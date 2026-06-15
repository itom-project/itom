/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    This file is a port and modified version of the
    CTK Common Toolkit (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QAbstractItemView>
#include <QApplication>
#include <QComboBox>
#include <QCompleter>
#include <QDebug>
#include <QFileSystemModel>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QLineEdit>
#include <qregularexpression.h>
#include <QSettings>
#include <QStyleOptionComboBox>
#include <QToolButton>

#include "pathLineEdit.h"
#include "utils.h"

//-----------------------------------------------------------------------------
class PathLineEditPrivate
{
  Q_DECLARE_PUBLIC(PathLineEdit);

protected:
  PathLineEdit* const q_ptr;

public:
  PathLineEditPrivate(PathLineEdit& object);
  void init();
  QSize recomputeSizeHint(QSize& sh)const;
  void updateFilter();

  void adjustPathLineEditSize();

  void _q_recomputeCompleterPopupSize();

  void createPathLineEditWidget(bool useComboBox);
  QString settingKey()const;

  QLineEdit*            LineEdit;
  QComboBox*            ComboBox;
  QToolButton*          BrowseButton;       //!< "..." button

  int                   MinimumContentsLength;
  PathLineEdit::SizeAdjustPolicy SizeAdjustPolicy;

  QString               Label;              //!< used in file dialogs
  QStringList           NameFilters;        //!< Regular expression (in wildcard mode) used to help the user to complete the line
  QDir::Filters         Filters;            //!< Type of path (file, dir...)
#ifdef USE_QFILEDIALOG_OPTIONS
  QFileDialog::Options DialogOptions;
#else
  PathLineEdit::Options DialogOptions;
#endif

  bool                  HasValidInput;      //!< boolean that stores the old state of valid input
  QString               SettingKey;

  static QString        sCurrentDirectory;   //!< Content the last value of the current directory
  static int            sMaxHistory;     //!< Size of the history, if the history is full and a new value is added, the oldest value is dropped

  mutable QSize SizeHint;
  mutable QSize MinimumSizeHint;
};

QString PathLineEditPrivate::sCurrentDirectory = "";
int PathLineEditPrivate::sMaxHistory = 5;

//-----------------------------------------------------------------------------
PathLineEditPrivate::PathLineEditPrivate(PathLineEdit& object)
  : q_ptr(&object)
  , LineEdit(0)
  , ComboBox(0)
  , BrowseButton(0)
  , MinimumContentsLength(0)
  , SizeAdjustPolicy(PathLineEdit::AdjustToContentsOnFirstShow)
  , Filters(QDir::AllEntries|QDir::NoDotAndDotDot|QDir::Readable)
  , HasValidInput(false)
{
}

//-----------------------------------------------------------------------------
void PathLineEditPrivate::init()
{
  Q_Q(PathLineEdit);

  QHBoxLayout* layout = new QHBoxLayout(q);
  layout->setContentsMargins(0,0,0,0);
  layout->setSpacing(0); // no space between the combobx and button

  this->createPathLineEditWidget(true);

  this->BrowseButton = new QToolButton(q);
  this->BrowseButton->setText("...");
  // Don't vertically stretch the path line edit unnecessary
  this->BrowseButton->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Ignored));
  this->BrowseButton->setToolTip(PathLineEdit::tr("Open a dialog"));

  QObject::connect(this->BrowseButton,SIGNAL(clicked()),
                   q, SLOT(browse()));

  layout->addWidget(this->BrowseButton);

  q->setSizePolicy(QSizePolicy(
                     QSizePolicy::Expanding, QSizePolicy::Fixed,
                     QSizePolicy::LineEdit));
}

//------------------------------------------------------------------------------
void PathLineEditPrivate::createPathLineEditWidget(bool useComboBox)
{
  Q_Q(PathLineEdit);

  QString path = q->currentPath();

  if (useComboBox)
    {
    this->ComboBox = new QComboBox(q);
    this->ComboBox->setEditable(true);
    this->ComboBox->setInsertPolicy(QComboBox::NoInsert);
    this->LineEdit = this->ComboBox->lineEdit();
    }
  else
    {
    this->ComboBox = 0;
    this->LineEdit = new QLineEdit(q);
    }

  if (q->layout() && q->layout()->itemAt(0))
    {
    delete q->layout()->itemAt(0)->widget();
    }
  qobject_cast<QHBoxLayout*>(q->layout())->insertWidget(
    0,
    this->ComboBox ? qobject_cast<QWidget*>(this->ComboBox) :
    qobject_cast<QWidget*>(this->LineEdit));

  this->updateFilter();
  q->retrieveHistory();
  q->setCurrentPath(path);

  QObject::connect(this->LineEdit, SIGNAL(textChanged(QString)),
                   q, SLOT(setCurrentDirectory(QString)));
  QObject::connect(this->LineEdit, SIGNAL(textChanged(QString)),
                   q, SLOT(updateHasValidInput()));
  q->updateGeometry();
}

//------------------------------------------------------------------------------
QSize PathLineEditPrivate::recomputeSizeHint(QSize& sh)const
{
  Q_Q(const PathLineEdit);
  if (!sh.isValid())
    {
    int frame = 0;
    if (this->ComboBox)
      {
      QStyleOptionComboBox option;
      int arrowWidth = this->ComboBox->style()->subControlRect(
            QStyle::CC_ComboBox, &option, QStyle::SC_ComboBoxArrow, this->ComboBox).width()
          + (this->ComboBox->hasFrame() ? 2 : 0);
      frame = 2 * (this->ComboBox->hasFrame() ? 3 : 0)
          + arrowWidth
          + 1; // for mac style, not sure why
      }
    else
      {
      QStyleOptionFrame option;
      int frameWidth = this->LineEdit->style()->pixelMetric(QStyle::PM_DefaultFrameWidth, &option, q);
      int horizontalMargin = 2; // QLineEditPrivate::horizontalMargin
      // See QLineEdit::sizeHint
      frame = 2 * frameWidth
          + this->LineEdit->textMargins().left()
          + this->LineEdit->textMargins().right()
          + this->LineEdit->contentsMargins().left()
          + this->LineEdit->contentsMargins().right()
          + 2 * horizontalMargin;
      }
    int browseWidth = 0;
    if (q->showBrowseButton())
      {
      browseWidth = this->BrowseButton->minimumSizeHint().width();
      }

    // text width
    int textWidth = 0;
    if (&sh == &this->SizeHint || this->MinimumContentsLength == 0)
      {
      switch (SizeAdjustPolicy)
        {
        case PathLineEdit::AdjustToContents:
        case PathLineEdit::AdjustToContentsOnFirstShow:
          if (this->LineEdit->text().isEmpty())
            {
#if (QT_VERSION >= QT_VERSION_CHECK(5,11,0))
            int character_pixel_width = this->LineEdit->fontMetrics().horizontalAdvance(QLatin1Char('x'));
#else
            int character_pixel_width = this->LineEdit->fontMetrics().width(QLatin1Char('x'));
#endif
            textWidth = 7 * character_pixel_width;
            }
          else
            {
            textWidth = this->LineEdit->fontMetrics().boundingRect(this->LineEdit->text()).width() + 8;
            }
          break;
        /*case QComboBox::AdjustToMinimumContentsLength:
        default:
          ;*/
        }
      }

    if (this->MinimumContentsLength > 0)
      {
#if (QT_VERSION >= QT_VERSION_CHECK(5,11,0))
        int character_pixel_width = this->LineEdit->fontMetrics().horizontalAdvance(QLatin1Char('X'));
#else
        int character_pixel_width = this->LineEdit->fontMetrics().width(QLatin1Char('X'));
#endif
        textWidth = qMax(textWidth, this->MinimumContentsLength * character_pixel_width);
      }

    int height = (this->ComboBox ? this->ComboBox->minimumSizeHint() :
                                   this->LineEdit->minimumSizeHint()).height();
    sh.rwidth() = frame + textWidth + browseWidth;
    sh.rheight() = height;
  }
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
  return sh;
#else
  return sh.expandedTo(QApplication::globalStrut());
#endif
}

//-----------------------------------------------------------------------------
void PathLineEditPrivate::updateFilter()
{
  Q_Q(PathLineEdit);
  // help completion for the QComboBox::QLineEdit

  QCompleter *newCompleter = new QCompleter(q);
  QFileSystemModel* fileSystemModel = new QFileSystemModel(newCompleter);
  fileSystemModel->setNameFilters(ctk::nameFiltersToExtensions(this->NameFilters));
  fileSystemModel->setFilter(this->Filters | QDir::NoDotAndDotDot | QDir::AllDirs);
  newCompleter->setModel(fileSystemModel);

  this->LineEdit->setCompleter(newCompleter);

  QObject::connect(this->LineEdit->completer()->completionModel(), SIGNAL(layoutChanged()),
                   q, SLOT(_q_recomputeCompleterPopupSize()));

  // don't accept invalid path
  QRegularExpressionValidator* validator = new QRegularExpressionValidator(
    ctk::nameFiltersToRegExp(this->NameFilters), q);
  this->LineEdit->setValidator(validator);
}

//-----------------------------------------------------------------------------
void PathLineEditPrivate::adjustPathLineEditSize()
{
  Q_Q(PathLineEdit);
  if (q->sizeAdjustPolicy() == PathLineEdit::AdjustToContents)
    {
    q->updateGeometry();
    q->adjustSize();
    q->update();
    }
}

//-----------------------------------------------------------------------------
void PathLineEditPrivate::_q_recomputeCompleterPopupSize()
{
  QSize lineEditSize = this->LineEdit->size();

  QAbstractItemView* view = this->LineEdit->completer()->popup();
  const QFontMetrics& fm = view->fontMetrics();

  int iconWidth = 0;
  int textWidth = 0;

  QStyleOptionFrame option;
  int frameWidth = view->style()->pixelMetric(QStyle::PM_DefaultFrameWidth, &option, view);
  int frame = 2 * frameWidth
      + view->contentsMargins().left()
      + view->contentsMargins().right();

  QAbstractItemModel* model = this->LineEdit->completer()->completionModel();
  for (int i = 0; i < model->rowCount(); ++i)
    {
    QVariant icon = model->data(model->index(i, 0), Qt::DecorationRole);
    if (icon.isValid() && icon.canConvert<QIcon>())
      {
      iconWidth = qMax(iconWidth, icon.value<QIcon>().availableSizes().front().width() + 4);
      }
    textWidth = qMax(textWidth, fm.boundingRect(model->data(model->index(i, 0)).toString()).width());
    }

  view->setMinimumWidth(qMax(frame + iconWidth + textWidth, lineEditSize.width()));
}

//-----------------------------------------------------------------------------
QString PathLineEditPrivate::settingKey()const
{
  Q_Q(const PathLineEdit);
  return QString("PathLineEdit/") +
    (this->SettingKey.isEmpty() ? q->objectName() : this->SettingKey);
}

//-----------------------------------------------------------------------------
PathLineEdit::PathLineEdit(QWidget *parentWidget)
  : QWidget(parentWidget)
  , d_ptr(new PathLineEditPrivate(*this))
{
  Q_D(PathLineEdit);
  d->init();

  this->setNameFilters(nameFilters());
  this->setFilters(filters());
}

//-----------------------------------------------------------------------------
PathLineEdit::PathLineEdit(const QString& label,
                                 const QStringList& nameFilters,
                                 Filters filters,
                                 QWidget *parentWidget)
  : QWidget(parentWidget)
  , d_ptr(new PathLineEditPrivate(*this))
{
  Q_D(PathLineEdit);
  d->init();

  this->setLabel(label);
  this->setNameFilters(nameFilters);
  this->setFilters(filters);
}

//-----------------------------------------------------------------------------
PathLineEdit::~PathLineEdit()
{
}

//-----------------------------------------------------------------------------
void PathLineEdit::setLabel(const QString &label)
{
  Q_D(PathLineEdit);
  d->Label = label;
}

//-----------------------------------------------------------------------------
const QString& PathLineEdit::label()const
{
  Q_D(const PathLineEdit);
  return d->Label;
}

//-----------------------------------------------------------------------------
void PathLineEdit::setNameFilters(const QStringList &nameFilters)
{
  Q_D(PathLineEdit);
  d->NameFilters = nameFilters;
  d->updateFilter();
}

//-----------------------------------------------------------------------------
const QStringList& PathLineEdit::nameFilters()const
{
  Q_D(const PathLineEdit);
  return d->NameFilters;
}

//-----------------------------------------------------------------------------
void PathLineEdit::setFilters(const Filters &filters)
{
  Q_D(PathLineEdit);
  d->Filters = QFlags<QDir::Filter>(static_cast<int>(filters));
  d->updateFilter();
}

//-----------------------------------------------------------------------------
PathLineEdit::Filters PathLineEdit::filters()const
{
  Q_D(const PathLineEdit);
  return QFlags<PathLineEdit::Filter>(static_cast<int>(d->Filters));
}

//-----------------------------------------------------------------------------
#ifdef USE_QFILEDIALOG_OPTIONS
void PathLineEdit::setOptions(const QFileDialog::Options& dialogOptions)
#else
void PathLineEdit::setOptions(const Options& dialogOptions)
#endif
{
  Q_D(PathLineEdit);
  d->DialogOptions = dialogOptions;
}

//-----------------------------------------------------------------------------
#ifdef USE_QFILEDIALOG_OPTIONS
const QFileDialog::Options& PathLineEdit::options()const
#else
const PathLineEdit::Options& PathLineEdit::options()const
#endif
{
  Q_D(const PathLineEdit);
  return d->DialogOptions;
}

//-----------------------------------------------------------------------------
void PathLineEdit::browse()
{
  Q_D(PathLineEdit);
  QString path = "";
  if ( d->Filters & QDir::Files ) //file
    {
    if ( d->Filters & QDir::Writable) // load or save
      {
      path = QFileDialog::getSaveFileName(
	this,
        tr("Select a file to save "),
        this->currentPath().isEmpty() ? PathLineEditPrivate::sCurrentDirectory :
	                                this->currentPath(),
	d->NameFilters.join(";;"),
	0,
#ifdef USE_QFILEDIALOG_OPTIONS
      d->DialogOptions);
#else
      QFlags<QFileDialog::Option>(int(d->DialogOptions)));
#endif
      }
    else
      {
      path = QFileDialog::getOpenFileName(
        this,
        QString(tr("Open a file")),
        this->currentPath().isEmpty()? PathLineEditPrivate::sCurrentDirectory :
	                               this->currentPath(),
        d->NameFilters.join(";;"),
	0,
#ifdef USE_QFILEDIALOG_OPTIONS
      d->DialogOptions);
#else
      QFlags<QFileDialog::Option>(int(d->DialogOptions)));
#endif
      }
    }
  else //directory
    {
    path = QFileDialog::getExistingDirectory(
      this,
      QString(tr("Select a directory...")),
      this->currentPath().isEmpty() ? PathLineEditPrivate::sCurrentDirectory :
                                      this->currentPath(),
#ifdef USE_QFILEDIALOG_OPTIONS
      d->DialogOptions);
#else
      QFlags<QFileDialog::Option>(int(d->DialogOptions)));
#endif
    }
  if (path.isEmpty())
    {
    return;
    }
  this->setCurrentPath(path);
}

//-----------------------------------------------------------------------------
void PathLineEdit::retrieveHistory()
{
  Q_D(PathLineEdit);
  if (d->ComboBox == 0)
    {
    return;
    }
  QString path = this->currentPath();
  bool wasBlocking = this->blockSignals(true);
  d->ComboBox->clear();
  // fill the combobox using the QSettings
  QSettings settings;
  QString key = d->settingKey();
  const QStringList history = settings.value(key).toStringList();
  foreach(const QString& path, history)
    {
    d->ComboBox->addItem(path);
    if (d->ComboBox->count() >= PathLineEditPrivate::sMaxHistory)
      {
      break;
      }
    }
  // Restore path or select the most recent file location if none set.
  if (path.isEmpty())
    {
    this->blockSignals(wasBlocking);
    d->ComboBox->setCurrentIndex(0);
    }
  else
    {
    this->setCurrentPath(path);
    this->blockSignals(wasBlocking);
    }
}

//-----------------------------------------------------------------------------
void PathLineEdit::addCurrentPathToHistory()
{
  Q_D(PathLineEdit);
  if (d->ComboBox == 0 ||
      this->currentPath().isEmpty())
    {
    return;
    }
  QSettings settings;
  //keep the same values, add the current value
  //if more than m_MaxHistory entrees, drop the oldest.
  QString key = d->settingKey();
  QStringList history = settings.value(key).toStringList();
  QString pathToAdd = this->currentPath();
  if (history.contains(pathToAdd))
    {
    history.removeAll(pathToAdd);
    }
  history.push_front(pathToAdd);
  settings.setValue(key, history);
  // don't fire intermediate events.
  bool wasBlocking = d->ComboBox->blockSignals(false);
  int index = d->ComboBox->findText(this->currentPath());
  if (index >= 0)
    {
    d->ComboBox->removeItem(index);
    }
  while (d->ComboBox->count() >= PathLineEditPrivate::sMaxHistory)
    {
    d->ComboBox->removeItem(d->ComboBox->count() - 1);
    }
  d->ComboBox->insertItem(0, pathToAdd);
  d->ComboBox->setCurrentIndex(0);
  d->ComboBox->blockSignals(wasBlocking);
}

//------------------------------------------------------------------------------
void PathLineEdit::setCurrentFileExtension(const QString& extension)
{
  QString filename = this->currentPath();
  QFileInfo fileInfo(filename);

  if (!fileInfo.suffix().isEmpty())
    {
    filename.replace(fileInfo.suffix(), extension);
    }
  else
    {
    filename.append(QString(".") + extension);
    }
  this->setCurrentPath(filename);
}

//------------------------------------------------------------------------------
QComboBox* PathLineEdit::comboBox() const
{
  Q_D(const PathLineEdit);
  return d->ComboBox;
}

//------------------------------------------------------------------------------
QString PathLineEdit::currentPath()const
{
  Q_D(const PathLineEdit);
  return d->LineEdit ? d->LineEdit->text() : QString();
}

//------------------------------------------------------------------------------
void PathLineEdit::setCurrentPath(const QString& path)
{
  Q_D(PathLineEdit);
  d->LineEdit->setText(path);
}

//------------------------------------------------------------------------------
void PathLineEdit::setCurrentDirectory(const QString& directory)
{
  PathLineEditPrivate::sCurrentDirectory = directory;
}

//------------------------------------------------------------------------------
void PathLineEdit::updateHasValidInput()
{
  Q_D(PathLineEdit);

  bool oldHasValidInput = d->HasValidInput;
  d->HasValidInput = d->LineEdit->hasAcceptableInput();
  if (d->HasValidInput)
    {
    QFileInfo fileInfo(this->currentPath());
    PathLineEditPrivate::sCurrentDirectory =
      fileInfo.isFile() ? fileInfo.absolutePath() : fileInfo.absoluteFilePath();
    emit currentPathChanged(this->currentPath());
    }
  if (d->HasValidInput != oldHasValidInput)
    {
    emit validInputChanged(d->HasValidInput);
    }

  if (d->SizeAdjustPolicy == AdjustToContents)
    {
    d->SizeHint = QSize();
    d->adjustPathLineEditSize();
    this->updateGeometry();
    }
}

//------------------------------------------------------------------------------
QString PathLineEdit::settingKey()const
{
  Q_D(const PathLineEdit);
  return d->SettingKey;
}

//------------------------------------------------------------------------------
void PathLineEdit::setSettingKey(const QString& key)
{
  Q_D(PathLineEdit);
  d->SettingKey = key;
  this->retrieveHistory();
}

//------------------------------------------------------------------------------
bool PathLineEdit::showBrowseButton()const
{
  Q_D(const PathLineEdit);
  return d->BrowseButton->isVisibleTo(const_cast<PathLineEdit*>(this));
}

//------------------------------------------------------------------------------
void PathLineEdit::setShowBrowseButton(bool visible)
{
  Q_D(PathLineEdit);
  d->BrowseButton->setVisible(visible);
}

//------------------------------------------------------------------------------
bool PathLineEdit::showHistoryButton()const
{
  Q_D(const PathLineEdit);
  return d->ComboBox ? true: false;
}

//------------------------------------------------------------------------------
void PathLineEdit::setShowHistoryButton(bool visible)
{
  Q_D(PathLineEdit);
  d->createPathLineEditWidget(visible);
}

//------------------------------------------------------------------------------
PathLineEdit::SizeAdjustPolicy PathLineEdit::sizeAdjustPolicy() const
{
  Q_D(const PathLineEdit);
  return d->SizeAdjustPolicy;
}

//------------------------------------------------------------------------------
void PathLineEdit::setSizeAdjustPolicy(PathLineEdit::SizeAdjustPolicy policy)
{
  Q_D(PathLineEdit);
  if (policy == d->SizeAdjustPolicy)
    return;

  d->SizeAdjustPolicy = policy;
  d->SizeHint = QSize();
  d->adjustPathLineEditSize();
  this->updateGeometry();
}

//------------------------------------------------------------------------------
int PathLineEdit::minimumContentsLength()const
{
  Q_D(const PathLineEdit);
  return d->MinimumContentsLength;
}

//------------------------------------------------------------------------------
void PathLineEdit::setMinimumContentsLength(int length)
{
  Q_D(PathLineEdit);
  if (d->MinimumContentsLength == length || length < 0) return;

  d->MinimumContentsLength = length;

  if (d->SizeAdjustPolicy == AdjustToContents ||
      d->SizeAdjustPolicy == AdjustToMinimumContentsLength)
    {
    d->SizeHint = QSize();
    d->adjustPathLineEditSize();
    this->updateGeometry();
    }
}

//------------------------------------------------------------------------------
QSize PathLineEdit::minimumSizeHint()const
{
  Q_D(const PathLineEdit);
  return d->recomputeSizeHint(d->MinimumSizeHint);
}

//------------------------------------------------------------------------------
QSize PathLineEdit::sizeHint()const
{
  Q_D(const PathLineEdit);
  return d->recomputeSizeHint(d->SizeHint);
}

#include "moc_pathLineEdit.h"
