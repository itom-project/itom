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
    Common framework (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QApplication>
#include <QAbstractItemView>
#include <QDebug>
#if QT_VERSION <= QT_VERSION_CHECK(6, 0, 0)
#include <QDesktopWidget>
#endif
#include <QItemDelegate>
#include <QLayout>
#include <QMouseEvent>
#include <QMenu>
#include <QPainter>
#include <QPointer>
#include <QPushButton>
#include <QStandardItemModel>
#include <QStyle>
#include <QStyleOptionButton>
#include <QStylePainter>
#include <QToolBar>
#include <QAbstractItemModel>

// CTK includes
#include "checkableComboBox.h"
#include "helper/checkableModelHelper.h"

// Similar to QComboBoxDelegate
class ComboBoxDelegate : public QItemDelegate
{
public:
    ComboBoxDelegate(QObject *parent, QComboBox *cmb)
      : QItemDelegate(parent), ComboBox(cmb)
    {}

    static bool isSeparator(const QModelIndex &index)
    {
      return index.data(Qt::AccessibleDescriptionRole).toString() == QLatin1String("separator");
    }
    static void setSeparator(QAbstractItemModel *model, const QModelIndex &index)
    {
      model->setData(index, QString::fromLatin1("separator"), Qt::AccessibleDescriptionRole);
      if (QStandardItemModel *m = qobject_cast<QStandardItemModel*>(model))
        {
        if (QStandardItem *item = m->itemFromIndex(index))
          {
          item->setFlags(item->flags() & ~(Qt::ItemIsSelectable|Qt::ItemIsEnabled));
          }
        }
    }

protected:
    void paint(QPainter *painter,
               const QStyleOptionViewItem &option,
               const QModelIndex &index) const
    {
      if (isSeparator(index))
        {
            QRect rect = option.rect;

            if (const QAbstractItemView *view = qobject_cast<const QAbstractItemView*>(option.widget))
            {
            rect.setWidth(view->viewport()->width());
            }

            QStyleOption opt;
            opt.rect = rect;
            this->ComboBox->style()->drawPrimitive(QStyle::PE_IndicatorToolBarSeparator, &opt, painter, this->ComboBox);
        }
      else
        {
        QItemDelegate::paint(painter, option, index);
        }
    }

    QSize sizeHint(const QStyleOptionViewItem &option,
                   const QModelIndex &index) const
    {
      if (isSeparator(index))
        {
        int pm = this->ComboBox->style()->pixelMetric(QStyle::PM_DefaultFrameWidth, 0, this->ComboBox);
        return QSize(pm, pm);
        }
      return this->QItemDelegate::sizeHint(option, index);
    }
private:
    QComboBox* ComboBox;
};

//-----------------------------------------------------------------------------
class CheckableComboBoxPrivate
{
  Q_DECLARE_PUBLIC(CheckableComboBox);
protected:
  CheckableComboBox* const q_ptr;
  QModelIndexList checkedIndexes()const;
  QModelIndexList uncheckedIndexes()const;

public:
  CheckableComboBoxPrivate(CheckableComboBox& object);
  void init();

  QModelIndexList cachedCheckedIndexes()const;
  void updateCheckedList();

  CheckableModelHelper* CheckableModelHelper_;
  bool MouseButtonPressed;

private:
  QModelIndexList persistentIndexesToModelIndexes(
    const QList<QPersistentModelIndex>& persistentModels)const;
  QList<QPersistentModelIndex> modelIndexesToPersistentIndexes(
    const QModelIndexList& modelIndexes)const;

  mutable QList<QPersistentModelIndex> CheckedList;
};

//-----------------------------------------------------------------------------
CheckableComboBoxPrivate::CheckableComboBoxPrivate(CheckableComboBox& object)
  : q_ptr(&object)
{
  this->CheckableModelHelper_ = 0;
  this->MouseButtonPressed = false;
}

//-----------------------------------------------------------------------------
void CheckableComboBoxPrivate::init()
{
  Q_Q(CheckableComboBox);
  this->CheckableModelHelper_ = new CheckableModelHelper(Qt::Horizontal, q);
  this->CheckableModelHelper_->setForceCheckability(true);

  q->setCheckableModel(q->model());
  q->view()->installEventFilter(q);
  q->view()->viewport()->installEventFilter(q);
  // QCleanLooksStyle uses a delegate that doesn't show the checkboxes in the
  // popup list.
  q->setItemDelegate(new ComboBoxDelegate(q->view(), q));
}

//-----------------------------------------------------------------------------
void CheckableComboBoxPrivate::updateCheckedList()
{
  Q_Q(CheckableComboBox);
  QList<QPersistentModelIndex> newCheckedPersistentList =
    this->modelIndexesToPersistentIndexes(this->checkedIndexes());
  if (newCheckedPersistentList == this->CheckedList)
    {
    return;
    }
  this->CheckedList = newCheckedPersistentList;
  emit q->checkedIndexesChanged();
}

//-----------------------------------------------------------------------------
QList<QPersistentModelIndex> CheckableComboBoxPrivate
::modelIndexesToPersistentIndexes(const QModelIndexList& indexes)const
{
  QList<QPersistentModelIndex> res;
  foreach(const QModelIndex& index, indexes)
    {
    QPersistentModelIndex persistent(index);
    if (persistent.isValid())
      {
      res << persistent;
      }
    }
  return res;
}

//-----------------------------------------------------------------------------
QModelIndexList CheckableComboBoxPrivate
::persistentIndexesToModelIndexes(
  const QList<QPersistentModelIndex>& indexes)const
{
  QModelIndexList res;
  foreach(const QPersistentModelIndex& index, indexes)
    {
    if (index.isValid())
      {
      res << index;
      }
    }
  return res;
}

//-----------------------------------------------------------------------------
QModelIndexList CheckableComboBoxPrivate::cachedCheckedIndexes()const
{
  return this->persistentIndexesToModelIndexes(this->CheckedList);
}

//-----------------------------------------------------------------------------
QModelIndexList CheckableComboBoxPrivate::checkedIndexes()const
{
  Q_Q(const CheckableComboBox);
  QModelIndex startIndex = q->model()->index(0,0, q->rootModelIndex());
  return q->model()->match(
    startIndex, Qt::CheckStateRole,
    static_cast<int>(Qt::Checked), -1, Qt::MatchRecursive);
}

//-----------------------------------------------------------------------------
QModelIndexList CheckableComboBoxPrivate::uncheckedIndexes()const
{
  Q_Q(const CheckableComboBox);
  QModelIndex startIndex = q->model()->index(0,0, q->rootModelIndex());
  return q->model()->match(
    startIndex, Qt::CheckStateRole,
    static_cast<int>(Qt::Unchecked), -1, Qt::MatchRecursive);
}

//-----------------------------------------------------------------------------
CheckableComboBox::CheckableComboBox(QWidget* parentWidget)
  : QComboBox(parentWidget)
  , d_ptr(new CheckableComboBoxPrivate(*this))
{
  Q_D(CheckableComboBox);
  d->init();
}

//-----------------------------------------------------------------------------
CheckableComboBox::~CheckableComboBox()
{
}

//-----------------------------------------------------------------------------
bool CheckableComboBox::eventFilter(QObject *o, QEvent *e)
{
  Q_D(CheckableComboBox);
  switch (e->type())
    {
    case QEvent::MouseButtonPress:
      {
      if (this->view()->isVisible())
        {
        d->MouseButtonPressed = true;
        }
      break;
      }
    case QEvent::MouseButtonRelease:
      {
      QMouseEvent *m = static_cast<QMouseEvent *>(e);
      if (this->view()->isVisible() &&
          this->view()->rect().contains(m->pos()) &&
          this->view()->currentIndex().isValid()
          //&& !blockMouseReleaseTimer.isActive()
          && (this->view()->currentIndex().flags() & Qt::ItemIsEnabled)
          && (this->view()->currentIndex().flags() & Qt::ItemIsSelectable))
        {
        // The signal to open the menu is fired when the mouse button is
        // pressed, we don't want to toggle the item under the mouse cursor
        // when the button used to open the popup is released.
        if (d->MouseButtonPressed)
          {
          // make the item current, it will then call QComboBox::update (and
          // repaint) when the current index data is changed (checkstate
          // toggled fires dataChanged signal which is observed).
          this->setCurrentIndex(this->view()->currentIndex().row());
          d->CheckableModelHelper_->toggleCheckState(this->view()->currentIndex());
          }
        d->MouseButtonPressed = false;
        return true;
        }
      d->MouseButtonPressed = false;
      break;
      }
    default:
        break;
    }
  return this->QComboBox::eventFilter(o, e);
}

//-----------------------------------------------------------------------------
void CheckableComboBox::setCheckableModel(QAbstractItemModel* newModel)
{
  Q_D(CheckableComboBox);
  this->disconnect(this->model(), SIGNAL(dataChanged(QModelIndex,QModelIndex)),
                   this, SLOT(onDataChanged(QModelIndex,QModelIndex)));
  if (newModel != this->model())
    {
    this->setModel(newModel);
    }
  this->connect(this->model(), SIGNAL(dataChanged(QModelIndex,QModelIndex)),
                this, SLOT(onDataChanged(QModelIndex,QModelIndex)));
  d->CheckableModelHelper_->setModel(newModel);
  d->updateCheckedList();
}

//-----------------------------------------------------------------------------
QAbstractItemModel* CheckableComboBox::checkableModel()const
{
  return this->model();
}

//-----------------------------------------------------------------------------
QModelIndexList CheckableComboBox::checkedIndexes()const
{
  Q_D(const CheckableComboBox);
  return d->cachedCheckedIndexes();
}

//-----------------------------------------------------------------------------
bool CheckableComboBox::allChecked()const
{
  Q_D(const CheckableComboBox);
  return d->uncheckedIndexes().count() == 0;
}

//-----------------------------------------------------------------------------
bool CheckableComboBox::noneChecked()const
{
  Q_D(const CheckableComboBox);
  return d->cachedCheckedIndexes().count() == 0;
}

//-----------------------------------------------------------------------------
void CheckableComboBox::setCheckState(const QModelIndex& index, Qt::CheckState check)
{
  Q_D(CheckableComboBox);
  return d->CheckableModelHelper_->setCheckState(index, check);
}

//-----------------------------------------------------------------------------
Qt::CheckState CheckableComboBox::checkState(const QModelIndex& index)const
{
  Q_D(const CheckableComboBox);
  return d->CheckableModelHelper_->checkState(index);
}

//-----------------------------------------------------------------------------
CheckableModelHelper* CheckableComboBox::checkableModelHelper()const
{
    Q_D(const CheckableComboBox);
    return d->CheckableModelHelper_;
}

//-----------------------------------------------------------------------------
void CheckableComboBox::onDataChanged(const QModelIndex& start, const QModelIndex& end)
{
  Q_D(CheckableComboBox);
  Q_UNUSED(start);
  Q_UNUSED(end);
  d->updateCheckedList();
}

//-----------------------------------------------------------------------------
void CheckableComboBox::paintEvent(QPaintEvent *)
{
  Q_D(CheckableComboBox);

  QStylePainter painter(this);
  painter.setPen(palette().color(QPalette::Text));

  // draw the combobox frame, focusrect and selected etc.
  QStyleOptionComboBox opt;
  this->initStyleOption(&opt);

  if (this->allChecked())
    {
    opt.currentText = "All";
    opt.currentIcon = QIcon();
    }
  else if (this->noneChecked())
    {
    opt.currentText = "None";
    opt.currentIcon = QIcon();
    }
  else
    {
    //search the checked items
    QModelIndexList indexes = d->cachedCheckedIndexes();
    if (indexes.count() == 1)
      {
      opt.currentText = this->model()->data(indexes[0], Qt::DisplayRole).toString();
      opt.currentIcon = qvariant_cast<QIcon>(this->model()->data(indexes[0], Qt::DecorationRole));
      }
    else
      {
      QStringList indexesText;
      foreach(QModelIndex checkedIndex, indexes)
        {
        indexesText << this->model()->data(checkedIndex, Qt::DisplayRole).toString();
        }
      opt.currentText = indexesText.join(", ");
      opt.currentIcon = QIcon();
      }
    }
  painter.drawComplexControl(QStyle::CC_ComboBox, opt);

  // draw the icon and text
  painter.drawControl(QStyle::CE_ComboBoxLabel, opt);
}

//-----------------------------------------------------------------------------
void CheckableComboBox::setCheckedIndices(const QVector<int> &indices)
{
    Q_D(CheckableComboBox);
    QAbstractItemModel *m = model();

    for (int i = 0; i < count(); ++i)
    {
        if (indices.contains(i))
        {
            d->CheckableModelHelper_->setCheckState(m->index(i, 0, rootModelIndex()), Qt::Checked);
        }
        else
        {
            d->CheckableModelHelper_->setCheckState(m->index(i, 0, rootModelIndex()), Qt::Unchecked);
        }
    }

    ;
}

//-----------------------------------------------------------------------------
QVector<int> CheckableComboBox::getCheckedIndices() const
{
    QModelIndexList indices = checkedIndexes();
    QVector<int> i;
    foreach(const QModelIndex &mi, indices)
    {
        i << mi.row();
    }
    return i;
}

//-----------------------------------------------------------------------------
void CheckableComboBox::setIndexState(int index, bool state)
{
	Q_D(CheckableComboBox);
	QAbstractItemModel *m = model();

	if (state)
		d->CheckableModelHelper_->setCheckState(m->index(index, 0, rootModelIndex()), Qt::Checked);
	else
		d->CheckableModelHelper_->setCheckState(m->index(index, 0, rootModelIndex()), Qt::Unchecked);

}
