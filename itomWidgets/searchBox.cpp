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
#include <QDebug>
#include <QIcon>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>
#include <QStyleOption>

// CTK includes
#include "searchBox.h"

// --------------------------------------------------
class SearchBoxPrivate
{
  Q_DECLARE_PUBLIC(SearchBox);
protected:
  SearchBox* const q_ptr;
public:
  SearchBoxPrivate(SearchBox& object);
  void init();

  /// Position and size for the clear icon in the QLineEdit
  QRect clearRect()const;
  /// Position and size for the search icon in the QLineEdit
  QRect searchRect()const;

  QIcon clearIcon;
  QIcon searchIcon;
  bool showSearchIcon;
  bool alwaysShowClearIcon;
  bool hideClearIcon;

#if QT_VERSION < 0x040700
  QString placeholderText;
#endif
};

// --------------------------------------------------
SearchBoxPrivate::SearchBoxPrivate(SearchBox &object)
  : q_ptr(&object)
{
  this->clearIcon = QIcon(":icons/clear.svg");
  this->searchIcon = QIcon(":icons/search.svg");
  this->showSearchIcon = false;
  this->alwaysShowClearIcon = false;
  this->hideClearIcon = true;
}

// --------------------------------------------------
void SearchBoxPrivate::init()
{
  Q_Q(SearchBox);

  // Set a text by default on the QLineEdit
  q->setPlaceholderText(QObject::tr("Search..."));

  QObject::connect(q, SIGNAL(textChanged(QString)),
                   q, SLOT(updateClearButtonState()));
}

// --------------------------------------------------
QRect SearchBoxPrivate::clearRect()const
{
  Q_Q(const SearchBox);
  QRect cRect = this->searchRect();
  cRect.moveLeft(q->width() - cRect.width() - cRect.left());
  return cRect;
}

// --------------------------------------------------
QRect SearchBoxPrivate::searchRect()const
{
  Q_Q(const SearchBox);
  QRect sRect = q->contentsRect();
  // If the QLineEdit has a frame, the icon must be shifted from
  // the frame line width
  if (q->hasFrame())
    {
    QStyleOptionFrame opt;
    q->initStyleOption(&opt);
    sRect.adjust(opt.lineWidth, opt.lineWidth, -opt.lineWidth, -opt.lineWidth);
    }
  // Hardcoded: shrink by 1 pixel because some styles have a focus frame inside
  // the line edit frame.
  sRect.adjust(1, 1, -1, -1);
  // Square size
  sRect.setWidth(sRect.height());
  return sRect;
}

// --------------------------------------------------
SearchBox::SearchBox(QWidget* _parent)
  : QLineEdit(_parent)
  , d_ptr(new SearchBoxPrivate(*this))
{
  Q_D(SearchBox);
  d->init();
}

// --------------------------------------------------
SearchBox::~SearchBox()
{
}

#if QT_VERSION < 0x040700
// --------------------------------------------------
QString SearchBox::placeholderText()const
{
  Q_D(const SearchBox);
  return d->placeholderText;
}

// --------------------------------------------------
void SearchBox::setPlaceholderText(const QString &defaultText)
{
  Q_D(SearchBox);
  d->placeholderText = defaultText;
  if (!this->hasFocus())
    {
    this->update();
    }
}
#endif

// --------------------------------------------------
void SearchBox::setShowSearchIcon(bool show)
{
  Q_D(SearchBox);
  d->showSearchIcon = show;
  this->update();
}

// --------------------------------------------------
bool SearchBox::showSearchIcon()const
{
  Q_D(const SearchBox);
  return d->showSearchIcon;
}

// --------------------------------------------------
void SearchBox::setAlwaysShowClearIcon(bool show)
{
  Q_D(SearchBox);
  d->alwaysShowClearIcon = show;
  if (show == true)
    {
    d->hideClearIcon = false;
    }
  this->update();
}

// --------------------------------------------------
bool SearchBox::alwaysShowClearIcon()const
{
  Q_D(const SearchBox);
  return d->alwaysShowClearIcon;
}

// --------------------------------------------------
void SearchBox::setSearchIcon(const QIcon& icon)
{
  Q_D(SearchBox);
  d->searchIcon = icon;
  this->update();
}

// --------------------------------------------------
QIcon SearchBox::searchIcon()const
{
  Q_D(const SearchBox);
  return d->searchIcon;
}

// --------------------------------------------------
void SearchBox::setClearIcon(const QIcon& icon)
{
  Q_D(SearchBox);
  d->clearIcon = icon;
  this->update();
}

// --------------------------------------------------
QIcon SearchBox::clearIcon()const
{
  Q_D(const SearchBox);
  return d->clearIcon;
}

// --------------------------------------------------
void SearchBox::paintEvent(QPaintEvent * event)
{
    Q_D(SearchBox);

    // Draw the line edit with text.
    // Text has already been shifted to the right (in resizeEvent()) to leave
    // space for the search icon.
    this->Superclass::paintEvent(event);

    QPainter p(this);

    QRect cRect = d->clearRect();
    QRect sRect = d->showSearchIcon ? d->searchRect() : QRect();

    // Draw clearIcon
    if (!d->hideClearIcon)
    {
        QPixmap closePixmap = d->clearIcon.pixmap(cRect.size(), this->isEnabled() ? QIcon::Normal : QIcon::Disabled);
        this->style()->drawItemPixmap(&p, cRect, Qt::AlignCenter, closePixmap);
    }

    // Draw searchIcon
    if (d->showSearchIcon)
    {
        QPixmap searchPixmap = d->searchIcon.pixmap(sRect.size(), this->isEnabled() ? QIcon::Normal : QIcon::Disabled);
        this->style()->drawItemPixmap(&p, sRect, Qt::AlignCenter, searchPixmap);
    }
}

// --------------------------------------------------
void SearchBox::mousePressEvent(QMouseEvent *e)
{
  Q_D(SearchBox);

  if(d->clearRect().contains(e->pos()))
    {
    this->clear();
    emit this->textEdited(this->text());
    return;
    }

  if(d->showSearchIcon && d->searchRect().contains(e->pos()))
    {
    this->selectAll();
    return;
    }

  this->Superclass::mousePressEvent(e);
}

// --------------------------------------------------
void SearchBox::mouseMoveEvent(QMouseEvent *e)
{
  Q_D(SearchBox);

  if(d->clearRect().contains(e->pos()) ||
     (d->showSearchIcon && d->searchRect().contains(e->pos())))
    {
    this->setCursor(Qt::PointingHandCursor);
    }
  else
    {
    this->setCursor(this->isReadOnly() ? Qt::ArrowCursor : Qt::IBeamCursor);
    }
  this->Superclass::mouseMoveEvent(e);
}

// --------------------------------------------------
void SearchBox::resizeEvent(QResizeEvent * event)
{
  Q_D(SearchBox);
  static int iconSpacing = 0; // hardcoded,
  QRect cRect = d->clearRect();
  QRect sRect = d->showSearchIcon ? d->searchRect() : QRect();
  // Set 2 margins each sides of the QLineEdit, according to the icons
  this->setTextMargins( sRect.right() + iconSpacing, 0,
                        event->size().width() - cRect.left() - iconSpacing,0);
}

// --------------------------------------------------
void SearchBox::updateClearButtonState()
{
  Q_D(SearchBox);
  if (!d->alwaysShowClearIcon)
    {
    d->hideClearIcon = this->text().isEmpty() ? true : false;
    }
}
