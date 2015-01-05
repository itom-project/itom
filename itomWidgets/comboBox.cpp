/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut f�r Technische
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
#include <QAbstractScrollArea>
#include <QApplication>
#include <QDebug>
#include <QScrollBar>
#include <QStylePainter>
#include <QWheelEvent>

// CTK includes
#include "comboBox.h"

// -------------------------------------------------------------------------
class ComboBoxPrivate
{
  Q_DECLARE_PUBLIC(ComboBox);
protected:
  ComboBox* const q_ptr;
public:
  ComboBoxPrivate(ComboBox& object);
  void initStyleOption(QStyleOptionComboBox* opt)const;
  QSize recomputeSizeHint(QSize &sh) const;
  QString DefaultText;
  QIcon   DefaultIcon;
  bool    ForceDefault;
  Qt::TextElideMode ElideMode;
  ComboBox::ScrollEffect ScrollWheelEffect;

  mutable QSize MinimumSizeHint;
  mutable QSize SizeHint;
};

// -------------------------------------------------------------------------
ComboBoxPrivate::ComboBoxPrivate(ComboBox& object)
  :q_ptr(&object)
{
  this->DefaultText = "";
  this->ForceDefault = false;
  this->ElideMode = Qt::ElideNone;
  this->ScrollWheelEffect = ComboBox::AlwaysScroll;
}

// -------------------------------------------------------------------------
QSize ComboBoxPrivate::recomputeSizeHint(QSize &sh) const
{
  Q_Q(const ComboBox);
  if (sh.isValid())
    {
    return sh.expandedTo(QApplication::globalStrut());
    }

  bool hasIcon = false;
  int count = q->count();
  QSize iconSize = q->iconSize();
  const QFontMetrics &fm = q->fontMetrics();

  // text width
  if (&sh == &this->SizeHint || q->minimumContentsLength() == 0)
    {
    switch (q->sizeAdjustPolicy())
      {
      case QComboBox::AdjustToContents:
      case QComboBox::AdjustToContentsOnFirstShow:
        if (count == 0 || this->ForceDefault)
          {
          sh.rwidth() = this->DefaultText.isEmpty() ?
            7 * fm.width(QLatin1Char('x')) :
            fm.boundingRect(this->DefaultText).width();
          if (!this->DefaultIcon.isNull())
            {
            hasIcon = true;
            sh.rwidth() += iconSize.width() + 4;
            }
          }
        for (int i = 0; i < count; ++i)
          {
          if (!q->itemIcon(i).isNull())
            {
            hasIcon = true;
            sh.setWidth(qMax(sh.width(), fm.boundingRect(q->itemText(i)).width() + iconSize.width() + 4));
            }
          else
            {
            sh.setWidth(qMax(sh.width(), fm.boundingRect(q->itemText(i)).width()));
            }
          }
        break;
      case QComboBox::AdjustToMinimumContentsLength:
        if ((count == 0 || this->ForceDefault) && !this->DefaultIcon.isNull())
          {
          hasIcon = true;
          }
        for (int i = 0; i < count && !hasIcon; ++i)
          {
          hasIcon = !q->itemIcon(i).isNull();
          }
        break;
      case QComboBox::AdjustToMinimumContentsLengthWithIcon:
        hasIcon = true;
        break;
      default:
        break;
      }
    }
  else // minimumsizehint is computing and minimumcontentslenght is > 0
    {
    if ((count == 0 || this->ForceDefault) && !this->DefaultIcon.isNull())
      {
      hasIcon = true;
      }

    for (int i = 0; i < count && !hasIcon; ++i)
      {
      hasIcon = !q->itemIcon(i).isNull();
      }
    }
  if (q->minimumContentsLength() > 0)
    {
    sh.setWidth(qMax(sh.width(),
                     q->minimumContentsLength() * fm.width(QLatin1Char('X'))
                     + (hasIcon ? iconSize.width() + 4 : 0)));
    }

  // height
  sh.setHeight(qMax(fm.height(), 14) + 2);
  if (hasIcon)
    {
    sh.setHeight(qMax(sh.height(), iconSize.height() + 2));
    }

  // add style and strut values
  QStyleOptionComboBox opt;
  this->initStyleOption(&opt);
  sh = q->style()->sizeFromContents(QStyle::CT_ComboBox, &opt, sh, q);
  return sh.expandedTo(QApplication::globalStrut());
}

// -------------------------------------------------------------------------
void ComboBoxPrivate::initStyleOption(QStyleOptionComboBox* opt)const
{
  Q_Q(const ComboBox);
  q->initStyleOption(opt);
  if (q->currentIndex() == -1 ||
      this->ForceDefault)
    {
    opt->currentText = this->DefaultText;
    opt->currentIcon = this->DefaultIcon;
    }
  QRect textRect = q->style()->subControlRect(
    QStyle::CC_ComboBox, opt, QStyle::SC_ComboBoxEditField, q);
  // TODO substract icon size
  opt->currentText = opt->fontMetrics.elidedText(opt->currentText,
                                                 this->ElideMode,
                                                 textRect.width());
}


// -------------------------------------------------------------------------
ComboBox::ComboBox(QWidget* _parent)
  : QComboBox(_parent)
  , d_ptr(new ComboBoxPrivate(*this))
{
}

// -------------------------------------------------------------------------
ComboBox::~ComboBox()
{
}

// -------------------------------------------------------------------------
void ComboBox::setDefaultText(const QString& newDefaultText)
{
  Q_D(ComboBox);
  d->DefaultText = newDefaultText;
  d->SizeHint = QSize();
  this->update();
}

// -------------------------------------------------------------------------
QString ComboBox::defaultText()const
{
  Q_D(const ComboBox);
  return d->DefaultText;
}

// -------------------------------------------------------------------------
void ComboBox::setDefaultIcon(const QIcon& newIcon)
{
  Q_D(ComboBox);
  d->DefaultIcon = newIcon;
  d->SizeHint = QSize();
  this->update();
}

// -------------------------------------------------------------------------
QIcon ComboBox::defaultIcon()const
{
  Q_D(const ComboBox);
  return d->DefaultIcon;
}

// -------------------------------------------------------------------------
void ComboBox::forceDefault(bool newForceDefault)
{
  Q_D(ComboBox);
  if (newForceDefault == d->ForceDefault)
    {
    return;
    }
  d->ForceDefault = newForceDefault;
  d->SizeHint = QSize();
  this->updateGeometry();
}

// -------------------------------------------------------------------------
void ComboBox::setElideMode(const Qt::TextElideMode& newMode)
{
  Q_D(ComboBox);
  d->ElideMode = newMode;
  this->update();
}
// -------------------------------------------------------------------------
Qt::TextElideMode ComboBox::elideMode()const
{
  Q_D(const ComboBox);
  return d->ElideMode;
}

// -------------------------------------------------------------------------
bool ComboBox::isDefaultForced()const
{
  Q_D(const ComboBox);
  return d->ForceDefault;
}

// -------------------------------------------------------------------------
ComboBox::ScrollEffect ComboBox::scrollWheelEffect()const
{
  Q_D(const ComboBox);
  return d->ScrollWheelEffect;
}

// -------------------------------------------------------------------------
void ComboBox::setScrollWheelEffect(ComboBox::ScrollEffect scroll)
{
  Q_D(ComboBox);
  d->ScrollWheelEffect = scroll;
  this->setFocusPolicy( d->ScrollWheelEffect == ComboBox::ScrollWithFocus ?
                        Qt::StrongFocus : Qt::WheelFocus );
}

// -------------------------------------------------------------------------
void ComboBox::paintEvent(QPaintEvent*)
{
  Q_D(ComboBox);
  QStylePainter painter(this);
  painter.setPen(palette().color(QPalette::Text));

  QStyleOptionComboBox opt;
  d->initStyleOption(&opt);

  // draw the combobox frame, focusrect and selected etc.
  painter.drawComplexControl(QStyle::CC_ComboBox, opt);
  // draw the icon and text
  painter.drawControl(QStyle::CE_ComboBoxLabel, opt);
}

// -------------------------------------------------------------------------
QSize ComboBox::minimumSizeHint() const
{
  Q_D(const ComboBox);
  return d->recomputeSizeHint(d->MinimumSizeHint);
}

// -------------------------------------------------------------------------
/*!
    \reimp

    This implementation caches the size hint to avoid resizing when
    the contents change dynamically. To invalidate the cached value
    change the \l sizeAdjustPolicy.
*/
QSize ComboBox::sizeHint() const
{
  Q_D(const ComboBox);
  return d->recomputeSizeHint(d->SizeHint);
}

// -------------------------------------------------------------------------
void ComboBox::changeEvent(QEvent *e)
{
  Q_D(const ComboBox);
  switch (e->type())
    {
    case QEvent::StyleChange:
    case QEvent::MacSizeChange:
    case QEvent::FontChange:
      d->SizeHint = QSize();
      d->MinimumSizeHint = QSize();
      break;
    default:
      break;
    }

  this->QComboBox::changeEvent(e);
}

// -------------------------------------------------------------------------
void ComboBox::wheelEvent(QWheelEvent* event)
{
  Q_D(ComboBox);
  bool scroll = false;
  switch (d->ScrollWheelEffect)
    {
    case AlwaysScroll:
      scroll = true;
      break;
    case ScrollWithFocus:
      scroll = this->hasFocus();
      break;
    case ScrollWithNoVScrollBar:
      scroll = true;
      for (QWidget* ancestor = this->parentWidget();
           ancestor; ancestor = ancestor->parentWidget())
        {
        if (QAbstractScrollArea* scrollArea =
            qobject_cast<QAbstractScrollArea*>(ancestor))
          {
          scroll = !scrollArea->verticalScrollBar()->isVisible();
          if (!scroll)
            {
            break;
            }
          }
        }
      break;
    default:
    case NeverScroll:
      break;
    }
  if (scroll)
    {
    this->QComboBox::wheelEvent(event);
    }
  else
    {
    event->ignore();
    }
}