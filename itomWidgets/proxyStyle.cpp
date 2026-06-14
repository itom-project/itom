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
#include <QApplication>
#include <QPointer>
#include <QStyleFactory>

// CTK includes
#include "proxyStyle.h"

// ----------------------------------------------------------------------------
class ProxyStylePrivate
{
  Q_DECLARE_PUBLIC(ProxyStyle)
protected:
  ProxyStyle* const q_ptr;
public:
  void setProxyStyle(QProxyStyle* proxy, QStyle *style)const;
  void setBaseStyle(QProxyStyle* proxyStyle, QStyle* baseStyle)const;
private:
  ProxyStylePrivate(ProxyStyle& object);
  mutable QPointer <QStyle> baseStyle;
  mutable bool ensureBaseStyleInProgress;
};

// ----------------------------------------------------------------------------
ProxyStylePrivate::ProxyStylePrivate(ProxyStyle& object)
  : q_ptr(&object)
  , ensureBaseStyleInProgress(false)
{
}

// ----------------------------------------------------------------------------
void ProxyStylePrivate::setProxyStyle(QProxyStyle* proxy, QStyle *style)const
{
  if (style->proxy() == proxy)
    {
    return;
    }
  this->setBaseStyle(proxy, style);
}
// ----------------------------------------------------------------------------
void ProxyStylePrivate::setBaseStyle(QProxyStyle* proxy, QStyle *style)const
{
  if (proxy->baseStyle() == style &&
      style->proxy() == proxy)
    {
    return;
    }
  QObject* parent = style->parent();
  QStyle* oldStyle = proxy->baseStyle();
  QObject* oldParent = oldStyle ? oldStyle->parent() : 0;
  if (oldParent == proxy)
    {
    oldStyle->setParent(0);// make sure setBaseStyle doesn't delete baseStyle
    }
  proxy->setBaseStyle(style);
  style->setParent(parent);
  if (oldParent == proxy)
    {
    oldStyle->setParent(oldParent);
    }
}

// ----------------------------------------------------------------------------
ProxyStyle::ProxyStyle(QStyle *style, QObject* parent)
  : d_ptr(new ProxyStylePrivate(*this))
{
  Q_D(ProxyStyle);
  d->baseStyle = style;
  this->setBaseStyle(style);
  this->setParent(parent);
}

// ----------------------------------------------------------------------------
ProxyStyle::~ProxyStyle()
{
  Q_D(ProxyStyle);
  if (!QApplication::closingDown() && d->baseStyle == QApplication::style())
    {
    d->baseStyle->setParent(qApp); // don't delete the application style.
    }
}

// ----------------------------------------------------------------------------
void ProxyStyle::ensureBaseStyle() const
{
  Q_D(const ProxyStyle);
  if (d->ensureBaseStyleInProgress)
  {
    // avoid infinite loop
    return;
  }
  d->ensureBaseStyleInProgress = true;
  d->baseStyle = this->baseStyle();
  // Set the proxy to the entire hierarchy.
  QProxyStyle* proxyStyle = const_cast<QProxyStyle*>(qobject_cast<const QProxyStyle*>(
    this->proxy() ? this->proxy() : this));
  QStyle* proxyBaseStyle = proxyStyle->baseStyle(); // calls ensureBaseStyle
  QStyle* baseStyle = proxyBaseStyle;
  while (baseStyle)
    {
    d->setProxyStyle(proxyStyle, baseStyle);// set proxy on itself to all children
    QProxyStyle* proxy = qobject_cast<QProxyStyle*>(baseStyle);
    baseStyle = proxy ? proxy->baseStyle() : 0;
    }
  d->setBaseStyle(proxyStyle, proxyBaseStyle);
  d->ensureBaseStyleInProgress = false;
}

// ----------------------------------------------------------------------------
void ProxyStyle::drawPrimitive(PrimitiveElement element, const QStyleOption *option, QPainter *painter, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->drawPrimitive(element, option, painter, widget);
}

// ----------------------------------------------------------------------------
void ProxyStyle::drawControl(ControlElement element, const QStyleOption *option, QPainter *painter, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->drawControl(element, option, painter, widget);
}

// ----------------------------------------------------------------------------
void ProxyStyle::drawComplexControl(ComplexControl control, const QStyleOptionComplex *option, QPainter *painter, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->drawComplexControl(control, option, painter, widget);
}

// ----------------------------------------------------------------------------
void ProxyStyle::drawItemText(QPainter *painter, const QRect &rect, int flags, const QPalette &pal, bool enabled,
                               const QString &text, QPalette::ColorRole textRole) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->drawItemText(painter, rect, flags, pal, enabled, text, textRole);
}

// ----------------------------------------------------------------------------
void ProxyStyle::drawItemPixmap(QPainter *painter, const QRect &rect, int alignment, const QPixmap &pixmap) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->drawItemPixmap(painter, rect, alignment, pixmap);
}

// ----------------------------------------------------------------------------
QSize ProxyStyle::sizeFromContents(ContentsType type, const QStyleOption *option, const QSize &size, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->sizeFromContents(type, option, size, widget);
}

// ----------------------------------------------------------------------------
QRect ProxyStyle::subElementRect(SubElement element, const QStyleOption *option, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->subElementRect(element, option, widget);
}

// ----------------------------------------------------------------------------
QRect ProxyStyle::subControlRect(ComplexControl cc, const QStyleOptionComplex *option, SubControl sc, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->subControlRect(cc, option, sc, widget);
}

// ----------------------------------------------------------------------------
QRect ProxyStyle::itemTextRect(const QFontMetrics &fm, const QRect &r, int flags, bool enabled, const QString &text) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->itemTextRect(fm, r, flags, enabled, text);
}

// ----------------------------------------------------------------------------
QRect ProxyStyle::itemPixmapRect(const QRect &r, int flags, const QPixmap &pixmap) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->itemPixmapRect(r, flags, pixmap);
}

// ----------------------------------------------------------------------------
QStyle::SubControl ProxyStyle::hitTestComplexControl(ComplexControl control, const QStyleOptionComplex *option, const QPoint &pos, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->hitTestComplexControl(control, option, pos, widget);
}

// ----------------------------------------------------------------------------
int ProxyStyle::styleHint(StyleHint hint, const QStyleOption *option, const QWidget *widget, QStyleHintReturn *returnData) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->styleHint(hint, option, widget, returnData);
}

// ----------------------------------------------------------------------------
int ProxyStyle::pixelMetric(PixelMetric metric, const QStyleOption *option, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->pixelMetric(metric, option, widget);
}

// ----------------------------------------------------------------------------
QPixmap ProxyStyle::standardPixmap(StandardPixmap standardPixmap, const QStyleOption *opt, const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->standardPixmap(standardPixmap, opt, widget);
}

// ----------------------------------------------------------------------------
QPixmap ProxyStyle::generatedIconPixmap(QIcon::Mode iconMode, const QPixmap &pixmap, const QStyleOption *opt) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->generatedIconPixmap(iconMode, pixmap, opt);
}

// ----------------------------------------------------------------------------
QPalette ProxyStyle::standardPalette() const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->standardPalette();
}

// ----------------------------------------------------------------------------
void ProxyStyle::polish(QWidget *widget)
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->polish(widget);
}

// ----------------------------------------------------------------------------
void ProxyStyle::polish(QPalette &pal)
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->polish(pal);
}

// ----------------------------------------------------------------------------
void ProxyStyle::polish(QApplication *app)
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->polish(app);
}

// ----------------------------------------------------------------------------
void ProxyStyle::unpolish(QWidget *widget)
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->unpolish(widget);
}

// ----------------------------------------------------------------------------
void ProxyStyle::unpolish(QApplication *app)
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    d->baseStyle->unpolish(app);
}

// ----------------------------------------------------------------------------
bool ProxyStyle::event(QEvent *e)
{
    Q_D(const ProxyStyle);
    if (e->type() != QEvent::ParentChange &&
        e->type() != QEvent::ChildRemoved &&
        e->type() != QEvent::ChildAdded)
      {
      this->ensureBaseStyle();
      }
    return !d->baseStyle.isNull() ? d->baseStyle->event(e) : false;
}

// ----------------------------------------------------------------------------
QIcon ProxyStyle::standardIconImplementation(StandardPixmap standardIcon,
                                              const QStyleOption *option,
                                              const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->standardIcon(standardIcon, option, widget);
}

// ----------------------------------------------------------------------------
int ProxyStyle::layoutSpacingImplementation(QSizePolicy::ControlType control1,
                                             QSizePolicy::ControlType control2,
                                             Qt::Orientation orientation,
                                             const QStyleOption *option,
                                             const QWidget *widget) const
{
    Q_D(const ProxyStyle);
    this->ensureBaseStyle();
    return d->baseStyle->layoutSpacing(control1, control2, orientation, option, widget);
}
