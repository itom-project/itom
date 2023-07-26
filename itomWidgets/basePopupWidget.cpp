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
#include <QDebug>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <QDesktopWidget>
#endif
#include <QDir>
#include <QEvent>
#include <QLabel>
#include <QLayout>
#include <QMouseEvent>
#include <QMoveEvent>
#include <QPainter>
#include <QPointer>
#include <QPropertyAnimation>
#include <QStyle>
#include <QTimer>

// CTK includes
#include "basePopupWidget_p.h"

// -------------------------------------------------------------------------
QGradient* duplicateGradient(const QGradient* gradient)
{
  QGradient* newGradient = 0;
  switch (gradient->type())
    {
    case QGradient::LinearGradient:
      {
      const QLinearGradient* linearGradient = static_cast<const QLinearGradient*>(gradient);
      newGradient = new QLinearGradient(linearGradient->start(), linearGradient->finalStop());
      break;
      }
    case QGradient::RadialGradient:
      {
      const QRadialGradient* radialGradient = static_cast<const QRadialGradient*>(gradient);
      newGradient = new QRadialGradient(radialGradient->center(), radialGradient->radius());
      break;
      }
    case QGradient::ConicalGradient:
      {
      const QConicalGradient* conicalGradient = static_cast<const QConicalGradient*>(gradient);
      newGradient = new QConicalGradient(conicalGradient->center(), conicalGradient->angle());
      break;
      }
    default:
      break;
    }
  if (!newGradient)
    {
    Q_ASSERT(gradient->type() != QGradient::NoGradient);
    return newGradient;
    }
  newGradient->setCoordinateMode(gradient->coordinateMode());
  newGradient->setSpread(gradient->spread());
  newGradient->setStops(gradient->stops());
  return newGradient;
}

// -------------------------------------------------------------------------
BasePopupWidgetPrivate::BasePopupWidgetPrivate(BasePopupWidget& object)
  :q_ptr(&object)
{
  this->Effect = BasePopupWidget::FadeEffect;
  this->EffectDuration = 30; // in ms
  this->EffectAlpha = 1.;
  this->AlphaAnimation = 0;
  this->ForcedTranslucent = false;
  this->ScrollAnimation = 0;
  this->PopupPixmapWidget = 0;
  // Geometry attributes
  this->Alignment = Qt::AlignJustify | Qt::AlignBottom;
  this->Orientations = Qt::Vertical;
  this->VerticalDirection = BasePopupWidget::TopToBottom;
  this->HorizontalDirection = Qt::LeftToRight;
}

// -------------------------------------------------------------------------
BasePopupWidgetPrivate::~BasePopupWidgetPrivate()
{
}

// -------------------------------------------------------------------------
void BasePopupWidgetPrivate::init()
{
  Q_Q(BasePopupWidget);
  // By default, Tooltips are shown only on active windows. In a popup widget
  // case, we sometimes aren't the active window but we still would like to
  // show the children tooltips.
  q->setAttribute(Qt::WA_AlwaysShowToolTips, true);

  this->AlphaAnimation = new QPropertyAnimation(q, "effectAlpha", q);
  this->AlphaAnimation->setDuration(this->EffectDuration);
  this->AlphaAnimation->setStartValue(0.);
  this->AlphaAnimation->setEndValue(1.);
  QObject::connect(this->AlphaAnimation, SIGNAL(finished()),
                   q, SLOT(onEffectFinished()));

  this->PopupPixmapWidget = new QLabel(q, Qt::ToolTip | Qt::FramelessWindowHint);

  this->ScrollAnimation = new QPropertyAnimation(q, "effectGeometry", q);
  this->ScrollAnimation->setDuration(this->EffectDuration);
  QObject::connect(this->ScrollAnimation, SIGNAL(finished()),
                   q, SLOT(onEffectFinished()));
  QObject::connect(this->ScrollAnimation, SIGNAL(finished()),
                   this->PopupPixmapWidget, SLOT(hide()));

  q->setAnimationEffect(this->Effect);
  q->setEasingCurve(QEasingCurve::OutCubic);
  q->setBaseWidget(q->parentWidget());
}

// -------------------------------------------------------------------------
QPropertyAnimation* BasePopupWidgetPrivate::currentAnimation()const
{
  return this->Effect == BasePopupWidget::ScrollEffect ?
    this->ScrollAnimation : this->AlphaAnimation;
}

// -------------------------------------------------------------------------
bool BasePopupWidgetPrivate::isOpening()const
{
  return this->currentAnimation()->state() == QAbstractAnimation::Running &&
    this->currentAnimation()->direction() == QAbstractAnimation::Forward;
}

// -------------------------------------------------------------------------
bool BasePopupWidgetPrivate::isClosing()const
{
  return this->currentAnimation()->state() == QAbstractAnimation::Running &&
    this->currentAnimation()->direction() == QAbstractAnimation::Backward;
}

// -------------------------------------------------------------------------
bool BasePopupWidgetPrivate::wasClosing()const
{
  Q_Q(const BasePopupWidget);
  return qobject_cast<QAbstractAnimation*>(q->sender())->direction()
    == QAbstractAnimation::Backward;
}

// -------------------------------------------------------------------------
QWidgetList BasePopupWidgetPrivate::focusWidgets(bool onlyVisible)const
{
  Q_Q(const BasePopupWidget);
  QWidgetList res;
  if (!onlyVisible || q->isVisible())
    {
    res << const_cast<BasePopupWidget*>(q);
    }
  if (!this->BaseWidget.isNull() && (!onlyVisible || this->BaseWidget->isVisible()))
    {
    res << this->BaseWidget;
    }
  if (this->PopupPixmapWidget && (!onlyVisible || this->PopupPixmapWidget->isVisible()))
    {
    res << this->PopupPixmapWidget;
    }
  return res;
}

// -------------------------------------------------------------------------
QWidget* BasePopupWidgetPrivate::mouseOver()
{
  QList<QWidget*> widgets = this->focusWidgets(true);
  foreach(QWidget* widget, widgets)
    {
    if (widget->underMouse())
      {
      return widget;
      }
    }
  // Warning QApplication::widgetAt(QCursor::pos()) can be a bit slow...
  const QPoint pos = QCursor::pos();
  QWidget* widgetUnderCursor = qApp->widgetAt(pos);
  foreach(const QWidget* focusWidget, widgets)
    {
    if (this->isAncestorOf(focusWidget, widgetUnderCursor) &&
        // Ignore when cursor is above a title bar of a focusWidget, underMouse
        // wouldn't have return false, but QApplication::widgetAt would return
        // the widget
        (focusWidget != widgetUnderCursor ||
         QRect(QPoint(0,0), focusWidget->size()).contains(
          focusWidget->mapFromGlobal(pos))))
      {
      return widgetUnderCursor;
      }
    }
  return 0;
}

// -------------------------------------------------------------------------
bool BasePopupWidgetPrivate::isAncestorOf(const QWidget* ancestor, const QWidget* child)const
{
  while (child)
    {
    if (child == ancestor)
      {
      return true;
      }
    child = child->parentWidget();
    }
  return false;
}

// -------------------------------------------------------------------------
void BasePopupWidgetPrivate::setupPopupPixmapWidget()
{
  Q_Q(BasePopupWidget);
  this->PopupPixmapWidget->setAlignment(this->pixmapAlignment());
  QPixmap pixmap;
  if (q->testAttribute(Qt::WA_TranslucentBackground))
    {
    // only QImage handle transparency correctly
    QImage image(q->geometry().size(), QImage::Format_ARGB32);
    image.fill(0);
    q->render(&image);
    pixmap = QPixmap::fromImage(image);
    }
  else
    {
      q->grab(QRect(QPoint(0, 0), q->geometry().size()));
    }
  this->PopupPixmapWidget->setPixmap(pixmap);
  this->PopupPixmapWidget->setAttribute(
    Qt::WA_TranslucentBackground, q->testAttribute(Qt::WA_TranslucentBackground));
  this->PopupPixmapWidget->setWindowOpacity(q->windowOpacity());
}

// -------------------------------------------------------------------------
Qt::Alignment BasePopupWidgetPrivate::pixmapAlignment()const
{
  Qt::Alignment alignment;
  if (this->VerticalDirection == BasePopupWidget::TopToBottom)
    {
    alignment |= Qt::AlignBottom;
    }
  else// if (this->VerticalDirection == BasePopupWidget::BottomToTop)
    {
    alignment |= Qt::AlignTop;
    }

  if (this->HorizontalDirection == Qt::LeftToRight)
    {
    alignment |= Qt::AlignRight;
    }
  else// if (this->VerticalDirection == BasePopupWidget::BottomToTop)
    {
    alignment |= Qt::AlignLeft;
    }
  return alignment;
}

// -------------------------------------------------------------------------
QRect BasePopupWidgetPrivate::closedGeometry()const
{
  Q_Q(const BasePopupWidget);
  return this->closedGeometry(q->geometry());
}

// -------------------------------------------------------------------------
QRect BasePopupWidgetPrivate::closedGeometry(QRect openGeom)const
{
  if (this->Orientations & Qt::Vertical)
    {
    if (this->VerticalDirection == BasePopupWidget::BottomToTop)
      {
      openGeom.moveTop(openGeom.bottom());
      }
    openGeom.setHeight(0);
    }
  if (this->Orientations & Qt::Horizontal)
    {
    if (this->HorizontalDirection == Qt::RightToLeft)
      {
      openGeom.moveLeft(openGeom.right());
      }
    openGeom.setWidth(0);
    }
  return openGeom;
}

// -------------------------------------------------------------------------
QRect BasePopupWidgetPrivate::baseGeometry()const
{
  if (this->BaseWidget.isNull())
    {
    return QRect();
    }
  return QRect(this->mapToGlobal(this->BaseWidget->geometry().topLeft()),
               this->BaseWidget->size());
}

// -------------------------------------------------------------------------
QPoint BasePopupWidgetPrivate::mapToGlobal(const QPoint& baseWidgetPoint)const
{
  QPoint mappedPoint = baseWidgetPoint;
  if (!this->BaseWidget.isNull() && this->BaseWidget->parentWidget())
    {
    mappedPoint = this->BaseWidget->parentWidget()->mapToGlobal(mappedPoint);
    }
  return mappedPoint;
}

// -------------------------------------------------------------------------
QRect BasePopupWidgetPrivate::desiredOpenGeometry()const
{
  return this->desiredOpenGeometry(this->baseGeometry());
}

// -------------------------------------------------------------------------
QRect BasePopupWidgetPrivate::desiredOpenGeometry(QRect baseGeometry)const
{
  Q_Q(const BasePopupWidget);
  QSize size = q->size();
  if (!q->testAttribute(Qt::WA_WState_Created))
    {
    size = q->sizeHint();
    }

  if (baseGeometry.isNull())
    {
    return QRect(q->pos(), size);
    }

  QRect geometry;
  if (this->Alignment & Qt::AlignJustify)
    {
    if (this->Orientations & Qt::Vertical)
      {
      size.setWidth(baseGeometry.width());
      }
    }
  if (this->Alignment & Qt::AlignTop &&
      this->Alignment & Qt::AlignBottom)
    {
    size.setHeight(baseGeometry.height());
    }

  geometry.setSize(size);

  QPoint topLeft = baseGeometry.topLeft();
  QPoint bottomRight = baseGeometry.bottomRight();

  if (this->Alignment & Qt::AlignLeft)
    {
    if (this->HorizontalDirection == Qt::LeftToRight)
      {
      geometry.moveLeft(topLeft.x());
      }
    else
      {
      geometry.moveRight(topLeft.x() - 1);
      }
    }
  else if (this->Alignment & Qt::AlignRight)
    {
    if (this->HorizontalDirection == Qt::LeftToRight)
      {
      geometry.moveLeft(bottomRight.x() + 1);
      }
    else
      {
      geometry.moveRight(bottomRight.x());
      }
    }
  else if (this->Alignment & Qt::AlignHCenter)
    {
    geometry.moveLeft((topLeft.x() + bottomRight.x()) / 2 - size.width() / 2);
    }
  else if (this->Alignment & Qt::AlignJustify)
    {
    geometry.moveLeft(topLeft.x());
    }

  if (this->Alignment & Qt::AlignTop)
    {
    if (this->VerticalDirection == BasePopupWidget::TopToBottom)
      {
      geometry.moveTop(topLeft.y());
      }
    else
      {
      geometry.moveBottom(topLeft.y() - 1);
      }
    }
  else if (this->Alignment & Qt::AlignBottom)
    {
    if (this->VerticalDirection == BasePopupWidget::TopToBottom)
      {
      geometry.moveTop(bottomRight.y() + 1);
      }
    else
      {
      geometry.moveBottom(bottomRight.y());
      }
    }
  else if (this->Alignment & Qt::AlignVCenter)
    {
    geometry.moveTop((topLeft.y() + bottomRight.y()) / 2 - size.height() / 2);
    }
  return geometry;
}

// -------------------------------------------------------------------------
void BasePopupWidgetPrivate::hideAll()
{
  Q_Q(BasePopupWidget);

  // It is possible to have the popup widget not being a popup but inside
  // a layout: maybe the popup has been pin-down in a way that it gets parented
  // In that case, there is no reason to hide the popup.
  if (!(q->windowFlags() & Qt::ToolTip))
    {
    return;
    }

  // Before hiding, transfer the active window flag to its parent, this will
  // prevent the application to send a ApplicationDeactivate signal that
  // doesn't need to be done.
  if (q->isActiveWindow() && !this->BaseWidget.isNull())
    {
    qApp->setActiveWindow(this->BaseWidget->window());
    }

  q->hide();
  this->PopupPixmapWidget->hide();

  // If there is a popup open in the BasePopupWidget children, then hide it
  // as well so we don't have a popup open while the BasePopupWidget is hidden.
  QPointer<QWidget> activePopupWidget = qApp->activePopupWidget();
  if (activePopupWidget && this->isAncestorOf(q, activePopupWidget))
    {
    activePopupWidget->close();
    }
}

// -------------------------------------------------------------------------
// Qt::FramelessWindowHint is required on Windows for Translucent background
// Qt::Toolip is preferred to Qt::Popup as it would close itself at the first
// click outside the widget (typically a click in the BaseWidget)
BasePopupWidget::BasePopupWidget(QWidget* parentWidget)
  //: Superclass(QApplication::desktop()->screen(QApplication::desktop()->screenNumber(parentWidget)),
  : Superclass(parentWidget,
               Qt::ToolTip | Qt::FramelessWindowHint)
  , d_ptr(new BasePopupWidgetPrivate(*this))
{
  Q_D(BasePopupWidget);
  d->init();
}

// -------------------------------------------------------------------------
BasePopupWidget::BasePopupWidget(BasePopupWidgetPrivate* pimpl, QWidget* parentWidget)
  //: //Superclass(QApplication::desktop()->screen(QApplication::desktop()->screenNumber(parentWidget)),
  : Superclass(parentWidget,
               Qt::ToolTip | Qt::FramelessWindowHint)
  , d_ptr(pimpl)
{
}

// -------------------------------------------------------------------------
BasePopupWidget::~BasePopupWidget()
{
}

// -------------------------------------------------------------------------
QWidget* BasePopupWidget::baseWidget()const
{
  Q_D(const BasePopupWidget);
  return d->BaseWidget;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setBaseWidget(QWidget* widget)
{
  Q_D(BasePopupWidget);
  if (!d->BaseWidget.isNull())
    {
    //disconnect(d->BaseWidget, SIGNAL(destroyed(QObject*)),
    //           this, SLOT(onBaseWidgetDestroyed()));
    }
  d->BaseWidget = widget;
  if (!d->BaseWidget.isNull())
    {
    //connect(d->BaseWidget, SIGNAL(destroyed(QObject*)),
    //        this, SLOT(onBaseWidgetDestroyed()));
    }
}

// -------------------------------------------------------------------------
void BasePopupWidget::onBaseWidgetDestroyed()
{
  Q_D(BasePopupWidget);
  d->hideAll();
  this->setBaseWidget(0);
  // could be a property.
  this->deleteLater();
}

// -------------------------------------------------------------------------
BasePopupWidget::AnimationEffect BasePopupWidget::animationEffect()const
{
  Q_D(const BasePopupWidget);
  return d->Effect;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setAnimationEffect(BasePopupWidget::AnimationEffect effect)
{
  Q_D(BasePopupWidget);
  /// TODO: handle the case where there is an animation running
  d->Effect = effect;
}

// -------------------------------------------------------------------------
int BasePopupWidget::effectDuration()const
{
  Q_D(const BasePopupWidget);
  return d->EffectDuration;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setEffectDuration(int duration)
{
  Q_D(BasePopupWidget);
  d->EffectDuration = duration;
  d->AlphaAnimation->setDuration(d->EffectDuration);
  d->ScrollAnimation->setDuration(d->EffectDuration);
}

// -------------------------------------------------------------------------
QEasingCurve::Type BasePopupWidget::easingCurve()const
{
  Q_D(const BasePopupWidget);
  return d->AlphaAnimation->easingCurve().type();
}

// -------------------------------------------------------------------------
void BasePopupWidget::setEasingCurve(QEasingCurve::Type easingCurve)
{
  Q_D(BasePopupWidget);
  d->AlphaAnimation->setEasingCurve(easingCurve);
  d->ScrollAnimation->setEasingCurve(easingCurve);
}

// -------------------------------------------------------------------------
Qt::Alignment BasePopupWidget::alignment()const
{
  Q_D(const BasePopupWidget);
  return d->Alignment;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setAlignment(Qt::Alignment alignment)
{
  Q_D(BasePopupWidget);
  d->Alignment = alignment;
}

// -------------------------------------------------------------------------
Qt::Orientations BasePopupWidget::orientation()const
{
  Q_D(const BasePopupWidget);
  return d->Orientations;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setOrientation(Qt::Orientations orientations)
{
  Q_D(BasePopupWidget);
  d->Orientations = orientations;
}

// -------------------------------------------------------------------------
BasePopupWidget::VerticalDirection BasePopupWidget::verticalDirection()const
{
  Q_D(const BasePopupWidget);
  return d->VerticalDirection;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setVerticalDirection(BasePopupWidget::VerticalDirection verticalDirection)
{
  Q_D(BasePopupWidget);
  d->VerticalDirection = verticalDirection;
}

// -------------------------------------------------------------------------
Qt::LayoutDirection BasePopupWidget::horizontalDirection()const
{
  Q_D(const BasePopupWidget);
  return d->HorizontalDirection;
}

// -------------------------------------------------------------------------
void BasePopupWidget::setHorizontalDirection(Qt::LayoutDirection horizontalDirection)
{
  Q_D(BasePopupWidget);
  d->HorizontalDirection = horizontalDirection;
}

// -------------------------------------------------------------------------
void BasePopupWidget::onEffectFinished()
{
  Q_D(BasePopupWidget);
  if (d->ForcedTranslucent)
    {
    d->ForcedTranslucent = false;
    this->setAttribute(Qt::WA_TranslucentBackground, false);
    }
  if (d->wasClosing())
    {
    d->hideAll();
    emit this->popupOpened(false);
    }
  else
    {
    this->show();
    emit this->popupOpened(true);
    }
}

// -------------------------------------------------------------------------
bool BasePopupWidget::event(QEvent* event)
{
  switch(event->type())
    {
    case QEvent::ParentChange:
      // For now the base widget is the parent widget
      this->setBaseWidget(this->parentWidget());
      break;
    default:
      break;
    }
  return this->Superclass::event(event);
}

// -------------------------------------------------------------------------
void BasePopupWidget::paintEvent(QPaintEvent* event)
{
  Q_D(BasePopupWidget);
  Q_UNUSED(event);

  QPainter painter(this);
  QBrush brush = this->palette().window();
  if (brush.style() == Qt::LinearGradientPattern ||
      brush.style() == Qt::ConicalGradientPattern ||
      brush.style() == Qt::RadialGradientPattern)
    {
    QGradient* newGradient = duplicateGradient(brush.gradient());
    QGradientStops stops;
    foreach(QGradientStop stop, newGradient->stops())
      {
      stop.second.setAlpha(stop.second.alpha() * d->EffectAlpha);
      stops.push_back(stop);
      }
    newGradient->setStops(stops);
    brush = QBrush(*newGradient);
    delete newGradient;
    }
  else
    {
    QColor color = brush.color();
    color.setAlpha(color.alpha() * d->EffectAlpha);
    brush.setColor(color);
    }
  //QColor semiTransparentColor = this->palette().window().color();
  //semiTransparentColor.setAlpha(d->CurrentAlpha);
  painter.fillRect(this->rect(), brush);
  painter.end();
  // Let the QFrame draw itself if needed
  this->Superclass::paintEvent(event);
}

// --------------------------------------------------------------------------
void BasePopupWidget::showPopup()
{
  Q_D(BasePopupWidget);

  if ((this->isVisible() &&
       d->currentAnimation()->state() == QAbstractAnimation::Stopped) ||
      (!d->BaseWidget.isNull() && !d->BaseWidget->isVisible()))
    {
    return;
    }

  // If the layout has never been activated, the widget doesn't know its
  // minSize/maxSize and we then wouldn't know what's its true geometry.
  if (this->layout() && !this->testAttribute(Qt::WA_WState_Created))
    {
    this->layout()->activate();
    }
  this->setGeometry(d->desiredOpenGeometry());
  /// Maybe the popup doesn't allow the desiredOpenGeometry if the widget
  /// minimum size is larger than the desired size.
  QRect openGeometry = this->geometry();
  QRect closedGeometry = d->closedGeometry();

  d->currentAnimation()->setDirection(QAbstractAnimation::Forward);

  switch(d->Effect)
    {
    case WindowOpacityFadeEffect:
      if (!this->testAttribute(Qt::WA_TranslucentBackground))
        {
        d->ForcedTranslucent = true;
        this->setAttribute(Qt::WA_TranslucentBackground, true);
        }
      this->show();
      break;
    case ScrollEffect:
      {
      d->PopupPixmapWidget->setGeometry(closedGeometry);
      d->ScrollAnimation->setStartValue(closedGeometry);
      d->ScrollAnimation->setEndValue(openGeometry);
      d->setupPopupPixmapWidget();
      d->PopupPixmapWidget->show();
      break;
      }
    default:
      break;
    }
  switch(d->currentAnimation()->state())
    {
    case QAbstractAnimation::Stopped:
      d->currentAnimation()->start();
      break;
    case QAbstractAnimation::Paused:
      d->currentAnimation()->resume();
      break;
    default:
    case QAbstractAnimation::Running:
      break;
    }
}

// --------------------------------------------------------------------------
void BasePopupWidget::hidePopup()
{
  Q_D(BasePopupWidget);

  if (!this->isVisible() &&
      d->currentAnimation()->state() == QAbstractAnimation::Stopped)
    {
    return;
    }
  d->currentAnimation()->setDirection(QAbstractAnimation::Backward);

  QRect openGeometry = this->geometry();
  QRect closedGeometry = d->closedGeometry();

  switch(d->Effect)
    {
    case WindowOpacityFadeEffect:
      if (!this->testAttribute(Qt::WA_TranslucentBackground))
        {
        d->ForcedTranslucent = true;
        this->setAttribute(Qt::WA_TranslucentBackground, true);
        }
      break;
    case ScrollEffect:
      {
      d->ScrollAnimation->setStartValue(closedGeometry);
      d->ScrollAnimation->setEndValue(openGeometry);
      d->setupPopupPixmapWidget();
      d->PopupPixmapWidget->setGeometry(this->geometry());
      d->PopupPixmapWidget->show();
      if (this->isActiveWindow())
        {
        qApp->setActiveWindow(!d->BaseWidget.isNull() ? d->BaseWidget->window() : 0);
        }
      this->hide();
      break;
      }
    default:
      break;
    }
  switch(d->currentAnimation()->state())
    {
    case QAbstractAnimation::Stopped:
      d->currentAnimation()->start();
      break;
    case QAbstractAnimation::Paused:
      d->currentAnimation()->resume();
      break;
    default:
    case QAbstractAnimation::Running:
      break;
    }
}

// --------------------------------------------------------------------------
double BasePopupWidget::effectAlpha()const
{
  Q_D(const BasePopupWidget);
  return d->EffectAlpha;
}

// --------------------------------------------------------------------------
void BasePopupWidget::setEffectAlpha(double alpha)
{
  Q_D(BasePopupWidget);
  d->EffectAlpha = alpha;
  this->repaint();
}

// --------------------------------------------------------------------------
QRect BasePopupWidget::effectGeometry()const
{
  Q_D(const BasePopupWidget);
  return d->PopupPixmapWidget->geometry();
}

// --------------------------------------------------------------------------
void BasePopupWidget::setEffectGeometry(QRect newGeometry)
{
  Q_D(BasePopupWidget);
  d->PopupPixmapWidget->setGeometry(newGeometry);
  d->PopupPixmapWidget->repaint();
}
