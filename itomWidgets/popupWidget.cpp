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
#include "popupWidget_p.h"

// -------------------------------------------------------------------------
PopupWidgetPrivate::PopupWidgetPrivate(PopupWidget& object)
  :Superclass(object)
{
  this->Active = false;
  this->AutoShow = true;
  this->ShowDelay = 20;
  this->AutoHide = true;
  this->HideDelay = 200;
}

// -------------------------------------------------------------------------
PopupWidgetPrivate::~PopupWidgetPrivate()
{
}

// -------------------------------------------------------------------------
void PopupWidgetPrivate::init()
{
  Q_Q(PopupWidget);
  this->setParent(q);
  q->setActive(true);
  this->Superclass::init();
}

// -------------------------------------------------------------------------
QWidget* PopupWidgetPrivate::mouseOver()
{
  Q_Q(PopupWidget);
  QWidget* widgetUnderCursor = this->Superclass::mouseOver();
  if (widgetUnderCursor &&
      !this->focusWidgets(true).contains(widgetUnderCursor))
    {
    widgetUnderCursor->installEventFilter(q);
    }
  return widgetUnderCursor;
}

// -------------------------------------------------------------------------
bool PopupWidgetPrivate::eventFilter(QObject* obj, QEvent* event)
{
  Q_Q(PopupWidget);
  QWidget* widget = qobject_cast<QWidget*>(obj);
  if (!widget)
  {
    return this->Superclass::eventFilter(obj, event);
  }

  // Here are the application events, it's a lot of events, so we need to be
  // careful to be fast.
  if (event->type() == QEvent::ApplicationDeactivate)
    {
    // We wait to see if there is no other window being active
    QTimer::singleShot(0, this, SLOT(onApplicationDeactivate()));
    }
  else if (event->type() == QEvent::ApplicationActivate)
    {
    QTimer::singleShot(0, this, SLOT(updateVisibility()));
    }
  if (this->BaseWidget.isNull())
    {
    return false;
    }
  if (event->type() == QEvent::Move && widget != this->BaseWidget)
    {
    if (widget->isAncestorOf(this->BaseWidget))
      {
      QMoveEvent* moveEvent = dynamic_cast<QMoveEvent*>(event);
      QPoint topLeft = widget->parentWidget() ? widget->parentWidget()->mapToGlobal(moveEvent->pos()) : moveEvent->pos();
      topLeft += this->BaseWidget->mapTo(widget, QPoint(0,0));
      //q->move(q->pos() + moveEvent->pos() - moveEvent->oldPos());
      QRect newBaseGeometry = this->baseGeometry();
	    newBaseGeometry.moveTopLeft(topLeft);
	    QRect desiredGeometry = this->desiredOpenGeometry(newBaseGeometry);
	    q->move(desiredGeometry.topLeft());
      }
    else if (widget->isWindow() &&
             widget->windowType() != Qt::ToolTip &&
             widget->windowType() != Qt::Popup)
      {
      QTimer::singleShot(0, this, SLOT(updateVisibility()));
      }
    }
  else if (event->type() == QEvent::Resize)
    {
    if (widget->isWindow() &&
        widget != this->BaseWidget->window() &&
        widget->windowType() != Qt::ToolTip &&
        widget->windowType() != Qt::Popup)
      {
      QTimer::singleShot(0, this, SLOT(updateVisibility()));
      }
    }
  else if (event->type() == QEvent::WindowStateChange &&
           widget != this->BaseWidget->window() &&
           widget->windowType() != Qt::ToolTip &&
           widget->windowType() != Qt::Popup)
    {
    QTimer::singleShot(0, this, SLOT(updateVisibility()));
    }
  else if ((event->type() == QEvent::WindowActivate ||
            event->type() == QEvent::WindowDeactivate) &&
           widget == this->BaseWidget->window())
    {
    QTimer::singleShot(0, this, SLOT(updateVisibility()));
    }
  else if (event->type() == QEvent::RequestSoftwareInputPanel)
    {
    qApp->setActiveWindow(widget->window());
    }
  return false;
}

// -------------------------------------------------------------------------
void PopupWidgetPrivate::onApplicationDeactivate()
{
  // Still no active window, that means the user now is controlling another
  // application, we have no control over when the other app moves over the
  // popup, so we hide the popup as it would show on top of the other app.
  if (!qApp->activeWindow())
    {
    this->temporarilyHiddenOn();
    }
}

// -------------------------------------------------------------------------
void PopupWidgetPrivate::updateVisibility()
{
  Q_Q(PopupWidget);
  // If the BaseWidget window is active, then there is no reason to cover the
  // popup.
  if (this->BaseWidget.isNull()  ||
      // the popupwidget active window is not active
      (!this->BaseWidget->window()->isActiveWindow() &&
      // and no other active window
       (!qApp->activeWindow() ||
      // or the active window is a popup/tooltip
        (qApp->activeWindow()->windowType() != Qt::ToolTip &&
         qApp->activeWindow()->windowType() != Qt::Popup))))
    {
    foreach(QWidget* topLevelWidget, qApp->topLevelWidgets())
      {
      // If there is at least 1 window (active or not) that covers the popup,
      // then we ensure the popup is hidden.
      // We have no way of knowing which toplevel is over (z-order) which one,
      // it is an OS specific information.
      // Of course, tooltips and popups don't count as covering windows.
      if (topLevelWidget->isVisible() &&
          !(topLevelWidget->windowState() & Qt::WindowMinimized) &&
          topLevelWidget->windowType() != Qt::ToolTip &&
          topLevelWidget->windowType() != Qt::Popup &&
          topLevelWidget != (this->BaseWidget ? this->BaseWidget->window() : 0) &&
          topLevelWidget->frameGeometry().intersects(q->geometry()))
        {
        //qDebug() << "hide" << q << "because of: " << topLevelWidget
        //         << " with windowType: " << topLevelWidget->windowType()
        //         << topLevelWidget->isVisible()
        //         << (this->BaseWidget ? this->BaseWidget->window() : 0)
        //         << topLevelWidget->frameGeometry();
        this->temporarilyHiddenOn();
        return;
        }
      }
    }
  // If the base widget is hidden or minimized, we don't want to restore the
  // popup.
  if (!this->BaseWidget.isNull() &&
      (!this->BaseWidget->isVisible() ||
        this->BaseWidget->window()->windowState() & Qt::WindowMinimized))
    {
    return;
    }
  // Restore the visibility of the popup if it was hidden
  this->temporarilyHiddenOff();
}

// -------------------------------------------------------------------------
void PopupWidgetPrivate::temporarilyHiddenOn()
{
  Q_Q(PopupWidget);
  if (!this->AutoHide &&
      (q->isVisible() || this->isOpening()) &&
      !(q->isHidden() || this->isClosing()))
    {
    this->setProperty("forcedClosed", this->isOpening() ? 2 : 1);
    }
  this->currentAnimation()->stop();
  this->hideAll();
}

// -------------------------------------------------------------------------
void PopupWidgetPrivate::temporarilyHiddenOff()
{
  Q_Q(PopupWidget);

  int forcedClosed = this->property("forcedClosed").toInt();
  if (forcedClosed > 0)
    {
    q->show();
    if (forcedClosed == 2)
      {
      emit q->popupOpened(true);
      }
    this->setProperty("forcedClosed", 0);
    }
  else
    {
    q->updatePopup();
    }
}

// -------------------------------------------------------------------------
// Qt::FramelessWindowHint is required on Windows for Translucent background
// Qt::Toolip is preferred to Qt::Popup as it would close itself at the first
// click outside the widget (typically a click in the BaseWidget)
PopupWidget::PopupWidget(QWidget* parentWidget)
  : Superclass(new PopupWidgetPrivate(*this), parentWidget)
{
  Q_D(PopupWidget);
  d->init();
}

// -------------------------------------------------------------------------
PopupWidget::~PopupWidget()
{
}

// -------------------------------------------------------------------------
bool PopupWidget::isActive()const
{
  Q_D(const PopupWidget);
  return d->Active;
}

// -------------------------------------------------------------------------
void PopupWidget::setActive(bool active)
{
  Q_D(PopupWidget);
  if (active == d->Active)
    {
    return;
    }
  d->Active = active;
  if (d->Active)
    {
    if (!d->BaseWidget.isNull())
      {
      d->BaseWidget->installEventFilter(this);
      }
    if (d->PopupPixmapWidget)
      {
      d->PopupPixmapWidget->installEventFilter(this);
      }
    qApp->installEventFilter(d);
    }
  else // not active
    {
    if (!d->BaseWidget.isNull())
      {
      d->BaseWidget->removeEventFilter(this);
      }
    if (d->PopupPixmapWidget)
      {
      d->PopupPixmapWidget->removeEventFilter(this);
      }
    qApp->removeEventFilter(d);
    }
}

// -------------------------------------------------------------------------
void PopupWidget::setBaseWidget(QWidget* widget)
{
  Q_D(PopupWidget);
  if (!d->BaseWidget.isNull())
    {
    d->BaseWidget->removeEventFilter(this);
    }
  this->Superclass::setBaseWidget(widget);
  if (!d->BaseWidget.isNull() && d->Active)
    {
    d->BaseWidget->installEventFilter(this);
    }
  QTimer::singleShot(d->ShowDelay, this, SLOT(updatePopup()));
}

// -------------------------------------------------------------------------
bool PopupWidget::autoShow()const
{
  Q_D(const PopupWidget);
  return d->AutoShow;
}

// -------------------------------------------------------------------------
void PopupWidget::setAutoShow(bool mode)
{
  Q_D(PopupWidget);
  d->AutoShow = mode;
  QTimer::singleShot(d->ShowDelay, this, SLOT(updatePopup()));
}

// -------------------------------------------------------------------------
int PopupWidget::showDelay()const
{
  Q_D(const PopupWidget);
  return d->ShowDelay;
}

// -------------------------------------------------------------------------
void PopupWidget::setShowDelay(int delay)
{
  Q_D(PopupWidget);
  d->ShowDelay = delay;
}

// -------------------------------------------------------------------------
bool PopupWidget::autoHide()const
{
  Q_D(const PopupWidget);
  return d->AutoHide;
}

// -------------------------------------------------------------------------
void PopupWidget::setAutoHide(bool mode)
{
  Q_D(PopupWidget);
  d->AutoHide = mode;
  QTimer::singleShot(d->HideDelay, this, SLOT(updatePopup()));
}

// -------------------------------------------------------------------------
int PopupWidget::hideDelay()const
{
  Q_D(const PopupWidget);
  return d->HideDelay;
}

// -------------------------------------------------------------------------
void PopupWidget::setHideDelay(int delay)
{
  Q_D(PopupWidget);
  d->HideDelay = delay;
}

// -------------------------------------------------------------------------
void PopupWidget::onEffectFinished()
{
  Q_D(PopupWidget);
  bool wasClosing = d->wasClosing();
  this->Superclass::onEffectFinished();
  if (wasClosing)
    {
    /// restore the AutoShow if needed.
    if (!this->property("AutoShowOnClose").isNull())
      {
      d->AutoShow = this->property("AutoShowOnClose").toBool();
      this->setProperty("AutoShowOnClose", QVariant());
      }
    }
}

// --------------------------------------------------------------------------
void PopupWidget::leaveEvent(QEvent* event)
{
  Q_D(PopupWidget);
  QTimer::singleShot(d->HideDelay, this, SLOT(updatePopup()));
  this->Superclass::leaveEvent(event);
}

// --------------------------------------------------------------------------
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
void PopupWidget::enterEvent(QEnterEvent* event)
{
  Q_D(PopupWidget);
  QTimer::singleShot(d->ShowDelay, this, SLOT(updatePopup()));
  this->Superclass::enterEvent(event);
}
#else
void PopupWidget::enterEvent(QEvent* event)
{
    Q_D(PopupWidget);
    QTimer::singleShot(d->ShowDelay, this, SLOT(updatePopup()));
    this->Superclass::enterEvent(event);
}
#endif

// --------------------------------------------------------------------------
bool PopupWidget::eventFilter(QObject* obj, QEvent* event)
{
  Q_D(PopupWidget);
  // Here we listen to PopupPixmapWidget, BaseWidget and PopupWidget
  // children popups that were under the mouse
  switch(event->type())
    {
    case QEvent::Move:
      {
      if (obj != d->BaseWidget)
        {
        break;
        }
      QMoveEvent* moveEvent = dynamic_cast<QMoveEvent*>(event);
      QRect newBaseGeometry = d->baseGeometry();
      newBaseGeometry.moveTopLeft(d->mapToGlobal(moveEvent->pos()));
      QRect desiredGeometry = d->desiredOpenGeometry(newBaseGeometry);
      this->move(desiredGeometry.topLeft());
      //this->move(this->pos() + moveEvent->pos() - moveEvent->oldPos());
      this->update();
      break;
      }
    case QEvent::Hide:
    case QEvent::Close:
      // if the mouse was in a base widget child popup, then when we leave
      // the popup we want to check if it needs to be closed.
      if (obj != d->BaseWidget)
        {
        if (obj != d->PopupPixmapWidget &&
            qobject_cast<QWidget*>(obj)->windowType() == Qt::Popup)
          {
          obj->removeEventFilter(this);
          QTimer::singleShot(d->HideDelay, this, SLOT(updatePopup()));
          }
        break;
        }
      d->temporarilyHiddenOn();
      break;
    case QEvent::Show:
      if (obj != d->BaseWidget)
        {
	      break;
	      }
	    this->setGeometry(d->desiredOpenGeometry());
	    d->temporarilyHiddenOff();
	    break;
	  case QEvent::Resize:
	    if (obj != d->BaseWidget ||
	        !(d->Alignment & Qt::AlignJustify ||
	         (d->Alignment & Qt::AlignTop && d->Alignment & Qt::AlignBottom)) ||
	         !(d->isOpening() || this->isVisible()))
	      {
	      break;
	      }
	    // TODO: bug when the effect is WindowOpacityFadeEffect
	    this->setGeometry(d->desiredOpenGeometry());
	    break;
    case QEvent::Enter:
      if ( d->currentAnimation()->state() == QAbstractAnimation::Stopped )
        {
        // Maybe the user moved the mouse on the widget by mistake, don't open
        // the popup instantly...
        QTimer::singleShot(d->ShowDelay, this, SLOT(updatePopup()));
        }
      else
        {
        // ... except if the popup is closing, we want to reopen it as sooon as
        // possible.
        this->updatePopup();
        }
      break;
    case QEvent::Leave:
      // Don't listen to base widget children that are popups as what
      // matters here is their close event instead
      if (obj != d->BaseWidget &&
          obj != d->PopupPixmapWidget &&
          qobject_cast<QWidget*>(obj)->windowType() == Qt::Popup)
        {
        break;
        }
      // The mouse might have left the area that keeps the popup open
      QTimer::singleShot(d->HideDelay, this, SLOT(updatePopup()));
      if (obj != d->BaseWidget &&
          obj != d->PopupPixmapWidget)
        {
        obj->removeEventFilter(this);
        }
      break;
    default:
      break;
    }
  return this->QObject::eventFilter(obj, event);
}

// --------------------------------------------------------------------------
void PopupWidget::updatePopup()
{
  Q_D(PopupWidget);

  // Querying mouseOver can be slow, don't do it if not needed.
  QWidget* mouseOver = (d->AutoShow || d->AutoHide) ? d->mouseOver() : 0;
  if ((d->AutoShow ||
     // Even if there is no AutoShow, we might still want to reopen the popup
     // when closing it inadvertently, except if we are un-pin-ing the popup
      (d->AutoHide && d->isClosing() && this->property("AutoShowOnClose").toBool())) &&
     // to be automatically open, the mouse has to be over a child widget
      mouseOver &&
     // disable opening the popup when the popup is disabled
      (d->BaseWidget.isNull() || d->BaseWidget->isEnabled()))
    {
    this->showPopup();
    }
  else if (d->AutoHide && !mouseOver)
    {
    this->hidePopup();
    }
}


// --------------------------------------------------------------------------
void PopupWidget::hidePopup()
{
  // just in case it was set.
  this->setProperty("forcedClosed", 0);

  this->Superclass::hidePopup();
}

// --------------------------------------------------------------------------
void PopupWidget::pinPopup(bool pin)
{
  Q_D(PopupWidget);
  this->setAutoHide(!pin);
  if (pin)
    {
    this->showPopup();
    }
  else
    {
    // When closing, we don't want to inadvertently re-open the menu.
    this->setProperty("AutoShowOnClose", this->autoShow());
    d->AutoShow = false;
    this->hidePopup();
    }
}
