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

#ifndef POPUPWIDGET_H
#define POPUPWIDGET_H

// CTK includes
#include "basePopupWidget.h"

class PopupWidgetPrivate;

/// \ingroup Widgets
/// Description:
class ITOMWIDGETS_EXPORT PopupWidget : public BasePopupWidget
{
  Q_OBJECT

  /// Control whether the popup listens to the application and baseWidget
  /// events and decides if it needs to be permanently or temporarily hidden.
  /// You might want to setActive(false) when embedding the popup
  /// into a static layout intead of having it top-level (no parent).
  /// Consider also removing its windowFlags (Qt::ToolTip |
  /// Qt::FramelessWindowHint) and removing the baseWidget.
  /// True by default
  Q_PROPERTY( bool active READ isActive WRITE setActive)

  /// Control wether the popup automatically opens when the mouse
  /// enter the widget. True by default
  Q_PROPERTY( bool autoShow READ autoShow WRITE setAutoShow)

  /// Time in ms to wait before opening the popup if autoShow is set.
  /// 20ms by default
  Q_PROPERTY( int showDelay READ showDelay WRITE setShowDelay)

  /// Control wether the popup automatically closes when the mouse
  /// leaves the widget. True by default
  Q_PROPERTY( bool autoHide READ autoHide WRITE setAutoHide)

  /// Time in ms to wait before closing the popup if autoHide is set.
  /// 200ms by default
  Q_PROPERTY( int hideDelay READ hideDelay WRITE setHideDelay)

public:
  typedef BasePopupWidget Superclass;
  explicit PopupWidget(QWidget* parent = 0);
  virtual ~PopupWidget();

  bool isActive()const;
  void setActive(bool);

  bool autoShow()const;
  /// Calling setAutoShow automatically updates opens the popup if the cursor
  /// is above the popup or the base widget.
  void setAutoShow(bool);

  int showDelay()const;
  void setShowDelay(int delay);

  bool autoHide()const;
  /// Don't automatically close the popup when leaving the widget.
  /// Calling setAutoHide automatically updates the state close the popup
  /// if the mouse is not over the popup nor the base widget.
  void setAutoHide(bool autoHide);

  int hideDelay()const;
  void setHideDelay(int delay);

public Q_SLOTS:
  /// Convenient function that calls setAutoHide(!pin) and opens the popup
  /// if pin is true regardless of the value of \a AutoShow.
  /// It is typically connected with a checkable button to anchor the popup.
  void pinPopup(bool pin);

public:
  /// Reimplemented for internal reasons
  virtual void hidePopup();

protected:
  virtual void leaveEvent(QEvent* event);
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
  virtual void enterEvent(QEnterEvent* event);
#else
  virtual void enterEvent(QEvent* event);
#endif
  virtual bool eventFilter(QObject* obj, QEvent* event);

  /// Widget the popup is attached to. It opens right under \a baseWidget
  /// and if the PopupWidget sizepolicy contains the growFlag/shrinkFlag,
  /// it tries to resize itself to fit the same width of \a baseWidget.
  virtual void setBaseWidget(QWidget* baseWidget);

protected Q_SLOTS:
  void updatePopup();
  virtual void onEffectFinished();

private:
  Q_DECLARE_PRIVATE(PopupWidget);
  Q_DISABLE_COPY(PopupWidget);
};

#endif
