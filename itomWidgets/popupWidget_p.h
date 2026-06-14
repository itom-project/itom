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

#ifndef POPUPWIDGET_P_H
#define POPUPWIDGET_P_H

// CTK includes
#include "basePopupWidget_p.h"
#include "popupWidget.h"

// -------------------------------------------------------------------------
/// \ingroup Widgets
class ITOMWIDGETS_EXPORT PopupWidgetPrivate
  : public BasePopupWidgetPrivate
{
  Q_OBJECT
  Q_DECLARE_PUBLIC(PopupWidget);
public:
  typedef BasePopupWidgetPrivate Superclass;
  PopupWidgetPrivate(PopupWidget& object);
  ~PopupWidgetPrivate();

  virtual void init();

  // Return the widget if the mouse cursor is above any of the focus widgets or their
  // children.
  // If the cursor is above a child widget, install the event filter to listen
  // when the cursor leaves the widget.
  virtual QWidget* mouseOver();

  virtual bool eventFilter(QObject* obj, QEvent* event);

  void temporarilyHiddenOn();
  void temporarilyHiddenOff();

public Q_SLOTS:
  void updateVisibility();
  void onApplicationDeactivate();

protected:
  bool Active;
  bool AutoShow;
  int  ShowDelay;
  bool AutoHide;
  int  HideDelay;
};

#endif
