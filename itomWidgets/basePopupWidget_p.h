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

#ifndef BASEPOPUPWIDGET_P_H
#define BASEPOPUPWIDGET_P_H

// Qt includes
#include <QPointer>
class QLabel;
class QPropertyAnimation;

// CTK includes
#include "basePopupWidget.h"

// -------------------------------------------------------------------------
/// \ingroup Widgets
class ITOMWIDGETS_EXPORT BasePopupWidgetPrivate
  : public QObject
{
  Q_OBJECT
  Q_DECLARE_PUBLIC(BasePopupWidget);
protected:
  BasePopupWidget* const q_ptr;
public:
  BasePopupWidgetPrivate(BasePopupWidget& object);
  ~BasePopupWidgetPrivate();
  virtual void init();

  bool isOpening()const;
  bool isClosing()const;
  /// Return true if the animation was closing (direction == backward).
  /// It doesn't indicate if the action is still running or finished.
  /// Can only be called in a slot as it uses sender().
  bool wasClosing()const;

  bool fitBaseWidgetSize()const;
  Qt::Alignment pixmapAlignment()const;
  void setupPopupPixmapWidget();

  QWidgetList focusWidgets(bool onlyVisible = false)const;

  // Return the widget if the mouse cursor is above any of the focus widgets or their
  // children.
  virtual QWidget* mouseOver();

  // Same as QWidget::isAncestorOf() but don't restrain to the same window
  // and apply it to all the focusWidgets
  bool isAncestorOf(const QWidget* ancestor, const QWidget* child)const;


  /// Return the closed geometry for the popup based on the current geometry
  QRect closedGeometry()const;
  /// Return the closed geometry for a given open geometry
  QRect closedGeometry(QRect openGeom)const;

  /// Return the desired geometry, maybe it won't happen if the size is too
  /// small for the popup.
  QRect desiredOpenGeometry()const;
  QRect desiredOpenGeometry(QRect baseGeometry)const;
  QRect baseGeometry()const;
  QPoint mapToGlobal(const QPoint& baseWidgetPoint)const;

  QPropertyAnimation* currentAnimation()const;

  //void temporarilyHiddenOn();
  //void temporarilyHiddenOff();

  void hideAll();

protected:
  QPointer<QWidget> BaseWidget;

  double EffectAlpha;

  BasePopupWidget::AnimationEffect Effect;
  int                 EffectDuration;
  QPropertyAnimation* AlphaAnimation;
  bool                ForcedTranslucent;
  QPropertyAnimation* ScrollAnimation;
  QLabel*             PopupPixmapWidget;

  // Geometry attributes
  Qt::Alignment    Alignment;
  Qt::Orientations Orientations;

  BasePopupWidget::VerticalDirection VerticalDirection;
  Qt::LayoutDirection HorizontalDirection;
};

#endif
