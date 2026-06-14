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

#ifndef BASEPOPUPWIDGET_H
#define BASEPOPUPWIDGET_H

// Qt includes
#include <QEasingCurve>
#include <QFrame>
#include <QMetaType>

#include "commonWidgets.h"

class BasePopupWidgetPrivate;

/// \ingroup Widgets
/// Description:
class ITOMWIDGETS_EXPORT BasePopupWidget : public QFrame
{
  Q_OBJECT

  /// ScrollEffect by default
  Q_PROPERTY( AnimationEffect animationEffect READ animationEffect WRITE setAnimationEffect)

  /// Effect duration in ms
  /// Default to 333ms
  Q_PROPERTY( int effectDuration READ effectDuration WRITE setEffectDuration);

  /// Opening/Closing curve
  /// QEasingCurve::InOutQuad by default
  Q_PROPERTY( QEasingCurve::Type easingCurve READ easingCurve WRITE setEasingCurve);

  /// Where is the popup in relation to the BaseWidget
  /// To vertically justify, use Qt::AlignTop | Qt::AlignBottom
  /// Qt::AlignJustify | Qt::AlignBottom by default
  Q_PROPERTY( Qt::Alignment alignment READ alignment WRITE setAlignment);

  /// Direction of the scrolling effect, can be Qt::Vertical, Qt::Horizontal or
  /// both Qt::Vertical|Qt::Horizontal.
  /// Vertical by default
  Q_PROPERTY( Qt::Orientations orientation READ orientation WRITE setOrientation);

  /// Control where the popup opens vertically.
  /// TopToBottom by default
  Q_PROPERTY( BasePopupWidget::VerticalDirection verticalDirection READ verticalDirection WRITE setVerticalDirection);

  /// Control where the popup opens horizontally.
  /// LeftToRight by default
  Q_PROPERTY( Qt::LayoutDirection horizontalDirection READ horizontalDirection WRITE setHorizontalDirection);

public:
  typedef QFrame Superclass;
  /// Although a popup widget is a top-level widget, if a parent is
  /// passed the popup widget will be deleted when that parent is
  /// destroyed (as with any other QObject).
  /// BasePopupWidget is a top-level widget (Qt::ToolTip), so
  /// even if a parent is passed, the popup will display outside the possible
  /// parent layout.
  /// \sa baseWidget().
  explicit BasePopupWidget(QWidget* parent = 0);
  virtual ~BasePopupWidget();

  /// Widget the popup is attached to. It opens right under \a baseWidget
  /// and if the BasePopupWidget sizepolicy contains the growFlag/shrinkFlag,
  /// it tries to resize itself to fit the same width of \a baseWidget.
  /// By default, baseWidget is the parent widget.
  QWidget* baseWidget()const;

  enum AnimationEffect
  {
    WindowOpacityFadeEffect = 0,
    ScrollEffect,
    FadeEffect
  };

  AnimationEffect animationEffect()const;
  void setAnimationEffect(AnimationEffect effect);

  int effectDuration()const;
  void setEffectDuration(int duration);

  QEasingCurve::Type easingCurve()const;
  void setEasingCurve(QEasingCurve::Type easingCurve);

  Qt::Alignment alignment()const;
  void setAlignment(Qt::Alignment alignment);

  Qt::Orientations orientation()const;
  void setOrientation(Qt::Orientations orientation);

  enum VerticalDirection{
    TopToBottom = 1,
    BottomToTop = 2
  };

  //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
  //values are always accessible.
  Q_ENUM(AnimationEffect);
  Q_ENUM(VerticalDirection);

  VerticalDirection verticalDirection()const;
  void setVerticalDirection(VerticalDirection direction);

  Qt::LayoutDirection horizontalDirection()const;
  void setHorizontalDirection(Qt::LayoutDirection direction);

public Q_SLOTS:
  /// Hide the popup if open or opening. It takes around 300ms
  /// for the fading effect to hide the popup.
  virtual void hidePopup();
  /// Open the popup if closed or closing. It takes around 300ms
  /// for the fading effect to open the popup.
  virtual void showPopup();
  /// Show/hide the popup. It can be conveniently linked to a QPushButton
  /// signal.
  inline void showPopup(bool show);

Q_SIGNALS:
  void popupOpened(bool open);

protected:
  explicit BasePopupWidget(BasePopupWidgetPrivate* pimpl, QWidget* parent = 0);
  QScopedPointer<BasePopupWidgetPrivate> d_ptr;
  Q_PROPERTY(double effectAlpha READ effectAlpha WRITE setEffectAlpha DESIGNABLE false)
  Q_PROPERTY(QRect effectGeometry READ effectGeometry WRITE setEffectGeometry DESIGNABLE false)

  double effectAlpha()const;
  QRect effectGeometry()const;

  virtual void setBaseWidget(QWidget* baseWidget);

  virtual bool event(QEvent* event);
  virtual void paintEvent(QPaintEvent*);

protected Q_SLOTS:
  virtual void onEffectFinished();
  void setEffectAlpha(double alpha);
  void setEffectGeometry(QRect geometry);
  void onBaseWidgetDestroyed();

private:
  Q_DECLARE_PRIVATE(BasePopupWidget);
  Q_DISABLE_COPY(BasePopupWidget);
};

Q_DECLARE_METATYPE(BasePopupWidget::AnimationEffect)
Q_DECLARE_METATYPE(BasePopupWidget::VerticalDirection)

// -------------------------------------------------------------------------
void BasePopupWidget::showPopup(bool show)
{
  if (show)
    {
    this->showPopup();
    }
  else
    {
    this->hidePopup();
    }
}

#endif
