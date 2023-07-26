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

#ifndef COMBOBOX_H
#define COMBOBOX_H

// Qt includes
#include <QComboBox>

// CTK includes
//#include "ctkPimpl.h"
#include "commonWidgets.h"
class ComboBoxPrivate;

/// \ingroup Widgets
/// \brief ComboBox is an advanced QComboBox.
/// It adds multiple features:
///  * Display a default text and/or icon when the combobox current index is
///    invalid (-1). A typical default text would be "Select a XXX...".
///    forceDefault can force the display of the default text at all time (with
///    a valid current index). The text displayed in the combo box can be
///    elided when the size is too small.
///  * Optionally prevent the mouse scroll events from changing the current
///    index.
/// ComboBox works exactly the same as QComboBox by default.
/// \sa QComboBox
class ITOMWIDGETS_EXPORT ComboBox : public QComboBox
{
  Q_OBJECT
  Q_PROPERTY(QString defaultText READ defaultText WRITE setDefaultText)
  Q_PROPERTY(QIcon defaultIcon READ defaultIcon WRITE setDefaultIcon)
  Q_PROPERTY(bool forceDefault READ isDefaultForced WRITE forceDefault)
  Q_PROPERTY(Qt::TextElideMode elideMode READ elideMode WRITE setElideMode)
  /// This property controls the behavior of the mouse scroll wheel.
  /// ScrollOn by default.
  /// /sa scrollWheelEffect, setScrollWheelEffect
  Q_PROPERTY(ScrollEffect scrollWheelEffect READ scrollWheelEffect WRITE setScrollWheelEffect)
  /// Current item's user data as string (Qt::UserRole role)
  Q_PROPERTY(QString currentUserDataAsString READ currentUserDataAsString WRITE setCurrentUserDataAsString)

public:
  /// Constructor, build a ComboBox that behaves like QComboBox.
  explicit ComboBox(QWidget* parent = 0);
  virtual ~ComboBox();

  /// Empty by default (same behavior as QComboBox)
  void setDefaultText(const QString&);
  QString defaultText()const;

  /// Empty by default (same behavior as QComboBox)
  void setDefaultIcon(const QIcon&);
  QIcon defaultIcon()const;

  /// Force the display of the text/icon at all time (not only when the
  /// current index is invalid). False by default.
  void forceDefault(bool forceDefault);
  bool isDefaultForced()const;

  /// setElideMode can elide the text displayed on the combobox.
  /// Qt::ElideNone by default (same behavior as QComboBox)
  void setElideMode(const Qt::TextElideMode& newMode);
  Qt::TextElideMode elideMode()const;

  /// \tbd turn into flags ?
  enum ScrollEffect
  {
    /// Scrolling is not possible with the mouse wheel.
    NeverScroll,
    /// Scrolling is always possible with the mouse wheel.
    AlwaysScroll,
    /// Scrolling is only possible if the combobox has the focus.
    /// The focus policy is automatically set to Qt::StrongFocus
    ScrollWithFocus,
    /// Scrolling is not possible when the combobox is inside a scroll area with
    /// a visible vertical scrollbar.
    ScrollWithNoVScrollBar
  };

  //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
  //values are always accessible.
  Q_ENUM(ScrollEffect);

  /// Return the scrollWheelEffect property value.
  /// \sa scrollEffect
  ScrollEffect scrollWheelEffect()const;
  /// Set the scrollWheelEffect property value.
  /// \sa scrollEffect
  void setScrollWheelEffect(ScrollEffect scroll);

  /// Reimplemented for internal reasons
  virtual QSize minimumSizeHint()const;
  /// Reimplemented for internal reasons
  virtual QSize sizeHint()const;

  /// Get current item's user data as string
  QString currentUserDataAsString()const;

public slots:
  /// Set current item based on user data
  void setCurrentUserDataAsString(QString userData);

protected:
  /// Reimplemented for internal reasons
  virtual void paintEvent(QPaintEvent* event);
  virtual void changeEvent(QEvent* event);
  virtual void wheelEvent(QWheelEvent* event);

protected:
  QScopedPointer<ComboBoxPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(ComboBox);
  Q_DISABLE_COPY(ComboBox);
};

#endif
