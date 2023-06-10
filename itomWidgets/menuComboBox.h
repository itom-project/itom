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

#ifndef MENUCOMBOBOX_H
#define MENUCOMBOBOX_H

// Qt includes
#include <QMenu>
#include <QMetaType>
#include <QWidget>
#include <qvariant.h>
#include <qlist.h>
class QComboBox;
class QToolButton;

// CTK includes
#include "commonWidgets.h"
class Completer;
class MenuComboBoxPrivate;

/// \ingroup Widgets
/// QComboBox linked with a QMenu. See MenuComboBox::setMenu()
/// MenuComboBox can be editable, disable,
/// editable on focus or editable on double click.
///   if it is editable :
/// the comboBox is always editable, you can filter the Menu or show it.
///   if it is editable on focus - on double click:
/// the combobox become editable when it has the focus in.
/// So MenuComboBox's purpose is to filter a menu, if you edit the current text
/// or show the menu, if you click on the arrow.
///   if it is disabled :
/// the MenuComboBox has the same behavior as a QPushButton. You can't filter the menu.

/// By default MenuComboBox is not editable with the search icon visible.
/// See MenuComboBox::setEditableType() to change the default behavior.
/// and setIconSearchVisible() to show/hide the icon.

class ITOMWIDGETS_EXPORT MenuComboBox : public QWidget
{
  Q_OBJECT

  /// This property holds the text shown on the combobox when there is no
  /// selected item.
  /// Empty by default.
  Q_PROPERTY(QString defaultText READ defaultText WRITE setDefaultText)
  /// This property holds the icon shown on the combobox when the current item
  /// (QAction) doesn't have any icon associated.
  /// Empty by default
  Q_PROPERTY(QIcon defaultIcon READ defaultIcon WRITE setDefaultIcon)
  /// This property holds the edit behavior of the combobox, it defines what
  /// action is needed to turn the combobox into a search mode where the user
  /// can type the name of the item to select using the combobox line edit.
  /// MenuComboBox::NotEditable by default
  /// \sa EditableType
  Q_PROPERTY(EditableBehavior editBehavior READ editableBehavior WRITE setEditableBehavior)
  /// This property controls whether the search tool button is visible or hidden.
  /// True by default
  Q_PROPERTY(bool searchIconVisible READ isSearchIconVisible WRITE setSearchIconVisible)
  /// This property holds whether the search tool button displays an icon only,
  /// text only, or text beside/below the icon.
  /// The default is Qt::ToolButtonIconOnly.
  /// \sa QToolButton::toolButtonStyle
  Q_PROPERTY(Qt::ToolButtonStyle toolButtonStyle READ toolButtonStyle WRITE setToolButtonStyle)
public:
  enum EditableBehavior{
    NotEditable = 0,
    Editable,
    EditableOnFocus,
    EditableOnPopup
  };

  //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
  //values are always accessible.
  Q_ENUM(EditableBehavior)

    /// Superclass typedef
    typedef QWidget Superclass;

  ///
  MenuComboBox(QWidget* parent = 0);
  virtual ~MenuComboBox();

  /// Set menu to both the QComboBox and the associated Completer.
  /// \sa setCompleterMenu(), searchCompleter()
  Q_INVOKABLE void setMenu(QMenu* menu);
  Q_INVOKABLE QMenu* menu()const;

  /// Set a specific menu to the Completer.
  ///
  /// This is useful when the menu displayed with the combobox is only a subset
  /// of the action that can be searched for.
  /// \sa setMenu(), searchCompleter()
  Q_INVOKABLE void setCompleterMenu(QMenu* menu);
  Q_INVOKABLE QMenu* completerMenu()const;

  void setDefaultText(const QString&);
  QString defaultText()const;

  void setDefaultIcon(const QIcon&);
  QIcon defaultIcon()const;

  void setEditableBehavior(EditableBehavior editBehavior);
  EditableBehavior editableBehavior()const;

  void setSearchIconVisible(bool state);
  bool isSearchIconVisible() const;

  Qt::ToolButtonStyle toolButtonStyle() const;

  /// Set the minimum width of the combobox.
  /// \sa QComboBox::setMinimumContentsLength()
  void setMinimumContentsLength(int characters);

  /// Return the internal combo box
  QComboBox* menuComboBoxInternal() const;

  /// Return the internal tool button
  QToolButton* toolButtonInternal() const;

  /// Return the internal completer
  Completer* searchCompleter() const;

protected:
    virtual bool eventFilter(QObject* target, QEvent* event);

public Q_SLOTS:
    void clearActiveAction();
    void setToolButtonStyle(Qt::ToolButtonStyle style);

    /// this slot only returns the argument list for unittest purposes.
    /* This slot has no further functionality and can be used
    to test the proper marshalling between Python, C++ and Qt.
    */
    QVariantList __unittestVariantList(const QVariantList &list);

    int __unittestInt(int value);
    qint64 __unittestInt64(qint64 value);
    quint64 __unittestUInt64(quint64 value);
    float __unittestFloat(float value);
    double __unittestDouble(double value);
    short __unittestShort(short value);

Q_SIGNALS:
    void actionChanged(QAction* action);
    void popupShown();

protected Q_SLOTS:
    /// Change the current text/icon on the QComboBox
    /// And trigger the action.
    /// action selected from the menu.
    void onActionSelected(QAction* action);
    /// action selected from the line edit or the completer.
    void onEditingFinished();

protected:
    QScopedPointer<MenuComboBoxPrivate> d_ptr;

private:
    Q_DECLARE_PRIVATE(MenuComboBox);
    Q_DISABLE_COPY(MenuComboBox);
};

Q_DECLARE_METATYPE(MenuComboBox::EditableBehavior)

#endif
