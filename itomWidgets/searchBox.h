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

#ifndef SEARCHBOX_H
#define SEARCHBOX_H

// QT includes
#include <QIcon>
#include <QLineEdit>

// CTK includes
#include "commonWidgets.h"

class SearchBoxPrivate;

/// \ingroup Widgets
/// QLineEdit with two QIcons on each side: search and clear.
/// "Search" selects all the text
/// "Clear" clears the current text
/// See QLineEdit::text to set/get the current text.
/// SearchBox's purpose is to be used to filter other widgets.
/// e.g.:
/// <code>
///  SearchBox searchBox;
///  QSortFilterProxyModel filterModel;
///  QObject::connect(&searchBox, SIGNAL(textChanged(QString)),
///                   &filterModel, SLOT(setFilterFixedString(QString)));
///  ...
/// </code>
class ITOMWIDGETS_EXPORT SearchBox : public QLineEdit
{
  Q_OBJECT
#if QT_VERSION < 0x040700
  /// Qt < 4.7 don't have a placeholderText property, as we need it, we define it
  /// manually.
  Q_PROPERTY(QString placeholderText READ placeholderText WRITE setPlaceholderText)
#endif
  /// Show an icon at left side of the line edit, indicating that the text
  /// field is used to search/filter something. The default is <code>false</code>.
  Q_PROPERTY(bool showSearchIcon READ showSearchIcon WRITE setShowSearchIcon)

  /// The QIcon to use for the search icon at the left. The default is a
  /// magnifying glass icon.
  Q_PROPERTY(QIcon searchIcon READ searchIcon WRITE setSearchIcon)
  /// The QIcon to use for the clear icon. The default is a round grey button
  /// with a white cross.
  Q_PROPERTY(QIcon clearIcon READ clearIcon WRITE setClearIcon)

public:
  /// Superclass typedef
  typedef QLineEdit Superclass;

  SearchBox(QWidget *parent = 0);
  virtual ~SearchBox();

#if QT_VERSION < 0x040700
  QString placeholderText()const;
  void setPlaceholderText(const QString& defaultText);
#endif
  /// False by default
  void setShowSearchIcon(bool show);
  bool showSearchIcon()const;

  /// False by default
  void setAlwaysShowClearIcon(bool show);
  bool alwaysShowClearIcon()const;

  /// Set the search icon.
  void setSearchIcon(const QIcon& icon);
  /// Get the current search icon.
  QIcon searchIcon()const;

  /// Set the clear icon.
  void setClearIcon(const QIcon& icon);
  /// Get the current clear icon.
  QIcon clearIcon()const;

protected Q_SLOTS:
  /// Change the clear icon's state to enabled or disabled.
  void updateClearButtonState();

protected:
  virtual void paintEvent(QPaintEvent*);
  virtual void mousePressEvent(QMouseEvent* event);
  virtual void mouseMoveEvent(QMouseEvent *event);
  virtual void resizeEvent(QResizeEvent * event);

  QScopedPointer<SearchBoxPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(SearchBox);
  Q_DISABLE_COPY(SearchBox);
};
#endif // __SearchBox_h
