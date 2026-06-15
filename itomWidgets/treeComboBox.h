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

#ifndef TREECOMBOBOX_H
#define TREECOMBOBOX_H

// Qt includes
#include <QComboBox>

// CTK includes
//#include <ctkPimpl.h>

#include "commonWidgets.h"

class TreeComboBoxPrivate;
class QTreeView;

/// \ingroup Widgets
/// Description:
/// ComboBox that displays the items as a tree view.
/// See below for a use case:
///    TreeComboBox combo;
///    QStandardItemModel model;
///    model.appendRow(new QStandardItem("Test1"));
///    model.item(0)->appendRow(new QStandardItem("Test1.1"));
///    model.item(0)->appendRow(new QStandardItem("Test1.2"));
///    model.item(0)->appendRow(new QStandardItem("Test1.3"));
///    model.appendRow(new QStandardItem("Test2"));
///    model.appendRow(new QStandardItem("Test3"));
///    combo.setModel(&model);
///    combo.show();
/// TODO fix size of the view
class ITOMWIDGETS_EXPORT TreeComboBox : public QComboBox
{
  Q_OBJECT
  /// Column index visible in the view. If \sa visibleModelColumn is -1
  /// (default) then all columns are visible.
  Q_PROPERTY(int visibleModelColumn READ visibleModelColumn WRITE setVisibleModelColumn)
public:
  typedef QComboBox Superclass;
  explicit TreeComboBox(QWidget* parent = 0);
  virtual ~TreeComboBox();

  int visibleModelColumn()const;
  void setVisibleModelColumn(int index);

  virtual bool eventFilter(QObject* object, QEvent* event);
  virtual void showPopup();
  virtual void hidePopup();

  /// TreeComboBox uses a QTreeView for its model view. treeView() is a
  /// utility function that cast QComboBox::view() into a QTreeView.
  /// \sa view()
  QTreeView* treeView()const;

protected:
  virtual void paintEvent(QPaintEvent*);

protected Q_SLOTS:
  void resizePopup();

signals:
  void popupShow();
  void popupHide();

protected:
  QScopedPointer<TreeComboBoxPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(TreeComboBox);
  Q_DISABLE_COPY(TreeComboBox);
};

#endif
