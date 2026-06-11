/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */
#ifndef ITOMQTWIDGETS_H
#define ITOMQTWIDGETS_H

#include <qevent.h>
#include <qlistview.h>
#include <qtabbar.h>
#include <qtableview.h>
#include <qtabwidget.h>
#include <qtreeview.h>
#include <qtreewidget.h>

namespace ito {
class FileSystemDockWidget; // forward declaration

//----------------------------------------------------------------------------------------------------------------------------------
class QTabWidgetItom : public QTabWidget
{
    Q_OBJECT

public:
    QTabWidgetItom(QWidget* parent = 0) : QTabWidget(parent)
    {
    }

    inline QTabBar* getTabBar()
    {
        return tabBar();
    };

protected:
    void contextMenuEvent(QContextMenuEvent* event)
    {
        emit tabContextMenuEvent(event);
        event->accept();
    };

signals:
    void tabContextMenuEvent(QContextMenuEvent* event);
};

//----------------------------------------------------------------------------------------------------------------------------------
class QTreeWidgetItom : public QTreeWidget
{
    Q_OBJECT

public:
    QTreeWidgetItom(QWidget* parent = 0) : QTreeWidget(parent)
    {
    }
    ~QTreeWidgetItom();

    QTreeWidgetItem* itemFromIndex2(const QModelIndex& index) const;
};

//----------------------------------------------------------------------------------------------------------------------------------
class QTreeViewItom : public QTreeView
{
    Q_OBJECT

public:
    QTreeViewItom(QWidget* parent = 0) : QTreeView(parent)
    {
    }
    ~QTreeViewItom();

    QModelIndexList selectedIndexes() const;

protected:
    virtual void selectionChanged(const QItemSelection& selected, const QItemSelection& deselected);
    virtual void mouseReleaseEvent(QMouseEvent* event);

signals:
    void selectedItemsChanged(const QItemSelection& selected, const QItemSelection& deselected);

Q_SIGNALS:
    void QTreeViewItomMouseReleased(QMouseEvent* event);
};

//----------------------------------------------------------------------------------------------------------------------------------
class QListViewItom : public QListView
{
    Q_OBJECT

public:
    QListViewItom(QWidget* parent = 0) : QListView(parent)
    {
    }
    ~QListViewItom();

    QModelIndexList selectedIndexes() const;

protected:
    virtual void selectionChanged(const QItemSelection& selected, const QItemSelection& deselected);

signals:
    void selectedItemsChanged(const QItemSelection& selected, const QItemSelection& deselected);
};

//----------------------------------------------------------------------------------------------------------------------------------
class QTableViewItom : public QTableView
{
    Q_OBJECT

public:
    QTableViewItom(QWidget* parent = 0) : QTableView(parent)
    {
    }
    ~QTableViewItom();

    QModelIndexList selectedIndexes() const;

protected:
    virtual void selectionChanged(const QItemSelection& selected, const QItemSelection& deselected);

signals:
    void selectedItemsChanged(const QItemSelection& selected, const QItemSelection& deselected);
};

} // end namespace ito

#endif
