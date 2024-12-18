/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut für Technische Optik (ITO),
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

#include "itomQWidgets.h"
#include "fileSystemDockWidget.h"
#include "mainWindow.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
QTreeWidgetItom::~QTreeWidgetItom()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QTreeWidgetItem* QTreeWidgetItom::itemFromIndex2(const QModelIndex& index) const
{
    return itemFromIndex(index);
}

//----------------------------------------------------------------------------------------------------------------------------------
QTreeViewItom::~QTreeViewItom()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QModelIndexList QTreeViewItom::selectedIndexes() const
{
    QModelIndexList retList;
    for (int i = 0; i < QTreeView::selectedIndexes().length(); ++i)
    {
        if (QTreeView::selectedIndexes().at(i).column() == 0)
        {
            retList.append(QTreeView::selectedIndexes().at(i));
        }
    }
    return retList;
}

//----------------------------------------------------------------------------------------------------------------------------------
void QTreeViewItom::selectionChanged(
    const QItemSelection& selected, const QItemSelection& deselected)
{
    QTreeView::selectionChanged(selected, deselected);
    emit selectedItemsChanged(selected, deselected);
}

//----------------------------------------------------------------------------------------------------------------------------------
void QTreeViewItom::mouseReleaseEvent(QMouseEvent* event)
{
    // Call the base class implementation
    QTreeView::mouseReleaseEvent(event);

    // Your custom code here
    emit QTreeViewItomMouseReleased(event);
}

//----------------------------------------------------------------------------------------------------------------------------------
QListViewItom::~QListViewItom()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QModelIndexList QListViewItom::selectedIndexes() const
{
    QModelIndexList retList;
    for (int i = 0; i < QListView::selectedIndexes().length(); ++i)
    {
        if (QListView::selectedIndexes().at(i).column() == 0)
        {
            retList.append(QListView::selectedIndexes().at(i));
        }
    }
    return retList;
}

//----------------------------------------------------------------------------------------------------------------------------------
void QListViewItom::selectionChanged(
    const QItemSelection& selected, const QItemSelection& deselected)
{
    QListView::selectionChanged(selected, deselected);
    emit selectedItemsChanged(selected, deselected);
}

//----------------------------------------------------------------------------------------------------------------------------------
QTableViewItom::~QTableViewItom()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QModelIndexList QTableViewItom::selectedIndexes() const
{
    QModelIndexList retList;
    for (int i = 0; i < QTableView::selectedIndexes().length(); ++i)
    {
        if (QTableView::selectedIndexes().at(i).column() == 0)
        {
            retList.append(QTableView::selectedIndexes().at(i));
        }
    }
    return retList;
}

//----------------------------------------------------------------------------------------------------------------------------------
void QTableViewItom::selectionChanged(
    const QItemSelection& selected, const QItemSelection& deselected)
{
    QTableView::selectionChanged(selected, deselected);
    emit selectedItemsChanged(selected, deselected);
}
} // namespace ito
