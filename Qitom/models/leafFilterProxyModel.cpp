/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#include "leafFilterProxyModel.h"
#include <QtDebug>

namespace ito
{

/*!
    \class LeafFilterProxyModel
    \brief  This class provides a special search function to the help model. The normal filtermodel hides nodes if they don't contain
            the searched term, even though the children contain it.

            Example
            Searchtext = "Leaf1"
            NodeA
            +- Leaf1
            +- NodeB

            The standard model would hide the entire NodeA because NodeA != Leaf1

            This Model here checks the children first and still shows the unmatching parent if a child matches the searched term.

*/

//----------------------------------------------------------------------------------------------------------------------------------
LeafFilterProxyModel::LeafFilterProxyModel(QObject *parent) :
    QSortFilterProxyModel(parent)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
bool LeafFilterProxyModel::filterAcceptsRow(int source_row, const QModelIndex &source_parent) const
{
    if (filterAcceptsRowItself(source_row, source_parent))
        return true;

    //accept if any of the parents is accepted on it's own merits
    QModelIndex parent = source_parent;
    while (parent.isValid()) {
        if (filterAcceptsRowItself(parent.row(), parent.parent()))
            return true;
        parent = parent.parent();
    }

    //accept if any of the children is accepted on it's own merits
    if (hasAcceptedChildren(source_row, source_parent)) {
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool LeafFilterProxyModel::filterAcceptsRowItself(int source_row, const QModelIndex &source_parent) const
{
    return QSortFilterProxyModel::filterAcceptsRow(source_row, source_parent);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool LeafFilterProxyModel::hasAcceptedChildren(int source_row, const QModelIndex &source_parent) const
{
    QModelIndex item = sourceModel()->index(source_row,0,source_parent);
    if (!item.isValid()) {
        //qDebug() << "item invalid" << source_parent << source_row;
        return false;
    }

    //check if there are children
    int childCount = item.model()->rowCount(item);
    if (childCount == 0)
        return false;

    for (int i = 0; i < childCount; ++i) {
        if (filterAcceptsRowItself(i, item))
            return true;
        //recursive call -> NOTICE that this is depth-first searching, you're probably better off with breadth first search...
        if (hasAcceptedChildren(i, item))
            return true;
    }

    return false;
}

} //end namespace ito
