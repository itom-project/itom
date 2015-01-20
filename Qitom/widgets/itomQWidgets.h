/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#include <qtabwidget.h>
#include <qtreeview.h>
#include <qtabbar.h>
#include <qevent.h>

namespace ito
{

/*!
    This class inherits QTabWidget and only has the additional inline function to get the member tabBar of QTabWidget.
    In QTabWidget this member is protected.
*/
class QTabWidgetItom : public QTabWidget
{
    Q_OBJECT
public:
    QTabWidgetItom(QWidget * parent = 0) : QTabWidget(parent) {};
 
    inline QTabBar* getTabBar() {return tabBar(); };

protected:
    void contextMenuEvent (QContextMenuEvent * event)
    {
        emit tabContextMenuEvent(event);
        event->accept();
    };

signals:
    void tabContextMenuEvent (QContextMenuEvent *event);
};


class QTreeViewItom : public QTreeView
{
    Q_OBJECT

public:
    QTreeViewItom ( QWidget * parent = 0 ) : QTreeView(parent) {}
    ~QTreeViewItom () {}

    QModelIndexList selectedIndexes() const
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

protected slots:
    virtual void selectionChanged ( const QItemSelection & selected, const QItemSelection & deselected )
    {
        QTreeView::selectionChanged(selected, deselected);
        emit selectedItemsChanged();
    }

signals:
    void selectedItemsChanged();
};

} //end namespace ito

#endif