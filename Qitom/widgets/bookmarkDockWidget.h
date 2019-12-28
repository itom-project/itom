/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2019, Institut fuer Technische Optik (ITO),
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

#ifndef BOOKMARKDOCKWIDGET_H
#define BOOKMARKDOCKWIDGET_H

#include "abstractDockWidget.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qitemselectionmodel.h>
#include "../models/bookmarkModel.h"

#include "itomQWidgets.h"


namespace ito
{
    class BookmarkDockWidget : public AbstractDockWidget
    {
        Q_OBJECT

        public:
            BookmarkDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL, 
                bool docked = true, bool isDockAvailable = true, 
                tFloatingStyle floatingStyle = floatingNone, 
                tMovingStyle movingStyle = movingEnabled);
            ~BookmarkDockWidget();

            void setBookmarkModel(BookmarkModel *model);

        protected:

            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){}
            void updateActions();
            void updatePythonActions(){ updateActions(); }

        private:
            QTreeViewItom   *m_bookmarkView;        /*!< QTreeViewItom derived from QTreeView with some special selection behaviour (see QItomWidgets)*/
            QToolBar    *m_pMainToolbar;            /*!< Toolbar with QActions */
            QMenu *m_pContextMenu;                  /*!< Context menu with the same actions as the toolbar */
            BookmarkModel *m_pModel;
            QAction *m_pSpacerAction;

        signals:

        private slots:
            void doubleClicked(const QModelIndex &index);
            void treeViewContextMenuRequested(const QPoint &pos);
            void treeViewSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);

            void dataChanged();
    };

} //end namespace ito

#endif
