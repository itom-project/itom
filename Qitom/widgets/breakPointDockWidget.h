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

#ifndef BREAKPOINTDOCKWIDGET_H
#define BREAKPOINTDOCKWIDGET_H

#include "abstractDockWidget.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qitemselectionmodel.h>

#include "itomQWidgets.h"


namespace ito
{
    class BreakPointDockWidget : public AbstractDockWidget
    {
        Q_OBJECT

        public:
            BreakPointDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL, bool docked = true, bool isDockAvailable = true, tFloatingStyle floatingStyle = floatingNone, tMovingStyle movingStyle = movingEnabled);
            ~BreakPointDockWidget();

        protected:

            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){}
            void updateActions();
            void updatePythonActions(){ updateActions(); }

        private:
            QTreeViewItom   *m_breakPointView;      /*!< QTreeViewItom derived from QTreeView with some special selection behaviour (see QItomWidgets)*/

            QToolBar    *m_pMainToolbar;            /*!< Toolbar with QActions */

            QMenu *m_pContextMenu;                  /*!< Contextmenu with the same actions as the toolbar */

            bool m_enOrDisAbleAllBreakpoints;        /*!< This is a flag used to signalize the enableBP method that it has to select all entries or just the ones the user selected  */

            ShortcutAction* m_pActDelBP;
            ShortcutAction* m_pActDelAllBPs;
            ShortcutAction* m_pActEditBP;
            ShortcutAction* m_pActToggleBP;
            ShortcutAction* m_pActToggleAllBPs;


        signals:

        private slots:
            void doubleClicked(const QModelIndex &index);
            void mnuDeleteBP();
            void mnuDeleteAllBPs();
            void mnuEditBreakpoint();
            void mnuEnOrDisAbleBrakpoint();
            void mnuEnOrDisAbleAllBrakpoints();
            void treeViewContextMenuRequested(const QPoint &pos);
            void treeViewSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
            void actualizeTree(const QModelIndex &parent, int start, int end);

			void dataChanged();

    };

} //end namespace ito

#endif
