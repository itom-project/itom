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

#pragma once

#include "../organizer/uiOrganizer.h"
#include "../../common/addInInterface.h"
#include "abstractDockWidget.h"
#include "../ui/helpTreeDockWidget.h"

#include <qdockwidget.h>
#include <qtreeview.h>
#include <qmenu.h>
#include <qaction.h>
#include <qsortfilterproxymodel.h>

namespace ito
{
    class PlugInModel; //forward declaration

    class AIManagerWidget : public AbstractDockWidget
    {
        Q_OBJECT

        Q_PROPERTY(QColor backgroundColorInstancesWithPythonRef READ backgroundColorInstancesWithPythonRef WRITE setBackgroundColorInstancesWithPythonRef)

        public:
//            AIManagerWidget();
            AIManagerWidget(const QString &title, const QString &objName, QWidget *parent = NULL, bool docked = true, bool isDockAvailable = true, tFloatingStyle floatingStyle = floatingNone, tMovingStyle movingStyle = movingEnabled);
            ~AIManagerWidget();

            //!< returns the background color for instances, that have been created by Python or which have at least one current reference by Python code
            QColor backgroundColorInstancesWithPythonRef() const;
            void setBackgroundColorInstancesWithPythonRef(const QColor &bgColor);

        protected:
//            QList<QAction*> getAlgoWidgetActions(const ito::AddInInterfaceBase *aib);
    //        void closeEvent(QCloseEvent *event) {};
            QMenu* m_pContextMenu;
            QMenu* m_pAIManagerViewSettingMenu;

            QToolBar* m_pMainToolbar;

            QAction *m_pShowConfDialog;
            QAction *m_pActDockWidget;
            QAction *m_pActDockWidgetToolbar;
            QAction *m_pActNewInstance;
            QAction *m_pActCloseInstance;
            QAction *m_pActCloseAllInstances;
            QAction *m_pActSendToPython;
            QAction *m_pActLiveImage;
            QAction *m_pActSnapDialog;
            QAction *m_pActAutoGrabbing;
            QAction *m_pActInfo;
            QAction *m_pActOpenWidget;
            QAction *m_pMainToolbarSeparator1;
            QAction *m_pMainToolbarSeparator2;

            ShortcutAction* m_pViewList;
            ShortcutAction* m_pViewDetails;

            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){}
            void updateActions();
            void updatePythonActions(){ updateActions(); }
            void closeInstance(const QModelIndex index);

        private:
            QTreeView *m_pAIManagerView;
            QSortFilterProxyModel *m_pSortFilterProxyModel;
            int *m_pColumnWidth;
            PlugInModel *m_pPlugInModel;

        public slots:

        private slots:
            void treeViewContextMenuRequested(const QPoint &pos);
            void selectionChanged(const QItemSelection& newSelection, const QItemSelection& oldSelection);
            void mnuShowConfdialog();
            void mnuToggleDockWidget();
            void mnuCreateNewInstance();
            void mnuCloseInstance();
            void mnuCloseAllInstances();
            void mnuSendToPython();
            void mnuShowAlgoWidget(ito::AddInAlgo::AlgoWidgetDef* awd);
            void mnuOpenWidget();
            void mnuToggleView();
            void mnuShowLiveImage();
            void mnuSnapDialog();
            void mnuToggleAutoGrabbing();
            void setTreeViewHideColumns(const bool &hide, const int colCount);
            void showList();
            void showDetails();
            void mnuShowInfo();

        signals:
            void showPluginInfo(QString name, HelpTreeDockWidget::HelpItemType type);
            void showDockWidget();
    };

};  // namespace ito
