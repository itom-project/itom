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

#ifndef HELPDOCKWIDGET_H
#define HELPDOCKWIDGET_H

#include "abstractDockWidget.h"

#include <qwidget.h>
#include <qaction.h>
#include <qwidgetaction.h>
#include <qtoolbar.h>

#include <qtreewidget.h>

class HelpTreeDockWidget;

namespace ito
{
    class HelpDockWidget : public AbstractDockWidget
    {
        Q_OBJECT

        public:
            HelpDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL, bool docked = true, bool isDockAvailable = true, tFloatingStyle floatingStyle = floatingNone, tMovingStyle movingStyle = movingEnabled);
            ~HelpDockWidget();

        protected:

            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){};
            void updateActions();
            void updatePythonActions(){ updateActions(); }

        private:
            QAction *m_pActBackward;
            QAction *m_pActForward;
            QAction *m_pActExpand;
            QAction *m_pActCollapse;
            QWidgetAction *m_pActChanged;
            QToolBar *m_pMainToolbar;
            QLineEdit *m_pFilterEdit;


            HelpTreeDockWidget *m_pHelpWidget;

        signals:
            void showPluginInfo(QString name, HelpTreeDockWidget::HelpItemType type, const QModelIndex modelIndex, bool fromLink);

        private slots:


        public slots:
            void mnuShowInfo(QString name, HelpTreeDockWidget::HelpItemType type);

    };

} //end namespace ito

#endif
