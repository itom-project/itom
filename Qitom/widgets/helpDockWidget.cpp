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

#include "../ui/helpTreeDockWidget.h"
#include "helpDockWidget.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../helper/guiHelper.h"

#include "../organizer/scriptEditorOrganizer.h"

#include <qheaderview.h>
#include <qsettings.h>
#include "../widgets/abstractDockWidget.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
HelpDockWidget::HelpDockWidget(const QString &title, const QString &objName, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_pHelpWidget(NULL),
    m_pActBackward(NULL),
    m_pActForward(NULL),
    m_pActChanged(NULL),
    m_pMainToolbar(NULL),
    m_pFilterEdit(NULL),
    m_pActExpand(NULL),
    m_pActCollapse(NULL)
{
    m_pHelpWidget = new HelpTreeDockWidget(this, this);

    m_pFilterEdit = new QLineEdit(this);
    m_pFilterEdit->sizePolicy().setHorizontalPolicy(QSizePolicy::MinimumExpanding);

    AbstractDockWidget::init();

    float dpiFactor = GuiHelper::screenDpiFactor();

    setMinimumSize(QSize(dpiFactor * 300, dpiFactor * 200));

    setContentWidget(m_pHelpWidget);
}

//----------------------------------------------------------------------------------------------------------------------------------
HelpDockWidget::~HelpDockWidget()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::createActions()
{
    m_pActBackward = new QAction(QIcon(":/editor/icons/editUndo.png"), tr("backwards"), this);
    connect(m_pActBackward, SIGNAL(triggered()), m_pHelpWidget, SLOT(navigateBackwards()));

    m_pActForward = new QAction(QIcon(":/editor/icons/editRedo.png"), tr("forwards"), this);
    connect(m_pActForward, SIGNAL(triggered()), m_pHelpWidget, SLOT(navigateForwards()));

    m_pActExpand = new QAction(QIcon(":/editor/icons/editSmartIndent.png"), tr("expand tree"), this);
    connect(m_pActExpand, SIGNAL(triggered()), m_pHelpWidget, SLOT(expandTree()));

    m_pActCollapse = new QAction(QIcon(":/editor/icons/editUnindent.png"), tr("collapse tree"), this);
    connect(m_pActCollapse, SIGNAL(triggered()), m_pHelpWidget, SLOT(collapseTree()));

    connect(this, &HelpDockWidget::showPluginInfo, m_pHelpWidget, &HelpTreeDockWidget::showPluginInfo);
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::createMenus()
{
    QMenu *winMenu = getMenuBar()->addMenu(tr("&Windows"));
    if (m_actStayOnTop)
    {
        winMenu->addAction(m_actStayOnTop);
    }
    if (m_actStayOnTopOfApp)
    {
        winMenu->addAction(m_actStayOnTopOfApp);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::createToolBars()
{
    m_pMainToolbar = new QToolBar(tr("navigation"), this);
    m_pMainToolbar->setObjectName("toolbarHelp");
    m_pMainToolbar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolbar->setFloatable(false);
    m_pMainToolbar->addAction(m_pActBackward);
    m_pMainToolbar->addAction(m_pActForward);
    m_pMainToolbar->addWidget(m_pFilterEdit);
    m_pFilterEdit->setToolTip(tr("type text to filter the keywords in the tree"));
    connect(m_pFilterEdit, SIGNAL(textChanged(QString)), m_pHelpWidget, SLOT(liveFilter(QString)));
    m_pMainToolbar->addAction(m_pActExpand);
    m_pMainToolbar->addAction(m_pActCollapse);

    addToolBar(m_pMainToolbar, "navigationToolbar");
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::updateActions()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpDockWidget::mnuShowInfo(QString name, HelpTreeDockWidget::HelpItemType type)
{
    emit showPluginInfo(name, type, QModelIndex(), true);
}


} //end namespace ito
