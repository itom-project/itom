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

#include "bookmarkDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include "../organizer/scriptEditorOrganizer.h"
#include "../helper/guiHelper.h"

#include <qheaderview.h>
#include <qsettings.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
BookmarkDockWidget::BookmarkDockWidget(const QString &title, const QString &objName,
        QWidget *parent, bool docked, bool isDockAvailable,
        tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_pModel(NULL),
    m_pMainToolbar(NULL),
    m_pSpacerAction(NULL),
    m_pContextMenu(NULL),
    m_bookmarkView(nullptr)
{
    m_bookmarkView = new QTreeViewItom(this);

    AbstractDockWidget::init();

    m_bookmarkView->setContextMenuPolicy(Qt::CustomContextMenu);

    connect(m_bookmarkView, SIGNAL(doubleClicked(const QModelIndex &)), this, SLOT(doubleClicked(const QModelIndex &)));
    connect(m_bookmarkView, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(treeViewContextMenuRequested(const QPoint &)));

    m_bookmarkView->setTextElideMode(Qt::ElideLeft);
    m_bookmarkView->setSortingEnabled(false);
    m_bookmarkView->setItemsExpandable(false);
    m_bookmarkView->setRootIsDecorated(false);
    m_bookmarkView->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_bookmarkView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_bookmarkView->setHeaderHidden(true);

    setContentWidget(m_bookmarkView);

    updateActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::setBookmarkModel(BookmarkModel *model)
{
    if (m_pModel != NULL || model == NULL)
    {
        return; //can only assign a model once
    }

    m_pModel = model;
    m_bookmarkView->setModel(m_pModel);

    QVariant width; //user defined role: UserRole + SizeHintRole to get width only
    int width_;
    bool ok;
    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)

    for (int i = 0; i < m_pModel->columnCount(); ++i)
    {
        width = m_pModel->headerData(i, Qt::Horizontal, Qt::UserRole + Qt::SizeHintRole);
        width_ = width.toInt(&ok);
        if (ok)
        {
            m_bookmarkView->setColumnWidth(i, dpiFactor * width_);
        }
    }

    if (m_pMainToolbar)
    {
        m_pMainToolbar->insertAction(m_pSpacerAction, m_pModel->bookmarkPreviousAction());
        m_pMainToolbar->insertAction(m_pSpacerAction, m_pModel->bookmarkNextAction());
        m_pMainToolbar->insertAction(m_pSpacerAction, m_pModel->bookmarkClearAllAction());
    }

    if (m_pContextMenu)
    {
        m_pContextMenu->addAction(m_pModel->bookmarkPreviousAction());
        m_pContextMenu->addAction(m_pModel->bookmarkNextAction());
        m_pContextMenu->addAction(m_pModel->bookmarkClearAllAction());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
BookmarkDockWidget::~BookmarkDockWidget()
{
    //m_pModel is already destroyed if this destructor is called
}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::createToolBars()
{
    QWidget *spacerWidget = new QWidget();
    QHBoxLayout *spacerLayout = new QHBoxLayout();
    spacerLayout->addItem(new QSpacerItem(5, 5, QSizePolicy::Expanding, QSizePolicy::Minimum));
    spacerLayout->setStretch(0, 2);
    spacerWidget->setLayout(spacerLayout);

    m_pMainToolbar = new QToolBar(tr("Bookmarks"), this);
    m_pMainToolbar->setObjectName("toolbarBreakpoints");
    m_pMainToolbar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolbar->setFloatable(false);
    addToolBar(m_pMainToolbar, "mainToolBar");

    m_pSpacerAction = m_pMainToolbar->addWidget(spacerWidget);
}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::createActions()
{

}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::treeViewContextMenuRequested(const QPoint &pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_bookmarkView->mapToGlobal(m_bookmarkView->pos()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::updateActions()
{

}

//----------------------------------------------------------------------------------------------------------------------------------
void BookmarkDockWidget::doubleClicked(const QModelIndex &index)
{
    QAbstractItemModel *m = m_bookmarkView->model();

    if (index.isValid() && m)
    {
        QModelIndex idx = m->index(index.row(), 0, index.parent());
        m_pModel->gotoBookmark(idx);
    }
}

} //end namespace ito
