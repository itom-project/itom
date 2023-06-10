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

#include "../python/pythonEngineInc.h"

#include "breakPointDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"
#include "ui/dialogEditBreakpoint.h"

#include "../organizer/scriptEditorOrganizer.h"
#include "../helper/guiHelper.h"

#include <qheaderview.h>
#include <qsettings.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
BreakPointDockWidget::BreakPointDockWidget(const QString &title, const QString &objName, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent)
{
    m_breakPointView = new QTreeViewItom(this);

    m_enOrDisAbleAllBreakpoints = false;

    AbstractDockWidget::init();

    m_breakPointView->setContextMenuPolicy(Qt::CustomContextMenu);

    connect(m_breakPointView, SIGNAL(doubleClicked(const QModelIndex &)), this, SLOT(doubleClicked(const QModelIndex &)));
    connect(m_breakPointView, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(treeViewContextMenuRequested(const QPoint &)));
    connect(m_breakPointView, SIGNAL(selectedItemsChanged(QItemSelection,QItemSelection)), this, SLOT(treeViewSelectionChanged(QItemSelection,QItemSelection)));

    m_breakPointView->setTextElideMode(Qt::ElideLeft);
    m_breakPointView->sortByColumn(0, Qt::AscendingOrder);
    m_breakPointView->setExpandsOnDoubleClick(false);       // to avoid collapse of item while trying to open it
    m_breakPointView->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_breakPointView->setSelectionMode(QAbstractItemView::ExtendedSelection);

    PythonEngine *pe = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pe)
    {
        m_breakPointView->setModel(pe->getBreakPointModel());
        connect(pe->getBreakPointModel(), SIGNAL(rowsInserted(const QModelIndex &, int, int)), this, SLOT(actualizeTree(const QModelIndex &, int, int)));
        // maybe it would be good to connect the rowsRemoved-Signal as well. Just to be shure!
    }

    QAbstractItemModel *model = m_breakPointView->model();
    if (model)
    {
        QVariant width; //user defined role: UserRole + SizeHintRole to get width only
        int width_;
        bool ok;
        float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)

        for (int i = 0; i < model->columnCount(); ++i)
        {
            width = model->headerData(i, Qt::Horizontal, Qt::UserRole + Qt::SizeHintRole);
            width_ = width.toInt(&ok);
            if (ok)
            {
                m_breakPointView->setColumnWidth(i, dpiFactor * width_);
            }
        }

		//to adjust colspan of filename-rows
		connect(model, SIGNAL(rowsInserted(QModelIndex, int, int)), this, SLOT(dataChanged()));
		connect(model, SIGNAL(rowsRemoved(QModelIndex, int, int)), this, SLOT(dataChanged()));
    }

    setContentWidget(m_breakPointView);

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("itomBreakPointDockWidget");
    int size = settings.beginReadArray("ColWidth");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        m_breakPointView->setColumnWidth(i, settings.value("width", 100).toInt());
        m_breakPointView->setColumnHidden(i, m_breakPointView->columnWidth(i) == 0);
    }
    settings.endArray();
    settings.endGroup();

    updateActions();

	dataChanged(); //initially adjust colspan of filename-rows
}

//----------------------------------------------------------------------------------------------------------------------------------
BreakPointDockWidget::~BreakPointDockWidget()
{
    QAbstractItemModel *model = m_breakPointView->model();
    if (model)
    {
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup("itomBreakPointDockWidget");
        settings.beginWriteArray("ColWidth");
        for (int i = 0; i < model->columnCount(); i++)
        {
            settings.setArrayIndex(i);
            settings.setValue("width", m_breakPointView->columnWidth(i));
        }
        settings.endArray();
        settings.endGroup();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::actualizeTree(const QModelIndex &parent, int start, int end)
{
    m_breakPointView->expandAll();
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::createToolBars()
{
    QWidget *spacerWidget = new QWidget();
    QHBoxLayout *spacerLayout = new QHBoxLayout();
    spacerLayout->addItem(new QSpacerItem(5, 5, QSizePolicy::Expanding, QSizePolicy::Minimum));
    spacerLayout->setStretch(0, 2);
    spacerWidget->setLayout(spacerLayout);

    m_pMainToolbar = new QToolBar(tr("Breakpoints"), this);
    m_pMainToolbar->setObjectName("toolbarBreakpoints");
    m_pMainToolbar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolbar->setFloatable(false);
    addToolBar(m_pMainToolbar, "mainToolBar");

    m_pMainToolbar->addAction(m_pActDelBP->action());
    m_pMainToolbar->addAction(m_pActDelAllBPs->action());
    m_pMainToolbar->addAction(m_pActEditBP->action());
    m_pMainToolbar->addAction(m_pActToggleBP->action());
    m_pMainToolbar->addAction(m_pActToggleAllBPs->action());
    m_pMainToolbar->addWidget(spacerWidget);
    //connect(m_pFileSystemSettingMenu->menuAction(),SIGNAL(triggered()), this, SLOT(mnuToggleView()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::createActions()
{
    m_pActDelBP         = new ShortcutAction(QIcon(":/breakpoints/icons/garbageBP.png"), tr("Delete Breakpoint"), this);
    m_pActDelBP->connectTrigger(this, SLOT(mnuDeleteBP()));
    m_pActDelAllBPs     = new ShortcutAction(QIcon(":/breakpoints/icons/garbageAllBPs.png"), tr("Delete All Breakpoints"), this);
    m_pActDelAllBPs->connectTrigger(this, SLOT(mnuDeleteAllBPs()));
    m_pActEditBP        = new ShortcutAction(QIcon(":/breakpoints/icons/itomcBreak.png"), tr("Edit Breakpoints"), this);
    m_pActEditBP->connectTrigger(this, SLOT(mnuEditBreakpoint()));
    m_pActToggleBP      = new ShortcutAction(QIcon(":/breakpoints/icons/itomBreakDisable.png"), tr("En- Or Disable Breakpoint"), this);
    m_pActToggleBP->connectTrigger(this, SLOT(mnuEnOrDisAbleBrakpoint()));
    m_pActToggleAllBPs  = new ShortcutAction(QIcon(":/breakpoints/icons/itomBreakDisabledAll.png"), tr("En- Or Disable All Breakpoints"), this);
    m_pActToggleAllBPs->connectTrigger(this, SLOT(mnuEnOrDisAbleAllBrakpoints()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
    m_pContextMenu->addAction(m_pActDelBP->action());
    m_pContextMenu->addAction(m_pActDelAllBPs->action());
    m_pContextMenu->addAction(m_pActEditBP->action());
    m_pContextMenu->addAction(m_pActToggleBP->action());
    m_pContextMenu->addAction(m_pActToggleAllBPs->action());
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::treeViewContextMenuRequested(const QPoint &pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_breakPointView->mapToGlobal(m_breakPointView->pos()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::treeViewSelectionChanged(const QItemSelection &/*selected*/, const QItemSelection &/*deselected*/)
{
    updateActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::mnuDeleteBP()
{
    BreakPointModel *model = qobject_cast<BreakPointModel*>(m_breakPointView->model());
    if (model)
    {
        QModelIndexList delList;
        QModelIndexList allList = model->getAllFileIndexes();
        QModelIndexList selList = m_breakPointView->selectedIndexes();
        for (int i = 0; i < selList.size(); ++i)
        {
            if (!allList.contains(selList.at(i)))
            {
                delList.append(selList.at(i));
            }
        }
        model->deleteBreakPoints(delList);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::mnuDeleteAllBPs()
{
    BreakPointModel *model = qobject_cast<BreakPointModel*>(m_breakPointView->model());
    if (model)
    {
        model->deleteAllBreakPoints();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::mnuEditBreakpoint()
{
    BreakPointModel *model = qobject_cast<BreakPointModel*>(m_breakPointView->model());
    if (model)
    {
        if (m_breakPointView->selectedIndexes().length() == 1)
        {
            QModelIndex sel = m_breakPointView->selectedIndexes()[0];
            BreakPointItem bp = model->getBreakPoint(sel);
            if (bp.lineIdx > -1)
            {
                DialogEditBreakpoint *dlg = new DialogEditBreakpoint(bp.filename, bp.lineIdx+1, bp.enabled, bp.temporary , bp.ignoreCount, bp.condition);
                dlg->exec();
                if (dlg->result() == QDialog::Accepted)
                {
                    dlg->getData(bp.enabled, bp.temporary, bp.ignoreCount, bp.condition);
                    bp.conditioned = (bp.condition != "") || (bp.ignoreCount > 0) || bp.temporary;

                    model->changeBreakPoint(sel, bp);
                }

                DELETE_AND_SET_NULL(dlg);

                model->changeBreakPoint(sel, bp, true);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::mnuEnOrDisAbleBrakpoint()
{
    BreakPointModel *model = qobject_cast<BreakPointModel*>(m_breakPointView->model());
    if (model)
    {
        QModelIndexList selected;
        if (m_enOrDisAbleAllBreakpoints)
        { // select all
            selected = model->getAllBreakPointIndizes();
        }
        else
        { // use selection
            selected = m_breakPointView->selectedIndexes();
        }
        for (int i = 0; i<selected.length(); ++i)
        {
            BreakPointItem bp = model->getBreakPoint(selected[i]);
            if (bp.lineIdx > -1) //else the item is not valid
            {
                bp.enabled = !bp.enabled;
                model->changeBreakPoint(selected[i], bp, true);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::mnuEnOrDisAbleAllBrakpoints()
{
    m_breakPointView->clearSelection();
    m_enOrDisAbleAllBreakpoints = true;
    mnuEnOrDisAbleBrakpoint();
    m_enOrDisAbleAllBreakpoints = false;
    m_breakPointView->clearSelection();
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::updateActions()
{
    //m_pActDelBP (only active if only one or more breakpoints are selected)
    //m_pActDelAllBPs (always active)
    //m_pActEditBP (only active if exactly one breakpoint is selected)
    //m_pActToggleBP (only active if only one or more breakpoints are selected)
    //m_pActToggleAllBPs (always active)
    QModelIndexList sel = m_breakPointView->selectedIndexes();

    if (sel.length() == 1 && sel.at(0).parent().isValid())
    {
        m_pActEditBP->setEnabled(true);
    }
    else
    {
        m_pActEditBP->setEnabled(false);
    }

    m_pActToggleBP->setEnabled(sel.length() >= 1);
    m_pActDelBP->setEnabled(sel.length() >= 1);

    for (int i = 0; i < sel.length(); ++i)
    {
        if (!sel.at(i).parent().isValid())
        {
            m_pActToggleBP->setEnabled(false);
            m_pActDelBP->setEnabled(false);
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::doubleClicked(const QModelIndex &index)
{
    QString canonicalPath;
    int lineNr = -1;
    QModelIndex idx;
    QAbstractItemModel *m = m_breakPointView->model();

    if (index.isValid() && m)
    {
        idx = m->index(index.row(), 0, index.parent());
        canonicalPath = m->data(idx, Qt::ToolTipRole).toString();

        idx = m->index(index.row(), 0, index.parent());
        lineNr = m->data(idx, Qt::DisplayRole).toInt() - 1;

        if (canonicalPath.isEmpty() == false && canonicalPath.contains("<") == false)
        {
            ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
            if (seo)
            {
                seo->openScript(canonicalPath, NULL, lineNr);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void BreakPointDockWidget::dataChanged()
{
	//this slot is only necessary until the span-method of AbstractItemModel will be automatically considered and changes the column span.
	// added if due to crash on itom startup ck 26/01/2018
	if (m_breakPointView && m_breakPointView->model())
	{
		for (int row = 0; row < m_breakPointView->model()->rowCount(); row++)
		{
			QSize span = m_breakPointView->model()->span(m_breakPointView->model()->index(row, 0));
			m_breakPointView->setFirstColumnSpanned(row, QModelIndex(), span.width() > 1);
		}
	}
}

} //end namespace ito
