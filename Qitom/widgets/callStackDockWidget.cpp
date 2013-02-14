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

#include "../python/pythonEngineInc.h"

#include "callStackDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qmessagebox.h>
#include <qapplication.h>
#include <qheaderview.h>
#include <qfileinfo.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class CallStackDockWidget
    \brief docking
*/


CallStackDockWidget::CallStackDockWidget(const QString &title, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, parent),
	m_table(NULL)
{
    m_table = new QTableWidget(this);

    AbstractDockWidget::init();

	m_table->setColumnCount(3);
	m_table->setSortingEnabled(false);
	m_table->setTextElideMode( Qt::ElideLeft );
	m_table->verticalHeader()->setDefaultSectionSize(20);
	m_table->horizontalHeader()->setStretchLastSection(true);
	m_table->setAlternatingRowColors(true);
	m_table->setCornerButtonEnabled(false);

	m_headers << tr("file") << tr("line") << tr("method");
	m_table->setHorizontalHeaderLabels(m_headers);
	

    setContentWidget(m_table);

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    
    if(eng)
    {
		connect(eng, SIGNAL( updateCallStack(QStringList,IntList,QStringList) ), this, SLOT( updateCallStack(QStringList,IntList,QStringList) ));
		connect(eng, SIGNAL(deleteCallStack()), this, SLOT(deleteCallStack()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
CallStackDockWidget::~CallStackDockWidget()
{
    DELETE_AND_SET_NULL(m_table);
}

//----------------------------------------------------------------------------------------------------------------------------------
////! loads the given python dictionary by calling the appropriate method in its workspaceWidget.
///*!
//    \param dict [in] is the global or local python dictionary (depending on the role of this widget)
//    \param semaphore [in,out] is the semaphore, which is released if the load-operation has terminated.
//    \return retOk
//    \sa loadDictionary
//*/

//! implementation for virtual method createActions in AbstractDockWidget.
/*!
    creates all actions related to this widget. These actions will be used in the toolbars.
*/
void CallStackDockWidget::createActions()
{
    /*m_actDelete = new ShortcutAction(QIcon(":/workspace/icons/document-close-4.png"), tr("delete item(s)"), this, QKeySequence::Delete, Qt::WidgetWithChildrenShortcut);
    m_actDelete->connectTrigger(this, SLOT(mnuDeleteItem()));
    m_actExport = new ShortcutAction(QIcon(":/workspace/icons/document-export.png"), tr("export item(s)"), this);
    m_actExport->connectTrigger(this, SLOT(mnuExportItem()));
    m_actImport = new ShortcutAction(QIcon(":/workspace/icons/document-import.png"), tr("import item(s)"), this);
    m_actImport->connectTrigger(this, SLOT(mnuImportItem()));
    m_actRename = new ShortcutAction(QIcon(":/workspace/icons/edit-rename.png"), tr("rename item"), this, QKeySequence(tr("F2")), Qt::WidgetWithChildrenShortcut);
    m_actRename->connectTrigger(this, SLOT(mnuRenameItem()));*/
}

//----------------------------------------------------------------------------------------------------------------------------------
//! implementation for virtual method createToolBars in AbstractDockWidget.
/*!
    Creates the toolbar for this dock-widget with the necessary buttons, connected to existing actions.
*/
void CallStackDockWidget::createToolBars()
{
    /*m_pMainToolBar = new QToolBar(tr("script editor"),this);
    m_pMainToolBar->setFloatable(false);
    m_pMainToolBar->setAllowedAreas(Qt::TopToolBarArea);
    addAndRegisterToolBar(m_pMainToolBar, "mainToolBar");

    m_pMainToolBar->addAction( m_actImport->action() );
    m_pMainToolBar->addAction( m_actExport->action() );
    m_pMainToolBar->addAction( m_actDelete->action() );
    m_pMainToolBar->addAction( m_actRename->action() );*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void CallStackDockWidget::createMenus()
{
    /*m_pContextMenu = new QMenu(this);
    m_pContextMenu->addAction( m_actDelete->action() );
    m_pContextMenu->addAction( m_actRename->action() );
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction( m_actExport->action() );
    m_pContextMenu->addAction( m_actImport->action() );*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void CallStackDockWidget::updateCallStack(QStringList filenames, IntList lines, QStringList methods)
{
	QTableWidgetItem *item;
	QFileInfo info;
	m_table->clear();

	m_table->setRowCount(filenames.count());
	m_table->setHorizontalHeaderLabels(m_headers);

	if(lines.count() < filenames.count()) return;
	if(methods.count() < filenames.count()) return;

	for(int i = 0 ; i < filenames.count() ; i++)
	{
		info = QFileInfo(filenames[i]);
		item = new QTableWidgetItem( info.fileName() );
		item->setData(Qt::ToolTipRole, info.canonicalFilePath() );
		m_table->setItem(i,0, item);
		item = new QTableWidgetItem( QString::number(lines[i]) );
		m_table->setItem(i,1,item);
		item = new QTableWidgetItem(methods[i]);
		m_table->setItem(i,2,item);
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
void CallStackDockWidget::deleteCallStack()
{
	m_table->clear();
	m_table->setRowCount(0);
	m_table->setHorizontalHeaderLabels(m_headers);
}



} //end namespace ito
