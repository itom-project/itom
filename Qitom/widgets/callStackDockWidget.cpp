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
#include "../organizer/scriptEditorOrganizer.h"

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


CallStackDockWidget::CallStackDockWidget(const QString &title, const QString &objName, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_table(NULL)
{
    m_table = new QTableWidget(this);

    AbstractDockWidget::init();

    m_table->setColumnCount(3);
    m_table->setSortingEnabled(false);
    m_table->setTextElideMode(Qt::ElideLeft);
    m_table->verticalHeader()->setDefaultSectionSize(20);
    m_table->horizontalHeader()->setStretchLastSection(true);
    m_table->setAlternatingRowColors(true);
    m_table->setCornerButtonEnabled(false);

    m_headers << tr("file") << tr("line") << tr("method");
    m_table->setHorizontalHeaderLabels(m_headers);

    connect(m_table, SIGNAL(itemDoubleClicked(QTableWidgetItem*)), this, SLOT(itemDoubleClicked(QTableWidgetItem*)));
    
    setContentWidget(m_table);

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    
    if (eng)
    {
        connect(eng, SIGNAL(updateCallStack(QStringList,IntList,QStringList)), this, SLOT(updateCallStack(QStringList,IntList,QStringList)));
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
}

//----------------------------------------------------------------------------------------------------------------------------------
//! implementation for virtual method createToolBars in AbstractDockWidget.
/*!
    Creates the toolbar for this dock-widget with the necessary buttons, connected to existing actions.
*/
void CallStackDockWidget::createToolBars()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void CallStackDockWidget::createMenus()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void CallStackDockWidget::updateCallStack(QStringList filenames, IntList lines, QStringList methods)
{
    QTableWidgetItem *item;
    Qt::ItemFlags flagsEnabled = Qt::ItemIsSelectable | Qt::ItemIsEnabled;
    Qt::ItemFlags flagsDisabled = Qt::ItemIsSelectable;
    Qt::ItemFlags flags;
    QFileInfo info;
    QString filename;
    m_table->clear();

    m_table->setRowCount(filenames.count());
    m_table->setHorizontalHeaderLabels(m_headers);

    if (lines.count() < filenames.count()) return;
    if (methods.count() < filenames.count()) return;

    for (int i = 0 ; i < filenames.count() ; i++)
    {
        info = QFileInfo(filenames[i]);
        filename = info.fileName();
        if (filename.contains("<"))
        {
            flags = flagsDisabled;
        }
        else
        {
            flags = flagsEnabled;
        }

        item = new QTableWidgetItem(filename);
        item->setFlags(flags);
        item->setData(Qt::ToolTipRole, info.canonicalFilePath());
        m_table->setItem(i,0, item);

        item = new QTableWidgetItem(QString::number(lines[i]));
        item->setFlags(flags);
        m_table->setItem(i,1,item);

        item = new QTableWidgetItem(methods[i]);
        item->setFlags(flags);
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

//----------------------------------------------------------------------------------------------------------------------------------
void CallStackDockWidget::itemDoubleClicked(QTableWidgetItem *item)
{
    QString canonicalPath;
    int lineNr = -1;

    if (item)
    {
        QTableWidgetItem *item2 = m_table->item(item->row(),0);
        if (item2)
        {
            canonicalPath = item2->data(Qt::ToolTipRole).toString();

            item2 = m_table->item(item->row(), 1);
            if (item2)
            {
                lineNr = item2->text().toInt() - 1;
            }

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
}



} //end namespace ito
