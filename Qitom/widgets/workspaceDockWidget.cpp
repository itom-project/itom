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

#include "workspaceDockWidget.h"
#include "../helper/IOHelper.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qmessagebox.h>
#include <qapplication.h>
#include <qurl.h>
#include <qfileinfo.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class WorkspaceDockWidget
    \brief docking widget for contents of type workspace widget
*/

//! constructor
/*!
    long description

    \param title [in] is the docking widget's title
    \param globalNotLocal defines whether this widget contains global (true) or local (false) variables
    \param parent is a pointer to the parent widget [default: NULL]
    \param docked indicates whether this widget should appear docked (true) or undocked (false) [default: true]
    \param isDockAvailable indicates if this widget can be docked (true) or not (false) [default: true]
    \param floatingStyle indicates the style for the floating mode [default: floatingNone]
    \param movingStyle indicates the style for movement of the docked widget [default: movingEnabled]
*/

WorkspaceDockWidget::WorkspaceDockWidget(const QString &title, bool globalNotLocal, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, parent),
    m_globalNotLocal(globalNotLocal),
    m_pWorkspaceWidget(NULL),
    m_actDelete(NULL),
    m_actRename(NULL),
    m_actExport(NULL),
    m_actImport(NULL),
    m_pMainToolBar(NULL),
    m_pContextMenu(NULL),
    m_firstCurrentItem(NULL),
    m_firstCurrentItemKey(QString::Null())
{
    m_pWorkspaceWidget = new WorkspaceWidget(m_globalNotLocal, this);
    m_pWorkspaceWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_pWorkspaceWidget->setContextMenuPolicy( Qt::CustomContextMenu);

    AbstractDockWidget::init();

    setContentWidget(m_pWorkspaceWidget);

    connect(m_pWorkspaceWidget, SIGNAL(itemSelectionChanged()), this, SLOT(treeWidgetItemSelectionChanged()));
    connect(m_pWorkspaceWidget, SIGNAL(itemChanged(QTreeWidgetItem*,int)), this, SLOT(treeWidgetItemChanged(QTreeWidgetItem*,int)));
    connect(m_pWorkspaceWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(treeViewContextMenuRequested(const QPoint &)));

    ito::PyWorkspaceContainer *cont = m_pWorkspaceWidget->getWorkspaceContainer();
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    
    if(eng && cont)
    {
        QMetaObject::invokeMethod(eng, "registerWorkspaceContainer", Q_ARG(PyWorkspaceContainer*,cont), Q_ARG(bool,true), Q_ARG(bool,m_globalNotLocal));
    }

    setAcceptDrops(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
WorkspaceDockWidget::~WorkspaceDockWidget()
{
    disconnect(m_pWorkspaceWidget, SIGNAL(itemSelectionChanged()), this, SLOT(treeWidgetItemSelectionChanged()));
    disconnect(m_pWorkspaceWidget, SIGNAL(itemChanged(QTreeWidgetItem*,int)), this, SLOT(treeWidgetItemChanged(QTreeWidgetItem*,int)));

    ito::PyWorkspaceContainer *cont = m_pWorkspaceWidget->getWorkspaceContainer();
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    
    if(eng && cont)
    {
        QMetaObject::invokeMethod(eng, "registerWorkspaceContainer",  Qt::BlockingQueuedConnection, Q_ARG(PyWorkspaceContainer*,cont), Q_ARG(bool,false), Q_ARG(bool,m_globalNotLocal));
    }

    m_pWorkspaceWidget->deleteLater(); //important, since the above invokation still needs the container
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
void WorkspaceDockWidget::createActions()
{
    m_actDelete = new ShortcutAction(QIcon(":/workspace/icons/document-close-4.png"), tr("delete item(s)"), this, QKeySequence::Delete, Qt::WidgetWithChildrenShortcut);
    m_actDelete->connectTrigger(this, SLOT(mnuDeleteItem()));
    m_actExport = new ShortcutAction(QIcon(":/workspace/icons/document-export.png"), tr("export item(s)"), this);
    m_actExport->connectTrigger(this, SLOT(mnuExportItem()));
    m_actImport = new ShortcutAction(QIcon(":/workspace/icons/document-import.png"), tr("import item(s)"), this);
    m_actImport->connectTrigger(this, SLOT(mnuImportItem()));
    m_actRename = new ShortcutAction(QIcon(":/workspace/icons/edit-rename.png"), tr("rename item"), this, QKeySequence(tr("F2")), Qt::WidgetWithChildrenShortcut);
    m_actRename->connectTrigger(this, SLOT(mnuRenameItem()));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! implementation for virtual method createToolBars in AbstractDockWidget.
/*!
    Creates the toolbar for this dock-widget with the necessary buttons, connected to existing actions.
*/
void WorkspaceDockWidget::createToolBars()
{
    m_pMainToolBar = new QToolBar(tr("script editor"),this);
    m_pMainToolBar->setFloatable(false);
    m_pMainToolBar->setAllowedAreas(Qt::TopToolBarArea);
    addToolBar(m_pMainToolBar,"mainToolBar");
    //addAndRegisterToolBar(m_pMainToolBar, "mainToolBar");

    m_pMainToolBar->addAction( m_actImport->action() );
    m_pMainToolBar->addAction( m_actExport->action() );
    m_pMainToolBar->addAction( m_actDelete->action() );
    m_pMainToolBar->addAction( m_actRename->action() );
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
    m_pContextMenu->addAction( m_actDelete->action() );
    m_pContextMenu->addAction( m_actRename->action() );
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction( m_actExport->action() );
    m_pContextMenu->addAction( m_actImport->action() );
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::treeViewContextMenuRequested(const QPoint & /*pos*/)
{
    updateActions();
//    m_pContextMenu->exec(pos + m_firstCurrentItem->mapToGlobal(m_firstCurrentItem->pos()));
    m_pContextMenu->exec(QCursor::pos());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the delete button has been clicked
/*!
    After accepting a security query, the selected variables will be deleted by invoking the slot deleteVariable
    in pythonEngine.

    \sa deleteVariable
*/
void WorkspaceDockWidget::mnuDeleteItem()
{
    if(m_pWorkspaceWidget != NULL && m_pWorkspaceWidget->numberOfSelectedMainItems() >= 1)
    {
         QMessageBox msgBox;
         msgBox.setText(tr("Do you really want to delete the selected variables?"));
         msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
         msgBox.setDefaultButton(QMessageBox::Yes);
         int ret = msgBox.exec();

         PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

         if(ret == QMessageBox::Yes && eng != NULL)
         {
             QList<QTreeWidgetItem*> itemList = m_pWorkspaceWidget->selectedItems();
             QStringList keyList;

             for(int i = 0; i < itemList.size(); i++)
             {
                 if(itemList.at(i)->parent() == NULL)
                 {
                    keyList.append(itemList.at(i)->data(0,Qt::DisplayRole).toString());
                 }
             }

             QMetaObject::invokeMethod(eng, "deleteVariable", Q_ARG(bool,m_globalNotLocal), Q_ARG(QStringList,keyList), Q_ARG(ItomSharedSemaphore*, NULL));
         }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the import button has been clicked
/*!
    A file-dialog appears where an idc (pickle)-file can be chosen, where the selected variables should be pickled to.
    An error message will appear if the export failed.

    \sa uiExportPyWorkspaceVars
*/
void WorkspaceDockWidget::mnuExportItem()
{
    if(m_pWorkspaceWidget != NULL && m_pWorkspaceWidget->numberOfSelectedMainItems() >= 1)
    {
        QList<QTreeWidgetItem*> itemList = m_pWorkspaceWidget->selectedItems();
        QStringList keyList;
        QVector<int> compatibleParamBaseTypes; //Type of ParamBase, which is compatible to this value, or 0 if not compatible
        QTreeWidgetItem * item;
        foreach(item, itemList)
        {
            if(item->parent() == NULL)
            {
                keyList.append(item->data(0,Qt::DisplayRole).toString());
                compatibleParamBaseTypes.append(item->data(0, Qt::UserRole + 2).toInt());;
            }
        }

        RetVal retValue = IOHelper::uiExportPyWorkspaceVars(m_globalNotLocal, keyList, compatibleParamBaseTypes, QString::Null(), this);
        if(retValue.containsError())
        {
            char *errorMsg = retValue.errorMessage();
            QString message = QString();
            if(errorMsg) message =errorMsg;
            //std::cerr << "error while exporting variables. reason: " << message.toAscii().data() << "\n" << std::endl;
            QMessageBox::critical(this, tr("Export data"), tr("Error while exporting variables: \n%1").arg( message ));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the import button has been clicked
/*!
    A file-dialog appears where an idc (pickle)-file can be chosen, whose content should be load to the python workspace.
    An error message will appear if the import failed.

    \sa uiImportPyWorkspaceVars
*/
void WorkspaceDockWidget::mnuImportItem()
{
    RetVal retValue = IOHelper::uiImportPyWorkspaceVars(m_globalNotLocal, IOHelper::IOFilters(IOHelper::IOPlugin |IOHelper::IOInput | IOHelper::IOWorkspace | IOHelper::IOMimeAll), QString::Null(), this);
    if(retValue.containsError())
    {
        char *errorMsg = retValue.errorMessage();
        QString message = QString();
        if(errorMsg) message = errorMsg;
        QMessageBox::critical(this, tr("Import data"), tr("Error while importing variables: \n%1").arg( message ));
        //std::cerr << "error while importing variables. reason: " << message.toAscii().data() << "\n" << std::endl;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the rename button has been clicked in the menu
/*!
    this slot forces the current item in the treeview to become editable (editMode)
*/
void WorkspaceDockWidget::mnuRenameItem()
{
    if(m_pWorkspaceWidget != NULL && m_pWorkspaceWidget->numberOfSelectedMainItems() == 1 && m_firstCurrentItem != NULL)
    {
        m_firstCurrentItemKey = m_firstCurrentItem->data(0, Qt::DisplayRole).toString();
        m_pWorkspaceWidget->editItem(m_firstCurrentItem,0);
    }
    else
    {
        m_firstCurrentItemKey = QString::Null();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates the status of all actions of this widget
/*!
    The update concerns mainly the visible and enabled-status of each action,
    depending on different influences, like e.g. the python status.

    \sa pythonInWaitingMode, pythonBusy
*/
void WorkspaceDockWidget::updateActions()
{
    if(m_pWorkspaceWidget != NULL)
    {
        int num = m_pWorkspaceWidget->numberOfSelectedMainItems();
        int i=0;

        if(num > 0)
        {
            m_firstCurrentItem = NULL;
            while(m_firstCurrentItem == NULL && i < m_pWorkspaceWidget->selectedItems().count())
            {
                m_firstCurrentItem = m_pWorkspaceWidget->selectedItems()[i++];
            }
        }
        else
        {
            m_firstCurrentItem = NULL;
        }

        if(m_globalNotLocal)
        {
            bool pythonFree = (pythonBusy() == false || pythonInWaitingMode());
            m_actDelete->setEnabled(num > 0 && pythonFree);
            m_actExport->setEnabled(num > 0 && pythonFree);
            m_actImport->setEnabled(pythonFree);
            m_actRename->setEnabled(num == 1 && pythonFree);
        }
        else
        {
            m_actDelete->setEnabled(num > 0 && pythonInWaitingMode());
            m_actExport->setEnabled(num > 0 && pythonInWaitingMode());
            m_actImport->setEnabled(pythonInWaitingMode());
            m_actRename->setEnabled(num == 1 && pythonInWaitingMode());
            m_pWorkspaceWidget->setEnabled( pythonInWaitingMode() );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if name of element in workspaceWidget (TreeView) has been changed.
/*!
    calls pythonEngine's method renameVariable in order to initiate the renaming operation in python.

    \param item [in] is the corresponding QTreeWidgetItem, whose name has manually been changed
    \sa renameVariable
*/
void WorkspaceDockWidget::treeWidgetItemChanged(QTreeWidgetItem * item, int /*column*/)
{
    QString newKey = item->data(0, Qt::DisplayRole).toString();

    if(newKey != m_firstCurrentItemKey && m_firstCurrentItemKey != "" && newKey != "")
    {
        
        PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));

        emit setStatusInformation(tr("renaming variable"),0);

        QMetaObject::invokeMethod(eng, "renameVariable", Q_ARG(bool,m_globalNotLocal), Q_ARG(QString,m_firstCurrentItemKey), Q_ARG(QString,newKey), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        m_firstCurrentItemKey = QString::Null();

        if(locker.getSemaphore()->waitAndProcessEvents(PLUGINWAIT))
        {
            emit setStatusInformation("",0);
        }
        else
        {
            emit setStatusInformation(tr("timeout while renaming variables"), 5000);
        }
        QApplication::restoreOverrideCursor();
    }
}

void WorkspaceDockWidget::dragEnterEvent(QDragEnterEvent *event)
{
    if(m_globalNotLocal == false && !pythonInWaitingMode())
    {
        //local workspace is only active if python is in waiting mode
        return;
    }

    if( event->mimeData()->hasFormat("text/uri-list") ) //or hasUrls() should be the same result
    {
        QList<QUrl> urls = event->mimeData()->urls();

        QStringList allPatterns;
        IOHelper::getFileFilters( IOHelper::IOFilters(IOHelper::IOPlugin |IOHelper::IOInput | IOHelper::IOWorkspace | IOHelper::IOMimeAll) , &allPatterns);
        QRegExp reg;
        bool ok = false;
        reg.setPatternSyntax( QRegExp::Wildcard );

        //check files
        foreach(const QUrl &url, urls)
        {
            qDebug() << url.toLocalFile();
#if QT_VERSION >= 0x040800
            if (url.isLocalFile() == false) //this method has been introduced in Qt 4.8
            {
#else
            if (url.scheme().compare( QLatin1String("file"), Qt::CaseInsensitive ) != 0)
            {
#endif
                return;
            }

            foreach(const QString &pat, allPatterns)
            {
                reg.setPattern(pat);
                if(reg.exactMatch( url.toLocalFile() ))
                {
                    ok = true;
                    break;
                }
            }

            if(!ok) return;
        }


        event->acceptProposedAction();
    }
}

void WorkspaceDockWidget::dropEvent(QDropEvent *event)
{
    QList<QUrl> urls = event->mimeData()->urls();
    bool ok = true;
    QFileInfo finfo;

    //check files
    foreach(const QUrl &url, urls)
    {
        IOHelper::openGeneralFile(url.toLocalFile(), false, true, this, 0, m_globalNotLocal);
    }
}

} //end namespace ito
