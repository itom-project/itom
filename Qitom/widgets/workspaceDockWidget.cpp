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

#include "../AppManagement.h"
#include "../common/typeDefs.h"
#include "../global.h"
#include "../helper/IOHelper.h"
#include "../helper/compatHelper.h"
#include "../organizer/uiOrganizer.h"
#include "workspaceDockWidget.h"

#include <qapplication.h>
#include <qfileinfo.h>
#include <qmessagebox.h>
#include <qmimedata.h>
#include <qsettings.h>
#include <qurl.h>
#include <qregularexpression.h>

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
    \param globalNotLocal defines whether this widget contains global (true) or local (false)
   variables \param parent is a pointer to the parent widget [default: NULL] \param docked indicates
   whether this widget should appear docked (true) or undocked (false) [default: true] \param
   isDockAvailable indicates if this widget can be docked (true) or not (false) [default: true]
    \param floatingStyle indicates the style for the floating mode [default: floatingNone]
    \param movingStyle indicates the style for movement of the docked widget [default:
   movingEnabled]
*/

WorkspaceDockWidget::WorkspaceDockWidget(
    const QString& title,
    const QString& objName,
    bool globalNotLocal,
    QWidget* parent,
    bool docked,
    bool isDockAvailable,
    tFloatingStyle floatingStyle,
    tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_globalNotLocal(globalNotLocal), m_pWorkspaceWidget(NULL), m_actDelete(NULL),
    m_actRename(NULL), m_actExport(NULL), m_actImport(NULL), m_dObjPlot1d(NULL), m_dObjPlot2d(NULL),
    m_dObjPlot25d(NULL), m_separatorSpecialActionsToolBar(NULL),
    m_separatorSpecialActionsContextMenu(NULL), m_separatorDisplayItemDetailsActionsToolBar(NULL),
    m_separatorDisplayItemDetailsActionsContextMenu(NULL), m_pMainToolBar(NULL),
    m_pContextMenu(NULL), m_firstCurrentItem(NULL), m_actClearAll(NULL),
    m_firstCurrentItemKey(QString())
{
    m_pWorkspaceWidget = new WorkspaceWidget(m_globalNotLocal, this);
    m_pWorkspaceWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_pWorkspaceWidget->setContextMenuPolicy(Qt::CustomContextMenu);

    AbstractDockWidget::init();

    setContentWidget(m_pWorkspaceWidget);

    connect(
        m_pWorkspaceWidget,
        SIGNAL(itemSelectionChanged()),
        this,
        SLOT(treeWidgetItemSelectionChanged()));
    connect(
        m_pWorkspaceWidget,
        SIGNAL(itemChanged(QTreeWidgetItem*, int)),
        this,
        SLOT(treeWidgetItemChanged(QTreeWidgetItem*, int)));
    connect(
        m_pWorkspaceWidget,
        SIGNAL(customContextMenuRequested(const QPoint&)),
        this,
        SLOT(treeViewContextMenuRequested(const QPoint&)));
    connect(
        AppManagement::getMainApplication(),
        SIGNAL(propertiesChanged()),
        this,
        SLOT(propertiesChanged()));

    ito::PyWorkspaceContainer* cont = m_pWorkspaceWidget->getWorkspaceContainer();
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (eng && cont)
    {
        QMetaObject::invokeMethod(
            eng,
            "registerWorkspaceContainer",
            Q_ARG(PyWorkspaceContainer*, cont),
            Q_ARG(bool, true),
            Q_ARG(bool, m_globalNotLocal));
    }

    setAcceptDrops(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
WorkspaceDockWidget::~WorkspaceDockWidget()
{
    disconnect(
        m_pWorkspaceWidget,
        SIGNAL(itemSelectionChanged()),
        this,
        SLOT(treeWidgetItemSelectionChanged()));
    disconnect(
        m_pWorkspaceWidget,
        SIGNAL(itemChanged(QTreeWidgetItem*, int)),
        this,
        SLOT(treeWidgetItemChanged(QTreeWidgetItem*, int)));
    disconnect(
        m_pWorkspaceWidget,
        SIGNAL(customContextMenuRequested(const QPoint&)),
        this,
        SLOT(treeViewContextMenuRequested(const QPoint&)));
    disconnect(
        AppManagement::getMainApplication(),
        SIGNAL(propertiesChanged()),
        this,
        SLOT(propertiesChanged()));

    ito::PyWorkspaceContainer* cont = m_pWorkspaceWidget->getWorkspaceContainer();
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (eng && cont)
    {
        QMetaObject::invokeMethod(
            eng,
            "registerWorkspaceContainer",
            Qt::BlockingQueuedConnection,
            Q_ARG(PyWorkspaceContainer*, cont),
            Q_ARG(bool, false),
            Q_ARG(bool, m_globalNotLocal));
    }

    m_pWorkspaceWidget
        ->deleteLater(); // important, since the above invokation still needs the container
}

//----------------------------------------------------------------------------------------------------------------------------------
////! loads the given python dictionary by calling the appropriate method in its workspaceWidget.
///*!
//    \param dict [in] is the global or local python dictionary (depending on the role of this
//    widget) \param semaphore [in,out] is the semaphore, which is released if the load-operation
//    has terminated. \return retOk \sa loadDictionary
//*/

//! implementation for virtual method createActions in AbstractDockWidget.
/*!
    creates all actions related to this widget. These actions will be used in the toolbars.
*/
void WorkspaceDockWidget::createActions()
{
    m_actDelete = new ShortcutAction(
        QIcon(":/workspace/icons/document-close-4.png"),
        tr("Delete Selected Item(s)"),
        this,
        QKeySequence::Delete,
        Qt::WidgetWithChildrenShortcut);
    m_actDelete->connectTrigger(this, SLOT(mnuDeleteItem()));
    m_actExport = new ShortcutAction(
        QIcon(":/workspace/icons/document-export.png"),
        tr("Export Selected Item(s)"),
        this,
        QKeySequence::Save,
        Qt::WidgetWithChildrenShortcut);
    m_actExport->connectTrigger(this, SLOT(mnuExportItem()));
    m_actImport = new ShortcutAction(
        QIcon(":/workspace/icons/document-import.png"), tr("Import Item(s)"), this);
    m_actImport->connectTrigger(this, SLOT(mnuImportItem()));
    m_actRename = new ShortcutAction(
        QIcon(":/workspace/icons/edit-rename.png"),
        tr("Rename Selected Item"),
        this,
        QKeySequence(tr("F2")),
        Qt::WidgetWithChildrenShortcut);
    m_actRename->connectTrigger(this, SLOT(mnuRenameItem()));

    m_dObjPlot1d =
        new ShortcutAction(QIcon(":/plots/icons/itom_icons/1d.png"), tr("1D Line Plot"), this);
    m_dObjPlot1d->connectTrigger(this, SLOT(mnuPlot1D()));
    m_dObjPlot2d =
        new ShortcutAction(QIcon(":/plots/icons/itom_icons/2d.png"), tr("2D Image Plot"), this);
    m_dObjPlot2d->connectTrigger(this, SLOT(mnuPlot2D()));
    m_dObjPlot25d = new ShortcutAction(
        QIcon(":/plots/icons/itom_icons/3d.png"), tr("2.5D Isometric Plot"), this);
    m_dObjPlot25d->connectTrigger(this, SLOT(mnuPlot25D()));
    m_dObjPlot3d = new ShortcutAction(
        QIcon(":/plots/icons/itom_icons/3d.png"), tr("3D Cloud Or Mesh Visualization"), this);
    m_dObjPlot3d->connectTrigger(this, SLOT(mnuPlot25D()));

    m_actUnpack =
        new QAction(QIcon(":/application/icons/unpack.png"), tr("Unpack Loaded Dictionary"), this);
    m_actUnpack->setToolTip(tr("Unpack loaded dictionary from idc or mat files to workspace"));
    m_actUnpack->setCheckable(true);
    connect(m_actUnpack, SIGNAL(triggered()), this, SLOT(mnuToggleUnpack()));
    checkToggleUnpack();

    m_actClearAll = new ShortcutAction(
        QIcon(":/workspace/icons/closeAll.png"), tr("Clear All Variables"), this);
    m_actClearAll->connectTrigger(this, SLOT(mnuClearAll()));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! implementation for virtual method createToolBars in AbstractDockWidget.
/*!
    Creates the toolbar for this dock-widget with the necessary buttons, connected to existing
   actions.
*/
void WorkspaceDockWidget::createToolBars()
{
    m_pMainToolBar = new QToolBar(tr("Workspace"), this);
    m_pMainToolBar->setObjectName("toolbarWorkspace");
    m_pMainToolBar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolBar->setFloatable(false);
    //    m_pMainToolBar->setAllowedAreas(Qt::TopToolBarArea);
    addToolBar(m_pMainToolBar, "mainToolBar");
    // addAndRegisterToolBar(m_pMainToolBar, "mainToolBar");

    m_pMainToolBar->addAction(m_actImport->action());
    m_pMainToolBar->addAction(m_actExport->action());
    m_pMainToolBar->addAction(m_actUnpack);
    m_pMainToolBar->addSeparator();
    m_pMainToolBar->addAction(m_actDelete->action());
    m_pMainToolBar->addAction(m_actClearAll->action());
    m_pMainToolBar->addAction(m_actRename->action());
    m_separatorSpecialActionsToolBar = m_pMainToolBar->addSeparator();
    m_pMainToolBar->addAction(m_dObjPlot1d->action());
    m_pMainToolBar->addAction(m_dObjPlot2d->action());
    m_pMainToolBar->addAction(m_dObjPlot25d->action());
    m_pMainToolBar->addAction(m_dObjPlot3d->action());
    m_separatorDisplayItemDetailsActionsToolBar = m_pMainToolBar->addSeparator();
    m_pMainToolBar->addAction(m_pWorkspaceWidget->m_displayItemDetails);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
    //    m_pContextMenu->addAction(m_actImport->action());
    m_pContextMenu->addAction(m_actExport->action());
    //    m_pContextMenu->addAction(m_actUnpack);
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_actDelete->action());
    m_pContextMenu->addAction(m_actRename->action());
    m_separatorSpecialActionsContextMenu = m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_dObjPlot1d->action());
    m_pContextMenu->addAction(m_dObjPlot2d->action());
    m_pContextMenu->addAction(m_dObjPlot25d->action());
    m_pContextMenu->addAction(m_dObjPlot3d->action());
    m_separatorSpecialActionsContextMenu = m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pWorkspaceWidget->m_displayItemDetails);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::treeViewContextMenuRequested(const QPoint& /*pos*/)
{
    updateActions();
    m_pContextMenu->exec(QCursor::pos());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the delete button has been clicked
/*!
    After accepting a security query, the selected variables will be deleted by invoking the slot
   deleteVariable in pythonEngine.

    \sa deleteVariable
*/
void WorkspaceDockWidget::mnuDeleteItem()
{
    if (m_pWorkspaceWidget != NULL && m_pWorkspaceWidget->numberOfSelectedItems() >= 1)
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Do you really want to delete the selected variables?"));
        msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);
        int ret = msgBox.exec();

        PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

        if (ret == QMessageBox::Yes && eng != NULL)
        {
            QList<QTreeWidgetItem*> itemList = m_pWorkspaceWidget->selectedItems();
            QStringList keyList;

            QTreeWidgetItem* parent;
            bool ignore;
            foreach (const QTreeWidgetItem* item, itemList)
            {
                if (item->parent() == NULL)
                {
                    keyList.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
                }
                else // check if parent or parent of parent is also selected. If so, ignore item
                {
                    parent = item->parent();
                    ignore = false;
                    while (parent && !ignore)
                    {
                        if (itemList.contains(parent))
                        {
                            ignore = true;
                        }

                        parent = parent->parent();
                    }

                    if (!ignore)
                    {
                        keyList.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
                    }
                }
            }

            QMetaObject::invokeMethod(
                eng,
                "deleteVariable",
                Q_ARG(bool, m_globalNotLocal),
                Q_ARG(QStringList, keyList));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::mnuClearAll()
{
    RetVal retVal;
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        retVal += RetVal(retError, 1, tr("Python engine not available").toLatin1().data());
    }
    else if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        retVal += RetVal(
            retError,
            2,
            tr("Variables cannot be plot since python is busy right now").toLatin1().data());
    }
    QMetaObject::invokeMethod(eng, "pythonClearAll");
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the import button has been clicked
/*!
    A file-dialog appears where an idc (pickle)-file can be chosen, where the selected variables
   should be pickled to. An error message will appear if the export failed.

    \sa uiExportPyWorkspaceVars
*/
void WorkspaceDockWidget::mnuExportItem()
{
    if (m_pWorkspaceWidget != NULL && m_pWorkspaceWidget->numberOfSelectedItems() >= 1)
    {
        QList<QTreeWidgetItem*> itemList = m_pWorkspaceWidget->selectedItems();
        QStringList keyList;
        QVector<int> compatibleParamBaseTypes; // Type of ParamBase, which is compatible to this
                                               // value, or 0 if not compatible
        QTreeWidgetItem* parent;
        bool ignore;
        foreach (const QTreeWidgetItem* item, itemList)
        {
            if (item->parent() == NULL)
            {
                keyList.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
                compatibleParamBaseTypes.append(
                    item->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt());
            }
            else // check if parent or parent of parent is also selected. If so, ignore item
            {
                parent = item->parent();
                ignore = false;
                while (parent && !ignore)
                {
                    if (itemList.contains(parent))
                    {
                        ignore = true;
                    }

                    parent = parent->parent();
                }

                if (!ignore)
                {
                    keyList.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
                    compatibleParamBaseTypes.append(
                        item->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt());
                }
            }
        }

        RetVal retValue = IOHelper::uiExportPyWorkspaceVars(
            m_globalNotLocal, keyList, compatibleParamBaseTypes, QString(), this);
        if (retValue.containsError())
        {
            const char* errorMsg = retValue.errorMessage();
            QString message = errorMsg ? QLatin1String(errorMsg) : QString();
            QMessageBox::critical(
                this, tr("Export data"), tr("Error while exporting variables:\n%1").arg(message));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the import button has been clicked
/*!
    A file-dialog appears where an idc (pickle)-file can be chosen, whose content should be load to
   the python workspace. An error message will appear if the import failed.

    \sa uiImportPyWorkspaceVars
*/
void WorkspaceDockWidget::mnuImportItem()
{
    RetVal retValue = IOHelper::uiImportPyWorkspaceVars(
        m_globalNotLocal,
        IOHelper::IOPlugin | IOHelper::IOInput | IOHelper::IOWorkspace | IOHelper::IOMimeAll,
        QString(),
        this);

    if (retValue.containsError())
    {
        const char* errorMsg = retValue.errorMessage();
        QString message = QString();
        if (errorMsg)
            message = errorMsg;
        QMessageBox::critical(
            this, tr("Import data"), tr("Error while importing variables:\n%1").arg(message));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the rename button has been clicked in the menu
/*!
    this slot forces the current item in the treeview to become editable (editMode)
*/
void WorkspaceDockWidget::mnuRenameItem()
{
    if (m_pWorkspaceWidget != NULL && m_pWorkspaceWidget->numberOfSelectedItems(true) == 1 &&
        m_firstCurrentItem != NULL)
    {
        m_firstCurrentItemKey = m_firstCurrentItem->data(0, Qt::DisplayRole).toString();
        m_pWorkspaceWidget->editItem(m_firstCurrentItem, 0);
    }
    else
    {
        m_firstCurrentItemKey = QString();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if the unpack dictionary button has been clicked in the menu
/*!
    when importing an *.idc or *.mat file to the workspace, it is either possible to unpack all
   values within the file and load them as separate variables to the workspace or to load the
   content of the file as one single dictionary (name of the dictionary will be requested by an
   input dialog)
*/
void WorkspaceDockWidget::mnuToggleUnpack()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Workspace");
    settings.setValue("importIdcMatUnpackDict", m_actUnpack->isChecked());
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::mnuPlot1D()
{
    mnuPlotGeneric("1d");
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::mnuPlot2D()
{
    mnuPlotGeneric("2d");
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::mnuPlot25D()
{
    mnuPlotGeneric("2.5d");
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::mnuPlotGeneric(const QString& plotClass)
{
    // try to open it with filters
    RetVal retVal;
    QSharedPointer<ito::ParamBase> value;
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    QStringList keyList;
    QVector<int> compatibleParamBaseTypes; // Type of ParamBase, which is compatible to this value,
                                           // or 0 if not compatible
    QStringList keyListFinal;
    QVector<int> compatibleParamBaseTypesFinal;
    QVector<const QTreeWidgetItem*> items;
    QVector<const QTreeWidgetItem*> itemsFinal;

    if (eng == NULL)
    {
        retVal += RetVal(retError, 1, tr("Python engine not available").toLatin1().data());
    }
    else if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        retVal += RetVal(
            retError,
            2,
            tr("Variables cannot be plot since python is busy right now").toLatin1().data());
    }
    else if (!m_pWorkspaceWidget)
    {
        retVal += RetVal(retError, 3, tr("Workspace not available").toLatin1().data());
    }
    else if (m_pWorkspaceWidget->numberOfSelectedItems() == 0)
    {
        retVal += RetVal(retError, 5, tr("Nothing selected").toLatin1().data());
    }
    else
    {
        QList<QTreeWidgetItem*> itemList = m_pWorkspaceWidget->selectedItems();
        QTreeWidgetItem* parent;
        bool ignore;
        foreach (const QTreeWidgetItem* item, itemList)
        {
            if (item->parent() == NULL)
            {
                keyList.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
                compatibleParamBaseTypes.append(
                    item->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt());
                items.append(item);
            }
            else // check if parent or parent of parent is also selected. If so, ignore item
            {
                parent = item->parent();
                ignore = false;
                while (parent && !ignore)
                {
                    if (itemList.contains(parent))
                    {
                        ignore = true;
                    }

                    parent = parent->parent();
                }

                if (!ignore)
                {
                    keyList.append(item->data(0, WorkspaceWidget::RoleFullName).toString());
                    compatibleParamBaseTypes.append(
                        item->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt());
                    items.append(item);
                }
            }
        }
    }

    if (!retVal.containsError())
    {
        keyListFinal.reserve(keyList.size());
        compatibleParamBaseTypesFinal.reserve(compatibleParamBaseTypesFinal.size());
        itemsFinal.reserve(items.size());

        // check that only dataObjects are plot
        for (int i = 0; i < keyList.size(); ++i)
        {
            if ((compatibleParamBaseTypes[i] == ito::ParamBase::DObjPtr) ||
                (compatibleParamBaseTypes[i] == ito::ParamBase::PointCloudPtr) ||
                (compatibleParamBaseTypes[i] == ito::ParamBase::PolygonMeshPtr))
            {
                keyListFinal.append(keyList[i]);
                compatibleParamBaseTypesFinal.append(compatibleParamBaseTypes[i]);
                itemsFinal.append(items[i]);
            }
            else
            {
                retVal += ito::RetVal(
                    ito::retWarning,
                    0,
                    tr("At least one variable cannot be plotted since it is no dataObject or "
                       "numpy.array. These values are ignored.")
                        .toLatin1()
                        .data());
            }
        }
    }

    if (!retVal.containsError())
    {
        // get values from workspace
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QSharedPointer<SharedParamBasePointerVector> values(new SharedParamBasePointerVector());
        QMetaObject::invokeMethod(
            eng,
            "getParamsFromWorkspace",
            Q_ARG(bool, m_globalNotLocal),
            Q_ARG(QStringList, keyListFinal),
            Q_ARG(QVector<int>, compatibleParamBaseTypesFinal),
            Q_ARG(QSharedPointer<SharedParamBasePointerVector>, values),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        if (!locker.getSemaphore()->wait(5000))
        {
            retVal += RetVal(
                retError, 0, tr("Timeout while getting value from workspace").toLatin1().data());
        }
        else
        {
            retVal += locker.getSemaphore()->returnValue;
        }

        if (values->size() != keyListFinal.size())
        {
            retVal += RetVal(
                retError,
                0,
                tr("The number of values returned from workspace does not correspond to requested "
                   "number")
                    .toLatin1()
                    .data());
        }

        if (!retVal.containsError())
        {
            QVariantMap properties;
            int areaCol = 0;
            int areaRow = 0;
            const ito::DataObject* obj = NULL;
#if ITOM_POINTCLOUDLIBRARY > 0
            const ito::PCLPointCloud* cloud = NULL;
            const ito::PCLPolygonMesh* mesh = NULL;
#endif

            for (int i = 0; i < values->size(); ++i)
            {
                ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());

                QSharedPointer<unsigned int> figHandle(new unsigned int);
                *figHandle = 0; // new figure will be requested

                UiOrganizer* uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
                ito::UiDataContainer dataCont;

                if (values->at(i)->getType() == (ito::ParamBase::DObjPtr))
                {
                    obj = (*values)[i]->getVal<const ito::DataObject*>();
                    dataCont = ito::UiDataContainer(
                        QSharedPointer<ito::DataObject>(new ito::DataObject(*obj)));
                }
#if ITOM_POINTCLOUDLIBRARY > 0
                else if (values->at(i)->getType() == (ito::ParamBase::PointCloudPtr))
                {
                    cloud = (*values)[i]->getVal<const ito::PCLPointCloud*>();
                    dataCont = ito::UiDataContainer(
                        QSharedPointer<ito::PCLPointCloud>(new ito::PCLPointCloud(*cloud)));
                    obj = NULL;
                }
                else if (values->at(i)->getType() == (ito::ParamBase::PolygonMeshPtr))
                {
                    mesh = (*values)[i]->getVal<const ito::PCLPolygonMesh*>();
                    dataCont = ito::UiDataContainer(
                        QSharedPointer<ito::PCLPolygonMesh>(new ito::PCLPolygonMesh(*mesh)));
                    obj = NULL;
                }
#endif
                else
                {
                    retVal += RetVal(
                        retError,
                        0,
                        tr("Invalid or unsupported data type for plotting.").toLatin1().data());
                    break;
                }

                if (obj && !obj->existTag("title"))
                {
                    properties["title"] = m_pWorkspaceWidget->getPythonReadableName(itemsFinal[i]);
                }
                else
                {
                    properties.remove("title");
                }

                QSharedPointer<unsigned int> objectID(new unsigned int);
                ito::UiDataContainer xAxisCont;

                QMetaObject::invokeMethod(
                    uiOrg,
                    "figurePlot",
                    Q_ARG(ito::UiDataContainer&, dataCont),
                    Q_ARG(ito::UiDataContainer&, xAxisCont),
                    Q_ARG(QSharedPointer<uint>, figHandle),
                    Q_ARG(QSharedPointer<uint>, objectID),
                    Q_ARG(int, areaRow),
                    Q_ARG(int, areaCol),
                    Q_ARG(QString, plotClass),
                    Q_ARG(QVariantMap, properties),
                    Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore()));
                if (!locker2.getSemaphore()->wait(PLUGINWAIT * 5))
                {
                    retVal += RetVal(
                        retError,
                        0,
                        tr("Timeout while plotting dataObject or numpy.array").toLatin1().data());
                    break;
                }
            }
        }
    }

    if (retVal.containsError())
    {
        const char* errorMsg = retVal.errorMessage();
        QString message = QString();
        if (errorMsg)
            message = errorMsg;
        QMessageBox::critical(
            this, tr("Plot data"), tr("Error while plotting value(s):\n%1").arg(message));
    }
    else if (retVal.containsWarning())
    {
        const char* errorMsg = retVal.errorMessage();
        QString message = QString();
        if (errorMsg)
            message = errorMsg;
        QMessageBox::warning(
            this, tr("Plot data"), tr("Warning while plotting value(s):\n%1").arg(message));
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
    if (m_pWorkspaceWidget != NULL)
    {
        int num = m_pWorkspaceWidget->numberOfSelectedItems();
        int numToBeRenamed = m_pWorkspaceWidget->numberOfSelectedItems(true);
        int i = 0;

        if (num > 0)
        {
            QList<QTreeWidgetItem*> items = m_pWorkspaceWidget->selectedItems();
            num = items.size(); // workaround: sometimes num != items.size() !!!

            m_firstCurrentItem = NULL;
            while (m_firstCurrentItem == NULL && i < items.count())
            {
                m_firstCurrentItem = items[i++];
            }

            if (num > 1)
            {
                bool ok;
                bool plotDObjOk = true;
                bool plotCloudOk = true;
                int compatibleTypes;
                for (int i = 0; i < num; ++i)
                {
                    compatibleTypes =
                        items[i]->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt(&ok);
                    if (!ok)
                    {
                        compatibleTypes = 0;
                    }

                    if (compatibleTypes != ito::ParamBase::DObjPtr)
                    {
                        plotDObjOk = false;
                    }
                    if (compatibleTypes != ito::ParamBase::PointCloudPtr &&
                        compatibleTypes != ito::ParamBase::PolygonMeshPtr)
                    {
                        plotCloudOk = false;
                    }
                }

                m_separatorSpecialActionsToolBar->setVisible(plotDObjOk || plotCloudOk);
                m_dObjPlot1d->setVisible(plotDObjOk);
                m_dObjPlot2d->setVisible(plotDObjOk);
                m_dObjPlot25d->setVisible(plotDObjOk);
                m_dObjPlot3d->setVisible(plotCloudOk);
            }
            else
            {
                bool ok;

                int compatibleTypes =
                    items[0]->data(0, WorkspaceWidget::RoleCompatibleTypes).toInt(&ok);
                if (!ok)
                {
                    compatibleTypes = 0;
                }

                if (compatibleTypes == ito::ParamBase::DObjPtr)
                {
                    m_separatorSpecialActionsToolBar->setVisible(true);
                    m_dObjPlot1d->setVisible(true);
                    m_dObjPlot2d->setVisible(true);
                    m_dObjPlot25d->setVisible(true);
                    m_dObjPlot3d->setVisible(false);
                    m_separatorDisplayItemDetailsActionsToolBar->setVisible(true);
                    m_pWorkspaceWidget->m_displayItemDetails->setVisible(true);
                }
                else if (
                    compatibleTypes == ito::ParamBase::PointCloudPtr ||
                    compatibleTypes == ito::ParamBase::PolygonMeshPtr)
                {
                    m_separatorSpecialActionsToolBar->setVisible(true);
                    m_dObjPlot1d->setVisible(false);
                    m_dObjPlot2d->setVisible(false);
                    m_dObjPlot25d->setVisible(false);
                    m_dObjPlot3d->setVisible(true);
                    m_separatorDisplayItemDetailsActionsToolBar->setVisible(true);
                    m_pWorkspaceWidget->m_displayItemDetails->setVisible(true);
                }
                else
                {
                    m_separatorSpecialActionsToolBar->setVisible(false);
                    m_dObjPlot1d->setVisible(false);
                    m_dObjPlot2d->setVisible(false);
                    m_dObjPlot25d->setVisible(false);
                    m_dObjPlot3d->setVisible(false);
                    m_separatorDisplayItemDetailsActionsToolBar->setVisible(true);
                    m_pWorkspaceWidget->m_displayItemDetails->setVisible(true);
                }
            }
        }
        else
        {
            m_firstCurrentItem = NULL;
            m_separatorSpecialActionsToolBar->setVisible(false);
            m_dObjPlot1d->setVisible(false);
            m_dObjPlot2d->setVisible(false);
            m_dObjPlot25d->setVisible(false);
            m_dObjPlot3d->setVisible(false);
            m_separatorDisplayItemDetailsActionsToolBar->setVisible(false);
            m_pWorkspaceWidget->m_displayItemDetails->setVisible(false);
        }
        m_separatorSpecialActionsContextMenu->setVisible(
            m_separatorSpecialActionsToolBar->isVisible());

        if (m_globalNotLocal)
        {
            bool pythonFree = (pythonBusy() == false || pythonInWaitingMode());
            m_actDelete->setEnabled(num > 0 && pythonFree);
            m_actExport->setEnabled(num > 0 && pythonFree);
            m_actImport->setEnabled(pythonFree);
            m_actRename->setEnabled(numToBeRenamed == 1 && pythonFree);
            m_actClearAll->setEnabled(pythonFree);
        }
        else
        {
            m_actDelete->setEnabled(num > 0 && pythonInWaitingMode());
            m_actExport->setEnabled(num > 0 && pythonInWaitingMode());
            m_actImport->setEnabled(pythonInWaitingMode());
            m_actRename->setEnabled(numToBeRenamed == 1 && pythonInWaitingMode());
            m_pWorkspaceWidget->setEnabled(pythonInWaitingMode());
            m_actClearAll->setEnabled(pythonInWaitingMode());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if name of element in workspaceWidget (TreeView) has been changed.
/*!
    calls pythonEngine's method renameVariable in order to initiate the renaming operation in
   python.

    \param item [in] is the corresponding QTreeWidgetItem, whose name has manually been changed
    \sa renameVariable
*/
void WorkspaceDockWidget::treeWidgetItemChanged(QTreeWidgetItem* item, int /*column*/)
{
    QString newKey = item->data(0, Qt::DisplayRole).toString();

    if (newKey != m_firstCurrentItemKey && m_firstCurrentItemKey != "" && newKey != "")
    {
        PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (eng)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));

            emit setStatusInformation(tr("renaming variable"), 0);

            QMetaObject::invokeMethod(
                eng,
                "renameVariable",
                Q_ARG(bool, m_globalNotLocal),
                Q_ARG(
                    QString, m_firstCurrentItem->data(0, WorkspaceWidget::RoleFullName).toString()),
                Q_ARG(QString, newKey),
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

            m_firstCurrentItemKey = QString();

            if (locker.getSemaphore()->waitAndProcessEvents(PLUGINWAIT))
            {
                emit setStatusInformation("", 0);
            }
            else
            {
                emit setStatusInformation(tr("Timeout while renaming variables"), PLUGINWAIT);
            }
            QApplication::restoreOverrideCursor();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::dragEnterEvent(QDragEnterEvent* event)
{
    if (m_globalNotLocal == false && !pythonInWaitingMode())
    {
        // local workspace is only active if python is in waiting mode
        return;
    }

    if (event->mimeData()->hasFormat("text/uri-list")) // or hasUrls() should be the same result
    {
        QList<QUrl> urls = event->mimeData()->urls();

        QStringList allPatterns;
        IOHelper::getFileFilters(
            IOHelper::IOFilters(
                IOHelper::IOPlugin | IOHelper::IOInput | IOHelper::IOWorkspace |
                IOHelper::IOMimeAll),
            &allPatterns);
        QRegularExpression reg("", QRegularExpression::CaseInsensitiveOption);
        bool ok = false;

        // check files
        foreach (const QUrl& url, urls)
        {
            if (url.isLocalFile() == false)
            {
                return;
            }

            foreach (const QString& pat, allPatterns)
            {
                reg.setPattern(CompatHelper::regExpAnchoredPattern(CompatHelper::wildcardToRegularExpression(pat)));

                if (url.toLocalFile().indexOf(reg) >= 0)
                {
                    ok = true;
                    break;
                }
            }

            if (!ok)
            {
                return;
            }
        }

        event->acceptProposedAction();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::dropEvent(QDropEvent* event)
{
    QList<QUrl> urls = event->mimeData()->urls();
    QString localFile;

    // check files
    foreach (const QUrl& url, urls)
    {
        localFile = url.toLocalFile();
        IOHelper::openGeneralFile(localFile, false, true, this, 0, m_globalNotLocal);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::checkToggleUnpack()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Workspace");
    m_actUnpack->setChecked(settings.value("importIdcMatUnpackDict", "true").toBool());
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WorkspaceDockWidget::propertiesChanged()
{
    checkToggleUnpack();
}

} // end namespace ito
