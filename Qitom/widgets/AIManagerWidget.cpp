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

#include "AIManagerWidget.h"

#include "../../AddInManager/addInManager.h"
#include "../ui/dialogNewPluginInstance.h"
#include "../ui/dialogOpenNewGui.h"
#include "../ui/dialogSnapshot.h"

#include "../AppManagement.h"
#include "../global.h"

#include "../../AddInManager/pluginModel.h"
#include <qabstractitemmodel.h>
#include <qaction.h>
#include <qdockwidget.h>
#include <qinputdialog.h>
#include <qmessagebox.h>

namespace ito {

//-------------------------------------------------------------------------------------
AIManagerWidget::AIManagerWidget(
    const QString& title,
    const QString& objName,
    QWidget* parent,
    bool docked,
    bool isDockAvailable,
    tFloatingStyle floatingStyle,
    tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_pContextMenu(nullptr), m_pShowConfDialog(nullptr), m_pActDockWidget(nullptr),
    m_pActDockWidgetToolbar(nullptr), m_pActNewInstance(nullptr), m_pActCloseInstance(nullptr),
    m_pActCloseAllInstances(nullptr), m_pActSendToPython(nullptr), m_pActLiveImage(nullptr),
    m_pActSnapDialog(nullptr), m_pActAutoGrabbing(nullptr), m_pActInfo(nullptr), m_pActOpenWidget(nullptr),
    m_pAIManagerView(nullptr), m_pSortFilterProxyModel(nullptr), m_pColumnWidth(nullptr),
    m_pMainToolbar(nullptr), m_pViewList(nullptr), m_pViewDetails(nullptr), m_pPlugInModel(nullptr)
{
    int size = 0;
    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    m_pAIManagerView = new QTreeView(this);
    m_pAIManagerView->setContextMenuPolicy(Qt::CustomContextMenu);
    m_pAIManagerView->setSortingEnabled(true);
    m_pAIManagerView->setDragEnabled(true);
    connect(
        m_pAIManagerView,
        &QTreeView::customContextMenuRequested,
        this,
        &AIManagerWidget::treeViewContextMenuRequested);

    m_pContextMenu = new QMenu(this);

    m_pActNewInstance =
        new QAction(QIcon(":/plugins/icons/pluginNewInstance.png"), tr("New Instance..."), this);
    connect(m_pActNewInstance, SIGNAL(triggered()), this, SLOT(mnuCreateNewInstance()));
    m_pContextMenu->addAction(m_pActNewInstance);

    m_pShowConfDialog =
        new QAction(QIcon(":/plugins/icons/pluginConfigure.png"), tr("Configuration Dialog"), this);
    connect(m_pShowConfDialog, SIGNAL(triggered()), this, SLOT(mnuShowConfdialog()));
    m_pContextMenu->addAction(m_pShowConfDialog);

    m_pActDockWidget = new QAction(QIcon(":/plugins/icons/pluginToolbox.png"), "", this);
    m_pActDockWidget->setCheckable(true);
    connect(m_pActDockWidget, SIGNAL(triggered()), this, SLOT(mnuToggleDockWidget()));
    m_pContextMenu->addAction(m_pActDockWidget);

    m_pActDockWidgetToolbar = new QAction(
        QIcon(":/plugins/icons/pluginToolbox.png"), tr("Show/Hide Plugin Toolbox"), this);
    connect(m_pActDockWidgetToolbar, SIGNAL(triggered()), this, SLOT(mnuToggleDockWidget()));

    m_pActCloseInstance =
        new QAction(QIcon(":/plugins/icons/pluginCloseInstance.png"), tr("Close Instance"), this);
    connect(m_pActCloseInstance, SIGNAL(triggered()), this, SLOT(mnuCloseInstance()));
    m_pContextMenu->addAction(m_pActCloseInstance);

    m_pActCloseAllInstances =
        new QAction(QIcon(":/plugins/icons/closeAll.png"), tr("Close All"), this);
    connect(m_pActCloseAllInstances, SIGNAL(triggered()), this, SLOT(mnuCloseAllInstances()));
    m_pContextMenu->addAction(m_pActCloseAllInstances);

    m_pContextMenu->addSeparator();

    m_pActLiveImage = new QAction(QIcon(":/plugins/icons/monitor.png"), tr("Live Image..."), this);
    connect(m_pActLiveImage, SIGNAL(triggered()), this, SLOT(mnuShowLiveImage()));
    m_pContextMenu->addAction(m_pActLiveImage);

    m_pActSnapDialog =
        new QAction(QIcon(":/measurement/icons/itom_icons/snap.png"), tr("Snap Dialog..."), this);
    connect(m_pActSnapDialog, SIGNAL(triggered()), this, SLOT(mnuSnapDialog()));
    m_pContextMenu->addAction(m_pActSnapDialog);

    m_pActAutoGrabbing = new QAction(QIcon(":/misc/icons/shell.png"), tr("Auto Grabbing"), this);
    m_pActAutoGrabbing->setCheckable(true);
    connect(m_pActAutoGrabbing, SIGNAL(triggered()), this, SLOT(mnuToggleAutoGrabbing()));
    m_pContextMenu->addAction(m_pActAutoGrabbing);

    m_pActOpenWidget = new QAction(QIcon(":/plugins/icons/window.png"), tr("Open Widget..."), this);
    connect(m_pActOpenWidget, SIGNAL(triggered()), this, SLOT(mnuOpenWidget()));
    m_pContextMenu->addAction(m_pActOpenWidget);

    m_pContextMenu->addSeparator();

    m_pActInfo = new QAction(QIcon(":/plugins/icons/info.png"), tr("Info..."), this);
    connect(m_pActInfo, SIGNAL(triggered()), this, SLOT(mnuShowInfo()));
    m_pContextMenu->addAction(m_pActInfo);

    m_pActSendToPython =
        new QAction(QIcon(":/plugins/icons/sendToPython.png"), tr("Send To Python..."), this);
    connect(m_pActSendToPython, SIGNAL(triggered()), this, SLOT(mnuSendToPython()));
    m_pContextMenu->addAction(m_pActSendToPython);

    m_pSortFilterProxyModel = new QSortFilterProxyModel(this);

    if (aim)
    {
        m_pPlugInModel = aim->getPluginModel();

        m_pSortFilterProxyModel->setSourceModel(m_pPlugInModel);
        m_pAIManagerView->setModel(m_pSortFilterProxyModel);
        m_pAIManagerView->sortByColumn(0, Qt::AscendingOrder);
        connect(
            m_pAIManagerView->selectionModel(),
            SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),
            this,
            SLOT(selectionChanged(const QItemSelection&, const QItemSelection&)),
            Qt::DirectConnection);

        // expanding DataIO node
        QModelIndex index = m_pPlugInModel->getTypeNode(typeDataIO);
        if (index.isValid() && m_pSortFilterProxyModel)
        {
            index = m_pSortFilterProxyModel->mapFromSource(index);
            m_pAIManagerView->expand(index);
        }

        QSettings* settings = new QSettings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings->beginGroup("itomPluginsDockWidget");
        size = settings->beginReadArray("ColWidth");
        for (int i = 0; i < size; ++i)
        {
            settings->setArrayIndex(i);
            m_pAIManagerView->setColumnWidth(i, settings->value("width", 100).toInt());
            m_pAIManagerView->setColumnHidden(i, m_pAIManagerView->columnWidth(i) == 0);
        }
        settings->endArray();

        m_pColumnWidth = new int[m_pPlugInModel->columnCount()];
        size = settings->beginReadArray("StandardColWidth");

        if (size != m_pPlugInModel->columnCount())
        {
            m_pColumnWidth[0] = 200;

            for (int i = 1; i < m_pPlugInModel->columnCount(); ++i)
            {
                m_pColumnWidth[i] = 120;
            }
        }

        for (int i = 0; i < size; ++i)
        {
            settings->setArrayIndex(i);
            m_pColumnWidth[i] = settings->value("width", 100).toInt();

            if (m_pColumnWidth[i] == 0)
            {
                m_pColumnWidth[i] = 120;
            }
        }

        settings->endArray();
        settings->endGroup();
        delete settings;
    }

    AbstractDockWidget::init();
    setContentWidget(m_pAIManagerView);

    updateActions();
}

//-------------------------------------------------------------------------------------
AIManagerWidget::~AIManagerWidget()
{
    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    if (aim)
    {
        PlugInModel* plugInModel = (PlugInModel*)(aim->getPluginModel());
        QString setFile(AppManagement::getSettingsFile());
        QSettings* settings = new QSettings(setFile, QSettings::IniFormat);

        settings->beginGroup("itomPluginsDockWidget");

        //    QByteArray state = m_pMainToolbar->saveGeometry();
        //    settings->setValue("stateToolBar", state);

        settings->beginWriteArray("ColWidth");

        for (int i = 0; i < plugInModel->columnCount(); i++)
        {
            settings->setArrayIndex(i);
            settings->setValue("width", m_pAIManagerView->columnWidth(i));
        }

        settings->endArray();
        settings->sync();

        settings->beginWriteArray("StandardColWidth");

        for (int i = 0; i < plugInModel->columnCount(); i++)
        {
            settings->setArrayIndex(i);
            settings->setValue("width", m_pColumnWidth[i]);
        }

        settings->endArray();
        settings->endGroup();
        settings->sync();
        delete settings;
    }

    disconnect(
        m_pAIManagerView,
        SIGNAL(customContextMenuRequested(const QPoint&)),
        this,
        SLOT(treeViewContextMenuRequested(const QPoint&)));
    disconnect(
        m_pAIManagerView->selectionModel(),
        SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)),
        this,
        SLOT(selectionChanged(const QItemSelection&, const QItemSelection&)));

    DELETE_AND_SET_NULL(m_pSortFilterProxyModel);
    DELETE_AND_SET_NULL(m_pAIManagerView);
    DELETE_AND_SET_NULL(m_pContextMenu);
    DELETE_AND_SET_NULL(m_pShowConfDialog);
    DELETE_AND_SET_NULL(m_pActNewInstance);
    DELETE_AND_SET_NULL(m_pActDockWidget);
    DELETE_AND_SET_NULL(m_pActDockWidgetToolbar);
    DELETE_AND_SET_NULL(m_pActCloseInstance);
    DELETE_AND_SET_NULL(m_pActCloseAllInstances);
    DELETE_AND_SET_NULL(m_pActSendToPython);
    DELETE_AND_SET_NULL(m_pActLiveImage);
    DELETE_AND_SET_NULL(m_pActSnapDialog);
    DELETE_AND_SET_NULL(m_pActAutoGrabbing);
    DELETE_AND_SET_NULL(m_pActInfo);
    DELETE_AND_SET_NULL(m_pActOpenWidget);
    DELETE_AND_SET_NULL(m_pAIManagerViewSettingMenu);
    DELETE_AND_SET_NULL_ARRAY(m_pColumnWidth);
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::createActions()
{
    m_pViewList = new ShortcutAction(QIcon(":/application/icons/kdb_form.png"), tr("List"), this);
    m_pViewList->connectTrigger(this, SLOT(showList()));
    m_pViewDetails = new ShortcutAction(QIcon(":/application/icons/list.png"), tr("Details"), this);
    m_pViewDetails->connectTrigger(this, SLOT(showDetails()));
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::createMenus()
{
    m_pAIManagerViewSettingMenu = new QMenu(tr("Settings"), this);
    m_pAIManagerViewSettingMenu->setIcon(QIcon(":/application/icons/adBlockAction.png"));
    m_pAIManagerViewSettingMenu->addAction(m_pViewList->action());
    m_pAIManagerViewSettingMenu->addAction(m_pViewDetails->action());
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::createToolBars()
{
    QWidget* spacerWidget = new QWidget();
    QHBoxLayout* spacerLayout = new QHBoxLayout();
    spacerLayout->addItem(new QSpacerItem(5, 5, QSizePolicy::Expanding, QSizePolicy::Minimum));
    spacerLayout->setStretch(0, 2);
    spacerWidget->setLayout(spacerLayout);

    m_pMainToolbar = new QToolBar(tr("plugins"), this);
    m_pMainToolbar->setObjectName("toolbarPlugins");
    m_pMainToolbar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolbar->setFloatable(false);
    addToolBar(m_pMainToolbar, "mainToolBar");

    m_pMainToolbar->addAction(m_pActNewInstance);
    m_pMainToolbar->addAction(m_pShowConfDialog);
    m_pMainToolbar->addAction(m_pActDockWidgetToolbar);
    m_pMainToolbar->addAction(m_pActCloseInstance);
    m_pMainToolbar->addAction(m_pActCloseAllInstances);
    m_pMainToolbar->addAction(m_pActOpenWidget);
    m_pMainToolbarSeparator1 = m_pMainToolbar->addSeparator();
    m_pMainToolbar->addAction(m_pActLiveImage);
    m_pMainToolbar->addAction(m_pActSnapDialog);
    m_pMainToolbar->addAction(m_pActAutoGrabbing);
    m_pMainToolbarSeparator2 = m_pMainToolbar->addSeparator();
    m_pMainToolbar->addAction(m_pActInfo);
    m_pMainToolbar->addAction(m_pActSendToPython);
    m_pMainToolbar->addWidget(spacerWidget);
    m_pMainToolbar->addAction(m_pAIManagerViewSettingMenu->menuAction());
    connect(
        m_pAIManagerViewSettingMenu->menuAction(),
        SIGNAL(triggered()),
        this,
        SLOT(mnuToggleView()));
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::updateActions()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        PlugInModel::tItemType itemType;
        size_t itemInternalData;
        PlugInModel* plugInModel = (PlugInModel*)(index.model());

        if (plugInModel->flags(index) & Qt::ItemIsEnabled)
        {
            if (plugInModel->getModelIndexInfo(index, itemType, itemInternalData))
            {
                //            bool isFixedNode = itemType & PlugInModel::itemCatAll; // ==
                //            PlugInModel::itemCatActuator || itemType == PlugInModel::itemCatAlgo
                //            || itemType == PlugInModel::itemCatDataIO || itemType ==
                //            PlugInModel::itemSubCategoryDataIO_Grabber;
                bool isPlugInNode = (itemType == PlugInModel::itemPlugin);
                bool isPlugInAlgoNode = plugInModel->getIsAlgoPlugIn(itemType, itemInternalData);
                bool isInstanceNode = (itemType == PlugInModel::itemInstance);
                bool isPlugInGrabberNode =
                    plugInModel->getIsGrabberInstance(itemType, itemInternalData);
                bool isFilterNode = (itemType == PlugInModel::itemFilter);
                bool isWidgetNode = (itemType == PlugInModel::itemWidget);

                m_pActCloseAllInstances->setVisible(isPlugInNode && !isPlugInAlgoNode);
                m_pActCloseInstance->setVisible(isInstanceNode);
                m_pActDockWidget->setVisible(isInstanceNode);
                m_pActDockWidgetToolbar->setVisible(isInstanceNode);
                m_pActInfo->setVisible(isPlugInNode || isFilterNode || isWidgetNode);
                m_pActLiveImage->setVisible(isPlugInGrabberNode);
                m_pActAutoGrabbing->setVisible(isPlugInGrabberNode);
                m_pActNewInstance->setVisible(isPlugInNode && !isPlugInAlgoNode);
                m_pActOpenWidget->setVisible(isWidgetNode);
                m_pActSendToPython->setVisible(isInstanceNode);
                m_pActSnapDialog->setVisible(isPlugInGrabberNode);
                m_pShowConfDialog->setVisible(isInstanceNode);

                if (isInstanceNode)
                {
                    ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();

                    m_pActCloseInstance->setEnabled(ais->createdByGUI() > 0);

                    QObject* engine = AppManagement::getPythonEngine();
                    m_pActSendToPython->setEnabled(engine);

                    m_pShowConfDialog->setEnabled(
                        (qobject_cast<QApplication*>(QCoreApplication::instance())) &&
                        ais->hasConfDialog());
                    m_pActDockWidget->setEnabled(ais->hasDockWidget());
                    m_pActDockWidgetToolbar->setEnabled(ais->hasDockWidget());

                    if (m_pActDockWidget->isEnabled())
                    {
                        if (ais->getDockWidget() &&
                            ais->getDockWidget()->toggleViewAction()->isChecked())
                        {
                            m_pActDockWidget->setText(tr("Hide Plugin Toolbox"));
                            m_pActDockWidget->setChecked(true);
                        }
                        else
                        {
                            m_pActDockWidget->setText(tr("Show Plugin Toolbox"));
                            m_pActDockWidget->setChecked(false);
                        }
                    }
                    else
                    {
                        m_pActDockWidget->setText(tr("Plugin Toolbox"));
                        m_pActDockWidget->setChecked(false);
                    }

                    if (m_pActAutoGrabbing->isVisible())
                    {
                        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();
                        if (ais)
                        {
                            m_pActAutoGrabbing->setChecked(
                                ((ito::AddInDataIO*)ais)->getAutoGrabbing());
                        }
                    }
                }
                else if (m_pActCloseAllInstances->isVisible())
                {
                    QModelIndex indexChild = plugInModel->index(0, 0, index);
                    m_pActCloseAllInstances->setEnabled(indexChild.isValid());
                }

                m_pActInfo->setEnabled(true);
            }
        }
    }
    else
    {
        m_pActCloseAllInstances->setVisible(false);
        m_pActCloseInstance->setVisible(false);
        m_pActDockWidget->setVisible(false);
        m_pActDockWidgetToolbar->setVisible(false);
        m_pActInfo->setVisible(false);
        m_pActLiveImage->setVisible(false);
        m_pActAutoGrabbing->setVisible(false);
        m_pActNewInstance->setVisible(false);
        m_pActOpenWidget->setVisible(false);
        m_pActSendToPython->setVisible(false);
        m_pActSnapDialog->setVisible(false);
        m_pShowConfDialog->setVisible(false);
    }

    m_pMainToolbarSeparator1->setVisible(
        m_pActLiveImage->isVisible() || m_pActSnapDialog->isVisible() ||
        m_pActAutoGrabbing->isVisible());
    m_pMainToolbarSeparator2->setVisible(
        (m_pActInfo->isVisible() || m_pActSendToPython->isVisible()) &&
        (m_pMainToolbarSeparator1->isVisible() || m_pActCloseAllInstances->isVisible() ||
         m_pActNewInstance->isVisible() || m_pShowConfDialog->isVisible() ||
         m_pActDockWidget->isVisible() || m_pActDockWidgetToolbar->isVisible() ||
         m_pActCloseInstance->isVisible() || m_pActOpenWidget->isVisible()));
}

//-------------------------------------------------------------------------------------
QColor AIManagerWidget::backgroundColorInstancesWithPythonRef() const
{
    if (m_pPlugInModel)
    {
        return m_pPlugInModel->backgroundColorInstancesWithPythonRef();
    }

    return QColor();
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::setBackgroundColorInstancesWithPythonRef(const QColor& bgColor)
{
    if (m_pPlugInModel)
    {
        m_pPlugInModel->setBackgroundColorInstancesWithPythonRef(bgColor);
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::selectionChanged(
    const QItemSelection& newSelection, const QItemSelection& oldSelection)
{
    updateActions();
}

//-------------------------------------------------------------------------------------
/*QList<QAction*> AIManagerWidget::getAlgoWidgetActions(const ito::AddInInterfaceBase *aib)
{
    QList<QAction*> actions;
    QAction* action = nullptr;
    QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> awList;
//    if (aib->getType() != ito::typeAlgo) return actions;

    QList<ito::AddInBase*> instList = aib->getInstList();
    if (instList.size() == 0) return actions;

    ito::AddInBase* ab = instList[0];
    ito::AddInAlgo *aia = static_cast<ito::AddInAlgo *>(ab);

    if (aia == nullptr) return actions;

    aia->getAlgoWidgetList(awList);

    foreach (const QString &key, awList.keys())
    {
        action = new QAction(QIcon(":/plugins/icons/window.png"), key, this);
        action->setData(key);
        actions.append(action);
    }

    return actions;
}*/

//-------------------------------------------------------------------------------------
void AIManagerWidget::closeInstance(const QModelIndex index)
{
    ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();
    if (ais)
    {
        if (ais->createdByGUI() == 0)
        {
            QMessageBox::warning(
                this,
                tr("Closing not possible"),
                tr("The instance '%1' cannot be closed by GUI since it has been created by Python")
                    .arg(index.model()->data(index).toString()));
        }
        else if (ais->getRefCount() > 1)
        {
            QMessageBox::warning(
                this,
                tr("Closing not possible"),
                tr("The instance '%1' can temporarily not be closed since it is still in use by "
                   "another element.")
                    .arg(index.model()->data(index).toString()));
        }
        else
        {
            // it may be that an instance has been created by gui and then a reference has been
            // created in python. If we now close the instance in the GUI, python still holds it,
            // therefore the createdByGUI-flag must be false after that the instance is closed by
            // the GUI-side
            ais->setCreatedByGUI(false);

            if (ais->getRefCount() > 0)
            {
                QMessageBox::information(
                    this,
                    tr("final closing not possible"),
                    tr("The instance '%1' can finally not be closed since there are still "
                       "references to this instance from other componentents, e.g. python "
                       "variables.")
                        .arg(index.model()->data(index).toString()));
            }

            ito::AddInManager* aim =
                qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
            ito::RetVal retValue = aim->closeAddIn(ais, nullptr);

            if (retValue.containsWarning())
            {
                QString message = tr("Warning while closing instance. Message: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::warning(this, tr("Warning while closing instance"), message);
            }
            else if (retValue.containsError())
            {
                QString message = tr("Error while closing instance. Message: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::critical(this, tr("Error while closing instance"), message);
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::treeViewContextMenuRequested(const QPoint& pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_pAIManagerView->mapToGlobal(m_pAIManagerView->pos()));
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuShowConfdialog()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();
        if ((qobject_cast<QApplication*>(QCoreApplication::instance())) && ais &&
            ais->hasConfDialog())
        {
            ito::RetVal retValue = ais->showConfDialog();

            if (retValue.containsWarning())
            {
                QString message = tr("Warning while showing configuration dialog. Message: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::warning(
                    this, tr("Warning while showing configuration dialog"), message);
            }
            else if (retValue.containsError())
            {
                QString message = tr("Error while showing configuration dialog. Message: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::critical(
                    this, tr("Error while showing configuration dialog"), message);
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuToggleDockWidget()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();
        if (ais)
        {
            QDockWidget* dockWidget = ais->getDockWidget();
            if (dockWidget)
            {
                dockWidget->toggleViewAction()->trigger();
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuCreateNewInstance()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
        ito::AddInInterfaceBase* aib = (ito::AddInInterfaceBase*)index.internalPointer();

        DialogNewPluginInstance* dialog = new DialogNewPluginInstance(index, aib);
        if (dialog->exec() == 1) // accepted
        {
            QVector<ito::ParamBase> paramsMandNew, paramsOptNew;
            QString pythonVarName = dialog->getPythonVariable();
            ito::RetVal retValue = ito::retOk;
            ito::AddInBase* basePlugin = nullptr;

            retValue += dialog->getFilledMandParams(paramsMandNew);
            retValue += dialog->getFilledOptParams(paramsOptNew);

            DELETE_AND_SET_NULL(dialog);

            if (retValue.containsError())
            {
                QString message = tr("Error while creating new instance. \nMessage: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::critical(this, tr("Error while creating new instance"), message);
                return;
            }

            int itemNum = aim->getItemIndexInList((void*)aib);

            if (itemNum < 0)
            {
                return;
            }

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));

            // the WaitCursor only becomes visible if the
            // event loop of the main thread is called once.
            //(it is not allowed to filter
            // QEventLoop::ExcludeUserInputEvents here out,
            // since mouse events have to be passed to the
            // operating system. Else the cursor is not
            // changed. - at least with Windows)
            QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers);

            if (aib->getType() & ito::typeDataIO)
            {
                ito::AddInDataIO* plugin = nullptr;
                retValue += aim->initAddIn(
                    itemNum,
                    aib->objectName(),
                    &plugin,
                    &paramsMandNew,
                    &paramsOptNew,
                    false,
                    nullptr);
                basePlugin = (ito::AddInBase*)(plugin);
            }
            else if (aib->getType() & ito::typeActuator)
            {
                ito::AddInActuator* plugin = nullptr;
                retValue += aim->initAddIn(
                    itemNum,
                    aib->objectName(),
                    &plugin,
                    &paramsMandNew,
                    &paramsOptNew,
                    false,
                    nullptr);
                basePlugin = (ito::AddInBase*)(plugin);
            }

            QApplication::restoreOverrideCursor();

            if (retValue.containsWarning())
            {
                QString message = tr("Warning while creating new instance. Message: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::warning(this, tr("Warning while creating new instance"), message);
            }
            else if (retValue.containsError())
            {
                QString message = tr("Error while creating new instance. Message: %1")
                                      .arg(QLatin1String(retValue.errorMessage()));
                QMessageBox::critical(this, tr("Error while creating new instance"), message);
            }

            if (basePlugin != nullptr)
            {
                basePlugin->setCreatedByGUI(1);

                if (pythonVarName != "")
                {
                    QObject* engine = AppManagement::getPythonEngine();
                    if (engine)
                    {
                        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                        ito::RetVal retValue = retOk;

                        QMetaObject::invokeMethod(
                            engine,
                            "registerAddInInstance",
                            Q_ARG(QString, pythonVarName),
                            Q_ARG(ito::AddInBase*, basePlugin),
                            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
                        if (!locker.getSemaphore()->wait(PLUGINWAIT))
                        {
                            QMessageBox::warning(
                                this,
                                tr("Timeout"),
                                tr("Python did not response to the request within a certain "
                                   "timeout."));
                        }

                        retValue = locker.getSemaphore()->returnValue;

                        if (retValue.containsWarning())
                        {
                            QString message =
                                tr("Warning while sending instance to python. Message: %1")
                                    .arg(QLatin1String(retValue.errorMessage()));
                            QMessageBox::warning(
                                this, tr("Warning while sending instance to python"), message);
                        }
                        else if (retValue.containsError())
                        {
                            QString message =
                                tr("Error while sending instance to python. Message: %1")
                                    .arg(QLatin1String(retValue.errorMessage()));
                            QMessageBox::critical(
                                this, tr("Error while sending instance to python"), message);
                        }
                    }
                    else
                    {
                        QMessageBox::warning(
                            this,
                            tr("Python not available"),
                            tr("The Python engine is not available"));
                    }
                }
            }
            m_pAIManagerView->expand(m_pAIManagerView->currentIndex());
        }
        else
        {
            DELETE_AND_SET_NULL(dialog);
        }
    }
    updateActions();
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuCloseInstance()
{
    QModelIndex index = m_pAIManagerView->currentIndex();
    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        closeInstance(index);
    }
    updateActions();
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuCloseAllInstances()
{
    QModelIndex index = m_pAIManagerView->currentIndex();
    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        const QAbstractItemModel* model = index.model();

        for (int i = model->rowCount(index) - 1; i > -1; --i)
        {
            closeInstance(model->index(i, 0, index));
        }
    }
    updateActions();
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuSendToPython()
{
    QModelIndex index = m_pAIManagerView->currentIndex();
    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();
        if (ais)
        {
            QObject* engine = AppManagement::getPythonEngine();
            if (engine)
            {
                bool ok = false;
                QString varname = QInputDialog::getText(
                    this,
                    tr("Python variable name"),
                    tr("Python variable name for saving this instance in global workspace"),
                    QLineEdit::Normal,
                    tr("instance"),
                    &ok);
                if (ok && varname != "")
                {
                    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                    ito::RetVal retValue = retOk;

                    QMetaObject::invokeMethod(
                        engine,
                        "registerAddInInstance",
                        Q_ARG(QString, varname),
                        Q_ARG(ito::AddInBase*, ais),
                        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
                    if (!locker.getSemaphore()->wait(PLUGINWAIT))
                    {
                        QMessageBox::warning(
                            this,
                            tr("Timeout"),
                            tr("Python did not response to the request within a certain timeout."));
                    }

                    retValue = locker.getSemaphore()->returnValue;

                    if (retValue.containsWarning())
                    {
                        QString message =
                            tr("Warning while sending instance to python. Message: %1")
                                .arg(QLatin1String(retValue.errorMessage()));
                        QMessageBox::warning(
                            this, tr("Warning while sending instance to python"), message);
                    }
                    else if (retValue.containsError())
                    {
                        QString message = tr("Error while sending instance to python. Message: %1")
                                              .arg(QLatin1String(retValue.errorMessage()));
                        QMessageBox::critical(
                            this, tr("Error while sending instance to python"), message);
                    }
                }
            }
            else
            {
                QMessageBox::warning(
                    this, tr("Python not available"), tr("The Python engine is not available"));
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuOpenWidget()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInAlgo::AlgoWidgetDef* awd =
            (ito::AddInAlgo::AlgoWidgetDef*)index.internalPointer();
        mnuShowAlgoWidget(awd);
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuShowAlgoWidget(ito::AddInAlgo::AlgoWidgetDef* awd)
{
    QMessageBox msgBox;

    ito::RetVal retValue(retOk);

    QVector<ito::Param> paramsMand;
    QVector<ito::Param> paramsOpt;
    QVector<ito::Param> paramsOut;
    QVector<ito::ParamBase> paramsMandBase;
    QVector<ito::ParamBase> paramsOptBase;

    retValue += awd->m_paramFunc(&paramsMand, &paramsOpt, &paramsOut);

    if (!retValue.containsError())
    {
        if (paramsMand.size() > 0 || paramsOpt.size() > 0)
        {
            DialogOpenNewGui* dialog = new DialogOpenNewGui(awd->m_name, paramsMand, paramsOpt);

            if (dialog->exec() == 1) // accepted
            {
                // QString pythonVarName = dialog->getPythonVariable();
                ito::RetVal retValue = ito::retOk;
                ito::AddInBase* basePlugin = nullptr;

                retValue += dialog->getFilledMandParams(paramsMandBase);
                retValue += dialog->getFilledOptParams(paramsOptBase);

                DELETE_AND_SET_NULL(dialog);
            }
            else
            {
                DELETE_AND_SET_NULL(dialog);

                return;
            }
        }

        UiOrganizer* uiOrganizer = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
        QSharedPointer<unsigned int> dialogHandle(new unsigned int);
        QSharedPointer<int> retCodeIfModal(new int);
        QSharedPointer<unsigned int> objectID(new unsigned int);
        QSharedPointer<QByteArray> className(new QByteArray());
        *objectID = 0;
        *dialogHandle = 0;

        int winType = 0xff;
        bool deleteOnClose = false;
        bool childOfMainWindow = false;
        Qt::DockWidgetArea dockWidgetArea = Qt::TopDockWidgetArea;
        int buttonBarType = UserUiDialog::bbTypeNo;
        StringMap dialogButtons;
        int uiDescription = UiOrganizer::createUiDescription(
            winType, buttonBarType, childOfMainWindow, deleteOnClose, dockWidgetArea);

        if (uiOrganizer)
        {
            retValue += uiOrganizer->loadPluginWidget(
                reinterpret_cast<void*>(awd->m_widgetFunc),
                uiDescription,
                dialogButtons,
                &paramsMandBase,
                &paramsOptBase,
                dialogHandle,
                objectID,
                className,
                nullptr);
            if (!retValue.containsError())
            {
                if (*dialogHandle > 0)
                {
                    retValue += uiOrganizer->setAttribute(
                        *dialogHandle,
                        Qt::WA_DeleteOnClose,
                        true,
                        nullptr); // forces the dialog to delete itself if the user closes the dialog,
                               // since no other instance/itom-component is holding a reference to
                               // it.
                    retValue += uiOrganizer->showDialog(*dialogHandle, false, retCodeIfModal, nullptr);
                }
                else
                {
                    retValue += ito::RetVal(
                        ito::retError,
                        0,
                        tr("User interface of plugin could not be created. Returned handle is "
                           "invalid.")
                            .toLatin1()
                            .data());
                }
            }
        }
        else
        {
            retValue += ito::RetVal(
                ito::retError, 0, tr("Could not find instance of UiOrganizer").toLatin1().data());
        }
    }

    if (retValue.containsError())
    {
        msgBox.setText(tr("Error while opening user interface from plugin."));

        if (retValue.hasErrorMessage())
        {
            msgBox.setDetailedText(QLatin1String(retValue.errorMessage()));
        }

        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retValue.containsWarning())
    {
        msgBox.setText(tr("Warning while opening user interface from plugin."));

        if (retValue.hasErrorMessage())
        {
            msgBox.setDetailedText(QLatin1String(retValue.errorMessage()));
        }

        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuShowLiveImage()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();

        if (ais && ais->inherits("ito::AddInGrabber"))
        {
            UiOrganizer* uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
            QString defaultPlotClassName;
            QSharedPointer<unsigned int> objectID(new unsigned int);
            QSharedPointer<unsigned int> figHandle(new unsigned int);
            *figHandle = 0; // new figure will be requested

            ito::RetVal retval = uiOrg->figureLiveImage(
                (ito::AddInDataIO*)ais,
                figHandle,
                objectID,
                0,
                0,
                defaultPlotClassName,
                QVariantMap(),
                nullptr);

            if (retval.containsError())
            {
                QMessageBox msgBox;
                msgBox.setText(QLatin1String(retval.errorMessage()));
                msgBox.setIcon(QMessageBox::Critical);
                msgBox.exec();
            }
            else if (retval.containsWarning())
            {
                QMessageBox msgBox;
                msgBox.setText(QLatin1String(retval.errorMessage()));
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.exec();
            }
        }
        else
        {
            QMessageBox msgBox;
            msgBox.setText(
                tr("This instance is no grabber. Therefore no live image is available."));
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuSnapDialog()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();

        if (ais && ais->inherits("ito::AddInGrabber"))
        {
            ito::RetVal retval = ito::retOk;
            QPointer<ito::AddInDataIO> aisPointer((ito::AddInDataIO*)ais);
            DialogSnapshot* snapDialog = new DialogSnapshot(this, aisPointer, retval);
            snapDialog->setAttribute(Qt::WA_DeleteOnClose, true);
            snapDialog->show();

            if (retval.containsError())
            {
                QMessageBox msgBox;
                msgBox.setText(QLatin1String(retval.errorMessage()));
                msgBox.setIcon(QMessageBox::Critical);
                msgBox.exec();
            }
            else if (retval.containsWarning())
            {
                QMessageBox msgBox;
                msgBox.setText(QLatin1String(retval.errorMessage()));
                msgBox.setIcon(QMessageBox::Warning);
                msgBox.exec();
            }
        }
        else
        {
            QMessageBox msgBox;
            msgBox.setText(
                tr("This instance is no grabber. Therefore no snap dialog is available."));
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuToggleAutoGrabbing()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        ito::AddInBase* ais = (ito::AddInBase*)index.internalPointer();

        if (ais)
        {
            ItomSharedSemaphore* waitCond = nullptr;
            waitCond = new ItomSharedSemaphore();
            if (m_pActAutoGrabbing->isChecked())
            {
                QMetaObject::invokeMethod(
                    ais, "enableAutoGrabbing", Q_ARG(ItomSharedSemaphore*, waitCond));
            }
            else
            {
                QMetaObject::invokeMethod(
                    ais, "disableAutoGrabbing", Q_ARG(ItomSharedSemaphore*, waitCond));
            }
            waitCond->deleteSemaphore();
            waitCond = nullptr;
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::setTreeViewHideColumns(const bool& hide, const int colCount)
{
    for (int i = 1; i < colCount; ++i)
    {
        m_pAIManagerView->setColumnHidden(i, hide);
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::showList()
{
    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    PlugInModel* plugInModel = (PlugInModel*)(aim->getPluginModel());
    bool isList = true;

    for (int i = 1; i < plugInModel->columnCount(); ++i)
    {
        isList = isList && m_pAIManagerView->isColumnHidden(i);
    }

    if (!isList)
    {
        for (int i = 0; i < plugInModel->columnCount(); ++i)
        {
            m_pColumnWidth[i] = m_pAIManagerView->columnWidth(i);
        }
    }

    setTreeViewHideColumns(true, plugInModel->columnCount());
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuToggleView()
{
    if (m_pAIManagerView->isColumnHidden(1))
    {
        showDetails();
    }
    else
    {
        showList();
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::showDetails()
{
    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    PlugInModel* plugInModel = (PlugInModel*)(aim->getPluginModel());
    bool isList = true;

    for (int i = 1; i < plugInModel->columnCount(); ++i)
    {
        isList = isList && m_pAIManagerView->isColumnHidden(i);
    }

    setTreeViewHideColumns(false, plugInModel->columnCount());

    if (isList)
    {
        for (int i = 0; i < plugInModel->columnCount(); ++i)
        {
            m_pAIManagerView->setColumnWidth(i, m_pColumnWidth[i]);
        }
    }
}

//-------------------------------------------------------------------------------------
void AIManagerWidget::mnuShowInfo()
{
    QModelIndex index = m_pAIManagerView->currentIndex();

    if (index.isValid() && m_pSortFilterProxyModel)
    {
        index = m_pSortFilterProxyModel->mapToSource(index);
    }

    if (index.isValid())
    {
        emit(showDockWidget());
        PlugInModel::tItemType itemType;
        size_t itemInternalData;
        PlugInModel* plugInModel = (PlugInModel*)(index.model());

        if (plugInModel->getModelIndexInfo(index, itemType, itemInternalData))
        {
            if (itemType & PlugInModel::itemFilter)
            { // Filter
                ito::AddInAlgo::FilterDef* awd = (ito::AddInAlgo::FilterDef*)itemInternalData;
                emit showPluginInfo(
                    "Algorithms." + awd->m_pBasePlugin->objectName() + "." + awd->m_name,
                    HelpTreeDockWidget::typeFilter);
            }
            else if (itemType & PlugInModel::itemWidget)
            { // Widget
                ito::AddInAlgo::AlgoWidgetDef* awd =
                    (ito::AddInAlgo::AlgoWidgetDef*)itemInternalData;
                emit showPluginInfo(
                    "Widgets." + awd->m_pBasePlugin->objectName() + "." + awd->m_name,
                    HelpTreeDockWidget::typeWidget);
            }
            else if (itemType & PlugInModel::itemPlugin)
            { // DataIO and Actuator and Plugins (eg BasicFilters)
                ito::AddInInterfaceBase* aib = (ito::AddInInterfaceBase*)itemInternalData;
                if (aib->getType() & ito::typeActuator)
                {
                    emit showPluginInfo(
                        "Actuator." + aib->objectName(), HelpTreeDockWidget::typeActuator);
                }
                else if (aib->getType() & ito::typeDataIO)
                {
                    if (aib->getType() & ito::typeADDA)
                    {
                        emit showPluginInfo(
                            "DataIO.ADDA." + aib->objectName(), HelpTreeDockWidget::typeDataIO);
                    }
                    else if (aib->getType() & ito::typeGrabber)
                    {
                        emit showPluginInfo(
                            "DataIO.Grabber." + aib->objectName(), HelpTreeDockWidget::typeDataIO);
                    }
                    else if (aib->getType() & ito::typeRawIO)
                    {
                        emit showPluginInfo(
                            "DataIO.Raw IO." + aib->objectName(), HelpTreeDockWidget::typeDataIO);
                    }
                }
                else if (aib->getType() & ito::typeAlgo)
                {
                    emit showPluginInfo(
                        "Algorithms." + aib->objectName(), HelpTreeDockWidget::typeFPlugin);
                }
            }
        }
    }
}

} // end namespace ito
