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

#include "dialogPluginPicker.h"

#include "dialogNewPluginInstance.h"
#include "../AppManagement.h"
#include "../AddInManager/pluginModel.h"
#include <qmessagebox.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogPluginPicker::DialogPluginPicker(bool allowNewInstances, ito::AddInBase *currentItem, int minimumPluginTypeMask, QString pluginName , QWidget *parent) :
    QDialog(parent),
    m_pFilterModel(NULL)
{
    ui.setupUi(this);

    ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    m_pFilterModel = new PickerSortFilterProxyModel(this);
    m_pFilterModel->setSourceModel(aim->getPluginModel());
    m_pFilterModel->setPluginMinimumMask(minimumPluginTypeMask);
    m_pFilterModel->setPluginName(pluginName);
    m_pFilterModel->showPluginsWithoutInstance(false);

    ui.cmdNewInstance->setVisible(allowNewInstances);
    ui.cmdNewInstance->setEnabled(false);

    ui.treeView->setSortingEnabled(true);
    ui.treeView->setModel(m_pFilterModel);

    ui.treeView->setColumnHidden(1, true);
    ui.treeView->setColumnHidden(2, true);
    ui.treeView->setColumnHidden(3, true);
    ui.treeView->setColumnHidden(4, true);
    ui.treeView->setColumnHidden(5, true);
    ui.treeView->setColumnHidden(6, true);
    ui.treeView->setColumnHidden(7, true);

    ui.treeView->sortByColumn(1, Qt::AscendingOrder);
    ui.treeView->expandAll();

    connect(ui.checkShowPluginsWithoutInstance, SIGNAL(toggled(bool)), this, SLOT(showPluginsWithoutInstance(bool)));
    connect(ui.treeView, SIGNAL(doubleClicked(QModelIndex)), this, SLOT(itemDblClicked(QModelIndex)));
    connect(ui.cmdNewInstance, SIGNAL(clicked(bool)), this, SLOT(createNewInstance(bool)));
    connect(ui.treeView->selectionModel(), SIGNAL(selectionChanged(const QItemSelection&, const QItemSelection&)), this, SLOT(selectionChanged(const QItemSelection&, const QItemSelection&)), Qt::DirectConnection);

    QModelIndex currentIndexSource = aim->getPluginModel()->getIndexByAddIn(currentItem);
    ui.treeView->setCurrentIndex(m_pFilterModel->mapFromSource(currentIndexSource));
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogPluginPicker::itemClicked(const QModelIndex &index)
{
    bool enabled = false;

    if (index.isValid())
    {
        QModelIndex indexMap = m_pFilterModel->mapToSource(index);

        ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
        const ito::PlugInModel *model = aim->getPluginModel();

        int itemType = model->data(indexMap, Qt::UserRole + 3).toInt();
        if (itemType == ito::PlugInModel::itemInstance)
        {
            enabled = true;
        }
        else if (itemType == ito::PlugInModel::itemPlugin)
        {
            ito::AddInInterfaceBase *aib = (ito::AddInInterfaceBase *)(indexMap.internalPointer());
            enabled = (aib  && (aib->getType() & ito::typeAlgo) == 0);
        }
    }

    ui.cmdNewInstance->setEnabled(enabled);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogPluginPicker::itemDblClicked(const QModelIndex &index)
{
    itemClicked(index);
    this->accept();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogPluginPicker::showPluginsWithoutInstance(bool checked)
{
    m_pFilterModel->showPluginsWithoutInstance(checked);
    ui.treeView->expandAll();
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::AddInBase* DialogPluginPicker::getSelectedInstance()
{
    QModelIndex idx = ui.treeView->currentIndex();
    idx = m_pFilterModel->mapToSource(idx);
    if (idx.isValid())
    {
        int itemType = m_pFilterModel->sourceModel()->data(idx, Qt::UserRole + 3).toInt();
        if (itemType == ito::PlugInModel::itemInstance)
        {
            ito::AddInBase *ais = (ito::AddInBase *)(idx.internalPointer());
            return ais;
        }
    }

    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogPluginPicker::createNewInstance(bool /*checked*/)
{
    QModelIndex index = ui.treeView->currentIndex();
    index = m_pFilterModel->mapToSource(index);

    ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    const ito::PlugInModel *model = aim->getPluginModel();

    if (index.isValid())
    {
        int itemType = model->data(index, Qt::UserRole + 3).toInt();
        if (itemType == ito::PlugInModel::itemInstance)
        {
            index = index.parent();
        }
        else if (itemType != ito::PlugInModel::itemPlugin)
        {
            return;
        }
    }

    //here: we assume that index is never a filter or algowidget!!!
    //index is now of type instance.

    if (index.isValid())
    {
        ito::AddInInterfaceBase *aib = (ito::AddInInterfaceBase *)(index.internalPointer());

        if (aib  && (aib->getType() & ito::typeAlgo) == 0)
        {
            DialogNewPluginInstance *dialog = new DialogNewPluginInstance(index, aib, false);
            if (dialog->exec() == 1) //accepted
            {
                QVector<ito::ParamBase> paramsMandNew, paramsOptNew;
                QString pythonVarName = dialog->getPythonVariable();
                ito::RetVal retValue = ito::retOk;
                ito::AddInBase *basePlugin = NULL;

                retValue += dialog->getFilledMandParams(paramsMandNew);
                retValue += dialog->getFilledOptParams(paramsOptNew);

                DELETE_AND_SET_NULL(dialog);

                if (retValue.containsError())
                {
                    QString message = tr("Error while creating new instance. \nMessage: %1").arg(QLatin1String(retValue.errorMessage()));
                    QMessageBox::critical(this, tr("Error while creating new instance"), message);
                    return;
                }

                int itemNum = aim->getItemIndexInList((void*)aib);
                if (itemNum < 0)
                {
                    return;
                }

                QApplication::setOverrideCursor (QCursor(Qt::WaitCursor));
                QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                                 //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                                 //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

                if (aib->getType() & ito::typeDataIO)
                {
                    ito::AddInDataIO *plugin = NULL;
                    retValue += aim->initAddIn(itemNum, aib->objectName(), &plugin, &paramsMandNew, &paramsOptNew, false, NULL);
                    basePlugin = (ito::AddInBase*)(plugin);

                }
                else if (aib->getType() & ito::typeActuator)
                {
                    ito::AddInActuator *plugin = NULL;
                    retValue += aim->initAddIn(itemNum, aib->objectName(), &plugin, &paramsMandNew, &paramsOptNew, false, NULL);
                    basePlugin = (ito::AddInBase*)(plugin);
                }

                QApplication::restoreOverrideCursor();

                if (retValue.containsWarning())
                {
                    QString message = tr("Warning while creating new instance. Message: %1").arg(QLatin1String(retValue.errorMessage()));
                    QMessageBox::warning(this, tr("Warning while creating new instance"), message);
                }
                else if (retValue.containsError())
                {
                    QString message = tr("Error while creating new instance. Message: %1").arg(QLatin1String(retValue.errorMessage()));
                    QMessageBox::critical(this, tr("Error while creating new instance"), message);
                }

                if (basePlugin != NULL)
                {
                    basePlugin->setCreatedByGUI(1);
                }

                m_pFilterModel->invalidate();
                ui.treeView->expandAll();

                if (basePlugin != NULL)
                {
                    QModelIndex currentIndexSource = aim->getPluginModel()->getIndexByAddIn(basePlugin);
                    ui.treeView->setCurrentIndex(m_pFilterModel->mapFromSource(currentIndexSource));
                }
            }
            else
            {
                DELETE_AND_SET_NULL(dialog);
            }
        }
    }
    else
    {
        QMessageBox::information(this, tr("Choose plugin"), tr("Please choose plugin you want to create a new instance from"));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogPluginPicker::selectionChanged(const QItemSelection& newSelection, const QItemSelection& oldSelection)
{
    itemClicked(ui.treeView->currentIndex());
}

} //end namespace ito
