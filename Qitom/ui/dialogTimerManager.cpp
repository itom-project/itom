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

#include "dialogTimerManager.h"
#include "../AppManagement.h"
#include "../organizer/uiOrganizer.h"
#include "../models/timerModel.h"

namespace ito
{

//-------------------------------------------------------------------------------------
DialogTimerManager::DialogTimerManager(QWidget *parent /*= nullptr*/) :
	QDialog(parent),
    m_pModel(nullptr)
{
	ui.setupUi(this);

    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrg)
    {
        m_pModel = uiOrg->getTimerModel();
        ui.listView->setModel(m_pModel);
        m_pModel->autoUpdateModel(true);

        QItemSelectionModel *selectionModel = ui.listView->selectionModel();
        connect(selectionModel, &QItemSelectionModel::currentChanged,
            this, &DialogTimerManager::listView_currentChanged);
        connect(m_pModel, &QAbstractItemModel::dataChanged,
            this, &DialogTimerManager::listView_dataChanged);
    }

    ui.btnStart->setEnabled(false);
    ui.btnStop->setEnabled(false);
    ui.btnStopAll->setEnabled(false);
}

//-------------------------------------------------------------------------------------
DialogTimerManager::~DialogTimerManager()
{
    if (m_pModel)
    {
        m_pModel->autoUpdateModel(false);
    }
}

//-------------------------------------------------------------------------------------
void DialogTimerManager::on_btnStop_clicked()
{
    if (m_pModel)
    {
        m_pModel->timerStop(ui.listView->currentIndex());
    }
}

//-------------------------------------------------------------------------------------
void DialogTimerManager::on_btnStart_clicked()
{
    if (m_pModel)
    {
        m_pModel->timerStart(ui.listView->currentIndex());
    }
}

//-------------------------------------------------------------------------------------
void DialogTimerManager::on_btnStopAll_clicked()
{
    if (m_pModel)
    {
        m_pModel->timerStopAll();
    }
}

//-------------------------------------------------------------------------------------
void DialogTimerManager::listView_currentChanged(const QModelIndex &current, const QModelIndex &/*previous*/)
{
    if (m_pModel && current.isValid())
    {
        bool active = m_pModel->data(current, Qt::UserRole).toBool();
        ui.btnStart->setEnabled(!active);
        ui.btnStop->setEnabled(active);
    }
    else
    {
        ui.btnStart->setEnabled(false);
        ui.btnStop->setEnabled(false);
    }

    ui.btnStopAll->setEnabled(m_pModel->rowCount() > 0);
}

//-------------------------------------------------------------------------------------
void DialogTimerManager::listView_dataChanged(const QModelIndex &/*topLeft*/, const QModelIndex &/*bottomRight*/)
{
    QModelIndex idx = ui.listView->currentIndex();
    listView_currentChanged(idx, QModelIndex());
}

} //end namespace ito
