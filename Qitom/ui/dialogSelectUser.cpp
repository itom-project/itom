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

#include "dialogSelectUser.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogSelectUser::DialogSelectUser(QWidget *parent) :
    QDialog(parent),
    m_userModel(NULL)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogSelectUser::~DialogSelectUser()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSelectUser::DialogInit(UserModel *model)
{
    m_userModel = model;
    QItemSelectionModel *selModel = ui.userList->selectionModel();
    QObject::connect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSelectUser::userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous)
{
    bool userExists = false;

    ui.permissionList->clear();

    if (m_userModel)
    {
        QModelIndex curIdx = ui.userList->currentIndex();
        if (curIdx.isValid())
        {
            userExists = true;

            ui.lineEdit_name->setText(m_userModel->index(curIdx.row(), 0).data().toString());
            ui.lineEdit_id->setText(m_userModel->index(curIdx.row(), 1).data().toString());
            ui.lineEdit_iniFile->setText(m_userModel->index(curIdx.row(), 3).data().toString());

            ito::UserRole role = m_userModel->index(curIdx.row(), 2).data().value<ito::UserRole>();
            UserFeatures features = m_userModel->index(curIdx.row(), 4).data().value<UserFeatures>();
            
            ui.permissionList->addItem(tr("Role") + ": " + m_userModel->getRoleName(role));
            

            if (features & featDeveloper)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featDeveloper));
            }

            if (features & featFileSystem)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featFileSystem));
            }

            if (features & featUserManag)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featUserManag));
            }

            if (features & featPlugins)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featPlugins));
            }

            if (features & featProperties)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featProperties));
            }

            if ((features & featConsoleReadWrite))
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featConsoleReadWrite));
            }
            else if (features & featConsoleRead)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featConsoleRead));
            }
        }
    }

    if (!userExists)
    {
        ui.lineEdit_name->setText("");
        ui.lineEdit_id->setText("");
        ui.lineEdit_iniFile->setText("");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSelectUser::on_userList_doubleClicked(const QModelIndex current)
{
    this->accept();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSelectUser::on_buttonBox_clicked(QAbstractButton* btn)
{
    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::AcceptRole)
    {
        accept(); //AcceptRole
    }
    else
    {
        reject(); //close dialog with reject
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
} //end namespace ito