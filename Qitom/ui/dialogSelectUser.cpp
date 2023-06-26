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

#include "dialogSelectUser.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"


#include <qmessagebox.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogSelectUser::DialogSelectUser(UserModel *model, QWidget *parent) :
    QDialog(parent),
    m_userModel(model)
{
    ui.setupUi(this);

    ui.userList->setModel(m_userModel);

    QItemSelectionModel *selModel = ui.userList->selectionModel();
    QObject::connect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &)));
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogSelectUser::~DialogSelectUser()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DialogSelectUser::selectUser(const QString &id)
{
    for (int curIdx = 0; curIdx < m_userModel->rowCount(); curIdx++)
    {
        QModelIndex midx = m_userModel->index(curIdx, 1); //id
        if (midx.isValid())
        {
			if (QString::compare(id, m_userModel->index(curIdx, UserModel::umiName).data().toString(), Qt::CaseInsensitive) == 0)
            {
                QModelIndex actIdx = m_userModel->index(curIdx, 0);
                ui.userList->setCurrentIndex(actIdx);
                return true;
            }
        }
    }

	ui.userList->setCurrentIndex(m_userModel->index(0,0));
    return false;
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
            int curRow = curIdx.row();

            ui.lineEdit_name->setText(m_userModel->index(curRow, UserModel::umiName).data().toString());
            ui.lineEdit_id->setText(m_userModel->index(curRow, UserModel::umiId).data().toString());
            ui.lineEdit_iniFile->setText(m_userModel->index(curRow, UserModel::umiIniFile).data().toString());

            ito::UserRole role = m_userModel->index(curRow, UserModel::umiRole).data().value<ito::UserRole>();
            UserFeatures features = m_userModel->index(curRow, UserModel::umiFeatures).data().value<UserFeatures>();
            QVariant paswd = m_userModel->index(curRow, UserModel::umiPassword).data();
            QByteArray password = m_userModel->index(curRow, UserModel::umiPassword).data().toByteArray();
            bool hasPassword = !password.isEmpty();
            if (hasPassword)
            {
                ui.lineEdit_password->setEnabled(true);
                ui.label_password->setEnabled(true);
            }
            else
            {
                ui.lineEdit_password->setEnabled(false);
                ui.label_password->setEnabled(false);
            }

            ui.permissionList->addItem(tr("Role") + ": " + m_userModel->getRoleName(role));

            if (features & featDeveloper)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featDeveloper));
            }

            if (features & featFileSystem)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featFileSystem));
            }

            if (features & featUserManagement)
            {
                ui.permissionList->addItem(m_userModel->getFeatureName(featUserManagement));
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
int DialogSelectUser::checkPassword()
{
    QModelIndex curIdx = ui.userList->currentIndex();
	return m_userModel->checkPassword(curIdx, ui.lineEdit_password->text().toUtf8());
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSelectUser::on_userList_doubleClicked(const QModelIndex current)
{
    if (checkPassword())
        this->accept();
    else
        QMessageBox::critical(this, tr("Wrong password"), tr("Wrong password, select another user or try again"));
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSelectUser::on_buttonBox_clicked(QAbstractButton* btn)
{
    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::AcceptRole)
    {
        if (checkPassword())
            this->accept();
        else
            QMessageBox::critical(this, tr("Wrong password"), tr("Wrong password, select another user or try again"));
    }
    else
    {
        reject(); //close dialog with reject
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
} //end namespace ito
