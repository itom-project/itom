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

#include "userManagement.h"
#include "userManagementEdit.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"

#include <QSettings>
#include <QDir>
#include <qmessagebox.h>
#include <qtimer.h>
#include <qdebug.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::readModel(const QModelIndex &index)
{
    ui.permissionList->clear();

    if (index.isValid())
    {
        ui.lineEdit_name->setText(m_userModel->index(index.row(), UserModel::umiName).data().toString());
        ui.lineEdit_id->setText(m_userModel->index(index.row(), UserModel::umiId).data().toString());
        ui.lineEdit_iniFile->setText(m_userModel->index(index.row(), UserModel::umiIniFile).data().toString());

        ito::UserRole role = m_userModel->index(index.row(), UserModel::umiRole).data().value<ito::UserRole>();
        UserFeatures features = m_userModel->index(index.row(), UserModel::umiFeatures).data().value<UserFeatures>();

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

        UserRole currentUserRole = m_userModel->getUserRole(m_userModel->currentUser());

        if (currentUserRole >= role)
        {
            ui.pushButton_delUser->setEnabled(
                m_currentUserName != ui.lineEdit_name->text() &&
                m_userModel->data(index, Qt::EditRole).isValid() &&
                m_userModel->index(index.row(), UserModel::umiIniFile).data().toString().contains("itom_"));
        }
        else
        {
            ui.pushButton_delUser->setEnabled(false);
        }

        ui.userList->setCurrentIndex(index);
    }
    else
    {
        ui.lineEdit_name->setText("");
        ui.lineEdit_id->setText("");
        ui.lineEdit_iniFile->setText("");

        ui.pushButton_editUser->setEnabled(false);
        ui.pushButton_delUser->setEnabled(false);

        ui.userList->setCurrentIndex(QModelIndex());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::loadUserList()
{
    QItemSelectionModel *selModel = ui.userList->selectionModel();
    disconnect(selModel, &QItemSelectionModel::currentChanged, this, &DialogUserManagement::userListCurrentChanged);

    // select the first user
    readModel(m_userModel->index(0, 0));

    selModel = ui.userList->selectionModel();
    connect(selModel, &QItemSelectionModel::currentChanged, this, &DialogUserManagement::userListCurrentChanged);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::DialogUserManagement(QWidget* parent, Qt::WindowFlags f) :
    QDialog(parent), m_userModel(NULL), m_currentUserName("")
{
    ui.setupUi(this);

    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    m_currentUserName = uOrg->getCurrentUserName();
    m_userModel = uOrg->getUserModel();
    ui.userList->setModel(m_userModel);
    setWindowTitle(tr("User Management - Current User: ") + m_currentUserName);

    loadUserList();
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::~DialogUserManagement()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous)
{
    readModel(ui.userList->currentIndex());
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_newUser_clicked()
{
    DialogUserManagementEdit *dlg = new DialogUserManagementEdit("", m_userModel);

    if (dlg->exec() == QDialog::Accepted)
    {
        loadUserList();

        //last is the standard user, new is the one before
        QModelIndex newIndex = m_userModel->index(m_userModel->rowCount() - 2, 0);

        if (newIndex.isValid())
        {
            readModel(newIndex);
        }
    }

    DELETE_AND_SET_NULL(dlg);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_delUser_clicked()
{
    QString uid = ui.lineEdit_id->text();
    QString iniFile = ui.lineEdit_iniFile->text();
    QString name = ui.lineEdit_name->text();
    QModelIndex startIdx = m_userModel->index(0, 1);
    QModelIndexList uidList = m_userModel->match(startIdx, Qt::DisplayRole, uid, -1, Qt::MatchExactly | Qt::MatchWrap);
    QDir appDir(QCoreApplication::applicationDirPath());

    if (uidList.isEmpty())
    {
        QMessageBox::warning(this, tr("Warning"),
            tr("User ID not found, aborting!"), QMessageBox::Ok);
        return;
    }

    QString tempPath = QDir::cleanPath(appDir.absoluteFilePath(QString("itomSettings/itom_").append(uid).append(".ini")));

    if (iniFile != tempPath)
    {
        QMessageBox::warning(this, tr("Warning"),
            tr("User ID and ini file name mismatch, aborting!"), QMessageBox::Ok);
        return;
    }

    QSettings settings(iniFile, QSettings::IniFormat);
    if (settings.value("ITOMIniFile/name").toString() != name)
    {
        QMessageBox::warning(this, tr("Warning"),
            tr("User name and ini file user name mismatch, aborting!"), QMessageBox::Ok);
        return;
    }

    if (uid == ((ito::UserOrganizer*)AppManagement::getUserOrganizer())->getCurrentUserId())
    {
        QMessageBox::warning(this, tr("Warning"),
            tr("Cannot delete current user, aborting!"), QMessageBox::Ok);
        return;
    }

    QString msg = tr("The settings file '%1' for user '%2' will be deleted!\nAre you sure?")
                    .arg(iniFile)
                    .arg(name);

    if (QMessageBox::question(this, tr("Warning"), msg, QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
    {
        QFile file(iniFile);
        if (!file.remove())
        {
            QString msg2 = QString("File '%1' could not be deleted!").arg(iniFile);
            QMessageBox::warning(this, tr("Warning"), msg2, QMessageBox::Ok);
        }

        foreach (const QModelIndex &mi, uidList)
        {
            m_userModel->removeUser(mi);
        }

        loadUserList();

        QModelIndex newIndex = m_userModel->index(ui.userList->currentIndex().row(), 0);

        if (newIndex.isValid())
        {
            readModel(newIndex);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_editUser_clicked()
{
    QModelIndex curIdx = ui.userList->currentIndex();
    on_userList_doubleClicked(curIdx);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_userList_doubleClicked(const QModelIndex & index)
{
    if (index.isValid() && ui.pushButton_editUser->isEnabled())
    {
        DialogUserManagementEdit *dlg = new DialogUserManagementEdit(
            m_userModel->index(index.row(), UserModel::umiIniFile).data().toString(),
            m_userModel,
            nullptr,
            Qt::WindowFlags(),
            !m_userModel->data(index, Qt::EditRole).isValid());

        if (dlg->exec() == QDialog::Accepted)
        {
            loadUserList();
            readModel(index);
        }

        DELETE_AND_SET_NULL(dlg);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito
