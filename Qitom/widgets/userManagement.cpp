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

#include "userManagement.h"
#include "userManagementEdit.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"

#include <QSettings>
#include <QDir>
#include <qmessagebox.h>
#include <qtimer.h>
#include <qdebug.h>
#include <QCryptographicHash>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::readModel(const QModelIndex &index)
{
    ui.permissionList->clear();

    if (index.isValid())
    {
        UserOrganizer *uio = (UserOrganizer*)AppManagement::getUserOrganizer();
        ui.lineEdit_name->setText(m_userModel->index(index.row(), 0).data().toString());
        ui.lineEdit_id->setText(m_userModel->index(index.row(), 1).data().toString());
        ui.lineEdit_iniFile->setText(m_userModel->index(index.row(), 3).data().toString());

        QString roleText;
        QModelIndex midx = m_userModel->index(index.row(), 2);
        if (midx.data().toString() == "developer")
        {
            roleText = uio->strConstRoleDeveloper;
        }
        else if (midx.data().toString() == "admin")
        {
            roleText = uio->strConstRoleAdministrator;
        }
        else
        {
            roleText = uio->strConstRoleUser;
        }
        ui.permissionList->addItem(uio->strConstRole + ": " + roleText);

        long flags = uio->getFlagsFromFile(m_userModel->index(index.row(), 3).data().toString());
        if (flags & featDeveloper)
        {
            ui.permissionList->addItem(uio->strConstFeatDeveloper);
        }

        if (flags & featFileSystem)
        {
            ui.permissionList->addItem(uio->strConstFeatFileSystem);
        }

        if (flags & featUserManag)
        {
            ui.permissionList->addItem(uio->strConstFeatUserManag);
        }

        if (flags & featPlugins)
        {
            ui.permissionList->addItem(uio->strConstFeatPlugins);
        }

        if (flags & featProperties)
        {
            ui.permissionList->addItem(uio->strConstFeatProperties);
        }

        if ((flags & featConsole) && (flags & featConsoleRW))
        {
            ui.permissionList->addItem(uio->strConstFeatConsole);
        }
        else if (flags & featConsole)
        {
            ui.permissionList->addItem(uio->strConstFeatConsoleRO);
        }

//        ui.userList->setCurrentIndex(index);
        ui.pushButton_editUser->setEnabled(true);
        ui.pushButton_delUser->setEnabled(m_currentUser != ui.lineEdit_name->text());
    }
    else
    {
        ui.lineEdit_name->setText("");
        ui.lineEdit_id->setText("");
        ui.lineEdit_iniFile->setText("");

        ui.pushButton_editUser->setEnabled(false);
        ui.pushButton_delUser->setEnabled(false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::loadUserList()
{
    QItemSelectionModel *selModel = ui.userList->selectionModel();
    QObject::disconnect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
    if (m_userModel)
    {
        m_userModel->deleteLater();
    }

    m_userModel = new UserModel();
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, "itomSettings");
    QSettings::setDefaultFormat(QSettings::IniFormat);

    QString settingsFile;
    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        appDir.mkdir("itomSettings");
        appDir.cd("itomSettings");
    }

    QStringList iniList = appDir.entryList(QStringList("itom_*.ini"));

    bool userModelHasItem = false;
    foreach(QString iniFile, iniList) 
    {
        QSettings settings(QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        if (settings.contains("name"))
        {
            qDebug() << "found user ini file: " << iniFile;
            m_userModel->addUser(UserInfoStruct(QString(settings.value("name").toString()), iniFile.mid(5, iniFile.length() - 9), QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QString(settings.value("role").toString())));
            userModelHasItem = true;
        }
        settings.endGroup();
    }

    ui.userList->setModel(m_userModel);

    if (!userModelHasItem)
    {
        ui.pushButton_editUser->setEnabled(false);
        ui.pushButton_delUser->setEnabled(false);
    }
    else
    {
        readModel(m_userModel->index(0, 1));
    }

    selModel = ui.userList->selectionModel();

    QObject::connect(selModel, SIGNAL(currentChanged(const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::openUserManagementEdit(const QString fileName, UserModel *userModel)
{
    DialogUserManagementEdit *dlg = new DialogUserManagementEdit(fileName, userModel);
    dlg->exec();
    if (dlg->result() == QDialog::Accepted)
    {
        loadUserList();
    }

    DELETE_AND_SET_NULL(dlg);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::DialogUserManagement(QWidget *parent, Qt::WindowFlags f) :
    QDialog(parent),
    m_userModel(NULL),
    m_currentUser("")
{
    ui.setupUi(this);

    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    m_currentUser = uOrg->getUserName();
    setWindowTitle(tr("User Management - Current User: ") + m_currentUser);

    loadUserList();
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::~DialogUserManagement()
{
    m_userModel->deleteLater();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous)
{
    readModel(ui.userList->currentIndex());
/*    ui.permissionList->clear();

    QModelIndex curIdx = ui.userList->currentIndex();
    if (curIdx.isValid())
    {
        UserOrganizer *uio = (UserOrganizer*)AppManagement::getUserOrganizer();
        ui.lineEdit_name->setText(m_userModel->index(curIdx.row(), 0).data().toString());
        ui.lineEdit_id->setText(m_userModel->index(curIdx.row(), 1).data().toString());
        ui.lineEdit_iniFile->setText(m_userModel->index(curIdx.row(), 3).data().toString());

        QString roleText;
        QModelIndex midx = m_userModel->index(curIdx.row(), 2);
        if (midx.data().toString() == "developer")
        {
            roleText = uio->strConstRoleDeveloper;
        }
        else if (midx.data().toString() == "admin")
        {
            roleText = uio->strConstRoleAdministrator;
        }
        else
        {
            roleText = uio->strConstRoleUser;
        }
        ui.permissionList->addItem(uio->strConstRole + ": " + roleText);

        long flags = uio->getFlagsFromFile(m_userModel->index(curIdx.row(), 3).data().toString());
        if (flags & featDeveloper)
        {
            ui.permissionList->addItem(uio->strConstFeatDeveloper);
        }

        if (flags & featFileSystem)
        {
            ui.permissionList->addItem(uio->strConstFeatFileSystem);
        }

        if (flags & featUserManag)
        {
            ui.permissionList->addItem(uio->strConstFeatUserManag);
        }

        if (flags & featPlugins)
        {
            ui.permissionList->addItem(uio->strConstFeatPlugins);
        }

        if (flags & featProperties)
        {
            ui.permissionList->addItem(uio->strConstFeatProperties);
        }

        if ((flags & featConsole) && (flags & featConsoleRW))
        {
            ui.permissionList->addItem(uio->strConstFeatConsole);
        }
        else if (flags & featConsole)
        {
            ui.permissionList->addItem(uio->strConstFeatConsoleRO);
        }

        ui.pushButton_editUser->setEnabled(true);
        ui.pushButton_delUser->setEnabled(m_currentUser != ui.lineEdit_name->text());
    }
    else
    {
        ui.lineEdit_name->setText("");
        ui.lineEdit_id->setText("");
        ui.lineEdit_iniFile->setText("");

        ui.pushButton_editUser->setEnabled(false);
        ui.pushButton_delUser->setEnabled(false);
    }*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_newUser_clicked()
{
    openUserManagementEdit("", m_userModel);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_delUser_clicked()
{
    QString uid = ui.lineEdit_id->text();
    QString iniFile = ui.lineEdit_iniFile->text();
    QString name = ui.lineEdit_name->text();
    QModelIndex startIdx = m_userModel->index(0, 1);
    QModelIndexList uidList = m_userModel->match(startIdx, Qt::DisplayRole, uid, -1);
    QDir appDir(QCoreApplication::applicationDirPath());

    if (uidList.isEmpty())
    {
        QMessageBox::warning(this, tr("Warning"), tr("User ID not found, aborting!"), QMessageBox::Ok);
        return;
    }

    QString tempPath = QDir::cleanPath(appDir.absoluteFilePath(QString("itomSettings/itom_").append(uid).append(".ini")));
    if (iniFile != tempPath)
    {
        QMessageBox::warning(this, tr("Warning"), tr("User ID and ini file name mismatch, aborting!"), QMessageBox::Ok);
        return;
    }

    QSettings settings(iniFile, QSettings::IniFormat);
    if (settings.value("ITOMIniFile/name").toString() != name)
    {
        QMessageBox::warning(this, tr("Warning"), tr("User name and ini file user name mismatch, aborting!"), QMessageBox::Ok);
        return;
    }

    if (uid == ((ito::UserOrganizer*)AppManagement::getUserOrganizer())->getUserID())
    {
        QMessageBox::warning(this, tr("Warning"), tr("Cannot delete current user, aborting!"), QMessageBox::Ok);
        return;
    }

    QString msg = QString("Warning the ini file\n").append(iniFile).append("\nfor user ").append(name).append(" will be deleted!\nAre you sure?");
    if (QMessageBox::warning(this, tr("Warning"), tr(msg.toLatin1().data()), QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
    {
        QFile file(iniFile);
        if (!file.remove())
            QMessageBox::warning(this, tr("Warning"), tr((QString("file: \n").append(iniFile).append("\ncould not be deleted!")).toLatin1().data()), QMessageBox::Ok);

        loadUserList();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_editUser_clicked()
{
    QModelIndex curIdx = ui.userList->currentIndex();
    if (curIdx.isValid())
    {
        openUserManagementEdit(m_userModel->index(curIdx.row(), 3).data().toString(), m_userModel);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_userList_doubleClicked(const QModelIndex & index)
{
    if (index.isValid())
    {
        openUserManagementEdit(m_userModel->index(index.row(), 3).data().toString(), m_userModel);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito