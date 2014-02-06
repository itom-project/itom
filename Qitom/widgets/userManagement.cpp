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
int DialogUserManagement::getFlags()
{
    int flags = 0;
    if (ui.checkBox_fileSystem->isChecked())
    {
        flags |= featFileSystem;
    }
    if (ui.checkBox_devTools->isChecked())
    {
        flags |= featDeveloper;
    }
    if (ui.checkBox_editProperties->isChecked())
    {
        flags |= featUserManag;
    }
    if (ui.checkBox_addInManager->isChecked())
    {
        flags |= featPlugins;
    }
    if (ui.radioButton_consoleNormal->isChecked())
    {
        flags |= featConsole | featConsoleRW;
    }
    if (ui.radioButton_consoleNormal->isChecked())
    {
        flags |= featConsole;
    }

    return flags;
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

//    int nUser = 0;
    foreach(QString iniFile, iniList) 
    {
        QSettings settings(QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        if (settings.contains("name"))
        {
            qDebug() << "found user ini file: " << iniFile;
            m_userModel->addUser(UserInfoStruct(QString(settings.value("name").toString()), iniFile.mid(5, iniFile.length() - 9), QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QString(settings.value("role").toString())));
        }
        settings.endGroup();
    }

    ui.userList->setModel(m_userModel);
    selModel = ui.userList->selectionModel();
    QObject::connect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::DialogUserManagement(QWidget *parent, Qt::WindowFlags f) :
    QDialog(parent),
    m_userModel(NULL)
{
    ui.setupUi(this);

    loadUserList();
/*
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

    int nUser = 0;
    foreach(QString iniFile, iniList) 
    {
        QSettings settings(QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        if (settings.contains("name"))
        {
            qDebug() << "found user ini file: " << iniFile;
            m_userModel->addUser(UserInfoStruct(QString(settings.value("name").toString()), iniFile.mid(5, iniFile.length() - 9), QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QString(settings.value("role").toString())));
        }
        settings.endGroup();
    }

    ui.userList->setModel(m_userModel);
    QItemSelectionModel *selModel = ui.userList->selectionModel();
    QObject::connect(selModel, SIGNAL(currentChanged (const QModelIndex &, const QModelIndex &)), this, SLOT(userListCurrentChanged(const QModelIndex &, const QModelIndex &))); 
*/
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::~DialogUserManagement()
{
    m_userModel->deleteLater();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::userListCurrentChanged(const QModelIndex &current, const QModelIndex &previous)
{
    QModelIndex curIdx = ui.userList->currentIndex();
    if (curIdx.isValid())
    {
        QModelIndex midx = m_userModel->index(curIdx.row(), 0);
        ui.lineEdit_name->setText(midx.data().toString());
        midx = m_userModel->index(curIdx.row(), 1);
        ui.lineEdit_id->setText(midx.data().toString());
        midx = m_userModel->index(curIdx.row(), 2);
        if (midx.data().toString() == "developer")
        {
            ui.comboBox_group->setCurrentIndex(0);
        }
        else if (midx.data().toString() == "admin")
        {
            ui.comboBox_group->setCurrentIndex(1);
        }
        else
        {
            ui.comboBox_group->setCurrentIndex(2);
        }
        midx = m_userModel->index(curIdx.row(), 3);
        ui.lineEdit_iniFile->setText(midx.data().toString());
        UserOrganizer *uio = (UserOrganizer*)AppManagement::getUserOrganizer();
        long flags = uio->getFlagsFromFile(midx.data().toString());

        if (flags & featFileSystem)
        {
            ui.checkBox_fileSystem->setChecked(1);
        }
        else
        {
            ui.checkBox_fileSystem->setChecked(0);
        }

        if (flags & featDeveloper)
        {
            ui.checkBox_devTools->setChecked(1);
        }
        else
        {
            ui.checkBox_devTools->setChecked(0);
        }

        if (flags & featUserManag)
        {
            ui.checkBox_editProperties->setChecked(1);
        }
        else
        {
            ui.checkBox_editProperties->setChecked(0);
        }

        if (flags & featPlugins)
        {
            ui.checkBox_addInManager->setChecked(1);
        }
        else
        {
            ui.checkBox_addInManager->setChecked(0);
        }

        if ((flags & featConsole) && (flags & featConsoleRW))
        {
            ui.radioButton_consoleNormal->setChecked(1);
        }
        else if (flags & featConsole)
        {
            ui.radioButton_consoleRO->setChecked(1);
        }
        else
        {
            ui.radioButton_consoleOff->setChecked(1);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_newUser_clicked()
{
    QString uid = ui.lineEdit_id->text();
    QString group;
    QString name;
    QString iniFile;
    QModelIndex startIdx = m_userModel->index(0, 1);
    QModelIndexList uidList = m_userModel->match(startIdx, Qt::DisplayRole, uid, -1);

    if (uid.isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("UserID is empty! Cannot create user!"), QMessageBox::Ok);
        return;
    }

    if (!uidList.isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("UserID already exists! Cannot create user!"), QMessageBox::Ok);
        return;
    }

    if (ui.comboBox_group->currentText() == "developer")
    {
        group = "developer";
    }
    else if (ui.comboBox_group->currentText() == "admin")
    {
        group = "admin";
    }
    else if (ui.comboBox_group->currentText() == "user")
    {
        group = "user";
    }
    else
    {
        QMessageBox::warning(this, tr("Warning"), tr("No or invalid group entered, setting to developer!"), QMessageBox::Ok);
        group = "developer";
    }
    
    if ((name = ui.lineEdit_name->text()).isEmpty())
    {
        QMessageBox::warning(this, tr("Warning"), tr("No user name entered, creating user with empty name!"), QMessageBox::Ok);
        name = "";
    }

    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        if (!appDir.exists("itomDefault.ini"))
        {
            QMessageBox::critical(this, tr("Error"), tr("Standard itom ini file not found, aborting!"), QMessageBox::Ok);
            return;
        }
    }
    else
    {
        QFile stdIniFile(QDir::cleanPath(appDir.absoluteFilePath(QString("itomDefault.ini"))));
        iniFile = QDir::cleanPath(appDir.absoluteFilePath(QString("itom_").append(uid).append(".ini")));
        if (!stdIniFile.copy(iniFile))
        {
            QMessageBox::critical(this, tr("Error"), tr("Could not copy standard itom ini file!"), QMessageBox::Ok);
            return;
        }
    }

    QSettings settings(iniFile, QSettings::IniFormat);
    settings.beginGroup("ITOMIniFile");
    settings.setValue("name", name);
    settings.setValue("role", group);
    UserOrganizer *uio = (UserOrganizer*)AppManagement::getUserOrganizer();
    uio->writeFlagsToFile(getFlags(), iniFile);
    settings.endGroup();

    loadUserList();
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

    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_resetGroup_clicked()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_pluginsEnableAll_clicked()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_pluginsDisableAll_clicked()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_buttonBox_apply()
{
}

//----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito