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
#include <QSettings>
#include <QDir>
#include <qmessagebox.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagement::DialogUserManagement(QWidget *parent, Qt::WindowFlags f) :
    QDialog(parent),
    m_userModel(NULL)
{
    ui.setupUi(this);

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

    if (!uidList.isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("UserID already exists! Cannot create user!"), QMessageBox::Ok);
        return;
    }

    if (ui.comboBox_group->currentText() == "developer")
        group = "developer";
    else if (ui.comboBox_group->currentText() == "admin")
        group = "admin";
    else if (ui.comboBox_group->currentText() == "user")
        group = "user";
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
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagement::on_pushButton_delUser_clicked()
{
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