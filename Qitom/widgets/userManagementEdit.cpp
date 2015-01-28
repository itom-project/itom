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

#include "UserManagementEdit.h"
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
int DialogUserManagementEdit::getFlags()
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
        flags |= featProperties;
    }

    if (ui.checkBox_userManag->isChecked())
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

    if (ui.radioButton_consoleRO->isChecked())
    {
        flags |= featConsole;
    }

    return flags;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DialogUserManagementEdit::saveUser()
{
    bool newUser = m_fileName == "";
    QString uid;
    QString group;
    QString name = ui.lineEdit_name->text();
    QString iniFile;

    if (newUser && ui.lineEdit_id->text().isEmpty())
    {
        uid = ui.lineEdit_name->text();
    }
    else
    {
        uid = ui.lineEdit_id->text();
    }

    if (name.isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("Name is empty! Cannot create user!"), QMessageBox::Ok);
        return false;
    }

    QModelIndex startIdx = m_userModel->index(0, 1);
    if (newUser && !m_userModel->match(startIdx, Qt::DisplayRole, uid, -1).isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("UserID already exists! Cannot create user!"), QMessageBox::Ok);
        return false;
    }

    if ((name = ui.lineEdit_name->text()).isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("No user name entered, aborting!"), QMessageBox::Ok);
        return false;
    }

    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        QMessageBox::critical(this, tr("Error"), tr("ItomSettings directory not found, aborting!"), QMessageBox::Ok);
        return false;
/*        if (!appDir.exists("itomDefault.ini"))
        {
            QMessageBox::critical(this, tr("Error"), tr("Standard itom ini file not found, aborting!"), QMessageBox::Ok);
        }*/
    }

    if (newUser)
    {
        QFile stdIniFile(QDir::cleanPath(appDir.absoluteFilePath(QString("itomDefault.ini"))));
        iniFile = QDir::cleanPath(appDir.absoluteFilePath(QString("itom_").append(uid).append(".ini")));
        if (!stdIniFile.copy(iniFile))
        {
            QMessageBox::critical(this, tr("Error"), tr("Could not copy standard itom ini file!"), QMessageBox::Ok);
            return false;
        }
    }
    else
    {
        iniFile = m_fileName;
    }

    if (ui.radioButton_roleDevel->isChecked())
    {
        group = "developer";
    }
    else if (ui.radioButton_roleAdmin->isChecked())
    {
        group = "admin";
    }
    else
    {
        group = "user";
    }
    
    QSettings settings(iniFile, QSettings::IniFormat);
    settings.beginGroup("ITOMIniFile");
    settings.setValue("name", name);
    settings.setValue("role", group);
    UserOrganizer *uio = (UserOrganizer*)AppManagement::getUserOrganizer();
    uio->writeFlagsToFile(getFlags(), iniFile);
    settings.endGroup();

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagementEdit::DialogUserManagementEdit(const QString fileName, UserModel *userModel, QWidget *parent, Qt::WindowFlags f) :
    QDialog(parent),
    m_userModel(userModel),
    m_fileName(fileName)
{
    ui.setupUi(this);

    if (m_fileName == "")
    {
        setWindowTitle(tr("User Management - New User"));
    }
    else
    {
        setWindowTitle(tr("User Management - Edit User"));

        QSettings settings(fileName, QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        if (settings.contains("name"))
        {
            ui.lineEdit_name->setText(QString(settings.value("name").toString()));

            QFileInfo file = m_fileName;
            QString fileName = file.fileName();
            ui.lineEdit_id->setText(fileName.mid(5, fileName.length() - 9));
            ui.lineEdit_id->setEnabled(false);

            QString roleStr = QString(settings.value("role").toString());
            if (roleStr == "developer")
            {
                ui.radioButton_roleDevel->setChecked(true);
            }
            else if (roleStr == "admin")
            {
                ui.radioButton_roleAdmin->setChecked(true);
            }
            else
            {
                ui.radioButton_roleUser->setChecked(true);
            }
        }
        settings.endGroup();

        UserOrganizer *uio = (UserOrganizer*)AppManagement::getUserOrganizer();
        long flags = uio->getFlagsFromFile(fileName);
        ui.checkBox_devTools->setChecked(flags & featDeveloper);
        ui.checkBox_fileSystem->setChecked(flags & featFileSystem);
        ui.checkBox_userManag->setChecked(flags & featUserManag);
        ui.checkBox_addInManager->setChecked(flags & featPlugins);
        ui.checkBox_editProperties->setChecked(flags & featProperties);

        if ((flags & featConsole) && (flags & featConsoleRW))
        {
            ui.radioButton_consoleNormal->setChecked(true);
        }
        else if (flags & featConsole)
        {
            ui.radioButton_consoleRO->setChecked(true);
        }
        else
        {
            ui.radioButton_consoleOff->setChecked(true);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagementEdit::~DialogUserManagementEdit()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_buttonBox_clicked(QAbstractButton* btn)
{
    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::AcceptRole)
    {
        if (saveUser())
        {
            accept(); //AcceptRole
        }
    }
    else
    {
        reject(); //close dialog with reject
    }
}

//----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito