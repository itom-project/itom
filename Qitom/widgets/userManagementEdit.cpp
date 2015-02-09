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

#include "userManagementEdit.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"

#include <QDir>
#include <qmessagebox.h>
#include <qtimer.h>
#include <qdebug.h>
#include <QCryptographicHash>
#include <qobject.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
bool DialogUserManagementEdit::saveUser()
{
    bool newUser = m_fileName == "";
    QString uid;
    QString group;
    QString username = ui.lineEdit_name->text();
    QString iniFile;

    if (newUser && ui.lineEdit_id->text().isEmpty())
    {
        uid = clearName(ui.lineEdit_name->text());
    }
    else
    {
        uid = clearName(ui.lineEdit_id->text());
    }

    if (username.isEmpty())
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

    if ((username = ui.lineEdit_name->text()).isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("No user name entered, aborting!"), QMessageBox::Ok);
        return false;
    }

    UserOrganizer *uio = qobject_cast<UserOrganizer*>(AppManagement::getUserOrganizer());
    if (uio)
    {
        UserRole role = userRoleBasic;
        if (ui.radioButton_roleDevel->isChecked())
        {
            role = userRoleDeveloper;
        }
        else if (ui.radioButton_roleAdmin->isChecked())
        {
            role = userRoleAdministrator;
        }

        UserFeatures flags;
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
            flags |= featConsoleReadWrite;
        }

        if (ui.radioButton_consoleRO->isChecked())
        {
            flags |= featConsoleRead;
        }

        ito::RetVal retval = uio->writeUserDataToFile(username, uid, flags, role); 
        if (retval.containsError())
        {
            QMessageBox::critical(this, tr("Error"), tr(retval.errorMessage()), QMessageBox::Ok);
            return false;
        }
    }
    else
    {
        QMessageBox::critical(this, tr("Error"), tr("UserOrganizer not found!"), QMessageBox::Ok);
        return false;
    }

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DialogUserManagementEdit::clearName(const QString &name)
{
    QString name_(name);
    name_.replace( QRegExp( "[" + QRegExp::escape( "\\/:*?\"<>|" ) + "]" ), QString( "_" ) );
    name_.replace("ä", "ae");
    name_.replace("ö", "oe");
    name_.replace("ü", "ue");
    name_.replace("Ä", "Ae");
    name_.replace("Ö", "Oe");
    name_.replace("Ü", "Ue");
    name_.replace("ß", "ss");

    return name_;
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagementEdit::DialogUserManagementEdit(const QString &filename, UserModel *userModel, QWidget *parent, Qt::WindowFlags f) :
    QDialog(parent, f),
    m_userModel(userModel),
    m_fileName(filename)
{
    ui.setupUi(this);

    if (m_fileName == "")
    {
        setWindowTitle(tr("User Management - New User"));
    }
    else
    {
        setWindowTitle(tr("User Management - Edit User"));

        UserOrganizer *uio = qobject_cast<UserOrganizer*>(AppManagement::getUserOrganizer());
        if (uio)
        {
            QString username;
            QString uid;
            UserFeatures features;
            UserRole role;
            if (uio->readUserDataFromFile(filename, username, uid, features, role) == ito::retOk)
            {
                ui.lineEdit_name->setText(username);
                
                ui.lineEdit_id->setText(uid);
                ui.lineEdit_id->setEnabled(false);
                
                switch (role)
                {
                case userRoleAdministrator:
                    ui.radioButton_roleAdmin->setChecked(true);
                    break;
                case userRoleDeveloper:
                    ui.radioButton_roleDevel->setChecked(true);
                    break;
                default:
                    ui.radioButton_roleUser->setChecked(true);
                }

                ui.checkBox_devTools->setChecked(features & featDeveloper);
                ui.checkBox_fileSystem->setChecked(features & featFileSystem);
                ui.checkBox_userManag->setChecked(features & featUserManag);
                ui.checkBox_addInManager->setChecked(features & featPlugins);
                ui.checkBox_editProperties->setChecked(features & featProperties);

                if ((features & featConsoleReadWrite))
                {
                    ui.radioButton_consoleNormal->setChecked(true);
                }
                else if (features & featConsoleRead)
                {
                    ui.radioButton_consoleRO->setChecked(true);
                }
                else
                {
                    ui.radioButton_consoleOff->setChecked(true);
                }

            }
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
