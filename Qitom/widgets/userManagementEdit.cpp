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

#include "userManagementEdit.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"

#include <QDir>
#include <qmessagebox.h>
#include <qtimer.h>
#include <qdebug.h>
#include <QCryptographicHash>
#include <qsettings.h>
#include <qregularexpression.h>
#include <qfiledialog.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
bool DialogUserManagementEdit::saveUser()
{
    bool newUser = m_fileName == "";
    QString uid;
    QString group;
    QString username = ui.lineEdit_name->text();
    QByteArray password;
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
        QMessageBox::critical(this, tr("Error"), tr("User ID already exists! Cannot create user!"), QMessageBox::Ok);
        return false;
    }

    if ((username = ui.lineEdit_name->text()).isEmpty())
    {
        QMessageBox::critical(this, tr("Error"), tr("No user name entered, aborting!"), QMessageBox::Ok);
        return false;
    }
    if (!m_showsStandardUser && (ui.lineEdit_name->text() == "Standard User"))
    {
        QMessageBox::critical(this, tr("Error"), tr("The user name \"Standard User\" is reserved!"), QMessageBox::Ok);
        return false;
    }
    if (!ui.lineEdit_name->text().isEmpty())
    {
        if (m_oldPassword != ui.lineEdit_password->text())
            if (!ui.lineEdit_password->text().isEmpty())
            {
                password = QCryptographicHash::hash(ui.lineEdit_password->text().toUtf8(), QCryptographicHash::Sha3_512);
            }
            else
            {
                password = QByteArray(); //clear password
            }
        else
            password = m_oldPassword;
    }
    else
        password = "";

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
            flags |= featUserManagement;
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

        ito::RetVal retval = uio->writeUserDataToFile(username, uid, flags, role, password, m_showsStandardUser);
        if (retval.containsError())
        {
            QMessageBox::critical(this, tr("Error"), QLatin1String(retval.errorMessage()), QMessageBox::Ok);
            return false;
        }

        QStringList files;
		const ito::UserModel* userModel = uio->getUserModel();
		QModelIndex index = userModel->getUser(uid);
		QString settingsFile = userModel->getUserSettingsFile(index);

        if (settingsFile != "")
        {
            QSettings settings(settingsFile, QSettings::IniFormat);
            settings.beginGroup("Python");
            settings.beginWriteArray("startupFiles");
            for (int i = 0; i < ui.lv_startUpScripts->count(); i++)
            {
                settings.setArrayIndex(i);
                settings.setValue("file", ui.lv_startUpScripts->item(i)->text());
                files.append(ui.lv_startUpScripts->item(i)->text());
            }

            settings.endArray();
            settings.endGroup();
        }
        else
        {
            QMessageBox::critical(this, tr("Error"), tr("Error retrieving user settings file name. Startup scripts not written to file."));
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
    name_.replace(QRegularExpression( "[" + QRegularExpression::escape( "\\/:*?\"<>|" ) + "]" ), QString( "_" ) );
    name_.replace(QChar(0x00, 0xE4), "ae"); //german umlaut 'a with diaresis' replaced by ae
    name_.replace(QChar(0x00, 0xF6), "oe"); //german umlaut 'o with diaresis' replaced by oe
    name_.replace(QChar(0x00, 0xFC), "ue"); //german umlaut 'u with diaresis' replaced by ue
    name_.replace(QChar(0x00, 0xC4), "Ae"); //german umlaut 'A with diaresis' replaced by Ae
    name_.replace(QChar(0x00, 0xD6), "Oe"); //german umlaut 'O with diaresis' replaced by Oe
    name_.replace(QChar(0x00, 0xDC), "Ue"); //german umlaut 'U with diaresis' replaced by Ue
    name_.replace(QChar(0x00, 0xDF), "ss"); //german sharp s replaced by ss
    name_.replace(" ", "_"); //replace space by underscore

    return name_;
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogUserManagementEdit::DialogUserManagementEdit(
    const QString& filename,
    UserModel* userModel,
    QWidget* parent,
    Qt::WindowFlags f,
    bool isStandardUser) :
    QDialog(parent, f),
    m_userModel(userModel), m_fileName(filename), m_osUser(""), m_showsStandardUser(isStandardUser)
{
    ui.setupUi(this);

    m_osUser = qgetenv("USERNAME");

    if (m_osUser.isEmpty())
    {
        m_osUser = qgetenv("USER");
    }

    ui.cmdUseWindowsUser->setVisible(m_osUser != "");

    if (m_fileName == "")
    {
        setWindowTitle(tr("User Management - New User"));

		UserOrganizer *uio = qobject_cast<UserOrganizer*>(AppManagement::getUserOrganizer());

		enableWidgetsByUserRole(uio->getCurrentUserRole(), uio->getCurrentUserFeatures(),
                                uio->getCurrentUserRole(), uio->getCurrentUserFeatures());

    }
    else
    {
        setWindowTitle(tr("User Management - Edit User"));

        UserOrganizer *uio = qobject_cast<UserOrganizer*>(AppManagement::getUserOrganizer());
        if (uio)
        {
            QString username;
            QString uid;
            QByteArray password;
            UserFeatures features;
            UserRole role;
            QDateTime modified;
            if (uio->readUserDataFromFile(filename, username, uid, features, role, password, modified) == ito::retOk)
            {
                ui.lineEdit_name->setText(username);

                ui.lineEdit_id->setText(uid);
                ui.lineEdit_id->setEnabled(false);
                m_oldPassword = password;
                ui.lineEdit_password->setText(password);

				enableWidgetsByUserRole(uio->getCurrentUserRole(), uio->getCurrentUserFeatures(), role, features);

                QSettings settings(filename, QSettings::IniFormat);
                settings.beginGroup("Python");

                int size = settings.beginReadArray("startupFiles");
                for (int i = 0; i < size; ++i)
                {
                    settings.setArrayIndex(i);
                    ui.lv_startUpScripts->addItem(settings.value("file", QString()).toString());
                }

                settings.endArray();
                settings.endGroup();
            }

            updateScriptButtons();
        }
    }

    QString label = ui.checkAddFileRel->text().arg(QCoreApplication::applicationDirPath());
    ui.checkAddFileRel->setText(label);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! sets the enabled and check state of many controls depending on the rights of the currently logged user and the user to be edited.
/*!
    If the current user has admin privileges, he has the right to edit all controls.
    If he is a developer, he can only switch the user level of the user to be edited
    to basic or developer. If he is a basic user, only basic can be chosen for the
    current user.

    \param currentUserRole is the user role of the currently logged-in user
    \param currentFeatures is the active feature set of the currently logged-in user
    \param userRole is the user role of the user, that is edited
    \param features is the current feature set of the user, that is edited.
*/
void DialogUserManagementEdit::enableWidgetsByUserRole(
    const UserRole currentUserRole,
    const UserFeatures &currentFeatures,
    const UserRole userRole,
    const UserFeatures &features)
{
    // set the state of the role radio buttons
    ui.radioButton_roleAdmin->setChecked(userRole == UserRole::userRoleAdministrator);
    ui.radioButton_roleDevel->setChecked(userRole == UserRole::userRoleDeveloper);
    ui.radioButton_roleUser->setChecked(userRole == UserRole::userRoleBasic);

    // enable / disable the role radio buttons
    // depending on the rights of the current user
    ui.radioButton_roleUser->setEnabled(!m_showsStandardUser);

	switch (currentUserRole)
	{
    default:
	case ito::userRoleBasic:
		ui.radioButton_roleAdmin->setEnabled(false);
		ui.radioButton_roleDevel->setEnabled(false);
		break;
	case ito::userRoleAdministrator:
        ui.radioButton_roleAdmin->setEnabled(!m_showsStandardUser);
        ui.radioButton_roleDevel->setEnabled(!m_showsStandardUser);
		break;
	case ito::userRoleDeveloper:
        ui.radioButton_roleAdmin->setEnabled(false);
        ui.radioButton_roleDevel->setEnabled(!m_showsStandardUser);
		break;
	}

    // set the check state of all features
    ui.checkBox_devTools->setChecked(features & featDeveloper);
    ui.checkBox_fileSystem->setChecked(features & featFileSystem);
    ui.checkBox_userManag->setChecked(features & featUserManagement);
    ui.checkBox_addInManager->setChecked(features & featPlugins);
    ui.checkBox_editProperties->setChecked(features & featProperties);

    if (features & featConsoleReadWrite)
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

    // set the enable state of all features depending on the user role
    // and the features of the currently logged-in user
    char rightCmp = 0;

    // 1: current user has higher rights than edited user
    // -1: current user has less rights than edited user (can edit nothing)
    // 0: both users have the same rights
    if (currentUserRole == userRoleAdministrator)
    {
        rightCmp = 1; //admin can do everything
    }
    else if (currentUserRole == userRoleDeveloper)
    {
        switch (userRole)
        {
        default:
        case ito::userRoleBasic:
            rightCmp = 1;
            break;
        case ito::userRoleAdministrator:
            rightCmp = -1;
            break;
        case ito::userRoleDeveloper:
            rightCmp = 0;
            break;
        }
    }
    else // current user is basic
    {
        switch (userRole)
        {
        default:
        case ito::userRoleBasic:
            rightCmp = 0;
            break;
        case ito::userRoleAdministrator:
        case ito::userRoleDeveloper:
            rightCmp = -1;
            break;
        }
    }

	ui.lineEdit_password->setEnabled(rightCmp >= 0); // admin has the permission to change the passwort of the standard user
	ui.cmdAutoID->setEnabled(rightCmp >= 0);

    if (m_showsStandardUser)
    {
        rightCmp = -1; // it is not allowed to reduce features of the standard user
    }

    if (rightCmp != 0)
    {
        ui.checkBox_devTools->setEnabled(rightCmp > 0);
        ui.checkBox_fileSystem->setEnabled(rightCmp > 0);
        ui.checkBox_userManag->setEnabled(rightCmp > 0);
        ui.checkBox_addInManager->setEnabled(rightCmp > 0);
        ui.checkBox_editProperties->setEnabled(rightCmp > 0);
        ui.radioButton_consoleNormal->setEnabled(rightCmp > 0);
        ui.radioButton_consoleRO->setEnabled(rightCmp > 0);
        ui.radioButton_consoleOff->setEnabled(rightCmp > 0);
    }
    else
    {
        ui.checkBox_devTools->setEnabled(currentFeatures & featDeveloper);
        ui.checkBox_fileSystem->setEnabled(currentFeatures & featFileSystem);
        ui.checkBox_userManag->setEnabled(currentFeatures & featUserManagement);
        ui.checkBox_addInManager->setEnabled(currentFeatures & featPlugins);
        ui.checkBox_editProperties->setEnabled(currentFeatures & featProperties);
        ui.radioButton_consoleNormal->setEnabled(currentFeatures & (featConsoleReadWrite));
        ui.radioButton_consoleRO->setEnabled(currentFeatures & (featConsoleReadWrite | featConsoleRead));
        ui.radioButton_consoleOff->setEnabled(true);
    }

    // the startup scripts can only be edited if the currently logged-in user
    // has the featProperties
    ui.groupStartupScripts->setEnabled(currentFeatures & featProperties);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_lv_startUpScripts_currentRowChanged(int row)
{
    updateScriptButtons();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::updateScriptButtons()
{
    int currentRow = ui.lv_startUpScripts->currentRow();
    int rows = ui.lv_startUpScripts->count();
    ui.pb_removeScript->setEnabled(currentRow >= 0);
    ui.pb_downScript->setEnabled((currentRow >= 0) && (currentRow < (rows - 1)));
    ui.pb_upScript->setEnabled(currentRow > 0);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_pb_addScript_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Load python script"), QDir::currentPath(), tr("Python script (*.py)"));

    if (!filenames.empty())
    {
        QDir::setCurrent(QFileInfo(filenames.first()).path());
        QDir baseDir(QCoreApplication::applicationDirPath());

        foreach(QString filename, filenames)
        {
            if (ui.checkAddFileRel->isChecked())
            {
                filename = baseDir.relativeFilePath(filename);
            }

            if (ui.lv_startUpScripts->findItems(filename, Qt::MatchExactly).isEmpty())
            {
                ui.lv_startUpScripts->addItem(filename);
            }
        }

        updateScriptButtons();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_pb_removeScript_clicked()
{
    qDeleteAll(ui.lv_startUpScripts->selectedItems());
    updateScriptButtons();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_pb_downScript_clicked()
{
    int currentRow = ui.lv_startUpScripts->currentRow();
    int numRows = ui.lv_startUpScripts->count();

    if (currentRow < (numRows - 1))
    {
        QListWidgetItem *item = ui.lv_startUpScripts->item(currentRow);
        QString text = item->text();
        DELETE_AND_SET_NULL(item);
        ui.lv_startUpScripts->insertItem(currentRow + 1, text);
        ui.lv_startUpScripts->setCurrentRow(currentRow + 1);
        updateScriptButtons();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_pb_upScript_clicked()
{
    int currentRow = ui.lv_startUpScripts->currentRow();

    if (currentRow > 0)
    {
        QListWidgetItem *item = ui.lv_startUpScripts->item(currentRow);
        QString text = item->text();
        DELETE_AND_SET_NULL(item);
        ui.lv_startUpScripts->insertItem(currentRow - 1, text);
        ui.lv_startUpScripts->setCurrentRow(currentRow - 1);
        updateScriptButtons();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_cmdUseWindowsUser_clicked()
{
    ui.lineEdit_name->setText(m_osUser);

    if (m_fileName == "" && ui.cmdAutoID->isChecked())
    {
        ui.lineEdit_id->setText(clearName(m_osUser));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_lineEdit_name_textChanged(const QString &text)
{
    if (m_fileName == "" && ui.cmdAutoID->isChecked())
    {
        ui.lineEdit_id->setText(clearName(text));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogUserManagementEdit::on_cmdAutoID_toggled(bool checked)
{
    if (checked && m_fileName == "")
    {
        ui.lineEdit_id->setText(clearName(ui.lineEdit_name->text()));
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
