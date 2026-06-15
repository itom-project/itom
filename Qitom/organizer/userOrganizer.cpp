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

#include "AppManagement.h"
#include "userOrganizer.h"

#include "./ui/dialogSelectUser.h"

#include <qsettings.h>
#include <qdir.h>
#include <qdebug.h>
#include <qinputdialog.h>
#include <QCryptographicHash>



namespace ito
{

/*!
    \class UserOrganizer
    \brief class handling users and their rights
*/

//! global variable reference used to store AddInManager reference, as the AIM is singleton this variable is principally only
//! accessed by the class itself. Its value is return also by the getReference \ref AddInManager::method of AIM
UserOrganizer* UserOrganizer::m_pUserOrganizer = NULL;
//----------------------------------------------------------------------------------------------------------------------------------
UserOrganizer::UserOrganizer(void) :
    QObject(),
    m_userModel(new UserModel())
{
    AppManagement::setUserOrganizer(this);

    m_strConstStdUserName = tr("Standard User");
	m_strConstStdUserId = "itom.ini"; //no translation
}

//----------------------------------------------------------------------------------------------------------------------------------
UserOrganizer::~UserOrganizer(void)
{
    DELETE_AND_SET_NULL(m_userModel);
}

//----------------------------------------------------------------------------------------------------------------------------------
/** getInstance method, retrieves Instance of the userOrganizer (or opens it if no instance exists)
*   @return instance of the userOrganizer
*
*   This method returns the instance of the userOrganizer, i.e. if the userOrganizer has not been started, it is started then.
*   Otherwise the reference to the open userOrganizer is returned
*/
UserOrganizer * UserOrganizer::getInstance(void)
{
    if (UserOrganizer::m_pUserOrganizer == NULL)
    {
        UserOrganizer::m_pUserOrganizer = new ito::UserOrganizer();
    }
    return UserOrganizer::m_pUserOrganizer;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** closeInstance
*   @return ito::retOk
*
*   closes the instance of the userOrganizer - should only be called at the very closing of the main program
*/
RetVal UserOrganizer::closeInstance(void)
{
    if (UserOrganizer::m_pUserOrganizer)
    {
        DELETE_AND_SET_NULL(UserOrganizer::m_pUserOrganizer);
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param userId
    \return RetVal
*/
ito::RetVal UserOrganizer::loadSettings(const QString &userId)
{
    ito::RetVal retval = scanSettingFilesAndLoadModel();

    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        appDir.mkdir("itomSettings");
        appDir.cd("itomSettings");
    }

    QString settingsFile;

    int numUserFiles = m_userModel->rowCount();

	if (userId != "")
	{
		QModelIndex userIndex = m_userModel->getUser(userId);

		if (userIndex.isValid())
		{
			bool loadUser = false;

			if (m_userModel->hasPassword(userIndex))
			{
				bool ok;
				QString pw = QInputDialog::getText(NULL, tr("User Password"), tr("Enter the password for the user %1").arg(userId), QLineEdit::Password, "", &ok);
				if (!ok)
				{
					return ito::retError;
				}
				else if (!m_userModel->checkPassword(m_userModel->index(userIndex.row(), UserModel::umiPassword), pw))
				{
					return ito::RetVal::format(ito::retError, 0, "The password for user '%s' is wrong.", userId.toLatin1().data());
				}
				else
				{
					loadUser = true;
				}
			}
			else
			{
				loadUser = true;
			}

			if (loadUser)
			{
				QModelIndex fIdx = m_userModel->index(userIndex.row(), UserModel::umiIniFile);
				settingsFile = fIdx.data().toString();
				m_userModel->setCurrentUser(getUserIdFromSettingsFilename(settingsFile));
				qDebug() << "settingsFile path: " << settingsFile;
			}
		}
		else
		{
			retval += ito::RetVal::format(ito::retError, 0, "No setting file (itom_%s.ini) available for the desired user '%s'.", userId.toLatin1().data(), userId.toLatin1().data());
		}
	}
    else if (numUserFiles > 1 || !m_userModel->index(0, UserModel::umiPassword).data().toByteArray().isEmpty())
    {
        bool foundDefaultUser = false;

        DialogSelectUser userDialog(m_userModel);

        // User(name) variable is not necessarily OS dependend. So this will give us the best chance to find the actual user name http://stackoverflow.com/questions/26552517/get-system-username-in-qt
        QString curSysUser = qgetenv("USERNAME");
        if (curSysUser.isEmpty())
        {
            curSysUser = qgetenv("USER");
        }

        if (userId.isEmpty())
        {
            if (!userDialog.selectUser(curSysUser))
            {
                //no profile found with the current user
                userDialog.selectUser(m_lastOpenedUserName);
            }
        }
        else
        {
            foundDefaultUser = userDialog.selectUser(userId);
        }

        if (!foundDefaultUser)
        {
            if (userDialog.exec() == QDialog::Rejected)
            {
                return ito::retError;
            }

            QModelIndex curIdx = userDialog.selectedIndex();
            QModelIndex fIdx = m_userModel->index(curIdx.row(), UserModel::umiIniFile);
            settingsFile = fIdx.data().toString();
        }
        else
        {
            settingsFile = QString("itom_").append(userId).append(".ini");
        }

        qDebug() << "settingsFile path: " << settingsFile;
		m_userModel->setCurrentUser(getUserIdFromSettingsFilename(settingsFile));
    }
    else if (userId == "")
    {
        settingsFile = QDir::cleanPath(appDir.absoluteFilePath("itom.ini"));

        QFileInfo settingsFileInfo(settingsFile);

        if (settingsFileInfo.exists() == false)
        {
            //try to create itom.ini as a copy from itomDefault.ini
            QFile defaultIni(QDir::cleanPath(appDir.absoluteFilePath("itomDefault.ini")));
            if (defaultIni.exists())
            {
                if (!defaultIni.copy(appDir.absoluteFilePath("itom.ini")))
                {
                    qDebug() << "error creating itom.ini from itomDefault.ini";
					retval += ito::RetVal::format(ito::retError, 0, "Error creating the default setting file itom.ini for the standard user The template itomDefault.ini could not be copied to itom.ini.");
                }
            }
			else
			{
				qDebug() << "error creating itom.ini from itomDefault.ini";
				retval += ito::RetVal::format(ito::retError, 0, "Error creating the default setting file itom.ini for the standard user. The template itomDefault.ini does not exist.");
			}
        }

		if (!retval.containsError())
		{
			qDebug() << "settingsFile path: " << settingsFile;
            m_userModel->setCurrentUser(getUserIdFromSettingsFilename(settingsFile));
		}
    }
	else
	{
		retval += ito::RetVal::format(ito::retError, 0, "No setting file (itom_%s.ini) available for the desired user '%s'.", userId.toLatin1().data(), userId.toLatin1().data());
	}

    return retval;
}

//----------------------------------------------------------------------------------------------------------
ito::RetVal UserOrganizer::scanSettingFilesAndLoadModel()
{
    ito::RetVal retval;

    QString currentUserId = m_userModel->getUserId(m_userModel->currentUser());

    m_userModel->removeAllUsers();

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
    bool userExists;
    QString absfile;
    QDateTime lastModified;
    QDateTime youngestModificationDate;

    foreach(QString iniFile, iniList)
    {
        absfile = QDir::cleanPath(appDir.absoluteFilePath(iniFile));
        QSettings settings(absfile, QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        userExists = settings.contains("name");
        settings.endGroup();

        if (userExists)
        {
            qDebug() << "found user ini file: " << iniFile;
            UserInfoStruct uis;
            uis.iniFile = absfile;
            uis.standardUser = false;

            if (readUserDataFromFile(absfile, uis.name, uis.id, uis.features, uis.role, uis.password, lastModified) == ito::retOk)
            {
                if (youngestModificationDate.isNull() || (youngestModificationDate < lastModified))
                {
                    youngestModificationDate = lastModified;
                    m_lastOpenedUserName = uis.name;
                }

                m_userModel->addUser(uis);
            }
        }
    }

    // 09/02/15 ck changed default role to developer
    // 04/09/20 jk changed default role to administrator
    QString itomIniPath = QDir::cleanPath(appDir.absoluteFilePath("itom.ini"));
    QByteArray tmpArray;
    UserInfoStruct uis(m_strConstStdUserName, m_strConstStdUserId, itomIniPath, userRoleAdministrator, ~UserFeatures(), tmpArray, true);

    QFileInfo fi(itomIniPath);

    if (fi.exists())
    {
        QSettings settings(itomIniPath, QSettings::IniFormat);
        settings.beginGroup("ITOMIniFile");
        uis.password = settings.value("password").toByteArray();
    }

    if (fi.exists() && fi.lastModified() > youngestModificationDate)
    {
        youngestModificationDate = fi.lastModified();
        m_lastOpenedUserName = uis.name;
    }

    m_userModel->addUser(uis);
    m_userModel->setCurrentUser(currentUserId);

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal UserOrganizer::readUserDataFromFile(
    const QString &filename, QString &username, QString &uid, UserFeatures &features,
    UserRole &role, QByteArray &password, QDateTime &lastModified)
{
    ito::RetVal retval;
    QFileInfo fi(filename);

    if (fi.exists())
    {
        lastModified = fi.lastModified();

        QSettings settings(filename, QSettings::IniFormat);
        settings.beginGroup("ITOMIniFile");

        //username
        username = settings.value("name", m_strConstStdUserName).toString();

        //uid
        uid = getUserIdFromSettingsFilename(filename);

        //user type
        // 09/02/15 ck changed default role to developer
        // 04/09/20 jk changed default role to administrator
        QString roleStr = settings.value("role", "administrator").toString().toLower();

        if (roleStr == "administrator")
        {
            role = userRoleAdministrator;
        }
        else if (roleStr == "developer")
        {
            role = userRoleDeveloper;
        }
        else
        {
            role = userRoleBasic;
        }

        if (settings.contains("password"))
        {
            password = settings.value("password").toByteArray();
        }
        else
        {
            password = "";
        }

        //features
        QByteArray featureSha1;
        QCryptographicHash nameHash(QCryptographicHash::Sha1);
        nameHash.addData(uid.toLatin1().data(), uid.length());

        if (settings.contains("userFeatures"))
        {
            featureSha1 = settings.value("userFeatures").toByteArray();
        }
        else
        {
            //compatibility to old, deprecated setting keyword 'flags'
            featureSha1 = settings.value("flags").toByteArray();
        }
        settings.endGroup();

        if (featureSha1.count() == 0)
        {
            //if no flags or userFeatures are given, all features are permitted
            features = ~UserFeatures();
        }
        else
        {
            QByteArray res;
            QByteArray nameHash_ = nameHash.result();
            for (int n = 0; n < nameHash_.length(); n++)
            {
                res.append(featureSha1[n] ^ nameHash_[n]);
            }

            features = UserFeatures(res.toInt());
        }
    }
    else
    {
        retval += ito::RetVal::format(ito::retError, 0, tr("file '%s' does not exist").toLatin1().data(), filename.toLatin1().data());
        lastModified = QDateTime();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal UserOrganizer::writeUserDataToFile(
    const QString &username, const QString &uid, const UserFeatures &features,
    const UserRole &role, const QByteArray &password, const bool& standardUser /*false*/)
{
    ito::RetVal retval;
    QString filename;

    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        retval += ito::RetVal(ito::retError, 0, tr("itomSettings directory not found, aborting!").toLatin1().data());
    }
    else
    {
        if (standardUser)
        {
            filename = QDir::cleanPath(appDir.absoluteFilePath(QString("itom").append(".ini")));
        }
        else
        {
            filename = QDir::cleanPath(appDir.absoluteFilePath(QString("itom_").append(uid).append(".ini")));
        }
        QFileInfo fi(filename);

        if (fi.exists() == false)
        {
            QFile stdIniFile(QDir::cleanPath(appDir.absoluteFilePath("itomDefault.ini")));
            if (!stdIniFile.copy(filename))
            {
                retval += ito::RetVal(ito::retError, 0, tr("Could not copy standard itom ini file!").toLatin1().data());
            }
        }
    }

    if (!retval.containsError())
    {
        QSettings settings(filename, QSettings::IniFormat);
        settings.beginGroup("ITOMIniFile");

        settings.setValue("name", username);

        switch (role)
        {
        case userRoleDeveloper:
            settings.setValue("role", "developer");
            break;
        case userRoleAdministrator:
            settings.setValue("role", "administrator");
            break;
        default:
            settings.setValue("role", "user");
            break;
        }

        settings.setValue("password", password);

        QCryptographicHash nameHash(QCryptographicHash::Sha1);
        nameHash.addData(uid.toLatin1().data(), uid.length());

        QByteArray fileFlags;
        QByteArray qbaFlags = QByteArray::number((int)(features));
        for (int n = 0; n < nameHash.result().length(); n++)
        {
            if (n >= qbaFlags.length())
                fileFlags.append(nameHash.result().at(n));
            else
                fileFlags.append(qbaFlags.at(n) ^ nameHash.result().at(n));
        }
        settings.setValue("userFeatures", fileFlags);

        settings.endGroup();
    }

    retval += scanSettingFilesAndLoadModel();

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param inifile
    \return QString
*/
QString UserOrganizer::getUserIdFromSettingsFilename(const QString &iniFile) const
{
	QFileInfo fileInfo(iniFile);
	if (fileInfo.fileName() == "itom.ini")
	{
		return m_strConstStdUserId;
	}
	else
	{
		QString fname = fileInfo.baseName(); //without .ini
		return fname.right(fname.length() - QString("itom_").size());
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< returns the user name of the current user
const QString UserOrganizer::getCurrentUserName() const
{
	return m_userModel->getUserName(m_userModel->currentUser());
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< returns the role of the current user (user, developer, administrator).
/*
The role is only used by the three python methods itom.userIsUser, itom.userIsDeveloper, itom.userIsAdministrator
*/
ito::UserRole UserOrganizer::getCurrentUserRole() const
{
	return m_userModel->getUserRole(m_userModel->currentUser());
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< returns the unique ID of the current user
QString UserOrganizer::getCurrentUserId(void) const
{
	return m_userModel->getUserId(m_userModel->currentUser());
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< returns the available features for the current user
UserFeatures UserOrganizer::getCurrentUserFeatures(void) const
{
	return m_userModel->getUserFeatures(m_userModel->currentUser());
}

//----------------------------------------------------------------------------------------------------------------------------------
bool UserOrganizer::currentUserHasFeature(const UserFeature &feature)
{
	UserFeatures features = m_userModel->getUserFeatures(m_userModel->currentUser());
	return features.testFlag(feature);
}

//----------------------------------------------------------------------------------------------------------------------------------
QString UserOrganizer::getCurrentUserSettingsFile() const
{
	return m_userModel->getUserSettingsFile(m_userModel->currentUser());
}

//----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito
