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

#include "AppManagement.h"
#include "userOrganizer.h"

#include "./ui/dialogSelectUser.h"

#include <qsettings.h>
#include <qdir.h>
#include <qdebug.h>
#include <QCryptographicHash>

/*!
    \class userOrganizer
    \brief class handling users and their rights
*/

//! global variable reference used to store AddInManager reference, as the AIM is singleton this variable is principally only
//! accessed by the class itself. Its value is return also by the getReference \ref AddInManager::method of AIM
ito::UserOrganizer* ito::UserOrganizer::m_pUserOrganizer = NULL;

namespace ito 
{

//! userOrganizer implementation
//----------------------------------------------------------------------------------------------------------------------------------
UserOrganizer::UserOrganizer(void) :
    QObject(),
    // 09/02/15 ck changed default role to developer
    m_userRole(userRoleDeveloper),
    m_features(~UserFeatures(0)),
    m_settingsFile(""),
    m_userModel(new UserModel())
{
    AppManagement::setUserOrganizer(this);

    m_strConstStdUser = tr("Standard User");
    m_userName = m_strConstStdUser;
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

    \param defUserName
    \return RetVal
*/
ito::RetVal UserOrganizer::loadSettings(const QString &defUserName)
{
    ito::RetVal retval = scanSettingFilesAndLoadModel();

    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        appDir.mkdir("itomSettings");
        appDir.cd("itomSettings");
    }

    QString settingsFile;

    if (m_userModel->rowCount() > 1) 
    {
        bool foundDefUser = false;

        DialogSelectUser userDialog(m_userModel);

#if linux
        QString curSysUser(qgetenv("USER")); ///for MAc or Linux
#else
        QString curSysUser(qgetenv("USERNAME")); //for windows
#endif

        if (defUserName.isEmpty())
        {
            userDialog.selectUser(curSysUser);
        }
        else
        {
            foundDefUser = userDialog.selectUser(defUserName);
        }

        if (!foundDefUser)
        {
            if (userDialog.exec() == QDialog::Rejected)
            {
                return ito::retError;
            }

            QModelIndex curIdx = userDialog.selectedIndex();
            QModelIndex fIdx = m_userModel->index(curIdx.row(), 3);
            settingsFile = fIdx.data().toString();
        }
        else
        {
            settingsFile = QString("itom_").append(defUserName).append(".ini");
        }

        qDebug() << "settingsFile path: " << settingsFile;
        m_settingsFile = settingsFile;

        QString uid;
        retval += readUserDataFromFile(settingsFile, m_userName, uid, m_features, m_userRole);
    }
    else
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
                }
            }
        }

        qDebug() << "settingsFile path: " << settingsFile;
        m_settingsFile = settingsFile;
        m_userName = m_strConstStdUser;
//        m_userRole = userRoleAdministrator;
        // 09/02/15 ck changed default role to developer
        m_userRole = userRoleDeveloper;
        m_features = ~UserFeatures();
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------
ito::RetVal UserOrganizer::scanSettingFilesAndLoadModel()
{
    ito::RetVal retval;
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
            if (readUserDataFromFile(absfile, uis.name, uis.id, uis.features, uis.role) == ito::retOk)
            {
                m_userModel->addUser(uis);
            }
        }
    }

    // 09/02/15 ck changed default role to developer
    UserInfoStruct uis(m_strConstStdUser, "itom.ini", QDir::cleanPath(appDir.absoluteFilePath("itom.ini")), userRoleDeveloper, ~UserFeatures(), true);
    m_userModel->addUser(uis);

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal UserOrganizer::readUserDataFromFile(const QString &filename, QString &username, QString &uid, UserFeatures &features, UserRole &role)
{
    ito::RetVal retval;
    QFileInfo fi(filename);

    if (fi.exists())
    {
        QSettings settings(filename, QSettings::IniFormat);
        settings.beginGroup("ITOMIniFile");

        //username
        username = settings.value("name", m_strConstStdUser).toString();

        //uid
        uid = getUserID(filename);

        //user type
        // 09/02/15 ck changed default role to developer
        QString roleStr = settings.value("role", "developer").toString().toLower();
        if (roleStr == "developer")
        {
            role = userRoleDeveloper;
        }
        else if (roleStr == "administrator")
        {
            role = userRoleAdministrator;
        }
        else
        {
            role = userRoleBasic;
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
        retval += ito::RetVal::format(ito::retError, 0, "file '%s' does not exist", filename.toLatin1().data());
    }
    
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal UserOrganizer::writeUserDataToFile(const QString &username, const QString &uid, const UserFeatures &features, const UserRole &role)
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
        filename = QDir::cleanPath(appDir.absoluteFilePath(QString("itom_").append(uid).append(".ini")));
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

    \return QString
*/
QString UserOrganizer::getUserID(void) const
{
    QString fname = QFileInfo(m_settingsFile).baseName();
    return fname.right(fname.length() - 5);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param inifile
    \return QString
*/
QString UserOrganizer::getUserID(const QString &iniFile) const
{
    QString fname = QFileInfo(iniFile).baseName();
    return fname.right(fname.length() - 5);
}

//----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito