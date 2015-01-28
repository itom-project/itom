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
#include "./models/UserModel.h"
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
    m_userRole(2),
    m_userName("ito"),
    m_features(allFeatures),
    m_settingsFile("") 
{
    AppManagement::setUserOrganizer(this);

    strConstFeatDeveloper = tr("Developer");
    strConstFeatFileSystem = tr("File System");
    strConstFeatUserManag = tr("User Management");
    strConstFeatPlugins = tr("Addin Manager (Plugins)");
    strConstFeatConsole = tr("Console");
    strConstFeatConsoleRO = tr("Console (Read Only)");
    strConstFeatProperties = tr("Properties");

    strConstRole = tr("Role");
    strConstRoleUser = tr("User");
    strConstRoleAdministrator = tr("Administrator");
    strConstRoleDeveloper = tr("Developer");
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
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, "itomSettings");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    UserModel curUserModel;

    QString settingsFile;
    QDir appDir(QCoreApplication::applicationDirPath());
    if (!appDir.cd("itomSettings"))
    {
        appDir.mkdir("itomSettings");
        appDir.cd("itomSettings");
    }
    QStringList iniList = appDir.entryList(QStringList("itom_*.ini"));

    foreach(QString iniFile, iniList) 
    {
        QSettings settings(QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QSettings::IniFormat);

        settings.beginGroup("ITOMIniFile");
        if (settings.contains("name"))
        {
            qDebug() << "found user ini file: " << iniFile;
            curUserModel.addUser(UserInfoStruct(QString(settings.value("name").toString()), iniFile.mid(5, iniFile.length() - 9), QDir::cleanPath(appDir.absoluteFilePath(iniFile)), QString(settings.value("role").toString())));
        }
        settings.endGroup();
    }

    if (curUserModel.rowCount() > 0) 
    {
        char foundDefUser = 0;

        curUserModel.addUser(UserInfoStruct(tr("Standard User"), "itom.ini", QDir::cleanPath(appDir.absoluteFilePath("itom.ini")), strConstRoleAdministrator));

        DialogSelectUser userDialog;
        userDialog.ui.userList->setModel(&curUserModel);
        userDialog.DialogInit(&curUserModel);
#if linux
        QString curSysUser(qgetenv("USER")); ///for MAc or Linux
#else
        QString curSysUser(qgetenv("USERNAME")); //for windows
#endif

        for (int curIdx = 0; curIdx < curUserModel.rowCount(); curIdx++)
        {
            QModelIndex midx = curUserModel.index(curIdx, 1);
            if (midx.isValid())
            {
                QString curUid(midx.data().toString());
                if (!defUserName.isEmpty())
                {
                    if (curUid == defUserName)
                    {
                        QModelIndex actIdx = curUserModel.index(curIdx, 0);
                        userDialog.ui.userList->setCurrentIndex(actIdx);
                        foundDefUser = 1;
                    }
                }
                else
                {
                    if (curUid == curSysUser)
                    {
                        QModelIndex actIdx = curUserModel.index(curIdx, 0);
                        userDialog.ui.userList->setCurrentIndex(actIdx);
                    }
                }
            }
        }

        if (foundDefUser == 0)
        {
            int ret = userDialog.exec();
            if (ret == 0)
            {
                return ito::retError;
            }

            QModelIndex curIdx = userDialog.ui.userList->currentIndex();
            QModelIndex fIdx = curUserModel.index(curIdx.row(), 3);
            settingsFile = QString(fIdx.data().toString());
        }
        else
        {
            settingsFile = QString("itom_").append(defUserName).append(".ini");
        }
        qDebug() << "settingsFile path: " << settingsFile;
        setSettingsFile(settingsFile);
        QSettings settings(settingsFile, QSettings::IniFormat);
        setUserName(settings.value("ITOMIniFile/name").toString());
        setUserRole(settings.value("ITOMIniFile/role").toString());

        setUiFlags((userFeatures)getFlagsFromFile());
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
        setSettingsFile(settingsFile);
        setUserRole("developer");
        setUiFlags((userFeatures)allFeatures);
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param fileName
    \return int
*/
int UserOrganizer::getFlagsFromFile(QString fileName)
{
    QString uid = getUserID(fileName);
    QCryptographicHash nameHash(QCryptographicHash::Sha1);
    nameHash.addData(uid.toLatin1().data(), uid.length());

    QSettings settings(fileName, QSettings::IniFormat);
    settings.beginGroup("ITOMIniFile");
    QByteArray fileFlags = settings.value("flags").toByteArray();
    settings.endGroup();

    if (fileFlags.count() == 0)
    {
        return allFeatures;
    }

    QByteArray res;
    for (int n = 0; n < nameHash.result().length(); n++)
    {
        res.append(fileFlags.at(n) ^ nameHash.result().at(n));
    }

    return res.toInt();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param flags
    \param iniFile
*/
void UserOrganizer::writeFlagsToFile(int flags, QString iniFile)
{
    QSettings settings(iniFile, QSettings::IniFormat);
    QString uid = getUserID(iniFile);
    settings.beginGroup("ITOMIniFile");
    QCryptographicHash nameHash(QCryptographicHash::Sha1);
    nameHash.addData(uid.toLatin1().data(), uid.length());

    QByteArray fileFlags;
    QByteArray qbaFlags = QByteArray::number(flags);
    for (int n = 0; n < nameHash.result().length(); n++)
    {
        if (n >= qbaFlags.length())
            fileFlags.append(nameHash.result().at(n));
        else
            fileFlags.append(qbaFlags.at(n) ^ nameHash.result().at(n));
    }
    settings.setValue("flags", fileFlags);
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \return QString
*/
QString UserOrganizer::getFeatureName(const userFeatures &feature) const
{
    switch(feature)
    {
        case featDeveloper:
            return strConstFeatDeveloper;
        case featFileSystem:
            return strConstFeatFileSystem;
        case featUserManag:
            return strConstFeatUserManag;
        case featPlugins:
            return strConstFeatPlugins;
        case featConsole:
            return strConstFeatConsole;
        case featConsoleRW:
            return strConstFeatConsoleRO;
        case featProperties:
            return strConstFeatProperties;
    }

    return "";
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \return QString
*/
QString UserOrganizer::getUserID(void) const
{
    QString fname = QFileInfo(m_settingsFile).baseName();
    fname = fname.right(fname.length() - 5);
    return fname;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param inifile
    \return QString
*/
QString UserOrganizer::getUserID(QString inifile) const
{
    QString fname = QFileInfo(inifile).baseName();
    fname = fname.right(fname.length() - 5);
    return fname;
}

//----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito