/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef USEROGRANIZER_H
#define USEROGRANIZER_H

#include "../global.h"
#include <qobject.h>
#include <qdatetime.h>
#include <qhash.h>
#include <qdir.h>
#include <qcoreapplication.h>

#include "models/UserModel.h"

namespace ito
{

class UserOrganizer : public QObject
{
    Q_OBJECT

    public:
        static UserOrganizer * getInstance(void);
        static RetVal closeInstance(void);

        //!< returns the user name of the current user
        inline const QString getUserName() const { return m_userName; }

        //!< returns the role of the current user (user, developer, administrator).
        /*
        The role is only used by the three python methods itom.userIsUser, itom.userIsDeveloper, itom.userIsAdministrator
        */
        inline ito::UserRole getUserRole() const { return m_userRole; }

        //!< returns the unique ID of the current user
        QString getUserID(void) const;

        inline UserModel* getUserModel() const { return m_userModel; }
        

        ito::RetVal readUserDataFromFile(const QString &filename, QString &username, QString &uid, UserFeatures &features, 
            UserRole &role, QByteArray &password, QDateTime &lastModified);
        ito::RetVal writeUserDataToFile(const QString &username, const QString &uid, const UserFeatures &features, 
            const UserRole &role, const QByteArray &password);

        UserFeatures getUserFeatures(void) const { return m_features; }

        inline QString getSettingsFile(const QString uid = "") const 
        {
            if (uid == "")
            {
                return m_settingsFile;
            }
            else
            {
                QDir appDir(QCoreApplication::applicationDirPath());
                if (!appDir.cd("itomSettings"))
                {
                    return "";
                }
                else
                {
                    QString filename = QDir::cleanPath(appDir.absoluteFilePath(QString("itom_").append(uid).append(".ini")));
                    QFileInfo fi(filename);

                    if (fi.exists() == false)
                    {
                        QFile stdIniFile(QDir::cleanPath(appDir.absoluteFilePath("itomDefault.ini")));
                        if (!stdIniFile.copy(filename))
                        {
                            return "";
                        }
                    }
                    return filename;
                }
            }
        }
        ito::RetVal loadSettings(const QString &defUserName);
        
        bool hasFeature(UserFeature feature)
        {
            return m_features.testFlag(feature);
        }

    private:
        UserOrganizer(void);
        UserOrganizer(UserOrganizer  &/*copyConstr*/) : QObject() {}
        ~UserOrganizer(void);
        static UserOrganizer *m_pUserOrganizer;

        QString getUserID(const QString &iniFile) const;
        ito::RetVal scanSettingFilesAndLoadModel();

        UserRole m_userRole;  /*< type of user: 0: "dumb" user, 1: admin user, 2: developer */
        QString m_userName;  /*< id of current user */
        UserFeatures m_features; /*< switch for enabeling and disabeling functions of itom */
        QString m_settingsFile;
        QByteArray m_password;

        QString m_strConstStdUser;
        QString m_lastOpenedUserName;

        UserModel *m_userModel;
        
};

} // namespace ito

#endif //USEROGRANIZER_H
