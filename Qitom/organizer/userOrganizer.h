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
		const QString getCurrentUserName() const;

        //!< returns the role of the current user (user, developer, administrator).
        /*
        The role is only used by the three python methods itom.userIsUser, itom.userIsDeveloper, itom.userIsAdministrator
        */
		ito::UserRole getCurrentUserRole() const;

        //!< returns the unique ID of the current user
        QString getCurrentUserId() const;

		//!< returns the available features for the current user
		UserFeatures getCurrentUserFeatures() const;

		QString getCurrentUserSettingsFile() const;

		bool currentUserHasFeature(const UserFeature &feature);

        inline UserModel* getUserModel() const { return m_userModel; }

        ito::RetVal readUserDataFromFile(const QString &filename, QString &username, QString &uid, UserFeatures &features,
            UserRole &role, QByteArray &password, QDateTime &lastModified);

        ito::RetVal writeUserDataToFile(const QString &username, const QString &uid, const UserFeatures &features,
            const UserRole &role, const QByteArray &password, const bool &standardUser = false);

        ito::RetVal loadSettings(const QString &userId); //use an empty userId to get the selection dialog of select the standard user

    private:
        UserOrganizer(void);
        UserOrganizer(UserOrganizer  &/*copyConstr*/) : QObject() {}
        ~UserOrganizer(void);
        static UserOrganizer *m_pUserOrganizer;

        QString getUserIdFromSettingsFilename(const QString &iniFile) const;
        ito::RetVal scanSettingFilesAndLoadModel();

        QString m_strConstStdUserName;
		QString m_strConstStdUserId;
        QString m_lastOpenedUserName;

        UserModel *m_userModel;

};

} // namespace ito

#endif //USEROGRANIZER_H
