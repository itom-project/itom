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

#ifndef USEROGRANIZER_H
#define USEROGRANIZER_H

#include "../global.h"
#include <qobject.h>

namespace ito
{

class userOrganizer : QObject
{
    Q_OBJECT

    public:
        static userOrganizer * getInstance(void);
        static RetVal closeInstance(void);

        inline void setUserName(const QString userName) { m_userName = userName; }
        inline const QString getUserName() { return m_userName; }
        inline void setUserRole(const int role) { m_userRole = role; }
        void setUserRole(const QString role) 
        { 
            if (role == "user")
                m_userRole = 0;
            else if (role == "admin")
                m_userRole = 1;
            else
                m_userRole = 2;
        }
        inline int getUserRole() { return m_userRole; }
        QString getUserID(void);
        void setUiFlags(long flags) { m_enabledUI = flags; }
        long getUiFlags(void) { return m_enabledUI; }
        void setSettingsFile(QString &settingsFile) { m_settingsFile = settingsFile; }
        inline QString& getSettingsFile() { return m_settingsFile; };
        ito::RetVal userOrganizer::loadSettings(const QString defUserName);

private:
        userOrganizer(void);
        userOrganizer(userOrganizer  &/*copyConstr*/) : QObject() {}
        ~userOrganizer(void) {};
        static userOrganizer *m_pUserOrganizer;

        int m_userRole;  /*< type of user: 0: "dumb" user, 1: admin user, 2: developer */
        QString m_userName;  /*< id of current user */
        long m_enabledUI; /*< switch for enabeling and disabeling functions of itom */
        QString m_settingsFile;
};

} // namespace ito

#endif //USEROGRANIZER_H