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

#ifndef USERMODEL_H
#define USERMODEL_H

#include <qabstractitemmodel.h>

namespace ito
{

//! Enumeration that defines some user roles
enum UserRole
{
    userRoleBasic = 0,         /* basic user with lowest rights */
    userRoleDeveloper = 1,     /* developer user with many rights */
    userRoleAdministrator = 2, /* most advanced user with all rights. */

};

//! Enumeration that defines some feature permissions for a user.
enum UserFeature
{
    featDeveloper           =   0x01, /* the user has the right to create, open, view or edit python scripts */
    featFileSystem          =   0x02,
    featUserManagement      =   0x04,
    featPlugins             =   0x08,
    featConsoleRead         =   0x10,
    featConsoleReadWrite    =   0x20,
    featProperties          =   0x40
};

Q_DECLARE_FLAGS(UserFeatures, UserFeature)

/*!
    \class UserInfo
    \brief holds the relevant user information
*/
struct UserInfoStruct
{
    UserInfoStruct() {};
    UserInfoStruct(const QString &sname, const QString &suid, const QString siniFile, UserRole srole,
        UserFeatures sfeatures, QByteArray &spassword, bool sStandardUser)
        : name(sname), id(suid), iniFile(siniFile), role(srole), password(spassword),
        features(sfeatures), standardUser(sStandardUser) {}
    QString name;
    QString id;
    QString iniFile;
    QByteArray password;
    UserRole role;
    UserFeatures features;
    bool standardUser;
};

/** @class UserModel
*   @brief class for for visualizing the available users
*
*   The UserModel is used in the initially shown user list. It contains the userId (which is the user name part of the ini-file name),
*   the plain text user name and the ini-file.
*/
class UserModel : public QAbstractItemModel
{
    Q_OBJECT

    public:
        UserModel(/*const QString &data, QObject *parent = 0*/);
        ~UserModel();

        enum UserModelIndex
        {
            umiName = 0,
            umiId = 1,
            umiRole = 2,
            umiIniFile = 3,
            umiFeatures = 4,
            umiPassword = 5
        };

        QString getRoleName(const UserRole &role) const;
        QString getFeatureName(const UserFeature &feature) const;

        QVariant data(const QModelIndex &index, int role) const;
        QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
        QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
        QModelIndex parent(const QModelIndex &index) const;
        int rowCount(const QModelIndex &parent = QModelIndex()) const;
        int columnCount(const QModelIndex &parent = QModelIndex()) const;

        int addUser(const UserInfoStruct &newUser);
        void removeAllUsers();
        bool removeUser(const QModelIndex &index);

		QModelIndex getUser(const QString &userId) const;
		bool hasPassword(const QModelIndex &index) const;
		bool checkPassword(const QModelIndex &index, const QString &password) const;
		QString getUserName(const QModelIndex &index) const;
		QString getUserId(const QModelIndex &index) const;
		UserRole getUserRole(const QModelIndex &index) const;
		UserFeatures getUserFeatures(const QModelIndex &index) const;
		QString getUserSettingsFile(const QModelIndex &index) const;
        QModelIndex currentUser() const { return m_currentUser; }
        bool setCurrentUser(const QString &userId);

    private:
        QList<QString> m_headers;               //!<  string list of names of column headers
        QList<QVariant> m_alignment;            //!<  list of alignments for the corresponding headers
        QList<UserInfoStruct> m_userInfo;       //!<  list with user information
        QModelIndex m_currentUser;              //!< the model index of the currently logged-in user
};
} //end namespace ito

Q_DECLARE_METATYPE(ito::UserRole);
Q_DECLARE_METATYPE(ito::UserFeatures);
Q_DECLARE_METATYPE(ito::UserFeature);



#endif //USERMODEL_H
