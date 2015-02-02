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

#ifndef USERMODEL_H
#define USERMODEL_H

#include <qabstractitemmodel.h>

namespace ito 
{

enum UserRole
{
    userRoleBasic = 0,
    userRoleAdministrator = 1,
    userRoleDeveloper = 2
};   

enum UserFeature
{
    featDeveloper           =   1,
    featFileSystem          =   2,
    featUserManag           =   4,
    featPlugins             =   8,
    featConsoleRead         =   16,
    featConsoleReadWrite    =   32,
    featProperties          =   64
};

Q_DECLARE_FLAGS(UserFeatures, UserFeature)

/*!
    \class UserInfo
    \brief holds the relevant user information
*/
struct UserInfoStruct
{
    UserInfoStruct() {};
    UserInfoStruct(const QString &sname, const QString &suid, const QString siniFile, UserRole srole, UserFeatures sfeatures, bool sStandardUser) : name(sname), id(suid), iniFile(siniFile), role(srole), features(sfeatures), standardUser(sStandardUser) {}
    QString name;
    QString id;
    QString iniFile;
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
            umiFeatures = 4
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

    private:
        QList<QString> m_headers;               //!<  string list of names of column headers
        QList<QVariant> m_alignment;            //!<  list of alignments for the corresponding headers
        QList<UserInfoStruct> m_userInfo;     //!<  list with user information
};
}

Q_DECLARE_METATYPE(ito::UserRole);
Q_DECLARE_METATYPE(ito::UserFeatures);
Q_DECLARE_METATYPE(ito::UserFeature);

#endif //USERMODEL_H
