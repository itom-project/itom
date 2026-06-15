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

#include "UserModel.h"
#include <qicon.h>
#include <qcryptographichash.h>

namespace ito
{

//-------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
UserModel::UserModel()
{
    m_headers << tr("Name") << tr("Id") << tr("role")
              << tr("iniFile") << tr("features") << tr("password");
    m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft)
                << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft);
}

//-------------------------------------------------------------------------------------
/** destructor - clean up, clear header and alignment list
*
*/
UserModel::~UserModel()
{
    m_headers.clear();
    m_alignment.clear();
    m_userInfo.clear();
    return;
}
//-------------------------------------------------------------------------------------
/** return parent element
*   @param [in] index   the element's index for which the parent should be returned
*   @return     the parent element.
*
*/
QModelIndex UserModel::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//-------------------------------------------------------------------------------------
/** return number of rows
*   @param [in] parent parent of current item
*   @return     returns number of users
*/
int UserModel::rowCount(const QModelIndex &parent) const
{
    return m_userInfo.length();
}

//-------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
*
*/
QVariant UserModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal )
    {
        if (section >= 0 && section < m_headers.size())
        {
            return m_headers.at(section);
        }
        return QVariant();
    }
    return QVariant();
}

//-------------------------------------------------------------------------------------
/** return data elements for a given row
*   @param [in] index   index for which the data elements should be delivered
*   @param [in] role    the current role of the model
*   @return data of the selected element, depending on the element's row and column (passed in index.row and index.column)
*
*   This method is actually used to fill the tree view. It returns the data for the selected element, depending as well on the
*   column of the selected element, passed in index.column. The method here is divded into two parts. The first one handels requests
*   for root elements (plugins) the second one is used for child elements (instances of plugins).
*/
QVariant UserModel::data(const QModelIndex &index, int role) const
{
    QString temp;

    if(!index.isValid())
    {
        return QVariant();
    }

    if(role == Qt::DisplayRole)
    {
        switch (index.column())
        {
            case umiName:
                temp = m_userInfo.at(index.row()).name;
                return QVariant(temp);
            case umiId:
                return m_userInfo.at(index.row()).id;
            case umiRole:
                return QVariant::fromValue<UserRole>(m_userInfo.at(index.row()).role);
            case umiIniFile:
                return m_userInfo.at(index.row()).iniFile;
            case umiFeatures:
                return QVariant::fromValue<UserFeatures>(m_userInfo.at(index.row()).features);
            case umiPassword:
                return m_userInfo.at(index.row()).password;
        }
    }
    else if (role == Qt::EditRole)
    {
        if (m_userInfo[index.row()].standardUser)
        {
            return QVariant(); //standard user is not editible
        }
        else
        {
            switch (index.column())
            {
                case umiName:
                    temp = m_userInfo.at(index.row()).name;
                    return QVariant(temp);
                case umiId:
                    return m_userInfo.at(index.row()).id;
                case umiRole:
                    return QVariant::fromValue<UserRole>(m_userInfo.at(index.row()).role);
                case umiIniFile:
                    return m_userInfo.at(index.row()).iniFile;
                case umiFeatures:
                    return QVariant::fromValue<UserFeatures>(m_userInfo.at(index.row()).features);
                case umiPassword:
                    return m_userInfo.at(index.row()).password;
            }
        }
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------
/** return column count
*   @param [in] parent parent of current item
*   @return     2 for child elements (instances) and the header size for root elements (plugins)
*/
int UserModel::columnCount(const QModelIndex & /*parent*/) const
{
//    return 1;
    return m_headers.size();
}

//-------------------------------------------------------------------------------------
/** return current index element
*   @param [in] row row of current element
*   @param [in] column column of current element
*   @param [in] parent  parent of current element
*   @return QModelIndex - element at current index
*
*   returns the passed row as index, as the users are arranged in a simple one dimensional list
*/
QModelIndex UserModel::index(int row, int column, const QModelIndex &parent) const
{
    if(parent.isValid() || row < 0 || row >= m_userInfo.length() || column < 0 || column >= m_headers.size())
    {
        return QModelIndex();
    }

    return createIndex(row, column);
}

//-------------------------------------------------------------------------------------
/** Adds a user to the current model
*   @param newUser Struct containing new User
*   @return QModelIndex - index of the position in the list where the user was added
*
*   returns the passed row as index, as the users are arranged in a simple one dimensional list
*/
int UserModel::addUser(const UserInfoStruct &newUser)
{
    foreach (const UserInfoStruct &userInfo, m_userInfo)
    {
        if (userInfo.id == newUser.id)
            return -1;
    }

    beginInsertRows(QModelIndex(), m_userInfo.length(), m_userInfo.length());
    m_userInfo.append(newUser);
    endInsertRows();

    return 0;
}

//-------------------------------------------------------------------------------------
bool UserModel::removeUser(const QModelIndex &index)
{
    if (index.row() >= 0 &&
        index.row() < rowCount() &&
        index.row() != m_currentUser.row())
    {
        beginRemoveRows(parent(index), index.row(), index.row());
        m_userInfo.removeAt(index.row());

        if (index.row() < m_currentUser.row())
        {
            m_currentUser = this->index(m_currentUser.row() - 1, 0);
        }

        endRemoveRows();

        return true;
    }

    return false;
}

//-------------------------------------------------------------------------------------
void UserModel::removeAllUsers()
{
    beginResetModel();
    m_userInfo.clear();
    m_currentUser = QModelIndex();
    endResetModel();
}

//-------------------------------------------------------------------------------------
/* Return the model index to the first column of the user with the given userId.
Returns an invalid QModelIndex if the user could not be found.
*/
QModelIndex UserModel::getUser(const QString &userId) const
{
    int row = -1;

    for (int row = 0; row < m_userInfo.size(); ++row)
    {
        if (m_userInfo[row].id == userId)
        {
            return createIndex(row, 0);
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------
/*
Returns true if the user with the given index (the column value of this index
is ignored) has a password set, else false.
*/
bool UserModel::hasPassword(const QModelIndex &index) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        return m_userInfo[index.row()].password != "";
    }
    else
    {
        return false;
    }
}

//-------------------------------------------------------------------------------------
bool UserModel::checkPassword(const QModelIndex &index, const QString &password) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        QByteArray passwordSha512 = m_userInfo[index.row()].password;
        if (passwordSha512 == "" && password == "")
        {
            return true;
        }
        else
        {
            QByteArray newPasswordSha512 = QCryptographicHash::hash(password.toUtf8(), QCryptographicHash::Sha3_512);
            return passwordSha512 == newPasswordSha512;
        }
    }
    else
    {
        return false;
    }
}

//-------------------------------------------------------------------------------------
QString UserModel::getUserName(const QModelIndex &index) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        return m_userInfo[index.row()].name;
    }
    else
    {
        return QString();
    }
}

//-------------------------------------------------------------------------------------
QString UserModel::getUserId(const QModelIndex &index) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        return m_userInfo[index.row()].id;
    }
    else
    {
        return QString();
    }
}

//-------------------------------------------------------------------------------------
UserRole UserModel::getUserRole(const QModelIndex &index) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        return m_userInfo[index.row()].role;
    }
    else
    {
        return userRoleBasic;
    }
}

//-------------------------------------------------------------------------------------
UserFeatures UserModel::getUserFeatures(const QModelIndex &index) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        return m_userInfo[index.row()].features;
    }
    else
    {
        return UserFeatures();
    }
}

//-------------------------------------------------------------------------------------
QString UserModel::getUserSettingsFile(const QModelIndex &index) const
{
    if (index.isValid() && index.row() >= 0 && index.row() < m_userInfo.size())
    {
        return m_userInfo[index.row()].iniFile;
    }
    else
    {
        return "";
    }
}

//-------------------------------------------------------------------------------------
QString UserModel::getRoleName(const UserRole &role) const
{
    switch (role)
    {
    case userRoleDeveloper:
        return tr("Developer");
    case userRoleAdministrator:
        return tr("Administrator");
    default:
        return tr("User");
    }
}

//-------------------------------------------------------------------------------------
QString UserModel::getFeatureName(const UserFeature &feature) const
{
    switch(feature)
    {
        case featDeveloper:
            return tr("Developer");
        case featFileSystem:
            return tr("File System");
        case featUserManagement:
            return tr("User Management");
        case featPlugins:
            return tr("Addin Manager (Plugins)");
        case featConsoleReadWrite:
            return tr("Console");
        case featConsoleRead:
            return tr("Console (Read Only)");
        case featProperties:
            return tr("Properties");
    }

    return "";
}

//-------------------------------------------------------------------------------------
bool UserModel::setCurrentUser(const QString &userId)
{
    m_currentUser = getUser(userId);

    return m_currentUser.isValid();
}

} //end namespace ito
