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

#include "UserModel.h"
#include <qicon.h>

using namespace ito;

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
UserModel::UserModel()
{
    m_headers << tr("Name") << tr("Id") << tr("role") << tr("iniFile");
    m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft);
}

//----------------------------------------------------------------------------------------------------------------------------------
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
//----------------------------------------------------------------------------------------------------------------------------------
/** return parent element
*   @param [in] index   the element's index for which the parent should be returned
*   @return     the parent element. 
*
*/
QModelIndex UserModel::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return number of rows
*   @param [in] parent parent of current item
*   @return     returns number of users
*/
int UserModel::rowCount(const QModelIndex &parent) const
{
    return m_userInfo.length();
}

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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
            case 0:
                temp = m_userInfo.at(index.row()).name;
                return QVariant(temp);
            case 1:
                return m_userInfo.at(index.row()).id;
            case 2:
                return m_userInfo.at(index.row()).role;
            case 3:
                return m_userInfo.at(index.row()).iniFile;
        }
    }

    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return column count
*   @param [in] parent parent of current item
*   @return     2 for child elements (instances) and the header size for root elements (plugins)
*/
int UserModel::columnCount(const QModelIndex & /*parent*/) const
{
//    return 1;
    return m_headers.size();
}

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
