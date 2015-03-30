/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut für Technische Optik (ITO),
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

#include "pipManager.h"

using namespace ito;

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor
*
*   contructor, creating column headers for the tree view
*/
PipManager::PipManager(QObject *parent /*= 0*/) :
    QAbstractItemModel(parent)
{
    m_headers << tr("Name") << tr("Version") << tr("Location") << tr("Requires") << tr("Status");
    m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft);
}

//----------------------------------------------------------------------------------------------------------------------------------
/** destructor - clean up, clear header and alignment list
*
*/
PipManager::~PipManager()
{
}
//----------------------------------------------------------------------------------------------------------------------------------
/** return parent element
*   @param [in] index   the element's index for which the parent should be returned
*   @return     the parent element. 
*
*/
QModelIndex PipManager::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return number of rows
*   @param [in] parent parent of current item
*   @return     returns number of users
*/
int PipManager::rowCount(const QModelIndex &parent) const
{
    return m_pythonPackages.length();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
*
*/
QVariant PipManager::headerData(int section, Qt::Orientation orientation, int role) const
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
*/
QVariant PipManager::data(const QModelIndex &index, int role) const
{
    if(!index.isValid())
    {
        return QVariant();
    }
 
    if(role == Qt::DisplayRole)
    {
        const PythonPackage &package = m_pythonPackages[index.row()];
        switch (index.column())
        {
            case idxName:
                return package.m_name;
            case idxVersion:
                return package.m_version;
            case idxLocation:
                return package.m_location;
            case idxRequires:
                return package.m_requires;
            case idxStatus:
                {
                    if (package.m_status == PythonPackage::Uptodate)
                    {
                        return tr("up to date");
                    }
                    else if (package.m_status == PythonPackage::Outdated)
                    {
                        return tr("new version %s available").arg(package.m_newVersion);
                    }
                    else
                    {
                        return tr("unknown");
                    }
                }
            default:
                return QVariant();
        }
    }
    
    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return column count
*   @param [in] parent parent of current item
*   @return     2 for child elements (instances) and the header size for root elements (plugins)
*/
int PipManager::columnCount(const QModelIndex & /*parent*/) const
{
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
QModelIndex PipManager::index(int row, int column, const QModelIndex &parent) const
{
    if(parent.isValid() || row < 0 || row >= m_pythonPackages.length() || column < 0 || column >= m_headers.size())
    {
        return QModelIndex();
    }
    
    return createIndex(row, column);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PipManager::isPipStarted() const
{
    return m_pipProcess.pid() != 0;
}