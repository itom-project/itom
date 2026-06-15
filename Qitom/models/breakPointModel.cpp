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

#include "breakPointModel.h"

#include "../AppManagement.h"

#include <algorithm>


#include <qfileinfo.h>
#include <qdir.h>
#include <qsize.h>
#include <qsettings.h>
#include <qdatastream.h>
#include <qicon.h>
#include <qsize.h>
#include <qhash.h>

namespace ito
{
    QDataStream &operator<<(QDataStream &out, const BreakPointItem &obj)
    {
        out << obj.filename << obj.lineIdx << obj.condition << obj.conditioned << obj.enabled << obj.ignoreCount << obj.temporary;
        return out;
    }

    QDataStream &operator>>(QDataStream &in, BreakPointItem &obj)
    {
        obj.pythonDbgBpNumber = -1; //invalid, only valid if python is deubugging
        in >> obj.filename >> obj.lineIdx >> obj.condition >> obj.conditioned >> obj.enabled >> obj.ignoreCount >> obj.temporary;
        return in;
    }

/*!
    \class BreakPointModel
    \brief model for management of all breakpoints. This model will be displayed by a viewer-widget in the main window
*/

//-------------------------------------------------------------------------------------------------------
//! constructor
/*!
    initializes headers and its alignment
*/
BreakPointModel::BreakPointModel() : QAbstractItemModel()
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    // must not be called any more in Qt6, since this is automatically done then.
    qRegisterMetaTypeStreamOperators<ito::BreakPointItem>("BreakPointItem");
#else
    qRegisterMetaType<ito::BreakPointItem>("BreakPointItem");
#endif

    m_headers   << tr("Line")          << tr("Condition")         << tr("Temporary")            << tr("Enabled")              << tr("Ignore count");
    m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignRight) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignHCenter) << QVariant(Qt::AlignHCenter) << QVariant(Qt::AlignRight);

#ifndef WIN32
    m_filenameCaseSensitivity = Qt::CaseSensitive;
#else
    m_filenameCaseSensitivity = Qt::CaseInsensitive;
#endif
}

//-------------------------------------------------------------------------------------------------------
//! destructor
BreakPointModel::~BreakPointModel()
{
    m_headers.clear();
    m_alignment.clear();
    m_breakpoints.clear();
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//! Saves the breakpoint model into the settings
/*!

*/
RetVal BreakPointModel::saveState()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    int counter = 0;
    settings.beginGroup("BreakPointModel");

    settings.beginWriteArray("breakPointStorage");

    foreach(const BreakPointItem &item, m_breakpoints)
    {
        QFileInfo fi(item.filename);

        if (fi.exists())
        {
            settings.setArrayIndex(counter++);
            settings.setValue("item", QVariant::fromValue<BreakPointItem>(item));
        }
    }

    settings.endArray();
    settings.endGroup();

    return ito::retOk;
}

//-------------------------------------------------------------------------------------------------------
//! Restores the breakpoint model from the settings
/*!

*/
RetVal BreakPointModel::restoreState()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("BreakPointModel");

    int size = settings.beginReadArray("breakPointStorage");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        QVariant v = settings.value("item");

        if (v.canConvert<BreakPointItem>())
        {
            BreakPointItem item = v.value<BreakPointItem>();

            QFileInfo fi(item.filename);
            if (fi.exists())
            {
                addBreakPoint(item);
            }
        }
    }

    settings.endArray();
    settings.endGroup();

    return ito::retOk;
}

//-------------------------------------------------------------------------------------------------------
//! adds given breakpoint to model
/*!
    if added, the signal breakPointAdded is emitted.

    \param[in] bp Breakpoint of type BreakPointItem
    \return retOk
*/
RetVal BreakPointModel::addBreakPoint(BreakPointItem bp)
{
#ifndef WIN32
    //in linux bp.filename is casesensitive and must fit
#else
    //correct incoming filename with respect to the real case-sensitive version:
    QFileInfo fi(bp.filename);
    QDir folder = fi.dir();
    QString filenameCaseInsensitive = fi.fileName();
    QStringList filesInDirectory = folder.entryList(QDir::Files);
    foreach (const QString &file, filesInDirectory)
    {
        if (QString::compare(file, filenameCaseInsensitive, Qt::CaseInsensitive) == 0)
        {
            bp.filename =  folder.absoluteFilePath(file);
            break;
        }
    }
#endif

    if (!m_scriptFiles.contains(bp.filename, m_filenameCaseSensitivity))
    {
        // Parent does not exist yet, so create it
        beginInsertRows(QModelIndex(), m_scriptFiles.size(), m_scriptFiles.size() + 1);
        m_scriptFiles.append(bp.filename);
        endInsertRows();
    }

    // now append the bp to the file:
    QModelIndex parent = getFilenameModelIndex(bp.filename);

    if (parent.isValid())
    {
        int nrOfExistingBreakpoints = nrOfBreakpointsInFile(parent.row());
        beginInsertRows(parent, nrOfExistingBreakpoints, nrOfExistingBreakpoints + 1);
        m_breakpoints.append(bp);
        emit(breakPointAdded(bp, m_breakpoints.size()-1));
        endInsertRows();
    }
    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------------------------
//! delete breakpoint given by its QModelIndex
/*!
    emits breakPointDeleted if deletion has been successfull.

    \param index QModelIndex of breakpoint which should be deleted
    \return retOk in case of success, if QModelIndex not valid retError
*/
RetVal BreakPointModel::deleteBreakPoint(const QModelIndex &index)
{
    if(index.isValid())
    {
        if (index.internalPointer() != NULL) // delete one single breakpoint, second level item
        {
            int breakPointIndex = getBreakPointIndex(index);

            if (breakPointIndex >= 0)
            {
                int row = index.row();
                QModelIndex fileItem = index.parent();
                int fileIndex = fileItem.row();
                const BreakPointItem &item = m_breakpoints[breakPointIndex];

                // remove the breakpoint
                beginRemoveRows(fileItem, row, row);
                emit(breakPointDeleted(item.filename, item.lineIdx, item.pythonDbgBpNumber));
                m_breakpoints.removeAt(breakPointIndex);
                endRemoveRows();

                if (nrOfBreakpointsInFile(fileIndex) <= 0)
                {
                    // this file has no breakpoints any more. Delete the entire file...
                    QModelIndex root = fileItem.parent();
                    beginRemoveRows(root, fileIndex, fileIndex);
                    m_scriptFiles.removeAt(fileIndex);
                    endRemoveRows();
                }

                return RetVal(retOk);
            }
        }
        else //delete all breakpoints of a file
        {
            // in this case, index is the parent, so delete all the children and the parent afterwards
            // problem with BreakPointModel::deleteBreakPoints because a list can contain files and BPs
            // In this case it might happen, that the file is deleted and the corresponding BPs
            // are still in the list from BreakPointModel::deleteBreakPoints!!!
        }
    }

    return RetVal(retError);
}

//-------------------------------------------------------------------------------------------------------
//! delete multiple breakpoints given by a list of QModelIndex
/*!
    calls deleteBreakPoint method for each element of QModelIndexList

    \param indizes list of QModelIndex
    \return retOk in case of total success, if any deletion returned with retError, the total return value will be retError, too.
    \sa deleteBreakPoint
*/
RetVal BreakPointModel::deleteBreakPoints(const QModelIndexList &indizes)
{
    RetVal retValue(retOk);

    QModelIndexList indizes_sorted = indizes;
    std::sort(indizes_sorted.begin(), indizes_sorted.end(), &BreakPointModel::compareRow);

    QModelIndexList::ConstIterator it;

    for(it = indizes_sorted.constBegin() ; it != indizes_sorted.constEnd() ; ++it)
    {
        void* k = it->internalPointer();
        if (k != NULL)
        {
            retValue += deleteBreakPoint(*it);
        }
    }

    return retValue;
}

//-------------------------------------------------------------------------------------------------------
//! delete all breakpoints
/*!
    \param indizes list of QModelIndex
    \return retOk
    \sa deleteBreakPoint, deleteBreakPoints
*/
RetVal BreakPointModel::deleteAllBreakPoints()
{
    beginResetModel();
    foreach (const BreakPointItem &item, m_breakpoints)
    {
        emit(breakPointDeleted(item.filename, item.lineIdx, item.pythonDbgBpNumber));
    }
    m_breakpoints.clear();
    m_scriptFiles.clear();
    endResetModel();

    return retOk;
}

//-------------------------------------------------------------------------------------------------------
//! counts number of breakpoints in this model
/*!
    \return number of elements
*/
int BreakPointModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid() == false) //first level, root item
    {
        return m_scriptFiles.count();
    }
    else if (parent.internalPointer() == NULL)
    {
        return nrOfBreakpointsInFile(parent.row());
    }
    else
    {
        return 0;
    }
}

//-------------------------------------------------------------------------------------------------------
//! counts number of columns in this model (corresponds to number of header-elements)
/*!
    \return number of columns
*/
int BreakPointModel::columnCount(const QModelIndex &parent) const
{
    return m_headers.size();
}

//-------------------------------------------------------------------------------------------------------
//! overwritten data method of QAbstractItemModel
/*!
    data method will be called by View-Widget in order to fill the table.

    \param index QModelIndex of item, whose content should be returned
    \return content of desired item and column
*/
QVariant BreakPointModel::data(const QModelIndex &index, int role) const
{
    if(!index.isValid())
    {
        return QVariant();
    }

    if (!index.parent().isValid()) // toplevel-item
    {
        if (index.column() == 0)
        {
            if (role == Qt::DisplayRole)
            {
                QFileInfo finfo(m_scriptFiles[index.row()]);
                return finfo.fileName();
            }
            else if(role == Qt::ToolTipRole)
            {
                QFileInfo finfo(m_scriptFiles[index.row()]);
                return finfo.absoluteFilePath();
            }
            else if(role == Qt::TextAlignmentRole)
            {
                return QVariant(Qt::Alignment(Qt::AlignLeft | Qt::AlignVCenter));
            }
            else if(role == Qt::DecorationRole)
            {
                return QIcon(":/files/icons/filePython.png");
            }
        }
        else // these columns are always empty in toplevel items
        {
            return QVariant();
        }
    }
    else // second level item
    {
        int breakPointIndex = getBreakPointIndex(index);

        //security check (the case already occurred that breakPointIndex returned -1)
        if (breakPointIndex < 0 || breakPointIndex >= m_breakpoints.size())
        {
            return QVariant();
        }

        const BreakPointItem item = m_breakpoints[breakPointIndex];

        if(role == Qt::DisplayRole)
        {
            switch(index.column())
            {
            case 0: //line
                return item.lineIdx + 1;
            case 1: //condition
                return item.conditioned ? item.condition : "";
            case 2: //temporary
                return item.temporary ? tr("yes") : tr("no");
            case 3: //enabled
                return item.enabled ? tr("yes") : tr("no");
            case 4: //ignore count
                return item.ignoreCount;
            }
        }
        else if(role == Qt::DecorationRole && index.column() == 0)
        {
            if (item.conditioned || item.temporary || item.ignoreCount != 0)
            { // conditioned
                if (item.enabled)
                {
                    return QIcon(":/breakpoints/icons/itomcBreak.png");
                }
                else
                {
                    return QIcon(":/breakpoints/icons/itomCBreakDisabled.png");
                }
            }
            else
            { // not conditioned
                if (item.enabled)
                {
                    return QIcon(":/breakpoints/icons/itomBreak.png");
                }
                else
                {
                    return QIcon(":/breakpoints/icons/itomBreakDisabled.png");
                }
            }
        }
        else if(role == Qt::ToolTipRole)
        {
            switch(index.column())
            {
            case 0:
                return item.filename;
            }
        }
        else if(role == Qt::TextAlignmentRole)
        {
            switch(index.column())
            {
            case 0: //line
                return QVariant(Qt::Alignment(Qt::AlignLeft | Qt::AlignVCenter));
            case 1: //condition
                return Qt::AlignCenter;
            case 2: //temporary
                return QVariant(Qt::Alignment(Qt::AlignLeft | Qt::AlignVCenter));
            case 3: //enabled
                return Qt::AlignCenter;
            case 4: //ignore count
                return Qt::AlignCenter;
            }
        }
    }
    return QVariant();
}

//-------------------------------------------------------------------------------------------------------
//! returns QModelIndex for given row and column
/*!
    \param row row of desired entry, corresponds to index in m_breakpoints list
    \param column column of desired entry
    \param parent since this model is no tree model, parent always points to a "virtual" root element
    \return empty QModelIndex if row or column are out of bound, else returns new valid QModelIndex for that combination of row and column
*/
QModelIndex BreakPointModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!parent.isValid()) //root item
    {
        if (row < 0 || column < 0 || column >= m_headers.size() || row >= m_scriptFiles.count())
        {
            return QModelIndex();
        }
        else
        {                           //???
            return createIndex(row, column, (void*)NULL); //internalId of root item is NULL
        }
    }
    else
    {
        int nrOfFiles = nrOfBreakpointsInFile(parent.row());
        if (row < 0 || row >= nrOfFiles || column < 0 || column >= m_headers.size())
        {
            return QModelIndex();
        }
        else
        {                           //???
            return createIndex(row, column, (void*)(&(m_scriptFiles[parent.row()])));
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------------------------
QSize BreakPointModel::span(const QModelIndex &index) const
{
	if (!index.isValid())
	{
		return QAbstractItemModel::span(index);
	}
	else if (index.internalPointer() == NULL)
	{
		//file
		return QSize(columnCount(), 1);
	}
	else
	{
		return QAbstractItemModel::span(index);
	}
}

//-------------------------------------------------------------------------------------------------------
//! returns parent of given QModelIndex
/*!
    since this model is not a tree model, returns always an empty QModelIndex
*/
QModelIndex BreakPointModel::parent(const QModelIndex &index) const
{
    if (index.isValid() && index.internalPointer() == NULL) //this index is a root level item
    {
        return QModelIndex();
    }
    else
    {
        int parentRow = getFileIndexFromInternalPtr(index.internalPointer());

        if (parentRow >= 0)
        {
            return createIndex(parentRow, 0, (void*)NULL);
        }
        else //may not occur
        {
            return QModelIndex();
        }
    }
}

//-------------------------------------------------------------------------------------------------------
int BreakPointModel::getFileIndexFromInternalPtr(const void* ptr) const
{
    for (int i = 0; i < m_scriptFiles.size(); ++i)
    {
        if (&(m_scriptFiles[i]) == ptr)
        {
            return i;
        }
    }

    return -1;
}

//-------------------------------------------------------------------------------------------------------
//! returns header element at given position
/*!
    \param section position in m_headers list
    \param orientation the model's orientation should be horizontal, no other orientation is supported
    \param role model is only prepared for DisplayRole
    \return name of header or empty QVariant value (if no header element available)
*/
QVariant BreakPointModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if( role == Qt::DisplayRole && orientation == Qt::Horizontal )
    {
        if(section >= 0 && section < m_headers.size())
        {
            return m_headers.at(section);
        }
        return QVariant();
    }
    else if (role == (Qt::UserRole + Qt::SizeHintRole))
    {
        //this is a "special role" that only returns the desired column width. The
        //view that uses these hints has to multiply these pixel values by the screen dpi resolution factor.
        switch(section)
        {
        case 0: //filename
            return 180;
        case 1: //condition
            return 70;
        case 2: //temporary
            return 60;
        case 3: //enabled
            return 50;
        case 4: //ignore count
            return 80;
        }
    }

    return QVariant();
}

//-------------------------------------------------------------------------------------------------------
//! This function returns the number of breakpoints in the given File
/*!
    \return the number is returned as an integer
*/
int BreakPointModel::nrOfBreakpointsInFile(const int fileIdx) const
{
    if (fileIdx >= 0 && fileIdx < m_scriptFiles.size())
    {
        QString filename = m_scriptFiles[fileIdx];
        int count = 0;

        for (int i = 0; i < m_breakpoints.size(); ++i)
        {
            if (QString::compare(m_breakpoints[i].filename, filename, m_filenameCaseSensitivity) == 0)
            {
                count++;
            }
        }
        return count;
    }

    return 0;
}

//-------------------------------------------------------------------------------------------------------
//! This function returns the QModelindex of the given File
/*!
    \return QModelIndex of the given file
*/
QModelIndex BreakPointModel::getFilenameModelIndex(const QString &filename) const
{
    for(int i = 0; i < m_scriptFiles.size(); ++i)
    {
        if (QString::compare(m_scriptFiles[i], filename, m_filenameCaseSensitivity) == 0)
        {
            return createIndex(i, 0, (void*)NULL);
        }
    }
    return QModelIndex();
}

//-------------------------------------------------------------------------------------------------------
//! returns QModelIndex for first breakpoint which is found in given filename and at given line number.
/*!
    \param filename Filename of Python macro file
    \param lineIdx line, where breakpoint is expected
    \return valid QModelIndex, if breakpoint could be found, else returns empty QModelIndex
*/
QModelIndex BreakPointModel::getFirstBreakPointIndex(const QString &filename, int lineIdx) const
{
    void *filePointer = NULL;

    //first find filename in m_scriptFiles
    for (int i = 0; i < m_scriptFiles.size(); ++i)
    {
        if (QString::compare(m_scriptFiles[i], filename, m_filenameCaseSensitivity) == 0)
        {
            filePointer = (void*)(&(m_scriptFiles[i]));
            break;
        }
    }

    if (filePointer)
    {
        int count = -1;

        for (int row = 0; row < m_breakpoints.size(); ++row)
        {
            if (QString::compare(m_breakpoints[row].filename, filename, m_filenameCaseSensitivity) == 0)
            {
                count++;

                if (m_breakpoints[row].lineIdx == lineIdx)
                {
                    return createIndex(count, 0, filePointer);
                }
            }
        }
    }

    return QModelIndex();
}

//-------------------------------------------------------------------------------------------------------
//! returns a list of QModelIndex for all breakpoints, which are registered in given file and at given line number.
/*!
    \param filename Filename of python macro
    \param lineIdx line, where breakpoint is expected
    \return list of detected QModelIndex, corresponding to each found breakpoint
    \sa getFirstBreakPointIndex
*/
QModelIndexList BreakPointModel::getBreakPointIndizes(const QString &filename, int lineIdx) const
{
    QModelIndexList list;
    void *filePointer = NULL;

    //first find filename in m_scriptFiles
    for (int i = 0; i < m_scriptFiles.size(); ++i)
    {
        if (QString::compare(m_scriptFiles[i], filename, m_filenameCaseSensitivity) == 0)
        {
            filePointer = (void*)(&(m_scriptFiles[i]));
            break;
        }
    }

    if (filePointer)
    {
        int count = -1;

        for (int row = 0; row < m_breakpoints.size(); ++row)
        {
			if (QString::compare(m_breakpoints[row].filename, filename, m_filenameCaseSensitivity) == 0)
            {
                count++;

                if (m_breakpoints[row].lineIdx == lineIdx)
                {
                    list.push_back(createIndex(count, 0, filePointer));
                }
            }
        }
    }

    return list;
}

//-------------------------------------------------------------------------------------
//! returns a List of all Breakpoints, doesn't matter in which file they are.
/*
    \return QModelIndexList of all Breakpoints whereever they are
*/
QModelIndexList BreakPointModel::getAllBreakPointIndizes()
{
    QModelIndexList allBPs;
    for (int i = 0; i < m_scriptFiles.size(); ++i)
    {
        QString filename = m_scriptFiles[i];
        allBPs.append(getBreakPointIndizes(filename));
    }
    return allBPs;
}

//-------------------------------------------------------------------------------------
//! returns BreakPointItem for breakpoint being in given file and at given line number
/*!
    \param filename Filename of python macro file
    \param lineIdx line number in given filename
    \return breakpoint element represented by a BreakPointItem-struct
*/
BreakPointItem BreakPointModel::getBreakPoint(const QString &filename, int lineIdx) const
{
    return getBreakPoint(getFirstBreakPointIndex(filename, lineIdx));
}

//-------------------------------------------------------------------------------------
//! returns BreakPointItem for given QModelIndex
/*!
    \param index given QModelIndex
    \return BreakPointItem to the QModelIndex taht was given
*/
BreakPointItem BreakPointModel::getBreakPoint(const QModelIndex &index) const
{
    if (index.isValid() && index.internalPointer() != NULL)
    {
        for (int i = 0; i < m_scriptFiles.size(); ++i)
        {
            if (&m_scriptFiles[i] == index.internalPointer())
            {
                QString filename = m_scriptFiles[i];
                int breakpointIndex = index.row();
                int count = -1;

                for (int j = 0; j < m_breakpoints.size(); ++j)
                {
                    if (QString::compare(m_breakpoints[j].filename, filename, m_filenameCaseSensitivity) == 0)
                    {
                        count++;
                    }

                    if (count == breakpointIndex)
                    {
                        return m_breakpoints[j];
                    }
                }
            }
        }
    }

    return BreakPointItem();
}

//-------------------------------------------------------------------------------------
//! returns the index for given QModelIndex
/*!
    \param index given QModelIndex
    \return index as an integer
*/
int BreakPointModel::getBreakPointIndex(const QModelIndex &index) const
{
    if (index.isValid() && index.internalPointer() != NULL)
    {
        for (int i = 0; i < m_scriptFiles.size(); ++i)
        {
            if (&m_scriptFiles[i] == index.internalPointer())
            {
                QString filename = m_scriptFiles[i];
                int breakpointIndex = index.row();
                int count = -1;

                for (int i = 0; i < m_breakpoints.size(); ++i)
                {
                    if (QString::compare(m_breakpoints[i].filename, filename, m_filenameCaseSensitivity) == 0)
                    {
                        count++;
                    }

                    if (count == breakpointIndex)
                    {
                        return i;
                    }
                }
            }
        }
    }

    return -1;
}

//-------------------------------------------------------------------------------------
//! This function returns a list of all files that contain breakpoints.
/*!
    \return QModelIndexList of all files that contain breakpoints.
*/
QModelIndexList BreakPointModel::getAllFileIndexes()
{
    QModelIndexList retList;
    for (int i = 0; i < m_scriptFiles.size(); ++i)
    {
        // TODO: Case Sensitivity chekcen
        retList.append(this->getFilenameModelIndex(m_scriptFiles.at(i)));
    }
    return retList;
}

//-------------------------------------------------------------------------------------
//! changes breakpoint, given by its QModelIndex to values, determined by BreakPointItem
/*!
    if indicated, emits signal emitBreakPointChanged with old and new BreakPointItem

    \param index QModelIndex of item, which should be changed
    \param bp BreakPointItem with new values for this breakpoint
    \param emitBreakPointChanged if signal should be emitted, this value must be true, else false
    \return retOk, if index has been valid, retError, else.
*/
RetVal BreakPointModel::changeBreakPoint(const QModelIndex index, BreakPointItem bp, bool emitBreakPointChanged)
{
    //it is only allowed to change second level breakpoints
    //no change of filename is allowed
    ito::RetVal retval;

    if (!index.isValid())
    {
        retval += ito::RetVal(ito::retError, 0, tr("Given modelIndex of breakpoint is invalid").toLatin1().data());
    }
    else if (index.internalPointer() != NULL)
    {
        //second level item
        int idx = getBreakPointIndex(index);

        if (idx >= 0)
        {
            if (m_breakpoints[idx].filename != bp.filename)
            {
                retval += ito::RetVal(ito::retError, 0, tr("Filename must not be changed").toLatin1().data());
            }
            else
            {
                BreakPointItem oldBp = m_breakpoints[idx];
                m_breakpoints[idx] = bp;
                emit(dataChanged(createIndex(index.row(),0,index.internalPointer()),createIndex(index.row(),0/*m_headers.size()-1*/,index.internalPointer())));

                if(emitBreakPointChanged) //!< should be false, if filename or line-nr of editor has changed.
                {
                    emit(breakPointChanged(oldBp, bp));
                }
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("Given modelIndex is no model index of a breakpoint").toLatin1().data());
    }

    return retval;
}

//-------------------------------------------------------------------------------------
//! returns QModelIndexList with all breakpoints being in one given file
/*!
    \param filename Filename of python macro file
    \return list of model indices
*/
QModelIndexList BreakPointModel::getBreakPointIndizes(const QString &filename) const
{
    void *filePointer = NULL;
    QModelIndexList list;

    //first find filename in m_scriptFiles
    for (int i = 0; i < m_scriptFiles.size(); ++i)
    {
        if (QString::compare(m_scriptFiles[i], filename, m_filenameCaseSensitivity) == 0)
        {
            filePointer = (void*)(&(m_scriptFiles[i]));
            break;
        }
    }

    if (filePointer)
    {
        int count = -1;

        for (int row = 0; row < m_breakpoints.size(); ++row)
        {
            if (QString::compare(m_breakpoints[row].filename, filename, m_filenameCaseSensitivity) == 0)
            {
                count++;
                list.push_back(createIndex(count, 0, filePointer));
            }
        }
    }

    return list;
}

//-------------------------------------------------------------------------------------
//! returns list of BreakPointItem corresponding to given list of model indices
/*!
    \param indizes list of model indices
    \return list of BreakPointItem
*/
QList<BreakPointItem> BreakPointModel::getBreakPoints(const QModelIndexList &indizes) const
{
    QList<BreakPointItem> bps;

    foreach(const QModelIndex &idx, indizes)
    {
        if (idx.parent().isValid())
        {
            bps.append(getBreakPoint(idx));
        }
    }

    return bps;
}

//-------------------------------------------------------------------------------------
//! change multiple breakpoints to data, given by list of BreakPointItem
/*!
    \param indizes list of model indices
    \param bps list of BreakPointItem
    \param emitBreakPointChanged true if the breakPointChanged signal should be emitted after having changed the property of one single breakpoint
    \return retOk, if everything was ok, else retError
    \sa changeBreakPoint
*/
RetVal BreakPointModel::changeBreakPoints(const QModelIndexList indizes, QList<BreakPointItem> bps, bool emitBreakPointChanged)
{
    RetVal retValue(retOk);
    if( indizes.size() == bps.size() )
    {
        for(int i=0; i<indizes.size() ; ++i)
        {
            retValue += changeBreakPoint(indizes.at(i), bps.at(i),emitBreakPointChanged);
        }
    }
    else
    {
        retValue += RetVal(retError);
    }
    return retValue;
}

//-------------------------------------------------------------------------------------------------------
//! resets all python breakpoint numbers to -1.
/*!
    every breakpoint only gets a valid python breakpoint number, if python is in debugging mode. This method is called,
    if python leaves the debugging mode.

    \return retOk
*/
RetVal BreakPointModel::resetAllPyBpNumbers()
{
    QList<BreakPointItem>::iterator it;
    for(it = m_breakpoints.begin() ; it != m_breakpoints.end() ; ++it)
    {
        (*it).pythonDbgBpNumber = -1;
    }                                                                    //???
    emit(dataChanged(createIndex(0,0),createIndex(m_breakpoints.size()-1,m_headers.size()-1)));
    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------------------------
//! set python breakpoint number of breakpoint at given row in the model
/*!
    If starting debugging a python macro, the whole breakpoint list is submitted to the python debugger, which assigns a python debugging number for each breakpoint.
    This method calls the changeBreakPoint method.

    \param row row of breakpoint in model
    \param pyBpNumber python breakpoint number for this breakpoint
    \return result of changeBreakPoint method
    \sa changeBreakPoint
*/
RetVal BreakPointModel::setPyBpNumber(const BreakPointItem &item, int pyBpNumber)
{                                                     //???
    //get modelIndex of file
    QModelIndex index = getFirstBreakPointIndex(item.filename, item.lineIdx);

    if (index.isValid())
    {
        BreakPointItem bp = item;
        bp.pythonDbgBpNumber = pyBpNumber;
        return changeBreakPoint(index,bp,false);
    }

    return ito::retError;
}

} //end namespace ito
