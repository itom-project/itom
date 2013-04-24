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

#include <algorithm>
#include "breakPointModel.h"

#include <qfileinfo.h>
#include <qsize.h>

/*!
    \class BreakPointModel
    \brief model for management of all breakpoints. This model will be displayed by a viewer-widget in the main window
*/

//! constructor
/*!
    initializes headers and its alignment
*/
BreakPointModel::BreakPointModel() : QAbstractItemModel()
{
    m_headers   << tr("filename")          << tr("line")               << tr("condition")         << tr("temporary")            << tr("enabled")              << tr("ignore count")       << tr("py bp nr");
    m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignRight) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignHCenter) << QVariant(Qt::AlignHCenter) << QVariant(Qt::AlignRight) << QVariant(Qt::AlignLeft);
}

//! destructor
BreakPointModel::~BreakPointModel()
{
    m_headers.clear();
    m_alignment.clear();
    m_breakpoints.clear();
}

//! adds given breakpoint to model
/*!
    if added, the signal breakPointAdded is emitted.

    \param[in] bp Breakpoint of type BreakPointItem
    \return retOk
*/
RetVal BreakPointModel::addBreakPoint(BreakPointItem bp)
{
    beginInsertRows(QModelIndex(), m_breakpoints.size(), m_breakpoints.size());
    m_breakpoints.append(bp);
	emit(breakPointAdded(bp, m_breakpoints.size()-1));
    endInsertRows();

    return RetVal(retOk);
}

//! delete breakpoint given by its QModelIndex
/*!
    emits breakPointDeleted if deletion has been successfull.

    \param index QModelIndex of breakpoint which should be deleted
    \return retOk in case of success, if QModelIndex not valid retError
*/
RetVal BreakPointModel::deleteBreakPoint(QModelIndex index)
{
    if(index.isValid())
    {
		BreakPointItem item = getBreakPoint(index);
        int row = index.row();
        beginRemoveRows(QModelIndex(), row, row);
        m_breakpoints.removeAt(row);
        emit(breakPointDeleted(item.filename, item.lineno, item.pythonDbgBpNumber));
        endRemoveRows();
        return RetVal(retOk);
    }
    else
    {
        return RetVal(retError);
    }
}

//! delete multiple breakpoints given by a list of QModelIndex
/*!
    calls deleteBreakPoint method for each element of QModelIndexList

    \param indizes list of QModelIndex
    \return retOk in case of total success, if any deletion returned with retError, the total return value will be retError, too.
    \sa deleteBreakPoint
*/
RetVal BreakPointModel::deleteBreakPoints(QModelIndexList indizes)
{
	RetVal retValue(retOk);

	std::sort(indizes.begin(), indizes.end(), &BreakPointModel::compareRow);
	
	QModelIndexList::Iterator it;

	for(it = indizes.begin() ; it != indizes.end() ; ++it)
	{
		retValue += deleteBreakPoint(*it);
	}

	return retValue;
}

//! counts number of breakpoints in this model
/*!
    \return number of elements
*/
int BreakPointModel::rowCount(const QModelIndex &/*parent*/) const
{
    return m_breakpoints.size();
}

//! counts number of columns in this model (corresponds to number of header-elements)
/*!
    \return number of columns
*/
int BreakPointModel::columnCount(const QModelIndex &/*parent*/) const
{
    return m_headers.size();
}

//! overwritten data method of QAbstractItemModel
/*!
    data method will be called by View-Widget in order to fill the table.

    \param index QModelIndex of item, whose content should be returned
    \return content of desired item and column
*/
QVariant BreakPointModel::data(const QModelIndex &index, int role) const
{
    const BreakPointItem& item = m_breakpoints.at(index.row());

	if(!index.isValid())
	{
		return QVariant();
	}
 
    if(role == Qt::DisplayRole)
    {
        switch(index.column())
        {
        case 0: //filename
            {
                QFileInfo finfo(item.filename);
                return finfo.fileName();
            }
        case 1: //line
            return item.lineno + 1;
        case 2: //condition
            return item.conditioned ? item.condition : "";
        case 3: //temporary
            return item.temporary ? tr("yes") : tr("no");
		case 4: //enabled
			return item.enabled ? tr("yes") : tr("no");
		case 5: //ignore count
			return item.ignoreCount;
        case 6: //pythonDbgBpNumber
            return item.pythonDbgBpNumber;
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
        case 0: //filename
            return Qt::AlignLeft;
        case 1: //line
            return Qt::AlignCenter;
        case 2: //condition
            return Qt::AlignLeft;
        case 3: //temporary
            return Qt::AlignCenter;
		case 4: //enabled
			return Qt::AlignCenter;
		case 5: //ignore count
			return Qt::AlignCenter;
        case 6: //pythonDbgBpNumber
            return Qt::AlignCenter;
        }
    }
    else if(role == Qt::SizeHintRole)
    {
        switch(index.column())
        {
        case 0: //filename
            return QSize(130,12);
        case 1: //line
            return QSize(20,12);
        case 2: //condition
            return QSize(50,12);
        case 3: //temporary
            return QSize(20,12);
		case 4: //enabled
			return QSize(20,12);
		case 5: //ignore count
			return QSize(20,12);
        case 6: //pythonDbgBpNumber
            return QSize(20,12);
        }
    }
 
    return QVariant();
}

//! returns QModelIndex for given row and column
/*!
    \param row row of desired entry, corresponds to index in m_breakpoints list
    \param column column of desired entry
    \param parent since this model is no tree model, parent always points to a "virtual" root element
    \return empty QModelIndex if row or column are out of bound, else returns new valid QModelIndex for that combination of row and column
*/
QModelIndex BreakPointModel::index(int row, int column, const QModelIndex &parent) const
{
    if(parent.isValid() || row < 0 || row >= m_breakpoints.size() || column < 0 || column >= m_headers.size())
    {
        return QModelIndex();
    }
    
    return createIndex(row, column);
}

//! returns parent of given QModelIndex
/*!
    since this model is not a tree model, returns always an empty QModelIndex
*/
QModelIndex BreakPointModel::parent(const QModelIndex &/*index*/) const
{
    return QModelIndex();
}


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
	return QVariant();
}

//! returns QModelIndex for first breakpoint which is found in given filename and at given line number.
/*!
    \param filename Filename of Python macro file
    \param lineNo line, where breakpoint is expected
    \return valid QModelIndex, if breakpoint could be found, else returns empty QModelIndex
*/
QModelIndex BreakPointModel::getFirstBreakPointIndex(const QString filename, int lineNo) const
{

	for (int row = 0; row < m_breakpoints.size(); ++row)
	{
		if(m_breakpoints.at(row).filename == filename && m_breakpoints.at(row).lineno == lineNo)
		{
			return createIndex(row, 1);
		}
	}
	return QModelIndex();
}

//! returns a list of QModelIndex for all breakpoints, which are registered in given file and at given line number.
/*!
    \param filename Filename of python macro
    \param lineNo line, where breakpoint is expected
    \return list of detected QModelIndex, corresponding to each found breakpoint
    \sa getFirstBreakPointIndex
*/
QModelIndexList BreakPointModel::getBreakPointIndizes(const QString filename, int lineNo) const
{
	QModelIndexList list;
	for (int row = 0; row < m_breakpoints.size(); ++row)
	{
		if(m_breakpoints.at(row).filename == filename && m_breakpoints.at(row).lineno == lineNo)
		{
			list.push_back(createIndex(row, 1));
		}
	}
	return list;
}

//! returns BreakPointItem for breakpoint being in given file and at given line number
/*!
    \param filename Filename of python macro file
    \param lineNo line number in given filename
    \return breakpoint element represented by a BreakPointItem-struct
*/
BreakPointItem BreakPointModel::getBreakPoint(const QString filename, int lineNo) const
{
	return getBreakPoint(getFirstBreakPointIndex(filename, lineNo));
}

//! returns BreakPointItem for given QModelIndex
/*!
    \param index given QModelIndex
    \return element of breakpoint list or empty BreakPointItem, if index is invalid
*/
BreakPointItem BreakPointModel::getBreakPoint(const QModelIndex index) const
{
	if(index.isValid() && index.row() < m_breakpoints.size() )
	{
		return m_breakpoints.at(index.row());
	}
	else
	{
		return BreakPointItem();
	}
}

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
	if(index.isValid()  && index.row() < m_breakpoints.size() )
	{
		BreakPointItem oldBp = m_breakpoints.at(index.row());
		m_breakpoints.replace(index.row(),bp);
		emit(dataChanged(index,index));
		this->setData(index, QVariant(), Qt::DisplayRole);
		emit(dataChanged(createIndex(index.row(),0),createIndex(index.row(),m_headers.size()-1)));
		if(emitBreakPointChanged) //!< should be false, if filename or line-nr of editor has changed.
		{
			emit(breakPointChanged(oldBp, bp));
		}
		return RetVal(retOk);
	}
	return RetVal(retError);

}

//! returns QModelIndexList with all breakpoints being in one given file
/*!
    \param filename Filename of python macro file
    \return list of model indices
*/
QModelIndexList BreakPointModel::getBreakPointIndizes(const QString filename) const
{
	QModelIndexList list;

	for (int row = 0; row < m_breakpoints.size(); ++row)
	{
		if(m_breakpoints.at(row).filename == filename)
		{
			list.push_back(createIndex(row,1));
		}
	}
	return list;
}

//! returns list of BreakPointItem corresponding to given list of model indices
/*!
    \param indizes list of model indices
    \return list of BreakPointItem
*/
QList<BreakPointItem> BreakPointModel::getBreakPoints(const QModelIndexList indizes) const
{
	QList<BreakPointItem> bps;
	for(int i=0; i<indizes.size(); ++i)
	{
		bps.push_back(getBreakPoint(indizes.at(i)));
	}
	return bps;
}

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
    }
    emit(dataChanged(createIndex(0,0),createIndex(m_breakpoints.size()-1,m_headers.size()-1)));
    return RetVal(retOk);
}

//! set python breakpoint number of breakpoint at given row in the model
/*!
    If starting debugging a python macro, the whole breakpoint list is submitted to the python debugger, which assigns a python debugging number for each breakpoint.
    This method calls the changeBreakPoint method.

    \param row row of breakpoint in model
    \param pyBpNumber python breakpoint number for this breakpoint
    \return result of changeBreakPoint method
    \sa changeBreakPoint
*/
RetVal BreakPointModel::setPyBpNumber(int row, int pyBpNumber)
{
    BreakPointItem bp = getBreakPoint(createIndex(row,1));
    bp.pythonDbgBpNumber=pyBpNumber;
    return changeBreakPoint(createIndex(row,1),bp,false);
}