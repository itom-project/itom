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

#ifndef BREAKPOINTMODEL_H
#define BREAKPOINTMODEL_H

#include "../common/sharedStructures.h"

#include <qabstractitemmodel.h>
#include <qlist.h>
#include <qstringlist.h>

#include <qstring.h>
#include <QDebug>


namespace ito {

//! item of BreakPointModel
/*!
    this struct corresponds to one item in the BreakPointModel
*/
struct BreakPointItem
{
    /*! constructor fills struct with default values */
    BreakPointItem(): filename(""), lineIdx(-1), enabled(true), temporary(false), conditioned(false), condition(""), ignoreCount(0), pythonDbgBpNumber(-1)  {}
    QString filename;       /*!<  filename of corresponding python file */
    int lineIdx;             /*!<  line number */
    bool enabled;           /*!<  indicates whether breakpoint is actually enabled */
    bool temporary;         /*!<  indicates whether breakpoint is temporary. If yes, debugger only stops one time at this breakpoint */
    bool conditioned;       /*!<  indicates whether breakpoint is conditioned */
    QString condition;      /*!<  if conditioned==true, the condition, which is evaluated by the debugger to check, whether to stop or not */
    int ignoreCount;        /*!<  number of times the debugger should ignore this breakpoint before stopping. If 0, debugger always stops at this breakpoint */
    int pythonDbgBpNumber;  /*!<  corresponding breakpoint number in the python debugger */
};

} //end namespace ito

Q_DECLARE_METATYPE(ito::BreakPointItem) //must be outside of namespace

namespace ito
{

QDataStream &operator<<(QDataStream &out, const BreakPointItem &obj);


QDataStream &operator>>(QDataStream &in, BreakPointItem &obj);

class BreakPointModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    BreakPointModel();
    ~BreakPointModel();

    RetVal saveState();
    RetVal restoreState();

    QVariant data(const QModelIndex &index, int role) const;
    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex parent(const QModelIndex &index) const;
    int rowCount(const QModelIndex &parent = QModelIndex()) const;
    int columnCount(const QModelIndex &parent = QModelIndex()) const;

    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

    RetVal addBreakPoint(BreakPointItem bp);
    RetVal deleteBreakPoint(const QModelIndex &index);
    RetVal deleteBreakPoints(const QModelIndexList &indizes);
    RetVal deleteAllBreakPoints();

    QModelIndex getFirstBreakPointIndex(const QString &filename, int lineIdx) const;
    QModelIndexList getBreakPointIndizes(const QString &filename, int lineIdx) const;
    QModelIndexList getBreakPointIndizes(const QString &filename) const;
    QModelIndexList getAllBreakPointIndizes();

    BreakPointItem getBreakPoint(const QString &filename, int lineIdx) const;
    BreakPointItem getBreakPoint(const QModelIndex &index) const;
    QList<BreakPointItem> getBreakPoints(const QModelIndexList &indizes) const;

    RetVal changeBreakPoint(const QModelIndex index, BreakPointItem bp, bool emitBreakPointChanged = true);
    RetVal changeBreakPoints(const QModelIndexList indizes, QList<BreakPointItem> bps, bool emitBreakPointChanged = true);

    QList<BreakPointItem> const getBreakpoints() { return m_breakpoints; };

    QModelIndexList getAllFileIndexes();

    RetVal resetAllPyBpNumbers();
    RetVal setPyBpNumber(const BreakPointItem &item, int pyBpNumber);

	QSize span(const QModelIndex &index) const;

protected:

private:
    int nrOfBreakpointsInFile(const int fileIdx) const;
    QModelIndex getFilenameModelIndex(const QString &filename) const;
    int getBreakPointIndex(const QModelIndex &index) const;
    int getFileIndexFromInternalPtr(const void* ptr) const;

    //! helper-method for sorting different breakpoints with respect to row-index of both given QModelIndex
    static inline bool compareRow(QModelIndex a, QModelIndex b) { return a.row()>b.row(); };

    QList<BreakPointItem> m_breakpoints;    /*!<  list of breakpoints (BreakPointItem) which are currently available in this application */
    QList<QString> m_headers;               /*!<  string list of names of column headers */
    QList<QVariant> m_alignment;            /*!<  list of alignments for the corresponding headers */
    QStringList m_scriptFiles;
    Qt::CaseSensitivity m_filenameCaseSensitivity;

signals:
    /*!<  emitted if breakpoint has been added to model at position row */
    void breakPointAdded(BreakPointItem bp, int row);

    /*!<  emitted if breakpoint in file filename at line lineIdx with python
    internal debugger number has been deleted from model */
    void breakPointDeleted(QString filename, int lineIdx, int pyBpNumber);

    /*!<  emitted if breakpoint oldBp has been changed to newBp */
    void breakPointChanged(BreakPointItem oldBp, BreakPointItem newBp);
};

} //end namespace ito


#endif
