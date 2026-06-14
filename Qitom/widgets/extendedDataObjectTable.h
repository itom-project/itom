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

#pragma once

#include <qwidget.h>
#include <qaction.h>

#include "itomWidgets/dataObjectTable.h"


namespace ito {


class ExtendedDataObjectTable : public DataObjectTable
{
    Q_OBJECT
public:
    ExtendedDataObjectTable(QWidget *parent = nullptr);
    virtual ~ExtendedDataObjectTable();

    void setTableName(const QString &name);

protected:
    virtual void selectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
    void showPlotGeneric(const QString &plotClass);
    bool coverAllRangesTheSameRows(const QItemSelection &ranges, int &nrOfColumns) const;
    bool coverAllRangesTheSameColumns(const QItemSelection &ranges, int &nrOfRows) const;

    QAction *m_pActPlot2d;
    QAction *m_pActPlot1d;
    QString m_name;

private slots:
    void showPlot2d();
    void showPlot1d();
};



} //end namespace ito
