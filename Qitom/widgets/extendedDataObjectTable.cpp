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

#include "extendedDataObjectTable.h"

#include <qdebug.h>

namespace ito
{

ExtendedDataObjectTable::ExtendedDataObjectTable(QWidget *parent /*= nullptr*/) :
    DataObjectTable(parent),
    m_pActPlot2d(nullptr),
    m_pActPlot1d(nullptr)
{
    QAction* a = new QAction(this);
    a->setSeparator(true);
    addAction(a);

    m_pActPlot2d = new QAction(QIcon(":/plots/icons/itom_icons/2d.png"), tr("2D Image Plot"), this);
    connect(m_pActPlot2d, &QAction::triggered, this, &ExtendedDataObjectTable::showPlot2d);
    m_pActPlot2d->setStatusTip(tr("Opens the current table or current selection in a 2d image plot."));
    m_pActPlot2d->setEnabled(false);
    addAction(m_pActPlot2d);
    
    m_pActPlot1d = new QAction(QIcon(":/plots/icons/itom_icons/1d.png"), tr("1D Line Plot"), this);
    connect(m_pActPlot1d, &QAction::triggered, this, &ExtendedDataObjectTable::showPlot1d);
    m_pActPlot1d->setStatusTip(tr("Opens the current table or current selection in a 1d line plot."));
    m_pActPlot1d->setEnabled(false);
    addAction(m_pActPlot1d);
}

//-------------------------------------------------------------------------------------
ExtendedDataObjectTable::~ExtendedDataObjectTable()
{

}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::selectionChanged(const QItemSelection &selected, const QItemSelection &deselected)
{
    DataObjectTable::selectionChanged(selected, deselected);

    // check if selected is empty, contains one rectangular range with > 1 elements or
    // multiple ranges, whose top/bottoms are equal or left/right positions.
    switch (selected.size())
    {
    case 0:
        m_pActPlot1d->setEnabled(true);
        m_pActPlot2d->setEnabled(true);
        break;
    case 1:
        {
        const QItemSelectionRange &range = selected[0];
        int rows = 1 + range.bottom() - range.top();
        int columns = 1 + range.right() - range.left();
        m_pActPlot1d->setEnabled(rows > 1 || columns > 1);
        m_pActPlot2d->setEnabled(rows > 1 && columns > 1);
        }
        break;
    default:
        {
            const QItemSelectionRange &range = selected[0];
            QList<int> rows, columns;

            foreach(const QItemSelectionRange &range, selected)
            {
                qDebug() << range.bottom() << range.left() << range.top() << range.right();
                rows << 1 + range.bottom() - range.top();
                columns << 1 + range.right() - range.left();
            }

            int r = rows[0];
            int c = columns[0];
            bool rowsEqual = true;
            bool columnsEqual = true;
            int rowsSum = r;
            int columnsSum = c;

            for (int idx = 1; idx < rows.size(); ++idx)
            {
                rowsEqual &= (rows[idx] == r);
                columnsEqual &= (columns[idx] == c);
                rowsSum += rows[idx];
                columnsSum += columns[idx];
            }

            m_pActPlot1d->setEnabled(rowsEqual || columnsEqual);
            m_pActPlot2d->setEnabled((rowsEqual && columnsSum > 1) || (columnsEqual && rowsSum > 1));
        }
        break;
    }
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::showPlot2d()
{

}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::showPlot1d()
{

}

} // end namespace ito