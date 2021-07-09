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

#include "common/sharedStructuresQt.h"
#include "../AppManagement.h"
#include "../organizer/uiOrganizer.h"

#include <qdebug.h>
#include <qmessagebox.h>

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

    const QItemSelection ranges = selectionModel()->selection();
    /*int idx = 0;

    foreach(const QItemSelectionRange &range, ranges)
    {
        qDebug() << idx++ << range.bottom() << range.left() << range.top() << range.right();
    }*/

    // check if selected is empty, contains one rectangular range with > 1 elements or
    // multiple ranges, whose top/bottoms are equal or left/right positions.
    switch (ranges.size())
    {
    case 0:
        m_pActPlot1d->setEnabled(true);
        m_pActPlot2d->setEnabled(true);
        break;
    case 1:
        {
        const QItemSelectionRange &range = ranges[0];
        int rows = 1 + range.bottom() - range.top();
        int columns = 1 + range.right() - range.left();
        m_pActPlot1d->setEnabled(rows > 1 || columns > 1);
        m_pActPlot2d->setEnabled(rows > 1 && columns > 1);
        }
        break;
    default:
        {
            const QItemSelectionRange &range = ranges[0];
            QList<int> rows, columns;

            foreach(const QItemSelectionRange &range, ranges)
            {
                rows << range.height();
                columns << range.width();
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
    const QItemSelection ranges = selectionModel()->selection();
    ito::DataObject dObj;
    ito::RetVal retVal;
    QSharedPointer<ito::DataObject> src = getData();

    if (src.isNull())
    {
        return;
    }

    switch (ranges.size())
    {
    case 0:
        dObj = *src;
        break;
    case 1:   
    {
        auto rowRange = ito::Range(ranges[0].top(), ranges[0].bottom() + 1);
        auto colRange = ito::Range(ranges[0].left(), ranges[0].right() + 1);
        dObj = src->at(rowRange, colRange);
    }
        break;
    default:
        break;
    }

    if (dObj.getDims() > 0)
    {
        QVariantMap properties;
        int areaCol = 0;
        int areaRow = 0;
        const ito::DataObject *obj = NULL;

        QSharedPointer<unsigned int> figHandle(new unsigned int);
        *figHandle = 0; //new figure will be requested

        UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
        ito::UiDataContainer dataCont = ito::UiDataContainer(QSharedPointer<ito::DataObject>(new ito::DataObject(dObj)));
        QSharedPointer<unsigned int> objectID(new unsigned int);
        ito::UiDataContainer xAxisCont;

        /*if (!dObj.existTag("title"))
        {
            properties["title"] = "";
        }
        else
        {
            properties.remove("title");
        }*/

        retVal += uiOrg->figurePlot(dataCont, xAxisCont, figHandle, objectID, areaRow, areaCol, "2D", properties, nullptr);
    }

    if (retVal.containsError())
    {
        const char *errorMsg = retVal.errorMessage();
        QString message = QString();
        if (errorMsg) message = errorMsg;
        QMessageBox::critical(this, tr("Plot data"), tr("Error while plotting value(s):\n%1").arg(message));
    }
    else if (retVal.containsWarning())
    {
        const char *errorMsg = retVal.errorMessage();
        QString message = QString();
        if (errorMsg) message = errorMsg;
        QMessageBox::warning(this, tr("Plot data"), tr("Warning while plotting value(s):\n%1").arg(message));
    }
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::showPlot1d()
{

}

} // end namespace ito