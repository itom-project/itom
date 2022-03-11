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

#include "../AppManagement.h"
#include "../organizer/uiOrganizer.h"
#include "common/sharedStructuresQt.h"

#include <qdebug.h>
#include <qmessagebox.h>

namespace ito {

//-------------------------------------------------------------------------------------
ExtendedDataObjectTable::ExtendedDataObjectTable(QWidget* parent /*= nullptr*/) :
    DataObjectTable(parent), m_pActPlot2d(nullptr), m_pActPlot1d(nullptr)
{
    QAction* a = new QAction(this);
    a->setSeparator(true);
    addAction(a);

    m_pActPlot2d = new QAction(QIcon(":/plots/icons/itom_icons/2d.png"), tr("2D Image Plot"), this);
    connect(m_pActPlot2d, &QAction::triggered, this, &ExtendedDataObjectTable::showPlot2d);
    m_pActPlot2d->setStatusTip(
        tr("Opens the current table or current selection in a 2d image plot."));
    m_pActPlot2d->setEnabled(false);
    addAction(m_pActPlot2d);

    m_pActPlot1d = new QAction(QIcon(":/plots/icons/itom_icons/1d.png"), tr("1D Line Plot"), this);
    connect(m_pActPlot1d, &QAction::triggered, this, &ExtendedDataObjectTable::showPlot1d);
    m_pActPlot1d->setStatusTip(
        tr("Opens the current table or current selection in a 1d line plot."));
    m_pActPlot1d->setEnabled(false);
    addAction(m_pActPlot1d);
}

//-------------------------------------------------------------------------------------
ExtendedDataObjectTable::~ExtendedDataObjectTable()
{
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::setTableName(const QString& name)
{
    m_name = name;
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::selectionChanged(
    const QItemSelection& selected, const QItemSelection& deselected)
{
    DataObjectTable::selectionChanged(selected, deselected);

    const QItemSelection ranges = selectionModel()->selection();

    // check if selected is empty, contains one rectangular range with > 1 elements or
    // multiple ranges, whose top/bottoms are equal or left/right positions.
    switch (ranges.size())
    {
    case 0:
        m_pActPlot1d->setEnabled(true);
        m_pActPlot2d->setEnabled(true);
        break;
    case 1: {
        const QItemSelectionRange& range = ranges[0];
        int rows = 1 + range.bottom() - range.top();
        int columns = 1 + range.right() - range.left();
        m_pActPlot1d->setEnabled(rows > 1 || columns > 1);
        m_pActPlot2d->setEnabled(rows > 1 && columns > 1);
    }
    break;
    default: {
        int columnsSum = 0;
        int rowsSum = 0;
        bool rowsEqual = coverAllRangesTheSameRows(ranges, columnsSum);
        bool columnsEqual = coverAllRangesTheSameColumns(ranges, rowsSum);

        m_pActPlot1d->setEnabled(rowsEqual || columnsEqual);
        m_pActPlot2d->setEnabled((rowsEqual && columnsSum > 1) || (columnsEqual && rowsSum > 1));
    }
    break;
    }
}

//-------------------------------------------------------------------------------------
bool ExtendedDataObjectTable::coverAllRangesTheSameRows(
    const QItemSelection& ranges, int& nrOfColumns) const
{
    const QItemSelectionRange& firstRange = ranges[0];
    nrOfColumns = firstRange.width();
    bool rowsEqual = true;

    // check if all ranges have the same height and start row
    for (int i = 1; i < ranges.size(); ++i)
    {
        if (ranges[i].height() != firstRange.height())
        {
            rowsEqual = false;
        }
        else if (ranges[i].top() != firstRange.top())
        {
            rowsEqual = false;
        }

        nrOfColumns += ranges[i].width();
    }

    return rowsEqual;
}

//-------------------------------------------------------------------------------------
bool ExtendedDataObjectTable::coverAllRangesTheSameColumns(
    const QItemSelection& ranges, int& nrOfRows) const
{
    const QItemSelectionRange& firstRange = ranges[0];
    nrOfRows = firstRange.height();
    bool columnsEqual = true;

    // check if all ranges have the same width and start column
    for (int i = 1; i < ranges.size(); ++i)
    {
        if (ranges[i].width() != firstRange.width())
        {
            columnsEqual = false;
        }
        else if (ranges[i].left() != firstRange.left())
        {
            columnsEqual = false;
        }

        nrOfRows += ranges[i].height();
    }

    return columnsEqual;
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::showPlot2d()
{
    showPlotGeneric("2D");
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::showPlot1d()
{
    showPlotGeneric("1D");
}

//-------------------------------------------------------------------------------------
void ExtendedDataObjectTable::showPlotGeneric(const QString& plotClass)
{
    QItemSelection ranges = selectionModel()->selection();
    QSharedPointer<ito::DataObject> dObj;
    ito::RetVal retVal;
    QSharedPointer<ito::DataObject> src = getData();

    if (src.isNull())
    {
        return;
    }

    switch (ranges.size())
    {
    case 0:
        dObj = src;
        break;
    case 1: {
        auto rowRange = ito::Range(ranges[0].top(), ranges[0].bottom() + 1);
        auto colRange = ito::Range(ranges[0].left(), ranges[0].right() + 1);

        if (rowRange.size() == src->getSize(0) && colRange.size() == src->getSize(1))
        {
            dObj = src;
        }
        else
        {
            dObj =
                QSharedPointer<ito::DataObject>(new ito::DataObject(src->at(rowRange, colRange)));
        }
    }
    break;
    default: {
        int nrOfRows, nrOfColumns;
        int idx = 0;

        if (coverAllRangesTheSameRows(ranges, nrOfColumns))
        {
            // sort ranges by column, ascending
            std::sort(
                ranges.begin(),
                ranges.end(),
                [](const QItemSelectionRange& a, const QItemSelectionRange& b) {
                    return a.left() <= b.left();
                });

            dObj = QSharedPointer<ito::DataObject>(
                new ito::DataObject(ranges[0].height(), nrOfColumns, src->getType()));

            auto rowRange = ito::Range(ranges[0].top(), ranges[0].bottom() + 1);

            for (int i = 0; i < ranges.size(); ++i)
            {
                auto colRangeSrc = ito::Range(ranges[i].left(), ranges[i].right() + 1);
                auto colRangeDest = ito::Range(idx, idx + ranges[i].width());
                ito::DataObject destination = dObj->at(ito::Range::all(), colRangeDest);
                idx += colRangeDest.size();
                src->at(rowRange, colRangeSrc)
                    .deepCopyPartial(destination);
            }
        }
        else if (coverAllRangesTheSameColumns(ranges, nrOfRows))
        {
            // sort ranges by row, ascending
            std::sort(
                ranges.begin(),
                ranges.end(),
                [](const QItemSelectionRange& a, const QItemSelectionRange& b) {
                    return a.top() <= b.top();
                });

            dObj = QSharedPointer<ito::DataObject>(
                new ito::DataObject(nrOfRows, ranges[0].width(), src->getType()));

            auto colRange = ito::Range(ranges[0].left(), ranges[0].right() + 1);

            for (int i = 0; i < ranges.size(); ++i)
            {
                auto rowRangeSrc = ito::Range(ranges[i].top(), ranges[i].bottom() + 1);
                auto rowRangeDest = ito::Range(idx, idx + ranges[i].height());
                ito::DataObject destination = dObj->at(rowRangeDest, ito::Range::all());
                idx += rowRangeDest.size();
                src->at(rowRangeSrc, colRange)
                    .deepCopyPartial(destination);
            }
        }

        if (dObj)
        {
            src->copyAxisTagsTo(*(dObj.data()));
            src->copyTagMapTo(*(dObj.data()));
        }
    }
    break;
    }

    if (dObj && dObj->getDims() > 0)
    {
        QVariantMap properties;
        int areaCol = 0;
        int areaRow = 0;
        const ito::DataObject* obj = NULL;

        QSharedPointer<unsigned int> figHandle(new unsigned int);
        *figHandle = 0; // new figure will be requested

        UiOrganizer* uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
        ito::UiDataContainer dataCont = ito::UiDataContainer(dObj);
        QSharedPointer<unsigned int> objectID(new unsigned int);
        ito::UiDataContainer xAxisCont;

        if (!dObj->existTag("title"))
        {
            properties["title"] = m_name;
        }
        else
        {
            properties.remove("title");
        }

        retVal += uiOrg->figurePlot(
            dataCont,
            xAxisCont,
            figHandle,
            objectID,
            areaRow,
            areaCol,
            plotClass,
            properties,
            nullptr);
    }

    if (retVal.containsError())
    {
        const char* errorMsg = retVal.errorMessage();
        QString message = errorMsg ? QLatin1String(errorMsg) : QString();
        QMessageBox::critical(
            this, tr("Plot data"), tr("Error while plotting value(s):\n%1").arg(message));
    }
    else if (retVal.containsWarning())
    {
        const char* errorMsg = retVal.errorMessage();
        QString message = errorMsg ? QLatin1String(errorMsg) : QString();
        QMessageBox::warning(
            this, tr("Plot data"), tr("Warning while plotting value(s):\n%1").arg(message));
    }
}

} // end namespace ito
