/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "markerLegendWidget.h"
//#include "../common/sharedStructuresPrimitives.h"
//---------------------------------------------------------------------------------------------------------
MarkerLegend::MarkerLegend(QWidget* parent /*= NULL*/) : QTreeWidget(this)
{
    
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    clear();

    insertTopLevelItem(0, new QTreeWidgetItem(this, QStringList("Picker")));
    insertTopLevelItem(1, new QTreeWidgetItem(this, QStringList("Geometric Elements")));
    insertTopLevelItem(2, new QTreeWidgetItem(this, QStringList("Plot Children")));
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::updatePicker(int index, QVector< float > position)
{
    QTreeWidgetItem *pickerEntries = topLevelItem(0);
    QTreeWidgetItem* myChild = NULL;
    for(int idx = 0; idx < pickerEntries->childCount(); idx++)
    {
        if(pickerEntries->child(idx)->data(0, Qt::UserRole).toInt() == index) 
        {
            myChild = pickerEntries->child(idx);
            break;
        }
    
    }

    if(myChild)
    {
        for(int pos = 0; pos < position.size(); pos++)
        {
            myChild->setData(pos + 1, Qt::DisplayRole , QString::number(position[pos]));
            myChild->setData(pos + 1, Qt::UserRole, position[pos]);
        }
    }
    else
    {
        myChild = new QTreeWidgetItem(this, 0);
        myChild->setData(0, Qt::DisplayRole , QString::number(index));
        myChild->setData(0, Qt::UserRole, index);

        for(int pos = 0; pos < position.size(); pos++)
        {
            myChild->setData(pos + 1, Qt::DisplayRole , QString::number(position[pos]));
            myChild->setData(pos + 1, Qt::UserRole, position[pos]);
        }

        pickerEntries->addChild(myChild);
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::updatePickers(QVector< int > indices, QVector< QVector< float > > positions)
{
    QTreeWidgetItem *pickerEntries = topLevelItem(0);

    if(indices.size() != positions.size())
    {
        qDebug("Could not update pickers, indices and positions missmatch");
        return;
    }

    for( int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
    {
        QTreeWidgetItem* myChild = NULL;
        for(int idx = 0; idx < pickerEntries->childCount(); idx++)
        {
            if(pickerEntries->child(idx)->data(0, Qt::UserRole).toInt() == indices[curSearchIndex]) 
            {
                myChild = pickerEntries->child(idx);
                break;
            }
    
        }

        if(myChild)
        {
            for(int pos = 0; pos < positions[curSearchIndex].size(); pos++)
            {
                myChild->setData(pos + 1, Qt::DisplayRole , QString::number(positions[curSearchIndex][pos]));
                myChild->setData(pos + 1, Qt::UserRole, positions[curSearchIndex][pos]);
            }
        }
        else
        {
            myChild = new QTreeWidgetItem(this, 0);
            myChild->setData(0, Qt::DisplayRole , QString::number(indices[curSearchIndex]));
            myChild->setData(0, Qt::UserRole, indices[curSearchIndex]);

            for(int pos = 0; pos < positions[curSearchIndex].size(); pos++)
            {
                myChild->setData(pos + 1, Qt::DisplayRole , QString::number(positions[curSearchIndex][pos]));
                myChild->setData(pos + 1, Qt::UserRole, positions[curSearchIndex][pos]);
            }

            pickerEntries->addChild(myChild);
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::updateGeometry(int index, QPair< int, QVector< float > > element)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::updateGeometries(QVector< int > index, QVector< QPair <int,  QVector< float > > > elements)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::updateLinePlot(int type, QVector<QPointF > positions)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::removePicker(int index)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::removePickers()
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::removeGeometry(int index)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::removeGeometries()
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::removeLinePlot()
{


    return;
}
//---------------------------------------------------------------------------------------------------------