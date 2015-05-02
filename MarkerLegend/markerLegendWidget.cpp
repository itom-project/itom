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
    

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegend::updatePickers(QVector< int > indices, QVector< QVector< float > > positions)
{
    

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