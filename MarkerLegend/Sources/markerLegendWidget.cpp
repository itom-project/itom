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

#include "../MarkerLegend/markerLegendWidget.h"

//---------------------------------------------------------------------------------------------------------
MarkerLegendWidget::MarkerLegendWidget(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    m_onlyTwoDims = true;
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    clear();

    this->setColumnCount(2);

    insertTopLevelItem(0, new QTreeWidgetItem(this, QStringList("Picker")));
    insertTopLevelItem(1, new QTreeWidgetItem(this, QStringList("Shapes")));
    insertTopLevelItem(2, new QTreeWidgetItem(this, QStringList("Plot Nodes")));
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updatePicker(const int index, const QPointF position)
{
	QTreeWidgetItem *pickerEntries = topLevelItem(0);
	QTreeWidgetItem* myChild = NULL;
	for (int idx = 0; idx < pickerEntries->childCount(); idx++)
	{
		if (pickerEntries->child(idx)->data(0, Qt::UserRole).toInt() == index)
		{
			myChild = pickerEntries->child(idx);
			break;
		}

	}

	if (!myChild)
	{
		myChild = new QTreeWidgetItem();
		myChild->setData(0, Qt::DisplayRole, QString::number(index));
		myChild->setData(0, Qt::UserRole, index);

		pickerEntries->addChild(myChild);
	}

	if (myChild)
	{
		myChild->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number(position.x()), QString::number(position.y())));
		myChild->setData(1, Qt::UserRole, position);
	}

	return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updatePickers(const QVector< int > indices, const QVector< QPointF> positions)
{
	QTreeWidgetItem *pickerEntries = topLevelItem(0);

	if (indices.size() != positions.size())
	{
		qDebug("Could not update pickers, indices and positions missmatch");
		return;
	}

	for (int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
	{
		QTreeWidgetItem* myChild = NULL;
		for (int idx = 0; idx < pickerEntries->childCount(); idx++)
		{
			if (pickerEntries->child(idx)->data(0, Qt::UserRole).toInt() == indices[curSearchIndex])
			{
				myChild = pickerEntries->child(idx);
				break;
			}

		}


		if (!myChild)
		{
			myChild = new QTreeWidgetItem();
			myChild->setData(0, Qt::DisplayRole, QString::number(indices[curSearchIndex]));
			myChild->setData(0, Qt::UserRole, indices[curSearchIndex]);

			pickerEntries->addChild(myChild);
		}

		if (myChild)
		{
			myChild->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y())));
			myChild->setData(1, Qt::UserRole, positions[curSearchIndex]);
		}
	}

	return;
}

//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updatePicker(const int index, const QVector3D position)
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

    if(!myChild)
    {
        myChild = new QTreeWidgetItem();
        myChild->setData(0, Qt::DisplayRole , QString::number(index));
        myChild->setData(0, Qt::UserRole, index);

        pickerEntries->addChild(myChild);
    }

    if(myChild)
    {
        myChild->setData(1, Qt::DisplayRole , QString("%1, %2, %3").arg(QString::number(position.x()), QString::number(position.y()), QString::number(position.z())));
        myChild->setData(1, Qt::UserRole, position);
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updatePickers(const QVector< int > indices, const QVector< QVector3D> positions)
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


        if(!myChild)
        {
            myChild = new QTreeWidgetItem();
            myChild->setData(0, Qt::DisplayRole , QString::number(indices[curSearchIndex]));
            myChild->setData(0, Qt::UserRole, indices[curSearchIndex]);

            pickerEntries->addChild(myChild);
        }

        if(myChild)
        {
            myChild->setData(1, Qt::DisplayRole , QString("%1, %2, %3").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y()), QString::number((positions[curSearchIndex]).z())));
            myChild->setData(1, Qt::UserRole, positions[curSearchIndex]);
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updateShape(const int index, const ito::Shape element)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updateShapes(const QVector< ito::Shape > elements)
{

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updateLinePlot(const int index, const int type, const QVector4D positionAndDirection)
{
    QTreeWidgetItem *lineEntries = topLevelItem(1);
    QTreeWidgetItem* myChild = NULL;

    if(lineEntries->childCount() == 0)
    {
        myChild = new QTreeWidgetItem();
		myChild->setData(0, Qt::DisplayRole, type == ito::Shape::Point ? "xy-Line" : type == ito::Shape::Line ? "zSlice" : "Slice");
        myChild->setData(0, Qt::UserRole, type);

        lineEntries->addChild(myChild);
    }

    if(myChild)
    {
        if(type == ito::Shape::Point)
        {
            myChild->setData(1, Qt::DisplayRole , QString("%1, %2").arg(QString::number(positionAndDirection.x()), QString::number(positionAndDirection.y())));
        }
        else
        {
            myChild->setData(1, Qt::DisplayRole , QString("%1, %2 to %3, %4").arg(  QString::number(positionAndDirection.x()), 
                                                                                    QString::number(positionAndDirection.y()), 
                                                                                    QString::number(positionAndDirection.z()), 
                                                                                    QString::number(positionAndDirection.w())));
        }
        myChild->setData(1, Qt::UserRole, positionAndDirection);
    }

    return;

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::updateLinePlots(const QVector<int> indices, const QVector<int> type, const QVector<QVector4D> positionAndDirection)
{

	return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::removePicker(int index)
{
    QTreeWidgetItem *pickerEntries = topLevelItem(0);
    QTreeWidgetItem* myChild = NULL;
    for(int idx = 0; idx < pickerEntries->childCount(); idx++)
    {
        if(pickerEntries->child(idx)->data(0, Qt::UserRole).toInt() == index) 
        {
            myChild = pickerEntries->child(idx);
            pickerEntries->removeChild(myChild);
            break;
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::removePickers()
{
    QTreeWidgetItem *pickerEntries = topLevelItem(0);
    QTreeWidgetItem* myChild = NULL;
    while (pickerEntries->childCount() > 0)
    {
        pickerEntries->removeChild(pickerEntries->child(pickerEntries->childCount()-1));
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::removeShape(int index)
{
    QTreeWidgetItem *shapeEntries = topLevelItem(1);
    QTreeWidgetItem* myChild = NULL;
	for (int idx = 0; idx < shapeEntries->childCount(); idx++)
    {
		if (shapeEntries->child(idx)->data(0, Qt::UserRole).toInt() == index)
        {
			myChild = shapeEntries->child(idx);
			shapeEntries->removeChild(myChild);
            break;
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::removeShapes()
{
	QTreeWidgetItem *shapeEntries = topLevelItem(1);
	while (shapeEntries->childCount() > 0)
    {
		shapeEntries->removeChild(shapeEntries->child(shapeEntries->childCount() - 1));
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::removeLinePlot(int index)
{
    QTreeWidgetItem *lineEntries = topLevelItem(2);

	QTreeWidgetItem* myChild = NULL;
	for (int idx = 0; idx < lineEntries->childCount(); idx++)
	{
		if (lineEntries->child(idx)->data(0, Qt::UserRole).toInt() == index)
		{
			myChild = lineEntries->child(idx);
			lineEntries->removeChild(myChild);
			break;
		}
	}

    return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerLegendWidget::removeLinePlots()
{
	QTreeWidgetItem *lineEntries = topLevelItem(2);
	while (lineEntries->childCount() > 0)
	{
		lineEntries->removeChild(lineEntries->child(lineEntries->childCount() - 1));
	}

	return;
}
//---------------------------------------------------------------------------------------------------------