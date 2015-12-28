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

#include "../plotLegends/infoWidgetShapes.h"

//---------------------------------------------------------------------------------------------------------
ShapesInfoWidget::ShapesInfoWidget(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    m_onlyTwoDims = true;
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    clear();

    this->setColumnCount(2);
	/*
    insertTopLevelItem(0, new QTreeWidgetItem(this, QStringList("Picker")));
    insertTopLevelItem(1, new QTreeWidgetItem(this, QStringList("Shapes")));
    insertTopLevelItem(2, new QTreeWidgetItem(this, QStringList("Plot Nodes")));
	*/
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::updateShape(const ito::Shape element)
{

	QTreeWidgetItem *curItem = NULL;

	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == element.index())
		{
			curItem = this->topLevelItem(idx);
			break;
		}
	}

	if (!curItem)
	{
		curItem = new QTreeWidgetItem();
		curItem->setData(0, Qt::UserRole, element.index());

		addTopLevelItem(curItem);
	}

	if (curItem)
	{
		switch (element.type() & ito::Shape::TypeMask)
		{
		case ito::Shape::Point:
			curItem->setData(0, Qt::DisplayRole, QString("Point %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		case ito::Shape::Line:
			curItem->setData(0, Qt::DisplayRole, QString("Line %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		case ito::Shape::Square:
			curItem->setData(0, Qt::DisplayRole, QString("Sqare %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		case ito::Shape::Rectangle:
			curItem->setData(0, Qt::DisplayRole, QString("Rectangle %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		case ito::Shape::Ellipse:
			curItem->setData(0, Qt::DisplayRole, QString("Ellipse %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		case ito::Shape::Circle:
			curItem->setData(0, Qt::DisplayRole, QString("Circle %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		case ito::Shape::Polygon:
			curItem->setData(0, Qt::DisplayRole, QString("Polygon %1").arg(QString::number(element.index())));
			curItem->setData(1, Qt::UserRole, element.basePoints());
			break;
		default:
			delete curItem;
			curItem = NULL;
			break;
		}

		//curItem->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y())));

	}

	
	return;
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::updateShapes(const QVector< ito::Shape > elements)
{
	if (elements.size() < 0)
	{
		qDebug("Could not update pickers");
		return;
	}

	for (int curSearchIndex = 0; curSearchIndex < elements.size(); curSearchIndex++)
	{

		QTreeWidgetItem *curItem = NULL;

		for (int idx = 0; idx < topLevelItemCount(); idx++)
		{
			if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == elements[curSearchIndex].index())
			{
				curItem = this->topLevelItem(idx);
				break;
			}
		}

		if (!curItem)
		{
			curItem = new QTreeWidgetItem();
			curItem->setData(0, Qt::UserRole, elements[curSearchIndex].index());

			addTopLevelItem(curItem);
		}

		if (curItem)
		{
			switch (elements[curSearchIndex].type() & ito::Shape::TypeMask)
			{
				case ito::Shape::Point:
					curItem->setData(0, Qt::DisplayRole, QString("Point %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints()); 
					break;
				case ito::Shape::Line:
					curItem->setData(0, Qt::DisplayRole, QString("Line %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints());
					break;
				case ito::Shape::Square:
					curItem->setData(0, Qt::DisplayRole, QString("Sqare %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints()); 
					break;
				case ito::Shape::Rectangle:
					curItem->setData(0, Qt::DisplayRole, QString("Rectangle %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints()); 
					break;
				case ito::Shape::Ellipse:
					curItem->setData(0, Qt::DisplayRole, QString("Ellipse %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints()); 
					break;
				case ito::Shape::Circle:
					curItem->setData(0, Qt::DisplayRole, QString("Circle %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints()); 
					break;
				case ito::Shape::Polygon:
					curItem->setData(0, Qt::DisplayRole, QString("Polygon %1").arg(QString::number(elements[curSearchIndex].index())));
					curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints());
					break;
				default:
					delete curItem;
					curItem = NULL;
					break;
			}
			
			//curItem->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y())));
			
		}

	}
	return;
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeShape(int index)
{
	for (int idx = 0; idx < topLevelItemCount(); idx++)
    {
		if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == index)
        {
			delete topLevelItem(idx);
            break;
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeShapes()
{
	while (topLevelItemCount() > 0)
    {
		delete topLevelItem(topLevelItemCount() - 1);
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::addRelation(const int index1, const int index2, const int relationType)
{

}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeRelation(const int index1, const int index2)
{

}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeRelations(const int index1, const int index2)
{

}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeRelations(const int index)
{

}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeRelations()
{

}
//---------------------------------------------------------------------------------------------------------