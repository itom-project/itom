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

}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Point(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Point %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());

	while (curItem->childCount() > 1)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 1)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	QPointF center = element.centerPoint();
	curItem->child(0)->setData(0, Qt::DisplayRole, "Position");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Line(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Line %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());
	while (curItem->childCount() > 3)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 3)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	const QPolygonF points = element.rbasePoints();
	double length = std::sqrt(std::pow(points[0].x() - points[1].x(), 2) + std::pow(points[0].y() - points[1].y(), 2));
	curItem->child(0)->setData(0, Qt::DisplayRole, "Start");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(points[0].x()), QString::number(points[0].y())));
	curItem->child(1)->setData(0, Qt::DisplayRole, "End");
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(points[1].x()), QString::number(points[1].y())));
	curItem->child(2)->setData(0, Qt::DisplayRole, "Length");
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(length)));
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Circle(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Circle %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());
	while (curItem->childCount() > 2)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 2)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	QPointF center = element.centerPoint();
	double radius = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x()) / 2;
	curItem->child(0)->setData(0, Qt::DisplayRole, "Position");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	curItem->child(1)->setData(0, Qt::DisplayRole, "Radius");
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(radius)));
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Ellipse(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Ellipse %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());
	while (curItem->childCount() > 3)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 3)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	QPointF center = element.centerPoint();

	curItem->child(0)->setData(0, Qt::DisplayRole, "Position");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	double radius = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x()) / 2;
	curItem->child(1)->setData(0, Qt::DisplayRole, "Radius1");
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(radius)));
	radius = std::abs(element.rbasePoints()[0].y() - element.rbasePoints()[1].y()) / 2;
	curItem->child(2)->setData(0, Qt::DisplayRole, "Radius2");
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(radius)));
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Square(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Sqare %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());
	while (curItem->childCount() > 2)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 2)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	QPointF center = element.centerPoint();

	curItem->child(0)->setData(0, Qt::DisplayRole, "Position");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	double side = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x());
	curItem->child(1)->setData(0, Qt::DisplayRole, "SizeX");
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(side)));
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Rect(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Rectangle %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());
	while (curItem->childCount() > 3)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 3)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	QPointF center = element.centerPoint();

	curItem->child(0)->setData(0, Qt::DisplayRole, "Position");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	double side = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x());
	curItem->child(1)->setData(0, Qt::DisplayRole, "SizeX");
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(side)));
	side = std::abs(element.rbasePoints()[0].y() - element.rbasePoints()[1].y());
	curItem->child(2)->setData(0, Qt::DisplayRole, "SizeY");
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(side)));
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::setItem2Poly(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Polygon %1").arg(QString::number(element.index())));
	curItem->setData(1, Qt::UserRole, element.rbasePoints());
	while (curItem->childCount() > 3)
	{
		curItem->removeChild(curItem->child(curItem->childCount() - 1));
	}
	while (curItem->childCount() < 3)
	{
		curItem->addChild(new QTreeWidgetItem());
	}
	QPointF center = element.centerPoint();
	curItem->child(0)->setData(0, Qt::DisplayRole, "Position");
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	curItem->child(1)->setData(0, Qt::DisplayRole, "Length");
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(element.circumference())));
	curItem->child(2)->setData(0, Qt::DisplayRole, "Notes");
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(element.rbasePoints().length())));
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
		curItem->setExpanded(true);
	}

	if (curItem)
	{
		switch (element.type() & ito::Shape::TypeMask)
		{
			case ito::Shape::Point:
			{
				setItem2Point(curItem, element);
			}
			break;
			case ito::Shape::Line:
			{
				setItem2Line(curItem, element);
			}
			break;
			case ito::Shape::Square:
			{
				setItem2Square(curItem, element);
			}
			break;
			case ito::Shape::Rectangle:
			{
				setItem2Rect(curItem, element);
			}
			break;
			case ito::Shape::Ellipse:
			{
				setItem2Ellipse(curItem, element);
			}
			break;
			case ito::Shape::Circle:
			{
				setItem2Circle(curItem, element);
			}
			break;
			case ito::Shape::Polygon:
			{
				setItem2Poly(curItem, element);
			}
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
	for each (const ito::Shape &element in elements)
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
			curItem->setExpanded(true);
		}

		if (curItem)
		{
			switch (element.type() & ito::Shape::TypeMask)
			{
			case ito::Shape::Point:
			{
				setItem2Point(curItem, element);
			}
			break;
			case ito::Shape::Line:
			{
				setItem2Line(curItem, element);
			}
			break;
			case ito::Shape::Square:
			{
				setItem2Square(curItem, element);
			}
			break;
			case ito::Shape::Rectangle:
			{
				setItem2Rect(curItem, element);
			}
			break;
			case ito::Shape::Ellipse:
			{
				setItem2Ellipse(curItem, element);
			}
			break;
			case ito::Shape::Circle:
			{
				setItem2Circle(curItem, element);
			}
			break;
			case ito::Shape::Polygon:
			{
				setItem2Poly(curItem, element);
			}
			break;
			default:
				delete curItem;
				curItem = NULL;
				break;
			}	
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