/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    University of Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
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

#include "plotInfoShapes.h"

#include <QtGui/qpainter.h>
#include <qpolygon.h>

//---------------------------------------------------------------------------------------------------------
PlotInfoShapes::PlotInfoShapes(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    m_onlyTwoDims = true;
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    clear();

    setColumnCount(2);

    QStringList headerLabels;
    headerLabels << tr("Property") << tr("Value");
    setHeaderLabels(headerLabels);

}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::adjustNumberOfChildItems(QTreeWidgetItem* curItem, int count)
{
    while (curItem->childCount() > count)
    {
        curItem->removeChild(curItem->child(curItem->childCount() - 1));
    }
    while (curItem->childCount() < count)
    {
        curItem->addChild(new QTreeWidgetItem());
    }
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Point(QTreeWidgetItem* curItem, const ito::Shape &element)
{
    curItem->setData(0, Qt::DisplayRole, tr("Point %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 1);

	QPointF center = element.centerPoint();
	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Position"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Line(QTreeWidgetItem* curItem, const ito::Shape &element)
{
    curItem->setData(0, Qt::DisplayRole, tr("Line %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 3);

	const QPolygonF points = element.rbasePoints();
	double length = std::sqrt(std::pow(points[0].x() - points[1].x(), 2) + std::pow(points[0].y() - points[1].y(), 2));
	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Start"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(points[0].x()), QString::number(points[0].y())));
	curItem->child(1)->setData(0, Qt::DisplayRole, tr("End"));
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(points[1].x()), QString::number(points[1].y())));
	curItem->child(2)->setData(0, Qt::DisplayRole, tr("Length"));
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(length)));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Circle(QTreeWidgetItem* curItem, const ito::Shape &element)
{
    curItem->setData(0, Qt::DisplayRole, tr("Circle %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 3);

	QPointF center = element.centerPoint();
	double radius = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x()) / 2;
	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Center"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	curItem->child(1)->setData(0, Qt::DisplayRole, tr("Radius"));
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(radius)));
    curItem->child(2)->setData(0, Qt::DisplayRole, tr("Rotation"));
    curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1%2").arg(QString::number(element.rotationAngleDeg())).arg(QChar(0xB0, 0x00)));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Ellipse(QTreeWidgetItem* curItem, const ito::Shape &element)
{
    curItem->setData(0, Qt::DisplayRole, tr("Ellipse %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 4);
	QPointF center = element.centerPoint();

	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Center"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	double radius = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x()) / 2;
	curItem->child(1)->setData(0, Qt::DisplayRole, tr("Radius 1"));
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(radius)));
	radius = std::abs(element.rbasePoints()[0].y() - element.rbasePoints()[1].y()) / 2;
	curItem->child(2)->setData(0, Qt::DisplayRole, tr("Radius 2"));
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(radius)));
    curItem->child(3)->setData(0, Qt::DisplayRole, tr("Rotation"));
    curItem->child(3)->setData(1, Qt::DisplayRole, QString("%1%2").arg(QString::number(element.rotationAngleDeg())).arg(QChar(0xB0, 0x00)));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Square(QTreeWidgetItem* curItem, const ito::Shape &element)
{
    curItem->setData(0, Qt::DisplayRole, QString("Square %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 3);
	QPointF center = element.centerPoint();

	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Center"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	double side = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x());
	curItem->child(1)->setData(0, Qt::DisplayRole, tr("Side Length"));
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(side)));
    curItem->child(2)->setData(0, Qt::DisplayRole, tr("Rotation"));
    curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1%2").arg(QString::number(element.rotationAngleDeg())).arg(QChar(0xB0, 0x00)));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Rect(QTreeWidgetItem* curItem, const ito::Shape &element)
{
	curItem->setData(0, Qt::DisplayRole, QString("Rectangle %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 4);
	QPointF center = element.centerPoint();

	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Center"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	double side = std::abs(element.rbasePoints()[0].x() - element.rbasePoints()[1].x());
	curItem->child(1)->setData(0, Qt::DisplayRole, tr("Width"));
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(side)));
	side = std::abs(element.rbasePoints()[0].y() - element.rbasePoints()[1].y());
	curItem->child(2)->setData(0, Qt::DisplayRole, tr("Height"));
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(side)));
    curItem->child(3)->setData(0, Qt::DisplayRole, tr("Rotation"));
    curItem->child(3)->setData(1, Qt::DisplayRole, QString("%1%2").arg(QString::number(element.rotationAngleDeg())).arg(QChar(0xB0, 0x00)));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::setItem2Poly(QTreeWidgetItem* curItem, const ito::Shape &element)
{
    curItem->setData(0, Qt::DisplayRole, QString("Polygon %1").arg(QString::number(element.index())));
    if (element.name() != "")
    {
        curItem->setData(1, Qt::DisplayRole, element.name());
    }

	curItem->setData(1, Qt::UserRole, element.rbasePoints());

    adjustNumberOfChildItems(curItem, 3);
	QPointF center = element.centerPoint();
	curItem->child(0)->setData(0, Qt::DisplayRole, tr("Center"));
	curItem->child(0)->setData(1, Qt::DisplayRole, QString("%1; %2").arg(QString::number(center.x()), QString::number(center.y())));
	curItem->child(1)->setData(0, Qt::DisplayRole, tr("Length"));
	curItem->child(1)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(element.circumference())));
	curItem->child(2)->setData(0, Qt::DisplayRole, tr("Nodes"));
	curItem->child(2)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(element.rbasePoints().size())));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::updateShape(const ito::Shape element)
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
        curItem->setFirstColumnSpanned(true);

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
void PlotInfoShapes::updateShapes(const QVector< ito::Shape > elements)
{
	foreach (const ito::Shape &element, elements)
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
            curItem->setFirstColumnSpanned(true);

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
void PlotInfoShapes::removeShape(int index)
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
void PlotInfoShapes::removeShapes()
{
	while (topLevelItemCount() > 0)
    {
		delete topLevelItem(topLevelItemCount() - 1);
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::addRelation(const int index1, const int index2, const int relationType)
{

}
//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::removeRelation(const int index1, const int index2)
{

}
//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::removeRelations(const int index1, const int index2)
{

}
//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::removeRelations(const int index)
{

}
//---------------------------------------------------------------------------------------------------------
void PlotInfoShapes::removeRelations()
{

}
//---------------------------------------------------------------------------------------------------------
QPainterPath PlotInfoShapes::renderToPainterPath(const int xsize, const int ysize, const int fontSize)
{
	QPainterPath destinationPath(QPoint(0,0));

	int ySpacing = 12;
	int ySpacingTp = 6;
	int xSpacing = 10;
	int yStartPos = 5;
	int linesize = iconSize().height() + ySpacing;

	//if(m_pContent->topLevelItemCount() > 0) yStartPos = (m_pContent->iconSize().height() - m_pContent->topLevelItem(0)->font(0).pixelSize()) / 2;

	QPoint pos(iconSize().width() + 4, yStartPos);
	QPoint posI(0, 0);

	for (int topItem = 0; topItem < topLevelItemCount(); topItem++)
	{
		pos.setX(iconSize().width() + xSpacing);
		posI.setX(0);
		destinationPath.addText(pos, topLevelItem(topItem)->font(0), topLevelItem(topItem)->text(0));
		//painter.drawPixmap(posI, topLevelItem(topItem)->icon(0).pixmap(iconSize()));
		pos.setY(pos.y() + linesize);
		posI.setY(posI.y() + linesize);
		if (topLevelItem(topItem)->childCount() > 0)
		{
			pos.setX(30 + iconSize().width() + xSpacing);
			posI.setX(30);
			for (int childItem = 0; childItem < topLevelItem(topItem)->childCount(); childItem++)
			{
				destinationPath.addText(pos, topLevelItem(topItem)->child(childItem)->font(0), topLevelItem(topItem)->child(childItem)->text(0));
				//painter.drawPixmap(posI, topLevelItem(topItem)->child(childItem)->icon(0).pixmap(iconSize()));
				pos.setY(pos.y() + linesize);
				posI.setY(posI.y() + linesize);
			}
		}
		pos.setY(pos.y() + ySpacingTp);
		posI.setY(posI.y() + ySpacingTp);
	}
	pos.setX(0);
	for (int col = 1; col < columnCount(); col++)
	{
		pos.setX(pos.x() + columnWidth(col - 1) + xSpacing);
		pos.setY(yStartPos);
		for (int topItem = 0; topItem < topLevelItemCount(); topItem++)
		{

			destinationPath.addText(pos, topLevelItem(topItem)->font(0), topLevelItem(topItem)->text(col));
			pos.setY(pos.y() + linesize);

			if (topLevelItem(topItem)->childCount() > 0)
			{
				for (int childItem = 0; childItem < topLevelItem(topItem)->childCount(); childItem++)
				{
					destinationPath.addText(pos, topLevelItem(topItem)->child(childItem)->font(0), topLevelItem(topItem)->child(childItem)->text(col));
					pos.setY(pos.y() + linesize);
				}
			}
			pos.setY(pos.y() + ySpacingTp);
		}
	}

	return destinationPath;
}
//---------------------------------------------------------------------------------------------------------
