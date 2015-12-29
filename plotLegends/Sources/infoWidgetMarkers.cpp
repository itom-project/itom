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

#include "../plotLegends/infoWidgetMarkers.h"

//---------------------------------------------------------------------------------------------------------
MarkerInfoWidget::MarkerInfoWidget(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    clear();

    this->setColumnCount(2);

}
//---------------------------------------------------------------------------------------------------------
void MarkerInfoWidget::updateMarker(const ito::Shape element)
{

	QTreeWidgetItem *curItem = NULL;

	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		if (topLevelItem(idx)->data(0, Qt::DisplayRole).toString() == element.name())
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
		while (curItem->childCount() > 0)
		{
			curItem->removeChild(curItem->child(curItem->childCount() - 1));
		}

		switch (element.type() & ito::Shape::TypeMask)
		{
		case ito::Shape::MultiPointPick:
			if (element.index() < 0)
			{
				curItem->setData(0, Qt::DisplayRole, "Picked Group");
			}
			else
			{
				curItem->setData(0, Qt::DisplayRole, element.name());
			}
			
			curItem->setData(1, Qt::UserRole, element.basePoints());
			for each (QPointF basePoint in element.basePoints())
			{
				curItem->addChild(new QTreeWidgetItem());
				curItem->child(curItem->childCount()-1)->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number(basePoint.x()), QString::number(basePoint.y())));
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
void MarkerInfoWidget::updateMarkers(const QVector< ito::Shape > elements)
{

	for (int curSearchIndex = 0; curSearchIndex < elements.size(); curSearchIndex++)
	{
		QTreeWidgetItem *curItem = NULL;

		for (int idx = 0; idx < topLevelItemCount(); idx++)
		{
			if (topLevelItem(idx)->data(0, Qt::DisplayRole).toString() == elements[curSearchIndex].name())
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
			while (curItem->childCount() > 0)
			{
				curItem->removeChild(curItem->child(curItem->childCount() - 1));
			}

			switch (elements[curSearchIndex].type() & ito::Shape::TypeMask)
			{
			case ito::Shape::MultiPointPick:
				if (elements[curSearchIndex].index() < 0)
				{
					curItem->setData(0, Qt::DisplayRole, "Picked Group");
				}
				else
				{
					curItem->setData(0, Qt::DisplayRole, elements[curSearchIndex].name());
				}				
				curItem->setData(1, Qt::UserRole, elements[curSearchIndex].basePoints());
				for each (QPointF basePoint in elements[curSearchIndex].basePoints())
				{
					curItem->addChild(new QTreeWidgetItem());
					curItem->child(curItem->childCount() - 1)->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number(basePoint.x()), QString::number(basePoint.y())));
				}
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
void MarkerInfoWidget::removeMarker(const QString setName)
{
	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		if (topLevelItem(idx)->data(0, Qt::UserRole).toString() == setName)
		{
			delete topLevelItem(idx);
			break;
		}
	}

	return;
}
//---------------------------------------------------------------------------------------------------------
void MarkerInfoWidget::removeMarkers()
{
	while (topLevelItemCount() > 0)
	{
		delete topLevelItem(topLevelItemCount() - 1);
	}

	return;
}
//---------------------------------------------------------------------------------------------------------