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
void ShapesInfoWidget::updateShape(const int index, const ito::Shape element)
{


    return;
}
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::updateShapes(const QVector< ito::Shape > elements)
{

    return;
}
//---------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------
void ShapesInfoWidget::removeShape(int index)
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
void ShapesInfoWidget::removeShapes()
{
	QTreeWidgetItem *shapeEntries = topLevelItem(1);
	while (shapeEntries->childCount() > 0)
    {
		shapeEntries->removeChild(shapeEntries->child(shapeEntries->childCount() - 1));
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