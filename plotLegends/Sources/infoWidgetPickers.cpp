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

#include "../plotLegends/infoWidgetPickers.h"

//---------------------------------------------------------------------------------------------------------
PickerInfoWidget::PickerInfoWidget(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    setSelectionBehavior( QAbstractItemView::SelectRows );
    setAlternatingRowColors(true);

    clear();

    this->setColumnCount(2);
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::updatePicker(const int index, const QPointF position)
{
	QTreeWidgetItem *curItem = NULL;
	
	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == index)
		{
			curItem = this->topLevelItem(idx);
			break;
		}
	}

	if (!curItem)
	{
		curItem = new QTreeWidgetItem(this);

		curItem->setData(0, Qt::DisplayRole, QString::number(index));
		curItem->setData(0, Qt::UserRole, index);

		this->addTopLevelItem(curItem);
	}

	if (curItem)
	{
		curItem->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number(position.x()), QString::number(position.y())));
		curItem->setData(1, Qt::UserRole, position);
	}

	return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::updatePickers(const QVector< int > indices, const QVector< QPointF> positions)
{
	QTreeWidgetItem *pickerEntries = topLevelItem(0);

	if (indices.size() != positions.size())
	{
		qDebug("Could not update pickers, indices and positions missmatch");
		return;
	}

	for (int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
	{

		QTreeWidgetItem *curItem = NULL;

		for (int idx = 0; idx < topLevelItemCount(); idx++)
		{
			if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == indices[curSearchIndex])
			{
				curItem = this->topLevelItem(idx);
				break;
			}
		}

		if (!curItem)
		{
			curItem = new QTreeWidgetItem(this);

			curItem->setData(0, Qt::DisplayRole, QString::number(indices[curSearchIndex]));
			curItem->setData(0, Qt::UserRole, indices[curSearchIndex]);

			this->addTopLevelItem(curItem);
		}

		if (curItem)
		{
			curItem->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y())));
			curItem->setData(1, Qt::UserRole, positions[curSearchIndex]);
		}

	}

	return;
}

//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::updatePicker(const int index, const QVector3D position)
{
	QTreeWidgetItem *curItem = NULL;

	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == index)
		{
			curItem = this->topLevelItem(idx);
			break;
		}
	}

	if (!curItem)
	{
		curItem = new QTreeWidgetItem(this);

		curItem->setData(0, Qt::DisplayRole, QString::number(index));
		curItem->setData(0, Qt::UserRole, index);

		this->addTopLevelItem(curItem);
	}

	if (curItem)
	{
		curItem->setData(1, Qt::DisplayRole, QString("%1, %2, %3").arg(QString::number(position.x()), QString::number(position.y()), QString::number(position.z())));
		curItem->setData(1, Qt::UserRole, position);
	}

	return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::updatePickers(const QVector< int > indices, const QVector< QVector3D> positions)
{
	QTreeWidgetItem *pickerEntries = topLevelItem(0);

	if (indices.size() != positions.size())
	{
		qDebug("Could not update pickers, indices and positions missmatch");
		return;
	}

	for (int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
	{

		QTreeWidgetItem *curItem = NULL;

		for (int idx = 0; idx < topLevelItemCount(); idx++)
		{
			if (topLevelItem(idx)->data(0, Qt::UserRole).toInt() == indices[curSearchIndex])
			{
				curItem = this->topLevelItem(idx);
				break;
			}
		}

		if (!curItem)
		{
			curItem = new QTreeWidgetItem(this);

			curItem->setData(0, Qt::DisplayRole, QString::number(indices[curSearchIndex]));
			curItem->setData(0, Qt::UserRole, indices[curSearchIndex]);

			this->addTopLevelItem(curItem);
		}

		if (curItem)
		{
			curItem->setData(1, Qt::DisplayRole, QString("%1, %2, %3").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y()), QString::number((positions[curSearchIndex]).z())));
			curItem->setData(1, Qt::UserRole, positions[curSearchIndex]);
		}

	}

	return;
    return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::removePicker(int index)
{
    QTreeWidgetItem *pickerEntry = NULL;
    for(int idx = 0; idx < topLevelItemCount(); idx++)
    {
        if(topLevelItem(idx)->data(0, Qt::UserRole).toInt() == index) 
        {
			delete topLevelItem(idx);
            break;
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::removePickers()
{
	clear();

    return;
}
//---------------------------------------------------------------------------------------------------------