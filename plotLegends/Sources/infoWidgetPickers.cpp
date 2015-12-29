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
		int position = 0;
		curItem = new QTreeWidgetItem();

		curItem->setData(0, Qt::DisplayRole, QString::number(index));
		curItem->setData(0, Qt::UserRole, index);

		if (topLevelItemCount() != 0)
		{
			while (position < topLevelItemCount() &&  !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
			{
				position++;
			}
		}
		insertTopLevelItem(position, curItem);
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
			int position = 0;
			curItem = new QTreeWidgetItem();

			curItem->setData(0, Qt::DisplayRole, QString::number(indices[curSearchIndex]));
			curItem->setData(0, Qt::UserRole, indices[curSearchIndex]);

			if (topLevelItemCount() != 0)
			{
				while (position < topLevelItemCount() && !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
				{
					position++;
				}
			}
			insertTopLevelItem(position, curItem);
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
		int position = 0;
		curItem = new QTreeWidgetItem();

		curItem->setData(0, Qt::DisplayRole, QString::number(index));
		curItem->setData(0, Qt::UserRole, index);

		if (topLevelItemCount() != 0)
		{
			while (position < topLevelItemCount() &&  !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
			{
				position++;
			}
		}
		insertTopLevelItem(position, curItem);
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
			int position = 0;
			curItem = new QTreeWidgetItem();

			curItem->setData(0, Qt::DisplayRole, QString::number(indices[curSearchIndex]));
			curItem->setData(0, Qt::UserRole, indices[curSearchIndex]);

			if (topLevelItemCount() != 0)
			{
				while (position < topLevelItemCount() &&  !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
				{
					position++;
				}
			}
			insertTopLevelItem(position, curItem);
		}

		if (curItem)
		{
			curItem->setData(1, Qt::DisplayRole, QString("%1, %2, %3").arg(QString::number((positions[curSearchIndex]).x()), QString::number((positions[curSearchIndex]).y()), QString::number((positions[curSearchIndex]).z())));
			curItem->setData(1, Qt::UserRole, positions[curSearchIndex]);
		}

	}

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
	QTreeWidgetItem *pickerEntry = NULL;
	for (int idx = topLevelItemCount(); idx > 0; idx--)
	{
		if (!(topLevelItem(idx)->data(0, Qt::UserRole).toInt() & 0xF000))
		{
			delete topLevelItem(idx - 1);
		}
	}

    return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::updateChildPlot(const int index, const int type, const QVector4D positionAndDirection)
{
	QTreeWidgetItem *entry = NULL;

	int searchIndex0 = index | 0x4000;
	int searchIndex1 = index | 0x8000;

	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		int curIndex = topLevelItem(idx)->data(0, Qt::UserRole).toInt();
		if (curIndex == searchIndex0 ||
			curIndex == searchIndex1)
		{
			entry = this->topLevelItem(idx);
			break;
		}
	}

	if (!entry)
	{
		entry = new QTreeWidgetItem();
		if (type == ito::Shape::Point)
		{
			entry->setData(0, Qt::DisplayRole, QString("Slice %1").arg(QString::number(index)));
			entry->setData(0, Qt::UserRole, searchIndex0);
		}
		else
		{
			entry->setData(0, Qt::DisplayRole, QString("Line %1").arg(QString::number(index)));
			entry->setData(0, Qt::UserRole, searchIndex1);
		}
		addTopLevelItem(entry);
	}

	if (entry)
	{
		if (type == ito::Shape::Point)
		{
			entry->setData(1, Qt::DisplayRole, QString("%1, %2").arg(QString::number(positionAndDirection.x()), QString::number(positionAndDirection.y())));
		}
		else
		{
			entry->setData(1, Qt::DisplayRole, QString("%1, %2 to %3, %4").arg(QString::number(positionAndDirection.x()),
				QString::number(positionAndDirection.y()),
				QString::number(positionAndDirection.z()),
				QString::number(positionAndDirection.w())));
		}
		entry->setData(1, Qt::UserRole, positionAndDirection);
	}

	return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::updateChildPlots(const QVector<int> indices, const QVector<int> type, const QVector<QVector4D> positionAndDirection)
{

	return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::removeChildPlot(int index)
{
	QTreeWidgetItem *pickerEntry = NULL;
	for (int idx = 0; idx < topLevelItemCount(); idx++)
	{
		int indexVal = topLevelItem(idx)->data(0, Qt::UserRole).toInt();
		if (indexVal == (index | 0x8000) || indexVal == (index | 0x4000))
		{
			delete topLevelItem(idx);
			break;
		}
	}
	return;
}
//---------------------------------------------------------------------------------------------------------
void PickerInfoWidget::removeChildPlots()
{
	QTreeWidgetItem *pickerEntry = NULL;
	for (int idx = topLevelItemCount(); idx > 0; idx--)
	{
		if ((topLevelItem(idx - 1)->data(0, Qt::UserRole).toInt() & 0x8000) ||
			(topLevelItem(idx - 1)->data(0, Qt::UserRole).toInt() & 0x4000))
		{
			delete topLevelItem(idx - 1);
		}
	}
	return;
}
//---------------------------------------------------------------------------------------------------------