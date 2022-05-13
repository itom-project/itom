/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2022, Institut fuer Technische Optik (ITO),
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

#include "plotInfoPicker.h"

#include <QtGui/qpainter.h>

//-------------------------------------------------------------------------------------
PlotInfoPicker::PlotInfoPicker(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setAlternatingRowColors(true);

    clear();

    setColumnCount(2);

    QStringList headerLabels;
    headerLabels << tr("Property") << tr("Value");
    setHeaderLabels(headerLabels);
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updatePicker(const int index, const QPointF position)
{
    QTreeWidgetItem* curItem = NULL;

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
            while (position < topLevelItemCount() &&
                   !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
            {
                position++;
            }
        }
        insertTopLevelItem(position, curItem);
    }

    if (curItem)
    {
        curItem->setData(
            1,
            Qt::DisplayRole,
            QString("%1, %2").arg(QString::number(position.x()), QString::number(position.y())));
        curItem->setData(1, Qt::UserRole, position);
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updatePickers(const QVector<int> indices, const QVector<QPointF> positions)
{
    if (indices.size() != positions.size())
    {
        qDebug("Could not update pickers, indices and positions missmatch");
        return;
    }

    for (int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
    {
        QTreeWidgetItem* curItem = nullptr;

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
                while (position < topLevelItemCount() &&
                       !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
                {
                    position++;
                }
            }
            insertTopLevelItem(position, curItem);
        }

        if (curItem)
        {
            curItem->setData(
                1,
                Qt::DisplayRole,
                QString("%1, %2").arg(
                    QString::number((positions[curSearchIndex]).x()),
                    QString::number((positions[curSearchIndex]).y())));
            curItem->setData(1, Qt::UserRole, positions[curSearchIndex]);
        }
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updatePickers(
    const QVector<int> indices,
    const QVector<QDateTime>& xpositions,
    const QVector<qreal>& ypositions)
{
    QString date;

    if (indices.size() != xpositions.size() || indices.size() != ypositions.size())
    {
        qDebug("Could not update pickers, indices and positions missmatch");
        return;
    }

    for (int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
    {
        QTreeWidgetItem* curItem = nullptr;

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
                while (position < topLevelItemCount() &&
                       !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
                {
                    position++;
                }
            }
            insertTopLevelItem(position, curItem);
        }

        if (curItem)
        {
            if (xpositions[curSearchIndex].time().msec() != 0)
            {
                date = xpositions[curSearchIndex].toString(Qt::ISODateWithMs);
            }
            else
            {
                date = xpositions[curSearchIndex].toString(Qt::DateFormat::ISODate);
            }

            curItem->setData(
                1,
                Qt::DisplayRole,
                QString("%1, %2").arg(
                    date,
                    QString::number(ypositions[curSearchIndex])));
            //curItem->setData(1, Qt::UserRole, positions[curSearchIndex]);
        }
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updatePicker(const int index, const QVector3D position)
{
    QTreeWidgetItem* curItem = nullptr;

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
            while (position < topLevelItemCount() &&
                   !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
            {
                position++;
            }
        }
        insertTopLevelItem(position, curItem);
    }

    if (curItem)
    {
        curItem->setData(
            1,
            Qt::DisplayRole,
            QString("%1, %2, %3")
                .arg(
                    QString::number(position.x()),
                    QString::number(position.y()),
                    QString::number(position.z())));
        curItem->setData(1, Qt::UserRole, position);
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updatePickers(const QVector<int> indices, const QVector<QVector3D> positions)
{
    if (indices.size() != positions.size())
    {
        qDebug("Could not update pickers, indices and positions missmatch");
        return;
    }

    for (int curSearchIndex = 0; curSearchIndex < indices.size(); curSearchIndex++)
    {
        QTreeWidgetItem* curItem = nullptr;

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
                while (position < topLevelItemCount() &&
                       !(topLevelItem(position)->data(0, Qt::UserRole).toInt() & 0xF000))
                {
                    position++;
                }
            }

            insertTopLevelItem(position, curItem);
        }

        if (curItem)
        {
            curItem->setData(
                1,
                Qt::DisplayRole,
                QString("%1, %2, %3")
                    .arg(
                        QString::number((positions[curSearchIndex]).x()),
                        QString::number((positions[curSearchIndex]).y()),
                        QString::number((positions[curSearchIndex]).z())));
            curItem->setData(1, Qt::UserRole, positions[curSearchIndex]);
        }
    }
}
//-------------------------------------------------------------------------------------
void PlotInfoPicker::removePicker(int index)
{
    QTreeWidgetItem* pickerEntry = nullptr;

    for (int idx = 0; idx < topLevelItemCount(); idx++)
    {
        if (index == -1 || topLevelItem(idx)->data(0, Qt::UserRole).toInt() == index)
        {
            delete topLevelItem(idx);
            break;
        }
    }

    return;
}
//-------------------------------------------------------------------------------------
void PlotInfoPicker::removePickers()
{
    QTreeWidgetItem* pickerEntry = nullptr;

    for (int idx = topLevelItemCount() - 1; idx >= 0; idx--)
    {
        if (!topLevelItem(idx))
            continue;

        QVariant test = topLevelItem(idx)->data(0, Qt::UserRole);

        if (!(test.toInt() & 0xF000))
        {
            delete topLevelItem(idx);
        }
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updateChildPlot(
    const int index, const int type, const QVector4D positionAndDirection)
{
    QTreeWidgetItem* entry = NULL;

    int searchIndex0 = index | 0x4000;
    int searchIndex1 = index | 0x8000;

    for (int idx = 0; idx < topLevelItemCount(); idx++)
    {
        int curIndex = topLevelItem(idx)->data(0, Qt::UserRole).toInt();
        if (curIndex == searchIndex0 || curIndex == searchIndex1)
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
            entry->setData(0, Qt::DisplayRole, QString("z-Slice"));
            entry->setData(0, Qt::UserRole, searchIndex0);
        }
        else
        {
            entry->setData(0, Qt::DisplayRole, QString("LineCut"));
            entry->setData(0, Qt::UserRole, searchIndex1);
        }
        addTopLevelItem(entry);
        entry->addChild(new QTreeWidgetItem());
        entry->child(0)->setData(0, Qt::DisplayRole, "Unique ID");
        entry->addChild(new QTreeWidgetItem());
        entry->child(1)->setData(0, Qt::DisplayRole, "Position");
        if (type == ito::Shape::Line)
        {
            entry->addChild(new QTreeWidgetItem());
            entry->child(2)->setData(0, Qt::DisplayRole, "End");
            entry->addChild(new QTreeWidgetItem());
            entry->child(3)->setData(0, Qt::DisplayRole, "Length");
        }
        entry->setExpanded(true);
    }

    if (entry)
    {
        entry->setData(1, Qt::UserRole, positionAndDirection);
        entry->child(0)->setData(1, Qt::DisplayRole, QString("%1").arg(QString::number(index)));
        entry->child(1)->setData(
            1,
            Qt::DisplayRole,
            QString("%1, %2").arg(
                QString::number(positionAndDirection.x()),
                QString::number(positionAndDirection.y())));
        if (type == ito::Shape::Line)
        {
            entry->child(2)->setData(
                1,
                Qt::DisplayRole,
                QString("%1, %2").arg(
                    QString::number(positionAndDirection.z()),
                    QString::number(positionAndDirection.w())));
            double length = std::sqrt(
                std::pow(positionAndDirection.x() - positionAndDirection.z(), 2) +
                std::pow(positionAndDirection.y() - positionAndDirection.w(), 2));
            entry->child(3)->setData(
                1, Qt::DisplayRole, QString("%1").arg(QString::number(length)));
        }
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::updateChildPlots(
    const QVector<int> indices,
    const QVector<int> type,
    const QVector<QVector4D> positionAndDirection)
{
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::removeChildPlot(int index)
{
    QTreeWidgetItem* pickerEntry = nullptr;

    for (int idx = 0; idx < topLevelItemCount(); idx++)
    {
        int indexVal = topLevelItem(idx)->data(0, Qt::UserRole).toInt();

        if (indexVal == (index | 0x8000) || indexVal == (index | 0x4000))
        {
            delete topLevelItem(idx);
            break;
        }
    }
}

//-------------------------------------------------------------------------------------
void PlotInfoPicker::removeChildPlots()
{
    QTreeWidgetItem* pickerEntry = nullptr;

    for (int idx = topLevelItemCount(); idx > 0; idx--)
    {
        if ((topLevelItem(idx - 1)->data(0, Qt::UserRole).toInt() & 0x8000) ||
            (topLevelItem(idx - 1)->data(0, Qt::UserRole).toInt() & 0x4000))
        {
            delete topLevelItem(idx - 1);
        }
    }
}
//-------------------------------------------------------------------------------------
QPainterPath PlotInfoPicker::renderToPainterPath(
    const int xsize, const int ysize, const int fontSize)
{
    QPainterPath destinationPath(QPoint(0, 0));

    int ySpacing = 12;
    int ySpacingTp = 6;
    int xSpacing = 10;
    int yStartPos = 5;
    int linesize = iconSize().height() + ySpacing;

    // if(m_pContent->topLevelItemCount() > 0) yStartPos = (m_pContent->iconSize().height() -
    // m_pContent->topLevelItem(0)->font(0).pixelSize()) / 2;

    QPoint pos(iconSize().width() + 4, yStartPos);
    QPoint posI(0, 0);

    for (int topItem = 0; topItem < topLevelItemCount(); topItem++)
    {
        pos.setX(iconSize().width() + xSpacing);
        posI.setX(0);
        destinationPath.addText(
            pos, topLevelItem(topItem)->font(0), topLevelItem(topItem)->text(0));
        // painter.drawPixmap(posI, topLevelItem(topItem)->icon(0).pixmap(iconSize()));
        pos.setY(pos.y() + linesize);
        posI.setY(posI.y() + linesize);

        if (topLevelItem(topItem)->childCount() > 0)
        {
            pos.setX(30 + iconSize().width() + xSpacing);
            posI.setX(30);

            for (int childItem = 0; childItem < topLevelItem(topItem)->childCount(); childItem++)
            {
                destinationPath.addText(
                    pos,
                    topLevelItem(topItem)->child(childItem)->font(0),
                    topLevelItem(topItem)->child(childItem)->text(0));
                // painter.drawPixmap(posI,
                // topLevelItem(topItem)->child(childItem)->icon(0).pixmap(iconSize()));
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
            destinationPath.addText(
                pos, topLevelItem(topItem)->font(0), topLevelItem(topItem)->text(col));
            pos.setY(pos.y() + linesize);

            if (topLevelItem(topItem)->childCount() > 0)
            {
                for (int childItem = 0; childItem < topLevelItem(topItem)->childCount();
                     childItem++)
                {
                    destinationPath.addText(
                        pos,
                        topLevelItem(topItem)->child(childItem)->font(0),
                        topLevelItem(topItem)->child(childItem)->text(col));
                    pos.setY(pos.y() + linesize);
                }
            }
            pos.setY(pos.y() + ySpacingTp);
        }
    }

    return destinationPath;
}
