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

#include "plotInfoMarker.h"

#include <QtGui/qpainter.h>

//---------------------------------------------------------------------------------------------------------
PlotInfoMarker::PlotInfoMarker(QWidget* parent /*= NULL*/) : QTreeWidget(parent)
{
    setSelectionBehavior(QAbstractItemView::SelectRows);
    setAlternatingRowColors(true);

    clear();

    setColumnCount(2);

    QStringList headerLabels;
    headerLabels << tr("Property") << tr("Value");
    setHeaderLabels(headerLabels);
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoMarker::updateMarker(const ito::Shape element)
{
    updateMarkers(QVector<ito::Shape>(1, element));
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoMarker::updateMarkers(const QVector<ito::Shape> elements)
{
    QTreeWidgetItem* curItem = nullptr;

    foreach (const ito::Shape &element, elements)
    {
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
            curItem->setFlags(curItem->flags() | Qt::ItemIsUserCheckable);
            curItem->setCheckState(0, Qt::CheckState::Checked);
            curItem->setData(0, Qt::UserRole, element.index());
            addTopLevelItem(curItem);
            curItem->setFirstColumnSpanned(true);
            curItem->setExpanded(true);
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

                foreach (const QPointF& basePoint, element.basePoints())
                {
                    curItem->addChild(new QTreeWidgetItem());
                    int curCnt = curItem->childCount() - 1;
                    curItem->child(curCnt)->setData(
                        0, Qt::DisplayRole, tr("Marker %1").arg(QString::number(curCnt)));
                    curItem->child(curCnt)->setData(
                        1,
                        Qt::DisplayRole,
                        QString("%1, %2").arg(
                            QString::number(basePoint.x()), QString::number(basePoint.y())));
                }
                break;
            default:
                delete curItem;
                curItem = nullptr;
                break;
            }
        }
    }

    return;
}
//---------------------------------------------------------------------------------------------------------
void PlotInfoMarker::removeMarker(const QString setName)
{
    for (int idx = 0; idx < topLevelItemCount(); idx++)
    {
        if (topLevelItem(idx)->data(0, Qt::DisplayRole).toString() == setName)
        {
            delete topLevelItem(idx);
            break;
        }
    }
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoMarker::removeMarkers()
{
    while (topLevelItemCount() > 0)
    {
        delete topLevelItem(topLevelItemCount() - 1);
    }
}

//---------------------------------------------------------------------------------------------------------
void PlotInfoMarker::checkMarkers(const QString setName, bool checked)
{
    blockSignals(true);

    for (int idx = 0; idx < topLevelItemCount(); idx++)
    {
        if (topLevelItem(idx)->data(0, Qt::DisplayRole).toString() == setName)
        {
            topLevelItem(idx)->setCheckState(0, checked ? Qt::Checked : Qt::Unchecked);
        }
    }

    blockSignals(false);
}

//---------------------------------------------------------------------------------------------------------
QPainterPath PlotInfoMarker::renderToPainterPath(
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
//---------------------------------------------------------------------------------------------------------
