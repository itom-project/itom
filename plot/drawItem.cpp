/* ********************************************************************
 i tom measurement sys*tem
 URL: http://www.uni-stuttgart.de/ito
 Copyright (C) 2012, Institut für Technische Optik (ITO),
 Universität Stuttgart, Germany

 This file is part of itom.

 itom is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 itom is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with itom. If not, see <http://www.gnu.org/licenses/>.
 *********************************************************************** */

#include "drawItem.h"
#include "common/sharedStructuresPrimitives.h"

#include <qwt_symbol.h>

QVector<int> DrawItem::idxVec;

//----------------------------------------------------------------------------------------------------------------------------------
DrawItem::DrawItem(QwtPlot *parent, char type, int id, const QString &title) : m_pparent(parent), m_type(type), m_active(0), m_idx(0), x1(-1), y1(-1), m_autoColor(true),
    x2(-1), y2(-1), m_selected(false)
{

    if (id <= 0)
    {
        int idxCtr = 0;
        do 
            idxCtr++;
        while (idxVec.contains(idxCtr));
        m_idx = idxCtr;
        idxVec.append(m_idx);
    }
    else 
    {
        m_idx = id;
        idxVec.append(id);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
DrawItem::~DrawItem()
{
    detach();
    for (int n = 0; n < m_marker.size(); n++)
    {
        m_marker[n]->detach();
//        m_marker.remove(n);
        delete m_marker[n];
    }
    m_marker.clear();
    idxVec.remove(idxVec.indexOf(m_idx));
}
//----------------------------------------------------------------------------------------------------------------------------------
void DrawItem::setSelected(const bool selected)
{
    m_selected = selected;

    if(m_type == ito::PrimitiveContainer::tPoint && m_marker.size() > 0)
    {
        QColor markerColor = m_marker[0]->linePen().color();
        m_marker[0]->setSymbol(new QwtSymbol(selected ? QwtSymbol::Rect : QwtSymbol::Triangle, QBrush(markerColor),
            QPen(QBrush(markerColor), 1), selected ? QSize(9,9) : QSize(7,7) ));
    }
    setPen(pen().color(), m_selected ? 3 : 1);
    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DrawItem::selected() const
{
    return m_selected;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DrawItem::setActive(int active)
{
    QColor markerColor = Qt::green;
    if(m_marker.size()) markerColor = m_marker[0]->linePen().color();

    if(m_type == ito::PrimitiveContainer::tPoint)
    {
        for (int n = 0; n < m_marker.size(); n++)
        {
            m_marker[n]->setLinePen(QPen(markerColor));
            m_marker[n]->setSymbol(new QwtSymbol(QwtSymbol::Triangle,QBrush(markerColor),
                QPen(QBrush(markerColor), 1),  QSize(7,7) ));
        }
    }
    else
    {
        for (int n = 0; n < m_marker.size(); n++)
        {
            m_marker[n]->setSymbol(new QwtSymbol(QwtSymbol::Diamond,QBrush(markerColor),
                QPen(QBrush(markerColor), 1),  QSize(7,7) ));
        }
    }

    if (active == 1)
    {
        if (m_marker.size() >= 1)
            m_marker[0]->setSymbol(new QwtSymbol(QwtSymbol::Rect, QBrush(markerColor),
                QPen(QBrush(markerColor), 1),  QSize(9,9) ));
    }
    else if (active == 2)
    {
        if (m_marker.size() >= 2)
            m_marker[1]->setSymbol(new QwtSymbol(QwtSymbol::Rect, QBrush(markerColor),
                QPen(QBrush(markerColor), 1),  QSize(9,9) ));
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
void DrawItem::setColor(const QColor &markerColor, const QColor &lineColor)
{
    if(m_type == ito::PrimitiveContainer::tPoint)
    {
        for (int n = 0; n < m_marker.size(); n++)
        {
            m_marker[n]->setLinePen(QPen(markerColor));
            m_marker[n]->setSymbol(new QwtSymbol(QwtSymbol::Triangle,QBrush(markerColor),
                QPen(QBrush(markerColor), 1),  QSize(7,7) ));
        }
    }
    else
    {
        for (int n = 0; n < m_marker.size(); n++)
        {
            m_marker[n]->setSymbol(new QwtSymbol(QwtSymbol::Diamond,QBrush(markerColor),
                QPen(QBrush(markerColor), 1),  QSize(7,7) ));
        }
    }
    setPen(QColor(lineColor), m_selected ? 3 : 1);
}
//----------------------------------------------------------------------------------------------------------------------------------
void DrawItem::setShape(const QPainterPath &path, const QColor &firstColor, const QColor &secondColor)
{
    QwtPlotMarker *marker = NULL;
    int numOfElements = path.elementCount();

    if (numOfElements <= 0)
        return;
    
    setPen(firstColor, m_selected ? 3 : 1);

    QwtPlotShapeItem::setShape(path);
    
    if (m_marker.size() > 0)
    {

        for (int n = 0; n < m_marker.size(); n++)
        {
            m_marker[n]->detach();
//            m_marker.remove(n);
            delete m_marker[n];
        }
        m_marker.clear();
    }
    //if (path.length() >= 1) // len gives the physical length, not the number of elements!!!
    if (numOfElements >= 1)
    {
        QPainterPath::Element el;
        marker = new QwtPlotMarker();


        if(m_type == ito::PrimitiveContainer::tPoint)
        {
            marker->setLinePen(QPen(secondColor));
            marker->setSymbol(new QwtSymbol(QwtSymbol::Triangle,QBrush(secondColor),
                QPen(QBrush(secondColor), 1),  QSize(7,7) ));
        }
        else
        {
            marker->setLinePen(QPen(secondColor));
            marker->setSymbol(new QwtSymbol(QwtSymbol::Diamond,QBrush(secondColor),
                QPen(QBrush(secondColor), 1),  QSize(7,7) ));
            
        }

        /*
        if (secondColor.isValid())
        {
            marker->setLinePen(QPen(secondColor));
        }
        else
        {
            marker->setLinePen(QPen(Qt::green));
        }
        */


        switch (m_type)
        {
            default:
            case ito::PrimitiveContainer::tPoint:
            case ito::PrimitiveContainer::tLine:
            case ito::PrimitiveContainer::tRectangle:
                el = path.elementAt(0);
                x1 = el.x;
                y1 = el.y;
            break;

            case ito::PrimitiveContainer::tEllipse:
                //if (path.length() >= 7) // len gives the physical length, not the number of elements!!!
                if (numOfElements >= 7)
                {
                    el = path.elementAt(6);
                    x1 = el.x;
                }
                //if (path.length() >= 10) // len gives the physical length, not the number of elements!!!
                if (numOfElements >= 10)
                {
                    el = path.elementAt(9);
                    y1 = el.y;
                }
            break;

        }

        marker->setXValue(x1);
        marker->setYValue(y1);
        marker->setVisible(true);
        marker->attach(m_pparent);
        m_marker.append(marker);
//        m_active = 1;
    }
    //if (path.length() >= 2) // len gives the physical length, not the number of elements!!!
    if (numOfElements >= 2)
    {
        QPainterPath::Element el;
        marker = new QwtPlotMarker();
        marker->setLinePen(QPen(secondColor));
        marker->setSymbol(new QwtSymbol(QwtSymbol::Diamond,QBrush(secondColor),
            QPen(QBrush(secondColor), 1),  QSize(7,7) ));
        

        switch (m_type)
        {
            default:
            case ito::PrimitiveContainer::tLine:
                el = path.elementAt(1);
                x2 = el.x;
                y2 = el.y;
            break;

            case ito::PrimitiveContainer::tRectangle:
                //if (path.length() >= 3) // len gives the physical length, not the number of elements!!!
                if (numOfElements >= 3)
                {
                    el = path.elementAt(2);
                    x2 = el.x;
                    y2 = el.y;
                }
            break;

            case ito::PrimitiveContainer::tEllipse:
                el = path.elementAt(0);
                x2 = el.x;
                //if (path.length() >= 4) // len gives the physical length, not the number of elements!!!
                if (numOfElements >= 4)
                {
                    el = path.elementAt(3);
                    y2 = el.y;
                }
            break;

        }

        marker->setXValue(x2);
        marker->setYValue(y2);
        marker->setVisible(true);
        marker->attach(m_pparent);
        m_marker.append(marker);
//        m_active = 2;
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
