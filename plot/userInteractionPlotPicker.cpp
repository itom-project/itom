/* ********************************************************************
   itom measurement system
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

#include "userInteractionPlotPicker.h"

#include <qwt_picker_machine.h>
#include <qwt_painter.h>

//----------------------------------------------------------------------------------------------------------------------------------
void UserInteractionPlotPicker::reset()
{
    //at the beginning no point is clicked, nevertheless the Abort-Key should abort the selection and
    // send activated(false) such that itom is able to continue
    if (isEnabled() && !isActive())
    {
        emit activated(false);
    }

    QwtPlotPicker::reset();
}

//----------------------------------------------------------------------------------------------------------------------------------
void UserInteractionPlotPicker::setBackgroundFillBrush( const QBrush &brush )
{
    if(brush != this->m_rectFillBrush)
    {
        m_rectFillBrush = brush;
        updateDisplay();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void UserInteractionPlotPicker::drawTracker( QPainter *painter ) const
{
    const QRect textRect = trackerRect( painter->font() );
    if ( !textRect.isEmpty() )
    {
        const QwtText label = trackerText( trackerPosition() );
        if ( !label.isEmpty() )
        {
            painter->fillRect(textRect, m_rectFillBrush);
            label.draw( painter, textRect );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void UserInteractionPlotPicker::drawRubberBand( QPainter *painter ) const
{
    if ( !isActive() || rubberBand() == NoRubberBand ||
        rubberBandPen().style() == Qt::NoPen )
    {
        return;
    }

    const QPolygon pa = adjustedPoints( pickedPoints() );

    QwtPickerMachine::SelectionType selectionType =
        QwtPickerMachine::NoSelection;

    if ( stateMachine() )
        selectionType = stateMachine()->selectionType();

    switch ( selectionType )
    {
        case QwtPickerMachine::NoSelection:
        case QwtPickerMachine::PointSelection:
        {
            if ( pa.count() < 1 )
                return;

            const QPoint pos = pa[0];

            const QRect pRect = pickArea().boundingRect().toRect();
            switch ( rubberBand() )
            {
                case VLineRubberBand:
                {
                    QwtPainter::drawLine( painter, pos.x(),
                        pRect.top(), pos.x(), pRect.bottom() );
                    break;
                }
                case HLineRubberBand:
                {
                    QwtPainter::drawLine( painter, pRect.left(),
                        pos.y(), pRect.right(), pos.y() );
                    break;
                }
                case CrossRubberBand:
                {
                    QwtPainter::drawLine( painter, pos.x(),
                        pRect.top(), pos.x(), pRect.bottom() );
                    QwtPainter::drawLine( painter, pRect.left(),
                        pos.y(), pRect.right(), pos.y() );
                    break;
                }
                default:
                    break;
            }
            break;
        }
        case QwtPickerMachine::RectSelection:
        {
            if ( pa.count() < 2 )
                return;

            const QRect rect = QRect( pa.first(), pa.last() ).normalized();
            switch ( rubberBand() )
            {
                case EllipseRubberBand:
                {
                    QwtPainter::drawEllipse( painter, rect );
                    break;
                }
                case RectRubberBand:
                {
                    QwtPainter::drawRect( painter, rect );
                    break;
                }
                default:
                    break;
            }
            break;
        }
        case QwtPickerMachine::PolygonSelection:
        {
            if ( rubberBand() == PolygonRubberBand )
            {
                painter->drawPolyline( pa );
            }
            else if ( rubberBand() == QwtPicker::UserRubberBand )
            {
                if (pa.size() > 0)
                {
                    const QPoint pos = pa.last();

                    const QRect pRect = pickArea().boundingRect().toRect();

                    QwtPainter::drawLine( painter, pos.x(),
                            pRect.top(), pos.x(), pRect.bottom() );
                    QwtPainter::drawLine( painter, pRect.left(),
                        pos.y(), pRect.right(), pos.y() );
                }
                break;
            }
            break;
        }
        default:
            break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
