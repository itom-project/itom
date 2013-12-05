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

#include "multiPointPickerMachine.h"

#include <qevent.h>
#include <qwt_event_pattern.h>

//----------------------------------------------------------------------------------------------------------------------------------
MultiPointPickerMachine::MultiPointPickerMachine() :
    QwtPickerPolygonMachine(), m_maxNrItems(0), m_currentNrItems(0)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void MultiPointPickerMachine::setMaxNrItems(int value)
{
    m_maxNrItems = value;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Transition
QList<QwtPickerMachine::Command> MultiPointPickerMachine::transition(
    const QwtEventPattern &eventPattern, const QEvent *event )
{
    QList<QwtPickerMachine::Command> cmdList;

    switch ( event->type() )
    {
        case QEvent::MouseButtonPress:
        {
            if ( eventPattern.mouseMatch(
                QwtEventPattern::MouseSelect1, static_cast<const QMouseEvent *>( event ) ) )
            {
                if ( state() == 0 )
                {
                    cmdList += Begin;
                    cmdList += Append;
                    m_currentNrItems = 1;
                    setState( 1 );
                }
                else
                {
                    cmdList += Append;
                    m_currentNrItems++;
                }

                if (m_currentNrItems > m_maxNrItems && m_maxNrItems >= 0) // "> m_maxNrItems" since the last item is always the currently "moved" point (not clicked yet)
                {
                    cmdList += End;
                    setState( 0 );
                }
            }
            if ( eventPattern.mouseMatch(
                QwtEventPattern::MouseSelect2, static_cast<const QMouseEvent *>( event ) ) )
            {
                if ( state() == 1 )
                {
                    cmdList += End;
                    setState( 0 );
                }
            }
            break;
        }
        case QEvent::MouseMove:
        case QEvent::Wheel:
        {
            if ( state() != 0 )
            {
                cmdList += Move;
            }
            break;
        }
        case QEvent::KeyPress:
        {
            if ( eventPattern.keyMatch(
                QwtEventPattern::KeySelect1, static_cast<const QKeyEvent *> ( event ) ) )
            {
                if ( state() == 0 )
                {
                    cmdList += Begin;
                    cmdList += Append;
                    m_currentNrItems = 1;
                    setState( 1 );
                }
                else
                {
                    cmdList += Append;
                    m_currentNrItems++;
                }

                if (m_currentNrItems > m_maxNrItems && m_maxNrItems >= 0) // "> m_maxNrItems" since the last item is always the currently "moved" point (not clicked yet)
                {
                    cmdList += End;
                    setState( 0 );
                }
            }
            else if ( eventPattern.keyMatch(
                QwtEventPattern::KeySelect2, static_cast<const QKeyEvent *> ( event ) ) )
            {
                if ( state() == 1 )
                {
                    cmdList += End;
                    setState( 0 );
                }
            }
            else if ( (static_cast<const QKeyEvent *>( event ))->key() == Qt::Key_M)
            {
                if ( state() == 0 )
                {
                    cmdList += Begin;
                    m_currentNrItems = 0;
                    setState( 1 );
                }
            }
            break;
        }
        default:
            break;
    }

    return cmdList;
}

//----------------------------------------------------------------------------------------------------------------------------------
