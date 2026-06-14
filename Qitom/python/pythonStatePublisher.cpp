/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "pythonStatePublisher.h"
#include "pythonEngine.h"
#include "../AppManagement.h"
#include "../mainApplication.h"

namespace ito
{

//-------------------------------------------------------------------------------------
PythonStatePublisher::PythonStatePublisher(const PythonEngine *engine) :
    m_delayMs(100)
{
    propertiesChanged(); // read real timeout time

    connect(
        engine,
        &PythonEngine::pythonStateChanged,
        this,
        &PythonStatePublisher::onPythonStateChanged);

    connect(
        qobject_cast<MainApplication*>(AppManagement::getMainApplication()),
        &MainApplication::propertiesChanged,
        this,
        &PythonStatePublisher::propertiesChanged);
}

//-------------------------------------------------------------------------------------
PythonStatePublisher::~PythonStatePublisher()
{
}

//-------------------------------------------------------------------------------------
void PythonStatePublisher::onPythonStateChanged(tPythonTransitions pyTransition, bool immediate)
{
    switch (pyTransition)
    {
    case pyTransBeginDebug:
    case pyTransEndDebug:
    case pyTransDebugWaiting:
    case pyTransDebugContinue:
    case pyTransDebugExecCmdBegin:
    case pyTransDebugExecCmdEnd:
        if (m_delayedTrans.timerId >= 0)
        {
            killTimer(m_delayedTrans.timerId);
            m_delayedTrans.timerId = -1;
        }

        emit pythonStateChanged(pyTransition);
        break;
    case pyTransEndRun:
        if (m_delayedTrans.timerId >= 0)
        {
            killTimer(m_delayedTrans.timerId);
            m_delayedTrans.timerId = -1;

            if (m_delayedTrans.transition == pyTransBeginRun)
            {
                // do not emit a state change, since the
                // initializing counterpart has not been
                // emitted yet.
                return;
            }
        }

        emit pythonStateChanged(pyTransition);
        break;

    case pyTransBeginRun:
        // publish this transition with a certain delay
        // if the transition is "reverted" earlier, nothing is emitted.
        if (m_delayedTrans.timerId >= 0)
        {
            killTimer(m_delayedTrans.timerId);
        }

        if (immediate || m_delayMs <= 0)
        {
            emit pythonStateChanged(pyTransition);
        }
        else
        {
            m_delayedTrans.timerId = startTimer(m_delayMs);
            m_delayedTrans.transition = pyTransition;
        }

        break;
    }
}

//-------------------------------------------------------------------------------------
void PythonStatePublisher::timerEvent(QTimerEvent *event)
{
    if (m_delayedTrans.timerId >= 0)
    {
        emit pythonStateChanged(m_delayedTrans.transition);
        killTimer(m_delayedTrans.timerId);
        m_delayedTrans.timerId = -1;
    }
}

//-------------------------------------------------------------------------------------
void PythonStatePublisher::propertiesChanged()
{
    // can be implemented if the delay should be configured by settings!
}

} //end namespace ito
