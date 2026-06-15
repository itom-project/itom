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

#pragma once

#include <qobject.h>
#include <qevent.h>

#include "../global.h"


namespace ito
{

class PythonEngine;

//-------------------------------------------------------------------------------------
/*!
    \class PythonStatePublisher
    \brief One instance of this class is created by MainApplication::setupApplication
         and runs in the main thread of itom. Its onPythonStateChanged slot is connected
         to the pythonStateChanged signal of PythonEngine. Other widgets should rather
         connect to pythonStateChanged of this class than to the direct signal of
         PythonEngine. This is mainly for one reason: Whenever a short script
         is run in Python (not debug), it might be that the execution is so short,
         that it is a waste of computing resources to switch the GUI to a busy state
         during this short exeuction. Therefore, the transition to the `run` state
         is signalled by this class with a short delay. Whenever, the 'idle' state
         is signalled by the PythonEngine before the delay exceeds, nothing is
         signalled by this class. This only holds for this transition. All other
         python state transitions are immediately reported to all connected
         classes.

    If the PythonEngine emits the signal pythonStateChanged with immediate = true,
    this transition is directly reported (e.g. necessary for commands, executed
    in the command line).
*/
class PythonStatePublisher : public QObject
{
    Q_OBJECT
public:

    explicit PythonStatePublisher(const PythonEngine *engine);
    ~PythonStatePublisher();

protected:
    void timerEvent(QTimerEvent *event);

private:
    struct DelayedTransition
    {
        DelayedTransition() : timerId(-1) {}

        int timerId; //!< -1 if no timer is currently set, else the timer id
        tPythonTransitions transition; //!< the scheduled transition
    };

    DelayedTransition m_delayedTrans;
    int m_delayMs; //!< delay time of non-immediate state changes in ms

private Q_SLOTS:
    void onPythonStateChanged(tPythonTransitions pyTransition, bool immediate);
    void propertiesChanged();

Q_SIGNALS:
    void pythonStateChanged(tPythonTransitions pyTransition);

};

} //end namespace ito
