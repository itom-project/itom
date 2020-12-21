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
        int timerId;
        tPythonTransitions transition;
    };

    DelayedTransition m_delayedTrans;

private Q_SLOTS:
    void onPythonStateChanged(tPythonTransitions pyTransition);

Q_SIGNALS:
    void pythonStateChanged(tPythonTransitions pyTransition);

};

} //end namespace ito

