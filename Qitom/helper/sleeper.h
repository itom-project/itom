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

#ifndef SLEEPER_H
#define SLEEPER_H

#include <qthread.h>

namespace ito
{
    //for Qt5 only, QThread::usleep... are all public, then this helper class is not needed any more.
    class Sleeper : public QThread
    {
    public:
        static void usleep(unsigned long usecs){QThread::usleep(usecs);}
        static void msleep(unsigned long msecs){QThread::msleep(msecs);}
        static void sleep(unsigned long secs){QThread::sleep(secs);}
    };
}

#endif //SLEEPER_H
