/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#include "sharedStructuresQt.h"

#include <qelapsedtimer.h>
#include <qcoreapplication.h>

//namespace ito {

/*!
*    \class ItomSharedSemaphore
*    \brief semaphore which can be used for asychronous thread communication. By using this class it is possible to realize timeouts.

        This semaphore is usually applied if any method invokes another method in another thread and should wait for the called
        method being terminated or the waiting routine drops into a possible timeout. Therefore, the calling method
        must create an instance of ItomSharedSemaphore with a number of listeners equal to one. Then, the pointer to ItomSharedSemaphore
        is transmitted to the called method (usually as last argument). If the called method is done or wants the caller to continue
        the release-method of ItomSharedSemaphore is called. The calling method calls the wait-method of the semaphore which blocks
        the method until the semaphore is released. Finally, both the caller and the calling method must call ItomSharedSemaphore::deleteSemaphore
        in order to decrease the reference counter from two to zero, which allows the system to delete the semaphore.

        Consider to guard an instance of ItomSharedSemaphore by the capsule-class ItomSharedSemaphoreLocker both in the caller and calling method,
        such that the decrease of the reference counter is executed if the ItomSharedSemaphoreLocker-variables are deleted, e.g. if they run out
        of scope, which is even after the return-command of any method.
*/

//! mutex initialization
//QMutex ItomSharedSemaphore::internalMutex;



//! The call of this method returns if a certain timeout has been expired or every listener released the semaphore.
/*
    This method does the same than \sa wait, but while waiting events within the event queue of the calling thread will continuously be checked and processed.

    @param [in] timeout in ms [-1 : no timeout]
    @param [in] flags is the flag-parameter of method processEvents
    @return true if caller (listener) released the lock within the given timeout time, false if timeout expired
    @sa release, wait
*/
bool ItomSharedSemaphore::waitAndProcessEvents(int timeout, QEventLoop::ProcessEventsFlags flags /* = QEventLoop::AllEvents*/)
{
    bool available = false;
    QElapsedTimer timer;
    timer.start();

    while(!available && (timer.elapsed() <= timeout || timeout == -1))
    {
        available = m_pSemaphore->tryAcquire(m_numOfListeners);

        QCoreApplication::processEvents(flags);
    }

    QMutexLocker mutexLocker(&internalMutex);
    if(available == false)
    {
        qDebug() << "ItomSharedSemaphore run into a timeout. Number of attempted listeners: " << m_numOfListeners << ", already freed: " << m_pSemaphore->available();
    }
    else
    {
        m_pSemaphore->release(m_numOfListeners);
    }

    m_callerStillWaiting = false;

    return available;
}

//} //end namespace ito
