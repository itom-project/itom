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

#ifndef SHAREDSTRUCTURES_QT_H
#define SHAREDSTRUCTURES_QT_H

#include "commonGlobal.h"
#include <qmutex.h>
#include <qsemaphore.h>
#include <qdebug.h>
#include <qeventloop.h>
#include <qmap.h>
#include <qstring.h>

#include "sharedStructures.h"

//definition of some tags for signals and slots
#ifndef Q_MOC_RUN
    //the following tags can be placed before the definition of slots and signals, they are then obtainable via the tag() method of QMetaMethod.

    //place ITOM_PYNOTACCESSIBLE before the definition of a method to not show this method in auto-parsed listings that are dedicated to python usage.
    #define ITOM_PYNOTACCESSIBLE
#endif

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

typedef QMap<QString, ito::Param> ParamMap;
typedef ParamMap::iterator ParamMapIterator;

//namespace ito
//{

class ITOMCOMMONQT_EXPORT ItomSharedSemaphore
{
    private:
        QSemaphore *m_pSemaphore;        /*!< underlying instance of QSemaphore. This semaphore is created and destructed in the constructor and destructor respectively. */
        int m_instCounter;               /*!<  counts how many instances are remaining, which are still participating at this wait condition */
        bool m_enableDelete;             /*!<  helper member variable avoiding that this instance is deleted directly by "delete"-keyword */
        int m_numOfListeners;            /*!< number of called methods (listeners) in different threads. Every listener must release this semaphore before it is finally unlocked. */
        bool m_callerStillWaiting;       /*!< true if caller is still waiting, false if caller finished or run into a timeout */
        QMutex internalMutex;            /*!<  static mutex internally used for protecting some commands */

    public:
        //! constructor
        /*!
            A new ItomSharedSemaphore is created and the underlying semaphore is already locked with a certain number.
            This number depends on the number of listeners (usually: 1). A listener is a method which is called in another
            thread. The caller creates the ItomSharedSemaphore in order to wait until the listener decreases the lock level
            of the semaphore by one or a certain timout time has been reached.

            \param [in] numberOfListeners (default: 1) are the number of different methods in other threads which should release this semaphore before the caller can go on.
        */
        inline ItomSharedSemaphore(int numberOfListeners = 1) :
                m_numOfListeners(numberOfListeners),
                m_instCounter(numberOfListeners + 1),
                m_enableDelete(false),
                m_callerStillWaiting(true),
                returnValue(ito::retOk)
        {
            QMutexLocker mutexLocker(&internalMutex);
            m_pSemaphore = new QSemaphore(m_numOfListeners);
            m_pSemaphore->acquire(m_numOfListeners);
        }

        //! destructor (do not call directly, instead free the semaphore by ItomSharedSemaphore::deleteSemaphore
        /*
        \sa ItomSharedSemaphore::deleteSemaphore
        */
        inline ~ItomSharedSemaphore()
        {
            Q_ASSERT_X(m_enableDelete, "~ItomSharedSemaphore", "it is not allowed to directly destroy ItomSharedSemaphore. Always use ItomSharedSemaphore::deleteSemaphore(...)");

            if(m_pSemaphore->available() < m_numOfListeners)
            {
                m_pSemaphore->release( m_numOfListeners - m_pSemaphore->available());
                qDebug("ItomSharedSemaphore is not fully available at moment of destruction");
            }
            if (m_pSemaphore)
                delete m_pSemaphore;
            m_pSemaphore = NULL;
        }

        //! The call of this method returns if a certain timeout has been expired or every listener released the semaphore
        /*
            The timeout time is indicated in milliseconds, a value equal to -1 means that no timeout is set and this methods
            only returns if the called method released the semaphore. In the constructor of ItomSharedSemaphore, the semaphore
            has been locked with a number equal to the number of listeners, which usually is 1. The semaphore is unlocked, if
            every listener releases the semaphore once (\sa release).

            A possible timeout is indicated by a debug-warning in the output window of the IDE.

            @param [in] timeout in ms [-1 : no timeout]
            @return true if caller (listener) released the lock within the given timeout time, false if timeout expired
            @sa release, waitAndProcessEvents
        */
        inline bool wait(int timeout)
        {
            bool success;

            if (timeout >= 0)
            {
                success = m_pSemaphore->tryAcquire(m_numOfListeners, timeout);
            }
            else
            {
                //due to a bug in linux (at least CentOS 7), tryAcquire causes a crash for infinite timeouts.
                //therefore, we use the specific case here, to handle these cases.
                m_pSemaphore->acquire(m_numOfListeners);
                success = true;
            }

            QMutexLocker mutexLocker(&internalMutex);
            if(success == false)
            {
                qDebug() << "ItomSharedSemaphore run into a timeout. Number of attempted listeners: " << m_numOfListeners << ", already freed: " << m_pSemaphore->available();
            }
            else
            {
                m_pSemaphore->release(m_numOfListeners);
            }

            m_callerStillWaiting = false;

            return success;
        }

        //! The call of this method returns if a certain timeout has been expired or every listener released the semaphore
        /*
        The timeout time is indicated in milliseconds, a value equal to -1 means that no timeout is set and this methods
        only returns if the called method released the semaphore. In the constructor of ItomSharedSemaphore, the semaphore
        has been locked with a number equal to the number of listeners, which usually is 1. The semaphore is unlocked, if
        every listener releases the semaphore once (\sa release).

        The only difference between this method and wait is, that this implementation continuously allows
        processing events in the event loop of the waiting thread. The type of allowed events is set by flags.

        A possible timeout is indicated by a debug-warning in the output window of the IDE.

        @param [in] timeout in ms [-1 : no timeout]
        @param [in] flags is the type of allowed events that can be processed in the calling thread during the wait.
        @return true if caller (listener) released the lock within the given timeout time, false if timeout expired
        @sa release, wait
        */
        bool waitAndProcessEvents(int timeout, QEventLoop::ProcessEventsFlags flags = QEventLoop::AllEvents);

        //! decreases the number of locks by one
        /*
            the called method in another thread must release the semaphore as soon as possible such that the caller can continue
        */
        inline void release() { /*qDebug() << "semaphore release";*/ m_pSemaphore->release(1); }

        //! checks whether the semaphore is still locked or not
        /*
            @return true if semaphore is still locked, else false
        */
        inline int available() const { return m_pSemaphore->available(); }

        //! indicates whether caller-method is still waiting that the lock is released by the listener(s).
        /*
            this method is not 100% thread-safe, that means it might occure, that the wait-method of the caller drops into the timeout
            during the call to the method isCallerStillWaiting. Therefore consider this method as pure information.

            @return true if caller-method is still waiting (in \sa wait-method) that all listeners are calling the release method
            in order to free the lock of the semaphore. else: false
        */
        inline bool isCallerStillWaiting() { QMutexLocker mutexLocker(&internalMutex); return m_callerStillWaiting; }

        //! static method to decrease the reference counter of any ItomSharedSemaphore or delete it if the reference counter drops to zero
        //
        //  Every listener and the caller-method must call this static method with the pointer to the corresponding ItomSharedSemaphore
        //  in order to guarantee the final deletion of the semaphore, if it is not needed any more by any participating method (caller or listener).
        //
        //  In every listener method (called method) you should call this method after you released the semaphore (\sa release).
        // The ItomSharedSemaphore consists of a internal reference counter, which is set to the number of listeners plus 1 at
        //  construction time. Every call to deleteSemaphore decreases this counter and if this counter drops to zero, no method
        //  uses the semaphore any more such that it is safely deleted. Be careful that you don't access the semaphore in a method
        //  where you already called deleteSemaphore with the semaphore-pointer as parameter.
        //
        //  In order to simplify the handling of ItomSharedSemaphore consider to used ItomSharedSemaphoreLocker.

        inline void deleteSemaphore(void)
        {
            QMutexLocker mutexLocker(&internalMutex);

            m_instCounter --;
            if(m_instCounter <= 0)
            {
                m_enableDelete = true;
                mutexLocker.unlock();
                delete this;
                return;
            }

            return;
        }

        ito::RetVal returnValue; /*!< public returnValue member variable of ItomSharedSemaphore. This return value can be used to return the result of any called method (listener) to the caller. Please access this value in caller only if the wait-method returned with true. Write to returnValue in called method (listener) before releasing the semaphore. This is important, since the returnValue is not fully thread safe.*/
};


/*!
*    \class ItomSharedSemaphoreLocker
*    \brief Locker-class for ItomSharedSemaphore. The functionality is equal to QMutexLocker in Qt. ItomSharedSemaphoreLocker is a guard for
        any ItomSharedSemaphore-pointer. If the variable of type ItomSharedSemaphoreLocker is deleted, e.g. if its containing method runs out of scope,
        the destructor calls ItomSharedSemaphore::deleteSemaphore(...). This is also the case if you assign another pointer to ItomSharedSemaphore to this
        locker-instance.
*/
class ItomSharedSemaphoreLocker
{
    public:
        //! constructor with ItomSharedSemaphore-pointer as parameter. This semaphore will be guarded by this locker.
        inline ItomSharedSemaphoreLocker(ItomSharedSemaphore* semaphore) : m_semaphore(semaphore) {}

        //! empty constructor. The locker will not guard any semaphore yet.
        inline ItomSharedSemaphoreLocker() : m_semaphore(NULL) {}

        //! destructor.
        inline ~ItomSharedSemaphoreLocker()
        {
            if(m_semaphore)
            {
                m_semaphore->deleteSemaphore();
                m_semaphore = NULL;
            }
        }

        //! returns the pointer to the guarded ItomSharedSemaphore.
        inline ItomSharedSemaphore* getSemaphore() const { return m_semaphore; }

        //! returns the pointer to the guarded ItomSharedSemaphore.
        ItomSharedSemaphore* operator ->() const { return m_semaphore; }

        //! assigns another ItomSharedSemaphore to this locker.
        /*
            If this locker already guards an instance of ItomSharedSemaphore its reference counter is decremented first. \sa ItomSharedSemaphore::deleteSemaphore.
        */
        inline ItomSharedSemaphoreLocker & operator = (ItomSharedSemaphore *newSemaphoreInstance)
        {
            if(m_semaphore)
            {
                m_semaphore->deleteSemaphore();
                m_semaphore = NULL;
            }
            m_semaphore = newSemaphoreInstance;
            return *this;
        }

    private:
        inline ItomSharedSemaphoreLocker(ItomSharedSemaphoreLocker & /*other*/ ) { /* forbidden */ }
        inline ItomSharedSemaphoreLocker & operator = (const ItomSharedSemaphoreLocker & /*other*/ ) { return *this; /* forbidden */ }
        ItomSharedSemaphore* m_semaphore;  /*!< pointer to ItomSharedSemaphore */
};

//} //end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
