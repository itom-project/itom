/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#ifndef READWRITELOCK
#define READWRITELOCK

#include "defines.h"

// Use Window or Posix
//#if defined (_WINDOWS) || defined (WIN32)
#ifdef WIN32
     #include <windows.h>
#else
     #ifndef POSIX
//          #warning POSIX will be used (but you did not define it)
     #endif
     #include <pthread.h>
#endif

/*!
    \class ReadWriteLock
    \brief objects of this class organize a read-write-lock mechanism. This class is mainly used by dataObject.
        If this object is locked for writing (lockWrite), it can not be used for reading nor writing until the writer released the protection
        with (unlock). On the other hand, the object can multiply be locked for reading if no write-operation is executed at the same time.
*/
namespace ito {

#ifdef WIN32
//#if defined _WINDOWS || defined WIN32

    class ReadWriteLock
    {
    private:

        ReadWriteLock(const ReadWriteLock &) {}

        int nrOfReaders;                //! nrOfReaders (>= 0)
        int status;                     //! state (-1 if object is in write mode, 0 if object is idle, else number of readers)
        bool writeGelocked;
        CRITICAL_SECTION mAtomicOp;     //! critical section for realizing atomic operation blocks
        CRITICAL_SECTION mReadLock;     //! critical section for realizing the read lock
        CRITICAL_SECTION mWriteLock;    //! critical section for realizing the write lock

    public:
        //! constructor
        /*!
            \param int lockStatus = 0, should normally be used with the default parameter, else: lockStatus: -1 writeLock, 0 no lock, >0 read lock with given number of readers
        */
        ReadWriteLock(int lockStatus = 0) : nrOfReaders(0), status(0), writeGelocked(false)
        {
            InitializeCriticalSection(&mAtomicOp);
            InitializeCriticalSection(&mReadLock);
            InitializeCriticalSection(&mWriteLock);

            if(lockStatus == -1)
            {
                lockWrite();
                status = -1;
            }
            else if(lockStatus > 0)
            {
                lockRead(lockStatus);
                status = lockStatus;
            }
        };

        //! deconstructor
        /*!
        */
        ~ReadWriteLock()
        {
            TryEnterCriticalSection(&mAtomicOp);
            LeaveCriticalSection(&mAtomicOp);
            DeleteCriticalSection(&mAtomicOp);
            TryEnterCriticalSection(&mReadLock);
            LeaveCriticalSection(&mReadLock);
            DeleteCriticalSection(&mReadLock);
            TryEnterCriticalSection(&mWriteLock);
            LeaveCriticalSection(&mWriteLock);
            DeleteCriticalSection(&mWriteLock);
        };

        //! locks for reading
        /*!
            locks this object for reading. This method waits until the object is not protected by any write operation and then locks it for reading.
            If other participants are already reading this object, the number of readers variable is just incremented

            \param int increment, number of new readers
        */
        void lockRead(int increment = 1)
        {
            if (status == -1)
            {
                std::cout << "info: try to lock for reading, but currently locked for writing\n" << std::endl;
            }

            //if(!TryEnterCriticalSection(&mWriteLock))
            //{
            //    int i=1;
            //    EnterCriticalSection(&mWriteLock);
            //    std::cout << "try failed2" << std::endl;
            //}
            EnterCriticalSection(&mWriteLock);

            writeGelocked = true;
            EnterCriticalSection(&mAtomicOp);
            TryEnterCriticalSection(&mReadLock);
            nrOfReaders+= increment;
            status = nrOfReaders;
            LeaveCriticalSection(&mAtomicOp);
            LeaveCriticalSection(&mWriteLock);
            writeGelocked = false;
        }

        //! locks for writing
        /*!
            locks this object for writing. This method waits until this object is in idle mode and then locks it for one single writing operation.

        */
        void lockWrite()
        {
            if (status == -1)
            {
                std::cout << "info: try to lock for writing, but currently locked for writing\n" << std::endl;
            }

            if (status > 0)
            {
                std::cout << "info: try to lock for writing, but currently locked for reading\n" << std::endl;
            }
            //if(!TryEnterCriticalSection(&mWriteLock))
            //{
            //    int i=1;
            //    EnterCriticalSection(&mWriteLock);
            //    std::cout << "try failed" << std::endl;
            //}
            EnterCriticalSection(&mWriteLock);

            writeGelocked = true;
            EnterCriticalSection(&mReadLock);
            EnterCriticalSection(&mAtomicOp);
            status = -1;
            LeaveCriticalSection(&mAtomicOp);
        }

        //! unlocks a read or write protection
        /*!
            If this object is in write-mode, this write-protection is unlocked. Else the number of readers
            is decremented. The reading-protection is finally unlocked, if no other reader is reading the object.
        */
        void unlock()
        {
            EnterCriticalSection(&mAtomicOp);
            if(nrOfReaders == 0)
            {
                while(mReadLock.RecursionCount > 0) LeaveCriticalSection(&mReadLock);
                while(mWriteLock.RecursionCount > 0) LeaveCriticalSection(&mWriteLock);
                writeGelocked = false;
                status = 0;
            }
            else
            {
                if(nrOfReaders > 0)  nrOfReaders--;
                status = nrOfReaders;
                if(nrOfReaders <= 0)
                {
                    while(mReadLock.RecursionCount > 0) LeaveCriticalSection(&mReadLock);
                }
            }
            LeaveCriticalSection(&mAtomicOp);
        }

        //! unlocks a read or write protection depending on the given integer value
        /*!
            \param int value (-1 : unlocks write protection, >0 unlocks the given number of reading protections)
        */
        void _unlock(int value)
        {
            EnterCriticalSection(&mAtomicOp);
            if(value == -1)
            {
                TryEnterCriticalSection(&mReadLock);
                while(mReadLock.RecursionCount > 0) LeaveCriticalSection(&mReadLock);
                TryEnterCriticalSection(&mWriteLock);
                while(mWriteLock.RecursionCount > 0) LeaveCriticalSection(&mWriteLock);
                writeGelocked = false;
                status = 0;
            }
            else
            {
                nrOfReaders -= value;
                if(nrOfReaders <= 0)
                {
                    nrOfReaders = 0;
                    TryEnterCriticalSection(&mReadLock);
                    while(mReadLock.RecursionCount > 0) LeaveCriticalSection(&mReadLock);
                }
                status = nrOfReaders;
            }
            LeaveCriticalSection(&mAtomicOp);
        }

        //! returns the protection state
        /*!
            \return -1 if object is in write mode, 0 if object is idle, else number of readers
        */
        inline int getLockStatus() const { return status; }
    };

#else

    class ReadWriteLock
    {
        public:
            //! constructor
            /*!

                \param int lockStatus = 0, should normally be used with the default parameter, else: lockStatus: -1 writeLock, 0 no lock, >0 read lock with given number of readers
            */
            ReadWriteLock(int lockStatus = 0) : nrOfReaders(0), status(0), writeGelocked(false)
            {
                pthread_mutexattr_t attr;
                pthread_mutexattr_init(&attr);
                pthread_mutexattr_settype(&attr,PTHREAD_MUTEX_RECURSIVE);
                pthread_mutex_init(&mAtomicOp,&attr);
                pthread_mutex_init(&mReadLock,&attr);
                pthread_mutex_init(&mWriteLock,&attr);
                pthread_mutexattr_destroy(&attr);

                if(lockStatus == -1)
                {
                    lockWrite();
                    status = -1;
                }
                else if(lockStatus > 0)
                {
                    lockRead(lockStatus);
                    status = lockStatus;
                }
            }

            //! deconstructor
            /*!
            */
            ~ReadWriteLock()
            {
                pthread_mutex_trylock(&mAtomicOp);
                pthread_mutex_unlock(&mAtomicOp);
                pthread_mutex_destroy(&mAtomicOp);
                pthread_mutex_trylock(&mReadLock);
                pthread_mutex_unlock(&mReadLock);
                pthread_mutex_destroy(&mReadLock);
                pthread_mutex_trylock(&mWriteLock);
                pthread_mutex_unlock(&mWriteLock);
                pthread_mutex_destroy(&mWriteLock);
            }

            //! locks for reading
            /*!
                locks this object for reading. This method waits until the object is not protected by any write operation and then locks it for reading.
                If other participants are already reading this object, the number of readers variable is just incremented

                \param int increment, number of new readers
            */
            void lockRead(int increment = 1)
            {
                pthread_mutex_lock(&mWriteLock);
                pthread_mutex_lock(&mAtomicOp);
                pthread_mutex_trylock(&mReadLock);
                nrOfReaders+=increment;
                status = nrOfReaders;
                pthread_mutex_unlock(&mAtomicOp);
                pthread_mutex_unlock(&mWriteLock);
            }

            //! locks for writing
            /*!
                locks this object for writing. This method waits until this object is in idle mode and then locks it for one single writing operation.

            */
            void lockWrite()
            {
                pthread_mutex_lock(&mWriteLock);
                pthread_mutex_lock(&mReadLock);
                pthread_mutex_lock(&mAtomicOp);
                status = -1;
                pthread_mutex_unlock(&mAtomicOp);
            }

            //! unlocks a read or write protection
            /*!
                If this object is in write-mode, this write-protection is unlocked. Else the number of readers
                is decremented. The reading-protection is finally unlocked, if no other reader is reading the object.
            */
            void unlock()
            {
                pthread_mutex_lock(&mAtomicOp);
                if(nrOfReaders == 0)
                {
                    pthread_mutex_unlock(&mReadLock);
                    pthread_mutex_unlock(&mWriteLock);
                    status = 0;
                }
                else
                {
                    if(nrOfReaders > 0)  nrOfReaders--;
                    status = nrOfReaders;
                    if(nrOfReaders <= 0) pthread_mutex_unlock(&mReadLock);
                }
                pthread_mutex_unlock(&mAtomicOp);
            }

            //! unlocks a read or write protection depending on the given integer value
            /*!
                \param int value (-1 : unlocks write protection, >0 unlocks the given number of reading protections)
            */
            void _unlock(int value)
            {
                pthread_mutex_lock(&mAtomicOp);
                if(value == -1)
                {
                    pthread_mutex_trylock(&mReadLock);
                    pthread_mutex_unlock(&mReadLock);
                    pthread_mutex_trylock(&mWriteLock);
                    pthread_mutex_unlock(&mWriteLock);
                    status = 0;
                }
                else
                {
                    nrOfReaders -= value;
                    if(nrOfReaders <= 0)
                    {
                        nrOfReaders = 0;
                        pthread_mutex_trylock(&mReadLock);
                        pthread_mutex_unlock(&mReadLock);
                    }
                    status = nrOfReaders;
                }
                pthread_mutex_unlock(&mAtomicOp);
            }

            //! returns the protection state
            /*!
                \return -1 if object is in write mode, 0 if object is idle, else number of readers
            */
            inline int getLockStatus() const { return status; }

        private:

            ReadWriteLock(const ReadWriteLock &) {}

            int nrOfReaders;                //! nrOfReaders (>= 0)
            int status;                     //! state (-1 if object is in write mode, 0 if object is idle, else number of readers)
            bool writeGelocked;
            pthread_mutex_t mAtomicOp;     //! critical section for realizing atomic operation blocks
            pthread_mutex_t mReadLock;     //! critical section for realizing the read lock
            pthread_mutex_t mWriteLock;    //! critical section for realizing the write lock
    };

#endif

} //namespace ito

#endif
