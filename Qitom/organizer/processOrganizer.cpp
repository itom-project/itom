/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "processOrganizer.h"

#include <qdebug.h>

#ifndef linux
#include <Windows.h>
#endif

namespace ito
{

ProcessOrganizer::ProcessOrganizer()
{
    m_processes.clear();    
}

ProcessOrganizer::~ProcessOrganizer()
{
    QMultiHash<QString, QPair<QProcess*, bool> >::iterator it = m_processes.begin();
    //0. delete all connections for processes
    it = m_processes.begin();

    while(it != m_processes.end())
    {
        if((*it).first != NULL)
        {
            disconnect((*it).first, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(processFinished(int,QProcess::ExitStatus)));
            disconnect((*it).first, SIGNAL(error(QProcess::ProcessError)), this, SLOT(processError(QProcess::ProcessError)));
        }
        ++it;
    }

    //1. first call garbage collector, to clear every killed process
    collectGarbage();

    //2. close every existing process, if closeOnShutdown-flag is set
    it = m_processes.begin();

    while(it != m_processes.end())
    {
        if( (*it).first->state() == QProcess::Running && (*it).second == true )
        {
            (*it).first->close();
        }
        ++it;
    }

    //3. wait for every existing process to be closed
    it = m_processes.begin();

    while(it != m_processes.end())
    {
        if( (*it).first->state() == QProcess::Running && (*it).second == true)
        {
            (*it).first->waitForFinished(30000);
        }
        else if( (*it).first->state() == QProcess::Starting && (*it).second == true)
        {
            (*it).first->waitForStarted(30000);
            (*it).first->close();
            (*it).first->waitForFinished(30000);
        }
        ++it;
    }

    //4. delete every existing process
    it = m_processes.begin();

    while(it != m_processes.end())
    {
        (*it).first->deleteLater();
        it = m_processes.erase(it);
    }

    m_processes.clear();
}

RetVal ProcessOrganizer::collectGarbage(bool forceToCloseAll /*= false*/)
{
    QMultiHash<QString, QPair<QProcess*, bool> >::iterator it = m_processes.begin();
    it = m_processes.begin();

    while(it != m_processes.end())
    {
        if( (*it).first->state() == QProcess::NotRunning || forceToCloseAll )
        {
            (*it).first->deleteLater();
            m_processStdOut.remove( it.key() ); //removes corresponding output messages
            it = m_processes.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return retOk;
}

QProcess* ProcessOrganizer::getFirstExistingProcess(const QString &name)
{
    QMultiHash<QString, QPair<QProcess*, bool> >::const_iterator i = m_processes.constFind(name);
    if(i != m_processes.constEnd())
    {
        return (*i).first;
    }
    return NULL;
}

QProcess* ProcessOrganizer::getProcess(const QString &name, bool tryToUseExistingProcess, bool &existingProcess, bool closeOnFinalize)
{
    QProcess *process = NULL;
    existingProcess = false;

    if(tryToUseExistingProcess)
    {
        process = getFirstExistingProcess(name);
        existingProcess = true;
    }

    if(process)
    {
        return process;
    }

    process = new QProcess();
    connect(process, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(processFinished(int,QProcess::ExitStatus)));
    connect(process, SIGNAL(error(QProcess::ProcessError)), this, SLOT(processError(QProcess::ProcessError)));
    connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(readyReadStandardOutput()));

    m_processes.insertMulti(name, QPair<QProcess*, bool>(process, closeOnFinalize) );

    return process;

}

void ProcessOrganizer::processFinished ( int /*exitCode*/, QProcess::ExitStatus /*exitStatus*/ )
{
    collectGarbage();
}

void ProcessOrganizer::processError (QProcess::ProcessError /*error*/ )
{
    collectGarbage();
}

void ProcessOrganizer::readyReadStandardOutput()
{
    //the output message is mainly useful in order to get the socket-number of the designer (started as server)
    //the designer returns this number over stdOut. When opening another ui-file, this ui-file is send over a socket
    //connection
    QProcess *sendingProcess = qobject_cast<QProcess*>(sender());
    if(sendingProcess)
    {
        QByteArray ba = sendingProcess->readAllStandardOutput();

        QHashIterator< QString, QPair<QProcess*, bool> > i(m_processes);
        while (i.hasNext()) 
        {
            i.next();
            if(i.value().first == sendingProcess)
            {
                QByteArray ba2 = m_processStdOut[ i.key() ];
                ba2.append(ba);
                m_processStdOut[ i.key() ] = ba2;
                break;
            }
        }

        
    }
}

bool ProcessOrganizer::bringWindowsOnTop(const QString &windowName)
{
#ifndef linux

#if UNICODE
    wchar_t *nameArray = new wchar_t[ windowName.size() + 2];
    int size = windowName.toWCharArray(nameArray);
    nameArray[size] = '\0';
    HWND THandle = FindWindow(NULL, nameArray );
    delete[] nameArray;
#else
    HWND THandle = FindWindow(NULL, windowName.toAscii().data() );
#endif
    

    if(THandle)
    {
        long result = GetWindowLong(THandle, GWL_STYLE);
        if(result & WS_MINIMIZE)
        {
            ShowWindow(THandle, 1); //SW_SHOW = 5, SW_NORMAL = 1
        }

        return SetForegroundWindow(THandle);

        //long d = WS_MINIMIZE;
        //long d2 = WS_VISIBLE;
        //long d3 = WS_MAXIMIZE;
        //long d4 = WS_ICONIC;
        //bool r = SetForegroundWindow(THandle);
        //qDebug() << r;
        //return r;
        //
        ////bringWindowToTop only works, if window is not minimized
        //ShowWindow(THandle, 1); //SW_SHOW = 5, SW_NORMAL = 1
        //return BringWindowToTop(THandle);
    }

#endif

    return false;

}

} //namespace ito