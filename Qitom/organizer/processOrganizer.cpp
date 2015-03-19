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
#include <qfileinfo.h>
#include <qdir.h>
#include <qcoreapplication.h>
#include <qlibraryinfo.h>

#if WIN32
    #include <Windows.h>
#endif

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
ProcessOrganizer::ProcessOrganizer()
{
    m_processes.clear();    
}

//----------------------------------------------------------------------------------------------------------------------------------
ProcessOrganizer::~ProcessOrganizer()
{
    QMultiHash<QString, QPair<QProcess*, bool> >::iterator it = m_processes.begin();
    //0. delete all connections for processes
    //it = m_processes.begin();

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

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param binaryName
    \return QString
*/
/*static*/ QString ProcessOrganizer::getAbsQtToolPath(const QString &binaryName)
{
#ifdef __APPLE__
    for( int i = 0; i < 4; ++i)
    {
        QDir dir;
        QString binaryName2 = binaryName;
        
        // second try: append .app
        if( i == 1)
        {
            if(!binaryName2.endsWith(".app"))
            {
                binaryName2.append(".app");
            }
        }
        // third try: first letter upper case
        else if( i == 2)
        {
            binaryName2[0] = binaryName2[0].toUpper();
        }
        // fourth try: first letter upper case and .app
        else if( i == 2)
        {
            binaryName2[0] = binaryName2[0].toUpper();
            if(!binaryName2.endsWith(".app"))
            {
                binaryName2.append(".app");
            }
        }
        
        //1. first try: in this application dir
        dir.setPath( QCoreApplication::applicationDirPath() );
        QStringList entryList = dir.entryList(QDir::Executable | QDir::Files | QDir::Dirs);
        if(entryList.contains(binaryName2)) //dir.exists(binaryName2))
        {
            return dir.absoluteFilePath( binaryName2 );
        }
        
        //2. next try: qt binary dir (when installing qt from sources)
        dir.setPath( QLibraryInfo::location( QLibraryInfo::BinariesPath ) );
        entryList = dir.entryList(QDir::Executable | QDir::Files | QDir::Dirs);
        if(entryList.contains(binaryName2))
        {
            return dir.absoluteFilePath( binaryName2 );
        }
       
        //3. next try: qt parent dir (when installing qt from sources)
        dir.setPath( QLibraryInfo::location( QLibraryInfo::PrefixPath ) );
        entryList = dir.entryList(QDir::Executable | QDir::Files | QDir::Dirs);
        if(entryList.contains(binaryName2))
        {
            return dir.absoluteFilePath( binaryName2 );
        }
        
        //4. next try: system applications directory
        dir.setPath( "/Applications");
        entryList = dir.entryList(QDir::Executable | QDir::Files | QDir::Dirs);
        if(entryList.contains(binaryName2))
        {
            return dir.absoluteFilePath( binaryName2 );
        }
        
        //5. next try: user applications directory
        dir.setPath( QDir::homePath() + "/Applications");
        entryList = dir.entryList(QDir::Executable | QDir::Files | QDir::Dirs);
        if(entryList.contains(binaryName2))
        {
            return dir.absoluteFilePath( binaryName2 );
        }
    }
    
    //6. return as is
    return binaryName;
#elif (!defined WIN32)
    QDir dir;
    QString binaryName2 = binaryName;

    //1. first try: in this application dir
    dir.setPath( QCoreApplication::applicationDirPath() );
    QStringList entryList = dir.entryList(QDir::Executable | QDir::Files);
    //qDebug() << dir << entryList << dir.entryList(QDir::Files);
    if(entryList.contains(binaryName2)) //dir.exists(binaryName2))
    {
        return dir.absoluteFilePath( binaryName2 );
    }

    //2. next try: qt binary dir (when installing qt from sources)
    dir.setPath( QLibraryInfo::location( QLibraryInfo::BinariesPath ) );
    entryList = dir.entryList(QDir::Executable | QDir::Files);
    //qDebug() << dir << entryList << dir.entryList(QDir::Files);
    if(entryList.contains(binaryName2))
    {
        return dir.absoluteFilePath( binaryName2 );
    }
    
    //3. return as is
    return binaryName;
#else
    QDir dir;
    QString binaryName2 = binaryName;
    if(!binaryName2.endsWith(".exe"))
    {
        binaryName2.append(".exe");
    }

    //1. first try: in this application dir
    dir.setPath( QCoreApplication::applicationDirPath() );
    if(dir.exists(binaryName2))
    {
        return dir.absoluteFilePath( binaryName2 );
    }

    //2. next try: qt binary dir (when installing qt from sources)
    dir.setPath( QLibraryInfo::location( QLibraryInfo::BinariesPath ) );
    if(dir.exists(binaryName2))
    {
        return dir.absoluteFilePath( binaryName2 );
    }

    //3. QTDIR
    QByteArray qtdirenv = qgetenv( "QTDIR" );
    if(qtdirenv.size() > 0)
    {
        dir.setPath( qtdirenv );
        if(dir.exists(binaryName2))
        {
            return dir.absoluteFilePath( binaryName2 );
        }
    }

    //4. return as is
    return binaryName;
#endif
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param forceToCloseAll
    \return RetVal
*/
RetVal ProcessOrganizer::collectGarbage(bool forceToCloseAll /*= false*/)
{
    QMultiHash<QString, QPair<QProcess*, bool> >::iterator it = m_processes.begin();
    //it = m_processes.begin();

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

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param name
    \return QProcess
*/
QProcess* ProcessOrganizer::getFirstExistingProcess(const QString &name)
{
    QMultiHash<QString, QPair<QProcess*, bool> >::const_iterator i = m_processes.constFind(name);
    if(i != m_processes.constEnd())
    {
        return (*i).first;
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param name
    \param tryToUseExistingProcess
    \param existingProcess
    \param closeOnFinalize
    \return QProcess
*/
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

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param exitCode
    \param exitStatus
*/
void ProcessOrganizer::processFinished ( int /*exitCode*/, QProcess::ExitStatus /*exitStatus*/ )
{
    collectGarbage();
}


//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param error
*/
void ProcessOrganizer::processError (QProcess::ProcessError /*error*/ )
{
    collectGarbage();
}


//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

*/
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

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param windowName
    \return bool
*/
bool ProcessOrganizer::bringWindowsOnTop(const QString &windowName)
{
#ifdef WIN32

#if UNICODE
    wchar_t *nameArray = new wchar_t[ windowName.size() + 2];
    int size = windowName.toWCharArray(nameArray);
    nameArray[size] = '\0';
    HWND THandle = FindWindow(NULL, nameArray );
    delete[] nameArray;
#else
    HWND THandle = FindWindow(NULL, windowName.toLatin1().data() );
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
