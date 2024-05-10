/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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

#include "processOrganizer.h"

#include <qcoreapplication.h>
#include <qdebug.h>
#include <qdir.h>
#include <qfileinfo.h>
#include <qlibraryinfo.h>
#include <qprocess.h>

#if WIN32
#include <Windows.h>
#endif

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
ProcessOrganizer::ProcessOrganizer()
{
    m_processes.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
ProcessOrganizer::~ProcessOrganizer()
{
    auto it = m_processes.constBegin();
    // 0. delete all connections for processes
    // it = m_processes.begin();

    while (it != m_processes.constEnd())
    {
        if (it->first != nullptr)
        {
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
            disconnect(
                it->first,
                SIGNAL(finished(int, QProcess::ExitStatus)),
                this,
                SLOT(processFinished(int, QProcess::ExitStatus)));
#else
            disconnect(it->first, &QProcess::finished, this, &ProcessOrganizer::processFinished);
#endif
            disconnect(
                it->first,
                &QProcess::errorOccurred,
                this,
                &ito::ProcessOrganizer::processError);
        }
        ++it;
    }

    // 1. first call garbage collector, to clear every killed process
    collectGarbage();

    // 2. close every existing process, if closeOnShutdown-flag is set
    it = m_processes.constBegin();

    while (it != m_processes.constEnd())
    {
        if (it->first->state() == QProcess::Running && it->second == true)
        {
            it->first->close();
        }

        ++it;
    }

    // 3. wait for every existing process to be closed
    it = m_processes.constBegin();

    while (it != m_processes.constEnd())
    {
        if (it->first->state() == QProcess::Running && it->second == true)
        {
            it->first->waitForFinished(30000);
        }
        else if (it->first->state() == QProcess::Starting && it->second == true)
        {
            it->first->waitForStarted(30000);
            it->first->close();
            it->first->waitForFinished(30000);
        }

        ++it;
    }

    // 4. delete every existing process
    it = m_processes.constBegin();

    while (it != m_processes.constEnd())
    {
        it->first->deleteLater();
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
/*static*/ QString ProcessOrganizer::getAbsQtToolPath(
    const QString& binaryName, bool* found /*= NULL*/)
{
    QList<QDir> dirList; // Possible directories
    QList<QString> binList; // Possible binary names

    // Define binary names and possible directories
#ifdef __APPLE__
    binList.append(binaryName); // unchanges file name
    if (!binaryName.endsWith(".app"))
    {
        // .app added
        binList.append(binaryName + ".app");
    }
    binList.append(binaryName);
    binList[2][0] = binList.at(2).at(0).toUpper(); // capitalize first letter
    if (!binaryName.endsWith(".app"))
    {
        // capitalize first letter and .app added
        binList.append(binList.at(2) + ".app");
    }

    dirList.append(QDir(QCoreApplication::applicationDirPath())); // itom app dir
    dirList.append(QDir(QLibraryInfo::location(QLibraryInfo::BinariesPath))); // Qt bin dir
    dirList.append(QDir(QLibraryInfo::location(QLibraryInfo::PrefixPath))); // Qt master dir
    dirList.append(QDir(QCoreApplication::applicationDirPath())); // bin dir in /Library/ and /usr/
    dirList.append(QDir("/Applications")); // global app dir
    dirList.append(QDir(QDir::homePath() + "/Applications")); // user app dir
#elif (defined linux)
    binList.append(binaryName); // unchanges file name

    dirList.append(QDir(QCoreApplication::applicationDirPath())); // itom app dir
    dirList.append(QDir(QLibraryInfo::location(QLibraryInfo::BinariesPath))); // Qt bin dir
#else // WIN32
    if (binaryName.endsWith(".exe"))
    {
        // unchanges file name
        binList.append(binaryName);
    }
    else
    {
        // .exe added
        binList.append(binaryName + ".exe");
    }

    dirList.append(QDir(QCoreApplication::applicationDirPath())); // itom app dir
    dirList.append(QDir(QLibraryInfo::location(QLibraryInfo::BinariesPath))); // Qt bin dir
    QByteArray qtdirenv = qgetenv("QTDIR");

    if (qtdirenv.size() > 0)
    {
        // Qt dir from global defines
        dirList.append(QDir((QString)qtdirenv));
    }
#endif

    // Loop through possible directories
    foreach (const QDir& directory, dirList)
    {
        // Loop through possible binary file names
        foreach (const QString& binary, binList)
        {
            // Check for binary file name to exist in directory
#ifdef WIN32
            if (directory.exists(binary))
            {
                if (found)
                {
                    *found = true;
                }
                return directory.absoluteFilePath(binary);
            }
#else // linux || __APPLE__
#ifdef __APPLE__
            QStringList entryList =
                directory.entryList(QDir::Executable | QDir::Files | QDir::Dirs);
#else // linux
            QStringList entryList = directory.entryList(QDir::Executable | QDir::Files);
#endif
            if (entryList.contains(binary))
            {
                if (found)
                {
                    *found = true;
                }
                return directory.absoluteFilePath(binary);
            }
#endif
        }
    }

    if (found)
    {
        *found = false;
    }

    // nothing found, return as is
    return binaryName;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param forceToCloseAll
    \return RetVal
*/
RetVal ProcessOrganizer::collectGarbage(bool forceToCloseAll /*= false*/)
{
    auto it = m_processes.begin();
    // it = m_processes.begin();

    while (it != m_processes.end())
    {
        if (it->first->state() == QProcess::NotRunning || forceToCloseAll)
        {
            it->first->deleteLater();
            m_processStdOut.remove(it.key()); // removes corresponding output messages
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
QProcess* ProcessOrganizer::getFirstExistingProcess(const QString& name)
{
    auto it = m_processes.constFind(name);

    if (it != m_processes.constEnd())
    {
        return it->first;
    }
    return nullptr;
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
QProcess* ProcessOrganizer::getProcess(
    const QString& name, bool tryToUseExistingProcess, bool& existingProcess, bool closeOnFinalize)
{
    QProcess* process = nullptr;
    existingProcess = false;

    if (tryToUseExistingProcess)
    {
        process = getFirstExistingProcess(name);
        existingProcess = true;
    }

    if (process)
    {
        return process;
    }

    process = new QProcess();

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    connect(
        process,
        SIGNAL(finished(int, QProcess::ExitStatus)),
        this,
        SLOT(processFinished(int, QProcess::ExitStatus)));
#else
    connect(process, &QProcess::finished, this, &ProcessOrganizer::processFinished);
#endif

    connect(process, &QProcess::errorOccurred, this, &ProcessOrganizer::processError);
    connect(process, &QProcess::readyReadStandardOutput, this, &ProcessOrganizer::readyReadStandardOutput);

    m_processes.insert(name, QPair<QProcess*, bool>(process, closeOnFinalize));

    return process;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param exitCode
    \param exitStatus
*/
void ProcessOrganizer::processFinished(int /*exitCode*/, QProcess::ExitStatus /*exitStatus*/)
{
    collectGarbage();
}


//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param error
*/
void ProcessOrganizer::processError(QProcess::ProcessError /*error*/)
{
    collectGarbage();
}


//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

*/
void ProcessOrganizer::readyReadStandardOutput()
{
    // the output message is mainly useful in order to get the socket-number of the designer
    // (started as server) the designer returns this number over stdOut. When opening another
    // ui-file, this ui-file is send over a socket connection
    QProcess* sendingProcess = qobject_cast<QProcess*>(sender());

    if (sendingProcess)
    {
        QByteArray ba = sendingProcess->readAllStandardOutput();
        auto it = m_processes.constBegin();

        while (it != m_processes.constEnd())
        {
            if (it.value().first == sendingProcess)
            {
                QByteArray ba2 = m_processStdOut[it.key()];
                ba2.append(ba);
                m_processStdOut[it.key()] = ba2;
                break;
            }

            it = it++;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param windowName
    \return bool
*/
bool ProcessOrganizer::bringWindowsOnTop(const QString& windowName)
{
#ifdef WIN32

#if UNICODE
    wchar_t* nameArray = new wchar_t[windowName.size() + 2];
    int size = windowName.toWCharArray(nameArray);
    nameArray[size] = '\0';
    HWND THandle = FindWindow(NULL, nameArray);
    delete[] nameArray;
#else
    HWND THandle = FindWindow(NULL, windowName.toLatin1().data());
#endif


    if (THandle)
    {
        long result = GetWindowLong(THandle, GWL_STYLE);
        if (result & WS_MINIMIZE)
        {
            ShowWindow(THandle, 1); // SW_SHOW = 5, SW_NORMAL = 1
        }

        return SetForegroundWindow(THandle);

        // long d = WS_MINIMIZE;
        // long d2 = WS_VISIBLE;
        // long d3 = WS_MAXIMIZE;
        // long d4 = WS_ICONIC;
        // bool r = SetForegroundWindow(THandle);
        // qDebug() << r;
        // return r;
        //
        ////bringWindowToTop only works, if window is not minimized
        // ShowWindow(THandle, 1); //SW_SHOW = 5, SW_NORMAL = 1
        // return BringWindowToTop(THandle);
    }

#endif

    return false;
}

} // namespace ito
