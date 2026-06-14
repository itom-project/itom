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
#include "mainApplication.h"
#include "main.h"
#include "AppManagement.h"
#include "organizer/userOrganizer.h"
#include "common/itomLog.h"

#define VISUAL_LEAK_DETECTOR 0 //1 if you want to active the Visual Leak Detector (MSVC and Debug only), else type 0, if build with CMake always set it to 0.
#if defined _DEBUG  && defined(_MSC_VER) && (VISUAL_LEAK_DETECTOR > 0 || defined(VISUAL_LEAK_DETECTOR_CMAKE))
    #include "vld.h"
#endif

//#include <QtGui/QApplication>
#include <qapplication.h>
#include <qmap.h>
#include <qhash.h>
#include <qtextstream.h>
#include <qfile.h>
#include <qdatetime.h>
#include <qdir.h>
#include <qmutex.h>
#include <qmessagebox.h>
#include <QSysInfo>
#include <QScreen>

#ifdef WIN32
#include <Windows.h>
#include <VersionHelpers.h>
#include <ShellScalingApi.h>
#endif

//#include "benchmarks.h"

/**
 * \mainpage itom
 *
 * \section intro_sec Introduction
 *
 * ITOM is an open source software suite for operating measurement systems, laboratory automation and data evaluation.
 * It can be used in a multitude of application areas, but especially is devoted to optical systems and image processing.
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS
//DOXYGEN FORMAT
//! brief description
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
#endif /*DOXYGEN_SHOULD_SKIP_THIS*/



//! OpenCV error handler
//!
//!  In case of a call of cv::error in any OpenCV method or the dataObject,
//!  this method is called. Afterwards the error is thrown. In this method, the
//!  error message is print to the output window.

int itomCvError( int status, const char* func_name,
            const char* err_msg, const char* file_name,
            int line, void* /*userdata*/ )
{
    QString err = QString("OpenCV Error: %1 (%2) in %3, file %4, line %5"). \
        arg(err_msg). \
        arg(status). \
        arg((func_name && strlen(func_name) > 0) ? func_name : "unknown function"). \
        arg(file_name). \
        arg(line);

    qWarning("Itom-Application has caught a cv::exception");
    qWarning() << err;

    return 0; //Return value is not used
}

#ifdef WIN32 // only for windows
DWORDLONG GetWindowsBuildAndServicePackVersion(bool onlyVersion = true)
{
    // https://stackoverflow.com/questions/32115255/c-how-to-detect-windows-10/52122386#52122386
    NTSTATUS(WINAPI * RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
    }

    if (onlyVersion)
    {
        return osInfo.dwMajorVersion | osInfo.dwMinorVersion |
            osInfo.wServicePackMajor | osInfo.wServicePackMinor;
    }
    else
    {
        return osInfo.dwMajorVersion | osInfo.dwMinorVersion | osInfo.dwBuildNumber | osInfo.wServicePackMajor |
                osInfo.wServicePackMinor;
    }
}

bool IsWindows10BuildVersionOrLater(uint32_t inVersion)
{
    return GetWindowsBuildAndServicePackVersion(false) >= inVersion ? true : false;
}

bool IsWindowsVersionOrLater(uint32_t inVersion)
{
    return GetWindowsBuildAndServicePackVersion() >= inVersion ? true : false;
}

bool IsWin7SP1OrLater()
{
    return IsWindowsVersionOrLater(0x06010100ul);
}

bool IsWin8OrLater()
{
    return IsWindowsVersionOrLater(0x06020000ul);
}

bool IsWin8Point1OrLater()
{
    return IsWindowsVersionOrLater(0x06030000ul);
}

bool IsWin10OrLater()
{
    return IsWindowsVersionOrLater(0x0a000000ul);
}

bool IsWin10November2015UpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(10586);
}

bool IsWin10AnniversaryUpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(14393);
}

bool IsWin10CreatorsUpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(15063);
}

bool IsWin10FallCreatorsUpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(16299);
}

bool IsWin10April2018UpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(17134);
}

bool IsWin10Sep2018UpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(17763);
}

bool IsWin10May2019UpdateOrLater()
{
    return IsWindows10BuildVersionOrLater(18362);
}

bool IsWin11OrLater()
{
    return IsWindows10BuildVersionOrLater(22000);
}

bool IsNotWin7PreRTM()
{
    return IsWin7SP1OrLater() || IsWindows10BuildVersionOrLater(7600);
}

#endif

//! starts application
//!
//!  Starts Application by instantiating MainApplication with a desired GuiType
//!  \sa MainApplication, tGuiType
int main(int argc, char *argv[])
{
#ifdef WIN32
#if WDK_NTDDI_VERSION >= 0x0A000002 // only for windows 10 anniversity update because SetProcessDpiAwarenessContext was introduced
    //  https://docs.microsoft.com/de-de/windows/win32/winprog/using-the-windows-headers
    //  https://naughter.wordpress.com/2017/02/14/changes-in-the-windows-v10-0-15021-sdk-compared-to-windows-v10-0-14393-sdk-part-one/
    //  https://searchfox.org/mozilla-central/source/mfbt/WindowsVersion.h#84
    //  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) was added in the Win10 Creator Update
    //  SetProcessDpiAwarenessContext() was added in the Win10 Anniversary Update
    //  SetProcessDpiAwareness() was added in Windows 8.1
    //  SetProcessDpiAware() was added in Windows Vista

    if (IsWin10CreatorsUpdateOrLater())
    {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
    else if (IsWin10AnniversaryUpdateOrLater())
    {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE);
    }
#endif
#endif
    qputenv("QT_AUTO_SCREEN_SCALE_FACTOR", "1");  // auto scale by qt

    //startBenchmarks();

    //parse arguments passed to the executable

    //  possible arguments are:

    //      *.py : opens the given python file in the script editor
    //      log : writes all messages sent via qDebug, qWarning... to the logfile itomlog.txt
    //              in the itom application directory.
    //      name=anyUsername : tries to start itom with the given username (different setting file)
    //      run=pathToPythonFile : runs the given script (if it exists) after possible autostart scripts added to the selected user role, put
    //                             'pathToPythonFile' in "..." if it contains spaces or other special characters. You can stack multiple run= items to execute multiple scripts.
    //      pipManager : only opens the Python Pip Manager to update packages like Numpy. Numpy cannot be updated if itom is running since Numpy is used and files are blocked.

    QStringList args;
    for (int i = 0; i < argc; ++i)
    {
        args << argv[i];
    }

    //it is possible to redirect all Qt messages sent via qDebug, qWarning... to the logfile itomlog.txt.
    //This option is enabled via the argument log passed to the executable.
    ito::Logger* logger = nullptr;
    if (!args.contains("nolog", Qt::CaseInsensitive))
    {
        QString logFileDir = "";
        QStringListIterator i(args);
        while (i.hasNext())
        {
            QString arg = i.next();
            if (arg.startsWith("log="))
            {
                logFileDir = arg.mid(4);
            }
        }
        logger = new ito::Logger("itomlog.txt", logFileDir, 5 * 1024 * 1024, 2);
        ito::AppManagement::setLogger((QObject*)logger);
    }

    //in debug mode uncaught exceptions as well as uncaught
    //cv::Exceptions will be parsed and also passed to qWarning and qFatal.
    cv::redirectError(itomCvError);


    QItomApplication itomApplication(argc, argv);

    //itom modifies its local environment variables like PATH such that plugin libraries, python... that are loaded later
    //benefit from necessary pathes that are then guaranteed to be found.

    //      These things are done:

    //      * Prepend the subfolder 'lib' of the itom application directory to the PATH environment variable.
    //          Plugins can place further required 3rd party libraries inside of this folder that is then
    //          searched during the load of any plugins.
    //      * Prepend the subfolder 'designer' of the itom application directory to the PATH environment variable.
    //          This subfolder contains designer plugins that can then be loaded by the QtDesigner started via itom.
    //      * Create the environment variable MPLCONFIGDIR whose value is the absolute path to the subfolder 'itom-packages/mpl_itom'.
    //          If you use the python package matplotlib, you can place a modified matplotlib config file there that is then
    //          used for matplotlib configurations. For instance it is recommended to modify the backend variable in this file,
    //          such that matplotlib renders its content inside of an itom widget per default.
    //parse lib path:
    QDir appLibPath = QDir(itomApplication.applicationDirPath());
    if(appLibPath.exists("lib"))
    {
        appLibPath.cd("lib");
    }
    else
    {
        appLibPath.cdUp();
        appLibPath.cd("lib");
    }
    QString libDir = QDir::cleanPath(appLibPath.filePath(""));
    libDir = QDir::toNativeSeparators( libDir );

    //and designer path
    appLibPath = QDir(itomApplication.applicationDirPath());
    if(appLibPath.exists("designer"))
    {
        appLibPath.cd("designer");
    }
    else
    {
        appLibPath.cdUp();
        appLibPath.cd("designer");
    }
    QString designerDir = QDir::cleanPath(appLibPath.filePath(""));

    //search for mpl_itom path in itom-packages
    appLibPath = QDir(itomApplication.applicationDirPath());
    if(appLibPath.exists("itom-packages"))
    {
        appLibPath.cd("itom-packages");
        appLibPath.cd("mpl_itom");
    }
    else
    {
        appLibPath.cdUp();
        appLibPath.cd("itom-packages");
        appLibPath.cd("mpl_itom");
    }
    QString mpl_itomDir = QDir::cleanPath(appLibPath.filePath(""));

#ifdef WIN32
    QByteArray oldpath = qgetenv("path");
    QChar pathSep = ';';
#else
    QByteArray oldpath = QByteArray(getenv("PATH"));
    QChar pathSep = ':';
#endif

    QByteArray newpath;
#ifdef WIN32
    newpath += "path=";

    #if WINVER > 0x0502
        if (IsWindowsVistaOrGreater())
        {
            SetDllDirectoryA(libDir.toLatin1().data());
        }
    #endif
#else
#endif
    newpath += libDir.toLatin1(); //set libDir at the beginning of the path-variable
    newpath += pathSep.toLatin1();
    newpath += designerDir.toLatin1();
    newpath += pathSep.toLatin1();
    newpath += oldpath;
#ifdef WIN32
    _putenv(newpath.constData());

    //this is for the matplotlib config file that is adapted for itom.
    mpl_itomDir = QString("MPLCONFIGDIR=%1").arg(mpl_itomDir);
    _putenv(mpl_itomDir.toLatin1().data());
    _putenv("MPLBACKEND=module://mpl_itom.backend_itomagg"); //set the default backend for matplotlib (only taken into account for matplotlib >= 1.5) to the itom backend
#else
    setenv("PATH", newpath, 1);
    setenv("MPLCONFIGDIR", mpl_itomDir.toLatin1().data(), 1);
    setenv("MPLBACKEND", "module://mpl_itom.backend_itomagg", 1);
#endif

    //itom has an user management. If you pass the string name=[anyUsername] to the executable,
    //a setting file itom_{anyUsername}.ini is searched and if found loaded. Pass itom.ini as anyUsername
    //to explicitely load the default setting file itom.ini. If no username is given and more than
    //one settings ini file is available, a selection dialog is shown.
    //another setting file than the default file itom.ini will be loaded for this session of itom.
    //Therefore all settings files in the folder itomSettings matching itom_*.ini are checked for
    //a group
    //
    //[ITOMIniFile]
    //name = anyUsername
    //
    //and if found, the setting file is used.
    QString defUserName;
    foreach (const QString &arg, args)
    {
        if (arg.startsWith("name="))
        {
            defUserName = arg.mid(5);
            break;
        }
    }

    QDir tmp(QDir::tempPath());
    if (tmp.exists("restart_itom_with_pip_manager.txt"))
    {
        args.append("pipManager");
    }

    //now the main things for loading itom are done:

    //  1. create MainApplication
    //  2. load the default or user defined setting file (*.ini)
    //  3. setupApplication()
    //  4. start the application's main loop
    //  5. finalizeApplication() if itom is closed

    int ret = QDialog::Accepted;
    ito::MainApplication mainApp(ito::MainApplication::standard);

    ito::RetVal userRetVal = ito::UserOrganizer::getInstance()->loadSettings(defUserName);

    if (userRetVal.containsError())
    {
        if (userRetVal.hasErrorMessage())
        {
            QMessageBox::critical(NULL, QObject::tr("User Management"), userRetVal.errorMessage());
            qDebug() << userRetVal.errorMessage();
        }

        ret = QDialog::Rejected;
        qDebug("load program aborted, possibly unknown username (check argument name=...)");
    }
    else if (args.contains("pipManager"))
    {
        if (ito::UserOrganizer::getInstance()->currentUserHasFeature(ito::featDeveloper))
        {
            ret = mainApp.execPipManagerOnly();
        }
        else
        {
            ret = QDialog::Rejected;
            qDebug("chosen user has no rights to start the Python Pip Manager");
        }

        if (tmp.exists("restart_itom_with_pip_manager.txt"))
        {
            QFile file(tmp.absoluteFilePath("restart_itom_with_pip_manager.txt"));
            if (!file.remove())
            {
                qDebug("the file %s could not be deleted. Please delete it manually", tmp.absoluteFilePath("restart_itom_with_pip_manager.txt").toLatin1().data());
            }
        }
    }

    if (ret == QDialog::Accepted)
    {
        //check if args contains entries with .py at the end, these files should be opened as scripts at startup
        QStringList scriptsToOpen;
        QStringList scriptsToExecute;
        foreach(const QString &a, args)
        {
            if (a.endsWith(".py", Qt::CaseInsensitive))
            {
                if (a.startsWith("run="))
                {
                    scriptsToExecute << a.mid(QString("run=").size());
                }
                else
                {
                    scriptsToOpen << a;
                }
            }
        }

        mainApp.setupApplication(scriptsToOpen, scriptsToExecute);

        qDebug("starting main event loop");

        ret = mainApp.exec();

        qDebug("application exited. call finalize");

        mainApp.finalizeApplication();

        qDebug("finalize done");
    }

    ito::UserOrganizer::closeInstance();

    qInstallMessageHandler(0);

    // close possible logger
    DELETE_AND_SET_NULL(logger);

    return ret;
}
