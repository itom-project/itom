/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include "organizer/userOrganizer.h"

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
#include <qsysinfo.h>

//#include "benchmarks.h"

//DOXYGEN FORMAT
//! brief description
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/

QTextStream *messageStream = NULL;
QMutex msgOutputProtection;

//! Message handler that redirects qDebug, qWarning and qFatal streams to the global messageStream
/*!
    This method is only registered for this redirection, if the global messageStream is related to the file itomlog.txt.

    The redirection is enabled via args passed to the main function.
*/
#if QT_VERSION < 0x050000
void myMessageOutput(QtMsgType type, const char *msg)
{
    msgOutputProtection.lock();

    switch (type) {
    case QtDebugMsg:
        (*messageStream) << "[qDebug    " <<  QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        break;
    case QtWarningMsg:
        (*messageStream) << "[qWarning  " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        break;
    case QtCriticalMsg:
        (*messageStream) << "[qCritical " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        break;
    case QtFatalMsg:
        (*messageStream) << "[qFatal    " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        abort();
    }

    messageStream->flush();
    msgOutputProtection.unlock();
}
#else
void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
//    myMessageOutput(type, msg.toLatin1().data());
    msgOutputProtection.lock();

    switch (type) 
    {
        case QtDebugMsg:
            (*messageStream) << "[qDebug    " <<  QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << " File: " << context.file << " Line: " << context.line << " Function: " << context.function << "\r\n";
            break;
        case QtWarningMsg:
            (*messageStream) << "[qWarning  " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << " File: " << context.file << " Line: " << context.line << " Function: " << context.function << "\r\n";
            break;
        case QtCriticalMsg:
            (*messageStream) << "[qCritical " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << " File: " << context.file << " Line: " << context.line << " Function: " << context.function << "\r\n";
            break;
        case QtFatalMsg:
            (*messageStream) << "[qFatal    " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << " File: " << context.file << " Line: " << context.line << " Function: " << context.function << "\r\n";
            abort();
    }

    messageStream->flush();
    msgOutputProtection.unlock();
}
#endif

//! OpenCV error handler
/*!
    In case of a call of cv::error in any OpenCV method or the dataObject,
    this method is called. Afterwards the error is thrown. In this method, the
    error message is print to the output window.
*/
int itomCvError( int status, const char* func_name,
            const char* err_msg, const char* file_name,
            int line, void* userdata )
{
    char buf[1 << 16];

    sprintf( buf, "OpenCV Error: %s (%i) in %s, file %s, line %d",
        err_msg, status, (func_name > 0 && strlen(func_name) > 0) ?
        func_name : "unknown function", file_name, line );
    qWarning("Itom-Application has caught a cv::exception");
    qWarning() << buf;

    return 0; //Return value is not used
}

//! starts application
/*!
    Starts Application by instantiating MainApplication with a desired GuiType
    \sa MainApplication, tGuiType
*/
int main(int argc, char *argv[])
{
#if linux
#if (((QT_VERSION & 0xFF0000) >= 0x40000) && ((QT_VERSION & 0X00FF0) >= 0x800))
    // http://labs.qt.nokia.com/2011/06/03/threaded-opengl-in-4-8/
    QCoreApplication::setAttribute(Qt::AA_X11InitThreads);
    bool mthread = QCoreApplication::testAttribute(Qt::AA_X11InitThreads);
#endif
#endif

    //startBenchmarks();
    
    //parse arguments passed to the executable
    /*
        possible arguments are:

        *.py : opens the given python file in the script editor
        log : writes all messages sent via qDebug, qWarning... to the logfile itomlog.txt
               in the itom application directory.
        name=anyUsername : tries to start itom with the given username (different setting file)
        pipManager : only opens the Python Pip Manager to update packages like Numpy. Numpy cannot be updated if itom is running since Numpy is used and files are blocked.
    */
    QStringList args;
    for (int i = 0; i < argc; ++i)
    {
        args << argv[i];
    }

    //it is possible to redirect all Qt messages sent via qDebug, qWarning... to the logfile itomlog.txt.
    //This option is enabled via the argument log passed to the executable.
    QFile logfile;
    if (args.contains("log", Qt::CaseInsensitive))
    {
        logfile.setFileName("itomlog.txt");
        logfile.open(QIODevice::WriteOnly);
        messageStream = new QTextStream(&logfile);
#if QT_VERSION < 0x050000
        qInstallMsgHandler(myMessageOutput);  //uncomment that line if you want to print all debug-information (qDebug, qWarning...) to file itomlog.txt
#else
        qInstallMessageHandler(myMessageOutput);
#endif
    }    

#if defined _DEBUG
    //in debug mode uncaught exceptions as well as uncaught cv::Exceptions will be parsed and also passed to qWarning and qFatal.
    cv::redirectError(itomCvError);
    QItomApplication a(argc, argv);
#else
    //in release an uncaught exception will exit the application.
    QApplication a(argc, argv);
#endif

    //itom modifies its local environment variables like PATH such that plugin libraries, python... that are loaded later
    //benefit from necessary pathes that are then guaranteed to be found.
    /*
        These things are done:

        * Prepend the subfolder 'lib' of the itom application directory to the PATH environment variable.
             Plugins can place further required 3rd party libraries inside of this folder that is then
             searched during the load of any plugins.
        * Prepend the subfolder 'designer' of the itom application directory to the PATH environment variable.
             This subfolder contains designer plugins that can then be loaded by the QtDesigner started via itom.
        * Create the environment variable MPLCONFIGDIR whose value is the absolute path to the subfolder 'itom-packages/mpl_itom'.
             If you use the python package matplotlib, you can place a modified matplotlib config file there that is then
             used for matplotlib configurations. For instance it is recommended to modify the backend variable in this file,
             such that matplotlib renders its content inside of an itom widget per default.
    */
    //parse lib path:
    QDir appLibPath = QDir(a.applicationDirPath());
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
    appLibPath = QDir(a.applicationDirPath());
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
    appLibPath = QDir(a.applicationDirPath());
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
    char *oldpath = getenv("path");
    char pathSep[] = ";";
#else
    char *oldpath = getenv("PATH");
    char pathSep[] = ":";
#endif
    char *newpath = (char*)malloc(strlen(oldpath) + libDir.size() + designerDir.size() + 11);
    newpath[0] = 0;
#ifdef WIN32
    strcat(newpath, "path=");

#if WINVER > 0x0502 
    if (QSysInfo::windowsVersion() > QSysInfo::WV_XP)
    {
#if UNICODE
        //sometimes LoadLibrary commands in plugins with files that are located in the lib folder cannot be loaded
        //even if the lib folder is add to the path variable in this funtion, too. The SetDllDirectory
        //is another approach to reach this (only available since Win XP).
        wchar_t *lib_path = new wchar_t[libDir.size() + 5];
        memset(lib_path, 0, (libDir.size() + 5) * sizeof(wchar_t));
        libDir.toWCharArray(lib_path);
        SetDllDirectory(lib_path);
        delete lib_path;
#else
        SetDllDirectory(libDir.toLatin1().data());
#endif
    }
#endif
#else
#endif
    strcat(newpath, libDir.toLatin1().data()); //set libDir at the beginning of the path-variable
    strcat(newpath, pathSep);
    strcat(newpath, designerDir.toLatin1().data());
    strcat(newpath, pathSep);
    strcat(newpath, oldpath);
#ifdef WIN32
    _putenv(newpath);

    HMODULE ximeaLib2 = LoadLibrary(L"m3apiX64.dll");

    //this is for the matplotlib config file that is adapted for itom.
    mpl_itomDir = QString("MPLCONFIGDIR=%1").arg(mpl_itomDir);
    _putenv(mpl_itomDir.toLatin1().data());
    _putenv("MPLBACKEND=module://mpl_itom.backend_itomagg"); //set the default backend for matplotlib (only taken into account for matplotlib >= 1.5) to the itom backend
#else
    setenv("PATH", newpath, 1);
    setenv("MPLCONFIGDIR", mpl_itomDir.toLatin1().data(), 1);
    setenv("MPLBACKEND", "module://mpl_itom.backend_itomagg", 1);
#endif
    free(newpath);

    //itom has an user management. If you pass the string name=[anyUsername] to the executable,
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
    /*
        1. create MainApplication
        2. load the default or user defined setting file (*.ini)
        3. setupApplication()
        4. start the application's main loop
        5. finalizeApplication() if itom is closed
    */
    int ret;
    ito::MainApplication m(ito::MainApplication::standard);
    if (ito::UserOrganizer::getInstance()->loadSettings(defUserName) != ito::retOk)
    {
        ret = 0; 
        qDebug("load program aborted, possibly unknown username (check argument name=...)");
    }
    else if (args.contains("pipManager"))
    {
        if (ito::UserOrganizer::getInstance()->hasFeature(ito::featDeveloper))
        {
            ret = m.execPipManagerOnly();
        }
        else
        {
            ret = 0; 
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
    else
    {
        //check if args contains entries with .py at the end, these files should be opened as scripts at startup
        QStringList scriptsToOpen;
        foreach(const QString &a, args)
        {
            if (a.endsWith(".py", Qt::CaseInsensitive))
            {
                scriptsToOpen << a;
            }
        }

        m.setupApplication(scriptsToOpen);

        qDebug("starting main event loop");

        ret = m.exec();

        qDebug("application exited. call finalize");

        m.finalizeApplication();

        qDebug("finalize done");
    }

    ito::UserOrganizer::closeInstance();

    #if QT_VERSION >= 0x050000
    qInstallMessageHandler(0);
    #else
    qInstallMsgHandler(0);
    #endif

    //close possible logfile
    DELETE_AND_SET_NULL(messageStream);
    if (logfile.fileName().isEmpty() == false)
    {
        logfile.close();
    }

    return ret;
}
