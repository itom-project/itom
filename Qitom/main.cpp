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
#include "organizer/userOrganizer.h"
#include "helper/guiHelper.h"

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

#if WIN32
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



QTextStream *messageStream = NULL;
QMutex msgOutputProtection;

//! Message handler that redirects qDebug, qWarning and qFatal streams to the global messageStream
//!
//!  This method is only registered for this redirection, if the global messageStream is related to the file itomlog.txt.
//!  The redirection is enabled via args passed to the main function.
void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
//    myMessageOutput(type, msg.toLatin1().data());
    msgOutputProtection.lock();

    switch (type)
    {
        case QtDebugMsg:
            (*messageStream) << "[qDebug    " <<  QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "     (File: " << context.file << " Line: " << context.line << " Function: " << context.function << ")\n";
            break;
        case QtWarningMsg:
            (*messageStream) << "[qWarning  " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "     (File: " << context.file << " Line: " << context.line << " Function: " << context.function << ")\n";
            break;
        case QtCriticalMsg:
            (*messageStream) << "[qCritical " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "     (File: " << context.file << " Line: " << context.line << " Function: " << context.function << ")\n";
            break;
        case QtFatalMsg:
            (*messageStream) << "[qFatal    " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "     (File: " << context.file << " Line: " << context.line << " Function: " << context.function << ")\n";
            abort();
    }

    messageStream->flush();
    msgOutputProtection.unlock();
}

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

//! starts application
//!
//!  Starts Application by instantiating MainApplication with a desired GuiType
//!  \sa MainApplication, tGuiType

int main(int argc, char *argv[])
{

    // enable high DPI scaling when it was checked in itom properties
    if (ito::GuiHelper::highDPIFileExists())
    {
        /*QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
        QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);*/
        // DPI_AWARENESS_CONTEXT_UNAWARE show unsharp on 4k monitor with scaling 120%
        // DPI_AWARENESS_CONTEXT_SYSTEM_AWARE looks ugly when move itom onto fullHD monitor with scaling 100% An advancement over the original per-monitor DPI awareness mode, which
        // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 enables applications to access new DPI-related scaling behaviors on a per top-level window basis. 
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2); 
        /*QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
        QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);*/
    } 

    int ret = 0;
#if linux
    // https://www.qt.io/blog/2011/06/03/threaded-opengl-in-4-8
    QCoreApplication::setAttribute(Qt::AA_X11InitThreads);
    bool mthread = QCoreApplication::testAttribute(Qt::AA_X11InitThreads);
#endif
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

    DPI_AWARENESS_CONTEXT SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT dpiContext);

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
        logfile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text);
        messageStream = new QTextStream(&logfile);

        //uncomment that line if you want to print all debug-information (qDebug, qWarning...) to file itomlog.txt
        qInstallMessageHandler(myMessageOutput);
        //first lines in log file
        logfile.write("------------------------------------------------------------------------------------------\n");
        logfile.write(QString(QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") + " Starting itom... \n").toLatin1().constData());
        logfile.write("------------------------------------------------------------------------------------------\n");
    }

    //in debug mode uncaught exceptions as well as uncaught
    //cv::Exceptions will be parsed and also passed to qWarning and qFatal.
    cv::redirectError(itomCvError);
   

    QItomApplication itomApplication(argc, argv);

    QList<QScreen*> screens = qApp->screens();
    for (int ii = 0; ii < screens.length(); ++ii)
    {
        QSize pixelSize = screens[ii]->size();
        QSizeF physicalSize = screens[ii]->physicalSize();
        double devicePixelRatio = screens[ii]->devicePixelRatio();
        double logicalDPIX = screens[ii]->logicalDotsPerInchX();
        double logicalDPIY = screens[ii]->logicalDotsPerInchY();
        double logicalDPI = screens[ii]->logicalDotsPerInch();
        double physicalDPIX = screens[ii]->physicalDotsPerInchX();
        double physicalDPIY = screens[ii]->physicalDotsPerInchY();
        double physicalDPI = screens[ii]->physicalDotsPerInch();

        double pixelValX = pixelSize.width();
        double pixelValY = pixelSize.height();
        double physicalSizeX_cm = physicalSize.width() / 10.0;
        double physicalSizeY_cm = physicalSize.height() / 10.0;
        double calcPixelPerCMX = pixelValX / physicalSizeX_cm;
        double calcPixelPerCMY = pixelValY / physicalSizeY_cm;

        double givenLogicalDotsPerCMX = logicalDPIX * 2.54;
        double givenLogicalDotsPerCMY = logicalDPIY * 2.54;
        double givenLogicalDotsPerCM = logicalDPI * 2.54;

        double givenPhysicalDotsPerCMX = physicalDPIX * 2.54;
        double givenPhysicalDotsPerCMY = physicalDPIY * 2.54;
        double givenPhysicalDotsPerCM = physicalDPI * 2.54;

        double ratioLogicalDPCMvsPPCMX = givenLogicalDotsPerCMX / calcPixelPerCMX;
        double ratioLogicalDPCMvsPPCMY = givenLogicalDotsPerCMY / calcPixelPerCMY;
        double ratioPhysicalDPCMvsPPCMX = givenPhysicalDotsPerCMX / calcPixelPerCMX;
        double ratioPhysicalDPCMvsPPCMY = givenPhysicalDotsPerCMY / calcPixelPerCMY;

        qDebug() << "\n\nScreen: " << ii;
        qDebug() << "logicalDPI: " << logicalDPI;
        qDebug() << "physicalDPI: " << physicalDPI;
        qDebug() << "Device Pixel Ratio: " << devicePixelRatio;
        qDebug() << "Pixel in X-Direction: " << pixelValX;
        qDebug() << "Pixel in Y-Direction: " << pixelValY;
        qDebug() << "Physical Size X-Direction in CM: " << physicalSizeX_cm;
        qDebug() << "Physical Size Y-Direction in CM: " << physicalSizeY_cm;
        qDebug() << "Calculated Pixel Per CM in X-Direction: " << calcPixelPerCMX;
        qDebug() << "Calculated Pixel Per CM in Y-Direction: " << calcPixelPerCMY;
        qDebug() << "Qt Logical Dots Per CM in X-Direction: " << givenLogicalDotsPerCMX;
        qDebug() << "Qt Logical Dots Per CM in Y-Direction: " << givenLogicalDotsPerCMY;
        qDebug() << "Qt Logical Dots Per CM Average: " << givenLogicalDotsPerCM;
        qDebug() << "Qt Physical Dots Per CM in X-Direction: " << givenPhysicalDotsPerCMX;
        qDebug() << "Qt Physical Dots Per CM in Y-Direction: " << givenPhysicalDotsPerCMY;
        qDebug() << "Qt Physical Dots Per CM Average: " << givenPhysicalDotsPerCM;
        qDebug() << "Ratio of Logical Dots Per CM vs Pixel Per CM in X-Direction: "
                 << ratioLogicalDPCMvsPPCMX;
        qDebug() << "Ratio of Logical Dots Per CM vs Pixel Per CM in Y-Direction: "
                 << ratioLogicalDPCMvsPPCMY;
        qDebug() << "Ratio of Physical Dots Per CM vs Pixel Per CM in X-Direction: "
                 << ratioPhysicalDPCMvsPPCMX;
        qDebug() << "Ratio of Physical Dots Per CM vs Pixel Per CM in Y-Direction: "
                 << ratioPhysicalDPCMvsPPCMY;
    }

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

    ret = QDialog::Accepted;
    ito::MainApplication m(ito::MainApplication::standard);

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
            ret = m.execPipManagerOnly();
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

        m.setupApplication(scriptsToOpen, scriptsToExecute);

        qDebug("starting main event loop");

        ret = m.exec();

        qDebug("application exited. call finalize");

        m.finalizeApplication();

        qDebug("finalize done");
    }

    ito::UserOrganizer::closeInstance();

    qInstallMessageHandler(0);

    //close possible logfile
    DELETE_AND_SET_NULL(messageStream);
    if (logfile.fileName().isEmpty() == false)
    {
        logfile.close();
    }

    return ret;
}