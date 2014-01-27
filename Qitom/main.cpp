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

//#if (defined WIN32 || defined WIN64)
//    #include <QtCore/qt_windows.h>
//#endif

#include "mainApplication.h"
#include "main.h"
#include "organizer/userOrganizer.h"

#define VISUAL_LEAK_DETECTOR 0 //1 if you want to active the Visual Leak Detector (MSVC and Debug only), else type 0, if build with CMake always set it to 0.
#if defined _DEBUG  && defined(_MSC_VER) && (VISUAL_LEAK_DETECTOR > 0 || defined(VISUAL_LEAK_DETECTOR_CMAKE))
    #include "vld.h"
#endif

#include <QtGui/QApplication>

#include <qmap.h>
#include <qhash.h>
#include <qtextstream.h>
#include <qfile.h>
#include <qdatetime.h>
#include <qdir.h>
#include <qmutex.h>

//#include "benchmarks.h"

QTextStream *messageStream = NULL;
QMutex msgOutputProtection;

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


//DOXYGEN FORMAT
//! brief description
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/

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
    

    QFile logfile("itomlog.txt");
    logfile.open(QIODevice::WriteOnly);
    messageStream = new QTextStream(&logfile);
    //qInstallMsgHandler(myMessageOutput);  //uncomment that line if you want to print all debug-information (qDebug, qWarning...) to file itomlog.txt

    

#if defined _DEBUG
    cv::redirectError(itomCvError);
    QItomApplication a(argc, argv);       //uncomment that line and comment the next line if you want to catch exceptions propagated through the Qt-event system.
#else
    QApplication a(argc, argv);
#endif

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

    //and designer path
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

#if (defined WIN32 || defined WIN64)
    libDir = QDir::toNativeSeparators( libDir );

    char *oldpath = getenv("path");
    char *newpath = (char*)malloc(strlen(oldpath) + libDir.size() + designerDir.size() + 11);
    newpath[0] = 0;
    strcat(newpath, "path=");
    strcat(newpath, libDir.toAscii().data()); //set libDir at the beginning of the path-variable
    strcat(newpath, ";");
    strcat(newpath, designerDir.toAscii().data());
    strcat(newpath, ";");
    strcat(newpath, oldpath);
    _putenv(newpath);
    free(newpath);
#endif

    QString defUserName;
    for (int nA = 0; nA < argc; nA++)
    {
        char *pNameFound = NULL;

        pNameFound = strstr(argv[nA], "name=");
        if (pNameFound)
        {
            defUserName = QString(((char*)argv[nA] + 6));
            break;
        }
    }

    int ret = 0;

    MainApplication m(MainApplication::standard);
    if (ito::UserOrganizer::getInstance()->loadSettings(defUserName) != 0)
    {
        qDebug("load program aborted by user");
        ret = 0;
        goto end;
    }

    m.setupApplication();

    qDebug("starting main event loop");

    ret = m.exec();

    qDebug("application exited. call finalize");

    m.finalizeApplication();

    qDebug("finalize done");

end:

    ito::UserOrganizer::closeInstance();

    qInstallMsgHandler(0);
    delete messageStream;
    messageStream = NULL;
    logfile.close();

    return ret;
}