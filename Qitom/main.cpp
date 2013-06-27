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
#include <qregexp.h>

void benchmarkTest1()
{
    int64 start, ende;
    double freq = cv::getTickFrequency();

    //1
    int size = 1000000;
    int temp;

    start = cv::getTickCount();
    std::vector<int> a1;
    a1.resize(size);
    for(int i=0;i<size;i++)
    {
        a1[i]=2;
        temp=a1[i];
    }
    a1.clear();
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    int* a2 = new int[size];
    for(int i=0;i<size;i++)
    {
        a2[i]=2;
        temp=a2[i];
    }
    delete[] a2;
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;
}

void benchmarkTest2()
{
    qDebug("benchmarkTest2");
    int64 start, ende;
    double freq = cv::getTickFrequency();


    //2
    int *test = (int*)(new cv::Mat());
    int size = 1000000;
    cv::Mat* ptr = NULL;

    start = cv::getTickCount();
    for(int i=0;i<size;i++)
    {
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for(int i=0;i<size;i++)
    {
        ptr = (cv::Mat*)test;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    for(int i=0;i<size;i++)
    {
        ptr = reinterpret_cast<cv::Mat*>(test);
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;


}

void benchmarkTest3()
{
    ito::DataObject *do1 = NULL; //new ito::DataObject(10000,100,100,ito::tFloat32);
    ito::DataObject *do2 = NULL;//new ito::DataObject(*do1);

    qDebug("benchmarkTest3");
    int64 start, ende;
    double freq = cv::getTickFrequency();

    start = cv::getTickCount();
    do1 = new ito::DataObject(10000,100,100,ito::tFloat32);
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    do2 = new ito::DataObject(*do1);
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    delete do2;
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    start = cv::getTickCount();
    delete do1;
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    //int i=1;
};

void benchmarkTest4()
{
    int64 start, ende;
    double freq = cv::getTickFrequency();
    QString str1 = "guten tag kih ihiu oiuziuzt iztfzutfu iztuztriuz iuztiuztiuztzutut";
    QString str2 = "guten tag kih ihiu oiuziuzt iztfzutfu iztuztriuz iuztiuztiuztzutut";
    QByteArray ba1 = str1.toAscii();
    QByteArray ba2 = str2.toAscii();
    char *c1 = ba1.data();
    char *c2 = ba2.data();
    int num = 10000000;
    int c = -num;
    size_t size = sizeof(char) * std::min( strlen(c1),strlen(c2));

    qDebug() << "benchmarkTest4: " << num;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(str1 == str2) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(ba1 == ba2) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(strcmp(c1,c2)) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;
    c = 0;
    start = cv::getTickCount();
    for(int i = 0; i< num;i++)
    {
        if(memcmp(c1,c2,size)) {c++;}else{c--;}
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq << " result: " << c;

    //int i=1;
};

void benchmarkTest5()
{
    ito::DataObject *do1 = NULL; //new ito::DataObject(10000,100,100,ito::tFloat32);
    ito::DataObject *do2 = NULL;//new ito::DataObject(*do1);

    qDebug("benchmarkTest5");
    int64 start, ende;
    double freq = cv::getTickFrequency();
    size_t j = 0;

    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; i++)
    {
        j += i;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;

    j = 0;
    start = cv::getTickCount();
    for (size_t i = 0 ; i < 1000000; ++i)
    {
        j += i;
    }
    ende = cv::getTickCount();
    qDebug() << "time: " << (ende-start)/freq;
};


//#include "memoryCheck/setDebugNew.h"
//#include "memoryCheck/reportingHook.h"

QTextStream *messageStream = NULL;
QMutex msgOutputProtection;

void myMessageOutput(QtMsgType type, const char *msg)
{

    msgOutputProtection.lock();

    switch (type) {
    case QtDebugMsg:
        (*messageStream) << "[qDebug    " <<  QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        /*fprintf(std, "Debug: %s\n", msg);*/
        break;
    case QtWarningMsg:
        (*messageStream) << "[qWarning  " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        //fprintf(stderr, "Warning: %s\n", msg);
        break;
    case QtCriticalMsg:
        (*messageStream) << "[qCritical " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        //fprintf(stderr, "Critical: %s\n", msg);
        break;
    case QtFatalMsg:
        (*messageStream) << "[qFatal    " << QDateTime::currentDateTime().toString("dd.MM.yy hh:mm:ss") << "] - " << msg << "\r\n";
        //fprintf(stderr, "Fatal: %s\n", msg);
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

    //QString t = "qitom_de_DE.qm";
    //QRegExp r("^qitom_(.*).qm$");
    ////r.setPatternSyntax(QRegExp::Wildcard);
    //int pos = r.indexIn(t);
    //QStringList x = r.capturedTexts();
    //QString v = r.cap(1);

    QFile logfile("itomlog.txt");
    logfile.open(QIODevice::WriteOnly);
    messageStream = new QTextStream(&logfile);
    //qInstallMsgHandler(myMessageOutput);  //uncomment that line if you want to print all debug-information (qDebug, qWarning...) to file itomlog.txt

    //benchmarkTest1();
    //benchmarkTest2();
    //benchmarkTest3();
    //benchmarkTest4();
    //benchmarkTest5();

    //QItomApplication a(argc, argv);       //uncomment that line and comment the next line if you want to catch exceptions propagated through the Qt-event system.
    QApplication a(argc, argv);

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

#if (defined WIN32 || defined WIN64)
    char *oldpath = getenv("path");
    char *newpath = (char*)malloc(strlen(oldpath) + libDir.size() + 10);
    newpath[0] = 0;
    strcat(newpath, "path=");
    strcat(newpath, oldpath);
    strcat(newpath, ";");
    strcat(newpath, libDir.toAscii().data());
    //strcat(newpath, ";D:\\itom\\trunk\\iTOM\\lib"); //;./lib;../lib");
    _putenv(newpath);
    free(newpath);
//    QString appDir = QApplication::applicationDirPath();
//    char *dllPath = new wchar_t[appDir.length() + 100];
////    appDir.truncate(appDir.length() - 2);
//    appDir.append("D:/NewITOM/svn/m12/trunk/iTOM/lib");
//    appDir.append(0);
//    appDir.toWCharArray(dllPath);
//    int nn = SetDllDirectory(dllPath);
//    delete dllPath;
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








