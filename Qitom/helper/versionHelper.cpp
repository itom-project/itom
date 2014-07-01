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

#include "versionHelper.h"
#include "../global.h"
#include "version.h"
#include "opencv2/core/version.hpp"
//#include "../../designerPluginSource/qwt/src/qwt_global.h"
#include "common/addInInterface.h"

#include "common/commonVersion.h"
#include "commonQt/commonQtVersion.h"
#include "PointCloud/pclVersion.h"
#include "DataObject/dataobjVersion.h"

#if ITOM_POINTCLOUDLIBRARY > 0
#include <pcl/pcl_config.h>
#endif

//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (!defined linux)
    #undef _DEBUG
    #include "patchlevel.h"
    #define _DEBUG
#else
    #include "patchlevel.h"
#endif

#ifndef PCL_REVISION_VERSION
    #define PCL_REVISION_VERSION 0
#endif

//QList<QPair<QString, QString> > ito::retrieveITOMVERSIONMAP()
//{
//    QList<QPair<QString, QString> > newList;
//    QPair<QString, QString> newPair;
//
//    char buf[7] = {0};
//    _snprintf(buf, 7, "%i.%i.%i", ITOM_VERSION_MAJOR, ITOM_VERSION_MINOR, ITOM_VERSION_PATCH);
//
//    newPair.first = "itom_Version";
//    newPair.second = QString("%1.%2.%3").arg(QString::number(ITOM_VERSION_MAJOR)).arg(QString::number(ITOM_VERSION_MINOR)).arg(QString::number(ITOM_VERSION_PATCH));
//    newList.append(newPair);
//
//    newPair.first = "itom_SysType";
//    
//#if (defined linux)
//    newPair.second = "Q_OS_LINUX";
//#elif (defined Q_OS_WIN32)
//    #if (defined Q_OS_WIN64)
//        newPair.second = "Windows 64-Bit";
//    #else
//        newPair.second = "Windows 32-Bit";
//    #endif
//    #if (defined _DEBUG) && (!defined linux)
//        newPair.second.append(" DEBUG");
//    #endif
//#else
//    newPair.second = "undefined system";
//#endif
//     
//    newList.append(newPair);
//
//#ifdef USING_SVN
//    newPair.first = "itom_SVNRevision";
//    newPair.second = SVN_REVISION;
//    newList.append(newPair);
//
//    newPair.first = "itom_SVNRevision_Date";
//    newPair.second = SVN_REVISION_DATE;
//    newList.append(newPair);
//
//    newPair.first = "ito_SVNRevision_URL";
//    newPair.second = SVN_REPOSITORY_URL;
//    newList.append(newPair);
//
//    newPair.first = "version_Warnings";
//    QString warning("");
//    bool isClean = SVN_CLEAN_BUILD_FLAG > 0? true:false;
//    if(!isClean)
//    {
//        warning.append(QObject::tr("Warning: The version contains locally changed code!\n"));
//    }
//    else
//    {
//        warning.append(QObject::tr("Build from a clean version.\n"));
//    }
//    newPair.second = warning;
//    newList.append(newPair);
//#elif (defined USING_GIT)
//    newPair.first = "itom_GITHASH";
//    newPair.second = GIT_HASHTAG;
//    newList.append(newPair);
//
//    newPair.first = "itom_GIT_Date";
//    newPair.second = GIT_REVISION_DATE;
//    newList.append(newPair);
//
//    newPair.first = "ito_GIT_URL";
//    newPair.second = GIT_REPOSITORY_URL;
//    newList.append(newPair);
//
//    newPair.first = "version_Warnings";
//    QString warning("");
//    bool isClean = GIT_CLEAN_BUILD_FLAG > 0? false:true;
//    if(!isClean)
//    {
//        warning.append(QObject::tr("Warning: The version contains locally changed and uncomitted code!\n"));
//    }
//    else
//    {
//        warning.append(QObject::tr("Build from a clean version.\n"));
//    }
//    newPair.second = warning;
//    newList.append(newPair);
//#else
//    newPair.first = "version_Warnings";
//    QString warning("");
//    warning.append(QObject::tr("This version of itom not under version control (no GIT or SVN)!\n"));
//    newPair.second = warning;
//    newList.append(newPair);
//#endif
//    
//    newPair.first = "openCV_Version";
//    newPair.second = CV_VERSION;
//    newList.append(newPair);
//
//#if ITOM_POINTCLOUDLIBRARY > 0
//    _snprintf(buf, 7, "%i.%i.%i", PCL_MAJOR_VERSION, PCL_MINOR_VERSION, PCL_REVISION_VERSION);
//    newPair.first = "PCL_Version";
//    newPair.second = buf;
//#else
//    newPair.first = "PCL_Version";
//    newPiar.second = "not compiled";
//#endif
//    newList.append(newPair);
//    
//    newPair.first = "QT_Version";
//    newPair.second = QT_VERSION_STR;
//    newList.append(newPair);
//
//    newPair.first = "QT_Your_Version";
//    newPair.second = qVersion();
//    newList.append(newPair);
//
//    newPair.first = "Py_Version";
//    newPair.second = PY_VERSION;
//    newList.append(newPair);
//
//    //newPair.first = "QwtPlot_Version";
//    //newPair.second = QWT_VERSION_STR;
//    //newList.append(newPair);
//
//    return newList;
//}


//! This function is called to get all version numbers that itom contains
/*!
    Use this function to get a map of all parts of itom that are used with 
    their corresponding version.

    \return returns a QMap<QString,QString> containing the module and the version of a module.
*/
QMap<QString, QString> ito::getItomVersionMap()
{
    QMap<QString, QString> items;

    //itom_version
    items["itom_Version"] = QString("%1.%2.%3").arg(QString::number(ITOM_VERSION_MAJOR)).arg(QString::number(ITOM_VERSION_MINOR)).arg(QString::number(ITOM_VERSION_PATCH));


#if (defined ITOMLIBS_SHARED)
    items["itom_sharedLibs"] = QString("dataobject v.%1; itomCommonLib v.%2; itomCommonQtLib v.%3; pointcloud v.%4").arg(DATAOBJ_VER_STRING, COMMON_VER_STRING , COMMON_QT_VER_STRING, PCL_WRAPPER_VER_STRING);
#else
    items["itom_sharedLibs"] = QObject::tr("none");
#endif

    //itom_SysType
#if (defined linux)
    items["itom_SysType"] = "Q_OS_LINUX";
#elif (defined Q_OS_WIN32)
    #if (defined Q_OS_WIN64)
        items["itom_SysType"] = "Windows 64-Bit";
    #else
        items["itom_SysType"] = "Windows 32-Bit";
    #endif
    #if (defined _DEBUG) && (!defined linux)
        items["itom_SysType"].append(" DEBUG");
    #endif
#else
    items["itom_SysType"] = "undefined system";
#endif
     
    items["itom_compileDate"] = __DATE__;
    items["itom_compileDate"].append(", ");
    items["itom_compileDate"].append( __TIME__);

    items["itom_SVN_Rev"] = "";
    items["itom_SVN_Date"] = "";
    items["itom_SVN_URL"] = "";
    items["itom_GIT_Rev"] = "";
    items["itom_GIT_Rev_Abbrev"] = "";
    items["itom_GIT_Date"] = "";
    items["itom_GIT_URL"] = "";

#ifdef USING_SVN
    items["itom_SVN_Rev"] = SVN_REVISION;
    items["itom_SVN_Date"] = SVN_REVISION_DATE;
    items["itom_SVN_URL"] = SVN_REPOSITORY_URL;

    bool isClean = SVN_CLEAN_BUILD_FLAG > 0? true:false;
    if(!isClean)
    {
        items["version_Warnings"] = QObject::tr("Warning: The version contains locally changed code!\n");
    }
    else
    {
        items["version_Warnings"] = QObject::tr("Build from a clean version.\n");
    }

#elif (defined USING_GIT)
    items["itom_GIT_Rev"] = GIT_HASHTAG;
    items["itom_GIT_Rev_Abbrev"] = GIT_HASHTAG_ABBREV;
    items["itom_GIT_Date"] = GIT_REVISION_DATE;
    items["itom_GIT_URL"] = GIT_REPOSITORY_URL;

    bool isClean = GIT_CLEAN_BUILD_FLAG > 0? false:true;
    if(!isClean)
    {
        items["version_Warnings"] = "Warning: ";
        if(GIT_CLEAN_BUILD_FLAG & 0x01) items["version_Warnings"].append(QObject::tr("The version contains locally changed code! "));
        if(GIT_CLEAN_BUILD_FLAG & 0x02) items["version_Warnings"].append(QObject::tr("The version contains unversioned files (e.g. from pyCache-files)!"));
        items["version_Warnings"].append("\n");
    }
    else
    {
        items["version_Warnings"] = QObject::tr("Build from a clean version.\n");
    }
#else
    items["version_Warnings"] = QObject::tr("This version of itom is not under version control (no GIT or SVN)!\n");
#endif
    
    //OpenCV
    items["openCV_Version"] = CV_VERSION;
    
    //PCL
#if ITOM_POINTCLOUDLIBRARY > 0
    items["PCL_Version"] = QString("%1.%2.%3").arg(PCL_MAJOR_VERSION).arg(PCL_MINOR_VERSION).arg(PCL_REVISION_VERSION);
#else
    items["PCL_Version"] = "not used";
#endif

    //Qt-Stuff
    items["QT_Version"] = QT_VERSION_STR;
    items["QT_Your_Version"] = qVersion();

    //Python
    items["Py_Version"] = PY_VERSION;

    //newPair.first = "QwtPlot_Version";
    //newPair.second = QWT_VERSION_STR;

    //addInInterface
    items["itom_pluginInterface_Version"] = ito_AddInInterface_CurrentVersion;

    return items;
}

