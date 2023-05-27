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
#include "plot/plotVersion.h"
#include "itomWidgets/itomWidgetsVersion.h"
#include "shape/shapeVersion.h"
#include "AddInManager/addInManagerVersion.h"
#include "QPropertyEditor/propertyEditorVersion.h"
#include "common/sharedStructures.h"

#include <QSysInfo>

#if ITOM_POINTCLOUDLIBRARY > 0
#include <pcl/pcl_config.h>
#endif

//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (defined WIN32)
    // see https://github.com/microsoft/onnxruntime/issues/9735#issuecomment-970718821
    #include <corecrt.h>

    #undef _DEBUG
    #include "patchlevel.h"
    #define _DEBUG
#else
    #include "patchlevel.h"
#endif

#ifndef PCL_REVISION_VERSION
    #define PCL_REVISION_VERSION 0
#endif



namespace ito
{

//! This function is called to get all version numbers that itom contains
/*!
    Use this function to get a map of all parts of itom that are used with
    their corresponding version.

    \return returns a QMap<QString,QString> containing the module and the version of a module.
*/
QMap<QString, QString> getItomVersionMap()
{
    QMap<QString, QString> items;

    //itom_version
    items["itom_Version"] = QString("%1.%2.%3").arg(QString::number(ITOM_VERSION_MAJOR)).arg(QString::number(ITOM_VERSION_MINOR)).arg(QString::number(ITOM_VERSION_PATCH));

    items["itom_EditionName"] = QString::fromLatin1(ITOM_ADDITIONAL_EDITION_NAME);


#if (defined ITOMLIBS_SHARED)
	items["itom_sharedLibs"] = QString("dataobject v.%1; itomCommonLib v.%2; itomCommonQtLib v.%3; pointcloud v.%4; plot v.%5; widgets v.%6; shape v.%7; addInManager v.%8; property editor v.%9").arg(\
		DATAOBJ_VERSION_STRING, COMMON_VERSION_STRING, COMMON_QT_VERSION_STRING, PCL_WRAPPER_VERSION_STRING, \
		COMMON_PLOT_VERSION_STRING, ITOM_WIDGETS_VERSION_STRING, SHAPE_VERSION_STRING, ADDINMANAGER_VERSION_STRING, PROPERTYEDITOR_VERSION_STRING);
#else
    items["itom_sharedLibs"] = QObject::tr("none");
#endif

    items["itom_SysType"] = QSysInfo::prettyProductName();

    //detect compiler
    items["itom_compiler"] = "Unknown compiler";

#ifdef Q_CC_BOR
    items["itom_compiler"] = "Borland/Turbo C++";
#endif
#ifdef Q_CC_CLANG
    items["itom_compiler"] = "Clang";
#endif
#ifdef Q_CC_GNU
    items["itom_compiler"] = "GNU C++";
#endif
#ifdef Q_CC_INTEL
    items["itom_compiler"] = "Intel C++ for Linux or Windows";
#endif
#ifdef Q_CC_MSVC
    items["itom_compiler"] = "Microsoft Visual C/C++";
#endif

#if (defined _WIN64)
    items["itom_compiler"].append(", 64-Bit");
#else
    items["itom_compiler"].append(", 32-Bit");
#endif

#if (defined _DEBUG)
    items["itom_compiler"].append(", Debug");
#else
    items["itom_compiler"].append(", Release");
#endif

    items["itom_compileDate"] = __DATE__;
    items["itom_compileDate"].append(", ");
    items["itom_compileDate"].append( __TIME__);

    items["itom_GIT_Rev"] = "";
    items["itom_GIT_Rev_Abbrev"] = "";
    items["itom_GIT_Date"] = "";
    items["itom_GIT_URL"] = "";

#ifdef USING_GIT
    items["itom_GIT_Rev"] = GIT_HASHTAG;
    items["itom_GIT_Rev_Abbrev"] = GIT_HASHTAG_ABBREV;
    items["itom_GIT_Date"] = GIT_REVISION_DATE;
    items["itom_GIT_URL"] = GIT_REPOSITORY_URL;

    bool isClean = GIT_CLEAN_BUILD_FLAG > 0? false:true;
    if(!isClean)
    {
        items["version_Warnings"] = "Warning: ";
        if(GIT_CLEAN_BUILD_FLAG & 0x01) items["version_Warnings"].append(QObject::tr("The version contains locally changed code! "));
        if(GIT_CLEAN_BUILD_FLAG & 0x02) items["version_Warnings"].append(QObject::tr("The version contains unversioned files (e.g. from __pycache__-files)!"));
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
    items["itom_pluginInterface_Version"] = ITOM_ADDININTERFACE_VERSION_STR;

    return items;
}

} //end namespace ito
