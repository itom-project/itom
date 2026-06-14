/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom and its software development toolkit (SDK).

itom is free software; you can redistribute it and/or modify it
under the terms of the GNU Library General Public Licence as published by
the Free Software Foundation; either version 2 of the Licence, or (at
your option) any later version.

In addition, as a special exception, the Institut fuer Technische
Optik (ITO) gives you certain additional rights.
These rights are described in the ITO LGPL Exception version 1.0,
which can be found in the file LGPL_EXCEPTION.txt in this package.

itom is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
General Public Licence for more details.

You should have received a copy of the GNU Library General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef ADDININTERFACEVERSION_H
#define ADDININTERFACEVERSION_H

#include "typeDefs.h"

//###########################################################################################################
//   Interface version:
//###########################################################################################################
//
//
// Please change the interface version, if you made any changes to this interface, files located in the common folder or to the dataObject.
//
// To add a new version, do the following steps
//
// 1. change ITOM_ADDININTERFACE_MAJOR, ITOM_ADDININTERFACE_Minor and/or ITOM_ADDININTERFACE_Patch
// 2. append the string behind the variable ito_AddInInterface_CurrentVersion (e.g. ito.AddIn.InterfaceBase/1.1) to the array ito_AddInInterface_OldVersions
// 3. change the version number in the string ito_AddInInterface_CurrentVersion
//
//
// This helps, that deprecated or "future" plugins, which fit not to the current implementation of the interface will not be loaded
// but a sophisticated error message is shown.
//
// From version 4.0.0 on, the version numbers follow the schematic of semantic versioning (semver.org).
// In general, any plugin can be loaded whose interface is based on the CREATE_ADDININTERFACE_MAJOR_VERSION_STR (major version number must fit).
// itom loads such a plugin if its minor version number is lower or equal than the minor version number, defined in the compiled version of the core
// application of itom. The patch number does not influence the ability to load a plugin library.

static const char* ito_AddInInterface_OldVersions[] = {
    "ito.AddIn.InterfaceBase/1.0",   //version from start of development until 2012-10-11
    "ito.AddIn.InterfaceBase/1.1",   //version until 2012-10-15 (outdated due to changes in dataObject)
    "ito.AddIn.InterfaceBase/1.1.1", //version until 2012-10-21 (outdated due to small changes in addInInterface)
    "ito.AddIn.InterfaceBase/1.1.2", //version until 2012-10-30 (outdated due to small changes in addInInterface)
    "ito.AddIn.InterfaceBase/1.1.3", //version until 2012-11-09 (outdated due to changes in the checkData()-function in addInGrabber)
    "ito.AddIn.InterfaceBase/1.1.4", //version until 2012-11-12 (outdated due to changes in the DataObject)
    "ito.AddIn.InterfaceBase/1.1.5", //version until 2012-11-18 (outdated: changes in the API structure)
    "ito.AddIn.InterfaceBase/1.1.6", //version until 2012-12-20 (outdated: added paramBase metatype)
    "ito.AddIn.InterfaceBase/1.1.7", //version until 2013-01-17 (outdated: last version for itom version 1.0.5)
    "ito.AddIn.InterfaceBase/1.1.8", //version until 2013-01-23 (outdated: changes in auto-grabbing of cameras, first interface for itom version 1.0.6)
    "ito.AddIn.InterfaceBase/1.1.9", //version until 2013-03-04 (outdated: changes in dataObjectHelper)
    "ito.AddIn.InterfaceBase/1.1.10",//version until 2013-03-12 (outdated: Added license and about string to the plugin)
    "ito.AddIn.InterfaceBase/1.1.11",//version until 2013-03-22 (outdated: bugfix in ito::ParamBase)
    "ito.AddIn.InterfaceBase/1.1.12",//version until 2013-03-25 (outdated: changes in api)
    "ito.AddIn.InterfaceBase/1.1.13",//version until 2013-04-08 (outdated: removed transpose flag in dataObject)
    "ito.AddIn.InterfaceBase/1.1.14",//version until 2013-04-17 (outdated: uniqueID and identifier inserted/changed)
    "ito.AddIn.InterfaceBase/1.1.15",//version until 2013-04-23 (outdated: made some tag-space related methods non-inline (due to linker errors in MSVC))
    "ito.AddIn.InterfaceBase/1.1.16",//version until 2013-06-07 (outdated: added qpluginloader to the interface for cleaner unloading of plugins)
    "ito.AddIn.InterfaceBase/1.1.17",//version until 2013-06-11 (outdated: apis extended, changes in param-class)
    "ito.AddIn.InterfaceBase/1.1.18",//version until 2013-06-18 (outdated: iterator and constIterator introduced for dataObject)
    "ito.AddIn.InterfaceBase/1.1.19",//version until 2013-08-15 (outdated: operators +, +=, -, -= introduced for scalar operands)
    "ito.AddIn.InterfaceBase/1.1.20",//version until 2013-10-10 (outdated: RGBA-type introduced into dataObjectTypes)
    "ito.AddIn.InterfaceBase/1.1.21",//version until 2013-10-15 (outdated: getSize(..) and getTotalSize(..) return int now and -1 if error. Consistency to documented behaviour)
    "ito.AddIn.InterfaceBase/1.1.22",//version until 2013-10-27 (outdated: class Rgba32Base in typedefs.h and inherited class Rgba32 in color.h introduced, improved data() method in dataObj)
    "ito.AddIn.InterfaceBase/1.1.23",//version until 2013-12-17 (outdated: changed dataObject internal size parameters (back) from size_t to int - hopfully last time)
    "ito.AddIn.InterfaceBase/1.1.24",//version until 2014-02-09 (outdated: restructuring to itomCommonLib and itomCommonQtLib for a better binary compatibility)
    "ito.AddIn.InterfaceBase/1.2.0", //outdated on 2014-03-14 due to change in AddInDataIO::setVal(const char *data, const int length, ItomSharedSemaphore *waitCond = NULL); (const void *data changed to const char *data) (Qt5 bugfix)
    "ito.AddIn.InterfaceBase/1.2.1", //outdated on 2014-10-06 due to changes in APIs, retVal.h and itomWidgets-project. The next version 1.3.0 is the version for the setup 1.3.0.
    "ito.AddIn.InterfaceBase/1.3.0", //outdated on 2014-10-27 due to insertion of ito::AutoInterval object and addition of further ito::ParamMeta classes.
    "ito.AddIn.InterfaceBase/1.3.1", //outdated on 2015-03-01 due to rework on data object
    "ito.AddIn.InterfaceBase/1.4.0", //outdated on 2015-07-03 due to removal of lock mechanism in data object, add of embedded line plots, qt5 incompatiblity changes and some refinements in addInInterface
    "ito.AddIn.InterfaceBase/2.0.0", //outdated on 2015-12-04 due to improvements in plot/figure interfaces, removal of deprecated classes helperActuator and helperGrabber and further removal of deprecated items
    "ito.AddIn.InterfaceBase/2.1.0", //outdated on 2016-02-01 due to improvements in PluginThreadCtrl, ActuatorThreadCtrl and DataIoThreadCtrl (as replacement for removed classes helperActuator and helperGrabber), new method ito::DataObject::getStep and some smaller rearrangements
    "ito.AddIn.InterfaceBase/2.2.0", //outdated on 2016-02-19 due to crash fixes if the main mindow is deleted and implicitely closes dock widgets of plugins, that are currently blocked by any other operation.
    "ito.AddIn.InterfaceBase/2.3.0", //outdated on 2016-06-14 due to changes in signal definitions in plots, introduction of complex and complexArray types in ParamBase and further smaller changes.
    "ito.AddIn.InterfaceBase/2.4.0", //outdated on 2016-07-12 due to new library itomCommonPlotLib.
    "ito.AddIn.InterfaceBase/2.5.0", //outdated on 2017-02-05 due to changes in ParamMeta classes
    "ito.AddIn.InterfaceBase/2.6.0", //outdated on 2017-02-05 since the AddInManager has been separated into its own shared library
    "ito.AddIn.InterfaceBase/3.0.0", //outdated on 2017-12-06 due to change of type (float to double) in ito::AutoInterval
    "ito.AddIn.InterfaceBase/3.1.0", //outdated on 2018-01-10 due to introduction of xData feature in plots
    "ito.AddIn.InterfaceBase/3.2.0", //outdated on 2019-03-03 due to cleanup in AddInInterface including Private-classes for all AddIn classes and the ability to return the last reported state and position of axes (even while the axis is currently moving)
    "ito.AddIn.InterfaceBase/3.3.0", //outdated on 2019-11-19 due to new FilterDefExt class and semver based interface numbering
    "ito.AddIn.InterfaceBase/4.0.0", //outdated on 2020-01-01 due to removal of Qt4 support and removal of #precompiler checks that differ between Qt4 and Qt5.
    "ito.AddIn.InterfaceBase/4.0.1", //outdated on 2020-03-31 due to atomic reference counting in ito::ByteArray
    "ito.AddIn.InterfaceBase/4.0.2", //outdated on 2020-09-15 due to additional property 'popupSlider' of 'ParamEditorWidget'
    "ito.AddIn.InterfaceBase/4.1.0", //outdated on 2020-12-14 due to new userMutex in AddInBase
    "ito.AddIn.InterfaceBase/4.2.0", //outdated on 2021-05-19 due to rework of ParamBase, Param, ParamMeta including the new StringList parameter type. Further changes in RetVal interface and bool operator of itom.dataObject.
    "ito.AddIn.InterfaceBase/5.0.0", //outdated on 2022-05-07 due to bugfix of DObjMeta class (see https://github.com/itom-project/itom/issues/187)
    NULL
};

//these defines exist since itom 1.3.1
#define CREATE_ADDININTERFACE_VERSION_STR(major,minor,patch) "ito.AddIn.InterfaceBase/"#major"."#minor"."#patch
#define CREATE_ADDININTERFACE_MAJOR_VERSION_STR(major) "ito.AddIn.InterfaceBase/"#major

//please indicate the major, minor and patch version in the following defines.
//Additionally put all three components of the version in the define ITOM_ADDININTERFACE_VERSION_STR
//and add the major version number only as argument of the macro in the last line.
#define ITOM_ADDININTERFACE_MAJOR 6
#define ITOM_ADDININTERFACE_MINOR 0
#define ITOM_ADDININTERFACE_PATCH 0
#define ITOM_ADDININTERFACE_VERSION CREATEVERSION(ITOM_ADDININTERFACE_MAJOR,ITOM_ADDININTERFACE_MINOR,ITOM_ADDININTERFACE_PATCH)
#define ITOM_ADDININTERFACE_VERSION_STR CREATE_ADDININTERFACE_VERSION_STR(6,0,0)
static constexpr const char* ito_AddInInterface_CurrentVersion = CREATE_ADDININTERFACE_MAJOR_VERSION_STR(6); //results in "ito.AddIn.InterfaceBase/x"; (the major version number 5 can not be replaced by the macros above. Does not work properly)

#endif
