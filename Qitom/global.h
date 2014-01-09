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

#ifndef GLOBAL_H
#define GLOBAL_H

//!< must be defined before undef _DEBUG scheiß
//#include <vector>
//#include <map>
//#include <deque>
//#include <queue>
//#include <algorithm>

#include <qmutex.h>
#include <qobject.h>
#include "../common/sharedStructures.h"
//#include "organizer/addInManager.h"

#include <qmap.h>
#include <qlist.h>
#include <qstring.h>
#include <qsharedpointer.h>

/* definition and macros */
#define ITOM_VERSION_MAJOR  0x01
#define ITOM_VERSION_MINOR  0x00
#define ITOM_VERSION_PATCH  0x0E
#define ITOM_VERSION        CREATEVERSION(ITOM_VERSION_MAJOR,ITOM_VERSION_MINOR,ITOM_VERSION_PATCH) //ITOM_VERSION is (major << 16) + (minor << 8) + patch
#define ITOM_VERSION_STR    "1.0.14"

#ifdef USEPCL
    #define ITOM_POINTCLOUDLIBRARY 1
#else
    #define ITOM_POINTCLOUDLIBRARY 0
#endif
#define ITOM_PYTHONMATLAB 0

#define DELETE_AND_SET_NULL(pointer) if(pointer != NULL) { delete pointer; pointer = NULL;};
#define DELETE_AND_SET_NULL_ARRAY(pointer) if(pointer != NULL) { delete[] pointer; pointer = NULL;};

/* global variables (avoid) */
typedef QMap<QString,QString> StringMap;
typedef QList<int> IntList;
typedef QVector<int> IntVector;
typedef QSharedPointer<ito::Param> SharedParamPointer;
typedef QVector<ito::Param> ParamVector;
typedef QSharedPointer<ito::ParamBase> SharedParamBasePointer;
typedef QVector<SharedParamBasePointer> SharedParamBasePointerVector;
typedef QVector<ito::Param> ParamBaseVector;


#endif
