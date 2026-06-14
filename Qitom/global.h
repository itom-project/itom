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

#ifndef GLOBAL_H
#define GLOBAL_H

//!< must be defined before undef _DEBUG
//#include <vector>
//#include <map>
//#include <deque>
//#include <queue>
//#include <algorithm>

#include "../common/param.h"
#include "common/sharedStructures.h"

#include <qmap.h>
#include <qlist.h>
#include <qvector.h>
#include <qstring.h>
#include <qsharedpointer.h>

/* definition and macros */
#define ITOM_VERSION_MAJOR  0x04
#define ITOM_VERSION_MINOR  0x02
#define ITOM_VERSION_PATCH  0x02
#define ITOM_VERSION        CREATEVERSION(ITOM_VERSION_MAJOR,ITOM_VERSION_MINOR,ITOM_VERSION_PATCH) //ITOM_VERSION is (major << 16) + (minor << 8) + patch
#define ITOM_VERSION_STR    "4.2.2"

#ifdef USEPCL
    #define ITOM_POINTCLOUDLIBRARY 1
#else
    #define ITOM_POINTCLOUDLIBRARY 0
#endif

#ifdef USEPYMATLAB
    #define ITOM_PYTHONMATLAB 1
#else
    #define ITOM_PYTHONMATLAB 0
#endif

#ifdef USEHELPVIEWER //Qt5 only
#if QT_VERSION >= 0x050600
    #define ITOM_USEHELPVIEWER 1
#endif
#endif

/* global variables (avoid) */
typedef QMap<QString,QString> StringMap;
typedef QList<int> IntList;
typedef QVector<int> IntVector;
typedef QSharedPointer<ito::Param> SharedParamPointer;
typedef QVector<ito::Param> ParamVector;
typedef QSharedPointer<ito::ParamBase> SharedParamBasePointer;
typedef QVector<SharedParamBasePointer> SharedParamBasePointerVector;
typedef QVector<ito::Param> ParamBaseVector;


namespace ito {

    /**
    * MsgType enumeration
    * This enum holds the possible values for any message type (for qDebugStream e.g.)
    *
    * This enumeration is deprecated and will be removed in future versions. It is currently unused. Don't use it again.
    */
    enum tMsgType
    {
        msgReturnInfo,
        msgReturnWarning,
        msgReturnError,
        msgTextInfo,
        msgTextWarning,
        msgTextError
    };


    enum tPythonDbgCmd
    {
        pyDbgNone=0,
        pyDbgContinue=1,
        pyDbgStep=2,
        pyDbgStepOut=4,
        pyDbgStepOver=8,
        pyDbgQuit=16
    };


    enum tPythonTransitions
    {
        pyTransBeginRun = 1,
        pyTransEndRun = 2,
        pyTransBeginDebug = 4,
        pyTransEndDebug = 8,
        pyTransDebugWaiting = 16,
        pyTransDebugContinue = 32,
        pyTransDebugExecCmdBegin = 64,
        pyTransDebugExecCmdEnd = 128
    };

    enum tPythonState
    {
        pyStateIdle = 1,
        pyStateRunning = 2,
        pyStateDebugging = 4,
        pyStateDebuggingWaiting = 8,
        pyStateDebuggingWaitingButBusy = 16
    };
}; //end namespace ito


#endif
