/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#ifndef __TYPEDEFS
#define __TYPEDEFS

#include <stdint.h>
#include <complex>

// WARNING it is very EVIL to include ANY QT STUFF here!!!

namespace ito
{
    #define PLUGINWAIT 5000

    

    /**
    * LogLevel enumeration
    * This enum holds all possible LogLevel values
    */
    enum tLogLevel
    {
        logNone     = 0x0,
        logError    = 0x1,
        logWarning  = 0x2,
        logInfo     = 0x4,
        logAll      = logInfo | logWarning | logError
    };


    /**
    * RetValue enumeration
    * This enum holds the three possible return states Ok, Warning and Error
    */
    enum tRetValue
    {
        retOk       = 0x0,  /*!< ok */ 
        retWarning  = 0x1,  /*!< warning */ 
        retError    = 0x2   /*!< error */ 
    };

    /**
    * MsgType enumeration
    * This enum holds the possible values for any message type (for qDebugStream e.g.)
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

    enum tCompareResult 
    { 
        tCmpEqual, 
        tCmpCompatible, 
        tCmpFailed 
    };

    enum tPythonState
    {
        pyStateIdle = 1,
        pyStateRunning = 2,
        pyStateDebugging = 4,
        pyStateDebuggingWaiting = 8,
		pyStateDebuggingWaitingButBusy = 16
    };

    /**
    * DataType enumeration
    * This enum holds the possible values for DataObject matrices.
    */
    enum tDataType
    {
        tInt8 = 0,      /*!< integer, 8bit */ 
        tUInt8 = 1,     /*!< unsigned integer, 8bit */ 
        tInt16 = 2,     /*!< integer, 16bit */ 
        tUInt16 = 3,    /*!< unsigned integer, 16bit */ 
        tInt32 = 4,     /*!< integer, 32bit */ 
        tUInt32 = 5,    /*!< unsigned integer, 32bit (not fully supported) */ 
        tFloat32 = 6,   /*!< float, 32bit */ 
        tFloat64 = 7,   /*!< double (64bit) */ 
        tComplex64 = 8, /*!< complex value with real and imaginary part of type float32 */ 
        tComplex128 = 9 /*!< complex value with real and imaginary part of type float64 */ 
    };

    /**
    * PCLPointType enumeration
    * This enum holds the possible values for point types supported by the wrapper of the point-cloud library in itom.
    */
    enum tPCLPointType
    {
        pclInvalid      = 0x0000, /*!< invalid point */
        pclXYZ          = 0x0001, /*!< point with x,y,z-value */
        pclXYZI         = 0x0002, /*!< point with x,y,z and intensity value */
        pclXYZRGBA      = 0x0004, /*!< point with x,y,z and r,g,b,a and curvature value */
        pclXYZNormal    = 0x0008, /*!< point with x,y,z value and its normal vector nx,ny,nz */
        pclXYZINormal   = 0x0010, /*!< point with the same values than pclXYZNormal and an additional intensity value */
        pclXYZRGBNormal = 0x0020  /*!< point with x,y,z and r,g,b and normal vector */
    };  

    // data types for images should always be the same size
    // so define them to fixed byte sizes here

   // data types for images should always be the same size
    // so define them to fixed byte sizes here


    /*< \todo #define bool bool */
    typedef int8_t  int8;  //__int8
    typedef int16_t int16; //__int16
    typedef int32_t int32; //__int32

#ifdef _WIN64
    //typedef int64_t int64;
    //typedef uint64_t uint64;
#endif

    //#define int int32 //commented by M. Gronle, 10.10.2011, since this caused problems while compiling with gcc and qtCreator
    //#define uint uint32   // impossible to define this, as in qglobal uint is also defined which causes problems

    typedef uint8_t uint8; //unsigned __int8
    typedef uint16_t uint16; //unsigned __int16
    typedef uint32_t uint32; //unsigned __int32

    typedef float float32;
    typedef double float64;

    //#define complex std::complex<double>
    typedef std::complex<ito::float32> complex64;
    typedef std::complex<ito::float64> complex128;


    #define GLOBAL_LOG_LEVEL tLogLevel(logAll)

#ifdef linux
    #define _strdup strdup
    #define _itoa itoa
    #define _snprintf snprintf
    #define Sleep(TIME) usleep(TIME*1000.0)
#endif

// this will be set on Visual Studio only, so this code is added for all other compilers
#ifndef _MSC_VER
    #define vsprintf_s(b,l,f,v) vsprintf(b,f,v);
    #define sprintf_s(b,l,f,v) sprintf(b,f,v);
    #define strcat_s(dest,len,source) strcat(dest,source);
#endif

    

} // namespace ito

#endif
