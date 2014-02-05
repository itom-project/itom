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

#ifndef SHAREDSTRUCTURES_H
#define SHAREDSTRUCTURES_H

/* includes */
#include "typeDefs.h"
#include "retVal.h"
#include "param.h"

/* definition and macros */
/* global variables (avoid) */
/* content */

namespace ito
{
    #define CREATEVERSION(major,minor,patch)    (major << 16) + (minor << 8) + patch
    #define MAJORVERSION(version)               version >> 16
    #define MINORVERSION(version)               (version >> 8) - (MAJORVERSION(version) << 8)
    #define PATCHVERSION(version)               version - ((version >> 8) << 8)
    #define MAXVERSION                          CREATEVERSION(999999,0,0)    //maximum possible version (that means no maximum version is indicated)
    #define MINVERSION                          CREATEVERSION(0,0,0)         //minimum possible version

    #define ItomDoc_VAR(name) static char name[]
    #define ItomDoc_STRVAR(name,str) ItomDoc_VAR(name) = str

} //end namespace ito

#endif
