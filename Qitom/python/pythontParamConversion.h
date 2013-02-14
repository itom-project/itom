/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#ifndef PYTHONPARAMCONVERSION_H
#define PYTHONPARAMCONVERSION_H

//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#ifdef _DEBUG
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
    #else
        #include "Python.h"
    #endif
    #define _DEBUG
#else
#ifdef linux
    #include "Python.h"
#else
    #include "Python.h"
#endif
#endif

#include "../../common/sharedStructures.h"

#include "../global.h"

namespace ito
{

class PythonParamConversion
{
public:

    static PyObject *ParamBaseToPyObject(const ito::ParamBase &param);
    static SharedParamBasePointer PyObjectToParamBase(PyObject* obj, const char *name, ito::RetVal &retVal, int paramBaseType = 0, bool strict = true);

private:
    static void PyObjectToParamBaseDeleter(ito::ParamBase *param);


};

} //namespace ito


#endif