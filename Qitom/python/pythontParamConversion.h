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

#pragma once

#ifndef Q_MOC_RUN
// python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (defined WIN32)
#undef _DEBUG
#include "python/pythonWrapper.h"
#define _DEBUG
#else
#include "python/pythonWrapper.h"
#endif
#endif

#include "../../common/sharedStructures.h"
#include "../global.h"

namespace ito
{

/*!
    \class PythonParamConversion
    \brief Static methods to convert between Python objects and ito::ParamBase.
*/
class PythonParamConversion
{
  public:
    // converts ito::ParamBase to the most appropriate PyObject
    static PyObject *ParamBaseToPyObject(const ito::ParamBase &param);

    // converts a given PyObject to an appropriate ito::ParamBase
    static SharedParamBasePointer PyObjectToParamBase(PyObject *obj, const char *name, ito::RetVal &retVal,
                                                      int paramBaseType = 0, bool strict = true);

  private:
    // special deleter for param, where the wrapped object is deleted, too.
    static void PyObjectToParamBaseDeleter(ito::ParamBase *param);
};

} // namespace ito
