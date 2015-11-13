/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "pythonNumeric.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyAMax_doc,"amax(dataObject) -> returns the maximum value of the data object.");
/*static */PyObject* PythonNumeric::PyAMax(PyObject * /*pSelf*/, PyObject * /*pArgs*/)
{
    PyErr_SetString(PyExc_RuntimeError, "not yet implemented (method amax)");
    return NULL;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//          PYTHON MODULES - - - PYTHON TYPES - - - PYTHON MODULES                                              //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PyMethodDef PythonNumeric::PythonMethodItomNumeric[] = {
    // "Python name", C Ffunction Code, Argument Flags, __doc__ description
    {"amax", (PyCFunction)PythonNumeric::PyAMax, METH_VARARGS, pyAMax_doc},
    {NULL, NULL, 0, NULL}
};

PyModuleDef PythonNumeric::PythonModuleItomNumeric = {
    PyModuleDef_HEAD_INIT, "numeric", NULL, -1, PythonNumeric::PythonMethodItomNumeric,
    NULL, NULL, NULL, NULL
};

} //end namespace ito
