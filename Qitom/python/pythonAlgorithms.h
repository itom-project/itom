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

/* includes */
#ifndef Q_MOC_RUN
    //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

#include "common/addInInterface.h"

#include <qstring.h>

namespace ito
{

class PythonAlgorithms
{

public:
    static PyMethodDef PythonMethodItomAlgorithms[];
    static PyModuleDef PythonModuleItomAlgorithms;

    static void AddAlgorithmFunctions(PyObject* mod);

    //!< calls an algorithm with a given name and the required args and kwds.
    static PyObject* PyGenericAlgorithm(const QString &algorithmName, PyObject *self, PyObject *args, PyObject *kwds);

    static PyObject* PyGenericAlgorithm2(const AddInAlgo::FilterDef *filterDef, PyObject *self, PyObject *args, PyObject *kwds);

private:

    //! PyAlgorithm is a thin wrapper class for calling any itom algorithm, defined in an algo plugin.
    /*
    For each algorithm, a PyAlgorithm object is created (in AddAlgorithmFunctions)
    and added as member to the itom.algorithms module. The PyAlgorithm class has
    the __call__ method implemented, that calls the real algorithm.

    Then, it is equal to call

    itom.filter("lowPassFilter", arg1, arg2, ...)

    or

    itom.algorithms.lowPassFilter(arg1, arg2, ...)

    However, the latter provides auto completion and calltips.
    */
    typedef struct
    {
        PyObject_HEAD
        const ito::AddInAlgo::FilterDef *filterDef;
    }
    PyAlgorithm;

    #define PyAlgorithm_Check(op) PyObject_TypeCheck(op, &ito::PythonAlgorithms::PyAlgorithmType)

    //-------------------------------------------------------------------------------------------------
    // constructor, destructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyAlgorithm_dealloc(PyAlgorithm *self);
    static PyObject* PyAlgorithm_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyAlgorithm_init(PyAlgorithm *self, PyObject *args, PyObject *kwds);
    static PyObject* PyAlgorithm_call(PyAlgorithm *self, PyObject *args, PyObject *kwds);
    static PyObject* PyAlgorithm_repr(PyAlgorithm *self);

    // creates a new PyAlgorithm object.
    static PyObject* createPyAlgorithm(const ito::AddInAlgo::FilterDef *filterDef);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    static PyTypeObject PyAlgorithmType;
    static PyModuleDef PyAlgorithmModule;
};

} //end namespace ito
