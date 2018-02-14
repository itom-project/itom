/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONMATLAB_H
#define PYTHONMATLAB_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (defined WIN32)
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #elif (defined __APPLE__) | (defined CMAKE)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
    #define _DEBUG
#else
    #if (defined linux)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #elif (defined __APPLE__)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
#endif
#include "structmember.h"

#include "../global.h"

#if ITOM_PYTHONMATLAB == 1
/* * *
 * Copyright 2010 Joakim Mller
 *
 * This file is part of pymatlab.
 * 
 * pymatlab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * pymatlab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with pymatlab.  If not, see <http://www.gnu.org/licenses/>.
 * * */

#include "common/retVal.h"
#include <qlibrary.h>
#include <engine.h>



namespace ito
{

class PythonMatlab
{

public:

    typedef struct 
    {
        PyObject_HEAD
        Engine *ep;
    } PyMatlabSessionObject;

    static PyObject * PyMatlabSessionObject_new(PyTypeObject *type, PyObject *args, PyObject * kwds);

    static int PyMatlabSessionObject_init(PyMatlabSessionObject *self, PyObject *args, PyObject *kwds);

    static void PyMatlabSessionObject_dealloc(PyMatlabSessionObject *self);

    static PyObject * PyMatlabSessionObject_run(PyMatlabSessionObject *self, PyObject *args);

    static PyObject * PyMatlabSessionObject_setValue(PyMatlabSessionObject *self, PyObject *args);

    static PyObject * PyMatlabSessionObject_setString(PyMatlabSessionObject *self, PyObject *args);

    static PyObject * PyMatlabSessionObject_getString(PyMatlabSessionObject *self, PyObject *args);

    static PyObject * PyMatlabSessionObject_GetValue(PyMatlabSessionObject * self, PyObject *args);

    static PyObject * PyMatlabSessionObject_close(PyMatlabSessionObject * self, PyObject *args);


    static PyMethodDef PyMatlabSessionObject_methods[];
    static PyMemberDef PyMatlabSessionObject_members[]; // = { { NULL } };
    static PyTypeObject PyMatlabSessionObjectType;
    static PyModuleDef PyMatlabSessionObject_Module;
    static PyObject* PyInit_matlab(void);

    static bool initialized;
    static ito::RetVal loadLibrary();

    static QLibrary engineLibrary;
    static QLibrary mxLibrary;

}; //end class PythonMatlab

} //end namespace ito


#endif

#endif