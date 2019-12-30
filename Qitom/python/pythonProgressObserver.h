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

#ifndef PYTHONPROGRESSOBSERVER_H
#define PYTHONPROGRESSOBSERVER_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
    #define NO_IMPORT_ARRAY

    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #include "python/pythonWrapper.h" 
        #define _DEBUG
    #else
        #include "python/pythonWrapper.h"   
    #endif
#endif

#include "common/functionCancellationAndObserver.h"
#include <qsharedpointer.h>

namespace ito
{

class PythonProgressObserver
{
public:
    typedef struct
    {
        PyObject_HEAD
        QSharedPointer<ito::FunctionCancellationAndObserver> *progressObserver;
    }
    PyProgressObserver;

    #define PyProgressObserver_Check(op) PyObject_TypeCheck(op, &ito::PythonProgressObserver::PyProgressObserverType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //------------------------------------------------------------------------------------------------- 
    static void PyProgressObserver_dealloc(PyProgressObserver *self);
    static PyObject* PyProgressObserver_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyProgressObserver_init(PyProgressObserver *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // general members
    //------------------------------------------------------------------------------------------------- 
    static PyObject* PyProgressObserver_repr(PyProgressObserver *self);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //------------------------------------------------------------------------------------------------- 
    static PyObject* PyProgressObserver_getProgressMinimum(PyProgressObserver *self, void *closure);
    static int PyProgressObserver_setProgressMinimum(PyProgressObserver *self, PyObject *value, void *closure);

    static PyObject* PyProgressObserver_getProgressMaximum(PyProgressObserver *self, void *closure);
    static int PyProgressObserver_setProgressMaximum(PyProgressObserver *self, PyObject *value, void *closure);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //------------------------------------------------------------------------------------------------- 
    //static PyMemberDef PyProgressObserver_members[];
    static PyMethodDef PyProgressObserver_methods[];
    static PyGetSetDef PyProgressObserver_getseters[];
    static PyTypeObject PyProgressObserverType;
    static PyModuleDef PyProgressObserverModule;

    static void PyProgressObserver_addTpDict(PyObject *tp_dict);
};

}; //end namespace ito


#endif