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

#ifndef PYTHONPROXY_H
#define PYTHONPROXY_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

namespace ito
{

class PythonProxy
{
public:
    typedef struct
    {
        PyObject_HEAD
        PyObject* inst;
        PyObject* func;
        PyObject* klass;
        PyObject* base;
    }
    PyProxy;

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyProxy_dealloc(PyProxy *self);
    static PyObject* PyProxy_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyProxy_init(PyProxy *self, PyObject *args, PyObject *kwds);


    static PyObject *PyProxy_richcompare(PyObject *v, PyObject *w, int op);
    static PyObject *PyProxy_call(PyProxy *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    static PyTypeObject PyProxyType;
    static PyModuleDef PyProxyModule;

    static void PyProxy_addTpDict(PyObject *tp_dict);
};

}; //end namespace ito


#endif
