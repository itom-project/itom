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

#ifndef PYTHONSHAPE_H
#define PYTHONSHAPE_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
    #define NO_IMPORT_ARRAY

    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #include "Python.h" 
        #define _DEBUG
    #else
        #include "Python.h"   
    #endif
#endif

namespace ito
{

class Shape; //forward declaration

class PythonShape
{
public:
    typedef struct
    {
        PyObject_HEAD
        Shape *shape;
    }
    PyShape;

    #define PyShape_Check(op) PyObject_TypeCheck(op, &ito::PythonShape::PyShapeType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //------------------------------------------------------------------------------------------------- 
    static void PyShape_dealloc(PyShape *self);
    static PyObject* PyShape_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyShape_init(PyShape *self, PyObject *args, PyObject *kwds);

    static PyObject* createPyShape(const Shape &shape);

    //-------------------------------------------------------------------------------------------------
    // general members
    //------------------------------------------------------------------------------------------------- 
    static PyObject* PyShape_repr(PyShape *self);

    //-------------------------------------------------------------------------------------------------
    // pickling
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyShape_Reduce(PyShape *self, PyObject *args);
    static PyObject* PyShape_SetState(PyShape *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //------------------------------------------------------------------------------------------------- 
    static PyObject* PyShape_getType(PyShape *self, void *closure);

    static PyObject* PyShape_getFlags(PyShape *self, void *closure);
    static int PyShape_setFlags(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getTransform(PyShape *self, void *closure);
    static int PyShape_setTransform(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getArea(PyShape *self, void *closure);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //------------------------------------------------------------------------------------------------- 
    //static PyMemberDef PyShape_members[];
    static PyMethodDef PyShape_methods[];
    static PyGetSetDef PyShape_getseters[];
    static PyTypeObject PyShapeType;
    static PyModuleDef PyShapeModule;

    static void PyShape_addTpDict(PyObject *tp_dict);

    

};

}; //end namespace ito


#endif