/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef PYTHONNPDATAOBJECT_H
#define PYTHONNPDATAOBJECT_H

#include "pythonDataObject.h"

//using namespace ito;
namespace ito {

class PythonNpDataObject
{

public:
    //-------------------------------------------------------------------------------------------------
    // typedefs
    //------------------------------------------------------------------------------------------------- 
    typedef struct 
    {
        PyArrayObject numpyArray;
        PyObject* tags;             //PyDictObject
        PyObject* axisScales;       //PyListObject
        PyObject* axisOffsets;      //PyListObject
        PyObject* axisDescriptions; //PyListObject
        PyObject* axisUnits;        //PyListObject
        PyObject* valueUnit;        //PyUnicode
        PyObject* valueDescription; //PyUnicode
        double valueOffset;
        double valueScale;
    } 
    PyNpDataObject;

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //------------------------------------------------------------------------------------------------- 
    static void PyNpDataObject_dealloc(PyNpDataObject *self);
    static PyObject* PyNpDataObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyDataObject_init(PyNpDataObject *self, PyObject *args, PyObject *kwds);

    //uncomment the following methods for finding the bug described in comment at begin of pythonNpDataObject.cpp
    //static PyObject * __PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype);
    //static int __PyArray_SetBaseObject(PyArrayObject *arr, PyObject *obj);

    //-------------------------------------------------------------------------------------------------
    // numpy subclassing methods
    //------------------------------------------------------------------------------------------------- 
    static PyObject* PyNpDataObject_Array_Finalize(PyNpDataObject *self, PyObject *args);
    //static PyObject* PyNpDataObject_Array_Wrap(PyNPDataObject *self, PyObject *args);
    //static PyObject* PyNpDataObject_Array_Prepare(PyNPDataObject *self, PyObject *args);

    static PyObject* PyNpDataObj_Reduce(PyNpDataObject *self, PyObject *args); //PyObject *NPY_UNUSED(args));
    static PyObject* PyNpDataObj_SetState(PyNpDataObject *self, PyObject *args);
    
    //-------------------------------------------------------------------------------------------------
    // general members
    //------------------------------------------------------------------------------------------------- 
    static PyObject *PyNpDataObject_name(PyNpDataObject *self);
    
    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyNpDataObject_getTags(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setTags(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getAxisScales(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setAxisScales(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getAxisOffsets(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setAxisOffsets(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getAxisDescriptions(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setAxisDescriptions(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getAxisUnits(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setAxisUnits(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getValueUnit(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setValueUnit(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getValueDescription(PyNpDataObject *self, void *closure);
    static int PyNpDataObject_setValueDescription(PyNpDataObject *self, PyObject *value, void *closure);
    static PyObject* PyNpDataObject_getValueScale(PyNpDataObject *self, void *closure);
    static PyObject* PyNpDataObject_getValueOffset(PyNpDataObject *self, void *closure);

    static PyObject* PyNpDataObject_getTagDict(PyNpDataObject *self, void *clousure);


    //-------------------------------------------------------------------------------------------------
    // type structures
    //------------------------------------------------------------------------------------------------- 
    //static PyMemberDef PyNpDataObject_members[];
    static PyMethodDef PyNpDataObject_methods[];
    static PyTypeObject PyNpDataObjectType;
    static PyModuleDef PyNpDataObjectModule;
    static PyGetSetDef PyNpDataObject_getseters[];

   
};

}; // namespace ito

#endif