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

#include "pythonEngineInc.h"
#include "pythonNpDataObject.h"
#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "structmember.h"
#endif

//------------------------------------------------------------------------------------------------------


// in npDataObject, there is a crash for numpy 1.7 if 
// //#define NPY_NO_DEPRECATED_API 0x00000007 //see comment in pythonNpDataObject.cpp
// is set. This is due to the following fact:
//
//  In type PyNpDataObject, the first memory block belongs to PyArrayObject.
//  In previous versions of numpy, PyArrayObject contained lots of members for the
//  numpy array. In the new version, this is only reduced to PyObject_HEAD and the
//  members are moved to PyArrayObject_fields. Nevertheless, numpy casts a lot of 
//  PyArrayObject-pointers to PyArrayObject_fields. This can be done, if PyArrayObject
//  is not subclassed, since then, an access to the member base of PyArrayObject_fields
//  points to the same memory than for instance unitValue. Right now, I (Marc Gronle)
//  don't know how to avoid this, therefore NPY_NO_DEPRECATED_API must not be defined.
//
//  In order to test the problem, you can uncomment the methods__PyArray_View and__PyArray_SetBaseObject
//  in this class and replace the call of PyArray_View by __PyArray_View. There finally is an
//  error in __PyArray_SetBaseObject.
//
//  futher references are:
//     see: http://osdir.com/ml/python-numeric-general/2012-03/msg00384.html
//          https://github.com/numpy/numpy/issues/2980
//
//  other subclasses of numpy can be found in:
//     http://trac.mcs.anl.gov/projects/ITAPS/browser/python/trunk/iMesh_array.inl?rev=3831

//PyNPDataObject
using namespace ito;

void PythonNpDataObject::PyNpDataObject_dealloc(PyNpDataObject* self)
{
    Py_XDECREF(self->tags);
    Py_XDECREF(self->axisScales);
    Py_XDECREF(self->axisOffsets);
    Py_XDECREF(self->axisDescriptions);
    Py_XDECREF(self->axisUnits);
    Py_XDECREF(self->valueUnit);
    Py_XDECREF(self->valueDescription);

    /*destructor blub = PyArray_Type.tp_dealloc;
    blub( (PyObject*) self );
    int i=1;
    (&PyArray_Type)->tp_dealloc((PyObject*)self);*/
    //self->base.ob_base.ob_type->tp_free( (PyObject*)self );
    Py_TYPE(&(self->numpyArray))->tp_free( (PyObject*)self );
};

PyDoc_STRVAR(npDataObject_tags_doc,             "tag dictionary for this data object.");
PyDoc_STRVAR(npDataObject_axisScales_doc,       "list with scale values for each axis [unit/px].");
PyDoc_STRVAR(npDataObject_axisOffsets_doc,      "list with offset values for each axis [px].");
PyDoc_STRVAR(npDataObject_axisDescriptions_doc, "list with descriptions for each axis.");
PyDoc_STRVAR(npDataObject_axisUnits_doc,        "list containing the units for each axis.");
PyDoc_STRVAR(npDataObject_valueUnit_doc,        "value unit string.");
PyDoc_STRVAR(npDataObject_valueDescription_doc, "value unit description.");
PyDoc_STRVAR(npDataObject_valueScale_doc,       "value scale [default: 1.0].");
PyDoc_STRVAR(npDataObject_valueOffset_doc,      "value offset [default: 0.0].");
PyDoc_STRVAR(npDataObject_getTagDict_doc,       "returns new dictionary with all tags inside");


//PyMemberDef PythonNpDataObject::PyNpDataObject_members[] = {
//        {"tags",             T_OBJECT_EX, offsetof(PyNpDataObject, tags)             , 0, npDataObject_tags             },
//        {"axisScales",       T_OBJECT_EX, offsetof(PyNpDataObject, axisScales)       , 0, npDataObject_axisScales       },
//        {"axisOffsets",      T_OBJECT_EX, offsetof(PyNpDataObject, axisOffsets)      , 0, npDataObject_axisOffsets      },
//        {"axisDescriptions", T_OBJECT_EX, offsetof(PyNpDataObject, axisDescriptions) , 0, npDataObject_axisDescriptions },
//        {"axisUnits",        T_OBJECT_EX, offsetof(PyNpDataObject, axisUnits)        , 0, npDataObject_axisUnits        },
//        {"valueUnit",        T_OBJECT_EX, offsetof(PyNpDataObject, valueUnit)        , 0, npDataObject_valueUnit        },
//        {"valueDescription", T_OBJECT_EX, offsetof(PyNpDataObject, valueDescription) , 0, npDataObject_valueDescription },
//        {"valueScale",       T_DOUBLE,    offsetof(PyNpDataObject, valueScale)       , READONLY, npDataObject_valueScale},
//        {"valueOffset",      T_DOUBLE,    offsetof(PyNpDataObject, valueOffset)      , READONLY, npDataObject_valueOffset},
//        {NULL}  /* Sentinel */
//    };

PyGetSetDef PythonNpDataObject::PyNpDataObject_getseters[] = {
    {"tags", (getter)PyNpDataObject_getTags, (setter)PyNpDataObject_setTags, npDataObject_tags_doc, NULL},
    {"axisScales", (getter)PyNpDataObject_getAxisScales, (setter)PyNpDataObject_setAxisScales, npDataObject_axisScales_doc, NULL},
    {"axisOffsets", (getter)PyNpDataObject_getAxisOffsets, (setter)PyNpDataObject_setAxisOffsets, npDataObject_axisOffsets_doc, NULL},
    {"axisDescriptions", (getter)PyNpDataObject_getAxisDescriptions, (setter)PyNpDataObject_setAxisDescriptions, npDataObject_axisDescriptions_doc, NULL},
    {"axisUnits", (getter)PyNpDataObject_getAxisUnits, (setter)PyNpDataObject_setAxisUnits, npDataObject_axisUnits_doc, NULL},
    {"valueUnit", (getter)PyNpDataObject_getValueUnit, (setter)PyNpDataObject_setValueUnit, npDataObject_valueUnit_doc, NULL},
    {"valueDescription", (getter)PyNpDataObject_getValueDescription, (setter)PyNpDataObject_setValueDescription, npDataObject_valueDescription_doc, NULL},
    {"valueScale", (getter)PyNpDataObject_getValueScale, NULL, npDataObject_valueScale_doc, NULL},
    {"valueOffset", (getter)PyNpDataObject_getValueOffset, NULL, npDataObject_valueOffset_doc, NULL},
    {"metaDict", (getter)PyNpDataObject_getTagDict, NULL, npDataObject_getTagDict_doc, NULL},
    {NULL}  /* Sentinel */
};

PyObject* PythonNpDataObject::PyNpDataObject_name(PyNpDataObject* /*self*/)
{
    return PyUnicode_FromString("npDataObject");
};



PyMethodDef PythonNpDataObject::PyNpDataObject_methods[] = {
        {"name", (PyCFunction)PyNpDataObject_name, METH_NOARGS, "name"},
        {"__array_finalize__", (PyCFunction)PythonNpDataObject::PyNpDataObject_Array_Finalize, METH_VARARGS, "__array_finalize__"},
        /*{"__array_wrap__", (PyCFunction)PythonDataObject::PyNPDataObject_Array_Wrap, METH_VARARGS, "__array_wrap__"},*/
        /*{"__array_prepare__", (PyCFunction)PythonDataObject::PyNPDataObject_Array_Prepare, METH_VARARGS, "__array_prepare__"},*/
        {"__reduce__", (PyCFunction)PythonNpDataObject::PyNpDataObj_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
        {"__setstate__", (PyCFunction)PythonNpDataObject::PyNpDataObj_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
        {NULL}  /* Sentinel */
    };

PyModuleDef PythonNpDataObject::PyNpDataObjectModule = {
        PyModuleDef_HEAD_INIT, "npDataObject", "Numpy compatible DataObject type in python", -1,
        NULL, NULL, NULL, NULL, NULL
    };

PyTypeObject PythonNpDataObject::PyNpDataObjectType = {
        PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
        "itom.npDataObject",             /* tp_name */
        sizeof(PyNpDataObject),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyNpDataObject_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        "npDataObject objects",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyNpDataObject_methods,             /* tp_methods */
        0, /*PyNpDataObject_members,*/             /* tp_members */
        PyNpDataObject_getseters,                         /* tp_getset */
        0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        0,                       /* tp_init */
        0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
        PyNpDataObject_new         /* tp_new */
    };


PyObject * PythonNpDataObject::PyNpDataObject_new(PyTypeObject *type, PyObject *args, PyObject * /*kwds*/)
{
    PyObject *obj = NULL;
    PyObject *arr = NULL;
    PyNpDataObject *self = NULL;
    PyObject *item = NULL;

    //argument must be a PyObject
    if(!PyArg_ParseTuple(args, "O", &obj)) //obj is borrowed reference
    {
        PyErr_SetString(PyExc_ValueError,"Argument of npDataObject must be of type ndarray or dataObject and not empty");
        return NULL;
    }

    if(PyArray_Check(obj))
    {
        arr = PyArray_FROM_O(obj);

        if(arr == NULL)
        {
            Py_XDECREF(arr);
            PyErr_SetString(PyExc_ValueError, "ndarray argument is invalid");
            return NULL;
        }

        //self = (PyNpDataObject*)__PyArray_View((PyArrayObject*)arr, NULL, type);
        self = (PyNpDataObject*)PyArray_View((PyArrayObject*)arr, NULL, type);
 
        Py_DECREF(arr);

        int dims = PyArray_NDIM(&(self->numpyArray)); //.nd;
        if(dims == 1) dims = 2;
        self->tags = PyDict_New();
        self->axisScales = PyList_New(dims);
        for(int i=0;i<dims;i++) PyList_SetItem(self->axisScales, i, PyFloat_FromDouble(1.0));
        self->axisOffsets = PyList_New(dims);
        for(int i=0;i<dims;i++) PyList_SetItem(self->axisOffsets, i, PyFloat_FromDouble(0.0));
        self->axisDescriptions = PyList_New(dims);
        for(int i=0;i<dims;i++) PyList_SetItem(self->axisDescriptions, i, PyUnicode_FromStringAndSize("",0));
        self->axisUnits = PyList_New(dims);
        for(int i=0;i<dims;i++) PyList_SetItem(self->axisUnits, i, PyUnicode_FromStringAndSize("px",2));
        self->valueUnit = PyUnicode_FromStringAndSize("mm",2);
        self->valueDescription = PyUnicode_FromStringAndSize("",0);
        self->valueOffset = 0.0;
        self->valueScale = 1.0;
    }
    else if(PyDataObject_Check(obj))
    {
        arr = PyArray_FromStructInterface(obj);  //!< calls PyDataObj_Array_StructGet of PyDataObject and increments its refcount (If the object does not contain this method then a borrowed reference to Py_NotImplemented is returned.)
        //Py_XDECREF(obj);

        if(arr == Py_NotImplemented)
        {
            //Py_XDECREF(arr);
            PyErr_SetString(PyExc_ValueError, "argument returned Py_NotImplemented (e.g. empty dataObject is not allowed)");
            return NULL;
        }
        else
        {
            self = (PyNpDataObject*)PyArray_View((PyArrayObject*)arr, NULL, type);
            //self = (PyNpDataObject*)__PyArray_View((PyArrayObject*)arr, NULL, type);

            //copy tags from obj to self
            //0. check-value (here: 21120)
            //1. PyObject* tags (dict)
            //2. PyObject* axisScales (list)
            //3. PyObject* axisOffsets (list)
            //4. PyObject* axisDescriptions (list)
            //5. PyObject* axisUnits (list)
            //6. PyObject* valueUnit (unicode)
            //7. PyObject* valueDescription (unicode)
            //8. double valueOffset
            //9. double valueScale

            DataObject *dObj = ((PythonDataObject::PyDataObject*)obj)->dataObject;
            int tagSize = dObj->getTagListSize();
            int dims = dObj->getDims();
            //std::string tempString;
            DataObjectTagType tempTag;
            std::string tempKey;
            std::string tempString;
            bool validOp;

            //1. tags
            self->tags = PyDict_New();
            for(int i=0;i<tagSize;i++)
            {
                //tempKey = dObj->getTagKey(i,validOp);
                //if(validOp)
                //{
                    //tempString = dObj->getTag(tempKey, validOp);
                    //if(validOp) PyDict_SetItem(self->tags, PyUnicode_FromString(tempKey.data()), PyUnicode_FromString(tempString.data()));
                //}
                dObj->getTagByIndex(i, tempKey, tempTag);
                if(tempTag.getType() == DataObjectTagType::typeDouble)
                {
                    item =  PyFloat_FromDouble(tempTag.getVal_ToDouble());
                    PyDict_SetItemString(self->tags, tempKey.data(), item);
                    Py_DECREF(item);
                }
                else
                {
                    //item =  PyUnicode_FromString(tempTag.getVal_ToString().data());
                    tempString = tempTag.getVal_ToString().data();
                    item =  PyUnicode_DecodeLatin1(tempString.data(), tempString.length(), NULL);
                    PyDict_SetItemString(self->tags, tempKey.data(), item);
                    Py_DECREF(item);
                }
            }

            //2. axisScales
            self->axisScales = PyList_New(dims);
            for(int i=0;i<dims;i++)
            {
                PyList_SetItem(self->axisScales, i, PyFloat_FromDouble(dObj->getAxisScale(i))); //steals ref to value
            }

            //3. axisOffsets
            self->axisOffsets = PyList_New(dims);
            for(int i=0;i<dims;i++)
            {
                PyList_SetItem(self->axisOffsets, i, PyFloat_FromDouble(dObj->getAxisOffset(i))); //steals ref to value
            }

            //4. axisDescriptions
            self->axisDescriptions = PyList_New(dims);
            for(int i=0;i<dims;i++)
            {
                tempString = dObj->getAxisDescription(i,validOp);
                PyList_SetItem(self->axisDescriptions, i, PyUnicode_DecodeLatin1(tempString.data(), tempString.length(), NULL)); //steals ref to value
            }

            //5. axisUnits
            self->axisUnits = PyList_New(dims);
            for(int i=0;i<dims;i++)
            {
                tempString = dObj->getAxisUnit(i,validOp);
                PyList_SetItem(self->axisUnits, i, PyUnicode_DecodeLatin1(tempString.data(), tempString.length(), NULL)); //steals ref to value
            }

            //6. valueUnit
            //self->valueUnit = PyUnicode_FromString(dObj->getValueUnit().data());
            tempString = dObj->getValueUnit();
            self->valueUnit = PyUnicode_DecodeLatin1(tempString.data(), tempString.length(), NULL);

            //7. valueDescription
            //self->valueDescription = PyUnicode_FromString(dObj->getValueDescription().data());
            tempString = dObj->getValueDescription();
            self->valueDescription = PyUnicode_DecodeLatin1(tempString.data(), tempString.length(), NULL);

            //8.
            self->valueOffset = dObj->getValueOffset();
            //9.
            self->valueScale  = dObj->getValueScale();
        }
        Py_DECREF(arr);
    }
    else
    {
        //Py_XDECREF(obj);
        PyErr_SetString(PyExc_ValueError, "Argument of npDataObject must be of type ndarray or dataObject");
        return NULL;
    }


    return (PyObject*)self;
}


//this method is taken from source code of numpy 1.7.0
//PyObject * PythonNpDataObject::__PyArray_View(PyArrayObject *self, PyArray_Descr *type, PyTypeObject *pytype)
//{
//    PyArrayObject *ret = NULL;
//    PyArray_Descr *dtype;
//    PyTypeObject *subtype;
//    int flags;
//
//    if (pytype) {
//        subtype = pytype;
//    }
//    else {
//        subtype = Py_TYPE(self);
//    }
//
//    flags = PyArray_FLAGS(self);
//
//    dtype = PyArray_DESCR(self);
//    Py_INCREF(dtype);
//    ret = (PyArrayObject *)PyArray_NewFromDescr(subtype,
//                               dtype,
//                               PyArray_NDIM(self), PyArray_DIMS(self),
//                               PyArray_STRIDES(self),
//                               PyArray_DATA(self),
//                               flags,
//                               (PyObject *)self);
//    if (ret == NULL) {
//        return NULL;
//    }
//
//    /* Set the base object */
//    Py_INCREF(self);
//    if (__PyArray_SetBaseObject(ret, (PyObject *)self) < 0) {
//        Py_DECREF(ret);
//        Py_DECREF(type);
//        return NULL;
//    }
//
//    if (type != NULL) {
//        if (PyObject_SetAttrString((PyObject *)ret, "dtype",
//                                   (PyObject *)type) < 0) {
//            Py_DECREF(ret);
//            Py_DECREF(type);
//            return NULL;
//        }
//        Py_DECREF(type);
//    }
//    return (PyObject *)ret;
//}
//
//
//
//int PythonNpDataObject::__PyArray_SetBaseObject(PyArrayObject *arr, PyObject *obj)
//{
//    if (obj == NULL) {
//        PyErr_SetString(PyExc_ValueError,
//                "Cannot set the NumPy array 'base' "
//                "dependency to NULL after initialization");
//        return -1;
//    }
//    /*
//     * Allow the base to be set only once. Once the object which
//     * owns the data is set, it doesn't make sense to change it.
//     */
//    if (PyArray_BASE(arr) != NULL) {
//        Py_DECREF(obj);
//        PyErr_SetString(PyExc_ValueError,
//                "Cannot set the NumPy array 'base' "
//                "dependency more than once");
//        return -1;
//    }
//
//    /*
//     * Don't allow infinite chains of views, always set the base
//     * to the first owner of the data.  
//     * That is, either the first object which isn't an array, 
//     * or the first object which owns its own data.
//     */
//
//    while (PyArray_Check(obj) && (PyObject *)arr != obj) {
//        PyArrayObject *obj_arr = (PyArrayObject *)obj;
//        PyObject *tmp;
//
//        /* Propagate WARN_ON_WRITE through views. */
//        if (PyArray_FLAGS(obj_arr) & (1 << 31) /*NPY_ARRAY_WARN_ON_WRITE*/) {
//            PyArray_ENABLEFLAGS(arr, (1 << 31) /*NPY_ARRAY_WARN_ON_WRITE*/);
//        }   
//
//        /* If this array owns its own data, stop collapsing */
//        if (PyArray_CHKFLAGS(obj_arr, NPY_ARRAY_OWNDATA)) {
//            break;
//        }   
//
//        tmp = PyArray_BASE(obj_arr);
//        /* If there's no base, stop collapsing */
//        if (tmp == NULL) {
//            break;
//        }
//        /* Stop the collapse new base when the would not be of the same 
//         * type (i.e. different subclass).
//         */
//        if (Py_TYPE(tmp) != Py_TYPE(arr)) {
//            break;
//        }
//
//
//        Py_INCREF(tmp);
//        Py_DECREF(obj);
//        obj = tmp;
//    }
//
//    /* Disallow circular references */
//    if ((PyObject *)arr == obj) {
//        Py_DECREF(obj);
//        PyErr_SetString(PyExc_ValueError,
//                "Cannot create a circular NumPy array 'base' dependency");
//        return -1;
//    }
//
//    ((PyArrayObject_fields *)arr)->base = obj;
//
//    return 0;
//}




PyObject* PythonNpDataObject::PyNpDataObject_Array_Finalize(PyNpDataObject *self, PyObject *args)
{
    PyObject* obj = NULL;
    PyNpDataObject* obj_cast = NULL;

    PyArg_ParseTuple(args, "O", &obj);

    if(obj && Py_TYPE(obj) == &PyNpDataObjectType)
    {
        obj_cast = (PyNpDataObject*)obj;

        Py_INCREF(obj_cast->tags);
        Py_INCREF(obj_cast->axisScales);
        Py_INCREF(obj_cast->axisOffsets);
        Py_INCREF(obj_cast->axisDescriptions);
        Py_INCREF(obj_cast->axisUnits);
        Py_INCREF(obj_cast->valueUnit);
        Py_INCREF(obj_cast->valueDescription);
        self->tags = obj_cast->tags;
        self->axisScales = obj_cast->axisScales;
        self->axisOffsets = obj_cast->axisOffsets;
        self->axisDescriptions = obj_cast->axisDescriptions;
        self->axisUnits = obj_cast->axisUnits;
        self->valueUnit = obj_cast->valueUnit;
        self->valueDescription = obj_cast->valueDescription;
        self->valueOffset = obj_cast->valueOffset;
        self->valueScale = obj_cast->valueScale;
    }
    else if(obj && Py_TYPE(obj) == &PyArray_Type)
    {
        int dims = PyArray_NDIM((PyArrayObject*)obj); // ((PyArrayObject*)obj)->nd;
        if(dims == 1) dims = 2;
        self->tags = PyDict_New();
        self->axisScales = PyList_New(dims);
        for(Py_ssize_t i=0;i<dims;i++) PyList_SetItem(self->axisScales, i, PyFloat_FromDouble(1.0));
        self->axisOffsets = PyList_New(dims);
        for(Py_ssize_t i=0;i<dims;i++) PyList_SetItem(self->axisOffsets, i, PyFloat_FromDouble(0.0));
        self->axisDescriptions = PyList_New(dims);
        for(Py_ssize_t i=0;i<dims;i++) PyList_SetItem(self->axisDescriptions, i, PyUnicode_FromStringAndSize("",0));
        self->axisUnits = PyList_New(dims);
        for(Py_ssize_t i=0;i<dims;i++) PyList_SetItem(self->axisUnits, i, PyUnicode_FromStringAndSize("px",2));
        self->valueUnit = PyUnicode_FromStringAndSize("mm",2);
        self->valueDescription = PyUnicode_FromStringAndSize("",0);
        self->valueOffset = 0.0;
        self->valueScale = 1.0;
    }
    else
    {
        self->tags = PyDict_New();
        self->axisScales = PyList_New(0);
        self->axisOffsets = PyList_New(0);
        self->axisDescriptions = PyList_New(0);
        self->axisUnits = PyList_New(0);
        self->valueUnit = PyUnicode_FromStringAndSize("mm",2);
        self->valueDescription = PyUnicode_FromStringAndSize("",0);
        self->valueOffset = 0.0;
        self->valueScale = 1.0;
    }

    Py_RETURN_NONE;
}

//PyObject* PythonNpDataObject::PyNPDataObject_Array_Prepare(PyNpDataObject *self, PyObject *args)
//{
//    PyObject* obj = NULL;
//
//    int i=PyTuple_Size(args);
//
//    //PyArg_ParseTuple(args, "O", &obj);
//    return (PyObject*)self;
//}

//PyObject* PythonNpDataObject::PyNPDataObject_Array_Wrap(PyNpDataObject *self, PyObject *args)
//{
//    int j=PyTuple_Size(args);
//}


PyObject* PythonNpDataObject::PyNpDataObj_Reduce(PyNpDataObject *self, PyObject * /*args*/)
{
    PyObject *mod = NULL; //borrowed
    PyObject *ndarray = NULL; //new
    PyObject *baseDump = NULL; //new
    PyObject *baseState = NULL; //borrowed
    PyObject *selfState = NULL; //new
    PyObject *state = NULL; //new

    //load numpy module
    mod = PyImport_AddModule("numpy"); //borrowed ref
    if( mod == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "the numpy module cannot be loaded. Please make sure, that numpy is properly installed");
        return NULL;
    }

    ndarray = PyObject_GetAttrString(mod, "ndarray"); //new ref
    if(ndarray == NULL)
    {
        Py_XDECREF(ndarray);
        return NULL;
    }

    baseDump = PyObject_CallMethodObjArgs( ndarray , PyUnicode_FromString("__reduce__"), (PyObject*)self, NULL); //new ref

    //baseDump is tuple. 1.elem -> constructor method of numpy, 2. elem -> arguments for constructor, 3. elem -> content-tuple, 4. and 5. elem -> unknown, only handled by base-type (ndarray)
    baseState = PyTuple_GetItem(baseDump, 2); //borrowed

    selfState = PyTuple_New(10); //new ref


    /*
    selfState contains of:
    0. check-value (here: 21120)
    1. PyObject* tags (dict)
    2. PyObject* axisScales (list)
    3. PyObject* axisOffsets (list)
    4. PyObject* axisDescriptions (list)
    5. PyObject* axisUnits (list)
    6. PyObject* valueUnit (unicode)
    7. PyObject* valueDescription (unicode)
    8. double valueOffset
    9. double valueScale
    */

    PyTuple_SetItem(selfState, 0, Py_BuildValue("i",21120));
    //PyTuple_SetItem(selfState, 1, PyDict_Copy(self->tags));

    //PyObject *axisScaleCopy = PyList_New(PyList_Size(self->axisScales)); //new
    //for(Py_ssize_t i = 0 ; i < PyList_Size(self->axisScales) ; i++) PyList_SetItem(axisScaleCopy, i, PyList_GetItem(self->axisScales, i));
    //PyTuple_SetItem(selfState, 2, axisScaleCopy); //vielleicht auch durch Py_INCREF auf org-list möglich copy zu vermeiden

    //PyObject *axisOffsetCopy = PyList_New(PyList_Size(self->axisOffsets)); //new
    //for(Py_ssize_t i = 0 ; i < PyList_Size(self->axisOffsets) ; i++) PyList_SetItem(axisOffsetCopy, i, PyList_GetItem(self->axisOffsets, i));
    //PyTuple_SetItem(selfState, 3, axisOffsetCopy); //steals ref of axisOffsetCopy

    //PyObject *axisDescriptionsCopy = PyList_New(PyList_Size(self->axisDescriptions)); //new
    //for(Py_ssize_t i = 0 ; i < PyList_Size(self->axisDescriptions) ; i++) PyList_SetItem(axisDescriptionsCopy, i, PyList_GetItem(self->axisDescriptions, i));
    //PyTuple_SetItem(selfState, 4, axisDescriptionsCopy); //steals ref of axisDescriptionsCopy

    //PyObject *axisUnitsCopy = PyList_New(PyList_Size(self->axisUnits)); //new
    //for(Py_ssize_t i = 0 ; i < PyList_Size(self->axisUnits) ; i++) PyList_SetItem(axisUnitsCopy, i, PyList_GetItem(self->axisUnits, i));
    //PyTuple_SetItem(selfState, 5, axisUnitsCopy); //steals ref of axisUnitsCopy


    Py_INCREF(self->tags);
    Py_INCREF(self->axisScales);
    Py_INCREF(self->axisOffsets);
    Py_INCREF(self->axisDescriptions);
    Py_INCREF(self->axisUnits);
    Py_INCREF(self->valueDescription);
    Py_INCREF(self->valueUnit);

    PyTuple_SetItem(selfState, 1, self->tags); //steals ref of self->tags
    PyTuple_SetItem(selfState, 2, self->axisScales); //steals ref of self->axisScales
    PyTuple_SetItem(selfState, 3, self->axisOffsets); //steals ref of self->axisOffsets
    PyTuple_SetItem(selfState, 4, self->axisDescriptions); //steals ref of self->axisDescriptions
    PyTuple_SetItem(selfState, 5, self->axisUnits); //steals ref of self->axisUnits


    PyTuple_SetItem(selfState, 6, self->valueUnit);

    PyTuple_SetItem(selfState, 7, self->valueDescription); //steals ref of self->axisDescription
    PyTuple_SetItem(selfState, 8, PyFloat_FromDouble(self->valueOffset));
    PyTuple_SetItem(selfState, 9, PyFloat_FromDouble(self->valueScale));

    state = PyTuple_New(PyTuple_Size(baseState)+1); //new
    PyObject* temp;
    for(Py_ssize_t i = 0 ; i < PyTuple_Size(baseState) ; i++)
    {
        temp = PyTuple_GetItem(baseState, i);
        Py_INCREF(temp);
        PyTuple_SetItem(state, i, temp);
    }
    PyTuple_SetItem(state, PyTuple_Size(baseState), selfState); //steals ref of selfState

    Py_DECREF(baseState);

    PyTuple_SetItem(baseDump, 2, state); //baseDump[2] = tuple( baseState (for ndarray) , selfState (for npDataObject) )

    Py_XDECREF(ndarray);
    return baseDump;
}

PyObject* PythonNpDataObject::PyNpDataObj_SetState(PyNpDataObject *self, PyObject *args)
{
    //load numpy module
    PyObject *mod = PyImport_AddModule("numpy"); //borrowed ref
    if( mod == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "the numpy module cannot be loaded. Please make sure, that numpy is properly installed.");
        return NULL;
    }

    PyObject *ndarray = PyObject_GetAttrString(mod, "ndarray"); //new ref
    if(ndarray == NULL)
    {
        return NULL;
    }

    PyObject *baseDump = NULL;

    if(!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &baseDump))
    {
        Py_XDECREF(ndarray);
        PyErr_Print();
        return NULL;
    }

    if(!PyTuple_Check(baseDump))
    {
        PyErr_SetString(PyExc_ValueError, "unpickling state must contain a tuple for type npDataObject");
        Py_XDECREF(ndarray);
        Py_XDECREF(baseDump);
        return NULL;
    }

    if(PyTuple_Size(baseDump)<6) //elem 1-5 for baseState of ndArray, 6 is state-Tuple from npDataObject
    {
        PyErr_SetString(PyExc_ValueError, "unpickling state must contain a tuple with 6 elements");
        Py_XDECREF(baseDump);
        return NULL;
    }

    Py_ssize_t i = PyTuple_Size(baseDump);

    //state tuple consists of the following tuple (elements for ndarray , one tuple for self)
    PyObject *baseState = PyTuple_GetSlice(baseDump,0,i-1); //new
    //int j = PyTuple_Size(baseState);
    PyObject *selfState = PyTuple_GetItem(baseDump,i-1); //borrowed
    //int z = PyTuple_Size(selfState);



    PyObject* baseResult = PyObject_CallMethodObjArgs( ndarray , PyUnicode_FromString("__setstate__"), (PyObject*)self, baseState , NULL);

    if(baseResult != Py_None)
    {
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        PyErr_Print();
        return NULL;
    }

    /*
    selfState contains of:
    0. check-value (here: 21120)
    1. PyObject* tags (dict)
    2. PyObject* axisScales (list)
    3. PyObject* axisOffsets (list)
    4. PyObject* axisDescriptions (list)
    5. PyObject* axisUnits (list)
    6. PyObject* valueUnit (unicode)
    7. PyObject* valueDescription (unicode)
    8. double valueOffset
    9. double valueScale
    */

    //now parse selfState and fill tags
    int checkValue = 0;

    if(!PyArg_ParseTuple(selfState, "iO!OOOOOOdd", &checkValue, &PyDict_Type, &self->tags, &self->axisScales, &self->axisOffsets, &self->axisDescriptions, &self->axisUnits, &self->valueUnit, &self->valueDescription, &self->valueOffset, &self->valueScale))
    {
        PyErr_SetString(PyExc_TypeError, "the content of the pickled array does not correspond to the necessary structure of npDataObject.");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }

    if(PySequence_Check(self->axisScales) == false)
    {
        PyErr_SetString(PyExc_TypeError, "axisScales is no sequence type.");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }
    if(PySequence_Check(self->axisOffsets) == false)
    {
        PyErr_SetString(PyExc_TypeError, "axisScales is no sequence type.");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }
    if(PySequence_Check(self->axisDescriptions) == false)
    {
        PyErr_SetString(PyExc_TypeError, "axisScales is no sequence type.");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }
    if(PySequence_Check(self->axisUnits) == false)
    {
        PyErr_SetString(PyExc_TypeError, "axisScales is no sequence type.");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }
    if(PyBytes_Check(self->valueDescription) == false && PyUnicode_Check(self->valueDescription) == false)
    {
        PyErr_SetString(PyExc_TypeError, "valueDescription is no string");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }
    if(PyBytes_Check(self->valueUnit) == false && PyUnicode_Check(self->valueUnit) == false)
    {
        PyErr_SetString(PyExc_TypeError, "valueUnit is no string");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }

    Py_INCREF(self->tags);
    Py_INCREF(self->axisScales);
    Py_INCREF(self->axisOffsets);
    Py_INCREF(self->axisDescriptions);
    Py_INCREF(self->axisUnits);
    Py_INCREF(self->valueDescription);
    Py_INCREF(self->valueUnit);

    if(checkValue != 21120)
    {
        PyErr_SetString(PyExc_TypeError, "the check value in the pickled array does not have the right value.");
        Py_XDECREF(baseDump);
        Py_XDECREF(baseState);
        Py_XDECREF(baseResult);
        Py_XDECREF(ndarray);
        return NULL;
    }

    Py_XDECREF(baseDump);
    Py_XDECREF(baseState);
    Py_XDECREF(baseResult);
    Py_XDECREF(ndarray);

    Py_RETURN_NONE;
}


//getter and setter methods
PyObject* PythonNpDataObject::PyNpDataObject_getTags(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->tags);
    return self->tags;
}

int PythonNpDataObject::PyNpDataObject_setTags(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (! PyDict_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "The tag attribute must be a dictionary");
        return -1;
    }

    Py_XDECREF(self->tags);
    Py_INCREF(value);
    self->tags = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getAxisScales(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->axisScales);
    return self->axisScales;
}

int PythonNpDataObject::PyNpDataObject_setAxisScales(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    int dims = PyArray_NDIM(&(self->numpyArray)); //self->base.nd;
    if(dims == 1) dims = 2;

    if(!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis scales must be a sequence");
        return -1;
    }
    if(PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis scale sequence must be equal to number of dimensions");
        return -1;
    }

    PyObject *valueItem = NULL;

    for(Py_ssize_t i = 0; i < PySequence_Size(value); i++)
    {
        valueItem = PySequence_GetItem(value,i); //new reference

        if(!PyLong_Check(valueItem) && !PyFloat_Check(valueItem))
        {
            Py_XDECREF(valueItem);
            PyErr_SetString(PyExc_TypeError, "axis scale values must have a numeric value");
            return -1;
        }
        Py_XDECREF(valueItem);
    }

    Py_XDECREF(self->axisScales);
    Py_INCREF(value);
    self->axisScales = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getAxisOffsets(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->axisOffsets);
    return self->axisOffsets;
}

int PythonNpDataObject::PyNpDataObject_setAxisOffsets(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    int dims = PyArray_NDIM(&(self->numpyArray)); //self->base.nd;
    if(dims == 1) dims = 2;

    if(!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis offsets must be a sequence");
        return -1;
    }
    if(PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis offset sequence must be equal to number of dimensions");
        return -1;
    }

    PyObject *valueItem = NULL;

    for(Py_ssize_t i = 0; i < PySequence_Size(value); i++)
    {
        valueItem = PySequence_GetItem(value,i); //new reference
        if(!PyLong_Check(valueItem) && !PyFloat_Check(valueItem))
        {
            Py_XDECREF(valueItem);
            PyErr_SetString(PyExc_TypeError, "axis offset values must have a numeric value");
            return -1;
        }
        Py_XDECREF(valueItem);
    }

    Py_XDECREF(self->axisOffsets);
    Py_INCREF(value);
    self->axisOffsets = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getAxisDescriptions(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->axisDescriptions);
    return self->axisDescriptions;
}

int PythonNpDataObject::PyNpDataObject_setAxisDescriptions(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    int dims = PyArray_NDIM(&(self->numpyArray)); //self->base.nd;
    if(dims == 1) dims = 2;

    if(!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis descriptions must be a sequence");
        return -1;
    }
    if(PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis descriptions sequence must be equal to number of dimensions");
        return -1;
    }

    Py_XDECREF(self->axisDescriptions);
    Py_INCREF(value);
    self->axisDescriptions = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getAxisUnits(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->axisUnits);
    return self->axisUnits;
}

int PythonNpDataObject::PyNpDataObject_setAxisUnits(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    int dims = PyArray_NDIM(&(self->numpyArray)); //self->base.nd;
    if(dims == 1) dims = 2;

    if(!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis units must be a sequence");
        return -1;
    }
    if(PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis unit sequence must be equal to number of dimensions");
        return -1;
    }

    Py_XDECREF(self->axisUnits);
    Py_INCREF(value);
    self->axisUnits = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getValueUnit(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->valueUnit);
    return self->valueUnit;
}

int PythonNpDataObject::PyNpDataObject_setValueUnit(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (! PyUnicode_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "The value unit must be a string");
        return -1;
    }

    Py_XDECREF(self->valueUnit);
    Py_INCREF(value);
    self->valueUnit = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getValueDescription(PyNpDataObject *self, void * /*closure*/)
{
    Py_INCREF(self->valueDescription);
    return self->valueDescription;
}

int PythonNpDataObject::PyNpDataObject_setValueDescription(PyNpDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (! PyUnicode_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "The value description must be a string");
        return -1;
    }

    Py_XDECREF(self->valueDescription);
    Py_INCREF(value);
    self->valueDescription = value;

    return 0;
}

PyObject* PythonNpDataObject::PyNpDataObject_getValueScale(PyNpDataObject *self, void * /*closure*/)
{
    return PyFloat_FromDouble(self->valueScale);
}

PyObject* PythonNpDataObject::PyNpDataObject_getValueOffset(PyNpDataObject *self, void * /*closure*/)
{
    return PyFloat_FromDouble(self->valueOffset);
}

PyObject* PythonNpDataObject::PyNpDataObject_getTagDict(PyNpDataObject *self, void * /*clousure*/)
{
    PyObject *dict = PyDict_New();
    //Py_INCREF(self->tags);
    PyDict_SetItemString(dict, "tags", self->tags);
    //Py_INCREF(self->axisOffsets);
    PyDict_SetItemString(dict, "axisOffsets", self->axisOffsets);
    //Py_INCREF(self->axisScales);
    PyDict_SetItemString(dict, "axisScales", self->axisScales);
    //Py_INCREF(self->axisDescriptions);
    PyDict_SetItemString(dict, "axisDescriptions", self->axisDescriptions);
    //Py_INCREF(self->axisUnits);
    PyDict_SetItemString(dict, "axisUnits", self->axisUnits);
    //Py_INCREF(self->valueUnit);
    PyDict_SetItemString(dict, "valueUnit", self->valueUnit);
    //Py_INCREF(self->valueDescription);
    PyDict_SetItemString(dict, "valueDescription", self->valueDescription);

    PyObject *val = PyFloat_FromDouble(self->valueOffset);
    PyDict_SetItemString(dict, "valueOffset", val);
    Py_DECREF(val);

    val = PyFloat_FromDouble(self->valueScale);
    PyDict_SetItemString(dict, "valueScale", val);
    Py_DECREF(val);

    return dict; //returns new reference
}
