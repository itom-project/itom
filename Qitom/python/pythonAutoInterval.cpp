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

#include "pythonEngineInc.h"
#include "pythonAutoInterval.h"

#include "../global.h"

#include "pythonQtConversion.h"

namespace ito
{

//------------------------------------------------------------------------------------------------------
void PythonAutoInterval::PyAutoInterval_dealloc(PyAutoInterval* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
};


//------------------------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyAutoInterval* self = (PyAutoInterval *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->interval.rmin() = -std::numeric_limits<float>::infinity();
        self->interval.rmax() = std::numeric_limits<float>::infinity();
        self->interval.rauto() = true;
    }

    return (PyObject *)self;
};


//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(autoIntervalInit_doc,"autoInterval([min=-inf, max=inf, auto=true]) -> creates a new auto interval object.\n\
\n\
Parameters \n\
----------- \n\
min : {float} \n\
    minimum value of interval (default: -infinity) \n\
max : {float}, \n\
    maximum value of interval (default: +infinity) \n\
auto : {bool} \n\
    false if interval is fixed, true if the interval can be scaled automatically (default)");
int PythonAutoInterval::PyAutoInterval_init(PyAutoInterval *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"min", "max", "auto", NULL};
    self->interval.rmin() = -std::numeric_limits<float>::infinity();
    self->interval.rmax() = std::numeric_limits<float>::infinity();
    self->interval.rauto() = true;

    if (args == NULL && kwds == NULL)
    {
        return 0; //call from createEmptyPyAutoInterval
    }

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|ffB", const_cast<char**>(kwlist), &(self->interval.rmin()), &(self->interval.rmax()), &(self->interval.rauto())))
    {
        return -1;
    }

    if (self->interval.rmin() > self->interval.rmax())
    {
        std::swap(self->interval.rmin(), self->interval.rmax());
    }

    return 0;
}


//------------------------------------------------------------------------------------------------------
PythonAutoInterval::PyAutoInterval* PythonAutoInterval::createEmptyPyAutoInterval()
{
    PyAutoInterval* result = (PyAutoInterval*)PyObject_Call((PyObject*)&PyAutoIntervalType, NULL, NULL);
    if(result != NULL)
    {
        return result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_name(PyAutoInterval* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("autoInterval");
    return result;
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_repr(PyAutoInterval *self)
{
    QString str;
    if (self->interval.minimum() == std::numeric_limits<float>::min())
    {
        if (self->interval.maximum() == std::numeric_limits<float>::max())
        {
            str = QString("autoInterval([-Inf,Inf], auto: %3)").arg(self->interval.isAuto());
        }
        else
        {
            str = QString("autoInterval([-Inf,%1], auto: %3)").arg(self->interval.maximum()).arg(self->interval.isAuto());
        }
    }
    else if (self->interval.maximum() == std::numeric_limits<float>::max())
    {
        str = QString("autoInterval([%,Inf], auto: %3)").arg(self->interval.minimum()).arg(self->interval.isAuto());
    }
    else
    {
        str = QString("autoInterval([%1,%2], auto: %3)").arg(self->interval.minimum()).arg(self->interval.maximum()).arg(self->interval.isAuto());
    }
    
    PyObject *result = PyUnicode_FromFormat("%s", str.toLatin1().data());
    return result;
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_Reduce(PyAutoInterval *self, PyObject * /*args*/)
{
    PyObject *stateTuple = PyTuple_New(0);

    PyObject *tempOut = Py_BuildValue("(O(ffB)O)", Py_TYPE(self), self->interval.rmin(), self->interval.rmax(), self->interval.rauto(), stateTuple);
    Py_DECREF(stateTuple);

    return tempOut;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_SetState(PyAutoInterval *self, PyObject *args)
{
    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_RichCompare(PyAutoInterval *self, PyObject *other, int cmp_op)
{
    if(other == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "compare object is empty.");
        return NULL;
    }

    //check type of other
    PyAutoInterval* otherInterval = NULL;
    ito::DataObject resDataObj;

    if(PyArg_Parse(other, "O!", &PyAutoIntervalType, &otherInterval))
    {
        switch (cmp_op)
        {
        //case Py_LT: resultRgba->rgba = self->rgba < otherRgba->rgba; break;
        //case Py_LE: resultRgba->rgba = self->rgba <= otherRgba->rgba; break;
        case Py_EQ: if (self->interval == otherInterval->interval) { Py_RETURN_TRUE; } else { Py_RETURN_FALSE; } break;
        case Py_NE: if (self->interval != otherInterval->interval) { Py_RETURN_TRUE; } else { Py_RETURN_FALSE; } break;
        //case Py_GT: resultRgba->rgba = self->rgba > otherRgba->rgba; break;
        //case Py_GE: resultRgba->rgba = self->rgba >= otherRgba->rgba; break;
        }
        PyErr_SetString(PyExc_TypeError, "comparison of autoInterval is only defined for == and !=");
        return NULL;

    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "it is only possible to compare to objects of type autoInterval");
        return NULL;
    }
}



PyMethodDef PythonAutoInterval::PyAutoInterval_methods[] = {
        {"name", (PyCFunction)PythonAutoInterval::PyAutoInterval_name, METH_NOARGS, ""},
        
        {"__reduce__", (PyCFunction)PythonAutoInterval::PyAutoInterval_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
        {"__setstate__", (PyCFunction)PythonAutoInterval::PyAutoInterval_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
        
        {NULL}  /* Sentinel */
    };

PyMemberDef PythonAutoInterval::PyAutoInterval_members[] = {
    #ifdef WIN32 //TODO offsetof with member in GCC not possible
        {"min", T_FLOAT, offsetof(PyAutoInterval, interval.rmin()), 0, "min"}, 
        {"max", T_FLOAT, offsetof(PyAutoInterval, interval.rmax()), 0, "max"}, 
        {"auto", T_BOOL, offsetof(PyAutoInterval, interval.rauto()), 0, "auto"},  
    #endif
        {NULL}  /* Sentinel */
    };

PyModuleDef PythonAutoInterval::PyAutoIntervalModule = {
        PyModuleDef_HEAD_INIT,
        "autoInterval",
        "Itom autoInterval type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
    };

PyGetSetDef PythonAutoInterval::PyAutoInterval_getseters[] = {
    
    {NULL}  /* Sentinel */
};

PyTypeObject PythonAutoInterval::PyAutoIntervalType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.autoInterval",               /* tp_name */
        sizeof(PyAutoInterval),            /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyAutoInterval_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyAutoInterval_repr,     /* tp_repr */
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
        autoIntervalInit_doc,                    /* tp_doc */
        0,                                    /* tp_traverse */
        0,                                    /* tp_clear */
        (richcmpfunc)PyAutoInterval_RichCompare,            /* tp_richcompare */
        0,                                    /* tp_weaklistoffset */
        0,                                    /* tp_iter */
        0,                                    /* tp_iternext */
        PyAutoInterval_methods,                        /* tp_methods */
        PyAutoInterval_members,                        /* tp_members */
        PyAutoInterval_getseters,                    /* tp_getset */
        0,                                    /* tp_base */
        0,                                    /* tp_dict */
        0,                                    /* tp_descr_get */
        0,                                    /* tp_descr_set */
        0,                                    /* tp_dictoffset */
        (initproc)PyAutoInterval_init,                /* tp_init */
        0,                                    /* tp_alloc */
        PyAutoInterval_new /*PyType_GenericNew*/    /* tp_new */
    };


} //end namespace ito
