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
#include "pythonRgba.h"

#include "../global.h"

#include "pythonQtConversion.h"

namespace ito
{

//------------------------------------------------------------------------------------------------------
void PythonRgba::PyRgba_dealloc(PyRgba* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
};


//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyRgba* self = (PyRgba *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->rgba.rgba = 0xFF000000; //alpha 255, rest: 0
    }

    return (PyObject *)self;
};


//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectInit_doc,"rgba(r, g, b [, alpha=255]) -> creates a new color value from red, green, blue and optional alpha\n\
\n\
Parameters \n\
----------- \n\
r : {uint8} \n\
	red component [0,255] \n\
g : {uint8}, \n\
	green component [0,255] \n\
b : {uint8} \n\
	blue component [0,255] \n\
alpha : {uint8}, optional \n\
	alpha component [0,255], default: 255 (no transparancy) \n\
\n\
Notes \n\
------ \n\
\n\
For a gray value set all colors to the same value.");
int PythonRgba::PyRgba_init(PyRgba *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"r", "g", "b", "alpha", NULL};
	self->rgba.a = 255;

	if (args == NULL && kwds == NULL)
	{
		return 0; //call from createEmptyPyRgba
	}

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "BBB|B", const_cast<char**>(kwlist), &(self->rgba.r), &(self->rgba.g), &(self->rgba.b), &(self->rgba.a)))
    {
        return -1;
    }

    return 0;
}


//------------------------------------------------------------------------------------------------------
PythonRgba::PyRgba* PythonRgba::createEmptyPyRgba()
{
    PyRgba* result = (PyRgba*)PyObject_Call((PyObject*)&PyRgbaType, NULL, NULL);
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
bool PythonRgba::checkPyRgba(int number, PyObject* rgba1 /*= NULL*/, PyObject* rgba2 /*= NULL*/, PyObject* rgba3 /*= NULL*/)
{
    PyObject *temp;
    for (int i = 0; i < number; ++i)
    {
        switch(i)
        {
        case 0:
            temp = rgba1;
            break;
        case 1:
            temp = rgba2;
            break;
        case 2:
            temp = rgba3;
            break;
        default:
            continue;
        }

        if (temp == NULL)
        {
            PyErr_Format(PyExc_TypeError, "%i. operand is NULL", i+1);
            return false;
        }
        else if(!PyRgba_Check(temp))
        {
            PyErr_Format(PyExc_TypeError, "%i. operand must be a valid rgba value", i);
            return false;
        }
    }
    return true;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbAdd(PyObject* o1, PyObject* o2)
{
    PyRgba *rgba1 = NULL;
    PyRgba *rgba2 = NULL;
    
    if (PyRgba_Check(o1) && PyRgba_Check(o2))
    {
        rgba1 = (PyRgba*)o1;
        rgba2 = (PyRgba*)o2;
    }
    else
    {
        return PyErr_Format(PyExc_RuntimeError, "both operands must be of type rgba");
    }

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	retRgba->rgba = rgba1->rgba + rgba2->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbSubtract(PyObject* o1, PyObject* o2)
{
    PyRgba *rgba1 = NULL;
    PyRgba *rgba2 = NULL;
    
    if (PyRgba_Check(o1) && PyRgba_Check(o2))
    {
        rgba1 = (PyRgba*)o1;
        rgba2 = (PyRgba*)o2;
    }
    else
    {
        return PyErr_Format(PyExc_RuntimeError, "both operands must be of type rgba");
    }

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	retRgba->rgba = rgba1->rgba - rgba2->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbMultiply(PyObject* o1, PyObject* o2)
{
    if(o1 == NULL || o2 == NULL) return NULL;

    if(Py_TYPE(o1) == &PyRgbaType && Py_TYPE(o2) == &PyRgbaType)
    {
        PyRgba *rgba1 = (PyRgba*)(o1);
        PyRgba *rgba2 = (PyRgba*)(o2);

        PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
		retRgba->rgba = rgba1->rgba * rgba2->rgba;

        return (PyObject*)retRgba;
    }
    
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbPositive(PyObject* o1)
{
    if(!checkPyRgba(1,o1)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference

    retRgba->rgba = rgba1->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbAbsolute(PyObject* o1)
{
    if(!checkPyRgba(1,o1)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference

    retRgba->rgba = rgba1->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbLshift(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(1,o1)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);

    int shift = PyLong_AsLong(o2);

    if(PyErr_Occurred()) return NULL;
    if(shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	// not implemented yet !!  retRgba->rgba = rgba1->rgba << shift;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbRshift(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(1,o1)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);

    int shift = PyLong_AsLong(o2);

    if(PyErr_Occurred()) return NULL;
    if(shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	// not implemented yet !!  retRgba->rgba = rgba1->rgba >> shift;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbAnd(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
    PyRgba *rgba2 = (PyRgba*)(o2);

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	// not implemented yet !!  retRgba->rgba = rgba1->rgba & rgba2->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbXor(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
    PyRgba *rgba2 = (PyRgba*)(o2);

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	// not implemented yet !!  retRgba->rgba = rgba1->rgba ^ rgba2->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbOr(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
    PyRgba *rgba2 = (PyRgba*)(o2);

    PyRgba* retRgba = PythonRgba::createEmptyPyRgba(); // new reference
	// not implemented yet !!  retRgba->rgba = rgba1->rgba | rgba2->rgba;

    return (PyObject*)retRgba;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceAdd(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
	PyRgba *rgba2 = (PyRgba*)(o2);

	rgba1->rgba += rgba2->rgba;

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceSubtract(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
	PyRgba *rgba2 = (PyRgba*)(o2);

	rgba1->rgba -= rgba2->rgba;

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceMultiply(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
	PyRgba *rgba2 = (PyRgba*)(o2);

	rgba1->rgba *= rgba2->rgba;

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceLshift(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(1,o1)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);

    int shift = PyLong_AsLong(o2);

    if(PyErr_Occurred()) return NULL;
    if(shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    Py_INCREF(o1);

    // not implemented yet !!  rgba1->rgba <<= shift;

    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceRshift(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(1,o1)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);

    int shift = PyLong_AsLong(o2);

    if(PyErr_Occurred()) return NULL;
    if(shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    Py_INCREF(o1);

    // not implemented yet !!  rgba1->rgba >>= shift;

    return (PyObject*)o1;
}

PyObject* PythonRgba::PyRgba_nbInplaceAnd(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
    PyRgba *rgba2 = (PyRgba*)(o2);
	// not implemented yet !!  rgba1->rgba &= rgba2->rgba;
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceXor(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
    PyRgba *rgba2 = (PyRgba*)(o2);
	// not implemented yet !!  rgba1->rgba ^= rgba2->rgba;
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_nbInplaceOr(PyObject* o1, PyObject* o2)
{
    if(!checkPyRgba(2,o1,o2)) return NULL;

    PyRgba *rgba1 = (PyRgba*)(o1);
    PyRgba *rgba2 = (PyRgba*)(o2);
	// not implemented yet !!  rgba1->rgba |= rgba2->rgba;
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_name(PyRgba* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("Rgba");
    return result;
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_repr(PyRgba *self)
{
    PyObject *result = PyUnicode_FromFormat("Rgba(%i,%i,%i alpha:%i)", self->rgba.r, self->rgba.g, self->rgba.b, self->rgba.a);
    return result;
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_Reduce(PyRgba *self, PyObject * /*args*/)
{
    PyObject *stateTuple = PyTuple_New(0);

    PyObject *tempOut = Py_BuildValue("(O(BBBB)O)", Py_TYPE(self), self->rgba.r, self->rgba.g, self->rgba.b, self->rgba.a, stateTuple);
    Py_DECREF(stateTuple);

    return tempOut;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_SetState(PyRgba *self, PyObject *args)
{
    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonRgba::PyRgba_RichCompare(PyRgba *self, PyObject *other, int cmp_op)
{
    if(other == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "compare object is empty.");
        Py_RETURN_NONE;
    }

    //check type of other
    PyRgba* otherRgba = NULL;
    ito::DataObject resDataObj;
    PyRgba* resultRgba = createEmptyPyRgba();

    if(PyArg_Parse(other, "O!", &PyRgbaType, &otherRgba))
    {
        switch (cmp_op)
        {
        //case Py_LT: resultRgba->rgba = self->rgba < otherRgba->rgba; break;
        //case Py_LE: resultRgba->rgba = self->rgba <= otherRgba->rgba; break;
        case Py_EQ: resultRgba->rgba = self->rgba == otherRgba->rgba; break;
        case Py_NE: resultRgba->rgba = self->rgba != otherRgba->rgba; break;
        //case Py_GT: resultRgba->rgba = self->rgba > otherRgba->rgba; break;
        //case Py_GE: resultRgba->rgba = self->rgba >= otherRgba->rgba; break;
        }
        return (PyObject*)resultRgba;

    }
    else
    {
        return NULL;
    }
}



PyMethodDef PythonRgba::PyRgba_methods[] = {
        {"name", (PyCFunction)PythonRgba::PyRgba_name, METH_NOARGS, ""},
        
        {"__reduce__", (PyCFunction)PythonRgba::PyRgba_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
        {"__setstate__", (PyCFunction)PythonRgba::PyRgba_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
        
        {NULL}  /* Sentinel */
    };

PyMemberDef PythonRgba::PyRgba_members[] = {
    #ifndef linux //TODO offsetof with member in GCC not possible
        {"r", T_UBYTE, offsetof(PyRgba, rgba.r), 0, "red"}, 
		{"g", T_UBYTE, offsetof(PyRgba, rgba.g), 0, "green"}, 
		{"b", T_UBYTE, offsetof(PyRgba, rgba.b), 0, "blue"}, 
		{"alpha", T_UBYTE, offsetof(PyRgba, rgba.a), 0, "alpha"}, 
    #endif
        {NULL}  /* Sentinel */
    };

PyModuleDef PythonRgba::PyRgbaModule = {
        PyModuleDef_HEAD_INIT,
        "rgba",
        "Itom Rgba color type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
    };

PyGetSetDef PythonRgba::PyRgba_getseters[] = {
    
    {NULL}  /* Sentinel */
};

PyTypeObject PythonRgba::PyRgbaType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.rgba",			   /* tp_name */
        sizeof(PyRgba),            /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyRgba_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyRgba_repr,     /* tp_repr */
        &PyRgba_numberProtocol,    /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        dataObjectInit_doc,					/* tp_doc */
        0,									/* tp_traverse */
        0,									/* tp_clear */
        (richcmpfunc)PyRgba_RichCompare,            /* tp_richcompare */
        0,									/* tp_weaklistoffset */
        0,									/* tp_iter */
        0,									/* tp_iternext */
        PyRgba_methods,						/* tp_methods */
        PyRgba_members,						/* tp_members */
        PyRgba_getseters,					/* tp_getset */
        0,									/* tp_base */
        0,									/* tp_dict */
        0,									/* tp_descr_get */
        0,									/* tp_descr_set */
        0,									/* tp_dictoffset */
        (initproc)PyRgba_init,				/* tp_init */
        0,									/* tp_alloc */
        PyRgba_new /*PyType_GenericNew*/    /* tp_new */
    };


PyNumberMethods PythonRgba::PyRgba_numberProtocol = {
    (binaryfunc)PyRgba_nbAdd,               /* nb_add */
    (binaryfunc)PyRgba_nbSubtract,          /* nb_subtract */
    (binaryfunc)PyRgba_nbMultiply,          /* nb_multiply */
    (binaryfunc)0,							/* nb_remainder */
    (binaryfunc)0,							/* nb_divmod */
    (ternaryfunc)0,							/* nb_power */
    (unaryfunc)0,							/* nb_negative */
    (unaryfunc)PyRgba_nbPositive,			/* nb_positive */
    (unaryfunc)PyRgba_nbAbsolute,           /* nb_absolute */
    (inquiry)0,								/* nb_bool */
    (unaryfunc)0,                           /* nb_invert */
//    (binaryfunc)PyRgba_nbLshift,            /* nb_lshift */
//    (binaryfunc)PyRgba_nbRshift,            /* nb_rshift */
//    (binaryfunc)PyRgba_nbAnd,               /* nb_and */
//    (binaryfunc)PyRgba_nbXor,               /* nb_xor */
//    (binaryfunc)PyRgba_nbOr,                /* nb_or */
    (binaryfunc)0,            /* nb_lshift */
    (binaryfunc)0,            /* nb_rshift */
    (binaryfunc)0,               /* nb_and */
    (binaryfunc)0,               /* nb_xor */
    (binaryfunc)0,                /* nb_or */
    0,										/* nb_int */
    0,                                      /* nb_reserved */
    0,										/* nb_float */
    (binaryfunc)PyRgba_nbInplaceAdd,        /* nb_inplace_add */
    (binaryfunc)PyRgba_nbInplaceSubtract,   /* nb_inplace_subtract */
    (binaryfunc)PyRgba_nbInplaceMultiply,   /* nb_inplace_multiply*/
    (binaryfunc)0,                          /* nb_inplace_remainder */
    (ternaryfunc)0,                         /* nb_inplace_power */
//    (binaryfunc)PyRgba_nbInplaceLshift,   /* nb_inplace_lshift */
//    (binaryfunc)PyRgba_nbInplaceRshift,   /* nb_inplace_rshift */
//    (binaryfunc)PyRgba_nbInplaceAnd,      /* nb_inplace_and */
//    (binaryfunc)PyRgba_nbInplaceXor,		/* nb_inplace_xor */
//    (binaryfunc)PyRgba_nbInplaceOr,		/* nb_inplace_or */
    (binaryfunc)0,                          /* nb_inplace_lshift */
    (binaryfunc)0,                          /* nb_inplace_rshift */
    (binaryfunc)0,                          /* nb_inplace_and */
    (binaryfunc)0,                          /* nb_inplace_xor */
    (binaryfunc)0,                          /* nb_inplace_or */
    (binaryfunc)0,                /* nb_floor_divide */
    (binaryfunc)0,                /* nb_true_divide */
    0,                            /* nb_inplace_floor_divide */
    0,                            /* nb_inplace_true_divide */
};

} //end namespace ito
