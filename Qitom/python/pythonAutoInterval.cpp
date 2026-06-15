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

#include "pythonEngineInc.h"
#include "pythonAutoInterval.h"

#include "../global.h"

#include "pythonQtConversion.h"

namespace ito
{

//-------------------------------------------------------------------------------------
void PythonAutoInterval::PyAutoInterval_dealloc(PyAutoInterval* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
};


//-------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyAutoInterval* self = (PyAutoInterval *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->interval.rmin() = -std::numeric_limits<double>::infinity();
        self->interval.rmax() = std::numeric_limits<double>::infinity();
        self->interval.rauto() = true;
    }

    return (PyObject *)self;
};


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(autoIntervalInit_doc,"autoInterval(min = -math.inf, max = math.inf, auto = True) -> autoInterval \n\
\n\
Creates a new (auto) interval object.\n\
\n\
\n\
Properties and slots of :class:`uiItem` objects (e.g. plots) sometimes have parameters \n\
of type :class:`autoInterval`. It is either possible to pass an :class:`autoInterval` \n\
instance, the string ``auto`` or a list or tuple with the two limit values ``[min, max]``. \n\
\n\
Example:: \n\
\n\
    [i,h] = plot(dataObject.randN([100, 100], 'int8'))\n\
    h['xAxisInterval'] = autoInterval(20, 80)\n\
    h['yAxisInterval'] = 'auto' \n\
    h['zAxisInterval'] = [-90, 90] \n\
\n\
Parameters \n\
---------- \n\
min : float, optional \n\
    minimum value of interval (default: -:obj:`math.inf`). \n\
max : float, optional \n\
    maximum value of interval (default: :obj:`math.inf`). \n\
auto : bool, optional \n\
    ``False`` if interval is fixed, ``True`` if the interval can be scaled \n\
    automatically (default).");
int PythonAutoInterval::PyAutoInterval_init(PyAutoInterval *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"min", "max", "auto", NULL};
    self->interval.rmin() = -std::numeric_limits<double>::infinity();
    self->interval.rmax() = std::numeric_limits<double>::infinity();
    self->interval.rauto() = true;

    if (args == NULL && kwds == NULL)
    {
        return 0; //call from createEmptyPyAutoInterval
    }

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|ddB", const_cast<char**>(kwlist), &(self->interval.rmin()), &(self->interval.rmax()), &(self->interval.rauto())))
    {
        return -1;
    }

    if (self->interval.rmin() > self->interval.rmax())
    {
        std::swap(self->interval.rmin(), self->interval.rmax());
    }

    return 0;
}


//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(autoInterval_name_doc, "name() -> str \n\
\n\
Returns the name of this object. \n\
\n\
Returns \n\
------- \n\
str \n\
    name of this object ``autoInterval``.");
PyObject* PythonAutoInterval::PyAutoInterval_name(PyAutoInterval* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("autoInterval");
    return result;
};

//-------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_repr(PyAutoInterval *self)
{
    QString str;
    if (self->interval.minimum() == std::numeric_limits<double>::min())
    {
        if (self->interval.maximum() == std::numeric_limits<double>::max())
        {
            str = QString("autoInterval([-Inf,Inf], auto: %3)").arg(self->interval.isAuto());
        }
        else
        {
            str = QString("autoInterval([-Inf,%1], auto: %3)").arg(self->interval.maximum()).arg(self->interval.isAuto());
        }
    }
    else if (self->interval.maximum() == std::numeric_limits<double>::max())
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

//-------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_Reduce(PyAutoInterval *self, PyObject * /*args*/)
{
    //method to pickle (save to IDC) autoInterval objects
    /*
    This method is mapped to itom.autoInterval.__reduce__.

    If this function exists, the pickle module is able to save autoInterval instances to pickled
    files (e.g. idc files). This method is called, whenever an autoInterval object should be pickled.
    Its purpose is to return a Python tuple, that only consists of basic python types, which can be pickled.

    If an idc file is loaded, pickle knows if an autoInterval is contained. If so, the object, returned
    by this function is used to reconstruct the object, also by the help of the method below, mapped to itom.autoInterval.__setstate__.

    The meaning of the returned tuple is as follows:

    (type of the autoInterval class, an arbitrary constructor-tuple, a user-defined object)

    If the object is reconstructed, the constructor of autoInterval is called with the unpacked version of the arbitrary constructor tuple.
    Afterwards the __setstate__ method is called (with the newly created instance and the user-defined object). The __setstate__ method
    can then adapt the new instance using the content of the user-defined object.

    Here, no user-defined object is necessary, since the autoInterval object can be fully reconstructed using the
    three values (min, max, auto), passed to the constructor (__init__).
    */
    PyObject *stateTuple = PyTuple_New(0);

    PyObject *tempOut = Py_BuildValue("(O(ddB)O)", Py_TYPE(self), self->interval.rmin(), self->interval.rmax(), self->interval.rauto(), stateTuple);
    Py_DECREF(stateTuple);

    return tempOut;
}

//-------------------------------------------------------------------------------------
PyObject* PythonAutoInterval::PyAutoInterval_SetState(PyAutoInterval *self, PyObject *args)
{
    /*documentation see method above*/
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
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

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(autoInterval_min_doc,
"float : Gets or sets the minimum value of the interval.");

PyObject* PythonAutoInterval::PyAutoInterval_getMin(PyAutoInterval *self, void *closure)
{
	return PyFloat_FromDouble(self->interval.minimum());
}

int PythonAutoInterval::PyAutoInterval_setMin(PyAutoInterval *self, PyObject *value, void *closure)
{
	bool ok;
	double minimum = PythonQtConversion::PyObjGetDouble(value, false, ok);

	if (ok)
	{
		self->interval.rmin() = (float)minimum;
		return 0;
	}

	PyErr_SetString(PyExc_TypeError, "minimum value must be a float");
	return -1;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(autoInterval_max_doc,
"float : Gets or sets the maximum value of the interval.");

PyObject* PythonAutoInterval::PyAutoInterval_getMax(PyAutoInterval *self, void *closure)
{
	return PyFloat_FromDouble(self->interval.maximum());
}

int PythonAutoInterval::PyAutoInterval_setMax(PyAutoInterval *self, PyObject *value, void *closure)
{
	bool ok;
	double maximum = PythonQtConversion::PyObjGetDouble(value, false, ok);

	if (ok)
	{
		self->interval.rmax() = (float)maximum;
		return 0;
	}

	PyErr_SetString(PyExc_TypeError, "maximum value must be a float");
	return -1;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(autoInterval_auto_doc,
"bool : Gets or sets if this interval has an automatic range.");

PyObject* PythonAutoInterval::PyAutoInterval_getAuto(PyAutoInterval *self, void *closure)
{
	return PyBool_FromLong(self->interval.isAuto() ? 1 : 0);
}

int PythonAutoInterval::PyAutoInterval_setAuto(PyAutoInterval *self, PyObject *value, void *closure)
{
	bool ok;
	bool auto_flag = PythonQtConversion::PyObjGetBool(value, false, ok);

	if (ok)
	{
		self->interval.rauto() = auto_flag;
		return 0;
	}

	PyErr_SetString(PyExc_TypeError, "auto value must be a bool");
	return -1;
}


//-------------------------------------------------------------------------------------
PyMethodDef PythonAutoInterval::PyAutoInterval_methods[] = {
        {"name", (PyCFunction)PythonAutoInterval::PyAutoInterval_name, METH_NOARGS, autoInterval_name_doc},

        {"__reduce__", (PyCFunction)PythonAutoInterval::PyAutoInterval_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
        {"__setstate__", (PyCFunction)PythonAutoInterval::PyAutoInterval_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},

        {NULL}  /* Sentinel */
    };

PyMemberDef PythonAutoInterval::PyAutoInterval_members[] = {
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
	{ "min", (getter)PyAutoInterval_getMin,         (setter)PyAutoInterval_setMin,    autoInterval_min_doc, NULL },
	{ "max", (getter)PyAutoInterval_getMax,         (setter)PyAutoInterval_setMax,    autoInterval_max_doc, NULL },
	{ "auto", (getter)PyAutoInterval_getAuto,         (setter)PyAutoInterval_setAuto,    autoInterval_auto_doc, NULL },
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
