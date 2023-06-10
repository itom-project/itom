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

#include "pythonProxy.h"

//-------------------------------------------------------------------------------------

namespace ito
{

//-------------------------------------------------------------------------------------
/*
    Our own proxy object which enables weak references to bound and unbound
    methods and arbitrary callables. Pulls information about the function,
    class, and instance out of a bound method. Stores a weak reference to the
    instance to support garbage collection.

    @organization: IBM Corporation
    @copyright: Copyright (c) 2005, 2006 IBM Corporation
    @license: The BSD License

    Idea from http://mindtrove.info/python-weak-references/ and the Linux Screen Reader project.
*/
void PythonProxy::PyProxy_addTpDict(PyObject * /*tp_dict*/)
{
}

//-------------------------------------------------------------------------------------
void PythonProxy::PyProxy_dealloc(PyProxy* self)
{
    Py_XDECREF(self->klass);
    Py_XDECREF(self->inst);
    Py_XDECREF(self->func);
    Py_XDECREF(self->base);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//-------------------------------------------------------------------------------------
PyObject* PythonProxy::PyProxy_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyProxy* self = (PyProxy *)type->tp_alloc(type, 0);

    if (self != nullptr)
    {
        self->klass = nullptr;
        self->inst = nullptr;
        self->func = nullptr;
        self->base = nullptr;
    }

    return (PyObject *)self;
};

//-------------------------------------------------------------------------------------
int PythonProxy::PyProxy_init(PyProxy *self, PyObject *args, PyObject * /*kwds*/)
{
    PyObject *method = nullptr;

    if (!PyArg_ParseTuple(args, "O", &method))
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "argument must be a bounded or unbounded method or function");
        return -1;
    }

    if (!PyMethod_Check(method) &&
        !PyCFunction_Check(method) &&
        !PyCallable_Check(method))
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "argument must be a bounded or unbounded method or function");
        return -1;
    }

    if (!PyCFunction_Check(method) && PyObject_HasAttrString(method, "__self__"))
    {
        PyObject *temp = PyObject_GetAttrString(method, "__self__"); //new reference
        self->inst = PyWeakref_NewRef(temp, nullptr); //new ref
        Py_DECREF(temp);

    }
    else
    {
        self->inst = nullptr;
    }

    Py_INCREF(Py_None);
    self->klass = Py_None;

    if (PyObject_HasAttrString(method, "__func__"))
    {
        self->func = PyObject_GetAttrString(method, "__func__"); //new reference
    }
    else
    {
        self->func = method;
        Py_INCREF(self->func);
    }

    return 0;
};

//-------------------------------------------------------------------------------------
/*
Compare the held function and instance with that held by another proxy.

@param other: Another proxy object
@type other: L{Proxy}
@return: Whether this func/inst pair is equal to the one in the other
proxy object or not
@rtype: boolean
*/
PyObject* PythonProxy::PyProxy_richcompare(PyObject *v, PyObject *w, int op)
{
    if (op == Py_EQ || op == Py_NE)
    {
        PyProxy *v2 = (PyProxy*)v;
        PyProxy *w2 = (PyProxy*)w;

        bool res = false;

        if (v2 == nullptr || w2 == nullptr)
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "both elements of the comparison must be of type PyProxy."
            );
            return nullptr;
        }

        if (v2->func == w2->func)
        {
            if (v2->inst == nullptr && w2->inst == nullptr)
            {
                res = true;
            }
            else if (PyWeakref_GetObject(v2->inst) == PyWeakref_GetObject(w2->inst))
            {
                res = true;
            }
        }

        if (op == Py_EQ)
        {
            if (res) Py_RETURN_TRUE;
            Py_RETURN_FALSE;
        }
        else
        {
            if (res) Py_RETURN_FALSE;
            Py_RETURN_TRUE;
        }
    }
    else
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "For the proxy-object, only the comparison operators == and != are allowed."
        );
        return nullptr;
    }
}

//-------------------------------------------------------------------------------------
/*
Proxy for a call to the weak referenced object. Take arbitrary params to
pass to the callable.

@raise ReferenceError: When the weak reference refers to a dead object
*/
PyObject* PythonProxy::PyProxy_call(PyProxy *self, PyObject *args, PyObject *kwds)
{
    PyObject *mtd = nullptr;
    PyObject *res = nullptr;
    PyObject *wr = nullptr;

    if (self->inst != nullptr)
    {
        wr = PyWeakref_GetObject(self->inst); //borrowed reference

        if (wr == Py_None)
        {
            PyErr_SetString(
                PyExc_ReferenceError,
                "The reference to the instance of the proxy object is dead"
            );
            return nullptr;
        }
        else
        {
            mtd = PyObject_CallMethod(self->func, "__get__", "OO", wr, self->klass); //new reference
        }
    }
    else
    {
        //not a bound method, just return the func
        mtd = self->func;
        Py_INCREF(mtd);
    }

    res = PyObject_Call(mtd, args, kwds); //new reference
    Py_XDECREF(mtd);
    return res;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonProxy::PyProxyModule = {
    PyModuleDef_HEAD_INIT, "Proxy", "Weak reference proxy object for (un)bounded method calls", -1,
    nullptr, nullptr, nullptr, nullptr, nullptr
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonProxy::PyProxyType = {
    PyVarObject_HEAD_INIT(nullptr, 0) /* here has been NULL,0 */
    "itom.proxy",             /* tp_name */
    sizeof(PyProxy),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PythonProxy::PyProxy_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    (ternaryfunc)PythonProxy::PyProxy_call,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    0,           /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    PythonProxy::PyProxy_richcompare,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    0,             /* tp_methods */
    0, /*PyNpDataObject_members,*/             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PythonProxy::PyProxy_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PythonProxy::PyProxy_new         /* tp_new */
};

} //end namespace ito
