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

#include "pythonUiTimer.h"

#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "structmember.h"
#endif

#include "../global.h"

#include "pythonEngineInc.h"
#include "AppManagement.h"

#include <iostream>
#include <qvector.h>
#include <qtimer.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
void TimerCallback::timeout()
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if(Py_IsInitialized() == 0)
    {
        qDebug("python is not available any more");
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    bool debug = false;
    if(pyEngine)
    {
        debug = pyEngine->execInternalCodeByDebugger();
    }

    if(m_boundedMethod == false)
    {
        PyObject *func = PyWeakref_GetObject(m_function);
        if((func != NULL) && (func != Py_None))
        {
            if(debug)
            {
                pyEngine->pythonDebugFunction(func, m_callbackArgs, true);
            }
            else
            {
                pyEngine->pythonRunFunction(func, m_callbackArgs, true);
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
            PyErr_PrintEx(0);
            PyErr_Clear();
        }
    }
    else
    {
        PyObject *func = PyWeakref_GetObject(m_function);
        PyObject *inst = PyWeakref_GetObject(m_boundedInstance);

        if((func == NULL) || (func == Py_None) || (inst == Py_None))
        {
            PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
            PyErr_PrintEx(0);
            PyErr_Clear();
        }
        else
        {

            PyObject *method = PyMethod_New(func, inst); //new ref

            if(debug)
            {
                pyEngine->pythonDebugFunction(method, m_callbackArgs, true);
            }
            else
            {
                pyEngine->pythonRunFunction(method, m_callbackArgs, true);
            }

            Py_XDECREF(method);
        }
    }

    PyGILState_Release(state);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonTimer::PyTimer_dealloc(PyTimer* self)
{
    if (self->timer)
    {
        self->timer->stop();
        DELETE_AND_SET_NULL(self->timer);
    }
    if (self->callbackFunc)
    {
        Py_XDECREF(self->callbackFunc->m_callbackArgs);
        DELETE_AND_SET_NULL(self->callbackFunc);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonTimer::PyTimer_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyTimer* self = (PyTimer *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->timer = NULL;
        self->callbackFunc = 0;
    }
    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerInit_doc,"timer(interval, callbackFunc [, argTuple, singleShot]) -> new callback timer \n\
\n\
Creates a timer object that continuously calls a python callback function or method with a certain interval. The timer is active after construction and \n\
stops when this instance is destroyed or stop() is called. \n\
\n\
Parameters \n\
----------- \n\
interval : {int} \n\
    time out interval in ms \n\
callbackFunc: {function, method} \n\
    Python function that should be called when timer event raises \n\
argTuple: {tuple}, optional \n\
    tuple of parameters passed as arguments to the callback function \n\
singleShot: {bool}, optional \n\
    defines if this timer only fires one time after its start (True) or continuously (False, default)");
int PythonTimer::PyTimer_init(PyTimer *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"interval", "callbackFunc", "argTuple", "singleShot", NULL};

    if(args == NULL || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    PyObject *tempObj = NULL;
    self->callbackFunc = new TimerCallback();
    int timeOut = -1;
    unsigned char singleShot = 0;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "iO|O!b", const_cast<char**>(kwlist), &timeOut, &tempObj, &PyTuple_Type, &self->callbackFunc->m_callbackArgs, &singleShot))
    {
        return -1;
    }

    if (timeOut < 0)
    {
        PyErr_SetString(PyExc_TypeError, "minimum timeout is 0 ms (immediate fire).");
        DELETE_AND_SET_NULL(self->callbackFunc);
        return -1;
    }

    if (self->callbackFunc->m_callbackArgs)
    {
        Py_INCREF(self->callbackFunc->m_callbackArgs);
    }
    else
    {
        self->callbackFunc->m_callbackArgs = PyTuple_New(0); //if the callback function of the timeout event is debugged, it must not get a NULL object but at least an empty tuple!
    }

    PyObject *temp = NULL;
    if(PyMethod_Check(tempObj))
    {
        self->callbackFunc->m_boundedMethod = true;
        Py_XDECREF(self->callbackFunc->m_boundedInstance);
        Py_XDECREF(self->callbackFunc->m_function);
        temp = PyMethod_Self(tempObj); //borrowed
        self->callbackFunc->m_boundedInstance = PyWeakref_NewRef(temp, NULL); //new ref (weak reference used to avoid cyclic garbage collection)
        temp = PyMethod_Function(tempObj); //borrowed
        self->callbackFunc->m_function = PyWeakref_NewRef(temp, NULL); //new ref
    }
    else if(PyFunction_Check(tempObj))
    {
        self->callbackFunc->m_boundedMethod = false;
        Py_XDECREF(self->callbackFunc->m_boundedInstance);
        Py_XDECREF(self->callbackFunc->m_function);
        self->callbackFunc->m_function = PyWeakref_NewRef(tempObj, NULL); //new ref
    }
    else
    {
        Py_XDECREF(self->callbackFunc->m_callbackArgs);
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        DELETE_AND_SET_NULL(self->callbackFunc);
        return -1;
    }

    self->timer = new QTimer();
    self->timer->setInterval(timeOut);
#if QT_VERSION >= 0x050000
    QMetaObject::Connection conn = QObject::connect(self->timer, SIGNAL(timeout()), self->callbackFunc, SLOT(timeout()));
#else
    int conn = QObject::connect(self->timer, SIGNAL(timeout()), self->callbackFunc, SLOT(timeout()));
#endif
    if (!conn)
    {
        DELETE_AND_SET_NULL(self->timer);
        Py_XDECREF(self->callbackFunc->m_callbackArgs);
        PyErr_SetString(PyExc_TypeError, "error connecting timeout signal/slot");
        DELETE_AND_SET_NULL(self->callbackFunc);
        return -1;
    }

    self->timer->setSingleShot(singleShot > 0);
    self->timer->start();
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonTimer::PyTimer_repr(PyTimer *self)
{
    PyObject *result;
    if(self->timer == 0)
    {
        result = PyUnicode_FromFormat("timer(empty)");
    }
    else
    {
        if (self->timer->isSingleShot())
        {
            result = PyUnicode_FromFormat("timer(interval %i ms, single shot)", self->timer->interval());
        }
        else
        {
            result = PyUnicode_FromFormat("timer(interval %i ms)", self->timer->interval());
        }
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerStart_doc,"start() -> starts timer\n\
\n\
Starts or restarts the timer with its timeout interval. If the timer is already running, it will be stopped and restarted.");
PyObject* PythonTimer::PyTimer_start(PyTimer *self) 
{ 
    if (self->timer) 
        self->timer->start(); 
    Py_RETURN_NONE; 
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerStop_doc,"stop() -> stops timer");
PyObject* PythonTimer::PyTimer_stop(PyTimer *self) 
{ 
    if (self->timer) 
        self->timer->stop(); 
    Py_RETURN_NONE; 
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerIsActive_doc,"isActive() -> returns timer status\n\
\n\
Returns \n\
------- \n\
status : {bool} \n\
    True if the timer is running, otherwise False.");
PyObject* PythonTimer::PyTimer_isActive(PyTimer *self)
{ 
    if (self->timer) 
    {
        if( self->timer->isActive() )
        {
            Py_RETURN_TRUE;
        }
        else
        {
            Py_RETURN_FALSE;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "timer is not available.");
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerSetInterval_doc,"setInterval(interval) -> sets timer interval in [ms]\n\
\n\
This method sets the timeout interval in milliseconds. The timer calls the callback function continuously after this interval (if started) \n\
\n\
Parameters \n\
----------- \n\
interval : {int} \n\
    timeout interval in milliseconds. The callback function is continuously called after this timeout once the timer is started.");
PyObject* PythonTimer::PyTimer_setInterval(PyTimer *self, PyObject *args)
{ 
    int timeout; 
    if(!PyArg_ParseTuple(args, "i", &timeout))
    {
        return NULL;
    } 

    if (self->timer) self->timer->setInterval(timeout);
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonTimer::PyTimer_methods[] = {
        {"start", (PyCFunction)PyTimer_start, METH_NOARGS, PyTimerStart_doc},
        {"stop", (PyCFunction)PyTimer_stop, METH_NOARGS, PyTimerStop_doc},
        {"isActive", (PyCFunction)PyTimer_isActive, METH_NOARGS, PyTimerIsActive_doc},
        {"setInterval", (PyCFunction)PyTimer_setInterval, METH_VARARGS, PyTimerSetInterval_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonTimer::PyTimer_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonTimer::PyTimerModule = {
        PyModuleDef_HEAD_INIT,
        "timer",
        "timer for callback function",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonTimer::PyTimer_getseters[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonTimer::PyTimerType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.timer",             /* tp_name */
        sizeof(PyTimer),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyTimer_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyTimer_repr,         /* tp_repr */
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
        PyTimerInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,            /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyTimer_methods,             /* tp_methods */
        PyTimer_members,             /* tp_members */
        PyTimer_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyTimer_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyTimer_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

} //end namespace ito

//----------------------------------------------------------------------------------------------------------------------------------