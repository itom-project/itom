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

//#include "../organizer/uiOrganizer.h"
//#include "../organizer/addInManager.h"
//
//#include "pythonQtConversion.h"
//
//#include <qmap.h>
//#include <qsharedpointer.h>
//#include <qmessagebox.h>
//#include <qmetaobject.h>
//#include <qcoreapplication.h>

#include <iostream>
#include <qvector.h>

#include "pythonEngineInc.h"
#include "AppManagement.h"

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
//    PyGILState_STATE state = PyGILState_Ensure();

    bool debug = false;
    if(pyEngine)
    {
        debug = pyEngine->execInternalCodeByDebugger();
    }

//    PyObject *argTuple = PyTuple_New(m_argTypeList.size());
//    PyObject *temp = NULL;

    //arguments[0] is return argument
/*
    for(int i=0;i<m_argTypeList.size();i++)
    {
        temp = PythonQtConversion::ConvertQtValueToPythonInternal(m_argTypeList[i],arguments[i+1]); //new reference
        if(temp)
        {
            PyTuple_SetItem(argTuple,i,temp); //steals reference
        }
    }
*/
    if(m_boundedMethod == false)
    {
        PyObject *func = PyWeakref_GetObject(m_function);
        if((func != NULL) && (func != Py_None))
        {
            if(debug)
            {
                pyEngine->pythonDebugFunction(func, m_callbackArgs);
            }
            else
            {
                pyEngine->pythonRunFunction(func, m_callbackArgs);
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
            PyErr_Print();
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
            PyErr_Print();
            PyErr_Clear();
        }
        else
        {

            PyObject *method = PyMethod_New(func, inst); //new ref

            if(debug)
            {
                pyEngine->pythonDebugFunction(method, m_callbackArgs);
            }
            else
            {
                pyEngine->pythonRunFunction(method, m_callbackArgs);
            }

            Py_XDECREF(method);
        }
    }

   /* if (m_callbackArgs)
        Py_DECREF(m_callbackArgs);*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonUiTimer::PyUiTimer_dealloc(PyUiTimer* self)
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
PyObject* PythonUiTimer::PyUiTimer_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyUiTimer* self = (PyUiTimer *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->timer = NULL;
        self->callbackFunc = 0;
    }
    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerInit_doc,"uiTimer(timeOut, callbackFunc, [args]) -> new callback timer \n\
                                Parameters: \n\
                                timeOut [int]: time out in ms \n\
                                callbackFunc: Python function that should be called when timer event raises \n\
                                args: parameters for Python function");
int PythonUiTimer::PyUiTimer_init(PyUiTimer *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"timeOut", "callbackFunc", "parameters", NULL};

    if(args == NULL || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    PyObject *tempObj = NULL;
    self->callbackFunc = new TimerCallback();
    double timeOut = -1;
    if(!PyArg_ParseTupleAndKeywords(args, kwds, "dO|O!", const_cast<char**>(kwlist), &timeOut, &tempObj, &PyTuple_Type, &self->callbackFunc->m_callbackArgs))
    {
        PyErr_Format(PyExc_TypeError,"Arguments do not fit to required list of arguments");
        return -1;
    }
    if (timeOut < 1)
    {
        PyErr_Format(PyExc_TypeError, "minimum timeout is 1ms.");
        DELETE_AND_SET_NULL(self->callbackFunc);
        return -1;
    }

    Py_XINCREF(self->callbackFunc->m_callbackArgs);

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
        PyErr_Format(PyExc_TypeError, "given method reference is not callable.");
        delete self->callbackFunc;
        return -1;
    }

    self->timer = new QTimer();
    self->timer->setInterval(timeOut);
    int ret;
    if (!(ret=QObject::connect(self->timer, SIGNAL(timeout()), self->callbackFunc, SLOT(timeout()))))
        return -1;
    self->timer->start();
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiTimer::PyUiTimer_repr(PyUiTimer *self)
{
    PyObject *result;
    if(self->timer == 0)
    {
        result = PyUnicode_FromFormat("UiTimer(empty)");
    }
    else
    {
        result = PyUnicode_FromFormat("UiTimer(timeOut %d, callbackFunc %s)", self->timer->interval());
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerStart_doc,"start() -> starts uiTimer\n\
\n\
Notes \n\
----- \n\
Starts uiTimer.\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonUiTimer::PyUiTimer_start(PyUiTimer *self) 
{ 
    if (self->timer) 
        self->timer->start(); 
    Py_RETURN_NONE; 
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerStop_doc,"stop() -> stopps uiTimer\n\
\n\
Notes \n\
----- \n\
Stopps uiTimer.\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonUiTimer::PyUiTimer_stop(PyUiTimer *self) 
{ 
    if (self->timer) 
        self->timer->stop(); 
    Py_RETURN_NONE; 
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerIsActive_doc,"isActive() -> returns timer status\n\
\n\
Returns \n\
------- \n\
status : {bool} \n\
\n\
Notes \n\
----- \n\
doctodo\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonUiTimer::PyUiTimer_isActive(PyUiTimer *self)
{ 
    if (self->timer) 
        return PyLong_FromLong((long)self->timer->isActive());
    else
        return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerSetInterval_doc,"setInterval(interval) -> sets timer interval in [ms]\n\
\n\
Parameters \n\
----------- \n\
interval : {int}\n\
    interval in ms\n\
\n\
Returns \n\
------- \n\
\n\
Notes \n\
----- \n\
sets the uiTimerinterval in ms.\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonUiTimer::PyUiTimer_setInterval(PyUiTimer *self, PyObject *args)
{ 
    int timeout; 
    if(!PyArg_ParseTuple(args, "i", &timeout))
    {
        PyErr_SetString(PyExc_RuntimeError, "Parameter modal must be a boolean value");
        return NULL;
    } 
    if (self->timer) self->timer->setInterval(timeout);
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonUiTimer::PyUiTimer_methods[] = {
        {"start", (PyCFunction)PyUiTimer_start, METH_NOARGS, pyUiTimerStart_doc},
        {"stop", (PyCFunction)PyUiTimer_stop, METH_NOARGS, pyUiTimerStop_doc},
        {"isActive", (PyCFunction)PyUiTimer_isActive, METH_NOARGS, pyUiTimerIsActive_doc},
        {"setInterval", (PyCFunction)PyUiTimer_setInterval, METH_VARARGS, pyUiTimerSetInterval_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonUiTimer::PyUiTimer_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonUiTimer::PyUiTimerModule = {
        PyModuleDef_HEAD_INIT,
        "uiTimer",
        "timer for callback function",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonUiTimer::PyUiTimer_getseters[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonUiTimer::PyUiTimerType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.uiTimer",             /* tp_name */
        sizeof(PyUiTimer),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyUiTimer_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyUiTimer_repr,         /* tp_repr */
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
        pyUiTimerInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,		               /* tp_traverse */
        0,		               /* tp_clear */
        0,            /* tp_richcompare */
        0,		               /* tp_weaklistoffset */
        0,		               /* tp_iter */
        0,		               /* tp_iternext */
        PyUiTimer_methods,             /* tp_methods */
        PyUiTimer_members,             /* tp_members */
        PyUiTimer_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyUiTimer_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyUiTimer_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

} //end namespace ito

//----------------------------------------------------------------------------------------------------------------------------------