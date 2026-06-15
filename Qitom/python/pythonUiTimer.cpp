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

#include "pythonUiTimer.h"

#include "structmember.h"
#include "../global.h"
#include "../organizer/uiOrganizer.h"
#include "pythonEngineInc.h"
#include "AppManagement.h"
#include <iostream>
#include <qvector.h>
#include <qtimer.h>

namespace ito
{

//-------------------------------------------------------------------------------------
TimerCallback::TimerCallback() :
    m_function(nullptr),
    m_boundedInstance(nullptr),
    m_callableType(CallableType::Callable_Invalid),
    m_callbackArgs(nullptr)
{
}

//-------------------------------------------------------------------------------------
TimerCallback::~TimerCallback()
{
    Py_XDECREF(m_function);
    Py_XDECREF(m_boundedInstance);
    Py_XDECREF(m_callbackArgs);
}

//-------------------------------------------------------------------------------------
void TimerCallback::timeout()
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (Py_IsInitialized() == 0)
    {
        qDebug("python is not available any more");
        return;
    }

    if (m_function == nullptr)
    {
        qDebug("invalid callable slot.");
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    bool debug = false;

    if (pyEngine)
    {
        debug = pyEngine->execInternalCodeByDebugger();
    }

    if (m_callableType == CallableType::Callable_Function)
    {
        if (debug)
        {
            pyEngine->pythonDebugFunction(m_function, m_callbackArgs, true);
        }
        else
        {
            pyEngine->pythonRunFunction(m_function, m_callbackArgs, true);
        }
    }
    else if (m_callableType == CallableType::Callable_Method)
    {
        PyObject *func = PyWeakref_GetObject(m_function);
        PyObject *inst = PyWeakref_GetObject(m_boundedInstance);

        if (func == Py_None || inst == Py_None)
        {
            PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
            PyErr_PrintEx(0);
            PyErr_Clear();
        }
        else
        {
            PyObject *method = PyMethod_New(func, inst); //new ref

            if (debug)
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
    else if (m_callableType == CallableType::Callable_CFunction)
    {
        PyCFunctionObject* cfunc = (PyCFunctionObject*)m_function;
        PyObject *method = PyCFunction_NewEx(cfunc->m_ml, cfunc->m_self, NULL);

        if (method)
        {
            if (debug)
            {
                pyEngine->pythonDebugFunction(method, m_callbackArgs, true);
            }
            else
            {
                pyEngine->pythonRunFunction(method, m_callbackArgs, true);
            }
        }

        Py_XDECREF(method);
    }
    else
    {
        qDebug("invalid m_callableType.");
    }

    PyGILState_Release(state);
}

//-------------------------------------------------------------------------------------
void PythonTimer::PyTimer_dealloc(PyTimer* self)
{
    if (self->timer)
    {
        self->timer->stop();
    }

    self->callbackFunc.clear();
    self->timer.clear();
}

//-------------------------------------------------------------------------------------
PyObject* PythonTimer::PyTimer_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyTimer* self = (PyTimer *)type->tp_alloc(type, 0);

    if (self != nullptr)
    {
        self->timer.clear();
        self->callbackFunc.clear();
    }

    return (PyObject *)self;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerInit_doc,"timer(interval, callbackFunc, argTuple = (), singleShot = False, name = \"\", startAfterInit = True) -> timer \n\
\n\
Creates a new timer object for (continously) triggering a callback function \n\
\n\
Creates a timer object that (continuously) calls a python callback function or method. \n\
The timer is active right from the beginning, hence, after creating this object. \n\
If ``singleShot`` is ``True``, the callback function is triggered once after the \n\
interval is passed (denoted as timeout). Else, the callback is continuously triggered \n\
with the given interval. \n\
\n\
Please note, that the timer objects may time out later than expected if Python is \n\
currently busy or the operating system is unable to provide the requested accuracy. \n\
In such a case of timeout overrun, the callback function is only triggered once, \n\
even if multiple timeouts have expired, and then will resume the original interval. \n\
\n\
An active timer can be stopped by the :meth:`stop` method, or if this object is \n\
deleted. Furthermore, itom provides the :ref:`gui-timermanager` dialog, where \n\
all or selected timers can be started or stopped. \n\
\n\
**New in itom 4.1**: Added optional ``startAfterInit`` argument. \n\
\n\
Parameters \n\
---------- \n\
interval : int \n\
    Time out interval in ms. \n\
callbackFunc : callable \n\
    Python method (bounded) or function (unbounded) that should be called whenever \n\
    the timer event raises. \n\
argTuple : tuple, optional \n\
    Tuple of parameters passed as arguments to the callback function. \n\
singleShot : bool, optional \n\
    Defines if this timer only fires one time after its start (``True``) or \n\
    continuously (``False``, default). \n\
name : str, optional \n\
    Is the optional name of this timer. This name is displayed in the timer \n\
    manager dialog (instead of the timer ID, if no name is given. \n\
startAfterInit : bool, optional \n\
    If this optional boolean is set to False the timer will not start after initialization. \n\
    The timer can be started later by using timer.start(). \n\
\n\
Examples \n\
-------- \n\
>>> import time \n\
... \n\
... def callbackFunc(startTime: float, a: int): \n\
...     print(\"%.2f sec elapsed: %i\" % (time.time() - startTime, a)) \n\
... \n\
... myTimer = timer(1000, callbackFunc, argTuple = (time.time(), 25)) \n\
\n\
1.00 sec elapsed: 25 \n\
2.01 sec elapsed : 25 \n\
3.01 sec elapsed : 25 \n\
4.01 sec elapsed : 25");
int PythonTimer::PyTimer_init(PyTimer *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"interval", "callbackFunc", "argTuple", "singleShot", "name", "startAfterInit", nullptr };

    if(args == nullptr || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    PyObject *callable = nullptr;
    self->callbackFunc = QSharedPointer<TimerCallback>(new TimerCallback());
    int timeOut = -1;
    unsigned char singleShot = 0;
    const char* name = nullptr;
    unsigned char startAfterInit = 1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "iO|O!bsb", const_cast<char**>(kwlist),
        &timeOut,
        &callable,
        &PyTuple_Type, &self->callbackFunc->m_callbackArgs,
        &singleShot,
        &name,
        &startAfterInit))
    {
        return -1;
    }

    if (timeOut < 0)
    {
        PyErr_SetString(PyExc_TypeError, "minimum timeout is 0 ms (immediate fire).");
        return -1;
    }

    if (self->callbackFunc->m_callbackArgs)
    {
        Py_INCREF(self->callbackFunc->m_callbackArgs);
    }
    else
    {
        // if the callback function of the timeout event is debugged,
        // it must not get a NULL object but at least an empty tuple!
        self->callbackFunc->m_callbackArgs = PyTuple_New(0);
    }

    PyObject *temp = nullptr;

    if (PyMethod_Check(callable))
    {
        self->callbackFunc->m_callableType = TimerCallback::CallableType::Callable_Method;
        PyObject *temp = PyMethod_Self(callable); //borrowed
        self->callbackFunc->m_boundedInstance = PyWeakref_NewRef(temp, nullptr); //new ref (weak reference used to avoid cyclic garbage collection)
        temp = PyMethod_Function(callable); //borrowed
        self->callbackFunc->m_function = PyWeakref_NewRef(temp, nullptr); //new ref
    }
    else if (PyCFunction_Check(callable))
    {
        self->callbackFunc->m_callableType = TimerCallback::CallableType::Callable_CFunction;
        Py_INCREF(callable);
        self->callbackFunc->m_function = callable; //new ref
    }
    else if (PyCallable_Check(callable))
    {
        // any other callable, especially PyFunction, but also functools.partial etc.
        self->callbackFunc->m_callableType = TimerCallback::CallableType::Callable_Function;
        Py_INCREF(callable);
        self->callbackFunc->m_function = callable; //new ref
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return -1;
    }

    self->timer = QSharedPointer<QTimer>(new QTimer());
    self->timer->setInterval(timeOut);

    QMetaObject::Connection conn = QObject::connect(
        self->timer.data(), &QTimer::timeout,
        self->callbackFunc.data(), &TimerCallback::timeout);

    if (!conn)
    {
        PyErr_SetString(PyExc_TypeError, "error connecting timeout signal/slot");
        return -1;
    }

    self->timer->setSingleShot(singleShot > 0);
    self->timer->start();

    if (self->timer->timerId() < 0 )
    {
        if (PyErr_WarnEx(
                PyExc_RuntimeWarning,
                "timer object could not be created (e.g. negative interval, timer can only "
                "be used in threads started with QThread or timers cannot be started "
                "from another thread)",
                1) == -1
            )
        {
            // exception is raised instead of warning (depending on user defined warning levels)
            return -1;
        }
    }
    else
    {
        //send timer to timer dialog of main window
        UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
        QWeakPointer<QTimer> qTimerPtr = self->timer.toWeakRef();
        QString nameStr;

        if (name != nullptr)
        {
            nameStr = QString(name);
        }

        QMetaObject::invokeMethod(
            uiOrg,
            "registerActiveTimer",
            Q_ARG(QWeakPointer<QTimer>, qTimerPtr),
            Q_ARG(QString, nameStr)
        );
    }

    if (!startAfterInit)
    {
        self->timer->stop();
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyObject* PythonTimer::PyTimer_repr(PyTimer *self)
{
    PyObject *result = nullptr;

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

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerStart_doc,"start() \n\
\n\
Starts the timer. \n\
\n\
This method starts or restarts the timer with its timeout interval. \n\
If the timer is already running, it will be stopped and restarted. \n\
\n\
See Also \n\
-------- \n\
isActive, stop");
PyObject* PythonTimer::PyTimer_start(PyTimer *self)
{
    if (self->timer)
    {
        self->timer->start();
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerStop_doc, "stop() \n\
\n\
Stops the timer. \n\
\n\
This method stop the timer (if currently active). \n\
\n\
See Also \n\
-------- \n\
isActive, start");
PyObject* PythonTimer::PyTimer_stop(PyTimer *self)
{
    if (self->timer)
    {
        self->timer->stop();
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerIsActive_doc,"isActive() -> bool \n\
\n\
Indicates if the timer is currently active. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if the timer is running, otherwise ``False``.");
PyObject* PythonTimer::PyTimer_isActive(PyTimer *self)
{
    if (self->timer)
    {
        if(self->timer->isActive())
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
        return nullptr;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyTimerSetInterval_doc,"setInterval(interval) \n\
\n\
Sets the timer interval in ms. \n\
\n\
This method sets the timeout interval in milliseconds. If the timer is started, \n\
the callback function is tried to be continously triggered whenever the interval \n\
expired. \n\
\n\
Parameters \n\
---------- \n\
interval : int \n\
    Timeout interval in milliseconds. \n\
\n\
Notes \n\
----- \n\
If Python is currently busy, a timer event can also be triggered at a later time, \n\
if the same trigger event is not already in the execution queue.");
PyObject* PythonTimer::PyTimer_setInterval(PyTimer *self, PyObject *args)
{
    int timeout;

    if(!PyArg_ParseTuple(args, "i", &timeout))
    {
        return nullptr;
    }

    if (self->timer)
    {
        self->timer->setInterval(timeout);
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyMethodDef PythonTimer::PyTimer_methods[] = {
        {"start", (PyCFunction)PyTimer_start, METH_NOARGS, PyTimerStart_doc},
        {"stop", (PyCFunction)PyTimer_stop, METH_NOARGS, PyTimerStop_doc},
        {"isActive", (PyCFunction)PyTimer_isActive, METH_NOARGS, PyTimerIsActive_doc},
        {"setInterval", (PyCFunction)PyTimer_setInterval, METH_VARARGS, PyTimerSetInterval_doc},
        {nullptr}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMemberDef PythonTimer::PyTimer_members[] = {
        {nullptr}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonTimer::PyTimerModule = {
        PyModuleDef_HEAD_INIT,
        "timer",
        "timer for callback function",
        -1,
        nullptr, nullptr, nullptr, nullptr, nullptr
};

//-------------------------------------------------------------------------------------
PyGetSetDef PythonTimer::PyTimer_getseters[] = {
    {nullptr}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonTimer::PyTimerType = {
        PyVarObject_HEAD_INIT(nullptr, 0)
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

//-------------------------------------------------------------------------------------
