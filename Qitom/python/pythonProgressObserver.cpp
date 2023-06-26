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

#include "pythonProgressObserver.h"

#include "../global.h"
#include <qsharedpointer.h>
#include "pythonUi.h"
#include "../organizer/uiOrganizer.h"
#include "../AppManagement.h"
#include "pythonQtConversion.h"
#include "pythonQtSignalMapper.h"


//---------------------------------------------------------------------------------------------

namespace ito
{

//---------------------------------------------------------------------------------------------
void PythonProgressObserver::PyProgressObserver_addTpDict(PyObject * tp_dict)
{
}

//---------------------------------------------------------------------------------------------
void PythonProgressObserver::PyProgressObserver_dealloc(PyProgressObserver* self)
{
    DELETE_AND_SET_NULL(self->progressObserver);
    DELETE_AND_SET_NULL(self->signalMapper);

    Py_TYPE(self)->tp_free((PyObject*)self);
};

//---------------------------------------------------------------------------------------------
PyObject* PythonProgressObserver::PyProgressObserver_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyProgressObserver* self = (PyProgressObserver*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->progressObserver = NULL;
        self->signalMapper = NULL;
    }

    return (PyObject *)self;
};

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyProgressObserver_doc,"progressObserver(progressBar = None, label = None, progressMinimum = 0, progressMaximum = 100) -> progressObserver \n\
\n\
Creates a progressObserver object. \n\
\n\
A :class:`progressObserver` object can be passed to functions, that might need some \n\
time to be finished, such that these functions can regularily report their current \n\
progress (as number as well as text) via this progress observer. These reported progress \n\
values are then displayed in the passed ``progressBar`` and / or ``label``. \n\
For more information see also this section: :ref:`filter_interruptible`. \n\
\n\
Target functions, that can make use of this :class:`progressObserver` can be contained in \n\
itom algorithm plugins. However these functions must implement the **FilterDefExt** \n\
interface, which is available from itom 3.3 on. Check the method :py:meth:`itom.filterHelp` \n\
or the help widget of itom in order to find out whether a filter \n\
in an algorithm plugin has this ability. \n\
\n\
If a filter accepts a :class:`progressObserver`, pass this object to the keyword \n\
argument ``_observe`` of the method :py:meth:`itom.filter`. Algorithms, that accept \n\
this kind of observer can also use the same observer to interrupt the algorithm once \n\
the additional interrupt flag of the observer is set. This flag is either set whenever \n\
a Python script execution is interrupted or if a signal of a widget has been emitted that \n\
was previously connected to this interrupt flag using the method \n\
:py:meth:`~itom.uiItem.invokeProgressObserverCancellation`. \n\
\n\
Parameters \n\
---------- \n\
progressBar : uiItem, optional \n\
    This is an optional handle to a progress bar in any user interface. The minimum \n\
    requirement is that the given widget has at least a slot 'setValue(int)', which \n\
    is called once this progress observer reports a new progress value (bound between \n\
    ``progressMinimum`` and ``progressMaximum``. \n\
label : uiItem, optional \n\
    This argument is very similar to ``progressBar``, however it requires a handle to a label \n\
    widget or any other widget that has a slot ``setText(QString)``. This slot is called \n\
    whenever the target algorithm for this observer reports a new progress text. \n\
progressMinimum : int, optional \n\
    Minimum progress value that should be used and reported by the target of this observer. \n\
progressMaximum : int, optional \n\
    Maximum progress value that should be used and reported by the target of this observer. \n\
\n\
Notes \n\
----- \n\
This class wraps the C++ class `ito::FunctionCancellationAndObserver`.");
int PythonProgressObserver::PyProgressObserver_init(PyProgressObserver *self, PyObject *args, PyObject * kwds)
{
    PyObject *progressBar = NULL;
    PyObject *label = NULL;
    int progressMinimum = 0;
    int progressMaximum = 100;

    const char *kwlist[] = {"progressBar", "label", "progressMinimum", "progressMaximum", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O!O!ii", const_cast<char**>(kwlist), &PythonUi::PyUiItemType, &progressBar, &PythonUi::PyUiItemType, &label, &progressMinimum, &progressMaximum))
    {
        return -1;
    }

    self->progressObserver = new QSharedPointer<ito::FunctionCancellationAndObserver>(new ito::FunctionCancellationAndObserver(progressMinimum, progressMaximum));

    PythonUi::PyUiItem *p = (PythonUi::PyUiItem*)progressBar;
    PythonUi::PyUiItem *l = (PythonUi::PyUiItem*)label;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();

    QSharedPointer<unsigned int> objectID(new unsigned int);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());


    QMetaObject::invokeMethod(uiOrg, "connectWidgetsToProgressObserver", Q_ARG(bool, p != NULL), Q_ARG(unsigned int, p ? p->objectID : 0),
        Q_ARG(bool, l != NULL), Q_ARG(unsigned int, l ? l->objectID : 0),
        Q_ARG(QSharedPointer<ito::FunctionCancellationAndObserver>, *(self->progressObserver)),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while connecting widgets to progressObserver");
        return -1;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return -1;
    }

    DELETE_AND_SET_NULL(self->signalMapper);
    self->signalMapper = new PythonQtSignalMapper();

    return 0;
};

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonProgressObserver::PyProgressObserver_repr(PyProgressObserver *self)
{
    PyObject *result = PyUnicode_FromFormat("progressObserver()");

    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_getProgressMinimum_doc,
"int : Gets the minimum value of the progress. \n\
\n\
The minimum progress value is the minimum scalar value that the observed \n\
function or algorithm should set as its lowest progress value.");
PyObject* PythonProgressObserver::PyProgressObserver_getProgressMinimum(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PyLong_FromLong((*(self->progressObserver))->progressMinimum());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_getProgressMaximum_doc,
"int : Gets the maximum value of the progress. \n\
\n\
The maximum progress value is the maximum scalar value that the observed \n\
function or algorithm should set as its highest progress value.");
PyObject* PythonProgressObserver::PyProgressObserver_getProgressMaximum(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PyLong_FromLong((*(self->progressObserver))->progressMaximum());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_progressValue_doc,
"int : gets or sets the current progress value.\n\
\n\
If the current progress value is set, the signal ``progressValueChanged(int)`` \n\
is emitted. It can for instance be connected to a ``setValue`` slot of a \n\
`QProgressBar`. The ``progressValue`` will be clipped to ``progressMinimum`` \n\
and  ``progressMaximum``.");
PyObject* PythonProgressObserver::PyProgressObserver_getProgressValue(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PyLong_FromLong((*(self->progressObserver))->progressValue());
}

//-------------------------------------------------------------------------------------
int PythonProgressObserver::PyProgressObserver_setProgressValue(PyProgressObserver *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return -1;
    }

    bool ok;
    int val = PythonQtConversion::PyObjGetInt(value, false, ok);
    if (ok)
    {
        (*(self->progressObserver))->setProgressValue(val);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Progress value must be an integer.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_progressText_doc, "str : the current progress text\n\
\n\
This attribute gives access to the current progress text.\n\
When set, the signal ``progressTextChanged`` is emitted. It can for instance be\n\
connected to a ``setText`` slot of a `QLabel`. The text should inform about \n\
the step, the long-running method is currently executing.");
PyObject* PythonProgressObserver::PyProgressObserver_getProgressText(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PythonQtConversion::QStringToPyObject((*(self->progressObserver))->progressText());
}

//-------------------------------------------------------------------------------------
int PythonProgressObserver::PyProgressObserver_setProgressText(PyProgressObserver *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return -1;
    }

    bool ok;
    QString text = PythonQtConversion::PyObjGetString(value, false, ok);
    if (ok)
    {
        (*(self->progressObserver))->setProgressText(text);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Error interpreting the progress text as string.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_isCancelled_doc,
"bool : returns ``True`` if a cancellation request has been signalled, otherwise ``False``.");
PyObject* PythonProgressObserver::PyProgressObserver_isCancelled(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PyBool_FromLong((*(self->progressObserver))->isCancelled());
}

//---------------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_requestCancellation_doc, "requestCancellation() \n\
\n\
Requests the cancellation of the filter.\n\
\n\
If this :class:`progressObserver` is currently passed to an object, filter or \n\
algorithm, that can be cancelled, a cancellation request is sent to this object. \n\
Calling this method will emit the ``cancellationRequested()`` signal.");
PyObject* PythonProgressObserver::PyProgressObserver_requestCancellation(PyProgressObserver *self /*self*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    (*(self->progressObserver))->requestCancellation(ito::FunctionCancellationAndObserver::ReasonGeneral);
    Py_RETURN_NONE;
};

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_reset_doc, "reset() \n\
\n\
Resets this object. \n\
\n\
Resets this object and empties the current progress text, resets the current \n\
progress value to its minimum and resets the cancellation request. \n\
Emits the ``resetDone`` signal.");
PyObject* PythonProgressObserver::PyProgressObserver_reset(PyProgressObserver *self)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    (*(self->progressObserver))->reset();
    Py_RETURN_NONE;
};

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_connect_doc, "connect(signalSignature, callableMethod, minRepeatInterval = 0)\n\
\n\
Connects the signal of the progressObserver with the given callable Python method. \n\
\n\
This object of :class:`progressObserver` wraps an underlying object of the C++ class \n\
``ito::FunctionCancellationAndObserver``, which can emit various signals. Use this \n\
method to connect any signal to any callable python method (bounded or unbounded). This \n\
method must have the same number of arguments than the signal and the types of the \n\
signal definition must be convertable into a python object. \n\
\n\
Possible signals are (among others): \n\
\n\
* progressTextChanged(QString) -> emitted when the observed function reports a new progress text, \n\
* progressValueChanged(int) -> emitted whenever the observed function reports a new progress value, \n\
* cancellationRequested() -> emitted if a cancellation of the observed function has been requested, \n\
* resetDone() -> emitted if the progressObserver has been reset. \n\
\n\
New in itom 4.1. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature. Possible signatures are: \n\
    ``progressTextChanged(QString)`` or ``progressValueChanged(int)``\n\
callableMethod : callable \n\
    Valid method or function that is called if the signal is emitted. The method must \n\
    provide one parameter for the string or number argument of the signal. \n\
minRepeatInterval : int, optional \n\
    If > 0, the same signal only invokes a slot once within the given interval (in ms). \n\
    Default: 0 (all signals will invoke the callable Python method. \n\
\n\
See Also \n\
-------- \n\
disconnect");
PyObject* PythonProgressObserver::PyProgressObserver_connect(PyProgressObserver *self, PyObject* args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", "minRepeatInterval", NULL };
    const char* signalSignature;
    PyObject *callableMethod;
    int signalIndex;
    int tempType;
    IntList argTypes;
    int minRepeatInterval = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|i", const_cast<char**>(kwlist), &signalSignature, &callableMethod, &minRepeatInterval))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }

    if (!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    if (!self || self->progressObserver == NULL || self->progressObserver->isNull())
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    ito::FunctionCancellationAndObserver *fcao = self->progressObserver->data();

    QByteArray signature(signalSignature);
    const QMetaObject *mo = fcao->metaObject();
    signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature));
    QMetaMethod metaMethod = mo->method(signalIndex);
    QList<QByteArray> names = metaMethod.parameterTypes();

    foreach(const QByteArray& name, names)
    {
        tempType = QMetaType::type(name.constData());

        if (tempType > 0)
        {
            argTypes.append(tempType);
        }
        else
        {
            QString msg = QString("parameter type %1 is unknown").arg(name.constData());
            PyErr_SetString(PyExc_RuntimeError, msg.toLatin1().data());
            signalIndex = -1;
            return NULL;
        }
    }
    if (self->signalMapper)
    {
        if (!self->signalMapper->addSignalHandler(fcao, signalSignature, signalIndex, callableMethod, argTypes, minRepeatInterval))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No signalMapper for this progressObserver could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_disconnect_doc, "disconnect(signalSignature, callableMethod) \n\
\n\
Disconnects a connection which must have been established with exactly the same parameters.\n\
\n\
New in itom 4.1. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature (``progressTextChanged(QString)`` or ``progressValueChanged(int)``)\n\
callableMethod : callable \n\
    valid method or function, that should not be called any more, if the given signal is \n\
    emitted. \n\
\n\
See Also \n\
-------- \n\
connect");
PyObject* PythonProgressObserver::PyProgressObserver_disconnect(PyProgressObserver *self, PyObject* args, PyObject* kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", NULL };
    int signalIndex;
    const char* signalSignature;
    PyObject *callableMethod;
    IntList argTypes;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", const_cast<char**>(kwlist), &signalSignature, &callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return
            NULL;
    }
    if (!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    if (!self || self->progressObserver == NULL || self->progressObserver->isNull())
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    ito::FunctionCancellationAndObserver *fcao = self->progressObserver->data();

    const QMetaObject *mo = fcao->metaObject();
    signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature));
    QMetaMethod metaMethod = mo->method(signalIndex);
    if (self->signalMapper)
    {
        if (!self->signalMapper->removeSignalHandler(fcao, signalIndex, callableMethod))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No signalMapper for this progressObserver could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_info_doc, "info(verbose = 0) \n\
\n\
Prints information about possible signals to the command line.\n\
\n\
Parameters \n\
---------- \n\
verbose : int \n\
    0: only signals from the plugin class are printed (default) \n\
    1: all signals from all inherited classes are printed");
PyObject* PythonProgressObserver::PyProgressObserver_info(PyProgressObserver* self, PyObject* args)
{
    int showAll = 0;

    if (!PyArg_ParseTuple(args, "|i", &showAll))
    {
        return NULL;
    }
    if (!self->progressObserver || self->progressObserver->isNull())
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of progressObserver available");
        return NULL;
    }
    //QList<QByteArray> signalSignatureList, slotSignatureList;
    QStringList signalSignatureList;
    const QMetaObject *mo = self->progressObserver->data()->metaObject();
    QMetaMethod metaFunc;
    bool again = true;
    int methodIdx;

    if (showAll == 0 || showAll == 1)
    {
        while (again)
        {
            for (methodIdx = mo->methodOffset(); methodIdx < mo->methodCount(); ++methodIdx)
            {
                metaFunc = mo->method(methodIdx);

                if (metaFunc.methodType() == QMetaMethod::Signal)
                {
                    signalSignatureList.append(metaFunc.methodSignature());

                }
            }

            if (showAll == 1)
            {
                mo = mo->superClass();

                if (mo)
                {
                    again = true;
                    continue;
                }
            }

            again = false;

        }
    }
    else
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Invalid verbose level. Use level 0 to display all signals defined "
            "by the progressObserver itself. Level 1 also displays all inherited signals.");
        return NULL;
    }
    signalSignatureList.sort();

    if (signalSignatureList.length() > 0)
    {
        //QByteArray val;
        QString val;
        QString previous;
        std::cout << "Signals: \n";

        foreach(val, signalSignatureList)
        {
            if (val != previous)
            {
                std::cout << "\t" << QString(val).toLatin1().data() << "\n";
            }
            previous = val;
        }
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyGetSetDef PythonProgressObserver::PyProgressObserver_getseters[] = {
    {"progressMinimum", (getter)PyProgressObserver_getProgressMinimum,  (setter)NULL, progressObserver_getProgressMinimum_doc, NULL},
    {"progressMaximum", (getter)PyProgressObserver_getProgressMaximum,  (setter)NULL, progressObserver_getProgressMaximum_doc, NULL},
    {"progressValue",   (getter)PyProgressObserver_getProgressValue,    (setter)PyProgressObserver_setProgressValue, progressObserver_progressValue_doc, NULL},
    {"progressText",    (getter)PyProgressObserver_getProgressText,     (setter)PyProgressObserver_setProgressText , progressObserver_progressText_doc, NULL },
    {"isCancelled",     (getter)PyProgressObserver_isCancelled,         (setter)NULL, progressObserver_isCancelled_doc, NULL},
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMethodDef PythonProgressObserver::PyProgressObserver_methods[] = {
    {"requestCancellation", (PyCFunction)PythonProgressObserver::PyProgressObserver_requestCancellation, METH_NOARGS, progressObserver_requestCancellation_doc },
    {"reset",               (PyCFunction)PythonProgressObserver::PyProgressObserver_reset, METH_NOARGS, progressObserver_reset_doc },
    {"connect",             (PyCFunction)PythonProgressObserver::PyProgressObserver_connect, METH_VARARGS | METH_KEYWORDS, progressObserver_connect_doc },
    {"disconnect",          (PyCFunction)PythonProgressObserver::PyProgressObserver_disconnect, METH_VARARGS | METH_KEYWORDS, progressObserver_disconnect_doc},
    {"info",                (PyCFunction)PythonProgressObserver::PyProgressObserver_info, METH_VARARGS, progressObserver_info_doc},
    {NULL}  /* Sentinel */
};




//-------------------------------------------------------------------------------------
PyModuleDef PythonProgressObserver::PyProgressObserverModule = {
    PyModuleDef_HEAD_INIT, "progressObserver", "Registers a label and / or progress bar to visualize the progress of a function call within algorithm plugins", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonProgressObserver::PyProgressObserverType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.progressObserver",             /* tp_name */
    sizeof(PyProgressObserver),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyProgressObserver_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyProgressObserver_repr,                         /* tp_repr */
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
    PyProgressObserver_doc,              /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    PyProgressObserver_methods,          /* tp_methods */
    0,                         /* tp_members */
    PyProgressObserver_getseters,        /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyProgressObserver_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PyProgressObserver_new     /* tp_new */
};



} //end namespace ito
