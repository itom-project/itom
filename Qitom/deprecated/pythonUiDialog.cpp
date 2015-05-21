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
/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2012, Institut für Technische Optik (ITO), 
   Universität Stuttgart, Germany 
 
   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "pythonUiDialog.h"



#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "include/structmember.h"
#endif

#include "../global.h"

#include "../organizer/uiOrganizer.h"
#include "../organizer/addInManager.h"

#include "pythonQtConversion.h"

#include <qmap.h>
#include <qsharedpointer.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qcoreapplication.h>

#include <iostream>
#include <qvector.h>

#include "pythonEngineInc.h"

using namespace ito;

//----------------------------------------------------------------------------------------------------------------------------------
void timerCallback::timeout()
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
*
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

    if (m_callbackArgs)
        Py_DECREF(m_callbackArgs);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonUiDialog::PyUiTimer_dealloc(PyUiTimer* self)
{
    if (self->timer)
    {
        self->timer->stop();
        DELETE_AND_SET_NULL(self->timer);
    }
    if (self->callbackFunc)
    {
        DELETE_AND_SET_NULL(self->callbackFunc);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiTimer_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
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
int PythonUiDialog::PyUiTimer_init(PyUiTimer *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"timeOut", "callbackFunc", "parameters", NULL};

    if(args == NULL || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    PyObject *tempObj = NULL;
    self->callbackFunc = new timerCallback();
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
PyDoc_STRVAR(pyUiTimerName_doc,"name() -> returns name for class UiTimer");
PyObject* PythonUiDialog::PyUiTimer_name(PyUiTimer* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("uiTimer");
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiTimer_repr(PyUiTimer *self)
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
PyDoc_STRVAR(pyUiTimerStart_doc,"start() -> starts uiTimer");
PyObject* PythonUiDialog::PyUiTimer_start(PyUiTimer *self) 
{ 
    if (self->timer) 
        self->timer->start(); 
    Py_RETURN_NONE; 
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerStop_doc,"stop() -> stops uiTimer");
PyObject* PythonUiDialog::PyUiTimer_stop(PyUiTimer *self) 
{ 
    if (self->timer) 
        self->timer->stop(); 
    Py_RETURN_NONE; 
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerIsActive_doc,"isActive() -> returns timer status");
PyObject* PythonUiDialog::PyUiTimer_isActive(PyUiTimer *self)
{ 
    if (self->timer) 
        return PyLong_FromLong((long)self->timer->isActive());
    else
        return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiTimerSetInterval_doc,"setInterval(int interval) -> sets timer interval in [ms]");
PyObject* PythonUiDialog::PyUiTimer_setInterval(PyUiTimer *self, PyObject *args)
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
PyMethodDef PythonUiDialog::PyUiTimer_methods[] = {
        {"name", (PyCFunction)PyUiTimer_name, METH_NOARGS, pyUiTimerName_doc},
        {"start", (PyCFunction)PyUiTimer_start, METH_NOARGS, pyUiTimerStart_doc},
        {"stop", (PyCFunction)PyUiTimer_stop, METH_NOARGS, pyUiTimerStop_doc},
        {"isActive", (PyCFunction)PyUiTimer_isActive, METH_NOARGS, pyUiTimerIsActive_doc},
        {"setInterval", (PyCFunction)PyUiTimer_setInterval, METH_VARARGS, pyUiTimerSetInterval_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonUiDialog::PyUiTimer_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonUiDialog::PyUiTimerModule = {
        PyModuleDef_HEAD_INIT,
        "uiTimer",
        "timer for callback function",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonUiDialog::PyUiTimer_getseters[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonUiDialog::PyUiTimerType = {
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
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,            /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
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

//----------------------------------------------------------------------------------------------------------------------------------
void PythonUiDialog::PyUiDialog_dealloc(PyUiDialog* self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga && self->uiHandle >= 0)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        ito::RetVal retValue = retOk;

        QMetaObject::invokeMethod(uiOrga, "deleteDialog", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
        if(!locker.getSemaphore()->wait(5000))
        {
            PyErr_Format(PyExc_RuntimeError, "timeout while closing dialog");
        }
    }

    DELETE_AND_SET_NULL( self->signalMapper );
    DELETE_AND_SET_NULL_ARRAY( self->filename );

    Py_XDECREF(self->dialogButtons);

    //clear weak reference to this object
    if (self->weakreflist != NULL)
    {
        PyObject_ClearWeakRefs((PyObject *) self);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiDialog_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyUiDialog* self = (PyUiDialog *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->uiHandle = -1; //default: invalid
        self->dialogButtons = PyDict_New();
        PyDict_SetItemString(self->dialogButtons, "AcceptRole", PyUnicode_FromString("OK"));
        PyDict_SetItemString(self->dialogButtons, "RejectRole", PyUnicode_FromString("Cancel"));
        Py_INCREF(Py_None);
        self->winType = 0;
        self->buttonBarType = 0;
        self->childOfMainWindow = true;
        self->filename = NULL;
        self->signalMapper = NULL;
        self->weakreflist = NULL;
    }

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogInit_doc,"uiDialog(filename, [type(TYPEDIALOG|TYPEWINDOW|TYPEDOCKWIDGET), dialogButtonBar(BUTTONBAR_NO|BUTTONBAR_HORIZONTAL|BUTTONBAR_VERTICAL), dialogButtons, childOfMainWindow]) -> new instance of user defined dialog \n\
                                Parameters: \n\
                                filename [string]: path to user interface file (*.ui) \n\
                                type [int]: display type: \n\
                                    - 0 (TYPEDIALOG): ui-file is embedded in auto-created dialog (default), \n\
                                    - 1 (TYPEWINDOW): ui-file is handled as main window, \n\
                                    - 2 (TYPEDOCKWIDGET): ui-file is handled as dock-widget and appended to the main-window dock area \n\
                                dialogButtonBar [int]: Only for type TYPEDIALOG (0). Indicates whether buttons should automatically be added to the dialog: \n\
                                    - 0 (BUTTONBAR_NO): do not add any buttons \n\
                                    - 1 (BUTTONBAR_HORIZONTAL): add horizontal button bar \n\
                                    - 2 (BUTTONBAR_VERTICAL): add vertical button bar \n\
                                dialogButtons [dict]: every dictionary-entry is one button. key is the role, value is the button text \n\
                                childOfMainWindow [bool]: for type TYPEDIALOG and TYPEWINDOW only. Indicates whether window should be a child of itom main window (default: True)");
int PythonUiDialog::PyUiDialog_init(PyUiDialog *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"filename", "type", "dialogButtonBar", "dialogButtons", "childOfMainWindow", NULL};
    PyObject *dialogButtons = NULL;
    PyObject *tmp;
    PyBytesObject *bytesFilename = NULL;
    char *internalFilename;
    //PyUnicode_FSConverter

    if(args == NULL || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iiO!b", const_cast<char**>(kwlist), &PyUnicode_FSConverter, &bytesFilename, &self->winType, &self->buttonBarType, &PyDict_Type, &dialogButtons, &self->childOfMainWindow))
    {
        PyErr_Format(PyExc_TypeError,"Arguments does not fit to required list of arguments. See help(uiDialog).");
        Py_XDECREF(bytesFilename);
        return -1;
    }

    //check values:
    if(self->winType < 0 || self->winType > 2)
    {
        PyErr_Format(PyExc_ValueError,"Argument 'type' must have one of the values TYPEDIALOG (0), TYPEWINDOW (1) or TYPEDOCKWIDGET (2)");
        Py_XDECREF(bytesFilename);
        return -1;
    }

    if(self->buttonBarType < 0 || self->buttonBarType > 2)
     {
        PyErr_Format(PyExc_ValueError,"Argument 'dialogButtonBar' must have one of the values BUTTONBAR_NO (0), BUTTONBAR_HORIZONTAL (1) or BUTTONBAR_VERTICAL (2)");
        Py_XDECREF(bytesFilename);
        return -1;
    }


    DELETE_AND_SET_NULL_ARRAY(self->filename);
    internalFilename = PyBytes_AsString((PyObject*)bytesFilename);
    self->filename = new char[ strlen(internalFilename)+1];
    strcpy(self->filename, internalFilename);
    internalFilename = NULL;
    Py_XDECREF(bytesFilename);

    if(dialogButtons)
    {
        tmp = self->dialogButtons;
        Py_INCREF(dialogButtons);
        self->dialogButtons = dialogButtons;
        if(PyDict_Check(tmp)) PyDict_Clear(tmp);
        Py_XDECREF(tmp);
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    QSharedPointer<unsigned int> dialogHandle(new unsigned int);
    QSharedPointer<unsigned int> initSlotCount(new unsigned int);
    *dialogHandle = 0;
    *initSlotCount = 0;
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    StringMap dialogButtonMap;
    ito::RetVal retValue;

    if(self->dialogButtons)
    {
        //transfer dialogButtons dict to dialogButtonMap
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QString keyString, valueString;
        bool ok=false;

        while (PyDict_Next(self->dialogButtons, &pos, &key, &value)) 
        {
            keyString = PythonQtConversion::PyObjGetString(key,true,ok);
            valueString = PythonQtConversion::PyObjGetString(value,true,ok);
            if(keyString.isNull() || valueString.isNull())
            {
                std::cout << "Warning while parsing dialogButtons-dictionary. At least one element does not contain a string as key and value" << std::endl;
            }
            else
            {
                dialogButtonMap[keyString] = valueString;
            }
        }
    }

    int uiDescription = UiOrganizer::createUiDescription(self->winType,self->buttonBarType,self->childOfMainWindow);
    QSharedPointer<QByteArray> className(new QByteArray());
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QMetaObject::invokeMethod(uiOrga, "createNewDialog",Q_ARG(QString,QString(self->filename)), Q_ARG(int, uiDescription), Q_ARG(StringMap, dialogButtonMap), Q_ARG(QSharedPointer<unsigned int>, dialogHandle),Q_ARG(QSharedPointer<unsigned int>, initSlotCount), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(QSharedPointer<QByteArray>, className), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(60000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while opening dialog");
        return -1;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return -1;

    self->uiHandle = static_cast<int>(*dialogHandle);
    DELETE_AND_SET_NULL( self->signalMapper );
    self->signalMapper = new PythonQtSignalMapper(*initSlotCount);

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogName_doc,"name() -> returns name for class UiDialog");
PyObject* PythonUiDialog::PyUiDialog_name(PyUiDialog* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("UiDialog");
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiDialog_repr(PyUiDialog *self)
{
    PyObject *result;
    if(self->uiHandle < 0)
    {
        result = PyUnicode_FromFormat("UiDialog(empty)");
    }
    else
    {
        if(self->filename == NULL)
        {
            result = PyUnicode_FromFormat("UiDialog(handle: %i)", self->uiHandle);
        }
        else
        {
            result = PyUnicode_FromFormat("UiDialog(filename: '%s', handle: %i)", self->filename, self->uiHandle);
        }
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogShow_doc,"show(modal) -> shows initialized UI-Dialog \n\
                                Parameters: \n\
                                modal: 1 modal, 0: unmodal (default)");
PyObject* PythonUiDialog::PyUiDialog_show(PyUiDialog *self, PyObject *args)
{
    bool modal = 0;

    if(!PyArg_ParseTuple(args, "|i", &modal))
    {
        PyErr_SetString(PyExc_RuntimeError, "Parameter modal must be a boolean value");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<int> retCodeIfModal(new int);
    *retCodeIfModal = -1;
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "showDialog", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)) , Q_ARG(bool,modal), Q_ARG(QSharedPointer<int>, retCodeIfModal), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(modal == true)
    {
        /*while(!locker.getSemaphore()->wait(10))
        {
            QCoreApplication::processEvents();
            QCoreApplication::sendPostedEvents();
        }
*/
        locker.getSemaphore()->waitAndProcessEvents(-1);
    }
    else
    {
        if(!locker.getSemaphore()->wait(30000))
        {
            PyErr_Format(PyExc_RuntimeError, "timeout while showing dialog");
            return NULL;
        }
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(*retCodeIfModal >= 0)
    {
        return Py_BuildValue("i",*retCodeIfModal);
    }
    else
    {
        Py_RETURN_NONE;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogHide_doc,"hide() -> hides initialized UI-Dialog");
PyObject* PythonUiDialog::PyUiDialog_hide(PyUiDialog *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "hideDialog", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while hiding dialog");
        return NULL;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogIsVisible_doc,"isVisible() -> returns true if dialog is still visible");
PyObject* PythonUiDialog::PyUiDialog_isVisible(PyUiDialog *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<bool> visible(new bool);
    *visible = false;
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "isVisible", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)), Q_ARG(QSharedPointer<bool>, visible), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while getting visible status");
        return NULL;
    }
    
    if(*visible)
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetPropertyInfo_doc,"getPropertyInfo(widgetName [,propertyName]) ->   ");
PyObject* PythonUiDialog::PyUiDialog_getPropertyInfo(PyUiDialog *self, PyObject *args)
{
    const char *widgetName = 0;
    const char *propertyName = 0;
    if(!PyArg_ParseTuple(args, "s|s", &widgetName, &propertyName))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument must contain widgetName (string) and propertyName (string, optional)");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());
    QMetaObject::invokeMethod(uiOrga, "getPropertyInfos", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)), Q_ARG(QString, QString(widgetName)), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while getting property information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if(retValue.containsError())
    {
        if(retValue.errorMessage())
        {
            PyErr_Format(PyExc_RuntimeError, "Error while getting property infos with error message: \n%s", retValue.errorMessage());
        }
        else
        {
            PyErr_Format(PyExc_RuntimeError, "Error while getting property infos.");
        }
        return NULL;
    }
    else if(retValue.containsWarning())
    {
        std::cout << "Warning while getting property infos with message: " << QObject::tr(retValue.errorMessage()).toAscii().data() << std::endl;
    }
    
    QStringList stringList = retPropMap->keys();
    QString propNameString = QString(propertyName);

    if(propertyName == NULL)
    {
        PyObject *retObj = PythonQtConversion::QStringListToPyList(stringList);
        return retObj;
    }
    else if(retPropMap->contains(propNameString))
    {
        int flags = retPropMap->value(propNameString).toInt();
        PyObject *retObj = PyDict_New();
        PyDict_SetItemString(retObj, "name", PythonQtConversion::QVariantToPyObject( propNameString ));
        PyDict_SetItemString(retObj, "valid", PythonQtConversion::GetPyBool( flags & UiOrganizer::propValid ));
        PyDict_SetItemString(retObj, "readable", PythonQtConversion::GetPyBool( flags & UiOrganizer::propReadable ));
        PyDict_SetItemString(retObj, "writable", PythonQtConversion::GetPyBool( flags & UiOrganizer::propWritable ));
        PyDict_SetItemString(retObj, "resettable", PythonQtConversion::GetPyBool( flags & UiOrganizer::propResettable ));
        PyDict_SetItemString(retObj, "final", PythonQtConversion::GetPyBool( flags & UiOrganizer::propFinal ));
        PyDict_SetItemString(retObj, "constant", PythonQtConversion::GetPyBool( flags & UiOrganizer::propConstant ));
        PyObject *proxyDict = PyDictProxy_New(retObj);
        Py_DECREF(retObj);
        return proxyDict;
    }
    else
    {
        PyErr_Format(PyExc_RuntimeError, QString("the property '%1' does not exist.").arg(propNameString).toAscii());
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetProperties_doc,"getProperty(widgetName, propertyName or list/tuple of propertyNames) ->   ");
PyObject* PythonUiDialog::PyUiDialog_getProperties(PyUiDialog *self, PyObject *args)
{
    const char *widgetName = NULL;
    PyObject *propertyNames = NULL;
    QStringList propNames;
    bool ok = false;

    if(!PyArg_ParseTuple(args, "sO", &widgetName, &propertyNames))
    {
        PyErr_Format(PyExc_RuntimeError, "argument list must be widgetName (string), propertyNames (string or tuple/list of strings)");
        return NULL;
    }

    Py_INCREF(propertyNames);

    if(PyBytes_Check(propertyNames) || PyUnicode_Check(propertyNames))
    {
        QString temp = PythonQtConversion::PyObjGetString(propertyNames, true, ok);
        if(ok)
        {
            propNames << temp;
        }
        else
        {
            Py_XDECREF(propertyNames);
            PyErr_Format(PyExc_RuntimeError, "property name string could not be parsed.");
            return NULL;
        }
    }
    else if(PySequence_Check(propertyNames))
    {
        propNames = PythonQtConversion::PyObjToStringList(propertyNames, true, ok);
        if(!ok)
        {
            Py_XDECREF(propertyNames);
            PyErr_Format(PyExc_RuntimeError, "property names list or tuple could not be converted to a list of strings");
            return NULL;
        }
    }
    else
    {
        Py_XDECREF(propertyNames);
        PyErr_Format(PyExc_RuntimeError, "property name must be a string or tuple/list of strings"); 
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        Py_XDECREF(propertyNames);
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        Py_XDECREF(propertyNames);
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());
    for(int i = 0 ; i < propNames.count() ; i++)
    {
        (*retPropMap)[propNames.at(i)] = QVariant();
    }

    QMetaObject::invokeMethod(uiOrga, "readProperties", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)), Q_ARG(QString, QString(widgetName)), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        Py_XDECREF(propertyNames);
        PyErr_Format(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    PyObject *retObj = PyList_New(propNames.count());
    for(int i = 0 ; i < propNames.count() ; i++)
    {
        PyList_SetItem(retObj,i, PythonQtConversion::QVariantToPyObject(retPropMap->value(propNames.at(i))));
    }

    Py_XDECREF(propertyNames);

    return retObj;

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogSetProperties_doc,"setProperty(widgetName [,propertyName]) ->   ");
PyObject* PythonUiDialog::PyUiDialog_setProperties(PyUiDialog *self, PyObject *args)
{
    const char *widgetName = NULL;
    const char *propertyName = NULL;
    PyObject *prop = NULL;
    PyObject *propDict = NULL;
    bool ok = false;
    QVariantMap propMap;

    if(PyArg_ParseTuple(args, "ssO", &widgetName, &propertyName, &prop))
    {
        Py_INCREF(prop);
        propDict = PyDict_New();
        PyDict_SetItemString(propDict, propertyName, prop);
        Py_DECREF(prop);
    }
    else
    {
        PyErr_Clear();
        if(!PyArg_ParseTuple(args, "sO!", &widgetName, &PyDict_Type, &propDict))
        {
            PyErr_Format(PyExc_RuntimeError, "argument list must be widgetName (string), property name (string), property or dict of name-property pairs");
            return NULL;
        }
        Py_INCREF(propDict);
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    QVariant valueV;
    QString keyS;

    while (PyDict_Next(propDict, &pos, &key, &value)) 
    {
        keyS = PythonQtConversion::PyObjGetString(key,true,ok);
        valueV = PythonQtConversion::PyObjToQVariant(value);
        if(valueV.isValid())
        {
            propMap[keyS] = valueV;
        }
        else
        {
            PyErr_Format(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
            Py_DECREF(propDict);
            return NULL;
        }
    }

    Py_DECREF(propDict); 

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "writeProperties", Q_ARG(unsigned int, static_cast<unsigned int>(self->uiHandle)), Q_ARG(QString, widgetName), Q_ARG(QVariantMap, propMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while writing property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiDialog_getattro(PyUiDialog *self, PyObject *args)
{
    PyObject *ret = PyObject_GenericGetAttr((PyObject*)self,args);
    if(ret != NULL)
    {
        return ret;
    }
    PyErr_Clear(); //genericgetattr throws an error, if attribute is not available, which it isn't for attributes pointing to widgetNames

    //return new instance of PyMetaObject
    PyUiDialogMetaObject::PyMetaObject *pyMetaObject;
    PyObject *arg2 = Py_BuildValue("OO", self, args);

    pyMetaObject = (PyUiDialogMetaObject::PyMetaObject *)PyObject_CallObject((PyObject *)&PyUiDialogMetaObject::PyMetaObjectType, arg2);
    Py_DECREF(arg2);

    if(pyMetaObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Could not create uiDialogMetaObject of requested widget");
        return NULL;
    }

    if(PyErr_Occurred())
    {
        Py_XDECREF(pyMetaObject);
        pyMetaObject = NULL;
    }

    return (PyObject *)pyMetaObject;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiDialog_setattro(PyUiDialog * /*self*/, PyObject * /*args*/)
{
    PyErr_SetString(PyExc_TypeError, "It is not possible to assign another widget to the given widget in the user interface.");
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetUiMetaObject_doc,"win() -> returns uiDialogMetaObject of entire dialog or main-window.");
PyObject* PythonUiDialog::PyUiDialog_getuimetaobject(PyUiDialog* self)
{
    
    //return new instance of PyMetaObject
    PyUiDialogMetaObject::PyMetaObject *pyMetaObject;
    PyObject *arg2 = Py_BuildValue("Os", self, "");

    pyMetaObject = (PyUiDialogMetaObject::PyMetaObject *)PyObject_CallObject((PyObject *)&PyUiDialogMetaObject::PyMetaObjectType, arg2);
    Py_DECREF(arg2);

    if(pyMetaObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Could not create uiDialogMetaObject of requested widget");
        return NULL;
    }

    if(PyErr_Occurred())
    {
        Py_XDECREF(pyMetaObject);
        pyMetaObject = NULL;
    }

    return (PyObject *)pyMetaObject;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetDouble_doc,"getDouble(title, label, defaultValue [, min, max, decimals]) -> function to get a floating point number from the user \n\
                                     Parameters: \n\
                                     title (string) is the dialog title \n\
                                     label (string) is the label above the text box \n\
                                     defaultValue (double) is the default value in the text box \n\
                                     min (double) is the allowed minimal value (default: -2147483647.0) \n\
                                     max (double) is the allowed maximal value (default: 2147483647.0) \n\
                                     decimals (int) are the number of shown decimals (default: 1)");
PyObject* PythonUiDialog::PyUiDialog_getDouble(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultValue", "min", "max", "decimals", NULL};
    const char *title = 0;
    const char *label = 0;
    double defaultValue = 0;
    double minValue = -2147483647;
    double maxValue = 2147483647;
    int decimals = 1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ssd|ddi", const_cast<char**>(kwlist), &title, &label, &defaultValue, &minValue, &maxValue, &decimals))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), default value (double), min (double, optional), max (double, optional), decimals (int, optional)");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<double> retDblValue(new double);
    *retDblValue = defaultValue;

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetDouble", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(double, defaultValue), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<double>, retDblValue), Q_ARG(double,minValue), Q_ARG(double,maxValue), Q_ARG(int,decimals), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }
    
    if(*retOk == true)
    {
        Py_INCREF(Py_True);
        return Py_BuildValue("dO", *retDblValue, Py_True );
    }
    else
    {
        Py_INCREF(Py_False);
        return Py_BuildValue("dO", defaultValue, Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetInt_doc,"getInt(title, label, defaultValue [, min, max, step]) -> function to get an integer number from the user \n\
                                     Parameters: \n\
                                     title (string) is the dialog title \n\
                                     label (string) is the label above the text box \n\
                                     defaultValue (int) is the default value in the text box \n\
                                     min (int) is the allowed minimal value (default: -2147483647) \n\
                                     max (int) is the allowed maximal value (default: 2147483647) \n\
                                     step (int) is the change step if user presses the up/down arrow (default: 1)");
PyObject* PythonUiDialog::PyUiDialog_getInt(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultValue", "min", "max", "step", NULL};
    const char *title = 0;
    const char *label = 0;
    int defaultValue = 0;
    int minValue = -2147483647;
    int maxValue = 2147483647;
    int step = 1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ssi|iii", const_cast<char**>(kwlist), &title, &label, &defaultValue, &minValue, &maxValue, &step))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), default value (int), min (int, optional), max (int, optional), step (int, optional)");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<int> retIntValue(new int);
    *retIntValue = defaultValue;

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetInt", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(int, defaultValue), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<int>, retIntValue), Q_ARG(int,minValue), Q_ARG(int,maxValue), Q_ARG(int,step), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }
    
    if(*retOk == true)
    {
        Py_INCREF(Py_True);
        return Py_BuildValue("iO", *retIntValue, Py_True );
    }
    else
    {
        Py_INCREF(Py_False);
        return Py_BuildValue("iO", defaultValue, Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetItem_doc,"getItem(title, label, stringList [, currentIndex, editable]) -> function to let the user select an item from a string list \n\
                                     Parameters: \n\
                                     title (string) is the dialog title \n\
                                     label (string) is the label above the text box \n\
                                     stringList (tuple or list) is a list or tuple of possible string values \n\
                                     currentIndex (int) defines the preselected value index (default: 0) \n\
                                     editable (bool) defines wether new entries can be added (True) or not (False, default)");
PyObject* PythonUiDialog::PyUiDialog_getItem(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "stringList", "currentIndex", "editable", NULL};
    const char *title = 0;
    const char *label = 0;
    PyObject *stringList = NULL;
    int currentIndex = 1;
    bool editable = true;
    QStringList stringListQt;
    QString temp;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|ib", const_cast<char**>(kwlist), &title, &label, &stringList, &currentIndex, &editable))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), string list (list, tuple), currentIndex (int), editable (bool)");
        return NULL;
    }

    if(!PySequence_Check(stringList))
    {
        PyErr_SetString(PyExc_TypeError, "string list must be a sequence of elements (tuple or list)");
        return NULL;
    }
    else
    {
        Py_ssize_t length = PySequence_Size(stringList);
        bool ok = false;
        for(Py_ssize_t i = 0 ; i < length ; i++)
        {
            temp = PythonQtConversion::PyObjGetString(PySequence_GetItem(stringList,i),true,ok);
            if(!temp.isNull()) 
            {
                stringListQt << temp;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "string list must only contain string values!");
                return NULL;
            }
        }
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    
    QSharedPointer<QString> retString(new QString());
    

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetItem", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(QStringList, stringListQt), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<QString>, retString), Q_ARG(int, currentIndex), Q_ARG(bool, editable), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }
    
    if(*retOk == true)
    {
        Py_INCREF(Py_True);
        QByteArray ba = retString->toAscii();
        return Py_BuildValue("sO", ba.data(), Py_True );
    }
    else
    {
        Py_INCREF(Py_False);
        return Py_BuildValue("sO", "", Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetText_doc,"getText(title, label, defaultString) -> function to get a string from the user \n\
                                     Parameters: \n\
                                     title (string) is the dialog title \n\
                                     label (string) is the label above the text box \n\
                                     defaultString (string) is the default string in the text box");
PyObject* PythonUiDialog::PyUiDialog_getText(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultString", NULL};
    const char *title = 0;
    const char *label = 0;
    const char *defaultString = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "sss", const_cast<char**>(kwlist), &title, &label, &defaultString))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), default string (string)");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<QString> retStringValue(new QString(defaultString));

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetText", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(QString, QString(defaultString)), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<QString>, retStringValue), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }
    
    if(*retOk == true)
    {
        Py_INCREF(Py_True);
        return Py_BuildValue("sO", retStringValue->toAscii().data(), Py_True );
    }
    else
    {
        Py_INCREF(Py_False);
        return Py_BuildValue("sO", defaultString, Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogMsgInformation_doc,"msgInformation(title, text [, buttons, defaultButton]) -> opens an information message box \n\
                                    Parameters: \n\
                                    title (string) is the message box title \n\
                                    text (string) is the message text \n\
                                    buttons is an or-combination of uiDialog.MsgBox[...]-constants indicating the buttons to display \n\
                                    defaultbutton is a value of uiDialg.MsgBox[...] which indicates the default button");
PyObject* PythonUiDialog::PyUiDialog_msgInformation(PyUiDialog *self, PyObject *args, PyObject *kwds)
{
    return PyUiDialog_msgGeneral(self,args,kwds,1);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogMsgQuestion_doc,"msgQuestion(title, text [, buttons, defaultButton]) -> opens a question message box \n\
                                    Parameters: \n\
                                    title (string) is the message box title \n\
                                    text (string) is the message text \n\
                                    buttons is an or-combination of uiDialog.MsgBox[...]-constants indicating the buttons to display \n\
                                    defaultbutton is a value of uiDialg.MsgBox[...] which indicates the default button");
PyObject* PythonUiDialog::PyUiDialog_msgQuestion(PyUiDialog *self, PyObject *args, PyObject *kwds)
{
    return PyUiDialog_msgGeneral(self,args,kwds,2);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogMsgWarning_doc,"msgWarning(title, text [, buttons, defaultButton]) -> opens a warning message box \n\
                                    Parameters: \n\
                                    title (string) is the message box title \n\
                                    text (string) is the message text \n\
                                    buttons is an or-combination of uiDialog.MsgBox[...]-constants indicating the buttons to display \n\
                                    defaultbutton is a value of uiDialg.MsgBox[...] which indicates the default button");
PyObject* PythonUiDialog::PyUiDialog_msgWarning(PyUiDialog *self, PyObject *args, PyObject *kwds)
{
    return PyUiDialog_msgGeneral(self,args,kwds,3);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogMsgCritical_doc,"msgCritical(title, text [, buttons, defaultButton]) -> opens a critical message box \n\
                                    Parameters: \n\
                                    title (string) is the message box title \n\
                                    text (string) is the message text \n\
                                    buttons is an or-combination of uiDialog.MsgBox[...]-constants indicating the buttons to display \n\
                                    defaultbutton is a value of uiDialg.MsgBox[...] which indicates the default button");
PyObject* PythonUiDialog::PyUiDialog_msgCritical(PyUiDialog *self, PyObject *args, PyObject *kwds)
{
    return PyUiDialog_msgGeneral(self,args,kwds,4);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUiDialog::PyUiDialog_msgGeneral(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds, int type)
{
    const char *kwlist[] = {"title", "text", "buttons", "defaultButton", NULL};
    const char *title = 0;
    const char *text = 0;
    int buttons = QMessageBox::Ok;
    int defaultButton = QMessageBox::NoButton;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ss|ii", const_cast<char**>(kwlist), &title, &text, &buttons, &defaultButton))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), buttons (combination of uiDialog.MsgBox[...]), defaultButton (uiDialog.MsgBox[...])");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<int> retButton(new int);
    *retButton = QMessageBox::Escape;
    QSharedPointer<QString> retButtonText(new QString());

    QMetaObject::invokeMethod(uiOrga, "showMessageBox", Q_ARG(int, type), Q_ARG(QString, QString(title)), Q_ARG(QString, QString(text)), Q_ARG(int, buttons), Q_ARG(int, defaultButton), Q_ARG(QSharedPointer<int>, retButton), Q_ARG(QSharedPointer<QString>, retButtonText), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing message box");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;
    
    return Py_BuildValue("is", *retButton, retButtonText->toAscii().data());
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetExistingDirectory_doc,"getExistingDirectory(caption, startDirectory [, options]) -> opens a dialog to choose an existing directory \n\
                                    Parameters: \n\
                                    caption (string) is the caption of this dialog \n\
                                    startDirectory (string) is the start directory \n\
                                    options is an or-combination of the following options (see QFileDialog::Option: \n\
                                    - ShowDirsOnly (1) [default] \n\
                                    - DontResolveSymlinks (2) \n\
                                    - DontConfirmOverwrite (4) \n\
                                    - DontUseNativeDialog (16) \n\
                                    ... (for others see Qt-Help)");
PyObject* PythonUiDialog::PyUiDialog_getExistingDirectory(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"caption", "directory", "options", NULL};
    const char *caption = 0;
    const char *directory = 0;
    int options = 1; //QFileDialog::ShowDirsOnly


    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|ssi", const_cast<char**>(kwlist), &caption, &directory, &options))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QString> sharedDir(new QString(directory));

    QMetaObject::invokeMethod(uiOrga, "showFileDialogExistingDir", Q_ARG(unsigned int, 0), Q_ARG(QString, QString(caption)), Q_ARG(QSharedPointer<QString>, sharedDir), Q_ARG(int, options), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(sharedDir->isEmpty() || sharedDir->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", sharedDir->toAscii().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetOpenFileName_doc,"getOpenFileName([caption, startDirectory, filters, selectedFilterIndex, options]) -> opens dialog for chosing an existing file. \n\
                                           Parameters: \n\
                                           - caption (String) [optional] \n\
                                           - startDirectory (String) [optional, if not indicated currentDirectory will be taken] \n\
                                           - filters (String) [optional], possible filter list, entries should be separated by ;; , e.g. 'Images (*.png *.jpg);;Text files (*.txt)' \n\
                                           - selectedFilterIndex [optional, default: 0] is the index of filters which is set by default (0 is first entry) \n\
                                           - options [optional, default: 0], or-combination of enum values QFileDialog::Options \n\
                                           \n\
                                           returns filename as string or empty string if dialog has been aborted.");
PyObject* PythonUiDialog::PyUiDialog_getOpenFileName(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    
    const char *kwlist[] = {"caption", "startDirectory", "filters", "selectedFilterIndex", "options", NULL};
    const char *caption = 0;
    const char *directory = 0;
    const char *filters = 0;
    int selectedFilterIndex = 0;
    int options = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sssii", const_cast<char**>(kwlist), &caption, &directory, &filters, &selectedFilterIndex, &options))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QString> file(new QString());
    //QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore
    QMetaObject::invokeMethod(uiOrga, "showFileOpenDialog", Q_ARG(unsigned int, 0), Q_ARG(QString, QString(caption)), Q_ARG(QString, QString(directory)), Q_ARG(QString, QString(filters)), Q_ARG(QSharedPointer<QString>, file), Q_ARG(int, selectedFilterIndex), Q_ARG(int, options), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(file->isEmpty() || file->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", file->toAscii().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogGetSaveFileName_doc,"getSaveFileName([caption, startDirectory, filters, selectedFilterIndex, options]) -> opens dialog for chosing a file to save. \n\
                                           Parameters: \n\
                                           - caption (String) [optional] \n\
                                           - startDirectory (String) [optional, if not indicated currentDirectory will be taken] \n\
                                           - filters (String) [optional], possible filter list, entries should be separated by ;; , e.g. 'Images (*.png *.jpg);;Text files (*.txt)' \n\
                                           - selectedFilterIndex [optional, default: 0] is the index of filters which is set by default (0 is first entry) \n\
                                           - options [optional, default: 0], or-combination of enum values QFileDialog::Options");
PyObject* PythonUiDialog::PyUiDialog_getSaveFileName(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    
    const char *kwlist[] = {"caption", "startDirectory", "filters", "selectedFilterIndex", "options", NULL};
    const char *caption = 0;
    const char *directory = 0;
    const char *filters = 0;
    int selectedFilterIndex = 0;
    int options = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sssii", const_cast<char**>(kwlist), &caption, &directory, &filters, &selectedFilterIndex, &options))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QString> file(new QString());
    //QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore
    QMetaObject::invokeMethod(uiOrga, "showFileSaveDialog", Q_ARG(unsigned int, 0), Q_ARG(QString, QString(caption)), Q_ARG(QString, QString(directory)), Q_ARG(QString, QString(filters)), Q_ARG(QSharedPointer<QString>, file), Q_ARG(int, selectedFilterIndex), Q_ARG(int, options), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(file->isEmpty() || file->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", file->toAscii().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiDialogCreateNewPluginWidget_doc, "createNewPluginWidget(widgetName[, mandparams, optparams]) -> creates widget defined by any algorithm plugin and returns the instance of type uiDialog \n\
                                Parameters: \n\
                                - 'widgetName' name algorithm widget \n\
                                - parameters to pass to the plugin. The parameters are parsed and unnamed parameters are used in their \
                                incoming order to fill first mandatory parameters and afterwards optional parameters. Parameters may be passed \
                                with name as well but after the first named parameter no more unnamed parameters are allowed.");
PyObject* PythonUiDialog::PyUiDialog_createNewAlgoWidget(PyUiDialog * /*self*/, PyObject *args, PyObject *kwds)
{
    int length = PyTuple_Size(args);

    if (length == 0)
    {
        PyErr_Format(PyExc_ValueError, QObject::tr("no widget name specified").toAscii());
        return NULL;
    }
    
    PyErr_Clear();
    QVector<ito::tParam> paramsMand;
    QVector<ito::tParam> paramsOpt;
    ito::RetVal retVal = 0;
    PyObject *pnameObj = NULL;
    PyObject *params = NULL;
    QString algoWidgetName;
    bool ok;

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        PyErr_Format(PyExc_RuntimeError, QObject::tr("no addin-manager found").toAscii());
        return NULL;
    }

    pnameObj = PyTuple_GetItem(args, 0);
    algoWidgetName = PythonQtConversion::PyObjGetString(pnameObj, true, ok);
    if(!ok)
    {
        PyErr_Format(PyExc_TypeError, QObject::tr("the first parameter must contain the widget name as string").toAscii());
        return NULL;
    }

    const ito::AddInAlgo::algoWidgetDef *def = AIM->getAlgoWidgetDef( algoWidgetName );
    if(def == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, QObject::tr("Could not find plugin widget with name '%1'").arg(algoWidgetName).toAscii().data());
        return NULL;
    }
    
    retVal = def->m_paramFunc(&paramsMand, &paramsOpt);
    if (retVal.containsWarningOrError())
    {
        PyErr_Format(PyExc_RuntimeError, QObject::tr("Could not load default parameter set for loading plugin widget. Error-Message: \n%s\n").toAscii(), QObject::tr(retVal.errorMessage()).toAscii().data());
        return NULL;
    }

    params = PyTuple_GetSlice(args, 1, PyTuple_Size(args));
    if (parseInitParams(&paramsMand, &paramsOpt, params, kwds) != ito::retOk)
    {
        PyErr_Format(PyExc_RuntimeError, "error while parsing parameters.");
        return NULL;
    }
    Py_DECREF(params);

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<unsigned int> dialogHandle(new unsigned int);
    QSharedPointer<unsigned int> initSlotCount(new unsigned int);
    *dialogHandle = 0;
    *initSlotCount = 0;
    QMetaObject::invokeMethod(uiOrga, "loadPluginWidget", Q_ARG(void*, reinterpret_cast<void*>(def->m_widgetFunc)), Q_ARG(QVector<ito::tParam> *, &paramsMand), Q_ARG(QVector<ito::tParam> *, &paramsOpt), Q_ARG(QSharedPointer<unsigned int>, dialogHandle), Q_ARG(QSharedPointer<unsigned int>, initSlotCount), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while loading plugin widget");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    PythonUiDialog::PyUiDialog *dialog;

    PyObject *emptyTuple = PyTuple_New(0);
    dialog = (PyUiDialog*)PyObject_Call((PyObject*)&PyUiDialogType, NULL, NULL); //new ref
    Py_XDECREF(emptyTuple);

    if(dialog == NULL)
    {
        if(*dialogHandle)
        {
            ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(uiOrga, "deleteDialog", Q_ARG(unsigned int, static_cast<unsigned int>(*dialogHandle)), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore()));
    
            if(!locker2.getSemaphore()->wait(5000))
            {
                PyErr_Format(PyExc_RuntimeError, "timeout while closing dialog");
            }
        }

        PyErr_Format(PyExc_RuntimeError, "could not create a new instance of uiDialog.");
        return NULL;
    }

    dialog->uiHandle = static_cast<int>(*dialogHandle);
    DELETE_AND_SET_NULL( dialog->signalMapper );
    dialog->signalMapper = new PythonQtSignalMapper(*initSlotCount);

    return (PyObject*)dialog;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonUiDialog::PyUiDialog_methods[] = {
        {"name", (PyCFunction)PyUiDialog_name, METH_NOARGS, pyUiDialogName_doc},
        {"show", (PyCFunction)PyUiDialog_show,     METH_VARARGS, pyUiDialogShow_doc},
        {"hide", (PyCFunction)PyUiDialog_hide, METH_NOARGS, pyUiDialogHide_doc},
        {"isVisible", (PyCFunction)PyUiDialog_isVisible, METH_NOARGS, pyUiDialogIsVisible_doc},
        {"getPropertyInfo", (PyCFunction)PyUiDialog_getPropertyInfo, METH_VARARGS, pyUiDialogGetPropertyInfo_doc},
        {"getProperty",(PyCFunction)PyUiDialog_getProperties, METH_VARARGS, pyUiDialogGetProperties_doc},
        {"setProperty",(PyCFunction)PyUiDialog_setProperties, METH_VARARGS, pyUiDialogSetProperties_doc},
        {"win",(PyCFunction)PyUiDialog_getuimetaobject, METH_NOARGS, pyUiDialogGetUiMetaObject_doc},
        {"getDouble",(PyCFunction)PyUiDialog_getDouble, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogGetDouble_doc},
        {"getInt",(PyCFunction)PyUiDialog_getInt, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogGetInt_doc},
        {"getItem",(PyCFunction)PyUiDialog_getItem, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogGetItem_doc},
        {"getText",(PyCFunction)PyUiDialog_getText, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogGetText_doc},
        {"msgInformation", (PyCFunction)PyUiDialog_msgInformation, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogMsgInformation_doc},
        {"msgQuestion", (PyCFunction)PyUiDialog_msgQuestion, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogMsgQuestion_doc},
        {"msgWarning", (PyCFunction)PyUiDialog_msgWarning, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogMsgWarning_doc},
        {"msgCritical", (PyCFunction)PyUiDialog_msgCritical, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogMsgCritical_doc},
        {"getExistingDirectory", (PyCFunction)PyUiDialog_getExistingDirectory, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiDialogGetExistingDirectory_doc},
        {"getOpenFileName", (PyCFunction)PyUiDialog_getOpenFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiDialogGetOpenFileName_doc},
        {"getSaveFileName", (PyCFunction)PyUiDialog_getSaveFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiDialogGetSaveFileName_doc},
        {"createNewPluginWidget", (PyCFunction)PyUiDialog_createNewAlgoWidget, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiDialogCreateNewPluginWidget_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonUiDialog::PyUiDialog_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonUiDialog::PyUiDialogModule = {
        PyModuleDef_HEAD_INIT,
        "uiDialog",
        "Itom userInterfaceDialog type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonUiDialog::PyUiDialog_getseters[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject PythonUiDialog::PyUiDialogType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.uiDialog",             /* tp_name */
        sizeof(PyUiDialog),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyUiDialog_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyUiDialog_repr,         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        (getattrofunc)PyUiDialog_getattro, /* tp_getattro */
        (setattrofunc)PyUiDialog_setattro,  /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        pyUiDialogInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,            /* tp_richcompare */
        offsetof(PyUiDialog, weakreflist),                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyUiDialog_methods,             /* tp_methods */
        PyUiDialog_members,             /* tp_members */
        PyUiDialog_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyUiDialog_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyUiDialog_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//----------------------------------------------------------------------------------------------------------------------------------
void PythonUiDialog::PyUiDialog_addTpDict(PyObject *tp_dict)
{
    PyObject *value;
    QMetaObject metaObject = QMessageBox::staticMetaObject;
    QMetaEnum metaEnum = metaObject.enumerator( metaObject.indexOfEnumerator( "StandardButtons" ));
    QString key;
    //auto-parsing of StandardButtons-enumeration for key-value-pairs
    for(int i = 0 ; i < metaEnum.keyCount() ; i++)
    {
        value = Py_BuildValue("i", metaEnum.value(i));
        key = metaEnum.key(i);
        key.prepend("MsgBox"); //Button-Constants will be accessed by uiDialog.MsgBoxOk, uiDialog.MsgBoxError...
        PyDict_SetItemString(tp_dict, key.toAscii().data(), value);
        Py_DECREF(value);
    }

    //add dialog types
    value = Py_BuildValue("i", 0);
    PyDict_SetItemString(tp_dict, "TYPEDIALOG", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 1);
    PyDict_SetItemString(tp_dict, "TYPEWINDOW", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 2);
    PyDict_SetItemString(tp_dict, "TYPEDOCKWIDGET", value);
    Py_DECREF(value);

    //add button orientation
    value = Py_BuildValue("i", 0);
    PyDict_SetItemString(tp_dict, "BUTTONBAR_NO", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 1);
    PyDict_SetItemString(tp_dict, "BUTTONBAR_HORIZONTAL", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 2);
    PyDict_SetItemString(tp_dict, "BUTTONBAR_VERTICAL", value);
    Py_DECREF(value);
}

// -------------------------------------------------------------------------------------------------------------------------
//
//  PyUiDialogMetaObject
//
// -------------------------------------------------------------------------------------------------------------------------
void PyUiDialogMetaObject::PyMetaObject_dealloc(PyMetaObject* self)
{
    if (self->dialog)
        Py_XDECREF(self->dialog);
    if (self->objName)
        DELETE_AND_SET_NULL_ARRAY(self->objName);
    if (self->methodList)
        DELETE_AND_SET_NULL(self->methodList);
    if (self->methodListHash)
        DELETE_AND_SET_NULL(self->methodListHash);

    //clear weak reference to this object
    if (self->weakreflist != NULL)
    {
        PyObject_ClearWeakRefs((PyObject *) self);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PyUiDialogMetaObject::PyMetaObject_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyMetaObject* self = (PyMetaObject *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->dialog = NULL;
        self->objName = NULL;
        self->objectID = 0; //invalid
        self->methodDescriptionListLoaded = false;
        self->methodList = NULL;
        self->methodListHash = NULL;
        self->weakreflist = NULL;
    }

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectInit_doc,"");
int PyUiDialogMetaObject::PyMetaObject_init(PyMetaObject *self, PyObject *args, PyObject * /*kwds*/)
{
    PyObject *baseObject = NULL;
    const char *objName = NULL;
    PythonUiDialog::PyUiDialog *uiDialog = NULL;

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ito::RetVal retValue = retOk;
    QSharedPointer<unsigned int> objectID(new unsigned int());
    *objectID = 0;
    DELETE_AND_SET_NULL_ARRAY(self->objName);

    if(PyArg_ParseTuple(args,"O!s", &PythonUiDialog::PyUiDialogType, &baseObject, &objName))
    {
        uiDialog = (PythonUiDialog::PyUiDialog*)baseObject;
        if(uiDialog->uiHandle < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of UiDialog");
            return NULL;
        }

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "getChildObject", Q_ARG(unsigned int, static_cast<unsigned int>(uiDialog->uiHandle)), Q_ARG(QString, QString(objName)), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if(!locker.getSemaphore()->wait(-1))
        {
            PyErr_Format(PyExc_RuntimeError, "timeout while getting attribute");
            return NULL;
        }
        retValue += locker.getSemaphore()->returnValue;

        if(*objectID == 0)
        {
            PyErr_Format(PyExc_RuntimeError, "attribute is no widget name of this user interface");
            return NULL;
        }
    }
    else if(PyErr_Clear(), PyArg_ParseTuple(args,"O!s", &PyUiDialogMetaObject::PyMetaObjectType, &baseObject, &objName))
    {
        PyUiDialogMetaObject::PyMetaObject* metaObj = (PyUiDialogMetaObject::PyMetaObject*)baseObject;
        if(metaObj->objectID < 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this instance of UiDialogMetaObject");
            return NULL;
        }
        uiDialog = metaObj->dialog;

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "getChildObject2", Q_ARG(unsigned int, static_cast<unsigned int>(metaObj->objectID)), Q_ARG(QString, QString(objName)), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if(!locker.getSemaphore()->wait(-1))
        {
            PyErr_Format(PyExc_RuntimeError, "timeout while getting attribute");
            return NULL;
        }
        retValue += locker.getSemaphore()->returnValue;

        if(*objectID == 0)
        {
            PyErr_Format(PyExc_RuntimeError, "attribute is no widget name of this user interface");
            return NULL;
        }
    }
    else
    {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "Arguments must be an object of type uiDialog or uiDialogMetaObject followed by an object name (string).");
        return NULL;
    }

    PythonUiDialog::PyUiDialog *tmp = self->dialog;
    Py_INCREF(uiDialog);
    self->dialog =uiDialog;
    Py_XDECREF(tmp);
    self->objName = new char[strlen(objName)+1];
    strcpy(self->objName, objName);
    self->objectID = *objectID;
    DELETE_AND_SET_NULL(self->methodList);
    DELETE_AND_SET_NULL(self->methodListHash);
    self->methodDescriptionListLoaded=false;

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectName_doc,"");
PyObject* PyUiDialogMetaObject::PyMetaObject_name(PyMetaObject* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("UiDialogMetaObject");
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PyUiDialogMetaObject::PyMetaObject_repr(PyMetaObject *self)
{
    if(self->objName)
    {
        return PyUnicode_FromFormat("UiDialogMetaObject(object name: '%s')", self->objName);
    }
    else
    {
        return PyUnicode_FromString("UiDialogMetaObject(unknown object name)");
    }
}

//--------------------------------------------------------------------------------------------
// mapping methods
//--------------------------------------------------------------------------------------------
int PyUiDialogMetaObject::PyMetaObject_mappingLength(PyMetaObject* self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return 0;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<int> classInfoCount(new int());
    QSharedPointer<int> enumeratorCount(new int());
    QSharedPointer<int> methodCount(new int());
    QSharedPointer<int> propertiesCount(new int());
    *classInfoCount = -1;
    *enumeratorCount = -1;
    *methodCount = -1;
    *propertiesCount = -1;

    QMetaObject::invokeMethod(uiOrga, "widgetMetaObjectCounts", Q_ARG(unsigned int, static_cast<unsigned int>(self->objectID)), Q_ARG(QSharedPointer<int>, classInfoCount), Q_ARG(QSharedPointer<int>, enumeratorCount), Q_ARG(QSharedPointer<int>, methodCount),Q_ARG(QSharedPointer<int>, propertiesCount), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while getting number of properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    return *propertiesCount; //nr of properties in the corresponding QMetaObject
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PyUiDialogMetaObject::PyMetaObject_mappingGetElem(PyMetaObject* self, PyObject* key)
{
    QStringList propNames;
    bool ok = false;
    QString propName = PythonQtConversion::PyObjGetString(key,false,ok);
    if(!ok)
    {
        PyErr_Format(PyExc_RuntimeError, "property name string could not be parsed.");
        return NULL;
    }
    propNames << propName;

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());
    for(int i = 0 ; i < propNames.count() ; i++)
    {
        (*retPropMap)[propNames.at(i)] = QVariant();
    }

    QMetaObject::invokeMethod(uiOrga, "readProperties", Q_ARG(unsigned int, self->objectID), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    return PythonQtConversion::QVariantToPyObject(retPropMap->value(propNames[0]));
}

//----------------------------------------------------------------------------------------------------------------------------------
int PyUiDialogMetaObject::PyMetaObject_mappingSetElem(PyMetaObject* self, PyObject* key, PyObject* value)
{
    QString keyString;
    bool ok = false;
    QVariantMap propMap;
    QVariant valueV;

    keyString = PythonQtConversion::PyObjGetString(key,false,ok);

    if(!ok)
    {
        PyErr_Format(PyExc_RuntimeError, "key must be a string");
        return NULL;
    }

    valueV = PythonQtConversion::PyObjToQVariant(value);
    if(valueV.isValid())
    {
        propMap[keyString] = valueV;
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "property value could not be parsed to QVariant.");
        return NULL;
    } 

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "writeProperties", Q_ARG(unsigned int, self->objectID), Q_ARG(QVariantMap, propMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while writing property");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return -1;

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectCall_doc,"call(slotOrPublicMethod [,argument1, argument2, ...]) -> calls any public slot of this widget or any accessible public method.  \n\
                                  - slotOrPublicMethod is the name of the slot or method (string)\n\
                                  - arguments are the required arguments for this slot, separated by comma");
PyObject* PyUiDialogMetaObject::PyMetaObject_call(PyMetaObject *self, PyObject* args)
{
    int argsSize = PyTuple_Size(args);
    int nrOfParams = argsSize - 1;
    bool ok;
    FctCallParamContainer *paramContainer;

    if(argsSize < 1)
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a function name string, optionally followed by the necessary function parameters");
        return NULL;
    }
    
    QString slotName = PythonQtConversion::PyObjGetString(PyTuple_GetItem(args,0),false,ok);
    if(!ok)
    {
        PyErr_SetString(PyExc_TypeError, "First given parameter cannot be interpreted as string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    if(!loadMethodDescriptionList(self)) return NULL;
   
    //scan for method
    //step 1: check if method exists
    QList<unsigned int> slotIndizes = self->methodListHash->values(slotName);
    if(slotIndizes.size() == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No slot or method with name %s available.", slotName.toAscii().data());
        return NULL;
    }

    //create function container
    paramContainer = new FctCallParamContainer(nrOfParams);
    void *ptr = NULL;
    int typeNr = 0;
    MethodDescription method;
    bool found = false;
    QString possibleSignatures = "";

    for(int i=0;i<slotIndizes.size();i++)
    {
        ok = true;
        method = self->methodList->at(slotIndizes[i]);
        if(method.checkMethod(slotName.toAscii(), nrOfParams))
        {
            paramContainer->initRetArg( method.retType() );

            for(int j=0;j<nrOfParams;j++)
            {
                if(PythonQtConversion::PyObjToVoidPtr(PyTuple_GetItem(args,j+1), &ptr, &typeNr, method.argTypes()[j]))
                {
                    paramContainer->setParamArg(j, ptr, typeNr);
                }
                else
                {
                    ok = false;
                    break;
                }
            }

            if(ok)
            {
                found = true;
                break; //everything ok, we found the method and could convert all given parameters
            }
            else
            {
                possibleSignatures.append(QString("'%1', ").arg( QString::fromAscii(method.signature().data()) ));
            }

        }
        else
        {
            possibleSignatures.append(QString("'%1', ").arg( QString::fromAscii(method.signature().data()) ));
        }
    }

    if(!found)
    {
        DELETE_AND_SET_NULL(paramContainer);
        PyErr_Format(PyExc_RuntimeError, "None of the following possible signatures fit to the given set of parameters: %s", possibleSignatures.toAscii().data());
        return NULL;
    }

    QSharedPointer<FctCallParamContainer> sharedParamContainer(paramContainer); //from now on, do not directly delete paramContainer any more
    ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());

    if(method.type() == QMetaMethod::Slot)
    {
        QMetaObject::invokeMethod(uiOrga, "callSlot", Q_ARG(unsigned int, self->objectID), Q_ARG(int, method.methodIndex()), Q_ARG(QSharedPointer<FctCallParamContainer>, sharedParamContainer), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore()));
    }   
    else if(method.type() == QMetaMethod::Method)
    {
        QMetaObject::invokeMethod(uiOrga, "callMethod", Q_ARG(unsigned int, self->objectID), Q_ARG(int, method.methodIndex()), Q_ARG(QSharedPointer<FctCallParamContainer>, sharedParamContainer), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore()));
    }
    else
    {
        PyErr_Format(PyExc_RuntimeError, "unknown method type.");
        return NULL;
    }

    if(!locker2.getSemaphore()->wait(50000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while calling slot");
        return NULL;
    }

    if(PythonCommon::transformRetValToPyException( locker2.getSemaphore()->returnValue ) == false) return NULL;

    if(sharedParamContainer->getRetType() > 0)
    {
        return PythonQtConversion::ConvertQtValueToPythonInternal(sharedParamContainer->getRetType(), sharedParamContainer->args()[0]);
    }

    Py_RETURN_NONE;

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectConnect_doc,"connect(signalSignature, callableMethod) -> connects the signal of the widget with the given callable python method \n\
                                     The signalSignature must be a valid Qt-signature, the callable method is the name of a bounded or unbounded python method.");
PyObject* PyUiDialogMetaObject::PyMetaObject_connect(PyMetaObject *self, PyObject* args)
{
    const char* signalSignature;
    PyObject *callableMethod;

    if(!PyArg_ParseTuple(args, "sO", &signalSignature, &callableMethod))
    {
        PyErr_Format(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }
    if(!PyCallable_Check(callableMethod))
    {
        PyErr_Format(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    //PyObject *fct = PyMethod_Function(callableMethod); //borrowed

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    QString signature(signalSignature);
    QSharedPointer<int> sigId(new int);
    QSharedPointer<QObject*> objPtr(new QObject*[1]);
    QSharedPointer<IntList> argTypes(new IntList);

    *sigId = -1;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "getSignalIndex", Q_ARG(unsigned int, self->objectID), Q_ARG(QString, signature), Q_ARG(QSharedPointer<int>, sigId), Q_ARG(QSharedPointer<QObject*>, objPtr), Q_ARG(QSharedPointer<IntList>, argTypes), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if(*sigId == -1)
    {
        PyErr_Format(PyExc_RuntimeError, "signal signature is invalid.");
        return NULL;
    }

    if(!self->dialog->signalMapper->addSignalHandler(*objPtr, signalSignature, *sigId, callableMethod, *argTypes))
    {
        PyErr_Format(PyExc_RuntimeError, "the connection could not be established.");
        return NULL;
    }


    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectConnectKeyboardInterrupt_doc,"invokeKeyboardInterrupt(signalSignature) -> connects the given signal with a slot immediately invoking a python interrupt signal. \n\
                                                      signalSignature must be a valid Qt-signature string, e.g. 'clicked(bool)'");
PyObject* PyUiDialogMetaObject::PyMetaObject_connectKeyboardInterrupt(PyMetaObject *self, PyObject* args)
{
    const char* signalSignature;

    if(!PyArg_ParseTuple(args, "s", &signalSignature))
    {
        PyErr_Format(PyExc_TypeError, "Arguments must be a signal signature");
        return NULL;
    }
    
    //PyObject *fct = PyMethod_Function(callableMethod); //borrowed

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    QString signature(signalSignature);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "connectWithKeyboardInterrupt", Q_ARG(unsigned int, self->objectID), Q_ARG(QString, signature), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if(!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectDisconnect_doc,"disconnect(signalSignature, callableMethod) -> disconnects a connection which must have been established with exactly the same parameters.");
PyObject* PyUiDialogMetaObject::PyMetaObject_disconnect(PyMetaObject *self, PyObject* args)
{
    const char* signalSignature;
    PyObject *callableMethod;

    if(!PyArg_ParseTuple(args, "sO", &signalSignature, &callableMethod))
    {
        PyErr_Format(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }
    if(!PyCallable_Check(callableMethod))
    {
        PyErr_Format(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    //PyObject *fct = PyMethod_Function(callableMethod); //borrowed

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    QString signature(signalSignature);
    QSharedPointer<int> sigId(new int);
    QSharedPointer<QObject*> objPtr(new QObject*[1]);
    QSharedPointer<IntList> argTypes(new IntList);

    *sigId = -1;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "getSignalIndex", Q_ARG(unsigned int, self->objectID), Q_ARG(QString, signature), Q_ARG(QSharedPointer<int>, sigId), Q_ARG(QSharedPointer<QObject*>, objPtr), Q_ARG(QSharedPointer<IntList>, argTypes), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if(*sigId == -1)
    {
        PyErr_Format(PyExc_RuntimeError, "signal signature is invalid.");
        return NULL;
    }

    if(!self->dialog->signalMapper->removeSignalHandler(*objPtr, signalSignature, *sigId, callableMethod))
    {
        PyErr_Format(PyExc_RuntimeError, "the connection could not be disconnected.");
        return NULL;
    }


    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectGetProperty_doc,"getProperty(property) -> returns tuple of requested properties (single property or tuple of properties)");
PyObject* PyUiDialogMetaObject::PyMetaObject_getProperties(PyMetaObject *self, PyObject *args)
{
    PyObject *propertyNames = NULL;
    QStringList propNames;
    bool ok = false;

    if(!PyArg_ParseTuple(args, "O", &propertyNames))
    {
        return NULL;
    }

    Py_INCREF(propertyNames);

    if(PyBytes_Check(propertyNames) || PyUnicode_Check(propertyNames))
    {
        QString temp = PythonQtConversion::PyObjGetString(propertyNames, true, ok);
        if(ok)
        {
            propNames << temp;
        }
        else
        {
            Py_XDECREF(propertyNames);
            PyErr_Format(PyExc_RuntimeError, "property name string could not be parsed.");
            return NULL;
        }
    }
    else if(PySequence_Check(propertyNames))
    {
        propNames = PythonQtConversion::PyObjToStringList(propertyNames, true, ok);
        if(!ok)
        {
            Py_XDECREF(propertyNames);
            PyErr_SetString(PyExc_RuntimeError, "property names list or tuple could not be converted to a list of strings");
            return NULL;
        }
    }
    else
    {
        Py_XDECREF(propertyNames);
        PyErr_SetString(PyExc_RuntimeError, "property name must be a string or tuple/list of strings"); 
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        Py_XDECREF(propertyNames);
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());
    for(int i = 0 ; i < propNames.count() ; i++)
    {
        (*retPropMap)[propNames.at(i)] = QVariant();
    }

    QMetaObject::invokeMethod(uiOrga, "readProperties", Q_ARG(unsigned int, self->objectID), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        Py_XDECREF(propertyNames);
        PyErr_SetString(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    PyObject *retObj = PyList_New(propNames.count());
    for(int i = 0 ; i < propNames.count() ; i++)
    {
        PyList_SetItem(retObj,i, PythonQtConversion::QVariantToPyObject(retPropMap->value(propNames.at(i))));
    }

    Py_XDECREF(propertyNames);

    return retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyMetaObjectSetProperty_doc,"setProperty(propertyDict) -> each property in the parameter dictionary is set to the dictionaries value.");
PyObject* PyUiDialogMetaObject::PyMetaObject_setProperties(PyMetaObject *self, PyObject *args)
{
    PyObject *propDict = NULL;
    bool ok = false;
    QVariantMap propMap;

    if(!PyArg_ParseTuple(args, "O!", &PyDict_Type, &propDict))
    {
        return NULL;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    QVariant valueV;
    QString keyS;

    while (PyDict_Next(propDict, &pos, &key, &value)) 
    {
        keyS = PythonQtConversion::PyObjGetString(key,true,ok);
        valueV = PythonQtConversion::PyObjToQVariant(value);
        if(valueV.isValid())
        {
            propMap[keyS] = valueV;
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
            Py_DECREF(propDict);
            return NULL;
        }
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "writeProperties", Q_ARG(unsigned int, static_cast<unsigned int>(self->objectID)), Q_ARG(QVariantMap, propMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(5000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while writing property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PyUiDialogMetaObject::loadMethodDescriptionList(PyMetaObject *self)
{
    if(self->methodDescriptionListLoaded == false)
    {
        UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
        if(uiOrga == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
            return NULL;
        }

        if(self->objectID <= 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this metaObject-instance");
            return NULL;
        }

        QSharedPointer<MethodDescriptionList> methodList(new MethodDescriptionList);
        ItomSharedSemaphoreLocker locker1(new ItomSharedSemaphore());

        QMetaObject::invokeMethod(uiOrga, "getMethodDescriptions", Q_ARG(unsigned int, self->objectID), Q_ARG(QSharedPointer<MethodDescriptionList>, methodList), Q_ARG(ItomSharedSemaphore*, locker1.getSemaphore()));
    
        if(!locker1.getSemaphore()->wait(5000))
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while analysing method description list");
            return false;
        }

        self->methodListHash = new QMultiHash<QString,unsigned int>();
        for(int i=0;i<methodList->size();i++)
        {
            self->methodListHash->insertMulti( methodList->at(i).name() , i );
        }

        self->methodDescriptionListLoaded = true;
        self->methodList = new MethodDescriptionList(*methodList);
    }

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PyUiDialogMetaObject::PyMetaObject_getattro(PyMetaObject *self, PyObject *args)
{
    PyObject *ret = PyObject_GenericGetAttr((PyObject*)self,args);
    if(ret != NULL)
    {
        return ret;
    }
    PyErr_Clear(); //genericgetattr throws an error, if attribute is not available, which it isn't for attributes pointing to widgetNames

    //return new instance of PyMetaObject
    PyUiDialogMetaObject::PyMetaObject *pyMetaObject;
    PyObject *arg2 = Py_BuildValue("OO", self, args);

    pyMetaObject = (PyUiDialogMetaObject::PyMetaObject *)PyObject_CallObject((PyObject *)&PyUiDialogMetaObject::PyMetaObjectType, arg2);
    Py_DECREF(arg2);

    if(pyMetaObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Could not create uiDialogMetaObject of requested widget");
        return NULL;
    }

    if(PyErr_Occurred())
    {
        Py_XDECREF(pyMetaObject);
        pyMetaObject = NULL;
    }

    return (PyObject *)pyMetaObject;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PyUiDialogMetaObject::PyMetaObject_setattro(PyMetaObject * /*self*/, PyObject * /*args*/)
{
    PyErr_SetString(PyExc_TypeError, "It is not possible to assign another widget to the given widget in the user interface.");
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PyUiDialogMetaObject::PyMetaObject_methods[] = {
        {"name", (PyCFunction)PyMetaObject_name, METH_NOARGS, pyMetaObjectName_doc},
        {"call", (PyCFunction)PyMetaObject_call, METH_VARARGS, pyMetaObjectCall_doc},
        {"connect", (PyCFunction)PyMetaObject_connect, METH_VARARGS, pyMetaObjectConnect_doc},
        {"disconnect", (PyCFunction)PyMetaObject_disconnect, METH_VARARGS, pyMetaObjectDisconnect_doc},
        {"setProperty", (PyCFunction)PyMetaObject_setProperties, METH_VARARGS, pyMetaObjectSetProperty_doc},
        {"getProperty", (PyCFunction)PyMetaObject_getProperties, METH_VARARGS, pyMetaObjectGetProperty_doc},
        {"invokeKeyboardInterrupt", (PyCFunction)PyMetaObject_connectKeyboardInterrupt, METH_VARARGS, pyMetaObjectConnectKeyboardInterrupt_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PyUiDialogMetaObject::PyMetaObject_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PyUiDialogMetaObject::PyMetaObjectModule = {
        PyModuleDef_HEAD_INIT,
        "uiDialogMetaObject",
        "MetaObject of any uiDialog widget",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PyUiDialogMetaObject::PyMetaObject_getseters[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PyUiDialogMetaObject::PyMetaObjectType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.uiDialogMetaObject",             /* tp_name */
        sizeof(PyMetaObject),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyMetaObject_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyMetaObject_repr,         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        &PyMetaObject_mappingProtocol,   /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        (getattrofunc)PyMetaObject_getattro, /* tp_getattro */
        (setattrofunc)PyMetaObject_setattro,  /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        pyMetaObjectInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,            /* tp_richcompare */
        offsetof(PyMetaObject, weakreflist),    /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyMetaObject_methods,             /* tp_methods */
        PyMetaObject_members,             /* tp_members */
        PyMetaObject_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyMetaObject_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyMetaObject_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMappingMethods PyUiDialogMetaObject::PyMetaObject_mappingProtocol = {
    (lenfunc)PyMetaObject_mappingLength,
    (binaryfunc)PyMetaObject_mappingGetElem,
    (objobjargproc)PyMetaObject_mappingSetElem
};

//----------------------------------------------------------------------------------------------------------------------------------
void PyUiDialogMetaObject::PyMetaObject_addTpDict(PyObject * /*tp_dict*/)
{
    //nothing
}

//----------------------------------------------------------------------------------------------------------------------------------