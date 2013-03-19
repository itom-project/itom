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

#include "pythonFigure.h"

#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "structmember.h"
#endif

#include "../global.h"
#include "../organizer/uiOrganizer.h"

#include "pythonQtConversion.h"
#include "AppManagement.h"

#include <qsharedpointer.h>
#include <qmessagebox.h>
#include <qmetaobject.h>


namespace ito
{
// -------------------------------------------------------------------------------------------------------------------------
//
//  PyFigure
//
// -------------------------------------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------------------------------------------OK
void PythonFigure::PyFigure_dealloc(PyFigure* self)
{
    self->guardedFigHandle.clear(); //if reference of semaphore drops to zero, the static method threadSafeDeleteUi of UiOrganizer is called that will finally delete the figure

    DELETE_AND_SET_NULL( self->signalMapper );

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonFigure::PyFigure_new(PyTypeObject *type, PyObject * args, PyObject * kwds)
{
    PyFigure* self = (PyFigure *)type->tp_alloc(type, 0);
    if(self != NULL)
    {
        self->guardedFigHandle.clear(); //default: invalid
        self->rows = 0;
        self->cols = 0;
        self->signalMapper = NULL;
    }

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyFigureInit_doc,"figure([handle]) -> plot figure\n\
\n\
doc");
int PythonFigure::PyFigure_init(PyFigure *self, PyObject *args, PyObject *kwds)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    int handle = -1;

    if(!PyArg_ParseTuple(args,"|i", &handle))
    {
        return -1;
    }

    QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>() );
    QSharedPointer<unsigned int> initSlotCount(new unsigned int);
    QSharedPointer<unsigned int> objectID(new unsigned int);
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue;

    if(handle != -1)
    {
        *guardedFigHandle = QSharedPointer<unsigned int>( new unsigned int );
        **guardedFigHandle = handle;
    }

    *initSlotCount = 0;

    QMetaObject::invokeMethod(uiOrga, "createFigure",Q_ARG(QSharedPointer< QSharedPointer<unsigned int> >,guardedFigHandle), Q_ARG(QSharedPointer<unsigned int>, initSlotCount), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(60000))
    {
        PyErr_Format(PyExc_RuntimeError, "timeout while opening figure");
        return -1;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return -1;

    self->guardedFigHandle = *guardedFigHandle;
    DELETE_AND_SET_NULL( self->signalMapper );
    self->signalMapper = new PythonQtSignalMapper(*initSlotCount);

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonFigure::PyFigure_repr(PyFigure *self)
{
    PyObject *result;
    if(self->guardedFigHandle.isNull())
    {
        result = PyUnicode_FromFormat("Figure(empty)");
    }
    else
    {
        result = PyUnicode_FromFormat("Figure(handle: %i)", *(self->guardedFigHandle) );
    }
    return result;
}


//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonFigure::PyFigure_methods[] = {
        //{"show", (PyCFunction)PyUi_show,     METH_VARARGS, pyUiShow_doc},
        //{"hide", (PyCFunction)PyUi_hide, METH_NOARGS, pyUiHide_doc},
        //{"isVisible", (PyCFunction)PyUi_isVisible, METH_NOARGS, pyUiIsVisible_doc},
        //
        //{"getDouble",(PyCFunction)PyUi_getDouble, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetDouble_doc},
        //{"getInt",(PyCFunction)PyUi_getInt, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetInt_doc},
        //{"getItem",(PyCFunction)PyUi_getItem, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetItem_doc},
        //{"getText",(PyCFunction)PyUi_getText, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetText_doc},
        //{"msgInformation", (PyCFunction)PyUi_msgInformation, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgInformation_doc},
        //{"msgQuestion", (PyCFunction)PyUi_msgQuestion, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgQuestion_doc},
        //{"msgWarning", (PyCFunction)PyUi_msgWarning, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgWarning_doc},
        //{"msgCritical", (PyCFunction)PyUi_msgCritical, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgCritical_doc},
        //{"getExistingDirectory", (PyCFunction)PyUi_getExistingDirectory, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetExistingDirectory_doc},
        //{"getOpenFileName", (PyCFunction)PyUi_getOpenFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetOpenFileName_doc},
        //{"getSaveFileName", (PyCFunction)PyUi_getSaveFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetSaveFileName_doc},
        //{"createNewPluginWidget", (PyCFunction)PyUi_createNewAlgoWidget, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiCreateNewPluginWidget_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonFigure::PyFigure_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonFigure::PyFigureModule = {
        PyModuleDef_HEAD_INIT,
        "figure",
        "itom figure type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonFigure::PyFigure_getseters[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject PythonFigure::PyFigureType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.figure",             /* tp_name */
        sizeof(PyFigure),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyFigure_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyFigure_repr,         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0, /* tp_getattro */
        0,  /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        pyFigureInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,    	               /* tp_traverse */
        0,		               /* tp_clear */
        0,            /* tp_richcompare */
        0,		               /* tp_weaklistoffset */
        0,		               /* tp_iter */
        0,		               /* tp_iternext */
        PyFigure_methods,             /* tp_methods */
        PyFigure_members,             /* tp_members */
        PyFigure_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyFigure_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyFigure_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//----------------------------------------------------------------------------------------------------------------------------------
void PythonFigure::PyFigure_addTpDict(PyObject *tp_dict)
{
    //PyObject *value;
    //QMetaObject metaObject = QMessageBox::staticMetaObject;
    //QMetaEnum metaEnum = metaObject.enumerator( metaObject.indexOfEnumerator( "StandardButtons" ));
    //QString key;
    ////auto-parsing of StandardButtons-enumeration for key-value-pairs
    //for(int i = 0 ; i < metaEnum.keyCount() ; i++)
    //{
    //    value = Py_BuildValue("i", metaEnum.value(i));
    //    key = metaEnum.key(i);
    //    key.prepend("MsgBox"); //Button-Constants will be accessed by ui.MsgBoxOk, ui.MsgBoxError...
    //    PyDict_SetItemString(tp_dict, key.toAscii().data(), value);
    //    Py_DECREF(value);
    //}

    ////add dialog types
    //value = Py_BuildValue("i", 0);
    //PyDict_SetItemString(tp_dict, "TYPEDIALOG", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 1);
    //PyDict_SetItemString(tp_dict, "TYPEWINDOW", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 2);
    //PyDict_SetItemString(tp_dict, "TYPEDOCKWIDGET", value);
    //Py_DECREF(value);

    ////add button orientation
    //value = Py_BuildValue("i", 0);
    //PyDict_SetItemString(tp_dict, "BUTTONBAR_NO", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 1);
    //PyDict_SetItemString(tp_dict, "BUTTONBAR_HORIZONTAL", value);
    //Py_DECREF(value);
    //value = Py_BuildValue("i", 2);
    //PyDict_SetItemString(tp_dict, "BUTTONBAR_VERTICAL", value);
    //Py_DECREF(value);
}


} //end namespace ito

