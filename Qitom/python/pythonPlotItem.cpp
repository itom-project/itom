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

#include "pythonPlotItem.h"

#include "structmember.h"
#include "pythonFigure.h"

#include "../organizer/uiOrganizer.h"
#include "../AppManagement.h"

namespace ito
{
// -------------------------------------------------------------------------------------------------------------------------
//
//  PyFigure
//
// -------------------------------------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------------------------------------------
void PythonPlotItem::PyPlotItem_dealloc(PyPlotItem* self)
{
    PythonUi::PyUiItemType.tp_dealloc( (PyObject*)self );
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonPlotItem::PyPlotItem_new(PyTypeObject *type, PyObject * args, PyObject * kwds)
{
    PyPlotItem *self = (PyPlotItem*)PythonUi::PyUiItemType.tp_new(type,args,kwds);
    if(self != NULL)
    {
        //self->signalMapper = NULL;
    }

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItemInit_doc,"figure([handle]) -> plot figure\n\
\n\
doc");
int PythonPlotItem::PyPlotItem_init(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    PythonFigure::PyFigure *figure = NULL;
    unsigned int subplotIndex = 0;
    ito::RetVal retval;

    if(!PyArg_ParseTuple(args, "O!I", &PythonFigure::PyFigureType, &figure, &subplotIndex))
    {
        return NULL;
    }


    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return -1;
    }

    QSharedPointer<unsigned int> objectID(new unsigned int);
    QSharedPointer<QByteArray> widgetClassName(new QByteArray());
    QSharedPointer<QByteArray> objectName(new QByteArray());
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QMetaObject::invokeMethod(uiOrga, "getSubplot", Q_ARG(QSharedPointer<unsigned int>, figure->guardedFigHandle), Q_ARG(unsigned int, subplotIndex), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(QSharedPointer<QByteArray>, objectName), Q_ARG(QSharedPointer<QByteArray>, widgetClassName), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    locker.getSemaphore()->wait(-1);
    retval += locker.getSemaphore()->returnValue;

    if(!PythonCommon::transformRetValToPyException(retval))
    {
        return -1;
    }

    Py_XINCREF(figure);
    self->uiItem.baseItem = (PyObject*)figure;
    DELETE_AND_SET_NULL_ARRAY(self->uiItem.objName);
    self->uiItem.objName = new char[objectName->size()+1];
    strcpy(self->uiItem.objName, objectName->data() );
    DELETE_AND_SET_NULL_ARRAY(self->uiItem.widgetClassName);
    self->uiItem.widgetClassName = new char[widgetClassName->size()+1];
    strcpy(self->uiItem.widgetClassName, widgetClassName->data() );
    self->uiItem.objectID = *objectID;

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonPlotItem::PyPlotItem_repr(PyPlotItem *self)
{
    return PyUnicode_FromFormat("PlotItem(%U)", PythonUi::PyUiItemType.tp_repr((PyObject*)self) );
}


//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonPlotItem::PyPlotItem_methods[] = {
    //{"createNewPluginWidget", (PyCFunction)PyUi_createNewAlgoWidget, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiCreateNewPluginWidget_doc},
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonPlotItem::PyPlotItem_members[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonPlotItem::PyPlotItemModule = {
    PyModuleDef_HEAD_INIT,
    "plotItem",
    "itom plotItem type in python",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonPlotItem::PyPlotItem_getseters[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonPlotItem::PyPlotItemType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "itom.plotItem",             /* tp_name */
    sizeof(PyPlotItem),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyPlotItem_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyPlotItem_repr,         /* tp_repr */
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
    pyPlotItemInit_doc,      /* tp_doc */
    0,    	                /* tp_traverse */
    0,		                /* tp_clear */
    0,                      /* tp_richcompare */
    0,		                /* tp_weaklistoffset */
    0,		                /* tp_iter */
    0,		                /* tp_iternext */
    PyPlotItem_methods,             /* tp_methods */
    PyPlotItem_members,             /* tp_members */
    PyPlotItem_getseters,            /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyPlotItem_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyPlotItem_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//----------------------------------------------------------------------------------------------------------------------------------
void PythonPlotItem::PyPlotItem_addTpDict(PyObject *tp_dict)
{
    
}


} //end namespace ito

