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
#include "common/sharedStructuresPrimitives.h"

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
PyDoc_STRVAR(pyPlotItemInit_doc,"plotItem(figure[, subplotIdx]) -> instance of the plot or subplot of a figure.\n\
\n\
Use can use this constructor to access any plot or subplot (if more than one plot) of a figure. The subplotIndex \n\
row-wisely addresses the subplots, beginning with 0. \n\
\n\
Parameters \n\
------------ \n\
doc");
int PythonPlotItem::PyPlotItem_init(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    PythonFigure::PyFigure *figure = NULL;
    unsigned int subplotIndex = 0;
    ito::RetVal retval;
    unsigned int objectID = 0;

    const char *kwlist1[] = {"figure", "subplotIdx", NULL};
    const char *kwlist2[] = {"figure", "objectID", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!|I", const_cast<char**>(kwlist1), &PythonFigure::PyFigureType, &figure, &subplotIndex))
    {
        PyErr_Clear();
        if(!PyArg_ParseTupleAndKeywords(args,kwds,"|O!I", const_cast<char**>(kwlist2), &PythonFigure::PyFigureType, &figure, &objectID))
        {
            return -1;
        }
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return -1;
    }

    QSharedPointer<unsigned int> objectIDShared(new unsigned int);
    QSharedPointer<QByteArray> widgetClassName(new QByteArray());
    QSharedPointer<QByteArray> objectName(new QByteArray());

    if(objectID == 0)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "getSubplot", Q_ARG(QSharedPointer<unsigned int>, figure->guardedFigHandle), Q_ARG(unsigned int, subplotIndex), Q_ARG(QSharedPointer<unsigned int>, objectIDShared), Q_ARG(QSharedPointer<QByteArray>, objectName), Q_ARG(QSharedPointer<QByteArray>, widgetClassName), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        locker.getSemaphore()->wait(-1);
        retval += locker.getSemaphore()->returnValue;

        if(!PythonCommon::transformRetValToPyException(retval))
        {
            return -1;
        }

        objectID = *objectIDShared;
    }
    else
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "getObjectInfo", Q_ARG(unsigned int, objectID), Q_ARG(QSharedPointer<QByteArray>, objectName), Q_ARG(QSharedPointer<QByteArray>, widgetClassName), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        locker.getSemaphore()->wait(-1);
        retval += locker.getSemaphore()->returnValue;

        if(!PythonCommon::transformRetValToPyException(retval))
        {
            return -1;
        }
    }

    Py_XINCREF(figure);
    self->uiItem.baseItem = (PyObject*)figure;
    DELETE_AND_SET_NULL_ARRAY(self->uiItem.objName);
    self->uiItem.objName = new char[objectName->size()+1];
    strcpy_s(self->uiItem.objName, objectName->size()+1, objectName->data() );
    DELETE_AND_SET_NULL_ARRAY(self->uiItem.widgetClassName);
    self->uiItem.widgetClassName = new char[widgetClassName->size()+1];
    strcpy_s(self->uiItem.widgetClassName, widgetClassName->size()+1, widgetClassName->data() );
    self->uiItem.objectID = objectID;

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonPlotItem::PyPlotItem_repr(PyPlotItem *self)
{
    return PyUnicode_FromFormat("PlotItem(%U)", PythonUi::PyUiItemType.tp_repr((PyObject*)self) );
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItem_pickPoints_doc,"pickPoints(points [,maxNrPoints]) -> method to let the user pick points on a plot (only if plot supports this) \n\
\n\
This method lets the user select one or multiple points (up to maxNrPoints) at the current plot (if the plot supports this).\n\
\n\
Parameters\n\
-----------\n\
points : {DataObject} \n\
    resulting data object containing the 2D positions of the selected points [2 x nrOfSelectedPoints].\n\
maxNrPoints: {int}, optional \n\
    let the user select up to this number of points [default: infinity]. Selection can be stopped pressing Space or Esc.");
/*static*/ PyObject* PythonPlotItem::PyPlotItem_pickPoints(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"points", "maxNrPoints", NULL};
    ito::RetVal retval;
    PyObject *dataObject = NULL;
    int maxNrPoints = -1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds,"O!|i",const_cast<char**>(kwlist), &PythonDataObject::PyDataObjectType, &dataObject, &maxNrPoints))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    
    bool ok;
    QSharedPointer<ito::DataObject> coords = PythonQtConversion::PyObjGetSharedDataObject(dataObject, ok);

    if (!ok)
    {
        retval += ito::RetVal(ito::retError,0,"data object cannot be converted to a shared data object");
    }
    else
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "figurePickPoints", Q_ARG(unsigned int, self->uiItem.objectID), Q_ARG(QSharedPointer<ito::DataObject>, coords), Q_ARG(int, maxNrPoints), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        bool finished = false;

        while(!finished)
        {
            if (PyErr_CheckSignals())
            {
                retval += ito::RetVal(ito::retError,0,"pick points operation interrupted by user");
                QMetaObject::invokeMethod(uiOrga, "figurePickPointsInterrupt", Q_ARG(unsigned int, self->uiItem.objectID));
            }
            else
            {
                finished = locker.getSemaphore()->wait(200);
            }
        }

        if (finished)
        {
            retval += locker.getSemaphore()->returnValue;
        }
    }

    if(!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItem_drawAndPickElement_doc,"drawAndPickElement(elementType, elementData, [,maxNrElements]) -> method to let the user draw geometric elements on a plot (only if plot supports this) \n\
\n\
This method lets the user select one or multiple elements of type (up to maxNrElements) at the current plot (if the plot supports this).\n\
\n\
Parameters\n\
-----------\n\
elementType : {int} \n\
    The element type to plot according to ito::PrimitiveContainer::tPrimitive.\n\
points : {DataObject} \n\
    resulting data object containing the 2D positions of the selected points [2 x nrOfSelectedPoints].\n\
maxNrPoints: {int}, optional \n\
    let the user select up to this number of points [default: infinity]. Selection can be stopped pressing Space or Esc.");
/*static*/ PyObject* PythonPlotItem::PyPlotItem_drawAndPickElement(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"elementType", "elementData", "maxNrElements", NULL};
    ito::RetVal retval;
    PyObject *dataObject = NULL;
    int maxNrPoints = -1;
    int elementType = -1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds,"iO!|i",const_cast<char**>(kwlist), &elementType, &PythonDataObject::PyDataObjectType, &dataObject, &maxNrPoints))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    
    switch(elementType)
    {
        case ito::PrimitiveContainer::tSquare:
        case ito::PrimitiveContainer::tCircle:
        case ito::PrimitiveContainer::tPolygon:
            PyErr_SetString(PyExc_RuntimeError, "Drawing of element type currently not supported");
            return NULL;

        case ito::PrimitiveContainer::tMultiPointPick:
        case ito::PrimitiveContainer::tPoint:
        case ito::PrimitiveContainer::tLine:
        case ito::PrimitiveContainer::tRectangle:
        case ito::PrimitiveContainer::tEllipse:
            break;
    }

    bool ok;
    QSharedPointer<ito::DataObject> coords = PythonQtConversion::PyObjGetSharedDataObject(dataObject, ok);

    if (!ok)
    {
        retval += ito::RetVal(ito::retError,0,"data object cannot be converted to a shared data object");
    }
    else
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "figureDrawGeometricElements", Q_ARG(unsigned int, self->uiItem.objectID), Q_ARG(QSharedPointer<ito::DataObject>, coords), Q_ARG(int, elementType), Q_ARG(int, maxNrPoints), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        bool finished = false;

        while(!finished)
        {
            if (PyErr_CheckSignals())
            {
                retval += ito::RetVal(ito::retError,0,"draw points operation interrupted by user");
                QMetaObject::invokeMethod(uiOrga, "figurePickPointsInterrupt", Q_ARG(unsigned int, self->uiItem.objectID));
            }
            else
            {
                finished = locker.getSemaphore()->wait(200);
            }
        }

        if (finished)
        {
            retval += locker.getSemaphore()->returnValue;
        }
    }

    if(!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonPlotItem::PyPlotItem_methods[] = {
    {"pickPoints", (PyCFunction)PyPlotItem_pickPoints, METH_KEYWORDS | METH_VARARGS, pyPlotItem_pickPoints_doc},
    {"drawAndPickElements", (PyCFunction)PyPlotItem_drawAndPickElement, METH_KEYWORDS | METH_VARARGS, pyPlotItem_drawAndPickElement_doc},
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

