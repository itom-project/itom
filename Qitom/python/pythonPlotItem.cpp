/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include "pythonPlotItem.h"

#include "structmember.h"
#include "pythonFigure.h"
#include "pythonEngine.h"
#include "pythonShape.h"

#include "../organizer/uiOrganizer.h"
#include "../AppManagement.h"
#include "common/shape.h"

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
PyDoc_STRVAR(pyPlotItemInit_doc,"plotItem(figure | uiItem[, subplotIdx]) -> instance of the plot or subplot of a figure.\n\
\n\
Use can use this constructor to access any plot or subplot (if more than one plot) of a figure. The subplotIndex \n\
row-wisely addresses the subplots, beginning with 0. \n\
\n\
As second possibility, the constructor can be used to cast 'uiItem' to 'plotItem' in order to access methods like 'pickPoints' \n\
or 'drawAndPickElement'. \n\
\n\
Parameters \n\
------------ \n\
figure : {???} \n\
\n\
subplotIdx: {???}\n\
\n\
");
int PythonPlotItem::PyPlotItem_init(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    PythonFigure::PyFigure *figure = NULL;
    PythonUi::PyUiItem *uiItem = NULL;
    unsigned int subplotIndex = 0;
    ito::RetVal retval;
    unsigned int objectID = 0;
    

    const char *kwlist1[] = {"figure", "subplotIdx", NULL};
    const char *kwlist2[] = {"figure", "objectID", NULL};
    const char *kwlist3[] = {"uiItem", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O!|I", const_cast<char**>(kwlist1), &PythonFigure::PyFigureType, &figure, &subplotIndex))
    {
        PyErr_Clear();
        if(!PyArg_ParseTupleAndKeywords(args,kwds,"|O!I", const_cast<char**>(kwlist2), &PythonFigure::PyFigureType, &figure, &objectID))
        {
            PyErr_Clear();
            if(!PyArg_ParseTupleAndKeywords(args,kwds,"O!", const_cast<char**>(kwlist3), &PythonUi::PyUiItemType, &uiItem))
            {
                return -1;
            }
        }
        else
        {
            if (objectID == 0 && figure == NULL)
            {
                //this avoid a crash if plotItem is instantiated without any arguments
                PyErr_SetString(PyExc_RuntimeError, "PlotItem requires an existing figure as argument and / or a valid subplotIdx or objectID");
                return -1;
            }
        }
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return -1;
    }

    if (uiItem)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QSharedPointer<uint> isFigureItem(new uint);
        QMetaObject::invokeMethod(uiOrga, "isFigureItem", Q_ARG(uint, uiItem->objectID), Q_ARG(QSharedPointer<uint>, isFigureItem), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        locker->wait(-1);
        
        if ((*isFigureItem) > 0)
        {
            //copy uiItem to member uiItem
            Py_XINCREF(uiItem->baseItem);
            self->uiItem.baseItem = uiItem->baseItem;
            DELETE_AND_SET_NULL_ARRAY(self->uiItem.objName);
            self->uiItem.objName = _strdup(uiItem->objName);
            DELETE_AND_SET_NULL_ARRAY(self->uiItem.widgetClassName);
            self->uiItem.widgetClassName = _strdup(uiItem->widgetClassName);
            self->uiItem.objectID = uiItem->objectID;
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "given uiItem cannot be cast to plotItem (no valid plot detected).");
            return -1;
        }
    }
    else
    {
        QSharedPointer<unsigned int> objectIDShared(new unsigned int);
        QSharedPointer<QByteArray> widgetClassName(new QByteArray());
        QSharedPointer<QByteArray> objectName(new QByteArray());

        if(objectID == 0)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(uiOrga, "getSubplot", Q_ARG(QSharedPointer<uint>, figure->guardedFigHandle), Q_ARG(uint, subplotIndex), Q_ARG(QSharedPointer<uint>, objectIDShared), Q_ARG(QSharedPointer<QByteArray>, objectName), Q_ARG(QSharedPointer<QByteArray>, widgetClassName), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

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
            QMetaObject::invokeMethod(uiOrga, "getObjectAndWidgetName", Q_ARG(uint, objectID), Q_ARG(QSharedPointer<QByteArray>, objectName), Q_ARG(QSharedPointer<QByteArray>, widgetClassName), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

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
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonPlotItem::PyPlotItem_repr(PyPlotItem *self)
{
    return PyUnicode_FromFormat("PlotItem(%U)", PythonUi::PyUiItemType.tp_repr((PyObject*)self));
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
        retval += ito::RetVal(ito::retError, 0, QObject::tr("data object cannot be converted to a shared data object").toLatin1().data());
    }
    else
    {
        QSharedPointer<QVector<ito::Shape> > shapes(new QVector<ito::Shape>());
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "figurePickPoints", Q_ARG(uint, self->uiItem.objectID), Q_ARG(QSharedPointer<QVector<ito::Shape> >, shapes), Q_ARG(int, maxNrPoints), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

        bool finished = false;

        while(!finished)
        {
            if (PythonEngine::isInterruptQueued())
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("pick points operation interrupted by user").toLatin1().data());
                QMetaObject::invokeMethod(uiOrga, "figurePickPointsInterrupt", Q_ARG(uint, self->uiItem.objectID)); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                finished = locker.getSemaphore()->wait(2000);
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

        if (!retval.containsError() &&shapes->size() > 0)
        {
            QPolygonF polygon = shapes->at(0).contour();
            ito::DataObject obj(2, polygon.size(), ito::tFloat64);
            ito::float64 *xPtr = obj.rowPtr<ito::float64>(0, 0);
            ito::float64 *yPtr = obj.rowPtr<ito::float64>(0, 1);

            for (int i = 0; i < polygon.size(); ++i)
            {
                xPtr[i] = polygon[i].x();
                yPtr[i] = polygon[i].y();
    }

            *coords = obj;
        }
        else
        {
            *coords = ito::DataObject();
        }
    }

    if(!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItem_drawAndPickElement_doc,"drawAndPickElement(elementType [,maxNrElements]) -> method to let the user draw geometric elements on a plot (only if plot supports this) \n\
\n\
This method lets the user select one or multiple elements of type (up to maxNrElements) at the current plot (if the plot supports this).\n\
\n\
Parameters\n\
-----------\n\
elementType : {int} \n\
    The element type to plot according to ito::PrimitiveContainer::tPrimitive.\n\
maxNrElements: {int}, optional \n\
    let the user select up to this number of points [default: infinity]. Selection can be stopped pressing Space or Esc. \n\
\n\
Return \n\
-------- \n\
Tuple of class itom.shape for all created geometric shapes.");
/*static*/ PyObject* PythonPlotItem::PyPlotItem_drawAndPickElement(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"elementType", "maxNrElements", NULL};
    ito::RetVal retval;
    int maxNrPoints = -1;
    int elementType = -1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds,"i|i",const_cast<char**>(kwlist), &elementType, &maxNrPoints))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    
    switch(elementType & ito::Shape::TypeMask)
    {
        case ito::Shape::Polygon:
//            PyErr_SetString(PyExc_RuntimeError, "Drawing of element type currently not supported");
//            return NULL;
        case ito::Shape::Square:
        case ito::Shape::Circle:
        case ito::Shape::MultiPointPick:
        case ito::Shape::Point:
        case ito::Shape::Line:
        case ito::Shape::Rectangle:
        case ito::Shape::Ellipse:
            break;
    }

    QSharedPointer<QVector<ito::Shape> > shapes(new QVector<ito::Shape>());

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QMetaObject::invokeMethod(uiOrga, "figureDrawGeometricShapes", Q_ARG(uint, self->uiItem.objectID), Q_ARG(QSharedPointer<QVector<ito::Shape> >, shapes), Q_ARG(int, elementType), Q_ARG(int, maxNrPoints), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

        bool finished = false;

        while(!finished)
        {
            if (PythonEngine::isInterruptQueued()) //PyErr_CheckSignals())
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("draw points operation interrupted by user").toLatin1().data());
                QMetaObject::invokeMethod(uiOrga, "figurePickPointsInterrupt", Q_ARG(uint, self->uiItem.objectID)); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                finished = locker.getSemaphore()->wait(2000);
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

    if(!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }

    PyObject *tuple = PyTuple_New(shapes->size());
    for (int i = 0; i < shapes->size(); ++i)
    {
        PyTuple_SetItem(tuple, i, ito::PythonShape::createPyShape(shapes->at(i))); //steals reference
    }

    return tuple;
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
    0,                        /* tp_traverse */
    0,                        /* tp_clear */
    0,                      /* tp_richcompare */
    0,                        /* tp_weaklistoffset */
    0,                        /* tp_iter */
    0,                        /* tp_iternext */
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
    PyObject *value;
    
    value = Py_BuildValue("i",ito::Shape::MultiPointPick);
    PyDict_SetItemString(tp_dict, "PrimitiveMultiPointPick", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Point);
    PyDict_SetItemString(tp_dict, "PrimitivePoint", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Line);
    PyDict_SetItemString(tp_dict, "PrimitiveLine", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Rectangle);
    PyDict_SetItemString(tp_dict, "PrimitiveRectangle", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Square);
    PyDict_SetItemString(tp_dict, "PrimitiveSquare", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Ellipse);
    PyDict_SetItemString(tp_dict, "PrimitiveEllipse", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Circle);
    PyDict_SetItemString(tp_dict, "PrimitiveCircle", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::Shape::Polygon);
    PyDict_SetItemString(tp_dict, "PrimitivePolygon", value);
    Py_DECREF(value);
}


} //end namespace ito

