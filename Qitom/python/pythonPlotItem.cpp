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
// ------------------------------------------------------------------------------------
//
//  PyFigure
//
// ------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------
void PythonPlotItem::PyPlotItem_dealloc(PyPlotItem* self)
{
    PythonUi::PyUiItemType.tp_dealloc( (PyObject*)self );
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlotItem::PyPlotItem_new(PyTypeObject *type, PyObject * args, PyObject * kwds)
{
    PyPlotItem *self = (PyPlotItem*)PythonUi::PyUiItemType.tp_new(type,args,kwds);
    if(self != NULL)
    {
        //self->signalMapper = NULL;
    }

    return (PyObject *)self;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItemInit_doc,"plotItem(figure, subplotIdx = 0) -> plotItem \\\n\
plotItem(uiItem) -> plotItem \\\n\
plotItem(figure, objectID = 0) -> plotItem \n\
\n\
The plotItem is a special :class:`uiItem` and represents itom plot designer widgets. \n\
\n\
This constructor can be used to get a reference of a plot in a :class:`figure`. \n\
The plot can also be in a subplot area of a figure. There are two main uses cases \n\
for the direct usage of this constructor: \n\
\n\
1. If a reference to a general :class:`uiItem` is available, but the referenced \n\
   widget / item is a plot, the :class:`uiItem` can be cast to :class:`plotItem` \n\
   such that additional methods like :meth:`pickPoints` or :meth:`drawAndPickElements` \n\
   become available. \n\
2. If a reference to a :class:`figure` is available, that contains one or more \n\
   plots, the reference to one of these plots can be obtained. The ``subplotIdx`` \n\
   indicates the specific plot, that should be referenced. The default is ``0``. \n\
   If the figure consists of more than one subplot, the index counts the subplots \n\
   row-by-row. \n\
\n\
The 3rd variant of this contructor, using the keyword-based argument ``objectID`` \n\
is only internally used and has no further meaning for a high-level usage. However, \n\
if it is used, ``objectID`` must be used as keyword argument, else ``subplotIdx`` is \n\
assumed to be initialized. \n\
\n\
Parameters \n\
---------- \n\
figure : figure \n\
    The :class:`figure` window, that contains an itom plot. \n\
subplotIdx : int, optional \n\
    The area index of the (sub)plot, that should be referenced. This index is \n\
    considered to be row-wise, such that the center plot in the 2nd row with \n\
    three plots in each row has the index ``4``. The first, left, plot in the \n\
    first row has the index ``0``, which is the default. \n\
objectID : int, optional \n\
    If the internal ``objectID`` of a :class:`uiItem` is available, it can be \n\
    tried to be casted to ``plotItem``. \n\
uiItem : uiItem \n\
    Try to cast this :class:`uiItem` to its inherited class :class:`plotItem`. \n\
    A :obj:`RuntimeError` is raised if this cast is not possible. \n\
\n\
Returns \n\
------- \n\
plotItem \n\
    The initialized :class:`plotItem`.");
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
                PyErr_SetString(
                    PyExc_RuntimeError,
                    "PlotItem requires an existing figure as argument and / or a valid subplotIdx or objectID"
                );
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

        QMetaObject::invokeMethod(
            uiOrga,
            "isFigureItem",
            Q_ARG(uint, uiItem->objectID),
            Q_ARG(QSharedPointer<uint>, isFigureItem),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        );

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
            PyErr_SetString(
                PyExc_RuntimeError,
                "given uiItem cannot be cast to plotItem (no valid plot detected)."
            );

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

            QMetaObject::invokeMethod(
                uiOrga,
                "getSubplot",
                Q_ARG(QSharedPointer<uint>, figure->guardedFigHandle),
                Q_ARG(uint, subplotIndex), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                Q_ARG(QSharedPointer<uint>, objectIDShared),
                Q_ARG(QSharedPointer<QByteArray>, objectName),
                Q_ARG(QSharedPointer<QByteArray>, widgetClassName),
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

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
            QMetaObject::invokeMethod(
                uiOrga,
                "getObjectAndWidgetName",
                Q_ARG(uint, objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                Q_ARG(QSharedPointer<QByteArray>, objectName),
                Q_ARG(QSharedPointer<QByteArray>, widgetClassName),
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

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

//-------------------------------------------------------------------------------------
PyObject* PythonPlotItem::PyPlotItem_repr(PyPlotItem *self)
{
    return PyUnicode_FromFormat("PlotItem(%U)", PythonUi::PyUiItemType.tp_repr((PyObject*)self));
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItem_pickPoints_doc,"pickPoints(points, maxNrPoints = -1) \n\
\n\
 -> method to let the user pick points on a plot (only if plot supports this) \n\
\n\
This method lets the user select one or multiple points (up to ``maxNrPoints``) in \n\
the canvas of the current plot (if the plot supports this). The operation can be \n\
aborted by pressing the ``Esc`` key. Then a ``RuntimeError` is raised. It can also \n\
be quit earlier by pressing the ``Space`` key. This is also required if an unlimited \n\
number of points can be selected (``maxNrPoints = -1``). \n\
\n\
A pick-points operation is not supported by all plot classes. It is for instance \n\
available for the classes ``itom1dqwtplot`` (see section :ref:`plot-line`) or \n\
``itom2dqwtplot`` (see section :ref:`plot-image`). \n\
\n\
Parameters\n\
----------\n\
points : dataObject \n\
    This object will be a ``2 x nrOfSelectedPoints`` :class:`dataObject` of dtype \n\
    ``float64`` after the successful call of this method. The first row contains \n\
    the ``x`` coordinates of the selected points, the 2nd row the ``y`` coordinates. \n\
maxNrPoints : int, optional \n\
    Let the user select up to this number of points. The selection \n\
    can be stopped by pressing Space or Esc. ``maxNrPoints`` must be -1 \n\
    for an unlimited number of picked points (default) or a number >= 1. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the specific plot class does not provide the necessary methods to support \n\
    this operation. \n\
RuntimeError \n\
    if the operation has been interrupted by the user. ``points`` is an empty \n\
    :class:`dataObject` then. An interruption can occur if the plot is closed \n\
    or if the user pressed the escape key during the operation.");
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

    if (maxNrPoints < -1 || maxNrPoints == 0)
    {
        PyErr_SetString(PyExc_ValueError, "maxNrPoints must be > 1 or -1 (for an infinite number of points).");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    // conversion must be strict, since the given dataObject should be changed by the subsequent calls.
    bool ok;
    QSharedPointer<ito::DataObject> coords = PythonQtConversion::PyObjGetSharedDataObject(dataObject, true, ok, &retval);

    if (ok)
    {
        QSharedPointer<QVector<ito::Shape> > shapes(new QVector<ito::Shape>());
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

        QMetaObject::invokeMethod(
            uiOrga,
            "figurePickPoints",
            Q_ARG(uint, self->uiItem.objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
            Q_ARG(QSharedPointer<QVector<ito::Shape> >, shapes),
            Q_ARG(int, maxNrPoints),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        bool finished = false;

        while(!finished)
        {
            if (PythonEngine::isInterruptQueued())
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("pick points operation interrupted by user").toLatin1().data());

                QMetaObject::invokeMethod(
                    uiOrga,
                    "figurePickPointsInterrupt",
                    Q_ARG(uint, self->uiItem.objectID)); // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

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
        return nullptr;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlotItem_drawAndPickElements_doc,"drawAndPickElements(elementType, maxNrElements = 1) -> Tuple[shape] \n\
\n\
This method forces the user to draw specific shapes on the canvas of the plot. \n\
\n\
If this method is called, the script execution in paused until the requested \n\
``maxNrElements`` of shapes of the given type ``elementType`` are drawn by the \n\
user on the canvas of the plot. The coordinates of the shapes is then returned \n\
by this method. If the script execution is aborted during the interactive operation \n\
or if the user presses the ``Esc`` key in the plot, this operation is stopped by \n\
a ``RuntimeError``. \n\
\n\
Parameters\n\
----------\n\
elementType : int \n\
    The element type that the user should interactively draw on the canvas of the plot. \n\
    Allowed types are: \n\
    \n\
    * ``plotItem.PrimitivePoint`` \n\
    * ``plotItem.PrimitiveLine`` \n\
    * ``plotItem.PrimitiveRectangle`` \n\
    * ``plotItem.PrimitiveSquare`` \n\
    * ``plotItem.PrimitiveEllipse`` \n\
    * ``plotItem.PrimitiveCircle`` \n\
    * ``plotItem.PrimitivePolygon`` \n\
    \n\
maxNrElements : int, optional \n\
    Number of elements of the given type, the user has to draw. \n\
    The operation can be aborted by clicking the ``Esc`` key. If this is the \n\
    case, a ``RuntimeError`` is raised. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the user interrupts the selection (by pressing the Esc key). \n\
\n\
Returns \n\
------- \n\
tuple of shape \n\
    A tuple with all drawn elements, represented as :class:`shape` objects is returned.");
/*static*/ PyObject* PythonPlotItem::PyPlotItem_drawAndPickElements(PyPlotItem *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"elementType", "maxNrElements", NULL};
    ito::RetVal retval;
    int maxNrElements = -1;
    int elementType = -1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds,"i|i",const_cast<char**>(kwlist), &elementType, &maxNrElements))
    {
        return NULL;
    }

    if (maxNrElements < 1)
    {
        PyErr_SetString(PyExc_ValueError, "maxNrElements must be >= 1.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }


    if (elementType  == ito::Shape::MultiPointPick)
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "elementType 'PrimitiveMultiPointPick' (1) not allowed. Use 'PrimitivePoint' (2) instead or call 'pickPoints'."
        );

        return NULL;
    }

    QSharedPointer<QVector<ito::Shape> > shapes(new QVector<ito::Shape>());
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QMetaObject::invokeMethod(
        uiOrga,
        "figureDrawGeometricShapes",
        Q_ARG(uint, self->uiItem.objectID), //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QSharedPointer<QVector<ito::Shape> >, shapes),
        Q_ARG(int, elementType),
        Q_ARG(int, maxNrElements),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        bool finished = false;

        while(!finished)
        {
            if (PythonEngine::isInterruptQueued())
            {
                retval += ito::RetVal(
                    ito::retError,
                    0,
                    QObject::tr("Draw element operation interrupted by user").toLatin1().data()
                );

                QMetaObject::invokeMethod(
                    uiOrga,
                    "figurePickPointsInterrupt",
                    Q_ARG(uint, self->uiItem.objectID) //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                );

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

//-------------------------------------------------------------------------------------
PyMethodDef PythonPlotItem::PyPlotItem_methods[] = {
    {"pickPoints", (PyCFunction)PyPlotItem_pickPoints, METH_KEYWORDS | METH_VARARGS, pyPlotItem_pickPoints_doc},
    {"drawAndPickElements", (PyCFunction)PyPlotItem_drawAndPickElements, METH_KEYWORDS | METH_VARARGS, pyPlotItem_drawAndPickElements_doc},
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMemberDef PythonPlotItem::PyPlotItem_members[] = {
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonPlotItem::PyPlotItemModule = {
    PyModuleDef_HEAD_INIT,
    "plotItem",
    "itom plotItem type in python",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyGetSetDef PythonPlotItem::PyPlotItem_getseters[] = {
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonPlotItem::PyPlotItemType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "itom.plotItem",           /* tp_name */
    sizeof(PyPlotItem),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyPlotItem_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyPlotItem_repr, /* tp_repr */
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
    pyPlotItemInit_doc,        /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyPlotItem_methods,        /* tp_methods */
    PyPlotItem_members,        /* tp_members */
    PyPlotItem_getseters,      /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyPlotItem_init, /* tp_init */
    0,                         /* tp_alloc */
    PyPlotItem_new             /* tp_new */
};

//-------------------------------------------------------------------------------------
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
