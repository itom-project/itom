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

#include "pythonShape.h"

#include "../global.h"
#include "pythonQtConversion.h"
#include "pythonDataObject.h"
#include "pythonRegion.h"
#include "pythonCommon.h"
#include "pythonRgba.h"
#include "../common/shape.h"
#include "../DataObject/dataobj.h"
#include "../DataObject/dataObjectFuncs.h"
#include <qrect.h>
#include <qiodevice.h>



//---------------------------------------------------------------------------------------------
namespace ito
{

//---------------------------------------------------------------------------------------------
void PythonShape::PyShape_addTpDict(PyObject * tp_dict)
{
    PyObject *value;

    value = Py_BuildValue("i", Shape::Invalid);
    PyDict_SetItemString(tp_dict, "Invalid", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Point);
    PyDict_SetItemString(tp_dict, "Point", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Line);
    PyDict_SetItemString(tp_dict, "Line", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Rectangle);
    PyDict_SetItemString(tp_dict, "Rectangle", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Square);
    PyDict_SetItemString(tp_dict, "Square", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Ellipse);
    PyDict_SetItemString(tp_dict, "Ellipse", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Circle);
    PyDict_SetItemString(tp_dict, "Circle", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::Polygon);
    PyDict_SetItemString(tp_dict, "Polygon", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::MoveLock);
    PyDict_SetItemString(tp_dict, "MoveLock", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::RotateLock);
    PyDict_SetItemString(tp_dict, "RotateLock", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", Shape::ResizeLock);
    PyDict_SetItemString(tp_dict, "ResizeLock", value);
    Py_DECREF(value);
}

//---------------------------------------------------------------------------------------------
void PythonShape::PyShape_dealloc(PyShape* self)
{
    DELETE_AND_SET_NULL(self->shape);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//---------------------------------------------------------------------------------------------
PyObject* PythonShape::PyShape_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyShape* self = (PyShape *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->shape = NULL;
    }

    return (PyObject *)self;
}

//---------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyShape_doc,"shape(type = shape.Invalid, param1 = None, param2 = None, index = -1, name = \"\") -> shape \n\
\n\
Creates a shape object of a specific type. \n\
\n\
A shape object is used to describe a vectorized object, that can for instance be displayed \n\
in plots or might also be passed to different methods, e.g. in order to define a masked \n\
area etc. A :class:`shape` object can also be converted into a :class:`region` object, \n\
however the vector information is then projected onto a raster with a given resolution. \n\
\n\
Depending on the ``type``, the following arguments are allowed, where the first \n\
argument must be given to ``param1`` and the 2nd one to ``param2``: \n\
\n\
* ``shape.Invalid``: - \n\
* ``shape.Point``: point \n\
* ``shape.Line``: start-point, end-point \n\
* ``shape.Rectangle``: top left point, bottom right point \n\
* ``shape.Square``: center point, side-length \n\
* ``shape.Ellipse``: top left point, bottom right point of bounding box \n\
* ``shape.Circle``: center point, radius \n\
* ``shape.Polygon``: 2xM float64 array with M points of polygon \n\
\n\
The parameters ``point``, ``start-point``, ... can be all array-like types (e.g. :class:`dataObject`,\n\
:obj:`list`, :obj:`tuple`, :obj:`np.ndarray`) that can be mapped to float64 and have two elements. \n\
\n\
Another possibility to create a :class:`shape` object for a certain type is to use \n\
one of the following static creation functions: \n\
\n\
* :meth:`createPoint` \n\
* :meth:`createLine` \n\
* :meth:`createCircle` \n\
* :meth:`createEllipse` \n\
* :meth:`createSquare` \n\
* :meth:`createRectangle` \n\
* :meth:`createPolygon` \n\
\n\
During construction, all shapes are aligned with respect to the x- and y-axis. Set a \n\
2d transformation (attribute :attr:`transform`) to rotate and move it. \n\
\n\
Parameters \n\
---------- \n\
type : int \n\
    Type of the shape (see list above). \n\
param1 : list of float or tuple of float or dataObject or numpy.ndarray, optional \n\
    1st initialization argument. This argument is depending on the ``type`` (see list above). \n\
param2 : list of float or tuple of float or dataObject or numpy.ndarray, optional \n\
    2nd initialization argument. This argument is depending on the ``type`` (see list above). \n\
index : int \n\
    index of the shape, or ``-1`` if not further specified (default). \n\
name : str  \n\
    name of the shape, can for instance be displayed next to shapes in plots \n\
    (depending on the parameterization of the plot).");
int PythonShape::PyShape_init(PyShape *self, PyObject *args, PyObject * kwds)
{
    int type = Shape::Invalid;
    PyObject *param1 = NULL;
    PyObject *param2 = NULL;
    int index = -1;
    const char* name = NULL;

    const char *kwlist[] = {"type", "param1", "param2", "index", "name", NULL};

    if (args == NULL && kwds == NULL)
    {
        return 0; //call from createPyShape
    }

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|OOis", const_cast<char**>(kwlist), &(type), &(param1), &(param2), &index, &name))
    {
        return -1;
    }

    self->shape = NULL;
    QPointF pt1, pt2;
    ito::RetVal retval;
    QString name_(name ? name : "");
    bool ok = false;
    double dbl;

    switch (type)
    {
        case Shape::Invalid:
        break;

        case Shape::Point:
            pt1 = PyObject2PointF(param1, retval, "param1");
            if (!retval.containsError())
            {
                self->shape = new ito::Shape(ito::Shape::fromPoint(pt1, index, name_));
            }
        break;

        case Shape::Line:
            pt1 = PyObject2PointF(param1, retval, "param1");
            pt2 = PyObject2PointF(param2, retval, "param2");
            if (!retval.containsError())
            {
                self->shape = new ito::Shape(ito::Shape::fromLine(pt1, pt2, index, name_));
            }
        break;

        case Shape::Rectangle:
            pt1 = PyObject2PointF(param1, retval, "param1");
            pt2 = PyObject2PointF(param2, retval, "param2");
            if (!retval.containsError())
            {
                self->shape = new ito::Shape(ito::Shape::fromRectangle(pt1.rx(), pt1.ry(), pt2.rx(), pt2.ry(), index, name_));
            }
        break;

        case Shape::Square:
            pt1 = PyObject2PointF(param1, retval, "param1");
            dbl = param2 ? PythonQtConversion::PyObjGetDouble(param2, false, ok) : 0.0;
            if (!ok)
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("param2 must be a double value.").toLatin1().data());
            }

            if (!retval.containsError())
            {
                self->shape = new ito::Shape(ito::Shape::fromSquare(pt1, dbl, index, name_));
            }
        break;

        case Shape::Polygon:
        {
            if (!param1)
            {
                PyErr_SetString(PyExc_RuntimeError, "param1: 2xM float64 array like object with polygon data required.");
                return -1;
            }

            PyObject *arr = PyArray_ContiguousFromAny(param1, NPY_DOUBLE, 2, 2); //new reference
            PyArrayObject* npArray = (PyArrayObject*)arr;
            if (arr)
            {
                const npy_intp *shape = PyArray_SHAPE(npArray);
                if (shape[0] == 2 && shape[1] >= 3)
                {
                    QPolygonF polygon;
                    polygon.reserve(shape[1]);
                    const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);
                    const npy_double *ptr2 = (const npy_double*)PyArray_GETPTR1(npArray, 1);
                    for (int i = 0; i < shape[1]; ++i)
                    {
                        polygon.push_back(QPointF(ptr1[i], ptr2[i]));
                    }
                    self->shape = new ito::Shape(ito::Shape::fromPolygon(polygon, index, name_));
                }
                else
                {
                    Py_XDECREF(arr);
                    PyErr_SetString(PyExc_RuntimeError, "param1: 2xM float64 array like object with polygon data required.");
                    return -1;
                }
            }
            else
            {
                return -1;
            }

            Py_XDECREF(arr);
        }
        break;

        case Shape::Ellipse:
        {
            pt1 = PyObject2PointF(param1, retval, "param1");
            pt2 = PyObject2PointF(param2, retval, "param2");
            if (!retval.containsError())
            {
                self->shape = new ito::Shape(ito::Shape::fromEllipse(pt1.rx(), pt1.ry(), pt2.rx(), pt2.ry(), index, name_));
            }
        }
        break;

        case Shape::Circle:
            pt1 = PyObject2PointF(param1, retval, "param1");
            dbl = param2 ? PythonQtConversion::PyObjGetDouble(param2, false, ok) : 0.0;
            if (!ok)
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("param2 must be a double value.").toLatin1().data());
            }

            if (!retval.containsError())
            {
                self->shape = new ito::Shape(ito::Shape::fromCircle(pt1, dbl, index, name_));
            }
        break;

        default:
            PyErr_SetString(PyExc_RuntimeError, "unknown type");
            return -1;
    }

    if (!PythonCommon::transformRetValToPyException(retval)) return -1;

    return 0;
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::createPyShape(const Shape &shape)
{
    PyShape* result = (PyShape*)PyObject_Call((PyObject*)&PyShapeType, NULL, NULL);
    if(result != NULL)
    {
        result->shape = new Shape(shape);
        return (PyObject*)result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticPoint_doc,  "createPoint(point, index = -1, name = '', flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Point``.\n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Point, point, index, name)\n\
    myShape.flags = flags  # optional\n\
\n\
Parameters \n\
----------\n\
point : sequence of float or dataObject or numpy.ndarray \n\
    (x,y) coordinate of the point, given as any type that can be interpreted as \n\
    array with two float64 values. \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticPoint(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *point = NULL;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"point", "index", "name", "flags", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|isi", const_cast<char**>(kwlist), &point, &index, &name, &flags))
    {
        return NULL;
    }

    QPointF pt = PyObject2PointF(point, retval, "point");

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;
    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromPoint(pt, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticLine_doc,  "createLine(point1, point2, index = -1, name = \"\", flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Line``.\n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Line, point1, point2, index, name)\n\
    myShape.flags = flags #optional\n\
\n\
Parameters \n\
----------\n\
point1 : sequence of float or dataObject or numpy.ndarray \n\
    (x,y) coordinate of the first point, given as any type that can be interpreted as \n\
    array with two values \n\
point2 : sequence of float or dataObject or numpy.ndarray \n\
    (x,y) coordinate of the 2nd point, given as any type that can be interpreted as \n\
    array with two values \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticLine(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *point1 = NULL;
    PyObject *point2 = NULL;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"point1", "point2", "index", "name", "flags", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|isi", const_cast<char**>(kwlist), &point1, &point2, &index, &name, &flags))
    {
        return NULL;
    }

    QPointF pt1 = PyObject2PointF(point1, retval, "point1");
    QPointF pt2 = PyObject2PointF(point2, retval, "point2");

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;
    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromLine(pt1, pt2, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticCircle_doc,  "createCircle(center, radius, index = -1, name = '', flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Circle``.\n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Circle, center, radius, index, name)\n\
    myShape.flags = flags #optional\n\
\n\
Parameters \n\
----------\n\
center : sequence of float or dataObject or numpy.ndarray \n\
    (x,y) coordinate of the center point, given as any type that can be interpreted \n\
    as array with two values \n\
radius : float \n\
    radius of the circle \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticCircle(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *center = NULL;
    double radius = 0.0;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"center", "radius", "index", "name", "flags", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|isi", const_cast<char**>(kwlist), &center, &radius, &index, &name, &flags))
    {
        return NULL;
    }

    QPointF c = PyObject2PointF(center, retval, "center");

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;
    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromCircle(c, radius, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticEllipse_doc,  "createEllipse(corner1 = None, corner2 = None, center = None, size = None, index = -1, name = \"\", flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Ellipse``.\n\
\n\
Basically, there are two different ways to construct the ellipse: \n\
Either by the top left and bottom right corner points of the outer bounding box (``corner1`` \n\
and ``corner2``), or by the ``center`` point (x,y) and the ``size``, as array of (width, height). \n\
\n\
Furthermore, you can indicate a ``size`` together with ``corner1`` OR ``corner2``, \n\
where corner1.x + width = corner2.x and corner1.y + height = corner2.y. \n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Ellipse, corner1, corner2, index, name)\n\
    myShape.flags = flags #optional\n\
\n\
Parameters \n\
----------\n\
corner1 : sequence of float or dataObject or numpy.ndarray, optional \n\
    (x,y) coordinate of the top, left corner point of the bounding box, given as \n\
    any type that can be interpreted as array with two values \n\
corner2 : sequence of float or dataObject or numpy.ndarray, optional \n\
    (x,y) coordinate of the bottom, right corner point of the bounding box, given as \n\
    any type that can be interpreted as array with two values \n\
center : sequence of float or dataObject or numpy.ndarray, optional \n\
    (x,y) coordinate of the center point, given as any type that can be interpreted \n\
    as array with two values \n\
size : sequence of float or dataObject or numpy.ndarray, optional \n\
    (width, height) of the rectangle, given as any type that can be interpreted as array \n\
    with two values \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticEllipse(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *corner1 = NULL;
    PyObject *corner2 = NULL;
    PyObject *center = NULL;
    PyObject *size = NULL;
    double sideLength = 0.0;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"corner1", "corner2", "center", "size", "index", "name", "flags", NULL};


    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOisi", const_cast<char**>(kwlist), &corner1, &corner2, &center, &size, &index, &name, &flags))
    {
        return NULL;
    }

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;
    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    QPointF pt1, pt2;
    QRectF rect;

    if (corner1 && corner2)
    {
        if (center || size)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'corner1' and 'corner2' are given, the parameters 'center' or 'size' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(corner1, retval, "corner1");
        pt2 = PyObject2PointF(corner2, retval, "corner2");
        rect = QRectF(pt1, pt2);
    }
    else if (center && size)
    {
        if (corner1 || corner2)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'center' and 'size' are given, the parameters 'corner1' or 'corner2' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(center, retval, "center");
        pt2 = PyObject2PointF(size, retval, "size");
        QSizeF size(pt2.x(), pt2.y());
        rect = QRectF(pt1 - pt2/2, size);
    }
    else if (corner1 && size)
    {
        if (corner2 || center)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'corner1' and 'size' are given, the parameters 'corner2' or 'center' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(corner1, retval, "corner1");
        pt2 = PyObject2PointF(size, retval, "size");
        QSizeF size(pt2.x(), pt2.y());
        rect = QRectF(pt1, size);
    }
    else if (corner2 && size)
    {
        if (corner1 || center)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'corner2' and 'size' are given, the parameters 'corner1' or 'center' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(corner2, retval, "corner2");
        pt2 = PyObject2PointF(size, retval, "size");
        QSizeF size(pt2.x(), pt2.y());
        rect = QRectF(pt1 - pt2, size);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid combination of parameters 'corner1', 'corner2', 'center' and 'size'.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromEllipse(rect, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticSquare_doc,  "createSquare(center, sideLength, index = -1, name = \"\", flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Square``.\n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Square, center, sideLength, index, name)\n\
    myShape.flags = flags #optional\n\
\n\
Parameters \n\
----------\n\
center : sequence of float or dataObject or numpy.ndarray \n\
    (x,y) coordinate of the center point, given as any type that can be interpreted \n\
    as array with two values \n\
sideLength : float \n\
    side length of the square \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticSquare(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *center = NULL;
    double sideLength = 0.0;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"center", "sideLength", "index", "name", "flags", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Od|isi", const_cast<char**>(kwlist), &center, &sideLength, &index, &name, &flags))
    {
        return NULL;
    }

    QPointF c = PyObject2PointF(center, retval, "center");

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;
    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromSquare(c, sideLength, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticRectangle_doc,  "createRectangle(corner1 = None, corner2 = None, center = None, size = None, index = -1, name = \"\", flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Rectangle``.\n\
\n\
Basically, there are two different ways to construct a rectangle: \n\
Either by the top left and bottom right corner points (``corner1`` and ``corner2``), \n\
or by the ``center`` point (x, y) and the ``size``, as array of (width, height). \n\
\n\
Furthermore, you can indicate a ``size`` together with ``corner1`` OR ``corner2``, \n\
where corner1.x + width = corner2.x and corner1.y + height = corner2.y. \n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Rectangle, corner1, corner2, index, name)\n\
    myShape.flags = flags #optional\n\
\n\
Parameters \n\
----------\n\
corner1 : sequence of float or dataObject or numpy.ndarray, optional \n\
    (x,y) coordinate of the top, left corner point, given as any type that can be \n\
    interpreted as array with two values \n\
corner2 : sequence of float or dataObject or numpy.ndarray, optional \n\
    (x,y) coordinate of the bottom, right corner point, given as any type that can be \n\
    interpreted as array with two values \n\
center : sequence of float or dataObject or numpy.ndarray, optional \n\
    (x,y) coordinate of the center point, given as any type that can be interpreted \n\
    as array with two values \n\
size : sequence of float or dataObject or numpy.ndarray, optional \n\
    (width, height) of the rectangle, given as any type that can be interpreted as array \n\
    with two values \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticRectangle(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *corner1 = NULL;
    PyObject *corner2 = NULL;
    PyObject *center = NULL;
    PyObject *size = NULL;
    double sideLength = 0.0;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"corner1", "corner2", "center", "size", "index", "name", "flags", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOisi", const_cast<char**>(kwlist), &corner1, &corner2, &center, &size, &index, &name, &flags))
    {
        return NULL;
    }

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;
    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    QPointF pt1, pt2;
    QRectF rect;

    if (corner1 && corner2)
    {
        if (center || size)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'corner1' and 'corner2' are given, the parameters 'center' or 'size' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(corner1, retval, "corner1");
        pt2 = PyObject2PointF(corner2, retval, "corner2");
        rect = QRectF(pt1, pt2);
    }
    else if (center && size)
    {
        if (corner1 || corner2)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'center' and 'size' are given, the parameters 'corner1' or 'corner2' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(center, retval, "center");
        pt2 = PyObject2PointF(size, retval, "size");
        QSizeF size(pt2.x(), pt2.y());
        rect = QRectF(pt1 - pt2/2, size);
    }
    else if (corner1 && size)
    {
        if (corner2 || center)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'corner1' and 'size' are given, the parameters 'corner2' or 'center' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(corner1, retval, "corner1");
        pt2 = PyObject2PointF(size, retval, "size");
        QSizeF size(pt2.x(), pt2.y());
        rect = QRectF(pt1, size);
    }
    else if (corner2 && size)
    {
        if (corner1 || center)
        {
            PyErr_SetString(PyExc_TypeError, "If the parameters 'corner2' and 'size' are given, the parameters 'corner1' or 'center' are not allowed.");
            return NULL;
        }

        pt1 = PyObject2PointF(corner2, retval, "corner2");
        pt2 = PyObject2PointF(size, retval, "size");
        QSizeF size(pt2.x(), pt2.y());
        rect = QRectF(pt1 - pt2, size);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Invalid combination of parameters 'corner1', 'corner2', 'center' and 'size'.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromRectangle(rect, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_staticPolygon_doc,  "createPolygon(points, index = -1, name = \"\", flags = 0) -> shape \n\
\n\
Returns a new shape object of type ``shape.Polygon``.\n\
\n\
This static method is equal to the command:: \n\
\n\
    myShape = shape(shape.Polygon, points, index, name)\n\
    myShape.flags = flags #optional\n\
\n\
Parameters \n\
----------\n\
points : sequence of sequence of float or dataObject or numpy.ndarray \n\
    An array-like object of shape ``2 x M`` (with M > 2), that can be converted \n\
    to float64. This object defines ``M`` points for the polygon (order: x, y). \n\
    If a sequence is given, it must look like this:: \n\
    \n\
        points = ((1, 2, 3), (4, 5, 6)) \n\
    \n\
    where the first inner tuple defines the x-coordinates, and the 2nd tuple \n\
    the y-coordinates. \n\
index : int, optional \n\
    index of this shape or -1 (default) if not further specified. \n\
name : str, optional \n\
    optional name of this shape (default: \"\"). This name can for instance be \n\
    displayed in a plot. \n\
flags : int, optional \n\
    if the user should not be able to rotate, resize and / or move this shape in any \n\
    plot canvas, then pass an or-combination of the restricitive flag values \n\
    ``shape.ResizeLock``, ``shape.RotateLock`` or ``shape.MoveLock``. \n\
\n\
Returns \n\
------- \n\
shape \n\
    The new shape object.");
/*static*/ PyObject* PythonShape::PyShape_StaticPolygon(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *points = NULL;
    int index = -1;
    char* name = NULL;
    int flags = 0;
    ito::RetVal retval;
    const char *kwlist[] = {"points", "index", "name", "flags", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|isi", const_cast<char**>(kwlist), &points, &index, &name, &flags))
    {
        return NULL;
    }

    QPolygonF polygon;
    PyObject *arr = PyArray_ContiguousFromAny(points, NPY_DOUBLE, 2, 2); //new reference
    PyArrayObject* npArray = (PyArrayObject*)arr;

    if (arr)
    {
        const npy_intp *shape = PyArray_SHAPE(npArray);
        if (shape[0] == 2 && shape[1] >= 3)
        {
            polygon.reserve(shape[1]);
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);
            const npy_double *ptr2 = (const npy_double*)PyArray_GETPTR1(npArray, 1);
            for (int i = 0; i < shape[1]; ++i)
            {
                polygon.push_back(QPointF(ptr1[i], ptr2[i]));
            }
        }
        else
        {
            Py_XDECREF(arr);
            PyErr_SetString(PyExc_RuntimeError, "points: 3xM float64 array like object with polygon data required.");
            return NULL;
        }
    }
    else
    {
        return NULL;
    }

    Py_XDECREF(arr);

    quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;

    if ((flags | allowedFlags) != allowedFlags)
    {
        PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return NULL;
    }
    else
    {
        PyShape* shape = (PyShape*)createPyShape(ito::Shape::fromPolygon(polygon, index, name));
        shape->shape->setFlags(flags & ito::Shape::FlagMask);

        return (PyObject*)shape;
    }
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::PyShape_repr(PyShape *self)
{
    PyObject *result;
    if(self->shape == NULL)
    {
        result = PyUnicode_FromFormat("shape(NULL)");
    }
    else
    {
        const QPolygonF &base = self->shape->rbasePoints();

        if (self->shape->transform().isIdentity())
        {
            switch (self->shape->type())
            {
            case Shape::Point:
                result = PyUnicode_FromFormat(
                    QString("shape(Point, (%1, %2), index: %3)").arg(base[0].x()).arg(base[0].y()).arg(self->shape->index()).toLatin1().data());
                break;
            case Shape::Line:
                result = PyUnicode_FromFormat(
                    QString("shape(Line, (%1, %2) - (%3, %4), index: %5)").arg(base[0].x()).arg(base[0].y()).arg(base[1].x()).arg(base[1].y()).arg(self->shape->index()).toLatin1().data());
                break;
            case Shape::Rectangle:
                result = PyUnicode_FromFormat(
                    QString("shape(Rectangle, (%1, %2) - (%3, %4), index: %5)").arg(base[0].x()).arg(base[0].y()).arg(base[1].x()).arg(base[1].y()).arg(self->shape->index()).toLatin1().data());
                break;
            case Shape::Square:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat(
                    QString("shape(Square, center (%1, %2), l: %3, index: %4)").arg(p.rx() / 2).arg(p.ry() / 2).arg(s.rx() / 2).arg(self->shape->index()).toLatin1().data());
                break;
            }
            case Shape::Polygon:
                result = PyUnicode_FromFormat("shape(Polygon, %i points, index: %i)", base.size(), self->shape->index());
                break;
            case Shape::Ellipse:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat(
                    QString("shape(Ellipse, center (%1, %2), (a=%3, b=%4), index: %5)").arg(p.rx() / 2).arg(p.ry() / 2).arg(s.rx() / 2).arg(s.ry() / 2).arg(self->shape->index()).toLatin1().data());
                break;
            }
            case Shape::Circle:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat(
                    QString("shape(Circle, center (%1, %2), r: %3, index: %4)").arg(p.rx() / 2).arg(p.ry() / 2).arg(s.rx() / 2).arg(self->shape->index()).toLatin1().data());
                break;
            }
            case Shape::Invalid:
                result = PyUnicode_FromFormat("shape(Invalid)");
                break;
            default:
                result = PyUnicode_FromFormat("shape(Unknown)");
                break;
            }
        }
        else
        {
            switch (self->shape->type())
            {
            case Shape::Point: {
                QPointF p = self->shape->centerPoint();
                result = PyUnicode_FromFormat(
                    "shape(Point, (%f, %f), index: %i)", p.x(), p.y(), self->shape->index());
            }
            break;
            case Shape::Line: {
                const QPolygonF& contour = self->shape->contour();
                result = PyUnicode_FromFormat(
                    "shape(Line, (%f, %f) - (%f, %f), index: %i)",
                    contour[0].x(),
                    contour[0].y(),
                    contour[1].x(),
                    contour[1].y(),
                    self->shape->index());
            }
            break;
            case Shape::Rectangle: {
                const QPolygonF& contour = self->shape->contour();
                result =
                    PyUnicode_FromFormat(QString("shape(Rectangle, (%1, %2) - (%3, %4), index: %5)")
                                             .arg(contour[0].x())
                                             .arg(contour[0].y())
                                             .arg(contour[2].x())
                                             .arg(contour[2].y())
                                             .arg(self->shape->index())
                                             .toLatin1()
                                             .data());
            }
                break;
            case Shape::Square:
            {
                QPointF p = self->shape->centerPoint();
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat(
                    QString("shape(Square, center (%1, %2), l: %3, index: %4)").arg(p.rx()).arg(p.ry()).arg(s.rx() / 2).arg(self->shape->index()).toLatin1().data());
            }
                break;
            case Shape::Polygon:
                result = PyUnicode_FromFormat("shape(Polygon, %i points, index: %i)", base.size(), self->shape->index());
                break;
            case Shape::Ellipse:
            {
                QPointF p = self->shape->centerPoint();
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat(
                    QString("shape(Ellipse, center (%1, %2), (a=%3, b=%4), index: %5)").arg(p.rx()).arg(p.ry()).arg(s.rx() / 2).arg(s.ry() / 2).arg(self->shape->index()).toLatin1().data());
            }
                break;
            case Shape::Circle:
            {
                QPointF p = self->shape->centerPoint();
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat(
                    QString("shape(Circle, center (%1, %2), l: %3, index: %4)").arg(p.rx()).arg(p.ry()).arg(s.rx() / 2).arg(self->shape->index()).toLatin1().data());
            }
                break;
            case Shape::Invalid:
                result = PyUnicode_FromFormat("shape(Invalid)");
                break;
            default:
                result = PyUnicode_FromFormat("shape(Unknown)");
                break;
            }
        }
    }
    return result;
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::PyShape_reduce(PyShape *self, PyObject *args)
{
    PyObject *stateTuple = NULL;

    if (self->shape)
    {
        QByteArray ba;
        QDataStream d(&ba, QIODevice::WriteOnly | QIODevice::Truncate);
        d << *(self->shape);

        stateTuple = PyBytes_FromStringAndSize(ba.data(), ba.size());
    }
    else
    {
        Py_INCREF(Py_None);
        stateTuple = Py_None;
    }

    //the stateTuple is simply a byte array with the stream data of the QRegion.
    PyObject *tempOut = Py_BuildValue("(O(i)O)", Py_TYPE(self), Shape::Invalid, stateTuple);
    Py_XDECREF(stateTuple);

    return tempOut;
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::PyShape_setState(PyShape *self, PyObject *args)
{
    PyObject *data = NULL;
    if (!PyArg_ParseTuple(args, "O", &data))
    {
        return NULL;
    }

    if (data == Py_None)
    {
        Py_RETURN_NONE;
    }
    else
    {
        QByteArray ba(PyBytes_AS_STRING(data), PyBytes_GET_SIZE(data));
        QDataStream d(&ba, QIODevice::ReadOnly);

		if (!self->shape)
		{
			self->shape = new ito::Shape();
		}

        d >> *(self->shape);
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getType_doc,
"int : Get the type of this shape. \n\
\n\
Possible types are: \n\
\n\
* ``shape.Line`` \n\
* ``shape.Point`` \n\
* ``shape.Rectangle`` \n\
* ``shape.Ellipse`` \n\
* ``shape.Circle`` \n\
* ``shape.Square``");
PyObject* PythonShape::PyShape_getType(PyShape *self, void * /*closure*/)
{
    if(!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyLong_FromLong(self->shape->type());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getValid_doc,
"bool : Returns True if this shape is valid, otherwise False. \n\
\n\
An invalid shape is the one constructed with the type ``shape.Invalid``. All other \n\
shapes are valid.");
PyObject* PythonShape::PyShape_getValid(PyShape *self, void * /*closure*/)
{
    if(!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    if (self->shape->type() == ito::Shape::Invalid)
        Py_RETURN_FALSE;
    else
        Py_RETURN_TRUE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getFlags_doc,
"int : Gets or sets a flag (bitmask) that define denied manipulation of this shape. \n\
\n\
It is possible to deny the following manipulations: \n\
\n\
* ``shape.MoveLock`` \n\
* `` shape.RotateLock`` \n\
* `` shape.ResizeLock``");
PyObject* PythonShape::PyShape_getFlags(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyLong_FromLong(self->shape->flags());
}

int PythonShape::PyShape_setFlags(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;
    quint64 flags = PythonQtConversion::PyObjGetULongLong(value, true, ok);

    if (ok)
    {
        quint64 allowedFlags = Shape::MoveLock | Shape::RotateLock | Shape::ResizeLock;

        if ((flags | allowedFlags) != allowedFlags)
        {
            PyErr_SetString(PyExc_TypeError, "at least one flag value is not supported.");
            return -1;
        }

        self->shape->setFlags(flags);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the flags as uint.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_point1_doc,
"tuple of float : Gets or sets the 1st point of the bounding box of this shape. \n\
\n\
The first point is the first point of a ``shape.Point`` or ``shape.Line`` or \n\
the upper left point of the bounding box of a ``shape.Rectangle``, ``shape.Square`` \n\
``shape.Ellipse`` or ``shape.Circle``. The point always considers a possible 2D \n\
coordinate transformation matrix.\n\
\n\
For setting this value, it is also possible to pass any other array-like object \n\
with two elements, that can be converted to float64. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this attribute is read or assigned for a type, that has no 2nd point defined.");
PyObject* PythonShape::PyShape_getPoint1(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    switch (self->shape->type())
    {
        case Shape::Point:
        case Shape::Line:
        case Shape::Rectangle:
        case Shape::Square:
        case Shape::Ellipse:
        case Shape::Circle:
        {
            QPointF point(self->shape->transform().map(self->shape->basePoints()[0]));
            return PointF2PyObject(point);
        }
        break;

        case Shape::Polygon:
        case Shape::Invalid:
        default:
        break;
    }

    return PyErr_Format(PyExc_TypeError, "This type of shape has no 'point1' defined.");
}

int PythonShape::PyShape_setPoint1(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    ito::RetVal retval;
    QPointF point = PyObject2PointF(value, retval, "point1");

    if (!retval.containsError())
    {
        switch (self->shape->type())
        {
            case Shape::Point:
            case Shape::Line:
            case Shape::Rectangle:
            case Shape::Ellipse:
            {
                QTransform inv = self->shape->transform().inverted();
                self->shape->rbasePoints()[0] = inv.map(point);
            }
            break;

            case Shape::Square:
            case Shape::Circle:
                retval += ito::RetVal(ito::retError, 0, QObject::tr("point1 cannot be changed for square and circle. Change center and width / height.").toLatin1().data());
            break;

            case Shape::Polygon:
            case Shape::Invalid:
            default:
                retval += ito::RetVal(ito::retError, 0, QObject::tr("This type of shape has no 'point1' defined.").toLatin1().data());
            break;
        }
    }

    if (!PythonCommon::transformRetValToPyException(retval)) return -1;
    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_point2_doc,
"tuple of float : Gets or sets the second point of the bounding box of this shape. \n\
\n\
The second point is the 2nd point of a ``shape.Line`` or the bottom right point of \n\
the bounding box (types: ``shape.Rectangle``, ``shape.Square``, ``shape.Ellipse`` or \n\
``shape.Circle``). The point always considers a possible 2D coordinate \n\
transformation matrix.\n\
\n\
For setting this value, it is also possible to pass any other array-like object \n\
with two elements, that can be converted to float64. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this attribute is read or assigned for a type, that has no 2nd point defined.");
PyObject* PythonShape::PyShape_getPoint2(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    switch (self->shape->type())
    {
        case Shape::Line:
        case Shape::Rectangle:
        case Shape::Square:
        case Shape::Ellipse:
        case Shape::Circle:
        {
            QPointF point(self->shape->transform().map(self->shape->basePoints()[1]));
            return PointF2PyObject(point);
        }
        break;

        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
        break;
    }

    return PyErr_Format(PyExc_TypeError, "This type of shape has no 'point2' defined.");
}

int PythonShape::PyShape_setPoint2(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    ito::RetVal retval;
    QPointF point = PyObject2PointF(value, retval, "point2");

    if (!retval.containsError())
    {
        switch (self->shape->type())
        {
            case Shape::Line:
            case Shape::Rectangle:
            case Shape::Ellipse:
            {
                QTransform inv = self->shape->transform().inverted();
                self->shape->rbasePoints()[1] = inv.map(point);
            }
            break;

            case Shape::Square:
            case Shape::Circle:
                retval += ito::RetVal(ito::retError, 0, QObject::tr("point2 cannot be changed for square and circle. Change center and width / height.").toLatin1().data());
            break;

            case Shape::Point:
            case Shape::Polygon:
            case Shape::Invalid:
            default:
                retval += ito::RetVal(ito::retError, 0, QObject::tr("This type of shape has no 'point2' defined.").toLatin1().data());
            break;
        }
    }

    if (!PythonCommon::transformRetValToPyException(retval)) return -1;
    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_center_doc,
"tuple of float : Gets or sets the center point of this shape. \n\
\n\
The center point is defined for all types of shapes, beside ``shape.Polygon``. \n\
Changing the center point will directly influence the base points of the shape. \n\
\n\
If the value is set, it is also possible to pass any other array-like object with \n\
two values that can be converted to float64. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this attribute is read or assigned for a type, that has no center defined.");
PyObject* PythonShape::PyShape_getCenter(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    switch (self->shape->type())
    {
        case Shape::Point:
        {
            QPointF point(self->shape->transform().map(self->shape->basePoints()[0]));
            return PointF2PyObject(point);
        }
        break;

        case Shape::Line:
        case Shape::Rectangle:
        case Shape::Square:
        case Shape::Ellipse:
        case Shape::Circle:
        {
            QPointF point((self->shape->transform().map(self->shape->basePoints()[0]) + self->shape->transform().map(self->shape->basePoints()[1])) / 2.0);
            return PointF2PyObject(point);
        }
        break;

        case Shape::Polygon:
        case Shape::Invalid:
        default:
        break;
    }

    return PyErr_Format(PyExc_TypeError, "This type of shape has no 'center' defined.");
}

int PythonShape::PyShape_setCenter(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    ito::RetVal retval;
    QPointF point = PyObject2PointF(value, retval, "center");

    if (!retval.containsError())
    {
        switch (self->shape->type())
        {
            case Shape::Point:
            {
                QTransform inv = self->shape->transform().inverted();
                self->shape->rbasePoints()[0] = inv.map(point);
            }
            break;

            case Shape::Line:
            case Shape::Rectangle:
            case Shape::Ellipse:
            case Shape::Square:
            case Shape::Circle:
            {
                QTransform inv = self->shape->transform().inverted();
                QPointF baseCenter = inv.map(point);
                QPointF oldCenter = (self->shape->basePoints()[0] + self->shape->basePoints()[1]) / 2.0;
                self->shape->rbasePoints()[0] += (baseCenter - oldCenter);
                self->shape->rbasePoints()[1] += (baseCenter - oldCenter);
            }
            break;

            case Shape::Polygon:
            case Shape::Invalid:
            default:
                retval += ito::RetVal(ito::retError, 0, QObject::tr("This type of shape has no 'center' defined.").toLatin1().data());
            break;
        }
    }

    if (!PythonCommon::transformRetValToPyException(retval)) return -1;
    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_angleDeg_doc,
"float : Gets or sets the current angle of rotation of the transformation matrix in degree. \n\
\n\
A rotation is always defined counter-clockwise.");
PyObject* PythonShape::PyShape_getAngleDeg(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyFloat_FromDouble(self->shape->rotationAngleDeg());
}

int PythonShape::PyShape_setAngleDeg(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;
    double angle = PythonQtConversion::PyObjGetDouble(value, false, ok);
    if (ok)
    {
        self->shape->setRotationAngleDeg(angle);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the angle as double.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_angleRad_doc,
"float : Gets or sets the current angle of rotation of the transformation matrix in Radians. \n\
\n\
A rotation is always defined counter-clockwise.");
PyObject* PythonShape::PyShape_getAngleRad(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyFloat_FromDouble(self->shape->rotationAngleRad());
}

int PythonShape::PyShape_setAngleRad(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;
    double angle = PythonQtConversion::PyObjGetDouble(value, false, ok);
    if (ok)
    {
        self->shape->setRotationAngleRad(angle);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the angle as double.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_radius_doc,
"float or tuple of float : Gets or sets the radius of this shape. \n\
\n\
A radius can only be set for shapes of type ``shape.Circle`` or ``shape.Ellipse``. \n\
For a circle, the radius is a scalar float value. For an ellipse, a tuple of \n\
two values ``(a, b)`` define the half side-length in x- and y-direction of \n\
the base coordinate system.\n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this attribute is read or assigned for a type, that has no radius defined.");
PyObject* PythonShape::PyShape_getRadius(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    switch (self->shape->type())
    {
        case Shape::Ellipse:
        {
            QPointF point(self->shape->basePoints()[1] - self->shape->basePoints()[0]);
            PyObject *tuple = PyTuple_New(2);
            PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(point.x() / 2.0));
            PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(point.y() / 2.0));
            return tuple;
        }
        break;

        case Shape::Circle:
        {
            QPointF point(self->shape->basePoints()[1] - self->shape->basePoints()[0]);
            return PyFloat_FromDouble(point.x() / 2.0);
        }
        break;

        case Shape::Line:
        case Shape::Rectangle:
        case Shape::Square:
        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
        break;
    }

    return PyErr_Format(PyExc_TypeError, "This type of shape has no 'radius' defined.");
}

int PythonShape::PyShape_setRadius(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;

    switch (self->shape->type())
    {
        case Shape::Ellipse:
        {
            ito::RetVal retval;
            QPointF ab = PyObject2PointF(value, retval, "radius (a,b)");
            if (!PythonCommon::transformRetValToPyException(retval)) return -1;

            QPointF dr(ab.x(), ab.y());
            QPolygonF &basePoints = self->shape->rbasePoints();
            basePoints[0] -= dr;
            basePoints[1] += dr;
            return 0;
        }
        break;

        case Shape::Circle:
        {
            double radius = PythonQtConversion::PyObjGetDouble(value, false, ok);
            if (ok)
            {
                QPointF dr(radius, radius);
                QPolygonF &basePoints = self->shape->rbasePoints();
                basePoints[0] -= dr;
                basePoints[1] += dr;
                return 0;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "error interpreting the radius as double.");
                return -1;
            }
        }
        break;

        case Shape::Line:
        case Shape::Rectangle:
        case Shape::Square:
        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
            PyErr_SetString(PyExc_TypeError, "type of shape has no 'radius' defined.");
            return -1;
        break;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_width_doc,  "float : Gets or sets the width of this shape. \n\
\n\
A width can only be set or read for shapes of type ``shape.Square`` and \n\
``shape.Rectangle``. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this attribute is read or assigned for a type, that has no width defined.");
PyObject* PythonShape::PyShape_getWidth(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    switch (self->shape->type())
    {
        case Shape::Rectangle:
        case Shape::Square:
        {
            QPointF point(self->shape->basePoints()[1] - self->shape->basePoints()[0]);
            return PyFloat_FromDouble(point.x());
        }
        break;

        case Shape::Line:
        case Shape::Ellipse:
        case Shape::Circle:
        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
        break;
    }

    return PyErr_Format(PyExc_TypeError, "This type of shape has no 'width' defined.");
}

int PythonShape::PyShape_setWidth(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;

    switch (self->shape->type())
    {
        case Shape::Square:
        {
            double width = PythonQtConversion::PyObjGetDouble(value, false, ok);
            if (ok)
            {
                QPolygonF &basePoints = self->shape->rbasePoints();
                QPointF center = 0.5 * (basePoints[0] + basePoints[1]);
                QPointF delta(width/2, 0.0);
                basePoints[0] = center - delta;
                basePoints[1] = center + delta;
                return 0;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "error interpreting the width as double.");
                return -1;
            }
        }
        break;

        case Shape::Rectangle:
        {
            double width = PythonQtConversion::PyObjGetDouble(value, false, ok);
            if (ok)
            {
                QPolygonF &basePoints = self->shape->rbasePoints();
                basePoints[1] = basePoints[0] + QPointF(width, 0.0);
                return 0;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "error interpreting the width as double.");
                return -1;
            }
        }
        break;

        case Shape::Ellipse:
        case Shape::Circle:
        case Shape::Line:
        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
            PyErr_SetString(PyExc_TypeError, "type of shape has no 'width' defined.");
            return -1;
         break;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_height_doc,  "float : Gets or sets the height of this shape. \n\
\n\
A height can only be set or read for shapes of type ``shape.Square`` and \n\
``shape.Rectangle``. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this attribute is read or assigned for a type, that has no height defined.");
PyObject* PythonShape::PyShape_getHeight(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    switch (self->shape->type())
    {
        case Shape::Rectangle:
        case Shape::Square:
        {
            QPointF point(self->shape->basePoints()[1] - self->shape->basePoints()[0]);
            return PyFloat_FromDouble(point.y());
        }
        break;

        case Shape::Line:
        case Shape::Ellipse:
        case Shape::Circle:
        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
        break;
    }

    return PyErr_Format(PyExc_TypeError, "This type of shape has no 'height' defined.");
}

int PythonShape::PyShape_setHeight(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;

    switch (self->shape->type())
    {
        case Shape::Square:
        {
            double height = PythonQtConversion::PyObjGetDouble(value, false, ok);
            if (ok)
            {
                QPolygonF &basePoints = self->shape->rbasePoints();
                QPointF center = 0.5 * (basePoints[0] + basePoints[1]);
                QPointF delta(0.0, height/2.0);
                basePoints[0] = center - delta;
                basePoints[1] = center + delta;
                return 0;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "error interpreting the height as double.");
                return -1;
            }
        }
        break;

        case Shape::Rectangle:
        {
            double height = PythonQtConversion::PyObjGetDouble(value, false, ok);
            if (ok)
            {
                QPolygonF &basePoints = self->shape->rbasePoints();
                basePoints[1] = basePoints[0] + QPointF(0.0, height);
                return 0;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "error interpreting the height as double.");
                return -1;
            }
        }
        break;

        case Shape::Ellipse:
        case Shape::Circle:
        case Shape::Line:
        case Shape::Point:
        case Shape::Polygon:
        case Shape::Invalid:
        default:
            PyErr_SetString(PyExc_TypeError, "type of shape has no 'height' defined.");
            return -1;
        break;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getIndex_doc,  "int : Gets or sets the index of this shape. \n\
\n\
The default is -1, however if the shape is a geometric shape of a plot, an \n\
auto-incremented index is assigned once the shape is drawn or set. \n\
If >= 0 it is possible to modify an existing shape with the same index.");
PyObject* PythonShape::PyShape_getIndex(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyLong_FromLong(self->shape->index());
}

int PythonShape::PyShape_setIndex(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;
    int index = PythonQtConversion::PyObjGetInt(value, true, ok);
    if (ok)
    {
        self->shape->setIndex(index);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the index as int.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getName_doc,  "str : Gets or sets the name (label) of this shape.");
PyObject* PythonShape::PyShape_getName(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PythonQtConversion::QStringToPyObject(self->shape->name());
}

int PythonShape::PyShape_setName(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok;
    QString name = PythonQtConversion::PyObjGetString(value, false, ok);
    if (ok)
    {
        self->shape->setName(name);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the name as str.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getColor_doc,
"None or rgba : Gets or sets color of this shape. \n\
\n\
The default color is an invalid color, given by the ``None`` value. \n\
The color of shapes is for instance be used for visualization purposes in plots.");
PyObject* PythonShape::PyShape_getColor(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

	QColor c = self->shape->color();

	if (c.isValid())
	{
		PythonRgba::PyRgba* retRgba = PythonRgba::createEmptyPyRgba();
		retRgba->rgba.r = c.red();
		retRgba->rgba.g = c.green();
		retRgba->rgba.b = c.blue();
		retRgba->rgba.a = c.alpha();
		return (PyObject*)retRgba;
	}
	else
	{
		Py_RETURN_NONE;
	}
}

int PythonShape::PyShape_setColor(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

	QColor color;
	bool ok;

	if (value == Py_None)
	{
		ok = true;
	}
	else if (PyRgba_Check(value))
	{
		ito::PythonRgba::PyRgba *rgba = (ito::PythonRgba::PyRgba*)value;
		color = QColor(rgba->rgba.r, rgba->rgba.g, rgba->rgba.b, rgba->rgba.a);
		ok = true;
	}

    if (ok)
    {
        self->shape->setColor(color);
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the color as itom.rgba or None (invalid color).");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getTransform_doc,
"dataObject : gets or sets the affine, non scaled 2D transformation matrix as dataObject. \n\
\n\
The returned matrix is a ``2 x 3``, float64 :class:`dataObject`, where the left \n\
``2 x 2`` matrix describes a rotation matrix, and the right ``2 x 1`` part is the \n\
translation vector.");
PyObject* PythonShape::PyShape_getTransform(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    const QTransform &t = self->shape->transform();
    ito::DataObject trafo(2, 3, ito::tFloat64);
    ito::float64 *ptr = trafo.rowPtr<ito::float64>(0, 0);
    ptr[0] = t.m11();
    ptr[1] = t.m21(); //m21() is the 2nd value in the first row. QTransform has a different indice notation than usual!
    ptr[2] = t.dx();
    ptr = trafo.rowPtr<ito::float64>(0, 1);
    ptr[0] = t.m12(); //m12() is the 1st value in the second row. QTransform has a different indice notation than usual!
    ptr[1] = t.m22();
    ptr[2] = t.dy();

    ito::PythonDataObject::PyDataObject *obj = PythonDataObject::createEmptyPyDataObject();
    if (obj)
    {
        obj->dataObject = new DataObject(trafo);
    }

    return (PyObject*)obj;
}

int PythonShape::PyShape_setTransform(PyShape *self, PyObject *value, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return -1;
    }

    bool ok = true; //true since PyArray_ContiguousFromAny may throw its own error.
    PyObject *arr = PyArray_ContiguousFromAny(value, NPY_DOUBLE, 2, 2);
    PyArrayObject* npArray = (PyArrayObject*)arr;
    if (arr)
    {
        ok = false;
        const npy_intp *shape = PyArray_SHAPE(npArray);
        if (shape[0] == 2 && shape[1] == 3)
        {
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);
            const npy_double *ptr2 = (const npy_double*)PyArray_GETPTR1(npArray, 1);
            ok = true;
            QTransform trafo(ptr1[0], ptr2[0], ptr1[1], ptr2[1], ptr1[2], ptr2[2]);
            if (trafo.isAffine() && !trafo.isScaling())
            {
                self->shape->setTransform(trafo);
            }
            else
            {
                Py_XDECREF(arr);
                PyErr_SetString(PyExc_RuntimeError, "2x3 transformation must be affine and not scaled [2x2 Rot,2x1 trans]");
                return -1;
            }
        }
    }

    Py_XDECREF(arr);

    if (!ok)
    {
        PyErr_SetString(PyExc_RuntimeError, "affine, non-scaled 2x3, float64 array for transformation required");
        return -1;
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_getArea_doc, "float : Get area of this shape \n\
\n\
Shapes of type ``shape.Line`` and ``shape.Point`` will always return 0.0.");
PyObject* PythonShape::PyShape_getArea(PyShape *self, void * /*closure*/)
{
    if(!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyFloat_FromDouble(self->shape->area());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_rotateDeg_doc, "rotateDeg(angle) \n\
\n\
Rotate shape by given angle in degrees around the center point of this shape \n\
(counterclockwise). This method only affects the :attr:`transform` matrix, not the \n\
base points themselfs. \n\
\n\
Parameters \n\
---------- \n\
angle : float \n\
    is the rotation angle (in radians) by which the shape is rotated by its center. \n\
\n\
See Also \n\
-------- \n\
translate, rotateRad");
PyObject* PythonShape::PyShape_rotateDeg(PyShape *self, PyObject *args)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    qreal rot = 0.0;
    if (!PyArg_ParseTuple(args, "d", &rot))
    {
        return NULL;
    }

    self->shape->rotateByCenterDeg(rot);

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_rotateRad_doc, "rotateRad(angle) \n\
\n\
Rotate shape by given angle in radians around the center point of this shape \n\
(counterclockwise). This method only affects the :attr:`transform` matrix, not the \n\
base points themselfs. \n\
\n\
Parameters \n\
---------- \n\
angle : float \n\
    is the rotation angle (in radians) by which the shape is rotated by its center. \n\
\n\
See Also \n\
-------- \n\
translate, rotateDeg");
PyObject* PythonShape::PyShape_rotateRad(PyShape *self, PyObject *args)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    qreal rot = 0.0;
    if (!PyArg_ParseTuple(args, "d", &rot))
    {
        return NULL;
    }

    self->shape->rotateByCenterRad(rot);

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_translate_doc, "translate(dxy) \n\
\n\
Translate shape by given (dx, dy) value. \n\
\n\
Moves the shape by dx and dy along the x- and y-axis of the base coordinate system. \n\
This means, that dx and dy are added to the existing tx and ty values of the current \n\
transformation matrix. \n\
\n\
Parameters \n\
---------- \n\
dxy : sequence of float or dataObject or numpy.ndarray \n\
    array-like object with two elements, that define the desired ``dx`` and ``dy`` \n\
    component. \n\
\n\
See Also \n\
-------- \n\
rotateRad, rotateDeg");
PyObject* PythonShape::PyShape_translate(PyShape *self, PyObject *args)
{
    PyObject *obj = NULL;

    if (!PyArg_ParseTuple(args, "O", &obj))
    {
        return NULL;
    }

    bool ok = true; //true since PyArray_ContiguousFromAny may throw its own error.
    double dx = 0.0;
    double dy = 0.0;
    PyObject *arr = PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 1, 2);
    PyArrayObject* npArray = (PyArrayObject*)arr;

    if (arr)
    {
        ok = false;

        if (PyArray_NDIM(npArray) == 2)
        {
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);

            if (PyArray_DIM(npArray, 0) == 2 && PyArray_DIM(npArray, 1) == 1) //2d, two rows, one col
            {
                dx = ptr1[0];
                dy = ((npy_double*)PyArray_GETPTR1(npArray, 1))[0];
                ok = true;
            }
            else if (PyArray_DIM(npArray, 0) == 1 && PyArray_DIM(npArray, 1) == 2) //2d, one row, two cols
            {
                dx = ptr1[0];
                dy = ptr1[1];
                ok = true;
            }
        }
        else
        {
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);

            if (PyArray_DIM(npArray, 0) == 2) //1d
            {
                dx = ptr1[0];
                dy = ptr1[1];
                ok = true;
            }
        }

        if (ok)
        {
            self->shape->translate(QPointF(dx,dy));
        }
    }

    Py_XDECREF(arr);

    if (!ok)
    {
        PyErr_SetString(PyExc_RuntimeError, "float64 array with two elements required (dx,dy)");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_basePoints_doc,
"dataObject : base points of this shape, given as ``2 x M``, float64 dataObject. \n\
\n\
The ``M`` base points are untransformed points that describe the shape \n\
dependent on its type: \n\
\n\
* ``shape.Point``: one point \n\
* ``shape.Line`` : start point, end point \n\
* ``shape.Rectangle``, ``shape.Square`` : top left point, bottom right point \n\
* ``shape.Ellipse``, ``shape.Circle`` : top left point, bottom right point of bounding box \n\
* ``shape.Polygon`` : points of polygon, the last and first point are connected, too.");
PyObject* PythonShape::PyShape_getBasePoints(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    QPolygonF basePoints = self->shape->basePoints();
    ito::DataObject bp;
    if (basePoints.size() > 0)
    {
        bp = ito::DataObject(2, basePoints.size(), ito::tFloat64);
        ito::float64 *ptr_x = bp.rowPtr<ito::float64>(0, 0);
        ito::float64 *ptr_y = bp.rowPtr<ito::float64>(0, 1);
        for (int i = 0; i < basePoints.size(); ++i)
        {
            ptr_x[i] = basePoints[i].rx();
            ptr_y[i] = basePoints[i].ry();
        }
    }

    ito::PythonDataObject::PyDataObject *obj = PythonDataObject::createEmptyPyDataObject();
    if (obj)
        obj->dataObject = new DataObject(bp);

    return (PyObject*)obj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_region_doc, "region() -> region \n\
\n\
Returns a region object from this shape. \n\
\n\
The :class:`region` object only contains valid regions if the shape has an area > 0. \n\
A region object is an integer based object (pixel raster), therefore the shapes \n\
are rounded to the nearest fixed-point coordinate. \n\
\n\
Returns \n\
------- \n\
region \n\
    The region, whose contour approximates this shape. The inner of \n\
    this shaped is part of the region.");
PyObject* PythonShape::PyShape_region(PyShape *self)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return ito::PythonRegion::createPyRegion(self->shape->region());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_contour_doc, "contour(applyTrafo = True, tol = -1.0) -> dataObject \n\
\n\
Returns the contour points of this shape as ``2 x N``, float64 :class:`dataObject`. \n\
\n\
For most shapes, the contour is exactly given by its corner points. However for \n\
circles or ellipses, the contour has to be approximated by line segments. Use the \n\
argument ``tol`` to set the maximum distance between each line segment and the \n\
real contour of the shape. If ``tol`` is set to -1.0, ``tol`` is assumed to be `1 %` \n\
of the smalles diameter. \n\
\n\
Shapes can have a transformation matrix (attribute :attr:`transform`). If ``applyTrafo`` \n\
is ``True``, the returned contour points correspond to the transformed base shape, else \n\
the contour with respect to the base points is returned. \n\
\n\
Parameters \n\
---------- \n\
applyTrafo : bool \n\
    Define if the transformation matrix (default: unity matrix, attribute :attr:`transform`) \n\
    should be considered for the returned contour points (``True``) or not (``False``). \n\
tol : float \n\
    Maximum tolerance to determine the approximated contour in the case of circular \n\
    or elliptical shapes. The approximated contour consists of line segments, that \n\
    can differ from the real contour by a maximum of ``tol``. If -1.0, the tolerance \n\
    is assumed to be one percent of the smallest diameter.");
PyObject* PythonShape::PyShape_contour(PyShape *self, PyObject *args, PyObject *kwds)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    double tol = -1.0;
    unsigned char applyTrafo = true;

    const char *kwlist[] = { "applyTrafo", "tol", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|bd", const_cast<char**>(kwlist), &applyTrafo, &tol))
    {
        return NULL;
    }

    QPolygonF contour = self->shape->contour(applyTrafo, tol);
    ito::DataObject cont;

    if (contour.size() > 0)
    {
        cont = ito::DataObject(2, contour.size(), ito::tFloat64);
        ito::float64 *ptr_x = cont.rowPtr<ito::float64>(0, 0);
        ito::float64 *ptr_y = cont.rowPtr<ito::float64>(0, 1);

        for (int i = 0; i < contour.size(); ++i)
        {
            ptr_x[i] = contour[i].rx();
            ptr_y[i] = contour[i].ry();
        }
    }

    ito::PythonDataObject::PyDataObject *obj = PythonDataObject::createEmptyPyDataObject();

    if (obj)
    {
        obj->dataObject = new DataObject(cont);
    }

    return (PyObject*)obj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_contains_doc, "contains(points) -> Union[bool, dataObject] \n\
\n\
Checks if one or multiple ``points`` are contained in this shape. \n\
\n\
Tests if one or multiple ``points`` lie within the contour of the given shape. If the \n\
shape has an empty area (e.g. points, line...) the test will always return ``False``.\n\
\n\
Parameters  \n\
----------\n\
points : sequence of float or dataObject or numpy.ndarray \n\
    The coordinates ``(x, y)`` of the point to be tested as sequence or an array-like \n\
    object (shape ``2 x N``), where the first row contains the x-coordinates and the \n\
    2nd-row the y-coordinates of ``N`` points to be tested. The array-like object \n\
    must be convertible to ``float64``, which is internally done before testing. \n\
\n\
Returns \n\
------- \n\
result : bool or dataObject \n\
    If one point is passed as sequence, ``True`` is returned if this point is within \n\
    the contour of this shape, otherwise ``False``. If ``points`` is given as array-like \n\
    object, a ``1 x N`` :class:`dataObject` with dtype ``uint8`` is returned, where \n\
    the value ``255`` indicates, that the corresponding point is inside of the \n\
    shape's contour and ``0`` outside.");
/*static*/ PyObject* PythonShape::PyShape_contains(PyShape *self, PyObject *args, PyObject *kwds)
{
    if (!self || self->shape == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    PyObject *points = nullptr;
    const char *kwlist[] = { "points", nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &points))
    {
        return nullptr;
    }

    bool ok;
    ito::RetVal pointsRetVal;
    QVector<double> values = PythonQtConversion::PyObjGetDoubleArray(points, false, ok);

    if (ok)
    {
        if (values.size() == 2)
        {
            QPointF point(values[0], values[1]);
            if (self->shape->contains(point))
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
            PyErr_SetString(PyExc_RuntimeError, "given point coordinate must be a sequence of two floats (x,y) or an 2xN dataObject / array.");
            return NULL;
        }
    }

    auto dataobj = PythonQtConversion::PyObjGetSharedDataObject(points, false, ok, &pointsRetVal);

    if (ok)
    {
        ito::DataObject dataobj_ = ito::dObjHelper::squeezeConvertCheck2DDataObject(
            dataobj.data(),
            "points",
            ito::Range(2, 2),
            ito::Range::all(),
            pointsRetVal,
            ito::tFloat64,
            8, ito::tUInt8, ito::tInt8, ito::tUInt16, ito::tInt16, ito::tUInt32, ito::tInt32, ito::tFloat32, ito::tFloat64);

        if (PythonCommon::transformRetValToPyException(pointsRetVal))
        {
            QPolygonF points_;
            points_.resize(dataobj_.getSize(1));
            const ito::float64 *row1 = dataobj_.rowPtr<const ito::float64>(0, 0);
            const ito::float64 *row2 = dataobj_.rowPtr<const ito::float64>(0, 1);

            for (int i = 0; i < points_.size(); ++i)
            {
                points_[i].rx() = row1[i];
                points_[i].ry() = row2[i];
            }

            QVector<bool> result = self->shape->contains(points_);
            ito::DataObject result_(1, points_.size(), ito::tUInt8);
            ito::uint8 *row3 = result_.rowPtr<ito::uint8>(0, 0);

            for (int i = 0; i < points_.size(); ++i)
            {
                row3[i] = (result[i] ? 255 : 0);
            }

            ito::PythonDataObject::PyDataObject *obj = PythonDataObject::createEmptyPyDataObject();

            if (obj)
            {
                obj->dataObject = new DataObject(result_);
            }

            return (PyObject*)obj;
        }
        else
        {
            return nullptr;
        }
    }

    PythonCommon::transformRetValToPyException(pointsRetVal);
    return nullptr;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_normalized_doc, "normalized() -> shape \n\
\n\
Returns the normalized version of this shape. \n\
\n\
The normalized shape guarantees that the bounding box of the shape never has a \n\
non-negative width or height. Therefore, the order or position of the two corner points \n\
(base points) is switched or changed, if necessary. Shapes different than \n\
rectangles, squares, circles or ellipses are not affected by this such that the \n\
original shape object is returned as it is. \n\
\n\
Returns \n\
------- \n\
normalized : shape \n\
    The normalized shape of this object (for types ``shape.Rectange``, ``shape.Square`` \n\
    ``shape.Circle`` or ``shape.Ellipse``) or this object (for all other types).");
/*static*/ PyObject* PythonShape::PyShape_normalized(PyShape *self)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return createPyShape(self->shape->normalized());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(shape_copy_doc, "copy() -> shape\n\
\n\
Returns a deep copy of this shape. \n\
\n\
Returns \n\
------- \n\
copy : shape \n\
    deep copy of this shape.");
/*static*/ PyObject* PythonShape::PyShape_copy(PyShape *self)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return createPyShape(ito::Shape(*self->shape));
}

//-------------------------------------------------------------------------------------
PyObject* PythonShape::PointF2PyObject(const QPointF &point)
{
    PyObject *tuple = PyTuple_New(2);
    PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(point.x()));
    PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(point.y()));
    return tuple;
}

//-------------------------------------------------------------------------------------
QPointF PythonShape::PyObject2PointF(PyObject *value, ito::RetVal &retval, const char* paramName)
{
    if (!value)
    {
        retval += ito::RetVal::format(ito::retError, 0, QObject::tr("%s missing").toLatin1().constData(), paramName);
        return QPointF();
    }

    bool ok = true; //true since PyArray_ContiguousFromAny may throw its own error.
    PyObject *arr = PyArray_ContiguousFromAny(value, NPY_DOUBLE, 1, 2);
    PyArrayObject* npArray = (PyArrayObject*)arr;
    QPointF point;

    if (arr)
    {
        ok = false;
        if (PyArray_NDIM(npArray) == 2)
        {
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);
            if (PyArray_DIM(npArray, 0) == 2 && PyArray_DIM(npArray, 1) == 1) //2d, two rows, one col
            {
                point.rx() = ptr1[0];
                point.ry() = ((npy_double*)PyArray_GETPTR1(npArray, 1))[0];
                ok = true;
            }
            else if (PyArray_DIM(npArray, 0) == 1 && PyArray_DIM(npArray, 1) == 2) //2d, one row, two cols
            {
                point.rx() = ptr1[0];
                point.ry() = ptr1[1];
                ok = true;
            }
        }
        else
        {
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);
            if (PyArray_DIM(npArray, 0) == 2) //1d
            {
                point.rx() = ptr1[0];
                point.ry() = ptr1[1];
                ok = true;
            }
        }
    }
	else
	{
		//if PyArray_ContiguousFromAny could not convert the input to an array, it has raised an exception,
		//that is transformed here to a retval. If this is not the case, ok = false --> retval is set to error at the end of this function
		retval += PythonCommon::checkForPyExceptions(true);
		if (!retval.containsError())
		{
			ok = false;
		}
	}

    Py_XDECREF(arr);

    if (!ok)
    {
        retval += ito::RetVal::format(ito::retError, 0, QObject::tr("%s: float64 array with two elements required (x,y)").toLatin1().constData(), paramName);
    }

    return point;
}

//-------------------------------------------------------------------------------------
PyGetSetDef PythonShape::PyShape_getseters[] = {
    {"valid",     (getter)PyShape_getValid,       (setter)NULL,                   shape_getValid_doc, NULL },
    {"type",      (getter)PyShape_getType,        (setter)NULL,                   shape_getType_doc, NULL },
    {"flags",     (getter)PyShape_getFlags,       (setter)PyShape_setFlags,       shape_getFlags_doc, NULL},
    {"index",     (getter)PyShape_getIndex,       (setter)PyShape_setIndex,       shape_getIndex_doc, NULL},
    {"name",      (getter)PyShape_getName,        (setter)PyShape_setName,        shape_getName_doc, NULL},
    {"transform", (getter)PyShape_getTransform,   (setter)PyShape_setTransform,   shape_getTransform_doc, NULL}, //only affine transformation, 2d, allowed
    {"area",      (getter)PyShape_getArea,        (setter)NULL,                   shape_getArea_doc, NULL},
    {"basePoints", (getter)PyShape_getBasePoints, (setter)NULL,                   shape_basePoints_doc, NULL},
    {"point1",    (getter)PyShape_getPoint1,      (setter)PyShape_setPoint1,      shape_point1_doc, NULL},
    {"point2",    (getter)PyShape_getPoint2,      (setter)PyShape_setPoint2,      shape_point2_doc, NULL},
    {"center",    (getter)PyShape_getCenter,      (setter)PyShape_setCenter,      shape_center_doc, NULL},
    {"angleDeg",  (getter)PyShape_getAngleDeg,    (setter)PyShape_setAngleDeg,    shape_angleDeg_doc, NULL},
    {"angleRad",  (getter)PyShape_getAngleRad,    (setter)PyShape_setAngleRad,    shape_angleRad_doc, NULL},
    {"radius",    (getter)PyShape_getRadius,      (setter)PyShape_setRadius,      shape_radius_doc, NULL},
    {"width",     (getter)PyShape_getWidth,       (setter)PyShape_setWidth,       shape_width_doc, NULL},
    {"height",    (getter)PyShape_getHeight,      (setter)PyShape_setHeight,      shape_height_doc, NULL},
	{"color",     (getter)PyShape_getColor,       (setter)PyShape_setColor,       shape_getColor_doc, NULL},
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMethodDef PythonShape::PyShape_methods[] = {
    { "__reduce__", (PyCFunction)PyShape_reduce, METH_VARARGS,      "__reduce__ method for handle pickling commands"},
    { "__setstate__", (PyCFunction)PyShape_setState, METH_VARARGS,  "__setstate__ method for handle unpickling commands"},
    { "rotateDeg", (PyCFunction)PyShape_rotateDeg, METH_VARARGS, shape_rotateDeg_doc },
    { "rotateRad", (PyCFunction)PyShape_rotateRad, METH_VARARGS, shape_rotateRad_doc },
    { "translate", (PyCFunction)PyShape_translate, METH_VARARGS, shape_translate_doc },
    { "region", (PyCFunction)PyShape_region, METH_NOARGS, shape_region_doc },
    { "contour", (PyCFunction)PyShape_contour, METH_VARARGS | METH_KEYWORDS, shape_contour_doc },
    { "contains", (PyCFunction)PyShape_contains, METH_VARARGS | METH_KEYWORDS, shape_contains_doc },
    { "normalized", (PyCFunction)PyShape_normalized, METH_NOARGS, shape_normalized_doc },
    { "copy", (PyCFunction)PyShape_copy, METH_NOARGS, shape_copy_doc },

    { "createPoint", (PyCFunction)PyShape_StaticPoint, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticPoint_doc },
    { "createLine", (PyCFunction)PyShape_StaticLine, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticLine_doc },
    { "createCircle", (PyCFunction)PyShape_StaticCircle, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticCircle_doc },
    { "createEllipse", (PyCFunction)PyShape_StaticEllipse, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticEllipse_doc },
    { "createSquare", (PyCFunction)PyShape_StaticSquare, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticSquare_doc },
    { "createRectangle", (PyCFunction)PyShape_StaticRectangle, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticRectangle_doc },
    { "createPolygon", (PyCFunction)PyShape_StaticPolygon, METH_VARARGS | METH_KEYWORDS | METH_STATIC, shape_staticPolygon_doc },
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonShape::PyShapeModule = {
    PyModuleDef_HEAD_INIT, "shape", "Shape (Point, Line, Rectangle, Ellipse, Polygon...)", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonShape::PyShapeType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.shape",             /* tp_name */
    sizeof(PyShape),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyShape_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyShape_repr,                         /* tp_repr */
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
    PyShape_doc,              /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    PyShape_methods,          /* tp_methods */
    0,                         /* tp_members */
    PyShape_getseters,        /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyShape_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PyShape_new               /* tp_new */
};


} //end namespace ito
