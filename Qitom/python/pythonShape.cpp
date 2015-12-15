/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include "../common/shape.h"



//------------------------------------------------------------------------------------------------------

namespace ito
{

//------------------------------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------------------------------
void PythonShape::PyShape_dealloc(PyShape* self)
{
    DELETE_AND_SET_NULL(self->shape);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonShape::PyShape_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyShape* self = (PyShape *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->shape = NULL;
    }

    return (PyObject *)self;
};

//------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyShape_doc,"shape() -> creates an empty shape object.");
int PythonShape::PyShape_init(PyShape *self, PyObject *args, PyObject * kwds)
{
    //int pointSize = 0;
    //int weight = -1;
    //bool italic = false;
    //const char* family = NULL;

    //const char *kwlist[] = {"family", "pointSize", "weight", "italic", NULL};

    //if (args == NULL && kwds == NULL)
    //{
    //    return 0; //call from createPyShape
    //}

    //if(!PyArg_ParseTupleAndKeywords(args, kwds, "s|iiB", const_cast<char**>(kwlist), &(family), &(pointSize), &(weight), &(italic)))
    //{
    //    return -1;
    //}

    self->shape = new Shape();

    return 0;
};

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::PyShape_repr(PyShape *self)
{
    PyObject *result;
    if(self->shape == NULL)
    {
        result = PyUnicode_FromFormat("shape(NULL)");
    }
    else
    {
        QPolygonF &base = self->shape->basePoints();

        if (self->shape->transform().isIdentity())
        {
            switch (self->shape->type())
            {
            case Shape::Point:
                result = PyUnicode_FromFormat("shape(Point, (%f, %f)", base[0].rx(), base[0].ry());
                break;
            case Shape::Line:
                result = PyUnicode_FromFormat("shape(Line, (%f, %f) - (%f, %f)", base[0].rx(), base[0].ry(), base[1].rx(), base[1].ry());
                break;
            case Shape::Rectangle:
                result = PyUnicode_FromFormat("shape(Rectangle, (%f, %f) - (%f, %f)", base[0].rx(), base[0].ry(), base[1].rx(), base[1].ry());
                break;
            case Shape::Square:
                result = PyUnicode_FromFormat("shape(Square, (%f, %f) - (%f, %f)", base[0].rx(), base[0].ry(), base[1].rx(), base[1].ry());
                break;
            case Shape::Polygon:
                result = PyUnicode_FromFormat("shape(Polygon, %i points", base.size());
                break;
            case Shape::Ellipse:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat("shape(Ellipse, center (%f, %f), (a=%f, b=%f)", p.rx() / 2, p.ry() / 2, s.rx(), s.ry());
                break;
            }
            case Shape::Circle:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat("shape(Circle, center (%f, %f), r=%f", p.rx() / 2, p.ry() / 2, s.rx());
                break;
            }
            default:
                result = PyUnicode_FromFormat("shape(Unknown)");
                break;
            }
        }
        else
        {
            QPolygonF &contour = self->shape->contour();

            switch (self->shape->type())
            {
            case Shape::Point:
                result = PyUnicode_FromFormat("shape(Point, (%f, %f)", contour[0].rx(), contour[0].ry());
                break;
            case Shape::Line:
                result = PyUnicode_FromFormat("shape(Line, (%f, %f) - (%f, %f)", contour[0].rx(), contour[0].ry(), contour[1].rx(), contour[1].ry());
                break;
            case Shape::Rectangle:
                result = PyUnicode_FromFormat("shape(Rectangle)");
                break;
            case Shape::Square:
                result = PyUnicode_FromFormat("shape(Square)");
                break;
            case Shape::Polygon:
                result = PyUnicode_FromFormat("shape(Polygon, %i points", base.size());
                break;
            case Shape::Ellipse:
                result = PyUnicode_FromFormat("shape(Ellipse)");
                break;
            case Shape::Circle:
                result = PyUnicode_FromFormat("shape(Circle)");
                break;
            default:
                result = PyUnicode_FromFormat("shape(Unknown)");
                break;
            }
        }
    }
    return result;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::PyShape_Reduce(PyShape *self, PyObject *args)
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
    PyObject *tempOut = Py_BuildValue("(O()O)", Py_TYPE(self), stateTuple);
    Py_XDECREF(stateTuple);

    return tempOut;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonShape::PyShape_SetState(PyShape *self, PyObject *args)
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

        if (self->shape)
        {
            d >> *(self->shape);
        }
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_getType_doc,  "...");
PyObject* PythonShape::PyShape_getType(PyShape *self, void * /*closure*/)
{
    if(!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyLong_FromLong(self->shape->type());
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_getFlags_doc,  "....");
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
        return NULL;
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
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the flags as uint.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_getTransform_doc,  "...");
PyObject* PythonShape::PyShape_getTransform(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return NULL; // PyLong_FromLong(self->font->pointSize());
}

int PythonShape::PyShape_setTransform(PyShape *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    quint64 pointSize = PythonQtConversion::PyObjGetULongLong(value, true, ok);
    if (ok)
    {
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the point size as uint.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_getArea_doc,  "Area of shape.");
PyObject* PythonShape::PyShape_getArea(PyShape *self, void * /*closure*/)
{
    if(!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    return PyFloat_FromDouble(self->shape->area());
}

//-----------------------------------------------------------------------------
PyGetSetDef PythonShape::PyShape_getseters[] = {
    { "type", (getter)PyShape_getType,          (setter)NULL,                   shape_getType_doc, NULL },
    {"flags", (getter)PyShape_getFlags,         (setter)PyShape_setFlags,       shape_getFlags_doc, NULL},
    {"transform", (getter)PyShape_getTransform, (setter)PyShape_setTransform,   shape_getTransform_doc, NULL}, //only affine transformation, 2d, allowed
    {"area", (getter)PyShape_getArea,           (setter)NULL,                   shape_getArea_doc, NULL},
    {NULL}  /* Sentinel */
};

//-----------------------------------------------------------------------------
PyMethodDef PythonShape::PyShape_methods[] = {
    {"__reduce__", (PyCFunction)PyShape_Reduce, METH_VARARGS,      "__reduce__ method for handle pickling commands"},
    {"__setstate__", (PyCFunction)PyShape_SetState, METH_VARARGS,  "__setstate__ method for handle unpickling commands"},
    {NULL}  /* Sentinel */
};




//-----------------------------------------------------------------------------
PyModuleDef PythonShape::PyShapeModule = {
    PyModuleDef_HEAD_INIT, "shape", "Shape (Point, Line, Rectangle, Ellipse, Polygon...)", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-----------------------------------------------------------------------------
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