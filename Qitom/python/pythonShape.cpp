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
#include "pythonDataObject.h"
#include "../common/shape.h"
#include "../DataObject/dataobj.h"



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
                result = PyUnicode_FromFormat("shape(Point, (%f, %f))", base[0].rx(), base[0].ry());
                break;
            case Shape::Line:
                result = PyUnicode_FromFormat("shape(Line, (%f, %f) - (%f, %f))", base[0].rx(), base[0].ry(), base[1].rx(), base[1].ry());
                break;
            case Shape::Rectangle:
                result = PyUnicode_FromFormat("shape(Rectangle, (%f, %f) - (%f, %f))", base[0].rx(), base[0].ry(), base[1].rx(), base[1].ry());
                break;
            case Shape::Square:
                result = PyUnicode_FromFormat("shape(Square, (%f, %f) - (%f, %f))", base[0].rx(), base[0].ry(), base[1].rx(), base[1].ry());
                break;
            case Shape::Polygon:
                result = PyUnicode_FromFormat("shape(Polygon, %i points)", base.size());
                break;
            case Shape::Ellipse:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat("shape(Ellipse, center (%f, %f), (a=%f, b=%f))", p.rx() / 2, p.ry() / 2, s.rx(), s.ry());
                break;
            }
            case Shape::Circle:
            {
                QPointF p = base[0] + base[1];
                QPointF s = base[1] - base[0];
                result = PyUnicode_FromFormat("shape(Circle, center (%f, %f), r=%f)", p.rx() / 2, p.ry() / 2, s.rx());
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
            QPolygonF &contour = self->shape->contour();

            switch (self->shape->type())
            {
            case Shape::Point:
                result = PyUnicode_FromFormat("shape(Point, (%f, %f))", contour[0].rx(), contour[0].ry());
                break;
            case Shape::Line:
                result = PyUnicode_FromFormat("shape(Line, (%f, %f) - (%f, %f))", contour[0].rx(), contour[0].ry(), contour[1].rx(), contour[1].ry());
                break;
            case Shape::Rectangle:
                result = PyUnicode_FromFormat("shape(Rectangle)");
                break;
            case Shape::Square:
                result = PyUnicode_FromFormat("shape(Square)");
                break;
            case Shape::Polygon:
                result = PyUnicode_FromFormat("shape(Polygon, %i points)", base.size());
                break;
            case Shape::Ellipse:
                result = PyUnicode_FromFormat("shape(Ellipse)");
                break;
            case Shape::Circle:
                result = PyUnicode_FromFormat("shape(Circle)");
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

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_getTransform_doc,  "Get/set the affine, non scaled 2D transformation matrix (2x3, float64, [2x2 Rot, 2x1 trans])");
PyObject* PythonShape::PyShape_getTransform(PyShape *self, void * /*closure*/)
{
    if (!self || self->shape == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "shape is not available");
        return NULL;
    }

    QTransform &t = self->shape->transform();
    ito::DataObject trafo(2, 3, ito::tFloat64);
    ito::float64 *ptr = trafo.rowPtr<ito::float64>(0, 0);
    ptr[0] = t.m11();
    ptr[1] = t.m12();
    ptr[2] = t.dx();
    ptr = trafo.rowPtr<ito::float64>(0, 1);
    ptr[0] = t.m21();
    ptr[1] = t.m22();
    ptr[2] = t.dy();

    ito::PythonDataObject::PyDataObject *obj = PythonDataObject::createEmptyPyDataObject();
    if (obj)
        obj->dataObject = new DataObject(trafo);

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
            QTransform trafo(ptr1[0], ptr1[1], ptr2[0], ptr2[1], ptr1[2], ptr2[2]);
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
PyDoc_STRVAR(shape_rotateDeg_doc, "Rotate shape by given angle in degree (counterclockwise).");
PyObject* PythonShape::PyShape_RotateDeg(PyShape *self, PyObject *args)
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

    self->shape->rtransform().rotate(rot, Qt::ZAxis);
    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_rotateRad_doc, "Rotate shape by given angle in radians (counterclockwise).");
PyObject* PythonShape::PyShape_RotateRad(PyShape *self, PyObject *args)
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

    self->shape->rtransform().rotateRadians(rot, Qt::ZAxis);
    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(shape_translate_doc, "Translate shape by given (dx,dy) value.");
PyObject* PythonShape::PyShape_Translate(PyShape *self, PyObject *args)
{
    PyObject *obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &obj))
    {
        return NULL;
    }

    bool ok = true; //true since PyArray_ContiguousFromAny may throw its own error.
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
                self->shape->rtransform().translate(ptr1[0], ((npy_double*)PyArray_GETPTR1(npArray, 1))[0]);
                ok = true;
            }
            else if (PyArray_DIM(npArray, 0) == 1 && PyArray_DIM(npArray, 1) == 2) //2d, one row, two cols
            {
                self->shape->rtransform().translate(ptr1[0], ptr1[1]);
                ok = true;
            }
        }
        else
        {
            const npy_double *ptr1 = (npy_double*)PyArray_GETPTR1(npArray, 0);
            if (PyArray_DIM(npArray, 0) == 2) //1d
            {
                self->shape->rtransform().translate(ptr1[0], ptr1[1]);
                ok = true;
            }
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
    { "__reduce__", (PyCFunction)PyShape_Reduce, METH_VARARGS,      "__reduce__ method for handle pickling commands"},
    { "__setstate__", (PyCFunction)PyShape_SetState, METH_VARARGS,  "__setstate__ method for handle unpickling commands"},
    { "rotateDeg", (PyCFunction)PyShape_RotateDeg, METH_VARARGS, shape_rotateDeg_doc },
    { "rotateRad", (PyCFunction)PyShape_RotateRad, METH_VARARGS, shape_rotateRad_doc },
    { "translate", (PyCFunction)PyShape_Translate, METH_VARARGS, shape_translate_doc },
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