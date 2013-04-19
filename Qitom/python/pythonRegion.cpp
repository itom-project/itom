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

#include "pythonRegion.h"

#include <qvector.h>
#include <qrect.h>

//------------------------------------------------------------------------------------------------------

namespace ito
{

void PythonRegion::PyRegion_addTpDict(PyObject * tp_dict)
{
    PyObject *value;
    
    value = Py_BuildValue("i",QRegion::Rectangle);
    PyDict_SetItemString(tp_dict, "RECTANGLE", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",QRegion::Ellipse);
    PyDict_SetItemString(tp_dict, "ELLIPSE", value);
    Py_DECREF(value);
}

void PythonRegion::PyRegion_dealloc(PyRegion* self)
{
    if(self->r)
    {
        delete self->r;
        self->r = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
};

PyObject* PythonRegion::PyRegion_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyRegion* self = (PyRegion *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->r = NULL;
    }

    return (PyObject *)self;
};


int PythonRegion::PyRegion_init(PyRegion *self, PyObject *args, PyObject * /*kwds*/)
{
    int x,y,w,h;
    int t = QRegion::Rectangle;
    PyObject *other = NULL;

    if(PyTuple_Size(args) == 0)
    {
        if(self->r) delete self->r;
        self->r = new QRegion();
    }
    else if(PyArg_ParseTuple(args,"O!",&PyRegionType,other))
    {
        if(self->r) delete self->r;
        PyRegion *otherRegion = (PyRegion*)other;
        if(otherRegion->r == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Region of other region object is NULL");
            return -1;
        }
        self->r = new QRegion(*(otherRegion->r));
    }
    else if(PyErr_Clear(), PyArg_ParseTuple(args,"iiii|i", &x, &y, &w, &h, &t))
    {
        if(t == QRegion::Rectangle)
        {
            if(self->r) delete self->r;
            self->r = new QRegion(x,y,w,h, QRegion::Rectangle);
        }
        else if(t == QRegion::Ellipse)
        {
            if(self->r) delete self->r;
            self->r = new QRegion(x,y,w,h, QRegion::Ellipse);
        }
    }
    else
    {
        return -1;
    }

    return 0;
};

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::createPyRegion(const QRegion &region)
{
    PyRegion* result = (PyRegion*)PyObject_Call((PyObject*)&PyRegionType, NULL, NULL);
    if(result != NULL)
    {
        result->r = new QRegion(region);
        return (PyObject*)result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_repr(PyRegion *self)
{
    PyObject *result;
    if(self->r == NULL)
    {
        result = PyUnicode_FromFormat("Region(NULL)");
    }
    else
    {
        result = PyUnicode_FromFormat("Figure(rects: %i)", self->r->rectCount() );
    }
    return result;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_contains(PyRegion *self, PyObject *args, PyObject *kwds)
{
    if(!self || self->r == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "region is not available");
    }

    int x,y;
    int w = -1;
    int h = -1;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};

    if( !PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
    {
        return NULL;
    }

    bool result;

    if(w == -1 && h == -1)
    {
        result = self->r->contains( QPoint(x,y) );
    }
    else
    {
        result = self->r->contains( QRect(x,y,w,h) );
    }

    if(result) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_intersected(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
    {
        QRegion reg = self->r->intersected( QRect(x,y,w,h) );
        return createPyRegion(reg);
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist2), &PyRegionType, &other) )
    {
        QRegion reg = self->r->intersected( *(((PyRegion*)other)->r) );
        return createPyRegion(reg);
    }
    else
    {
        return NULL;
    }
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_intersects(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    bool result;

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
    {
        result = self->r->intersects( QRect(x,y,w,h) );
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist2), &PyRegionType, &other) )
    {
        result = self->r->intersects( *(((PyRegion*)other)->r) );
    }
    else
    {
        return NULL;
    }

    if(result) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_subtracted(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
    {
        QRegion reg = self->r->subtracted( QRegion(QRect(x,y,w,h)) );
        return createPyRegion(reg);
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist2), &PyRegionType, &other) )
    {
        QRegion reg = self->r->subtracted( *(((PyRegion*)other)->r) );
        return createPyRegion(reg);
    }
    else
    {
        return NULL;
    }
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_translate(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y;

    const char *kwlist[] = {"x", "y", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist), &x, &y) )
    {
        self->r->translate( QPoint(x,y) );
    }
    else
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_translated(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y;

    const char *kwlist[] = {"x", "y", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii", const_cast<char**>(kwlist), &x, &y) )
    {
        QRegion reg = self->r->translated( QPoint(x,y) );
        return createPyRegion(reg);
    }
    else
    {
        return NULL;
    }
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_united(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
    {
        QRegion reg = self->r->united( QRect(x,y,w,h) );
        return createPyRegion(reg);
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist2), &PyRegionType, &other) )
    {
        QRegion reg = self->r->united( *(((PyRegion*)other)->r) );
        return createPyRegion(reg);
    }
    else
    {
        return NULL;
    }
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_xored(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
    {
        QRegion reg = self->r->xored( QRegion(QRect(x,y,w,h)) );
        return createPyRegion(reg);
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist2), &PyRegionType, &other) )
    {
        QRegion reg = self->r->xored( *(((PyRegion*)other)->r) );
        return createPyRegion(reg);
    }
    else
    {
        return NULL;
    }
}


//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_getEmpty(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "region is not available");
    }

    if(self->r->isEmpty())
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyFigure_getRectCount(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "region is not available");
    }

    return Py_BuildValue( "i", self->r->rectCount() );
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyFigure_getRects(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "region is not available");
    }

    QVector<QRect> rects = self->r->rects();

    PyObject *ret = PyList_New( rects.size() );
    Py_ssize_t i = 0;
    PyObject *t;

    foreach(const QRect &r, rects)
    {
        t = PyList_New(4);
        PyList_SetItem(t,0, PyLong_FromLong(r.x()));
        PyList_SetItem(t,1, PyLong_FromLong(r.y()));
        PyList_SetItem(t,2, PyLong_FromLong(r.width()));
        PyList_SetItem(t,3, PyLong_FromLong(r.height()));
        PyList_SetItem(ret,i++,t); //steals reference
    }

    return ret;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyFigure_getBoundingRect(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        return PyErr_Format(PyExc_RuntimeError, "region is not available");
    }

    QRect b = self->r->boundingRect();

    PyObject *t = PyList_New(4);
    PyList_SetItem(t,0, PyLong_FromLong(b.x()));
    PyList_SetItem(t,1, PyLong_FromLong(b.y()));
    PyList_SetItem(t,2, PyLong_FromLong(b.width()));
    PyList_SetItem(t,3, PyLong_FromLong(b.height()));

    return t;
}



//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbAdd(PyRegion* o1, PyRegion* o2)
{
    return createPyRegion( *(o1->r) + *(o2->r) );
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbSubtract(PyRegion* o1, PyRegion* o2)
{
    return createPyRegion( *(o1->r) - *(o2->r) );
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbAnd(PyRegion* o1, PyRegion* o2)
{
    return createPyRegion( *(o1->r) & *(o2->r) );
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbXor(PyRegion* o1, PyRegion* o2)
{
    return createPyRegion( *(o1->r) ^ *(o2->r) );
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbOr(PyRegion* o1, PyRegion* o2)
{
    return createPyRegion( *(o1->r) | *(o2->r) );
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbInplaceAdd(PyRegion* o1, PyRegion* o2)
{
    *(o1->r) += *(o2->r);
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbInplaceSubtract(PyRegion* o1, PyRegion* o2)
{
    *(o1->r) -= *(o2->r);
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbInplaceAnd(PyRegion* o1, PyRegion* o2)
{
    *(o1->r) &= *(o2->r);
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbInplaceXor(PyRegion* o1, PyRegion* o2)
{
    *(o1->r) ^= *(o2->r);
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_nbInplaceOr(PyRegion* o1, PyRegion* o2)
{
    *(o1->r) |= *(o2->r);
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-----------------------------------------------------------------------------
PyGetSetDef PythonRegion::PyRegion_getseters[] = {
    {"empty", (getter)PyRegion_getEmpty, NULL, "Returns True if region is empty, else False", NULL},
    {"rectCount", (getter)PyFigure_getRectCount, NULL, "Returns True if region is empty, else False", NULL},
    {"rects", (getter)PyFigure_getRects, NULL, "Returns list of rectangles", NULL},
    {NULL}  /* Sentinel */
};

//-----------------------------------------------------------------------------
PyMethodDef PythonRegion::PyRegion_methods[] = {
    {"contains", (PyCFunction)PyRegion_contains, METH_VARARGS | METH_KEYWORDS, NULL},
    {"intersected", (PyCFunction)PyRegion_intersected, METH_VARARGS | METH_KEYWORDS, NULL},
    {"intersects", (PyCFunction)PyRegion_intersects, METH_VARARGS | METH_KEYWORDS, NULL},
    {"subtracted", (PyCFunction)PyRegion_subtracted, METH_VARARGS | METH_KEYWORDS, NULL},
    {"translate", (PyCFunction)PyRegion_translate, METH_VARARGS | METH_KEYWORDS, NULL},
    {"translated", (PyCFunction)PyRegion_translated, METH_VARARGS | METH_KEYWORDS, NULL},
    {"united", (PyCFunction)PyRegion_united, METH_VARARGS | METH_KEYWORDS, NULL},
    {"xored", (PyCFunction)PyRegion_xored, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL}  /* Sentinel */
};




//-----------------------------------------------------------------------------
PyModuleDef PythonRegion::PyRegionModule = {
    PyModuleDef_HEAD_INIT, "Region", "Region (wrapper for QRegion)", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-----------------------------------------------------------------------------
PyTypeObject PythonRegion::PyRegionType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.Region",             /* tp_name */
    sizeof(PyRegion),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyRegion_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyRegion_repr,                         /* tp_repr */
    &PyRegion_numberProtocol,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    0,                         /* tp_doc */
    0,		                   /* tp_traverse */
    0,		                   /* tp_clear */
    0,		                   /* tp_richcompare */
    0,		                   /* tp_weaklistoffset */
    0,		                   /* tp_iter */
    0,		                   /* tp_iternext */
    PyRegion_methods,          /* tp_methods */
    0,                         /* tp_members */
    PyRegion_getseters,        /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyRegion_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PyRegion_new               /* tp_new */
};

//-----------------------------------------------------------------------------
PyNumberMethods PythonRegion::PyRegion_numberProtocol = {
    (binaryfunc)PyRegion_nbAdd,                 /* nb_add */
    (binaryfunc)PyRegion_nbSubtract,            /* nb_subtract */
    0,                    /* nb_multiply */
    0,              /* nb_remainder */
    0,                 /* nb_divmod */
    0,                   /* nb_power */
    0,                     /* nb_negative */
    0,                     /* nb_positive */
    0,                     /* nb_absolute */
    (inquiry)0,                      /* nb_bool */
    0,                                          /* nb_invert */
    0,                                          /* nb_lshift */
    0,                                          /* nb_rshift */
    (binaryfunc)PyRegion_nbAnd,                 /* nb_and */
    (binaryfunc)PyRegion_nbXor,                 /* nb_xor */
    (binaryfunc)PyRegion_nbOr,                  /* nb_or */
    0,                                /* nb_int */
    0,                                          /* nb_reserved */
    0,                              /* nb_float */
    (binaryfunc)PyRegion_nbInplaceAdd,          /* nb_inplace_add */
    (binaryfunc)PyRegion_nbInplaceSubtract,     /* nb_inplace_subtract */
    0,                                          /* nb_inplace_multiply*/
    0,                                          /* nb_inplace_remainder */
    0,                                          /* nb_inplace_power */
    0,                                          /* nb_inplace_lshift */
    0,                                          /* nb_inplace_rshift */
    (binaryfunc)PyRegion_nbInplaceAnd,          /* nb_inplace_and */
    (binaryfunc)PyRegion_nbInplaceXor,          /* nb_inplace_xor */
    (binaryfunc)PyRegion_nbInplaceOr,           /* nb_inplace_or */
    (binaryfunc)0,                /* nb_floor_divide */
    (binaryfunc)0,                    /* nb_true_divide */
    0,                                          /* nb_inplace_floor_divide */
    0,                                          /* nb_inplace_true_divide */
};



} //end namespace ito