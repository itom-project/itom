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

#include "pythonRegion.h"

#include "pythonDataObject.h"
#include "../global.h"

#include <qvector.h>
#include <qrect.h>
#include <qiodevice.h>


//-----------------------------------------------------------------------------

namespace ito
{

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
void PythonRegion::PyRegion_dealloc(PyRegion* self)
{
    DELETE_AND_SET_NULL(self->r);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//-----------------------------------------------------------------------------
PyObject* PythonRegion::PyRegion_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyRegion* self = (PyRegion *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->r = NULL;
    }

    return (PyObject *)self;
};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyRegion_doc,"region() -> region \\\n\
region(otherRegion) -> region \\\n\
region(x, y, w, h, type = region.RECTANGLE) -> region \n\
\n\
Creates a rectangular or elliptical region. \n\
\n\
This class is a wrapper for the class ``QRegion`` of `Qt`. It provides possibilities for \n\
creating pixel-based regions. Furtherone you can calculate new regions based on the \n\
intersection, union or subtraction of other regions. Based on the region it is \n\
possible to get a uint8 masked dataObject, where every point within the entire \n\
region has the value 255 and all other values 0 \n\
\n\
If the constructor is called without argument, an empty region is created. \n\
\n\
Parameters \n\
---------- \n\
otherRegion : region \n\
    Pass this object of :class:`region` to create a copied object of it. \n\
x : int\n\
    x-coordinate of the reference corner of the region \n\
y : int\n\
    y-coordinate of the reference corner of the region \n\
w : int\n\
    width of the region \n\
h : int\n\
    height of the region \n\
type : int, optional \n\
    ``region.RECTANGLE`` creates a rectangular region (default). \n\
    ``region.ELLIPSE`` creates an elliptical region, which is placed inside of the \n\
    given boundaries.");
int PythonRegion::PyRegion_init(PyRegion *self, PyObject *args, PyObject * kwds)
{
    int x,y,w,h;
    int t = QRegion::Rectangle;
    PyObject *other = NULL;
	const char *kwlist[] = { "x", "y", "w", "h", "type", NULL };

    if(!args || PyTuple_Size(args) == 0)
    {
        DELETE_AND_SET_NULL(self->r);
        self->r = new QRegion();
    }
    else if(PyArg_ParseTuple(args,"O!",&PyRegionType,other))
    {
        DELETE_AND_SET_NULL(self->r);
        PyRegion *otherRegion = (PyRegion*)other;
        if(otherRegion->r == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Region of other region object is NULL");
            return -1;
        }
        self->r = new QRegion(*(otherRegion->r));
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "iiii|i", const_cast<char**>(kwlist), &x, &y, &w, &h, &t))
    {
        if (w < 1 || h < 1)
        {
            PyErr_SetString(PyExc_RuntimeError, "Width and height of region must be > 0.");
            return -1;
        }

        //QRegion has an upper limit for too many internal rectangles. If this limit is reached, a QWarning is emitted
        //and self->r->rect() returns 0.
        if(t == QRegion::Rectangle)
        {
            DELETE_AND_SET_NULL(self->r);
            self->r = new QRegion(x,y,w,h, QRegion::Rectangle);
        }
        else if(t == QRegion::Ellipse)
        {
            DELETE_AND_SET_NULL(self->r);
            self->r = new QRegion(x,y,w,h, QRegion::Ellipse);
        }

        if (self->r->rectCount() == 0)
        {
            delete self->r;
            self->r = NULL;
            PyErr_SetString(PyExc_RuntimeError, "Region cannot be created from a huge polygon. Upper limit reached.");
            return -1;
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
        result = PyUnicode_FromFormat("region(empty)");
    }
    else
    {
        result = PyUnicode_FromFormat("region(rects: %i)", self->r->rectCount() );
    }
    return result;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyRegionContains_doc,"contains(x, y, w = -1, h = -1) -> bool \n\
\n\
This method returns True, if the given point (x,y) or rectangle (x,y,w,h) is fully \n\
contained in this region. Otherwise returns False.\n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    x-coordinate of one corner of the new rectangular region \n\
y : int \n\
    y-coordinate of one corner of the new rectangular region \n\
w : int, optional \n\
    width of the new rectangular region. If not given, point is assumed. \n\
h : int, optional \n\
    height of the new rectangular region. If not given, point is assumed.\n\
\n\
Returns \n\
------- \n\
bool \n\
    True if point or rectangle is contained in region, otherwise False.");
/*static*/ PyObject* PythonRegion::PyRegion_contains(PyRegion *self, PyObject *args, PyObject *kwds)
{
    if(!self || self->r == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "region is not available");
        return NULL;
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
PyDoc_STRVAR(pyRegionIntersected_doc,"intersected(x, y, w, h) -> region \\\n\
intersected(region) -> region \n\
\n\
Returns a new region which is the intersection of the given region and this region. \n\
\n\
The intersection only contains points that are part of both regions. \n\
The given region can either by a :class:`region` object or a rectangular \n\
region, defined by its corner points (``x``, ``y``) and its width ``w`` \n\
and height ``h``. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    x-coordinate of one corner of the new rectangular region \n\
y : int \n\
    y-coordinate of one corner of the new rectangular region \n\
w : int \n\
    width of the new rectangular region \n\
h : int \n\
    height of the new rectangular region \n\
region : region \n\
    another instance of region \n\
\n\
Returns \n\
------- \n\
region \n\
    new intersected region.");
/*static*/ PyObject* PythonRegion::PyRegion_intersected(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "iiii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
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
PyDoc_STRVAR(pyRegionIntersects_doc,"intersects(x, y, w, h) -> bool \\\n\
intersects(region) -> bool \n\
\n\
Returns True if this region intersects with the given region, otherwise False. \n\
\n\
The given region can either by a :class:`region` object or a rectangular \n\
region, defined by its corner points (``x``, ``y``) and its width ``w`` \n\
and height ``h``. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    x-coordinate of one corner of the new rectangular region \n\
y : int \n\
    y-coordinate of one corner of the new rectangular region \n\
w : int \n\
    width of the new rectangular region \n\
h : int \n\
    height of the new rectangular region \n\
region : region \n\
    another instance of region \n\
\n\
Returns \n\
------- \n\
bool \n\
    True if both regions intersect, otherwise False.");
/*static*/ PyObject* PythonRegion::PyRegion_intersects(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    bool result;

    if(PyArg_ParseTupleAndKeywords(args, kwds, "iiii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
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
PyDoc_STRVAR(pyRegionSubtracted_doc,"subtracted(x, y, w, h) -> region \\\n\
subtracted(region) -> region \n\
\n\
This method returns a new region, which is the given, new region subtracted from this region. \n\
\n\
The given region can either by a :class:`region` object or a rectangular \n\
region, defined by its corner points (``x``, ``y``) and its width ``w`` \n\
and height ``h``. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    x-coordinate of one corner of the new rectangular region \n\
y : int \n\
    y-coordinate of one corner of the new rectangular region \n\
w : int \n\
    width of the new rectangular region \n\
h : int \n\
    height of the new rectangular region \n\
region : region \n\
    another instance of region \n\
\n\
Returns \n\
------- \n\
region \n\
    new, subtraced region.");
/*static*/ PyObject* PythonRegion::PyRegion_subtracted(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "iiii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
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
PyDoc_STRVAR(pyRegionTranslate_doc,"translate(x, y)\n\
\n\
This method translates this region by the given translation values. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    translation in x-direction \n\
y : int \n\
    translation in y-direction \n\
\n\
See Also \n\
-------- \n\
translated");
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
PyDoc_STRVAR(pyRegionTranslated_doc,"translated(x, y) -> region\n\
\n\
This method returns a new region, which is translated by the given distances in x and y direction. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    translation in x-direction \n\
y : int \n\
    translation in y-direction \n\
\n\
Returns \n\
------- \n\
region \n\
    new translated region.\n\
\n\
See Also \n\
-------- \n\
translate");
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
PyDoc_STRVAR(pyRegionUnited_doc,"united(x, y, w, h) -> region \\\n\
united(region) -> region \n\
\n\
returns a region which is the union of the given region with this region. \n\
\n\
This method returns a new region, which is the union of this region with the given region. \n\
The union contains all areas, that are contained in any of both regions. \n\
\n\
The given region can either by a :class:`region` object or a rectangular \n\
region, defined by its corner points (``x``, ``y``) and its width ``w`` \n\
and height ``h``. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    x-coordinate of one corner of a rectangular region \n\
y : int \n\
    y-coordinate of one corner of a rectangular region \n\
w : int \n\
    width of the new rectangular region \n\
h : int \n\
    height of the new rectangular region \n\
region : region \n\
    another instance of region \n\
\n\
Returns \n\
------- \n\
region \n\
    new united region.");
/*static*/ PyObject* PythonRegion::PyRegion_united(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "iiii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
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
PyDoc_STRVAR(pyRegionXored_doc,"xored(x, y, w, h) -> region \\\n\
xored(region) -> region \n\
\n\
returns a region which is an xor combination of the given region with this region. \n\
\n\
This method returns a new region, which is defined by an xor-combination of this \n\
region with the given region. \n\
\n\
The given region can either by a :class:`region` object or a rectangular \n\
region, defined by its corner points (``x``, ``y``) and its width ``w`` \n\
and height ``h``. \n\
\n\
Parameters \n\
---------- \n\
x : int \n\
    x-coordinate of one corner of a rectangular region \n\
y : int \n\
    y-coordinate of one corner of a rectangular region \n\
w : int \n\
    width of the new rectangular region \n\
h : int \n\
    height of the new rectangular region \n\
region : region \n\
    another instance of region \n\
\n\
Returns \n\
------- \n\
region \n\
    new xored region.");
/*static*/ PyObject* PythonRegion::PyRegion_xored(PyRegion *self, PyObject *args, PyObject *kwds)
{
    int x,y,w,h;
    PyObject *other = NULL;

    const char *kwlist[] = {"x", "y", "w", "h", NULL};
    const char *kwlist2[] = {"region", NULL};

    if(PyArg_ParseTupleAndKeywords(args, kwds, "iiii", const_cast<char**>(kwlist), &x, &y, &w, &h) )
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
PyDoc_STRVAR(pyRegion_getEmpty_doc,
"bool: Returns True if this region is empty, otherwise False");
/*static*/ PyObject* PythonRegion::PyRegion_getEmpty(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "region is not available");
        return NULL;
    }

    if(self->r->isEmpty())
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyRegion_getRectCount_doc,
"int: Returns the number of rectangles in this region");
/*static*/ PyObject* PythonRegion::PyRegion_getRectCount(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "region is not available");
        return NULL;
    }

    return Py_BuildValue( "i", self->r->rectCount() );
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyRegion_getRects_doc,
"list of list of int: Returns list of rectangles, whose union defines this region. \n\
\n\
Each rectangle is given by a list of (x, y, width, height).");
/*static*/ PyObject* PythonRegion::PyRegion_getRects(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "region is not available");
        return NULL;
    }

#if (QT_VERSION >= QT_VERSION_CHECK(5,8,0))
    auto it = self->r->begin();
    auto it_end = self->r->end();
#else
    QVector<QRect> rects = self->r->rects();
    auto it = rects.constBegin();
    auto it_end = rects.constEnd();
#endif

    PyObject *ret = PyList_New( self->r->rectCount() );
    Py_ssize_t i = 0;
    PyObject *t;

    for (; it != it_end; ++it)
    {
        t = PyList_New(4);
        PyList_SetItem(t,0, PyLong_FromLong(it->x()));
        PyList_SetItem(t,1, PyLong_FromLong(it->y()));
        PyList_SetItem(t,2, PyLong_FromLong(it->width()));
        PyList_SetItem(t,3, PyLong_FromLong(it->height()));
        PyList_SetItem(ret, i++, t); //steals reference
    }

    return ret;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyRegion_getBoundingRect_doc,
"list of int or None: Returns the bounding rectangle of this region or None if it is empty. \n\
\n\
The bounding rectangle is given by a list (x, y, width, height).");
/*static*/ PyObject* PythonRegion::PyRegion_getBoundingRect(PyRegion *self, void * /*closure*/)
{
    if(!self || self->r == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "region is not available");
        return NULL;
    }

    QRect b = self->r->boundingRect();

    if(b.isNull())
    {
        Py_RETURN_NONE;
    }

    PyObject *t = PyList_New(4);
    PyList_SetItem(t,0, PyLong_FromLong(b.x()));
    PyList_SetItem(t,1, PyLong_FromLong(b.y()));
    PyList_SetItem(t,2, PyLong_FromLong(b.width()));
    PyList_SetItem(t,3, PyLong_FromLong(b.height()));

    return t;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyRegionCreateMask_doc,"createMask(boundingRegion = None) -> dataObject \n\
\n\
Returns a :class:`~itom.dataObject` with dtype ``uint8`` whose shape corresponds to the \n\
width and height of the bounding rectangle. All pixels contained in the region have a \n\
value of ``255`` while the rest is set to ``0``. The offset value of the dataObject is \n\
set such that it fits to the real position of the region, since the first element \n\
in the dataObject corresponds to the left upper corner of the bounding rectangle.\n\
\n\
Indicate a ``boundingRegion`` in order to increase the size of the returned data object. \n\
Its size will have the size of the union between the boundingRegion and the region.\n\
\n\
Parameters \n\
---------- \n\
boundingRegion : region, optional\n\
    If a :class:`region` object is given, the shape of the returned :class:`dataObject`\n\
    is the maximum (union) between this ``boundingRegion`` and this region. \n\
\n\
Returns \n\
------- \n\
mask : dataObject");
/*static*/ PyObject* PythonRegion::PyRegion_createMask(PyRegion *self, PyObject *args, PyObject *kwds)
{
    if(!self || self->r == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "region is not available");
        return NULL;
    }

    PyObject *boundingRegion = NULL;
    const char *kwlist[] = {"boundingRegion", NULL};
    QRect bounds;
    QRect r = self->r->boundingRect();

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O!", const_cast<char**>(kwlist), &PyRegionType, &boundingRegion) )
    {
        return NULL;
    }
    else
    {
        if (boundingRegion)
        {
            QRegion reg = self->r->united( *(((PyRegion*)boundingRegion)->r) );
            bounds = reg.boundingRect().united(self->r->boundingRect());
        }
        else
        {
            bounds = r;
        }
    }

    int w = bounds.width();
    int h = bounds.height();
    int x = bounds.x();
    int y = bounds.y();


    ito::DataObject *d = new ito::DataObject();
    d->zeros(h, w, ito::tUInt8);
    d->setAxisOffset(0,-y);
    d->setAxisOffset(1,-x);

    ito::uint8 *ptr = ((cv::Mat*)(d->get_mdata()[0]))->ptr(0); //continuous

#if (QT_VERSION >= QT_VERSION_CHECK(5,8,0))
    auto it = self->r->begin();
    auto it_end = self->r->end();
#else
    QVector<QRect> rects = self->r->rects();
    auto it = rects.constBegin();
    auto it_end = rects.constEnd();
#endif

    for(; it != it_end; ++it)
    {
        for(int m = it->y(); m < (it->y() + it->height()) ; m++)
        {
            for(int n = it->x(); n < (it->x() + it->width()) ; n++)
            {
                ptr[ (n-x) + (m-y)*w ] = 255;
            }
        }
    }

    ito::PythonDataObject::PyDataObject *dObj = ito::PythonDataObject::createEmptyPyDataObject();
    dObj->dataObject = d;
    return (PyObject*)dObj;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonRegion::PyRegion_Reduce(PyRegion *self, PyObject *args)
{
    PyObject *stateTuple = NULL;

    if(self->r)
    {
        QByteArray ba;
        QDataStream d(&ba, QIODevice::WriteOnly | QIODevice::Truncate);
        d << *(self->r);

        stateTuple = PyBytes_FromStringAndSize( ba.data(), ba.size() );
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
/*static*/ PyObject* PythonRegion::PyRegion_SetState(PyRegion *self, PyObject *args)
{
    PyObject *data = NULL;
    if(!PyArg_ParseTuple(args, "O", &data))
    {
        return NULL;
    }

    if(data == Py_None)
    {
        Py_RETURN_NONE;
    }
    else
    {
        QByteArray ba( PyBytes_AS_STRING(data), PyBytes_GET_SIZE(data) );
        QDataStream d(&ba, QIODevice::ReadOnly);

        if(self->r)
        {
            d >> *(self->r);
        }
    }

    Py_RETURN_NONE;
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
    {"empty", (getter)PyRegion_getEmpty, NULL, pyRegion_getEmpty_doc, NULL},
    {"rectCount", (getter)PyRegion_getRectCount, NULL, pyRegion_getRectCount_doc, NULL },
    {"rects", (getter)PyRegion_getRects, NULL, pyRegion_getRects_doc, NULL },
    {"boundingRect", (getter)PyRegion_getBoundingRect, NULL, pyRegion_getBoundingRect_doc, NULL },
    {NULL}  /* Sentinel */
};

//-----------------------------------------------------------------------------
PyMethodDef PythonRegion::PyRegion_methods[] = {
    {"contains", (PyCFunction)PyRegion_contains, METH_VARARGS | METH_KEYWORDS, pyRegionContains_doc},
    {"intersected", (PyCFunction)PyRegion_intersected, METH_VARARGS | METH_KEYWORDS, pyRegionIntersected_doc},
    {"intersects", (PyCFunction)PyRegion_intersects, METH_VARARGS | METH_KEYWORDS, pyRegionIntersects_doc},
    {"subtracted", (PyCFunction)PyRegion_subtracted, METH_VARARGS | METH_KEYWORDS, pyRegionSubtracted_doc},
    {"translate", (PyCFunction)PyRegion_translate, METH_VARARGS | METH_KEYWORDS, pyRegionTranslate_doc},
    {"translated", (PyCFunction)PyRegion_translated, METH_VARARGS | METH_KEYWORDS, pyRegionTranslated_doc},
    {"united", (PyCFunction)PyRegion_united, METH_VARARGS | METH_KEYWORDS, pyRegionUnited_doc},
    {"xored", (PyCFunction)PyRegion_xored, METH_VARARGS | METH_KEYWORDS, pyRegionXored_doc},
    {"__reduce__", (PyCFunction)PyRegion_Reduce, METH_VARARGS,      "__reduce__ method for handle pickling commands"},
    {"__setstate__", (PyCFunction)PyRegion_SetState, METH_VARARGS,  "__setstate__ method for handle unpickling commands"},
    {"createMask", (PyCFunction)PyRegion_createMask, METH_VARARGS | METH_KEYWORDS, pyRegionCreateMask_doc },
    {NULL}  /* Sentinel */
};




//-----------------------------------------------------------------------------
PyModuleDef PythonRegion::PyRegionModule = {
    PyModuleDef_HEAD_INIT, "region", "Region (wrapper for QRegion)", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-----------------------------------------------------------------------------
PyTypeObject PythonRegion::PyRegionType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.region",             /* tp_name */
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
    pyRegion_doc,              /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
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
