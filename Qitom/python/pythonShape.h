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

#ifndef PYTHONSHAPE_H
#define PYTHONSHAPE_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

#include <qpoint.h>
#include "../common/retVal.h"

namespace ito
{

class Shape; //forward declaration

class PythonShape
{
public:
    typedef struct
    {
        PyObject_HEAD
        Shape *shape;
    }
    PyShape;

    #define PyShape_Check(op) PyObject_TypeCheck(op, &ito::PythonShape::PyShapeType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyShape_dealloc(PyShape *self);
    static PyObject* PyShape_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyShape_init(PyShape *self, PyObject *args, PyObject *kwds);

    static PyObject* createPyShape(const Shape &shape);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyShape_repr(PyShape *self);
    static PyObject* PyShape_rotateDeg(PyShape *self, PyObject *args);
    static PyObject* PyShape_rotateRad(PyShape *self, PyObject *args);
    static PyObject* PyShape_translate(PyShape *self, PyObject *args);
    static PyObject* PyShape_region(PyShape *self);
    static PyObject* PyShape_normalized(PyShape *self);
    static PyObject* PyShape_contour(PyShape *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_contains(PyShape *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_copy(PyShape *self);

    //-------------------------------------------------------------------------------------------------
    // static members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyShape_StaticPoint(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_StaticLine(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_StaticCircle(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_StaticEllipse(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_StaticSquare(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_StaticRectangle(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyShape_StaticPolygon(PyObject *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // pickling
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyShape_reduce(PyShape *self, PyObject *args);
    static PyObject* PyShape_setState(PyShape *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyShape_getType(PyShape *self, void *closure);

    static PyObject* PyShape_getValid(PyShape *self, void *closure);

    static PyObject* PyShape_getFlags(PyShape *self, void *closure);
    static int PyShape_setFlags(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getIndex(PyShape *self, void *closure);
    static int PyShape_setIndex(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getName(PyShape *self, void *closure);
    static int PyShape_setName(PyShape *self, PyObject *value, void *closure);

	static PyObject* PyShape_getColor(PyShape *self, void *closure);
	static int PyShape_setColor(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getTransform(PyShape *self, void *closure);
    static int PyShape_setTransform(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getArea(PyShape *self, void *closure);

    static PyObject* PyShape_getBasePoints(PyShape *self, void *closure);

    static PyObject* PyShape_getPoint1(PyShape *self, void *closure);
    static int PyShape_setPoint1(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getPoint2(PyShape *self, void *closure);
    static int PyShape_setPoint2(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getCenter(PyShape *self, void *closure);
    static int PyShape_setCenter(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getAngleDeg(PyShape *self, void *closure);
    static int PyShape_setAngleDeg(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getAngleRad(PyShape *self, void *closure);
    static int PyShape_setAngleRad(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getRadius(PyShape *self, void *closure);
    static int PyShape_setRadius(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getWidth(PyShape *self, void *closure);
    static int PyShape_setWidth(PyShape *self, PyObject *value, void *closure);

    static PyObject* PyShape_getHeight(PyShape *self, void *closure);
    static int PyShape_setHeight(PyShape *self, PyObject *value, void *closure);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    //static PyMemberDef PyShape_members[];
    static PyMethodDef PyShape_methods[];
    static PyGetSetDef PyShape_getseters[];
    static PyTypeObject PyShapeType;
    static PyModuleDef PyShapeModule;

    static void PyShape_addTpDict(PyObject *tp_dict);

private:
    static QPointF PyObject2PointF(PyObject *value, ito::RetVal &retval, const char* paramName);
    static PyObject* PointF2PyObject(const QPointF &point);

};

}; //end namespace ito


#endif
