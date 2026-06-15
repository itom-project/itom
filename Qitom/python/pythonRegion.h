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

#ifndef PYTHONREGION_H
#define PYTHONREGION_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

#include <qregion.h>

namespace ito
{

class PythonRegion
{
public:
    typedef struct
    {
        PyObject_HEAD
        QRegion *r;
    }
    PyRegion;

    #define PyRegion_Check(op) PyObject_TypeCheck(op, &ito::PythonRegion::PyRegionType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyRegion_dealloc(PyRegion *self);
    static PyObject* PyRegion_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyRegion_init(PyRegion *self, PyObject *args, PyObject *kwds);

    static PyObject* createPyRegion(const QRegion &region);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyRegion_repr(PyRegion *self);

    static PyObject* PyRegion_contains(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_intersected(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_intersects(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_subtracted(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_translate(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_translated(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_united(PyRegion *self, PyObject *args, PyObject *kwds);
    static PyObject* PyRegion_xored(PyRegion *self, PyObject *args, PyObject *kwds);

    static PyObject* PyRegion_createMask(PyRegion *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // pickling
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyRegion_Reduce(PyRegion *self, PyObject *args);
    static PyObject* PyRegion_SetState(PyRegion *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // number protocol
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyRegion_nbAdd(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbSubtract(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbAnd(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbXor(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbOr(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbInplaceAdd(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbInplaceSubtract(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbInplaceAnd(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbInplaceXor(PyRegion* o1, PyRegion* o2);
    static PyObject* PyRegion_nbInplaceOr(PyRegion* o1, PyRegion* o2);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyRegion_getEmpty(PyRegion *self, void *closure);
    static PyObject* PyRegion_getRectCount(PyRegion *self, void *closure);
    static PyObject* PyRegion_getRects(PyRegion *self, void *closure);
    static PyObject* PyRegion_getBoundingRect(PyRegion *self, void *closure);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    //static PyMemberDef PyRegion_members[];
    static PyMethodDef PyRegion_methods[];
    static PyGetSetDef PyRegion_getseters[];
    static PyTypeObject PyRegionType;
    static PyModuleDef PyRegionModule;
    static PyNumberMethods PyRegion_numberProtocol;

    static void PyRegion_addTpDict(PyObject *tp_dict);



};

}; //end namespace ito


#endif
