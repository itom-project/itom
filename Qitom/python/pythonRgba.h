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

#ifndef PYTHONRGBA
#define PYTHONRGBA

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY
    #include "python/pythonWrapper.h"
#endif

#include "../../common/typeDefs.h"
#include "../../common/color.h"
#include "structmember.h"
#include <qobject.h>

namespace ito
{
class PythonRgba
    {

    public:

        //-------------------------------------------------------------------------------------------------
        // typedefs
        //-------------------------------------------------------------------------------------------------
        typedef struct
        {
            PyObject_HEAD
            ito::Rgba32 rgba;
        }
        PyRgba;


        #define PyRgba_Check(op) PyObject_TypeCheck(op, &ito::PythonRgba::PyRgbaType)


        //-------------------------------------------------------------------------------------------------
        // constructor, deconstructor, alloc, dellaoc
        //-------------------------------------------------------------------------------------------------

        static void PyRgba_dealloc(PyRgba *self);
        static PyObject *PyRgba_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
        static int PyRgba_init(PyRgba *self, PyObject *args, PyObject *kwds);

        static PyRgba* createEmptyPyRgba();
        static bool checkPyRgba(int number, PyObject* rgba1 = NULL, PyObject* rgba2 = NULL, PyObject* rgba3 = NULL);


        //-------------------------------------------------------------------------------------------------
        // general members
        //-------------------------------------------------------------------------------------------------
        static PyObject *PyRgba_name(PyRgba *self);

        static PyObject* PyRgba_repr(PyRgba *self);

        static PyObject *PyRgba_toGray(PyRgba *self);

        static PyObject *PyRgba_toColor(PyRgba *self);

        static PyObject* PyRgba_RichCompare(PyRgba *self, PyObject *other, int cmp_op);

        static PyGetSetDef PyRgba_getseters[];

        static PyObject* PyRgba_getValue(PyRgba *self, void *closure);
        static int PyRgba_setValue(PyRgba *self, PyObject *value, void *closure);

        static PyObject* PyRgba_Reduce(PyRgba *self, PyObject *args);
        static PyObject* PyRgba_SetState(PyRgba *self, PyObject *args);

        //-------------------------------------------------------------------------------------------------
        // number protocol
        //
        // python note: Binary and ternary functions must check the type of all their operands, and implement
        //    the necessary conversions (at least one of the operands is an instance of the defined type).
        //    If the operation is not defined for the given operands, binary and ternary functions must return
        //    Py_NotImplemented, if another error occurred they must return NULL and set an exception.
        //-------------------------------------------------------------------------------------------------
        static PyObject* PyRgba_nbAdd(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbSubtract(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbMultiply(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbPositive(PyObject* o1);
        static PyObject* PyRgba_nbAbsolute(PyObject* o1);
        static PyObject* PyRgba_nbInvert(PyObject* o1);
        static PyObject* PyRgba_nbLshift(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbRshift(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbAnd(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbXor(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbOr(PyObject* o1, PyObject* o2);

        static PyObject* PyRgba_nbInplaceAdd(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceSubtract(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceMultiply(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceLshift(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceRshift(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceAnd(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceXor(PyObject* o1, PyObject* o2);
        static PyObject* PyRgba_nbInplaceOr(PyObject* o1, PyObject* o2);


        //-------------------------------------------------------------------------------------------------
        // type structures
        //-------------------------------------------------------------------------------------------------
        static PyMemberDef PyRgba_members[];
        static PyMethodDef PyRgba_methods[];
        static PyTypeObject PyRgbaType;
        static PyModuleDef PyRgbaModule;

        static PyNumberMethods PyRgba_numberProtocol;

        //-------------------------------------------------------------------------------------------------
        // helper methods
        //-------------------------------------------------------------------------------------------------

        //-------------------------------------------------------------------------------------------------
        // static type methods
        //-------------------------------------------------------------------------------------------------


};

} //end namespace ito

#endif
