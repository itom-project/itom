/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#include "Common/typedefs.h"

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
    #define NO_IMPORT_ARRAY
//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#ifdef _DEBUG
    #undef _DEBUG
    #include "Python.h" 
    #define _DEBUG
#else
    #include "Python.h"   
#endif
#endif

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
			union
			{
				union
				{
					struct
					{
					ito::uint8 b;
					ito::uint8 g;
					ito::uint8 r;
					ito::uint8 a;
					};
					float rgb;
				};
				ito::uint32 rgba;
			};
        }
        PyRgba;


        #define PyRgba_Check(op) PyObject_TypeCheck(op, &PythonRgba::PyRgbaType)

        
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
        static PyObject* PyRgba_StaticZeros(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyRgba_StaticOnes(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyRgba_StaticRand(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyRgba_StaticRandN(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyRgba_StaticEye(PyObject *self, PyObject *args /*, PyObject *kwds*/);


};

} //end namespace ito

#endif
