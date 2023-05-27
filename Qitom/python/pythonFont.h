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

#ifndef PYTHONFONT_H
#define PYTHONFONT_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

#include <qfont.h>

namespace ito
{

class PythonFont
{
public:
    typedef struct
    {
        PyObject_HEAD
        QFont *font;
    }
    PyFont;

    #define PyFont_Check(op) PyObject_TypeCheck(op, &ito::PythonFont::PyFontType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyFont_dealloc(PyFont *self);
    static PyObject* PyFont_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyFont_init(PyFont *self, PyObject *args, PyObject *kwds);

    static PyObject* createPyFont(const QFont &font);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFont_repr(PyFont *self);

    //-------------------------------------------------------------------------------------------------
    // pickling
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFont_Reduce(PyFont *self, PyObject *args);
    static PyObject* PyFont_SetState(PyFont *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFont_getFamily(PyFont *self, void *closure);
    static int PyFont_setFamily(PyFont *self, PyObject *value, void *closure);

    static PyObject* PyFont_getPointSize(PyFont *self, void *closure);
    static int PyFont_setPointSize(PyFont *self, PyObject *value, void *closure);

    static PyObject* PyFont_getWeight(PyFont *self, void *closure);
    static int PyFont_setWeight(PyFont *self, PyObject *value, void *closure);

    static PyObject* PyFont_getItalic(PyFont *self, void *closure);
    static int PyFont_setItalic(PyFont *self, PyObject *value, void *closure);

    static PyObject* PyFont_getUnderline(PyFont *self, void *closure);
    static int PyFont_setUnderline(PyFont *self, PyObject *value, void *closure);

    static PyObject* PyFont_getStrikeOut(PyFont *self, void *closure);
    static int PyFont_setStrikeOut(PyFont *self, PyObject *value, void *closure);

    //-------------------------------------------------------------------------------------------------
    // static
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFont_isFamilyInstalled(PyFont *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFont_installedFontFamilies(PyFont * self);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    //static PyMemberDef PyFont_members[];
    static PyMethodDef PyFont_methods[];
    static PyGetSetDef PyFont_getseters[];
    static PyTypeObject PyFontType;
    static PyModuleDef PyFontModule;

    static void PyFont_addTpDict(PyObject *tp_dict);



};

}; //end namespace ito


#endif
