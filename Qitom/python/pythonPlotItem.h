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

#ifndef PYTHONPLOTITEM_H
#define PYTHONPLOTITEM_H

#include "pythonCommon.h"
#include "pythonUi.h"
#include "pythonQtConversion.h"
#include "pythonQtSignalMapper.h"
#include "pythonItomMetaObject.h"

#include <qstring.h>
#include <qvariant.h>
#include <qobject.h>
#include <qhash.h>
#include <qsharedpointer.h>
#include <qpointer.h>

namespace ito
{

class PythonPlotItem
{
public:

    //#################################################################################################
    // Figure
    //#################################################################################################

    //-------------------------------------------------------------------------------------------------
    // typedefs
    //-------------------------------------------------------------------------------------------------

    typedef struct
    {
        PythonUi::PyUiItem uiItem;
    }
    PyPlotItem;

    //-------------------------------------------------------------------------------------------------
    // macros
    //-------------------------------------------------------------------------------------------------
    #define PyPlotItem_Check(op) PyObject_TypeCheck(op, &ito::PythonPlotItem::PyPlotItemType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyPlotItem_dealloc(PyPlotItem *self);
    static PyObject *PyPlotItem_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyPlotItem_init(PyPlotItem *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPlotItem_repr(PyPlotItem *self);

    static PyObject* PyPlotItem_pickPoints(PyPlotItem *self, PyObject *args, PyObject *kwds);
    static PyObject* PyPlotItem_drawAndPickElements(PyPlotItem *self, PyObject *args, PyObject *kwds);
    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------

    //-------------------------------------------------------------------------------------------------
    // static members
    //-------------------------------------------------------------------------------------------------


    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    static PyGetSetDef  PyPlotItem_getseters[];
    static PyMemberDef  PyPlotItem_members[];
    static PyMethodDef  PyPlotItem_methods[];
    static PyTypeObject PyPlotItemType;
    static PyModuleDef  PyPlotItemModule;
    static void PyPlotItem_addTpDict(PyObject *tp_dict);

private:

};

}; //end namespace ito


#endif
