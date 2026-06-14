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

#ifndef PYTHONFIGURE_H
#define PYTHONFIGURE_H

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

class PythonFigure
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
        QSharedPointer<unsigned int> guardedFigHandle;
        int rows;
        int cols;
        int currentSubplotIdx;
        PythonQtSignalMapper *signalMapper;
    }
    PyFigure;

    //-------------------------------------------------------------------------------------------------
    // macros
    //-------------------------------------------------------------------------------------------------
    #define PyFigure_Check(op) PyObject_TypeCheck(op, &ito::PythonFigure::PyFigureType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyFigure_dealloc(PyFigure *self);
    static PyObject *PyFigure_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyFigure_init(PyFigure *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFigure_repr(PyFigure *self);
    static PyObject* PyFigure_show(PyFigure *self, PyObject *args);
    static PyObject* PyFigure_hide(PyFigure *self);
    static PyObject* PyFigure_plot(PyFigure *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFigure_plot1(PyFigure *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFigure_plot2(PyFigure *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFigure_plot25(PyFigure *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFigure_liveImage(PyFigure *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFigure_matplotlib(PyFigure *self, PyObject *args, PyObject *kwds);
    static PyObject* PyFigure_plotly(PyFigure *self, PyObject *args, PyObject *kwds);

    static PyObject* PyFigure_getSubplot(PyFigure *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFigure_getHandle(PyFigure *self, void *closure);
    static PyObject* PyFigure_getDocked(PyFigure *self, void *closure);
    static int       PyFigure_setDocked(PyFigure *self, PyObject *value, void *closure);

    //-------------------------------------------------------------------------------------------------
    // static members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyFigure_close(PyFigure *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    static PyGetSetDef  PyFigure_getseters[];
    static PyMemberDef  PyFigure_members[];
    static PyMethodDef  PyFigure_methods[];
    static PyTypeObject PyFigureType;
    static PyModuleDef  PyFigureModule;
    static void PyFigure_addTpDict(PyObject *tp_dict);

private:

};

}; //end namespace ito


#endif
