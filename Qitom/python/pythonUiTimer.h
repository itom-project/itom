/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONUITIMER_H
#define PYTHONUITIMER_H

#include "pythonCommon.h"

#include <qstring.h>
#include <qvariant.h>
#include <qobject.h>

namespace ito 
{

class TimerCallback : public QObject
{
    Q_OBJECT
    public:
        TimerCallback() : m_function(NULL), m_boundedInstance(NULL), m_callbackArgs(0),  m_boundedMethod(0) {}
        ~TimerCallback() {}
        PyObject *m_function; //pyFunctionObject
        PyObject *m_boundedInstance; //self if bounded method, else null
//        PyObject m_callbackFunc;
//        PyObject m_callbackArgs;
        PyObject *m_callbackArgs; //
        int m_boundedMethod;

    public slots:
        void timeout(); 
};

class PythonTimer
{
public:

    //-------------------------------------------------------------------------------------------------
    // typedefs
    //------------------------------------------------------------------------------------------------- 
    typedef struct
    {
        PyObject_HEAD
        QTimer *timer;
        PyObject* base;
        TimerCallback *callbackFunc;
    }
    PyTimer;

    //-------------------------------------------------------------------------------------------------
    // Timer
    //------------------------------------------------------------------------------------------------- 
    static void PyTimer_dealloc(PyTimer *self);
    static PyObject *PyTimer_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyTimer_init(PyTimer *self, PyObject *args, PyObject *kwds);
    static PyObject *PyTimer_repr(PyTimer *self);

    static PyGetSetDef  PyTimer_getseters[];
    static PyMemberDef  PyTimer_members[];
    static PyMethodDef  PyTimer_methods[];
    static PyTypeObject PyTimerType;
    static PyModuleDef  PyTimerModule;
    static PyObject *PyTimer_start(PyTimer *self);
    static PyObject *PyTimer_stop(PyTimer *self);
    static PyObject *PyTimer_isActive(PyTimer *self);
    static PyObject *PyTimer_setInterval(PyTimer *self, PyObject *args);
};

}; //end namespace ito

#endif
