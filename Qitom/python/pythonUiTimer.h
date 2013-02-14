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

#ifndef PYTHONUITIMER_H
#define PYTHONUITIMER_H

////python
//// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
//#ifdef _DEBUG
//    #undef _DEBUG
//    #ifdef linux
//        #include "Python.h"
//    #else
//        #include "include\Python.h"
//    #endif
//    #define _DEBUG
//#else
//    #ifdef linux
//        #include "Python.h"
//    #else
//        #include "include\Python.h"
//    #endif
//#endif

#include "pythonCommon.h"
//#include "pythonQtConversion.h"
//#include "pythonQtSignalMapper.h"

#include <qstring.h>
#include <qvariant.h>
#include <qobject.h>
//#include <qhash.h>


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

class PythonUiTimer
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
    PyUiTimer;

    //-------------------------------------------------------------------------------------------------
    // Timer
    //------------------------------------------------------------------------------------------------- 
    static void PyUiTimer_dealloc(PyUiTimer *self);
    static PyObject *PyUiTimer_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyUiTimer_init(PyUiTimer *self, PyObject *args, PyObject *kwds);
    static PyObject *PyUiTimer_repr(PyUiTimer *self);

    static PyGetSetDef  PyUiTimer_getseters[];
    static PyMemberDef  PyUiTimer_members[];
    static PyMethodDef  PyUiTimer_methods[];
    static PyTypeObject PyUiTimerType;
    static PyModuleDef  PyUiTimerModule;
    static PyObject *PyUiTimer_start(PyUiTimer *self);
    static PyObject *PyUiTimer_stop(PyUiTimer *self);
    static PyObject *PyUiTimer_isActive(PyUiTimer *self);
    static PyObject *PyUiTimer_setInterval(PyUiTimer *self, PyObject *args);
};

}; //end namespace ito

#endif
