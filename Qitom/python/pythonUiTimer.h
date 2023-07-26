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

#ifndef PYTHONUITIMER_H
#define PYTHONUITIMER_H

#include "pythonCommon.h"

#include <qobject.h>
#include <qsharedpointer.h>

class QTimer; //forward declaration

namespace ito
{

class TimerCallback : public QObject
{
    Q_OBJECT
    public:
        TimerCallback();
        ~TimerCallback();

        enum class CallableType
        {
            Callable_Invalid, //!< the callable is invalid

            //!< class method (written in python), the function is stored in m_function, the self object is stored in m_boundedInstance
            Callable_Method,

            //!< unbounded python method, the function is stored in m_function, m_boundedInstance is NULL
            Callable_Function,

            //!< function, written in C, stored in m_function. m_boundedInstance is NULL, since the potential self object is also contained in the CFunction object
            Callable_CFunction
        };

        /* If the target is a bounded method, this member holds a Python
        weak reference (new ref) to the method, that acts as slot.
        m_boundedInstance is != nullptr then.

        If the target is an unbounded function,
        this member holds a new reference to the function itself (that acts as slot).
        m_boundedInstance is nullptr then. */
        PyObject *m_function;

        /* weak reference to the python-class instance of the
        function (if the function is bounded) or nullptr if the function is unbounded*/
        PyObject *m_boundedInstance;

        //!< type of the python callable (see CallableType)
        CallableType m_callableType;

        //!< new reference to a (empty) tuple with arguments passed to the callable function
        PyObject *m_callbackArgs;

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
        QSharedPointer<QTimer> timer;
        QSharedPointer<TimerCallback> callbackFunc;
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
