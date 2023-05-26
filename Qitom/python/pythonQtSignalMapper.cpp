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

#include "pythonEngineInc.h"
#include "pythonQtSignalMapper.h"
#include "pythonQtConversion.h"

#include "../AppManagement.h"

#include <qmetaobject.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class PythonQtSignalMapper
    \brief This class provides the possibility to redirect any signal emitted in an user-defined GUI to different python methods

    Every user-defined dialog, main window or widget that is loaded by the python-class ui or that is loaded from a plugin,
    contains one instance of this class. Any signal from any item of this user interface, that is connected by a python
    script with any appropriate bounded or unbounded pyhton method, is handled by this instance. This works as follows:
    The Qt-signal-slot system obtains a virtual slot that fits to the requirements of the signal. Both are connected.
    Once the slot as a member of this class is called, the call is catched by the overwritten method qt_metacall. If anything
    is ok, the call is then redirected to the registered python method. Any python method that acts as slot for any signal
    of the dialog or window, is one instance of the class PythonQtSignalTarget.

    Since the mapper usually is created in the python thread, while the signaling widgets are in the main thread,
    the necessary thread-change is already handled when connection the signal of the widget with the virtual slot of this mapper
    instance.

    \sa PythonQtSignalMapperBase, PythonQtSignalTarget
*/

//! constructor
/*!
    Creates an instance of PythonQtSignalMapper and initializes the slot counter
    with the given value. Usually this initial slot counter is set to the highest slot number
    of the graphical user interface this mapper is assigned to.

    \param initSlotCount should be set to the methodOffset() value of the underlying
           QObject, in order to separate default signals and slots of the base class
           from new, virtually created slots.

    \todo: probably, m_slotCount can also be set to methodOffset() of the QObject
           base class of PythonQtSignalMapper and should not be given as argument.
           it should not be offset of the object, that emits the signal but of
           this object, that has the virtual slot!
*/
    PythonQtSignalMapper::PythonQtSignalMapper()
    {
        m_slotCount = this->metaObject()->methodOffset();
    }

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    Destroys this signal mapper and deletes the managed targets (virtual slots).
    The connected signals are automatically disconnected by Qt.
*/
PythonQtSignalMapper::~PythonQtSignalMapper()
{
    m_targets.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates signal-slot connection between the signal of any widget and a python method as slot
/*!
    The connection is established as follows:
    1. An instance of PythonQtSignalTarget is created as virtual slot for the corresponding python method
    2. This instance is appended to the target list m_targets.
    3. Using Qt-methods, the widget's signal is connected to the slot of this virtual target (auto-connection).
    4. This virtual slot gets the index of the member m_slotCount, that is incremented afterwards

    \param [in] obj is the instance derived from QObject that is the signaling instance
    \param [in] signal is the signature of the signal (Qt-syntax)
    \param [in] sigId is the Qt-internal ID of the signal (obtained by QMetaObject-system)
    \param [in] callable is a reference to the real python method, that should act as
                slot. This method can be bounded or unbounded.
    \param [in] argTypeList is a list of integer values that describe the Qt-internal
                type number for all arguments of the signal (type number with respect to
                QMetaType)
    \param [in] minRepeatInterval is a minimum amount of time (in ms) which has to be
                passed until the same signal-slot-connection is accepted again (additional
                signal emissions are blocked), default: 0 (no timeout)
    \return true if the connection could be established, else false.
*/
bool PythonQtSignalMapper::addSignalHandler(
    QObject *obj, const char* signal, int sigId,
    PyObject* callable, IntList &argTypeList, int minRepeatInterval)
{
    bool success = false;

    if (!PyCallable_Check(callable))
    {
        return success;
    }

    if (sigId >= 0)
    {
        PythonQtSignalTarget t(argTypeList, m_slotCount, sigId, callable, signal, minRepeatInterval);

        // now connect to ourselves with the new slot id
        if (QMetaObject::connect(obj, sigId, this, m_slotCount, Qt::AutoConnection, 0))
        {
            m_slotCount++;
            success = true;
            m_targets[t.slotId()] = t;
        }
    }

    return success;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! disconnects a certain connection
/*!
    Disconnects a certain signal-slot connection, that has previously been connected.
    This connection is described by the signaling object, the index of the signal (and its signature)
    and the python callable object (as virtual slot)

    \param [in] obj is the instance derived from QObject that is the signaling instance
    \param [in] sigId is the Qt-internal ID of the signal (obtained by QMetaObject-system)
    \param [in] callable is a reference to the real python method, that should act as slot. This method can be bounded or unbounded.
    \return true if the connection could be disconnected, else false.
*/
bool PythonQtSignalMapper::removeSignalHandler(QObject *obj, int sigId, PyObject* callable)
{
    bool found = false;

    if (sigId >= 0)
    {
        TargetMap::iterator it = m_targets.begin();

        while (it != m_targets.end())
        {
            if (it->isSame(sigId, callable))
            {
                QMetaObject::disconnect(obj, sigId, this, it->slotId());
                it = m_targets.erase(it);
                found = true;
                break;
            }
            else
            {
                it++;
            }
        }
    }

    return found;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! disconnects all signal-slot connections managed by this instane of PythonQtSignalMapper
/*!
    This disconnection is easily done by deleting the list of targets.
*/
void PythonQtSignalMapper::removeSignalHandlers()
{
    m_targets.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method invoked by Qt if a connected signal is emitted
/*!
    This method is overwritten from the method created by the Qt-moc process. It is called
    whenever a signal, connected to a slot or virtual (python) slot of this instance. At first,
    the instance of qt_metacall of the derived from QObject is called with the same parameters, in order
    to allow the usual Qt-communcation. If the slot-id could not be handled by the base implementation,
    all registered PythonQtSignalTarget instances are searched. If their internal slot-index corresponds to the
    index given as argument to this function, the PythonQtSignalTarget instance is called (method call) with
    the given arguments.

    \param [in] c provide basic information about the call (only used for passing to the base implementation)
    \param [in] id is the unique slot index of the slot to call.
    \param [in] arguments is an array of different argument variables, whose types corresponds to the type-number list, passed when the
                connection has been registered.
*/
int PythonQtSignalMapper::qt_metacall(QMetaObject::Call c, int id, void **arguments)
{
    if (c != QMetaObject::InvokeMetaMethod)
    {
        QObject::qt_metacall(c, id, arguments);
    }

    TargetMap::iterator it = m_targets.find(id);

    if (it != m_targets.end())
    {
        it->call(arguments);
    }

    return 0;
}


//-------------------------------------------------------------------------------------
//! empty constructor
PythonQtSignalTarget::PythonQtSignalTarget() :
    m_slotId(-1),
    m_signalId(-1),
    m_function(NULL),
    m_boundedInstance(NULL),
    m_callableType(Callable_Invalid)

{
};

//-------------------------------------------------------------------------------------
//! constructor
/*!
    Constructs the virtual slot as target for any signal. If this slot is
    invoked, the given python method is executed.

    If the python method is a method (hence bounded), both a weak reference
    of the method and its containing instance is stored. If it is an
    unbounded function, only the weak reference to the function is saved.

    \param [in] argTypeList is a list of integer-based type number, describing
                the type of each argument as given by QMetaType
    \param [in] slotId is the assigned index for this slot (must be unique)
    \param [in] signalId is the index of the emitting signal
    \param [in] callabel is a python method or function (bounded or unbounded)
                that should be called if the slot is invoked
    \param [in] signal is the signature of the signal (for debugging reasons)
    \param [in] minRepeatInterval is a minimum amount of time (in ms) which has
                to be passed until the same signal-slot-connection is accepted
                again (additional signal emissions are blocked), default: 0
                (no timeout)
*/
PythonQtSignalTarget::PythonQtSignalTarget(
    IntList &argTypeList, int slotId, int signalId,
    PyObject* callable, const char *signal, int minRepeatInterval) :
        m_slotId(slotId),
        m_signalId(signalId),
        m_function(NULL),
        m_boundedInstance(NULL),
        m_callableType(Callable_Invalid),
        m_signalName(signal),
        m_minRepeatInterval(minRepeatInterval),
        m_argTypeList(argTypeList)
{
    m_elapsedTimer.invalidate();

    if (PyMethod_Check(callable))
    {
        m_callableType = Callable_Method;
        PyObject *temp = PyMethod_Self(callable); //borrowed
        m_boundedInstance = PyWeakref_NewRef(temp, NULL); //new ref (weak reference used to avoid cyclic garbage collection)
        temp = PyMethod_Function(callable); //borrowed
        m_function = PyWeakref_NewRef(temp, NULL); //new ref
    }
    else if (PyCFunction_Check(callable))
    {
        m_callableType = Callable_CFunction;
        Py_INCREF(callable);
        m_function = callable; //new ref
    }
    else if (PyCallable_Check(callable))
    {
        // any other callable, especially PyFunction, but also functools.partial etc.
        m_callableType = Callable_Function;
        Py_INCREF(callable);
        m_function = callable; //new ref
    }
};

//-------------------------------------------------------------------------------------
//! copy constructor
PythonQtSignalTarget::PythonQtSignalTarget(const PythonQtSignalTarget &copy) :
    m_slotId(copy.m_slotId),
    m_signalId(copy.m_signalId),
    m_argTypeList(copy.m_argTypeList),
    m_function(copy.m_function),
    m_boundedInstance(copy.m_boundedInstance),
    m_callableType(copy.m_callableType),
    m_signalName(copy.m_signalName),
    m_minRepeatInterval(copy.m_minRepeatInterval)
{
    Py_XINCREF(m_function);
    Py_XINCREF(m_boundedInstance);
    m_elapsedTimer.invalidate();
}

//-------------------------------------------------------------------------------------
PythonQtSignalTarget& PythonQtSignalTarget::operator=(const PythonQtSignalTarget &rhs)
{
    Py_XDECREF(m_boundedInstance);
    Py_XDECREF(m_function);

    m_signalName = rhs.m_signalName;
    m_slotId = rhs.m_slotId;
    m_signalId = rhs.m_signalId;
    m_argTypeList = rhs.m_argTypeList;

    m_callableType = rhs.m_callableType;
    m_function = rhs.m_function;
    Py_XINCREF(m_function);
    m_boundedInstance = rhs.m_boundedInstance;
    Py_XINCREF(m_boundedInstance);

    m_minRepeatInterval = rhs.m_minRepeatInterval;
    m_elapsedTimer.invalidate();

    return *this;
}

//-------------------------------------------------------------------------------------
//! destructor
PythonQtSignalTarget::~PythonQtSignalTarget()
{
    Py_XDECREF(m_boundedInstance);
    Py_XDECREF(m_function);
    m_argTypeList.clear();
}

//-------------------------------------------------------------------------------------
//! Compares this signal target with given values
/*! checks whether the given signal index and the reference to the python method
    is the same than the values of this instance of PythonQtSignalTarget

    \param [in] signalId is the signal index (source of the signal-slot connection)
    \param [in] callable is the python slot method (slot, destination of the signal-slot connection)
    \return true if they are equal, else false.
*/
bool PythonQtSignalTarget::isSame(int signalId, PyObject* callable) const
{
    if (signalId == m_signalId)
    {
        // bounded, m_function is a weakref
        if (PyMethod_Check(callable))
        {
            return PyMethod_Self(callable) == PyWeakref_GetObject(m_boundedInstance) &&
                PyMethod_Function(callable) == PyWeakref_GetObject(m_function);
        }
        else //function or cfunction
        {
            //unbounded function, m_function is the function itself.
            return callable == m_function;
        }
    }

    return false;
}


//-------------------------------------------------------------------------------------
//! invokes the python method or function
/*!
    If the slot is invoked, the python method or function is executed by this function.
    Usually the method is directly executed. If the user toggled the debug-button in the main
    window, the method is started in debug-mode. However, this is only done if python is in idle-mode
    at the point of the start of the execution.

    The exeuction is as follows:
    1. It is checked whether the method should be executed or started in debug-mode
    2. The given arguments are marshalled to PyObject-values and added to a tuple.
    3. The python method or function is called.

    \param [in] arguments are the arguments of the emitted signal.
*/
void PythonQtSignalTarget::call(void ** arguments)
{
    if (m_minRepeatInterval > 0)
    {
        if (m_elapsedTimer.isValid() == false)
        {
            //start the timer for the first time
            m_elapsedTimer.start();
        }
        else
        {
            if (m_elapsedTimer.elapsed() < m_minRepeatInterval)
            {
                //the same signal has been fired to the same slot less than m_minRepeatInterval ms ago: ignore this call
                return;
            }
            else
            {
                m_elapsedTimer.restart();
            }
        }
    }

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (Py_IsInitialized() == 0)
    {
        qDebug("python is not available any more");
        return;
    }

    if (m_function == NULL)
    {
        qDebug("invalid callable slot.");
        return;
    }

    PyGILState_STATE state = PyGILState_Ensure();

    bool debug = false;

    if (pyEngine)
    {
        debug = pyEngine->execInternalCodeByDebugger();
    }

    PyObject *argTuple = PyTuple_New(m_argTypeList.size());
    PyObject *temp = NULL;
    bool argParsingError = false;

    //arguments[0] is return argument

    for (int i = 0; i < m_argTypeList.size(); i++)
    {
        temp = PythonQtConversion::ConvertQtValueToPythonInternal(m_argTypeList[i], arguments[i + 1]); //new reference

        if (temp)
        {
            PyTuple_SetItem(argTuple, i, temp); //steals reference
        }
        else //error message is set in ConvertQtValueToPythonInternal
        {
            PyErr_PrintEx(0);
            PyErr_Clear();
            argParsingError = true;
            break;
        }
    }

    if (!argParsingError)
    {
        if (m_callableType == Callable_Function)
        {
            if (debug)
            {
                pyEngine->pythonDebugFunction(m_function, argTuple, true);
            }
            else
            {
                pyEngine->pythonRunFunction(m_function, argTuple, true);
            }
        }
        else if (m_callableType == Callable_Method)
        {
            PyObject *func = PyWeakref_GetObject(m_function);
            PyObject *inst = PyWeakref_GetObject(m_boundedInstance);

            if (func == Py_None || inst == Py_None)
            {
                PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
                PyErr_PrintEx(0);
                PyErr_Clear();
            }
            else
            {
                PyObject *method = PyMethod_New(func, inst); //new ref

                if (debug)
                {
                    pyEngine->pythonDebugFunction(method, argTuple, true);
                }
                else
                {
                    pyEngine->pythonRunFunction(method, argTuple, true);
                }

                Py_XDECREF(method);
            }
        }
        else if (m_callableType == Callable_CFunction)
        {
            PyCFunctionObject* cfunc = (PyCFunctionObject*)m_function;
            PyObject *method = PyCFunction_NewEx(cfunc->m_ml, cfunc->m_self, NULL);

            if (method)
            {
                if (debug)
                {
                    pyEngine->pythonDebugFunction(method, argTuple, true);
                }
                else
                {
                    pyEngine->pythonRunFunction(method, argTuple, true);
                }
            }

            Py_XDECREF(method);
        }
        else
        {
            qDebug("invalid m_callableType.");
        }
    }

    Py_DECREF(argTuple);

    PyGILState_Release(state);
}

} //end namespace ito
