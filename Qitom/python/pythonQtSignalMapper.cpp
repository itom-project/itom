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

#include "pythonEngineInc.h"
#include "pythonQtSignalMapper.h"
#include "pythonQtConversion.h"

#include "../AppManagement.h"

#include <qmetaobject.h>

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
*/
PythonQtSignalMapper::PythonQtSignalMapper(unsigned int initSlotCount) : m_slotCount(initSlotCount) {}

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
    \param [in] callable is a reference to the real python method, that should act as slot. This method can be bounded or unbounded.
    \param [in] argTypeList is a list of integer values that describe the Qt-internal type number for all arguments of the signal (type number with respect to QMetaType)
    \return true if the connection could be established, else false.
*/
bool PythonQtSignalMapper::addSignalHandler(QObject *obj, const char* signal, int sigId, PyObject* callable, IntList &argTypeList)
{
    bool flag = false;
    if (sigId>=0)
    {
        PythonQtSignalTarget t(argTypeList, m_slotCount, sigId, callable, signal);
        m_targets.append(t);
        // now connect to ourselves with the new slot id
        if (QMetaObject::connect(obj, sigId, this, m_slotCount, Qt::AutoConnection, 0))
        {
            m_slotCount++;
            flag = true;
        }
    }
    return flag;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! disconnects a certain connection
/*!
    Disconnects a certain signal-slot connection, that has previously been connected.
    This connection is described by the signaling object, the index of the signal (and its signature)
    and the python callable object (as virtual slot)

    \param [in] obj is the instance derived from QObject that is the signaling instance
    \param [in] signal is the signature of the signal (Qt-syntax)
    \param [in] sigId is the Qt-internal ID of the signal (obtained by QMetaObject-system)
    \param [in] callable is a reference to the real python method, that should act as slot. This method can be bounded or unbounded.
    \return true if the connection could be disconnected, else false.
*/
bool PythonQtSignalMapper::removeSignalHandler(QObject *obj, const char* /*signal*/, int sigId, PyObject* callable)
{
    bool found = false;
    if (sigId>=0)
    {
        QMutableListIterator<PythonQtSignalTarget> i(m_targets);
        while (i.hasNext())
        {
            if (i.next().isSame(sigId, callable))
            {
                QMetaObject::disconnect(obj, sigId, this, i.value().slotId());
                i.remove();
                found = true;
                break;
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
#if QT_VERSION < 0x050000
    int PythonQtSignalMapper::qt_metacall(QMetaObject::Call c, int id, void **arguments)
#else
    int PythonQtSignalMapper::qt_metacall(QMetaObject::Call c, int id, char **arguments)
#endif
{
    if (c != QMetaObject::InvokeMetaMethod)
    {
        QObject::qt_metacall(c, id, arguments);
    }

//    bool found = false;
    foreach(const PythonQtSignalTarget& t, m_targets)
    {
        if (t.slotId() == id)
        {
//            found = true;
            t.call(arguments);
            break;
        }
    }
  return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
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
#if QT_VERSION < 0x050000
    void PythonQtSignalTarget::call(void ** arguments) const
#else
    void PythonQtSignalTarget::call(char ** arguments) const
#endif
{
    //qDebug() << "signaltarget::call in thread: " << QThread::currentThreadId ();


    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (Py_IsInitialized() == 0)
    {
        qDebug("python is not available any more");
        return;
    }
//    PyGILState_STATE state = PyGILState_Ensure();

    bool debug = false;
    if (pyEngine)
    {
        debug = pyEngine->execInternalCodeByDebugger();
    }

    PyObject *argTuple = PyTuple_New(m_argTypeList.size());
    PyObject *temp = NULL;

    //arguments[0] is return argument

    for (int i=0;i<m_argTypeList.size();i++)
    {
        temp = PythonQtConversion::ConvertQtValueToPythonInternal(m_argTypeList[i],arguments[i+1]); //new reference
        if (temp)
        {
            PyTuple_SetItem(argTuple,i,temp); //steals reference
        }
    }

    //qDebug() << m_signalName.toLatin1().data() << endl;
    if (m_boundedMethod == false)
    {
        PyObject *func = PyWeakref_GetObject(m_function);
        if (func != Py_None)
        {
            if (debug)
            {
                pyEngine->pythonDebugFunction(func, argTuple);
            }
            else
            {
                pyEngine->pythonRunFunction(func, argTuple);
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else
    {
        PyObject *func = PyWeakref_GetObject(m_function);
        PyObject *inst = PyWeakref_GetObject(m_boundedInstance);

        if (func == Py_None || inst == Py_None)
        {
            PyErr_SetString(PyExc_RuntimeError, "The python slot method is not longer available");
            PyErr_Print();
            PyErr_Clear();
        }
        else
        {
            PyObject *method = PyMethod_New(func, inst); //new ref

            if (debug)
            {
                pyEngine->pythonDebugFunction(method, argTuple);
            }
            else
            {
                pyEngine->pythonRunFunction(method, argTuple);
            }

            Py_XDECREF(method);
        }
    }

    Py_DECREF(argTuple);

//    PyGILState_Release(state);
}

