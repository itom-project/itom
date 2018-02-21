/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONQTSIGNALMAPPER_H
#define PYTHONQTSIGNALMAPPER_H

#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API

    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #include "Python.h"
        #define _DEBUG
    #else
        #include "Python.h"
    #endif
#endif

#include "../global.h"

#include <qvariant.h>
#include <qobject.h>

namespace ito
{

class PythonQtSignalTarget
{
public:

    //! empty constructor
    PythonQtSignalTarget() :
        m_slotId(-1),
        m_signalId(-1),
        m_function(NULL),
        m_boundedInstance(NULL),
        m_boundedMethod(false)

    {
    };

    //! constructor
    /*!
        Constructs the virtual slot as target for any signal. If this slot is invoked, the given python method is executed.

        If the python method is a method (hence bounded), both a weak reference of the method and its containing instance is
        stored. If it is an unbounded function, only the weak reference to the function is saved.

        \param [in] argTypeList is a list of integer-based type number, describing the type of each argument as given by QMetaType
        \param [in] slotId is the assigned index for this slot (must be unique)
        \param [in] signalId is the index of the emitting signal
        \param [in] callabel is a python method or function (bounded or unbounded) that should be called if the slot is invoked
        \param [in] signal is the signature of the signal (for debugging reasons)
    */
    PythonQtSignalTarget(IntList &argTypeList, int slotId, int signalId, PyObject* callable, const char *signal) :
            m_slotId(slotId),
            m_signalId(signalId),
            m_function(NULL),
            m_boundedInstance(NULL),
            m_boundedMethod(false),
            m_signalName(signal)
    {
        m_argTypeList = argTypeList;
        PyObject *temp = NULL;

        if(PyMethod_Check(callable))
        {
            m_boundedMethod = true;
            Py_XDECREF(m_boundedInstance);
            Py_XDECREF(m_function);
            temp = PyMethod_Self(callable); //borrowed
            m_boundedInstance = PyWeakref_NewRef(temp, NULL); //new ref (weak reference used to avoid cyclic garbage collection)
            temp = PyMethod_Function(callable); //borrowed
            m_function = PyWeakref_NewRef(temp, NULL); //new ref
        }
        else if(PyFunction_Check(callable))
        {
            m_boundedMethod = false;
            Py_XDECREF(m_boundedInstance);
            Py_XDECREF(m_function);
            m_function = PyWeakref_NewRef(callable, NULL); //new ref
        }
    };
    
    //! copy constructor
    PythonQtSignalTarget(const PythonQtSignalTarget &copy) :
        m_slotId(-1),
        m_signalId(-1),
        m_function(NULL),
        m_boundedInstance(NULL),
        m_boundedMethod(false),
        m_signalName(copy.m_signalName)
    {
        Py_XDECREF(m_boundedInstance);
        Py_XDECREF(m_function);
        m_slotId = copy.slotId();
        m_signalId = copy.signalId();
        m_argTypeList = copy.argTypeList();

        m_boundedMethod = copy.m_boundedMethod;
        m_function = copy.m_function;
        Py_XINCREF(m_function);
        m_boundedInstance = copy.m_boundedInstance;
        Py_XINCREF(m_boundedInstance);
    }
    
    //! destructor
    ~PythonQtSignalTarget()
    {
        Py_XDECREF(m_boundedInstance);
        Py_XDECREF(m_function);
        m_argTypeList.clear();
    }

    //! gets the id of the original signal
    inline int signalId() const { return m_signalId; }

    //! gets the id that was assigned to this simulated slot
    inline int slotId()  const { return m_slotId; }

    // call the python callable with the given arguments (docs see source file)
    void call(void ** arguments) const;

    //! returns list of type-numbers of arguments
    inline IntList argTypeList() const { return m_argTypeList; };

    //! Compares this signal target with given values
    /*! checks whether the given signal index and the reference to the python method
        is the same than the values of this instance of PythonQtSignalTarget

        \param [in] signalId is the signal index (source of the signal-slot connection)
        \param [in] callable is the python slot method (slot, destination of the signal-slot connection)
        \return true if they are equal, else false.
    */
    bool isSame(int signalId, PyObject* callable) const 
    { 
        if(signalId == m_signalId)
        {
            if(PyMethod_Check(callable))
            {
                return PyMethod_Self(callable) == PyWeakref_GetObject(m_boundedInstance) && PyMethod_Function(callable) == PyWeakref_GetObject(m_function);
            }
            return callable == PyWeakref_GetObject(m_function);
        }
        return false;
    }

private:
    int m_slotId;                //!< index of this slot
    int m_signalId;                //!< index of the connected signal
    IntList m_argTypeList;        //!< type id's from QMetaType::type("..."), describing the arguments of the function-call
    PyObject *m_function;        //!< weak reference to the python-function, that acts as slot
    PyObject *m_boundedInstance; //!< weak reference to the python-class instance of the function (if the function is bounded) or NULL if the function is unbounded
    bool m_boundedMethod;        //!< true if the python function is bounded (non-static member of a class), else false
    QString m_signalName;        //!< signature of the signal (mainly used for debugging reasons)
};

/*! \class PythonQtSignalMapperBase
    \brief base class for PythonQtSignalMapper

    Since the class PythonQtSignalMapper needs to overwrite the method
    qt_metacall, which is automatically defined in the moc-file of any class having
    the Q_OBJECT macro, PythonQtSignalMapperBase has the Q_OBJECT macro defined such that
    the method qt_metacall is defined in its moc-file. The class PythonQtSignalMapper is then
    derived from PythonQtSignalMapperBase but has no Q_OBJECT macro defined. Then, no second
    moc-file is created and PythonQtSignalMapper can overwrite the method qt_metacall from
    PythonQtSignalMapperBase.

    The base idea of this class has been taken from the project PythonQt,
    MeVis Medical Solutions AG, 28359 Bremen.

    \sa PythonQtSignalMapper
*/
class PythonQtSignalMapperBase : public QObject 
{
    Q_OBJECT
public:
    PythonQtSignalMapperBase() {}; //!< constructor (no further functionality)
};


class PythonQtSignalMapper : public PythonQtSignalMapperBase
{

public:
    PythonQtSignalMapper(unsigned int initSlotCount); 
    ~PythonQtSignalMapper();

    bool addSignalHandler(QObject *obj, const char* signal, int sigId, PyObject* callable, IntList &argTypeList);
    bool removeSignalHandler(QObject *obj, const char* signal, int sigId, PyObject* callable);
    void removeSignalHandlers();
    virtual int qt_metacall(QMetaObject::Call c, int id, void **arguments);
   
private:
    QList<PythonQtSignalTarget> m_targets;    //!< list with all virtual slot targets that are the destination for any registered signal-slot-connection
    int m_slotCount;                        //!< index of the last virtual slot managed by this instance (auto-incremented)
};

} //end namespace ito

#endif
