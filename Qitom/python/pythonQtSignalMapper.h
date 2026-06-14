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

#ifndef PYTHONQTSIGNALMAPPER_H
#define PYTHONQTSIGNALMAPPER_H

#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API

    #include "python/pythonWrapper.h"
#endif

#include "../global.h"

#include <qvariant.h>
#include <qobject.h>
#include <qelapsedtimer.h>
#include <qhash.h>

namespace ito
{

class PythonQtSignalTarget
{
public:

    //! empty constructor
    PythonQtSignalTarget();

    //! constructor
    PythonQtSignalTarget(IntList &argTypeList, int slotId, int signalId, PyObject* callable, const char *signal, int minRepeatInterval);

    //! copy constructor
    PythonQtSignalTarget(const PythonQtSignalTarget &copy);

    //! destructor
    ~PythonQtSignalTarget();

    //! assignment operator
    PythonQtSignalTarget &operator=(const PythonQtSignalTarget &rhs);

    //! gets the id of the original signal
    inline int signalId() const { return m_signalId; }

    //! gets the id that was assigned to this simulated slot
    inline int slotId()  const { return m_slotId; }

    // call the python callable with the given arguments (docs see source file)
    void call(void ** arguments);

    //! returns list of type-numbers of arguments
    inline IntList argTypeList() const { return m_argTypeList; };

    //! Compares this signal target with given values
    bool isSame(int signalId, PyObject* callable) const;

private:
    int m_slotId;                  //!< index of this slot
    int m_signalId;                //!< index of the connected signal
    IntList m_argTypeList;         //!< type id's from QMetaType::type("..."), describing the arguments of the function-call

    enum CallableType
    {
        Callable_Invalid, //!< the callable is invalid

        //!< class method (written in python), the function is stored in m_function, the self object is stored in m_boundedInstance
        Callable_Method,

        //!< unbounded python method, the function is stored in m_function, m_boundedInstance is NULL
        Callable_Function,

        //!< function, written in C, stored in m_function. m_boundedInstance is NULL, since the potential self object is also contained in the CFunction object
        Callable_CFunction
    };

    /* If the target is a bounded method, m_boundedMethod is true and
    this member holds a Python weak reference to the method, that acts as slot.
    m_boundedInstance is != NULL then.

    If the target is an unbounded function, m_boundedMethod is false and
    this member holds a new reference to the function itself (that acts as slot).
    m_boundedInstance is NULL then. */
    PyObject *m_function;
    PyObject *m_boundedInstance;   //!< weak reference to the python-class instance of the function (if the function is bounded) or NULL if the function is unbounded
    CallableType m_callableType;   //!< type of the python callable (see CallableType)
    QString m_signalName;          //!< signature of the signal (mainly used for debugging reasons)
    QElapsedTimer m_elapsedTimer;  //!< see m_minRepeatInterval

    /*
    if > 0, every call of a certain slot by a certain signal with restart m_elapsedTimer. If the same signal-slot-connection is
    called another time, while less than m_minRepeatInterval ms have expired, this new invoke will be ignored.

    This can be used to prevent that very fast signal emissions will overflow the call queue for the connected callable python method.
    */
    int m_minRepeatInterval;
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

/*! \class PythonQtSignalMapper

    This class must not define the Q_OBJECT macro, since the method qt_metacall,
    that is usually generated by the Qt moc-process (run, if Q_OBJECT is defined),
    is manually created within this class (this is the special thing). The method qt_metacall
    overwrites the moc-generated method qt_metacall of the baseclass. Since the baseclass has no
    slots or signals defined, its specific method qt_metacall is 'empty'.

    Every python object that wants to define slots on a python level, that can be connected
    to real signals of Qt classes (like widgets or plugins), should create an object of PythonQtSignalMapper.
    It is also possible that several python objects share one instance of PythonQtSignalMapper (e.g. all widgets
    share the PythonQtSignalMapper object of their top-level-widget (the dialog, main window...)).

    The PythonQtSignalMapper acts like a 'virtual' slot, can be bind to a signal of a QObject derived class.
    The slots are not defined in this class definition, but they are added at runtime and stored in the m_targets list.
    This list knows which python method or function should be called if a specific signal, indicated by its unique signal index,
    of a specific QObject instance (widget, plugin...) is emitted. The standard Qt signal-slot connection is hereby
    initialized in the addSignalHandler method. It is removed by removeSignalHandler or removeSignalHandlers.

    \sa PythonQtSignalMapperBase
*/
class PythonQtSignalMapper : public PythonQtSignalMapperBase
{

public:
    PythonQtSignalMapper();
    ~PythonQtSignalMapper();

    bool addSignalHandler(QObject *obj, const char* signal, int sigId, PyObject* callable, IntList &argTypeList, int minRepeatInterval);
    bool removeSignalHandler(QObject *obj, int sigId, PyObject* callable);
    void removeSignalHandlers();

    //! overwrites qt_metacall from PythonQtSignalMapperBase.
    virtual int qt_metacall(QMetaObject::Call c, int id, void **arguments);

private:
    typedef QMap<int, PythonQtSignalTarget> TargetMap;

    //!< list with all virtual slot targets that are the destination for any registered signal-slot-connection
    /* This list is generated as map, that maps the slotId of the PythonQtSignalTarget
    to the target itself (for a faster indexing).
    */
    TargetMap m_targets;

    //!< index of the last virtual slot managed by this instance (auto-incremented)
    int m_slotCount;

};

} //end namespace ito

#endif
