/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#pragma once

/*
    Small inline helpers that hide differences between the CPython versions supported by itom.
    Python.h is included through pythonWrapper.h, which handles the debug build on Windows and the
    'slots' macro of Qt; never include Python.h directly.
*/

#ifndef Q_MOC_RUN
    #include "pythonWrapper.h"
#endif

namespace ito
{
namespace PythonCompat
{

//! returns a new reference to the object referenced by a weak reference, or NULL
/*!
    Replacement for PyWeakref_GetObject, which returns a borrowed reference and is deprecated
    since Python 3.13 (removed in 3.15), because a borrowed reference to a weakly referenced
    object can become dangling at any time (the object may be finalized by another thread in a
    free-threaded build or by a callback). PyWeakref_GetRef (Python 3.13) returns a strong
    reference instead.

    \param ref the weak reference object
    \return a new reference to the referenced object, or NULL if the weak reference is dead,
            ref is NULL or ref is not a weak reference. No Python exception is set in any case.
*/
inline PyObject* weakrefGetRef(PyObject* ref)
{
    if (ref == nullptr)
    {
        return nullptr;
    }

#if PY_VERSION_HEX >= 0x030D0000
    PyObject* obj = nullptr;

    // 1: alive (obj is a new reference), 0: dead (obj is NULL), -1: error (ref is no weakref)
    if (PyWeakref_GetRef(ref, &obj) < 0)
    {
        PyErr_Clear();
        return nullptr;
    }

    return obj;
#else
    PyObject* obj = PyWeakref_GetObject(ref); // borrowed, Py_None if dead, NULL + SystemError if no weakref

    if (obj == nullptr)
    {
        PyErr_Clear();
        return nullptr;
    }

    if (obj == Py_None)
    {
        return nullptr;
    }

    Py_INCREF(obj);
    return obj;
#endif
}

} // namespace PythonCompat
} // namespace ito
