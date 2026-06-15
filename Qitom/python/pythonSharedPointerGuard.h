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

#pragma once

//python
#ifndef Q_MOC_RUN
#include "python/pythonWrapper.h"
#endif

#include "../../common/sharedStructures.h"
#include "../global.h"

#include <qhash.h>
#include <qsharedpointer.h>

namespace ito
{

//! Guard for a shared pointer of a PyObject.
/*
If a PyObject wraps an object, like a dataObject, pointCloud etc,
which should accessed via a QSharedPointer, this class can be
used to safely get the sharedPointer. As long as the shared pointer exists,
the refcount of the original PyObject is incremented to make sure,
that the internal object is not destroyed. If the shared pointer is
destroyed, the PythonSharedPointerGuard::deleter is called and decrements
the refcount of the guarded PyObject.
*/
class PythonSharedPointerGuard
{
private:
    //!< if the caller of the deleter currently does not own the Python GIL,
    //!< this method is called to safely decrement the refcount of the
    //!< owner PyObject. This is done by a worker thread, that tries to
    //!< acquire the GIL, decrease the refcount and releas the GIL.
    static void safeDecrefPyObject2Async(PyObject* obj);

    //!< user-defined deleter for the QSharedPointer, released by createPythonSharedPointer
    template<typename _Tp> static void deleter(_Tp *sharedPointerData)
    {
        auto iter = m_hashTable.find((void*)sharedPointerData);

        if (iter != m_hashTable.end())
        {
            PyObject *val = iter.value();

            if (val)
            {
                if (PyGILState_Check())
                {
                    Py_DECREF(val);
                }
                else
                {
                    safeDecrefPyObject2Async(val);
                }
            }

            m_hashTable.erase(iter);
        }
    }

public:
    //!< main method to get the guarded shared pointer
    /*
    \param sharedPointerData is the object, owned by pyObjOwner, that
        should be returned within the QSharedPointer.
    \param pyObjOwner is the PyObject, that is the owner of the
        sharedPointerData. Its ref count is incremented by one as long
        as the returned shared pointer exists.
    */
    template<typename _Tp> static QSharedPointer<_Tp> createPythonSharedPointer(_Tp *sharedPointerData, PyObject *pyObjOwner)
    {
        Py_XINCREF(pyObjOwner);
        m_hashTable.insert((void*)sharedPointerData, pyObjOwner);
        return QSharedPointer<_Tp>(sharedPointerData, deleter<_Tp>);
    }

private:

    /* elements in this hash-table are pyObjects (type byteArray, unicode...),
    whose inlying char*-pointer is given to a QSharedPointer<char>. Before giving
    it to the shared pointer, the reference of the pyObject is incremented.
    If the deleter of the sharedPointer is called, it does not delete the char-array,
    but decrements the PyObject and deletes it from this hash-table (used in getVal) */
    static QHash<void* /*sharedPointerData*/, PyObject* /*pyObjOwner*/> m_hashTable;

};

} //namespace ito
