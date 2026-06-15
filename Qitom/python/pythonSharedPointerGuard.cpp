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

#include "pythonSharedPointerGuard.h"

#include <QtConcurrent/qtconcurrentrun.h>

namespace ito
{

QHash<void*, PyObject*> PythonSharedPointerGuard::m_hashTable = QHash<void*, PyObject*>();

//----------------------------------------------------------------------------------------------------------------------------------
// the following method is only called by PythonSharedPointerGuard::deleter within a QtConcurrent::run worker thread
void safeDecrefPyObject2(PyObject *obj)
{
    if (PyGILState_Check())
    {
        Py_DECREF(obj);
    }
    else
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_DECREF(obj);
        PyGILState_Release(gstate);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ void PythonSharedPointerGuard::safeDecrefPyObject2Async(PyObject* obj)
{
    //the current thread has no Python GIL. However, it might be
    //that the GIL is currently hold by another thread, which has called
    //the current thread, such that directly waiting for the GIL here might
    //lead to a dead-lock. Therefore, we open a worker thread to finally delete the guarded base object!
    QtConcurrent::run(safeDecrefPyObject2, obj);
}


} //end namespace ito
