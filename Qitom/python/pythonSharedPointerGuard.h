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

#ifndef PYTHONSHAREDPOINTERGUARD_H
#define PYTHONSHAREDPOINTERGUARD_H

//python
#ifndef Q_MOC_RUN
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #include "Python.h"
        #define _DEBUG
    #else
        #include "Python.h"
    #endif
#endif

#include "../../common/sharedStructures.h"
#include "../global.h"
#include "patchlevel.h"

#include <qhash.h>
#include <qsharedpointer.h>

namespace ito
{

class PythonSharedPointerGuard
{
public:
    static PyObject *tParamToPyObject(ito::ParamBase &param);

    template<typename _Tp> static void deleter(_Tp *sharedPointerData)
    {
        QHash<void*, PyObject*>::iterator i = m_hashTable.find((void*)sharedPointerData);
        if(i != m_hashTable.end())
        {
            if (i.value())
            {
#if (PY_VERSION_HEX >= 0x03040000)
                if (PyGILState_Check())
                {
                    Py_DECREF(i.value());
                }
                else
                {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    Py_DECREF(i.value());
                    PyGILState_Release(gstate);
                }
#else
                //we don't know if we need to acquire the GIL here, or not.
                Py_DECREF(i.value());
#endif
            }

            m_hashTable.erase(i);
        }
    }

    template<typename _Tp> static QSharedPointer<_Tp> createPythonSharedPointer(_Tp *sharedPointerData, PyObject *pyObjOwner)
    {



        Py_XINCREF(pyObjOwner);
        m_hashTable.insert((void*)sharedPointerData, pyObjOwner);
        return QSharedPointer<_Tp>(sharedPointerData, deleter<_Tp>);
    }

private:
    static QHash<void* /*sharedPointerData*/, PyObject* /*pyObjOwner*/> m_hashTable;  //elements in this hash-table are pyObjects (type byteArray, unicode...), whose inlying char*-pointer is given to a QSharedPointer<char>. Before giving it to the shared pointer, the reference of the pyObject is incremented. If the deleter of the sharedPointer is called, it does not delete the char-array, but decrements the PyObject and deletes it from this hash-table (used in getVal)

};

} //namespace ito

#endif
