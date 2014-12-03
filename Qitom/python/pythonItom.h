/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONITOM_H
#define PYTHONITOM_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
    #define NO_IMPORT_ARRAY

    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (!defined linux)
        #undef _DEBUG
        #include "Python.h"
        #define _DEBUG
    #else
        #include "Python.h"
    #endif
#endif

#include "../global.h"

#include <qhash.h>
#include <qstring.h>

namespace ito 
{

class FuncWeakRef; //forward declaration

class PythonItom
{

public:
    static PyMethodDef PythonMethodItom[];
    static PyModuleDef PythonModuleItom;
    static PyObject* PyInitItom(void);

    //!< methods of module itom
    static PyObject* PyOpenEmptyScriptEditor(PyObject *pSelf, PyObject *pArgs);
    static PyObject* PyNewScript(PyObject *pSelf, PyObject *pArgs);
    static PyObject* PyOpenScript(PyObject *pSelf, PyObject *pArgs);
    static PyObject* PyClearCommandLine(PyObject *pSelf);

    static PyObject* PyPlotImage(PyObject *pSelf, PyObject *pArgs, PyObject *pKwds);
    static PyObject* PyLiveImage(PyObject *pSelf, PyObject *pArgs, PyObject *pKwds);

    static PyObject* PyFilter(PyObject *pSelf, PyObject *pArgs, PyObject *kwds);
    static PyObject* PyFilterHelp(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PyWidgetHelp(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PyPluginHelp(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PyPlotHelp(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PyITOMVersion(PyObject* pSelf, PyObject* pArgs);

    static PyObject* PyLoadIDC(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PySaveIDC(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);

    static PyObject* PyPluginLoaded(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyFilterLoaded(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyPlotLoaded(PyObject* pSelf, PyObject* pArgs);

    static PyObject* PySaveDataObject(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PyLoadDataObject(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);

    static PyObject* PyAddButton(PyObject* pSelf, PyObject* pArgs, PyObject *kwds);
    static PyObject* PyRemoveButton(PyObject* pSelf, PyObject* pArgs);

    static PyObject* PyAddMenu(PyObject* pSelf, PyObject* args, PyObject *kwds);
    static PyObject* PyRemoveMenu(PyObject* pSelf, PyObject* args, PyObject *kwds);

    static PyObject* PyCheckSignals(PyObject* pSelf);
    static PyObject* PyProcessEvents(PyObject* pSelf);

    static PyObject* PySaveMatlabMat(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyLoadMatlabMat(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyMatlabMatDataObjectConverter(PyObject *element);  /*!< returns new reference to element.checks whether element is a npDataObject or dataObject and if so,  */

    static PyObject* PyGetDebugger(PyObject* pSelf);
    static PyObject* PyGCStartTracking(PyObject *pSelf);
    static PyObject* PyGCEndTracking(PyObject *pSelf);
    //static PyObject* PyGetGlobalDict(PyObject *pSelf);

    static PyObject* PyAutoReloader(PyObject* pSelf, PyObject *args, PyObject *kwds);

    static PyObject* PyGetScreenInfo(PyObject* pSelf);

    static PyObject* getDefaultScaleableUnits(PyObject* pSelf);
    static PyObject* scaleValueAndUnit(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);

    static PyObject* getAppPath(PyObject* pSelf);
    static PyObject* getCurrentPath(PyObject* pSelf);
    static PyObject* setCurrentPath(PyObject* pSelf, PyObject* pArgs);

    static PyObject* PyGetPalette(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PySetPalette(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyGetPaletteList(PyObject* pSelf, PyObject* pArgs);

    static PyObject* compressData(PyObject* pSelf, PyObject* pArgs);
    static PyObject* uncompressData(PyObject* pSelf, PyObject* pArgs);

    static PyObject* setApplicationCursor(PyObject* pSelf, PyObject* pArgs);

    static PyObject* userCheckIsAdmin(PyObject* pSelf);
    static PyObject* userCheckIsDeveloper(PyObject* pSelf);
    static PyObject* userCheckIsUser(PyObject* pSelf);
    static PyObject* userGetUserInfo(PyObject* pSelf);

protected:
    static QHash<size_t, QString> m_gcTrackerList; //!< list with objects currently tracked by python garbage collector.

    static ito::FuncWeakRef* hashButtonOrMenuCode(PyObject *code, PyObject *argtuple, ito::RetVal &retval, QString &codeString);
    static ito::RetVal unhashButtonOrMenuCode(const size_t &funcID);
    static ito::RetVal unhashButtonOrMenuCode(const ito::FuncWeakRef *funcWeakRef);
};

} //end namespace ito
#endif