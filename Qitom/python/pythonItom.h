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

#ifndef PYTHONITOM_H
#define PYTHONITOM_H

/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
    #define NO_IMPORT_ARRAY
//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#ifdef _DEBUG
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
        //#include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        //#include "../Lib\site-packages\numpy\core\include\numpy\arrayobject.h" //for numpy arrays
    #endif
    #define _DEBUG
#else
    #ifdef linux
        #include "Python.h"
        //#include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        //#include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
#endif
#endif

#include "../global.h"

#include <qhash.h>
#include <qstring.h>

namespace ito 
{

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

    static PyObject* PyPlotImage(PyObject *pSelf, PyObject *pArgs, PyObject *pKwds);
    static PyObject* PyLiveImage(PyObject *pSelf, PyObject *pArgs, PyObject *pKwds);
    //static PyObject* PyLiveLine(PyObject *pSelf, PyObject *pArgs, PyObject *pKwds);
    /*static PyObject* PyCloseFigure(PyObject *pSelf, PyObject *pArgs);
    static PyObject* PySetFigParam(PyObject *pSelf, PyObject *pArgs);
    static PyObject* PyGetFigParam(PyObject *pSelf, PyObject *pArgs);*/

    static PyObject* PyFilter(PyObject *pSelf, PyObject *pArgs, PyObject *kwds);
    static PyObject* PyFilterHelp(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyWidgetHelp(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyPluginHelp(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PyITOMVersion(PyObject* pSelf, PyObject* pArgs);

    static PyObject* PyLoadIDC(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);
    static PyObject* PySaveIDC(PyObject* pSelf, PyObject* pArgs, PyObject *pKwds);

    static PyObject* PyPluginLoaded(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyFilterLoaded(PyObject* pSelf, PyObject* pArgs);

    static PyObject* PySaveDataObject(PyObject* pSelf, PyObject* pArgs);
    static PyObject* PyLoadDataObject(PyObject* pSelf, PyObject* pArgs);

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

    static PyObject* PyGetScreenInfo(PyObject* pSelf);

    static PyObject* getDefaultScaleAbleUnits(PyObject* pSelf);
    static PyObject* ScaleValueAndUnit(PyObject* pSelf, PyObject* pArgs);

    static PyObject* getAppPath(PyObject* pSelf);
    static PyObject* getCurrentPath(PyObject* pSelf);
    static PyObject* setCurrentPath(PyObject* pSelf, PyObject* pArgs);

    static PyObject* setApplicationCursor(PyObject* pSelf, PyObject* pArgs);


protected:
    static QHash<unsigned int, QString> m_gcTrackerList; //!< list with objects currently tracked by python garbage collector.
};

} //end namespace ito
#endif