/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONCOMMON_H
#define PYTHONCOMMON_H

#ifndef NPY_1_7_API_VERSION
    #define NPY_1_7_API_VERSION 0x00000007
#endif
#ifndef NPY_1_8_API_VERSION
    #define NPY_1_8_API_VERSION 0x00000008
#endif

#ifndef Q_MOC_RUN
    /* includes */
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

#include "../../common/sharedStructures.h"

#include <qvector.h>
#include <qvariant.h>

namespace ito
{

    ito::RetVal checkAndSetParamVal(PyObject *tempObj, ito::Param *param, int *set);
    ito::RetVal checkAndSetParamVal(PyObject *pyObj, const ito::Param *defaultParam, ito::ParamBase &outParam, int *set);
    //!< This function is used to print out parameters to a dictionary and the itom-console
    PyObject* PrntOutParams(const QVector<ito::Param> *params, bool asErr, bool addInfos, const int num, bool printToStdStream = true);
    PyObject *parseParamMetaAsDict(const ito::ParamMeta *meta);
    void errOutInitParams(const QVector<ito::Param> *params, const int num, const char *reason);
    ito::RetVal parseInitParams(const QVector<ito::Param> *defaultParamListMand, const QVector<ito::Param> *defaultParamListOpt, PyObject *args, PyObject *kwds, QVector<ito::ParamBase> &paramListMandOut, QVector<ito::ParamBase> &paramListOptOut);
    ito::RetVal copyParamVector(const QVector<ito::ParamBase> *paramVecIn, QVector<ito::ParamBase> &paramVecOut);
    ito::RetVal copyParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::Param> &paramVecOut);
    ito::RetVal copyParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::ParamBase> &paramVecOut);
    ito::RetVal createEmptyParamBaseFromParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::ParamBase> &paramVecOut);

    //!< This function searches for reserves Keywords (e.g. autoLoadParams) sets the corresponding bool parameter to the right value and than deletes the keyword
    ito::RetVal findAndDeleteReservedInitKeyWords(PyObject *kwds, bool * enableAutoLoadParams);

    PyObject* buildFilterOutputValues(QVector<QVariant> *outVals, ito::RetVal &retValue);

    class PythonCommon
    {
        public:

            enum tErrMsg
            {
                noMsg = 0,
                loadPlugin = 1,
                execFunc = 2,
                invokeFunc = 3,
                getProperty = 4,
                runFunc = 5
            
            };

            static bool transformRetValToPyException(ito::RetVal &retVal, PyObject *exceptionIfError = PyExc_RuntimeError, PyObject *exceptionIfWarning = PyExc_RuntimeWarning);
            static bool setReturnValueMessage(ito::RetVal &retVal, const QString &objName, const tErrMsg &errorMSG, PyObject *exceptionIfError = PyExc_RuntimeError, PyObject *exceptionIfWarning = PyExc_RuntimeWarning);
            static bool setReturnValueMessage(ito::RetVal &retVal, const char *objName, const tErrMsg &errorMSG, PyObject *exceptionIfError = PyExc_RuntimeError, PyObject *exceptionIfWarning = PyExc_RuntimeWarning);
    };

} //end namespace ito

#endif
