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

#ifndef PYTHONCOMMON_H
#define PYTHONCOMMON_H

#ifndef NPY_1_7_API_VERSION
    #define NPY_1_7_API_VERSION 0x00000007
#endif
#ifndef NPY_1_8_API_VERSION
    #define NPY_1_8_API_VERSION 0x00000008
#endif

#ifndef Q_MOC_RUN
    #include "python/pythonWrapper.h"
#endif

#include "../../common/sharedStructures.h"

#include <qvector.h>
#include <qvariant.h>

namespace ito
{

    ito::RetVal checkAndSetParamVal(PyObject *tempObj, ito::Param *param, int *set);
    ito::RetVal checkAndSetParamVal(PyObject *pyObj, const ito::Param *defaultParam, ito::ParamBase &outParam, int *set);
    //!< This function is used to print out parameters to a dictionary and the itom-console
    PyObject* printOutParams(const QVector<ito::Param> *params, bool asErr, bool addInfos, const int num, bool printToStdStream = true);
    PyObject *parseParamMetaAsDict(const ito::ParamMeta *meta, const ito::Param* param = nullptr);
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

            //!< mode that can be chosen for the code checker
            /*
            The values must only be changed in accordance to the
            argument 'mode' of the method 'check' of the python
            module itomSyntaxCheck.py (itom-packages subfolder).

            @seealso WidgetPropEditorCodeCheckers
            */
            enum CodeCheckerMode
            {
                NoCodeChecker = 0,       //!< no code checker active
                CodeCheckerPyFlakes = 1, //!< syntax error and further static code analysis based on PyFlakes
                CodeCheckerFlake8 = 2,   //!< extended code checks (syntax, style, doc style, complexity...) based on Flake8
                CodeCheckerAuto = 3      //!< chose Flake8 if flake8 is available, else chose PyFlakes if pyflakes is available, else No Checker
            };

            //!< type of a message from any code checker
            enum CodeCheckerMessageType
            {
                TypeInfo = 0,    //!< the assigned message is displayed as information (blue dot)
                TypeWarning = 1, //!< the assigned message is displayed as warning (orange dot)
                TypeError = 2    //!< the assigned message is displayed as error (bug symbol)
            };

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

            //!> checks if a Python exception is currently set and returns ito::retError with the corresponding error message,
            //!> if this is the case (else retOk). The python exception is only cleared, if clearError is true (default)
            //!> The caller of this method must already hold the GIL!
            static ito::RetVal checkForPyExceptions(bool clearError = true);
    };

} //end namespace ito

#endif
