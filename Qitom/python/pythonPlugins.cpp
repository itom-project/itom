/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
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

#include "pythonPlugins.h"
#include "pythonDataObject.h"
#include "pythonQtConversion.h"
#include "pythonCommon.h"

#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "structmember.h"   //python structmember
#endif
#include "../../AddInManager/addInManager.h"
#include "../AppManagement.h"
#include <qlist.h>
#include <qmap.h>
#include <qobject.h>
#include "pythonQtSignalMapper.h"

#include <qsharedpointer.h>
#include "../../AddInManager/paramHelper.h"
#include "../../common/helperCommon.h"

#include "pythontParamConversion.h"
#include "pythonSharedPointerGuard.h"
#include <qdockwidget.h>
#include <qaction.h>

using namespace ito;


namespace ito
{

//-------------------------------------------------------------------------------------
/** returns the names of the parameters available in a plugin
*   @param [in] aib the plugin for which the parameter names are requested
*   @return     python object with a string list with the parameters' names
*/
PyObject * getParamList(ito::AddInBase *aib)
{
    PyObject *result = NULL;
    QMap<QString, ito::Param> *paramList = NULL;
    const char *name;

    aib->getParamList(&paramList);

    if (paramList)
    {
        result = PyList_New(0);
        QMap<QString, ito::Param>::const_iterator paramIt;
        PyObject *temp = NULL;

        for (paramIt = paramList->constBegin(); paramIt != paramList->constEnd(); ++paramIt)
        {
            name = paramIt.value().getName();
            if (name)
            {
                temp = PyUnicode_DecodeLatin1(name, strlen(name), NULL); //new ref
                PyList_Append(result, temp);
            }
            else
            {
                temp = PyUnicode_FromString("<invalid name>"); //new ref
                PyList_Append(result, temp);
            }
            Py_XDECREF(temp);
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------
/** returns the all information of the parameters available in a plugin
*   @param [in] aib     the plugin for which the parameter names are requested
*   @param [in] args    1 Item-Vector with bool request for additional dictionary return
*   @return     python list of python tuple with the parameters' names, min, max, current value, (infostring)
*/
PyObject * getParamListInfo(ito::AddInBase *aib, PyObject *args)
{
   PyObject *result = NULL;
   int length = PyTuple_Size(args);
   int output = 0;

   QMap<QString, ito::Param> *paramList = NULL;

   aib->getParamList(&paramList);

    if (length == 1)
    {
        if (!PyArg_ParseTuple(args, "i", &output))
        {
            PyErr_SetString(PyExc_ValueError, "wrong input parameter");
            return NULL;
        }
    }
    else if (length > 1)
    {
        PyErr_SetString(PyExc_ValueError, "wrong number of input arguments");
        return NULL;
    }

    if (paramList)
    {
        if (output == 0)
            std::cout << "Plugin parameters are:\n";

        QVector<ito::Param> parameter = paramList->values().toVector();
        result = printOutParams(&parameter, false, true, -1, output == 0);
    }
    else
    {
        result = PyDict_New();

        if (output == 0)
            std::cout << " \nPlugin does not accept parameters! \n";
    }

   //std::cout << "\n";

    if ((length == 0) || (output==0))
    {
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
    else
        return result;

}

//-------------------------------------------------------------------------------------
/** returns a dictionary with all meta information of one parameter
*   @param [in] aib     the plugin for which the parameter names are requested
*   @param [in] args    1 Item-Vector with bool request for additional dictionary return
*   @return     python list of python tuple with the parameters' names, min, max, current value, (infostring)
*/
PyObject * getParamInfo(ito::AddInBase *aib, PyObject *args)
{
    PyObject *result = NULL;
    char *name = nullptr;// = "";

    if (!PyArg_ParseTuple(args, "s", &name))
    {
        return NULL;
    }

    QMap<QString, ito::Param> *paramList = NULL;
    aib->getParamList(&paramList);

    if (paramList->contains(name))
    {
        const ito::Param &p = (*paramList)[name];
        return parseParamMetaAsDict(p.getMeta());
    }
    else
    {
        return PyErr_Format(PyExc_ValueError, "parameter '%s' does not exist", name);
    }
}

//-------------------------------------------------------------------------------------
PyObject* plugin_showConfiguration(ito::AddInBase *aib)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retval;

    if (aib)
    {
        if ((qobject_cast<QApplication*>(QCoreApplication::instance())) && aib->hasConfDialog())
        {
            ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
            ItomSharedSemaphore *sem = locker.getSemaphore();
            QMetaObject::invokeMethod(aim, "showConfigDialog", Q_ARG(ito::AddInBase*, aib), Q_ARG(ItomSharedSemaphore*, sem));

            locker.getSemaphore()->wait(-1);
            retval += locker.getSemaphore()->returnValue;
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("plugin has no configuration dialog").toLatin1().data());
        }
    }

    if (!PythonCommon::setReturnValueMessage(retval, "showConfiguration", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------


/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*
*   This method simply open the widget
*/
PyObject* plugin_showToolbox(ito::AddInBase *aib)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retval;

    if (aib)
    {
        ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

        ItomSharedSemaphore *sem = locker.getSemaphore();
        int mode = 1; // MUST be an lvalue for Q_ARG
        if (QMetaObject::invokeMethod(
            aim,
            "showDockWidget",
            Q_ARG(ito::AddInBase*, aib),
            Q_ARG(int, mode),
            Q_ARG(ItomSharedSemaphore*, sem)))
        {
            if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("timeout while showing toolbox").toLatin1().data());
            }
            else
            {
                retval += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("Member 'showDockWidget' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }
    }
    if (!PythonCommon::setReturnValueMessage(retval, "showToolbox", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*
*   This method simply close the widget
*/
PyObject* plugin_hideToolbox(ito::AddInBase *aib)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retval;

    if (aib)
    {
        ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

        ItomSharedSemaphore *sem = locker.getSemaphore();
        int mode = 0; // MUST be an lvalue for Q_ARG
        if (QMetaObject::invokeMethod(
            aim,
            "showDockWidget",
            Q_ARG(ito::AddInBase*, aib),
            Q_ARG(int, mode),
            Q_ARG(ItomSharedSemaphore*, sem)))
        {
            if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("timeout while hiding toolbox").toLatin1().data());
            }
            else
            {
                retval += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Member 'showDockWidget' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }
    }

    if (!PythonCommon::setReturnValueMessage(retval, "hideToolbox", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}
