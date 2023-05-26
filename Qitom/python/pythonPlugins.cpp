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
/** returns the all informations of the parameters available in a plugin
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
            QMetaObject::invokeMethod(aim, "showConfigDialog", Q_ARG(ito::AddInBase*, aib), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

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

        if (QMetaObject::invokeMethod(
            aim,
            "showDockWidget",
            Q_ARG(ito::AddInBase*, aib),
            Q_ARG(int, 1),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
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

        if (QMetaObject::invokeMethod(
            aim,
            "showDockWidget",
            Q_ARG(ito::AddInBase*, aib),
            Q_ARG(int, 0),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
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


//-------------------------------------------------------------------------------------
PyObject* plugin_userMutexLock(ito::AddInBase *aib, PyObject* args, PyObject* kwds, bool &userMutexLocked)
{
    if (aib->getBasePlugin()->getAddInInterfaceVersion() < 0x040200)
    {
        return PyErr_Format(PyExc_RuntimeError, "The plugin must implement the AddInInterface version >= 4.2 to support the user mutex.");
    }

    const char *kwlist[] = { "timeout",  NULL };
    int timeout = 3000;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", const_cast<char**>(kwlist), &timeout))
    {
        return NULL;
    }

    bool r = aib->getUserMutex().tryLock(timeout);

    userMutexLocked = r;

    if (r)
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//-------------------------------------------------------------------------------------
PyObject* plugin_userMutexUnlock(ito::AddInBase *aib, bool &userMutexLocked)
{
    if (aib->getBasePlugin()->getAddInInterfaceVersion() < 0x040200)
    {
        return PyErr_Format(PyExc_RuntimeError, "The plugin must implement the AddInInterface version >= 4.2 to support the user mutex.");
    }

    // unlock will lead to an undefined error if it is not locked, yet.
    // Therefore, try to lock it immediately... such that unlock will always work.
    aib->getUserMutex().tryLock(0);

    aib->getUserMutex().unlock();

    userMutexLocked = false;

    Py_RETURN_NONE;
}



//-------------------------------------------------------------------------------------
/** returns the names of extended Functionality available in a plugin
*   @param [in] aib the plugin for which the parameter names are requested
*   @return     python object with a string list with the execFuncs' names
*/
PyObject* getExecFuncsList(ito::AddInBase *aib)
{
    PyObject *result = PyList_New(0);

    QMap<QString, ExecFuncParams> *funcList = NULL;
    aib->getExecFuncList(&funcList);

    if (funcList && !funcList->isEmpty())
    {
        QMap<QString, ExecFuncParams>::const_iterator fn;

        for (fn = funcList->constBegin(); fn != funcList->constEnd(); fn++)
        {
            PyObject* temp = NULL;
            QByteArray name = fn.key().toLatin1();

            if (name != "")
            {
                temp = PyUnicode_DecodeLatin1(name.constData(), name.length(), NULL); //new ref
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
/** returns a list of execFunction available in a plugin similar to filterHelp
*   @param [in] aib   the plugin for which the execFuncs names are requested
*   @param [in] args  2 Item-Vector with integer request for additional dictionary return
*   @return           python dictionary with list of functions or specific dictionary
*                     for one execFunc with the parameters' names, min, max, current
*                     value, (infostring)
*/
PyObject * getExecFuncsInfo(ito::AddInBase *aib, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"funcName", "detailLevel", NULL};
    char* funcName = NULL;
    int detailLevel = 0; //0: show text in std::cout, 1: return dict or list with items
    PyObject *result = NULL;
    QString funcNameString("");

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|si", const_cast<char**>(kwlist), &funcName, &detailLevel))
    {
        return NULL;
    }

    QMap<QString, ExecFuncParams> *funcList = NULL;
    bool printToStream = (detailLevel != 1);

    aib->getExecFuncList(&funcList);
    result = PyDict_New();

    if (funcList && funcList->size() > 0)
    {
        if (funcName != NULL)
        {
            funcNameString = QString(funcName);
        }

        QStringList execFuncs = funcList->keys();
        PyObject *execFuncslist = NULL;

        if (execFuncs.size() > 0)
        {
            if (!funcNameString.isEmpty() && execFuncs.contains(funcNameString))    // got an exect match
            {
                (*funcList)[funcNameString].infoString;
                (*funcList)[funcNameString].paramsMand;
                (*funcList)[funcNameString].paramsOpt;

                if (printToStream)
                {
                    std::cout << "Parameters\n-----------------\n";
                }

                const QVector<ito::Param> *parameter = &((*funcList)[funcNameString].paramsMand);

                if (parameter->size() > 0)
                {
                    if (printToStream)
                    {
                        std::cout << "\nMandatory parameters:\n";
                    }
                    execFuncslist = printOutParams(parameter, false, true, -1, printToStream);
                    PyDict_SetItemString(result, "Mandatory Parameters", execFuncslist);
                    Py_DECREF(execFuncslist);
                }
                else if (printToStream)
                {
                    std::cout << "\nMandatory parameters:\n- no mandatory parameters -";
                }

                parameter = &((*funcList)[funcNameString].paramsOpt);

                if (parameter->size() > 0)
                {
                    if (printToStream)
                    {
                        std::cout << "\nOptional parameters:\n";
                    }

                    execFuncslist = printOutParams(parameter, false, true, -1, printToStream);
                    PyDict_SetItemString(result, "Optional Parameters", execFuncslist);
                    Py_DECREF(execFuncslist);
                }
                else if (printToStream)
                {
                    std::cout << "\nOptional parameters:\n- no optional parameters -";
                }

                std::cout << "\n";

                parameter = &((*funcList)[funcNameString].paramsOut);

                if (parameter->size())
                {
                    if (printToStream)
                    {
                        std::cout << "\nOutput values:\n";
                    }

                    execFuncslist = printOutParams(parameter, false, true, -1, printToStream);
                    PyDict_SetItemString(result, "Output Parameters", execFuncslist);
                    Py_DECREF(execFuncslist);
                }
                else if (printToStream)
                {
                    std::cout << "\nOutput values:\n- no output parameters -";
                }

            }
            else
            {
                execFuncs.sort();

                if (printToStream)
                {
                    std::cout << "Plugin 'exec' functions are:\n\n";
                }

                QList<QPair<QString, QString> > outPut;
                outPut.clear();
                int longname = 0;

                for (int funcs = 0; funcs < execFuncs.size(); funcs++)
                {
                    if (longname < execFuncs.value(funcs).length())
                        longname = execFuncs.value(funcs).length();

                    outPut.append(QPair<QString, QString>(execFuncs.value(funcs), (*funcList)[execFuncs.value(funcs)].infoString));

                    PyObject *text = PythonQtConversion::QByteArrayToPyUnicodeSecure((*funcList)[execFuncs.value(funcs)].infoString.toLatin1());
                    PyDict_SetItemString(result, execFuncs.value(funcs).toLatin1().data() , text);
                    Py_DECREF(text);
                    text = NULL;
                }

                longname+= 3;

                if (printToStream)
                {
                    std::cout << "No " << QString("Name").leftJustified(longname, ' ', false).toLatin1().data() << "   \tInfostring\n";

                    for (int funcs = 0; funcs < outPut.size(); funcs++)
                    {
                        std::cout << funcs << "  " << outPut.value(funcs).first.leftJustified(longname, ' ', false).toLatin1().data() << "  \t'" << outPut.value(funcs).second.toLatin1().data() << "'\n";
                    }

                    std::cout << "\nUse inst.getExecFuncsInfo('execfuncname') to get detailed information about one 'exec' function.\n";
                }
            }
        }
        else if (printToStream)
        {
            std::cout << " \nPlugin has no additional 'exec' functions. \n";
        }

    }
    else if (printToStream)
    {
        std::cout << " \nPlugin has no additional 'exec' functions. \n";
    }

    if (printToStream)
    {
        Py_DECREF(result);
        Py_RETURN_NONE;
    }
    else
    {
        return result;
    }

}

//-------------------------------------------------------------------------------------
/** returns the name of a python plugin
*   @param [in] addInObj    the plugin whoes name should be returned
*   @return     the plugin name
*/
PyObject* getName(ito::AddInBase *addInObj)
{
    ito::RetVal ret = ito::retOk;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<ito::Param> qsParam(new ito::Param("name", ito::ParamBase::String, "", NULL));

    if (QMetaObject::invokeMethod(
        addInObj,
        "getParam",
        Q_ARG(QSharedPointer<ito::Param>, qsParam),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
    {
        bool timeout = false;

        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!addInObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while getting name parameter").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'getParam' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    if (!PythonCommon::setReturnValueMessage(ret, "getName", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    //return PyUnicode_FromString((*qsParam).getVal<char*>());
    char* val = (*qsParam).getVal<char*>();
    QString val2 = QString("%1 (%2)").arg(val).arg(addInObj->getRefCount());
    QByteArray val2_ = val2.toLatin1();
    const char* val3 = val2_.constData();
    return PyUnicode_DecodeLatin1(val3, strlen(val3), NULL);
}

//-------------------------------------------------------------------------------------

PyObject* execFunc(ito::AddInBase *aib, PyObject *args, PyObject *kwds)
{
    ito::RetVal ret = ito::retOk;
    QMap<QString, ExecFuncParams> *funcList;
    QSharedPointer<QVector<ito::ParamBase> > paramsMand(new QVector<ito::ParamBase>(), PythonPlugins::paramBaseVectorDeleter); //the deleter are important, else it crashes sometimes if the execFunc of the plugin releases the waitCond much earlier (nobody knows why)
    QSharedPointer<QVector<ito::ParamBase> > paramsOpt(new QVector<ito::ParamBase>(), PythonPlugins::paramBaseVectorDeleter); //the deleter are important, else it crashes sometimes if the execFunc of the plugin releases the waitCond much earlier (nobody knows why)
    QSharedPointer<QVector<ito::ParamBase> > paramsOut(new QVector<ito::ParamBase>(), PythonPlugins::paramBaseVectorDeleter); //the deleter are important, else it crashes sometimes if the execFunc of the plugin releases the waitCond much earlier (nobody knows why)
    QString name;
    int argsLength = PyTuple_Size(args);
    PyObject *pyObj = NULL;
    bool ok;

    if (argsLength < 1)
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("you must provide at least one parameter with the name of the function").toLatin1().data());
    }
    else
    {
        pyObj = PyTuple_GET_ITEM(args,0); //borrowed
        name = PythonQtConversion::PyObjGetString(pyObj,true,ok);
        if (!ok)
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("the first function name parameter can not be interpreted as string").toLatin1().data());
        }
    }

    if (!ret.containsError())
    {
        ret += aib->getExecFuncList(&funcList);
        QMap<QString, ExecFuncParams>::const_iterator it = funcList->constFind(name);

        if (it == funcList->constEnd())
        {
            ret += ito::RetVal::format(ito::retError, 0, QObject::tr("plugin does not provide an execution of function '%s'").toLatin1().data(),name.toLatin1().data());
        }
        else
        {
            //split first argument from args
            pyObj = PyTuple_GetSlice(args, 1, argsLength); //new ref

            //parses python-parameters with respect to the default values given py (*it).paramsMand and (*it).paramsOpt and returns default-initialized ParamBase-Vectors paramsMand and paramsOpt.
            ret += parseInitParams(&(it->paramsMand), &(it->paramsOpt), pyObj, kwds, *paramsMand, *paramsOpt);

            //makes deep copy from default-output parameters (*it).paramsOut and returns it in paramsOut (ParamBase-Vector)
            ret += copyParamVector(&(it->paramsOut), *paramsOut);

            Py_XDECREF(pyObj);

            if (!ret.containsError())
            {
                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                if (QMetaObject::invokeMethod(aib, "execFunc", Q_ARG(QString, name), Q_ARG(QSharedPointer<QVector<ito::ParamBase> >, paramsMand), Q_ARG(QSharedPointer<QVector<ito::ParamBase> >, paramsOpt), Q_ARG(QSharedPointer<QVector<ito::ParamBase> >, paramsOut), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
                {
                    bool timeout = false;

                    while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
                    {
                        if (!aib->isAlive())
                        {
                            ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling specific function in plugin.").toLatin1().data());
                            timeout = true;
                            break;
                        }
                    }

                    if (!timeout)
                    {
                        ret += locker.getSemaphore()->returnValue;
                    }
                }
                else
                {
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'execFunc' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
                }
            }

        }
    }

    QByteArray name_ba = name.toLatin1();
    if (!PythonCommon::setReturnValueMessage(ret, name_ba.data(), PythonCommon::execFunc))
    {
        return NULL;
    }
    else
    {
        if (paramsOut->size() == 0)
        {
            Py_RETURN_NONE;
        }
        else if (paramsOut->size() == 1)
        {
            PyObject* out = PythonParamConversion::ParamBaseToPyObject((*paramsOut)[0]); //new ref
            if (!PythonCommon::setReturnValueMessage(ret, name_ba.data(), PythonCommon::execFunc))
            {
                return NULL;
            }
            else
            {
                return out;
            }
        }
        else
        {
            //parse output vector to PyObject-Tuple
            PyObject* out = PyTuple_New(paramsOut->size());
            PyObject* temp;
            Py_ssize_t i = 0;

            foreach(const ito::ParamBase &p, *paramsOut)
            {
                temp = PythonParamConversion::ParamBaseToPyObject(p); //new ref
                if (temp)
                {
                    PyTuple_SetItem(out,i,temp); //steals ref
                    i++;
                }
                else
                {
                    break;
                }
            }
            if (!PythonCommon::setReturnValueMessage(ret, name_ba.data(), PythonCommon::execFunc))
            {
                Py_DECREF(out);
                return NULL;
            }
            else
            {
                return out;
            }
        }
    }


}

//-------------------------------------------------------------------------------------
/** return a parameter value
*   @param [in] addInObj    the addIn whoes parameter is requested
*   @param [in] args        the parameter name
*   @return     python object with the parameter value on success (parameter exists), NULL otherwise
*
*   The function tries to retrieve the value of the parameter with the name given in args. If the parameter does not exist
*   NULL is returned. To actually retrieve the value the getParam function of the plugin is invoked.
*/
PyObject* getParam(ito::AddInBase *addInObj, PyObject *args)
{
    PyObject *result = NULL;
    const char *paramName = NULL;
    //bool paramNameCheck;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    ito::RetVal ret = ito::retOk;

    if (!PyArg_ParseTuple(args, "s", &paramName))
    {
        PyErr_SetString(PyExc_ValueError, "no parameter name specified");
        return NULL;
    }

    //check parameter name and split it into its components
    bool hasIndex;
    QString nameOnly;
    int index;
    QString additionalTag;
    if(ito::parseParamName(paramName, nameOnly, hasIndex, index, additionalTag).containsError())
    {
        PyErr_SetString(PyExc_TypeError, "parameter name is invalid. It must have the following format: paramName['['index']'][:additionalTag]");
        return NULL;
    }

    //now get pointer to the parameter-map from plugin and check whether paramName is available
    QMap<QString, Param> *params;
    (addInObj)->getParamList(&params); //always returns ok

    //find parameter in params
    auto it = params->constFind(nameOnly);

    if (it == params->constEnd())
    {
        PyErr_Format(PyExc_ValueError, "Parameter '%s' not contained in plugin.", nameOnly.toLatin1().data());
        return NULL;
    }

    //create a container for the returned parameter. This value is initialized by the full name including the type of the corresponding parameter of the m_params map.
    //Usually, this type is correct, such that setVal can directly used within the plugin. However, the plugin is also allowed to change the type.
    QSharedPointer<ito::Param> qsParam(new ito::Param(paramName, it->getType() | it->getFlags()));

    if (QMetaObject::invokeMethod(addInObj, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
    {
        bool timeout = false;
        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!addInObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while getting parameter").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'getParam' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    result = ito::PythonParamConversion::ParamBaseToPyObject(*qsParam);

    if (!PythonCommon::setReturnValueMessage(ret, "getParam", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    return result;
}

//-------------------------------------------------------------------------------------
//general docstrings BEGIN
PyDoc_STRVAR(pyPluginName_doc, "name() -> str \n\
\n\
Returns the name of this plugin object.\n\
\n\
Returns \n\
------- \n\
name : str \n\
    name of the plugin, which corresponds to ``getParam(\"name\")`` \n\
\n\
See Also \n\
-------- \n\
getParam");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginGetExecFuncsList_doc, "getExecFuncsList() -> List[str] \n\
\n\
Gets a list of the names of additional callable functions of this plugin.\n\
\n\
Each plugin may define a set of functions, extending the standard interface. \n\
These functions are not common to plugins of the same type. They are \n\
executed using:: \n\
    \n\
    instance.exec(\"funcname\", arg1, arg2, ...)\n\
\n\
To get more information about one specific function, call \n\
:meth:`getExecFuncsInfo`. \n\
\n\
Returns \n\
------- \n\
list of str \n\
    is a list of additional, callable function names of this plugin object.");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginGetParamList_doc, "getParamList() -> List[str] \n\
\n\
Returns a list of the names of all available parameters of this plugin object.\n\
\n\
Each plugin defines a set of parameters. Each of these parameters maps its ``name`` \n\
to a certain ``value``. The value is represented by the C++ class \n\
:class:`ito::ParamBase` and can have one of the following types \n\
(Python equivalent in brackets): \n\
\n\
* String (str) \n\
* Char (int, [-127, 128]) \n\
* Integer, (int) \n\
* Double (float) \n\
* CharArray (sequence of int) \n\
* IntegerArray (sequence of int) \n\
* DoubleArray (sequence of float) \n\
* DataObject (:class:`dataObject`) \n\
* PolygonMesh (:class:`polygonMesh`) \n\
* PointCloud (:class:`pointCloud`) \n\
* Another plugin instance (:class:`dataIO` or :class:`actuator`) \n\
\n\
Using one of the parameter names, its current value can be obtained by \n\
``getParam(\"name\")`` and can be set by ``setParam(\"name\", newValue)`` \n\
(if not read-only). \n\
\n\
Usually, every plugin object can define its own set of parameters. However, there are \n\
conventions about certain parameters, that must be available and have a specific \n\
meaning for a type of plugin object. \n\
\n\
Returns \n\
------- \n\
list of str \n\
    list of available parameter names in this plugin. \n\
\n\
See Also \n\
-------- \n\
getParam, setParam, getParamListInfo");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginGetParamListInfo_doc, "getParamListInfo(detailLevel = 1) -> Optional[dict] \n\
\n\
Prints or returns detailed information about all parameters of this plugin object. \n\
\n\
Each plugin defines a set of parameters. Each of these parameters maps its ``name`` \n\
to a certain ``value``. The value is represented by the C++ class \n\
:class:`ito::ParamBase` and can have one of the following types \n\
(Python equivalent in brackets): \n\
\n\
* String (str) \n\
* Char (int, [-127, 128]) \n\
* Integer, (int) \n\
* Double (float) \n\
* CharArray (sequence of int) \n\
* IntegerArray (sequence of int) \n\
* DoubleArray (sequence of float) \n\
* DataObject (:class:`dataObject`) \n\
* PolygonMesh (:class:`polygonMesh`) \n\
* PointCloud (:class:`pointCloud`) \n\
* Another plugin instance (:class:`dataIO` or :class:`actuator`) \n\
\n\
Using one of the parameter names, its current value can be obtained by \n\
``getParam(\"name\")`` and can be set by ``setParam(\"name\", newValue)`` \n\
(if not read-only). \n\
\n\
This method prints a detailed listing with the `name`, `current value`, \n\
`description string` and further `meta information` of every plugin parameter. \n\
Additionally, the column ``R/W`` indicates if this parameter is writable or read-only. \n\
\n\
Dependin`g on ``detailLevel``, this method will not print the listing to the command line \n\
but returns it using a nested :class:`dict`. \n\
\n\
Parameters \n\
---------- \n\
detailLevel : dict, optional \n\
    if ``detailLevel`` is set to ``1``, this method returns a nested dictionary with all \n\
    information about all parameters of this plugin. Otherwise ``None`` is returned and \n\
    the listing is printed in a readable form to the command line (default). \n\
\n\
Returns \n\
------- \n\
None or dict \n\
    See the parameter ``detailLevel`` for the difference in returned values. \n\
\n\
See Also \n\
-------- \n\
getParam, setParam, getParamInfo, getParamList, getParamInfo");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginGetParamInfo_doc, "getParamInfo(name) -> dict \n\
\n\
Returns a nested dictionary with meta information of the desired parameter. \n\
\n\
Plugin parameters in itom not only hold a value, but they can also be equipped \n\
with further meta information, like the minimum or maximum value range, a certain \n\
step size, allowed string values etc. \n\
\n\
These values are returned as nested dictionary (if available, else the dict is \n\
more or less empty). \n\
\n\
Parameters \n\
---------- \n\
name : str \n\
    Name of the plugin parameter. \n\
\n\
Returns \n\
------- \n\
dict \n\
    nested dictionary with meta information assigned to the plugin parameter ``name``.");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginGetParam_doc, "getParam(name) -> Union[int, float, str, Tuple[int], Tuple[float], dataObject, polygonMesh, pointCloud, dataIO, actuator] \n\
\n\
Returns the current value of the plugin parameter ``name``. \n\
\n\
The type of the returned value depends on the real type of the internal plugin, \n\
which can be: \n\
\n\
* String -> :obj:`str` \n\
* Char, Integer -> :obj:`int` \n\
* Double -> :obj:`float` \n\
* CharArray, IntegerArray -> :obj:`tuple` of :obj:`int` \n\
* DoubleArray -> :obj:`tuple` of :obj:`float` \n\
* DataObject -> :class:`dataObject` \n\
* PolygonMesh -> :class:`polygonMesh` \n\
* PointCloud -> :class:`pointCloud` \n\
* Another plugin instance -> :class:`dataIO` or :class:`actuator` \n\
\n\
The ``name`` of the parameter must have the following form: \n\
\n\
* **name** \n\
* **name:additionalTag** (``additionalTag`` can be a special feature of some plugins) \n\
* **name[index]** (only possible if parameter is an array type and you only want to get \n\
  one single value, specified by the integer index ``[0, len(array) - 1]``) \n\
* **name[index]:additionalTag** (a combination of the two possibilies above) \n\
\n\
Parameters \n\
---------- \n\
name : str\n\
    Name of the requested parameter.\n\
\n\
Returns \n\
------- \n\
int or float or str or tuple of int or tuple of float or dataObject or polygonMesh or pointCloud or dataIO or actuator\n\
    Current value of the parameter ``name``. \n\
\n\
Raises \n\
------ \n\
ValueError \n\
    if parameter does not exist \n\
\n\
See Also \n\
-------- \n\
setParam, getParamList, getParamListInfo");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginSetParam_doc, "setParam(name, value) \n\
\n\
Sets a writeable parameter ``name`` of this plugin object to ``value``. \n\
\n\
Sets the internal plugin parameter with 'name' to a new value. The plugin itsself \n\
can decide whether the given value is accepted as new value. This may depend on the \n\
type of the given value, but also on the allowed value range indicated by further \n\
meta information of the internal parameter. Parameters that are (currently) set to \n\
read-only cannot be set. \n\
\n\
The ``name`` of the parameter must have the following form: \n\
\n\
* **name** \n\
* **name:additionalTag** (additionalTag can be a special feature of some plugins) \n\
* **name[index]** (only possible if parameter is an array type and you only want to get \n\
  one single value, specified by the integer index [0,nrOfArrayItems-1]) \n\
* **name[index]:additionalTag** (a combination of the two possibilies above) \n\
\n\
Parameters \n\
---------- \n\
name : str\n\
    Name of the parameter. \n\
value : int or float or str or tuple of int or tuple of float or dataObject or polygonMesh or pointCloud or dataIO or actuator\n\
    The ``value`` that will be set. The plugin will check if this ``value`` fits \n\
    to possible constraints, given by the parameters's meta information or further \n\
    limitations. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the new ``value`` is (currently) not accepted. \n\
\n\
See Also \n\
-------- \n\
getParam, getParamList, getParamListInfo");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginShowToolbox_doc, "showToolbox() \n\
\n\
Opens the (optional) toolbox of this plugin object in the itom main window. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if this plugin does not provide a toolbox. \n\
\n\
See Also \n\
-------- \n\
hideToolbox");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginHideToolbox_doc, "hideToolbox() \n\
\n\
Hides the visible toolbox of this plugin object. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if this plugin does not provide a toolbox. \n\
\n\
See Also \n\
-------- \n\
showToolbox");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPluginShowConfiguration_doc, "showConfiguration() \n\
\n\
Shows the (optional) configuration dialog of this plugin as modal dialog. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if this plugin does not provide a configuration dialog.");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPlugInGetExecFuncsInfo_doc, "getExecFuncsInfo(funcName = \"\", detailLevel = 0) -> Optional[dict] \n\
\n\
Lists all available additional functions of this plugin or gives a detailed description of one specific ``funcName``. \n\
\n\
Every plugin can define further functions, that are called by the method :meth:`exec`. \n\
This can for instance be used in order to call specific calibration routines of \n\
cameras or actuators. \n\
\n\
This method either prints requested information in a readable form to the command line \n\
or returns this information as nested dictionary. \n\
\n\
Parameters \n\
---------- \n\
funcName : str, optional \n\
    is the fullname or a part of any name of such an additional plugin function. \n\
    If ``funcName`` is an empty string or does not match any plugin function \n\
    (case sensitive), a list of all suitable additional plugin function names is given. \n\
    Else, detailed information about the desired ``funcName`` is given, like its description \n\
    or (optional) arguments that are needed to execute this function. \n\
detailLevel : dict, optional \n\
    if ``detailLevel == 1``, this returns a nested dictionary with detailed information, else \n\
    it is printed to the command line in a readable form (default). \n\
\n\
Returns \n\
------- \n\
None or dict\n\
    The return value depends on the argument ``detailLevel``. \n\
\n\
See Also \n\
-------- \n\
exec");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyPlugin_execFunc_doc, "exec(funcName, *args, **kwds) -> Union[Any, Tuple[Any]] \n\
\n\
Calls the additional function ``funcName`` of this plugin. \n\
\n\
Every plugin can define special, additional functions (denoted as ``exec functions``) \n\
that can for instance be used in order to call specific calibration routines \n\
of cameras or actuators. This generic method is used to call one of these specific \n\
functions, that has to be registered in the plugin under the name ``funcName``. \n\
\n\
Every function can define a set of mandatory and / or optional parameters. See \n\
:meth:`getExecFuncsInfo` or the plugin help viewer of itom for more information. \n\
Pass the mandatory and optional parameters as arguments ``param1``, ``param2`` ... to \n\
this method. \n\
\n\
Additionally, every function can return one or multiple values. Either the single value \n\
or a tuple of all returned values is returned by this method. \n\
\n\
Parameters \n\
---------- \n\
funcName : str \n\
    The name of the additional function.\n\
*args : Any \n\
    Further positional arguments, that are assigned first to all mandatory parameters, \n\
    followed by the optional ones. The mandatory or optional parameters of the called \n\
    function can also given as keyword arguments (see ``**kwds``). \n\
**kwds : Any, optional \n\
    Keyword-based arguments, see ``*args`` above. \n\
\n\
Returns \n\
------- \n\
any or tuple of any \n\
    The returned values depend on the function itself.\n\
\n\
See Also \n\
-------- \n\
getExecFuncsInfo");


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyPlugin_userMutex_tryLock_doc, "tryLock(timeout = 3000) -> bool \n\
\n\
Tries to lock the user mutex of this plugin. \n\
\n\
Every plugin contains a user mutex, that can be used for arbitrary purposes. \n\
It is not used for any official purposes. You can for instance use this mutex \n\
both from Python and other C++ threads to protect a series of calls to this \n\
plugin to not to be interrupted by other participants. However, it is the \n\
full responsibility of the programmer to carefully use this mutex. \n\
\n\
Please be careful, that this method can lead to a deadlock if ``timeout`` is \n\
set to a negative value (inifinite wait) and if the mutex is not released \n\
by any other thread. Hint: A Python thread is no `real` thread, it must be \n\
a real C++ thread. If you want to use the mutex within two or more Python \n\
threads, it is recommended, to call this method with a defined ``timeout`` \n\
within a loop and wait for this method to return ``True``. This allows \n\
unlocking the mutex by another thread. \n\
\n\
This method is new for plugins that implement the AddInInterface >= 4.2. \n\
\n\
Parameters \n\
---------- \n\
timeout : int \n\
    This method will wait for at most ``timeout`` milliseconds for the \n\
    mutex to become available. If this value is negative, it will wait forever \n\
    until the mutex become available.\n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if the user defined mutex could be locked, else ``False``. \n\
\n\
See Also \n\
-------- \n\
unlock");

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyPlugin_userMutex_unlock_doc, "unlock() \n\
\n\
Tries to unlock the user mutex of this plugin. \n\
\n\
Every plugin contains a user mutex, that can be used for arbitrary purposes. \n\
It is not used for any official purposes. You can for instance use this mutex \n\
both from Python and other C++ threads to protect a series of calls to this \n\
plugin to not to be interrupted by other participants. However, it is the \n\
full responsibility of the programmer to carefully use this mutex. \n\
fu\n\
This method is new for plugins that implement the AddInInterface >= 4.2. \n\
\n\
See Also \n\
-------- \n\
lock");
//general docstrings END



//-------------------------------------------------------------------------------------
/** set a parameter value
*   @param [in] addInObj    the addIn whoes parameter is requested
*   @param [in] args        the parameter name and value in a python object
*   @return     Py_Return_None on success, NULL otherwise
*
*   The function tries to set the value of the parameter with the name given in args. If the parameter does not exist
*   or is incompatible with the value passed, NULL is returned. To actually set the value the setParam function of the plugin is invoked.
*/
PyObject* setParam(ito::AddInBase *addInObj, PyObject *args)
{
    const char *key = NULL;
    ItomSharedSemaphore *waitCond = NULL;
    ito::RetVal ret = ito::retOk;
    PyObject *value = NULL;

    QSharedPointer<ito::ParamBase> qsParam;

    if(!PyArg_ParseTuple(args, "sO", &key, &value))
    {
        PyErr_SetString(PyExc_ValueError, "Parameter name and its value required.");
        return NULL;
    }

    //check parameter name and split it into its components
    bool hasIndex;
    QString paramName;
    int index;
    QString additionalTag;
    if(ito::parseParamName(key, paramName, hasIndex, index, additionalTag).containsError())
    {
        PyErr_SetString(PyExc_TypeError, "parameter name is invalid. It must have the following format: paramName['['index']'][:additionalTag]");
        return NULL;
    }

    //now get pointer to the parameter-map from plugin and check whether paramName is available
    QMap<QString, Param> *params;
    QMap<QString, Param>::iterator it;
    addInObj->getParamList(&params); //always returns ok

    //find parameter in params
    it = params->find(paramName);
    if (it == params->end())
    {
        PyErr_Format(PyExc_ValueError, "Parameter '%s' not contained in plugin.", paramName.toLatin1().data());
        return NULL;
    }

    if(hasIndex)
    {
        switch(it->getType())
        {
        case ito::ParamBase::CharArray:
            qsParam = PythonParamConversion::PyObjectToParamBase(value, key, ret, ito::ParamBase::Char, false);
            break;
        case ito::ParamBase::IntArray:
            qsParam = PythonParamConversion::PyObjectToParamBase(value, key, ret, ito::ParamBase::Int, false);
            break;
        case ito::ParamBase::DoubleArray:
            qsParam = PythonParamConversion::PyObjectToParamBase(value, key, ret, ito::ParamBase::Double, false);
            break;
        case ito::ParamBase::ComplexArray:
            qsParam = PythonParamConversion::PyObjectToParamBase(value, key, ret, ito::ParamBase::Complex, false);
            break;
        case ito::ParamBase::StringList:
            qsParam = PythonParamConversion::PyObjectToParamBase(value, key, ret, ito::ParamBase::String, false);
            break;
        default:
            PyErr_Format(PyExc_ValueError, "Parameter '%s' of plugin is no array.", paramName.toLatin1().data());
            return NULL;
        }
    }
    else
    {
        qsParam = PythonParamConversion::PyObjectToParamBase(value, key, ret, it->getType(), false);
    }

    if(ret.containsError())
    {
        PyErr_Format(PyExc_ValueError, "The given value could not be transformed to the type of parameter.", paramName.toLatin1().data());
        return NULL;
    }
    else
    {
        bool timeout = false;
        waitCond = new ItomSharedSemaphore();
        if (QMetaObject::invokeMethod(addInObj, "setParam", Q_ARG(QSharedPointer<ito::ParamBase>, qsParam), Q_ARG(ItomSharedSemaphore*, waitCond)))
        {

            while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!addInObj->isAlive())
                {
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout.").toLatin1().data());
                    timeout = true;
                    break;
                }
            }

            if (!timeout)
            {
                ret += waitCond->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'setParam' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

         waitCond->deleteSemaphore();
         waitCond = NULL;
    }

    if (!PythonCommon::setReturnValueMessage(ret, "setParam", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------------
/** desctructor for actuator object in python
*   @param [in] self
*
*   Destructs an actuator object (plugin), i.e. deletes the according python variable and invokes
*   the closeAddIn function. The object itself is only deleted if the object's reference
*   counter is zero.
*/
void PythonPlugins::PyActuatorPlugin_dealloc(PyActuatorPlugin* self)
{
    if (self->weakreflist != NULL)
    {
        PyObject_ClearWeakRefs((PyObject *)self);
    }

    if (self->actuatorObj)
    {
        ito::AddInInterfaceBase *aib = self->actuatorObj->getBasePlugin();
        if (!aib)
        {
            std::cerr << "error closing plugin" << std::endl;
            //PyErr_Format(PyExc_RuntimeError, "error closing plugin");
        }
        else
        {
            ito::RetVal retval(ito::retOk);
            ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

            ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

            if (QMetaObject::invokeMethod(aim, "closeAddIn", Q_ARG(ito::AddInBase*, (ito::AddInBase*)self->actuatorObj), Q_ARG(ItomSharedSemaphore*, waitCond)))
            {
                waitCond->wait(-1);
                retval += waitCond->returnValue;
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Member 'closeAddIn' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
            }

            waitCond->deleteSemaphore();
            waitCond = NULL;

            PythonCommon::transformRetValToPyException(retval);
        }
    }

    DELETE_AND_SET_NULL(self->signalMapper);

    if (self->userMutexLocked)
    {
        // unlock the user mutex (if it has not been locked by Python yet, this
        // will be temporarily done in the unlock method)
        PyActuatorPlugin_userMutex_unlock(self);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//-------------------------------------------------------------------------------------
/** constructor for actuator object in python
*   @param [in] type
*   @return     new python actuator object
*
*   Creates a new pythonActuator object. The actual actuator object (itom) is only created later.
*/
PyObject* PythonPlugins::PyActuatorPlugin_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
   PyActuatorPlugin *self = NULL;

   self = (PyActuatorPlugin *)type->tp_alloc(type, 0);
   if (self != NULL)
   {
      self->actuatorObj = NULL;
      self->base = NULL;
      self->weakreflist = NULL;
      self->signalMapper = new PythonQtSignalMapper();
   }

   return (PyObject *)self;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorInit_doc, "actuator(name, *args, **kwds) -> actuator \n\
\n\
Creates a new instance of the actuator plugin ``name``. \n\
\n\
This is the constructor for an :class:`actuator` plugin. It initialises an new \n\
instance of the plugin with the given ``name``. The initialisation parameters are \n\
parsed and unnamed parameters are used in their incoming order to fill first \n\
mandatory parameters and afterwards optional parameters. Parameters may be passed \n\
with their name as keyword, too. However, as usual, no positional parameters are \n\
allowed after a keyword-based one.\n\
\n\
See :meth:`pluginHelp` for detailed information about the specific initialisation \n\
parameters.\n\
\n\
Parameters \n\
---------- \n\
name : str \n\
    is the fullname (case sensitive) of an :class:`actuator`-plugin. \n\
*args : Any \n\
    Every ``actuator`` plugin defines a list of mandatory and optional initialization \n\
    parameters. Pass these arguments either as positional (``*args``) or keyword \n\
    based (``**kwds``) arguments, where the mandatory parameters must be given first, \n\
    followed by the optional ones. Not every optional initialization argument must be \n\
    given, else its default value is used. \n\
**kwds : Any \n\
    Further keyword based parameters. See also ``*args``. \n\
\n\
Returns \n\
------- \n\
actuator \n\
    new instance of the desired actuator plugin.");

/** constructor for actuator object (plugin) accessible from python
*   @param [in] self    the according pythonActuator object
*   @param [in] args    unnamed arguments passed to the constructor in python
*   @param [in] kwds    keyword parameters passed to the constructor in pyhton
*   @return             -1 in case an error occured, else 0
*
*   At first the list of available plugins is searched whether the plugin can be found (by name). If it was found
*   the plugin's manadtory ans optional initialization parameters are retrieved and a parameter check is done. In
*   case everything went right a new instance of a plugin is created with the parameters passed to the constructor.
*/
int PythonPlugins::PyActuatorPlugin_init(PyActuatorPlugin *self, PyObject *args, PyObject *kwds)
{
    self->actuatorObj = NULL;

    if (args == NULL) //args is only NULL, instance of actuator is created by a c-code fragment. Then the content of the type-struct has to be filled by the c-code, too.
    {
        return 0;
    }

    int length = PyTuple_Size(args);

    if (length == 0)
    {
        PyErr_SetString(PyExc_ValueError, "no plugin specified");
        return -1;
    }
    else if (length == 1) //!< copy constructor or name only
    {
        PyActuatorPlugin* copyPlugin = NULL;

        if (PyArg_ParseTuple(args, "O!", &PyActuatorPluginType, &copyPlugin))
        {
            //try to increment reference of copyPlugin->actuatorObj
            if (copyPlugin->actuatorObj)
            {
                copyPlugin->actuatorObj->getBasePlugin()->incRef(copyPlugin->actuatorObj);
            }

            self->actuatorObj = copyPlugin->actuatorObj;
            self->base = copyPlugin->base;
            return 0;
        }
    }

    PyErr_Clear();

    QVector<ito::Param> *paramsMand = NULL;
    QVector<ito::Param> *paramsOpt = NULL;
    ito::RetVal retval = ito::retOk;
    int pluginNum = -1;
    PyObject *pnameObj = NULL;
    PyObject *params = NULL;
    QString pluginName = NULL;

    QVector<ito::ParamBase> paramsMandCpy;
    QVector<ito::ParamBase> paramsOptCpy;

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, "no addin-manager found");
        return -1;
    }

    pnameObj = PyTuple_GetItem(args, 0);

    if (PyUnicode_Check(pnameObj))
    {
        bool ok = false;
        pluginName = PythonQtConversion::PyObjGetString(pnameObj,false,ok);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "invalid parameters");
        return -1;
    }


    retval = AIM->getInitParams(pluginName, ito::typeActuator, &pluginNum, paramsMand, paramsOpt);

    if (retval.containsWarningOrError())
    {
        PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin);

        return -1;
    }

    bool enableAutoLoadParams = false;
    retval = findAndDeleteReservedInitKeyWords(kwds, &enableAutoLoadParams);

    if (retval.containsWarningOrError())
    {
        PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin);

        return -1;
    }
    else
    {
        params = PyTuple_GetSlice(args, 1, PyTuple_Size(args));

        if (parseInitParams(paramsMand, paramsOpt, params, kwds, paramsMandCpy, paramsOptCpy) != ito::retOk)
        {
            Py_XDECREF(params);
            PyErr_SetString(PyExc_RuntimeError, "error while parsing parameters.");
            return -1;
        }
        Py_XDECREF(params);

        ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
        if (QMetaObject::invokeMethod(AIM, "initAddIn", Q_ARG(int, pluginNum), Q_ARG(QString, pluginName), Q_ARG(ito::AddInActuator**, &self->actuatorObj), Q_ARG(QVector<ito::ParamBase>*, &paramsMandCpy), Q_ARG(QVector<ito::ParamBase>*, &paramsOptCpy), Q_ARG(bool, enableAutoLoadParams), Q_ARG(ItomSharedSemaphore*, waitCond)))
        {
            waitCond->wait(-1);
            retval += waitCond->returnValue;
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Member 'initAddIn' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

        waitCond->deleteSemaphore();
        waitCond = NULL;

        paramsMandCpy.clear();
        paramsOptCpy.clear();
    }

    if (!PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin))
    {
        return -1;
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyActuatorPlugin_repr(PyActuatorPlugin *self)
{
    PyObject *result;

    if (self->actuatorObj == NULL)
    {
        result = PyUnicode_FromFormat("empty actuator plugin");
    }
    else
    {
        PyObject *tempObj = NULL;

        if ((tempObj = getName(self->actuatorObj)) != NULL)
        {
            QString ident = self->actuatorObj->getIdentifier();

            if(ident != "")
            {
                result = PyUnicode_FromFormat("Actuator-Plugin(%U, %s, ID: %i)", tempObj, ident.toLatin1().data(), self->actuatorObj->getID());
            }
            else
            {
                result = PyUnicode_FromFormat("Actuator-Plugin(%U, ID: %i)", tempObj, self->actuatorObj->getID());
            }

            Py_DECREF(tempObj);
        }
        else
        {
            result = NULL;
        }
    }
    return result;
}

//-------------------------------------------------------------------------------------
PyMemberDef PythonPlugins::PyActuatorPlugin_members[] = {
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
//PyDoc_STRVAR(pyActuatorName_doc, -> see pyPluginName_doc);
/** Returns the plugin's name
*   @param [in] self    the plugin object
*   @return             the name of the plugin
*
*                       Queries the name of a plugin by invoking a getParam on the plugin for the name parameter
*/
PyObject* PythonPlugins::PyActuatorPlugin_name(PyActuatorPlugin* self)
{
    return getName(self->actuatorObj);
}

//-------------------------------------------------------------------------------------
/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*   @return             a string with all available parameters for this actuator
*
*   All parameters of the plugin are shown. This can be useful as there are only few standard parameters
*   for an actuator. The majority is depending on the actual hardware and accordingly is different for each
*   plugin.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getParamList(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return getParamList(aib);
}

//-------------------------------------------------------------------------------------
/** returns the list of available parameters and additional information about the plugin
*   @param [in] self    the actuator object (python)
*   @return             a string with all available parameters for this actuator
*
*   All parameters of the plugin are shown with additional information as min, max and infostring.
*   This can be useful as there are only few standard parameters for an actuator. The majority is
*   depending on the actual hardware and accordingly is different for each plugin.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getParamListInfo(PyActuatorPlugin* self, PyObject *args)
{
    ito::AddInBase *aib = self->actuatorObj;
    return getParamListInfo(aib, args);
}

/** returns the list of available ExecFunctions' names
*   @param [in] self    the actuator object (python)
*   @return             a list with all available parameters for this actuator
*
*   All ExecFunctions of the plugin are shown or one specific ExecFunctions with additional information as min, max and infostring is shown.
*   This can be useful as there are only few standard parameters for an actuator. The majority is
*   depending on the actual hardware and accordingly is different for each plugin.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getExecFuncsList(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return getExecFuncsList(aib);
}


/** returns the list of available parameters and additional information about the plugin ExecFunctions
*   @param [in] self    the actuator object (python)
*   @return             a dictionary with all available parameters for this actuator
*
*   All ExecFunctions of the plugin are shown or one specific ExecFunctions with additional information as min, max and infostring is shown.
*   This can be useful as there are only few standard parameters for an actuator. The majority is
*   depending on the actual hardware and accordingly is different for each plugin.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getExecFuncsInfo(PyActuatorPlugin* self, PyObject *args, PyObject *kwds)
{
    ito::AddInBase *aib = self->actuatorObj;
    return getExecFuncsInfo(aib, args, kwds);
}

//-------------------------------------------------------------------------------------


/** gets a parameter value
*   @param [in] self    the actuator object (python)
*   @param [in] args    the parameter name
*   @return             the parameter value
*
*   The getParam method of the plugin is invoked and the actual parameter value is returned. If the parameter
*   doesn't exist an error is returned.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getParam(PyActuatorPlugin* self, PyObject * args)
{
    return getParam(self->actuatorObj, args);
}

//-------------------------------------------------------------------------------------
/** set a parameter to a new value
*   @param [in] self    the actuator object (python)
*   @param [in] args    the parameter name and new value
*   @return             an error if the parameter wasn't found or the passed value is out of the limits
*
*   The setParam method of the plugin is invoked and the parameter is set to the new value in case the passed value
*   is within the limits.
*/
PyObject* PythonPlugins::PyActuatorPlugin_setParam(PyActuatorPlugin* self, PyObject * args)
{
    return setParam(self->actuatorObj, args);
}

//-------------------------------------------------------------------------------------
/** returns dictionary with meta information about desired parameter
*   @param [in] self    the actuator object (python)
*   @param [in] args    the parameter name
*   @return             an error if the parameter wasn't found
*
*   The getParamInfo method of the plugin is invoked
*/
PyObject* PythonPlugins::PyActuatorPlugin_getParamInfo(PyActuatorPlugin* self, PyObject * args)
{
    return getParamInfo(self->actuatorObj, args);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorCalib_doc, "calib(axisIndex1, *args) \n\
\n\
Starts a calibration or homing routine of one or multiple axes. \n\
\n\
Most actuators have the possibility to calibrate or home certain axes. \n\
Use this command to start the calibration. \n\
\n\
Parameters \n\
---------- \n\
axisIndex1 : int\n\
    Index of the first axis to be calibrated or homed (e.g. 0 for first axis). \n\
*args : int \n\
    Pass further axis indices as 2nd, 3rd, etc. parameter to this function \n\
    if more than one axis should be calibrated or homed. \n\
\n\
Raises \n\
------ \n\
NotImplemented \n\
    if calibration routine not available in this plugin.");
/** calibrate actuator axi(e)s
*   @param [in] self    the actuator object (python)
*   @param [in] args    the axi(e)s numbers
*   @return             status of calibration
*
*   Invokes the calibrate method on an actuator object with the numbers of the axis passed. The status is
*   of the calibration is returned or an error.
*/
PyObject* PythonPlugins::PyActuatorPlugin_calib(PyActuatorPlugin* self, PyObject * args)
{
    ito::RetVal ret = ito::retOk;
    bool ok = true;
    QVector<int> axisVec = PythonQtConversion::PyObjGetIntArray(args, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "At least one given axis index cannot be interpreted as integer number.");
        return nullptr;
    }
    else if (axisVec.size() == 0)
    {
        PyErr_SetString(PyExc_ValueError, "no axis specified");
        return nullptr;
    }

    // if a Python script has been interrupted, depending on itom settings, an interrupt is sent to all connected actuators.
    // If these actuators did not check for this flag in the meantime, reset it now.
    self->actuatorObj->resetInterrupt();

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    bool invokeOk;

    if (axisVec.size() == 1)
    {
        invokeOk = QMetaObject::invokeMethod(self->actuatorObj, "calib", Q_ARG(int, axisVec[0]), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }
    else
    {
        invokeOk = QMetaObject::invokeMethod(self->actuatorObj, "calib", Q_ARG(QVector<int>, axisVec), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }

    if (invokeOk)
    {
        bool timeout = false;

        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->actuatorObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calibration").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'calib' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    if (!PythonCommon::setReturnValueMessage(ret, "calib", PythonCommon::invokeFunc))
    {
        return nullptr;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorSetOrigin_doc, "setOrigin(axisIndex1, *args) \n\
\n\
Defines the current position of the given axes to have the value ``0``. \n\
\n\
The current positions of all indicated axes (``axisIndex1``, ``*args``) are considered \n\
to be ``0`` such that future positioning commands are relative with respect to this \n\
current position. \n\
\n\
Parameters \n\
---------- \n\
axisIndex1 : int\n\
    index of the first axis (e.g. 0 for first axis) \n\
*args : int \n\
    Pass further axis indices as 2nd, 3rd, etc. parameter to this function \n\
    if more than one axis should be origined. \n\
\n\
Raises \n\
------ \n\
NotImplemented \n\
    if actuator does not support this feature");
/** set the origin of axi(e)s
*   @param [in] self    the actuator object (python)
*   @param [in] args    the axi(e)s
*   @return             status of setOrigin
*
*   The axi(e)s current position is set as new origin of the axi(e)s.
*/
PyObject* PythonPlugins::PyActuatorPlugin_setOrigin(PyActuatorPlugin* self, PyObject * args)
{
    ito::RetVal ret = ito::retOk;
    bool ok = true;

    QVector<int> axisVec = PythonQtConversion::PyObjGetIntArray(args, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "At least one given axis index cannot be interpreted as integer number.");
        return nullptr;
    }
    else if (axisVec.size() == 0)
    {
        PyErr_SetString(PyExc_ValueError, "no axis specified");
        return nullptr;
    }

    // if a Python script has been interrupted, depending on itom settings, an interrupt is sent to all connected actuators.
    // If these actuators did not check for this flag in the meantime, reset it now.
    self->actuatorObj->resetInterrupt();

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    bool invokeOk;
    if (axisVec.size() == 1)
    {
        invokeOk = QMetaObject::invokeMethod(self->actuatorObj, "setOrigin", Q_ARG(int, axisVec[0]), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }
    else
    {
        invokeOk = QMetaObject::invokeMethod(self->actuatorObj, "setOrigin", Q_ARG(QVector<int>, axisVec), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }

    if (invokeOk)
    {
        bool timeout = false;
        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->actuatorObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while setting origin").toLatin1().data());
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'setOrigin' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    if (!PythonCommon::setReturnValueMessage(ret, "setOrigin", PythonCommon::invokeFunc))
    {
        return nullptr;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorGetStatus_doc, "getStatus(axis = -1) -> Union[int, List[int]] \n\
\n\
Returns the status for one single axis or all axes of the actuator object. \n\
\n\
Each axis of an actuator plugin has got a status value that is used for informing \n\
about the current status of the axis. \n\
\n\
The status value is a bitmask (flag), that might contain a combination of the \n\
following values: \n\
\n\
Moving flags: \n\
\n\
* actuatorUnknown     = 0x0001 : unknown current moving status \n\
* actuatorInterrupted = 0x0002 : movement has been interrupted by the user or another \n\
  error during the movement occurred \n\
* actuatorMoving      = 0x0004 : axis is currently moving \n\
* actuatorAtTarget    = 0x0008 : axis reached the target position \n\
* actuatorTimeout     = 0x0010 : timout during movement. Unknown status of the movement \n\
\n\
Switches flags: \n\
\n\
* actuatorEndSwitch   = 0x0100 : axis reached any end switch (e.g. if only one end switch \n\
  is available) \n\
* actuatorEndSwitch1  = 0x0200 : axis reached the specified left end switch (if set, also \n\
  set actuatorEndSwitch)\n\
* actuatorEndSwitch2  = 0x0400 : axis reached the specified left end switch (if set, also \n\
  set actuatorEndSwitch)\n\
* actuatorRefSwitch   = 0x0800 : axis reached any reference switch (e.g. for calibration...) \n\
* actuatorRefSwitch1  = 0x1000 : axis reached the specified right reference switch \n\
  (if set, also set actuatorRefSwitch)\n\
* actuatorRefSwitch2  = 0x2000 : axis reached the specified right reference switch \n\
  (if set, also set actuatorRefSwitch)\n\
\n\
Status flags: \n\
\n\
* actuatorAvailable   = 0x4000 : the axis is available \n\
* actuatorEnabled     = 0x8000 : the axis is currently enabled and can be moved \n\
* actuatorError       = 0x10000 : axis has encountered error/reports error\n\
\n\
Parameters \n\
---------- \n\
axis : int, optional\n\
    If an index >= 0 is passed, the status of this specific axis is returned. \n\
    Else, a list of status values for all axes is returned (default). \n\
\n\
Returns \n\
------- \n\
int or list of int \n\
    Single status value or a list of status values as combination of the \n\
    possible flag values, given above.");
/** get the status of an actuator
*   @param [in] self    the actuator object (python)
*   @param [in] args    the axi(e)s numbers
*   @return             an error if the parameter wasn't found or the passed value is out of the limits
*
*   Returns the status of the axi(e)s passed as parameter.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getStatus(PyActuatorPlugin* self, PyObject * args)
{
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    ito::RetVal ret = ito::retOk;
    int length = PyTuple_Size(args);

    int axis = -1;
    PyObject *result = NULL;

    if (!PyArg_ParseTuple(args, "|i", &axis))
    {
        return NULL;
    }

    if (axis == -1)
    {
        QSharedPointer<QVector<int> > status(new QVector<int>());

        if (QMetaObject::invokeMethod(self->actuatorObj, "getStatus", Q_ARG(QSharedPointer<QVector<int> >, status), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            bool timeout = false;

            while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->actuatorObj->isAlive())
                {
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while getting status").toLatin1().data());
                    timeout = true;
                    break;
                }
            }

            if (!timeout)
            {
                ret += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'getStatus' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

        if (!PythonCommon::setReturnValueMessage(ret, "getStatus", PythonCommon::invokeFunc))
        {
            return NULL;
        }

        int size = status->size();
        result = PyList_New(size); //new ref

        for (int i = 0; i < size; ++i)
        {
            PyList_SetItem(result, i, PyLong_FromLong((*status)[i]));
        }
    }
    else
    {
        QSharedPointer<int> status(new int);

        if (QMetaObject::invokeMethod(self->actuatorObj, "getStatus", Q_ARG(int, axis), Q_ARG(QSharedPointer<int>, status), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            bool timeout = false;
            while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->actuatorObj->isAlive())
                {
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while getting status").toLatin1().data());
                    timeout = true;
                    break;
                }
            }

            if (!timeout)
            {
                ret += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'getStatus' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

        if (!PythonCommon::setReturnValueMessage(ret, "getStatus", PythonCommon::invokeFunc))
        {
            return NULL;
        }

        result = PyLong_FromLong(*status);
    }

    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorGetPos_doc, "getPos(axisIndex1, *args) -> Union[float, Tuple[float]] \n\
\n\
Returns the current position(s) of the given axis or axes (in mm or degree).\n\
\n\
This method requests the current position(s) of the given axes and returns it or them. \n\
\n\
Parameters \n\
---------- \n\
axisIndex1 : int\n\
    index of the first axis (e.g. 0 for first axis) \n\
*args : int\n\
    Pass further indices of more axes as additional parameters. \n\
\n\
Returns \n\
------- \n\
positions : float or tuple of float \n\
    Current position as float value if only one axis is given or a tuple of floats \n\
    if multiple axis indices are given. The unit is **mm** or **degree**. \n\
\n\
See Also \n\
-------- \n\
setPosRel, setPosAbs");

/** get the current position of axi(e)S
*   @param [in] self    the actuator object (python)
*   @param [in] args    the axi(e)s numbers
*   @return             the axi(e)s position(s)
*
*   Reads the position of the axi(e)s passed as parameter.
*/
PyObject* PythonPlugins::PyActuatorPlugin_getPos(PyActuatorPlugin* self, PyObject * args)
{
    ito::RetVal ret = ito::retOk;
    bool ok = true;

    QVector<int> axisVec = PythonQtConversion::PyObjGetIntArray(args, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "At least one given axis index cannot be interpreted as integer number.");
        return nullptr;
    }
    else if (axisVec.size() == 0)
    {
        PyErr_SetString(PyExc_ValueError, "no axis specified");
        return nullptr;
    }

    QSharedPointer<double> pos(new double);
    *pos = 0.0;
    QSharedPointer<QVector<double> > posVec(new QVector<double>(axisVec.size(), 0.0));
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    bool invokeOk;

    if (axisVec.size() == 1)
    {
        invokeOk = QMetaObject::invokeMethod(
            self->actuatorObj,
            "getPos",
            Q_ARG(int, axisVec[0]),
            Q_ARG(QSharedPointer<double>, pos),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }
    else
    {
        invokeOk = QMetaObject::invokeMethod(
            self->actuatorObj,
            "getPos",
            Q_ARG(QVector<int>, axisVec),
            Q_ARG(QSharedPointer<QVector<double> >, posVec),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }

    if (invokeOk)
    {
        bool timeout = false;
        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->actuatorObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while getting position values").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("Member 'getPos' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    PyObject* result = nullptr;

    if (axisVec.size() > 1)
    {
        result = PyTuple_New(axisVec.size());

        for (int n = 0; n < axisVec.size(); ++n)
        {
            PyTuple_SetItem(result, n, PyFloat_FromDouble((*posVec)[n])); //steals a ref
        }
    }
    else
    {
        result = PyFloat_FromDouble(*pos);
    }


    if (!PythonCommon::setReturnValueMessage(ret, "getPos", PythonCommon::invokeFunc))
    {
        return nullptr;
    }

    return result;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyActuatorPlugin_getType_doc, "getType() -> int \n\
\n\
Returns the type value of this actuator plugin (always: 0x2). \n\
\n\
Returns \n\
------- \n\
int \n\
    actuator type value (``0x2``).");
/** returns the type of the actuator object
*   @param [in] self    the actuator object (python)
*   @return             a string with the type
*
*   This method simply returns the type of the actuator object
*/
PyObject* PythonPlugins::PyActuatorPlugin_getType(PyActuatorPlugin *self)
{
    PyObject *result = NULL;
    if (self == NULL || self->actuatorObj == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"empty actuator plugin");
        return NULL;
    }
    else
    {
        ito::AddInInterfaceBase *aib = self->actuatorObj->getBasePlugin();
        if (aib)
        {
            result = PyLong_FromLong(aib->getType());
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError,"interface of plugin is NULL");
            return NULL;
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyActuatorPlugin_execFunc(PyActuatorPlugin *self, PyObject *args, PyObject *kwds)
{
    return execFunc(self->actuatorObj, args, kwds);
}

//-------------------------------------------------------------------------------------
/** open configuration dialog
*   @param [in] self    the actuator object (python)
*
*   This method simply open the configuration dialog
*/
PyObject* PythonPlugins::PyActuatorPlugin_showConfiguration(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return plugin_showConfiguration(aib);
}

//-------------------------------------------------------------------------------------


/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*
*   This method simply open the widget
*/
PyObject* PythonPlugins::PyActuatorPlugin_showToolbox(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return plugin_showToolbox(aib);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorSetInterrupt_doc, "setInterrupt() \n\
\n\
Request the interruption of the movement of this actuator. \n\
\n\
Sets the interrupt flag of the :class:`actuator`. The actuator interrupts \n\
the movement of all running axes as soon as this flag is checked and handled again.");

/** sets the interrupt flag of the actuator in order to interrupt a movement
*/
PyObject* PythonPlugins::PyActuatorPlugin_setInterrupt(PyActuatorPlugin *self)
{
    if (self->actuatorObj)
    {
        //direct call is thread-safe since the flag is protected by a mutex.
        self->actuatorObj->setInterrupt();
    }
    else
    {
        return PyErr_Format(PyExc_RuntimeError, "actuator is invalid");
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorConnect_doc, "connect(signalSignature, callableMethod, minRepeatInterval = 0) \n\
\n\
Connects a signal of this actuator with the given callable Python method. \n\
\n\
Every :class:`actuator` object can emit different signals whenever a certain event \n\
occurs. Use the method :meth:`info` to get a print-out of a list of possible signals \n\
of the actuator. This method is used to connect a certain callable Python callback \n\
method or function to a specific signal. The callable function can be bounded as well \n\
as unbounded. \n\
\n\
The connection is described by the string signature of the signal (hence the source of \n\
the connection). Such a signature is the name of the signal, followed by the types of \n\
its arguments (the original C++ types). An example is ``targetChanged(QVector<double>)``, \n\
emitted whenever the target position of one or multiple axes changed. This signal can \n\
be connected to a callback function, that accepts one argument (in case of a bounded method, \n\
the ``self`` argument must be an additional first parameter. \n\
\n\
The C++ datatype ``QVector<double>`` will be transformed to ``tuple of float``, for \n\
more type conversions see the table in section :ref:`qtdesigner-datatypes`. In general, \n\
a ``callableMethod`` must be a method or function with the same number of parameters than \n\
the signal has (besides the ``self`` argument). The types are converted based on the itom \n\
C++ <-> Python conversion table (:ref:`qtdesigner-datatypes`). \n\
\n\
If a signal is emitted very often, it can be necessary to limit the call of the callback \n\
function to a certain minimum time interval. This can be given by the ``minRepeatInterval`` \n\
parameter. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. ``targetChanged(QVector<double>)``) \n\
callableMethod : callable \n\
    valid method or function that is called if the signal is emitted. \n\
minRepeatInterval : int, optional \n\
    If > 0, the same signal only invokes a slot once within the given interval (in ms). \n\
    Default: 0 (all signals will invoke the callable python method. \n\
\n\
See Also \n\
-------- \n\
disconnect, info");
PyObject* PythonPlugins::PyActuatorPlugin_connect(PyActuatorPlugin *self, PyObject* args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", "minRepeatInterval", NULL };
    const char* signalSignature;
    PyObject *callableMethod;
    int signalIndex;
    int tempType;
    IntList argTypes;
    int minRepeatInterval = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|i", const_cast<char**>(kwlist), &signalSignature, &callableMethod, &minRepeatInterval))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }
    if (!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }
    if (!self->actuatorObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }

    QByteArray signature(signalSignature);
    const QMetaObject *mo = self->actuatorObj->metaObject();
    signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature));
    QMetaMethod metaMethod = mo->method(signalIndex);
    QList<QByteArray> names = metaMethod.parameterTypes();

    foreach(const QByteArray& name, names)
    {
        tempType = QMetaType::type(name.constData());
        if (tempType > 0)
        {
            argTypes.append(tempType);
        }
        else
        {
            QString msg = QString("parameter type %1 is unknown").arg(name.constData());
            PyErr_SetString(PyExc_RuntimeError, msg.toLatin1().data());
            signalIndex = -1;
            return NULL;
        }
    }
    if (self->signalMapper)
    {
        if (!self->signalMapper->addSignalHandler(self->actuatorObj, signalSignature, signalIndex, callableMethod, argTypes, minRepeatInterval))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No signalMapper for this plugin could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorDisconnect_doc, "disconnect(signalSignature, callableMethod) \n\
\n\
Disconnects a connection which must have been established before with exactly the same parameters.\n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. ``clicked(bool)``) \n\
callableMethod : callable \n\
    valid method or function, that should not be called any more if the \n\
    given signal is emitted. \n\
\n\
See Also \n\
-------- \n\
connect, info");
PyObject *PythonPlugins::PyActuatorPlugin_disconnect(PyActuatorPlugin *self, PyObject* args, PyObject* kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", NULL };
    int signalIndex;
    const char* signalSignature;
    PyObject *callableMethod;
    IntList argTypes;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", const_cast<char**>(kwlist), &signalSignature, &callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }
    if (!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }
    if (!self->actuatorObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }

    const QMetaObject *mo = self->actuatorObj->metaObject();
    signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature));
    QMetaMethod metaMethod = mo->method(signalIndex);
    if (self->signalMapper)
    {
        if (!self->signalMapper->removeSignalHandler(self->actuatorObj, signalIndex, callableMethod))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No signalMapper for this plugin could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorInfo_doc, "info(verbose = 0) \n\
\n\
Prints out information about signal and callable slots of this actuator.\n\
\n\
Parameters \n\
---------- \n\
verbose : int \n\
    0: only slots and signals from the plugin class are printed (default) \n\
    1: all slots and signals from all inherited classes are printed\n\
\n\
See Also \n\
-------- \n\
connect, disconnect");
PyObject* PythonPlugins::PyActuatorPlugin_info(PyActuatorPlugin* self, PyObject* args)
{
    int showAll = 0;

    if (!PyArg_ParseTuple(args, "|i", &showAll))
    {
        return NULL;
    }
    if (!self->actuatorObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }
    //QList<QByteArray> signalSignatureList, slotSignatureList;
    QStringList signalSignatureList, slotSignatureList;
    const QMetaObject *mo = self->actuatorObj->metaObject();
    QMetaMethod metaFunc;
    bool again = true;
    int methodIdx;
    if (showAll == 0 || showAll == 1)
    {
        while (again)
        {
            for (methodIdx = mo->methodOffset(); methodIdx < mo->methodCount(); ++methodIdx)
            {
                metaFunc = mo->method(methodIdx);
                if (metaFunc.methodType() == QMetaMethod::Signal)
                {
                    signalSignatureList.append(metaFunc.methodSignature());

                }
                if (metaFunc.methodType() == QMetaMethod::Slot)
                {
                    slotSignatureList.append(metaFunc.methodSignature());
                }

            }
            if (showAll == 1)
            {
                mo = mo->superClass();
                if (mo)
                {
                    again = true;
                    continue;
                }
            }
            again = false;

        }
    }
    else
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Invalid verbose level. Use level 0 to display all signals and slots "
            "defined by the plugin itself. Level 1 also displays all inherited signals and slots");
        return NULL;
    }

    signalSignatureList.sort();
    slotSignatureList.sort();

    if (signalSignatureList.length() || slotSignatureList.length())
    {
        //QByteArray val;
        QString val;
        QString previous;
        std::cout << "Signals: \n";

        foreach(val, signalSignatureList)
        {
            if (val != previous)
            {
                std::cout << "\t" << QString(val).toLatin1().data() << "\n";
            }

            previous = val;
        }

        std::cout << "\nSlots: \n";

        foreach(val, slotSignatureList)
        {
            if (val != previous)
            {
                std::cout << "\t" << QString(val).toLatin1().data() << "\n";
            }

            previous = val;
        }
    }

    Py_RETURN_NONE;
}
//-------------------------------------------------------------------------------------
/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*
*   This method simply close the widget
*/
PyObject* PythonPlugins::PyActuatorPlugin_hideToolbox(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return plugin_hideToolbox(aib);
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyActuatorPlugin_userMutex_tryLock(PyActuatorPlugin* self, PyObject* args, PyObject* kwds)
{
    ito::AddInBase *aib = self->actuatorObj;
    return plugin_userMutexLock(aib, args, kwds, self->userMutexLocked);
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyActuatorPlugin_userMutex_unlock(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return plugin_userMutexUnlock(aib, self->userMutexLocked);
}

//-------------------------------------------------------------------------------------
/** helper function to parse the positioning parameters for an actuator
*   @param [in]  args       arguments passed to the function (in python)
*   @param [out]    axisVec Vector with axes numbers
*   @param [out]    posVec  Vector with position values
*   @return                 retOk of parameters could be parsed, retError otherwise
*
*   Parses the parameters passed to a setPos command in python. For each axis that should be positioned
*   an axis number and a position value are expected.
*/
ito::RetVal parsePosParams(PyObject *args, QVector<int> &axisVec, QVector<double> &posVec)
{
    ito::RetVal ret = ito::retOk;
    int length = PyTuple_Size(args);
    bool ok = true;
    axisVec.clear();
    posVec.clear();

    if (length < 2)
    {
        PyErr_SetString(PyExc_ValueError, "At least one axis index and position must be given.");
        return ito::retError;
    }
    else if ((length % 2) != 0)
    {
        PyErr_SetString(PyExc_ValueError, "Number of axis indices and position arguments must be equal.");
        return ito::retError;
    }
    else
    {
        PyObject *axisVal = nullptr;
        PyObject *posVal = nullptr;
        axisVec.resize(length / 2);
        posVec.resize(length / 2);

        for (int idx = 0; idx < length / 2; ++idx)
        {
            axisVal = PyTuple_GetItem(args, idx * 2); // borrowed
            posVal = PyTuple_GetItem(args, idx * 2 + 1); // borrowed

            axisVec[idx] = PythonQtConversion::PyObjGetInt(axisVal, true, ok);

            if (!ok)
            {
                PyErr_Format(PyExc_TypeError, "The %i. axis index is no integer number.", idx + 1);
                return ito::retError;
            }

            posVec[idx] = PythonQtConversion::PyObjGetDouble(posVal, false, ok);

            if (!ok)
            {
                PyErr_Format(PyExc_TypeError, "The %i. position value is no float value.", idx + 1);
                return ito::retError;
            }
        }
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorSetPosAbs_doc, "setPosAbs(axisIndex1, pos1, *args) \n\
\n\
Moves the given axis or axes to the indicated absolute position(s) (in mm or degree).\n\
\n\
The parameters of this function are always alternating between the index of one axis \n\
and its new absolute target position as following parameter. As an example, moving \n\
the first three axes would look like::  \n\
    \n\
    myMotor.setPosAbs(0, 10.0, 1, -5.2, 2, 0.7)  # axes 0, 1 and 2 are absolutely moved \n\
\n\
This method starts the absolute positioning of all given axes. If the ``async`` parameter \n\
(see :meth:`getParam` and :meth:`setParam`) of the plugin is ``0`` (usually default), \n\
a synchronous positioning is started, hence, this method returns after that all \n\
axes reached their target positions or a timeout occurred. Else, (``async = 1``) this \n\
method immediately returns and the actuator continuous its movement. \n\
\n\
Parameters \n\
---------- \n\
axisIndex1 : int \n\
    index of the first axis, that should be moved. \n\
pos1 : float \n\
    absolute target position for this first axis ``axisIndex1`` (in mm or degree) \n\
*args \n\
    Pass more arguments of the form ``axisIndexX, posX`` to move more than one axis. \n\
\n\
See Also \n\
-------- \n\
getPos, setPosRel");

/** set actuator axi(e)s to new absolute position(s)
*   @param [in] self    the actuator object (python)
*   @param [in] args    the axi(e)s numbers
*   @return             status of positioning command
*
*   The passed parameters are parsed using the helper function \ref parsePosParams and in case a meaningful
*   number of axi(e)s and position(s) are found the setPosAbs method of the actuator object is invoked.
*/
PyObject* PythonPlugins::PyActuatorPlugin_setPosAbs(PyActuatorPlugin* self, PyObject * args)
{
    ito::RetVal ret = ito::retOk;
    QVector<int> axisVec;
    QVector<double> posVec;

    if ((ret = parsePosParams(args, axisVec, posVec)) != ito::retOk)
    {
        return nullptr;
    }

    // if a Python script has been interrupted, depending on itom settings, an interrupt is sent to all connected actuators.
    // If these actuators did not check for this flag in the meantime, reset it now.
    self->actuatorObj->resetInterrupt();

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    bool invokeOk;

    if (axisVec.size() == 1)
    {
        invokeOk = QMetaObject::invokeMethod(
            self->actuatorObj,
            "setPosAbs",
            Q_ARG(int, axisVec[0]),
            Q_ARG(double, posVec[0]),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        );
    }
    else
    {
        invokeOk = QMetaObject::invokeMethod(
            self->actuatorObj,
            "setPosAbs",
            Q_ARG(QVector<int>, axisVec),
            Q_ARG(QVector<double>, posVec),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        );
    }

    if (invokeOk)
    {
        bool timeout = false;
        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->actuatorObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while setting absolute position").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0,
            QObject::tr("Member 'setPosAbs' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    if (!PythonCommon::setReturnValueMessage(ret, "setPosAbs", PythonCommon::invokeFunc))
    {
        return nullptr;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorSetPosRel_doc, "setPosRel(axisIndex1, offset1, *args) \n\
\n\
Moves the given axis or axes to the indicated relative position(s) (in mm or degree).\n\
\n\
The parameters of this function are always alternating between the index of one axis \n\
and its new absolute target position as following parameter. As an example, moving \n\
the first three axes would look like::  \n\
    \n\
    myMotor.setPosAbs(0, 10.0, 1, -5.2, 2, 0.7)  # axes 0, 1 and 2 are absolutely moved \n\
\n\
This method starts the relative positioning of all given axes. If the ``async`` parameter \n\
(see :meth:`getParam` and :meth:`setParam`) of the plugin is ``0`` (usually default), \n\
a synchronous positioning is started, hence, this method returns after that all \n\
axes reached their target positions or a timeout occurred. Else, (``async = 1``) this \n\
method immediately returns and the actuator continuous its movement. \n\
\n\
Parameters \n\
---------- \n\
axisIndex1 : int \n\
    index of the first axis, that should be moved. \n\
offset1 : float \n\
    The new target position for the first axis ``axisIndex1`` is given by the current \n\
    position of this axis plus this ``offset1`` value (in mm or degree) \n\
*args \n\
    Pass more arguments of the form ``axisIndexX, offsetX`` to move more than one axis. \n\
\n\
See Also \n\
-------- \n\
getPos, setPosAbs");
/** set actuator axi(e)s to new relative position(s)
*   @param [in] self    the actuator object (python)
*   @param [in] args    the axi(e)s numbers
*   @return             status of positioning command
*
*   The passed parameters are parsed using the helper function \ref parsePosParams and in case a meaningful
*   number of axi(e)s and position(s) are found the setPosRel method of the actuator object is invoked.
*/
PyObject* PythonPlugins::PyActuatorPlugin_setPosRel(PyActuatorPlugin* self, PyObject * args)
{
    ito::RetVal ret = ito::retOk;
    QVector<int> axisVec;
    QVector<double> posVec;

    if ((ret = parsePosParams(args, axisVec, posVec)) != ito::retOk)
    {
        return nullptr;
    }

    // if a Python script has been interrupted, depending on itom settings, an interrupt is sent to all connected actuators.
    // If these actuators did not check for this flag in the meantime, reset it now.
    self->actuatorObj->resetInterrupt();

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    bool invokeOk;

    if (axisVec.size() == 1)
    {
        invokeOk = QMetaObject::invokeMethod(
            self->actuatorObj,
            "setPosRel",
            Q_ARG(int, axisVec[0]),
            Q_ARG(double, posVec[0]),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }
    else
    {
        invokeOk = QMetaObject::invokeMethod(
            self->actuatorObj,
            "setPosRel",
            Q_ARG(QVector<int>, axisVec),
            Q_ARG(QVector<double>, posVec),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    }

    if (invokeOk)
    {
        bool timeout = false;
        while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->actuatorObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while setting relative position").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += locker.getSemaphore()->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0,
            QObject::tr("Member 'setPosRel' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    if (!PythonCommon::setReturnValueMessage(ret, "setPosRel", PythonCommon::invokeFunc))
    {
        return nullptr;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorGetCurrentStatus_doc,
"tuple of int : Gets the current status (flag mask, see :py:meth:`~itom.actuator.getStatus`) of all axes \n\
\n\
This property returns a tuple whose size corresponds to the number of axes of this \n\
actuator. The returned tuple contains the current positions of all axes (in mm or degree). \n\
This property is always updated if the plugin signals a change of any current position \n\
via the signal 'actuatorStatusChanged'. Instead of reading this property, you can also \n\
connect to this signal in order to get instantly informed about new current positions. \n\
\n\
The difference between this property and the method :py:meth:`~itom.actuator.getStatus` \n\
is that `getStatus` will only return if the actuator plugin is currently idle. This \n\
property always returns immediately, however it only contains the last reported values \n\
which can slightly differ from the real current positions (if the plugin rarely emits its \n\
current states for instance due to performance reasons).");
/*static*/ PyObject* PythonPlugins::PyActuatorPlugin_getCurrentStatus(PyActuatorPlugin *self, void * /*closure*/)
{
    if (!self->actuatorObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }

    QVector<int> status;
    QVector<double> currentPosition;
    QVector<double> targetPosition;
    ito::RetVal ret = self->actuatorObj->getLastSignalledStates(status, currentPosition, targetPosition);

    if (!PythonCommon::setReturnValueMessage(ret, "currentStatus", PythonCommon::getProperty))
    {
        return NULL;
    }

    PyObject* result = PyTuple_New(status.size());
    int i = 0;

    foreach(int s, status)
    {
        PyTuple_SET_ITEM(result, i, PyLong_FromLong(s)); //steals a reference
        i++;
    }

    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorGetCurrentPositions_doc,
"tuple of float : Gets the current positions (in mm or degree) of all axes. \n\
\n\
This property returns a tuple whose size corresponds to the number of axes of this \n\
actuator. The returned tuple contains the current positions of all axes (in mm or degree). \n\
This property is always updated if the plugin signals a change of any current position \n\
via the signal ``actuatorStatusChanged``. Instead of reading this property, you can also \n\
connect to this signal in order to get instantly informed about new current positions. \n\
\n\
This property always returns immediately, however it \n\
only contains the last reported values which can slightly differ from the real current \n\
positions (if the plugin rarely emits its current states for instance due to performance \n\
reasons).");
/*static*/ PyObject* PythonPlugins::PyActuatorPlugin_getCurrentPositions(PyActuatorPlugin *self, void * /*closure*/)
{
    if (!self->actuatorObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }

    QVector<int> status;
    QVector<double> currentPosition;
    QVector<double> targetPosition;
    ito::RetVal ret = self->actuatorObj->getLastSignalledStates(status, currentPosition, targetPosition);

    if (!PythonCommon::setReturnValueMessage(ret, "currentPosition", PythonCommon::getProperty))
    {
        return NULL;
    }

    PyObject* result = PyTuple_New(currentPosition.size());
    int i = 0;
    foreach(double s, currentPosition)
    {
        PyTuple_SET_ITEM(result, i, PyFloat_FromDouble(s)); //steals a reference
        i++;
    }

    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyActuatorGetTargetPositions_doc,
"tuple of float : Gets the target positions (in mm or degree) of all axes. \n\
\n\
This property returns a tuple whose size corresponds to the number of axes of this \n\
actuator. The returned tuple contains the current target positions of all axes \n\
(in mm or degree). This property is always updated if the plugin signals a change of \n\
any target position via the signal ``targetChanged``. Instead of reading this property, \n\
you can also connect to this signal in order to get instantly informed about new \n\
target positions. \n\
\n\
This property always returns immediately, however it only contains the last reported \n\
values which can slightly differ from the real target positions (if the plugin rarely \n\
emits its current states for instance due to performance reasons).");
/*static*/ PyObject* PythonPlugins::PyActuatorPlugin_getTargetPositions(PyActuatorPlugin *self, void * /*closure*/)
{
    if (!self->actuatorObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }

    QVector<int> status;
    QVector<double> currentPosition;
    QVector<double> targetPosition;
    ito::RetVal ret = self->actuatorObj->getLastSignalledStates(status, currentPosition, targetPosition);

    if (!PythonCommon::setReturnValueMessage(ret, "targetPosition", PythonCommon::getProperty))
    {
        return NULL;
    }

    PyObject* result = PyTuple_New(targetPosition.size());
    int i = 0;
    foreach(double s, targetPosition)
    {
        PyTuple_SET_ITEM(result, i, PyFloat_FromDouble(s)); //steals a reference
        i++;
    }

    return result;
}

//-----------------------------------------------------------------------------
PyGetSetDef PythonPlugins::PyActuatorPlugin_getseters[] = {
    {"currentStatus",   (getter)PyActuatorPlugin_getCurrentStatus,    (setter)NULL, pyActuatorGetCurrentStatus_doc, NULL },
    {"currentPositions", (getter)PyActuatorPlugin_getCurrentPositions,  (setter)NULL, pyActuatorGetCurrentPositions_doc, NULL },
    {"targetPositions",  (getter)PyActuatorPlugin_getTargetPositions,   (setter)NULL, pyActuatorGetTargetPositions_doc, NULL},
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMethodDef PythonPlugins::PyActuatorPlugin_methods[] = {
   {"getParamList", (PyCFunction)PythonPlugins::PyActuatorPlugin_getParamList, METH_NOARGS, pyPluginGetParamList_doc},
   {"getParamInfo", (PyCFunction)PythonPlugins::PyActuatorPlugin_getParamInfo, METH_VARARGS, pyPluginGetParamInfo_doc},
   {"getParamListInfo", (PyCFunction)PythonPlugins::PyActuatorPlugin_getParamListInfo, METH_VARARGS, pyPluginGetParamListInfo_doc},
   {"getExecFuncsList", (PyCFunction)PythonPlugins::PyActuatorPlugin_getExecFuncsList, METH_NOARGS, pyPluginGetExecFuncsList_doc},/*wrong doc atm.*/
   {"getExecFuncsInfo", (PyCFunction)PythonPlugins::PyActuatorPlugin_getExecFuncsInfo, METH_VARARGS | METH_KEYWORDS, pyPlugInGetExecFuncsInfo_doc},
   {"name", (PyCFunction)PythonPlugins::PyActuatorPlugin_name, METH_NOARGS, pyPluginName_doc},
   {"getParam", (PyCFunction)PythonPlugins::PyActuatorPlugin_getParam, METH_VARARGS, pyPluginGetParam_doc},
   {"setParam", (PyCFunction)PythonPlugins::PyActuatorPlugin_setParam, METH_VARARGS, pyPluginSetParam_doc},
   {"calib", (PyCFunction)PythonPlugins::PyActuatorPlugin_calib, METH_VARARGS, pyActuatorCalib_doc},
   {"setOrigin", (PyCFunction)PythonPlugins::PyActuatorPlugin_setOrigin, METH_VARARGS, pyActuatorSetOrigin_doc},
   {"getStatus", (PyCFunction)PythonPlugins::PyActuatorPlugin_getStatus, METH_VARARGS, pyActuatorGetStatus_doc},
   {"getPos", (PyCFunction)PythonPlugins::PyActuatorPlugin_getPos, METH_VARARGS, pyActuatorGetPos_doc},
   {"setPosAbs", (PyCFunction)PythonPlugins::PyActuatorPlugin_setPosAbs, METH_VARARGS, pyActuatorSetPosAbs_doc},
   {"setPosRel", (PyCFunction)PythonPlugins::PyActuatorPlugin_setPosRel, METH_VARARGS, pyActuatorSetPosRel_doc},
   {"getType", (PyCFunction)PythonPlugins::PyActuatorPlugin_getType, METH_NOARGS, PyActuatorPlugin_getType_doc},
   {"exec", (PyCFunction)PythonPlugins::PyActuatorPlugin_execFunc, METH_KEYWORDS | METH_VARARGS, PyPlugin_execFunc_doc},
   {"showConfiguration", (PyCFunction)PythonPlugins::PyActuatorPlugin_showConfiguration, METH_NOARGS, pyPluginShowConfiguration_doc},
   {"showToolbox", (PyCFunction)PythonPlugins::PyActuatorPlugin_showToolbox, METH_NOARGS, pyPluginShowToolbox_doc},
   {"hideToolbox", (PyCFunction)PythonPlugins::PyActuatorPlugin_hideToolbox, METH_NOARGS, pyPluginHideToolbox_doc},
   {"setInterrupt", (PyCFunction)PythonPlugins::PyActuatorPlugin_setInterrupt, METH_NOARGS, pyActuatorSetInterrupt_doc},
   {"connect", (PyCFunction)PythonPlugins::PyActuatorPlugin_connect, METH_VARARGS | METH_KEYWORDS, pyActuatorConnect_doc},
   {"disconnect", (PyCFunction)PythonPlugins::PyActuatorPlugin_disconnect, METH_VARARGS | METH_KEYWORDS, pyActuatorDisconnect_doc},
   {"info",(PyCFunction)PythonPlugins::PyActuatorPlugin_info, METH_VARARGS, pyActuatorInfo_doc},
   {"userMutexTryLock", (PyCFunction)PythonPlugins::PyActuatorPlugin_userMutex_tryLock, METH_VARARGS | METH_KEYWORDS, PyPlugin_userMutex_tryLock_doc },
   {"userMutexUnlock", (PyCFunction)PythonPlugins::PyActuatorPlugin_userMutex_unlock, METH_NOARGS, PyPlugin_userMutex_unlock_doc },
   {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonPlugins::PyActuatorPluginModule = {
   PyModuleDef_HEAD_INIT,
   "actuator",
   QObject::tr("Itom actuator plugin object").toLatin1().data(),
   -1,
   NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonPlugins::PyActuatorPluginType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "itom.actuator",                         /* tp_name */
   sizeof(PyActuatorPlugin),                /* tp_basicsize */
   0,                                       /* tp_itemsize */
   (destructor)PyActuatorPlugin_dealloc,    /* tp_dealloc */
   0,                                       /* tp_print */
   0,                                       /* tp_getattr */
   0,                                       /* tp_setattr */
   0,                                       /* tp_reserved */
   (reprfunc)PyActuatorPlugin_repr,         /* tp_repr */
   0,                                       /* tp_as_number */
   0,                                       /* tp_as_sequence */
   0,                                       /* tp_as_mapping */
   0,                                       /* tp_hash  */
   0,                                       /* tp_call */
   0,                                       /* tp_str */
   0,                                       /* tp_getattro */
   0,                                       /* tp_setattro */
   0,                                       /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
   pyActuatorInit_doc,                      /* tp_doc */
   0,                                       /* tp_traverse */
   0,                                       /* tp_clear */
   0,                                       /* tp_richcompare */
   offsetof(PyActuatorPlugin, weakreflist), /* tp_weaklistoffset */
   0,                                       /* tp_iter */
   0,                                       /* tp_iternext */
   PyActuatorPlugin_methods,                /* tp_methods */
   PyActuatorPlugin_members,                /* tp_members */
   PyActuatorPlugin_getseters,              /* tp_getset */
   0,                                       /* tp_base */
   0,                                       /* tp_dict */
   0,                                       /* tp_descr_get */
   0,                                       /* tp_descr_set */
   0,                                       /* tp_dictoffset */
   (initproc)PythonPlugins::PyActuatorPlugin_init,      /* tp_init */
   0,                                       /* tp_alloc */
   PyActuatorPlugin_new                     /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};



/*static*/ void PythonPlugins::PyActuatorPlugin_addTpDict(PyObject* tp_dict)
{
    PyObject *value;
    //Status Moving
    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorUnknown);
    PyDict_SetItemString(tp_dict, "actuatorUnknown", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorInterrupted);
    PyDict_SetItemString(tp_dict, "actuatorInterrupted", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorMoving);
    PyDict_SetItemString(tp_dict, "actuatorMoving", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorAtTarget);
    PyDict_SetItemString(tp_dict, "actuatorAtTarget", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorTimeout);
    PyDict_SetItemString(tp_dict, "actuatorTimeout", value);
    Py_DECREF(value);

    //status switches
    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorEndSwitch);
    PyDict_SetItemString(tp_dict, "actuatorEndSwitch", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorEndSwitch1);
    PyDict_SetItemString(tp_dict, "actuatorEndSwitch1", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorEndSwitch2);
    PyDict_SetItemString(tp_dict, "actuatorEndSwitch2", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorRefSwitch);
    PyDict_SetItemString(tp_dict, "actuatorRefSwitch", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorRefSwitch1);
    PyDict_SetItemString(tp_dict, "actuatorRefSwitch1", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorRefSwitch2);
    PyDict_SetItemString(tp_dict, "actuatorRefSwitch2", value);
    Py_DECREF(value);

    //status Flags
    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorAvailable);
    PyDict_SetItemString(tp_dict, "actuatorAvailable", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorEnabled);
    PyDict_SetItemString(tp_dict, "actuatorEnabled", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actuatorError);
    PyDict_SetItemString(tp_dict, "actuatorError", value);
    Py_DECREF(value);

    /*value = Py_BuildValue("i", ito::tActuatorStatus::actMovingMask);
    PyDict_SetItemString(tp_dict, "actuatorMovingMask", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actEndSwitchMask);
    PyDict_SetItemString(tp_dict, "actuatorEndSwitchMask", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actRefSwitchMask);
    PyDict_SetItemString(tp_dict, "actuatorRefSwitchMask", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actSwitchesMask);
    PyDict_SetItemString(tp_dict, "actuatorSwitchesMask", value);
    Py_DECREF(value);

    value = Py_BuildValue("i", ito::tActuatorStatus::actStatusMask);
    PyDict_SetItemString(tp_dict, "actuatorStatusMask", value);
    Py_DECREF(value);*/
}



//-------------------------------------------------------------------------------------
/** desctructor for dataIO object in python
*   @param [in] self
*
*   Destructs an instance of a dataIO object (plugin), i.e. deletes the according python variable and invokes
*   the closeAddIn function. The object itself is only deleted if the object's <<<erence
*   counter is zero.
*/
void PythonPlugins::PyDataIOPlugin_dealloc(PyDataIOPlugin* self)
{
    if (self->weakreflist != NULL)
    {
        PyObject_ClearWeakRefs((PyObject *) self);
    }

    if (self->dataIOObj)
    {
        ito::AddInInterfaceBase *aib = self->dataIOObj->getBasePlugin();
        if (!aib)
        {
            std::cerr << "error closing plugin" << std::endl;
            //PyErr_SetString(PyExc_RuntimeError, "error closing plugin");
        }
        else
        {
            ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
            ito::RetVal retval(ito::retOk);

            ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

            if (QMetaObject::invokeMethod(aim, "closeAddIn", Q_ARG(ito::AddInBase*, (ito::AddInBase*)self->dataIOObj), Q_ARG(ItomSharedSemaphore*, waitCond)))
            {
                waitCond->wait(-1);
                retval += waitCond->returnValue;
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "Member 'closeAddIn' of plugin could not be invoked (error in signal/slot connection).");
            }
            waitCond->deleteSemaphore();
            waitCond = NULL;

            PythonCommon::transformRetValToPyException(retval);
        }
    }

    DELETE_AND_SET_NULL(self->signalMapper);

    if (self->userMutexLocked)
    {
        // unlock the user mutex (if it has not been locked by Python yet, this
        // will be temporarily done in the unlock method)
        PyDataIOPlugin_userMutex_unlock(self);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//-------------------------------------------------------------------------------------
/** constructor for dataIO object in python
*   @param [in] type
*   @return     new python dataIO object
*
*   Creates a new pythonDataIO object. The actual dataIO object (itom) is only created later.
*/
PyObject* PythonPlugins::PyDataIOPlugin_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyDataIOPlugin *self = NULL;

    self = (PyDataIOPlugin *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->dataIOObj = NULL;
        self->base = NULL;
        self->weakreflist = NULL;
        self->signalMapper = new PythonQtSignalMapper();
    }

    return (PyObject *)self;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataIOInit_doc, "dataIO(name, *args, **kwds) -> dataIO \n\
\n\
Creates a new instance of the dataIO plugin ``name``. \n\
\n\
This is the constructor for an :class:`dataIO` plugin. It initialises an new \n\
instance of the plugin with the given ``name``. The initialisation parameters are \n\
parsed and unnamed parameters are used in their incoming order to fill first \n\
mandatory parameters and afterwards optional parameters. Parameters may be passed \n\
with their name as keyword, too. However, as usual, no positional parameters are \n\
allowed after a keyword-based one.\n\
\n\
See :meth:`pluginHelp` for detailed information about the specific initialisation \n\
parameters.\n\
\n\
Parameters \n\
---------- \n\
name : str \n\
    is the fullname (case sensitive) of an :class:`dataIO`-plugin. \n\
*args : Any \n\
    Every ``actuator`` plugin defines a list of mandatory and optional initialization \n\
    parameters. Pass these arguments either as positional (``*args``) or keyword \n\
    based (``**kwds``) arguments, where the mandatory parameters must be given first, \n\
    followed by the optional ones. Not every optional initialization argument must be \n\
    given, else its default value is used. \n\
**kwds : Any \n\
    Further keyword based parameters. See also ``*args``. \n\
\n\
Returns \n\
------- \n\
actuator \n\
    new instance of the desired dataIO plugin.");

/** constructor for dataIO object
*   @param [in] self    the according dataIO object
*   @param [in] args    unnamed arguments passed to the constructor in python
*   @return             -1 in case an error occured, else 0
*
*   The dataIO passed must be a valid dataIO object. In case the autoloading of parameters is activated for this
*   plugin the default parameters are loaded.
*/
int PythonPlugins::PyDataIOPlugin_init(PyDataIOPlugin *self, PyObject *args, PyObject *kwds)
{

    if (args == NULL) //args is only NULL, instance of dataIO is created by a c-code fragment. Then the content of the type-struct has to be filled by the c-code, too.
    {
        return 0;
    }

    self->dataIOObj = NULL;

    int length = PyTuple_Size(args);

    if (length == 0)
    {
        PyErr_SetString(PyExc_ValueError, "no plugin specified");
        return -1;
    }
    else if (length == 1) //!< copy constructor or name only
    {
        PyDataIOPlugin* copyPlugin = NULL;

        if (PyArg_ParseTuple(args, "O!", &PyDataIOPluginType, &copyPlugin))
        {
            //try to increment reference of copyPlugin->dataIOObj
            if (copyPlugin->dataIOObj)
            {
                copyPlugin->dataIOObj->getBasePlugin()->incRef(copyPlugin->dataIOObj);
            }

            self->dataIOObj = copyPlugin->dataIOObj;
            self->base = copyPlugin->base;

            return 0;
        }
    }

    PyErr_Clear();
    QVector<ito::Param> *paramsMand = NULL;
    QVector<ito::Param> *paramsOpt = NULL;
    ito::RetVal retval = ito::retOk;
    int pluginNum = -1;
    PyObject *pnameObj = NULL;
    PyObject *params = NULL;
    QString pluginName;

    QVector<ito::ParamBase> paramsMandCpy;
    QVector<ito::ParamBase> paramsOptCpy;

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, "no addin-manager found");
        return -1;
    }

    pnameObj = PyTuple_GetItem(args, 0);
    if (PyUnicode_Check(pnameObj))
    {
        bool ok = false;
        pluginName = PythonQtConversion::PyObjGetString(pnameObj,false,ok);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "invalid parameters");
        return -1;
    }

    retval = AIM->getInitParams(pluginName, ito::typeDataIO, &pluginNum, paramsMand, paramsOpt);
    if (retval.containsWarningOrError())
    {
        PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin);
        return -1;
    }

    bool enableAutoLoadParams = false;
    retval = findAndDeleteReservedInitKeyWords(kwds, &enableAutoLoadParams);
    if (retval.containsWarningOrError())
    {
        PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin);
        return -1;
    }
    else
    {
        params = PyTuple_GetSlice(args, 1, PyTuple_Size(args)); //new reference

        if (parseInitParams(paramsMand, paramsOpt, params, kwds, paramsMandCpy, paramsOptCpy) != ito::retOk)
        {
            Py_XDECREF(params);
            PyErr_SetString(PyExc_ValueError, "error while parsing parameters.");
            return -1;
        }
        Py_XDECREF(params);

        ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

        if (QMetaObject::invokeMethod(AIM, "initAddIn", Q_ARG(int, pluginNum), Q_ARG(QString, pluginName), Q_ARG(ito::AddInDataIO**, &self->dataIOObj), Q_ARG(QVector<ito::ParamBase>*, &paramsMandCpy), Q_ARG(QVector<ito::ParamBase>*, &paramsOptCpy), Q_ARG(bool, enableAutoLoadParams), Q_ARG(ItomSharedSemaphore*, waitCond)))
        {
            waitCond->wait(-1);
            retval += waitCond->returnValue;
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Member 'initAddIn' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

        waitCond->deleteSemaphore();
        waitCond = NULL;

        paramsMandCpy.clear();
        paramsOptCpy.clear();
    }

    if (!PythonCommon::setReturnValueMessage(retval, pluginName, PythonCommon::loadPlugin))
    {
        return -1;
    }

    return 0;
}



//-------------------------------------------------------------------------------------
//PyDoc_STRVAR(PyDataIOPlugin_name_doc, -> see pyPluginName_doc);

/** Returns the plugin's name
*   @param [in] self    the plugin object
*   @return             the name of the plugin
*
*                       Queries the name of a plugin by invoking a getParam on the plugin for the name parameter
*/
PyObject* PythonPlugins::PyDataIOPlugin_name(PyDataIOPlugin* self)
{
    return getName(self->dataIOObj);
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyDataIOPlugin_repr(PyDataIOPlugin *self)
{
    PyObject *result;
    if (self->dataIOObj == NULL)
    {
        result = PyUnicode_FromFormat("empty dataIO plugin");
    }
    else
    {
        PyObject *name = getName(self->dataIOObj);
        if (name)
        {
            QString ident = self->dataIOObj->getIdentifier();
            if(ident != "")
            {
                result = PyUnicode_FromFormat("DataIO-Plugin(%U, %s, ID: %i)", name, ident.toLatin1().data(), self->dataIOObj->getID());
            }
            else
            {
                result = PyUnicode_FromFormat("DataIO-Plugin(%U, ID: %i)", name, self->dataIOObj->getID());
            }
            Py_DECREF(name);
        }
        else
            result = PyUnicode_FromFormat("dataIO-Plugin, time out reading name");
    }
    return result;
}

//-------------------------------------------------------------------------------------
/** returns the list of available parameters
*   @param [in] self    the dataIO object (python)
*   @return             a string with all available parameters for this dataIO
*
*   All parameters of the plugin are shown. This can be useful as there are only few standard parameters
*   for an dataIO. The majority is depending on the actual hardware and accordingly is different for each
*   plugin.
*/
PyObject* PythonPlugins::PyDataIOPlugin_getParamList(PyDataIOPlugin* self)
{
    ito::AddInBase *aib = self->dataIOObj;
    return getParamList(aib);
}
//-------------------------------------------------------------------------------------
/** returns the list of available parameters and additional information about the plugin
*   @param [in] self    the dataIO object (python)
*   @return             a string with all available parameters for this dataIO
*
*   All parameters of the plugin are shown with additional information as min, max and infostring.
*   This can be useful as there are only few standard parameters for an dataIO. The majority is
*   depending on the actual hardware and accordingly is different for each plugin.
*/
PyObject* PythonPlugins::PyDataIOPlugin_getParamListInfo(PyDataIOPlugin* self, PyObject *args)
{
    ito::AddInBase *aib = self->dataIOObj;
    return getParamListInfo(aib, args);
}

/** returns the list of available ExecFunctions' names
*   @param [in] self    the actuator object (python)
*   @return             a List with all available parameters for this actuator
*
*   All ExecFunctions of the plugin are shown or one specific ExecFunctions with additional information as min, max and infostring is shown.
*   This can be useful as there are only few standard parameters for an actuator. The majority is
*   depending on the actual hardware and accordingly is different for each plugin.
*/
PyObject* PythonPlugins::PyDataIOPlugin_getExecFuncsList(PyActuatorPlugin* self)
{
    ito::AddInBase *aib = self->actuatorObj;
    return getExecFuncsList(aib);
}
/** returns the list of available parameters and additional information about the plugin ExecFunctions
*   @param [in] self    the actuator object (python)
*   @return             a dictionary with all available parameters for this actuator
*
*   All ExecFunctions of the plugin are shown or one specific ExecFunctions with additional information as min, max and infostring is shown.
*   This can be useful as there are only few standard parameters for an dataIO. The majority is
*   depending on the actual hardware and accordingly is different for each plugin.
*/
PyObject* PythonPlugins::PyDataIOPlugin_getExecFuncsInfo(PyDataIOPlugin* self, PyObject *args, PyObject *kwds)
{
    ito::AddInBase *aib = self->dataIOObj;
    return getExecFuncsInfo(aib, args, kwds);
}

//-------------------------------------------------------------------------------------
/** return a parameter value
*   @param [in] self        the addIn whoes parameter is requested
*   @param [in] args        the parameter name
*   @return     python object with the parameter value on success (parameter exists), NULL otherwise
*
*   The function tries to retrieve the value of the parameter with the name given in args. If the parameter does not exist
*   NULL is returned. To actually retrieve the value the getParam function of the plugin is invoked.
*/
PyObject* PythonPlugins::PyDataIOPlugin_getParam(PyDataIOPlugin *self, PyObject *args)
{
    return getParam(self->dataIOObj, args);
}

//-------------------------------------------------------------------------------------
/** set a parameter to a new value
*   @param [in] self    the actuator object (python)
*   @param [in] args    the parameter name and new value
*   @return             an error if the parameter wasn't found or the passed value is out of the limits
*
*   The setParam method of the plugin is invoked and the parameter is set to the new value in case the passed value
*   is within the limits.
*/
PyObject* PythonPlugins::PyDataIOPlugin_setParam(PyDataIOPlugin *self, PyObject *args)
{
    return setParam(self->dataIOObj, args);
}

//-------------------------------------------------------------------------------------
/** returns dictionary with meta information about desired parameter
*   @param [in] self    the actuator object (python)
*   @param [in] args    the parameter name
*   @return             an error if the parameter wasn't found
*
*   The getParamInfo method of the plugin is invoked
*/
PyObject* PythonPlugins::PyDataIOPlugin_getParamInfo(PyDataIOPlugin* self, PyObject * args)
{
    return getParamInfo(self->dataIOObj, args);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_startDevice_doc,"startDevice(count = 1) \n\
\n\
Starts the given dataIO plugin object. \n\
\n\
This command starts the dataIO plugin such that it is ready for data acquisition. \n\
Call this method before you start using commands like :meth:`acquire`, :meth:`getVal` \n\
or :meth:`copyVal`. If the device already is started, an internal start-counter is \n\
incremented by the parameter ``count``. The corresponding :meth:`stopDevice` method \n\
then decrements this counter and finally stops the device once the counter drops to \n\
zero again. \n\
\n\
The counter is necessary, since every connected live image needs to start the device \n\
without knownledge about any previous start. No acquisition is possible, if the device \n\
has not been started, hence the counter is 0. \n\
\n\
Parameters \n\
---------- \n\
count : int, optional \n\
    Number of increments to the internal start-counter (default: 1). \n\
\n\
See Also \n\
-------- \n\
stopDevice");
/** start a dataIO device, i.e. prepare it for recording data
*   @param [in] self    the dataIO object (python)
*   @param [in] args    should be empty
*   @return             an error if the device could not be started
*
*   Start a dataIO device, i.e. prepare it for the acquisition of data. Usually the device is first initialized, then
*   then it is started using this function afterwards a number of acquisitions is done (using \ref PyDataIOPlugin_acquire and
*   \ref PyDataIOPlugin_getVal) and only after the last data set is recorded it is stopped again.
*/
PyObject* PythonPlugins::PyDataIOPlugin_startDevice(PyDataIOPlugin *self, PyObject *args)
{
    int count = 1;

    if (!PyArg_ParseTuple(args, "|i", &count))
    {
        return NULL;
    }

    if (count < 0)
    {
        PyErr_SetString(PyExc_ValueError, "argument 'count' must be >= 0");
        return NULL;
    }
    ito::RetVal ret = ito::retOk;
    ItomSharedSemaphore *waitCond = NULL;
    bool timeout = false;

    for (int i = 0 ; i < count ; i++)
    {
        waitCond = new ItomSharedSemaphore();

        if (QMetaObject::invokeMethod(self->dataIOObj, "startDevice", Q_ARG(ItomSharedSemaphore*, waitCond)))
        {

            while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->dataIOObj->isAlive())
                {
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'startDevice'").toLatin1().data());
                    timeout = true;
                    break;
                }
            }

            if (!timeout)
            {
                ret += waitCond->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'startDevice' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

        waitCond->deleteSemaphore();
        waitCond = NULL;

        if (!PythonCommon::setReturnValueMessage(ret, "startDevice", PythonCommon::invokeFunc))
        {
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_stopDevice_doc,"stopDevice(count = 1) -> Optional[int] \n\
\n\
Stops the given dataIO plugin object. \n\
\n\
If this method is called as many times as he corresponding :meth:`startDevice` method  \n\
(or if the ``counts`` are equal), the :class:`dataIO` device is stopped (not deleted) \n\
and it is not possible to acquire further data. \n\
\n\
Once a live image is connected to a camera, :meth:`startDevice` is automatically called \n\
at start of the live acquisition and :meth:`stopDevice` at shutdown. \n\
\n\
Parameters \n\
---------- \n\
count : int, optional\n\
    if ``count`` > 1, :meth:`stopDevice` is executed ``count`` times, in order to \n\
    decrement the grabber internal start counter. You can also set ``count = -1``, \n\
    then :meth:`stopDevice` is called in a loop until the internal start counter \n\
    drops to 0. The number of effective counts is then returned.\n\
\n\
Returns \n\
------- \n\
counts : None or int \n\
    If ``count = -1`` the number of required calls to ``stopDevice`` to finally \n\
    stop the device is returned. For ``count >= 0``, ``None`` is returned. \n\
\n\
See Also \n\
-------- \n\
startDevice");

/** stop a dataIO device
*   @param [in] self    the dataIO object (python)QString
*   @param [in] args    should be empty
*   @return             an error if the device could not be stopped
*
*   Stop a dataIO device, i.e. it is no longer possible to acquire data with it. See also \ref PyDataIOPlugin_startDevice
*/
PyObject* PythonPlugins::PyDataIOPlugin_stopDevice(PyDataIOPlugin *self, PyObject *args)
{
    int count = 1;

    if (!PyArg_ParseTuple(args, "|i", &count))
    {
        return NULL;
    }

    ito::RetVal ret = ito::retOk;
    ItomSharedSemaphore *waitCond = NULL;
    bool timeout = false;

    if (count >= 0)
    {
        for (int i = 0 ; i < count ; i++)
        {
            waitCond = new ItomSharedSemaphore();
            if (QMetaObject::invokeMethod(self->dataIOObj, "stopDevice", Q_ARG(ItomSharedSemaphore*, waitCond)))
            {
                while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
                {
                    if (!self->dataIOObj->isAlive())
                    {
                        ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while stopping device").toLatin1().data());
                        timeout = true;
                        break;
                    }
                }

                if (!timeout)
                {
                    ret += waitCond->returnValue;
                }
            }
            else
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'stopDevice' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
            }

            waitCond->deleteSemaphore();
            waitCond = NULL;

            if (!PythonCommon::setReturnValueMessage(ret, "stopDevice", PythonCommon::invokeFunc))
            {
                return NULL;
            }
        }

        Py_RETURN_NONE;
    }
    else if (count == -1)
    {
        count = -1;
        bool timeout = false;
        while(!ret.containsWarningOrError())
        {
            count++;
            waitCond = new ItomSharedSemaphore();
            QMetaObject::invokeMethod(self->dataIOObj, "stopDevice", Q_ARG(ItomSharedSemaphore*, waitCond));

            while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->dataIOObj->isAlive())
                {
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while stopping device").toLatin1().data());
                    timeout = true;
                    break;
                }
            }

            ret += waitCond->returnValue;
            waitCond->deleteSemaphore();
            waitCond = NULL;
        }

        if (timeout)
        {
            if (!PythonCommon::setReturnValueMessage(ret, "stopDevice", PythonCommon::invokeFunc))
            {
                return NULL;
            }
            return NULL;
        }
        else
        {
            return Py_BuildValue("i",count);
        }

    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "argument 'count' must be >= 0 or -1");
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_acquire_doc,"acquire(trigger = dataIO.TRIGGER_SOFTWARE) \n\
\n\
Triggers a new camera acquisition. \n\
\n\
This method triggers a new data acquisition. This method immediately returns even if \n\
the acquisition is not finished yet. Use :meth:`getVal` or :meth:`copyVal` to get the \n\
acquired data. Both methods will then block until the data is available or a timeout \n\
occurred. \n\
\n\
Before calling :meth:`acquire`, the device must have been started using \n\
:meth:`startDevice`. \n\
\n\
Parameters \n\
---------- \n\
trigger : int, optional\n\
    Type of the trigger: \n\
    \n\
    * ``dataIO.TRIGGER_SOFTWARE = 0`` : a software trigger is started, hence, the \n\
      acquisition is immediately started when calling this method.\n\
    * others : depending on your camera, this parameter can be used to set other \n\
      triggers, like hardware trigger with raising or falling edges... Please consider \n\
      the documentation of the specific device for possible values.");

/** acquire data with a dataIO device
*   @param [in] self    the dataIO object (python)
*   @param [in] args    number of acquisitions
*   @return             an error if no data could be acquired
*
*   After the device has been initialized and started this method can be used to trigger the acquisition of a
*   data set. The data is then recorded depending on the actually set parameters of the device and can
*   be afterwards retrieved from the device using the \ref getVal method.
*/
PyObject* PythonPlugins::PyDataIOPlugin_acquire(PyDataIOPlugin *self, PyObject *args)
{
    int trigger = 0;
    ito::RetVal ret = ito::retOk;

    if (PyArg_ParseTuple(args, "|i", &trigger) == false)
    {
        return NULL;
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    bool timeout = false;
    if (QMetaObject::invokeMethod(self->dataIOObj, "acquire", Q_ARG(int, trigger), Q_ARG(ItomSharedSemaphore*, waitCond)))
    {

        while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->dataIOObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'acquire'").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += waitCond->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'acquire' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "acquire", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_stop_doc, "stop() \n\
\n\
Stops a started, continuous acquisition. \n\
\n\
This method stops a previously started, continuous data acquisition. This method is not \n\
implemented in every plugin. A common example for its implementation is to stop an \n\
infinite, continuous acquisition job of an AD-converter plugin. \n\
\n\
See also\n\
--------\n\
acquire");

/** stop continuous acquisiiton with a dataIO device
*   @param [in] self    the dataIO object (python)
*   @return             an error if no data could be acquired
*/
PyObject* PythonPlugins::PyDataIOPlugin_stop(PyDataIOPlugin *self)
{
    ito::RetVal ret = ito::retOk;

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    bool timeout = false;

    if (QMetaObject::invokeMethod(self->dataIOObj, "stop", Q_ARG(ItomSharedSemaphore*, waitCond)))
    {
        while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->dataIOObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'stop'").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += waitCond->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'stop' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "stop", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_getVal_doc,"getVal(dataObj) -> None \\\n\
getVal(buffer, length = INT_MAX) -> int \n\
\n\
Gets shallow copy of internal camera image if `dataObj` is provided. Else values from the plugins are copied to given buffer. \n\
\n\
Returns a reference (shallow copy) of the recently acquired image (located in the \n\
internal memory if the plugin) if the plugin is a grabber or camera and the buffer is a \n\
:class:`dataObject`. Please consider that the values of the :class:`dataObject` might \n\
change if a new image is acquired since it is only a reference. Therefore consider copying \n\
the :class:`dataObject` or directly use :meth:`copyVal`. \n\
\n\
If no acquisition has been triggered, this method raises a :obj`RuntimeError`. If the \n\
acquisition is not finished yet, this method blocks and waits until the end of the \n\
acquisition. \n\
\n\
If the plugin is another type than a grabber or camera (e.g. serialIO), this method \n\
requires any :obj:`buffer` object that is preallocated with a reasonable size (e.g. \n\
:obj:`bytearray`, :obj:`bytes` or unicode :obj:`str`. Then, the currently available \n\
data is copied into this buffer object and the size of the copied data is returned. If \n\
the buffer is too small, only the data that fits into the buffer is copied. Another \n\
call to :meth:`getVal` will copy the rest. \n\
\n\
Parameters \n\
---------- \n\
dataObj : dataObject \n\
    Usually for cameras and grabber: A reference (shallow copy) to the internal memory \n\
    of the camera plugin is set to the given data object. Therefore its content may \n\
    change if a new image is being acquired by the camera. Consider taking a deep copy \n\
    if the image (:meth:`dataObject.copy`) or use the method :meth:`copyVal`. \n\
buffer : bytearray or bytes or str \n\
    Usually for all other IO devices or AD-converters: The buffer must be an object \n\
    of type :obj:`bytearray`, :obj:`bytes` or unicode :obj:`str`. The ``length`` \n\
    parameter is then set to the size of the allocated buffer. This buffer is then \n\
    filled with data and the filled size is returned (max: ``length``). \n\
length : int, optional \n\
    Size of the given buffer. This value is usually automatically determined and \n\
    must not be given. \n\
\n\
Returns \n\
------- \n\
None or int \n\
    ``None`` if ``dataObj`` is given, else the size of the values filled into the given \n\
    ``buffer``. \n\
\n\
See Also \n\
-------- \n\
copyVal");

/** get values from a dataIO device
*   @param [in] self    the dataIO object (python)
*   @param [in] args    input buffer
*   @return             an error if no data could be retrieved
*
*   After a device has been started and an acquisition was triggered the result can be retrieved from the device with the
*   getVal method. As argument the input buffer is passed.
*/
PyObject* PythonPlugins::PyDataIOPlugin_getVal(PyDataIOPlugin *self, PyObject *args)
{
    ito::RetVal ret = ito::retOk;
    PyObject* bufferObj = NULL;
    PythonDataObject::PyDataObject* bufferDataObj = NULL;
    Py_ssize_t length = std::numeric_limits<Py_ssize_t>::max();
    ItomSharedSemaphoreLocker locker;
    unsigned int invokeMethod = -1;
    QSharedPointer<int> maxLength(new int);
    *maxLength = 0;
    QSharedPointer<char> sharedBuffer;
    char* tempBuf = NULL;

    //check whether object is a data object
    if (PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &bufferDataObj))
    {
        if (bufferDataObj->dataObject == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "given data object is empty (internal dataObject-pointer is NULL)");
            return NULL;
        }

        locker = (new ItomSharedSemaphore());

        QMetaObject::invokeMethod(
            self->dataIOObj,
            "getVal",
            Q_ARG(void*, (void*)(bufferDataObj->dataObject)),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        );

        invokeMethod = 1;
    }
    else if (PyErr_Clear(), PyArg_ParseTuple(args, "O|i", &bufferObj, &length))
    {
        if (PyByteArray_Check(bufferObj))
        {
            tempBuf  = (char *)PyByteArray_AsString(bufferObj);
            sharedBuffer = PythonSharedPointerGuard::createPythonSharedPointer<char>(tempBuf, bufferObj);
            *maxLength = static_cast<int>(length < 0 ? PyByteArray_Size(bufferObj) : qMin(PyByteArray_Size(bufferObj),length));
        }
        else if (PyBytes_Check(bufferObj))
        {
            tempBuf  = (char *)PyBytes_AsString(bufferObj);
            sharedBuffer = PythonSharedPointerGuard::createPythonSharedPointer<char>(tempBuf, bufferObj);
            *maxLength = static_cast<int>(length < 0 ? PyBytes_Size(bufferObj) : qMin(PyBytes_Size(bufferObj),length));
        }
        else
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "arguments of method must be a byte array or byte object");
            return NULL;
        }

        if (*maxLength <= 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "length of given buffer is zero.");
            return NULL;
        }

        locker = (new ItomSharedSemaphore());

        QMetaObject::invokeMethod(
            self->dataIOObj,
            "getVal",
            Q_ARG(QSharedPointer<char>, sharedBuffer),
            Q_ARG(QSharedPointer<int>, maxLength),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        );

        invokeMethod = 2;
    }
    else
    {
        PyErr_Clear();
        PyErr_SetString(
            PyExc_RuntimeError,
            "arguments of method must be either one data object, byte array or "
            "byte object.");
        return NULL;
    }

    bool timeout = false;

    while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
    {
        if (!self->dataIOObj->isAlive())
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'getVal'").toLatin1().data());
            timeout = true;
            break;
        }
    }

    if (!timeout)
    {
        ret += locker.getSemaphore()->returnValue;
    }

    if (!PythonCommon::setReturnValueMessage(ret, "getVal", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    if (invokeMethod == 1)
    {
        Py_RETURN_NONE; //in case of data-object
    }
    else
    {
        return PyLong_FromLong(*maxLength);
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_copyVal_doc,"copyVal(destObject) \n\
\n\
Gets deep copy of data of this plugin, stored in the given data object. \n\
\n\
Returns a deep copy of the recently acquired data (for grabber and ADDA only) of the \n\
camera or AD-converter device. The deep copy sometimes requires one copy operation \n\
more than the similar command :meth:`getVal`. However, :meth:`getVal` only returns \n\
a reference to the plugin internal data structure whose values might be changed if \n\
another data acquisition is started. \n\
\n\
If no acquisition has been triggered, this method raises a RuntimeError. If the \n\
acquisition is not finished yet, this method blocks and waits until the end of the \n\
acquisition. \n\
\n\
Parameters \n\
---------- \n\
destObject : dataObject\n\
    `dataObject` where the plugin data is copied to. Either provide an empty \n\
    :class:`dataObject` or a :class:`dataObject` whose shape exactly fits to the \n\
    shape of the available data of the plugin. Therefore you can allocate a \n\
    3D data object, set a region of interest to one plane such that the data from \n\
    the plugin is copied into this plane. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the dataIO plugin is anything else than ADDA or grabber\n\
    or if no acquisition has been triggered \n\
\n\
See Also \n\
-------- \n\
getVal");
/** copy values from a dataIO device to an existing dataObject
*   @param [in] self    the dataIO object (python)
*   @param [in] args    input buffer
*   @return             an error if no data could be retrieved
*
*   After a device has been started and an acquisition was triggered the result can be retrieved from the device with the
*   getVal method. As argument the input buffer is passed.
*/
PyObject* PythonPlugins::PyDataIOPlugin_copyVal(PyDataIOPlugin *self, PyObject *args)
{
    int length = PyTuple_Size(args);
    PyObject *tempObj = NULL;
    ito::RetVal ret = ito::retOk;

    if (self->dataIOObj->getBasePlugin()->getType() & ito::typeGrabber)
    {
        if (length != 1)
        {
            PyErr_Format(PyExc_ValueError, "too many parameters");
            return NULL;
        }

        tempObj = PyTuple_GetItem(args, 0);
        ito::DataObject *dObj = NULL;

        if ((Py_TYPE(tempObj) == &PythonDataObject::PyDataObjectType))
        {
            dObj = ((PythonDataObject::PyDataObject *)tempObj)->dataObject;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "argument must be of type itom.dataObject.");
            return NULL;
        }

        if (dObj == NULL)
        {
            PyErr_SetString(PyExc_ValueError, "invalid dataObject");
            return NULL;
        }

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

        if (QMetaObject::invokeMethod(self->dataIOObj, "copyVal", Q_ARG(void*, (void *)dObj), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            bool timeout = false;

            while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->dataIOObj->isAlive())
                {
                    timeout = true;
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'copyVal'").toLatin1().data());
                    break;
                }
            }

            if (!timeout)
            {
                ret += locker.getSemaphore()->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'copyVal' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }

    }
    else if (self->dataIOObj->getBasePlugin()->getType() & ito::typeADDA)
    {
        if (length != 1)
        {
            PyErr_SetString(PyExc_ValueError, "too many parameters");
            return NULL;
        }

        ito::DataObject *dObj = NULL;
        tempObj = PyTuple_GetItem(args, 0);

        if ((Py_TYPE(tempObj) == &PythonDataObject::PyDataObjectType))
        {
            dObj = ((PythonDataObject::PyDataObject *)tempObj)->dataObject;
        }

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

        if (QMetaObject::invokeMethod(self->dataIOObj, "copyVal", Q_ARG(void *, (void *)dObj), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            bool timeout = false;

            while (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->dataIOObj->isAlive())
                {
                    timeout = true;
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'copyVal'").toLatin1().data());
                    break;
                }
            }

            ret += locker.getSemaphore()->returnValue;

        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, QObject::tr("Member 'copyVal' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, QObject::tr("copyVal function only implemented for typeADDA and typeGrabber").toLatin1().data());
    }

    if (!PythonCommon::setReturnValueMessage(ret, "copyVal", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_setVal_doc, "setVal(dataObj) -> None \\\n\
setVal(buffer, length = 1) -> None \n\
\n\
Transfers a dataObject to an ADDA plugin for write, or a bytearray to other dataIO plugins for general purposes. \n\
\n\
If the :class:`dataIO` plugin has the subtype ``ADDA`` (analog-digital converter), \n\
this method is used to send data to one or more analog outputs of the device. \n\
In this case a :class:`dataObject` must be given as first and only argument. \n\
and the second argument ``length`` must be 1. \n\
\n\
For other dataIO plugins, the first argument must be any buffer object, like \n\
a :obj:`bytearray`, :obj:`bytes` or unicode :obj:`str`. The ``length`` is then extracted \n\
from this value. However it is also possible to define a user-defined size using the \n\
``length`` argument. \n\
\n\
Parameters \n\
---------- \n\
dataObj : dataObject \n\
    The array, that should be transmitted to the output of an analog-digital converter. \n\
    Usually, the shape of this array is ``M x N``, where ``M`` channels will obtain up \n\
    to ``N`` new values. This argument is used for ``ADDA`` :class:`dataIO` devices.\n\
buffer : bytearray or bytes or str \n\
    Other :class:`dataIO` devices than ``ADDA`` need to pass a buffer object, \n\
    like a :obj:`bytearray`, :obj:`bytes` or unicode :obj:`str`. \n\
length : int, optional \n\
    Usually, this value is not required, since the length of the ``buffer`` is \n\
    automatically extracted from the given object.");
/** write values to a dataIO device
*   @param [in] self    the dataIO object (python)
*   @param [in] args    output buffer
*   @return             an error if no data could be retrieved
*
*   Analog to the \ref getVal method this method writes data to a dataIO device (e.g. a DA converter or serial port).The
*   data passed in the output buffer is written to the device according to its current parameters.
*/
PyObject* PythonPlugins::PyDataIOPlugin_setVal(PyDataIOPlugin *self, PyObject *args, PyObject *kwds)
{
    ito::RetVal ret = ito::retOk;

    if (self->dataIOObj->getBasePlugin()->getType() & ito::typeADDA)
    {
        PythonDataObject::PyDataObject* dObj = NULL;
        const char *kwlist[] = { "dataObj", NULL };

        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", const_cast<char**>(kwlist), &PythonDataObject::PyDataObjectType, &dObj))
        {
            return NULL;
        }

        ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

        if (QMetaObject::invokeMethod(
            self->dataIOObj,
            "setVal",
            Q_ARG(const char *, (const char *)(dObj->dataObject)),
            Q_ARG(int, 1),
            Q_ARG(ItomSharedSemaphore*, waitCond)))
        {
            bool timeout = false;

            while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->dataIOObj->isAlive())
                {
                    timeout = true;
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'setVal'").toLatin1().data());
                    break;
                }
            }

            if (!timeout)
            {
                ret += waitCond->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("Member 'setVal' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data()
            );
        }

        waitCond->deleteSemaphore();
        waitCond = NULL;
    }
    else
    {
        const char *kwlist[] = { "buffer", "length", NULL };
        int datalen = -1;
        int datalen_temp = 0;
        PyObject *bufferObj = NULL;

        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", const_cast<char**>(kwlist), &bufferObj, &datalen))
        {
            return NULL;
        }

        QString tempString;
        QByteArray ba;
        const char* buf;

        if (PyByteArray_Check(bufferObj))
        {
            buf = PyByteArray_AsString(bufferObj);
            datalen_temp = PyByteArray_Size(bufferObj);
        }
        else if (PyBytes_Check(bufferObj))
        {
            buf = PyBytes_AsString(bufferObj);
            datalen_temp = PyBytes_Size(bufferObj);
        }
        else if (PyUnicode_Check(bufferObj))
        {
            Py_ssize_t wstring_size;
            wchar_t* wstring = PyUnicode_AsWideCharString(bufferObj, &wstring_size);

            if (wstring != nullptr)
            {
                tempString = QString::fromWCharArray(wstring, wstring_size);
                ba = tempString.toLatin1();
                buf = ba.data();
                datalen_temp = ba.length();
                PyMem_Free(wstring);
                wstring = nullptr;
            }
            else
            {
                PyErr_Format(PyExc_TypeError, "given unicode cannot be parsed to an latin1 string.");
                return NULL;
            }

            if (datalen == -1)
            {
                datalen = datalen_temp;
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "wrong parameter type (char buffer | byte array)");
            return NULL;
        }

        ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

        if (QMetaObject::invokeMethod(
            self->dataIOObj,
            "setVal",
            Q_ARG(const char *, (const char *)buf),
            Q_ARG(int, datalen),
            Q_ARG(ItomSharedSemaphore*, waitCond)))
        {
            bool timeout = false;

            while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
            {
                if (!self->dataIOObj->isAlive())
                {
                    timeout = true;
                    ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'setVal'").toLatin1().data());
                    break;
                }
            }

            if (!timeout)
            {
                ret += waitCond->returnValue;
            }
        }
        else
        {
            ret += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("Member 'setVal' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data()
            );
        }

        waitCond->deleteSemaphore();
        waitCond = NULL;
    }

    if (!PythonCommon::setReturnValueMessage(ret, "setVal", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_enableAutoGrabbing_doc,"enableAutoGrabbing() \n\
\n\
Enables auto grabbing for the grabber (camera...). \n\
\n\
If the auto grabbing flag is set, the camera acquisition is continuously triggered \n\
if at least one live image is connected to the camera. The default and minimum interval \n\
between two grabs is 20 ms. It can be changed via :meth:`setAutoGrabbingInterval`. \n\
If the grabbing process is slower, the camera tries to acquire new images as fast \n\
as possible. \n\
\n\
Enabling this auto grabbing mechanism can be undesired behaviour for instance if a \n\
measurement is started where the acquisition should be controlled by a specific \n\
script or something similar. In this case, disable the auto grabbing property. \n\
All connected live images will then get new images only if :meth:`getVal` or \n\
:meth:`copyVal` is called. \n\
\n\
This method enables the auto grabbing timer. \n\
\n\
See Also \n\
-------- \n\
setAutoGrabbing, disableAutoGrabbing, getAutoGrabbing, setAutoGrabbingInterval");

/** enable timer triggered autograbbing of a dataIO device
*   @param [in] self    the dataIO object (python)
*   @param [in] args    empty
*   @return             an error if autograbbing is not possible
*
*   For live viewing of data incoming from a dataIO device the autograbbing must be enabled. It starts a timer which triggers
*   periodically the data acquisition and pushes the new buffer to the output graph(s).
*/
PyObject *PythonPlugins::PyDataIOPlugin_enableAutoGrabbing(PyDataIOPlugin *self, PyObject * /*args*/)
{
    ito::RetVal ret = ito::retOk;
    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

    if (QMetaObject::invokeMethod(self->dataIOObj, "enableAutoGrabbing", Q_ARG(ItomSharedSemaphore*, waitCond)))
    {
        bool timeout = false;
        while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->dataIOObj->isAlive())
            {
                timeout = true;
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'enableAutoGrabbing'").toLatin1().data());
                break;
            }
        }

        if (!timeout)
        {
            ret += waitCond->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("Member 'enableAutoGrabbing' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "enableAutoGrabbing", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_disableAutoGrabbing_doc,"disableAutoGrabbing() \n\
\n\
Disables auto grabbing for this grabber (camera...). \n\
\n\
If the auto grabbing flag is set, the camera acquisition is continuously triggered \n\
if at least one live image is connected to the camera. The default and minimum interval \n\
between two grabs is 20 ms. It can be changed via :meth:`setAutoGrabbingInterval`. \n\
If the grabbing process is slower, the camera tries to acquire new images as fast \n\
as possible. \n\
\n\
Enabling this auto grabbing mechanism can be undesired behaviour for instance if a \n\
measurement is started where the acquisition should be controlled by a specific \n\
script or something similar. In this case, disable the auto grabbing property. \n\
All connected live images will then get new images only if :meth:`getVal` or \n\
:meth:`copyVal` is called. \n\
\n\
This method disables the auto grabbing timer. \n\
\n\
See Also \n\
-------- \n\
setAutoGrabbing, enableAutoGrabbing, getAutoGrabbing, setAutoGrabbingInterval");
/** disable timer triggered autograbbing of a dataIO device
*   @param [in] self    the dataIO object (python)
*   @param [in] args    empty
*   @return             an error if autograbbing could not be stopped
*
*   When the autograbbing is stopped the device will no longer periodically acquire data triggered by an internal
*   timer event. Anyway if the connected live view is still present it will receive data recorded by manual
*   acquisitions. This may be usful when a sort of live view is desired within a measurement loop where a
*   timer based autograbbing would disturbe the measurement process.
*/
PyObject *PythonPlugins::PyDataIOPlugin_disableAutoGrabbing(PyDataIOPlugin *self, PyObject * /*args*/)
{
    ito::RetVal ret = ito::retOk;
    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();

    if (QMetaObject::invokeMethod(self->dataIOObj, "disableAutoGrabbing", Q_ARG(ItomSharedSemaphore*, waitCond)))
    {
        bool timeout = false;
        while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->dataIOObj->isAlive())
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'disableAutoGrabbing'").toLatin1().data());
                timeout = true;
                break;
            }
        }

        if (!timeout)
        {
            ret += waitCond->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("Member 'disableAutoGrabbing' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "disableAutoGrabbing", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_setAutoGrabbing_doc,"setAutoGrabbing(enable) \n\
\n\
Enables or disables the auto grabbing property of this grabber device. \n\
\n\
If the auto grabbing flag is set, the camera acquisition is continuously triggered \n\
if at least one live image is connected to the camera. \n\
\n\
Enabling this auto grabbing mechanism can be undesired behaviour for instance if a \n\
measurement is started where the acquisition should be controlled by a specific \n\
script or something similar. In this case, disable the auto grabbing property. \n\
All connected live images will then get new images only if :meth:`getVal` or \n\
:meth:`copyVal` is called. \n\
\n\
Parameters \n\
---------- \n\
enable : bool \n\
    ``True`` will enable the auto grabbing timer, ``False`` disables it. \n\
\n\
See Also \n\
-------- \n\
enableAutoGrabbing, disableAutoGrabbing, getAutoGrabbing");
PyObject *PythonPlugins::PyDataIOPlugin_setAutoGrabbing(PyDataIOPlugin *self, PyObject * args)
{
    ito::RetVal ret = ito::retOk;
    bool val;

    if (!PyArg_ParseTuple(args, "b", &val))
    {
        return NULL;
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    bool invokeOk;

    if (val)
    {
        invokeOk = QMetaObject::invokeMethod(self->dataIOObj, "enableAutoGrabbing", Q_ARG(ItomSharedSemaphore*, waitCond));
    }
    else
    {
        invokeOk = QMetaObject::invokeMethod(self->dataIOObj, "disableAutoGrabbing", Q_ARG(ItomSharedSemaphore*, waitCond));
    }

    if (invokeOk)
    {
        bool timeout = false;
        while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
        {
            if (!self->dataIOObj->isAlive())
            {
                timeout = true;
                ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while calling 'enable/disableAutoGrabbing'").toLatin1().data());
                break;
            }
        }

        if (!timeout)
        {
            ret += waitCond->returnValue;
        }
    }
    else
    {
        ret += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("Member 'enableAutoGrabbing' or 'disableAutoGrabbing' of plugin could not be invoked (error in signal/slot connection).").toLatin1().data());
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "setAutoGrabbing", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_getAutoGrabbing_doc,"getAutoGrabbing() -> bool \n\
\n\
Returns if the auto grabbing property of this grabber device is enabled or disabled. \n\
\n\
If the auto grabbing flag is set, the camera acquisition is continuously triggered \n\
if at least one live image is connected to the camera. \n\
\n\
Enabling this auto grabbing mechanism can be undesired behaviour for instance if a \n\
measurement is started where the acquisition should be controlled by a specific \n\
script or something similar. In this case, disable the auto grabbing property. \n\
All connected live images will then get new images only if :meth:`getVal` or \n\
:meth:`copyVal` is called. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if the auto grabbing timer is currently active, otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
enableAutoGrabbing, disableAutoGrabbing, setAutoGrabbing");

/** return the status of the autograbbing
*   @param [in] self    the dataIO object (python)
*   @param [in] args    empty
*   @return             the status of the autograbbing
*
*   This method simply returns the status of the autograbbing.
*/
PyObject *PythonPlugins::PyDataIOPlugin_getAutoGrabbing(PyDataIOPlugin *self, PyObject * /*args*/)
{
    if (self->dataIOObj->getAutoGrabbing())
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_setAutoGrabbingInterval_doc, "setAutoGrabbingInterval() \n\
\n\
Changes the minimum auto grabbing interval (in ms) between two auto-grabbed datasets. \n\
\n\
If auto grabbing is enabled for a grabber device, a timer is set that continuously \n\
acquires data or images from the devices and sends it to all connected plots or \n\
other listeners. The timer event will occur with a certain interval (in ms). However, \n\
if the image acquisition requires more time than the interval, several timer events \n\
will be automatically omitted, such that the next image is only acquired if the \n\
grabber device is in an idle state again. Hence, the interval is considered to be a \n\
minimum value. \n\
\n\
The default interval of newly started grabber devices in 20 ms. It is possible to \n\
change this interval even if auto grabbing is currently disabled. The new interval \n\
will be considered from the next activation on. \n\
\n\
Parameters \n\
---------- \n\
interval : int\n\
    New minimum auto grabbing timer interval in `ms`. \n\
\n\
See Also \n\
-------- \n\
enableAutoGrabbing, disableAutoGrabbing, getAutoGrabbing, setAutoGrabbing, getAutoGrabbingInterval");
PyObject *PythonPlugins::PyDataIOPlugin_setAutoGrabbingInterval(PyDataIOPlugin *self, PyObject *args)
{
    ito::RetVal ret;

    int val;

    if (!PyArg_ParseTuple(args, "i", &val))
    {
        return NULL;
    }
    else if (val <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "interval must be > 0.");
        return NULL;
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QSharedPointer<int> interval(new int);
    *interval = val;

    QMetaObject::invokeMethod(
        self->dataIOObj,
        "setAutoGrabbingInterval",
        Q_ARG(QSharedPointer<int>, interval),
        Q_ARG(ItomSharedSemaphore*, waitCond));

    bool timeout = false;

    while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
    {
        if (!self->dataIOObj->isAlive())
        {
            timeout = true;
            ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while setting the current 'autoGrabbingInterval'").toLatin1().data());
            break;
        }
    }

    if (!timeout)
    {
        ret += waitCond->returnValue;
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "setAutoGrabbingInterval", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_getAutoGrabbingInterval_doc, "getAutoGrabbingInterval() -> int \n\
\n\
Returns the current auto grabbing interval (in ms), even if auto grabbing is disabled. \n\
\n\
If auto grabbing is enabled for a grabber device, a timer is set that continuously \n\
acquires data or images from the devices and sends it to all connected plots or \n\
other listeners. The timer event will occur with a certain interval (in ms). However, \n\
if the image acquisition requires more time than the interval, several timer events \n\
will be automatically omitted, such that the next image is only acquired if the \n\
grabber device is in an idle state again. Hence, the interval is considered to be a \n\
minimum value. \n\
\n\
The default interval of newly started grabber devices in 20 ms. \n\
\n\
Returns \n\
------- \n\
int \n\
    the current auto grabbing timer interval in `ms`. \n\
\n\
See Also \n\
-------- \n\
enableAutoGrabbing, disableAutoGrabbing, getAutoGrabbing, setAutoGrabbing, setAutoGrabbingInterval");
PyObject *PythonPlugins::PyDataIOPlugin_getAutoGrabbingInterval(PyDataIOPlugin *self)
{
    ito::RetVal ret;

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QSharedPointer<int> interval(new int);
    *interval = 0; //setAutoGrabbingInterval with interval=0 only returns the current interval
    QMetaObject::invokeMethod(self->dataIOObj, "setAutoGrabbingInterval", Q_ARG(QSharedPointer<int>, interval), Q_ARG(ItomSharedSemaphore*, waitCond));

    bool timeout = false;
    while (!waitCond->wait(AppManagement::timeouts.pluginGeneral))
    {
        if (!self->dataIOObj->isAlive())
        {
            timeout = true;
            ret += ito::RetVal(ito::retError, 0, QObject::tr("timeout while obtaining the current 'autoGrabbingInterval'").toLatin1().data());
            break;
        }
    }

    if (!timeout)
    {
        ret += waitCond->returnValue;
    }

    waitCond->deleteSemaphore();
    waitCond = NULL;

    if (!PythonCommon::setReturnValueMessage(ret, "getAutoGrabbingInterval", PythonCommon::invokeFunc))
    {
        return NULL;
    }

    return Py_BuildValue("i", *interval);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_connect_doc, "connect(signalSignature, callableMethod, minRepeatInterval = 0) \n\
\n\
Connects a signal of this dataIO device with the given callable Python method. \n\
\n\
Every :class:`dataIO` object can emit different signals whenever a certain event \n\
occurs. Use the method :meth:`info` to get a print-out of a list of possible signals \n\
of the dataIO device. This method is used to connect a certain callable Python callback \n\
method or function to a specific signal. The callable function can be bounded as well \n\
as unbounded. \n\
\n\
The connection is described by the string signature of the signal (hence the source of \n\
the connection). Such a signature is the name of the signal, followed by the types of \n\
its arguments (the original C++ types). An example is ``destroyed()``, \n\
emitted if this device is internally deleted. This signal can \n\
be connected to a callback function with no arguments, since the signal has no arguments, \n\
too. In case of a bounded method, the ``self`` argument must be given in any case. \n\
\n\
If the signal should have further arguments with specific datatypes, they are transformed \n\
into corresponding Python data types. A table of supported conversions is given in section \n\
:ref:`qtdesigner-datatypes`. In general, a ``callableMethod`` must be a method or \n\
function with the same number of parameters than the signal has (besides the \n\
``self`` argument). \n\
\n\
If a signal is emitted very often, it can be necessary to limit the call of the callback \n\
function to a certain minimum time interval. This can be given by the ``minRepeatInterval`` \n\
parameter. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. ``targetChanged(QVector<double>)``) \n\
callableMethod : callable \n\
    valid method or function that is called if the signal is emitted. \n\
minRepeatInterval : int, optional \n\
    If > 0, the same signal only invokes a slot once within the given interval (in ms). \n\
    Default: 0 (all signals will invoke the callable python method. \n\
\n\
See Also \n\
-------- \n\
disconnect, info");
PyObject *PythonPlugins::PyDataIOPlugin_connect(PyDataIOPlugin *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", "minRepeatInterval", NULL };
    const char* signalSignature;
    PyObject *callableMethod;
    int signalIndex;
    int tempType;
    IntList argTypes;
    int minRepeatInterval = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|i", const_cast<char**>(kwlist), &signalSignature, &callableMethod, &minRepeatInterval))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }

    if (!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    if (!self->dataIOObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of dataIO available");
        return NULL;
    }

    QByteArray signature(signalSignature);
    const QMetaObject *mo = self->dataIOObj->metaObject();
    signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature));
    QMetaMethod metaMethod = mo->method(signalIndex);
    QList<QByteArray> names = metaMethod.parameterTypes();
    foreach(const QByteArray& name, names)
    {
        tempType = QMetaType::type(name.constData());
        if (tempType > 0)
        {
            argTypes.append(tempType);
        }
        else
        {
            QString msg = QString("parameter type %1 is unknown").arg(name.constData());
            PyErr_SetString(PyExc_RuntimeError, msg.toLatin1().data());
            signalIndex = -1;
            return NULL;
        }
    }
    if (self->signalMapper)
    {
        if (!self->signalMapper->addSignalHandler(self->dataIOObj, signalSignature, signalIndex, callableMethod, argTypes, minRepeatInterval))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established. Maybe a wrong sifnature is used");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No signalMapper for this plugin could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_disconnect_doc, "disconnect(signalSignature, callableMethod) \n\
\n\
Disconnects a connection which must have been established before with exactly the same parameters.\n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. ``clicked(bool)``) \n\
callableMethod : callable \n\
    valid method or function, that should not be called any more if the \n\
    given signal is emitted. \n\
\n\
See Also \n\
-------- \n\
connect, info");
PyObject *PythonPlugins::PyDataIOPlugin_disconnect(PyDataIOPlugin *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", NULL };
    int signalIndex;
    const char* signalSignature;
    PyObject *callableMethod;
    IntList argTypes;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", const_cast<char**>(kwlist), &signalSignature, &callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }
    if (!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }
    if (!self->dataIOObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of actuator available");
        return NULL;
    }

    const QMetaObject *mo = self->dataIOObj->metaObject();
    signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature));
    QMetaMethod metaMethod = mo->method(signalIndex);
    if (self->signalMapper)
    {
        if (!self->signalMapper->removeSignalHandler(self->dataIOObj, signalIndex, callableMethod))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No signalMapper for this plugin could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_getType_doc, "getType() -> int \n\
\n\
Returns the type value of this specific dataIO plugin. \n\
\n\
Possible values are: \n\
\n\
* ``0x081``: a camera or general grabber device \n\
* ``0x101``: a analog-digital converter device \n\
* ``0x201``: any other kind of device (e.g. data transfer, like serial ports, \n\
  USB ports, ... but also other devices like a power supply...). \n\
\n\
Returns \n\
------- \n\
int \n\
    dataIO type indentifier.");
/** returns the type of the dataIO object
*   @param [in] self    the dataIO object (python)
*   @return             a string with the type
*
*   This method simply returns the type of the dataIO object
*/
PyObject* PythonPlugins::PyDataIOPlugin_getType(PyDataIOPlugin *self)
{
    PyObject *result = NULL;
    if (self == NULL || self->dataIOObj == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,"empty dataIO plugin");
        return NULL;
    }
    else
    {
        ito::AddInInterfaceBase *aib = self->dataIOObj->getBasePlugin();
        if (aib)
        {
            result = PyLong_FromLong(aib->getType());
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError,"interface of plugin is NULL");
            return NULL;
        }
    }

    return result;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyDataIOPlugin_info_doc, "info(verbose = 0) \n\
\n\
Prints out information about signal and callable slots of this actuator.\n\
\n\
Parameters \n\
---------- \n\
verbose : int \n\
    0: only slots and signals from the plugin class are printed (default) \n\
    1: all slots and signals from all inherited classes are printed\n\
\n\
See Also \n\
-------- \n\
connect, disconnect");
PyObject* PythonPlugins::PyDataIOPlugin_info(PyDataIOPlugin* self, PyObject* args)
{
    int showAll = 0;

    if (!PyArg_ParseTuple(args, "|i", &showAll))
    {
        return NULL;
    }

    if (!self->dataIOObj)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid instance of dataIO available");
        return NULL;
    }

    QStringList signalSignatureList, slotSignatureList;
    const QMetaObject *mo = self->dataIOObj->metaObject();
    QMetaMethod metaFunc;
    bool again = true;
    int methodIdx;

    if (showAll == 0 || showAll == 1)
    {
        while (again)
        {
            for (methodIdx = mo->methodOffset(); methodIdx < mo->methodCount(); ++methodIdx)
            {
                metaFunc = mo->method(methodIdx);

                if (metaFunc.methodType() == QMetaMethod::Signal)
                {
                    signalSignatureList.append(metaFunc.methodSignature());

                }

                if (metaFunc.methodType() == QMetaMethod::Slot)
                {
                    slotSignatureList.append(metaFunc.methodSignature());
                }

            }

            if (showAll == 1)
            {
                mo = mo->superClass();
                if (mo)
                {
                    again = true;
                    continue;
                }
            }
            again = false;

        }
    }
    else
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Invalid verbose level. Use level 0 to display all signals and slots defined "
            "by the plugin itself. Level 1 also displays all inherited signals and slots");

        return NULL;
    }
    signalSignatureList.sort();
    slotSignatureList.sort();

    if (signalSignatureList.length() || slotSignatureList.length())
    {
        //QByteArray val;
        QString val;
        QString previous;
        std::cout << "Signals: \n";

        foreach(val, signalSignatureList)
        {
            if (val != previous)
            {
                std::cout << "\t" << QString(val).toLatin1().data() << "\n";
            }
            previous = val;
        }

        std::cout << "\nSlots: \n";

        foreach(val, slotSignatureList)
        {
            if (val != previous)
            {
                std::cout << "\t" << QString(val).toLatin1().data() << "\n";
            }
            previous = val;
        }
    }

    Py_RETURN_NONE;
}
//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyDataIOPlugin_execFunc(PyDataIOPlugin *self, PyObject *args, PyObject *kwds)
{
    return execFunc(self->dataIOObj, args, kwds);
}

//-------------------------------------------------------------------------------------
/** open configuration dialog
*   @param [in] self    the actuator object (python)
*
*   This method simply open the configuration dialog
*/
PyObject* PythonPlugins::PyDataIOPlugin_showConfiguration(PyDataIOPlugin* self)
{
    ito::AddInBase *aib = self->dataIOObj;
    return plugin_showConfiguration(aib);
}

//-------------------------------------------------------------------------------------
/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*
*   This method simply open the widget
*/
PyObject* PythonPlugins::PyDataIOPlugin_showToolbox(PyDataIOPlugin* self)
{
    ito::AddInBase *aib = self->dataIOObj;
    return plugin_showToolbox(aib);
}

//-------------------------------------------------------------------------------------
/** returns the list of available parameters
*   @param [in] self    the actuator object (python)
*
*   This method simply close the widget
*/
PyObject* PythonPlugins::PyDataIOPlugin_hideToolbox(PyDataIOPlugin* self)
{
    ito::AddInBase *aib = self->dataIOObj;
    return plugin_hideToolbox(aib);
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyDataIOPlugin_userMutex_tryLock(PyDataIOPlugin* self, PyObject* args, PyObject* kwds)
{
    ito::AddInBase *aib = self->dataIOObj;
    return plugin_userMutexLock(aib, args, kwds, self->userMutexLocked);
}

//-------------------------------------------------------------------------------------
PyObject* PythonPlugins::PyDataIOPlugin_userMutex_unlock(PyDataIOPlugin* self)
{
    ito::AddInBase *aib = self->dataIOObj;
    return plugin_userMutexUnlock(aib, self->userMutexLocked);
}

//-------------------------------------------------------------------------------------
PyMemberDef PythonPlugins::PyDataIOPlugin_members[] = {
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMethodDef PythonPlugins::PyDataIOPlugin_methods[] = {
   {"getParamList", (PyCFunction)PythonPlugins::PyDataIOPlugin_getParamList, METH_NOARGS, pyPluginGetParamList_doc},
   {"getParamInfo", (PyCFunction)PythonPlugins::PyDataIOPlugin_getParamInfo, METH_VARARGS, pyPluginGetParamInfo_doc},
   {"getParamListInfo", (PyCFunction)PythonPlugins::PyDataIOPlugin_getParamListInfo, METH_VARARGS, pyPluginGetParamListInfo_doc},
   {"getExecFuncsList", (PyCFunction)PythonPlugins::PyDataIOPlugin_getExecFuncsList, METH_NOARGS, pyPluginGetExecFuncsList_doc },
   {"getExecFuncsInfo", (PyCFunction)PythonPlugins::PyDataIOPlugin_getExecFuncsInfo, METH_VARARGS | METH_KEYWORDS, pyPlugInGetExecFuncsInfo_doc},
   {"name", (PyCFunction)PythonPlugins::PyDataIOPlugin_name, METH_NOARGS, pyPluginName_doc},
   {"getParam", (PyCFunction)PythonPlugins::PyDataIOPlugin_getParam, METH_VARARGS, pyPluginGetParam_doc},
   {"setParam", (PyCFunction)PythonPlugins::PyDataIOPlugin_setParam, METH_VARARGS, pyPluginSetParam_doc},
   {"startDevice", (PyCFunction)PythonPlugins::PyDataIOPlugin_startDevice, METH_VARARGS, PyDataIOPlugin_startDevice_doc},
   {"stopDevice", (PyCFunction)PythonPlugins::PyDataIOPlugin_stopDevice, METH_VARARGS, PyDataIOPlugin_stopDevice_doc},
   {"acquire", (PyCFunction)PythonPlugins::PyDataIOPlugin_acquire, METH_VARARGS, PyDataIOPlugin_acquire_doc},
   {"stop", (PyCFunction)PythonPlugins::PyDataIOPlugin_stop, METH_NOARGS, PyDataIOPlugin_stop_doc},
   {"getVal", (PyCFunction)PythonPlugins::PyDataIOPlugin_getVal, METH_VARARGS, PyDataIOPlugin_getVal_doc},
   {"copyVal", (PyCFunction)PythonPlugins::PyDataIOPlugin_copyVal, METH_VARARGS, PyDataIOPlugin_copyVal_doc},
   {"setVal", (PyCFunction)PythonPlugins::PyDataIOPlugin_setVal, METH_VARARGS | METH_KEYWORDS, PyDataIOPlugin_setVal_doc},
   {"enableAutoGrabbing", (PyCFunction)PythonPlugins::PyDataIOPlugin_enableAutoGrabbing, METH_NOARGS, PyDataIOPlugin_enableAutoGrabbing_doc},
   {"disableAutoGrabbing", (PyCFunction)PythonPlugins::PyDataIOPlugin_disableAutoGrabbing, METH_NOARGS, PyDataIOPlugin_disableAutoGrabbing_doc},
   {"setAutoGrabbing", (PyCFunction)PythonPlugins::PyDataIOPlugin_setAutoGrabbing, METH_VARARGS, PyDataIOPlugin_setAutoGrabbing_doc},
   {"getAutoGrabbing", (PyCFunction)PythonPlugins::PyDataIOPlugin_getAutoGrabbing, METH_NOARGS, PyDataIOPlugin_getAutoGrabbing_doc},
   {"setAutoGrabbingInterval", (PyCFunction)PythonPlugins::PyDataIOPlugin_setAutoGrabbingInterval, METH_VARARGS, PyDataIOPlugin_setAutoGrabbingInterval_doc },
   {"getAutoGrabbingInterval", (PyCFunction)PythonPlugins::PyDataIOPlugin_getAutoGrabbingInterval, METH_NOARGS, PyDataIOPlugin_getAutoGrabbingInterval_doc },
   {"getType", (PyCFunction)PythonPlugins::PyDataIOPlugin_getType, METH_NOARGS, PyDataIOPlugin_getType_doc},
   {"exec", (PyCFunction)PythonPlugins::PyDataIOPlugin_execFunc, METH_KEYWORDS | METH_VARARGS, PyPlugin_execFunc_doc},
   {"showConfiguration", (PyCFunction)PythonPlugins::PyDataIOPlugin_showConfiguration, METH_NOARGS, pyPluginShowConfiguration_doc},
   {"showToolbox", (PyCFunction)PythonPlugins::PyDataIOPlugin_showToolbox, METH_NOARGS, pyPluginShowToolbox_doc},
   {"hideToolbox", (PyCFunction)PythonPlugins::PyDataIOPlugin_hideToolbox, METH_NOARGS, pyPluginHideToolbox_doc},
   {"connect", (PyCFunction)PythonPlugins::PyDataIOPlugin_connect, METH_VARARGS | METH_KEYWORDS, PyDataIOPlugin_connect_doc},
   {"disconnect", (PyCFunction)PythonPlugins::PyDataIOPlugin_disconnect, METH_VARARGS | METH_KEYWORDS, PyDataIOPlugin_disconnect_doc },
   { "info",(PyCFunction)PythonPlugins::PyDataIOPlugin_info, METH_VARARGS,PyDataIOPlugin_info_doc },
   {"userMutexTryLock", (PyCFunction)PythonPlugins::PyDataIOPlugin_userMutex_tryLock, METH_VARARGS |METH_KEYWORDS, PyPlugin_userMutex_tryLock_doc },
   {"userMutexUnlock", (PyCFunction)PythonPlugins::PyDataIOPlugin_userMutex_unlock, METH_NOARGS, PyPlugin_userMutex_unlock_doc },
   {NULL}  /* Sentinel */
};






//-------------------------------------------------------------------------------------
PyModuleDef PythonPlugins::PyDataIOPluginModule = {
   PyModuleDef_HEAD_INIT,
   "dataIO",
   QObject::tr("Itom dataIO plugin object").toLatin1().data(),
   -1,
   NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonPlugins::PyDataIOPluginType = {
   PyVarObject_HEAD_INIT(NULL, 0)
   "itom.dataIO",                       /* tp_name */
   sizeof(PyDataIOPlugin),              /* tp_basicsize */
   0,                                   /* tp_itemsize */
   (destructor)PyDataIOPlugin_dealloc,  /* tp_dealloc */
   0,                                   /* tp_print */
   0,                                   /* tp_getattr */
   0,                                   /* tp_setattr */
   0,                                   /* tp_reserved */
   (reprfunc)PyDataIOPlugin_repr,       /* tp_repr */
   0,                                   /* tp_as_number */
   0,                                   /* tp_as_sequence */
   0,                                   /* tp_as_mapping */
   0,                                   /* tp_hash  */
   0,                                   /* tp_call */
   0,                                   /* tp_str */
   0,                                   /* tp_getattro */
   0,                                   /* tp_setattro */
   0,                                   /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
   pyDataIOInit_doc,                    /* tp_doc */
   0,                                    /* tp_traverse */
   0,                                    /* tp_clear */
   0,                                    /* tp_richcompare */
   offsetof(PyDataIOPlugin, weakreflist),/* tp_weaklistoffset */
   0,                                    /* tp_iter */
   0,                                    /* tp_iternext */
   PyDataIOPlugin_methods,              /* tp_methods */
   PyDataIOPlugin_members,              /* tp_members */
   0,                                   /* tp_getset */
   0,                                   /* tp_base */
   0,                                   /* tp_dict */
   0,                                   /* tp_descr_get */
   0,                                   /* tp_descr_set */
   0,                                   /* tp_dictoffset */
   (initproc)PythonPlugins::PyDataIOPlugin_init,      /* tp_init */
   0,                                   /* tp_alloc */
   PyDataIOPlugin_new                   /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

/*static*/ void PythonPlugins::PyDataIOPlugin_addTpDict(PyObject *tp_dict)
{
    PyObject *value = NULL;

    //add dialog types
    value = Py_BuildValue("i", 0);
    PyDict_SetItemString(tp_dict, "TRIGGER_SOFTWARE", value);
    Py_DECREF(value);

}


} //end namespace ito

//-------------------------------------------------------------------------------------
