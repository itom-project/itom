/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#include "pythonCommon.h"

#include "pythonQtConversion.h"

#include "helper/paramHelper.h"

#include <iostream>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
/** Helper function to check and set initialisation parameters in the initialisation parameter list
*   @param [in]         tempObj python object holding the value to set
*   @param [in, out]    param   the param in the parameter list, that is set
*   @param [out]        set     indicator whether the parameter was set or not
*   @return             retOk on success, retError otherwise
*
*   The function checks if the types of the passed python parameter and the parameter are compatible and sets the parameter
*   value if it is possible. If the paramter cannot be set an error is returned.
*/
ito::RetVal checkAndSetParamVal(PyObject *tempObj, ito::Param *param, int *set)
{
    return checkAndSetParamVal(tempObj, param, *param, set);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal checkAndSetParamVal(PyObject *pyObj, const ito::Param *defaultParam, ito::ParamBase &outParam, int *set)
{
    ito::RetVal retval;
    //outParam must have same type than defaultParam
    Q_ASSERT( defaultParam->getType() == outParam.getType() );
    /*PyObject *item = NULL;*/

    switch (defaultParam->getType())
    {
    case ito::ParamBase::Char & ito::paramTypeMask:
    case ito::ParamBase::Int & ito::paramTypeMask:
        if (PyLong_Check(pyObj))
        {
            *set = 1;
            outParam.setVal<int>( PyLong_AsLong(pyObj) );
        }
        else
        {
            return ito::retError;
        }
    break;

    case ito::ParamBase::Double & ito::paramTypeMask:
        if (PyFloat_Check(pyObj))
        {
            *set = 1;
            outParam.setVal<double>(PyFloat_AsDouble(pyObj));
        }
        else if (PyErr_Clear(), PyLong_Check(pyObj))
        {
            *set = 1;
            outParam.setVal<double>(PyLong_AsDouble(pyObj));
        }
        else
        {
            return ito::retError;
        }
    break;

    case ito::ParamBase::DoubleArray & ito::paramTypeMask:
        {
            bool ok;
            QVector<double> v = PythonQtConversion::PyObjGetDoubleArray(pyObj, false, ok);
            
            if(ok)
            {
                *set = 1;
                outParam.setVal<double*>( v.data(), v.size() );
            }
            else
            {
                return ito::retError;
            }
        }
    break;

    case ito::ParamBase::IntArray & ito::paramTypeMask:
        {
            bool ok;
            QVector<int> v = PythonQtConversion::PyObjGetIntArray(pyObj, false, ok);
            
            if(ok)
            {
                *set = 1;
                outParam.setVal<int*>( v.data(), v.size() );
            }
            else
            {
                return ito::retError;
            }
        }
    break;

    case ito::ParamBase::String & ito::paramTypeMask:
        if (PyUnicode_Check(pyObj))
        {
            *set = 1;
            bool ok = false;
            QByteArray ba = PythonQtConversion::PyObjGetBytes(pyObj,false,ok);
            if(ok == false)
            {
                return ito::RetVal(ito::retError,0,"error while converting python object to string");
            }
            outParam.setVal<char *>(ba.data());
        }
        else
        {
            return ito::retError;
        }
    break;

    case ito::ParamBase::HWRef & ito::paramTypeMask:
        if (Py_TYPE(pyObj) == &PythonPlugins::PyDataIOPluginType)
        {
            *set = 1;
            outParam.setVal<void *>((void*)(((PythonPlugins::PyDataIOPlugin *)pyObj)->dataIOObj));
        }
        else if (Py_TYPE(pyObj) == &PythonPlugins::PyActuatorPluginType)
        {
            *set = 1;
            outParam.setVal<void *>((void*)(((PythonPlugins::PyActuatorPlugin *)pyObj)->actuatorObj));
        }
        else if (Py_TYPE(pyObj) == &PythonPlugins::PyAlgoPluginType)
        {
            *set = 1;
            outParam.setVal<void *>((void*)(((PythonPlugins::PyAlgoPlugin *)pyObj)->algoObj));
        }
        else
        {
            return ito::retError;
        }
    break;

    case ito::ParamBase::DObjPtr & ito::paramTypeMask:
        if ((Py_TYPE(pyObj) == &ito::PythonDataObject::PyDataObjectType))
        {
            *set = 1;
            outParam.setVal<void*>(((ito::PythonDataObject::PyDataObject *)pyObj)->dataObject);
        }
        else
        {
            return ito::retError;
        }
    break;

#if ITOM_POINTCLOUDLIBRARY > 0
    case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
        if ((Py_TYPE(pyObj) == &ito::PythonPCL::PyPointCloudType))
        {
            *set = 1;
            outParam.setVal<void*>(((ito::PythonPCL::PyPointCloud *)pyObj)->data);
        }
        else
        {
            return ito::retError;
        }
    break;

    case ito::ParamBase::PointPtr & ito::paramTypeMask:
        if ((Py_TYPE(pyObj) == &ito::PythonPCL::PyPointType))
        {
            *set = 1;
            outParam.setVal<void*>(((ito::PythonPCL::PyPoint *)pyObj)->point);
        }
        else
        {
            return ito::retError;
        }
    break;

    case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
        if ((Py_TYPE(pyObj) == &ito::PythonPCL::PyPolygonMeshType))
        {
            *set = 1;
            outParam.setVal<void*>(((ito::PythonPCL::PyPolygonMesh *)pyObj)->polygonMesh);
        }
        else
        {
            return ito::retError;
        }
    break;
#endif //ITOM_POINTCLOUDLIBRARY > 0

    default:
        return ito::RetVal(ito::retError, 0, QObject::tr("Unknown parameter type").toAscii().data());
    }

    //validate parameter (due to possible meta information)
    if(defaultParam->getMeta() != NULL)
    {
        retval += ParamHelper::validateParam(*defaultParam, outParam, true, false);

        if(retval.containsError())
        {
            *set = 0;
        }
    }

    return retval;
}



PyObject* PrntOutParams(const QVector<ito::Param> *params, bool asErr, bool addInfos, const int num)
{
    PyObject *p_pyDic = NULL;
    PyObject *p_pyLine = NULL;
    PyObject *item = NULL;
    QString type;
    QString temp;
    QMap<QString, QStringList> values;
    values["number"] = QStringList();
    values["name"] = QStringList();
    values["type"] = QStringList();
    values["values"] = QStringList();
    values["description"] = QStringList();

    p_pyDic = PyDict_New(); // new reference

    for (int n = 0; n < params->size(); n++)
    {
        if ((*params)[n].getType() != 0)
        {
            p_pyLine = PyDict_New();    // new reference

            switch(((*params)[n]).getType())
            {
                case ito::ParamBase::Char & ito::paramTypeMask:
                    type = ("char");
                break;

                case ito::ParamBase::Int & ito::paramTypeMask:
                    type = ("int");
                break;

                case ito::ParamBase::Double & ito::paramTypeMask:
                    type = ("double");
                break;

                case ito::ParamBase::String & ito::paramTypeMask:
                    type = ("string (char*)");
                break;

                case ito::ParamBase::CharArray & ito::paramTypeMask:
                    type = ("char*");
                break;

                case ito::ParamBase::IntArray & ito::paramTypeMask:
                    type = ("int*");
                break;

                case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                    type = ("double*");
                break;

                case ((ito::ParamBase::Pointer|ito::ParamBase::HWRef) & ito::paramTypeMask):
                    type = ("Plugin");
                break;

                case (ito::ParamBase::Pointer & ito::paramTypeMask):
                    type = ("void*");
                break;

                case (ito::ParamBase::DObjPtr & ito::paramTypeMask):
                    type = ("dataObject");
                break;

                case (ito::ParamBase::PointCloudPtr & ito::paramTypeMask):
                    type = ("pointCloud");
                break;

                case (ito::ParamBase::PointPtr & ito::paramTypeMask):
                    type = ("point");
                break;

                case (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask):
                    type = ("polygonMesh");
                break;

                default:
                    type = ("type error");
                break;
            }

            values["type"].append( type );
            temp = QString::number(n+1) + ".";
            values["number"].append( temp );
            values["name"].append( ((*params)[n]).getName() );

            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(((*params)[n]).getName());
            PyDict_SetItemString(p_pyLine, "name", item);
            Py_DECREF(item);

            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(type.toAscii());
            PyDict_SetItemString(p_pyLine, "type", item);
            Py_DECREF(item);

            item = PyLong_FromLong(n);
            PyDict_SetItemString(p_pyLine, "index", item);
            Py_DECREF(item);

            if(addInfos)
            {
                char* tempinfobuf = NULL;
                tempinfobuf = const_cast<char*>(((*params)[n]).getInfo());
                if(tempinfobuf)
                {
                    temp = QString(tempinfobuf);
                    values["description"].append( temp );
                }
                else
                {
                    values["description"].append( "<no description>" );
                }

                switch(((*params)[n]).getType())
                {
                    case ito::ParamBase::Char & ito::paramTypeMask:
                    case ito::ParamBase::Int & ito::paramTypeMask:
                        {
                        const ito::IntMeta *intMeta = static_cast<const ito::IntMeta*>((*params)[n].getMeta());
                        int mi, ma;
                        if(intMeta)
                        {
                            mi = intMeta->getMin();
                            ma = intMeta->getMax();
                        }
                        else
                        {
                            const ito::CharMeta *charMeta = static_cast<const ito::CharMeta*>((*params)[n].getMeta());
                            if(charMeta)
                            {
                                mi = static_cast<int>( charMeta->getMin() );
                                ma = static_cast<int>( charMeta->getMax() );
                            }
                            else
                            {
                                mi = std::numeric_limits<int>::min();
                                ma = std::numeric_limits<int>::max();
                            }
                        }
                        int va =  ((*params)[n]).getVal<int>();

                        temp = QString("current: %1, [%2,%3]").arg(va).arg(mi).arg(ma);
                        values["values"].append(temp);

                        item = PyLong_FromLong(mi);
                        PyDict_SetItemString(p_pyLine, "min", item);
                        Py_DECREF(item);
                        item = PyLong_FromLong(ma);
                        PyDict_SetItemString(p_pyLine, "max", item);
                        Py_DECREF(item);
                        item = PyLong_FromLong(va);
                        PyDict_SetItemString(p_pyLine, "value", item);
                        Py_DECREF(item);
                        }
                    break;

                    case ito::ParamBase::Double & ito::paramTypeMask:
                        {
                        const ito::DoubleMeta *dblMeta = static_cast<const ito::DoubleMeta*>((*params)[n].getMeta());
                        double mi, ma;
                        if(dblMeta)
                        {
                            mi = dblMeta->getMin();
                            ma = dblMeta->getMax();
                        }
                        else
                        {
                            ma = std::numeric_limits<double>::max();
                            mi = -ma;
                        }
                        double va =  ((*params)[n]).getVal<double>();

                        temp = QString("current: %1, [%2,%3]").arg(va).arg(mi).arg(ma);
                        values["values"].append(temp);

                        item = PyFloat_FromDouble(mi);
                        PyDict_SetItemString(p_pyLine, "min", item);
                        Py_DECREF(item);

                        item = PyFloat_FromDouble(ma);
                        PyDict_SetItemString(p_pyLine, "max", item);
                        Py_DECREF(item);

                        item = PyFloat_FromDouble(va);
                        PyDict_SetItemString(p_pyLine, "value", item);
                        Py_DECREF(item);
                        }
                    break;

                    case (ito::ParamBase::String & ito::paramTypeMask):
                    {
                        char* tempbuf = ((*params)[n]).getVal<char*>();
                        if(tempbuf == NULL)
                        {
                            item = PyUnicode_FromString("");
                            values["values"].append("");
                            PyDict_SetItemString(p_pyLine, "value", item);
                            Py_DECREF(item);
                        }
                        else
                        {
                            temp = tempbuf;
                            if(temp.size() > 20)
                            {
                                temp = QString("\"%1...\"").arg( temp.left(20) );
                            }
                            else
                            {
                                temp = QString("\"%1\"").arg(temp);
                            }
                            values["values"].append( temp );
                            item = PythonQtConversion::QByteArrayToPyUnicodeSecure( tempbuf );
                            PyDict_SetItemString(p_pyLine, "value", item);
                            Py_DECREF(item);
                        }
                    }
                    break;

                    case ito::ParamBase::CharArray & ito::paramTypeMask:
                    case ito::ParamBase::IntArray & ito::paramTypeMask:
                    case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                        temp = QString("%1 elements").arg( QString::number(((*params)[n]).getLen()));
                        values["values"].append( temp );
                    break;

                    case ((ito::ParamBase::Pointer | ito::ParamBase::HWRef) & ito::paramTypeMask):
                    case (ito::ParamBase::Pointer & ito::paramTypeMask):
                    case (ito::ParamBase::DObjPtr & ito::paramTypeMask):
                    case (ito::ParamBase::PointCloudPtr & ito::paramTypeMask):
                    case (ito::ParamBase::PointPtr & ito::paramTypeMask):
                    case (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask):
                        values["values"].append( "<Object-Pointer>" );
                    break;

                    default:
                        values["values"].append("<unknown>");
                    break;

                }
            }
            if(((*params)[n]).getInfo())
            {
                item = PythonQtConversion::QByteArrayToPyUnicodeSecure( ((*params)[n]).getInfo() );
                PyDict_SetItemString(p_pyLine, "info", item);
                Py_DECREF(item);
            }
            
            PyDict_SetItemString(p_pyDic, (*params)[n].getName(), p_pyLine);    // Add reference
            Py_DECREF(p_pyLine);    // Reduce reference
        }
    }

    //now construct final output
    int numberLength = 2;
    int nameLength = 4;
    int typeLength = 4;
    int valuesLength = 5;
    QString output;

    foreach(const QString &str, values["number"])
    {
        numberLength = qMax(numberLength, str.length());
    }
    foreach(const QString &str, values["name"])
    {
        nameLength = qMax(nameLength, str.length());
    }
    foreach(const QString &str, values["type"])
    {
        typeLength = qMax(typeLength, str.length());
    }
    foreach(const QString &str, values["values"])
    {
        valuesLength = qMax(valuesLength, str.length());
    }

    //truncate length by max-max-value
    nameLength = qMin(nameLength, 50);
    valuesLength = qMin(valuesLength, 50);

    numberLength += 1;
    nameLength += 1;
    typeLength += 1;
    valuesLength += 2;

    // write a heading
    if(asErr)
    {
        output.append("#");
    }
    else
    {
        output.append("'"); //mark as unclosed string
    }

    temp = QString("No").leftJustified(numberLength,' ');
    output.append( temp );
    temp = QString("Name").leftJustified(nameLength,' ');
    output.append( temp );
    temp = QString("type").leftJustified(typeLength,' ');
    output.append( temp );
    if(addInfos)
    {
        temp = QString("value").leftJustified(valuesLength,' ');
        output.append( temp );
        output.append("description");
    }
    output.append("\n");

    for(int i=0;i<values["number"].length();i++)
    {
        if(asErr)
        {
            output.append("#");
        }
        else
        {
            output.append("'"); //mark as unclosed string
        }
        temp = values["number"][i].leftJustified(numberLength,' ', true);
        output.append( temp );
        temp = values["name"][i].leftJustified(nameLength,' ');
        output.append( temp );
        temp = values["type"][i].leftJustified(typeLength,' ', true);
        output.append( temp );
        if(addInfos)
        {
            temp = values["values"][i].leftJustified(valuesLength,' ', true);
            output.append( temp );
            output.append( values["description"][i] );
        }

        if(num == i)
        {
            output.append( " <-- erroneous parameter");
        }
        output.append("\n");
    }

    if(asErr)
    {
        std::cerr << output.toAscii().data() << std::endl;
    }
    else
    {
        std::cout << output.toAscii().data() << std::endl;
    }

    return p_pyDic;
}















//----------------------------------------------------------------------------------------------------------------------------------
/** Helper function for error output
*   @param [in] params  parameters expected by the plugin
*   @param [in] num     parameter where the error occured
*   @param [in] reason  the reason for the error (e.g. parameter missing, wrong type, ...)
*
*   Function used for writing error messages occured during the parsing of the parameters passed for the initialisation
*   of a plugin. The function uses the cerr stream to "post" the error message. If possible the parameter where the error
*   occured is marked with an arrow. Except the error all parameters necessary and optional including their type are written
*   to the console.
*/
void errOutInitParams(const QVector<ito::Param> *params, const int num, const QString reason)
{
    PyErr_Print();
    std::cerr << "\n";
    std::cerr << reason.toAscii().data() << "\n";
    if (params)
    {
        PyObject* dummy = PrntOutParams(params, true, false, num);
        Py_DecRef(dummy);
    }
    else
    {
        std::cerr << "Plugin does not accept parameters!" << "\n";
    }
    std::cerr << "\n";
    PyErr_Print();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** Function to read mandatory and optional parameter lists from a given python parameter list according to the plugins definition
*   @param [in, out]    initParamListMand   vector holding the mandatory initialisation parameters (filled with default values),
*                                           the default values are overwritten with the passed values
*   @param [in, out]    initParamListOpt    vector holding the optional initialisation parameters (filled with default values),
*                                           the default values are overwritten with the passed values
*   @param [in]         args                list with python arguments
*   @param [in]         kwds                list with named python arguments
*   @return             returns ito::retOk if the number and type of parameters was correct, ito::retError otherwise
*
*   The function takes as input the vectors with the madatory and optional input parameters used for the plugin initialisation. These
*   vectors must be previously be read using the function \ref getInitParams. The default values of the parameters are overwritten with
*   the values given by python. In case the number or parameters or a parameter type is incorrect the function will abort with an error.
*/
ito::RetVal parseInitParams(QVector<ito::Param> *initParamListMand, QVector<ito::Param> *initParamListOpt, PyObject *args, PyObject *kwds)
{
    int len;
    int numMandParams = initParamListMand == NULL ? 0 : initParamListMand->size();
    int numOptParams = initParamListOpt == NULL ? 0 : initParamListOpt->size();
    int *mandPParsed = (int*)calloc(numMandParams, sizeof(int));
    int *optPParsed = (int*)calloc(numOptParams, sizeof(int));
    int argsLen = 0;
    int kwdsLen = 0;
    int mandKwd = 0;
    PyObject *tempObj = NULL;

    if (args != NULL)
    {
        argsLen = PyTuple_Size(args);
    }
    if (kwds != NULL)
    {
        kwdsLen = PyDict_Size(kwds);
    }

    // Check if number of given parameters is in an acceptable range
    if (((argsLen + kwdsLen) < numMandParams)
        || ((argsLen + kwdsLen) > (numMandParams + numOptParams)))
    {
            errOutInitParams(initParamListMand, -1, "wrong number of parameters. Mandatory parameters are:");
            errOutInitParams(initParamListOpt, -1, "optional parameters are:");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
    }

    len = argsLen > numMandParams ? numMandParams : argsLen;

    // Check if paramters are passed as arg and keyword
    if (kwds != NULL)
    {
        for (int n = 0; n < len; n++)
        {
            const char *tkey = (*initParamListMand)[n].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                if (mandPParsed)
                    free(mandPParsed);
                if (optPParsed)
                    free(optPParsed);
                return ito::RetVal::format(ito::retError,0,"parameter %d - %s passed as arg and keyword!",n,tkey);
            }
        }
        for (int n = len; n < argsLen; n++)
        {
            const char *tkey = (*initParamListOpt)[n - len].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                if (mandPParsed)
                    free(mandPParsed);
                if (optPParsed)
                    free(optPParsed);
                return ito::RetVal::format(ito::retError,0,"optional parameter %d - %s passed as arg and keyword!",n,tkey);
            }
        }
    }

    // argsLen ist not sufficient for mandatory parameters so check if we can complete with keywords
    if (argsLen < numMandParams)
    {
        for (int n = argsLen; n < numMandParams; n++)
        {
            const char *tkey = (*initParamListMand)[n].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                mandKwd++;
            }
        }
        if ((argsLen + mandKwd) < numMandParams)
        {
            errOutInitParams(initParamListMand, -1, "wrong number of parameters\n Mandatory parameters are:\n");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }

    // read in mandatory parameters
    for (int n = 0; n < len; n++)
    {
        tempObj = PyTuple_GetItem(args, n);
        if (checkAndSetParamVal(tempObj, &((*initParamListMand)[n]), &(mandPParsed[n])) != ito::retOk)
        {
            errOutInitParams(initParamListMand, n, "wrong parameter type");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }
    for (int n = 0; n < mandKwd; n++)
    {
        const char *tkey = (*initParamListMand)[len + n].getName();
        tempObj = PyDict_GetItemString(kwds, tkey);
        if (checkAndSetParamVal(tempObj, &((*initParamListMand)[n + len]), &(mandPParsed[n + len])) != ito::retOk)
        {
            errOutInitParams(initParamListMand, n, "wrong parameter type");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }

    // read in remaining (optional) parameters
    for (int n = numMandParams; n < argsLen; n++)
    {
        tempObj = PyTuple_GetItem(args, n);
        if (checkAndSetParamVal(tempObj, &((*initParamListOpt)[n - numMandParams]), &(optPParsed[n - numMandParams])) != ito::retOk)
        {
            errOutInitParams(initParamListOpt, n-numMandParams, "wrong parameter type");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }
    if (kwds)
    {
        for (int n = 0; n < numOptParams; n++)
        {
            const char *tkey = (*initParamListOpt)[n].getName();
            tempObj = PyDict_GetItemString(kwds, tkey);
            if (tempObj)
            {
                if (checkAndSetParamVal(tempObj, &((*initParamListOpt)[n]), &(optPParsed[n])) != ito::retOk)
                {
                    errOutInitParams(initParamListOpt, n, "wrong parameter type");
                    if (mandPParsed)
                        free(mandPParsed);
                    if (optPParsed)
                        free(optPParsed);
                    return ito::retError;
                }
            }
        }
    }

    if (mandPParsed)
        free(mandPParsed);
    if (optPParsed)
        free(optPParsed);

    return ito::retOk;
}


ito::RetVal parseInitParams(const QVector<ito::Param> *defaultParamListMand, const QVector<ito::Param> *defaultParamListOpt, PyObject *args, PyObject *kwds, QVector<ito::ParamBase> &paramListMandOut, QVector<ito::ParamBase> &paramListOptOut)
{
    int len;
    int numMandParams = defaultParamListMand == NULL ? 0 : defaultParamListMand->size();
    int numOptParams = defaultParamListOpt == NULL ? 0 : defaultParamListOpt->size();

    paramListMandOut.clear();
    paramListOptOut.clear();

    int *mandPParsed = (int*)calloc(numMandParams, sizeof(int));
    int *optPParsed = (int*)calloc(numOptParams, sizeof(int));
    int argsLen = 0;
    int kwdsLen = 0;
    int mandKwd = 0;
    PyObject *tempObj = NULL;

    if (args != NULL)
    {
        argsLen = PyTuple_Size(args);
    }
    if (kwds != NULL)
    {
        kwdsLen = PyDict_Size(kwds);
    }

    // Check if number of given parameters is in an acceptable range
    if (((argsLen + kwdsLen) < numMandParams)
        || ((argsLen + kwdsLen) > (numMandParams + numOptParams)))
    {
            errOutInitParams(defaultParamListMand, -1, "wrong number of parameters. Mandatory parameters are:");
            errOutInitParams(defaultParamListOpt, -1, "optional parameters are:");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
    }

    len = argsLen > numMandParams ? numMandParams : argsLen;

    // Check if parameters are passed as arg and keyword
    if (kwds != NULL)
    {
        for (int n = 0; n < len; n++)
        {
            const char *tkey = (*defaultParamListMand)[n].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                if (mandPParsed)
                    free(mandPParsed);
                if (optPParsed)
                    free(optPParsed);
                return ito::RetVal::format(ito::retError,0,"parameter %d - %s passed as arg and keyword!",n,tkey);
            }
        }
        for (int n = len; n < argsLen; n++)
        {
            const char *tkey = (*defaultParamListOpt)[n - len].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                if (mandPParsed)
                    free(mandPParsed);
                if (optPParsed)
                    free(optPParsed);
                return ito::RetVal::format(ito::retError,0,"optional parameter %d - %s passed as arg and keyword!",n,tkey);
            }
        }
    }

    // check if any key is given, which does not exist in kwds-dictionary
    Py_ssize_t foundKwds = 0;
    foreach(const ito::Param p, *defaultParamListMand)
    {
        if ( PyDict_GetItemString(kwds, p.getName() ) ) 
        {
            foundKwds++;
        }
    }
    foreach(const ito::Param p, *defaultParamListOpt)
    {
        if ( PyDict_GetItemString(kwds, p.getName() ) ) 
        {
            foundKwds++;
        }
    }

    //this is a keyword-parameter, that can be passed without being part of the mandatory or optional parameters
    if(PyDict_GetItemString(kwds, "autoLoadParams"))
    {
        foundKwds++;
    }

    if(foundKwds != PyDict_Size(kwds))
    {
        if (mandPParsed) 
            free(mandPParsed);
        if (optPParsed)  
            free(optPParsed);
        std::cerr << "there are keyword arguments that does not exist in mandatory or optional parameters." << std::endl;
        errOutInitParams(defaultParamListMand, -1, "Mandatory parameters are:");
        errOutInitParams(defaultParamListOpt, -1, "Optional parameters are:");
        return ito::RetVal(ito::retError,0,"there are keyword arguments that does not exist in mandatory or optional parameters.");
    }


    // argsLen ist not sufficient for mandatory parameters so check if we can complete with keywords
    if (argsLen < numMandParams)
    {
        for (int n = argsLen; n < numMandParams; n++)
        {
            const char *tkey = (*defaultParamListMand)[n].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                mandKwd++;
            }
        }
        if ((argsLen + mandKwd) < numMandParams)
        {
            errOutInitParams(defaultParamListMand, -1, "wrong number of parameters\n Mandatory parameters are:\n");
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }

    //create default out-vectors
    copyParamVector(defaultParamListMand, paramListMandOut);
    copyParamVector(defaultParamListOpt, paramListOptOut);

    ito::RetVal retval;

    // read in mandatory parameters
    for (int n = 0; n < len; n++)
    {
        tempObj = PyTuple_GetItem(args, n);
        retval = checkAndSetParamVal(tempObj, &((*defaultParamListMand)[n]), paramListMandOut[n], &(mandPParsed[n]));
        if(retval.containsError())
        {
            if(retval.errorMessage() == NULL)
            {
                errOutInitParams(defaultParamListMand, n, "wrong parameter type");
            }
            else
            {
                errOutInitParams(defaultParamListMand, n, retval.errorMessage());
            }
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }

    for (int n = 0; n < mandKwd; n++)
    {
        const char *tkey = (*defaultParamListMand)[len + n].getName();
        tempObj = PyDict_GetItemString(kwds, tkey);
        
        retval = checkAndSetParamVal(tempObj, &((*defaultParamListMand)[n + len]), paramListMandOut[n + len], &(mandPParsed[n + len]));
        if(retval.containsError())
        {
            if(retval.errorMessage() == NULL)
            {
                errOutInitParams(defaultParamListMand, n, "wrong parameter type");
            }
            else
            {
                errOutInitParams(defaultParamListMand, n, retval.errorMessage());
            }
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }

    // read in remaining (optional) parameters
    for (int n = numMandParams; n < argsLen; n++)
    {
        tempObj = PyTuple_GetItem(args, n);

        retval = checkAndSetParamVal(tempObj, &((*defaultParamListOpt)[n - numMandParams]), paramListOptOut[n - numMandParams], &(optPParsed[n - numMandParams]));
        if(retval.containsError())
        {
            if(retval.errorMessage() == NULL)
            {
                errOutInitParams(defaultParamListOpt, n, "wrong parameter type");
            }
            else
            {
                errOutInitParams(defaultParamListOpt, n, retval.errorMessage());
            }
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::retError;
        }
    }
    if (kwds)
    {
        for (int n = 0; n < numOptParams; n++)
        {
            const char *tkey = (*defaultParamListOpt)[n].getName();
            tempObj = PyDict_GetItemString(kwds, tkey);
            if (tempObj)
            {
                if (checkAndSetParamVal(tempObj, &((*defaultParamListOpt)[n]), paramListOptOut[n], &(optPParsed[n])) != ito::retOk)
                {
                    errOutInitParams(defaultParamListOpt, n, "wrong parameter type");
                    if (mandPParsed)
                        free(mandPParsed);
                    if (optPParsed)
                        free(optPParsed);
                    return ito::retError;
                }
            }
        }
    }

    if (mandPParsed)
        free(mandPParsed);
    if (optPParsed)
        free(optPParsed);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** makes a deep copy of a vector with values of type ParamBase
*   
*   @param [in]     paramVecIn is a pointer to a vector of ParamBase-values
*   @param [out]    paramVecOut is a reference to a vector which is first cleared and then filled with a deep copy of every element of paramVecIn
*/
ito::RetVal copyParamVector(const QVector<ito::ParamBase> *paramVecIn, QVector<ito::ParamBase> &paramVecOut)
{
    if(paramVecIn)
    {
        paramVecOut.clear();
        for(int i=0;i<paramVecIn->size();i++)
        {
            paramVecOut.append( ito::ParamBase( paramVecIn->value(i) ) );
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError,0,"paramVecIn is NULL");
}

//----------------------------------------------------------------------------------------------------------------------------------
/** makes a deep copy of a vector with values of type Param
*   
*   @param [in]     paramVecIn is a pointer to a vector of Param-values
*   @param [out]    paramVecOut is a reference to a vector which is first cleared and then filled with a deep copy of every element of paramVecIn
*/
ito::RetVal copyParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::Param> &paramVecOut)
{
    if(paramVecIn)
    {
        paramVecOut.clear();
        for(int i=0;i<paramVecIn->size();i++)
        {
            paramVecOut.append( ito::Param( paramVecIn->value(i) ) );
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError,0,"paramVecIn is NULL");
}

ito::RetVal copyParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::ParamBase> &paramVecOut)
{
    if(paramVecIn)
    {
        paramVecOut.clear();
        for(int i=0;i<paramVecIn->size();i++)
        {
            paramVecOut.append( ito::ParamBase( paramVecIn->value(i) ) );
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError,0,"paramVecIn is NULL");
}


ito::RetVal createEmptyParamBaseFromParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::ParamBase> &paramVecOut)
{
    if(paramVecIn)
    {
//        const ito::Param temp;
        paramVecOut.clear();
        for(int i=0;i<paramVecIn->size();i++)
        {
//            temp = (paramVecIn->value(i));
            paramVecOut.append( ito::ParamBase(paramVecIn->value(i).getName(), paramVecIn->value(i).getType()) );
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError,0,"paramVecIn is NULL");
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param [in|out]  kwds                   list with named python arguments
*   @param [out]     enableAutoLoadParams   if keyword autoLoadParams is found, value of this is set to kwds-item value else false it is set to false.
*
*/
ito::RetVal findAndDeleteReservedInitKeyWords(PyObject *kwds, bool * enableAutoLoadParams)
{
    * enableAutoLoadParams = false;
    if(kwds)
    {
        if(PyDict_GetItemString(kwds, "autoLoadParams"))
        {
            if (PyLong_Check(PyDict_GetItemString(kwds, "autoLoadParams")))
            {
                *enableAutoLoadParams = (bool)(PyLong_AsLong(PyDict_GetItemString(kwds, "autoLoadParams")));
            }
            else
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("Keyword autoLoadParams not of integer type").toAscii().data());
            }
            if(PyDict_DelItemString(kwds, "autoLoadParams"))
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("Could not delete Keyword: autoLoadParams").toAscii().data());
            }
        }
    }
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------
//PyObject* transformQVariant2PyObject(QVariant *value, ito::RetVal &retValue)
//{
//    retValue += ito::retOk;
//
//    switch(value->type())
//    {
//    case QVariant::Invalid:
//        retValue += ito::RetVal(ito::retError, 0, QObject::tr("invalid return value").toAscii().data());
//        return NULL;
//        break;
//    case QVariant::Bool:
//        if(value->toBool() == true)
//        {
//            Py_INCREF(Py_True);
//            return Py_True;
//        }
//        else
//        {
//            Py_INCREF(Py_False);
//            return Py_False;
//        }
//        break;
//    case QVariant::Int:
//        return Py_BuildValue("i",value->toInt());
//        break;
//    case QVariant::Double:
//        return Py_BuildValue("d",value->toDouble());
//        break;
//    case QVariant::String:
//        return Py_BuildValue("s", value->toString().toAscii().data());
//        break;
//    default:
//        retValue += ito::RetVal(ito::retError, 0, QObject::tr("unknown parameter of type QVariant").toAscii().data());
//        return NULL;
//        break;
//    }
//}

//------------------------------------------------------------------------------------------------------------------
PyObject* buildFilterOutputValues(QVector<QVariant> *outVals, ito::RetVal &retValue)
{
    PyObject *tuple = NULL;
    QVariant *elem;

    if(outVals->size() <= 0)
    {
        retValue += ito::RetVal(ito::retOk);
        Py_RETURN_NONE;
    }
    else if(outVals->size() == 1)
    {
        elem = &(outVals->data()[0]);
        //tuple = transformQVariant2PyObject(elem, retValue);
        tuple = PythonQtConversion::QVariantToPyObject(*elem);
        if(tuple == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "unknown parameter of type QVariant");
        }
    }
    else
    {
        tuple = PyTuple_New(outVals->size());
        PyObject *temp = NULL;
        for(int i=0;i <outVals->size();i++)
        {
            elem = &(outVals->data()[i]);
            temp = PythonQtConversion::QVariantToPyObject(*elem);
            if(temp == NULL)
            {
                PyErr_SetString(PyExc_RuntimeError, "unknown parameter of type QVariant");
            }
            else
            {
                PyTuple_SetItem(tuple,i, temp); //steals reference
            }
        }
    }

    return tuple;

}

//------------------------------------------------------------------------------------------------------------------
bool PythonCommon::transformRetValToPyException(ito::RetVal &retVal, PyObject *exceptionIfError)
{
    QString baseMsg, msg;
    if(retVal.containsWarningOrError())
    {
        char *temp = retVal.errorMessage();
        if(temp == NULL)
        {
            msg = QObject::tr("- unknown message -");
        }
        else
        {
            msg = retVal.errorMessage();
        }

        if(retVal.containsError())
        {
            PyErr_Format(exceptionIfError, msg.toAscii().data());
            return false;
        }
        else
        {
            std::cout << "Warning: " << msg.toAscii().data() << std::endl;
        }
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------
} //end namespace ito
