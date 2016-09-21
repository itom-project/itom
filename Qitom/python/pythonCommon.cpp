/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "pythonCommon.h"

#include "pythonRgba.h"
#include "pythonQtConversion.h"
#include "pythonPlugins.h"
#include "pythonDataObject.h"
#include "pythonPCL.h"

#include "helper/paramHelper.h"
#include "../common/numeric.h"

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
    Q_ASSERT(defaultParam->getType() == outParam.getType());
    /*PyObject *item = NULL;*/

    switch (defaultParam->getType())
    {
    case ito::ParamBase::Char & ito::paramTypeMask:
    case ito::ParamBase::Int & ito::paramTypeMask:
        if (PyRgba_Check(pyObj))
        {
            PythonRgba::PyRgba *pyRgba = (PythonRgba::PyRgba*)(pyObj);
            outParam.setVal<int>((int)(pyRgba->rgba.rgba));
        }
        else
        {
            bool ok;
            outParam.setVal<int>(PythonQtConversion::PyObjGetInt(pyObj, false, ok));
            if (ok)
            {
                *set = 1;
            }
            else
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value could not be converted to integer").toLatin1().data());
            }
        }
    break;

    case ito::ParamBase::Double & ito::paramTypeMask:
        {
            bool ok;
            outParam.setVal<double>(PythonQtConversion::PyObjGetDouble(pyObj, false, ok));
            if (ok)
            {
                *set = 1;
            }
            else
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value could not be converted to double").toLatin1().data());
            }
        }
    break;

    case ito::ParamBase::Complex & ito::paramTypeMask:
        {
            bool ok;
            outParam.setVal<ito::complex128>(PythonQtConversion::PyObjGetComplex(pyObj, false, ok));
            if (ok)
            {
                *set = 1;
            }
            else
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value could not be converted to complex").toLatin1().data());
            }
        }
    break;

    case ito::ParamBase::CharArray & ito::paramTypeMask:
        {
            if (PyByteArray_Check(pyObj))
            {
                char *buf  = (char *)PyByteArray_AsString(pyObj);
                Py_ssize_t listlen = PyByteArray_Size(pyObj);
                outParam.setVal<char*>(buf, listlen);
            }
            else
            {
                return ito::retError;
            }
        }
    break;

    case ito::ParamBase::DoubleArray & ito::paramTypeMask:
        {
            bool ok;
            QVector<double> v = PythonQtConversion::PyObjGetDoubleArray(pyObj, false, ok);
            
            if (ok)
            {
                *set = 1;
                outParam.setVal<double*>(v.data(), v.size());
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
            
            if (ok)
            {
                *set = 1;
                outParam.setVal<int*>(v.data(), v.size());
            }
            else
            {
                return ito::retError;
            }
        }
    break;

    case ito::ParamBase::ComplexArray & ito::paramTypeMask:
        {
            bool ok;
            QVector<ito::complex128> v = PythonQtConversion::PyObjGetComplexArray(pyObj, false, ok);
            
            if (ok)
            {
                *set = 1;
                outParam.setVal<ito::complex128*>(v.data(), v.size());
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
            if (ok == false)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("error while converting python object to string").toLatin1().data());
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
        return ito::RetVal(ito::retError, 0, QObject::tr("Unknown parameter type").toLatin1().data());
    }

    //validate parameter (due to possible meta information)
    if (defaultParam->getMeta() != NULL)
    {
        retval += ParamHelper::validateParam(*defaultParam, outParam, true, false);

        if (retval.containsError())
        {
            *set = 0;
        }
    }

    return retval;
}


//--------------------------------------------------------------------------------------------------------------
PyObject* PrntOutParams(const QVector<ito::Param> *params, bool asErr, bool addInfos, const int num, bool printToStdStream /*= true*/)
{
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
    values["readwrite"] = QStringList();
    bool readonly;
    const ito::ParamMeta *meta = NULL;

    PyObject *pVector = PyTuple_New( params->size() ); // new reference

    for (int n = 0; n < params->size(); n++)
    {
        const ito::Param &p = params->at(n);
        meta = p.getMeta();
        ito::ParamMeta::MetaRtti metaType = meta ? meta->getType() : ito::ParamMeta::rttiUnknown;

        if (p.getType() != 0)
        {
            p_pyLine = PyDict_New();    // new reference

            if (p.getFlags() & ito::ParamBase::Readonly)
            {
                values["readwrite"].append("r");
                readonly = true;
            }
            else
            {
                values["readwrite"].append("rw");
                readonly = false;
            }

            switch(p.getType())
            {
                case ito::ParamBase::Char & ito::paramTypeMask:
                    type = ("int (char)");
                break;

                case ito::ParamBase::Int & ito::paramTypeMask:
                    type = ("int");
                break;

                case ito::ParamBase::Double & ito::paramTypeMask:
                    type = ("float");
                break;

                case ito::ParamBase::Complex & ito::paramTypeMask:
                    type = ("complex");
                break;

                case ito::ParamBase::String & ito::paramTypeMask:
                    type = ("str");
                break;

                case ito::ParamBase::CharArray & ito::paramTypeMask:
                    type = ("seq. of int (char)");
                break;

                case ito::ParamBase::IntArray & ito::paramTypeMask:
                    switch (metaType)
                    {
                    case ito::ParamMeta::rttiIntervalMeta:
                        type = "int interval [v1,v2]";
                        break;
                    case ito::ParamMeta::rttiRangeMeta:
                        type = "int range [v1,v2)";
                        break;
                    case ito::ParamMeta::rttiRectMeta:
                        type = "int rect [x0,y0,width,height]";
                        break;
                    default:
                        type = ("seq. of int");
                    }
                break;

                case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                    switch (metaType)
                    {
                    case ito::ParamMeta::rttiDoubleIntervalMeta:
                        type = "float interval [v1,v2]";
                        break;
                    default:
                        type = ("seq. of float");
                    }
                break;

                case ito::ParamBase::ComplexArray & ito::paramTypeMask:
                    type = ("seq. of complex");
                break;

                case ((ito::ParamBase::Pointer|ito::ParamBase::HWRef) & ito::paramTypeMask):
                    type = ("dataIO|actuator");
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
                    type = ("unknown type");
                break;
            }

            values["type"].append(type);
            temp = QString::number(n+1) + ".";
            values["number"].append(temp);
            values["name"].append(p.getName());

            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(p.getName());
            PyDict_SetItemString(p_pyLine, "name", item);
            Py_DECREF(item);

            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(type.toLatin1());
            PyDict_SetItemString(p_pyLine, "type", item);
            Py_DECREF(item);

            item = PyLong_FromLong(n);
            PyDict_SetItemString(p_pyLine, "index", item);
            Py_DECREF(item);

            PyDict_SetItemString(p_pyLine, "readonly", readonly ? Py_True : Py_False);

            if (addInfos)
            {
                char* tempinfobuf = NULL;
                tempinfobuf = const_cast<char*>(p.getInfo());
                if (tempinfobuf)
                {
                    temp = QString::fromLatin1(tempinfobuf);
                    values["description"].append(temp);
                }
                else
                {
                    values["description"].append("<no description>");
                }

                switch(p.getType())
                {
                    case ito::ParamBase::Char & ito::paramTypeMask:
                    case ito::ParamBase::Int & ito::paramTypeMask:
                        {
                        const ito::IntMeta *intMeta = static_cast<const ito::IntMeta*>(meta);
                        int mi, ma, step;
                        if (intMeta)
                        {
                            mi = intMeta->getMin();
                            ma = intMeta->getMax();
                            step = intMeta->getStepSize();
                        }
                        else
                        {
                            const ito::CharMeta *charMeta = static_cast<const ito::CharMeta*>(meta);
                            if (charMeta)
                            {
                                mi = static_cast<int>(charMeta->getMin());
                                ma = static_cast<int>(charMeta->getMax());
                                step = static_cast<int>(charMeta->getStepSize());
                            }
                            else
                            {
                                mi = std::numeric_limits<int>::min();
                                ma = std::numeric_limits<int>::max();
                                step = 1;
                            }
                        }
                        int va =  p.getVal<int>();

                        if (mi == std::numeric_limits<int>::min() && ma == std::numeric_limits<int>::max() && step == 1)
                        {
                            temp = QString("current: %1").arg(va);
                        }
                        else if (step == 1)
                        {
                            temp = QString("current: %1, [%2,%3]").arg(va).arg(mi).arg(ma);
                        }
                        else
                        {
                            temp = QString("current: %1, [%2,%3], step: %4").arg(va).arg(mi).arg(ma).arg(step);
                        }
                        values["values"].append(temp);

                        item = PyLong_FromLong(va);
                        PyDict_SetItemString(p_pyLine, "value", item);
                        Py_DECREF(item);

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                        }
                    break;

                    case ito::ParamBase::Double & ito::paramTypeMask:
                        {
                        const ito::DoubleMeta *dblMeta = static_cast<const ito::DoubleMeta*>(meta);
                        double mi, ma, step;
                        if (dblMeta)
                        {
                            mi = dblMeta->getMin();
                            ma = dblMeta->getMax();
                            step = dblMeta->getStepSize();
                        }
                        else
                        {
                            ma = std::numeric_limits<double>::max();
                            mi = -ma;
                            step = 0.0;
                        }
                        double va = p.getVal<double>();

                        if (qAbs(ma - std::numeric_limits<double>::max()) < std::numeric_limits<double>::epsilon() && qAbs(mi + std::numeric_limits<double>::max()) < std::numeric_limits<double>::epsilon() && step == 0.0)
                        {
                            temp = QString("current: %1").arg(va);
                        }
                        else if (step == 0.0)
                        {
                            temp = QString("current: %1, [%2,%3]").arg(va).arg(mi).arg(ma);
                        }
                        else
                        {
                            temp = QString("current: %1, [%2,%3], step: %4").arg(va).arg(mi).arg(ma).arg(step);
                        }
                        values["values"].append(temp);

                        item = PyFloat_FromDouble(va);
                        PyDict_SetItemString(p_pyLine, "value", item);
                        Py_DECREF(item);

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                        }
                    break;

                    case (ito::ParamBase::String & ito::paramTypeMask):
                    {
                        char* tempbuf = p.getVal<char*>();
                        if (tempbuf == NULL)
                        {
                            item = PyUnicode_FromString("");
                            values["values"].append("");
                            PyDict_SetItemString(p_pyLine, "value", item);
                            Py_DECREF(item);
                        }
                        else
                        {
                            temp = tempbuf;
                            temp.replace("\n","\\n");
                            temp.replace("\r","\\r");
                            if (temp.size() > 20)
                            {
                                temp = QString("\"%1...\"").arg(temp.left(20));
                            }
                            else
                            {
                                temp = QString("\"%1\"").arg(temp);
                            }
                            values["values"].append(temp);
                            item = PythonQtConversion::QByteArrayToPyUnicodeSecure(tempbuf);
                            PyDict_SetItemString(p_pyLine, "value", item);
                            Py_DECREF(item);
                        }
                    }
                    break;

                    case ito::ParamBase::CharArray & ito::paramTypeMask:
                    {
                        int len = p.getLen();
                        char *ptr = p.getVal<char*>();
                        switch (len)
                        {
                        case 0:
                            values["values"].append("empty");
                            break;
                        case 1:
                            temp = QString("[%1]").arg(ptr[0]);
                            values["values"].append(temp);
                            break;
                        case 2:
                            temp = QString("[%1,%2]").arg(ptr[0]).arg(ptr[1]);
                            values["values"].append(temp);
                            break;
                        case 3:
                            temp = QString("[%1,%2,%3]").arg(ptr[0]).arg(ptr[1]).arg(ptr[2]);
                            values["values"].append(temp);
                            break;
                        case 4:
                            temp = QString("[%1,%2,%3,%4]").arg(ptr[0]).arg(ptr[1]).arg(ptr[2]).arg(ptr[3]);
                            values["values"].append(temp);
                            break;
                        default:
                            temp = QString("%1 elements").arg(len);
                            values["values"].append(temp);
                            break;
                        }

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                    }
                    break;

                    case ito::ParamBase::IntArray & ito::paramTypeMask:
                    {
                        int len = p.getLen();
                        int *ptr = p.getVal<int*>();
                        switch (len)
                        {
                        case 0:
                            values["values"].append("empty");
                            break;
                        case 1:
                            temp = QString("[%1]").arg(ptr[0]);
                            values["values"].append(temp);
                            break;
                        case 2:
                            temp = QString("[%1,%2]").arg(ptr[0]).arg(ptr[1]);
                            values["values"].append(temp);
                            break;
                        case 3:
                            temp = QString("[%1,%2,%3]").arg(ptr[0]).arg(ptr[1]).arg(ptr[2]);
                            values["values"].append(temp);
                            break;
                        case 4:
                            temp = QString("[%1,%2,%3,%4]").arg(ptr[0]).arg(ptr[1]).arg(ptr[2]).arg(ptr[3]);
                            values["values"].append(temp);
                            break;
                        default:
                            temp = QString("%1 elements").arg(len);
                            values["values"].append(temp);
                            break;
                        }

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                    }
                    break;

                    case ito::ParamBase::DoubleArray & ito::paramTypeMask:
                    {
                        int len = p.getLen();
                        double *ptr = p.getVal<double*>();
                        switch (len)
                        {
                        case 0:
                            values["values"].append("empty");
                            break;
                        case 1:
                            temp = QString("[%1]").arg(ptr[0]);
                            values["values"].append(temp);
                            break;
                        case 2:
                            temp = QString("[%1,%2]").arg(ptr[0]).arg(ptr[1]);
                            values["values"].append(temp);
                            break;
                        case 3:
                            temp = QString("[%1,%2,%3]").arg(ptr[0]).arg(ptr[1]).arg(ptr[2]);
                            values["values"].append(temp);
                            break;
                        case 4:
                            temp = QString("[%1,%2,%3,%4]").arg(ptr[0]).arg(ptr[1]).arg(ptr[2]).arg(ptr[3]);
                            values["values"].append(temp);
                            break;
                        default:
                            temp = QString("%1 elements").arg(len);
                            values["values"].append(temp);
                            break;
                        }

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                    }
                    break;

                    case ((ito::ParamBase::Pointer | ito::ParamBase::HWRef) & ito::paramTypeMask):
                    case (ito::ParamBase::Pointer & ito::paramTypeMask):
                    case (ito::ParamBase::DObjPtr & ito::paramTypeMask):
                    case (ito::ParamBase::PointCloudPtr & ito::paramTypeMask):
                    case (ito::ParamBase::PointPtr & ito::paramTypeMask):
                    case (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask):
                        values["values"].append("<Object-Pointer>");

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                    break;

                    default:
                        values["values"].append("<unknown>");

                        item = parseParamMetaAsDict(meta);
                        PyDict_Merge(p_pyLine, item, 1);
                        Py_DECREF(item);
                    break;

                }
            }
            if (p.getInfo())
            {
                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(p.getInfo());
                PyDict_SetItemString(p_pyLine, "info", item);
                Py_DECREF(item);
            }
            
            PyTuple_SetItem(pVector, n, p_pyLine); //steals reference to p_pyLine
        }
		else
		{
			std::cerr << "The plugin parameter at position " << n + 1 << " contains neither type or name. This is an invalid parameter. Please check the plugin\n" << std::endl;
			//this is an error case, params vector contains a type-less element. Check the plugin, this must not happen.
			Py_INCREF(Py_None);
			PyTuple_SetItem(pVector, n, Py_None); //steals reference of Py_None
		}
    }

    //now construct final output
    int numberLength = 2;
    int nameLength = 4;
    int typeLength = 4;
    int valuesLength = 5;
    int readWriteLength = 3;
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
    readWriteLength += 1;

    // write a heading
    if (asErr)
    {
        output.append("#");
    }
    else
    {
        output.append("'"); //mark as unclosed string
    }

    temp = QString("No").leftJustified(numberLength,' ');
    output.append(temp);
    temp = QString("Name").leftJustified(nameLength,' ');
    output.append(temp);
    temp = QString("Type").leftJustified(typeLength,' ');
    output.append(temp);
    if (addInfos)
    {
        temp = QString("Value").leftJustified(valuesLength,' ');
        output.append(temp);
        temp = QString("R/W").leftJustified(readWriteLength, ' ');
        output.append(temp);
        output.append("Description");
    }
    output.append("\n");

    for (int i = 0; i < values["number"].length(); i++)
    {
        if (asErr)
        {
            output.append("#");
        }
        else
        {
            output.append("'"); //mark as unclosed string
        }
        temp = values["number"][i].leftJustified(numberLength,' ', true);
        output.append(temp);
        temp = values["name"][i].leftJustified(nameLength,' ');
        output.append(temp);
        temp = values["type"][i].leftJustified(typeLength,' ', true);
        output.append(temp);
        if (addInfos)
        {
            temp = values["values"][i].leftJustified(valuesLength,' ', true);
            output.append(temp);
            temp = values["readwrite"][i].leftJustified(readWriteLength,' ', true);
            output.append(temp);
            output.append(values["description"][i]);
        }

        if (num == i)
        {
            output.append(" <-- erroneous parameter");
        }
        output.append("\n");
    }

    if (printToStdStream)
    {
        if (asErr)
        {
            std::cerr << output.toLatin1().data() << std::endl;
        }
        else
        {
            std::cout << output.toLatin1().data() << std::endl;
        }
    }

    return pVector;
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
void errOutInitParams(const QVector<ito::Param> *params, const int num, const char *reason)
{
    PyErr_PrintEx(0);
    std::cerr << "\n";
    if (reason)
    {
        std::cerr << reason << "\n";
    }
    else
    {
        std::cerr << "unknown error\n";
    }

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
    PyErr_PrintEx(0);
}

//-----------------------------------------------------------------------------------------------------
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
        errOutInitParams(defaultParamListMand, -1, QObject::tr("wrong number of parameters. Mandatory parameters are:").toLatin1().data());
        errOutInitParams(defaultParamListOpt, -1, QObject::tr("optional parameters are:").toLatin1().data());
        if (mandPParsed)
        {
            free(mandPParsed);
            mandPParsed = NULL;
        }
        if (optPParsed)
        {
            free(optPParsed);
            optPParsed = NULL;
        }
        return ito::RetVal::format(ito::retError, 0, QObject::tr("wrong number of parameters (%i given, %i mandatory and %i optional required)").toLatin1().data(), 
            argsLen + kwdsLen, numMandParams, numOptParams);
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
                {
                    free(mandPParsed);
                    mandPParsed = NULL;
                }
                if (optPParsed)
                {
                    free(optPParsed);
                    optPParsed = NULL;
                }
                return ito::RetVal::format(ito::retError, 0, QObject::tr("parameter %d - %s passed as arg and keyword!").toLatin1().data(), n, tkey);
            }
        }
        for (int n = len; n < argsLen; n++)
        {
            const char *tkey = (*defaultParamListOpt)[n - len].getName();
            if (PyDict_GetItemString(kwds, tkey))
            {
                if (mandPParsed)
                {
                    free(mandPParsed);
                    mandPParsed = NULL;
                }
                if (optPParsed)
                {
                    free(optPParsed);
                    optPParsed = NULL;
                }
                return ito::RetVal::format(ito::retError, 0, QObject::tr("optional parameter %d - %s passed as arg and keyword!").toLatin1().data(), n, tkey);
            }
        }
    }

    if(kwds)
    {
        // check if any key is given, which does not exist in kwds-dictionary
        Py_ssize_t foundKwds = 0;
        foreach(const ito::Param p, *defaultParamListMand)
        {
            if (PyDict_GetItemString(kwds, p.getName())) 
            {
                foundKwds++;
            }
        }
        foreach(const ito::Param p, *defaultParamListOpt)
        {
            if (PyDict_GetItemString(kwds, p.getName())) 
            {
                foundKwds++;
            }
        }

        //this is a keyword-parameter, that can be passed without being part of the mandatory or optional parameters
        if (PyDict_GetItemString(kwds, "autoLoadParams"))
        {
            foundKwds++;
        }

        if (foundKwds != PyDict_Size(kwds))
        {
            if (mandPParsed) 
            {
                free(mandPParsed);
                mandPParsed = NULL;
            }
            if (optPParsed)  
            {
                free(optPParsed);
                optPParsed = NULL;
            }
            std::cerr << "there are keyword arguments that does not exist in mandatory or optional parameters." << std::endl;
            errOutInitParams(defaultParamListMand, -1, "Mandatory parameters are:");
            errOutInitParams(defaultParamListOpt, -1, "Optional parameters are:");
            return ito::RetVal(ito::retError, 0, QObject::tr("there are keyword arguments that does not exist in mandatory or optional parameters.").toLatin1().data());
        }
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
            errOutInitParams(defaultParamListMand, -1, QObject::tr("wrong number of parameters\n Mandatory parameters are:\n").toLatin1().data());
            if (mandPParsed)
            {
                free(mandPParsed);
                mandPParsed = NULL;
            }
            if (optPParsed)
            {
                free(optPParsed);
                optPParsed = NULL;
            }
            return ito::RetVal::format(ito::retError, 0, QObject::tr("wrong number of parameters (%i given, %i mandatory and %i optional required)").toLatin1().data(), 
                argsLen + kwdsLen, numMandParams, numOptParams);
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
        if (retval.containsError())
        {
            if (retval.hasErrorMessage() == false)
            {
                errOutInitParams(defaultParamListMand, n, QObject::tr("wrong parameter type").toLatin1().data());
            }
            else
            {
                errOutInitParams(defaultParamListMand, n, retval.errorMessage());
            }
            if (mandPParsed)
            {
                free(mandPParsed);
                mandPParsed = NULL;
            }
            if (optPParsed)
            {
                free(optPParsed);
                optPParsed = NULL;
            }
            return ito::retError;
        }
    }

    for (int n = 0; n < mandKwd; n++)
    {
        const char *tkey = (*defaultParamListMand)[len + n].getName();
        tempObj = PyDict_GetItemString(kwds, tkey);
        
        retval = checkAndSetParamVal(tempObj, &((*defaultParamListMand)[n + len]), paramListMandOut[n + len], &(mandPParsed[n + len]));
        if (retval.containsError())
        {
            if (retval.hasErrorMessage() == false)
            {
                errOutInitParams(defaultParamListMand, n, QObject::tr("wrong parameter type").toLatin1().data());
            }
            else
            {
                errOutInitParams(defaultParamListMand, n, retval.errorMessage());
            }
            if (mandPParsed)
            {
                free(mandPParsed);
                mandPParsed = NULL;
            }
            if (optPParsed)
            {
                free(optPParsed);
                optPParsed = NULL;
            }
            return ito::retError;
        }
    }

    // read in remaining (optional) parameters
    for (int n = numMandParams; n < argsLen; n++)
    {
        tempObj = PyTuple_GetItem(args, n);

        retval = checkAndSetParamVal(tempObj, &((*defaultParamListOpt)[n - numMandParams]), paramListOptOut[n - numMandParams], &(optPParsed[n - numMandParams]));
        if (retval.containsError())
        {
            if (retval.hasErrorMessage() == false)
            {
                errOutInitParams(defaultParamListOpt, n - numMandParams, QObject::tr("wrong parameter type").toLatin1().data());
            }
            else
            {
                errOutInitParams(defaultParamListOpt, n - numMandParams, retval.errorMessage());
            }
            if (mandPParsed)
            {
                free(mandPParsed);
                mandPParsed = NULL;
            }
            if (optPParsed)
            {
                free(optPParsed);
                optPParsed = NULL;
            }
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
                retval = checkAndSetParamVal(tempObj, &((*defaultParamListOpt)[n]), paramListOptOut[n], &(optPParsed[n])); 
                if (retval.containsError())
                {
                    if (retval.hasErrorMessage())
                    {
                        errOutInitParams(defaultParamListOpt, n, retval.errorMessage());
                    }
                    else
                    {
                        errOutInitParams(defaultParamListOpt, n, QObject::tr("wrong parameter type").toLatin1().data());
                    }
                    if (mandPParsed)
                    {
                        free(mandPParsed);
                        mandPParsed = NULL;
                    }
                    if (optPParsed)
                    {
                        free(optPParsed);
                        optPParsed = NULL;
                    }
                    return ito::retError;
                }
            }
        }
    }

    if (mandPParsed)
    {
        free(mandPParsed);
        mandPParsed = NULL;
    }
    if (optPParsed)
    {
        free(optPParsed);
        optPParsed = NULL;
    }

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
    if (paramVecIn)
    {
        paramVecOut.clear();
        for (int i=0;i<paramVecIn->size();i++)
        {
            paramVecOut.append(ito::ParamBase(paramVecIn->value(i)));
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError, 0, QObject::tr("paramVecIn is NULL").toLatin1().data());
}

//----------------------------------------------------------------------------------------------------------------------------------
/** makes a deep copy of a vector with values of type Param
*   
*   @param [in]     paramVecIn is a pointer to a vector of Param-values
*   @param [out]    paramVecOut is a reference to a vector which is first cleared and then filled with a deep copy of every element of paramVecIn
*/
ito::RetVal copyParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::Param> &paramVecOut)
{
    if (paramVecIn)
    {
        paramVecOut.clear();
        for (int i=0;i<paramVecIn->size();i++)
        {
            paramVecOut.append(ito::Param(paramVecIn->value(i)));
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError, 0, QObject::tr("paramVecIn is NULL").toLatin1().data());
}

//----------------------------------------------------------------------------------------------------------------------------------
/** makes a deep copy of a vector with values of type Param
*   
*   @param [in]     paramVecIn is a pointer to a vector of Param-values
*   @param [out]    paramVecOut is a reference to a vector which is first cleared and then filled with a deep copy of every element of paramVecIn (casted to ito::ParamBase)
*/
ito::RetVal copyParamVector(const QVector<ito::Param> *paramVecIn, QVector<ito::ParamBase> &paramVecOut)
{
    if (paramVecIn)
    {
        paramVecOut.clear();
        for (int i=0;i<paramVecIn->size();i++)
        {
            paramVecOut.append(ito::ParamBase(paramVecIn->value(i)));
        }

        return ito::retOk;
    }
    return ito::RetVal(ito::retError, 0, QObject::tr("paramVecIn is NULL").toLatin1().data());
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
    if (kwds)
    {
        if (PyDict_GetItemString(kwds, "autoLoadParams"))
        {
            if (PyLong_Check(PyDict_GetItemString(kwds, "autoLoadParams")))
            {
                *enableAutoLoadParams = (bool)(PyLong_AsLong(PyDict_GetItemString(kwds, "autoLoadParams")));
            }
            else
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("Keyword autoLoadParams not of integer type").toLatin1().data());
            }
            if (PyDict_DelItemString(kwds, "autoLoadParams"))
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("Could not delete Keyword: autoLoadParams").toLatin1().data());
            }
        }
    }
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------
PyObject* buildFilterOutputValues(QVector<QVariant> *outVals, ito::RetVal &retValue)
{
    PyObject *tuple = NULL;
    QVariant *elem;

    if (outVals->size() <= 0)
    {
        retValue += ito::RetVal(ito::retOk);
        Py_RETURN_NONE;
    }
    else if (outVals->size() == 1)
    {
        elem = &(outVals->data()[0]);
        //tuple = transformQVariant2PyObject(elem, retValue);
        tuple = PythonQtConversion::QVariantToPyObject(*elem);
        if (tuple == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, QObject::tr("unknown parameter of type QVariant").toLatin1().data());
        }
    }
    else
    {
        tuple = PyTuple_New(outVals->size());
        PyObject *temp = NULL;
        for (int i=0; i <outVals->size(); i++)
        {
            elem = &(outVals->data()[i]);
            temp = PythonQtConversion::QVariantToPyObject(*elem);
            if (temp == NULL)
            {
                PyErr_SetString(PyExc_RuntimeError, QObject::tr("unknown parameter of type QVariant").toLatin1().data());
            }
            else
            {
                PyTuple_SetItem(tuple, i, temp); //steals reference
            }
        }
    }

    return tuple;

}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject *parseParamMetaAsDict(const ito::ParamMeta *meta)
{
    if (meta)
    {
        PyObject *dict = PyDict_New();
        PyObject *temp = NULL;
        switch (meta->getType())
        {
        case ito::ParamMeta::rttiCharMeta:
            {
                const ito::CharMeta *cm = static_cast<const ito::CharMeta*>(meta);
                temp = PyUnicode_FromString("char meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiIntMeta:
            {
                const ito::IntMeta *cm = static_cast<const ito::IntMeta*>(meta);
                temp = PyUnicode_FromString("int meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiDoubleMeta:
            {
                const ito::DoubleMeta *cm = static_cast<const ito::DoubleMeta*>(meta);
                temp = PyUnicode_FromString("float meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                if (ito::isZeroValue(cm->getStepSize(), std::numeric_limits<double>::epsilon()))
                {
                    PyDict_SetItemString(dict, "step", Py_None);
                }
                else
                {
                    temp = PyFloat_FromDouble(cm->getStepSize());
                    PyDict_SetItemString(dict, "step", temp);
                    Py_DECREF(temp);
                }
            }
            break;
        case ito::ParamMeta::rttiStringMeta:
            {
                const ito::StringMeta *cm = static_cast<const ito::StringMeta*>(meta);
                temp = PyUnicode_FromString("string meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                switch (cm->getStringType())
                {
                case ito::StringMeta::String:
                    temp = PyUnicode_FromString("String");
                    break;
                case ito::StringMeta::Wildcard:
                    temp = PyUnicode_FromString("Wildcard");
                    break;
                case ito::StringMeta::RegExp:
                    temp = PyUnicode_FromString("RegExp");
                    break;
                }
                PyDict_SetItemString(dict, "stringType", temp);
                Py_DECREF(temp);

                temp = PyTuple_New(cm->getLen());
                for (int i = 0 ; i < cm->getLen(); ++i)
                {
                    PyTuple_SetItem(temp, i, PyUnicode_FromString(cm->getString(i))); //steals reference
                }
                PyDict_SetItemString(dict, "allowedItems", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiHWMeta:
            {
                const ito::HWMeta *cm = static_cast<const ito::HWMeta*>(meta);
                temp = PyUnicode_FromString("hardware plugin meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyUnicode_FromString(cm->getHWAddInName().data());
                PyDict_SetItemString(dict, "requiredPluginName", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiDObjMeta:
            {
                const ito::DObjMeta *cm = static_cast<const ito::DObjMeta*>(meta);
                temp = PyUnicode_FromString("data object meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMinDim());
                PyDict_SetItemString(dict, "dimMin", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMaxDim());
                PyDict_SetItemString(dict, "dimMax", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiIntArrayMeta:
            {
                const ito::IntArrayMeta *cm = static_cast<const ito::IntArrayMeta*>(meta);
                temp = PyUnicode_FromString("int sequence meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getNumMin());
                PyDict_SetItemString(dict, "numMin", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getNumMax());
                PyDict_SetItemString(dict, "numMax", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getNumStepSize());
                PyDict_SetItemString(dict, "numStep", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiDoubleArrayMeta:
            {
                const ito::DoubleArrayMeta *cm = static_cast<const ito::DoubleArrayMeta*>(meta);
                temp = PyUnicode_FromString("float sequence meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getNumMin());
                PyDict_SetItemString(dict, "numMin", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getNumMax());
                PyDict_SetItemString(dict, "numMax", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getNumStepSize());
                PyDict_SetItemString(dict, "numStep", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiCharArrayMeta:
            {
                const ito::CharArrayMeta *cm = static_cast<const ito::CharArrayMeta*>(meta);
                temp = PyUnicode_FromString("char sequence meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getNumMin());
                PyDict_SetItemString(dict, "numMin", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getNumMax());
                PyDict_SetItemString(dict, "numMax", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getNumStepSize());
                PyDict_SetItemString(dict, "numStep", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiIntervalMeta:
            {
                const ito::IntervalMeta *cm = static_cast<const ito::IntervalMeta*>(meta);
                temp = PyUnicode_FromString("integer interval meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getSizeMin());
                PyDict_SetItemString(dict, "sizeMin", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getSizeMax());
                PyDict_SetItemString(dict, "sizeMax", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getSizeStepSize());
                PyDict_SetItemString(dict, "sizeStep", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiDoubleIntervalMeta:
            {
                const ito::DoubleIntervalMeta *cm = static_cast<const ito::DoubleIntervalMeta*>(meta);
                temp = PyUnicode_FromString("float interval meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                if (ito::isZeroValue(cm->getStepSize(), std::numeric_limits<double>::epsilon()))
                {
                    PyDict_SetItemString(dict, "step", Py_None);
                }
                else
                {
                    temp = PyFloat_FromDouble(cm->getStepSize());
                    PyDict_SetItemString(dict, "step", temp);
                    Py_DECREF(temp);
                }

                temp = PyFloat_FromDouble(cm->getSizeMin());
                PyDict_SetItemString(dict, "sizeMin", temp);
                Py_DECREF(temp);

                temp = PyFloat_FromDouble(cm->getSizeMax());
                PyDict_SetItemString(dict, "sizeMax", temp);
                Py_DECREF(temp);

                if (ito::isZeroValue(cm->getSizeStepSize(), std::numeric_limits<double>::epsilon()))
                {
                    PyDict_SetItemString(dict, "sizeStep", Py_None);
                }
                else
                {
                    temp = PyFloat_FromDouble(cm->getSizeStepSize());
                    PyDict_SetItemString(dict, "sizeStep", temp);
                    Py_DECREF(temp);
                }
            }
            break;
        case ito::ParamMeta::rttiRangeMeta:
            {
                const ito::RangeMeta *cm = static_cast<const ito::RangeMeta*>(meta);
                temp = PyUnicode_FromString("integer range meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMin());
                PyDict_SetItemString(dict, "min", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getMax());
                PyDict_SetItemString(dict, "max", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getStepSize());
                PyDict_SetItemString(dict, "step", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getSizeMin());
                PyDict_SetItemString(dict, "sizeMin", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getSizeMax());
                PyDict_SetItemString(dict, "sizeMax", temp);
                Py_DECREF(temp);

                temp = PyLong_FromLong(cm->getSizeStepSize());
                PyDict_SetItemString(dict, "sizeStep", temp);
                Py_DECREF(temp);
            }
            break;
        case ito::ParamMeta::rttiRectMeta:
            {
                const ito::RectMeta *cm = static_cast<const ito::RectMeta*>(meta);
                temp = PyUnicode_FromString("integer rect meta");
                PyDict_SetItemString(dict, "metaTypeStr", temp);
                Py_DECREF(temp);

                temp = parseParamMetaAsDict(&cm->getWidthRangeMeta());
                PyDict_SetItemString(dict, "widthMeta", temp);
                Py_DECREF(temp);

                temp = parseParamMetaAsDict(&cm->getHeightRangeMeta());
                PyDict_SetItemString(dict, "heightMeta", temp);
                Py_DECREF(temp);
            }
            break;
        default:
            temp = PyUnicode_FromString("unknown meta");
            PyDict_SetItemString(dict, "metaTypeStr", temp);
            Py_DECREF(temp);
            break;
        }

        temp = PyLong_FromLong(meta->getType());
        PyDict_SetItemString(dict, "metaType", temp);
        Py_DECREF(temp);

        return dict;
    }
    else
    {
        return PyDict_New();
    }
}

//------------------------------------------------------------------------------------------------------------------
/* transforms a possible warning or error in retVal into a Python exception or warning
    
   returns true if retVal contained no error or if a warning became "only" a warning in Python.
   returns false if a Python exception was created or if the warning level in Python was set such that the
          warning contained in retVal also raised a Python exception.
*/
bool PythonCommon::transformRetValToPyException(ito::RetVal &retVal, PyObject *exceptionIfError /*= PyExc_RuntimeError*/, PyObject *exceptionIfWarning /*= PyExc_RuntimeWarning*/)
{
    QByteArray msg;
    if (retVal.containsWarningOrError())
    {
        const char *temp = retVal.errorMessage();
        if (temp == NULL)
        {
            msg = QObject::tr("- unknown message -").toUtf8();
        }
        else
        {
            msg = QString::fromLatin1(temp).toUtf8();
        }

        if (retVal.containsError())
        {
            PyErr_SetString(exceptionIfError, msg.data());
            return false;
        }
        else
        {
            if (PyErr_WarnEx(exceptionIfWarning, msg.data(), 1) == -1)
            {
                return false; //warning was turned into a real exception, 
            }
            else
            {
                return true; //warning is a warning, go on with the script
            }
        }
    }

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------
bool PythonCommon::setReturnValueMessage(ito::RetVal &retVal, const QString &objName, const tErrMsg &errorMSG, PyObject *exceptionIfError /*= PyExc_RuntimeError*/, PyObject *exceptionIfWarning /*= PyExc_RuntimeWarning*/)
{
    QByteArray msgSpecified;
    QByteArray msgUnspecified;

    if (retVal.containsError())
    {
        switch(errorMSG)
        {
            case PythonCommon::noMsg:
            default:
                msgSpecified = QObject::tr("%s with message: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("%s with unspecified error.").toLatin1().data();
                break;
            case PythonCommon::loadPlugin:
                msgSpecified = QObject::tr("Could not load plugin %s with error message: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Could not load plugin %s with unspecified error.").toLatin1().data();
                break;
            case PythonCommon::runFunc:
                msgSpecified = QObject::tr("Error executing function %s with error message: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Error executing function %s with unspecified error.").toLatin1().data();
                break;
            case PythonCommon::invokeFunc:
                msgSpecified = QObject::tr("Error invoking function %s with error message: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified error invoking function %s.").toLatin1().data();
                break;
            case PythonCommon::getProperty:
                msgSpecified = QObject::tr("Error while getting property info %s with error message: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified error while getting property info %s.").toLatin1().data();
                break;
            case PythonCommon::execFunc:
                msgSpecified = QObject::tr("Error invoking exec-function %s with error message: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Error invoking exec-function %s with unspecified error.").toLatin1().data();
                break;
        }

        if (retVal.hasErrorMessage())
        {
            PyErr_Format(exceptionIfError, msgSpecified.data(), objName.toUtf8().data(), QString::fromLatin1(retVal.errorMessage()).toUtf8().data());
        }
        else
        {
            PyErr_Format(exceptionIfError, msgUnspecified.data(), objName.toUtf8().data());
        }
        return false;
    }
    else if (retVal.containsWarning())
    {
        switch(errorMSG)
        {
            case PythonCommon::noMsg:
            default:
                msgSpecified = QObject::tr("Warning while %s: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("%s with unspecified warning.").toLatin1().data();
                break;
            case PythonCommon::loadPlugin:
                msgSpecified = QObject::tr("Warning while loading plugin %s: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified warning while loading plugin %s.").toLatin1().data();
                break;
            case PythonCommon::runFunc:
                msgSpecified = QObject::tr("Warning while executing function %s: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified warning while executing function %s.").toLatin1().data();
                break;
            case PythonCommon::invokeFunc:
                msgSpecified = QObject::tr("Warning while invoking function %s: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified warning while invoking function %s.").toLatin1().data();
                break;
            case PythonCommon::getProperty:
                msgSpecified = QObject::tr("Warning while getting property info %s: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified warning while getting property info %s.").toLatin1().data();
                break;
            case PythonCommon::execFunc:
                msgSpecified = QObject::tr("Warning while invoking exec-function %s: \n%s").toLatin1().data();
                msgUnspecified = QObject::tr("Unspecified warning invoking exec-function %s.").toLatin1().data();
                break;
        }

        int level;

        if (retVal.hasErrorMessage())
        {
            level = PyErr_WarnFormat(exceptionIfWarning, 1, msgSpecified.data(), objName.toUtf8().data(), QString::fromLatin1(retVal.errorMessage()).toUtf8().data());
        }
        else
        {
            level = PyErr_WarnFormat(exceptionIfWarning, 1, msgUnspecified.data(), objName.toUtf8().data());
        }

        if (level == -1)
        {
            return false; //the warning was turned into an exception due to the settings of the python warning module
        }
        else
        {
            return true; // the warning is a warning, go on with the script execution
        }
    }

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PythonCommon::setReturnValueMessage(ito::RetVal &retVal,const char *objName, const tErrMsg &errorMSG, PyObject *exceptionIfError /*= PyExc_RuntimeError*/, PyObject *exceptionIfWarning /*= PyExc_RuntimeWarning*/)
{
    QString pName(objName);
    return PythonCommon::setReturnValueMessage(retVal, pName, errorMSG, exceptionIfError, exceptionIfWarning);
}
//------------------------------------------------------------------------------------------------------------------
} //end namespace ito
