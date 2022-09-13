/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "helperCommon.h"

#include <qmap.h>
#include <qobject.h>
#include <qregularexpression.h>
#include <qsharedpointer.h>
#include <qstringlist.h>

#include "common/addInInterface.h"

namespace ito {

//-------------------------------------------------------------------------------------
//! checks param vector to be not a nullptr vector.
/*!
    \param [in] params is a pointer to QVector<ito::Param>. This pointer is checked.
    \return ito::RetVal, that contains an error if params is nullptr
*/
ito::RetVal checkParamVector(const QVector<ito::Param>* params)
{
    if (params == nullptr)
    {
        return ito::RetVal(
            ito::retError, 0, QObject::tr("parameter vector is not initialized").toLatin1().data());
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
//! verifies that the three param vectors are not nullptr and clears all vectors.
/*!
    If any of the given input parameters of type QVector<ito::Param>* are nullptr, a ito::RetVal is
   returned, that contains an error. Use this method in any algorithm-method in order to check the
   given input.

    \param [in] paramsMand is the first parameter vector
    \param [in] paramsOpt is the second parameter vector
    \param [in] paramsOut is the third parameter vector
    \return ito::RetVal, that contains an error if params is nullptr
    \sa checkParamVector
*/
ito::RetVal checkParamVectors(
    QVector<ito::Param>* paramsMand, QVector<ito::Param>* paramsOpt, QVector<ito::Param>* paramsOut)
{
    if (paramsMand == nullptr)
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("mandatory parameter vector is not initialized").toLatin1().data());
    }

    if (paramsOpt == nullptr)
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("optional parameter vector is not initialized").toLatin1().data());
    }

    if (paramsOut == nullptr)
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("output parameter vector is not initialized").toLatin1().data());
    }

    paramsMand->clear();
    paramsOpt->clear();
    paramsOut->clear();

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
//! returns a parameter from the given vector, selected by its name.
/*!
    This method checks the given parameter vector for any parameter, whose name
    is equal to the given name and returns the pointer to this parameter.

    Hint: The returned pointer is only valid as long as the given paramVec is
    not deleted or changed.

    \param [in] paramVec is a pointer to a vector of ito::Param
    \param [in] name is the name of the wanted parameter
    \param [in/out] retval (optional). If given, a retError is appended to
        this retval, if the parameter could not be found.
    \return the found parameter in the given vector or nullptr if nothing found.
*/
ito::Param* getParamByName(QVector<ito::Param>* paramVec, const char* name, ito::RetVal* retval)
{
    if (paramVec)
    {
        ito::Param* data = paramVec->data();
        const char* temp;

        for (int i = 0; i < paramVec->size(); ++i)
        {
            temp = data[i].getName();

            if (strcmp(temp, name) == 0)
            {
                return &(data[i]);
            }
        }
    }

    if (retval)
    {
        *retval += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("parameter '%1' cannot be found in given parameter vector")
                .arg(name)
                .toLatin1()
                .data());
    }

    return nullptr;
}

//-------------------------------------------------------------------------------------
//! returns a parameter from the given vector, selected by its name.
/*!
    This method checks the given parameter vector for any parameter, whose name
    is equal to the given name and returns the pointer to this parameter.

    Hint: The returned pointer is only valid as long as the given paramVec is
    not deleted or changed.

    \param [in] paramVec is a pointer to a vector of ito::Param
    \param [in] name is the name of the wanted parameter
    \param [in/out] retval (optional). If given, a retError is appended to
        this retval, if the parameter could not be found.
    \return the found parameter in the given vector or nullptr if nothing found.
*/
ito::ParamBase* getParamByName(
    QVector<ito::ParamBase>* paramVec, const char* name, ito::RetVal* retval)
{
    if (paramVec)
    {
        ito::ParamBase* data = paramVec->data();
        const char* temp;

        for (int i = 0; i < paramVec->size(); ++i)
        {
            temp = data[i].getName();
            if (strcmp(temp, name) == 0)
            {
                return &(data[i]);
            }
        }
    }

    if (retval)
    {
        *retval += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("parameter '%1' cannot be found in given parameter vector")
                .arg(name)
                .toLatin1()
                .data());
    }

    return nullptr;
}

//-------------------------------------------------------------------------------------
//! Check if the numeric value is within the min/max range of the meta info of param.
/*!
    If param is a real numeric type, value is checked if it lies within the
    limits of the optionally given meta information. If no meta information is given,
    value fits always. An optional step size is not considered.

    \param [in] param is the parameter, whose meta information is used as reference.
    \param [in] value is the value to be checked.
    \param [in/out] ok (optional). If given: true if the check could be executed,
        else false.
    \return false, if param is not numeric or if value is out of bounds. If param
        does not contain meta information, true is returned, too.
*/
bool checkNumericParamRange(const ito::Param& param, double& value, bool* ok)
{
    bool done = false;
    bool result = false;

    if (param.isNumeric())
    {
        const ito::ParamMeta* meta = param.getMeta();

        if (meta)
        {
            done = true;

            switch (meta->getType())
            {
            case ito::ParamMeta::rttiCharMeta:
            case ito::ParamMeta::rttiCharArrayMeta: {
                const ito::CharMeta* cMeta = (const ito::CharMeta*)meta;

                if (value >= cMeta->getMin() && value <= cMeta->getMax())
                {
                    result = true;
                }
            }
            break;
            case ito::ParamMeta::rttiIntMeta:
            case ito::ParamMeta::rttiIntArrayMeta:
            case ito::ParamMeta::rttiIntervalMeta:
            case ito::ParamMeta::rttiRangeMeta: {
                const ito::IntMeta* iMeta = (const ito::IntMeta*)meta;

                if (value >= iMeta->getMin() && value <= iMeta->getMax())
                {
                    result = true;
                }
            }
            break;
            case ito::ParamMeta::rttiDoubleMeta:
            case ito::ParamMeta::rttiDoubleArrayMeta:
            case ito::ParamMeta::rttiDoubleIntervalMeta: {
                const ito::DoubleMeta* dMeta = (const ito::DoubleMeta*)meta;

                if (value >= dMeta->getMin() && value <= dMeta->getMax())
                {
                    result = true;
                }
            }
            break;
            default:
                done = false;
                result = false;
                break;
            }
        }
        else
        {
            done = true;
            result = true;
        }
    }

    if (ok)
    {
        *ok = done;
    }

    return result;
}

//-------------------------------------------------------------------------------------
//! Get a parameter from a params map, given by a full key.
/*!
    The searched parameter from the given params map is given by the key.
    If found, the parameter is returned by the argument val, its name is
    returned by name and its index by index (or -1 if no array/list type or the
    entire array/list).

    \param [in] params is the given map of parameters.
    \param [in] key is the full key, which can be the name of the parameter only,
        or name[index] if the parameter is an array or list type and the single value
        param of at the given index position should be returned. It is also allowed
        to add a ":suffix" string to this key, which is however ignored.
    \param [out] val is the detected parameter.
    \param [out] name is the detected name of the parameter from key.
    \param [out] index is the detected index in the argument key, or -1 if not given.
    \return ito::retOk if the parameter could be found, ito::retWarning if key
        contains a sub-index but the parameter is no array or list type, else ito::retError.
*/
ito::RetVal getParamValue(
    const QMap<QString, Param>* params,
    const QString& key,
    ito::Param& val,
    QString& name,
    int& index)
{
    ito::RetVal retValue(ito::retOk);
    index = -1;
    name = key;

    if (key == "")
    {
        retValue += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("name of requested parameter is empty.").toLatin1().data());
    }
    else if (!params)
    {
        retValue += ito::RetVal(ito::retError, 0, "empty params map");
    }
    else
    {
        QString paramName;
        bool hasIndex;
        QString additionalTag;

        retValue += parseParamName(key, paramName, hasIndex, index, additionalTag);

        if (retValue.containsError() || paramName.isEmpty())
        {
            retValue = ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("the parameter name '%1' is invald").arg(key).toLatin1().data());
        }
        else
        {
            if (!hasIndex)
            {
                index = -1;
            }

            auto paramIt = params->constFind(paramName);

            if (paramIt != params->constEnd())
            {
                name = paramName;

                const auto& value = paramIt.value();

                switch (value.getType())
                {
                case ito::ParamBase::IntArray:
                case ito::ParamBase::DoubleArray:
                case ito::ParamBase::ComplexArray:
                case ito::ParamBase::CharArray:
                case ito::ParamBase::StringList:
                    if (index < 0)
                    {
                        val = value;
                    }
                    else if (index < value.getLen())
                    {
                        val = value[index];
                    }
                    else
                    {
                        val = ito::Param();
                        retValue += ito::RetVal(
                            ito::retError,
                            0,
                            QObject::tr("array index of parameter out of bounds.")
                                .toLatin1()
                                .data());
                    }
                    break;
                default:
                    if (index >= 0)
                    {
                        retValue += ito::RetVal(
                            ito::retWarning,
                            0,
                            QObject::tr("given index of parameter name ignored since parameter is "
                                        "no array type")
                                .toLatin1()
                                .data());
                    }

                    val = value;
                    break;
                }
            }
            else
            {
                retValue += ito::RetVal(
                    ito::retError,
                    0,
                    QObject::tr("parameter not found in m_params.").toLatin1().data());
            }
        }
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
//! internal helper method, no error checks here.
template <typename _Tp>
void paramHelperSetArrayValue_(ito::Param& param, ito::ParamBase value, const int pos)
{
    _Tp* dPtr = param.getVal<_Tp*>();
    _Tp val = value.getVal<_Tp>();

    if (pos >= 0)
    {
        dPtr[pos] = val;
    }
    else
    {
        for (int num = 0; num < param.getLen(); ++num)
        {
            dPtr[num] = val;
        }
    }
}

//-------------------------------------------------------------------------------------
//! internal helper method, no error checks here.
void paramHelperSetArrayValue(ito::Param& param, ito::ParamBase value, const int pos)
{
    assert(param.getType() & ito::Param::Pointer);

    switch (param.getType())
    {
    case ito::Param::CharArray:
        paramHelperSetArrayValue_<char>(param, value, pos);
        break;
    case ito::Param::IntArray:
        paramHelperSetArrayValue_<int>(param, value, pos);
        break;
    case ito::Param::DoubleArray:
        paramHelperSetArrayValue_<ito::float64>(param, value, pos);
        break;
    case ito::Param::ComplexArray:
        paramHelperSetArrayValue_<ito::complex128>(param, value, pos);
        break;
    case ito::Param::StringList: {
        ito::ByteArray* dPtr = param.getVal<ito::ByteArray*>();
        const char* val = value.getVal<const char*>();

        if (pos >= 0)
        {
            dPtr[pos] = val;
        }
        else
        {
            for (int num = 0; num < param.getLen(); ++num)
            {
                dPtr[num] = val;
            }
        }
    }
    break;
    }
}

//-------------------------------------------------------------------------------------
//! Sets the value of a parameter to a given new value val.
/*!
    This method searches for a parameter, based on its key, from a params map
    and if detected, sets its value to the given new value val.

    The found parameter and the new value must have the same type. However,
    if the found parameter is an array or list, the new value can either be
    the same type (then the entire content is set to the new value) or it can
    be the corresponding scalar type. Then, it depends if an index is given or not.
    If the index is not given or -1, all values in the current parameter are set
    to the same new value, else only the item at the given index position is
    overwritten.

    \param [in] params is the parameter map
    \param [in] key is the full key, which can be the name of the parameter only,
        or name[index] if the parameter is an array or list type and the single value
        param of at the given index position should be returned. It is also allowed
        to add a ":suffix" string to this key, which is however ignored.
    \param [out] val is the new parameter.
    \param [out] name is the detected name of the parameter from key.
    \param [out] index is the detected index in the argument key, or -1 if not given.
    \return ito::retOk if the parameter could be found and the type of val
        corresponds to the detected parameter, ito::retWarning if key
        contains a sub-index but the parameter is no array or list type, else ito::retError.
*/
ito::RetVal setParamValue(
    QMap<QString, Param>* params,
    const QString& key,
    const ito::ParamBase& val,
    QString& name,
    int& index)
{
    ito::RetVal retValue;
    name = key;
    index = -1;

    if (key == "")
    {
        retValue += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("name of requested parameter is empty.").toLatin1().data());
    }
    else if (params == nullptr)
    {
        retValue += ito::RetVal(ito::retError, 0, "empty params map");
    }
    else
    {
        QString paramName;
        bool hasIndex;
        QString additionalTag;

        retValue += parseParamName(key, paramName, hasIndex, index, additionalTag);

        if (retValue.containsError() || paramName.isEmpty())
        {
            retValue = ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("the parameter name '%1' is invalid").arg(key).toLatin1().data());
        }
        else
        {
            if (!hasIndex)
            {
                index = -1;
            }

            auto paramIt = params->find(paramName);

            if (paramIt != params->end())
            {
                name = paramName;
                auto& value = paramIt.value();

                if (value.getType() == val.getType())
                {
                    if (index >= 0)
                    {
                        retValue += ito::RetVal(
                            ito::retWarning,
                            0,
                            QObject::tr("given index of parameter name ignored since parameter is "
                                        "no array type")
                                .toLatin1()
                                .data());
                    }

                    value.copyValueFrom(&val);
                }
                else
                {
                    switch (value.getType())
                    {
                    case ito::ParamBase::IntArray:
                    case ito::ParamBase::DoubleArray:
                    case ito::ParamBase::ComplexArray:
                    case ito::ParamBase::CharArray: {
                        if ((value.getType() & ~ito::ParamBase::Pointer) != val.getType())
                        {
                            retValue += ito::RetVal(
                                ito::retError,
                                0,
                                QObject::tr("The type of the new value does not fit to the given "
                                            "array or list type.")
                                    .toLatin1()
                                    .data());
                        }
                        else if (index < 0)
                        {
                            paramHelperSetArrayValue(value, val, -1);
                        }
                        else if (index < value.getLen())
                        {
                            paramHelperSetArrayValue(value, val, index);
                        }
                        else
                        {
                            retValue += ito::RetVal(
                                ito::retError,
                                0,
                                QObject::tr("array index out of bounds.").toLatin1().data());
                        }
                    }
                    break;
                    case ito::ParamBase::StringList: {
                        if (val.getType() != ito::ParamBase::String)
                        {
                            retValue += ito::RetVal(
                                ito::retError,
                                0,
                                QObject::tr("The type of the new value does not fit to the given "
                                            "array or list type.")
                                    .toLatin1()
                                    .data());
                        }
                        else if (index < 0)
                        {
                            paramHelperSetArrayValue(value, val, -1);
                        }
                        else if (index < value.getLen())
                        {
                            paramHelperSetArrayValue(value, val, index);
                        }
                        else
                        {
                            retValue += ito::RetVal(
                                ito::retError,
                                0,
                                QObject::tr("array index out of bounds.").toLatin1().data());
                        }
                    }
                    break;
                    default: {
                        retValue += ito::RetVal(
                            ito::retError,
                            0,
                            QObject::tr("The parameter is either no array or list, or the type of "
                                        "the new value does not correspond to this parameter.")
                                .toLatin1()
                                .data());
                    }
                    break;
                    }
                }
            }
            else
            {
                retValue += ito::RetVal(
                    ito::retError,
                    0,
                    QObject::tr("parameter not found in m_params.").toLatin1().data());
            }
        }
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
//! parses parameter name with respect to regular expression, assigned for parameter-communcation
//! with plugins
/*!
    This method parses any parameter-name with respect to the rules defined for possible names of
   plugin-parameters.

    The regular expression used for the check is "^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$"

    Then the components are:

    [0] full string
    [1] PARAMNAME
    [2] [INDEX] or empty-string if no index is given
    [3] INDEX or empty-string if no index is given
    [4] :ADDITIONALTAG or empty-string if no tag is given
    [5] ADDITIONALTAG or empty-string if no tag is given

    \param [in] name is the raw parameter name
    \param [out] paramName is the real parameter name (first part of name; part before the first
   opening bracket ('[') or if not available the first colon (':')) \param [out] hasIndex indicates
   whether the name contains an index part (defined by a number within two brackets (e.g.
   '[NUMBER]'), which has to be appended to the paramName \param [out] index is the fixed-point
   index value or -1 if hasIndex is false \param [out] additionalTag is the remaining string of name
   which is the part after the first colon (':'). If an index part exists, the first colon after the
   index part is taken.
*/
ito::RetVal parseParamName(
    const QString& key, QString& paramName, bool& hasIndex, int& index, QString& additionalTag)
{
    ito::RetVal retValue = ito::retOk;
    paramName = QString();
    hasIndex = false;
    index = -1;
    additionalTag = QString();

    QRegularExpression rx("^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$");

    auto regexpmatch = rx.match(key);

    if (!regexpmatch.hasMatch())
    {
        retValue +=
            ito::RetVal(ito::retError, 0, QObject::tr("invalid parameter name").toLatin1().data());
    }
    else
    {
        QStringList pname = regexpmatch.capturedTexts();
        paramName = pname[1];

        if (pname.size() >= 4)
        {
            if (!pname[3].isEmpty())
            {
                index = pname[3].toInt(&hasIndex);
            }
        }

        if (pname.size() >= 6)
        {
            additionalTag = pname[5];
        }
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
//! Returns a data-type dependent string representation of the given number
QString boundedNumberText(char value, ito::ParamMeta::tRepresentation representation)
{
    if (value == std::numeric_limits<char>::min())
    {
        return "-inf";
    }
    else if (value == std::numeric_limits<char>::max())
    {
        return "inf";
    }

    switch (representation)
    {
    case ito::ParamMeta::HexNumber:
        return (value < 0 ? QString("-") : QString()) +
            QString("0x%1").arg(QString::number(value, 16));
    default:
        return QString::number(value);
    }
}

//-------------------------------------------------------------------------------------
//! Returns a data-type dependent string representation of the given number
QString boundedNumberText(int value, ito::ParamMeta::tRepresentation representation)
{
    if (value == std::numeric_limits<int>::min())
    {
        return "-inf";
    }
    else if (value == std::numeric_limits<int>::max())
    {
        return "inf";
    }

    switch (representation)
    {
    case ito::ParamMeta::HexNumber:
        return (value < 0 ? QString("-") : QString()) +
            QString("0x%1").arg(QString::number(value, 16));
    default:
        return QString::number(value);
    }
}

//-------------------------------------------------------------------------------------
//! Returns a data-type dependent string representation of the given number
QString boundedNumberText(
    double value, ito::DoubleMeta::tDisplayNotation displayNotation, int displayPrecision)
{
    if (value < 0 && std::isinf(value))
    {
        return "-inf";
    }
    else if (
        value <= (-std::numeric_limits<double>::max() + std::numeric_limits<double>::epsilon()))
    {
        return "-inf";
    }
    else if (value > 0 && std::isinf(value))
    {
        return "inf";
    }
    else if (value >= (std::numeric_limits<double>::max() - std::numeric_limits<double>::epsilon()))
    {
        return "inf";
    }
    else if (std::isnan(value))
    {
        return "nan";
    }

    switch (displayNotation)
    {
    default:
    case ito::DoubleMeta::Automatic:
        return QString::number(value, 'g');
    case ito::DoubleMeta::Fixed:
        return QString::number(value, 'f', displayPrecision);
    case ito::DoubleMeta::Scientific:
        return QString::number(value, 'e', displayPrecision);
    }
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderArrayContent(
    const char* values, size_t num, ito::ParamMeta::tRepresentation representation)
{
    QStringList items;
    QString content;

    if (num < 10)
    {
        for (size_t i = 0; i < num; ++i)
        {
            items << boundedNumberText(values[i], representation);
        }
    }
    else if (num >= 10)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            items << boundedNumberText(values[i], representation);
        }

        items << "...";

        for (size_t i = num - 4; i < num; ++i)
        {
            items << boundedNumberText(values[i], representation);
        }
    }

    content = items.join(", ");

    if (content.size() == 0)
    {
        content = " ";
    }

    return QString("[%1]").arg(content);
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderArrayContent(
    const int* values, size_t num, ito::ParamMeta::tRepresentation representation)
{
    QStringList items;
    QString content;

    if (num < 10)
    {
        for (size_t i = 0; i < num; ++i)
        {
            items << boundedNumberText(values[i], representation);
        }
    }
    else if (num >= 10)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            items << boundedNumberText(values[i], representation);
        }

        items << "...";

        for (size_t i = num - 4; i < num; ++i)
        {
            items << boundedNumberText(values[i], representation);
        }
    }

    content = items.join(", ");

    if (content.size() == 0)
    {
        content = " ";
    }

    return QString("[%1]").arg(content);
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderArrayContent(
    const double* values,
    size_t num,
    ito::DoubleMeta::tDisplayNotation displayNotation,
    int displayPrecision)
{
    QStringList items;
    QString content;

    if (num < 10)
    {
        for (size_t i = 0; i < num; ++i)
        {
            items << boundedNumberText(values[i], displayNotation, displayPrecision);
        }
    }
    else if (num >= 10)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            items << boundedNumberText(values[i], displayNotation, displayPrecision);
        }

        items << "...";

        for (size_t i = num - 4; i < num; ++i)
        {
            items << boundedNumberText(values[i], displayNotation, displayPrecision);
        }
    }

    content = items.join(", ");

    if (content.size() == 0)
    {
        content = " ";
    }

    return QString("[%1]").arg(content);
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderArrayContent(const ito::ByteArray* values, size_t num)
{
    QStringList items;
    QString content;

    auto parse = [](const ito::ByteArray& ba) {
        if (ba.empty())
        {
            return QLatin1String("\"\"");
        }
        else
        {
            return QLatin1String(ba.data(), ba.size());
        }
    }; // end of lambda expression

    if (num < 10)
    {
        for (size_t i = 0; i < num; ++i)
        {
            items << parse(values[i]);
        }
    }
    else if (num >= 10)
    {
        for (size_t i = 0; i < 4; ++i)
        {
            items << parse(values[i]);
        }

        items << "...";

        for (size_t i = num - 4; i < num; ++i)
        {
            items << parse(values[i]);
        }
    }

    content = items.join(", ");

    if (content.size() == 0)
    {
        content = " ";
    }

    return QString("[%1]").arg(content);
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderListMetaInfo(const ito::ListMeta* meta, bool translate)
{
    QString content;

    if (meta->getNumMin() == meta->getNumMax())
    {
        content =
            (translate ? QObject::tr("%1 values required") : QLatin1String("%1 values required"))
                .arg(meta->getNumMin());
    }
    else
    {
        content = translate ? QObject::tr("Allowed number of values: %1 - %2")
                            : QString("Allowed number of values: %1 - %2");
        content = content.arg(meta->getNumMin()).arg(meta->getNumMax());

        if (meta->getNumStepSize() != 1)
        {
            QString content2 = (translate ? QObject::tr(", step: %1") : QString(", step: %1"))
                                   .arg(meta->getNumStepSize());
            content += content2;
        }
    }

    return content;
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderDoubleMetaInfo(const ito::DoubleMeta* meta, bool translate)
{
    QString temp;
    const double minimum = meta->getMin();
    const double maximum = meta->getMax();
    const double eps = std::numeric_limits<double>::epsilon();
    const double dblmax = std::numeric_limits<double>::max();
    auto prec = meta->getDisplayPrecision();
    auto disnot = meta->getDisplayNotation();

    if (meta->getStepSize() == 0.0)
    {
        if ((minimum < 0) && ((minimum >= (-dblmax + eps)) || std::isinf(minimum)) &&
            (maximum > 0) && ((maximum >= (dblmax - eps)) || std::isinf(maximum)))
        {
            temp =
                translate ? QObject::tr("All values allowed") : QLatin1String("All values allowed");
        }
        else
        {
            temp = translate ? QObject::tr("Value range: [%1, %2]")
                             : QLatin1String("Value range: [%1, %2]");
            temp = temp.arg(boundedNumberText(minimum, disnot, prec))
                       .arg(boundedNumberText(maximum, disnot, prec));
        }
    }
    else
    {
        temp = translate ? QObject::tr("Value range: [%1:%2:%3]")
                         : QLatin1String("Value range: [%1:%2:%3]");
        temp = temp.arg(boundedNumberText(minimum, disnot, prec))
                   .arg(meta->getStepSize())
                   .arg(boundedNumberText(maximum, disnot, prec));
    }

    const auto& unit = meta->getUnit();

    if (!unit.empty())
    {
        temp += (translate ? QObject::tr(", Unit: %1") : QLatin1String(", Unit: %1"))
                    .arg(QLatin1String(unit.data(), unit.size()));
    }

    return temp;
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderCharMetaInfo(const ito::CharMeta* meta, bool translate)
{
    QString temp;
    char minimum = meta->getMin();
    char maximum = meta->getMax();
    auto repr = meta->getRepresentation();

    if (meta->getStepSize() == 1)
    {
        if ((minimum == std::numeric_limits<char>::min()) &&
            (maximum == std::numeric_limits<char>::max()))
        {
            temp =
                translate ? QObject::tr("All values allowed") : QLatin1String("All values allowed");
        }
        else
        {
            temp = translate ? QObject::tr("Value range: [%1, %2]")
                             : QLatin1String("Value range: [%1, %2]");
            temp = temp.arg(boundedNumberText(minimum, repr)).arg(boundedNumberText(maximum, repr));
        }
    }
    else
    {
        temp = translate ? QObject::tr("Value range: [%1:%2:%3]")
                         : QLatin1String("Value range: [%1:%2:%3]");
        temp = temp.arg(boundedNumberText(minimum, repr))
                   .arg(meta->getStepSize())
                   .arg(boundedNumberText(maximum, repr));
    }

    const auto& unit = meta->getUnit();

    if (!unit.empty())
    {
        temp += (translate ? QObject::tr(", Unit: %1") : QLatin1String(", Unit: %1"))
                    .arg(QLatin1String(unit.data(), unit.size()));
    }

    return temp;
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderIntMetaInfo(const ito::IntMeta* meta, bool translate)
{
    QString temp;
    int minimum = meta->getMin();
    int maximum = meta->getMax();
    auto repr = meta->getRepresentation();

    if (meta->getStepSize() == 1)
    {
        if ((minimum == std::numeric_limits<int>::min()) &&
            (maximum == std::numeric_limits<int>::max()))
        {
            temp =
                translate ? QObject::tr("All values allowed") : QLatin1String("All values allowed");
        }
        else
        {
            temp = translate ? QObject::tr("Value range: [%1, %2]")
                             : QLatin1String("Value range: [%1, %2]");
            temp = temp.arg(boundedNumberText(minimum, repr)).arg(boundedNumberText(maximum, repr));
        }
    }
    else
    {
        temp = translate ? QObject::tr("Value range: [%1:%2:%3]")
                         : QLatin1String("Value range: [%1:%2:%3]");
        temp = temp.arg(boundedNumberText(minimum, repr))
                   .arg(meta->getStepSize())
                   .arg(boundedNumberText(maximum, repr));
    }

    const auto& unit = meta->getUnit();

    if (!unit.empty())
    {
        temp += (translate ? QObject::tr(", Unit: %1") : QLatin1String(", Unit: %1"))
                    .arg(QLatin1String(unit.data(), unit.size()));
    }

    return temp;
}

//-------------------------------------------------------------------------------------
//!< helper method for getMetaDocstringFromParam
QString renderStrMetaInfo(const ito::StringMeta* meta, bool translate)
{
    QString temp;

    switch (meta->getStringType())
    {
    case ito::StringMeta::RegExp:
        temp = translate ? QObject::tr("RegularExpression") : "RegularExpression";

        if (meta->getLen() == 1)
        {
            temp = temp + QString(": \"%1\"").arg(QLatin1String(meta->getString(0)));
        }
        else if (meta->getLen() > 1)
        {
            QStringList allowed;
            for (int i = 0; i < meta->getLen(); ++i)
            {
                allowed += QString("\"%1\"").arg(QLatin1String(meta->getString(i)));
            }
            temp = temp + QString(": [%1]").arg(allowed.join("; "));
        }
        else if (meta->getLen() == 0)
        {
            temp = translate ? QObject::tr("RegExp: <no pattern given>")
                             : "RegExp: <no pattern given>";
        }
        break;
    case ito::StringMeta::String:
        temp = translate ? QObject::tr("Match") : "Match";

        if (meta->getLen() == 1)
        {
            temp = temp + QString(": \"%1\"").arg(QLatin1String(meta->getString(0)));
        }
        else if (meta->getLen() > 1)
        {
            QStringList allowed;

            for (int i = 0; i < meta->getLen(); ++i)
            {
                allowed += QString("\"%1\"").arg(QLatin1String(meta->getString(i)));
            }

            temp = temp + QString(": [%1]").arg(allowed.join(", "));
        }
        else if (meta->getLen() == 0)
        {
            temp =
                translate ? QObject::tr("Match: <no pattern given>") : "Match: <no pattern given>";
        }
        break;
    case ito::StringMeta::Wildcard:
        temp = translate ? QObject::tr("Wildcard") : "Wildcard";

        if (meta->getLen() == 1)
        {
            temp = temp + QString(": \"%1\"").arg(QLatin1String(meta->getString(0)));
        }
        else if (meta->getLen() > 1)
        {
            QStringList allowed;
            for (int i = 0; i < meta->getLen(); ++i)
            {
                allowed += QString("\"%1\"").arg(QLatin1String(meta->getString(i)));
            }
            temp = temp + QString("Wildcard: [%1]").arg(allowed.join("; "));
        }
        else if (meta->getLen() == 0)
        {
            temp = translate ? QObject::tr("Wildcard: <no pattern given>")
                             : "Wildcard: <no pattern given>";
        }

        break;
    }

    return temp;
}

//-------------------------------------------------------------------------------------
QString getMetaDocstringFromParam(const Param& param, bool translate, QString& pythonLikeTypename)
{
    QString meta;
    QString temp;
    QString defaultStr;

    switch (param.getType())
    {
    case ito::ParamBase::Int: {
        pythonLikeTypename = "int";
        const ito::IntMeta* pMeta = dynamic_cast<const ito::IntMeta*>(param.getMeta());
        ito::ParamMeta::tRepresentation repr = ito::ParamMeta::PureNumber;

        if (pMeta)
        {
            repr = pMeta->getRepresentation();
        }

        defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"))
                         .arg(boundedNumberText(param.getVal<ito::int32>(), repr));

        if (pMeta)
        {
            meta = renderIntMetaInfo(pMeta, translate) + ", " + defaultStr;
        }
        else
        {
            meta = defaultStr;
        }
    }
    break;
    case ito::ParamBase::Char: {
        // Never tested ... no filter holding metadata as char available
        pythonLikeTypename = "int (char)";
        const ito::CharMeta* pMeta = dynamic_cast<const ito::CharMeta*>(param.getMeta());
        ito::ParamMeta::tRepresentation repr = ito::ParamMeta::PureNumber;

        if (pMeta)
        {
            repr = pMeta->getRepresentation();
        }

        defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"))
                         .arg(boundedNumberText(param.getVal<char>(), repr));

        if (pMeta)
        {
            meta = renderCharMetaInfo(pMeta, translate) + ", " + defaultStr;
        }
        else
        {
            meta = defaultStr;
        }
    }
    break;
    case ito::ParamBase::Double: {
        pythonLikeTypename = "float (double-precision)";
        const ito::DoubleMeta* pMeta = dynamic_cast<const ito::DoubleMeta*>(param.getMeta());

        if (pMeta)
        {
            defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"))
                             .arg(boundedNumberText(
                                 param.getVal<float64>(),
                                 pMeta->getDisplayNotation(),
                                 pMeta->getDisplayPrecision()));

            meta = renderDoubleMetaInfo(pMeta, translate) + ", " + defaultStr;
        }
        else
        {
            defaultStr =
                (translate ? QObject::tr("Default: %1") : QString("Default: %1"))
                    .arg(boundedNumberText(param.getVal<float64>(), ito::DoubleMeta::Automatic, 3));

            meta = defaultStr;
        }
    }
    break;
    case ito::ParamBase::Complex: {
        pythonLikeTypename = "complex (double-precision)";
        ito::float64 real = param.getVal<ito::complex128>().real();
        ito::float64 imag = param.getVal<ito::complex128>().imag();

        if (imag >= 0)
        {
            temp = translate ? QObject::tr("Default: %1+%2i") : QLatin1String("Default: %1+%2i");
            meta = temp.arg(real).arg(imag);
        }
        else
        {
            temp = translate ? QObject::tr("Default: %1-%2i") : QLatin1String("Default: %1-%2i");
            meta = temp.arg(real).arg(-imag);
        }
    }
    break;
    case ito::ParamBase::String: {
        pythonLikeTypename = "str";
        const ito::StringMeta* pMeta = dynamic_cast<const ito::StringMeta*>(param.getMeta());
        defaultStr = (translate ? QObject::tr("Default: \"%1\"") : QString("Default: \"%1\""));
        const char* str = param.getVal<const char*>();

        if (str == nullptr)
        {
            defaultStr = (translate ? QObject::tr("Default: <nullptr / None>") : QString("Default: <nullptr / None>"));
        }
        else if (strlen(str) == 0)
        {
            defaultStr = (translate ? QObject::tr("Default: <empty str>") : QString("Default: <empty str>"));
        }
        else
        {
            defaultStr = defaultStr.arg(QLatin1String(str));
        }

        if (pMeta)
        {
            meta = renderStrMetaInfo(pMeta, translate);
            meta += ", " + defaultStr;
        }
        else
        {
            meta = defaultStr;
        }
    }
    break;
    case ito::ParamBase::CharArray: {
        const ito::ParamMeta* m = param.getMeta();
        const auto pMeta = dynamic_cast<const ito::CharArrayMeta*>(param.getMeta());

        pythonLikeTypename = "Sequence[int (char)]";

        ito::ParamMeta::tRepresentation repr = ito::ParamMeta::PureNumber;

        if (pMeta)
        {
            repr = pMeta->getRepresentation();
        }

        defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"));
        defaultStr =
            defaultStr.arg(renderArrayContent(param.getVal<const char*>(), param.getLen(), repr));

        if (m && m->getType() == ito::ParamMeta::rttiCharArrayMeta)
        {
            meta = renderListMetaInfo(pMeta, translate);
            meta += ", " + renderCharMetaInfo(pMeta, translate);
            meta += ", " + defaultStr;
        }
        else
        {
            meta = defaultStr;
        }
    }

    break;
    case ito::ParamBase::IntArray: {
        const ito::ParamMeta* m = param.getMeta();

        ito::ParamMeta::tRepresentation repr = ito::ParamMeta::PureNumber;

        defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"));


        if (!m)
        {
            pythonLikeTypename = "Sequence[int]";
            defaultStr = defaultStr.arg(
                renderArrayContent(param.getVal<const int*>(), param.getLen(), repr));

            meta = defaultStr;
        }
        else if (m->getType() == ito::ParamMeta::rttiIntArrayMeta)
        {
            pythonLikeTypename = "Sequence[int]";

            const auto pMeta = dynamic_cast<const ito::IntArrayMeta*>(param.getMeta());
            defaultStr = defaultStr.arg(renderArrayContent(
                param.getVal<const int*>(), param.getLen(), pMeta->getRepresentation()));
            meta = renderListMetaInfo(pMeta, translate);
            meta += ", " + renderIntMetaInfo(pMeta, translate);
            meta += ", " + defaultStr;
        }
        else if (m->getType() == ito::ParamMeta::rttiIntervalMeta)
        {
            pythonLikeTypename = "interval [first: int, last: int]";

            const auto pMeta = dynamic_cast<const ito::IntervalMeta*>(param.getMeta());
            defaultStr = defaultStr.arg(renderArrayContent(
                param.getVal<const int*>(), param.getLen(), pMeta->getRepresentation()));
            meta += ", " + renderIntMetaInfo(pMeta, translate);
            meta += ", " + defaultStr;
        }
        else if (m->getType() == ito::ParamMeta::rttiRangeMeta)
        {
            pythonLikeTypename = "range [first: int, last: int]";

            const auto pMeta = dynamic_cast<const ito::RangeMeta*>(param.getMeta());
            defaultStr = defaultStr.arg(renderArrayContent(
                param.getVal<const int*>(), param.getLen(), pMeta->getRepresentation()));
            meta += ", " + renderIntMetaInfo(pMeta, translate);
            meta += ", " + defaultStr;
        }
        else if (m->getType() == ito::ParamMeta::rttiRectMeta)
        {
            pythonLikeTypename = "rectangle [left: int, top: int, width: int, height: int]";

            const auto pMeta = dynamic_cast<const ito::RectMeta*>(param.getMeta());
            defaultStr = defaultStr.arg(
                renderArrayContent(param.getVal<const int*>(), param.getLen(), repr));

            const auto widthMeta = pMeta->getWidthRangeMeta();
            const auto heightMeta = pMeta->getHeightRangeMeta();
            meta = (translate ? QObject::tr("Width: %1") : QLatin1String("Width: %1"))
                       .arg(renderIntMetaInfo(&widthMeta, translate));
            meta = (translate ? QObject::tr("Height: %1") : QLatin1String("Height: %1"))
                       .arg(renderIntMetaInfo(&heightMeta, translate));
            meta += ", " + defaultStr;
        }
    }

    break;
    case ito::ParamBase::DoubleArray: {
        const ito::ParamMeta* m = param.getMeta();

        defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"));


        if ((m && m->getType() == ito::ParamMeta::rttiDoubleArrayMeta) || !m)
        {
            pythonLikeTypename = "Sequence[float]";

            const auto pMeta = dynamic_cast<const ito::DoubleArrayMeta*>(param.getMeta());

            if (pMeta)
            {
                meta = renderListMetaInfo(pMeta, translate);
                meta += ", " + renderDoubleMetaInfo(pMeta, translate);
                defaultStr = defaultStr.arg(renderArrayContent(
                    param.getVal<const double*>(),
                    param.getLen(),
                    pMeta->getDisplayNotation(),
                    pMeta->getDisplayPrecision()));
                meta += ", " + defaultStr;
            }
            else
            {
                defaultStr = defaultStr.arg(renderArrayContent(
                    param.getVal<const double*>(), param.getLen(), ito::DoubleMeta::Automatic, 3));
                meta = defaultStr;
            }
        }
        else if (m && m->getType() == ito::ParamMeta::rttiDoubleIntervalMeta)
        {
            pythonLikeTypename = "interval [first: float, last: float]";

            const auto pMeta = dynamic_cast<const ito::DoubleIntervalMeta*>(param.getMeta());

            if (pMeta)
            {
                meta += ", " + renderDoubleMetaInfo(pMeta, translate);
                defaultStr = defaultStr.arg(renderArrayContent(
                    param.getVal<const double*>(),
                    param.getLen(),
                    pMeta->getDisplayNotation(),
                    pMeta->getDisplayPrecision()));
                meta += ", " + defaultStr;
            }
            else
            {
                defaultStr = defaultStr.arg(renderArrayContent(
                    param.getVal<const double*>(), param.getLen(), ito::DoubleMeta::Automatic, 3));
                meta = defaultStr;
            }
        }
        else
        {
            defaultStr = defaultStr.arg(renderArrayContent(
                param.getVal<const double*>(), param.getLen(), ito::DoubleMeta::Automatic, 3));
            meta = defaultStr;
        }
    }

    break;
    case ito::ParamBase::StringList: {
        pythonLikeTypename = "Sequence[str]";
        const ito::ParamMeta* m = param.getMeta();

        defaultStr = (translate ? QObject::tr("Default: %1") : QString("Default: %1"));
        defaultStr = defaultStr.arg(
            renderArrayContent(param.getVal<const ito::ByteArray*>(), param.getLen()));

        if ((m && m->getType() == ito::ParamMeta::rttiStringListMeta) || !m)
        {
            const auto pMeta = dynamic_cast<const ito::StringListMeta*>(param.getMeta());

            if (pMeta)
            {
                meta = renderListMetaInfo(pMeta, translate);
                meta += ", " +
                    (translate ? QObject::tr("Value rules: ") : QLatin1String("Value rules: ")) +
                    renderStrMetaInfo(pMeta, translate);
                meta += ", " + defaultStr;
            }
            else
            {
                meta = defaultStr;
            }
        }
    }

    break;
    case ito::ParamBase::ComplexArray: {
        pythonLikeTypename = "Sequence[complex], double precision";
    }

    break;
    case ito::ParamBase::DObjPtr: {
        pythonLikeTypename = "itom.dataObject";

        const ito::DObjMeta* pMeta = dynamic_cast<const ito::DObjMeta*>(param.getMeta());

        if (pMeta)
        {
            meta = "";

            if (pMeta->getMinDim() > 0)
            {
                meta += (translate ? QObject::tr("Min. dims: %1") : QString("Min. dims: %1"))
                            .arg(pMeta->getMinDim());
            }

            if (pMeta->getMaxDim() > 0)
            {
                if (meta != "")
                {
                    meta += ", ";
                }

                meta += (translate ? QObject::tr("Max. dims: %1") : QString("Max. dims: %1"))
                            .arg(pMeta->getMaxDim());
            }
        }
    }

    break;
    case ito::ParamBase::PointCloudPtr: {
        pythonLikeTypename = "itom.pointCloud";
    }

    break;
    case ito::ParamBase::PointPtr: {
        pythonLikeTypename = "itom.point";
    }

    break;
    case ito::ParamBase::PolygonMeshPtr: {
        pythonLikeTypename = "itom.polygonMesh";
    }

    break;
    case ito::ParamBase::HWRef: {
        pythonLikeTypename = "Union[itom.dataIO, itom.actuator]";

        if (param.getMeta() != nullptr)
        {
            const ito::HWMeta* pMeta = dynamic_cast<const ito::HWMeta*>(param.getMeta());
            ito::ByteArray name = pMeta->getHWAddInName();

            if (name.length() > 0)
            {
                temp = translate ? QObject::tr("Only plugin \"%1\" is allowed")
                                 : QString("Only plugin \"%1\" is allowed");
                meta = temp.arg(name.data());
            }
            else
            {
                meta = "";
                bool dataIOIncluded = false;

                if (pMeta->getMinType() & ito::typeActuator)
                {
                    meta.append("Actuator, ");
                }
                if (pMeta->getMinType() & ito::typeAlgo)
                {
                    meta.append("Algorithms, ");
                }
                if (pMeta->getMinType() & ito::typeGrabber)
                {
                    meta.append("DataIO::Grabber, ");
                    dataIOIncluded = true;
                }
                else if (pMeta->getMinType() & ito::typeADDA)
                {
                    meta.append("DataIO::ADDA, ");
                    dataIOIncluded = true;
                }
                else if (pMeta->getMinType() & ito::typeRawIO)
                {
                    meta.append("DataIO::Raw IO, ");
                    dataIOIncluded = true;
                }
                else if (!dataIOIncluded && pMeta->getMinType() & ito::typeDataIO)
                {
                    meta.append("DataIO, ");
                }

                meta = meta.mid(0, meta.size() - 2);
                meta = (translate ? QObject::tr("Allowed plugin type(s): \"%1\".")
                                  : QString("Allowed plugin type(s): \"%1\"."))
                           .arg(meta);
            }
        }
    }

    break;
    }

    return meta;
}

} // end namespace ito
