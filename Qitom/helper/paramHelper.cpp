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

#include "paramHelper.h"

#include "../../common/addInInterface.h"

#include <qobject.h>
#include <qregexp.h>


namespace ito {

    //----------------------------------------------------------------------------------------------------------------------------------
    tCompareResult ParamHelper::compareParam(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret)
    {
        //check whether type is equal
        if (paramTemplate.getType() != param.getType())
        {
            ret += ito::RetVal::format(ito::retError, 0, QObject::tr("Types of parameter '%s' is unequal to required type of interface parameter '%s'").toLatin1().data(), param.getName(), paramTemplate.getName());
            return tCmpFailed;
        }

        //check in/out flags
        int inOutFlags = ito::ParamBase::In | ito::ParamBase::Out;
        if ((paramTemplate.getFlags() & inOutFlags) != (param.getFlags() & inOutFlags))
        {
            ret += ito::RetVal::format(ito::retError, 0, QObject::tr("In/Out flags of parameter '%s' are unequal to required flags of interface parameter '%s'").toLatin1().data(), param.getName(), paramTemplate.getName());
            return tCmpFailed;
        }

        //check meta information
        const ito::ParamMeta *metaTemplate = paramTemplate.getMeta();
        const ito::ParamMeta *meta = param.getMeta();

        return compareMetaParam(metaTemplate, meta, paramTemplate.getName(), param.getName(), ret);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    tCompareResult ParamHelper::compareMetaParam(const ito::ParamMeta *metaTemplate, const ito::ParamMeta *meta, const char* nameTemplate, const char *name, ito::RetVal &ret)
    {
        if (metaTemplate == NULL && meta == NULL) 
        {
            return tCmpEqual;
        }
        else if (meta == NULL)
        {
            return tCmpCompatible; //param is compatible to paramTemplate, since it has no meta block defined, but paramTemplate has. A meta bock always is more restrictive than no.
        }
        else if (metaTemplate == NULL)
        {
            ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The parameter '%s' is restricted by meta information while the interface parameter '%s' is not.").toLatin1().data(), name, nameTemplate);
            return tCmpFailed;
        }

        if (metaTemplate->getType() != meta->getType())
        {
            ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
            return tCmpFailed;
        }

        switch(metaTemplate->getType())
        {
            case ito::ParamBase::Int:
            {
                const ito::IntMeta *mT = static_cast<const ito::IntMeta*>(metaTemplate);
                const ito::IntMeta *m = static_cast<const ito::IntMeta*>(meta);
                if (!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }
                if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
                {
                    return tCmpEqual;
                }
                else if (m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
                {
                    return tCmpCompatible;
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The allowed integer range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }
            }
            break;

            case ito::ParamBase::Char:
            {
                const ito::CharMeta *mT = static_cast<const ito::CharMeta*>(metaTemplate);
                const ito::CharMeta *m = static_cast<const ito::CharMeta*>(meta);
                if (!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }
                if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
                {
                    return tCmpEqual;
                }
                else if (m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
                {
                    return tCmpCompatible;
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The allowed char range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }
            }
            break;

            case ito::ParamBase::Double:
            {
                const ito::DoubleMeta *mT = static_cast<const ito::DoubleMeta*>(metaTemplate);
                const ito::DoubleMeta *m = static_cast<const ito::DoubleMeta*>(meta);
                if (!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }
                if (m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
                {
                    return tCmpEqual;
                }
                else if (m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
                {
                    return tCmpCompatible;
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The allowed double range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }
            }
            break;

            case ito::ParamBase::String:
            {
                const ito::StringMeta *mT = static_cast<const ito::StringMeta*>(metaTemplate);
                const ito::StringMeta *m = static_cast<const ito::StringMeta*>(meta);
                if (!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }

                if (m->getStringType() != mT->getStringType())
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The string type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }

                const char* sT = NULL;
                bool found = false;

                for (int i = 0; i < mT->getLen(); i++)
                {
                    sT = mT->getString(i);
                    found = false;

                    for (int j = 0; j < m->getLen(); j++)
                    {
                        if (strcmp(sT, m->getString(j)) == 0)
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("String '%s', requested by meta data of interface parameter '%s' could not be found in meta data of parameter '%s'.").toLatin1().data(), sT, nameTemplate, name);
                        return tCmpFailed;
                    }
                }

                if (m->getLen() == mT->getLen())
                {
                    return tCmpEqual;
                }
                else
                {
                    return tCmpCompatible;
                }
                
            }
            break;

            case ito::ParamBase::DObjPtr & ito::paramTypeMask:
            {
                const ito::DObjMeta *mT = static_cast<const ito::DObjMeta*>(metaTemplate);
                const ito::DObjMeta *m = static_cast<const ito::DObjMeta*>(meta);
                if (!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }

                //all bits in allowedTypes of mT must be set in m, too
                if ((m->getAllowedTypes() & mT->getAllowedTypes()) != mT->getAllowedTypes())
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The allowed data object types of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }

                if (m->getAllowedTypes() == mT->getAllowedTypes())
                {
                    if (m->getMinDim() == mT->getMinDim() && m->getMaxDim() == mT->getMaxDim())
                    {
                        return tCmpEqual;
                    }
                    else if (m->getMinDim() <= mT->getMinDim() && m->getMaxDim() >= mT->getMaxDim())
                    {
                        return tCmpCompatible;
                    }
                    else
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The minimum and maximum dimensions of the data object of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                        return tCmpFailed;
                    }
                }
                else
                {
                    if (m->getMinDim() <= mT->getMinDim() && m->getMaxDim() >= mT->getMaxDim())
                    {
                        return tCmpCompatible;
                    }
                    else
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The minimum and maximum dimensions of the data object of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                        return tCmpFailed;
                    }
                }
                
            }
            break;

            case ito::ParamBase::HWRef & ito::paramTypeMask:
            {
                const ito::HWMeta *mT = static_cast<const ito::HWMeta*>(metaTemplate);
                const ito::HWMeta *m = static_cast<const ito::HWMeta*>(meta);
                if (!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                    return tCmpFailed;
                }

                if (mT->getHWAddInName() != NULL)
                {
                    if (m->getHWAddInName() != NULL)
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The meta data of the interface parameter '%s' requires a plugin with name '%s', but parameter '%s' does it not.").toLatin1().data(), nameTemplate, mT->getHWAddInName(), name);
                        return tCmpFailed;
                    }
                    else if (strcmp(mT->getHWAddInName(), m->getHWAddInName()) != 0)
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("Both parameter '%s' and interface parameter '%s' require different plugins.").toLatin1().data(), name, nameTemplate);
                        return tCmpFailed;
                    }
                }

                //every bit set in m->getMinType() must be set in mT->getMinType()
                if (mT->getMinType() == m->getMinType())
                {
                    return tCmpEqual;
                }
                else if ((mT->getMinType() & m->getMinType()) == m->getMinType())
                {
                    return tCmpCompatible;
                }

                ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The minimum plugin type bit mask of parameter '%s' is more restrictive than this of the interface parameter '%s'.").toLatin1().data(), name, nameTemplate);
                return tCmpFailed;
                
            }
            break;

            default:
            {
                ret += ito::RetVal::format(ito::retError, 0, QObject::tr("meta data of interface parameter '%s' is unknown.").toLatin1().data(), nameTemplate);
                return tCmpFailed;
            }
            
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateStringMeta(const ito::StringMeta *meta, const char* value, bool mandatory)
    {
        QString pattern;
        if (meta && meta->getLen() > 0 && value)
        {
            bool found = false;
            QRegExp reg;
            switch(meta->getStringType())
            {
            case ito::StringMeta::String:
                reg.setPatternSyntax(QRegExp::FixedString);
                break;
            case ito::StringMeta::Wildcard:
                reg.setPatternSyntax(QRegExp::Wildcard);
                break;
            case ito::StringMeta::RegExp:
                reg.setPatternSyntax(QRegExp::RegExp);
                break;
            }

            for (int i = 0; i < meta->getLen(); i++)
            {
                pattern = meta->getString(i);
                reg.setPattern(pattern);
                if (reg.indexIn(value, 0) > -1)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return ito::RetVal::format(ito::retError, 0, QObject::tr("String '%s' does not fit to given string-constraints.").toLatin1().data(), value);
            }

        }

        if (mandatory && (value == NULL))
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("AddIn must not be NULL").toLatin1().data());
        }

        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateDoubleMeta(const ito::DoubleMeta *meta, double value)
    {
        if (meta)
        {
            double minVal = meta->getMin();
            double maxVal = meta->getMax();

            if (value < minVal || value > maxVal)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal).toLatin1().data());
            }

            double step = meta->getStepSize();
            if (step > 0.0)   
            {
                double div = (value - minVal) / step;
                div = qRound(div) * step + minVal;
                if (std::abs(div - value) > std::numeric_limits<double>::epsilon())
                {
                    return ito::RetVal(ito::retError, 0, QObject::tr("value does not fit to given step size [%1:%2:%3]").arg(minVal).arg(step).arg(maxVal).toLatin1().data());
                }
            }
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateIntMeta(const ito::IntMeta *meta, int value)
    {
        if (meta)
        {
            int minVal = meta->getMin();
            int maxVal = meta->getMax();

            if (value < minVal || value > maxVal)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal).toLatin1().data());
            }

            int step = meta->getStepSize();
            if (step > 1 && ((value - minVal) % step) != 0)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value does not fit to given step size [%1:%2:%3]").arg(minVal).arg(step).arg(maxVal).toLatin1().data());
            }
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateCharMeta(const ito::CharMeta *meta, char value)
    {
        if (meta)
        {
            char minVal = meta->getMin();
            char maxVal = meta->getMax();

            if (value < minVal || value > maxVal)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value out of range [%1, %2]").arg(minVal).arg(maxVal).toLatin1().data());
            }

            char step = meta->getStepSize();
            if (step > 1 && ((value - minVal) % step) != 0)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("value does not fit to given step size [%1:%2:%3]").arg(minVal).arg(step).arg(maxVal).toLatin1().data());
            }
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateHWMeta(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory)
    {
        if (meta && value)
        {
            int minType = meta->getMinType();
            if ((minType & value->getBasePlugin()->getType()) != minType)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("AddIn does not fit to minimum required type(s).").toLatin1().data()); 
            }
            if (meta->getHWAddInName() && QString::compare(meta->getHWAddInName(), value->getBasePlugin()->objectName(), Qt::CaseInsensitive) != 0)
            {
                return ito::RetVal::format(ito::retError, 0, QObject::tr("AddIn must be of the following plugin: '%s'.").toLatin1().data(), meta->getHWAddInName());
            }
        }
        else if (mandatory && value == NULL)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("AddIn must not be NULL").toLatin1().data());
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateParam(const ito::Param &templateParam, const ito::ParamBase &param, bool strict /*= true*/, bool mandatory /*= false*/)
    {
        ito::RetVal retVal;
        bool hasIndex = false;
        int index;

        //check whether param has an index
        QRegExp rx("^([a-zA-Z]+\\w*)(\\[(\\d+)\\])(:(.*)){0,1}$");
        if (rx.indexIn(param.getName()) >= 0)
        {
            hasIndex = true;
            index = rx.capturedTexts()[3].toInt();
        }

        if (!hasIndex && (templateParam.getType() == param.getType()))
        {

            switch(templateParam.getType())
            {
            case ito::ParamBase::Char:
                {
                    retVal += validateCharMeta(dynamic_cast<const ito::CharMeta*>(templateParam.getMeta()), param.getVal<char>()); 
                }
                break;
            case ito::ParamBase::Int:
                {
                    retVal += validateIntMeta(dynamic_cast<const ito::IntMeta*>(templateParam.getMeta()), param.getVal<int>()); 
                }
                break;
            case ito::ParamBase::Double:
                {
                    retVal += validateDoubleMeta(dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta()), param.getVal<double>()); 
                }
                break;
            case ito::ParamBase::CharArray:
                {
                    const ito::CharMeta *meta = dynamic_cast<const ito::CharMeta*>(templateParam.getMeta());
                    char* vals = param.getVal<char*>();
                    if (meta)
                    {
                        for (int i = 0; i < param.getLen(); i++)
                        {
                            retVal += validateCharMeta(meta, vals[i]);
                        }
                    }
                }
                break;
            case ito::ParamBase::IntArray:
                {
                    const ito::IntMeta *meta = dynamic_cast<const ito::IntMeta*>(templateParam.getMeta());
                    int* vals = param.getVal<int*>();
                    if (meta)
                    {
                        for (int i = 0; i < param.getLen(); i++)
                        {
                            retVal += validateIntMeta(meta, vals[i]);
                        }
                    }
                }
                break;
            case ito::ParamBase::DoubleArray:
                {
                    const ito::DoubleMeta *meta = dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta());
                    double* vals = param.getVal<double*>();
                    if (meta)
                    {
                        for (int i = 0; i < param.getLen(); i++)
                        {
                            retVal += validateDoubleMeta(meta, vals[i]);
                        }
                    }
                }
                break;
            case ito::ParamBase::String:
                {
                    retVal += validateStringMeta(dynamic_cast<const ito::StringMeta*>(templateParam.getMeta()), param.getVal<char*>(), mandatory); 
                }
                break;
            case ito::ParamBase::HWRef & ito::paramTypeMask:
                {
                    retVal += validateHWMeta(dynamic_cast<const ito::HWMeta*>(templateParam.getMeta()), (ito::AddInBase*)param.getVal<void*>(), mandatory);
                }
                break;
            }
        }
        else if (hasIndex && (templateParam.getType() & ito::ParamBase::Pointer) && (templateParam.getType() == (param.getType() ^ ito::ParamBase::Pointer)))
        {
            if (index < 0 || index >= templateParam.getLen())
            {
                retVal += ito::RetVal::format(ito::retError, 0, QObject::tr("Index value is out of range [0, %i]").toLatin1().data(), templateParam.getLen()-1);
            }

            switch(templateParam.getType())
            {
            case ito::ParamBase::CharArray:
                {
                    retVal += validateCharMeta(dynamic_cast<const ito::CharMeta*>(templateParam.getMeta()), param.getVal<char>()); 
                }
                break;
            case ito::ParamBase::IntArray:
                {
                    retVal += validateIntMeta(dynamic_cast<const ito::IntMeta*>(templateParam.getMeta()), param.getVal<int>()); 
                }
                break;
            case ito::ParamBase::DoubleArray:
                {
                    retVal += validateDoubleMeta(dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta()), param.getVal<double>()); 
                }
                break;
            default:
                {
                    retVal += ito::RetVal(ito::retError, 0, QObject::tr("Index-based parameter name requires an array-type parameter.").toLatin1().data());
                }
                break;
            }
        }
        else if (!strict)
        {
            bool ok = false;
            ito::ParamBase p = convertParam(param, templateParam.getType(), &ok);
            if (ok)
            {
                retVal += validateParam(templateParam, p, true);
            }
            else
            {
                retVal += ito::RetVal(ito::retError, 0, QObject::tr("Parameter could not be converted to destination type.").toLatin1().data());
            }
        }
        else
        {
            retVal += ito::RetVal(ito::retError, 0, QObject::tr("type of parameter does not fit to requested parameter type").toLatin1().data());
        }

        return retVal;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::ParamBase ParamHelper::convertParam(const ito::ParamBase &source, int destType, bool *ok /*= NULL*/)
    {
        int sourceType = source.getType();
        bool ok2;
        if (ok) *ok = true;

        if (sourceType == (destType & (int)ito::paramTypeMask)) return source;


        switch(destType & ito::paramTypeMask)
        {
        case ito::ParamBase::Int:
            {
                if (source.isNumeric())
                {
                    return ito::ParamBase(source.getName(), destType, source.getVal<int>());
                }
                else if (sourceType & ito::ParamBase::String)
                {
                    int val = QString(source.getVal<char*>()).toInt(&ok2);
                    if (ok2)
                    {
                        return ito::ParamBase(source.getName(), destType, val);
                    }
                }
            }
            break;
        
        case ito::ParamBase::Char:
            {
                if (source.isNumeric())
                {
                    return ito::ParamBase(source.getName(), destType, source.getVal<char>());
                }
                else if (sourceType & ito::ParamBase::String)
                {
                    char val = QString(source.getVal<char*>()).toShort(&ok2);
                    if (ok2)
                    {
                        return ito::ParamBase(source.getName(), destType, val);
                    }
                }
            }
            break;
        
        case ito::ParamBase::Double:
            {
                if (source.isNumeric())
                {
                    return ito::ParamBase(source.getName(), destType, source.getVal<double>());
                }
                else if (sourceType & ito::ParamBase::String)
                {
                    double val = QString(source.getVal<char*>()).toDouble(&ok2);
                    if (ok2)
                    {
                        return ito::ParamBase(source.getName(), destType, val);
                    }
                }
            }
            break;

        case ito::ParamBase::String:
            {
                if (source.isNumeric())
                {
                    return ito::ParamBase(source.getName(), destType, QString::number(source.getVal<double>()).toLatin1().data());
                }
            }
            break;
        }

        if (ok) *ok = false;
        return ParamBase();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::getParamFromMapByKey(QMap<QString, ito::Param> &paramMap, const QString &key, QMap<QString, ito::Param>::iterator &found, bool errorIfReadOnly)
    {
        if (key == "")
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("Name of given parameter is empty.").toLatin1().data());
        }

        QMap<QString, ito::Param>::iterator it = paramMap.find(key);
        if (it != paramMap.end())
        {
            if (errorIfReadOnly && (it->getFlags() & ito::ParamBase::Readonly))
            {
                return ito::RetVal::format(ito::retError, 0, QObject::tr("Parameter '%1' is read only.").arg(key).toLatin1().data());
            }

            found = it;
        }
        else
        {
            return ito::RetVal::format(ito::retError, 0, QObject::tr("Parameter '%1' not found.").arg(key).toLatin1().data());
        }

        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! parses parameter name with respect to regular expression, assigned for parameter-communcation with plugins
    /*!
        This method parses any parameter-name with respect to the rules defined for possible names of plugin-parameters.

        The regular expression used for the check is "^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$"

        Then the components are:

        [0] full string
        [1] PARAMNAME
        [2] [INDEX] or empty-string if no index is given
        [3] INDEX or empty-string if no index is given
        [4] :ADDITIONALTAG or empty-string if no tag is given
        [5] ADDITIONALTAG or empty-string if no tag is given

        \param [in] name is the raw parameter name
        \param [out] paramName is the real parameter name (first part of name; part before the first opening bracket ('[') or if not available the first colon (':'))
        \param [out] hasIndex indicates whether the name contains an index part (defined by a number within two brackets (e.g. '[NUMBER]'), which has to be appended to the paramName
        \param [out] index is the fixed-point index value or -1 if hasIndex is false
        \param [out] additionalTag is the remaining string of name which is the part after the first colon (':'). If an index part exists, the first colon after the index part is taken.
    */
    ito::RetVal ParamHelper::parseParamName(const QString &name, QString &paramName, bool &hasIndex, int &index, QString &additionalTag)
    {
        ito::RetVal retValue = ito::retOk;
        paramName = QString();
        hasIndex = false;
        index = -1;
        additionalTag = QString();

        QRegExp rx("^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$");
        if (rx.indexIn(name) == -1)
        {
            retValue += ito::RetVal(ito::retError, 0, QObject::tr("invalid parameter name").toLatin1().data());
        }
        else
        {
            QStringList pname = rx.capturedTexts();
            paramName = pname[1];
            if (pname.size()>=4)
            {
                if (!pname[3].isEmpty())
                {
                    index = pname[3].toInt(&hasIndex);
                }
            }
            if (pname.size() >=6)
            {
                additionalTag = pname[5];
            }
        }
        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*static*/ ito::RetVal ParamHelper::getItemFromArray(const ito::Param &arrayParam, const int index, ito::Param &itemParam)
    {
        ito::RetVal retval;
        int len = arrayParam.getLen();

        if (index < 0 || index >= len)
        {
            return ito::RetVal::format(ito::retError, 0, QObject::tr("index is ouf of range [0, %i]").toLatin1().data(), len-1);
        }

        QString newName = QString("%1[%2]").arg(arrayParam.getName()).arg(index);

        switch(arrayParam.getType())
        {
        case ito::ParamBase::IntArray & ito::paramTypeMask:
            {
                int *val = arrayParam.getVal<int*>();
                itemParam = ito::Param(newName.toLatin1().data(), (arrayParam.getType(false) ^ ito::ParamBase::IntArray) | ito::ParamBase::Int, val[index], NULL, arrayParam.getInfo());
                const ito::ParamMeta *m = arrayParam.getMeta();
                if (m)
                {
                    itemParam.copyMetaFrom(m);
                }
            }
            break;
        case ito::ParamBase::DoubleArray & ito::paramTypeMask:
            {
                double *val = arrayParam.getVal<double*>();
                itemParam = ito::Param(newName.toLatin1().data(), (arrayParam.getType(false) ^ ito::ParamBase::DoubleArray) | ito::ParamBase::Double, val[index], NULL, arrayParam.getInfo());
                const ito::ParamMeta *m = arrayParam.getMeta();
                if (m)
                {
                    itemParam.copyMetaFrom(m);
                }
            }
            break;
        case ito::ParamBase::CharArray & ito::paramTypeMask:
            {
                char *val = arrayParam.getVal<char*>();
                itemParam = ito::Param(newName.toLatin1().data(), (arrayParam.getType(false) ^ ito::ParamBase::CharArray) | ito::ParamBase::Char, val[index], NULL, arrayParam.getInfo());
                const ito::ParamMeta *m = arrayParam.getMeta();
                if (m)
                {
                    itemParam.copyMetaFrom(m);
                }
            }
            break;
        default:
            retval += ito::RetVal(ito::retError, 0, QObject::tr("param is no array").toLatin1().data());
            break;
        }
        return retval;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*static*/ ito::Param ParamHelper::getParam(const ito::Param &param, const bool hasIndex, const int index, ito::RetVal &ret)
    {
        if (!hasIndex)
        {
            return param;
        }
        else
        {
            //check whether param is an array type
            ito::uint32 type = param.getType();
            if (type == (ito::ParamBase::IntArray & ito::paramTypeMask) ||
                type == (ito::ParamBase::DoubleArray & ito::paramTypeMask) ||
                type == (ito::ParamBase::CharArray & ito::paramTypeMask))
            {
                ito::Param p;
                ret += getItemFromArray(param, index, p);
                return p;
            }
            else
            {
                ret += ito::RetVal(ito::retError, 0, QObject::tr("Paramater is no array type. Indexing not possible.").toLatin1().data());
            }
        }
        return ito::Param();
    }

} //end namespace ito
