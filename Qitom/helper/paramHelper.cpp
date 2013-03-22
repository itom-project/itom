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

#include "paramHelper.h"

#include "../../common/addInInterface.h"

#include <qobject.h>
#include <qregexp.h>


namespace ito {

    //----------------------------------------------------------------------------------------------------------------------------------

    tCompareResult ParamHelper::compareParam(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret)
    {
        //check whether type is equal
        if(paramTemplate.getType() != param.getType())
        {
            ret += ito::RetVal::format(ito::retError,0, QObject::tr("Types of parameter '%s' is unequal to required type of interface parameter '%s'").toAscii().data(),param.getName(),paramTemplate.getName());
            return tCmpFailed;
        }

        //check in/out flags
        int inOutFlags = ito::ParamBase::In | ito::ParamBase::Out;
        if( (paramTemplate.getFlags() & inOutFlags) != (param.getFlags() & inOutFlags) )
        {
            ret += ito::RetVal::format(ito::retError,0, QObject::tr("In/Out flags of parameter '%s' are unequal to required flags of interface parameter '%s'").toAscii().data(),param.getName(),paramTemplate.getName());
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
        if(metaTemplate == NULL && meta == NULL) 
        {
            return tCmpEqual;
        }
        else if(meta == NULL)
        {
            return tCmpCompatible; //param is compatible to paramTemplate, since it has no meta block defined, but paramTemplate has. A meta bock always is more restrictive than no.
        }
        else if(metaTemplate == NULL)
        {
            ret += ito::RetVal::format(ito::retError,0, QObject::tr("The parameter '%s' is restricted by meta information while the interface parameter '%s' is not.").toAscii().data(),name,nameTemplate);
            return tCmpFailed;
        }

        if(metaTemplate->getType() != meta->getType())
        {
            ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
            return tCmpFailed;
        }

        switch( metaTemplate->getType() )
        {
            case ito::ParamBase::Int:
            {
                const ito::IntMeta *mT = static_cast<const ito::IntMeta*>(metaTemplate);
                const ito::IntMeta *m = static_cast<const ito::IntMeta*>(meta);
                if(!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }
                if(m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
                {
                    return tCmpEqual;
                }
                else if(m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
                {
                    return tCmpCompatible;
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The allowed integer range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }
            }
            break;

            case ito::ParamBase::Char:
            {
                const ito::CharMeta *mT = static_cast<const ito::CharMeta*>(metaTemplate);
                const ito::CharMeta *m = static_cast<const ito::CharMeta*>(meta);
                if(!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }
                if(m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
                {
                    return tCmpEqual;
                }
                else if(m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
                {
                    return tCmpCompatible;
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The allowed char range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }
            }
            break;

            case ito::ParamBase::Double:
            {
                const ito::DoubleMeta *mT = static_cast<const ito::DoubleMeta*>(metaTemplate);
                const ito::DoubleMeta *m = static_cast<const ito::DoubleMeta*>(meta);
                if(!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }
                if(m->getMax() == mT->getMax() && m->getMin() == mT->getMin())
                {
                    return tCmpEqual;
                }
                else if(m->getMax() >= mT->getMax() && m->getMin() <= mT->getMin())
                {
                    return tCmpCompatible;
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The allowed double range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }
            }
            break;

            case ito::ParamBase::String:
            {
                const ito::StringMeta *mT = static_cast<const ito::StringMeta*>(metaTemplate);
                const ito::StringMeta *m = static_cast<const ito::StringMeta*>(meta);
                if(!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                if(m->getStringType() != mT->getStringType())
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The string type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                const char* sT = NULL;
                bool found = false;

                for(size_t i=0;i<mT->getLen();i++)
                {
                    sT = mT->getString(i);
                    found = false;

                    for(size_t j=0;j<m->getLen();j++)
                    {
                        if( strcmp(sT, m->getString(j)) == 0)
                        {
                            found = true;
                            break;
                        }
                    }

                    if(!found)
                    {
                        ret += ito::RetVal::format(ito::retError,0, QObject::tr("String '%s', requested by meta data of interface parameter '%s' could not be found in meta data of parameter '%s'.").toAscii().data(),sT,nameTemplate,name);
                        return tCmpFailed;
                    }
                }

                if(m->getLen() == mT->getLen())
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
                if(!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                //all bits in allowedTypes of mT must be set in m, too
                if( (m->getAllowedTypes() & mT->getAllowedTypes()) != mT->getAllowedTypes() )
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The allowed data object types of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                if( m->getAllowedTypes() == mT->getAllowedTypes() )
                {
                    if(m->getMinDim() == mT->getMinDim() && m->getMaxDim() == mT->getMaxDim())
                    {
                        return tCmpEqual;
                    }
                    else if(m->getMinDim() <= mT->getMinDim() && m->getMaxDim() >= mT->getMaxDim())
                    {
                        return tCmpCompatible;
                    }
                    else
                    {
                        ret += ito::RetVal::format(ito::retError,0, QObject::tr("The minimum and maximum dimensions of the data object of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                        return tCmpFailed;
                    }
                }
                else
                {
                    if(m->getMinDim() <= mT->getMinDim() && m->getMaxDim() >= mT->getMaxDim())
                    {
                        return tCmpCompatible;
                    }
                    else
                    {
                        ret += ito::RetVal::format(ito::retError,0, QObject::tr("The minimum and maximum dimensions of the data object of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                        return tCmpFailed;
                    }
                }
                
            }
            break;

            case ito::ParamBase::HWRef & ito::paramTypeMask:
            {
                const ito::HWMeta *mT = static_cast<const ito::HWMeta*>(metaTemplate);
                const ito::HWMeta *m = static_cast<const ito::HWMeta*>(meta);
                if(!mT || !m)
                {
                    ret += ito::RetVal::format(ito::retError,0, QObject::tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                if(mT->getHWAddInName() != NULL)
                {
                    if(m->getHWAddInName() != NULL)
                    {
                        ret += ito::RetVal::format(ito::retError,0, QObject::tr("The meta data of the interface parameter '%s' requires a plugin with name '%s', but parameter '%s' does it not.").toAscii().data(),nameTemplate,mT->getHWAddInName(),name);
                        return tCmpFailed;
                    }
                    else if( strcmp(mT->getHWAddInName(),m->getHWAddInName()) != 0)
                    {
                        ret += ito::RetVal::format(ito::retError,0, QObject::tr("Both parameter '%s' and interface parameter '%s' require different plugins.").toAscii().data(),name,nameTemplate);
                        return tCmpFailed;
                    }
                }

                //every bit set in m->getMinType() must be set in mT->getMinType()
                if( mT->getMinType() == m->getMinType() )
                {
                    return tCmpEqual;
                }
                else if( (mT->getMinType() & m->getMinType()) == m->getMinType() )
                {
                    return tCmpCompatible;
                }

                ret += ito::RetVal::format(ito::retError,0, QObject::tr("The minimum plugin type bit mask of parameter '%s' is more restrictive than this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                return tCmpFailed;
                
            }
            break;

            default:
            {
                ret += ito::RetVal::format(ito::retError,0, QObject::tr("meta data of interface parameter '%s' is unknown.").toAscii().data(),nameTemplate);
                return tCmpFailed;
            }
            
        }
    }



    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateStringMeta(const ito::StringMeta *meta, const char* value, bool mandatory)
    {
        QString pattern;
        if(meta && meta->getLen() > 0 && value)
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

            for(size_t i = 0 ; i < meta->getLen() ; i++)
            {
                pattern = meta->getString(i);
                reg.setPattern( pattern );
                if(reg.indexIn( value, 0 ) > -1)
                {
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                return ito::RetVal::format(ito::retError,0, QObject::tr("String '%s' does not fit to given string-constraints.").toAscii().data(), value);
            }

        }

        if(mandatory && (value == NULL))
        {
            return ito::RetVal(ito::retError,0,QObject::tr("AddIn must not be NULL").toAscii().data());
        }

        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateDoubleMeta(const ito::DoubleMeta *meta, double value)
    {
        if(meta)
        {
            if(value < meta->getMin() || value > meta->getMax())
            {
                return ito::RetVal(ito::retError,0, QObject::tr("value out of range [%1,%2]").arg(meta->getMin()).arg(meta->getMax()).toAscii().data());
            }
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateIntMeta(const ito::IntMeta *meta, int value)
    {
        if(meta)
        {
            if(value < meta->getMin() || value > meta->getMax())
            {
                return ito::RetVal(ito::retError,0, QObject::tr("value out of range [%1,%2]").arg((int)meta->getMin()).arg((int)meta->getMax()).toAscii().data());
            }
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateCharMeta(const ito::CharMeta *meta, char value)
    {
        if(meta)
        {
            if(value < meta->getMin() || value > meta->getMax())
            {
                return ito::RetVal(ito::retError,0, QObject::tr("Value out of range [%1,%2]").arg((char)meta->getMin()).arg((char)meta->getMax()).toAscii().data());
            }
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateHWMeta(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory)
    {
        if(meta && value)
        {
            int minType = meta->getMinType();
            if( (minType & value->getBasePlugin()->getType()) != minType)
            {
                return ito::RetVal(ito::retError,0, QObject::tr("AddIn does not fit to minimum required type(s).").toAscii().data()); 
            }
            if(meta->getHWAddInName() && QString::compare(meta->getHWAddInName(), value->getBasePlugin()->objectName(), Qt::CaseInsensitive) != 0)
            {
                return ito::RetVal::format(ito::retError,0, QObject::tr("AddIn must be of the following plugin: '%s'.").toAscii().data(), meta->getHWAddInName());
            }
        }
        else if(mandatory && value == NULL)
        {
            return ito::RetVal(ito::retError,0,QObject::tr("AddIn must not be NULL").toAscii().data());
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal ParamHelper::validateParam(const ito::Param &templateParam, const ito::ParamBase &param, bool strict /*= true*/, bool mandatory /*= false*/)
    {
        ito::RetVal retVal;

        if(templateParam.getType() == param.getType())
        {

            switch(templateParam.getType())
            {
            case ito::ParamBase::Char:
                {
					retVal += validateCharMeta( dynamic_cast<const ito::CharMeta*>(templateParam.getMeta()), param.getVal<char>() ); 
                }
				break;
			case ito::ParamBase::Int:
                {
					retVal += validateIntMeta( dynamic_cast<const ito::IntMeta*>(templateParam.getMeta()), param.getVal<int>() ); 
                }
				break;
			case ito::ParamBase::Double:
                {
					retVal += validateDoubleMeta( dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta()), param.getVal<double>() ); 
                }
				break;
			case ito::ParamBase::CharArray:
				{
					const ito::CharMeta *meta = dynamic_cast<const ito::CharMeta*>(templateParam.getMeta());
					char* vals = param.getVal<char*>();
					if(meta)
					{
						for(int i = 0 ; i < param.getLen() ; i++)
						{
							retVal += validateCharMeta( meta, vals[i] );
						}
					}
				}
				break;
			case ito::ParamBase::IntArray:
				{
					const ito::IntMeta *meta = dynamic_cast<const ito::IntMeta*>(templateParam.getMeta());
					int* vals = param.getVal<int*>();
					if(meta)
					{
						for(int i = 0 ; i < param.getLen() ; i++)
						{
							retVal += validateIntMeta( meta, vals[i] );
						}
					}
				}
				break;
			case ito::ParamBase::DoubleArray:
				{
					const ito::DoubleMeta *meta = dynamic_cast<const ito::DoubleMeta*>(templateParam.getMeta());
					double* vals = param.getVal<double*>();
					if(meta)
					{
						for(int i = 0 ; i < param.getLen() ; i++)
						{
							retVal += validateDoubleMeta( meta, vals[i] );
						}
					}
				}
				break;
			case ito::ParamBase::String:
                {
					retVal += validateStringMeta( dynamic_cast<const ito::StringMeta*>(templateParam.getMeta()), param.getVal<char*>(), mandatory ); 
                }
				break;
			case ito::ParamBase::HWRef & ito::paramTypeMask:
				{
					retVal += validateHWMeta( dynamic_cast<const ito::HWMeta*>(templateParam.getMeta()), (ito::AddInBase*)param.getVal<void*>(), mandatory );
				}
                break;
            }
        }
        else if(!strict)
        {
            bool ok = false;
            ito::ParamBase p = convertParam(param, templateParam.getType(), &ok);
            if(ok)
            {
                retVal += validateParam(templateParam, p, true);
            }
            else
            {
                retVal += ito::RetVal(ito::retError,0,QObject::tr("Parameter could not be converted to destination type.").toAscii().data());
            }
        }
        else
        {
            retVal += ito::RetVal(ito::retError,0,QObject::tr("type of parameter does not fit to requested parameter type").toAscii().data());
        }

        return retVal;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::ParamBase ParamHelper::convertParam(const ito::ParamBase &source, int destType, bool *ok /*= NULL*/)
    {
        int sourceType = source.getType();
        bool ok2;
        if(ok) *ok = true;

        if(sourceType == (destType & (int)ito::paramTypeMask) ) return source;


        switch(destType & ito::paramTypeMask)
        {
        case ito::ParamBase::Int:
            {
                if(source.isNumeric())
                {
                    return ito::ParamBase(source.getName(),destType,source.getVal<int>());
                }
                else if(sourceType & ito::ParamBase::String)
                {
                    int val = QString(source.getVal<char*>()).toInt(&ok2);
                    if(ok2)
                    {
                        return ito::ParamBase(source.getName(),destType,val);
                    }
                }
            }
            break;
        
        case ito::ParamBase::Char:
            {
                if(source.isNumeric())
                {
                    return ito::ParamBase(source.getName(),destType,source.getVal<char>());
                }
                else if(sourceType & ito::ParamBase::String)
                {
                    char val = QString(source.getVal<char*>()).toShort(&ok2);
                    if(ok2)
                    {
                        return ito::ParamBase(source.getName(),destType,val);
                    }
                }
            }
            break;
        
        case ito::ParamBase::Double:
            {
                if(source.isNumeric())
                {
                    return ito::ParamBase(source.getName(),destType,source.getVal<double>());
                }
                else if(sourceType & ito::ParamBase::String)
                {
                    double val = QString(source.getVal<char*>()).toDouble(&ok2);
                    if(ok2)
                    {
                        return ito::ParamBase(source.getName(),destType,val);
                    }
                }
            }
            break;

        case ito::ParamBase::String:
            {
                if(source.isNumeric())
                {
                    return ito::ParamBase(source.getName(),destType,QString::number(source.getVal<double>()).toAscii().data() );
                }
            }
            break;
        }

        if(ok) *ok = false;
        return ParamBase();
    }

    //----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito
