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

#include "algoInterfaceValidator.h"
//#include "addInManager.h"
#include "../global.h"
#include <qregexp.h>

namespace ito
{
	/*!
		\class AlgoInterfaceValidator
		\brief The class AlgoInterfaceValidator provides validators and checks in order to verify that
			a certain filter or algoWidget that pretends to implement a certain interface really fits
			to the requirements and conditions of this interface.

		In the current implementation this class is instatiated once by the class AddInManager, hence,
		it can be considered as singleton class.

		\sa AddInManager, AddInAlgo::tAlgoInterface, AddInAlgo::FilterDef, AddInAlgo::AlgoWidgetDef
	*/

	//! constructor
	/*!
		Calls method init in order to load the requirements and conditions for each interface, defined
		in the enumeration tAlgoInterface
		\sa AddInAlgo::tAlgoInterface
	*/
    AlgoInterfaceValidator::AlgoInterfaceValidator(ito::RetVal &retValue) : QObject()
    {
        retValue += init();
    }

	//! destructor
    AlgoInterfaceValidator::~AlgoInterfaceValidator()
    {
        m_interfaces.clear();
    }


    ito::RetVal AlgoInterfaceValidator::getInterfaceParameters(ito::AddInAlgo::tAlgoInterface iface, QVector<ito::ParamBase> &mandParams, QVector<ito::ParamBase> &outParams) const
    {
        QMap<int,AlgoInterface>::const_iterator it = m_interfaces.constFind( (int)iface );
        if(it != m_interfaces.constEnd())
        {
            mandParams.clear();
            outParams.clear();
            const QVector<ito::Param> &mands = it->mandParams;
            const QVector<ito::Param> &outs = it->outParams;
            for(int i = 0 ; i < mands.size(); i++)
            {
                mandParams << ito::ParamBase( mands[i] );
            }
            for(int i = 0 ; i < outs.size(); i++)
            {
                outParams << ito::ParamBase( outs[i] );
            }
            
            return ito::retOk;
        }
        return ito::RetVal(ito::retError,0,tr("interface not found").toAscii().data());
    }

	//! loads the requirements for every interface defined in the enumeration AddInAlgo::tAlgoInterface
	/*!
		The requirements of every interface can be given by several things:
		- You can indicate the first n mandatory parameters. This is done by filling a vector of class ito::Param.
		- You can define the first m output parameters.
		- You can indicate a maximum number of mandatory, optional and output parameters.
		must be greater or equal than the previously indicated vectors.

		If any mandatory, optional or output parameters are defined, the corresponding filter, that pretends
		to fit to this interface, must have a parameter at the same position whose type is equal to the given parameter
		and if the indicated parameter of the interface has any restricitions given in form of a meta information instance,
		the corresponding parameter of the filter must have a restriction which is equal or "stricter" than the restriction
		defined in this method.

		For every interface defined in the enumeration AddInAlgo::tAlgoInterface, you should call the method addInInterface
		in this method in order to create the requirements. The last requirement of every interface is the possible structure
		or content of the meta information string, that can or sometimes must be appended when creating the FilterDef-instance
		in any plugin. This meta information string is checked and parsed in the method getTags.

		\sa AddInAlgo::tAlgoInterface, ito::Param, addInterface, getTags
	*/
    ito::RetVal AlgoInterfaceValidator::init(void)
    {
        ito::RetVal retVal;
        QVector<ito::Param> pMand;
        QVector<ito::Param> pOut;
        size_t maxNum = std::numeric_limits<size_t>::max();

        //1. ito::AddInAlgo::iReadDataObject
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("dataObject", ito::ParamBase::DObjPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("DataObject [in/out]").toAscii().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iReadDataObject, pMand, pOut, maxNum, maxNum, 0);

        //2. ito::AddInAlgo::iReadPointCloud
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("pointCloud", ito::ParamBase::PointCloudPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("PointCloud [in/out]").toAscii().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iReadPointCloud, pMand, pOut, maxNum, maxNum, 0);

        //3. ito::AddInAlgo::iReadPolygonMesh
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("PolygonMesh [in/out]").toAscii().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iReadPolygonMesh, pMand, pOut, maxNum, maxNum, 0);

        //4. ito::AddInAlgo::iWriteDataObject
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("dataObject", ito::ParamBase::DObjPtr | ito::ParamBase::In, NULL, tr("DataObject [in]").toAscii().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iWriteDataObject, pMand, pOut, maxNum, maxNum, 0);

        //5. ito::AddInAlgo::iWritePointCloud
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("pointCloud", ito::ParamBase::PointCloudPtr | ito::ParamBase::In, NULL, tr("PointCloud [in]").toAscii().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iWritePointCloud, pMand, pOut, maxNum, maxNum, 0);

        //6. ito::AddInAlgo::iWritePolygonMesh
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr | ito::ParamBase::In, NULL, tr("PolygonMesh [in]").toAscii().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iWritePolygonMesh, pMand, pOut, maxNum, maxNum, 0);

        //7. ito::AddInAlgo::iPlotSingleObject
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("dataObject", ito::ParamBase::DObjPtr | ito::ParamBase::In, NULL, tr("DataObject [in]").toAscii().data());
        retVal += addInterface(ito::AddInAlgo::iPlotSingleObject, pMand, pOut, maxNum, maxNum, 0);

        return retVal;
    }

	//! verifies and parses the meta information string of any filter or algoWidget
	/*!
		Sometimes a certain algorithm interface needs that the user gives additional information about the 
		filter when creating this filter in the plugin. This additional information can be given by the meta information
		string in the structs FilterDef and AlgoWidgetDef. In the filter or algoWidget pretends to fit to a certain
		algorithm interface, this method is called with the interface number and the meta information string. Then, this
		string is checked if it fits the requirements of the interface and if so it can be parsed. The parsed elements
		are then returned in form of a string list. The meaning and definition of the content of this string list must
		be interpreted by the method which wishes to use filters of a certain interface type.

		\param [in] iface is the interface number
		\param [in] metaInformation is the meta information string connected to the filter in any plugin
		\param [out] tags is the parsed string list (or empty in case of an error)
		\return true if the meta information string has been valid and could be parsed, else false

		\sa AddInAlgo::tAlgoInterface
	*/
    bool AlgoInterfaceValidator::getTags(const ito::AddInAlgo::tAlgoInterface iface, const QString &metaInformation, QStringList &tags) const
    {
        tags.clear();
        switch(iface)
        {
        case ito::AddInAlgo::iNotSpecified:
        case ito::AddInAlgo::iPlotSingleObject:
            return true;
        case ito::AddInAlgo::iReadDataObject:
        case ito::AddInAlgo::iReadPointCloud:
        case ito::AddInAlgo::iReadPolygonMesh:
        case ito::AddInAlgo::iWriteDataObject:
        case ito::AddInAlgo::iWritePointCloud:
        case ito::AddInAlgo::iWritePolygonMesh:
            {
                QStringList l = metaInformation.split(";;");
                //the regExp checks for strings like "Text Files (*.txt *.dat *.bat)"
                QRegExp regExp("^[a-zA-Z0-9-_ ]+\\(((\\*\\.[a-zA-Z0-9]{2,4} )*\\*\\.[a-zA-Z0-9]{2,4})\\)$");
                QStringList l2;
                foreach(const QString &s, l)
                {
                    if(regExp.indexIn(s) >= 0)
                    {
                        l2 = regExp.capturedTexts()[1].split(" ");
                        foreach(const QString &s2,l2)
                        {
                            if(s2.size() > 2)
                            {
                                tags.append( s2.mid(2) );
                            }
                        }
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
        default:
            return false;
        }
    }

	//! addInterface
	/*!
		Registers the requirements for any interface given by the enumeration value AddInAlgo::tAlgoInterface.

		\param [in] iface is the interface enumeration value whose requirements should be registered
		\param [in] mandParams is a vector indicating the first m mandatory parameters every filter that implements this interface must provide, too.
		\param [in] outParams is a vector indicating the fist n output parameters every filter that implements this interface must provide, too. 
					Remember that every parameter in this vector must have the Out-flag set and no In-flag.
		\param [in] maxNumMand is the maximum number of mandatory parameters
		\param [in] maxNumOpt is the maximum number of optional parameters
		\param [in] maxNumOut is the maximum number of output parameters

		\sa AddInAlgo::tAlgoInterface
	*/
    ito::RetVal AlgoInterfaceValidator::addInterface(ito::AddInAlgo::tAlgoInterface iface, QVector<ito::Param> &mandParams, QVector<ito::Param> &outParams, size_t maxNumMand, size_t maxNumOpt, size_t maxNumOut)
    {
        QMap<int,AlgoInterface>::const_iterator it = m_interfaces.constFind( (int)iface );
        if(it == m_interfaces.constEnd())
        {
            AlgoInterface ai;
            ai.mandParams = mandParams;
            ai.outParams = outParams;
            ai.maxNumMand = maxNumMand;
            ai.maxNumOpt = maxNumOpt;
            ai.maxNumOut = maxNumOut;
            m_interfaces[ (int)iface ] = ai;
        }
        else
        {
            return ito::RetVal(ito::retError,0,"interface could not be added since its enumeration ID already has been added");
        }
        return ito::retOk;
    }


	//! verifies a given filter with respect to its indicated interface
	/*!
		If the given filter pretends to implement a certain interface, the parameters of the filter are
		checked with respect to the requirements of the interface and the additional meta information string is checked
		and parsed.

		\param [in] filter is the filter-definition struct of the filter to check
		\param [in/out] ret is the result of the check
		\param [out] tags are an optional string list containing tags extracted from the meta information string
		\return true if filter is valid with respect to the interface

		\sa isValid, getTags
	*/
    bool AlgoInterfaceValidator::isValidFilter(const ito::AddInAlgo::FilterDef &filter, ito::RetVal &ret, QStringList &tags) const
    {
        return isValid(filter.m_interface, filter.m_paramFunc, ret) & getTags(filter.m_interface, filter.m_interfaceMeta, tags);
    }

	//! verifies a given algo-widget with respect to its indicated interface
	/*!
		If the given algo-widget pretends to implement a certain interface, the parameters of the algo-widget are
		checked with respect to the requirements of the interface and the additional meta information string is checked
		and parsed.

		\param [in] widget is the algo-widget-definition struct of the algo-widget to check
		\param [in/out] ret is the result of the check
		\param [out] tags are an optional string list containing tags extracted from the meta information string
		\return true if the algo-widget is valid with respect to the interface

		\sa isValid, getTags
	*/
    bool AlgoInterfaceValidator::isValidWidget(const ito::AddInAlgo::AlgoWidgetDef &widget, ito::RetVal &ret, QStringList &tags) const
    {
        return isValid(widget.m_interface, widget.m_paramFunc, ret) & getTags(widget.m_interface, widget.m_interfaceMeta, tags);
    }

    bool AlgoInterfaceValidator::isValid(const ito::AddInAlgo::tAlgoInterface iface, const ito::AddInAlgo::t_filterParam filterParamFunc, ito::RetVal &ret) const
    {
        if(iface == ito::AddInAlgo::iNotSpecified) return true;

        int iface2 = (int)iface;
        QMap<int,AlgoInterface>::const_iterator it = m_interfaces.constFind(iface2);
        if(it != m_interfaces.constEnd())
        {
            QVector<ito::Param> paramsMand;
            QVector<ito::Param> paramsOpt;
            QVector<ito::Param> paramsOut;

            ret += filterParamFunc(&paramsMand, &paramsOpt, &paramsOut);
            if(ret.containsError())
            {
                return false;
            }

            if((size_t)paramsMand.size() > it->maxNumMand)
            {
                ret += ito::RetVal(ito::retError,0,tr("Number of mandatory parameters of given algorithm exceed the maximum value, given by algorithm interface.").toAscii().data());
                return false;
            }

            if((size_t)paramsOpt.size() > it->maxNumOpt)
            {
                ret += ito::RetVal(ito::retError,0,tr("Number of optional parameters of given algorithm exceed the maximum value, given by algorithm interface.").toAscii().data());
                return false;
            }

            if((size_t)paramsOut.size() > it->maxNumOut)
            {
                ret += ito::RetVal(ito::retError,0,tr("Number of output parameters of given algorithm exceed the maximum value, given by algorithm interface.").toAscii().data());
                return false;
            }

            AlgoInterfaceValidator::tCompareResult result;
            
            for(int i=0; i < it->mandParams.size(); i++)
            {
                result = compareParam( it->mandParams[i], (paramsMand)[i], ret);
                if(result == tCmpFailed)
                {
                    return false;
                }
            }

            for(int i=0; i < it->outParams.size(); i++)
            {
                result = compareParam( it->outParams[i], (paramsOut)[i], ret);
                if(result == tCmpFailed)
                {
                    return false;
                }
            }

        }
        else
        {
            ret += ito::RetVal(ito::retError,0,tr("The given algorithm interface is unknown").toAscii().data());
            return false;
        }
        return true;
    }

    AlgoInterfaceValidator::tCompareResult AlgoInterfaceValidator::compareParam(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret) const
    {
        //check whether type is equal
        if(paramTemplate.getType() != param.getType())
        {
            ret += ito::RetVal::format(ito::retError,0,tr("Types of parameter '%s' is unequal to required type of interface parameter '%s'").toAscii().data(),param.getName(),paramTemplate.getName());
            return tCmpFailed;
        }

        //check in/out flags
        int inOutFlags = ito::ParamBase::In | ito::ParamBase::Out;
        if( (paramTemplate.getFlags() & inOutFlags) != (param.getFlags() & inOutFlags) )
        {
            ret += ito::RetVal::format(ito::retError,0,tr("In/Out flags of parameter '%s' are unequal to required flags of interface parameter '%s'").toAscii().data(),param.getName(),paramTemplate.getName());
            return tCmpFailed;
        }

        //check meta information
        const ito::ParamMeta *metaTemplate = paramTemplate.getMeta();
        const ito::ParamMeta *meta = param.getMeta();

        return compareMetaParam(metaTemplate, meta, paramTemplate.getName(), param.getName(), ret);
    }

    AlgoInterfaceValidator::tCompareResult AlgoInterfaceValidator::compareMetaParam(const ito::ParamMeta *metaTemplate, const ito::ParamMeta *meta, const char* nameTemplate, const char *name, ito::RetVal &ret) const
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
            ret += ito::RetVal::format(ito::retError,0,tr("The parameter '%s' is restricted by meta information while the interface parameter '%s' is not.").toAscii().data(),name,nameTemplate);
            return tCmpFailed;
        }

        if(metaTemplate->getType() != meta->getType())
        {
            ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The allowed integer range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The allowed char range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The allowed double range of parameter '%s' is smaller than the requested range from interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                if(m->getStringType() != mT->getStringType())
                {
                    ret += ito::RetVal::format(ito::retError,0,tr("The string type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                        ret += ito::RetVal::format(ito::retError,0,tr("String '%s', requested by meta data of interface parameter '%s' could not be found in meta data of parameter '%s'.").toAscii().data(),sT,nameTemplate,name);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                //all bits in allowedTypes of mT must be set in m, too
                if( (m->getAllowedTypes() & mT->getAllowedTypes()) != mT->getAllowedTypes() )
                {
                    ret += ito::RetVal::format(ito::retError,0,tr("The allowed data object types of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                        ret += ito::RetVal::format(ito::retError,0,tr("The minimum and maximum dimensions of the data object of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                        ret += ito::RetVal::format(ito::retError,0,tr("The minimum and maximum dimensions of the data object of parameter '%s' are more restrictive than these required by the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
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
                    ret += ito::RetVal::format(ito::retError,0,tr("The type of the meta information of parameter '%s' is unequal to this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                    return tCmpFailed;
                }

                if(mT->getHWAddInName() != NULL)
                {
                    if(m->getHWAddInName() != NULL)
                    {
                        ret += ito::RetVal::format(ito::retError,0,tr("The meta data of the interface parameter '%s' requires a plugin with name '%s', but parameter '%s' does it not.").toAscii().data(),nameTemplate,mT->getHWAddInName(),name);
                        return tCmpFailed;
                    }
                    else if( strcmp(mT->getHWAddInName(),m->getHWAddInName()) != 0)
                    {
                        ret += ito::RetVal::format(ito::retError,0,tr("Both parameter '%s' and interface parameter '%s' require different plugins.").toAscii().data(),name,nameTemplate);
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

                ret += ito::RetVal::format(ito::retError,0,tr("The minimum plugin type bit mask of parameter '%s' is more restrictive than this of the interface parameter '%s'.").toAscii().data(),name,nameTemplate);
                return tCmpFailed;
                
            }
            break;

            default:
            {
                ret += ito::RetVal::format(ito::retError,0,tr("meta data of interface parameter '%s' is unknown.").toAscii().data(),nameTemplate);
                return tCmpFailed;
            }
            
        }
    }

} //end namespace ito