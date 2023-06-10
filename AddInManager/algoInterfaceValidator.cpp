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

#include "algoInterfaceValidator.h"
#include "paramHelper.h"

#include <qmap.h>
#include <qregularexpression.h>

namespace ito
{
    class AlgoInterfaceValidatorPrivate
    {
    public:
        AlgoInterfaceValidatorPrivate() {};
        ~AlgoInterfaceValidatorPrivate() {};

        struct AlgoInterface
        {
            AlgoInterface() : maxNumMand(0), maxNumOpt(0), maxNumOut(0) {}
            QVector<ito::Param> mandParams;
            QVector<ito::Param> outParams;
            int maxNumMand;
            int maxNumOpt;
            int maxNumOut;
        };

        QMap<int, AlgoInterface> m_interfaces;
    };

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \class AlgoInterfaceValidator
        \brief The class AlgoInterfaceValidator provides validators and checks in order to verify that
            a certain filter or algoWidget that pretends to implement a certain interface really fits
            to the requirements and conditions of this interface.

        In the current implementation this class is instantiated once by the class AddInManager, hence,
        it can be considered as singleton class.

        \sa AddInManager, AddInAlgo::tAlgoInterface, AddInAlgo::FilterDef, AddInAlgo::AlgoWidgetDef
    */

    //----------------------------------------------------------------------------------------------------------------------------------
    //! constructor
    /*!
        Calls method init in order to load the requirements and conditions for each interface, defined
        in the enumeration tAlgoInterface
        \sa AddInAlgo::tAlgoInterface
    */
    AlgoInterfaceValidator::AlgoInterfaceValidator(ito::RetVal &retValue) : QObject(), d_ptr(new AlgoInterfaceValidatorPrivate())
    {
        retValue += init();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! destructor
    AlgoInterfaceValidator::~AlgoInterfaceValidator()
    {
        Q_D(AlgoInterfaceValidator);
        d->m_interfaces.clear();
    }


    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AlgoInterfaceValidator::getInterfaceParameters(ito::AddInAlgo::tAlgoInterface iface, QVector<ito::ParamBase> &mandParams, QVector<ito::ParamBase> &outParams) const
    {
        Q_D(const AlgoInterfaceValidator);
        QMap<int, AlgoInterfaceValidatorPrivate::AlgoInterface>::const_iterator it = d->m_interfaces.constFind((int)iface);
        if (it != d->m_interfaces.constEnd())
        {
            mandParams.clear();
            outParams.clear();
            const QVector<ito::Param> &mands = it->mandParams;
            const QVector<ito::Param> &outs = it->outParams;

            for (int i = 0; i < mands.size(); i++)
            {
                mandParams << ito::ParamBase( mands[i] );
            }

            for (int i = 0; i < outs.size(); i++)
            {
                outParams << ito::ParamBase( outs[i] );
            }

            return ito::retOk;
        }

        return ito::RetVal(ito::retError,0,tr("interface not found").toLatin1().data());
    }

    //----------------------------------------------------------------------------------------------------------------------------------
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
        int maxNum = std::numeric_limits<int>::max();

        //1. ito::AddInAlgo::iReadDataObject
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("dataObject", ito::ParamBase::DObjPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("DataObject [in/out]").toLatin1().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iReadDataObject, pMand, pOut, maxNum, maxNum, 0);

        //2. ito::AddInAlgo::iReadPointCloud
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("pointCloud", ito::ParamBase::PointCloudPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("PointCloud [in/out]").toLatin1().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iReadPointCloud, pMand, pOut, maxNum, maxNum, 0);

        //3. ito::AddInAlgo::iReadPolygonMesh
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("PolygonMesh [in/out]").toLatin1().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iReadPolygonMesh, pMand, pOut, maxNum, maxNum, 0);

        //4. ito::AddInAlgo::iWriteDataObject
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("dataObject", ito::ParamBase::DObjPtr | ito::ParamBase::In, NULL, tr("DataObject [in]").toLatin1().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iWriteDataObject, pMand, pOut, maxNum, maxNum, 0);

        //5. ito::AddInAlgo::iWritePointCloud
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("pointCloud", ito::ParamBase::PointCloudPtr | ito::ParamBase::In, NULL, tr("PointCloud [in]").toLatin1().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iWritePointCloud, pMand, pOut, maxNum, maxNum, 0);

        //6. ito::AddInAlgo::iWritePolygonMesh
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("polygonMesh", ito::ParamBase::PolygonMeshPtr | ito::ParamBase::In, NULL, tr("PolygonMesh [in]").toLatin1().data());
        pMand << ito::Param("filename", ito::ParamBase::String | ito::ParamBase::In, "", tr("filename").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iWritePolygonMesh, pMand, pOut, maxNum, maxNum, 0);

        //7. ito::AddInAlgo::iPlotSingleObject
        pMand.clear();
        pOut.clear();
        pMand << ito::Param("dataObject", ito::ParamBase::DObjPtr | ito::ParamBase::In, NULL, tr("DataObject [in]").toLatin1().data());
        retVal += addInterface(ito::AddInAlgo::iPlotSingleObject, pMand, pOut, maxNum, maxNum, 0);

        return retVal;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
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
                QRegularExpression regExp("^[a-zA-Z0-9-_ ]+\\(((\\*\\.[a-zA-Z0-9]{2,4} )*\\*\\.[a-zA-Z0-9]{2,4})\\)$");
                QStringList l2;
                QRegularExpressionMatch match;

                foreach(const QString &s, l)
                {
                    match = regExp.match(s);
                    if (match.hasMatch())
                    {
                        l2 = match.capturedTexts()[1].split(" ");
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

    //----------------------------------------------------------------------------------------------------------------------------------
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
    ito::RetVal AlgoInterfaceValidator::addInterface(ito::AddInAlgo::tAlgoInterface iface, QVector<ito::Param> &mandParams, QVector<ito::Param> &outParams, int maxNumMand, int maxNumOpt, int maxNumOut)
    {
        Q_D(AlgoInterfaceValidator);
        QMap<int, AlgoInterfaceValidatorPrivate::AlgoInterface>::const_iterator it = d->m_interfaces.constFind((int)iface);
        if(it == d->m_interfaces.constEnd())
        {
            AlgoInterfaceValidatorPrivate::AlgoInterface ai;
            ai.mandParams = mandParams;
            ai.outParams = outParams;
            ai.maxNumMand = maxNumMand;
            ai.maxNumOpt = maxNumOpt;
            ai.maxNumOut = maxNumOut;
            d->m_interfaces[ (int)iface ] = ai;
        }
        else
        {
            return ito::RetVal(ito::retError, 0, tr("interface could not be added since its enumeration ID already has been added").toLatin1().data());
        }
        return ito::retOk;
    }


    //----------------------------------------------------------------------------------------------------------------------------------
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
        bool valid = isValid(filter.m_interface, filter.m_paramFunc, ret);
        if (!valid && !ret.containsError())
        {
            ret += ito::RetVal(ito::retError, 0, tr("The parameters of the filter does not fit to the requirements given by the specified filter interface.").toLatin1().data());
        }

        bool valid2 = getTags(filter.m_interface, filter.m_interfaceMeta, tags);
        if (!valid2 && !ret.containsError())
        {
            ret += ito::RetVal(ito::retError, 0, tr("The filter does not have the required tags defined.").toLatin1().data());
        }

        return valid && valid2;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
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
        bool valid = isValid(widget.m_interface, widget.m_paramFunc, ret);
        if (!valid && !ret.containsError())
        {
            ret += ito::RetVal(ito::retError, 0, tr("The parameters of the widget does not fit to the requirements given by the specified widget interface.").toLatin1().data());
        }

        bool valid2 = getTags(widget.m_interface, widget.m_interfaceMeta, tags);
        if (!valid2 && !ret.containsError())
        {
            ret += ito::RetVal(ito::retError, 0, tr("The widget does not have the required tags defined.").toLatin1().data());
        }

        return valid && valid2;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \param iface
        \param filterParamFunc
        \param ret
        \return bool

        \sa isValid, getTags
    */
    bool AlgoInterfaceValidator::isValid(const ito::AddInAlgo::tAlgoInterface iface, const ito::AddInAlgo::t_filterParam filterParamFunc, ito::RetVal &ret) const
    {
        if(iface == ito::AddInAlgo::iNotSpecified) return true;

        Q_D(const AlgoInterfaceValidator);

        int iface2 = (int)iface;
        QMap<int, AlgoInterfaceValidatorPrivate::AlgoInterface>::const_iterator it = d->m_interfaces.constFind(iface2);
        if(it != d->m_interfaces.constEnd())
        {
            QVector<ito::Param> paramsMand;
            QVector<ito::Param> paramsOpt;
            QVector<ito::Param> paramsOut;

            ret += filterParamFunc(&paramsMand, &paramsOpt, &paramsOut);
            if(ret.containsError())
            {
                return false;
            }

            if((int)paramsMand.size() > it->maxNumMand)
            {
                ret += ito::RetVal(ito::retError, 0, tr("Number of mandatory parameters of given algorithm exceed the maximum value, given by algorithm interface.").toLatin1().data());
                return false;
            }

            if((int)paramsOpt.size() > it->maxNumOpt)
            {
                ret += ito::RetVal(ito::retError, 0, tr("Number of optional parameters of given algorithm exceed the maximum value, given by algorithm interface.").toLatin1().data());
                return false;
            }

            if((int)paramsOut.size() > it->maxNumOut)
            {
                ret += ito::RetVal(ito::retError, 0, tr("Number of output parameters of given algorithm exceed the maximum value, given by algorithm interface.").toLatin1().data());
                return false;
            }

            ito::tCompareResult result;

            // checking if the number of parameters is equal, otherwise this might crash
            if (paramsMand.size() < it->mandParams.size())
                return false;

            for (int i=0; i < it->mandParams.size(); i++)
            {
                result = ParamHelper::compareParam(it->mandParams[i], (paramsMand)[i], ret);
                if(result == tCmpFailed)
                {
                    return false;
                }
            }

            // checking if the number of parameters is equal, otherwise this might crash
            if (paramsOut.size() < it->outParams.size())
                return false;

            for(int i=0; i < it->outParams.size(); i++)
            {
                result = ParamHelper::compareParam( it->outParams[i], (paramsOut)[i], ret);
                if(result == tCmpFailed)
                {
                    return false;
                }
            }

        }
        else
        {
            ret += ito::RetVal(ito::retError, 0, tr("The given algorithm interface is unknown").toLatin1().data());
            return false;
        }
        return true;
    }

} //end namespace ito
