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

#include "../organizer/addInManager.h"
#include "../helper/paramHelper.h"
#include "apiFunctions.h"
#include "../Qitom/AppManagement.h"
#include "../organizer/paletteOrganizer.h"

static ito::apiFunctions singleApiFunctions;

namespace ito
{
    void **ITOM_API_FUNCS;

	void *ITOM_API_FUNCS_ARR[] = {
		(void*)&singleApiFunctions.mfilterGetFunc,      /* [0] */
		(void*)&singleApiFunctions.mfilterCall,         /* [1] */
		(void*)&singleApiFunctions.mfilterParam,        /* [2] */
        (void*)&singleApiFunctions.mfilterParamBase,    /* [3] */
		(void*)&singleApiFunctions.maddInGetInitParams, /* [4] */
		(void*)&singleApiFunctions.maddInOpenActuator,  /* [5] */
		(void*)&singleApiFunctions.maddInOpenDataIO,    /* [6] */
        (void*)&ParamHelper::validateStringMeta,        /* [7] */
        (void*)&ParamHelper::validateDoubleMeta,        /* [8] */
        (void*)&ParamHelper::validateIntMeta,           /* [9] */
        (void*)&ParamHelper::validateCharMeta,          /* [10] */
        (void*)&ParamHelper::validateHWMeta,            /* [11] */
        (void*)&ParamHelper::compareParam,              /* [12] */
        (void*)&ParamHelper::validateParam,             /* [13] */
        (void*)&ParamHelper::getParamFromMapByKey,      /* [14] */
        (void*)&ParamHelper::parseParamName,            /* [15] */
		NULL
	};

//------------------------------------------------------------------------------------------------------------------------------------------------------
apiFunctions::apiFunctions() : m_loadFPointer(0)
{ 
	ITOM_API_FUNCS = ITOM_API_FUNCS_ARR;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
apiFunctions::~apiFunctions()
{}

//----------------------------------------------------------------------------------------------------------------------------------
/** Helper function to check and set initialisation parameters in the initialisation parameter list
*   @param [in]         tempParam QVariant holding the value to set
*   @param [in, out]    param   the param in the parameter list, that is set
*   @param [out]        set     indicator whether the parameter was set or not
*   @return             retOk on success, retError otherwise
*
*   The function checks if the types of the passed python parameter and the parameter are compatible and sets the parameter
*   value if it is possible. If the paramter cannot be set an error is returned.
*/
ito::RetVal apiFCheckAndSetParamVal(QVariant *tempParam, ito::ParamBase *param, int *set)
{
    switch (param->getType())
    {
        case ito::ParamBase::Char & ito::paramTypeMask:
        case ito::ParamBase::Int & ito::paramTypeMask:
            if (tempParam->type() == QVariant::Char)
            {
                *set = 1;
                param->setVal<int>(tempParam->toInt());
            }
            else
            {
                return ito::retError;
            }
        break;

        case ito::ParamBase::Double & ito::paramTypeMask:
            if ((tempParam->type() == QVariant::Double) || (tempParam->type() == QVariant::Int)
                || (tempParam->type() == QVariant::Char))
            {
                *set = 1;
                param->setVal<double>(tempParam->toDouble());
            }
            else
            {
                return ito::retError;
            }
        break;

        case ito::ParamBase::String & ito::paramTypeMask:
            if ((tempParam->type() == QVariant::String) || (tempParam->type() == QVariant::ByteArray))
            {
                *set = 1;
                QString tempStr = tempParam->toString();
                param->setVal<char *>((char*)tempStr.data_ptr(), tempStr.length());
            }
            else
            {
                return ito::retError;
            }
        break;

        case ito::ParamBase::HWRef & ito::paramTypeMask:
        case ito::ParamBase::DObjPtr & ito::paramTypeMask:
        case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
        case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
            if (tempParam->type() == QVariant::UserType)
            {
                *set = 1;
                param->setVal<void *>((void*)(tempParam->value<void*>()));
            }
            else
            {
                return ito::retError;
            }
        break;

        default:
            return ito::RetVal(ito::retError, 0, QObject::tr("Unknown parameter type").toAscii().data());
    }

    return ito::retOk;
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
ito::RetVal apiFParseInitParams(QVector<ito::ParamBase> *initParamListMand, QVector<ito::ParamBase> *initParamListOpt, QVector<QVariant> *params)
{
    int len;
    int numMandParams = initParamListMand == NULL ? 0 : initParamListMand->size();
    int numOptParams = initParamListOpt == NULL ? 0 : initParamListOpt->size();
    int *mandPParsed = (int*)calloc(numMandParams, sizeof(int));
    int *optPParsed = (int*)calloc(numOptParams, sizeof(int));
    int numParams = 0;

    numParams = params->size();

    // Check if number of given parameters is in an acceptable range
    if (((numParams) < numMandParams)
        || ((numParams) > (numMandParams + numOptParams)))
    {
        if (mandPParsed)
            free(mandPParsed);
        if (optPParsed)
            free(optPParsed);

        return ito::RetVal(ito::retError, 0, QObject::tr("Wrong number of parameters").toAscii().data());
    }

    len = numParams > numMandParams ? numMandParams : numParams;

    // read in mandatory parameters
    for (int n = 0; n < numMandParams; n++)
    {
        QVariant tempParam = params->at(n);
        if (apiFCheckAndSetParamVal(&tempParam, &((*initParamListMand)[n]), &(mandPParsed[n])) != ito::retOk)
        {
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::RetVal(ito::retError, 0, QObject::tr("Wrong parameter type").toAscii().data());
        }
    }

    // read in remaining (optional) parameters
    for (int n = numMandParams; n < numParams; n++)
    {
        QVariant tempParam = params->at(n);
        if (apiFCheckAndSetParamVal(&tempParam, &((*initParamListOpt)[n - numMandParams]), &(optPParsed[n - numMandParams])) != ito::retOk)
        {
            if (mandPParsed)
                free(mandPParsed);
            if (optPParsed)
                free(optPParsed);
            return ito::RetVal(ito::retError, 0, QObject::tr("Wrong parameter type").toAscii().data());
        }
    }

    if (mandPParsed)
        free(mandPParsed);
    if (optPParsed)
        free(optPParsed);

    return ito::retOk;
}



//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctions::setFPointer()
{
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctions::mfilterGetFunc(const QString &name, ito::AddInAlgo::FilterDef *&FilterDef)
{
    if (name.length() < 1)
    {
        FilterDef = NULL;
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toAscii().data());
    }

    ito::AddInManager *aim = ito::AddInManager::getInstance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        FilterDef = NULL;
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter not found").toAscii().data());
    }

    FilterDef = cfit.value();
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal apiFunctions::mfilterCall(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval(ito::retOk);

    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toAscii().data());
    }

    ito::AddInManager *aim = ito::AddInManager::getInstance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Filter '%s' not found").toAscii().data(), name.toAscii().data() );
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    retval += (*(fFunc->m_filterFunc))(paramsMand, paramsOpt, paramsOut);

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal apiFunctions::mfilterParamBase(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toAscii().data());
    }
    if(!paramsMand || !paramsOpt || !paramsOut)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Vectors paramsMand, paramsOpt and paramsOut must not be NULL").toAscii().data());
    }

    ito::AddInManager *aim = ito::AddInManager::getInstance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Filter '%s' not found").toAscii().data(), name.toAscii().data() );
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    const ito::FilterParams *fp = aim->getHashedFilterParams(fFunc->m_paramFunc);
    if(!fp)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter parameters not found in hash table.").toAscii().data());
    }

    paramsMand->clear();
    for(int i = 0; i < fp->paramsMand.size(); i++)
    {
        paramsMand->append( ito::ParamBase( fp->paramsMand[i] ) );
    }

    paramsOpt->clear();
    for(int i = 0; i < fp->paramsOpt.size(); i++)
    {
        paramsOpt->append( ito::ParamBase( fp->paramsOpt[i] ) );
    }

    paramsOut->clear();
    for(int i = 0; i < fp->paramsOut.size(); i++)
    {
        paramsOut->append( ito::ParamBase( fp->paramsOut[i] ) );
    }

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal apiFunctions::mfilterParam(const QString &name, QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toAscii().data());
    }
    if(!paramsMand || !paramsOpt || !paramsOut)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Vectors paramsMand, paramsOpt and paramsOut must not be NULL").toAscii().data());
    }

    ito::AddInManager *aim = ito::AddInManager::getInstance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Filter '%s' not found").toAscii().data(), name.toAscii().data() );
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    const ito::FilterParams *fp = aim->getHashedFilterParams(fFunc->m_paramFunc);
    if(!fp)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter parameters not found in hash table.").toAscii().data());
    }

    *paramsMand = fp->paramsMand;
    *paramsOpt = fp->paramsOpt;
    *paramsOut = fp->paramsOut;

    return ito::retOk;
}


//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctions::maddInGetInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt)
{
    *pluginNum = -1;

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toAscii().data());
    }

    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No plugin name specified.").toAscii().data());
    }

    return AIM->getInitParams(name, pluginType, pluginNum, paramsMand, paramsOpt);
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctions::maddInOpenActuator(const QString &name, const int pluginNum, const bool autoLoadParams, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::AddInActuator *&actuator)
{
    ito::RetVal retval(ito::retOk);

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toAscii().data());
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QMetaObject::invokeMethod(AIM, "initAddIn", Q_ARG(const int, pluginNum), Q_ARG(const QString&, name), Q_ARG(ito::AddInActuator**, &actuator), Q_ARG(QVector<ito::ParamBase>*, paramsMand), Q_ARG(QVector<ito::ParamBase>*, paramsOpt), Q_ARG(bool, autoLoadParams), Q_ARG(ItomSharedSemaphore*, waitCond));
    //retval = AIM->initAddIn(pluginNum, pluginName, &self->actuatorObj, paramsMand, paramsOpt, enableAutoLoadParams);
    waitCond->wait(-1);
    retval += waitCond->returnValue;
    waitCond->deleteSemaphore();
    waitCond = NULL;

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctions::maddInOpenDataIO(const QString &name, const int pluginNum, const bool autoLoadParams, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::AddInDataIO *&dataIO)
{
    ito::RetVal retval(ito::retOk);

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toAscii().data());
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QMetaObject::invokeMethod(AIM, "initAddIn", Q_ARG(const int, pluginNum), Q_ARG(const QString&, name), Q_ARG(ito::AddInDataIO**, &dataIO), Q_ARG(QVector<ito::ParamBase>*, paramsMand), Q_ARG(QVector<ito::ParamBase>*, paramsOpt), Q_ARG(bool, autoLoadParams), Q_ARG(ItomSharedSemaphore*, waitCond));
    //retval = AIM->initAddIn(pluginNum, pluginName, &self->actuatorObj, paramsMand, paramsOpt, enableAutoLoadParams);
    waitCond->wait(-1);
    retval += waitCond->returnValue;
    waitCond->deleteSemaphore();
    waitCond = NULL;

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
}; // namespace ito
