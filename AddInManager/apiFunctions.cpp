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

#include "addInManager.h"
#include "paramHelper.h"
#include "apiFunctions.h"
#include "../common/sharedFunctionsQt.h"
#include "../common/abstractAddInConfigDialog.h"
#include <qdir.h>

static ito::ApiFunctions singleApiFunctions; //singleton instance, forces the construction where the ITOM_API_FUNCS_ARR pointer is propagated to ITOM_API_FUNCS
QString ito::ApiFunctions::m_settingsFile("");

namespace ito
{
    void **ITOM_API_FUNCS;

    void *ITOM_API_FUNCS_ARR[] = {
        (void*)&ApiFunctions::mfilterGetFunc,             /* [0] */
        (void*)&ApiFunctions::mfilterCall,                /* [1] */
        (void*)&ApiFunctions::mfilterParam,               /* [2] */
        (void*)&ApiFunctions::mfilterParamBase,           /* [3] */
        (void*)&ApiFunctions::maddInGetInitParams,        /* [4] */
        (void*)&ApiFunctions::maddInOpenActuator,         /* [5] */
        (void*)&ApiFunctions::maddInOpenDataIO,           /* [6] */
        (void*)&ParamHelper::validateStringMeta,          /* [7] */
        (void*)&ParamHelper::validateDoubleMeta,          /* [8] */
        (void*)&ParamHelper::validateIntMeta,             /* [9] */
        (void*)&ParamHelper::validateCharMeta,            /* [10] */
        (void*)&ParamHelper::validateHWMeta,              /* [11] */
        (void*)&ParamHelper::compareParam,                /* [12] */
        (void*)&ParamHelper::validateParam,               /* [13] */
        (void*)&ParamHelper::getParamFromMapByKey,        /* [14] */
        (void*)&ParamHelper::parseParamName,              /* [15] */
        (void*)&ParamHelper::getItemFromArray,            /* [16] */
        (void*)&saveQLIST2XML,                            /* [17] */
        (void*)&loadXML2QLIST,                            /* [18] */
        (void*)&ApiFunctions::mcreateFromDataObject,      /* [19] */
        (void*)&ParamHelper::getParam,                    /* [20] */
        (void*)&ApiFunctions::getCurrentWorkingDir,       /* [21] */
        (void*)&ApiFunctions::mshowConfigurationDialog,   /* [22] */
        (void*)&ParamHelper::updateParameters,            /* [23] */
        (void*)&ApiFunctions::mcreateFromNamedDataObject, /* [24] */
        (void*)&ParamHelper::validateAndCastParam,        /* [25] */
        (void*)&ParamHelper::validateIntArrayMeta,        /* [26] */
        (void*)&ParamHelper::validateCharArrayMeta,       /* [27] */
        (void*)&ParamHelper::validateDoubleArrayMeta,     /* [28] */
//        (void*)&ApiFunctions::sendParamToPyWorkspaceThreadSafe,      /* [29] */
//        (void*)&ApiFunctions::sendParamsToPyWorkspaceThreadSafe,     /* [30] */
        (void*)&ApiFunctions::removed,                    /* [29] */
        (void*)&ApiFunctions::removed,                    /* [30] */
        (void*)&ApiFunctions::maddInClose,                /* [31] */
//        (void*)&QPropertyHelper::readProperty,            /* [32] */
//        (void*)&QPropertyHelper::writeProperty,           /* [33] */
        (void*)&ApiFunctions::removed,                    /* [32] */
        (void*)&ApiFunctions::removed,                    /* [33] */
        (void*)&ApiFunctions::getSettingsFile,            /* [34] */
        (void*)&ApiFunctions::mfilterVersion,             /* [35] */
        (void*)&ApiFunctions::mfilterAuthor,              /* [36] */
        (void*)&ApiFunctions::mfilterPluginName,          /* [37] */
        (void*)&ApiFunctions::mfilterCallExt,             /* [38] */
        NULL
    };

//------------------------------------------------------------------------------------------------------------------------------------------------------
ApiFunctions::ApiFunctions() : m_loadFPointer(0)
{
    ITOM_API_FUNCS = ITOM_API_FUNCS_ARR;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ApiFunctions::~ApiFunctions()
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
        case ito::ParamBase::Char:
        case ito::ParamBase::Int:
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

        case ito::ParamBase::Double:
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

        case ito::ParamBase::String:
            if ((tempParam->type() == QVariant::String) || (tempParam->type() == QVariant::ByteArray))
            {
                *set = 1;
                QString tempStr = tempParam->toString();
                param->setVal<char*>((char*)tempStr.toLatin1().data(), tempStr.length());
            }
            else
            {
                return ito::retError;
            }
        break;

        case ito::ParamBase::HWRef:
        case ito::ParamBase::DObjPtr:
        case ito::ParamBase::PointCloudPtr:
        case ito::ParamBase::PolygonMeshPtr:
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
            return ito::RetVal(ito::retError, 0, QObject::tr("Unknown parameter type").toLatin1().data());
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
//    int len;
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

        return ito::RetVal(ito::retError, 0, QObject::tr("Wrong number of parameters").toLatin1().data());
    }

    numParams > numMandParams ? numMandParams : numParams;

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
            return ito::RetVal(ito::retError, 0, QObject::tr("Wrong parameter type").toLatin1().data());
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
            return ito::RetVal(ito::retError, 0, QObject::tr("Wrong parameter type").toLatin1().data());
        }
    }

    if (mandPParsed)
        free(mandPParsed);
    if (optPParsed)
        free(optPParsed);

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::removed(...)
{
    return ito::RetVal(ito::retError, 0, QObject::tr("function removed from apiFunctions, check apiFunctionsGraph").toLatin1().data());
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::mfilterGetFunc(const QString &name, ito::AddInAlgo::FilterDef *&FilterDef)
{
    if (name.length() < 1)
    {
        FilterDef = NULL;
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        FilterDef = NULL;
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter not found").toLatin1().data());
    }

    FilterDef = cfit.value();
    return ito::retOk;
}

//-------------------------------------------------------------------------------------
/*static*/ ito::RetVal ApiFunctions::mfilterCall(
    const QString &name,
    QVector<ito::ParamBase> *paramsMand,
    QVector<ito::ParamBase> *paramsOpt,
    QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval(ito::retOk);

    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);

    if (cfit == flist->end())
    {
        return ito::RetVal::format(
            ito::retError, 0, QObject::tr("Filter '%s' not found").toLatin1().data(), name.toLatin1().data() );
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    retval += (*(fFunc->m_filterFunc))(paramsMand, paramsOpt, paramsOut);

    return retval;
}

//-------------------------------------------------------------------------------------
/*static*/ ito::RetVal ApiFunctions::mfilterCallExt(
    const QString &name,
    QVector<ito::ParamBase> *paramsMand,
    QVector<ito::ParamBase> *paramsOpt,
    QVector<ito::ParamBase> *paramsOut,
    QSharedPointer<ito::FunctionCancellationAndObserver> observer)
{
    ito::RetVal retval(ito::retOk);

    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);

    if (cfit == flist->end())
    {
        return ito::RetVal::format(
            ito::retError, 0, QObject::tr("Filter '%s' not found").toLatin1().data(), name.toLatin1().data());
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();

    //try to cast cfit to FilterDefExt
    ito::AddInAlgo::FilterDefExt *fFuncExt = dynamic_cast<ito::AddInAlgo::FilterDefExt*>(fFunc);

    if (fFuncExt == NULL)
    {
        return ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("Filter '%s' has no progress observer and cancellation interface").toLatin1().data(),
            name.toLatin1().data());
    }

    retval += (*(fFuncExt->m_filterFuncExt))(paramsMand, paramsOpt, paramsOut, observer);

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal ApiFunctions::mfilterParamBase(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }
    if(!paramsMand || !paramsOpt || !paramsOut)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Vectors paramsMand, paramsOpt and paramsOut must not be NULL").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Filter '%s' not found").toLatin1().data(), name.toLatin1().data() );
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    const ito::FilterParams *fp = aim->getHashedFilterParams(fFunc->m_paramFunc);
    if(!fp)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter parameters not found in hash table.").toLatin1().data());
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
/*static*/ ito::RetVal ApiFunctions::mfilterParam(const QString &name, QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }
    if(!paramsMand || !paramsOpt || !paramsOut)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Vectors paramsMand, paramsOpt and paramsOut must not be NULL").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Filter '%s' not found").toLatin1().data(), name.toLatin1().data() );
    }

    ito::AddInAlgo::FilterDef * fFunc = cfit.value();
    const ito::FilterParams *fp = aim->getHashedFilterParams(fFunc->m_paramFunc);
    if(!fp)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter parameters not found in hash table.").toLatin1().data());
    }

    *paramsMand = fp->paramsMand;
    *paramsOpt = fp->paramsOpt;
    *paramsOut = fp->paramsOut;

    return ito::retOk;
}


//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal ApiFunctions::mfilterVersion(const QString &name, int &version)
{
    if (name.length() < 1)
    {
        version = 0;
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        version = 0;
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter not found").toLatin1().data());
    }

    version = cfit.value()->m_pBasePlugin->getVersion();

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal ApiFunctions::mfilterAuthor(const QString &name, QString &author)
{
    if (name.length() < 1)
    {
        author = "";
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        author = "";
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter not found").toLatin1().data());
    }

    author = cfit.value()->m_pBasePlugin->getAuthor();

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal ApiFunctions::mfilterPluginName(const QString &name, QString &pluginName)
{
    if (name.length() < 1)
    {
        pluginName = "";
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter name empty").toLatin1().data());
    }

    ito::AddInManager *aim = AddInManager::instance();
    const QHash<QString, ito::AddInAlgo::FilterDef *> *flist = aim->getFilterList();
    QHash<QString, ito::AddInAlgo::FilterDef *>::ConstIterator cfit = flist->find(name);
    if (cfit == flist->end())
    {
        pluginName = "";
        return ito::RetVal(ito::retError, 0, QObject::tr("Filter not found").toLatin1().data());
    }

    pluginName = cfit.value()->m_pBasePlugin->objectName();

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::maddInGetInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt)
{
    *pluginNum = -1;

    ito::AddInManager *AIM = AddInManager::instance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toLatin1().data());
    }

    if (name.length() < 1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No plugin name specified.").toLatin1().data());
    }

    return AIM->getInitParams(name, pluginType, pluginNum, paramsMand, paramsOpt);
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::maddInOpenActuator(const QString &name, const int pluginNum, const bool autoLoadParams, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::AddInActuator *&actuator)
{
    ito::RetVal retval(ito::retOk);

    ito::AddInManager *AIM = AddInManager::instance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toLatin1().data());
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QMetaObject::invokeMethod(AIM, "initAddIn", Q_ARG(int, pluginNum), Q_ARG(QString, name), Q_ARG(ito::AddInActuator**, &actuator), Q_ARG(QVector<ito::ParamBase>*, paramsMand), Q_ARG(QVector<ito::ParamBase>*, paramsOpt), Q_ARG(bool, autoLoadParams), Q_ARG(ItomSharedSemaphore*, waitCond));
    waitCond->wait(-1);
    retval += waitCond->returnValue;
    waitCond->deleteSemaphore();
    waitCond = NULL;

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::maddInOpenDataIO(const QString &name, const int pluginNum, const bool autoLoadParams, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::AddInDataIO *&dataIO)
{
    ito::RetVal retval(ito::retOk);

    ito::AddInManager *AIM = AddInManager::instance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toLatin1().data());
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QMetaObject::invokeMethod(AIM, "initAddIn", Q_ARG(int, pluginNum), Q_ARG(QString, name), Q_ARG(ito::AddInDataIO**, &dataIO), Q_ARG(QVector<ito::ParamBase>*, paramsMand), Q_ARG(QVector<ito::ParamBase>*, paramsOpt), Q_ARG(bool, autoLoadParams), Q_ARG(ItomSharedSemaphore*, waitCond));
    waitCond->wait(-1);
    retval += waitCond->returnValue;
    waitCond->deleteSemaphore();
    waitCond = NULL;

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::maddInClose(ito::AddInBase *instance)
{
    ito::RetVal retval(ito::retOk);

    ito::AddInManager *AIM = AddInManager::instance();
    if (!AIM)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Fatal error! Could not get addInManager instance!").toLatin1().data());
    }

    ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
    QMetaObject::invokeMethod(AIM, "closeAddIn", Q_ARG(ito::AddInBase*, instance), Q_ARG(ItomSharedSemaphore*, waitCond));
    waitCond->deleteSemaphore();
    waitCond = NULL;

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::DataObject* ApiFunctions::mcreateFromDataObject(const ito::DataObject *dObj, int nrDims, ito::tDataType type, int *sizeLimits /*= NULL*/, ito::RetVal *retval /*= NULL*/)
{
    return mcreateFromNamedDataObject(dObj, nrDims, type, NULL, sizeLimits, retval);
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::DataObject* ApiFunctions::mcreateFromNamedDataObject(const ito::DataObject *dObj, int nrDims, ito::tDataType type, const char *name /*= NULL*/, int *sizeLimits /*= NULL*/, ito::RetVal *retval /*= NULL*/)
{
    ito::DataObject *output = NULL;
    ito::RetVal ret;

    if (dObj)
    {
        if (dObj->getDims() != nrDims)
        {
            if (name)
            {
                ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The data object '%s' must have %i dimensions (%i given)").toLatin1().data(), name, nrDims, dObj->getDims());
            }
            else
            {
                ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The given data object must have %i dimensions (%i given)").toLatin1().data(), nrDims, dObj->getDims());
            }
        }
        else if(sizeLimits) //check sizeLimits (must be twice as lang as nrDims)
        {
            for (int i = 0; i < nrDims; ++i)
            {
                int s = dObj->getSize(i);
                if (s < sizeLimits[i * 2] || s > sizeLimits[i * 2 + 1])
                {
                    if (name)
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The size of the %i. dimension of data object '%s' exceeds the given boundaries [%i, %i]").toLatin1().data(), i+1, name, sizeLimits[i * 2], sizeLimits[i * 2 + 1]);
                    }
                    else
                    {
                        ret += ito::RetVal::format(ito::retError, 0, QObject::tr("The size of the %i. dimension exceeds the given boundaries [%i, %i]").toLatin1().data(), i+1, sizeLimits[i * 2], sizeLimits[i * 2 + 1]);
                    }
                    break;
                }
            }
        }

        if (!ret.containsError())
        {
            if (dObj->getType() == type)
            {
                output = new ito::DataObject(*dObj);
            }
            else
            {
                output = new ito::DataObject();
                ret += dObj->convertTo(*output, type);
            }
        }
    }

    if (ret.containsError())
    {
        DELETE_AND_SET_NULL(output);
    }

    if (retval) *retval += ret;
    return output;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/** Helper function returning the currently selected directory of the filesystemwidget
*   @param [in, out]    dir   the currently selected directory
*   @return             retOk on success, retError otherwise
*
*   The function checks if the types of the passed python parameter and the parameter are compatible and sets the parameter
*   value if it is possible. If the paramter cannot be set an error is returned.
*/
QString ApiFunctions::getCurrentWorkingDir(void)
{
    return QDir::cleanPath(QDir::currentPath());
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::mshowConfigurationDialog(ito::AddInBase *plugin, ito::AbstractAddInConfigDialog *configDialogInstance)
{
    ito::RetVal retval;

    if (QObject::connect(plugin, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), configDialogInstance, SLOT(parametersChanged(QMap<QString, ito::Param>))))
    {
        if (!QMetaObject::invokeMethod(plugin, "sendParameterRequest"))
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Error invoking 'sendParameterRequest' of the plugin").toLatin1().data());
        }
        else
        {
            if (configDialogInstance->exec())
            {
                QObject::disconnect(plugin, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), configDialogInstance, SLOT(parametersChanged(QMap<QString, ito::Param>)));
                configDialogInstance->applyParameters(); //retval is not checked since the messages should be displayed via the configuration dialog itself.
            }
            else
            {
                QObject::disconnect(plugin, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), configDialogInstance, SLOT(parametersChanged(QMap<QString, ito::Param>)));
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("The signal/slot 'parametersChanged' could not be connected").toLatin1().data());
    }

    configDialogInstance->deleteLater();

    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
/*
ito::RetVal ApiFunctions::sendParamToPyWorkspaceThreadSafe(const QString &varname, const QSharedPointer<ito::ParamBase> &value)
{
    return sendParamsToPyWorkspaceThreadSafe(QStringList(varname), QVector<QSharedPointer<ito::ParamBase> >(1, value));
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::sendParamsToPyWorkspaceThreadSafe(const QStringList &varnames, const QVector<QSharedPointer<ito::ParamBase> > &values)
{
    ito::RetVal retval;
    PythonEngine *pyEng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyEng)
    {
        if (QThread::currentThreadId() == pyEng->getPythonThreadId())
        {
            retval += pyEng->putParamsToWorkspace(true, varnames, values, NULL);
        }
        else
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(pyEng, "putParamsToWorkspace", Q_ARG(bool,true), Q_ARG(QStringList,varnames), Q_ARG(QVector<SharedParamBasePointer >, values), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
            if (locker->wait(AppManagement::timeouts.pluginGeneral))
            {
                retval += locker->returnValue;
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("timeout while sending variables to python workspace. Python is maybe busy. Try it later again.").toLatin1().data());
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("Python is not available.").toLatin1().data());
    }

    return retval;
}
*/
//------------------------------------------------------------------------------------------------------------------------------------------------------
/*static*/ QString ApiFunctions::getSettingsFile(void)
{
    return m_settingsFile;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ApiFunctions::setSettingsFile(QString settingsFile)
{
    m_settingsFile = settingsFile;
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------

}; // namespace ito
