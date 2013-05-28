#include "algoPlugin.h"

#include "common/helperCommon.h"

#include <QtCore/QtPlugin>
#include <qstringlist.h>
#include <qvariant.h>

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPluginInterface::getAddInInst(ito::AddInBase **addInInst)
{
    FittingFilters* newInst = new FittingFilters();
    newInst->setBasePlugin(this);
    *addInInst = qobject_cast<ito::AddInBase*>(newInst);
    QList<QString> keyList = newInst->m_filterList.keys();
    for (int i = 0; i < newInst->m_filterList.size(); i++)
    {
        newInst->m_filterList[keyList[i]]->m_pBasePlugin = this;
    }

    m_InstList.append(*addInInst);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPluginInterface::closeThisInst(ito::AddInBase **addInInst)
{
       if (*addInInst)
    {
        FittingFilters * thisInst = qobject_cast<FittingFilters*>(*addInInst);
        if(thisInst)
        {
            delete thisInst;
            int idx = m_InstList.indexOf(*addInInst);
            m_InstList.removeAt(idx);
        }
        else
        {
            return ito::RetVal(ito::retError, 0, tr("plugin-instance cannot be converted to class FittingFilters. Close operation failed").toAscii().data());
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
AlgoPluginInterface::AlgoPluginInterface()
{
    m_type = ito::typeAlgo;
    setObjectName("FittingFilters");

    m_description = QObject::tr("Filter-Plugin for fitting-methods.");
    m_author = "ITO";
    m_license = QObject::tr("LGPL with ITO itom-exception");
}

//----------------------------------------------------------------------------------------------------------------------------------
AlgoPluginInterface::~AlgoPluginInterface()
{
    m_initParamsMand.clear();
    m_initParamsOpt.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(AlgoPluginInterface, AlgoPluginInterface)

//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
AlgoPlugin::AlgoPlugin() : AddInAlgo()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
AlgoPlugin::~AlgoPlugin()
{
}


//----------------------------------------------------------------------------------------------------------------------------------
RetVal AlgoPlugin::init(QVector<ito::ParamBase> * /*paramsMand*/, QVector<ito::ParamBase> * /*paramsOpt*/, ItomSharedSemaphore * /*waitCond*/)
{
    ito::RetVal retval = ito::retOk;
    FilterDef *filter = NULL;

    filter = new FilterDef(AlgoPlugin::algo1, FittingFilters::algo1Params, tr("description"));
    m_filterList.insert("algo1Name", filter);

    setInitialized(true); //init method has been finished (independent on retval)
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AlgoPlugin::close(ItomSharedSemaphore * /*waitCond*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);

    ito::RetVal retval = ito::retOk;

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal FittingFilters::fitPlaneParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::Param param;
    ito::RetVal retval = ito::retOk;
    retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
    if(retval.containsError()) return retval;

    paramsMand->clear();
    paramsMand->append( ito::Param("mandParam1", ito::ParamBase::PointCloudPtr | ito::ParamBase::In, NULL, "description") );
    
    paramsOpt->clear();
    paramsOpt->append( ito::Param("optParam1", ito::ParamBase::String, "default", "description") );

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal FittingFilters::fitPlane(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
    return retval;
}