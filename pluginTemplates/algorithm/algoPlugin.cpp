/* ********************************************************************
    Template for an algorithm plugin for the software itom
    
    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI

#include "algoPlugin.h"
#include "pluginVersion.h"

#include <QtCore/QtPlugin>

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPluginInterface::getAddInInst(ito::AddInBase **addInInst)
{
    NEW_PLUGININSTANCE(AlgoPlugin)
    REGISTER_FILTERS_AND_WIDGETS
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPluginInterface::closeThisInst(ito::AddInBase **addInInst)
{
    REMOVE_PLUGININSTANCE(AlgoPlugin)
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Constructor of Interface Class.
/*!
    \todo add necessary information about your plugin here.
*/
AlgoPluginInterface::AlgoPluginInterface()
{
    m_type = ito::typeAlgo;
    setObjectName("AlgoPlugin");

    m_description = QObject::tr("AlgoPlugin");

    //for the docstring, please don't set any spaces at the beginning of the line.
    char docstring[] = \
"This template can be used for implementing a new type of algorithm plugin \n\
\n\
Put a detailed description about what the plugin is doing, what is needed to get it started, limitations...";
    m_detaildescription = QObject::tr(docstring);

    m_author = "Authors of the plugin";
    m_version = (PLUGIN_VERSION_MAJOR << 16) + (PLUGIN_VERSION_MINOR << 8) + PLUGIN_VERSION_PATCH;
    m_minItomVer = MINVERSION;
    m_maxItomVer = MAXVERSION;
    m_license = QObject::tr("The plugin's license string");
    m_aboutThis = QObject::tr("");
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Destructor of Interface Class.
/*!
    
*/
AlgoPluginInterface::~AlgoPluginInterface()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
#if QT_VERSION < 0x050000
    Q_EXPORT_PLUGIN2(algoplugininterface, AlgoPluginInterface) //the second parameter must correspond to the class-name of the interface class, the first parameter is arbitrary (usually the same with small letters only)
#endif

//----------------------------------------------------------------------------------------------------------------------------------
AlgoPlugin::AlgoPlugin() : AddInAlgo()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
AlgoPlugin::~AlgoPlugin()
{
}


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPlugin::init(QVector<ito::ParamBase> * /*paramsMand*/, QVector<ito::ParamBase> * /*paramsOpt*/, ItomSharedSemaphore * /*waitCond*/)
{
    ito::RetVal retval = ito::retOk;
    FilterDef *filter = NULL;
    
    //register each algorithm with the following code snippet
    filter = new FilterDef(AlgoPlugin::algo1, AlgoPlugin::algo1Params, tr(algo1doc));
    m_filterList.insert("algo1Name", filter);

    setInitialized(true); //init method has been finished (independent on retval)
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPlugin::close(ItomSharedSemaphore * waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);

    ito::RetVal retval;

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
const char* AlgoPlugin::algo1doc = "detailed description of the algorithm 'algo1'.\n\
\n\
You can use line breaks here, but always start in the first column.";


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPlugin::algo1Params(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::Param param;
    ito::RetVal retval = ito::retOk;
    retval += prepareParamVectors(paramsMand,paramsOpt,paramsOut);
    if(retval.containsError()) return retval;

    paramsMand->clear();
    paramsMand->append( ito::Param("mandParam1", ito::ParamBase::DObjPtr | ito::ParamBase::In, NULL, "description") );
    
    paramsOpt->clear();
    paramsOpt->append( ito::Param("optParam1", ito::ParamBase::String | ito::ParamBase::In, "default", "description") );
    
    paramsOut->append( ito::Param("outParam1", ito::ParamBase::Double | ito::ParamBase::Out, 0.0, ito::DoubleMeta::all(), "description") );

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal AlgoPlugin::algo1(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
    ito::DataObject *dObj = paramsMand->at(0).getVal<ito::DataObject*>();
    QString str = paramsOpt->at(0).getVal<char*>();
    
    (*paramsOut)[0].setVal<double>(1.0);
    
    return retval;
}