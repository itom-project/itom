#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI

#include "CameraPlugin.h"
#include "pluginVersion.h"

#define _USE_MATH_DEFINES  // needs to be defined to enable standard declartions of PI constant
#include "math.h"

#if (defined linux) //| (defined CMAKE)
    #include <unistd.h>
#endif
#include <qstring.h>
#include <qstringlist.h>
#include <QtCore/QtPlugin>
#include <qmetaobject.h>

#include <qdockwidget.h>
#include <qpushbutton.h>
#include <qmetaobject.h>
//#include "dockWidgetDummyGrabber.h"

#include "common/helperCommon.h"

Q_DECLARE_METATYPE(ito::DataObject)


/*!
    \class CameraPluginInterface
    \brief Small interface class for class CameraPlugin. This class contains basic information about CameraPlugin as is able to
        create one or more new instances of CameraPlugin.
*/

//----------------------------------------------------------------------------------------------------------------------------------
//! creates new instance of CameraPlugin and returns the instance-pointer.
/*!
    \param [in,out] addInInst is a double pointer of type ito::AddInBase. The newly created DummyGrabber-instance is stored in *addInInst
    \return retOk
    \sa DummyGrabber
*/
ito::RetVal CameraPluginInterface::getAddInInst(ito::AddInBase **addInInst)
{
    CameraPlugin* newInst = new CameraPlugin();
    newInst->setBasePlugin(this);
    *addInInst = qobject_cast<ito::AddInBase*>(newInst);

    m_InstList.append(*addInInst);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! deletes instance of CameraPlugin. This instance is given by parameter addInInst.
/*!
    \param [in] double pointer to the instance which should be deleted.
    \return retOk
    \sa DummyGrabber
*/
ito::RetVal CameraPluginInterface::closeThisInst(ito::AddInBase **addInInst)
{
   if (*addInInst)
    {
        m_InstList.removeOne(*addInInst);
        delete ((CameraPlugin *)*addInInst);
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor for interace
/*!
    defines the plugin type (dataIO and grabber) and sets the plugins object name. If the real plugin (here: DummyGrabber) should or must
    be initialized (e.g. by a Python call) with mandatory or optional parameters, please initialize both vectors m_initParamsMand
    and m_initParamsOpt within this constructor.
*/
CameraPluginInterface::CameraPluginInterface()
{
    m_autoLoadPolicy = ito::autoLoadKeywordDefined;
    m_autoSavePolicy = ito::autoSaveAlways;

    m_type = ito::typeDataIO | ito::typeGrabber;
    setObjectName("CameraPlugin");

    m_description = QObject::tr("CameraPlugin description");
    m_detaildescription = QObject::tr("The CameraPlugin is a template for cameras.");
    m_author = "ITO";
    m_license = tr("LGPL");
    
    m_version = PLUGIN_VERSION_MAJOR << 16 + PLUGIN_VERSION_MINOR << 8 + PLUGIN_VERSION_PATCH;
    
    m_initParamsMand.clear();
    m_initParamsOpt.clear();

    /*ito::Param param("maxXSize", ito::ParamBase::Int, 640, new ito::IntMeta(1,4096), QObject::tr("description").toAscii().data());
    m_initParamsOpt.append(param);

    param = ito::Param("maxYSize", ito::ParamBase::Int, 480, new ito::IntMeta(1,4096), QObject::tr("description").toAscii().data());
    m_initParamsOpt.append(param);

    param = ito::Param("bpp", ito::ParamBase::Int, 8, new ito::IntMeta(1,32), QObject::tr("description").toAscii().data());
    m_initParamsOpt.append(param);*/

    return;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    clears both vectors m_initParamsMand and m_initParamsOpt.
*/
CameraPluginInterface::~CameraPluginInterface()
{
    m_initParamsMand.clear();
    m_initParamsOpt.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
// this makro registers the class CameraPluginInterface with the name CameraPluginInterface as plugin for the Qt-System (see Qt-DOC)
Q_EXPORT_PLUGIN2(CameraPlugininterface, CameraPluginInterface)

//----------------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------------
CameraPlugin::~CameraPlugin() {};

//----------------------------------------------------------------------------------------------------------------------------------
CameraPlugin::CameraPlugin() :
	AddInGrabber()
{
	//create internal parameter map
    ito::Param paramVal("name", ito::ParamBase::String | ito::ParamBase::Readonly, "CameraPlugin", NULL);
    m_params.insert(paramVal.getName(), paramVal);
    
    paramVal = ito::Param("integration_time", ito::ParamBase::Double, 0.005, 100.0, 12.5, tr("Integrationtime of CCD [s]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("gain", ito::ParamBase::Double, 0.0, 1.0, 1.0, tr("Virtual gain").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("offset", ito::ParamBase::Double, 0.0, 1.0, 0.5, tr("Virtual offset").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("sizex", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 2048, 2048, tr("size in x (cols) [px]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("sizey", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 2048, 2048, tr("size in y (rows) [px]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("x0", ito::ParamBase::Int, 0, 2047, 0, tr("first pixel in x (cols) within ROI [zero-based, <= x1]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("y0", ito::ParamBase::Int, 0, 2047, 0, tr("first pixel in y (rows) within ROI [zero-based, <= y1]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("x1", ito::ParamBase::Int, 0, 2047, 2047, tr("last pixel in x (cols) within ROI [zero-based, >= x0]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("y1", ito::ParamBase::Int, 0, 2047, 2047, tr("last pixel in y (rows) within ROI [zero-based, >= y0]").toAscii().data());
    m_params.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("bpp", ito::ParamBase::Int, 0, 32, 8, "bit depth of camera");
    m_params.insert(paramVal.getName(), paramVal);
    //... add further parameters to map
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;

    //use the content of paramsMand and paramsOpt (order and type
    // with respect to m_initParamsMand and m_initParamsOpt of
    // interface class) in order to initialize the hardware and
    // change values of the m_params-map if necessary.

    // emit signal about changed parameters
    emit parametersChanged(m_params);

    //release the wait condition and set its returnValue before
    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    setInitialized(true); //plugin is initialized
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::close(ItomSharedSemaphore *waitCond)
{
	ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;

    //your code comes here

    if (waitCond)
    {
        waitCond->release();
        waitCond->returnValue = retValue;
    }
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;
    QString key;
    bool hasIndex = false;
    int index;
    QString suffix;
    QMap<QString,ito::Param>::iterator it;

    //parse the given parameter-name (if you support indexed or suffix-based parameters)
    retValue += apiParseParamName(val->getName(), key, hasIndex, index, suffix);

    if (retValue == ito::retOk)
    {
            //gets the parameter key from m_params map (read-only is allowed, since we only want to get the value).
            retValue += apiGetParamFromMapByKey(m_params, key, it, false);
    }

    if (!retValue.containsError())
    {
            *val = it.value();
    }

    if (waitCond)
    {
            waitCond->returnValue = retValue;
            waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    QString key;
    bool hasIndex;
    int index;
    QString suffix;
    QMap<QString, ito::Param>::iterator it;

    //parse the given parameter-name (if you support indexed or suffix-based parameters)
    retValue += apiParseParamName( val->getName(), key, hasIndex, index, suffix );

    if(!retValue.containsError())
    {
        //gets the parameter key from m_params map (read-only is not allowed and leads to ito::retError).
        retValue += apiGetParamFromMapByKey(m_params, key, it, true);
    }

    if(!retValue.containsError())
    {
        //here the new parameter is checked whether it's type corresponds or can be cast into the
        // value in m_params and whether the new type fits to the requirements of any possible
        // meta structure.
        retValue += apiValidateParam(*it, *val, false, true);
    }

    if(!retValue.containsError())
    {
        if(key == "bpp")
        {
            //check the new value and if ok, assign it to the internal parameter
            retValue += it->copyValueFrom( &(*val) );
        }
        else if(key == "x0")
        {
            //check the new value and if ok, assign it to the internal parameter
            retValue += it->copyValueFrom( &(*val) );
        }
        else
        {
            //all parameters that don't need further checks can simply be assigned
            //to the value in m_params (the rest is already checked above)
            retValue += it->copyValueFrom( &(*val) );
        }
    }

    if(!retValue.containsError())
    {
            emit parametersChanged(m_params); //send changed parameters to any connected dialogs or dock-widgets
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::startDevice(ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue = ito::retOk;

    checkData(); //this will be reallocated in this method.

    incGrabberStarted();

    if(grabberStartedCount() == 1)
    {
        /*int ret = myCam.prepareCamera();
        if(ret)
        {
            retValue += ito::retError;
        }*/
    }

    if(waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::stopDevice(ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue = ito::retOk;

    decGrabberStarted();
    if(grabberStartedCount() == 0)
    {
        /*int ret = myCam.stopCamera();
        if(ret)
        {
            retValue += ito::retError;
        }*/

    }
    else if(grabberStartedCount() < 0)
    {
        retValue += ito::RetVal(ito::retWarning, 1001, tr("StopDevice of DummyGrabber can not be executed, since camera has not been started.").toAscii().data());
        setGrabberStarted(0);
    }


    if(waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::acquire(const int trigger, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue = ito::retOk;

    if(grabberStartedCount() <= 0)
    {
        retValue += ito::RetVal(ito::retError, 1002, tr("Acquire of DummyGrabber can not be executed, since camera has not been started.").toAscii().data());
    }
    else
    {
        /*this->m_isgrabbing = true;
        int ret = myCam.acquireImage();
        if(ret)
        {
            retValue += ito::retError;
        }*/

    }

    if(waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::getVal(void *vpdObj, ItomSharedSemaphore *waitCond)
{
	ItomSharedSemaphoreLocker locker(waitCond);
    ito::DataObject *dObj = reinterpret_cast<ito::DataObject *>(vpdObj);

    ito::RetVal retValue(ito::retOk);

    retValue += retrieveData();

    if(!retValue.containsError())
    {
        if(dObj == NULL)
        {
            retValue += ito::RetVal(ito::retError, 1004, tr("data object of getVal is NULL or cast failed").toAscii().data());
        }
        else
        {
            retValue += sendDataToListeners(0); //don't wait for live image, since user should get the image as fast as possible.

            (*dObj) = this->m_data;
        }
    }

    if (waitCond)
    {
        waitCond->returnValue=retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::copyVal(void *vpdObj, ItomSharedSemaphore *waitCond)
{
	ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    ito::DataObject *dObj = reinterpret_cast<ito::DataObject *>(vpdObj);

    if(!dObj)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Empty object handle retrieved from caller").toAscii().data());
    }
    else
    {
        retValue += checkData(dObj);  
    }

    if(!retValue.containsError())
	{
        retValue += retrieveData(dObj);  
    }

    if(!retValue.containsError())
    {
        sendDataToListeners(0); //don't wait for live image, since user should get the image as fast as possible.
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal CameraPlugin::retrieveData(ito::DataObject *externalDataObject /*= NULL*/)
{
	return ito::RetVal();
}

