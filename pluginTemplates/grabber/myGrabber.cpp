/* ********************************************************************
    Template for a camera / grabber plugin for the software itom
    
    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI

#include "myGrabber.h"
#include "pluginVersion.h"
#include "gitVersion.h"

#include <qstring.h>
#include <qstringlist.h>
#include <qplugin.h>
#include <qmessagebox.h>

#include "dockWidgetMyGrabber.h"

//----------------------------------------------------------------------------------------------------------------------------------
//! Constructor of Interface Class.
/*!
    \todo add necessary information about your plugin here.
*/
MyGrabberInterface::MyGrabberInterface()
{
    m_type = ito::typeDataIO | ito::typeGrabber; //any grabber is a dataIO device AND its subtype grabber (bitmask -> therefore the OR-combination).
    setObjectName("MyGrabber");

    m_description = QObject::tr("MyGrabber");

    //for the docstring, please don't set any spaces at the beginning of the line.
    char docstring[] = \
"This template can be used for implementing a new type of camera or grabber plugin \n\
\n\
Put a detailed description about what the plugin is doing, what is needed to get it started, limitations...";
    m_detaildescription = QObject::tr(docstring);

    m_author = PLUGIN_AUTHOR;
    m_version = PLUGIN_VERSION;
    m_minItomVer = MINVERSION;
    m_maxItomVer = MAXVERSION;
    m_license = QObject::tr("The plugin's license string");
    m_aboutThis = QObject::tr(GITVERSION); 

    //add mandatory and optional parameters for the initialization here.
    //append them to m_initParamsMand or m_initParamsOpt.
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Destructor of Interface Class.
/*!
    
*/
MyGrabberInterface::~MyGrabberInterface()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyGrabberInterface::getAddInInst(ito::AddInBase **addInInst)
{
    NEW_PLUGININSTANCE(MyGrabber) //the argument of the macro is the classname of the plugin
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyGrabberInterface::closeThisInst(ito::AddInBase **addInInst)
{
   REMOVE_PLUGININSTANCE(MyGrabber) //the argument of the macro is the classname of the plugin
   return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
#if QT_VERSION < 0x050000
    Q_EXPORT_PLUGIN2(mygrabberinterface, MyGrabberInterface) //the second parameter must correspond to the class-name of the interface class, the first parameter is arbitrary (usually the same with small letters only)
#endif




//----------------------------------------------------------------------------------------------------------------------------------
//! Constructor of plugin.
/*!
    \todo add internal parameters of the plugin to the map m_params. It is allowed to append or remove entries from m_params
    in this constructor or later in the init method
*/
MyGrabber::MyGrabber() : AddInGrabber(), m_isgrabbing(false)
{
    ito::Param paramVal("name", ito::ParamBase::String | ito::ParamBase::Readonly, "MyGrabber", NULL);
    m_params.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("x0", ito::ParamBase::Int | ito::ParamBase::In, 0, 2048, 0, tr("first pixel index in ROI (x-direction)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("y0", ito::ParamBase::Int | ito::ParamBase::In, 0, 2048, 0, tr("first pixel index in ROI (y-direction)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("x1", ito::ParamBase::Int | ito::ParamBase::In, 0, 1279, 1279, tr("last pixel index in ROI (x-direction)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("y1", ito::ParamBase::Int | ito::ParamBase::In, 0, 1023, 1023, tr("last pixel index in ROI (y-direction)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("sizex", ito::ParamBase::Int | ito::ParamBase::Readonly | ito::ParamBase::In, 1, 2048, 2048, tr("width of ROI (x-direction)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("sizey", ito::ParamBase::Int | ito::ParamBase::Readonly | ito::ParamBase::In, 1, 2048, 2048, tr("height of ROI (y-direction)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("bpp", ito::ParamBase::Int | ito::ParamBase::In, 8, 8, 8, tr("bpp").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    paramVal = ito::Param("integrationTime", ito::ParamBase::Double | ito::ParamBase::In, 0.0, 1.0, 0.01, tr("Integrationtime of CCD [0..1] (no unit)").toLatin1().data());
    m_params.insert(paramVal.getName(), paramVal);
    
    //the following lines create and register the plugin's dock widget. Delete these lines if the plugin does not have a dock widget.
    DockWidgetMyGrabber *dw = new DockWidgetMyGrabber(this);
    
    Qt::DockWidgetAreas areas = Qt::AllDockWidgetAreas;
    QDockWidget::DockWidgetFeatures features = QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable;
    createDockWidget(QString(m_params["name"].getVal<char *>()), features, areas, dw);   
}

//----------------------------------------------------------------------------------------------------------------------------------
MyGrabber::~MyGrabber()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
//! initialization of plugin
/*!
    \sa close
*/
ito::RetVal MyGrabber::init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);

    //steps todo:
    // - get all initialization parameters
    // - try to detect your device
    // - establish a connection to the device
    // - synchronize the current parameters of the device with the current values of parameters inserted in m_params
    // - if an identifier string of the device is available, set it via setIdentifier("yourIdentifier")
    // - call checkData() in order to reconfigure the temporary image buffer m_data (or other structures) depending on the current size, image type...
    // - call emit parametersChanged(m_params) in order to propagate the current set of parameters in m_params to connected dock widgets...
    // - call setInitialized(true) to confirm the end of the initialization (even if it failed)
    
    if (!retValue.containsError())
    {        
        retValue += checkData();
    }
    
    if (!retValue.containsError())
    {
        emit parametersChanged(m_params);
    }
    
    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    setInitialized(true); //init method has been finished (independent on retval)
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! shutdown of plugin
/*!
    \sa init
*/
ito::RetVal MyGrabber::close(ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    
    //todo:
    // - disconnect the device if not yet done
    // - this funtion is considered to be the "inverse" of init.

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }
    
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyGrabber::getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond)
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
        //put your switch-case.. for getting the right value here

        //finally, save the desired value in the argument val (this is a shared pointer!)
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
ito::RetVal MyGrabber::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond)
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

    if (!retValue.containsError())
    {
        //gets the parameter key from m_params map (read-only is not allowed and leads to ito::retError).
        retValue += apiGetParamFromMapByKey(m_params, key, it, true);
    }

    if (!retValue.containsError())
    {
        //here the new parameter is checked whether its type corresponds or can be cast into the
        // value in m_params and whether the new type fits to the requirements of any possible
        // meta structure.
        retValue += apiValidateParam(*it, *val, false, true);
    }

    if (!retValue.containsError())
    {
        if (key == "demoKey1")
        {
            //check the new value and if ok, assign it to the internal parameter
            retValue += it->copyValueFrom( &(*val) );
        }
        else if (key == "demoKey2")
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

    if (!retValue.containsError())
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
ito::RetVal MyGrabber::startDevice(ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    
    incGrabberStarted(); //increment a counter to see how many times startDevice has been called
    
    //todo:
    // if this function has been called for the first time (grabberStartedCount() == 1),
    // start the camera, allocate necessary buffers or do other work that is necessary
    // to prepare the camera for image acquisitions.
    
    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }
    return retValue;
}
         
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyGrabber::stopDevice(ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);

    decGrabberStarted(); //decrements the counter (see startDevice)

    if (grabberStartedCount() < 0)
    {
        retValue += ito::RetVal(ito::retWarning, 0, tr("The grabber has already been stopped.").toLatin1().data());
        setGrabberStarted(0);
    }
    
    //todo:
    // if the counter (obtained by grabberStartedCount()) drops to zero again, stop the camera, free all allocated
    // image buffers of the camera... (it is the opposite from all things that have been started, allocated... in startDevice)

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }
    return ito::retOk;
}
         
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyGrabber::acquire(const int trigger, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    bool RetCode = false;

    if (grabberStartedCount() <= 0)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Tried to acquire an image without having started the device.").toLatin1().data());
    }
    else
    {
        m_isgrabbing = true;
    }
    
    //todo:
    // trigger the camera for starting the acquisition of a new image (software trigger or make hardware trigger ready (depending on trigger (0: software trigger, default))
    
    //now the wait condition is released, such that itom (caller) stops waiting and continuous with its own execution.
    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();  
    }
    
    //todo:
    // it is possible now, to wait here until the acquired image is ready
    // if you want to do this here, wait for the finished image, get it and save it
    // to any accessible buffer, for instance the m_data dataObject that is finally delivered
    // via getVal or copyVal.
    // 
    // you can also implement this waiting and obtaining procedure in retrieveImage.
    // If you do it here, the camera thread is blocked until the image is obtained, such that calls to getParam, setParam, stopDevice...
    // are not executed during the waiting operation. They are queued and executed once the image is acquired and transmitted to the plugin.

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyGrabber::retrieveData(ito::DataObject *externalDataObject)
{
    //todo: this is just a basic example for getting the buffered image to m_data or the externally given data object
    //enhance it and adjust it for your needs
    ito::RetVal retValue(ito::retOk);

    ito::DataObject *dataObj = externalDataObject ? externalDataObject : &m_data;

    bool hasListeners = (m_autoGrabbingListeners.size() > 0);
    bool copyExternal = (externalDataObject != NULL);
    
    const int bufferWidth = m_params["sizex"].getVal<int>();
    const int bufferHeight = m_params["sizey"].getVal<int>();



    if (m_isgrabbing == false)
    {
        retValue += ito::RetVal(ito::retWarning, 0, tr("Tried to get picture without triggering exposure").toLatin1().data());
    }
    else
    {
        //step 1: create m_data (if not yet available)
        if (externalDataObject && hasListeners)
        {
            retValue += checkData(NULL); //update m_data
            retValue += checkData(externalDataObject); //update external object
        }
        else
        {
            retValue += checkData(externalDataObject); //update external object or m_data
        }
        
        if (!retValue.containsError())
        {
            if (m_data.getType() == ito::tUInt8)
            {
                if (copyExternal)
                {
                    retValue += externalDataObject->copyFromData2D<ito::uint8>((ito::uint8*) bufferPtr, bufferWidth, bufferHeight);
                }
                if (!copyExternal || hasListeners)
                {
                    retValue += m_data.copyFromData2D<ito::uint8>((ito::uint8*) bufferPtr, bufferWidth, bufferHeight);
                }
            }
            else if (m_data.getType() == ito::tUInt16)
            {
                if (copyExternal)
                {
                    retValue += externalDataObject->copyFromData2D<ito::uint16>((ito::uint16*) bufferPtr, bufferWidth, bufferHeight);
                }
                if (!copyExternal || hasListeners)
                {
                    retValue += m_data.copyFromData2D<ito::uint16>((ito::uint16*) bufferPtr, bufferWidth, bufferHeight);            
                }
            }
            else
            {
                retValue += ito::RetVal(ito::retError, 1002, tr("copying image buffer not possible since unsupported type.").toLatin1().data());
            }
        }

        m_isgrabbing = false;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
// usually it is not necessary to implement the checkData method, since the default implementation from AddInGrabber is already
// sufficient.
//
// What is does:
// - it obtains the image size from sizex, sizey, bpp
// - it checks whether the rows, cols and type of m_data are unequal to the requested dimensions and type
// - if so, m_data is reallocated, else nothing is done
// - if an external data object is given (from copyVal), this object is checked in place of m_data
// - the external data object is only reallocated if it is empty, else its size or its region of interest must exactly
//    fit to the given size restrictions
//
// if you need to do further things, overload checkData and implement your version there
/*ito::RetVal MyGrabber::checkData(ito::DataObject *externalDataObject)
{
    return ito::retOk;
}*/

//----------------------------------------------------------------------------------------------------------------------------------
//! Returns the grabbed camera frame as reference.
/*!
    This method returns a reference to the recently acquired image. Therefore this camera size must fit to the data structure of the 
    DataObject.
    
    This method returns a reference to the internal dataObject m_data of the camera where the currently acquired image data is copied to (either
    in the acquire method or in retrieve data). Please remember, that the reference may directly change if a new image is acquired.

    \param [in,out] vpdObj is the pointer to a given dataObject (this pointer should be cast to ito::DataObject*). After the call, the dataObject is a reference to the internal m_data dataObject of the camera.
    \param [in] waitCond is the semaphore (default: NULL), which is released if this method has been terminated
    \return retOk if everything is ok, retError is camera has not been started or no image has been acquired by the method acquire.
    
    \sa retrieveImage, copyVal
*/
ito::RetVal MyGrabber::getVal(void *vpdObj, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    ito::DataObject *dObj = reinterpret_cast<ito::DataObject *>(vpdObj);
    
    //call retrieveData without argument. Retrieve data should then put the currently acquired image into the dataObject m_data of the camera.
    retValue += retrieveData();

    if (!retValue.containsError())
    {
        //send newly acquired image to possibly connected live images
        sendDataToListeners(0); //don't wait for live data, since user should get the data as fast as possible.

        if (dObj)
        {
            (*dObj) = m_data; //copy reference to externally given object
        }
    }

    if (waitCond) 
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Returns the grabbed camera frame as a deep copy.
/*!
    This method copies the recently grabbed camera frame to the given DataObject. 
    
    The given dataObject must either have an empty size (then it is resized to the size and type of the camera image) or its size or adjusted region of
    interest must exactly fit to the size of the camera. Then, the acquired image is copied inside of the given region of interest (copy into a subpart of
    an image stack is possible then)

    \param [in,out] vpdObj is the pointer to a given dataObject (this pointer should be cast to ito::DataObject*) where the acquired image is deep copied to.
    \param [in] waitCond is the semaphore (default: NULL), which is released if this method has been terminated
    \return retOk if everything is ok, retError is camera has not been started or no image has been acquired by the method acquire.
    
    \sa retrieveImage, getVal
*/
ito::RetVal MyGrabber::copyVal(void *vpdObj, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    ito::DataObject *dObj = reinterpret_cast<ito::DataObject *>(vpdObj);
    
    if (!dObj)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Empty object handle retrieved from caller").toLatin1().data());
    }
    
    if (!retValue.containsError())
    {
        //this method calls retrieveData with the passed dataObject as argument such that retrieveData is able to copy the image obtained
        //by the camera directly into the given, external dataObject
        retValue += retrieveData(dObj);  //checkData is executed inside of retrieveData
    }

    if (!retValue.containsError())
    {
        //send newly acquired image to possibly connected live images
        sendDataToListeners(0); //don't wait for live data, since user should get the data as fast as possible.
    }
    
    if (waitCond) 
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }
    
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot called if the dock widget of the plugin becomes (in)visible
/*!
    Overwrite this method if the plugin has a dock widget. If so, you can connect the parametersChanged signal of the plugin
    with the dock widget once its becomes visible such that no resources are used if the dock widget is not visible. Right after
    a re-connection emit parametersChanged(m_params) in order to send the current status of all plugin parameters to the dock widget.
*/
void MyGrabber::dockWidgetVisibilityChanged(bool visible)
{
    if (getDockWidget())
    {
        QWidget *widget = getDockWidget()->widget();
        if (visible)
        {
            connect(this, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), widget, SLOT(parametersChanged(QMap<QString, ito::Param>)));

            emit parametersChanged(m_params);
        }
        else
        {
            disconnect(this, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), widget, SLOT(parametersChanged(QMap<QString, ito::Param>)));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method called to show the configuration dialog
/*!
    This method is called from the main thread from itom and should show the configuration dialog of the plugin.
    If the instance of the configuration dialog has been created, its slot 'parametersChanged' is connected to the signal 'parametersChanged'
    of the plugin. By invoking the slot sendParameterRequest of the plugin, the plugin's signal parametersChanged is immediately emitted with
    m_params as argument. Therefore the configuration dialog obtains the current set of parameters and can be adjusted to its values.
    
    The configuration dialog should emit reject() or accept() depending if the user wanted to close the dialog using the ok or cancel button.
    If ok has been clicked (accept()), this method calls applyParameters of the configuration dialog in order to force the dialog to send
    all changed parameters to the plugin. If the user clicks an apply button, the configuration dialog itsself must call applyParameters.
    
    If the configuration dialog is inherited from AbstractAddInConfigDialog, use the api-function apiShowConfigurationDialog that does all
    the things mentioned in this description.
    
    Remember that you need to implement hasConfDialog in your plugin and return 1 in order to signalize itom that the plugin
    has a configuration dialog.
    
    \sa hasConfDialog
*/
const ito::RetVal MyGrabber::showConfDialog(void)
{
    return apiShowConfigurationDialog(this, new DialogMyGrabber(this));
}