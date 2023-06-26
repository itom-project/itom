/* ********************************************************************
    Template for an actuator plugin for the software itom

    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI

#include "myActuator.h"
#include "pluginVersion.h"
#include "gitVersion.h"

#include <qstring.h>
#include <qstringlist.h>
#include <qplugin.h>
#include <qmessagebox.h>
#include <qdatetime.h>
#include <qwaitcondition.h>

#include "dockWidgetMyActuator.h"

//----------------------------------------------------------------------------------------------------------------------------------
//! Constructor of Interface Class.
/*!
    \todo add necessary information about your plugin here.
*/
MyActuatorInterface::MyActuatorInterface()
{
    m_type = ito::typeActuator;
    setObjectName("MyActuator");

    m_description = QObject::tr("MyActuator");

    //for the docstring, please don't set any spaces at the beginning of the line.
    char docstring[] = \
"This template can be used for implementing a new type of actuator plugin \n\
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
MyActuatorInterface::~MyActuatorInterface()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyActuatorInterface::getAddInInst(ito::AddInBase **addInInst)
{
    NEW_PLUGININSTANCE(MyActuator) //the argument of the macro is the classname of the plugin
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MyActuatorInterface::closeThisInst(ito::AddInBase **addInInst)
{
   REMOVE_PLUGININSTANCE(MyActuator) //the argument of the macro is the classname of the plugin
   return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
#if QT_VERSION < 0x050000
    Q_EXPORT_PLUGIN2(myactuatorinterface, MyActuatorInterface) //the second parameter must correspond to the class-name of the interface class, the first parameter is arbitrary (usually the same with small letters only)
#endif




//----------------------------------------------------------------------------------------------------------------------------------
//! Constructor of plugin.
/*!
    \todo add internal parameters of the plugin to the map m_params. It is allowed to append or remove entries from m_params
    in this constructor or later in the init method
*/
MyActuator::MyActuator() : AddInActuator(), m_async(0), m_nrOfAxes(3)
{
    ito::Param paramVal("name", ito::ParamBase::String | ito::ParamBase::Readonly, "MyActuator", NULL);
    m_params.insert(paramVal.getName(), paramVal);

    m_params.insert( "async", ito::Param("async", ito::ParamBase::Int, 0, 1, m_async, tr("asynchronous move (1), synchronous (0) [default]").toLatin1().data()));

    //initialize the current position vector, the status vector and the target position vector
    m_currentPos.fill(0.0,m_nrOfAxes);
    m_currentStatus.fill(0,m_nrOfAxes);
    m_targetPos.fill(0.0,m_nrOfAxes);

    //the following lines create and register the plugin's dock widget. Delete these lines if the plugin does not have a dock widget.
    DockWidgetMyActuator *dw = new DockWidgetMyActuator(this);

    Qt::DockWidgetAreas areas = Qt::AllDockWidgetAreas;
    QDockWidget::DockWidgetFeatures features = QDockWidget::DockWidgetClosable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable;
    createDockWidget(QString(m_params["name"].getVal<char *>()), features, areas, dw);
}

//----------------------------------------------------------------------------------------------------------------------------------
MyActuator::~MyActuator()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
//! initialization of plugin
/*!
    \sa close
*/
ito::RetVal MyActuator::init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);

    //steps todo:
    // - get all initialization parameters
    // - try to detect your device
    // - establish a connection to the device
    // - synchronize the current parameters of the device with the current values of parameters inserted in m_params
    // - if an identifier string of the device is available, set it via setIdentifier("yourIdentifier")
    // - set m_nrOfAxes to the number of axes
    // - resize and refill m_currentStatus, m_currentPos and m_targetPos with the corresponding values
    // - call emit parametersChanged(m_params) in order to propagate the current set of parameters in m_params to connected dock widgets...
    // - call setInitialized(true) to confirm the end of the initialization (even if it failed)


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
ito::RetVal MyActuator::close(ItomSharedSemaphore *waitCond)
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
ito::RetVal MyActuator::getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond)
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

    if(retValue == ito::retOk)
    {
        //gets the parameter key from m_params map (read-only is allowed, since we only want to get the value).
        retValue += apiGetParamFromMapByKey(m_params, key, it, false);
    }

    if(!retValue.containsError())
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
ito::RetVal MyActuator::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond)
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

    if(isMotorMoving()) //this if-case is for actuators only.
    {
        retValue += ito::RetVal(ito::retError, 0, tr("any axis is moving. Parameters cannot be set.").toLatin1().data());
    }

    if(!retValue.containsError())
    {
        //gets the parameter key from m_params map (read-only is not allowed and leads to ito::retError).
        retValue += apiGetParamFromMapByKey(m_params, key, it, true);
    }

    if(!retValue.containsError())
    {
        //here the new parameter is checked whether its type corresponds or can be cast into the
        // value in m_params and whether the new type fits to the requirements of any possible
        // meta structure.
        retValue += apiValidateParam(*it, *val, false, true);
    }

    if(!retValue.containsError())
    {
        if(key == "async")
        {
            m_async = val->getVal<int>();
            //check the new value and if ok, assign it to the internal parameter
            retValue += it->copyValueFrom( &(*val) );
        }
        else if(key == "demoKey2")
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
//! calib
/*!
    the given axis should be calibrated (e.g. by moving to a reference switch).
*/
ito::RetVal MyActuator::calib(const int axis, ItomSharedSemaphore *waitCond)
{
    return calib(QVector<int>(1,axis), waitCond);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! calib
/*!
    the given axes should be calibrated (e.g. by moving to a reference switch).
*/
ito::RetVal MyActuator::calib(const QVector<int> axis, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue = ito::RetVal(ito::retWarning, 0, tr("calibration not possible").toLatin1().data());

    if(isMotorMoving())
    {
        retValue += ito::RetVal(ito::retError, 0, tr("motor is running. Further action is not possible").toLatin1().data());
    }

    if (!retValue.containsError())
    {
        //todo:
        //start calibrating the given axes and don't forget to regularily call setAlive().
        //this is important if the calibration needs more time than the timeout time of itom (e.g. 5sec).
        //itom regularily checks the alive flag and only drops to a timeout if setAlive() is not regularily called (at least all 3-4 secs).
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();

    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! setOrigin
/*!
    the given axis should be set to origin. That means (if possible) its current position should be
    considered to be the new origin (zero-position). If this operation is not possible, return a
    warning.
*/
ito::RetVal MyActuator::setOrigin(const int axis, ItomSharedSemaphore *waitCond)
{
    return setOrigin(QVector<int>(1,axis), waitCond);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! setOrigin
/*!
    the given axes should be set to origin. That means (if possible) their current position should be
    considered to be the new origin (zero-position). If this operation is not possible, return a
    warning.
*/
ito::RetVal MyActuator::setOrigin(QVector<int> axis, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);

    if(isMotorMoving())
    {
        retValue += ito::RetVal(ito::retError, 0, tr("motor is running. Additional actions are not possible.").toLatin1().data());
    }
    else
    {
        foreach(const int &i,axis)
        {
            if (i >= 0 && i < m_nrOfAxes)
            {
                //todo: set axis i to origin (current position is considered to be the 0-position).
            }
            else
            {
                retValue += ito::RetVal::format(ito::retError, 1, tr("axis %i not available").toLatin1().data(), i);
            }
        }

        retValue += updateStatus();
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! getStatus
/*!
    re-checks the status (current position, available, end switch reached, moving, at target...) of all axes and
    returns the status of each axis as vector. Each status is an or-combination of the enumeration ito::tActuatorStatus.
*/
ito::RetVal MyActuator::getStatus(QSharedPointer<QVector<int> > status, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);

    retValue += updateStatus();
    *status = m_currentStatus;

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! getPos
/*!
    returns the current position (in mm or degree) of the given axis
*/
ito::RetVal MyActuator::getPos(const int axis, QSharedPointer<double> pos, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    QSharedPointer<QVector<double> > pos2(new QVector<double>(1,0.0));
    ito::RetVal retValue = getPos(QVector<int>(1,axis), pos2, NULL); //forward to multi-axes version
    *pos = (*pos2)[0];

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! getPos
/*!
    returns the current position (in mm or degree) of all given axes
*/
ito::RetVal MyActuator::getPos(QVector<int> axis, QSharedPointer<QVector<double> > pos, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);

    foreach (const int i, axis)
    {
        if (i >= 0 && i < m_nrOfAxes)
        {
            //obtain current position of axis i
            //transform tempPos to angle
            m_currentPos[i] = 0.0; //set m_currentPos[i] to the obtained position
            (*pos)[i] = m_currentPos[i];
        }
        else
        {
            retValue += ito::RetVal::format(ito::retError, 1, tr("axis %i not available").toLatin1().data(),i);
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
//! setPosAbs
/*!
    starts moving the given axis to the desired absolute target position

    depending on m_async this method directly returns after starting the movement (async = 1) or
    only returns if the axis reached the given target position (async = 0)

    In some cases only relative movements are possible, then get the current position, determine the
    relative movement and call the method relatively move the axis.
*/
ito::RetVal MyActuator::setPosAbs(const int axis, const double pos, ItomSharedSemaphore *waitCond)
{
    return setPosAbs(QVector<int>(1,axis), QVector<double>(1,pos), waitCond);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! setPosAbs
/*!
    starts moving all given axes to the desired absolute target positions

    depending on m_async this method directly returns after starting the movement (async = 1) or
    only returns if all axes reached their given target positions (async = 0)

    In some cases only relative movements are possible, then get the current position, determine the
    relative movement and call the method relatively move the axis.
*/
ito::RetVal MyActuator::setPosAbs(QVector<int> axis, QVector<double> pos, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    bool released = false;

    if(isMotorMoving())
    {
        retValue += ito::RetVal(ito::retError, 0, tr("motor is running. Additional actions are not possible.").toLatin1().data());
    }
    else
    {
        foreach(const int i, axis)
        {
            if (i < 0 || i >= m_nrOfAxes)
            {
                retValue += ito::RetVal::format(ito::retError, 1, tr("axis %i not available").toLatin1().data(), i);
            }
            else
            {
                m_targetPos[i] = 0.0; //todo: set the absolute target position to the desired value in mm or degree
            }
        }

        if (!retValue.containsError())
        {
            //set status of all given axes to moving and keep all flags related to the status and switches
            setStatus(axis, ito::actuatorMoving, ito::actSwitchesMask | ito::actStatusMask);

            //todo: start the movement

            //emit the signal targetChanged with m_targetPos as argument, such that all connected slots gets informed about new targets
            sendTargetUpdate();

            //emit the signal sendStatusUpdate such that all connected slots gets informed about changes in m_currentStatus and m_currentPos.
            sendStatusUpdate();

            //release the wait condition now, if async is true (itom considers this method to be finished now due to the threaded call)
            if(m_async && waitCond && !released)
            {
                waitCond->returnValue = retValue;
                waitCond->release();
                released = true;
            }

            //call waitForDone in order to wait until all axes reached their target or a given timeout expired
            //the m_currentPos and m_currentStatus vectors are updated within this function
            retValue += waitForDone(60000, axis); //WaitForAnswer(60000, axis);

            //release the wait condition now, if async is false (itom waits until now if async is false, hence in the synchronous mode)
            if(!m_async && waitCond && !released)
            {
                waitCond->returnValue = retValue;
                waitCond->release();
                released = true;
            }
        }
    }

    //if the wait condition has not been released yet, do it now
    if (waitCond && !released)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! setPosRel
/*!
    starts moving the given axis by the given relative distance

    depending on m_async this method directly returns after starting the movement (async = 1) or
    only returns if the axis reached the given target position (async = 0)

    In some cases only absolute movements are possible, then get the current position, determine the
    new absolute target position and call setPosAbs with this absolute target position.
*/
ito::RetVal MyActuator::setPosRel(const int axis, const double pos, ItomSharedSemaphore *waitCond)
{
    return setPosRel(QVector<int>(1,axis), QVector<double>(1,pos), waitCond);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! setPosRel
/*!
    starts moving the given axes by the given relative distances

    depending on m_async this method directly returns after starting the movement (async = 1) or
    only returns if all axes reached the given target positions (async = 0)

    In some cases only absolute movements are possible, then get the current positions, determine the
    new absolute target positions and call setPosAbs with these absolute target positions.
*/
ito::RetVal MyActuator::setPosRel(QVector<int> axis, QVector<double> pos, ItomSharedSemaphore *waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    bool released = false;

    if(isMotorMoving())
    {
        retValue += ito::RetVal(ito::retError, 0, tr("motor is running. Additional actions are not possible.").toLatin1().data());
    }
    else
    {
        foreach(const int i, axis)
        {
            if (i < 0 || i >= m_nrOfAxes)
            {
                retValue += ito::RetVal::format(ito::retError, 1, tr("axis %i not available").toLatin1().data(), i);
            }
            else
            {
                m_targetPos[i] = 0.0; //todo: set the absolute target position to the desired value in mm or degree
                                      //(obtain the absolute position with respect to the given relative distances)
            }
        }

        if (!retValue.containsError())
        {
            //set status of all given axes to moving and keep all flags related to the status and switches
            setStatus(axis, ito::actuatorMoving, ito::actSwitchesMask | ito::actStatusMask);

            //todo: start the movement

            //emit the signal targetChanged with m_targetPos as argument, such that all connected slots gets informed about new targets
            sendTargetUpdate();

            //emit the signal sendStatusUpdate such that all connected slots gets informed about changes in m_currentStatus and m_currentPos.
            sendStatusUpdate();

            //release the wait condition now, if async is true (itom considers this method to be finished now due to the threaded call)
            if(m_async && waitCond && !released)
            {
                waitCond->returnValue = retValue;
                waitCond->release();
                released = true;
            }

            //call waitForDone in order to wait until all axes reached their target or a given timeout expired
            //the m_currentPos and m_currentStatus vectors are updated within this function
            retValue += waitForDone(60000, axis); //WaitForAnswer(60000, axis);

            //release the wait condition now, if async is false (itom waits until now if async is false, hence in the synchronous mode)
            if(!m_async && waitCond && !released)
            {
                waitCond->returnValue = retValue;
                waitCond->release();
                released = true;
            }
        }
    }

    //if the wait condition has not been released yet, do it now
    if (waitCond && !released)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method must be overwritten from ito::AddInActuator
/*!
    WaitForDone should wait for a moving motor until the indicated axes (or all axes of nothing is indicated) have stopped or a timeout or user interruption
    occurred. The timeout can be given in milliseconds, or -1 if no timeout should be considered. The flag-parameter can be used for your own purpose.
*/
ito::RetVal MyActuator::waitForDone(const int timeoutMS, const QVector<int> axis, const int flags)
{
    ito::RetVal retVal(ito::retOk);
    bool done = false;
    bool timeout = false;
    char motor;
    QElapsedTimer timer;
    QMutex waitMutex;
    QWaitCondition waitCondition;
    long delay = 100; //[ms]

    timer.start();

    //if axis is empty, all axes should be observed by this method
    QVector<int> _axis = axis;
    if (_axis.size() == 0) //all axis
    {
        for (int i=0;i<m_nrOfAxes;i++)
        {
            _axis.append(i);
        }
    }

    while (!done && !timeout)
    {
        //todo: obtain the current position, status... of all given axes

        done = true; //assume all axes at target
        motor = 0;
        foreach(const int &i,axis)
        {
            m_currentPos[i] = 0.0; //todo: if possible, set the current position if axis i to its current position

            if (1 /*axis i is still moving*/)
            {
                setStatus(m_currentStatus[i], ito::actuatorMoving, ito::actSwitchesMask | ito::actStatusMask);
                done = false; //not done yet
            }
            else if (1 /*axis i is at target*/)
            {
                setStatus(m_currentStatus[i], ito::actuatorAtTarget, ito::actSwitchesMask | ito::actStatusMask);
                done = false; //not done yet
            }
        }

        //emit actuatorStatusChanged with both m_currentStatus and m_currentPos as arguments
        sendStatusUpdate(false);

        //now check if the interrupt flag has been set (e.g. by a button click on its dock widget)
        if (!done && isInterrupted())
        {
            //todo: force all axes to stop

            //set the status of all axes from moving to interrupted (only if moving was set before)
            replaceStatus(_axis, ito::actuatorMoving, ito::actuatorInterrupted);
            sendStatusUpdate(true);

            retVal += ito::RetVal(ito::retError,0,"interrupt occurred");
            done = true;
            return retVal;
        }

        //short delay
        waitMutex.lock();
        waitCondition.wait(&waitMutex, delay);
        waitMutex.unlock();

        //raise the alive flag again, this is necessary such that itom does not drop into a timeout if the
        //positioning needs more time than the allowed timeout time.
        setAlive();

        if (timeoutMS > -1)
        {
            if (timer.elapsed() > timeoutMS) timeout = true;
        }
    }

    if (timeout)
    {
        //timeout occured, set the status of all currently moving axes to timeout
        replaceStatus(_axis, ito::actuatorMoving, ito::actuatorTimeout);
        retVal += ito::RetVal(ito::retError,9999,"timeout occurred");
        sendStatusUpdate(true);
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method obtains the current position, status of all axes
/*!
    This is a helper function, it is not necessary to implement a function like this, but it might help.
*/
ito::RetVal MyActuator::updateStatus()
{
    for (int i=0;i<m_nrOfAxes;i++)
    {
        m_currentStatus[i] = m_currentStatus[i] | ito::actuatorAvailable; //set this if the axis i is available, else use
        //m_currentStatus[i] = m_currentStatus[i] ^ ito::actuatorAvailable;

        m_currentPos[i] = 0.0; //todo fill in here the current position of axis i in mm or degree

        //if you know that the axis i is at its target position, change from moving to target if moving has been set, therefore:
        replaceStatus(m_currentStatus[i], ito::actuatorMoving, ito::actuatorAtTarget);

        //if you know that the axis i is still moving, set this bit (all other moving-related bits are unchecked, but the status bits and switches bits
        //kept unchanged
        setStatus(m_currentStatus[i], ito::actuatorMoving, ito::actSwitchesMask | ito::actStatusMask);
    }

    //emit actuatorStatusChanged with m_currentStatus and m_currentPos in order to inform connected slots about the current status and position
    sendStatusUpdate();

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot called if the dock widget of the plugin becomes (in)visible
/*!
    Overwrite this method if the plugin has a dock widget. If so, you can connect the parametersChanged signal of the plugin
    with the dock widget once its becomes visible such that no resources are used if the dock widget is not visible. Right after
    a re-connection emit parametersChanged(m_params) in order to send the current status of all plugin parameters to the dock widget.
*/
void MyActuator::dockWidgetVisibilityChanged(bool visible)
{
    if (getDockWidget())
    {
        QWidget *widget = getDockWidget()->widget();
        if (visible)
        {
            connect(this, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), widget, SLOT(parametersChanged(QMap<QString, ito::Param>)));
            connect(this, SIGNAL(actuatorStatusChanged(QVector<int>, QVector<double>)), widget, SLOT(actuatorStatusChanged(QVector<int>, QVector<double>)));
            connect(this, SIGNAL(targetChanged(QVector<double>)), widget, SLOT(targetChanged(QVector<double>)));

            emit parametersChanged(m_params);
            sendTargetUpdate();
            sendStatusUpdate(false);
        }
        else
        {
            disconnect(this, SIGNAL(parametersChanged(QMap<QString, ito::Param>)), widget, SLOT(parametersChanged(QMap<QString, ito::Param>)));
            disconnect(this, SIGNAL(actuatorStatusChanged(QVector<int>, QVector<double>)), widget, SLOT(actuatorStatusChanged(QVector<int>, QVector<double>)));
            disconnect(this, SIGNAL(targetChanged(QVector<double>)), widget, SLOT(targetChanged(QVector<double>)));
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
const ito::RetVal MyActuator::showConfDialog(void)
{
    return apiShowConfigurationDialog(this, new DialogMyActuator(this));
}
