/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "pluginThreadCtrl.h"

#include <qelapsedtimer.h>
#include <qvector.h>
#include <qapplication.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
PluginThreadCtrl::PluginThreadCtrl() :
    m_pPlugin(NULL)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \return (void)
    \sa CameraThreadCtrl
*/
PluginThreadCtrl::PluginThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval /*= NULL*/) :
    m_pPlugin(NULL)
{
    void *plugin = pluginParameter.getVal<void *>();

    if (plugin && (reinterpret_cast<ito::AddInBase *>(plugin)->getBasePlugin()->getType() & (ito::typeDataIO | ito::typeActuator)))
    {
        m_pPlugin = (ito::AddInBase *)(plugin);

        m_pPlugin->getBasePlugin()->incRef(m_pPlugin); //increment reference, such that camera is not deleted
    }

    if (m_pPlugin == NULL)
    {
        if (retval)
        {
            (*retval) += ito::RetVal(ito::retError, 0, QObject::tr("No or invalid plugin given.").toLatin1().data());
        }
        return;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PluginThreadCtrl::PluginThreadCtrl(ito::AddInBase *plugin, ito::RetVal *retval /*= NULL*/) :
    m_pPlugin(plugin)
{
    if (m_pPlugin == NULL)
    {
        if (retval)
        {
            (*retval) += ito::RetVal(ito::retError, 0, QObject::tr("No or invalid plugin given").toLatin1().data());
        }
        return;
    }
    else
    {
        m_pPlugin->getBasePlugin()->incRef(m_pPlugin); //increment reference, such that camera is not deleted
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail The destructor
*/
PluginThreadCtrl::~PluginThreadCtrl()
{
    if (m_pPlugin)
    {
        m_pPlugin->getBasePlugin()->decRef(m_pPlugin); //decrement reference
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PluginThreadCtrl::PluginThreadCtrl(const PluginThreadCtrl &other) :
    m_pPlugin(other.m_pPlugin)
{
    if (m_pPlugin)
    {
        m_pPlugin->getBasePlugin()->incRef(m_pPlugin); //increment reference, such that camera is not deleted
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PluginThreadCtrl& PluginThreadCtrl::operator =(const PluginThreadCtrl &other)
{
    ito::AddInBase *op = other.m_pPlugin;
    if (op)
    {
        op->getBasePlugin()->incRef(op); //increment reference, such that camera is not deleted
    }

    if (m_pPlugin)
    {
        m_pPlugin->getBasePlugin()->decRef(m_pPlugin); //decrement reference
        m_pPlugin = NULL;
    }

    m_pPlugin = op;

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail After the invoke-command this thread must wait / be synchronize with the plugin-thread.
        Therefore the wait-Function of m_semaphoreLocker is called. If the plugin do not answer within timeOutMS and the pMyCamera is not alive anymore, the function returns a timeout.

    \param [in] timeOutMS    timout for the wait. -1: endless wait until the plugin finished the last invokation or it is not 'alive' anymore, 0: no wait (this method does nothing), >0: time in ms

    \return retOk if semaphore was successfully released, retError if semaphore returned an error message or if a timeout occurred (error code: 256)
*/
//-----------------------------------------------------------------------------------------------
ito::RetVal PluginThreadCtrl::waitForSemaphore(int timeOutMS /*= PLUGINWAIT*/)
{
    ito::RetVal retval;

    if (m_semaphoreLocker.getSemaphore() != NULL && timeOutMS != 0)
    {
        ItomSharedSemaphore *waitCond = m_semaphoreLocker.getSemaphore();
        bool timeout = false;

        if (timeOutMS < 0) //endless wait until it is done or not alive anymore
        {
            while(waitCond->wait(PLUGINWAIT) == false)
            {
                if (m_pPlugin->isAlive() == false)
                {
                    retval += ito::RetVal(ito::retError, 256, QObject::tr("Timeout while waiting for answer from camera.").toLatin1().data());
                    timeout = true;
                    break;
                }
            }
        }
        else
        {
            QElapsedTimer timer;
            timer.start();
            int t = std::min(timeOutMS, PLUGINWAIT);

            while(!timeout && waitCond->wait(t) == false)
            {
                if (timer.elapsed() > timeOutMS)
                {
                    retval += ito::RetVal(ito::retError, 256, QObject::tr("Timeout while waiting for answer from camera.").toLatin1().data());
                    timeout = true;
                }
                else if (m_pPlugin->isAlive() == false)
                {
                    retval += ito::RetVal(ito::retError, 256, QObject::tr("Timeout while waiting for answer from camera.").toLatin1().data());
                    timeout = true;
                }
            }
        }

        if (!timeout)
        {
            retval += waitCond->returnValue;
            m_semaphoreLocker = NULL; //delete semaphore in locker
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Get any parameter of the camera defined by val.name. val must be initialised and name must be correct. After correct execution, val has the correct value.

    \param [in|out] val      Initialised tParam (correct name | in)
    \param [in] timeOutMS    TimeOut for the semaphore-wait

    \return retOk or retError
*/
ito::RetVal PluginThreadCtrl::getParam(ito::Param &val, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    ito::RetVal retval(ito::retOk);
    QSharedPointer<ito::Param> qsParam(new ito::Param(val));
    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "getParam", conType, Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking getParam").toLatin1().data());
    }

    retval = waitForSemaphore(timeOutMS);

    if (!retval.containsError() && timeOutMS != 0)
    {
        val = *qsParam;
    }
    else if (timeOutMS == 0)
    {
        retval += ito::RetVal(ito::retWarning, 0, QObject::tr("No parameter can be returned if timeout = 0").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Get the parameter of the plugin defined by val.name to the value of val.

    \param [in] val         Initialised tParam (correct name | value)
    \param [in] timeOutMS   TimeOut for the semaphore-wait

    \return retOk or retError
*/
ito::RetVal PluginThreadCtrl::setParam(ito::ParamBase val, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("no camera available").toLatin1().data());
    }

    QSharedPointer<ito::ParamBase> qsParam(new ito::ParamBase(val));
    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "setParam", conType, Q_ARG(QSharedPointer<ito::ParamBase>, qsParam), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("error invoking setParam").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);
}

//----------------------------------------------------------------------------------------------------------------------------------
DataIOThreadCtrl::DataIOThreadCtrl() :
    PluginThreadCtrl()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
DataIOThreadCtrl::DataIOThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval /*= NULL*/) :
    PluginThreadCtrl(pluginParameter, retval)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
DataIOThreadCtrl::DataIOThreadCtrl(ito::AddInDataIO *plugin, ito::RetVal *retval /*= NULL*/) :
    PluginThreadCtrl(plugin, retval)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
DataIOThreadCtrl::DataIOThreadCtrl(const DataIOThreadCtrl &other) :
    PluginThreadCtrl(other)
{

}

//----------------------------------------------------------------------------------------------------------------------------------
DataIOThreadCtrl::~DataIOThreadCtrl()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Every capture procedure starts with the startDevice() to set the camera active and is ended with stopDevice().

    \param [in] timeOutMS    TimeOut for the semaphore-wait

    \return retOk or retError
    \sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::acquire, DataIOThreadCtrl::getVal, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::startDevice(int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("no camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "startDevice", conType, Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking startDevice").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Every capture procedure starts with the startDevice() to set the camera active and is ended with stopDevice().

    \param [in] timeOutMS    TimeOut for the semaphore-wait

    \return retOk or retError
    \sa DataIOThreadCtrl::startDevice, DataIOThreadCtrl::acquire, DataIOThreadCtrl::getVal, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::stopDevice(int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "stopDevice", conType, Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking stopDevice").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail The acquire()-function triggers a new exposure of the camera and returns afterwards. It can only be executed after startDevice().
            The function does not wait until the exposure is done. This is performed by the getVal or copyVal-method.

    \param [in] trigger     A currently not implemented constant to define trigger-modes during exposure of the camera
    \param [in] timeOutMS   TimeOut for the semaphore-wait

    \return retOk or retError
    \sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::startDevice, DataIOThreadCtrl::getVal, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::acquire(const int trigger /*= 0*/, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("no camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "acquire", conType, Q_ARG(int, trigger), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking acquire").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
\detail The getAutoGrabbing function is used to get the auto grabbing setting.

\return retOk or retError
\sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::acquire, DataIOThreadCtrl::startDevice, DataIOThreadCtrl::copyVal
*/
bool DataIOThreadCtrl::getAutoGrabbing()
{
    if (!m_pPlugin)
    {
        return false;
    }
    else
    {
        return static_cast<ito::AddInDataIO*>(m_pPlugin)->getAutoGrabbing();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
\detail Enables the timer for auto grabbing (live image), if any live image has signed on.

\param [in] timeOutMS   TimeOut for the semaphore-wait

\return retOk or retError
\sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::startDevice, DataIOThreadCtrl::getVal, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::enableAutoGrabbing(int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "enableAutoGrabbing", conType, Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking enableAutoGrabbing").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
\detail Disables the timer for auto grabbing (live image).

\param [in] timeOutMS   TimeOut for the semaphore-wait

\return retOk or retError
\sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::startDevice, DataIOThreadCtrl::getVal, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::disableAutoGrabbing(int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "disableAutoGrabbing", conType, Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking disableAutoGrabbing").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
\detail Sets a new interval for the auto-grabbing timer (in ms). If interval <= 0 is passed, nothing is changed,
        but the current interval is returned. This method does not enable or disable the timer.

\param [in] interval    Timer (in ms)
\param [in] timeOutMS   TimeOut for the semaphore-wait

\return retOk or retError
\sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::startDevice, DataIOThreadCtrl::getVal, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::setAutoGrabbingInterval(QSharedPointer<int> interval, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "setAutoGrabbingInterval", conType, Q_ARG(QSharedPointer<int>, interval), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking setAutoGrabbingInterval").toLatin1().data());
    }
    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail The getVal function is used to wait until an exposure is finished. Than it gives a shallow copy of the inner dataObject within the grabber to the dObj-argument.
            Before the getVal()-function can be used an acquire() is neccessary.
            If the content of dObj is not deepcopied to another object, the data is lost after the next acquire() - getVal() combination and overwritten by the newly captured image.

    \param [in|out] dObj    IN: an dataObject | OUT: an dataObject containing an shallow copy of the last captured image
    \param [in] timeOutMS   TimeOut for the semaphore-wait

    \return retOk or retError
    \sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::acquire, DataIOThreadCtrl::startDevice, DataIOThreadCtrl::copyVal
*/
ito::RetVal DataIOThreadCtrl::getVal(ito::DataObject &dObj, int timeOutMS /*= PLUGINWAIT*/)   /*! < Get a shallow-copy of the dataObject */
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "getVal", conType, Q_ARG(void*, (void *)&dObj), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking getVal").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail The copyVal function is used to wait until an exposure is finished. Than it gives a deep copy of the inner dataObject within the grabber to the dObj-argument.
            Before the copyVal()-function can be used an acquire() is neccessary.
            If the content of dObj do not need to be deepcopied to another object and will not be overwritten after the next acquire() - getVal() combination.

    \param [in|out] dObj    IN: an dataObject | OUT: an dataObject containing an shallow copy of the last captured image
    \param [in] timeOutMS   TimeOut for the semaphore-wait

    \return retOk or retError
    \sa DataIOThreadCtrl::stopDevice, DataIOThreadCtrl::acquire, DataIOThreadCtrl::getVal, DataIOThreadCtrl::startDevice
*/
ito::RetVal DataIOThreadCtrl::copyVal(ito::DataObject &dObj, int timeOutMS /*= PLUGINWAIT*/)  /*! < Get a deep-copy of the dataObject */
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "copyVal", conType, Q_ARG(void*, (void *)&dObj), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking copyVal").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Get the most important parameter of the camera.

    \param [out] bpp        Number of Bits this camera grabs
    \param [out] xsize      Size of the camera in x (cols)
    \param [out] ysize      Size of the camera in y (rows)

    \return retOk or retError
*/

ito::RetVal DataIOThreadCtrl::getImageParams(int &bpp, int &sizex, int &sizey, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No camera available").toLatin1().data());
    }

    QSharedPointer<ito::Param> param1(new ito::Param("bpp", ito::ParamBase::Int));
    QSharedPointer<ito::Param> param2(new ito::Param("sizex", ito::ParamBase::Int));
    QSharedPointer<ito::Param> param3(new ito::Param("sizey", ito::ParamBase::Int));
    QVector<QSharedPointer<ito::Param> > params;
    params << param1 << param2 << param3;
    ito::RetVal retval(ito::retOk);
    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "getParamVector", conType, Q_ARG(QVector<QSharedPointer<ito::Param> >, params), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking copyVal").toLatin1().data());
    }

    retval = waitForSemaphore(timeOutMS);

    if (!retval.containsError() && timeOutMS != 0)
    {
        bpp = params[0]->getVal<int>();
        sizex = params[1]->getVal<int>();
        sizey = params[2]->getVal<int>();
    }
    else if (timeOutMS == 0)
    {
        retval += ito::RetVal(ito::retWarning, 0, QObject::tr("No image parameters can be returned if timeout = 0").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
ActuatorThreadCtrl::ActuatorThreadCtrl() :
    PluginThreadCtrl(),
    m_numAxes(-1)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ActuatorThreadCtrl::ActuatorThreadCtrl(const ito::ParamBase &pluginParameter, ito::RetVal *retval /*= NULL*/) :
    PluginThreadCtrl(pluginParameter, retval),
    m_numAxes(-1)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ActuatorThreadCtrl::ActuatorThreadCtrl(ito::AddInActuator *plugin, ito::RetVal *retval /*= NULL*/) :
    PluginThreadCtrl(plugin, retval),
    m_numAxes(-1)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ActuatorThreadCtrl::ActuatorThreadCtrl(const ActuatorThreadCtrl &other) :
    PluginThreadCtrl(other)
{
    m_numAxes = other.m_numAxes;
}

//----------------------------------------------------------------------------------------------------------------------------------
ActuatorThreadCtrl::~ActuatorThreadCtrl()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Move the axis in axisVec with a distance defined in stepSizeVec relative to current position.
            The axisVec and stepSizeVec must be same size. After the invoke-command this thread must wait / synchronize with the actuator-thread.
            Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
            To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
            The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

    \param [in] axisVec         Vector with the axis to move
    \param [in] stepSizeVec     Vector with the distances for every axis
    \param [in] timeOutMS       TimeOut for the semaphore-wait, if (0) the waitForSemaphore is not called and must be called seperate by the algorithm

    \return retOk or retError
    \sa ActuatorThreadCtrl::setPosAbs
*/
ito::RetVal ActuatorThreadCtrl::setPosRel(const QVector<int> &axes, const QVector<double> &relPositions, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No actuator available").toLatin1().data());
    }

    if (relPositions.size() != axes.size())
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error during setPosRel: Vectors differ in size").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "setPosRel", conType, Q_ARG(QVector<int>, axes), Q_ARG(QVector<double>, relPositions), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking setPosRel").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Move the axis in axisVec to the positions given in posVec.
            The axisVec and posVec must be same size. After the invoke-command this thread must wait / synchronize with the actuator-thread.
            Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
            To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
            The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

    \param [in] axisVec         Vector with the axis to move
    \param [in] posVec          Vector with the new absolute positions
    \param [in] timeOutMS       TimeOut for the semaphore-wait, if (0) the waitForSemaphore is not called and must be called seperate by the algorithm

    \return retOk or retError
    \sa ActuatorThreadCtrl::setPosRel
*/
ito::RetVal ActuatorThreadCtrl::setPosAbs(const QVector<int> &axes, const QVector<double> &absPositions, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No actuator available").toLatin1().data());
    }

    if (absPositions.size() != axes.size())
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error during setPosAbs: Vectors differ in size").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "setPosAbs", conType, Q_ARG(QVector<int>, axes), Q_ARG(QVector<double>, absPositions), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking setPosAbs").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Move a single axis specified by axis  with a distance defined in stepSize relative to current position. After the invoke-command this thread must wait / synchronize with the actuator-thread.
            Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
            To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
            The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

    \param [in] axis         Number of the axis
    \param [in] stepSize     Distances from current position
    \param [in] timeOutMS       TimeOut for the semaphore-wait, if (0) the waitForSemaphore is not called and must be called seperate by the algorithm

    \return retOk or retError
    \sa ActuatorThreadCtrl::setPosAbs
*/
ito::RetVal ActuatorThreadCtrl::setPosRel(int axis, double relPosition, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No actuator available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "setPosRel", conType, Q_ARG(int, axis), Q_ARG(double, relPosition), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking setPosRel").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Move a single axis specified by axis to the position pos. After the invoke-command this thread must wait / synchronize with the actuator-thread.
            Therefore the semaphore->wait is called via the function threadActuator::waitForSemaphore(timeOutMS)
            To enable the algorithm to process data during movement, the waitForSemaphore(timeOutMS) can be skipped by callWait = false.
            The threadActuator::waitForSemaphore(timeOutMS)-function must than be called by the algorithms afterwards / before the next command is send to the actuator.

    \param [in] axis         Number of the axis
    \param [in] pos          New position of the axis
    \param [in] timeOutMS       TimeOut for the semaphore-wait, if (0) the waitForSemaphore is not called and must be called seperate by the algorithm

    \return retOk or retError
    \sa ActuatorThreadCtrl::setPosRel
*/
ito::RetVal ActuatorThreadCtrl::setPosAbs(int axis, double absPosition, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No actuator available").toLatin1().data());
    }

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "setPosAbs", conType, Q_ARG(int, axis), Q_ARG(double, absPosition), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking setPosAbs").toLatin1().data());
    }

    return waitForSemaphore(timeOutMS);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Get the position of a single axis specified by axis.

    \param [in] axis         Number of the axis
    \param [out] pos          position of the axis
    \param [in] timeOutMS    TimeOut for the semaphore-wait

    \return retOk or retError
*/
ito::RetVal ActuatorThreadCtrl::getPos(int axis, double &position, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No actuator available").toLatin1().data());
    }

    QSharedPointer<double> posSP(new double);
    *posSP = 0.0;
    ito::RetVal retval(ito::retOk);

    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "getPos", conType, Q_ARG(int, (const int)axis), Q_ARG(QSharedPointer<double>, posSP), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking getPos").toLatin1().data());
    }

    retval = waitForSemaphore(timeOutMS);

    if (!retval.containsError() && timeOutMS != 0)
    {
        position = *posSP;
    }
    else if (timeOutMS == 0)
    {
        retval += ito::RetVal(ito::retWarning, 0, QObject::tr("No position value can be returned if timeout = 0").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Get the position of a number of axis specified by axisVec.

    \param [in] axisVec         Number of the axis
    \param [out] posVec         Vecotr with position of the axis
    \param [in] timeOutMS    TimeOut for the semaphore-wait

    \return retOk or retError
*/
ito::RetVal ActuatorThreadCtrl::getPos(QVector<int> axes, QVector<double> &positions, int timeOutMS /*= PLUGINWAIT*/)
{
    if (!m_pPlugin)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("No actuator available").toLatin1().data());
    }

    positions.resize(axes.size());
    QSharedPointer<QVector<double> > posVecSP(new QVector<double>());
    posVecSP->fill(0.0, axes.size());

    ito::RetVal retval(ito::retOk);
    Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;
    m_semaphoreLocker = new ItomSharedSemaphore();
    if (!QMetaObject::invokeMethod(m_pPlugin, "getPos", conType, Q_ARG(QVector<int>, axes), Q_ARG(QSharedPointer<QVector<double> >, posVecSP), Q_ARG(ItomSharedSemaphore*, m_semaphoreLocker.getSemaphore())))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Error invoking getPos").toLatin1().data());
    }

    retval = waitForSemaphore(timeOutMS);

    if (!retval.containsError() && timeOutMS != 0)
    {
        positions = *posVecSP;
    }
    else if (timeOutMS == 0)
    {
        retval += ito::RetVal(ito::retWarning, 0, QObject::tr("No position value(s) can be returned if timeout = 0").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail Check if a specific axis is within the axisSpace of this actuator

    \param [in] axisNum  axis index to be checked

    \return retOk or retError
*/
ito::RetVal ActuatorThreadCtrl::checkAxis(int axisNum)
{
    if (m_numAxes == -1 && m_pPlugin)
    {
        ito::Param p("numaxes");
        ito::RetVal retval2 = getParam(p);
        if (!retval2.containsError())
        {
            m_numAxes = p.getVal<int>();
        }
        else
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("Failed to ask for number of axes of actuator").toLatin1().data());
        }
    }

    if (m_numAxes == -1)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Failed to ask for number of axes of actuator").toLatin1().data());
    }

    if (axisNum < 0 || axisNum >= m_numAxes)
    {
        return ito::retError;
    }
    else
    {
        return ito::retOk;
    }
}


} //end namespace ito
