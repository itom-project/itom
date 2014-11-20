/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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


#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI
#include "addInInterface.h"
#include <qdebug.h>
#include <qmetaobject.h>
#include <qcoreapplication.h>
#include "abstractAddInDockWidget.h"

#if defined _DEBUG  && defined(_MSC_VER) && defined(VISUAL_LEAK_DETECTOR_CMAKE)
    #include "vld.h"
#endif

namespace ito
{
    int AddInBase::m_instCounter = 0;

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInInterfaceBase::~AddInInterfaceBase()
    { 
        m_initParamsMand.clear();
        m_initParamsOpt.clear();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInInterfaceBase::closeInst(ito::AddInBase **addInInst) 
    { 
        ito::RetVal ret = closeThisInst(addInInst); 
        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void AddInInterfaceBase::incRef(ito::AddInBase *addIn)
    {
        addIn->incRefCount();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void AddInInterfaceBase::decRef(ito::AddInBase *addIn) 
    { 
        addIn->decRefCount();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    int AddInInterfaceBase::getRef(ito::AddInBase *addIn) 
    { 
        return addIn->getRefCount();
    }

    //! set api function pointer
    void AddInInterfaceBase::setApiFunctions(void **apiFunctions) 
    { 
        m_apiFunctionsBasePtr = apiFunctions;
        this->importItomApi(apiFunctions); //this is the virtual function call in order to also propagate the api pointer to the plugin dll
        ito::ITOM_API_FUNCS = apiFunctions; //this propagates the api pointer to the itomCommonQt dll where this source file has been compiled
    }

    void AddInInterfaceBase::setApiFunctionsGraph(void ** apiFunctionsGraph) 
    { 
        m_apiFunctionsGraphBasePtr = apiFunctionsGraph;
        this->importItomApiGraph(apiFunctionsGraph); //this is the virtual function call in order to also propagate the api pointer to the plugin dll
        ito::ITOM_API_FUNCS_GRAPH = apiFunctionsGraph; //this propagates the api pointer to the itomCommonQt dll where this source file has been compiled
    }

    bool AddInInterfaceBase::event(QEvent *e)
    {
        //the event User+123 is emitted by AddInManager, if the API has been prepared and can
        //transmitted to the plugin. This assignment cannot be done directly, since 
        //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
        //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
        //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
        //also is necessary if any methods of the plugin are directly called from itom).
        if (e->type() == (QEvent::User+123))
        {
            this->importItomApi(m_apiFunctionsBasePtr);
            this->importItomApiGraph(m_apiFunctionsGraphBasePtr);
        }   
        return QObject::event(e);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! Constructor.
    /*!
        This constructor is called by any constructor of classes AddInActuator, AddInDataIO or AddInAlgo.
        Please make sure to call these constructors in your plugin constructor, since this is not automatically be done (due to the single
        parameter of the constructor).

        This constructor simply initializes several member variables of this class.

        \param [in] uniqueID is the unique identifier of this plugin instance. This identifier can be changed in the constructor or
                    finally at the beginning of in the init-method. Afterwards it is used by different organizers and GUI components.
    */
    AddInBase::AddInBase() :
        m_pThread(NULL), 
        m_pBasePlugin(NULL), 
        m_uniqueID(++m_instCounter), 
        m_refCount(0), 
        m_createdByGUI(0),
        m_dockWidget(NULL),
        m_alive(0),
        m_initialized(false)
    {
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! Destructor.
    /*!
        This destructor is automatically called if any plugin instance is destroyed. It does the following steps:

        - Deletes the dock widget if available.
        - Clears the internal parameter-vector.
        - If the plugin instance is executed in its own thread, this thread is stopped and finally deleted.
    */
    AddInBase::~AddInBase() //will be called from main thread
    {
        if (m_dockWidget)
        {
            //the dock widget has not been destroyed yet (by deconstructor of mainWindow, where it is attached)
            m_dockWidget->deleteLater();
            m_dockWidget = NULL;
        }

        m_params.clear();

        //delete own thread if not already happened
        if (m_pThread != NULL)
        {
            m_pThread->quit();
            m_pThread->wait(5000);
            delete m_pThread;
            m_pThread = NULL;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method for setting various parameters in a sequence
    /*!
        Using this method, only one over-thread call needs to be executed in order to set various parameters
        by calling setParam for each parameter.

        \param values is a vector of parameters to set
        \param waitCond is the locked semaphore that is released at the end of the method.
        \sa setParam, ParamBase
    */
    ito::RetVal AddInBase::setParamVector(const QVector<QSharedPointer<ito::ParamBase> > values, ItomSharedSemaphore *waitCond)
    {
        ItomSharedSemaphoreLocker locker(waitCond);

        ito::RetVal retValue = ito::retOk;

        foreach(const QSharedPointer<ito::ParamBase> &param, values)
        {
            retValue += setParam(param,NULL);
            setAlive();
        }

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method for setting various parameters in a sequence
    /*!
        Using this method, only one over-thread call needs to be executed in order to set various parameters
        by calling setParam for each parameter.

        \param values is a vector of parameters to set
        \param waitCond is the locked semaphore that is released at the end of the method.
        \sa setParam, ParamBase
    */
    ito::RetVal AddInBase::getParamVector(const QVector<QSharedPointer<ito::Param> > values, ItomSharedSemaphore *waitCond)
    {
        ItomSharedSemaphoreLocker locker(waitCond);

        ito::RetVal retValue = ito::retOk;

        foreach(const QSharedPointer<ito::Param> &param, values)
        {
            retValue += getParam(param,NULL);
            setAlive();
        }

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! this method can handle additional functions of your plugin.
    /*!
        Use registerExecFunc to register a specific function name and a set of mandatory and optional default parameters.
        It one on those functions is called (for instance by a python-call), this method is executed. Implement a switch
        case for the function name and start the execution. The mandatory and optional parameters are handled like it is
        the case for the creation (init-method) of a plugin. Additionally define an optional vector of output parameters,
        that are finally filled with a valid input during the execution of the function.

        \param funcName is the function name
        \param paramsMand is the vector of mandatory parameters for the specific function name
        \param paramsOpt is the vector of optional parameters for the specific function name
        \param paramsOut is the vector of parameters (must have flag OUT, not IN), that are the return value(s) of the specific function call
        \param waitCond is the semaphore in order guarantee, that the caller of this method waits until the function has been executed.
        \sa registerExecFunc, init
    */
    ito::RetVal AddInBase::execFunc(const QString /*funcName*/, QSharedPointer<QVector<ito::ParamBase> > /*paramsMand*/, QSharedPointer<QVector<ito::ParamBase> > /*paramsOpt*/, QSharedPointer<QVector<ito::ParamBase> > /*paramsOut*/, ItomSharedSemaphore *waitCond)
    {
        ItomSharedSemaphoreLocker locker(waitCond);

        ito::RetVal retValue = ito::RetVal(ito::retError, 0, tr("function execution unused in this plugin").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! Creates the dock-widget for this plugin
    /*
        Call this method ONLY in the constructor of your plugin, since it must be executed in the main thread.

        By this method, the dock-widget for this plugin is created, where you can define the content-widget of the dock-widget,
        some style-features of the dock-widget, the areas in the main window, where it is allowed to move the dock-widget...

        \param [in] title is the dock-widget's title
        \param [in] features is an OR-combination of QDockWidget::DockWidgetFeature
        \param [in] allowedAreas indicate the allowed areas as OR-combination of Qt::DockWidgetArea
        \param [in] content is the new content-widget for the dock-widget

        \sa dockWidgetDefaultStyle
    */
    void AddInBase::createDockWidget(QString title, QDockWidget::DockWidgetFeatures features, Qt::DockWidgetAreas allowedAreas, QWidget *content)
    {
        if (m_dockWidget == NULL)
        {
            m_dockWidget = new QDockWidget(title + " - " + tr("Toolbox"));
            connect(m_dockWidget, SIGNAL(destroyed()), this, SLOT(dockWidgetDestroyed())); //this signal is established in order to check if the docking widget already has been deleted while destruction of mainWindows
            connect(m_dockWidget, SIGNAL(visibilityChanged(bool)), this, SLOT(dockWidgetVisibilityChanged(bool)));
        }
        m_dockWidget->setObjectName(title.simplified() + "_dockWidget#" + QString::number(m_uniqueID));
        m_dockWidget->setFeatures(features);
        m_dockWidget->setAllowedAreas(allowedAreas);

        if (content) 
        {
            m_dockWidget->setWidget(content);
            content->setParent(m_dockWidget);
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void AddInBase::setIdentifier(const QString &identifier)
    {
        m_identifier = identifier;

        if (m_dockWidget)
        {
            ito::AbstractAddInDockWidget *adw = qobject_cast<ito::AbstractAddInDockWidget*>(m_dockWidget->widget());
            if (adw)
            {
                QMetaObject::invokeMethod(adw, "identifierChanged", Q_ARG(const QString &, identifier));
            }
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! Registers an additional function with specific name and default parameters
    /*
        After having registered the function, the method execFunc can be called with the specific function name
        and a set of parameters, fitting to the given default ones. Then execFunc needs to be implemented, such
        that the approparite function, depending on its name, is executed.

        \param [in] funcName is the unique name of the additional function
        \param [in] paramsMand is the vector with default mandatory parameters (must all have a set IN-flag)
        \param [in] paramsOpt is the vector with default optional parameters (must all have a set IN-flag)
        \param [in] paramsOut is the vector with default output parameters (must only have the OUT-flag, no IN-flag)
        \param [in] infoString is a description of this additional function

        \sa execFunc
    */
    ito::RetVal AddInBase::registerExecFunc(const QString funcName, const QVector<ito::Param> &paramsMand, const QVector<ito::Param> &paramsOpt, const QVector<ito::Param> &paramsOut, const QString infoString)
    {
        QMap<QString, ExecFuncParams>::const_iterator it = m_execFuncList.constFind(funcName);
        ito::RetVal retValue = ito::retOk;

        if (it == m_execFuncList.constEnd())
        {
#ifndef NDEBUG
            //check flags of paramsMand, paramsOpt and paramsOut
            foreach(const ito::Param &p, paramsMand)
            {
                //mandatory parameters can be of every type, but their flags must be In Or In|Out (NOT Out)
                if ((p.getFlags() & ito::ParamBase::Out) && !(p.getFlags() & ito::ParamBase::In))
                {
                    QString err = QString("Mandatory parameter '%1' cannot be defined as Out-Parameter").arg(p.getName());
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    //throw std::logic_error(err.toLatin1().data());
                    break;
                }
            }
            foreach(const ito::Param &p, paramsOpt)
            {
                //optional parameters can be of every type, but their flags must be In Or In|Out (NOT Out)
                if ((p.getFlags() & ito::ParamBase::Out) && !(p.getFlags() & ito::ParamBase::In))
                {
                    QString err = QString("Optional parameter '%1' cannot be defined as Out-Parameter").arg(p.getName());
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    //throw std::logic_error(err.toLatin1().data());
                    break;
                }
            }
            foreach(const ito::Param &p, paramsOut)
            {
                //output parameters must have flag Out and not In, only types Int(Array),Char(Array),Double(Array) or String are allowed
                if ((p.getFlags() & ito::ParamBase::In) || !(p.getFlags() & ito::ParamBase::Out))
                {
                    QString err = QString("Output parameter '%1' must be defined as Out-Parameter").arg(p.getName());
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    //throw std::logic_error(err.toLatin1().data());
                    break;
                }
                if ((p.getType() & (ito::ParamBase::Int | ito::ParamBase::Char | ito::ParamBase::Double)) == 0)
                {
                    QString err = QString("Output parameter '%1' must be of type Int(-Array), Char(-Array), Double(-Array) or String.").arg(p.getName());
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    //throw std::logic_error(err.toLatin1().data());
                    break;
                }
            }
#endif
            if (!retValue.containsError())
            {
                ExecFuncParams newParam;
                newParam.paramsMand = paramsMand; //implicitly shared (see Qt-doc QVector(const QVector<T> & other))
                newParam.paramsOpt  = paramsOpt;  //implicitly shared (see Qt-doc QVector(const QVector<T> & other))
                newParam.paramsOut  = paramsOut;  //implicitly shared (see Qt-doc QVector(const QVector<T> & other))
                newParam.infoString = infoString;
                m_execFuncList[funcName] = newParam;
            }
        }
        else
        {
            retValue += ito::RetVal(ito::retError,0,"function with this name is already registered.");
        }

        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns default style properties for dock-widget of plugin
    /*
        This method is called by the AddInManager at initialization of a plugin instance. Then,
        the AddInManager gets informed about the default behaviour of the dock widget. Overwrite this
        method if you want to create your own default behaviour.

        \param [out] floating defines whether the dock widget should per default be shown in floating mode (true) or docked (false, default)
        \param [out] visible defines whether the dock widget should be visible at startup of plugin (true) or not (false, default)
        \param [out] defaultArea defines the default area in the main window, where the dock widget is shown
        \sa AddInManager::initDockWidget
    */
     void AddInBase::dockWidgetDefaultStyle(bool &floating, bool &visible, Qt::DockWidgetArea &defaultArea) const
     {
         if (m_dockWidget)
         {
             floating = false;
             visible = false;
             defaultArea = Qt::RightDockWidgetArea;
         }
         else
         {
             floating = false;
             visible = false;
             defaultArea = Qt::NoDockWidgetArea;
         }
     }


    //----------------------------------------------------------------------------------------------------------------------------------
    //! method indicates whether this plugin instance has a configuration dialog.
    /*!
        Overwrite this method if your plugin provides such a configuration dialog by simply returning 1 instead of 0.

        \return 0 since the base implementation of a plugin does not have a configuration dialog. If there is a configuration dialog
                overwrite this method and return 1.
    */
    int AddInBase::hasConfDialog(void)
    {
        return 0;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method called if the configuration dialog of the plugin should be shown.
    /*!
        Overwrite this method if your plugin provides a configuration dialog. This method is directly called by the main (GUI) thread.
        Therefore you can directly show your configuration dialog, connect it with signals and slots of your plugin instance (which possibly is executed
        in another thread), wait for any commands or that the dialog is closed and handle the result.

        \return retWarning since you have to overwrite this method in order to show your configuration dialog.
    */
    const ito::RetVal AddInBase::showConfDialog(void)
    {
        return ito::RetVal(ito::retWarning,0, tr("Your plugin is supposed to have a configuration dialog, but you did not implement the showConfDialog-method").toLatin1().data());
    }



    //----------------------------------------------------------------------------------------------------------------------------------
    AddInDataIO::AddInDataIO() : 
        AddInBase(),
        m_timerID(0),
        m_timerIntervalMS(20),
        m_autoGrabbingEnabled(true)
    {
        qDebug() << "AddInDataIO constructor. ThreadID: " << QThread::currentThreadId();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInDataIO::~AddInDataIO()
    {
        if (m_timerID > 0)
        {
            killTimer(m_timerID);
            m_timerID = 0;
        }
    }


    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::startDeviceAndRegisterListener(QObject* obj, ItomSharedSemaphore *waitCond)
    {
        qDebug("begin: startDeviceAndRegisterListener");
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retValue(ito::retOk);

        if (obj->metaObject()->indexOfSlot(QMetaObject::normalizedSignature("setSource(QSharedPointer<ito::DataObject>,ItomSharedSemaphore*)")) == -1)
        {
            retValue += ito::RetVal(ito::retError, 2002, tr("listener does not have a slot ").toLatin1().data());
        }
        else if (m_autoGrabbingListeners.contains(obj))
        {
            retValue += ito::RetVal(ito::retWarning, 1011, tr("this object already has been registered as listener").toLatin1().data());
        }
        else
        {
            retValue += startDevice(NULL);

            if (m_autoGrabbingEnabled == true && m_autoGrabbingListeners.size() >= 0 && m_timerID == 0)
            {
                m_timerID = startTimer(m_timerIntervalMS);

                if (m_timerID == 0)
                {
                    retValue += ito::RetVal(ito::retError, 2001, tr("timer could not be set").toLatin1().data());
                }
            }

            m_autoGrabbingListeners.insert(obj);
        }

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        qDebug("end: startDeviceAndRegisterListener");
        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::stopDeviceAndUnregisterListener(QObject* obj, ItomSharedSemaphore *waitCond)
    {
        qDebug("start: stopDeviceAndUnregisterListener");
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retValue(ito::retOk);

        if (!m_autoGrabbingListeners.remove(obj))
        {
            retValue += ito::RetVal(ito::retWarning, 1012, tr("the object could not been removed from the listener list").toLatin1().data());
        }
        else
        {
            qDebug("live image has been removed from listener list");
        }

        if (m_autoGrabbingListeners.size() <= 0)
        {
            if (m_timerID) //stop timer if no other listeners are registered
            {
                killTimer(m_timerID);
                m_timerID = 0;
            }

            retValue += stopDevice(NULL);
        }

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        qDebug("end: stopDeviceAndUnregisterListener");
        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::disableAutoGrabbing(ItomSharedSemaphore *waitCond)
    {
        m_autoGrabbingEnabled = false;

        if (m_timerID)
        {
            killTimer(m_timerID);
            m_timerID = 0;
        }

        if (waitCond)
        {
            waitCond->release();
            waitCond->deleteSemaphore();
            waitCond = NULL;
        }

        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::enableAutoGrabbing(ItomSharedSemaphore *waitCond)
    {
        m_autoGrabbingEnabled = true;

        if (m_autoGrabbingListeners.size() > 0 && m_timerID == 0)
        {
            m_timerID = startTimer(m_timerIntervalMS);
        }

        if (waitCond)
        {
            waitCond->release();
            waitCond->deleteSemaphore();
            waitCond = NULL;
        }

        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void AddInDataIO::runStatusChanged(bool deviceStarted)
    {
        if (deviceStarted && m_autoGrabbingEnabled) 
        {
            //auto grabbing flag is set, the device has probably been stopped as is now restarted -> restart auto-grabbing timer as well
            if (m_autoGrabbingListeners.size() > 0 && m_timerID == 0)
            {
                m_timerID = startTimer(m_timerIntervalMS);
            }
        }
        else 
        {
            //device is stopped -> also stop the live grabbing timer, if set. The auto grabbing flag is not changed by this method
            if (m_timerID)
            {
                killTimer(m_timerID);
                m_timerID = 0;
            }
        }
    }


    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::startDevice(ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::startDevice", tr("not implemented").toLatin1().data());

        ItomSharedSemaphoreLocker locker(waitCond);

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method startDevice() is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::stopDevice(ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::stopDevice", tr("not implemented").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method stopDevice() is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::acquire(const int /*trigger*/, ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::acquire", tr("not implemented").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method acquire() is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::getVal(void * /*data*/, ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::getVal(ito::RetVal, void *data, ItomSharedSemaphore *waitCond)", tr("not implemented").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method getVal(void*, ItomSharedSemaphore*) is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::getVal(QSharedPointer<char> /*data*/, QSharedPointer<int> /*length*/, ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::getVal(ito::RetVal, QSharedPointer<char> data, QSharedPointer<int> length, ItomSharedSemaphore *waitCond)", tr("not implemented").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method getVal(QSharedPointer<char>, QSharedPointer<int>, ItomSharedSemaphore*) is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::copyVal(void * /*data*/, ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::copyVal(void *data, ItomSharedSemaphore *waitCond)", tr("not implemented").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method copyVal(void*,ItomSharedSemaphore*) is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInDataIO::setVal(const char * /*data*/, const int /*length*/, ItomSharedSemaphore *waitCond)
    {
        Q_ASSERT_X(1, "AddInDataIO::setVal(const char *data, const int length, ItomSharedSemaphore *waitCond)", tr("not implemented").toLatin1().data());

        if (waitCond)
        {
            waitCond->returnValue += ito::RetVal(ito::retError,0,"method setVal(const char*, const int, ItomSharedSemaphore*) is not implemented in this plugin");
            waitCond->release();
            
            return waitCond->returnValue;
        }
        else
        {
            return ito::retError;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInActuator::AddInActuator() 
        : AddInBase(), 
        m_nrOfStatusChangedConnections(0), 
        m_nrOfTargetChangedConnections(0), 
        m_interruptFlag(false)
    {
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInActuator::~AddInActuator()
    {
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! overwritten method from QObject in order to get informed about connected signal-slot connections.
    /*!
        This method is called if any signal-slot connection is established. If it is connected to the signal actuatorStatusChanged, the
        counter m_nrOfStatusChangedConnections is incremented (in order to have the number of connections). The same holds for the signal 
        targetChanged and counter variable m_nrOfTargetChangedConnections.

        If you want to overwrite this method in your actuator implementation, please call the implementation of AddInActuator in your implementation
        such that the functionality described above is still working.

        \param [in] signal is the normalized connection string.
    */
    void AddInActuator::connectNotify(const char * signal)
    {
        if (QLatin1String(signal) == SIGNAL(actuatorStatusChanged(QVector<int>,QVector<double>)))
        {
            m_nrOfStatusChangedConnections++;
        }
        else if (QLatin1String(signal) == SIGNAL(targetChanged(QVector<double>)))
        {
            m_nrOfTargetChangedConnections++;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! overwritten method from QObject in order to get informed about disconnected signal-slot connections.
    /*!
        This method is called if any signal-slot connection to this instance is destroyed. If it is disconnected from the signal actuatorStatusChanged, the
        counter m_nrOfStatusChangedConnections is decremented (in order to have the right number of connections). The same holds for the signal 
        targetChanged and counter variable m_nrOfTargetChangedConnections.

        If you want to overwrite this method in your actuator implementation, please call the implementation of AddInActuator in your implementation
        such that the functionality described above is still working.

        \param [in] signal is the normalized signal string of the connection which is destroyed.
    */
    void AddInActuator::disconnectNotify(const char * signal)
    {
        if (QLatin1String(signal) == SIGNAL(actuatorStatusChanged(QVector<int>,QVector<double>)))
        {
            m_nrOfStatusChangedConnections--;
        }
        else if (QLatin1String(signal) == SIGNAL(targetChanged(QVector<double>)))
        {
            m_nrOfTargetChangedConnections--;
        }
    }
    

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method emits the actuatorStatusChanged signal if any slot is connected to this signal.
    /*!
        The emitted values are the member variables m_currentStatus and m_currentPos (optional).

        \param [in] statusOnly indicates whether the status only should be emitted or the current position vector, too. In case of status only, the
                current position vector is empty, hence has a length of zero. This should be considered by the slot.
    */
    void AddInActuator::sendStatusUpdate(const bool statusOnly)
    {
        if (m_nrOfStatusChangedConnections>0)
        {
            if (statusOnly)
            {
                emit actuatorStatusChanged(m_currentStatus, QVector<double>());
            }
            else
            {
                emit actuatorStatusChanged(m_currentStatus, m_currentPos);
            }
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method emits the targetChanged signal if any slot is connected to this signal.
    /*!
        The emitted values is the member variable m_targetPos
    */
    void AddInActuator::sendTargetUpdate()
    {
        if (m_nrOfTargetChangedConnections>0)
        {
            emit targetChanged(m_targetPos);
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method invoked in order to force a re-emitation of the current status, the current positions (if desired) and the target positions (if desired)
    /*!
        This method is mainly invoked by a dock widget of the actuator such that the plugin re-emits the current values, that are then
        received by the dock widget.

        Overload this method if you want to update the values before emitting them.
    */
    ito::RetVal AddInActuator::requestStatusAndPosition(bool sendActPosition, bool sendTargetPos)
    {
        ito::RetVal retval;

        //in your real motor, overload this function and update m_currentStatus, m_currentPos and/or m_targetPos
        //before emitting them using the methods above

        sendStatusUpdate(!sendActPosition);

        if (sendTargetPos)
        {
            sendTargetUpdate();
        }

        return retval;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInAlgo::AddInAlgo() : AddInBase()
    {
        Q_ASSERT_X(1, "AddInAlgo::AddInAlgo", tr("Constructor must be overwritten").toLatin1().data());
        return;
    }

   //----------------------------------------------------------------------------------------------------------------------------------
    AddInAlgo::~AddInAlgo()
    {
        FilterDef *filter;
        foreach(filter, m_filterList)
        {
            delete filter;
        }
        m_filterList.clear();

        AlgoWidgetDef *algoWidget;
        foreach(algoWidget, m_algoWidgetList)
        {
            delete algoWidget;
        }
        m_algoWidgetList.clear();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInAlgo::getFilterList(QHash<QString, FilterDef *> &fList) const
    {
        fList = m_filterList;
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInAlgo::getAlgoWidgetList(QHash<QString, AlgoWidgetDef *> &awList) const
    {
        awList = m_algoWidgetList;
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInAlgo::rejectFilter(const QString &name)
    {
        QHash<QString, FilterDef *>::iterator it = m_filterList.find(name);
        if (it != m_filterList.end())
        {
            delete *it;
            m_filterList.erase(it);
            return ito::retOk;
        }
        return ito::retError;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal AddInAlgo::rejectAlgoWidget(const QString &name)
    {
        QHash<QString, AlgoWidgetDef *>::iterator it = m_algoWidgetList.find(name);
        if (it != m_algoWidgetList.end())
        {
            delete *it;
            m_algoWidgetList.erase(it);
            return ito::retOk;
        }
        return ito::retError;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito
