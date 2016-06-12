/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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


#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI
#include "addInInterface.h"
#include <qdebug.h>
#include <qmetaobject.h>
#include <qcoreapplication.h>
#include <qpointer.h>

#include "abstractAddInDockWidget.h"
#include "opencv/cv.h"

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


    //------------------------------------------------------------------------------------------------------------------------
    class AddInBasePrivate
    {
    public:
        AddInBasePrivate() :
            m_pThread(NULL),
            m_pBasePlugin(NULL),
            m_createdByGUI(0),
            m_initialized(false)
        {}

        QPointer<QDockWidget> m_dockWidget;     //!< safe pointer to dock widget. This pointer is automatically NULL if the dock widget is deleted e.g. by a previous deletion of the main window.
        QThread *m_pThread;                     //!< the instance's thread
        AddInInterfaceBase *m_pBasePlugin;      //!< the AddInInterfaceBase instance of this plugin
        int m_uniqueID;                         //!< uniqueID (automatically given by constructor of AddInBase with auto-incremented value)
        int m_createdByGUI;                     //!< 1 if this instance has firstly been created by GUI, 0: this instance has been created by c++ or python
        bool m_initialized;                     //!< true: init-method has been returned with any RetVal, false (default): init-method has not been finished yet

    };

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
        m_refCount(0), 
        m_alive(0)
    {
        aibp = new AddInBasePrivate();
        aibp->m_uniqueID = (++m_instCounter);
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
        if (aibp->m_dockWidget)
        {
            //the dock widget has not been destroyed yet (by deconstructor of mainWindow, where it is attached)
            aibp->m_dockWidget->deleteLater();
        }

        m_params.clear();

        //delete own thread if not already happened
        if (aibp->m_pThread != NULL)
        {
            aibp->m_pThread->quit();
            aibp->m_pThread->wait(5000);
            DELETE_AND_SET_NULL(aibp->m_pThread);
        }

        delete aibp;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! retrieve the uniqueID of this instance
    int AddInBase::getID() const 
    { 
        return aibp->m_uniqueID; 
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! increments reference counter of this plugin (thread-safe)
    void AddInBase::incRefCount(void)
    {
        m_refCountMutex.lock();
        m_refCount++;
        m_refCountMutex.unlock();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! decrements reference counter of this plugin (thread-safe)
    void AddInBase::decRefCount(void)
    {
        m_refCountMutex.lock();
        m_refCount--;
        m_refCountMutex.unlock();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns true if this instance has firstly been created by the GUI
    int AddInBase::createdByGUI() const 
    { 
        return aibp->m_createdByGUI;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method to set whether this instance has been firstly created by the GUI (true) or by any other component (Python, C++, other plugin,..) (false)
    void AddInBase::setCreatedByGUI(int value) 
    { 
        aibp->m_createdByGUI = value;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns in a thread-safe way the status of the m_initialized-member variable. This variable should be set to true at the end of the init-method.
    bool AddInBase::isInitialized(void) const
    {
        //QMutexLocker locker(&m_atomicMutex);
        return aibp->m_initialized;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! sets in a thread-safe way the status of the m_initialized-member
    /*
    \param [in] initialized is the value to set
    */
    void AddInBase::setInitialized(bool initialized)
    {
        //QMutexLocker locker(&m_atomicMutex);
        aibp->m_initialized = initialized;
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
    //! creates new thread for the class instance and moves this instance to the new thread
    ito::RetVal AddInBase::MoveToThread(void)
    {
        aibp->m_pThread = new QThread();
        moveToThread(aibp->m_pThread);
        aibp->m_pThread->start();

		/*set new seed for random generator of OpenCV. 
		This is required to have real random values for any randn or randu command.
		The seed must be set in every thread. This is for the main thread.
		*/
		cv::theRNG().state = (uint64)cv::getCPUTickCount();
		/*seed is set*/

        return retOk;
    }

    //! method to retrieve a parameter from the parameter map (m_params)
    /*!
        returns parameter from m_params vector. If the parameter could not be found or if the given name is invalid an invalid Param is returned.
        If you provide the nameCheckOk-pointer, you will return a boolean value describing whether your name matched the possible regular expression.

        The parameter name, that is search can have the following form:

        - Name (where Name consists of numbers, characters (a-z) or the symbols _-)
        - Name[Idx] (where Idx is a fixed-point number
        - Name[Idx]:suffix (where suffix is any string - suffix is ignored by this method)
        - Name:suffix

        \warn until now, the Idx is ignored by this method.

        \param name is the name of the parameter
        \param nameCheckOk returns true if name corresponds to the necessary syntax, else false
        \return Param as copy of the internal m_params-map or empty Param, if name could not be resolved or found
    */
    const Param AddInBase::getParamRec(const QString name, bool *nameCheckOk /*= NULL*/) const
    {
        QRegExp rx("^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$");
        if (rx.indexIn(name) == -1)
        {
            if (nameCheckOk)
            {
                *nameCheckOk = false;
            }
            return Param();
        }
        else
        {
            QStringList pname = rx.capturedTexts();
            if (pname.length() > 1)
            {
                if (nameCheckOk)
                {
                    *nameCheckOk = true;
                }
                ito::Param tempParam = m_params.value(pname[1]);
                if (pname[2].length())
                {
                }
                if (pname[4].length())
                {
                }
                return tempParam; //returns default constructor if value not available in m_params. Default constructor has member isValid() => false
            }
        }
        return Param();
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
    bool AddInBase::hasDockWidget(void) const 
    { 
        return !aibp->m_dockWidget.isNull(); 
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! Returns the reference to the dock widget of this plugin or NULL, if no dock widget is provided.
    /*
    \sa hasDockWidget
    */
    QDockWidget* AddInBase::getDockWidget(void) const { return aibp->m_dockWidget; }

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
        if (aibp->m_dockWidget.isNull())
        {
            aibp->m_dockWidget = QPointer<QDockWidget>(new QDockWidget(title + QLatin1String(" - ") + tr("Toolbox")));
            connect(aibp->m_dockWidget, SIGNAL(visibilityChanged(bool)), this, SLOT(dockWidgetVisibilityChanged(bool)));
        }
        aibp->m_dockWidget->setObjectName(title.simplified() + QLatin1String("_dockWidget#") + QString::number(aibp->m_uniqueID));
        aibp->m_dockWidget->setFeatures(features);
        aibp->m_dockWidget->setAllowedAreas(allowedAreas);

        if (content) 
        {
            aibp->m_dockWidget->setWidget(content);
            content->setParent(aibp->m_dockWidget);
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void AddInBase::setIdentifier(const QString &identifier)
    {
        m_identifier = identifier;

        if (aibp->m_dockWidget)
        {
            ito::AbstractAddInDockWidget *adw = qobject_cast<ito::AbstractAddInDockWidget*>(aibp->m_dockWidget->widget());
            if (adw)
            {
                QMetaObject::invokeMethod(adw, "identifierChanged", Q_ARG(const QString &, identifier));
            }
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! sets the interface of this instance to base. \sa AddInInterfaceBase
    void AddInBase::setBasePlugin(AddInInterfaceBase *base) 
    { 
        aibp->m_pBasePlugin = base; 
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInInterfaceBase* AddInBase::getBasePlugin(void) const 
    { 
        return aibp->m_pBasePlugin;
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
                    QString err = QString("Mandatory parameter '%1' of exec function %2 cannot be defined as out parameter only").arg(QLatin1String(p.getName()), funcName);
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    qDebug() << err;
                    break;
                }
            }
            foreach(const ito::Param &p, paramsOpt)
            {
                //optional parameters can be of every type, but their flags must be In Or In|Out (NOT Out)
                if ((p.getFlags() & ito::ParamBase::Out) && !(p.getFlags() & ito::ParamBase::In))
                {
                    QString err = QString("Optional parameter '%1' of exec function %2 cannot be defined as out parameter only").arg(QLatin1String(p.getName()), funcName);
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    qDebug() << err;
                    break;
                }
            }
            foreach(const ito::Param &p, paramsOut)
            {
                //output parameters must have flag Out and not In, only types Int(Array),Char(Array),Double(Array) or String are allowed
                if ((p.getFlags() & ito::ParamBase::In) || !(p.getFlags() & ito::ParamBase::Out))
                {
                    QString err = QString("Output parameter '%1' of exec function %2 must be defined as out parameter").arg(QLatin1String(p.getName()), funcName);
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    qDebug() << err;
                    break;
                }
                if ((p.getType() & (ito::ParamBase::Int | ito::ParamBase::Char | ito::ParamBase::Double | ito::ParamBase::String)) == 0)
                {
                    QString err = QString("Output parameter '%1' of exec function %2 must be of type Int(-Array), Char(-Array), Double(-Array) or String.").arg(QLatin1String(p.getName()), funcName);
                    retValue += ito::RetVal(ito::retError,0,err.toLatin1().data());
                    qDebug() << err;
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
            else
            {
                qDebug() << "exec function " << funcName << " was rejected due to invalid argument definitions.";
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
         if (aibp->m_dockWidget)
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
    //! method invoked by AddInManager if the plugin should be pulled back to the main thread of itom.
    /*!
        Do not invoke this method in any other case. It should only be invoked by AddInManager of the itom core.
        After having moved the thread to the main thread of itom, the plugin's thread m_pThread can be closed and
        deleted. However this cannot be done in this method, since a thread can only be killed and closed by another
        thread. Therefore, this is done in the destructor of the AddIn.

        Qt does not allow pushing an object from the object's thread to the caller's thread. Only the object
        itself can move its thread to another thread.
    */
    ito::RetVal AddInBase::moveBackToApplicationThread(ItomSharedSemaphore *waitCond /*= NULL*/)
    {
        ItomSharedSemaphoreLocker locker(waitCond);

        if (aibp->m_pThread) //only push this plugin to the main thread, if it currently lives in a second thread.
        {
            moveToThread(QCoreApplication::instance()->thread());
        }

        if (waitCond)
        {
            waitCond->returnValue = retOk;
            waitCond->release();
        }
        return retOk;
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

            if (!retValue.containsError() && m_autoGrabbingEnabled == true && m_autoGrabbingListeners.size() >= 0 && m_timerID == 0)
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
    ito::RetVal AddInDataIO::setAutoGrabbingInterval(QSharedPointer<int> interval, ItomSharedSemaphore *waitCond /*= NULL*/)
    {
        ito::RetVal retval;

        if (!interval.isNull())
        {
            if (*interval > 0)
            {
                if (m_autoGrabbingEnabled)
                {
                    retval += disableAutoGrabbing();
                    m_timerIntervalMS = *interval;
                    retval += enableAutoGrabbing();
                }
                else
                {
                    m_timerIntervalMS = *interval;
                }
            }

            *interval = m_timerIntervalMS;
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("empty interval buffer has been given").toLatin1().data());
        }

        if (waitCond)
        {
            waitCond->release();
            waitCond->deleteSemaphore();
            waitCond = NULL;
        }

        return retval;
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
        m_interruptFlag(false)
    {
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    AddInActuator::~AddInActuator()
    {
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
        if (statusOnly)
        {
            emit actuatorStatusChanged(m_currentStatus, QVector<double>());
        }
        else
        {
            emit actuatorStatusChanged(m_currentStatus, m_currentPos);
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! method emits the targetChanged signal if any slot is connected to this signal.
    /*!
        The emitted values is the member variable m_targetPos
    */
    void AddInActuator::sendTargetUpdate()
    {
        emit targetChanged(m_targetPos);
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
