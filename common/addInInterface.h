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

#ifndef ADDININTERFACE_H
#define ADDININTERFACE_H

#include "commonGlobal.h"

#include "apiFunctionsInc.h"
#include "apiFunctionsGraphInc.h"

#include "addInInterfaceVersion.h"
#include "sharedStructuresQt.h"
#include "sharedStructures.h"
#include "functionCancellationAndObserver.h"

#include <qlist.h>
#include <qmap.h>
#include <qpair.h>
#include <qset.h>
#include <qthread.h>
#include <qsharedpointer.h>
#include <qmutex.h>
#include <qapplication.h>
#include <qscopedpointer.h>
#include <QtWidgets/qdockwidget.h>

//plugins define VISUAL_LEAK_DETECTOR_CMAKE in their CMake configuration file
#if defined _DEBUG  && defined(_MSC_VER) && defined(VISUAL_LEAK_DETECTOR_CMAKE)
#ifndef NOMINAX
#define NOMINMAX //instead min, max is defined as macro in winDef.h, included by vld.h
#include "vld.h"
#undef NOMINMAX
#else
#include "vld.h"
#endif
#endif

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

//! macro to create a new plugin instance in the method getAddInInst of any plugin
/*!
Insert this macro at the first line of the method getAddInInst of your plugin.
Pass the name of the corresponding plugin class (not its interface class)
*/
#define NEW_PLUGININSTANCE(PluginClass) \
    PluginClass* newInst = new PluginClass(); \
    newInst->setBasePlugin(this); \
    *addInInst = qobject_cast<ito::AddInBase*>(newInst); \
    m_InstList.append(*addInInst);

//! macro to delete a plugin instance in the method closeThisInst of any plugin
/*!
Insert this macro at the first line of the method closeThisInst of your plugin.
Pass the name of the corresponding plugin class (not its interface class).
This macro is the opposite of NEW_PLUGININSTANCE
*/
#define REMOVE_PLUGININSTANCE(PluginClass) \
   if (*addInInst) \
      { \
      (*addInInst)->deleteLater(); \
      m_InstList.removeOne(*addInInst); \
      }

//! macro to set the pointer of the plugin to all its defined filters and widgets
/*!
Insert this macro right after NEW_PLUGININSTANCE in all plugins that are
algo plugins.
*/
#define REGISTER_FILTERS_AND_WIDGETS \
    foreach(ito::AddInAlgo::FilterDef *f, newInst->m_filterList) \
        { \
        f->m_pBasePlugin = this; \
        } \
    foreach(ito::AddInAlgo::AlgoWidgetDef *w, newInst->m_algoWidgetList) \
        { \
        w->m_pBasePlugin = this; \
        }

//write this macro right after Q_INTERFACE(...) in your interface class definition
#define PLUGIN_ITOM_API \
        protected: \
            void importItomApi(void** apiPtr) \
                            {ito::ITOM_API_FUNCS = apiPtr;} \
            void importItomApiGraph(void** apiPtr) \
                            { ito::ITOM_API_FUNCS_GRAPH = apiPtr;} \
        public: \
            virtual int getAddInInterfaceVersion() const \
                            { return ITOM_ADDININTERFACE_VERSION; } \
            //.

QT_BEGIN_NAMESPACE
class QPluginLoader;
QT_END_NAMESPACE

namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    //! tPluginType enumeration
    /*!
    used to describe the plugin type and subtype (in case of DataIO main type)
    e.g. typeDataIO|typeGrabber for a frame grabber
    */
    enum tPluginType
    {
        typeDataIO = 0x1,     /*!< base type for data input and output (cameras, AD-converter, display windows...) */
        typeActuator = 0x2,     /*!< base type for actuators and motors */
        typeAlgo = 0x4,     /*!< base type for algorithm plugin */
        typeGrabber = 0x80,    /*!< subtype of dataIO for cameras (grabbers), use this type in combination with typeDataIO (OR-combination) */
        typeADDA = 0x100,   /*!< subtype of dataIO for AD and DA-converters, use this type in combination with typeDataIO (OR-combination) */
        typeRawIO = 0x200   /*!< subtype of dataIO for further input-output-devices (like display windows), use this type in combination with typeDataIO (OR-combination) */
    };

    //! tActuatorStatus enumeration
    /*!
    flags used for describing the status of one axis of an actuator plugin.

    These flags are intended to be combined in the status bitmask.
    Usually the bitmask for each mask is saved in the vector ito::AddInActuator::m_currentStatus
    of an actuator plugin.

    The bitmask is divided into different topical areas (moving flags, switches, general status).
    */
    enum tActuatorStatus
    {
        //moving flags
        actuatorUnknown = 0x0001, /*!< moving status of axis is unknown */
        actuatorInterrupted = 0x0002, /*!< movement has been interrupted by the user, axis is immediately stopped */
        actuatorMoving = 0x0004, /*!< axis is currently moving */
        actuatorAtTarget = 0x0008, /*!< axis reached target */
        actuatorTimeout = 0x0010, /*!< no signal from axis, timeout */
        //switches
        actuatorEndSwitch = 0x0100, /*!< axis reached an undefined end switch */
        actuatorLeftEndSwitch = 0x0200, /*!< axis reached the specified left end switch (if set, also set actuatorEndSwitch), deprecated */
        actuatorRightEndSwitch = 0x0400, /*!< axis reached the specified right end switch (if set, also set actuatorEndSwitch), deprecated */
        actuatorEndSwitch1 = 0x0200,/*!< axis reached the specified left end switch (if set, also set actuatorEndSwitch) */
        actuatorEndSwitch2 = 0x0400,    /*!< axis reached the specified left end switch (if set, also set actuatorEndSwitch) */
        actuatorRefSwitch = 0x0800, /*!< axis reached an undefined reference switch */
        actuatorLeftRefSwitch = 0x1000, /*!< axis reached the specified left reference switch (if set, also set actuatorRefSwitch), deprecated */
        actuatorRightRefSwitch = 0x2000, /*!< axis reached the specified right reference switch (if set, also set actuatorRefSwitch), deprecated */
        actuatorRefSwitch1 = 0x1000,/*!< axis reached the specified right reference switch (if set, also set actuatorRefSwitch)*/
        actuatorRefSwitch2 = 0x2000,/*!< axis reached the specified right reference switch (if set, also set actuatorRefSwitch)*/

        //status flags
        actuatorAvailable = 0x4000, /*!< axis is generally available */
        actuatorEnabled = 0x8000, /*!< axis is enabled for movements */
        actuatorError = 0x10000,/*axis has encountered error/reports error*/

        actMovingMask = actuatorUnknown | actuatorInterrupted | actuatorMoving | actuatorAtTarget | actuatorTimeout, /*!< bitmask that marks all bits related to the movement */

        actEndSwitchMask = actuatorEndSwitch \
        | actuatorEndSwitch1 | actuatorEndSwitch2, /*!< bitmask that marks all bits related to end switches */

        actRefSwitchMask = \
        actuatorRefSwitch \
        | actuatorRefSwitch1 | actuatorRefSwitch2, /*!< bitmask that marks all bits related to reference switches */

        actSwitchesMask = actEndSwitchMask | actRefSwitchMask,                                /*!< bitmask that marks all bits related to reference and end switches */

        actStatusMask = actuatorAvailable | actuatorEnabled | actuatorError                     /*!< bitmask that marks all status flags */
    };

    enum tAutoLoadPolicy
    {
        autoLoadAlways = 0x1, /*!< always loads xml file by addInManager */
        autoLoadNever = 0x2, /*!< never automatically loads parameters from xml-file (default) */
        autoLoadKeywordDefined = 0x4  /*!< only loads parameters if keyword autoLoadParams=1 exists in python-constructor */
    };

    enum tAutoSavePolicy
    {
        autoSaveAlways = 0x1, /*!< always saves parameters to xml-file at shutdown */
        autoSaveNever = 0x2  /*!< never saves parameters to xml-file at shutdown (default) */
    };

    struct ExecFuncParams
    {
        ExecFuncParams() : infoString("") {}
        QVector<Param> paramsMand; /*!< mandatory parameters (default set), must have flag In or In|Out */
        QVector<Param> paramsOpt;  /*!< optional parameters (default set), must have flag In or In|Out */
        QVector<Param> paramsOut;  /*!< return parameters (default set), must have Out. Only types Int,Char,Double,String,IntArray,CharArray or DoubleArray are allowed. */
        QString infoString;
    };

    struct FilterParams
    {
        QVector<Param> paramsMand;
        QVector<Param> paramsOpt;
        QVector<Param> paramsOut;
    };

    class AddInBase;        //!< forward declaration
    class DataObject;
    class AddInBasePrivate;          //!< forward declaration to private container class of AddInBase
    class AddInInterfaceBasePrivate; //!< forward declaration to private container class of AddInInterfaceBase
    class AddInActuatorPrivate;      //!< forward declaration to private container class of AddInActuator
    class AddInDataIOPrivate;        //!< forward declaration to private container class of AddInDataIO
    class AddInAlgoPrivate;          //!< forward declaration to private container class of AddInAlog

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class AddInInterfaceBase
    *   @brief class of the AddIn - Library (DLL) - Interface
    *
    *   The measurement program can (and should) be expanded with additional functionality by "plugins". The aim of separating
    *   part of the program into plugins is to speed up developement and to reduce complexity for plugin developers.
    *   The main program holds the necessary functionality to load and use plugins with either the integrated python interpreter
    *   or within c/c++ functions.
    *   All plugins are divded into two parts:
    *       - the AddInInterfaceBase
    *       - the AddIn (derived from the specific addIn-type that should be created which is derived from AddInBase
    *
    *   The Interface is a small light weight class which is used when loading the dll into the main program. It holds information
    *   about the plugin itself, e.g. name, version, parameters and so on.
    *   When loading the plugin is tested for compability with the current version of the main program based on the information in the
    *   interface class.
    *   The AddInXXX class provides the plugin functionality. Instances of this class are only created when the plugin "is used" either
    *   by python or within other functions. For a description about the loading, using and unloading process see \ref AddInBase, \ref AddInActuator,
    *   \ref AddInDataIO and \ref AddInAlgo.
    *   The instantiation of an AddIn class is a two step process. At first the necessary and optional parameter values as well as the
    *   plugin's reference number are retrieved from the AddInManager using the getInitParams \ref getInitParams method. Then a new instance
    *   is obtained using one of the initAddIn \ref initAddIn methods. Which first create a new instance, move the instance to a new thread
    *   and at last call the classes init method
    */
    class ITOMCOMMONQT_EXPORT AddInInterfaceBase : public QObject
    {
        Q_OBJECT

    private:
        //!< internal function used within the closing process
        virtual ito::RetVal closeThisInst(ito::AddInBase **addInInst) = 0;

        QScopedPointer<AddInInterfaceBasePrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
        Q_DECLARE_PRIVATE(AddInInterfaceBase);

    protected:
        int m_type;                                     //!< plugin type
        int m_version;                                  //!< plugin version
        QString m_filename;                             //!< plugin (library) filename on the disc
        int m_maxItomVer;                               //!< minimum required version of the main program
        int m_minItomVer;                               //!< maximum supported version of the main program
        QString m_author;                                //!< the plugin author
        QString m_description;                          //!< a brief descrition of the plugin
        QString m_detaildescription;                    //!< a detail descrition of the plugin
        QString m_license;                              //!< a short license string for the plugin, default value is "LGPL with ITO itom-exception"
        QString m_aboutThis;
        QList<ito::AddInBase *> m_InstList;             //!< vector holding a list of the actual instantiated classes of the plugin
        QVector<ito::Param> m_initParamsMand;          //!< vector with the mandatory initialisation parameters, please only read this vector within the init-method of AddInBase (afterwards it might have been changed)
        QVector<ito::Param> m_initParamsOpt;           //!< vector with the optional initialisation parameters, please only read this vector within the init-method of AddInBase (afterwards it might have been changed)
        tAutoLoadPolicy m_autoLoadPolicy;               /*!< defines the auto-load policy for automatic loading of parameters from xml-file at startup of any instance */
        tAutoSavePolicy m_autoSavePolicy;               /*!< defines the auto-save policy for automatic saving of parameters in xml-file at shutdown of any instance */
        bool m_callInitInNewThread;                     /*!< true (default): the init-method of addIn will be called after that the plugin-instance has been moved to new thread (my addInManager). false: the init-method is called in main(gui)-thread, and will be moved to new thread afterwards (this should only be chosen, if not otherwise feasible) */


        virtual void importItomApi(void** apiPtr) = 0; //this methods are implemented in the plugin itsself. Therefore place ITOM_API right after Q_INTERFACE in the header file and replace Q_EXPORT_PLUGIN2 by Q_EXPORT_PLUGIN2_ITOM in the source file.
        virtual void importItomApiGraph(void** apiPtr) = 0;

        //!> check if we have gui support
        inline bool hasGuiSupport()
        {
            if (qobject_cast<QApplication*>(QCoreApplication::instance()))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

    public:
        void **m_apiFunctionsBasePtr;
        void **m_apiFunctionsGraphBasePtr;

        //! destructor
        virtual ~AddInInterfaceBase();

        //! default constructor
        AddInInterfaceBase();

        //! pure virtual function that returns the addin interface version of the plugin
        /* This method is automatically implemented by the PLUGIN_ITOM_API macro.
        The definition is 0xAABBCC where AA is the major, BB the minor and CC the patch.
        */
        virtual int getAddInInterfaceVersion() const = 0;

        //! returns addIn type
        inline int getType(void) const { return m_type; }
        //! returns addIn version
        inline int getVersion(void) const { return m_version; }
        //! returns minimum required version of main program
        inline int getMinItomVer(void) const { return m_minItomVer; }
        //! returns maximum supported version of main program
        inline int getMaxItomVer(void) const { return m_maxItomVer; }

        //! returns whether init-method should be called in new thread (default) or still in main thread
        inline bool getCallInitInNewThread(void) const { return m_callInitInNewThread; }

        //! returns true if the plugin allows his own parameter load to be autoloaded by addin manager
        inline tAutoLoadPolicy getAutoLoadPolicy(void) const { return m_autoLoadPolicy; }

        //! returns true if the plugin allows his own parameter save to be autoloaded by addin manager
        inline tAutoSavePolicy getAutoSavePolicy(void) const { return m_autoSavePolicy; }

        //! returns plugin author
        const QString getAuthor(void) const { return m_author; }
        //! returns a brief description of the plugin
        const QString getDescription(void) const { return m_description; }
        //! returns a detailed description of the plugin
        const QString getDetailDescription(void) const { return m_detaildescription; }
        //! returns a detailed description of the plugin license
        const QString getLicenseInfo(void) const { return m_license; }
        //! returns a detailed description of the plugin compile informations
        const QString getAboutInfo(void) const { return m_aboutThis; }
        //! returns the plugin's filename
        const QString getFilename(void) const { return m_filename; }

        const ito::RetVal setFilename(const QString &name) { m_filename = name; return ito::retOk; }
        //! returns a list of the actual intantiated classes from this plugin
        inline QList<ito::AddInBase *> getInstList(void) { return m_InstList; }
        inline const QList<ito::AddInBase *> getInstList(void) const { return m_InstList; }
        //! method for closing an instance
        ito::RetVal closeInst(ito::AddInBase **addInInst);
        //! returns a vector with the mandatory initialisation parameters
        virtual QVector<ito::Param>* getInitParamsMand(void) { return &m_initParamsMand; }
        //! returns a vector with the optional initialisation parameters
        virtual QVector<ito::Param>* getInitParamsOpt(void) { return &m_initParamsOpt; }
        //! method to instantiate a new class of the plugin
        virtual ito::RetVal getAddInInst(ito::AddInBase **addInInst) = 0;
        //! increment use reference
        void incRef(ito::AddInBase *addIn);
        //! decrement use reference
        void decRef(ito::AddInBase *addIn);
        //! get reference counter
        int getRef(ito::AddInBase *addIn);
        //! get number instantiated plugins
        int getInstCount() { return m_InstList.length(); }
        //! set api function pointer
        void setApiFunctions(void **apiFunctions);
        void setApiFunctionsGraph(void ** apiFunctionsGraph);

        void setLoader(QPluginLoader *loader);
        QPluginLoader * getLoader(void) const;

        bool event(QEvent *e);
    };

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class AddInBase
    *   @brief Base class for all plugins.
    *
    *   The common methods and members are defined here. The available plugin type (actuator \ref AddInActuator,
    *   dataIO \ref AddInDataIO and algo \ref AddInAlgo) are derived from this class. At the main program startup all available plugins
    *   located in the plugin directory are searched and matched against the current plugin interface version. Then all compatible
    *   plugins can be check with the AddInManager. Up to that stage for each plugin only a lightweight AddInInterface \ref AddInInterfaceBase
    *   class has been loaded. To use a plugin instances of the plugin class have to be instantiated. The AddInInterface is run in the
    *   calling thread whilst the plugin classes are run in separate threads. Therefore the plugin functions are implemented as slots which can be
    *   used e.g. with the invokeMethod function.
    *   The base functionality included in this base class is getting the plugin's parameter list, getting the classes uniqueID (which is
    *   used e.g. for saveing the parameter values) and optinally to bring up a configuration dialog.
    */
    class ITOMCOMMONQT_EXPORT AddInBase : public QObject
    {
        Q_OBJECT

    public:

        struct AddInRef {
            AddInRef() : type(-1), ptr(0) {}
            AddInRef(void *p, int t) : type(t), ptr(p) {}
            int type;
            void *ptr;
        };

        //! method to retrieve a parameter from the parameter map (m_params)
        const Param getParamRec(const QString name, bool *nameCheckOk = NULL) const;

        //! returns the interface of this instance. \sa AddInInterfaceBase
        AddInInterfaceBase* getBasePlugin(void) const;

        //! creates new thread for the class instance and moves this instance to the new thread
        ito::RetVal MoveToThread(void);

        //! returns a map with the parameters of this plugin.
        /*
        Use the method setParam in order to change any parameter.

        \param paramNames [out]. The pointer contains a pointer to the map after the call of this function
        \return RetVal returns retOk.
        */
        inline const ito::RetVal getParamList(QMap<QString, Param> **paramNames) { *paramNames = &m_params; return ito::retOk; }

        //! returns list of registered additional functions
        /*
        \param [out] funcs is the pointer to a map, that points to the internal map of additional functions after the method-call.
        \return retOk
        \sa registerExecFunc
        */
        inline const ito::RetVal getExecFuncList(QMap<QString, ExecFuncParams> **funcs) { *funcs = &m_execFuncList; return ito::retOk; }

        //! retrieve the uniqueID of this instance
        int getID() const;

        //! retrieve the unique identifier of this instance
        inline QString getIdentifier() const { return m_identifier; }

        //! determine if a configuration dialog is available
        virtual int hasConfDialog(void);

        //! open configuration dialog
        virtual const ito::RetVal showConfDialog(void);

        //! returns true if this instance has firstly been created by the GUI
        int createdByGUI() const;

        //! method to set whether this instance has been firstly created by the GUI (true) or by any other component (Python, C++, other plugin,..) (false)
        void setCreatedByGUI(int value);

        //! Returns the reference counter of this instance.
        /*
        The reference counter is zero-based, hence, the value zero means that one reference is pointing to this instance
        */
        int getRefCount(void) const;

        //! Returns true if this plugin provides a dock widget, that can be shown in the main window.
        /*
        \sa getDockWidget
        */
        bool hasDockWidget(void) const;

        //! Returns the reference to the dock widget of this plugin or NULL, if no dock widget is provided or if it is already deleted.
        /*
        \sa hasDockWidget
        */
        QDockWidget* getDockWidget(void) const;

        // doc in source
        virtual void dockWidgetDefaultStyle(bool &floating, bool &visible, Qt::DockWidgetArea &defaultArea) const;

        //! returns the alive-flag of this plugin
        /*
        Any time-consuming operation of the plugin should regularly set the alive-flag to true
        by calling setAlive. The state of this flag is returned by this method and afterwards
        reset to 0. This method is thread-safe.

        \return current status of alive-flag (1 if "still alive", else 0)
        \sa setAlive
        */
        int isAlive(void);

        //! sets the alive-flag to 1 ("still alive")
        /*
        This method is thread-safe.

        \sa isAlive
        */
        void setAlive(void);

        //! returns in a thread-safe way the status of the m_initialized-member variable. This variable should be set to true at the end of the init-method.
        bool isInitialized(void) const;

        //! sets in a thread-safe way the status of the m_initialized-member
        /*
        \param [in] initialized is the value to set
        */
        void setInitialized(bool initialized);

        //! returns vector of AddInRef instances.
        /*
        This vector contains all plugin-instances, that have been passed to the init
        method of this plugin. The reference counter of these plugin is incremented at
        initialization of this plugin and decremented if this plugin will be destroyed.

        \sa AddInRef, init
        */
        QVector<ito::AddInBase::AddInRef *> * getArgAddIns(void) { return &m_hwDecList; }

        //! returns the user mutex of this plugin, that can be used for user-defined purposes.
        /* This mutex has no designed task in the plugin, however it can be used by
        the user to for instance protect a sequence of different calls to this plugin.

        This can be important, if the plugin is for instance a communication object,
        that is used by different other hardware instances (e.g. a SerialIO to
        an arduino, that controls different motors, sensors etc.). Then, it might
        be important, that every hardware plugin object, that uses the serialIO
        plugin, can protect a setVal / getVal sequence without that any other
        plugin instance interrupts its. However, it is the task of the user to
        implement that protection. This mutex can only help for this.
        */
        QMutex& getUserMutex();

        static int getMaximumThreadCount();

        static RetVal setMaximumThreadCount(int threadCount);

    protected:
        // constructor (doc in source)
        AddInBase();

        // destructor (doc in source)
        virtual ~AddInBase();

        //! sets the identifier of the plugin. The slot AbstractAddInDockWidget::identifierChanged is invoked if a corresponding dock widget is available.
        void setIdentifier(const QString &identifier);

        //doc in source
        void createDockWidget(QString title, QDockWidget::DockWidgetFeatures features, Qt::DockWidgetAreas allowedAreas = Qt::AllDockWidgetAreas, QWidget *content = NULL);

        // doc in source
        ito::RetVal registerExecFunc(const QString funcName, const QVector<ito::Param> &paramsMand, const QVector<ito::Param> &paramsOpt, const QVector<ito::Param> &paramsOut, const QString infoString);

        //! sets the interface of this instance to base. \sa AddInInterfaceBase
        void setBasePlugin(AddInInterfaceBase *base);

        QMap<QString, Param> m_params;                        //!< map of the available parameters

        QString m_identifier;                               //!< unique identifier (serial number, com-port...)

        //! check if we have gui support
        inline bool hasGuiSupport()
        {
            if (qobject_cast<QApplication*>(QCoreApplication::instance()))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

    private:
        Q_DISABLE_COPY(AddInBase)

        //! increments reference counter of this plugin (thread-safe)
        void incRefCount(void);

        //! decrements reference counter of this plugin (thread-safe)
        void decRefCount(void);

        QVector<ito::AddInBase::AddInRef *> m_hwDecList;  //!< list of hardware that was passed to the plugin on initialisation and whose refcounter was incremented
        QMap<QString, ExecFuncParams> m_execFuncList;     //!< map with registered additional functions. funcExec-name -> (default mandParams, default optParams, default outParams, infoString)

        QScopedPointer<AddInBasePrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed). pointer to private class of AddInBase defined in AddInInterface.cpp. This container is used to allow flexible changes in the interface without destroying the binary compatibility
        Q_DECLARE_PRIVATE(AddInBase);

        friend class AddInInterfaceBase;                  //!< AddInBase is friend with AddInInterfaceBase, such that the interface can access methods like the protected constructor or destructor of this plugin class.

        static int m_instCounter;
        static int maxThreadCount;                        //!< maximum number of threads algorithms can use e.g. with OpenMP parallelization. This is a number between 1 and QThread::idealThreadCount()

    Q_SIGNALS:
        //! This signal usually is emitted if the vector m_params is changed.
        /*!
        Emit this signal for instance in setParam if the parameter has been changed in order
        to inform connected dock-widgets... about the change. This signal is also emitted
        if you invoke the slot sendParameterRequest.

        \param params is the parameter-vector to send (usually m_params)
        \sa m_params, sendParameterRequest
        */
        void parametersChanged(QMap<QString, ito::Param> params);

    public Q_SLOTS:
        //! method for the initialisation of a new instance of the class (must be overwritten)
        virtual ito::RetVal init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! method for closing an instance (must be overwritten)
        virtual ito::RetVal close(ItomSharedSemaphore *waitCond) = 0;

        //! method for the retrieval of a parameter. The actual value is always passed as ito::Param (must be overwritten). See also \ref setParam
        virtual ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! method to set a parameter. The actual value is always passed as ito::ParamBase (must be overwritten). See also \ref getParam
        virtual ito::RetVal setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond = NULL) = 0;

        //! method for setting various parameters (can be used instead of multiple calls to setParam, this can safe multiple invocations)
        ito::RetVal setParamVector(const QVector<QSharedPointer<ito::ParamBase> > values, ItomSharedSemaphore *waitCond = NULL);

        //! method for getting various parameters (can be used instead of multiple calls to getParam, this can safe multiple invocations)
        ito::RetVal getParamVector(const QVector<QSharedPointer<ito::Param> > values, ItomSharedSemaphore *waitCond = NULL);

        //! overwrite this function if you registered exec funcs. Once the exec function is called, this method is executed.
        virtual ito::RetVal execFunc(const QString funcName, QSharedPointer<QVector<ito::ParamBase> > paramsMand, QSharedPointer<QVector<ito::ParamBase> > paramsOpt, QSharedPointer<QVector<ito::ParamBase> > paramsOut, ItomSharedSemaphore *waitCond = NULL);

        //! method invoked by AddInManager if the plugin should be pulled back to the main thread of itom. (not for direct use in plugins)
        ito::RetVal moveBackToApplicationThread(ItomSharedSemaphore *waitCond = NULL);

        //! immediately emits the signal parametersChanged
        /*!
        call or invoke this method for instance after creating a configuration dialog for the plugin.
        Then the dialog gets the current parameter map m_params, if it has been connected to the signal
        parametersChanged (must be done before).

        \sa parametersChanged, m_params
        */
        void sendParameterRequest(){ emit parametersChanged(m_params); };

    private Q_SLOTS:

        //! overwrite this slot if you want to get informed when the dock-widget of the plugin becomes (in)visible
        /*!
        It is recommended to use this method, in order to connect the widget to signals like parametersChanged,
        actuatorStatusChanged or targetChanged (both actuator only) if the dock-widget becomes visible and disconnect
        them if it is hidden. This is useful in order to avoid intense function calls if the dock-widget is not
        visible.

        \sa parametersChanged
        */
        virtual void dockWidgetVisibilityChanged(bool /*visible*/) {};
    };

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class AddInDataIO
    *   @brief base class for all dataIO plugin classes
    *
    *   This class is one step further down the line from \ref AddInBase. DataIO plugins must be derived from this class which
    *   is derived from AddInBase. In this class only the methods specific to dataIO plugins are declared. A dataIO
    *   device (e.g. a framegrabber / camera) should use this sequence to operate:
    *   1. startDevice - device is now ready to acquire data
    *   2. acquire - now a data set is freezed and prepared for retrieval
    *   3. getVal - method to retrieve the previously freezed data (shallow copy, if you want to work with this data, make a deep copy)
    *   4. stopDevice - acquisition is stopped, device is no longer ready to record data
    *
    *   the steps 2. and 3. can be repeated until the desired number of "frames" has been read. The device
    *   should be only started once before the first acquisition of a sequence and stop only at their end.
    *   The device itself MUST NOT allocate memory for the data to be stored. This HAS TO BE DONE in the main programm
    *   or in the calling method!
    *
    *   If a live image is listening this device, its source node calls startDeviceAndRegisterListener. If the autoGrabbing-flag is enabled,
    *   a timer will be started, which triggers the method 'timerEvent' (should be implemented by any camera). If this flag is disabled, the
    *   live image is registered, but no images will be regularily aquired. In this case, only manually taken images will be passed to any registered
    *   source node. If the flag is enabled again, the timer is restarted and every live image will automatically get new images. This is done by
    *   invoking the slot 'setSource' of every registered source node.
    *
    *   Every camera will only return shallow copies of its internal image both to the live image and to the user. This image can be read by everybody.
    *   If the user wants to change values in this image, he should make a deep copy first.
    *
    */
    class ITOMCOMMONQT_EXPORT AddInDataIO : public AddInBase
    {
        Q_OBJECT

    private:
        Q_DISABLE_COPY(AddInDataIO)

        QScopedPointer<AddInDataIOPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
        Q_DECLARE_PRIVATE(AddInDataIO);

    protected:
        virtual ~AddInDataIO();
        AddInDataIO();

        void runStatusChanged(bool deviceStarted);

        //void timerEvent (QTimerEvent *event) = 0; //implement this event in your plugin, if you are a device, which is accessible by any liveImage!!!

        QSet<QObject*> m_autoGrabbingListeners;  /*!< list of listeners (live image source nodes), which want to have updates from this camera */
        int m_timerID;               /*!< internal ID of the timer, which acquires images for any live view (if allowed) */
        int m_timerIntervalMS;       /*!<  timer interval (in ms)*/
        bool m_autoGrabbingEnabled;  /*!<  defines, whether the auto-grabbing timer for any live image can be activated. If this variable becomes false and any timer is activated, this timer is killed.*/

    public:
        inline int getAutoGrabbing() { return m_autoGrabbingEnabled; }  /*!< returns the state of m_autoGrabbingEnabled; consider this method as final */

    Q_SIGNALS:

    public Q_SLOTS:
        //! method to start the device - i.e. get ready to record data
        virtual ito::RetVal startDevice(ItomSharedSemaphore *waitCond);

        //! method to stop the device, it is no longer possible to acquire data
        virtual ito::RetVal stopDevice(ItomSharedSemaphore *waitCond);

        //! freeze the current data and prepare it for retrieval
        virtual ito::RetVal acquire(const int trigger, ItomSharedSemaphore *waitCond = NULL);

        //! stops a continuous acquisition (usually only required by AD/DA converters). This method has not to be implemented in every plugin. New from itom.AddIn.Interface/4.0.0 on
        virtual ito::RetVal stop(ItomSharedSemaphore *waitCond = NULL);

        //! read data from the device into a dataObject (which is passed as void pointer actually). Output is a shallow-copy to the grabber internal buffer-object.
        virtual ito::RetVal getVal(void* data, ItomSharedSemaphore *waitCond = NULL);

        //! read data from the device into a "raw data pointer" (in this case a char * is passed, pointing to the start of the preallocated memory)
        virtual ito::RetVal getVal(QSharedPointer<char> data, QSharedPointer<int> length, ItomSharedSemaphore *waitCond = NULL);

        //! read data from the device into a dataObject (which is passed as void pointer actually). Output is a deep-copy to the grabber internal object.
        virtual ito::RetVal copyVal(void *dObj, ItomSharedSemaphore *waitCond);

        //! write data, e.g. to the DA part of an ADDA card
        virtual ito::RetVal setVal(const char *data, const int length, ItomSharedSemaphore *waitCond = NULL);

        //! enables the timer for auto grabbing (live image), if any live image has signed on (usually this method must not be overwritten)
        ito::RetVal enableAutoGrabbing(ItomSharedSemaphore *waitCond = NULL); //consider this method as final

        //! disables the timer for auto grabbing (live image) (usually this method must not be overwritten)
        ito::RetVal disableAutoGrabbing(ItomSharedSemaphore *waitCond = NULL); //consider this method as final

        //! sets a new interval for the auto-grabbing timer (in ms). If interval <= 0 is passed, nothing is changed, but the current interval is returned. This method does not enable or disable the timer.
        ito::RetVal setAutoGrabbingInterval(QSharedPointer<int> interval, ItomSharedSemaphore *waitCond = NULL); //consider this method as final

        //! starts device and registers obj as listener (live image). This listener must have a slot void setSource(QSharedPointer<ito::DataObject>, ItomSaredSemaphore).
        ito::RetVal startDeviceAndRegisterListener(QObject* obj, ItomSharedSemaphore *waitCond = NULL); //consider this method as final

        //! stops device and unregisters obj (live image).
        ito::RetVal stopDeviceAndUnregisterListener(QObject* obj, ItomSharedSemaphore *waitCond = NULL); //consider this method as final
    };

    //----------------------------------------------------------------------------------------------------------------------------------

    /** @class AddInActuator
    *   @brief base class for all actuator plugin classes
    *
    *   This class is one step further down the line from \ref AddInBase. Actuator plugins must be derived from this class which
    *   is derived from AddInBase. In this class only the methods specific to actuator plugins are declared.
    */
    class ITOMCOMMONQT_EXPORT AddInActuator : public AddInBase
    {
        Q_OBJECT

    private:
        Q_DISABLE_COPY(AddInActuator)

        QScopedPointer<AddInActuatorPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
        Q_DECLARE_PRIVATE(AddInActuator);

    protected:
        virtual ~AddInActuator();
        AddInActuator();

        QVector<int>    m_currentStatus;  /*!< vector (same length than number of axes) containing the status of every axis. The status is a combination of enumeration ito::tActuatorStatus. */
        QVector<double> m_currentPos;  /*!< vector (same length than number of axes) containing the current position (mm or degree) of every axis. The current position should be updated with a reasonable frequency (depending on the actuator and situation)*/
        QVector<double> m_targetPos;  /*!< vector (same length than number of axes) containing the target position (mm or degree) of every axis */

        //! checks whether any axis is still moving (moving flag is set)
        bool isMotorMoving() const;

        void sendStatusUpdate(const bool statusOnly = false); /* emits actuatorStatusChanged signal with the vector of currentStatus (and currentPos if statusOnly = false) to notify connected listeners about the current status (and position)*/
        void sendTargetUpdate(); /* emits targetChanged with the vector of targetPositions to notify connected listeners about the change of the target position(s) */

        //! this method must be overwritten.
        /*!
        WaitForDone should wait for a moving motor until the indicated axes (or all axes of nothing is indicated) have stopped or a timeout or user interruption
        occurred. The timeout can be given in milliseconds, or -1 if no timeout should be considered. The flag-parameter can be used for your own purpose.
        */
        virtual ito::RetVal waitForDone(const int timeoutMS = -1, const QVector<int> axis = QVector<int>() /*if empty -> all axis*/, const int flags = 0 /*for your use*/) = 0;

        //! sets status flags of given status variable
        /*!
        Use this small inline method in order to set the status of given status variable. The status is an OR-combination of the enumeration ito::tActuatorStatus.
        You can assign a mask (keepMask). Bits whithin this mask will be unchanged.

        \param [in,out] status is the status variable which is changed.
        \param [in] newFlags    is an OR-combination of ito::tActuatorStatus which is assigned to status.
        \param [in] keepMask    is a mask whose bits are not deleted if they are not contained in newFlags.

        \sa tActuatorStatus
        */
        void setStatus(int &status, const int newFlags, const int keepMask = 0);

        //! sets status flags of the status of the given axes
        /*!
        This method calls setStatus for the status of every given axis-number. The status is directly changed in the member variable m_currentStatus.
        For further information see the description of method setStatus (for one single status variable)

        \param [in] axis        is the vector with axis-numbers.
        \param [in] newFlags    is an OR-combination of ito::tActuatorStatus which is assigned to status.
        \param [in] keepMask    is a mask whose bits are not deleted if they are not contained in newFlags.

        \sa setStatus
        */
        void setStatus(const QVector<int> &axis, const int newFlags, const int keepMask = 0);

        //! changes the status bit of the given status value from one existing to a new value.
        /*!
        If the given existingFlag bitmask of the status-value is set, it is completely replaced by the flags set by replaceFlag.

        \param [in,out] status is the status variable which is changed.
        \param [in] existingFlag    is the bitmask which must be set and is replace by replaceFlag
        \param [in] replaceFlag     is the bitmask which is used for the replacement.

        \sa tActuatorStatus
        */
        void replaceStatus(int &status, const int existingFlag, const int replaceFlag);

        //! changes the status flags of the status of the given axes from one existing to a new value
        /*!
        This method calls replaceStatus for the status of every given axis-number. The status is directly changed in the member variable m_currentStatus.
        For further information see the description of method replaceStatus (for one single status variable)

        \param [in] axis        is the vector with axis-numbers.
        \param [in] existingFlag    is the bitmask which must be set and is replace by replaceFlag
        \param [in] replaceFlag     is the bitmask which is used for the replacement.

        \sa replaceStatus
        */
        void replaceStatus(const QVector<int> &axis, const int existingFlag, const int replaceFlag);

        //! initializes the current status, current position and target position vectors to the same size and the given start values
        /*!
        If sendUpdateSignals is true, the signals targetChanged and actuatorStatusChanged are emitted with the new size and content.
        In any case, the last signalled states are set to these initial values, too.
        */
        void initStatusAndPositions(int numAxes, int status, double currentPosition = 0.0, double targetPosition = 0.0, bool sendUpdateSignals = true);

        //! returns interrupt flag (thread-safe)
        /*!
        This methods returns true if the interrupt flag has been set using setInterrupt. Once this method is called, the interrupt flag is reset.

        \return true if interrupt flag has been set, else false.
        \sa setInterrupt
        */
        bool isInterrupted();

    public:
        //! set interrupt flag (thread-safe)
        /*!
        call this method (even direct call from different thread is possible) if you want to set the interrupt flag in order to stop a moving actuator.
        This flag has to be continously be checked in waitForDone using the method isInterrupted, since this method only sets the flag without initiating
        further actions.

        \sa waitForDone, isInterrupted
        */
        void setInterrupt();

        //! resets the interrupt flag (thread-safe)
        /*!
        call this method (even direct call from different thread is possible) if you want to reset the interrupt flag.
        This method is called if setOrigin, setPosAbs or setPosRel is called from Python since the interrupt flag can be set if a Python
        script is interrupted (depending on itom property).
        */
        void resetInterrupt();

        //! put the latest signalled states (current status, position and target position) to the given arguments.
        /*!
        "last signalled" signifies that whenever the signal targetChanged or actuatorStatusChanged is emitted,
        the current value of these states is stored and can be obtained by this method.

        The call to this method is thread-safe.

        \return ito::retOk if all states could be read, ito::retError if no status or position value has been reported up to now.
        */
        ito::RetVal getLastSignalledStates(QVector<int> &status, QVector<double> &currentPos, QVector<double> &targetPos);

    Q_SIGNALS:
        //! signal emitted if status or actual position of any axis has been changed.
        /*!
        Usually this signal can be sent using the method sendStatusUpdate(...). This method firstly checks if any slot has been connected to this signal
        and only fires the signal of at least one slot is connected (e.g. any docking-widget).

        Usually the status and positions are taken from the member variables m_currentStatus and m_currentPos.

        \params status is a vector with the length of the number of axis and each value contains a OR-combination of ito::tActuatorStatus
        \params actPosition is a vector with the length of the number of axis containing the actual axis positions, or length 0, if no actual positions are sended.
        */
        void actuatorStatusChanged(QVector<int> status, QVector<double> actPosition);

        //! signal emitted if target position of any axis has changed.
        /*!
        Usually this signal can be sent using the method sendTargetUpdate(). This method firstly checks if any slot has been connected to this signal
        and only fires the signal of at least one slot is connected (e.g. any docking-widget).

        Usually the target is taken from the member variable m_targetPos.

        \params targetPositions is a vector with the length of the number of axis containing the desired target positions for ALL axis.
        */
        void targetChanged(QVector<double> targetPositions);

    public Q_SLOTS:
        //! method to calibrate a single axis
        virtual ito::RetVal calib(const int axis, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! method to calibrate a number of axis. The axis' numbers are given in the axis vector
        virtual ito::RetVal calib(const QVector<int> axis, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! method to set the origin of one axis to the current position
        virtual ito::RetVal setOrigin(const int axis, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! method to set the origin of a number of axis to their current positions. The axis' numbers are given in the axis vector
        virtual ito::RetVal setOrigin(const QVector<int> axis, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! retrieve the status of the actuator
        virtual ito::RetVal getStatus(QSharedPointer<QVector<int> > status, ItomSharedSemaphore *waitCond) = 0;
        //! retrieve the status of one axis of the actuator. By default, this method uses the general implementation of getStatus and returns the requested axis only.
        virtual ito::RetVal getStatus(const int axis, QSharedPointer<int> status, ItomSharedSemaphore *waitCond);
        //! read the position of one axis
        virtual ito::RetVal getPos(const int axis, QSharedPointer<double> pos, ItomSharedSemaphore *waitCond) = 0;
        //! read the position of a number of axis. The axis' numbers are given in the axis vector
        virtual ito::RetVal getPos(const QVector<int> axis, QSharedPointer<QVector<double> > pos, ItomSharedSemaphore *waitCond) = 0;
        //! move a single axis to a new absolute position
        virtual ito::RetVal setPosAbs(const int axis, const double pos, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! move a number of axis to a new absolute position. The axis' numbers are given in the axis vector
        virtual ito::RetVal setPosAbs(const QVector<int> axis, QVector<double> pos, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! increment/decrement a single axis by position value
        virtual ito::RetVal setPosRel(const int axis, const double pos, ItomSharedSemaphore *waitCond = NULL) = 0;
        //! increment/decrement a number of axis by position values. The axis' numbers are given in the axis vector
        virtual ito::RetVal setPosRel(const QVector<int> axis, QVector<double> pos, ItomSharedSemaphore *waitCond = NULL) = 0;

        //! overload this function to update the current status and position values, followed by calling sendStatusUpdate and/or sendTargetUpdate
        virtual ito::RetVal requestStatusAndPosition(bool sendCurrentPos, bool sendTargetPos);
    };

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @class AddInAlgo
    *   @brief base class for all "algorithm" plugin classes
    *
    *   This class is one step further down the line from \ref AddInBase. "Algorithm" plugins must be derived from this class which
    *   is derived from AddInBase. Compared to \ref AddInDataIO and \ref AddInActuator the AddInAlgo class is fairly simple. It does
    *   not have an init function or a close function. In the algo base class at the moment no further methods or variables are declared -
    *   it serves more as an organisation class, putting all actual plugins to the same level of inheritance.
    */
    class ITOMCOMMONQT_EXPORT AddInAlgo : public AddInBase
    {
        Q_OBJECT

    private:
        Q_DISABLE_COPY(AddInAlgo)

        QScopedPointer<AddInAlgoPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
        Q_DECLARE_PRIVATE(AddInAlgo);

    public:
        typedef ito::RetVal (*t_filter)     (QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        typedef ito::RetVal (*t_filterExt)  (QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut, QSharedPointer<ito::FunctionCancellationAndObserver> observer);
        typedef QWidget*    (*t_algoWidget) (QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::RetVal &retValue);
        typedef ito::RetVal (*t_filterParam)(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

        //!< possible categories for filter or widget-methods
        enum tAlgoCategory
        {
            catNone = 0x0000, //!< default: no category
            catDiskIO = 0x0001, //!< category for saving or loading data from hard drive
            catAnalyseDataObject = 0x0002, //!< category for methods analysing data objects
            catPlotDataObject = 0x0004 //!< category for methods plotting data objects
        };

        //!< possible algorithm interfaces
        enum tAlgoInterface
        {
            iNotSpecified = 0x0000, //!< default: filter or widget does not fit to any interface
            iReadDataObject = 0x0001, //!< interface for loading content of files into a data object
            iWriteDataObject = 0x0002, //!< interface for saving data object to file
            iReadPointCloud = 0x0004, //!< interface for loading content of files into a point cloud
            iWritePointCloud = 0x0008, //!< interface for saving point cloud to file
            iReadPolygonMesh = 0x0010, //!< interface for loading content of files into a polygon mesh
            iWritePolygonMesh = 0x0020, //!< interface for saving polygon mesh to file
            iPlotSingleObject = 0x0040  //!< interface for ploting dataObjects via the GUI
        };

        //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
        //values are always accessible.
        Q_ENUM(tAlgoCategory)
        Q_ENUM(tAlgoInterface)

        //! container for publishing filters provided by any plugin
        class FilterDef
        {
        public:
            //!< empty, default constructor
            FilterDef() :
                m_filterFunc(NULL),
                m_paramFunc(NULL),
                m_pBasePlugin(NULL),
                m_category(ito::AddInAlgo::catNone),
                m_interface(ito::AddInAlgo::iNotSpecified)
            {}

            //!< constructor with all necessary arguments.
            FilterDef(AddInAlgo::t_filter filterFunc, AddInAlgo::t_filterParam filterParamFunc,
                    QString description = QString(), ito::AddInAlgo::tAlgoCategory category = ito::AddInAlgo::catNone,
                    ito::AddInAlgo::tAlgoInterface interf = ito::AddInAlgo::iNotSpecified,
                    QString interfaceMeta = QString()) :
                m_filterFunc(filterFunc),
                m_paramFunc(filterParamFunc),
                m_pBasePlugin(NULL),
                m_description(description),
                m_category(category),
                m_interface(interf),
                m_interfaceMeta(interfaceMeta)
            {}

            virtual ~FilterDef() {}

            t_filter m_filterFunc;                      //!< function pointer (unbounded, static) for filter-method
            t_filterParam m_paramFunc;                  //!< function pointer (unbounded, static) for filter's default parameter method
            ito::AddInInterfaceBase *m_pBasePlugin;     //!< interface (factory) instance of this plugin (will be automatically filled)
            QString m_name;                             //!< name of filter
            QString m_description;                      //!< description of filter
            ito::AddInAlgo::tAlgoCategory m_category;   //!< category, filter belongs to (default: catNone)
            ito::AddInAlgo::tAlgoInterface m_interface; //!< algorithm interface, filter fits to (default: iNotSpecified)
            QString m_interfaceMeta;                    //!< meta information if required by algorithm interface
        private:
            FilterDef(const FilterDef & /*p*/); //disable copy constructor
        };

        //! extended FilterDef (derived from FilterDef) with a filterFunc of type f_filterExt instead of t_filter. This method has an additional argument of type FunctionCancellationAndObserver
        class FilterDefExt : public FilterDef
        {
        public:
            //!< empty, default constructor
            FilterDefExt() :
                FilterDef(),
                m_filterFuncExt(NULL)
            {}

            //!< constructor with all necessary arguments.
            FilterDefExt(AddInAlgo::t_filterExt filterFuncExt, AddInAlgo::t_filterParam filterParamFunc,
                    QString description = QString(), ito::AddInAlgo::tAlgoCategory category = ito::AddInAlgo::catNone,
                    ito::AddInAlgo::tAlgoInterface interf = ito::AddInAlgo::iNotSpecified,
                    QString interfaceMeta = QString(), bool hasStatusInfo = true, bool isCancellable = true) :
                FilterDef(NULL, filterParamFunc, description, category, interf, interfaceMeta),
                m_filterFuncExt(filterFuncExt),
                m_hasStatusInformation(hasStatusInfo),
                m_isCancellable(isCancellable)
            {}

            virtual ~FilterDefExt() {}

            t_filterExt m_filterFuncExt;                      //!< extended function pointer (unbounded, static) for filter-method
            bool m_hasStatusInformation;                      //!< true, if filter updates status information to the optional observer
            bool m_isCancellable;                             //!< true, if filter listens to a possible interrupt flag in the optional observer and cancels the execution if set

        private:
            FilterDefExt(const FilterDefExt & /*p*/); //disable copy constructor
        };

        //! container for publishing widgets provided by any plugin
        class AlgoWidgetDef
        {
        public:
            //!< empty, default constructor
            AlgoWidgetDef() :
                m_widgetFunc(NULL),
                m_paramFunc(NULL),
                m_pBasePlugin(NULL),
                m_category(ito::AddInAlgo::catNone),
                m_interface(ito::AddInAlgo::iNotSpecified)
            {}

            //!< constructor with all necessary arguments.
            AlgoWidgetDef(AddInAlgo::t_algoWidget algoWidgetFunc, AddInAlgo::t_filterParam algoWidgetParamFunc, QString description = QString(), ito::AddInAlgo::tAlgoCategory category = ito::AddInAlgo::catNone, ito::AddInAlgo::tAlgoInterface interf = ito::AddInAlgo::iNotSpecified, QString interfaceMeta = QString()) :
                m_widgetFunc(algoWidgetFunc),
                m_paramFunc(algoWidgetParamFunc),
                m_pBasePlugin(NULL),
                m_description(description),
                m_category(category),
                m_interface(interf),
                m_interfaceMeta(interfaceMeta)
            {}

            virtual ~AlgoWidgetDef() {}    //!< destructor

            t_algoWidget m_widgetFunc;    //!< function pointer (unbounded, static) for widget-method
            t_filterParam m_paramFunc;    //!< function pointer (unbounded, static) for widget's default parameter method
            ito::AddInInterfaceBase *m_pBasePlugin;        //!< interface (factory) instance of this plugin (will be automatically filled)
            QString m_name;                //!< name of widget
            QString m_description;        //!< description of widget
            ito::AddInAlgo::tAlgoCategory m_category;    //!< category, widget belongs to (default: catNone)
            ito::AddInAlgo::tAlgoInterface m_interface; //!< algorithm interface, widget fits to (default: iNotSpecified)
            QString m_interfaceMeta;    //!< meta information if required by algorithm interface

        private:
            AlgoWidgetDef(const AlgoWidgetDef & /*p*/); //disable copy constructor
        };

        ito::RetVal getFilterList(QHash<QString, FilterDef *> &fList) const;
        ito::RetVal getAlgoWidgetList(QHash<QString, AlgoWidgetDef *> &awList) const;
        ito::RetVal rejectFilter(const QString &name);
        ito::RetVal rejectAlgoWidget(const QString &name);

    protected:
        virtual ~AddInAlgo();
        AddInAlgo();
        QHash<QString, FilterDef *> m_filterList;
        QHash<QString, AlgoWidgetDef *> m_algoWidgetList;

        //! small check and cleaning of all parameter vectors (can be used in filterParam methods)
        static ito::RetVal prepareParamVectors(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

    public Q_SLOTS:
        virtual ito::RetVal getParam(QSharedPointer<ito::Param> /*val*/, ItomSharedSemaphore * /*waitCond*/ = NULL) { return ito::retOk; }
        virtual ito::RetVal setParam(QSharedPointer<ito::ParamBase> /*val*/, ItomSharedSemaphore * /*waitCond*/ = NULL) { return ito::retOk; }
    };

    //----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

//! must be out of namespace ito, otherwise it results in a strange compiler error (template ...)
Q_DECLARE_INTERFACE(ito::AddInInterfaceBase, ito_AddInInterface_CurrentVersion /*"ito.AddIn.InterfaceBase/4"*/)



#endif
