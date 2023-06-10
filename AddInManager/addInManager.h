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

#ifndef ADDINMANAGER_H
#define ADDINMANAGER_H

#include "addInMgrDefines.h"

#include "../common/addInInterface.h"
#include "../DataObject/dataobj.h"
#ifdef USEPCL
#define ITOM_POINTCLOUDLIBRARY 1
#else
#define ITOM_POINTCLOUDLIBRARY 0
#endif
#if ITOM_POINTCLOUDLIBRARY > 0
    #include "../PointCloud/pclStructures.h"
#endif

#include <qobject.h>
#include <qscopedpointer.h>

//#include "algoInterfaceValidator.h"



#if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    // forward declaration for private AddInManager functions, declared and implemented
    // in addInManagerPrivate.cpp
    class AddInManagerPrivate;

    class PlugInModel; //forward declaration
    class AlgoInterfaceValidator; //forward declaration

    /** @class AddInManager
    *   @brief class for AddIn management
    *
    *   This class is internally used for plugin handling, i.e. detecting available plugins which can be loaded,
    *   maintaining a list (widget \ref AddInModel) of available and loaded plugins, loading and unloading of plugins.
    *   The plugins themselfs are based on the addInInterface, declared in \ref addInInterface. The AddInManager is
    *   implemented as singleton class because it must exist only one instance of it (which would also be possible using
    *   a static class) but which also does a clean up of the instantiated plugin classes at program exit.
    */
    class ADDINMGR_EXPORT AddInManager : public QObject
    {
        Q_OBJECT

        public:
            //!> create a new instance of AddInManager as singleton class or returns the recently opened instance
            static AddInManager* createInstance(QString itomSettingsFile, void **apiFuncsGraph, QObject *mainWindow = NULL, QObject *mainApplication = NULL);

            //!> close the singleton class of AddInManager
            static RetVal closeInstance();

            //!> returns the instantiated singleton class or NULL if it has not been loaded, yet
            static AddInManager* instance() { return staticInstance; }

            //!> scan directory at path for loadable plugins, if checkQCoreApp is 1 it is checked wether an instance of Q(Core)Application
            //!> is already running, which is necessary for multithreading, i.e. asyncronous use of plugins. If no instance is found a new
            //!> one is created
            const RetVal scanAddInDir(const QString &path, const int checkQCoreApp = 1);

            //!> return list of all dataIO plugins
            const QList<QObject *> * getDataIOList(void) const;

            //!> return list of actuator plugins
            const QList<QObject *> * getActList(void)    const;

            //!> return list of algorithm plugins
            const QList<QObject *> * getAlgList(void)    const;

            //!> return list of all filters
            const QHash<QString, ito::AddInAlgo::FilterDef *> * getFilterList(void) const;

            //!> return list of algorithm widgets
            const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> * getAlgoWidgetList(void) const;

            //!> return parameters used for / within a filter based on the filter function pointer
            const ito::FilterParams* getHashedFilterParams(ito::AddInAlgo::t_filterParam filterParam) const;

            //!> return status of all plugins
            const QList<struct PluginLoadStatus> getPluginLoadStatus() const;

            //!>
            const AlgoInterfaceValidator * getAlgoInterfaceValidator(void) const;

            //!>
            const ito::AddInAlgo::AlgoWidgetDef * getAlgoWidgetDef( QString algoWidgetName, QString algoPluginName = QString() );

            //!> returns pointer to plugin model, usable in a model/view relationship
            PlugInModel * getPluginModel(void);

            //!> Reload plugin library (dll)
            const RetVal reloadAddIn(const QString &name);

            //!> returns the overall number of loaded plugins
            int getTotalNumAddIns(void) const;

            //!> get plugin pointer
            void * getAddInPtr(const int itemNum);

            //!> get index in plugin list based on plugin pointer
            int getItemIndexInList(const void *item);

            //!> forces the plugin model to be updated
            void updateModel(void);

            //!> get parameters for plugin initialization, based on plugin number \ref getItemIndexList
            const RetVal getInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt);

            //!> returns the a string containing the about string based on plugin number and type
            const RetVal getAboutInfo(const QString &name, QString &versionString);

            //!> return plugin information based on plugin number and type
            const RetVal getPluginInfo(const QString &name, int &pluginType, int &pluginNum, int &version, QString &typeString, QString &author, QString &description, QString &detaildescription, QString &license, QString &about);

            //!> increment plugin reference counter, use e.g. when making a copy of the plugin pointer to avoid plugin being closed while still holding a reference
            const RetVal incRef(ito::AddInBase *plugin);

            //!> decrement plugin reference counter, use to ensure proper closing / deletion of plugin after incrementing reference counter
            const RetVal decRef(ito::AddInBase **plugin);

            //!> set plugin time outs (initialization / closing and general)
            const RetVal setTimeOuts(const int initClose, const int general);

            //!> pass main window pointer to addInManager, which is used for the construction / displaying of plugin widgets
            const RetVal setMainWindow(QObject *mainWindow);

            //!> check if given instance of plugin still reacts
            bool isPluginInstanceDead(const ito::AddInBase *plugin) const;

            //!> return list of filter matching the passed interface
            const QList<ito::AddInAlgo::FilterDef *> getFilterByInterface(ito::AddInAlgo::tAlgoInterface iface, const QString tag = QString()) const;

            //!> return list of filter matching the passed category
            const QList<ito::AddInAlgo::FilterDef *> getFiltersByCategory(ito::AddInAlgo::tAlgoCategory cat) const;

            //!> return list of filter matching the passed interface and category
            const QList<ito::AddInAlgo::FilterDef *> getFilterByInterfaceAndCategory(ito::AddInAlgo::tAlgoInterface iface, ito::AddInAlgo::tAlgoCategory cat, const QString tag = QString()) const;

            //!> return itomApi functions pointer (e.g. used in plugins to call itom api functions)
            void **getItomApiFuncsPtr(void);

        private:
            //!> private constructor. Use createInstance to get an instance of AddInManager
            AddInManager(QString itomSettingsFile, void **apiFuncsGraph, QObject *mainWindow = NULL, QObject *mainApplication = NULL);

            ~AddInManager(void);

            static AddInManager *staticInstance; //!> static instance pointer
            QScopedPointer<AddInManagerPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
            Q_DECLARE_PRIVATE(AddInManager);
            Q_DISABLE_COPY(AddInManager);

        signals:
            //!> show plugin load splash screen
            void splashLoadMessage(const QString &message);

        public slots:
            //!> show plugin configuration dialog, don't call this method if no Qt gui application is available.
            ito::RetVal showConfigDialog(ito::AddInBase *addin, ItomSharedSemaphore *waitCond = NULL);

            //!> show plugin dock widget, don't call this method if no Qt gui application is available.
            ito::RetVal showDockWidget(ito::AddInBase *addin, int visible, ItomSharedSemaphore *waitCond = NULL);

            //!> initialize dataIO plugin based on number and name
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInDataIO **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);

            //!> initialize actuator plugin based on number and name
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInActuator **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);

            //!> initialize algorithm plugin based on number and name
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInAlgo **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);

            //!> close passed plugin
            ito::RetVal closeAddIn(ito::AddInBase *addIn, ItomSharedSemaphore *aimWait = NULL);

            //!> interrupts all active actuator instances
            ito::RetVal interruptAllActuatorInstances(ItomSharedSemaphore *aimWait = NULL);
    };
} //namespace ito
#endif // #if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL)
#endif // #if ADDINMANAGER_H
