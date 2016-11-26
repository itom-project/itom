/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "../DataObject/dataobj.h"
#if ITOM_POINTCLOUDLIBRARY > 0
    #include "../PointCloud/pclStructures.h"
#endif
#include "PlugInModel.h"
#include "algoInterfaceValidator.h"

#if !defined(Q_MOC_RUN) || defined(ADDINMGR_DLL) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    // forward declaration for private AddInManager functions, declared and implemented
    // in addInManager.cpp
    class AddInManagerPrivate; 

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
        //PLUGIN_ITOM_API

        public:
            AddInManager(QString itomSettingsFile, void **apiFuncsGraph, QObject *mainWindow = NULL, QObject *mainApplication = NULL);
            //AddInManager * getInstance(const void *mainWindow = NULL, const void *mainApplication = NULL);
            RetVal closeInstance(void);
            const RetVal scanAddInDir(const QString &path);
            const QList<QObject *> * getDataIOList(void) const;
            const QList<QObject *> * getActList(void)    const;
            const QList<QObject *> * getAlgList(void)    const;
            const QHash<QString, ito::AddInAlgo::FilterDef *>     * getFilterList(void)     const;
            const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> * getAlgoWidgetList(void) const;
            const ito::FilterParams* getHashedFilterParams(ito::AddInAlgo::t_filterParam filterParam) const;
            const QList<PluginLoadStatus> getPluginLoadStatus() const;
            const AlgoInterfaceValidator * getAlgoInterfaceValidator(void) const;

            const ito::AddInAlgo::AlgoWidgetDef * getAlgoWidgetDef( QString algoWidgetName, QString algoPluginName = QString() );

            PlugInModel * getPluginModel(void);
            const RetVal reloadAddIn(const QString &name);
            int getNumTotItems(void) const;
            void * getAddInPtr(const int itemNum);
            int getItemIndexInList(const void *item);

            void updateModel(void);
            const RetVal getInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt);
            const RetVal getPluginInfo(const QString &name, int &pluginType, int &pluginNum, int &version, QString &typeString, QString &author, QString &description, QString &detaildescription, QString &license, QString &about);
            const RetVal incRef(ito::AddInBase *plugin);
            const RetVal decRef(ito::AddInBase **plugin);
            const RetVal setTimeOuts(const int initClose, const int general);

            bool isPluginInstanceDead(const ito::AddInBase *plugin) const;

            const QList<ito::AddInAlgo::FilterDef *> getFilterByInterface(ito::AddInAlgo::tAlgoInterface iface, const QString tag = QString::Null()) const;
            const QList<ito::AddInAlgo::FilterDef *> getFiltersByCategory(ito::AddInAlgo::tAlgoCategory cat) const;
            const QList<ito::AddInAlgo::FilterDef *> getFilterByInterfaceAndCategory(ito::AddInAlgo::tAlgoInterface iface, ito::AddInAlgo::tAlgoCategory cat, const QString tag = QString::Null()) const;

            void **getItomApiFuncsPtr(void);

        private:
            AddInManager(AddInManager  &/*copyConstr*/);
            ~AddInManager(void);
/*
            //!< singleton nach: http://www.oop-trainer.de/Themen/Singleton.html
            class AddInSingleton
            {
                public:
                    ~AddInSingleton()
                    {
                        #pragma omp critical
                        {
                            if( AddInManager::m_pAddInManager != NULL)
                            {
                                delete AddInManager::m_pAddInManager;
                                AddInManager::m_pAddInManager = NULL;
                            }
                        }
                    }
            };
            friend class AddInSingleton;
*/
        signals:
            void splashLoadMessage(const QString &message, int alignment = Qt::AlignLeft, const QColor &color = Qt::black);

        public slots:
            ito::RetVal showConfigDialog(ito::AddInBase *addin, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal showDockWidget(ito::AddInBase *addin, int visible, ItomSharedSemaphore *waitCond = NULL);

            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInDataIO **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInActuator **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInAlgo **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);

            ito::RetVal closeAddIn(ito::AddInBase *addIn, ItomSharedSemaphore *aimWait = NULL);
    };
} //namespace ito   
#endif // #if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) 
#endif
