/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "../global.h"

#include "../common/addInInterface.h"
#include "../models/PlugInModel.h"
#include "algoInterfaceValidator.h"

//#include <qcoreapplication.h>
#include <qmetatype.h>
#include <qvector.h>
#include <qsharedpointer.h>
#include <qhash.h>
#include <qtimer.h>
#include <qtranslator.h>

// in the invokeMethod function parameters are passed with the Q_ARG macro, which works only with preregistered data types
// the registration of "new" data types is done in two steps. First they are declared with the Q_DECLARE_METATYPE macro
// second they are registered for use with the function qRegisterMetaType. For the data types used within the iTom plugin
// system this is done here and in the constructor of the AddInManager
Q_DECLARE_METATYPE(ItomSharedSemaphore *)
Q_DECLARE_METATYPE(const char *)
Q_DECLARE_METATYPE(const char **)
Q_DECLARE_METATYPE(char *)
Q_DECLARE_METATYPE(char **)
Q_DECLARE_METATYPE(double)
Q_DECLARE_METATYPE(double *)
//Q_DECLARE_METATYPE(const double)
Q_DECLARE_METATYPE(const double *)
Q_DECLARE_METATYPE(int *)
Q_DECLARE_METATYPE(const int *)
//Q_DECLARE_METATYPE(int)
Q_DECLARE_METATYPE(ito::AddInInterfaceBase *)
Q_DECLARE_METATYPE(ito::AddInBase *)
Q_DECLARE_METATYPE(ito::AddInBase **)
Q_DECLARE_METATYPE(ito::AddInDataIO **)
Q_DECLARE_METATYPE(ito::AddInActuator **)
Q_DECLARE_METATYPE(ito::AddInAlgo **)
//Q_DECLARE_METATYPE(ito::ActuatorAxis **)
Q_DECLARE_METATYPE(ito::RetVal *)
Q_DECLARE_METATYPE(ito::RetVal)
Q_DECLARE_METATYPE(const void*)
Q_DECLARE_METATYPE(QVector<ito::Param> *)
Q_DECLARE_METATYPE(QVector<ito::ParamBase> *)
Q_DECLARE_METATYPE(QVector<int>)
Q_DECLARE_METATYPE(QVector<double>)

Q_DECLARE_METATYPE(QSharedPointer<double> );
Q_DECLARE_METATYPE(QSharedPointer<int>);
Q_DECLARE_METATYPE(QSharedPointer<QVector<double> >);
Q_DECLARE_METATYPE(QSharedPointer<char>);
Q_DECLARE_METATYPE(QSharedPointer<ito::Param>);
Q_DECLARE_METATYPE(QSharedPointer<ito::ParamBase>);

Q_DECLARE_METATYPE(QVector<QSharedPointer<ito::ParamBase> >);
Q_DECLARE_METATYPE(StringMap);

//Q_DECLARE_METATYPE(ito::PCLPointCloud)
//Q_DECLARE_METATYPE(ito::PCLPoint)
//Q_DECLARE_METATYPE(ito::PCLPolygonMesh)

namespace ito
{
    

    /** @class AddInManager
    *   @brief class for AddIn management
    *
    *   This class is internally used for plugin handling, i.e. detecting available plugins which can be loaded,
    *   maintaining a list (widget \ref AddInModel) of available and loaded plugins, loading and unloading of plugins.
    *   The plugins themselfs are based on the addInInterface, declared in \ref addInInterface. The AddInManager is
    *   implemented as singleton class because it must exist only one instance of it (which would also be possible using
    *   a static class) but which also does a clean up of the instantiated plugin classes at program exit.
    */
    class AddInManager : public QObject
    {
        Q_OBJECT

        public:
            static AddInManager * getInstance(void);
            static RetVal closeInstance(void);
            const RetVal scanAddInDir(const QString &path);
            inline const QList<QObject *> * getDataIOList(void) const { return &m_addInListDataIO; }
            inline const QList<QObject *> * getActList(void)    const { return &m_addInListAct; }
            inline const QList<QObject *> * getAlgList(void)    const { return &m_addInListAlgo; }
            inline const QHash<QString, ito::AddInAlgo::FilterDef *>     * getFilterList(void)     const { return &m_filterList;     }
            inline const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> * getAlgoWidgetList(void) const { return &m_algoWidgetList; }
            const ito::FilterParams* getHashedFilterParams(ito::AddInAlgo::t_filterParam filterParam) const;
            const QList<PluginLoadStatus> getPluginLoadStatus() const { return m_pluginLoadStatus; }
            inline const AlgoInterfaceValidator * getAlgoInterfaceValidator(void) const { return m_algoInterfaceValidator; }

            const ito::AddInAlgo::AlgoWidgetDef * getAlgoWidgetDef( QString algoWidgetName, QString algoPluginName = QString() );

            inline PlugInModel * getPluginModel(void) { return &m_plugInModel; }
            const RetVal reloadAddIn(const QString &name);
            inline int getNumTotItems(void) const { return m_addInListDataIO.size() + m_addInListAct.size() + m_addInListAlgo.size(); }
            void * getAddInPtr(const int itemNum)
            {
                int num = itemNum;

                if (num < m_addInListAct.size())
                {
                    return (void *)m_addInListAct[num];
                }
                else if (num -= m_addInListAct.size(), num < m_addInListAlgo.size())
                {
                    return (void *)m_addInListAlgo[num];
                }
                else if (num -= m_addInListAlgo.size(), num < m_addInListDataIO.size())
                {
                    return (void *)m_addInListDataIO[num];
                }
                else
                {
                    return NULL;
                }
            }
            int getItemNum(const void *item)
            {
                int num = 0;
                if ((num = m_addInListAct.indexOf((QObject*)item)) != -1)
                {
                    return num;
                }
                else if ((num = m_addInListAlgo.indexOf((QObject*)item)) != -1)
                {
                    return num + m_addInListAct.size();
                }
                else if ((num = m_addInListDataIO.indexOf((QObject*)item)) != -1)
                {
                    return num + m_addInListAct.size() + m_addInListAlgo.size();
                }
                else
                {
                    return -1;
                }
            }
            int getItemIndexInList(const void *item)
            {
                int num = 0;
                if ((num = m_addInListAct.indexOf((QObject*)item)) != -1)
                {
                    return num;
                }
                else if ((num = m_addInListAlgo.indexOf((QObject*)item)) != -1)
                {
                    return num;
                }
                else if ((num = m_addInListDataIO.indexOf((QObject*)item)) != -1)
                {
                    return num;
                }
                else
                {
                    return -1;
                }
            }
            int getPluginNum(const QString &name, ito::AddInInterfaceBase *&addIn)
            {

                addIn = NULL;
//                int num = -1;
                for (int n = 0; n < m_addInListAct.size(); n++)
                {
                    if ((m_addInListAct[n])->objectName() == name)
                    {
                        addIn = (ito::AddInInterfaceBase*)m_addInListAct[n];
                        return n;
                    }
                }
                for (int n = 0; n < m_addInListDataIO.size(); n++)
                {
                    if ((m_addInListDataIO[n])->objectName() == name)
                    {
                        addIn = (ito::AddInInterfaceBase*)m_addInListDataIO[n];
                        return n;
                    }
                }
                for (int n = 0; n < m_addInListAlgo.size(); n++)
                {
                    if ((m_addInListAlgo[n])->objectName() == name)
                    {
                        addIn = (ito::AddInInterfaceBase*)m_addInListAlgo[n];
                        return n;
                    }
                }
                return -1;
            }

//            inline void updateModel(void) { m_addInModel.update(); }
            inline void updateModel(void) { m_plugInModel.update(); }
            const RetVal saveParamVals(ito::AddInBase *plugin);
            const RetVal loadParamVals(ito::AddInBase *plugin);
            const RetVal getInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt);
            const RetVal getPlugInInfo(const QString &name, int *pluginType, int *pluginNum, char **pluginTypeString, char ** author, char ** discription, char ** detaildiscription, int * version);
            const RetVal incRef(ito::AddInBase *plugin);
            const RetVal decRef(ito::AddInBase **plugin);

            bool isPluginInstanceDead(const ito::AddInBase *plugin) const;

            const QList<ito::AddInAlgo::FilterDef *> getFilterByInterface(ito::AddInAlgo::tAlgoInterface iface, const QString tag = QString::Null()) const;
            const QList<ito::AddInAlgo::FilterDef *> getFiltersByCategory(ito::AddInAlgo::tAlgoCategory cat) const;
            const QList<ito::AddInAlgo::FilterDef *> getFilterByInterfaceAndCategory(ito::AddInAlgo::tAlgoInterface iface, ito::AddInAlgo::tAlgoCategory cat, const QString tag = QString::Null()) const;

        protected:

            RetVal initDockWidget(const ito::AddInBase *addIn);
            RetVal loadAddIn(QString &filename);

            RetVal loadAddInDataIO(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus);
            RetVal loadAddInActuator(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus);
            RetVal loadAddInAlgo(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus);

            RetVal registerPluginAsDeadPlugin(ito::AddInBase *addIn);


        private:
            AddInManager(void);
            AddInManager(AddInManager  &/*copyConstr*/) : QObject() {}
            ~AddInManager(void);
            QVector<QTranslator*> m_Translator;

            static AddInManager *m_pAddInManager;
            static QList<QObject *> m_addInListDataIO;
            static QList<QObject *> m_addInListAct;
            static QList<QObject *> m_addInListAlgo;
            static QHash<QString, ito::AddInAlgo::FilterDef *> m_filterList;
            static QMultiHash<QString, ito::AddInAlgo::FilterDef *> m_filterListInterfaceTag; //hash value is "{interface-number}_{tag}"
            static QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> m_algoWidgetList;
            static QHash<void*, ito::FilterParams *> filterParamHash;
            static QList<PluginLoadStatus> m_pluginLoadStatus;

            AlgoInterfaceValidator *m_algoInterfaceValidator;

            //AddInModel m_addInModel;
            PlugInModel m_plugInModel;

            QList< QWeakPointer<ito::AddInBase> > m_deadPlugins;
            QTimer m_deadPluginTimer;
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

        public slots:
            ito::RetVal showConfigDialog(ito::AddInBase *addin);
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInDataIO **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInActuator **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);
            ito::RetVal initAddIn(const int pluginNum, const QString &name, ito::AddInAlgo **addIn, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);

            ito::RetVal closeAddIn(ito::AddInBase **addIn, ItomSharedSemaphore *aimWait = NULL);

        private slots:
            RetVal closeDeadPlugins();
    };
} //namespace ito

#endif
