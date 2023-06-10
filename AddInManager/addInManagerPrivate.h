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

#ifndef ADDINMANAGERIMPL_H
#define ADDINMANAGERIMPL_H

#include "algoInterfaceValidator.h"
#include "../common/sharedFunctionsQt.h"
#include "../common/addInInterface.h"
#include "pluginModel.h"

#include <qobject.h>
#include <qtimer.h>
#include <qlist.h>
#include <qhash.h>
#include <qstring.h>
#include <qglobal.h>
#include <qfileinfo.h>
#include <qsettings.h>
#include <QDir>
#include <QDirIterator>
#include <QTranslator>
#include <qmainwindow.h>
#include <qpair.h>
#include <qpointer.h>
#include <qsharedpointer.h>


namespace ito
{

class AddInManager;

class AddInManagerPrivate : public QObject
{
    Q_OBJECT

    Q_DECLARE_PUBLIC(AddInManager);
    public:
        AddInManagerPrivate(AddInManager* addInMgr);
        ~AddInManagerPrivate();

        // two helper methods for Qt < 5.12. They can be removed once the minimum Qt version is 5.12.
        static QString regExpAnchoredPattern(const QString& expression);
        static QString wildcardToRegularExpression(const QString &pattern);

    protected:
        AddInManager* const q_ptr;

    private:
        QVector<QTranslator*> m_Translator;

        QList<QObject *> m_addInListDataIO;
        QList<QObject *> m_addInListAct;
        QList<QObject *> m_addInListAlgo;
        QHash<QString, ito::AddInAlgo::FilterDef *> m_filterList;
        QMultiHash<QString, ito::AddInAlgo::FilterDef *> m_filterListInterfaceTag; //hash value is "{interface-number}_{tag}"
        QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> m_algoWidgetList;
        QHash<void*, ito::FilterParams *> filterParamHash;
        QList<PluginLoadStatus> m_pluginLoadStatus;
        QObject *m_pMainWindow;
        QObject *m_pMainApplication;

        AlgoInterfaceValidator *m_algoInterfaceValidator;
        PlugInModel *m_plugInModel;

        QCoreApplication *m_pQCoreApp;
        QList< QPointer<ito::AddInBase> > m_deadPlugins;
        QTimer m_deadPluginTimer;
        int m_timeOutInitClose;
        int m_timeOutGeneral;

        int getItemNum(const void *item);
        int getPluginNum(const QString &name, ito::AddInInterfaceBase *&addIn);
        const RetVal saveParamVals(ito::AddInBase *plugin);
        const RetVal loadParamVals(ito::AddInBase *plugin);
        void incRefParamPlugins(ito::AddInBase *ai, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt);
        ito::RetVal decRefParamPlugins(ito::AddInBase *ai);
        const RetVal decRef(ito::AddInBase **plugin);
        const RetVal closeAddIn(AddInBase *addIn, ItomSharedSemaphore *aimWait = NULL);

        template<typename _Tp> const RetVal initAddInActuatorOrDataIO(
            bool actuatorNotDataIO,
            const int pluginNum, const QString &name,
            _Tp** addIn, QVector<ito::ParamBase> *paramsMand,
            QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams,
            ItomSharedSemaphore *aimWait = NULL);

        const RetVal initAddInAlgo(
            const int pluginNum, const QString &name, ito::AddInAlgo **addIn,
            QVector<ito::ParamBase> * paramsMand, QVector<ito::ParamBase> * paramsOpt,
            bool autoLoadPluginParams, ItomSharedSemaphore *aimWait = NULL);

    protected:
        void setItomProperties(void *propPtr) {};
        RetVal initDockWidget(const ito::AddInBase *addIn);
        RetVal loadAddIn(QString &filename);

        RetVal loadAddInDataIO(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus);
        RetVal loadAddInActuator(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus);
        RetVal loadAddInAlgo(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus);

        RetVal registerPluginAsDeadPlugin(ito::AddInBase *addIn);

    private slots:
        RetVal closeDeadPlugins();
        void propertiesChanged();
};

} //end namespace ito



#endif // ADDINMANAGERIMPL_H
