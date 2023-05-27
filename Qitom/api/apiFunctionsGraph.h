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

#ifndef APIFUNCTIONSGRAPH_H
#define APIFUNCTIONSGRAPH_H

#include "../../common/apiFunctionsGraphInc.h"

#include "../../common/sharedStructures.h"
#include "../../common/sharedStructuresGraphics.h"
#include "../../common/itomPlotHandle.h"

#include <qvariant.h>
#include <qpointer.h>
#include <qsharedpointer.h>

namespace ito
{
    class apiFunctionsGraph
    {
        public:
            apiFunctionsGraph();
            ~apiFunctionsGraph();

            static ito::RetVal mnumberOfColorBars(int &number);
            static ito::RetVal mgetColorBarName(const QString &name, ito::ItomPalette &palette);
            static ito::RetVal mgetColorBarIdx(const int number, ito::ItomPalette &palette);
            //static ito::RetVal mgetFigure(ito::uint32 &UID, const QString plugin, QWidget **newFigure);
            static ito::RetVal mgetPluginList(const ito::PluginInfo &requirements, QHash<QString, ito::PluginInfo> &pluginList, const QString preference);
            static ito::RetVal mconnectLiveData(QObject *liveDataSource, QObject *liveDataView);
            static ito::RetVal mstartLiveData(QObject* liveDataSource, QObject *liveDataView);
            static ito::RetVal mstopLiveData(QObject* liveDataSource, QObject *liveDataView);
            static ito::RetVal mdisconnectLiveData(QObject* liveDataSource, QObject *liveDataView);
            static ito::RetVal mgetColorBarIdxFromName(const QString &name, ito::int32 & index);

            static ito::RetVal mgetFigure(const QString &figCategoryName, const QString &figClassName, ito::uint32 &UID, QWidget **figure, QWidget *parent = NULL);
            static QVariant mgetFigureSetting(const QObject *figureClass, const QString &key, const QVariant &defaultValue = QVariant(), ito::RetVal *retval = NULL);

            static ito::RetVal mgetPluginWidget(char* algoWidgetFunc, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QPointer<QWidget> *widget);
            static ito::RetVal mgetFigureUIDByHandle(QObject *figure, ito::uint32 &figureUID);
            static ito::RetVal mgetPlotHandleByID(const ito::uint32 &figureUID, ito::ItomPlotHandle &plotHandle);

            //! function called by apiSendParamToPyWorkspace
            static ito::RetVal sendParamToPyWorkspaceThreadSafe(const QString &varname, const QSharedPointer<ito::ParamBase> &value);
            //! function called by apiSendParamsToPyWorkspace
            static ito::RetVal sendParamsToPyWorkspaceThreadSafe(const QStringList &varnames, const QVector<QSharedPointer<ito::ParamBase> > &values);

            static ito::RetVal mConnectToOutputAndErrorStream(const QObject *receiver, const char *method, ito::tStreamMessageType messageType);
            static ito::RetVal mDisconnectFromOutputAndErrorStream(const QObject *receiver, const char *method, ito::tStreamMessageType messageType);

        private:
    };
}

#endif // APIFUNCTIONSGRAPH_H
