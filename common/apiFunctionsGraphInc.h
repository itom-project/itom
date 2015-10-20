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

#ifndef APIFUNCTIONSGRAPHINC_H
#define APIFUNCTIONSGRAPHINC_H
#ifndef Q_MOC_RUN

//#include "sharedStructures.h"
//#include "sharedStructuresGraphics.h"

namespace ito
{

    #if defined(ITOM_IMPORT_PLOTAPI) && !defined(ITOM_CORE)
        void **ITOM_API_FUNCS_GRAPH;
    #else
        extern void **ITOM_API_FUNCS_GRAPH;
    #endif

    /**
    * \defgroup ITOM_API_GRAPH itom figure and plot API
    *
    * \brief The itom plot and figure API contains a bunch of functions that can be called by the core application itom as 
    * well as by every designer plugin.
    *
    * Every function is given by a certain preprocessor definition that describes
    * the return value, parameters and function name of the method to call. Each preprocessor definition is redirected
    * to a function pointer that becomes valid at runtime. The initialization of these function pointers in any plugins
    * is an automatic process by itom, called when loading the plugin.
    *
    * How to read the following definitions?
    *
    * Basically the first word after the #define word is the method to call. After the first star the return value
    * follows (the star into brackets is not part of the return value). Then there is a list of different parameters
    * for this method.
    *
    * \{
    */

    #define apiPaletteGetNumberOfColorBars \
        (*(ito::RetVal (*)(int &number)) ito::ITOM_API_FUNCS_GRAPH[0])

    #define apiPaletteGetColorBarName \
        (*(ito::RetVal (*)(const QString &name, ito::ItomPalette &palette)) ito::ITOM_API_FUNCS_GRAPH[1])

    #define apiPaletteGetColorBarIdx \
        (*(ito::RetVal (*)(const int index, ito::ItomPalette &palette)) ito::ITOM_API_FUNCS_GRAPH[2])

    #define apiGetFigure \
        (*(ito::RetVal (*)(const QString &figCategoryName, const QString &figClassName, ito::uint32 &UID, QWidget **figure, QWidget *parent)) ito::ITOM_API_FUNCS_GRAPH[3])

    #define apiGetPluginList \
        (*(ito::RetVal (*)(const ito::pluginInfo requirements, QHash<QString, ito::pluginInfo> &pluginList, const QString preference)) ito::ITOM_API_FUNCS_GRAPH[4])

    #define apiStartLiveData \
        (*(ito::RetVal (*)(QObject *liveDataSource, QObject *liveDataView)) ito::ITOM_API_FUNCS_GRAPH[5])

    #define apiStopLiveData \
        (*(ito::RetVal (*)(QObject *liveDataSource, QObject *liveDataView)) ito::ITOM_API_FUNCS_GRAPH[6])

    #define apiConnectLiveData \
        (*(ito::RetVal (*)(QObject *liveDataSource, QObject *liveDataView)) ito::ITOM_API_FUNCS_GRAPH[7])

    #define apiDisconnectLiveData \
        (*(ito::RetVal (*)(QObject *liveDataSource, QObject *liveDataView)) ito::ITOM_API_FUNCS_GRAPH[8])

    #define apiPaletteGetColorBarIdxFromName \
        (*(ito::RetVal (*)(const QString &name, ito::int32 &index)) ito::ITOM_API_FUNCS_GRAPH[9])

    #define apiGetFigureSetting \
        (*(QVariant (*)(const QObject *figureClass, const QString &key, const QVariant &defaultValue, ito::RetVal *retval)) ito::ITOM_API_FUNCS_GRAPH[10])

    #define apiGetPluginWidget \
        (*(ito::RetVal (*)(void *, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QPointer<QWidget>*widget)) ito::ITOM_API_FUNCS_GRAPH[11])

    #define apiGetFigureIDbyHandle \
        (*(ito::RetVal (*)(QObject *figure, ito::uint32 &figureUID)) ito::ITOM_API_FUNCS_GRAPH[12])

    #define apiGetItomPlotHandleByID \
        (*(ito::RetVal (*)(const ito::uint32 &figureUID, ito::ItomPlotHandle &plotHandle)) ito::ITOM_API_FUNCS_GRAPH[13])
    /** \} */

    /*#if defined(ITOM_IMPORT_PLOTAPI)
    static int importItomPlotApi(void** apiArray)
    {
        ito::ITOM_API_FUNCS_GRAPH = apiArray;
        return 0;
    }
    #endif*/

} //end namespace ito

#endif // Q_MOC_RUN
#endif // APIFUNCTIONSGRAPHINC_H
