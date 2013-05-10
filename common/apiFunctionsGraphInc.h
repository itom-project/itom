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

//#include "sharedStructures.h"
//#include "sharedStructuresGraphics.h"

namespace ito
{

#if defined(ITOM_IMPORT_PLOTAPI) && !defined(ITOM_CORE)
    void **ITOM_API_FUNCS_GRAPH;
#else
    extern void **ITOM_API_FUNCS_GRAPH;
#endif


#define apiPaletteGetNumberOfColorBars \
	(*(ito::RetVal (*)(int &)) ito::ITOM_API_FUNCS_GRAPH[0])

#define apiPaletteGetColorBarName \
	(*(ito::RetVal (*)(const QString &, ito::ItomPalette &)) ito::ITOM_API_FUNCS_GRAPH[1])

#define apiPaletteGetColorBarIdx \
	(*(ito::RetVal (*)(const int, ito::ItomPalette &)) ito::ITOM_API_FUNCS_GRAPH[2])

#define apiGetFigure \
    (*(ito::RetVal (*)(const QString &, const QString &, ito::uint32 &, QWidget **, QWidget *parent)) ito::ITOM_API_FUNCS_GRAPH[3])

	//(*(ito::RetVal (*)(ito::uint32 &, const QString, QWidget **)) ito::ITOM_API_FUNCS_GRAPH[3])

#define apiGetPluginList \
	(*(ito::RetVal (*)(const ito::pluginInfo, QHash<QString, ito::pluginInfo> &, const QString)) ito::ITOM_API_FUNCS_GRAPH[4])

#define apiStartLiveData \
	(*(ito::RetVal (*)(QObject *, QObject *)) ito::ITOM_API_FUNCS_GRAPH[5])

#define apiStopLiveData \
	(*(ito::RetVal (*)(QObject *, QObject *)) ito::ITOM_API_FUNCS_GRAPH[6])

#define apiConnectLiveData \
	(*(ito::RetVal (*)(QObject *, QObject *)) ito::ITOM_API_FUNCS_GRAPH[7])

#define apiDisconnectLiveData \
	(*(ito::RetVal (*)(QObject *, QObject *)) ito::ITOM_API_FUNCS_GRAPH[8])

#define apiPaletteGetColorBarIdxFromName \
	(*(ito::RetVal (*)(const QString &, ito::int32 &)) ito::ITOM_API_FUNCS_GRAPH[9])

#define apiGetFigureSetting \
    (*(QVariant (*)(const QObject *, const QString &, const QVariant &, ito::RetVal *)) ito::ITOM_API_FUNCS_GRAPH[10])


#if defined(ITOM_IMPORT_PLOTAPI)
static int importItomPlotApi(void** apiArray)
{
    ito::ITOM_API_FUNCS_GRAPH = apiArray;
    return 0;
}
#endif

} //end namespace ito

#endif // APIFUNCTIONSGRAPHINC_H
