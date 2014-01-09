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

#include "../helper/paramHelper.h"
#include "apiFunctionsGraph.h"
#include "../Qitom/AppManagement.h"
#include "../organizer/addInManager.h"
#include "../organizer/paletteOrganizer.h"
#include "../organizer/designerWidgetOrganizer.h"
#include "../Qitom/organizer/uiOrganizer.h"

#include <qmetaobject.h>
#include <qcoreapplication.h>
#include <qsettings.h>

static ito::apiFunctionsGraph singleApiFunctionsGraph;

namespace ito
{
    void **ITOM_API_FUNCS_GRAPH;

    void *ITOM_API_FUNCS_GRAPH_ARR[] = {
        (void*)&singleApiFunctionsGraph.mnumberOfColorBars,     /* [0] */
        (void*)&singleApiFunctionsGraph.mgetColorBarName,       /* [1] */
        (void*)&singleApiFunctionsGraph.mgetColorBarIdx,        /* [2] */
        (void*)&singleApiFunctionsGraph.mgetFigure,             /* [3] */
        (void*)&singleApiFunctionsGraph.mgetPluginList,         /* [4] */
        (void*)&singleApiFunctionsGraph.mstartLiveData,         /* [5] */
        (void*)&singleApiFunctionsGraph.mstopLiveData,          /* [6] */
        (void*)&singleApiFunctionsGraph.mconnectLiveData,       /* [7] */
        (void*)&singleApiFunctionsGraph.mdisconnectLiveData,    /* [8] */
        (void*)&singleApiFunctionsGraph.mgetColorBarIdxFromName,/* [9] */
        (void*)&singleApiFunctionsGraph.mgetFigureSetting,      /* [10] */
        NULL
    };

//------------------------------------------------------------------------------------------------------------------------------------------------------
apiFunctionsGraph::apiFunctionsGraph()
{ 
    ITOM_API_FUNCS_GRAPH = ITOM_API_FUNCS_GRAPH_ARR;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
apiFunctionsGraph::~apiFunctionsGraph()
{}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mnumberOfColorBars(int &number)
{
    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (!paletteOrganizer)
        return ito::retError;
    number = paletteOrganizer->numberOfColorBars();
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetColorBarName(const QString &name, ito::ItomPalette &palette)
{
    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (!paletteOrganizer)
    {
        return ito::retError;
    }

    bool found;
    palette = paletteOrganizer->getColorBar(name, &found).getPalette();

    if (found)
    {
        return ito::retOk;
    }
    else
    {
        return ito::RetVal::format(ito::retError,0,"color map '%s' not found", name.toAscii().data());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetColorBarIdx(const int number, ito::ItomPalette &palette)
{
    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (!paletteOrganizer || (number > (paletteOrganizer->numberOfColorBars() - 1)))
        return ito::retError;
    palette = paletteOrganizer->getColorBar(number).getPalette();
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetColorBarIdxFromName(const QString &name, ito::int32 & index)
{
    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (!paletteOrganizer)
    {
        index = 0;
        return ito::retError;
    }

    bool found;
    index = paletteOrganizer->getColorBarIndex(name, &found);

    if (found)
    {
        return ito::retOk;
    }
    else
    {
        return ito::RetVal::format(ito::retError,0,"color map '%s' not found", name.toAscii().data());
    }
}

////------------------------------------------------------------------------------------------------------------------------------------------------------
//ito::RetVal apiFunctionsGraph::mgetFigure(ito::uint32 &UID, const QString plugin, QWidget **newFigure)
//{
//    ito::RetVal retval = ito::retOk;
//    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
//
//    if (UID)
//    {
//        *newFigure = uiOrg->getPluginReference(UID);
//        if (*newFigure)
//            return ito::retOk;
//    }
//
////    retval += uiOrg->getNewPluginWindow(plugin, UID, newFigure);
//    retval += uiOrg->getNewPluginWindow(plugin, UID, newFigure);
//
//    return retval;
//}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetFigure(const QString &figCategoryName, const QString &figClassName, ito::uint32 &UID, QWidget **figure, QWidget *parent /*= NULL*/)
{
    ito::RetVal retval;
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    DesignerWidgetOrganizer *dwOrg = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());

    if(uiOrg)
    {
        if(UID > 0)
        {
            *figure = qobject_cast<QWidget*>(uiOrg->getPluginReference(UID));
            if(*figure && parent)
            {
                (*figure)->setParent( parent );
            }
            if(*figure == NULL)
            {
                UID = 0;
            }
        }

        if(UID == 0 && dwOrg)
        {
            QString className = dwOrg->getFigureClass(figCategoryName, figClassName, retval);
            if(!retval.containsError())
            {
                retval += uiOrg->getNewPluginWindow(className, UID, figure, parent);
            }
        }
        else if(!dwOrg)
        {
            retval += ito::RetVal(ito::retError,0,"designerWidgetOrganizer is not available");
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError,0,"uiOrganizer is not available");
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetPluginList(const ito::PluginInfo requirements, QHash<QString, ito::PluginInfo> &pluginList, const QString preference)
{
    //UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();
    //QHash<QString, ito::PluginInfo> itomPluginList = uiOrg->getLoadedPluginList();

    //if (pluginList.contains(preference))
    //{
    //    pluginList.clear();
    //    pluginList.insert(preference, itomPluginList[preference]);
    //    return ito::retOk;
    //}
    //else
    //{
    //    pluginList.clear();
    //    int temp1, temp2;

    //    QHashIterator<QString, ito::PluginInfo> i(itomPluginList);
    //    while (i.hasNext()) 
    //    {
    //        i.next();
    //        //check whether all flags set in requirements are set in PluginInfo of any plugin. If so, append it to pluginList
    //        temp1 = i.value().m_plotDataFormats;
    //        temp2 = requirements.m_plotDataFormats;
    //        if( (temp1 & temp2) == temp2)
    //        {
    //            temp1 = i.value().m_plotDataTypes;
    //            temp2 = requirements.m_plotDataTypes;
    //            if( (temp1 & temp2) == temp2)
    //            {
    //                temp1 = i.value().m_plotFeatures;
    //                temp2 = requirements.m_plotFeatures;
    //                if( (temp1 & temp2) == temp2)
    //                {
    //                    pluginList.insert(i.key(), i.value());
    //                }
    //            }
    //        }
    //    }
    //}

    //return ito::retOk;
    return ito::retError;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mstartLiveData(QObject *liveDataSource, QObject *liveDataView)
{
    QMetaObject::invokeMethod(liveDataSource, "startDeviceAndRegisterListener", Q_ARG(QObject*, liveDataView), Q_ARG(ItomSharedSemaphore*, NULL));
    return ito::retOk;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mstopLiveData(QObject *liveDataSource, QObject *liveDataView)
{
    RetVal retValue(retOk);

    qDebug() << "stopLiveView Thread: " << QThread::currentThreadId();

    ItomSharedSemaphoreLocker locker1(new ItomSharedSemaphore());

    QCoreApplication::sendPostedEvents(liveDataSource,0);
    QMetaObject::invokeMethod(liveDataSource, "stopDeviceAndUnregisterListener", Q_ARG(QObject*, liveDataView), Q_ARG(ItomSharedSemaphore*, locker1.getSemaphore()));
    QCoreApplication::sendPostedEvents(liveDataSource,0);

    if(!locker1.getSemaphore()->wait(10000))
    {
        retValue += RetVal(retError, 1001, QObject::tr("timeout while unregistering live image from camera.").toAscii().data());
    }
    else
    {
        retValue += locker1.getSemaphore()->returnValue;
//        m_started = false;
    }

    qDebug() << "stopLiveView done";

    return retValue;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mconnectLiveData(QObject *liveDataSource, QObject *liveDataView)
{
    ito::RetVal retval = ito::retOk;

    if (liveDataSource && liveDataView)
    {
        if(liveDataSource->inherits("ito::AddInDataIO"))
        {
            ito::AddInManager *aim = ito::AddInManager::getInstance();
            retval += aim->incRef((ito::AddInBase*)liveDataSource);
        }
        else
        {
            retval += ito::RetVal(ito::retError,0,"liveDataSource is no instance of ito::AddInDataIO");
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError,0,"liveDataSource or liveDataView are NULL");
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mdisconnectLiveData(QObject *liveDataSource, QObject *liveDataView)
{
    ito::RetVal retval = ito::retOk;

    if (liveDataSource && liveDataView)
    {
        if(liveDataSource->inherits("ito::AddInDataIO"))
        {
            //retval += mstopLiveData(liveDataSource, liveDataView);
        ito::AddInManager *aim = ito::AddInManager::getInstance();
        retval += aim->decRef((ito::AddInBase**)&liveDataSource);
        }
        else
        {
            retval += ito::RetVal(ito::retError,0,"liveDataSource is no instance of ito::AddInDataIO");
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError,0,"liveDataSource or liveDataView are NULL");
    }

    return retval;
}



//------------------------------------------------------------------------------------------------------------------------------------------------------
QVariant apiFunctionsGraph::mgetFigureSetting(const QObject *figureClass, const QString &key, const QVariant &defaultValue /*= QVariant()*/, ito::RetVal *retval/* = NULL*/)
{
    if(!figureClass)
    {
        if(retval) (*retval) += ito::RetVal(ito::retError,0,"figureClass is NULL. No settings could be retrieved");
        return defaultValue;
    }
    else if(figureClass->inherits("ito::AbstractFigure") == false)
    {
        if(retval) (*retval) += ito::RetVal(ito::retError,0,"figureClass is not inherited from AbstractFigure. No settings could be retrieved");
        return defaultValue;
    }

    const QMetaObject *mo = figureClass->metaObject();
    bool found = false;
    QVariant value = defaultValue;
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DesignerPlugins");

    while(mo && !found)
    {
        settings.beginGroup(mo->className());
        if(settings.contains(key))
        {
            value = settings.value(key,defaultValue);
            found = true;
        }
        settings.endGroup();

        mo = mo->superClass();
    }

    settings.endGroup();
    
    return value;
}

}; // namespace ito
