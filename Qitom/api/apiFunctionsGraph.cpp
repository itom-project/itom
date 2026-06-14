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

#include "../python/pythonEngine.h"
#include "../../AddInManager/paramHelper.h"
#include "apiFunctionsGraph.h"
#include "../Qitom/AppManagement.h"
#include "../../AddInManager/addInManager.h"
#include "../organizer/paletteOrganizer.h"
#include "../organizer/designerWidgetOrganizer.h"
#include "../Qitom/organizer/uiOrganizer.h"
#include "../helper/qpropertyHelper.h"

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
        (void*)&singleApiFunctionsGraph.mgetPluginWidget,       /* [11] */
        (void*)&singleApiFunctionsGraph.mgetFigureUIDByHandle,  /* [12] */
        (void*)&singleApiFunctionsGraph.mgetPlotHandleByID,     /* [13] */
        (void*)&singleApiFunctionsGraph.sendParamToPyWorkspaceThreadSafe,       /* [14] */
        (void*)&singleApiFunctionsGraph.sendParamsToPyWorkspaceThreadSafe,      /* [15] */
        (void*)&QPropertyHelper::readProperty,                                  /* [16] */
        (void*)&QPropertyHelper::writeProperty,                                 /* [17] */
        (void*)&singleApiFunctionsGraph.mConnectToOutputAndErrorStream,         /* [18] */
        (void*)&singleApiFunctionsGraph.mDisconnectFromOutputAndErrorStream,    /* [19] */
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
    number = paletteOrganizer->numberOfColorPalettes();
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
    palette = paletteOrganizer->getColorPalette(name, &found).getPalette();

    if (found)
    {
        return ito::retOk;
    }
    else
    {
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Color map '%s' not found").toLatin1().data(), name.toLatin1().data());
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetColorBarIdx(const int number, ito::ItomPalette &palette)
{
    ito::PaletteOrganizer *paletteOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (!paletteOrganizer || (number > (paletteOrganizer->numberOfColorPalettes() - 1)))
        return ito::retError;
    palette = paletteOrganizer->getColorPalette(number).getPalette();
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
        return ito::RetVal::format(ito::retError, 0, QObject::tr("Color map '%s' not found").toLatin1().data(), name.toLatin1().data());
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
//! tries to get an existing figure of a given UID (if UID > 0) or tries to open a new figure from the given figCategoryName and / or figClassName
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

            if(*figure && parent && !(*figure)->parent())
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

                if (*figure)
                {
                    //minimum size of a new figure window (see also uiOrganizer::figurePlot)
                    QSize minimumFigureSize(700, 400);
                    QSize sz = (*figure)->sizeHint();
                    sz.rwidth() = qMax(minimumFigureSize.width(), sz.width());
                    sz.rheight() = qMax(minimumFigureSize.height(), sz.height());
                    (*figure)->resize(sz);
                }
            }
        }
        else if(!dwOrg)
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("DesignerWidgetOrganizer is not available").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("uiOrganizer is not available").toLatin1().data());
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetPluginList(const ito::PluginInfo &requirements, QHash<QString, ito::PluginInfo> &pluginList, const QString preference)
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

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

    QCoreApplication::sendPostedEvents(liveDataSource,0);
    QMetaObject::invokeMethod(liveDataSource, "stopDeviceAndUnregisterListener", Q_ARG(QObject*, liveDataView), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    QCoreApplication::sendPostedEvents(liveDataSource,0);

    if(!locker.getSemaphore()->wait(10000))
    {
        retValue += RetVal(retError, 1001, QObject::tr("Timeout while unregistering live image from camera.").toLatin1().data());
    }
    else
    {
        retValue += locker.getSemaphore()->returnValue;
    }

    qDebug() << "stopLiveView done";

    return retValue;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mconnectLiveData(QObject *liveDataSource, QObject *liveDataView)
{
    ito::RetVal retval(ito::retOk);

    if (liveDataSource && liveDataView)
    {
        if(liveDataSource->inherits("ito::AddInDataIO"))
        {
            ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
            retval += aim->incRef((ito::AddInBase*)liveDataSource);
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("LiveDataSource is no instance of ito::AddInDataIO").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("LiveDataSource or liveDataView are NULL").toLatin1().data());
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mdisconnectLiveData(QObject *liveDataSource, QObject *liveDataView)
{
    ito::RetVal retval(ito::retOk);

    if (liveDataSource && liveDataView)
    {
        if(liveDataSource->inherits("ito::AddInDataIO"))
        {
            ito::AddInManager *aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
            retval += aim->decRef((ito::AddInBase**)&liveDataSource);
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("LiveDataSource is no instance of ito::AddInDataIO").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("LiveDataSource or liveDataView are NULL").toLatin1().data());
    }

    return retval;
}



//------------------------------------------------------------------------------------------------------------------------------------------------------
QVariant apiFunctionsGraph::mgetFigureSetting(const QObject *figureClass, const QString &key, const QVariant &defaultValue /*= QVariant()*/, ito::RetVal *retval/* = NULL*/)
{
    if(!figureClass)
    {
        if (retval) (*retval) += ito::RetVal(ito::retError, 0, QObject::tr("FigureClass is NULL. No settings could be retrieved").toLatin1().data());
        return defaultValue;
    }
    else if(figureClass->inherits("ito::AbstractFigure") == false)
    {
        if (retval) (*retval) += ito::RetVal(ito::retError, 0, QObject::tr("FigureClass is not inherited from AbstractFigure. No settings could be retrieved").toLatin1().data());
        return defaultValue;
    }

    //there are two possibilities where a figure can get a setting from:
    //1. If the parent() of the figureClass is also inherited from ito::AbstractFigure, the parent figure will asked for
    //   a property with the same name. If this exists, it will be taken
    //2. Else: the settings file will be asked
    bool found = false;
    QVariant value = defaultValue;

    if (figureClass->parent() && qobject_cast<const AbstractFigure*>(figureClass->parent()))
    {
        const QObject *par = qobject_cast<const AbstractFigure*>(figureClass->parent());
        int idx = par->metaObject()->indexOfProperty(key.toLatin1().data());
        if (idx >= 0)
        {
            value = par->metaObject()->property(idx).read(par);
            found = true;
        }
    }

    if (!found) //check ini setting file
    {
        const QMetaObject *mo = figureClass->metaObject();

        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup("DesignerPlugins");

        while (mo && !found)
        {
            settings.beginGroup(mo->className());
            if (settings.contains(key))
            {
                value = settings.value(key, defaultValue);
                found = true;
            }
            settings.endGroup();

            mo = mo->superClass();
        }

        settings.endGroup();
    }

    return value;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetPluginWidget(char* algoWidgetFunc, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QPointer<QWidget> *widget)
{
    ito::RetVal retval(ito::retOk);
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    DesignerWidgetOrganizer *dwOrg = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());

    if(uiOrg)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QSharedPointer<unsigned int> dialogHandle(new unsigned int);
        QSharedPointer<unsigned int> objectID(new unsigned int);
        QSharedPointer<int> modalRet(new int);
        QSharedPointer<QByteArray> className(new QByteArray());
        *dialogHandle = 0;
        *objectID = 0;
        UiContainer *widgetContainer = NULL;
        QMetaObject::invokeMethod(uiOrg, "loadPluginWidget", Q_ARG(void*, reinterpret_cast<char*>(algoWidgetFunc)), Q_ARG(int, uiOrg->createUiDescription(0, 0, 0, 1, 4)), Q_ARG(const StringMap, StringMap()), Q_ARG(QVector<ito::ParamBase>*, paramsMand), Q_ARG(QVector<ito::ParamBase>*, paramsOpt), Q_ARG(QSharedPointer<unsigned int>, dialogHandle), Q_ARG(QSharedPointer<unsigned int>, objectID), Q_ARG(QSharedPointer<QByteArray>, className), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginInitClose))
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Timeout while loading plugin widget").toLatin1().data());
            return retval;
        }

        retval += locker.getSemaphore()->returnValue;
        if (retval.containsError())
            return retval;

        QMetaObject::invokeMethod(uiOrg, "getUiDialogByHandle", Qt::BlockingQueuedConnection, Q_RETURN_ARG(UiContainer*, widgetContainer), Q_ARG(uint, *dialogHandle)); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        if (!(*widget = widgetContainer->getUiWidget()))
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Error retrieving widget pointer").toLatin1().data());
            ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(uiOrg, "deleteDialog", Q_ARG(uint, static_cast<unsigned int>(*dialogHandle)), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

            if (!locker2.getSemaphore()->wait(AppManagement::timeouts.pluginGeneral))
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Error closing dialog").toLatin1().data());
            }
            return retval;
        }

        ItomSharedSemaphoreLocker locker3(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrg, "showDialog", Q_ARG(uint, *dialogHandle) , Q_ARG(int, 0), Q_ARG(QSharedPointer<int>, modalRet), Q_ARG(ItomSharedSemaphore*, locker3.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        if (!locker3.getSemaphore()->wait(AppManagement::timeouts.pluginInitClose))
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Timeout showing dialog").toLatin1().data());
            return retval;
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("UI-Organizer is not available!").toLatin1().data());
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
//! return the figure UID for the given figure
/*
\param figure is the pointer to the figure, whose figure UID should be determined and returned
\param figureUID is the resulting figure UID of figure
\return ito::retOk if figureUID could be found, else ito::retError
*/
ito::RetVal apiFunctionsGraph::mgetFigureUIDByHandle(QObject *figure, ito::uint32 &figureUID)
{
    ito::RetVal retval;
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrg)
    {
        QSharedPointer<unsigned int > uid(new unsigned int) ;
        *uid = 0;
        uiOrg->getObjectID(figure, uid);
        figureUID = *uid;
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("uiOrganizer is not available").toLatin1().data());
    }

    return retval;
}


//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mgetPlotHandleByID(const ito::uint32 &figureUID, ito::ItomPlotHandle &plotHandle)
{
    ito::RetVal retval;
    UiOrganizer *uiOrg = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrg)
    {
        QObject *obj = uiOrg->getPluginReference(figureUID);

        if (obj)
        {
            plotHandle = ito::ItomPlotHandle(obj->objectName().toLatin1().data(), obj->metaObject()->className(), figureUID);
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, QObject::tr("Plot widget does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("uiOrganizer is not available").toLatin1().data());
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::sendParamToPyWorkspaceThreadSafe(const QString &varname, const QSharedPointer<ito::ParamBase> &value)
{
    return sendParamsToPyWorkspaceThreadSafe(QStringList(varname), QVector<QSharedPointer<ito::ParamBase> >(1, value));
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::sendParamsToPyWorkspaceThreadSafe(const QStringList &varnames, const QVector<QSharedPointer<ito::ParamBase> > &values)
{
    ito::RetVal retval;
    PythonEngine *pyEng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyEng)
    {
        if (QThread::currentThreadId() == pyEng->getPythonThreadId())
        {
            retval += pyEng->putParamsToWorkspace(true, varnames, values, NULL);
        }
        else
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(pyEng, "putParamsToWorkspace", Q_ARG(bool,true), Q_ARG(QStringList,varnames), Q_ARG(QVector<SharedParamBasePointer >, values), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
            if (locker->wait(AppManagement::timeouts.pluginGeneral))
            {
                retval += locker->returnValue;
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Timeout while sending variables to python workspace. Python is maybe busy. Try it later again.").toLatin1().data());
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, QObject::tr("Python is not available.").toLatin1().data());
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mConnectToOutputAndErrorStream(const QObject *receiver, const char *method, ito::tStreamMessageType messageType)
{
    ito::RetVal retval;
    QObject *sender = NULL;
    switch (messageType)
    {
    case ito::msgStreamOut:
        sender = AppManagement::getCoutStream();
        break;
    case ito::msgStreamErr:
        sender = AppManagement::getCerrStream();
        break;
    default:
        retval += ito::RetVal(ito::retError, 0, "connection only possible for output or error stream");
        break;
    }

    if (!sender)
    {
        retval += ito::RetVal(ito::retError, 0, "output or error stream is not available");
    }
    else
    {
        QMetaObject::Connection conn = QObject::connect(sender, SIGNAL(flushStream(QString, ito::tStreamMessageType)), receiver, method);

        if (!conn)
        {
            retval += ito::RetVal(ito::retError, 0, "connection cannot be established");
        }
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal apiFunctionsGraph::mDisconnectFromOutputAndErrorStream(const QObject *receiver, const char *method, ito::tStreamMessageType messageType)
{
    ito::RetVal retval;
    QObject *sender = NULL;
    switch (messageType)
    {
    case ito::msgStreamOut:
        sender = AppManagement::getCoutStream();
        break;
    case ito::msgStreamErr:
        sender = AppManagement::getCerrStream();
        break;
    default:
        retval += ito::RetVal(ito::retError, 0, "connection only possible for output or error stream");
        break;
    }

    if (!sender)
    {
        retval += ito::RetVal(ito::retError, 0, "output or error stream is not available");
    }
    else
    {
        bool conn = QObject::disconnect(sender, SIGNAL(flushStream(QString, ito::tStreamMessageType)), receiver, method);

        if (!conn)
        {
            retval += ito::RetVal(ito::retError, 0, "connection cannot be disconnected");
        }
    }

    return retval;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------

}; // namespace ito
