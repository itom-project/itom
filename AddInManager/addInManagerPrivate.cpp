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

#include "addInManagerPrivate.h"
#include "pluginModel.h"
#include "addInManager.h"
#include "paramHelper.h"

#include <QtCore/qpluginloader.h>
#include <qregularexpression.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
AddInManagerPrivate::AddInManagerPrivate(AddInManager* addInMgr) :
    q_ptr(addInMgr),
    m_pMainWindow(NULL),
    m_pMainApplication(NULL),
    m_algoInterfaceValidator(NULL),
    m_timeOutInitClose(30000),
    m_timeOutGeneral(5000),
    m_pQCoreApp(NULL)
{
    m_plugInModel = new PlugInModel(addInMgr, NULL);
    connect(&this->m_deadPluginTimer, SIGNAL(timeout()), this, SLOT(closeDeadPlugins()));
}

//----------------------------------------------------------------------------------------------------------------------------------
AddInManagerPrivate::~AddInManagerPrivate(void)
{
    DELETE_AND_SET_NULL(m_plugInModel);
}

//----------------------------------------------------------------------------------------------------------------------------------
/** decrements the reference counter of arguments passed to a plugin if necessary
*   @param [in] ai          AddIn to which the parameters are passed
*   @param [in] paramsMand  mandatory argument parameters
*   @param [in] paramsOpt   optional argument parameters
*
*   This function decrements the reference counter of plugins passed to other plugins as parameters, to enable
*   a closing of the passed plugins when they are no longer used by any other plugin.
*/
ito::RetVal AddInManagerPrivate::decRefParamPlugins(ito::AddInBase *ai)
{
    ito::RetVal retval(ito::retOk);
    QVector<ito::AddInBase::AddInRef *> *argAddInList = ai->getArgAddIns();

    for (int n = 0; n < argAddInList->size(); n++)
    {
        ito::AddInBase::AddInRef *ref = (*argAddInList)[n];

        ito::AddInBase *closeAi = reinterpret_cast<ito::AddInBase*>((*argAddInList)[n]->ptr);
        ito::AddInInterfaceBase *aib = ai->getBasePlugin();
        if (aib)
        {
            decRef(&closeAi);
        }
        else
        {
            retval += ito::retError;
        }

        delete ref;
    }

    argAddInList->clear();

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** increments the reference counter of arguments passed to a plugin if necessary
*   @param [in] ai          AddIn to which the parameters are passed
*   @param [in] paramsMand  mandatory argument parameters
*   @param [in] paramsOpt   optional argument parameters
*
*   This function increments the reference counter of plugins passed to other plugins as parameters, to avoid
*   the passed plugins are closed while they are still in use by the other plugin.
*/
void AddInManagerPrivate::incRefParamPlugins(ito::AddInBase *ai, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt)
{
    void* hwRefPtr = NULL;

    if (paramsMand)
    {
        QVector<ito::AddInBase::AddInRef *> *addInArgList = ai->getArgAddIns();
        for (int n = 0; n < paramsMand->size(); n++)
        {
            ito::ParamBase *param = &((*paramsMand)[n]);

            if (param->getType() == ParamBase::HWRef)
            {
                hwRefPtr = param->getVal<void *>();
                if (hwRefPtr)
                {
                    /*if (!(param->getFlags() & ito::ParamBase::Axis))
                    {*/
                        ito::AddInBase *aib = reinterpret_cast<ito::AddInBase*>(hwRefPtr);
                        if (aib)
                        {
                            ito::AddInInterfaceBase *ab = aib->getBasePlugin();
                            ab->incRef(aib);
                            ito::AddInBase::AddInRef *ref = new ito::AddInBase::AddInRef(aib, param->getType() & param->getFlags());
                            (*addInArgList).append(ref);
                        }
                    //}
                }
            }
        }
    }

    if (paramsOpt)
    {
        QVector<ito::AddInBase::AddInRef*> *addInArgList = ai->getArgAddIns();
        for (int n = 0; n < paramsOpt->size(); n++)
        {
            ito::ParamBase *param = &((*paramsOpt)[n]);
            if (param->getType() == ParamBase::HWRef)
            {
                hwRefPtr = param->getVal<void *>();
                if (hwRefPtr)
                {
                    /*if (!(param->getFlags() & ito::ParamBase::Axis))
                    {*/
                        ito::AddInBase *aib = reinterpret_cast<ito::AddInBase*>(hwRefPtr);
                        if (aib)
                        {
                            ito::AddInInterfaceBase *ab = aib->getBasePlugin();
                            ab->incRef(aib);
                            ito::AddInBase::AddInRef *ref = new ito::AddInBase::AddInRef(aib, param->getType() & param->getFlags());
                            (*addInArgList).append(ref);
                        }
                    //}
                }
            }
        }
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
int AddInManagerPrivate::getPluginNum(const QString &name, ito::AddInInterfaceBase *&addIn)
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


//----------------------------------------------------------------------------------------------------------------------------------
int AddInManagerPrivate::getItemNum(const void *item)
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


//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @return RetVal
*/
RetVal AddInManagerPrivate::closeDeadPlugins()
{
    RetVal retval(retOk);
    QList< QPointer< ito::AddInBase > >::iterator it;
    it = m_deadPlugins.begin();
    ito::AddInBase *aib = NULL;

    while (it != m_deadPlugins.end())
    {
        aib = it->data();
        if (it->isNull()) //weak pointer does not live any more
        {
            it = m_deadPlugins.erase(it);
        }
        else if (aib->isInitialized()) //plugin finished init-method (late, but finally it did finish ;)), we can kill it now
        {
            retval += closeAddIn(aib, NULL);
            it = m_deadPlugins.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (m_deadPlugins.count() == 0)
    {
        m_deadPluginTimer.stop();
    }
    else if (m_deadPluginTimer.interval() < 30000)
    {
        //the interval is incremented by 2000ms after each run until a max-value of 30secs.
        //This gives the chance to delete newly added dead plugins fast and decrease the
        //intents the longer it did not work.
        m_deadPluginTimer.setInterval(m_deadPluginTimer.interval() + 2000);
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param addIn
*   @return RetVal
*/
RetVal AddInManagerPrivate::registerPluginAsDeadPlugin(ito::AddInBase *addIn)
{
    QPointer<ito::AddInBase> ptr(addIn);
    m_deadPlugins.push_back(ptr);

    if (m_deadPluginTimer.isActive() == false)
    {
        m_deadPluginTimer.start(2000); //the interval is incremented by 2000ms after each run until a max-value of 30secs. This gives the chance to delete newly added dead plugins fast and decrease the intents the longer it did not work.
    }
    return ito::retOk;
}


//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param filename
*   @return RetVal
*/
RetVal AddInManagerPrivate::loadAddIn(QString &filename)
{
    Q_Q(AddInManager);

    RetVal retValue(retOk);
    QString message;
    QFileInfo finfo(filename);
    ito::PluginLoadStatus pls;

    if (QLibrary::isLibrary(filename) == false)
    {
        message = tr("Filename is no itom plugin library: %1").arg(filename);
        qDebug() << message;
        retValue += RetVal(retError, 1001, message.toLatin1().data());
    }
    else
    {
        emit q->splashLoadMessage(tr("Scan and load plugins (%1)").arg(finfo.fileName()));
        QCoreApplication::processEvents();

        QString language("en");
        if (ITOM_API_FUNCS)
        {
            //load translation file
            QString settingsFile = apiGetSettingsFile();
            QSettings settings(settingsFile, QSettings::IniFormat);
            QStringList startupScripts;

            settings.beginGroup("Language");
            language = settings.value("language", "en").toString();
            QByteArray codec = settings.value("codec", "UTF-8").toByteArray();
            settings.endGroup();
        }

        QFileInfo fileInfo(filename);
        QDir fileInfoDir = fileInfo.dir();
        fileInfoDir.cdUp();
        if (language != "en_US" && fileInfoDir.absolutePath() == qApp->applicationDirPath() + "/plugins")
        {
            QLocale local = QLocale(language); //language can be "language[_territory][.codeset][@modifier]"
            QString translationPath = fileInfo.path() + "/translation";
            QString languageStr = local.name().left(local.name().indexOf("_", 0, Qt::CaseInsensitive));
            QDirIterator it(translationPath, QStringList("*_" + languageStr + ".qm"), QDir::Files);
            if (it.hasNext())
            {
                QString translationLocal = it.next();
                m_Translator.append(new QTranslator);
                m_Translator.last()->load(translationLocal, translationPath);
                if (m_Translator.last()->isEmpty())
                {
                    message = tr("Unable to load translation file '%1'. Translation file is empty.").arg(translationPath + '/' + translationLocal);
                    qDebug() << message;
                    pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfWarning, message));
                }
                else
                {
                    QCoreApplication::instance()->installTranslator(m_Translator.last());
                }
            }
            else
            {
                message = tr("Unable to find translation file.");
                qDebug() << message;
                pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfWarning, message));
            }
        }

        try
        {
            QPluginLoader *loader = new QPluginLoader(filename);
            QObject *plugin = loader->instance();
            if (plugin)
            {
                ito::AddInInterfaceBase *ain = qobject_cast<ito::AddInInterfaceBase *>(plugin);

                if (ain && (ITOM_ADDININTERFACE_MAJOR >= 4))
                {
                    //for major interfaces >= 4, semver holds. Check the minor number of the plugin.
                    //if it is <= the minor of itom, the plugin can be loaded. Else not.
                    int minor = MINORVERSION(ain->getAddInInterfaceVersion());
                    if (minor > ITOM_ADDININTERFACE_MINOR)
                    {
                        ain = NULL;
                    }
                }

                pls.filename = filename;

                if (ain)
                {
                    ain->setFilename(filename);
                    ain->setApiFunctions(ITOM_API_FUNCS);
                    ain->setApiFunctionsGraph(ITOM_API_FUNCS_GRAPH);
                    ain->setLoader(loader);
                    //the event User+123 is emitted by AddInManager, if the API has been prepared and can
                    //transmitted to the plugin. This assignment cannot be done directly, since
                    //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
                    //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
                    //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
                    //also is necessary if any methods of the plugin are directly called from itom).
                    QEvent evt((QEvent::Type)(QEvent::User + 123));
                    QCoreApplication::sendEvent(ain, &evt);

                    switch (ain->getType() & (ito::typeDataIO | ito::typeAlgo | ito::typeActuator))
                    {
                    case typeDataIO:
                        if ((ain->getType() & (ito::typeADDA | ito::typeRawIO | ito::typeGrabber)) == 0)
                        {
                            message = tr("Plugin with filename '%1' is a dataIO type, but no subtype is given in the type flag.").arg(filename);
                            qDebug() << message;

                            pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfError, message));
                        }
                        else
                        {
                            retValue += loadAddInDataIO(plugin, pls);
                        }

                        break;

                    case typeActuator:
                        retValue += loadAddInActuator(plugin, pls);
                        break;

                    case typeAlgo:
                        retValue += loadAddInAlgo(plugin, pls);
                        break;

                    default:
                        message = tr("Plugin with filename '%1' is unknown.").arg(filename);
                        qDebug() << message;

                        pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfError, message));
                        break;
                    }
                    m_pluginLoadStatus.append(pls);
                }
                else
                {
                    //check whether this instance is an older or newer version of AddInInterface
                    QObject *obj = qobject_cast<QObject*>(plugin);
                    if (obj)
                    {
                        if (obj->qt_metacast("ito::AddInInterfaceBase") != NULL)
                        {
                            int i = 0;
                            const char* oldName = ito_AddInInterface_OldVersions[0];

                            while (oldName != NULL)
                            {
                                if (obj->qt_metacast(oldName) != NULL)
                                {
                                    message = tr("AddIn '%1' fits to the obsolete interface %2. The AddIn interface of this version of 'itom' is %3.").arg(filename).arg(oldName).arg(ITOM_ADDININTERFACE_VERSION_STR);
                                    break;
                                }
                                oldName = ito_AddInInterface_OldVersions[++i];
                            }
                            if (oldName == NULL)
                            {
                                message = tr("AddIn '%1' fits to a new addIn-interface, which is not supported by this version of itom. The AddIn interface of this version of 'itom' is %2.").arg(filename).arg(ITOM_ADDININTERFACE_VERSION_STR);
                            }
                        }
                        else
                        {
                            message = tr("AddIn '%1' does not fit to the general interface AddInInterfaceBase").arg(filename);
                        }
                    }
                    else
                    {
                        message = tr("AddIn '%1' is not derived from class QObject.").arg(filename).arg(loader->errorString());
                    }
                    qDebug() << message;

                    pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfError, message));
                    m_pluginLoadStatus.append(pls);

                    //                    delete plugin;
                    loader->unload();
                    DELETE_AND_SET_NULL(loader);
                }
            }
            else
            {
                QString notValidQtLibraryMsg = tr("The file '%1' is not a valid Qt plugin.").arg("*");
                QRegularExpression rx = QRegularExpression(AddInManagerPrivate::wildcardToRegularExpression(notValidQtLibraryMsg));
                if (rx.match(loader->errorString()).hasMatch())
                {
                    message = tr("Library '%1' was ignored. Message: %2").arg(filename).arg(loader->errorString());
                    qDebug() << message;
                    pls.filename = filename;
                    pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfIgnored, message));
                    m_pluginLoadStatus.append(pls);
                }
                else
                {
                    //This regular expression is used to check whether the error message during loading a plugin contains the words
                    //'debug' or 'release'. This means, that a release plugin is tried to be loaded with a debug version of itom or vice-versa
                    QRegularExpression regExpDebugRelease = QRegularExpression(".*(release|debug).*");
                    regExpDebugRelease.setPatternOptions(QRegularExpression::CaseInsensitiveOption);

                    if (regExpDebugRelease.match(loader->errorString()).hasMatch())
                    {
                        message = tr("AddIn '%1' could not be loaded. Error message: %2").arg(filename).arg(loader->errorString());
                        qDebug() << message;
                        pls.filename = filename;
                        ito::PluginLoadStatusFlags flags(plsfWarning | plsfRelDbg);
                        pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(flags, message));
                        m_pluginLoadStatus.append(pls);
                    }
                    else
                    {
                        message = tr("AddIn '%1' could not be loaded. Error message: %2").arg(filename).arg(loader->errorString());
                        qDebug() << message;

                        pls.filename = filename;
                        pls.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfError, message));
                        m_pluginLoadStatus.append(pls);
                    }
                }
                loader->unload();
                DELETE_AND_SET_NULL(loader);
            }
        }
        catch (std::logic_error &ex)
        {
            const char* what = ex.what();
            retValue += ito::RetVal(ito::retError, 0, tr("Caught an exception when loading the plugin '%1'\nReason: %2").arg(filename).arg(what).toLatin1().data());
        }
        catch (...)
        {
            retValue += ito::RetVal(ito::retError, 0, tr("Caught an exception when loading the plugin '%1'").arg(filename).toLatin1().data());
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param plugin
*   @param pluginLoadStatus
*   @return RetVal
*/
RetVal AddInManagerPrivate::loadAddInDataIO(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus)
{
    if (!m_addInListDataIO.contains(plugin))
    {
        m_addInListDataIO.append(plugin);
        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfOk, tr("%1 (DataIO) loaded").arg(plugin->objectName())));
        return retOk;
    }
    else
    {
        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfWarning, tr("Plugin %1 (DataIO) already exists. Duplicate rejected.").arg(plugin->objectName())));
        return retWarning;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param plugin
*   @param pluginLoadStatus
*   @return RetVal
*/
RetVal AddInManagerPrivate::loadAddInActuator(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus)
{
    if (!m_addInListAct.contains(plugin))
    {
        m_addInListAct.append(plugin);
        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfOk, tr("%1 (Actuator) loaded").arg(plugin->objectName())));
        return retOk;
    }
    else
    {
        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfWarning, tr("Plugin %1 (Actuator) already exists. Duplicate rejected.").arg(plugin->objectName())));
        return retWarning;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param plugin
*   @param pluginLoadStatus
*   @return RetVal
*/
RetVal AddInManagerPrivate::loadAddInAlgo(QObject *plugin, ito::PluginLoadStatus &pluginLoadStatus)
{
    QString message;
    ito::RetVal retValue;
    if (!m_addInListAlgo.contains(plugin))
    {
        m_addInListAlgo.append(plugin);

        ito::AddInAlgo *algoInst = NULL;
        QVector<ito::ParamBase> paramsMand, paramsOpt;
        retValue += initAddInAlgo(m_addInListAlgo.size() - 1, plugin->objectName(), &algoInst, &paramsMand, &paramsOpt, true);
        if (!algoInst)
        {
            message = tr("Error initializing plugin: %1").arg(plugin->objectName());
            qDebug() << message;
            retValue += RetVal(retError, 1002, message.toLatin1().data());
            pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
        }
        else
        {
            QHash<QString, ito::AddInAlgo::FilterDef *> funcList;
            algoInst->getFilterList(funcList);

            QHash<QString, ito::AddInAlgo::FilterDef *>::const_iterator it = funcList.constBegin();
            ito::AddInInterfaceBase *ain = qobject_cast<ito::AddInInterfaceBase *>(plugin);

            ito::AddInAlgo::FilterDef *fd;
            ito::RetVal validRet;
            QVector<ito::Param> paramsMand, paramsOpt, paramsOut;
            QStringList tags;
            while (it != funcList.constEnd())
            {
                fd = *it;
                if (m_filterList.contains(it.key()))
                {
                    algoInst->rejectFilter(it.key());
                    message = tr("Filter '%1' rejected since a filter with the same name already exists in global filter list.\n").arg(it.key());
                    qDebug() << message;
                    retValue += RetVal(retWarning, 1004, message.toLatin1().data());
                    pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfWarning, message));
                }
                else
                {
                    //1. first check if filter has a valid interface (if indicated)
                    validRet = ito::retOk;
                    tags.clear();
                    if (fd->m_interface == 0 || m_algoInterfaceValidator->isValidFilter(*fd, validRet, tags))
                    {

                        //2. hash the mand, opt and out param vectors from the filter (if not yet done, since multiple filters can use the same paramFunc-function.
                        paramsMand.clear();
                        paramsOpt.clear();
                        paramsOut.clear();
                        if (!filterParamHash.contains((void*)fd->m_paramFunc))
                        {
                            validRet += fd->m_paramFunc(&paramsMand, &paramsOpt, &paramsOut);

                            if (!validRet.containsError())
                            {
                                foreach(const ito::Param &p, paramsMand)
                                {
                                    if ((p.getName() != NULL) && (strcmp(p.getName(), "_observer") == 0))
                                    {
                                        validRet += ito::RetVal::format(ito::retError, 0, "The parameter name '_observer' is a reserved name and may not be used as real parameter name.");
                                        break;
                                    }
                                }

                                foreach(const ito::Param &p, paramsOpt)
                                {
                                    if ((p.getName() != NULL) && (strcmp(p.getName(), "_observer") == 0))
                                    {
                                        validRet += ito::RetVal::format(ito::retError, 0, "The parameter name '_observer' is a reserved name and may not be used as real parameter name.");
                                        break;
                                    }
                                }

                                if (!validRet.containsError())
                                {
                                    ito::FilterParams *fp = new ito::FilterParams();
                                    fp->paramsMand = paramsMand;
                                    fp->paramsOpt = paramsOpt;
                                    fp->paramsOut = paramsOut;
                                    filterParamHash[(void*)fd->m_paramFunc] = fp;
                                }
                            }
                        }

                        if (!validRet.containsError())
                        {
                            fd->m_pBasePlugin = ain; //put pointer to corresponding AddInInterfaceBase to this filter
                            fd->m_name = it.key();
                            m_filterList.insert(it.key(), fd);
                            pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfOk, tr("Filter %1 loaded").arg(it.key())));

                            if (tags.size() == 0) tags.append("");
                            foreach(const QString &tag, tags)
                            {
                                m_filterListInterfaceTag.insert(QString::number(fd->m_interface) + "_" + tag, fd);
                            }
                        }
                        else
                        {
                            algoInst->rejectFilter(it.key());
                            if (validRet.hasErrorMessage())
                            {
                                message = tr("Filter '%1' rejected. The filter parameters could not be loaded: %2").arg(it.key()).arg(QLatin1String(validRet.errorMessage()));
                            }
                            else
                            {
                                message = tr("Filter '%1' rejected. The filter parameters could not be loaded.").arg(it.key());
                            }
                            qDebug() << message;
                            pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
                        }
                    }
                    else if (validRet.containsError() || fd->m_interface != 0) //the !=0 check is only to make sure that we always get into that case if the filter is somehow wrong
                    {
                        algoInst->rejectFilter(it.key());
                        if (validRet.hasErrorMessage())
                        {
                            message = tr("Filter '%1' rejected. It does not correspond to the algorithm interface: %2").arg(it.key()).arg(QLatin1String(validRet.errorMessage()));
                        }
                        else
                        {
                            message = tr("Filter '%1' rejected. It does not correspond to the algorithm interface.").arg(it.key());
                        }
                        qDebug() << message;
                        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
                    }
                }
                ++it;
            }

            QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> algoWidgetList;
            ito::AddInAlgo::AlgoWidgetDef *ad;
            algoInst->getAlgoWidgetList(algoWidgetList);

            QHash<QString, ito::AddInAlgo::AlgoWidgetDef *>::const_iterator jt = algoWidgetList.constBegin();
            while (jt != algoWidgetList.constEnd())
            {
                if (m_algoWidgetList.contains(jt.key()))
                {
                    algoInst->rejectAlgoWidget(jt.key());
                    message = QObject::tr("Widget '%1' rejected since widget with the same name already exists in global plugin widget list").arg(jt.key());
                    qDebug() << message;
                    retValue += RetVal(retWarning, 1005, message.toLatin1().data());
                }
                else
                {
                    ad = *jt;
                    //1. first check if filter has a valid interface (if indicated)
                    validRet = ito::retOk;
                    tags.clear();
                    if (ad->m_interface == 0 || m_algoInterfaceValidator->isValidWidget(*ad, validRet, tags))
                    {

                        //2. hash the mand, opt and out param vectors from the widget  (if not yet done, since multiple filters can use the same paramFunc-function.
                        paramsMand.clear();
                        paramsOpt.clear();
                        paramsOut.clear();
                        if (!filterParamHash.contains((void*)ad->m_paramFunc))
                        {
                            ad->m_paramFunc(&paramsMand, &paramsOpt, &paramsOut);
                            ito::FilterParams *fp = new ito::FilterParams();
                            fp->paramsMand = paramsMand;
                            fp->paramsOpt = paramsOpt;
                            fp->paramsOut = paramsOut;
                            filterParamHash[(void*)ad->m_paramFunc] = fp;
                        }

                        ad->m_pBasePlugin = ain; //put pointer to corresponding AddInInterfaceBase to this filter
                        ad->m_name = jt.key();
                        m_algoWidgetList.insert(jt.key(), ad);
                        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfOk, QObject::tr("Widget %1 loaded").arg(jt.key())));
                    }
                    else if (validRet.containsError())
                    {
                        algoInst->rejectAlgoWidget(jt.key());
                        if (validRet.hasErrorMessage())
                        {
                            message = tr("Widget '%1' rejected. It does not correspond to the algorithm interface: %2").arg(jt.key()).arg(QLatin1String(validRet.errorMessage()));
                        }
                        else
                        {
                            message = tr("Widget '%1' rejected. It does not correspond to the algorithm interface.").arg(jt.key());
                        }
                        qDebug() << message;
                        pluginLoadStatus.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
                    }
                }
                ++jt;
            }
        }
    }

    return retValue;
}


//----------------------------------------------------------------------------------------------------------------------------------
/** initAddInAlgo initialize new instance of a algo addIn class
*   @param [in]  pluginNum      number of the plugin in the plugin list, retrieved with \ref getInitParams
*   @param [in]  name           name of the plugin to be initialized, this just a check that number and name correspond, principally it should not be necessary to pass the name
*   @param [out] addIn          pointer to the new instance of the plugin class
*   @param [in]  paramsMand     mandatory initialisation parameters which are required by the initialisation. As this vector should(must) be retrieved from the plugin
*                               previously with the \ref getInitParams method it should always be filled with meaningful values
*   @param [in]  paramsOpt      mandatory initialisation parameters which may optionally be passed to the initialisation. As this vector should(must) be retrieved from the plugin
*                               previously with the \ref getInitParams method it should always be filled with meaningful values
*   @return      on success ito::retOk, ito::retError otherwise
*
*   new instance from the addIn class is created. In contrast to the dataIO and actuator plugins the new object is not moved to a new thread and no init method is called.
*   As a last step the plugins parameters are loaded from the plugins parameters xml file \ref loadParamVals.
*/
const ito::RetVal AddInManagerPrivate::initAddInAlgo(
    const int pluginNum,
    const QString &name,
    ito::AddInAlgo **addIn,
    QVector<ito::ParamBase> *paramsMand,
    QVector<ito::ParamBase> *paramsOpt,
    bool autoLoadPluginParams,
    ItomSharedSemaphore *aimWait)
{
    ItomSharedSemaphoreLocker locker(aimWait);
    ito::RetVal retval = ito::retOk;
    ito::tAutoLoadPolicy policy = ito::autoLoadNever;
    ito::AddInInterfaceBase *aib = NULL;

    if (QString::compare((m_addInListAlgo[pluginNum])->objectName(), name, Qt::CaseInsensitive) != 0)
    {
        retval += ito::RetVal(ito::retError, 0, tr("Wrong plugin name").toLatin1().data());
    }
    else
    {
        aib = qobject_cast<ito::AddInInterfaceBase *>(m_addInListAlgo[pluginNum]);
        retval += aib->getAddInInst(reinterpret_cast<ito::AddInBase **>(addIn));
        if ((!addIn) || (!*addIn))
        {
            retval += ito::RetVal(ito::retError, 0, tr("Plugin instance is invalid (NULL)").toLatin1().data());
        }
    }

    if (!retval.containsError())
    {
        //ref-count of plugin must be zero (that means one instance is holder a single reference), this is rechecked in the following line
        if (aib->getRef(*addIn) != 0)
        {
            retval += ito::RetVal(ito::retWarning, 0, tr("Reference counter of plugin has to be initialized with zero. This is not the case for this plugin (Please contact the plugin developer).").toLatin1().data());
        }

        (*addIn)->init(paramsMand, paramsOpt);

        if (!((*addIn)->getBasePlugin()->getType() & ito::typeAlgo) || retval.containsError())
        {
            if (*addIn != NULL)
            {
                retval += closeAddIn(*addIn);
            }
            *addIn = NULL;
        }
    }

    if (!retval.containsError())
    {
        policy = (*addIn)->getBasePlugin()->getAutoLoadPolicy();

        // ck 31.12.16 removed warning about ignoring autoload settings
        //        if (autoLoadPluginParams && policy != ito::autoLoadKeywordDefined)
        //        {
        //            retval += ito::RetVal(ito::retWarning, 0, tr("Parameter has own parameter management. Keyword 'autoLoadParams' is ignored.").toLatin1().data());
        //        }

        if (policy == ito::autoLoadAlways || (policy == ito::autoLoadKeywordDefined && autoLoadPluginParams))
        {
            retval += loadParamVals(*addIn);
        }
    }

    if (aimWait)
    {
        aimWait->returnValue = retval;
        aimWait->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** initAddIn initialize new instance of a dataIO addIn class
*   @param [in]  actuatorNotDataIO true if an actuator plugin should be initialized, else false
*   @param [in]  pluginNum      number of the plugin in the plugin list, retrieved with \ref getInitParams
*   @param [in]  name           name of the plugin to be initialized, this just a check that number and name correspond, principally it should not be necessary to pass the name
*   @param [out] addIn          pointer to the new instance of the plugin class
*   @param [in]  paramsMand     mandatory initialisation parameters which are required by the initialisation. As this vector should(must) be retrieved from the plugin
*                               previously with the \ref getInitParams method it should always be filled with meaningful values
*   @param [in]  paramsOpt      mandatory initialisation parameters which may optionally be passed to the initialisation. As this vector should(must) be retrieved from the plugin
*                               previously with the \ref getInitParams method it should always be filled with meaningful values
*   @param [in, out] aimWait    wait condition for calls from other threads. See also \ref ItomSharedSemaphore
*   @return      on success ito::retOk, ito::retError otherwise
*
*   A new instance from the addIn class is created then the newly created object is moved into a new thread. Afterwards the classes init method is invoked with
*   the passed mandatory and optional parameters. As a last step the plugins parameters are loaded from the plugins parameters xml file \ref loadParamVals.
*/
template<typename _Tp> const ito::RetVal AddInManagerPrivate::initAddInActuatorOrDataIO(
    bool actuatorNotDataIO,
    const int pluginNum,
    const QString &name,
    _Tp** addIn,
    QVector<ito::ParamBase> *paramsMand,
    QVector<ito::ParamBase> *paramsOpt,
    bool autoLoadPluginParams,
    ItomSharedSemaphore *aimWait)
{
    ItomSharedSemaphoreLocker locker(aimWait);
    ito::RetVal retval = ito::retOk;
    ItomSharedSemaphore *waitCond = NULL;
    ito::tAutoLoadPolicy policy = ito::autoLoadNever;
    ito::AddInInterfaceBase *aib = NULL;
    bool callInitInNewThread;
    bool timeoutOccurred = false;
    bool modelInsertAnnounced = false;

    QObjectList *addInList = (actuatorNotDataIO) ? &m_addInListAct : &m_addInListDataIO;

    if (addIn == NULL)
    {
        retval += ito::RetVal(ito::retError, 0, "Empty plugin pointer");
    }
    else if (QString::compare(((*addInList)[pluginNum])->objectName(), name, Qt::CaseInsensitive) != 0)
    {
        retval += ito::RetVal(ito::retError, 0, tr("Wrong plugin name").toLatin1().data());
    }
    else
    {
        aib = qobject_cast<ito::AddInInterfaceBase *>((*addInList)[pluginNum]);
        m_plugInModel->insertInstance(aib, true); //begin insert
        modelInsertAnnounced = true;

        try
        {
            retval += aib->getAddInInst(reinterpret_cast<ito::AddInBase **>(addIn));
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError, 0, "Exception during call of constructor of plugin.");
        }

        if (!retval.containsError() && (*addIn) == NULL)
        {
            retval += ito::RetVal(ito::retError, 0, tr("Plugin instance is invalid (NULL)").toLatin1().data());
        }
    }

    if (!retval.containsError())
    {
        //ref-count of plugin must be zero (that means one instance is holder a single reference), this is rechecked in the following line
        if (aib->getRef((*addIn)) != 0)
        {
            retval += ito::RetVal(ito::retWarning, 0,
                tr("Reference counter of plugin has to be initialized with zero. "
                    "This is not the case for this plugin (Please contact the plugin developer).").toLatin1().data());
        }

        if ((*addIn)->getBasePlugin() == NULL || (*addIn)->getBasePlugin()->getType() == 0)
        {
            retval += ito::RetVal(ito::retError, 2000, tr("Base plugin or appropriate plugin type not indicated for this plugin.").toLatin1().data());
        }
    }

    if (!retval.containsError())
    {
        retval += initDockWidget((*addIn));

        callInitInNewThread = (*addIn)->getBasePlugin()->getCallInitInNewThread();
        if (QApplication::instance() && callInitInNewThread)
        {
            (*addIn)->MoveToThread();
        }

        waitCond = new ItomSharedSemaphore();
        Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;

        QMetaObject::invokeMethod(
            (*addIn),
            "init",
            conType,
            Q_ARG(QVector<ito::ParamBase>*, paramsMand),
            Q_ARG(QVector<ito::ParamBase>*, paramsOpt),
            Q_ARG(ItomSharedSemaphore*, waitCond));

        //this gives the plugin's init method to invoke a slot of any gui instance of the plugin within its init method. Else this slot is called after
        //having finished this initAddIn method (since main thread is blocked).
        while (!waitCond->waitAndProcessEvents(m_timeOutInitClose, QEventLoop::ExcludeUserInputEvents | QEventLoop::ExcludeSocketNotifiers))
        {
            if (!(*addIn)->isAlive())
            {
                retval += ito::RetVal(ito::retError, 0, tr("Timeout while initializing dataIO").toLatin1().data());
                timeoutOccurred = true;
                break;
            }
        }

        retval += waitCond->returnValue;
        waitCond->deleteSemaphore();
        waitCond = NULL;

        if (QApplication::instance() && !callInitInNewThread)
        {
            (*addIn)->MoveToThread();
        }

        if (timeoutOccurred == true)
        {
            //increment depending addIns in order to keep their reference alive while this plugin is in a undefined status.
            incRefParamPlugins((*addIn), paramsMand, paramsOpt);

            retval += registerPluginAsDeadPlugin((*addIn));
            (*addIn) = NULL;
        }
        else
        {
            //no timeout
            if (retval.containsError() ||
                (actuatorNotDataIO && !((*addIn)->getBasePlugin()->getType() & ito::typeActuator)) ||
                (!actuatorNotDataIO && !((*addIn)->getBasePlugin()->getType() & ito::typeDataIO)))
            {
                if (modelInsertAnnounced)
                {
                    m_plugInModel->insertInstance(aib, false); //end insert, since closeAddIn will call beginRemoveRows...
                    modelInsertAnnounced = false;
                }

                retval += closeAddIn((*addIn));

                (*addIn) = NULL;
            }
            else
            {
                incRefParamPlugins((*addIn), paramsMand, paramsOpt);

                policy = (*addIn)->getBasePlugin()->getAutoLoadPolicy();

                if (policy == ito::autoLoadAlways || (policy == ito::autoLoadKeywordDefined && autoLoadPluginParams))
                {
                    retval += loadParamVals((*addIn));
                }
            }
        }
    }

    if (modelInsertAnnounced)
    {
        m_plugInModel->insertInstance(aib, false); //end insert
    }

    if (aimWait)
    {
        aimWait->returnValue = retval;
        aimWait->release();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
/** closeAddIn  close an instance of an actuator addIn object
*   @param [in]  addIn  the addIn to close
*   @return      on success ito::retOk, ito::retError otherwise
*
*   At first the close method of the plugin class is invoked.
*   Then the \ref closeInst method of the addInInterfaceBase is called.
*/
const ito::RetVal AddInManagerPrivate::closeAddIn(AddInBase *addIn, ItomSharedSemaphore *aimWait)
{
    ItomSharedSemaphoreLocker locker(aimWait);
    ito::RetVal retval = ito::retOk;
    ItomSharedSemaphore *waitCond = NULL;
    bool timeout = false;

    ito::AddInInterfaceBase *aib = addIn->getBasePlugin();

    if (aib->getRef(addIn) <= 0) //this instance holds the last reference of the plugin. close it now.
    {
        // if the plugin is an actuator, raise its interrupt flag
        // to make sure that a possible movement is stopped as
        // soon as possible (if implemented in the specific actuator
        // plugin). Then the instance can be closed.
        ito::AddInActuator *aia = qobject_cast<ito::AddInActuator*>(addIn);

        if (aia)
        {
            aia->setInterrupt();
        }

        if (QApplication::instance())
        {
            //we always promised that if the init-method is called in the main thread,
            // the close method is called in the main thread, too.
            //Therefore pull it to the main thread, if necessary.
            if (!aib->getCallInitInNewThread())
            {
                ItomSharedSemaphoreLocker moveToThreadLocker(new ItomSharedSemaphore());

                if (QMetaObject::invokeMethod(
                        addIn,
                        "moveBackToApplicationThread",
                        Q_ARG(ItomSharedSemaphore*,
                        moveToThreadLocker.getSemaphore())))
                {
                    if (moveToThreadLocker->wait(m_timeOutInitClose) == false)
                    {
                        retval += ito::RetVal(ito::retWarning, 0,
                            tr("Timeout while pulling plugin back to main thread.").toLatin1().data());
                    }
                }
                else
                {
                    moveToThreadLocker->deleteSemaphore();
                    retval += ito::RetVal(ito::retWarning, 0,
                        tr("Error invoking method 'moveBackToApplicationThread' of plugin.").toLatin1().data());
                }
            }

            waitCond = new ItomSharedSemaphore();

            QMetaObject::invokeMethod(addIn, "close", Q_ARG(ItomSharedSemaphore*, waitCond));

            while (waitCond->wait(m_timeOutInitClose) == false && !timeout)
            {
                if (addIn->isAlive() == 0)
                {
                    retval += ito::RetVal(ito::retError, 0, tr("Timeout while closing plugin").toLatin1().data());
                    timeout = true;
                    break;
                }
            }
        }

        if (!timeout)
        {
            retval += waitCond->returnValue;
            if (QApplication::instance())
            {
                if (aib->getCallInitInNewThread())
                {
                    ItomSharedSemaphoreLocker moveToThreadLocker(new ItomSharedSemaphore());

                    if (QMetaObject::invokeMethod(addIn, "moveBackToApplicationThread", Q_ARG(ItomSharedSemaphore*, moveToThreadLocker.getSemaphore())))
                    {
                        if (moveToThreadLocker->wait(m_timeOutInitClose) == false)
                        {
                            retval += ito::RetVal(ito::retWarning, 0, tr("Timeout while pulling plugin back to main thread.").toLatin1().data());
                        }
                    }
                    else
                    {
                        moveToThreadLocker->deleteSemaphore();
                        retval += ito::RetVal(ito::retWarning, 0, tr("Error invoking method 'moveBackToApplicationThread' of plugin.").toLatin1().data());
                    }
                }
            }

            if (aib->getAutoSavePolicy() == ito::autoSaveAlways)
            {
                retval += saveParamVals(addIn);
            }

            m_plugInModel->deleteInstance(addIn, true); //begin remove

            retval += decRefParamPlugins(addIn);
            retval += aib->closeInst(&addIn);

            m_plugInModel->deleteInstance(addIn, false); //end remove
        }
        else
        {
            qDebug() << "Plugin could not be safely removed. Unknown state for plugin.";
            //TODO: what happens in the case that the close method does not return???
            //until now, we can not put it to the dead-plugin stack since this stack only handles plugins
            //that could not be initialized.
        }

        waitCond->deleteSemaphore();
        waitCond = NULL;

    }
    else
    {
        aib->decRef(addIn);
    }

    if (aimWait)
    {
        aimWait->returnValue = retval;
        aimWait->release();
    }

    return retval;
}


//-------------------------------------------------------------------------------------
/** decRef  decrement reference counter of addin and close it if necessary
*   @param [in]  addIn  the addIn to increment reference
*   @return      on success ito::retOk, ito::retError otherwise
*
*   The method decrements the reference counter of the addin.
*/
const ito::RetVal AddInManagerPrivate::decRef(ito::AddInBase **addIn)
{
    ito::AddInInterfaceBase *aib = (*addIn)->getBasePlugin();
    if (aib->getRef(*addIn) <= 0) //this instance holds the last reference of the plugin, therefore it is closed now
    {
        ito::RetVal retval(ito::retOk);
        retval += closeAddIn(*addIn);
        return retval;
    }
    else //at least one other instance is holding a reference of the plugin. just decrement the reference counter
    {
        aib->decRef(*addIn);
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** saveParamVals saves the plugins parameter values to the plugin parameter xml file
*   @param [in] plugin  plugin for which the parameter should be saved
*
*   A xml file with the same name as the plugin library in the plugin directory is used to save the plugin parameters. The xml file
*   is checked for the current plugin-file version and type when opened. The parameters are stored underneath the unique ID of
*   the instance currently closed. This enables to have a several parameter sets for one plugin. Each parameter is stored with its
*   name, type and value. The type may be either number or string.
*/
const ito::RetVal AddInManagerPrivate::saveParamVals(ito::AddInBase *plugin)
{
    ito::RetVal ret = ito::retOk;
    QFile paramFile;
    QString pluginUniqueId = plugin->getIdentifier();
    if (pluginUniqueId == "")
    {
        pluginUniqueId = QString::number(plugin->getID());
    }

    // Generate the filename

    ito::AddInInterfaceBase *aib = plugin->getBasePlugin();
    QString fname = aib->getFilename();

    if ((ret = generateAutoSaveParamFile(fname, paramFile)) == ito::retError)
    {
        return ret;
    }

    // Get the paremterlist
    QMap<QString, ito::Param> *paramList;


    if ((ret = plugin->getParamList(&paramList)) == ito::retError)
    {
        return ret;
    }

    QMap<QString, ito::Param> paramListCpy(*paramList);

    // Write parameter list to file
    if ((ret = saveQLIST2XML(&paramListCpy, pluginUniqueId, paramFile)) == ito::retError)
    {
        return ret;
    }

    if (ret.containsWarning())
    {
        return ret;
    }
    else
    {
        return ito::retOk;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** loadParamVals loads the plugins parameter values from the plugin parameter xml file
*   @param [in] plugin  plugin for which the parameter should be loaded
*
*   A xml file with the same name as the plugin library in the plugin directory is used to load the plugin parameters. The xml file
*   is checked for the current plugin-file version and type when opened. The parameters are set using the invokeMethod function on the
*   plugins' setParam method.
*/
const ito::RetVal AddInManagerPrivate::loadParamVals(ito::AddInBase *plugin)
{
    ito::RetVal ret = ito::retOk;
    QFile paramFile;
    QString pluginUniqueId = plugin->getIdentifier();
    if (pluginUniqueId == "")
    {
        pluginUniqueId = QString::number(plugin->getID());
    }
    ito::Param param1;

    ito::AddInInterfaceBase *aib = plugin->getBasePlugin();
    QString fname = aib->getFilename();

    if ((ret = generateAutoSaveParamFile(fname, paramFile)) == ito::retError)
    {
        return ito::RetVal(ito::retWarning, 0, ret.errorMessage());
    }

    // Get the paremterlist
    QMap<QString, ito::Param> *paramListPlugin;
    QMap<QString, ito::Param> paramListXML;

    if ((ret = plugin->getParamList(&paramListPlugin)) == ito::retError)
    {
        return ito::RetVal(ito::retWarning, 0, ret.errorMessage());
    }

    QMap<QString, ito::Param> paramListPluginCpy(*paramListPlugin);

    // Write parameter list to file
    if ((ret = loadXML2QLIST(&paramListXML, pluginUniqueId, paramFile)) == ito::retError)
    {
        return ito::RetVal(ito::retWarning, 0, ret.errorMessage());
    }

    if ((ret = mergeQLists(&paramListPluginCpy, &paramListXML, true, true)) == ito::retError)
    {
        return ito::RetVal(ito::retWarning, 0, ret.errorMessage());
    }

    ItomSharedSemaphore *waitCond = NULL;
    foreach(param1, paramListPluginCpy)
    {
        if (!strlen(param1.getName()))
        {
            continue;
        }

        QSharedPointer<ito::ParamBase> qsParam(new ito::ParamBase(param1));

        waitCond = new ItomSharedSemaphore();
        Qt::ConnectionType conType = (QApplication::instance() != NULL) ? Qt::AutoConnection : Qt::DirectConnection;

        QMetaObject::invokeMethod(plugin, "setParam", conType, Q_ARG(QSharedPointer<ito::ParamBase>, qsParam), Q_ARG(ItomSharedSemaphore*, waitCond));
        ret += waitCond->returnValue;
        waitCond->wait(m_timeOutGeneral);
        waitCond->deleteSemaphore();
        waitCond = NULL;
    }

    if (ret.containsError())
    {
        ret = ito::RetVal(ito::retWarning, 0, ret.errorMessage());
    }
    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AddInManagerPrivate::propertiesChanged()
{
    QString settingsFile("");
    if (ITOM_API_FUNCS)
    {
        settingsFile = apiGetSettingsFile();
        QSettings settings(settingsFile, QSettings::IniFormat);
        settings.beginGroup("AddInManager");
        if (QThread::idealThreadCount() < 0)
        {
            ito::AddInBase::setMaximumThreadCount(qBound(1, settings.value("maximumThreadCount", 2).toInt(), 2));
        }
        else
        {
            ito::AddInBase::setMaximumThreadCount(qBound(1, settings.value("maximumThreadCount", QThread::idealThreadCount()).toInt(), QThread::idealThreadCount()));
        }
        settings.endGroup();
    }
    else
    {
        ito::AddInBase::setMaximumThreadCount(QThread::idealThreadCount());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** initAddIn initialize new instance of a dataIO addIn class
*   @param [in]  addIn      pointer to newly initialized pluginIn
*   @return      on success ito::retOk
*
*   checks if addIn has a docking widget and if so, registers this docking widget to mainWindow
*/
ito::RetVal AddInManagerPrivate::initDockWidget(const ito::AddInBase *addIn)
{
    if (qobject_cast<QApplication*>(QCoreApplication::instance()))
    {
        QMainWindow *win = qobject_cast<QMainWindow*>(m_pMainWindow);
        if (addIn->getDockWidget() && win)
        {
            QDockWidget* dockWidget = addIn->getDockWidget();
            Qt::DockWidgetArea area;
            bool floating;
            bool visible;
            addIn->dockWidgetDefaultStyle(floating, visible, area);
            win->addDockWidget(area, dockWidget);
            dockWidget->setFloating(floating);
            dockWidget->setVisible(visible);

            /*bool restored =*/ win->restoreDockWidget(dockWidget);
        }
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------
QString AddInManagerPrivate::regExpAnchoredPattern(const QString& expression)
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    return QRegularExpression::anchoredPattern(expression);
#else
    return QString() + QLatin1String("\\A(?:") + expression + QLatin1String(")\\z");
#endif
}

//-------------------------------------------------------------------------------
QString AddInManagerPrivate::wildcardToRegularExpression(const QString &pattern)
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    // conversion should be anchored, hence, strict. Not partial.
    return QRegularExpression::wildcardToRegularExpression(pattern);
#else
    // from the Qt source code
    const qsizetype wclen = pattern.size();
    QString rx;
    rx.reserve(wclen + wclen / 16);
    qsizetype i = 0;
    const QChar *wc = pattern.data();
#ifdef Q_OS_WIN
    const QLatin1Char nativePathSeparator('\\');
    const QLatin1String starEscape("[^/\\\\]*");
    const QLatin1String questionMarkEscape("[^/\\\\]");
#else
    const QLatin1Char nativePathSeparator('/');
    const QLatin1String starEscape("[^/]*");
    const QLatin1String questionMarkEscape("[^/]");
#endif
    while (i < wclen) {
        const QChar c = wc[i++];
        switch (c.unicode()) {
        case '*':
            rx += starEscape;
            break;
        case '?':
            rx += questionMarkEscape;
            break;
        case '\\':
#ifdef Q_OS_WIN
        case '/':
            rx += QLatin1String("[/\\\\]");
            break;
#endif
        case '$':
        case '(':
        case ')':
        case '+':
        case '.':
        case '^':
        case '{':
        case '|':
        case '}':
            rx += QLatin1Char('\\');
            rx += c;
            break;
        case '[':
            rx += c;
            // Support for the [!abc] or [!a-c] syntax
            if (i < wclen) {
                if (wc[i] == QLatin1Char('!')) {
                    rx += QLatin1Char('^');
                    ++i;
                }
                if (i < wclen && wc[i] == QLatin1Char(']'))
                    rx += wc[i++];
                while (i < wclen && wc[i] != QLatin1Char(']')) {
                    // The '/' appearing in a character class invalidates the
                    // regular expression parsing. It also concerns '\\' on
                    // Windows OS types.
                    if (wc[i] == QLatin1Char('/') || wc[i] == nativePathSeparator)
                        return rx;
                    if (wc[i] == QLatin1Char('\\'))
                        rx += QLatin1Char('\\');
                    rx += wc[i++];
                }
            }
            break;
        default:
            rx += c;
            break;
        }
    }
    return regExpAnchoredPattern(rx);
#endif
}

} //end namespace ito


//explicit template instantiation
template const ito::RetVal ito::AddInManagerPrivate::initAddInActuatorOrDataIO<ito::AddInActuator>(
    bool, const int, const QString&, ito::AddInActuator**,
    QVector<ito::ParamBase>*, QVector<ito::ParamBase>*, bool, ItomSharedSemaphore*);

template const ito::RetVal ito::AddInManagerPrivate::initAddInActuatorOrDataIO<ito::AddInDataIO>(
    bool, const int, const QString&, ito::AddInDataIO**,
    QVector<ito::ParamBase>*, QVector<ito::ParamBase>*, bool, ItomSharedSemaphore*);
