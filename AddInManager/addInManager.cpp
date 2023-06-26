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
//#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI

//#include "global.h"
#include "addInManager.h"
#include "apiFunctions.h"
#include "../common/sharedFunctionsQt.h"
#include "../common/apiFunctionsGraphInc.h"
#include "addInManagerPrivate.h"

#include <qapplication.h>
#include <qmainwindow.h>
#include <QDirIterator>
#include <qaction.h>
#include <qsettings.h>
#include <qpointer.h>
#include <qtimer.h>
#include <qtranslator.h>
#include <qlibrary.h>
#include <qsharedpointer.h>

#include <QtCore/qpluginloader.h>

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
Q_DECLARE_METATYPE(const double *)
Q_DECLARE_METATYPE(int *)
Q_DECLARE_METATYPE(const int *)
Q_DECLARE_METATYPE(ito::AddInInterfaceBase *)
Q_DECLARE_METATYPE(ito::AddInBase *)
Q_DECLARE_METATYPE(ito::AddInBase **)
Q_DECLARE_METATYPE(ito::AddInDataIO **)
Q_DECLARE_METATYPE(ito::AddInActuator **)
Q_DECLARE_METATYPE(ito::AddInAlgo **)
//Q_DECLARE_METATYPE(ito::ActuatorAxis **)
Q_DECLARE_METATYPE(ito::RetVal *)
Q_DECLARE_METATYPE(ito::RetVal)
//Q_DECLARE_METATYPE(const void*)
Q_DECLARE_METATYPE(QVector<ito::Param> *)
Q_DECLARE_METATYPE(QVector<ito::ParamBase> *)
Q_DECLARE_METATYPE(QVector<int>)
Q_DECLARE_METATYPE(QVector<double>)

Q_DECLARE_METATYPE(QSharedPointer<double>)
Q_DECLARE_METATYPE(QSharedPointer<int>)
Q_DECLARE_METATYPE(QSharedPointer<QVector<double> >)
Q_DECLARE_METATYPE(QSharedPointer<char>)
Q_DECLARE_METATYPE(QSharedPointer<QByteArray>)
Q_DECLARE_METATYPE(QSharedPointer<ito::Param>)
Q_DECLARE_METATYPE(QSharedPointer<ito::ParamBase>)
Q_DECLARE_METATYPE(QSharedPointer<ito::DataObject>)

Q_DECLARE_METATYPE(QVector<QSharedPointer<ito::ParamBase> >)
Q_DECLARE_METATYPE(QSharedPointer<QVector<ito::ParamBase> >)
//Q_DECLARE_METATYPE(StringMap)

Q_DECLARE_METATYPE(ito::DataObject)


namespace ito
{

    AddInManager* AddInManager::staticInstance = NULL;

//----------------------------------------------------------------------------------------------------------------------------------
void **AddInManager::getItomApiFuncsPtr(void)
{
    return ITOM_API_FUNCS;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!> create a new instance of AddInManager as singleton class or returns the recently opened instance
/*static*/ AddInManager* AddInManager::createInstance(QString itomSettingsFile, void **apiFuncsGraph, QObject *mainWindow /*= NULL*/, QObject *mainApplication /*= NULL*/)
{
    if (staticInstance)
    {
        return staticInstance;
    }
    else
    {
        staticInstance = new AddInManager(itomSettingsFile, apiFuncsGraph, mainWindow, mainApplication);
        return staticInstance;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** closeInstance
*   @return ito::retOk
*
*   closes the instance of the AddInManager - should only be called at the very closing of the main program
*/
/*static*/ RetVal AddInManager::closeInstance(void)
{
    DELETE_AND_SET_NULL(staticInstance);
    return ito::retOk;
}



//----------------------------------------------------------------------------------------------------------------------------------
const QList<QObject *> * AddInManager::getDataIOList(void) const
{
    Q_D(const AddInManager);
    return &d->m_addInListDataIO;
}

//----------------------------------------------------------------------------------------------------------------------------------
const QList<QObject *> * AddInManager::getActList(void) const
{
    Q_D(const AddInManager);
    return &d->m_addInListAct;
}

//----------------------------------------------------------------------------------------------------------------------------------
const QList<QObject *> * AddInManager::getAlgList(void) const
{
    Q_D(const AddInManager);
    return &d->m_addInListAlgo;
}

//----------------------------------------------------------------------------------------------------------------------------------
const QHash<QString, ito::AddInAlgo::FilterDef *> * AddInManager::getFilterList(void) const
{
    Q_D(const AddInManager);
    return &d->m_filterList;
}

//----------------------------------------------------------------------------------------------------------------------------------
const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> * AddInManager::getAlgoWidgetList(void) const
{
    Q_D(const AddInManager);
    return &d->m_algoWidgetList;
}

//----------------------------------------------------------------------------------------------------------------------------------
const QList<PluginLoadStatus> AddInManager::getPluginLoadStatus() const
{
    Q_D(const AddInManager);
    return d->m_pluginLoadStatus;
}

//----------------------------------------------------------------------------------------------------------------------------------
const AlgoInterfaceValidator * AddInManager::getAlgoInterfaceValidator(void) const
{
    Q_D(const AddInManager);
    return d->m_algoInterfaceValidator;
}

//----------------------------------------------------------------------------------------------------------------------------------
PlugInModel * AddInManager::getPluginModel(void)
{
    Q_D(AddInManager);
    return d->m_plugInModel;
}

//----------------------------------------------------------------------------------------------------------------------------------
int AddInManager::getTotalNumAddIns(void) const
{
    Q_D(const AddInManager);
    return d->m_addInListDataIO.size()
        + d->m_addInListAct.size()
        + d->m_addInListAlgo.size();
}

//----------------------------------------------------------------------------------------------------------------------------------
void * AddInManager::getAddInPtr(const int itemNum)
{
    Q_D(const AddInManager);
    int num = itemNum;

    if (num < d->m_addInListAct.size())
    {
        return (void *)d->m_addInListAct[num];
    }
    else if (num -= d->m_addInListAct.size(), num < d->m_addInListAlgo.size())
    {
        return (void *)d->m_addInListAlgo[num];
    }
    else if (num -= d->m_addInListAlgo.size(), num < d->m_addInListDataIO.size())
    {
        return (void *)d->m_addInListDataIO[num];
    }
    else
    {
        return NULL;
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
int AddInManager::getItemIndexInList(const void *item)
{
    Q_D(const AddInManager);
    int num = 0;
    if ((num = d->m_addInListAct.indexOf((QObject*)item)) != -1)
    {
        return num;
    }
    else if ((num = d->m_addInListAlgo.indexOf((QObject*)item)) != -1)
    {
        return num;
    }
    else if ((num = d->m_addInListDataIO.indexOf((QObject*)item)) != -1)
    {
        return num;
    }
    else
    {
        return -1;
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
void AddInManager::updateModel(void)
{
    Q_D(AddInManager);
    d->m_plugInModel->update();
}



//----------------------------------------------------------------------------------------------------------------------------------
/** scanAddInDir
*   @param path directory path to search in (currently unused)
*   @return     returns ito::retOk on success otherwise ito::retError
*
*   This method searches the plugin directory which is currently assumed to be in the main programs folder
*   and must habe the name "plugins" for loadable plugins. The found plugins are sorted into the three lists
*   with the available plugins (ito::AddInManager::m_addInListDataIO, ito::AddInManager::m_addInListAct, ito::AddInManager::m_addInListAlg)
*/
const RetVal AddInManager::scanAddInDir(const QString &path, const int checkQCoreApp)
{
    Q_D(AddInManager);

    RetVal retValue = retOk;
    bool firstStart = false;
    bool pluginsFolderExists = true;
    QDir pluginsDir;

    // first we check if there exists any sort of QCoreApplication, which is required for qt event loop.
    // if not we start a QCoreApplication which is preferred for gui less systems. If the AddInManager has
    // been started via itom a QCoreApplication should (must) already exist here and nothing will happen
    if (checkQCoreApp && !QApplication::instance())
    {
        int argc = 0;
        d->m_pQCoreApp = new QCoreApplication(argc, NULL);
    }

    if (path.isEmpty() || path == "")
    {
        firstStart = true;
        d->m_plugInModel->resetModel(true);

        //search for base plugin folder
        pluginsDir = QDir(qApp->applicationDirPath());
        QString a = pluginsDir.dirName();

#if defined(WIN32)
        if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
        {
            pluginsDir.cdUp();
        }
#elif defined(__APPLE__)
        if (pluginsDir.dirName() == "MacOS")
        {
            pluginsDir.cdUp();
            pluginsDir.cdUp();
            pluginsDir.cdUp();
        }
#endif
        if (!pluginsDir.cd("plugins"))
        {
            //plugins-folder could not be found.
            pluginsFolderExists = false;
        }
    }
    else
    {
        pluginsDir.setPath(path);
    }

    if (pluginsDir.exists() == false)
    {
        QString dirErr = tr("Directory '%1' could not be found").arg(pluginsDir.canonicalPath());
        retValue += RetVal(retError, 0, dirErr.toLatin1().data());
    }
    else if (!pluginsFolderExists)
    {
        retValue += RetVal(retWarning, 0, tr("Plugins folder could not be found").toLatin1().data());
    }
    else
    {
        QString absoluteAddInPath;

        foreach (const QString &folderName, pluginsDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
        {
            absoluteAddInPath = QDir::cleanPath(pluginsDir.absoluteFilePath(folderName));
            retValue += scanAddInDir(absoluteAddInPath);
        }

        QStringList filters;
#ifdef linux
        filters << "*.a" << "*.so";
#elif (defined __APPLE__)
        filters << "*.a" << "*.dylib";
#else
        filters << "*.dll";
#endif

        foreach (const QString &fileName, pluginsDir.entryList(filters, QDir::Files))
        {
            absoluteAddInPath = QDir::cleanPath(pluginsDir.absoluteFilePath(fileName));
            if (QLibrary::isLibrary(absoluteAddInPath))
            {
                retValue += d->loadAddIn(absoluteAddInPath);
            }
        }
    }

    if (firstStart)
    {
        d->m_plugInModel->resetModel(false);
    }

    return retValue;
}



//----------------------------------------------------------------------------------------------------------------------------------
/** getInitParams
*   @param [in]  name        plugin name for which the initialisation parameters should be retrieved
*   @param [in]  pluginType  plugin type, i.e. in which of the plugin lists should be searched for the plugin
*   @param [out] pluginNum   number of the plugin in the plugin list, this number is needed later to create an instance of the plugin class
*   @param [out] paramsMand  mandatory initialisation parameters
*   @param [out] paramsOpt   optional initialisation parameters
*   @return      ito::retOk on success ito::retError otherwise
*
*   The getInitParams method searchs the plugin list given by plugin type for a plugin with the name 'name'. In case the according
*   plugin is found its number, mandatory and optional initialisation parameters are returned.
*
*   Please consider that this method returns pointers to the original initialization parameter vectors. If you change the value of these elements
*   consider to copy the complete vector.
*/
const RetVal AddInManager::getInitParams(const QString &name, const int pluginType, int *pluginNum, QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt)
{
    Q_D(AddInManager);

    ito::RetVal ret;
    *pluginNum = -1;

    if (pluginType & ito::typeActuator)
    {
        for (int n = 0; n < d->m_addInListAct.size(); n++)
        {
            if (QString::compare(d->m_addInListAct[n]->objectName(), name, Qt::CaseInsensitive) == 0)
            {
                *pluginNum = n;
                paramsMand = (qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListAct[n]))->getInitParamsMand();
                paramsOpt = (qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListAct[n]))->getInitParamsOpt();
                ret = ito::retOk;
                break;
            }
        }
    }
    else if (pluginType & ito::typeDataIO)
    {
        for (int n = 0; n < d->m_addInListDataIO.size(); n++)
        {
            if (QString::compare(d->m_addInListDataIO[n]->objectName(), name, Qt::CaseInsensitive) == 0)
            {
                *pluginNum = n;
                paramsMand = (qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListDataIO[n]))->getInitParamsMand();
                paramsOpt = (qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListDataIO[n]))->getInitParamsOpt();
                ret = ito::retOk;
                break;
            }
        }
    }
    else if (pluginType & ito::typeAlgo)
    {
        for (int n = 0; n < d->m_addInListAlgo.size(); n++)
        {
            if (QString::compare(d->m_addInListAlgo[n]->objectName(), name, Qt::CaseInsensitive) == 0)
            {
                *pluginNum = n;
                ret = ito::retOk;
                break;
            }
        }
    }
    else
    {
        ret += ito::RetVal(ito::retError, 0, tr("Invalid plugin type. Only typeDataIO, typeActuator or typeAlgo are allowed.").toLatin1().data());
    }

    if (*pluginNum < 0)
    {
        ret += ito::RetVal(ito::retError, 0, tr("Plugin '%1' not found in list of given type").arg(name).toLatin1().data());
    }

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** getAboutString
*   @param [in]  name               plugin name for which type and number should be retrieved
*   @param [out] versionString      plugin version string
*   @return      ito::retOk on success ito::retError otherwise
*
*   The getVersionString method searches in all three plugin lists for a plugin wirh the name 'name'. In case the according
*   plugin is found its inforamtion about the plugin version string is returned. In contrast to the version number returned
*   by the function 'getPluginInfo' the plugin version string can also handle alphanumerical signs.
*/
const RetVal ito::AddInManager::getAboutInfo(const QString &name, QString &versionString)
{
    Q_D(AddInManager);

    ito::RetVal ret = ito::RetVal(ito::retError, 0, QObject::tr("plugin not found").toLatin1().data());
    int found = 0;
    ito::AddInInterfaceBase *aib = NULL;
    try
    {
        //test actuator (objectName)
        for (int n = 0; n < d->m_addInListAct.size(); n++)
        {
            if (QString::compare(d->m_addInListAct[n]->objectName(), name, Qt::CaseInsensitive) == 0)
            {

                aib = qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListAct[n]);
                found = 1;
                break;
            }
        }
        if (!found) //test dataIO (objectName)
        {
            for (int n = 0; n < d->m_addInListDataIO.size(); n++)
            {
                if (QString::compare(d->m_addInListDataIO[n]->objectName(), name, Qt::CaseInsensitive) == 0)
                {


                    aib = qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListDataIO[n]);
                    found = 1;
                    break;
                }
            }
        }
        if (!found) //test Algorithm (objectName)
        {
            for (int n = 0; n < d->m_addInListAlgo.size(); n++)
            {
                if (QString::compare(d->m_addInListAlgo[n]->objectName(), name, Qt::CaseInsensitive) == 0)
                {
                    aib = qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListAlgo[n]);
                    found = 1;
                    break;
                }
            }
        }
        if (aib && found)
        {
            versionString = aib->getAboutInfo();

            ret = ito::retOk;
        }
    }
    catch (...)
    {
        ret += ito::RetVal(ito::retError, 0, tr("Caught exception during getPluginInfo of: %1").arg(name).toLatin1().data());
    }

    return ret;
}
//----------------------------------------------------------------------------------------------------------------------------------
/** getPlugInInfo
*   @param [in]  name               plugin name for which type and number should be retrieved
*   @param [out] pluginType         plugin type, i.e. in which of the plugin lists should be searched for the plugin
*   @param [out] pluginNum          number of the plugin in the plugin list, this number is needed later to create an instance of the plugin class
*   @param [out] pluginTypeString   type of the plugin as string
*   @param [out] author             author name or company
*   @param [out] description        short discribtion of the plugin
*   @param [out] detaildescription  detail discription of the plugin
*   @param [out] version            plugin version number
*   @return      ito::retOk on success ito::retError otherwise
*
*   The getPlugInInfo method searchs in all three plugin lists for a plugin with the name 'name'. In case the according
*   plugin is found its information about number, name ... returned. For all parameters of type char** provide the address to a char*-variable.
*   Then, a newly allocated \0-terminated string is returned. Don't forget to free this pointer after using it (free not delete!).
*/
const RetVal AddInManager::getPluginInfo(const QString &name, int &pluginType,
                                         int &pluginNum, int &version, QString &typeString,
                                         QString &author, QString &description,
                                         QString &detaildescription, QString &license,
                                         QString &about)
{
    Q_D(AddInManager);

    ito::RetVal ret = ito::RetVal(ito::retError, 0, QObject::tr("plugin not found").toLatin1().data());
    int found = 0;
    ito::AddInInterfaceBase *aib = NULL;

    try
    {
        //test actuator (objectName)
        for (int n = 0; n < d->m_addInListAct.size(); n++)
        {
            if (QString::compare(d->m_addInListAct[n]->objectName(), name, Qt::CaseInsensitive) == 0)
            {
                pluginNum = n;
                pluginType = ito::typeActuator;

                typeString = "Actuator";
                aib = qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListAct[n]);
                found = 1;
                break;
            }
        }

        if (!found) //test dataIO (objectName)
        {
            for (int n = 0; n < d->m_addInListDataIO.size(); n++)
            {
                if (QString::compare(d->m_addInListDataIO[n]->objectName(), name, Qt::CaseInsensitive) == 0)
                {
                    pluginNum = n;
                    pluginType = ito::typeDataIO;

                    typeString = "DataIO";
                    aib = qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListDataIO[n]);
                    found = 1;
                    break;
                }
            }
        }

        if (!found) //test Algorithm (objectName)
        {
            for (int n = 0; n < d->m_addInListAlgo.size(); n++)
            {
                if (QString::compare(d->m_addInListAlgo[n]->objectName(), name, Qt::CaseInsensitive) == 0)
                {
                    pluginNum = n;
                    pluginType = ito::typeAlgo;

                    typeString = "Algorithm";
                    aib = qobject_cast<ito::AddInInterfaceBase *>(d->m_addInListAlgo[n]);
                    found = 1;
                    break;
                }
            }
        }

        if (aib && found)
        {
            author = aib->getAuthor();
            description = aib->getDescription();
            detaildescription = aib->getDetailDescription();
            version = aib->getVersion();
            license = aib->getLicenseInfo();
            about = aib->getAboutInfo();
            ret = ito::retOk;
        }
    }
    catch (...)
    {
        ret += ito::RetVal(ito::retError, 0, tr("Caught exception during getPluginInfo of: %1").arg(name).toLatin1().data());
    }

    return ret;
}



//----------------------------------------------------------------------------------------------------------------------------------
/** initAddIn initialize new instance of a dataIO addIn class
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
ito::RetVal AddInManager::initAddIn(
    const int pluginNum, const QString &name,
    ito::AddInDataIO **addIn, QVector<ito::ParamBase> *paramsMand,
    QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams,
    ItomSharedSemaphore *aimWait)
{
    Q_D(AddInManager);

    try
    {
        return d->initAddInActuatorOrDataIO<ito::AddInDataIO>(false, pluginNum, name, addIn, paramsMand, paramsOpt, autoLoadPluginParams, aimWait);
    }
    catch (...)
    {
        QString txt = tr("Caught exception during initAddIn of: %1").arg(name);
        qDebug() << txt;
        return ito::RetVal(ito::retError, 0, txt.toLatin1().data());
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
/** initAddIn initialize new instance of a actuator addIn class
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
ito::RetVal AddInManager::initAddIn(
    const int pluginNum, const QString &name,
    ito::AddInActuator **addIn, QVector<ito::ParamBase> *paramsMand,
    QVector<ito::ParamBase> *paramsOpt, bool autoLoadPluginParams,
    ItomSharedSemaphore *aimWait)
{
    Q_D(AddInManager);

    try
    {
        return d->initAddInActuatorOrDataIO<ito::AddInActuator>(true, pluginNum, name, addIn, paramsMand, paramsOpt, autoLoadPluginParams, aimWait);
    }
    catch (...)
    {
        QString txt = tr("Caught exception during initAddIn of: %1").arg(name);
        qDebug() << txt;
        return ito::RetVal(ito::retError, 0, txt.toLatin1().data());
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
/** initAddIn initialize new instance of a algo addIn class
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
ito::RetVal AddInManager::initAddIn(
    const int pluginNum, const QString &name,
    ito::AddInAlgo **addIn, QVector<ito::ParamBase> * paramsMand,
    QVector<ito::ParamBase> * paramsOpt, bool autoLoadPluginParams,
    ItomSharedSemaphore *aimWait)
{
    Q_D(AddInManager);

    try
    {
        return d->initAddInAlgo(pluginNum, name, addIn, paramsMand, paramsOpt, autoLoadPluginParams, aimWait);
    }
    catch (...)
    {
        QString txt = tr("Caught exception during initAddIn of: %1").arg(name);
        qDebug() << txt;
        return ito::RetVal(ito::retError, 0, txt.toLatin1().data());
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
/** closeAddIn  close an instance of an actuator addIn object
*   @param [in]  addIn  the addIn to close
*   @return      on success ito::retOk, ito::retError otherwise
*
*   At first the close method of the plugin class is invoked. Then the \ref closeInst method of the addInInterfaceBase is called.
*/
ito::RetVal AddInManager::closeAddIn(AddInBase *addIn, ItomSharedSemaphore *aimWait)
{
    Q_D(AddInManager);

    try
    {
        return d->closeAddIn(addIn, aimWait);
    }
    catch (...)
    {
        QString txt = tr("Caught exception during closeAddIn of: %1").arg(addIn->getBasePlugin()->getFilename());
        qDebug() << txt;
        return ito::RetVal(ito::retError, 0, txt.toLatin1().data());
    }
}



//----------------------------------------------------------------------------------------------------------------------------------
/** incRef  increment reference counter of addin
*   @param [in]  addIn  the addIn to increment reference
*   @return      on success ito::retOk, ito::retError otherwise
*
*   The method increments the reference counter of the addin.
*/
const ito::RetVal AddInManager::incRef(ito::AddInBase *addIn)
{
    ito::AddInInterfaceBase *aib = addIn->getBasePlugin();
    aib->incRef(addIn);
    return ito::retOk;
}



//----------------------------------------------------------------------------------------------------------------------------------
/** decRef  decrement reference counter of addin and close it if necessary
*   @param [in]  addIn  the addIn to increment reference
*   @return      on success ito::retOk, ito::retError otherwise
*
*   The method decrements the reference counter of the addin.
*/
const ito::RetVal AddInManager::decRef(ito::AddInBase **addIn)
{
    Q_D(AddInManager);

    return d->decRef(addIn);
}



//----------------------------------------------------------------------------------------------------------------------------------
AddInManager::AddInManager(QString itomSettingsFile, void **apiFuncsGraph, QObject *mainWindow, QObject *mainApplication) :
    d_ptr(new AddInManagerPrivate(this))
{
    Q_D(AddInManager);
    ito::RetVal retValue;

    ApiFunctions::setSettingsFile(itomSettingsFile);
    ito::ITOM_API_FUNCS_GRAPH = apiFuncsGraph;

    // this needs to be done in combination with Q_DECLARE_METATYPE to register a user data type
    qRegisterMetaType<char*>("char*");
    qRegisterMetaType<char**>("char**");
    qRegisterMetaType<const char*>("const char*");
    qRegisterMetaType<const char**>("const char**");
    // incomplete type, i.e. void* are illegal in Qt5!
    //qRegisterMetaType<const void *>("const void *");
    qRegisterMetaType<double>("double");
    qRegisterMetaType<double *>("double*");
    //qRegisterMetaType<const double>();
    qRegisterMetaType<const double *>("const double*");
    qRegisterMetaType<int *>("int*");
    qRegisterMetaType<const int *>("const int*");
    //qRegisterMetaType<int>();
    qRegisterMetaType<ItomSharedSemaphore*>("ItomSharedSemaphore*");
    qRegisterMetaType<ito::AddInInterfaceBase*>("ito::AddInInterfaceBase*");
    qRegisterMetaType<ito::AddInBase*>("ito::AddInBase*");
    qRegisterMetaType<ito::AddInBase*>("ito::AddInBase**");
    qRegisterMetaType<ito::AddInDataIO**>("ito::AddInDataIO**");
    qRegisterMetaType<ito::AddInActuator**>("ito::AddInActuator**");
    qRegisterMetaType<ito::AddInAlgo**>("ito::AddInAlgo**");
//        qRegisterMetaType<ito::ActuatorAxis*>("ito::ActuatorAxis**");
    qRegisterMetaType<ito::RetVal>("ito::RetVal");
    qRegisterMetaType<ito::RetVal*>("ito::RetVal*");
//        qRegisterMetaType<const void*>("const void*");
    qRegisterMetaType<QVector<ito::Param>*>("QVector<ito::Param>*");
    qRegisterMetaType<QVector<ito::ParamBase>*>("QVector<ito::ParamBase>*");
    qRegisterMetaType<QVector<int> >("QVector<int>");
    qRegisterMetaType<QVector<double> >("QVector<double>");
    // used in plotItemsChanged do not remove
    qRegisterMetaType<QVector<float> >("QVector<float>");

    qRegisterMetaType<QSharedPointer<double> >("QSharedPointer<double>");
    qRegisterMetaType<QSharedPointer<QVector<double> > >("QSharedPointer<QVector<double> >");
    qRegisterMetaType<QSharedPointer<int> >("QSharedPointer<int>");
//    qRegisterMetaType<QSharedPointer<IntVector> >("QSharedPointer<IntVector>");
    qRegisterMetaType<QSharedPointer<char*> >("QSharedPointer<char>");
    qRegisterMetaType<QSharedPointer<QByteArray> >("QSharedPointer<QByteArray>");
    qRegisterMetaType<QSharedPointer<ito::Param> >("QSharedPointer<ito::Param>");
    qRegisterMetaType<QSharedPointer<ito::ParamBase> >("QSharedPointer<ito::ParamBase>");
    qRegisterMetaType<QSharedPointer<ito::DataObject> >("QSharedPointer<ito::DataObject>");

    qRegisterMetaType<ito::DataObject>("ito::DataObject");
    qRegisterMetaType<QMap<QString, ito::Param> >("QMap<QString, ito::Param>");
    qRegisterMetaType<QMap<QString, ito::Param> >("QMap<QString, ito::ParamBase>");
    qRegisterMetaType<QSharedPointer<QVector<ito::ParamBase> > >("QSharedPointer<QVector<ito::ParamBase> >");
    qRegisterMetaType<QVector<QSharedPointer<ito::ParamBase> > >("QVector<QSharedPointer<ito::ParamBase> >");

#if ITOM_POINTCLOUDLIBRARY > 0
    qRegisterMetaType<ito::PCLPointCloud >("ito::PCLPointCloud");
    qRegisterMetaType<ito::PCLPolygonMesh >("ito::PCLPolygonMesh");
    qRegisterMetaType<ito::PCLPoint >("ito::PCLPoint");
    qRegisterMetaType<QSharedPointer<ito::PCLPointCloud> >("QSharedPointer<ito::PCLPointCloud>");
    qRegisterMetaType<QSharedPointer<ito::PCLPolygonMesh> >("QSharedPointer<ito::PCLPolygonMesh>");
    qRegisterMetaType<QSharedPointer<ito::PCLPoint> >("QSharedPointer<ito::PCLPoint>");
#endif //#if ITOM_POINTCLOUDLIBRARY > 0

    d->m_deadPlugins.clear();

    d->propertiesChanged();

    d->m_algoInterfaceValidator = new AlgoInterfaceValidator(retValue);

    d->m_pMainApplication = mainApplication;
    if (mainApplication)
    {
        connect(d->m_pMainApplication, SIGNAL(propertiesChanged()), d, SLOT(propertiesChanged()));

    }
    d->m_pMainWindow = mainWindow;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** destructor, closes all instances of plugins and plugins
*
*   Before the AddInManager itself is closed it closes all instances of plugins that are in the plugins' instance lists.
*   Afterwards the AddInInterfaceBase for each plugin (i.e. the library) is closed an it is removed from the plugin list.
*   This is done for dataIO, actuator and algo plugins.
*/
AddInManager::~AddInManager(void)
{
    Q_D(AddInManager);

    ito::RetVal retval;
    AddInBase *addInInstance = NULL;
    QList<AddInBase*> addInInstancesCpy;
    AddInInterfaceBase *aib = NULL;

    //if there are still plugins in the "deadPlugin"-stack, try to kill them now
    d->closeDeadPlugins();
    d->m_deadPluginTimer.stop();
    foreach(QTranslator *Translator, d->m_Translator)
    {
        delete Translator;
    }
    d->m_Translator.clear();

    //we need to apply two steps in order to close all hardware-references
    //1. first -> close all opened instances (instances that keep reference to others must delete them after their deletion; ref-count values which are kept by other instances are ignored)
    //2. second -> delete all AddInInterfaceBase classes


    //step 1:
    QObjectList allHardwarePlugins = d->m_addInListDataIO + d->m_addInListAct;
    QMap<ito::AddInBase*, int> openedInstances;
    ito::AddInBase *aib2;

    //identify all opened devices including their current ref count
    foreach(QObject *obj, allHardwarePlugins)
    {
        aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
        foreach(ito::AddInBase* aib, aib->getInstList())
        {
            if (aib->getRefCount() >= 0)
            {
                openedInstances[aib] = aib->getRefCount();
            }
        }
    }

    //screen all opened devices and if they are referencing another plugin instance, decrement the target instance's ref count from openedInstances (since they are decremented by closing the owner plugin
    foreach(QObject *obj, allHardwarePlugins)
    {
        aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
        foreach(ito::AddInBase* aib, aib->getInstList())
        {
            QVector<ito::AddInBase::AddInRef *>* referencedInstances = aib->getArgAddIns();
            for (int i = 0; i < referencedInstances->size(); ++i)
            {
                aib2 = (ito::AddInBase*)referencedInstances->at(i)->ptr;
                if (openedInstances.contains(aib2))
                {
                    openedInstances[aib2]--;
                }
            }
        }
    }

    //try to close all remaining opened instances, whose ref count is still >= 0:
    QMapIterator<ito::AddInBase*, int> iter(openedInstances);
    while (iter.hasNext())
    {
        iter.next();
        if (iter.value() >= 0)
        {
            retval += closeAddIn(iter.key(), NULL);
        }
    }

    //now all plugin instances should be closed. If this is not the case, close it now... this is a safety thing.
    foreach(QObject *obj, allHardwarePlugins)
    {
        aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
        addInInstancesCpy = aib->getInstList(); //this copy is necessary in order to close every instance exactly one times (even if one instance is not deleted here but later, since another plugin still holds a reference to it)

        while (addInInstancesCpy.size() > 0)
        {
            addInInstance = (addInInstancesCpy[0]);
            if (addInInstance)
            {
                retval += closeAddIn(addInInstance, NULL);
            }
            addInInstancesCpy.removeFirst();
        }
    }

    //step 2:
    while (d->m_addInListDataIO.size() > 0)
    {
        QObject *qaib = d->m_addInListDataIO[0];
        AddInInterfaceBase *aib = (qobject_cast<ito::AddInInterfaceBase *>(qaib));
        d->m_addInListDataIO.removeFirst();
        QPluginLoader *loader = aib->getLoader();
        //loader->unload(); //under windows, unloading the plugin will sometimes not return. Therefore, no unload() here.
        DELETE_AND_SET_NULL(loader);
    }

    while (d->m_addInListAct.size() > 0)
    {
        QObject *qaib = d->m_addInListAct[0];
        AddInInterfaceBase *aib = (qobject_cast<ito::AddInInterfaceBase *>(qaib));
        d->m_addInListAct.removeFirst();
        QPluginLoader *loader = aib->getLoader();
        //loader->unload(); //under windows, unloading the plugin will sometimes not return. Therefore, no unload() here.
        DELETE_AND_SET_NULL(loader);
    }


    QHashIterator<void*, ito::FilterParams*> i(d->filterParamHash);
    while (i.hasNext())
    {
        i.next();
        delete i.value();
    }
    d->filterParamHash.clear();

    //remove all algorithms
    while (d->m_addInListAlgo.size() > 0)
    {
        QObject *qaib = d->m_addInListAlgo[0];
        AddInInterfaceBase *aib = (qobject_cast<ito::AddInInterfaceBase *>(qaib));
        while (aib->getInstList().size() > 0)
        {
            AddInAlgo *ail = reinterpret_cast<AddInAlgo *>(aib->getInstList()[0]);
            if (ail)
            {
                QHash<QString, ito::AddInAlgo::FilterDef *> funcList;
                ail->getFilterList(funcList);
                QList<QString> keyList = funcList.keys();
                for (int n = 0; n < keyList.size(); n++)
                {
                    if (d->m_filterList.contains(keyList[n]))
                    {
                        d->m_filterList.remove(keyList[n]);
                    }
                }

                aib->closeInst(reinterpret_cast<ito::AddInBase **>(&ail));
            }
        }
        d->m_addInListAlgo.removeFirst();
        QPluginLoader *loader = aib->getLoader();
        //loader->unload(); //under windows, unloading the plugin will sometimes not return. Therefore, no unload() here.
        DELETE_AND_SET_NULL(loader);
    }

    DELETE_AND_SET_NULL(d->m_algoInterfaceValidator);
}

//----------------------------------------------------------------------------------------------------------------------------------
const ito::RetVal AddInManager::setMainWindow(QObject *mainWindow)
{
    ito::RetVal retval(ito::retOk);
    Q_D(AddInManager);

    if (d->m_pMainWindow && d->m_pMainWindow != mainWindow)
    {
        retval += ito::RetVal(ito::retWarning, 0, tr("AddInManager already has an instance of mainWindow, ignoring new window").toLatin1().data());
    }
    else
    {
        d->m_pMainWindow = mainWindow;
    }

    return retval;
}



//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param name
*   @return ito::RetVal
*/
const ito::RetVal AddInManager::reloadAddIn(const QString &name)
{
    Q_D(AddInManager);

    ito::AddInInterfaceBase *aib = NULL;
    int pluginNum = d->getPluginNum(name, aib);
    QString filename = aib->getFilename();
    QList<ito::AddInBase *> instList;
    ito::AddInAlgo *algo = NULL;

    if ((((aib->getType() == ito::typeDataIO) || (aib->getType() == ito::typeActuator)) && (aib->getInstCount() != 0))
        || ((aib->getType() == ito::typeAlgo) && (aib->getInstCount() != 1)))
    {
        return ito::RetVal(ito::retError, 0, tr("Reference counter not zero. Only unused plugins can be reloaded.").toLatin1().data());
    }

    switch (aib->getType())
    {
        case ito::typeActuator:
            d->m_addInListAct.removeAt(pluginNum);
        break;

        case ito::typeDataIO:
            d->m_addInListDataIO.removeAt(pluginNum);
        break;

        case ito::typeAlgo:
            instList = aib->getInstList();
            for (int n = 0; n < instList.length(); n++)
            {
                algo = (ito::AddInAlgo*)instList.at(n);
                closeAddIn((ito::AddInBase*)algo);
            }
            d->m_addInListAlgo.removeAt(pluginNum);
        break;
    }

    delete aib;
    d->loadAddIn(filename);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** showConfigDialog    show the plugin's configuration dialog
*   @param [in] addin   addin from which the dialog should be called
*
*   This method opens the configuration dialog of a plugin. The dialog can be opened using a right click on an instance of the plugin
*   in the addInModel list or using showConfiguration command in python. An implementation of a configuration dialog is not mandatory,
*   so in case there is no dialog implemented nothing happens.
*/
ito::RetVal AddInManager::showConfigDialog(ito::AddInBase *addin, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;

    if (qobject_cast<QApplication*>(QCoreApplication::instance()))
    {
        //itom has been loaded with GUI support
        if (addin && addin->hasConfDialog())
        {
            retval += addin->showConfDialog();
        }
        else
        {
            retval += ito::RetVal(ito::retWarning, 0, tr("No configuration dialog available").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retWarning, 0, tr("No suitable qapplication / window could be detected, not loading dockWidget").toLatin1().data());
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** showDockWidget              show or hide the plugin's widget
*   @param [in] addin           addin from which the dialog should be called
*   @param [in] visible         1=show, 0=hide, -1=toggle
*   @param [in, out] waitCond   wait condition for calls from other threads. See also \ref ItomSharedSemaphore
*
*   This method opens or closes the widget of a plugin. The widget can be opened or closed using a right click on an instance of the
*   plugin in the addInModel list or using showToolbox or hideToolbox command in python. An implementation of a configuration dialog
*   is not mandatory, so in case there is no dialog implemented nothing happens.
*/
ito::RetVal AddInManager::showDockWidget(ito::AddInBase *addin, int visible, ItomSharedSemaphore *waitCond /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;

    if (qobject_cast<QApplication*>(QCoreApplication::instance()))
    {
        if (addin)
        {
            QDockWidget *dw = addin->getDockWidget();
            if (dw)
            {
                QAction *toggleAction = dw->toggleViewAction();
                if (visible == 0) //hide
                {
                    if (toggleAction->isChecked()) //dock widget is currently visible -> hide it now
                    {
                        toggleAction->trigger();
                    }
                }
                else if (visible == 1) //show
                {
                    if (toggleAction->isChecked() == false) //dock widget is currently hidden -> show it now
                    {
                        toggleAction->trigger();
                    }
                }
                else //toggle
                {
                    toggleAction->trigger();
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, tr("No toolbox available").toLatin1().data());
            }
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("Plugin not available").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retWarning, 0, tr("No suitable qapplication / window could be detected, not loading dockWidget").toLatin1().data());
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param algoWidgetName algoPluginName
*   @param algoPluginName
*   @return ito::AddInAlgo::AlgoWidgetDef
*/
const ito::AddInAlgo::AlgoWidgetDef * AddInManager::getAlgoWidgetDef(QString algoWidgetName, QString algoPluginName)
{
    //at the moment algoPluginName do not really influence the search, but maybe it might become necessary to also search for plugin-widgets by "pluginName.widgetName"

    const QHash<QString, ito::AddInAlgo::AlgoWidgetDef *> *list = getAlgoWidgetList();
    QHash<QString, ito::AddInAlgo::AlgoWidgetDef *>::ConstIterator iter = list->find(algoWidgetName);
    if (iter == list->end() || iter.value() == NULL)
    {
        return NULL;
    }
    else
    {
        ito::AddInInterfaceBase *aib = iter.value()->m_pBasePlugin;
        if (aib && (aib->objectName() == algoPluginName  || algoPluginName.isEmpty() || algoPluginName == ""))
        {
            return iter.value();
        }
        else if (aib == NULL && (algoPluginName.isEmpty() || algoPluginName == ""))
        {
            return iter.value();
        }
    }

    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param filterParam filterParam
*   @return ito::FilterParams
*/
const ito::FilterParams* AddInManager::getHashedFilterParams(ito::AddInAlgo::t_filterParam filterParam) const
{
    Q_D(const AddInManager);

    QHash<void*,ito::FilterParams*>::ConstIterator it = d->filterParamHash.constFind((void*)filterParam);
    if (it != d->filterParamHash.constEnd())
    {
        return *it;
    }
    return NULL;
}



//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param plugin
*   @return bool
*/
bool AddInManager::isPluginInstanceDead(const ito::AddInBase *plugin) const
{
    Q_D(const AddInManager);

    foreach(const QPointer<ito::AddInBase> ptr, d->m_deadPlugins)
    {
        if (!ptr.isNull() && ptr.data() == plugin)
        {
            return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param iface
*   @param tag
*   @return QList<ito::AddInAlgo::FilterDef *>
*/
const QList<ito::AddInAlgo::FilterDef *> AddInManager::getFilterByInterface(ito::AddInAlgo::tAlgoInterface iface, const QString tag) const
{
    Q_D(const AddInManager);

    if (tag.isNull())
    {
        QList<ito::AddInAlgo::FilterDef *> res;
        QHash<QString, ito::AddInAlgo::FilterDef *>::const_iterator it = d->m_filterList.constBegin();
        while (it != d->m_filterList.constEnd())
        {
            if (it.value()->m_interface == iface) res.append(*it);
            ++it;
        }
        return res;
    }
    else
    {
        QString key = QString::number(iface) + "_" + tag;
        return d->m_filterListInterfaceTag.values(key);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param cat
*   @return QList<ito::AddInAlgo::FilterDef *>
*/
const QList<ito::AddInAlgo::FilterDef *> AddInManager::getFiltersByCategory(ito::AddInAlgo::tAlgoCategory cat) const
{
    Q_D(const AddInManager);

    QList<ito::AddInAlgo::FilterDef *> res;
    QHash<QString, ito::AddInAlgo::FilterDef *>::const_iterator it = d->m_filterList.constBegin();
    while (it != d->m_filterList.constEnd())
    {
        if (it.value()->m_category == cat)
        {
            res.append(*it);
        }
        ++it;
    }
    return res;
}

//----------------------------------------------------------------------------------------------------------------------------------
/**
*   @param iface
*   @param cat
*   @param tag
*   @return QList<ito::AddInAlgo::FilterDef
*/
const QList<ito::AddInAlgo::FilterDef *> AddInManager::getFilterByInterfaceAndCategory(ito::AddInAlgo::tAlgoInterface iface, ito::AddInAlgo::tAlgoCategory cat, const QString tag) const
{
    QList<ito::AddInAlgo::FilterDef *> res = getFilterByInterface(iface,tag);
    QList<ito::AddInAlgo::FilterDef *> res2;
    for(int i=0; i<res.size(); i++)
    {
        if (res[i]->m_category == cat) res2.append(res[i]);
    }
    return res2;
}

//----------------------------------------------------------------------------------------------------------------------------------
const ito::RetVal AddInManager::setTimeOuts(const int initClose, const int general)
{
    Q_D(AddInManager);
    d->m_timeOutInitClose = initClose;
    d->m_timeOutGeneral = general;
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!> interrupts all active actuator instances
ito::RetVal AddInManager::interruptAllActuatorInstances(ItomSharedSemaphore *aimWait /*= NULL*/)
{
    ItomSharedSemaphoreLocker locker(aimWait);
    ito::RetVal retval;

    const QList<QObject*>* actuatorPlugins = getActList();

    if (actuatorPlugins)
    {
        const ito::AddInInterfaceBase *aib;
        ito::AddInActuator *aia;

        for (int i = 0; i < actuatorPlugins->size(); ++i)
        {
            aib = qobject_cast<const ito::AddInInterfaceBase*>(actuatorPlugins->at(i));
            if (aib)
            {
                foreach(ito::AddInBase* aib, aib->getInstList())
                {
                    aia = qobject_cast<ito::AddInActuator*>(aib);
                    if (aia)
                    {
                        aia->setInterrupt();
                    }
                }
            }
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, "actuatorPlugins invalid.");
    }

    if (aimWait)
    {
        aimWait->returnValue = retval;
        aimWait->release();
    }

    return retval;
}

} // namespace ito
