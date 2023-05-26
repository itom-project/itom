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

#include "designerWidgetOrganizer.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../common/apiFunctionsGraphInc.h"
#include "../common/apiFunctionsInc.h"
#include "../common/abstractApiWidget.h"
#include "plot/AbstractFigure.h"
#include "plot/AbstractDObjFigure.h"
#include "../common/semVerVersion.h"

#include <qmetaobject.h>
#include <qpluginloader.h>
#include <QtUiPlugin/QDesignerCustomWidgetInterface>
#include <qsettings.h>
#include <qcoreapplication.h>
#include <qdir.h>
#include <QDirIterator>
#include <qapplication.h>
#include <qregularexpression.h>
#include <qpen.h>
#include <qnamespace.h>

/*!
    \class DesignerWidgetOrganizer
    \brief
*/

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!

*/
DesignerWidgetOrganizer::DesignerWidgetOrganizer(ito::RetVal &retValue)
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DesignerPlugins");
    settings.beginGroup("ito::AbstractFigure");
    if (settings.contains("titleFont") == false) settings.setValue("titleFont", QFont("Verdana", 12));
    if (settings.contains("labelFont") == false) settings.setValue("labelFont", QFont("Verdana", 10));
    if (settings.contains("axisFont") == false) settings.setValue("axisFont", QFont("Verdana", 10));
    if (settings.contains("lineStyle") == false) settings.setValue("lineStyle", (int)Qt::SolidLine);
    if (settings.contains("lineWidth") == false) settings.setValue("lineWidth", qreal(1.0));
    if (settings.contains("legendFont") == false) settings.setValue("legendFont", QFont("Verdana", 8));
    if (settings.contains("zoomRubberBandPen") == false) settings.setValue("zoomRubberBandPen", QPen(QBrush(Qt::red), 2, Qt::DashLine));
    if (settings.contains("trackerPen") == false) settings.setValue("trackerPen", QPen(QBrush(Qt::red), 2));
    if (settings.contains("trackerFont") == false) settings.setValue("trackerFont", QFont("Verdana", 10));
    if (settings.contains("trackerBackground") == false) settings.setValue("trackerBackground", QBrush(QColor(255, 255, 255, 255), Qt::SolidPattern));
    if (settings.contains("centerMarkerSize") == false) settings.setValue("centerMarkerSize", QSize(25, 25));
    if (settings.contains("centerMarkerPen") == false) settings.setValue("centerMarkerPen", QPen(QBrush(Qt::red), 1));
    settings.endGroup();
    settings.endGroup();

    //create figure categories (for property dialog...)
   ito::PlotDataFormats allFormats = ~(ito::PlotDataFormats(0)); //(~ito::Format_Gray8); // | ito::Format_Gray8; //(ito::PlotDataFormats(0));

    m_figureCategories["DObjLiveLine"] = FigureCategory("Data Object, Line Plot, Live", ito::DataObjLine, allFormats, ito::Live | ito::PlotLine, ito::PlotFeature(), "Itom1DQwtPlot");
   m_figureCategories["DObjLiveImage"] = FigureCategory("Data Object, 2D Image Plot, Live", ito::DataObjPlane, allFormats, ito::Live | ito::PlotImage, ito::PlotFeature(), "Itom2DQwtPlot");
    m_figureCategories["DObjStaticLine"] = FigureCategory("Data Object, Line Plot, Static", ito::DataObjLine, allFormats, ito::Static | ito::PlotLine, ito::PlotFeature(), "Itom1DQwtPlot");
   m_figureCategories["DObjStaticImage"] = FigureCategory("Data Object, 2D Image Plot, Static", ito::DataObjPlane | ito::DataObjPlaneStack, allFormats, ito::Static | ito::PlotImage, ito::PlotFeature(), "Itom2DQwtPlot");
    m_figureCategories["DObjStaticGeneralPlot"] = FigureCategory("Data Object, Any Planar Plot, Static", ito::DataObjLine | ito::DataObjPlane | ito::DataObjPlaneStack, allFormats, ito::Static, ito::Plot3D | ito::PlotISO, "Itom2DQwtPlot");
    m_figureCategories["PerspectivePlot"] = FigureCategory("Data Object, Any Planar Plot, Point Clouds, Polygon Meshes, Static", ito::DataObjPlane | ito::DataObjPlaneStack | ito::PointCloud | ito::PolygonMesh, allFormats, ito::Static | ito::Plot3D | ito::PlotISO, ito::Live | ito::PlotLine, "TwipOGLFigure");

    retValue += scanDesignerPlugins();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!

*/
DesignerWidgetOrganizer::~DesignerWidgetOrganizer()
{
    foreach(ito::FigurePlugin p, m_figurePlugins)
    {
        p.factory->unload();
        DELETE_AND_SET_NULL(p.factory);
    }
    m_figurePlugins.clear();
    m_figureCategories.clear();

    foreach (QTranslator *Translator, m_Translator)
    {
        delete Translator;
    }
    m_Translator.clear();
}

//------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \return
*/
RetVal DesignerWidgetOrganizer::scanDesignerPlugins()
{
    QDir dir(QCoreApplication::applicationDirPath());
    dir.cd("designer");
    QStringList candidates = dir.entryList(QDir::Files);
    QString absolutePluginPath;
    FigurePlugin infoStruct;
    PluginLoadStatus status;
    QString message;
    QPluginLoader *loader = NULL;
    SemVerVersion requiredItomDesignerPluginInterface =
        SemVerVersion::fromInt(ITOM_DESIGNERPLUGININTERFACE_VERSION);
    SemVerVersion requiredAddInInterfaceVersion(
            ITOM_ADDININTERFACE_MAJOR,
            ITOM_ADDININTERFACE_MINOR,
            ITOM_ADDININTERFACE_PATCH);
    const QMetaObject *metaObj = NULL;

    //This regular expression is used to check whether the error message
    //during loading a plugin contains the words
    //'debug' or 'release'. This means, that a release plugin is tried to be
    //loaded with a debug version of itom or vice-versa
    QRegularExpression regExpDebugRelease(".*(release|debug).*", QRegularExpression::CaseInsensitiveOption);

    foreach(const QString &plugin, candidates)
    {
        if (plugin.indexOf("itomWidgets", 0, Qt::CaseInsensitive) == -1)
        {
            absolutePluginPath = QDir::cleanPath(dir.absoluteFilePath(plugin));
            status.filename = absolutePluginPath;
            status.messages.clear();

            if (QLibrary::isLibrary(absolutePluginPath))
            {
                //load translation file
                QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
                QStringList startupScripts;

                settings.beginGroup("Language");
                QString language = settings.value("language", "en").toString();
                QByteArray codec =  settings.value("codec", "UTF-8").toByteArray();
                settings.endGroup();

                QFileInfo fileInfo(absolutePluginPath);
                QDir fileInfoDir = fileInfo.dir();
                fileInfoDir.cdUp();
                if (language != "en_US" && fileInfoDir.absolutePath() == qApp->applicationDirPath())
                {
                    QLocale local = QLocale(language); //language can be "language[_territory][.codeset][@modifier]"
                    QString translationPath = fileInfo.path() + "/translation";
                    QString languageStr = local.name().left(local.name().indexOf("_", 0, Qt::CaseInsensitive));
                    QString baseFileName = fileInfo.baseName();
                    baseFileName.replace("d", "*");
                    QDirIterator it(translationPath, QStringList(baseFileName + "_" + languageStr + ".qm"), QDir::Files);
                    if (it.hasNext())
                    {
                        QString translationLocal = it.next();
                        m_Translator.append(new QTranslator);
                        m_Translator.last()->load(translationLocal, translationPath);
                        if (m_Translator.last()->isEmpty())
                        {
                            message = QObject::tr("Unable to load translation file '%1'. Translation file is empty.").arg(translationPath + '/' + translationLocal);
                            qDebug() << message;
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfWarning, message));
                        }
                        else
                        {
                            QCoreApplication::instance()->installTranslator(m_Translator.last());
                        }
                    }
                    else
                    {
                        message = QObject::tr("Unable to find translation file.");
                        qDebug() << message;
                        status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfWarning, message));
                    }
                }

                bool success = false;
                ito::PluginLoadStatusFlags pluginStatus = plsfOk;

                loader = new QPluginLoader(absolutePluginPath);

                QJsonObject metaData = loader->metaData();
                QJsonValue metaDataDebug = metaData["debug"];
                QJsonValue metaDataIid = metaData["IID"];
                QJsonValue metaDataUser = metaData["MetaData"];
                QStringList keys = metaData.keys();

                if (metaData.isEmpty() ||
                    metaDataDebug.isUndefined() ||
                    metaDataIid.isUndefined())
                {
                    message = QString("Could not load meta data or mandatory items for the library '%1'. This file is probably no valid plugin.").\
                        arg(absolutePluginPath);
                    pluginStatus = plsfIgnored;
                    status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                }
                else if (metaDataIid.toString() != "org.qt-project.Qt.QDesignerCustomWidgetInterface")
                {
                    message = QString("The file '%1' is no valid Qt designer plugin inherited from QDesignerCustomWidgetInterface").arg(absolutePluginPath);
                    pluginStatus = plsfWarning;
                    status.messages.append(
                            QPair<ito::PluginLoadStatusFlags,
                            QString>(pluginStatus, message));
                }
                else
                {
                    bool debug = metaDataDebug.toBool();
#ifdef _DEBUG
                    if (!debug)
                    {
                        message = QString("The designer plugin '%1' seems to be a release version and cannot be loaded in a debug build of itom.").
                            arg(absolutePluginPath);
                        pluginStatus = ito::PluginLoadStatusFlags(plsfWarning | plsfRelDbg);
                        status.messages.append(QPair<ito::PluginLoadStatusFlags,
                                QString>(pluginStatus, message));
                    }
#else
                    if (debug)
                    {
                        message = QString("The designer plugin '%1' seems to be a debug version and cannot be loaded in a release build of itom.").
                            arg(absolutePluginPath);
                        pluginStatus = ito::PluginLoadStatusFlags(plsfWarning | plsfRelDbg);
                        status.messages.append(QPair<ito::PluginLoadStatusFlags,
                                QString>(pluginStatus, message));
                    }
#endif
                }

                if (pluginStatus == plsfOk)
                {
                    QJsonObject obj = metaDataUser.toObject();
                    QJsonValue aiiVersion = obj.contains("ito.addInInterface.version") ? obj["ito.addInInterface.version"] : QJsonValue();
                    QJsonValue idpVersion = obj.contains("ito.itomDesignerPlugin.version") ? obj["ito.itomDesignerPlugin.version"] : QJsonValue();

                    bool a = metaDataUser.isUndefined();
                    bool b = obj.isEmpty();
                    bool c = idpVersion.isUndefined();

                    if (metaDataUser.isUndefined() || obj.isEmpty() ||
                        idpVersion.isUndefined() ||
                        aiiVersion.isUndefined())
                    {
                        message = QString("The designer plugin '%1' does not contain valid meta information for an itom designer plugin (itom > 3.2.1). Maybe this plugin is too old.").
                            arg(absolutePluginPath);
                        pluginStatus = ito::PluginLoadStatusFlags(plsfError | plsfIncompatible);
                        status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                    }
                    else
                    {
                        SemVerVersion dpIfaceVersion = SemVerVersion::fromString(idpVersion.toString());
                        SemVerVersion addInIfaceVersion = SemVerVersion::fromString(aiiVersion.toString());

                        if (!dpIfaceVersion.isValid() && !addInIfaceVersion.isValid())
                        {
                            message = tr("The ito.itomDesignerPlugin.version of the meta data of the designer plugin in file '%1' is invalid."). \
                                arg(absolutePluginPath);
                            pluginStatus = ito::PluginLoadStatusFlags(plsfError | plsfIncompatible);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }
                        else if (!dpIfaceVersion.isValid())
                        {
                            message = tr("The ito.itomDesignerPlugin.version of the meta data of the designer plugin in file '%1' is invalid."). \
                                arg(absolutePluginPath);
                            pluginStatus = ito::PluginLoadStatusFlags(plsfError | plsfIncompatible);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }
                        else if (!addInIfaceVersion.isValid())
                        {
                            message = tr("The ito.addInInterface.version of the meta data of the designer plugin in file '%1' is invalid."). \
                                arg(absolutePluginPath);
                            pluginStatus = ito::PluginLoadStatusFlags(plsfError | plsfIncompatible);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }
                        else if (!requiredAddInInterfaceVersion.isCompatible(addInIfaceVersion))
                        {
                            message = tr("The ito.addInInterface.version (%1) of the meta data of the designer plugin in file '%2' is incompatible to the required version %3."). \
                                arg(aiiVersion.toString()).\
                                arg(absolutePluginPath). \
                                arg(requiredAddInInterfaceVersion.toString());
                            pluginStatus = ito::PluginLoadStatusFlags(plsfError | plsfIncompatible);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }
                        else if (!requiredItomDesignerPluginInterface.isCompatible(dpIfaceVersion))
                        {
                            message = tr("The ito.itomDesignerPlugin.version (%1) of the meta data of the designer plugin in file '%2' is incompatible to the required version %3."). \
                                arg(dpIfaceVersion.toString()).\
                                arg(absolutePluginPath). \
                                arg(requiredItomDesignerPluginInterface.toString());
                            pluginStatus = ito::PluginLoadStatusFlags(plsfError | plsfIncompatible);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }
                    }
                }

                if (pluginStatus == plsfOk)
                {
                    QDesignerCustomWidgetInterface *iface = NULL;
                    QObject *instance = loader->instance();

                    if (instance == NULL)
                    {
                        message = loader->errorString();
                        loader->unload();

                        if (message.indexOf(regExpDebugRelease) >= 0) //debug/release conflict is only a warning, no error
                        {
                            pluginStatus = ito::PluginLoadStatusFlags(plsfWarning | plsfRelDbg);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }
                        else
                        {
                            pluginStatus = ito::PluginLoadStatusFlags(plsfError);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                        }

                        DELETE_AND_SET_NULL(loader);
                    }
                    // try with a normal plugin, we do not support collections
                    else if (iface = qobject_cast<QDesignerCustomWidgetInterface *>(instance))
                    {
                        if (instance->inherits("ito::AbstractItomDesignerPlugin"))
                        {
                            //1. check if instance also implements the ItomDesignerPluginInterface (new from itom > 3.2.1 on):
                            ito::AbstractItomDesignerPlugin *itomDesignerPlugin = qobject_cast<ito::AbstractItomDesignerPlugin*>(instance);

                            if (itomDesignerPlugin)
                            {
                                ito::AbstractItomDesignerPlugin *absIDP = (ito::AbstractItomDesignerPlugin *)instance;
                                infoStruct.filename = absolutePluginPath;
                                infoStruct.classname = iface->name();
                                infoStruct.plotDataFormats = absIDP->getPlotDataFormats();
                                infoStruct.plotDataTypes = absIDP->getPlotDataTypes();
                                infoStruct.plotFeatures = absIDP->getPlotFeatures();
                                infoStruct.icon = iface->icon();
                                infoStruct.factory = loader; //now, loader is organized by m_figurePlugins-list
                                m_figurePlugins.append(infoStruct);

                                absIDP->setItomSettingsFile(AppManagement::getSettingsFile());

                                message = tr("DesignerWidget '%1' successfully loaded").arg(iface->name());
                                status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfOk, message));
                                success = true;
                            }
                            else
                            {
                                pluginStatus = ito::PluginLoadStatusFlags(plsfError);
                                message = tr("The designer plugin in file '%1' does not implement the ItomDesignerPluginInterface, required by itom > 3.2.1."). \
                                    arg(status.filename);
                                status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                            }
                        }
                        else
                        {
                            message = tr("Plugin in file '%1' is a Qt designer plugin but no itom plot widget that inherits 'ito.AbstractItomDesignerPlugin'").arg(status.filename);
                            status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfIgnored, message));
                        }
                    }
                    else
                    {
                        message = QString("The file '%1' is no valid Qt designer plugin inherited from QDesignerCustomWidgetInterface").arg(absolutePluginPath);
                        pluginStatus = plsfWarning;
                        status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(pluginStatus, message));
                    }
                }

                if (!success && loader)
                {
                    loader->unload();
                    DELETE_AND_SET_NULL(loader);
                }

                m_pluginLoadStatus.append(status);
            }
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param className
    \return bool
*/
bool DesignerWidgetOrganizer::figureClassExists(const QString &className)
{
    foreach(const FigurePlugin &plugin, m_figurePlugins)
    {
        if (QString::compare(plugin.classname, className, Qt::CaseInsensitive) == 0)
        {
            return true;
        }
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param className
    \param plotDataTypesMask
    \param plotDataFormatsMask
    \param plotFeaturesMask
    \param ok
    \return ito::RetVal
*/
ito::RetVal DesignerWidgetOrganizer::figureClassMinimumRequirementCheck(const QString &className, int plotDataTypesMask, int plotDataFormatsMask, int plotFeaturesMask, bool *ok)
{
    ito::RetVal retVal;
    bool success = false;

    foreach(const FigurePlugin &plugin, m_figurePlugins)
    {
        if (className == plugin.classname)
        {
            if ((plugin.plotDataTypes & plotDataTypesMask) != plotDataTypesMask)
            {
                retVal += ito::RetVal::format(ito::retError, 0, tr("Figure '%s' does not correspond to the minimum requirements").toLatin1().data(), className.toLatin1().data());
                break;
            }
            if ((plugin.plotDataFormats & plotDataFormatsMask) != plotDataFormatsMask)
            {
                retVal += ito::RetVal::format(ito::retError, 0, tr("Figure '%s' does not correspond to the minimum requirements").toLatin1().data(), className.toLatin1().data());
                break;
            }
            if ((plugin.plotFeatures & plotFeaturesMask) != plotFeaturesMask)
            {
                retVal += ito::RetVal::format(ito::retError, 0, tr("Figure '%s' does not correspond to the minimum requirements").toLatin1().data(), className.toLatin1().data());
                break;
            }

            success = true;
        }
    }

    if (retVal == ito::retOk && success == false)
    {
        retVal += ito::RetVal::format(ito::retError, 0, tr("Figure '%s' not found").toLatin1().data(), className.toLatin1().data());
    }

    if (ok)
    {
        *ok = success;
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param plotDataTypesMask
    \param plotDataFormatsMask
    \param plotFeaturesMask
    \return QList<FigurePlugin>
*/
QList<FigurePlugin> DesignerWidgetOrganizer::getPossibleFigureClasses(int plotDataTypesMask, int plotDataFormatsMask, int plotFeaturesMask)
{
    QList<FigurePlugin> figurePlugins;

    foreach(const FigurePlugin &plugin, m_figurePlugins)
    {
        if ((plugin.plotDataTypes & plotDataTypesMask) == plotDataTypesMask &&
            (plugin.plotDataFormats & plotDataFormatsMask) == plotDataFormatsMask &&
            (plugin.plotFeatures & plotFeaturesMask) == plotFeaturesMask)
        {
            figurePlugins.append(plugin);
        }
    }
    return figurePlugins;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param figureCat
    \return QList<FigurePlugin>
*/
QList<FigurePlugin> DesignerWidgetOrganizer::getPossibleFigureClasses(const FigureCategory &figureCat)
{
    QList<FigurePlugin> figurePlugins;

        QString         m_description;
    PlotDataTypes   m_allowedPlotDataTypes;
    PlotDataFormats m_allowedPlotDataFormats;
    PlotFeatures    m_requiredPlotFeatures;
    PlotFeatures    m_excludedPlotFeatures;
    QString         m_defaultClassName;

    foreach(const FigurePlugin &plugin, m_figurePlugins)
    {
        if ((plugin.plotDataTypes & figureCat.m_allowedPlotDataTypes) &&
            (plugin.plotDataFormats & figureCat.m_allowedPlotDataFormats) &&
            (plugin.plotFeatures & figureCat.m_excludedPlotFeatures) == 0 &&
            ((plugin.plotFeatures & figureCat.m_requiredPlotFeatures) == figureCat.m_requiredPlotFeatures))
        {
            figurePlugins.append(plugin);
        }
    }
    return figurePlugins;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param figureCategory
    \param defaultClassName
    \param retVal
    \return QString
*/
QString DesignerWidgetOrganizer::getFigureClass(
        const QString &figureCategory,
        const QString &defaultClassName,
        ito::RetVal &retVal)
{
    if (!m_figureCategories.contains(figureCategory))
    {
        retVal += ito::RetVal::format(ito::retError, 0, tr("The figure category '%s' is unknown").toLatin1().data(), figureCategory.data());
        return "";
    }

    FigureCategory figureCat = m_figureCategories[figureCategory];
    QList<FigurePlugin> figurePlugins;

    foreach(const FigurePlugin &plugin, m_figurePlugins)
    {
        if ((plugin.plotDataTypes & figureCat.m_allowedPlotDataTypes) &&
            (plugin.plotDataFormats & figureCat.m_allowedPlotDataFormats) &&
            (plugin.plotFeatures & figureCat.m_excludedPlotFeatures) == 0 &&
            ((plugin.plotFeatures & figureCat.m_requiredPlotFeatures) == figureCat.m_requiredPlotFeatures))
        {
            figurePlugins.append(plugin);
        }
    }

    if (defaultClassName != "")
    {
        foreach(const FigurePlugin &plugin, figurePlugins)
        {
            if (QString::compare(plugin.classname, defaultClassName, Qt::CaseInsensitive) == 0)
            {
                return defaultClassName; //the given class name fits to the figureCategory and exists
            }
        }

        //check for obsolete class names
        QString replaceClassName;
        if (QString::compare(defaultClassName, "matplotlibfigure", Qt::CaseInsensitive) == 0)
        {
            replaceClassName = "matplotlibplot";
        }
        else if (QString::compare(defaultClassName, "itom1dqwtfigure", Qt::CaseInsensitive) == 0)
        {
            replaceClassName = "itom1dqwtplot";
        }
        else if (QString::compare(defaultClassName, "itom2dqwtfigure", Qt::CaseInsensitive) == 0)
        {
            replaceClassName = "itom2dqwtplot";
        }
        else if (QString::compare(defaultClassName, "itom2DGVFigure", Qt::CaseInsensitive) == 0)
        {
            replaceClassName = "GraphicViewPlot";
        }

        if (replaceClassName != "")
        {
            foreach(const FigurePlugin &plugin, figurePlugins)
            {
                if (QString::compare(plugin.classname, replaceClassName, Qt::CaseInsensitive) == 0)
                {
                    return defaultClassName; //the given class name fits to the figureCategory and exists
                }
            }
        }

        retVal += ito::RetVal::format(ito::retWarning, 0, tr("The figure class '%1' could not be found or does not support displaying the given type of data. The default class for the given data is used instead.").arg(defaultClassName).toLatin1().data());
    }
    // #if no class name present, default is being loaded from settings file...
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DesignerPlotWidgets");
    QString settingsClassName = settings.value(figureCategory, figureCat.m_defaultClassName).toString();
    settings.endGroup();

    bool repeat = true;

    while(repeat && !retVal.containsError())
    {

        foreach(const FigurePlugin &plugin, figurePlugins)
        {
            if (QString::compare(plugin.classname, settingsClassName, Qt::CaseInsensitive) == 0)
            {
                return settingsClassName; //the given class name fits to the figureCategory and exists
            }
        }

        repeat = false;

        //There are some obsolete figures. If they cannot be found, try to find their equivalent successor
        if (QString::compare(settingsClassName, "itom2dqwtfigure", Qt::CaseInsensitive) == 0)
        {
            settingsClassName = "itom2dqwtplot";
            repeat = true;
        }
        else if (QString::compare(settingsClassName, "matplotlibfigure", Qt::CaseInsensitive) == 0)
        {
            settingsClassName = "matplotlibplot";
            repeat = true;
        }
        else if (QString::compare(settingsClassName, "itom1dqwtfigure", Qt::CaseInsensitive) == 0)
        {
            settingsClassName = "itom1dqwtplot";
            repeat = true;
        }
        else if (QString::compare(defaultClassName, "itom2DGVFigure", Qt::CaseInsensitive) == 0)
        {
            settingsClassName = "GraphicViewPlot";
            repeat = true;
        }
    }

    if (figurePlugins.count() > 0)
    {
        return figurePlugins[0].classname;
    }

    retVal += ito::RetVal(ito::retError, 0, tr("no plot figure plugin could be found that fits to the given category.").toLatin1().data());
    return "";
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param figureCategory
    \param defaultClassName
    \return RetVal
*/
RetVal DesignerWidgetOrganizer::setFigureDefaultClass(const QString &figureCategory, const QString &defaultClassName)
{
    if (!m_figureCategories.contains(figureCategory))
    {
        return ito::RetVal::format(ito::retError, 0, tr("The figure category '%s' is unknown").toLatin1().data(), figureCategory.data());
    }

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DesignerPlotWidgets");
    settings.setValue(figureCategory, defaultClassName);
    settings.endGroup();
    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! short
/*! long

    \param className
    \param parentWidget
    \param winMode
    \return QWidget
*/
QWidget* DesignerWidgetOrganizer::createWidget(const QString &className, QWidget *parentWidget, AbstractFigure::WindowMode winMode /*= AbstractFigure::ModeStandaloneInUi*/)
{
    QPluginLoader *factory = NULL;
    foreach(const FigurePlugin &plugin, m_figurePlugins)
    {
        if (QString::compare(plugin.classname, className, Qt::CaseInsensitive) == 0)
        {
            factory = plugin.factory;
            break;
        }
    }

    if (factory)
    {
        //qDebug() << "create instance\n";
        ito::AbstractItomDesignerPlugin *fac = (ito::AbstractItomDesignerPlugin*)(factory->instance());
        QWidget *w = fac->createWidgetWithMode(winMode, parentWidget);
        if (w)
        {
            //if w is a plot (hence, inherited from ito::AbstractFigure), send the API function pointers to it.
            setApiPointersToWidgetAndChildren(w);
        }

        return w;
    }

    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Return plot input types as string list
/*! Return plot input types from plot type identifier to string list.

    \param plotInputType
    \return QStringList
*/
QStringList DesignerWidgetOrganizer::getPlotInputTypes(const int plotInputTypes)
{

    QStringList sl;

    if (plotInputTypes & ito::DataObjLine)
    {
        sl.append(tr("DataObject - Line"));
    }
    if (plotInputTypes & ito::DataObjPlane)
    {
        sl.append(tr("DataObject - Plane"));
    }
    if (plotInputTypes & ito::DataObjPlaneStack)
    {
        sl.append(tr("DataObject - Plane Stack"));
    }
    if (plotInputTypes & ito::PointCloud)
    {
        sl.append(tr("Point Cloud"));
    }
    if (plotInputTypes & ito::PolygonMesh)
    {
        sl.append(tr("PolygonMesh"));
    }

    if (sl.length() == 0)
        sl.append(tr("invalid type or no type defined"));

    return sl;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Return plot data formats as string list
/*! Return plot data formats from plot data format identifier to string list.

    \param plotDataFormat
    \return QStringList
*/
QStringList DesignerWidgetOrganizer::getPlotDataFormats(const int plotDataFormats)
{
    QStringList sl;

    if (plotDataFormats & ito::Format_Gray8)
    {
        sl.append(tr("Gray8"));
    }
    if (plotDataFormats & ito::Format_Gray16)
    {
        sl.append(tr("Gray16"));
    }
    if (plotDataFormats & ito::Format_Gray32)
    {
        sl.append(tr("Gray32"));
    }
    if (plotDataFormats & ito::Format_RGB32)
    {
        sl.append(tr("RGB32"));
    }
    if (plotDataFormats & ito::Format_ARGB32)
    {
        sl.append(tr("ARGB32"));
    }
    if (plotDataFormats & ito::Format_CMYK32)
    {
        sl.append(tr("CMYK32"));
    }
    if (plotDataFormats & ito::Format_Float32)
    {
        sl.append(tr("Float32"));
    }
    if (plotDataFormats & ito::Format_Float64)
    {
        sl.append(tr("Float64"));
    }
    if (plotDataFormats & ito::Format_Complex)
    {
        sl.append(tr("Complex"));
    }

    if (sl.length() == 0)
        sl.append(tr("invalid type or no type defined"));

    return sl;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Return plot features as string list
/*! Return plot features from plot features identifier to string list.

    \param plotFeatures
    \return QStringList
*/
QStringList DesignerWidgetOrganizer::getPlotFeatures(const int plotFeatures)
{
    QStringList sl;

    if (plotFeatures & ito::Static)
    {
        sl.append(tr("Static"));
    }
    if (plotFeatures & ito::Live)
    {
        sl.append(tr("Live"));
    }
    if (plotFeatures & ito::Cartesian)
    {
        sl.append(tr("Cartesian"));
    }
    if (plotFeatures & ito::Polar)
    {
        sl.append(tr("Polar"));
    }
    if (plotFeatures & ito::Cylindrical)
    {
        sl.append(tr("Cylindrical"));
    }
    if (plotFeatures & ito::OpenGl)
    {
        sl.append(tr("OpenGl"));
    }
    if (plotFeatures & ito::Cuda)
    {
        sl.append(tr("Cuda"));
    }
    if (plotFeatures & ito::X3D)
    {
        sl.append(tr("X3D"));
    }

    if (sl.length() == 0)
        sl.append(tr("invalid type or no type defined"));

    return sl;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Return plot input types as string list
/*! Return plot input types from plot type identifier to string list.

    \param plotInputType
    \return QStringList
*/
QStringList DesignerWidgetOrganizer::getPlotType(const int plotType)
{
    QStringList sl;

    if (plotType & ito::PlotLine)
    {
        sl.append(tr("Line Plot"));
    }
    if (plotType & ito::PlotImage)
    {
        sl.append(tr("Image Plot"));
    }
    if (plotType & ito::PlotISO)
    {
        sl.append(tr("Isometric Plot"));
    }
    if (plotType & ito::Plot3D)
    {
        sl.append(tr("3D Plot"));
    }

    if (sl.length() == 0)
        sl.append(tr("invalid type or no type defined"));

    return sl;
}

//----------------------------------------------------------------------------------------------------------------------------------
QStringList DesignerWidgetOrganizer::getListOfIncompatibleDesignerPlugins() const
{
    QStringList output;

    foreach(const PluginLoadStatus &plugin, m_pluginLoadStatus)
    {
        for (int j = 0; j < plugin.messages.size(); ++j)
        {
            if (plugin.messages[j].first & tPluginLoadStatusFlag::plsfIncompatible)
            {
                output << plugin.messages[j].second;
                break;
            }
        }
    }

    return output;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DesignerWidgetOrganizer::setApiPointersToWidgetAndChildren(QWidget *widget)
{
    //this method is also implemented in uiOrganizer!

    if (widget)
    {
        if (widget->inherits("ito::AbstractFigure"))
        {
            ((ito::AbstractFigure*)widget)->setApiFunctionGraphBasePtr(ITOM_API_FUNCS_GRAPH);
            ((ito::AbstractFigure*)widget)->setApiFunctionBasePtr(ITOM_API_FUNCS);

            //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
            //transmitted to the plugin. This assignment cannot be done directly, since
            //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
            //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
            //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
            //also is necessary if any methods of the plugin are directly called from itom).
            QEvent evt((QEvent::Type)(QEvent::User + 123));
            QCoreApplication::sendEvent(widget, &evt);
        }
        else if (widget->inherits("ito::AbstractApiWidget"))
        {
            ((ito::AbstractApiWidget*)widget)->setApiFunctionGraphBasePtr(ITOM_API_FUNCS_GRAPH);
            ((ito::AbstractApiWidget*)widget)->setApiFunctionBasePtr(ITOM_API_FUNCS);

            //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
            //transmitted to the plugin. This assignment cannot be done directly, since
            //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
            //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
            //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
            //also is necessary if any methods of the plugin are directly called from itom).
            QEvent evt((QEvent::Type)(QEvent::User + 123));
            QCoreApplication::sendEvent(widget, &evt);
        }

        QObjectList list = widget->children();
        foreach(QObject *obj, list)
        {
            setApiPointersToWidgetAndChildren(qobject_cast<QWidget*>(obj));
        }
    }
}

} //end namespace ito
