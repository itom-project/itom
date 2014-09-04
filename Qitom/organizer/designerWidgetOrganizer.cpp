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

#include "designerWidgetOrganizer.h"

#include "../global.h"
#include "../AppManagement.h"
#include "common/apiFunctionsGraphInc.h"
#include "plot/AbstractFigure.h"
#include "plot/AbstractDObjFigure.h"

#include <qmetaobject.h>
#include <qpluginloader.h>
#include <QtDesigner/QDesignerCustomWidgetInterface>
#include <qsettings.h>
#include <qcoreapplication.h>
#include <qdir.h>
#include <QDirIterator>
#include <qapplication.h>

#include <qpen.h>

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
    settings.beginGroup("ito::AbstractDObjFigure");
    if (settings.contains("zoomRubberBandPen") == false) settings.setValue("zoomRubberBandPen", QPen(QBrush(Qt::red), 2, Qt::DashLine));
    if (settings.contains("trackerPen") == false) settings.setValue("trackerPen", QPen(QBrush(Qt::red), 2));
    if (settings.contains("trackerFont") == false) settings.setValue("trackerFont", QFont("Verdana", 10));
    if (settings.contains("trackerBackground") == false) settings.setValue("trackerBackground", QBrush(QColor(255, 255, 255, 155), Qt::SolidPattern));
    if (settings.contains("centerMarkerSize") == false) settings.setValue("centerMarkerSize", QSize(10, 10));
    if (settings.contains("centerMarkerPen") == false) settings.setValue("centerMarkerPen", QPen(QBrush(Qt::red), 1));
    settings.endGroup();
    settings.endGroup();

    //create figure categories (for property dialog...)
   ito::PlotDataFormats allFormats = ~(ito::PlotDataFormats(0)); //(~ito::Format_Gray8); // | ito::Format_Gray8; //(ito::PlotDataFormats(0));

    m_figureCategories["DObjLiveLine"] = FigureCategory("Data Object, Line Plot, Live", ito::DataObjLine, allFormats, ito::Live | ito::PlotLine, 0, "Itom1DQwtPlot");
    m_figureCategories["DObjLiveImage"] = FigureCategory("Data Object, 2D Image Plot, Live", ito::DataObjPlane, allFormats, ito::Live | ito::PlotImage, 0, "GraphicViewPlot");
    m_figureCategories["DObjStaticLine"] = FigureCategory("Data Object, Line Plot, Static", ito::DataObjLine, allFormats, ito::Static | ito::PlotLine, 0, "Itom1DQwtPlot");
    m_figureCategories["DObjStaticImage"] = FigureCategory("Data Object, 2D Image Plot, Static", ito::DataObjPlane | ito::DataObjPlaneStack, allFormats, ito::Static | ito::PlotImage, 0, "Itom2DQwtPlot");
    m_figureCategories["DObjStaticGeneralPlot"] = FigureCategory("Data Object, Any Planar Plot, Static", ito::DataObjLine | ito::DataObjPlane | ito::DataObjPlaneStack, allFormats, ito::Static, ito::Plot3D | ito::PlotISO, "Itom2DQwtPlot");
    m_figureCategories["PerspectivePlot"] = FigureCategory("Data Object, Any Planar Plot, Point Clouds, Polygon Meshes, Static", ito::DataObjPlane | ito::DataObjPlaneStack | ito::PointCloud | ito::PolygonMesh, allFormats, ito::Static | ito::Plot3D | ito::PlotISO, ito::Live | ito::PlotLine, "ItomIsoOGLPlot");

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
    QString requiredInterface = "0.0.0";
    const QMetaObject *metaObj = NULL;
    bool allowedInterface;

    //This regular expression is used to check whether the error message during loading a plugin contains the words
    //'debug' or 'release'. This means, that a release plugin is tried to be loaded with a debug version of itom or vice-versa
    QRegExp regExpDebugRelease(".*(release|debug).*", Qt::CaseInsensitive); 

    //get version of the required AbstractItomDesignerPlugin
    AbstractItomDesignerPlugin *dummyPlugin = new DummyItomDesignerPlugin(NULL);
    if (dummyPlugin)
    {
        metaObj = dummyPlugin->metaObject();
        for (int i = 0; i < metaObj->classInfoCount() ; i++)
        {
            if (qstrcmp(metaObj->classInfo(i).name(), "ito.AbstractItomDesignerPlugin") == 0)
            {
                requiredInterface = metaObj->classInfo(i).value();
                break;
            }
        }
    }

    DELETE_AND_SET_NULL(dummyPlugin);

    if (requiredInterface == "0.0.0")
    {
        return RetVal(retError, 0, tr("could not read interface 'ito.AbstractItomDesignerPlugin'").toLatin1().data());
    }

    foreach(const QString &plugin, candidates)
    {
//        if (plugin.indexOf("itomWidgets", 0, Qt::CaseInsensitive) > 0)
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
            QByteArray codec =  settings.value("codec", "UTF-8" ).toByteArray();
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
                        message = QObject::tr("Unable to load translation file '%1'.").arg(translationPath + '/' + translationLocal);
                        qDebug() << message;
                        status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(plsfError, message));
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

            loader = new QPluginLoader(absolutePluginPath);


            QDesignerCustomWidgetInterface *iface = NULL;
            QObject *instance = loader->instance();

            if (instance == NULL)
            {
                message = loader->errorString();
                loader->unload();

                if (regExpDebugRelease.exactMatch(message)) //debug/release conflict is only a warning, no error
                {
                    ito::PluginLoadStatusFlags flags(plsfWarning | plsfRelDbg);
                    status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(flags, message));
                }
                else
                {
                    status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
                }
                
                DELETE_AND_SET_NULL(loader);
            }
            // try with a normal plugin, we do not support collections
            else if (iface = qobject_cast<QDesignerCustomWidgetInterface *>(instance))
            {
                if (instance->inherits("ito::AbstractItomDesignerPlugin"))
                {
                    allowedInterface = false;

                    //check interface
                    metaObj = ((ito::AbstractItomDesignerPlugin*)instance)->metaObject();
                    for (int i = 0; i < metaObj->classInfoCount() ; i++)
                    {
                        if (qstrcmp(metaObj->classInfo(i).name(), "ito.AbstractItomDesignerPlugin") == 0)
                        {
                            if (requiredInterface == metaObj->classInfo(i).value())
                            {
                                allowedInterface = true;
                            }
                            break;
                        }
                    }

                    if (allowedInterface)
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
                    }
                    else
                    {
//                        delete instance;
                        loader->unload();
                        message = tr("The version 'ito.AbstractItomDesignerPlugin' in file '%1' does not correspond to the requested version (%2)").arg(status.filename).arg(requiredInterface);
                        status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
                        DELETE_AND_SET_NULL(loader);
                    }
                }
                else
                {

#if QT_VERSION >= 0x040800 
                    /* it seems that it is not allowed to unload a designer plugin (but no plot plugin) here, 
                       since it is then also unloaded in the member m_uiLoader from uiOrganizer. TODO 

                       \todo this bug seems only to be there with Qt 4.7.x
                    */
                    loader->unload();
#endif
                    message = tr("Plugin in file '%1' is a Qt Designer widget but no itom plot widget that inherits 'ito.AbtractItomDesignerPlugin'").arg(status.filename);
                    status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfIgnored, message));
                    DELETE_AND_SET_NULL(loader);
                }
            }
            else
            {
                loader->unload();
                message = tr("Plugin in file '%1' is no Qt DesignerWidget inherited from QDesignerCustomWidgetInterface").arg(status.filename);
                status.messages.append(QPair<ito::PluginLoadStatusFlags, QString>(ito::plsfError, message));
                DELETE_AND_SET_NULL(loader);
            }

            m_pluginLoadStatus.append(status);
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
    
    if (ok) *ok = success;
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
QString DesignerWidgetOrganizer::getFigureClass(const QString &figureCategory, const QString &defaultClassName, ito::RetVal &retVal)
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
    }

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("DesignerPlotWidgets");
    QString settingsClassName = settings.value(figureCategory, figureCat.m_defaultClassName).toString();
    settings.endGroup();

    bool repeat = true;

    while(repeat)
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
    \param name
    \param winMode
    \return QWidget
*/
QWidget* DesignerWidgetOrganizer::createWidget(const QString &className, QWidget *parentWidget, const QString &name /*= QString()*/, AbstractFigure::WindowMode winMode /*= AbstractFigure::ModeStandaloneInUi*/)
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
        return fac->createWidgetWithMode(winMode, parentWidget);
    }
    return NULL;
}


//----------------------------------------------------------------------------------------------------------------------------------

} //end namespace ito
