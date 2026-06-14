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

#ifndef DESIGNERWIDGETORGANIZER_H
#define DESIGNERWIDGETORGANIZER_H

#include "../../common/sharedStructures.h"
#include "../../common/sharedStructuresGraphics.h"
#include "../../AddInManager/pluginModel.h"
#include "plot/AbstractItomDesignerPlugin.h"

#include <qobject.h>
#include <qlist.h>
#include <qtranslator.h>

class QPluginLoader; //forward declaration

namespace ito
{

struct FigurePlugin
{
    FigurePlugin() : filename(""), classname(""), plotDataTypes(DataObjLine), plotFeatures(Static), factory(NULL) {}
    QString filename;
    QString classname;
    PlotDataTypes plotDataTypes;
    PlotDataFormats plotDataFormats;
    PlotFeatures plotFeatures;
    QIcon icon;
    QPluginLoader *factory;
};

struct FigureCategory
{
public:
    FigureCategory(const QString description, const PlotDataTypes allowedPlotDataTypes, const PlotDataFormats allowedPlotDataFormats, const PlotFeatures requiredPlotFeatures, const PlotFeatures excludedPlotFeatures, const QString defaultClassName)
        : m_description(description),
        m_allowedPlotDataTypes(allowedPlotDataTypes),
        m_allowedPlotDataFormats(allowedPlotDataFormats),
        m_requiredPlotFeatures(requiredPlotFeatures),
        m_excludedPlotFeatures(excludedPlotFeatures),
        m_defaultClassName(defaultClassName)
    {
    }

    FigureCategory() : m_description("") {}

    QString         m_description;
    PlotDataTypes   m_allowedPlotDataTypes;
    PlotDataFormats m_allowedPlotDataFormats;
    PlotFeatures    m_requiredPlotFeatures;
    PlotFeatures    m_excludedPlotFeatures;
    QString         m_defaultClassName;
};

class DesignerWidgetOrganizer : public QObject
{
    Q_OBJECT

public:

    DesignerWidgetOrganizer(ito::RetVal &retValue);
    ~DesignerWidgetOrganizer();

    const QList<PluginLoadStatus> getPluginLoadStatus() const { return m_pluginLoadStatus; }
    const QMap<QString, FigureCategory> getFigureCategories() const { return m_figureCategories; }

    QStringList getListOfIncompatibleDesignerPlugins() const;

    bool figureClassExists( const QString &className );
    ito::RetVal figureClassMinimumRequirementCheck( const QString &className, int plotDataTypesMask, int plotDataFormatsMask, int plotFeaturesMask, bool *ok = NULL );
    QList<FigurePlugin> getPossibleFigureClasses( int plotDataTypesMask, int plotDataFormatsMask, int plotFeaturesMask );
    QList<FigurePlugin> getPossibleFigureClasses( const FigureCategory &figureCat );
    QString getFigureClass( const QString &figureCategory, const QString &defaultClassName, ito::RetVal &retVal );
    RetVal setFigureDefaultClass( const QString &figureCategory, const QString &defaultClassName);
    QStringList getPlotInputTypes(const int plotInputType);
    QStringList getPlotType(const int plotType);
    QStringList getPlotFeatures(const int plotFeatures);
    QStringList getPlotDataFormats(const int plotDataFormats);

    QWidget* createWidget(const QString &className, QWidget *parentWidget, AbstractFigure::WindowMode winMode = AbstractFigure::ModeStandaloneInUi);

protected:
    RetVal scanDesignerPlugins();
    void setApiPointersToWidgetAndChildren(QWidget *widget);

private:
    QList<FigurePlugin> m_figurePlugins;
    QList<PluginLoadStatus> m_pluginLoadStatus;
    QMap<QString, FigureCategory> m_figureCategories;
    QVector<QTranslator*> m_Translator;

signals:

public slots:

private slots:

};

} //namespace ito

#endif
