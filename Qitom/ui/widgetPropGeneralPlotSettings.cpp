/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include "widgetPropGeneralPlotSettings.h"

#include "../global.h"
#include "../AppManagement.h"
#include <qdebug>
#include <qsettings.h>

namespace ito
{
    WidgetPropGeneralPlotSettings::WidgetPropGeneralPlotSettings(QWidget *parent) :
        AbstractPropertyPageWidget(parent)
    {
        ui.setupUi(this);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void WidgetPropGeneralPlotSettings::readSettings()
    {
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup("DesignerPlugins");
        settings.beginGroup("ito::AbstractFigure");


        ui.titleFontBtn->setCurrentFont(settings.value("titleFont", ("Helvetica", 12)).value<QFont>());
        ui.labelFontBtn->setCurrentFont(settings.value("labelFont", ("Helvetica", 12)).value<QFont>());
        ui.axisFontBtn->setCurrentFont(settings.value("axisFont", ("Helvetica", 12)).value<QFont>());
        QPen pen;
        pen.setStyle((Qt::PenStyle)settings.value("lineStyle", (int)Qt::SolidLine).toInt());
        pen.setWidth(settings.value("lineWidth", 1.0).value<qreal>());
        ui.lineStyleBtn->setPen(pen);
        ui.legendFontBtn->setCurrentFont(settings.value("legendFont", ("Helvetica", 8)).value<QFont>());
        ui.zoomRubberBandPenBtn->setPen(settings.value("zoomRubberBandPen", QPen((Qt::red), 2, Qt::DashLine)).value<QPen>());
        ui.trackerPenBtn->setPen(settings.value("trackerPen", (Qt::red, 2)).value<QPen>());
        ui.trackerFontBtn->setCurrentFont(settings.value("trackerFont", ("Verdana", 10)).value<QFont>());
        ui.trackerBackgroundBtn->setBrush(settings.value("trackerBackground", QBrush((255, 255, 255, 255), Qt::SolidPattern)).value<QBrush>());
        ui.centerMerkerPenBtn->setPen(settings.value("centerMarkerPen", (Qt::red, 1)).value<QPen>());
        ui.centerMarkerSizeSpin->setValue(settings.value("centerMarkerSize", (25, 25)).value<QSize>().width());


        settings.endGroup();
        settings.endGroup();
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void WidgetPropGeneralPlotSettings::writeSettings()
    {
      QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
      settings.beginGroup("DesignerPlugins");
      settings.beginGroup("ito::AbstractFigure");
      settings.setValue("titleFont", ui.titleFontBtn->currentFont());
      settings.setValue("labelFont", ui.labelFontBtn->currentFont());
      settings.setValue("axisFont", ui.axisFontBtn->currentFont());
      settings.setValue("lineStyle", (int) ui.lineStyleBtn->getPen().style());
      settings.setValue("lineWidth", (qreal)ui.lineStyleBtn->getPen().width());
      settings.setValue("legendFont", ui.legendFontBtn->currentFont());
      settings.setValue("zoomRubberBandPen", ui.zoomRubberBandPenBtn->getPen());
      settings.setValue("trackerPen", ui.trackerPenBtn->getPen());
      settings.setValue("trackerFont", ui.trackerFontBtn->currentFont());
      settings.setValue("trackerBackground", ui.trackerBackgroundBtn->getBrush());
      settings.setValue("centerMarkerPen", ui.centerMerkerPenBtn->getPen());
      settings.setValue("centerMarkerSize", QSize(ui.centerMarkerSizeSpin->value(), ui.centerMarkerSizeSpin->value()));


      settings.endGroup();
      settings.endGroup();
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    void WidgetPropGeneralPlotSettings::on_defaultBtn_clicked()
    {
        ui.titleFontBtn->setCurrentFont(QFont("Helvetica", 12));
        ui.labelFontBtn->setCurrentFont(QFont("Helvetica", 12));
        ui.axisFontBtn->setCurrentFont(QFont("Helvetica", 12));
        QPen pen(Qt::SolidLine);
        pen.setWidth(1.0);
        ui.lineStyleBtn->setPen(pen);
        ui.legendFontBtn->setCurrentFont(QFont("Helvetica", 8));
        ui.zoomRubberBandPenBtn->setPen(QPen(Qt::red, 2, Qt::DashLine));
        ui.trackerPenBtn->setPen(QPen(Qt::red, 2));
        ui.trackerFontBtn->setCurrentFont(QFont("Verdana", 10));
        ui.trackerBackgroundBtn->setBrush( QBrush((255, 255, 255, 255), Qt::SolidPattern));
        ui.centerMerkerPenBtn->setPen(QPen(Qt::red, 1));
        ui.centerMarkerSizeSpin->setValue(25);
        this->update();
    }

}//endNamespace ito