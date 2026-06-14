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

#include "widgetPropEditorScripts.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../codeEditor/codeEditor.h"
#include <qmenu.h>

#include <qsettings.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorScripts::WidgetPropEditorScripts(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorScripts::~WidgetPropEditorScripts()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorScripts::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // Code Outline
    ui.checkOutlineAutoUpdateEnabled->setChecked(settings.value("outlineAutoUpdateEnabled", true).toBool());
    ui.spinOutlineAutoUpdateDelay->setValue(settings.value("outlineAutoUpdateDelay", 0.5).toDouble());
    ui.checkOutlineShowNavigation->setChecked(settings.value("outlineShowNavigation", true).toBool());

    // edge mode
    switch ((CodeEditor::EdgeMode)(settings.value("edgeMode", CodeEditor::EdgeNone).toInt()))
    {
    case CodeEditor::EdgeNone:
        ui.comboEdgeMode->setCurrentIndex(0);
        break;
    case CodeEditor::EdgeLine:
        ui.comboEdgeMode->setCurrentIndex(1);
        break;
    case CodeEditor::EdgeBackground:
        ui.comboEdgeMode->setCurrentIndex(2);
        break;
    }

    on_comboEdgeMode_currentIndexChanged(ui.comboEdgeMode->currentIndex());

    ui.spinEdgeColumn->setValue(settings.value("edgeColumn", 88).toInt());
    ui.colorEdgeBg->setColor(settings.value("edgeColor", QColor(Qt::black)).value<QColor>());

    int elideMode = settings.value("tabElideMode", Qt::ElideNone).toInt();

    switch (elideMode)
    {
    case Qt::ElideLeft:
        ui.radioElideLeft->setChecked(true);
        break;
    case Qt::ElideRight:
        ui.radioElideRight->setChecked(true);
        break;
    case Qt::ElideMiddle:
        ui.radioElideMiddle->setChecked(true);
        break;
    default:
        ui.radioElideNone->setChecked(true);
        break;
    }


    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorScripts::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // Class Navigator
    settings.setValue("outlineAutoUpdateEnabled", ui.checkOutlineAutoUpdateEnabled->isChecked());
    settings.setValue("outlineAutoUpdateDelay", ui.spinOutlineAutoUpdateDelay->value());
    settings.setValue("outlineShowNavigation", ui.checkOutlineShowNavigation->isChecked());

    //edge mode
    //edge mode
    int idx = ui.comboEdgeMode->currentIndex();
    settings.setValue("edgeMode", idx == 0 ? CodeEditor::EdgeNone : (idx == 1 ? CodeEditor::EdgeLine : CodeEditor::EdgeBackground));
    settings.setValue("edgeColumn", ui.spinEdgeColumn->value());
    settings.setValue("edgeColor", ui.colorEdgeBg->color());

    //elide mode (filename shortening in tabs of scriptDockWidget)
    if (ui.radioElideLeft->isChecked())
    {
        settings.setValue("tabElideMode", Qt::ElideLeft);
    }
    else if (ui.radioElideRight->isChecked())
    {
        settings.setValue("tabElideMode", Qt::ElideRight);
    }
    else if (ui.radioElideMiddle->isChecked())
    {
        settings.setValue("tabElideMode", Qt::ElideMiddle);
    }
    else
    {
        settings.setValue("tabElideMode", Qt::ElideNone);
    }

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorScripts::on_comboEdgeMode_currentIndexChanged(int index)
{
    switch (index)
    {
    case 0:
        ui.comboEdgeMode->setCurrentIndex(0);
        ui.spinEdgeColumn->setVisible(false);
        ui.lblEdgeColumn->setVisible(false);
        ui.colorEdgeBg->setVisible(false);
        ui.lblEdgeBg->setVisible(false);
        break;
    case 1:
        ui.comboEdgeMode->setCurrentIndex(1);
        ui.spinEdgeColumn->setVisible(true);
        ui.lblEdgeColumn->setVisible(true);
        ui.colorEdgeBg->setVisible(true);
        ui.lblEdgeBg->setVisible(true);
        ui.lblEdgeBg->setText(tr("Line color:"));
        break;
    case 2:
        ui.comboEdgeMode->setCurrentIndex(2);
        ui.spinEdgeColumn->setVisible(true);
        ui.lblEdgeColumn->setVisible(true);
        ui.colorEdgeBg->setVisible(true);
        ui.lblEdgeBg->setVisible(true);
        ui.lblEdgeBg->setText(tr("Background color:"));
        break;
    }
}

} //end namespace ito
