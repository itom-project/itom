/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

    // Syntax Checker
    ui.groupSyntaxCheck->setChecked(settings.value("syntaxChecker", true).toBool());
    ui.checkIncludeItom->setChecked(settings.value("syntaxIncludeItom", true).toBool());
    ui.spinSyntaxInterval->setValue(settings.value("syntaxInterval", 1.00).toDouble());

    // Class Navigator
    ui.groupClassNavigator->setChecked(settings.value("classNavigator", true).toBool());
    ui.checkActiveClassNavigatorTimer->setChecked(settings.value("classNavigatorTimerActive", true).toBool());
    ui.spinClassNavigatorInterval->setValue(settings.value("classNavigatorInterval", 2.00).toDouble());

    //edge mode
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

    ui.spinEdgeColumn->setValue(settings.value("edgeColumn", 0).toInt());
    ui.colorEdgeBg->setColor(settings.value("edgeColor", QColor(Qt::black)).value<QColor>());
    

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorScripts::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // Syntax Checker
    settings.setValue("syntaxInterval", ui.spinSyntaxInterval->value());
    settings.setValue("syntaxChecker", ui.groupSyntaxCheck->isChecked());
    settings.setValue("syntaxIncludeItom", ui.checkIncludeItom->isChecked());

    // Class Navigator
    settings.setValue("classNavigator", ui.groupClassNavigator->isChecked());
    settings.setValue("classNavigatorTimerActive", ui.checkActiveClassNavigatorTimer->isChecked());
    settings.setValue("classNavigatorInterval", ui.spinClassNavigatorInterval->value());

    //edge mode
    //edge mode
    int idx = ui.comboEdgeMode->currentIndex();
    settings.setValue("edgeMode", idx == 0 ? CodeEditor::EdgeNone : (idx == 1 ? CodeEditor::EdgeLine : CodeEditor::EdgeBackground));
    settings.setValue("edgeColumn", ui.spinEdgeColumn->value());
    settings.setValue("edgeColor", ui.colorEdgeBg->color());

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

