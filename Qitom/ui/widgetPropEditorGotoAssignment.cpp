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

#include "widgetPropEditorGotoAssignment.h"

#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorGotoAssignment::WidgetPropEditorGotoAssignment(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorGotoAssignment::~WidgetPropEditorGotoAssignment()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorGotoAssignment::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    ui.groupGotoDefinition->setChecked( settings.value("gotoAssignmentEnabled", true).toBool());
    ui.groupMouseClick->setChecked( settings.value("gotoAssignmentMouseClickEnabled", true).toBool());
    ui.comboWordClickMode->setCurrentIndex(settings.value("gotoAssignmentMouseClickMode", 1).toInt());
	ui.comboWordClickKey->setCurrentIndex(settings.value("gotoAssignmentMouseClickKey", 1).toInt());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorGotoAssignment::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    settings.setValue("gotoAssignmentEnabled", ui.groupGotoDefinition->isChecked());
    settings.setValue("gotoAssignmentMouseClickEnabled", ui.groupMouseClick->isChecked());
    settings.setValue("gotoAssignmentMouseClickMode", ui.comboWordClickMode->currentIndex());
	settings.setValue("gotoAssignmentMouseClickKey", ui.comboWordClickKey->currentIndex());
    settings.endGroup();
}

} //end namespace ito
