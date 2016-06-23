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

#include "widgetPropEditorScripts.h"

#include "../global.h"
#include "../AppManagement.h"
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
    settings.beginGroup("PyScintilla");

    // Syntax Checker
    ui.groupSyntaxCheck->setChecked(settings.value("syntaxChecker", true).toBool());
    ui.checkIncludeItom->setChecked(settings.value("syntaxIncludeItom", true).toBool());
    ui.spinSyntaxInterval->setValue(settings.value("syntaxInterval", 1.00).toDouble());

    // Class Navigator
    ui.groupClassNavigator->setChecked(settings.value("classNavigator", true).toBool());
    ui.checkActiveClassNavigatorTimer->setChecked(settings.value("classNavigatorTimerActive", true).toBool());
    ui.spinClassNavigatorInterval->setValue(settings.value("classNavigatorInterval", 2.00).toDouble());

    

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorScripts::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    // Syntax Checker
    settings.setValue("syntaxInterval", ui.spinSyntaxInterval->value());
    settings.setValue("syntaxChecker", ui.groupSyntaxCheck->isChecked());
    settings.setValue("syntaxIncludeItom", ui.checkIncludeItom->isChecked());

    // Class Navigator
    settings.setValue("classNavigator", ui.groupClassNavigator->isChecked());
    settings.setValue("classNavigatorTimerActive", ui.checkActiveClassNavigatorTimer->isChecked());
    settings.setValue("classNavigatorInterval", ui.spinClassNavigatorInterval->value());

    settings.endGroup();
}

} //end namespace ito

