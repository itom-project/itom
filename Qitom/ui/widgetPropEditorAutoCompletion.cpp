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

#include "widgetPropEditorAutoCompletion.h"

#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorAutoCompletion::WidgetPropEditorAutoCompletion(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorAutoCompletion::~WidgetPropEditorAutoCompletion()
{

}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAutoCompletion::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    ui.checkCaseSensitivity->setChecked( settings.value("autoComplCaseSensitive", false).toBool());
    ui.groupAutoCompletion->setChecked( settings.value("autoComplEnabled", true).toBool());
    ui.checkShowTooltips->setChecked( settings.value("autoComplShowTooltips", false).toBool());
    ui.spinThreshold->setValue( settings.value("autoComplThreshold", 2).toInt());

    int filterMode = qBound(0, settings.value("autoComplFilterMode", 2).toInt(), 2);
    ui.comboFilterMode->setCurrentIndex(filterMode);

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAutoCompletion::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    settings.setValue("autoComplCaseSensitive", ui.checkCaseSensitivity->isChecked() );
    settings.setValue("autoComplShowTooltips", ui.checkShowTooltips->isChecked() );
    settings.setValue("autoComplEnabled", ui.groupAutoCompletion->isChecked() );

    settings.setValue("autoComplThreshold", ui.spinThreshold->value());
    settings.setValue("autoComplFilterMode", ui.comboFilterMode->currentIndex());

    settings.endGroup();
}

} //end namespace ito
