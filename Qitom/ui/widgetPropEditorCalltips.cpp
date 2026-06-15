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

#include "widgetPropEditorCalltips.h"

#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorCalltips::WidgetPropEditorCalltips(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorCalltips::~WidgetPropEditorCalltips()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorCalltips::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    ui.checkCallTips->setChecked( settings.value("calltipsEnabled", true).toBool());
    ui.checkHelpTooltip->setChecked(settings.value("helpTooltipEnabled", true).toBool());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorCalltips::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    settings.setValue("calltipsEnabled", ui.checkCallTips->isChecked());
    settings.setValue("helpTooltipEnabled", ui.checkHelpTooltip->isChecked());
    settings.endGroup();
}

} //end namespace ito
