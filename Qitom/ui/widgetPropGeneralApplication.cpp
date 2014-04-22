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

#include "widgetPropGeneralApplication.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{

WidgetPropGeneralApplication::WidgetPropGeneralApplication(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

WidgetPropGeneralApplication::~WidgetPropGeneralApplication()
{
}

void WidgetPropGeneralApplication::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("MainWindow");

    ui.checkAskBeforeExit->setChecked( settings.value("askBeforeClose", false).toBool() );

    settings.endGroup();
}

void WidgetPropGeneralApplication::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("MainWindow");

    settings.setValue("askBeforeClose", ui.checkAskBeforeExit->isChecked() );

    settings.endGroup();
}

} //end namespace ito