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

#include "widgetPropHelpDock.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <global.h>


namespace ito
{
//----------------------------------------------------------------------------------------------------------------------------------
// Constructor
WidgetPropHelpDock::WidgetPropHelpDock(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Destructor
WidgetPropHelpDock::~WidgetPropHelpDock()
{

}

//----------------------------------------------------------------------------------------------------------------------------------
// get settings from ini
void WidgetPropHelpDock::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference"); //keep this name fix here!

    ui.checkPlaintext->setChecked( settings.value("plaintext", false).toBool() );

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
// set settings from ini
void WidgetPropHelpDock::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");

    settings.setValue("plaintext", ui.checkPlaintext->isChecked());

    settings.endGroup();
}

} //end namespace ito
