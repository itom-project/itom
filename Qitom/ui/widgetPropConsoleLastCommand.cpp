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

#include "widgetPropConsoleLastCommand.h"

#include "../global.h"
#include "../AppManagement.h"
#include <qcolor.h>
#include "../widgets/lastCommandDockWidget.h"

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropConsoleLastCommand::WidgetPropConsoleLastCommand(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropConsoleLastCommand::~WidgetPropConsoleLastCommand()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropConsoleLastCommand::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup(objectName());
    ui.groupLastCommand->setChecked(settings.value("lastCommandEnabled", "true").toBool());
    QString dateColor = settings.value("lastCommandDateColor", "green").toString();
    ui.checkShowTimeStamp->setChecked(settings.value("lastCommandTimeStamp", "false").toBool());
    ui.checkHideDoubleCommand->setChecked(settings.value("lastCommandHideDoubleCommand", "false").toBool());
    ui.spinCommandNumbers->setValue(settings.value("lastCommandCommandNumbers", "100").toInt());
    settings.endGroup();

    int x = 0;
    QPixmap icon(12, 12);
    foreach(QString color, QColor::colorNames())
    {
        icon.fill(QColor(color));
        ui.comboBoxColor->addItem(icon, color);
        if (color == dateColor)
        {
            ui.comboBoxColor->setCurrentIndex(x);
        }
        ++x;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropConsoleLastCommand::writeSettings()
{
    QString dateColor = ui.comboBoxColor->itemText(ui.comboBoxColor->currentIndex());

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup(objectName());

    settings.setValue("lastCommandEnabled", ui.groupLastCommand->isChecked());
    settings.setValue("lastCommandDateColor", dateColor);
    settings.setValue("lastCommandTimeStamp", ui.checkShowTimeStamp->isChecked());
    settings.setValue("lastCommandHideDoubleCommand", ui.checkHideDoubleCommand->isChecked());
    settings.setValue("lastCommandCommandNumbers", ui.spinCommandNumbers->value());

    if (!ui.groupLastCommand->isChecked())
    {
        settings.remove("lastUsedCommands");
    }

    settings.endGroup();
}