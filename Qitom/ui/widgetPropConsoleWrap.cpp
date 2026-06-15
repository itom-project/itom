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

#include "widgetPropConsoleWrap.h"

#include "../global.h"
#include "../AppManagement.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropConsoleWrap::WidgetPropConsoleWrap(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    ui.groupVisualFlags->setVisible(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropConsoleWrap::~WidgetPropConsoleWrap()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropConsoleWrap::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    bool ok = false;

    int wrapMode = settings.value("WrapMode", 0).toInt(&ok);
    if (!ok)
    {
        wrapMode = 0;
    }
    ui.radioWrapMode1->setChecked(wrapMode == 0);
    ui.radioWrapMode2->setChecked(wrapMode == 1);
    ui.radioWrapMode3->setChecked(wrapMode == 2);

    int indent = settings.value("WrapIndent", 2).toInt(&ok);
    if (!ok)
    {
        indent = 2;
    }
    ui.spinFlagIndent->setValue(indent);

    ui.groupTextLineSplit->setChecked(settings.value("SplitLongLines", true).toBool());
    ui.spinMaxLineLength->setValue(settings.value("SplitLongLinesMaxLength", 200).toInt());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropConsoleWrap::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    int wrapMode;
    if (ui.radioWrapMode1->isChecked())
    {
        wrapMode = 0;
    }
    if (ui.radioWrapMode2->isChecked())
    {
        wrapMode = 1;
    }
    if (ui.radioWrapMode3->isChecked())
    {
        wrapMode = 2;
    }
    settings.setValue("WrapMode", wrapMode);

    settings.setValue("WrapIndent", ui.spinFlagIndent->value());

    settings.setValue("SplitLongLines", ui.groupTextLineSplit->isChecked());
    settings.setValue("SplitLongLinesMaxLength", ui.spinMaxLineLength->value());

    settings.endGroup();
}

} //end namespace ito
