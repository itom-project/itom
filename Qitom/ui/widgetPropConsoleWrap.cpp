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
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropConsoleWrap::~WidgetPropConsoleWrap()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropConsoleWrap::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    bool ok = false;
    int index;

    int wrapMode = settings.value("WrapMode", 0).toInt(&ok);
    if (!ok)
    {
        wrapMode = 0;
    }
    ui.radioWrapMode1->setChecked(wrapMode == 0);
    ui.radioWrapMode2->setChecked(wrapMode == 1);
    ui.radioWrapMode3->setChecked(wrapMode == 2);

    QString flagStart = settings.value("WrapFlagStart", "NoFlag").toString();
    if (flagStart == "NoFlag")
    {
        index = 0;
    }
    if (flagStart == "FlagText")
    {
        index = 1;
    }
    if (flagStart == "FlagBorder")
    {
        index = 2;
    }
    ui.comboFlagStart->setCurrentIndex(index);

    QString flagEnd = settings.value("WrapFlagEnd", "NoFlag").toString();
    if (flagEnd == "NoFlag")
    {
        index = 0;
    }
    if (flagEnd == "FlagText")
    {
        index = 1;
    }
    if (flagEnd == "FlagBorder")
    {
        index = 2;
    }
    ui.comboFlagEnd->setCurrentIndex(index);

    int indent = settings.value("WrapIndent", 2).toInt(&ok);
    if (!ok)
    {
        indent = 2;
    }
    ui.spinFlagIndent->setValue(indent);

    int indentMode = settings.value("WrapIndentMode", 0).toInt(&ok);
    if (!ok)
    {
        indentMode = 0;
    }
    ui.radioIndentMode1->setChecked(indentMode == 0);
    ui.radioIndentMode2->setChecked(indentMode == 1);
    ui.radioIndentMode3->setChecked(indentMode == 2);

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropConsoleWrap::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

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

    QString flagStart, flagEnd;
    switch(ui.comboFlagStart->currentIndex())
    {
    case 0: flagStart = "NoFlag"; break;
    case 1: flagStart = "FlagText"; break;
    case 2: flagStart = "FlagBorder"; break;
    }
    switch(ui.comboFlagEnd->currentIndex())
    {
    case 0: flagEnd = "NoFlag"; break;
    case 1: flagEnd = "FlagText"; break;
    case 2: flagEnd = "FlagBorder"; break;
    }

    settings.setValue("WrapFlagStart", flagStart);
    settings.setValue("WrapFlagEnd", flagEnd);

    settings.setValue("WrapIndent", ui.spinFlagIndent->value());

    int wrapIndentMode;
    if (ui.radioIndentMode1->isChecked())
    {
        wrapIndentMode = 0;
    }
    if (ui.radioIndentMode2->isChecked())
    {
        wrapIndentMode = 1;
    }
    if (ui.radioIndentMode3->isChecked())
    {
        wrapIndentMode = 2;
    }
    settings.setValue("WrapIndentMode", wrapIndentMode);

    settings.endGroup();
}

} //end namespace ito


