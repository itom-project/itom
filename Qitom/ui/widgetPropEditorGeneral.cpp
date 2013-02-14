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

#include "widgetPropEditorGeneral.h"

#include "../global.h"
#include "../AppManagement.h"

WidgetPropEditorGeneral::WidgetPropEditorGeneral(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

WidgetPropEditorGeneral::~WidgetPropEditorGeneral()
{
}

void WidgetPropEditorGeneral::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    QString eolMode = settings.value("eolMode", "EolUnix").toString();
    ui.radioEOL1->setChecked( eolMode == "EolWindows" );
    ui.radioEOL2->setChecked( eolMode == "EolUnix" );
    ui.radioEOL3->setChecked( eolMode == "EolMac" );

    ui.checkAutoIndent->setChecked( settings.value("autoIndent", true).toBool());
    ui.checkIndentUseTabs->setChecked(settings.value("indentationUseTabs", false).toBool());
    ui.spinIndentWidth->setValue( settings.value("indentationWidth", 4).toInt());

    QString indentationWarning = settings.value("indentationWarning", "Inconsistent").toString();
    ui.radioIndentWarn1->setChecked( indentationWarning == "NoWarning" );
    ui.radioIndentWarn2->setChecked( indentationWarning == "Inconsistent" );
    ui.radioIndentWarn3->setChecked( indentationWarning == "TabsAfterSpaces" );
    ui.radioIndentWarn4->setChecked( indentationWarning == "Spaces" );
    ui.radioIndentWarn5->setChecked( indentationWarning == "Tabs" );

    ui.checkIndentShowGuides->setChecked( settings.value("showIndentationGuides", true).toBool());
    ui.checkShowWhitespace->setChecked( settings.value("showWhitespace", true).toBool());

    settings.endGroup();
}

void WidgetPropEditorGeneral::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    if(ui.radioEOL1->isChecked())
    {
        settings.setValue("eolMode", "EolWindows");
    }
    else if(ui.radioEOL2->isChecked())
    {
        settings.setValue("eolMode", "EolUnix");
    }
    else
    {
        settings.setValue("eolMode", "EolMac");
    }

    settings.setValue("autoIndent", ui.checkAutoIndent->isChecked() );
    settings.setValue("indentationUseTabs", ui.checkIndentUseTabs->isChecked() );
    settings.setValue("indentationWidth", ui.spinIndentWidth->value() );
    settings.setValue("showIndentationGuides", ui.checkIndentShowGuides->isChecked() );
    settings.setValue("showWhitespace", ui.checkShowWhitespace->isChecked() );

    if(ui.radioIndentWarn1->isChecked())
    {
        settings.setValue("indentationWarning", "NoWarning");
    }
    else if(ui.radioIndentWarn2->isChecked())
    {
        settings.setValue("indentationWarning", "Inconsistent");
    }
    else if(ui.radioIndentWarn3->isChecked())
    {
        settings.setValue("indentationWarning", "TabsAfterSpaces");
    }
    else if(ui.radioIndentWarn4->isChecked())
    {
        settings.setValue("indentationWarning", "Spaces");
    }
    else
    {
        settings.setValue("indentationWarning", "Tabs");
    }

    settings.endGroup();
}



