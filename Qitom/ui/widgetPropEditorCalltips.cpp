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

#include "widgetPropEditorCalltips.h"

#include "../global.h"
#include "../AppManagement.h"

WidgetPropEditorCalltips::WidgetPropEditorCalltips(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

WidgetPropEditorCalltips::~WidgetPropEditorCalltips()
{
}

void WidgetPropEditorCalltips::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    ui.groupCallTips->setChecked( settings.value("calltipsEnabled",true).toBool());
    ui.spinNoOfCalltips->setValue( settings.value("calltipsNoVisible",3).toInt());

    QString style = settings.value("calltipsStyle","NoContext").toString();
    ui.radioCalltipsContext1->setChecked( style == "NoContext" );
    ui.radioCalltipsContext2->setChecked( style == "NoAutoCompletionContext" );
    ui.radioCalltipsContext3->setChecked( style == "Context" );

    settings.endGroup();
}

void WidgetPropEditorCalltips::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    settings.setValue("calltipsEnabled", ui.groupCallTips->isChecked());
    settings.setValue("calltipsNoVisible", ui.spinNoOfCalltips->value());
    
    if(ui.radioCalltipsContext1->isChecked())
    {
        settings.setValue("calltipsStyle","NoContext");
    }
    else if(ui.radioCalltipsContext2->isChecked())
    {
        settings.setValue("calltipsStyle","NoAutoCompletionContext");
    }
    else
    {
        settings.setValue("calltipsStyle","Context");
    }

    settings.endGroup();
}