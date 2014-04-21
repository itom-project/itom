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
    settings.beginGroup("PyScintilla");

    ui.checkCaseSensitivity->setChecked( settings.value("autoComplCaseSensitive", false).toBool());
    ui.checkFillUps->setChecked( settings.value("autoComplFillUps", true).toBool());
    ui.checkReplaceWord->setChecked( settings.value("autoComplReplaceWord", false).toBool());
    ui.checkShowSingle->setChecked( settings.value("autoComplShowSingle", false).toBool());
    ui.groupAutoCompletion->setChecked( settings.value("autoComplEnabled", true).toBool());

    ui.spinThreshold->setValue( settings.value("autoComplThreshold", 3).toInt());

    QString source = settings.value("autoComplSource", "AcsAPIs").toString();

    ui.radioACSource1->setChecked( source == "AcsAll" );
    ui.radioACSource2->setChecked( source == "AcsDocument" );
    ui.radioACSource3->setChecked( source == "AcsAPIs" );

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAutoCompletion::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    settings.setValue("autoComplCaseSensitive", ui.checkCaseSensitivity->isChecked() );
    settings.setValue("autoComplFillUps", ui.checkFillUps->isChecked() );
    settings.setValue("autoComplReplaceWord", ui.checkReplaceWord->isChecked() );
    settings.setValue("autoComplShowSingle", ui.checkShowSingle->isChecked() );
    settings.setValue("autoComplEnabled", ui.groupAutoCompletion->isChecked() );

    settings.setValue("autoComplThreshold", ui.spinThreshold->value());

    if (ui.radioACSource1->isChecked())
    {
        settings.setValue("autoComplSource", "AcsAll" );
    }
    else if (ui.radioACSource2->isChecked())
    {
        settings.setValue("autoComplSource", "AcsDocument" );
    }
    else
    {
        settings.setValue("autoComplSource", "AcsAPIs" );
    }

    settings.endGroup();
}

} //end namespace ito