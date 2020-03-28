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

#include "widgetPropEditorSyntaxCheck.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../codeEditor/codeEditor.h"
#include <qmenu.h>

#include <qsettings.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorSyntaxCheck::WidgetPropEditorSyntaxCheck(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorSyntaxCheck::~WidgetPropEditorSyntaxCheck()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorSyntaxCheck::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // General Settings
    ui.checkIncludeItom->setChecked(settings.value("syntaxIncludeItom", true).toBool());
    ui.spinSyntaxInterval->setValue(settings.value("syntaxInterval", 1.00).toDouble());
	ui.comboShowMinLevel->setCurrentIndex(
			qBound(0, 
				settings.value("syntaxCheckerShowMinCategoryLevel", 0).toInt(),
				ui.comboShowMinLevel->count()));

    // Basic Options
	switch (settings.value("syntaxCheckerMode", 1).toInt())
	{
	case 0:
		ui.radioDisableChecks->setChecked(true);
		break;
	default:
	case 1:
		ui.radioEnableSyntaxCheck->setChecked(true);
		break;
	case 2:
		ui.radioEnableAllChecks->setChecked(true);
		break;
	}
	
	//Basic Options (PyFlakes)
	ui.comboSyntaxCategories->setCurrentText(settings.value("syntaxCheckerBasicCategory", "Error").toString());

	//All Options (Flake8)
	ui.spinAllChecksLineLength->setValue(settings.value("syntaxCheckerAllMaxLineLength", 79).toInt());

	ui.checkAllChecksSelect->setChecked( settings.value("syntaxCheckerAllSelectEnabled", false).toBool());
	ui.lblAllChecksSelect->setText( settings.value("syntaxCheckerAllSelectValues", "").toString());

	ui.checkAllChecksIgnore->setChecked( settings.value("syntaxCheckerAllIgnoreEnabled", false).toBool());
	ui.lblAllChecksIgnore->setText( settings.value("syntaxCheckerAllIgnoreValues", "").toString());

    ui.comboAllChecksDocstyle->setCurrentText(settings.value("syntaxCheckerAllDocstyle", "pep257").toString());

	ui.checkAllChecksMaxComplexity->setChecked(settings.value("syntaxCheckerAllMaxComplexityEnabled", true).toBool());
	ui.spinAllChecksMaxComplexity->setValue(settings.value("syntaxCheckerAllMaxComplexity", 10).toInt());

	ui.lblAllChecksErrorNumbers->setText(settings.value("syntaxCheckerAllErrorNumbers", "F").toString());
	ui.lblAllCheckWarningNumbers->setText(settings.value("syntaxCheckerAllWarningNumbers", "E, C").toString());

	ui.txtAllChecksMoreOptions->setPlainText(settings.value("syntaxCheckerAllOtherOptions", "").toString());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorSyntaxCheck::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

	// General Settings
    settings.setValue("syntaxInterval", ui.spinSyntaxInterval->value());
    settings.setValue("syntaxIncludeItom", ui.checkIncludeItom->isChecked());
	settings.setValue("syntaxCheckerShowMinCategoryLevel", qMax(0, ui.comboShowMinLevel->currentIndex()));

	// Basic Options
	if (ui.radioDisableChecks->isChecked())
	{
		settings.setValue("syntaxCheckerMode", 0);
	}
	else if (ui.radioEnableSyntaxCheck->isChecked())
	{
		settings.setValue("syntaxCheckerMode", 1);
	}
	else
	{
		settings.setValue("syntaxCheckerMode", 2);
	}

	//Basic Options (PyFlakes)
	settings.setValue("syntaxCheckerBasicCategory", ui.comboSyntaxCategories->currentText());

	//All Options (Flake8)
	settings.setValue("syntaxCheckerAllMaxLineLength", ui.spinAllChecksLineLength->value());

	settings.setValue("syntaxCheckerAllSelectEnabled", ui.checkAllChecksSelect->isChecked());
	settings.setValue("syntaxCheckerAllSelectValues", ui.lblAllChecksSelect->text());

	settings.setValue("syntaxCheckerAllIgnoreEnabled", ui.checkAllChecksIgnore->isChecked());
	settings.setValue("syntaxCheckerAllIgnoreValues", ui.lblAllChecksIgnore->text());

	settings.setValue("syntaxCheckerAllDocstyle", ui.comboAllChecksDocstyle->currentText());

	settings.setValue("syntaxCheckerAllMaxComplexityEnabled", ui.checkAllChecksMaxComplexity->isChecked());
	settings.setValue("syntaxCheckerAllMaxComplexity", ui.spinAllChecksMaxComplexity->value());

	settings.setValue("syntaxCheckerAllErrorNumbers", ui.lblAllChecksErrorNumbers->text());
	settings.setValue("syntaxCheckerAllWarningNumbers", ui.lblAllCheckWarningNumbers->text());

	settings.setValue("syntaxCheckerAllOtherOptions", ui.txtAllChecksMoreOptions->toPlainText());

    settings.endGroup();
}

} //end namespace ito

