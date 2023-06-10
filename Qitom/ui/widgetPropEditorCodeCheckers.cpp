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

#include "widgetPropEditorCodeCheckers.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../codeEditor/codeEditor.h"
#include "../python/pythonCommon.h" //required for some enumeration values related to code checkers
#include <qmenu.h>
#include <qsettings.h>
#include <qmap.h>
#include <qvariant.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorCodeCheckers::WidgetPropEditorCodeCheckers(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorCodeCheckers::~WidgetPropEditorCodeCheckers()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorCodeCheckers::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // General Settings
    ui.checkIncludeItom->setChecked(settings.value("codeCheckerAutoImportItem", true).toBool());
    ui.spinSyntaxInterval->setValue(settings.value("codeCheckerInterval", 1.00).toDouble());

    // Basic Options
    switch (settings.value("codeCheckerMode", PythonCommon::CodeCheckerAuto).toInt())
    {
    case PythonCommon::NoCodeChecker:
        ui.radioDisableChecks->setChecked(true);
        break;
    case PythonCommon::CodeCheckerPyFlakes:
        ui.radioEnableSyntaxCheck->setChecked(true);
        break;
    case PythonCommon::CodeCheckerFlake8:
        ui.radioEnableAllChecks->setChecked(true);
        break;
    case PythonCommon::CodeCheckerAuto:
    default:
        ui.radioEnableAutoChecks->setChecked(true);
        break;
    }

    switch (settings.value("codeCheckerShowMinCategoryLevel", PythonCommon::TypeInfo).toInt())
    {
    case PythonCommon::TypeInfo: //information
    default:
        ui.comboShowMinLevel->setCurrentIndex(0);
        break;
    case PythonCommon::TypeWarning: //warning
        ui.comboShowMinLevel->setCurrentIndex(1);
        break;
    case PythonCommon::TypeError: //error
        ui.comboShowMinLevel->setCurrentIndex(2);
        break;
    }

    QVariantMap syntaxCheckerParams = settings.value("codeCheckerProperties").toMap();

    //Basic Options (PyFlakes)
    switch (syntaxCheckerParams.value("codeCheckerPyFlakesCategory",
        PythonCommon::TypeError).toInt())
    {
    case PythonCommon::TypeInfo: //information
        ui.comboSyntaxCategories->setCurrentIndex(0);
        break;
    case PythonCommon::TypeWarning: //warning
        ui.comboSyntaxCategories->setCurrentIndex(1);
        break;
    case PythonCommon::TypeError: //error
    default:
        ui.comboSyntaxCategories->setCurrentIndex(2);
        break;
    }

    //All Options (Flake8)
    ui.spinAllChecksLineLength->setValue(syntaxCheckerParams.value("codeCheckerFlake8MaxLineLength", 79).toInt());

    ui.checkAllChecksSelect->setChecked(syntaxCheckerParams.value("codeCheckerFlake8SelectEnabled", false).toBool());
    ui.lblAllChecksSelect->setText(syntaxCheckerParams.value("codeCheckerFlake8SelectValues", "").toString());

    ui.checkAllChecksIgnore->setChecked(syntaxCheckerParams.value("codeCheckerFlake8IgnoreEnabled", false).toBool());
    ui.lblAllChecksIgnore->setText(syntaxCheckerParams.value("codeCheckerFlake8IgnoreValues", "").toString());

    ui.checkAllChecksIgnoreExtend->setChecked(syntaxCheckerParams.value("codeCheckerFlake8IgnoreExtendEnabled", true).toBool());
    ui.lblAllChecksIgnoreExtend->setEnabled(ui.checkAllChecksIgnoreExtend->isChecked());
    ui.lblAllChecksIgnoreExtend->setText(syntaxCheckerParams.value("codeCheckerFlake8IgnoreExtendValues", "W293").toString());

    ui.comboAllChecksDocstyle->setCurrentText(syntaxCheckerParams.value("codeCheckerFlake8Docstyle", "pep257").toString());

    ui.checkAllChecksMaxComplexity->setChecked(syntaxCheckerParams.value("codeCheckerFlake8MaxComplexityEnabled", false).toBool());
    ui.spinAllChecksMaxComplexity->setValue(syntaxCheckerParams.value("codeCheckerFlake8MaxComplexity", 10).toInt());

    ui.lblAllChecksErrorNumbers->setText(syntaxCheckerParams.value("codeCheckerFlake8ErrorNumbers", "F").toString());
    ui.lblAllCheckWarningNumbers->setText(syntaxCheckerParams.value("codeCheckerFlake8WarningNumbers", "E, C").toString());

    ui.txtAllChecksMoreOptions->setPlainText(syntaxCheckerParams.value("codeCheckerFlake8OtherOptions", "").toString());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
/*

Please consider to set all default values, which are directly passed to pyflakes
or flake8 (contained in codeCheckerProperties) in the python script itomSyntaxCheck.py, too.
*/
void WidgetPropEditorCodeCheckers::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // General Settings
    settings.setValue("codeCheckerInterval", ui.spinSyntaxInterval->value());
    settings.setValue("codeCheckerAutoImportItem", ui.checkIncludeItom->isChecked());

    // Basic Options
    if (ui.radioDisableChecks->isChecked())
    {
        settings.setValue("codeCheckerMode", PythonCommon::NoCodeChecker);
    }
    else if (ui.radioEnableSyntaxCheck->isChecked())
    {
        settings.setValue("codeCheckerMode", PythonCommon::CodeCheckerPyFlakes);
    }
    else if (ui.radioEnableAllChecks->isChecked())
    {
        settings.setValue("codeCheckerMode", PythonCommon::CodeCheckerFlake8);
    }
    else
    {
        settings.setValue("codeCheckerMode", PythonCommon::CodeCheckerAuto);
    }

    switch (ui.comboShowMinLevel->currentIndex())
    {
    case 0: //information
    default:
        settings.setValue("codeCheckerShowMinCategoryLevel", PythonCommon::TypeInfo);
        break;
    case 1: //warning
        settings.setValue("codeCheckerShowMinCategoryLevel", PythonCommon::TypeWarning);
        break;
    case 2: //error
        settings.setValue("codeCheckerShowMinCategoryLevel", PythonCommon::TypeError);
        break;
    }

    QVariantMap syntaxCheckerParams = settings.value("codeCheckerProperties", QVariantMap()).toMap();

    //Basic Options (PyFlakes)
    switch (ui.comboSyntaxCategories->currentIndex())
    {
    case 0: //information
        syntaxCheckerParams["codeCheckerPyFlakesCategory"] = PythonCommon::TypeInfo;
        break;
    case 1: //warning
        syntaxCheckerParams["codeCheckerPyFlakesCategory"] = PythonCommon::TypeWarning;
        break;
    case 2: //error
    default:
        syntaxCheckerParams["codeCheckerPyFlakesCategory"] = PythonCommon::TypeError;
        break;
    }

    //All Options (Flake8)
    syntaxCheckerParams["codeCheckerFlake8MaxLineLength"] = ui.spinAllChecksLineLength->value();

    syntaxCheckerParams["codeCheckerFlake8SelectEnabled"] = ui.checkAllChecksSelect->isChecked();
    syntaxCheckerParams["codeCheckerFlake8SelectValues"] = ui.lblAllChecksSelect->text();

    syntaxCheckerParams["codeCheckerFlake8IgnoreEnabled"] = ui.checkAllChecksIgnore->isChecked();
    syntaxCheckerParams["codeCheckerFlake8IgnoreValues"] = ui.lblAllChecksIgnore->text();

    syntaxCheckerParams["codeCheckerFlake8IgnoreExtendEnabled"] = ui.checkAllChecksIgnoreExtend->isChecked();
    syntaxCheckerParams["codeCheckerFlake8IgnoreExtendValues"] = ui.lblAllChecksIgnoreExtend->text();

    syntaxCheckerParams["codeCheckerFlake8Docstyle"] = ui.comboAllChecksDocstyle->currentText();

    syntaxCheckerParams["codeCheckerFlake8MaxComplexityEnabled"] = ui.checkAllChecksMaxComplexity->isChecked();
    syntaxCheckerParams["codeCheckerFlake8MaxComplexity"] = ui.spinAllChecksMaxComplexity->value();

    syntaxCheckerParams["codeCheckerFlake8ErrorNumbers"] = ui.lblAllChecksErrorNumbers->text();
    syntaxCheckerParams["codeCheckerFlake8WarningNumbers"] = ui.lblAllCheckWarningNumbers->text();

    QString text = ui.txtAllChecksMoreOptions->toPlainText();
    QStringList texts = text.trimmed().split("\n");
    QStringList texts2;

    foreach(const QString &t, texts)
    {
        if (t.trimmed() != "")
        {
            texts2.append(t.trimmed());
        }
    }

    syntaxCheckerParams["codeCheckerFlake8OtherOptions"] = texts2.join("\n");

    settings.setValue("codeCheckerProperties", syntaxCheckerParams);

    settings.endGroup();
}

} //end namespace ito
