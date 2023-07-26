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

#include "widgetPropEditorAutoCodeFormat.h"

#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qmessagebox.h>

namespace ito
{

//-------------------------------------------------------------------------------------
WidgetPropEditorAutoCodeFormat::WidgetPropEditorAutoCodeFormat(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    m_demoCode = "import libB\nimport libA\n#comment\nif True:\n  print('test')\n  abc=[1,2,3]";
}

//-------------------------------------------------------------------------------------
WidgetPropEditorAutoCodeFormat::~WidgetPropEditorAutoCodeFormat()
{
}

//-------------------------------------------------------------------------------------
void WidgetPropEditorAutoCodeFormat::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    ui.groupAutoCodeFormat->setChecked(
        settings.value("autoCodeFormatEnabled", true).toBool()
    );

    ui.txtCmd->setText(
        settings.value("autoCodeFormatCmd", "black --line-length 88 --quiet -").toString()
    );

    ui.txtPreCmd->setText(
        settings.value("autoCodeFormatImportsSortCmd", "isort --py 3 --profile black").toString()
    );

    ui.groupImportsSorting->setChecked(
        settings.value("autoCodeFormatEnableImportsSort", false).toBool()
    );

    settings.endGroup();
}

//-------------------------------------------------------------------------------------
void WidgetPropEditorAutoCodeFormat::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    settings.setValue("autoCodeFormatEnabled", ui.groupAutoCodeFormat->isChecked());
    settings.setValue("autoCodeFormatCmd", ui.txtCmd->toPlainText());
    settings.setValue("autoCodeFormatImportsSortCmd", ui.txtPreCmd->text());
    settings.setValue("autoCodeFormatEnableImportsSort", ui.groupImportsSorting->isChecked());

    settings.endGroup();
}

//-------------------------------------------------------------------------------------
/*static*/ void WidgetPropEditorAutoCodeFormat::deleteLater(QObject *obj)
{
    obj->deleteLater();
}

//-------------------------------------------------------------------------------------
void WidgetPropEditorAutoCodeFormat::on_btnTest_clicked()
{
    m_pyCodeFormatter = QSharedPointer<PyCodeFormatter>(
        new PyCodeFormatter(this),
        WidgetPropEditorAutoCodeFormat::deleteLater
    );

    connect(m_pyCodeFormatter.data(), &PyCodeFormatter::formattingDone,
        this, &WidgetPropEditorAutoCodeFormat::testCodeFormatterDone);

    QString importSortCmd = "";

    if (ui.groupImportsSorting->isChecked())
    {
        importSortCmd = ui.txtPreCmd->text();
    }

    ito::RetVal retval = m_pyCodeFormatter->startSortingAndFormatting(
        importSortCmd, ui.txtCmd->toPlainText(), m_demoCode, this);

    if (retval.containsError())
    {
        QMessageBox::critical(
            this,
            tr("Test failed"),
            tr("The code formatting could not be started: %1").arg(retval.errorMessage())
        );
    }
}

//-------------------------------------------------------------------------------------
void WidgetPropEditorAutoCodeFormat::on_btnTake_clicked()
{
    ui.txtCmd->setText(ui.comboDefault->currentText());
}

//-------------------------------------------------------------------------------------
void WidgetPropEditorAutoCodeFormat::testCodeFormatterDone(bool success, QString code)
{
    QStringList codeIn = m_demoCode.split("\n");

    for (int i = 0; i < codeIn.size(); ++i)
    {
        codeIn[i].prepend(">> ");
    }

    if (success && code.trimmed() != "")
    {
        QStringList codeOut = code.split("\n");

        for (int i = 0; i < codeOut.size(); ++i)
        {
            codeOut[i].prepend(">> ");
        }

        QMessageBox::information(
            this,
            tr("Successful test"),
            tr("The test python code\n\n%1\n\nhas been successful formatted to\n\n%2")
            .arg(codeIn.join("\n"))
            .arg(codeOut.join("\n")));
    }
    else
    {
        QMessageBox::critical(
            this,
            tr("Test failed"),
            tr("The test python code\n\n%1\n\ncould not been formatted. Reason:\n%2")
            .arg(codeIn.join("\n"))
            .arg(code != "" ? code : tr("empty code returned."))
        );
    }
}

} //end namespace ito
