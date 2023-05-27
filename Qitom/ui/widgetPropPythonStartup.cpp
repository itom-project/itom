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

#include "widgetPropPythonStartup.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qcoreapplication.h>
#include <qfiledialog.h>
#include <qstringlist.h>
#include <qdir.h>
#include <qfileinfo.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPythonStartup::WidgetPropPythonStartup(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    QString label = ui.checkAddFileRel->text().arg(QCoreApplication::applicationDirPath());
    ui.checkAddFileRel->setText(label);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPythonStartup::~WidgetPropPythonStartup()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Python");

    int size = settings.beginReadArray("startupFiles");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);

        ui.listWidget->addItem(settings.value("file", QString()).toString());
    }

    settings.endArray();
    settings.endGroup();

    updateScriptButtons();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::writeSettings()
{
    QStringList files;
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Python");
    settings.beginWriteArray("startupFiles");
    for (int i = 0; i < ui.listWidget->count(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("file", ui.listWidget->item(i)->text());
        files.append(ui.listWidget->item(i)->text());
    }

    settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* /*previous*/)
{
    updateScriptButtons();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_btnAdd_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Load python script"), QDir::currentPath(), tr("Python script (*.py)"));

    if (!filenames.empty())
    {
        QDir::setCurrent(QFileInfo(filenames.first()).path());
        QDir baseDir(QCoreApplication::applicationDirPath());

        foreach (QString filename, filenames)
        {
            if (ui.checkAddFileRel->isChecked())
            {
                filename = baseDir.relativeFilePath(filename);
            }

            if (ui.listWidget->findItems(filename, Qt::MatchExactly).isEmpty())
            {
                ui.listWidget->addItem(filename);
            }
        }

        updateScriptButtons();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_btnRemove_clicked()
{
    qDeleteAll(ui.listWidget->selectedItems());
    updateScriptButtons();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::updateScriptButtons()
{
    int currentRow = ui.listWidget->currentRow();
    int rows = ui.listWidget->count();
    ui.btnRemove->setEnabled(currentRow >= 0);
    ui.btnDownScript->setEnabled((currentRow >= 0) && (currentRow < (rows - 1)));
    ui.btnUpScript->setEnabled(currentRow > 0);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_btnDownScript_clicked()
{
    int currentRow = ui.listWidget->currentRow();
    int numRows = ui.listWidget->count();

    if (currentRow < (numRows - 1))
    {
        QListWidgetItem *item = ui.listWidget->item(currentRow);
        QString text = item->text();
        DELETE_AND_SET_NULL(item);
        ui.listWidget->insertItem(currentRow + 1, text);
        ui.listWidget->setCurrentRow(currentRow + 1);
        updateScriptButtons();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_btnUpScript_clicked()
{
    int currentRow = ui.listWidget->currentRow();

    if (currentRow > 0)
    {
        QListWidgetItem *item = ui.listWidget->item(currentRow);
        QString text = item->text();
        DELETE_AND_SET_NULL(item);
        ui.listWidget->insertItem(currentRow - 1, text);
        ui.listWidget->setCurrentRow(currentRow - 1);
        updateScriptButtons();
    }
}

} //end namespace ito
