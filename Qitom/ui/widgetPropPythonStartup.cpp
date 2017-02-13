/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

    ui.lblBasePath->setText(tr("Base path for relative pathes: ") + QCoreApplication::applicationDirPath());
    ui.btnAdd->setEnabled(true);
    ui.btnRemove->setEnabled(false);
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
    ui.btnRemove->setEnabled(current != NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_btnAdd_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Load python script"), QDir::currentPath(), tr("Python script (*.py)"));

    if (!filenames.empty())
    {
        QDir::setCurrent(QFileInfo(filenames.first()).path());

        foreach (QString filename, filenames)
        {
            if (ui.listWidget->findItems(filename, Qt::MatchExactly).isEmpty())
            {
                ui.listWidget->addItem(filename);
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_btnRemove_clicked()
{
    qDeleteAll(ui.listWidget->selectedItems());
/*    if (ui.listWidget->currentItem())
    {
        ui.listWidget->takeItem(ui.listWidget->currentIndex().row());
    }*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonStartup::on_listWidget_itemActivated(QListWidgetItem* item)
{
    if (item)
    {
/*        item->setFlags(item->flags() | Qt::ItemIsEditable);
        ui.listWidget->editItem(item);*/
    }
}

} //end namespace ito