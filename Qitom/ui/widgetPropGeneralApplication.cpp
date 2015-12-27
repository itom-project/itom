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

#include "widgetPropGeneralApplication.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qfiledialog.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropGeneralApplication::WidgetPropGeneralApplication(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    ui.btnAdd->setEnabled(true);
    ui.btnRemove->setEnabled(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropGeneralApplication::~WidgetPropGeneralApplication()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("MainWindow");
    ui.checkAskBeforeExit->setChecked( settings.value("askBeforeClose", false).toBool() );
    settings.endGroup();

    settings.beginGroup("Application");
    ui.spinBoxTimeoutGeneral->setValue(settings.value("timeoutGeneral", PLUGINWAIT).toInt());
    ui.spinBoxTimeoutInitClose->setValue(settings.value("timeoutInitClose", 10000).toInt());
    ui.spinBoxTimeoutFileSaveLoad->setValue(settings.value("timeoutFileSaveLoad", 60000).toInt());

	QListWidgetItem *lwi = new QListWidgetItem(QCoreApplication::applicationDirPath() + "/lib", ui.listWidget, QListWidgetItem::UserType);
    lwi->setTextColor(Qt::gray);

    int size = settings.beginReadArray("searchPathes");
    for (int i = 0; i < size; ++i) 
    {
        settings.setArrayIndex(i);
        ui.listWidget->addItem(settings.value("path", QString()).toString());
    }
	settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("MainWindow");
    settings.setValue("askBeforeClose", ui.checkAskBeforeExit->isChecked() );
    settings.endGroup();

    QStringList files;
    settings.beginGroup("Application");
    settings.setValue("timeoutGeneral", ui.spinBoxTimeoutGeneral->value());
    settings.setValue("timeoutInitClose", ui.spinBoxTimeoutInitClose->value());
    settings.setValue("timeoutFileSaveLoad", ui.spinBoxTimeoutFileSaveLoad->value());
    //apply the new timeout times in the current session, too.
    AppManagement::timeouts.pluginInitClose = ui.spinBoxTimeoutInitClose->value();
    AppManagement::timeouts.pluginGeneral = ui.spinBoxTimeoutGeneral->value();
    AppManagement::timeouts.pluginFileSaveLoad = ui.spinBoxTimeoutFileSaveLoad->value();

    settings.beginWriteArray("searchPathes");
    for (int i = 1; i < ui.listWidget->count(); i++)
    {
        settings.setArrayIndex(i - 1);
        settings.setValue("path", ui.listWidget->item(i)->text());
        files.append(ui.listWidget->item(i)->text());
    }
    settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* /*previous*/)
{
	ui.btnRemove->setEnabled(current != NULL && current->type() != QListWidgetItem::UserType);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_btnAdd_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("load directory"), QDir::currentPath());

    if (dir != "" && ui.listWidget->findItems(dir, Qt::MatchExactly).isEmpty())
    {
        ui.listWidget->addItem(dir);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_btnRemove_clicked()
{
    qDeleteAll(ui.listWidget->selectedItems());
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_listWidget_itemActivated(QListWidgetItem* item)
{
    if (item)
    {
/*        item->setFlags(item->flags() | Qt::ItemIsEditable);
        ui.listWidget->editItem(item);*/
    }
}

} //end namespace ito
