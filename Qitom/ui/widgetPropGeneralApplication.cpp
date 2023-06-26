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
    ui.btnMoveUp->setEnabled(false);
    ui.btnMoveDown->setEnabled(false);
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
	ui.checkAskBeforeExit->setChecked(settings.value("askBeforeClose", true).toBool());

    settings.endGroup();

    settings.beginGroup("Application");
    ui.spinBoxTimeoutGeneral->setValue(settings.value("timeoutGeneral", PLUGINWAIT).toInt());
    ui.spinBoxTimeoutInitClose->setValue(settings.value("timeoutInitClose", 10000).toInt());
    ui.spinBoxTimeoutFileSaveLoad->setValue(settings.value("timeoutFileSaveLoad", 60000).toInt());

	QListWidgetItem *lwi = new QListWidgetItem(QCoreApplication::applicationDirPath() + "/lib", ui.listWidget, QListWidgetItem::UserType);
    lwi->setForeground(Qt::gray);

    QStringList prepend;
    QStringList append;

    int size = settings.beginReadArray("searchPathes");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        if (settings.value("prepend", true).toBool())
        {
            prepend.append(settings.value("path", QString()).toString());
        }
        else
        {
            append.append(settings.value("path", QString()).toString());
        }
    }
	settings.endArray();
    settings.endGroup();

    foreach(const QString &p, prepend)
    {
        ui.listWidget->addItem(p);
    }

    lwi = new QListWidgetItem("pathes from global PATH variable", ui.listWidget, QListWidgetItem::UserType + 1);
    lwi->setForeground(Qt::gray);

    foreach(const QString &p, append)
    {
        ui.listWidget->addItem(p);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("MainWindow");
    settings.setValue("askBeforeClose", ui.checkAskBeforeExit->isChecked());
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

    bool prependToPath = true;

    settings.beginWriteArray("searchPathes");
    int counter = 0;
    for (int i = 0; i < ui.listWidget->count(); i++)
    {
        switch (ui.listWidget->item(i)->type())
        {
        case QListWidgetItem::UserType:
            break;
        case QListWidgetItem::UserType + 1:
            prependToPath = false;
            break;
        default:
            settings.setArrayIndex(counter);
            settings.setValue("path", ui.listWidget->item(i)->text());
            settings.setValue("prepend", prependToPath);
            files.append(ui.listWidget->item(i)->text());
            counter++;
            break;
        }
    }

    settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* /*previous*/)
{
    ui.btnRemove->setEnabled(current != NULL && current->type() != QListWidgetItem::UserType && current->type() != QListWidgetItem::UserType + 1);
    ui.btnMoveUp->setEnabled(current != NULL && current->type() != QListWidgetItem::UserType && ui.listWidget->row(current) > 1);
    ui.btnMoveDown->setEnabled(current != NULL && current->type() != QListWidgetItem::UserType && ui.listWidget->row(current) < ui.listWidget->count() - 1);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_btnAdd_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("load directory"), QDir::currentPath());

    if (dir != "" && ui.listWidget->findItems(dir, Qt::MatchExactly).isEmpty())
    {
        ui.listWidget->insertItem(qMax(1, ui.listWidget->currentRow()), dir);
    }

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_btnRemove_clicked()
{
    qDeleteAll(ui.listWidget->selectedItems());

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_btnMoveUp_clicked()
{
    QListWidgetItem *current = ui.listWidget->currentItem();
    if (current)
    {
        int currentRow = ui.listWidget->currentRow();
        int previousRow = currentRow - 1;
        QListWidgetItem *take = ui.listWidget->takeItem(previousRow);
        ui.listWidget->insertItem(currentRow, take);
    }

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropGeneralApplication::on_btnMoveDown_clicked()
{
    QListWidgetItem *current = ui.listWidget->currentItem();
    if (current)
    {
        int currentRow = ui.listWidget->currentRow();
        int nextRow = currentRow + 1;
        QListWidgetItem *take = ui.listWidget->takeItem(nextRow);
        ui.listWidget->insertItem(currentRow, take);
    }

    on_listWidget_currentItemChanged(ui.listWidget->currentItem(), NULL);
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
