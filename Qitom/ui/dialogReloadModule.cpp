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

#include "../python/pythonEngine.h"

#include "dialogReloadModule.h"

#include "../global.h"
#include "../../common/sharedStructuresQt.h"
#include "../AppManagement.h"

#include <qmessagebox.h>
#include <qsharedpointer.h>
#include <qmetaobject.h>
#include <qtimer.h>

namespace ito
{

DialogReloadModule::DialogReloadModule(QWidget* parent) :
    QDialog(parent)
{
    ui.setupUi(this);
    connect(ui.btnReload, SIGNAL(clicked()), this, SLOT(dialogAccepted()));
    connect(ui.treeWidget, SIGNAL(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)), this, SLOT(currentItemChanged(QTreeWidgetItem*,QTreeWidgetItem*)));
    connect(ui.checkShowBuildin, SIGNAL(clicked(bool)), this, SLOT(checkBuildinClicked(bool)));
    connect(ui.checkShowFromPythonPath, SIGNAL(clicked(bool)), this, SLOT(checkPythonPathClicked(bool)));

    ui.treeWidget->setHeaderLabel(tr("Module Name"));
    ui.treeWidget->sortByColumn(0, Qt::AscendingOrder);
    ui.treeWidget->setSelectionMode(QAbstractItemView::MultiSelection);

    enableUI(false);

    QTimer::singleShot(0, this, SLOT(loadModules()));

}

void DialogReloadModule::loadModules()
{
    int pos;
    QString key;
    QString part1, part2;
    QTreeWidgetItem *twi;

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if(pyEngine == NULL)
    {
        QMessageBox::critical(this, tr("Python Engine is invalid"), tr("The Python Engine could not be found"));
        enableUI(false);
    }
    else
    {

        QSharedPointer<QStringList> modNames = QSharedPointer<QStringList>(new QStringList());
        QSharedPointer<QStringList> modFilenames = QSharedPointer<QStringList>(new QStringList());
        QSharedPointer<IntList> modTypes = QSharedPointer<IntList>(new IntList());
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(pyEngine, "getSysModules", Q_ARG(QSharedPointer<QStringList>,modNames), Q_ARG(QSharedPointer<QStringList>,modFilenames), Q_ARG(QSharedPointer<IntList>,modTypes), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

        //clear old
        ui.treeWidget->clear();
        m_items.clear();

        if(!locker.getSemaphore()->waitAndProcessEvents(PLUGINWAIT)) //this is important, since the garbage collector might be called when calling getSysModules. If the gc is destructing an ui-instance, the uiOrganizer is invoked, which lives in the same thread than this dialog.
        {
            QMessageBox::critical(this, tr("Connection problem"), tr("No information about loaded modules could be retrieved by python."));
            enableUI(false);
        }
        else if(locker.getSemaphore()->returnValue != ito::retOk)
        {
            if(locker.getSemaphore()->returnValue.hasErrorMessage())
            {
                QMessageBox::critical(this, tr("Error while getting module list"), QLatin1String(locker.getSemaphore()->returnValue.errorMessage()));
            }
            else
            {
                QMessageBox::critical(this, tr("Error while getting module list"), tr("Unknown error"));
            }
        }
        else
        {
            //refill

            for(int i=0;i<modNames->size();i++)
            {
                key = (*modNames)[i];
                pos = key.lastIndexOf( ".");
                part1 = pos >= 0 ? key.left(pos) : "";
                part2 = pos >= 0 ? key.mid(pos+1) : key;

                if(m_items.contains(part1))
                {
                    twi = new QTreeWidgetItem(m_items[part1]);
                    twi->setText(0, part2);
                    twi->setData(0, Qt::UserRole + 1, (*modFilenames)[i]);
                    twi->setData(0, Qt::UserRole + 2, key);
                    twi->setData(0, Qt::UserRole + 3, (*modTypes)[i]);
                    m_items[key] = twi;
                }
                else
                {
                    twi = new QTreeWidgetItem(ui.treeWidget);
                    twi->setText(0, key);
                    twi->setData(0, Qt::UserRole + 1, (*modFilenames)[i]);
                    twi->setData(0, Qt::UserRole + 2, key);
                    twi->setData(0, Qt::UserRole + 3, (*modTypes)[i]);
                    m_items[key] = twi;
                }
            }

            filterItems();
            enableUI(true);
        }
    }

}

void DialogReloadModule::dialogAccepted()
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if(pyEngine == NULL)
    {
        QMessageBox::critical(this, tr("Python Engine is invalid"), tr("The Python Engine could not be found"));
        emit accept();
    }
    else
    {

        QList<QTreeWidgetItem*> selected = ui.treeWidget->selectedItems();
        QSharedPointer<QStringList> mods = QSharedPointer<QStringList>(new QStringList());
        foreach(QTreeWidgetItem *item, selected)
        {
            (*mods).append(item->data(0, Qt::UserRole + 2).toString());
        }

        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(pyEngine, "reloadSysModules", Q_ARG(QSharedPointer<QStringList>,mods), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));

        if(locker.getSemaphore()->waitAndProcessEvents(PLUGINWAIT) == false) //this is important, since the garbage collector might be called when calling getSysModules. If the gc is destructing an ui-instance, the uiOrganizer is invoked, which lives in the same thread than this dialog.
        {
            QMessageBox::critical(this, tr("connection problem"), tr("Timeout while forcing python to reload modules."));
        }
        else if(locker.getSemaphore()->returnValue != ito::retOk)
        {
            if(locker.getSemaphore()->returnValue.hasErrorMessage())
            {
                QMessageBox::critical(this, tr("Error while reloading modules"), QLatin1String(locker.getSemaphore()->returnValue.errorMessage()));
            }
            else
            {
                QMessageBox::critical(this, tr("Error while reloading modules"), tr("Unknown error"));
            }
        }
        else if(mods->count() > 0) //some modules could not be loaded
        {
            QMessageBox::information(this, tr("Module reload"), tr("The following modules could not be reloaded:\n") + mods->join("\n"));
        }

        emit accept();
    }
}

void DialogReloadModule::enableUI(bool enabled)
{
    ui.btnReload->setEnabled(enabled);
}

void DialogReloadModule::currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem * /*previous*/)
{
    if(current)
    {
        ui.lblModuleName->setText( current->data(0, Qt::UserRole + 2).toString());
        ui.lblPath->setText( current->data(0, Qt::UserRole + 1).toString());
    }
    else
    {
        ui.lblModuleName->setText( tr("<click on item to view detailed information>"));
        ui.lblPath->setText("");
    }
}

void DialogReloadModule::filterItems()
{
    bool showBuildins = ui.checkShowBuildin->isChecked();
    bool showPythonPath = ui.checkShowFromPythonPath->isChecked();
    int type;

    foreach(QTreeWidgetItem *item, m_items)
    {
        type = item->data(0, Qt::UserRole + 3).toInt();
        if(type == 1 && !showBuildins)
        {
            item->setHidden(true);
        }
        else if(type == 2 && !showPythonPath)
        {
            item->setHidden(true);
        }
        else
        {
            item->setHidden(false);
        }
    }
}

} //end namespace ito
