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

#include "WidgetPropHelpDock.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>



WidgetPropHelpDock::WidgetPropHelpDock(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
	m_pDBFiles(NULL),
	m_pdbPath(qApp->applicationDirPath()+"/help/")
{
    ui.setupUi(this);
	m_pDBFiles = new QStringList();
	m_plistChanged = false;
	ui.label->hide();
}

WidgetPropHelpDock::~WidgetPropHelpDock()
{

}


void WidgetPropHelpDock::on_listWidget_itemChanged(QListWidgetItem * item)
{
	m_plistChanged = true;
	ui.label->show();
}

void WidgetPropHelpDock::refreshDBs()
{
	QDirIterator it(m_pdbPath, QStringList("*.db"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
	QStringList foundDBs;
	QString dbName;
	foundDBs.clear();
    while(it.hasNext())
    {
		QString iter = it.next();
		dbName = iter.right(iter.length()-1-iter.lastIndexOf('/'));
		foundDBs.append(dbName);
	}

	// Remove entries from Listview if they are not in the new list anymore
	int j = 0;
	while (j < ui.listWidget->count())
	{
		int g = ui.listWidget->count();
		QString cont = ui.listWidget->item(j)->text();
		if (foundDBs.indexOf(QRegExp(cont)) == -1)
		{
			ui.listWidget->takeItem(j);
		}
		else
		{
			j++;
		}
	}

	// Remove entries from new List that already exist in the Listview
	//int j = 0;
	for (int i = 0; i < ui.listWidget->count() ; i++)
	{
		if (foundDBs.contains(ui.listWidget->item(i)->text()))
		{
			foundDBs.removeAt(foundDBs.indexOf(QRegExp(ui.listWidget->item(i)->text())));
		}
		//if (ui.listWidget->findItems(foundDBs[j], Qt::MatchExactly).count() > 0)
		
	}

	// Put %% infront of all entries
	QListWidgetItem *nextDBItem;
	for (int k = 0; k < foundDBs.length(); k++)
	{
		nextDBItem = new QListWidgetItem(foundDBs[k]);
		nextDBItem->setCheckState(Qt::Unchecked);
		ui.listWidget->addItem(nextDBItem);
	}
}

void WidgetPropHelpDock::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("helpTreeDockWidget");

    ui.checkBox->setChecked( settings.value("OpenExtLinks", true).toBool() );
	ui.checkBox_2->setChecked( settings.value("Plaintext", false).toBool() );

	// Read the List of Databases and if they are checked ($ = checked, % = unchecked)
	int size = settings.beginReadArray("Databases");
	QListWidgetItem *nextDBItem;
	for (int i = 0; i < size; ++i)
	{
		settings.setArrayIndex(i);
		QString dbNameC = settings.value("DB", QString()).toString();
		QString dbName = dbNameC;
		dbName.remove(0,2);
		nextDBItem = new QListWidgetItem(dbName);
		if (dbNameC.startsWith("$"))	
		{// Check the new Item
			nextDBItem->setCheckState(Qt::Checked);
		}
		else if (dbNameC.startsWith("%")) 
		{// Don´t check the new Item
			nextDBItem->setCheckState(Qt::Unchecked);
		}
		ui.listWidget->addItem(nextDBItem);
	}
    settings.endGroup();
	refreshDBs();
}

void WidgetPropHelpDock::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("helpTreeDockWidget");

    settings.setValue("OpenExtLinks", ui.checkBox->isChecked() );
	settings.setValue("Plaintext", ui.checkBox_2->isChecked() );
	settings.setValue("reLoadDBs", m_plistChanged );

	// Write the checkstate with the List into the ini File
	settings.beginWriteArray("Databases");
    for (int i = 0; i < ui.listWidget->count(); ++i)
    {
        settings.setArrayIndex(i);
		if (ui.listWidget->item(i)->checkState() == Qt::Checked)
		{// Item was checked => add $$
			settings.setValue("DB", "$$"+ui.listWidget->item(i)->text());
		}
		else
		{// Item was unchecked => add %%
			settings.setValue("DB", "%%"+ui.listWidget->item(i)->text());
		}
    }
    settings.endArray();
    settings.endGroup();
}

