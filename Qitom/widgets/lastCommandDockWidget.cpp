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

#include "../python/pythonEngineInc.h"

#include "lastCommandDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include "../organizer/scriptEditorOrganizer.h"

#include <qheaderview.h>
#include <qsettings.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
LastCommandDockWidget::LastCommandDockWidget(const QString &title, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, parent),
    m_lastCommandTreeWidget(NULL),
    m_pActClearList(NULL)
{
    AbstractDockWidget::init();

    m_lastCommandTreeWidget = new QTreeWidget(this);
    setContentWidget(m_lastCommandTreeWidget);
    m_lastCommandTreeWidget->header()->hide();
    connect(m_lastCommandTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("lastCommandDockWidget");
    int size = settings.beginReadArray("lastUsedCommands");
    QTreeWidgetItem *parentItem = NULL;
    QTreeWidgetItem *childItem = NULL;
    QString date = "";
    QString tmpDate = "";
    for (int i = 0; i < size; ++i) 
    {
        settings.setArrayIndex(i);
        QString cmd = settings.value("cmd", QString()).toString();
        tmpDate = cmd.mid(0,10);
        if (QString::compare(tmpDate, date) != 0)
        {
            date = tmpDate;
            parentItem = new QTreeWidgetItem(m_lastCommandTreeWidget);
            parentItem->setText(0, date);
            parentItem->setTextColor(0, QColor("green"));
            m_lastCommandTreeWidget->addTopLevelItem(parentItem);
        }

        childItem = new QTreeWidgetItem(parentItem);
        childItem->setText(0, cmd.mid(11));
        childItem->setTextColor(0, QColor("black"));
        parentItem->addChild(childItem);
    }
    settings.endArray();
    settings.endGroup();

    m_lastCommandTreeWidget->setItemsExpandable(true);
    m_lastCommandTreeWidget->setExpandsOnDoubleClick(false);
    m_lastCommandTreeWidget->expandAll();
    m_lastCommandTreeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_lastCommandTreeWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(treeWidgetContextMenuRequested(const QPoint &)));

    if (childItem)
    {
        m_lastCommandTreeWidget->scrollToItem(childItem);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
LastCommandDockWidget::~LastCommandDockWidget()
{
    QString date = "";
    QStringList cmdList;
    cmdList.clear();
    for (int x = 0 ; x < m_lastCommandTreeWidget->topLevelItemCount() ; ++x)
    {
        date = m_lastCommandTreeWidget->topLevelItem(x)->text(0);
        for (int y = 0 ; y < m_lastCommandTreeWidget->topLevelItem(x)->childCount() ; ++y)
        {
            cmdList.append(date + " " + m_lastCommandTreeWidget->topLevelItem(x)->child(y)->text(0));
        }
    }

    int firstListIndex = 0;
    if (cmdList.count() > 99)
    {
        firstListIndex = cmdList.count() - 99;
    }

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    QStringList files;
    settings.beginGroup("lastCommandDockWidget");
    settings.beginWriteArray("lastUsedCommands");
    for (int i = firstListIndex ; i < cmdList.count() ; ++i)
    {
        settings.setArrayIndex(i);
        settings.setValue("cmd", cmdList[i]);
    }
    settings.endArray();
    settings.endGroup();

    disconnect(m_lastCommandTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));
    
    DELETE_AND_SET_NULL(m_lastCommandTreeWidget);
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::createActions()
{
    m_pActClearList = new ShortcutAction(QIcon(":/editor/icons/editDelete.png"), tr("clear list"), this);
    m_pActClearList->connectTrigger(this, SLOT(mnuClearList()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
    m_pContextMenu->addAction(m_pActClearList->action());
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::createToolBars()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::updateActions()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::addLastCommand(const QString cmd)
{
    QDate date(QDate::currentDate());
    QString strDate = date.toString(Qt::ISODate);
    QTreeWidgetItem *parentItem;
    QList<QTreeWidgetItem *> itemList = m_lastCommandTreeWidget->findItems(strDate, Qt::MatchExactly);
    if (itemList.empty())
    {
        parentItem = new QTreeWidgetItem(m_lastCommandTreeWidget);
        parentItem->setText(0, strDate);
        parentItem->setTextColor(0, QColor("green"));
        m_lastCommandTreeWidget->addTopLevelItem(parentItem);
    }
    else
    {
        parentItem = itemList[0];
    }

    QTreeWidgetItem *childItem = new QTreeWidgetItem(parentItem);
    childItem->setText(0, cmd);
    childItem->setTextColor(0, QColor("black"));
    parentItem->addChild(childItem);
    parentItem->setExpanded(true);
    m_lastCommandTreeWidget->scrollToItem(childItem);
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::itemDoubleClicked(QTreeWidgetItem *item, int column)
{
    if (item->parent() != NULL)
    {
        emit runPythonCommand(item->text(0));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::mnuClearList()
{
    m_lastCommandTreeWidget->clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::treeWidgetContextMenuRequested(const QPoint &pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_lastCommandTreeWidget->mapToGlobal(m_lastCommandTreeWidget->pos()));
}


} //end namespace ito
