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
QStringList LastCommandTreeWidget::mimeTypes() const
{
    QStringList types = QTreeWidget::mimeTypes();

    if (types.contains("text/plain") == false)
    {
        types.append("text/plain");
    }

    return types;
}

//----------------------------------------------------------------------------------------------------------------------------------
QMimeData * LastCommandTreeWidget::mimeData(const QList<QTreeWidgetItem *> items) const
{
    QMimeData *mimeData = QTreeWidget::mimeData(items);
    QStringList texts;

    //data from a itom-internal model (e.g. last command history)
    QByteArray encoded = mimeData->data("application/x-qabstractitemmodeldatalist");
    QDataStream stream(&encoded, QIODevice::ReadOnly);

    //check if it is really data from the command history (needs to only have strings and one column)
    while (!stream.atEnd())
    {
        int row, col;
        QMap<int,  QVariant> roleDataMap;
        stream >> row >> col >> roleDataMap;
        texts.append( roleDataMap[0].toString() );
    }

    mimeData->setData("text/plain", texts.join("\n").toAscii() );
    return mimeData;
}

//----------------------------------------------------------------------------------------------------------------------------------
LastCommandDockWidget::LastCommandDockWidget(const QString &title, const QString &objName, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_lastCommandTreeWidget(NULL),
    m_pActClearList(NULL),
    m_lastTreeWidgetParent(NULL)
{
    AbstractDockWidget::init();

    m_lastCommandTreeWidget = new LastCommandTreeWidget(this);
    m_lastCommandTreeWidget->installEventFilter(this);
    setContentWidget(m_lastCommandTreeWidget);
    m_lastCommandTreeWidget->header()->hide();
    m_lastCommandTreeWidget->setDragDropMode( QAbstractItemView::DragOnly );
    m_lastCommandTreeWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    connect(m_lastCommandTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));
    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("itomLastCommandDockWidget");
    m_enabled = settings.value("lastCommandEnabled", "true").toBool();
    m_dateColor = settings.value("lastCommandDateColor", "green").toString();
    m_timeStamp = settings.value("lastCommandTimeStamp", "false").toBool();
    m_doubleCommand = settings.value("lastCommandHideDoubleCommand", "false").toBool();

    QTreeWidgetItem *childItem = NULL;
    if (m_enabled)
    {
        int size = settings.beginReadArray("lastUsedCommands");
        QTreeWidgetItem *parentItem = NULL;
        QString date = "";
        QString tmpDate = "";
        for (int i = 0; i < size; ++i)
        {
            settings.setArrayIndex(i);
            QString cmd = settings.value("cmd", QString()).toString();
            tmpDate = cmd.mid(0, 10);
            if (QString::compare(tmpDate, date) != 0)
            {
                date = tmpDate;
                parentItem = new QTreeWidgetItem(m_lastCommandTreeWidget);
                parentItem->setText(0, date);
                parentItem->setTextColor(0, QColor(m_dateColor));
                parentItem->setFlags(Qt::ItemIsEnabled);
                m_lastCommandTreeWidget->addTopLevelItem(parentItem);
                m_lastTreeWidgetParent = parentItem;
            }

            childItem = new QTreeWidgetItem(parentItem);
            childItem->setText(0, cmd.mid(11));
//            childItem->setTextColor(0, QColor(0x80000008));  // Color of text in windows
            parentItem->addChild(childItem);
        }
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
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("itomLastCommandDockWidget");
    int commandNumbers = settings.value("lastCommandCommandNumbers", "100").toInt();

    QString date = "";
    QStringList cmdList;
    cmdList.clear();
    for (int x = 0; x < m_lastCommandTreeWidget->topLevelItemCount(); ++x)
    {
        date = m_lastCommandTreeWidget->topLevelItem(x)->text(0);
        for (int y = 0; y < m_lastCommandTreeWidget->topLevelItem(x)->childCount(); ++y)
        {
            cmdList.append(date + " " + m_lastCommandTreeWidget->topLevelItem(x)->child(y)->text(0));
        }
    }

    int firstListIndex = 0;
    if (cmdList.count() > commandNumbers)
    {
        firstListIndex = cmdList.count() - commandNumbers;
    }

    settings.remove("lastUsedCommands");
    settings.beginWriteArray("lastUsedCommands");
    for (int i = firstListIndex; i < cmdList.count(); ++i)
    {
        settings.setArrayIndex(i - firstListIndex);
        settings.setValue("cmd", cmdList[i]);
    }
    settings.endArray();
    settings.endGroup();

    disconnect(m_lastCommandTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));
    disconnect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));
    
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
    if (m_enabled)
    {
        QTreeWidgetItem *lastDateItem = NULL;
        QString lastSavedCommand;
        if (m_lastCommandTreeWidget->topLevelItemCount() > 0)
        {
            lastDateItem = m_lastCommandTreeWidget->topLevelItem(m_lastCommandTreeWidget->topLevelItemCount() - 1);
//            QTreeWidgetItem *lastCommand = NULL;
            if (lastDateItem->childCount() > 0)
            {
                lastSavedCommand = lastDateItem->child(lastDateItem->childCount() - 1)->text(0);
                if (m_timeStamp)
                {
                    lastSavedCommand = lastSavedCommand.mid(9);
                }
            }
        }

        if (!m_doubleCommand || QString::compare(lastSavedCommand, cmd, Qt::CaseInsensitive) != 0)
        {
            QDate date(QDate::currentDate());
            QString strDate = date.toString(Qt::ISODate);
            if (!m_lastTreeWidgetParent || m_lastTreeWidgetParent->text(0) != strDate)
            {
                QTreeWidgetItem *parentItem;
                parentItem = new QTreeWidgetItem(m_lastCommandTreeWidget);
                parentItem->setText(0, strDate);
                parentItem->setTextColor(0, QColor(m_dateColor));
                parentItem->setFlags(Qt::ItemIsEnabled);
                m_lastCommandTreeWidget->addTopLevelItem(parentItem);
                m_lastTreeWidgetParent = parentItem;
            }

            QTreeWidgetItem *childItem = new QTreeWidgetItem(m_lastTreeWidgetParent);
            QString addCmd;
            if (m_timeStamp)
            {
                QTime time(QTime::currentTime());
                addCmd = time.toString("hh:mm:ss") + " " + cmd;
            }
            else
            {
                addCmd = cmd;
            }
            childItem->setText(0, addCmd);
            m_lastTreeWidgetParent->addChild(childItem);
            m_lastTreeWidgetParent->setExpanded(true);
            m_lastCommandTreeWidget->scrollToItem(childItem);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::propertiesChanged()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("itomLastCommandDockWidget");
    m_enabled = settings.value("lastCommandEnabled", "true").toBool();
    m_dateColor = settings.value("lastCommandDateColor", "green").toString();
    m_timeStamp = settings.value("lastCommandTimeStamp", "false").toBool();
    m_doubleCommand = settings.value("lastCommandHideDoubleCommand", "false").toBool();
    settings.endGroup();

    if (m_enabled)
    {
        for (int x = 0; x < m_lastCommandTreeWidget->topLevelItemCount(); ++x)
        {
            m_lastCommandTreeWidget->topLevelItem(x)->setTextColor(0, QColor(m_dateColor));
        }
    }
    else
    {
        m_lastCommandTreeWidget->clear();
        m_lastTreeWidgetParent = NULL;
    }
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
    m_lastTreeWidgetParent = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void LastCommandDockWidget::treeWidgetContextMenuRequested(const QPoint &pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_lastCommandTreeWidget->mapToGlobal(m_lastCommandTreeWidget->pos()));
}


} //end namespace ito
