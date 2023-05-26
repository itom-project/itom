/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include "../python/pythonEngineInc.h"

#include "pythonMessageDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "consoleWidget.h"

#include <qheaderview.h>
#include <qsettings.h>
#include <qmimedata.h>
#include <QAbstractScrollArea>
#include <qscrollbar.h>


namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
QStringList PythonMessageTreeWidget::mimeTypes() const
{
    QStringList types = QTreeWidget::mimeTypes();

    if (types.contains("text/plain") == false)
    {
        types.append("text/plain");
    }

    return types;
}

//----------------------------------------------------------------------------------------------------------------------------------
QMimeData * PythonMessageTreeWidget::mimeData(const QList<QTreeWidgetItem *> items) const
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
        texts.append(roleDataMap[0].toString());
    }

    mimeData->setData("text/plain", texts.join("\n").toLatin1());
    return mimeData;
}

//----------------------------------------------------------------------------------------------------------------------------------
PythonMessageDockWidget::PythonMessageDockWidget(const QString &title, const QString &objName, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_pythonMessageTreeWidget(NULL),
    m_pActClearList(NULL),
    m_pythonTreeWidgetParent(NULL),
    m_message("")
{
    AbstractDockWidget::init();

    m_pythonMessageTreeWidget = new PythonMessageTreeWidget(this);
    m_pythonMessageTreeWidget->installEventFilter(this);
    setContentWidget(m_pythonMessageTreeWidget);
    m_pythonMessageTreeWidget->header()->hide();
    m_pythonMessageTreeWidget->setDragDropMode(QAbstractItemView::DragOnly);
    m_pythonMessageTreeWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    connect(m_pythonMessageTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));
    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));

    m_pythonMessageTreeWidget->setItemsExpandable(true);
    m_pythonMessageTreeWidget->setExpandsOnDoubleClick(false);
    m_pythonMessageTreeWidget->expandAll();
    m_pythonMessageTreeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_pythonMessageTreeWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(treeWidgetContextMenuRequested(const QPoint &)));
}

//----------------------------------------------------------------------------------------------------------------------------------
PythonMessageDockWidget::~PythonMessageDockWidget()
{
    disconnect(m_pythonMessageTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(itemDoubleClicked(QTreeWidgetItem*, int)));
    disconnect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));

    DELETE_AND_SET_NULL(m_pythonMessageTreeWidget);
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::createActions()
{
    m_pActClearList = new ShortcutAction(QIcon(":/editor/icons/editDelete.png"), tr("Clear List"), this);
    m_pActClearList->connectTrigger(this, SLOT(mnuClearList()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
    m_pContextMenu->addAction(m_pActClearList->action());
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::createToolBars()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::updateActions()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::addPythonMessage(QString cmd)
{
    if (m_enabled)
    {
        if (cmd.compare("\n") == 0)
        {
            QDate date(QDate::currentDate());
            QTime time(QTime::currentTime());
            QString strDate = date.toString(Qt::ISODate) + " " + time.toString(Qt::ISODate);
            QTreeWidgetItem *parentItem;
            parentItem = new QTreeWidgetItem(m_pythonMessageTreeWidget);
            parentItem->setText(0, strDate);
            parentItem->setTextColor(0, QColor(m_dateColor));
            parentItem->setFlags(Qt::ItemIsEnabled);
            m_pythonMessageTreeWidget->addTopLevelItem(parentItem);
            m_pythonTreeWidgetParent = parentItem;

            QTreeWidgetItem *childItem = new QTreeWidgetItem(m_pythonTreeWidgetParent);
            childItem->setText(0, m_message);
            m_pythonTreeWidgetParent->addChild(childItem);
            m_pythonTreeWidgetParent->setExpanded(true);

            if (m_pythonMessageTreeWidget->verticalScrollBar()->value() == m_pythonMessageTreeWidget->verticalScrollBar()->maximum())
            {
                m_pythonMessageTreeWidget->scrollToItem(childItem);
            }

            m_message = "";
        }
        else
        {
            m_message = m_message + cmd;// .trimmed();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::propertiesChanged()
{
/*    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("itomPythonMessageDockWidget");
    m_enabled = settings.value("lastCommandEnabled", "true").toBool();
    m_dateColor = settings.value("lastCommandDateColor", "green").toString();
    m_doubleCommand = settings.value("lastCommandHideDoubleCommand", "false").toBool();
    settings.endGroup();

    if (m_enabled)
    {
        for (int x = 0; x < m_pythonMessageTreeWidget->topLevelItemCount(); ++x)
        {
            m_pythonMessageTreeWidget->topLevelItem(x)->setTextColor(0, QColor(m_dateColor));
        }
    }
    else
    {
        m_pythonMessageTreeWidget->clear();
        m_pythonMessageTreeWidget = NULL;
    }*/
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::itemDoubleClicked(QTreeWidgetItem *item, int column)
{
    if (item->parent() != NULL)
    {
        emit runPythonCommand(item->text(0));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::mnuClearList()
{
    m_pythonMessageTreeWidget->clear();
    m_pythonTreeWidgetParent = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonMessageDockWidget::treeWidgetContextMenuRequested(const QPoint &pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_pythonMessageTreeWidget->mapToGlobal(m_pythonMessageTreeWidget->pos()));
}


} //end namespace ito
