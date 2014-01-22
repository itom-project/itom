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

#include "dialogLoadedPlugins.h"

#include "../AppManagement.h"
#include "../organizer/addInManager.h"
#include "../organizer/designerWidgetOrganizer.h"

#include <qfileinfo.h>

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogLoadedPlugins::DialogLoadedPlugins(QWidget *parent) :
    QDialog(parent),
    m_fileIconProvider(NULL)
{
    ui.setupUi(this);
    m_fileIconProvider = new QFileIconProvider();

    ui.cmdError->setIcon(QIcon(":/application/icons/dialog-error-4.png"));
    ui.cmdWarning->setIcon(QIcon(":/application/icons/dialog-warning-4.png"));
    ui.cmdMessage->setIcon(QIcon(":/application/icons/dialog-information-4.png"));
    ui.cmdIgnored->setIcon(QIcon(":/plugins/icons_m/ignored.png"));

    init();
    filter();

    ui.tree->expandAll();
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogLoadedPlugins::~DialogLoadedPlugins()
{
    DELETE_AND_SET_NULL(m_fileIconProvider);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogLoadedPlugins::init()
{
    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    if (AIM)
    {
        m_content = AIM->getPluginLoadStatus();
    }

    ito::DesignerWidgetOrganizer *dwo = qobject_cast<ito::DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo)
    {
        m_content.append(dwo->getPluginLoadStatus());
    }

    m_windowTitle = ui.groupBox_2->title();
    m_cmdMessage = ui.cmdMessage->text();
    m_cmdWarning = ui.cmdWarning->text();
    m_cmdError = ui.cmdError->text();
    m_cmdIgnored = ui.cmdIgnored->text();

    foreach (const PluginLoadStatus& item, m_content)
    {
        int overallStatus = 0;
        QFileInfo info(item.filename);
        const QPair<ito::tPluginLoadStatusFlag, QString> *message;
        QTreeWidgetItem *child = NULL;

        QTreeWidgetItem *plugin = new QTreeWidgetItem();
        plugin->setData(0, Qt::DisplayRole, info.fileName());
        plugin->setData(0, Qt::ToolTipRole, info.absoluteFilePath());
        plugin->setData(0, Qt::DecorationRole, m_fileIconProvider->icon(info));

        for (int i = 0; i < item.messages.size(); i++)
        {
            message = &(item.messages[i]);
            if (message->first == ito::plsfOk)
            {
//                overallStatus |= ito::retError * 2; //retOk is 0, that is bad, therefore use another value for retOk
                overallStatus |= 8; //retOk is 0, that is bad, therefore use another value for retOk
            }
            else
            {
                overallStatus |= message->first;
            }

            child = new QTreeWidgetItem(plugin);
            child->setData(0, Qt::DisplayRole, message->second);
            child->setData(0, Qt::ToolTipRole, message->second);
            if (message->first == ito::plsfOk)
            {
                child->setData(0, Qt::DecorationRole, QIcon(":/application/icons/dialog-information-4.png"));
                m_items.append(QPair<int,QTreeWidgetItem*>(8, child)); //plsfOk is 0, that is bad, therefore use another value for retOk
            }
            else if (message->first == ito::plsfWarning)
            {
                child->setData(0, Qt::DecorationRole, QIcon(":/application/icons/dialog-warning-4.png"));
                m_items.append(QPair<int,QTreeWidgetItem*>(ito::plsfWarning, child));
            }
            else if (message->first == ito::plsfIgnored)
            {
                child->setData(0, Qt::DecorationRole, QIcon(":/plugins/icons_m/ignored.png"));
                m_items.append(QPair<int,QTreeWidgetItem*>(ito::plsfIgnored, child));
            }
            else
            {
                child->setData(0, Qt::DecorationRole, QIcon(":/application/icons/dialog-error-4.png"));
                m_items.append(QPair<int,QTreeWidgetItem*>(ito::plsfError, child));
            }
        }
        
        m_items.append(QPair<int,QTreeWidgetItem*>(overallStatus, plugin));
        ui.tree->addTopLevelItem(plugin);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogLoadedPlugins::filter()
{
    int stateCount[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // we need 1: plsfWarning, 2: plsfError, 4: plsfIgnored, 8: plsfOk

    int flag = ui.cmdMessage->isChecked() * 8 + 
                ui.cmdWarning->isChecked() * ito::plsfWarning +
                ui.cmdError->isChecked() * ito::plsfError +
                ui.cmdIgnored->isChecked() * ito::plsfIgnored;

    bool filterEditNotEmpty = ui.filterEdit->text() != "";
    QString filterEditText = "*" + ui.filterEdit->text() + "*";
    QRegExp rx(filterEditText, Qt::CaseInsensitive, QRegExp::Wildcard);

    for (int i = m_items.size() - 1; i >= 0; i--)
    {
        bool hiddenItem = false;
        if (filterEditNotEmpty)
        {
            if (m_items[i].second->parent())
            {
                hiddenItem = m_items[i].second->parent()->isHidden();
            }
            else
            {
                hiddenItem = !rx.exactMatch(m_items[i].second->text(0));
            }
        }

        m_items[i].second->setHidden(!(m_items[i].first & flag) || hiddenItem);
        if (m_items[i].second->parent() && !m_items[i].second->isHidden())
        {
            stateCount[m_items[i].first]++;
        }
    }
    ui.groupBox_2->setTitle(QString("%1 (%2)").arg(m_windowTitle).arg(stateCount[8] + stateCount[1] + stateCount[2] + stateCount[4]));
    ui.cmdMessage->setText(QString("%1 (%2)").arg(m_cmdMessage).arg(stateCount[8]));
    ui.cmdWarning->setText(QString("%1 (%2)").arg(m_cmdWarning).arg(stateCount[1]));
    ui.cmdError->setText(QString("%1 (%2)").arg(m_cmdError).arg(stateCount[2]));
    ui.cmdIgnored->setText(QString("%1 (%2)").arg(m_cmdIgnored).arg(stateCount[4]));
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogLoadedPlugins::on_tree_itemSelectionChanged()
{
    QList<QTreeWidgetItem*> items = ui.tree->selectedItems();
    if (items.size() >= 1)
    {
        ui.lblText->setText(items[0]->data(0, Qt::ToolTipRole).toString());
    }
    else
    {
        ui.lblText->setText("");
    }
}

} //end namespace ito