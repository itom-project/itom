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

#include "dialogIconBrowser.h"
#include "../global.h"

#include <qclipboard.h>
#include <qboxlayout.h>
#include <qresource.h>

#include <QApplication>
#include <QDirIterator>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
DialogIconBrowser::DialogIconBrowser(QWidget *parent) :
    QDialog(parent)
{
    ui.setupUi(this);

    QStringList list;
    QStringList sublist;
    QList<QTreeWidgetItem *> items;

//    list.clear();

    QDirIterator it(":", QDirIterator::NoIteratorFlags);
    while (it.hasNext())
    {
        list.clear();
        QString curDir(it.next());
        list.append(curDir);

        items.append(new QTreeWidgetItem(list, QTreeWidgetItem::DontShowIndicatorWhenChildless));
        QDirIterator subIT(curDir, QDirIterator::Subdirectories);
        while (subIT.hasNext())
        {
            //sublist.append(subIT.next());
            //QResource temp(subIT.filePath());
            QIcon thisIcon(subIT.next());
            if (subIT.fileName().contains(".png"))
            {
                QTreeWidgetItem *icon = new QTreeWidgetItem(QTreeWidgetItem::DontShowIndicatorWhenChildless);
                icon->setIcon(0, thisIcon);
                icon->setText(0, subIT.filePath());
                items.last()->addChild(icon);
            }
        }
    }

    ui.treeWidget->addTopLevelItems(items);
    ui.treeWidget->setItemsExpandable(true);
    ui.treeWidget->setExpandsOnDoubleClick(true);
    ui.treeWidget->expandAll();
    ui.treeWidget->setIconSize(QSize(16, 16));
    ui.treeWidget->sortItems(0, Qt::AscendingOrder);

    this->setWindowIcon(QIcon(QString(":/editor/icons/iconList.png")));
    this->setWindowTitle(tr("Icon Browser"));
//    this->setWhatsThis(tr("itom resource file browser\nDouble-Click icon to copy icon path to the clipboard\nand close this window."));
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogIconBrowser::~DialogIconBrowser()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogIconBrowser::on_treeWidget_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    bool enable = current->parent() != NULL;
    ui.pushButtonClipboard->setEnabled(enable);
    ui.pushButtonInsert->setEnabled(enable);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogIconBrowser::on_pushButtonClipboard_clicked(bool value)
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(ui.treeWidget->currentItem()->text(0), QClipboard::Clipboard);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogIconBrowser::on_pushButtonInsert_clicked(bool value)
{
    emit sendIconBrowserText(ui.treeWidget->currentItem()->text(0));
}

} //end namespace ito
