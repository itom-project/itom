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

#include "iconBrowserDialog.h"
#include "../global.h"

#include <qclipboard.h>
#include <qboxlayout.h>
#include <qresource.h>

#include <QApplication>
#include <QDirIterator>
#include <QClipboard>

//----------------------------------------------------------------------------------------------------------------------------------
IconBrowserDialog::IconBrowserDialog(QWidget *parent) :  QDialog(parent),  
    m_pTreeWidget(NULL)    
{
    
    m_pTreeWidget = new IconRescourcesTreeView(this);
    QStringList list;
    QStringList sublist;
    list.clear();

    int longestName = 0;
    int lineCnt = 0;

    QList<QTreeWidgetItem *> items;

    //QDirIterator it(":", QDirIterator::Subdirectories);

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
            if(subIT.fileName().contains(".png"))
            {
                QTreeWidgetItem *icon = new QTreeWidgetItem(QTreeWidgetItem::DontShowIndicatorWhenChildless);
                icon->setIcon(0, thisIcon);
                icon->setText(0, subIT.filePath());
                items.last()->addChild(icon);
                if(subIT.filePath().length() > longestName)
                {
                    longestName = subIT.filePath().length();
                }
                lineCnt++;
            }
        }  
    }

    longestName += 20;
    longestName *= 5;
    if(longestName > 420)
        longestName = 420;

    lineCnt *= 10;
    if(lineCnt > 500)
        lineCnt = 500;

    m_pTreeWidget->headerItem()->setHidden(true);

    //m_pTreeWidget->setGeometry(0, 0, 300, 400);
    m_pTreeWidget->setItemsExpandable(true);
    m_pTreeWidget->addTopLevelItems(items);
    m_pTreeWidget->setItemsExpandable(true);
    m_pTreeWidget->setExpandsOnDoubleClick(true);
    //m_pTreeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
    m_pTreeWidget->expandAll();
    m_pTreeWidget->setIconSize(QSize(16, 16));

    m_pTreeWidget->sortItems(0, Qt::AscendingOrder);
    //m_pTreeWidget->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

    //m_pTreeWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

   // m_pTreeWidget->add
    
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->setContentsMargins(0, 0, 0, 2);
    mainLayout->setSpacing(1);
    mainLayout->addWidget(m_pTreeWidget);
    setLayout(mainLayout);

    QRect curGeo = geometry();
    curGeo.setCoords(50, 50, longestName, lineCnt);
    setGeometry(curGeo);

    this->setWindowIcon(QIcon(QString(":/editor/icons/iconList.png")));
    this->setWindowTitle("Icon Browser");

    this->setWhatsThis("itom resource file browser\nDouble-Click icon to copy icon path to the clipboard\nand close this window.");

    connect(m_pTreeWidget, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(copyCurrentName()));
}

//----------------------------------------------------------------------------------------------------------------------------------
IconBrowserDialog::~IconBrowserDialog()
{
    disconnect(m_pTreeWidget, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(copyCurrentName()));
    DELETE_AND_SET_NULL(m_pTreeWidget);
}
//----------------------------------------------------------------------------------------------------------------------------------
void IconBrowserDialog::copyCurrentName()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(m_pTreeWidget->currentItem()->text(0), QClipboard::Clipboard);
    this->close();
    
}


