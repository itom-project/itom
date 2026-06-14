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

#include "dialogIconBrowser.h"
#include "../global.h"
#include "helper/guiHelper.h"

#include <qclipboard.h>
#include <qboxlayout.h>
#include <qresource.h>

#include <QApplication>
#include <QDirIterator>
#include <QtConcurrent/qtconcurrentrun.h>
#include <qlist.h>
#include <qtreewidget.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
DialogIconBrowser::DialogIconBrowser(QWidget *parent) :
    QDialog(parent)
{
    ui.setupUi(this);

    ui.treeWidget->setItemsExpandable(true);
    ui.treeWidget->setExpandsOnDoubleClick(true);
    ui.treeWidget->expandAll();
    int size = 20 * GuiHelper::screenDpiFactor();
    ui.treeWidget->setIconSize(QSize(size, size));
    ui.treeWidget->sortItems(0, Qt::AscendingOrder);

    ui.txtCurrentName->setText(tr("loading..."));

    setWindowIcon(QIcon(QString(":/editor/icons/iconList.png")));
    setWindowTitle(tr("Icon Browser"));

    ui.cancelButton->setEnabled(false);
    ui.pushButtonClipboard->setEnabled(false);
    ui.pushButtonInsert->setEnabled(false);
    ui.treeWidget->setEnabled(false);
    ui.txtFilter->setEnabled(false);

    connect(&m_loadWatcher, SIGNAL(finished()), this, SLOT(loadFinished()));

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    QFuture<QList<QTreeWidgetItem*> > f1 = QtConcurrent::run(&DialogIconBrowser::loadIcons, this);
#else
    QFuture<QList<QTreeWidgetItem*> > f1 = QtConcurrent::run(this, &DialogIconBrowser::loadIcons);
#endif
    m_loadWatcher.setFuture(f1);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogIconBrowser::~DialogIconBrowser()
{
    if (m_loadWatcher.isRunning())
    {
        //a previous load of icons QtConcurrent::run is still running, wait for it to be finished
        m_loadWatcher.waitForFinished();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QTreeWidgetItem*> DialogIconBrowser::loadIcons()
{
    QStringList list;
    QStringList sublist;
    QList<QTreeWidgetItem *> items;

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
            QIcon thisIcon(subIT.next());
            if (subIT.fileName().endsWith(".png") || subIT.fileName().endsWith(".svg"))
            {
                QTreeWidgetItem *icon = new QTreeWidgetItem(QTreeWidgetItem::DontShowIndicatorWhenChildless);
                icon->setIcon(0, thisIcon);
                icon->setText(0, subIT.filePath());
                items.last()->addChild(icon);
            }
        }
    }

    return items;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! this method is called if the async load mechanism of the icons has been finished
void DialogIconBrowser::loadFinished()
{
    ui.treeWidget->setEnabled(true);
    ui.txtFilter->setEnabled(true);
    ui.cancelButton->setEnabled(true);
    ui.txtCurrentName->setText("");

    //adding the elements has to be done in the main GUI thread
    ui.treeWidget->addTopLevelItems(m_loadWatcher.future().result());

    ui.treeWidget->expandAll();
    ui.treeWidget->sortItems(0, Qt::AscendingOrder);

    ui.txtFilter->setFocus();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogIconBrowser::on_treeWidget_currentItemChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    bool enable = current->parent() != NULL;
    ui.pushButtonClipboard->setEnabled(enable);
    ui.pushButtonInsert->setEnabled(enable);
    ui.txtCurrentName->setText(current ? current->text(0) : "");
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

//----------------------------------------------------------------------------------------------------------------------------------
void DialogIconBrowser::on_txtFilter_textChanged(const QString &text)
{
    int numTopLevels = ui.treeWidget->topLevelItemCount();
    bool showAtLeastOne;
    QTreeWidgetItem *toplevel;
    QString childText;
    QTreeWidgetItem *child;

    for (int top = 0; top < numTopLevels; ++top)
    {
        toplevel = ui.treeWidget->topLevelItem(top);
        showAtLeastOne = false;

        if (toplevel)
        {
            for (int i = 0; i < toplevel->childCount(); ++i)
            {
                child = toplevel->child(i);
                childText = child->text(0);
                if (childText.contains(text, Qt::CaseInsensitive))
                {
                    child->setHidden(false);
                    showAtLeastOne = true;
                }
                else
                {
                    child->setHidden(true);
                }
            }

            toplevel->setHidden(!showAtLeastOne);
        }
    }
}

} //end namespace ito
