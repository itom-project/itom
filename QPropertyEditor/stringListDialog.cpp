/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "stringListDialog.h"

#include <qicon.h>
#include <qlistwidget.h>

////////////////// List editor ///////////////
StringListDialog::StringListDialog(const QStringList& stringList, QWidget* parent) :
    QDialog(parent), m_updating(false)
{
    ui.setupUi(this);

    QIcon upIcon(":/arrows/icons/up-32.png");
    QIcon downIcon(":/arrows/icons/down-32.png");
    QIcon minusIcon(":/arrows/icons/minus.png");
    QIcon plusIcon(":/arrows/icons/plus.png");
    ui.moveListItemUpButton->setIcon(upIcon);
    ui.moveListItemDownButton->setIcon(downIcon);
    ui.newListItemButton->setIcon(plusIcon);
    ui.deleteListItemButton->setIcon(minusIcon);

    foreach (const QString& stringItem, stringList)
    {
        QListWidgetItem* item = new QListWidgetItem(stringItem);
        item->setFlags(item->flags() | Qt::ItemIsEditable);
        ui.listWidget->addItem(item);
    }

    if (ui.listWidget->count() > 0)
    {
        ui.listWidget->setCurrentRow(0);
    }
    else
    {
        updateEditor();
    }
}

//-------------------------------------------------------------------------------------
QStringList StringListDialog::getStringList()
{
    QStringList stringlist;
    for (int i = 0; i < ui.listWidget->count(); ++i)
    {
        stringlist.append(ui.listWidget->item(i)->data(Qt::DisplayRole).toString());
    }

    return stringlist;
}

//-------------------------------------------------------------------------------------
void StringListDialog::setCurrentIndex(int idx)
{
    m_updating = true;
    ui.listWidget->setCurrentRow(idx);
    m_updating = false;
}

//-------------------------------------------------------------------------------------
void StringListDialog::on_newListItemButton_clicked()
{
    int row = ui.listWidget->currentRow() + 1;

    QListWidgetItem* item = new QListWidgetItem(m_newItemText);
    item->setFlags(item->flags() | Qt::ItemIsEditable);
    if (row < ui.listWidget->count())
    {
        ui.listWidget->insertItem(row, item);
    }
    else
    {
        ui.listWidget->addItem(item);
    }

    ui.listWidget->setCurrentItem(item);
    ui.listWidget->editItem(item);
}

//-------------------------------------------------------------------------------------
void StringListDialog::on_deleteListItemButton_clicked()
{
    int row = ui.listWidget->currentRow();

    if (row != -1)
    {
        delete ui.listWidget->takeItem(row);
    }

    if (row == ui.listWidget->count())
    {
        row--;
    }
    if (row < 0)
    {
        updateEditor();
    }
    else
    {
        ui.listWidget->setCurrentRow(row);
    }
}

//-------------------------------------------------------------------------------------
void StringListDialog::on_moveListItemUpButton_clicked()
{
    int row = ui.listWidget->currentRow();
    if (row <= 0)
    {
        return; // nothing to do
    }

    ui.listWidget->insertItem(row - 1, ui.listWidget->takeItem(row));
    ui.listWidget->setCurrentRow(row - 1);
}

//-------------------------------------------------------------------------------------
void StringListDialog::on_moveListItemDownButton_clicked()
{
    int row = ui.listWidget->currentRow();
    if (row == -1 || row == ui.listWidget->count() - 1)
    {
        return; // nothing to do
    }

    ui.listWidget->insertItem(row + 1, ui.listWidget->takeItem(row));
    ui.listWidget->setCurrentRow(row + 1);
}

//-------------------------------------------------------------------------------------
void StringListDialog::on_listWidget_currentRowChanged()
{
    updateEditor();
}

//-------------------------------------------------------------------------------------
void StringListDialog::setItemData(int role, const QVariant& v)
{
    QListWidgetItem* item = ui.listWidget->currentItem();
    bool reLayout = false;
    if ((role == Qt::EditRole &&
         (v.toString().count(QLatin1Char('\n')) !=
          item->data(role).toString().count(QLatin1Char('\n')))) ||
        role == Qt::FontRole)
    {
        reLayout = true;
    }

    QVariant newValue = v;
    if (role == Qt::FontRole && newValue.type() == QVariant::Font)
    {
        QFont oldFont = ui.listWidget->font();
        QFont newFont = qvariant_cast<QFont>(newValue).resolve(oldFont);
        newValue = QVariant::fromValue(newFont);
        item->setData(role, QVariant()); // force the right font with the current resolve mask is
                                         // set (item view bug)
    }

    item->setData(role, newValue);
    if (reLayout)
    {
        ui.listWidget->doItemsLayout();
    }
}

//-------------------------------------------------------------------------------------
QVariant StringListDialog::getItemData(int role) const
{
    return ui.listWidget->currentItem()->data(role);
}

//-------------------------------------------------------------------------------------
void StringListDialog::updateEditor()
{
    bool currentItemEnabled = false;

    bool moveRowUpEnabled = false;
    bool moveRowDownEnabled = false;

    QListWidgetItem* item = ui.listWidget->currentItem();
    if (item)
    {
        currentItemEnabled = true;
        int currentRow = ui.listWidget->currentRow();
        if (currentRow > 0)
        {
            moveRowUpEnabled = true;
        }

        if (currentRow < ui.listWidget->count() - 1)
        {
            moveRowDownEnabled = true;
        }
    }

    ui.moveListItemUpButton->setEnabled(moveRowUpEnabled);
    ui.moveListItemDownButton->setEnabled(moveRowDownEnabled);
    ui.deleteListItemButton->setEnabled(currentItemEnabled);
}

//-------------------------------------------------------------------------------------
void StringListDialog::on_listWidget_itemDoubleClicked(QListWidgetItem* item)
{
    ui.listWidget->editItem(item);
}
