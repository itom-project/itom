/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "paramInputDialog.h"

#include <qicon.h>
#include <qlistwidget.h>
#include <qlineedit.h>
#include <qspinbox.h>
#include <qtimer.h>
#include "itomWidgets/doubleSpinBox.h"
#include "../global.h"

#if QT_VERSION < 0x050000
#include <qmessagebox.h>
#else
#include <QtWidgets/qmessagebox.h>
#endif

namespace ito {

//-------------------------------------------------------------------------------------
LineEditDelegate::LineEditDelegate(const double minVal, const double maxVal, const tParamType paramType, QObject *parent /*= 0*/) : 
    QStyledItemDelegate(parent),
    m_minVal(minVal),
    m_maxVal(maxVal),
    m_paramType(paramType)
{
}

//-------------------------------------------------------------------------------------
QWidget* LineEditDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem &/* option */, const QModelIndex &/* index */) const
{
    if ((m_paramType == intArray) || (m_paramType == charArray))
    {
        QSpinBox *spinbox = new QSpinBox(parent);
        spinbox->setMinimum(m_minVal);
        spinbox->setMaximum(m_maxVal);
        return spinbox;
    }
    else if (m_paramType == doubleArray)
    {
        //DoubleSpinBox is not directly derived from QDoubleSpinBox, therefore selectAll is not directly called for this by Qt.
        //We have to do it by a 0-ms timer, to verify that the widget is properly initialized.
        DoubleSpinBox *spinbox = new DoubleSpinBox(parent);
        spinbox->setMinimum(m_minVal);
        spinbox->setMaximum(m_maxVal);
        QTimer::singleShot(0, spinbox->spinBox(), SLOT(selectAll()));
        return spinbox;
    }

    return NULL;
}

//-------------------------------------------------------------------------------------
void LineEditDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    bool ok;

    if ((m_paramType == intArray) || (m_paramType == charArray))
    {
        int val = index.model()->data(index, Qt::EditRole).toInt(&ok);
        if (ok)
        {
            QSpinBox *spinbox = qobject_cast<QSpinBox*>(editor);
            spinbox->setValue(val);
        }
    }
    else if (m_paramType == doubleArray)
    {
        double val = index.model()->data(index, Qt::EditRole).toDouble(&ok);
        if (ok)
        {
            DoubleSpinBox *spinbox = qobject_cast<DoubleSpinBox*>(editor);
            spinbox->setValue(val);
        }
    }
}

//-------------------------------------------------------------------------------------
void LineEditDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    bool ok;

    if ((m_paramType == intArray) || (m_paramType == charArray))
    {
        QSpinBox *spinbox = qobject_cast<QSpinBox*>(editor);
        model->setData(index, spinbox->value(), Qt::EditRole);
    }
    else if (m_paramType == doubleArray)
    {
        DoubleSpinBox *spinbox = qobject_cast<DoubleSpinBox*>(editor);
        model->setData(index, spinbox->value(), Qt::EditRole);
    }
}

//-------------------------------------------------------------------------------------
void LineEditDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &/* index */) const
{
    editor->setGeometry(option.rect);
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
////////////////// List editor ///////////////
ParamInputDialog::ParamInputDialog(const QStringList &stringList, const ito::ParamMeta *meta, const tParamType paramType, QWidget *parent):
    QDialog(parent),
    m_RegExp(""),
    m_updating(false),
    m_lineEditDel(NULL),
    m_minSize(0),
    m_maxSize(std::numeric_limits<int>::max()),
    m_stepSize(1)
{
    ui.setupUi(this);

    if (paramType == intArray)
    {
        setWindowTitle(tr("IntArray"));
        m_minVal = std::numeric_limits<int>::min();
        m_maxVal = std::numeric_limits<int>::max();

        if (meta)
        {
            if (static_cast<const ito::IntArrayMeta*>(meta))
            {
                const ito::IntArrayMeta* dam = static_cast<const ito::IntArrayMeta*>(meta);
                m_minSize = dam->getNumMin();
                m_maxSize = dam->getNumMax();
                m_stepSize = dam->getNumStepSize();
                m_minVal = dam->getMin();
                m_maxVal = dam->getMax();
            }
        }
    }
    else if (paramType == doubleArray)
    {
        setWindowTitle(tr("DoubleArray"));
        m_minVal = -std::numeric_limits<double>::max();
        m_maxVal = std::numeric_limits<double>::max();

        if (meta)
        {
            if (static_cast<const ito::DoubleArrayMeta*>(meta))
            {
                const ito::DoubleArrayMeta* dam = static_cast<const ito::DoubleArrayMeta*>(meta);
                m_minSize = dam->getNumMin();
                m_maxSize = dam->getNumMax();
                m_stepSize = dam->getNumStepSize();
                m_minVal = dam->getMin();
                m_maxVal = dam->getMax();
            }
        }
    }
    else if (paramType == charArray)
    {
        setWindowTitle(tr("CharArray"));
        m_minVal = std::numeric_limits<char>::min();
        m_maxVal = std::numeric_limits<char>::max();

        if (meta)
        {
            if (static_cast<const ito::CharArrayMeta*>(meta))
            {
                const ito::CharArrayMeta* dam = static_cast<const ito::CharArrayMeta*>(meta);
                m_minSize = dam->getNumMin();
                m_maxSize = dam->getNumMax();
                m_stepSize = dam->getNumStepSize();
                m_minVal = dam->getMin();
                m_maxVal = dam->getMax();
            }
        }
    }

    QIcon upIcon(":/arrows/icons/up-32.png");
    QIcon downIcon(":/arrows/icons/down-32.png");
    QIcon minusIcon(":/arrows/icons/minus.png");
    QIcon plusIcon(":/arrows/icons/plus.png");
    ui.moveListItemUpButton->setIcon(upIcon);
    ui.moveListItemDownButton->setIcon(downIcon);
    ui.newListItemButton->setIcon(plusIcon);
    ui.deleteListItemButton->setIcon(minusIcon);

    m_lineEditDel = new LineEditDelegate(m_minVal, m_maxVal, paramType, ui.listWidget);
    ui.listWidget->setItemDelegate(m_lineEditDel);

    foreach(const QString &stringItem, stringList)
    {
        QListWidgetItem *item = new QListWidgetItem(stringItem);
        item->setFlags(item->flags() | Qt::ItemIsEditable);
        item->setSizeHint(QSize(item->sizeHint().width(), 20));
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

//----------------------------------------------------------------------------------------------------------------------------------
ParamInputDialog::~ParamInputDialog()
{
    DELETE_AND_SET_NULL(m_lineEditDel);
}

//-------------------------------------------------------------------------------------
QStringList ParamInputDialog::getStringList()
{
    QStringList stringlist;
    for (int i = 0; i < ui.listWidget->count(); ++i)
    {
        stringlist.append(ui.listWidget->item(i)->data(Qt::DisplayRole).toString());
    }

    return stringlist;
}

//-------------------------------------------------------------------------------------
void ParamInputDialog::setCurrentIndex(int idx)
{
    m_updating = true;
    ui.listWidget->setCurrentRow(idx);
    m_updating = false;
}

//-------------------------------------------------------------------------------------
void ParamInputDialog::on_newListItemButton_clicked()
{
    int row = ui.listWidget->currentRow() + 1;

    QListWidgetItem *item = new QListWidgetItem(m_newItemText);
    item->setFlags(item->flags() | Qt::ItemIsEditable);
    item->setSizeHint(QSize(item->sizeHint().width(), 20));
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
void ParamInputDialog::on_deleteListItemButton_clicked()
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
void ParamInputDialog::on_moveListItemUpButton_clicked()
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
void ParamInputDialog::on_moveListItemDownButton_clicked()
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
void ParamInputDialog::on_listWidget_currentRowChanged()
{
    updateEditor();
}

//-------------------------------------------------------------------------------------
void ParamInputDialog::setItemData(int role, const QVariant &v)
{
    QListWidgetItem *item = ui.listWidget->currentItem();
    bool reLayout = false;
    if ((role == Qt::EditRole && (v.toString().count(QLatin1Char('\n')) != item->data(role).toString().count(QLatin1Char('\n'))))
        || role == Qt::FontRole)
    {
        reLayout = true;
    }

    QVariant newValue = v;
    if (role == Qt::FontRole && newValue.type() == QVariant::Font)
    {
        QFont oldFont = ui.listWidget->font();
        QFont newFont = qvariant_cast<QFont>(newValue).resolve(oldFont);
        newValue = QVariant::fromValue(newFont);
        item->setData(role, QVariant()); // force the right font with the current resolve mask is set (item view bug)
    }

    item->setData(role, newValue);
    if (reLayout)
    {
        ui.listWidget->doItemsLayout();
    }
}

//-------------------------------------------------------------------------------------
QVariant ParamInputDialog::getItemData(int role) const
{
    return ui.listWidget->currentItem()->data(role);
}

//-------------------------------------------------------------------------------------
void ParamInputDialog::updateEditor()
{
    bool currentItemEnabled = false;
    bool moveRowUpEnabled = false;
    bool moveRowDownEnabled = false;

    QListWidgetItem *item = ui.listWidget->currentItem();
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
    ui.newListItemButton->setEnabled(ui.listWidget->count() < m_maxSize);
}

//-------------------------------------------------------------------------------------
void ParamInputDialog::on_buttonBox_clicked(QAbstractButton* btn)
{
    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::AcceptRole)
    {
        if ((ui.listWidget->count() <= m_maxSize) && (ui.listWidget->count() - m_minSize) % m_stepSize == 0)
        {
            accept(); //AcceptRole
        }
        else
        {
            QMessageBox msgBox;
            msgBox.setText(tr("The number of value does not match the step size"));
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
    }
    else
    {
        reject(); //close dialog with reject
    }
}

//-------------------------------------------------------------------------------------

void ParamInputDialog::on_listWidget_itemDoubleClicked(QListWidgetItem *item)
{
    ui.listWidget->editItem(item);
}

} //end namespace ito
