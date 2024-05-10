/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

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

#include "paramInputDialog.h"

#include "../global.h"
#include "itomWidgets/doubleSpinBox.h"
#include "../helper/compatHelper.h"
#include "AddInManager/paramHelper.h"

#include <QtWidgets/qmessagebox.h>
#include <qcombobox.h>
#include <qicon.h>
#include <qlineedit.h>
#include <qlabel.h>
#include <qlistwidget.h>
#include <qsharedpointer.h>
#include <qspinbox.h>

#include <qtimer.h>

namespace ito
{

//-------------------------------------------------------------------------------------
LineEditDelegate::LineEditDelegate(const ito::ParamMeta *meta, int paramType,
                                   QObject *parent /*= 0*/)
    : QStyledItemDelegate(parent), m_meta(nullptr), m_paramType(paramType)
{
    switch (m_paramType)
    {
    case ParamBase::CharArray: {
        auto cam = dynamic_cast<const ito::CharArrayMeta *>(meta);

        if (cam)
        {
            m_meta = QSharedPointer<ito::ParamMeta>(new ito::CharArrayMeta(*cam));
        }
    }
    break;
    case ParamBase::IntArray: {
        auto cam = dynamic_cast<const ito::IntArrayMeta *>(meta);

        if (cam)
        {
            m_meta = QSharedPointer<ito::ParamMeta>(new ito::IntArrayMeta(*cam));
        }
    }
    break;
    case ParamBase::DoubleArray: {
        auto cam = dynamic_cast<const ito::IntArrayMeta *>(meta);

        if (cam)
        {
            m_meta = QSharedPointer<ito::ParamMeta>(new ito::IntArrayMeta(*cam));
        }
    }
    break;
    case ParamBase::ComplexArray: {
        // = nullptr is only possible since Qt 5.8
        m_meta = QSharedPointer<ito::ParamMeta>();
    }
    break;
    case ParamBase::StringList: {
        auto cam = dynamic_cast<const ito::StringListMeta *>(meta);

        if (cam)
        {
            m_meta = QSharedPointer<ito::ParamMeta>(new ito::StringListMeta(*cam));
        }
    }
    break;
    }
}

//-------------------------------------------------------------------------------------
QWidget *LineEditDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem & /* option */,
                                        const QModelIndex & /* index */) const
{
    if (m_paramType == ParamBase::CharArray)
    {
        QSpinBox *spinbox = new QSpinBox(parent);
        auto meta = dynamic_cast<const CharMeta *>(m_meta.data());

        if (meta)
        {
            spinbox->setMinimum(meta->getMin());
            spinbox->setMaximum(meta->getMax());
            spinbox->setSingleStep(meta->getStepSize());
        }
        else
        {
            spinbox->setMinimum(std::numeric_limits<char>::min());
            spinbox->setMaximum(std::numeric_limits<char>::max());
        }

        return spinbox;
    }
    else if (m_paramType == ParamBase::IntArray)
    {
        QSpinBox *spinbox = new QSpinBox(parent);
        auto meta = dynamic_cast<const IntMeta *>(m_meta.data());

        if (meta)
        {
            spinbox->setMinimum(meta->getMin());
            spinbox->setMaximum(meta->getMax());
            spinbox->setSingleStep(meta->getStepSize());
        }
        else
        {
            spinbox->setMinimum(std::numeric_limits<int>::min());
            spinbox->setMaximum(std::numeric_limits<int>::max());
        }

        return spinbox;
    }
    else if (m_paramType == ParamBase::DoubleArray)
    {
        // DoubleSpinBox is not directly derived from QDoubleSpinBox, therefore selectAll is not directly called for
        // this by Qt. We have to do it by a 0-ms timer, to verify that the widget is properly initialized.
        DoubleSpinBox *spinbox = new DoubleSpinBox(parent);
        spinbox->setFocusProxy(spinbox->spinBox());
        auto meta = dynamic_cast<const DoubleMeta *>(m_meta.data());

        if (meta)
        {
            spinbox->setMinimum(meta->getMin());
            spinbox->setMaximum(meta->getMax());

            if (meta->getStepSize() != 0.0)
            {
                spinbox->setSingleStep(meta->getStepSize());
            }
        }
        else
        {
            spinbox->setMinimum(-std::numeric_limits<double>::max());
            spinbox->setMaximum(std::numeric_limits<double>::max());
        }

        QTimer::singleShot(0, spinbox->spinBox(), &QDoubleSpinBox::selectAll);

        return spinbox;
    }
    else if (m_paramType == ParamBase::ComplexArray)
    {
        // DoubleSpinBox is not directly derived from QDoubleSpinBox, therefore selectAll is not directly called for
        // this by Qt. We have to do it by a 0-ms timer, to verify that the widget is properly initialized.
        DoubleSpinBox *spinboxReal = new DoubleSpinBox(parent);
        spinboxReal->setFocusProxy(spinboxReal->spinBox());
        spinboxReal->setMinimum(-std::numeric_limits<double>::max());
        spinboxReal->setMaximum(std::numeric_limits<double>::max());

        DoubleSpinBox *spinboxImag = new DoubleSpinBox(parent);
        spinboxImag->setFocusProxy(spinboxImag->spinBox());
        spinboxImag->setMinimum(-std::numeric_limits<double>::max());
        spinboxImag->setMaximum(std::numeric_limits<double>::max());
        spinboxImag->setSuffix("i");

        QLabel *label = new QLabel(parent);
        label->setText("+");

        //QTimer::singleShot(0, spinboxReal->spinBox(), &QDoubleSpinBox::selectAll);

        QWidget *widget = new QWidget(parent);
        QHBoxLayout *layout = new QHBoxLayout();
        layout->addWidget(spinboxReal);
        layout->addWidget(label);
        layout->addWidget(spinboxImag);
        widget->setLayout(layout);
        widget->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(1);
        layout->setContentsMargins(0, 0, 0, 0);

        return widget;
    }
    else if (m_paramType == ParamBase::StringList)
    {
        auto meta = dynamic_cast<const StringMeta *>(m_meta.data());
        bool lineedit = !meta || (meta->getStringType() != ito::StringMeta::String) || (meta->getLen() <= 0);

        if (lineedit)
        {
            QLineEdit *lineEdit = new QLineEdit(parent);

            if (meta && meta->getLen() > 0)
            {
                switch (meta->getStringType())
                {
                case ito::StringMeta::String:
                    lineEdit->setValidator(nullptr);
                    break;
                case ito::StringMeta::Wildcard: {
                    QRegularExpression regexp(CompatHelper::wildcardToRegularExpression(meta->getString(0)));
                    lineEdit->setValidator(new QRegularExpressionValidator(regexp, lineEdit));
                    break;
                }
                case ito::StringMeta::RegExp: {
                    QRegularExpression regexp(QLatin1String(meta->getString(0)));
                    lineEdit->setValidator(new QRegularExpressionValidator(regexp, lineEdit));
                    break;
                }
                }
            }
            else
            {
                lineEdit->setValidator(nullptr);
            }

            return lineEdit;
        }
        else
        {
            QComboBox *comboBox = new QComboBox(parent);

            for (int i = 0; i < meta->getLen(); ++i)
            {
                comboBox->addItem(meta->getString(i));
            }

            return comboBox;
        }
    }

    return nullptr;
}

//-------------------------------------------------------------------------------------
void LineEditDelegate::setEditorData(QWidget *editor, const QModelIndex &index) const
{
    bool ok;

    if ((m_paramType == ParamBase::IntArray) || (m_paramType == ParamBase::CharArray))
    {
        int val = index.model()->data(index, Qt::UserRole).toInt(&ok);
        if (ok)
        {
            QSpinBox *spinbox = qobject_cast<QSpinBox *>(editor);
            spinbox->setValue(val);
        }
    }
    else if (m_paramType == ParamBase::DoubleArray)
    {
        double val = index.model()->data(index, Qt::UserRole).toDouble(&ok);
        if (ok)
        {
            DoubleSpinBox *spinbox = qobject_cast<DoubleSpinBox *>(editor);
            spinbox->setValue(val);
        }
    }
    else if (m_paramType == ParamBase::StringList)
    {
        QByteArray val = index.model()->data(index, Qt::UserRole).toByteArray();
        QLineEdit *lineEdit = qobject_cast<QLineEdit *>(editor);
        QComboBox *comboBox = qobject_cast<QComboBox *>(editor);

        if (lineEdit)
        {
            lineEdit->setText(val);
        }
        else
        {
            comboBox->setCurrentText(val);
        }
    }
    else if (m_paramType == ParamBase::ComplexArray)
    {
        ito::complex128 value = index.model()->data(index, Qt::UserRole).value<ito::complex128>();
        auto layout = editor->layout();

        if (layout && layout->count() > 2)
        {
            DoubleSpinBox *spinboxReal = qobject_cast<DoubleSpinBox *>(layout->itemAt(0)->widget());
            DoubleSpinBox *spinboxImag = qobject_cast<DoubleSpinBox *>(layout->itemAt(2)->widget());

            if (spinboxReal && spinboxImag)
            {
                spinboxReal->setValue(value.real());
                spinboxImag->setValue(value.imag());
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void LineEditDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
    if ((m_paramType == ParamBase::IntArray) || (m_paramType == ParamBase::CharArray))
    {
        QSpinBox *spinbox = qobject_cast<QSpinBox *>(editor);
        model->setData(index, spinbox->value(), Qt::UserRole);
        model->setData(index, QString::number(spinbox->value()), Qt::DisplayRole);
    }
    else if (m_paramType == ParamBase::DoubleArray)
    {
        DoubleSpinBox *spinbox = qobject_cast<DoubleSpinBox *>(editor);
        model->setData(index, spinbox->value(), Qt::UserRole);
        model->setData(index, QString::number(spinbox->value(), 'g', 4), Qt::DisplayRole);
    }
    else if (m_paramType == ParamBase::ComplexArray)
    {
        auto layout = editor->layout();

        if (layout && layout->count() > 2)
        {
            DoubleSpinBox *spinboxReal = qobject_cast<DoubleSpinBox *>(layout->itemAt(0)->widget());
            DoubleSpinBox *spinboxImag = qobject_cast<DoubleSpinBox *>(layout->itemAt(2)->widget());

            if (spinboxReal && spinboxImag)
            {
                auto cmplx = ito::complex128(spinboxReal->value(), spinboxImag->value());
                auto value = QVariant::fromValue<ito::complex128>(cmplx);
                model->setData(index, value, Qt::UserRole);
                QString s;

                if (cmplx.imag() >= 0)
                {
                    s = QString::number(cmplx.real(), 'g', 4) + "+" + QString::number(cmplx.imag(), 'g', 4) + "i";
                }
                else
                {
                    s = QString::number(cmplx.real(), 'g', 4) + "-" + QString::number(std::abs(cmplx.imag()), 'g', 4) + "i";
                }

                model->setData(index, s, Qt::DisplayRole);
            }
        }
    }
    else if (m_paramType == ParamBase::StringList)
    {
        QLineEdit *lineEdit = qobject_cast<QLineEdit *>(editor);
        QComboBox *comboBox = qobject_cast<QComboBox *>(editor);

        if (lineEdit)
        {
            model->setData(index, lineEdit->text(), Qt::UserRole);
        }
        else
        {
            model->setData(index, comboBox->currentText(), Qt::UserRole);
        }

        model->setData(index, model->data(index, Qt::UserRole).toString(), Qt::DisplayRole);
    }
}

//-------------------------------------------------------------------------------------
void LineEditDelegate::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option,
                                            const QModelIndex & /* index */) const
{
    editor->setGeometry(option.rect);
}

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
////////////////// List editor ///////////////
ParamInputDialog::ParamInputDialog(const Param &param, QWidget *parent /*= nullptr*/) :
    QDialog(parent),
    m_RegExp(""),
    m_updating(false),
    m_lineEditDel(nullptr),
    m_minSize(0),
    m_maxSize(std::numeric_limits<int>::max()),
    m_stepSize(1),
    m_param(param)
{
    assert((param.getType() == ParamBase::CharArray)
        || (param.getType() == ParamBase::IntArray)
        || (param.getType() == ParamBase::DoubleArray)
        || (param.getType() == ParamBase::ComplexArray)
        || (param.getType() == ParamBase::StringList));

    ui.setupUi(this);

    QIcon upIcon(":/arrows/icons/up-32.png");
    QIcon downIcon(":/arrows/icons/down-32.png");
    QIcon minusIcon(":/arrows/icons/minus.png");
    QIcon plusIcon(":/arrows/icons/plus.png");
    ui.moveListItemUpButton->setIcon(upIcon);
    ui.moveListItemDownButton->setIcon(downIcon);
    ui.newListItemButton->setIcon(plusIcon);
    ui.deleteListItemButton->setIcon(minusIcon);

    m_lineEditDel = new LineEditDelegate(param.getMeta(), param.getType(), ui.listWidget);
    ui.listWidget->setItemDelegate(m_lineEditDel);

    foreach (const auto value, parseListItems(param))
    {
        QListWidgetItem *item = new QListWidgetItem(value.first);
        item->setData(Qt::UserRole, value.second);
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

    updateButtonState();
}

//-------------------------------------------------------------------------------------
ParamInputDialog::~ParamInputDialog()
{
    DELETE_AND_SET_NULL(m_lineEditDel);
}

//-------------------------------------------------------------------------------------
QList<QPair<QString, QVariant>> ParamInputDialog::parseListItems(const ito::Param &param) const
{
    QList<QPair<QString, QVariant>> items;
    int num = param.getLen();

    switch (param.getType())
    {
    case ito::ParamBase::CharArray:
    {
        auto vals = param.getVal<const char*>();
        for (int i = 0; i < num; ++i)
        {
            items.append(
                qMakePair(
                    QString::number(vals[i]),
                    vals[i]
                )
            );
        }
        break;
    }
    case ito::ParamBase::IntArray:
    {
        auto vals = param.getVal<const int*>();
        for (int i = 0; i < num; ++i)
        {
            items.append(
                qMakePair(
                    QString::number(vals[i]),
                    vals[i]
                )
            );
        }
        break;
    }
    case ito::ParamBase::DoubleArray:
    {
        auto vals = param.getVal<const double*>();
        for (int i = 0; i < num; ++i)
        {
            items.append(
                qMakePair(
                    QString::number(vals[i], 'g', 4),
                    vals[i]
                )
            );
        }
        break;
    }
    case ito::ParamBase::ComplexArray:
    {
        auto vals = param.getVal<const ito::complex128*>();
        QString s;

        for (int i = 0; i < num; ++i)
        {
            if (vals[i].imag() >= 0)
            {
                s = QString::number(vals[i].real(), 'g', 4) + "+" + QString::number(vals[i].imag(), 'g', 4) + "i";
            }
            else
            {
                s = QString::number(vals[i].real(), 'g', 4) + "-" + QString::number(std::abs(vals[i].imag()), 'g', 4) + "i";
            }

            items.append(
                qMakePair(
                    s,
                    QVariant::fromValue<ito::complex128>(vals[i])
                )
            );

        }
        break;
    }
    case ito::ParamBase::StringList:
    {
        auto vals = param.getVal<const ito::ByteArray*>();

        for (int i = 0; i < num; ++i)
        {
            const char* s = vals[i].data();
            items.append(
                qMakePair(
                    QLatin1String(s),
                    QByteArray(s)
                )
            );
        }
        break;
    }
    }

    return items;
}

//-------------------------------------------------------------------------------------
void ParamInputDialog::updateButtonState()
{
    ito::ListMeta* listMeta = dynamic_cast<ito::ListMeta*>(m_param.getMeta());

    if (listMeta)
    {
        ui.newListItemButton->setEnabled(
            ui.listWidget->count() < listMeta->getNumMax()
        );

        ui.deleteListItemButton->setEnabled(
            ui.listWidget->count() > listMeta->getNumMin()
        );
    }
}

//-------------------------------------------------------------------------------------
Param ParamInputDialog::getItems(RetVal &retValue) const
{
    Param result = m_param;
    int num = ui.listWidget->count();
    bool ok;

    switch (m_param.getType())
    {
    case ParamBase::CharArray:
    {
        char *arr = new char[num];
        int val;

        for (int i = 0; i < num; ++i)
        {
            val = ui.listWidget->item(i)->data(Qt::UserRole).toInt(&ok);

            if (!ok || val < std::numeric_limits<char>::min() || val > std::numeric_limits<char>::max())
            {
                retValue += ito::RetVal::format(
                    ito::retError,
                    0,
                    tr("The %i.th value in the list cannot be parsed to a char value.").toLatin1().data(),
                    i);
            }
            else
            {
                arr[i] = (char)val;
            }
        }

        result.setVal<char*>(arr, num);
        delete[] arr;
    }
    break;

    case ParamBase::IntArray:
    {
        int *arr = new int[num];
        int val;

        for (int i = 0; i < num; ++i)
        {
            val = ui.listWidget->item(i)->data(Qt::UserRole).toInt(&ok);

            if (!ok)
            {
                retValue += ito::RetVal::format(
                    ito::retError,
                    0,
                    tr("The %i.th value in the list cannot be parsed to an integer value.").toLatin1().data(),
                    i);
            }
            else
            {
                arr[i] = val;
            }
        }

        result.setVal<int*>(arr, num);
        delete[] arr;
    }
    break;

    case ParamBase::DoubleArray:
    {
        double *arr = new double[num];
        double val;

        for (int i = 0; i < num; ++i)
        {
            val = ui.listWidget->item(i)->data(Qt::UserRole).toDouble(&ok);

            if (!ok)
            {
                retValue += ito::RetVal::format(
                    ito::retError,
                    0,
                    tr("The %i.th value in the list cannot be parsed to a double value.").toLatin1().data(),
                    i);
            }
            else
            {
                arr[i] = val;
            }
        }

        result.setVal<double*>(arr, num);
        delete[] arr;
    }
    break;

    case ParamBase::StringList:
    {
        ByteArray *arr = new ByteArray[num];
        QByteArray val;

        for (int i = 0; i < num; ++i)
        {
            val = ui.listWidget->item(i)->data(Qt::UserRole).toByteArray();
            arr[i] = ByteArray(val.constData());
        }

        result.setVal<ByteArray*>(arr, num);
        delete[] arr;
    }
    break;

    case ParamBase::ComplexArray:
    {
        auto arr = new ito::complex128[num];
        ito::complex128 val;

        for (int i = 0; i < num; ++i)
        {
            val = ui.listWidget->item(i)->data(Qt::UserRole).value<ito::complex128>();
            arr[i] = ito::complex128(val);
        }

        result.setVal<ito::complex128*>(arr, num);
        delete[] arr;
    }
    break;
    }

    if (!retValue.containsError())
    {
        retValue += ito::ParamHelper::validateParam(result, result, true);
    }

    return result;
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
    updateButtonState();
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

    updateButtonState();
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
    if ((role == Qt::EditRole &&
         (v.toString().count(QLatin1Char('\n')) != item->data(role).toString().count(QLatin1Char('\n')))) ||
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
void ParamInputDialog::on_buttonBox_clicked(QAbstractButton *btn)
{
    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::AcceptRole)
    {
        ito::RetVal retValue;
        getItems(retValue);

        if (retValue.containsError())
        {
            QMessageBox msgBox(this);
            msgBox.setText(retValue.errorMessage());
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
        else if (ui.listWidget->count() < m_minSize ||
            ui.listWidget->count() >= m_maxSize)
        {
            QMessageBox msgBox(this);
            msgBox.setText(tr("The number of values must be in the range [%1, %2]").arg(m_minSize).arg(m_maxSize));
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
        else if ((ui.listWidget->count() - m_minSize) % m_stepSize == 0)
        {
            accept(); // AcceptRole
        }
        else
        {
            QMessageBox msgBox(this);
            msgBox.setText(tr("The number of value does not match the step size"));
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
    }
    else
    {
        reject(); // close dialog with reject
    }
}

//-------------------------------------------------------------------------------------

void ParamInputDialog::on_listWidget_itemDoubleClicked(QListWidgetItem *item)
{
    ui.listWidget->editItem(item);
}

} // end namespace ito
