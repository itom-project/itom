/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2021, Institut fuer Technische Optik (ITO),
   Universitaet Stuttgart, Germany

   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.

   In addition, as a special exception, the Institut fuer Technische
   Optik (ITO) gives you certain additional rights.
   These rights are described in the ITO LGPL Exception version 1.0,
   which can be found in the file LGPL_EXCEPTION.txt in this package.
*********************************************************************** */

#include "dataObjectDelegate.h"
#include "dataObjectModel.h"

#include <qboxlayout.h>
#include <qcolordialog.h>
#include <qdatetimeedit.h>
#include <qitemdelegate.h>
#include <qlabel.h>
#include <qspinbox.h>

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectDelegate::DataObjectDelegate(QObject* parent /*= 0*/) :
    QItemDelegate(parent), m_min(-std::numeric_limits<double>::max()),
    m_max(std::numeric_limits<double>::max()), m_editorDecimals(3)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectDelegate::~DataObjectDelegate()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* DataObjectDelegate::createEditor(
    QWidget* parent, const QStyleOptionViewItem& /*option*/, const QModelIndex& index) const
{
    const DataObjectModel* model = qobject_cast<const DataObjectModel*>(index.model());
    int type = model->getType();

    QWidget* result = NULL;

    // this is a workaround, since the saturate_cast from double::max() to limits of int8 for
    // instance if malignious.
    int intMin =
        m_min < (double)(std::numeric_limits<int>::min()) ? std::numeric_limits<int>::min() : m_min;
    int intMax =
        m_max > (double)(std::numeric_limits<int>::max()) ? std::numeric_limits<int>::max() : m_max;

    QString suffix;
    if (m_suffixes.size() > 0)
    {
        suffix = m_suffixes[std::min((int)m_suffixes.size() - 1, index.column())];
    }

    switch (type)
    {
    case ito::tInt8: {
        QSpinBox* editor = new QSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setMinimum(
            std::max(std::numeric_limits<ito::int8>::min(), cv::saturate_cast<ito::int8>(intMin)));
        editor->setMaximum(
            std::min(std::numeric_limits<ito::int8>::max(), cv::saturate_cast<ito::int8>(intMax)));
        result = editor;
    }
    break;
    case ito::tUInt8: {
        QSpinBox* editor = new QSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setMinimum(std::max(
            std::numeric_limits<ito::uint8>::min(), cv::saturate_cast<ito::uint8>(intMin)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::uint8>::max(), cv::saturate_cast<ito::uint8>(intMax)));
        result = editor;
    }
    break;
    case ito::tInt16: {
        QSpinBox* editor = new QSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setMinimum(std::max(
            std::numeric_limits<ito::int16>::min(), cv::saturate_cast<ito::int16>(intMin)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::int16>::max(), cv::saturate_cast<ito::int16>(intMax)));
        result = editor;
    }
    break;
    case ito::tUInt16: {
        QSpinBox* editor = new QSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setMinimum(std::max(
            std::numeric_limits<ito::uint16>::min(), cv::saturate_cast<ito::uint16>(intMin)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::uint16>::max(), cv::saturate_cast<ito::uint16>(intMax)));
        result = editor;
    }
    break;
    case ito::tInt32: {
        QSpinBox* editor = new QSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setMinimum(std::max(
            std::numeric_limits<ito::int32>::min(), cv::saturate_cast<ito::int32>(intMin)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::int32>::max(), cv::saturate_cast<ito::int32>(intMax)));
        result = editor;
    }
    break;
    case ito::tUInt32: {
        QSpinBox* editor = new QSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setMinimum(std::max(
            std::numeric_limits<ito::uint32>::min(), cv::saturate_cast<ito::uint32>(intMin)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::uint32>::max(), cv::saturate_cast<ito::uint32>(intMax)));
        result = editor;
    }
    break;
    case ito::tFloat32: {
        QDoubleSpinBox* editor = new QDoubleSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setDecimals(m_editorDecimals);
        editor->setMinimum(std::max(
            -std::numeric_limits<ito::float32>::max(), cv::saturate_cast<ito::float32>(m_min)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::float32>::max(), cv::saturate_cast<ito::float32>(m_max)));
        result = editor;
    }
    break;
    case ito::tFloat64: {
        QDoubleSpinBox* editor = new QDoubleSpinBox(parent);
        editor->setSuffix(suffix);
        editor->setDecimals(m_editorDecimals);
        editor->setMinimum(std::max(
            -std::numeric_limits<ito::float64>::max(), cv::saturate_cast<ito::float64>(m_min)));
        editor->setMaximum(std::min(
            std::numeric_limits<ito::float64>::max(), cv::saturate_cast<ito::float64>(m_max)));
        result = editor;
    }
    break;
    case ito::tComplex64: {
        QWidget* editor = new QWidget(parent);
        QHBoxLayout* layout = new QHBoxLayout();
        layout->setSpacing(0);
        layout->setContentsMargins(0, 0, 0, 0);

        QDoubleSpinBox* realEditor = new QDoubleSpinBox(parent);
        realEditor->setSuffix(suffix);
        realEditor->setDecimals(m_editorDecimals);
        realEditor->setMinimum(std::max(
            -std::numeric_limits<ito::float32>::max(), cv::saturate_cast<ito::float32>(m_min)));
        realEditor->setMaximum(std::min(
            std::numeric_limits<ito::float32>::max(), cv::saturate_cast<ito::float32>(m_max)));
        realEditor->setToolTip(tr("Real part"));
        realEditor->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        QDoubleSpinBox* imagEditor = new QDoubleSpinBox(parent);
        imagEditor->setSuffix(suffix);
        imagEditor->setDecimals(m_editorDecimals);
        imagEditor->setMinimum(std::max(
            -std::numeric_limits<ito::float32>::max(), cv::saturate_cast<ito::float32>(m_min)));
        imagEditor->setMaximum(std::min(
            std::numeric_limits<ito::float32>::max(), cv::saturate_cast<ito::float32>(m_max)));
        imagEditor->setToolTip(tr("Imaginary part"));
        imagEditor->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        QLabel* label = new QLabel("+i");
        label->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

        layout->addWidget(realEditor);
        layout->addWidget(label);
        layout->addWidget(imagEditor);

        editor->setLayout(layout);

        return editor;
    }
    break;
    case ito::tComplex128: {
        QWidget* editor = new QWidget(parent);
        QHBoxLayout* layout = new QHBoxLayout();
        layout->setSpacing(0);
        layout->setContentsMargins(0, 0, 0, 0);

        QDoubleSpinBox* realEditor = new QDoubleSpinBox(parent);
        realEditor->setSuffix(suffix);
        realEditor->setDecimals(m_editorDecimals);
        realEditor->setMinimum(std::max(
            -std::numeric_limits<ito::float64>::max(), cv::saturate_cast<ito::float64>(m_min)));
        realEditor->setMaximum(std::min(
            std::numeric_limits<ito::float64>::max(), cv::saturate_cast<ito::float64>(m_max)));
        realEditor->setToolTip(tr("Real part"));
        realEditor->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        QDoubleSpinBox* imagEditor = new QDoubleSpinBox(parent);
        imagEditor->setSuffix(suffix);
        imagEditor->setDecimals(m_editorDecimals);
        imagEditor->setMinimum(std::max(
            -std::numeric_limits<ito::float64>::max(), cv::saturate_cast<ito::float64>(m_min)));
        imagEditor->setMaximum(std::min(
            std::numeric_limits<ito::float64>::max(), cv::saturate_cast<ito::float64>(m_max)));
        imagEditor->setToolTip(tr("Imaginary part"));
        imagEditor->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

        QLabel* label = new QLabel("+i");
        label->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

        layout->addWidget(realEditor);
        layout->addWidget(label);
        layout->addWidget(imagEditor);

        editor->setLayout(layout);

        return editor;
    }
    break;
    case ito::tRGBA32: {
        QColorDialog* colorDialog = new QColorDialog(parent);
        colorDialog->setOption(QColorDialog::ShowAlphaChannel, true);
        return colorDialog;
    }
    break;
    case ito::tTimeDelta: {
        // no editor available for this
    }
    break;
    case ito::tDateTime: {
        QDateTimeEdit* dte = new QDateTimeEdit(parent);
        dte->setCalendarPopup(true);
        return dte;
    }
    break;
    }

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
    const DataObjectModel* model = qobject_cast<const DataObjectModel*>(index.model());
    int type = model->getType();

    //    QWidget *result = NULL;

    switch (type)
    {
    case ito::tInt8:
    case ito::tInt16:
    case ito::tInt32: {
        QSpinBox* spinBox = static_cast<QSpinBox*>(editor);
        int value = model->data(index, Qt::EditRole).toInt();
        spinBox->setValue(value);
    }
    break;

    case ito::tUInt8:
    case ito::tUInt16:
    case ito::tUInt32: {
        QSpinBox* spinBox = static_cast<QSpinBox*>(editor);
        uint value = model->data(index, Qt::EditRole).toUInt();
        spinBox->setValue(value);
    }
    break;

    case ito::tFloat32:
    case ito::tFloat64: {
        QDoubleSpinBox* spinBox = static_cast<QDoubleSpinBox*>(editor);
        double value = model->data(index, Qt::EditRole).toDouble();
        spinBox->setValue(value);
    }
    break;
    case ito::tComplex64: {
        QWidget* widget = static_cast<QWidget*>(editor);
        ito::complex64 value = model->data(index, Qt::EditRole).value<ito::complex64>();
        QDoubleSpinBox* realSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(0)->widget());
        QDoubleSpinBox* imagSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(2)->widget());
        realSpinBox->setValue(value.real());
        imagSpinBox->setValue(value.imag());
    }
    break;
    case ito::tComplex128: {
        QWidget* widget = static_cast<QWidget*>(editor);
        ito::complex128 value = model->data(index, Qt::EditRole).value<ito::complex128>();
        QDoubleSpinBox* realSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(0)->widget());
        QDoubleSpinBox* imagSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(2)->widget());
        realSpinBox->setValue(value.real());
        imagSpinBox->setValue(value.imag());
    }
    break;
    case ito::tRGBA32: {
        QColorDialog* colorDialog = static_cast<QColorDialog*>(editor);
        QColor color = model->data(index, Qt::EditRole).value<QColor>();
        colorDialog->setCurrentColor(color);
    }
    break;
    case ito::tDateTime: {
        QDateTimeEdit* dte = static_cast<QDateTimeEdit*>(editor);
        QDateTime dt = model->data(index, Qt::EditRole).value<QDateTime>();
        dte->setDateTime(dt);
    }
    break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectDelegate::setModelData(
    QWidget* editor, QAbstractItemModel* model, const QModelIndex& index) const
{
    const DataObjectModel* model2 = qobject_cast<DataObjectModel*>(model);
    int type = model2->getType();

    //    QWidget *result = NULL;

    switch (type)
    {
    case ito::tInt8:
    case ito::tInt16:
    case ito::tInt32: {
        QSpinBox* spinBox = static_cast<QSpinBox*>(editor);
        model->setData(index, spinBox->value(), Qt::EditRole);
    }
    break;

    case ito::tUInt8:
    case ito::tUInt16:
    case ito::tUInt32: {
        QSpinBox* spinBox = static_cast<QSpinBox*>(editor);
        model->setData(index, spinBox->value(), Qt::EditRole);
    }
    break;

    case ito::tFloat32:
    case ito::tFloat64: {
        QDoubleSpinBox* spinBox = static_cast<QDoubleSpinBox*>(editor);
        model->setData(index, spinBox->value(), Qt::EditRole);
    }
    break;
    case ito::tComplex64: {
        QWidget* widget = static_cast<QWidget*>(editor);
        QDoubleSpinBox* realSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(0)->widget());
        QDoubleSpinBox* imagSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(2)->widget());
        model->setData(
            index,
            QVariant::fromValue<ito::complex64>(
                ito::complex64(realSpinBox->value(), imagSpinBox->value())),
            Qt::EditRole);
    }
    break;
    case ito::tComplex128: {
        QWidget* widget = static_cast<QWidget*>(editor);
        QDoubleSpinBox* realSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(0)->widget());
        QDoubleSpinBox* imagSpinBox =
            static_cast<QDoubleSpinBox*>(widget->layout()->itemAt(2)->widget());
        model->setData(
            index,
            QVariant::fromValue<ito::complex128>(
                ito::complex128(realSpinBox->value(), imagSpinBox->value())),
            Qt::EditRole);
    }
    break;
    case ito::tRGBA32: {
        QColorDialog* colorDialog = static_cast<QColorDialog*>(editor);
        QColor color = colorDialog->currentColor();
        model->setData(index, color, Qt::EditRole);
    }
    break;
    case ito::tDateTime: {
        QDateTimeEdit* dte = static_cast<QDateTimeEdit*>(editor);
        QDateTime datetime = dte->dateTime();
        model->setData(index, datetime, Qt::EditRole);
    }
    break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObjectDelegate::updateEditorGeometry(
    QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index) const
{
    editor->setGeometry(option.rect);
}
