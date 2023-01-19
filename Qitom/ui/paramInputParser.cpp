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

#include "paramInputParser.h"

#include <qcombobox.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <qspinbox.h>
#include <qtoolbutton.h>
#include <qregularexpression.h>

#include "../../AddInManager/paramHelper.h"
#include "../helper/guiHelper.h"
#include "../helper/compatHelper.h"
#include "dialogPluginPicker.h"
#include "paramInputDialog.h"

namespace ito {

//-------------------------------------------------------------------------------------
ParamInputParser::ParamInputParser(QWidget* canvas) : QObject(canvas)
{
    m_canvas = QPointer<QWidget>(canvas);
    m_iconInfo = QIcon(":/plugins/icons/info.png");
}

//-------------------------------------------------------------------------------------
ParamInputParser::~ParamInputParser()
{
}

//-------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::createInputMask(const QVector<ito::Param>& params)
{
    ito::RetVal retValue;
    QWidget* parent = m_canvas.data();
    if (parent == nullptr)
    {
        return ito::RetVal(
            ito::retError, 0, tr("Canvas widget does not exist any more").toLatin1().data());
    }

    if (parent->layout())
    {
        delete parent->layout();
    }

    m_params = params;
    QVBoxLayout* vLayout = new QVBoxLayout(parent);
    parent->setLayout(vLayout);
    QGridLayout* gridLayout = new QGridLayout();

    QLabel* m_lblInfo = nullptr;
    QLabel* m_lblName = nullptr;
    QLabel* m_lblType = nullptr;
    QWidget* m_content = nullptr;
    int i = 0;

    m_internalData.resize(params.size());

    float dpiFactor = GuiHelper::screenDpiFactor(); // factor related to 96dpi (1.0)

    foreach (const ito::Param& param, params)
    {
        m_lblInfo = new QLabel(parent);
        m_lblInfo->setMaximumSize(24 * dpiFactor, 24 * dpiFactor);
        m_lblInfo->setPixmap(m_iconInfo.pixmap(16 * dpiFactor, 16));

        QString info = QLatin1String(param.getInfo());

        if (info == "")
        {
            m_lblInfo->setToolTip(tr("[no description]"));
        }
        else if (info.length() < 120)
        {
            m_lblInfo->setToolTip(info);
        }
        else
        {
            // try to split string into parts of around 120 characters and replace separatring
            // spaces by \n to have a multi-line tool tip text
            int l = 119;
            while (l < info.length())
            {
                if ((l = info.indexOf(" ", l)) >= 0)
                {
                    info.replace(l, 1, '\n');
                    l += 120;
                }
                else
                {
                    break;
                }
            }
            m_lblInfo->setToolTip(info);
        }

        m_lblName = new QLabel(QString(param.getName()).append(":"), parent);
        m_lblType = new QLabel(parent);

        switch (param.getType())
        {
        case ito::ParamBase::Int:
            m_lblType->setText(tr("[Integer]"));
            m_content = renderTypeInt(param, i, parent);
            break;
        case ito::ParamBase::Char:
            m_lblType->setText(tr("[Char]"));
            m_content = renderTypeChar(param, i, parent);
            break;
        case ito::ParamBase::Double:
            m_lblType->setText(tr("[Double]"));
            m_content = renderTypeDouble(param, i, parent);
            break;
        case ito::ParamBase::String:
            m_lblType->setText(tr("[String]"));
            m_content = renderTypeString(param, i, parent);
            break;
        case ito::ParamBase::IntArray:
            m_lblType->setText(tr("[IntArray]"));
            m_content = renderTypeGenericArray(param, i, parent, ParamBase::IntArray);
            break;
        case ito::ParamBase::DoubleArray:
            m_lblType->setText(tr("[DoubleArray]"));
            m_content = renderTypeGenericArray(param, i, parent, ParamBase::DoubleArray);
            break;
        case ito::ParamBase::CharArray:
            m_lblType->setText(tr("[CharArray]"));
            m_content = renderTypeGenericArray(param, i, parent, ParamBase::CharArray);
            break;
        case ito::ParamBase::HWRef:
            m_lblType->setText(tr("[HW-Instance]"));
            m_content = renderTypeHWRef(param, i, parent);
            break;
        case ito::ParamBase::ComplexArray:
            m_lblType->setText(tr("[ComplexArray]"));
            m_content = renderTypeGenericArray(param, i, parent, ParamBase::ComplexArray);
            break;
        case ito::ParamBase::StringList:
            m_lblType->setText(tr("[StringList]"));
            m_content = renderTypeGenericArray(param, i, parent, ParamBase::StringList);
            break;
        default:
            m_lblType->setText(tr("[unknown]"));
            m_content = new QLabel(tr(" - - error - - "), parent);
            break;
        }

        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(m_content->sizePolicy().hasHeightForWidth());
        m_content->setSizePolicy(sizePolicy);

        if (param.getType() & ito::ParamBase::Readonly)
        {
            m_content->setEnabled(false);
        }

        gridLayout->addWidget(m_lblName, i, 0);
        gridLayout->addWidget(m_content, i, 1);
        gridLayout->addWidget(m_lblType, i, 2);
        gridLayout->addWidget(m_lblInfo, i, 3);
        i++;
    }

    vLayout->addLayout(gridLayout);
    QSpacerItem* verticalSpacer =
        new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding);
    vLayout->addItem(verticalSpacer);

    return retValue;
}

//-------------------------------------------------------------------------------------
bool ParamInputParser::validateInput(
    bool mandatoryValues, ito::RetVal& retValue, bool showMessages /*= false*/)
{
    QWidget* parent = m_canvas.data();

    if (parent == nullptr)
    {
        retValue += ito::RetVal(
            ito::retError, 0, tr("Canvas widget does not exist any more").toLatin1().data());
        return false;
    }

    int i = 0;
    QGridLayout* gridLayout = nullptr;
    ito::ParamBase tempParam;
    QVBoxLayout* temp = qobject_cast<QVBoxLayout*>(parent->layout());
    gridLayout = qobject_cast<QGridLayout*>(temp->itemAt(0)->layout());

    if (gridLayout == nullptr)
    {
        retValue += ito::RetVal(
            ito::retError,
            0,
            tr("QT error: Grid layout could not be identified").toLatin1().data());
        return false;
    }

    foreach (const ito::Param& param, m_params)
    {
        // copy orgParams to params
        tempParam = ito::ParamBase(param);

        switch (param.getType())
        {
        case ito::ParamBase::Int:
            retValue += getIntValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                mandatoryValues);
            break;
        case ito::ParamBase::Char:
            retValue += getCharValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                mandatoryValues);
            break;
        case ito::ParamBase::Double:
            retValue += getDoubleValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                mandatoryValues);
            break;
        case ito::ParamBase::String:
            retValue += getStringValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                mandatoryValues);
            break;
        case ito::ParamBase::IntArray:
        case ito::ParamBase::DoubleArray:
        case ito::ParamBase::CharArray:
        case ito::ParamBase::StringList:
            break;
        case ito::ParamBase::HWRef:
            retValue += getHWValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                mandatoryValues);
            break;
        default:
            break;
        }

        if (retValue.containsError())
        {
            if (showMessages)
            {
                QString text = QString(tr("The parameter '%1' is invalid.")).arg(param.getName());

                if (retValue.hasErrorMessage())
                {
                    text.append("\n\n").append(QLatin1String(retValue.errorMessage()));
                }

                QMessageBox::critical(parent, tr("Invalid input"), text);
            }
            else
            {
            }
            return false;
        }

        i++;
    }

    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getParameters(QVector<ito::ParamBase>& params)
{
    ito::RetVal retValue;
    QWidget* parent = m_canvas.data();

    if (parent == nullptr)
    {
        return ito::RetVal(
            ito::retError, 0, tr("Canvas widget does not exist any more").toLatin1().data());
    }

    int i = 0;
    QGridLayout* gridLayout = nullptr;
    ito::ParamBase tempParam;
    QVBoxLayout* temp = qobject_cast<QVBoxLayout*>(parent->layout());
    gridLayout = qobject_cast<QGridLayout*>(temp->itemAt(0)->layout());
    params.clear();

    if (gridLayout == nullptr)
    {
        retValue += ito::RetVal(
            ito::retError,
            0,
            tr("QT error: Grid layout could not be identified").toLatin1().data());
        return retValue;
    }

    foreach (const ito::Param& param, m_params)
    {
        // copy orgParams to params
        tempParam = ito::ParamBase(param);

        switch (param.getType())
        {
        case ito::ParamBase::Int:
            retValue += getIntValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                false);
            break;
        case ito::ParamBase::Char:
            retValue += getCharValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                false);
            break;
        case ito::ParamBase::Double:
            retValue += getDoubleValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                false);
            break;
        case ito::ParamBase::String:
            retValue += getStringValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                false);
            break;
        case ito::ParamBase::IntArray:
        case ito::ParamBase::DoubleArray:
        case ito::ParamBase::CharArray:
        case ito::ParamBase::StringList:
            tempParam = param;
            break;
        case ito::ParamBase::HWRef:
            retValue += getHWValue(
                tempParam,
                param,
                gridLayout->itemAtPosition(i, 1)->widget(),
                m_internalData[i],
                false);
            break;
        default:
            break;
        }

        i++;
        params.append(tempParam);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeInt(
    const ito::Param& param, int /*virtualIndex*/, QWidget* parent)
{
    const ito::IntMeta* meta = static_cast<const ito::IntMeta*>(param.getMeta());

    if (meta && meta->getMin() == 0 && meta->getMax() == 1)
    {
        // special case, we use a checkbox instead of the spinbox
        QCheckBox* box = new QCheckBox(parent);
        box->setToolTip(tr("checked: 1, unchecked: 0"));
        box->setChecked(param.getVal<int>() > 0);
        return box;
    }
    else
    {
        QSpinBox* box = new QSpinBox(parent);
        box->setDisplayIntegerBase(10);

        if (meta)
        {
            box->setMinimum(meta->getMin());
            box->setMaximum(meta->getMax());
            box->setSingleStep(meta->getStepSize());

            if (meta->getMin() >= 0 && meta->getRepresentation() == ito::ParamMeta::HexNumber)
            {
                // numbers >= 0 and in Hex-representation can be handled...
                box->setDisplayIntegerBase(16);
                box->setPrefix("0x");

                box->setToolTip(tr("min: 0x%1, max: 0x%2, step: 0x%3")
                    .arg(meta->getMin(), 0, 16)
                    .arg(meta->getMax(), 0, 16)
                    .arg(meta->getStepSize(), 0, 16));
            }
            else
            {
                box->setToolTip(tr("min: %1, max: %2, step: %3")
                    .arg(meta->getMin())
                    .arg(meta->getMax())
                    .arg(meta->getStepSize()));
            }
        }
        else
        {
            box->setMinimum(std::numeric_limits<int>::min());
            box->setMaximum(std::numeric_limits<int>::max());
            box->setSingleStep(1);
            box->setToolTip(tr("unlimited"));
        }

        box->setValue(param.getVal<int>());

        return box;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeChar(
    const ito::Param& param, int /*virtualIndex*/, QWidget* parent)
{
    QSpinBox* box = new QSpinBox(parent);

    const ito::CharMeta* meta = static_cast<const ito::CharMeta*>(param.getMeta());

    if (meta)
    {
        box->setMinimum((int)meta->getMin());
        box->setMaximum((int)meta->getMax());
        box->setSingleStep((int)meta->getStepSize());

        if (meta->getMin() >= 0 && meta->getRepresentation() == ito::ParamMeta::HexNumber)
        {
            // numbers >= 0 and in Hex-representation can be handled...
            box->setDisplayIntegerBase(16);
            box->setPrefix("0x");

            box->setToolTip(tr("min: 0x%1, max: 0x%2, step: 0x%3")
                .arg((int)meta->getMin(), 0, 16)
                .arg((int)meta->getMax(), 0, 16)
                .arg((int)meta->getStepSize(), 0, 16));
        }
        else
        {
            box->setToolTip(tr("min: %1, max: %2, step: %3")
                .arg((int)meta->getMin())
                .arg((int)meta->getMax())
                .arg((int)meta->getStepSize()));
        }
    }
    else
    {
        box->setMinimum(std::numeric_limits<char>::min());
        box->setMaximum(std::numeric_limits<char>::max());
        box->setSingleStep(1);
        box->setToolTip(tr("unlimited"));
    }

    box->setValue(param.getVal<int>());

    return box;
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeDouble(
    const ito::Param& param, int /*virtualIndex*/, QWidget* parent)
{
    QDoubleSpinBox* box = new QDoubleSpinBox(parent);
    box->setDecimals(4);
    const ito::DoubleMeta* meta = static_cast<const ito::DoubleMeta*>(param.getMeta());

    if (meta)
    {
        box->setMinimum(meta->getMin());
        box->setMaximum(meta->getMax());

        if (meta->getStepSize() != 0.0)
        {
            box->setSingleStep(meta->getStepSize());
            box->setToolTip(tr("min: %1, max: %2, step: %3")
                                .arg(meta->getMin())
                                .arg(meta->getMax())
                                .arg(meta->getStepSize()));
        }
        else
        {
            box->setToolTip(tr("min: %1, max: %2").arg(meta->getMin()).arg(meta->getMax()));
        }
    }
    else
    {
        box->setMinimum(std::numeric_limits<double>::min());
        box->setMaximum(std::numeric_limits<double>::max());
        box->setToolTip(tr("unlimited"));
    }

    box->setValue(param.getVal<double>());

    return box;
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeString(
    const ito::Param& param, int /*virtualIndex*/, QWidget* parent)
{
    const ito::StringMeta* meta = static_cast<const ito::StringMeta*>(param.getMeta());

    if (meta && meta->getStringType() == ito::StringMeta::String && meta->getLen() > 0)
    {
        QComboBox* cmb = new QComboBox(parent);
        int cur = -1;

        for (int i = 0; i < meta->getLen(); i++)
        {
            cmb->addItem(QLatin1String(meta->getString(i)));

            if (param.getVal<const char*>() &&
                (QLatin1String(param.getVal<const char*>()) == QLatin1String(meta->getString(i))))
            {
                cur = i;
            }
        }

        if (cur >= 0)
        {
            cmb->setCurrentIndex(cur);
        }

        return cmb;
    }
    else
    {
        QLineEdit* txt = new QLineEdit(parent);
        QString value = QLatin1String(param.getVal<const char*>());
        value.replace('\r', "\\r");
        value.replace('\n', "\\n");
        value.replace('\t', "\\t");
        value.replace('\\', "\\");

        if (meta && meta->getStringType() == ito::StringMeta::Wildcard && meta->getLen() > 0)
        {
            if (meta->getLen() == 1)
            {
                QString pattern = CompatHelper::regExpAnchoredPattern(CompatHelper::wildcardToRegularExpression(QLatin1String(meta->getString(0))));
                QRegularExpression reg(pattern);
                txt->setValidator(new QRegularExpressionValidator(reg, txt));
                QString toolTip = tr("%1 [Wildcard]").arg(reg.pattern());
                txt->setToolTip(toolTip);
            }
        }
        else if (meta && meta->getStringType() == ito::StringMeta::RegExp && meta->getLen() > 0)
        {
            if (meta->getLen() == 1)
            {
                QRegularExpression reg(QLatin1String(meta->getString(0)));
                txt->setValidator(new QRegularExpressionValidator(reg, txt));
                QString toolTip = tr("%1 [Regular Expression]").arg(reg.pattern());
                txt->setToolTip(toolTip);
            }
        }

        txt->setText(value);

        return txt;
    }
}

//-------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeHWRef(
    const ito::Param& param, int virtualIndex, QWidget* parent)
{
    QWidget* container = new QWidget(parent);
    QHBoxLayout* layout = new QHBoxLayout();
    QLineEdit* txt = new QLineEdit(container);
    txt->setObjectName(arrayTypeObjectName(param.getType(), virtualIndex));
    txt->setText(tr("[None]"));
    txt->setEnabled(false);

    QToolButton* tool = new QToolButton(container);
    tool->setIcon(QIcon(":/files/icons/browser.png"));
    connect(tool, &QToolButton::clicked, [=]() { browsePluginPicker(virtualIndex); });

    layout->addWidget(txt);
    layout->addWidget(tool);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    container->setLayout(layout);

    return container;
}

//-------------------------------------------------------------------------------------
QString ParamInputParser::getTypeGenericArrayPreview(const ito::Param& param) const
{
    const int maxNum = 10;

    QStringList items;
    int num = param.getLen();

    switch (param.getType())
    {
    case ito::ParamBase::CharArray: {
        auto vals = param.getVal<const char*>();
        for (int i = 0; i < std::min(maxNum, num); ++i)
        {
            items.append(QString::number(vals[i]));
        }
        break;
    }
    case ito::ParamBase::IntArray: {
        auto vals = param.getVal<const int*>();
        for (int i = 0; i < std::min(maxNum, num); ++i)
        {
            items.append(QString::number(vals[i]));
        }
        break;
    }
    case ito::ParamBase::DoubleArray: {
        auto vals = param.getVal<const double*>();
        for (int i = 0; i < std::min(maxNum, num); ++i)
        {
            items.append(QString::number(vals[i], 'g', 4));
        }
        break;
    }
    case ito::ParamBase::ComplexArray: {
        auto vals = param.getVal<const ito::complex128*>();
        for (int i = 0; i < std::min(maxNum, num); ++i)
        {
            if (vals[i].imag() >= 0)
            {
                items.append(
                    QString::number(vals[i].real(), 'g', 4) + "+" +
                    QString::number(vals[i].imag(), 'g', 4) + "i");
            }
            else
            {
                items.append(
                    QString::number(vals[i].real(), 'g', 4) + "-" +
                    QString::number(vals[i].imag(), 'g', 4) + "i");
            }
        }
        break;
    }
    case ito::ParamBase::StringList: {
        auto vals = param.getVal<const ito::ByteArray*>();
        QString s;

        for (int i = 0; i < std::min(maxNum, num); ++i)
        {
            s = QLatin1String(vals[i].data());

            if (s.size() > 20)
            {
                s = s.left(17) + "...";
            }

            items.append(s);
        }
        break;
    }
    default:
        return QString();
    }

    QString content;

    if (num > maxNum)
    {
        content = items.join(";") + ";...";
    }
    else
    {
        content = items.join(";");
    }

    return QString("[%1]").arg(content);
}

//-------------------------------------------------------------------------------------
QString ParamInputParser::arrayTypeObjectName(int paramType, int index) const
{
    // do not translate these strings
    switch (paramType)
    {
    case ParamBase::CharArray:
        return QString("ArrayChar_%1").arg(index);
    case ParamBase::IntArray:
        return QString("ArrayInt_%1").arg(index);
    case ParamBase::DoubleArray:
        return QString("ArrayDbl_%1").arg(index);
    case ParamBase::ComplexArray:
        return QString("ArrayCmplx_%1").arg(index);
    case ParamBase::StringList:
        return QString("ListString_%1").arg(index);
    case ParamBase::HWRef:
        return QString("HWRef_%1").arg(index);
    default:
        return "";
    }
}

//-------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeGenericArray(
    const ito::Param& param, const int virtualIndex, QWidget* parent, int paramType)
{
    QWidget* container = new QWidget(parent);
    QHBoxLayout* layout = new QHBoxLayout();
    QLineEdit* lineEdit = new QLineEdit(container);
    lineEdit->setObjectName(arrayTypeObjectName(paramType, virtualIndex));
    lineEdit->setReadOnly(true);
    lineEdit->setText(getTypeGenericArrayPreview(param));

    QToolButton* tool = new QToolButton(container);
    tool->setIcon(QIcon(":/application/icons/list.png"));
    connect(tool, &QToolButton::clicked, [=]() { browseArrayPicker(virtualIndex); });

    layout->addWidget(lineEdit);
    layout->addWidget(tool);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    container->setLayout(layout);

    return container;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getIntValue(
    ito::ParamBase& param,
    const ito::Param& orgParam,
    QWidget* contentWidget,
    void* /*internalData*/,
    bool /*mandatory*/)
{
    ito::RetVal retVal;
    QSpinBox* box = qobject_cast<QSpinBox*>(contentWidget);
    int value = 0.0;

    if (box)
    {
        value = box->value();
    }
    else
    {
        QCheckBox* c = qobject_cast<QCheckBox*>(contentWidget);

        if (c)
        {
            value = c->isChecked() ? 1 : 0;
        }
        else
        {
            return ito::RetVal(
                ito::retError,
                0,
                tr("Qt error: Spin box widget could not be found").toLatin1().data());
        }
    }

    retVal += ito::ParamHelper::validateIntMeta(
        static_cast<const ito::IntMeta*>(orgParam.getMeta()), value);

    if (!retVal.containsError())
    {
        param.setVal<int>(value);
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getCharValue(
    ito::ParamBase& param,
    const ito::Param& orgParam,
    QWidget* contentWidget,
    void* /*internalData*/,
    bool /*mandatory*/)
{
    ito::RetVal retVal;
    QSpinBox* box = qobject_cast<QSpinBox*>(contentWidget);

    if (box == nullptr)
    {
        return ito::RetVal(
            ito::retError, 0, tr("Qt error: Spin box widget could not be found").toLatin1().data());
    }

    retVal += ito::ParamHelper::validateCharMeta(
        static_cast<const ito::CharMeta*>(orgParam.getMeta()), box->value());

    if (!retVal.containsError())
    {
        param.setVal<char>(box->value());
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getDoubleValue(
    ito::ParamBase& param,
    const ito::Param& orgParam,
    QWidget* contentWidget,
    void* /*internalData*/,
    bool /*mandatory*/)
{
    ito::RetVal retVal;
    QDoubleSpinBox* box = qobject_cast<QDoubleSpinBox*>(contentWidget);

    if (box == nullptr)
    {
        return ito::RetVal(
            ito::retError,
            0,
            tr("Qt error: Double spin box widget could not be found").toLatin1().data());
    }

    retVal += ito::ParamHelper::validateDoubleMeta(
        static_cast<const ito::DoubleMeta*>(orgParam.getMeta()), box->value());

    if (!retVal.containsError())
    {
        param.setVal<double>(box->value());
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getStringValue(
    ito::ParamBase& param,
    const ito::Param& orgParam,
    QWidget* contentWidget,
    void* /*internalData*/,
    bool mandatory)
{
    ito::RetVal retVal;
    QString string;
    QLineEdit* txt = qobject_cast<QLineEdit*>(contentWidget);

    if (txt == nullptr)
    {
        QComboBox* cmb = qobject_cast<QComboBox*>(contentWidget);

        if (cmb == nullptr)
        {
            return ito::RetVal(
                ito::retError,
                0,
                tr("Qt error: String input widget could not be found").toLatin1().data());
        }

        string = cmb->currentText();
    }
    else
    {
        string = txt->text();
    }

    retVal += ito::ParamHelper::validateStringMeta(
        static_cast<const ito::StringMeta*>(orgParam.getMeta()),
        string.toLatin1().data(),
        mandatory);

    if (!retVal.containsError())
    {
        string.replace("\\\\", "\a");
        string.replace("\\n", "\n");
        string.replace("\\r", "\r");
        string.replace("\\t", "\t");
        string.replace("\a", "\\");
        QByteArray ba = string.toLatin1();
        char* temp = ba.data();
        param.setVal<char*>(temp);
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getHWValue(
    ito::ParamBase& param,
    const ito::Param& orgParam,
    QWidget* /*contentWidget*/,
    void* internalData,
    bool mandatory)
{
    ito::RetVal retValue = ito::ParamHelper::validateHWMeta(
        static_cast<const ito::HWMeta*>(orgParam.getMeta()),
        (ito::AddInBase*)internalData,
        mandatory);
    if (!retValue.containsError())
    {
        param.setVal<void*>(internalData);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ParamInputParser::browsePluginPicker(int i)
{
    ito::AddInBase* aib = (ito::AddInBase*)(m_internalData[i]);

    ito::Param p;
    p = m_params[i];

    ito::HWMeta* hwmeta = static_cast<ito::HWMeta*>(p.getMeta());
    QString pluginName = QString();
    int minimumPluginType = 0x0;

    if (hwmeta)
    {
        if (!hwmeta->getHWAddInName().empty())
        {
            pluginName = hwmeta->getHWAddInName().data();
        }
        minimumPluginType = hwmeta->getMinType();
    }

    QWidget* canvas = m_canvas.data();

    if (canvas)
    {
        DialogPluginPicker* dialog =
            new DialogPluginPicker(true, aib, minimumPluginType, pluginName, canvas);

        if (dialog->exec() == 1) // accepted
        {
            aib = dialog->getSelectedInstance();
            m_internalData[i] = (void*)aib;

            QString name = arrayTypeObjectName(ParamBase::HWRef, i);
            QLineEdit* le = canvas->findChild<QLineEdit*>(name);

            if (le && aib)
            {
                if (aib->getIdentifier() != "")
                {
                    le->setText(tr("%1, Identifier: %2")
                                    .arg(aib->getBasePlugin()->objectName())
                                    .arg(aib->getIdentifier()));
                }
                else
                {
                    le->setText(
                        tr("%1, ID: %2").arg(aib->getBasePlugin()->objectName()).arg(aib->getID()));
                }
            }
            else if (le && !aib)
            {
                le->setText(tr("[None]"));
            }
        }

        DELETE_AND_SET_NULL(dialog);
    }
}

//-------------------------------------------------------------------------------------
void ParamInputParser::browseArrayPicker(int i)
{
    const ito::Param p = m_params[i];
    QString leString;
    QLineEdit* lineEdit = nullptr;

    if (m_canvas)
    {
        QString name = arrayTypeObjectName(p.getType(), i);
        lineEdit = m_canvas->findChild<QLineEdit*>(name);
    }

    if (lineEdit)
    {
        ParamInputDialog* dialog = new ParamInputDialog(p, m_canvas.data());

        if (dialog->exec() == 1) // accepted
        {
            RetVal retValue;
            Param newParam = dialog->getItems(retValue);

            if (retValue.containsError())
            {
                QMessageBox msgBox;
                msgBox.setWindowTitle(tr("Invalid input"));
                msgBox.setText(retValue.errorMessage());
                msgBox.setIcon(QMessageBox::Critical);
                msgBox.exec();
            }
            else
            {
                m_params[i].copyValueFrom(&newParam);
            }

            lineEdit->setText(getTypeGenericArrayPreview(m_params[i]));
        }

        DELETE_AND_SET_NULL(dialog);
    }
}

} // end namespace ito
