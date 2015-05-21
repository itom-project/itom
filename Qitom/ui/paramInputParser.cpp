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

#include "paramInputParser.h"
#include <qmessagebox.h>
#include <qspinbox.h>
#include <qcombobox.h>
#include <qlineedit.h>
#include <qtoolbutton.h>
#include <qregexp.h>
#include "../global.h"
#include "dialogPluginPicker.h"
#include "../helper/paramHelper.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
ParamInputParser::ParamInputParser(QWidget *canvas) :
    QObject(canvas)
{
    m_canvas = QPointer<QWidget>(canvas);
    m_iconInfo = QIcon(":/plugins/icons/info.png");

    m_pSignalMapper = new QSignalMapper(this);
    connect(m_pSignalMapper, SIGNAL(mapped(int)), this, SLOT(browsePluginPicker(int)));
}

//----------------------------------------------------------------------------------------------------------------------------------
ParamInputParser::~ParamInputParser()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::createInputMask(const QVector<ito::Param> &params)
{
    ito::RetVal retValue;
    QWidget *parent = m_canvas.data();
    if (parent == NULL)
    {
        return ito::RetVal(ito::retError, 0, tr("Canvas widget does not exist any more").toLatin1().data());
    }

    if (parent->layout())
    {
        delete parent->layout();
    }

    m_params = params;
    QVBoxLayout* vLayout = new QVBoxLayout(parent);
    parent->setLayout(vLayout);
    QGridLayout *gridLayout = new QGridLayout();

    QLabel *m_lblInfo = NULL;
    QLabel *m_lblName = NULL;
    QLabel *m_lblType = NULL;
    QWidget *m_content = NULL;
    int i = 0;

    m_internalData.resize(params.size());

    foreach (const ito::Param &param, params)
    {
        m_lblInfo = new QLabel(parent);
        m_lblInfo->setMaximumSize(24,24);
        m_lblInfo->setPixmap(m_iconInfo.pixmap(16,16));
        m_lblInfo->setToolTip(param.getInfo());

        m_lblName = new QLabel(QString(param.getName()).append(":"), parent);
        m_lblType = new QLabel(parent);

        switch(param.getType() & ito::paramTypeMask)
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
        case ito::ParamBase::HWRef & ito::paramTypeMask:
            m_lblType->setText(tr("[HW-Instance]"));
            m_content = renderTypeHWRef(param, i, parent);
            break;
        default:
            m_lblType->setText(tr("[unknown]"));
            m_content = new QLabel(" - - error - - ",parent);
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
    QSpacerItem *verticalSpacer = new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding);
    vLayout->addItem(verticalSpacer);

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool ParamInputParser::validateInput(bool mandatoryValues, ito::RetVal &retValue, bool showMessages /*= false*/)
{
    QWidget *parent = m_canvas.data();
    if (parent == NULL)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Canvas widget does not exist any more").toLatin1().data());
        return false;
    }

    int i = 0;
    QGridLayout *gridLayout = NULL;
    ito::ParamBase tempParam;
    QVBoxLayout *temp = qobject_cast<QVBoxLayout*>(parent->layout());
    gridLayout = qobject_cast<QGridLayout*>(temp->itemAt(0)->layout());

    if (gridLayout == NULL)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("QT error: Grid layout could not be identified").toLatin1().data());
        return false;
    }

    foreach (const ito::Param &param, m_params)
    {        
        //copy orgParams to params
        tempParam =  ito::ParamBase(param);

        switch(param.getType() & ito::paramTypeMask)
        {
        case ito::ParamBase::Int:
            retValue += getIntValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], mandatoryValues);
            break;
        case ito::ParamBase::Char:
            retValue += getCharValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], mandatoryValues);
            break;
        case ito::ParamBase::Double:
            retValue += getDoubleValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], mandatoryValues);
            break;
        case ito::ParamBase::String:
            retValue += getStringValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], mandatoryValues);
            break;
        case ito::ParamBase::HWRef & ito::paramTypeMask:
            retValue += getHWValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], mandatoryValues);
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
                    text.append("\n\n").append(retValue.errorMessage());
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
ito::RetVal ParamInputParser::getParameters(QVector<ito::ParamBase> &params)
{
    ito::RetVal retValue;
    QWidget *parent = m_canvas.data();
    if (parent == NULL)
    {
        return ito::RetVal(ito::retError, 0, tr("Canvas widget does not exist any more").toLatin1().data());
    }

    int i = 0;
    QGridLayout *gridLayout = NULL;
    ito::ParamBase tempParam;
    QVBoxLayout *temp = qobject_cast<QVBoxLayout*>(parent->layout());
    gridLayout = qobject_cast<QGridLayout*>(temp->itemAt(0)->layout());
    params.clear();
    
    if (gridLayout == NULL)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("QT error: Grid layout could not be identified").toLatin1().data());
        return retValue;
    }

    foreach (const ito::Param &param, m_params)
    {
        //copy orgParams to params
        tempParam =  ito::ParamBase(param);

        switch(param.getType() & ito::paramTypeMask)
        {
        case ito::ParamBase::Int:
            retValue += getIntValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], false);
            break;
        case ito::ParamBase::Char:
            retValue += getCharValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], false);
            break;
        case ito::ParamBase::Double:
            retValue += getDoubleValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], false);
            break;
        case ito::ParamBase::String:
            retValue += getStringValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], false);
            break;
        case ito::ParamBase::HWRef & ito::paramTypeMask:
            retValue += getHWValue(tempParam, param, gridLayout->itemAtPosition(i,1)->widget(), m_internalData[i], false);
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
QWidget* ParamInputParser::renderTypeInt(const ito::Param &param, int /*virtualIndex*/, QWidget *parent)
{
    const ito::IntMeta *meta = static_cast<const ito::IntMeta*>(param.getMeta());
    if (meta && meta->getMin() == 0 && meta->getMax() == 1)
    {
        //special case, we use a checkbox instead of the spinbox
        QCheckBox *box = new QCheckBox(parent);
        box->setToolTip(tr("checked: 1, unchecked: 0"));
        box->setChecked(param.getVal<int>() > 0);
        return box;
    }
    else
    {
        QSpinBox *box = new QSpinBox(parent);
        
        if (meta)
        {
            box->setMinimum(meta->getMin());
            box->setMaximum(meta->getMax());
            box->setSingleStep(meta->getStepSize());
            box->setToolTip(QString("min: %1, max: %2, step: %3").arg(meta->getMin()).arg(meta->getMax()).arg(meta->getStepSize()));
        }
        else
        {
            box->setMinimum(std::numeric_limits<int>::min());
            box->setMaximum(std::numeric_limits<int>::max());
            box->setSingleStep(1);
            box->setToolTip("unlimited");
        }

        box->setValue(const_cast<ito::Param&>(param).getVal<int>());

        return box;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeChar(const ito::Param &param, int /*virtualIndex*/, QWidget *parent)
{
    QSpinBox *box = new QSpinBox(parent);
    
    const ito::CharMeta *meta = static_cast<const ito::CharMeta*>(param.getMeta());
    if (meta)
    {
        box->setMinimum((int)meta->getMin());
        box->setMaximum((int)meta->getMax());
        box->setSingleStep((int)meta->getStepSize());
        box->setToolTip(QString("min: %1, max: %2, step: %3").arg(meta->getMin()).arg(meta->getMax()).arg(meta->getStepSize()));
    }
    else
    {
        box->setMinimum(std::numeric_limits<char>::min());
        box->setMaximum(std::numeric_limits<char>::max());
        box->setSingleStep(1);
        box->setToolTip("unlimited");
    }

    box->setValue(const_cast<ito::Param&>(param).getVal<int>());

    return box;
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeDouble(const ito::Param &param, int /*virtualIndex*/, QWidget *parent)
{
    QDoubleSpinBox *box = new QDoubleSpinBox(parent);
    box->setDecimals(4);
    
    const ito::DoubleMeta *meta = static_cast<const ito::DoubleMeta*>(param.getMeta());
    if (meta)
    {
        box->setMinimum(meta->getMin());
        box->setMaximum(meta->getMax());
        if (meta->getStepSize() != 0.0)
        {
            box->setSingleStep(meta->getStepSize());
            box->setToolTip(QString("min: %1, max: %2, step: %3").arg(meta->getMin()).arg(meta->getMax()).arg(meta->getStepSize()));
        }
        else
        {
            box->setToolTip(QString("min: %1, max: %2").arg(meta->getMin()).arg(meta->getMax()));
        }
    }
    else
    {
        box->setMinimum(std::numeric_limits<double>::min());
        box->setMaximum(std::numeric_limits<double>::max());
        box->setToolTip("unlimited");
    }

    box->setValue(const_cast<ito::Param&>(param).getVal<double>());

    return box;
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeString(const ito::Param &param, int /*virtualIndex*/, QWidget *parent)
{
    const ito::StringMeta *meta = static_cast<const ito::StringMeta*>(param.getMeta());

    if (meta && meta->getStringType() == ito::StringMeta::String && meta->getLen() > 0)
    {
        QComboBox *cmb = new QComboBox(parent);
        int cur = -1;
        for (int i = 0; i < meta->getLen(); i++)
        {
            cmb->addItem(meta->getString(i));
            if (param.getVal<char*>() && strcmp(param.getVal<char*>(), meta->getString(i)) == 0)
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
        QLineEdit *txt = new QLineEdit(parent);
        QString value = const_cast<ito::Param&>(param).getVal<char*>();
        value.replace('\r', "\\r");
        value.replace('\n', "\\n");
        value.replace('\t', "\\t");
        value.replace('\\', "\\");

        if (meta && meta->getStringType() == ito::StringMeta::Wildcard && meta->getLen() > 0)
        {
            if (meta->getLen() == 1)
            {
                QRegExp reg(meta->getString(0), Qt::CaseSensitive, QRegExp::Wildcard);
                txt->setValidator(new QRegExpValidator(reg, txt));
                QString toolTip = tr("%1 [Wildcard]").arg(reg.pattern());
                txt->setToolTip(toolTip);
            }
        }
        else if (meta && meta->getStringType() == ito::StringMeta::RegExp && meta->getLen() > 0)
        {
            if (meta->getLen() == 1)
            {
                QRegExp reg(meta->getString(0), Qt::CaseSensitive, QRegExp::RegExp);
                txt->setValidator(new QRegExpValidator(reg, txt));
                QString toolTip = tr("%1 [Regular Expression]").arg(reg.pattern());
                txt->setToolTip(toolTip);
            }
        }

        txt->setText(value);
        return txt;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* ParamInputParser::renderTypeHWRef(const ito::Param & /*param*/, int virtualIndex, QWidget *parent)
{
    QWidget *container = new QWidget(parent);
    QHBoxLayout *layout = new QHBoxLayout();
    QLineEdit *txt = new QLineEdit(container);
    txt->setObjectName(QString("HWRef_%1").arg(virtualIndex));
    txt->setText(tr("[None]"));
    txt->setEnabled(false);

    QToolButton *tool = new QToolButton(container);
    tool->setIcon(QIcon(":/files/icons/browser.png"));
    connect(tool, SIGNAL(clicked()), m_pSignalMapper, SLOT(map()));
    m_pSignalMapper->setMapping(tool, virtualIndex);

    layout->addWidget(txt);
    layout->addWidget(tool);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    container->setLayout(layout);

    return container;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getIntValue(ito::ParamBase &param, const ito::Param &orgParam, QWidget *contentWidget, void* /*internalData*/, bool /*mandatory*/)
{
    ito::RetVal retVal;
    QSpinBox *box = qobject_cast<QSpinBox*>(contentWidget);
    if (box == NULL)
    {
        QCheckBox *c = qobject_cast<QCheckBox*>(contentWidget);
        if (c)
        {
            int val = c->isChecked() ? 1 : 0;
            retVal += ito::ParamHelper::validateIntMeta(static_cast<const ito::IntMeta*>(orgParam.getMeta()), val);
            if (!retVal.containsError())
            {
                param.setVal<int>(val);
            }
        }
        else
        {
            return ito::RetVal(ito::retError, 0, tr("QT error: Spin box widget could not be found").toLatin1().data());
        }
    }
    else
    {
        retVal += ito::ParamHelper::validateIntMeta(static_cast<const ito::IntMeta*>(orgParam.getMeta()), box->value());

        if (!retVal.containsError())
        {
            param.setVal<int>(box->value());
        }
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getCharValue(ito::ParamBase &param, const ito::Param &orgParam, QWidget *contentWidget, void* /*internalData*/, bool /*mandatory*/)
{
    ito::RetVal retVal;
    QSpinBox *box = qobject_cast<QSpinBox*>(contentWidget);
    if (box == NULL)
    {
        return ito::RetVal(ito::retError, 0, tr("QT error: Spin box widget could not be found").toLatin1().data());
    }
    
    retVal += ito::ParamHelper::validateCharMeta(static_cast<const ito::CharMeta*>(orgParam.getMeta()), box->value());

    if (!retVal.containsError())
    {
        param.setVal<char>(box->value());
    }
    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getDoubleValue(ito::ParamBase &param, const ito::Param &orgParam, QWidget *contentWidget, void* /*internalData*/, bool /*mandatory*/)
{
    ito::RetVal retVal;
    QDoubleSpinBox *box = qobject_cast<QDoubleSpinBox*>(contentWidget);
    if (box == NULL)
    {
        return ito::RetVal(ito::retError, 0, tr("QT error: Double spin box widget could not be found").toLatin1().data());
    }

    retVal += ito::ParamHelper::validateDoubleMeta(static_cast<const ito::DoubleMeta*>(orgParam.getMeta()), box->value());

    if (!retVal.containsError())
    {
        param.setVal<double>(box->value());
    }
    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getStringValue(ito::ParamBase &param, const ito::Param &orgParam, QWidget *contentWidget, void* /*internalData*/, bool mandatory)
{
    ito::RetVal retVal;
    QString string;
    QLineEdit *txt = qobject_cast<QLineEdit*>(contentWidget);
    if (txt == NULL)
    {
        QComboBox *cmb = qobject_cast<QComboBox*>(contentWidget);
        if (cmb == NULL)
        {
            return ito::RetVal(ito::retError, 0, tr("QT error: String input widget could not be found").toLatin1().data());
        }
        string = cmb->currentText();
    }
    else
    {
        string = txt->text();
    }

    retVal += ito::ParamHelper::validateStringMeta(static_cast<const ito::StringMeta*>(orgParam.getMeta()), string.toLatin1().data(), mandatory);

    if (!retVal.containsError())
    {
        string.replace("\\\\", "#~!?üü@@?!~#");
        string.replace("\\n", "\n");
        string.replace("\\r", "\r");
        string.replace("\\t", "\t");
        string.replace("#~!?üü@@?!~#", "\\");
        QByteArray ba = string.toLatin1();
        char *temp = ba.data();
        param.setVal<char*>(temp);
    }
    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamInputParser::getHWValue(ito::ParamBase &param, const ito::Param &orgParam, QWidget* /*contentWidget*/, void* internalData, bool mandatory)
{
    ito::RetVal retValue = ito::ParamHelper::validateHWMeta(static_cast<const ito::HWMeta*>(orgParam.getMeta()), (ito::AddInBase*)internalData, mandatory);
    if (!retValue.containsError())
    {
        param.setVal<void*>(internalData);
    }
    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ParamInputParser::browsePluginPicker(int i)
{
    ito::AddInBase *aib = NULL;
    aib = (ito::AddInBase*)(m_internalData[i]);

    ito::Param p;
    p = m_params[i];
    
    ito::HWMeta* hwmeta = static_cast<ito::HWMeta*>(p.getMeta());
    QString pluginName = QString::Null();
    int minimumPluginType = 0x0;
    if (hwmeta)
    {
        if (!hwmeta->getHWAddInName().empty())
        {
            pluginName = hwmeta->getHWAddInName().data();
        }
        minimumPluginType = hwmeta->getMinType();
    }

    QWidget *canvas = m_canvas.data();
    if (canvas)
    {
        DialogPluginPicker *dialog = new DialogPluginPicker(true, aib, minimumPluginType, pluginName, canvas);

        if (dialog->exec() == 1) //accepted
        {
            aib = dialog->getSelectedInstance();
            m_internalData[i] = (void*)aib;

            QString name = QString("HWRef_%1").arg(i);
            QLineEdit *le = canvas->findChild<QLineEdit*>(name);
            if (le && aib)
            {
                if (aib->getIdentifier() != "")
                {
                    le->setText(QString("%1, Identifier: %2").arg(aib->getBasePlugin()->objectName()).arg(aib->getIdentifier()));
                }
                else
                {
                    le->setText(QString("%1, ID: %2").arg(aib->getBasePlugin()->objectName()).arg(aib->getID()));
                }
            }
            else if (le && !aib)
            {
                le->setText("[None]");
            }
        }

        DELETE_AND_SET_NULL(dialog);
    }
}

} //end namespace ito

