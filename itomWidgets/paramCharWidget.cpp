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

#include "paramCharWidget.h"

#include <qspinbox.h>
#include <qcheckbox.h>
#include "sliderWidget.h"
#include "doubleSpinBox.h"
#include <qlayout.h>
#include "qtpropertybrowserutils_p.h"

namespace ito
{

class ParamCharWidgetPrivate
{
public:
    ParamCharWidget *q_ptr;
    ParamCharWidgetPrivate() {}

    QtBoolEdit *m_pCheckBox;
    QSpinBox *m_pSpinBox;
    SliderWidget *m_pSliderWidget;
    ito::Param m_param;

    //--------------------------------------------------
    void slotValueChanged(int value)
    {
        //value changed signal from spinbox
        if (round(value))
        {
            bool blocked = m_pSpinBox->signalsBlocked();
            m_pSpinBox->blockSignals(true);
            m_pSpinBox->setValue(value);
            m_pSpinBox->blockSignals(blocked);
        }

        emit q_ptr->valueChanged(value);
    }

    //--------------------------------------------------
    void slotValueChanged(double value)
    {
        int value2 = (int)value;
        //value changed signal from slider widget
        if (round(value2))
        {
            bool blocked = m_pSliderWidget->signalsBlocked();
            m_pSliderWidget->blockSignals(true);
            m_pSliderWidget->setValue(value2);
            m_pSliderWidget->blockSignals(blocked);
        }

        emit q_ptr->valueChanged(value2);
    }

    //--------------------------------------------------
    void slotChecked(bool checked)
    {
        //value changed signal from checkbox
        emit q_ptr->valueChanged(checked ? 1 : 0);
    }

private:
    //--------------------------------------------------
    bool round(int &value)
    {
        //check if value fits to optional step size of meta information.
        //If it does not fit, round it to the next possible value and return true (changed)
        const ito::CharMeta *meta = m_param.getMetaT<const ito::CharMeta>();
        char step = meta->getStepSize();
        char minimum = meta->getMin();
        if (step != 1 && ((value - minimum) % step) != 0)
        {
            value = qBound<int>(minimum, minimum + step * qRound(float(value - minimum) / step), meta->getMax());
            return true;
        }
        else
        {
            return false;
        }
    }

    Q_DISABLE_COPY(ParamCharWidgetPrivate);
};

//---------------------------------------------------------------------------
ParamCharWidget::ParamCharWidget(QWidget *parent /*= NULL*/) :
    QWidget(parent),
    d_ptr(new ParamCharWidgetPrivate())
{
    d_ptr->q_ptr = this;

    Q_D(ParamCharWidget);

    char minChar = std::numeric_limits<char>::min();
    char maxChar = std::numeric_limits<char>::max();
    d->m_param = ito::Param("", ito::ParamBase::Char, minChar, maxChar, (char)0, "");

    QHBoxLayout *layout = new QHBoxLayout();
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);;

    d->m_pCheckBox = new QtBoolEdit(this);
    layout->addWidget(d->m_pCheckBox);
    d->m_pCheckBox->setVisible(false);
    connect(d->m_pCheckBox, SIGNAL(toggled(bool)), this, SLOT(slotChecked(bool)));

    d->m_pSpinBox = new QSpinBox(this);
    layout->addWidget(d->m_pSpinBox);
    d->m_pSpinBox->setVisible(false);
    connect(d->m_pSpinBox, SIGNAL(valueChanged(int)), this, SLOT(slotValueChanged(int)));

    d->m_pSliderWidget = new SliderWidget(this);
    layout->addWidget(d->m_pSliderWidget);
    d->m_pSliderWidget->setVisible(false);
    d->m_pSliderWidget->setDecimals(0);
    connect(d->m_pSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(slotValueChanged(double)));

    setLayout(layout);
    setKeyboardTracking(false);
}

//---------------------------------------------------------------------------
ParamCharWidget::~ParamCharWidget()
{
}

//---------------------------------------------------------------------------
ito::Param ParamCharWidget::param() const
{
    Q_D(const ParamCharWidget);
    return d->m_param;
}

//---------------------------------------------------------------------------
void ParamCharWidget::setParam(const ito::Param &param, bool forceValueChanged /*= false*/)
{
    Q_D(ParamCharWidget);

    if (param.getType() == ito::ParamBase::Char)
    {
        const ito::CharMeta *metaOld = d->m_param.getMetaT<const ito::CharMeta>(); //always != NULL
        const ito::CharMeta *metaNew = param.getMetaT<const ito::CharMeta>();

        bool valChanged = forceValueChanged || (param != d->m_param);
        bool metaChanged = ((metaNew && (*metaOld != *metaNew)) || \
                            (!metaNew));

        if (valChanged || metaChanged)
        {
            bool check = metaNew && ((metaNew->getRepresentation() == ito::ParamMeta::Boolean) || \
                         (metaNew->getMin() == 0 && metaNew->getMax() == 1 && metaNew->getStepSize() == 1));
            bool slider = metaNew && ((metaNew->getRepresentation() == ito::ParamMeta::Linear)  || \
                         (metaNew->getRepresentation() == ito::ParamMeta::Logarithmic));

            if (metaChanged)
            {
                d->m_pCheckBox->setVisible(check);
                d->m_pSliderWidget->setVisible(!check && slider);
                d->m_pSpinBox->setVisible(!check && !slider);
            }

            if (check) //metaNew always valid
            {
                if (valChanged)
                {
                    d->m_pCheckBox->blockCheckBoxSignals(true);
                    d->m_pCheckBox->setChecked(param.getVal<char>() > 0);
                    d->m_pCheckBox->blockCheckBoxSignals(false);
                }
            }
            else if (slider) //metaNew always valid
            {
                if (metaChanged)
                {
                    d->m_pSliderWidget->setSuffix(metaNew->getUnit().empty() ? "" : QString(" %1").arg(metaNew->getUnit().data()));
                    d->m_pSliderWidget->setRange(metaNew->getMin(), metaNew->getMax());
                    d->m_pSliderWidget->setSingleStep(metaNew->getStepSize());
                }

                if (valChanged)
                {
                    d->m_pSliderWidget->setValue(param.getVal<char>());
                }
            }
            else
            {
                if (metaChanged)
                {
                    if (metaNew)
                    {
                        d->m_pSpinBox->setSuffix(metaNew->getUnit().empty() ? "" : QString(" %1").arg(metaNew->getUnit().data()));
                        d->m_pSpinBox->setRange(metaNew->getMin(), metaNew->getMax());
                        d->m_pSpinBox->setSingleStep(metaNew->getStepSize());
                    }
                    else
                    {
                        d->m_pSpinBox->setSuffix("");
                        d->m_pSpinBox->setRange(std::numeric_limits<char>::min(), std::numeric_limits<char>::max());
                        d->m_pSpinBox->setSingleStep(1);
                    }
                }

                if (valChanged)
                {
                    d->m_pSpinBox->setValue(param.getVal<char>());
                }
            }

            d->m_param = param;

            if (metaNew == NULL)
            {
                d->m_param.setMeta(new ito::CharMeta(std::numeric_limits<char>::min(), std::numeric_limits<char>::max(), 1), true);
            }

            if (valChanged)
            {
                emit valueChanged(param.getVal<char>());
            }
        }
    }
}

//---------------------------------------------------------------------------
bool ParamCharWidget::keyboardTracking() const
{
    Q_D(const ParamCharWidget);
    return d->m_pSpinBox->keyboardTracking();
}

//---------------------------------------------------------------------------
void ParamCharWidget::setKeyboardTracking(bool tracking)
{
    Q_D(ParamCharWidget);
    d->m_pSpinBox->setKeyboardTracking(tracking);
    d->m_pSliderWidget->spinBox()->setKeyboardTracking(tracking);
}

//---------------------------------------------------------------------------
char ParamCharWidget::value() const
{
    Q_D(const ParamCharWidget);
    return d->m_param.getVal<char>();
}

//---------------------------------------------------------------------------
void ParamCharWidget::setValue(char value)
{
    Q_D(ParamCharWidget);

    if (d->m_param.getVal<char>() != value)
    {
        if (d->m_pCheckBox->isVisible())
        {
            d->m_pCheckBox->setChecked(value > 0);
        }
        else if (d->m_pSliderWidget->isVisible())
        {
            d->m_pSliderWidget->setValue(value);
        }
        else
        {
            d->m_pSpinBox->setValue(value);
        }

        d->m_param.setVal<char>(value);

        emit valueChanged(value);
    }
}

//---------------------------------------------------------------------------
ito::CharMeta ParamCharWidget::meta() const
{
    Q_D(const ParamCharWidget);
    return *(d->m_param.getMetaT<ito::CharMeta>());
}

//---------------------------------------------------------------------------
void ParamCharWidget::setMeta(const ito::CharMeta &meta)
{
    Q_D(ParamCharWidget);
    ito::Param p = d->m_param;
    p.setMeta(new ito::CharMeta(meta), true);
    setParam(p);
}

} //end namespace ito

#include "moc_paramCharWidget.cpp"
#include "paramCharWidget.moc"
