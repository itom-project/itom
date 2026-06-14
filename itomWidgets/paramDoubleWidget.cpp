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

#include "paramDoubleWidget.h"

#include <qspinbox.h>
#include <qcheckbox.h>
#include "sliderWidget.h"
#include "doubleSpinBox.h"
#include <qlayout.h>
#include "qtpropertybrowserutils_p.h"

namespace ito
{

class ParamDoubleWidgetPrivate
{
public:
    ParamDoubleWidget *q_ptr;
    ParamDoubleWidgetPrivate() {}

    QtBoolEdit *m_pCheckBox;
    DoubleSpinBox *m_pSpinBox;
    SliderWidget *m_pSliderWidget;
    ito::Param m_param;

    //--------------------------------------------------
    void slotValueChanged(double value)
    {
        //value changed signal from slider widget
        if (round(value))
        {
            bool blocked = m_pSliderWidget->signalsBlocked();
            m_pSliderWidget->blockSignals(true);
            m_pSliderWidget->setValue(value);
            m_pSliderWidget->blockSignals(blocked);

            blocked = m_pSpinBox->signalsBlocked();
            m_pSpinBox->blockSignals(true);
            m_pSpinBox->setValue(value);
            m_pSpinBox->blockSignals(blocked);
        }

        emit q_ptr->valueChanged(value);
    }

    //--------------------------------------------------
    void slotChecked(bool checked)
    {
        //value changed signal from checkbox
        emit q_ptr->valueChanged(checked ? 1 : 0);
    }

private:
    //--------------------------------------------------
    bool round(double &value)
    {
        //check if value fits to optional step size of meta information.
        //If it does not fit, round it to the next possible value and return true (changed)
        const ito::DoubleMeta *meta = m_param.getMetaT<const ito::DoubleMeta>();
        ito::float64 step = meta->getStepSize();

        if (qFuzzyCompare(step, 0.0))
        {
            return true;
        }
        else
        {
            ito::float64 minimum = meta->getMin();
            if (qFuzzyCompare(std::floor((value - minimum) / step), (value - minimum) / step))
            {
                value = qBound(minimum, minimum + step * qRound(float(value - minimum) / step), meta->getMax());
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    Q_DISABLE_COPY(ParamDoubleWidgetPrivate);
};

//---------------------------------------------------------------------------
ParamDoubleWidget::ParamDoubleWidget(QWidget *parent /*= NULL*/) :
    QWidget(parent),
    d_ptr(new ParamDoubleWidgetPrivate())
{
    d_ptr->q_ptr = this;

    Q_D(ParamDoubleWidget);

    d->m_param = ito::Param("", ito::ParamBase::Double, 0.0, 0.0, 0.0, "");

    QHBoxLayout *layout = new QHBoxLayout();
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);;

    d->m_pCheckBox = new QtBoolEdit(this);
    layout->addWidget(d->m_pCheckBox);
    d->m_pCheckBox->setVisible(false);
    connect(d->m_pCheckBox, SIGNAL(toggled(bool)), this, SLOT(slotChecked(bool)));

    d->m_pSpinBox = new DoubleSpinBox(this);
    layout->addWidget(d->m_pSpinBox);
    d->m_pSpinBox->setVisible(false);
    connect(d->m_pSpinBox, SIGNAL(valueChanged(double)), this, SLOT(slotValueChanged(double)));

    d->m_pSliderWidget = new SliderWidget(this);
    d->m_pSliderWidget->setPopupSlider(true);
    layout->addWidget(d->m_pSliderWidget);
    d->m_pSliderWidget->setVisible(false);
    connect(d->m_pSliderWidget, SIGNAL(valueChanged(double)), this, SLOT(slotValueChanged(double)));

    setLayout(layout);
    setKeyboardTracking(false);
}

//---------------------------------------------------------------------------
ParamDoubleWidget::~ParamDoubleWidget()
{
}

//---------------------------------------------------------------------------
ito::Param ParamDoubleWidget::param() const
{
    Q_D(const ParamDoubleWidget);
    return d->m_param;
}

//---------------------------------------------------------------------------
void ParamDoubleWidget::setParam(const ito::Param &param, bool forceValueChanged /*= false*/)
{
    Q_D(ParamDoubleWidget);

    if (param.getType() == ito::ParamBase::Double)
    {
        const ito::DoubleMeta *metaOld = d->m_param.getMetaT<const ito::DoubleMeta>(); //always != NULL
        const ito::DoubleMeta *metaNew = param.getMetaT<const ito::DoubleMeta>();

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
                    d->m_pCheckBox->setChecked(param.getVal<double>() > 0);
                    d->m_pCheckBox->blockCheckBoxSignals(false);
                }
            }
            else if (slider) //metaNew always valid
            {
                if (metaChanged)
                {
                    d->m_pSliderWidget->setSuffix(metaNew->getUnit().empty() ? "" : QString(" %1").arg(metaNew->getUnit().data()));
                    d->m_pSliderWidget->setRange(metaNew->getMin(), metaNew->getMax());
                    double step = metaNew->getStepSize();
                    if (qFuzzyCompare(metaNew->getStepSize(), 0.0))
                    {
                        step = qMin(1.0, (metaNew->getMax() - metaNew->getMin()) / 100.0);
                    }
                    d->m_pSliderWidget->setSingleStep(step);
                    d->m_pSliderWidget->setDecimals(metaNew->getDisplayPrecision());
                }

                if (valChanged)
                {
                    d->m_pSliderWidget->setValue(param.getVal<double>());
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
                        double step = metaNew->getStepSize();
                        if (qFuzzyCompare(step, 0.0))
                        {
                            step = qMin(1.0, (metaNew->getMax() - metaNew->getMin()) / 100.0);
                        }
                        d->m_pSpinBox->setSingleStep(step);
                        d->m_pSpinBox->setDecimals(metaNew->getDisplayPrecision());
                    }
                    else
                    {
                        d->m_pSpinBox->setSuffix("");
                        d->m_pSpinBox->setRange(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
                        d->m_pSpinBox->setSingleStep(1.0);
                        d->m_pSpinBox->setDecimals(3);
                    }
                }

                if (valChanged)
                {
                    d->m_pSpinBox->setValue(param.getVal<double>());
                }
            }

            d->m_param = param;

            if (metaNew == NULL)
            {
                d->m_param.setMeta(new ito::DoubleMeta(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 1), true);
            }

            if (valChanged)
            {
                emit valueChanged(param.getVal<double>());
            }
        }
    }
}

//---------------------------------------------------------------------------
bool ParamDoubleWidget::keyboardTracking() const
{
    Q_D(const ParamDoubleWidget);
    return d->m_pSliderWidget->hasTracking();
}

//---------------------------------------------------------------------------
void ParamDoubleWidget::setKeyboardTracking(bool tracking)
{
    Q_D(ParamDoubleWidget);
    d->m_pSpinBox->setKeyboardTracking(tracking);
    d->m_pSliderWidget->setTracking(tracking);
}

//---------------------------------------------------------------------------
bool ParamDoubleWidget::hasPopupSlider() const
{
    Q_D(const ParamDoubleWidget);
    return d->m_pSliderWidget->hasPopupSlider();
}

//---------------------------------------------------------------------------
void ParamDoubleWidget::setPopupSlider(bool popup)
{
    Q_D(ParamDoubleWidget);
    d->m_pSliderWidget->setPopupSlider(popup);
}

//---------------------------------------------------------------------------
double ParamDoubleWidget::value() const
{
    Q_D(const ParamDoubleWidget);
    return d->m_param.getVal<double>();
}

//---------------------------------------------------------------------------
void ParamDoubleWidget::setValue(double value)
{
    Q_D(ParamDoubleWidget);

    if (d->m_param.getVal<double>() != value)
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

        d->m_param.setVal<double>(value);

        emit valueChanged(value);
    }
}

//---------------------------------------------------------------------------
ito::DoubleMeta ParamDoubleWidget::meta() const
{
    Q_D(const ParamDoubleWidget);
    return *(d->m_param.getMetaT<ito::DoubleMeta>());
}

//---------------------------------------------------------------------------
void ParamDoubleWidget::setMeta(const ito::DoubleMeta &meta)
{
    Q_D(ParamDoubleWidget);
    ito::Param p = d->m_param;
    p.setMeta(new ito::DoubleMeta(meta), true);
    setParam(p);
}

} //end namespace ito

#include "moc_paramDoubleWidget.cpp"
#include "paramDoubleWidget.moc"
