/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    This file is a port and modified version of the 
    CTK Common Toolkit (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QDebug>
#include <QMouseEvent>
#include <QWeakPointer>

// CTK includes
#include "rangeWidget.h"
#include "valueProxy.h"
#include "ui_RangeWidget.h"

// STD includes
#include <cmath>
#include <limits>

//-----------------------------------------------------------------------------
class RangeWidgetPrivate: public Ui_RangeWidget
{
  Q_DECLARE_PUBLIC(RangeWidget);
protected:
  RangeWidget* const q_ptr;
public:
  RangeWidgetPrivate(RangeWidget& object);
  void connectSlider();

  void updateSpinBoxWidth();
  int synchronizedSpinBoxWidth()const;
  void synchronizeSiblingSpinBox(int newWidth);
  bool equal(double v1, double v2)const;
  void relayout();

  bool          Tracking;
  bool          Changing;
  bool          SettingSliderRange;
  bool          BlockSliderUpdate;
  double        MinimumValueBeforeChange;
  double        MaximumValueBeforeChange;
  bool          AutoSpinBoxWidth;
  Qt::Alignment SpinBoxAlignment;
  QWeakPointer<ValueProxy> Proxy;
};

// --------------------------------------------------------------------------
bool RangeWidgetPrivate::equal(double v1, double v2)const
{
  if (v1 == v2)
    {// don't bother computing difference as it could fail for infinity numbers
    return true;
    }
  if (v1 != v1 && v2 != v2)
    {// NaN check
    return true;
    }
  return qAbs(v1 - v2) < pow(10., -this->MinimumSpinBox->decimals());
}

// --------------------------------------------------------------------------
RangeWidgetPrivate::RangeWidgetPrivate(RangeWidget& object)
  :q_ptr(&object)
{
  this->Tracking = true;
  this->Changing = false;
  this->SettingSliderRange = false;
  this->BlockSliderUpdate = false;
  this->MinimumValueBeforeChange = 0.;
  this->MaximumValueBeforeChange = 0.;
  this->AutoSpinBoxWidth = true;
  this->SpinBoxAlignment = Qt::AlignVCenter;
}

// --------------------------------------------------------------------------
void RangeWidgetPrivate::connectSlider()
{
  Q_Q(RangeWidget);
  QObject::connect(this->Slider, SIGNAL(valuesChanged(double,double)),
                   q, SLOT(changeValues(double,double)));
  QObject::connect(this->Slider, SIGNAL(minimumValueChanged(double)),
                   q, SLOT(changeMinimumValue(double)));
  QObject::connect(this->Slider, SIGNAL(maximumValueChanged(double)),
                   q, SLOT(changeMaximumValue(double)));

  QObject::connect(this->MinimumSpinBox, SIGNAL(valueChanged(double)),
                   q, SLOT(setSliderValues()));
  QObject::connect(this->MaximumSpinBox, SIGNAL(valueChanged(double)),
                   q, SLOT(setSliderValues()));
  QObject::connect(this->MinimumSpinBox, SIGNAL(valueChanged(double)),
                   q, SLOT(setMinimumToMaximumSpinBox(double)));
  QObject::connect(this->MaximumSpinBox, SIGNAL(valueChanged(double)),
                   q, SLOT(setMaximumToMinimumSpinBox(double)));
  QObject::connect(this->MinimumSpinBox, SIGNAL(decimalsChanged(int)),
                   q, SLOT(setDecimals(int)));
  QObject::connect(this->MaximumSpinBox, SIGNAL(decimalsChanged(int)),
                   q, SLOT(setDecimals(int)));

  QObject::connect(this->Slider, SIGNAL(sliderPressed()),
                   q, SLOT(startChanging()));
  QObject::connect(this->Slider, SIGNAL(sliderReleased()),
                   q, SLOT(stopChanging()));
  QObject::connect(this->Slider, SIGNAL(rangeChanged(double,double)),
                   q, SLOT(onSliderRangeChanged(double,double)));
}

// --------------------------------------------------------------------------
void RangeWidgetPrivate::updateSpinBoxWidth()
{
  int spinBoxWidth = this->synchronizedSpinBoxWidth();
  if (this->AutoSpinBoxWidth)
    {
    this->MinimumSpinBox->setMinimumWidth(spinBoxWidth);
    this->MaximumSpinBox->setMinimumWidth(spinBoxWidth);
    }
  else
    {
    this->MinimumSpinBox->setMinimumWidth(0);
    this->MaximumSpinBox->setMinimumWidth(0);
    }
  this->synchronizeSiblingSpinBox(spinBoxWidth);
}

// --------------------------------------------------------------------------
int RangeWidgetPrivate::synchronizedSpinBoxWidth()const
{
  Q_Q(const RangeWidget);
  //Q_ASSERT(this->MinimumSpinBox->sizeHint() == this->MaximumSpinBox->sizeHint());
  int maxWidth = qMax(this->MinimumSpinBox->sizeHint().width(),
                      this->MaximumSpinBox->sizeHint().width());
  if (!q->parent())
    {
    return maxWidth;
    }
  QList<RangeWidget*> siblings =
    q->parent()->findChildren<RangeWidget*>();
  foreach(RangeWidget* sibling, siblings)
    {
    maxWidth = qMax(maxWidth, qMax(sibling->d_func()->MaximumSpinBox->sizeHint().width(),
                                   sibling->d_func()->MaximumSpinBox->sizeHint().width()));
    }
  return maxWidth;
}

// --------------------------------------------------------------------------
void RangeWidgetPrivate::synchronizeSiblingSpinBox(int width)
{
  Q_Q(const RangeWidget);
  QList<RangeWidget*> siblings =
    q->parent()->findChildren<RangeWidget*>();
  foreach(RangeWidget* sibling, siblings)
    {
    if (sibling != q && sibling->isAutoSpinBoxWidth())
      {
      sibling->d_func()->MinimumSpinBox->setMinimumWidth(width);
      sibling->d_func()->MaximumSpinBox->setMinimumWidth(width);
      }
    }
}

// --------------------------------------------------------------------------
void RangeWidgetPrivate::relayout()
{
  this->GridLayout->removeWidget(this->MinimumSpinBox);
  this->GridLayout->removeWidget(this->MaximumSpinBox);
  this->GridLayout->removeWidget(this->Slider);
  if (this->SpinBoxAlignment & Qt::AlignTop)
    {
    this->GridLayout->addWidget(this->MinimumSpinBox,0,0);
    this->GridLayout->addWidget(this->MaximumSpinBox,0,2);
    this->GridLayout->addWidget(this->Slider,1,0,1,3);
    }
  else if (this->SpinBoxAlignment & Qt::AlignBottom)
    {
    this->GridLayout->addWidget(this->MinimumSpinBox,1,0);
    this->GridLayout->addWidget(this->MaximumSpinBox,1,2);
    this->GridLayout->addWidget(this->Slider,0, 0, 1, 3);
    }
  else if (this->SpinBoxAlignment & Qt::AlignRight)
    {
    this->GridLayout->addWidget(this->Slider, 0, 0);
    this->GridLayout->addWidget(this->MinimumSpinBox,0,1);
    this->GridLayout->addWidget(this->MaximumSpinBox,0,2);
    }
  else if (this->SpinBoxAlignment & Qt::AlignLeft)
    {
    this->GridLayout->addWidget(this->MinimumSpinBox,0,0);
    this->GridLayout->addWidget(this->MaximumSpinBox,0,1);
    this->GridLayout->addWidget(this->Slider, 0, 2);
    }
  else // Qt::AlignVCenter (or any other bad alignment)
    {
    this->GridLayout->addWidget(this->MinimumSpinBox,0,0);
    this->GridLayout->addWidget(this->Slider,0,1);
    this->GridLayout->addWidget(this->MaximumSpinBox,0,2);
    }
}

// --------------------------------------------------------------------------
RangeWidget::RangeWidget(QWidget* _parent) : Superclass(_parent)
  , d_ptr(new RangeWidgetPrivate(*this))
{
  Q_D(RangeWidget);
  
  d->setupUi(this);

  d->MinimumSpinBox->setRange(d->Slider->minimum(), d->Slider->maximum());
  d->MaximumSpinBox->setRange(d->Slider->minimum(), d->Slider->maximum());
  d->MinimumSpinBox->setValue(d->Slider->minimumValue());
  d->MaximumSpinBox->setValue(d->Slider->maximumValue());
  
  d->connectSlider();

  d->MinimumSpinBox->installEventFilter(this);
  d->MaximumSpinBox->installEventFilter(this);
}

// --------------------------------------------------------------------------
RangeWidget::~RangeWidget()
{
}

// --------------------------------------------------------------------------
double RangeWidget::minimum()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->MinimumSpinBox->minimum(),d->Slider->minimum()));
  return d->Slider->minimum();
}

// --------------------------------------------------------------------------
double RangeWidget::maximum()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->MaximumSpinBox->maximum(), d->Slider->maximum()));
  return d->Slider->maximum();
}

// --------------------------------------------------------------------------
void RangeWidget::setMinimum(double min)
{
  Q_D(RangeWidget);
  bool blocked = d->MinimumSpinBox->blockSignals(true);
  blocked = d->MaximumSpinBox->blockSignals(true);
  d->MinimumSpinBox->setMinimum(min);
  d->MaximumSpinBox->setMinimum(min);
  d->MinimumSpinBox->blockSignals(blocked);
  d->MaximumSpinBox->blockSignals(blocked);
  // SpinBox can truncate min (depending on decimals).
  // use Spinbox's min to set Slider's min
  d->SettingSliderRange = true;
  d->Slider->setMinimum(d->MinimumSpinBox->minimum());
  d->SettingSliderRange = false;
  Q_ASSERT(d->equal(d->MinimumSpinBox->minimum(),d->Slider->minimum()));
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
void RangeWidget::setMaximum(double max)
{
  Q_D(RangeWidget);
  bool blocked = d->MinimumSpinBox->blockSignals(true);
  blocked = d->MaximumSpinBox->blockSignals(true);
  d->MinimumSpinBox->setMaximum(max);
  d->MaximumSpinBox->setMaximum(max);
  d->MinimumSpinBox->blockSignals(blocked);
  d->MaximumSpinBox->blockSignals(blocked);
  // SpinBox can truncate max (depending on decimals).
  // use Spinbox's max to set Slider's max
  d->SettingSliderRange = true;
  d->Slider->setMaximum(d->MaximumSpinBox->maximum());
  d->SettingSliderRange = false;
  Q_ASSERT(d->equal(d->MaximumSpinBox->maximum(), d->Slider->maximum()));
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
void RangeWidget::setRange(double min, double max)
{
  Q_D(RangeWidget);

  double oldMin = d->MinimumSpinBox->minimum();
  double oldMax = d->MaximumSpinBox->maximum();
  bool blocked = d->MinimumSpinBox->blockSignals(true);
  d->MinimumSpinBox->setRange(qMin(min,max), qMax(min,max));
  d->MinimumSpinBox->blockSignals(blocked);
  blocked = d->MaximumSpinBox->blockSignals(true);
  d->MaximumSpinBox->setRange(qMin(min,max), qMax(min,max));
  d->MaximumSpinBox->blockSignals(blocked);
  // SpinBox can truncate the range (depending on decimals).
  // use Spinbox's range to set Slider's range
  d->SettingSliderRange = true;
  d->Slider->setRange(d->MinimumSpinBox->minimum(), d->MaximumSpinBox->maximum());
  d->SettingSliderRange = false;
  Q_ASSERT(d->equal(d->MinimumSpinBox->minimum(), d->Slider->minimum()));
  Q_ASSERT(d->equal(d->MaximumSpinBox->maximum(), d->Slider->maximum()));
  Q_ASSERT(d->equal(d->Slider->minimumValue(), d->MinimumSpinBox->value()));
  Q_ASSERT(d->equal(d->Slider->maximumValue(), d->MaximumSpinBox->value()));
  d->updateSpinBoxWidth();
  if (oldMin != d->MinimumSpinBox->minimum() ||
      oldMax != d->MaximumSpinBox->maximum())
    {
    emit rangeChanged(d->MinimumSpinBox->minimum(), d->MaximumSpinBox->maximum());
    }
}

// --------------------------------------------------------------------------
void RangeWidget::range(double range[2])const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->MinimumSpinBox->maximum(), d->Slider->minimum()));
  Q_ASSERT(d->equal(d->MaximumSpinBox->maximum(), d->Slider->maximum()));
  range[0] = d->Slider->minimum();
  range[1] = d->Slider->maximum();
}

// --------------------------------------------------------------------------
void RangeWidget::onSliderRangeChanged(double min, double max)
{
  Q_D(RangeWidget);
  if (!d->SettingSliderRange)
    {
    this->setRange(min, max);
    }
}

/*
// --------------------------------------------------------------------------
double RangeWidget::sliderPosition()const
{
  return d->Slider->sliderPosition();
}

// --------------------------------------------------------------------------
void RangeWidget::setSliderPosition(double position)
{
  d->Slider->setSliderPosition(position);
}
*/
/*
// --------------------------------------------------------------------------
double RangeWidget::previousSliderPosition()
{
  return d->Slider->previousSliderPosition();
}
*/

// --------------------------------------------------------------------------
void RangeWidget::values(double &minValue, double &maxValue)const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->Slider->minimumValue(), d->MinimumSpinBox->value()));
  Q_ASSERT(d->equal(d->Slider->maximumValue(), d->MaximumSpinBox->value()));
  minValue = d->Changing ? d->MinimumValueBeforeChange : d->Slider->minimumValue();
  maxValue = d->Changing ? d->MaximumValueBeforeChange : d->Slider->maximumValue();
}

// --------------------------------------------------------------------------
double RangeWidget::minimumValue()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->Slider->minimumValue(), d->MinimumSpinBox->value()));
  const double minValue =
    d->Changing ? d->MinimumValueBeforeChange : d->Slider->minimumValue();
  return minValue;
}

// --------------------------------------------------------------------------
double RangeWidget::maximumValue()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->Slider->maximumValue(), d->MaximumSpinBox->value()));
  const double maxValue =
    d->Changing ? d->MaximumValueBeforeChange : d->Slider->maximumValue();
  return maxValue;
}

// --------------------------------------------------------------------------
void RangeWidget::setMinimumValue(double _value)
{
  Q_D(RangeWidget);
  // disable the tracking temporally to emit the
  // signal valueChanged if changeValue() is called
  bool isChanging = d->Changing;
  d->Changing = false;
  d->MinimumSpinBox->setValue(_value);
  Q_ASSERT(d->equal(d->Slider->minimumValue(), d->MinimumSpinBox->value()));
  // restore the prop
  d->Changing = isChanging;
}

// --------------------------------------------------------------------------
void RangeWidget::setMaximumValue(double _value)
{
  Q_D(RangeWidget);
  // disable the tracking temporally to emit the
  // signal valueChanged if changeValue() is called
  bool isChanging = d->Changing;
  d->Changing = false;
  d->MaximumSpinBox->setValue(_value);
  Q_ASSERT(d->equal(d->Slider->maximumValue(), d->MaximumSpinBox->value()));
  // restore the prop
  d->Changing = isChanging;
}

// --------------------------------------------------------------------------
void RangeWidget::setValues(double newMinimumValue, double newMaximumValue)
{
  Q_D(RangeWidget);
  if (newMinimumValue > newMaximumValue)
    {
    qSwap(newMinimumValue, newMaximumValue);
    }
  // This test must take into account NaN values
  const bool minimumFirst = !(newMinimumValue > this->maximumValue());

  // disable the tracking temporally to emit the
  // signal valueChanged if changeValue() is called
  bool isChanging = d->Changing;
  d->Changing = false;
  // \todo: setting the spinbox separately is currently firing 2 signals and
  // between the signals, the state of the widget is inconsistent.
  bool wasBlocking = d->BlockSliderUpdate;
  d->BlockSliderUpdate = true;
  if (minimumFirst)
    {
    d->MinimumSpinBox->setValue(newMinimumValue);
    d->MaximumSpinBox->setValue(newMaximumValue);
    }
  else
    {
    d->MaximumSpinBox->setValue(newMaximumValue);
    d->MinimumSpinBox->setValue(newMinimumValue);
    }
  d->BlockSliderUpdate = wasBlocking;
  this->setSliderValues();

  Q_ASSERT(d->equal(d->Slider->minimumValue(), d->MinimumSpinBox->value()));
  Q_ASSERT(d->equal(d->Slider->maximumValue(), d->MaximumSpinBox->value()));
  // restore the prop
  d->Changing = isChanging;
}

// --------------------------------------------------------------------------
void RangeWidget::setSliderValues()
{
  Q_D(RangeWidget);
  if (d->BlockSliderUpdate)
    {
    return;
    }
  d->Slider->setValues(d->MinimumSpinBox->value(), d->MaximumSpinBox->value());
}

// --------------------------------------------------------------------------
void RangeWidget::setMinimumToMaximumSpinBox(double minimum)
{
  Q_D(RangeWidget);
  if (minimum != minimum) // NaN check
    {
    return;
    }
  d->MaximumSpinBox->setRange(minimum, d->Slider->maximum());
}

// --------------------------------------------------------------------------
void RangeWidget::setMaximumToMinimumSpinBox(double maximum)
{
  Q_D(RangeWidget);
  if (maximum != maximum) // NaN check
    {
    return;
    }
  d->MinimumSpinBox->setRange(d->Slider->minimum(), maximum);
}

// --------------------------------------------------------------------------
void RangeWidget::startChanging()
{
  Q_D(RangeWidget);
  if (d->Tracking)
    {
    return;
    }
  d->Changing = true;
  d->MinimumValueBeforeChange = this->minimumValue();
  d->MaximumValueBeforeChange = this->maximumValue();
}

// --------------------------------------------------------------------------
void RangeWidget::stopChanging()
{
  Q_D(RangeWidget);
  if (d->Tracking)
    {
    return;
    }
  d->Changing = false;
  bool emitMinValChanged = qAbs(this->minimumValue() - d->MinimumValueBeforeChange) > (this->singleStep() * 0.000000001);
  bool emitMaxValChanged = qAbs(this->maximumValue() - d->MaximumValueBeforeChange) > (this->singleStep() * 0.000000001);
  if (emitMinValChanged || emitMaxValChanged)
    {
    // emit the valuesChanged signal first
    emit this->valuesChanged(this->minimumValue(), this->maximumValue());
    }
  if (emitMinValChanged)
    {
    emit this->minimumValueChanged(this->minimumValue());
    }
  if (emitMaxValChanged)
    {
    emit this->maximumValueChanged(this->maximumValue());
    }
}

// --------------------------------------------------------------------------
void RangeWidget::changeMinimumValue(double newValue)
{
  Q_D(RangeWidget);
  //if (d->Tracking)
    {
    emit this->minimumValueIsChanging(newValue);
    }
  if (!d->Changing)
    {
    emit this->minimumValueChanged(newValue);
    }
}

// --------------------------------------------------------------------------
void RangeWidget::changeMaximumValue(double newValue)
{
  Q_D(RangeWidget);
  //if (d->Tracking)
    {
    emit this->maximumValueIsChanging(newValue);
    }
  if (!d->Changing)
    {
    emit this->maximumValueChanged(newValue);
    }
}

// --------------------------------------------------------------------------
void RangeWidget::changeValues(double newMinValue, double newMaxValue)
{
  Q_D(RangeWidget);
  bool wasBlocking = d->BlockSliderUpdate;
  d->BlockSliderUpdate = true;
  d->MinimumSpinBox->setValue(newMinValue);
  d->MaximumSpinBox->setValue(newMaxValue);
  d->BlockSliderUpdate = wasBlocking;
  if (!d->Changing)
    {
    emit this->valuesChanged(newMinValue, newMaxValue);
    }
}

// --------------------------------------------------------------------------
bool RangeWidget::eventFilter(QObject *obj, QEvent *event)
 {
   if (event->type() == QEvent::MouseButtonPress)
     {
     QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
     if (mouseEvent->button() & Qt::LeftButton)
       {
       this->startChanging();
       }
     }
   else if (event->type() == QEvent::MouseButtonRelease) 
     {
     QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
     if (mouseEvent->button() & Qt::LeftButton)
       {
       // here we might prevent RangeWidget::stopChanging
       // from sending a valueChanged() event as the spinbox might
       // send a valueChanged() after eventFilter() is done.
       this->stopChanging();
       }
     } 
   // standard event processing
   return this->Superclass::eventFilter(obj, event);
 }

// --------------------------------------------------------------------------
double RangeWidget::singleStep()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->equal(d->Slider->singleStep(), d->MinimumSpinBox->singleStep()));
  Q_ASSERT(d->equal(d->Slider->singleStep(), d->MaximumSpinBox->singleStep()));
  return d->Slider->singleStep();
}

// --------------------------------------------------------------------------
void RangeWidget::setSingleStep(double step)
{
  Q_D(RangeWidget);
  if (!d->Slider->isValidStep(step))
    {
    qWarning() << "RangeWidget::setSingleStep(" << step << ")"
               << "is outside valid bounds";
    return;
    }
  d->MinimumSpinBox->setSingleStep(step);
  d->MaximumSpinBox->setSingleStep(step);
  d->Slider->setSingleStep(d->MinimumSpinBox->singleStep());
  Q_ASSERT(d->equal(d->Slider->singleStep(), d->MinimumSpinBox->singleStep()));
  Q_ASSERT(d->equal(d->Slider->singleStep(), d->MaximumSpinBox->singleStep()));
  Q_ASSERT(d->equal(d->Slider->minimumValue(), d->MinimumSpinBox->value()));
  Q_ASSERT(d->equal(d->Slider->maximumValue(), d->MaximumSpinBox->value()));
}

// --------------------------------------------------------------------------
int RangeWidget::decimals()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->MinimumSpinBox->decimals() == d->MaximumSpinBox->decimals());
  return d->MinimumSpinBox->decimals();
}

// --------------------------------------------------------------------------
void RangeWidget::setDecimals(int newDecimals)
{
  Q_D(RangeWidget);
  d->MinimumSpinBox->setDecimals(newDecimals);
  d->MaximumSpinBox->setDecimals(newDecimals);
  // The number of decimals can change the range values
  // i.e. 50.55 with 2 decimals -> 51 with 0 decimals
  // As the SpinBox range change doesn't fire signals, 
  // we have to do the synchronization manually here
  d->Slider->setRange(d->MinimumSpinBox->minimum(), d->MaximumSpinBox->maximum());
}

// --------------------------------------------------------------------------
QString RangeWidget::prefix()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->MinimumSpinBox->prefix() == d->MaximumSpinBox->prefix());
  return d->MinimumSpinBox->prefix();
}

// --------------------------------------------------------------------------
void RangeWidget::setPrefix(const QString& newPrefix)
{
  Q_D(RangeWidget);
  d->MinimumSpinBox->setPrefix(newPrefix);
  d->MaximumSpinBox->setPrefix(newPrefix);
}

// --------------------------------------------------------------------------
QString RangeWidget::suffix()const
{
  Q_D(const RangeWidget);
 Q_ASSERT(d->MinimumSpinBox->suffix() == d->MaximumSpinBox->suffix());
  return d->MinimumSpinBox->suffix();
}

// --------------------------------------------------------------------------
void RangeWidget::setSuffix(const QString& newSuffix)
{
  Q_D(RangeWidget);
  d->MinimumSpinBox->setSuffix(newSuffix);
  d->MaximumSpinBox->setSuffix(newSuffix);
}

// --------------------------------------------------------------------------
double RangeWidget::tickInterval()const
{
  Q_D(const RangeWidget);
  return d->Slider->tickInterval();
}

// --------------------------------------------------------------------------
void RangeWidget::setTickInterval(double ti)
{ 
  Q_D(RangeWidget);
  d->Slider->setTickInterval(ti);
}

// -------------------------------------------------------------------------
void RangeWidget::reset()
{
  this->setValues(this->minimum(), this->maximum());
}

// -------------------------------------------------------------------------
void RangeWidget::setSpinBoxAlignment(Qt::Alignment alignment)
{
  Q_D(RangeWidget);
  if (d->SpinBoxAlignment == alignment)
    {
    return;
    }
  d->SpinBoxAlignment = alignment;
  d->relayout();
}

// -------------------------------------------------------------------------
Qt::Alignment RangeWidget::spinBoxAlignment()const
{
  Q_D(const RangeWidget);
  return d->SpinBoxAlignment;
}

// -------------------------------------------------------------------------
void RangeWidget::setSpinBoxTextAlignment(Qt::Alignment alignment)
{
  Q_D(RangeWidget);
  d->MinimumSpinBox->setAlignment(alignment);
  d->MaximumSpinBox->setAlignment(alignment);
}

// -------------------------------------------------------------------------
Qt::Alignment RangeWidget::spinBoxTextAlignment()const
{
  Q_D(const RangeWidget);
  Q_ASSERT(d->MinimumSpinBox->alignment() == d->MaximumSpinBox->alignment());
  return d->MinimumSpinBox->alignment();
}

// -------------------------------------------------------------------------
void RangeWidget::setTracking(bool enable)
{
  Q_D(RangeWidget);
  d->Tracking = enable;
}

// -------------------------------------------------------------------------
bool RangeWidget::hasTracking()const
{
  Q_D(const RangeWidget);
  return d->Tracking;
}

// -------------------------------------------------------------------------
bool RangeWidget::isAutoSpinBoxWidth()const
{
  Q_D(const RangeWidget);
  return d->AutoSpinBoxWidth;
}

// -------------------------------------------------------------------------
void RangeWidget::setAutoSpinBoxWidth(bool autoWidth)
{
  Q_D(RangeWidget);
  d->AutoSpinBoxWidth = autoWidth;
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
bool RangeWidget::symmetricMoves()const
{
  Q_D(const RangeWidget);
  return d->Slider->symmetricMoves();
}

// --------------------------------------------------------------------------
void RangeWidget::setSymmetricMoves(bool symmetry)
{
  Q_D(RangeWidget);
  d->Slider->setSymmetricMoves(symmetry);
}

// -------------------------------------------------------------------------
DoubleRangeSlider* RangeWidget::slider()const
{
  Q_D(const RangeWidget);
  return d->Slider;
}

// -------------------------------------------------------------------------
void RangeWidget::setSlider(DoubleRangeSlider* slider)
{
  Q_D(RangeWidget);

  slider->setOrientation(d->Slider->orientation());
  slider->setRange(d->Slider->minimum(), d->Slider->maximum());
  slider->setValues(d->Slider->minimumValue(), d->Slider->maximumValue());
  slider->setSingleStep(d->Slider->singleStep());
  slider->setTracking(d->Slider->hasTracking());
  slider->setTickInterval(d->Slider->tickInterval());

  delete d->Slider;
  d->Slider = slider;

  d->connectSlider();

  d->relayout();
}

// -------------------------------------------------------------------------
DoubleSpinBox* RangeWidget::minimumSpinBox()const
{
  Q_D(const RangeWidget);
  return d->MinimumSpinBox;
}

// -------------------------------------------------------------------------
DoubleSpinBox* RangeWidget::maximumSpinBox()const
{
  Q_D(const RangeWidget);
  return d->MaximumSpinBox;
}

//----------------------------------------------------------------------------
void RangeWidget::setValueProxy(ValueProxy* proxy)
{
  Q_D(RangeWidget);
  if (proxy == d->Proxy.data())
    {
    return;
    }

  this->onValueProxyAboutToBeModified();

  if (d->Proxy)
    {
    disconnect(d->Proxy.data(), SIGNAL(proxyAboutToBeModified()),
               this, SLOT(onValueProxyAboutToBeModified()));
    disconnect(d->Proxy.data(), SIGNAL(proxyModified()),
               this, SLOT(onValueProxyModified()));
    }

  d->Proxy = proxy;

  if (d->Proxy)
    {
    connect(d->Proxy.data(), SIGNAL(proxyAboutToBeModified()),
            this, SLOT(onValueProxyAboutToBeModified()));
    }

  this->slider()->setValueProxy(proxy);
  this->minimumSpinBox()->setValueProxy(proxy);
  this->maximumSpinBox()->setValueProxy(proxy);

  if (d->Proxy)
    {
    connect(d->Proxy.data(), SIGNAL(proxyModified()),
            this, SLOT(onValueProxyModified()));
    }

  this->onValueProxyModified();
}

//----------------------------------------------------------------------------
ValueProxy* RangeWidget::valueProxy() const
{
  Q_D(const RangeWidget);
  return d->Proxy.data();
}

//-----------------------------------------------------------------------------
void RangeWidget::onValueProxyAboutToBeModified()
{
}

//-----------------------------------------------------------------------------
void RangeWidget::onValueProxyModified()
{
}