/*=========================================================================

  Library:   CTK

  Copyright (c) Kitware Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0.txt

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/

// QT includes
#include <QDebug>
#include <QHBoxLayout>
#include <QHelpEvent>
#include <QStyle>
#include <QStyleOptionSlider>
#include <QToolTip>
#include <QPointer>

#include "doubleSlider.h"
#include "valueProxy.h"

// STD includes
#include <limits>

//-----------------------------------------------------------------------------
// slider

//-----------------------------------------------------------------------------
class Slider: public QSlider
{
public:
  Slider(QWidget* parent);
  using QSlider::initStyleOption;
};

//-----------------------------------------------------------------------------
Slider::Slider(QWidget* parent): QSlider(parent)
{
}

//-----------------------------------------------------------------------------
// DoubleSliderPrivate

//-----------------------------------------------------------------------------
class DoubleSliderPrivate
{
  Q_DECLARE_PUBLIC(DoubleSlider);
protected:
  DoubleSlider* const q_ptr;
public:
  DoubleSliderPrivate(DoubleSlider& object);
  int toInt(double value)const;
  double fromInt(int value)const;
  double safeFromInt(int value)const;
  void init();
  void updateOffset(double value);

  Slider*    slider;
  QString       HandleToolTip;
  double      Minimum;
  double      Maximum;
  bool        SettingRange;
  // we should have a Offset and SliderPositionOffset (and MinimumOffset?)
  double      Offset;
  double      SingleStep;
  double      PageStep;
  double      Value;
  /// Converts input value with displayed value
  QPointer<ValueProxy> Proxy;
};

// --------------------------------------------------------------------------
DoubleSliderPrivate::DoubleSliderPrivate(DoubleSlider& object)
  :q_ptr(&object)
{
  this->slider = 0;
  this->Minimum = 0.;
  this->Maximum = 100.;
  this->SettingRange = false;
  this->Offset = 0.;
  this->SingleStep = 1.;
  this->PageStep = 10.;
  this->Value = 0.;
}

// --------------------------------------------------------------------------
void DoubleSliderPrivate::init()
{
  Q_Q(DoubleSlider);
  this->slider = new Slider(q);
  this->slider->installEventFilter(q);
  QHBoxLayout* l = new QHBoxLayout(q);
  l->addWidget(this->slider);
  l->setContentsMargins(0,0,0,0);

  this->Minimum = this->slider->minimum();
  this->Maximum = this->slider->maximum();
  // this->slider->singleStep is always 1
  this->SingleStep = this->slider->singleStep();
  this->PageStep = this->slider->pageStep();
  this->Value = this->slider->value();

  q->connect(this->slider, SIGNAL(valueChanged(int)), q, SLOT(onValueChanged(int)));
  q->connect(this->slider, SIGNAL(sliderMoved(int)), q, SLOT(onSliderMoved(int)));
  q->connect(this->slider, SIGNAL(sliderPressed()), q, SIGNAL(sliderPressed()));
  q->connect(this->slider, SIGNAL(sliderReleased()), q, SIGNAL(sliderReleased()));
  q->connect(this->slider, SIGNAL(rangeChanged(int,int)),
             q, SLOT(onRangeChanged(int,int)));

  q->setSizePolicy(this->slider->sizePolicy());
  q->setAttribute(Qt::WA_WState_OwnSizePolicy, false);
}

// --------------------------------------------------------------------------
int DoubleSliderPrivate::toInt(double doubleValue)const
{
  double tmp = doubleValue / this->SingleStep;
  static const double minInt = std::numeric_limits<int>::min();
  static const double maxInt = std::numeric_limits<int>::max();
#ifndef QT_NO_DEBUG
  static const double maxDouble = std::numeric_limits<double>::max();
  if ( (tmp < minInt || tmp > maxInt) &&
       // If the value is the min or max double, there is no need
       // to warn. It is expected that the number is outside of bounds.
       (doubleValue != -maxDouble && doubleValue != maxDouble) )
    {
    qWarning() << __FUNCTION__ << ": value " << doubleValue
               << " for singleStep " << this->SingleStep
               << " is out of integer bounds !";
    }
#endif
  tmp = qBound(minInt, tmp, maxInt);
  int intValue = qRound(tmp);
  //qDebug() << __FUNCTION__ << doubleValue << tmp << intValue;
  return intValue;
}

// --------------------------------------------------------------------------
double DoubleSliderPrivate::fromInt(int intValue)const
{
  double doubleValue = this->SingleStep * (this->Offset + intValue) ;
  //qDebug() << __FUNCTION__ << intValue << doubleValue;
  return doubleValue;
}

// --------------------------------------------------------------------------
double DoubleSliderPrivate::safeFromInt(int intValue)const
{
  return qBound(this->Minimum, this->fromInt(intValue), this->Maximum);
}

// --------------------------------------------------------------------------
void DoubleSliderPrivate::updateOffset(double value)
{
  this->Offset = (value / this->SingleStep) - this->toInt(value);
}

//-----------------------------------------------------------------------------
// DoubleSlider

// --------------------------------------------------------------------------
DoubleSlider::DoubleSlider(QWidget* _parent) : Superclass(_parent)
  , d_ptr(new DoubleSliderPrivate(*this))
{
  Q_D(DoubleSlider);
  d->init();
}

// --------------------------------------------------------------------------
DoubleSlider::DoubleSlider(Qt::Orientation _orientation, QWidget* _parent)
  : Superclass(_parent)
  , d_ptr(new DoubleSliderPrivate(*this))
{
  Q_D(DoubleSlider);
  d->init();
  this->setOrientation(_orientation);
}

// --------------------------------------------------------------------------
DoubleSlider::~DoubleSlider()
{
}

// --------------------------------------------------------------------------
void DoubleSlider::setMinimum(double min)
{
  this->setRange(min, qMax(min, this->maximum()));
}

// --------------------------------------------------------------------------
void DoubleSlider::setMaximum(double max)
{
  this->setRange(qMin(this->minimum(), max), max);
}

// --------------------------------------------------------------------------
void DoubleSlider::setRange(double newMin, double newMax)
{
  Q_D(DoubleSlider);
  if (d->Proxy)
    {
    newMin = d->Proxy.data()->proxyValueFromValue(newMin);
    newMax = d->Proxy.data()->proxyValueFromValue(newMax);
    }

  if (newMin > newMax)
    {
    qSwap(newMin, newMax);
    }

  double oldMin = d->Minimum;
  double oldMax = d->Maximum;

  d->Minimum = newMin;
  d->Maximum = newMax;

  if (d->Minimum >= d->Value)
    {
    d->updateOffset(d->Minimum);
    }
  if (d->Maximum <= d->Value)
    {
    d->updateOffset(d->Maximum);
    }
  bool wasSettingRange = d->SettingRange;
  d->SettingRange = true;
  d->slider->setRange(d->toInt(newMin), d->toInt(newMax));
  d->SettingRange = wasSettingRange;
  if (!wasSettingRange && (d->Minimum != oldMin || d->Maximum != oldMax))
    {
    emit this->rangeChanged(this->minimum(), this->maximum());
    }
  /// In case QSlider::setRange(...) didn't notify the value
  /// has changed.
  this->setValue(this->value());
}

// --------------------------------------------------------------------------
double DoubleSlider::minimum()const
{
  Q_D(const DoubleSlider);
  double min = d->Minimum;
  double max = d->Maximum;
  if (d->Proxy)
    {
    min = d->Proxy.data()->valueFromProxyValue(min);
    max = d->Proxy.data()->valueFromProxyValue(max);
    }
  return qMin(min, max);
}

// --------------------------------------------------------------------------
double DoubleSlider::maximum()const
{
  Q_D(const DoubleSlider);
  double min = d->Minimum;
  double max = d->Maximum;
  if (d->Proxy)
    {
    min = d->Proxy.data()->valueFromProxyValue(min);
    max = d->Proxy.data()->valueFromProxyValue(max);
    }
  return qMax(min, max);
}

// --------------------------------------------------------------------------
double DoubleSlider::sliderPosition()const
{
  Q_D(const DoubleSlider);
  int intPosition = d->slider->sliderPosition();
  double position = d->safeFromInt(intPosition);
  if (d->Proxy)
    {
    position = d->Proxy.data()->valueFromProxyValue(position);
    }
  return position;
}

// --------------------------------------------------------------------------
void DoubleSlider::setSliderPosition(double newPosition)
{
  Q_D(DoubleSlider);
  if (d->Proxy)
    {
    newPosition = d->Proxy.data()->proxyValueFromValue(newPosition);
    }
  int newIntPosition = d->toInt(newPosition);
  d->slider->setSliderPosition(newIntPosition);
}

// --------------------------------------------------------------------------
double DoubleSlider::value()const
{
  Q_D(const DoubleSlider);
  double val = d->Value;
  if (d->Proxy)
    {
    val = d->Proxy.data()->valueFromProxyValue(val);
    }
  return val;
}

// --------------------------------------------------------------------------
void DoubleSlider::setValue(double newValue)
{
  Q_D(DoubleSlider);
  if (d->Proxy)
    {
    newValue = d->Proxy.data()->proxyValueFromValue(newValue);
    }

  newValue = qBound(d->Minimum, newValue, d->Maximum);
  d->updateOffset(newValue);
  int newIntValue = d->toInt(newValue);
  if (newIntValue != d->slider->value())
    {
    // d->slider will emit a valueChanged signal that is connected to
    // DoubleSlider::onValueChanged
    d->slider->setValue(newIntValue);
    }
  else
    {
    double oldValue = d->Value;
    d->Value = newValue;
    // don't emit a valuechanged signal if the new value is quite
    // similar to the old value.
    if (qAbs(newValue - oldValue) > (d->SingleStep * 0.000000001))
      {
      emit this->valueChanged(this->value());
      }
    }
}

// --------------------------------------------------------------------------
double DoubleSlider::singleStep()const
{
  Q_D(const DoubleSlider);
  double step = d->SingleStep;
  return step;
}

// --------------------------------------------------------------------------
void DoubleSlider::setSingleStep(double newStep)
{
  Q_D(DoubleSlider);
  if (!this->isValidStep(newStep))
    {
    qWarning() << "DoubleSlider::setSingleStep("<< newStep <<")"
               << "is outside of valid bounds.";
    return;
    }
  d->SingleStep = newStep;
  d->updateOffset(d->Value);
  // update the new values of the QSlider
  bool oldBlockSignals = d->slider->blockSignals(true);
  d->slider->setRange(d->toInt(d->Minimum), d->toInt(d->Maximum));
  d->slider->setValue(d->toInt(d->Value));
  d->slider->setPageStep(d->toInt(d->PageStep));
  d->slider->blockSignals(oldBlockSignals);
  Q_ASSERT(qFuzzyCompare(d->Value,d->safeFromInt(d->slider->value())));
}

// --------------------------------------------------------------------------
bool DoubleSlider::isValidStep(double step)const
{
  Q_D(const DoubleSlider);
  if (d->Minimum == d->Maximum)
    {
    return true;
    }
  const double minStep = qMax(d->Maximum / std::numeric_limits<double>::max(),
                              std::numeric_limits<double>::epsilon());
  const double maxStep = qMin(d->Maximum - d->Minimum,
                              static_cast<double>(std::numeric_limits<int>::max()));
  return step >= minStep && step <= maxStep;
}

// --------------------------------------------------------------------------
double DoubleSlider::pageStep()const
{
  Q_D(const DoubleSlider);
  return d->PageStep;
}

// --------------------------------------------------------------------------
void DoubleSlider::setPageStep(double newStep)
{
  Q_D(DoubleSlider);
  d->PageStep = newStep;
  int intPageStep = d->toInt(d->PageStep);
  d->slider->setPageStep(intPageStep);
}

// --------------------------------------------------------------------------
double DoubleSlider::tickInterval()const
{
  Q_D(const DoubleSlider);
  // No need to apply Offset
  double interval = d->SingleStep * d->slider->tickInterval();
  return interval;
}

// --------------------------------------------------------------------------
void DoubleSlider::setTickInterval(double newInterval)
{
  Q_D(DoubleSlider);
  int newIntInterval = d->toInt(newInterval);
  d->slider->setTickInterval(newIntInterval);
}

// --------------------------------------------------------------------------
QSlider::TickPosition DoubleSlider::tickPosition()const
{
  Q_D(const DoubleSlider);
  return d->slider->tickPosition();
}

// --------------------------------------------------------------------------
void DoubleSlider::setTickPosition(QSlider::TickPosition newTickPosition)
{
  Q_D(DoubleSlider);
  d->slider->setTickPosition(newTickPosition);
}

// --------------------------------------------------------------------------
bool DoubleSlider::hasTracking()const
{
  Q_D(const DoubleSlider);
  return d->slider->hasTracking();
}

// --------------------------------------------------------------------------
void DoubleSlider::setTracking(bool enable)
{
  Q_D(DoubleSlider);
  d->slider->setTracking(enable);
}

// --------------------------------------------------------------------------
bool DoubleSlider::invertedAppearance()const
{
  Q_D(const DoubleSlider);
  return d->slider->invertedAppearance();
}

// --------------------------------------------------------------------------
void DoubleSlider::setInvertedAppearance(bool invertedAppearance)
{
  Q_D(DoubleSlider);
  d->slider->setInvertedAppearance(invertedAppearance);
}

// --------------------------------------------------------------------------
bool DoubleSlider::invertedControls()const
{
  Q_D(const DoubleSlider);
  return d->slider->invertedControls();
}

// --------------------------------------------------------------------------
void DoubleSlider::setInvertedControls(bool invertedControls)
{
  Q_D(DoubleSlider);
  d->slider->setInvertedControls(invertedControls);
}

// --------------------------------------------------------------------------
void DoubleSlider::triggerAction( QAbstractSlider::SliderAction action)
{
  Q_D(DoubleSlider);
  d->slider->triggerAction(action);
}

// --------------------------------------------------------------------------
Qt::Orientation DoubleSlider::orientation()const
{
  Q_D(const DoubleSlider);
  return d->slider->orientation();
}

// --------------------------------------------------------------------------
void DoubleSlider::setOrientation(Qt::Orientation newOrientation)
{
  Q_D(DoubleSlider);
  if (this->orientation() == newOrientation)
    {
    return;
    }
  if (!testAttribute(Qt::WA_WState_OwnSizePolicy))
    {
    QSizePolicy sp = this->sizePolicy();
    sp.transpose();
    this->setSizePolicy(sp);
    this->setAttribute(Qt::WA_WState_OwnSizePolicy, false);
    }
  // d->slider will take care of calling updateGeometry
  d->slider->setOrientation(newOrientation);
}

// --------------------------------------------------------------------------
QString DoubleSlider::handleToolTip()const
{
  Q_D(const DoubleSlider);
  return d->HandleToolTip;
}

// --------------------------------------------------------------------------
void DoubleSlider::setHandleToolTip(const QString& toolTip)
{
  Q_D(DoubleSlider);
  d->HandleToolTip = toolTip;
}

// --------------------------------------------------------------------------
void DoubleSlider::onValueChanged(int newValue)
{
  Q_D(DoubleSlider);
  double doubleNewValue = d->safeFromInt(newValue);
/*
  qDebug() << "onValueChanged: " << newValue << "->"<< d->fromInt(newValue+d->Offset)
           << " old: " << d->Value << "->" << d->toInt(d->Value)
           << "offset:" << d->Offset << doubleNewValue;
*/
  if (d->Value == doubleNewValue)
    {
    return;
    }
  d->Value = doubleNewValue;
  emit this->valueChanged(this->value());
}

// --------------------------------------------------------------------------
void DoubleSlider::onSliderMoved(int newPosition)
{
  Q_D(const DoubleSlider);
  emit this->sliderMoved(d->safeFromInt(newPosition));
}

// --------------------------------------------------------------------------
void DoubleSlider::onRangeChanged(int newIntMin, int newIntMax)
{
  Q_D(const DoubleSlider);
  if (d->SettingRange)
    {
    return;
    }
  double newMin = d->fromInt(newIntMin);
  double newMax = d->fromInt(newIntMax);
  if (d->Proxy)
    {
    newMin = d->Proxy.data()->valueFromProxyValue(newMin);
    newMax = d->Proxy.data()->valueFromProxyValue(newMax);
    }
  this->setRange(newMin, newMax);
}

// --------------------------------------------------------------------------
bool DoubleSlider::eventFilter(QObject* watched, QEvent* event)
{
  Q_D(DoubleSlider);
  if (watched == d->slider)
    {
    switch(event->type())
      {
      case QEvent::ToolTip:
        {
        QHelpEvent* helpEvent = static_cast<QHelpEvent*>(event);
        QStyleOptionSlider opt;
        d->slider->initStyleOption(&opt);
        QStyle::SubControl hoveredControl =
          d->slider->style()->hitTestComplexControl(
            QStyle::CC_Slider, &opt, helpEvent->pos(), this);
        if (!d->HandleToolTip.isEmpty() &&
            hoveredControl == QStyle::SC_SliderHandle)
          {
          QToolTip::showText(helpEvent->globalPos(), d->HandleToolTip.arg(this->value()));
          event->accept();
          return true;
          }
        }
      default:
        break;
      }
    }
  return this->Superclass::eventFilter(watched, event);
}

//----------------------------------------------------------------------------
void DoubleSlider::setValueProxy(ValueProxy* proxy)
{
  Q_D(DoubleSlider);
  if (proxy == d->Proxy.data())
    {
    return;
    }

  this->onValueProxyAboutToBeModified();

  if (d->Proxy.data())
    {
    disconnect(d->Proxy.data(), 0, this, 0);
    }

  d->Proxy = proxy;

  if (d->Proxy)
    {
    connect(d->Proxy.data(), SIGNAL(proxyAboutToBeModified()),
            this, SLOT(onValueProxyAboutToBeModified()));
    connect(d->Proxy.data(), SIGNAL(proxyModified()),
            this, SLOT(onValueProxyModified()));
    }

  this->onValueProxyModified();
}

//----------------------------------------------------------------------------
ValueProxy* DoubleSlider::valueProxy() const
{
  Q_D(const DoubleSlider);
  return d->Proxy.data();
}

// --------------------------------------------------------------------------
void DoubleSlider::onValueProxyAboutToBeModified()
{
  Q_D(DoubleSlider);
  d->slider->setProperty("inputValue", this->value());
  d->slider->setProperty("inputMinimum", this->minimum());
  d->slider->setProperty("inputMaximum", this->maximum());
}

// --------------------------------------------------------------------------
void DoubleSlider::onValueProxyModified()
{
  Q_D(DoubleSlider);
  bool wasBlockingSignals = this->blockSignals(true);
  bool wasSettingRange = d->SettingRange;
  d->SettingRange = true;
  this->setRange(d->slider->property("inputMinimum").toDouble(),
                 d->slider->property("inputMaximum").toDouble());
  d->SettingRange = wasSettingRange;
  this->setValue(d->slider->property("inputValue").toDouble());
  this->blockSignals(wasBlockingSignals);
}
