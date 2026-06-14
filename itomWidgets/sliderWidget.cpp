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

    This file is a port and modified version of the
    CTK Common Toolkit (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QDebug>
#include <QMouseEvent>
#include <QPointer>
#include <QDoubleSpinBox>

// CTK includes
#include "popupWidget.h"
#include "sliderWidget.h"
#include "valueProxy.h"
#include "ui_sliderWidget.h"

// STD includes
#include <cmath>

//-----------------------------------------------------------------------------
class SliderWidgetPrivate: public Ui_SliderWidget
{
  Q_DECLARE_PUBLIC(SliderWidget);
protected:
  SliderWidget* const q_ptr;

public:
  SliderWidgetPrivate(SliderWidget& object);
  virtual ~SliderWidgetPrivate();

  void updateSpinBoxWidth();
  void updateSpinBoxDecimals();

  int synchronizedSpinBoxWidth() const;

  void synchronizeSiblingWidth(int width);
  void synchronizeSiblingDecimals(int decimals);
  bool equal(double spinBoxValue, double sliderValue)const
  {
    if (this->Proxy)
      {
      spinBoxValue = this->Proxy.data()->proxyValueFromValue(spinBoxValue);
      sliderValue = this->Proxy.data()->proxyValueFromValue(sliderValue);
      }
    return qAbs(sliderValue - spinBoxValue) < std::pow(10., -this->SpinBox->decimals());
  }

  bool   Tracking;
  bool   Changing;
  double ValueBeforeChange;
  bool   BlockSetSliderValue;
  SliderWidget::SynchronizeSiblings SynchronizeMode;
  PopupWidget* SliderPopup;
  QPointer<ValueProxy> Proxy;
};

// --------------------------------------------------------------------------
SliderWidgetPrivate::SliderWidgetPrivate(SliderWidget& object)
  :q_ptr(&object)
{
  qRegisterMetaType<SliderWidget::SynchronizeSiblings>(
    "SliderWidget::SynchronizeSiblings");
  this->Tracking = true;
  this->Changing = false;
  this->ValueBeforeChange = 0.;
  this->BlockSetSliderValue = false;
  this->SynchronizeMode = SliderWidget::SynchronizeWidth;
  this->SliderPopup = 0;
}

// --------------------------------------------------------------------------
SliderWidgetPrivate::~SliderWidgetPrivate()
{
}

// --------------------------------------------------------------------------
void SliderWidgetPrivate::updateSpinBoxWidth()
{
  int spinBoxWidth = this->synchronizedSpinBoxWidth();
  if (this->SynchronizeMode.testFlag(SliderWidget::SynchronizeWidth))
    {
    this->SpinBox->setMinimumWidth(spinBoxWidth);
    }
  else
    {
    this->SpinBox->setMinimumWidth(0);
    }

  this->synchronizeSiblingWidth(spinBoxWidth);
}

// --------------------------------------------------------------------------
void SliderWidgetPrivate::updateSpinBoxDecimals()
{
  if (this->SynchronizeMode.testFlag(SliderWidget::SynchronizeDecimals))
    {
    this->synchronizeSiblingDecimals(this->SpinBox->decimals());
    }
}

// --------------------------------------------------------------------------
int SliderWidgetPrivate::synchronizedSpinBoxWidth()const
{
  Q_Q(const SliderWidget);
  int maxWidth = this->SpinBox->sizeHint().width();
  if (!q->parent())
    {
    return maxWidth;
    }
  QList<SliderWidget*> siblings =
    q->parent()->findChildren<SliderWidget*>();
  foreach(SliderWidget* sibling, siblings)
    {
    maxWidth = qMax(maxWidth, sibling->d_func()->SpinBox->sizeHint().width());
    }
  return maxWidth;
}

// --------------------------------------------------------------------------
void SliderWidgetPrivate::synchronizeSiblingWidth(int width)
{
  Q_UNUSED(width);
  Q_Q(const SliderWidget);
  QList<SliderWidget*> siblings =
    q->parent()->findChildren<SliderWidget*>();
  foreach(SliderWidget* sibling, siblings)
    {
    if (sibling != q
      && sibling->synchronizeSiblings().testFlag(SliderWidget::SynchronizeWidth))
      {
      sibling->d_func()->SpinBox->setMinimumWidth(
        this->SpinBox->minimumWidth());
      }
    }
}

// --------------------------------------------------------------------------
void SliderWidgetPrivate::synchronizeSiblingDecimals(int decimals)
{
  Q_UNUSED(decimals);
  Q_Q(const SliderWidget);
  QList<SliderWidget*> siblings =
    q->parent()->findChildren<SliderWidget*>();
  foreach(SliderWidget* sibling, siblings)
    {
    if (sibling != q
      && sibling->synchronizeSiblings().testFlag(SliderWidget::SynchronizeDecimals))
      {
      sibling->d_func()->SpinBox->setDecimals(this->SpinBox->decimals());
      }
    }
}

// --------------------------------------------------------------------------
SliderWidget::SliderWidget(QWidget* _parent) : Superclass(_parent)
  , d_ptr(new SliderWidgetPrivate(*this))
{
  Q_D(SliderWidget);

  d->setupUi(this);

  d->Slider->setMaximum(d->SpinBox->maximum());
  d->Slider->setMinimum(d->SpinBox->minimum());

  this->connect(d->SpinBox, SIGNAL(valueChanged(double)), this, SLOT(setSliderValue(double)));
  this->connect(d->SpinBox, SIGNAL(decimalsChanged(int)), this, SLOT(setDecimals(int)));

  this->connect(d->Slider, SIGNAL(sliderPressed()), this, SLOT(startChanging()));
  this->connect(d->Slider, SIGNAL(sliderReleased()), this, SLOT(stopChanging()));
  // setSpinBoxValue will fire the valueChanged signal.
  this->connect(d->Slider, SIGNAL(valueChanged(double)), this, SLOT(setSpinBoxValue(double)));
  d->SpinBox->installEventFilter(this);
}

// --------------------------------------------------------------------------
SliderWidget::~SliderWidget()
{
}

// --------------------------------------------------------------------------
double SliderWidget::minimum()const
{
  Q_D(const SliderWidget);
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  return d->Slider->minimum();
}

// --------------------------------------------------------------------------
double SliderWidget::maximum()const
{
  Q_D(const SliderWidget);
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  return d->Slider->maximum();
}

// --------------------------------------------------------------------------
void SliderWidget::setMinimum(double min)
{
  Q_D(SliderWidget);
  bool wasBlockSetSliderValue = d->BlockSetSliderValue;
  d->BlockSetSliderValue = true;
  d->SpinBox->setMinimum(min);
  d->BlockSetSliderValue = wasBlockSetSliderValue;

  // SpinBox can truncate min (depending on decimals).
  // use Spinbox's min to set Slider's min
  d->Slider->setMinimum(d->SpinBox->minimum());
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
void SliderWidget::setMaximum(double max)
{
  Q_D(SliderWidget);
  bool wasBlockSetSliderValue = d->BlockSetSliderValue;
  d->BlockSetSliderValue = true;
  d->SpinBox->setMaximum(max);
  d->BlockSetSliderValue = wasBlockSetSliderValue;

  // SpinBox can truncate max (depending on decimals).
  // use Spinbox's max to set Slider's max
  d->Slider->setMaximum(d->SpinBox->maximum());
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
void SliderWidget::setRange(double min, double max)
{
  Q_D(SliderWidget);

  bool wasBlockSetSliderValue = d->BlockSetSliderValue;
  d->BlockSetSliderValue = true;
  d->SpinBox->setRange(min, max);
  d->BlockSetSliderValue = wasBlockSetSliderValue;

  // SpinBox can truncate the range (depending on decimals).
  // use Spinbox's range to set Slider's range
  d->Slider->setRange(d->SpinBox->minimum(), d->SpinBox->maximum());
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  d->updateSpinBoxWidth();
}
/*
// --------------------------------------------------------------------------
double SliderWidget::sliderPosition()const
{
  return d->Slider->sliderPosition();
}

// --------------------------------------------------------------------------
void SliderWidget::setSliderPosition(double position)
{
  d->Slider->setSliderPosition(position);
}
*/
/*
// --------------------------------------------------------------------------
double SliderWidget::previousSliderPosition()
{
  return d->Slider->previousSliderPosition();
}
*/

// --------------------------------------------------------------------------
double SliderWidget::value()const
{
  Q_D(const SliderWidget);
  Q_ASSERT(d->equal(d->SpinBox->value(), d->Slider->value()));
  // The slider is the most precise as it does not round the value with the
  // decimals number.
  return d->Changing ? d->ValueBeforeChange : d->Slider->value();
}

// --------------------------------------------------------------------------
void SliderWidget::setValue(double _value)
{
  Q_D(SliderWidget);
  // disable the tracking temporally to emit the
  // signal valueChanged if setSpinBoxValue() is called
  bool isChanging = d->Changing;
  d->Changing = false;
  d->SpinBox->setValue(_value);
  // Why do we need to set the value to the slider ?
  //d->Slider->setValue(d->SpinBox->value());
  //double spinBoxValue = d->SpinBox->value();
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  // restore the prop
  d->Changing = isChanging;
}

// --------------------------------------------------------------------------
void SliderWidget::startChanging()
{
  Q_D(SliderWidget);
  if (d->Tracking)
    {
    return;
    }
  d->ValueBeforeChange = this->value();
  d->Changing = true;
}

// --------------------------------------------------------------------------
void SliderWidget::stopChanging()
{
  Q_D(SliderWidget);
  if (d->Tracking)
    {
    return;
    }
  d->Changing = false;
  if (qAbs(this->value() - d->ValueBeforeChange) > (this->singleStep() * 0.000000001))
    {
    emit this->valueChanged(this->value());
    }
}

// --------------------------------------------------------------------------
void SliderWidget::setSliderValue(double spinBoxValue)
{
  Q_D(SliderWidget);
  if (d->BlockSetSliderValue)
    {
    return;
    }
  d->Slider->setValue(spinBoxValue);
}

// --------------------------------------------------------------------------
void SliderWidget::setSpinBoxValue(double sliderValue)
{
  Q_D(SliderWidget);

  bool wasBlockSetSliderValue = d->BlockSetSliderValue;
  d->BlockSetSliderValue = true;
  d->SpinBox->setValue(sliderValue);
  d->BlockSetSliderValue = wasBlockSetSliderValue;
  Q_ASSERT(d->equal(d->SpinBox->value(), d->Slider->value()));

  if (!d->Tracking)
    {
    emit this->valueIsChanging(sliderValue);
    }
  if (!d->Changing)
    {
    emit this->valueChanged(sliderValue);
    }
}

// --------------------------------------------------------------------------
bool SliderWidget::eventFilter(QObject *obj, QEvent *event)
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
       // here we might prevent SliderWidget::stopChanging
       // from sending a valueChanged() event as the spinbox might
       // send a valueChanged() after eventFilter() is done.
       this->stopChanging();
       }
     }
   // standard event processing
   return this->Superclass::eventFilter(obj, event);
 }

// --------------------------------------------------------------------------
double SliderWidget::singleStep()const
{
  Q_D(const SliderWidget);
  Q_ASSERT(d->equal(d->SpinBox->singleStep(), d->Slider->singleStep()));
  return d->Slider->singleStep();
}

// --------------------------------------------------------------------------
void SliderWidget::setSingleStep(double newStep)
{
  Q_D(SliderWidget);
  if (!d->Slider->isValidStep(newStep))
    {
    qWarning() << "SliderWidget::setSingleStep() " << newStep << "is out of bounds." <<
      this->minimum() << this->maximum() <<this->value();
    return;
    }
  d->SpinBox->setSingleStep(newStep);
  d->Slider->setSingleStep(d->SpinBox->singleStep());
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
}

// --------------------------------------------------------------------------
double SliderWidget::pageStep()const
{
  Q_D(const SliderWidget);
  return d->Slider->pageStep();
}

// --------------------------------------------------------------------------
void SliderWidget::setPageStep(double step)
{
  Q_D(SliderWidget);
  d->Slider->setPageStep(step);
}

// --------------------------------------------------------------------------
int SliderWidget::decimals()const
{
  Q_D(const SliderWidget);
  return d->SpinBox->decimals();
}

// --------------------------------------------------------------------------
void SliderWidget::setDecimals(int newDecimals)
{
  Q_D(SliderWidget);
  d->SpinBox->setDecimals(newDecimals);
  // The number of decimals can change the range values
  // i.e. 50.55 with 2 decimals -> 51 with 0 decimals
  // As the SpinBox range change doesn't fire signals,
  // we have to do the synchronization manually here
  d->Slider->setRange(d->SpinBox->minimum(), d->SpinBox->maximum());
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  // Last time the value was set on the spinbox, the value might have been
  // rounded by the previous number of decimals. The slider however never rounds
  // the value. Now, if the number of decimals is higher, such rounding is lost
  // precision. The "true" value must be set again to the spinbox to "recover"
  // the precision.
  this->setSpinBoxValue(d->Slider->value());
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
  d->updateSpinBoxDecimals();
  emit decimalsChanged(d->SpinBox->decimals());
}

// --------------------------------------------------------------------------
QString SliderWidget::prefix()const
{
  Q_D(const SliderWidget);
  return d->SpinBox->prefix();
}

// --------------------------------------------------------------------------
void SliderWidget::setPrefix(const QString& newPrefix)
{
  Q_D(SliderWidget);
  d->SpinBox->setPrefix(newPrefix);
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
QString SliderWidget::suffix()const
{
  Q_D(const SliderWidget);
  return d->SpinBox->suffix();
}

// --------------------------------------------------------------------------
void SliderWidget::setSuffix(const QString& newSuffix)
{
  Q_D(SliderWidget);
  d->SpinBox->setSuffix(newSuffix);
  d->updateSpinBoxWidth();
}

// --------------------------------------------------------------------------
double SliderWidget::tickInterval()const
{
  Q_D(const SliderWidget);
  return d->Slider->tickInterval();
}

// --------------------------------------------------------------------------
void SliderWidget::setTickInterval(double ti)
{
  Q_D(SliderWidget);
  d->Slider->setTickInterval(ti);
}

// --------------------------------------------------------------------------
QSlider::TickPosition SliderWidget::tickPosition()const
{
  Q_D(const SliderWidget);
  return d->Slider->tickPosition();
}

// --------------------------------------------------------------------------
void SliderWidget::setTickPosition(QSlider::TickPosition newTickPosition)
{
  Q_D(SliderWidget);
  d->Slider->setTickPosition(newTickPosition);
}

// -------------------------------------------------------------------------
void SliderWidget::reset()
{
  this->setValue(0.);
}

// -------------------------------------------------------------------------
void SliderWidget::setSpinBoxAlignment(Qt::Alignment alignment)
{
  Q_D(SliderWidget);
  return d->SpinBox->setAlignment(alignment);
}

// -------------------------------------------------------------------------
Qt::Alignment SliderWidget::spinBoxAlignment()const
{
  Q_D(const SliderWidget);
  return d->SpinBox->alignment();
}

// -------------------------------------------------------------------------
void SliderWidget::setTracking(bool enable)
{
  Q_D(SliderWidget);
  d->SpinBox->spinBox()->setKeyboardTracking(enable);
  d->Tracking = enable;
}

// -------------------------------------------------------------------------
bool SliderWidget::hasTracking()const
{
  Q_D(const SliderWidget);
  return d->Tracking;
}

// --------------------------------------------------------------------------
bool SliderWidget::invertedAppearance()const
{
  Q_D(const SliderWidget);
  return d->Slider->invertedAppearance();
}

// --------------------------------------------------------------------------
void SliderWidget::setInvertedAppearance(bool invertedAppearance)
{
  Q_D(SliderWidget);
  d->Slider->setInvertedAppearance(invertedAppearance);
}

// --------------------------------------------------------------------------
bool SliderWidget::invertedControls()const
{
  Q_D(const SliderWidget);
  return d->Slider->invertedControls() && d->SpinBox->invertedControls();
}

// --------------------------------------------------------------------------
void SliderWidget::setInvertedControls(bool invertedControls)
{
  Q_D(SliderWidget);
  d->Slider->setInvertedControls(invertedControls);
  d->SpinBox->setInvertedControls(invertedControls);
}

// -------------------------------------------------------------------------
SliderWidget::SynchronizeSiblings
SliderWidget::synchronizeSiblings() const
{
  Q_D(const SliderWidget);
  return d->SynchronizeMode;
}

// -------------------------------------------------------------------------
void SliderWidget
::setSynchronizeSiblings(SliderWidget::SynchronizeSiblings flag)
{
  Q_D(SliderWidget);
  d->SynchronizeMode = flag;
  d->updateSpinBoxWidth();
  d->updateSpinBoxDecimals();
}

// -------------------------------------------------------------------------
bool SliderWidget::isSpinBoxVisible()const
{
  Q_D(const SliderWidget);
  return d->SpinBox->isVisibleTo(const_cast<SliderWidget*>(this));
}

// -------------------------------------------------------------------------
void SliderWidget::setSpinBoxVisible(bool visible)
{
  Q_D(SliderWidget);
  d->SpinBox->setVisible(visible);
}

// --------------------------------------------------------------------------
bool SliderWidget::hasPopupSlider()const
{
  Q_D(const SliderWidget);
  return d->SliderPopup != 0;
}

// --------------------------------------------------------------------------
void SliderWidget::setPopupSlider(bool popup)
{
  Q_D(SliderWidget);
  if (this->hasPopupSlider() == popup)
    {
    return;
    }
  if (popup)
    {
    d->SliderPopup = new PopupWidget(this);
    d->SliderPopup->setObjectName("DoubleSliderPopup");

    QHBoxLayout* layout = new QHBoxLayout(d->SliderPopup);
    layout->setContentsMargins(3, 1, 3, 1);
    /// If the Slider has already been created, it will try to keep its
    /// size.
    layout->addWidget(d->Slider);

    d->SliderPopup->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    d->SliderPopup->setOrientation(Qt::Horizontal);
    d->SliderPopup->setHorizontalDirection(Qt::RightToLeft);
    }
  else
    {
    qobject_cast<QHBoxLayout*>(this->layout())->insertWidget(0,d->Slider);
    d->SliderPopup->deleteLater();
    d->SliderPopup = 0;
    }
}

// --------------------------------------------------------------------------
PopupWidget* SliderWidget::popup()const
{
  Q_D(const SliderWidget);
  return d->SliderPopup;
}

// --------------------------------------------------------------------------
DoubleSpinBox* SliderWidget::spinBox()
{
  Q_D(SliderWidget);
  return d->SpinBox;
}

// --------------------------------------------------------------------------
DoubleSlider* SliderWidget::slider()
{
  Q_D(SliderWidget);
  return d->Slider;
}

// --------------------------------------------------------------------------
void SliderWidget::setValueProxy(ValueProxy* proxy)
{
  Q_D(SliderWidget);
  if (d->Proxy.data() == proxy)
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
  this->spinBox()->setValueProxy(proxy);

  if (d->Proxy)
    {
    connect(d->Proxy.data(), SIGNAL(proxyModified()),
            this, SLOT(onValueProxyModified()));
    }
  this->onValueProxyModified();
}

// --------------------------------------------------------------------------
ValueProxy* SliderWidget::valueProxy() const
{
  Q_D(const SliderWidget);
  return d->Proxy.data();
}

// --------------------------------------------------------------------------
void SliderWidget::onValueProxyAboutToBeModified()
{
}

// --------------------------------------------------------------------------
void SliderWidget::onValueProxyModified()
{
  Q_D(SliderWidget);
  Q_ASSERT(d->equal(d->SpinBox->minimum(),d->Slider->minimum()));
  Q_ASSERT(d->equal(d->SpinBox->maximum(),d->Slider->maximum()));
  // resync as the modification of proxy could have discarded decimals
  // in the process. The slider always keeps the exact value (no rounding).
  d->SpinBox->setValue(d->Slider->value());
  Q_ASSERT(d->equal(d->SpinBox->value(),d->Slider->value()));
}
