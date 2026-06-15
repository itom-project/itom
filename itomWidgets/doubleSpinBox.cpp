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

//  includes
#include "doubleSpinBox_p.h"
#include "utils.h"
#include "valueProxy.h"

// Qt includes
#include <QApplication>
#include <QDebug>
#include <QEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLineEdit>
#include <QShortcut>
#include <QSizePolicy>
#include <QStyle>
#include <QStyleOptionSpinBox>
#include <QVariant>

//-----------------------------------------------------------------------------
// QDoubleSpinBox
//----------------------------------------------------------------------------
itomQDoubleSpinBox::itomQDoubleSpinBox(DoubleSpinBoxPrivate* pimpl,
                                     QWidget* spinBoxParent)
  : QDoubleSpinBox(spinBoxParent)
  , d_ptr(pimpl)
{
  this->InvertedControls = false;
}

//----------------------------------------------------------------------------
QLineEdit* itomQDoubleSpinBox::lineEdit()const
{
  return this->QDoubleSpinBox::lineEdit();
}
//----------------------------------------------------------------------------
void itomQDoubleSpinBox::initStyleOptionSpinBox(QStyleOptionSpinBox* option)
{
  this->initStyleOption(option);
}

//----------------------------------------------------------------------------
void itomQDoubleSpinBox::setInvertedControls(bool invertedControls)
{
  this->InvertedControls = invertedControls;
}

//----------------------------------------------------------------------------
bool itomQDoubleSpinBox::invertedControls() const
{
  return this->InvertedControls;
}

//----------------------------------------------------------------------------
void itomQDoubleSpinBox::stepBy(int steps)
{
  if (this->InvertedControls)
    {
    steps = -steps;
    }
  this->Superclass::stepBy(steps);
}

//----------------------------------------------------------------------------
void itomQDoubleSpinBox::focusOutEvent(QFocusEvent * event)
{
    QDoubleSpinBox::focusOutEvent(event);
    QObject *p = parent();
    QCoreApplication::sendEvent(p, event);
}

//----------------------------------------------------------------------------
QAbstractSpinBox::StepEnabled itomQDoubleSpinBox::stepEnabled() const
{
  if (!this->InvertedControls)
    {
    return this->Superclass::stepEnabled();
    }

  if (this->isReadOnly())
    {
    return StepNone;
    }

  if (this->wrapping())
    {
    return StepEnabled(StepUpEnabled | StepDownEnabled);
    }

  StepEnabled ret = StepNone;
  double value = this->value();
  if (value < this->maximum())
    {
    ret |= StepDownEnabled;
    }
  if (value > this->minimum())
    {
    ret |= StepUpEnabled;
    }
  return ret;
}

//-----------------------------------------------------------------------------
double itomQDoubleSpinBox::valueFromText(const QString &text) const
{
  Q_D(const DoubleSpinBox);

  QString copy = text;
  int pos = this->lineEdit()->cursorPosition();
  QValidator::State state = QValidator::Acceptable;
  int decimals = 0;
  double value = d->validateAndInterpret(copy, pos, state, decimals);
  return value;
}

//-----------------------------------------------------------------------------
QString itomQDoubleSpinBox::textFromValue(double value) const
{
  Q_D(const DoubleSpinBox);
  QString text = this->QDoubleSpinBox::textFromValue(value);
  if (text.isEmpty())
    {
    text = "0";
    }
  // If there is no decimal, it does not mean there won't be any.
  if (d->DOption & DoubleSpinBox::DecimalPointAlwaysVisible &&
      text.indexOf(this->locale().decimalPoint()) == -1)
    {
    text += this->locale().decimalPoint();
    }
  return text;
}

//-----------------------------------------------------------------------------
int itomQDoubleSpinBox::decimalsFromText(const QString &text) const
{
  Q_D(const DoubleSpinBox);

  QString copy = text;
  int pos = this->lineEdit()->cursorPosition();
  int decimals = 0;
  QValidator::State state = QValidator::Acceptable;
  d->validateAndInterpret(copy, pos, state, decimals);
  return decimals;
}

//-----------------------------------------------------------------------------
QValidator::State itomQDoubleSpinBox::validate(QString &text, int &pos) const
{
  Q_D(const DoubleSpinBox);

  QValidator::State state = QValidator::Acceptable;
  int decimals = 0;
  d->validateAndInterpret(text, pos, state, decimals);
  return state;
}

//-----------------------------------------------------------------------------
// DoubleSpinBoxPrivate
//-----------------------------------------------------------------------------
DoubleSpinBoxPrivate::DoubleSpinBoxPrivate(DoubleSpinBox& object)
  : q_ptr(&object)
{
  qRegisterMetaType<DoubleSpinBox::SetMode>("DoubleSpinBox::SetMode");
  qRegisterMetaType<DoubleSpinBox::DecimalsOptions>("DoubleSpinBox::DecimalsOption");
  this->SpinBox = 0;
  this->Mode = DoubleSpinBox::SetIfDifferent;
  this->DefaultDecimals = 2;
  // InsertDecimals is not a great default, but it is QDoubleSpinBox's default.
  this->DOption = DoubleSpinBox::DecimalsByShortcuts
    | DoubleSpinBox::InsertDecimals;
  this->InvertedControls = false;
  this->SizeHintPolicy = DoubleSpinBox::SizeHintByMinMax;
  this->InputValue = 0.;
  this->InputRange[0] = 0.;
  this->InputRange[1] = 99.99;
  this->ForceInputValueUpdate = false;
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::init()
{
  Q_Q(DoubleSpinBox);
  this->SpinBox = new itomQDoubleSpinBox(this, q);
  this->SpinBox->setInvertedControls(this->InvertedControls);
  // QDoubleSpinBox needs to be first to receive textChanged() signals.
  QLineEdit* lineEdit = new QLineEdit(q);
  QObject::connect(lineEdit, SIGNAL(textChanged(QString)),
                   this, SLOT(editorTextChanged(QString)));
  this->SpinBox->setLineEdit(lineEdit);
  lineEdit->setObjectName(QLatin1String("qt_spinbox_lineedit"));
  this->InputValue = this->SpinBox->value();
  this->InputRange[0] = this->SpinBox->minimum();
  this->InputRange[1] = this->SpinBox->maximum();

  QObject::connect(this->SpinBox, SIGNAL(valueChanged(double)),
    this, SLOT(onValueChanged()));
  QObject::connect(this->SpinBox, SIGNAL(editingFinished()),
    q, SIGNAL(editingFinished()));

  QHBoxLayout* l = new QHBoxLayout(q);
  l->addWidget(this->SpinBox);
  l->setContentsMargins(0,0,0,0);
  q->setLayout(l);
  q->setSizePolicy(QSizePolicy(QSizePolicy::Minimum,
    QSizePolicy::Fixed, QSizePolicy::ButtonBox));

  this->SpinBox->installEventFilter(q);
}

//-----------------------------------------------------------------------------
bool DoubleSpinBoxPrivate::compare(double x1, double x2) const
{
  Q_Q(const DoubleSpinBox);
  return q->round(x1) == q->round(x2);
}

//-----------------------------------------------------------------------------
double DoubleSpinBoxPrivate::round(double value, int decimals) const
{
  return QString::number(value, 'f', decimals).toDouble();
}

//-----------------------------------------------------------------------------
QString DoubleSpinBoxPrivate::stripped(const QString& text, int* pos) const
{
  Q_Q(const DoubleSpinBox);
  QString strip(text);
  if (strip.startsWith(q->prefix()))
    {
    strip.remove(0, q->prefix().size());
    }
  if (strip.endsWith(q->suffix()))
    {
    strip.chop(q->suffix().size());
    }
  strip = strip.trimmed();
  if (pos)
    {
    int stripInText = text.indexOf(strip);
    *pos = qBound(0, *pos - stripInText, strip.size());
    }
  return strip;
}

//-----------------------------------------------------------------------------
int DoubleSpinBoxPrivate::boundDecimals(int dec)const
{
  Q_Q(const DoubleSpinBox);
  if (dec == -1)
    {
    return q->decimals();
    }
  int min = (this->DOption & DoubleSpinBox::DecimalsAsMin) ?
    this->DefaultDecimals : 0;
  int max = (this->DOption & DoubleSpinBox::DecimalsAsMax) ?
    this->DefaultDecimals : 323; // see QDoubleSpinBox::decimals doc
  return qBound(min, dec, max);
}

//-----------------------------------------------------------------------------
int DoubleSpinBoxPrivate::decimalsForValue(double value) const
{
  int decimals = this->DefaultDecimals;
  if (this->DOption & DoubleSpinBox::DecimalsByValue)
    {
    decimals = ctk::significantDecimals(value, decimals);
    }
  return this->boundDecimals(decimals);
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::setValue(double value, int dec)
{
  Q_Q(DoubleSpinBox);
  dec = this->boundDecimals(dec);
  const bool changeDecimals = dec != q->decimals();
  if (changeDecimals)
  {
     // don't fire valueChanged signal because we will change the value
     // right after anyway.
     const bool blockValueChangedSignal = (this->round(this->SpinBox->value(), dec) != value);
     bool wasBlocked = false;
     if (blockValueChangedSignal)
     {
       wasBlocked = this->SpinBox->blockSignals(true);
     }
     // don't fire decimalsChanged signal yet, wait for the value to be
     // up-to-date.
     this->SpinBox->setDecimals(dec);
     if (blockValueChangedSignal)
     {
       this->SpinBox->blockSignals(wasBlocked);
     }
  }
  this->SpinBox->setValue(value); // re-do the text (calls textFromValue())
  if (changeDecimals)
  {
    emit q->decimalsChanged(dec);
  }
  if (this->SizeHintPolicy == DoubleSpinBox::SizeHintByValue)
  {
    this->CachedSizeHint = QSize();
    this->CachedMinimumSizeHint = QSize();
    q->updateGeometry();
  }
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::setDecimals(int dec)
{
  Q_Q(DoubleSpinBox);
  dec = this->boundDecimals(dec);
  this->SpinBox->setDecimals(dec);
  emit q->decimalsChanged(dec);
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::editorTextChanged(const QString& text)
{
  if (this->SpinBox->keyboardTracking())
    {
    QString tmp = text;
    int pos = this->SpinBox->lineEdit()->cursorPosition();
    QValidator::State state = QValidator::Invalid;
    int decimals = 0;
    this->validateAndInterpret(tmp, pos, state, decimals);
    if (state == QValidator::Acceptable)
      {
      double newValue = this->SpinBox->valueFromText(tmp);
      int decimals = this->boundDecimals(this->SpinBox->decimalsFromText(tmp));
      bool changeDecimals = this->DOption & DoubleSpinBox::DecimalsByKey &&
        decimals != this->SpinBox->decimals();
      if (changeDecimals)
        {
        this->ForceInputValueUpdate = true;
        this->setValue(newValue, decimals);
        this->ForceInputValueUpdate = false;
        }
      // else, let QDoubleSpinBox process the validation.
      }
    }
}

//-----------------------------------------------------------------------------
double DoubleSpinBoxPrivate
::validateAndInterpret(QString &input, int &pos,
                       QValidator::State &state, int &decimals) const
{
  Q_Q(const DoubleSpinBox);
  if (this->CachedText == input)
    {
    state = this->CachedState;
    decimals = this->CachedDecimals;
    return this->CachedValue;
    }
  const double max = this->SpinBox->maximum();
  const double min = this->SpinBox->minimum();

  int posInValue = pos;
  QString text = this->stripped(input, &posInValue);
  // posInValue can change, track the offset.
  const int oldPosInValue = posInValue;
  state = QValidator::Acceptable;
  decimals = 0;

  double value = min;
  const int dec = text.indexOf(q->locale().decimalPoint());

  bool ok = false;
  value = q->locale().toDouble(text, &ok);

  // could be in an intermediate state
  if (!ok  && state == QValidator::Acceptable)
    {
    if (text.isEmpty() ||
        text == "." ||
        text == "-" ||
        text == "+" ||
        text == "-." ||
        text == "+.")
      {
      state = QValidator::Intermediate;
      }
    }
  // could be because of group separators:
  if (!ok && state == QValidator::Acceptable)
    {
#if (QT_VERSION >= QT_VERSION_CHECK(6, 0, 0))
      QChar groupSeparator;

      if (q->locale().groupSeparator().size() == 1)
      {
          groupSeparator = q->locale().groupSeparator()[0];
      }
      // else: group separator does not necessarily fit into a QChar (https://bugreports.qt.io/browse/QTBUG-69324)
      // but CTK only support group separators if they fit into a QChar
#else
      QChar groupSeparator = q->locale().groupSeparator();
#endif
      if (groupSeparator.isPrint())
      {
      int start = (dec == -1 ? text.size() : dec)- 1;
      int lastGroupSeparator = start;
      for (int digit = start; digit >= 0; --digit)
        {
        if (text.at(digit) == q->locale().groupSeparator())
          {
          if (digit != lastGroupSeparator - 3)
            {
            state = QValidator::Invalid;
            break;
            }
          text.remove(digit, 1);
          lastGroupSeparator = digit;
          }
        }
      }
    // try again without the group separators
    value = q->locale().toDouble(text, &ok);
    }
  // test the decimalPoint
  if (!ok && state == QValidator::Acceptable)
    {
    // duplicate decimal points probably means the user typed another decimal points,
    // move the cursor pos to the right then
    if (dec + 1 < text.size() &&
        text.at(dec + 1) == q->locale().decimalPoint() &&
        posInValue == dec + 1)
      {
      text.remove(dec + 1, 1);
      value = q->locale().toDouble(text, &ok);
      }
    }
  if (ok && state != QValidator::Invalid)
    {
    if (dec != -1)
      {
      decimals = text.size() - (dec + 1);
      if (decimals > q->decimals())
        {
        // With ReplaceDecimals on, key strokes replace decimal digits
        if (posInValue > dec && posInValue < text.size())
          {
          const int extraDecimals = decimals - q->decimals();
          if (this->DOption & DoubleSpinBox::ReplaceDecimals)
            {
            text.remove(posInValue, extraDecimals);
            decimals = q->decimals();
            value = q->locale().toDouble(text, &ok);
            }
          else if (!(this->DOption & DoubleSpinBox::InsertDecimals))
            {
            text.remove(text.size() - extraDecimals, extraDecimals);
            decimals = q->decimals();
            value = q->locale().toDouble(text, &ok);
            }
          }
        }
      // When DecimalsByKey is set, it is possible to extend the number of decimals
      if (decimals > q->decimals() &&
          !(this->DOption & DoubleSpinBox::DecimalsByKey) )
        {
        state = QValidator::Invalid;
        }
      }
    }
  if (state == QValidator::Acceptable)
    {
    if (!ok)
      {
      state = QValidator::Invalid;
      }
    else if (value >= min && value <= max)
      {
      state = QValidator::Acceptable;
      }
    else if (max == min)
      { // when max and min is the same the only non-Invalid input is max (or min)
      state = QValidator::Invalid;
      }
    else if ((value >= 0 && value > max) || (value < 0 && value < min))
      {
      state = QValidator::Invalid;
      }
    else
      {
      state = QValidator::Intermediate;
      }
    }

  if (state != QValidator::Acceptable)
    {
    value = max > 0 ? min : max;
    }

  pos += posInValue - oldPosInValue;
  input = q->prefix() + text + q->suffix();
  this->CachedText = input;
  this->CachedState = state;
  this->CachedValue = value;
  this->CachedDecimals = decimals;
  return value;
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::onValueChanged()
{
  Q_Q(DoubleSpinBox);
  double newValue = this->SpinBox->value();
  double oldValue = q->value();
  if (this->Proxy)
    {
    oldValue = this->Proxy.data()->proxyValueFromValue(oldValue);
    }
  // Don't trigger value changed signal if the difference only happened on the
  // precision.
  if (this->compare(oldValue, newValue) && !this->ForceInputValueUpdate)
    {
    return;
    }
  // Force it only once (when the user typed a new number that could have change
  // the number of decimals which could have make the compare test always pass.
  this->ForceInputValueUpdate = false;

  double minimum = q->minimum();
  double maximum = q->maximum();
  if (this->Proxy)
    {
    minimum = this->Proxy.data()->proxyValueFromValue(minimum);
    maximum = this->Proxy.data()->proxyValueFromValue(maximum);
    }
  // Special case to return max precision
  if (this->compare(minimum, newValue))
    {
    newValue = q->minimum();
    }
  else if (this->compare(maximum, newValue))
    {
    newValue = q->maximum();
    }
  else if (this->Proxy)
    {
    newValue = this->Proxy.data()->valueFromProxyValue(newValue);
    }
  this->InputValue = newValue;
  emit q->valueChanged(newValue);
  // \tbd The string might not make much sense when using proxies.
  emit q->valueChanged(
    QString::number(newValue, 'f', this->SpinBox->decimals()));
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::onValueProxyAboutToBeModified()
{
}

//-----------------------------------------------------------------------------
void DoubleSpinBoxPrivate::onValueProxyModified()
{
  Q_Q(DoubleSpinBox);
  int oldDecimals = q->decimals();
  double oldValue = this->InputValue;
  DoubleSpinBox::SetMode oldSetMode = this->Mode;

  // Only the display is changed, not the programatic value, no need to trigger
  // signals
  bool wasBlocking = q->blockSignals(true);
  // Enforce a refresh. Signals are blocked so it should not trigger unwanted
  // signals
  this->Mode = DoubleSpinBox::SetAlways;
  q->setRange(this->InputRange[0], this->InputRange[1]);
  q->setValue(oldValue);
  this->Mode = oldSetMode;
  q->blockSignals(wasBlocking);
  // Decimals might change when value proxy is modified.
  if (oldDecimals != q->decimals())
    {
    emit q->decimalsChanged(q->decimals());
    }
}

//-----------------------------------------------------------------------------
// DoubleSpinBox
//-----------------------------------------------------------------------------
DoubleSpinBox::DoubleSpinBox(QWidget* newParent)
  : QWidget(newParent)
  , d_ptr(new DoubleSpinBoxPrivate(*this))
{
  Q_D(DoubleSpinBox);
  d->init();
  //setFocusProxy(d->SpinBox);
}

//-----------------------------------------------------------------------------
DoubleSpinBox::DoubleSpinBox(DoubleSpinBox::SetMode mode, QWidget* newParent)
  : QWidget(newParent)
  , d_ptr(new DoubleSpinBoxPrivate(*this))
{
  Q_D(DoubleSpinBox);
  d->init();
  this->setSetMode(mode);
}

//-----------------------------------------------------------------------------
DoubleSpinBox::~DoubleSpinBox()
{
}

//-----------------------------------------------------------------------------
double DoubleSpinBox::value() const
{
  Q_D(const DoubleSpinBox);
  return d->InputValue;
}

//-----------------------------------------------------------------------------
double DoubleSpinBox::displayedValue() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->value();
}

//----------------------------------------------------------------------------
void DoubleSpinBox::setDisplayedValue(double value)
{
  Q_D(DoubleSpinBox);
  d->SpinBox->setValue(value);
}

//-----------------------------------------------------------------------------
QString DoubleSpinBox::text() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->text();
}

//-----------------------------------------------------------------------------
QString DoubleSpinBox::cleanText() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->cleanText();
}

//-----------------------------------------------------------------------------
Qt::Alignment DoubleSpinBox::alignment() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->alignment();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setAlignment(Qt::Alignment flag)
{
  Q_D(const DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent && flag == d->SpinBox->alignment())
    {
    return;
    }

  d->SpinBox->setAlignment(flag);
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setFrame(bool frame)
{
  Q_D(const DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent && frame == d->SpinBox->hasFrame())
    {
    return;
    }

  d->SpinBox->setFrame(frame);
}

//-----------------------------------------------------------------------------
bool DoubleSpinBox::hasFrame() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->hasFrame();
}

//-----------------------------------------------------------------------------
QString DoubleSpinBox::prefix() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->prefix();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setPrefix(const QString &prefix)
{
  Q_D(const DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent && prefix == d->SpinBox->prefix())
    {
    return;
    }

#if QT_VERSION < 0x040800
  /// Setting the prefix doesn't recompute the sizehint, do it manually here:
  /// See: http://bugreports.qt.nokia.com/browse/QTBUG-9530
  d->SpinBox->setRange(d->SpinBox->minimum(), d->SpinBox->maximum());
#endif

  d->SpinBox->setPrefix(prefix);
}

//-----------------------------------------------------------------------------
QString DoubleSpinBox::suffix() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->suffix();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setSuffix(const QString &suffix)
{
  Q_D(const DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent && suffix == d->SpinBox->suffix())
    {
    return;
    }

#if QT_VERSION < 0x040800
  /// Setting the suffix doesn't recompute the sizehint, do it manually here:
  /// See: http://bugreports.qt.nokia.com/browse/QTBUG-9530
  d->SpinBox->setRange(d->SpinBox->minimum(), d->SpinBox->maximum());
#endif

  d->SpinBox->setSuffix(suffix);
}

//-----------------------------------------------------------------------------
double DoubleSpinBox::singleStep() const
{
  Q_D(const DoubleSpinBox);
  double step = d->SpinBox->singleStep();
  return step;
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setSingleStep(double newStep)
{
  Q_D(DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent
    && d->compare(newStep, this->singleStep()))
    {
    return;
    }

  d->SpinBox->setSingleStep(newStep);
}

//-----------------------------------------------------------------------------
double DoubleSpinBox::minimum() const
{
  Q_D(const DoubleSpinBox);
  return d->InputRange[0];
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setMinimum(double newMin)
{
  this->setRange(newMin, qMax(newMin, this->maximum()));
}

//-----------------------------------------------------------------------------
double DoubleSpinBox::maximum() const
{
  Q_D(const DoubleSpinBox);
  return d->InputRange[1];
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setMaximum(double newMax)
{
  this->setRange(qMin(newMax, this->minimum()), newMax);
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setRange(double newMin, double newMax)
{
  Q_D(DoubleSpinBox);
  if (newMin > newMax)
    {
    qSwap(newMin, newMax);
    }
  if (d->Mode == DoubleSpinBox::SetIfDifferent
      && newMin == d->InputRange[0]
      && newMax == d->InputRange[1])
    {
    return;
    }
  d->InputRange[0] = newMin;
  d->InputRange[1] = newMax;
  if (d->Proxy)
    {
    newMin = d->Proxy.data()->proxyValueFromValue(newMin);
    newMax = d->Proxy.data()->proxyValueFromValue(newMax);
    if (newMin > newMax)
      {
      qSwap(newMin, newMax);
      }
    }

  d->SpinBox->setRange(newMin, newMax);
}

//-----------------------------------------------------------------------------
int DoubleSpinBox::decimals() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->decimals();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setDecimals(int dec)
{
  Q_D(DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent
      && dec == this->decimals()
      && dec == d->DefaultDecimals)
    {
    return;
    }

  d->DefaultDecimals = dec;
  // The number of decimals may or may not depend on the value. Recompute the
  // new number of decimals.
  double currentValue = this->value();
  if (d->Proxy)
    {
    currentValue = d->Proxy.data()->proxyValueFromValue(currentValue);
    }
  int newDecimals = d->decimalsForValue(currentValue);
  d->setValue(currentValue, newDecimals);
}

//-----------------------------------------------------------------------------
double DoubleSpinBox::round(double value) const
{
  Q_D(const DoubleSpinBox);
  return QString::number(value, 'f', d->SpinBox->decimals()).toDouble();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setKeyboardTracking(bool kt)
{
    Q_D(const DoubleSpinBox);
    d->SpinBox->setKeyboardTracking(kt);
}

//-----------------------------------------------------------------------------
bool DoubleSpinBox::keyboardTracking() const
{
    Q_D(const DoubleSpinBox);
    return d->SpinBox->keyboardTracking();
}

//-----------------------------------------------------------------------------
QDoubleSpinBox* DoubleSpinBox::spinBox() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox;
}

//-----------------------------------------------------------------------------
QLineEdit* DoubleSpinBox::lineEdit() const
{
  Q_D(const DoubleSpinBox);
  return d->SpinBox->lineEdit();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setValue(double value)
{
  Q_D(DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent)
    {
    this->setValueIfDifferent(value);
    }
  else
    {
    this->setValueAlways(value);
    }
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setValueIfDifferent(double newValue)
{
  Q_D(DoubleSpinBox);
  if (newValue == d->InputValue)
    {
    return;
    }
  this->setValueAlways(newValue);
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setValueAlways(double newValue)
{
  Q_D(DoubleSpinBox);
  newValue = qBound(d->InputRange[0], newValue, d->InputRange[1]);
  const bool valueModified = d->InputValue != newValue;
  d->InputValue = newValue;
  double newValueToDisplay = newValue;
  if (d->Proxy)
    {
    newValueToDisplay = d->Proxy.data()->proxyValueFromValue(newValueToDisplay);
    }
  const int decimals = d->decimalsForValue(newValueToDisplay);
  d->setValue(newValueToDisplay, decimals);
  const bool signalsEmitted = (newValue != d->InputValue);
  if (valueModified && !signalsEmitted)
    {
    emit valueChanged(d->InputValue);
    emit valueChanged(QString::number(d->InputValue, 'f', d->SpinBox->decimals()));
    }
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::stepUp()
{
  Q_D(const DoubleSpinBox);
  d->SpinBox->stepUp();
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::stepDown()
{
  Q_D(const DoubleSpinBox);
  d->SpinBox->stepDown();
}

//-----------------------------------------------------------------------------
DoubleSpinBox::SetMode DoubleSpinBox::setMode() const
{
  Q_D(const DoubleSpinBox);
  return d->Mode;
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setSetMode(DoubleSpinBox::SetMode newMode)
{
  Q_D(DoubleSpinBox);
  d->Mode = newMode;
}

//-----------------------------------------------------------------------------
DoubleSpinBox::DecimalsOptions DoubleSpinBox::decimalsOption()
{
  Q_D(const DoubleSpinBox);
  return d->DOption;
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::setDecimalsOption(DoubleSpinBox::DecimalsOptions option)
{
  Q_D(DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent && option == d->DOption)
    {
    return;
    }

  d->DOption = option;
  this->setValueAlways(this->value());
}

//----------------------------------------------------------------------------
void DoubleSpinBox::setInvertedControls(bool invertedControls)
{
  Q_D(DoubleSpinBox);
  d->InvertedControls = invertedControls;
  d->SpinBox->setInvertedControls(d->InvertedControls);
}

//----------------------------------------------------------------------------
bool DoubleSpinBox::invertedControls() const
{
  Q_D(const DoubleSpinBox);
  return d->InvertedControls;
}

//----------------------------------------------------------------------------
void DoubleSpinBox
::setSizeHintPolicy(DoubleSpinBox::SizeHintPolicy newSizeHintPolicy)
{
  Q_D(DoubleSpinBox);
  if (d->Mode == DoubleSpinBox::SetIfDifferent
      && newSizeHintPolicy == d->SizeHintPolicy)
    {
    return;
    }
  d->SizeHintPolicy = newSizeHintPolicy;
  d->CachedSizeHint = QSize();
  d->CachedMinimumSizeHint = QSize();
  this->updateGeometry();
}

//----------------------------------------------------------------------------
DoubleSpinBox::SizeHintPolicy DoubleSpinBox::sizeHintPolicy() const
{
  Q_D(const DoubleSpinBox);
  return d->SizeHintPolicy;
}

//----------------------------------------------------------------------------
void DoubleSpinBox::setValueProxy(ValueProxy* proxy)
{
  Q_D(DoubleSpinBox);
  if (proxy == d->Proxy.data())
    {
    return;
    }

  d->onValueProxyAboutToBeModified();

  if (d->Proxy)
    {
    disconnect(d->Proxy.data(), SIGNAL(proxyAboutToBeModified()),
               d, SLOT(onValueProxyAboutToBeModified()));
    disconnect(d->Proxy.data(), SIGNAL(proxyModified()),
               d, SLOT(onValueProxyModified()));
    }

  d->Proxy = proxy;

  if (d->Proxy)
    {
    connect(d->Proxy.data(), SIGNAL(proxyAboutToBeModified()),
            d, SLOT(onValueProxyAboutToBeModified()));
    connect(d->Proxy.data(), SIGNAL(proxyModified()),
            d, SLOT(onValueProxyModified()));
    }

  d->onValueProxyModified();
}

//----------------------------------------------------------------------------
ValueProxy* DoubleSpinBox::valueProxy() const
{
  Q_D(const DoubleSpinBox);
  return d->Proxy.data();
}

//----------------------------------------------------------------------------
QSize DoubleSpinBox::sizeHint() const
{
  Q_D(const DoubleSpinBox);
  if (d->SizeHintPolicy == DoubleSpinBox::SizeHintByMinMax)
    {
    return this->Superclass::sizeHint();
    }
  if (!d->CachedSizeHint.isEmpty())
    {
    return d->CachedSizeHint;
    }

  QSize newSizeHint;
  newSizeHint.setHeight(this->lineEdit()->sizeHint().height());

  QString extraString = " "; // give some room
  QString s = this->text() + extraString;
  s.truncate(18);
  int extraWidth = 2; // cursor width

  this->ensurePolished(); // ensure we are using the right font
  const QFontMetrics fm(this->fontMetrics());
#if (QT_VERSION >= QT_VERSION_CHECK(5,11,0))
  int width_in_pixels = fm.horizontalAdvance(s + extraString);
#else
  int width_in_pixels = fm.width(s + extraString);
#endif
  newSizeHint.setWidth(width_in_pixels + extraWidth);

  QStyleOptionSpinBox opt;
  d->SpinBox->initStyleOptionSpinBox(&opt);
#if QT_VERSION < QT_VERSION_CHECK(5,1,0)
  QSize extraSize(35, 6);
  opt.rect.setSize(newSizeHint + extraSize);
  extraSize += newSizeHint - this->style()->subControlRect(
      QStyle::CC_SpinBox, &opt,
      QStyle::SC_SpinBoxEditField, this).size();
  // Converging size hint...
  opt.rect.setSize(newSizeHint + extraSize);
  extraSize += newSizeHint - this->style()->subControlRect(
      QStyle::CC_SpinBox, &opt,
      QStyle::SC_SpinBoxEditField, this).size();
  newSizeHint += extraSize;
#endif
  opt.rect = this->rect();
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
  d->CachedSizeHint = this->style()
                          ->sizeFromContents(QStyle::CT_SpinBox, &opt, newSizeHint, this);
#else
  d->CachedSizeHint = this->style()
                          ->sizeFromContents(QStyle::CT_SpinBox, &opt, newSizeHint, this)
                          .expandedTo(QApplication::globalStrut());
#endif
  return d->CachedSizeHint;
}

//----------------------------------------------------------------------------
QSize DoubleSpinBox::minimumSizeHint() const
{
    Q_D(const DoubleSpinBox);
    if (d->SizeHintPolicy == DoubleSpinBox::SizeHintByMinMax)
    {
        // For some reasons, Superclass::minimumSizeHint() returns the spinbox
        // sizeHint()
        return this->spinBox()->minimumSizeHint();
    }
    if (!d->CachedMinimumSizeHint.isEmpty())
    {
        return d->CachedMinimumSizeHint;
    }
    QSize newSizeHint;
    newSizeHint.setHeight(this->lineEdit()->minimumSizeHint().height());
    QString extraString = " "; // give some room
    QString s = this->text() + extraString;
    s.truncate(18);
    int extraWidth = 2; // cursor width

    this->ensurePolished(); // ensure we are using the right font
    const QFontMetrics fm(this->fontMetrics());

#if (QT_VERSION >= QT_VERSION_CHECK(5,11,0))
    int width_in_pixels = fm.horizontalAdvance(s + extraString);
#else
    int width_in_pixels = fm.width(s + extraString);
#endif
    newSizeHint.setWidth(width_in_pixels + extraWidth);

    QStyleOptionSpinBox opt;
    d->SpinBox->initStyleOptionSpinBox(&opt);
#if QT_VERSION < QT_VERSION_CHECK(5,1,0)
    QSize extraSize(35, 6);
    opt.rect.setSize(newSizeHint + extraSize);
    extraSize += newSizeHint - this->style()->subControlRect(
        QStyle::CC_SpinBox, &opt,
        QStyle::SC_SpinBoxEditField, this).size();
    // Converging size hint...
    opt.rect.setSize(newSizeHint + extraSize);
    extraSize += newSizeHint - this->style()->subControlRect(
        QStyle::CC_SpinBox, &opt,
        QStyle::SC_SpinBoxEditField, this).size();
    newSizeHint += extraSize;
#endif
    opt.rect = this->rect();
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
    d->CachedMinimumSizeHint = this->style()
                                   ->sizeFromContents(QStyle::CT_SpinBox, &opt, newSizeHint, this);
#else
    d->CachedMinimumSizeHint = this->style()
                                   ->sizeFromContents(QStyle::CT_SpinBox, &opt, newSizeHint, this)
                                   .expandedTo(QApplication::globalStrut());
#endif
    return d->CachedMinimumSizeHint;
}

//-----------------------------------------------------------------------------
void DoubleSpinBox::keyPressEvent(QKeyEvent* event)
{
  Q_D(DoubleSpinBox);
  const bool accept = this->eventFilter(d->SpinBox, event);
  event->setAccepted(accept);
}

//-----------------------------------------------------------------------------
bool DoubleSpinBox::eventFilter(QObject* obj, QEvent* event)
{
  Q_D(DoubleSpinBox);
  if (d->DOption & DoubleSpinBox::DecimalsByShortcuts &&
    obj == d->SpinBox && event->type() == QEvent::KeyPress)
    {
    QKeyEvent* keyEvent = static_cast<QKeyEvent*>(event);
    Q_ASSERT(keyEvent);
    int newDecimals = -1;
    if (keyEvent->modifiers() & Qt::ControlModifier)
      {
      if (keyEvent->key() == Qt::Key_Plus
        || keyEvent->key() == Qt::Key_Equal
        || keyEvent->key() == Qt::Key_BracketRight) //bracketRight is a workaround, since Ctrl+ is sometimes not properly recognized: http://www.qtforum.org/article/36551/wrong-value-for-ctrl-plus-in-keypressevent.html
        {
        newDecimals = this->decimals() + 1;
        }
      else if (keyEvent->key() == Qt::Key_Minus)
        {
        newDecimals = this->decimals() - 1;
        }
      else if (keyEvent->key() == Qt::Key_0)
        {
        newDecimals = d->DefaultDecimals;
        }
      }
    if (newDecimals != -1)
      {
      double currentValue = this->value();
      if (d->Proxy)
        {
        currentValue = d->Proxy.data()->proxyValueFromValue(currentValue);
        }
      // increasing the number of decimals should restore lost precision
      d->setValue(currentValue, newDecimals);
      return true;
      }
    return QWidget::eventFilter(obj, event);
    }
  else if (obj == d->SpinBox && event->type() == QEvent::FocusOut)
  {
      // pass the event on to the parent class
      return QWidget::eventFilter(obj, event);
  }
  else
  {
    // pass the event on to the parent class
    return QWidget::eventFilter(obj, event);
  }
}
