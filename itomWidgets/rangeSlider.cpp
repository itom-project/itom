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
#include <QKeyEvent>
#include <QStyleOptionSlider>
#include <QApplication>
#include <QStylePainter>
#include <QStyle>
#include <QToolTip>

//  includes
#include "rangeSlider.h"
#include "../common/paramMeta.h"

class RangeSliderPrivate
{
  Q_DECLARE_PUBLIC(RangeSlider);
protected:
  RangeSlider* const q_ptr;
public:
  /// Boolean indicates the selected handle
  ///   True for the minimum range handle, false for the maximum range handle
  enum Handle {
    NoHandle = 0x0,
    MinimumHandle = 0x1,
    MaximumHandle = 0x2
  };
  Q_DECLARE_FLAGS(Handles, Handle);

  RangeSliderPrivate(RangeSlider& object);
  void init();

  /// Return the handle at the given pos, or none if no handle is at the pos.
  /// If a handle is selected, handleRect is set to the handle rect.
  /// otherwise return NoHandle and handleRect is set to the combined rect of
  /// the min and max handles
  Handle handleAtPos(const QPoint& pos, QRect& handleRect)const;

  qint64 bound(qint64 min, qint64 max, qint64 step, qint64 value, bool snapToBoundaries = true) const;
  uint boundUnsigned(uint min, uint max, uint step, uint value, bool snapToBoundaries = true) const;
  void rangeBound(int valLimitMin, int valLimitMax, Handle handleChangePriority, int &valMin, int &valMax);

  /// Copied verbatim from QSliderPrivate class (see QSlider.cpp)
  int pixelPosToRangeValue(int pos) const;
  int pixelPosFromRangeValue(int val) const;

  /// Draw the bottom and top sliders.
  void drawMinimumSlider( QStylePainter* painter, const QRect &rect ) const;
  void drawMaximumSlider( QStylePainter* painter, const QRect &rect ) const;
  void drawMinMaxSlider(QStylePainter *painter, const QRect &rect, bool isSliderDown, QStyleOptionSlider &option) const;

  /// End points of the range on the Model
  int m_MaximumValue;
  int m_MinimumValue;

  /// End points of the range on the GUI. This is synced with the model.
  int m_MaximumPosition;
  int m_MinimumPosition;

  uint m_PositionStepSize;
  uint m_MinimumRange;
  uint m_MaximumRange;
  uint m_StepSizeRange;
  bool m_RangeIncludesLimits;

  /// Controls selected ?
  QStyle::SubControl m_MinimumSliderSelected;
  QStyle::SubControl m_MaximumSliderSelected;

  /// See QSliderPrivate::clickOffset.
  /// Overrides this ivar
  int m_SubclassClickOffset;

  /// See QSliderPrivate::position
  /// Overrides this ivar.
  int m_SubclassPosition;

  /// Original width between the 2 bounds before any moves
  float m_SubclassWidth;

  RangeSliderPrivate::Handles m_SelectedHandles;

  /// When symmetricMoves is true, moving a handle will move the other handle
  /// symmetrically, otherwise the handles are independent.
  bool m_SymmetricMoves;

  bool m_UseStyleSheets;

  float m_HandleBorderRadius;

  QString m_HandleToolTip;

private:
  Q_DISABLE_COPY(RangeSliderPrivate);
};

// --------------------------------------------------------------------------
RangeSliderPrivate::RangeSliderPrivate(RangeSlider& object)
  :q_ptr(&object)
{
  this->m_MinimumValue = 0;
  this->m_MaximumValue = 100;
  this->m_MinimumPosition = 0;
  this->m_MaximumPosition = 100;
  this->m_MinimumSliderSelected = QStyle::SC_None;
  this->m_MaximumSliderSelected = QStyle::SC_None;
  this->m_SubclassClickOffset = 0;
  this->m_SubclassPosition = 0;
  this->m_SubclassWidth = 0.0;
  this->m_SelectedHandles = RangeSliderPrivate::Handle();
  this->m_SymmetricMoves = false;
  this->m_UseStyleSheets = false; // per default, the default OS depending styling is enabled.
  this->m_HandleBorderRadius = 0.0;

  this->m_PositionStepSize = 1;
  this->m_MinimumRange = 0;
  this->m_RangeIncludesLimits = false;
  this->m_MaximumRange = (this->m_RangeIncludesLimits ? 1 : 0) + this->m_MaximumValue - this->m_MinimumValue;
  this->m_StepSizeRange = 1;

}

// --------------------------------------------------------------------------
void RangeSliderPrivate::init()
{
  Q_Q(RangeSlider);
  this->m_MinimumValue = q->minimum();
  this->m_MaximumValue = q->maximum();
  this->m_MinimumPosition = q->minimum();
  this->m_MaximumPosition = q->maximum();

  this->m_MinimumRange = 0;
  this->m_MaximumRange = (uint)((this->m_RangeIncludesLimits ? 1 : 0) + (qint64)this->m_MaximumValue - (qint64)this->m_MinimumValue);

  q->connect(q, SIGNAL(rangeChanged(int,int)), q, SLOT(onRangeChanged(int,int)));
}

// --------------------------------------------------------------------------
RangeSliderPrivate::Handle RangeSliderPrivate::handleAtPos(const QPoint& pos, QRect& handleRect)const
{
  Q_Q(const RangeSlider);

  QStyleOptionSlider option;
  q->initStyleOption( &option );

  // The functinos hitTestComplexControl only know about 1 handle. As we have
  // 2, we change the position of the handle and test if the pos correspond to
  // any of the 2 positions.

  // Test the MinimumHandle
  option.sliderPosition = this->m_MinimumPosition;
  option.sliderValue    = this->m_MinimumValue;

  QStyle::SubControl minimumControl = q->style()->hitTestComplexControl(
    QStyle::CC_Slider, &option, pos, q);
  QRect minimumHandleRect = q->style()->subControlRect(
      QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, q);

  // Test if the pos is under the Maximum handle
  option.sliderPosition = this->m_MaximumPosition;
  option.sliderValue    = this->m_MaximumValue;

  QStyle::SubControl maximumControl = q->style()->hitTestComplexControl(
    QStyle::CC_Slider, &option, pos, q);
  QRect maximumHandleRect = q->style()->subControlRect(
      QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, q);

  // The pos is above both handles, select the closest handle
  if (minimumControl == QStyle::SC_SliderHandle &&
      maximumControl == QStyle::SC_SliderHandle)
    {
    int minDist = 0;
    int maxDist = 0;
    if (q->orientation() == Qt::Horizontal)
      {
      minDist = pos.x() - minimumHandleRect.left();
      maxDist = maximumHandleRect.right() - pos.x();
      }
    else //if (q->orientation() == Qt::Vertical)
      {
      minDist = minimumHandleRect.bottom() - pos.y();
      maxDist = pos.y() - maximumHandleRect.top();
      }
    Q_ASSERT( minDist >= 0 && maxDist >= 0);
    minimumControl = minDist < maxDist ? minimumControl : QStyle::SC_None;
    }

  if (minimumControl == QStyle::SC_SliderHandle)
    {
    handleRect = minimumHandleRect;
    return MinimumHandle;
    }
  else if (maximumControl == QStyle::SC_SliderHandle)
    {
    handleRect = maximumHandleRect;
    return MaximumHandle;
    }
  handleRect = minimumHandleRect.united(maximumHandleRect);
  return NoHandle;
}

// --------------------------------------------------------------------------
qint64 RangeSliderPrivate::bound(qint64 min, qint64 max, qint64 step, qint64 value, bool snapToBoundaries /*= true*/) const
{
    if (step == 1)
    {
        return qBound(min, value, max);
    }
    else
    {
        value = qBound(min, value, max);

        //try to round to nearest value following the step size
        qint64 remainder = (value - min) % step;
        if (remainder == 0)
        {
            //value = value;
        }
        else if (snapToBoundaries && ((value - remainder) == min))
        {
            value = min;
        }
        else if (snapToBoundaries && ((value + (step - remainder)) == max))
        {
            value = max;
        }
        else if (remainder > (step/2)) //we want to round up
        {
            //check upper limit
            if (value + remainder <= max)
            {
                //not exceeded, then go to the next upper step value
                value += (step - remainder);
            }
            else
            {
                //decrementing is always allowed
                value -= remainder;
            }
        }
        else
        {
            //decrementing is always allowed
            value -= remainder;
        }
        return value;
    }
}

// --------------------------------------------------------------------------
uint RangeSliderPrivate::boundUnsigned(uint min, uint max, uint step, uint value, bool snapToBoundaries /*= true*/) const
{
    if (step == 1)
    {
        return qBound(min, value, max);
    }
    else
    {
        value = qBound(min, value, max);

        //try to round to nearest value following the step size
        uint remainder = (value - min) % step;
        if (remainder == 0)
        {
            //value = value;
        }
        else if (snapToBoundaries && ((value - remainder) == min))
        {
            value = min;
        }
        else if (snapToBoundaries && ((value + (step - remainder)) == max))
        {
            value = max;
        }
        else if (remainder > (step/2)) //we want to round up
        {
            //check upper limit
            if (value + remainder <= max)
            {
                //not exceeded, then go to the next upper step value
                value += (step - remainder);
            }
            else
            {
                //decrementing is always allowed
                value -= remainder;
            }
        }
        else
        {
            //decrementing is always allowed
            value -= remainder;
        }
        return value;
    }
}

// --------------------------------------------------------------------------
void RangeSliderPrivate::rangeBound(int valLimitMin, int valLimitMax, Handle handleChangePriority, int &valMin, int &valMax)
{
    int offset = (this->m_RangeIncludesLimits ? 1 : 0);
    int range = offset + valMax - valMin;

    //try to fix left boundary and move right one
    if (handleChangePriority == MaximumHandle || (handleChangePriority == NoHandle && (qAbs(valLimitMin - valMin) < qAbs(valLimitMax - valMax))))
    {
        valMin = bound(valLimitMin, valLimitMax - m_MinimumRange, m_PositionStepSize, valMin);
        qint64 maxRange = bound((qint64)m_MinimumRange, (qint64)valLimitMax - (qint64)valMin + (qint64)offset, (qint64)m_StepSizeRange, (qint64)m_MaximumRange);
        range = bound((qint64)m_MinimumRange, (qint64)maxRange, (qint64)m_StepSizeRange, (qint64)valMax - (qint64)valMin + (qint64)offset, false);
        valMax = valMin + range - offset;
    }
    else //try to fix right boundary and move left one
    {
        valMax = bound(qMin((qint64)valLimitMin + (qint64)m_MinimumRange - (qint64)offset, (qint64)valLimitMax), valLimitMax, m_PositionStepSize, valMax);
        qint64 maxRange = bound((qint64)m_MinimumRange, (qint64)valMax - (qint64)valLimitMin + (qint64)offset, (qint64)m_StepSizeRange, (qint64)m_MaximumRange);
        range = bound((qint64)m_MinimumRange, (qint64)maxRange, (qint64)m_StepSizeRange, (qint64)valMax - (qint64)valMin + (qint64)offset, false);
        valMin = valMax - range + offset;
    }
}

// --------------------------------------------------------------------------
// Copied verbatim from QSliderPrivate::pixelPosToRangeValue. See QSlider.cpp
//
int RangeSliderPrivate::pixelPosToRangeValue( int pos ) const
{
  Q_Q(const RangeSlider);
  QStyleOptionSlider option;
  q->initStyleOption( &option );

  QRect gr = q->style()->subControlRect( QStyle::CC_Slider,
                                            &option,
                                            QStyle::SC_SliderGroove,
                                            q );
  QRect sr = q->style()->subControlRect( QStyle::CC_Slider,
                                            &option,
                                            QStyle::SC_SliderHandle,
                                            q );
  int sliderMin, sliderMax, sliderLength;
  if (option.orientation == Qt::Horizontal)
    {
    sliderLength = sr.width();
    sliderMin = gr.x();
    sliderMax = gr.right() - sliderLength + 1;
    }
  else
    {
    sliderLength = sr.height();
    sliderMin = gr.y();
    sliderMax = gr.bottom() - sliderLength + 1;
    }

  return QStyle::sliderValueFromPosition( q->minimum(),
                                          q->maximum(),
                                          pos - sliderMin,
                                          sliderMax - sliderMin,
                                          option.upsideDown );
}

//---------------------------------------------------------------------------
int RangeSliderPrivate::pixelPosFromRangeValue( int val ) const
{
  Q_Q(const RangeSlider);
  QStyleOptionSlider option;
  q->initStyleOption( &option );

  QRect gr = q->style()->subControlRect( QStyle::CC_Slider,
                                            &option,
                                            QStyle::SC_SliderGroove,
                                            q );
  QRect sr = q->style()->subControlRect( QStyle::CC_Slider,
                                            &option,
                                            QStyle::SC_SliderHandle,
                                            q );
  int sliderMin, sliderMax, sliderLength;
  if (option.orientation == Qt::Horizontal)
    {
    sliderLength = sr.width();
    sliderMin = gr.x();
    sliderMax = gr.right() - sliderLength + 1;
    }
  else
    {
    sliderLength = sr.height();
    sliderMin = gr.y();
    sliderMax = gr.bottom() - sliderLength + 1;
    }

  return QStyle::sliderPositionFromValue( q->minimum(),
                                          q->maximum(),
                                          val,
                                          sliderMax - sliderMin,
                                          option.upsideDown ) + sliderMin;
}

//---------------------------------------------------------------------------
void RangeSliderPrivate::drawMinMaxSlider(QStylePainter *painter, const QRect &rect, bool isSliderDown, QStyleOptionSlider &option) const
{
    Q_Q(const RangeSlider);

    option.subControls = QStyle::SC_SliderHandle;

    if (isSliderDown)
    {
        option.activeSubControls = QStyle::SC_SliderHandle;
        option.state |= QStyle::State_Sunken;
    }
#ifdef Q_OS_MAC
    // On mac style, drawing just the handle actually draws also the groove.
    QRect clip = q->style()->subControlRect(QStyle::CC_Slider, &option,
        QStyle::SC_SliderHandle, q);
    painter->setClipRect(clip);
#endif

    if (m_UseStyleSheets)
    {
        // the filling is given by the following css styles:
        // if the slider is down: 'selection-color', else 'color'.
        // For a disabled widget, us the disabled selector.
        QColor filling;
        QColor pen;

        if (isSliderDown)
        {
            if (q->isEnabled())
            {
                filling = q->palette().color(QPalette::Normal, QPalette::HighlightedText);
            }
            else
            {
                filling = q->palette().color(QPalette::Disabled, QPalette::HighlightedText);
            }
        }
        else
        {
            if (q->isEnabled())
            {
                filling = q->palette().color(QPalette::Normal, QPalette::WindowText);
            }
            else
            {
                filling = q->palette().color(QPalette::Disabled, QPalette::WindowText);
            }
        }

        pen = filling;

        if (q->hasFocus())
        {
            if (q->isEnabled())
            {
                pen = q->palette().color(QPalette::Normal, QPalette::HighlightedText);
            }
            else
            {
                pen = q->palette().color(QPalette::Disabled, QPalette::HighlightedText);
            }
        }

        painter->setPen(QPen(pen, 1));
        painter->setBrush(filling);

        bool wasAntialiased = painter->renderHints() & QPainter::Antialiasing;
        painter->setRenderHint(QPainter::Antialiasing);

        if (m_HandleBorderRadius <= 0.001)
        {
            painter->drawRect(rect);
        }
        else
        {
            painter->drawRoundedRect(rect, m_HandleBorderRadius, m_HandleBorderRadius);
        }

        painter->setRenderHint(QPainter::Antialiasing, wasAntialiased);
    }
    else
    {
        painter->drawComplexControl(QStyle::CC_Slider, option);
    }
}

//---------------------------------------------------------------------------
// Draw slider at the bottom end of the range
void RangeSliderPrivate::drawMinimumSlider( QStylePainter* painter, const QRect &rect ) const
{
  Q_Q(const RangeSlider);
  QStyleOptionSlider option;
  q->initMinimumSliderStyleOption( &option );

  option.sliderValue = m_MinimumValue;
  option.sliderPosition = m_MinimumPosition;

  drawMinMaxSlider(painter, rect, q->isMinimumSliderDown(), option);
}

//---------------------------------------------------------------------------
// Draw slider at the top end of the range
void RangeSliderPrivate::drawMaximumSlider(QStylePainter* painter, const QRect &rect) const
{
    Q_Q(const RangeSlider);
    QStyleOptionSlider option;
    q->initMaximumSliderStyleOption(&option);

    option.sliderValue = m_MaximumValue;
    option.sliderPosition = m_MaximumPosition;

    drawMinMaxSlider(painter, rect, q->isMaximumSliderDown(), option);
}

// --------------------------------------------------------------------------
RangeSlider::RangeSlider(QWidget* _parent)
  : QSlider(_parent)
  , d_ptr(new RangeSliderPrivate(*this))
{
  Q_D(RangeSlider);
  d->init();
}

// --------------------------------------------------------------------------
RangeSlider::RangeSlider( Qt::Orientation o,
                                  QWidget* parentObject )
  :QSlider(o, parentObject)
  , d_ptr(new RangeSliderPrivate(*this))
{
  Q_D(RangeSlider);
  d->init();
}

// --------------------------------------------------------------------------
RangeSlider::RangeSlider(RangeSliderPrivate* impl, QWidget* _parent)
  : QSlider(_parent)
  , d_ptr(impl)
{
  Q_D(RangeSlider);
  d->init();
}

// --------------------------------------------------------------------------
RangeSlider::RangeSlider( RangeSliderPrivate* impl, Qt::Orientation o,
                                QWidget* parentObject )
  :QSlider(o, parentObject)
  , d_ptr(impl)
{
  Q_D(RangeSlider);
  d->init();
}

// --------------------------------------------------------------------------
RangeSlider::~RangeSlider()
{
}

// --------------------------------------------------------------------------
int RangeSlider::minimumValue() const
{
  Q_D(const RangeSlider);
  return d->m_MinimumValue;
}

// --------------------------------------------------------------------------
void RangeSlider::setMinimumValue( int min )
{
  Q_D(RangeSlider);
  this->setValues( min, qMax(d->m_MaximumValue,min) );
}

// --------------------------------------------------------------------------
int RangeSlider::maximumValue() const
{
  Q_D(const RangeSlider);
  return d->m_MaximumValue;
}

// --------------------------------------------------------------------------
void RangeSlider::setMaximumValue( int max )
{
  Q_D(RangeSlider);
  this->setValues( qMin(d->m_MinimumValue, max), max );
}

// --------------------------------------------------------------------------
uint RangeSlider::stepSizePosition() const
{
  Q_D(const RangeSlider);
  return d->m_PositionStepSize;
}

// --------------------------------------------------------------------------
void RangeSlider::setStepSizePosition(uint stepSize)
{
  Q_D(RangeSlider);
  d->m_PositionStepSize = stepSize;
  d->m_StepSizeRange = d->boundUnsigned(stepSize, std::numeric_limits<uint>::max(), stepSize, d->m_StepSizeRange);
  this->setValues( d->m_MinimumValue, d->m_MaximumValue );
}

// --------------------------------------------------------------------------
uint RangeSlider::minimumRange() const
{
  Q_D(const RangeSlider);
  return d->m_MinimumRange;
}

// --------------------------------------------------------------------------
void RangeSlider::setMinimumRange(uint min)
{
  Q_D(RangeSlider);
  d->m_MinimumRange = d->boundUnsigned(0, d->m_MaximumRange, d->m_StepSizeRange, min);
  this->setValues( d->m_MinimumValue, d->m_MaximumValue );
}

// --------------------------------------------------------------------------
uint RangeSlider::maximumRange() const
{
  Q_D(const RangeSlider);
  return d->m_MaximumRange;
}

// --------------------------------------------------------------------------
void RangeSlider::setMaximumRange(uint max)
{
  Q_D(RangeSlider);
  d->m_MaximumRange = d->boundUnsigned(d->m_MinimumRange, qMin(max, std::numeric_limits<uint>::max() - d->m_StepSizeRange) + d->m_StepSizeRange, d->m_StepSizeRange, max);
  this->setValues( d->m_MinimumValue, d->m_MaximumValue );
}

// --------------------------------------------------------------------------
uint RangeSlider::stepSizeRange() const
{
  Q_D(const RangeSlider);
  return d->m_StepSizeRange;
}

// --------------------------------------------------------------------------
void RangeSlider::setStepSizeRange(uint stepSize)
{
  Q_D(RangeSlider);
  d->m_StepSizeRange = d->boundUnsigned(d->m_PositionStepSize, std::numeric_limits<uint>::max(), d->m_PositionStepSize, stepSize);
  d->m_MaximumRange = d->bound(d->m_MinimumRange, qMin(d->m_MaximumRange, std::numeric_limits<uint>::max() - stepSize) + stepSize, stepSize, d->m_MaximumRange);
  this->setValues( d->m_MinimumValue, d->m_MaximumValue );
}

// --------------------------------------------------------------------------
bool RangeSlider::rangeIncludeLimits() const
{
  Q_D(const RangeSlider);
  return d->m_RangeIncludesLimits;
}

// --------------------------------------------------------------------------
void RangeSlider::setRangeIncludeLimits(bool include)
{
  Q_D(RangeSlider);
  d->m_RangeIncludesLimits = include;
  this->setValues( d->m_MinimumValue, d->m_MaximumValue );
}

// --------------------------------------------------------------------------
void RangeSlider::setValues(int l, int u)
{
  Q_D(RangeSlider);
  int minValue = qMin(l,u);
  int maxValue = qMax(l,u);

  RangeSliderPrivate::Handle handleChangePriority = RangeSliderPrivate::NoHandle; //both handles per default
  if ((minValue != d->m_MinimumValue) && maxValue == d->m_MaximumValue)
  {
      handleChangePriority = RangeSliderPrivate::MinimumHandle;
  }
  else if ((minValue == d->m_MinimumValue) && maxValue != d->m_MaximumValue)
  {
      handleChangePriority = RangeSliderPrivate::MaximumHandle;
  }

  d->rangeBound(this->minimum(), this->maximum(), handleChangePriority, minValue, maxValue);

  bool emitMinValChanged = (minValue != d->m_MinimumValue);
  bool emitMaxValChanged = (maxValue != d->m_MaximumValue);

  d->m_MinimumValue = minValue;
  d->m_MaximumValue = maxValue;

  bool emitMinPosChanged =
    (minValue != d->m_MinimumPosition);
  bool emitMaxPosChanged =
    (maxValue != d->m_MaximumPosition);
  d->m_MinimumPosition = minValue;
  d->m_MaximumPosition = maxValue;

  if (isSliderDown())
    {
    if (emitMinPosChanged || emitMaxPosChanged)
      {
      emit positionsChanged(d->m_MinimumPosition, d->m_MaximumPosition);
      }
    if (emitMinPosChanged)
      {
      emit minimumPositionChanged(d->m_MinimumPosition);
      }
    if (emitMaxPosChanged)
      {
      emit maximumPositionChanged(d->m_MaximumPosition);
      }
    }
  if (emitMinValChanged || emitMaxValChanged)
    {
    emit valuesChanged(d->m_MinimumValue,
                       d->m_MaximumValue);
    }
  if (emitMinValChanged)
    {
    emit minimumValueChanged(d->m_MinimumValue);
    }
  if (emitMaxValChanged)
    {
    emit maximumValueChanged(d->m_MaximumValue);
    }
  if (emitMinPosChanged || emitMaxPosChanged ||
      emitMinValChanged || emitMaxValChanged)
    {
    this->update();
    }
}

// --------------------------------------------------------------------------
int RangeSlider::minimumPosition() const
{
  Q_D(const RangeSlider);
  return d->m_MinimumPosition;
}

// --------------------------------------------------------------------------
int RangeSlider::maximumPosition() const
{
  Q_D(const RangeSlider);
  return d->m_MaximumPosition;
}

// --------------------------------------------------------------------------
void RangeSlider::setMinimumPosition(int l)
{
  Q_D(const RangeSlider);
  this->setPositions(l, qMax(l, d->m_MaximumPosition));
}

// --------------------------------------------------------------------------
void RangeSlider::setMaximumPosition(int u)
{
  Q_D(const RangeSlider);
  this->setPositions(qMin(d->m_MinimumPosition, u), u);
}

// --------------------------------------------------------------------------
void RangeSlider::setPositions(int min, int max)
{
  Q_D(RangeSlider);
  int minPosition = qMin(min,max);
  int maxPosition = qMax(min,max);

  RangeSliderPrivate::Handle handleChangePriority = RangeSliderPrivate::NoHandle; //both handles per default
  if ((minPosition != d->m_MinimumPosition) && maxPosition == d->m_MaximumPosition)
  {
      handleChangePriority = RangeSliderPrivate::MinimumHandle;
  }
  else if ((minPosition == d->m_MinimumPosition) && maxPosition != d->m_MaximumPosition)
  {
      handleChangePriority = RangeSliderPrivate::MaximumHandle;
  }

  d->rangeBound(this->minimum(), this->maximum(), handleChangePriority, minPosition, maxPosition);

  bool emitMinPosChanged = (minPosition != d->m_MinimumPosition);
  bool emitMaxPosChanged = (maxPosition != d->m_MaximumPosition);

  if (!emitMinPosChanged && !emitMaxPosChanged)
    {
    return;
    }

  d->m_MinimumPosition = minPosition;
  d->m_MaximumPosition = maxPosition;

  if (!this->hasTracking())
    {
    this->update();
    }
  if (isSliderDown())
    {
    if (emitMinPosChanged)
      {
      emit minimumPositionChanged(d->m_MinimumPosition);
      }
    if (emitMaxPosChanged)
      {
      emit maximumPositionChanged(d->m_MaximumPosition);
      }
    if (emitMinPosChanged || emitMaxPosChanged)
      {
      emit positionsChanged(d->m_MinimumPosition, d->m_MaximumPosition);
      }
    }
  if (this->hasTracking())
    {
    this->triggerAction(SliderMove);
    this->setValues(d->m_MinimumPosition, d->m_MaximumPosition);
    }
}

// --------------------------------------------------------------------------
void RangeSlider::setSymmetricMoves(bool symmetry)
{
  Q_D(RangeSlider);
  d->m_SymmetricMoves = symmetry;
}

// --------------------------------------------------------------------------
bool RangeSlider::symmetricMoves()const
{
  Q_D(const RangeSlider);
  return d->m_SymmetricMoves;
}

// --------------------------------------------------------------------------
bool RangeSlider::useStyleSheets()const
{
    Q_D(const RangeSlider);
    return d->m_UseStyleSheets;
}
// --------------------------------------------------------------------------
void RangeSlider::setUseStyleSheets(bool useStyleSheets)
{
    Q_D(RangeSlider);
    d->m_UseStyleSheets = useStyleSheets;
    this->update();
}

// --------------------------------------------------------------------------
float RangeSlider::handleBorderRadius()const
{
    Q_D(const RangeSlider);
    return d->m_HandleBorderRadius;
}
// --------------------------------------------------------------------------
void RangeSlider::setHandleBorderRadius(float radius)
{
    Q_D(RangeSlider);
    d->m_HandleBorderRadius = radius;
    this->update();
}

// --------------------------------------------------------------------------
void RangeSlider::onRangeChanged(int _minimum, int _maximum)
{
  Q_UNUSED(_minimum);
  Q_UNUSED(_maximum);
  Q_D(RangeSlider);
  this->setValues(d->m_MinimumValue, d->m_MaximumValue);
}

// --------------------------------------------------------------------------
// Render
void RangeSlider::paintEvent( QPaintEvent* )
{
  Q_D(RangeSlider);
  QStyleOptionSlider option;
  this->initStyleOption(&option);

  QStylePainter painter(this);
  option.subControls = QStyle::SC_SliderGroove;
  // Move to minimum to not highlight the SliderGroove.
  // On mac style, drawing just the slider groove also draws the handles,
  // therefore we give a negative (outside of view) position.
  option.sliderValue = this->minimum() - this->maximum();
  option.sliderPosition = this->minimum() - this->maximum();
  painter.drawComplexControl(QStyle::CC_Slider, option);

  option.sliderPosition = d->m_MinimumPosition;
  const QRect lr = style()->subControlRect( QStyle::CC_Slider,
                                            &option,
                                            QStyle::SC_SliderHandle,
                                            this);

  option.sliderPosition = d->m_MaximumPosition;
  const QRect ur = style()->subControlRect( QStyle::CC_Slider,
                                            &option,
                                            QStyle::SC_SliderHandle,
                                            this);

  QRect grooveTotal = style()->subControlRect( QStyle::CC_Slider,
                                      &option,
                                      QStyle::SC_SliderGroove,
                                      this);
  QRect rangeBox;

  if (useStyleSheets())
  {
      // it seems that if style sheets are used, the height of a horizontal groove
      // in grooveTotal corresponds to the real height of this groove. Without style sheets,
      // subControlRect seems to return the total height of the widget for the groove, too.
      if (option.orientation == Qt::Horizontal)
      {
          rangeBox = QRect(
              QPoint(qMin(lr.center().x(), ur.center().x()), grooveTotal.top()),
              QPoint(qMax(lr.center().x(), ur.center().x()), grooveTotal.bottom()));
      }
      else
      {
          rangeBox = QRect(
              QPoint(grooveTotal.left(), qMin(lr.center().y(), ur.center().y())),
              QPoint(grooveTotal.right(), qMax(lr.center().y(), ur.center().y())));
      }
  }
  else
  {
      if (option.orientation == Qt::Horizontal)
      {
          rangeBox = QRect(
              QPoint(qMin(lr.center().x(), ur.center().x()), grooveTotal.center().y() - 2),
              QPoint(qMax(lr.center().x(), ur.center().x()), grooveTotal.center().y() + 1));
      }
      else
      {
          rangeBox = QRect(
              QPoint(grooveTotal.center().x() - 2, qMin(lr.center().y(), ur.center().y())),
              QPoint(grooveTotal.center().x() + 1, qMax(lr.center().y(), ur.center().y())));
      }
  }


  // -----------------------------
  // Render the range
  //
  grooveTotal.adjust(0, 0, -1, 0);

  // Create default colors based on the transfer function.
  //
  // The main color of the groove within the two handes (active groove)
  // are given by the css selector 'selection-background-color' (either
  // the default RangeSlider section or the disabled section for disabled
  // widgets).
  //
  QColor highlight = isEnabled() ?
      this->palette().color(QPalette::Normal, QPalette::Highlight) :
      this->palette().color(QPalette::Disabled, QPalette::Highlight);

  if (!useStyleSheets())
  {
      QLinearGradient gradient;
      if (option.orientation == Qt::Horizontal)
      {
          gradient = QLinearGradient(grooveTotal.center().x(), grooveTotal.top(),
              grooveTotal.center().x(), grooveTotal.bottom());
      }
      else
      {
          gradient = QLinearGradient(grooveTotal.left(), grooveTotal.center().y(),
              grooveTotal.right(), grooveTotal.center().y());
      }

      // TODO: Set this based on the supplied transfer function
      //QColor l = Qt::darkGray;
      //QColor u = Qt::black;

      gradient.setColorAt(0, highlight.darker(120));
      gradient.setColorAt(1, highlight.lighter(160));

      painter.setBrush(gradient);
  }
  else
  {
    painter.setBrush(QBrush(highlight));
  }

  painter.setPen(QPen(highlight.darker(150)));
  painter.drawRect( rangeBox.intersected(grooveTotal) );

  //  -----------------------------------
  // Render the sliders
  //
    if (this->isMinimumSliderDown())
    {
        d->drawMaximumSlider(&painter, ur);
        d->drawMinimumSlider(&painter, lr);
    }
    else
    {
        d->drawMinimumSlider(&painter, lr);
        d->drawMaximumSlider(&painter, ur);
    }
}

// --------------------------------------------------------------------------
// Standard Qt UI events
void RangeSlider::mousePressEvent(QMouseEvent* mouseEvent)
{
  Q_D(RangeSlider);
  if (minimum() == maximum() || (mouseEvent->buttons() ^ mouseEvent->button()))
    {
    mouseEvent->ignore();
    return;
    }
  int mepos = this->orientation() == Qt::Horizontal ?
    mouseEvent->pos().x() : mouseEvent->pos().y();

  QStyleOptionSlider option;
  this->initStyleOption( &option );

  QRect handleRect;
  RangeSliderPrivate::Handle handle_ = d->handleAtPos(mouseEvent->pos(), handleRect);

  if (handle_ != RangeSliderPrivate::NoHandle)
    {
    d->m_SubclassPosition = (handle_ == RangeSliderPrivate::MinimumHandle)?
      d->m_MinimumPosition : d->m_MaximumPosition;

    // save the position of the mouse inside the handle for later
    d->m_SubclassClickOffset = mepos - (this->orientation() == Qt::Horizontal ?
      handleRect.left() : handleRect.top());

    this->setSliderDown(true);

    if (d->m_SelectedHandles != handle_)
      {
      d->m_SelectedHandles = handle_;
      this->update(handleRect);
      }
    // Accept the mouseEvent
    mouseEvent->accept();
    return;
    }

  // if we are here, no handles have been pressed
  // Check if we pressed on the groove between the 2 handles

  QStyle::SubControl control = this->style()->hitTestComplexControl(
    QStyle::CC_Slider, &option, mouseEvent->pos(), this);
  QRect sr = style()->subControlRect(
    QStyle::CC_Slider, &option, QStyle::SC_SliderGroove, this);
  int minCenter = (this->orientation() == Qt::Horizontal ?
    handleRect.left() : handleRect.top());
  int maxCenter = (this->orientation() == Qt::Horizontal ?
    handleRect.right() : handleRect.bottom());
  if (control == QStyle::SC_SliderGroove &&
      mepos > minCenter && mepos < maxCenter)
    {
    // warning lost of precision it might be fatal
    d->m_SubclassPosition = (d->m_MinimumPosition + d->m_MaximumPosition) / 2.;
    d->m_SubclassClickOffset = mepos - d->pixelPosFromRangeValue(d->m_SubclassPosition);
    d->m_SubclassWidth = (d->m_MaximumPosition - d->m_MinimumPosition) / 2.;
    qMax(d->m_SubclassPosition - d->m_MinimumPosition, d->m_MaximumPosition - d->m_SubclassPosition);
    this->setSliderDown(true);
    if (!this->isMinimumSliderDown() || !this->isMaximumSliderDown())
      {
      d->m_SelectedHandles =
        QFlags<RangeSliderPrivate::Handle>(RangeSliderPrivate::MinimumHandle) |
        QFlags<RangeSliderPrivate::Handle>(RangeSliderPrivate::MaximumHandle);
      this->update(handleRect.united(sr));
      }
    mouseEvent->accept();
    return;
    }
  mouseEvent->ignore();
}

// --------------------------------------------------------------------------
// Standard Qt UI events
void RangeSlider::mouseMoveEvent(QMouseEvent* mouseEvent)
{
  Q_D(RangeSlider);

  if (!d->m_SelectedHandles)
    {
    mouseEvent->ignore();
    return;
    }
  int mepos = this->orientation() == Qt::Horizontal ?
    mouseEvent->pos().x() : mouseEvent->pos().y();

  QStyleOptionSlider option;
  this->initStyleOption(&option);

  const int m = style()->pixelMetric( QStyle::PM_MaximumDragDistance, &option, this );

  int newPosition = d->pixelPosToRangeValue(mepos - d->m_SubclassClickOffset);

  if (m >= 0)
    {
    const QRect r = rect().adjusted(-m, -m, m, m);
    if (!r.contains(mouseEvent->pos()))
      {
      newPosition = d->m_SubclassPosition;
      }
    }

  // Only the lower/left slider is down
  if (this->isMinimumSliderDown() && !this->isMaximumSliderDown())
    {
    double newMinPos = qMin(newPosition,d->m_MaximumPosition);
    this->setPositions(newMinPos, d->m_MaximumPosition +
      (d->m_SymmetricMoves ? d->m_MinimumPosition - newMinPos : 0));
    }
  // Only the upper/right slider is down
  else if (this->isMaximumSliderDown() && !this->isMinimumSliderDown())
    {
    double newMaxPos = qMax(d->m_MinimumPosition, newPosition);
    this->setPositions(d->m_MinimumPosition -
      (d->m_SymmetricMoves ? newMaxPos - d->m_MaximumPosition: 0),
      newMaxPos);
    }
  // Both handles are down (the user clicked in between the handles)
  else if (this->isMinimumSliderDown() && this->isMaximumSliderDown())
    {
    this->setPositions(newPosition - static_cast<int>(d->m_SubclassWidth),
                       newPosition + static_cast<int>(d->m_SubclassWidth + .5));
    }
  mouseEvent->accept();
}

// --------------------------------------------------------------------------
// Standard Qt UI mouseEvents
void RangeSlider::mouseReleaseEvent(QMouseEvent* mouseEvent)
{
  Q_D(RangeSlider);
  this->QSlider::mouseReleaseEvent(mouseEvent);

  setSliderDown(false);
  d->m_SelectedHandles = RangeSliderPrivate::Handle();

  this->update();
}

// --------------------------------------------------------------------------
void RangeSlider::keyPressEvent(QKeyEvent* keyEvent)
{
    keyEvent->ignore();

    if (keyEvent->key() == Qt::Key_Left || keyEvent->key() == Qt::Key_Down)
    {
        keyEvent->accept();

        if (keyEvent->modifiers() & Qt::ShiftModifier)
        {
            int valMin = qMax(minimum(), minimumValue() - (int)stepSizePosition());
            int width = maximumValue() - minimumValue();

            setValues(valMin, valMin + width);
        }
        else if (keyEvent->key() == Qt::Key_Left)
        {
            int val = qMax(minimum(), minimumValue() - (int)stepSizePosition());
            setMinimumValue(val);
        }
        else
        {
            int val = qMax(maximumValue() - (int)stepSizePosition(), minimum());
            setMaximumValue(val);
        }
    }
    else if (keyEvent->key() == Qt::Key_Right || keyEvent->key() == Qt::Key_Up)
    {
        keyEvent->accept();

        if (keyEvent->modifiers() & Qt::ShiftModifier)
        {
            int valMax = qMin(maximum(), maximumValue() + (int)stepSizePosition());
            int width = maximumValue() - minimumValue();

            setValues(valMax - width, valMax);
        }
        else if (keyEvent->key() == Qt::Key_Right)
        {
            int val = qMin(maximum(), minimumValue() + (int)stepSizePosition());
            setMinimumValue(val);
        }
        else
        {
            int val = qMin(maximumValue() + (int)stepSizePosition(), maximum());
            setMaximumValue(val);
        }
    }

    if (!keyEvent->isAccepted())
    {
        this->Superclass::keyPressEvent(keyEvent);
    }
}

// --------------------------------------------------------------------------
bool RangeSlider::isMinimumSliderDown()const
{
  Q_D(const RangeSlider);
  return d->m_SelectedHandles & RangeSliderPrivate::MinimumHandle;
}

// --------------------------------------------------------------------------
bool RangeSlider::isMaximumSliderDown()const
{
  Q_D(const RangeSlider);
  return d->m_SelectedHandles & RangeSliderPrivate::MaximumHandle;
}

// --------------------------------------------------------------------------
void RangeSlider::initMinimumSliderStyleOption(QStyleOptionSlider* option) const
{
  this->initStyleOption(option);
}

// --------------------------------------------------------------------------
void RangeSlider::initMaximumSliderStyleOption(QStyleOptionSlider* option) const
{
  this->initStyleOption(option);
}

// --------------------------------------------------------------------------
QString RangeSlider::handleToolTip()const
{
  Q_D(const RangeSlider);
  return d->m_HandleToolTip;
}

// --------------------------------------------------------------------------
void RangeSlider::setHandleToolTip(const QString& _toolTip)
{
  Q_D(RangeSlider);
  d->m_HandleToolTip = _toolTip;
}

// --------------------------------------------------------------------------
bool RangeSlider::event(QEvent* _event)
{
  Q_D(RangeSlider);
  switch(_event->type())
    {
    case QEvent::ToolTip:
      {
      QHelpEvent* helpEvent = static_cast<QHelpEvent*>(_event);
      QStyleOptionSlider opt;
      // Test the MinimumHandle
      opt.sliderPosition = d->m_MinimumPosition;
      opt.sliderValue = d->m_MinimumValue;
      this->initStyleOption(&opt);
      QStyle::SubControl hoveredControl =
        this->style()->hitTestComplexControl(
          QStyle::CC_Slider, &opt, helpEvent->pos(), this);
      if (!d->m_HandleToolTip.isEmpty() &&
          hoveredControl == QStyle::SC_SliderHandle)
        {
        QToolTip::showText(helpEvent->globalPos(), d->m_HandleToolTip.arg(this->minimumValue()));
        _event->accept();
        return true;
        }
      // Test the MaximumHandle
      opt.sliderPosition = d->m_MaximumPosition;
      opt.sliderValue = d->m_MaximumValue;
      this->initStyleOption(&opt);
      hoveredControl = this->style()->hitTestComplexControl(
        QStyle::CC_Slider, &opt, helpEvent->pos(), this);
      if (!d->m_HandleToolTip.isEmpty() &&
          hoveredControl == QStyle::SC_SliderHandle)
        {
        QToolTip::showText(helpEvent->globalPos(), d->m_HandleToolTip.arg(this->maximumValue()));
        _event->accept();
        return true;
        }
      }
    default:
      break;
    }
  return this->Superclass::event(_event);
}

// --------------------------------------------------------------------------
void RangeSlider::setLimitsFromIntervalMeta(const ito::IntervalMeta &intervalMeta)
{
    Q_D(RangeSlider);

    d->m_RangeIncludesLimits = !intervalMeta.isIntervalNotRange();
    int offset = d->m_RangeIncludesLimits ? 1 : 0;

    //first: set step sizes, then: boundaries
    d->m_PositionStepSize = intervalMeta.getStepSize();
    d->m_StepSizeRange = d->bound(d->m_PositionStepSize, std::numeric_limits<int>::max(), d->m_PositionStepSize, intervalMeta.getSizeStepSize());

    blockSignals(true); //slider-changed signal should only be emitted after the setMaximum call (in order to avoid intermediate state, that can crash due to wrong Q_ASSERT checks!
    setMinimum(d->bound(0, intervalMeta.getMin() + d->m_PositionStepSize, d->m_PositionStepSize, intervalMeta.getMin()));
    blockSignals(false);
    setMaximum(d->bound(minimum(), intervalMeta.getMax() + d->m_PositionStepSize + offset, d->m_PositionStepSize, intervalMeta.getMax() + offset) - offset);

    d->m_MinimumRange = d->boundUnsigned(0, intervalMeta.getSizeMin() + d->m_StepSizeRange, d->m_StepSizeRange, intervalMeta.getSizeMin());
    setMaximumRange( std::min(intervalMeta.getMax() - intervalMeta.getMin() + offset,intervalMeta.getSizeMax()) ); //using setMaximumRange in order to finally adapt the current values to allowed values
}
