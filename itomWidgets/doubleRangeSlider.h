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
    Common framework (http://www.commontk.org)
*********************************************************************** */

#ifndef DOUBLERANGESLIDER_H
#define DOUBLERANGESLIDER_H

// Qt includes
#include <QWidget>
#include <QSlider>

#include "commonWidgets.h"

class RangeSlider;
class DoubleRangeSliderPrivate;
class ValueProxy;

/// \ingroup Widgets
/// DoubleRangeSlider is a slider that controls 2 numbers as double.
/// DoubleRangeSlider internally aggregates a RangeSlider (not in the
/// API to prevent misuse). Only subclasses can have access to it.
/// \sa RangeSlider, DoubleSlider, RangeWidget
class ITOMWIDGETS_EXPORT DoubleRangeSlider : public QWidget
{
  Q_OBJECT
  Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
  Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
  Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)
  Q_PROPERTY(double minimumValue READ minimumValue WRITE setMinimumValue)
  Q_PROPERTY(double maximumValue READ maximumValue WRITE setMaximumValue)
  Q_PROPERTY(double minimumPosition READ minimumPosition WRITE setMinimumPosition)
  Q_PROPERTY(double maximumPosition READ maximumPosition WRITE setMaximumPosition)
  Q_PROPERTY(bool tracking READ hasTracking WRITE setTracking)
  Q_PROPERTY(Qt::Orientation orientation READ orientation WRITE setOrientation)
  Q_PROPERTY(double tickInterval READ tickInterval WRITE setTickInterval)
  Q_PROPERTY(QSlider::TickPosition tickPosition READ tickPosition WRITE setTickPosition)
  Q_PROPERTY(bool symmetricMoves READ symmetricMoves WRITE setSymmetricMoves)
public:
  // Superclass typedef
  typedef QWidget Superclass;

  /// Constructor, builds a DoubleRangeSlider whose default values are the same
  /// as RangeSlider.
  DoubleRangeSlider( Qt::Orientation o, QWidget* par= 0 );

  /// Constructor, builds a DoubleRangeSlider whose default values are the same
  /// as RangeSlider.
  DoubleRangeSlider( QWidget* par = 0 );

  /// Destructor
  virtual ~DoubleRangeSlider();

  ///
  /// This property holds the single step.
  /// The smaller of two natural steps that an abstract sliders provides and
  /// typically corresponds to the user pressing an arrow key
  /// \sa isValidStep()
  void setSingleStep(double ss);
  double singleStep()const;

  /// Return true if the step can be handled by the slider, false otherwise.
  /// An invalid step is a step that can't be used to convert from double
  /// to int (too large or too small).
  /// \sa singleStep
  bool isValidStep(double step)const;

  ///
  /// This property holds the interval between tickmarks.
  /// This is a value interval, not a pixel interval. If it is 0, the slider
  /// will choose between lineStep() and pageStep().
  /// The default value is 0.
  void setTickInterval(double ti);
  double tickInterval()const;

  ///
  /// This property holds the tickmark position for this slider.
  /// The valid values are described by the QSlider::TickPosition enum.
  /// The default value is QSlider::NoTicks.
  void setTickPosition(QSlider::TickPosition position);
  QSlider::TickPosition tickPosition()const;

  ///
  /// This property holds the sliders's minimum value.
  /// When setting this property, the maximum is adjusted if necessary to
  /// ensure that the range remains valid. Also the slider's current values
  /// are adjusted to be within the new range.
  double minimum()const;
  void setMinimum(double min);

  ///
  /// This property holds the slider's maximum value.
  /// When setting this property, the minimum is adjusted if necessary to
  /// ensure that the range remains valid. Also the slider's current values
  /// are adjusted to be within the new range.
  double maximum()const;
  void setMaximum(double max);

  ///
  /// Sets the slider's minimum to min and its maximum to max.
  /// If max is smaller than min, min becomes the only legal value.
  void setRange(double min, double max);

  ///
  /// This property holds the slider's current minimum value.
  /// The slider forces the minimum value to be within the legal range:
  /// minimum <= minvalue <= maxvalue <= maximum.
  /// Changing the minimumValue also changes the minimumPosition.
  double minimumValue() const;

  ///
  /// This property holds the slider's current maximum value.
  /// The slider forces the maximum value to be within the legal range:
  /// minimum <= minvalue <= maxvalue <= maximum.
  /// Changing the maximumValue also changes the maximumPosition.
  double maximumValue() const;

  ///
  /// This property holds the current slider minimum position.
  /// If tracking is enabled (the default), this is identical to minimumValue.
  double minimumPosition() const;
  void setMinimumPosition(double minPos);

  ///
  /// This property holds the current slider maximum position.
  /// If tracking is enabled (the default), this is identical to maximumValue.
  double maximumPosition() const;
  void setMaximumPosition(double maxPos);

  ///
  /// Utility function that set the minimum position and
  /// maximum position at once.
  void setPositions(double minPos, double maxPos);

  ///
  /// This property holds whether slider tracking is enabled.
  /// If tracking is enabled (the default), the slider emits the minimumValueChanged()
  /// signal while the left/bottom handler is being dragged and the slider emits
  /// the maximumValueChanged() signal while the right/top handler is being dragged.
  /// If tracking is disabled, the slider emits the minimumValueChanged()
  /// and maximumValueChanged() signals only when the user releases the slider.
  void setTracking(bool enable);
  bool hasTracking()const;

  ///
  /// Triggers a slider action on the current slider. Possible actions are
  /// SliderSingleStepAdd, SliderSingleStepSub, SliderPageStepAdd,
  /// SliderPageStepSub, SliderToMinimum, SliderToMaximum, and SliderMove.
  void triggerAction(QAbstractSlider::SliderAction action);

  ///
  /// This property holds the orientation of the slider.
  /// The orientation must be Qt::Vertical (the default) or Qt::Horizontal.
  Qt::Orientation orientation()const;
  void setOrientation(Qt::Orientation orientation);

  ///
  /// When symmetricMoves is true, moving a handle will move the other handle
  /// symmetrically, otherwise the handles are independent. False by default
  bool symmetricMoves()const;
  void setSymmetricMoves(bool symmetry);

  /// Set/Get the value proxy of the internal range slider.
  /// \sa setValueProxy(), valueProxy()
  void setValueProxy(ValueProxy* proxy);
  ValueProxy* valueProxy() const;

signals:
  ///
  /// This signal is emitted when the slider minimum value has changed,
  /// with the new slider value as argument.
  void minimumValueChanged(double minVal);

  ///
  /// This signal is emitted when the slider maximum value has changed,
  /// with the new slider value as argument.
  void maximumValueChanged(double maxVal);

  ///
  /// Utility signal that is fired when minimum or maximum values have changed.
  void valuesChanged(double minVal, double maxVal);

  ///
  /// This signal is emitted when sliderDown is true and the slider moves.
  /// This usually happens when the user is dragging the minimum slider.
  /// The value is the new slider minimum position.
  /// This signal is emitted even when tracking is turned off.
  void minimumPositionChanged(double minPos);

  ///
  /// This signal is emitted when sliderDown is true and the slider moves.
  /// This usually happens when the user is dragging the maximum slider.
  /// The value is the new slider maximum position.
  /// This signal is emitted even when tracking is turned off.
  void maximumPositionChanged(double maxPos);

  ///
  /// Utility signal that is fired when minimum or maximum positions
  /// have changed.
  void positionsChanged(double minPos, double maxPos);

  ///
  /// This signal is emitted when the user presses one slider with the mouse,
  /// or programmatically when setSliderDown(true) is called.
  void sliderPressed();

  ///
  /// This signal is emitted when the user releases one slider with the mouse,
  /// or programmatically when setSliderDown(false) is called.
  void sliderReleased();

  ///
  /// This signal is emitted when the slider range has changed, with min being
  /// the new minimum, and max being the new maximum.
  /// Warning: don't confound with valuesChanged(double, double);
  /// \sa QAbstractSlider::rangeChanged()
  void rangeChanged(double min, double max);

public slots:
  ///
  /// This property holds the slider's current minimum value.
  /// The slider forces the minimum value to be within the legal range:
  /// minimum <= minvalue <= maxvalue <= maximum.
  /// Changing the minimumValue also changes the minimumPosition.
  void setMinimumValue(double minVal);

  ///
  /// This property holds the slider's current maximum value.
  /// The slider forces the maximum value to be within the legal range:
  /// minimum <= minvalue <= maxvalue <= maximum.
  /// Changing the maximumValue also changes the maximumPosition.
  void setMaximumValue(double maxVal);

  ///
  /// Utility function that set the minimum value and maximum value at once.
  void setValues(double minVal, double maxVal);

protected slots:
  void onValuesChanged(int min, int max);

  void onMinPosChanged(int value);
  void onMaxPosChanged(int value);
  void onPositionsChanged(int min, int max);
  void onRangeChanged(int min, int max);

  void onValueProxyAboutToBeModified();
  void onValueProxyModified();

protected:
  RangeSlider* slider()const;
  /// Subclasses can change the internal slider
  void setSlider(RangeSlider* slider);

protected:
  QScopedPointer<DoubleRangeSliderPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(DoubleRangeSlider);
  Q_DISABLE_COPY(DoubleRangeSlider);
};

#endif
