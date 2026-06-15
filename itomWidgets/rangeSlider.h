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

#ifndef RANGESLIDER_H
#define RANGESLIDER_H

#include <QSlider>

#include "commonWidgets.h"


class QStylePainter;
class RangeSliderPrivate;

namespace ito {
    class IntervalMeta; //forward declaration
}

/// \ingroup Widgets
///
/// A RangeSlider is a slider that lets you input 2 values instead of one
/// (see QSlider). These values are typically a lower and upper bound.
/// Values are comprised between the range of the slider. See setRange(),
/// minimum() and maximum(). The upper bound can't be smaller than the
/// lower bound and vice-versa.
/// When setting new values (setMinimumValue(), setMaximumValue() or
/// setValues()), make sure they lie between the range (minimum(), maximum())
/// of the slider, they would be forced otherwised. If it is not the behavior
/// you desire, you can set the range first (setRange(), setMinimum(),
/// setMaximum())
/// TODO: support triggerAction(QAbstractSlider::SliderSingleStepSub) that
/// moves both values at a time.
/// \sa DoubleRangeSlider, DoubleSlider, RangeWidget
class ITOMWIDGETS_EXPORT RangeSlider : public QSlider
{
  Q_OBJECT
  Q_PROPERTY(int minimumValue READ minimumValue WRITE setMinimumValue)
  Q_PROPERTY(int maximumValue READ maximumValue WRITE setMaximumValue)
  Q_PROPERTY(int minimumPosition READ minimumPosition WRITE setMinimumPosition)
  Q_PROPERTY(int maximumPosition READ maximumPosition WRITE setMaximumPosition)
  Q_PROPERTY(uint stepSizePosition READ stepSizePosition WRITE setStepSizePosition)
  Q_PROPERTY(uint minimumRange READ minimumRange WRITE setMinimumRange)
  Q_PROPERTY(uint maximumRange READ maximumRange WRITE setMaximumRange)
  Q_PROPERTY(uint stepSizeRange READ stepSizeRange WRITE setStepSizeRange)
  Q_PROPERTY(bool rangeIncludeLimits READ rangeIncludeLimits WRITE setRangeIncludeLimits)
  Q_PROPERTY(bool symmetricMoves READ symmetricMoves WRITE setSymmetricMoves)
  Q_PROPERTY(QString handleToolTip READ handleToolTip WRITE setHandleToolTip)
  Q_PROPERTY(bool useStyleSheets READ useStyleSheets WRITE setUseStyleSheets) // special property to allow a basic support for style sheets (else one handle is not displayed among others)
  Q_PROPERTY(float handleBorderRadius READ handleBorderRadius WRITE setHandleBorderRadius) // special property to indicate the border radius of the handles (only if useStyleSheets is true)

public:
  // Superclass typedef
  typedef QSlider Superclass;
  /// Constructor, builds a RangeSlider that ranges from 0 to 100 and has
  /// a lower and upper values of 0 and 100 respectively, other properties
  /// are set the QSlider default properties.
  explicit RangeSlider( Qt::Orientation o, QWidget* par= 0 );
  explicit RangeSlider( QWidget* par = 0 );
  virtual ~RangeSlider();

  ///
  /// This property holds the slider's current minimum value.
  /// The slider silently forces minimumValue to be within the legal range:
  /// minimum() <= minimumValue() <= maximumValue() <= maximum().
  /// Changing the minimumValue also changes the minimumPosition.
  int minimumValue() const;

  ///
  /// This property holds the slider's current maximum value.
  /// The slider forces the maximum value to be within the legal range:
  /// The slider silently forces maximumValue to be within the legal range:
  /// Changing the maximumValue also changes the maximumPosition.
  int maximumValue() const;

  ///
  /// This property holds the current slider minimum position.
  /// If tracking is enabled (the default), this is identical to minimumValue.
  int minimumPosition() const;
  void setMinimumPosition(int min);

  ///
  /// This property holds the current slider maximum position.
  /// If tracking is enabled (the default), this is identical to maximumValue.
  int maximumPosition() const;
  void setMaximumPosition(int max);

  ///
  /// This property holds the step size for the left or right slider position.
  /// If the stepSize is equal to 1, this property has no impact.
  uint stepSizePosition() const;
  void setStepSizePosition(uint stepSize);

  ///
  /// This property holds the minimum allowed range.
  /// The range is (1+maximumRange-minimumRange) if rangeIncludeLimits is true, else (maximumRange-minimumRange)
  uint minimumRange() const;
  void setMinimumRange(uint min);

  ///
  /// This property holds the maximum allowed range.
  /// The range is (1+maximumRange-minimumRange) if rangeIncludeLimits is true, else (maximumRange-minimumRange)
  uint maximumRange() const;
  void setMaximumRange(uint max);

  ///
  /// This property holds the step size of the allowed range.
  /// The range is (1+maximumRange-minimumRange) if rangeIncludeLimits is true, else (maximumRange-minimumRange)
  uint stepSizeRange() const;
  void setStepSizeRange(uint stepSize);

  ///
  /// This property indicates if the range is assumed to be (1+maximumRange-minimumRange) (true)
  /// or (maximumRange-minimumRange) (false). The first case is important if the rangeSlider
  /// is used for ROIs of cameras, where the first and last value are inside of the ROI.
  bool rangeIncludeLimits() const;
  void setRangeIncludeLimits(bool include);

  ///
  /// Utility function that set the minimum position and
  /// maximum position at once.
  void setPositions(int min, int max);

  ///
  /// When symmetricMoves is true, moving a handle will move the other handle
  /// symmetrically, otherwise the handles are independent. False by default
  bool symmetricMoves()const;
  void setSymmetricMoves(bool symmetry);

  ///
  /// When useStyleSheets is enabled, the groove and handles are not rendered
  /// by the native style methods, but by special methods, that obtain colors from
  /// the current palette. This option should be enabled by style sheets, since the
  /// Qt QStyleSheetStyle is not able to properly render two handles and a colored groove.
  bool useStyleSheets()const;
  void setUseStyleSheets(bool useStyleSheets);

  ///
  /// When useStyleSheets is enabled, the groove and handles are not rendered
  /// by the native style methods, but by special methods, that obtain colors from
  /// the current palette. This option should be enabled by style sheets, since the
  /// Qt QStyleSheetStyle is not able to properly render two handles and a colored groove.
  float handleBorderRadius()const;
  void setHandleBorderRadius(float radius);

  ///
  /// Controls the text to display for the handle tooltip. It is in addition
  /// to the widget tooltip.
  /// "%1" is replaced by the current value of the slider.
  /// Empty string (by default) means no tooltip.
  QString handleToolTip()const;
  void setHandleToolTip(const QString& toolTip);

  /// Returns true if the minimum value handle is down, false if it is up.
  /// \sa isMaximumSliderDown()
  bool isMinimumSliderDown()const;
  /// Returns true if the maximum value handle is down, false if it is up.
  /// \sa isMinimumSliderDown()
  bool isMaximumSliderDown()const;

  ///
  /// The range slider can be used for parameters whose meta data is of type ito::IntervalMeta
  /// or ito::RangeMeta. In this case, the limits, step sizes... of this rangeSlider can
  /// be automatically adapted to the requirements given by the corresponding parameter.
  void setLimitsFromIntervalMeta(const ito::IntervalMeta &intervalMeta);

signals:
  ///
  /// This signal is emitted when the slider minimum value has changed,
  /// with the new slider value as argument.
  void minimumValueChanged(int min);
  ///
  /// This signal is emitted when the slider maximum value has changed,
  /// with the new slider value as argument.
  void maximumValueChanged(int max);
  ///
  /// Utility signal that is fired when minimum or maximum values have changed.
  void valuesChanged(int min, int max);

  ///
  /// This signal is emitted when sliderDown is true and the slider moves.
  /// This usually happens when the user is dragging the minimum slider.
  /// The value is the new slider minimum position.
  /// This signal is emitted even when tracking is turned off.
  void minimumPositionChanged(int min);

  ///
  /// This signal is emitted when sliderDown is true and the slider moves.
  /// This usually happens when the user is dragging the maximum slider.
  /// The value is the new slider maximum position.
  /// This signal is emitted even when tracking is turned off.
  void maximumPositionChanged(int max);

  ///
  /// Utility signal that is fired when minimum or maximum positions
  /// have changed.
  void positionsChanged(int min, int max);

public slots:
  ///
  /// This property holds the slider's current minimum value.
  /// The slider silently forces min to be within the legal range:
  /// minimum() <= min <= maximumValue() <= maximum().
  /// Note: Changing the minimumValue also changes the minimumPosition.
  /// \sa stMaximumValue, setValues, setMinimum, setMaximum, setRange
  void setMinimumValue(int min);

  ///
  /// This property holds the slider's current maximum value.
  /// The slider silently forces max to be within the legal range:
  /// minimum() <= minimumValue() <= max <= maximum().
  /// Note: Changing the maximumValue also changes the maximumPosition.
  /// \sa stMinimumValue, setValues, setMinimum, setMaximum, setRange
  void setMaximumValue(int max);

  ///
  /// Utility function that set the minimum value and maximum value at once.
  /// The slider silently forces min and max to be within the legal range:
  /// minimum() <= min <= max <= maximum().
  /// Note: Changing the minimumValue and maximumValue also changes the
  /// minimumPosition and maximumPosition.
  /// \sa setMinimumValue, setMaximumValue, setMinimum, setMaximum, setRange
  void setValues(int min, int max);

protected slots:
  void onRangeChanged(int minimum, int maximum);

protected:
  RangeSlider( RangeSliderPrivate* impl, Qt::Orientation o, QWidget* par= 0 );
  RangeSlider( RangeSliderPrivate* impl, QWidget* par = 0 );

  // Description:
  // Standard Qt UI events
  virtual void mousePressEvent(QMouseEvent* ev);
  virtual void mouseMoveEvent(QMouseEvent* ev);
  virtual void mouseReleaseEvent(QMouseEvent* ev);

  virtual void keyPressEvent(QKeyEvent* ev);

  // Description:
  // Rendering is done here.
  virtual void paintEvent(QPaintEvent* ev);
  virtual void initMinimumSliderStyleOption(QStyleOptionSlider* option) const;
  virtual void initMaximumSliderStyleOption(QStyleOptionSlider* option) const;

  // Description:
  // Reimplemented for the tooltips
  virtual bool event(QEvent* event);

protected:
  QScopedPointer<RangeSliderPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(RangeSlider);
  Q_DISABLE_COPY(RangeSlider);
};

#endif
