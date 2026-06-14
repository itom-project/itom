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

#ifndef DOUBLERANGEWIDGET_H
#define DOUBLERANGEWIDGET_H

// Qt includes
#include <QSlider>


#include "commonWidgets.h"

class DoubleRangeSlider;
class DoubleSpinBox;
class DoubleRangeWidgetPrivate;
class ValueProxy;

/// \ingroup Widgets
///
/// DoubleRangeWidget is a wrapper around a DoubleRangeSlider and 2 QSpinBoxes
/// \image html http://www.commontk.org/images/1/14/RangeWidget.png
/// \sa ctkSliderSpinBoxWidget, DoubleRangeSlider, QSpinBox
class ITOMWIDGETS_EXPORT DoubleRangeWidget : public QWidget
{
  Q_OBJECT
  Q_PROPERTY(int decimals READ decimals WRITE setDecimals)
  Q_PROPERTY(double singleStep READ singleStep WRITE setSingleStep)
  Q_PROPERTY(double minimum READ minimum WRITE setMinimum)
  Q_PROPERTY(double maximum READ maximum WRITE setMaximum)
  Q_PROPERTY(double minimumValue READ minimumValue WRITE setMinimumValue)
  Q_PROPERTY(double maximumValue READ maximumValue WRITE setMaximumValue)
  Q_PROPERTY(QString prefix READ prefix WRITE setPrefix)
  Q_PROPERTY(QString suffix READ suffix WRITE setSuffix)
  Q_PROPERTY(double tickInterval READ tickInterval WRITE setTickInterval)
  Q_PROPERTY(bool autoSpinBoxWidth READ isAutoSpinBoxWidth WRITE setAutoSpinBoxWidth)
  Q_PROPERTY(Qt::Alignment spinBoxTextAlignment READ spinBoxTextAlignment WRITE setSpinBoxTextAlignment)
  Q_PROPERTY(Qt::Alignment spinBoxAlignment READ spinBoxAlignment WRITE setSpinBoxAlignment)
  Q_PROPERTY(bool tracking READ hasTracking WRITE setTracking)
  Q_PROPERTY(bool symmetricMoves READ symmetricMoves WRITE setSymmetricMoves)

public:
  /// Superclass typedef
  typedef QWidget Superclass;

  /// Constructor
  /// If \li parent is null, DoubleRangeWidget will be a top-leve widget
  /// \note The \li parent can be set later using QWidget::setParent()
  explicit DoubleRangeWidget(QWidget* parent = 0);

  /// Destructor
  virtual ~DoubleRangeWidget();

  ///
  /// This property holds the sliders and spinbox minimum value.
  /// FIXME: Test following specs.
  /// When setting this property, the maximum is adjusted if necessary
  /// to ensure that the range remains valid.
  /// Also the slider's current value is adjusted to be within the new range.
  virtual double minimum()const;
  virtual void setMinimum(double minimum);

  ///
  /// This property holds the sliders and spinbox minimum value.
  /// FIXME: Test following specs.
  /// When setting this property, the maximum is adjusted if necessary
  /// to ensure that the range remains valid.
  /// Also the slider's current value is adjusted to be within the new range.
  virtual double maximum()const;
  virtual void setMaximum(double maximum);
  /// Description
  /// Utility function that set the min/max in once
  virtual void setRange(double min, double max);
  virtual void range(double minimumAndMaximum[2])const;

  ///
  /// This property holds the slider and spinbox minimum value.
  /// RangeWidget forces the value to be within the
  /// legal range: minimum <= minimumValue <= maximumValue <= maximum.
  virtual double minimumValue()const;

  ///
  /// This property holds the slider and spinbox maximum value.
  /// RangeWidget forces the value to be within the
  /// legal range: minimum <= minimumValue <= maximumValue <= maximum.
  virtual double maximumValue()const;

  ///
  /// Utility function that returns both values at the same time
  /// Returns minimumValue and maximumValue
  virtual void values(double &minValue, double &maxValue)const;

  ///
  /// This property holds the single step.
  /// The smaller of two natural steps that the
  /// slider provides and typically corresponds to the
  /// user pressing an arrow key.
  virtual double singleStep()const;
  virtual void setSingleStep(double step);

  ///
  /// This property holds the precision of the spin box, in decimals.
  virtual int decimals()const;

  ///
  /// This property holds the spin box's prefix.
  /// The prefix is prepended to the start of the displayed value.
  /// Typical use is to display a unit of measurement or a currency symbol
  virtual QString prefix()const;
 virtual  void setPrefix(const QString& prefix);

  ///
  /// This property holds the spin box's suffix.
  /// The suffix is appended to the end of the displayed value.
  /// Typical use is to display a unit of measurement or a currency symbol
  virtual QString suffix()const;
  virtual void setSuffix(const QString& suffix);

  ///
  /// This property holds the interval between tickmarks.
  /// This is a value interval, not a pixel interval.
  /// If it is 0, the slider will choose between lineStep() and pageStep().
  /// The default value is 0.
  virtual double tickInterval()const;
  virtual void setTickInterval(double ti);

  ///
  /// This property holds the alignment of the spin boxes.
  /// Possible Values are Qt::AlignTop, Qt::AlignBottom, and Qt::AlignVCenter.
  /// By default, the alignment is Qt::AlignVCenter
  virtual void setSpinBoxAlignment(Qt::Alignment alignment);
  virtual Qt::Alignment spinBoxAlignment()const;

  ///
  /// This property holds the alignment of the text inside the spin boxes.
  /// Possible Values are Qt::AlignLeft, Qt::AlignRight, and Qt::AlignHCenter.
  /// By default, the alignment is Qt::AlignLeft
  virtual void setSpinBoxTextAlignment(Qt::Alignment alignment);
  virtual Qt::Alignment spinBoxTextAlignment()const;

  ///
  /// This property holds whether slider tracking is enabled.
  /// If tracking is enabled (the default), the widget emits the valueChanged()
  /// signal while the slider or spinbox is being dragged. If tracking is
  /// disabled, the widget emits the valueChanged() signal only when the user
  /// releases the slider or spinbox.
  virtual void setTracking(bool enable);
  virtual bool hasTracking()const;

  ///
  /// Set/Get the auto spinbox width
  /// When the autoSpinBoxWidth property is on, the width of the SpinBox is
  /// set to the same width of the largest QSpinBox of its
  // RangeWidget siblings.
  virtual bool isAutoSpinBoxWidth()const;
  virtual void setAutoSpinBoxWidth(bool autoWidth);

  ///
  /// When symmetricMoves is true, moving a handle will move the other handle
  /// symmetrically, otherwise the handles are independent. False by default
  virtual bool symmetricMoves()const;
  virtual void setSymmetricMoves(bool symmetry);

  /// Return the slider of the range widget.
  /// \sa minimumSpinBox(), maximumSpinBox()
  virtual DoubleRangeSlider* slider()const;
  /// Return the minimum spinbox.
  /// \sa maximumSpinBox(), slider()
  virtual DoubleSpinBox* minimumSpinBox()const;
  /// Return the maximum spinbox.
  /// \sa minimumSpinBox(), slider()
  virtual DoubleSpinBox* maximumSpinBox()const;

  /// Set/Get the value proxy of the slider and spinboxes.
  /// \sa setValueProxy(), valueProxy()
  virtual void setValueProxy(ValueProxy* proxy);
  virtual ValueProxy* valueProxy() const;

public Q_SLOTS:
  ///
  /// Reset the slider and spinbox to zero (value and position)
  virtual void reset();
  virtual void setMinimumValue(double value);
  virtual void setMaximumValue(double value);
  ///
  /// Utility function that set the min and max values at once
  virtual void setValues(double minValue, double maxValue);

  /// Sets how many decimals the spinbox will use for displaying and
  /// interpreting doubles.
  virtual void setDecimals(int decimals);

Q_SIGNALS:
  /// Use with care:
  /// sliderMoved is emitted only when the user moves the slider
  //void sliderMoved(double position);
  void minimumValueChanged(double value);
  void minimumValueIsChanging(double value);
  void maximumValueChanged(double value);
  void maximumValueIsChanging(double value);
  void valuesChanged(double minValue, double maxValue);
  void rangeChanged(double min, double max);

protected Q_SLOTS:
  virtual void startChanging();
  virtual void stopChanging();
  virtual void changeValues(double newMinValue, double newMaxValue);
  virtual void changeMinimumValue(double value);
  virtual void changeMaximumValue(double value);
  /// A spinbox value has been modified, update the slider.
  virtual void setSliderValues();
  virtual void setMinimumToMaximumSpinBox(double minimum);
  virtual void setMaximumToMinimumSpinBox(double maximum);
  virtual void onSliderRangeChanged(double min, double max);

  virtual void onValueProxyAboutToBeModified();
  virtual void onValueProxyModified();

protected:
  virtual bool eventFilter(QObject *obj, QEvent *event);

  /// can be used to change the slider by a custom one
  virtual void setSlider(DoubleRangeSlider* slider);

protected:
  QScopedPointer<DoubleRangeWidgetPrivate> d_ptr;

private:
  Q_DECLARE_PRIVATE(DoubleRangeWidget);
  Q_DISABLE_COPY(DoubleRangeWidget);

};

#endif
