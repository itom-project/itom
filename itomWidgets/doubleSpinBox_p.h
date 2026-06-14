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
#include "doubleSpinBox.h"

// Qt includes
#include <QDoubleSpinBox>
#include <QPointer>
class DoubleSpinBoxPrivate;
class ValueProxy;
class QFocusEvent;

//-----------------------------------------------------------------------------
class itomQDoubleSpinBox: public QDoubleSpinBox
{
  Q_OBJECT
  /// This property controls whether decreasing the value by the mouse
  /// button or mouse wheel increases the value of the widget, and inverts the
  /// control similarly in the other way round or not. The property is switched off by
  /// default.
  /// \sa invertedControls(), setInvertedControls()
  Q_PROPERTY(bool invertedControls READ invertedControls WRITE setInvertedControls)
public:
  typedef QDoubleSpinBox Superclass;
  itomQDoubleSpinBox(DoubleSpinBoxPrivate* pimpl, QWidget* widget);
  void setInvertedControls(bool invertedControls);
  bool invertedControls() const;

  /// Overrides QDoubleSpinBox::stepBy(int) and negates the step number if the
  /// invertedControls property is true.
  virtual void stepBy(int steps);

  /// Expose lineEdit() publicly.
  /// \sa QAbstractSpinBox::lineEdit()
  virtual QLineEdit* lineEdit()const;

  virtual double valueFromText(const QString &text) const;
  virtual QString textFromValue(double value) const;
  virtual int decimalsFromText(const QString &text) const;
  virtual QValidator::State validate(QString& input, int& pos)const;

  /// Expose publicly QAbstractSpinBox::initStyleOption()
  void initStyleOptionSpinBox(QStyleOptionSpinBox* option);
protected:
  DoubleSpinBoxPrivate* const d_ptr;

  void focusOutEvent(QFocusEvent * event);

  /// If the invertedControls property is false (by default) then this function
  /// behavesLike QDoubleSpinBox::stepEnabled(). If the property is true then
  /// stepping down is allowed if the value is less then the maximum, and
  /// stepping up is allowed if the value is greater then the minimum.
  virtual StepEnabled stepEnabled () const;

  bool InvertedControls;
private:
  Q_DECLARE_PRIVATE(DoubleSpinBox);
  Q_DISABLE_COPY(itomQDoubleSpinBox);
};

//-----------------------------------------------------------------------------
class DoubleSpinBoxPrivate: public QObject
{
  Q_OBJECT
  Q_DECLARE_PUBLIC(DoubleSpinBox);
protected:
  DoubleSpinBox* const q_ptr;
public:
  DoubleSpinBoxPrivate(DoubleSpinBox& object);

  itomQDoubleSpinBox* SpinBox;
  DoubleSpinBox::SetMode Mode;
  int DefaultDecimals;
  DoubleSpinBox::DecimalsOptions DOption;
  bool InvertedControls;
  DoubleSpinBox::SizeHintPolicy SizeHintPolicy;

  double InputValue;
  double InputRange[2];

  mutable QString CachedText;
  mutable double CachedValue;
  mutable QValidator::State CachedState;
  mutable int CachedDecimals;
  mutable QSize CachedSizeHint;
  mutable QSize CachedMinimumSizeHint;
  bool ForceInputValueUpdate;

  QPointer<ValueProxy> Proxy;

  void init();
  /// Compare two double previously rounded according to the number of decimals
  bool compare(double x1, double x2) const;
  /// Return a value rounded with the number of decimals
  double round(double value, int decimals)const;

  /// Remove prefix and suffix
  QString stripped(const QString& text, int* pos)const;

  /// Return the number of decimals bounded by the allowed min and max number of
  /// decimals.
  /// If -1, returns the current number of decimals.
  int boundDecimals(int decimals)const;
  /// Return the number of decimals to use to display the value.
  /// Note that if DecimalsByValue is not set, the number of decimals to use
  /// is DefaultDecimals.
  int decimalsForValue(double value)const;
  /// Set the number of decimals of the spinbox and emit the signal
  /// No check if they are the same.
  void setDecimals(int dec);
  /// Set value with a specific number of decimals. -1 means the number of
  /// decimals stays the same.
  void setValue(double value, int dec = -1);

  /// Ensure the spinbox text is meaningful.
  /// It is called multiple times when the spinbox line edit is modified,
  /// therefore values are cached.
  double validateAndInterpret(QString &input, int &pos,
                              QValidator::State &state, int &decimals) const;

  void connectSpinBoxValueChanged();
  void disconnectSpinBoxValueChanged();

public slots:
  void editorTextChanged(const QString& text);
  void onValueChanged();

  void onValueProxyAboutToBeModified();
  void onValueProxyModified();
};
