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

#include "valueProxy.h"

// --------------------------------------------------------------------------
// ValueProxyPrivate

class ITOMWIDGETS_EXPORT ValueProxyPrivate
{
  Q_DECLARE_PUBLIC(ValueProxy);

public:
  ValueProxy* q_ptr;
  ValueProxyPrivate(ValueProxy& object);
  ~ValueProxyPrivate();

  double Value;
  double ProxyValue;
};

// --------------------------------------------------------------------------
// ValueProxyPrivate methods

// --------------------------------------------------------------------------
ValueProxyPrivate::ValueProxyPrivate(ValueProxy& object)
  : q_ptr(&object)
{
  this->Value = 0.0;
  this->ProxyValue = 0.0;
}

// --------------------------------------------------------------------------
ValueProxyPrivate::~ValueProxyPrivate()
{
}

// --------------------------------------------------------------------------
// ValueProxy methods

// --------------------------------------------------------------------------
ValueProxy::ValueProxy(QObject* _parent) : Superclass(_parent)
  , d_ptr(new ValueProxyPrivate(*this))
{
}

// --------------------------------------------------------------------------
ValueProxy::~ValueProxy()
{
}

// --------------------------------------------------------------------------
double ValueProxy::value() const
{
  Q_D(const ValueProxy);
  return d->Value;
}

// --------------------------------------------------------------------------
void ValueProxy::setValue(double newValue)
{
  Q_D(ValueProxy);
  if (d->Value == newValue)
    {
    return;
    }

  d->Value = newValue;
  emit this->valueChanged(d->Value);
  this->updateProxyValue();
}

// --------------------------------------------------------------------------
double ValueProxy::proxyValue() const
{
  Q_D(const ValueProxy);
  return d->ProxyValue;
}

// --------------------------------------------------------------------------
void ValueProxy::setProxyValue(double newProxyValue)
{
  Q_D(ValueProxy);
  if (d->ProxyValue == newProxyValue)
    {
    return;
    }

  d->ProxyValue = newProxyValue;
  emit this->proxyValueChanged(d->ProxyValue);
  this->updateValue();
}

// --------------------------------------------------------------------------
void ValueProxy::updateProxyValue()
{
  Q_D(ValueProxy);
  double newProxyValue = this->proxyValueFromValue(d->Value);
  if (newProxyValue == d->ProxyValue)
    {
    return;
    }

  d->ProxyValue = newProxyValue;
  emit this->proxyValueChanged(d->ProxyValue);
}

// --------------------------------------------------------------------------
void ValueProxy::updateValue()
{
  Q_D(ValueProxy);
  double newValue = this->valueFromProxyValue(d->ProxyValue);
  if (newValue == d->Value)
    {
    return;
    }

  d->Value = newValue;
  emit this->valueChanged(d->Value);
}
