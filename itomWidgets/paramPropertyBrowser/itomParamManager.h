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
*********************************************************************** */


#ifndef ITOMPARAMMANAGER_H
#define ITOMPARAMMANAGER_H

#include "qtpropertybrowser.h"
#include "common/param.h"
#include "common/paramMeta.h"

#include <qicon.h>

namespace ito
{

class AbstractParamPropertyManagerPrivate;

/*
Abstract base class for all property managers that are responsible for ito::Param values.
*/
class ITOMWIDGETS_EXPORT AbstractParamPropertyManager : public QtAbstractPropertyManager
{
    Q_OBJECT
public:
    AbstractParamPropertyManager(QObject *parent = 0);
    ~AbstractParamPropertyManager();

    const ito::ParamBase &paramBase(const QtProperty *property) const;
    const ito::Param &param(const QtProperty *property) const;

public Q_SLOTS:
    virtual void setParam(QtProperty *property, const ito::Param &param) = 0;

protected:
    virtual QString valueText(const QtProperty *property) const;
    virtual QIcon valueIcon(const QtProperty *property) const;
    virtual void initializeProperty(QtProperty *property) = 0;
    virtual void uninitializeProperty(QtProperty *property);

    AbstractParamPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(AbstractParamPropertyManager)
    Q_DISABLE_COPY(AbstractParamPropertyManager)
};


/*
Property Manager for parameters of type ito::ParamBase::Int
*/
class ITOMWIDGETS_EXPORT ParamIntPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamIntPropertyManager(QObject *parent = 0);
    ~ParamIntPropertyManager();

protected:
    QString valueText(const QtProperty *property) const;
    QIcon valueIcon(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);

Q_SIGNALS:
    void valueChanged(QtProperty *property, int val);
    void metaChanged(QtProperty *property, ito::IntMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int value);

private:
    Q_DISABLE_COPY(ParamIntPropertyManager)
};

/*
Property Manager for parameters of type ito::ParamBase::Char
*/
class ITOMWIDGETS_EXPORT ParamCharPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamCharPropertyManager(QObject *parent = 0);
    ~ParamCharPropertyManager();

protected:
    QString valueText(const QtProperty *property) const;
    QIcon valueIcon(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);

Q_SIGNALS:
    void valueChanged(QtProperty *property, char val);
    void metaChanged(QtProperty *property, ito::CharMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, char value);

private:
    Q_DISABLE_COPY(ParamCharPropertyManager)
};

/*
Property Manager for parameters of type ito::ParamBase::Double
*/
class ITOMWIDGETS_EXPORT ParamDoublePropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamDoublePropertyManager(QObject *parent = 0);
    ~ParamDoublePropertyManager();

    bool hasPopupSlider() const;
    void setPopupSlider(bool popup);

protected:
    QString valueText(const QtProperty *property) const;
    QIcon valueIcon(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);

Q_SIGNALS:
    void valueChanged(QtProperty *property, double val);
    void metaChanged(QtProperty *property, ito::DoubleMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, double value);

private:
    Q_DISABLE_COPY(ParamDoublePropertyManager)

    bool m_popupSlider;
};


/*
Property Manager for parameters of type ito::ParamBase::String
*/
class ITOMWIDGETS_EXPORT ParamStringPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamStringPropertyManager(QObject *parent = 0);
    ~ParamStringPropertyManager();

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);

Q_SIGNALS:
    void valueChanged(QtProperty *property, const QByteArray &value);
    void metaChanged(QtProperty *property, ito::StringMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, const QByteArray &value);

private:
    Q_DISABLE_COPY(ParamStringPropertyManager)
};

/*
Property Manager for parameters of type ito::ParamBase::Interval and ito::ParamBase::Range
*/
class ITOMWIDGETS_EXPORT ParamIntervalPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamIntervalPropertyManager(QObject *parent = 0);
    ~ParamIntervalPropertyManager();

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);

Q_SIGNALS:
    void valueChanged(QtProperty *property, int min, int max);
    void metaChanged(QtProperty *property, ito::IntervalMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int min, int max);

private:
    Q_DISABLE_COPY(ParamIntervalPropertyManager)
};


class ParamRectPropertyManagerPrivate;

/*
Property Manager for parameters of type ito::ParamBase::Rect
*/
class ITOMWIDGETS_EXPORT ParamRectPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamRectPropertyManager(QObject *parent = 0);
    ~ParamRectPropertyManager();

Q_SIGNALS:
    void valueChanged(QtProperty *property, int left, int top, int width, int height);
    void metaChanged(QtProperty *property, ito::RectMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int left, int top, int width, int height);

    ParamIntervalPropertyManager *subIntervalPropertyManager() const;

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);
    ParamRectPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(ParamRectPropertyManager)
    Q_DISABLE_COPY(ParamRectPropertyManager)

    Q_PRIVATE_SLOT(d_func(), void slotIntervalChanged(QtProperty *, int, int))
    Q_PRIVATE_SLOT(d_func(), void slotPropertyDestroyed(QtProperty *))
};


class ParamCharArrayPropertyManagerPrivate;

/*
Property Manager for parameters of type ito::ParamBase::CharArray
*/
class ITOMWIDGETS_EXPORT ParamCharArrayPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamCharArrayPropertyManager(QObject *parent = 0);
    ~ParamCharArrayPropertyManager();

    typedef char DataType;

Q_SIGNALS:
    void valueChanged(QtProperty *property, int num, const char* values);
    void metaChanged(QtProperty *property, ito::CharArrayMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int num, const char* values);

    ParamCharPropertyManager *subPropertyManager() const;

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);
    ParamCharArrayPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(ParamCharArrayPropertyManager)
    Q_DISABLE_COPY(ParamCharArrayPropertyManager)

    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(QtProperty *, char))
    Q_PRIVATE_SLOT(d_func(), void slotPropertyDestroyed(QtProperty *))
};

class ParamIntArrayPropertyManagerPrivate;

/*
Property Manager for parameters of type ito::ParamBase::IntArray
*/
class ITOMWIDGETS_EXPORT ParamIntArrayPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamIntArrayPropertyManager(QObject *parent = 0);
    ~ParamIntArrayPropertyManager();

    typedef int DataType;

Q_SIGNALS:
    void valueChanged(QtProperty *property, int num, const int* values);
    void metaChanged(QtProperty *property, ito::IntArrayMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int num, const int* values);

    ParamIntPropertyManager *subPropertyManager() const;

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);
    ParamIntArrayPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(ParamIntArrayPropertyManager)
    Q_DISABLE_COPY(ParamIntArrayPropertyManager)

    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(QtProperty *, int))
    Q_PRIVATE_SLOT(d_func(), void slotPropertyDestroyed(QtProperty *))
};

class ParamDoubleArrayPropertyManagerPrivate;

/*
Property Manager for parameters of type ito::ParamBase::DoubleArray
*/
class ITOMWIDGETS_EXPORT ParamDoubleArrayPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamDoubleArrayPropertyManager(QObject *parent = 0);
    ~ParamDoubleArrayPropertyManager();

    typedef double DataType;

Q_SIGNALS:
    void valueChanged(QtProperty *property, int num, const double* values);
    void metaChanged(QtProperty *property, ito::DoubleArrayMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int num, const double* values);

    ParamDoublePropertyManager *subPropertyManager() const;

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);
    ParamDoubleArrayPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(ParamDoubleArrayPropertyManager)
    Q_DISABLE_COPY(ParamDoubleArrayPropertyManager)

    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(QtProperty *, double))
    Q_PRIVATE_SLOT(d_func(), void slotPropertyDestroyed(QtProperty *))
};

class ParamStringListPropertyManagerPrivate;

/*
Property Manager for parameters of type ito::ParamBase::StringList
*/
class ITOMWIDGETS_EXPORT ParamStringListPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamStringListPropertyManager(QObject *parent = 0);
    ~ParamStringListPropertyManager();

    typedef ito::ByteArray DataType;

Q_SIGNALS:
    void valueChanged(QtProperty *property, int num, const ito::ByteArray* values);
    void metaChanged(QtProperty *property, ito::StringListMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int num, const ito::ByteArray* values);

    ParamStringPropertyManager *subPropertyManager() const;

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);
    ParamStringListPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(ParamStringListPropertyManager)
    Q_DISABLE_COPY(ParamStringListPropertyManager)

    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(QtProperty *, const QByteArray &))
    Q_PRIVATE_SLOT(d_func(), void slotPropertyDestroyed(QtProperty *))
};

/*
Property Manager for parameters of type ito::ParamBase::HWRef, ito::ParamBase::DObjPtr,
ito::ParamBase::PolygonMeshPtr, ito::ParamBase::PointCloudPtr, ito::ParamBase::PointPtr
*/
class ITOMWIDGETS_EXPORT ParamOtherPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamOtherPropertyManager(QObject *parent = 0);
    ~ParamOtherPropertyManager();

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);

private:
    Q_DISABLE_COPY(ParamOtherPropertyManager)
};



} //end namespace ito

#endif
