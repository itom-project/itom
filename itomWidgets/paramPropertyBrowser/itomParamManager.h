/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2017, Institut fuer Technische Optik (ITO),
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

protected:
    QString valueText(const QtProperty *property) const;
    void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);
    ParamRectPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(ParamRectPropertyManager)
    Q_DISABLE_COPY(ParamRectPropertyManager)
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
