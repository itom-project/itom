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
    virtual void setValue(QtProperty *property, int value) = 0;

protected:
    virtual QString valueText(const QtProperty *property) const;
    virtual QIcon valueIcon(const QtProperty *property) const;
    virtual void initializeProperty(QtProperty *property);
    virtual void uninitializeProperty(QtProperty *property);

    AbstractParamPropertyManagerPrivate *d_ptr;

private:
    Q_DECLARE_PRIVATE(AbstractParamPropertyManager)
    Q_DISABLE_COPY(AbstractParamPropertyManager)
};



class ITOMWIDGETS_EXPORT ParamIntPropertyManager : public AbstractParamPropertyManager
{
    Q_OBJECT
public:
    ParamIntPropertyManager(QObject *parent = 0);
    ~ParamIntPropertyManager();

protected:
    QString valueText(const QtProperty *property) const;
    QIcon valueIcon(const QtProperty *property) const;

Q_SIGNALS:
    void valueChanged(QtProperty *property, int val);
    void metaChanged(QtProperty *property, ito::IntMeta meta);

public Q_SLOTS:
    void setParam(QtProperty *property, const ito::Param &param);
    void setValue(QtProperty *property, int value);

private:
    Q_DISABLE_COPY(ParamIntPropertyManager)
};

} //end namespace ito

#endif
