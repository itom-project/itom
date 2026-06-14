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

#ifndef ITOMPARAMFACTORY_H
#define ITOMPARAMFACTORY_H

#include "itomParamManager.h"

#include "common/paramMeta.h"
#include "../commonWidgets.h"

namespace ito
{

class ParamIntPropertyFactoryPrivate;

class ITOMWIDGETS_EXPORT ParamIntPropertyFactory : public QtAbstractEditorFactory<ParamIntPropertyManager>
{
    Q_OBJECT
public:
    ParamIntPropertyFactory(QObject *parent = 0);
    ~ParamIntPropertyFactory();
protected:
    void connectPropertyManager(ParamIntPropertyManager *manager);
    QWidget *createEditor(ParamIntPropertyManager *manager, QtProperty *property, QWidget *parent);
    void disconnectPropertyManager(ParamIntPropertyManager *manager);
private:
    ParamIntPropertyFactoryPrivate *d_ptr;
    Q_DECLARE_PRIVATE(ParamIntPropertyFactory)
    Q_DISABLE_COPY(ParamIntPropertyFactory)
    Q_PRIVATE_SLOT(d_func(), void slotPropertyChanged(QtProperty *, int))
    Q_PRIVATE_SLOT(d_func(), void slotMetaChanged(QtProperty *, const ito::IntMeta &))
    Q_PRIVATE_SLOT(d_func(), void slotSetValue(int))
    Q_PRIVATE_SLOT(d_func(), void slotEditorDestroyed(QObject *))
};



class ParamDoublePropertyFactoryPrivate;

class ITOMWIDGETS_EXPORT ParamDoublePropertyFactory : public QtAbstractEditorFactory<ParamDoublePropertyManager>
{
    Q_OBJECT
public:
    ParamDoublePropertyFactory(QObject *parent = 0);
    ~ParamDoublePropertyFactory();
protected:
    void connectPropertyManager(ParamDoublePropertyManager *manager);
    QWidget *createEditor(ParamDoublePropertyManager *manager, QtProperty *property, QWidget *parent);
    void disconnectPropertyManager(ParamDoublePropertyManager *manager);
private:
    ParamDoublePropertyFactoryPrivate *d_ptr;
    Q_DECLARE_PRIVATE(ParamDoublePropertyFactory)
    Q_DISABLE_COPY(ParamDoublePropertyFactory)
    Q_PRIVATE_SLOT(d_func(), void slotPropertyChanged(QtProperty *, double))
    Q_PRIVATE_SLOT(d_func(), void slotMetaChanged(QtProperty *, const ito::DoubleMeta &))
    Q_PRIVATE_SLOT(d_func(), void slotSetValue(double))
    Q_PRIVATE_SLOT(d_func(), void slotEditorDestroyed(QObject *))
};


class ParamCharPropertyFactoryPrivate;

class ITOMWIDGETS_EXPORT ParamCharPropertyFactory : public QtAbstractEditorFactory<ParamCharPropertyManager>
{
    Q_OBJECT
public:
    ParamCharPropertyFactory(QObject *parent = 0);
    ~ParamCharPropertyFactory();
protected:
    void connectPropertyManager(ParamCharPropertyManager *manager);
    QWidget *createEditor(ParamCharPropertyManager *manager, QtProperty *property, QWidget *parent);
    void disconnectPropertyManager(ParamCharPropertyManager *manager);
private:
    ParamCharPropertyFactoryPrivate *d_ptr;
    Q_DECLARE_PRIVATE(ParamCharPropertyFactory)
    Q_DISABLE_COPY(ParamCharPropertyFactory)
    Q_PRIVATE_SLOT(d_func(), void slotPropertyChanged(QtProperty *, char))
    Q_PRIVATE_SLOT(d_func(), void slotMetaChanged(QtProperty *, const ito::CharMeta &))
    Q_PRIVATE_SLOT(d_func(), void slotSetValue(char))
    Q_PRIVATE_SLOT(d_func(), void slotEditorDestroyed(QObject *))
};


class ParamStringPropertyFactoryPrivate;

class ITOMWIDGETS_EXPORT ParamStringPropertyFactory : public QtAbstractEditorFactory<ParamStringPropertyManager>
{
    Q_OBJECT
public:
    ParamStringPropertyFactory(QObject *parent = 0);
    ~ParamStringPropertyFactory();
protected:
    void connectPropertyManager(ParamStringPropertyManager *manager);
    QWidget *createEditor(ParamStringPropertyManager *manager, QtProperty *property, QWidget *parent);
    void disconnectPropertyManager(ParamStringPropertyManager *manager);
private:
    ParamStringPropertyFactoryPrivate *d_ptr;
    Q_DECLARE_PRIVATE(ParamStringPropertyFactory)
    Q_DISABLE_COPY(ParamStringPropertyFactory)
    Q_PRIVATE_SLOT(d_func(), void slotPropertyChanged(QtProperty *, const QByteArray &))
    Q_PRIVATE_SLOT(d_func(), void slotMetaChanged(QtProperty *, const ito::StringMeta &))
    Q_PRIVATE_SLOT(d_func(), void slotSetValue(const QByteArray &))
    Q_PRIVATE_SLOT(d_func(), void slotEditorDestroyed(QObject *))
};

class ParamIntervalPropertyFactoryPrivate;

class ITOMWIDGETS_EXPORT ParamIntervalPropertyFactory : public QtAbstractEditorFactory<ParamIntervalPropertyManager>
{
    Q_OBJECT
public:
    ParamIntervalPropertyFactory(QObject *parent = 0);
    ~ParamIntervalPropertyFactory();
protected:
    void connectPropertyManager(ParamIntervalPropertyManager *manager);
    QWidget *createEditor(ParamIntervalPropertyManager *manager, QtProperty *property, QWidget *parent);
    void disconnectPropertyManager(ParamIntervalPropertyManager *manager);
private:
    ParamIntervalPropertyFactoryPrivate *d_ptr;
    Q_DECLARE_PRIVATE(ParamIntervalPropertyFactory)
    Q_DISABLE_COPY(ParamIntervalPropertyFactory)
    Q_PRIVATE_SLOT(d_func(), void slotPropertyChanged(QtProperty *, int, int))
    Q_PRIVATE_SLOT(d_func(), void slotMetaChanged(QtProperty *, const ito::IntervalMeta &))
    Q_PRIVATE_SLOT(d_func(), void slotSetValue(int, int))
    Q_PRIVATE_SLOT(d_func(), void slotEditorDestroyed(QObject *))
};

} //end namespace ito

#endif
