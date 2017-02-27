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

#ifndef ITOMPARAMFACTORY_H
#define ITOMPARAMFACTORY_H

#include "itomParamManager.h"

#include "common/paramMeta.h"

namespace ito 
{

class ParamIntPropertyFactoryPrivate;

class QT_QTPROPERTYBROWSER_EXPORT ParamIntPropertyFactory : public QtAbstractEditorFactory<ParamIntPropertyManager>
{
    Q_OBJECT
public:
    ParamIntPropertyFactory(QObject *parent = 0);
    ~ParamIntPropertyFactory();
protected:
    void connectPropertyManager(ParamIntPropertyManager *manager);
    QWidget *createEditor(ParamIntPropertyManager *manager, QtProperty *property,
                QWidget *parent);
    void disconnectPropertyManager(ParamIntPropertyManager *manager);
private:
    ParamIntPropertyFactoryPrivate *d_ptr;
    Q_DECLARE_PRIVATE(ParamIntPropertyFactory)
    Q_DISABLE_COPY(ParamIntPropertyFactory)
    Q_PRIVATE_SLOT(d_func(), void slotPropertyChanged(QtProperty *, int))
    Q_PRIVATE_SLOT(d_func(), void slotMetaChanged(QtProperty *, const ito::IntMeta &))
    //Q_PRIVATE_SLOT(d_func(), void slotSingleStepChanged(QtProperty *, int))
    //Q_PRIVATE_SLOT(d_func(), void slotSetValue(int))
    Q_PRIVATE_SLOT(d_func(), void slotEditorDestroyed(QObject *))
};

} //end namespace ito

#endif