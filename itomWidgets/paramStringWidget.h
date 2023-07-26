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

#ifndef PARAMSTRINGWIDGET_H
#define PARAMSTRINGWIDGET_H

#include <QWidget>

#include "common/param.h"

#include "commonWidgets.h"

namespace ito
{

class ParamStringWidgetPrivate; // forward declare

class ITOMWIDGETS_EXPORT ParamStringWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ParamStringWidget(QWidget *parent = 0);
    virtual ~ParamStringWidget();

    ito::Param param() const;
    QByteArray value() const;
    ito::StringMeta meta() const;

Q_SIGNALS:
    void valueChanged(const QByteArray &value);

public Q_SLOTS:
    void setParam(const ito::Param &param, bool forceValueChanged = false);
    void setValue(const QByteArray &value);
    void setMeta(const ito::StringMeta &meta);

protected:
    QScopedPointer<ParamStringWidgetPrivate> d_ptr; // QScopedPointer to forward declared class

private:
    Q_DECLARE_PRIVATE(ParamStringWidget);
    Q_DISABLE_COPY(ParamStringWidget);

    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(QString))
	Q_PRIVATE_SLOT(d_func(), void slotEditingFinished())
};

} //end namespace ito

#endif // PARAMINTWIDGET_H
