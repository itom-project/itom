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

#ifndef PARAMCHARWIDGET_H
#define PARAMCHARWIDGET_H

#include <QWidget>

#include "common/param.h"

#include "commonWidgets.h"

namespace ito
{

class ParamCharWidgetPrivate; // forward declare

class ITOMWIDGETS_EXPORT ParamCharWidget : public QWidget
{
    Q_OBJECT

    Q_PROPERTY(bool keyboardTracking READ keyboardTracking WRITE setKeyboardTracking);

public:
    explicit ParamCharWidget(QWidget *parent = 0);
    virtual ~ParamCharWidget();

    ito::Param param() const;
    bool keyboardTracking() const;
    char value() const;
    ito::CharMeta meta() const;

Q_SIGNALS:
    void valueChanged(char value);

public Q_SLOTS:
    void setParam(const ito::Param &param, bool forceValueChanged = false);
    void setKeyboardTracking(bool tracking);
    void setValue(char value);
    void setMeta(const ito::CharMeta &meta);

protected:
    QScopedPointer<ParamCharWidgetPrivate> d_ptr; // QScopedPointer to forward declared class

private:
    Q_DECLARE_PRIVATE(ParamCharWidget);
    Q_DISABLE_COPY(ParamCharWidget);

    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(int))
    Q_PRIVATE_SLOT(d_func(), void slotValueChanged(double))
    Q_PRIVATE_SLOT(d_func(), void slotChecked(bool))
};

} //end namespace ito

#endif // ParamCharWidget_H
