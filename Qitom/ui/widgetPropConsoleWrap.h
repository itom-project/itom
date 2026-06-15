/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef WIDGETPROPCONSOLEWRAP_H
#define WIDGETPROPCONSOLEWRAP_H

#include "abstractPropertyPageWidget.h"

#include <QtGui>

#include "ui_widgetPropConsoleWrap.h"

namespace ito
{

class WidgetPropConsoleWrap: public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropConsoleWrap(QWidget *parent = NULL);
    ~WidgetPropConsoleWrap();

    void readSettings();
    void writeSettings();

protected:

private:
    Ui::WidgetPropConsoleWrap ui;


signals:

public slots:

private slots:

};

} //end namespace ito

#endif
