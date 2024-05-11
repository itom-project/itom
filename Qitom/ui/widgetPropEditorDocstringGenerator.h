/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#pragma once

#include "abstractPropertyPageWidget.h"

#include <qwidget.h>

#include "ui_widgetPropEditorDocstringGenerator.h"

namespace ito
{

class WidgetPropEditorDocstringGenerator : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropEditorDocstringGenerator(QWidget *parent = NULL);
    ~WidgetPropEditorDocstringGenerator();

    void readSettings();
    void writeSettings();

protected:

private:
    Ui::WidgetPropEditorDocstringGenerator ui;


signals:

public slots:

private slots:

};

} //end namespace ito
