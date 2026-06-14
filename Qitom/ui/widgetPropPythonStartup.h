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

#ifndef WIDGETPROPPYTHONSTARTUP_H
#define WIDGETPROPPYTHONSTARTUP_H

#include "abstractPropertyPageWidget.h"
#include <qlistwidget.h>
#include <qwidget.h>
#include "ui_widgetPropPythonStartup.h"

namespace ito
{

class WidgetPropPythonStartup : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropPythonStartup(QWidget *parent = NULL);
    ~WidgetPropPythonStartup();

    void readSettings();
    void writeSettings();

protected:
    void updateScriptButtons();

private:
    Ui::WidgetPropPythonStartup ui;

signals:

public slots:

private slots:
    void on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* previous);
    void on_btnAdd_clicked();
    void on_btnRemove_clicked();
    void on_btnDownScript_clicked();
    void on_btnUpScript_clicked();
};

} //end namespace ito

#endif
