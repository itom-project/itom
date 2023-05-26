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

#ifndef WIDGETPROPPYTHONGENERAL_H
#define WIDGETPROPPYTHONGENERAL_H

#include "abstractPropertyPageWidget.h"

#include <qlistwidget.h>

#include <qwidget.h>

#include "ui_widgetPropPythonGeneral.h"

namespace ito
{

class WidgetPropPythonGeneral : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropPythonGeneral(QWidget *parent = NULL);
    ~WidgetPropPythonGeneral();

    void readSettings();
    void writeSettings();

protected:

private:
    Ui::WidgetPropPythonGeneral ui;

    //holds presets for common 3rd party textviewers...
    QMap<QString,QString> pyExtHelpers;

signals:

public slots:

private slots:
    void on_rbPyHomeSub_clicked();
    void on_rbPyHomeSys_clicked();
    void on_rbPyHomeUse_clicked();

    //3rd Party HelpViewer Buttons
    void on_cbbPyUse3rdPartyPresets_currentTextChanged(QString caption);
    void on_cbPyUse3rdPartyHelp_stateChanged(int checked);
    void on_pbApplyPyUse3rdPartyHelpViewer_clicked();

};

} //end namespace ito

#endif
