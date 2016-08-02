/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "widgetPropPythonGeneral.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qsettings.h>
#include <qcoreapplication.h>
#include <qfiledialog.h>
#include <qstringlist.h>
#include <qdir.h>
#include <qfileinfo.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPythonGeneral::WidgetPropPythonGeneral(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPythonGeneral::~WidgetPropPythonGeneral()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonGeneral::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Python");

    // Save script state before execution (0: ask user, 1: always save, 2: never save)
    int index = settings.value("saveScriptStateBeforeExecution", 0).toInt();
    ui.comboSaveScriptBeforeExecution->setCurrentIndex(index);

    QString pythonHomeDirectory = settings.value("pyHome", "").toString();
    if (pythonHomeDirectory == "" || QDir(pythonHomeDirectory).exists() == false)
    {
        ui.groupPyHome->setChecked(false);
        ui.pathLineEditPyHome->setCurrentPath("");
    }
    else
    {
        ui.groupPyHome->setChecked(true);
        ui.pathLineEditPyHome->setCurrentPath(pythonHomeDirectory);
    }

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonGeneral::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    QStringList files;
    settings.beginGroup("Python");

    settings.setValue("saveScriptStateBeforeExecution", ui.comboSaveScriptBeforeExecution->currentIndex());

    if (ui.groupPyHome->isChecked())
    {
        settings.setValue("pyHome", ui.pathLineEditPyHome->currentPath());
    }
    else
    {
        settings.setValue("pyHome", "");
    }

    settings.endGroup();
}

} //end namespace ito