/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifdef WIN32
    ui.rbPyHomeSub->setVisible(true);
#else
    ui.rbPyHomeSub->setVisible(false);
#endif
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

    int pythonDirState = settings.value("pyDirState", -1).toInt();
    ui.rbPyHomeSub->setChecked(pythonDirState == 0);
    ui.rbPyHomeSys->setChecked(pythonDirState == 1);
    ui.rbPyHomeUse->setChecked(pythonDirState == 2);

    QString pythonHomeDirectory = settings.value("pyHome", "").toString();
    if (pythonHomeDirectory == "" || QDir(pythonHomeDirectory).exists() == false)
    {
        ui.pathLineEditPyHome->setCurrentPath("");
    }
    else
    {
        ui.pathLineEditPyHome->setCurrentPath(pythonHomeDirectory);
    }
    ui.pathLineEditPyHome->setEnabled(pythonDirState == 2);

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonGeneral::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    QStringList files;
    settings.beginGroup("Python");

    settings.setValue("saveScriptStateBeforeExecution", ui.comboSaveScriptBeforeExecution->currentIndex());

    int pythonDirState = -1;
    if (ui.rbPyHomeSub->isChecked())
    {
        pythonDirState = 0;
    }
    else if (ui.rbPyHomeSys->isChecked())
    {
        pythonDirState = 1;
    }
    else if (ui.rbPyHomeUse->isChecked())
    {
        pythonDirState = 2;
    }

    settings.setValue("pyDirState", pythonDirState);
    settings.setValue("pyHome", ui.pathLineEditPyHome->currentPath());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonGeneral::on_rbPyHomeSub_clicked()
{
    ui.pathLineEditPyHome->setEnabled(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonGeneral::on_rbPyHomeSys_clicked()
{
    ui.pathLineEditPyHome->setEnabled(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPythonGeneral::on_rbPyHomeUse_clicked()
{
    ui.pathLineEditPyHome->setEnabled(true);
}

} //end namespace ito