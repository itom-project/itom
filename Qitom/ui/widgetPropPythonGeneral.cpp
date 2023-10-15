/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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
#include <qprocess.h>

namespace ito
{
//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPythonGeneral::WidgetPropPythonGeneral(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    pyExtHelpers["notepad"] = "notepad";
    pyExtHelpers["notepad++"] = "notepad++ -multiInst";

#ifdef WIN32
    ui.rbPyHomeSub->setVisible(true);
#else
    ui.rbPyHomeSub->setVisible(false);
#endif

    //populate combobox
    QMap<QString, QString>::const_iterator it;
    for(it = pyExtHelpers.constBegin(); it != pyExtHelpers.constEnd(); it++)
    {
        ui.cbbPyUse3rdPartyPresets->addItem(it.key());
    }
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

    //initialize GUI to show current pythonHelpViewer's parameters
    bool python3rdPartyHelperUse = settings.value("python3rdPartyHelperUse",0).toBool();
    QString python3rdPartyHelperCommand = settings.value("python3rdPartyHelperCommand", "notepad").toString();
    ui.cbPyUse3rdPartyHelp->setChecked(python3rdPartyHelperUse);
    on_cbPyUse3rdPartyHelp_stateChanged(python3rdPartyHelperUse);
    ui.lePyUse3rdPartyCommand->setText(python3rdPartyHelperCommand);
    QMap<QString, QString>::const_iterator it;

    for (it = pyExtHelpers.constBegin(); it != pyExtHelpers.constEnd(); ++it)
    {
        if (it.value() == python3rdPartyHelperCommand)
        {
            ui.cbbPyUse3rdPartyPresets->setCurrentText(it.key());
            on_cbbPyUse3rdPartyPresets_currentTextChanged(it.key());
        }
    }

    bool closeItomWithPySysExit = settings.value("closeItomWithPySysExit", false).toBool();
    ui.checkCloseItomByPySysExit->setChecked(closeItomWithPySysExit);

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

    settings.setValue("python3rdPartyHelperUse",ui.cbPyUse3rdPartyHelp->isChecked());
    settings.setValue("python3rdPartyHelperCommand", ui.lePyUse3rdPartyCommand->text());

    bool closeItomWithPySysExit = ui.checkCloseItomByPySysExit->isChecked();
    settings.setValue("closeItomWithPySysExit", closeItomWithPySysExit);

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


//----------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------- 3rdPartyHelp Section ---------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------


void WidgetPropPythonGeneral::on_cbPyUse3rdPartyHelp_stateChanged(int checked)
{
    //disable helpViewer by deleting the contents of the LineEdit
    ui.lePyUse3rdPartyCommand->setEnabled(checked);
    ui.cbbPyUse3rdPartyPresets->setEnabled(checked);
}

void WidgetPropPythonGeneral::on_cbbPyUse3rdPartyPresets_currentTextChanged(QString caption)
{
    ui.lePyUse3rdPartyCommand->setText(pyExtHelpers[caption]);
}

//this is a workaround for the external cmd to be opened... and i don't want this to happen everytime
//the properties dialogue is accepted...
void WidgetPropPythonGeneral::on_pbApplyPyUse3rdPartyHelpViewer_clicked()
{
    if (ui.cbPyUse3rdPartyHelp->isChecked())
    {
        QString tt = ui.lePyUse3rdPartyCommand->text();
        qputenv("PAGER", tt.toLatin1());
#if WIN32
        //better change this to something more appropriate.
        QString msg = QString("setx PAGER \"%1\"").arg(tt);
        system(msg.toLatin1());
#endif
    }

}
} //end namespace ito
