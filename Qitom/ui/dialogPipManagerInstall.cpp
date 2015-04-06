/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut für Technische Optik (ITO),
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

#include "dialogPipManagerInstall.h"

#include "../global.h"


namespace ito {

//--------------------------------------------------------------------------------
DialogPipManagerInstall::DialogPipManagerInstall(QWidget *parent ) :
    QDialog(parent)
{
    ui.setupUi(this);
}

//--------------------------------------------------------------------------------
DialogPipManagerInstall::~DialogPipManagerInstall()
{
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::getResult(int &type, QString &packageName, bool &upgrade, bool &installDeps, QString &findLinks, bool &ignoreIndex)
{
    if (ui.radioWhl->isChecked())
    {
        type = 0;
    }
    else if (ui.radioTarGz->isChecked())
    {
        type = 1;
    }
    else
    {
        type = 2;
    }

    packageName = ui.txtPackage->text();
    upgrade = ui.checkUpgrade->isChecked();
    installDeps = ui.checkInstallDeps->isChecked();
    findLinks = (ui.checkFindLinks->isChecked()) ? ui.txtFindLinks->text() : "";
    ignoreIndex = ui.checkNoIndex->isChecked();

}

} //end namespace ito