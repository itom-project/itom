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

#include "dialogPipManagerInstall.h"

#include "../global.h"

#include <qfiledialog.h>
#include <qmessagebox.h>


namespace ito {

/*static*/ DialogPipManagerInstallDefaults DialogPipManagerInstall::defaultsInstall = DialogPipManagerInstallDefaults();
/*static*/ DialogPipManagerInstallDefaults DialogPipManagerInstall::defaultsUpgrade = DialogPipManagerInstallDefaults();

//--------------------------------------------------------------------------------
DialogPipManagerInstall::DialogPipManagerInstall(QWidget *parent, QString package) :
    QDialog(parent),
    m_selectedType(typeWhl)
{
    ui.setupUi(this);

    on_radioWhl_clicked(true);

#if WIN32
    ui.checkRunSudo->setVisible(false);
#endif

    if (package == "")
    {
        setWindowTitle(tr("Install Package"));
        ui.radioWhl->setChecked(true);
        ui.checkUpgrade->setChecked(false);
        ui.txtPackage->setText("");
        m_upgradeMode = false;

        if (defaultsInstall.valid == true)
        {
            ui.checkFindLinks->setChecked(defaultsInstall.findLinks);
            ui.checkInstallDeps->setChecked(defaultsInstall.installDependencies);
            ui.checkNoIndex->setChecked(defaultsInstall.ignorePypi);
            ui.checkRunSudo->setChecked(defaultsInstall.runSudo);
            ui.checkUpgrade->setChecked(defaultsInstall.upgradeIfNewer);
            ui.txtFindLinks->setText(defaultsInstall.findLinksPath);
        }
    }
    else
    {
        setWindowTitle(tr("Update Package"));
        ui.radioSearchIndex->setChecked(true);
        ui.checkUpgrade->setChecked(true);
        ui.txtPackage->setText(package);
        m_upgradeMode = true;

        if (defaultsUpgrade.valid == true)
        {
            ui.checkFindLinks->setChecked(defaultsUpgrade.findLinks);
            ui.checkInstallDeps->setChecked(defaultsUpgrade.installDependencies);
            ui.checkNoIndex->setChecked(defaultsUpgrade.ignorePypi);
            ui.checkRunSudo->setChecked(defaultsUpgrade.runSudo);
            //ui.checkUpgrade->setChecked(defaultsUpgrade.upgradeIfNewer); //-> should always be true on update
            ui.txtFindLinks->setText(defaultsUpgrade.findLinksPath);
        }
    }

    ui.txtFindLinks->setEnabled(ui.checkFindLinks->isChecked());
}

//--------------------------------------------------------------------------------
DialogPipManagerInstall::~DialogPipManagerInstall()
{
    DialogPipManagerInstallDefaults *defaults;
    if (m_upgradeMode)
    {
        defaults = &defaultsUpgrade;
    }
    else
    {
        defaults = &defaultsInstall;
    }

    defaults->findLinks = ui.checkFindLinks->isChecked();
    defaults->installDependencies = ui.checkInstallDeps->isChecked();
    defaults->ignorePypi = ui.checkNoIndex->isChecked();
    defaults->runSudo = ui.checkRunSudo->isChecked();
    defaults->upgradeIfNewer = ui.checkUpgrade->isChecked();
    defaults->findLinksPath = ui.txtFindLinks->text();
    defaults->valid = true;
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::getResult(int &type, QString &packageName, bool &upgrade, bool &installDeps, QString &findLinks, bool &ignoreIndex, bool &runAsSudo)
{
    if (ui.radioWhl->isChecked())
    {
        type = typeWhl;
    }
    else if (ui.radioTarGz->isChecked())
    {
        type = typeTarGz;
    }
    else
    {
        type = typeSearchIndex;
    }

    packageName = ui.txtPackage->text();
    upgrade = ui.checkUpgrade->isChecked();
    installDeps = ui.checkInstallDeps->isChecked();
    findLinks = (ui.checkFindLinks->isChecked()) ? ui.txtFindLinks->text() : "";
    ignoreIndex = ui.checkNoIndex->isChecked();
    runAsSudo = ui.checkRunSudo->isChecked();
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::on_btnPackage_clicked()
{
    static QString btnPackageDirectory;
	QString filter = (m_selectedType == typeWhl) ? "Python Wheel (*.whl)" : "Python archives (*.tar.gz *.zip)";
    QString name = QFileDialog::getOpenFileName(this, tr("Select package archive"), btnPackageDirectory, filter);
    if (name != "")
    {
        btnPackageDirectory = QDir(name).absolutePath();
        ui.txtPackage->setText(name);
    }
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::on_btnFindLinks_clicked()
{
    static QString btnFindLinksDirectory;
    QString directory = QFileDialog::getExistingDirectory(this, tr("Select directory"), btnFindLinksDirectory);
    if (directory != "")
    {
        btnFindLinksDirectory = directory;
        ui.txtFindLinks->setText(directory);
    }
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::on_radioWhl_clicked(bool checked)
{
    m_selectedType = typeWhl;
    ui.btnPackage->setEnabled(true);
    ui.txtPackage->setText("");
    ui.txtPackage->setPlaceholderText(tr("choose whl archive..."));
    ui.txtPackage->setReadOnly(true);
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::on_radioTarGz_clicked(bool checked)
{
    m_selectedType = typeTarGz;
    ui.btnPackage->setEnabled(true);
    ui.txtPackage->setText("");
    ui.txtPackage->setPlaceholderText(tr("choose tar.gz or zip archive..."));
    ui.txtPackage->setReadOnly(true);
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::on_radioSearchIndex_clicked(bool checked)
{
    m_selectedType = typeSearchIndex;
    ui.btnPackage->setEnabled(false);
    ui.txtPackage->setText("");
    ui.txtPackage->setReadOnly(false);
    ui.txtPackage->setPlaceholderText(tr("package-name"));
    ui.txtPackage->setFocus();
}

//--------------------------------------------------------------------------------
void DialogPipManagerInstall::on_buttonBox_accepted()
{
    if (ui.txtPackage->text() != "")
    {
        this->accept();
    }
    else
    {
        QMessageBox::warning(this, tr("Missing package"), tr("You need to indicate a package"));
    }
}

} //end namespace ito