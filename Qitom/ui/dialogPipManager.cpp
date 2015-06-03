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

#include "dialogPipManager.h"

#include <qmessagebox.h>
#include <qscrollbar.h>
#include <qdir.h>


#include "../global.h"

#include "dialogPipManagerInstall.h"


namespace ito {

//--------------------------------------------------------------------------------
DialogPipManager::DialogPipManager(QWidget *parent /*= NULL*/, bool standalone /*= false*/) :
    QDialog(parent),
    m_pPipManager(NULL),
    m_lastLogEntry(-1),
    m_outputSilent(false),
    m_standalone(standalone)
{
    ui.setupUi(this);

    m_pPipManager = new PipManager(this);

    connect(m_pPipManager, SIGNAL(pipVersion(QString)), this, SLOT(pipVersion(QString)));
    connect(m_pPipManager, SIGNAL(outputAvailable(QString,bool)), this, SLOT(outputReceived(QString,bool)));
    connect(m_pPipManager, SIGNAL(pipRequestStarted(PipManager::Task,QString,bool)), this, SLOT(pipRequestStarted(PipManager::Task,QString,bool)));
    connect(m_pPipManager, SIGNAL(pipRequestFinished(PipManager::Task,QString,bool)), this, SLOT(pipRequestFinished(PipManager::Task,QString,bool)));
    connect(ui.tablePackages, SIGNAL(selectedItemsChanged(QItemSelection,QItemSelection)), this, SLOT(treeViewSelectionChanged(QItemSelection,QItemSelection)));

    m_pPipManager->checkPipAvailable(createOptions());

    ui.tablePackages->setModel(m_pPipManager);
    ui.groupPipSettings->setCollapsed(true);

#if WIN32
    ui.btnSudoUninstall->setVisible(false);
#endif
}

//--------------------------------------------------------------------------------
DialogPipManager::~DialogPipManager()
{
    DELETE_AND_SET_NULL(m_pPipManager);
}

//--------------------------------------------------------------------------------
PipGeneralOptions DialogPipManager::createOptions() const
{
    PipGeneralOptions pgo;
    pgo.isolated = ui.checkIsolated->isChecked();
    pgo.logPath = "";
    pgo.proxy = ui.txtProxy->text();
    pgo.timeout = ui.spinTimeout->value();
    pgo.retries = ui.spinRetries->value();
    return pgo;
}

//--------------------------------------------------------------------------------
void DialogPipManager::pipVersion(const QString &version)
{
    ui.lblPipVersion->setText(version);
}

//--------------------------------------------------------------------------------
void DialogPipManager::outputReceived(const QString &text, bool success)
{
    QString text_html = text;
    text_html.replace("\n", "<br>");

    if (success)
    {
        if (!m_outputSilent)
        {
            switch (m_lastLogEntry)
            {
            case -1:
                logHtml = QString("<p style='color:#000000;'>%1").arg(text_html);
                break;
            case 0:
                logHtml += text_html;
                break;
            default:
                logHtml += QString("</p><p style='color:#000000;'>%1").arg(text_html);
                break;
            }

            m_lastLogEntry = 0;
        }
    }
    else
    {
        switch (m_lastLogEntry)
        {
        case -1:
            logHtml = QString("<p style='color:#ff0000;'>%1").arg(text_html);
            break;
        case 1:
            logHtml += text_html;
            break;
        default:
            logHtml += QString("</p><p style='color:#ff0000;'>%1").arg(text_html);
            break;
        }

        m_lastLogEntry = 1;
    }
    QString output;
    output = QString("<html><head></head><body style='font-size:8pt; font-weight:400; font-style:normal;'>%1</p></body></html>").arg(logHtml);
    ui.txtLog->setHtml(output);
    QScrollBar *sb = ui.txtLog->verticalScrollBar();
    sb->setValue(sb->maximum());
}

//--------------------------------------------------------------------------------
void DialogPipManager::pipRequestStarted(const PipManager::Task &task, const QString &text, bool outputSilent)
{
    outputReceived(text, true);

    m_outputSilent = outputSilent;

    ui.btnInstall->setEnabled(false);
    ui.btnUpdate->setEnabled(false);
    ui.btnUninstall->setEnabled(false);
    ui.btnSudoUninstall->setEnabled(false);
    ui.btnReload->setEnabled(false);
    ui.btnOk->setEnabled(false);
    ui.btnCheckForUpdates->setEnabled(false);

    m_currentTask = task;
}

//--------------------------------------------------------------------------------
void DialogPipManager::pipRequestFinished(const PipManager::Task &task, const QString &text, bool success)
{
    m_outputSilent = false;

    if (text != "")
    {
        outputReceived(text, success);
    }

    m_currentTask = PipManager::taskNo;

    ui.btnInstall->setEnabled(true);
    ui.btnUninstall->setEnabled(m_pPipManager->rowCount() > 0);
    ui.btnSudoUninstall->setEnabled(m_pPipManager->rowCount() > 0);
    ui.btnReload->setEnabled(true);
    ui.btnOk->setEnabled(true);
    ui.btnCheckForUpdates->setEnabled(true);

    if (task == PipManager::taskCheckAvailable && success)
    {
        m_pPipManager->listAvailablePackages(createOptions());
    }
}

//--------------------------------------------------------------------------------
void DialogPipManager::closeEvent(QCloseEvent *e)
{
    if (m_currentTask != PipManager::taskNo)
    {
        if (QMessageBox::question(this, tr("Abort"), tr("The pip process is still running. Do you want to interrupt it?"), QMessageBox::Yes | QMessageBox::No, QMessageBox::No) == QMessageBox::No)
        {
            e->ignore();
        }
        else
        {
            m_pPipManager->interruptPipProcess();
            e->accept();
        }
    }
}

//--------------------------------------------------------------------------------
void DialogPipManager::on_btnReload_clicked()
{
    m_pPipManager->listAvailablePackages(createOptions());
}

//--------------------------------------------------------------------------------
void DialogPipManager::on_btnCheckForUpdates_clicked()
{
    m_pPipManager->checkPackageUpdates(createOptions());

    // TODO!!!
    QModelIndex mi = ui.tablePackages->currentIndex();
    bool updatedAvailabe = m_pPipManager->data(mi, Qt::UserRole + 1).toBool();
    ui.btnUpdate->setEnabled(updatedAvailabe);
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnInstall_clicked()
{
    installOrUpdatePackage();
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnUpdate_clicked()
{
    installOrUpdatePackage()
}
//---------------------------------------------------------------------------------
void DialogPipManager::installOrUpdatePackage()
{
    DialogPipManagerInstall *dpmi = new DialogPipManagerInstall(this);
    if (dpmi->exec() == QDialog::Accepted)
    {
        PipInstall install;
        dpmi->getResult(*((int*)&install.type), install.packageName, install.upgrade, install.installDeps, install.findLinks, install.ignoreIndex, install.runAsSudo);

        if (!m_standalone && \
            ((install.type == ito::PipInstall::typeWhl && install.packageName.indexOf("numpy-", 0, Qt::CaseInsensitive) >= 0) \
            || (install.type != ito::PipInstall::typeWhl && install.packageName.compare("numpy", Qt::CaseInsensitive) == 0)))
        {
             QMessageBox msgBox(this);
             msgBox.setWindowTitle("Pip Manager");
             msgBox.setIcon(QMessageBox::Warning);
             msgBox.setText("Warning installing Numpy if itom is already running.");
             msgBox.setInformativeText(QString("If you try to install / upgrade Numpy if itom is already running, \
a file access error might occur, since itom already uses parts of Numpy. \n\n\
Click ignore if you want to try to continue the installation or click OK in order to stop the \
installation. \n\n\
In the latter case, the file 'restart_itom_with_pip_manager.txt' is created in the directory '%1', \
such that the pip manager is started one time as standalone application once you restart itom. \
Then, close all instances of itom or other software accessing Numpy, restart itom and try \
to upgrade Numpy.").arg(QDir::tempPath()));
             msgBox.setStandardButtons(QMessageBox::Ignore | QMessageBox::Ok);
             msgBox.setDefaultButton(QMessageBox::Ok);
             int ret = msgBox.exec();

             if (ret == QMessageBox::Ok)
             {
                 QDir tmp(QDir::tempPath());
                 if (!tmp.exists("restart_itom_with_pip_manager.txt"))
                 {
                     QFile file(tmp.absoluteFilePath("restart_itom_with_pip_manager.txt"));
                     if (file.open(QIODevice::ReadWrite))
                     {
                         file.close();
                     }
                 }
             }
             else
             {
                 m_pPipManager->installPackage(install, createOptions());
             }
        }
        else
        {
            m_pPipManager->installPackage(install, createOptions());
        }
    }

    DELETE_AND_SET_NULL(dpmi);
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnUninstall_clicked()
{
    QModelIndex mi = ui.tablePackages->currentIndex();
    if (mi.isValid())
    {
        QString packageName = m_pPipManager->data(m_pPipManager->index(mi.row(), 0), Qt::DisplayRole).toString();
        bool doIt = false;

        if (m_pPipManager->isPackageInUseByOther(mi))
        {
            if (QMessageBox::warning(this, tr("Uninstall package"), tr("The package '%1' is used by at least one other package. Do you really want to uninstall it?").arg(packageName), QMessageBox::Yes | QMessageBox::No, QMessageBox::No) == QMessageBox::Yes)
            {
                doIt = true;
            }
        }
        else
        {
            if (QMessageBox::information(this, tr("Uninstall package"), tr("Do you really want to uninstall the package '%1'?").arg(packageName), QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes) == QMessageBox::Yes)
            {
                doIt = true;
            }
        }

        if (doIt)
        {
            m_pPipManager->uninstallPackage(packageName, false, createOptions());
        }
    }
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnSudoUninstall_clicked()
{
    QModelIndex mi = ui.tablePackages->currentIndex();
    if (mi.isValid())
    {
        QString packageName = m_pPipManager->data(m_pPipManager->index(mi.row(), 0), Qt::DisplayRole).toString();
        bool doIt = false;

        if (m_pPipManager->isPackageInUseByOther(mi))
        {
            if (QMessageBox::warning(this, tr("Uninstall package"), tr("The package '%1' is used by at least one other package. Do you really want to uninstall it?").arg(packageName), QMessageBox::Yes | QMessageBox::No, QMessageBox::No) == QMessageBox::Yes)
            {
                doIt = true;
            }
        }
        else
        {
            if (QMessageBox::information(this, tr("Uninstall package"), tr("Do you really want to uninstall the package '%1'?").arg(packageName), QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes) == QMessageBox::Yes)
            {
                doIt = true;
            }
        }

        if (doIt)
        {
            m_pPipManager->uninstallPackage(packageName, true, createOptions());
        }
    }
}

//---------------------------------------------------------------------------------
void DialogPipManager::treeViewSelectionChanged(const QItemSelection & selected, const QItemSelection & deselected)
{
    bool updatedAvailabe = false;

    foreach (const QModelIndex &mi, selected.indexes())
    {
        if (mi.column() == 0)
        {
            updatedAvailabe = m_pPipManager->data(mi, Qt::UserRole + 1).toBool();
        }
    }
    ui.btnUpdate->setEnabled(updatedAvailabe);
}

} //end namespace ito