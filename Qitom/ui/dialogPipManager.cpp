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

#include "dialogPipManager.h"

#include <qmessagebox.h>
#include <qscrollbar.h>
#include <qdir.h>
#include <qheaderview.h>
#include <qsettings.h>
#include <qmenu.h>
#include <qaction.h>
#include <qfiledialog.h>
#include <qclipboard.h>
#include <qfile.h>

#include "../global.h"
#include "../AppManagement.h"
#include "../helper/guiHelper.h"

#include "dialogPipManagerInstall.h"


namespace ito {

QString DialogPipManager::invisiblePwStr = "[invisible_password]";

//--------------------------------------------------------------------------------
DialogPipManager::DialogPipManager(QWidget *parent /*= NULL*/, bool standalone /*= false*/) :
    QDialog(parent),
    m_pPipManager(NULL),
    m_lastLogEntry(-1),
    m_outputSilent(false),
    m_standalone(standalone),
    m_colorMessage(Qt::black),
    m_colorError(Qt::red),
    m_currentTask(PipManager::taskNo)
{
    setWindowFlags(windowFlags() | Qt::WindowMaximizeButtonHint);

    ui.setupUi(this);

    ui.btnExit->setVisible(standalone);
    ui.btnStartItom->setVisible(standalone);
    ui.btnOk->setVisible(!standalone);

    ui.progressCancelFetchDetails->setVisible(false);
    ui.btnCancelFetchDetails->setVisible(false);

    ito::RetVal retval;
    m_pPipManager = new PipManager(retval, this);

    if (!retval.containsError())
    {
        connect(m_pPipManager, &PipManager::pipVersion, this, &DialogPipManager::pipVersion);
        connect(m_pPipManager, &PipManager::outputAvailable, this, &DialogPipManager::outputReceived);
        connect(m_pPipManager, SIGNAL(pipRequestStarted(PipManager::Task, QString, bool)), this, SLOT(pipRequestStarted(PipManager::Task, QString, bool)));
        connect(m_pPipManager, SIGNAL(pipRequestFinished(PipManager::Task, QString, bool)), this, SLOT(pipRequestFinished(PipManager::Task, QString, bool)));
        connect(ui.tablePackages, SIGNAL(selectedItemsChanged(QItemSelection, QItemSelection)), this, SLOT(treeViewSelectionChanged(QItemSelection, QItemSelection)));
        connect(ui.tablePackages, SIGNAL(customContextMenuRequested(QPoint)),this, SLOT(tableCustomContextMenuRequested(QPoint)));
        connect(m_pPipManager, &PipManager::pipFetchDetailsProgress, this, &DialogPipManager::pipFetchDetailsProgress);

        m_pPipManager->checkPipAvailable(createOptions());

        ui.tablePackages->setModel(m_pPipManager);
        ui.tablePackages->setWordWrap(false);
        ui.tablePackages->setShowGrid(false);
        ui.tablePackages->horizontalHeader()->setStretchLastSection(true);
        ui.tablePackages->horizontalHeader()->setHighlightSections(false);
        ui.tablePackages->verticalHeader()->setVisible(false);
        ui.tablePackages->verticalHeader()->setDefaultSectionSize(ui.tablePackages->verticalHeader()->minimumSectionSize());
        ui.tablePackages->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui.tablePackages->setSelectionMode(QAbstractItemView::SingleSelection);
        ui.tablePackages->setColumnWidth(1, 50 * GuiHelper::screenDpiFactor());
        ui.tablePackages->setColumnWidth(2, 200 * GuiHelper::screenDpiFactor());
        ui.tablePackages->setContextMenuPolicy(Qt::CustomContextMenu);

        ui.groupPipSettings->setCollapsed(true);

#if WIN32
        ui.btnSudoUninstall->setVisible(false);
#endif
    }
    else
    {
        ui.tablePackages->setEnabled(false);
        ui.btnInstall->setEnabled(false);
        ui.btnUpdate->setEnabled(false);
        ui.btnUninstall->setEnabled(false);
        ui.btnSudoUninstall->setEnabled(false);
        ui.btnReload->setEnabled(false);
        ui.btnCheckForUpdates->setEnabled(false);
        ui.btnOk->setEnabled(false);
        ui.btnCheckForUpdates->setEnabled(false);

        QMessageBox::critical(this, tr("Python initialization error"), retval.errorMessage());
    }


    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("pipManager");

    QString proxy = settings.value("proxy", "").toString();

    if (proxy.contains(invisiblePwStr))
    {
        outputReceived(tr("A proxy server was restored from the settings file. This proxy contained a password, which was not saved in the settings. Please set it again in the pip settings (below)."), false);
    }

    ui.txtProxy->setText(proxy);

    ui.spinTimeout->setValue(settings.value("timeout", 15).toInt());
    ui.spinRetries->setValue(settings.value("retries", 5).toInt());
    ui.checkTrustedHosts->setChecked(settings.value("trustedHostsEnabled", false).toBool());
    ui.txtTrustedHosts->setText(settings.value("trustedHosts", "pypi.python.org; files.pythonhosted.org; pypi.org").toString());
    ui.txtTrustedHosts->setEnabled(ui.checkTrustedHosts->isChecked());
    ui.checkIsolated->setChecked(settings.value("isolated", false).toBool());

    settings.endGroup();
}

//--------------------------------------------------------------------------------
DialogPipManager::~DialogPipManager()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("pipManager");

    settings.setValue("timeout", ui.spinTimeout->value());
    settings.setValue("retries", ui.spinRetries->value());
    settings.setValue("trustedHostsEnabled", ui.checkTrustedHosts->isChecked());
    settings.setValue("trustedHosts", ui.txtTrustedHosts->text());
    settings.setValue("isolated", ui.checkIsolated->isChecked());

    QString proxy = ui.txtProxy->text();
    int atidx = proxy.indexOf("@");
    if (atidx > 1)
    {
        QString left = proxy.left(atidx);
        int colonidx = left.indexOf(":");
        if (colonidx >= 0)
        {
            left = left.left(colonidx + 1) + invisiblePwStr;
        }
        proxy = left + proxy.mid(atidx);
    }

    settings.setValue("proxy", proxy);

    settings.setValue("isolated", ui.checkIsolated->isChecked());

    settings.endGroup();

    DELETE_AND_SET_NULL(m_pPipManager);
}

//--------------------------------------------------------------------------------
void DialogPipManager::setColorMessage(const QColor &color)
{
    m_colorMessage = color;

    outputReceived("", true);
}

//--------------------------------------------------------------------------------
void DialogPipManager::setColorError(const QColor &color)
{
    m_colorError = color;

    outputReceived("", true);
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
    pgo.useTrustedHosts = ui.checkTrustedHosts->isChecked();
    QStringList trustedHosts = ui.txtTrustedHosts->text().split(";");
    pgo.trustedHosts.clear();
    QString temp;
    foreach(const QString &th, trustedHosts)
    {
        temp = th.trimmed();
        if (temp != "")
        {
            if (temp.contains(" "))
            {
                pgo.trustedHosts.append(QString("\"%1\"").arg(temp));
            }
            else
            {
                pgo.trustedHosts.append(temp);
            }
        }
    }
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
    if (text != "")
    {
        QString text_html = text.toHtmlEscaped();
        text_html.replace("\n", "<br>");

        if (success)
        {
            if (!m_outputSilent)
            {
                switch (m_lastLogEntry)
                {
                case -1:
                    m_logHtml = QString("<p class='message'>%1").arg(text_html);
                    break;
                case 0:
                    m_logHtml += text_html;
                    break;
                default:
                    m_logHtml += QString("</p><p class='message'>%1").arg(text_html);
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
                m_logHtml = QString("<p class='error'>%1").arg(text_html);
                break;
            case 1:
                m_logHtml += text_html;
                break;
            default:
                m_logHtml += QString("</p><p class='error'>%1").arg(text_html);
                break;
            }

            m_lastLogEntry = 1;
        }
    }

    float factor = GuiHelper::screenDpiFactor();
    QString output;
    output = QString("<html><head><style>body{ font-size:%4pt; font-weight:400; } \
    p.message{ color:%2; } \
    p.error{ color:%3; }</style> \
    <body>%1</p></body></html>"). \
        arg(m_logHtml).arg(m_colorMessage.name()).arg(m_colorError.name()).arg(8 * factor);
    ui.txtLog->setHtml(output);
    QScrollBar *sb = ui.txtLog->verticalScrollBar();
    sb->setValue(sb->maximum());
}

//--------------------------------------------------------------------------------
void DialogPipManager::pipRequestStarted(const PipManager::Task &task, const QString &text, bool outputSilent)
{
    m_outputSilent = false;

    outputReceived(text, true);

    m_outputSilent = outputSilent;

    ui.btnInstall->setEnabled(false);
    ui.btnUpdate->setEnabled(false);
    ui.btnUninstall->setEnabled(false);
    ui.btnSudoUninstall->setEnabled(false);
    ui.btnReload->setEnabled(false);
    ui.btnCheckForUpdates->setEnabled(false);
    ui.btnOk->setEnabled(false);
    ui.btnVerifyInstalledPackages->setEnabled(false);

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
    ui.btnCheckForUpdates->setEnabled(true);
    ui.btnOk->setEnabled(true);
    ui.btnVerifyInstalledPackages->setEnabled(true);

    if (task == PipManager::taskCheckAvailable && success)
    {
        m_pPipManager->listAvailablePackages(createOptions());
    }
    else if (task == PipManager::taskCheckUpdates && success)
    {
        QModelIndex mi = ui.tablePackages->currentIndex();
        QItemSelection ItemSelection(mi, mi);
        treeViewSelectionChanged(ItemSelection, ItemSelection);
    }
}

//-------------------------------------------------------------------------------------
void DialogPipManager::pipFetchDetailsProgress(int totalNumberOfUnfetchedDetails, int recentlyFetchedDetails, bool finished)
{
    ui.progressCancelFetchDetails->setVisible(!finished);
    ui.btnCancelFetchDetails->setVisible(!finished);
    ui.progressCancelFetchDetails->setMaximum(totalNumberOfUnfetchedDetails);
    ui.progressCancelFetchDetails->setValue(recentlyFetchedDetails);
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
    m_pPipManager->listAvailablePackages(createOptions(), true);
}

//--------------------------------------------------------------------------------
void DialogPipManager::on_btnVerifyInstalledPackages_clicked()
{
    m_pPipManager->checkVerifyInstalledPackages(createOptions());
}

//-------------------------------------------------------------------------------------
void DialogPipManager::on_btnCancelFetchDetails_clicked()
{
    m_pPipManager->interruptPipProcess();
    ui.progressCancelFetchDetails->setVisible(false);
    ui.btnCancelFetchDetails->setVisible(false);
}

//--------------------------------------------------------------------------------
void DialogPipManager::on_btnCheckForUpdates_clicked()
{
    m_pPipManager->checkPackageUpdates(createOptions());
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnInstall_clicked()
{
    installOrUpdatePackage(false);
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnUpdate_clicked()
{
    installOrUpdatePackage(true);
}
//---------------------------------------------------------------------------------
void DialogPipManager::installOrUpdatePackage(bool update /*=false*/)
{
    const QModelIndex &mi = ui.tablePackages->currentIndex();

    QString package = ""; //pre-defined package

    if (update)
    {
        /*
        //if an update is available of the currently selected item, use this
        if (m_pPipManager->data(mi, Qt::UserRole + 1).toBool())
        {
            QModelIndex miCol0 = m_pPipManager->index(mi.row(), 0);
            package = m_pPipManager->data(miCol0, 0).toString();
        }*/

        //always pre-set the currently set package
        if (mi.isValid())
        {
            QModelIndex miCol0 = m_pPipManager->index(mi.row(), 0);
            package = m_pPipManager->data(miCol0, 0).toString();
        }
    }

    DialogPipManagerInstall *dpmi = new DialogPipManagerInstall(this, package);
    if (dpmi->exec() == QDialog::Accepted)
    {
        PipInstall install;
        dpmi->getResult(*((int*)&install.type), install.packageName, install.upgrade, install.installDeps, install.findLinks, install.ignoreIndex, install.runAsSudo);

        if (!m_standalone && \
            ((install.type == ito::PipInstall::typeWhl && install.packageName.indexOf("numpy-", 0, Qt::CaseInsensitive) >= 0) \
            || (install.type != ito::PipInstall::typeWhl && install.packageName.compare("numpy", Qt::CaseInsensitive) == 0)))
        {
             QMessageBox msgBox(this);
             msgBox.setWindowTitle(tr("Pip Manager"));
             msgBox.setIcon(QMessageBox::Warning);
             msgBox.setText(tr("Warning installing Numpy if itom is already running."));
             msgBox.setInformativeText(tr("If you try to install / upgrade Numpy if itom is already running, \
a file access error might occur, since itom already uses parts of Numpy. \n\n\
You have now three possibilities: \n\
1. Try to continue the installation in spite of possible problems by clicking 'Ignore' \n\
2. Click 'OK', close and restart itom. Then this package manager is opened as standalone application \
and you can install or upgrade Numpy and other packages. After another restart, itom is restarted as usual. \n\
3. Click 'Cancel' to cancel the installation process without any changes. \n\
\n\
Information: \n\
If the case of the restart ('OK'), an empty file 'restart_itom_with_pip_manager.txt' is created in the directory '%1'. \
If itom locates this file at startup, the pip manager is directly started. \n\
\n\
It is also possible to directly start the package manager by calling the itom application with the argument 'pipManager'.").arg(QDir::tempPath()));
             msgBox.setStandardButtons(QMessageBox::Ignore | QMessageBox::Ok | QMessageBox::Cancel);
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
             else if (ret == QMessageBox::Ignore)
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
    bool updateAvailable = false;

    if (selected.size() > 0)
    {
        updateAvailable = true;
    }

    ui.btnUpdate->setEnabled(updateAvailable && ui.btnInstall->isEnabled());
    ui.btnUninstall->setEnabled(updateAvailable && ui.btnInstall->isEnabled());
    ui.btnSudoUninstall->setEnabled(updateAvailable && ui.btnInstall->isEnabled());
}

//---------------------------------------------------------------------------------
void DialogPipManager::tableCustomContextMenuRequested(const QPoint &pos)
{
    QModelIndex index = ui.tablePackages->indexAt(pos);

    QMenu *menu = new QMenu(this);
    QAction *copyToClipboard = menu->addAction(QIcon(":/files/icons/clipboard.png"), tr("Export table to clipboard"));
    connect(copyToClipboard, SIGNAL(triggered()), this, SLOT(exportTableToClipboard()));
    QAction *saveToCsv = menu->addAction(QIcon(":/files/icons/fileSave.png"), tr("Export table to csv-file..."));
    connect(saveToCsv, SIGNAL(triggered()), this, SLOT(exportTableToCsv()));
    menu->popup(ui.tablePackages->viewport()->mapToGlobal(pos));
}

//---------------------------------------------------------------------------------
void DialogPipManager::exportTableToClipboard()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(exportPackageTableToString(), QClipboard::Clipboard);
}

//---------------------------------------------------------------------------------
void DialogPipManager::exportTableToCsv()
{
    QString filters(tr("CSV files (*.csv);;All files (*.*)"));
    QString defaultFilter(tr("CSV files (*.csv)"));
    QString fileName = QFileDialog::getSaveFileName(0, tr("Export to file"), QCoreApplication::applicationDirPath(),
        filters, &defaultFilter);

    if (fileName != "")
    {
        QFile file(fileName);

        if (file.open(QFile::WriteOnly | QFile::Truncate))
        {
            QTextStream data(&file);

            data << exportPackageTableToString();

            file.close();
        }
        else
        {
            QMessageBox::warning(this, tr("Export to file"), tr("The file '%1' could not be opened").arg(fileName));
        }
    }
}

//---------------------------------------------------------------------------------
QString DialogPipManager::exportPackageTableToString() const
{
    QStringList output;
    QStringList strList;
    for (int i = 0; i < m_pPipManager->columnCount(); i++)
    {
        if (m_pPipManager->headerData(i, Qt::Horizontal, Qt::DisplayRole).toString().length() > 0)
        {
            strList.append("\"" + m_pPipManager->headerData(i, Qt::Horizontal, Qt::DisplayRole).toString() + "\"");
        }
        else
        {
            strList.append("");
        }
    }


    output << strList.join(";") << "\n";
    for (int i = 0; i < m_pPipManager->rowCount(); i++)
    {
        strList.clear();
        for (int j = 0; j < m_pPipManager->columnCount(); j++) {

            if (m_pPipManager->data(m_pPipManager->index(i, j), Qt::DisplayRole).toString().length() > 0)
            {
                strList.append("\"" + m_pPipManager->data(m_pPipManager->index(i, j), Qt::DisplayRole).toString() + "\"");
            }
            else
            {
                strList.append("");
            }
        }
        output << strList.join(";") + "\n";
    }

    return output.join("");
}

} //end namespace ito
