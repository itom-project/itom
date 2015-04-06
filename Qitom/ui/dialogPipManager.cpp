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

#include "../global.h"

#include "dialogPipManagerInstall.h"


namespace ito {

//--------------------------------------------------------------------------------
DialogPipManager::DialogPipManager(QWidget *parent ) :
    QDialog(parent),
    m_pPipManager(NULL),
    m_lastLogEntry(-1)
{
    ui.setupUi(this);

    m_pPipManager = new PipManager(this);

    connect(m_pPipManager, SIGNAL(pipVersion(QString)), this, SLOT(pipVersion(QString)));
    connect(m_pPipManager, SIGNAL(outputAvailable(QString,bool)), this, SLOT(outputReceived(QString,bool)));
    connect(m_pPipManager, SIGNAL(pipRequestStarted(PipManager::Task,QString)), this, SLOT(pipRequestStarted(PipManager::Task,QString)));
    connect(m_pPipManager, SIGNAL(pipRequestFinished(PipManager::Task,QString,bool)), this, SLOT(pipRequestFinished(PipManager::Task,QString,bool)));

    m_pPipManager->checkPipAvailable(createOptions());

    ui.tablePackages->setModel(m_pPipManager);
    ui.groupPipSettings->setCollapsed(true);
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
void DialogPipManager::pipRequestStarted(const PipManager::Task &task, const QString &text)
{
    outputReceived(text, true);

    ui.btnInstall->setEnabled(false);
    ui.btnReload->setEnabled(false);
    ui.btnOk->setEnabled(false);
    ui.btnCheckForUpdates->setEnabled(false);

    m_currentTask = task;
}

//--------------------------------------------------------------------------------
void DialogPipManager::pipRequestFinished(const PipManager::Task &task, const QString &text, bool success)
{
    if (text != "")
    {
        outputReceived(text, success);
    }

    m_currentTask = PipManager::taskNo;

    ui.btnInstall->setEnabled(true);
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
}

//---------------------------------------------------------------------------------
void DialogPipManager::on_btnInstall_clicked()
{
    DialogPipManagerInstall *dpmi = new DialogPipManagerInstall(this);
    if (dpmi->exec() == QDialog::Accepted)
    {

    }

    DELETE_AND_SET_NULL(dpmi);
}

} //end namespace ito