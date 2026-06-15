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

#pragma once

#include "../../common/addInInterface.h"
#include "../../common/sharedStructures.h"

#include "../models/pipManager.h"

#include <qdialog.h>
#include <qvector.h>
#include <qevent.h>
#include <qcolor.h>

#include "ui_dialogPipManager.h"

namespace ito {

class DialogPipManager : public QDialog
{
    Q_OBJECT

    Q_PROPERTY(QColor colorMessage READ colorMessage WRITE setColorMessage DESIGNABLE true);
    Q_PROPERTY(QColor colorError READ colorError WRITE setColorError DESIGNABLE true);

public:
    DialogPipManager(QWidget *parent = NULL, bool standalone = false);
    ~DialogPipManager();

    QColor colorMessage() { return m_colorMessage; }
    void setColorMessage(const QColor &color);

    QColor colorError() { return m_colorError; }
    void setColorError(const QColor &color);

protected:
    void closeEvent(QCloseEvent *e);
    PipGeneralOptions createOptions() const;

    void installOrUpdatePackage(bool update = false);
    QString exportPackageTableToString() const;

private:
    PipManager *m_pPipManager;
    Ui::DialogPipManager ui;
    QString m_logHtml;
    PipManager::Task m_currentTask;
    int m_lastLogEntry; //-1: nothing yet, 0: standard text, 1: error text
    bool m_outputSilent;
    bool m_standalone;
    QColor m_colorMessage;
    QColor m_colorError;

    static QString invisiblePwStr;



private slots:
    void pipVersion(const QString &version);
    void outputReceived(const QString &text, bool success);
    void pipRequestStarted(const PipManager::Task &task, const QString &text, bool outputSilent);
    void pipRequestFinished(const PipManager::Task &task, const QString &text, bool success);
    void on_btnReload_clicked();
    void on_btnCheckForUpdates_clicked();
    void on_btnInstall_clicked();
    void on_btnUninstall_clicked();
    void on_btnUpdate_clicked();
    void on_btnSudoUninstall_clicked();
    void on_btnVerifyInstalledPackages_clicked();
    void on_btnCancelFetchDetails_clicked();
    void treeViewSelectionChanged(const QItemSelection & selected, const QItemSelection & deselected);
    void tableCustomContextMenuRequested(const QPoint &pos);
    void exportTableToClipboard();
    void exportTableToCsv();
    void pipFetchDetailsProgress(int totalNumberOfUnfetchedDetails, int recentlyFetchedDetails, bool finished);
};

} //end namespace ito
