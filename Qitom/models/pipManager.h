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

#ifndef PIPMANAGER_H
#define PIPMANAGER_H

#include <qabstractitemmodel.h>
#include <qprocess.h>

#include "../common/sharedStructures.h"

namespace ito
{

struct PythonPackage
{
    enum Status {Unknown, Uptodate, Outdated};
    PythonPackage() : m_status(Unknown), m_detailsFetched(false) {};
    PythonPackage(const QString &name, const QString &version) :
        m_name(name), m_version(version), m_location(""), m_requires(""), m_status(Unknown), m_newVersion(""), m_detailsFetched(false)
    {}
    QString m_name;
    QString m_version;
    QString m_location;
    QString m_requires;
    Status  m_status;
    QString m_newVersion;
    QString m_summary;
    QString m_homepage;
    QString m_license;
    bool m_detailsFetched;
};

struct PipGeneralOptions
{
    PipGeneralOptions() : isolated(false), logPath(""), proxy(""), timeout(15), retries(5), useTrustedHosts(false) {}
    bool isolated;          //if true, --isolated is added to pip calls
    QString logPath;        //if != "" --log <logPath> is added to pip calls
    QString proxy;          //if != "" --proxy <proxy> is added to pip calls
    int timeout;            //if >= 0 --timeout <sec> is added to pip calls where timeout denotes the number of seconds
    int retries;            //if > 0 --retries <retries> is added to pip calls where retries denotes the number of tries if one command failed.
    bool useTrustedHosts;   //if true, each host in the corresponding text field (semicolon separated list) will be appended via --trusted-host to any pip command
    QStringList trustedHosts; //trusted hosts, only considered if useTrustedHosts is true
};

struct PipInstall
{
    enum Type
    {
        typeWhl = 0,
        typeTarGz = 1,
        typeSearchIndex = 2,
        typeRequirements = 3,
        typePackageSource = 4
    }; //these are the same types than in DialogPipManager

    Type type;
    QString packageName;
    bool upgrade;
    bool installDeps;
    QString findLinks;
    bool ignoreIndex;
    bool runAsSudo;
};

class PipManager : public QAbstractItemModel
{
    Q_OBJECT

    public:
        PipManager(ito::RetVal &retval, QObject *parent = 0);
        ~PipManager();

        enum Task {
            taskNo,
            taskCheckAvailable,
            taskListPackages,
            taskFetchPackagesDetails,
            taskCheckUpdates,
            taskInstall,
            taskUninstall,
            taskVerifyInstalledPackages
        };

        enum PipMode {
            //!< directly call pip as process (might cause encoding errors under windows)
            pipModeDirect,

            //!< call pip via runPipUtf8.py module (uses a non-official interface of pip,
            //!< but sets the cout and cerr streams to UTF8; recommended under Windows)
            pipModeRunPipUtf8
        };

        QVariant data(const QModelIndex &index, int role) const;
        QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
        QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
        QModelIndex parent(const QModelIndex &index) const;
        int rowCount(const QModelIndex &parent = QModelIndex()) const;
        int columnCount(const QModelIndex &parent = QModelIndex()) const;

        void startPipProcess();
        bool isPipStarted() const;
        inline int getPipVersion() const { return m_pipVersion; }

        void checkPipAvailable(const PipGeneralOptions &options = PipGeneralOptions());
        void listAvailablePackages(const PipGeneralOptions &options = PipGeneralOptions(), bool forceReloadDetails = false);
        void checkPackageUpdates(const PipGeneralOptions &options = PipGeneralOptions());
        void checkVerifyInstalledPackages(const PipGeneralOptions &options = PipGeneralOptions());
        void installPackage(const PipInstall &installSettings, const PipGeneralOptions &options = PipGeneralOptions());
        void uninstallPackage(const QString &packageName, bool runAsSudo, const PipGeneralOptions &options = PipGeneralOptions());


        void interruptPipProcess();

        bool isPackageInUseByOther(const QModelIndex &index);


    private:
        QStringList parseGeneralOptions(const PipGeneralOptions &options, bool ignoreRetries = false, bool ignoreVersionCheck = true) const;
        void clearBuffers();
        ito::RetVal initPythonIfStandalone();
        ito::RetVal checkCallMode(); //!< check if pip can be called via the itom-packages/pipProcess/runPipUtf8.py script or directly
        void startProcess(const QStringList &arguments); //!< the arguments string list must not contain -m as first entry.

        QList<QString> m_headers;               //!<  string list of names of column headers
        QList<QVariant> m_alignment;            //!<  list of alignments for the corresponding headers
        QList<PythonPackage> m_pythonPackages;  //!<  list with installed python packages
        QProcess m_pipProcess;
        bool m_pipAvailable;
        QByteArray m_standardOutputBuffer;
        QByteArray m_standardErrorBuffer;
        Task m_currentTask;
        PipGeneralOptions m_generalOptionsCache;
        QString m_pythonPath;
        int m_pipVersion; //form 0x060100 for 6.1.0
        wchar_t *m_pUserDefinedPythonHome; //only used for standalone usage
        PipMode m_pipCallMode;
        QString m_runPipUtf8Path; //!< only valid if m_pipCallMode == pipModeRunPipUtf8

        int m_numberOfUnfetchedPackageDetails;
        int m_numberOfNewlyObtainedPackageDetails;
        bool m_fetchDetailCancelRequested;

        void fetchPackageDetails(const QStringList& names, int totalNumberOfUnfetchedDetails, bool firstCall);
        void updatePythonPackageDetails(const PythonPackage& details);
        bool triggerFetchDetailsForOpenPackages(bool firstCall);

        void finalizeTask(int exitCode = 0);
        void finalizeTaskCheckAvailable(const QString& error, const QString& output, int exitCode);
        void finalizeTaskListPackages(const QString& error, const QString& output);
        void finalizeTaskFetchPackagesDetails(const QString& error, const QString& output);
        void finalizeTaskCheckUpdates(const QString& error, const QString& output);
        void finalizeTaskVerifyInstalledPackages(const QString& error, const QString& output);
        void finalizeTaskInstall(const QString& error, const QString& output);
        void finalizeTaskUninstall(const QString& error, const QString& output);

    private slots:
        void processError(QProcess::ProcessError error);
        void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
        void processReadyReadStandardError();
        void processReadyReadStandardOutput();

    signals:
        void pipManagerBusy();
        void outputAvailable(const QString &text, bool success);
        void pipVersion(const QString &version);
        void pipRequestStarted(const PipManager::Task &task, const QString &text, bool outputSilent = false);
        void pipRequestFinished(const PipManager::Task &task, const QString &text, bool success);
        void pipFetchDetailsProgress(int totalNumberOfUnfetchedDetails, int recentlyFetchedDetails, bool finished);
};

}

#endif //PIPMANAGER_H
