/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2025, Institut für Technische Optik (ITO),
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

#ifndef ZUBANLSPBACKEND_H
#define ZUBANLSPBACKEND_H

#include "languageServerBackend.h"
#include "lspClient.h"

#include <QObject>
#include <QString>
#include <QMap>
#include <QPointer>
#include <QFileInfo>

namespace ito {

/**
 * @brief ZubanLS language server backend - uses LSP protocol
 * 
 * This backend communicates with ZubanLS (or any LSP-compatible Python server)
 * via the Language Server Protocol. It converts between LSP data structures
 * and itom's Jedi-compatible data structures.
 */
class ZubanLspBackend : public ILanguageServerBackend
{
    Q_OBJECT

public:
    /**
     * @brief Constructor
     * @param zubanExecutablePath Path to zuban executable (empty = auto-detect)
     * @param parent Parent QObject
     */
    explicit ZubanLspBackend(const QString& zubanExecutablePath = QString(), QObject* parent = nullptr);

    /**
     * @brief Destructor
     */
    virtual ~ZubanLspBackend();

    // ILanguageServerBackend interface
    virtual BackendType backendType() const override { return ZubanLS; }
    virtual bool isAvailable() const override;
    virtual bool initialize(const QString& includeItomImportString) override;
    virtual int requestCompletion(const JediCompletionRequest& request) override;
    virtual int requestCalltip(const JediCalltipRequest& request) override;
    virtual int requestGoToAssignment(const JediAssignmentRequest& request) override;
    virtual int requestGetHelp(const JediGetHelpRequest& request) override;
    virtual int requestRename(const JediRenameRequest& request) override;
    virtual void closeDocument(const QString& filePath) override;
    virtual void setProjectDirectory(const QString& directory) override;

    /**
     * @brief Set the path to the ZubanLS executable
     * @param path Path to executable (empty = auto-detect)
     */
    void setExecutablePath(const QString& path);

    /**
     * @brief Get the current executable path
     * @return Path to ZubanLS executable
     */
    QString executablePath() const { return m_executablePath; }

    /**
     * @brief Restart the Zuban server with updated workspace folders.
     *
     * The server is shut down and restarted. Pending requests are cancelled.
     * Use this when sys.path changes and new module search paths need to be
     * announced to the server.
     *
     * @param additionalWorkspaceFolders URIs of additional workspace folders (e.g., sys.path entries)
     * @return true if restart initiated successfully, false otherwise
     */
    bool restart(const QStringList& additionalWorkspaceFolders);

private slots:
    // LSP Client signal handlers
    void onLspInitialized();
    void onLspError(const QString& message);
    void onLspCompletionReceived(int requestId, const QJsonArray& items);
    void onLspSignatureHelpReceived(int requestId, const QJsonObject& signatureHelp);
    void onLspDefinitionReceived(int requestId, const QJsonArray& locations);
    void onLspHoverReceived(int requestId, const QJsonObject& hover);
    void onLspRenameReceived(int requestId, const QJsonObject& workspaceEdit);

    //!< handles the shutdown of the LSP client during a restart
    void onLspShutdownForRestart();

private:
    // Helper methods
    QString pathToUri(const QString& path) const;
    QString uriToPath(const QString& uri) const;

    /**
     * @brief Find ZubanLS executable path
     * 
     * Search order:
     * 1. Check settings: CodeEditor/zubanLsPath. If this path exists and is valid, use it.
     * 2. Check the scripts subdirectory ('Scripts' on Windows, 'bin' else) of the
     *    Python root directory, that is used by itom, for the executable 'zuban'.
     * 3. Otherwise, search in system PATH using 'where'/'which'
     * 4. Return empty string if not found
     * 
     * @return Path to zuban executable, or empty string if not found
     */
    QString findZubanExecutable() const;

    //!< returns true if the given path points to an existing, executable file.
    static bool isExecutableFile(const QString& path);

    void ensureDocumentOpen(const QString& uri, const QString& source);
    int trackRequest(int lspRequestId, const QPointer<QObject>& sender, const QByteArray& callbackName = QByteArray());

    // Conversion methods: LSP -> Jedi
    JediCompletion convertCompletionItem(const QJsonObject& item) const;
    JediCalltip convertSignatureHelp(const QJsonObject& signatureHelp) const;
    JediAssignment convertLocation(const QJsonObject& location) const;
    JediGetHelp convertHover(const QJsonObject& hover) const;
    QList<JediRename> convertWorkspaceEdit(const QJsonObject& workspaceEdit) const;

    // LSP Client
    LspClient* m_lspClient;
    QString m_executablePath;
    bool m_initialized;

    //!< uri of the project folder (current directory of itom), announced as workspace folder.
    QString m_rootUri;

    //!< uri of the itom-packages folder, that contains the itom-stubs package
    //!< (empty, if it does not exist). It is announced as additional workspace folder.
    QString m_stubsFolderUri;

    //!< current working directory of the edited Python file
    //!< (used to allow Zuban to resolve modules in the same directory)
    QString m_currentFileDirectory;

    //!< workspace folders to be used for the next restart (while shutdown is in progress)
    QStringList m_pendingAdditionalWorkspaceFolders;

    //!< true if a restart is in progress (waiting for shutdown to complete)
    bool m_restartPending;

    // Request tracking (LSP request ID -> itom context)
    struct RequestContext {
        int jediRequestId;
        QPointer<QObject> sender;
        QByteArray callbackName;
    };
    QMap<int, RequestContext> m_pendingRequests;
    int m_nextJediRequestId;

    // Document tracking (to avoid reopening same document)
    QMap<QString, int> m_openDocuments; // uri -> version
};

} // namespace ito

#endif // ZUBANLSPBACKEND_H
