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
    virtual void requestCompletion(const JediCompletionRequest& request) override;
    virtual void requestCalltip(const JediCalltipRequest& request) override;
    virtual void requestGoToAssignment(const JediAssignmentRequest& request) override;
    virtual void requestGetHelp(const JediGetHelpRequest& request) override;
    virtual void requestRename(const JediRenameRequest& request) override;

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

private slots:
    // LSP Client signal handlers
    void onLspInitialized();
    void onLspError(const QString& message);
    void onLspCompletionReceived(int requestId, const QJsonArray& items);
    void onLspSignatureHelpReceived(int requestId, const QJsonObject& signatureHelp);
    void onLspDefinitionReceived(int requestId, const QJsonArray& locations);
    void onLspHoverReceived(int requestId, const QJsonObject& hover);
    void onLspRenameReceived(int requestId, const QJsonObject& workspaceEdit);

private:
    // Helper methods
    QString pathToUri(const QString& path) const;
    QString uriToPath(const QString& uri) const;

    /**
     * @brief Find ZubanLS executable path
     * 
     * Search order:
     * 1. Check settings: CodeEditor/zubanLsPath
     * 2. If settings path exists and is valid, use it
     * 3. Otherwise, search in system PATH using 'where'/'which'
     * 4. Return empty string if not found
     * 
     * @return Path to zuban executable, or empty string if not found
     */
    QString findZubanExecutable() const;

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
    QString m_rootUri;

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
