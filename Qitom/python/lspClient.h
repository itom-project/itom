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

#ifndef LSPCLIENT_H
#define LSPCLIENT_H

#include <QObject>
#include <QProcess>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QString>
#include <QByteArray>
#include <QMap>
#include <QPointer>

namespace ito {

/**
 * @brief Language Server Protocol (LSP) Client for itom
 * 
 * Implements JSON-RPC 2.0 communication with LSP-compatible language servers
 * (e.g., ZubanLS, pylsp, pyright). Compatible with Qt 5.6+ and LSP spec 3.17.
 * 
 * The client manages the lifecycle of an LSP server process, handles message
 * serialization/deserialization, and provides a Qt-friendly API for LSP requests.
 */
class LspClient : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief LSP client state machine
     */
    enum State {
        NotStarted,    //!< Server process not yet started
        Starting,      //!< Server process starting
        Initializing,  //!< Waiting for initialize response
        Running,       //!< Server ready for requests
        ShuttingDown,  //!< Shutdown requested
        Stopped,       //!< Server stopped cleanly
        Error          //!< Server encountered an error
    };

    /**
     * @brief Constructor
     * @param serverExecutable Path to LSP server executable
     * @param parent Parent QObject
     */
    explicit LspClient(const QString& serverExecutable, QObject* parent = nullptr);

    /**
     * @brief Destructor - ensures clean server shutdown
     */
    virtual ~LspClient();

    /**
     * @brief Start the LSP server and initialize
     * @param rootUri Root URI for the workspace (file:///path/to/project)
     * @param serverArgs Additional command-line arguments for the server
     * @return true if server started successfully, false otherwise
     */
    bool start(const QString& rootUri, const QStringList& serverArgs = QStringList());

    /**
     * @brief Shutdown the LSP server cleanly
     */
    void shutdown();

    /**
     * @brief Get current state of the client
     * @return Current state
     */
    State state() const { return m_state; }

    /**
     * @brief Check if client is ready for requests
     * @return true if state is Running
     */
    bool isReady() const { return m_state == Running; }

    // Document Synchronization Methods

    /**
     * @brief Notify server that a document was opened
     * @param uri Document URI (file:///path/to/file.py)
     * @param languageId Language identifier (e.g., "python")
     * @param text Full document text
     * @param version Document version (default 1)
     */
    void didOpen(const QString& uri, const QString& languageId, const QString& text, int version = 1);

    /**
     * @brief Notify server that a document was changed
     * @param uri Document URI
     * @param text New full document text (full sync mode)
     * @param version New document version
     */
    void didChange(const QString& uri, const QString& text, int version);

    /**
     * @brief Notify server that a document was saved
     * @param uri Document URI
     * @param text Optional document text (if server requests it)
     */
    void didSave(const QString& uri, const QString& text = QString());

    /**
     * @brief Notify server that a document was closed
     * @param uri Document URI
     */
    void didClose(const QString& uri);

    // Language Feature Request Methods

    /**
     * @brief Request code completion at a position
     * @param uri Document URI
     * @param line Line number (0-based)
     * @param character Character position (0-based)
     * @return Request ID, or -1 if not ready
     */
    int requestCompletion(const QString& uri, int line, int character);

    /**
     * @brief Request signature help (calltips) at a position
     * @param uri Document URI
     * @param line Line number (0-based)
     * @param character Character position (0-based)
     * @return Request ID, or -1 if not ready
     */
    int requestSignatureHelp(const QString& uri, int line, int character);

    /**
     * @brief Request go-to-definition at a position
     * @param uri Document URI
     * @param line Line number (0-based)
     * @param character Character position (0-based)
     * @return Request ID, or -1 if not ready
     */
    int requestDefinition(const QString& uri, int line, int character);

    /**
     * @brief Request hover information at a position
     * @param uri Document URI
     * @param line Line number (0-based)
     * @param character Character position (0-based)
     * @return Request ID, or -1 if not ready
     */
    int requestHover(const QString& uri, int line, int character);

    /**
     * @brief Request rename refactoring
     * @param uri Document URI
     * @param line Line number (0-based)
     * @param character Character position (0-based)
     * @param newName New name for the symbol
     * @return Request ID, or -1 if not ready
     */
    int requestRename(const QString& uri, int line, int character, const QString& newName);

    /**
     * @brief Cancel a pending request
     * @param requestId Request ID to cancel
     */
    void cancelRequest(int requestId);

signals:
    /**
     * @brief Emitted when server initialization is complete
     */
    void initialized();

    /**
     * @brief Emitted when server shutdown is complete
     */
    void shutdownComplete();

    /**
     * @brief Emitted when an error occurs
     * @param message Error message
     */
    void errorOccurred(const QString& message);

    /**
     * @brief Emitted when completion results are received
     * @param requestId Request ID
     * @param items JSON array of completion items
     */
    void completionReceived(int requestId, const QJsonArray& items);

    /**
     * @brief Emitted when signature help is received
     * @param requestId Request ID
     * @param signatureHelp JSON object with signature information
     */
    void signatureHelpReceived(int requestId, const QJsonObject& signatureHelp);

    /**
     * @brief Emitted when definition locations are received
     * @param requestId Request ID
     * @param locations JSON array of location objects
     */
    void definitionReceived(int requestId, const QJsonArray& locations);

    /**
     * @brief Emitted when hover information is received
     * @param requestId Request ID
     * @param hover JSON object with hover information
     */
    void hoverReceived(int requestId, const QJsonObject& hover);

    /**
     * @brief Emitted when rename edits are received
     * @param requestId Request ID
     * @param workspaceEdit JSON object with workspace edit
     */
    void renameReceived(int requestId, const QJsonObject& workspaceEdit);

    /**
     * @brief Emitted when diagnostics are published by the server
     * @param uri Document URI
     * @param diagnostics JSON array of diagnostic objects
     */
    void diagnosticsReceived(const QString& uri, const QJsonArray& diagnostics);

private slots:
    void onProcessReadyRead();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void onProcessError(QProcess::ProcessError error);

private:
    // Message sending
    void sendRequest(const QString& method, const QJsonObject& params, int requestId);
    void sendNotification(const QString& method, const QJsonObject& params);
    void sendResponse(int id, const QJsonValue& result);
    void sendMessage(const QJsonObject& message);

    // Message handling
    void processMessages();
    void handleMessage(const QJsonObject& message);
    void handleResponse(const QJsonObject& response);
    void handleRequest(const QJsonObject& request);
    void handleNotification(const QJsonObject& notification);

    // Helper methods
    int nextRequestId();
    QJsonObject createInitializeParams(const QString& rootUri);

    // State
    State m_state;
    QString m_serverExecutable;
    QProcess* m_process;

    // Request tracking
    int m_nextRequestId;
    struct PendingRequest {
        QString method;
        qint64 timestamp;
    };
    QMap<int, PendingRequest> m_pendingRequests;

    // Message buffer for streaming JSON-RPC
    QByteArray m_messageBuffer;

    // Server capabilities (from initialize response)
    QJsonObject m_serverCapabilities;
};

} // namespace ito

Q_DECLARE_METATYPE(ito::LspClient::State)

#endif // LSPCLIENT_H
