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

#include "lspClient.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDebug>
#include <QRegularExpression>
#include <QUrl>

namespace ito {

//--------------------------------------------------------------------------------------
LspClient::LspClient(const QString& serverExecutable, QObject* parent)
    : QObject(parent),
      m_state(NotStarted),
      m_serverExecutable(serverExecutable),
      m_process(nullptr),
      m_nextRequestId(1)
{
}

//--------------------------------------------------------------------------------------
LspClient::~LspClient()
{
    if (m_process && m_process->state() != QProcess::NotRunning) {
        shutdown();
        m_process->waitForFinished(3000);
    }
}

//--------------------------------------------------------------------------------------
bool LspClient::start(const QString& rootUri, const QStringList& serverArgs)
{
    if (m_state != NotStarted && m_state != Stopped) {
        qWarning() << "LspClient: Cannot start, already running or starting";
        return false;
    }

    m_state = Starting;

    // Create and configure process
    m_process = new QProcess(this);
    m_process->setProcessChannelMode(QProcess::SeparateChannels);

    connect(m_process, &QProcess::readyReadStandardOutput, this, &LspClient::onProcessReadyRead);

    // Qt 5.6 compatible: use old-style signal for QProcess::finished
    connect(m_process, static_cast<void(QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished),
            this, &LspClient::onProcessFinished);

    // Qt version compatibility: error signal was renamed in Qt 5.6
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
    connect(m_process, &QProcess::errorOccurred,
            this, &LspClient::onProcessError);
#else
    connect(m_process, SIGNAL(error(QProcess::ProcessError)),
            this, SLOT(onProcessError(QProcess::ProcessError)));
#endif

    // Start server process
    m_process->start(m_serverExecutable, serverArgs);

    if (!m_process->waitForStarted(5000)) {
        m_state = Error;
        emit errorOccurred("Failed to start LSP server process");
        return false;
    }

    qDebug() << "LspClient: Server started, sending initialize...";

    // Send initialize request
    m_state = Initializing;
    QJsonObject params = createInitializeParams(rootUri);
    sendRequest("initialize", params, nextRequestId());

    return true;
}

//--------------------------------------------------------------------------------------
void LspClient::shutdown()
{
    if (m_state != Running) {
        return;
    }

    m_state = ShuttingDown;
    sendRequest("shutdown", QJsonObject(), nextRequestId());
}

//--------------------------------------------------------------------------------------
void LspClient::didOpen(const QString& uri, const QString& languageId, const QString& text, int version)
{
    if (m_state != Running) {
        qWarning() << "LspClient: Not running, cannot send didOpen";
        return;
    }

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    textDocument["languageId"] = languageId;
    textDocument["version"] = version;
    textDocument["text"] = text;
    params["textDocument"] = textDocument;

    sendNotification("textDocument/didOpen", params);
}

//--------------------------------------------------------------------------------------
void LspClient::didChange(const QString& uri, const QString& text, int version)
{
    if (m_state != Running) {
        return;
    }

    // Full document sync
    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    textDocument["version"] = version;
    params["textDocument"] = textDocument;

    QJsonArray contentChanges;
    QJsonObject change;
    change["text"] = text;
    contentChanges.append(change);
    params["contentChanges"] = contentChanges;

    sendNotification("textDocument/didChange", params);
}

//--------------------------------------------------------------------------------------
void LspClient::didSave(const QString& uri, const QString& text)
{
    if (m_state != Running) {
        return;
    }

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    if (!text.isEmpty()) {
        params["text"] = text;
    }

    sendNotification("textDocument/didSave", params);
}

//--------------------------------------------------------------------------------------
void LspClient::didClose(const QString& uri)
{
    if (m_state != Running) {
        return;
    }

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    sendNotification("textDocument/didClose", params);
}

//--------------------------------------------------------------------------------------
int LspClient::requestCompletion(const QString& uri, int line, int character)
{
    if (m_state != Running) {
        return -1;
    }

    int reqId = nextRequestId();

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    QJsonObject position;
    position["line"] = line;
    position["character"] = character;
    params["position"] = position;

    sendRequest("textDocument/completion", params, reqId);
    return reqId;
}

//--------------------------------------------------------------------------------------
int LspClient::requestSignatureHelp(const QString& uri, int line, int character)
{
    if (m_state != Running) {
        return -1;
    }

    int reqId = nextRequestId();

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    QJsonObject position;
    position["line"] = line;
    position["character"] = character;
    params["position"] = position;

    sendRequest("textDocument/signatureHelp", params, reqId);
    return reqId;
}

//--------------------------------------------------------------------------------------
int LspClient::requestDefinition(const QString& uri, int line, int character)
{
    if (m_state != Running) {
        return -1;
    }

    int reqId = nextRequestId();

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    QJsonObject position;
    position["line"] = line;
    position["character"] = character;
    params["position"] = position;

    sendRequest("textDocument/definition", params, reqId);
    return reqId;
}

//--------------------------------------------------------------------------------------
int LspClient::requestHover(const QString& uri, int line, int character)
{
    if (m_state != Running) {
        return -1;
    }

    int reqId = nextRequestId();

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    QJsonObject position;
    position["line"] = line;
    position["character"] = character;
    params["position"] = position;

    sendRequest("textDocument/hover", params, reqId);
    return reqId;
}

//--------------------------------------------------------------------------------------
int LspClient::requestRename(const QString& uri, int line, int character, const QString& newName)
{
    if (m_state != Running) {
        return -1;
    }

    int reqId = nextRequestId();

    QJsonObject params;
    QJsonObject textDocument;
    textDocument["uri"] = uri;
    params["textDocument"] = textDocument;

    QJsonObject position;
    position["line"] = line;
    position["character"] = character;
    params["position"] = position;

    params["newName"] = newName;

    sendRequest("textDocument/rename", params, reqId);
    return reqId;
}

//--------------------------------------------------------------------------------------
void LspClient::cancelRequest(int requestId)
{
    QJsonObject params;
    params["id"] = requestId;
    sendNotification("$/cancelRequest", params);

    m_pendingRequests.remove(requestId);
}

//--------------------------------------------------------------------------------------
void LspClient::sendRequest(const QString& method, const QJsonObject& params, int requestId)
{
    QJsonObject message;
    message["jsonrpc"] = "2.0";
    message["id"] = requestId;
    message["method"] = method;
    message["params"] = params;

    PendingRequest pending;
    pending.method = method;
    pending.timestamp = QDateTime::currentMSecsSinceEpoch();
    m_pendingRequests[requestId] = pending;

    sendMessage(message);
}

//--------------------------------------------------------------------------------------
void LspClient::sendNotification(const QString& method, const QJsonObject& params)
{
    QJsonObject message;
    message["jsonrpc"] = "2.0";
    message["method"] = method;
    message["params"] = params;

    sendMessage(message);
}

//--------------------------------------------------------------------------------------
void LspClient::sendMessage(const QJsonObject& message)
{
    if (!m_process || m_process->state() != QProcess::Running) {
        qWarning() << "LspClient: Cannot send message, process not running";
        return;
    }

    QJsonDocument doc(message);
    QByteArray json = doc.toJson(QJsonDocument::Compact);

    // LSP uses Content-Length header
    QString header = QString("Content-Length: %1\r\n\r\n").arg(json.size());

    m_process->write(header.toUtf8());
    m_process->write(json);
    m_process->waitForBytesWritten(1000);

    // Debug output
    QString method = message["method"].toString();
    if (!method.isEmpty()) {
        qDebug() << "LspClient >>> Sent:" << method
                 << (message.contains("id") ? QString("(id=%1)").arg(message["id"].toInt()) : "");
    }
}

//--------------------------------------------------------------------------------------
void LspClient::onProcessReadyRead()
{
    m_messageBuffer.append(m_process->readAllStandardOutput());
    processMessages();
}

//--------------------------------------------------------------------------------------
void LspClient::processMessages()
{
    while (true) {
        // Look for Content-Length header
        int headerEnd = m_messageBuffer.indexOf("\r\n\r\n");
        if (headerEnd == -1) {
            break; // Need more data
        }

        // Parse Content-Length
        QString header = QString::fromUtf8(m_messageBuffer.left(headerEnd));
        QRegularExpression regex("Content-Length: (\\d+)");
        QRegularExpressionMatch match = regex.match(header);

        if (!match.hasMatch()) {
            qWarning() << "LspClient: Invalid header:" << header;
            m_messageBuffer.clear();
            break;
        }

        int contentLength = match.captured(1).toInt();
        int messageStart = headerEnd + 4;

        // Check if we have the full message
        if (m_messageBuffer.size() < messageStart + contentLength) {
            break; // Need more data
        }

        // Extract message
        QByteArray messageData = m_messageBuffer.mid(messageStart, contentLength);
        m_messageBuffer.remove(0, messageStart + contentLength);

        // Parse JSON
        QJsonParseError error;
        QJsonDocument doc = QJsonDocument::fromJson(messageData, &error);

        if (error.error != QJsonParseError::NoError) {
            qWarning() << "LspClient: JSON parse error:" << error.errorString();
            continue;
        }

        // Handle message
        handleMessage(doc.object());
    }
}

//--------------------------------------------------------------------------------------
void LspClient::handleMessage(const QJsonObject& message)
{
    if (message.contains("method")) {
        // Request or Notification
        if (message.contains("id")) {
            handleRequest(message);
        } else {
            handleNotification(message);
        }
    } else if (message.contains("id")) {
        // Response
        handleResponse(message);
    } else {
        qWarning() << "LspClient: Unknown message type";
    }
}

//--------------------------------------------------------------------------------------
void LspClient::handleResponse(const QJsonObject& response)
{
    int id = response["id"].toInt();

    if (!m_pendingRequests.contains(id)) {
        qWarning() << "LspClient: Received response for unknown request" << id;
        return;
    }

    QString method = m_pendingRequests.take(id).method;

    qDebug() << "LspClient <<< Received response for:" << method << "(id=" << id << ")";

    // Handle specific responses
    if (method == "initialize") {
        QJsonObject result = response["result"].toObject();
        m_serverCapabilities = result["capabilities"].toObject();
        m_state = Running;

        // Send initialized notification
        sendNotification("initialized", QJsonObject());

        emit initialized();
    }
    else if (method == "shutdown") {
        // Send exit notification
        sendNotification("exit", QJsonObject());
        m_state = Stopped;
        emit shutdownComplete();
    }
    else if (method == "textDocument/completion") {
        QJsonValue result = response["result"];
        QJsonArray itemsArray;

        // Result can be CompletionList or CompletionItem[]
        if (result.isObject()) {
            itemsArray = result.toObject()["items"].toArray();
        } else if (result.isArray()) {
            itemsArray = result.toArray();
        }

        emit completionReceived(id, itemsArray);
    }
    else if (method == "textDocument/signatureHelp") {
        QJsonObject result = response["result"].toObject();
        emit signatureHelpReceived(id, result);
    }
    else if (method == "textDocument/definition") {
        QJsonValue result = response["result"];
        QJsonArray locationsArray;

        // Result can be Location or Location[]
        if (result.isArray()) {
            locationsArray = result.toArray();
        } else if (result.isObject()) {
            locationsArray.append(result);
        }

        emit definitionReceived(id, locationsArray);
    }
    else if (method == "textDocument/hover") {
        QJsonObject result = response["result"].toObject();
        emit hoverReceived(id, result);
    }
    else if (method == "textDocument/rename") {
        QJsonObject result = response["result"].toObject();
        emit renameReceived(id, result);
    }
}

//--------------------------------------------------------------------------------------
void LspClient::handleNotification(const QJsonObject& notification)
{
    QString method = notification["method"].toString();

    qDebug() << "LspClient <<< Received notification:" << method;

    if (method == "textDocument/publishDiagnostics") {
        QJsonObject params = notification["params"].toObject();
        QString uri = params["uri"].toString();
        QJsonArray diagnostics = params["diagnostics"].toArray();

        emit diagnosticsReceived(uri, diagnostics);
    }
}

//--------------------------------------------------------------------------------------
void LspClient::handleRequest(const QJsonObject& request)
{
    // Servers can send requests to clients (e.g., workspace/configuration)
    QString method = request["method"].toString();
    int id = request["id"].toInt();

    qDebug() << "LspClient <<< Received request:" << method << "(id=" << id << ")";

    // For now, send empty response for all requests
    sendResponse(id, QJsonValue());
}

//--------------------------------------------------------------------------------------
void LspClient::sendResponse(int id, const QJsonValue& result)
{
    QJsonObject message;
    message["jsonrpc"] = "2.0";
    message["id"] = id;
    message["result"] = result;

    sendMessage(message);
}

//--------------------------------------------------------------------------------------
int LspClient::nextRequestId()
{
    return m_nextRequestId++;
}

//--------------------------------------------------------------------------------------
QJsonObject LspClient::createInitializeParams(const QString& rootUri)
{
    QJsonObject params;
    params["processId"] = QCoreApplication::applicationPid();
    params["rootUri"] = rootUri;

    // Client capabilities
    QJsonObject capabilities;
    QJsonObject textDocument;

    // Completion capability
    QJsonObject completion;
    QJsonObject completionItem;
    completionItem["snippetSupport"] = false;
    completion["completionItem"] = completionItem;
    textDocument["completion"] = completion;

    // Signature help capability
    QJsonObject signatureHelp;
    QJsonObject signatureInformation;
    QJsonObject parameterInformation;
    parameterInformation["labelOffsetSupport"] = false;
    signatureInformation["parameterInformation"] = parameterInformation;
    signatureHelp["signatureInformation"] = signatureInformation;
    textDocument["signatureHelp"] = signatureHelp;

    capabilities["textDocument"] = textDocument;
    params["capabilities"] = capabilities;

    return params;
}

//--------------------------------------------------------------------------------------
void LspClient::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    qDebug() << "LspClient: Server process finished. Exit code:" << exitCode;

    if (m_state != ShuttingDown && m_state != Stopped) {
        m_state = Error;
        emit errorOccurred(QString("Server process terminated unexpectedly (code %1)").arg(exitCode));
    }
}

//--------------------------------------------------------------------------------------
void LspClient::onProcessError(QProcess::ProcessError error)
{
    QString errorMsg;

    switch (error) {
        case QProcess::FailedToStart:
            errorMsg = "Server failed to start";
            break;
        case QProcess::Crashed:
            errorMsg = "Server crashed";
            break;
        default:
            errorMsg = QString("Server process error: %1").arg(m_process->errorString());
    }

    m_state = Error;
    emit errorOccurred(errorMsg);
}

} // namespace ito
