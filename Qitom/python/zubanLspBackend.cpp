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

#include "zubanLspBackend.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QDebug>
#include <QUrl>
#include <QCoreApplication>
#include <QSettings>

#include "../AppManagement.h"

namespace ito {

//--------------------------------------------------------------------------------------
ZubanLspBackend::ZubanLspBackend(const QString& zubanExecutablePath, QObject* parent)
    : ILanguageServerBackend(parent),
      m_lspClient(nullptr),
      m_executablePath(zubanExecutablePath),
      m_initialized(false),
      m_nextJediRequestId(1)
{
}

//--------------------------------------------------------------------------------------
ZubanLspBackend::~ZubanLspBackend()
{
    if (m_lspClient) {
        m_lspClient->shutdown();
        delete m_lspClient;
    }
}

//--------------------------------------------------------------------------------------
bool ZubanLspBackend::isAvailable() const
{
    // Check if zuban executable exists or can be found
    if (!m_executablePath.isEmpty()) {
        return QFileInfo::exists(m_executablePath);
    }

    // Try to find in PATH
    QString found = findZubanExecutable();
    return !found.isEmpty();
}

//--------------------------------------------------------------------------------------
bool ZubanLspBackend::initialize(const QString& includeItomImportString)
{
    if (m_initialized) {
        return true;
    }

    // Find executable if not set
    if (m_executablePath.isEmpty()) {
        m_executablePath = findZubanExecutable();
    }

    if (m_executablePath.isEmpty() || !QFileInfo::exists(m_executablePath)) {
        emit errorOccurred("ZubanLS executable not found");
        return false;
    }

    // Create LSP client
    m_lspClient = new LspClient(m_executablePath, this);

    // Connect signals
    connect(m_lspClient, &LspClient::initialized, this, &ZubanLspBackend::onLspInitialized);
    connect(m_lspClient, &LspClient::errorOccurred, this, &ZubanLspBackend::onLspError);
    connect(m_lspClient, &LspClient::completionReceived, this, &ZubanLspBackend::onLspCompletionReceived);
    connect(m_lspClient, &LspClient::signatureHelpReceived, this, &ZubanLspBackend::onLspSignatureHelpReceived);
    connect(m_lspClient, &LspClient::definitionReceived, this, &ZubanLspBackend::onLspDefinitionReceived);
    connect(m_lspClient, &LspClient::hoverReceived, this, &ZubanLspBackend::onLspHoverReceived);
    connect(m_lspClient, &LspClient::renameReceived, this, &ZubanLspBackend::onLspRenameReceived);

    // Determine root URI (workspace root)
    // For now, use current working directory
    m_rootUri = pathToUri(QDir::currentPath());

    // Start the LSP client
    QStringList args;
    bool started = m_lspClient->start(m_rootUri, args);

    if (!started) {
        emit errorOccurred("Failed to start ZubanLS server");
        return false;
    }

    // Note: m_initialized will be set to true in onLspInitialized()
    return true;
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::setExecutablePath(const QString& path)
{
    m_executablePath = path;
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::requestCompletion(const JediCompletionRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character, Jedi uses 1-based line and column
    int lspRequestId = m_lspClient->requestCompletion(uri, request.m_line - 1, request.m_col);

    if (lspRequestId >= 0) {
        trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
        m_pendingRequests[lspRequestId].jediRequestId = request.m_requestId;
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::requestCalltip(const JediCalltipRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character
    int lspRequestId = m_lspClient->requestSignatureHelp(uri, request.m_line - 1, request.m_col);

    if (lspRequestId >= 0) {
        trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::requestGoToAssignment(const JediAssignmentRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character
    int lspRequestId = m_lspClient->requestDefinition(uri, request.m_line - 1, request.m_col);

    if (lspRequestId >= 0) {
        trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::requestGetHelp(const JediGetHelpRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character
    int lspRequestId = m_lspClient->requestHover(uri, request.m_line - 1, request.m_col);

    if (lspRequestId >= 0) {
        trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::requestRename(const JediRenameRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return;
    }

    QString uri = pathToUri(request.m_filepath);
    ensureDocumentOpen(uri, request.m_code);

    // LSP uses 0-based line/character
    // Note: For rename, we need to extract the new name from somewhere
    // The request doesn't contain a new name directly, so this is a placeholder
    QString newName = "newName"; // TODO: Get actual new name from request context

    int lspRequestId = m_lspClient->requestRename(uri, request.m_line - 1, request.m_col, newName);

    if (lspRequestId >= 0) {
        trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspInitialized()
{
    m_initialized = true;
    qDebug() << "ZubanLspBackend: LSP server initialized successfully";
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspError(const QString& message)
{
    qWarning() << "ZubanLspBackend: LSP error:" << message;
    emit errorOccurred(message);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspCompletionReceived(int requestId, const QJsonArray& items)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);

    // Convert LSP completion items to Jedi completions
    QList<JediCompletion> completions;
    for (const QJsonValue& itemValue : items) {
        completions.append(convertCompletionItem(itemValue.toObject()));
    }

    emit completionReady(context.jediRequestId, completions, context.sender);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspSignatureHelpReceived(int requestId, const QJsonObject& signatureHelp)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    JediCalltip calltip = convertSignatureHelp(signatureHelp);

    emit calltipReady(calltip, context.sender);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspDefinitionReceived(int requestId, const QJsonArray& locations)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);

    // Take the first location (Jedi returns single assignment)
    if (!locations.isEmpty()) {
        JediAssignment assignment = convertLocation(locations[0].toObject());
        emit goToAssignmentReady(assignment, context.sender);
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspHoverReceived(int requestId, const QJsonObject& hover)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    JediGetHelp help = convertHover(hover);

    emit getHelpReady(help, context.sender);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspRenameReceived(int requestId, const QJsonObject& workspaceEdit)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    QList<JediRename> renames = convertWorkspaceEdit(workspaceEdit);

    emit renameReady(renames, context.sender);
}

//--------------------------------------------------------------------------------------
QString ZubanLspBackend::pathToUri(const QString& path) const
{
    // Convert file path to URI (file:///path/to/file)
    QFileInfo fileInfo(path);
    QString absolutePath = fileInfo.absoluteFilePath();

    // Use QUrl for proper URI encoding (Qt 5.6+)
    QUrl url = QUrl::fromLocalFile(absolutePath);
    return url.toString();
}

//--------------------------------------------------------------------------------------
QString ZubanLspBackend::uriToPath(const QString& uri) const
{
    // Convert URI to local file path
    QUrl url(uri);
    return url.toLocalFile();
}

//--------------------------------------------------------------------------------------
QString ZubanLspBackend::findZubanExecutable() const
{
    // First, try to get path from settings
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    QString settingsPath = settings.value("zubanLsPath", "").toString();
    settings.endGroup();

    if (!settingsPath.isEmpty()) {
        QFileInfo fileInfo(settingsPath);
        if (fileInfo.exists()) {
#ifdef Q_OS_WIN
            // On Windows, check if it's a file (no executable bit)
            if (fileInfo.isFile()) {
                return settingsPath;
            }
#else
            // On Unix, check if it's executable
            if (fileInfo.isExecutable()) {
                return settingsPath;
            }
#endif
        }
        // If path from settings is invalid, log a warning but continue to auto-detection
        qWarning() << "ZubanLS path from settings is invalid or not executable:" << settingsPath;
    }

    // Fallback: Try to find zuban in PATH
    QProcess process;

#ifdef Q_OS_WIN
    process.start("where", QStringList() << "zuban");
#else
    process.start("which", QStringList() << "zuban");
#endif

    if (process.waitForFinished(3000) && process.exitCode() == 0) {
        QString output = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
        if (!output.isEmpty()) {
            return output.split('\n').first(); // Take first match
        }
    }

    return QString();
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::ensureDocumentOpen(const QString& uri, const QString& source)
{
    if (!m_openDocuments.contains(uri)) {
        // Open the document
        m_lspClient->didOpen(uri, "python", source, 1);
        m_openDocuments[uri] = 1;
    } else {
        // Update the document
        int version = ++m_openDocuments[uri];
        m_lspClient->didChange(uri, source, version);
    }
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::trackRequest(int lspRequestId, const QPointer<QObject>& sender, const QByteArray& callbackName)
{
    RequestContext context;
    context.jediRequestId = m_nextJediRequestId++;
    context.sender = sender;
    context.callbackName = callbackName;

    m_pendingRequests[lspRequestId] = context;

    return context.jediRequestId;
}

//--------------------------------------------------------------------------------------
JediCompletion ZubanLspBackend::convertCompletionItem(const QJsonObject& item) const
{
    // Convert LSP CompletionItem to JediCompletion
    QString label = item["label"].toString();
    QString detail = item["detail"].toString();
    QString documentation = item["documentation"].toString();
    int kind = item["kind"].toInt();

    // Map LSP kind to icon name
    QString icon;
    switch (kind) {
        case 3: icon = "code-function"; break;      // Function
        case 7: icon = "code-class"; break;         // Class
        case 6: icon = "code-variable"; break;      // Variable
        case 9: icon = "code-module"; break;        // Module
        case 14: icon = "quickopen"; break;         // Keyword
        default: icon = "code-variable"; break;
    }

    QStringList tooltips;
    if (!detail.isEmpty()) {
        tooltips.append(detail);
    }

    return JediCompletion(label, tooltips, icon, documentation);
}

//--------------------------------------------------------------------------------------
JediCalltip ZubanLspBackend::convertSignatureHelp(const QJsonObject& signatureHelp) const
{
    // Convert LSP SignatureHelp to JediCalltip
    QJsonArray signatures = signatureHelp["signatures"].toArray();
    int activeParameter = signatureHelp["activeParameter"].toInt(-1);

    if (signatures.isEmpty()) {
        return JediCalltip();
    }

    QJsonObject signature = signatures[0].toObject();
    QString label = signature["label"].toString();

    // Parse parameters
    QStringList params;
    QJsonArray parameters = signature["parameters"].toArray();
    for (const QJsonValue& param : parameters) {
        params.append(param.toObject()["label"].toString());
    }

    // Extract method name from label (before '(')
    QString methodName = label.left(label.indexOf('('));

    return JediCalltip(methodName, params, activeParameter, -1, -1);
}

//--------------------------------------------------------------------------------------
JediAssignment ZubanLspBackend::convertLocation(const QJsonObject& location) const
{
    // Convert LSP Location to JediAssignment
    QString uri = location["uri"].toString();
    QString path = uriToPath(uri);

    QJsonObject range = location["range"].toObject();
    QJsonObject start = range["start"].toObject();

    int line = start["line"].toInt() + 1;  // Convert to 1-based
    int column = start["character"].toInt();

    return JediAssignment(path, line, column, QString());
}

//--------------------------------------------------------------------------------------
JediGetHelp ZubanLspBackend::convertHover(const QJsonObject& hover) const
{
    // Convert LSP Hover to JediGetHelp
    QJsonValue contents = hover["contents"];

    QString description;
    if (contents.isString()) {
        description = contents.toString();
    } else if (contents.isObject()) {
        description = contents.toObject()["value"].toString();
    }

    return JediGetHelp(description, QStringList());
}

//--------------------------------------------------------------------------------------
QList<JediRename> ZubanLspBackend::convertWorkspaceEdit(const QJsonObject& workspaceEdit) const
{
    // Convert LSP WorkspaceEdit to list of JediRename
    QList<JediRename> renames;

    QJsonObject changes = workspaceEdit["changes"].toObject();

    for (const QString& uri : changes.keys()) {
        JediRename rename;
        rename.m_filePath = uriToPath(uri);
        rename.m_untitledFile = false;
        rename.m_mainFile = true;
        rename.m_fileInProject = true;

        QJsonArray edits = changes[uri].toArray();
        for (const QJsonValue& editValue : edits) {
            QJsonObject edit = editValue.toObject();
            QJsonObject range = edit["range"].toObject();
            QJsonObject start = range["start"].toObject();

            FileRenameItem item;
            item.lineNumber = start["line"].toInt() + 1; // Convert to 1-based
            item.startColumnIndex = start["character"].toInt();
            item.oldWordSize = edit["newText"].toString().length();

            rename.m_items.append(item);
        }

        renames.append(rename);
    }

    return renames;
}

} // namespace ito
