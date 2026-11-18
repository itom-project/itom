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
#include <QFileInfo>
#include <QProcess>
#include <QDebug>
#include <QUrl>
#include <QCoreApplication>
#include <QSettings>
#include <QCryptographicHash>

#include "../AppManagement.h"
#include "../python/pythonEngine.h"

namespace ito {

//--------------------------------------------------------------------------------------
ZubanLspBackend::ZubanLspBackend(const QString& zubanExecutablePath, QObject* parent)
         : ILanguageServerBackend(parent),
          m_lspClient(nullptr),
          m_executablePath(zubanExecutablePath),
          m_initialized(false),
          m_nextJediRequestId(1),
          m_restartPending(false),
          m_currentFileDirectory(QString())
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
    m_includeItomImportString = includeItomImportString;

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

    // Determine root URI (workspace root): the current directory of itom is
    // considered as root folder of the Python project.
    m_rootUri = pathToUri(QDir::currentPath());

    // Build additional workspace folders: include itom-packages (for itom-stubs)
    // and current sys.path entries (gathered on first access to the backend).
    QStringList additionalWorkspaceFolders;

    // The folder itom-packages contains the itom-stubs package.
    const QString stubBaseFolder =
        QDir(QCoreApplication::applicationDirPath()).absoluteFilePath("itom-packages");
    if (QDir(stubBaseFolder).exists("itom-stubs")) {
        m_stubsFolderUri = pathToUri(stubBaseFolder);
        additionalWorkspaceFolders << m_stubsFolderUri;
    } else {
        qWarning() << "ZubanLspBackend: stub package not found in" << stubBaseFolder;
    }

    // Start the LSP client.
    // The zuban executable requires the subcommand 'server' to start the language
    // server, that communicates via stdin / stdout. Without this argument, zuban only
    // prints its usage to stderr and terminates with the exit code 2.
    QStringList args;
    args << "server";

    bool started = m_lspClient->start(m_rootUri, additionalWorkspaceFolders, args);

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
int ZubanLspBackend::requestCompletion(const CompletionRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    // Update the current file directory from the request
    QFileInfo fileInfo(request.m_path);
    setProjectDirectory(fileInfo.absolutePath());

    const bool injectItomImport =
        !m_includeItomImportString.isEmpty() &&
        request.m_path.endsWith("__console_temp__.py", Qt::CaseInsensitive);
    const int lineOffset = injectItomImport ? 1 : 0;
    const QString source = injectItomImport ?
        (m_includeItomImportString + "\n" + request.m_source) :
        request.m_source;

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, source);

    // Both LSP and the JediXYZRequest structs use 0-based line and character values
    // (the 1-based line number, required by jedi, is only created within the jedi runner).
    // The code completion mode passes the position of the start of the already typed
    // prefix (as required by jedi). A language server expects the real cursor position,
    // such that it can filter and rank its results by the typed prefix.
    int lspRequestId = m_lspClient->requestCompletion(
        uri, request.m_line + lineOffset, request.m_col + request.m_prefix.size());
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(
            lspRequestId,
            request.m_sender,
            request.m_callbackFctName,
            lineOffset,
            request.m_path);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestCalltip(const CalltipRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    // Update the current file directory from the request
    QFileInfo fileInfo(request.m_path);
    setProjectDirectory(fileInfo.absolutePath());

    const bool injectItomImport =
        !m_includeItomImportString.isEmpty() &&
        request.m_path.endsWith("__console_temp__.py", Qt::CaseInsensitive);
    const int lineOffset = injectItomImport ? 1 : 0;
    const QString source = injectItomImport ?
        (m_includeItomImportString + "\n" + request.m_source) :
        request.m_source;

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, source);

    // LSP uses 0-based line/character, as the CalltipRequest does
    int lspRequestId = m_lspClient->requestSignatureHelp(uri, request.m_line + lineOffset, request.m_col);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(
            lspRequestId,
            request.m_sender,
            request.m_callbackFctName,
            lineOffset,
            request.m_path);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestGoToAssignment(const AssignmentRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    // Update the current file directory from the request
    QFileInfo fileInfo(request.m_path);
    setProjectDirectory(fileInfo.absolutePath());

    const bool injectItomImport =
        !m_includeItomImportString.isEmpty() &&
        request.m_path.endsWith("__console_temp__.py", Qt::CaseInsensitive);
    const int lineOffset = injectItomImport ? 1 : 0;
    const QString source = injectItomImport ?
        (m_includeItomImportString + "\n" + request.m_source) :
        request.m_source;

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, source);

    // LSP uses 0-based line/character
    // mode mapping (Jedi-compatible intent):
    // 0: goto definition -> definition
    // 1: goto assignment (no follow imports) -> declaration (fallback: definition)
    // 2: goto assignment (follow imports) -> definition
    int lspRequestId = -1;

    if (request.m_mode == 1 && m_lspClient->supportsDeclaration()) {
        lspRequestId = m_lspClient->requestDeclaration(uri, request.m_line + lineOffset, request.m_col);
    } else {
        lspRequestId = m_lspClient->requestDefinition(uri, request.m_line + lineOffset, request.m_col);
    }

    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(
            lspRequestId,
            request.m_sender,
            request.m_callbackFctName,
            lineOffset,
            request.m_path);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestGetHelp(const GetHelpRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    // Update the current file directory from the request
    QFileInfo fileInfo(request.m_path);
    setProjectDirectory(fileInfo.absolutePath());

    const bool injectItomImport =
        !m_includeItomImportString.isEmpty() &&
        request.m_path.endsWith("__console_temp__.py", Qt::CaseInsensitive);
    const int lineOffset = injectItomImport ? 1 : 0;
    const QString source = injectItomImport ?
        (m_includeItomImportString + "\n" + request.m_source) :
        request.m_source;

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, source);

    // LSP uses 0-based line/character
    int lspRequestId = m_lspClient->requestHover(uri, request.m_line + lineOffset, request.m_col);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(
            lspRequestId,
            request.m_sender,
            request.m_callbackFctName,
            lineOffset,
            request.m_path);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestRename(const RenameRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    // Update the current file directory from the request
    QFileInfo fileInfo(request.m_filepath);
    setProjectDirectory(fileInfo.absolutePath());

    QString uri = pathToUri(request.m_filepath);
    ensureDocumentOpen(uri, request.m_code);

    // LSP uses 0-based line/character, whereas RenameRequest::m_line is 1-based.
    // Only the ranges of the returned edits are used. The real new name is entered
    // by the user afterwards, therefore a placeholder is used here.
    const QString newName = "__itom_rename__";

    int lspRequestId = m_lspClient->requestRename(uri, request.m_line - 1, request.m_col, newName);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
        m_pendingRenameRequests[requestId] = request;
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspInitialized()
{
    m_initialized = true;
    qDebug() << "ZubanLspBackend: LSP server initialized successfully";

    emit initialized();
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
    QList<CompletionItem> completions;
    for (const QJsonValue& itemValue : items) {
        completions.append(convertCompletionItem(itemValue.toObject()));
    }

    emit completionReady(context.jediRequestId, completions);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspSignatureHelpReceived(int requestId, const QJsonObject& signatureHelp)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    CalltipItem calltip = convertSignatureHelp(signatureHelp);

    QVector<ito::CalltipItem> calltips;
    calltips << calltip;

    emit calltipReady(context.jediRequestId, calltips);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspDefinitionReceived(int requestId, const QJsonArray& locations)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);

    QVector<AssignmentItem> assignments;

    const QString sourcePath = QDir::cleanPath(context.sourcePath);

    for (const QJsonValue& location : locations) {
        AssignmentItem item = convertLocation(location.toObject());

        if (context.lineOffset > 0 && !sourcePath.isEmpty() &&
            QDir::cleanPath(item.m_path) == sourcePath)
        {
            item.m_line = qMax(0, item.m_line - context.lineOffset);
        }

        assignments.append(item);
    }

    // always emit the signal, such that the caller knows about empty results
    emit goToAssignmentReady(context.jediRequestId, assignments);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspHoverReceived(int requestId, const QJsonObject& hover)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    HelpItem help = convertHover(hover);

    emit getHelpReady(context.jediRequestId, help);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspRenameReceived(int requestId, const QJsonObject& workspaceEdit)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    QList<RenameFileChanges> renames = convertWorkspaceEdit(workspaceEdit);
    const RenameRequest request = m_pendingRenameRequests.take(context.jediRequestId);

    // determine the old value from the word under the requested position
    QString oldValue;
    const QStringList lines = request.m_code.split('\n');
    const int lineIdx = request.m_line - 1;

    if (lineIdx >= 0 && lineIdx < lines.size())
    {
        const QString& text = lines[lineIdx];
        int start = qBound(0, request.m_col, (int)text.size());
        int end = start;

        while (start > 0 && (text[start - 1].isLetterOrNumber() || text[start - 1] == '_'))
        {
            --start;
        }

        while (end < text.size() && (text[end].isLetterOrNumber() || text[end] == '_'))
        {
            ++end;
        }

        oldValue = text.mid(start, end - start);
    }

    QDir requestDir = QFileInfo(request.m_filepath).absoluteDir();
    const QString requestPath = QFileInfo(request.m_filepath).absoluteFilePath();
    QVector<RenameFileChanges> result;

    for (RenameFileChanges rename : renames)
    {
        const bool isRequestFile = QFileInfo(rename.m_filePath).absoluteFilePath() == requestPath;
        rename.m_mainFile = isRequestFile;
        rename.m_fileInProject = true;

        if (isRequestFile && request.m_untitledFile)
        {
            rename.m_untitledFile = true;
            rename.m_untitledFilename = request.m_untitledName;
        }

        result.append(rename);
    }

    emit renameReady(context.jediRequestId, result, oldValue, !result.isEmpty(), QString());
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
/* Returns true if the given path points to an existing file that can be executed.
*/
/*static*/ bool ZubanLspBackend::isExecutableFile(const QString& path)
{
    QFileInfo fileInfo(path);

    if (!fileInfo.exists() || !fileInfo.isFile())
    {
        return false;
    }

#ifdef Q_OS_WIN
    // On Windows, there is no executable bit
    return true;
#else
    return fileInfo.isExecutable();
#endif
}

//--------------------------------------------------------------------------------------
QString ZubanLspBackend::findZubanExecutable() const
{
    // 1. try to get the path from the settings
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");
    QString settingsPath = settings.value("zubanLsPath", "").toString();
    settings.endGroup();

    if (!settingsPath.isEmpty()) {
        if (isExecutableFile(settingsPath))
        {
            return settingsPath;
        }

        // If path from settings is invalid, log a warning but continue to auto-detection
        qWarning() << "ZubanLS path from settings is invalid or not executable:" << settingsPath;
    }

    // 2. try to find zuban in the scripts directory of the Python root, used by itom.
    const PythonEngine* pyEng = qobject_cast<const PythonEngine*>(AppManagement::getPythonEngine());

    if (pyEng)
    {
        const QString pythonExecutable = pyEng->getPythonExecutable();

        if (!pythonExecutable.isEmpty())
        {
            QDir pythonRoot = QFileInfo(pythonExecutable).absoluteDir();

#ifdef Q_OS_WIN
            // the python executable is located in the Python root itself,
            // scripts are located in the subdirectory 'Scripts'.
            const QString scriptsDirName = "Scripts";
            const QString zubanFileName = "zuban.exe";
#else
            // the python executable is already located in the 'bin' subdirectory
            // of the Python root, which also contains the scripts.
            const QString scriptsDirName = "bin";
            const QString zubanFileName = "zuban";

            if (pythonRoot.dirName() == scriptsDirName)
            {
                pythonRoot.cdUp();
            }
#endif

            if (pythonRoot.cd(scriptsDirName))
            {
                const QString zubanPath = pythonRoot.absoluteFilePath(zubanFileName);

                if (isExecutableFile(zubanPath))
                {
                    return zubanPath;
                }
            }
        }
    }

    // 3. Fallback: Try to find zuban in PATH
    QProcess process;

#ifdef Q_OS_WIN
    process.start("where", QStringList() << "zuban");
#else
    process.start("which", QStringList() << "zuban");
#endif

    if (process.waitForFinished(3000) && process.exitCode() == 0) {
        QString output = QString::fromUtf8(process.readAllStandardOutput()).trimmed();
        if (!output.isEmpty()) {
            return output.split('\n').first().trimmed(); // Take first match
        }
    }

    return QString();
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::closeDocument(const QString& filePath)
{
    if (filePath.isEmpty()) {
        return;
    }

    const QString uri = pathToUri(filePath);

    if (m_openDocuments.remove(uri) > 0 && m_lspClient) {
        m_lspClient->didClose(uri);
    }
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::ensureDocumentOpen(const QString& uri, const QString& source)
{
    // The hash is only used to detect changes of the text. MD5 is sufficient here (no
    // security relevance) and makes an undetected change practically impossible.
    const QByteArray hash = QCryptographicHash::hash(source.toUtf8(), QCryptographicHash::Md5);

    QMap<QString, OpenDocument>::iterator it = m_openDocuments.find(uri);

    if (it == m_openDocuments.end()) {
        // Open the document
        m_lspClient->didOpen(uri, "python", source, 1);

        OpenDocument doc;
        doc.version = 1;
        doc.hash = hash;
        m_openDocuments.insert(uri, doc);
    } else if (it->hash != hash) {
        // Update the document, since its content changed
        it->version++;
        it->hash = hash;
        m_lspClient->didChange(uri, source, it->version);
    }
    // else: the server already knows the current content, nothing to send.
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::trackRequest(
    int lspRequestId,
    const QPointer<QObject>& sender,
    const QByteArray& callbackName,
    int lineOffset,
    const QString& sourcePath)
{
    RequestContext context;
    context.jediRequestId = m_nextJediRequestId++;
    context.sender = sender;
    context.callbackName = callbackName;
    context.lineOffset = lineOffset;
    context.sourcePath = sourcePath;

    m_pendingRequests[lspRequestId] = context;

    return context.jediRequestId;
}

//--------------------------------------------------------------------------------------
CompletionItem ZubanLspBackend::convertCompletionItem(const QJsonObject& item) const
{
    // Convert LSP CompletionItem to CompletionItem
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

    return CompletionItem(label, tooltips, icon, documentation);
}

//--------------------------------------------------------------------------------------
/* ParameterInformation::label is either the label itself or a pair of utf-16 code unit
   offsets into the label of the enclosing signature (if the client announced
   labelOffsetSupport). Both variants are accepted here. */
static QString extractParameterLabel(const QJsonValue& labelValue, const QString& signatureLabel)
{
    if (labelValue.isString())
    {
        return labelValue.toString();
    }
    else if (labelValue.isArray())
    {
        const QJsonArray offsets = labelValue.toArray();

        if (offsets.size() == 2)
        {
            const int start = offsets[0].toInt(-1);
            const int end = offsets[1].toInt(-1);

            if (start >= 0 && end > start && end <= signatureLabel.size())
            {
                return signatureLabel.mid(start, end - start);
            }
        }
    }

    return QString();
}

//--------------------------------------------------------------------------------------
/* Splits the argument list of a signature label like 'foo(a: int, b: Dict[str, int])'
   into its single arguments. Commas within nested brackets or strings (e.g. default
   values or generic types) do not separate arguments. */
static QStringList splitSignatureParameters(const QString& signatureLabel)
{
    const int openIdx = signatureLabel.indexOf(QLatin1Char('('));

    if (openIdx < 0)
    {
        return QStringList();
    }

    QStringList params;
    QString current;
    int depth = 0;
    QChar quote;

    for (int idx = openIdx; idx < signatureLabel.size(); ++idx)
    {
        const QChar c = signatureLabel.at(idx);

        if (!quote.isNull())
        {
            current += c;

            if (c == quote)
            {
                quote = QChar();
            }

            continue;
        }

        if (c == QLatin1Char('\'') || c == QLatin1Char('"'))
        {
            quote = c;
            current += c;
        }
        else if (c == QLatin1Char('(') || c == QLatin1Char('[') || c == QLatin1Char('{'))
        {
            depth++;

            if (depth > 1)
            {
                // the opening bracket of the signature itself is not part of any argument
                current += c;
            }
        }
        else if (c == QLatin1Char(')') || c == QLatin1Char(']') || c == QLatin1Char('}'))
        {
            depth--;

            if (depth <= 0)
            {
                // the closing bracket of the signature terminates the argument list
                break;
            }

            current += c;
        }
        else if (c == QLatin1Char(',') && depth == 1)
        {
            params << current.trimmed();
            current.clear();
        }
        else
        {
            current += c;
        }
    }

    if (!current.trimmed().isEmpty())
    {
        params << current.trimmed();
    }

    return params;
}

//--------------------------------------------------------------------------------------
/* Returns true, if at least one of the given arguments carries a type annotation or a
   default value, i.e. if it provides more information than the bare argument name. */
static bool hasArgumentDetails(const QStringList& params)
{
    foreach (const QString& param, params)
    {
        if (param.contains(QLatin1Char(':')) || param.contains(QLatin1Char('=')))
        {
            return true;
        }
    }

    return false;
}

//--------------------------------------------------------------------------------------
/* ZubanLS reports optional arguments as 'name: type =' without the default value.
   Remove this dangling '=' to avoid an incomplete looking calltip. */
static QString stripEmptyDefaultValue(const QString& param)
{
    QString p = param.trimmed();

    if (p.endsWith(QLatin1Char('=')))
    {
        p.chop(1);
        p = p.trimmed();
    }

    return p;
}

//--------------------------------------------------------------------------------------
CalltipItem ZubanLspBackend::convertSignatureHelp(const QJsonObject& signatureHelp) const
{
    // Convert LSP SignatureHelp to CalltipItem. The result must have the same format
    // as the one created by the jedi backend in itomJediLib.calltips: the parameters
    // contain their type annotations and the currently edited parameter is marked bold.
    const QJsonArray signatures = signatureHelp["signatures"].toArray();

    if (signatures.isEmpty()) {
        return CalltipItem();
    }

    int activeSignature = signatureHelp["activeSignature"].toInt(0);

    if (activeSignature < 0 || activeSignature >= signatures.size())
    {
        activeSignature = 0;
    }

    const QJsonObject signature = signatures[activeSignature].toObject();

    // the label contains the entire signature, including all type annotations
    const QString label = signature["label"].toString();

    // since LSP 3.16 a single signature may override the active parameter
    const int activeParameter = signature.contains("activeParameter") ?
        signature["activeParameter"].toInt(-1) :
        signatureHelp["activeParameter"].toInt(-1);

    QStringList params;
    const QJsonArray parameters = signature["parameters"].toArray();

    for (const QJsonValue& param : parameters) {
        params.append(extractParameterLabel(param.toObject()["label"], label));
    }

    // Some language servers omit the parameters, provide empty labels or only report
    // the bare argument names. The type annotations and default values are always part
    // of the signature label, therefore parse the argument list from there in this case.
    bool labelsUsable = !params.isEmpty();

    foreach (const QString& param, params)
    {
        if (param.trimmed().isEmpty())
        {
            labelsUsable = false;
            break;
        }
    }

    const QStringList parsedParams = splitSignatureParameters(label);

    if (!labelsUsable)
    {
        if (!parsedParams.isEmpty())
        {
            params = parsedParams;
        }
    }
    else if (!hasArgumentDetails(params) && hasArgumentDetails(parsedParams) &&
             parsedParams.size() == params.size())
    {
        // the parameter labels are just the argument names, but the signature label
        // provides the annotated variant, which is more informative.
        params = parsedParams;
    }

    for (int idx = 0; idx < params.size(); ++idx)
    {
        params[idx] = stripEmptyDefaultValue(params[idx]);
    }

    // the calltip is displayed as rich text, hence escape the parameters and mark
    // the currently edited one as bold.
    for (int idx = 0; idx < params.size(); ++idx)
    {
        params[idx] = params[idx].trimmed().toHtmlEscaped();

        if (idx == activeParameter)
        {
            params[idx] = QString("<b>%1</b>").arg(params[idx]);
        }
    }

    // Extract method name from label (before '('). If the label does not contain any
    // bracket, QString::left(-1) returns the entire label.
    QString methodName = label.left(label.indexOf(QLatin1Char('('))).trimmed();

    if (methodName.startsWith(QLatin1String("def ")))
    {
        methodName = methodName.mid(4).trimmed();
    }

    // m_column and m_bracketStartCol are only used to shift the tooltip to the opening
    // bracket of the call. The LSP response does not provide this position, therefore
    // use equal values to place the tooltip at the current cursor position.
    return CalltipItem(methodName.toHtmlEscaped(), params, 0, -1, 0);
}

//--------------------------------------------------------------------------------------
AssignmentItem ZubanLspBackend::convertLocation(const QJsonObject& location) const
{
    // Convert LSP Location or LocationLink to AssignmentItem.
    // Location: {uri, range}
    // LocationLink: {originSelectionRange, targetUri, targetRange, targetSelectionRange}
    QString uri;
    QJsonObject range;

    if (location.contains("targetUri"))
    {
        uri = location["targetUri"].toString();

        // the selection range points to the name of the symbol, the target range
        // to the entire definition (e.g. including decorators or docstrings).
        range = location.contains("targetSelectionRange") ?
            location["targetSelectionRange"].toObject() :
            location["targetRange"].toObject();
    }
    else
    {
        uri = location["uri"].toString();
        range = location["range"].toObject();
    }

    QString path = uriToPath(uri);
    QJsonObject start = range["start"].toObject();

    // AssignmentItem::m_line is 0-based, like the LSP position
    int line = start["line"].toInt();
    int column = start["character"].toInt();

    return AssignmentItem(path, line, column, QString());
}

//--------------------------------------------------------------------------------------
HelpItem ZubanLspBackend::convertHover(const QJsonObject& hover) const
{
    // Convert LSP Hover to HelpItem
    QJsonValue contents = hover["contents"];

    QString description;
    bool markdown = false;

    auto markedString = [&](const QJsonValue& v) -> QString {
        if (v.isString()) {
            return v.toString();
        }

        // deprecated MarkedString {language, value}
        const QJsonObject o = v.toObject();
        const QString language = o["language"].toString();
        const QString value = o["value"].toString();
        return language.isEmpty() ? value : "```" + language + "\n" + value + "\n```";
    };

    if (contents.isString()) {
        description = contents.toString();
        markdown = true; // MarkedString plain strings are interpreted as markdown
    } else if (contents.isArray()) {
        QStringList parts;
        for (const QJsonValue& v : contents.toArray()) {
            parts << markedString(v);
        }
        description = parts.join("\n\n");
        markdown = true;
    } else if (contents.isObject()) {
        const QJsonObject obj = contents.toObject();

        if (obj.contains("kind")) {
            // MarkupContent
            description = obj["value"].toString();
            markdown = (obj["kind"].toString() == "markdown");
        } else {
            description = markedString(obj);
            markdown = true;
        }
    }

    HelpItem help(description, QStringList());
    help.m_markdown = markdown;
    return help;
}

//--------------------------------------------------------------------------------------
QList<RenameFileChanges> ZubanLspBackend::convertWorkspaceEdit(const QJsonObject& workspaceEdit) const
{
    // Convert LSP WorkspaceEdit to list of RenameFileChanges
    QList<RenameFileChanges> renames;

    QJsonObject changes = workspaceEdit["changes"].toObject();

    // newer servers may return documentChanges instead of changes
    const QJsonArray documentChanges = workspaceEdit["documentChanges"].toArray();

    for (const QJsonValue& docChange : documentChanges) {
        QJsonObject dc = docChange.toObject();
        QString uri = dc["textDocument"].toObject()["uri"].toString();

        if (!uri.isEmpty()) {
            changes[uri] = dc["edits"].toArray();
        }
    }

    for (const QString& uri : changes.keys()) {
        RenameFileChanges rename;
        rename.m_filePath = uriToPath(uri);
        rename.m_untitledFile = false;
        rename.m_mainFile = true;
        rename.m_fileInProject = true;

        QJsonArray edits = changes[uri].toArray();
        for (const QJsonValue& editValue : edits) {
            QJsonObject edit = editValue.toObject();
            QJsonObject range = edit["range"].toObject();
            QJsonObject start = range["start"].toObject();
            QJsonObject end = range["end"].toObject();

            RenameTextItem item;
            item.lineNumber = start["line"].toInt() + 1; // Convert to 1-based
            item.startColumnIndex = start["character"].toInt();
            item.oldWordSize = end["character"].toInt() - item.startColumnIndex;

            rename.m_items.append(item);
        }

        renames.append(rename);
    }

    return renames;
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::setProjectDirectory(const QString& directory)
{
    // If the directory changed, we need to add it to Zuban's workspace folders
    // so that Zuban can resolve modules in the same directory as the current file.
    if (directory == m_currentFileDirectory || directory.isEmpty())
    {
        return; // No change or invalid directory
    }

    m_currentFileDirectory = directory;

    if (!m_initialized || !m_lspClient)
    {
        return; // Backend not yet initialized
    }

    // Add the current file directory as workspace folder using didChangeWorkspaceFolders
    // This allows Zuban to resolve modules in the same directory without a full restart
    QString directoryUri = pathToUri(directory);

    // Check if this folder is already in the workspace folders
    // We don't add it if it's already there or if it's the root or stubs folder
    if (directoryUri == m_rootUri || directoryUri == m_stubsFolderUri)
    {
        return; // Already in workspace folders
    }

    // Send the workspace folder change notification to Zuban
    m_lspClient->didChangeWorkspaceFolders(QStringList() << directoryUri, QStringList());
}

//--------------------------------------------------------------------------------------
bool ZubanLspBackend::restart(const QStringList& additionalWorkspaceFolders)
{
    if (!m_lspClient || !m_initialized)
    {
        return false;
    }

    // Mark pending to ignore any signals during shutdown
    m_restartPending = true;
    m_pendingAdditionalWorkspaceFolders = additionalWorkspaceFolders;

    // Disconnect signals to avoid handling them during shutdown
    disconnect(m_lspClient, nullptr, this, nullptr);

    // Connect to shutdown completion signal
    connect(m_lspClient, &LspClient::shutdownComplete,
        this, &ZubanLspBackend::onLspShutdownForRestart, Qt::QueuedConnection);

    // Shutdown the server
    m_lspClient->shutdown();

    return true;
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspShutdownForRestart()
{
    // Shutdown is complete, now reinitialize with new workspace folders
    if (!m_restartPending || !m_lspClient)
    {
        return;
    }

    // Delete the old client
    delete m_lspClient;
    m_lspClient = nullptr;
    m_initialized = false;
    m_pendingRequests.clear();
    m_openDocuments.clear();

    // Create and connect new LSP client
    m_lspClient = new LspClient(m_executablePath, this);

    connect(m_lspClient, &LspClient::initialized, this, &ZubanLspBackend::onLspInitialized);
    connect(m_lspClient, &LspClient::errorOccurred, this, &ZubanLspBackend::onLspError);
    connect(m_lspClient, &LspClient::completionReceived, this, &ZubanLspBackend::onLspCompletionReceived);
    connect(m_lspClient, &LspClient::signatureHelpReceived, this, &ZubanLspBackend::onLspSignatureHelpReceived);
    connect(m_lspClient, &LspClient::definitionReceived, this, &ZubanLspBackend::onLspDefinitionReceived);
    connect(m_lspClient, &LspClient::hoverReceived, this, &ZubanLspBackend::onLspHoverReceived);
    connect(m_lspClient, &LspClient::renameReceived, this, &ZubanLspBackend::onLspRenameReceived);

    // Start the LSP client with updated workspace folders
    QStringList args;
    args << "server";

    bool started = m_lspClient->start(m_rootUri, m_pendingAdditionalWorkspaceFolders, args);

    if (!started)
    {
        emit errorOccurred("Failed to restart ZubanLS server");
    }

    m_restartPending = false;
}

} // namespace ito
