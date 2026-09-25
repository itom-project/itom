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
#include "../python/pythonEngine.h"

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

    // Start the LSP client.
    // The zuban executable requires the subcommand 'server' to start the language
    // server, that communicates via stdin / stdout. Without this argument, zuban only
    // prints its usage to stderr and terminates with the exit code 2.
    QStringList args;
    args << "server";

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
int ZubanLspBackend::requestCompletion(const JediCompletionRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // Both LSP and the JediXYZRequest structs use 0-based line and character values
    // (the 1-based line number, required by jedi, is only created within the jedi runner).
    int lspRequestId = m_lspClient->requestCompletion(uri, request.m_line, request.m_col);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestCalltip(const JediCalltipRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character, as the JediCalltipRequest does
    int lspRequestId = m_lspClient->requestSignatureHelp(uri, request.m_line, request.m_col);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestGoToAssignment(const JediAssignmentRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character
    int lspRequestId = m_lspClient->requestDefinition(uri, request.m_line, request.m_col);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestGetHelp(const JediGetHelpRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    QString uri = pathToUri(request.m_path);
    ensureDocumentOpen(uri, request.m_source);

    // LSP uses 0-based line/character
    int lspRequestId = m_lspClient->requestHover(uri, request.m_line, request.m_col);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
    }

    return requestId;
}

//--------------------------------------------------------------------------------------
int ZubanLspBackend::requestRename(const JediRenameRequest& request)
{
    if (!m_lspClient || !m_initialized) {
        return -1;
    }

    QString uri = pathToUri(request.m_filepath);
    ensureDocumentOpen(uri, request.m_code);

    // LSP uses 0-based line/character, whereas JediRenameRequest::m_line is 1-based
    // Note: For rename, we need to extract the new name from somewhere
    // The request doesn't contain a new name directly, so this is a placeholder
    QString newName = "newName"; // TODO: Get actual new name from request context

    int lspRequestId = m_lspClient->requestRename(uri, request.m_line - 1, request.m_col, newName);
    int requestId = -1;

    if (lspRequestId >= 0) {
        requestId = trackRequest(lspRequestId, request.m_sender, request.m_callbackFctName);
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
    QList<JediCompletion> completions;
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
    JediCalltip calltip = convertSignatureHelp(signatureHelp);

    QVector<ito::JediCalltip> calltips;
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

    // Take the first location (Jedi returns single assignment)
    if (!locations.isEmpty()) {
        JediAssignment assignment = convertLocation(locations[0].toObject());
        emit goToAssignmentReady(context.jediRequestId, assignment);
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

    emit getHelpReady(context.jediRequestId, help);
}

//--------------------------------------------------------------------------------------
void ZubanLspBackend::onLspRenameReceived(int requestId, const QJsonObject& workspaceEdit)
{
    if (!m_pendingRequests.contains(requestId)) {
        return;
    }

    RequestContext context = m_pendingRequests.take(requestId);
    QList<JediRename> renames = convertWorkspaceEdit(workspaceEdit);

    emit renameReady(context.jediRequestId, renames);
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
JediCalltip ZubanLspBackend::convertSignatureHelp(const QJsonObject& signatureHelp) const
{
    // Convert LSP SignatureHelp to JediCalltip. The result must have the same format
    // as the one created by the jedi backend in itomJediLib.calltips: the parameters
    // contain their type annotations and the currently edited parameter is marked bold.
    const QJsonArray signatures = signatureHelp["signatures"].toArray();

    if (signatures.isEmpty()) {
        return JediCalltip();
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
    return JediCalltip(methodName.toHtmlEscaped(), params, 0, -1, 0);
}

//--------------------------------------------------------------------------------------
JediAssignment ZubanLspBackend::convertLocation(const QJsonObject& location) const
{
    // Convert LSP Location to JediAssignment
    QString uri = location["uri"].toString();
    QString path = uriToPath(uri);

    QJsonObject range = location["range"].toObject();
    QJsonObject start = range["start"].toObject();

    // JediAssignment::m_line is 0-based, like the LSP position
    int line = start["line"].toInt();
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
