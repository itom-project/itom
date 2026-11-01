/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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

#include "LlmChatDockWidget.h"

#include "../AppManagement.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "scriptDockWidget.h"
#include "scriptEditorWidget.h"
#include "../llm/llmCredentialService.h"
#include "../llm/llmProviderFactory.h"
#include "../llm/llmSettingsManager.h"
#include "../llm/projectDiscoveryService.h"

#include <qtextcursor.h>
#include <qfile.h>
#include <qtextstream.h>
#include <qdir.h>
#include <qmessagebox.h>
#include <qlabel.h>
#include <qcombobox.h>
#include <qplaintextedit.h>
#include <qpushbutton.h>
#include <qtextedit.h>
#include <qboxlayout.h>
#include <qmenu.h>

namespace ito
{

LlmChatDockWidget::LlmChatDockWidget(
    const QString &title,
    const QString &objName,
    QWidget *parent,
    bool docked,
    bool isDockAvailable,
    tFloatingStyle floatingStyle,
    tMovingStyle movingStyle) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_pContentWidget(NULL),
    m_pActSend(NULL),
    m_pActStop(NULL),
    m_pActApply(NULL),
    m_pActShowPatch(NULL),
    m_pMainToolbar(NULL),
    m_autoApplySession(false)
{
    setupUi();

    connect(
        &m_controller,
        SIGNAL(streamingChunkReceived(QString,int,bool)),
        this,
        SLOT(onStreamingChunkReceived(QString,int,bool)));
    connect(
        &m_controller,
        SIGNAL(streamingFinished(QString,QString)),
        this,
        SLOT(onStreamingFinished(QString,QString)));
    connect(
        &m_controller,
        SIGNAL(streamingFailed(QString,QString)),
        this,
        SLOT(onStreamingFailed(QString,QString)));

    AbstractDockWidget::init();
    setContentWidget(m_pContentWidget);

    m_controller.setMetricsCollector(&m_metricsCollector);

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(reloadSettings()));

    refreshModeIndicator();
    updateActions();
}

LlmChatDockWidget::~LlmChatDockWidget()
{
    disconnect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(reloadSettings()));
}

void LlmChatDockWidget::setLlmSettings(const llm::LlmSettings &settings)
{
    m_settings = settings;
    m_controller.setSettings(settings);
    refreshModeIndicator();
    updateActions();
}

void LlmChatDockWidget::setLlmProvider(const QSharedPointer<llm::ILlmProvider> &provider)
{
    m_controller.setProvider(provider);
}

void LlmChatDockWidget::createActions()
{
    m_pActSend = new QAction(tr("Send"), this);
    connect(m_pActSend, SIGNAL(triggered()), this, SLOT(sendRequest()));

    m_pActStop = new QAction(tr("Stop"), this);
    connect(m_pActStop, SIGNAL(triggered()), this, SLOT(stopRequest()));

    m_pActApply = new QAction(tr("Apply"), this);
    connect(m_pActApply, SIGNAL(triggered()), this, SLOT(applySuggestion()));

    m_pActShowPatch = new QAction(tr("Show Patch"), this);
    connect(m_pActShowPatch, SIGNAL(triggered()), this, SLOT(showPatchPreview()));

    QAction *actToggleAutoApply = new QAction(tr("Auto Apply (Session)"), this);
    actToggleAutoApply->setCheckable(true);
    connect(actToggleAutoApply, SIGNAL(triggered()), this, SLOT(toggleAutoApplySession()));

    QAction *actAcceptAll = new QAction(tr("Accept All Pending"), this);
    connect(actAcceptAll, SIGNAL(triggered()), this, SLOT(acceptPendingChanges()));

    QAction *actRejectAll = new QAction(tr("Reject All Pending"), this);
    connect(actRejectAll, SIGNAL(triggered()), this, SLOT(rejectPendingChanges()));

    addAction(actToggleAutoApply);
    addAction(actAcceptAll);
    addAction(actRejectAll);
}

void LlmChatDockWidget::createMenus()
{
    QMenu *winMenu = getMenuBar()->addMenu(tr("&Windows"));
    if (m_actStayOnTop)
    {
        winMenu->addAction(m_actStayOnTop);
    }
    if (m_actStayOnTopOfApp)
    {
        winMenu->addAction(m_actStayOnTopOfApp);
    }
}

void LlmChatDockWidget::createToolBars()
{
    m_pMainToolbar = new QToolBar(tr("LLM Chat"), this);
    m_pMainToolbar->setObjectName("toolbarLlmChat");
    m_pMainToolbar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolbar->setFloatable(false);
    m_pMainToolbar->addAction(m_pActSend);
    m_pMainToolbar->addAction(m_pActStop);
    m_pMainToolbar->addSeparator();
    m_pMainToolbar->addAction(m_pActApply);
    m_pMainToolbar->addAction(m_pActShowPatch);

    addToolBar(m_pMainToolbar, "llmChatToolbar");
}

void LlmChatDockWidget::updateActions()
{
    const bool llmEnabled = m_settings.enabled && m_settings.mode != llm::LlmMode::Disabled;
    const bool isStreaming = m_currentStreamHandle.isValid();
    const bool hasSuggestion = !m_latestAssistantSuggestion.trimmed().isEmpty();

    const bool canSend = llmEnabled && !isStreaming;
    const bool canStop = llmEnabled && isStreaming;
    const bool canApply = !isStreaming && hasSuggestion;

    if (m_pActSend)
    {
        m_pActSend->setEnabled(canSend);
    }
    if (m_pSendButton)
    {
        m_pSendButton->setEnabled(canSend);
    }

    if (m_pActStop)
    {
        m_pActStop->setEnabled(canStop);
    }
    if (m_pStopButton)
    {
        m_pStopButton->setEnabled(canStop);
    }

    if (m_pActApply)
    {
        m_pActApply->setEnabled(canApply);
    }
    if (m_pApplyButton)
    {
        m_pApplyButton->setEnabled(canApply);
    }

    if (m_pActShowPatch)
    {
        m_pActShowPatch->setEnabled(canApply);
    }
    if (m_pPatchButton)
    {
        m_pPatchButton->setEnabled(canApply);
    }
}

void LlmChatDockWidget::setupUi()
{
    m_pContentWidget = new QWidget(this);

    QVBoxLayout *rootLayout = new QVBoxLayout(m_pContentWidget);
    rootLayout->setContentsMargins(4, 4, 4, 4);

    QHBoxLayout *headerLayout = new QHBoxLayout();
    m_pModeLabel = new QLabel(tr("Mode: Disabled"), m_pContentWidget);
    m_pSessionInfoLabel = new QLabel(tr("Session-only history"), m_pContentWidget);
    headerLayout->addWidget(m_pModeLabel);
    headerLayout->addStretch(1);
    headerLayout->addWidget(m_pSessionInfoLabel);

    QHBoxLayout *taskLayout = new QHBoxLayout();
    QLabel *taskLabel = new QLabel(tr("Action:"), m_pContentWidget);
    m_pTaskCombo = new QComboBox(m_pContentWidget);
    m_pTaskCombo->addItem(tr("Generate"));
    m_pTaskCombo->addItem(tr("Explain"));
    m_pTaskCombo->addItem(tr("Fix"));
    taskLayout->addWidget(taskLabel);
    taskLayout->addWidget(m_pTaskCombo, 1);

    m_pConversationView = new QTextEdit(m_pContentWidget);
    m_pConversationView->setReadOnly(true);

    m_pPromptEdit = new QPlainTextEdit(m_pContentWidget);
    m_pPromptEdit->setPlaceholderText(tr("Describe what you want to generate, explain, or fix..."));

    QHBoxLayout *buttonLayout = new QHBoxLayout();
    m_pSendButton = new QPushButton(tr("Send"), m_pContentWidget);
    m_pStopButton = new QPushButton(tr("Stop"), m_pContentWidget);
    m_pApplyButton = new QPushButton(tr("Apply"), m_pContentWidget);
    m_pPatchButton = new QPushButton(tr("Show Patch"), m_pContentWidget);
    buttonLayout->addStretch(1);
    buttonLayout->addWidget(m_pPatchButton);
    buttonLayout->addWidget(m_pApplyButton);
    buttonLayout->addWidget(m_pSendButton);
    buttonLayout->addWidget(m_pStopButton);

    connect(m_pSendButton, SIGNAL(clicked()), this, SLOT(sendRequest()));
    connect(m_pStopButton, SIGNAL(clicked()), this, SLOT(stopRequest()));
    connect(m_pApplyButton, SIGNAL(clicked()), this, SLOT(applySuggestion()));
    connect(m_pPatchButton, SIGNAL(clicked()), this, SLOT(showPatchPreview()));

    rootLayout->addLayout(headerLayout);
    rootLayout->addLayout(taskLayout);
    rootLayout->addWidget(m_pConversationView, 2);
    rootLayout->addWidget(m_pPromptEdit, 1);
    rootLayout->addLayout(buttonLayout);
}

llm::LlmTaskType LlmChatDockWidget::currentTaskType() const
{
    const int idx = m_pTaskCombo ? m_pTaskCombo->currentIndex() : 0;

    switch (idx)
    {
    case 1:
        return llm::LlmTaskType::Explain;
    case 2:
        return llm::LlmTaskType::Fix;
    case 0:
    default:
        return llm::LlmTaskType::Generate;
    }
}

llm::ContextAssemblyInput LlmChatDockWidget::createContextInput() const
{
    llm::ContextAssemblyInput input;

    ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
    if (!seo)
    {
        return input;
    }

    ScriptDockWidget *activeDock = seo->getActiveDockWidget();
    ScriptEditorWidget *activeEditor = activeDock ? activeDock->getCurrentEditor() : NULL;

    if (activeEditor)
    {
        QTextCursor cursor = activeEditor->textCursor();
        if (cursor.hasSelection())
        {
            input.selectedText = cursor.selectedText().replace(QChar(0x2029), '\n');
        }

        input.activeFilePath = activeEditor->getFilename();
        if (input.activeFilePath.isEmpty())
        {
            input.activeFilePath = activeEditor->getUntitledName();
        }

        input.activeFileContent = activeEditor->toPlainText();
    }

    const QStringList opened = seo->openedScripts();
    for (int i = 0; i < opened.size(); ++i)
    {
        const QString &file = opened[i];
        if (file.isEmpty() || file == input.activeFilePath)
        {
            continue;
        }

        ScriptEditorWidget *editor = seo->getEditorFromCanonicalFilepath(file);
        if (!editor)
        {
            continue;
        }

        llm::ContextFileContent additional;
        additional.filePath = file;
        additional.content = editor->toPlainText();

        input.additionalFiles << additional;
    }

    const QString workingDirectory = QDir::currentPath();
    llm::ProjectDiscoveryService discoveryService;
    llm::ProjectDiscoveryResult discovery =
        discoveryService.discover(input.activeFilePath, workingDirectory);

    for (int i = 0; i < discovery.filePaths.size(); ++i)
    {
        const QString &filePath = discovery.filePaths[i];
        if (filePath.isEmpty() || filePath == input.activeFilePath)
        {
            continue;
        }

        bool alreadyAdded = false;
        for (int j = 0; j < input.additionalFiles.size(); ++j)
        {
            if (input.additionalFiles[j].filePath == filePath)
            {
                alreadyAdded = true;
                break;
            }
        }

        if (alreadyAdded)
        {
            continue;
        }

        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            continue;
        }

        QTextStream stream(&file);

        llm::ContextFileContent additional;
        additional.filePath = filePath;
        additional.content = stream.readAll();
        input.additionalFiles << additional;
    }

    QString knowledgeQuery = input.selectedText;
    if (knowledgeQuery.isEmpty())
    {
        knowledgeQuery = input.activeFileContent;
    }

    QList<llm::ContextFileContent> knowledgeFiles =
        m_knowledgeContextService.collectKnowledgeFiles(m_settings, knowledgeQuery);

    for (int i = 0; i < knowledgeFiles.size(); ++i)
    {
        const llm::ContextFileContent &additional = knowledgeFiles[i];

        bool alreadyAdded = false;
        for (int j = 0; j < input.additionalFiles.size(); ++j)
        {
            if (input.additionalFiles[j].filePath == additional.filePath)
            {
                alreadyAdded = true;
                break;
            }
        }

        if (!alreadyAdded)
        {
            input.additionalFiles << additional;
        }
    }

    return input;
}

void LlmChatDockWidget::appendSystemMessage(const QString &message)
{
    if (m_pConversationView)
    {
        m_pConversationView->append(QString("<b>System:</b> %1").arg(message.toHtmlEscaped()));
    }
}

void LlmChatDockWidget::appendUserMessage(const QString &message)
{
    if (m_pConversationView)
    {
        m_pConversationView->append(QString("<b>User:</b><br/>%1")
                                        .arg(message.toHtmlEscaped().replace("\n", "<br/>")));
    }
}

void LlmChatDockWidget::appendAssistantMessage(const QString &message, bool replaceLast)
{
    if (!m_pConversationView)
    {
        return;
    }

    QString html = m_pConversationView->toHtml();

    if (replaceLast)
    {
        const int markerPos = html.lastIndexOf("<b>Assistant:</b>");
        if (markerPos >= 0)
        {
            html = html.left(markerPos);
            m_pConversationView->setHtml(html);
        }
    }

    m_pConversationView->append(QString("<b>Assistant:</b><br/>%1")
                                    .arg(message.toHtmlEscaped().replace("\n", "<br/>")));
}

void LlmChatDockWidget::refreshModeIndicator()
{
    if (!m_pModeLabel)
    {
        return;
    }

    QString modeText = tr("Disabled");

    if (m_settings.mode == llm::LlmMode::Cloud)
    {
        modeText = tr("Cloud");
    }
    else if (m_settings.mode == llm::LlmMode::Offline)
    {
        modeText = tr("Offline");
    }

    m_pModeLabel->setText(tr("Mode: %1").arg(modeText));
}

void LlmChatDockWidget::sendRequest()
{
    if (!m_settings.enabled || m_settings.mode == llm::LlmMode::Disabled)
    {
        appendSystemMessage(tr("LLM support is disabled."));
        return;
    }

    const QString prompt = m_pPromptEdit ? m_pPromptEdit->toPlainText().trimmed() : QString();
    if (prompt.isEmpty())
    {
        appendSystemMessage(tr("Please enter a prompt."));
        return;
    }

    const llm::ContextAssemblyInput context = createContextInput();

    appendUserMessage(prompt);
    appendSystemMessage(
        tr("Context: active file '%1', additional files: %2")
            .arg(context.activeFilePath)
            .arg(context.additionalFiles.size()));

    m_streamingAssistantText.clear();
    m_latestAssistantSuggestion.clear();
    appendAssistantMessage(tr("... streaming ..."), false);

    if (m_pActSend)
    {
        m_pActSend->setEnabled(false);
    }
    if (m_pSendButton)
    {
        m_pSendButton->setEnabled(false);
    }
    if (m_pActStop)
    {
        m_pActStop->setEnabled(true);
    }
    if (m_pStopButton)
    {
        m_pStopButton->setEnabled(true);
    }

    llm::LlmError error;
    m_currentStreamHandle = llm::StreamHandle();

    RetVal retval = m_controller.runStreamingChat(
        currentTaskType(),
        prompt,
        context,
        m_currentStreamHandle,
        &error);

    if (retval.containsError())
    {
        onStreamingFailed(error.userMessage, error.technicalMessage);
    }
}

void LlmChatDockWidget::stopRequest()
{
    if (m_currentStreamHandle.isValid())
    {
        m_controller.cancel(m_currentStreamHandle);
        m_currentStreamHandle = llm::StreamHandle();
        appendSystemMessage(tr("Request cancelled."));
    }

    updateActions();
}

void LlmChatDockWidget::applySuggestion()
{
    if (m_currentStreamHandle.isValid())
    {
        appendSystemMessage(tr("Cannot apply while streaming is active."));
        return;
    }

    if (m_latestAssistantSuggestion.trimmed().isEmpty())
    {
        appendSystemMessage(tr("No assistant suggestion available to apply."));
        return;
    }

    ScriptEditorOrganizer *seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
    ScriptDockWidget *activeDock = seo ? seo->getActiveDockWidget() : NULL;
    ScriptEditorWidget *activeEditor = activeDock ? activeDock->getCurrentEditor() : NULL;

    const llm::WorkspaceChangeSet changeSet =
        m_workspaceChangeSetParser.parse(m_latestAssistantSuggestion);

    if (!changeSet.isEmpty())
    {
        m_diffReviewService.clearPreview(activeEditor, m_pendingDiffReviewItems);
        m_pendingChangeSet = changeSet;

        const llm::WorkspacePolicyDecision decision =
            m_securityPolicy.evaluateWorkspaceChangeSet(changeSet, m_autoApplySession);

        if (!decision.allowed)
        {
            m_metricsCollector.onPolicyRejected();
            appendSystemMessage(tr("Policy gate: %1").arg(decision.reason));
            m_pendingDiffReviewItems = m_diffReviewService.preview(activeEditor, changeSet);
            appendSystemMessage(tr("Please review and confirm with 'Accept All Pending'."));
            return;
        }

        if (m_autoApplySession && !decision.requiresConfirmation)
        {
            acceptPendingChanges();
            return;
        }

        m_pendingDiffReviewItems = m_diffReviewService.preview(activeEditor, changeSet);

        if (decision.requiresConfirmation)
        {
            appendSystemMessage(tr("High-risk changes require explicit confirmation: %1").arg(decision.reason));
        }

        appendSystemMessage(
            tr("Diff review prepared for %1 changes. Use 'Accept All Pending' or 'Reject All Pending'.")
                .arg(changeSet.changes.size()));
        return;
    }

    llm::LlmApplyResult result;
    RetVal retval = m_applyService.applySuggestion(
        activeEditor,
        m_latestAssistantSuggestion,
        llm::LlmApplyMode::InsertAtCursor,
        &result);

    if (retval.containsError())
    {
        appendSystemMessage(result.message.isEmpty() ? tr("Apply failed.") : result.message);
        return;
    }

    appendSystemMessage(tr("Suggestion applied to active editor."));
}

void LlmChatDockWidget::showPatchPreview()
{
    if (m_latestAssistantSuggestion.trimmed().isEmpty())
    {
        appendSystemMessage(tr("No assistant suggestion available for patch preview."));
        return;
    }

    QMessageBox::information(
        this,
        tr("Patch Preview"),
        tr("Proposed code to apply:\n\n%1").arg(m_latestAssistantSuggestion));
}

void LlmChatDockWidget::toggleAutoApplySession()
{
    m_autoApplySession = !m_autoApplySession;
    appendSystemMessage(
        m_autoApplySession ? tr("Auto-apply session mode enabled.") : tr("Auto-apply session mode disabled."));
}

void LlmChatDockWidget::acceptPendingChanges()
{
    if (m_pendingChangeSet.isEmpty())
    {
        appendSystemMessage(tr("No pending change set."));
        return;
    }

    llm::WorkspaceApplyResult applyResult;
    RetVal ret = m_workspaceApplyService.apply(QDir::currentPath(), m_pendingChangeSet, &applyResult);

    if (ret.containsError())
    {
        appendSystemMessage(tr("Workspace apply reported errors."));
        for (int i = 0; i < applyResult.errors.size(); ++i)
        {
            appendSystemMessage(applyResult.errors[i]);
        }
    }
    else
    {
        m_metricsCollector.onWorkspaceApply(
            applyResult.createdFiles,
            applyResult.updatedFiles,
            applyResult.deletedFiles);
        appendSystemMessage(
            tr("Pending changes applied. Created: %1, Updated: %2, Deleted: %3")
                .arg(applyResult.createdFiles)
                .arg(applyResult.updatedFiles)
                .arg(applyResult.deletedFiles));
    }

    ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
    ScriptDockWidget *activeDock = seo ? seo->getActiveDockWidget() : NULL;
    ScriptEditorWidget *activeEditor = activeDock ? activeDock->getCurrentEditor() : NULL;
    m_diffReviewService.clearPreview(activeEditor, m_pendingDiffReviewItems);
    m_pendingChangeSet.changes.clear();
}

void LlmChatDockWidget::rejectPendingChanges()
{
    ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
    ScriptDockWidget *activeDock = seo ? seo->getActiveDockWidget() : NULL;
    ScriptEditorWidget *activeEditor = activeDock ? activeDock->getCurrentEditor() : NULL;

    m_diffReviewService.clearPreview(activeEditor, m_pendingDiffReviewItems);
    m_pendingChangeSet.changes.clear();
    appendSystemMessage(tr("Pending changes rejected."));
}

void LlmChatDockWidget::reloadSettings()
{
    llm::LlmSettings settings;
    llm::LlmSettingsManager::load(settings);

    llm::LlmCredentialService credentialService;
    QString apiKey;
    if (!credentialService.loadApiKey(settings, apiKey).containsError())
    {
        settings.apiKey = apiKey;
    }

    QSharedPointer<llm::ILlmProvider> provider;
    llm::LlmError error;
    RetVal retval = llm::LlmProviderFactory::createProvider(settings, provider, &error);

    setLlmProvider(provider);
    setLlmSettings(settings);

    if (retval.containsError())
    {
        const QString msg = error.userMessage.isEmpty() ? tr("LLM provider configuration is invalid.") : error.userMessage;
        appendSystemMessage(msg);
    }
}

void LlmChatDockWidget::onStreamingChunkReceived(const QString &textDelta, int, bool)
{
    m_streamingAssistantText += textDelta;
    m_latestAssistantSuggestion = m_streamingAssistantText;
    appendAssistantMessage(m_streamingAssistantText, true);
}

void LlmChatDockWidget::onStreamingFinished(const QString &fullText, const QString &finishReason)
{
    const QString finalText = fullText.isEmpty() ? m_streamingAssistantText : fullText;
    m_latestAssistantSuggestion = finalText;
    appendAssistantMessage(finalText, true);

    if (!finishReason.isEmpty())
    {
        appendSystemMessage(tr("Finish reason: %1").arg(finishReason));
    }

    m_streamingAssistantText.clear();
    m_currentStreamHandle = llm::StreamHandle();
    updateActions();
}

void LlmChatDockWidget::onStreamingFailed(const QString &userMessage, const QString &technicalMessage)
{
    const QString msg = userMessage.isEmpty() ? tr("Streaming failed.") : userMessage;
    appendSystemMessage(msg);

    if (!technicalMessage.isEmpty())
    {
        appendSystemMessage(technicalMessage);
    }

    m_streamingAssistantText.clear();
    m_currentStreamHandle = llm::StreamHandle();
    updateActions();
}

} // namespace ito
