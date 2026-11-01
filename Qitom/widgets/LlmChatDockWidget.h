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

#ifndef LLMCHATDOCKWIDGET_H
#define LLMCHATDOCKWIDGET_H

#include "abstractDockWidget.h"
#include "../llm/knowledgeContextService.h"
#include "../llm/llmApplyService.h"
#include "../llm/llmAssistantController.h"
#include "../llm/llmDiffReviewService.h"
#include "../llm/llmSecurityPolicy.h"
#include "../llm/iLlmProvider.h"
#include "../llm/workspaceApplyService.h"
#include "../llm/workspaceChangeSet.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qpointer.h>

class QLabel;
class QComboBox;
class QPlainTextEdit;
class QPushButton;
class QTextEdit;
class QWidget;

namespace ito
{
    class LlmChatDockWidget : public AbstractDockWidget
    {
        Q_OBJECT

        public:
            LlmChatDockWidget(
                const QString &title,
                const QString &objName,
                QWidget *parent = NULL,
                bool docked = true,
                bool isDockAvailable = true,
                tFloatingStyle floatingStyle = floatingNone,
                tMovingStyle movingStyle = movingEnabled);
            ~LlmChatDockWidget();

            void setLlmSettings(const llm::LlmSettings &settings);
            void setLlmProvider(const QSharedPointer<llm::ILlmProvider> &provider);

        protected:
            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){}
            void updateActions();
            void updatePythonActions(){ updateActions(); }

        private:
            void setupUi();
            llm::LlmTaskType currentTaskType() const;
            llm::ContextAssemblyInput createContextInput() const;
            void appendSystemMessage(const QString &message);
            void appendUserMessage(const QString &message);
            void appendAssistantMessage(const QString &message, bool replaceLast = false);
            void refreshModeIndicator();

            QWidget *m_pContentWidget;
            QPointer<QLabel> m_pModeLabel;
            QPointer<QLabel> m_pSessionInfoLabel;
            QPointer<QComboBox> m_pTaskCombo;
            QPointer<QTextEdit> m_pConversationView;
            QPointer<QPlainTextEdit> m_pPromptEdit;
            QPointer<QPushButton> m_pSendButton;
            QPointer<QPushButton> m_pStopButton;
            QPointer<QPushButton> m_pApplyButton;
            QPointer<QPushButton> m_pPatchButton;

            QAction *m_pActSend;
            QAction *m_pActStop;
            QAction *m_pActApply;
            QAction *m_pActShowPatch;
            QToolBar *m_pMainToolbar;

            llm::LlmAssistantController m_controller;
            llm::LlmApplyService m_applyService;
            llm::LlmMetricsCollector m_metricsCollector;
            llm::LlmSecurityPolicy m_securityPolicy;
            llm::WorkspaceApplyService m_workspaceApplyService;
            llm::WorkspaceChangeSetParser m_workspaceChangeSetParser;
            llm::LlmDiffReviewService m_diffReviewService;
            llm::KnowledgeContextService m_knowledgeContextService;
            llm::LlmSettings m_settings;
            llm::StreamHandle m_currentStreamHandle;
            QList<llm::DiffReviewItem> m_pendingDiffReviewItems;
            llm::WorkspaceChangeSet m_pendingChangeSet;
            bool m_autoApplySession;
            QString m_streamingAssistantText;
            QString m_latestAssistantSuggestion;

        private slots:
            void sendRequest();
            void stopRequest();
            void applySuggestion();
            void showPatchPreview();
            void toggleAutoApplySession();
            void acceptPendingChanges();
            void rejectPendingChanges();
            void reloadSettings();
            void onStreamingChunkReceived(const QString &textDelta, int sequenceNo, bool isFinal);
            void onStreamingFinished(const QString &fullText, const QString &finishReason);
            void onStreamingFailed(const QString &userMessage, const QString &technicalMessage);
    };

} // namespace ito

#endif
