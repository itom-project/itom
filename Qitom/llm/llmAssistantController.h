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

#pragma once

#include "contextAssemblyService.h"
#include "iLlmProvider.h"
#include "llmMetricsCollector.h"
#include "llmSecurityPolicy.h"
#include "llmSettings.h"
#include "promptPipeline.h"

#include "../../common/retVal.h"

#include <QObject>
#include <QSharedPointer>

namespace ito
{
namespace llm
{

class LlmAssistantController : public QObject, public IStreamObserver
{
    Q_OBJECT

    public:
        explicit LlmAssistantController(QObject *parent = NULL);

        void setProvider(const QSharedPointer<ILlmProvider> &provider);
        QSharedPointer<ILlmProvider> provider() const;

        void setSettings(const LlmSettings &settings);
        LlmSettings settings() const;

        void setMetricsCollector(LlmMetricsCollector *collector);

        RetVal runChat(
            LlmTaskType taskType,
            const QString &userPrompt,
            const ContextAssemblyInput &contextInput,
            LlmResponse &response,
            LlmError *errorOut = NULL);

        RetVal runStreamingChat(
            LlmTaskType taskType,
            const QString &userPrompt,
            const ContextAssemblyInput &contextInput,
            StreamHandle &handle,
            LlmError *errorOut = NULL);

        void cancel(const StreamHandle &handle);

        virtual void onChunk(const LlmChunk &chunk);
        virtual void onCompleted(const LlmResponse &finalResponse);
        virtual void onError(const LlmError &error);

    signals:
        void streamingChunkReceived(const QString &textDelta, int sequenceNo, bool isFinal);
        void streamingFinished(const QString &fullText, const QString &finishReason);
        void streamingFailed(const QString &userMessage, const QString &technicalMessage);

    private:
        LlmRequest createRequest(
            LlmTaskType taskType,
            const QString &userPrompt,
            const ContextAssemblyInput &contextInput) const;

        QSharedPointer<ILlmProvider> m_provider;
        ContextAssemblyService m_contextAssemblyService;
        PromptPipeline m_promptPipeline;
        LlmSecurityPolicy m_securityPolicy;
        LlmSettings m_settings;
        LlmMetricsCollector *m_metricsCollector;

        QString m_streamBuffer;
        StreamHandle m_lastStreamHandle;
};

} // namespace llm
} // namespace ito
