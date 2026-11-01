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

#include "llmAssistantController.h"

namespace ito
{
namespace llm
{

LlmAssistantController::LlmAssistantController(QObject *parent) :
    QObject(parent),
    m_metricsCollector(NULL)
{
}

void LlmAssistantController::setProvider(const QSharedPointer<ILlmProvider> &provider)
{
    m_provider = provider;
}

QSharedPointer<ILlmProvider> LlmAssistantController::provider() const
{
    return m_provider;
}

void LlmAssistantController::setSettings(const LlmSettings &settings)
{
    m_settings = settings;
}

LlmSettings LlmAssistantController::settings() const
{
    return m_settings;
}

void LlmAssistantController::setMetricsCollector(LlmMetricsCollector *collector)
{
    m_metricsCollector = collector;
}

RetVal LlmAssistantController::runChat(
    LlmTaskType taskType,
    const QString &userPrompt,
    const ContextAssemblyInput &contextInput,
    LlmResponse &response,
    LlmError *errorOut)
{
    if (m_metricsCollector)
    {
        m_metricsCollector->onRequestStarted();
    }

    if (m_provider.isNull())
    {
        if (errorOut)
        {
            errorOut->category = LlmErrorCategory::Configuration;
            errorOut->userMessage = tr("No LLM provider configured.");
            errorOut->technicalMessage = QString("Provider pointer is null.");
        }

        if (m_metricsCollector)
        {
            m_metricsCollector->onError(QString("Configuration"));
            m_metricsCollector->onRequestFinished(false);
        }

        return RetVal(retError, 0, "No LLM provider configured.");
    }

    LlmRequest request = createRequest(taskType, userPrompt, contextInput);
    RetVal retval = m_securityPolicy.validateRequest(m_settings, request, errorOut);
    if (retval.containsError())
    {
        if (m_metricsCollector)
        {
            m_metricsCollector->onError(QString("Policy"));
            m_metricsCollector->onRequestFinished(false);
        }

        return retval;
    }

    response = m_provider->chat(request, errorOut);

    if (errorOut && !errorOut->userMessage.isEmpty())
    {
        if (m_metricsCollector)
        {
            m_metricsCollector->onError(QString("ProviderError"));
            m_metricsCollector->onRequestFinished(false);
        }

        const QByteArray msg = errorOut->userMessage.toUtf8();
        return RetVal(retError, 0, msg.constData());
    }

    if (m_metricsCollector)
    {
        m_metricsCollector->onRequestFinished(true);
    }

    return RetVal(retOk);
}

RetVal LlmAssistantController::runStreamingChat(
    LlmTaskType taskType,
    const QString &userPrompt,
    const ContextAssemblyInput &contextInput,
    StreamHandle &handle,
    LlmError *errorOut)
{
    if (m_metricsCollector)
    {
        m_metricsCollector->onRequestStarted();
    }

    if (m_provider.isNull())
    {
        if (errorOut)
        {
            errorOut->category = LlmErrorCategory::Configuration;
            errorOut->userMessage = tr("No LLM provider configured.");
            errorOut->technicalMessage = QString("Provider pointer is null.");
        }

        if (m_metricsCollector)
        {
            m_metricsCollector->onError(QString("Configuration"));
            m_metricsCollector->onRequestFinished(false);
        }

        return RetVal(retError, 0, "No LLM provider configured.");
    }

    m_streamBuffer.clear();

    LlmRequest request = createRequest(taskType, userPrompt, contextInput);
    RetVal retval = m_securityPolicy.validateRequest(m_settings, request, errorOut);
    if (retval.containsError())
    {
        if (m_metricsCollector)
        {
            m_metricsCollector->onError(QString("Policy"));
            m_metricsCollector->onRequestFinished(false);
        }

        return retval;
    }

    handle = m_provider->streamChat(request, this, errorOut);

    if (!handle.isValid())
    {
        if (m_metricsCollector)
        {
            m_metricsCollector->onError(QString("ProviderError"));
            m_metricsCollector->onRequestFinished(false);
        }

        const QByteArray msg = errorOut ? errorOut->userMessage.toUtf8() : QByteArray("Could not start stream.");
        return RetVal(retError, 0, msg.constData());
    }

    m_lastStreamHandle = handle;

    if (m_metricsCollector)
    {
        m_metricsCollector->onRequestFinished(true);
        m_metricsCollector->onStreamStarted(handle.value);
    }

    return RetVal(retOk);
}

void LlmAssistantController::cancel(const StreamHandle &handle)
{
    if (!m_provider.isNull() && handle.isValid())
    {
        m_provider->cancel(handle);

        if (m_metricsCollector)
        {
            m_metricsCollector->onStreamFinished(handle.value, true);
        }

        if (m_lastStreamHandle.value == handle.value)
        {
            m_lastStreamHandle = StreamHandle();
        }
    }
}

void LlmAssistantController::onChunk(const LlmChunk &chunk)
{
    m_streamBuffer += chunk.textDelta;

    if (m_metricsCollector)
    {
        m_metricsCollector->onStreamChunk();
    }

    emit streamingChunkReceived(chunk.textDelta, chunk.sequenceNo, chunk.isFinal);
}

void LlmAssistantController::onCompleted(const LlmResponse &finalResponse)
{
    if (!finalResponse.content.isEmpty())
    {
        m_streamBuffer = finalResponse.content;
    }

    if (m_metricsCollector && m_lastStreamHandle.isValid())
    {
        m_metricsCollector->onStreamFinished(m_lastStreamHandle.value, false);
    }

    m_lastStreamHandle = StreamHandle();

    emit streamingFinished(m_streamBuffer, finalResponse.finishReason);
}

void LlmAssistantController::onError(const LlmError &error)
{
    if (m_metricsCollector)
    {
        QString category;
        switch (error.category)
        {
        case LlmErrorCategory::Auth: category = QString("Auth"); break;
        case LlmErrorCategory::Network: category = QString("Network"); break;
        case LlmErrorCategory::Timeout: category = QString("Timeout"); break;
        case LlmErrorCategory::Policy: category = QString("Policy"); break;
        case LlmErrorCategory::UserCancel: category = QString("UserCancel"); break;
        case LlmErrorCategory::Configuration: category = QString("Configuration"); break;
        case LlmErrorCategory::ProviderError:
        default:
            category = QString("ProviderError");
            break;
        }

        m_metricsCollector->onError(category);

        if (m_lastStreamHandle.isValid())
        {
            m_metricsCollector->onStreamFinished(m_lastStreamHandle.value, false);
        }
    }

    m_lastStreamHandle = StreamHandle();

    emit streamingFailed(error.userMessage, error.technicalMessage);
}

LlmRequest LlmAssistantController::createRequest(
    LlmTaskType taskType,
    const QString &userPrompt,
    const ContextAssemblyInput &contextInput) const
{
    const ContextEnvelope context = m_contextAssemblyService.build(contextInput);

    LlmRequest request;
    request.mode = m_settings.mode;
    request.provider = m_settings.provider;
    request.model = m_settings.model;
    request.endpoint = m_settings.endpoint;
    request.apiKey = m_settings.apiKey;
    request.timeoutMs = m_settings.timeoutMs;
    request.taskType = taskType;

    request.contextFilePaths = context.referencedFilePaths;
    if (!context.activeFilePath.isEmpty())
    {
        request.contextFilePaths.prepend(context.activeFilePath);
    }

    request.messages = m_promptPipeline.buildMessages(taskType, context, userPrompt);

    return request;
}

} // namespace llm
} // namespace ito
