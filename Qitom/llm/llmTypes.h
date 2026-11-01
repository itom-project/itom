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

#include <QString>
#include <QStringList>

namespace ito
{
namespace llm
{

enum class LlmMode
{
    Disabled,
    Cloud,
    Offline
};

enum class LlmTaskType
{
    Explain,
    Generate,
    Fix
};

enum class LlmErrorCategory
{
    Auth,
    Network,
    Timeout,
    Policy,
    UserCancel,
    ProviderError,
    Configuration
};

struct LlmMessage
{
    QString role;
    QString content;
};

struct LlmRequest
{
    LlmMode mode = LlmMode::Disabled;
    QString provider;
    QString model;
    QString endpoint;
    QString apiKey;
    int timeoutMs = 0;
    LlmTaskType taskType = LlmTaskType::Generate;
    QStringList contextFilePaths;
    QList<LlmMessage> messages;
};

struct LlmChunk
{
    int sequenceNo = 0;
    QString textDelta;
    bool isFinal = false;
    QString finishReason;
};

struct LlmUsage
{
    int promptTokens = 0;
    int completionTokens = 0;
    int totalTokens = 0;
};

struct LlmResponse
{
    QString content;
    QString finishReason;
    bool hasUsage = false;
    LlmUsage usage;
};

struct LlmError
{
    LlmErrorCategory category = LlmErrorCategory::ProviderError;
    QString userMessage;
    QString technicalMessage;
};

struct StreamHandle
{
    quint64 value = 0;

    bool isValid() const
    {
        return value != 0;
    }
};

struct ProviderCapabilities
{
    bool supportsStreaming = false;
    bool supportsCancellation = false;
};

struct ProviderHealth
{
    bool available = false;
    QString statusMessage;
};

} // namespace llm
} // namespace ito
