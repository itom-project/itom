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

#include "llmSecurityPolicy.h"

namespace ito
{
namespace llm
{

RetVal LlmSecurityPolicy::validateRequest(
    const LlmSettings &settings,
    const LlmRequest &request,
    LlmError *errorOut) const
{
    if (!settings.enabled || settings.mode == LlmMode::Disabled)
    {
        return fail(
            LlmErrorCategory::Policy,
            QString("LLM support is disabled."),
            QString("Request blocked by llm.enabled=false or mode=disabled."),
            errorOut);
    }

    if (request.mode != settings.mode)
    {
        return fail(
            LlmErrorCategory::Policy,
            QString("LLM mode mismatch."),
            QString("Request mode differs from active settings mode."),
            errorOut);
    }

    if (settings.mode == LlmMode::Cloud)
    {
        if (settings.provider.isEmpty() || settings.model.isEmpty() || settings.endpoint.isEmpty() ||
            settings.apiKey.isEmpty())
        {
            return fail(
                LlmErrorCategory::Configuration,
                QString("Cloud configuration is incomplete."),
                QString("provider/model/endpoint/apiKey must be set for cloud mode."),
                errorOut);
        }

        if (settings.restrictToKnownModels && !m_knownModels.isEmpty() &&
            !m_knownModels.contains(settings.model, Qt::CaseSensitive))
        {
            return fail(
                LlmErrorCategory::Policy,
                QString("Selected model is not allowed for this startup profile."),
                QString("Model '%1' is not in known model set.").arg(settings.model),
                errorOut);
        }
    }

    if (settings.mode == LlmMode::Offline)
    {
        if (settings.endpoint.isEmpty())
        {
            return fail(
                LlmErrorCategory::Configuration,
                QString("Offline endpoint is missing."),
                QString("endpoint must be set for offline mode."),
                errorOut);
        }
    }

    return RetVal(retOk);
}

WorkspacePolicyDecision LlmSecurityPolicy::evaluateWorkspaceChangeSet(
    const WorkspaceChangeSet &changeSet,
    bool autoApplyRequested) const
{
    WorkspacePolicyDecision decision;

    if (changeSet.isEmpty())
    {
        decision.allowed = false;
        decision.requiresConfirmation = false;
        decision.reason = QString("Empty change set.");
        return decision;
    }

    int deleteCount = 0;
    for (int i = 0; i < changeSet.changes.size(); ++i)
    {
        if (changeSet.changes[i].type == WorkspaceChangeType::DeleteFile)
        {
            deleteCount += 1;
        }
    }

    const bool bulkChange = changeSet.changes.size() > 8;
    const bool riskyChange = (deleteCount > 0) || bulkChange;

    decision.allowed = true;
    decision.requiresConfirmation = riskyChange;

    if (deleteCount > 0)
    {
        decision.reason = QString("Change set contains delete operations.");
    }
    else if (bulkChange)
    {
        decision.reason = QString("Change set contains many file operations.");
    }
    else
    {
        decision.reason = QString("Change set is low risk.");
    }

    if (autoApplyRequested && riskyChange)
    {
        decision.allowed = false;
        decision.requiresConfirmation = true;
        decision.reason = QString("Auto-apply blocked for high-risk change set.");
    }

    return decision;
}

QString LlmSecurityPolicy::maskSensitive(const QString &value) const
{
    if (value.length() <= 4)
    {
        return QString("****");
    }

    return value.left(2) + QString("****") + value.right(2);
}

void LlmSecurityPolicy::setKnownModels(const QStringList &models)
{
    m_knownModels = models;
}

RetVal LlmSecurityPolicy::fail(
    LlmErrorCategory category,
    const QString &userMessage,
    const QString &technicalMessage,
    LlmError *errorOut) const
{
    if (errorOut)
    {
        errorOut->category = category;
        errorOut->userMessage = userMessage;
        errorOut->technicalMessage = technicalMessage;
    }

    const QByteArray msg = userMessage.toUtf8();
    return RetVal(retError, 0, msg.constData());
}

} // namespace llm
} // namespace ito
