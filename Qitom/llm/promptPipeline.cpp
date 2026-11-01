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

#include "promptPipeline.h"

namespace ito
{
namespace llm
{

QList<LlmMessage> PromptPipeline::buildMessages(
    LlmTaskType taskType,
    const ContextEnvelope &context,
    const QString &userPrompt) const
{
    QList<LlmMessage> messages;

    messages << LlmMessage{ "system", QString("You are an assistant for itom Python scripting.") };
    messages << LlmMessage{ "system", taskInstruction(taskType) };

    if (!context.selectedText.isEmpty())
    {
        messages << LlmMessage{ "system", QString("Selected code block:\n%1").arg(context.selectedText) };
    }

    if (!context.activeFileContent.isEmpty())
    {
        const QString path = context.activeFilePath.isEmpty() ? QString("<active-script>") : context.activeFilePath;
        messages << LlmMessage{
            "system",
            QString("Active file context (%1):\n%2").arg(path, context.activeFileContent) };
    }

    for (int i = 0; i < context.additionalFiles.size(); ++i)
    {
        const ContextFileContent &file = context.additionalFiles[i];
        const QString path = file.filePath.isEmpty() ? QString("<unnamed>") : file.filePath;
        messages << LlmMessage{
            "system",
            QString("Additional file context (%1):\n%2").arg(path, file.content) };
    }

    if (!context.truncationReport.isEmpty())
    {
        messages << LlmMessage{ "system", QString("Context truncation report: %1").arg(context.truncationReport) };
    }

    messages << LlmMessage{ "system", QString("Estimated context tokens: %1").arg(context.estimatedTokenCount) };
    messages << LlmMessage{ "user", userPrompt };

    return messages;
}

QString PromptPipeline::taskInstruction(LlmTaskType taskType) const
{
    switch (taskType)
    {
    case LlmTaskType::Explain:
        return QString("Explain the behavior and root causes based on the provided context.");
    case LlmTaskType::Fix:
        return QString("Suggest a minimal fix and explain side effects.");
    case LlmTaskType::Generate:
    default:
        return QString("Generate Python code for itom based on the provided context.");
    }
}

} // namespace llm
} // namespace ito
