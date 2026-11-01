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

#include "contextAssemblyService.h"

namespace ito
{
namespace llm
{

ContextEnvelope ContextAssemblyService::build(const ContextAssemblyInput &input) const
{
    const int maxSelectedChars = 4000;
    const int maxActiveFileChars = 6000;
    const int maxAdditionalFileChars = 2000;
    const int maxTotalChars = 14000;

    ContextEnvelope envelope;

    bool selectedTruncated = false;
    bool activeTruncated = false;

    envelope.selectedText = truncateText(input.selectedText, maxSelectedChars, selectedTruncated);
    envelope.activeFilePath = input.activeFilePath;
    envelope.activeFileContent =
        truncateText(input.activeFileContent, maxActiveFileChars, activeTruncated);

    QStringList truncationNotes;

    if (selectedTruncated)
    {
        truncationNotes << QString("selectedText truncated to %1 chars").arg(maxSelectedChars);
    }

    if (activeTruncated)
    {
        truncationNotes << QString("activeFileContent truncated to %1 chars").arg(maxActiveFileChars);
    }

    int runningChars = envelope.selectedText.size() + envelope.activeFileContent.size();

    for (int i = 0; i < input.additionalFiles.size(); ++i)
    {
        const ContextFileContent &additional = input.additionalFiles[i];

        if (runningChars >= maxTotalChars)
        {
            truncationNotes << QString("additional files omitted due to total context limit");
            break;
        }

        bool truncated = false;
        ContextFileContent block;
        block.filePath = additional.filePath;
        block.content = truncateText(additional.content, maxAdditionalFileChars, truncated);

        const int remaining = maxTotalChars - runningChars;
        if (block.content.size() > remaining)
        {
            bool totalTruncated = false;
            block.content = truncateText(block.content, remaining, totalTruncated);
            truncated = truncated || totalTruncated;
        }

        if (block.content.isEmpty())
        {
            continue;
        }

        envelope.additionalFiles << block;
        envelope.referencedFilePaths << block.filePath;
        runningChars += block.content.size();

        if (truncated)
        {
            truncationNotes << QString("additional file '%1' truncated").arg(block.filePath);
        }
    }

    envelope.estimatedTokenCount = estimateTokenCount(envelope);
    envelope.truncationReport = truncationNotes.join("; ");

    return envelope;
}

QString ContextAssemblyService::truncateText(const QString &text, int maxChars, bool &truncated) const
{
    truncated = false;

    if (maxChars <= 0)
    {
        truncated = !text.isEmpty();
        return QString();
    }

    if (text.size() <= maxChars)
    {
        return text;
    }

    truncated = true;

    const QString marker("\n...[truncated]...");
    if (maxChars <= marker.size())
    {
        return text.left(maxChars);
    }

    return text.left(maxChars - marker.size()) + marker;
}

int ContextAssemblyService::estimateTokenCount(const ContextEnvelope &envelope) const
{
    int totalChars = envelope.selectedText.size() + envelope.activeFileContent.size();

    for (int i = 0; i < envelope.additionalFiles.size(); ++i)
    {
        totalChars += envelope.additionalFiles[i].content.size();
    }

    // rough estimate, to be replaced by model-aware tokenization in a follow-up step
    return totalChars / 4;
}

} // namespace llm
} // namespace ito
