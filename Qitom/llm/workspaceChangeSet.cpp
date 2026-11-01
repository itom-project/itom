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

#include "workspaceChangeSet.h"

#include <QRegularExpression>

namespace ito
{
namespace llm
{

WorkspaceChangeSet WorkspaceChangeSetParser::parse(const QString &assistantText) const
{
    WorkspaceChangeSet changeSet;

    QRegularExpression blockRegex(
        "<<<LLM_CHANGE\\s+(create|update|delete)\\s+path=([^>]+)>>>\\s*([\\s\\S]*?)\\s*<<<END_LLM_CHANGE>>>",
        QRegularExpression::CaseInsensitiveOption);

    QRegularExpressionMatchIterator it = blockRegex.globalMatch(assistantText);

    while (it.hasNext())
    {
        const QRegularExpressionMatch m = it.next();

        WorkspaceChange change;
        const QString op = m.captured(1).trimmed().toLower();
        change.filePath = m.captured(2).trimmed();
        change.content = m.captured(3);

        if (op == "create")
        {
            change.type = WorkspaceChangeType::CreateFile;
        }
        else if (op == "delete")
        {
            change.type = WorkspaceChangeType::DeleteFile;
            change.content.clear();
        }
        else
        {
            change.type = WorkspaceChangeType::UpdateFile;
        }

        if (!change.filePath.isEmpty())
        {
            changeSet.changes << change;
        }
    }

    return changeSet;
}

} // namespace llm
} // namespace ito
