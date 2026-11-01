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

#include "llmApplyService.h"

#include "../widgets/scriptEditorWidget.h"

#include <qcoreapplication.h>
#include <qtextcursor.h>

namespace ito
{
namespace llm
{

RetVal LlmApplyService::applySuggestion(
    ScriptEditorWidget *editor,
    const QString &suggestion,
    LlmApplyMode mode,
    LlmApplyResult *resultOut) const
{
    if (!editor)
    {
        return fail(QCoreApplication::translate("LlmApplyService", "No active script editor available."), resultOut);
    }

    if (editor->isReadOnly())
    {
        return fail(QCoreApplication::translate("LlmApplyService", "Active script is read-only."), resultOut);
    }

    if (suggestion.trimmed().isEmpty())
    {
        return fail(QCoreApplication::translate("LlmApplyService", "No suggestion text available to apply."), resultOut);
    }

    QTextCursor cursor = editor->textCursor();
    return applyWithCursor(editor, cursor, suggestion, mode, resultOut);
}

RetVal LlmApplyService::fail(const QString &message, LlmApplyResult *resultOut) const
{
    if (resultOut)
    {
        resultOut->applied = false;
        resultOut->message = message;
    }

    const QByteArray msg = message.toUtf8();
    return RetVal(retError, 0, msg.constData());
}

RetVal LlmApplyService::applyWithCursor(
    ScriptEditorWidget *editor,
    QTextCursor &cursor,
    const QString &suggestion,
    LlmApplyMode mode,
    LlmApplyResult *resultOut) const
{
    const bool hasSelection = cursor.hasSelection();

    if (mode == LlmApplyMode::ReplaceSelection && !hasSelection)
    {
        return fail(
            QCoreApplication::translate("LlmApplyService", "Replace selection requested, but no text is selected."),
            resultOut);
    }

    cursor.beginEditBlock();

    if (mode == LlmApplyMode::ReplaceSelection)
    {
        cursor.removeSelectedText();
    }

    cursor.insertText(suggestion);
    cursor.endEditBlock();

    editor->setTextCursor(cursor);

    if (resultOut)
    {
        resultOut->applied = true;
        resultOut->hadSelection = hasSelection;
        resultOut->message = QCoreApplication::translate("LlmApplyService", "Suggestion applied.");
    }

    return RetVal(retOk);
}

} // namespace llm
} // namespace ito
