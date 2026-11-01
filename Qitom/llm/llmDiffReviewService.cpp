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

#include "llmDiffReviewService.h"

#include "../widgets/scriptEditorWidget.h"
#include "../codeEditor/textDecoration.h"
#include "../codeEditor/managers/textDecorationsManager.h"

#include <QBrush>
#include <QColor>
#include <QFile>
#include <QFileInfo>
#include <QTextDocument>
#include <QTextStream>

namespace ito
{
namespace llm
{

QList<DiffReviewItem> LlmDiffReviewService::preview(
    ScriptEditorWidget *editor,
    const WorkspaceChangeSet &changeSet) const
{
    QList<DiffReviewItem> items;

    if (!editor)
    {
        return items;
    }

    const QString activeFile = QFileInfo(editor->getFilename()).absoluteFilePath();

    for (int i = 0; i < changeSet.changes.size(); ++i)
    {
        const WorkspaceChange &change = changeSet.changes[i];
        const QString target = QFileInfo(change.filePath).absoluteFilePath();
        if (activeFile.isEmpty() || target != activeFile)
        {
            continue;
        }

        QString oldText = editor->toPlainText();
        QString newText = (change.type == WorkspaceChangeType::DeleteFile) ? QString() : change.content;

        DiffReviewItem item;
        item.change = change;
        item.decorations = addLineHighlights(editor, oldText, newText);
        items << item;
    }

    return items;
}

void LlmDiffReviewService::clearPreview(
    ScriptEditorWidget *editor,
    QList<DiffReviewItem> &items) const
{
    if (!editor)
    {
        items.clear();
        return;
    }

    for (int i = 0; i < items.size(); ++i)
    {
        for (int j = 0; j < items[i].decorations.size(); ++j)
        {
            editor->decorations()->remove(items[i].decorations[j]);
        }
    }

    items.clear();
}

QList<QSharedPointer<TextDecoration> > LlmDiffReviewService::addLineHighlights(
    ScriptEditorWidget *editor,
    const QString &oldText,
    const QString &newText) const
{
    QList<QSharedPointer<TextDecoration> > decorations;

    if (!editor)
    {
        return decorations;
    }

    const QStringList oldLines = oldText.split('\n');
    const QStringList newLines = newText.split('\n');
    const int maxLines = qMax(oldLines.size(), newLines.size());

    for (int i = 0; i < maxLines; ++i)
    {
        const QString oldLine = (i < oldLines.size()) ? oldLines[i] : QString();
        const QString newLine = (i < newLines.size()) ? newLines[i] : QString();

        if (oldLine == newLine)
        {
            continue;
        }

        const int lineNumber = qMin(i, qMax(0, editor->lineCount() - 1));
        QSharedPointer<TextDecoration> deco(new TextDecoration(editor->document(), -1, -1, lineNumber, lineNumber, 200, "LLM diff", true));

        if (oldLine.isEmpty() && !newLine.isEmpty())
        {
            deco->setBackground(QBrush(QColor(210, 255, 210)));
        }
        else if (!oldLine.isEmpty() && newLine.isEmpty())
        {
            deco->setBackground(QBrush(QColor(255, 220, 220)));
        }
        else
        {
            deco->setBackground(QBrush(QColor(255, 245, 200)));
        }

        editor->decorations()->append(deco);
        decorations << deco;
    }

    return decorations;
}

} // namespace llm
} // namespace ito
