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

#include "knowledgeContextService.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QTextStream>

#include <algorithm>

namespace ito
{
namespace llm
{

struct ScoredKnowledgeFile
{
    ContextFileContent file;
    int score = 0;
};

QList<ContextFileContent> KnowledgeContextService::collectKnowledgeFiles(
    const LlmSettings &settings,
    const QString &queryText,
    int maxFiles,
    int maxCharsPerFile) const
{
    QList<ContextFileContent> combined;

    const QStringList terms = tokenizeQuery(queryText);

    if (settings.stubsContextEnabled && !settings.stubsPath.trimmed().isEmpty())
    {
        combined << collectFromRoot(
            settings.stubsPath,
            QStringList() << "*.py" << "*.pyi",
            terms,
            qMax(1, maxFiles / 2),
            maxCharsPerFile,
            QString("stub"));
    }

    if (settings.docsContextEnabled && !settings.docsPath.trimmed().isEmpty())
    {
        combined << collectFromRoot(
            settings.docsPath,
            QStringList() << "*.rst" << "*.md" << "*.txt",
            terms,
            qMax(1, maxFiles / 2),
            maxCharsPerFile,
            QString("doc"));
    }

    while (combined.size() > maxFiles)
    {
        combined.removeLast();
    }

    return combined;
}

QList<ContextFileContent> KnowledgeContextService::collectFromRoot(
    const QString &rootPath,
    const QStringList &nameFilters,
    const QStringList &queryTerms,
    int maxFiles,
    int maxCharsPerFile,
    const QString &prefixLabel) const
{
    QList<ContextFileContent> result;

    QDir root(rootPath);
    if (!root.exists())
    {
        return result;
    }

    QList<ScoredKnowledgeFile> scored;

    QDirIterator it(
        root.absolutePath(),
        nameFilters,
        QDir::Files,
        QDirIterator::Subdirectories);

    while (it.hasNext())
    {
        const QString filePath = it.next();

        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            continue;
        }

        QTextStream stream(&file);
        QString content = stream.readAll();

        if (content.size() > maxCharsPerFile)
        {
            content = content.left(maxCharsPerFile) + QString("\n...[truncated]...");
        }

        ScoredKnowledgeFile entry;
        entry.file.filePath = QString("%1:%2").arg(prefixLabel, filePath);
        entry.file.content = content;
        entry.score = scoreText(content, queryTerms);

        if (entry.score > 0 || queryTerms.isEmpty())
        {
            scored << entry;
        }
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredKnowledgeFile &a, const ScoredKnowledgeFile &b) {
        return a.score > b.score;
    });

    for (int i = 0; i < scored.size() && result.size() < maxFiles; ++i)
    {
        result << scored[i].file;
    }

    return result;
}

QStringList KnowledgeContextService::tokenizeQuery(const QString &queryText) const
{
    QString simplified = queryText.toLower();
    simplified.replace(QRegularExpression("[^a-z0-9_]+"), QString(" "));

    QStringList rawTerms = simplified.split(' ', Qt::SkipEmptyParts);
    QStringList terms;

    for (int i = 0; i < rawTerms.size(); ++i)
    {
        const QString t = rawTerms[i].trimmed();
        if (t.size() >= 3)
        {
            terms << t;
        }
    }

    terms.removeDuplicates();
    return terms;
}

int KnowledgeContextService::scoreText(const QString &text, const QStringList &terms) const
{
    if (terms.isEmpty())
    {
        return 1;
    }

    const QString haystack = text.toLower();
    int score = 0;

    for (int i = 0; i < terms.size(); ++i)
    {
        const QString &term = terms[i];
        if (haystack.contains(term))
        {
            score += 1;
        }
    }

    return score;
}

} // namespace llm
} // namespace ito
