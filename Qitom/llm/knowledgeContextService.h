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
#include "llmSettings.h"

#include <QStringList>

namespace ito
{
namespace llm
{

/*! \class ito::llm::KnowledgeContextService
    \brief Collects additional context snippets from configured knowledge roots.

    The service scans configured stubs and docs folders and returns a bounded
    list of relevant text snippets for prompt enrichment.

    \sa ContextFileContent, LlmSettings
*/
class KnowledgeContextService
{
    public:
        //! Collect context files relevant to a query text.
        /*! \param [in] settings Active LLM settings with knowledge root configuration.
            \param [in] queryText Query text used for lexical scoring.
            \param [in] maxFiles Maximum number of snippets to return.
            \param [in] maxCharsPerFile Maximum number of characters per snippet.
            \return List of collected context file snippets.
            \sa collectFromRoot
        */
        QList<ContextFileContent> collectKnowledgeFiles(
            const LlmSettings &settings,
            const QString &queryText,
            int maxFiles = 8,
            int maxCharsPerFile = 2000) const;

    private:
        //! Collect and score files from one configured root path.
        /*! \param [in] rootPath Root directory to scan.
            \param [in] nameFilters File-name filters for traversal.
            \param [in] queryTerms Tokenized query terms.
            \param [in] maxFiles Maximum number of snippets to return.
            \param [in] maxCharsPerFile Maximum number of characters per snippet.
            \param [in] prefixLabel Prefix to annotate origin in file path.
            \return List of collected context file snippets.
            \sa scoreText
        */
        QList<ContextFileContent> collectFromRoot(
            const QString &rootPath,
            const QStringList &nameFilters,
            const QStringList &queryTerms,
            int maxFiles,
            int maxCharsPerFile,
            const QString &prefixLabel) const;

        //! Tokenize query text into searchable terms.
        /*! \param [in] queryText Query text to tokenize.
            \return List of normalized query terms.
            \sa collectKnowledgeFiles
        */
        QStringList tokenizeQuery(const QString &queryText) const;

        //! Compute lexical relevance score for a text snippet.
        /*! \param [in] text Text snippet to score.
            \param [in] terms Tokenized query terms.
            \return Integer relevance score.
            \sa collectFromRoot
        */
        int scoreText(const QString &text, const QStringList &terms) const;
};

} // namespace llm
} // namespace ito
