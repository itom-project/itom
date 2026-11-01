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

#include "workspaceChangeSet.h"

#include <QList>
#include <QSharedPointer>

namespace ito
{

class ScriptEditorWidget;
struct TextDecoration;

namespace llm
{

struct DiffReviewItem
{
    WorkspaceChange change; //!< Workspace change represented by this review item.
    QList<QSharedPointer<TextDecoration> > decorations; //!< Decorations created for preview.
};

/*! \class ito::llm::LlmDiffReviewService
    \brief Builds and clears in-editor diff decorations for pending changes.

    The service creates line-based decorations for change review and can remove
    all previously generated decorations from an editor.

    \sa DiffReviewItem, WorkspaceChangeSet
*/
class LlmDiffReviewService
{
    public:
        //! Create in-editor preview decorations for a change set.
        /*! \param [in,out] editor Target script editor.
            \param [in] changeSet Change set to preview.
            \return Review items with associated decorations.
            \sa clearPreview
        */
        QList<DiffReviewItem> preview(
            ScriptEditorWidget *editor,
            const WorkspaceChangeSet &changeSet) const;

        //! Remove preview decorations for existing review items.
        /*! \param [in,out] editor Target script editor.
            \param [in,out] items Review items to clear.
            \sa preview
        */
        void clearPreview(ScriptEditorWidget *editor, QList<DiffReviewItem> &items) const;

    private:
        //! Create line highlight decorations from old/new text.
        /*! \param [in,out] editor Target script editor.
            \param [in] oldText Text before applying change.
            \param [in] newText Text after applying change.
            \return Created text decorations.
            \sa preview
        */
        QList<QSharedPointer<TextDecoration> > addLineHighlights(
            ScriptEditorWidget *editor,
            const QString &oldText,
            const QString &newText) const;
};

} // namespace llm
} // namespace ito
