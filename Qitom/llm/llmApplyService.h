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

#include "../../common/retVal.h"

#include <QString>

class QTextCursor;

namespace ito
{

class ScriptEditorWidget;

namespace llm
{

enum class LlmApplyMode
{
    InsertAtCursor, //!< Insert suggestion text at the current cursor position.
    ReplaceSelection //!< Replace current selection with suggestion text.
};

struct LlmApplyResult
{
    bool applied = false; //!< True if the suggestion was applied successfully.
    bool hadSelection = false; //!< True if an editor selection existed before apply.
    QString message; //!< User-facing result message.
};

/*! \class ito::llm::LlmApplyService
    \brief Applies assistant suggestions to a script editor.

    The service validates editor availability and writeability, then performs
    either insertion or replacement operations using the editor cursor.

    \sa LlmApplyMode, LlmApplyResult
*/
class LlmApplyService
{
    public:
        //! Apply a suggestion to the given script editor.
        /*! \param [in,out] editor Target script editor widget.
            \param [in] suggestion Suggested text from assistant.
            \param [in] mode Apply mode for insertion or replacement.
            \param [out] resultOut Optional apply result details.
            \return retOk on success, otherwise retError.
            \sa applyWithCursor
        */
        RetVal applySuggestion(
            ScriptEditorWidget *editor,
            const QString &suggestion,
            LlmApplyMode mode,
            LlmApplyResult *resultOut = NULL) const;

    private:
        //! Build an error result and return a RetVal error.
        /*! \param [in] message Error message for user-facing result.
            \param [out] resultOut Optional result object to fill.
            \return retError with the given message.
            \sa applySuggestion
        */
        RetVal fail(const QString &message, LlmApplyResult *resultOut) const;

        //! Apply suggestion using an existing text cursor.
        /*! \param [in,out] editor Target script editor widget.
            \param [in,out] cursor Cursor used for edit operations.
            \param [in] suggestion Suggested text from assistant.
            \param [in] mode Apply mode for insertion or replacement.
            \param [out] resultOut Optional apply result details.
            \return retOk on success, otherwise retError.
            \sa applySuggestion
        */
        RetVal applyWithCursor(
            ScriptEditorWidget *editor,
            QTextCursor &cursor,
            const QString &suggestion,
            LlmApplyMode mode,
            LlmApplyResult *resultOut) const;
};

} // namespace llm
} // namespace ito
