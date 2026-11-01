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

#include <QString>
#include <QList>

namespace ito
{
namespace llm
{

enum class WorkspaceChangeType
{
    CreateFile, //!< Create a new file with provided content.
    UpdateFile, //!< Update an existing file with provided content.
    DeleteFile //!< Delete an existing file.
};

struct WorkspaceChange
{
    WorkspaceChangeType type = WorkspaceChangeType::UpdateFile; //!< Requested workspace operation.
    QString filePath; //!< Relative or absolute target file path.
    QString content; //!< File content for create or update operations.
}; //!< One parsed workspace change operation.

struct WorkspaceChangeSet
{
    QList<WorkspaceChange> changes; //!< Ordered list of parsed workspace operations.

    //! Check whether the change set contains any operations.
    /*! \return True if no changes are present; otherwise false.
        \sa changes
    */
    bool isEmpty() const
    {
        return changes.isEmpty();
    }
};

/*! \class ito::llm::WorkspaceChangeSetParser
    \brief Parses assistant output into structured workspace file changes.

    The parser extracts fenced change blocks and converts them into
    WorkspaceChange entries for review and application flows.

    \sa WorkspaceChangeSet, WorkspaceChange
*/
class WorkspaceChangeSetParser
{
    public:
        //! Parse assistant text and return a structured change set.
        /*! \param [in] assistantText Raw assistant response text.
            \return Parsed workspace change set.
            \sa WorkspaceChangeSet
        */
        WorkspaceChangeSet parse(const QString &assistantText) const;
};

} // namespace llm
} // namespace ito
