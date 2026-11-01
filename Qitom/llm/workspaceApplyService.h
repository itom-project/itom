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

#include "../../common/retVal.h"

#include <QStringList>

namespace ito
{
namespace llm
{

struct WorkspaceApplyResult
{
    int createdFiles = 0; //!< Number of successfully created files.
    int updatedFiles = 0; //!< Number of successfully updated files.
    int deletedFiles = 0; //!< Number of successfully deleted files.
    QStringList errors; //!< Collected apply errors per change operation.
}; //!< Aggregated result for workspace apply operation.

/*! \class ito::llm::WorkspaceApplyService
    \brief Applies parsed workspace changes to files under a workspace root.

    The service validates target paths against the workspace root and applies
    create, update, or delete operations while collecting operation statistics.

    \sa WorkspaceChangeSet, WorkspaceApplyResult
*/
class WorkspaceApplyService
{
    public:
        //! Apply a complete change set to the workspace.
        /*! \param [in] workspaceRoot Root directory of the active workspace.
            \param [in] changeSet Parsed set of workspace changes.
            \param [out] resultOut Optional aggregate apply result.
            \return retOk if all operations succeed, otherwise retError aggregate.
            \sa applySingleChange
        */
        RetVal apply(
            const QString &workspaceRoot,
            const WorkspaceChangeSet &changeSet,
            WorkspaceApplyResult *resultOut = NULL) const;

    private:
        //! Resolve and validate a target path within workspace root boundaries.
        /*! \param [in] workspaceRoot Root directory of the active workspace.
            \param [in] relativeOrAbsolutePath Candidate target path.
            \return Absolute target path if safe; empty string otherwise.
            \sa applySingleChange
        */
        QString resolveSafePath(const QString &workspaceRoot, const QString &relativeOrAbsolutePath) const;

        //! Apply one workspace change operation.
        /*! \param [in] workspaceRoot Root directory of the active workspace.
            \param [in] change Single change operation to apply.
            \param [out] resultOut Optional aggregate apply result.
            \return retOk on success, otherwise retError.
            \sa apply
        */
        RetVal applySingleChange(
            const QString &workspaceRoot,
            const WorkspaceChange &change,
            WorkspaceApplyResult *resultOut) const;
};

} // namespace llm
} // namespace ito
