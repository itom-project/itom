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
#include <QStringList>

namespace ito
{
namespace llm
{

struct ProjectDiscoveryResult
{
    QString rootPath; //!< Discovered project root directory.
    QString stopReason; //!< Reason why directory ascent stopped.
    QStringList filePaths; //!< Candidate project files used for context.
}; //!< Result payload of project discovery.

/*! \class ito::llm::ProjectDiscoveryService
    \brief Discovers project root and candidate context files.

    The service infers a project root from the active file and working
    directory, then collects a bounded set of source files for prompt context.

    \sa ProjectDiscoveryResult
*/
class ProjectDiscoveryService
{
    public:
        //! Discover project root and related context files.
        /*! \param [in] activeFilePath Path of active editor file.
            \param [in] workingDirectory Current process working directory.
            \return Discovery result with root and collected file paths.
            \sa collectProjectFiles
        */
        ProjectDiscoveryResult discover(
            const QString &activeFilePath,
            const QString &workingDirectory) const;

    private:
        //! Detect a project root by ascending from a start directory.
        /*! \param [in] startDirectory Directory used as search start.
            \param [out] stopReasonOut Reason for search stop condition.
            \return Detected project root path or empty string.
            \sa discover
        */
        QString detectProjectRoot(
            const QString &startDirectory,
            QString &stopReasonOut) const;

        //! Collect context candidate files from the discovered root.
        /*! \param [in] rootPath Discovered project root path.
            \param [in] maxFiles Maximum number of files to collect.
            \return List of collected file paths.
            \sa isExcludedDirectory
        */
        QStringList collectProjectFiles(const QString &rootPath, int maxFiles) const;

        //! Check whether a directory should be excluded from discovery.
        /*! \param [in] directoryName Basename of a directory.
            \return True if directory is excluded; otherwise false.
            \sa collectProjectFiles
        */
        bool isExcludedDirectory(const QString &directoryName) const;
};

} // namespace llm
} // namespace ito
