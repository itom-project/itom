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

#include <qlist.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

//! one root folder of a project
struct ProjectFolder
{
    QString path; //!< absolute, cleaned path of the folder
    QString name; //!< display name; empty means: use the folder name
};

//! Reads and writes itom project files (*.itomproject, JSON) and keeps the project settings.
/*!
    The file format (version 1):

    \verbatim
    {
      "version": 1,
      "name": "My Project",
      "folders": [ { "path": ".", "name": "root" }, { "path": "../shared_lib" } ],
      "mainScript": "main.py",
      "workingDirectory": ".",
      "pythonPath": [ "lib" ],
      "excludes": [ "__pycache__", "*.pyc", ".git" ]
    }
    \endverbatim

    Paths in the file are relative to the directory of the project file where possible.
    In memory all paths are absolute and cleaned. Per-user state (open scripts, expanded
    nodes) is not part of this file.

    This class is QtCore only and reports errors via ito::RetVal.
*/
class ProjectFile
{
public:
    static const int currentVersion = 1;

    //! file extension without dot
    static QString fileExtension();

    //! default exclude patterns for new projects
    static QStringList defaultExcludes();

    ProjectFile();

    //! creates a new, unsaved project with one root folder. The project file is placed
    //! in the folder as <name>.itomproject (name defaults to the folder name).
    static ProjectFile fromFolder(const QString& folder, const QString& name = QString());

    //! loads the project from filePath. Returns an error if the file cannot be read or
    //! parsed, and a warning if root folders do not exist (they are kept in the list).
    ito::RetVal load(const QString& filePath);

    //! saves the project. If filePath is empty, the current file path is used.
    ito::RetVal save(const QString& filePath = QString());

    bool isValid() const
    {
        return !m_filePath.isEmpty();
    }

    QString filePath() const
    {
        return m_filePath;
    }

    //! directory of the project file (base for all relative paths)
    QString projectDir() const;

    QString name() const
    {
        return m_name;
    }
    void setName(const QString& name)
    {
        m_name = name;
    }

    QList<ProjectFolder> folders() const
    {
        return m_folders;
    }

    //! appends a root folder. Returns false if the folder is already in the list.
    bool addFolder(const QString& path, const QString& name = QString());

    //! removes a root folder. Returns false if it was not in the list.
    bool removeFolder(const QString& path);

    //! true if the given file or folder is inside one of the root folders
    bool contains(const QString& path) const;

    //! absolute path of the main script or an empty string
    QString mainScript() const
    {
        return m_mainScript;
    }
    void setMainScript(const QString& path);

    //! absolute working directory, defaults to projectDir()
    QString workingDirectory() const;
    void setWorkingDirectory(const QString& path);

    //! absolute additional python path entries
    QStringList pythonPath() const
    {
        return m_pythonPath;
    }
    void setPythonPath(const QStringList& paths);

    QStringList excludes() const
    {
        return m_excludes;
    }
    void setExcludes(const QStringList& patterns)
    {
        m_excludes = patterns;
    }

    //! converts a path to an absolute, cleaned path based on projectDir()
    QString toAbsolute(const QString& path) const;

    //! converts an absolute path to a path relative to projectDir(), if possible
    QString toRelative(const QString& path) const;

    //! normalizes a user given path (cleaned, forward slashes, absolute if possible)
    static QString normalize(const QString& path);

private:
    QString m_filePath;
    QString m_name;
    QList<ProjectFolder> m_folders;
    QString m_mainScript;
    QString m_workingDirectory;
    QStringList m_pythonPath;
    QStringList m_excludes;
};

} // namespace ito
