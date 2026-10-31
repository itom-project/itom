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

#include <qobject.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

class ProjectOrganizer;

//! Applies the per-project session when a project is opened or closed.
/*!
    - Open scripts: when a project is opened, the current scripts are saved to the
      global settings group (or the group of the previous project), closed and the
      scripts of the new project are restored (ScriptEditorOrganizer). When the project
      is closed, its scripts are saved to its group and the global scripts are restored.
      This can be switched off with the setting "Projects/restoreScripts".
    - Python path: the "pythonPath" entries of the project are prepended to sys.path
      while the project is open and removed again when it is closed.

    The object lives in the main thread (created by MainApplication after the
    ScriptEditorOrganizer).
*/
class ProjectSession : public QObject
{
    Q_OBJECT

public:
    explicit ProjectSession(ProjectOrganizer* organizer, QObject* parent = nullptr);
    ~ProjectSession();

    //! python statement that adds the paths to sys.path (public for tests)
    static QString addToSysPathStatement(const QStringList& paths);

    //! python statement that removes the paths from sys.path (public for tests)
    static QString removeFromSysPathStatement(const QStringList& paths);

private slots:
    void projectOpened(const QString& filePath);
    void projectClosed(const QString& filePath);

private:
    void runPython(const QString& statement);
    void switchScripts(const QString& saveGroup, const QString& restoreGroup);
    static QString pyStringList(const QStringList& paths);

    ProjectOrganizer* m_pOrganizer;
    QStringList m_currentPythonPath; //!< entries added to sys.path for the open project
};

} // namespace ito
