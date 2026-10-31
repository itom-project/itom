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

#include "projectFile.h"

#include "../../common/retVal.h"
#include "../../common/sharedStructuresQt.h"

#include <qobject.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

//! Owns the currently opened project (see ProjectFile) and the list of recent projects.
/*!
    The organizer lives in the main thread (it is created by MainApplication). Other
    threads (e.g. the Python thread) call the public slots via QMetaObject::invokeMethod
    with an ItomSharedSemaphore and read the result from the semaphore's returnValue.

    Settings (group "Projects"): "recent" (QStringList of project files, newest first),
    "setWorkingDirectoryOnOpen" (bool, default true).
*/
class ProjectOrganizer : public QObject
{
    Q_OBJECT

public:
    enum
    {
        maxRecentProjects = 10
    };

    explicit ProjectOrganizer(QObject* parent = nullptr);
    ~ProjectOrganizer();

    bool hasProject() const
    {
        return m_project.isValid();
    }

    //! the current project (invalid if no project is opened)
    const ProjectFile& project() const
    {
        return m_project;
    }

    //! modifiable access for the GUI. Call projectModified() afterwards.
    ProjectFile& project()
    {
        return m_project;
    }

    QStringList recentProjects() const
    {
        return m_recentProjects;
    }

    bool setWorkingDirectoryOnOpen() const
    {
        return m_setWorkingDirectoryOnOpen;
    }

    //! file filter for open/save dialogs
    static QString projectFileFilter();

public slots:
    //! opens the project file. A warning (e.g. missing folders) does not prevent opening.
    ito::RetVal openProject(const QString& filePath, ItomSharedSemaphore* waitCond = nullptr);

    //! creates a new project for the folder (not saved yet), and opens it
    ito::RetVal newProjectFromFolder(
        const QString& folder, const QString& name = QString(), ItomSharedSemaphore* waitCond = nullptr);

    //! saves the current project (to filePath if given)
    ito::RetVal saveProject(const QString& filePath = QString(), ItomSharedSemaphore* waitCond = nullptr);

    //! closes the current project (nothing happens if no project is open)
    ito::RetVal closeProject(ItomSharedSemaphore* waitCond = nullptr);

    //! notifies all listeners that the project settings changed (folders, main script, ...)
    void projectModified();

    void removeRecentProject(const QString& filePath);
    void clearRecentProjects();
    void setSetWorkingDirectoryOnOpen(bool enabled);

signals:
    void projectOpened(const QString& filePath);
    void projectClosed();
    void projectChanged(); //!< folders, main script etc. of the open project changed
    void recentProjectsChanged();
    void workingDirectoryChanged(); //!< the cwd was changed by the organizer

private:
    void addRecentProject(const QString& filePath);
    void loadSettings();
    void saveSettings() const;
    void applyWorkingDirectory();

    ProjectFile m_project;
    QStringList m_recentProjects;
    bool m_setWorkingDirectoryOnOpen;
};

} // namespace ito
