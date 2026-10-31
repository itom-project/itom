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

#include "projectOrganizer.h"

#include "../AppManagement.h"

#include <qdir.h>
#include <qfileinfo.h>
#include <qcryptographichash.h>
#include <qsettings.h>

namespace ito {

//-------------------------------------------------------------------------------------
ProjectOrganizer::ProjectOrganizer(QObject* parent) :
    QObject(parent), m_setWorkingDirectoryOnOpen(true), m_restoreScripts(true)
{
    loadSettings();
}

//-------------------------------------------------------------------------------------
ProjectOrganizer::~ProjectOrganizer()
{
    saveSettings();
}

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectOrganizer::projectFileFilter()
{
    return tr("itom project files (*.%1)").arg(ProjectFile::fileExtension());
}

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectOrganizer::settingsGroup(const QString& projectFilePath)
{
    const QByteArray hash = QCryptographicHash::hash(
        ProjectFile::normalize(projectFilePath).toUtf8(), QCryptographicHash::Sha1);
    return "Projects/" + QString::fromLatin1(hash.toHex().left(16));
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::reloadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Projects");
    m_setWorkingDirectoryOnOpen = settings.value("setWorkingDirectoryOnOpen", true).toBool();
    m_restoreScripts = settings.value("restoreScripts", true).toBool();
    settings.endGroup();
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Projects");
    m_recentProjects.clear();

    foreach (const QString& p, settings.value("recent").toStringList())
    {
        const QString abs = ProjectFile::normalize(p);

        if (!abs.isEmpty() && !m_recentProjects.contains(abs))
        {
            m_recentProjects.append(abs);
        }
    }

    while (m_recentProjects.size() > maxRecentProjects)
    {
        m_recentProjects.removeLast();
    }

    m_setWorkingDirectoryOnOpen = settings.value("setWorkingDirectoryOnOpen", true).toBool();
    m_restoreScripts = settings.value("restoreScripts", true).toBool();
    settings.endGroup();
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::saveSettings() const
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("Projects");
    settings.setValue("recent", m_recentProjects);
    settings.setValue("setWorkingDirectoryOnOpen", m_setWorkingDirectoryOnOpen);
    settings.setValue("restoreScripts", m_restoreScripts);
    settings.endGroup();
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::addRecentProject(const QString& filePath)
{
    const QString abs = ProjectFile::normalize(filePath);
    m_recentProjects.removeAll(abs);
    m_recentProjects.prepend(abs);

    while (m_recentProjects.size() > maxRecentProjects)
    {
        m_recentProjects.removeLast();
    }

    saveSettings();
    emit recentProjectsChanged();
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::removeRecentProject(const QString& filePath)
{
    if (m_recentProjects.removeAll(ProjectFile::normalize(filePath)) > 0)
    {
        saveSettings();
        emit recentProjectsChanged();
    }
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::clearRecentProjects()
{
    if (!m_recentProjects.isEmpty())
    {
        m_recentProjects.clear();
        saveSettings();
        emit recentProjectsChanged();
    }
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::setSetWorkingDirectoryOnOpen(bool enabled)
{
    if (m_setWorkingDirectoryOnOpen != enabled)
    {
        m_setWorkingDirectoryOnOpen = enabled;
        saveSettings();
    }
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::applyWorkingDirectory()
{
    if (!m_setWorkingDirectoryOnOpen || !m_project.isValid())
    {
        return;
    }

    const QString wd = m_project.workingDirectory();

    if (QFileInfo(wd).isDir() && QDir::cleanPath(QDir::currentPath()) != wd)
    {
        if (QDir::setCurrent(wd))
        {
            emit workingDirectoryChanged();
        }
    }
}

//-------------------------------------------------------------------------------------
ito::RetVal ProjectOrganizer::openProject(const QString& filePath, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;
    ProjectFile project;

    retval += project.load(filePath);

    if (!retval.containsError())
    {
        if (m_project.isValid())
        {
            retval += closeProject();
        }

        m_project = project;
        addRecentProject(m_project.filePath());
        emit projectOpened(m_project.filePath());
        applyWorkingDirectory();
    }
    else
    {
        // a project file that cannot be loaded is removed from the recent list
        removeRecentProject(filePath);
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
ito::RetVal ProjectOrganizer::newProjectFromFolder(
    const QString& folder, const QString& name, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;

    if (!QFileInfo(folder).isDir())
    {
        retval += ito::RetVal(
            ito::retError,
            0,
            tr("The folder '%1' does not exist.").arg(folder).toLatin1().data());
    }
    else
    {
        if (m_project.isValid())
        {
            retval += closeProject();
        }

        m_project = ProjectFile::fromFolder(folder, name);
        emit projectOpened(m_project.filePath());
        applyWorkingDirectory();
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
ito::RetVal ProjectOrganizer::saveProject(const QString& filePath, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;

    if (!m_project.isValid())
    {
        retval += ito::RetVal(ito::retError, 0, tr("No project is opened.").toLatin1().data());
    }
    else
    {
        retval += m_project.save(filePath);

        if (!retval.containsError())
        {
            addRecentProject(m_project.filePath());
            emit projectChanged();
        }
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
ito::RetVal ProjectOrganizer::closeProject(ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retval;

    if (m_project.isValid())
    {
        const QString filePath = m_project.filePath();
        m_project = ProjectFile();
        emit projectClosed(filePath);
    }

    if (waitCond)
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
void ProjectOrganizer::projectModified()
{
    if (m_project.isValid())
    {
        emit projectChanged();
    }
}

} // namespace ito
