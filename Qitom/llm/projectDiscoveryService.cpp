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

#include "projectDiscoveryService.h"

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

namespace ito
{
namespace llm
{

ProjectDiscoveryResult ProjectDiscoveryService::discover(
    const QString &activeFilePath,
    const QString &workingDirectory) const
{
    ProjectDiscoveryResult result;

    QString startDirectory;
    if (!activeFilePath.trimmed().isEmpty())
    {
        QFileInfo fi(activeFilePath);
        startDirectory = fi.exists() ? fi.absolutePath() : QString();
    }

    if (startDirectory.isEmpty())
    {
        QDir wd(workingDirectory);
        if (wd.exists())
        {
            startDirectory = wd.absolutePath();
        }
    }

    if (startDirectory.isEmpty())
    {
        startDirectory = QDir::currentPath();
    }

    result.rootPath = detectProjectRoot(startDirectory, result.stopReason);
    result.filePaths = collectProjectFiles(result.rootPath, 40);

    return result;
}

QString ProjectDiscoveryService::detectProjectRoot(
    const QString &startDirectory,
    QString &stopReasonOut) const
{
    QDir current(startDirectory);
    QString fallbackRoot = current.absolutePath();
    QString lastInitRoot;

    while (current.exists())
    {
        const QString currentPath = current.absolutePath();

        if (QFileInfo::exists(QDir(currentPath).filePath("pyproject.toml")))
        {
            stopReasonOut = QString("pyproject.toml");
            return currentPath;
        }

        if (QFileInfo::exists(QDir(currentPath).filePath("__init__.py")))
        {
            lastInitRoot = currentPath;
        }

        if (!current.cdUp())
        {
            break;
        }

        if (current.absolutePath() == currentPath)
        {
            break;
        }
    }

    if (!lastInitRoot.isEmpty())
    {
        stopReasonOut = QString("nearest __init__.py ancestor");
        return lastInitRoot;
    }

    stopReasonOut = QString("fallback start directory");
    return fallbackRoot;
}

QStringList ProjectDiscoveryService::collectProjectFiles(const QString &rootPath, int maxFiles) const
{
    QStringList files;
    if (rootPath.isEmpty())
    {
        return files;
    }

    QDir root(rootPath);
    if (!root.exists())
    {
        return files;
    }

    if (QFileInfo::exists(root.filePath("pyproject.toml")))
    {
        files << root.filePath("pyproject.toml");
    }

    QDirIterator it(
        rootPath,
        QStringList() << "*.py",
        QDir::Files,
        QDirIterator::Subdirectories);

    while (it.hasNext() && files.size() < maxFiles)
    {
        const QString filePath = it.next();
        const QFileInfo fi(filePath);

        QString relativeDir = root.relativeFilePath(fi.absolutePath());
        const QStringList parts = relativeDir.split('/', Qt::SkipEmptyParts);

        bool excluded = false;
        for (int i = 0; i < parts.size(); ++i)
        {
            if (isExcludedDirectory(parts[i]))
            {
                excluded = true;
                break;
            }
        }

        if (excluded)
        {
            continue;
        }

        files << fi.absoluteFilePath();
    }

    return files;
}

bool ProjectDiscoveryService::isExcludedDirectory(const QString &directoryName) const
{
    const QString name = directoryName.trimmed().toLower();

    return name == ".git" ||
        name == ".svn" ||
        name == ".hg" ||
        name == ".venv" ||
        name == "venv" ||
        name == "env" ||
        name == "__pycache__" ||
        name == "build" ||
        name == "dist";
}

} // namespace llm
} // namespace ito
