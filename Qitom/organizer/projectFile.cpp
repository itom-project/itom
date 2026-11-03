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

#include "projectFile.h"

#include <qdir.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qobject.h>

namespace ito {

namespace {

ito::RetVal error(const QString& msg)
{
    return ito::RetVal(ito::retError, 0, msg.toLatin1().data());
}

ito::RetVal warning(const QString& msg)
{
    return ito::RetVal(ito::retWarning, 0, msg.toLatin1().data());
}

} // namespace

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectFile::fileExtension()
{
    return "itomproject";
}

//-------------------------------------------------------------------------------------
/*static*/ QStringList ProjectFile::defaultExcludes()
{
    return QStringList() << "__pycache__" << "*.pyc" << ".git" << ".svn" << ".hg";
}

//-------------------------------------------------------------------------------------
ProjectFile::ProjectFile() : m_excludes(defaultExcludes())
{
}

//-------------------------------------------------------------------------------------
/*static*/ ProjectFile ProjectFile::fromFolder(const QString& folder, const QString& name)
{
    ProjectFile project;
    const QString absFolder = normalize(folder);
    const QString projectName = name.isEmpty() ? QFileInfo(absFolder).fileName() : name;

    project.m_filePath = absFolder + "/" + projectName + "." + fileExtension();
    project.m_name = projectName;
    project.addFolder(absFolder);
    return project;
}

//-------------------------------------------------------------------------------------
/*static*/ QString ProjectFile::normalize(const QString& path)
{
    if (path.isEmpty())
    {
        return QString();
    }

    QString p = QDir::fromNativeSeparators(path);
    QFileInfo fi(p);

    if (fi.isRelative())
    {
        p = fi.absoluteFilePath();
    }

    p = QDir::cleanPath(p);

    // remove a trailing slash (except for a root like "/" or "C:/")
    if (p.size() > 1 && p.endsWith('/') && !p.endsWith(":/"))
    {
        p.chop(1);
    }

    return p;
}

//-------------------------------------------------------------------------------------
QString ProjectFile::projectDir() const
{
    if (m_filePath.isEmpty())
    {
        return QString();
    }

    return QFileInfo(m_filePath).absolutePath();
}

//-------------------------------------------------------------------------------------
QString ProjectFile::toAbsolute(const QString& path) const
{
    if (path.isEmpty())
    {
        return QString();
    }

    const QString p = QDir::fromNativeSeparators(path);

    if (QFileInfo(p).isRelative() && !projectDir().isEmpty())
    {
        return normalize(QDir(projectDir()).absoluteFilePath(p));
    }

    return normalize(p);
}

//-------------------------------------------------------------------------------------
QString ProjectFile::toRelative(const QString& path) const
{
    if (path.isEmpty())
    {
        return QString();
    }

    const QString abs = normalize(path);
    const QString dir = projectDir();

    if (dir.isEmpty())
    {
        return abs;
    }

    if (abs == dir)
    {
        return ".";
    }

    // QDir::relativeFilePath returns the absolute path if there is no common
    // root (e.g. different drives on Windows)
    return QDir(dir).relativeFilePath(abs);
}

//-------------------------------------------------------------------------------------
bool ProjectFile::addFolder(const QString& path, const QString& name)
{
    const QString abs = toAbsolute(path);

    if (abs.isEmpty())
    {
        return false;
    }

    foreach (const ProjectFolder& f, m_folders)
    {
        if (f.path == abs)
        {
            return false;
        }
    }

    ProjectFolder folder;
    folder.path = abs;
    folder.name = name;
    m_folders.append(folder);
    return true;
}

//-------------------------------------------------------------------------------------
bool ProjectFile::removeFolder(const QString& path)
{
    const QString abs = toAbsolute(path);

    for (int i = 0; i < m_folders.size(); ++i)
    {
        if (m_folders[i].path == abs)
        {
            m_folders.removeAt(i);
            return true;
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------
bool ProjectFile::contains(const QString& path) const
{
    const QString abs = toAbsolute(path);

    foreach (const ProjectFolder& f, m_folders)
    {
        if (abs == f.path || abs.startsWith(f.path + "/"))
        {
            return true;
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------
void ProjectFile::setMainScript(const QString& path)
{
    m_mainScript = toAbsolute(path);
}

//-------------------------------------------------------------------------------------
QString ProjectFile::workingDirectory() const
{
    return m_workingDirectory.isEmpty() ? projectDir() : m_workingDirectory;
}

//-------------------------------------------------------------------------------------
void ProjectFile::setWorkingDirectory(const QString& path)
{
    m_workingDirectory = toAbsolute(path);
}

//-------------------------------------------------------------------------------------
void ProjectFile::setPythonPath(const QStringList& paths)
{
    m_pythonPath.clear();

    foreach (const QString& p, paths)
    {
        const QString abs = toAbsolute(p);

        if (!abs.isEmpty() && !m_pythonPath.contains(abs))
        {
            m_pythonPath.append(abs);
        }
    }
}

//-------------------------------------------------------------------------------------
ito::RetVal ProjectFile::load(const QString& filePath)
{
    ito::RetVal retval;
    const QString absFilePath = normalize(filePath);
    QFile file(absFilePath);

    if (!file.open(QIODevice::ReadOnly))
    {
        return error(QObject::tr("The project file '%1' could not be opened: %2")
                         .arg(absFilePath, file.errorString()));
    }

    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    file.close();

    if (parseError.error != QJsonParseError::NoError || !doc.isObject())
    {
        return error(QObject::tr("The project file '%1' is no valid JSON document: %2")
                         .arg(absFilePath, parseError.errorString()));
    }

    const QJsonObject root = doc.object();
    const int version = root.value("version").toInt(0);

    if (version < 1 || version > currentVersion)
    {
        return error(QObject::tr("The project file '%1' has the unsupported version %2 "
                                 "(supported: 1 to %3).")
                         .arg(absFilePath)
                         .arg(version)
                         .arg(currentVersion));
    }

    // from here on, the object is replaced
    m_filePath = absFilePath;
    m_folders.clear();
    m_mainScript.clear();
    m_workingDirectory.clear();
    m_pythonPath.clear();

    m_name = root.value("name").toString();

    if (m_name.isEmpty())
    {
        m_name = QFileInfo(absFilePath).completeBaseName();
    }

    const QJsonValue folders = root.value("folders");

    if (folders.isArray())
    {
        foreach (const QJsonValue& v, folders.toArray())
        {
            QString path;
            QString name;

            if (v.isObject())
            {
                path = v.toObject().value("path").toString();
                name = v.toObject().value("name").toString();
            }
            else
            {
                path = v.toString();
            }

            if (path.isEmpty())
            {
                continue;
            }

            addFolder(path, name);
        }
    }

    if (m_folders.isEmpty())
    {
        // a project without folders shows the directory of the project file
        addFolder(".");
    }

    foreach (const ProjectFolder& f, m_folders)
    {
        if (!QFileInfo(f.path).isDir())
        {
            retval += warning(
                QObject::tr("The project folder '%1' does not exist.").arg(f.path));
        }
    }

    setMainScript(root.value("mainScript").toString());
    setWorkingDirectory(root.value("workingDirectory").toString());

    QStringList pythonPath;

    foreach (const QJsonValue& v, root.value("pythonPath").toArray())
    {
        pythonPath << v.toString();
    }

    setPythonPath(pythonPath);

    if (root.contains("excludes") && root.value("excludes").isArray())
    {
        m_excludes.clear();

        foreach (const QJsonValue& v, root.value("excludes").toArray())
        {
            if (!v.toString().isEmpty())
            {
                m_excludes << v.toString();
            }
        }
    }
    else
    {
        m_excludes = defaultExcludes();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
ito::RetVal ProjectFile::save(const QString& filePath)
{
    if (!filePath.isEmpty())
    {
        m_filePath = normalize(filePath);
    }

    if (m_filePath.isEmpty())
    {
        return error(QObject::tr("No file name is given for the project."));
    }

    QJsonObject root;
    root.insert("version", currentVersion);
    root.insert("name", m_name);

    QJsonArray folders;

    foreach (const ProjectFolder& f, m_folders)
    {
        QJsonObject obj;
        obj.insert("path", toRelative(f.path));

        if (!f.name.isEmpty())
        {
            obj.insert("name", f.name);
        }

        folders.append(obj);
    }

    root.insert("folders", folders);

    if (!m_mainScript.isEmpty())
    {
        root.insert("mainScript", toRelative(m_mainScript));
    }

    if (!m_workingDirectory.isEmpty())
    {
        root.insert("workingDirectory", toRelative(m_workingDirectory));
    }

    QJsonArray pythonPath;

    foreach (const QString& p, m_pythonPath)
    {
        pythonPath.append(toRelative(p));
    }

    root.insert("pythonPath", pythonPath);
    root.insert("excludes", QJsonArray::fromStringList(m_excludes));

    QFile file(m_filePath);

    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return error(QObject::tr("The project file '%1' could not be written: %2")
                         .arg(m_filePath, file.errorString()));
    }

    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();
    return ito::retOk;
}

} // namespace ito
