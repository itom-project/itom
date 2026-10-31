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

#include "fileOperations.h"

#include "editorFileType.h"

#include <qdir.h>
#include <qfile.h>
#include <qobject.h>
#include <qprocess.h>
#include <qset.h>

namespace ito {
namespace FileOperations {

//-------------------------------------------------------------------------------------
QModelIndexList uniqueRowIndexes(const QModelIndexList& indexes)
{
    // filter indexList such that only one index per row and parent is available
    QModelIndexList indexList;
    bool found;

    foreach (const QModelIndex& idx, indexes)
    {
        found = false;

        foreach (const QModelIndex& idx2, indexList)
        {
            if (idx.row() == idx2.row() && idx.parent() == idx2.parent())
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            indexList.append(idx);
        }
    }

    return indexList;
}

//-------------------------------------------------------------------------------------
QString uniqueChildName(const QString& parentDir, const QString& baseName, const QString& suffix)
{
    const QDir dir(parentDir);
    QString name = baseName + suffix;

    if (dir.exists(name))
    {
        int i = 0;
        bool done = false;

        while (!done)
        {
            i++;
            name = baseName + " (" + QString::number(i) + ")" + suffix;
            done = !dir.exists(name);
        }
    }

    return name;
}

//-------------------------------------------------------------------------------------
ito::RetVal createEmptyFile(const QString& filePath)
{
    QFile file(filePath);

    if (file.exists())
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("The file '%1' already exists.").arg(filePath).toLatin1().data());
    }

    if (!file.open(QIODevice::WriteOnly))
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("The file '%1' could not be created: %2")
                .arg(filePath, file.errorString())
                .toLatin1()
                .data());
    }

    file.close();
    return ito::retOk;
}

//-------------------------------------------------------------------------------------
QStringList editableFilePaths(const QFileInfoList& fileInfos)
{
    QStringList files;
    QSet<QString> seen;

    foreach (const QFileInfo& fi, fileInfos)
    {
        if (fi.isFile() && EditorFileType::isEditable(fi.filePath()) &&
            !seen.contains(fi.filePath()))
        {
            seen.insert(fi.filePath());
            files.append(fi.filePath());
        }
    }

    return files;
}

//-------------------------------------------------------------------------------------
namespace {

QString cleanAbs(const QString& path)
{
    return QDir::cleanPath(QFileInfo(QDir::fromNativeSeparators(path)).absoluteFilePath());
}

ito::RetVal copyDirRecursively(const QString& source, const QString& target)
{
    QDir sourceDir(source);

    if (!QDir().mkpath(target))
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("The directory '%1' could not be created.").arg(target).toLatin1().data());
    }

    const QFileInfoList entries =
        sourceDir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System);

    foreach (const QFileInfo& fi, entries)
    {
        const QString targetPath = target + "/" + fi.fileName();

        if (fi.isDir() && !fi.isSymLink())
        {
            ito::RetVal ret = copyDirRecursively(fi.absoluteFilePath(), targetPath);

            if (ret.containsError())
            {
                return ret;
            }
        }
        else if (!QFile::copy(fi.absoluteFilePath(), targetPath))
        {
            return ito::RetVal(
                ito::retError,
                0,
                QObject::tr("The file '%1' could not be copied to '%2'.")
                    .arg(fi.absoluteFilePath(), targetPath)
                    .toLatin1()
                    .data());
        }
    }

    return ito::retOk;
}

} // namespace

//-------------------------------------------------------------------------------------
bool isInsideDirectory(const QString& path, const QString& dir)
{
    const QString p = cleanAbs(path);
    const QString d = cleanAbs(dir);
    return p == d || p.startsWith(d.endsWith('/') ? d : d + "/");
}

//-------------------------------------------------------------------------------------
ito::RetVal copyToDirectory(const QString& sourcePath, const QString& targetDir, QString* newPath)
{
    const QString source = cleanAbs(sourcePath);
    const QString target = cleanAbs(targetDir);
    const QFileInfo sourceInfo(source);

    if (!sourceInfo.exists())
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("'%1' does not exist.").arg(source).toLatin1().data());
    }

    if (!QFileInfo(target).isDir())
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("The target '%1' is no directory.").arg(target).toLatin1().data());
    }

    if (sourceInfo.isDir() && isInsideDirectory(target, source))
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("A directory cannot be copied into itself.").toLatin1().data());
    }

    // free name: "name", "name - Copy", "name - Copy (2)", ... (suffix is kept)
    const QString suffix = sourceInfo.isDir() || sourceInfo.suffix().isEmpty()
        ? QString()
        : "." + sourceInfo.suffix();
    const QString base = sourceInfo.isDir() || sourceInfo.suffix().isEmpty()
        ? sourceInfo.fileName()
        : sourceInfo.completeBaseName();
    QString name = base + suffix;

    if (QDir(target).exists(name))
    {
        const QString copyBase = base + " - " + QObject::tr("Copy");
        name = copyBase + suffix;
        int i = 1;

        while (QDir(target).exists(name))
        {
            ++i;
            name = copyBase + " (" + QString::number(i) + ")" + suffix;
        }
    }

    const QString destination = target + "/" + name;
    ito::RetVal retval;

    if (sourceInfo.isDir())
    {
        retval += copyDirRecursively(source, destination);
    }
    else if (!QFile::copy(source, destination))
    {
        retval += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("The file '%1' could not be copied to '%2'.")
                .arg(source, destination)
                .toLatin1()
                .data());
    }

    if (!retval.containsError() && newPath)
    {
        *newPath = destination;
    }

    return retval;
}

//-------------------------------------------------------------------------------------
ito::RetVal moveToDirectory(const QString& sourcePath, const QString& targetDir, QString* newPath)
{
    const QString source = cleanAbs(sourcePath);
    const QString target = cleanAbs(targetDir);
    const QFileInfo sourceInfo(source);

    if (!sourceInfo.exists())
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("'%1' does not exist.").arg(source).toLatin1().data());
    }

    if (!QFileInfo(target).isDir())
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("The target '%1' is no directory.").arg(target).toLatin1().data());
    }

    if (sourceInfo.absolutePath() == target)
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("'%1' is already in '%2'.").arg(source, target).toLatin1().data());
    }

    if (sourceInfo.isDir() && isInsideDirectory(target, source))
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("A directory cannot be moved into itself.").toLatin1().data());
    }

    const QString destination = target + "/" + sourceInfo.fileName();

    if (QFileInfo(destination).exists())
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("'%1' already exists.").arg(destination).toLatin1().data());
    }

    bool ok = QDir().rename(source, destination);

    if (!ok)
    {
        // e.g. across file systems: copy, then remove the source
        ito::RetVal ret;

        if (sourceInfo.isDir())
        {
            ret += copyDirRecursively(source, destination);
            ok = !ret.containsError() && QDir(source).removeRecursively();
        }
        else
        {
            ok = QFile::copy(source, destination) && QFile::remove(source);
        }
    }

    if (!ok)
    {
        return ito::RetVal(
            ito::retError,
            0,
            QObject::tr("'%1' could not be moved to '%2'.").arg(source, target).toLatin1().data());
    }

    if (newPath)
    {
        *newPath = destination;
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
void showInGraphicalShell(const QString& filePath)
{
#ifdef __APPLE__
    QStringList args;
    args << "-e";
    args << "tell application \"Finder\"";
    args << "-e";
    args << "activate";
    args << "-e";
    args << "select POSIX file \"" + filePath + "\"";
    args << "-e";
    args << "end tell";
    QProcess::startDetached("osascript", args);
#endif

#ifdef WIN32
    QStringList args;
    args << "/select," << QDir::toNativeSeparators(filePath);
    QProcess::startDetached("explorer", args);
#endif

#if !defined(__APPLE__) && !defined(WIN32)
    Q_UNUSED(filePath);
#endif
}

} // namespace FileOperations
} // namespace ito
