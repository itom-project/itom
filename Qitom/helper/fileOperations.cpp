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
