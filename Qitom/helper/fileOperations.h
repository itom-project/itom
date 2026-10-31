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

#include <qabstractitemmodel.h>
#include <qfileinfo.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

//! GUI independent helpers for the file browser views (file system mode and project mode).
/*!
    These functions were extracted from FileSystemDockWidget so that they can be shared
    between different tree views and tested without a GUI. They never show dialogs and
    report errors via ito::RetVal.
*/
namespace FileOperations {

//! returns only one index per row (the index of the first column that was seen).
/*!
    Tree views with several columns report one index per cell in selectedIndexes().
    The order of the first occurrence of every row is kept.
*/
QModelIndexList uniqueRowIndexes(const QModelIndexList& indexes);

//! returns a child name that does not exist yet in parentDir.
/*!
    The name is baseName + suffix. If this already exists, " (1)", " (2)", ... is
    inserted between baseName and suffix until a free name is found, e.g.
    "New Folder (2)" or "New Script (1).py".

    \param parentDir absolute path of the directory
    \param baseName name without suffix, e.g. "New Script"
    \param suffix suffix including the dot, e.g. ".py", or an empty string
*/
QString uniqueChildName(const QString& parentDir, const QString& baseName, const QString& suffix);

//! creates an empty file. Fails if the file exists or cannot be written.
ito::RetVal createEmptyFile(const QString& filePath);

//! returns the unique file paths of all entries that are files and can be opened
//! in the script editor (see EditorFileType::isEditable). The order is kept.
QStringList editableFilePaths(const QFileInfoList& fileInfos);

//! shows the file or directory in the file manager of the operating system
//! (Explorer, Finder). Does nothing on other platforms.
void showInGraphicalShell(const QString& filePath);

//! copies a file or a directory (recursively) into the directory targetDir.
/*!
    The copy gets the name of the source. If an entry with this name already exists in
    targetDir, " - Copy", " - Copy (2)", ... is appended to the base name (like a file
    manager does). Copying a directory into itself or one of its sub directories fails.

    \param newPath receives the path of the copy (optional)
*/
ito::RetVal copyToDirectory(
    const QString& sourcePath, const QString& targetDir, QString* newPath = nullptr);

//! moves a file or a directory into the directory targetDir (keeps the name).
/*!
    Fails if an entry with the same name exists in targetDir, if the source is a parent
    of targetDir, or if it is already in targetDir (nothing to do, returns an error).
    Moving across file systems is done by copy and delete.

    \param newPath receives the new path (optional)
*/
ito::RetVal moveToDirectory(
    const QString& sourcePath, const QString& targetDir, QString* newPath = nullptr);

//! true if path is dir itself or inside dir
bool isInsideDirectory(const QString& path, const QString& dir);

} // namespace FileOperations
} // namespace ito
