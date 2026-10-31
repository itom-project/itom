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

#include "gtest/gtest.h"

#include "helper/fileOperations.h"

#include <qdir.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qstandarditemmodel.h>
#include <qtemporarydir.h>

using namespace ito;

namespace {

void touch(const QString& path)
{
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly)) << path.toStdString();
    f.close();
}

} // namespace

//-------------------------------------------------------------------------------------
TEST(FileOperations, uniqueRowIndexesKeepsOneIndexPerRow)
{
    // model with 2 columns and 3 rows, the third row has a child row
    QStandardItemModel model(3, 2);
    QStandardItem* parent = new QStandardItem("parent");
    parent->appendRow(QList<QStandardItem*>() << new QStandardItem("c0") << new QStandardItem("c1"));
    model.setItem(2, 0, parent);

    QModelIndexList all;
    all << model.index(0, 1) << model.index(0, 0) << model.index(1, 0) << model.index(1, 1);
    all << model.index(0, 0, model.index(2, 0)) << model.index(0, 1, model.index(2, 0));

    QModelIndexList unique = FileOperations::uniqueRowIndexes(all);

    ASSERT_EQ(unique.size(), 3);
    // order of first occurrence is kept, including the column of the first index
    EXPECT_EQ(unique[0], model.index(0, 1));
    EXPECT_EQ(unique[1], model.index(1, 0));
    EXPECT_EQ(unique[2], model.index(0, 0, model.index(2, 0)));

    EXPECT_TRUE(FileOperations::uniqueRowIndexes(QModelIndexList()).isEmpty());
}

//-------------------------------------------------------------------------------------
TEST(FileOperations, uniqueChildNameReturnsBaseNameIfFree)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    EXPECT_EQ(FileOperations::uniqueChildName(dir.path(), "New Folder", ""), "New Folder");
    EXPECT_EQ(FileOperations::uniqueChildName(dir.path(), "New Script", ".py"), "New Script.py");
}

//-------------------------------------------------------------------------------------
TEST(FileOperations, uniqueChildNameAppendsCounter)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    QDir d(dir.path());

    ASSERT_TRUE(d.mkdir("New Folder"));
    EXPECT_EQ(FileOperations::uniqueChildName(dir.path(), "New Folder", ""), "New Folder (1)");

    ASSERT_TRUE(d.mkdir("New Folder (1)"));
    EXPECT_EQ(FileOperations::uniqueChildName(dir.path(), "New Folder", ""), "New Folder (2)");

    // files and directories with the same name count as "exists"
    touch(d.filePath("New Script.py"));
    touch(d.filePath("New Script (1).py"));
    ASSERT_TRUE(d.mkdir("New Script (2).py"));
    EXPECT_EQ(FileOperations::uniqueChildName(dir.path(), "New Script", ".py"), "New Script (3).py");
}

//-------------------------------------------------------------------------------------
TEST(FileOperations, createEmptyFileCreatesFile)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString path = QDir(dir.path()).filePath("a.py");

    RetVal ret = FileOperations::createEmptyFile(path);
    EXPECT_FALSE(ret.containsError()) << ret.errorMessage();
    EXPECT_TRUE(QFileInfo(path).isFile());
    EXPECT_EQ(QFileInfo(path).size(), 0);
}

//-------------------------------------------------------------------------------------
TEST(FileOperations, createEmptyFileFailsIfExistsOrNotWritable)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString path = QDir(dir.path()).filePath("a.py");

    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write("content");
    }

    // existing file must not be truncated
    RetVal ret = FileOperations::createEmptyFile(path);
    EXPECT_TRUE(ret.containsError());
    EXPECT_EQ(QFileInfo(path).size(), 7);

    // parent directory does not exist
    ret = FileOperations::createEmptyFile(QDir(dir.path()).filePath("missing/b.py"));
    EXPECT_TRUE(ret.containsError());
    EXPECT_FALSE(QString(ret.errorMessage()).isEmpty());
}

//-------------------------------------------------------------------------------------
TEST(FileOperations, editableFilePathsFiltersAndDeduplicates)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    QDir d(dir.path());

    touch(d.filePath("script.py"));
    touch(d.filePath("notes.txt"));
    touch(d.filePath("image.png"));
    ASSERT_TRUE(d.mkdir("folder.py"));

    QFileInfoList infos;
    infos << QFileInfo(d.filePath("script.py")) << QFileInfo(d.filePath("notes.txt"))
          << QFileInfo(d.filePath("script.py")) // duplicate (second column of a tree view)
          << QFileInfo(d.filePath("image.png")) // not editable
          << QFileInfo(d.filePath("folder.py")) // a directory
          << QFileInfo(d.filePath("does_not_exist.py")); // not a file

    QStringList files = FileOperations::editableFilePaths(infos);

    ASSERT_EQ(files.size(), 2);
    EXPECT_EQ(files[0], d.filePath("script.py"));
    EXPECT_EQ(files[1], d.filePath("notes.txt"));
}
