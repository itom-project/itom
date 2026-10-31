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

#include "models/projectTreeModel.h"
#include "organizer/projectFile.h"

#include <QAbstractItemModelTester>
#include <qcoreapplication.h>
#include <qdir.h>
#include <qfile.h>
#include <qsignalspy.h>
#include <qtemporarydir.h>
#include <qtest.h>

using namespace ito;

namespace {

void touch(const QString& path, const QByteArray& content = QByteArray())
{
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly)) << path.toStdString();
    f.write(content);
    f.close();
}

//! temp dir with a small tree:
//! root/
//!   b_folder/      (sub.py)
//!   A_folder/      (empty)
//!   __pycache__/   (x.pyc)
//!   main.py
//!   notes.txt
//!   image.png
//!   cache.pyc
//!   .hidden
struct TempTree
{
    QTemporaryDir dir;
    QString root;

    TempTree() : root(ProjectFile::normalize(dir.path()))
    {
        QDir d(root);
        d.mkdir("b_folder");
        d.mkdir("A_folder");
        d.mkdir("__pycache__");
        touch(d.filePath("b_folder/sub.py"));
        touch(d.filePath("__pycache__/x.pyc"));
        touch(d.filePath("main.py"));
        touch(d.filePath("notes.txt"));
        touch(d.filePath("image.png"));
        touch(d.filePath("cache.pyc"));
        touch(d.filePath(".hidden"));
    }

    QString path(const QString& rel) const
    {
        return root + "/" + rel;
    }
};

QStringList childNames(const ProjectTreeModel& model, const QModelIndex& parent)
{
    QStringList names;

    for (int r = 0; r < model.rowCount(parent); ++r)
    {
        names << model.index(r, 0, parent).data().toString();
    }

    return names;
}

QModelIndex loadedRoot(ProjectTreeModel& model, int row = 0)
{
    QModelIndex root = model.index(row, 0);
    if (model.canFetchMore(root))
    {
        model.fetchMore(root);
    }
    return root;
}

} // namespace

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, emptyModel)
{
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);

    EXPECT_EQ(model.rowCount(), 0);
    EXPECT_EQ(model.columnCount(), 1);
    EXPECT_FALSE(model.hasChildren());
    EXPECT_FALSE(model.index(0, 0).isValid());
    EXPECT_EQ(model.filePath(QModelIndex()), "");
    EXPECT_TRUE(model.rootFolders().isEmpty());
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, rootsAreLazyAndSorted)
{
    TempTree tt;
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);

    ProjectFile project = ProjectFile::fromFolder(tt.root, "Demo");
    project.addFolder("b_folder", "Second");
    model.setProject(project);

    ASSERT_EQ(model.rowCount(), 2);
    QModelIndex root0 = model.index(0, 0);
    QModelIndex root1 = model.index(1, 0);

    // roots are not read yet, but show an expand indicator
    EXPECT_EQ(model.rowCount(root0), 0);
    EXPECT_TRUE(model.hasChildren(root0));
    EXPECT_TRUE(model.canFetchMore(root0));
    EXPECT_EQ(root0.data().toString(), QFileInfo(tt.root).fileName()); // no name given
    EXPECT_EQ(root1.data().toString(), "Second");
    EXPECT_EQ(model.filePath(root1), tt.path("b_folder"));
    EXPECT_EQ(model.nodeType(root1), ProjectTreeModel::RootFolderNode);
    EXPECT_TRUE(model.isDir(root1));
    EXPECT_FALSE(model.parent(root1).isValid());

    QSignalSpy spy(&model, &ProjectTreeModel::folderLoaded);
    model.fetchMore(root0);
    EXPECT_FALSE(model.canFetchMore(root0));
    EXPECT_EQ(spy.count(), 1);

    // default excludes of the project hide __pycache__ and *.pyc, hidden files are off
    EXPECT_EQ(
        childNames(model, root0),
        QStringList() << "A_folder" << "b_folder" << "image.png" << "main.py" << "notes.txt");

    QModelIndex mainPy = model.index(3, 0, root0);
    EXPECT_EQ(model.filePath(mainPy), tt.path("main.py"));
    EXPECT_EQ(model.nodeType(mainPy), ProjectTreeModel::FileNode);
    EXPECT_FALSE(model.hasChildren(mainPy));
    EXPECT_FALSE(model.canFetchMore(mainPy));
    EXPECT_EQ(model.parent(mainPy), root0);
    EXPECT_EQ(mainPy.data(ProjectTreeModel::FilePathRole).toString(), tt.path("main.py"));
    EXPECT_FALSE(mainPy.data(ProjectTreeModel::IsDirRole).toBool());
    EXPECT_FALSE(mainPy.data(Qt::DecorationRole).isNull());

    QModelIndex bFolder = model.index(1, 0, root0);
    EXPECT_EQ(model.nodeType(bFolder), ProjectTreeModel::FolderNode);
    EXPECT_TRUE(model.hasChildren(bFolder));
    model.fetchMore(bFolder);
    EXPECT_EQ(childNames(model, bFolder), QStringList() << "sub.py");
    EXPECT_EQ(model.parent(model.index(0, 0, bFolder)), bFolder);
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, excludesNameFiltersAndHiddenFiles)
{
    TempTree tt;
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);

    QList<ProjectFolder> folders;
    ProjectFolder f;
    f.path = tt.root;
    folders << f;
    model.setRootFolders(folders);
    QModelIndex root = loadedRoot(model);

    // no excludes at all
    EXPECT_EQ(
        childNames(model, root),
        QStringList() << "__pycache__" << "A_folder" << "b_folder" << "cache.pyc" << "image.png"
                      << "main.py" << "notes.txt");

    model.setExcludes(QStringList() << "*.pyc" << "__pycache__" << "a_*"); // case insensitive
    EXPECT_EQ(
        childNames(model, root),
        QStringList() << "b_folder" << "image.png" << "main.py" << "notes.txt");

    // name filters only apply to files
    model.setNameFilters(QStringList() << "*.py" << "*.txt");
    EXPECT_EQ(childNames(model, root), QStringList() << "b_folder" << "main.py" << "notes.txt");

    model.setNameFilters(QStringList());
    model.setShowHiddenFiles(true);
    EXPECT_EQ(
        childNames(model, root),
        QStringList() << "b_folder" << ".hidden" << "image.png" << "main.py" << "notes.txt");

    model.setShowHiddenFiles(false);
    EXPECT_EQ(
        childNames(model, root),
        QStringList() << "b_folder" << "image.png" << "main.py" << "notes.txt");
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, refreshAppliesFileSystemChanges)
{
    TempTree tt;
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);
    model.setProject(ProjectFile::fromFolder(tt.root));
    QModelIndex root = loadedRoot(model);
    QModelIndex bFolder = model.indexForPath(tt.path("b_folder"), true);
    ASSERT_TRUE(bFolder.isValid());
    model.fetchMore(bFolder);
    EXPECT_EQ(childNames(model, bFolder), QStringList() << "sub.py");

    QSignalSpy inserted(&model, &QAbstractItemModel::rowsInserted);
    QSignalSpy removed(&model, &QAbstractItemModel::rowsRemoved);

    QDir d(tt.root);
    touch(d.filePath("aaa.py"));
    touch(d.filePath("zzz.py"));
    ASSERT_TRUE(QFile::remove(d.filePath("notes.txt")));
    ASSERT_TRUE(d.mkdir("c_folder"));
    touch(d.filePath("b_folder/other.txt"));

    model.refresh();

    EXPECT_EQ(
        childNames(model, root),
        QStringList() << "A_folder" << "b_folder" << "c_folder" << "aaa.py" << "image.png"
                      << "main.py" << "zzz.py");
    EXPECT_EQ(childNames(model, bFolder), QStringList() << "other.txt" << "sub.py");
    EXPECT_EQ(inserted.count(), 4); // c_folder, aaa.py, zzz.py, b_folder/other.txt
    EXPECT_EQ(removed.count(), 1); // notes.txt

    // the persistent index of an unchanged entry stays valid
    QPersistentModelIndex mainPy(model.indexForPath(tt.path("main.py")));
    model.refresh(root);
    EXPECT_TRUE(mainPy.isValid());
    EXPECT_EQ(model.filePath(mainPy), tt.path("main.py"));

    // a file that becomes a folder is replaced
    ASSERT_TRUE(QFile::remove(d.filePath("zzz.py")));
    ASSERT_TRUE(d.mkdir("zzz.py"));
    model.refresh(root);
    EXPECT_TRUE(model.isDir(model.indexForPath(tt.path("zzz.py"))));
    EXPECT_EQ(childNames(model, root)[3], "zzz.py");
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, watcherRefreshesLoadedFolders)
{
    TempTree tt;
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);
    model.setProject(ProjectFile::fromFolder(tt.root));
    QModelIndex root = loadedRoot(model);
    const int before = model.rowCount(root);

    touch(QDir(tt.root).filePath("watched.py"));

    // the file system watcher delivers the change through the event loop
    bool ok = QTest::qWaitFor([&]() { return model.rowCount(root) == before + 1; }, 5000);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(model.indexForPath(tt.path("watched.py")).isValid());
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, indexForPath)
{
    TempTree tt;
    ProjectTreeModel model;
    model.setProject(ProjectFile::fromFolder(tt.root));

    EXPECT_FALSE(model.indexForPath("").isValid());
    EXPECT_FALSE(model.indexForPath("/no/such/root").isValid());
    EXPECT_EQ(model.indexForPath(tt.root), model.index(0, 0));

    // not loaded yet: without load nothing is found, with load the folders are read
    EXPECT_FALSE(model.indexForPath(tt.path("b_folder/sub.py")).isValid());
    EXPECT_EQ(model.rowCount(model.index(0, 0)), 0);
    QModelIndex sub = model.indexForPath(tt.path("b_folder/sub.py"), true);
    ASSERT_TRUE(sub.isValid());
    EXPECT_EQ(model.filePath(sub), tt.path("b_folder/sub.py"));
    EXPECT_EQ(model.filePath(model.parent(sub)), tt.path("b_folder"));

    // excluded and missing entries are not found
    EXPECT_FALSE(model.indexForPath(tt.path("cache.pyc"), true).isValid());
    EXPECT_FALSE(model.indexForPath(tt.path("missing.py"), true).isValid());
    EXPECT_FALSE(model.indexForPath(tt.path("main.py/x"), true).isValid());

    // native separators and trailing slash are accepted
    EXPECT_EQ(model.indexForPath(QDir::toNativeSeparators(tt.path("b_folder")) + QDir::separator()),
              model.parent(sub));
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, nestedRootsPreferTheLongestMatch)
{
    TempTree tt;
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);
    ProjectFile project = ProjectFile::fromFolder(tt.root);
    project.addFolder("b_folder");
    model.setProject(project);

    QModelIndex sub = model.indexForPath(tt.path("b_folder/sub.py"), true);
    ASSERT_TRUE(sub.isValid());
    EXPECT_EQ(model.parent(sub), model.index(1, 0)); // the second root, not root/b_folder
    EXPECT_EQ(model.rowCount(model.index(0, 0)), 0); // first root was not loaded

    model.clear();
    EXPECT_EQ(model.rowCount(), 0);
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeModel, missingRootFolderHasNoChildren)
{
    ProjectTreeModel model;
    QAbstractItemModelTester tester(&model, QAbstractItemModelTester::FailureReportingMode::Fatal);
    QList<ProjectFolder> folders;
    ProjectFolder f;
    f.path = "/this/folder/does/not/exist";
    f.name = "Gone";
    folders << f;
    model.setRootFolders(folders);

    QModelIndex root = loadedRoot(model);
    EXPECT_EQ(root.data().toString(), "Gone");
    EXPECT_EQ(model.rowCount(root), 0);
    EXPECT_FALSE(model.hasChildren(root));
    EXPECT_FALSE(model.canFetchMore(root));
}
