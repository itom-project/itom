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

// Headless tests of ProjectTreeModel inside a real QTreeView (platform "offscreen"):
// lazy loading through QTreeView::expand, keyboard navigation with QTest, selection
// and the behaviour of expanded nodes and the selection during refresh().

#include "gtest/gtest.h"

#include "helper/fileOperations.h"
#include "models/projectTreeModel.h"
#include "organizer/projectFile.h"

#include <qdir.h>
#include <qfile.h>
#include <qtemporarydir.h>
#include <qtest.h>
#include <qtreeview.h>

using namespace ito;

namespace {

void touch(const QString& path)
{
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly)) << path.toStdString();
    f.close();
}

struct ViewFixture
{
    QTemporaryDir dir;
    QString root;
    ProjectTreeModel model;
    QTreeView view;

    ViewFixture() : root(ProjectFile::normalize(dir.path()))
    {
        QDir d(root);
        d.mkpath("src/pkg");
        d.mkdir("data");
        touch(d.filePath("src/main.py"));
        touch(d.filePath("src/pkg/__init__.py"));
        touch(d.filePath("readme.txt"));

        model.setProject(ProjectFile::fromFolder(root, "View"));
        view.setModel(&model);
        view.setSelectionMode(QAbstractItemView::ExtendedSelection);
        view.resize(300, 400);
        view.show();
        QTest::qWaitForWindowExposed(&view);
    }

    QString path(const QString& rel) const
    {
        return root + "/" + rel;
    }

    QStringList visibleNames(const QModelIndex& parent) const
    {
        QStringList names;

        for (int r = 0; r < model.rowCount(parent); ++r)
        {
            names << model.index(r, 0, parent).data().toString();
        }

        return names;
    }
};

} // namespace

//-------------------------------------------------------------------------------------
TEST(ProjectTreeView, expandLoadsLazily)
{
    ViewFixture f;
    const QModelIndex root = f.model.index(0, 0);

    EXPECT_EQ(f.model.rowCount(root), 0); // nothing read yet
    f.view.expand(root);
    QApplication::processEvents();

    EXPECT_TRUE(f.view.isExpanded(root));
    EXPECT_EQ(f.visibleNames(root), QStringList() << "data" << "src" << "readme.txt");

    // sub folders are still unread until they are expanded
    const QModelIndex src = f.model.indexForPath(f.path("src"));
    ASSERT_TRUE(src.isValid());
    EXPECT_EQ(f.model.rowCount(src), 0);
    f.view.expand(src);
    QApplication::processEvents();
    EXPECT_EQ(f.visibleNames(src), QStringList() << "pkg" << "main.py");
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeView, keyboardNavigationExpandsAndSelects)
{
    ViewFixture f;
    const QModelIndex root = f.model.index(0, 0);
    f.view.setCurrentIndex(root);
    f.view.setFocus();

    QTest::keyClick(&f.view, Qt::Key_Right); // expand root
    QApplication::processEvents();
    EXPECT_TRUE(f.view.isExpanded(root));

    QTest::keyClick(&f.view, Qt::Key_Down); // "data"
    QTest::keyClick(&f.view, Qt::Key_Down); // "src"
    EXPECT_EQ(f.model.filePath(f.view.currentIndex()), f.path("src"));

    QTest::keyClick(&f.view, Qt::Key_Right); // expand "src"
    QTest::keyClick(&f.view, Qt::Key_Down); // "pkg"
    QTest::keyClick(&f.view, Qt::Key_Down); // "main.py"
    EXPECT_EQ(f.model.filePath(f.view.currentIndex()), f.path("src/main.py"));

    // extend the selection upwards: main.py and pkg
    QTest::keyClick(&f.view, Qt::Key_Up, Qt::ShiftModifier);
    const QModelIndexList rows =
        FileOperations::uniqueRowIndexes(f.view.selectionModel()->selectedIndexes());
    ASSERT_EQ(rows.size(), 2);

    QStringList selected;
    foreach (const QModelIndex& idx, rows)
    {
        selected << f.model.filePath(idx);
    }
    selected.sort();
    EXPECT_EQ(selected, QStringList() << f.path("src/main.py") << f.path("src/pkg"));
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeView, refreshKeepsExpansionAndSelection)
{
    ViewFixture f;
    const QModelIndex root = f.model.index(0, 0);
    f.view.expand(root);
    const QModelIndex src = f.model.indexForPath(f.path("src"));
    f.view.expand(src);
    QApplication::processEvents();

    const QModelIndex mainPy = f.model.indexForPath(f.path("src/main.py"));
    f.view.setCurrentIndex(mainPy);
    f.view.selectionModel()->select(mainPy, QItemSelectionModel::ClearAndSelect);

    // changes on disk: new file before main.py, a new folder, readme removed
    QDir d(f.root);
    touch(d.filePath("src/a.py"));
    ASSERT_TRUE(d.mkdir("build"));
    ASSERT_TRUE(QFile::remove(d.filePath("readme.txt")));

    f.model.refresh();
    QApplication::processEvents();

    EXPECT_TRUE(f.view.isExpanded(root));
    EXPECT_TRUE(f.view.isExpanded(f.model.indexForPath(f.path("src"))));
    EXPECT_EQ(f.visibleNames(root), QStringList() << "build" << "data" << "src");
    EXPECT_EQ(
        f.visibleNames(f.model.indexForPath(f.path("src"))),
        QStringList() << "pkg" << "a.py" << "main.py");

    // the selection and the current index still point to main.py (now in row 2)
    EXPECT_EQ(f.model.filePath(f.view.currentIndex()), f.path("src/main.py"));
    EXPECT_EQ(f.view.currentIndex().row(), 2);
    const QModelIndexList rows =
        FileOperations::uniqueRowIndexes(f.view.selectionModel()->selectedIndexes());
    ASSERT_EQ(rows.size(), 1);
    EXPECT_EQ(f.model.filePath(rows.first()), f.path("src/main.py"));
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeView, renameOnDiskIsPickedUpByRefresh)
{
    ViewFixture f;
    const QModelIndex root = f.model.index(0, 0);
    f.view.expand(root);
    QApplication::processEvents();

    const QString newName = FileOperations::uniqueChildName(f.root, "readme", ".txt");
    EXPECT_EQ(newName, "readme (1).txt"); // readme.txt exists
    ASSERT_TRUE(QFile::rename(f.path("readme.txt"), f.root + "/" + newName));

    f.model.refresh(root);
    EXPECT_FALSE(f.model.indexForPath(f.path("readme.txt")).isValid());
    EXPECT_TRUE(f.model.indexForPath(f.root + "/" + newName).isValid());
    EXPECT_EQ(f.visibleNames(root), QStringList() << "data" << "src" << newName);
}

//-------------------------------------------------------------------------------------
TEST(ProjectTreeView, closingProjectClearsView)
{
    ViewFixture f;
    f.view.expand(f.model.index(0, 0));
    QApplication::processEvents();
    EXPECT_GT(f.model.rowCount(f.model.index(0, 0)), 0);

    f.model.clear();
    QApplication::processEvents();
    EXPECT_EQ(f.model.rowCount(), 0);
    EXPECT_FALSE(f.view.currentIndex().isValid());

    // opening again works
    f.model.setProject(ProjectFile::fromFolder(f.root));
    EXPECT_EQ(f.model.rowCount(), 1);
    f.view.expand(f.model.index(0, 0));
    QApplication::processEvents();
    EXPECT_EQ(f.visibleNames(f.model.index(0, 0)), QStringList() << "data" << "src" << "readme.txt");
}
