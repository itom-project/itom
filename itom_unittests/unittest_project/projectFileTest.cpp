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

#include "organizer/projectFile.h"

#include <qdir.h>
#include <qfile.h>
#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qtemporarydir.h>

using namespace ito;

namespace {

void writeFile(const QString& path, const QByteArray& content)
{
    QFile f(path);
    ASSERT_TRUE(f.open(QIODevice::WriteOnly)) << path.toStdString();
    f.write(content);
    f.close();
}

QByteArray readFile(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
    {
        return QByteArray();
    }
    return f.readAll();
}

struct TempProject
{
    QTemporaryDir dir;
    QString root; // normalized path of the temp dir

    TempProject() : root(ProjectFile::normalize(dir.path()))
    {
        QDir d(root);
        d.mkpath("src/lib");
        d.mkpath("data");
        QFile f(d.filePath("src/main.py"));
        f.open(QIODevice::WriteOnly);
        f.close();
    }

    QString path(const QString& rel) const
    {
        return root + "/" + rel;
    }
};

} // namespace

//-------------------------------------------------------------------------------------
TEST(ProjectFile, normalizeCleansPaths)
{
    EXPECT_EQ(ProjectFile::normalize(""), "");
    EXPECT_EQ(ProjectFile::normalize("/a/b/../c/"), "/a/c");
    EXPECT_EQ(ProjectFile::normalize("/a//b"), "/a/b");
    EXPECT_EQ(ProjectFile::normalize("/"), "/");
    // relative paths are made absolute based on the cwd
    EXPECT_TRUE(QFileInfo(ProjectFile::normalize("x/y")).isAbsolute());
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, fromFolderCreatesUnsavedProject)
{
    TempProject tp;
    ProjectFile p = ProjectFile::fromFolder(tp.root);

    EXPECT_TRUE(p.isValid());
    EXPECT_EQ(p.projectDir(), tp.root);
    EXPECT_EQ(p.name(), QFileInfo(tp.root).fileName());
    EXPECT_EQ(p.filePath(), tp.root + "/" + p.name() + ".itomproject");
    ASSERT_EQ(p.folders().size(), 1);
    EXPECT_EQ(p.folders()[0].path, tp.root);
    EXPECT_EQ(p.workingDirectory(), tp.root);
    EXPECT_EQ(p.mainScript(), "");
    EXPECT_EQ(p.excludes(), ProjectFile::defaultExcludes());
    EXPECT_FALSE(QFile::exists(p.filePath()));

    ProjectFile named = ProjectFile::fromFolder(tp.root, "Demo");
    EXPECT_EQ(named.name(), "Demo");
    EXPECT_EQ(named.filePath(), tp.root + "/Demo.itomproject");
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, saveLoadRoundTrip)
{
    TempProject tp;
    ProjectFile p = ProjectFile::fromFolder(tp.root, "Round");
    EXPECT_TRUE(p.addFolder("data", "Data"));
    EXPECT_FALSE(p.addFolder(tp.path("data"))); // duplicate (absolute vs. relative)
    p.setMainScript("src/main.py");
    p.setWorkingDirectory("src");
    p.setPythonPath(QStringList() << "src/lib" << tp.path("src/lib") << "data");
    p.setExcludes(QStringList() << "*.tmp");

    RetVal ret = p.save();
    ASSERT_FALSE(ret.containsError()) << ret.errorMessage();
    ASSERT_TRUE(QFile::exists(p.filePath()));

    ProjectFile q;
    ret = q.load(p.filePath());
    ASSERT_FALSE(ret.containsWarningOrError()) << ret.errorMessage();

    EXPECT_EQ(q.filePath(), p.filePath());
    EXPECT_EQ(q.name(), "Round");
    ASSERT_EQ(q.folders().size(), 2);
    EXPECT_EQ(q.folders()[0].path, tp.root);
    EXPECT_EQ(q.folders()[0].name, "");
    EXPECT_EQ(q.folders()[1].path, tp.path("data"));
    EXPECT_EQ(q.folders()[1].name, "Data");
    EXPECT_EQ(q.mainScript(), tp.path("src/main.py"));
    EXPECT_EQ(q.workingDirectory(), tp.path("src"));
    EXPECT_EQ(q.pythonPath(), QStringList() << tp.path("src/lib") << tp.path("data"));
    EXPECT_EQ(q.excludes(), QStringList() << "*.tmp");
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, saveWritesRelativePaths)
{
    TempProject tp;
    ProjectFile p = ProjectFile::fromFolder(tp.root, "Rel");
    p.addFolder("data");
    p.setMainScript("src/main.py");
    p.setWorkingDirectory(tp.root);
    ASSERT_FALSE(p.save().containsError());

    QJsonObject root = QJsonDocument::fromJson(readFile(p.filePath())).object();
    EXPECT_EQ(root.value("version").toInt(), 1);
    EXPECT_EQ(root.value("name").toString(), "Rel");
    EXPECT_EQ(root.value("folders").toArray().size(), 2);
    EXPECT_EQ(root.value("folders").toArray()[0].toObject().value("path").toString(), ".");
    EXPECT_EQ(root.value("folders").toArray()[1].toObject().value("path").toString(), "data");
    EXPECT_EQ(root.value("mainScript").toString(), "src/main.py");
    EXPECT_EQ(root.value("workingDirectory").toString(), ".");
    EXPECT_FALSE(readFile(p.filePath()).contains(tp.root.toUtf8()));
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, loadMinimalFileUsesDefaults)
{
    TempProject tp;
    const QString file = tp.path("min.itomproject");
    writeFile(file, "{ \"version\": 1 }");

    ProjectFile p;
    RetVal ret = p.load(file);
    ASSERT_FALSE(ret.containsWarningOrError()) << ret.errorMessage();

    EXPECT_EQ(p.name(), "min"); // file name without extension
    ASSERT_EQ(p.folders().size(), 1);
    EXPECT_EQ(p.folders()[0].path, tp.root); // "." by default
    EXPECT_EQ(p.workingDirectory(), tp.root);
    EXPECT_EQ(p.mainScript(), "");
    EXPECT_TRUE(p.pythonPath().isEmpty());
    EXPECT_EQ(p.excludes(), ProjectFile::defaultExcludes());
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, loadAcceptsPlainStringFolders)
{
    TempProject tp;
    const QString file = tp.path("str.itomproject");
    writeFile(file, "{ \"version\": 1, \"folders\": [\"src\", \"data\"] }");

    ProjectFile p;
    ASSERT_FALSE(p.load(file).containsWarningOrError());
    ASSERT_EQ(p.folders().size(), 2);
    EXPECT_EQ(p.folders()[0].path, tp.path("src"));
    EXPECT_EQ(p.folders()[1].path, tp.path("data"));
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, loadWarnsAboutMissingFoldersButKeepsThem)
{
    TempProject tp;
    const QString file = tp.path("missing.itomproject");
    writeFile(file, "{ \"version\": 1, \"folders\": [\"src\", \"gone\"] }");

    ProjectFile p;
    RetVal ret = p.load(file);
    EXPECT_TRUE(ret.containsWarning());
    EXPECT_FALSE(ret.containsError());
    EXPECT_TRUE(QString(ret.errorMessage()).contains("gone"));
    ASSERT_EQ(p.folders().size(), 2);
    EXPECT_TRUE(p.isValid());
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, loadFailsOnMissingFileInvalidJsonAndUnknownVersion)
{
    TempProject tp;
    ProjectFile p = ProjectFile::fromFolder(tp.root, "keep");
    const QString oldFilePath = p.filePath();

    EXPECT_TRUE(p.load(tp.path("does_not_exist.itomproject")).containsError());

    writeFile(tp.path("broken.itomproject"), "{ not json");
    EXPECT_TRUE(p.load(tp.path("broken.itomproject")).containsError());

    writeFile(tp.path("array.itomproject"), "[1, 2]");
    EXPECT_TRUE(p.load(tp.path("array.itomproject")).containsError());

    writeFile(tp.path("v2.itomproject"), "{ \"version\": 2 }");
    EXPECT_TRUE(p.load(tp.path("v2.itomproject")).containsError());

    writeFile(tp.path("v0.itomproject"), "{ \"name\": \"no version\" }");
    EXPECT_TRUE(p.load(tp.path("v0.itomproject")).containsError());

    // a failed load does not change the object
    EXPECT_EQ(p.filePath(), oldFilePath);
    EXPECT_EQ(p.name(), "keep");
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, saveFailsWithoutFileNameOrUnwritableDir)
{
    ProjectFile p;
    EXPECT_FALSE(p.isValid());
    EXPECT_TRUE(p.save().containsError());

    TempProject tp;
    EXPECT_TRUE(p.save(tp.path("no_such_dir/x.itomproject")).containsError());
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, foldersCanBeRemovedAndQueried)
{
    TempProject tp;
    ProjectFile p = ProjectFile::fromFolder(tp.root);
    p.addFolder("data");

    EXPECT_TRUE(p.contains(tp.path("src/main.py")));
    EXPECT_TRUE(p.contains(tp.root));
    EXPECT_FALSE(p.contains(tp.root + "_other/file.py"));
    EXPECT_FALSE(p.contains(QFileInfo(tp.root).absolutePath()));

    EXPECT_TRUE(p.removeFolder(tp.root));
    EXPECT_FALSE(p.removeFolder(tp.root));
    ASSERT_EQ(p.folders().size(), 1);
    EXPECT_EQ(p.folders()[0].path, tp.path("data"));
    EXPECT_TRUE(p.contains(tp.path("data/x.txt")));
    EXPECT_FALSE(p.contains(tp.path("src/main.py")));
}

//-------------------------------------------------------------------------------------
TEST(ProjectFile, relativeAndAbsoluteConversion)
{
    TempProject tp;
    ProjectFile p = ProjectFile::fromFolder(tp.root);

    EXPECT_EQ(p.toAbsolute("src/../data"), tp.path("data"));
    EXPECT_EQ(p.toAbsolute(tp.path("data")), tp.path("data"));
    EXPECT_EQ(p.toAbsolute(""), "");
    EXPECT_EQ(p.toRelative(tp.root), ".");
    EXPECT_EQ(p.toRelative(tp.path("src/main.py")), "src/main.py");
    EXPECT_EQ(p.toRelative(QFileInfo(tp.root).absolutePath() + "/other"), "../other");
    EXPECT_EQ(p.toRelative(""), "");

    // without a project file, paths stay absolute
    ProjectFile empty;
    EXPECT_EQ(empty.toRelative(tp.root), tp.root);
}
