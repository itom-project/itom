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

#include "organizer/projectOrganizer.h"
#include "organizer/projectSession.h"

#include <qdir.h>
#include <qfile.h>
#include <qsettings.h>
#include <qsignalspy.h>
#include <qtemporarydir.h>

using namespace ito;

namespace ito {
namespace test {
QString settingsFilePath();
}
} // namespace ito

namespace {

struct CwdRestore
{
    QString cwd;
    CwdRestore() : cwd(QDir::currentPath())
    {
    }
    ~CwdRestore()
    {
        QDir::setCurrent(cwd);
    }
};

void clearProjectSettings()
{
    QSettings settings(ito::test::settingsFilePath(), QSettings::IniFormat);
    settings.remove("Projects");
    settings.sync();
}

} // namespace

//-------------------------------------------------------------------------------------
TEST(ProjectOrganizer, openSaveCloseEmitSignalsAndChangeCwd)
{
    CwdRestore cwd;
    clearProjectSettings();
    QTemporaryDir dir;
    const QString root = ProjectFile::normalize(dir.path());
    QDir(root).mkdir("work");

    ProjectOrganizer org;
    EXPECT_FALSE(org.hasProject());
    EXPECT_TRUE(org.recentProjects().isEmpty());
    EXPECT_TRUE(org.setWorkingDirectoryOnOpen());

    QSignalSpy opened(&org, &ProjectOrganizer::projectOpened);
    QSignalSpy closed(&org, &ProjectOrganizer::projectClosed);
    QSignalSpy changed(&org, &ProjectOrganizer::projectChanged);
    QSignalSpy recent(&org, &ProjectOrganizer::recentProjectsChanged);
    QSignalSpy cwdChanged(&org, &ProjectOrganizer::workingDirectoryChanged);

    // new project from folder: open, not saved, cwd set to the folder
    RetVal ret = org.newProjectFromFolder(root, "Demo");
    ASSERT_FALSE(ret.containsError()) << ret.errorMessage();
    EXPECT_TRUE(org.hasProject());
    EXPECT_EQ(org.project().name(), "Demo");
    EXPECT_EQ(opened.count(), 1);
    EXPECT_EQ(opened.last().at(0).toString(), root + "/Demo.itomproject");
    EXPECT_EQ(cwdChanged.count(), 1);
    EXPECT_EQ(QDir::cleanPath(QDir::currentPath()), root);
    EXPECT_TRUE(org.recentProjects().isEmpty()); // not saved yet
    EXPECT_FALSE(QFile::exists(org.project().filePath()));

    // modify and save: file exists, recent list and projectChanged
    org.project().setWorkingDirectory("work");
    org.projectModified();
    EXPECT_EQ(changed.count(), 1);
    ret = org.saveProject();
    ASSERT_FALSE(ret.containsError()) << ret.errorMessage();
    EXPECT_TRUE(QFile::exists(root + "/Demo.itomproject"));
    EXPECT_EQ(org.recentProjects(), QStringList() << root + "/Demo.itomproject");
    EXPECT_EQ(recent.count(), 1);
    EXPECT_EQ(changed.count(), 2);

    // close
    ret = org.closeProject();
    EXPECT_FALSE(ret.containsError());
    EXPECT_FALSE(org.hasProject());
    EXPECT_EQ(closed.count(), 1);
    EXPECT_EQ(closed.last().at(0).toString(), root + "/Demo.itomproject");
    EXPECT_FALSE(org.closeProject().containsError()); // closing twice is fine
    EXPECT_EQ(closed.count(), 1);

    // reopen from file: working directory from the file is applied
    QDir::setCurrent(dir.path());
    ret = org.openProject(root + "/Demo.itomproject");
    ASSERT_FALSE(ret.containsWarningOrError()) << ret.errorMessage();
    EXPECT_TRUE(org.hasProject());
    EXPECT_EQ(opened.count(), 2);
    EXPECT_EQ(QDir::cleanPath(QDir::currentPath()), root + "/work");
    EXPECT_EQ(cwdChanged.count(), 2);

    // opening another project closes the first one
    QDir(root).mkdir("other");
    ret = org.newProjectFromFolder(root + "/other");
    ASSERT_FALSE(ret.containsError());
    EXPECT_EQ(closed.count(), 2);
    EXPECT_EQ(opened.count(), 3);
    EXPECT_EQ(org.project().name(), "other");
}

//-------------------------------------------------------------------------------------
TEST(ProjectOrganizer, errorsAndRecentList)
{
    CwdRestore cwd;
    clearProjectSettings();
    QTemporaryDir dir;
    const QString root = ProjectFile::normalize(dir.path());

    ProjectOrganizer org;
    QSignalSpy opened(&org, &ProjectOrganizer::projectOpened);

    EXPECT_TRUE(org.saveProject().containsError()); // no project
    EXPECT_TRUE(org.newProjectFromFolder(root + "/missing").containsError());
    EXPECT_TRUE(org.openProject(root + "/missing.itomproject").containsError());
    EXPECT_FALSE(org.hasProject());
    EXPECT_EQ(opened.count(), 0);

    // fill the recent list with saved projects
    for (int i = 0; i < ProjectOrganizer::maxRecentProjects + 2; ++i)
    {
        QDir(root).mkdir(QString("p%1").arg(i));
        ASSERT_FALSE(org.newProjectFromFolder(root + QString("/p%1").arg(i)).containsError());
        ASSERT_FALSE(org.saveProject().containsError());
    }

    QStringList recent = org.recentProjects();
    EXPECT_EQ(recent.size(), ProjectOrganizer::maxRecentProjects);
    EXPECT_EQ(recent.first(), root + "/p11/p11.itomproject"); // newest first
    EXPECT_EQ(recent.last(), root + "/p2/p2.itomproject");

    // opening an existing entry moves it to the front
    ASSERT_FALSE(org.openProject(root + "/p5/p5.itomproject").containsError());
    EXPECT_EQ(org.recentProjects().first(), root + "/p5/p5.itomproject");
    EXPECT_EQ(org.recentProjects().size(), ProjectOrganizer::maxRecentProjects);

    // a project file that cannot be opened is removed from the list
    ASSERT_TRUE(QFile::remove(root + "/p7/p7.itomproject"));
    EXPECT_TRUE(org.openProject(root + "/p7/p7.itomproject").containsError());
    EXPECT_FALSE(org.recentProjects().contains(root + "/p7/p7.itomproject"));
    EXPECT_TRUE(org.hasProject()); // p5 is still open

    // the list is persisted in the settings and read by a new organizer
    org.setSetWorkingDirectoryOnOpen(false);
    {
        ProjectOrganizer org2;
        EXPECT_EQ(org2.recentProjects(), org.recentProjects());
        EXPECT_FALSE(org2.setWorkingDirectoryOnOpen());

        // no cwd change if the option is off
        const QString before = QDir::currentPath();
        QSignalSpy cwdChanged(&org2, &ProjectOrganizer::workingDirectoryChanged);
        ASSERT_FALSE(org2.openProject(root + "/p3/p3.itomproject").containsError());
        EXPECT_EQ(QDir::currentPath(), before);
        EXPECT_EQ(cwdChanged.count(), 0);
    }

    org.clearRecentProjects();
    EXPECT_TRUE(org.recentProjects().isEmpty());
}

//-------------------------------------------------------------------------------------
TEST(ProjectOrganizer, settingsGroupIsStableAndPathNormalized)
{
    const QString g1 = ProjectOrganizer::settingsGroup("/a/b/x.itomproject");
    const QString g2 = ProjectOrganizer::settingsGroup("/a/c/../b/x.itomproject");
    const QString g3 = ProjectOrganizer::settingsGroup("/a/b/y.itomproject");
    EXPECT_TRUE(g1.startsWith("Projects/"));
    EXPECT_EQ(g1, g2);
    EXPECT_NE(g1, g3);
    EXPECT_FALSE(g1.contains(' '));
}

//-------------------------------------------------------------------------------------
TEST(ProjectOrganizer, reloadSettingsReadsOptions)
{
    clearProjectSettings();
    ProjectOrganizer org;
    EXPECT_TRUE(org.restoreScripts());
    EXPECT_TRUE(org.setWorkingDirectoryOnOpen());

    {
        QSettings settings(ito::test::settingsFilePath(), QSettings::IniFormat);
        settings.beginGroup("Projects");
        settings.setValue("restoreScripts", false);
        settings.setValue("setWorkingDirectoryOnOpen", false);
        settings.endGroup();
    }

    org.reloadSettings();
    EXPECT_FALSE(org.restoreScripts());
    EXPECT_FALSE(org.setWorkingDirectoryOnOpen());
}

//-------------------------------------------------------------------------------------
TEST(ProjectOrganizer, lastProjectIsRemembered)
{
    CwdRestore cwd;
    clearProjectSettings();
    QTemporaryDir dir;
    const QString root = ProjectFile::normalize(dir.path());
    const QString file = root + "/last.itomproject";

    {
        ProjectOrganizer org;
        EXPECT_FALSE(org.reopenLastProject());
        EXPECT_TRUE(org.lastProject().isEmpty());

        ASSERT_FALSE(org.newProjectFromFolder(root, "last").containsError());
        EXPECT_TRUE(org.lastProject().isEmpty()); // not saved yet
        ASSERT_FALSE(org.saveProject().containsError());
        EXPECT_EQ(org.lastProject(), file);
        // destructor saves the settings while the project is open
    }

    {
        ProjectOrganizer org;
        EXPECT_EQ(org.lastProject(), file);
        ASSERT_FALSE(org.openProject(file).containsError());
        org.closeProject();
        EXPECT_TRUE(org.lastProject().isEmpty());
    }

    ProjectOrganizer org;
    EXPECT_TRUE(org.lastProject().isEmpty());
}

//-------------------------------------------------------------------------------------
TEST(ProjectOrganizer, semaphoreReceivesReturnValue)
{
    CwdRestore cwd;
    clearProjectSettings();
    QTemporaryDir dir;
    ProjectOrganizer org;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    org.openProject(dir.path() + "/none.itomproject", locker.getSemaphore());
    EXPECT_TRUE(locker.getSemaphore()->wait(1000));
    EXPECT_TRUE(locker.getSemaphore()->returnValue.containsError());

    ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());
    org.newProjectFromFolder(dir.path(), QString(), locker2.getSemaphore());
    EXPECT_TRUE(locker2.getSemaphore()->wait(1000));
    EXPECT_FALSE(locker2.getSemaphore()->returnValue.containsError());
    EXPECT_TRUE(org.hasProject());
}

//-------------------------------------------------------------------------------------
TEST(ProjectSession, sysPathStatements)
{
    EXPECT_TRUE(ProjectSession::addToSysPathStatement(QStringList()).isEmpty());
    EXPECT_TRUE(ProjectSession::removeFromSysPathStatement(QStringList()).isEmpty());

    QStringList paths;
    paths << "/a/lib" << "/b/it's";
    const QString add = ProjectSession::addToSysPathStatement(paths);
    // reversed order, so that /a/lib ends up first in sys.path
    EXPECT_TRUE(add.contains("insert(0, _itom_p)"));
    EXPECT_LT(add.indexOf("it\\'s"), add.indexOf("/a/lib"));
    EXPECT_TRUE(add.contains("not in _itom_sys.path"));

    const QString remove = ProjectSession::removeFromSysPathStatement(paths);
    EXPECT_TRUE(remove.contains("_itom_sys.path.remove(_itom_p)"));
    EXPECT_LT(remove.indexOf("/a/lib"), remove.indexOf("it\\'s"));
}
