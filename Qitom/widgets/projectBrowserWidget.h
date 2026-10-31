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

#include "../models/projectTreeModel.h"

#include <qaction.h>
#include <qlabel.h>
#include <qmenu.h>
#include <qprocess.h>
#include <qpushbutton.h>
#include <qstackedwidget.h>
#include <qtreeview.h>
#include <qwidget.h>

namespace ito {

class ProjectOrganizer;

//! Project view of the file system dock widget (page "Project").
/*!
    Shows the root folders of the project opened in the ProjectOrganizer as a tree
    (ProjectTreeModel). Without a project a placeholder with buttons to open or create
    a project is shown. All file operations work directly on the file system; the model
    picks up the changes through its file system watcher.
*/
class ProjectBrowserWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ProjectBrowserWidget(QWidget* parent = nullptr);
    ~ProjectBrowserWidget();

    ProjectTreeModel* model() const
    {
        return m_pModel;
    }

    QTreeView* treeView() const
    {
        return m_pTreeView;
    }

    //! paths of the selected entries (one per row)
    QStringList selectedPaths() const;

    //! the folder of the current item (the item itself for folders, its parent for files,
    //! the first root folder if nothing is selected). Empty if there is no project.
    QString currentFolder() const;

signals:
    void openProjectRequested();
    void newProjectRequested();

    //! the user wants to see the folder in the file system view
    void showInFileSystemRequested(const QString& folder);

    //! the current working directory was changed by this widget
    void workingDirectoryChanged();

public slots:
    void projectOpened();
    void projectClosed(const QString& filePath);
    void projectChanged();
    void updateActions();

    //! stores the expanded folders of the current project in the settings
    void saveExpandedState();

private slots:
    void itemActivated(const QModelIndex& index);
    void contextMenuRequested(const QPoint& pos);
    void mnuOpenFile();
    void mnuOpenInEditor();
    void mnuExecuteFile();
    void mnuLocateOnDisk();
    void mnuRenameItem();
    void mnuDeleteItems();
    void mnuNewDir();
    void mnuNewPyFile();
    void mnuRefresh();
    void mnuAddFolder();
    void mnuRemoveFolder();
    void mnuSetWorkingDirectory();
    void mnuSetMainScript();
    void mnuRunMainScript();
    void mnuShowInFileSystem();
    void processError(QProcess::ProcessError error);

private:
    void createActions();
    void createMenus();
    ProjectOrganizer* organizer() const;
    void updateHeader();
    void showError(const QString& title, const QString& message);
    void restoreExpandedState();
    void collectExpanded(const QModelIndex& parent, QStringList& expanded) const;

    ProjectTreeModel* m_pModel;
    QTreeView* m_pTreeView;
    QStackedWidget* m_pStack;
    QWidget* m_pNoProjectPage;
    QWidget* m_pProjectPage;
    QLabel* m_pLblProjectName;
    QLabel* m_pLblMainScript;
    QMenu* m_pContextMenu;

    QAction* m_pActOpenFile;
    QAction* m_pActOpenInEditor;
    QAction* m_pActExecuteFile;
    QAction* m_pActLocateOnDisk;
    QAction* m_pActRenameItem;
    QAction* m_pActDeleteItems;
    QAction* m_pActNewDir;
    QAction* m_pActNewPyFile;
    QAction* m_pActRefresh;
    QAction* m_pActAddFolder;
    QAction* m_pActRemoveFolder;
    QAction* m_pActSetWorkingDir;
    QAction* m_pActSetMainScript;
    QAction* m_pActRunMainScript;
    QAction* m_pActShowInFileSystem;
};

} // namespace ito
