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

#include "projectBrowserWidget.h"

#include "../AppManagement.h"
#include "../helper/IOHelper.h"
#include "../helper/editorFileType.h"
#include "../helper/fileOperations.h"
#include "../organizer/projectOrganizer.h"
#include "../organizer/userOrganizer.h"
#include "../python/pythonEngine.h"

#include <qboxlayout.h>
#include <qdir.h>
#include <qfiledialog.h>
#include <qheaderview.h>
#include <qinputdialog.h>
#include <qmessagebox.h>
#include <qset.h>
#include <qsettings.h>

namespace ito {

//-------------------------------------------------------------------------------------
ProjectBrowserWidget::ProjectBrowserWidget(QWidget* parent) :
    QWidget(parent), m_pModel(nullptr), m_pTreeView(nullptr), m_pStack(nullptr),
    m_pNoProjectPage(nullptr), m_pProjectPage(nullptr), m_pLblProjectName(nullptr),
    m_pLblMainScript(nullptr), m_pContextMenu(nullptr), m_pActOpenFile(nullptr),
    m_pActOpenInEditor(nullptr), m_pActExecuteFile(nullptr), m_pActLocateOnDisk(nullptr),
    m_pActRenameItem(nullptr), m_pActDeleteItems(nullptr), m_pActNewDir(nullptr),
    m_pActNewPyFile(nullptr), m_pActRefresh(nullptr), m_pActAddFolder(nullptr),
    m_pActRemoveFolder(nullptr), m_pActSetWorkingDir(nullptr), m_pActSetMainScript(nullptr),
    m_pActRunMainScript(nullptr), m_pActShowInFileSystem(nullptr)
{
    m_pModel = new ProjectTreeModel(this);

    // page without project
    m_pNoProjectPage = new QWidget(this);
    QLabel* lblNoProject = new QLabel(tr("No project opened."), m_pNoProjectPage);
    lblNoProject->setAlignment(Qt::AlignCenter);
    lblNoProject->setWordWrap(true);
    QPushButton* btnOpen = new QPushButton(tr("Open Project..."), m_pNoProjectPage);
    QPushButton* btnNew = new QPushButton(tr("New Project From Folder..."), m_pNoProjectPage);
    connect(btnOpen, &QPushButton::clicked, this, &ProjectBrowserWidget::openProjectRequested);
    connect(btnNew, &QPushButton::clicked, this, &ProjectBrowserWidget::newProjectRequested);

    QVBoxLayout* noProjectLayout = new QVBoxLayout(m_pNoProjectPage);
    noProjectLayout->addStretch();
    noProjectLayout->addWidget(lblNoProject);
    noProjectLayout->addWidget(btnOpen);
    noProjectLayout->addWidget(btnNew);
    noProjectLayout->addStretch();

    // page with project
    m_pProjectPage = new QWidget(this);
    m_pLblProjectName = new QLabel(m_pProjectPage);
    QFont boldFont = m_pLblProjectName->font();
    boldFont.setBold(true);
    m_pLblProjectName->setFont(boldFont);
    m_pLblMainScript = new QLabel(m_pProjectPage);

    m_pTreeView = new QTreeView(m_pProjectPage);
    m_pTreeView->setModel(m_pModel);
    m_pTreeView->setHeaderHidden(true);
    m_pTreeView->setIndentation(15);
    m_pTreeView->setAnimated(true);
    m_pTreeView->setUniformRowHeights(true);
    m_pTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
    m_pTreeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_pTreeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_pTreeView->setObjectName("projectTreeView");
    connect(m_pTreeView, &QTreeView::activated, this, &ProjectBrowserWidget::itemActivated);
    connect(
        m_pTreeView,
        &QTreeView::customContextMenuRequested,
        this,
        &ProjectBrowserWidget::contextMenuRequested);

    QVBoxLayout* projectLayout = new QVBoxLayout(m_pProjectPage);
    projectLayout->setContentsMargins(2, 2, 2, 0);
    projectLayout->setSpacing(1);
    projectLayout->addWidget(m_pLblProjectName);
    projectLayout->addWidget(m_pLblMainScript);
    projectLayout->addWidget(m_pTreeView);

    m_pStack = new QStackedWidget(this);
    m_pStack->addWidget(m_pNoProjectPage);
    m_pStack->addWidget(m_pProjectPage);

    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(m_pStack);

    createActions();
    createMenus();

    ProjectOrganizer* org = organizer();

    if (org)
    {
        connect(org, &ProjectOrganizer::projectOpened, this, &ProjectBrowserWidget::projectOpened);
        connect(org, &ProjectOrganizer::projectClosed, this, &ProjectBrowserWidget::projectClosed);
        connect(org, &ProjectOrganizer::projectChanged, this, &ProjectBrowserWidget::projectChanged);

        if (org->hasProject())
        {
            projectOpened();
        }
        else
        {
            projectClosed(QString());
        }
    }
    else
    {
        projectClosed(QString());
    }
}

//-------------------------------------------------------------------------------------
ProjectBrowserWidget::~ProjectBrowserWidget()
{
    saveExpandedState();
}

//-------------------------------------------------------------------------------------
ProjectOrganizer* ProjectBrowserWidget::organizer() const
{
    return qobject_cast<ProjectOrganizer*>(AppManagement::getProjectOrganizer());
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::createActions()
{
    ito::UserOrganizer* uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
    const bool developer = uOrg && uOrg->currentUserHasFeature(featDeveloper);

    if (developer)
    {
        m_pActOpenFile = new QAction(QIcon(":/files/icons/open.png"), tr("Open File"), this);
        connect(m_pActOpenFile, &QAction::triggered, this, &ProjectBrowserWidget::mnuOpenFile);
        m_pActOpenInEditor =
            new QAction(QIcon(":/files/icons/fileText.png"), tr("Open in Script Editor"), this);
        connect(
            m_pActOpenInEditor, &QAction::triggered, this, &ProjectBrowserWidget::mnuOpenInEditor);
        m_pActExecuteFile =
            new QAction(QIcon(":/script/icons/runScript.png"), tr("Execute File"), this);
        connect(m_pActExecuteFile, &QAction::triggered, this, &ProjectBrowserWidget::mnuExecuteFile);
        m_pActRunMainScript =
            new QAction(QIcon(":/script/icons/runScript.png"), tr("Run Main Script"), this);
        connect(
            m_pActRunMainScript, &QAction::triggered, this, &ProjectBrowserWidget::mnuRunMainScript);
        m_pActSetMainScript = new QAction(tr("Set as Main Script"), this);
        connect(
            m_pActSetMainScript, &QAction::triggered, this, &ProjectBrowserWidget::mnuSetMainScript);
    }

    m_pActLocateOnDisk = new QAction(QIcon(":/files/icons/browser.png"), tr("Locate On Disk"), this);
    connect(m_pActLocateOnDisk, &QAction::triggered, this, &ProjectBrowserWidget::mnuLocateOnDisk);
    m_pActRenameItem = new QAction(QIcon(":/workspace/icons/edit-rename.png"), tr("Rename..."), this);
    m_pActRenameItem->setShortcut(QKeySequence(tr("F2")));
    m_pActRenameItem->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    connect(m_pActRenameItem, &QAction::triggered, this, &ProjectBrowserWidget::mnuRenameItem);
    m_pActDeleteItems = new QAction(QIcon(":/editor/icons/editDelete.png"), tr("Delete"), this);
    m_pActDeleteItems->setShortcut(QKeySequence::Delete);
    m_pActDeleteItems->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    connect(m_pActDeleteItems, &QAction::triggered, this, &ProjectBrowserWidget::mnuDeleteItems);
    m_pActNewDir = new QAction(QIcon(":/files/icons/newDir.png"), tr("Create New Folder"), this);
    connect(m_pActNewDir, &QAction::triggered, this, &ProjectBrowserWidget::mnuNewDir);
    m_pActNewPyFile = new QAction(QIcon(":/files/icons/new.png"), tr("Create New Python File"), this);
    connect(m_pActNewPyFile, &QAction::triggered, this, &ProjectBrowserWidget::mnuNewPyFile);
    m_pActRefresh = new QAction(tr("Refresh"), this);
    m_pActRefresh->setShortcut(QKeySequence::Refresh);
    m_pActRefresh->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    connect(m_pActRefresh, &QAction::triggered, this, &ProjectBrowserWidget::mnuRefresh);

    m_pActAddFolder = new QAction(QIcon(":/files/icons/dirOpen.png"), tr("Add Folder to Project..."), this);
    connect(m_pActAddFolder, &QAction::triggered, this, &ProjectBrowserWidget::mnuAddFolder);
    m_pActRemoveFolder = new QAction(tr("Remove Folder from Project"), this);
    connect(m_pActRemoveFolder, &QAction::triggered, this, &ProjectBrowserWidget::mnuRemoveFolder);
    m_pActSetWorkingDir = new QAction(tr("Set as Working Directory"), this);
    connect(
        m_pActSetWorkingDir, &QAction::triggered, this, &ProjectBrowserWidget::mnuSetWorkingDirectory);
    m_pActShowInFileSystem = new QAction(tr("Show in File System View"), this);
    connect(
        m_pActShowInFileSystem, &QAction::triggered, this, &ProjectBrowserWidget::mnuShowInFileSystem);

    // the shortcuts must be known to the widget
    addAction(m_pActRenameItem);
    addAction(m_pActDeleteItems);
    addAction(m_pActRefresh);
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);

    if (m_pActOpenFile)
    {
        m_pContextMenu->addAction(m_pActOpenFile);
    }

    if (m_pActOpenInEditor)
    {
        m_pContextMenu->addAction(m_pActOpenInEditor);
    }

    if (m_pActExecuteFile)
    {
        m_pContextMenu->addAction(m_pActExecuteFile);
    }

    m_pContextMenu->addAction(m_pActLocateOnDisk);
    m_pContextMenu->addAction(m_pActShowInFileSystem);
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pActRenameItem);
    m_pContextMenu->addAction(m_pActDeleteItems);
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pActNewDir);
    m_pContextMenu->addAction(m_pActNewPyFile);
    m_pContextMenu->addAction(m_pActRefresh);
    m_pContextMenu->addSeparator();

    if (m_pActSetMainScript)
    {
        m_pContextMenu->addAction(m_pActSetMainScript);
    }

    if (m_pActRunMainScript)
    {
        m_pContextMenu->addAction(m_pActRunMainScript);
    }

    m_pContextMenu->addAction(m_pActSetWorkingDir);
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pActAddFolder);
    m_pContextMenu->addAction(m_pActRemoveFolder);
}

//-------------------------------------------------------------------------------------
QStringList ProjectBrowserWidget::selectedPaths() const
{
    QStringList paths;

    foreach (
        const QModelIndex& idx, FileOperations::uniqueRowIndexes(m_pTreeView->selectionModel()->selectedIndexes()))
    {
        paths << m_pModel->filePath(idx);
    }

    return paths;
}

//-------------------------------------------------------------------------------------
QString ProjectBrowserWidget::currentFolder() const
{
    const QModelIndex current = m_pTreeView->currentIndex();

    if (current.isValid() && m_pTreeView->selectionModel()->isSelected(current))
    {
        return m_pModel->isDir(current) ? m_pModel->filePath(current)
                                        : m_pModel->filePath(current.parent());
    }

    if (m_pModel->rowCount() > 0)
    {
        return m_pModel->filePath(m_pModel->index(0, 0));
    }

    return QString();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::projectOpened()
{
    ProjectOrganizer* org = organizer();

    if (org && org->hasProject())
    {
        m_pModel->setProject(org->project());
        m_pStack->setCurrentWidget(m_pProjectPage);

        restoreExpandedState();

        // expand the first root
        if (m_pModel->rowCount() > 0)
        {
            m_pTreeView->expand(m_pModel->index(0, 0));
        }
    }

    updateHeader();
    updateActions();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::projectClosed(const QString& filePath)
{
    if (!filePath.isEmpty() && m_pModel->rowCount() > 0)
    {
        QStringList expanded;
        collectExpanded(QModelIndex(), expanded);
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup(ProjectOrganizer::settingsGroup(filePath));
        settings.setValue("expandedFolders", expanded);
        settings.endGroup();
    }

    m_pModel->clear();
    m_pStack->setCurrentWidget(m_pNoProjectPage);
    updateHeader();
    updateActions();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::projectChanged()
{
    ProjectOrganizer* org = organizer();

    if (org && org->hasProject())
    {
        // only reset the model if the roots or excludes changed, since a reset
        // collapses the tree
        if (org->project().folders().size() != m_pModel->rootFolders().size() ||
            org->project().excludes() != m_pModel->excludes())
        {
            m_pModel->setProject(org->project());
        }
        else
        {
            const QList<ProjectFolder> a = org->project().folders();
            const QList<ProjectFolder> b = m_pModel->rootFolders();

            for (int i = 0; i < a.size(); ++i)
            {
                if (a[i].path != b[i].path)
                {
                    m_pModel->setProject(org->project());
                    break;
                }
            }
        }
    }

    updateHeader();
    updateActions();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::collectExpanded(const QModelIndex& parent, QStringList& expanded) const
{
    for (int r = 0; r < m_pModel->rowCount(parent); ++r)
    {
        const QModelIndex idx = m_pModel->index(r, 0, parent);

        if (m_pModel->isDir(idx) && m_pTreeView->isExpanded(idx))
        {
            expanded << m_pModel->filePath(idx);
            collectExpanded(idx, expanded);
        }
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::saveExpandedState()
{
    ProjectOrganizer* org = organizer();

    if (org && org->hasProject())
    {
        QStringList expanded;
        collectExpanded(QModelIndex(), expanded);
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup(ProjectOrganizer::settingsGroup(org->project().filePath()));
        settings.setValue("expandedFolders", expanded);
        settings.endGroup();
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::restoreExpandedState()
{
    ProjectOrganizer* org = organizer();

    if (!org || !org->hasProject())
    {
        return;
    }

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup(ProjectOrganizer::settingsGroup(org->project().filePath()));
    const QStringList expanded = settings.value("expandedFolders").toStringList();
    settings.endGroup();

    // parents come before their children in the stored list
    foreach (const QString& path, expanded)
    {
        const QModelIndex idx = m_pModel->indexForPath(path, true);

        if (idx.isValid())
        {
            m_pTreeView->expand(idx);
        }
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::updateHeader()
{
    ProjectOrganizer* org = organizer();

    if (org && org->hasProject())
    {
        const ProjectFile& p = org->project();
        m_pLblProjectName->setText(p.name());
        m_pLblProjectName->setToolTip(QDir::toNativeSeparators(p.filePath()));

        if (p.mainScript().isEmpty())
        {
            m_pLblMainScript->setText(tr("Main script: -"));
        }
        else
        {
            m_pLblMainScript->setText(tr("Main script: %1").arg(p.toRelative(p.mainScript())));
        }

        m_pLblMainScript->setToolTip(QDir::toNativeSeparators(p.mainScript()));
    }
    else
    {
        m_pLblProjectName->clear();
        m_pLblMainScript->clear();
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::updateActions()
{
    ProjectOrganizer* org = organizer();
    const bool hasProject = org && org->hasProject();
    const QModelIndexList rows = FileOperations::uniqueRowIndexes(m_pTreeView->selectionModel()->selectedIndexes());
    const bool selected = !rows.isEmpty();
    const bool one = (rows.size() == 1);
    bool isFile = false;
    bool isPyFile = false;
    bool isOtherEditable = false;
    bool isRoot = false;
    bool isDir = false;
    bool areFiles = selected;
    bool pyNotRun = false;

    if (one)
    {
        const QFileInfo fi = m_pModel->fileInfo(rows.first());
        isFile = fi.isFile();
        isDir = fi.isDir();
        isPyFile = isFile && fi.suffix().toLower() == "py";
        isOtherEditable = isFile && !isPyFile && EditorFileType::isEditable(fi.filePath());
        isRoot = m_pModel->nodeType(rows.first()) == ProjectTreeModel::RootFolderNode;
    }

    foreach (const QModelIndex& idx, rows)
    {
        if (m_pModel->isDir(idx))
        {
            areFiles = false;
            break;
        }
    }

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    pyNotRun = eng && !(eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting());

    if (m_pActOpenFile)
    {
        m_pActOpenFile->setVisible(isFile || areFiles);
    }

    if (m_pActOpenInEditor)
    {
        m_pActOpenInEditor->setVisible(isOtherEditable);
    }

    if (m_pActExecuteFile)
    {
        m_pActExecuteFile->setVisible(isPyFile);
        m_pActExecuteFile->setEnabled(isPyFile && pyNotRun);
    }

    m_pActLocateOnDisk->setVisible(selected);
    m_pActShowInFileSystem->setVisible(one);
    m_pActRenameItem->setVisible(one && !isRoot);
    m_pActDeleteItems->setVisible(selected);
    m_pActNewDir->setVisible(hasProject && (one || !selected));
    m_pActNewPyFile->setVisible(hasProject && (one || !selected));
    m_pActRefresh->setVisible(hasProject);

    if (m_pActSetMainScript)
    {
        m_pActSetMainScript->setVisible(isPyFile);
        m_pActSetMainScript->setEnabled(
            isPyFile && hasProject && org->project().mainScript() != m_pModel->filePath(rows.first()));
    }

    if (m_pActRunMainScript)
    {
        m_pActRunMainScript->setVisible(hasProject && !org->project().mainScript().isEmpty());
        m_pActRunMainScript->setEnabled(pyNotRun);
    }

    m_pActSetWorkingDir->setVisible(one && isDir);
    m_pActAddFolder->setVisible(hasProject);
    m_pActRemoveFolder->setVisible(one && isRoot && m_pModel->rowCount() > 1);

    // the delete action must not delete a root folder of the project from disk
    bool rootSelected = false;

    foreach (const QModelIndex& idx, rows)
    {
        if (m_pModel->nodeType(idx) == ProjectTreeModel::RootFolderNode)
        {
            rootSelected = true;
            break;
        }
    }

    m_pActDeleteItems->setEnabled(selected && !rootSelected);
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::contextMenuRequested(const QPoint& pos)
{
    updateActions();
    m_pContextMenu->exec(m_pTreeView->viewport()->mapToGlobal(pos));
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::itemActivated(const QModelIndex& index)
{
    if (!index.isValid())
    {
        return;
    }

    if (m_pModel->isDir(index))
    {
        m_pTreeView->setExpanded(index, !m_pTreeView->isExpanded(index));
    }
    else
    {
        ito::UserOrganizer* uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();

        if (uOrg && uOrg->currentUserHasFeature(featDeveloper))
        {
            IOHelper::openGeneralFile(
                m_pModel->filePath(index),
                true,
                true,
                this,
                SLOT(processError(QProcess::ProcessError)));
        }
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuOpenFile()
{
    foreach (const QModelIndex& idx, FileOperations::uniqueRowIndexes(m_pTreeView->selectionModel()->selectedIndexes()))
    {
        if (!m_pModel->isDir(idx))
        {
            itemActivated(idx);
        }
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuOpenInEditor()
{
    QFileInfoList infos;

    foreach (const QString& path, selectedPaths())
    {
        infos << QFileInfo(path);
    }

    foreach (const QString& filePath, FileOperations::editableFilePaths(infos))
    {
        IOHelper::openPythonScript(filePath);
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuExecuteFile()
{
    const QStringList paths = selectedPaths();
    PythonEngine* pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (!paths.isEmpty() && pyEngine)
    {
        QMetaObject::invokeMethod(pyEngine, "pythonRunFile", Q_ARG(QString, paths.first()));
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuRunMainScript()
{
    ProjectOrganizer* org = organizer();
    PythonEngine* pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (org && org->hasProject() && !org->project().mainScript().isEmpty() && pyEngine)
    {
        const QString script = org->project().mainScript();

        if (!QFileInfo(script).isFile())
        {
            showError(tr("Run Main Script"), tr("The main script '%1' does not exist.").arg(script));
            return;
        }

        QMetaObject::invokeMethod(pyEngine, "pythonRunFile", Q_ARG(QString, script));
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuLocateOnDisk()
{
    QSet<QString> paths;

    foreach (const QString& path, selectedPaths())
    {
        const QString canonical = QFileInfo(path).canonicalFilePath();

        if (!canonical.isEmpty() && !paths.contains(canonical))
        {
            paths.insert(canonical);
            FileOperations::showInGraphicalShell(canonical);
        }
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuShowInFileSystem()
{
    const QStringList paths = selectedPaths();

    if (paths.size() == 1)
    {
        const QFileInfo fi(paths.first());
        emit showInFileSystemRequested(fi.isDir() ? fi.absoluteFilePath() : fi.absolutePath());
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuRenameItem()
{
    const QStringList paths = selectedPaths();

    if (paths.size() != 1)
    {
        return;
    }

    const QFileInfo fi(paths.first());
    bool ok = false;
    const QString newName = QInputDialog::getText(
        this, tr("Rename"), tr("New name:"), QLineEdit::Normal, fi.fileName(), &ok);

    if (!ok || newName.isEmpty() || newName == fi.fileName())
    {
        return;
    }

    if (newName.contains('/') || newName.contains('\\'))
    {
        showError(tr("Rename"), tr("The name must not contain path separators."));
        return;
    }

    const QString newPath = fi.absolutePath() + "/" + newName;
    const bool success = fi.isDir() ? QDir().rename(fi.absoluteFilePath(), newPath)
                                    : QFile::rename(fi.absoluteFilePath(), newPath);

    if (!success)
    {
        showError(tr("Rename"), tr("'%1' could not be renamed to '%2'.").arg(fi.fileName(), newName));
    }
    else
    {
        m_pModel->refresh(m_pModel->indexForPath(fi.absolutePath()));
        m_pTreeView->setCurrentIndex(m_pModel->indexForPath(newPath));
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuDeleteItems()
{
    const QStringList paths = selectedPaths();

    if (paths.isEmpty())
    {
        return;
    }

    const QString itemName = (paths.size() == 1) ? "'" + QFileInfo(paths.first()).fileName() + "'"
                                                 : tr("the selected items");

    if (QMessageBox::question(
            this,
            tr("Delete"),
            tr("Do you really want to delete %1?").arg(itemName),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No) != QMessageBox::Yes)
    {
        return;
    }

    foreach (const QString& path, paths)
    {
        const QFileInfo fi(path);

        if (m_pModel->nodeType(m_pModel->indexForPath(path)) == ProjectTreeModel::RootFolderNode)
        {
            continue; // never delete a root folder of the project
        }

        const bool success = fi.isDir() ? QDir(path).removeRecursively() : QFile::remove(path);

        if (!success)
        {
            showError(tr("Delete"), tr("Error while deleting '%1'!").arg(path));
            break;
        }
    }

    m_pModel->refresh();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuNewDir()
{
    const QString parentDir = currentFolder();

    if (parentDir.isEmpty())
    {
        return;
    }

    const QString name = FileOperations::uniqueChildName(parentDir, tr("New Folder"), "");

    if (!QDir(parentDir).mkdir(name))
    {
        showError(tr("New Folder"), tr("Failed to create a new directory"));
        return;
    }

    m_pModel->refresh(m_pModel->indexForPath(parentDir));
    const QModelIndex idx = m_pModel->indexForPath(parentDir + "/" + name);

    if (idx.isValid())
    {
        m_pTreeView->setCurrentIndex(idx);
        mnuRenameItem();
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuNewPyFile()
{
    const QString parentDir = currentFolder();

    if (parentDir.isEmpty())
    {
        return;
    }

    const QString name = FileOperations::uniqueChildName(parentDir, tr("New Script"), ".py");
    const QString filePath = parentDir + "/" + name;

    if (FileOperations::createEmptyFile(filePath).containsError())
    {
        showError(tr("New Script"), tr("Failed to create a new script"));
        return;
    }

    m_pModel->refresh(m_pModel->indexForPath(parentDir));
    const QModelIndex idx = m_pModel->indexForPath(filePath);

    if (idx.isValid())
    {
        m_pTreeView->setCurrentIndex(idx);
        mnuRenameItem();
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuRefresh()
{
    m_pModel->refresh();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuAddFolder()
{
    ProjectOrganizer* org = organizer();

    if (!org || !org->hasProject())
    {
        return;
    }

    const QString folder = QFileDialog::getExistingDirectory(
        this, tr("Add Folder to Project"), org->project().projectDir());

    if (folder.isEmpty())
    {
        return;
    }

    if (!org->project().addFolder(folder))
    {
        showError(tr("Add Folder to Project"), tr("The folder is already part of the project."));
        return;
    }

    org->projectModified();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuRemoveFolder()
{
    ProjectOrganizer* org = organizer();
    const QStringList paths = selectedPaths();

    if (!org || !org->hasProject() || paths.size() != 1)
    {
        return;
    }

    if (org->project().folders().size() <= 1)
    {
        showError(
            tr("Remove Folder from Project"), tr("The last folder of a project cannot be removed."));
        return;
    }

    if (org->project().removeFolder(paths.first()))
    {
        org->projectModified();
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuSetWorkingDirectory()
{
    ProjectOrganizer* org = organizer();
    const QStringList paths = selectedPaths();

    if (!org || !org->hasProject() || paths.size() != 1 || !QFileInfo(paths.first()).isDir())
    {
        return;
    }

    org->project().setWorkingDirectory(paths.first());
    org->projectModified();

    if (QDir::setCurrent(paths.first()))
    {
        emit workingDirectoryChanged();
    }
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::mnuSetMainScript()
{
    ProjectOrganizer* org = organizer();
    const QStringList paths = selectedPaths();

    if (!org || !org->hasProject() || paths.size() != 1)
    {
        return;
    }

    org->project().setMainScript(paths.first());
    org->projectModified();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::processError(QProcess::ProcessError error)
{
    QMessageBox msgBox(this);
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(tr("An external process could not be started (%1).").arg(error));
    msgBox.exec();
}

//-------------------------------------------------------------------------------------
void ProjectBrowserWidget::showError(const QString& title, const QString& message)
{
    QMessageBox::warning(this, title, message);
}

} // namespace ito
