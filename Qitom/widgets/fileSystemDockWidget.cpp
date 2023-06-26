/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#include "python/pythonEngineInc.h"
#include "fileSystemDockWidget.h"
#include "../global.h"
#include "../AppManagement.h"
#include "../helper/IOHelper.h"
#include "../organizer/userOrganizer.h"

#include <qheaderview.h>
#include <qclipboard.h>
#include <qsettings.h>
#include <qfiledialog.h>
#include <qlayout.h>
#include <qmessagebox.h>
#include <qsizepolicy.h>
#include <qdesktopservices.h>
#include <qurl.h>
#include <qprocess.h>
#include <qcombobox.h>
#include <qapplication.h>
#include <qshortcut.h>
#include <qmimedata.h>

namespace ito {


//----------------------------------------------------------------------------------------------------------------------------------
FileSystemDockWidget::FileSystemDockWidget(const QString &title, const QString &objName, QWidget *parent, bool docked, bool isDockAvailable, tFloatingStyle floatingStyle, tMovingStyle movingStyle, const QString &baseDirectory) :
    AbstractDockWidget(docked, isDockAvailable, floatingStyle, movingStyle, title, objName, parent),
    m_pShowDirListMenu(NULL),
    m_pFileSystemSettingMenu(NULL),
    m_pContextMenu(NULL),
    m_pPathEdit(NULL),
    m_pMainToolbar(NULL),
    m_pTreeView(NULL),
    m_pLblFilter(NULL),
    m_pCmbFilter(NULL),
    m_pFileSystemModel(NULL),
    baseDirectory(QString()),
    m_pColumnWidth(NULL),
    m_pActMoveCDUp(NULL),
    m_pActSelectCD(NULL),
    m_pActOpenFile(NULL),
    m_pActExecuteFile(NULL),
    m_pActLocateOnDisk(NULL),
    m_pActRenameItem(NULL),
    m_pActDeleteItems(NULL),
    m_pActCutItems(NULL),
    m_pActCopyItems(NULL),
    m_pActPasteItems(NULL),
    m_pActNewDir(NULL),
    m_pActNewPyFile(NULL),
    m_pViewList(NULL),
    m_pViewDetails(NULL),
    m_lastMovedShowDirAction(NULL),
    m_linkColor(Qt::blue)
{
    int size = 0;
    QAction *act = NULL;
    QString actCheckedStr = "";
    QString actDir = "";
    QIcon actIcon;  // we cannot assign NULL to a qicon for gcc, so rely on default constructor ... hope this works

    m_pShowDirListMenu = new QMenu(tr("Last used directories"), this);
    m_pShowDirListMenu->setIcon(QIcon(":/files/icons/browser.png"));

    m_pShowDirListMenu->installEventFilter(this);

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("itomFileSystemDockWidget");
    size = settings.beginReadArray("lastUsedDirs");
    int count = 0;

    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        QString dir = settings.value("dir", QString()).toString();

        if (dir.mid(0, 1) == "@")
        {
            actIcon= QIcon(":/application/icons/pinChecked.png");
            actCheckedStr = "@";
            actDir = dir.mid(1);
        }
        else
        {
            actIcon= QIcon(":/application/icons/empty.png");
            actCheckedStr = "";
            actDir = dir;
        }

        if (QDir(actDir).exists())
        {
            act = m_pShowDirListMenu->addAction(QString::number(count + 1) + " " + actDir);
            act->setIcon(actIcon);
            act->setData(actDir);
            act->setCheckable(false);
            act->setWhatsThis(actCheckedStr);
            connect(act, &QAction::triggered, [=]() {
                newDirSelected(actDir);
            });
            count++;
        }
    }

    settings.endArray();

    m_pPathEdit = new QTextBrowser(this); //QTextEdit(this);
    m_pPathEdit->setReadOnly(true);
    m_pPathEdit->setMaximumHeight(25);
    m_pPathEdit->setLineWrapMode(QTextEdit::NoWrap);
    m_pPathEdit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_pPathEdit->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_pPathEdit->setOpenLinks(false);

    connect(m_pPathEdit, SIGNAL(anchorClicked(const QUrl&)), this, SLOT(pathAnchorClicked(const QUrl&)));
//m_pPathEdit->setEnabled(false);
/*    QColor color = 433;
//    m_pPathEdit->setTextBackgroundColor(color);
    QPalette::ColorGroup cg;
    m_pPathEdit->palette().setColor(cg, color);*/

    m_pLblFilter = new QLabel(tr("Filter:"), this);
    m_pLblFilter->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
    m_pLblFilter->setMaximumSize(m_pLblFilter->minimumSizeHint());

    m_pCmbFilter = new QComboBox(this);
    m_pCmbFilter->setEditable(true);
    m_pCmbFilter->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    m_pCmbFilter->setToolTip(tr("file name filters (semicolon or space separated list)"));

    connect(m_pCmbFilter, SIGNAL(editTextChanged(const QString&)), this, SLOT(cmbFilterEditTextChanged(const QString &)));

    m_pFileSystemModel = new ItomFileSystemModel(m_pTreeView);
    m_pFileSystemModel->setRootPath("");
    m_pFileSystemModel->setReadOnly(false);

    m_pTreeView = new ito::QTreeViewItom(this);
    connect(m_pTreeView, SIGNAL(activated(const QModelIndex&)), this, SLOT(openFile(const QModelIndex&)));
    connect(m_pTreeView, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(treeViewContextMenuRequested(const QPoint &)));
    m_pTreeView->setModel(m_pFileSystemModel);

    // Demonstrating look and feel features
    m_pTreeView->setAnimated(true);
    m_pTreeView->setIndentation(15);
    m_pTreeView->setSortingEnabled(true);
    m_pTreeView->setDefaultDropAction(Qt::MoveAction);

    //TODO save to ini-file

    int column = settings.value("sortColumn", 0).toInt();

    if (settings.value("sortOrder", Qt::AscendingOrder).toInt() == Qt::DescendingOrder)
    {
        m_pTreeView->sortByColumn(column, Qt::DescendingOrder);
    }
    else
    {
        m_pTreeView->sortByColumn(column, Qt::AscendingOrder);
    }

    m_pTreeView->setContextMenuPolicy(Qt::CustomContextMenu);
    m_pTreeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_pTreeView->setDragEnabled(true);
    m_pTreeView->setAcceptDrops(true);
    m_pTreeView->setEditTriggers(QAbstractItemView::NoEditTriggers);
    connect(m_pTreeView, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(itemDoubleClicked(const QModelIndex&)));

    size = settings.beginReadArray("ColWidth");

    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        m_pTreeView->setColumnWidth(i, settings.value("width", 100).toInt());
        m_pTreeView->setColumnHidden(i, m_pTreeView->columnWidth(i) == 0);
    }
    settings.endArray();

    m_pColumnWidth = new int[m_pFileSystemModel->columnCount()];
    size = settings.beginReadArray("StandardColWidth");

    if (size != m_pFileSystemModel->columnCount())
    {
        for (int i = 0; i < m_pFileSystemModel->columnCount(); ++i)
        {
            m_pColumnWidth[i] = 120;
        }
    }

    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        m_pColumnWidth[i] = settings.value("width", 100).toInt();
        if (m_pColumnWidth[i] == 0)
        {
            m_pColumnWidth[i] = 120;
        }
    }

    settings.endArray();
    settings.endGroup();

    AbstractDockWidget::init();

    QDir baseDir(baseDirectory);
    QString baseDirectoryTemp = baseDirectory;

    if (!baseDir.exists())
    {
        baseDirectoryTemp = QDir::currentPath();
    }

    changeBaseDirectory(baseDirectoryTemp);

    m_pFileSystemModel->setNameFilterDisables(false);

    QHBoxLayout *filterLayout = new QHBoxLayout;
    filterLayout->addWidget(m_pLblFilter);
    filterLayout->addWidget(m_pCmbFilter);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->setContentsMargins(0, 0, 0, 2);
    mainLayout->setSpacing(1);
    mainLayout->addWidget(m_pPathEdit);
    mainLayout->addWidget(m_pTreeView);
    mainLayout->addLayout(filterLayout);

    QWidget *centerWidget = new QWidget(this);
    centerWidget->setLayout(mainLayout);

    setContentWidget(centerWidget);

    fillFilterList();

//    QObject::dumpObjectTree();
}

//----------------------------------------------------------------------------------------------------------------------------------
FileSystemDockWidget::~FileSystemDockWidget()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    QStringList files;

    settings.beginGroup("itomFileSystemDockWidget");
    settings.beginWriteArray("lastUsedDirs");
    for (int i = 0; i < m_pShowDirListMenu->actions().count(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("dir", m_pShowDirListMenu->actions()[i]->whatsThis() + m_pShowDirListMenu->actions()[i]->data().toString());
    }

    settings.endArray();

    settings.beginWriteArray("ColWidth");
    for (int i = 0; i < m_pFileSystemModel->columnCount(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("width", m_pTreeView->columnWidth(i));
    }
    settings.endArray();

    settings.beginWriteArray("StandardColWidth");
    for (int i = 0; i < m_pFileSystemModel->columnCount(); i++)
    {
        settings.setArrayIndex(i);
        settings.setValue("width", m_pColumnWidth[i]);
    }
    settings.endArray();

    settings.setValue("sortColumn", m_pTreeView->header()->sortIndicatorSection());
    settings.setValue("sortOrder", m_pTreeView->header()->sortIndicatorOrder());

    settings.endGroup();

    DELETE_AND_SET_NULL(m_pShowDirListMenu);
    DELETE_AND_SET_NULL(m_pFileSystemSettingMenu);
    DELETE_AND_SET_NULL(m_pContextMenu);
    DELETE_AND_SET_NULL(m_pPathEdit);
    DELETE_AND_SET_NULL(m_pMainToolbar);
    DELETE_AND_SET_NULL(m_pTreeView);
    DELETE_AND_SET_NULL(m_pLblFilter);
    DELETE_AND_SET_NULL(m_pCmbFilter);
    DELETE_AND_SET_NULL(m_pFileSystemModel);
    DELETE_AND_SET_NULL_ARRAY(m_pColumnWidth);
    DELETE_AND_SET_NULL(m_pActMoveCDUp);
    DELETE_AND_SET_NULL(m_pActSelectCD);
    DELETE_AND_SET_NULL(m_pActOpenFile);
    DELETE_AND_SET_NULL(m_pActExecuteFile);
    DELETE_AND_SET_NULL(m_pActLocateOnDisk);
    DELETE_AND_SET_NULL(m_pActRenameItem);
    DELETE_AND_SET_NULL(m_pActDeleteItems);
    DELETE_AND_SET_NULL(m_pActCutItems);
    DELETE_AND_SET_NULL(m_pActCopyItems);
    DELETE_AND_SET_NULL(m_pActPasteItems);
    DELETE_AND_SET_NULL(m_pActNewDir);
    DELETE_AND_SET_NULL(m_pActNewPyFile);
    DELETE_AND_SET_NULL(m_pViewList);
    DELETE_AND_SET_NULL(m_pViewDetails);
    //DELETE_AND_SET_NULL(m_lastMovedShowDirAction);
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::setLinkColor(const QColor &color)
{
    //write in a qss file:
    /*
    ito--FileSystemDockWidget
    {
        qproperty-linkColor: rgb(255, 0, 0);
    }

    to influence the link color
    */
    m_linkColor = color;
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::createActions()
{
    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();

    m_pActSelectCD = new ShortcutAction(QIcon(":/files/icons/dirOpen.png"), tr("Open New Folder"), this);
    m_pActSelectCD->connectTrigger(this, SLOT(mnuSelectCD()));
    m_pActMoveCDUp = new ShortcutAction(QIcon(":/files/icons/dir-parent-folder.png"), tr("Change To Parent Folder"), this);
    m_pActMoveCDUp->connectTrigger(this, SLOT(mnuMoveCDUp()));
    m_pActCopyDir = new ShortcutAction(QIcon(":/files/icons/dirCopy.png"), tr("Copy Path To Clipboard"), this);
    m_pActCopyDir->connectTrigger(this, SLOT(mnuCopyDir()));
    m_pActPasteDir = new ShortcutAction(QIcon(":/files/icons/dirPaste.png"), tr("Get Path From Clipboard"), this);
    m_pActPasteDir->connectTrigger(this, SLOT(mnuPasteDir()));

    if (uOrg->currentUserHasFeature(featDeveloper))
    {
        m_pActOpenFile = new ShortcutAction(QIcon(":/files/icons/open.png"), tr("Open File"), this);
        m_pActOpenFile->connectTrigger(this, SLOT(mnuOpenFile()));
        m_pActExecuteFile = new ShortcutAction(QIcon(":/script/icons/runScript.png"), tr("Execute File"), this);
        m_pActExecuteFile->connectTrigger(this, SLOT(mnuExecuteFile()));
    }

    m_pActLocateOnDisk = new ShortcutAction(QIcon(":/files/icons/browser.png"), tr("Locate On Disk"), this);
    m_pActLocateOnDisk->connectTrigger(this, SLOT(mnuLocateOnDisk()));
    m_pActRenameItem = new ShortcutAction(QIcon(":/workspace/icons/edit-rename.png"), tr("Rename"), this, QKeySequence(tr("F2")), Qt::WidgetWithChildrenShortcut);
    m_pActRenameItem->connectTrigger(this, SLOT(mnuRenameItem()));
    m_pActDeleteItems = new ShortcutAction(QIcon(":/editor/icons/editDelete.png"), tr("Delete"), this, QKeySequence::Delete, Qt::WidgetWithChildrenShortcut);
    m_pActDeleteItems->connectTrigger(this, SLOT(mnuDeleteItems()));
    m_pActCutItems = new ShortcutAction(QIcon(":/editor/icons/editCut.png"), tr("Cut"), this, QKeySequence::Cut, Qt::WidgetWithChildrenShortcut);
    m_pActCutItems->connectTrigger(this, SLOT(mnuCutItems()));
    m_pActCopyItems = new ShortcutAction(QIcon(":/editor/icons/editCopy.png"), tr("Copy"), this, QKeySequence::Copy, Qt::WidgetWithChildrenShortcut);
    m_pActCopyItems->connectTrigger(this, SLOT(mnuCopyItems()));
    m_pActPasteItems = new ShortcutAction(QIcon(":/editor/icons/editPaste.png"), tr("Paste"), this, QKeySequence::Paste, Qt::WidgetWithChildrenShortcut);
    m_pActPasteItems->connectTrigger(this, SLOT(mnuPasteItems()));
    m_pActNewDir = new ShortcutAction(QIcon(":/files/icons/newDir.png"), tr("Create New Folder"), this);
    m_pActNewDir->connectTrigger(this, SLOT(mnuNewDir()));
    m_pActNewPyFile = new ShortcutAction(QIcon(":/files/icons/new.png"), tr("Create New Python File"), this);
    m_pActNewPyFile->connectTrigger(this, SLOT(mnuNewPyFile()));

    m_pViewList = new ShortcutAction(QIcon(":/application/icons/kdb_form.png"), tr("List"), this);
    m_pViewList->connectTrigger(this, SLOT(showList()));
    m_pViewDetails = new ShortcutAction(QIcon(":/application/icons/list.png"), tr("Details"), this);
    m_pViewDetails->connectTrigger(this, SLOT(showDetails()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::createMenus()
{
    m_pContextMenu = new QMenu(this);
    if (m_pActOpenFile)
    {
        m_pContextMenu->addAction(m_pActOpenFile->action());
    }

    if (m_pActExecuteFile)
    {
        m_pContextMenu->addAction(m_pActExecuteFile->action());
    }

    m_pContextMenu->addAction(m_pActLocateOnDisk->action());
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pActRenameItem->action());
    m_pContextMenu->addAction(m_pActDeleteItems->action());
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pActCutItems->action());
    m_pContextMenu->addAction(m_pActCopyItems->action());
    m_pContextMenu->addAction(m_pActPasteItems->action());
    m_pContextMenu->addSeparator();
    m_pContextMenu->addAction(m_pActMoveCDUp->action());
    m_pContextMenu->addAction(m_pActNewDir->action());
    m_pContextMenu->addAction(m_pActNewPyFile->action());

    m_pFileSystemSettingMenu = new QMenu(tr("Settings"), this);
    m_pFileSystemSettingMenu->setIcon(QIcon(":/application/icons/adBlockAction.png"));
    m_pFileSystemSettingMenu->addAction(m_pViewList->action());
    m_pFileSystemSettingMenu->addAction(m_pViewDetails->action());
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::createToolBars()
{
    QWidget *spacerWidget = new QWidget();
    QHBoxLayout *spacerLayout = new QHBoxLayout();
    spacerLayout->addItem(new QSpacerItem(5, 5, QSizePolicy::Expanding, QSizePolicy::Minimum));
    spacerLayout->setStretch(0, 2);
    spacerWidget->setLayout(spacerLayout);

    m_pMainToolbar = new QToolBar(tr("File System"), this);
    m_pMainToolbar->setObjectName("toolbarFileSystem");
    m_pMainToolbar->setContextMenuPolicy(Qt::PreventContextMenu);
    m_pMainToolbar->setFloatable(false);
    addToolBar(m_pMainToolbar, "mainToolBar");

    m_pMainToolbar->addAction(m_pShowDirListMenu->menuAction());
    m_pMainToolbar->addAction(m_pActSelectCD->action());
    m_pMainToolbar->addAction(m_pActMoveCDUp->action());
    m_pMainToolbar->addAction(m_pActCopyDir->action());
    m_pMainToolbar->addAction(m_pActPasteDir->action());
    m_pMainToolbar->addWidget(spacerWidget);
    m_pMainToolbar->addAction(m_pFileSystemSettingMenu->menuAction());
    connect(m_pFileSystemSettingMenu->menuAction(),SIGNAL(triggered()), this, SLOT(mnuToggleView()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::updateActions()
{
    QModelIndexList indexList;

    //filter indexList such that only one index per row and parent is available
    bool found;
    foreach(const QModelIndex &idx, m_pTreeView->selectedIndexes())
    {
        found = false;
        foreach(const QModelIndex &idx2, indexList)
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

    bool selected = indexList.count();

    bool isFile = false;
    bool areFiles = false;
    bool isPyFile = false;
    bool pyNotRun = false;
    bool selectedOnlyOne = (indexList.count() == 1);

    if (selectedOnlyOne)
    {
        QFileInfo fileinfo = m_pFileSystemModel->fileInfo(indexList.first());
        isFile = fileinfo.isFile();
        isPyFile = isFile && (fileinfo.suffix().toLower() == "py");
        if (isPyFile)
        {
            PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
            pyNotRun = eng && !(eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting());
        }
    }
    else
    {
        areFiles = indexList.size() == 0 ? false : true;

        foreach (const QModelIndex &idx, indexList)
        {
            QFileInfo fileinfo = m_pFileSystemModel->fileInfo(idx);
            areFiles = fileinfo.isFile();

            if (!areFiles)
            {
                break;
            }
        }
    }

    //shortcuts are always enabled, since the selection-changed signal of the tree-view does not call updateActions.
    if (m_pActMoveCDUp)
    {
        QDir baseDir(baseDirectory);
        m_pActMoveCDUp->setEnabled(baseDir.exists() && baseDir.cdUp(), true);
    }

    if (m_pActOpenFile)
    {
        m_pActOpenFile->setVisible(isFile || areFiles, true);
    }

    if (m_pActExecuteFile)
    {
        m_pActExecuteFile->setVisible(isPyFile, true);
        m_pActExecuteFile->setEnabled(isPyFile && pyNotRun, true);
    }

    if (m_pActLocateOnDisk)
    {
        m_pActLocateOnDisk->setVisible(selected, true);
    }

    if (m_pActRenameItem)
    {
        m_pActRenameItem->setVisible(selectedOnlyOne, true);
    }

    if (m_pActDeleteItems)
    {
        m_pActDeleteItems->setVisible(selected, true);
    }

    if (m_pActCutItems)
    {
        m_pActCutItems->setVisible(selected, true);
    }

    if (m_pActCopyItems)
    {
        m_pActCopyItems->setVisible(selected, true);
    }

    if (m_pActPasteItems)
    {
        QClipboard* clipboard = QApplication::clipboard();
        m_pActPasteItems->setEnabled(clipboard->mimeData()->hasUrls(), true);
    }

    if (m_pActNewDir)
    {
        m_pActNewDir->setVisible(selectedOnlyOne || !selected, true);
    }

    if (m_pActNewPyFile)
    {
        m_pActNewPyFile->setVisible(selectedOnlyOne || !selected, true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::fillFilterList()
{
    int defFilterNumber = 0;
    int cnt = 0;
    QString itomFiles = IOHelper::getAllItomFilesName();
    QString filters = IOHelper::getFileFilters(IOHelper::IOFilters(IOHelper::IOInput | IOHelper::IOOutput | IOHelper::IOPlugin | IOHelper::IOAllFiles | IOHelper::IOMimeAll));
    QStringList fList = filters.split(";;");
    defaultFilterPatterns.clear();
    QRegularExpression regExp("^[a-zA-Z0-9-_ ]+ \\((\\*\\..*)\\)$");
    QRegularExpressionMatch regExpMatch;
    QStringList suffixes;

    foreach(const QString &s, fList)
    {
        m_pCmbFilter->addItem(s, s);
        regExpMatch = regExp.match(s);

        if (regExpMatch.hasMatch())
        {
            suffixes = regExpMatch.captured(1).split(" ");
            defaultFilterPatterns[s] << suffixes;
        }

        if (s.contains(itomFiles))
        {
            defFilterNumber = cnt;
        }
        cnt++;
    }
    m_pCmbFilter->setCurrentIndex(defFilterNumber);
    cmbFilterEditTextChanged(m_pCmbFilter->currentText());
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::cmbFilterEditTextChanged(const QString &text)
{
    if (defaultFilterPatterns.contains(text))
    {
        this->m_pFileSystemModel->setNameFilters(defaultFilterPatterns[text]);
    }
    else
    {
        //text can contain a semicolon and/or space separated list of filters, e.g. "*.png *.jpg" or "*.png;*.jpg" or "*.png; *.jpg"
        QString text_ = text.trimmed();
        QStringList filters = text.split(";");
        QStringList filters2;
        foreach (const QString &f, filters)
        {
            filters2.append(f.trimmed().split(" "));
        }
        m_pFileSystemModel->setNameFilters(filters2);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal FileSystemDockWidget::changeBaseDirectory(QString dir)
{
    QAction* act = NULL;

    QMutexLocker mutexLocker(&baseDirChangeMutex);
    QDir newDir(dir);
    RetVal retValue(retOk);

    if (dir == baseDirectory)
    {
        return retValue;
    }

    if (newDir.exists())
    {
        baseDirectory = dir;
        QDir::setCurrent(baseDirectory);
    }
    else
    {
        retValue += ito::RetVal(ito::retError, 0, tr("Directory '%1' does not exist!").arg(dir).toLatin1().data());
    }

    // not existing
    if (retValue.containsError())
    {
        int i = 0;
        while ((i < m_pShowDirListMenu->actions().count()) && (m_pShowDirListMenu->actions()[i]->data().toString() != dir))
        {
            i++;
        }
        // remove action if in list
        if (i < m_pShowDirListMenu->actions().count())
        {
            removeActionFromDirList(i);
            m_lastMovedShowDirAction = NULL;
        }
    }
    else
    {
        m_pTreeView->setRootIndex(m_pFileSystemModel->index(baseDirectory)); //setCurrentIndex

        bool isInList = false;
        foreach (act, m_pShowDirListMenu->actions())
        {
            if (act->data().toString() == baseDirectory)
            {
                m_pShowDirListMenu->removeAction(act);
                isInList = true;
                break;
            }
        }

        if (!isInList)
        {
            QDir baseDir(baseDirectory);
            act = new QAction(baseDirectory, m_pShowDirListMenu);
            act->setData(baseDirectory);
            act->setWhatsThis("");
            act->setIcon(QIcon(":/application/icons/empty.png"));
            act->setCheckable(false);
            connect(act, &QAction::triggered, [=]() {
                newDirSelected(baseDirectory);
            });
        }

        if (m_pShowDirListMenu->actions().count() > 0)
        {
            m_pShowDirListMenu->insertAction(m_pShowDirListMenu->actions()[0], act);
        }
        else
        {
            m_pShowDirListMenu->insertAction(NULL, act);
        }

        if (m_pShowDirListMenu->actions().count() == 11)
        {
            int i = 10;
            while (m_pShowDirListMenu->actions()[i]->whatsThis() == "@")
            {
                --i;
            }
            removeActionFromDirList(i);
        }
    }

    // renew numbering
    for (int x = 0; x < m_pShowDirListMenu->actions().count(); x++)
    {
        m_pShowDirListMenu->actions()[x]->setText(QString::number(x+1) + " " + m_pShowDirListMenu->actions()[x]->data().toString());
    }

    m_pPathEdit->setToolTip(baseDirectory);
    m_pPathEdit->setHtml(getHtmlTag(baseDirectory));
    //m_pPathEdit->scrollContentsBy(500,0);

    //m_pPathEdit->scrollToAnchor("last");
//    m_pPathEdit->setText(baseDirectory);
    //m_pPathEdit->textCursor().setPosition(40); //movePosition(QTextCursor::End);
    //m_pPathEdit->ensureCursorVisible();

    updateActions();

    mutexLocker.unlock();

    if (!retValue.containsError())
    {
        emit currentDirChanged();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuSelectCD()
{
    QString newDirectory = QFileDialog::getExistingDirectory(this, tr("Select base directory"), baseDirectory);

    if (!newDirectory.isEmpty() && !newDirectory.isNull())
    {
        QDir baseDir(newDirectory);
        changeBaseDirectory(QDir::cleanPath(baseDir.absolutePath()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuMoveCDUp()
{
    QDir baseDir(baseDirectory);

    if (baseDir.exists() && baseDir.cdUp())
    {
        changeBaseDirectory(QDir::cleanPath(baseDir.absolutePath()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuCopyDir()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(QDir::toNativeSeparators(baseDirectory));
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuPasteDir()
{
    QClipboard *clipboard = QApplication::clipboard();
    QString text = clipboard->text();
    QFileInfo file(text);
    if (file.exists())
    {
        if (QString(file.fileName()).indexOf('.') > -1)
        {
            text = file.absoluteDir().path();
        }
        QDir baseDir(text);
        changeBaseDirectory(QDir::cleanPath(baseDir.absolutePath()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuLocateOnDisk()
{
    QModelIndexList indexList = m_pTreeView->selectedIndexes();
    QModelIndex index;
    QFile file;
    QFileInfo fileInfo;
    QSet<QString> dirSet;
    QDir dir;

    foreach(index, indexList)
    {
        if (index.isValid())
        {
            fileInfo = m_pFileSystemModel->fileInfo(index);
            if (fileInfo.isDir())
            {
                dirSet.insert(fileInfo.canonicalFilePath());
            }
            else
            {
                dirSet.insert(fileInfo.canonicalFilePath());
                //dirSet.insert(fileInfo.absoluteDir().canonicalPath());
            }
        }
    }

    //QString dirString;
    foreach(const QString &dirString, dirSet)
    {
        //QDesktopServices::openUrl(QUrl("file:///" + dirString, QUrl::StrictMode));
        showInGraphicalShell(dirString);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuExecuteFile()
{
    if (m_pTreeView->selectedIndexes().count() > 0)
    {
        PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (pyEngine)
        {
            QMetaObject::invokeMethod(pyEngine, "pythonRunFile", Q_ARG(QString, m_pFileSystemModel->fileInfo(m_pTreeView->selectedIndexes().first()).filePath()));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuOpenFile()
{
    QModelIndexList indexList;

    //filter indexList such that only one index per row and parent is available
    bool found;
    foreach(const QModelIndex &idx, m_pTreeView->selectedIndexes())
    {
        found = false;
        foreach(const QModelIndex &idx2, indexList)
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

    foreach (const QModelIndex &idx, indexList)
    {
        openFile(idx);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::newDirSelected(const QString& text)
{
    RetVal retValue(retOk);

    if (baseDirChangeMutex.tryLock(0))
    {
        baseDirChangeMutex.unlock();
        retValue += changeBaseDirectory(text);
        if (retValue.containsError())
        {
            QMessageBox msgBox;
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.setText(QLatin1String(retValue.errorMessage()));
            msgBox.exec();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::openFile(const QModelIndex& index)
{
    if (index.isValid())
    {
        if (!this->m_pFileSystemModel->fileInfo(index).isDir())
        {
            ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
            if (uOrg->currentUserHasFeature(featDeveloper))
            {
                IOHelper::openGeneralFile(m_pFileSystemModel->filePath(index), true, true, this, SLOT(processError(QProcess::ProcessError)));
            }
        }
        else
        {
            changeBaseDirectory(m_pFileSystemModel->filePath(index));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::treeViewContextMenuRequested(const QPoint &pos)
{
    updateActions();
    m_pContextMenu->exec(pos + m_pTreeView->mapToGlobal(m_pTreeView->pos()));
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuRenameItem()
{
    //filter indexList such that only one index per row and parent is available
    bool found;
    QModelIndexList indexList;
    foreach(const QModelIndex &idx, m_pTreeView->selectedIndexes())
    {
        found = false;
        foreach(const QModelIndex &idx2, indexList)
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

    if (indexList.count() == 1)
    {
        m_pTreeView->edit(indexList[0]);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuDeleteItems()
{
    QString itemName;
    QStringList fileList;

    //filter indexList such that only one index per row and parent is available
    bool found;
    QModelIndexList indexList;
    foreach(const QModelIndex &idx, m_pTreeView->selectedIndexes())
    {
        found = false;
        foreach(const QModelIndex &idx2, indexList)
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

    if (indexList.count() == 1)
    {
        itemName = "'" + m_pFileSystemModel->fileInfo(indexList[0]).fileName() + "'";
    }
    else
    {
        itemName = tr("the selected items");
    }

    if (QMessageBox::question(this, tr("Delete"), tr("Do you really want to delete %1?").arg(itemName), QMessageBox::Yes, QMessageBox::No, QMessageBox::NoButton) == QMessageBox::Yes)
    {
        // first we have to create a list with all selected files
        foreach(const QModelIndex &idx, indexList)
        {
            fileList.append(m_pFileSystemModel->fileInfo(idx).filePath());
        }

        // now we can delete the files in the list
        for (int i = 0; i < fileList.count(); i++)
        {
            if (!m_pFileSystemModel->remove(m_pFileSystemModel->index(fileList[i])))
            {
                QMessageBox::warning(this, tr("Delete"), tr("Error while deleting '%1'!").arg(fileList[i]));
                break;
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuCutItems()
{
    QModelIndexList filesToCut;
    QClipboard *clipboard = QApplication::clipboard();

    foreach (const QModelIndex &idx, m_pTreeView->selectedIndexes())
    {
        if (idx.column() == 0)
        {
            filesToCut.append(idx);
        }
    }

    if (filesToCut.count() > 0)
    {
        QMimeData *mimeData = m_pFileSystemModel->mimeData(filesToCut);
        m_clipboardCutData = mimeData->urls();
        clipboard->setMimeData(mimeData, QClipboard::Clipboard);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuCopyItems()
{
    QModelIndexList filesToCopy;
    QClipboard *clipboard = QApplication::clipboard();

    m_clipboardCutData.clear();

    foreach (const QModelIndex &idx, m_pTreeView->selectedIndexes())
    {
        if (idx.column() == 0)
        {
            filesToCopy.append(idx);
        }
    }

    if (filesToCopy.count() > 0)
    {
        QMimeData *mimeData = m_pFileSystemModel->mimeData(filesToCopy);
        clipboard->setMimeData(mimeData, QClipboard::Clipboard);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuPasteItems()
{
    QModelIndex index;
    QClipboard *clipboard = QApplication::clipboard();
    const QMimeData *mimeData = clipboard->mimeData(QClipboard::Clipboard);
    bool result = false;

    if (mimeData->hasUrls())
    {
        if (m_pTreeView->selectedIndexes().count() == 0)
        {
            index = m_pTreeView->rootIndex();
        }
        else
        {
            if (m_pFileSystemModel->fileInfo(m_pTreeView->currentIndex()).isDir())
            {
                index = m_pTreeView->currentIndex();
            }
            else
            {
                index = m_pTreeView->currentIndex().parent();
            }
        }

        if (mimeData->urls() == m_clipboardCutData) //this mime data is result of a cut-action
        {
            result = m_pFileSystemModel->dropMimeData(mimeData, Qt::MoveAction, 0 /*unused in Qt*/, 0 /*unused in Qt*/, index);
            m_clipboardCutData.clear();
            clipboard->clear();
        }
        else
        {
            result = m_pFileSystemModel->dropMimeData(mimeData, Qt::CopyAction, 0 /*unused in Qt*/, 0 /*unused in Qt*/, index);
        }

        if (!result)
        {
            QMessageBox::warning(this, tr("Error pasting or copying files"), tr("At least one of the selected items could not be moved or copied. Maybe an existing file should have been overwritten, but could not be deleted first."));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuNewDir()
{
    QModelIndex index;
    if (!m_pTreeView->selectedIndexes().count())
    {
        index = m_pTreeView->rootIndex();
    }
    else
    {
        if (m_pFileSystemModel->fileInfo(m_pTreeView->currentIndex()).isDir())
        {
            index = m_pTreeView->currentIndex();
        }
        else
        {
            index = m_pTreeView->currentIndex().parent();
        }
    }

    QString folderName = tr("New Folder");
    QString filePath = m_pFileSystemModel->fileInfo(index).absoluteFilePath() + "/" + folderName;
    QDir dir(filePath);
    if (dir.exists())
    {
        int i = 0;
        bool done = false;
        while (!done)
        {
            i++;
            dir.setPath(filePath + " (" + QString::number(i) + ")");
            done = !dir.exists();
        }
        folderName = folderName + " (" + QString::number(i) + ")";
    }

    index = m_pFileSystemModel->mkdir(index, folderName);
    if (index.isValid())
    {
        m_pTreeView->setCurrentIndex(index);
        m_pTreeView->edit(index);
    }
    else
    {
        QMessageBox::warning(this, tr("New Folder"), tr("Failed to create a new directory"));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuNewPyFile()
{
    QModelIndex index;
    if (!m_pTreeView->selectedIndexes().count())
    {
        index = m_pTreeView->rootIndex();
    }
    else
    {
        if (m_pFileSystemModel->fileInfo(m_pTreeView->currentIndex()).isDir())
        {
            index = m_pTreeView->currentIndex();
        }
        else
        {
            index = m_pTreeView->currentIndex().parent();
        }
    }

    QString fileName = tr("New Script");
    QString filePath = m_pFileSystemModel->fileInfo(index).absoluteFilePath() + "/" + fileName;
    QFile file(filePath + ".py");
    if (file.exists())
    {
        int i = 0;
        bool done = false;
        while (!done)
        {
            i++;
            file.setFileName(filePath + " (" + QString::number(i) + ").py");
            done = !file.exists();
        }
    }

    if (file.open(QIODevice::WriteOnly))
    {
        file.close();

        index = m_pFileSystemModel->index(file.fileName());
        if (index.isValid())
        {
            m_pTreeView->setCurrentIndex(index);
            m_pTreeView->edit(index);
        }
    }
    else
    {
        QMessageBox::warning(this, tr("New Script"), tr("Failed to create a new script"));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::itemDoubleClicked(const QModelIndex &index)
{
    if (index.isValid() && m_pFileSystemModel->fileInfo(index).isDir())
    {
        QDir currentDir(QDir::currentPath());
        QString selectedDir = index.data().toString();
        if (currentDir.exists(selectedDir))
        {
            changeBaseDirectory(currentDir.absoluteFilePath(selectedDir));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::showInGraphicalShell(const QString & filePath)
{
#ifdef __APPLE__
    QStringList args;
    args << "-e";
    args << "tell application \"Finder\"";
    args << "-e";
    args << "activate";
    args << "-e";
    args << "select POSIX file \""+filePath+"\"";
    args << "-e";
    args << "end tell";
    QProcess::startDetached("osascript", args);
#endif

#ifdef WIN32
    QStringList args;
    args << "/select," << QDir::toNativeSeparators(filePath);
    QProcess::startDetached("explorer", args);
#endif
//
//    // Mac, Windows support folder or file.
//#if defined(WIN32)
//    const QString explorer = QProcess::systemEnvironment().searchInPath(QLatin1String("explorer.exe"));
//    if (explorer.isEmpty()) {
//        QMessageBox::warning(parent,
//                             tr("Launching Windows Explorer failed"),
//                             tr("Could not find explorer.exe in path to launch Windows Explorer."));
//        return;
//    }
//    QString param;
//    if (!QFileInfo(filePath).isDir())
//        param = QLatin1String("/select,");
//    param += QDir::toNativeSeparators(filePath);
//    QProcess::startDetached(explorer, QStringList(param));
//#elif defined(Q_OS_MAC)
//    Q_UNUSED(parent)
//    QStringList scriptArgs;
//    scriptArgs << QLatin1String("-e")
//               << QString::fromLatin1("tell application \"Finder\" to reveal POSIX file \"%1\"")
//                                     .arg(filePath);
//    QProcess::execute(QLatin1String("/usr/bin/osascript"), scriptArgs);
//    scriptArgs.clear();
//    scriptArgs << QLatin1String("-e")
//               << QLatin1String("tell application \"Finder\" to activate");
//    QProcess::execute("/usr/bin/osascript", scriptArgs);
//#else
//    // we cannot select a file here, because no file browser really supports it...
//    const QFileInfo fileInfo(filePath);
//    const QString folder = fileInfo.absoluteFilePath();
//    const QString app = Utils::UnixUtils::fileBrowser(Core::ICore::instance()->settings());
//    QProcess browserProc;
//    const QString browserArgs = Utils::UnixUtils::substituteFileBrowserParameters(app, folder);
//    if (debug)
//        qDebug() <<  browserArgs;
//    bool success = browserProc.startDetached(browserArgs);
//    const QString error = QString::fromLocal8Bit(browserProc.readAllStandardError());
//    success = success && error.isEmpty();
//    if (!success)
//        showGraphicalShellError(parent, app, error);
//#endif
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::setTreeViewHideColumns(const bool &hide)
{
    for (int i = 1; i < m_pFileSystemModel->columnCount(); ++i)
    {
        m_pTreeView->setColumnHidden(i, hide);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::showList()
{
    bool isList = true;

    for (int i = 1; i < m_pFileSystemModel->columnCount(); ++i)
    {
        isList = isList && m_pTreeView->isColumnHidden(i);
    }

    if (!isList)
    {
        for (int i = 0; i < m_pFileSystemModel->columnCount(); ++i)
        {
            m_pColumnWidth[i] = m_pTreeView->columnWidth(i);
        }
    }

    setTreeViewHideColumns(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::showDetails()
{
    bool isList = true;

    for (int i = 1; i < m_pFileSystemModel->columnCount(); ++i)
    {
        isList = isList && m_pTreeView->isColumnHidden(i);
    }

    setTreeViewHideColumns(false);

    if (isList)
    {
        for (int i = 0; i < m_pFileSystemModel->columnCount(); ++i)
        {
            m_pTreeView->setColumnWidth(i, m_pColumnWidth[i]);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::mnuToggleView()
{
    if (m_pTreeView->isColumnHidden(1))
    {
        showDetails();
    }
    else
    {
        showList();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::removeActionFromDirList(const int &pos)
{
    QAction *act = m_pShowDirListMenu->actions()[pos];
    m_pShowDirListMenu->removeAction(act);
    DELETE_AND_SET_NULL(act);
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::processError(QProcess::ProcessError error)
{
    QMessageBox msgBox(this);
    msgBox.setIcon(QMessageBox::Information);
    msgBox.setText(QString("An external process could not be started (%1).").arg(error));
    msgBox.exec();
}

//----------------------------------------------------------------------------------------------------------------------------------
void FileSystemDockWidget::pathAnchorClicked(const QUrl &link)
{
    if (link.isLocalFile())
    {
        QString dir = link.toLocalFile();

        if (dir.size() == 2)
        {
            dir += "/";
        }
        newDirSelected(dir);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QString FileSystemDockWidget::getHtmlTag(const QString &tag)
{
#ifndef WIN32
    QChar separator = '/';
#else
    QChar separator = '\\';
#endif

    QString link = "";
    QStringList tagList = tag.split("/");

    QString text = QString("<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.0//EN' 'http://www.w3.org/TR/REC-html40/strict.dtd'>\n\
<html><head><meta name='qrichtext' content='1' /><style type='text/css'>\n\
p, li { white-space: pre-wrap; }\n\
a { color: %1; }\n\
</style></head><body style=' font-family:'MS Shell Dlg 2'; font-size:8.25pt; font-weight:400; font-style:normal;'>\n\
<p style=' margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;'><span style=' font-size:8pt;'>").arg(m_linkColor.name());
    for (int x = 0; x < tagList.size(); x++)
    {
        if (x > 0)
        {
            text += separator;
            link += '/';
        }
        link += tagList[x];

		if (tagList[x].size() > 0)
		{
			if (link.startsWith("//"))
			{
				text += "<a href='file://" + link + "'>" + tagList[x] + "</a>";
			}
			else
			{
				text += "<a href='file:///" + link + "'>" + tagList[x] + "</a>";
			}
		}
    }
    text += "<a name=\"last\"></a></span></p></body></html>";
    //qDebug() << txt;
    return text;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool FileSystemDockWidget::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseMove)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        QAction *actionUnderMouse = m_pShowDirListMenu->actionAt(mouseEvent->pos());
        if (m_lastMovedShowDirAction != actionUnderMouse)
        {
            if (m_lastMovedShowDirAction && m_lastMovedShowDirAction->whatsThis() == "")
            {
                m_lastMovedShowDirAction->setIcon(QIcon(":/application/icons/empty.png"));
            }
            m_lastMovedShowDirAction = actionUnderMouse;
        }

        if (actionUnderMouse && actionUnderMouse->whatsThis() == "")
        {
            if (mouseEvent->pos().x() < 24 && mouseEvent->pos().x() > 1)
            {
                actionUnderMouse->setIcon(QIcon(":/application/icons/pin.png"));
            }
            else
            {
                actionUnderMouse->setIcon(QIcon(":/application/icons/empty.png"));
            }
        }
    }
    else if (event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        QAction *actionUnderMouse = m_pShowDirListMenu->actionAt(mouseEvent->pos());
        if (actionUnderMouse && (mouseEvent->pos().x() < 24 && mouseEvent->pos().x() > 1))
        {
            if (actionUnderMouse->whatsThis() == "")
            {
                actionUnderMouse->setIcon(QIcon(":/application/icons/pinChecked.png"));
                actionUnderMouse->setWhatsThis("@");
            }
            else
            {
                actionUnderMouse->setIcon(QIcon(":/application/icons/pin.png"));
                actionUnderMouse->setWhatsThis("");
            }
            return true;
        }
    }
    return QObject::eventFilter(obj, event);
}

} //end namespace ito
