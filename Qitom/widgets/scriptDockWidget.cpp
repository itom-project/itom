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

#include "../python/pythonEngineInc.h"

#include "scriptDockWidget.h"
#include "scriptEditorWidget.h"

#include "../widgets/mainWindow.h"
#include "../global.h"

#include <qlist.h>
#include <qfileinfo.h>
#include <qdebug.h>
#include <qboxlayout.h>
#include <qmessagebox.h>
#include <qdir.h>

#include "../ui/dialogGoto.h"
#include "../ui/dialogReplace.h"

#include <qsignalmapper.h>
#include <qstatusbar.h>

#include "../ui/dialogIconBrowser.h"
#include "../Qitom/AppManagement.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "../helper/IOHelper.h"

namespace ito {

/*static*/ QPointer<ScriptEditorWidget> ScriptDockWidget::currentSelectedCallstackLineEditor = QPointer<ScriptEditorWidget>();
/*static*/ const char* ScriptDockWidget::statusBarStatePropertyName = "_statusBarState";

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class ScriptDockWidget
    \brief widget containing one or multiple script editors (tabbed). This widget can either be a docking widget, docked in a docking
    area in the main window or it can be a floatable window which has a standard window behaviour.

    \todo right now, the floatable window does not have an single icon in the windows taskbar and is always on top of the main window.
          This behavior can be changed e.g. by defining an own titleWidget of the dockWidget.
*/

//! constructor
/*!
    creates a tab widget (QTabWidgetItom), which contains multiple instances of ScriptEditorWidget.
    Creates actions, menus, toolbars and sets some other settings.

    \param title Title of the DockWidget
    \param docked true: this widget should be docked at creating time, else: false
    \param isDockAvailable indicates whether docking functionality is available, if not, docked is always set to false
    \param commonActions is a collection of common actions, managed and created by the ScriptEditorOrganizer. These actions can be used in the dock widget or its child editors.

    \sa AbstractDockWidget::AbstractDockWidget
*/
ScriptDockWidget::ScriptDockWidget(const QString &title, const QString &objName,
    bool docked, bool isDockAvailable,
    const ScriptEditorActions &commonActions,
    BookmarkModel *bookmarkModel,
    QWidget* parent, Qt::WindowFlags /*flags*/) :
    AbstractDockWidget(docked, isDockAvailable, floatingWindow, movingEnabled, title, objName, parent),
    m_tab(nullptr),
    m_pWidgetFindWord(nullptr),
    m_pDialogReplace(nullptr),
    m_actTabIndex(-1),
    m_tabContextMenu(NULL),
    m_winMenu(NULL),
    m_commonActions(commonActions),
    m_pBookmarkModel(bookmarkModel),
    m_outlineShowNavigation(true),
    m_pStatusBarWidget(nullptr)
{
    qRegisterMetaType<QSharedPointer<OutlineItem> >("QSharedPointer<OutlineItem>");

    m_tab = new QTabWidgetItom(this);

    //!< tab-settings
    m_tab->setElideMode(Qt::ElideNone);
    m_tab->setTabShape(QTabWidget::Rounded);
    m_tab->setTabsClosable(true);
    m_tab->setMovable(true);
    m_tab->setTabPosition(QTabWidget::South);
    m_tab->setContentsMargins(2,0,2,0);

    const MainWindow *mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());

    connect(m_tab, SIGNAL(tabContextMenuEvent(QContextMenuEvent*)), this, SLOT(tabContextMenuEvent(QContextMenuEvent*)));
    connect(this, SIGNAL(pythonRunSelection(QString)), mainWin, SLOT(pythonRunSelection(QString)));

    AbstractDockWidget::init();

    //this is an example shortcut. This shortcut is the same than the shortcut directly set to the corresponding action.
    //The action is not triggered by shortcut in docked-mode. Therefore we have the following QShortcut-instance.
    //In undocked mode, the parent-dock-widget of the QShortcut is invisble, hence disabled, and the QAction-shortcut is triggered.
    //The shortcut is deleted when this instance is deleted (due to parent-indication).
    //QShortcut *s = new QShortcut(QKeySequence::Find, this, SLOT(mnuFindTextExpr()), 0, Qt::WidgetWithChildrenShortcut);

    resize(700,400);

    connect(m_tab, SIGNAL(currentChanged(int)), this, SLOT(currentTabChanged(int)));
    if (m_tabContextMenu != NULL)
    {
        connect(m_tabContextMenu, SIGNAL(aboutToShow()), this, SLOT(updateTabContextActions()));
    }
    connect(m_tab, SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested(int)));

    m_pWidgetFindWord = new WidgetFindWord(this);
    connect(m_pWidgetFindWord, SIGNAL(findNext(QString,bool,bool,bool,bool,bool,bool)), this, SLOT(findTextExpr(QString,bool,bool,bool,bool,bool,bool)));
    connect(m_pWidgetFindWord, SIGNAL(hideSearchBar()), this, SLOT(findWordWidgetFinished()));

    // Layoutbox
    m_pVBox = new QVBoxLayout();
    m_pVBox->setContentsMargins(0, 0, 0, 0);
    m_pVBox->setSpacing(1);

    // Create new widget for class navigation (bar)
    m_classMenuBar = new QWidget(this);
    m_classMenuBar->setContentsMargins(0,0,0,0);

    // These two comboboxes go inside
    m_classBox = new QComboBox(m_classMenuBar);
    m_classBox->setMinimumHeight(20);
    m_classBox->setMaxCount(500000);
    m_classBox->setContentsMargins(2, 0, 2, 0);
    m_classBox->setStyleSheet("QComboBox {border: 0px; border-radius: 0px; padding: 0px 18px 0px 3px;}");

    m_methodBox = new QComboBox(m_classMenuBar);
    m_methodBox->setMinimumHeight(20);
    m_methodBox->setMaxCount(500000);
    m_methodBox->setContentsMargins(2, 0, 2, 0);
    m_methodBox->setStyleSheet("QComboBox {border: 0px; border-radius: 0px; padding: 0px 18px 0px 3px;}");

    // Layout inside the widget (two comboboxes)
    QHBoxLayout *hLayoutBox = new QHBoxLayout();
    hLayoutBox->setSpacing(0);
    hLayoutBox->addWidget(m_classBox);
    hLayoutBox->addWidget(m_methodBox);
    hLayoutBox->setContentsMargins(0, 0, 0, 0);
    m_classMenuBar->setLayout(hLayoutBox);

    // Set size for widget
    m_classMenuBar->setMinimumHeight(20);
    m_classMenuBar->setMaximumHeight(20);
    m_pVBox->addWidget(m_classMenuBar, 1);

    connect(m_classBox, SIGNAL(activated(int)), this, SLOT(navigatorClassSelected(int)));
    connect(m_methodBox, SIGNAL(activated(int)), this, SLOT(navigatorMethodSelected(int)));

    // Add EditorTab
    m_pVBox->addWidget(m_tab);
    m_pVBox->addWidget(m_pWidgetFindWord);

    m_pCenterWidget = new QWidget();
    m_pCenterWidget->setLayout(m_pVBox);
    m_pCenterWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    setContentWidget(m_pCenterWidget);
    setFocusPolicy(Qt::StrongFocus);

    loadSettings();

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(loadSettings()));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    cancels connections and closes every tab.
*/
ScriptDockWidget::~ScriptDockWidget()
{
    disconnect(m_pWidgetFindWord, SIGNAL(findNext(QString,bool,bool,bool,bool,bool,bool)), this, SLOT(findTextExpr(QString,bool,bool,bool,bool,bool,bool)));
    disconnect(m_tab,SIGNAL(currentChanged(int)), this, SLOT(currentTabChanged(int)));
    if (m_tabContextMenu != NULL)
    {
        disconnect(m_tabContextMenu, SIGNAL(aboutToShow()), this, SLOT(updateTabContextActions()));
    }
    disconnect(m_tab, SIGNAL(tabCloseRequested(int)), this, SLOT(tabCloseRequested(int)));

    for (int i = m_tab->count() - 1; i >= 0; i--)
    {
        closeTab(i, false);
    }

    DELETE_AND_SET_NULL(m_tab);
    DELETE_AND_SET_NULL(m_pWidgetFindWord);
    DELETE_AND_SET_NULL(m_pVBox);
    DELETE_AND_SET_NULL(m_pCenterWidget);
    DELETE_AND_SET_NULL(m_pDialogReplace);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // Code Outline
    m_outlineShowNavigation = settings.value("outlineShowNavigation", true).toBool();
    showOutlineNavigationBar(m_outlineShowNavigation);

    int elideMode = settings.value("tabElideMode", Qt::ElideNone).toInt();

    switch (elideMode)
    {
    case Qt::ElideLeft:
        m_tab->setElideMode(Qt::ElideLeft);
        break;
    case Qt::ElideRight:
        m_tab->setElideMode(Qt::ElideRight);
        break;
    case Qt::ElideMiddle:
        m_tab->setElideMode(Qt::ElideMiddle);
        break;
    default:
        m_tab->setElideMode(Qt::ElideNone);
        break;
    }

    m_autoCodeFormatCmd = settings.value("autoCodeFormatCmd", "black --line-length 88 --quiet -").toString();

    m_autoCodeFormatAction->action()->setVisible(settings.value("autoCodeFormatEnabled", true).toBool());

    m_autoCodeFormatAction->setEnabled(m_autoCodeFormatCmd != "" && getCurrentEditor() != nullptr);

    settings.endGroup();
}

//------------------------------------------------------------------------------------
/* Recursive fill of all classes in the outline to the class combo box. */
void ScriptDockWidget::fillNavigationClassComboBox(
    const QSharedPointer<OutlineItem> &parent,
    const QString &prefix)
{
    if (!parent.isNull())
    {
        QString name;

        if (parent->m_type == OutlineItem::typeRoot)
        {
            QVariant userData = QVariant::fromValue(parent);
            m_classBox->addItem(
                parent->icon(),
                parent->m_name,
                userData
            );
        }

        foreach(auto item, parent->m_childs)
        {
            if (item->m_type == OutlineItem::typeClass)
            {
                QVariant userData = QVariant::fromValue(item);

                if (prefix != "")
                {
                    name = QString("class %1.%2").arg(prefix, item->m_name);
                }
                else
                {
                    name = QString("class %1").arg(item->m_name);
                }

                m_classBox->addItem(
                    item->icon(),
                    name,
                    userData
                );

                if (item->m_childs.size() > 0)
                {
                    if (prefix == "")
                    {
                        fillNavigationClassComboBox(item, item->m_name);
                    }
                    else
                    {
                        fillNavigationClassComboBox(item, QString("%1.%2").arg(prefix).arg(item->m_name));
                    }
                }
            }
        }
    }
}

//--------------------------------------------------------------------
QString argsWordWrap(QString text, int width)
{
    QString result;
    int j, i;
    bool firstWrap = true;

    for (;;)
    {
        i = std::min(width, (int)text.length());
        j = text.lastIndexOf(", ", i);

        if (j == -1)
        {
            j = text.indexOf(", ", i);
        }

        if (j > 0)
        {
            result += text.left(j);
            result += ",\n    ";
            text = text.mid(j + 2);

            if (firstWrap)
            {
                firstWrap = false;
                width -= 4;
            }
        }
        else
        {
            break;
        }

        if (width >= text.length())
        {
            break;
        }
    }

    return result + text;
}

//-------------------------------------------------------------------------------------
void methodBoxAddItem(
    QComboBox *methodBox,
    const QIcon &icon,
    const QString &methPre,
    const QString &methArgs,
    const QString &methPost,
    const QVariant &userData)
{
    QString fullSig = QString("%1(%2)").arg(methPre, methArgs);

    if (methPost != "")
    {
        fullSig += " -> " + methPost;
    }

    const int maxLength = 150;

    if (fullSig.size() <= maxLength)
    {
        methodBox->addItem(icon, fullSig, userData);
        methodBox->setItemData(methodBox->count() - 1, fullSig, Qt::ToolTipRole);
    }
    else
    {
        // todo: it seems that eliding the text of the combobox is not relevant
        // if no stylesheets are applied. Only if stylesheets are used, the
        // minimumSize of the comboBox seems to be adapted to the necessary
        // size of the real text in all entries (maybe a bug in Qt???).
        QString methArgsElide = methArgs.left(
            std::max(0, maxLength - 4 - (int)methPre.size() - (int)methPost.size())
        ) + "...";
        fullSig = QString("%1(%2)").arg(methPre, methArgsElide);

        if (methPost != "")
        {
            fullSig += " -> " + methPost;
        }

        methodBox->addItem(
            icon,
            fullSig,
            userData
        );

        QString methArgsWrapped = argsWordWrap(methArgs, 100);
        fullSig = QString("%1(\n    %2\n)").arg(methPre, methArgsWrapped);

        if (methPost != "")
        {
            fullSig += " -> " + methPost;
        }

        methodBox->setItemData(methodBox->count() - 1, fullSig, Qt::ToolTipRole);
    }
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::fillNavigationMethodComboBox(
    const QSharedPointer<OutlineItem> &parent,
    const QString &prefix)
{
    // insert empty dummy item
    if (prefix == "")
    {
        auto invalid = QVariant::fromValue(QSharedPointer<OutlineItem>());
        m_methodBox->addItem(QIcon(), "", invalid);
    }

    if (parent.isNull())
    {
        return;
    }

    foreach(const auto &item, parent->m_childs)
    {
        auto userData = QVariant::fromValue(item);

        switch (item->m_type)
        {
        case OutlineItem::typeFunction:
        case OutlineItem::typeMethod:
            if (item->m_async)
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "async def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            else
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            break;
        case OutlineItem::typePropertyGet:
            if (item->m_async)
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[get] async def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            else
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[get] def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            break;
        case OutlineItem::typePropertySet:
            if (item->m_async)
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[set] async def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            else
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[set] def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            break;
        case OutlineItem::typeStaticMethod:
            if (item->m_async)
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[static] async def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            else
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[static] def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            break;
        case OutlineItem::typeClassMethod:
            if (item->m_async)
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[classmethod] async def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            else
            {
                methodBoxAddItem(
                    m_methodBox,
                    item->icon(),
                    prefix + "[classmethod] def " + item->m_name,
                    item->m_args,
                    item->m_returnType,
                    userData);
            }
            break;
        default:
            // class...
            continue;
        }

        if (item->m_childs.size() > 0)
        {
            fillNavigationMethodComboBox(item, prefix + "... ");
        }
    }
}

//-------------------------------------------------------------------------------------
// public Slot invoked by outlineModelChanged from EditorWidget or by tabchange etc.
void ScriptDockWidget::updateCodeNavigation(ScriptEditorWidget *editor, QSharedPointer<OutlineItem> rootItem)
{
    if (m_outlineShowNavigation && editor)
    {
        if (m_tab->currentIndex() == m_tab->indexOf(editor))
        {
            int line = 0;
            editor->getCursorPosition(&line, nullptr);

            m_methodBox->blockSignals(true);
            m_classBox->blockSignals(true);
            m_classBox->clear();
            m_methodBox->clear();

            fillNavigationClassComboBox(rootItem, "");

            // check if there is a match in the class list concerning the current line
            OutlineItem *item;
            int rowCandidate = 0; //default is the global class section

            for (int row = 0; row < m_classBox->count(); ++row)
            {
                item = m_classBox->itemData(row, Qt::UserRole)
                    .value<QSharedPointer<OutlineItem>>().data();

                if (item &&
                    item->m_startLineIdx <= line &&
                    item->m_endLineIdx >= line)
                {
                    rowCandidate = row;
                }
            }

            if (rowCandidate >= 0)
            {
                m_classBox->setCurrentIndex(rowCandidate);
                auto parent = m_classBox->itemData(rowCandidate, Qt::UserRole)
                    .value<QSharedPointer<OutlineItem>>();
                fillNavigationMethodComboBox(parent, "");
                rowCandidate = -1;

                for (int row = 0; row < m_methodBox->count(); ++row)
                {
                    item = m_methodBox->itemData(row, Qt::UserRole)
                        .value<QSharedPointer<OutlineItem>>().data();

                    if (item &&
                        item->m_startLineIdx <= line &&
                        item->m_endLineIdx >= line)
                    {
                        rowCandidate = row;
                    }
                }

                if (rowCandidate >= 0)
                {
                    m_methodBox->setCurrentIndex(rowCandidate);
                }
            }

            m_classBox->blockSignals(false);
            m_methodBox->blockSignals(false);
        }
    }
}

//-------------------------------------------------------------------------------------
// Slot called if any entry in the class navigation combobox is selected.
/* The definition of the selected class (if a class is selected) is
highlighted in the current script. If the class has child items, the
method box is initialized with these children.
*/
void ScriptDockWidget::navigatorClassSelected(int row)
{
    QSharedPointer<OutlineItem> classItem =
        m_classBox->itemData(row, Qt::UserRole).value<QSharedPointer<OutlineItem>>();

    if (classItem.isNull())
    {
        m_methodBox->clear();
    }
    else
    {
        const auto editor = getCurrentEditor();

        if (editor)
        {
            editor->showLineAndHighlightWord(
                classItem->m_startLineIdx,
                classItem->m_name);
        }

        m_methodBox->clear();
        fillNavigationMethodComboBox(classItem, "");
    }
}

//-------------------------------------------------------------------------------------
// Slot called if any entry in the method navigation combobox is selected.
/* The definition of the selected method (if a method is selected) is
highlighted in the current script.
*/
void ScriptDockWidget::navigatorMethodSelected(int row)
{
    QSharedPointer<OutlineItem> methodItem =
        m_methodBox->itemData(row, Qt::UserRole)
        .value<QSharedPointer<OutlineItem>>();

    if (!methodItem.isNull())
    {
        const auto editor = getCurrentEditor();

        if (editor)
        {
            editor->showLineAndHighlightWord(
                methodItem->m_startLineIdx,
                methodItem->m_name);
        }
    }
}

//-------------------------------------------------------------------------------------
//!< displays or hides the entire outline navigation bar (class and method combo box)
void ScriptDockWidget::showOutlineNavigationBar(bool show)
{
    if (show)
    {
        ScriptEditorWidget *editorWidget =
            static_cast<ScriptEditorWidget*>(m_tab->widget(m_actTabIndex));

        if (editorWidget)
        {
            // update the content of the navigation combo boxes
            updateCodeNavigation(editorWidget, editorWidget->parseOutline());
        }
    }

    m_classMenuBar->setVisible(show);
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<ito::ScriptEditorStorage> ScriptDockWidget::saveScriptState() const
{
    QList<ito::ScriptEditorStorage> state;
    ScriptEditorWidget *sew;

    for (int idx = 0; idx < m_tab->count(); ++idx)
    {
        sew = static_cast<ScriptEditorWidget *>(m_tab->widget(idx));

        if (sew)
        {
            if (sew->hasNoFilename() == false) //don't save the state of non-saved scripts
            {
                state << sew->saveState();
            }
        }
    }

    return state;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptDockWidget::restoreScriptState(const QList<ito::ScriptEditorStorage> &states)
{
    RetVal retVal;

    if (!retVal.containsError())
    {
        foreach(const ito::ScriptEditorStorage &ses, states)
        {
            QFileInfo fi(ses.filename);

            if (ses.filename.isNull() == false && fi.exists())
            {
                ScriptEditorWidget* sew = new ScriptEditorWidget(m_pBookmarkModel, m_tab);
                if (sew->restoreState(ses).containsError())
                {
                    sew->deleteLater();
                }
                else
                {
                    retVal += appendEditor(sew);
                }
            }
        }
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns a list of filenames, which have been modified in this ScriptDockWidget
/*!
    long description

    \param ignoreUnsavedFiles if true, ignores scripts, which do not have any filename yet
    \param exludeIndex tab-index which should be ignored, set to -1 in order to consider every tab
    \return string list
*/
QStringList ScriptDockWidget::getModifiedFileNames(bool ignoreUnsavedFiles, int excludeIndex) const
{
    QStringList list;
    ScriptEditorWidget* sew;

    for (int i = 0; i < m_tab->count(); i++)
    {
        sew = getEditorByIndex(i);
        if (sew != NULL && sew->isModified() && i != excludeIndex)
        {
            if (!ignoreUnsavedFiles || !sew->hasNoFilename())
            {
                if (sew->hasNoFilename())
                {
                    list.append(sew->getUntitledName());
                }
                else
                {
                    list.append(sew->getFilename());
                }
            }
        }
    }

    return list;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns a list of all filenames of all opened scripts (besides new scripts)
/*!
\return string list
*/
QStringList ScriptDockWidget::getAllFilenames() const
{
    QStringList list;
    ScriptEditorWidget* sew;

    for (int i = 0; i < m_tab->count(); i++)
    {
        sew = getEditorByIndex(i);
        if (sew != NULL)
        {
            if (!sew->hasNoFilename())
            {
                list.append(sew->getFilename());
            }
            else
            {
                list.append(sew->getUntitledName());
            }
        }
    }

    return list;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates new instance of ScriptEditorWidget and appends it to the tab-widget
/*!
    \return result of method appendEditor
    \sa appendEditor
*/
RetVal ScriptDockWidget::newScript()
{
    ScriptEditorWidget* sew = new ScriptEditorWidget(m_pBookmarkModel, m_tab);
    ito::RetVal retval = appendEditor(sew);
    sew->setFocus();
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method to open an existing script which can be indicated by the user by a getOpenFileName-dialog.
/*!
    the script is not directly opened by this method, but the signal openScriptRequest is emitted which invokes a slot in the scriptEditorOrganizer.
    Then the organizer can check, if that filename is already opened in another tab and activate it instead of opening a new editor.

    \return retOk, if filename has been chosen, else retError (also in case of user cancellation)
    \sa (see also) keywords (comma-separated)
*/
RetVal ScriptDockWidget::openScript()
{
    QString fileName;

    fileName = QFileDialog::getOpenFileName(getActiveInstance(), tr("File Open"), QDir::currentPath(), "Python (*.py)");

    if (!fileName.isEmpty())
    {
        QDir::setCurrent(QFileInfo(fileName).path());
        emit (openScriptRequest(fileName, this));
        return RetVal(retOk);
    }

    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! opens a given filename as new tab in this ScriptDockWidget
/*!
    Opens new ScriptEditorWidget, appends it to the tab widget and opens filename in the newly created instance.

    \param silent if true, no warning about invalid filename or invalid file format is shown
    \return retOk if file has been opened, else retError
*/
RetVal ScriptDockWidget::openScript(QString filename, bool silent)
{
    QFileInfo fileInfo(filename);

    if (!fileInfo.exists())
    {
        if (!silent)
        {
            QMessageBox msg(QMessageBox::Warning, tr("Open Script"), tr("The file '%1' could not be found.").arg(filename));
            msg.exec();
        }
        return RetVal(retError);
    }
    else
    {

        if (fileInfo.suffix().toLower() != "py")
        {
            if (!silent)
            {
                QMessageBox msg(QMessageBox::Warning, tr("Open Script"), tr("The file '%1' is not a python script.").arg(filename));
                msg.exec();
            }
            return RetVal(retError);
        }
    }

    ScriptEditorWidget* sew = new ScriptEditorWidget(m_pBookmarkModel, m_tab);

    QString absoluteFilename = fileInfo.absoluteFilePath();

    // under Windows, pathes must not be case sensitive. Therefore
    // filename can be case insensitive. To show the correct name,
    // try to figure out how the case-correct filename would be.
    // Hint: this does not consider the pathes so far.
    QDir path(fileInfo.absolutePath());
    QStringList nameFilters;
    nameFilters << "*.py";
    QFileInfoList filesInfo = path.entryInfoList(nameFilters, QDir::Files);

    foreach(const QFileInfo &f, filesInfo)
    {
        if (QString::compare(f.absoluteFilePath(), absoluteFilename, Qt::CaseInsensitive) == 0)
        {
            absoluteFilename = f.absoluteFilePath();
            break;
        }
    }

    RetVal retValue = sew->openFile(absoluteFilename, false);

    if (retValue.containsError())
    {
        DELETE_AND_SET_NULL(sew);
    }
    else
    {
        appendEditor(sew);
        sew->setFocus();
        sew->reportCurrentCursorAsGoBackNavigationItem("open", sew->getUID());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! tries to save all opened scripts in this ScriptDockWidget
/*!
    First, all unsaved and modified scripts are indentified and listed. Then the user is asked for confirmation of
    saving these modified documents. Then these scripts will be saved, if desired.

    \param askFirst if true, the user is asked for confirmation, else all scripts are silently saved
    \param ignoreNewScripts if true do not consider new scripts, which does not have any filename yet
    \param excludeIndex ignore tab with this index, set it to -1 in order to consider every tab (default: -1)
    \return retOk if all identified scripts could be saved or have been discarded, else retError (in order to cancel the execution)
*/
RetVal ScriptDockWidget::saveAllScripts(bool askFirst, bool ignoreNewScripts, int excludeIndex)
{
    RetVal retValue(retOk);
    ScriptEditorWidget *sew;

    if (askFirst)
    {
        QStringList list = this->getModifiedFileNames(ignoreNewScripts);
        QMessageBox msgBox;

        if (list.size() > 0)
        {
            msgBox.setText(tr("The following files have been changed and should be safed:"));
            msgBox.setInformativeText(list.join("\n"));
            msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
            msgBox.setDefaultButton(QMessageBox::Save);
            int ret = msgBox.exec();

            if (ret & QMessageBox::Cancel)
            {
                return RetVal(retError);
            }
            else if (ret & QMessageBox::Discard)
            {
                return RetVal(retOk);
            }
        }
    }

    for (int i = 0; i < m_tab->count(); i++)
    {
        sew = getEditorByIndex(i);
        if ((!sew->hasNoFilename() || !ignoreNewScripts) && i != excludeIndex)
        {
            retValue += saveTab(i, false, false);
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! closes all opened scripts in this ScriptDockWidget
/*!
    if indicated, scripts will be saved first. Methods returns if saving process fails, else scripts will be closed.

    \param saveFirst if true, unsaved scripts are tried to be saved first
    \param askFirst if true, user will be asked about script saving
    \param ignoreNewScripts if true, new scripts which do not have any filename are not considered in the saving method
    \param excludeIndex tab with given index will not be considered in saving method, default: -1 (consider all scripts)
    \return retOk if every script could be closed, else retError
    \sa saveAllScripts
*/
RetVal ScriptDockWidget::closeAllScripts(bool saveFirst, bool askFirst, bool ignoreNewScripts, int excludeIndex, bool closeScriptWidgetIfLastTabClosed /*= true*/)
{
    RetVal retValue(retOk);

    if (saveFirst)
    {
        retValue += saveAllScripts(askFirst, ignoreNewScripts, excludeIndex);
    }

    if (!retValue.containsError())
    {
        QList<ScriptEditorWidget*> list;
        QList<ScriptEditorWidget*>::iterator it;
        for (int i = 0; i < m_tab->count(); i++)
        {
            if (i != excludeIndex)
            {
                list.append(getEditorByIndex(i));
            }
        }

        for (it = list.begin(); it != list.end(); ++it)
        {
            retValue += closeTab(getIndexByEditor(*it), false, closeScriptWidgetIfLastTabClosed);
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! append given ScriptEditorWidget as new tab
/*!
    \param editorWidget ScriptEditorWidget to append
    \return retOk
*/
RetVal ScriptDockWidget::appendEditor(ScriptEditorWidget* editorWidget)
{
    QString name = editorWidget->getFilename();
    QString toolTip = name;
    if (name == "")
    {
        name = editorWidget->getUntitledName();
        toolTip = name;
    }
    else
    {
        QFileInfo info(name);
        name = info.fileName();
    }
    m_tab->addTab(editorWidget, QIcon(":/files/icons/filePython.png"), name);

    // add the new index to the stackHistory.
    m_stackHistory.prepend(m_tab->count() - 1);

    //!< activate appended tab
    m_tab->setCurrentIndex(m_tab->count() - 1);
    m_tab->setTabToolTip(m_tab->count() - 1, toolTip);
    m_tab->setTabText(m_tab->count() - 1, name);

    connect(editorWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
    connect(editorWidget, SIGNAL(copyAvailable(bool)), this, SLOT(updateEditorActions()));
    connect(editorWidget, SIGNAL(closeRequest(ScriptEditorWidget*, bool)), this, SLOT(tabCloseRequested(ScriptEditorWidget*, bool)));
    connect(editorWidget, SIGNAL(marginChanged()), this, SLOT(editorMarginChanged()));
    connect(editorWidget, SIGNAL(updateActions()), this, SLOT(updateEditorActions()));
    connect(editorWidget, SIGNAL(addGoBackNavigationItem(GoBackNavigationItem)), this, SIGNAL(addGoBackNavigationItem(GoBackNavigationItem)));
    connect(
        editorWidget, &ScriptEditorWidget::tabChangeRequested,
        this, &ScriptDockWidget::tabChangedRequest
    );
    connect(
        editorWidget, &ScriptEditorWidget::findSymbolsShowRequested,
        this, &ScriptDockWidget::mnuFindSymbolsShow);

    // Load the right Class->Method model for this Editor
    connect(editorWidget, &ScriptEditorWidget::outlineModelChanged,
        this, &ScriptDockWidget::updateCodeNavigation);

    updateEditorActions();
    updatePythonActions();

    //scriptModificationChanged(editorWidget->isModified());
    if (editorWidget->isModified())
    {
        m_tab->setWindowModified(true);
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! removes ScriptEditorWidget at given tab position from tab-widget and returns its reference
/*!
    \param index tab index of editor, which should be removed
    \return instance to recently removed ScriptEditorWidget, NULL if index exceeds limits
*/
ScriptEditorWidget* ScriptDockWidget::removeEditor(int index)
{
    if (index < 0 || index >= m_tab->count())
    {
        return nullptr;
    }

    ScriptEditorWidget* removedWidget = static_cast<ScriptEditorWidget*>(m_tab->widget(index));

    // adapt m_stackHistory
    m_stackHistory.removeOne(index); //remove the index

    for (int i = 0; i < m_stackHistory.size(); ++i)
    {
        if (m_stackHistory[i] > index)
        {
            // decrement index at pos i, since index is removed
            m_stackHistory[i]--;
        }
    }

    m_tab->removeTab(index);
    disconnect(removedWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
    disconnect(removedWidget, SIGNAL(copyAvailable(bool)), this, SLOT(updateEditorActions()));
    disconnect(removedWidget, SIGNAL(closeRequest(ScriptEditorWidget*, bool)), this, SLOT(tabCloseRequested(ScriptEditorWidget*, bool)));
    disconnect(removedWidget, SIGNAL(marginChanged()), this, SLOT(editorMarginChanged()));
    disconnect(removedWidget, SIGNAL(updateActions()), this, SLOT(updateEditorActions()));
    disconnect(removedWidget, SIGNAL(addGoBackNavigationItem(GoBackNavigationItem)), this, SIGNAL(addGoBackNavigationItem(GoBackNavigationItem)));
    disconnect(
        removedWidget, &ScriptEditorWidget::tabChangeRequested,
        this, &ScriptDockWidget::tabChangedRequest
    );

    disconnect(
        removedWidget, &ScriptEditorWidget::findSymbolsShowRequested,
        this, &ScriptDockWidget::mnuFindSymbolsShow);

    // Load the right Class->Method model for this Editor
    disconnect(removedWidget, &ScriptEditorWidget::outlineModelChanged,
        this, &ScriptDockWidget::updateCodeNavigation);

    updateEditorActions();
    updatePythonActions();

    if (index > 0)
    {
        m_tab->setCurrentIndex(index - 1);
    }
    else
    {
        m_tab->setCurrentIndex(-1);
    }

    return removedWidget;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! checks whether any editor in this ScriptDockWidget has no filename
/*!
    \return true if any script has no filename, else false
*/
bool ScriptDockWidget::containsNewScripts() const //!< new means unsaved (without filename)
{
    bool newScripts = false;
    ScriptEditorWidget *editorWidget;

    for (int i = 0; i < m_tab->count(); i++)
    {
        editorWidget = static_cast<ScriptEditorWidget*>(m_tab->widget(i));
        if (editorWidget)
        {
            newScripts = newScripts || editorWidget->hasNoFilename();
        }
    }

    return newScripts;
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::tabFilenameOrModificationChanged(int index)
{
    if (index >= 0)
    {
        ScriptEditorWidget *editorWidget = static_cast<ScriptEditorWidget*>(m_tab->widget(index));
        setWindowModified(editorWidget->isModified());

        // ClassNavigator: set the right classes in comboboxes
        updateCodeNavigation(editorWidget, editorWidget->parseOutline());

        if (editorWidget->hasNoFilename())
        {
            if (index == m_actTabIndex)
            {
                setAdvancedWindowTitle(
                    editorWidget->getUntitledName().prepend(" - ").append("[*]"), true
                );
            }


            if (editorWidget->isModified())
            {
                m_tab->setTabText(index, editorWidget->getUntitledName().append("*"));
            }
            else
            {
                m_tab->setTabText(index, editorWidget->getUntitledName());
            }
        }
        else
        {
            if (index == m_actTabIndex)
            {
                setAdvancedWindowTitle(
                    editorWidget->getFilename().prepend(" - ").append("[*]"), true
                );
            }

            QFileInfo info(editorWidget->getFilename());

            if (editorWidget->isModified())
            {
                m_tab->setTabText(index, info.fileName().append("*"));
            }
            else
            {
                m_tab->setTabText(index, info.fileName());
            }
        }
    }
    else
    {
        setWindowModified(false);
        setAdvancedWindowTitle("", true);
    }

    updateEditorActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by tab-widget if current tab changed
/*!
    modifies title of this ScriptDockWidget instance, depending on the active tab.

    \param index tab-index of changed editor
*/
void ScriptDockWidget::currentTabChanged(int index)
{
    m_actTabIndex = index;

    if (m_stackHistory.contains(index))
    {
        // move the current index to the front
        m_stackHistory.removeOne(index);
        m_stackHistory.prepend(index);
    }

    tabFilenameOrModificationChanged(index);

    auto currentEditor = getCurrentEditor();

    if (currentEditor)
    {
        currentScriptCursorPositionChanged();

        // disconnect all previous connections
        disconnect(this, SLOT(currentScriptCursorPositionChanged()));

        connect(currentEditor, &ScriptEditorWidget::cursorPositionChanged, this, &ScriptDockWidget::currentScriptCursorPositionChanged);
    }
    else
    {
        emit statusBarInformationChanged(this, "", -1, -1);
    }
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::currentScriptCursorPositionChanged()
{
    auto currentEditor = getCurrentEditor();

    if (currentEditor)
    {
        auto charsetEncoding = currentEditor->charsetEncoding();
        int line = currentEditor->currentLineNumber() + 1;
        int col = currentEditor->currentColumnNumber() + 1;

        emit statusBarInformationChanged(
            this,
            charsetEncoding.displayNameShort,
            line,
            col
        );

        m_pStatusBarWidget->setText(tr("Ln %1, Col %2, %3 ").arg(line).arg(col).arg(charsetEncoding.displayNameShort));
    }
    else
    {
        emit statusBarInformationChanged(this, "", -1, -1);
        m_pStatusBarWidget->setText("");
    }
}

//-------------------------------------------------------------------------------------
//! slot connected to each ScriptEditorWidget instance. Invoked if any content in any script changed.
/*!
    calls slot currentTabChanged with tab index of scriptEditorWidget that sent the signal or
    the active tab index if no sender is available.

    \sa currentTabChanged
*/
void ScriptDockWidget::scriptModificationChanged(bool /*changed*/)
{
    // in case of save-all or other commands that change other scripts than the active on,
    // this slot needs to know the sender of the signal:
    const QObject *senderObject = sender();

    if (senderObject)
    {
        for (int i = 0; i < m_tab->count(); ++i)
        {
            if (qobject_cast<QObject*>(getEditorByIndex(i)) == senderObject)
            {
                tabFilenameOrModificationChanged(i);
            }
        }
    }
    else
    {
        tabFilenameOrModificationChanged(m_actTabIndex);
    }

    updateEditorActions();
}

//-------------------------------------------------------------------------------------
//! slot invoked if close button of any tab of m_tab (QTabWidgetItom) has been pressed
/*!
    tries to close the tab in question

    \param index tab-index of tab in question
*/
void ScriptDockWidget::tabCloseRequested(int index)
{
    ScriptEditorWidget *sew = getEditorByIndex(index);

    if (sew == nullptr)
    {
        return;
    }

    closeTab(index, true);
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::tabCloseRequested(ScriptEditorWidget* sew, bool ignoreModifications)
{
    if (sew == nullptr)
    {
        return;
    }

    int index = getIndexByEditor(sew);

    closeTab(index, !ignoreModifications);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! public method to close any specific tab with or without saving its script first
/*!
    \param index tab-index of tab in question
    \param saveFirst save changes in editor first (true) or ignore changes (false)
    \return retOk if tab has been saved (or not) and closed, retError if saving failed
    \sa saveTab, removeEditor
*/
RetVal ScriptDockWidget::closeTab(int index, bool saveFirst, bool closeScriptWidgetIfLastTabClosed /*= true*/)
{
    if (index < 0 || index >= m_tab->count())
    {
        return RetVal(retError);
    }

    RetVal retValue(retOk);

    if (saveFirst)
    {
        retValue += saveTab(index, false);
    }

    if (!retValue.containsError())
    {
        ScriptEditorWidget *sew = this->removeEditor(index);
        sew->deleteLater();
        sew = nullptr;
    }

    if (m_tab->count() == 0 && closeScriptWidgetIfLastTabClosed)
    {
        QCloseEvent evt;
        QApplication::sendEvent(this, &evt);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! saves tab
/*!
    \param index tab-index of tab in question
    \param forceSaveAs true if script should be saved with new filename, else false
    \param askFirst true if user must confirm that script will be saved
    \return retOk if script in tab could be successfully saved, else retError
*/
RetVal ScriptDockWidget::saveTab(int index, bool forceSaveAs, bool askFirst)
{
    if (index < 0 || index >= m_tab->count())
    {
        return RetVal(retError);
    }

    ScriptEditorWidget* sew = getEditorByIndex(index);
    if (sew == NULL)
    {
        return RetVal(retError);
    }

    if (forceSaveAs)
    {
        return sew->saveAsFile(askFirst);
    }
    else
    {
        return sew->saveFile(askFirst);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns ScriptEditorWidget by given tab-index
/*!
    long description

    \param index tab-index
    \return reference to ScriptEditorWidget or NULL, if index has been out of boundaries
*/
ScriptEditorWidget* ScriptDockWidget::getEditorByIndex(int index) const
{
    if (index < 0 || index >= m_tab->count())
    {
        return NULL;
    }
    else
    {
        ScriptEditorWidget *sew = static_cast<ScriptEditorWidget *>(m_tab->widget(index));
        return sew;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns tab-index by given reference to ScriptEditorWidget
/*!
    \param sew reference to ScriptEditorWidget
    \return tab-index of given ScriptEditorWidget or -1 if this widget could not be found
*/
int ScriptDockWidget::getIndexByEditor(const ScriptEditorWidget* sew) const
{
    if (sew == NULL)
    {
        return -1;
    }
    for (int i = 0; i < m_tab->count(); i++)
    {
        if (getEditorByIndex(i) == sew)
        {
            return i;
        }
    }

    return -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns reference to current ScriptEditorWidget
/*!
    \return reference to current ScriptEditorWidget or nullptr
*/
ScriptEditorWidget* ScriptDockWidget::getCurrentEditor() const
{
    if (m_tab->count() == 0)
    {
        return nullptr;
    }

    return static_cast<ScriptEditorWidget*>(m_tab->currentWidget());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method is invoked, if a context menu is requested
/*!
    checks if mouse click directly has been located at any tab and if yes actualizes the member m_actTabIndex.
*/
void ScriptDockWidget::tabContextMenuEvent(QContextMenuEvent * event)
{
    QRect tabRectangle;
    event->accept();

    for (int i = 0; i < m_tab->count(); i++)
    {
        tabRectangle = m_tab->getTabBar()->tabRect(i);
        int eventX = event->pos().x();
//qDebug() << "tabRectangle: " << tabRectangle << ", event->pos(): " << event->pos() << ", m_tab->pos():" << m_tab->pos() << ", m_tab->getTabBar()->pos(): " << m_tab->getTabBar()->pos();

//        if (tabRectangle.contains(event->pos() - m_tab->pos() - m_tab->getTabBar()->pos()))
        if (tabRectangle.x() <= eventX && (tabRectangle.x() + tabRectangle.width()) >= eventX)
        {
            m_tab->setCurrentIndex(i);
            m_actTabIndex = i;
        }
    }

    m_tabContextMenu->exec(event->globalPos());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates actions which deal with editor commands but are not dependent on the state of python.
void ScriptDockWidget::updateEditorActions()
{
    int tabCount = m_tab->count();
    const ScriptEditorWidget* sew = getCurrentEditor();

    m_saveAllScriptsAction->setEnabled(false);

    for (int i = 0; i < tabCount; i++)
    {
        if (static_cast<ScriptEditorWidget *>(m_tab->widget(i))->isModified())
        {
            m_saveAllScriptsAction->setEnabled(true);
            break;
        }
    }

    m_saveScriptAction->setEnabled(tabCount > 0 && sew != nullptr && sew->isModified());
    m_saveScriptAsAction->setEnabled(tabCount > 0 && sew != nullptr);
    m_copyAction->setEnabled(sew != nullptr && sew->getCanCopy());
    m_tabCloseAction->setEnabled(m_actTabIndex > -1);
    m_tabCloseAllAction->setEnabled(m_actTabIndex > -1);
    m_findTextExprAction->setEnabled(m_actTabIndex > -1);
    m_gotoAction->setEnabled(m_actTabIndex > -1);
    m_openIconBrowser->setEnabled(m_actTabIndex > -1);
    m_bookmarkToggle->setEnabled(sew != nullptr);

    if (m_pWidgetFindWord != nullptr)
    {
        m_pWidgetFindWord->setFindBarEnabled(m_actTabIndex > -1, false);
    }

    updatePythonActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates actions which deal with python commands or which are dependent on the python state
//! Read-only is also python-dependent.
void ScriptDockWidget::updatePythonActions()
{
    const ScriptEditorWidget* sew = getCurrentEditor();
    bool busy1 = pythonBusy();
    bool busy2 = busy1 && pythonDebugMode() && pythonInWaitingMode();
    int tabCount = m_tab->count();

    m_scriptRunAction->setEnabled(!busy1);
    m_scriptRunSelectionAction->setEnabled(sew && (!busy1 || pythonInWaitingMode()));
    m_scriptDebugAction->setEnabled(!busy1);
    m_scriptStopAction->setEnabled(busy1);
    m_scriptContinueAction->setEnabled(busy2);
    m_scriptStepAction->setEnabled(busy2);
    m_scriptStepOverAction->setEnabled(busy2);
    m_scriptStepOutAction->setEnabled(busy2);

    m_scriptRunSelectionAction->setEnabled(
        sew != nullptr &&
        (!pythonBusy() || pythonInWaitingMode()));

    m_replaceTextExprAction->setEnabled(
        !busy1 &&
        m_actTabIndex > -1);

    m_cutAction->setEnabled(
        sew != nullptr &&
        sew->getCanCopy() &&
        !busy1);

    m_pasteAction->setEnabled(
        tabCount > 0 &&
        !busy1);
    m_undoAction->setEnabled(!busy1 && sew != nullptr && sew->isUndoAvailable());
    m_redoAction->setEnabled(!busy1 && sew != nullptr && sew->isRedoAvailable());
    m_commentAction->setEnabled(!busy1 && tabCount > 0 && sew != nullptr);
    m_uncommentAction->setEnabled(!busy1 && tabCount > 0 && sew != nullptr);
    m_indentAction->setEnabled(!busy1 && tabCount > 0 && sew != nullptr);
    m_unindentAction->setEnabled(!busy1 && tabCount > 0 && sew != nullptr);

    m_autoCodeFormatAction->setEnabled(
        !busy1 &&
        m_autoCodeFormatCmd != "" &&
        tabCount > 0);

    m_pyDocstringGeneratorAction->setEnabled(
        !busy1 &&
        tabCount > 0 &&
        sew != nullptr &&
        sew->currentLineCanHaveDocstring());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates actions before context menu of tabs is shown
void ScriptDockWidget::updateTabContextActions()
{
    bool notFirst = m_actTabIndex > 0;
    bool notLast = m_actTabIndex > -1 && m_actTabIndex < m_tab->count() - 1;
    bool tabExist = m_actTabIndex > -1;
    m_tabMoveLeftAction->setEnabled(notFirst);
    m_tabMoveRightAction->setEnabled(notLast);
    m_tabMoveFirstAction->setEnabled(notFirst);
    m_tabMoveLastAction->setEnabled(notLast);

    m_tabCloseAction->setEnabled(tabExist);
    m_tabCloseAllAction->setEnabled(tabExist);
    m_tabCloseOthersAction->setEnabled(tabExist && m_tab->count() > 1);

    ScriptEditorWidget *sew = NULL;
    if (m_actTabIndex > -1)
    {
        sew = static_cast<ScriptEditorWidget *>(m_tab->widget(m_actTabIndex));
    }

    m_tabDockAction->setVisible(!docked());
    m_tabUndockAction->setVisible(true); //docked());

    m_saveScriptAction->setEnabled(m_tab->count()>0 && sew != NULL && sew->isModified());
    m_saveScriptAsAction->setEnabled(m_tab->count()>0 && sew != NULL);
    m_copyFilename->setEnabled(m_tab->count()>0 && sew != NULL && !sew->hasNoFilename());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates actions
void ScriptDockWidget::createActions()
{
    m_tabMoveLeftAction = new ShortcutAction(QIcon(":/arrows/icons/1leftarrow.png"), tr("Move Left"), this);
    m_tabMoveLeftAction->connectTrigger(this, SLOT(mnuTabMoveLeft()));

    m_tabMoveRightAction = new ShortcutAction(QIcon(":/arrows/icons/1rightarrow.png"), tr("Move Right"), this);
    m_tabMoveRightAction->connectTrigger(this, SLOT(mnuTabMoveRight()));

    m_tabMoveFirstAction = new ShortcutAction(QIcon(":/arrows/icons/2leftarrow.png"), tr("Move First"), this);
    m_tabMoveFirstAction->connectTrigger(this, SLOT(mnuTabMoveFirst()));

    m_tabMoveLastAction = new ShortcutAction(QIcon(":/arrows/icons/2rightarrow.png"), tr("Move Last"), this);
    m_tabMoveLastAction->connectTrigger(this, SLOT(mnuTabMoveLast()));

    m_tabCloseAction = new ShortcutAction(QIcon(":/files/icons/close.png"), tr("Close"),
        this, QKeySequence::Close, Qt::WidgetWithChildrenShortcut);
    m_tabCloseAction->connectTrigger(this, SLOT(mnuTabClose()));

    m_tabCloseOthersAction = new ShortcutAction(QIcon(), tr("Close Others"), this);
    m_tabCloseOthersAction->connectTrigger(this, SLOT(mnuTabCloseOthers()));

    m_tabCloseAllAction = new ShortcutAction(QIcon(":/plugins/icons/closeAll.png"), tr("Close All"), this);
    m_tabCloseAllAction->connectTrigger(this, SLOT(mnuTabCloseAll()));

    m_tabDockAction = new ShortcutAction(QIcon(":/dockWidget/icons/dockButtonGlyph.png"), tr("Dock"), this);
    m_tabDockAction->connectTrigger(this, SLOT(mnuTabDock()));

    m_tabUndockAction = new ShortcutAction(QIcon(":/dockWidget/icons/undockButtonGlyph.png"), tr("Undock"), this);
    m_tabUndockAction->connectTrigger(this, SLOT(mnuTabUndock()));

    m_newScriptAction = new ShortcutAction(QIcon(":/files/icons/new.png"), tr("New"),
        this, QKeySequence::New, Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_newScriptAction->connectTrigger(this, SLOT(mnuNewScript()));

    m_openScriptAction = new ShortcutAction(QIcon(":/files/icons/open.png"), tr("Open"),
        this, QKeySequence::Open, Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_openScriptAction->connectTrigger(this, SLOT(mnuOpenScript()));

    m_saveScriptAction = new ShortcutAction(QIcon(":/files/icons/fileSave.png"), tr("Save"),
        this, QKeySequence::Save, Qt::WidgetWithChildrenShortcut);
    m_saveScriptAction->connectTrigger(this, SLOT(mnuSaveScript()));

    m_saveScriptAsAction = new ShortcutAction(QIcon(":/files/icons/fileSaveAs.png"), tr("Save As..."),
        this, QKeySequence::SaveAs, Qt::WidgetWithChildrenShortcut);
    m_saveScriptAsAction->connectTrigger(this, SLOT(mnuSaveScriptAs()));

    m_saveAllScriptsAction = new ShortcutAction(QIcon(":/files/icons/fileSaveAll.png"), tr("Save All"), this);
    m_saveAllScriptsAction->connectTrigger(this, SLOT(mnuSaveAllScripts()));

    m_printAction = new ShortcutAction(QIcon(":/plots/icons/print.png"), tr("Print..."),
        this, QKeySequence::Print, Qt::WidgetWithChildrenShortcut);
    m_printAction->connectTrigger(this, SLOT(mnuPrint()));

    m_cutAction = new ShortcutAction(QIcon(":/editor/icons/editCut.png"), tr("Cut"),
        this, QKeySequence::Cut, Qt::WidgetWithChildrenShortcut);
    m_cutAction->connectTrigger(this, SLOT(mnuCut()));

    m_copyAction = new ShortcutAction(QIcon(":/editor/icons/editCopy.png"), tr("Copy"),
        this, QKeySequence::Copy, Qt::WidgetWithChildrenShortcut);
    m_copyAction->connectTrigger(this, SLOT(mnuCopy()));

    m_pasteAction = new ShortcutAction(QIcon(":/editor/icons/editPaste.png"), tr("Paste"),
        this, QKeySequence::Paste, Qt::WidgetWithChildrenShortcut);
    m_pasteAction->connectTrigger(this, SLOT(mnuPaste()));

    m_undoAction = new ShortcutAction(QIcon(":/editor/icons/editUndo.png"), tr("Undo"),
        this, QKeySequence::Undo, Qt::WidgetWithChildrenShortcut);
    m_undoAction->connectTrigger(this, SLOT(mnuUndo()));

    m_redoAction = new ShortcutAction(QIcon(":/editor/icons/editRedo.png"), tr("Redo"),
        this, QKeySequence::Redo, Qt::WidgetWithChildrenShortcut);
    m_redoAction->connectTrigger(this, SLOT(mnuRedo()));

    m_commentAction = new ShortcutAction(QIcon(":/editor/icons/editComment.png"), tr("Comment"),
        this, QKeySequence(tr("Ctrl+R", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_commentAction->connectTrigger(this, SLOT(mnuComment()));

    m_uncommentAction = new ShortcutAction(QIcon(":/editor/icons/editUncomment.png"), tr("Uncomment"),
        this, QKeySequence(tr("Ctrl+Shift+R", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_uncommentAction->connectTrigger(this, SLOT(mnuUncomment()));

    m_indentAction = new ShortcutAction(QIcon(":/editor/icons/editIndent.png"), tr("Indent"),
        this, QKeySequence(tr("Tab", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_indentAction->connectTrigger(this, SLOT(mnuIndent()));

    m_unindentAction = new ShortcutAction(QIcon(":/editor/icons/editUnindent.png"), tr("Unindent"),
        this, QKeySequence(tr("Shift+Tab", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_unindentAction->connectTrigger(this, SLOT(mnuUnindent()));

    m_autoCodeFormatAction = new ShortcutAction(QIcon(":/editor/icons/leftAlign.png"), tr("Auto Format File"),
        this, QKeySequence(tr("Ctrl+Alt+I", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_autoCodeFormatAction->connectTrigger(this, SLOT(mnuPyCodeFormatting()));

    m_pyDocstringGeneratorAction = new ShortcutAction(QIcon(), tr("Generate Docstring"),
        this, QKeySequence(tr("Ctrl+Alt+D", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_pyDocstringGeneratorAction->connectTrigger(this, SLOT(mnuPyDocstringGenerator()));

    m_scriptRunAction = new ShortcutAction(QIcon(":/script/icons/runScript.png"), tr("Run"),
        this, QKeySequence(tr("F5", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptRunAction->connectTrigger(this, SLOT(mnuScriptRun()));

    m_scriptRunSelectionAction = new ShortcutAction(QIcon(":/script/icons/runScript.png"), tr("Run Selection"),
        this, QKeySequence(tr("F9", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptRunSelectionAction->connectTrigger(this, SLOT(mnuScriptRunSelection()));

    m_scriptDebugAction = new ShortcutAction(QIcon(":/script/icons/debugScript.png"), tr("Debug"),
        this, QKeySequence(tr("F6", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptDebugAction->connectTrigger(this, SLOT(mnuScriptDebug()));

    m_scriptStopAction = new ShortcutAction(QIcon(":/script/icons/stopScript.png"), tr("Stop"),
        this, QKeySequence(tr("Shift+F5", "QShortcut")), Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_scriptStopAction->connectTrigger(this, SLOT(mnuScriptStop()));

    m_scriptContinueAction = new ShortcutAction(QIcon(":/script/icons/continue.png"), tr("Continue"),
        this, QKeySequence(tr("F6", "QShortcut")), Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_scriptContinueAction->connectTrigger(this, SLOT(mnuScriptContinue()));

    m_scriptStepAction = new ShortcutAction(QIcon(":/script/icons/step.png"), tr("Step"),
        this, QKeySequence(tr("F11", "QShortcut")), Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_scriptStepAction->connectTrigger(this, SLOT(mnuScriptStep()));

    m_scriptStepOverAction = new ShortcutAction(QIcon(":/script/icons/stepOver.png"), tr("Step Over"),
        this, QKeySequence(tr("F10", "QShortcut")), Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_scriptStepOverAction->connectTrigger(this, SLOT(mnuScriptStepOver()));

    m_scriptStepOutAction = new ShortcutAction(QIcon(":/script/icons/stepOut.png"), tr("Step Out"),
        this, QKeySequence(tr("Shift+F11", "QShortcut")), Qt::WidgetShortcut, Qt::WidgetWithChildrenShortcut);
    m_scriptStepOutAction->connectTrigger(this, SLOT(mnuScriptStepOut()));

    m_findTextExprAction = new ShortcutAction(QIcon(":/editor/icons/find.png"), tr("Quick Search..."),
        this, QKeySequence::Find, Qt::WidgetWithChildrenShortcut);
    m_findTextExprAction->connectTrigger(this, SLOT(mnuFindTextExpr()));
//    m_findTextExprAction->action()->setCheckable(true);

    // To add a secound shortcut. It works, but I don't know why!
    m_findTextExprActionSC = new ShortcutAction(QIcon(":/editor/icons/find.png"), tr("Quick Search..."),
        this, QKeySequence(tr("F3", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_findTextExprActionSC->connectTrigger(this, SLOT(mnuFindTextExpr()));

    m_replaceTextExprAction = new ShortcutAction(QIcon(":/editor/icons/editReplace.png"), tr("Find And Replace..."),
        this, QKeySequence(tr("Ctrl+H", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_replaceTextExprAction->connectTrigger(this, SLOT(mnuReplaceTextExpr()));

    m_openIconBrowser = new ShortcutAction(QIcon(":/editor/icons/iconList.png"), tr("Icon &Browser..."),
        this, QKeySequence(tr("Ctrl+B", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_openIconBrowser->connectTrigger(this, SLOT(mnuOpenIconBrowser()));

    m_gotoAction = new ShortcutAction(QIcon(), tr("Goto..."),
        this, QKeySequence(tr("Ctrl+G", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_gotoAction->connectTrigger(this, SLOT(mnuGoto()));

    m_bookmarkToggle = new ShortcutAction(QIcon(":/bookmark/icons/bookmarkToggle.png"), tr("&Toggle Bookmark"), this);
    m_bookmarkToggle->connectTrigger(this, SLOT(mnuToggleBookmark()));

    m_insertCodecAct = new ShortcutAction(tr("&Insert Codec..."), this);
    m_insertCodecAct->connectTrigger(this, SLOT(mnuInsertCodec()));

    m_copyFilename = new ShortcutAction(QIcon(":/editor/icons/editCopy.png"), tr("Copy Filename"), this);
    m_copyFilename->connectTrigger(this, SLOT(mnuCopyFilename()));

    m_findSymbols = new ShortcutAction(QIcon(":/classNavigator/icons/at.png"), tr("Fast Symbol Search..."),
        this, QKeySequence(tr("Ctrl+D", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_findSymbols->connectTrigger(this, SLOT(mnuFindSymbolsShow()));

    updatePythonActions();
    updateTabContextActions();
    updateEditorActions();
}

//-------------------------------------------------------------------------------------
/*Slot aboutToOpen*/
void ScriptDockWidget::menuLastFilesAboutToShow()
{
    // Delete old actions
    for (int i = 0; i < m_lastFilesMenu->actions().length(); ++i)
    {
        m_lastFilesMenu->actions().at(i)->deleteLater();
    }

    m_lastFilesMenu->clear();

    // Get StringList of last Files
    QStringList fileList;
    QObject *seoO = AppManagement::getScriptEditorOrganizer();

    if (seoO)
    {
        ScriptEditorOrganizer *sEO = qobject_cast<ScriptEditorOrganizer*>(seoO);

        if (sEO)
        {
            if (sEO->getRecentlyUsedFiles().isEmpty())
            {
                QAction *a = m_lastFilesMenu->addAction(tr("No Entries"));
                a->setEnabled(false);
            }
            else
            {
                ShortcutAction *a;

                // Create new menus
                foreach(const QString &path, sEO->getRecentlyUsedFiles())
                {
                    QString displayedPath = path;
                    IOHelper::elideFilepathMiddle(displayedPath, 200);
                    a = new ShortcutAction(QIcon(":/files/icons/filePython.png"), displayedPath, this);
                    m_lastFilesMenu->addAction(a->action());
                    connect(a->action(), &QAction::triggered, [=]() {
                        lastFileOpen(path);
                    });
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Slot that is invoked by the lastfile Buttons over the signalmapper
void ScriptDockWidget::lastFileOpen(const QString &path)
{
    QString fileName;

    fileName = path;

    if (!fileName.isEmpty())
    {
        QDir::setCurrent(QFileInfo(fileName).path());
        emit (openScriptRequest(fileName, this));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! create menus
void ScriptDockWidget::createMenus()
{
    m_fileMenu = getMenuBar()->addMenu(tr("&File"));
    m_fileMenu->addAction(m_newScriptAction->action());
    m_fileMenu->addAction(m_openScriptAction->action());
    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_saveScriptAction->action());
    m_fileMenu->addAction(m_saveScriptAsAction->action());
    m_fileMenu->addAction(m_saveAllScriptsAction->action());

    // dynamically created Menu with the last files
    m_lastFilesMenu = m_fileMenu->addMenu(QIcon(":/files/icons/filePython.png"), tr("Recently Used Files"));
    connect(this->m_lastFilesMenu, SIGNAL(aboutToShow()), this, SLOT(menuLastFilesAboutToShow()));
    // Add these menus dynamically

    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_printAction->action());
    m_fileMenu->addSeparator();
    m_fileMenu->addAction(m_tabCloseAction->action());
    m_fileMenu->addAction(m_tabCloseAllAction->action());

//    m_viewMenu = getMenuBar()->addMenu(tr("&View"));
//    m_viewMenu->addAction();
/*    QMenu *dockWidgets = createPopupMenu();
    if (dockWidgets)
    {
        dockWidgets->menuAction()->setIcon(QIcon(":/application/icons/preferences-general.png"));
        dockWidgets->menuAction()->setText(tr("Toolboxes"));
        m_viewMenu->addMenu(dockWidgets);
    }*/

    m_editMenu = getMenuBar()->addMenu(tr("&Edit"));
    m_editMenu->addAction(m_undoAction->action());
    m_editMenu->addAction(m_redoAction->action());
    m_editMenu->addSeparator();
    m_editMenu->addAction(m_cutAction->action());
    m_editMenu->addAction(m_copyAction->action());
    m_editMenu->addAction(m_pasteAction->action());
    m_editMenu->addSeparator();
    m_editMenu->addAction(m_commentAction->action());
    m_editMenu->addAction(m_uncommentAction->action());
    m_editMenu->addAction(m_indentAction->action());
    m_editMenu->addAction(m_unindentAction->action());
    m_editMenu->addAction(m_autoCodeFormatAction->action());
    m_editMenu->addAction(m_pyDocstringGeneratorAction->action());
    m_editMenu->addSeparator();
    m_editMenu->addAction(m_findTextExprAction->action());
    m_editMenu->addAction(m_replaceTextExprAction->action());
    m_editMenu->addAction(m_gotoAction->action());
    m_editMenu->addAction(m_findSymbols->action());
    m_editMenu->addAction(m_openIconBrowser->action());
    m_editMenu->addAction(m_insertCodecAct->action());
    m_editMenu->addSeparator();
    m_bookmark = m_editMenu->addMenu(QIcon(":/bookmark/icons/bookmark.png"), tr("Bookmark"));
    m_bookmark->addAction(m_bookmarkToggle->action());
    m_bookmark->addAction(m_pBookmarkModel->bookmarkPreviousAction());
    m_bookmark->addAction(m_pBookmarkModel->bookmarkNextAction());
    m_bookmark->addAction(m_pBookmarkModel->bookmarkClearAllAction());
    m_bookmark->addSeparator();
    m_bookmark->addAction(m_commonActions.actNavigationBackward);
    m_bookmark->addAction(m_commonActions.actNavigationForward);

    m_scriptMenu = getMenuBar()->addMenu(tr("&Script"));
    m_scriptMenu->addAction(m_scriptRunAction->action());
    m_scriptMenu->addAction(m_scriptRunSelectionAction->action());
    m_scriptMenu->addAction(m_scriptDebugAction->action());
    m_scriptMenu->addAction(m_scriptStopAction->action());
    m_scriptMenu->addSeparator();
    m_scriptMenu->addAction(m_scriptContinueAction->action());
    m_scriptMenu->addAction(m_scriptStepAction->action());
    m_scriptMenu->addAction(m_scriptStepOverAction->action());
    m_scriptMenu->addAction(m_scriptStepOutAction->action());

    m_winMenu = getMenuBar()->addMenu(tr("&Windows"));
    if (m_actStayOnTop)
    {
        m_winMenu->addAction(m_actStayOnTop);
    }
    if (m_actStayOnTopOfApp)
    {
        m_winMenu->addAction(m_actStayOnTopOfApp);
    }

    m_tabContextMenu = new QMenu(this);
    m_tabContextMenu->addAction(m_tabMoveLeftAction->action());
    m_tabContextMenu->addAction(m_tabMoveRightAction->action());
    m_tabContextMenu->addAction(m_tabMoveFirstAction->action());
    m_tabContextMenu->addAction(m_tabMoveLastAction->action());
    m_tabContextMenu->addSeparator();
    m_tabContextMenu->addAction(m_copyFilename->action());
    m_tabContextMenu->addAction(m_saveScriptAction->action());
    m_tabContextMenu->addAction(m_saveScriptAsAction->action());
    m_tabContextMenu->addAction(m_tabCloseAction->action());
    m_tabContextMenu->addAction(m_tabCloseOthersAction->action());
    m_tabContextMenu->addAction(m_tabCloseAllAction->action());
    m_tabContextMenu->addSeparator();
    m_tabContextMenu->addAction(m_tabDockAction->action());
    m_tabContextMenu->addAction(m_tabUndockAction->action());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! create toolbars
void ScriptDockWidget::createToolBars()
{
    m_fileToolBar = new QToolBar(tr("File Toolbar"), this);
    addToolBar(m_fileToolBar, "fileToolBar");
    m_fileToolBar->addAction(m_newScriptAction->action());
    m_fileToolBar->addAction(m_openScriptAction->action());
    m_fileToolBar->addAction(m_saveScriptAction->action());
    m_fileToolBar->addAction(m_saveScriptAsAction->action());
    m_fileToolBar->addAction(m_saveAllScriptsAction->action());
    m_fileToolBar->addAction(m_printAction->action());
    m_fileToolBar->setFloatable(false);

    m_editToolBar = new QToolBar(tr("Edit Toolbar"), this);
    addToolBar(m_editToolBar, "editToolBar");
    m_editToolBar->addAction(m_cutAction->action());
    m_editToolBar->addAction(m_copyAction->action());
    m_editToolBar->addAction(m_pasteAction->action());
    m_editToolBar->addAction(m_undoAction->action());
    m_editToolBar->addAction(m_redoAction->action());
    m_editToolBar->addAction(m_findTextExprAction->action());
    m_editToolBar->addAction(m_replaceTextExprAction->action());
    m_editToolBar->addAction(m_openIconBrowser->action());
    m_editToolBar->addAction(m_findSymbols->action());
    m_editToolBar->addAction(m_autoCodeFormatAction->action());
    m_editToolBar->setFloatable(false);

    m_scriptToolBar = new QToolBar(tr("Script Toolbar"), this);
    addToolBar(m_scriptToolBar, "scriptToolBar");
    m_scriptToolBar->addAction(m_scriptRunAction->action());
//    m_scriptToolBar->addAction(m_scriptRunSelectionAction->action());
    m_scriptToolBar->addAction(m_scriptDebugAction->action());
    m_scriptToolBar->addAction(m_scriptStopAction->action());
    m_scriptToolBar->addAction(m_scriptContinueAction->action());
    m_scriptToolBar->addAction(m_scriptStepAction->action());
    m_scriptToolBar->addAction(m_scriptStepOverAction->action());
    m_scriptToolBar->addAction(m_scriptStepOutAction->action());
    m_scriptToolBar->setFloatable(false);

    m_bookmarkToolBar = new QToolBar(tr("Bookmark and Navigation Toolbar"), this);
    addToolBar(m_bookmarkToolBar, "bookmarkToolBar");
    m_bookmarkToolBar->addAction(m_commonActions.actNavigationBackward);
    m_bookmarkToolBar->addAction(m_commonActions.actNavigationForward);
    m_bookmarkToolBar->addSeparator();
    m_bookmarkToolBar->addAction(m_bookmarkToggle->action());
    m_bookmarkToolBar->addAction(m_pBookmarkModel->bookmarkPreviousAction());
    m_bookmarkToolBar->addAction(m_pBookmarkModel->bookmarkNextAction());
    m_bookmarkToolBar->addAction(m_pBookmarkModel->bookmarkClearAllAction());
    m_bookmarkToolBar->setFloatable(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! init status bar \todo right now, this is an empty method
void ScriptDockWidget::createStatusBar()
{
    m_pStatusBarWidget = new QLabel(this);
    m_pStatusBarWidget->setProperty(statusBarStatePropertyName, 0);
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::windowStateChanged(bool windowNotToolbox)
{
    int state = m_pStatusBarWidget->property(statusBarStatePropertyName).toInt();

    if (windowNotToolbox)
    {
        m_pStatusBarWidget->setVisible(true);

        switch (state)
        {
        case 0: //not added yet
        {
            QStatusBar* sb = getCanvas()->statusBar();
            sb->addPermanentWidget(m_pStatusBarWidget);
            sb->setVisible(true);
            m_pStatusBarWidget->setProperty(statusBarStatePropertyName, 1);
            break;
        }
        case 1: //already added to own status bar
            break;
        }
    }
    else
    {
        m_pStatusBarWidget->setVisible(false);

        switch (state)
        {
        case 0: //not added yet
        {
            // todo: add to status bar of main window
            break;
        }
        case 1: // currently added to own status bar -> shift it to main window
        {
            QStatusBar* sb = getCanvas()->statusBar();
            sb->setVisible(false);
            sb->removeWidget(m_pStatusBarWidget);
            m_pStatusBarWidget->setProperty(statusBarStatePropertyName, 0);
        }
        break;
        }
    }

    // force the emit of new line, column and encoding updates.
    // This has to be done with a small delay, since the main window connects
    // to the statusBarInformationChanged signal after the call to this method.
    // However, this connection has to be established before currentScriptCursorPositionChanged
    // is called, to send the current values to the main window (if docked).
    QTimer::singleShot(20, this, &ScriptDockWidget::currentScriptCursorPositionChanged);

}

//----------------------------------------------------------------------------------------------------------------------------------
//! activates tab with script whose filename corresponds to the filename parameter (or the UID, if >= 0 for scripts without current filename).
/*!
    \param filename Filename of the script which should be activated
    \param line is the marked debugging line (default: -1, no arrow)
    \param UID if >= 0 and if a script has no filename, its UID is compared to the given one
    \return true if filename has been found and activated, else false.
*/
bool ScriptDockWidget::activateTabByFilename(const QString &filename, int currentDebugLine /* = -1*/, int UID /* = -1*/)
{
    ScriptEditorWidget *sew = NULL;
    QString temp2;
    QFileInfo finfo1(filename);
    QString filename2 = finfo1.canonicalFilePath().toLower();
    QFileInfo finfo2;
    bool found = false;

    for (int i = 0; i < m_tab->count(); i++)
    {
        sew = static_cast<ScriptEditorWidget *>(m_tab->widget(i));

        if (filename2 != "" && !sew->hasNoFilename())
        {
            finfo2.setFile(sew->getFilename());
            temp2 = finfo2.canonicalFilePath().toLower();

            if (filename2 == temp2)
            {
                m_tab->setCurrentIndex(i);
                found = true;
                break;
            }
        }
        else if (UID >= 0)
        {
            if (sew->getUID() == UID)
            {
                m_tab->setCurrentIndex(i);
                found = true;
                break;
            }
        }
        else
        {
            if (filename == sew->getUntitledName())
            {
                m_tab->setCurrentIndex(i);
                found = true;
                break;
            }
        }
    }

    if (found && sew)
    {
        raiseAndActivate();

        if (currentDebugLine >= 0)
        {
            sew->pythonDebugPositionChanged(filename2, currentDebugLine);
        }

        return true;
    }

    return false;
}

//-------------------------------------------------------------------------------------
bool ScriptDockWidget::activeTabEnsureLineVisible(
    const int lineNr,
    bool errorMessageClick /*= false*/,
    bool showSelectedCallstackLine /*= false*/)
{
    if (m_actTabIndex >= 0)
    {
        ScriptEditorWidget *sew = static_cast<ScriptEditorWidget *>(m_tab->widget(m_actTabIndex));

        if (sew)
        {
            if (showSelectedCallstackLine &&
                currentSelectedCallstackLineEditor.data() != sew &&
                currentSelectedCallstackLineEditor.data())
            {
                currentSelectedCallstackLineEditor->removeCurrentCallstackLine();
            }

            sew->setCursorPosAndEnsureVisible(lineNr, errorMessageClick, showSelectedCallstackLine);

            if (showSelectedCallstackLine)
            {
                currentSelectedCallstackLineEditor = QPointer<ScriptEditorWidget>(sew);
            }

            return true;
        }
    }

    return false;
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::activeTabShowLineAndHighlightWord(
    const int line,
    const QString &highlightedText,
    Qt::CaseSensitivity caseSensitivity /*= Qt::CaseInsensitive*/)
{
    if (m_actTabIndex >= 0)
    {
        ScriptEditorWidget *sew = static_cast<ScriptEditorWidget *>(m_tab->widget(m_actTabIndex));

        if (sew)
        {
            sew->showLineAndHighlightWord(line, highlightedText, caseSensitivity);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// menu slots:
//! moves active tab by one position to the left
void ScriptDockWidget::mnuTabMoveLeft()
{
    if (m_actTabIndex > 0)
    {
        m_tab->getTabBar()->moveTab(m_actTabIndex, m_actTabIndex - 1);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! moves active tab by one position to the right
void ScriptDockWidget::mnuTabMoveRight()
{
    if (m_actTabIndex < m_tab->count() - 1)
    {
        m_tab->getTabBar()->moveTab(m_actTabIndex, m_actTabIndex + 1);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Open the icon browser
void ScriptDockWidget::mnuOpenIconBrowser()
{
    DialogIconBrowser *iconBrowser = new DialogIconBrowser(getCanvas());
    connect(iconBrowser, SIGNAL(sendIconBrowserText(QString)), this, SLOT(insertIconBrowserText(QString)));

    if (iconBrowser->exec())
    {

    }

    DELETE_AND_SET_NULL(iconBrowser);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! moves active tab to the first position
void ScriptDockWidget::mnuTabMoveFirst()
{
    if (m_actTabIndex >= 0)
    {
       m_tab->getTabBar()->moveTab(m_actTabIndex, 0);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! moves active tab to the last position
void ScriptDockWidget::mnuTabMoveLast()
{
    if (m_actTabIndex <= m_tab->count() - 1)
    {
        m_tab->getTabBar()->moveTab(m_actTabIndex, m_tab->count() - 1);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! closes active tab
void ScriptDockWidget::mnuTabClose()
{
    if (m_actTabIndex >= 0)
    {
        tabCloseRequested (m_actTabIndex);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! closes every tab besides active one (asks for saving tabs, which should be closed)
void ScriptDockWidget::mnuTabCloseOthers()
{
    closeAllScripts(true, true, false, m_actTabIndex);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! closes every tab, asks for saving first
void ScriptDockWidget::mnuTabCloseAll()
{
    closeAllScripts(true, true, false);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! dock the active tab and closes this ScriptDockWidget, if it is not docked and empty after docking
void ScriptDockWidget::mnuTabDock()
{
    emit (dockScriptTab(this, m_actTabIndex, !docked()));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! undock the active tab and closes this ScriptDockWidget, if it is not docked and empty after docking
void ScriptDockWidget::mnuTabUndock()
{
    bool undockToNewScriptWindow = !docked();
    emit(undockScriptTab(this, m_actTabIndex, undockToNewScriptWindow, true)); // !docked()));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to open a new script
void ScriptDockWidget::mnuNewScript()
{
    newScript();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to open an existing script
void ScriptDockWidget::mnuOpenScript()
{
    openScript();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to save active script
void ScriptDockWidget::mnuSaveScript()
{
    ScriptEditorWidget *sew = getEditorByIndex(m_actTabIndex);
    if (sew != NULL)
    {
        sew->saveFile(false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to save active script with new filename (save as)
void ScriptDockWidget::mnuSaveScriptAs()
{
    ScriptEditorWidget *sew = getEditorByIndex(m_actTabIndex);
    if (sew != NULL)
    {
        ito::RetVal retval = sew->saveAsFile(false);

        if (retval == ito::retOk)
        {
            currentTabChanged(m_actTabIndex);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to save all opened scripts
void ScriptDockWidget::mnuSaveAllScripts()
{
    ScriptEditorWidget *sew;
    RetVal retValue(retOk);
    for (int i = 0; i < m_tab->count() && !retValue.containsError(); ++i)
    {
        sew = getEditorByIndex(i);
        if (sew != NULL)
        {
            retValue += sew->saveFile(false);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuPrint()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->print();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a cut command in active script editor
void ScriptDockWidget::mnuCut()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuCut();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a copy command in active script editor
void ScriptDockWidget::mnuCopy()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuCopy();
    }
}

//-------------------------------------------------------------------------------------
//! slot invoked to execute a paste command in active script editor
void ScriptDockWidget::mnuPaste()
{
    ScriptEditorWidget *sew = getCurrentEditor();

    if (sew != nullptr)
    {
        sew->menuPaste();
    }
}

//-------------------------------------------------------------------------------------
//! slot invoked to execute an undo command in active script editor
void ScriptDockWidget::mnuUndo()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    sew->startUndoRedo(true);
}

//-------------------------------------------------------------------------------------
//! slot invoked to execute a redo command in active script editor
void ScriptDockWidget::mnuRedo()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    sew->startUndoRedo(false);
}

//-------------------------------------------------------------------------------------
//! slot invoked to execute a comment command in active script editor
void ScriptDockWidget::mnuComment()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuComment();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute an uncomment command in active script editor
void ScriptDockWidget::mnuUncomment()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuUncomment();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute an indentation command in active script editor
void ScriptDockWidget::mnuIndent()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuIndent();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute an unindentation command in active script editor
void ScriptDockWidget::mnuUnindent()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuUnindent();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to run the active script in python engine
void ScriptDockWidget::mnuScriptRun()
{
    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew == NULL) return;

    RetVal retValue(retOk);
    if (sew->hasNoFilename())
    {
        retValue += sew->saveAsFile(true);
    }

    if (!retValue.containsError())
    {
        emit (pythonRunFileRequest(sew->getFilename()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuScriptRunSelection()
{
    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew == NULL) return;

    sew->menuRunSelection();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to debug the active script in python engine
void ScriptDockWidget::mnuScriptDebug()
{
    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew == NULL) return;

    RetVal retValue(retOk);
    if (sew->hasNoFilename())
    {
        retValue += sew->saveAsFile(true);
    }

    if (!retValue.containsError())
    {
        emit (pythonDebugFileRequest(sew->getFilename()));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to stop python script execution
void ScriptDockWidget::mnuScriptStop()
{
    PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyeng)
    {
        pyeng->pythonInterruptExecutionThreadSafe();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to continue debugging process if actually waiting at breakpoint
void ScriptDockWidget::mnuScriptContinue()
{
    emit (pythonDebugCommand(ito::pyDbgContinue));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a python debugging step
void ScriptDockWidget::mnuScriptStep()
{
    emit (pythonDebugCommand(ito::pyDbgStep));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a python debugging step over
void ScriptDockWidget::mnuScriptStepOver()
{
    emit (pythonDebugCommand(ito::pyDbgStepOver));
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a python debugging step out
void ScriptDockWidget::mnuScriptStepOut()
{
    emit (pythonDebugCommand(ito::pyDbgStepOut));
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuFindTextExpr()
{
    m_findTextExprActionSC->setEnabled(false);
    m_pWidgetFindWord->show();

    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew != NULL)
    {
        int lineFrom = -1;
        int lineTo = -1;
        int indexFrom = -1;
        int indexTo = -1;
        bool multiLineSelection = false;
        QString defaultText = "";

        //check whether text has been marked
        sew->getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

        if (lineFrom >= 0 && lineTo == lineFrom)
        {
            QString marked = sew->selectedText();
            m_pWidgetFindWord->setText(marked);
        }
    }

    m_pWidgetFindWord->setCursorToTextField();
    m_findTextExprAction->action()->setChecked(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuReplaceTextExpr()
{
    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew == NULL)
    {
        return;
    }

    int lineFrom = -1;
    int lineTo = -1;
    int indexFrom = -1;
    int indexTo = -1;
//    bool textSelected = false;
    bool multiLineSelection = false;
    QString defaultText = "";

    //check whether text has been marked
    sew->getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

    if (lineFrom >= 0)
    {
        //text has been marked
//        textSelected = true;
        multiLineSelection = lineTo > lineFrom;

        if (multiLineSelection == false)
        {
            defaultText = sew->selectedText();
        }
    }
    else
    {
        //nothing selected, get cursor position
        sew->getCursorPosition(&lineFrom, &indexFrom);
//        textSelected = false;
//        multiLineSelection = false;
        defaultText = sew->getWordAtPosition(lineFrom, indexFrom);
    }

    if (m_pDialogReplace == NULL)
    {
        m_pDialogReplace = new DialogReplace(getCanvas());
        m_pDialogReplace->setModal(false);
        connect(m_pDialogReplace, SIGNAL(findNext(QString, bool, bool, bool, bool, bool, bool)), this, SLOT(findTextExpr(QString, bool, bool, bool, bool, bool, bool)));
        connect(m_pDialogReplace, SIGNAL(replaceSelection(QString, QString)), this, SLOT(replaceTextExpr(QString, QString)));
        connect(m_pDialogReplace, SIGNAL(replaceAll(QString, QString, bool, bool, bool, bool)), this, SLOT(replaceAllExpr(QString, QString, bool, bool, bool, bool)));
    }

    m_pDialogReplace->setData(defaultText, (lineTo == -1) || (lineTo == lineFrom));
//    m_pDialogReplace->setData(defaultText, lineFrom, indexFrom, lineTo, indexTo);
    m_pDialogReplace->show();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuGoto()
{
    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew == NULL) return;

    bool lineNotChar;
    int curValue;
    int curLine;
    int curIndex;
    sew->getCursorPosition(&curLine,&curIndex);

    DialogGoto *d = new DialogGoto(sew->lineCount(), curLine + 1, sew->length(), sew->positionFromLineIndex(curLine, curIndex), getCanvas());

    if (d->exec())
    {
        d->getData(lineNotChar,curValue);

        if (lineNotChar)
        {
            curLine = curValue - 1;
            sew->setCursorPosAndEnsureVisible(curLine);
        }
        else
        {
            sew->lineIndexFromPosition(curValue, &curLine, &curIndex);
            sew->setCursorPosAndEnsureVisible(curLine);
            sew->setCursorPosition(curLine, curIndex);
        }

        sew->reportCurrentCursorAsGoBackNavigationItem("goto");
    }

    DELETE_AND_SET_NULL(d);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuToggleBookmark()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->toggleBookmark(-1);
        updateEditorActions();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuInsertCodec()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuScriptCharsetEncoding();
    }
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::mnuPyCodeFormatting()
{
    ScriptEditorWidget *sew = getCurrentEditor();

    if (sew != nullptr)
    {
        sew->menuPyCodeFormatting();
    }
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::mnuPyDocstringGenerator()
{
    ScriptEditorWidget *sew = getCurrentEditor();

    if (sew != nullptr)
    {
        sew->menuGenerateDocstring();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! this method is invoked if this ScriptDockWidget should be closed.
/*!
    First, tries to save every script. If this process is successfully executed, the close event is accepted in order to close this instance,
    else the event is ignored. If event is accepted, the signal removeAndDeleteScriptDockWidget is emitted, such that the ScriptEditorOrganizer can
    manage the deletion of this instance.

    \param event Event of type QCloseEvent
    \sa removeAndDeleteScriptDockWidget
*/
void ScriptDockWidget::closeEvent(QCloseEvent *event)
{
    RetVal retValue = closeAllScripts(true, true, false, false);

    if (retValue.containsError())
    {
        event->ignore();
    }
    else
    {
        event->accept();
        emit statusBarInformationChanged(this, "", -1, -1);
        emit (removeAndDeleteScriptDockWidget(this));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::findTextExpr(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward, bool isQuickSeach)
{
    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew != NULL)
    {
        if (!forward)
        {
            int lineFrom, indexFrom, lineTo, indexTo;
            sew->getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
            if (lineFrom != -1)
            {
                sew->setCursorPosition(lineFrom, indexFrom);
            }
        }

        bool success = sew->findFirst(expr, regExpr, caseSensitive, wholeWord, wrap, forward, -1, -1, true);

        if (isQuickSeach)
        {
            QMetaObject::invokeMethod(m_pWidgetFindWord, "setSuccessState", Q_ARG(bool,success));
        }
        else
        {
            if (!success)
            {
                QMessageBox::information(m_pDialogReplace, tr("Find And Replace"), tr("'%1' was not found").arg(expr));
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::replaceTextExpr(QString expr, QString replace)
{
    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew != nullptr)
    {
        sew->replace(replace);
        sew->updateSyntaxCheck();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::replaceAllExpr(QString expr, QString replace, bool regExpr, bool caseSensitive, bool wholeWord, bool findInSel)
{
    bool success = true;
    bool inRange = true;
    int count = 0;

    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew != nullptr)
    {
        int tempLineFrom, tempIndexFrom, tempLineTo, tempIndexTo;
        int lastLineFrom = -1;
        int lastIndexFrom = -1;
        int lastLineTo = -1;
        int lastIndexTo = -1;
        int lineFrom = -1;
        int indexFrom = -1;
        int lineTo = -1;
        int indexTo = -1;
        if (findInSel)
        {
            sew->getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
        }

        sew->beginUndoAction();
        sew->setCursorPosition(0, 0);
        success = sew->findFirst(expr, regExpr, caseSensitive, wholeWord, false, true, lineFrom, indexFrom, true);

        if (findInSel)
        {
            sew->getSelection(&tempLineFrom, &tempIndexFrom, &tempLineTo, &tempIndexTo);
            inRange = (lineTo > tempLineTo) || ((lineTo == tempLineTo) && (indexTo >= tempIndexTo));

            if (inRange)
            {
                lastLineFrom= tempLineFrom;
                lastIndexFrom = tempIndexFrom;
                lastLineTo = tempLineTo;
                lastIndexTo = tempIndexTo;
            }
        }

        while (success && inRange)
        {
            sew->replace(replace);
            success = sew->findNext();

            if (findInSel)
            {
                sew->getSelection(&tempLineFrom, &tempIndexFrom, &tempLineTo, &tempIndexTo);
                inRange = (lineTo > tempLineTo) || ((lineTo == tempLineTo) && (indexTo >= tempIndexTo));

                if (inRange)
                {
                    lastLineFrom= tempLineFrom;
                    lastIndexFrom = tempIndexFrom;
                    lastLineTo = tempLineTo;
                    lastIndexTo = tempIndexTo;
                }
            }

            count++;
        }
        sew->endUndoAction();

        if (!inRange && lastLineFrom > -1)
        {
            sew->setSelection(lastLineFrom, lastIndexFrom, lastLineTo, lastIndexTo);
        }

        sew->updateSyntaxCheck();
    }

    if (count == 1)
    {
        QMessageBox::information(m_pDialogReplace, tr("Find And Replace"), tr("One occurrence was replaced"));
    }
    else
    {
        QMessageBox::information(m_pDialogReplace, tr("Find And Replace"), tr("%1 occurrences were replaced").arg(count));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::insertIconBrowserText(QString iconLink)
{
    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew != NULL)
    {
        int line, index;
        sew->insertPlainText(iconLink);
        sew->getCursorPosition(&line, &index);
        sew->setCursorPosition(line, index + iconLink.length());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::editorMarginChanged()
{
    updateEditorActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::findWordWidgetFinished()
{
    m_pWidgetFindWord->hide();

    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->setFocus();
    }

    m_findTextExprActionSC->setEnabled(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::setCurrentIndex(int index)
{
    m_tab->setCurrentIndex(index);

    if (m_stackHistory.contains(index))
    {
        m_stackHistory.removeOne(index);
        m_stackHistory.prepend(index);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to copy the filename to clipboard
void ScriptDockWidget::mnuCopyFilename()
{
    ScriptEditorWidget *sew = getEditorByIndex(m_actTabIndex);
    if (sew != NULL)
    {
        QClipboard *clipboard = QApplication::clipboard();
        clipboard->setText(sew->getFilename(), QClipboard::Clipboard);
    }
}

//-------------------------------------------------------------------------------------
QList<OutlineSelectorWidget::EditorOutline> ScriptDockWidget::getAllOutlines(int &activeIndex) const
{
    QList<OutlineSelectorWidget::EditorOutline> outlines;
    activeIndex = m_actTabIndex;
    const ScriptEditorWidget *sew;

    for (int i = 0; i < m_tab->count(); ++i)
    {
        OutlineSelectorWidget::EditorOutline item;
        sew = getEditorByIndex(i);
        item.filename = sew->hasNoFilename() ? "" : sew->getFilename();
        item.editorUID = sew->getUID();
        item.rootOutline = sew->parseOutline();
        outlines << item;
    }

    return outlines;
}

//-------------------------------------------------------------------------------------
void ScriptDockWidget::mnuFindSymbolsShow()
{
    auto *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

    if (seo)
    {
        int currentIndex = -1; //index of the current tab in the returned outline list
        auto outlines = seo->getAllOutlines(this, currentIndex);

        if (currentIndex >= 0)
        {
            m_outlineSelectorWidget = QSharedPointer<OutlineSelectorWidget>(
                new OutlineSelectorWidget(
                    outlines,
                    currentIndex,
                    this,
                    getActiveInstance()));

            m_outlineSelectorWidget->show();
        }
    }
}

//-------------------------------------------------------------------------------------
/*
Tab navigation with "most recently used" behaviour.
It's fired when pressing the Ctrl+Tab shortcut.
*/
void ScriptDockWidget::tabChangedRequest()
{
    // pass getActiveInstance as parent, since this is either the
    // itom main window (docked mode) or the window of the abstractDockWidget
    // if undocked.
    m_tabSwitcherWidget = QSharedPointer<TabSwitcherWidget>(
        new TabSwitcherWidget(
            m_tab,
            m_stackHistory,
            this,
            getActiveInstance()));

    m_tabSwitcherWidget->show();
    m_tabSwitcherWidget->selectRow(1);
    m_tabSwitcherWidget->setFocus();
}

} //end namespace ito
