/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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
#include "../ui/dialogGoto.h"
#include "../ui/dialogReplace.h"

//#include <QPrinter>
//#include <QPrintDialog>
#include <qsignalmapper.h>

#include "../ui/dialogIconBrowser.h"
#include "../Qitom/AppManagement.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "../helper/IOHelper.h"

namespace ito {

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
    \return description
    \sa AbstractDockWidget::AbstractDockWidget
*/
ScriptDockWidget::ScriptDockWidget(const QString &title, const QString &objName, bool docked, bool isDockAvailable, QWidget *parent, Qt::WindowFlags /*flags*/) :
    AbstractDockWidget(docked, isDockAvailable, floatingWindow, movingEnabled, title, objName, parent),
    m_tab(NULL),
    m_pWidgetFindWord(NULL),
    m_pDialogReplace(NULL),
    m_actTabIndex(-1),
    m_tabContextMenu(NULL),
    m_winMenu(NULL)
{
    m_tab = new QTabWidgetItom(this);

    //!< tab-settings
    m_tab->setElideMode(Qt::ElideLeft);
    m_tab->setTabShape(QTabWidget::Rounded);
    m_tab->setTabsClosable(true);
    m_tab->setMovable(true);
    m_tab->setTabPosition(QTabWidget::South);

    // Signalmapper for dynamic lastFile Menu
    m_lastFilesMapper = new QSignalMapper(this);
    connect(m_lastFilesMapper, SIGNAL(mapped(const QString &)), this, SLOT(lastFileOpen(const QString &)));

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
    connect(m_pWidgetFindWord, SIGNAL(hideSearchBar()), this, SLOT(mnuFindTextExpr()));

    m_pVBox = new QVBoxLayout();
    m_pVBox->setContentsMargins(2,2,2,2);
    m_pVBox->addWidget(m_tab);
    m_pVBox->addWidget(m_pWidgetFindWord);

    m_pCenterWidget = new QWidget();
    m_pCenterWidget->setLayout(m_pVBox);

    setContentWidget(m_pCenterWidget);
    //setContentWidget(m_tab);

    setFocusPolicy(Qt::StrongFocus);
//    setAcceptDrops(true);
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
                ScriptEditorWidget* sew = new ScriptEditorWidget(m_tab);
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
//! creates new instance of ScriptEditorWidget and appends it to the tab-widget
/*!
    \return result of method appendEditor
    \sa appendEditor
*/
RetVal ScriptDockWidget::newScript()
{
    ScriptEditorWidget* sew = new ScriptEditorWidget(m_tab);
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

    fileName = QFileDialog::getOpenFileName(getActiveInstance(), tr("file open"), QDir::currentPath(), "Python (*.py)");

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
    QFile file(filename);

    if (!file.exists())
    {
        if (!silent)
        {
            QMessageBox msg(QMessageBox::Warning, tr("file not found"), tr("the file %1 could not be found").arg(filename));
            msg.exec();
        }
        return RetVal(retError);
    }
    else
    {
        QFileInfo info(file);
        if (info.suffix().toLower() != "py")
        {
            if (!silent)
            {
                QMessageBox msg(QMessageBox::Warning, tr("invalid file format"), tr("the file %1 is no python macro").arg(filename));
                msg.exec();
            }
            return RetVal(retError);
        }
    }

    ScriptEditorWidget* sew = new ScriptEditorWidget(m_tab);

    RetVal retValue = sew->openFile(filename, false);

    if (retValue.containsError())
    {
        DELETE_AND_SET_NULL(sew);
    }
    else
    {
        appendEditor(sew);
        sew->setFocus();
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
    \return retOk if all identified scripts could be saved, else retError
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
RetVal ScriptDockWidget::closeAllScripts(bool saveFirst, bool askFirst, bool ignoreNewScripts, int excludeIndex)
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
            retValue += closeTab(getIndexByEditor(*it), false);
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
    //!< activate appended tab
    m_tab->setCurrentIndex(m_tab->count() - 1);
    m_tab->setTabToolTip(m_tab->count() - 1, toolTip);
    m_tab->setTabText(m_tab->count() - 1, name);
    
    connect(editorWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
    connect(editorWidget, SIGNAL(copyAvailable(bool)), this, SLOT(updateEditorActions()));
    connect(editorWidget, SIGNAL(closeRequest(ScriptEditorWidget*, bool)), this, SLOT(tabCloseRequested(ScriptEditorWidget*, bool)));
    connect(editorWidget, SIGNAL(marginChanged()), this, SLOT(editorMarginChanged()));

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
    if (index < 0 || index >= m_tab->count()) return NULL;

    ScriptEditorWidget* removedWidget = static_cast<ScriptEditorWidget*>(m_tab->widget(index));
    m_tab->removeTab(index);
    disconnect(removedWidget, SIGNAL(modificationChanged(bool)), this, SLOT(scriptModificationChanged(bool)));
    disconnect(removedWidget, SIGNAL(copyAvailable(bool)), this, SLOT(updateEditorActions()));
    disconnect(removedWidget, SIGNAL(closeRequest(ScriptEditorWidget*, bool)), this, SLOT(tabCloseRequested(ScriptEditorWidget*, bool)));
    disconnect(removedWidget, SIGNAL(marginChanged()), this, SLOT(editorMarginChanged()));

    updateEditorActions();
    updatePythonActions();

    if (index>0)
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
            newScripts = newScripts | editorWidget->hasNoFilename();
        }
    }

    return newScripts;
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

    if (index >= 0)
    {
        ScriptEditorWidget *editorWidget = static_cast<ScriptEditorWidget*>(m_tab->widget(m_actTabIndex));
        setWindowModified(editorWidget->isModified());

        if (editorWidget->hasNoFilename())
        {
            setAdvancedWindowTitle(editorWidget->getUntitledName().prepend(" - ").append("[*]"), true);

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
            setAdvancedWindowTitle(editorWidget->getFilename().prepend(" - ").append("[*]"), true);

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
//! slot connected to each ScriptEditorWidget instance. Invoked if any content in any script changed.
/*!
    calls slot currentTabChanged with active tab index as parameter.

    \sa currentTabChanged
*/
void ScriptDockWidget::scriptModificationChanged(bool /*changed*/)
{
    currentTabChanged(m_actTabIndex);
    updateEditorActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if close button of any tab of m_tab (QTabWidgetItom) has been pressed
/*!
    tries to close the tab in question

    \param index tab-index of tab in question
*/
void ScriptDockWidget::tabCloseRequested(int index)
{
    ScriptEditorWidget *sew = getEditorByIndex(index);
    if (sew == NULL) return;

    RetVal retValue = closeTab(index, true);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::tabCloseRequested(ScriptEditorWidget* sew, bool ignoreModifications)
{
    if (sew == NULL) return;
    int index = getIndexByEditor(sew);

    RetVal retValue = closeTab(index, ignoreModifications);
    // TODO: What to do with retValue???
}

//----------------------------------------------------------------------------------------------------------------------------------
//! public method to close any specific tab with or without saving its script first
/*!
    \param index tab-index of tab in question
    \param saveFirst save changes in editor first (true) or ignore changes (false)
    \return retOk if tab has been saved (or not) and closed, retError if saving failed
    \sa saveTab, removeEditor
*/
RetVal ScriptDockWidget::closeTab(int index, bool saveFirst)
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
        //delete sew;
        sew = NULL;
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
int ScriptDockWidget::getIndexByEditor(ScriptEditorWidget* sew) const
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
    \return reference to current ScriptEditorWidget or NULL
*/
ScriptEditorWidget* ScriptDockWidget::getCurrentEditor() const
{
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

        if (tabRectangle.contains(event->pos()-m_tab->pos()-m_tab->getTabBar()->pos()))
        {
            m_tab->setCurrentIndex(i);
            m_actTabIndex = i;
        }
    }

    m_tabContextMenu->exec(event->globalPos());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates actions which deal with editor commands
void ScriptDockWidget::updateEditorActions()
{
    m_saveAllScriptsAction->setEnabled(false);
    for (int i = 0; i < m_tab->count(); i++)
    {
        if (static_cast<ScriptEditorWidget *>(m_tab->widget(i))->isModified())
        {
            m_saveAllScriptsAction->setEnabled(true);
        }
    }

    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    const ScriptEditorWidget *sew = NULL;
    if (m_actTabIndex > -1)
    {
        sew = static_cast<ScriptEditorWidget *>(m_tab->widget(m_actTabIndex));
    }

    m_saveScriptAction->setEnabled(m_tab->count()>0 && sew != NULL && sew->isModified());
    m_saveScriptAsAction->setEnabled(m_tab->count()>0 && sew != NULL);

    m_cutAction->setEnabled(sew != NULL && sew->getCanCopy());
    m_copyAction->setEnabled(sew != NULL && sew->getCanCopy());
    m_pasteAction->setEnabled(true, true); //!< todo
    m_undoAction->setEnabled(sew != NULL && sew->isUndoAvailable());
    m_redoAction->setEnabled(sew != NULL && sew->isRedoAvailable());
    m_commentAction->setEnabled(m_tab->count()>0 && sew != NULL);
    m_uncommentAction->setEnabled(m_tab->count()>0 && sew != NULL);
    m_indentAction->setEnabled(m_tab->count()>0 && sew != NULL);
    m_unindentAction->setEnabled(m_tab->count()>0 && sew != NULL);

    m_tabCloseAction->setEnabled(m_actTabIndex > -1);
    m_tabCloseAllAction->setEnabled(m_actTabIndex > -1);
    m_findTextExprAction->setEnabled(m_actTabIndex > -1);
    m_replaceTextExprAction->setEnabled(m_actTabIndex > -1);
    m_gotoAction->setEnabled(m_actTabIndex > -1);
    m_openIconBrowser->setEnabled(m_actTabIndex > -1);
    m_bookmarkToggle->setEnabled(sew != NULL);
    m_bookmarkNext->setEnabled(sew != NULL && sew->isBookmarked());
    m_bookmarkPrevious->setEnabled(sew != NULL && sew->isBookmarked());
    m_bookmarkClearAll->setEnabled(sew != NULL && sew->isBookmarked());

    m_scriptRunSelectionAction->setEnabled(sew != NULL && sew->getCanCopy() && (!pyEngine->isPythonBusy() || pyEngine->isPythonDebuggingAndWaiting()));

//    QMetaObject::invokeMethod(m_pWidgetFindWord,"setFindBarEnabled",Q_ARG(bool,m_actTabIndex > -1));
    if (m_pWidgetFindWord != NULL)
    {
        m_pWidgetFindWord->setFindBarEnabled(m_actTabIndex > -1);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! updates actions which deal with python commands
void ScriptDockWidget::updatePythonActions()
{
    int lineFrom = -1;
    int lineTo = -1;
    int indexFrom = -1;
    int indexTo = -1;

    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew)
    {
        sew->getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
    }

    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    bool busy1 = pythonBusy();
    bool busy2 = busy1 && pythonDebugMode() && pythonInWaitingMode();

    m_scriptRunAction->setEnabled(!busy1);
    m_scriptRunSelectionAction->setEnabled(lineFrom != -1 && (!pyEngine->isPythonBusy() || pyEngine->isPythonDebuggingAndWaiting()));
    m_scriptDebugAction->setEnabled(!busy1);
    m_scriptStopAction->setEnabled(busy1);
    m_scriptContinueAction->setEnabled(busy2);
    m_scriptStepAction->setEnabled(busy2);
    m_scriptStepOverAction->setEnabled(busy2);
    m_scriptStepOutAction->setEnabled(busy2);
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
}

//----------------------------------------------------------------------------------------------------------------------------------
//! creates actions
void ScriptDockWidget::createActions()
{
    m_tabMoveLeftAction = new ShortcutAction(QIcon(":/arrows/icons/1leftarrow.png"), tr("move left"), this);
    m_tabMoveLeftAction->connectTrigger(this, SLOT(mnuTabMoveLeft()));

    m_tabMoveRightAction = new ShortcutAction(QIcon(":/arrows/icons/1rightarrow.png"), tr("move right"), this);
    m_tabMoveRightAction->connectTrigger(this, SLOT(mnuTabMoveRight()));

    m_tabMoveFirstAction = new ShortcutAction(QIcon(":/arrows/icons/2leftarrow.png"), tr("move first"), this);
    m_tabMoveFirstAction->connectTrigger(this, SLOT(mnuTabMoveFirst()));

    m_tabMoveLastAction = new ShortcutAction(QIcon(":/arrows/icons/2rightarrow.png"), tr("move last"), this);
    m_tabMoveLastAction->connectTrigger(this, SLOT(mnuTabMoveLast()));

    m_tabCloseAction = new ShortcutAction(QIcon(":/files/icons/close.png"), tr("close"), this, QKeySequence::Close, Qt::WidgetWithChildrenShortcut);
    m_tabCloseAction->connectTrigger(this, SLOT(mnuTabClose()));

    m_tabCloseOthersAction = new ShortcutAction(QIcon(), tr("close others"), this);
    m_tabCloseOthersAction->connectTrigger(this, SLOT(mnuTabCloseOthers()));

    m_tabCloseAllAction = new ShortcutAction(QIcon(), tr("close all"), this);
    m_tabCloseAllAction->connectTrigger(this, SLOT(mnuTabCloseAll()));

    m_tabDockAction = new ShortcutAction(QIcon(), tr("dock"), this);
    m_tabDockAction->connectTrigger(this, SLOT(mnuTabDock()));

    m_tabUndockAction = new ShortcutAction(QIcon(), tr("undock"), this);
    m_tabUndockAction->connectTrigger(this, SLOT(mnuTabUndock()));

    m_newScriptAction = new ShortcutAction(QIcon(":/files/icons/new.png"), tr("new"), this, QKeySequence::New, Qt::WidgetWithChildrenShortcut);
    m_newScriptAction->connectTrigger(this, SLOT(mnuNewScript()));

    m_openScriptAction = new ShortcutAction(QIcon(":/files/icons/open.png"), tr("open"), this, QKeySequence::Open, Qt::WidgetWithChildrenShortcut);
    m_openScriptAction->connectTrigger(this, SLOT(mnuOpenScript()));

    m_saveScriptAction = new ShortcutAction(QIcon(":/files/icons/fileSave.png"), tr("save"), this, QKeySequence::Save, Qt::WidgetWithChildrenShortcut);
    m_saveScriptAction->connectTrigger(this, SLOT(mnuSaveScript()));

    m_saveScriptAsAction = new ShortcutAction(QIcon(":/files/icons/fileSaveAs.png"), tr("save as..."), this, QKeySequence::SaveAs, Qt::WidgetWithChildrenShortcut);
    m_saveScriptAsAction->connectTrigger(this, SLOT(mnuSaveScriptAs()));

    m_saveAllScriptsAction = new ShortcutAction(QIcon(":/files/icons/fileSaveAll.png"), tr("save all"), this);
    m_saveAllScriptsAction->connectTrigger(this, SLOT(mnuSaveAllScripts()));

    m_printAction = new ShortcutAction(QIcon(":/plots/icons/print.png"), tr("print..."), this, QKeySequence::Print, Qt::WidgetWithChildrenShortcut);
    m_printAction->connectTrigger(this, SLOT(mnuPrint()));

    m_cutAction = new ShortcutAction(QIcon(":/editor/icons/editCut.png"), tr("cut"), this, QKeySequence::Cut, Qt::WidgetWithChildrenShortcut);
    m_cutAction->connectTrigger(this, SLOT(mnuCut()));

    m_copyAction = new ShortcutAction(QIcon(":/editor/icons/editCopy.png"), tr("copy"), this, QKeySequence::Copy, Qt::WidgetWithChildrenShortcut);
    m_copyAction->connectTrigger(this, SLOT(mnuCopy()));

    m_pasteAction = new ShortcutAction(QIcon(":/editor/icons/editPaste.png"), tr("paste"), this, QKeySequence::Paste, Qt::WidgetWithChildrenShortcut);
    m_pasteAction->connectTrigger(this, SLOT(mnuPaste()));

    m_undoAction = new ShortcutAction(QIcon(":/editor/icons/editUndo.png"), tr("undo"), this, QKeySequence::Undo, Qt::WidgetWithChildrenShortcut);
    m_undoAction->connectTrigger(this, SLOT(mnuUndo()));

    m_redoAction = new ShortcutAction(QIcon(":/editor/icons/editRedo.png"), tr("redo"), this, QKeySequence::Redo, Qt::WidgetWithChildrenShortcut);
    m_redoAction->connectTrigger(this, SLOT(mnuRedo()));

    m_commentAction = new ShortcutAction(QIcon(":/editor/icons/editComment.png"), tr("comment"), this, QKeySequence(tr("Ctrl+R", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_commentAction->connectTrigger(this, SLOT(mnuComment()));

    m_uncommentAction = new ShortcutAction(QIcon(":/editor/icons/editUncomment.png"), tr("uncomment"), this, QKeySequence(tr("Ctrl+Shift+R", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_uncommentAction->connectTrigger(this, SLOT(mnuUncomment()));

    m_indentAction = new ShortcutAction(QIcon(":/editor/icons/editIndent.png"), tr("indent"), this);
    m_indentAction->connectTrigger(this, SLOT(mnuIndent()));

    m_unindentAction = new ShortcutAction(QIcon(":/editor/icons/editUnindent.png"), tr("unindent"), this);
    m_unindentAction->connectTrigger(this, SLOT(mnuUnindent()));

    m_scriptRunAction = new ShortcutAction(QIcon(":/script/icons/runScript.png"), tr("run"), this, QKeySequence(tr("F5", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptRunAction->connectTrigger(this, SLOT(mnuScriptRun()));

    m_scriptRunSelectionAction = new ShortcutAction(QIcon(":/script/icons/runScript.png"), tr("run selection"), this);
    m_scriptRunSelectionAction->connectTrigger(this, SLOT(mnuScriptRunSelection()));

    m_scriptDebugAction = new ShortcutAction(QIcon(":/script/icons/debugScript.png"), tr("debug"), this, QKeySequence(tr("F6", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptDebugAction->connectTrigger(this, SLOT(mnuScriptDebug()));

    m_scriptStopAction = new ShortcutAction(QIcon(":/script/icons/stopScript.png"), tr("stop"), this, QKeySequence(tr("Shift+F10", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptStopAction->connectTrigger(this, SLOT(mnuScriptStop()));

    m_scriptContinueAction = new ShortcutAction(QIcon(":/script/icons/continue.png"), tr("continue"), this, QKeySequence(tr("F6", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptContinueAction->connectTrigger(this, SLOT(mnuScriptContinue()));

    m_scriptStepAction = new ShortcutAction(QIcon(":/script/icons/step.png"), tr("step"), this, QKeySequence(tr("F11", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptStepAction->connectTrigger(this, SLOT(mnuScriptStep()));

    m_scriptStepOverAction = new ShortcutAction(QIcon(":/script/icons/stepOver.png"), tr("step over"), this, QKeySequence(tr("F10", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptStepOverAction->connectTrigger(this, SLOT(mnuScriptStepOver()));

    m_scriptStepOutAction = new ShortcutAction(QIcon(":/script/icons/stepOut.png"), tr("step out"), this, QKeySequence(tr("Shift+F11", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_scriptStepOutAction->connectTrigger(this, SLOT(mnuScriptStepOut()));

    m_findTextExprAction = new ShortcutAction(QIcon(":/editor/icons/find.png"), tr("quick search..."), this, QKeySequence::Find, Qt::WidgetWithChildrenShortcut);
    m_findTextExprAction->connectTrigger(this, SLOT(mnuFindTextExpr()));
    m_findTextExprAction->action()->setCheckable(true);

    m_replaceTextExprAction = new ShortcutAction(QIcon(":/editor/icons/editReplace.png"), tr("find and replace..."), this, QKeySequence(tr("Ctrl+H", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_replaceTextExprAction->connectTrigger(this, SLOT(mnuReplaceTextExpr()));

    m_openIconBrowser = new ShortcutAction(QIcon(":/editor/icons/iconList.png"), tr("icon &browser..."),this, QKeySequence(tr("Ctrl+B", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_openIconBrowser->connectTrigger(this, SLOT(mnuOpenIconBrowser()));

    m_gotoAction = new ShortcutAction(QIcon(), tr("goto..."), this, QKeySequence(tr("Ctrl+G", "QShortcut")), Qt::WidgetWithChildrenShortcut);
    m_gotoAction->connectTrigger(this, SLOT(mnuGoto()));

    m_bookmarkToggle = new ShortcutAction(QIcon(":/bookmark/icons/bookmarkToggle.png"), tr("&toggle bookmark"), this);
    m_bookmarkToggle->connectTrigger(this, SLOT(mnuToggleBookmark()));

    m_bookmarkNext = new ShortcutAction(QIcon(":/bookmark/icons/bookmarkNext.png"), tr("&next bookmark"), this);
    m_bookmarkNext->connectTrigger(this, SLOT(mnuGotoNextBookmark()));

    m_bookmarkPrevious = new ShortcutAction(QIcon(":/bookmark/icons/bookmarkPrevious.png"), tr("&previous bookmark"), this);
    m_bookmarkPrevious->connectTrigger(this, SLOT(mnuGotoPreviousBookmark()));

    m_bookmarkClearAll = new ShortcutAction(QIcon(":/bookmark/icons/bookmarkClearAll.png"), tr("&clear all bookmarks"), this);
    m_bookmarkClearAll->connectTrigger(this, SLOT(mnuClearAllBookmarks()));

    updatePythonActions();
    updateTabContextActions();
    updateEditorActions();
}

//----------------------------------------------------------------------------------------------------------------------------------
/*Slot aboutToOpen*/
void ScriptDockWidget::menuLastFilesAboutToShow()
{
    // Delete old actions
    for(int i = 0; i < m_lastFilesMenu->actions().length(); ++i)
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
            // Create new menus
            foreach (const QString &path, sEO->m_lastUsedFiles) 
            {
                QString displayedPath = path;
                IOHelper::shortenFilepathInMiddle(displayedPath, 200);
                m_lastFileAct = new ShortcutAction(QIcon(":/files/icons/filePython.png"), displayedPath, this);
                m_lastFilesMenu->addAction(m_lastFileAct->action());
                m_lastFileAct->connectTrigger(m_lastFilesMapper, SLOT(map()));
                m_lastFilesMapper->setMapping(m_lastFileAct->action(), path);
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
//void ScriptDockWidget::windowStateChanged(bool windowNotToolbox)
//{
//    //Qt::ShortcutContext context = Qt::WidgetWithChildrenShortcut;
//    //if (windowNotToolbox)
//    //{
//    //    context = Qt::WindowShortcut;
//    //}
//
//    //m_findTextExprAction->setShortcutContext(context);
//    //m_gotoAction->setShortcutContext(context);
//    //m_openIconBrowser->setShortcutContext(context);
//    //m_uncommentAction->setShortcutContext(context);
//    //m_commentAction->setShortcutContext(context);
//    //m_redoAction->setShortcutContext(context);
//    //m_undoAction->setShortcutContext(context);
//    //m_pasteAction->setShortcutContext(context);
//    //m_copyAction->setShortcutContext(context);
//    //m_cutAction->setShortcutContext(context);
//    ////m_saveScriptAction->setShortcutContext(context);
//    ////m_saveScriptAsAction->setShortcutContext(context);
//    //m_newScriptAction->setShortcutContext(context);
//    //m_tabCloseAction->setShortcutContext(context);
//
//}

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
    m_lastFilesMenu = m_fileMenu->addMenu(QIcon(":/files/icons/filePython.png"), tr("Open last files"));
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
    m_editMenu->addSeparator();
    m_editMenu->addAction(m_findTextExprAction->action());
    m_editMenu->addAction(m_replaceTextExprAction->action());
    m_editMenu->addAction(m_gotoAction->action());
    m_editMenu->addAction(m_openIconBrowser->action());
    m_editMenu->addSeparator();
    m_bookmark = m_editMenu->addMenu(QIcon(":/bookmark/icons/bookmark.png"), tr("bookmark"));
    m_bookmark->addAction(m_bookmarkToggle->action());
    m_bookmark->addAction(m_bookmarkNext->action());
    m_bookmark->addAction(m_bookmarkPrevious->action());
    m_bookmark->addAction(m_bookmarkClearAll->action());

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
    m_fileToolBar = new QToolBar(tr("file toolbar"), this);
    addToolBar(m_fileToolBar, "fileToolBar");
    m_fileToolBar->addAction(m_newScriptAction->action());
    m_fileToolBar->addAction(m_openScriptAction->action());
    m_fileToolBar->addAction(m_saveScriptAction->action());
    m_fileToolBar->addAction(m_saveScriptAsAction->action());
    m_fileToolBar->addAction(m_saveAllScriptsAction->action());
    m_fileToolBar->setFloatable(false);

    m_editToolBar = new QToolBar(tr("edit toolbar"), this);
    addToolBar(m_editToolBar, "editToolBar");
    m_editToolBar->addAction(m_cutAction->action());
    m_editToolBar->addAction(m_copyAction->action());
    m_editToolBar->addAction(m_pasteAction->action());
    m_editToolBar->addAction(m_undoAction->action());
    m_editToolBar->addAction(m_redoAction->action());
    m_editToolBar->addAction(m_findTextExprAction->action());
    m_editToolBar->addAction(m_replaceTextExprAction->action());
    m_editToolBar->setFloatable(false);

    m_scriptToolBar = new QToolBar(tr("script toolbar"), this);
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

    m_bookmarkToolBar = new QToolBar(tr("bookmark toolbar"), this);
    addToolBar(m_bookmarkToolBar, "bookmarkToolBar");
    m_bookmarkToolBar->addAction(m_bookmarkToggle->action());
    m_bookmarkToolBar->addAction(m_bookmarkNext->action());
    m_bookmarkToolBar->addAction(m_bookmarkPrevious->action());
    m_bookmarkToolBar->addAction(m_bookmarkClearAll->action());
    m_bookmarkToolBar->setFloatable(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! init status bar \todo right now, this is an empty method
void ScriptDockWidget::createStatusBar()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
//! activates tab with script whose filename corresponds to the filename parameter.
/*!
    \param filename Filename of the script which should be activated
    \param line is the marked debugging line (default: -1, no arrow)
    \return true if filename has been found and activated, else false.
*/
bool ScriptDockWidget::activateTabByFilename(QString filename, int line /* = -1*/)
{
    ScriptEditorWidget *sew = NULL;
    QString temp, temp2;
    QFileInfo finfo1(filename);
    filename = finfo1.canonicalFilePath().toLower();
    QFileInfo finfo2;

    for (int i = 0; i < m_tab->count(); i++)
    {
        sew = static_cast<ScriptEditorWidget *>(m_tab->widget(i));

        if (!sew->hasNoFilename())
        {
            finfo2.setFile(sew->getFilename());
            temp = finfo1.canonicalFilePath().toLower();
                temp2 = finfo2.canonicalFilePath().toLower();
            if (filename == finfo2.canonicalFilePath().toLower())
            {
                m_tab->setCurrentIndex(i);
                raiseAndActivate();

                if (line >= 0)
                {
                    sew->pythonDebugPositionChanged(filename, line);
                }

                return true;
            }
        }
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool ScriptDockWidget::activeTabEnsureLineVisible(int lineNr)
{
    if (m_actTabIndex >= 0)
    {
        ScriptEditorWidget *sew = static_cast<ScriptEditorWidget *>(m_tab->widget(m_actTabIndex));
        if (sew)
        {
            sew->setCursorPosAndEnsureVisible(lineNr);
            return true;
        }
    }
    return false;
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
    DialogIconBrowser *m_iconBrowser = new DialogIconBrowser(getCanvas());
    connect(m_iconBrowser, SIGNAL(sendIconBrowserText(QString)), this, SLOT(insertIconBrowserText(QString)));
    if (m_iconBrowser->exec())
    {
        
    }
    delete m_iconBrowser;
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
    emit (undockScriptTab(this, m_actTabIndex, undockToNewScriptWindow, !docked()));
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
        sew->saveAsFile(false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by action to save all opened scripts
void ScriptDockWidget::mnuSaveAllScripts()
{
    ScriptEditorWidget *sew;
    RetVal retValue(retOk);
    for (int i = 0; i < m_tab->count() && !retValue.containsError(); i++)
    {
        sew = getEditorByIndex(m_actTabIndex);
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

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a paste command in active script editor
void ScriptDockWidget::mnuPaste()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuPaste();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute an undo command in active script editor
void ScriptDockWidget::mnuUndo()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    sew->undo();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked to execute a redo command in active script editor
void ScriptDockWidget::mnuRedo()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    sew->redo();
}

//----------------------------------------------------------------------------------------------------------------------------------
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
    if (pythonDebugMode() && pythonInWaitingMode())
    {
        emit (pythonDebugCommand(ito::pyDbgQuit));
        //activateWindow(); (if you uncomment this line,the script window will always dissappear in the background - that's a little bit crazy, therefore don't activate it here)
    }
    else if (PythonEngine::getInstance())
    {
        PythonEngine::getInstance()->pythonInterruptExecution();
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
    if (!m_pWidgetFindWord->isVisible()) 
    {
        m_pWidgetFindWord->show();
        m_pWidgetFindWord->setCursorToTextField();
        m_findTextExprAction->action()->setChecked(true);
    }
    else
    {
        m_pWidgetFindWord->hide();
        m_findTextExprAction->action()->setChecked(false);
    }
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
        connect(m_pDialogReplace, SIGNAL(findNext(QString,bool,bool,bool,bool,bool,bool)), this, SLOT(findTextExpr(QString,bool,bool,bool,bool,bool,bool)));
        connect(m_pDialogReplace, SIGNAL(replaceSelection(QString,QString)), this, SLOT(replaceTextExpr(QString,QString)));
        connect(m_pDialogReplace, SIGNAL(replaceAll(QString,QString,bool,bool,bool)), this, SLOT(replaceAllExpr(QString,QString,bool,bool,bool)));
    }

    m_pDialogReplace->setData(defaultText, lineFrom, indexFrom, lineTo, indexTo);
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

    DialogGoto *d = new DialogGoto(sew->lines(), curLine + 1, sew->length(), sew->positionFromLineIndex(curLine, curIndex), getCanvas());
    
    if (d->exec())
    {
        d->getData(lineNotChar,curValue);
        if (lineNotChar)
        {
            sew->setCursorPosAndEnsureVisible(curValue - 1);
        }
        else
        {
            sew->lineIndexFromPosition(curValue, &curLine, &curIndex);
            sew->setCursorPosAndEnsureVisible(curLine);
            sew->setCursorPosition(curLine, curIndex);
        }
    }
    DELETE_AND_SET_NULL(d);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuToggleBookmark()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuToggleBookmark();
        updateEditorActions();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuClearAllBookmarks()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuClearAllBookmarks();
        updateEditorActions();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuGotoNextBookmark()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuGotoNextBookmark();
        updateEditorActions();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::mnuGotoPreviousBookmark()
{
    ScriptEditorWidget *sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->menuGotoPreviousBookmark();
        updateEditorActions();
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
    RetVal retValue = closeAllScripts(true, true, false);

    if (retValue.containsError())
    {
        event->ignore();
    }
    else
    {
        event->accept();
        emit (removeAndDeleteScriptDockWidget(this));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::findTextExpr(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward, bool isQuickSeach)
{
    ScriptEditorWidget* sew = getCurrentEditor();

    if (sew != NULL)
    {
        int lineFrom, indexFrom, lineTo, indexTo;
        sew->getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
        if (lineFrom != -1 && forward) sew->setCursorPosition(lineTo, indexTo);
        if (lineFrom != -1 && !forward) sew->setCursorPosition(lineFrom, indexFrom);
        bool success = sew->findFirst(expr, regExpr, caseSensitive, wholeWord, wrap, forward, -1, -1, true);

        if (isQuickSeach)
        {
            QMetaObject::invokeMethod(m_pWidgetFindWord, "setSuccessState", Q_ARG(bool,success));
        }
        else
        {
            if (!success)
            {
                QMessageBox::information(m_pDialogReplace, tr("find and replace"), tr("'%1' was not found").arg(expr));
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::replaceTextExpr(QString expr, QString replace)
{
    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->replace(replace);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::replaceAllExpr(QString expr, QString replace, bool regExpr, bool caseSensitive, bool wholeWord)
{
    bool success = true;
    int count = 0;

    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew != NULL)
    {
        sew->beginUndoAction();
        sew->setCursorPosition(0, 0);
        success = sew->findFirst(expr, regExpr, caseSensitive, wholeWord, false, true, -1, -1, true);

        while (success)
        {
            sew->replace(replace);
            success = sew->findNext();
            count++;
        }
        sew->endUndoAction();
    }

    QMessageBox::information(m_pDialogReplace, tr("find and replace"), tr("%1 occurrence(s) was replaced").arg(count));
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::insertIconBrowserText(QString iconLink)
{
    ScriptEditorWidget* sew = getCurrentEditor();
    if (sew != NULL)
    {
        int line, index;
        sew->insert(iconLink);
        sew->getCursorPosition(&line, &index);
        sew->setCursorPosition(line, index + iconLink.length());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptDockWidget::editorMarginChanged()
{
    updateEditorActions();
}

} //end namespace ito
