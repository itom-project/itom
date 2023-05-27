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

#ifndef SCRIPTDOCKWIDGET_H
#define SCRIPTDOCKWIDGET_H

#include "abstractDockWidget.h"
#include "itomQWidgets.h"
#include "scriptEditorWidget.h"
#include "tabSwitcherWidget.h"
#include "outlineSelectorWidget.h"

#include <qaction.h>
#include <qstring.h>
#include <qtoolbar.h>
#include <qcombobox.h>
#include <qpointer.h>
#include <qsharedpointer.h>

#include "../models/outlineItem.h"
#include "../models/bookmarkModel.h"

#include <qevent.h>

#include "../ui/widgetFindWord.h"

namespace ito {

class DialogReplace; //forward declaration

//! this struct can hold common actions for all script editor and script dock widgets
struct ScriptEditorActions
{
    QAction *actNavigationForward;
    QAction *actNavigationBackward;
};


class ScriptDockWidget : public AbstractDockWidget
{
    Q_OBJECT
public:
    ScriptDockWidget(const QString &title, const QString &objName, bool docked, bool isDockAvailable,
        const ScriptEditorActions &commonActions, BookmarkModel *bookmarkModel,
        QWidget* parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());
    ~ScriptDockWidget();

    QStringList getModifiedFileNames(bool ignoreNewScripts = false, int excludeIndex = -1) const;
    QStringList getAllFilenames() const;

    RetVal newScript();
    RetVal openScript();
    RetVal openScript(QString filename, bool silent = false);
    RetVal saveAllScripts(bool askFirst = true, bool ignoreNewScripts = false, int excludeIndex = -1);
    RetVal closeAllScripts(bool saveFirst = true, bool askFirst = true, bool ignoreNewScripts = false, \
                            int excludeIndex = -1, bool closeScriptWidgetIfLastTabClosed = true);

    inline bool isTabIndexValid(int tabIndex) const { return (tabIndex >= 0 && tabIndex < m_tab->count()); }   /*!<  checks wether given tab-index is valid (true) or not (false) */
    inline int getTabCount() const { return m_tab->count(); }      /*!<  returns number of tabs */
    inline int getCurrentIndex() const { return m_tab->currentIndex(); }
    void setCurrentIndex(int index);

    bool containsNewScripts() const;

    RetVal appendEditor(ScriptEditorWidget* editorWidget);         /*!<  appends widget, without creating it (for drag&drop, (un)-docking...) */
    ScriptEditorWidget* removeEditor(int index);                    /*!<  removes widget, without deleting it (for drag&drop, (un)-docking...) */
    bool activateTabByFilename(const QString &filename, int currentDebugLine = -1, int UID = -1);
    bool activeTabEnsureLineVisible(const int lineNr, bool errorMessageClick = false, bool showSelectedCallstackLine = false);
    void activeTabShowLineAndHighlightWord(const int line, const QString &highlightedText, Qt::CaseSensitivity caseSensitivity = Qt::CaseInsensitive);
    const QTabWidget* tabWidget() const { return m_tab;  }

    QList<ito::ScriptEditorStorage> saveScriptState() const;
    RetVal restoreScriptState(const QList<ito::ScriptEditorStorage> &states);

    //!< return all outlines in the tabs of this script dock widget
    QList<OutlineSelectorWidget::EditorOutline> getAllOutlines(int &activeIndex) const;

protected:
    ScriptEditorWidget* getEditorByIndex(int index) const;
    ScriptEditorWidget* getCurrentEditor() const;

    int getIndexByEditor(const ScriptEditorWidget* sew) const;
    void tabFilenameOrModificationChanged(int index);

    void createActions();
    //void deleteActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void closeEvent(QCloseEvent *event);
    virtual void windowStateChanged(bool windowNotToolbox);

    RetVal closeTab(int index, bool saveFirst = true, bool closeScriptWidgetIfLastTabClosed = true);
    RetVal saveTab(int index, bool forceSaveAs = false, bool askFirst = true);

private:
    QWidget *m_pCenterWidget;
    QVBoxLayout *m_pVBox;
    QTabWidgetItom* m_tab;              /*!<  reference to QTabWidgetItom instance */
    WidgetFindWord *m_pWidgetFindWord;
    DialogReplace *m_pDialogReplace;
    BookmarkModel* m_pBookmarkModel; //! borrowed reference to the bookmark model. This model is owned by the script editor organizer.
    QLabel* m_pStatusBarWidget;

    static const char* statusBarStatePropertyName;

    int m_actTabIndex;                  /*!<  member indicating the tab-index of the active script editor */

    //!< indices of the tabs. The most recently activated tab index is at the front.
    /*
    This list is used to initialize the tabSwitcherWidget if Ctrl+Tab is pressed.
    */
    QList<int> m_stackHistory;

    QSharedPointer<TabSwitcherWidget> m_tabSwitcherWidget;
    QSharedPointer<OutlineSelectorWidget> m_outlineSelectorWidget;

    // ACTIONS
    ShortcutAction *m_tabMoveLeftAction;
    ShortcutAction *m_tabMoveRightAction;
    ShortcutAction *m_tabMoveFirstAction;
    ShortcutAction *m_tabMoveLastAction;
    ShortcutAction *m_tabCloseAction;
    ShortcutAction *m_tabCloseOthersAction;
    ShortcutAction *m_tabCloseAllAction;
    ShortcutAction *m_tabDockAction;
    ShortcutAction *m_tabUndockAction;
    ShortcutAction *m_newScriptAction;
    ShortcutAction *m_openScriptAction;
    ShortcutAction *m_saveScriptAction;
    ShortcutAction *m_saveScriptAsAction;
    ShortcutAction *m_saveAllScriptsAction;
    ShortcutAction *m_printAction;
    ShortcutAction *m_cutAction;
    ShortcutAction *m_copyAction;
    ShortcutAction *m_pasteAction;
    ShortcutAction *m_undoAction;
    ShortcutAction *m_redoAction;
    ShortcutAction *m_commentAction;
    ShortcutAction *m_uncommentAction;
    ShortcutAction *m_indentAction;
    ShortcutAction *m_unindentAction;
    ShortcutAction *m_autoCodeFormatAction;
    ShortcutAction *m_pyDocstringGeneratorAction;
    ShortcutAction *m_scriptRunAction;
    ShortcutAction *m_scriptRunSelectionAction;
    ShortcutAction *m_scriptDebugAction;
    ShortcutAction *m_scriptStopAction;
    ShortcutAction *m_scriptContinueAction;
    ShortcutAction *m_scriptStepAction;
    ShortcutAction *m_scriptStepOverAction;
    ShortcutAction *m_scriptStepOutAction;
    ShortcutAction *m_findTextExprAction;
    ShortcutAction *m_findTextExprActionSC;
    ShortcutAction *m_replaceTextExprAction;
    ShortcutAction *m_gotoAction;
    ShortcutAction *m_openIconBrowser;
    ShortcutAction *m_bookmarkToggle;

    ShortcutAction *m_insertCodecAct;
    ShortcutAction *m_copyFilename;
    ShortcutAction *m_findSymbols;

    ScriptEditorActions m_commonActions;

    QMenu *m_tabContextMenu;
    QMenu *m_fileMenu;
    QMenu *m_lastFilesMenu;
    QMenu *m_viewMenu;
    QMenu *m_editMenu;
    QMenu *m_scriptMenu;
    QMenu *m_winMenu;
    QMenu *m_bookmark;

    QToolBar* m_fileToolBar;
    QToolBar* m_editToolBar;
    QToolBar* m_scriptToolBar;
    QToolBar* m_bookmarkToolBar;

    QString m_autoCodeFormatCmd;

    // ClassNavigator
    QWidget *m_classMenuBar;
    QComboBox *m_classBox;
    QComboBox *m_methodBox;
    bool m_outlineShowNavigation;
    void fillNavigationClassComboBox(const QSharedPointer<OutlineItem> &parent, const QString &prefix);
    void fillNavigationMethodComboBox(const QSharedPointer<OutlineItem> &parent, const QString &prefix);
    void showOutlineNavigationBar(bool show);

    static QPointer<ScriptEditorWidget> currentSelectedCallstackLineEditor; //this static variable holds the (weak) pointer to the script editor widget that received the last "selected callstack line" selector.


signals:
    void removeAndDeleteScriptDockWidget(ScriptDockWidget* widget);                             /*!<  signal emitted if given ScriptDockWidget should be closed and removed by ScriptEditorOrganizer */

    void openScriptRequest(const QString &filename, ScriptDockWidget* scriptDockWidget);               /*!<  signal emitted if script with given filename should be opened in scriptDockWidget */

    void dockScriptTab(ScriptDockWidget* widget, int index, bool closeDockIfEmpty = false);     /*!<  signal emitted if tab with given index of given ScriptDockWidget should be docked in a docked ScriptDockWidget */
    void undockScriptTab(ScriptDockWidget* widget, int index, bool undockToNewScriptWindow = false, bool closeDockIfEmpty = false);   /*!<  signal emitted if tab with given index of given ScriptDockWidget should be undocked in an undocked ScriptDockWidget */

    void pythonRunFileRequest(QString filename);                                                /*!<  will be received by scriptEditorOrganizer, in order to save all unsaved changes first */
    void pythonDebugFileRequest(QString filename);                                              /*!<  will be received by scriptEditorOrganizer, in order to save all unsaved changes first */
    void pythonInterruptExecution();                                                            /*!<  will be received by PythonThread, directly */
    void pythonDebugCommand(tPythonDbgCmd cmd);                                                 /*!<  will be received by PythonThread, directly */
    void pythonRunSelection(QString selectionText);                                             /*!<  will be received by consoleWidget, directly */

    void addGoBackNavigationItem(const GoBackNavigationItem &item);

    void statusBarInformationChanged(
        const QPointer<ScriptDockWidget> sourceDockWidget,
        const QString &encoding,
        int line,
        int column);

private slots:
    void tabContextMenuEvent (QContextMenuEvent * event);

    void findTextExpr(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward, bool isQuickSeach);
    void replaceTextExpr(QString expr, QString replace);
    void replaceAllExpr(QString expr, QString replace, bool regExpr, bool caseSensitive, bool wholeWord, bool findInSel);
    void insertIconBrowserText(QString iconLink);

    void currentTabChanged(int index);
    void tabCloseRequested(int index);
    void tabCloseRequested(ScriptEditorWidget* sew, bool ignoreModifications);
    void scriptModificationChanged(bool changed);

    void updateEditorActions();
    void updatePythonActions();
    void updateTabContextActions();

    void mnuOpenIconBrowser();

    void mnuTabMoveLeft();
    void mnuTabMoveRight();
    void mnuTabMoveFirst();
    void mnuTabMoveLast();
    void mnuTabClose();
    void mnuTabCloseOthers();
    void mnuTabCloseAll();
    void mnuTabDock();
    void mnuTabUndock();
    void mnuNewScript();
    void mnuOpenScript();
    void mnuSaveScript();
    void mnuSaveScriptAs();
    void mnuSaveAllScripts();
    void mnuPrint();
    void mnuCut();
    void mnuCopy();
    void mnuPaste();
    void mnuUndo();
    void mnuRedo();
    void mnuComment();
    void mnuUncomment();
    void mnuIndent();
    void mnuUnindent();
    void mnuScriptRun();
    void mnuScriptRunSelection();
    void mnuScriptDebug();
    void mnuScriptStop();
    void mnuScriptContinue();
    void mnuScriptStep();
    void mnuScriptStepOver();
    void mnuScriptStepOut();
    void mnuFindTextExpr();
    void mnuReplaceTextExpr();
    void mnuGoto();
    void mnuToggleBookmark();
    void mnuInsertCodec();
    void mnuCopyFilename();
    void mnuPyCodeFormatting();
    void mnuPyDocstringGenerator();


    void menuLastFilesAboutToShow();
    void lastFileOpen(const QString &path);

    // Class Navigator
    void navigatorClassSelected(int row);
    void navigatorMethodSelected(int row);

    void loadSettings();
    void findWordWidgetFinished();

    void currentScriptCursorPositionChanged();

public slots:
    void editorMarginChanged();
    void updateCodeNavigation(ScriptEditorWidget *editor, QSharedPointer<OutlineItem> rootItem);
    void tabChangedRequest();
    void mnuFindSymbolsShow();
};

} //end namespace ito

#endif
