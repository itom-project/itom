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

#ifndef SCRIPTEDITORWIDGET_H
#define SCRIPTEDITORWIDGET_H

#include "../models/breakPointModel.h"

#include "abstractCodeEditorWidget.h"
#include "../codeEditor/panels/foldingPanel.h"
#include "../codeEditor/panels/checkerBookmarkPanel.h"
#include "../codeEditor/panels/breakpointPanel.h"
#include "../codeEditor/modes/errorLineHighlight.h"
#include "../codeEditor/modes/pyGotoAssignment.h"
#include "../codeEditor/modes/wordHoverTooltip.h"
#include "../codeEditor/panels/lineNumber.h"
#include "../codeEditor/codeCheckerItem.h"

#include "../global.h"

#include <qfilesystemwatcher.h>
#include <qmutex.h>
#include <qwidget.h>
#include <qstring.h>
#include <qmenu.h>
#include <qevent.h>
#include <qmetaobject.h>
#include <qsharedpointer.h>
#include "../models/classNavigatorItem.h"
#include "../models/bookmarkModel.h"

#include <QtPrintSupport/qprinter.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

  

namespace ito
{

struct ScriptEditorStorage
{
    QString     filename;
    int         firstVisibleLine;
    QList<int>  bookmarkLines; //! this is deprecated, since bookmarks are now managed by the global bookmarkModel
};


struct GoBackNavigationItem
{
    QString filename;
    int UID;
    QString shortText;
    int line;
    int column;
    QString origin;
};

}

Q_DECLARE_METATYPE(ito::ScriptEditorStorage) //must be outside of namespace
Q_DECLARE_METATYPE(QList<ito::ScriptEditorStorage>) //must be outside of namespace
Q_DECLARE_METATYPE(ito::GoBackNavigationItem) //must be outside of namespace
Q_DECLARE_METATYPE(QList<ito::GoBackNavigationItem>) //must be outside of namespace

namespace ito
{


class ScriptEditorWidget : public AbstractCodeEditorWidget
{
    Q_OBJECT

public:
    ScriptEditorWidget(BookmarkModel *bookmarkModel, QWidget* parent = NULL);
    ~ScriptEditorWidget();

    RetVal saveFile(bool askFirst = true);
    RetVal saveAsFile(bool askFirst = true);

    RetVal openFile(QString file, bool ignorePresentDocument = false);

    bool keepIndentationOnPaste() const;
    void setKeepIndentationOnPaste(bool value);

    inline QString getFilename() const {return m_filename; }
    inline bool hasNoFilename() const { return m_filename.isNull(); }
    inline int getUID() const { return m_uid; }
    bool getCanCopy() const;
    inline QString getUntitledName() const { return tr("Untitled%1").arg(m_uid); }
    inline QString getCurrentClass() const { return m_currentClass; } //currently chosen class in class navigator for this script editor widget
    inline QString getCurrentMethod() const { return m_currentMethod; } //currently chosen method in class navigator for this script editor widget

    RetVal setCursorPosAndEnsureVisible(const int line, bool errorMessageClick = false, bool showSelectedCallstackLine = false);
    RetVal setCursorPosAndEnsureVisibleWithSelection(const int line, const QString &currentClass, const QString &currentMethod);

    void removeCurrentCallstackLine(); //!< removes the current-callstack-line arrow from the breakpoint panel, if currently displayed

    const ScriptEditorStorage saveState() const;
    RetVal restoreState(const ScriptEditorStorage &data);

    RetVal toggleBookmark(int line);

    virtual bool removeTextBlockUserData(TextBlockUserData* userData);

    //!< if UidFilter is -1, the current cursor position is always reported, else only if its editorUID is equal to UIDFilter
    void reportCurrentCursorAsGoBackNavigationItem(const QString &reason, int UIDFilter = -1);

    static QString filenameFromUID(int UID, bool &found);

protected:

    bool canInsertFromMimeData(const QMimeData *source) const;
    void insertFromMimeData(const QMimeData *source);

    void dropEvent(QDropEvent *event);
    virtual void loadSettings();
    bool event(QEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);

    virtual void contextMenuAboutToShow(int contextMenuLine);

    void reportGoBackNavigationCursorMovement(const CursorPosition &cursor, const QString &origin) const;

private:
    enum markerType
    {   
        markerBookmark = 1,
        markerPyBug = 2,
        markerBookmarkAndPyBug = markerBookmark | markerPyBug
    };

    RetVal initEditor();
    void initMenus();
    
    bool lineAcceptsBPs(int line);

    RetVal changeFilename(const QString &newFilename);

    QFileSystemWatcher *m_pFileSysWatcher;
    QMutex m_fileSystemWatcherMutex;

    // the following variables are related to the code checker feature of Python
    bool m_codeCheckerEnabled;
    int m_codeCheckerInterval;      //!< timeout time after the last key press, when the next code check is called.
    QTimer *m_codeCheckerCallTimer; //!< timer, that is used to call a new code check after a certain time after the key press

    Qt::CaseSensitivity m_filenameCaseSensitivity;

    //!< menus
    QMenu *m_contextMenu;
    std::map<QString,QAction*> m_editorMenuActions;

    QString m_filename; //!< canonical filename of the script or empty if no script name has been given yet
    int m_uid;

    //go back navigation features
    bool m_cursorJumpLastAction; //!< true if the last cursor movement was a jump by a mouse click
    CursorPosition m_cursorBeforeMouseClick;

    bool m_pythonBusy; //!< true: python is executing or debugging a script, a command...
    bool m_pythonExecutable;

    bool m_canCopy;
    bool m_keepIndentationOnPaste;

    BookmarkModel *m_pBookmarkModel; //! borrowed reference to the bookmark model. The owner of this model is the ScriptEditorOrganizer.

    QSharedPointer<FoldingPanel> m_foldingPanel;
    QSharedPointer<CheckerBookmarkPanel> m_checkerBookmarkPanel;
    QSharedPointer<BreakpointPanel> m_breakpointPanel;
    QSharedPointer<ErrorLineHighlighterMode> m_errorLineHighlighterMode;
    QSharedPointer<LineNumberPanel> m_lineNumberPanel;
    QSharedPointer<PyGotoAssignmentMode> m_pyGotoAssignmentMode;
    QSharedPointer<WordHoverTooltipMode> m_wordHoverTooltipMode;

    static const QString lineBreak;
    static int currentMaximumUID;
    static CursorPosition currentGlobalEditorCursorPos; //! the current cursor position within all opened editor widgets
    static QHash<int, ScriptEditorWidget*> editorByUID; //! hash table that maps the UID to its instance of ScriptEditorWidget*

    // Class Navigator
    bool m_classNavigatorEnabled;               // Enable Class-Navigator
    QTimer *m_classNavigatorTimer;              // Class Navigator Timer
    bool m_classNavigatorTimerEnabled;          // Class Navigator Timer Enable
    int m_classNavigatorInterval;               // Class Navigator Timer Interval
    QString m_currentClass;
    QString m_currentMethod;

    int buildClassTree(ClassNavigatorItem *parent, int parentDepth, int lineNumber, int singleIndentation = -1);
    int getIndentationLength(const QString &str) const;

signals:
    void pythonRunFile(QString filename);
    void pythonRunSelection(QString selectionText);
    void pythonDebugFile(QString filename);
    void closeRequest(ScriptEditorWidget* sew, bool ignoreModifications); //signal emitted if this tab should be closed without considering any save-state
    void marginChanged();
    void requestModelRebuild(ScriptEditorWidget *editor);
    void addGoBackNavigationItem(const GoBackNavigationItem &item);
    void tabChangeRequested();

public slots:
    void triggerCodeChecker();
    void codeCheckResultsReady(QList<ito::CodeCheckerItem> codeCheckerItems);
    void codeCheckerResultsChanged(const QList<ito::CodeCheckerItem> &codeCheckerItems);

    void menuCut();
    void menuCopy();
    void menuPaste();
    void menuIndent();
    void menuUnindent();
    void menuComment();
    void menuUncomment();

    void menuFoldAll();
    void menuUnfoldAll();
    void menuFoldUnfoldToplevel();
    void menuFoldUnfoldAll();

    void menuRunScript();
    void menuRunSelection();
    void menuDebugScript();
    void menuStopScript();

    void menuInsertCodec();

    void pythonStateChanged(tPythonTransitions pyTransition);
    void pythonDebugPositionChanged(QString filename, int lineno);

    void breakPointAdd(BreakPointItem bp, int row);
    void breakPointDelete(QString filename, int lineNo, int pyBpNumber);
    void breakPointChange(BreakPointItem oldBp, BreakPointItem newBp);

    void updateSyntaxCheck();

    // Class Navigator  
    ClassNavigatorItem* getPythonNavigatorRoot(); //creates new tree of current python code structure and returns its root pointer. Caller must delete the root pointer after usage.

    void print();

private slots:
    void toggleBookmarkRequested(int line);
    void onBookmarkAdded(const BookmarkItem &item);  
    void onBookmarkDeleted(const BookmarkItem &item);
    

    RetVal toggleBreakpoint(int line);
    RetVal toggleEnableBreakpoint(int line);
    RetVal editBreakpoint(int line);
    RetVal clearAllBreakpoints();
    RetVal gotoNextBreakPoint();
    RetVal gotoPreviousBreakPoint();

    void gotoAssignmentOutOfDoc(PyAssignment ref);

    void copyAvailable(const bool yes);

    void classNavTimerElapsed();

    void nrOfLinesChanged();

    void fileSysWatcherFileChanged ( const QString & path );
    void printPreviewRequested(QPrinter *printer);

    void dumpFoldsToConsole(bool);
    void onCursorPositionChanged();

    void tabChangeRequest();
};

} //end namespace ito

#endif
