/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
#include "../codeEditor/panels/lineNumber.h"

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

#include <QtPrintSupport/qprinter.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE



namespace ito
{

struct ScriptEditorStorage
{
    QByteArray  filename;
    int         firstVisibleLine;
    QList<int>  bookmarkLines;
};

}

Q_DECLARE_METATYPE(ito::ScriptEditorStorage) //must be outside of namespace
Q_DECLARE_METATYPE(QList<ito::ScriptEditorStorage>) //must be outside of namespace

namespace ito
{


class ScriptEditorWidget : public AbstractCodeEditorWidget
{
    Q_OBJECT

public:
    ScriptEditorWidget(QWidget* parent = NULL);
    ~ScriptEditorWidget();

    RetVal saveFile(bool askFirst = true);
    RetVal saveAsFile(bool askFirst = true);

    RetVal openFile(QString file, bool ignorePresentDocument = false);

    inline QString getFilename() const {return m_filename; }
    inline bool hasNoFilename() const { return m_filename.isNull(); }
    inline bool getCanCopy() const { return canCopy; }
    bool isBookmarked() const;
    inline QString getUntitledName() const { return tr("Untitled%1").arg(unnamedNumber); }
    inline QString getCurrentClass() const { return m_currentClass; } //currently chosen class in class navigator for this script editor widget
    inline QString getCurrentMethod() const { return m_currentMethod; } //currently chosen method in class navigator for this script editor widget

    RetVal setCursorPosAndEnsureVisible(const int line, bool errorMessageClick = false);
    RetVal setCursorPosAndEnsureVisibleWithSelection(const int line, const QString &currentClass, const QString &currentMethod);

    const ScriptEditorStorage saveState() const;
    RetVal restoreState(const ScriptEditorStorage &data);

    RetVal toggleBookmark(int line);
    RetVal clearAllBookmarks();
    RetVal gotoNextBookmark();
    RetVal gotoPreviousBookmark();

    virtual bool removeTextBlockUserData(TextBlockUserData* userData);

protected:
    bool canInsertFromMimeData(const QMimeData *source) const;

    void dropEvent(QDropEvent *event);
    virtual void loadSettings();
    bool event(QEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

    virtual void contextMenuAboutToShow(int contextMenuLine);

    virtual void addContextAction(QAction *action, const QString &categoryName);

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
    QMutex fileSystemWatcherMutex;

    bool m_syntaxCheckerEnabled;
    int m_syntaxCheckerInterval;
    QTimer *m_syntaxTimer;

    //!< menus
    QMenu *m_contextMenu;

    std::map<QString,QAction*> bookmarkMenuActions;
    std::map<QString,QAction*> m_editorMenuActions;

    QString m_filename; //!< canonical filename of the script or empty if no script name has been given yet
    int unnamedNumber;

    bool pythonBusy; //!< true: python is executing or debugging a script, a command...
    bool m_pythonExecutable;

    bool canCopy;

    QSharedPointer<FoldingPanel> m_foldingPanel;
    QSharedPointer<CheckerBookmarkPanel> m_checkerBookmarkPanel;
    QSharedPointer<BreakpointPanel> m_breakpointPanel;
    QSharedPointer<ErrorLineHighlighterMode> m_errorLineHighlighterMode;
    QSharedPointer<LineNumberPanel> m_lineNumberPanel;
    QSharedPointer<PyGotoAssignmentMode> m_pyGotoAssignmentMode;

    static const QString lineBreak;
    static int unnamedAutoIncrement;

    // Class Navigator
    bool m_ClassNavigatorEnabled;               // Enable Class-Navigator
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

public slots:
    
    void checkSyntax();
    void syntaxCheckResult(QString a, QString b);
    void errorListChange(const QStringList &errorList);

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
    void gotoBookmarkRequested(bool next);
    void clearAllBookmarksRequested();

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
};

} //end namespace ito

#endif
