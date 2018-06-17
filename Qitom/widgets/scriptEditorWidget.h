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

#define USE_PYQODE 1

#include "../models/breakPointModel.h"

#ifdef USE_PYQODE
    #include "abstractCodeEditorWidget.h"
    #include "../codeEditor/panels/foldingPanel.h"
    #include "../codeEditor/panels/checkerBookmarkPanel.h"
    #include "../codeEditor/panels/breakpointPanel.h"
    #include "../codeEditor/modes/errorLineHighlight.h"
    #include "../codeEditor/panels/lineNumber.h"
    #include "../codeEditor/modes/pyCalltips.h"
#else
    #include "abstractPyScintillaWidget.h"
#endif

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

#if QT_VERSION >= 0x050000
    #include <QtPrintSupport/qprinter.h>
#else
    #include <qprinter.h>
#endif

#ifndef USE_PYQODE
    #include <Qsci/qsciprinter.h>
#endif

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

#ifdef USE_PYQODE
    class ScriptEditorWidget : public AbstractCodeEditorWidget
#else
    class ScriptEditorWidget : public AbstractPyScintillaWidget
#endif
{
    Q_OBJECT

public:
    ScriptEditorWidget(QWidget* parent = NULL);
    ~ScriptEditorWidget();

#ifndef USE_PYQODE
    bool m_errorMarkerVisible;
    int m_errorMarkerNr; //number of indicator which marks the line with current erro
#endif

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

#ifdef USE_PYQODE
    virtual bool removeTextBlockUserData(TextBlockUserData* userData);
#endif

protected:
    //void keyPressEvent (QKeyEvent *event);
    bool canInsertFromMimeData(const QMimeData *source) const;

#ifndef USE_PYQODE
    void autoAdaptLineNumberColumnWidth();
#endif

    void dropEvent(QDropEvent *event);
    virtual void loadSettings();
    bool event(QEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);

private:

    enum markerType
    {   
        markerBookmark = 1,
        markerPyBug = 2,
        markerBookmarkAndPyBug = markerBookmark | markerPyBug
    };

    RetVal initEditor();

    void contextMenuEvent (QContextMenuEvent * event);

    int getMarginNumber(int xPos);

    RetVal initMenus();

    bool lineAcceptsBPs(int line);

    RetVal changeFilename(const QString &newFilename);

    QFileSystemWatcher *m_pFileSysWatcher;
    QMutex fileSystemWatcherMutex;

#ifndef USE_PYQODE
    //!< marker handling
    struct BookmarkErrorEntry
    {
        int handle;
        int type; //entry of enum markerType
        QString errorMessage;
        QString errorComment;
        int errorPos;
    };
    QList<BookmarkErrorEntry> bookmarkErrorHandles;
    int syntaxErrorHandle;
#endif

    bool m_syntaxCheckerEnabled;
    int m_syntaxCheckerInterval;
    QTimer *m_syntaxTimer;

#ifndef USE_PYQODE
    struct BPMarker
    {
        int bpHandle;
        int lineNo;
        bool markedForDeletion;
    };

    QList<BPMarker> m_breakPointMap; //!< <int bpHandle, int lineNo>

    unsigned int markBreakPoint;
    unsigned int markCBreakPoint;
    unsigned int markBreakPointDisabled;
    unsigned int markCBreakPointDisabled;
    unsigned int markBookmark;
    unsigned int markSyntaxError;
    unsigned int markBookmarkSyntaxError;

    unsigned int markCurrentLine;
    int markCurrentLineHandle;

    unsigned int markMask1;
    unsigned int markMask2;
    unsigned int markMaskBreakpoints;
#endif

    //!< menus
#ifndef USE_PYQODE
    QMenu *bookmarkMenu;
#endif
    QMenu *breakpointMenu;
    QMenu *editorMenu;

    std::map<QString,QAction*> bookmarkMenuActions;
    std::map<QString,QAction*> breakpointMenuActions;
    std::map<QString,QAction*> editorMenuActions;

    int contextMenuLine;

    QString m_filename; //!< canonical filename of the script or empty if no script name has been given yet
    int unnamedNumber;

    bool pythonBusy; //!< true: python is executing or debugging a script, a command...
    bool m_pythonExecutable;

    bool canCopy;

#ifdef USE_PYQODE
    QSharedPointer<FoldingPanel> m_foldingPanel;
    QSharedPointer<CheckerBookmarkPanel> m_checkerBookmarkPanel;
    QSharedPointer<BreakpointPanel> m_breakpointPanel;
    QSharedPointer<ErrorLineHighlighterMode> m_errorLineHighlighterMode;
    QSharedPointer<LineNumberPanel> m_lineNumberPanel;
    QSharedPointer<PyCalltipsMode> m_pyCallbackMode;
#endif

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

#ifndef USE_PYQODE
    void menuToggleBookmark();
    void menuClearAllBookmarks();
    void menuGotoNextBookmark();
    void menuGotoPreviousBookmark();

    void menuToggleBreakpoint();
    void menuToggleEnableBreakpoint();
    void menuEditBreakpoint();
    void menuClearAllBreakpoints();
    void menuGotoNextBreakPoint();
    void menuGotoPreviousBreakPoint();
#endif

    void menuCut();
    void menuCopy();
    void menuPaste();
    void menuIndent();
    void menuUnindent();
    void menuComment();
    void menuUncomment();

#ifdef USE_PYQODE
    void menuFoldAll();
#endif
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

#ifdef USE_PYQODE
    void toggleBookmarkRequested(int line);
    void gotoBookmarkRequested(bool next);
    void clearAllBookmarksRequested();

    RetVal toggleBreakpoint(int line);
    RetVal toggleEnableBreakpoint(int line);
    RetVal editBreakpoint(int line);
    RetVal clearAllBreakpoints();
    RetVal gotoNextBreakPoint();
    RetVal gotoPreviousBreakPoint();
#else
    void marginClicked(int margin, int line, Qt::KeyboardModifiers state);
#endif

    void copyAvailable(const bool yes);

    void classNavTimerElapsed();

    void nrOfLinesChanged();

    RetVal preShowContextMenuMargin();
    RetVal preShowContextMenuEditor();

    void fileSysWatcherFileChanged ( const QString & path );
    void printPreviewRequested(QPrinter *printer);
};

#ifdef USE_PYQODE
    class ScriptEditorPrinter : public QPrinter
#else
    class ScriptEditorPrinter : public QsciPrinter
#endif
{
public:

#ifdef USE_PYQODE
    ScriptEditorPrinter(QPrinter::PrinterMode mode=QPrinter::ScreenResolution) : QPrinter(mode)
    {
        setColorMode(QPrinter::Color);
        setPageOrder(QPrinter::FirstPageFirst);
    }
#else
    ScriptEditorPrinter(QPrinter::PrinterMode mode=QPrinter::ScreenResolution) : QsciPrinter(mode) {}
#endif

    virtual void formatPage( QPainter &painter, bool drawing, QRect &area, int pagenr );
};

} //end namespace ito

#endif
