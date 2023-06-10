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
#include "../codeEditor/modes/pyDocstringGenerator.h"
#include "../codeEditor/modes/wordHoverTooltip.h"
#include "../codeEditor/panels/lineNumber.h"
#include "../codeEditor/codeCheckerItem.h"
#include "../codeEditor/pyCodeFormatter.h"

#include "../global.h"

#include <qfilesystemwatcher.h>
#include <qwidget.h>
#include <qstring.h>
#include <qmenu.h>
#include <qevent.h>
#include <qpointer.h>
#include <qmetaobject.h>
#include <qsharedpointer.h>
#include <qregularexpression.h>
#include "../models/outlineItem.h"
#include "../models/bookmarkModel.h"
#include "../helper/IOHelper.h"
#include "../ui/dialogScriptCharsetEncoding.h"

#include <QtPrintSupport/qprinter.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE



namespace ito
{

class BreakPointModel;

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
    ScriptEditorWidget(BookmarkModel *bookmarkModel, QWidget* parent = nullptr);
    ~ScriptEditorWidget();

    RetVal saveFile(bool askFirst = true);
    RetVal saveAsFile(bool askFirst = true);

    RetVal openFile(const QString &fileName, bool ignorePresentDocument = false, QWidget *parent = nullptr);

    bool keepIndentationOnPaste() const;
    void setKeepIndentationOnPaste(bool value);

    inline QString getFilename() const {return m_filename; }
    inline bool hasNoFilename() const { return m_filename.isNull(); }
    inline int getUID() const { return m_uid; }
    bool getCanCopy() const;
    inline QString getUntitledName() const { return tr("Untitled%1").arg(m_uid); }

    RetVal setCursorPosAndEnsureVisible(const int line, bool errorMessageClick = false, bool showSelectedCallstackLine = false);
    RetVal showLineAndHighlightWord(const int line, const QString &highlightedText, Qt::CaseSensitivity caseSensitivity = Qt::CaseInsensitive);

    void removeCurrentCallstackLine(); //!< removes the current-callstack-line arrow from the breakpoint panel, if currently displayed

    const ScriptEditorStorage saveState() const;
    RetVal restoreState(const ScriptEditorStorage &data);

    RetVal toggleBookmark(int line);

    virtual bool removeTextBlockUserData(TextBlockUserData* userData);

    //!< if UidFilter is -1, the current cursor position is always reported, else only if its editorUID is equal to UIDFilter
    void reportCurrentCursorAsGoBackNavigationItem(const QString &reason, int UIDFilter = -1);

    //!< wrapper for undo() or redo() that tries to keep breakpoints and bookmarks
    void startUndoRedo(bool unundoNotRedo);

    QSharedPointer<OutlineItem> parseOutline(bool forceParsing = false) const;

    //!< returns true if the current line can be a trigger to insert a template docstring
    //!< for a possible method / function, this line belongs to.
    bool currentLineCanHaveDocstring() const;

    IOHelper::CharsetEncodingItem charsetEncoding() const { return m_charsetEncoding; }

    static QString filenameFromUID(int UID, bool &found);

protected:

    bool canInsertFromMimeData(const QMimeData *source) const;
    void insertFromMimeData(const QMimeData *source);

    void dropEvent(QDropEvent *event);
    void dragEnterEvent(QDragEnterEvent *event);
    void dragLeaveEvent(QDragLeaveEvent *event);
    virtual void loadSettings();
    bool event(QEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);

    virtual void contextMenuAboutToShow(int contextMenuLine);

    void reportGoBackNavigationCursorMovement(const CursorPosition &cursor, const QString &origin) const;

    void replaceSelectionAndKeepBookmarksAndBreakpoints(QTextCursor &cursor, const QString &newString);
    QVector<int> compareTexts(const QString &oldText, const QString &newText);

    BreakPointModel* getBreakPointModel();
    const BreakPointModel* getBreakPointModel() const;

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

    IOHelper::CharsetEncodingItem guessEncoding(const QByteArray &content) const;

    void changeFileSaveEncoding(const IOHelper::CharsetEncodingItem &encoding);

    QFileSystemWatcher *m_pFileSysWatcher;

    // the following variables are related to the code checker feature of Python
    bool m_codeCheckerEnabled;
    int m_codeCheckerInterval;      //!< timeout time after the last key press, when the next code check is called.
    QTimer *m_codeCheckerCallTimer; //!< timer, that is used to call a new code check after a certain time after the key press

    Qt::CaseSensitivity m_filenameCaseSensitivity;

    //!< menus
    std::map<QString,QAction*> m_editorMenuActions;

    QString m_filename; //!< canonical filename of the script or empty if no script name has been given yet
    int m_uid;

    //go back navigation features
    bool m_cursorJumpLastAction; //!< true if the last cursor movement was a jump by a mouse click
    CursorPosition m_cursorBeforeMouseClick;

    bool m_pythonBusy; //!< true: python is executing or debugging a script, a command...
    bool m_pythonExecutable;

    //!< to accept drop events of other files dropped onto this file, the script
    //!< must not be readonly. Therefore a readonly script will be temporary set in a read/write mode
    bool m_wasReadonly;
    bool m_canCopy;
    bool m_keepIndentationOnPaste;
    int m_textBlockLineIdxAboutToBeDeleted; //!< if != -1, a TextBlockUserData in the line index is about to be removed.
    BookmarkModel *m_pBookmarkModel; //! borrowed reference to the bookmark model. The owner of this model is the ScriptEditorOrganizer.

    QSharedPointer<PyCodeFormatter> m_pyCodeFormatter;

    //!< the current command string for the python auto code formatting.
    QString m_autoCodeFormatCmd;

    //!< the current command string for the python imports sorting (or empty, if this pre-step is not enabled)
    QString m_autoCodeFormatPreCmd;

    //!< this is the encoding of this script, hence,
    //!< the encoding that was used to load this script from
    //!< a file and will also be used to store it in a file.
    IOHelper::CharsetEncodingItem m_charsetEncoding;
    bool m_charsetDefined;
    bool m_charsetEncodingAutoGuess;

    QSharedPointer<FoldingPanel> m_foldingPanel;
    QSharedPointer<CheckerBookmarkPanel> m_checkerBookmarkPanel;
    QSharedPointer<BreakpointPanel> m_breakpointPanel;
    QSharedPointer<ErrorLineHighlighterMode> m_errorLineHighlighterMode;
    QSharedPointer<LineNumberPanel> m_lineNumberPanel;
    QSharedPointer<PyGotoAssignmentMode> m_pyGotoAssignmentMode;
    QSharedPointer<WordHoverTooltipMode> m_wordHoverTooltipMode;
    QSharedPointer<PyDocstringGeneratorMode> m_pyDocstringGeneratorMode;

    static const QString lineBreak;
    static int currentMaximumUID;
    static CursorPosition currentGlobalEditorCursorPos; //! the current cursor position within all opened editor widgets
    static QHash<int, ScriptEditorWidget*> editorByUID; //! hash table that maps the UID to its instance of ScriptEditorWidget*

    // Outline
    QTimer *m_outlineTimer; //!< timer to recreate the outline model with a certain delay
    bool m_outlineTimerEnabled; //!<
    int m_currentLineIndex; //!< current line index of the cursor
    mutable QSharedPointer<OutlineItem> m_rootOutlineItem; //!< cache for the latest outline items
    mutable bool m_outlineDirty;
    QRegularExpression m_regExpClass; //!< regular expression to parse the definition of a class
    QRegularExpression m_regExpDecorator; //!< regular expression to parse a decorator
    QRegularExpression m_regExpMethodStart; //!< regular expression to parse the start of a method definition
    QRegularExpression m_regExpMethod; //!< regular expression to parse a full method definition

    void parseOutlineRecursive(QSharedPointer<OutlineItem> &parent) const;
    QSharedPointer<OutlineItem> checkBlockForOutlineItem(int startLineIdx, int endLineIdx) const;

signals:
    void pythonRunFile(QString filename);
    void pythonRunSelection(QString selectionText);
    void pythonDebugFile(QString filename);
    void closeRequest(ScriptEditorWidget* sew, bool ignoreModifications); //signal emitted if this tab should be closed without considering any save-state
    void marginChanged();
    void outlineModelChanged(ScriptEditorWidget *editor, QSharedPointer<OutlineItem> rootItem);
    void addGoBackNavigationItem(const GoBackNavigationItem &item);
    void tabChangeRequested();
    void findSymbolsShowRequested();

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

    void menuPyCodeFormatting();
    void menuGenerateDocstring();
    void menuScriptCharsetEncoding();

    void pythonStateChanged(tPythonTransitions pyTransition);
    void pythonDebugPositionChanged(QString filename, int lineno);

    void breakPointAdd(BreakPointItem bp, int row);
    void breakPointDelete(QString filename, int lineIdx, int pyBpNumber);
    void breakPointChange(BreakPointItem oldBp, BreakPointItem newBp);

    void updateSyntaxCheck();

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

    void outlineTimerElapsed();

    void nrOfLinesChanged();

    void fileSysWatcherFileChanged(const QString &path);
    void fileSysWatcherFileChangedStep2(const QString &path);
    void printPreviewRequested(QPrinter *printer);

    void dumpFoldsToConsole(bool);
    void onCursorPositionChanged();
    void onTextChanged();
    void tabChangeRequest();

    void pyCodeFormatterDone(bool success, QString code);
};

} //end namespace ito

#endif
