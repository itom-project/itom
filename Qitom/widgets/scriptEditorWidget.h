/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#ifndef SCRIPTEDITORWIDGET_H
#define SCRIPTEDITORWIDGET_H

#include "../models/breakPointModel.h"

#include "abstractPyScintillaWidget.h"

#include <qfilesystemwatcher.h>
#include <qmutex.h>
#include <qwidget.h>
#include <qstring.h>
#include <qmenu.h>
#include <qevent.h>


using namespace ito;

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

class ScriptEditorWidget : public AbstractPyScintillaWidget
{
    Q_OBJECT


public:
    ScriptEditorWidget(QWidget* parent = NULL);
    ~ScriptEditorWidget();

    RetVal saveFile(bool askFirst = true);
    RetVal saveAsFile(bool askFirst = true);

    RetVal openFile(QString file, bool ignorePresentDocument = false);

    inline QString getFilename() const {return filename; }
    inline bool hasNoFilename() const { return filename.isNull(); }
    inline bool getCanCopy() const { return canCopy; }
    inline bool isBookmarked() const { return bookmarkHandles.size() > 0; }
    inline QString getUntitledName() const { return tr("Untitled%1").arg(unnamedNumber); }

    RetVal setCursorPosAndEnsureVisible(int line);

protected:
    //void keyPressEvent (QKeyEvent *event);
    bool canInsertFromMimeData(const QMimeData *source) const;
//    void dragEnterEvent(QDragEnterEvent *event);
    void dropEvent(QDropEvent *event);

    virtual void loadSettings();

private:
    enum msgType
    {
        msgReturnInfo,
        msgReturnWarning,
        msgReturnError,
        msgTextInfo,
        msgTextWarning,
        msgTextError
    };

    RetVal initEditor();

    void contextMenuEvent (QContextMenuEvent * event);

    int getMarginNumber(int xPos);

    RetVal initMenus();

    RetVal toggleBookmark(int line);
    RetVal clearAllBookmarks();
    RetVal gotoNextBookmark();
    RetVal gotoPreviousBookmark();

    RetVal toggleBreakpoint(int line);
    RetVal toggleEnableBreakpoint(int line);
    RetVal editBreakpoint(int line);
    RetVal clearAllBreakpoints();
    RetVal gotoNextBreakPoint();
    RetVal gotoPreviousBreakPoint();

    RetVal changeFilename(QString newFilename);

    QFileSystemWatcher *m_pFileSysWatcher;
    QMutex fileSystemWatcherMutex;

    //!< marker handling
    std::list<int> bookmarkHandles;
    int syntaxErrorHandle;

    std::list<QPair<int,int> > breakPointMap; //!< <int bpHandle, int lineNo>

    unsigned int markBreakPoint;
    unsigned int markCBreakPoint;
    unsigned int markBreakPointDisabled;
    unsigned int markCBreakPointDisabled;
    unsigned int markBookmark;
    unsigned int markSyntaxError;

    unsigned int markCurrentLine;
    int markCurrentLineHandle;

    unsigned int markMask1;
    unsigned int markMask2;
    unsigned int markMaskBreakpoints;

    //QsciLexerPython* qSciLex;
    //QsciAPIs* qSciApi;

    //!< menus
    QMenu *bookmarkMenu;
    QMenu *syntaxErrorMenu;
    QMenu *breakpointMenu;
    QMenu *editorMenu;

    std::map<QString,QAction*> bookmarkMenuActions;
    std::map<QString,QAction*> breakpointMenuActions;
    std::map<QString,QAction*> editorMenuActions;

    int contextMenuLine;

    QString filename;
    int unnamedNumber;

    bool pythonBusy; //!< true: python is executing or debugging a script, a command...

    bool canCopy;

    static const QString lineBreak;
    static int unnamedAutoIncrement;

signals:
    void pythonRunFile(QString filename);
    void pythonRunSelection(QString selectionText);
    void pythonDebugFile(QString filename);
    void closeRequest(ScriptEditorWidget* sew, bool ignoreModifications); //signal emitted if this tab should be closed without considering any save-state
    void marginChanged();    

public slots:
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

    void menuCut();
    void menuCopy();
    void menuPaste();
    void menuIndent();
    void menuUnindent();
    void menuComment();
    void menuUncomment();

    void menuRunScript();
    void menuRunSelection();
    void menuDebugScript();
    void menuStopScript();

    void pythonStateChanged(tPythonTransitions pyTransition);
    void pythonDebugPositionChanged(QString filename, int lineno);

    void breakPointAdd(BreakPointItem bp, int row);
    void breakPointDelete(QString filename, int lineNo, int pyBpNumber);
    void breakPointChange(BreakPointItem oldBp, BreakPointItem newBp);


private slots:
    void marginClicked(int margin, int line, Qt::KeyboardModifiers state);
    void copyAvailable(bool yes);

    void nrOfLinesChanged();

    RetVal preShowContextMenuMargin();
    RetVal preShowContextMenuEditor();

    void fileSysWatcherFileChanged ( const QString & path );
};

#endif
