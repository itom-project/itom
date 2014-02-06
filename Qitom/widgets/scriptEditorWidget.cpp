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

#include "../python/pythonEngineInc.h"
#include "../widgets/mainWindow.h"
#include "scriptEditorWidget.h"

#include "../global.h"
#include "../Qitom/AppManagement.h"

#include <qfileinfo.h>
#include "../ui/dialogEditBreakpoint.h"

#include <qsciprinter.h>

namespace ito 
{

//!< constants
const QString ScriptEditorWidget::lineBreak = QString("\n");

int ScriptEditorWidget::unnamedAutoIncrement = 1;

//----------------------------------------------------------------------------------------------------------------------------------
ScriptEditorWidget::ScriptEditorWidget(QWidget* parent) :
    AbstractPyScintillaWidget(parent), 
    m_pFileSysWatcher(NULL), 
    contextMenuLine(-1), 
    pythonBusy(false), 
    canCopy(false)
{
    filename = QString();

    unnamedNumber = ScriptEditorWidget::unnamedAutoIncrement++;

    breakPointMap.clear();

    bookmarkHandles.clear();
    bookmarkMenuActions.clear();

    initEditor();

    initMenus();

    m_pFileSysWatcher = new QFileSystemWatcher(this);
    connect(m_pFileSysWatcher, SIGNAL(fileChanged(const QString&)), this, SLOT(fileSysWatcherFileChanged(const QString&)));

    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    const MainWindow *mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());

    if (pyEngine) 
    {
        pythonBusy = pyEngine->isPythonBusy();
        connect(pyEngine, SIGNAL(pythonDebugPositionChanged(QString, int)), this, SLOT(pythonDebugPositionChanged(QString, int)));
        connect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
    
        connect(this, SIGNAL(pythonRunFile(QString)), pyEngine, SLOT(pythonRunFile(QString)));
        connect(this, SIGNAL(pythonDebugFile(QString)), pyEngine, SLOT(pythonDebugFile(QString)));

        connect(this, SIGNAL(pythonRunSelection(QString)), mainWin, SLOT(pythonRunSelection(QString)));

        const BreakPointModel *bpModel = pyEngine->getBreakPointModel();

        connect(bpModel, SIGNAL(breakPointAdded(BreakPointItem, int)), this, SLOT(breakPointAdd(BreakPointItem, int)));
        connect(bpModel, SIGNAL(breakPointDeleted(QString, int, int)), this, SLOT(breakPointDelete(QString, int, int)));
        connect(bpModel, SIGNAL(breakPointChanged(BreakPointItem, BreakPointItem)), this, SLOT(breakPointChange(BreakPointItem, BreakPointItem)));

        //!< check if BreakPointModel already contains breakpoints for this editor and load them
        if (getFilename() != "")
        {
            QModelIndexList modelIndexList = bpModel->getBreakPointIndizes(getFilename());
            QList<BreakPointItem> bpItems = bpModel->getBreakPoints(modelIndexList);

            for (int i = 0; i < bpItems.size(); i++)
            {
                breakPointAdd(bpItems.at(i), i);
            }

        }
    }

    connect(this, SIGNAL(linesChanged()), this, SLOT(nrOfLinesChanged()));
    connect(this, SIGNAL(copyAvailable(bool)), this, SLOT(copyAvailable(bool)));
    setAcceptDrops(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
ScriptEditorWidget::~ScriptEditorWidget()
{
    const PythonEngine *pyEngine = PythonEngine::getInstance();
    const MainWindow *mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());

    if (pyEngine)
    {
        const BreakPointModel *bpModel = pyEngine->getBreakPointModel();

        disconnect(pyEngine, SIGNAL(pythonDebugPositionChanged(QString, int)), this, SLOT(pythonDebugPositionChanged(QString, int)));
        disconnect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));

        disconnect(this, SIGNAL(pythonRunFile(QString)), pyEngine, SLOT(pythonRunFile(QString)));
        disconnect(this, SIGNAL(pythonDebugFile(QString)), pyEngine, SLOT(pythonDebugFile(QString)));

        disconnect(this, SIGNAL(pythonRunSelection(QString)), mainWin, SLOT(pythonRunSelection(QString)));

        disconnect(bpModel, SIGNAL(breakPointAdded(BreakPointItem, int)), this, SLOT(breakPointAdd(BreakPointItem, int)));
        disconnect(bpModel, SIGNAL(breakPointDeleted(QString, int, int)), this, SLOT(breakPointDelete(QString, int, int)));
        disconnect(bpModel, SIGNAL(breakPointChanged(BreakPointItem, BreakPointItem)), this, SLOT(breakPointChange(BreakPointItem, BreakPointItem)));

        //!< delete remaining break-points (not neccesary)
        /*if (0)
        {
            QModelIndexList list = bpModel->getBreakPointIndizes(getFilename());
            bpModel->deleteBreakPoints(list);
        }*/
    }

    disconnect(this, SIGNAL(linesChanged()), this, SLOT(nrOfLinesChanged()));
    disconnect(this, SIGNAL(copyAvailable(bool)), this, SLOT(copyAvailable(bool)));

    DELETE_AND_SET_NULL(m_pFileSysWatcher);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::initEditor()
{
    setPaper(QColor(1, 81, 107));

    //reset standard margins settings
    for (int i = 1; i <= 4; i++)
    {
        setMarginLineNumbers(i, false);
        setMarginMarkerMask(i, 0);
        setMarginWidth(i, 0);
        setMarginSensitivity(i, false);
    }

    setMarginWidth(1, 16);
    setMarginWidth(2, 35);
    setMarginWidth(3, 18);
    setMarginWidth(4, 18);

    setMarginSensitivity(1, true);
    setMarginSensitivity(3, true);

    setMarginLineNumbers(2, true);

    setMarginType(1, QsciScintilla::SymbolMargin); //!< bookmark margin
    setMarginType(2, QsciScintilla::NumberMargin); //!< line number
    setMarginType(3, QsciScintilla::SymbolMargin); //!< breakpoint, syntax error margin
    setMarginType(4, QsciScintilla::SymbolMargin); //!< folding margin

    setFolding(QsciScintilla::PlainFoldStyle, 4);

    markBreakPoint = markerDefine(QPixmap(":/breakpoints/icons/itomBreak.png"));
    markCBreakPoint = markerDefine(QPixmap(":/breakpoints/icons/itomcBreak.png"));
    markBreakPointDisabled = markerDefine(QPixmap(":/breakpoints/icons/itomBreakDisabled.png"));
    markCBreakPointDisabled = markerDefine(QPixmap(":/breakpoints/icons/itomCBreakDisabled.png"));
    markBookmark = markerDefine(QPixmap(":/bookmark/icons/bookmark.png"));
    markSyntaxError = markerDefine(QPixmap(":/script/icons/syntaxError.png"));

    markCurrentLine = markerDefine(QPixmap(":/script/icons/currentLine.png"));
    markCurrentLineHandle = -1;

    markMaskBreakpoints = (1 << markBreakPoint) | (1 << markCBreakPoint)  | (1 << markBreakPointDisabled)  | (1 << markCBreakPointDisabled) | (1 << markCurrentLine);
    markMask1 = markMaskBreakpoints;
    markMask2 = (1 << markBookmark) | (1 << markSyntaxError);

    setMarginMarkerMask(1, markMask2);
    setMarginMarkerMask(3, markMask1);

    setBraceMatching(QsciScintilla::StrictBraceMatch); 
    setMatchedBraceBackgroundColor(QColor("lightGray"));
    setMatchedBraceForegroundColor(QColor("blue"));

    connect(this, SIGNAL(marginClicked(int, int, Qt::KeyboardModifiers)), this, SLOT(marginClicked(int, int, Qt::KeyboardModifiers)));

    loadSettings();

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    if (settings.value("showWhitespace", true).toBool())
    {
        setWhitespaceVisibility(QsciScintilla::WsVisible);
    }
    else
    {
        setWhitespaceVisibility(QsciScintilla::WsInvisible);
    }

    AbstractPyScintillaWidget::loadSettings();
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::initMenus()
{
    bookmarkMenu = new QMenu(this);
    bookmarkMenuActions["toggleBM"] = bookmarkMenu->addAction(QIcon(":/bookmark/icons/bookmarkToggle.png"), tr("&toggle bookmark"), this, SLOT(menuToggleBookmark()));
    bookmarkMenuActions["nextBM"] = bookmarkMenu->addAction(QIcon(":/bookmark/icons/bookmarkNext.png"), tr("next bookmark"), this, SLOT(menuGotoNextBookmark()));
    bookmarkMenuActions["prevBM"] = bookmarkMenu->addAction(QIcon(":/bookmark/icons/bookmarkPrevious.png"), tr("previous bookmark"), this, SLOT(menuGotoPreviousBookmark()));
    bookmarkMenuActions["clearAllBM"] = bookmarkMenu->addAction(QIcon(":/bookmark/icons/bookmarkClearAll.png"), tr("clear all bookmarks"), this, SLOT(menuClearAllBookmarks()));

    connect(bookmarkMenu, SIGNAL(aboutToShow()), this, SLOT(preShowContextMenuMargin()));

    breakpointMenu = new QMenu(this);
    breakpointMenuActions["toggleBP"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/itomBreak.png"), tr("&toggle breakpoint"), this, SLOT(menuToggleBreakpoint()));
    breakpointMenuActions["toggleBPEnabled"] = breakpointMenu->addAction(tr("&disable breakpoint"), this, SLOT(menuToggleEnableBreakpoint()));
    breakpointMenuActions["editConditionBP"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/itomcBreak.png"), tr("&edit condition"), this, SLOT(menuEditBreakpoint()));
    breakpointMenuActions["nextBP"] = breakpointMenu->addAction(tr("&next breakpoint"), this, SLOT(menuGotoNextBreakPoint()));
    breakpointMenuActions["prevBP"] = breakpointMenu->addAction(tr("&previous breakpoint"), this, SLOT(menuGotoPreviousBreakPoint()));
    breakpointMenuActions["clearALLBP"] = breakpointMenu->addAction(tr("&clear all breakpoint"), this, SLOT(menuClearAllBreakpoints()));

    connect(breakpointMenu, SIGNAL(aboutToShow()), this, SLOT(preShowContextMenuMargin()));

    editorMenu = new QMenu(this);
    editorMenuActions["cut"] = editorMenu->addAction(QIcon(":/editor/icons/editCut.png"), tr("&cut"), this, SLOT(menuCut()));
    editorMenuActions["copy"] = editorMenu->addAction(QIcon(":/editor/icons/editCopy.png"), tr("cop&y"), this, SLOT(menuCopy()));
    editorMenuActions["paste"] = editorMenu->addAction(QIcon(":/editor/icons/editPaste.png"), tr("&paste"), this, SLOT(menuPaste()));
    editorMenu->addSeparator();
    editorMenuActions["indent"] = editorMenu->addAction(QIcon(":/editor/icons/editIndent.png"), tr("&indent"), this, SLOT(menuIndent()));
    editorMenuActions["unindent"] = editorMenu->addAction(QIcon(":/editor/icons/editUnindent.png"), tr("&unindent"), this, SLOT(menuUnindent()));
    editorMenuActions["comment"] = editorMenu->addAction(QIcon(":/editor/icons/editComment.png"), tr("&comment"), this, SLOT(menuComment()));
    editorMenuActions["uncomment"] = editorMenu->addAction(QIcon(":/editor/icons/editUncomment.png"), tr("unc&omment"), this, SLOT(menuUncomment()));
    //editorMenu->addSeparator();
    //editorMenuActions["open"] = editorMenu->addAction(QIcon("icons/open.png"), tr("&open"), this, SLOT(menuOpen()));
    //editorMenuActions["save"] = editorMenu->addAction(QIcon("icons/fileSave.png"), tr("&save"), this, SLOT(menuSave()), tr("Ctrl+S"));
    //editorMenuActions["saveas"] = editorMenu->addAction(QIcon("icons/fileSaveAs.png"), tr("save &as"), this, SLOT(menuSaveAs()));
    editorMenu->addSeparator();
    editorMenuActions["runScript"] = editorMenu->addAction(QIcon(":/script/icons/runScript.png"), tr("&run script"), this, SLOT(menuRunScript()));
    editorMenuActions["runSelection"] = editorMenu->addAction(QIcon(":/script/icons/runScript.png"), tr("run &selection"), this, SLOT(menuRunSelection()));
    editorMenuActions["debugScript"] = editorMenu->addAction(QIcon(":/script/icons/debugScript.png"), tr("&debug script"), this, SLOT(menuDebugScript()));
    editorMenuActions["stopScript"] = editorMenu->addAction(QIcon(":/script/icons/stopScript.png"), tr("sto&p script"), this, SLOT(menuStopScript()));
    editorMenu->addSeparator();
    editorMenu->addAction(bookmarkMenuActions["toggleBM"]);
    editorMenu->addAction(bookmarkMenuActions["nextBM"]);
    editorMenu->addAction(bookmarkMenuActions["prevBM"]);
    editorMenu->addAction(bookmarkMenuActions["clearAllBM"]);

    //this->addAction(editorMenuActions["save"]);

    connect(editorMenu, SIGNAL(aboutToShow()), this, SLOT(preShowContextMenuEditor()));

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::contextMenuEvent (QContextMenuEvent * event)
{
    event->accept();

    int line, index;
    int lineFrom, indexFrom, lineTo, indexTo;

    long chpos = SendScintilla(SCI_POSITIONFROMPOINT, event->pos().x(), event->pos().y());
    lineIndexFromPosition(chpos, &line, &index);

    switch (getMarginNumber(event->x()))
    {
    case 1: //!< bookmarks
        contextMenuLine = line;
        bookmarkMenu->exec(event->globalPos());
        break;
    case 2: //!< line numbers
        break;
    case 3: //!< break points
        contextMenuLine = line;
        breakpointMenu->exec(event->globalPos());
        break;
    case 4: //!< folds
        //do nothing
        break;
    default:
        contextMenuLine = line;

        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

        if (lineFrom >= 0) //area is selected
        {

        }
        else //no selection
        {
            setCursorPosition(line, index);
        }

        editorMenu->exec(event->globalPos());
        break;
    }

    contextMenuLine = -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::preShowContextMenuEditor()
{
    const PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    int lineFrom, indexFrom, lineTo, indexTo;

    getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

    editorMenuActions["cut"]->setEnabled(lineFrom != -1);
    editorMenuActions["copy"]->setEnabled(lineFrom != -1);
    //editorMenuActions["iconBrowser"]->setEnabled(!pythonBusy);
    //editorMenuActions["save"]->setEnabled(isModified());
    editorMenuActions["paste"]->setEnabled(contextMenuLine >= 0);
    //editorMenuActions["save"]->setEnabled(isModified());

    editorMenuActions["runScript"]->setEnabled(!pythonBusy);
    editorMenuActions["runSelection"]->setEnabled(lineFrom != -1 && (!pythonBusy || pyEngine->isPythonDebuggingAndWaiting()));
    editorMenuActions["debugScript"]->setEnabled(!pythonBusy);
    editorMenuActions["stopScript"]->setEnabled(pythonBusy);

    bookmarkMenuActions["toggleBM"]->setEnabled(true);
    bookmarkMenuActions["nextBM"]->setEnabled(!bookmarkHandles.empty());
    bookmarkMenuActions["prevBM"]->setEnabled(!bookmarkHandles.empty());
    bookmarkMenuActions["clearAllBM"]->setEnabled(!bookmarkHandles.empty());

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::preShowContextMenuMargin()
{
    bookmarkMenuActions["toggleBM"]->setEnabled(true);
    bookmarkMenuActions["nextBM"]->setEnabled(!bookmarkHandles.empty());
    bookmarkMenuActions["prevBM"]->setEnabled(!bookmarkHandles.empty());
    bookmarkMenuActions["clearAllBM"]->setEnabled(!bookmarkHandles.empty());

    breakpointMenuActions["nextBP"]->setEnabled(!breakPointMap.empty());
    breakpointMenuActions["prevBP"]->setEnabled(!breakPointMap.empty());
    breakpointMenuActions["clearALLBP"]->setEnabled(!breakPointMap.empty());

    if (contextMenuLine >= 0 && getFilename()!="") //!< breakpoints only if filename != ""
    {
        if (markersAtLine(contextMenuLine) & markMaskBreakpoints)
        {
            breakpointMenuActions["toggleBP"]->setEnabled(true);
            breakpointMenuActions["toggleBPEnabled"]->setEnabled(true);
            breakpointMenuActions["editConditionBP"]->setEnabled(true);

            if (markersAtLine(contextMenuLine) & ((1 << markBreakPoint) | (1 << markCBreakPoint)))
            {
                breakpointMenuActions["toggleBPEnabled"]->setText(tr("&disable breakpoint"));
            }
            else
            {
                breakpointMenuActions["toggleBPEnabled"]->setText(tr("&enable breakpoint"));
            }
        }
        else
        {
            breakpointMenuActions["toggleBP"]->setEnabled(true);
            breakpointMenuActions["toggleBPEnabled"]->setEnabled(false);
            breakpointMenuActions["editConditionBP"]->setEnabled(false);
        }

    }
    else
    {
        breakpointMenuActions["toggleBP"]->setEnabled(false);
        breakpointMenuActions["toggleBPEnabled"]->setEnabled(false);
        breakpointMenuActions["editConditionBP"]->setEnabled(false);
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool ScriptEditorWidget::canInsertFromMimeData(const QMimeData *source) const
{
    //test = source->has
    //qDebug() << "MIME-Type:" << source->formats();
//    qDebug() << "URL:" << source->urls();
    if (source->hasText() == false && (source->hasFormat("FileName") || source->hasFormat("text/uri-list")))
    {
        if (source->urls().length() == 1)
        {
            QString fext = QFileInfo(source->urls().at(0).toString()).suffix().toLower();
//            qDebug() << fext.toLatin1().data();
            if ((fext == "txt") || (fext == "py") || (fext == "c") || (fext == "cpp")
                || (fext == "h") || (fext == "hpp") || (fext == "cxx") || (fext == "hxx"))
                return 1;
        }
    }
    else
    {
        return AbstractPyScintillaWidget::canInsertFromMimeData(source);
    }

    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::dropEvent(QDropEvent *event)
{
//    qDebug() << "MIME-Type2:" << event->mimeData()->formats();
    QObject *sew = AppManagement::getScriptEditorOrganizer();

    if (sew != NULL)
    {
        if (event->mimeData()->hasText() == false && (event->mimeData()->hasFormat("FileName") || event->mimeData()->hasFormat("text/uri-list")))
        {
            if (event->mimeData()->urls().length() == 1)
            {
                QString fext = QFileInfo(event->mimeData()->urls().at(0).toString()).suffix().toLower();
    //            qDebug() << fext.toLatin1().data();
                if ((fext == "txt") || (fext == "py") || (fext == "c") || (fext == "cpp")
                    || (fext == "h") || (fext == "hpp") || (fext == "cxx") || (fext == "hxx"))
                    QMetaObject::invokeMethod(sew, "openScript", Q_ARG(QString, event->mimeData()->urls().at(0).toLocalFile()), Q_ARG(ItomSharedSemaphore*, NULL));
            }
        }
        else
        {
            AbstractPyScintillaWidget::dropEvent(event);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
int ScriptEditorWidget::getMarginNumber(int xPos)
{
    int tempWidth = 0;
    int nr = 1;
    while (nr <= 4)
    {
        tempWidth += marginWidth(nr);
        if (xPos <= tempWidth)
        {
            return nr;
        }
    nr++;
    }
    return -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::copyAvailable(bool yes)
{
    canCopy = yes;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::setCursorPosAndEnsureVisible(int line)
{
    setCursorPosition(line, 0);
    ensureLineVisible(line);
    ensureCursorVisible();
    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuToggleBookmark()
{
    if (contextMenuLine>=0)
    {
        toggleBookmark(contextMenuLine);
    }
    else
    {
        int line, index;
        getCursorPosition(&line, &index);
        toggleBookmark(line);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuClearAllBookmarks()
{
    clearAllBookmarks();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuGotoNextBookmark()
{
    gotoNextBookmark();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuGotoPreviousBookmark()
{
    gotoPreviousBookmark();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuToggleBreakpoint()
{
   if (contextMenuLine>=0)
    {
        toggleBreakpoint(contextMenuLine);
    }
    else
    {
        int line, index;
        getCursorPosition(&line, &index);
        toggleBreakpoint(line);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuToggleEnableBreakpoint()
{
    if (contextMenuLine>=0)
    {
        toggleEnableBreakpoint(contextMenuLine);
    }
    else
    {
        int line, index;
        getCursorPosition(&line, &index);
        toggleEnableBreakpoint(line);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuEditBreakpoint()
{
    if (contextMenuLine>=0)
    {
        editBreakpoint(contextMenuLine);
    }
    else
    {
        int line, index;
        getCursorPosition(&line, &index);
        editBreakpoint(line);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuClearAllBreakpoints()
{
    clearAllBreakpoints();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuGotoNextBreakPoint()
{
    gotoNextBreakPoint();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuGotoPreviousBreakPoint()
{
    gotoPreviousBreakPoint();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuCut()
{
    cut();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuCopy()
{
    copy();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuPaste()
{
    paste();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuIndent()
{
    if (isReadOnly() == false)
    {
        int lineFrom, lineTo, indexFrom, indexTo;

        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
        if (lineFrom < 0)
        {
            getCursorPosition(&lineFrom, &indexFrom);
            lineTo = lineFrom;
//            indexTo = indexFrom;
        }

        for (int i = lineFrom; i <= lineTo; i++)
        {
            indent(i);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuUnindent()
{
    if (isReadOnly() == false)
    {
        int lineFrom, lineTo, indexFrom, indexTo;

        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
        if (lineFrom < 0)
        {
            getCursorPosition(&lineFrom, &indexFrom);
            lineTo = lineFrom;
//            indexTo = indexFrom;
        }

        for (int i = lineFrom; i <= lineTo; i++)
        {
            unindent(i);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuComment()
{
    if (isReadOnly() == false)
    {
        int lineFrom, lineTo, indexFrom, indexTo;
        QString lineText;
        QString lineTextTrimmed;
        int searchIndex;

        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
        if (lineFrom < 0)
        {
            getCursorPosition(&lineFrom, &indexFrom);
            lineTo = lineFrom;
            indexTo = indexFrom;
        }

        for (int i = lineFrom; i <= lineTo; i++)
        {
            lineText = text(i);
            lineTextTrimmed = lineText.trimmed();

            searchIndex = lineText.indexOf(lineTextTrimmed);
            if (searchIndex >= 0)
            {
                insertAt(QString("#"), i, searchIndex);
                if (i == lineFrom)
                {
                    indexFrom++;
                }
                if (i == lineTo)
                {
                    indexTo++;
                }
            }
        }

        if (lineFrom != lineTo || indexFrom != indexTo)
        {
            setSelection(lineFrom, indexFrom, lineTo, indexTo);
        }
        else
        {
            setCursorPosition(lineTo, indexTo);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuUncomment()
{
    if (isReadOnly() == false)
    {
        int lineFrom, lineTo, indexFrom, indexTo;
        QString lineText;
        int searchIndex;
        QString lineTextTrimmed;

        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
        if (lineFrom < 0)
        {
            getCursorPosition(&lineFrom, &indexFrom);
            lineTo = lineFrom;
            indexTo = indexFrom;
        }

        for (int i = lineFrom; i <= lineTo; i++)
        {
            lineText = text(i);
            lineTextTrimmed = lineText.trimmed();

            if (lineTextTrimmed.left(1) == "#")
            {
                searchIndex = lineText.indexOf("#");
                if (searchIndex >= 0)
                {
                    setSelection(i, searchIndex, i, searchIndex + 1);
                    removeSelectedText();
                }
            }
        }

        if (lineFrom != indexFrom || lineTo != indexTo)
        {
            setSelection(lineFrom, indexFrom, lineTo, indexTo);
        }
        else
        {
            setCursorPosition(lineFrom, indexFrom);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuRunScript()
{
    RetVal retValue(retOk);

    retValue += saveFile(false);

    if (!retValue.containsError())
    {
        //retValue += checkSaveStateForExecution();

        if (!retValue.containsError())
        {
            emit pythonRunFile(getFilename());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuRunSelection()
{
    int lineFrom = -1;
    int lineTo = -1;
    int indexFrom = -1;
    int indexTo = -1;

    //check whether text has been marked
    getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
    if (lineFrom >= 0)
    {
        //text has been marked
        if (lineFrom != lineTo)
        {
            indexFrom = 0;
            if (lineTo == lines() - 1)
            {
                indexTo = lineLength(lineTo);
            }
            else
            {
                indexTo = lineLength(lineTo) - 1;
            }

            setSelection(lineFrom, indexFrom, lineTo, indexTo);
        }
        QString defaultText = selectedText();

        emit pythonRunSelection(defaultText);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuDebugScript()
{
    RetVal retValue(retOk);

    if (getFilename() == "")
    {
        retValue += saveFile(true);
    }

    if (!retValue.containsError())
    {
        //retValue += checkSaveStateForExecution();

        if (!retValue.containsError())
        {
            emit pythonDebugFile(getFilename());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::menuStopScript()
{
    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

    if (eng != NULL)
    {
        if (eng->isPythonDebugging() && eng->isPythonDebuggingAndWaiting())
        {
            QMetaObject::invokeMethod(eng, "pythonDebugCommand", Q_ARG(tPythonDbgCmd, ito::pyDbgQuit));
        }
        else
        {
            eng->pythonInterruptExecution();
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::openFile(QString fileName, bool ignorePresentDocument)
{
    //!< check for modifications in the present document first
    if (!ignorePresentDocument)
    {
        if (isModified())
        {
            int ret = QMessageBox::information(this, tr("unsaved changes"), tr("there are unsaved changes in the current document. Do you want to save it first?"), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, QMessageBox::Yes);

            if (ret & QMessageBox::Cancel)
            {
                return RetVal(retOk);
            }
            else if (ret & QMessageBox::Yes)
            {
                saveFile(false);
                setModified(false);
            }
        }
    }

    QFile file(fileName);
    if (! file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("error while opening file"), tr("file %1 could not be loaded").arg(fileName));
    }

    QString text(file.readAll());
    file.close();

    clearAllBookmarks();
    clearAllBreakpoints();
    setText(text);

    this->filename = fileName;

    QStringList watchedFiles = m_pFileSysWatcher->files();
    if (watchedFiles.size() > 0)
    {
        m_pFileSysWatcher->removePaths(watchedFiles);
    }
    m_pFileSysWatcher->addPath(this->filename);

    //!< check if BreakPointModel already contains breakpoints for this editor and load them
    if (getFilename() != "")
    {
        BreakPointModel *bpModel = PythonEngine::getInstance() ? PythonEngine::getInstance()->getBreakPointModel() : NULL;
        if (bpModel)
        {
            QModelIndexList modelIndexList = bpModel->getBreakPointIndizes(getFilename());
            QList<BreakPointItem> bpItems = bpModel->getBreakPoints(modelIndexList);

            for (int i=0; i<bpItems.size(); i++)
            {
                breakPointAdd(bpItems.at(i), i);
            }
        }
    }

    setModified(false);

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::saveFile(bool askFirst)
{
    if (!isModified())
    {
        return RetVal(retOk);
    }
    if (this->getFilename().isNull())
    {
        return saveAsFile(askFirst);
    }

    if (askFirst)
    {
        int ret = QMessageBox::information(this, tr("unsaved changes"), tr("there are unsaved changes in the document '%1'. Do you want to save it first?").arg(getFilename()), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, QMessageBox::Yes);
        if (ret & QMessageBox::Cancel)
        {
            return RetVal(retError);
        }
        else if (ret & QMessageBox::No)
        {
            return RetVal(retOk);
        }
    }

    m_pFileSysWatcher->removePath(getFilename());

    QFile file(getFilename());
    if (! file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("error while accessing file"), tr("file %1 could not be accessed").arg(getFilename()));
        return RetVal(retError);
    }

    convertEols(QsciScintilla::EolUnix);

    file.write(text().toLatin1());
    file.close();

    setModified(false);

    m_pFileSysWatcher->addPath(getFilename());

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::saveAsFile(bool askFirst)
{
    if (askFirst)
    {
        int ret = QMessageBox::information(this, tr("unsaved changes"), tr("there are unsaved changes in the current document. Do you want to save it first?"), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, QMessageBox::Yes);
        if (ret & QMessageBox::Cancel)
        {
            return RetVal(retError);
        }
        else if (ret & QMessageBox::No)
        {
            return RetVal(retOk);
        }
    }

    QString defaultPath = QDir::currentPath();
    QFile file;

    QString tempFileName = QFileDialog::getSaveFileName(this, tr("save as..."), defaultPath, "Python (*.py)");
    if (!tempFileName.isEmpty())
    {
        QDir::setCurrent(QFileInfo(tempFileName).path());
        file.setFileName(tempFileName);
    }
    else
    {
        return RetVal(retError);
    }

    if (! file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QMessageBox::warning(this, tr("error while accessing file"), tr("file %1 could not be accessed").arg(getFilename()));
        return RetVal(retError);
    }

    m_pFileSysWatcher->removePath(getFilename());

    convertEols(QsciScintilla::EolUnix);
    file.write(text().toLatin1());
    file.close();

    changeFilename(tempFileName);
    setModified(false);

    m_pFileSysWatcher->addPath(tempFileName);

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< bookmark handling
RetVal ScriptEditorWidget::toggleBookmark(int line)
{

    //!< markerLine(handle) returns -1, if marker doesn't exist any more (because lines have been deleted...)
    std::list<int>::iterator it;
    bool found = false;

    for (it=bookmarkHandles.begin(); it != bookmarkHandles.end() && !found; ++it)
    {
        if (markerLine(*it) == line)
        {
            markerDeleteHandle(*it);
            *it = -1; //!< in order to mark it for removal
            found = true;
        }
    }

    bookmarkHandles.remove(-1);

    if (!found)
    {
        bookmarkHandles.push_back(markerAdd(line, markBookmark));
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::clearAllBookmarks()
{
    markerDeleteAll(markBookmark);
    bookmarkHandles.clear();

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::gotoNextBookmark()
{
    int line, index;
    int bookmarkLine;
    getCursorPosition(&line, &index);

    line += 1;

    if (line == lines())
    {
        line = 0;
    }

    bookmarkLine = markerFindNext(line, 1 << markBookmark);
    if (bookmarkLine < 0)
    {
        bookmarkLine = markerFindNext(0, 1 << markBookmark);
    }

    if (bookmarkLine >= 0)
    {
        setCursorPosAndEnsureVisible(bookmarkLine);
        return RetVal(retOk);
    }

    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::gotoPreviousBookmark()
{
    int line, index;
    int bookmarkLine;
    getCursorPosition(&line, &index);

    if (line == 0)
    {
        line = lines()-1;
    }
    else
    {
        line -= 1;
    }

    bookmarkLine = markerFindPrevious(line, 1 << markBookmark);
    if (bookmarkLine < 0)
    {
        bookmarkLine = markerFindPrevious(lines() - 1, 1 << markBookmark);
    }

    if (bookmarkLine >= 0)
    {
        setCursorPosAndEnsureVisible(bookmarkLine);
        return RetVal(retOk);
    }
    return RetVal(retError);

}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::toggleBreakpoint(int line)
{
    if (getFilename() == "") return RetVal(retError);

    //!< markerLine(handle) returns -1, if marker doesn't exist any more (because lines have been deleted...)
//    std::list<QPair<int, int> >::iterator it;
    BreakPointModel *bpModel = PythonEngine::getInstance()->getBreakPointModel();
    QModelIndexList indexList = bpModel->getBreakPointIndizes(getFilename(), line);

    if (indexList.size()>0)
    {
        bpModel->deleteBreakPoints(indexList);
    }
    else
    {
        BreakPointItem bp;
        bp.filename = getFilename();
        bp.lineno = line;
        bp.conditioned = false;
        bp.condition = "";
        bp.enabled = true;
        bp.temporary = false;
        bp.ignoreCount = 0;
        bpModel->addBreakPoint(bp);
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::toggleEnableBreakpoint(int line)
{
    if (getFilename() == "") return RetVal(retError);

    BreakPointModel *bpModel = PythonEngine::getInstance()->getBreakPointModel();
    QModelIndexList indexList = bpModel->getBreakPointIndizes(getFilename() , line);
    BreakPointItem item;

    if (indexList.size()>0)
    {
        for (int i = 0; i < indexList.size(); i++)
        {
            item = bpModel->getBreakPoint(indexList.at(i));
            item.enabled = !item.enabled;
            bpModel->changeBreakPoint(indexList.at(i), item);
        }
        return RetVal(retOk);
    }

    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::editBreakpoint(int line)
{
    if (getFilename() == "") return RetVal(retError);

    BreakPointModel *bpModel = PythonEngine::getInstance()->getBreakPointModel();
    QModelIndex index;
    BreakPointItem item;
    RetVal retValue(retOk);

    if (markersAtLine(line) & markMaskBreakpoints)
    {
        index = bpModel->getFirstBreakPointIndex(getFilename(), line);

        if (index.isValid())
        {
            item = bpModel->getBreakPoint(index);

            DialogEditBreakpoint *dlg = new DialogEditBreakpoint(item.filename, line+1, item.enabled, item.temporary , item.ignoreCount, item.condition);
            dlg->exec();
            if (dlg->result() == QDialog::Accepted)
            {
                dlg->getData(item.enabled, item.temporary, item.ignoreCount, item.condition);
                item.conditioned = (item.condition != "") || (item.ignoreCount > 0) || item.temporary;

                bpModel->changeBreakPoint(index, item);
            }

            DELETE_AND_SET_NULL(dlg);

            return RetVal(retOk);
        }
    }

    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::clearAllBreakpoints()
{
    if (getFilename() == "") return RetVal(retError);

    BreakPointModel *bpModel = PythonEngine::getInstance() ? PythonEngine::getInstance()->getBreakPointModel() : NULL;

    if (bpModel)
    {
        bpModel->deleteBreakPoints(bpModel->getBreakPointIndizes(getFilename()));
    }

    //!< the following lines are not neccesary, since the delete-slot is invoked for each breakPoint by the BreakPointModel
    /*markerDeleteAll(markBreakPoint);
    markerDeleteAll(markCBreakPoint);
    markerDeleteAll(markCBreakPointDisabled);
    markerDeleteAll(markBreakPointDisabled);
    breakPointMap.clear();*/

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::gotoNextBreakPoint()
{
    int line, index;
    int breakPointLine;
    getCursorPosition(&line, &index);

    line += 1;

    if (line == lines())
    {
        line = 0;
    }

    breakPointLine = markerFindNext(line, markMaskBreakpoints);
    if (breakPointLine < 0)
    {
        breakPointLine = markerFindNext(0, markMaskBreakpoints);
    }

    if (breakPointLine >= 0)
    {
        setCursorPosAndEnsureVisible(breakPointLine);
        return RetVal(retOk);
    }

    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::gotoPreviousBreakPoint()
{
    int line, index;
    int breakPointLine;
    getCursorPosition(&line, &index);

    if (line == 0)
    {
        line = lines()-1;
    }
    else
    {
        line -= 1;
    }

    breakPointLine = markerFindPrevious(line, markMaskBreakpoints);
    if (breakPointLine < 0)
    {
        breakPointLine = markerFindPrevious(lines() - 1, markMaskBreakpoints);
    }

    if (breakPointLine >= 0)
    {
        setCursorPosAndEnsureVisible(breakPointLine);
        return RetVal(retOk);
    }
    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::marginClicked(int margin, int line, Qt::KeyboardModifiers /*state*/)
{
    if (margin == 1) //!< bookmarks
    {
        toggleBookmark(line);
    }
    else if (margin == 3) //!< set or remove breakpoint (standard form)
    {
        toggleBreakpoint(line);
    }
    emit marginChanged();
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< slot, invoked by BreakPointModel
void ScriptEditorWidget::breakPointAdd(BreakPointItem bp, int /*row*/)
{
    int newHandle = -1;

    if (bp.filename == getFilename() && bp.filename != "")
    {
                std::list<QPair<int, int> >::iterator it;
        bool found = false;

        for (it = breakPointMap.begin(); it != breakPointMap.end() && !found; ++it)
        {
            if (it->second == bp.lineno)
            {
                found = true;
            }
        }

        if (found) return; //!< there is already a breakpoint in this line, do not add the new one


        if (bp.enabled)
        {
            if (bp.conditioned)
            {
                newHandle = markerAdd(bp.lineno, markCBreakPoint);
            }
            else
            {
                newHandle = markerAdd(bp.lineno, markBreakPoint);
            }
        }
        else
        {
            if (bp.conditioned)
            {
                newHandle = markerAdd(bp.lineno, markCBreakPointDisabled);
            }
            else
            {
                newHandle = markerAdd(bp.lineno, markBreakPointDisabled);
            }
        }
        breakPointMap.push_back(QPair<int, int>(newHandle, bp.lineno));

    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< slot, invoked by BreakPointModel
void ScriptEditorWidget::breakPointDelete(QString filename, int lineNo, int /*pyBpNumber*/)
{
    if (filename == getFilename() && filename != "")
    {
                std::list<QPair<int, int> >::iterator it;

        it=breakPointMap.begin();

        while(it != breakPointMap.end())
        {
            if (it->second == lineNo)
            {
                markerDeleteHandle(it->first);
                it = breakPointMap.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< slot, invoked by BreakPointModel
void ScriptEditorWidget::breakPointChange(BreakPointItem oldBp, BreakPointItem newBp)
{
    if (oldBp.filename == getFilename())
    {
        breakPointDelete(oldBp.filename, oldBp.lineno, oldBp.pythonDbgBpNumber);
    }
    if (newBp.filename == getFilename())
    {
        breakPointAdd(newBp, -1); //!< -1 has no task
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::print()
{
    if (lines() == 0 || text() == "")
    {
        QMessageBox::warning(this, tr("Print"), tr("There is nothing to print"));
    }
    else
    {
        ItomQsciPrinter printer(QPrinter::HighResolution);
        printer.setWrapMode(WrapWord);

        if (hasNoFilename() == false)
        {
            printer.setDocName( getFilename() );
        }
        else
        {
            printer.setDocName( tr("unnamed") );
        }

        printer.setPageMargins(20,15,20,15,QPrinter::Millimeter);
        printer.setMagnification(-1); //size one point smaller than the one displayed in itom.
        QPrintPreviewDialog printPreviewDialog(&printer);
        printPreviewDialog.setWindowFlags(Qt::Window);
        connect(&printPreviewDialog, SIGNAL(paintRequested(QPrinter*)), this, SLOT(printPreviewRequested(QPrinter*)));
        printPreviewDialog.exec();

    //    QPrintDialog printDialog(&printer);
    ////    QPrintPreviewDialog printDialog(&printer);
    //    if (printDialog.exec())
    //    {
    //        printer.
    //        printer.printRange(this);
    //    }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::printPreviewRequested(QPrinter *printer)
{
    ItomQsciPrinter *p = static_cast<ItomQsciPrinter*>(printer);
    if (p)
    {
        p->printRange(this);
    }
}

    //def printPreviewFile(self):
    //    """
    //    Public slot to show a print preview of the text.
    //    """
    //    from PyQt4.QtGui import QPrintPreviewDialog
    //    
    //    printer = Printer(mode=QPrinter.HighResolution)
    //    fn = self.getFileName()
    //    if fn is not None:
    //        printer.setDocName(os.path.basename(fn))
    //    else:
    //        printer.setDocName(self.noName)
    //    preview = QPrintPreviewDialog(printer, self)
    //    preview.paintRequested.connect(self.__printPreview)
    //    preview.exec_()
    //

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::changeFilename(QString newFilename)
{
    BreakPointModel *bpModel = PythonEngine::getInstance() ? PythonEngine::getInstance()->getBreakPointModel() : NULL;
    QModelIndexList modelIndexList;

    if (newFilename == "" || newFilename.isNull())
    {
        if (bpModel)
        {
            modelIndexList = bpModel->getBreakPointIndizes(getFilename());
            bpModel->deleteBreakPoints(modelIndexList);
        }
        filename = QString();
    }
    else
    {
        if (bpModel)
        {
            modelIndexList = bpModel->getBreakPointIndizes(getFilename());
            QList<BreakPointItem> lists = bpModel->getBreakPoints(modelIndexList);
            BreakPointItem temp;
            QList<BreakPointItem> newList;
            for (int i = 0; i < lists.size(); i++)
            {
                temp = lists.at(i);
                temp.filename = newFilename;
                newList.push_back(temp);
            }
            bpModel->changeBreakPoints(modelIndexList, newList, false);
        }
        filename = newFilename;
    }
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::nrOfLinesChanged()
{
    std::list<QPair<int, int> >::iterator it;
    int line;
    BreakPointModel *bpModel = PythonEngine::getInstance() ? PythonEngine::getInstance()->getBreakPointModel() : NULL;
    BreakPointItem item;
    QModelIndex index;

    it = breakPointMap.begin();

    while(it != breakPointMap.end())
    {
        line = markerLine(it->first);

        if (line == -1)
        {
            //!< marker has been deleted:
            index = bpModel->getFirstBreakPointIndex(getFilename(), it->second);
            if (index.isValid())
            {
                bpModel->deleteBreakPoint(index);
            }
            it = breakPointMap.erase(it);
        }
        else if (line != it->second)
        {
            index = bpModel->getFirstBreakPointIndex(getFilename(), it->second);

            item = bpModel->getBreakPoint(index);
            item.lineno = line;
            bpModel->changeBreakPoint(index, item, false);
            it->second = line;

            ++it;
        }
        else
        {
            ++it;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::pythonDebugPositionChanged(QString filename, int lineno)
{
    if (!hasNoFilename() && (QFileInfo(filename) == QFileInfo(getFilename())))
    {
        if (markCurrentLineHandle != -1)
        {
            markerDeleteHandle(markCurrentLineHandle);
        }
        markCurrentLineHandle = markerAdd(lineno-1, markCurrentLine);
        ensureLineVisible(lineno-1);
        raise();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//void ScriptEditorWidget::pythonCodeExecContinued()
//{
//    if (markCurrentLineHandle != -1)
//    {
//        markerDeleteHandle(markCurrentLineHandle);
//    }
//}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::pythonStateChanged(tPythonTransitions pyTransition)
{
    switch(pyTransition)
    {
    case pyTransBeginRun:
    case pyTransBeginDebug:
        if (!hasNoFilename()) setReadOnly(true);
        pythonBusy = true;
        break;
    case pyTransDebugContinue:
        if (markCurrentLineHandle != -1)
        {
            markerDeleteHandle(markCurrentLineHandle);
        }
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
        setReadOnly(false);
        if (markCurrentLineHandle != -1)
        {
            markerDeleteHandle(markCurrentLineHandle);
        }
        pythonBusy = false;
        break;
    case pyTransDebugWaiting:
    case pyTransDebugExecCmdBegin:
    case pyTransDebugExecCmdEnd:

        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::fileSysWatcherFileChanged(const QString & path) //this signal may be emitted multiple times at once for the same file, therefore the mutex protection is introduced
{
    if (fileSystemWatcherMutex.tryLock(1))
    {
        QMessageBox msgBox(this);
        msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);

        if (path == getFilename())
        {
            QFile file(path);

            if (!file.exists()) //file deleted
            {
                msgBox.setText(tr("The file '%1' does not exist any more.").arg(path));
                msgBox.setInformativeText(tr("Keep this file in editor?"));

                int ret = msgBox.exec();

                if (ret == QMessageBox::No)
                {
                    emit closeRequest(this, true);
                }
                else
                {
                    insertAt("a", 0, 0); //workaround in order to set the modified-flag of QScintilla to TRUE (can not be done manually)
                    setSelection(0, 0, 0, 1);
                    removeSelectedText();
                }
            }
            else //file changed
            {
                msgBox.setText(tr("The file '%1' has been modified by another programm.").arg(path));
                msgBox.setInformativeText(tr("Do you want to reload it?"));
                int ret = msgBox.exec();

                if (ret == QMessageBox::Yes)
                {
                    openFile(path, true);
                }
                else
                {
                    insertAt("a", 0, 0); //workaround in order to set the modified-flag of QScintilla to TRUE (can not be done manually)
                    setSelection(0, 0, 0, 1);
                    removeSelectedText();
                }
            }
        }

        fileSystemWatcherMutex.unlock();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//void ScriptEditorWidget::keyPressEvent (QKeyEvent *event)
//{
//    int key = event->key();
//    Qt::KeyboardModifiers modifiers = event->modifiers();
//    bool acceptEvent = true;
//    bool forwardEvent = true;
//
//    if (key != Qt::Key_Control && (modifiers & Qt::ControlModifier))
//    {
//        if (key == Qt::Key_T)
//        {
//            acceptEvent = false;
//        }
//        else if (key == Qt::Key_R)
//        {
//            forwardEvent = false;
//        }
//    }
//
//    if (acceptEvent && forwardEvent)
//    {
//        QsciScintilla::keyPressEvent(event);
//    }
//    else if (!acceptEvent)
//    {
//        event->ignore();
//    }
//    else if (acceptEvent && !forwardEvent)
//    {
//        event->accept();
//    }
//
//}

//----------------------------------------------------------------------------------------------------------------------------------
void ItomQsciPrinter::formatPage( QPainter &painter, bool drawing, QRect &area, int pagenr )
{
    QString filename = this->docName();
    QString date = QDateTime::currentDateTime().toString(Qt::LocalDate);
    QString page = QString::number(pagenr);
    int width = area.width();
    int dateWidth = painter.fontMetrics().width(date);
    filename = painter.fontMetrics().elidedText( filename, Qt::ElideMiddle, 0.8 * (width - dateWidth) );
        
    painter.save();
    painter.setFont( QFont("Helvetica", 10, QFont::Normal, false) );
    painter.setPen(QColor(Qt::black)); 
    if (drawing)
    {
        //painter.drawText(area.right() - painter.fontMetrics().width(header), area.top() + painter.fontMetrics().ascent(), header);
        painter.drawText(area.left() - 25, area.top() + painter.fontMetrics().ascent(), filename);
        painter.drawText(area.right() + 25 - painter.fontMetrics().width(date), area.top() + painter.fontMetrics().ascent(), date);
        painter.drawText((area.left() + area.right())*0.5, area.bottom() - painter.fontMetrics().ascent(), page);
    }
    area.setTop(area.top() + painter.fontMetrics().height() + 30);
    area.setBottom(area.bottom() - painter.fontMetrics().height() - 50);
    painter.restore();
}

} // end namespace ito
