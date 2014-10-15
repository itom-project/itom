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
#include "qpair.h"

#include "../global.h"
#include "../Qitom/AppManagement.h"

#include <qfileinfo.h>
#include "../ui/dialogEditBreakpoint.h"

#include <Qsci/qsciprinter.h>
#include <qmessagebox.h>
#if QT_VERSION >= 0x050000
#include <QtPrintSupport/qprintpreviewdialog.h>
#else
#include <qprintpreviewdialog.h>
#endif
#include <qtooltip.h>
#include <qtimer.h>
#include <qpainter.h>
#include <qmimedata.h>

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
    m_filename(QString()),
    unnamedNumber(ScriptEditorWidget::unnamedAutoIncrement++),
    pythonBusy(false), 
    m_pythonExecutable(true),
    canCopy(false),
    m_syntaxTimer(NULL),
    m_classNavigatorTimer(NULL)
{
    bookmarkErrorHandles.clear();
    bookmarkMenuActions.clear();

    m_syntaxTimer = new QTimer(this);
    connect(m_syntaxTimer, SIGNAL(timeout()), this, SLOT(updateSyntaxCheck()));
    m_syntaxTimer->setInterval(1000);

    m_classNavigatorTimer = new QTimer(this);
    connect(m_classNavigatorTimer, SIGNAL(timeout()), this, SLOT(classNavTimerElapsed()));
    m_classNavigatorTimer->setInterval(2000);

    initEditor();

    initMenus();

    m_pFileSysWatcher = new QFileSystemWatcher(this);
    connect(m_pFileSysWatcher, SIGNAL(fileChanged(const QString&)), this, SLOT(fileSysWatcherFileChanged(const QString&)));

    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
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

    setMarginWidth(3, 18);
    setMarginWidth(4, 18);

    setMarginSensitivity(1, true);
    setMarginSensitivity(3, true);
    autoAdaptLineNumberColumnWidth();
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
    markBookmarkSyntaxError = markerDefine(QPixmap(":/script/icons/bookmarkSyntaxError.png"));

    markCurrentLine = markerDefine(QPixmap(":/script/icons/currentLine.png"));
    markCurrentLineHandle = -1;

    markMaskBreakpoints = (1 << markBreakPoint) | (1 << markCBreakPoint)  | (1 << markBreakPointDisabled)  | (1 << markCBreakPointDisabled) | (1 << markCurrentLine);
    markMask1 = markMaskBreakpoints;
    markMask2 = (1 << markBookmark) | (1 << markSyntaxError) | (1 << markBookmarkSyntaxError);

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

    // SyntaxChecker
    m_syntaxCheckerEnabled = settings.value("syntaxChecker", true).toBool();
    m_syntaxCheckerInterval = (int)(settings.value("syntaxInterval", 1).toDouble()*1000);
    m_syntaxTimer->stop();
    m_syntaxTimer->setInterval(m_syntaxCheckerInterval);
    if (m_syntaxCheckerEnabled)
    { // empty call: all bugs disappear
        checkSyntax();
    }
    else
    {
        errorListChange(QStringList());
    }

    // Class Navigator
    m_ClassNavigatorEnabled = settings.value("classNavigator", true).toBool();

    m_classNavigatorTimerEnabled = settings.value("classNavigatorTimerActive", true).toBool();
    m_classNavigatorInterval = (int)(settings.value("classNavigatorInterval", 2.00).toDouble()*1000);
    m_classNavigatorTimer->stop();
    m_classNavigatorTimer->setInterval(m_classNavigatorInterval);

    settings.endGroup();

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
    breakpointMenuActions["toggleBPEnabled"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/itomBreakDisable.png"), tr("&disable breakpoint"), this, SLOT(menuToggleEnableBreakpoint()));
    breakpointMenuActions["editConditionBP"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/itomcBreak.png"), tr("&edit condition"), this, SLOT(menuEditBreakpoint()));
    breakpointMenuActions["nextBP"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/breakpointNext.png"), tr("&next breakpoint"), this, SLOT(menuGotoNextBreakPoint()));
    breakpointMenuActions["prevBP"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/breakpointPrevious.png"),tr("&previous breakpoint"), this, SLOT(menuGotoPreviousBreakPoint()));
    breakpointMenuActions["clearALLBP"] = breakpointMenu->addAction(QIcon(":/breakpoints/icons/garbageAllBPs.png"), tr("&delete all breakpoints"), this, SLOT(menuClearAllBreakpoints()));

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
const ScriptEditorStorage ScriptEditorWidget::saveState() const
{
    ScriptEditorStorage storage;
    storage.filename = getFilename().toLatin1();
    storage.firstVisibleLine = firstVisibleLine();

    foreach(const BookmarkErrorEntry &e, bookmarkErrorHandles)
    {
        if (e.type & markerBookmark)
        {
            storage.bookmarkLines << markerLine(e.handle);
        }
    }

    return storage;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::autoAdaptLineNumberColumnWidth()
{
    int l = lines();
    QString s; //make the width always a little bit bigger than necessary

    if (l < 10)
    {
        s = QString::number(10);
    }
    else if (l < 100)
    {
        s = QString::number(100);
    }
    else if (l < 1000)
    {
        s = QString::number(1000);
    }
    else if (l < 10000)
    {
        s = QString::number(10000);
    }
    else
    {
        s = QString::number(100000);
    }

    setMarginWidth(2, s);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::restoreState(const ScriptEditorStorage &data)
{
    RetVal retVal = openFile(data.filename, true);

    if (!retVal.containsError())
    {
        setFirstVisibleLine(data.firstVisibleLine);

        clearAllBookmarks();
        foreach(const int &bookmarkLine, data.bookmarkLines)
        {
            toggleBookmark(bookmarkLine);
        }
    }
    
    return retVal;
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
    bookmarkMenuActions["nextBM"]->setEnabled(!bookmarkErrorHandles.empty());
    bookmarkMenuActions["prevBM"]->setEnabled(!bookmarkErrorHandles.empty());
    bookmarkMenuActions["clearAllBM"]->setEnabled(!bookmarkErrorHandles.empty());

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::preShowContextMenuMargin()
{
    bookmarkMenuActions["toggleBM"]->setEnabled(true);
    bookmarkMenuActions["nextBM"]->setEnabled(!bookmarkErrorHandles.empty());
    bookmarkMenuActions["prevBM"]->setEnabled(!bookmarkErrorHandles.empty());
    bookmarkMenuActions["clearAllBM"]->setEnabled(!bookmarkErrorHandles.empty());

    breakpointMenuActions["nextBP"]->setEnabled(!m_breakPointMap.empty());
    breakpointMenuActions["prevBP"]->setEnabled(!m_breakPointMap.empty());
    breakpointMenuActions["clearALLBP"]->setEnabled(!m_breakPointMap.empty());

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
    if ((source->hasFormat("FileName") || source->hasFormat("text/uri-list")))
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
        if ((event->mimeData()->hasFormat("FileName") || event->mimeData()->hasFormat("text/uri-list")))
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

            //this snipped is based on a QScintilla mailing list thread:
            //http://www.riverbankcomputing.com/pipermail/qscintilla/2014-September/000996.html
            if (event->source()->objectName() == "console")
            {
                //we never want to move text out of the console, text should always be copied
                if (event->dropAction() == Qt::MoveAction)
                {
                    event->setDropAction(Qt::CopyAction);
                    event->accept();
                }
            }
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
RetVal ScriptEditorWidget::setCursorPosAndEnsureVisibleWithSelection(int line, QString name)
{
    setCursorPosAndEnsureVisible(line);
    // regular expression for Classes and Methods
    QRegExp reg("(\\s*)(class||def)\\s(.+)\\(.*");
    reg.setMinimal(true);
    reg.indexIn(this->text(line), 0);
    this->setSelection(line, reg.pos(3), line, reg.pos(3) + reg.cap(3).length());
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
        QString defaultText = selectedText();

        //in linux, double-clicking at one line entirely marks this line and sometimes includes a \n to the next line. remove this:
        const QChar *data = defaultText.constData();
        int signsToRemove = 0;
        int len = defaultText.size() - 1;
        
        while (defaultText[len-signsToRemove] == '\n' || defaultText[len-signsToRemove] == '\r' || defaultText[len-signsToRemove] == ' ')
        {
            signsToRemove++;
        }

        defaultText.truncate(len - signsToRemove + 1);
        

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
            QMetaObject::invokeMethod(eng, "pythonDebugCommand", Q_ARG(tPythonDbgCmd, pyDbgQuit));
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
    else
    {

        QString text(file.readAll());
        file.close();

        clearAllBookmarks();
        clearAllBreakpoints();
        setText(text);

        changeFilename(fileName);

        QStringList watchedFiles = m_pFileSysWatcher->files();
        if (watchedFiles.size() > 0)
        {
            m_pFileSysWatcher->removePaths(watchedFiles);
        }
        m_pFileSysWatcher->addPath(m_filename);

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

        QObject *seo = AppManagement::getScriptEditorOrganizer();
        if (seo)
        {
            QMetaObject::invokeMethod(seo, "fileOpenedOrSaved", Q_ARG(QString, m_filename));
        }
    }

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

    QFileInfo fi(getFilename());
    if (fi.exists())
    {
        QObject *seo = AppManagement::getScriptEditorOrganizer();
        if (seo)
        {
            QMetaObject::invokeMethod(seo, "fileOpenedOrSaved", Q_ARG(QString, m_filename));
        }
    }

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

    QFileInfo fi(getFilename());
    if (fi.exists())
    {
        QObject *seo = AppManagement::getScriptEditorOrganizer();
        if (seo)
        {
            QMetaObject::invokeMethod(seo, "fileOpenedOrSaved", Q_ARG(QString, m_filename));
        }
    }

    setModified(false);

    m_pFileSysWatcher->addPath(tempFileName);

    return RetVal(retOk);
}


//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by pythonEnginge::pythonSyntaxCheck
/*!
    This function is automatically called to deliver the results of the syntax checker

    \sa checkSyntax
*/
void ScriptEditorWidget::syntaxCheckResult(QString a, QString b)
{ // this event occurs when the syntax checker is delivering results
    QStringList errorList = b.split("\n");
    for (int i = 0; i<errorList.length(); ++i)
    {
        errorList.removeAt(errorList.indexOf("",i));
        //errorList.at(i).
    }
    errorListChange(errorList);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Updates the List of Bookmarks and Errors when new Errorlist appears
/*!
    \param errorList Error list of this editor. Including all bugs and bookmarks.
*/
void ScriptEditorWidget::errorListChange(const QStringList &errorList)
{ 
    QList<BookmarkErrorEntry>::iterator it;
    it = bookmarkErrorHandles.begin();
    while (it != bookmarkErrorHandles.end())
    {
        if (it->type == markerPyBug)
        { // only Bug => Remove
            markerDeleteHandle(it->handle);
            it = bookmarkErrorHandles.erase(it);
        }
        else if (it->type == markerBookmarkAndPyBug)
        { // Bookmark and Bug => set to 1 and change Icon
            int line = markerLine(it->handle);
            markerDeleteHandle(it->handle);
            it->handle = markerAdd(line, markBookmark);
            it->type = markerBookmark;
            ++it;
        }
        else
        {
            ++it;
        }
    }

    for (int i = 0; i < errorList.length(); i++)
    {
        QRegExp regError(":(\\d+):(.*)");
        regError.indexIn(errorList.at(i),0);
        int line = regError.cap(1).toInt();
        bool found = false;
        it = bookmarkErrorHandles.begin();
        while (it != bookmarkErrorHandles.end())
        {
            if (line == markerLine(it->handle) + 1 && it->type == markerBookmark)
            { // this entry exists and is a bookmark, so make it 3 (BM & Err)
                markerDeleteHandle(it->handle);
                it->type = markerBookmarkAndPyBug;
                it->handle = markerAdd(line-1, markBookmarkSyntaxError);
                it->errorMessage = regError.cap(2);
                found = true;
            }
            ++it;
        }
        if (found == false)
        {
            BookmarkErrorEntry newE;
            newE.type = markerPyBug;
            newE.handle = markerAdd(line-1, markSyntaxError);
            newE.errorMessage = regError.cap(2);
            bookmarkErrorHandles.append(newE);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Sends the code to the Syntax Checker
/*!
    This function is called to send the content of this ScriptEditorWidget to the syntax checker

    \sa syntaxCheckResult
*/
void ScriptEditorWidget::checkSyntax()
{
    PythonEngine *pyEng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyEng->pySyntaxCheckAvailable())
    {
        QMetaObject::invokeMethod(pyEng, "pythonSyntaxCheck", Q_ARG(QString, this->text()), Q_ARG(QPointer<QObject>, QPointer<QObject>(this)));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by timer
/*!
    This slot is invoked by the timer to trigger the syntax check. The intervall is set in the option dialog.
    \sa syntaxCheckResult, checkSyntax
*/
void ScriptEditorWidget::updateSyntaxCheck()
{
    m_syntaxTimer->stop();
    checkSyntax();
}

//----------------------------------------------------------------------------------------------------------------------------------
bool ScriptEditorWidget::event (QEvent * event)
{ // This function is called when staying over an error icon to display the hint
    if (event->type() == QEvent::ToolTip)
    {
        //see http://www.riverbankcomputing.com/pipermail/qscintilla/2008-November/000381.html
        QHelpEvent *evt = static_cast<QHelpEvent*>(event);
        QPoint point = evt->pos();
        int sensAreaX = QsciScintilla::marginWidth(1);
        int posX = point.rx();
        int posY = point.ry();
        // Check that it´s in the right column (margin)
        if (posX <= sensAreaX)
        {
            QStringList texts;

            point.rx() = QsciScintilla::SendScintilla(QsciScintilla::SCI_POINTXFROMPOSITION, 0);
            int line = QsciScintilla::lineAt(point);
            point.rx() = posX;
            
            QList<BookmarkErrorEntry>::iterator it;
            it = bookmarkErrorHandles.begin();
            while (it != bookmarkErrorHandles.end())
            {
                int l = markerLine(it->handle);
                if (l == line && (it->type & markerBookmarkAndPyBug))
                {
                    texts << it->errorMessage;
                }
                ++it;
            }

            if (texts.size() > 0)
            {
                point = mapToGlobal(point);
                QToolTip::showText(point, texts.join("\n"), this);
            }
        }
    }
    else if(event->type() == QEvent::KeyRelease)
    {
        // SyntaxCheck   
        if (m_pythonExecutable && m_syntaxCheckerEnabled)
        {
            m_syntaxTimer->start(); //starts or restarts the timer
        }
        if(m_ClassNavigatorEnabled && m_classNavigatorTimerEnabled)
        {   // Class Navigator if Timer is active
            m_classNavigatorTimer->start();
        }
    }
    return QsciScintilla::event(event);
}


//----------------------------------------------------------------------------------------------------------------------------------
//!< bookmark handling
RetVal ScriptEditorWidget::toggleBookmark(int line)
{
    QList<BookmarkErrorEntry>::iterator it;
    bool createNew = true;

    it = bookmarkErrorHandles.begin();
    while (it != bookmarkErrorHandles.end())
    {
        if (markerLine(it->handle) == line)
        {
            // Delete old Handle
            markerDeleteHandle(it->handle);

            if (it->type == markerBookmark)
            { // bookmark => leave it empty and delete entry in List
                it = bookmarkErrorHandles.erase(it);
                createNew = false;
            }
            else if (it->type == markerPyBug)
            { // bug => create bug with bookmark
                it->handle = markerAdd(line, markBookmarkSyntaxError);
                it->type = markerBookmarkAndPyBug;
                createNew = false;
                ++it;
            }
            else if (it->type == markerBookmarkAndPyBug)
            { // bookmark and bug => create bug without bookmark
                it->handle = markerAdd(line, markSyntaxError);
                it->type = markerPyBug;
                createNew = false;
                ++it;
            }
            else
            {
                ++it;
            }
        }
        else
        {
            ++it;
        }
    }
    if (createNew && line >= 0 && line < lines())
    {    
        BookmarkErrorEntry newE;
        newE.type = markerBookmark;
        newE.handle = markerAdd(line, markBookmark);
        bookmarkErrorHandles.append(newE);
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::clearAllBookmarks()
{
    QList<BookmarkErrorEntry>::iterator it;
    bool createNew = true;

    it = bookmarkErrorHandles.begin();
    while (it != bookmarkErrorHandles.end())
    {
        if (it->type == markerBookmark)
        { // bookmark => delete it
            markerDeleteHandle(it->handle);
            it = bookmarkErrorHandles.erase(it);
            createNew = false;
        }
        else if (it->type == markerBookmarkAndPyBug)
        { // bookmark and bug => create bug without bookmark
            int line = markerLine(it->handle);
            markerDeleteHandle(it->handle);
            it->handle = markerAdd(line, markSyntaxError);
            it->type = markerPyBug;
            createNew = false;
            ++it;
        }
        else
        {
            ++it;
        }
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::gotoNextBookmark()
{
    int line, index;
    int closestLine = lines();
    getCursorPosition(&line, &index);
    QList<BookmarkErrorEntry>::iterator it;
    it = bookmarkErrorHandles.begin();

    line += 1;

    if (line == lines())
    {
        line = 0;
    }

    while (it != bookmarkErrorHandles.end())
    {
        if ((it->type & markerBookmark) && markerLine(it->handle) < closestLine && markerLine(it->handle) > line)
        {
            closestLine = markerLine(it->handle);
        }
        ++it;
        if (it == bookmarkErrorHandles.end() && closestLine == lines())
        { // eoF reached without finding a bookmark
            it = bookmarkErrorHandles.begin();
            line = 0;
        }
    }
    setCursorPosAndEnsureVisible(closestLine);
    return RetVal(retOk);
    return RetVal(retError);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::gotoPreviousBookmark()
{
    int line, index;
    int closestLine = 0;
    getCursorPosition(&line, &index);
    QList<BookmarkErrorEntry>::iterator it;
    it = bookmarkErrorHandles.begin();

    if (line == 0)
    {
        line = lines()-1;
    }
    else
    {
        line -= 1;
    }

    while (it != bookmarkErrorHandles.end())
    {
        if ((it->type & markerBookmark) && markerLine(it->handle) > closestLine && markerLine(it->handle) < line)
        {
            closestLine = markerLine(it->handle);
        }
        ++it;
        if (it == bookmarkErrorHandles.end() && closestLine == 0)
        { // eoF reached without finding a bookmark
            it = bookmarkErrorHandles.begin();
            line = lines();
        }
    }
    setCursorPosAndEnsureVisible(closestLine);
    return RetVal(retOk);
    return RetVal(retError);

}

//----------------------------------------------------------------------------------------------------------------------------------
// Breakpoint Handling
//----------------------------------------------------------------------------------------------------------------------------------
bool ScriptEditorWidget::lineAcceptsBPs(int line)
{
    // Check if it's a blank or comment line 
    for (int i = 0; i < this->lineLength(line); ++i)
    {
        QChar c = this->text(line).at(i);
        if (c != '\t' && c != ' ' && c != '#' && c != '\n')
        { // it must be a character
            return true;
        }
        else if (this->text(line)[i] == '#' || i == this->lineLength(line)-1)
        { // up to now there have only been '\t'or' ' if there is a '#' now, return ORend of line reached an nothing found
            return false;
        }
    }
    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::toggleBreakpoint(int line)
{
    if (getFilename() == "") return RetVal(retError);

    //!< markerLine(handle) returns -1, if marker doesn't exist any more (because lines have been deleted...)
    std::list<QPair<int, int> >::iterator it;
    BreakPointModel *bpModel = PythonEngine::getInstance()->getBreakPointModel();
    QModelIndexList indexList = bpModel->getBreakPointIndizes(getFilename(), line);

    if (indexList.size()>0)
    {
        bpModel->deleteBreakPoints(indexList);
    }
    else if (lineAcceptsBPs(line))
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

        foreach( const BPMarker &bpEntry, m_breakPointMap)
        {
            if (bpEntry.lineNo == bp.lineno && !bpEntry.markedForDeletion)
            {
                //there is already a breakPoint in this line
                found = true;
                break;
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

        BPMarker m;
        m.bpHandle = newHandle;
        m.lineNo = bp.lineno;
        m.markedForDeletion = false;
        m_breakPointMap.append( m );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< slot, invoked by BreakPointModel
void ScriptEditorWidget::breakPointDelete(QString filename, int lineNo, int /*pyBpNumber*/)
{
    bool found = false;
    if (filename == getFilename() && filename != "")
    {
        QList<BPMarker>::iterator it = m_breakPointMap.begin();

        //markedForDeletion comes prior
        while(it != m_breakPointMap.end())
        {
            if (it->lineNo == lineNo && it->markedForDeletion == true)
            {
                markerDeleteHandle(it->bpHandle);
                it = m_breakPointMap.erase(it);
                found = true;
            }
            else
            {
                ++it;
            }
        }

        if (!found)
        {
            it = m_breakPointMap.begin();
            while(it != m_breakPointMap.end())
            {
                if (it->lineNo == lineNo)
                {
                    markerDeleteHandle(it->bpHandle);
                    it = m_breakPointMap.erase(it);
                }
                else
                {
                    ++it;
                }
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

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ScriptEditorWidget::changeFilename(const QString &newFilename)
{
    QString oldFilename = getFilename();
    
    if (oldFilename.isNull())
    {
        if (newFilename == "" || newFilename.isNull())
        {
            m_filename = QString();
        }
        else
        {
            QFileInfo newFileInfo(newFilename);
            m_filename = newFileInfo.canonicalFilePath();
        }
    }
    else
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
            m_filename = QString();
        }
        else
        {
            QFileInfo newFileInfo(newFilename);
            if (bpModel)
            {
                modelIndexList = bpModel->getBreakPointIndizes(getFilename());
                QList<BreakPointItem> lists = bpModel->getBreakPoints(modelIndexList);
                BreakPointItem temp;
                QList<BreakPointItem> newList;
                for (int i = 0; i < lists.size(); i++)
                {
                    temp = lists.at(i);
                    temp.filename = newFileInfo.canonicalFilePath();
                    newList.push_back(temp);
                }
                bpModel->changeBreakPoints(modelIndexList, newList, false);
            }
            m_filename = newFileInfo.canonicalFilePath();
        }
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::nrOfLinesChanged()
{
    BreakPointModel *bpModel = PythonEngine::getInstance() ? PythonEngine::getInstance()->getBreakPointModel() : NULL;
    BreakPointItem item;
    QModelIndex index;

    QHash<int,int> currentLineHash;
    QModelIndexList oldItemsToDelete; //QList contains the old line numbers whose break points should be deleted in the end
    QList< QPair<QModelIndex, BreakPointItem> > itemsToChange;

    //get current line number of each item in m_breakPointMap (m_breakPointMap still reflects the state before the line change)
    foreach (const BPMarker &marker, m_breakPointMap)
    {
        currentLineHash[marker.bpHandle] = markerLine(marker.bpHandle);
        //qDebug() << "handle " << marker.bpHandle << " was in " << marker.lineNo << " and is in " << markerLine(marker.bpHandle);
    }

    foreach (const BPMarker &marker, m_breakPointMap)
    {
        if (currentLineHash[marker.bpHandle] == -1) //break point does not exist any more, delete it
        {
            oldItemsToDelete.append( bpModel->getFirstBreakPointIndex(getFilename(), marker.lineNo) );
            (const_cast<BPMarker*>(&marker))->markedForDeletion = true;
        }
        else if (currentLineHash[marker.bpHandle] != marker.lineNo) //line has been changed, if there is another breakpoint that is now in this line and that was in a smaller line number before, delete this one
        {
            bool found = false;
            foreach (const BPMarker &other, m_breakPointMap)
            {
                if (currentLineHash[other.bpHandle] == currentLineHash[marker.bpHandle] && other.lineNo < marker.lineNo && !other.markedForDeletion)
                {
                    //two breakpoints now in the same line AND
                    //the other breakpoint has been in a line above the former line nr of marker AND
                    //the other breakpoint has not been deleted for deletion
                    found = true;
                    break;
                }
            }

            if (found) //another unchanged breakpoint was and still is in the new line of marker, therefore delete marker
            {
                oldItemsToDelete.append( bpModel->getFirstBreakPointIndex(getFilename(), marker.lineNo) );
                (const_cast<BPMarker*>(&marker))->markedForDeletion = true;
            }
            else
            {
                //this breakpoint changed its line, but should stay alive
                // marker moved because a line was added or removed
                index = bpModel->getFirstBreakPointIndex(getFilename(), marker.lineNo);
                item = bpModel->getBreakPoint(index);
                item.lineno = currentLineHash[marker.bpHandle]; //new line

                itemsToChange.append( QPair<QModelIndex,BreakPointItem>(index,item) );
            }
        }
    }

    //now send all changes and afterwards delete the items that should be deleted
    for (int i = 0; i < itemsToChange.size(); ++i)
    {
        bpModel->changeBreakPoint(itemsToChange[i].first, itemsToChange[i].second, true);
    }

    bpModel->deleteBreakPoints(oldItemsToDelete);

    //unmark all for deletion flags
    for (int i=0; i < m_breakPointMap.size(); ++i)
    {
        m_breakPointMap[i].markedForDeletion = false;
    }

    // SyntaxCheck   
    if (m_pythonExecutable && m_syntaxCheckerEnabled)
    {
        m_syntaxTimer->start(); //starts or restarts the timer
    }
    if (m_ClassNavigatorEnabled && m_classNavigatorTimerEnabled)
    {
        m_classNavigatorTimer->start(); //starts or restarts the timer
    }
    autoAdaptLineNumberColumnWidth();
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
        m_pythonExecutable = false;
        break;
    case pyTransDebugContinue:
        if (markCurrentLineHandle != -1)
        {
            markerDeleteHandle(markCurrentLineHandle);
        }
        m_pythonExecutable = false;
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
        setReadOnly(false);
        if (markCurrentLineHandle != -1)
        {
            markerDeleteHandle(markCurrentLineHandle);
        }
        pythonBusy = false;
        m_pythonExecutable = true;
        break;
    case pyTransDebugWaiting:
        m_pythonExecutable = true;
        break;
    case pyTransDebugExecCmdBegin:
        m_pythonExecutable = false;
        break;
    case pyTransDebugExecCmdEnd:
        m_pythonExecutable = true;
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorWidget::fileSysWatcherFileChanged(const QString &path) //this signal may be emitted multiple times at once for the same file, therefore the mutex protection is introduced
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
void ItomQsciPrinter::formatPage(QPainter &painter, bool drawing, QRect &area, int pagenr)
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




//----------------------------------------------------------------------------------------------------------------------------------
// Class-Navigator
//----------------------------------------------------------------------------------------------------------------------------------
int ScriptEditorWidget::buildClassTree(ClassNavigatorItem *parent, int parentDepth, int lineNumber)
{
    int i = lineNumber;
    int depth = parentDepth;
    // read from settings
    int tabLength = 4;
    QString line = "";
    QString decoLine;   // @-Decorator Line in previous line of a function
    
    // regular expression for Classes
    QRegExp classes("(\\s*)(class)\\s(.+)\\((.*)\\):\\s*(#?.*)");
    classes.setMinimal(true);

    // regular expression for methods              |> this part might be not in the same line due multiple line parameter set
    QRegExp methods("(\\s*)(def)\\s(_*)?(.+)\\((.*)(\\):\\s*(#?.*))?");
    methods.setMinimal(true);

    // regular expresseion for decorator
    QRegExp decorator("(\\s*)(@)(.+)\\s*(#?.*)");
    methods.setMinimal(true);
    int size = this->lines();
    while(i < size)
    {
        decoLine = this->text(i-1);
        line = this->text(i);

        // CLASS
        if (classes.indexIn(line) != -1)
        {
            if (classes.cap(1).length() >= depth*tabLength)
            { 
                ClassNavigatorItem *classt = new ClassNavigatorItem();
                // Line indented => Subclass of parent
                ++depth;
                classt->m_name = classes.cap(3);
                // classt->m_args = classes.cap(4); // normally not needed
                classt->setInternalType(ClassNavigatorItem::typePyClass);
                classt->m_priv = false; // Class is usually not private
                classt->m_lineno = i;
                parent->m_member.append(classt);
                ++i;
                i = buildClassTree(classt, depth, i);
                --depth;
                continue;
            }
            else 
            {
                return i;
            }
        }
        // METHOD
        else if (methods.indexIn(line) != -1)
        {
            // Methode
            //checken ob line-1 == @declarator besitzt
            ClassNavigatorItem *meth = new ClassNavigatorItem();
            meth->m_name = methods.cap(3) + methods.cap(4);
            meth->m_args = methods.cap(5);
            meth->m_lineno = i;
            if (methods.cap(3) == "_" || methods.cap(3) == "__")
            {
                meth->m_priv = true;                    
            }
            else
            {
                meth->m_priv = false;
            }
            // Check for indentation:
            if (methods.cap(1).length() == 0)
            {// No indentation => Global Method
                if (parent->m_internalType == ClassNavigatorItem::typePyRoot)
                {
                    meth->setInternalType(ClassNavigatorItem::typePyGlobal);
                    parent->m_member.append(meth);
                }
                else
                {
                    DELETE_AND_SET_NULL(meth);
                    return i;
                }
            }            
            else if (methods.cap(1).length() == depth*tabLength)
            {// Child des parents
                if (decorator.indexIn(decoLine) != -1)
                {
                    if (decorator.cap(3) == "staticmethod")
                    {
                        meth->setInternalType(ClassNavigatorItem::typePyStaticDef);
                    }
                    else if (decorator.cap(3) == "classmethod")
                    {
                        meth->setInternalType(ClassNavigatorItem::typePyClMethDef);
                    }
                }
                else
                {
                    meth->setInternalType(ClassNavigatorItem::typePyDef);
                }
                parent->m_member.append(meth);
                ++i;
                continue;
            }
            
            else if (methods.cap(1).length() < depth*tabLength)
            {// Negativ indentation => it must be a child of a parental class
                DELETE_AND_SET_NULL(meth);
                return i;
            }
        }
        ++i;
    }
    return i;
}

//----------------------------------------------------------------------------------------------------------------------------------
// This function is just a workaround because the elapsed timer and requestClassModel cannot connect because of parameterset
void ScriptEditorWidget::classNavTimerElapsed()
{
    m_classNavigatorTimer->stop();
    emit requestModelRebuild(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Slot invoked by Dockwidget when Tabs change (new Tab, other Tab selected, etc)
// This method is used to start the build process of the class tree and the linear model or update the Comboboxes after a Tab change
ClassNavigatorItem* ScriptEditorWidget::getPythonNavigatorRoot()
{
    if (m_ClassNavigatorEnabled)
    {
        // create new Root-Element
        ClassNavigatorItem *rootElement = new ClassNavigatorItem();
        rootElement->m_name = "{Global Scope}";
        rootElement->m_lineno = 0;
        rootElement->setInternalType(ClassNavigatorItem::typePyRoot);

        // create Class-Tree
        buildClassTree(rootElement, 0, 0);

        // send rootItem to DockWidget
        return rootElement;
    }
    else // Otherwise the ClassNavigator is Disabled
    {
        return NULL;
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







} // end namespace ito
