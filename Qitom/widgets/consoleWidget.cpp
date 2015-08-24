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
#include "consoleWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qmessagebox.h>
#include <qfile.h>
#include <qmimedata.h>
#include <qurl.h>
#include <qsettings.h>
#include <qfileinfo.h>
#include <qregexp.h>
#include <QClipboard>

#include "../organizer/userOrganizer.h"
#include "../organizer/scriptEditorOrganizer.h"

namespace ito
{

//!< constants
const QString ConsoleWidget::lineBreak = QString("\n");

//----------------------------------------------------------------------------------------------------------------------------------
ConsoleWidget::ConsoleWidget(QWidget* parent) :
    AbstractPyScintillaWidget(parent),
    startLineBeginCmd(-1),
    canCopy(false),
    canCut(false),
    waitForCmdExecutionDone(false),
    pythonBusy(false),
    cmdList(NULL),
    qout(NULL),
    qerr(NULL)
{
    qDebug("console widget start constructor");
    initEditor();

    loadSettings();

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(reloadSettings()));

    qRegisterMetaType<tMsgType>("tMsgType");

    //redirect cout and cerr to this console
    qout = new QDebugStream(std::cout, msgReturnInfo, ConsoleWidget::lineBreak);
    qerr = new QDebugStream(std::cerr, msgReturnError, ConsoleWidget::lineBreak);
    
    connect(this, SIGNAL(wantToCopy()), SLOT(copy()));
    connect(this, SIGNAL(selectionChanged()), SLOT(selChanged()));
    connect(this, SIGNAL(SCN_DOUBLECLICK(int,int,int)), SLOT(textDoubleClicked(int,int,int)));

    if (qout)
    {
        connect(qout, SIGNAL(flushStream(QString, tMsgType)), this, SLOT(receiveStream(QString, tMsgType)));
    }
    if (qerr)
    {
        connect(qerr, SIGNAL(flushStream(QString, tMsgType)), this, SLOT(receiveStream(QString, tMsgType)));
    }

    const QObject *pyEngine = AppManagement::getPythonEngine(); //PythonEngine::getInstance();

    if (pyEngine)
    {
        connect(this, SIGNAL(pythonExecuteString(QString)), pyEngine, SLOT(pythonRunString(QString)));
        connect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
    }

    cmdList = new DequeCommandList(20);
    QString settingsName(AppManagement::getSettingsFile());
    QSettings *settings = new QSettings(settingsName, QSettings::IniFormat);
    settings->beginGroup("ConsoleDequeCommandList");
    int size = settings->beginReadArray("LastCommandList");
    for (int i = size - 1; i > -1; --i)
    {
        settings->setArrayIndex(i);
        cmdList->add(settings->value("cmd", "").toString());
    }
    settings->endArray();
    settings->endGroup();
    delete settings;

    //!< empty queue
    while (!cmdQueue.empty())
    {
        cmdQueue.pop();
    }

    startNewCommand(true);

    /*freopen ("D:\\test.txt","w",stdout);
    fprintf(stdout, "Test");
    fclose(stdout);*/
}

//----------------------------------------------------------------------------------------------------------------------------------
ConsoleWidget::~ConsoleWidget()
{
    cmdList->moveLast();
    QString settingsName(AppManagement::getSettingsFile());
    QSettings *settings = new QSettings(settingsName, QSettings::IniFormat);
    settings->beginGroup("ConsoleDequeCommandList");
    settings->beginWriteArray("LastCommandList");
    int i = 0;
    QString cmd = cmdList->getPrevious(); 
    while (cmd != "")
    {
        settings->setArrayIndex(i);
        settings->setValue("cmd", cmd);
        cmd = cmdList->getPrevious();
        ++i;
    }
    settings->endArray();
    settings->endGroup();
    delete settings;

    const QObject *pyEngine = AppManagement::getPythonEngine(); //PythonEngine::getInstance();
    if (pyEngine)
    {
        disconnect(this, SIGNAL(pythonExecuteString(QString)), pyEngine, SLOT(pythonRunString(QString)));
        disconnect(pyEngine, SIGNAL(pythonStateChanged(tPythonTransitions)), this, SLOT(pythonStateChanged(tPythonTransitions)));
    }

    DELETE_AND_SET_NULL(cmdList);
    DELETE_AND_SET_NULL(qout);
    DELETE_AND_SET_NULL(qerr);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::initEditor()
{
    setPaper(QColor(1, 81, 107));

    setFolding(QsciScintilla::NoFoldStyle);
    autoAdaptLineNumberColumnWidth(); //setMarginWidth(1,25);
    setMarginSensitivity(1, false);
    setMarginLineNumbers(1, true);

    setWrapMode(QsciScintilla::WrapWord);
    setWrapVisualFlags(QsciScintilla::WrapFlagByBorder, QsciScintilla::WrapFlagNone , 2);

    //with some QScintilla versions, there is a bug if the markerBackgroundColor contains transparancy. Then
    //the marker gets a black border. Problem: PlatQt.cpp of QScintilla:
    //
    //no transparancy: void SurfaceImpl::RoundedRectangle is called -> sets painter->setPen(convertQColor(fore)) or setPen(NoPen)
    //with transparancy: void SurfaceImpl::AlphaRectangle is called -> no impact on setPen, uses the lastly used settings

    markErrorLine = markerDefine(QsciScintilla::Background) ;
    setMarkerBackgroundColor(QColor(255, 192, 192), markErrorLine); //has been (255,0,0,25) -> equal to (255,192,192) on white background

    markCurrentLine = markerDefine(QsciScintilla::Background);
    setMarkerBackgroundColor(QColor(255, 255, 128), markCurrentLine); //has been (255, 255, 0, 50) -> equal to (255,255,128) on white background

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");

    bool ok = false;
    QsciScintilla::WrapVisualFlag start, end;

    int wrapMode = settings.value("WrapMode", 0).toInt(&ok);
    if (!ok)
    {
        wrapMode = 0;
    }
    switch (wrapMode)
    {
        case 0: setWrapMode(QsciScintilla::WrapNone); break;
        case 1: setWrapMode(QsciScintilla::WrapWord); break;
        case 2: setWrapMode(QsciScintilla::WrapCharacter); break;
    };

    QString flagStart = settings.value("WrapFlagStart", "NoFlag").toString();
    if (flagStart == "NoFlag")
    {
        start = QsciScintilla::WrapFlagNone;
    }
    if (flagStart == "FlagText")
    {
        start = QsciScintilla::WrapFlagByText;
    }
    if (flagStart == "FlagBorder")
    {
        start = QsciScintilla::WrapFlagByBorder;
    }

    QString flagEnd = settings.value("WrapFlagEnd", "NoFlag").toString();
    if (flagEnd == "NoFlag")
    {
        end = QsciScintilla::WrapFlagNone;
    }
    if (flagEnd == "FlagText")
    {
        end = QsciScintilla::WrapFlagByText;
    }
    if (flagEnd == "FlagBorder")
    {
        end = QsciScintilla::WrapFlagByBorder;
    }

    int indent = settings.value("WrapIndent", 2).toInt(&ok);
    if (!ok)
    {
        indent = 2;
    }

    setWrapVisualFlags(end, start, indent);

    int indentMode = settings.value("WrapIndentMode", 0).toInt(&ok);
    if (!ok)
    {
        indentMode = 0;
    }
    switch (indentMode)
    {
        case 0: setWrapIndentMode(QsciScintilla::WrapIndentFixed); break;
        case 1: setWrapIndentMode(QsciScintilla::WrapIndentSame); break;
        case 2: setWrapIndentMode(QsciScintilla::WrapIndentIndented); break;
    };

    settings.endGroup();

    AbstractPyScintillaWidget::loadSettings();
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::pythonStateChanged(tPythonTransitions pyTransition)
{
    switch (pyTransition)
    {
    case pyTransBeginRun:
    case pyTransBeginDebug:
    case pyTransDebugContinue:
    case pyTransDebugExecCmdBegin:
        if (!waitForCmdExecutionDone)
        {
            //this part is only executed if a script or other python code is executed but
            //not from the command line. Then, the text that is not executed yet, is
            //temporarily removed and finally added again when python has been finished

            //copy text from startLineBeginCmd on to temporaryRemovedCommands
            QStringList temp;

            for (int i = startLineBeginCmd; i <= lines() - 1; i++)
            {
                temp.push_back(text(i));
            }
            temporaryRemovedCommands = temp.join("");

            setSelection(startLineBeginCmd, 0, lines() - 1, lineLength(lines() - 1));

            removeSelectedText();
        }
        else
        {
            //temporaryRemovedCommands = "";
        }

        pythonBusy = true;
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
    case pyTransDebugWaiting:
    case pyTransDebugExecCmdEnd:

        if (!waitForCmdExecutionDone)
        {
            startLineBeginCmd = lines() - 1;
            append(temporaryRemovedCommands);
            temporaryRemovedCommands = "";
            moveCursorToEnd();
        }
        else
        {
            executeCmdQueue();
        }

        pythonBusy = false;

        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::clearEditor()
{
    startNewCommand(true);
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< new command is a new line starting with ">>"
RetVal ConsoleWidget::startNewCommand(bool clearEditorFirst)
{
    if (clearEditorFirst)
    {
        markerDeleteAll(markErrorLine);
        clear();
    }

    if (text() == "")
    {
        //!< empty editor, just start new command
        append(">>");
        moveCursorToEnd();
        startLineBeginCmd = lines() - 1;
    }
    else
    {
        //!< append at the end of the existing text
        if (lineLength(lines() - 1) > 0)
        {
            append(ConsoleWidget::lineBreak);
        }
        append(">>");
        moveCursorToEnd();
        startLineBeginCmd = lines() - 1;
    }
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::autoAdaptLineNumberColumnWidth()
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

    setMarginWidth(1, s);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::useCmdListCommand(int dir)
{
    QString cmd("");
    int lineFrom, lineTo, indexFrom, indexTo;

    if (startLineBeginCmd >= 0)
    {
        if (dir==1)
        {
            cmd = cmdList->getPrevious();
        }
        else
        {
            cmd = cmdList->getNext();
        }

        //delete possible commands after startLineBeginCmd:
        lineFrom = startLineBeginCmd;
        lineTo = lines() - 1;
        indexFrom = 2;
        indexTo = lineLength(lineTo);
        setSelection(lineFrom, indexFrom, lineTo, indexTo);
        removeSelectedText();
        setCursorPosition(lineFrom, indexFrom);
        append(cmd);

        moveCursorToEnd();
        //startLineBeginCmd = lines()-1;
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
//!> reimplementation to process the keyReleaseEvent
void ConsoleWidget::keyPressEvent(QKeyEvent* event)
{
    int key = event->key();
    Qt::KeyboardModifiers modifiers = event->modifiers();
    int lineFrom, lineTo, indexFrom, indexTo;
    bool acceptEvent = false;
    bool forwardEvent = false;

    if (key == Qt::Key_C && (modifiers & Qt::ControlModifier) && (modifiers & Qt::ShiftModifier))
    {

        PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (pyeng)
        {
            pyeng->pythonInterruptExecution();
        }

        acceptEvent = false; //!< no action necessary
        forwardEvent = false;
    }
    else if (hasFocus() && !waitForCmdExecutionDone && !pythonBusy)
    {
        switch (key)
        {
        case Qt::Key_Up:

            if (isCallTipActive() || isListActive())
            {
                acceptEvent = true;
                forwardEvent = true;
            }
            else
            {
                Qt::KeyboardModifiers modifiers = event->modifiers();
                if ((modifiers &  Qt::ShiftModifier) || (modifiers &  Qt::ControlModifier))
                {
                    acceptEvent = true;
                    forwardEvent = true;
                }
                else
                {
                    getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

                    if (lineFrom == -1)
                    {
                        getCursorPosition(&lineFrom, &indexFrom);
                    }

                    if (lineFrom <= startLineBeginCmd)
                    {
                        acceptEvent = true;
                        forwardEvent = false;
                        useCmdListCommand(1);
                    }
                    else
                    {
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                }
            }
            break;

        case Qt::Key_Down:
            if (isCallTipActive() || isListActive())
            {
                acceptEvent = true;
                forwardEvent = true;
            }
            else
            {
                Qt::KeyboardModifiers modifiers = event->modifiers();
                if ((modifiers &  Qt::ShiftModifier) || (modifiers &  Qt::ControlModifier))
                {
                    acceptEvent = true;
                    forwardEvent = true;
                }
                else
                {
                    getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

                    if (lineFrom == -1)
                    {
                        getCursorPosition(&lineFrom, &indexFrom);
                    }

                    if (lineFrom == lines() - 1 || lineFrom < startLineBeginCmd)
                    {
                        acceptEvent = true;
                        forwardEvent = false;
                        useCmdListCommand(-1);
                    }
                    else
                    {
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                }
            }
            break;

        case Qt::Key_Left:
        case Qt::Key_Right:
        case Qt::Key_NumLock:
        case Qt::Key_Print:
        case Qt::Key_Pause:
        case Qt::Key_Insert:
        case Qt::Key_End:
        case Qt::Key_PageUp:
        case Qt::Key_PageDown:
        case Qt::Key_CapsLock:
        //case Qt::Key_Escape:
            acceptEvent = true;
            forwardEvent = true;
            break;
        
        // Löscht die aktuelle Eingabe
        case Qt::Key_Escape:
            if (isListActive() == false)
            {
                lineTo = lines() - 1;
                indexTo = lineLength(lineTo);

                setSelection(startLineBeginCmd, 2, lineTo, indexTo);
                removeSelectedText();

                if (isCallTipActive())
                {
                    SendScintilla(SCI_CALLTIPCANCEL);
                }
                acceptEvent = true;
                forwardEvent = false;
            }
            else
            {
                acceptEvent = true;
                forwardEvent = true;
            }
            break;
        
        case Qt::Key_Home: //Pos1
            getCursorPosition(&lineFrom, &indexFrom);

            if (lineFrom == startLineBeginCmd && indexFrom>=2)
            {
                if (modifiers & Qt::ShiftModifier)
                {
                    setSelection(startLineBeginCmd,2,lineFrom,indexFrom);
                }
                else
                {
                    setCursorPosition(startLineBeginCmd, 2);
                }
            }
            else
            {
                if (modifiers & Qt::ShiftModifier)
                {
                    setSelection(lineFrom,0,lineFrom,indexFrom);
                }
                else
                {
                    setCursorPosition(lineFrom, 0);
                }
            }
            acceptEvent = true;
            forwardEvent = false;
            break;

        case Qt::Key_Return:
        case Qt::Key_Enter:
            if ((modifiers & Qt::ShiftModifier) == 0)
            {
                //!> return pressed
                if (startLineBeginCmd >= 0 && !pythonBusy)
                {
                    /*if (isListActive())
                    {
                        acceptEvent = true;
                        forwardEvent = false;
                        SendScintilla(SCI_TAB);
                    }
                    else
                    {*/
                        waitForCmdExecutionDone = true;
                        //!< new line for possible error or message
                        append(ConsoleWidget::lineBreak);

                        execCommand(startLineBeginCmd, lines() - 2);
                        acceptEvent = true;
                        forwardEvent = false;

                        //!< do not emit keyPressEvent in QsciScintilla!!
                    //}
                }
                else
                {
                    acceptEvent = false;
                    forwardEvent = false;
                    //!< do not emit keyPressEvent in QsciScintilla!!
                }
                SendScintilla(SCI_CALLTIPCANCEL);
                SendScintilla(SCI_AUTOCCANCEL);
            }
            else
            {
                moveCursorToValidRegion();
                acceptEvent = true;
                forwardEvent = true;
            }
            break;

        case Qt::Key_Backspace:
            //!< check that del and backspace is only pressed in valid cursor context
            getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
            if (lineFrom == -1) //!< no selection
            {
                getCursorPosition(&lineFrom, &indexFrom);

                if (lineFrom < startLineBeginCmd || (lineFrom == startLineBeginCmd && indexFrom<=2))
                {
                    acceptEvent = false;
                    forwardEvent = false;
                }
                else
                {
                    acceptEvent = true;
                    forwardEvent = true;
                }
            }
            else
            {
                if (lineFrom > startLineBeginCmd || (lineFrom == startLineBeginCmd && indexFrom>=2))
                {
                    acceptEvent = true;
                    forwardEvent = true;
                }
                else if ((lineTo == startLineBeginCmd && indexTo > 2) || (lineTo > startLineBeginCmd))
                {
                    setSelection(startLineBeginCmd, 2, lineTo, indexTo);
                    acceptEvent = true;
                    forwardEvent = true;
                }
                else
                {
                    acceptEvent = false;
                    forwardEvent = false;
                }
            }
            break;

        case Qt::Key_Delete:
            //!< check that del and backspace is only pressed in valid cursor context
            getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
            if (lineFrom == -1)
            {
                getCursorPosition(&lineFrom, &indexFrom);

                if (lineFrom < startLineBeginCmd || (lineFrom == startLineBeginCmd && indexFrom<2))
                {
                    acceptEvent = false;
                    forwardEvent = false;
                }
                else
                {
                    acceptEvent = true;
                    forwardEvent = true;
                }
            }
            else
            {
                if (lineFrom > startLineBeginCmd || (lineFrom == startLineBeginCmd && indexFrom>=2))
                {
                    acceptEvent = true;
                    forwardEvent = true;
                }
                else if ((lineTo == startLineBeginCmd && indexTo>2) || (lineTo > startLineBeginCmd))
                {
                    setSelection(startLineBeginCmd, 2, lineTo, indexTo);
                    acceptEvent = true;
                    forwardEvent = true;
                }
                else
                {
                    acceptEvent = false;
                    forwardEvent = false;
                }
            }
            break;

         case Qt::Key_C:
            if ((modifiers & Qt::ControlModifier))
            {
                copy();
                acceptEvent = true;
                forwardEvent = false;
            }
            else
            {
                moveCursorToValidRegion();
                acceptEvent = true;
                forwardEvent = true;
            }
            break;

        case Qt::Key_X:
            if ((modifiers & Qt::ControlModifier))
            {
                cut();
                acceptEvent = true;
                forwardEvent = false;
            }
            else
            {
                moveCursorToValidRegion();
                acceptEvent = true;
                forwardEvent = true;
            }
            break;

        case Qt::Key_V:
            if ((modifiers & Qt::ControlModifier))
            {
                paste();
                acceptEvent = true;
                forwardEvent = false;
            }
            else
            {
                moveCursorToValidRegion();
                acceptEvent = true;
                forwardEvent = true;
            }
            break;

        case Qt::Key_Control:
        case Qt::Key_Shift:
        case Qt::Key_Alt:
        case Qt::Key_AltGr:
            acceptEvent = false; //!< no action necessary
            forwardEvent = false;
            break;

        default:
            moveCursorToValidRegion();
            acceptEvent = true;
            forwardEvent = true;
            break;
        }

        if (acceptEvent && forwardEvent)
        {
            AbstractPyScintillaWidget::keyPressEvent(event);
        }
        else if (!acceptEvent)
        {
            event->ignore();
        }
        else if (acceptEvent && !forwardEvent)
        {
            event->accept();
        }
    }
    else
    {
        event->ignore();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::textDoubleClicked(int position, int line, int modifiers)
{
    if (modifiers == 0)
    {
        QString selectedText = text(line);

        //check for the following style '  File "x:\...py", line xxx, in ... and if found open the script at the given line to jump to the indicated error location in the script
        if (selectedText.startsWith("  File \""))
        {
            QRegExp rx("^  File \"(.*\\.[pP][yY])\", line (\\d+)(, in )?.*$");
            if (rx.indexIn(selectedText) >= 0)
            {
                ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
                if (seo)
                {
                    bool ok;
                    int line = rx.cap(2).toInt(&ok);
                    if (ok)
                    {
                        seo->openScript(rx.cap(1), NULL, line - 1);
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::clearCommandLine()
{
    clear();
    startLineBeginCmd = -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::executeCmdQueue()
{
    cmdQueueStruct value;

    if (cmdQueue.empty())
    {
        markerDeleteAll(markCurrentLine);
        if (waitForCmdExecutionDone)
        {
            waitForCmdExecutionDone = false;
            startNewCommand(false);

            //if further text has been removed within execCommand, it is appended now.
            //text, that is removed due to another run of python (not invoked by this command line),
            //is added in the pythonStateChanged method
            append(temporaryRemovedCommands);
            temporaryRemovedCommands = "";
            moveCursorToEnd();
        }
    }
    else
    {
        waitForCmdExecutionDone = true;
        canCut = false;
        canCopy = false;

        value = cmdQueue.front();
        cmdQueue.pop();

        markerDeleteAll(markCurrentLine);
        for (int i = 0; i < value.m_nrOfLines; i++)
        {
            markerAdd(value.m_lineBegin + i,markCurrentLine);
        }

        if (value.singleLine == "")
        {
            //do nothing, emit end of command
            executeCmdQueue();
        }
        else if (value.singleLine == "clc" || value.singleLine == "clear")
        {
            clear();
            startLineBeginCmd = -1;
            cmdList->add(value.singleLine);
            executeCmdQueue();
            emit sendToLastCommand(value.singleLine);
        }
        else
        {
            //emit pythonExecuteString(value.singleLine);

            QObject *pyEngine = AppManagement::getPythonEngine(); //qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
            if (pyEngine)
            {
                QMetaObject::invokeMethod(pyEngine, "pythonExecStringFromCommandLine", Q_ARG(QString, value.singleLine));
            }
            else
            {
                QMessageBox::critical(this, tr("script execution"), tr("Python is not available"));
            }

            //connect(this, SIGNAL(pythonExecuteString(QString)), pyEngine, SLOT(pythonRunString(QString)));

            //pyThread->pythonInterruptExecution();

            cmdList->add(value.singleLine);
            emit sendToLastCommand(value.singleLine);
        }

        autoAdaptLineNumberColumnWidth();
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::execCommand(int beginLine, int endLine)
{
    if (endLine<beginLine)
    {
        return RetVal(retError);
    }

    QString singleLine;
    QStringList buffer;

    if (beginLine == endLine)
    {
        singleLine = text(beginLine);
        if (singleLine.endsWith('\n'))
        {
            singleLine.chop(1);
        }
        if (singleLine.startsWith(">>"))
        {
            singleLine.remove(0, 2);
        }
        cmdQueue.push(cmdQueueStruct(singleLine, beginLine, 1));
    }
    else
    {
        for (int i = beginLine; i <= endLine; i++)
        {
            singleLine = text(i);
            if (singleLine.endsWith('\n'))
            {
                singleLine.chop(1);
            }
            if (singleLine.startsWith(">>"))
            {
                singleLine.remove(0, 2);
            }

            buffer.append(singleLine);
        }

        const PythonEngine *pyEng = PythonEngine::getInstance();
        QStringList temp;
        QByteArray encoding;
        singleLine = buffer.join("\n");
        QList<int> lines = pyEng->parseAndSplitCommandInMainComponents(singleLine.toLatin1().data(), encoding); //clc command will be accepted and parsed as single command -> this leads to our desired behaviour

        //if lines is empty, a syntax error occurred in the file and the python error indicator is set.
        //This will be checked in subsequent call of run-string or debug-string method.
        if (lines.length() == 0 || (encoding.length() > 0 && lines.length() == 1)) //probably error while execution, execute it in one block
        {
            if (encoding.length() > 0)
            {
                cmdQueue.push(cmdQueueStruct(singleLine, beginLine, 2));
            }
            else
            {
                cmdQueue.push(cmdQueueStruct(singleLine, beginLine, buffer.length()));
            }
        }
        else
        {
            lines.append(buffer.length() + 1); //append last index
            for (int i = 0; i < lines.length() - 1; i++)
            {
                temp = buffer.mid(lines[i] - 1 , lines[i+1] - lines[i]);
                singleLine = temp.join("\n");

                if (encoding.length() > 0)
                {
                    singleLine.prepend("#coding=" + encoding + "\n");
                }

                cmdQueue.push(cmdQueueStruct(singleLine, beginLine + lines[i] - 1, temp.length()));

            }
        }
    }

    //if endLine does not correspond to last line in command line, remove this part
    //and add it to temporaryRemovedCommands. It is added again after that python has been finished
    QStringList temp;

    for (int i = endLine + 1; i < lines(); i++)
    {
        temp.push_back(text(i));
    }
    temporaryRemovedCommands = temp.join("");
    setSelection(endLine + 1, 0, lines() - 1, lineLength(lines() - 1));
    removeSelectedText();

    waitForCmdExecutionDone = true;
    executeCmdQueue();

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::receiveStream(QString text, tMsgType msgType)
{
    printMessage(text, msgType);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::printMessage(QStringList msg, tMsgType type)
{
    QString totalMsg = msg.join(ConsoleWidget::lineBreak);
    int fromLine, toLine;

    switch (type)
    {
        case msgReturnError:
            //!> insert msg after last line
            //append(consoleWidget::lineBreak);
//setLexer(0);
            fromLine = lines() - 1;
            append(totalMsg);
            toLine = lines() - 1;
            if (lineLength(toLine) == 0)
            {
                toLine--;
            }

            for (int i = fromLine; i <= toLine; i++)
            {
                markerAdd(i, markErrorLine);
            }
            moveCursorToEnd();
            startLineBeginCmd = -1;
            if (!pythonBusy)
            {
                this->startNewCommand(false);
            }
            break;

        case msgReturnInfo:
        case msgReturnWarning:
            //!> insert msg after last line
            //append(consoleWidget::lineBreak);
            append(totalMsg);
            moveCursorToEnd();
            startLineBeginCmd = -1;

            if (!pythonBusy)
            {
                this->startNewCommand(false);
            }
            break;

        case msgTextInfo:
        case msgTextWarning:
        case msgTextError:
            //!> insert msg before last line containing ">>" (startLineBeginCmd)
            //totalMsg = totalMsg.append(consoleWidget::lineBreak);
            insertAt(totalMsg, startLineBeginCmd, 0);
            startLineBeginCmd += msg.length();
            moveCursorToEnd();
            break;
    }

    autoAdaptLineNumberColumnWidth();

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::printMessage(QString msg, tMsgType type)
{
    if (msg != "")
    {
        return printMessage(QStringList(msg), type);
    }
    else
    {
        return RetVal(retOk);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::moveCursorToEnd()
{
    int lastLine = lines() - 1;
    setCursorPosition(lastLine, lineLength(lastLine));
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::dropEvent(QDropEvent * event)
{
    const QMimeData *md = event->mimeData();

    //check if a local python file will be dropped -> allow this
//    if (md->hasText() == false && (md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
    if ((md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
    {
        QObject *sew = AppManagement::getScriptEditorOrganizer();
        ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();

        if (uOrg->hasFeature(featDeveloper))
        {
            foreach (const QUrl &url, md->urls())
            {
                if (!url.isLocalFile() || !url.isValid())
                {
                    break;
                }

                QFileInfo file(url.toLocalFile());
                QString suffix = file.suffix().toLower();
                if (suffix != "py")
                {
                    break;
                }

                if (sew != NULL)
                {
                    QMetaObject::invokeMethod(sew, "openScript", Q_ARG(QString,QString(file.absoluteFilePath())), Q_ARG(ItomSharedSemaphore*,NULL));
                }
            }   
        }
    }
    else
    {
        QsciScintilla::dropEvent(event);
    }
    setFocus(); //set focus to this widget such that a key-press (e.g. return) after a drop is directly executed (useful if code from callstack is dropped)
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::dragEnterEvent(QDragEnterEvent * event)
{
    const QMimeData *md = event->mimeData();

    if (md->hasText())
    {
        event->acceptProposedAction();
    }
    else
    {
        ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();
        if (uOrg->hasFeature(featDeveloper))
        {
            //check if a local python file will be dropped -> allow this
            if (md->hasText() == false && (md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
            {
                foreach (const QUrl &url, md->urls())
                {
                    if (!url.isLocalFile() || !url.isValid())
                    {
                        return; //not good
                    }

                    QString suffix = QFileInfo(url.toString()).suffix().toLower();
                    if (suffix != "py")
                    {
                        return; //not good
                    }
                }

                event->acceptProposedAction();
   
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::dragMoveEvent(QDragMoveEvent * event)
{
    const QMimeData *md = event->mimeData();

    //check if a local python file will be dropped -> allow this
    if ((md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
    {
        event->accept();
    }
    else
    {
        QsciScintilla::dragMoveEvent(event);

        //!< if text selected in this widget, starting point before valid region and move action -> ignore
        int lineFrom, lineTo, indexFrom, indexTo;

        //!< check, that selections are only in valid area
        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

        bool dragFromConsole = (event->source() == this);

        if (dragFromConsole && (lineFrom < startLineBeginCmd || (lineFrom == startLineBeginCmd && indexFrom < 2)))
        {
            if (event->dropAction() & Qt::MoveAction)
            {
                event->ignore();
            }
        }
        else
        {
            int res = checkValidDropRegion(event->pos());

            if (res==0)
            {
                event->ignore();
            }
            else
            {
                event->accept();
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*
@return 0: pos invalid, 1: pos valid, 2: pos below last line
*/
int ConsoleWidget::checkValidDropRegion(const QPoint &pos)
{
    if (waitForCmdExecutionDone || pythonBusy)
    {
        return 0;
    }
    else
    {
        long position;
        int line, index;
        QPoint pos2 = pos;

        int margin = marginWidth(1) + marginWidth(2) + marginWidth(3) + marginWidth(4);

        pos2.setX(1+ margin);

        position = SendScintilla(SCI_POSITIONFROMPOINT, pos2.x(), pos2.y());
        if (position>=0)
        {
            lineIndexFromPosition(position, &line, &index);
        }
        else
        {
            line = -1;
        }

        if (line == -1)
        {
            //!< pos is below last line
            return 2;
        }
        else if (line == startLineBeginCmd)
        {
            if (pos.x() <= margin)
            {
                //!< mouse over margin left
                return 0;
            }
            else
            {
                position = SendScintilla(SCI_POSITIONFROMPOINT, pos.x(),pos.y());

                if (position == -1)
                {
                    return 2; //!< mouse at the end of this line
                }
                else
                {
                    lineIndexFromPosition(position, &line, &index);

                    if (index<2)
                    {
                        return 0;
                    }
                    else
                    {
                        return 1;
                    }
                }
            }
        }
        else if (line > startLineBeginCmd)
        {
            return 1;
        }
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::selChanged()
{
    if (waitForCmdExecutionDone)
    {
        canCut = false;
        canCopy = false;
    }
    else
    {
        int lineFrom, lineTo, indexFrom, indexTo;
        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

        if (lineFrom == -1) //nothing selected
        {
            canCut = false;
            canCopy = false;
        }
        else if (lineFrom < startLineBeginCmd)
        {
            canCut = false;
            canCopy = true;
        }
        else if (lineFrom == startLineBeginCmd && indexFrom < 2)
        {
            canCut = false;
            canCopy = true;
        }
        else
        {
            canCut = true;
            canCopy = true;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::copy()
{
    if (canCopy)
    {
        QsciScintilla::copy();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::paste()
{
    moveCursorToValidRegion();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");
    bool formatPastCode = settings.value("formatPastCode", "false").toBool();
    settings.endGroup();

    QClipboard *clipboard = QApplication::clipboard();
    QString clipboardSave = "";

    if (formatPastCode)
    {
        if (clipboard->mimeData()->hasText()) 
        {
            clipboardSave = clipboard->text();
            clipboard->setText(formatPhytonCodePart(clipboard->text()));
        }
    }

    QsciScintilla::paste();

    if (clipboardSave != "")
    {
        clipboard->setText(clipboardSave);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::cut()
{
    if (canCut)
    {
        QsciScintilla::cut();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal ConsoleWidget::moveCursorToValidRegion()
{
    int lineFrom, lineTo, indexFrom, indexTo;

    //!< check, that selections are only in valid area
    getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
    if (lineFrom == -1)
    {
        getCursorPosition(&lineFrom, &indexFrom);
    }

    if (lineFrom == -1)
    {
        //do nothing
    }
    else if (lineFrom < startLineBeginCmd)
    {
        moveCursorToEnd();
    }
    else if (lineFrom == startLineBeginCmd && indexFrom < 2)
    {
        moveCursorToEnd();
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ConsoleWidget::pythonRunSelection(QString selectionText)
{
    // we have to remove the indent
    if (selectionText.length() > 0)
    {
//        waitForCmdExecutionDone = false;

        // 1. identify the indent typ
        QChar indentTyp = 0;
        if (selectionText[0] == '\t')
        {
            indentTyp = '\t';
        }
        else if (selectionText[0] == ' ')
        {
            indentTyp = ' ';
        }

        if (indentTyp != 0)
        {
            // 2. if any indent typ read the first indent
            QString indent = ConsoleWidget::lineBreak + ' ';
            int indentSize = 1;
            while (indentSize < selectionText.length() && selectionText[indentSize] == indentTyp)
            {
                ++indentSize;
                indent += indentTyp;
            }

            // 3. now we have to remove this indent size in first line
            selectionText.remove(0, indentSize); 

            // 4. now we have to remove this indent size in every other lines
            selectionText.replace(indent, ConsoleWidget::lineBreak);
        }

        selectionText += ConsoleWidget::lineBreak;

        insertAt(selectionText, startLineBeginCmd, 2);

        execCommand(startLineBeginCmd, startLineBeginCmd + selectionText.count(ConsoleWidget::lineBreak, Qt::CaseInsensitive) - 1);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
DequeCommandList::DequeCommandList(int maxLength)
{
    maxItems = maxLength;
    cmdList.clear();
    cmdList.push_back(QString());
    rit = cmdList.rbegin();
}

//----------------------------------------------------------------------------------------------------------------------------------
DequeCommandList::~DequeCommandList()
{
    cmdList.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal DequeCommandList::add(QString cmd)
{
    moveLast();
    *rit = cmd;
    cmdList.push_back(QString());

    if (static_cast<int>(cmdList.size()) > this->maxItems)
    {
        cmdList.pop_front();
    }

    moveLast();

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal DequeCommandList::moveLast()
{
    rit = cmdList.rbegin();
    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DequeCommandList::getPrevious()
{
    if (cmdList.size() > 1)
    {
        if (rit < cmdList.rend())
        {
            if ((++rit) < cmdList.rend())
            {
                return *rit;
            }
        }
        else
        {
            moveLast();
            return getPrevious();
        }
    }

    return QString();
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DequeCommandList::getNext()
{
    if (cmdList.size() > 1)
    {
        if (rit > cmdList.rbegin())
        {
            --rit;
            return *rit;
        }
        else
        {
            return QString();
        }
    }
    else
    {
        return QString();
    }
}

} //end namespace ito