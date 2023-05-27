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

#include "../python/pythonEngineInc.h"
#include "../python/pythonStatePublisher.h"
#include "../python/qDebugStream.h"
#include "consoleWidget.h"
#include "../global.h"
#include "../AppManagement.h"

#include <qmessagebox.h>
#include <qfile.h>
#include <qmimedata.h>
#include <qurl.h>
#include <qsettings.h>
#include <qfileinfo.h>

#include <qevent.h>
#include <qdebug.h>
#include <qtextboundaryfinder.h>

#include "../codeEditor/managers/panelsManager.h"
#include "../codeEditor/managers/modesManager.h"
#include "../codeEditor/textBlockUserData.h"

#include "../organizer/userOrganizer.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "../helper/IOHelper.h"

namespace ito
{

//!< constants
const QString ConsoleWidget::lineBreak = QString("\n");
const QString ConsoleWidget::longLineWrapPrefix = QString("... ");
const QString ConsoleWidget::newCommandPrefix = QString(">>");

//-------------------------------------------------------------------------------------
ConsoleWidget::ConsoleWidget(QWidget* parent) :
    AbstractCodeEditorWidget(parent),
    m_startLineBeginCmd(-1),
    m_canCopy(false),
    m_canCut(false),
    m_waitForCmdExecutionDone(false),
    m_pythonBusy(false),
    m_pCmdList(NULL),
    m_inputStreamWaitCond(NULL),
    m_inputStartLine(0),
    m_autoWheel(true),
    m_splitLongLines(true),
    m_splitLongLinesMaxLength(200)
{
    m_inputTextMode.inputModeEnabled = false;

    setSelectLineOnCopyEmpty(false);

    m_receiveStreamBuffer.msgType = ito::msgStreamOut;
    connect(&m_receiveStreamBufferTimer, SIGNAL(timeout()), this, SLOT(processStreamBuffer()));
    //m_receiveStreamBufferTimer.setSingleShot(true);
    m_receiveStreamBufferTimer.setInterval(50);
    m_receiveStreamBufferTimer.start();

    qDebug("console widget start constructor");

    initEditor();
    initMenus();

    loadSettings();

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(reloadSettings()));

    connect(this, SIGNAL(wantToCopy()), SLOT(copy()));
    connect(this, SIGNAL(selectionChanged()), SLOT(selChanged()));

    if (AppManagement::getCoutStream())
    {
        connect(AppManagement::getCoutStream(), SIGNAL(flushStream(QString, ito::tStreamMessageType)), this, SLOT(receiveStream(QString, ito::tStreamMessageType)));
    }
    if (AppManagement::getCerrStream())
    {
        connect(AppManagement::getCerrStream(), SIGNAL(flushStream(QString, ito::tStreamMessageType)), this, SLOT(receiveStream(QString, ito::tStreamMessageType)));
    }

	qDebug() << "Streams connected";

    const PythonStatePublisher *pyStatePublisher = qobject_cast<PythonStatePublisher*>(AppManagement::getPythonStatePublisher());

    if (pyStatePublisher)
    {
        connect(pyStatePublisher, &PythonStatePublisher::pythonStateChanged,
            this, &ConsoleWidget::pythonStateChanged);
    }

    m_pCmdList = new DequeCommandList(20);
    QString settingsName(AppManagement::getSettingsFile());
    QSettings *settings = new QSettings(settingsName, QSettings::IniFormat);
    settings->beginGroup("ConsoleDequeCommandList");
    int size = settings->beginReadArray("LastCommandList");
    for (int i = size - 1; i > -1; --i)
    {
        settings->setArrayIndex(i);
        m_pCmdList->add(settings->value("cmd", "").toString());
    }
    settings->endArray();
    settings->endGroup();
    delete settings;

    //!< empty queue
    while (!m_cmdQueue.empty())
    {
        m_cmdQueue.pop();
    }

    startNewCommand(true);
}

//-------------------------------------------------------------------------------------
ConsoleWidget::~ConsoleWidget()
{
    m_pCmdList->moveLast();
    QString settingsName(AppManagement::getSettingsFile());
    QSettings *settings = new QSettings(settingsName, QSettings::IniFormat);
    settings->beginGroup("ConsoleDequeCommandList");
    settings->beginWriteArray("LastCommandList");
    int i = 0;
    QString cmd = m_pCmdList->getPrevious();
    while (cmd != "")
    {
        settings->setArrayIndex(i);
        settings->setValue("cmd", cmd);
        cmd = m_pCmdList->getPrevious();
        ++i;
    }
    settings->endArray();
    settings->endGroup();
    delete settings;

    const PythonStatePublisher *pyStatePublisher = qobject_cast<PythonStatePublisher*>(AppManagement::getPythonStatePublisher());

    if (pyStatePublisher)
    {
        disconnect(pyStatePublisher, &PythonStatePublisher::pythonStateChanged,
            this, &ConsoleWidget::pythonStateChanged);
    }

    DELETE_AND_SET_NULL(m_pCmdList);
}

//-------------------------------------------------------------------------------------
RetVal ConsoleWidget::initEditor()
{
    m_lineNumberPanel = QSharedPointer<LineNumberPanel>(new LineNumberPanel("LineNumberPanel"));
    panels()->append(m_lineNumberPanel.dynamicCast<ito::Panel>());
    m_lineNumberPanel->setOrderInZone(3);

    m_markErrorLineMode = QSharedPointer<LineBackgroundMarkerMode>(new LineBackgroundMarkerMode("MarkErrorLineMode", QColor(255, 192, 192)));
    modes()->append(m_markErrorLineMode.dynamicCast<ito::Mode>());

    m_markCurrentLineMode = QSharedPointer<LineBackgroundMarkerMode>(new LineBackgroundMarkerMode("MarkCurrentLineMode", QColor(255, 255, 128)));
    modes()->append(m_markCurrentLineMode.dynamicCast<ito::Mode>());

    m_markInputLineMode = QSharedPointer<LineBackgroundMarkerMode>(new LineBackgroundMarkerMode("MarkInputLineMode", QColor(179, 222, 171)));
    modes()->append(m_markInputLineMode.dynamicCast<ito::Mode>());

    m_pyAutoIndentMode->setKeyPressedModifiers(Qt::ShiftModifier);

    m_codeCompletionMode->setSelectWithReturn(false);

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::loadSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    bool ok = false;
    int wrapMode = settings.value("WrapMode", 0).toInt(&ok);
    if (!ok)
    {
        wrapMode = 0;
    }

    switch (wrapMode)
    {
        case 0:
            setLineWrapMode(QPlainTextEdit::NoWrap);
            break;
        case 1:
            setLineWrapMode(QPlainTextEdit::WidgetWidth);
            setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
            break;
        case 2:
            setLineWrapMode(QPlainTextEdit::WidgetWidth);
            setWordWrapMode(QTextOption::WrapAnywhere);
            break;
    };


    m_splitLongLines = settings.value("SplitLongLines", true).toBool();
    m_splitLongLinesMaxLength = qMax(10, settings.value("SplitLongLinesMaxLength", 200).toInt());

    /*

    int indent = settings.value("WrapIndent", 2).toInt(&ok);
    if (!ok)
    {
        indent = 2;
    }*/

    m_markErrorLineMode->setBackground(QColor(settings.value("markerErrorForegroundColor", QColor(255, 192, 192)).toString()));
    m_markCurrentLineMode->setBackground(QColor(settings.value("markerCurrentBackgroundColor", QColor(255, 255, 128)).toString()));
    m_markInputLineMode->setBackground(QColor(settings.value("markerInputForegroundColor", QColor(179, 222, 171)).toString()));

    settings.endGroup();

    AbstractCodeEditorWidget::loadSettings();
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::pythonStateChanged(tPythonTransitions pyTransition)
{
    processStreamBuffer();

    switch (pyTransition)
    {
    case pyTransBeginRun:
    case pyTransBeginDebug:
    case pyTransDebugContinue:
    case pyTransDebugExecCmdBegin:
        if (!m_waitForCmdExecutionDone)
        {
            //this part is only executed if a script or other python code is executed but
            //not from the command line. Then, the text that is not executed yet, is
            //temporarily removed and finally added again when python has been finished

            //copy text from m_startLineBeginCmd on to m_temporaryRemovedCommands
            QStringList temp;

            for (int i = m_startLineBeginCmd; i <= lineCount() - 1; i++)
            {
                temp.push_back(lineText(i));

                if (!lineText(i).endsWith(lineBreak) && i < (lineCount() - 1))
                {
                    //lines with a smooth line break have no endline character. add it to distinguish these lines
                    temp.push_back(lineBreak);
                }
            }
            m_temporaryRemovedCommands = temp.join("");
            setSelection(m_startLineBeginCmd, 0, lineCount() - 1, lineLength(lineCount() - 1));
            removeSelectedText();
        }

        m_pythonBusy = true;
        break;
    case pyTransEndRun:
    case pyTransEndDebug:
    case pyTransDebugWaiting:
    case pyTransDebugExecCmdEnd:

        if (!m_waitForCmdExecutionDone)
        {
            if (lineCount() > 0)
            {
                QString txt = lineText(lineCount() - 1);

                if (txt != "" && !txt.endsWith(lineBreak))
                {
                    append(lineBreak);
                }
            }

            m_startLineBeginCmd = lineCount() - 1;

            append(m_temporaryRemovedCommands);
            m_temporaryRemovedCommands = "";
            moveCursorToEnd();
        }
        else
        {
            executeCmdQueue();
        }

        m_pythonBusy = false;

        break;
    }
}

//-------------------------------------------------------------------------------------
RetVal ConsoleWidget::clearEditor()
{
    startNewCommand(true);
    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
//!< new command is a new line starting with ">>" (newCommandPrefix)
RetVal ConsoleWidget::startNewCommand(bool clearEditorFirst)
{
    if (clearEditorFirst)
    {
        m_markErrorLineMode->clearAllMarkers();
        clear();
        m_autoWheel = true;
    }

    if (toPlainText() == "")
    {
        //!< empty editor, just start new command
        append(ConsoleWidget::newCommandPrefix);
        //qDebug() << (">> (startNewCommand)");
        moveCursorToEnd();
        m_startLineBeginCmd = lineCount() - 1;
    }
    else
    {
        //!< append at the end of the existing text
        if (lineLength(lineCount() - 1) > 0)
        {
            append(ConsoleWidget::lineBreak);
        }
        append(ConsoleWidget::newCommandPrefix);
        //qDebug() << (">> (startNewCommand 2)");
        moveCursorToEnd();
        m_startLineBeginCmd = lineCount() - 1;
    }

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::clearAndStartNewCommand()
{
    if (m_pythonBusy)
    {
        clearCommandLine();
    }
    else
    {
        startNewCommand(true);
    }
}

//-------------------------------------------------------------------------------------
RetVal ConsoleWidget::useCmdListCommand(int dir)
{
    QString cmd("");
    int lineFrom, lineTo, indexFrom, indexTo;

    if (m_startLineBeginCmd >= 0)
    {
        if (dir==1)
        {
            cmd = m_pCmdList->getPrevious();
        }
        else
        {
            cmd = m_pCmdList->getNext();
        }

        if (cmd != "")
        {
            //delete possible commands after m_startLineBeginCmd:
            lineFrom = m_startLineBeginCmd;
            lineTo = lineCount() - 1;
            indexFrom = ConsoleWidget::newCommandPrefix.size();
            indexTo = lineLength(lineTo);
            setSelection(lineFrom, indexFrom, lineTo, indexTo);
            removeSelectedText();
            setCursorPosition(lineFrom, indexFrom);
            append(cmd);

            moveCursorToEnd();
        }
    }

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
//!> reimplementation to process the keyReleaseEvent
bool ConsoleWidget::keyPressInternalEvent(QKeyEvent *event)
{
    int key = event->key();
    Qt::KeyboardModifiers modifiers = event->modifiers();
    int lineFrom, lineTo, indexFrom, indexTo;
    bool acceptEvent = false;
    bool forwardEvent = false;

    if (key == Qt::Key_F5 && (modifiers & Qt::ShiftModifier))
    {
        if (m_inputStreamWaitCond)
        {
            m_markInputLineMode->clearAllMarkers();
            m_caretLineHighlighter->setBlocked(false);
            m_inputStreamBuffer->clear();
            m_inputStreamWaitCond->release();
            m_inputStreamWaitCond->deleteSemaphore();
            m_inputStreamWaitCond = NULL;
            disableInputTextMode();
            append(ConsoleWidget::lineBreak);
        }
        else
        {
            PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
            if (pyeng)
            {
                pyeng->pythonInterruptExecutionThreadSafe();
            }
        }

        acceptEvent = false; //!< no action necessary
        forwardEvent = false;
    }
    else if (hasFocus() && (m_inputStreamWaitCond != NULL || (!m_waitForCmdExecutionDone && !m_pythonBusy)))
    {
        switch (key)
        {
        case Qt::Key_Up:
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

                if (lineFrom <= m_startLineBeginCmd)
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
            break;

        case Qt::Key_Down:
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

                if (lineFrom == lineCount() - 1 || lineFrom < m_startLineBeginCmd)
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

        // clears the current input or interrupts an input
        case Qt::Key_Escape:
            //todo
            if (1) //isListActive() == false)
            {
                if (!m_inputStreamWaitCond)
                {
                    lineTo = lineCount() - 1;
                    indexTo = lineLength(lineTo);

                    setSelection(m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size(), lineTo, indexTo);
                    removeSelectedText();
                }
                else
                {
                    m_markInputLineMode->clearAllMarkers();
                    m_caretLineHighlighter->setBlocked(false);
                    m_inputStreamBuffer->clear();
                    m_inputStreamWaitCond->release();
                    m_inputStreamWaitCond->deleteSemaphore();
                    m_inputStreamWaitCond = NULL;
                    disableInputTextMode();
                    append(ConsoleWidget::lineBreak);
                }

                //TODO: something to do?
                /*if (isCallTipActive())
                {
                    SendScintilla(SCI_CALLTIPCANCEL);
                }*/
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

            if (m_inputStreamWaitCond && lineFrom == m_inputStartLine && indexFrom >= m_inputStartCol)
            {
                if (modifiers & Qt::ShiftModifier)
                {
                    setSelection(m_inputStartLine, m_inputStartCol, lineFrom, indexFrom);
                }
                else
                {
                    setCursorPosition(m_inputStartLine, m_inputStartCol);
                }
            }
            else if (lineFrom == m_startLineBeginCmd && indexFrom >= ConsoleWidget::newCommandPrefix.size())
            {
                if (modifiers & Qt::ShiftModifier)
                {
                    setSelection(m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size(), lineFrom, indexFrom);
                }
                else
                {
                    setCursorPosition(m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size());
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
            //todo
            //if ((modifiers & Qt::ShiftModifier) == 0 && !isListActive())
            if ((modifiers & Qt::ShiftModifier) == 0)
            {
                //!> return pressed
                if (m_startLineBeginCmd >= 0 && !m_pythonBusy)
                {
                    m_waitForCmdExecutionDone = true;
                    //!< new line for possible error or message
                    append(ConsoleWidget::lineBreak);

                    execCommand(m_startLineBeginCmd, lineCount() - ConsoleWidget::newCommandPrefix.size());
                    acceptEvent = true;
                    forwardEvent = false;

                    //!< do not emit keyPressEvent in QsciScintilla!!
                }
                else if (m_inputStreamWaitCond) //startInputCommandLine was called before by pythonStream.cpp to wait for a string input (python command 'input(...)'). The semaphore m_inputStreamWaitCond is blocked until the input is obtained.
                {
                    QStringList texts(lineText(m_inputStartLine).mid(m_inputStartCol));
                    for (int i = m_inputStartLine + 1; i < lineCount(); i++)
                    {
                        texts.append(lineText(i));
                    }

                    QByteArray ba = texts.join("").toUtf8().data();

                    if (m_inputStreamBuffer->size() == 0)
                    {
                        *m_inputStreamBuffer = ba;
                    }
                    else
                    {
                        *m_inputStreamBuffer = ba.left(m_inputStreamBuffer->size());
                    }

                    m_markInputLineMode->clearAllMarkers();
                    m_caretLineHighlighter->setBlocked(false);

                    m_inputStreamWaitCond->release();
                    m_inputStreamWaitCond->deleteSemaphore();
                    m_inputStreamWaitCond = NULL;
                    disableInputTextMode();

                    append(ConsoleWidget::lineBreak);
                    acceptEvent = true;
                    forwardEvent = false;
                    //!< do not emit keyPressEvent in QsciScintilla!!
                }
                else
                {
                    acceptEvent = false;
                    forwardEvent = false;
                    //!< do not emit keyPressEvent in QsciScintilla!!
                }
                //TODO: something to do?
                //SendScintilla(SCI_CALLTIPCANCEL);
                //SendScintilla(SCI_AUTOCCANCEL);
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

                if (lineFrom < m_startLineBeginCmd || (lineFrom == m_startLineBeginCmd && indexFrom <= ConsoleWidget::newCommandPrefix.size()))
                {
                    acceptEvent = false;
                    forwardEvent = false;
                }
                else if (m_inputStreamWaitCond && (lineFrom < m_inputStartLine || (lineFrom == m_inputStartLine && indexFrom <= m_inputStartCol)))
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
                if (m_inputStreamWaitCond)
                {
                    if (lineFrom > m_inputStartLine || (lineFrom == m_inputStartLine && indexFrom > m_inputStartCol))
                    {
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else if ((lineTo == m_inputStartLine && indexTo > m_inputStartCol) || (lineTo > m_inputStartLine))
                    {
                        setSelection(m_inputStartLine, m_inputStartCol, lineTo, indexTo);
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else
                    {
                        acceptEvent = false;
                        forwardEvent = false;
                    }
                }
                else
                {
                    if (lineFrom > m_startLineBeginCmd || (lineFrom == m_startLineBeginCmd && indexFrom >= ConsoleWidget::newCommandPrefix.size()))
                    {
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else if ((lineTo == m_startLineBeginCmd && indexTo > ConsoleWidget::newCommandPrefix.size()) || (lineTo > m_startLineBeginCmd))
                    {
                        setSelection(m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size(), lineTo, indexTo);
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else
                    {
                        acceptEvent = false;
                        forwardEvent = false;
                    }
                }
            }
            break;

        case Qt::Key_Delete:
            //!< check that del and backspace is only pressed in valid cursor context
            getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);
            if (lineFrom == -1)
            {
                getCursorPosition(&lineFrom, &indexFrom);

                if (lineFrom < m_startLineBeginCmd || (lineFrom == m_startLineBeginCmd && indexFrom < ConsoleWidget::newCommandPrefix.size()))
                {
                    acceptEvent = false;
                    forwardEvent = false;
                }
                else if (m_inputStreamWaitCond && (lineFrom < m_inputStartLine || (lineFrom == m_inputStartLine && indexFrom < m_inputStartCol)))
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
                if (m_inputStreamWaitCond)
                {
                    if (lineFrom > m_inputStartLine || (lineFrom == m_inputStartLine && indexFrom > m_inputStartCol))
                    {
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else if ((lineTo == m_inputStartLine && indexTo > m_inputStartCol) || (lineTo > m_inputStartLine))
                    {
                        setSelection(m_inputStartLine, m_inputStartCol, lineTo, indexTo);
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else
                    {
                        acceptEvent = false;
                        forwardEvent = false;
                    }
                }
                else
                {
                    if (lineFrom > m_startLineBeginCmd || (lineFrom == m_startLineBeginCmd && indexFrom >= ConsoleWidget::newCommandPrefix.size()))
                    {
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else if ((lineTo == m_startLineBeginCmd && indexTo > ConsoleWidget::newCommandPrefix.size()) || (lineTo > m_startLineBeginCmd))
                    {
                        setSelection(m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size(), lineTo, indexTo);
                        acceptEvent = true;
                        forwardEvent = true;
                    }
                    else
                    {
                        acceptEvent = false;
                        forwardEvent = false;
                    }
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
            event->ignore();
            //AbstractCodeEditorWidget::keyPressEvent(event);
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

    return forwardEvent;
}

//---------------------------------------------------------------------------------------
void ConsoleWidget::mouseDoubleClickEvent(QMouseEvent *e)
{
    CodeEditor::mouseDoubleClickEvent(e);
    int line;
    int column;

    QPoint pos = viewport()->mapFromGlobal(e->globalPos());

    lineIndexFromPosition(pos, &line, &column);

    textDoubleClicked(column, line, e->modifiers());
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::textDoubleClicked(int position, int line, int modifiers)
{
    ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();

    if (uOrg->currentUserHasFeature(featDeveloper) && modifiers == 0)
    {
        QString selectedText = lineText(line);

        if (selectedText.startsWith(ConsoleWidget::newCommandPrefix))
        {
            selectedText = selectedText.mid(ConsoleWidget::newCommandPrefix.size()); //remove trailing >>
        }

        //check for the following style '  File "x:\...py", line xxx, in ... and if found open the script at the given line to jump to the indicated error location in the script
        if (selectedText.contains("file ", Qt::CaseInsensitive) || selectedText.contains("Warning:", Qt::CaseSensitive))
        {
			//check if there are subsequent lines, that start with longLineWrapPrefix, if so, append their texts to the selectedText
			while (lineText(++line).startsWith(longLineWrapPrefix))
			{
				selectedText.append(lineText(line).mid(longLineWrapPrefix.size()));
			}

            QRegularExpression rx("^  File \"(.*\\.[pP][yY])\", line (\\d+)(, in )?.*$");
            auto rxMatch = rx.match(selectedText);

            if (rxMatch.hasMatch())
            {
                ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

                if (seo)
                {
                    bool ok;
                    int lineInMsg = rxMatch.captured(2).toInt(&ok);

                    if (ok)
                    {
                        seo->openScript(rxMatch.captured(1), nullptr, lineInMsg - 1, true);
                    }
                }
            }
            else
            {
                rx.setPattern("^.*Line (\\d+) in file \"(.*\\.[pP][yY])\".*$");
                rxMatch = rx.match(selectedText);

                if (rxMatch.hasMatch())
                {
                    ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
                    if (seo)
                    {
                        bool ok;
                        int line = rxMatch.captured(1).toInt(&ok);
                        if (ok)
                        {
                            seo->openScript(rxMatch.captured(2), NULL, line - 1, true);
                        }
                    }
                }
                else
                {
                    rx.setPattern("^(.*\\.[pP][yY]):(\\d+): (\\w)*Warning:.*$");
                    rxMatch = rx.match(selectedText);

                    if (rxMatch.hasMatch())
                    {
                        ScriptEditorOrganizer *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());
                        if (seo)
                        {
                            bool ok;
                            int line = rxMatch.captured(2).toInt(&ok);
                            if (ok)
                            {
                                seo->openScript(rxMatch.captured(1), NULL, line - 1, true);
                            }
                        }
                    }
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::clearCommandLine()
{
    m_receiveStreamBuffer.text = "";
    clear();
    m_markErrorLineMode->clearAllMarkers();
    m_markCurrentLineMode->clearAllMarkers();

    if (m_codeCompletionMode)
    {
        m_codeCompletionMode->hidePopup();
    }

    m_autoWheel = true;
    m_startLineBeginCmd = -1;
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::startInputCommandLine(QSharedPointer<QByteArray> buffer, ItomSharedSemaphore *inputWaitCond)
{
    enableInputTextMode();

    processStreamBuffer(); //
    m_inputStreamWaitCond = inputWaitCond;
    m_inputStreamBuffer = buffer;
    m_inputStartLine = lineCount() - 1;
    m_inputStartCol = lineText(m_inputStartLine).size();
    m_markInputLineMode->addMarker(m_inputStartLine);
    m_caretLineHighlighter->setBlocked(true);
    setFocus();
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::enableInputTextMode()
{
    if (!m_inputTextMode.inputModeEnabled)
    {
        m_inputTextMode.inputModeEnabled = true;
        m_inputTextMode.autoCompletionModeCurrentState = m_codeCompletionMode->enabled();
        m_inputTextMode.calltipsModeCurrentState = m_calltipsMode->enabled();
        m_codeCompletionMode->setEnabled(false);
        m_calltipsMode->setEnabled(false);
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::disableInputTextMode()
{
    if (m_inputTextMode.inputModeEnabled)
    {
        m_inputTextMode.inputModeEnabled = false;
        m_codeCompletionMode->setEnabled(m_inputTextMode.autoCompletionModeCurrentState);
        m_calltipsMode->setEnabled(m_inputTextMode.calltipsModeCurrentState);
    }
}

//-------------------------------------------------------------------------------------
RetVal ConsoleWidget::executeCmdQueue()
{
    processStreamBuffer();

    if (m_cmdQueue.empty())
    {
        m_currentlyExecutedCommand = CmdQueueItem();
        m_markCurrentLineMode->clearAllMarkers();

        if (m_waitForCmdExecutionDone)
        {
            m_waitForCmdExecutionDone = false;
            startNewCommand(false);

            //if further text has been removed within execCommand, it is appended now.
            //text, that is removed due to another run of python (not invoked by this command line),
            //is added in the pythonStateChanged method
            append(m_temporaryRemovedCommands);
            //qDebug() << "temp commands: " << m_temporaryRemovedCommands;
            m_temporaryRemovedCommands = "";
            moveCursorToEnd();
        }
    }
    else
    {
        m_waitForCmdExecutionDone = true;
        m_canCut = false;
        m_canCopy = false;

        const CmdQueueItem value = m_cmdQueue.front();
        m_cmdQueue.pop();

        m_markCurrentLineMode->clearAllMarkers();
        m_markCurrentLineMode->addMarker(
            value.m_lineBegin,
            value.m_lineBegin + value.m_nrOfLines - 1);

        if (value.m_singleLine == "")
        {
            //do nothing, emit end of command
            executeCmdQueue();
        }
        else if (value.m_singleLine == "clc" || value.m_singleLine == "clear")
        {
            clearCommandLine();
            m_pCmdList->add(value.m_singleLine);
            executeCmdQueue();
            emit sendToLastCommand(value.m_singleLine);
        }
        else if (value.m_singleLine == "clearAll")
        {
            QObject *pyEngine = AppManagement::getPythonEngine();

            if (pyEngine)
            {
                QMetaObject::invokeMethod(pyEngine, "pythonClearAll");
                executeCmdQueue();
                m_pCmdList->add(value.m_singleLine);
                emit sendToLastCommand(value.m_singleLine);
            }
            else
            {
                QMessageBox::critical(this, tr("Script Execution"), tr("Python is not available"));
            }

            autoLineDelete();
        }
        else
        {
            QObject *pyEngine = AppManagement::getPythonEngine();

            if (pyEngine)
            {
                QMetaObject::invokeMethod(pyEngine, "pythonExecStringFromCommandLine", Q_ARG(QString, value.m_singleLine));
            }
            else
            {
                QMessageBox::critical(this, tr("Script Execution"), tr("Python is not available"));
            }

            m_pCmdList->add(value.m_singleLine);

            emit sendToLastCommand(value.m_singleLine);
        }

		autoLineDelete();
    }

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
RetVal ConsoleWidget::execCommand(int beginLine, int endLine)
{
    if (endLine < beginLine)
    {
        return RetVal(retError);
    }

    QString singleLine;
    QStringList buffer;

    if (beginLine == endLine)
    {
        singleLine = lineText(beginLine);
        if (singleLine.endsWith('\n'))
        {
            singleLine.chop(1);
        }
        if (singleLine.startsWith(ConsoleWidget::newCommandPrefix))
        {
            singleLine.remove(0, ConsoleWidget::newCommandPrefix.size());
        }
        m_cmdQueue.push(CmdQueueItem(singleLine, beginLine, 1));
    }
    else
    {
        for (int i = beginLine; i <= endLine; i++)
        {
            singleLine = lineText(i);
            if (singleLine.endsWith('\n'))
            {
                singleLine.chop(1);
            }
            if (singleLine.startsWith(ConsoleWidget::newCommandPrefix))
            {
                singleLine.remove(0, ConsoleWidget::newCommandPrefix.size());
            }

            buffer.append(singleLine);
        }

        const PythonEngine *pyEng = PythonEngine::getInstance();
        QStringList temp;
        QByteArray encoding;
        singleLine = buffer.join("\n");

        //clc command will be accepted and parsed as single command -> this leads to our desired behaviour
        QList<int> lines = pyEng->parseAndSplitCommandInMainComponents(singleLine, encoding);

        //if lines is empty, a syntax error occurred in the file and the python error indicator is set.
        //This will be checked in subsequent call of run-string or debug-string method.
        if (lines.length() == 0 || (encoding.length() > 0 && lines.length() == 1)) //probably error while execution, execute it in one block
        {
            if (encoding.length() > 0)
            {
                m_cmdQueue.push(CmdQueueItem(singleLine, beginLine, 2));
            }
            else
            {
                m_cmdQueue.push(CmdQueueItem(singleLine, beginLine, buffer.length()));
            }
        }
        else
        {
            lines.append(buffer.length() + 1); //append last index
            for (int i = 0; i < lines.length() - 1; i++)
            {
                temp = buffer.mid(lines[i] - 1, lines[i + 1] - lines[i]);

                // remove empty (besides whitechars) lines at the end of each block,
                // else an error can occur if the block is indented
                while (temp.size() > 1)
                {
                    if (temp.last().trimmed() == "")
                    {
                        temp.removeLast();
                    }
                    else
                    {
                        //there is a line with content.
                        break;
                    }
                }

                singleLine = temp.join("\n");

                if (encoding.length() > 0)
                {
                    singleLine.prepend("#coding=" + encoding + "\n");
                }

                m_cmdQueue.push(CmdQueueItem(singleLine, beginLine + lines[i] - 1, temp.length()));
            }
        }
    }

    //if endLine does not correspond to last line in command line, remove this part
    //and add it to m_temporaryRemovedCommands. It is added again after that python has been finished
    QStringList temp;

    for (int i = endLine + 1; i < lineCount(); i++)
    {
        temp.push_back(lineText(i));
        if (!lineText(i).endsWith(lineBreak) && i < (lineCount() - 1))
        {
            //lines with a smooth line break have no endline character. add it to distinguish these lines
            temp.push_back(lineBreak);
        }
    }
    m_temporaryRemovedCommands = temp.join("");
    setSelection(endLine + 1, 0, lineCount() - 1, lineLength(lineCount() - 1));
    removeSelectedText();

    m_waitForCmdExecutionDone = true;
    executeCmdQueue();

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
/*virtual*/ QString ConsoleWidget::codeText(int &line, int &column) const
{
    QStringList textLines;
    const ito::TextBlockUserData *userData = NULL;
    QString temp;

    for (int lineIdx = 0; lineIdx < lineCount(); ++lineIdx)
    {
        userData = getConstTextBlockUserData(lineIdx);
        if (userData == NULL || userData->m_syntaxStyle == ito::TextBlockUserData::StylePython)
        {
            temp = lineText(lineIdx);

            if (temp.startsWith(ConsoleWidget::newCommandPrefix))
            {
                temp = temp.mid(ConsoleWidget::newCommandPrefix.size());
                if (line == lineIdx)
                {
                    column -= ConsoleWidget::newCommandPrefix.size();
                    line = textLines.size();
                }
            }
            else if (line == lineIdx)
            {
                line = textLines.size();
            }

            textLines.append(temp);
        }
    }

    return textLines.join(ConsoleWidget::lineBreak);
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::receiveStream(QString text, ito::tStreamMessageType msgType)
{
    if (m_receiveStreamBuffer.msgType != msgType)
    {
        processStreamBuffer();
        m_receiveStreamBuffer.text = text;
        m_receiveStreamBuffer.msgType = msgType;
        //m_receiveStreamBufferTimer.start();
    }
    else
    {
        m_receiveStreamBuffer.text += text;

        //if (m_receiveStreamBuffer.text.size() > 1500)
        //{
        //    processStreamBuffer();
        ////    repaint();
        //}
        //else
        //{
        //    m_receiveStreamBufferTimer.start();
        //}
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::processStreamBuffer()
{

    if (m_receiveStreamBuffer.text == "")
    {
        return;
    }

    int fromLine, toLine;
    TextBlockUserData *userData = NULL;

    //process very long lines
    if (m_splitLongLines && m_receiveStreamBuffer.text.size() > m_splitLongLinesMaxLength)
    {
        QStringList outputs;
        QStringList splits = m_receiveStreamBuffer.text.split("\n");
        int lineStartPos = 0;
        int prevPos;
        QString substr;

        foreach(const QString &t, splits)
        {
            if (t.size() > m_splitLongLinesMaxLength)
            {
                if (m_receiveStreamBuffer.msgType == ito::msgStreamErr)
                {
                    //more sophisticated word wrap at word boundaries
                    QTextBoundaryFinder finder(QTextBoundaryFinder::Line, t);
                    lineStartPos = 0;
                    finder.setPosition(0);

                    while (lineStartPos < t.size())
                    {
                        finder.setPosition(lineStartPos + m_splitLongLinesMaxLength);
                        if (finder.isAtBoundary())
                        {
                            prevPos = finder.position();
                        }
                        else
                        {
                            prevPos = finder.toPreviousBoundary();
                        }

                        if (prevPos <= lineStartPos)
                        {
                            substr = t.mid(lineStartPos, m_splitLongLinesMaxLength);
                            if (lineStartPos == 0)
                            {
                                outputs.append(substr);
                            }
                            else
                            {
                                outputs.append(longLineWrapPrefix + substr);
                            }
                            lineStartPos += substr.size();
                        }
                        else
                        {
                            substr = t.mid(lineStartPos, prevPos - lineStartPos);
                            if (lineStartPos == 0)
                            {
                                outputs.append(substr);
                            }
                            else
                            {
                                outputs.append(longLineWrapPrefix + substr);
                            }
                            lineStartPos = prevPos;
                        }
                    }
                }
                else
                {
                    //simple (and faster) split after m_splitLongLinesMaxLength characters each
                    outputs.append(t.left(m_splitLongLinesMaxLength));
                    QString rest = t.mid(m_splitLongLinesMaxLength);
                    while (rest.size() > m_splitLongLinesMaxLength)
                    {
                        outputs.append(longLineWrapPrefix + rest.left(m_splitLongLinesMaxLength));
                        rest.remove(0, m_splitLongLinesMaxLength);
                    }

                    if (rest.size() > 0)
                    {
                        outputs.append(longLineWrapPrefix + rest);
                    }
                }
            }
            else
            {
                outputs.append(t);
            }
        }

        m_receiveStreamBuffer.text = outputs.join("\n");

    }

    switch (m_receiveStreamBuffer.msgType)
    {
    case ito::msgStreamErr:
        //case msgReturnError:
        //!> insert msg after last line
        fromLine = lineCount() - 1;
        if (lineLength(fromLine) > 0)
        {
            fromLine++;
        }

        append(m_receiveStreamBuffer.text);

        toLine = lineCount() - 1;
        if (lineLength(toLine) == 0)
        {
            toLine--;
        }

        if (fromLine <= toLine)
        {
            m_markErrorLineMode->addMarker(fromLine, toLine);
        }

        for (int lineIdx = fromLine; lineIdx <= toLine; ++lineIdx)
        {
            userData = getTextBlockUserData(lineIdx, true);
            userData->m_syntaxStyle = ito::TextBlockUserData::StyleError;
        }
        rehighlightBlock(fromLine, toLine);

        moveCursorToEnd();

        if (!m_pythonBusy && m_receiveStreamBuffer.text.right(1) == ConsoleWidget::lineBreak)
        {
            startNewCommand(false);
        }
        else
        {
            m_startLineBeginCmd = lineCount() - 1;
        }

        emit sendToPythonMessage(m_receiveStreamBuffer.text);
        break;

    case ito::msgStreamOut:

        fromLine = lineCount() - 1;
        if (lineLength(fromLine) > 0)
        {
            fromLine++;
        }

        //!> insert msg after last line
        append(m_receiveStreamBuffer.text);

        toLine = lineCount() - 1;
        if (lineLength(toLine) == 0)
        {
            toLine--;
        }

        for (int lineIdx = fromLine; lineIdx <= toLine; ++lineIdx)
        {
            userData = getTextBlockUserData(lineIdx, true);
            userData->m_syntaxStyle = ito::TextBlockUserData::StyleOutput;
        }
        rehighlightBlock(fromLine, toLine);

        moveCursorToEnd();

        if (!m_pythonBusy && m_receiveStreamBuffer.text.right(1) == ConsoleWidget::lineBreak)
        {
            startNewCommand(false);
        }
        else
        {
            m_startLineBeginCmd = lineCount() - 1;
        }

        break;
    }

    m_receiveStreamBuffer.text = "";

	autoLineDelete();

    repaint();
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::moveCursorToEnd()
{
    if (m_autoWheel)
    {
        moveCursor(QTextCursor::End);
    }
}

//-------------------------------------------------------------------------------------
bool ConsoleWidget::canInsertFromMimeData(const QMimeData *source) const
{
    return source->hasText() || source->hasUrls();
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::dropEvent(QDropEvent * event)
{
    const QMimeData *md = event->mimeData();

    //check if a local python file will be dropped -> allow this
//    if (md->hasText() == false && (md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
    if ((md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
    {
        ito::UserOrganizer *uOrg = (UserOrganizer*)AppManagement::getUserOrganizer();

        if (uOrg->currentUserHasFeature(featDeveloper))
        {
            foreach (const QUrl &url, md->urls())
            {
                if (!url.isLocalFile() || !url.isValid())
                {
                    break;
                }

                if (IOHelper::openGeneralFile(url.toLocalFile(), false, true, this, NULL, true).containsError())
                {
                    break;
                }
            }
        }
    }
    else
    {
        bool dragFromConsole = (event->source() == this) || (event->source() && event->source()->parent() == this);

        if (dragFromConsole)
        {
            //!< if text selected in this widget, starting point before valid region and move action -> ignore
            int lineFrom, lineTo, indexFrom, indexTo;

            //!< check, that selections are only in valid area
            getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

            if ((lineFrom < m_startLineBeginCmd ||
                (lineFrom == m_startLineBeginCmd && indexFrom < ConsoleWidget::newCommandPrefix.size())))
            {
                event->setDropAction(Qt::CopyAction);
                event->accept();
            }
        }

        CodeEditor::dropEvent(event);
    }

    setFocus();
}

//-------------------------------------------------------------------------------------
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
        if (uOrg->currentUserHasFeature(featDeveloper))
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

//-------------------------------------------------------------------------------------
void ConsoleWidget::dragMoveEvent(QDragMoveEvent * event)
{
    const QMimeData *md = event->mimeData();

    //check if a local python file will be dropped -> allow this
    if ((md->hasFormat("FileName") || md->hasFormat("text/uri-list")))
    {
        event->acceptProposedAction();
    }
    else
    {
        int validDropRegionResult = checkValidDropRegion(event->pos());

        if (validDropRegionResult == 0)
        {
            event->ignore();
        }
        else
        {
            //!< if text selected in this widget, starting point before valid region and move action -> ignore
            int lineFrom, lineTo, indexFrom, indexTo;

            //!< check, that selections are only in valid area
            getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

            bool dragFromConsole = (event->source() == this) || (event->source() && event->source()->parent() == this);

            if (dragFromConsole && (lineFrom < m_startLineBeginCmd ||
                (lineFrom == m_startLineBeginCmd && indexFrom < ConsoleWidget::newCommandPrefix.size())))
            {
                //turn a move action into a copy action if parts of the read-only section should be dragged.
                event->setDropAction(Qt::CopyAction);
                event->accept();
            }
            else
            {
                event->acceptProposedAction();
            }

            CodeEditor::dragMoveEvent(event);
        }
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::wheelEvent(QWheelEvent *event)
{
    m_autoWheel = false;
    AbstractCodeEditorWidget::wheelEvent(event);
}

//-------------------------------------------------------------------------------------
/*
@return 0: pos invalid, 1: pos valid, 2: pos below last line
*/
int ConsoleWidget::checkValidDropRegion(const QPoint &pos)
{
    if (m_waitForCmdExecutionDone || m_pythonBusy)
    {
        return 0;
    }
    else
    {
        long position;
        int line, index;
        QPoint pos2 = pos;


        pos2.setX(1);
        int margin = 0;
        //TODO: is this correct
        position = cursorForPosition(pos2).position();
        if (position >= 0)
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
        else if (line == m_startLineBeginCmd)
        {
            if (pos.x() <= margin)
            {
                //!< mouse over margin left
                return 0;
            }
            else
            {
                position = cursorForPosition(pos).position();

                if (position == -1)
                {
                    return 2; //!< mouse at the end of this line
                }
                else
                {
                    lineIndexFromPosition(position, &line, &index);

                    if (index < 2)
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
        else if (line > m_startLineBeginCmd)
        {
            return 1;
        }
    }

    return 0;
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::selChanged()
{
    if (m_waitForCmdExecutionDone)
    {
        m_canCut = false;
        m_canCopy = false;
    }
    else
    {
        int lineFrom, lineTo, indexFrom, indexTo;
        getSelection(&lineFrom, &indexFrom, &lineTo, &indexTo);

        if (lineFrom == -1) //nothing selected
        {
            m_canCut = false;
            m_canCopy = false;
        }
        else if (lineFrom < m_startLineBeginCmd)
        {
            m_canCut = false;
            m_canCopy = true;
        }
        else if (lineFrom == m_startLineBeginCmd && indexFrom < ConsoleWidget::newCommandPrefix.size())
        {
            m_canCut = false;
            m_canCopy = true;
        }
        else
        {
            m_canCut = true;
            m_canCopy = true;
        }
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::copy()
{
    if (m_canCopy)
    {
        AbstractCodeEditorWidget::copy();
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::paste()
{
    moveCursorToValidRegion();

    AbstractCodeEditorWidget::paste();
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::cut()
{
    if (m_canCut)
    {
        AbstractCodeEditorWidget::cut();
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::moveCursorToValidRegion()
{
    m_autoWheel = true;

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
    else if (lineFrom < m_startLineBeginCmd)
    {
        moveCursorToEnd();
    }
    else if (lineFrom == m_startLineBeginCmd && indexFrom < ConsoleWidget::newCommandPrefix.size())
    {
        moveCursorToEnd();
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::pythonRunSelection(QString selectionText)
{
    // we have to remove the indent
    if (selectionText.length() > 0)
    {
        int lineCount = 0;
        selectionText = formatCodeBeforeInsertion(selectionText, lineCount, true, "");

        if (selectionText.endsWith("\n"))
        {
            insertAt(selectionText, m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size());
        }
        else
        {
            insertAt(selectionText + "\n", m_startLineBeginCmd, ConsoleWidget::newCommandPrefix.size());
        }

        execCommand(m_startLineBeginCmd, m_startLineBeginCmd + lineCount - 1);
    }
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::initMenus()
{
    QMenu *menu = contextMenu();

    m_contextMenuActions["undo"] = menu->addAction(QIcon(":/editor/icons/editUndo.png"), tr("&Undo"), this, SLOT(undo()));
    m_contextMenuActions["redo"] = menu->addAction(QIcon(":/editor/icons/editRedo.png"), tr("&Redo"), this, SLOT(redo()));
    m_contextMenuActions["undo_redo_separator"] = menu->addSeparator();
    m_contextMenuActions["cut"] = menu->addAction(QIcon(":/editor/icons/editCut.png"), tr("&Cut"), this, SLOT(cut()));
    m_contextMenuActions["copy"] = menu->addAction(QIcon(":/editor/icons/editCopy.png"), tr("Cop&y"), this, SLOT(copy()));
    m_contextMenuActions["paste"] = menu->addAction(QIcon(":/editor/icons/editPaste.png"), tr("&Paste"), this, SLOT(paste()));
    m_contextMenuActions["delete"] = menu->addAction(QIcon(":/editor/icons/editDelete.png"), tr("Clear Command Line"), this, SLOT(clearAndStartNewCommand()));
    menu->addSeparator();
    m_contextMenuActions["select_all"] = menu->addAction(tr("Select All"), this, SLOT(selectAll()));
    menu->addSeparator();
    m_contextMenuActions["auto_scroll"] = menu->addAction(tr("Auto Scroll"), this, SLOT(toggleAutoWheel(bool)));
}


//-------------------------------------------------------------------------------------
void ConsoleWidget::contextMenuAboutToShow(int contextMenuLine)
{
    bool read_only = isReadOnly();
    bool has_selection = hasSelectedText();

    m_contextMenuActions["undo"]->setVisible(!read_only);
    m_contextMenuActions["redo"]->setVisible(!read_only);
    m_contextMenuActions["undo_redo_separator"]->setVisible(!read_only);
    m_contextMenuActions["cut"]->setVisible(!read_only);
    m_contextMenuActions["paste"]->setVisible(!read_only);
    m_contextMenuActions["delete"]->setVisible(true);

    if (!read_only)
    {
        m_contextMenuActions["undo"]->setEnabled(isUndoAvailable());
        m_contextMenuActions["redo"]->setEnabled(isRedoAvailable());
        m_contextMenuActions["cut"]->setEnabled(has_selection && m_canCut);
    }

    m_contextMenuActions["copy"]->setEnabled(has_selection && m_canCopy);

    if (!read_only)
    {
        m_contextMenuActions["paste"]->setEnabled(canPaste());
        m_contextMenuActions["delete"]->setEnabled(length() != 0);
    }

    m_contextMenuActions["select_all"]->setEnabled(length() != 0);
    m_contextMenuActions["auto_scroll"]->setCheckable(true);
    m_contextMenuActions["auto_scroll"]->setChecked(m_autoWheel);

    AbstractCodeEditorWidget::contextMenuAboutToShow(contextMenuLine);
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::autoLineDelete()
{
    const int cutoffLine = 50000;
    const int removeLines = 25000;

	if (lineCount() > cutoffLine)
	{
		setSelection(0, 0, removeLines, lineLength(removeLines));
		removeSelectedText();

        //adapt lines numbers of items in execution queue
        std::queue<CmdQueueItem> newQueue;
        while (m_cmdQueue.empty() == false)
        {
            CmdQueueItem q = m_cmdQueue.front();
            m_cmdQueue.pop();
            if (q.m_lineBegin > removeLines)
            {
                q.m_lineBegin -= (removeLines + 1);
                newQueue.push(q);
            }
        }

        m_cmdQueue = newQueue;
	}
}

//-------------------------------------------------------------------------------------
void ConsoleWidget::toggleAutoWheel(bool enable)
{
    m_autoWheel = enable;
}

//-------------------------------------------------------------------------------------
int ConsoleWidget::startLineOffset(int lineIdx) const
{
    if (lineIdx == m_startLineBeginCmd)
    {
        return 2;
    }

    return 0;
}


//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
DequeCommandList::DequeCommandList(int maxLength)
{
    m_maxItems = maxLength;
    m_cmdList.clear();
    m_cmdList.push_back(QString());
    m_rit = m_cmdList.rbegin();
}

//-------------------------------------------------------------------------------------
DequeCommandList::~DequeCommandList()
{
    m_cmdList.clear();
}

//-------------------------------------------------------------------------------------
RetVal DequeCommandList::add(const QString &cmd)
{
    moveLast();
    *m_rit = cmd;
    m_cmdList.push_back(QString());

    if (static_cast<int>(m_cmdList.size()) > m_maxItems)
    {
        m_cmdList.pop_front();
    }

    moveLast();

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
RetVal DequeCommandList::moveLast()
{
    m_rit = m_cmdList.rbegin();
    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
QString DequeCommandList::getPrevious()
{
    if (m_cmdList.size() > 1)
    {
        if (m_rit < m_cmdList.rend())
        {
            if ((++m_rit) < m_cmdList.rend())
            {
                return *m_rit;
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

//-------------------------------------------------------------------------------------
QString DequeCommandList::getNext()
{
    if (m_cmdList.size() > 1 && m_rit > m_cmdList.rbegin())
    {
        --m_rit;
        return *m_rit;
    }

    return QString();
}

} //end namespace ito
