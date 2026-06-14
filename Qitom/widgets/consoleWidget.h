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

#ifndef CONSOLEWIDGET_H
#define CONSOLEWIDGET_H

#include <queue>

#include "common/sharedStructures.h"
#include "common/sharedStructuresQt.h"
#include "../python/qDebugStream.h"
#include "../global.h"

#include "abstractCodeEditorWidget.h"
#include "../codeEditor/modes/lineBackgroundMarker.h"
#include "../codeEditor/modes/pyGotoAssignment.h"
#include "../codeEditor/panels/lineNumber.h"

#include <QKeyEvent>
#include <QDropEvent>
#include <qevent.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qdebug.h>
#include <qsettings.h>
#include <qtimer.h>
#include <qelapsedtimer.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace ito
{

class DequeCommandList;

class ConsoleWidget : public AbstractCodeEditorWidget
{
    Q_OBJECT

public:
    ConsoleWidget(QWidget* parent = NULL);
    ~ConsoleWidget();

    static const QString lineBreak;
    static const QString newCommandPrefix;

    virtual QString codeText(int &line, int &column) const;

protected:
    virtual void loadSettings();
    virtual void contextMenuAboutToShow(int contextMenuLine);
    virtual int startLineOffset(int lineIdx) const;
    void initMenus();

public slots:
    virtual void copy();
    virtual void paste();
    virtual void cut();
    void receiveStream(QString text, ito::tStreamMessageType msgType);
    void pythonRunSelection(QString selectionText);
    void pythonStateChanged(tPythonTransitions pyTransition);
    void clearCommandLine();
    void startInputCommandLine(QSharedPointer<QByteArray> buffer, ItomSharedSemaphore *inputWaitCond);

signals:
    void wantToCopy();
    void sendToLastCommand(QString cmd);
    void sendToPythonMessage(QString cmd);


protected:
    virtual bool keyPressInternalEvent(QKeyEvent *event);
    void dropEvent (QDropEvent *event);
    void dragEnterEvent (QDragEnterEvent *event);
    void dragMoveEvent (QDragMoveEvent *event);
    void wheelEvent(QWheelEvent *event);
    bool canInsertFromMimeData(const QMimeData *source) const;
    void mouseDoubleClickEvent(QMouseEvent *e);

private slots:
    void selChanged();
    void textDoubleClicked(int position, int line, int modifiers);
    void clearAndStartNewCommand();
    void toggleAutoWheel(bool enable);
    void processStreamBuffer();

private:
    struct CmdQueueItem
    {
        CmdQueueItem() :
            m_lineBegin(-1),
            m_nrOfLines(1)
        {}

        CmdQueueItem(const QString &text, int lineBegin, int nrOfLines)  :
            m_singleLine(text),
            m_lineBegin(lineBegin),
            m_nrOfLines(nrOfLines)
        {}

        QString m_singleLine;
        int m_lineBegin;
        int m_nrOfLines;
    };

    struct StreamBuffer
    {
        QString text;
        ito::tStreamMessageType msgType;
    };

    struct InputTextMode
    {
        bool inputModeEnabled;
        bool calltipsModeCurrentState; //current enable state of calltips mode before starting input text mode (only valid if inputModeEnabled = true)
        bool autoCompletionModeCurrentState; //current enable state of completion mode before starting input text mode (only valid if inputModeEnabled = true)
    };

    RetVal initEditor();
    RetVal clearEditor();
    RetVal startNewCommand(bool clearEditorFirst = false);
    RetVal execCommand(int lineBegin, int lineEnd);
    RetVal useCmdListCommand(int dir);

    RetVal executeCmdQueue();

    void autoLineDelete(); //!< delete the first N lines if the command line has more than M (M>=N) lines
    void moveCursorToEnd();
    void moveCursorToValidRegion();

    int checkValidDropRegion(const QPoint &pos);

    void enableInputTextMode();
    void disableInputTextMode();

    int m_startLineBeginCmd; //!< zero-based, first-line of actual (not evaluated command), last line which starts with ">>", -1: no command active

    DequeCommandList *m_pCmdList;

    std::queue<CmdQueueItem> m_cmdQueue; //!< upcoming events to handle

    bool m_canCopy;
    bool m_canCut;
    StreamBuffer m_receiveStreamBuffer;
    QTimer m_receiveStreamBufferTimer;
    bool m_splitLongLines;
    int m_splitLongLinesMaxLength;
    CmdQueueItem m_currentlyExecutedCommand;

    QSharedPointer<LineBackgroundMarkerMode> m_markErrorLineMode;
    QSharedPointer<LineBackgroundMarkerMode> m_markCurrentLineMode;
    QSharedPointer<LineBackgroundMarkerMode> m_markInputLineMode;
    QSharedPointer<LineNumberPanel> m_lineNumberPanel;

    bool m_waitForCmdExecutionDone; //!< true: command in this console is being executed and sends a finish-event, when done.
    bool m_pythonBusy; //!< true: python is executing or debugging a script, a command...

    QString m_temporaryRemovedCommands; //!< removed commands, if python busy, caused by another console instance or script (we assume that these removed commands are always python commands and no printed text)

    ItomSharedSemaphore *m_inputStreamWaitCond; //!< if this is != NULL, a input(...) command is currently running in Python and the command line is ready to receive inputs from the user.
    QSharedPointer<QByteArray> m_inputStreamBuffer;
    int m_inputStartLine; //!< if python-input command is currently used to ask for user-input, this variable holds the line of the input command
    int m_inputStartCol; //!< if python-input command is currently used to ask for user-input, this variable holds the column in line m_inputStartLine, where the first input character starts
    bool m_autoWheel; //!< true if command line should automatically move to the last line if new lines are appended, this is set to false upon a wheel event and will be reset to true if the command line is cleared (clc) or if a new input is added

    QMap<QString, QAction*> m_contextMenuActions;

    InputTextMode m_inputTextMode;

    static const QString longLineWrapPrefix;

};

class DequeCommandList
{
public:
    DequeCommandList(int maxLength);
    ~DequeCommandList();

    RetVal add(const QString &cmd);
    RetVal moveLast();
    QString getPrevious();
    QString getNext();

private:
    int m_maxItems;
    std::deque<QString> m_cmdList;
    std::deque<QString>::reverse_iterator m_rit;
};

} //end namespace ito

#endif
