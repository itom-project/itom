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

#ifndef CONSOLEWIDGET_H
#define CONSOLEWIDGET_H

#include <queue>

#include "../common/sharedStructures.h"
#include "../python/qDebugStream.h"

#include "abstractPyScintillaWidget.h"

//// Under Windows, define QSCINTILLA_MAKE_DLL to create a Scintilla DLL, or
//// define QSCINTILLA_DLL to link against a Scintilla DLL, or define neither
//// to either build or link against a static Scintilla library.
////!< this text is coming from qsciglobal.h
//#define QSCINTILLA_DLL  //http://www.riverbankcomputing.com/pipermail/qscintilla/2007-March/000034.html
//
//#include <Qsci/qsciscintilla.h>
//#include <Qsci/qscilexerpython.h>
//#include <Qsci/qsciapis.h>
#include <QKeyEvent>
#include <QDropEvent>
#include <qstring.h>
#include <qstringlist.h>
#include <qdebug.h>
#include <qsettings.h>

QT_BEGIN_NAMESPACE

QT_END_NAMESPACE

namespace ito
{

class DequeCommandList;

class ConsoleWidget : public AbstractPyScintillaWidget
{
    Q_OBJECT


public:

    ConsoleWidget(QWidget* parent = NULL);
    ~ConsoleWidget();
    
    RetVal printMessage(QStringList msg, ito::tMsgType type = msgTextInfo);
    RetVal printMessage(QString msg, ito::tMsgType type = msgTextInfo);

protected:
    virtual void loadSettings();
    void autoAdaptLineNumberColumnWidth();

public slots:
    virtual void copy();
    virtual void paste();
    virtual void cut();
    void receiveStream(QString text, tMsgType msgType);
    void pythonRunSelection(QString selectionText);
    void pythonStateChanged(tPythonTransitions pyTransition);

signals:
    void wantToCopy();
    void pythonExecuteString(QString cmd);
    void sendToLastCommand(QString cmd);


protected:
    void keyPressEvent (QKeyEvent *event);
    void dropEvent (QDropEvent *event);
    void dragEnterEvent (QDragEnterEvent *event);
    void dragMoveEvent (QDragMoveEvent *event);

private slots:
    void selChanged(); 

private:
    struct cmdQueueStruct
    { 
        cmdQueueStruct() { singleLine = ""; m_lineBegin = -1; m_nrOfLines = 1; }
        cmdQueueStruct(QString text, int lineBegin, int nrOfLines) {singleLine = text; m_lineBegin = lineBegin; m_nrOfLines = nrOfLines; }
        QString singleLine;
        int m_lineBegin;
        int m_nrOfLines;
    };

    RetVal initEditor();
    RetVal clearEditor();
    RetVal startNewCommand(bool clearEditorFirst = false);
    RetVal execCommand(int lineBegin, int lineEnd);
    RetVal useCmdListCommand(int dir);

    RetVal executeCmdQueue();

    RetVal moveCursorToEnd();

    RetVal moveCursorToValidRegion();

    int checkValidDropRegion(const QPoint &pos);
    
    int startLineBeginCmd; //!< zero-based, first-line of actual (not evaluated command), last line which starts with ">>", -1: no command active
    
    DequeCommandList *cmdList; 

    std::queue<cmdQueueStruct> cmdQueue; //!< upcoming events to handle

    bool canCopy;
    bool canCut;

    QDebugStream *qout;
    QDebugStream *qerr;

    unsigned int markErrorLine;
    unsigned int markCurrentLine;

    bool waitForCmdExecutionDone; //!< true: command in this console is being executed and sends a finish-event, when done.
    bool pythonBusy; //!< true: python is executing or debugging a script, a command...

    QString temporaryRemovedCommands; //!< removed text, if python busy, caused by another console instance or script.

    static const QString lineBreak;

};

class DequeCommandList
{
public:
    DequeCommandList(int maxLength);
    ~DequeCommandList();

    RetVal add(QString cmd);
    RetVal moveLast();
    QString getPrevious();
    QString getNext();

private:
    int maxItems;
    std::deque<QString> cmdList;
    std::deque<QString>::reverse_iterator rit;
};

} //end namespace ito

#endif
