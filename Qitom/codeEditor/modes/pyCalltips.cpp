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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source 
    code editor PyQode from Colin Duquesnoy and others 
    (see https://github.com/pyQode). PyQode itself is licensed under 
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors. 

*********************************************************************** */

#include "pyCalltips.h"

#include "../codeEditor.h"
#include "../utils/utils.h"
#include "../managers/panelsManager.h"
#include "AppManagement.h"
#include "../../widgets/scriptEditorWidget.h"

#include "python/pythonEngine.h"

#include <qdir.h>
#include <qtooltip.h>

namespace ito {

PyCalltipsMode::PyCalltipsMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent),
    m_pPythonEngine(NULL),
    m_requestCount(0)
{
    m_pPythonEngine = AppManagement::getPythonEngine();
    /*if (m_pPythonEngine)
    {
        connect(this, SIGNAL(jediCalltipRequested(QString,int,int,QString,QString,QByteArray)), m_pPythonEngine, SLOT(jediCalltipRequested(QString,int,int,QString,QString,QByteArray)));
    }*/

    m_disablingKeys << Qt::Key_ParenRight << \
            Qt::Key_Return << \
            Qt::Key_Left << \
            Qt::Key_Right << \
            Qt::Key_Up << \
            Qt::Key_Down << \
            Qt::Key_End << \
            Qt::Key_Home << \
            Qt::Key_PageDown << \
            Qt::Key_PageUp << \
            Qt::Key_Backspace << \
            Qt::Key_Delete;
}

//----------------------------------------------------------
/*
*/
PyCalltipsMode::~PyCalltipsMode()
{
}


//----------------------------------------------------------
/*
*/
void PyCalltipsMode::onStateChanged(bool state)
{
    if (m_pPythonEngine)
    {
        if (state)
        {
            connect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
        }
        else
        {
            disconnect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
        }   
    }
}

//----------------------------------------------------------
/*
Auto indent if the released key is the return key.
:param event: the key event
*/
void PyCalltipsMode::onKeyReleased(QKeyEvent *e)
{
    if (e->key() == Qt::Key_ParenLeft || \
            e->key() == Qt::Key_Comma)
    {
        QTextCursor tc = editor()->textCursor();
        int line = tc.blockNumber();
        int col = tc.columnNumber();

        QString encoding = "utf8";

        QString source = editor()->codeText(line, col); // line and col might be changed if code is a virtual code (e.g. for command line, containing all its history)
        // jedi has a bug if the statement has a closing parenthesis
        // remove it!
        QStringList lines = Utils::splitlines(source);
        QString l;

        if (line >= 0 && line < lines.size())
        {
            l = Utils::rstrip(lines[line]);
        }
        else
        {
            // at the beginning of the last line (empty)
            return;
        }

        if (l.endsWith(")"))
        {
            lines[line] = l.left(l.size() - 1);
            if (col > lines[line].size())
            {
                col = lines[line].size();
            }
        }

        source = lines.join("\n");
        requestCalltip(source, line, col, encoding);
    }
    else if (m_disablingKeys.contains(e->key())) 
    {
        QToolTip::hideText();
    }
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::requestCalltip(const QString &source, int line, int col, const QString &encoding)
{
    PythonEngine *pyEng = (PythonEngine*)m_pPythonEngine;
    if (pyEng && (m_requestCount == 0))
    {
        ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());
        QString filename;
        if (sew)
        {
            filename = sew->getFilename();
        }

        if (filename == "")
        {
            filename = QDir::cleanPath(QDir::current().absoluteFilePath("__temporaryfile__.py"));
        }

        if (pyEng->tryToLoadJediIfNotYetDone())
        {
            m_requestCount += 1;

            ito::JediCalltipRequest request;
            request.m_callbackFctName = "onJediCalltipResultAvailable";
            request.m_col = col;
            request.m_encoding = encoding;
            request.m_line = line;
            request.m_path = filename;
            request.m_sender = this;
            request.m_source = source;

            pyEng->enqueueJediCalltipRequest(request);
            //emit jediCalltipRequested(source, line, col, filename, encoding, "onJediCalltipResultAvailable");
        }
        else
        {
            onStateChanged(false);
        }
    }
}

//--------------------------------------------------------------------------------
bool PyCalltipsMode::isLastChardEndOfWord() const
{
    QTextCursor tc = editor()->wordUnderCursor(false);
    tc.setPosition(tc.position());
    tc.movePosition(QTextCursor::StartOfLine, QTextCursor::KeepAnchor);
    QString l = tc.selectedText();
    if (l.size() > 0)
    {
        QChar lastChar = l[l.size() - 1];
        QString seps = editor()->wordSeparators();
        QString symbols = ", (";
        return (seps.contains(lastChar)) && !(symbols.contains(lastChar));
    }
    else
    {
        return false;
    }
}

//--------------------------------------------------------------------------------
void PyCalltipsMode::onJediCalltipResultAvailable(QVector<ito::JediCalltip> calltips)
{
     m_requestCount--;

    if (isLastChardEndOfWord() || calltips.size() == 0)
    {
        return;
    }

    JediCalltip calltip = calltips[0];

    /*
    int index = args["call.index"].toInt();
    int col = args["column"].toInt();

    // create a formatted calltip (current index appear in bold)
    QString calltip = QString("<p style='white-space:pre'>%1.%2(").arg(args["call.module.name"].toString()).arg(args["call.call_name"].toString());
    QStringList callParams = args["call.params"].toString().split(";;");
    for (int i = 0; i < callParams.size(); ++i)
    {
        QString param = callParams[i];
        if ((i < callParams.size() - 1) && !param.endsWith(','))
        {
            param += ", ";
        }
        if (param.endsWith(','))
        {
            param += " ";  // pep8 calltip
        }
        if (i == index)
        {
            calltip += "<b>";
        }
        calltip += param;
        if (i == index)
        {
            calltip += "</b>";
        }
    calltip += ")</p>";
    */

    // set tool tip position at the start of the bracket
    int char_width = editor()->fontMetrics().width('A');
    int w_offset = (calltip.m_column - calltip.m_bracketStartCol) * char_width;
    QPoint position(
        editor()->cursorRect().x() - w_offset,
        editor()->cursorRect().y() + char_width +
        editor()->panels()->marginSize(ito::Panel::Top));
    position = editor()->mapToGlobal(position);
    // show tooltip
    QToolTip::showText(position, calltip.m_calltipText, editor());
}

} //end namespace ito