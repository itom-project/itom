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
#include "AppManagement.h"

#include <qtooltip.h>

namespace ito {

PyCalltipsMode::PyCalltipsMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent),
    m_pPythonEngine(NULL),
    m_requestCount(0)
{
    m_pPythonEngine = AppManagement::getPythonEngine();
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

        //QString fn = "";
        QString encoding = "utf8";

        /*ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());
        if (sew)
        {
            fn = sew->getFilename();
        }*/

        QString source = editor()->toPlainText();
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
    if (m_requestCount == 0)
    {
        m_requestCount += 1;
        emit jediCalltipRequested(source, line, col, encoding);
    }
}

} //end namespace ito