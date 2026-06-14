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

#include "wordclick.h"

#include "../codeEditor.h"
#include "../utils/utils.h"
#include "../managers/textDecorationsManager.h"
#include "../delayJobRunner.h"

namespace ito {

//----------------------------------------------------------
/*
*/
WordClickMode::WordClickMode(const QString &name /*="WordClickMode"*/, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    QObject(parent),
    Mode(name, description),
    m_previousCursorStart(-1),
    m_previousCursorEnd(-1),
    m_pTimer(NULL)
{
    m_cursor = QTextCursor();
    m_pTimer = new DelayJobRunnerArgTextCursor<WordClickMode, void(WordClickMode::*)(QTextCursor)>(200);
	m_mouseMoveKeyboardModifiers = Qt::ControlModifier | Qt::ShiftModifier;
}

//----------------------------------------------------------
/*
*/
WordClickMode::~WordClickMode()
{
    delete m_pTimer;
    m_pTimer = NULL;
}



//----------------------------------------------------------
/*
*/
/*virtual*/ void WordClickMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(mouseMoved(QMouseEvent*)), this, SLOT(onMouseMoved(QMouseEvent*)));
        connect(editor(), SIGNAL(mouseReleased(QMouseEvent*)), this, SLOT(onMouseReleased(QMouseEvent*)));
        connect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
        connect(editor(), SIGNAL(mouseDoubleClicked(QMouseEvent*)), this, SLOT(onMouseDoubleClicked(QMouseEvent*)));
    }
    else
    {
        disconnect(editor(), SIGNAL(mouseMoved(QMouseEvent*)), this, SLOT(onMouseMoved(QMouseEvent*)));
        disconnect(editor(), SIGNAL(mouseReleased(QMouseEvent*)), this, SLOT(onMouseReleased(QMouseEvent*)));
        disconnect(editor(), SIGNAL(keyReleased(QKeyEvent*)), this, SLOT(onKeyReleased(QKeyEvent*)));
        disconnect(editor(), SIGNAL(mouseDoubleClicked(QMouseEvent*)), this, SLOT(onMouseDoubleClicked(QMouseEvent*)));
        clearSelection();
    }
}


//--------------------------------------------------------------
/*
*/
void WordClickMode::onMouseDoubleClicked(QMouseEvent *e)
{
    m_pTimer->cancelRequests();
}

//--------------------------------------------------------------
/*
*/
void WordClickMode::onKeyReleased(QKeyEvent *e)
{
	if ((e->modifiers() & m_mouseMoveKeyboardModifiers) == 0)
	{
		clearSelection();
		m_deco.clear();
	}
}

//--------------------------------------------------------------
/*
Selects the word under the mouse cursor.
*/
void WordClickMode::selectWordCursor()
{
    QTextCursor cursor = editor()->wordUnderMouseCursor();
    if ((m_previousCursorStart != cursor.selectionStart()) && \
            m_previousCursorEnd != cursor.selectionEnd())
    {
        removeDecoration();
        addDecoration(cursor);
    }
    m_previousCursorStart = cursor.selectionStart();
    m_previousCursorEnd = cursor.selectionEnd();
}

//--------------------------------------------------------------
/*
*/
void WordClickMode::clearSelection()
{
    removeDecoration();
    editor()->setMouseCursor(Qt::IBeamCursor);
    m_previousCursorStart = -1;
    m_previousCursorEnd = -1;
}

//--------------------------------------------------------------
/*
mouse moved callback
*/
void WordClickMode::onMouseMoved(QMouseEvent *e)
{
    if ((e->modifiers() & m_mouseMoveKeyboardModifiers) == m_mouseMoveKeyboardModifiers)
    {
        QTextCursor cursor = editor()->wordUnderMouseCursor();
        if (!cursor.isNull() && (m_cursor.isNull() || cursor.position() != m_cursor.position()))
        {
            checkWordCursor(cursor);
        }

        m_cursor = cursor;
    }
    else
    {
        m_cursor = QTextCursor();
        clearSelection();
    }
}


//--------------------------------------------------------------
/*
mouse pressed callback
*/
void WordClickMode::onMouseReleased(QMouseEvent *e)
{
    if (e->button() == Qt::LeftButton && m_deco)
    {
        QTextCursor cursor = editor()->wordUnderMouseCursor();
        if (cursor.isNull() == false && cursor.selectedText() != "")
        {
            DELAY_JOB_RUNNER_ARGTEXTCURSOR(m_pTimer, WordClickMode, void(WordClickMode::*)(QTextCursor))->requestJob( \
                this, &WordClickMode::emitWordClicked, cursor);
        }
    }
}

//--------------------------------------------------------------
/*
Adds a decoration for the word under ``cursor``.
*/
void WordClickMode::addDecoration(const QTextCursor &cursor)
{
    if (m_deco.isNull())
    {
        if (cursor.selectedText() != "")
        {
            m_deco = TextDecoration::Ptr(new TextDecoration(cursor));
            if (editor()->background().lightness() < 128)
            {
                m_deco->setForeground(QColor("#0681e0"));
            }
            else
            {
                m_deco->setForeground(Qt::blue);
            }
            m_deco->setAsUnderlined();
            editor()->decorations()->append(m_deco);
            editor()->setMouseCursor(Qt::PointingHandCursor);
        }
        else
        {
            editor()->setMouseCursor(Qt::IBeamCursor);
        }
    }
}

//--------------------------------------------------------------
/*
Removes the word under cursor's decoration
*/
void WordClickMode::removeDecoration()
{
    if (m_deco)
    {
        editor()->decorations()->remove(m_deco);
        m_deco.clear();
    }
}

} //end namespace ito
