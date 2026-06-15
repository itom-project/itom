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

#include "caretLineHighlight.h"

#include "../codeEditor.h"
#include "../managers/textDecorationsManager.h"
#include "../utils/utils.h"

#include <qbrush.h>

namespace ito {

CaretLineHighlighterMode::CaretLineHighlighterMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode("CaretLineHighlighterMode", description),
    QObject(parent),
    m_decoration(NULL),
    m_color(QColor()),
    m_blocked(false)
{
}

//----------------------------------------------------------
/*
*/
CaretLineHighlighterMode::~CaretLineHighlighterMode()
{
}

//----------------------------------------------------------
/*
Background color of the caret line. Default is to use a color slightly
darker/lighter than the background color. You can override the
automatic color by setting up this property
*/
QColor CaretLineHighlighterMode::background() const
{
    if (m_color.isValid() || !editor())
    {
        return m_color;
    }
    else
    {
        return Utils::driftColor(editor()->background(), 110);
    }
}

//----------------------------------------------------------
/*
*/
void CaretLineHighlighterMode::setBackground(const QColor &color)
{
    if (m_color != color)
    {
        m_color = color;
        refresh();
    }
}

//----------------------------------------------------------
/*
*/
bool CaretLineHighlighterMode::blocked() const
{
    return m_blocked;
}

//----------------------------------------------------------
/*
*/
void CaretLineHighlighterMode::setBlocked(bool blocked)
{
    if (m_blocked != blocked)
    {
        m_blocked = blocked;
        refresh();
    }
}

//----------------------------------------------------------
/*
*/

void CaretLineHighlighterMode::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
    refresh();
}

//----------------------------------------------------------
/*
*/
void CaretLineHighlighterMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(refresh()));
        connect(editor(), SIGNAL(newTextSet()), this, SLOT(refresh()));
        refresh();
    }
    else
    {
        disconnect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(refresh()));
        disconnect(editor(), SIGNAL(newTextSet()), this, SLOT(refresh()));
        clearDeco();
    }

}

//----------------------------------------------------------
/*
Updates the current line decoration
*/
void CaretLineHighlighterMode::refresh()
{
    if (enabled() && !m_blocked)
    {
        QBrush brush;
        clearDeco();

        if (m_color.isValid())
        {
            brush = QBrush(m_color);
        }
        else
        {
            brush = Utils::driftColor(editor()->background(), 110);
        }

        m_decoration = TextDecoration::Ptr(
            new TextDecoration(editor()->textCursor(), -1, -1, -1, -1, 100, "", true)
        );
        m_decoration->setBackground(brush);
        m_decoration->setFullWidth();
        editor()->decorations()->append(m_decoration);
    }
    else
    {
        clearDeco();
    }
}

//----------------------------------------------------------
/*
Clear line decoration
*/
void CaretLineHighlighterMode::clearDeco()
{
    if (m_decoration.isNull() == false)
    {
        editor()->decorations()->remove(m_decoration);
    }

    m_decoration.clear();
}

} //end namespace ito
