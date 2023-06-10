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

#include "errorLineHighlight.h"

#include "../codeEditor.h"
#include "../managers/textDecorationsManager.h"
#include "../utils/utils.h"

#include <qbrush.h>

namespace ito {

ErrorLineHighlighterMode::ErrorLineHighlighterMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode("ErrorLineHighlighterMode", description),
    QObject(parent),
    m_decoration(NULL),
    m_color(QColor(255, 192, 192))
{
}

//----------------------------------------------------------
/*
*/
ErrorLineHighlighterMode::~ErrorLineHighlighterMode()
{
}

//----------------------------------------------------------
/*
Background color of the caret line. Default is to use a color slightly
darker/lighter than the background color. You can override the
automatic color by setting up this property
*/
QColor ErrorLineHighlighterMode::background() const
{
    return m_color;
}

//----------------------------------------------------------
/*
*/
void ErrorLineHighlighterMode::setBackground(const QColor &color)
{
    m_color = color;

    if (m_decoration)
    {
        m_decoration->setBackground(QBrush(m_color));
    }
    //refresh();
}

//----------------------------------------------------------
/*
*/
void ErrorLineHighlighterMode::setErrorLine(int line)
{
    if (m_decoration)
    {
        editor()->decorations()->remove(m_decoration);
    }

    m_decoration = TextDecoration::Ptr(new TextDecoration(editor()->document(), -1, -1, line, line, 101));
    m_decoration->setBackground(QBrush(m_color));
    m_decoration->setFullWidth();
    editor()->decorations()->append(m_decoration);
}

//----------------------------------------------------------
/*
*/
void ErrorLineHighlighterMode::clearErrorLine()
{
    if (m_decoration.isNull() == false)
    {
        editor()->decorations()->remove(m_decoration);
    }

    m_decoration.clear();
}

//----------------------------------------------------------
/*
*/

void ErrorLineHighlighterMode::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
}

//----------------------------------------------------------
/*
*/
void ErrorLineHighlighterMode::onStateChanged(bool state)
{
    if (state)
    {
    }
    else
    {
        clearErrorLine();
    }

}


} //end namespace ito
