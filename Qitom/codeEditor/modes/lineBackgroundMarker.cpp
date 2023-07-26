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

#include "lineBackgroundMarker.h"

#include "../codeEditor.h"
#include "../managers/textDecorationsManager.h"
#include "../utils/utils.h"

#include <qbrush.h>
//#include <qdebug.h>

namespace ito {

LineBackgroundMarkerMode::LineBackgroundMarkerMode(
    const QString &name,
    const QColor &bgcolor,
    const QString &description /*= ""*/,
    QObject *parent /*= NULL*/) :
        Mode(name, description),
        QObject(parent),
        m_color(bgcolor)
{
}

//----------------------------------------------------------
/*
*/
LineBackgroundMarkerMode::~LineBackgroundMarkerMode()
{
}

//----------------------------------------------------------
/*
Background color of the caret line. Default is to use a color slightly
darker/lighter than the background color. You can override the
automatic color by setting up this property
*/
QColor LineBackgroundMarkerMode::background() const
{
    return m_color;
}

//----------------------------------------------------------
/*
*/
void LineBackgroundMarkerMode::setBackground(const QColor &color)
{
    m_color = color;

    foreach (TextDecoration::Ptr deco, m_decorations)
    {
        deco->setBackground(QBrush(m_color));
    }
}

//----------------------------------------------------------
/*
*/
void LineBackgroundMarkerMode::addMarker(int line)
{
    //qDebug() << "add marker in line" << line << m_color;
    TextDecoration::Ptr deco = TextDecoration::Ptr(new TextDecoration(editor()->document(), -1, -1, line, line, 101,"", true));
    deco->setBackground(QBrush(m_color));
    editor()->decorations()->append(deco);
    m_decorations.append(deco);
}

//----------------------------------------------------------
/*
*/
void LineBackgroundMarkerMode::addMarker(int fromLine, int toLine)
{
    //qDebug() << "add marker in lines <<" << fromLine << ":" << toLine << "->" << m_color;
    TextDecoration::Ptr deco = TextDecoration::Ptr(new TextDecoration(editor()->document(), -1, -1, fromLine, toLine, 101, "", true));
    deco->setBackground(QBrush(m_color));
    editor()->decorations()->append(deco);
    m_decorations.append(deco);

    if (toLine > fromLine) //there seems to be a bug that the last line is not selected --> add another special selection in the last line
    {
        deco = TextDecoration::Ptr(new TextDecoration(editor()->document(), -1, -1, toLine, toLine, 101, "", true));
        deco->setBackground(QBrush(m_color));
        editor()->decorations()->append(deco);
        m_decorations.append(deco);
    }
}

//----------------------------------------------------------
/*
*/
void LineBackgroundMarkerMode::clearAllMarkers()
{
    foreach (TextDecoration::Ptr deco, m_decorations)
    {
        //qDebug() << "remove marker " << deco->format.background().color();
        editor()->decorations()->remove(deco);
    }

    m_decorations.clear();
}

//----------------------------------------------------------
/*
*/

void LineBackgroundMarkerMode::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
}

//----------------------------------------------------------
/*
*/
void LineBackgroundMarkerMode::onStateChanged(bool state)
{
    if (state)
    {
    }
    else
    {
        clearAllMarkers();
    }

}


} //end namespace ito
