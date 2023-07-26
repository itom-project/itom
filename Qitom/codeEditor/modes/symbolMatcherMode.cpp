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

#include "symbolMatcherMode.h"

#include "../codeEditor.h"
#include "../managers/textDecorationsManager.h"
#include "../utils/utils.h"


#include <qbrush.h>

namespace ito {

/*static*/ const QByteArray SymbolMatcherMode::chars = "()[]{}";

//----------------------------------------------------------
/*
*/
SymbolMatcherMode::SymbolMatcherMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode("SymbolMatcherMode", description),
    QObject(parent)
{
    setMatchForeground("red");
}

//----------------------------------------------------------
/*
*/
SymbolMatcherMode::~SymbolMatcherMode()
{
}

//-------------------------------------------------------------
/*
Background color of matching symbols.
*/
QBrush SymbolMatcherMode::matchBackground() const
{
    return m_matchBackground;
}

void SymbolMatcherMode::setMatchBackground(const QBrush &value)
{
    m_matchBackground = value;
    refreshDecorations();
}

//-------------------------------------------------------------
/*
Foreground color of matching symbols.
*/
QColor SymbolMatcherMode::matchForeground() const
{
    return m_matchForeground;
}

void SymbolMatcherMode::setMatchForeground(const QColor &value)
{
    m_matchForeground = value;
    refreshDecorations();
}

//-------------------------------------------------------------
/*
Background color of non-matching symbols.
*/
QBrush SymbolMatcherMode::unmatchBackground() const
{
    return m_unmatchBackground;
}

void SymbolMatcherMode::setUnmatchBackground(const QBrush &value)
{
    m_unmatchBackground = value;
    refreshDecorations();
}

//-------------------------------------------------------------
/*
Foreground color of non-matching symbols.
*/
QColor SymbolMatcherMode::unmatchForeground() const
{
    return m_unmatchForeground;
}

void SymbolMatcherMode::setUnmatchForeground(const QColor &value)
{
    m_unmatchForeground = value;
    refreshDecorations();
}

//-------------------------------------------------------------
void SymbolMatcherMode::clearDecorations()
{
    foreach(TextDecoration::Ptr deco, m_decorations)
    {
        editor()->decorations()->remove(deco);
    }
    m_decorations.clear();
}

//-------------------------------------------------------------
void SymbolMatcherMode::match(SymbolMatcherMode::Symbols symbol, QList<Utils::ParenthesisInfo> &data, int cursorPos)
{
    int pos;

    for (int i = 0; i < data.size(); ++i)
    {
        const Utils::ParenthesisInfo &info = data[i];
        pos = (editor()->textCursor().position() -
                editor()->textCursor().block().position());
        if ((info.character == chars[symbol + Open]) && (info.position == pos))
        {
            createDecoration(cursorPos + info.position, \
                matchLeft(symbol, editor()->textCursor().block(), i + 1, 0));
        }
        else if ((info.character == chars[symbol + Close]) && (info.position == (pos - 1)))
        {
            createDecoration(cursorPos + info.position, \
                matchRight(symbol, editor()->textCursor().block(), i - 1, 0));
        }
    }
}

//-------------------------------------------------------------
bool SymbolMatcherMode::matchLeft(SymbolMatcherMode::Symbols symbol, const QTextBlock &currentBlock, int i, int cpt)
{
    QTextBlock current_block = currentBlock;

    QList<Utils::ParenthesisInfo> parentheses;
    QList<Utils::ParenthesisInfo> squareBrackets;
    QList<Utils::ParenthesisInfo> braces;
    QList<Utils::ParenthesisInfo> used;

    while (current_block.isValid())
    {
        Utils::getBlockSymbolData(editor(), current_block, parentheses, squareBrackets, braces);

        switch (symbol)
        {
        case Paren:
            used = parentheses;
            break;
        case Square:
            used = squareBrackets;
            break;
        case Brace:
            used = braces;
            break;
        }

        //foreach (const Utils::ParenthesisInfo &info, used)
        for (int j = i; j < used.size(); ++j)
        {
            Utils::ParenthesisInfo &info = used[j];

            if (info.character == chars[symbol + Open])
            {
                cpt ++;
                continue;
            }
            if ((info.character == chars[symbol + Close]) && (cpt == 0))
            {
                createDecoration(current_block.position() + info.position);
                return true;
            }
            else if (info.character == chars[symbol + Close])
            {
                cpt --;
            }
        }
        current_block = current_block.next();
        i = 0;
    }
    return false;
}

//-------------------------------------------------------------
bool SymbolMatcherMode::matchRight(SymbolMatcherMode::Symbols symbol, const QTextBlock &currentBlock, int i, int nbRightParen)
{
    QTextBlock current_block = currentBlock;

    QList<Utils::ParenthesisInfo> parentheses;
    QList<Utils::ParenthesisInfo> squareBrackets;
    QList<Utils::ParenthesisInfo> braces;
    QList<Utils::ParenthesisInfo> used;

     while (current_block.isValid())
     {
        Utils::getBlockSymbolData(editor(), current_block, parentheses, squareBrackets, braces);

        switch (symbol)
        {
        case Paren:
            used = parentheses;
            break;
        case Square:
            used = squareBrackets;
            break;
        case Brace:
            used = braces;
            break;
        }

        for (int j = i; j >= 0; --j)
        {
            //if (j >= 0)
            //{
                Utils::ParenthesisInfo &info = used[j];
            //}
            if (info.character == chars[symbol + Close])
            {
                nbRightParen ++;
                continue;
            }
            if (info.character == chars[symbol + Open])
            {
                if (nbRightParen == 0)
                {
                    createDecoration(current_block.position() + info.position);
                    return true;
                }
                else
                {
                    nbRightParen --;
                }
            }
        }

        current_block = current_block.previous();

        Utils::getBlockSymbolData(editor(), current_block, parentheses, squareBrackets, braces);

        switch (symbol)
        {
        case Paren:
            used = parentheses;
            break;
        case Square:
            used = squareBrackets;
            break;
        case Brace:
            used = braces;
            break;
        }

        i = used.size() - 1;
     }
    return false;
}

//-------------------------------------------------------------
/*
Performs symbols matching.
*/
void SymbolMatcherMode::doSymbolsMatching()
{
    clearDecorations();
    QTextBlock current_block = editor()->textCursor().block();

    QList<Utils::ParenthesisInfo> parenthesis, squares, braces;
    Utils::getBlockSymbolData(editor(), current_block, parenthesis, squares, braces);
    int pos = editor()->textCursor().block().position();

    match(Paren, parenthesis, pos);
    match(Square, squares, pos);
    match(Brace, braces, pos);
}

//-------------------------------------------------------------
QTextCursor SymbolMatcherMode::createDecoration(int pos, bool match /*= true*/)
{
    QTextCursor cursor = editor()->textCursor();
    cursor.setPosition(pos);
    cursor.movePosition(QTextCursor::NextCharacter, QTextCursor::KeepAnchor);
    TextDecoration::Ptr deco(new TextDecoration(cursor, -1, -1, -1 , -1, 500));
    deco->properties()["line"] = cursor.blockNumber();
    deco->properties()["column"] = cursor.columnNumber();
    deco->properties()["character"] = cursor.selectedText();
    deco->properties()["match"] = match;

    if (match)
    {
        deco->setForeground(m_matchForeground);
        deco->setBackground(m_matchBackground);
    }
    else
    {
        deco->setForeground(m_unmatchForeground);
        deco->setBackground(m_unmatchBackground);
    }
    m_decorations.append(deco);
    editor()->decorations()->append(deco);
    return cursor;
}

//-------------------------------------------------------------
void SymbolMatcherMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(doSymbolsMatching()));
    }
    else
    {
        disconnect(editor(), SIGNAL(cursorPositionChanged()), this, SLOT(doSymbolsMatching()));
    }
}


//-------------------------------------------------------------
void SymbolMatcherMode::refreshDecorations()
{
    foreach(TextDecoration::Ptr deco, m_decorations)
    {
        editor()->decorations()->remove(deco);
        if (deco->properties()["match"].toBool())
        {
            deco->setForeground(m_matchForeground);
            deco->setBackground(m_matchBackground);
        }
        else
        {
            deco->setForeground(m_unmatchForeground);
            deco->setBackground(m_unmatchBackground);
        }
        editor()->decorations()->append(deco);
    }
}

//-------------------------------------------------------------
/*
Find the corresponding symbol position (line, column) of the specified
symbol. If symbol type is PAREN and character_type is OPEN, the
function will look for '('.

:param cursor: QTextCursor
:param character_type: character type to look for (open or close char)
:param symbol_type: symbol type (index in the SYMBOLS map).

Return -1, -1 if nothing found, QPoint.x is line, QPoint.y is column
*/
QPoint SymbolMatcherMode::symbolPos(const QTextCursor &cursor, SymbolMatcherMode::CharType charType /*= Open*/, SymbolMatcherMode::Symbols symbolType /*=Paren*/)
{
    QPoint retval(-1, -1);
    QTextCursor original_cursor = editor()->textCursor();
    editor()->setTextCursor(cursor);
    QTextBlock block = cursor.block();

    QList<Utils::ParenthesisInfo> parentheses;
    QList<Utils::ParenthesisInfo> squareBrackets;
    QList<Utils::ParenthesisInfo> braces;
    Utils::getBlockSymbolData(editor(), block, parentheses, squareBrackets, braces);

    switch (symbolType)
    {
    case Paren:
        match(symbolType, parentheses, block.position());
        break;
    case Square:
        match(symbolType, squareBrackets, block.position());
        break;
    case Brace:
        match(symbolType, braces, block.position());
        break;
    }

    QString character;

    foreach(TextDecoration::Ptr deco, m_decorations)
    {
        character = deco->properties()["character"].toString();
        if (character.size() > 0 && (character[0] == chars[symbolType + charType]))
        {
            int line = deco->properties()["line"].toInt();
            int column = deco->properties()["column"].toInt();
            retval = QPoint(line, column);
            break;
        }
    }
    editor()->setTextCursor(original_cursor);
    clearDecorations();
    return retval;
}

} //end namespace ito
