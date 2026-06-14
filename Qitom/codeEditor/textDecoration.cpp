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

#include "textDecoration.h"

#include <QTextBlock>
#include <qdebug.h>

namespace ito {

TextDecoration::TextDecoration()
    :
    QTextEdit::ExtraSelection(),
    m_signals(new TextDecorationsSignals()),
    m_drawOrder(-1),
    m_tooltip("")
{
}

//-----------------------------------------------------------
/*Creates a text decoration.

.. note:: start_pos/end_pos and start_line/end_line pairs let you
    easily specify the selected text. You should use one pair or the
    other or they will conflict between each others. If you don't
    specify any values, the selection will be based on the cursor.

:param cursor_or_bloc_or_doc: Reference to a valid
    QTextCursor/QTextBlock/QTextDocument
:param start_pos: Selection start position
:param end_pos: Selection end position
:param start_line: Selection start line.
:param end_line: Selection end line.
:param draw_order: The draw order of the selection, highest values will
    appear on top of the lowest values.
:param tooltip: An optional tooltips that will be automatically shown
    when the mouse cursor hover the decoration.
:param full_width: True to select the full line width.
.. note:: Use the cursor selection if startPos and endPos are none.
*/
TextDecoration::TextDecoration(const QTextCursor &cursor, int startPos /*=-1*/, int endPos /*=-1*/, \
    int startLine /*=-1*/, int endLine /*=-1*/, int drawOrder /*=0*/, const QString &tooltip /*= ""*/, \
    bool fullWidth /*= false*/) :
    QTextEdit::ExtraSelection(),
    m_signals(new TextDecorationsSignals()),
    m_drawOrder(drawOrder),
    m_tooltip(tooltip)
{
    this->cursor = cursor;

    if (fullWidth)
    {
        setFullWidth(fullWidth);
    }

    if (startPos >= 0)
    {
        this->cursor.setPosition(startPos);
    }

    if (endPos >= 0)
    {
        this->cursor.setPosition(endPos, QTextCursor::KeepAnchor);
    }

    if (startLine >= 0)
    {
        //this->cursor.movePosition(QTextCursor::Start, QTextCursor::MoveAnchor);
        //this->cursor.movePosition(QTextCursor::Down, QTextCursor::MoveAnchor, startLine);

        QTextBlock b = cursor.document()->findBlockByNumber(startLine);
        this->cursor.setPosition(b.position(), QTextCursor::MoveAnchor);
    }

    if (endLine >= 0)
    {
        //this->cursor.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor, endLine - startLine);

        QTextBlock b = cursor.document()->findBlockByNumber(endLine);
        if (b.isValid())
        {
            this->cursor.setPosition(b.position(), QTextCursor::KeepAnchor);
        }
        else
        {
            this->cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
        }
    }
}

//-----------------------------------------------------------
/*Creates a text decoration.

.. note:: start_pos/end_pos and start_line/end_line pairs let you
    easily specify the selected text. You should use one pair or the
    other or they will conflict between each others. If you don't
    specify any values, the selection will be based on the cursor.

:param cursor_or_bloc_or_doc: Reference to a valid
    QTextCursor/QTextBlock/QTextDocument
:param start_pos: Selection start position
:param end_pos: Selection end position
:param start_line: Selection start line.
:param end_line: Selection end line.
:param draw_order: The draw order of the selection, highest values will
    appear on top of the lowest values.
:param tooltip: An optional tooltips that will be automatically shown
    when the mouse cursor hover the decoration.
:param full_width: True to select the full line width.
.. note:: Use the cursor selection if startPos and endPos are none.
*/
TextDecoration::TextDecoration(QTextDocument *document, int startPos /*=-1*/, int endPos /*=-1*/, \
    int startLine /*=-1*/, int endLine /*=-1*/, int drawOrder /*=0*/, const QString &tooltip /*= ""*/, \
    bool fullWidth /*= false*/) :
    QTextEdit::ExtraSelection(),
    m_signals(new TextDecorationsSignals()),
    m_drawOrder(drawOrder),
    m_tooltip(tooltip)
{
    this->cursor = QTextCursor(document);

    if (fullWidth)
    {
        setFullWidth(fullWidth);
    }

    if (startPos >= 0)
    {
        this->cursor.setPosition(startPos);
    }

    if (endPos >= 0)
    {
        this->cursor.setPosition(endPos, QTextCursor::KeepAnchor);
    }

    if (startLine >= 0)
    {
        QTextBlock b = document->findBlockByNumber(startLine);

        if (b.isValid())
        {
            cursor.setPosition(b.position(), QTextCursor::MoveAnchor);
        }
        else
        {
            cursor.movePosition(QTextCursor::End, QTextCursor::MoveAnchor);
        }

    }

    if (endLine >= 0)
    {
        QTextBlock b = document->findBlockByNumber(endLine);

        if (b.isValid())
        {
            cursor.setPosition(b.position(), QTextCursor::KeepAnchor);
        }
        else
        {
            cursor.movePosition(QTextCursor::End, QTextCursor::KeepAnchor);
        }
    }
}

//-----------------------------------------------------------
TextDecoration::~TextDecoration()
{
}

//-----------------------------------------------------------
bool TextDecoration::operator==(const TextDecoration &other) const
{
    bool f = (format == other.format);
    return ((cursor == other.cursor) && \
        (m_drawOrder == other.m_drawOrder) && \
        (m_tooltip == other.m_tooltip) && f);
}

//-----------------------------------------------------------
/*
Checks if the textCursor is in the decoration

:param cursor: The text cursor to test
:type cursor: QtGui.QTextCursor
:returns: True if the cursor is over the selection
*/
bool TextDecoration::containsCursor(const QTextCursor &cursor) const
{
    int start = this->cursor.selectionStart();
    int end = this->cursor.selectionEnd();
    if (cursor.atBlockEnd())
    {
        end -= 1;
    }
    return (start <= cursor.position()) && (cursor.position() <= end);
}

//----------------------------------------------------------
/*
Uses bold text
*/
void TextDecoration::setAsBold()
{
    format.setFontWeight(QFont::Bold);
}

//----------------------------------------------------------
/*
Sets the foreground color.

:param color: Color
:type color: QtGui.QColor
*/
void TextDecoration::setForeground(const QColor &color)
{
    format.setForeground(color);
}

//----------------------------------------------------------
/*
Sets the background brush.

:param brush: Brush
:type brush: QtGui.QBrush
*/
void TextDecoration::setBackground(const QBrush &brush)
{
    format.setBackground(brush);
}

//----------------------------------------------------------
/*
Uses an outline rectangle.

:param color: Color of the outline rect
:type color: QtGui.QColor
*/
void TextDecoration::setOutline(const QColor &color)
{
    format.setProperty(QTextFormat::OutlinePen, QPen(color));
}

//----------------------------------------------------------
/*
Select the entire line but starts at the first non whitespace character
and stops at the non-whitespace character.

:return:
*/
void TextDecoration::selectLine()
{
    cursor.movePosition(QTextCursor::StartOfBlock);
    QString text = cursor.block().text();
    int lindent = 0;

    while (lindent < text.size() && text[lindent].isSpace())
    {
        lindent++;
    }

    cursor.setPosition(cursor.block().position() + lindent);
    cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
}

//----------------------------------------------------------
/*
Enables FullWidthSelection (the selection does not stops at after the
character instead it goes up to the right side of the widget).

:param flag: True to use full width selection.
:type flag: bool
:param clear: True to clear any previous selection. Default is True.
:type clear: bool
*/
void TextDecoration::setFullWidth(bool flag /*= true*/, bool clear /*= true*/)
{
    if (clear)
    {
        cursor.clearSelection();
    }

    format.setProperty(QTextFormat::FullWidthSelection, flag);
}

//----------------------------------------------------------
/*
Underlines the text

:param color: underline color.
*/
void TextDecoration::setAsUnderlined(const QColor &color /*= QColor("blue")*/)
{
    format.setUnderlineStyle(QTextCharFormat::SingleUnderline);
    format.setUnderlineColor(color);
}

//----------------------------------------------------------
/*
Underlines text as a spellcheck error.

:param color: Underline color
:type color: QtGui.QColor
*/
void TextDecoration::setAsSpellCheck(const QColor &color /*= QColor("blue")*/)
{
    format.setUnderlineStyle(QTextCharFormat::SpellCheckUnderline);
    format.setUnderlineColor(color);
}

//----------------------------------------------------------
/*
Highlights text as a syntax error.

:param color: Underline color
:type color: QtGui.QColor
*/
void TextDecoration::setAsError(const QColor &color /*= QColor("red")*/)
{
    format.setUnderlineStyle(QTextCharFormat::WaveUnderline);
    format.setUnderlineColor(color);
}

//----------------------------------------------------------
/*
Highlights text as a syntax warning

:param color: Underline color
:type color: QtGui.QColor
*/
void TextDecoration::setAsWarning(const QColor &color /*= QColor("orange")*/)
{
    format.setUnderlineStyle(QTextCharFormat::WaveUnderline);
    format.setUnderlineColor(color);
}


//-----------------------------------------------
void TextDecoration::emitClicked(TextDecoration::Ptr selection) const
{
    emit m_signals->clicked(selection);
}

//-----------------------------------------------
QMetaObject::Connection TextDecoration::connect(const char* signal, QObject *receiver, const char *slot)
{
    return QObject::connect(m_signals.data(), signal, receiver, slot);
}

} //end namespace ito
