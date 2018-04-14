#include "symbolMatcherMode.h"

#include "codeEditor.h"
#include "managers/textDecorationsManager.h"
#include "utils/utils.h"

#include <qbrush.h>


SymbolMatcherMode::SymbolMatcherMode(QObject *parent /*= NULL*/) :
    Mode("SymbolMatcherMode"),
    QObject(parent),
    m_decoration(NULL),
    m_pos(-1),
    m_color(QColor())
{
}

//----------------------------------------------------------
/*
*/
SymbolMatcherMode::~SymbolMatcherMode()
{
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

    void SymbolMatcherMode::match(SymbolMatcherMode::Symbols symbol, QList<Utils::ParenthesisInfo> &data, int cursorPos)
    {
        symbols = data[symbol]
        for i, info in enumerate(symbols):
            pos = (editor()->textCursor().position() -
                   editor()->textCursor().block().position())
            if info.character == self.SYMBOLS[symbol][OPEN] and \
                    info.position == pos:
                self._create_decoration(
                    cursor_pos + info.position,
                    self._match_left(
                        symbol, editor()->textCursor().block(), i + 1, 0))
            elif info.character == self.SYMBOLS[symbol][CLOSE] and \
                    info.position == pos - 1:
                self._create_decoration(
                    cursor_pos + info.position,
                    self._match_right(
                        symbol, editor()->textCursor().block(), i - 1, 0))
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
    TextDecoration::Ptr deco(new TextDecoration(cursor, -1, -1, -1 , -1, 10));
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
    m_editor->decorations()->append(deco);
    return cursor;
}