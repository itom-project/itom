#include "pyAutoIndent.h"

#include "../codeEditor.h"
#include "../modes/symbolMatcherMode.h"
#include "../managers/modesManager.h"


PyAutoIndentMode::PyAutoIndentMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    AutoIndentMode("PyAutoIndentMode", description, parent)
{
}

//----------------------------------------------------------
/*
*/
PyAutoIndentMode::~PyAutoIndentMode()
{
}


//----------------------------------------------------------
/*
*/
void PyAutoIndentMode::onInstall(CodeEditor *editor)
{
    AutoIndentMode::onInstall(editor);   
}

//----------------------------------------------------------
/*
Return the indentation text (a series of spaces or tabs)

:param cursor: QTextCursor

:returns: Tuple (text before new line, text after new line)
*/
QPair<QString, QString> PyAutoIndentMode::getIndent(const QTextCursor &cursor) const
{
    int ln, column;
    editor()->cursorPosition(ln, column);
    QString fullline = getFullLine(cursor);
    QString line = fullline.left(column - 1); //todo: maybe column and not column - 1
    QPair<QString, QString> pre_post =AutoIndentMode::getIndent(cursor);
    
    if (atBlockStart(cursor, line))
    {
        return pre_post;
    }

    QString pre = pre_post.first;
    QString post = pre_post.second;

    // return pressed in comments
    QTextCursor c2(cursor);
    if (c2.atBlockEnd())
    {
        c2.movePosition(QTextCursor::Left);
    }
    QList<ColorScheme::Keys> formats;
    formats << ColorScheme::KeyComment << ColorScheme::KeyDocstring;
    if (editor()->isCommentOrString(c2, formats) || \
            fullline.endsWith(("\"\"\"", "'''")))
    {
        if (Utils::strip(line).startsWith("#") && (column != fullline.size()))
        {
            post += "# ";
        }
        return QPair<QString, QString>(pre, post);

    }
    // between parens
    else if (betweenParen(cursor, column))
    {
        return handleIndentBetweenParen(column, line, QPair<QString, QString>(pre, post), cursor);
    } 
    else 
    {
        QString lastword = getLastWord(cursor);
        QString lastwordu = getLastWordUnstripped(cursor);
        bool end_with_op = fullline.endsWith("+") || \
            fullline.endsWith("-") || \
            fullline.endsWith("*") || \
            fullline.endsWith("/") || \
            fullline.endsWith("=") || \
            fullline.endsWith(" &&") || \
            fullline.endsWith(" ||") || \
            fullline.endsWith("%");

        QPair<bool, QChar> temp = isInStringDef(fullline, column);
        bool in_string_def = temp.first;
        QChar c =temp.second;
        if (in_string_def)
        {
            handleIndentInsideString(c, cursor, fullline, post, pre);
        }
        else if (Utils::rstrip(fullline).endsWith(":") &&
                Utils::rstrip(lastword).endsWith(':') &&
                atBlockEnd(cursor, fullline))
        {
            post = handleNewScopeIndentation(cursor, fullline);
        }
        else if (line.endsWith("\\"))
        {
            // if (user typed \ && press enter -> indent is always
            // one level higher
            post += QString(editor()->tabLength(), ' ');
        }
        else if ((fullline.endsWith((')', '}', ']')) &&
                lastword.endsWith((')', '}', ']'))))
        {
            handleIndentAfterParen(cursor, post);
        }
        else if (!fullline.endsWith("\\") &&
                (end_with_op || !atBlockEnd(cursor, fullline)))
        {
            handleIndentInStatement(fullline, lastwordu, post, pre);
        }
        else if ((atBlockEnd(cursor, fullline) &&
                Utils::strip(fullline).startsWith("return")) ||
                lastword == "pass")
        {
            post.chop(editor()->tabLength());
        }
    }
    return QPair<QString, QString>(pre, post);
}

//---------------------------------------------------------------------------
void PyAutoIndentMode::parensCountForBlock(int column, const QTextBlock &block, int &numOpenParentheses, int &numClosedParentheses) const
{
    QList<Utils::ParenthesisInfo> parentheses, squareBrackets, braces, all;
    
    Utils::getBlockSymbolData(editor(), block, parentheses, squareBrackets, braces);
    all = parentheses + squareBrackets + braces;

    numOpenParentheses = 0;
    numClosedParentheses = 0;

    foreach (const Utils::ParenthesisInfo &paren, all)
    {
        if (paren.position >= column)
        {
            continue;
        }
        if (isParenOpen(paren))
        {
            if (column < 0)
            {
                numOpenParentheses = -1;
                numOpenParentheses = -1;
                return;
            }
            numOpenParentheses++;
        }
        if (isParenClosed(paren))
        {
            numClosedParentheses++;
        }
    }
}


//---------------------------------------------------------------------------
bool PyAutoIndentMode::betweenParen(const QTextCursor &cursor, int column) const
{
    Mode::Ptr symbolMatcherMode = editor()->modes()->get("SymbolMatcherMode");
    if (!symbolMatcherMode)
    {
        return false;
    }

    QTextBlock block = cursor.block();
    int nb_open = 0;
    int nb_closed = 0;
    int o, c;

    while (block.isValid() && (block.text().trimmed() != ""))
    {
        parensCountForBlock(column, block, o, c);
        nb_open += o;
        nb_closed += c;
        block = block.previous();
        column = block.text().size();
    }
    return nb_open > nb_closed;
}

//---------------------------------------------------------------
int PyAutoIndentMode::getIndentOfOpeningParen(const QTextCursor &cursor) const
{
    QTextCursor cursor2(cursor);
    cursor2.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor);
    QChar character = cursor.selectedText()[0];
    cursor2.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor);

    QMap<QChar, QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols> > mapping;
    mapping[')'] = QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols>(SymbolMatcherMode::Open, SymbolMatcherMode::Paren);
    mapping[']'] = QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols>(SymbolMatcherMode::Open, SymbolMatcherMode::Square);
    mapping['}'] = QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols>(SymbolMatcherMode::Open, SymbolMatcherMode::Brace);
    
    if (mapping.contains(character))
    {
        Mode::Ptr symMatcherModePtr = editor()->modes()->get("SymbolMatcherMode");
        SymbolMatcherMode *symMatcherMode = symMatcherModePtr ? dynamic_cast<SymbolMatcherMode*>(symMatcherModePtr.data()) : NULL;

        if (symMatcherMode)
        {
            QPoint pt = symMatcherMode->symbolPos(cursor2, mapping[character].first, mapping[character].second);
            int ol = pt.x();
            //int oc = pt.y();
            QString line = editor()->lineText(ol);
            return (line.size() - Utils::lstrip(line).size());
        }
        return -2;
    }
    else
    {
        return -1;
    }
}

//---------------------------------------------------------------
bool cmpParenthesisByPosReversed(const Utils::ParenthesisInfo &a, const Utils::ParenthesisInfo &b)
{
    return a.position <= b.position; //todo: verify
}


QPair<int, QChar> PyAutoIndentMode::getFirstOpenParen(const QTextCursor &cursor, int column) const
{
    int pos = -1; //None
    QChar character;
    int ln = cursor.blockNumber();
    QTextCursor tc_trav(cursor);
    QMap<QChar, QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols> > mapping;
    mapping['('] = QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols>(SymbolMatcherMode::Close, SymbolMatcherMode::Paren);
    mapping['['] = QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols>(SymbolMatcherMode::Close, SymbolMatcherMode::Square);
    mapping['{'] = QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols>(SymbolMatcherMode::Close, SymbolMatcherMode::Brace);

    QList<Utils::ParenthesisInfo> parentheses, squareBrackets, braces, all_symbols;
    QPair<SymbolMatcherMode::CharType, SymbolMatcherMode::Symbols> mappingItem;
    SymbolMatcherMode::CharType ch;
    SymbolMatcherMode::Symbols ch_type;

    Mode::Ptr symMatcherModePtr = editor()->modes()->get("SymbolMatcherMode");
    SymbolMatcherMode *symMatcherMode = symMatcherModePtr ? dynamic_cast<SymbolMatcherMode*>(symMatcherModePtr.data()) : NULL;
    int l, c;

    while (ln >= 0 && (Utils::strip(cursor.block().text()) != ""))
    {
        tc_trav.movePosition(QTextCursor::StartOfLine, QTextCursor::MoveAnchor);
        Utils::getBlockSymbolData(editor(), tc_trav.block(), parentheses, squareBrackets, braces);
        all_symbols = parentheses + squareBrackets + braces;
        std::sort(all_symbols.begin(), all_symbols.end(), cmpParenthesisByPosReversed);

        foreach (const Utils::ParenthesisInfo &paren, all_symbols)
        {
            if (paren.position < column)
            {
                if (isParenOpen(paren))
                {
                    if (paren.position > column)
                    {
                        continue;
                    } 
                    else 
                    {
                        pos = tc_trav.position() + paren.position;
                        character = paren.character;
                        // ensure it does not have a closing paren on
                        // the same line
                        QTextCursor tc3(cursor);
                        tc3.setPosition(pos);

                        if (mapping.contains(paren.character) && symMatcherMode)
                        {
                            mappingItem = mapping[paren.character];
                            ch = mappingItem.first;
                            ch_type = mappingItem.second;
                            QPoint pt = symMatcherMode->symbolPos(tc3, ch, ch_type);
                            l = pt.x();
                            c = pt.y();
                        }
                        else
                        {
                            continue;
                        }

                        if ((l == ln) && (c < column))
                        {
                            continue;
                        }
                        return QPair<int, QChar>(pos, character);
                    }
                }
            }
        }
        // check previous line
        tc_trav.movePosition(QTextCursor::Up, QTextCursor::MoveAnchor);
        ln = tc_trav.blockNumber();
        column = editor()->lineText(ln).size();
    }

    return QPair<int, QChar>(pos, character);
}

//-------------------------------------------------------------------------
void PyAutoIndentMode::getParenPos(const QTextCursor &cursor, int column, int &ol, int &oc, int &cl, int &cc) const
{
    QPair<int, QChar> pos_char = getFirstOpenParen(cursor, column);
    Mode::Ptr symMatcherModePtr = editor()->modes()->get("SymbolMatcherMode");
    SymbolMatcherMode *symMatcherMode = symMatcherModePtr ? dynamic_cast<SymbolMatcherMode*>(symMatcherModePtr.data()) : NULL;

    QMap<QChar, SymbolMatcherMode::Symbols> mapping;
    mapping['('] = SymbolMatcherMode::Paren;
    mapping['['] = SymbolMatcherMode::Square;
    mapping['{'] = SymbolMatcherMode::Brace;

    QTextCursor tc2(cursor);
    tc2.setPosition(pos_char.first);
    QPoint ol_oc = symMatcherMode->symbolPos(tc2, SymbolMatcherMode::Open, mapping[pos_char.second]);
    QPoint cl_cc = symMatcherMode->symbolPos(tc2, SymbolMatcherMode::Close, mapping[pos_char.second]);
    ol = ol_oc.x();
    oc = ol_oc.y();
    cl = cl_cc.x();
    cc = cl_cc.y();
}

//---------------------------------------------------------------------------
/*
Handle indent between symbols such as parenthesis, braces,...
*/
QPair<QString, QString> PyAutoIndentMode::handleIndentBetweenParen(int column, const QString &line, const QPair<QString, QString> &parent_impl, const QTextCursor &cursor) const
{
    QString pre = parent_impl.first;
    QString post = parent_impl.second;

    QChar next_char = getNextChar(cursor);
    QChar prev_char = getPrevChar(cursor);
    bool prev_open = QString("[({").contains(prev_char);
    bool next_close = QString("])}").contains(next_char);

    int open_line, open_symbol_col, close_line, close_col;
    getParenPos(cursor, column, open_line, open_symbol_col, close_line, close_col);
    QString open_line_txt = editor()->lineText(open_line);
    int open_line_indent = open_line_txt.size() - Utils::lstrip(open_line_txt).size();
    if (prev_open)
    {
        post = QString((open_line_indent + editor()->tabLength()), ' ');
    }
    else if (next_close && (prev_char != ','))
    {
        post = QString(open_line_indent, ' ');
    }
    else if (cursor.block().blockNumber() == open_line)
    {
        post = QString(open_symbol_col, ' ');
    }

    // adapt indent if cursor on closing line and next line have same
    // indent -> PEP8 compliance
    if (close_line && close_col)
    {
        QString txt = editor()->lineText(close_line);
        int bn = cursor.block().blockNumber();
        bool flg = (bn == close_line);
        QString next_indent = QString(editor()->lineIndent(bn + 1), ' ');
        if (flg && Utils::strip(txt).endsWith(':') && (next_indent == post))
        {
            // | look at how the previous line ( ``':'):`` ) was
            // over-indented, this is actually what we are trying to
            // achieve here
            post += QString(editor()->tabLength(), ' ');
        }
    }

    QTextCursor cursor2(cursor);

    // breaking string
    if (QString("\"'").contains(next_char))
    {
        cursor2.movePosition(QTextCursor::Left);
    }

    QList<ColorScheme::Keys> formats;
    formats << ColorScheme::KeyString;

    bool is_string = editor()->isCommentOrString(cursor, formats);
    if (QString("\"'").contains(next_char))
    {
        cursor2.movePosition(QTextCursor::Right);
    }
    if (is_string)
    {
        QTextCursor trav = QTextCursor(cursor);
        
        while (editor()->isCommentOrString(trav, formats))
        {
            trav.movePosition(QTextCursor::Left);
        }
        trav.movePosition(QTextCursor::Right);
        QString symbol = getNextChar(trav);
        pre += symbol;
        post += symbol;
    }

    return QPair<QString,QString>(pre, post);
}

//---------------------------------------------------------------------------
void PyAutoIndentMode::handleIndentInsideString(const QString &c, const QTextCursor &cursor, const QString &fullline, QString &post, QString &pre) const
{
    // break string with a '\' at the end of the original line, always
    // breaking strings enclosed by parens is done in the
    // _handle_between_paren method
    int n = editor()->tabLength();
    pre = QString("%1 \\").arg(c);
    post += QString(n, ' ');
    if (fullline.endsWith(':'))
    {
        post += QString(n, ' ');
    }
    post += c;
}

//----------------------------------------------------------------------------
bool checkKwInLine(const QStringList &kwds, const QString &lparam)
{
    foreach (const QString &kw, kwds)
    {
        if (lparam.contains(kw, Qt::CaseSensitive))
        {
            return true;
        }
    }
    return false;
}


//----------------------------------------------------------------------------
QString PyAutoIndentMode::handleNewScopeIndentation(const QTextCursor &cursor, const QString &fullline) const
{
    QString post;

    int indent = getIndentOfOpeningParen(cursor);
    if (indent >= 0)
    {
        indent += editor()->tabLength();
        post = QString(indent, ' ');
    }
    else
    {
        // e.g indent is None (meaning the line does not ends with ):, ]:
        // or }:
        QStringList kw;
        kw << "if" << "class" << "def" << "while" << "for" << \
              "else" << "elif" << "except" << "finally" << "try" << "with";
        QString l = fullline;
        int ln = cursor.blockNumber();

        while (!checkKwInLine(kw, l) && ln > 0)
        {
            ln -= 1;
            l = editor()->lineText(ln);
        }

        QString indentStr = QString((l.size() - Utils::strip(l).size()), ' ');
        indentStr += QString(editor()->tabLength(), ' ');
        post = indentStr;
    }
    return post;
}

//----------------------------------------------------------------------------
void PyAutoIndentMode::handleIndentInStatement(const QString &fullline, const QString &lastword, QString &post, QString &pre) const
{
    if (lastword[-1] != ':')
    {
        if (lastword != "" && lastword.right(0) != " ")
        {
            pre += " \\";
        }
        else
        {
            pre += '\\';
        }
        post += QString(editor()->tabLength(), ' ');
        if (fullline.endsWith(':'))
        {
            post += QString(editor()->tabLength(), ' ');
        }
    }
}

//----------------------------------------------------------------------------
void PyAutoIndentMode::handleIndentAfterParen(const QTextCursor &cursor, QString &post) const
{
    int indent = getIndentOfOpeningParen(cursor);
    if (indent >= 0)
    {
        post = QString(indent, ' ');
    }
}

//----------------------------------------------------------------------------
/*static*/ QPair<bool, QChar> PyAutoIndentMode::isInStringDef(const QString &fullline, int column)
{
    int count = 0;
    QChar c = '\'';
    for (int i = 0; i < fullline.size(); ++i)
    {
        if (fullline[i] == '\'' || fullline[i] == '"')
        {
            count += 1;
        }
        if (fullline[i] == '"' && (i < column))
        {
            c = '"';
        }
    }
    int count_after_col = 0;
    for (int i = column; i < fullline.size(); ++i)
    {
        if (fullline[i] == '\'' || fullline[i] == '"')
        {
            count_after_col += 1;
        }
    }
    return QPair<bool, QChar>(((count % 2) == 0) && \
        ((count_after_col % 2) == 1), c);
}

//----------------------------------------------------------------------------
/*static*/ bool PyAutoIndentMode::isParenOpen(const Utils::ParenthesisInfo &paren)
{
    return (paren.character == '(' || paren.character == '[' || paren.character == '{');
}

//----------------------------------------------------------------------------
/*static*/ bool PyAutoIndentMode::isParenClosed(const Utils::ParenthesisInfo &paren)
{
    return (paren.character == ')' || paren.character == ']' || paren.character == '}');
}

//----------------------------------------------------------------------------
/*static*/ QString PyAutoIndentMode::getFullLine(const QTextCursor &cursor)
{
    QTextCursor tc2(cursor);
    tc2.select(QTextCursor::LineUnderCursor);
    QString full_line = tc2.selectedText();
    return full_line;
}

//----------------------------------------------------------------------------
/*static*/ QString PyAutoIndentMode::getLastWordUnstripped(const QTextCursor &cursor)
{
    QTextCursor tc2 = QTextCursor(cursor);
    tc2.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, 1);
    tc2.movePosition(QTextCursor::WordLeft, QTextCursor::KeepAnchor);
    return Utils::strip(tc2.selectedText());
}

//----------------------------------------------------------------------------
/*static*/ QString PyAutoIndentMode::getLastWord(const QTextCursor &cursor)
{
    QTextCursor tc2 = QTextCursor(cursor);
    tc2.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor, 1);
    tc2.movePosition(QTextCursor::WordLeft, QTextCursor::KeepAnchor);
    return Utils::strip(tc2.selectedText());
}

//----------------------------------------------------------------------------
/*static*/ QChar PyAutoIndentMode::getPrevChar(const QTextCursor &cursor)
{
    QTextCursor tc2 = QTextCursor(cursor);
    tc2.movePosition(QTextCursor::PreviousCharacter, QTextCursor::KeepAnchor);
    QCharRef c = tc2.selectedText()[0];
    while (c == ' ')
    {
        tc2.movePosition(QTextCursor::PreviousCharacter, QTextCursor::KeepAnchor);
        c = tc2.selectedText()[0];
    }
    return c; //TODO: check this //Utils::strip(c)[0];
}

//----------------------------------------------------------------------------
/*static*/ QChar PyAutoIndentMode::getNextChar(const QTextCursor &cursor)
{
    QTextCursor tc2 = QTextCursor(cursor);
    tc2.movePosition(QTextCursor::NextCharacter, QTextCursor::KeepAnchor);
    return tc2.selectedText()[0];
}

//----------------------------------------------------------------------------
/* Improve QTextCursor.atBlockStart to ignore spaces
*/
/*static*/ bool PyAutoIndentMode::atBlockEnd(const QTextCursor &cursor, const QString &fullline)
{
    if (cursor.atBlockEnd())
    {
        return true;
    }
    int column = cursor.columnNumber();
    return column >= (Utils::rstrip(fullline).size() - 1);
}

//----------------------------------------------------------------------------
/*static*/ bool PyAutoIndentMode::atBlockStart(const QTextCursor &cursor, const QString &line)
{
    if (cursor.atBlockStart())
    {
        return true;
    }
    int column = cursor.columnNumber();
    int indentation = line.size() - Utils::lstrip(line).size();
    return column <= indentation;
}