#include "indenter.h"

#include "codeEditor.h"
#include "utils/utils.h"
#include <qtextdocumentfragment.h>
#include <qdebug.h>


//----------------------------------------------------------
/*
*/
IndenterMode::IndenterMode(const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    QObject(parent),
    Mode("IndenterMode", description)
{
}

//----------------------------------------------------------
/*
*/
IndenterMode::~IndenterMode()
{
}



//----------------------------------------------------------
/*
*/
/*virtual*/ void IndenterMode::onStateChanged(bool state)
{
    if (state)
    {
        connect(editor(), SIGNAL(indentRequested()), this, SLOT(indent()));
        connect(editor(), SIGNAL(unindentRequested()), this, SLOT(unindent()));
    }
    else
    {
        disconnect(editor(), SIGNAL(indentRequested()), this, SLOT(indent()));
        disconnect(editor(), SIGNAL(unindentRequested()), this, SLOT(unindent()));
    }
}

//----------------------------------------------------------
/*
Indent selected text

:param cursor: QTextCursor
*/
void IndenterMode::indentSelection(QTextCursor cursor) const
{
        int tab_len = editor()->tabLength();
        cursor.beginEditBlock();
        int nb_lines = Utils::numlines(cursor.selection().toPlainText());
        QTextCursor c = editor()->textCursor();
        if (c.atBlockStart() && c.position() == c.selectionEnd())
        {
            nb_lines += 1;
        }

        QTextBlock block = editor()->document()->findBlock(cursor.selectionStart());
        int i = 0;
        QTextCursor cursor2;

        QString tab_text = editor()->useSpacesInsteadOfTabs() ? QString(tab_len, ' ') : "\t";
        // indent every lines
        while (i < nb_lines)
        {
            cursor2 = QTextCursor(block);
            cursor2.movePosition(QTextCursor::StartOfLine, QTextCursor::MoveAnchor);
            //qDebug() << cursor2.position() << block.text();
            cursor2.insertText(tab_text);
            block = block.next();
            i += 1;
        }
        cursor.endEditBlock();
}

//----------------------------------------------------------
/*
Un-indents selected text

:param cursor: QTextCursor
*/
QTextCursor IndenterMode::unindentSelection(QTextCursor cursor) const
{
    int tab_len = editor()->tabLength();
    QString txt;
    int indentation;
    QTextCursor c;
    int nb_lines = Utils::numlines(cursor.selection().toPlainText());
    if (nb_lines == 0)
    {
        nb_lines = 1;
    }
    QTextBlock block = editor()->document()->findBlock(cursor.selectionStart());
    int i = 0;
    //debug('unindent selection: %d lines', nb_lines)
    while (i < nb_lines)
    {
        txt = block.text();
        //debug('line to unindent: %s', txt)
        //debug('editor()->use_spaces_instead_of_tabs: %r',
        //      editor()->use_spaces_instead_of_tabs)
        if (editor()->useSpacesInsteadOfTabs())
        {
            indentation = (txt.size() - Utils::lstrip(txt).size());
        }
        else
        {
            indentation = txt.size() - txt.replace("\t", "").size();
        }
        //debug('unindent line %d: %d spaces', i, indentation)
        if (indentation > 0)
        {
            c = QTextCursor(block);
            c.movePosition(QTextCursor::StartOfLine, QTextCursor::MoveAnchor);
            for (int j = 0; j < tab_len; ++j)
            {
                txt = block.text();
                if (txt.size() && txt[0] == ' ')
                {
                    c.deleteChar();
                }
            }
        }
        block = block.next();
        i += 1;
    }
    return cursor;
}

//----------------------------------------------------------
/*
Indents text at cursor position.
*/
void IndenterMode::indent() const
{
    QTextCursor cursor = editor()->textCursor();
    //assert isinstance(cursor, QTextCursor)
    if (cursor.hasSelection())
    {
        indentSelection(cursor);
    }
    else
    {
        // simply insert indentation at the cursor position
        int tab_len = editor()->tabLength();
        cursor.beginEditBlock();
        if (editor()->useSpacesInsteadOfTabs())
        {
            int nb_space_to_add = tab_len - cursor.positionInBlock() % tab_len;
            cursor.insertText(QString(nb_space_to_add, ' '));
        }
        else
        {
            cursor.insertText("\t");
        }
        cursor.endEditBlock();
    }
}

//----------------------------------------------------------
/*
Un-indents text at cursor position.
*/
void IndenterMode::unindent() const
{
        QTextCursor cursor = editor()->textCursor();
        //debug('cursor has selection %r', cursor.hasSelection())
        if (cursor.hasSelection())
        {
            cursor.beginEditBlock();
            unindentSelection(cursor);
            cursor.endEditBlock();
            editor()->setTextCursor(cursor);
        }
        else
        {
            int tab_len = editor()->tabLength();
            int indentation = cursor.positionInBlock();
            int max_spaces = tab_len - (indentation - (indentation % tab_len));
            int spaces = countDeletableSpaces(cursor, max_spaces);
            //debug('deleting %d space before cursor' % spaces)
            cursor.beginEditBlock();
            if (spaces)
            {
                // delete spaces before cursor
                for (int i = 0; i < spaces; ++i)
                {
                    cursor.deletePreviousChar();
                }
            }
            else
            {
                // un-indent whole line
                //debug('un-indent whole line')
                cursor = unindentSelection(cursor);
            }
            cursor.endEditBlock();
            editor()->setTextCursor(cursor);
            //debug(cursor.block().text());
        }
}

//----------------------------------------------------------
/*
*/
int IndenterMode::countDeletableSpaces(const QTextCursor &cursor, int maxSpaces) const
{
    // count the number of spaces deletable, stop at tab len
    int max_spaces = std::abs(maxSpaces);
    max_spaces = std::max(max_spaces, editor()->tabLength());
    int spaces = 0;
    QTextCursor trav_cursor = QTextCursor(cursor);
    int pos;
    QString c;
    while ((spaces < max_spaces) || trav_cursor.atBlockStart())
    {
        pos = trav_cursor.position();
        trav_cursor.movePosition(QTextCursor::Left, QTextCursor::KeepAnchor);
        c = trav_cursor.selectedText();
        if (c == " ")
        {
            spaces += 1;
        }
        else
        {
            break;
        }
        trav_cursor.setPosition(pos - 1);
    }
    return spaces;
}
