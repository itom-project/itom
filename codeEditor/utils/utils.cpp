#include "utils.h"

#include "codeEditor.h"


namespace Utils
{
    //------------------------------------------------------------------------
    /*
    Retuns  a list of symbols found in the block text

    :param editor: code edit instance
    :param block: block to parse
    :param character: character to look for.
    */
    QList<ParenthesisInfo> listSymbols(CodeEditor *editor, const QTextBlock &block, const QChar &character)
    {
        QString text = block.text();
        QList<ParenthesisInfo> symbols;
        QTextCursor cursor(block);
        cursor.movePosition(QTextCursor::StartOfBlock);
        int pos = text.indexOf(character, 0);
        cursor.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor, pos);

        while (pos != -1)
        {
            if (!editor->isCommentOrString(cursor))
            {
                //skips symbols in string literal or comment
                ParenthesisInfo info(pos, character);
                symbols.append(info);
            }
            pos = text.indexOf(character, pos + 1);
            cursor.movePosition(QTextCursor::StartOfBlock);
            cursor.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor, pos);
        }
        return symbols;
    }

    bool sortParenthesisInfo(const ParenthesisInfo &a, const ParenthesisInfo &b)
    {
        return a.position < b.position;
    }

    //-----------------------------------------------------------
    /*
    Gets the list of ParenthesisInfo for specific text block.

    :param editor: Code edit instance
    :param block: block to parse
    */
    void getBlockSymbolData(CodeEditor *editor, const QTextBlock &block, QList<ParenthesisInfo> &parentheses, QList<ParenthesisInfo> &squareBrackets, QList<ParenthesisInfo> &braces)
    {
        parentheses = listSymbols(editor, block, '(') + listSymbols(editor, block, ')');
        qSort(parentheses.begin(), parentheses.end(), sortParenthesisInfo);
        squareBrackets = listSymbols(editor, block, '[') + listSymbols(editor, block, ']');
        qSort(parentheses.begin(), parentheses.end(), sortParenthesisInfo);
        braces = listSymbols(editor, block, '{') + listSymbols(editor, block, '}');
        qSort(parentheses.begin(), parentheses.end(), sortParenthesisInfo);
    }

    //-------------------------------------------------------------
    /*
    Gets the user state, generally used for syntax highlighting.

    :param block: block to access
    :return: The block state
    */
    int TextBlockHelper::getState(const QTextBlock &block)
    {
        if (!block.isValid())
        {
            return -1;
        }
        int state = block.userState();
        if (state == -1)
        {
            return state;
        }
        return state & 0x0000FFFF;
    }

    //-------------------------------------------------------------
    /*
    Sets the user state, generally used for syntax highlighting.

    :param block: block to modify
    :param state: new state value.
    :return:
    */
    void TextBlockHelper::setState(QTextBlock &block, int state)
    {
        if (!block.isValid())
        {
            return;
        }
        int user_state = block.userState();
        if (user_state == -1)
        {
            user_state = 0;
        }
        int higher_part = user_state & 0x7FFF0000;
        state &= 0x0000FFFF;
        state |= higher_part;
        block.setUserState(state);
    }

    //-------------------------------------------------------------
    /*
    Gets the block fold level

    :param block: block to access.
    :returns: The block fold level
    */
    int TextBlockHelper::getFoldLvl(const QTextBlock &block)
    {
        if (!block.isValid())
        {
            return 0;
        }
        int state = block.userState();
        if (state == -1)
        {
            state = 0;
        }
        return (state & 0x03FF0000) >> 16;
    }

    //-------------------------------------------------------------
    /*
    Sets the block fold level.

    :param block: block to modify
    :param val: The new fold level [0-7]
    */
    void TextBlockHelper::setFoldLvl(QTextBlock &block, int val)
    {
        if (!block.isValid())
        {
            return;
        }
        int state = block.userState();
        if (state == -1)
        {
            state = 0;
        }
        if (val >= 0x3FF)
        {
            val = 0x3FF;
        }
        state &= 0x7C00FFFF;
        state |= val << 16;
        block.setUserState(state);
    }

    //-------------------------------------------------------------
    /*
    Checks if the block is a fold trigger.

    :param block: block to check
    :return: True if the block is a fold trigger (represented as a node in
        the fold panel)
    */
    bool TextBlockHelper::isFoldTrigger(const QTextBlock &block)
    {
        if (!block.isValid())
        {
            return false;
        }
        int state = block.userState();
        if (state == -1)
        {
            state = 0;
        }
        return (bool)(state & 0x04000000);
    }

    //-------------------------------------------------------------
    /*
    Set the block fold trigger flag (True means the block is a fold
    trigger).

    :param block: block to set
    :param val: value to set
    */
    void TextBlockHelper::setFoldTrigger(QTextBlock &block, int val)
    {
        if (!block.isValid())
        {
            return;
        }
        int state = block.userState();
        if (state == -1)
        {
            state = 0;
        }
        state &= 0x7BFFFFFF;
        state |= int(val) << 26;
        block.setUserState(state);
    }

    //-------------------------------------------------------------
    /*
     Checks if the block is expanded or collased.

    :param block: QTextBlock
    :return: False for an open trigger, True for for closed trigger
    */
    bool TextBlockHelper::isCollapsed(const QTextBlock &block)
    {
        if (!block.isValid())
        {
            return false;
        }
        int state = block.userState();
        if (state == -1)
        {
            state = 0;
        }
        return (bool)(state & 0x08000000);
    }

    //-------------------------------------------------------------
    /*
    Sets the fold trigger state (collapsed or expanded).

    :param block: The block to modify
    :param val: The new trigger state (True=collapsed, False=expanded)
    */
    void TextBlockHelper::setCollapsed(QTextBlock &block, int val)
    {
        if (!block.isValid())
        {
            return;
        }
        int state = block.userState();
        if (state == -1)
        {
            state = 0;
        }
        state &= 0x77FFFFFF;
        state |= int(val) << 27;
        block.setUserState(state);
    }

    //---------------------------------------------------------------------------
    /*
    Return color that is lighter or darker than the base color.

    If base_color.lightness is higher than 128, the returned color is darker
    otherwise is is lighter.

    :param base_color: The base color to drift from
    ;:param factor: drift factor (%)
    :return A lighter or darker color.
    */
    QColor driftColor(const QColor &baseColor, int factor /*= 110*/)
    {
        if (baseColor.lightness() > 128)
        {
            return baseColor.darker(factor);
        }
        else
        {
            if (baseColor == QColor("#000000"))
            {
                return driftColor(QColor("#101010"), factor + 20);
            }
            else
            {
                return baseColor.lighter(factor + 10);
            }
        }
    }

    //---------------------------------------------------------------------------
    QString lstrip(const QString &string)
    {
        //remove whitespaces from the beginning
        for (int n = 0; n < string.size(); ++n)
        {
            if (!string.at(n).isSpace())
            {
                return string.mid(n);
            }
        }
        return "";
    }

    //---------------------------------------------------------------------------
    QString rstrip(const QString &string)
    {
        //remove whitespaces from the end
        int n = string.size() - 1;
        for (; n >= 0; --n) 
        {
            if (!string.at(n).isSpace()) 
            {
                return string.left(n + 1);
            }
        }
        return "";
    }

    //---------------------------------------------------------------------------
    QString strip(const QString &string)
    {
        return string.trimmed();
    }

    //---------------------------------------------------------------------------
    int numlines(const QString &string)
    {
        int num = 1;
        QString str(string);
        int l = str.size();
        str.replace("\r\n", "");
        num += (l - str.size()) / 2;
        l = str.size();
        str.replace("\r", "");
        str.replace("\n", "");
        num += (l - str.size());
        return num;
    }
};