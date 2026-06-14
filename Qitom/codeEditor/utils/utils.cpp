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

#include "utils.h"

#include "../codeEditor.h"

namespace ito {

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

        if (pos >= 0)
        {
            cursor.setPosition(cursor.position() + pos, QTextCursor::MoveAnchor);
        }

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

            if (pos >= 0)
            {
                cursor.setPosition(cursor.position() + pos, QTextCursor::MoveAnchor);
            }
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
        std::sort(parentheses.begin(), parentheses.end(), sortParenthesisInfo);
        squareBrackets = listSymbols(editor, block, '[') + listSymbols(editor, block, ']');
        std::sort(squareBrackets.begin(), squareBrackets.end(), sortParenthesisInfo);
        braces = listSymbols(editor, block, '{') + listSymbols(editor, block, '}');
        std::sort(braces.begin(), braces.end(), sortParenthesisInfo);
    }

    //-------------------------------------------------------------
    /*
    Gets the user state, generally used for syntax highlighting.

    The user state is limited to the bits 0xFFFF of the
    32bit integer state. The higher level bits 0xFFFF0000
    are used by the fold detection.

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

    The `state` is limited to 0xFFFF and stored in the
    user state of the block. The higher level bits 0xFFFF0000
    are kept and are used by the fold detection.

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
        if (val >= 0x3FF) //maximum fold level
        {
            val = 0x3FF;
        }
        state &= 0x7C00FFFF;
        state |= (val << 16);
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

    //---------------------------------------------------------------------------
    QStringList splitlines(const QString &string)
    {
        QString text = string;
        text.replace("\r\n", "\n");
        text.replace("\r", "\n");
        return text.split("\n");
    }

    QString signatureWordWrapCropString(const QString &str, int totalMaxLineWidth)
    {
        if (totalMaxLineWidth > 0)
        {
            if (str.size() > totalMaxLineWidth)
            {
                return str.left(totalMaxLineWidth - 3) + "...";
            }
        }

        return str;
    }

    //---------------------------------------------------------------------------
    /* Wraps a signature by its arguments, separated by ', ' into
    multiple lines, where each line has a maximum length of 'width'.
    Each following line is indented by four spaces.*/
    QString signatureWordWrap(QString signature, int width, int totalMaxLineWidth /*= -1*/)
    {
        QString result;
        int j, i;
        bool firstWrap = true;

        for (;;)
        {
            i = std::min(width, (int)signature.length());
            j = signature.lastIndexOf(", ", i);

            if (j == -1)
            {
                j = signature.indexOf(", ", i);
            }

            if (j > 0)
            {
                result += signatureWordWrapCropString(signature.left(j), totalMaxLineWidth);
                result += ",\n    ";
                signature = signature.mid(j + 2);

                if (firstWrap)
                {
                    firstWrap = false;
                    width -= 4;
                }
            }
            else
            {
                break;
            }

            if (width >= signature.length())
            {
                break;
            }
        }

        return result + signatureWordWrapCropString(signature, totalMaxLineWidth);
    }

    //---------------------------------------------------------------------------
    /*
    the signature is represented as <code> monospace section.
    this requires much more space than ordinary letters.
    Therefore reduce the maximum line length to 88/2.
    */
    QStringList parseStyledTooltipsFromSignature(
        const QStringList &signatures,
        const QString &docstring,
        int maxLineLength /*= 44*/,
        int maxDocStrLength /*= -1*/)
    {
        QStringList styledTooltips;
        QStringList defs = signatures;

        for (int i = 0; i < defs.size(); ++i)
        {
            if (defs[i].size() > maxLineLength)
            {
                defs[i] = Utils::signatureWordWrap(defs[i], maxLineLength, 3 * maxLineLength);
            }
        }

        // sometimes one element of a signature is still very long. Then cut
        if (defs.size() > 20)
        {
            QStringList newDefs;

            for (int i = 0; i < 10; ++i)
            {
                newDefs << defs[i];
            }

            newDefs << "...";

            for (int i = defs.size() - 10; i < defs.size(); ++i)
            {
                newDefs << defs[i];
            }

            defs = newDefs;
        }

        QString sigs = defs.join("\n");
        sigs = sigs.toHtmlEscaped().replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;");

        // search for the last occurence of ") ->" and replaces the arrow
        // by a real arrow, since the -> arrow will be wrapped (although <nobr>).
        const QString pattern(") -&gt; ");
        int idx = sigs.lastIndexOf(pattern);

        if (idx >= 0)
        {
            sigs = sigs.replace(idx, pattern.size(), ") &#8594; ");
        }

#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
        QStringList s = sigs.split('\n', Qt::SkipEmptyParts);
#else
        QStringList s = sigs.split('\n', QString::SkipEmptyParts);
#endif

        // for unbreakable paragraphs, do not use <nobr>, but <p style=...>
        // see also: https://bugreports.qt.io/browse/QTBUG-1135
        const QString pstart = "<p style=\"white-space:pre; padding-bottom:0px; margin-bottom:0px;\">";
        const QString br = "<br>";

        if (s.size() > 0)
        {
            sigs = pstart + s.join(br) + "</p>";
        }
        else
        {
            sigs = "";
        }

        QString docstr = docstring.toHtmlEscaped().replace('\n', br);

        if (maxDocStrLength > 3 && docstr.size() > maxDocStrLength)
        {
            docstr = docstr.left(maxDocStrLength - 3) + "...";
        }

        if (sigs != "" && docstr != "")
        {
            styledTooltips.append(QString("<code>%1</code><hr>%2%3</p>")
                .arg(sigs).arg(pstart).arg(docstr));
        }
        else if (sigs != "")
        {
            styledTooltips.append(QString("<code>%1</code>").arg(sigs));
        }
        else if (docstr != "")
        {
            styledTooltips.append(QString("%1%2</p>").arg(pstart).arg(docstr));
        }

        return styledTooltips;
    }
};

} //end namespace ito
