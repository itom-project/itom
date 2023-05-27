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

#ifndef UTILS_H
#define UTILS_H

#include <qcolor.h>
#include <qtextedit.h>
#include <qchar.h>

namespace ito {

class CodeEditor;

namespace Utils
{
    /*
    Stores information about a parenthesis in a line of code.
    */
    class ParenthesisInfo
    {
    public:
        ParenthesisInfo(int pos, const QChar &chara) :
            position(pos),
            character(chara)
        {
            //: Position of the parenthesis, expressed as a number of character
            //: The parenthesis character, one of "(", ")", "{", "}", "[", "]"
        }

        int position;
        QChar character;
    };

    /*
    Helps retrieving the various part of the user state bitmask.

    This helper should be used to replace calls to

    ``QTextBlock.setUserState``/``QTextBlock.getUserState`` as well as
    ``QSyntaxHighlighter.setCurrentBlockState``/
    ``QSyntaxHighlighter.currentBlockState`` and
    ``QSyntaxHighlighter.previousBlockState``.

    The bitmask is made up of the following fields:

        - bit0 -> bit26: User state (for syntax highlighting)
        - bit26: fold trigger state
        - bit27-bit29: fold level (8 level max)
        - bit30: fold trigger flag
        - bit0 -> bit15: 16 bits for syntax highlighter user state (
          for syntax highlighting)
        - bit16-bit25: 10 bits for the fold level (1024 levels)
        - bit26: 1 bit for the fold trigger flag (trigger or not trigger)
        - bit27: 1 bit for the fold trigger state (expanded/collapsed)
    */
    class TextBlockHelper
    {
    public:
        static int getState(const QTextBlock &block);
        static void setState(QTextBlock &block, int state);
        static int getFoldLvl(const QTextBlock &block);
        static void setFoldLvl(QTextBlock &block, int val);
        static bool isFoldTrigger(const QTextBlock &block);
        static void setFoldTrigger(QTextBlock &block, int val);
        static bool isCollapsed(const QTextBlock &block);
        static void setCollapsed(QTextBlock &block, int val);
    };

    /*
    Return color that is lighter or darker than the base color.*/
    QColor driftColor(const QColor &baseColor, int factor = 110);

    QList<ParenthesisInfo> listSymbols(CodeEditor *editor, const QTextBlock &block, const char* character);
    void getBlockSymbolData(CodeEditor *editor, const QTextBlock &block, QList<ParenthesisInfo> &parentheses, QList<ParenthesisInfo> &squareBrackets, QList<ParenthesisInfo> &braces);

    QString lstrip(const QString &string);
    QString rstrip(const QString &string);
    QString strip(const QString &string);
    int numlines(const QString &string);
    QStringList splitlines(const QString &string);
    QString signatureWordWrap(QString signature, int width, int totalMaxLineWidth = -1);
    QStringList parseStyledTooltipsFromSignature(const QStringList &signatures, const QString &docstring, int maxLineLength = 44, int maxDocStrLength = -1);
};

} //end namespace ito

#endif
