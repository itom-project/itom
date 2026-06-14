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

#ifndef SYMBOLMATCHERMODE_H
#define SYMBOLMATCHERMODE_H

/*
This module contains the symbol matcher mode
*/

#include "../textDecoration.h"
#include "../mode.h"
#include "../utils/utils.h"

#include <qcolor.h>
#include <qbrush.h>
#include <qcolor.h>
#include <qtextcursor.h>

namespace ito {

/*
Highlights matching symbols (parentheses, braces,...)

.. note:: This mode requires the document to be filled with
    :class:`pyqode.core.api.TextBlockUserData`, i.e. a
    :class:`pyqode.core.api.SyntaxHighlighter` must be installed on
    the editor instance.
*/
class SymbolMatcherMode : public QObject, public Mode
{
    Q_OBJECT
public:

    // symbols indices
    enum Symbols { Paren = 0, Square = 2, Brace = 4 }; //Position of Symbol in chars bytearray

    enum CharType { Open = 0, Close = 1}; //Sub-Position of Symbol in chars bytearray

    static const QByteArray chars;

    SymbolMatcherMode(const QString &description = "", QObject *parent = NULL);
    virtual ~SymbolMatcherMode();

    QBrush matchBackground() const;
    void setMatchBackground(const QBrush &value);

    QColor matchForeground() const;
    void setMatchForeground(const QColor &value);

    QBrush unmatchBackground() const;
    void setUnmatchBackground(const QBrush &value);

    QColor unmatchForeground() const;
    void setUnmatchForeground(const QColor &value);

    virtual void onStateChanged(bool state);

    QPoint symbolPos(const QTextCursor &cursor, CharType charType = Open, Symbols symbolType =Paren); //!< return value is line (x) and column (y)

private slots:
    void doSymbolsMatching();

protected:
    QTextCursor createDecoration(int pos, bool match = true);
    void clearDecorations();
    void match(Symbols symbol, QList<Utils::ParenthesisInfo> &data, int cursorPos);
    bool matchLeft(SymbolMatcherMode::Symbols symbol, const QTextBlock &currentBlock, int i, int cpt);
    bool matchRight(SymbolMatcherMode::Symbols symbol, const QTextBlock &currentBlock, int i, int nbRightParen);
    void refreshDecorations();

    QBrush m_matchBackground;
    QColor m_matchForeground;
    QBrush m_unmatchBackground;
    QColor m_unmatchForeground;
    QList<TextDecoration::Ptr> m_decorations;
};

} //end namespace ito

#endif
