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

#ifndef PYAUTOINDENT_H
#define PYAUTOINDENT_H

/*
Contains python smart indent modes
*/

#include "autoindent.h"
#include "../utils/utils.h"

#include <qtextcursor.h>
#include <qstring.h>
#include <qregularexpression.h>

namespace ito {


/*
Automatically indents text, respecting the PEP8 conventions.

Customised :class:`pyqode.core.modes.AutoIndentMode` for python
that tries its best to follow the pep8 indentation guidelines.
*/
class PyAutoIndentMode : public AutoIndentMode
{
    Q_OBJECT
public:
    PyAutoIndentMode(const QString &description = "", QObject *parent = NULL);
    virtual ~PyAutoIndentMode();

    virtual void onInstall(CodeEditor *editor);

private slots:

protected:
    virtual QPair<QString, QString> getIndent(const QTextCursor &cursor) const;

    QPair<QString, QString> handleIndentBetweenParen(int column, const QString &line, const QPair<QString, QString> &parent_impl, const QTextCursor &cursor) const;
    void handleIndentInsideString(const QChar &c, const QTextCursor &cursor, const QString &fullline, QString &post, QString &pre) const;
    QString handleNewScopeIndentation(const QTextCursor &cursor, const QString &fullline) const;
    void handleIndentInStatement(const QString &fullline, const QString &lastword, QString &post, QString &pre) const;
    void handleIndentAfterParen(const QTextCursor &cursor, QString &post) const;
    bool betweenParen(const QTextCursor &cursor, int column) const;
    QPair<int, QChar> getFirstOpenParen(const QTextCursor &tc, int column) const;
    void getParenPos(const QTextCursor &cursor, int column, int &ol, int &oc, int &cl, int &cc) const;
    void parensCountForBlock(int column, const QTextBlock &block, int &numOpenParentheses, int &numClosedParentheses) const;
    int getIndentOfOpeningParen(const QTextCursor &cursor) const;
    bool checkKwInLine(const QStringList &kwds, const QString &lparam) const;

    static QPair<bool, QChar> isInStringDef(const QString &fullline, int column);
    static bool isParenOpen(const Utils::ParenthesisInfo &paren);
    static bool isParenClosed(const Utils::ParenthesisInfo &paren);
    static QString getFullLine(const QTextCursor &cursor);
    static QString getLastWordUnstripped(const QTextCursor &cursor);
    static QString getLastWord(const QTextCursor &cursor);
    static QChar getPrevChar(const QTextCursor &cursor);
    static QChar getNextChar(const QTextCursor &cursor);
    static bool atBlockEnd(const QTextCursor &cursor, const QString &fullline);
    static bool atBlockStart(const QTextCursor &cursor, const QString &line);

    static QStringList newScopeKeywords;

};

} //end namespace ito

#endif
