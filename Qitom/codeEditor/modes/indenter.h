

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

#ifndef INDENTER_H
#define INDENTER_H

/*
Contains the default indenter.
*/

#include "../mode.h"
#include <qobject.h>
#include <qstring.h>
#include <qtextcursor.h>

namespace ito {

/*
Implements classic indentation/tabulation (Tab/Shift+Tab)

It inserts/removes tabulations (a series of spaces defined by the
tabLength settings) at the cursor position if there is no selection,
otherwise it fully indents/un-indents selected lines.

To trigger an indentation/un-indentation programatically, you must emit
:attr:`pyqode.core.api.CodeEdit.indent_requested` or
:attr:`pyqode.core.api.CodeEdit.unindent_requested`.
*/
class IndenterMode : public QObject, public Mode
{
    Q_OBJECT
public:
    IndenterMode(const QString &description = "", QObject *parent = NULL);
    virtual ~IndenterMode();

    virtual void onStateChanged(bool state);

    void indentSelection(QTextCursor cursor) const;
    QTextCursor unindentSelection(QTextCursor cursor) const;

    int countDeletableSpaces(const QTextCursor &cursor, int maxSpaces) const;

private slots:
    void indent() const;
    void unindent() const;

protected:


};

} //end namespace ito

#endif
