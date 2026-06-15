

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

#ifndef WORDCLICK_H
#define WORDCLICK_H

/*
This module contains the WordClickMode
*/

#include "../mode.h"
#include "../textDecoration.h"
#include "../delayJobRunner.h"
#include <qobject.h>
#include <qstring.h>
#include <qtextcursor.h>
#include <qevent.h>

namespace ito {

/*
Adds support for word click events.

It will highlight the click-able word when the user press control and move
the mouse over a word.

Detecting whether a word is click-able is the responsability of the
subclasses. You must override ``_check_word_cursor`` and call
``_select_word_cursor`` if this is a click-able word (this
process might be asynchrone) otherwise _clear_selection.

:attr:`pyqode.core.modes.WordClickMode.word_clicked` is emitted
when the word is clicked by the user (while keeping control pressed).
*/
class WordClickMode : public QObject, public Mode
{
    Q_OBJECT
public:
    WordClickMode(const QString &name = "WordClickMode", const QString &description = "", QObject *parent = NULL);
    virtual ~WordClickMode();

    virtual void onStateChanged(bool state);

    Qt::KeyboardModifiers wordClickModifiers() const { return m_mouseMoveKeyboardModifiers; }
    void setWordClickModifiers(Qt::KeyboardModifiers modifiers) { m_mouseMoveKeyboardModifiers = modifiers; }

protected:
    void selectWordCursor();
    virtual void clearSelection();
    virtual void checkWordCursor(const QTextCursor &cursor) = 0;
    void addDecoration(const QTextCursor &cursor);
    void removeDecoration();

private:
    int m_previousCursorStart;
    int m_previousCursorEnd;
    DelayJobRunnerBase *m_pTimer;
    TextDecoration::Ptr m_deco;
    QTextCursor m_cursor;

    void emitWordClicked(QTextCursor cursor)
    {
        emit wordClicked(cursor);
    }

    Qt::KeyboardModifiers m_mouseMoveKeyboardModifiers;

private slots:
    void onMouseDoubleClicked(QMouseEvent *e);
    void onMouseMoved(QMouseEvent *e);
    void onMouseReleased(QMouseEvent *e);
    void onKeyReleased(QKeyEvent *e);

signals:
    void wordClicked(const QTextCursor &cursor);


};

} //end namespace ito

#endif
