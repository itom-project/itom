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

#include "autoindent.h"

#include "../codeEditor.h"

#include <qdebug.h>

namespace ito {

AutoIndentMode::AutoIndentMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent),
    m_keyPressedModifiers(),
    m_autoStripTrailingSpacesAfterReturn(false),
    m_enableAutoIndent(true)
{
}

//----------------------------------------------------------
/*
*/
AutoIndentMode::~AutoIndentMode()
{
}


//----------------------------------------------------------
/*
*/
void AutoIndentMode::onStateChanged(bool state)
{
    if (state &&
        (m_enableAutoIndent || m_autoStripTrailingSpacesAfterReturn))
    {
        // maybe the connection already exists.
        connect(
            editor(), &CodeEditor::keyPressed,
            this, &AutoIndentMode::onKeyPressed,
            Qt::UniqueConnection);
    }
    else
    {
        disconnect(
            editor(), &CodeEditor::keyPressed,
            this, &AutoIndentMode::onKeyPressed);
    }
}

//----------------------------------------------------------
/*
Return the indentation text (a series of spaces or tabs)

:param cursor: QTextCursor

:returns: Tuple (text before new line, text after new line)
*/
QPair<QString, QString> AutoIndentMode::getIndent(const QTextCursor &cursor) const
{
    QString indent = QString(editor()->lineIndent(-1), indentChar());
    return QPair<QString,QString>("", indent);
}

//---------------------------------------------------------
QChar AutoIndentMode::indentChar() const
{
    if (editor()->useSpacesInsteadOfTabs())
    {
        return ' ';
    }
    else
    {
        return '\t';
    }
}

//---------------------------------------------------------
QString AutoIndentMode::singleIndent() const
{
    if (editor()->useSpacesInsteadOfTabs())
    {
        return QString(editor()->tabLength(), ' ');
    }
    else
    {
        return "\t";
    }
}

//---------------------------------------------------------
void AutoIndentMode::setKeyPressedModifiers(Qt::KeyboardModifiers modifiers)
{
    m_keyPressedModifiers = modifiers;
}

//---------------------------------------------------------
Qt::KeyboardModifiers AutoIndentMode::keyPressedModifiers() const
{
    return m_keyPressedModifiers;
}

//---------------------------------------------------------
void AutoIndentMode::setAutoStripTrailingSpacesAfterReturn(bool strip)
{
    m_autoStripTrailingSpacesAfterReturn = strip;

    onStateChanged(enabled());
}

//---------------------------------------------------------
bool AutoIndentMode::autoStripTrailingSpacesAfterReturn() const
{
    return m_autoStripTrailingSpacesAfterReturn;
}

//---------------------------------------------------------
void AutoIndentMode::enableAutoIndent(bool autoIndent)
{
    m_enableAutoIndent = autoIndent;

    onStateChanged(enabled());
}

//---------------------------------------------------------
bool AutoIndentMode::isAutoIndentEnabled() const
{
    return m_enableAutoIndent;
}

//----------------------------------------------------------
/*
Auto indent if the released key is the return key.
:param event: the key event
*/
void AutoIndentMode::onKeyPressed(QKeyEvent *e)
{
    if (!e->isAccepted())
    {
        //if Key_Enter on keypad is pressed, KeypadModifier is set, too --> ignore it
        if ((e->modifiers() & (~Qt::KeypadModifier)) == m_keyPressedModifiers && \
            ((e->key() == Qt::Key_Return) || (e->key() == Qt::Key_Enter)))
        {
            QTextCursor cursor = editor()->textCursor();
            int pos = cursor.position();
            cursor.beginEditBlock();

            if (m_enableAutoIndent)
            {
                QPair<QString, QString> pre_post = getIndent(cursor);

                cursor.insertText(QString("%1\n%2").arg(pre_post.first, pre_post.second));

                //eats possible whitespaces
                cursor.movePosition(QTextCursor::WordRight, QTextCursor::KeepAnchor);
                QString txt = cursor.selectedText();

                if (txt.startsWith(" "))
                {
                    QString new_txt = txt.replace(" ", "");

                    if (txt.size() > new_txt.size())
                    {
                        cursor.insertText(new_txt);
                    }
                }
            }
            else
            {
                cursor.insertText("\n");  // the line break must always be set.
            }

            // check if the line, where the original cursor is in,
            // contains trailing spaces or tabs. If so, remove them.
            // PEP8: empty lines should not contain any whitespaces
            if (m_autoStripTrailingSpacesAfterReturn)
            {
                cursor.setPosition(pos, QTextCursor::MoveAnchor);
                cursor.movePosition(QTextCursor::StartOfLine, QTextCursor::KeepAnchor);
                QString selected = cursor.selectedText();

                if (selected != "")
                {
                    QString rstrip = Utils::rstrip(selected);

                    if (rstrip != selected)
                    {
                        cursor.setPosition(pos - selected.size() + rstrip.size(), QTextCursor::MoveAnchor);
                        cursor.movePosition(QTextCursor::EndOfLine, QTextCursor::KeepAnchor);
                        cursor.removeSelectedText();
                    }

                }
            }

            cursor.endEditBlock();

            e->accept();
        }
    }
}

} //end namespace ito
