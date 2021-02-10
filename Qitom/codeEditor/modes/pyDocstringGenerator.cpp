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

#include "pyDocstringGenerator.h"

#include "../codeEditor.h"
#include "../../widgets/scriptEditorWidget.h"

#include <qdebug.h>
#include <qregularexpression.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyDocstringGeneratorMode::PyDocstringGeneratorMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent)
{
}

//-------------------------------------------------------------------------------------
/*
*/
PyDocstringGeneratorMode::~PyDocstringGeneratorMode()
{
}


//-------------------------------------------------------------------------------------
/*
*/
void PyDocstringGeneratorMode::onStateChanged(bool state)
{
    if (state)
    {
        // maybe the connection already exists.
        connect(
            editor(), &CodeEditor::keyPressed, 
            this, &PyDocstringGeneratorMode::onKeyPressed,
            Qt::UniqueConnection);
    }
    else
    {
        disconnect(
            editor(), &CodeEditor::keyPressed, 
            this, &PyDocstringGeneratorMode::onKeyPressed);
    }       
}

//-------------------------------------------------------------------------------------
QSharedPointer<OutlineItem> PyDocstringGeneratorMode::getOutlineOfLineIdx(int lineIdx) const
{
    ScriptEditorWidget *sew = qobject_cast<ScriptEditorWidget*>(editor());

    if (!sew)
    {
        return QSharedPointer<OutlineItem>();
    }

    auto current = sew->parseOutline();
    auto result = QSharedPointer<OutlineItem>();
    bool found = true;

    while (found && !current.isNull())
    {
        found = false;

        foreach(const QSharedPointer<OutlineItem> &c, current->m_childs)
        {
            if (lineIdx >= c->m_startLineIdx
                && lineIdx <= c->m_endLineIdx)
            {
                result = c;
                current = c;
                found = true;
                break;
            }
        }
    }

    if (!result.isNull())
    {
        if (lastLineIdxOfDefinition(result) < 0)
        {
            result.clear();
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------
void PyDocstringGeneratorMode::insertDocstring(const QTextCursor &cursor) const
{
    if (cursor.isNull())
    {
        return;
    }

    auto outline = getOutlineOfLineIdx(cursor.blockNumber());

    if (outline.isNull())
    {
        return;
    }

    CodeEditor *e = editor();

    int lineIdx = lastLineIdxOfDefinition(outline);

    if (lineIdx < 0)
    {
        return;
    }

    QTextCursor insertCursor = e->gotoLine(lineIdx, 0);
    insertCursor.movePosition(QTextCursor::EndOfLine);

    // get the indentation for the new docstring
    int initIndent = e->lineIndent(outline->m_startLineIdx);
    QString indent = editor()->useSpacesInsteadOfTabs() ? 
        QString(initIndent + editor()->tabLength(), ' ') : 
        QString(initIndent + 1, '\t');
    
    QString docstring = "\"\"\"DOCSTRING\n\nARGS: \n--------\nparam1 : int\n    ARG\n\"\"\"";

    // add the indentation to all lines of the docstring
    QStringList lines = docstring.split("\n");

    for (int i = 0; i < lines.size(); ++i)
    {
        lines[i].prepend(indent);
    }

    // insert the docstring
    insertCursor.insertText("\n" + lines.join("\n"));

    insertCursor.movePosition(QTextCursor::NextBlock);
    e->setTextCursor(insertCursor);
}

//-------------------------------------------------------------------------------------
int PyDocstringGeneratorMode::lastLineIdxOfDefinition(const QSharedPointer<OutlineItem> &item) const
{
    CodeEditor *e = editor();
    QString text;

    if (!e)
    {
        return -1;
    }

    QRegularExpression re("^(.*)(:\\s*#)(.*)$");

    for (int idx = item->m_startLineIdx; idx <= item->m_endLineIdx; ++idx)
    {
        text = e->lineText(idx).trimmed();

        if (text.endsWith(":"))
        {
            return idx;
        }

        QRegularExpressionMatch match = re.match(text);

        if (match.hasMatch())
        {
            return idx;
        }
    }

    return -1;
}

} //end namespace ito
