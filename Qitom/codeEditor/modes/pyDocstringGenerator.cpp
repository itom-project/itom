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
#include "../managers/panelsManager.h"
#include "../../widgets/scriptEditorWidget.h"
#include "../../widgets/menuOnlyForEnter.h"

#include <qdebug.h>
#include <iostream>
#include <qregularexpression.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyDocstringGeneratorMode::PyDocstringGeneratorMode(const QString &name, const QString &description /*= ""*/, QObject *parent /*= NULL*/) :
    Mode(name, description),
    QObject(parent),
    m_docstringStyle(Style::GoogleStyle),
    m_overwriteEndLineIndex(-1)
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
/*
*/
void PyDocstringGeneratorMode::onKeyPressed(QKeyEvent *e)
{
    if (!e->isAccepted())
    {
        if (e->key() == Qt::Key_QuoteDbl ||
            e->key() == Qt::Key_Apostrophe) // " or '
        {
            CodeEditor *ed = editor();
            QTextCursor cursor = ed->textCursor();
            int lineIdx = cursor.blockNumber();
            cursor.movePosition(QTextCursor::StartOfBlock, QTextCursor::KeepAnchor);

            if (cursor.hasSelection())
            {
                QString sel = cursor.selectedText().trimmed();

                if ((e->key() == Qt::Key_QuoteDbl && sel == "\"\"") ||
                    (e->key() == Qt::Key_Apostrophe && sel == "''"))
                {

                    QSharedPointer<OutlineItem> item = getOutlineOfLineIdx(lineIdx);

                    if (!item.isNull())
                    {
                        qDebug() << item->m_name << item->m_startLineIdx << item->m_endLineIdx;
                        m_overwriteEndLineIndex = item->m_endLineIdx;

                        if (lastLineIdxOfDefinition(item) == lineIdx - 1)
                        {
                            // the first """ sign in the line right after the
                            // definition.
                            QPoint pt = ed->cursorRect().bottomRight();
                            pt.rx() += ed->panels()->marginSize(ito::Panel::Left);
                            pt.ry() += ed->panels()->marginSize(ito::Panel::Top);
                            pt = ed->mapToGlobal(pt);

                            m_popupMenu = QSharedPointer<QMenu>(new MenuOnlyForEnter(ed));
                            QAction *a = m_popupMenu->addAction(
                                QIcon(":/arrows/icons/plus.png"),
                                tr("Generate docstring"),
                                this,
                                &PyDocstringGeneratorMode::mnuInsertDocstring
                            );
                            m_popupMenu->setActiveAction(a);
                            m_popupMenu->popup(pt);
                        }
                    }
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------
void PyDocstringGeneratorMode::mnuInsertDocstring()
{
    CodeEditor *e = editor();
    QTextCursor cursor = e->textCursor();

    cursor.movePosition(QTextCursor::StartOfBlock, QTextCursor::MoveAnchor);
    cursor.movePosition(QTextCursor::NextBlock, QTextCursor::KeepAnchor);

    if (!cursor.hasSelection())
    {
        cursor.movePosition(QTextCursor::EndOfBlock, QTextCursor::KeepAnchor);
    }

    if (cursor.hasSelection())
    {
        if (cursor.selectedText().trimmed() == "\"\"\"")
        {
            cursor.movePosition(QTextCursor::PreviousBlock);
            insertDocstring(cursor, "\"\"\"", false, m_overwriteEndLineIndex);
        }
        else if (cursor.selectedText().trimmed() == "'''")
        {
            cursor.movePosition(QTextCursor::PreviousBlock);
            insertDocstring(cursor, "'''", false, m_overwriteEndLineIndex);
        }

        m_overwriteEndLineIndex = -1;
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

    auto current = sew->parseOutline(false);
    auto result = QSharedPointer<OutlineItem>();
    bool found = true;

    while (found && !current.isNull() && current->m_childs.size() > 0)
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
        else if (result->m_type == OutlineItem::typeClass)
        {
            result.clear();
        }
    }

    return result;
}


//-------------------------------------------------------------------------------------
/*
overwriteEndLineIdx : If this method is called by the popup menu, the method
    is currently much longer than in reality, since the three opening quotes
    create an undesired multiline comment. Therefore, the outline after having
    added the three quotes is much longer than before having inserted it. Therefore
    the last line of the outline can be overwritten by this value (if != -1).
*/
void PyDocstringGeneratorMode::insertDocstring(
    const QTextCursor &cursor,
    const QString &quotes /*= "\"\"\""*/,
    bool insertOpeningQuotes /*= true*/,
    int overwriteEndLineIdx /*= -1*/) const
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

    if (overwriteEndLineIdx >= outline->m_startLineIdx)
    {
        // deep copy of outline and replace m_endLineIdx
        outline = QSharedPointer<ito::OutlineItem>(new ito::OutlineItem(*outline));
        outline->m_endLineIdx = overwriteEndLineIdx;

        auto it = outline->m_childs.begin();

        while (it != outline->m_childs.end())
        {
            if ((*it)->m_startLineIdx >= overwriteEndLineIdx)
            {
                // remove this child since it is outside of overwriteEndLineIdx
                it = outline->m_childs.erase(it);
                continue;
            }
            else if ((*it)->m_endLineIdx > overwriteEndLineIdx)
            {
                (*it)->m_endLineIdx = overwriteEndLineIdx;
            }

            it++;
        }
    }

    CodeEditor *e = editor();

    int lineIdx = lastLineIdxOfDefinition(outline);

    if (lineIdx < 0)
    {
        return;
    }

    QTextCursor insertCursor = e->gotoLine(lineIdx, 0);
    insertCursor.movePosition(QTextCursor::EndOfLine);

    if (!insertOpeningQuotes)
    {
        // move one line down and to the end, we expect
        // the opening quotes to be at the end.
        insertCursor.movePosition(QTextCursor::NextBlock);
        insertCursor.movePosition(QTextCursor::EndOfLine);
    }

    // get the indentation for the new docstring
    int initIndent = e->lineIndent(outline->m_startLineIdx);
    QString indent = editor()->useSpacesInsteadOfTabs() ?
        QString(initIndent + editor()->tabLength(), ' ') :
        QString(initIndent + 1, '\t');

    FunctionInfo finfo = parseFunctionInfo(outline, lineIdx);
    int cursorPos = 0;

    QString docstring;

    if (m_docstringStyle == GoogleStyle)
    {
        docstring = generateGoogleDoc(outline, finfo, cursorPos);
    }
    else
    {
        docstring = generateNumpyDoc(outline, finfo, cursorPos);
    }

    if (insertOpeningQuotes)
    {
        docstring = QString("%1%2\n%1").arg(quotes).arg(docstring);
    }
    else
    {
        docstring = QString("%2\n%1").arg(quotes).arg(docstring);
    }

    // add the indentation to all lines of the docstring
    QStringList lines = docstring.split("\n");

    for (int i = 0; i < lines.size(); ++i)
    {
        if (!insertOpeningQuotes && i == 0)
        {
            // do not indent the first line, since it is already
            // right after the existing opening quotes
            continue;
        }

        lines[i] = Utils::rstrip(indent + lines[i]);
    }

    // insert the docstring
    insertCursor.beginEditBlock();

    if (insertOpeningQuotes)
    {
        insertCursor.insertText("\n" + lines.join("\n"));
    }
    else
    {
        insertCursor.insertText(lines.join("\n"));
    }

    insertCursor.endEditBlock();

    e->setCursorPosition(lineIdx + 1, indent.size() + quotes.size() + cursorPos);

    e->textChanged();
}

//-------------------------------------------------------------------------------------
PyDocstringGeneratorMode::FunctionInfo PyDocstringGeneratorMode::parseFunctionInfo(
    const QSharedPointer<OutlineItem> &item,
    int lastLineIdxOfDefinition) const
{
    FunctionInfo info;

    // parse arguments
    parseArgList(item, info);

    // get code
    QStringList codelines;
    const CodeEditor *e = editor();
    int startIdx = std::max(item->m_startLineIdx, lastLineIdxOfDefinition + 1);

    for (int i = startIdx; i <= item->m_endLineIdx; ++i)
    {
        codelines.append(e->lineText(i));
    }

    foreach(const QSharedPointer<OutlineItem> &c, item->m_childs)
    {
        // remove code from subfunctions
        for (int i = c->m_startLineIdx; i <= c->m_endLineIdx; ++i)
        {
            if (i < startIdx || i >= item->m_endLineIdx)
            {
                // error. This should never happen.
#if _DEBUG
                std::cout << "Warning. Improper outline structure detected when parsing the docstring.\n" << std::endl;
#endif
                break;
            }

            codelines[i - startIdx] = "";
        }
    }

    codelines.removeAll("");

    QString code = codelines.join("\n");

    // raise
    QRegularExpression re("[ \\t]raise ([a-zA-Z0-9_]*)");
    QRegularExpressionMatchIterator reIter = re.globalMatch(code);

    while (reIter.hasNext())
    {
        QRegularExpressionMatch match = reIter.next();
        info.m_raises.append(match.captured(1));
    }

    info.m_hasYield = false;

    // generic return type
    if (item->m_returnType != "")
    {
        info.m_returnTypes << item->m_returnType;
    }

    // yield
    QRegularExpressionMatchIterator yieldIter =
        QRegularExpression("[ \\t]yield ").globalMatch(code);

    if (yieldIter.hasNext())
    {
        info.m_hasYield = true;

        if (info.m_returnTypes.size() == 0)
        {
            info.m_returnTypes << "TYPE";
        }
    }
    else
    {
        // return
        QRegularExpressionMatchIterator returnIter =
            QRegularExpression("[ \\t]return ").globalMatch(code);

        if (returnIter.hasNext())
        {
            if (info.m_returnTypes.size() == 0)
            {
                info.m_returnTypes << "TYPE";
            }
        }
    }

    return info;
}

//-------------------------------------------------------------------------------------
void PyDocstringGeneratorMode::parseArgList(
    const QSharedPointer<OutlineItem> &item, FunctionInfo &info) const
{
    bool expectSelfOrCls =
        item->m_type == OutlineItem::typeClassMethod ||
        item->m_type == OutlineItem::typeMethod ||
        item->m_type == OutlineItem::typePropertyGet ||
        item->m_type == OutlineItem::typePropertySet;

    QString argstr = item->m_args.trimmed();
    QStringList args;
    int lastpos = 0;
    QList<QChar> specialCharStack;
    int idx1, idx2;

    for (int pos = 0; pos < argstr.size(); ++pos)
    {
        if (argstr[pos] == ',' && specialCharStack.size() == 0)
        {
            if (pos - lastpos > 0)
            {
                args.append(argstr.mid(lastpos, pos - lastpos));
            }

            lastpos = pos + 1; //ignore the comma
            continue;
        }

        QChar lastChar = specialCharStack.size() == 0 ? QChar() : specialCharStack.last();

        if (lastChar != '"' && lastChar != '\'')
        {
            switch (argstr[pos].toLatin1())
            {
            case '(':
                specialCharStack.append('(');
                break;
            case ')':
                if (lastChar == '(')
                {
                    specialCharStack.removeLast();
                }
                break;
            case '[':
                specialCharStack.append('[');
                break;
            case ']':
                if (lastChar == '[')
                {
                    specialCharStack.removeLast();
                }
                break;
            case '{':
                specialCharStack.append('{');
                break;
            case '}':
                if (lastChar == '{')
                {
                    specialCharStack.removeLast();
                }
                break;
            case '"':
                if (lastChar == '"')
                {
                    specialCharStack.removeLast();
                }
                else
                {
                    specialCharStack.append('"');
                }
                break;
            case '\'':
                if (lastChar == '\'')
                {
                    specialCharStack.removeLast();
                }
                else
                {
                    specialCharStack.append('\'');
                }
                break;
            }
        }
        else // last char = " or '
        {
            if (argstr[pos] == lastChar)
            {
                specialCharStack.removeLast();
            }
        }
    }

    if (lastpos < argstr.size())
    {
        // append last section
        args.append(argstr.mid(lastpos));
    }

    int count = 0;

    foreach(const QString &arg, args)
    {
        idx1 = arg.indexOf(":");
        if (idx1 == -1 && arg.indexOf("=") >= 0)
        {
            idx2 = arg.indexOf("=");
        }
        else
        {
            idx2 = arg.indexOf("=", idx1);
        }

        ArgInfo a;
        a.m_isOptional = (idx2 >= 0);

        if (idx1 >= 0)
        {
            a.m_name = arg.left(idx1).trimmed();
            a.m_type = idx2 >= 0 ? arg.mid(idx1 + 1, idx2 - idx1 - 1).trimmed()
                                 : arg.mid(idx1 + 1).trimmed();
        }
        else if (idx2 >= 0)
        {
            a.m_name = arg.left(idx2).trimmed();
        }
        else
        {
            a.m_name = arg.trimmed();
        }

        if (a.m_isOptional)
        {
            a.m_defaultValue = arg.mid(idx2 + 1).trimmed();
        }

        if (count == 0 && expectSelfOrCls && (a.m_name == "self" || a.m_name == "cls"))
        {
            //pass
        }
        else
        {
            info.m_args.append(a);
        }



        count++;
    }
}

//-------------------------------------------------------------------------------------
QString PyDocstringGeneratorMode::generateGoogleDoc(
    const QSharedPointer<OutlineItem> &item, const FunctionInfo &info, int &cursorPos) const
{
    QString docs = "";
    cursorPos = 0;

    if (item->m_type == OutlineItem::typePropertyGet ||
        item->m_type == OutlineItem::typePropertySet)
    {
        if (info.m_returnTypes.size() == 1)
        {
            docs += info.m_returnTypes[0] + ": ";
            cursorPos = docs.size();
            docs += "DESCRIPTION";
        }

        if (info.m_raises.size() > 0)
        {
            docs += "\n\nRaises:";

            foreach(const QString &exc, info.m_raises)
            {
                docs += QString("\n    %1: DESCRIPTION").arg(exc);
            }
        }
    }
    else
    {
        if (info.m_args.size() > 0)
        {
            docs += "\n\nArgs:";

            foreach(const ArgInfo &arg, info.m_args)
            {
                if (arg.m_name != "")
                {
                    QString typ = arg.m_type != "" ? arg.m_type : "TYPE";

                    if (arg.m_isOptional)
                    {
                        // Defaults notation according https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
                        docs += QString("\n    %1 (%2, optional): DESCRIPTION. Defaults to %3.")
                            .arg(arg.m_name).arg(typ).arg(arg.m_defaultValue);
                    }
                    else
                    {
                        docs += QString("\n    %1 (%2): DESCRIPTION")
                            .arg(arg.m_name).arg(typ);
                    }
                }
            }
        }

        if (info.m_raises.size() > 0)
        {
            docs += "\n\nRaises:";

            foreach(const QString &exc, info.m_raises)
            {
                docs += QString("\n    %1: DESCRIPTION").arg(exc);
            }
        }

        if (info.m_returnTypes.size() > 0)
        {
            if (info.m_hasYield)
            {
                docs += "\n\nYields:";
            }
            else
            {
                docs += "\n\nReturns:";
            }

            foreach(const QString &returnType, info.m_returnTypes)
            {
                docs += QString("\n    %1: DESCRIPTION").arg(returnType);
            }
        }
    }

    return docs;
}

//-------------------------------------------------------------------------------------
QString PyDocstringGeneratorMode::generateNumpyDoc(
    const QSharedPointer<OutlineItem> &item, const FunctionInfo &info, int &cursorPos) const
{
    QString docs = "";
    cursorPos = 0;

    if (item->m_type == OutlineItem::typePropertyGet ||
        item->m_type == OutlineItem::typePropertySet)
    {
        if (info.m_returnTypes.size() == 1)
        {
            docs += info.m_returnTypes[0] + ": ";
            cursorPos = docs.size();
            docs += "DESCRIPTION";
        }

        if (info.m_raises.size() > 0)
        {
            docs += "\n\nRaises\n------";

            foreach(const QString &exc, info.m_raises)
            {
                docs += QString("\n%1\n    DESCRIPTION").arg(exc);
            }
        }
    }
    else
    {
        if (info.m_args.size() > 0)
        {
            docs += "\n\nParameters\n----------";

            foreach(const ArgInfo &arg, info.m_args)
            {
                if (arg.m_name != "")
                {
                    QString typ = arg.m_type != "" ? arg.m_type : "TYPE";

                    if (arg.m_isOptional)
                    {
                        docs += QString("\n%1 : %2, optional\n    DESCRIPTION, by default %3")
                            .arg(arg.m_name).arg(typ).arg(arg.m_defaultValue);
                    }
                    else
                    {
                        docs += QString("\n%1 : %2\n    DESCRIPTION")
                            .arg(arg.m_name).arg(typ);
                    }
                }
            }
        }

        if (info.m_raises.size() > 0)
        {
            docs += "\n\nRaises\n------";

            foreach(const QString &exc, info.m_raises)
            {
                docs += QString("\n%1\n    DESCRIPTION").arg(exc);
            }
        }

        if (info.m_returnTypes.size() > 0)
        {
            if (info.m_hasYield)
            {
                docs += "\n\nYields\n------";
            }
            else
            {
                docs += "\n\nReturns\n-------";
            }

            foreach(const QString &returnType, info.m_returnTypes)
            {
                docs += QString("\n%1\n    DESCRIPTION").arg(returnType);
            }
        }
    }

    return docs;
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
