/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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
*********************************************************************** */

#include "codeEditorTestHelper.h"

#include "codeEditor/managers/modesManager.h"
#include "codeEditor/utils/utils.h"

#include <qcoreapplication.h>
#include <qtextlayout.h>

namespace ito {
namespace test {

static QString g_settingsFile;

//-------------------------------------------------------------------------------------
QString settingsFilePath()
{
    return g_settingsFile;
}

//-------------------------------------------------------------------------------------
void setSettingsFilePath(const QString& path)
{
    g_settingsFile = path;
}

//-------------------------------------------------------------------------------------
// the unique color of a style type: red = 1 + enum value (KeyDefault -> 1, ...),
// green = 100 (to not collide with any other default color in the editor)
static QColor uniqueColor(StyleItem::StyleType type)
{
    int v = (int)type;

    if (v >= 1000)
    {
        v = v - 1000 + 100; // KeyBackground and following
    }

    return QColor(v + 1, 100, 0);
}

//-------------------------------------------------------------------------------------
QSharedPointer<CodeEditorStyle> createUniqueColorStyle()
{
    QSharedPointer<CodeEditorStyle> style(new CodeEditorStyle());

    foreach (StyleItem::StyleType type, StyleItem::availableStyleTypes())
    {
        style->rformat(type).setForeground(QBrush(uniqueColor(type)));
    }

    return style;
}

//-------------------------------------------------------------------------------------
StyleItem::StyleType styleFromColor(const QColor& color)
{
    if (color.green() != 100 || color.blue() != 0)
    {
        return StyleItem::KeyDefault;
    }

    foreach (StyleItem::StyleType type, StyleItem::availableStyleTypes())
    {
        if (uniqueColor(type) == color)
        {
            return type;
        }
    }

    return StyleItem::KeyDefault;
}

//-------------------------------------------------------------------------------------
QChar styleLetter(StyleItem::StyleType type)
{
    switch (type)
    {
    case StyleItem::KeyDefault:
        return ' ';
    case StyleItem::KeyComment:
        return 'C';
    case StyleItem::KeyNumber:
        return 'N';
    case StyleItem::KeyString:
        return 'S';
    case StyleItem::KeyKeyword:
        return 'K';
    case StyleItem::KeyClass:
        return 'L'; // class / headline / label / section
    case StyleItem::KeyFunction:
        return 'F';
    case StyleItem::KeyOperator:
        return 'O';
    case StyleItem::KeyDecorator:
        return 'D';
    case StyleItem::KeyBuiltin:
        return 'B';
    case StyleItem::KeyTag:
        return 'T';
    case StyleItem::KeyPunctuation:
        return 'P';
    case StyleItem::KeyConstant:
        return 'X';
    case StyleItem::KeyOperatorWord:
        return 'W';
    default:
        return '?';
    }
}

//-------------------------------------------------------------------------------------
HighlightedEditor::HighlightedEditor(EditorFileType::Type type) :
    m_editor(new CodeEditor(nullptr, false)), m_style(createUniqueColorStyle())
{
    m_highlighter = QSharedPointer<GenericSyntaxHighlighter>(
        new GenericSyntaxHighlighter(type, m_editor->document(), m_style));
    m_highlighter->setFoldDetector(GenericSyntaxHighlighter::createFoldDetector(type));
    m_editor->modes()->append(m_highlighter.dynamicCast<Mode>());
}

//-------------------------------------------------------------------------------------
HighlightedEditor::~HighlightedEditor()
{
    m_editor->modes()->remove(m_highlighter.dynamicCast<Mode>());
    m_highlighter.clear();
    delete m_editor;
}

//-------------------------------------------------------------------------------------
void HighlightedEditor::setText(const QString& text)
{
    m_editor->setPlainText(text);
    // QSyntaxHighlighter may defer the first highlighting to the event loop
    QCoreApplication::processEvents();
    m_highlighter->QSyntaxHighlighter::rehighlight();
}

//-------------------------------------------------------------------------------------
int HighlightedEditor::lineCount() const
{
    return m_editor->document()->blockCount();
}

//-------------------------------------------------------------------------------------
StyleItem::StyleType HighlightedEditor::styleAt(int line, int column) const
{
    QTextBlock block = m_editor->document()->findBlockByNumber(line);

    if (!block.isValid())
    {
        return StyleItem::KeyDefault;
    }

    foreach (const QTextLayout::FormatRange& range, block.layout()->formats())
    {
        if (column >= range.start && column < range.start + range.length)
        {
            return styleFromColor(range.format.foreground().color());
        }
    }

    return StyleItem::KeyDefault;
}

//-------------------------------------------------------------------------------------
QString HighlightedEditor::styleString(int line) const
{
    QTextBlock block = m_editor->document()->findBlockByNumber(line);

    if (!block.isValid())
    {
        return QString();
    }

    QString result(block.text().size(), ' ');

    foreach (const QTextLayout::FormatRange& range, block.layout()->formats())
    {
        QChar letter = styleLetter(styleFromColor(range.format.foreground().color()));

        for (int i = range.start; i < range.start + range.length && i < result.size(); ++i)
        {
            result[i] = letter;
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------
int HighlightedEditor::blockState(int line) const
{
    QTextBlock block = m_editor->document()->findBlockByNumber(line);
    return block.isValid() ? Utils::TextBlockHelper::getState(block) : 0;
}

} // end namespace test
} // end namespace ito
