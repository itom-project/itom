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

#ifndef CODEEDITORTESTHELPER_H
#define CODEEDITORTESTHELPER_H

#include "codeEditor/codeEditor.h"
#include "codeEditor/syntaxHighlighter/codeEditorStyle.h"
#include "codeEditor/syntaxHighlighter/genericSyntaxHighlighter.h"
#include "helper/editorFileType.h"

#include <qsharedpointer.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qvector.h>

namespace ito {
namespace test {

//! path of the settings file, that is returned by the AppManagement stub
QString settingsFilePath();
void setSettingsFilePath(const QString& path);

//! a style, where every style type has a unique foreground color. This allows to detect
//! the style type from the format of a character in the document (see styleAt).
QSharedPointer<CodeEditorStyle> createUniqueColorStyle();

//! the style type that belongs to a unique color (KeyDefault if not found)
StyleItem::StyleType styleFromColor(const QColor& color);

//! letter used in the style string for a style type (e.g. 'C' for KeyComment, ' ' for default)
QChar styleLetter(StyleItem::StyleType type);

//! Owns a code editor with a generic syntax highlighter for the given file type. All text
//! is highlighted synchronously, such that the formats can be checked immediately.
class HighlightedEditor
{
public:
    explicit HighlightedEditor(EditorFileType::Type type);
    ~HighlightedEditor();

    //! sets the text and runs the highlighter over the whole document
    void setText(const QString& text);

    //! style of the character at the given line (0-based) and column
    StyleItem::StyleType styleAt(int line, int column) const;

    //! one letter per character of the line, see styleLetter. Unformatted = ' '.
    QString styleString(int line) const;

    //! user state of the text block (multi-line state of the highlighter)
    int blockState(int line) const;

    int lineCount() const;

    CodeEditor* editor() const
    {
        return m_editor;
    }

    GenericSyntaxHighlighter* highlighter() const
    {
        return m_highlighter.data();
    }

private:
    CodeEditor* m_editor;
    QSharedPointer<GenericSyntaxHighlighter> m_highlighter;
    QSharedPointer<CodeEditorStyle> m_style;
};

} // end namespace test
} // end namespace ito

#endif
