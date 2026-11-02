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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

*********************************************************************** */

#ifndef GENERICSYNTAXHIGHLIGHTER_H
#define GENERICSYNTAXHIGHLIGHTER_H

#include "syntaxHighlighterBase.h"

#include "../../helper/editorFileType.h"

#include <qlist.h>
#include <qregularexpression.h>

namespace ito {

//! Rule based syntax highlighter for all text-like files besides python (json, yaml, xml, ...).
/*!
    The highlighter is configured by a list of single-line rules (regular expression plus style)
    and optionally by multi-line constructs (e.g. block comments), whose state is stored in the
    user state of the text blocks. The style items are the same as used by the python highlighter,
    such that the color settings of the editor apply to all file types.
*/
class GenericSyntaxHighlighter : public SyntaxHighlighterBase
{
    Q_OBJECT
public:
    //! first block state value used for multi-line constructs (0 = normal). The values 1 and 2
    //! are used by the python highlighter and interpreted by the indent fold detector.
    static const int FirstMultiLineState = 10;

    GenericSyntaxHighlighter(
        EditorFileType::Type type,
        QTextDocument* parent,
        QSharedPointer<CodeEditorStyle> editorStyle = QSharedPointer<CodeEditorStyle>());

    virtual ~GenericSyntaxHighlighter();

    EditorFileType::Type fileType() const
    {
        return m_type;
    }

    //! creates a fold detector that fits to the given file type (null if folding is not useful).
    static QSharedPointer<FoldDetector> createFoldDetector(EditorFileType::Type type);

    void highlight_block(const QString& text, QTextBlock& block);

    void default_highlight_block(const QString& text, const TextBlockUserData* textBlockUserData);

    virtual void rehighlight();

private:
    struct Rule
    {
        Rule(const QString& pattern, StyleItem::StyleType style_, int group_ = 0) :
            regExp(pattern), style(style_), group(group_)
        {
        }
        QRegularExpression regExp;
        StyleItem::StyleType style;
        int group; //!< capture group that is colored (0: whole match)
    };

    struct MultiLine
    {
        MultiLine(const QString& begin, const QString& end, StyleItem::StyleType style_) :
            beginExp(begin), endExp(end), style(style_)
        {
        }
        QRegularExpression beginExp;
        QRegularExpression endExp;
        StyleItem::StyleType style;
    };

    void setupRules();

    inline void addRule(const QString& pattern, StyleItem::StyleType style, int group = 0)
    {
        m_rules.append(Rule(pattern, style, group));
    }

    void addKeywords(
        const QStringList& words,
        StyleItem::StyleType style,
        bool caseInsensitive = false);

    QTextCharFormat getFormatFromStyle(StyleItem::StyleType token) const;

    EditorFileType::Type m_type;
    QList<Rule> m_rules;
    QList<MultiLine> m_multiLines;
};

} // end namespace ito

#endif
