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

#ifndef PYSYNTAXHIGHLIGHER_H
#define PYSYNTAXHIGHLIGHER_H

#include "syntaxHighlighterBase.h"

#include <qregularexpression.h>
#include <qtextformat.h>

namespace ito {

/*
This module contains a native python syntax highlighter, strongly inspired from
spyderlib.widgets.source_code.syntax_higlighter.PythonSH but modified to
highlight docstrings with a different color than the string color and to
highlight decorators and self parameters.

It is approximately 3 time faster then :class:`pyqode.core.modes.PygmentsSH`.
*/
class PythonSyntaxHighlighter : public SyntaxHighlighterBase
{
    Q_OBJECT
public:
    enum State //!< Syntax highlighting states (from one text block to another):
    {
        Normal = 0,
        InsideSq3String = 1, // the line is within a '''....''' multiline comment
        InsideDq3String = 2, // the line is within a """....""" multiline comment
        InsideSqString = 3, // the line is within a '...' string
        InsideDqString = 4 // the line is within a "..." string
    };

    PythonSyntaxHighlighter(QTextDocument *parent, const QString &description = "", QSharedPointer<CodeEditorStyle> editorStyle = QSharedPointer<CodeEditorStyle>());

    virtual ~PythonSyntaxHighlighter();

    /*
    Abstract method. Override this to apply syntax highlighting.

    :param text: Line of text to highlight.
    :param block: current block
    */
    void highlight_block(const QString &text, QTextBlock &block);

    void default_highlight_block(const QString &text, bool outputNotError);

    virtual void rehighlight();

private:

    struct NamedRegExp
    {
        NamedRegExp(const QString &groupName_, const QRegularExpression &regExp_) : regExp(regExp_), groupNames(groupName_) {}
        NamedRegExp(const QStringList &groupNames_, const QRegularExpression &regExp_) : regExp(regExp_), groupNames(groupNames_) {}
        QRegularExpression regExp;
        QStringList groupNames;
    };

    //syntax highlighting rules
    static QList<NamedRegExp> regExpProg;
    static QRegularExpression regExpIdProg;
    static QRegularExpression regExpAsProg;
    static QRegularExpression regExpOeComment; //comments suitable for outline explorer

    QTextCharFormat getFormatFromStyle(StyleItem::StyleType token) const;
    const QTextCharFormat getTextCharFormat(const QString &colorName, const QString &style = QString());

    static QList<NamedRegExp> makePythonPatterns(const QStringList &additionalKeywords = QStringList(), const QStringList &additionalBuiltins = QStringList());
};

} //end namespace ito

#endif
