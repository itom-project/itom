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

#ifndef SYNTAXHIGHLIGHTERBASE_H
#define SYNTAXHIGHLIGHTERBASE_H

#include <qstring.h>
#include <qtextedit.h>
#include <qsyntaxhighlighter.h>
#include <qpointer.h>
#include <qmap.h>
#include <qregularexpression.h>

#include "../textBlockUserData.h"
#include "../mode.h"
#include "../foldDetector/foldDetector.h"
#include "codeEditorStyle.h"

namespace ito {

class CodeEditor; //forware declaration

/*
Syntax Highlighters should derive from this mode instead of mode directly.
*/
class SyntaxHighlighterBase : public QSyntaxHighlighter, public Mode
{
    Q_OBJECT
public:
    SyntaxHighlighterBase(const QString &name, QTextDocument *parent, const QString &description = "", QSharedPointer<CodeEditorStyle> editorStyle = QSharedPointer<CodeEditorStyle>());

    virtual ~SyntaxHighlighterBase();

    void setFoldDetector(QSharedPointer<FoldDetector> foldDetector);

    virtual void onStateChanged(bool state);
    virtual void onInstall(CodeEditor *editor);

    QSharedPointer<CodeEditorStyle> editorStyle() const { return m_editorStyle; }

    /*
    Highlights a block of text. Please do not override, this method.
    Instead you should implement
    :func:`pyqode.core.api.SyntaxHighlighter.highlight_block`.

    :param text: text to highlight.
    */
    void highlightBlock(const QString &text);

    void refreshEditor(QSharedPointer<CodeEditorStyle> editorStyle);

    /*
    Abstract method. Override this to apply syntax highlighting.

    :param text: Line of text to highlight.
    :param block: current block
    */
    virtual void highlight_block(const QString &text, QTextBlock &block) = 0;

    virtual void default_highlight_block(const QString &text, bool outputNotError) = 0;

    virtual void rehighlight();

protected:
    static QTextBlock findPrevNonBlankBlock(const QTextBlock &currentBlock);

    void highlightWhitespaces(const QString &text);

    QRegularExpression m_regWhitespaces;
    QRegularExpression m_regSpacesPtrn;
    QSharedPointer<CodeEditorStyle> m_editorStyle;
    QSharedPointer<FoldDetector> m_foldDetector;
};

} //end namespace ito

#endif
