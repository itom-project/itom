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

#include "syntaxHighlighterBase.h"

#include "../codeEditor.h"
#include <qapplication.h>
#include <qtextdocument.h>
#include <qdebug.h>

#include "../managers/modesManager.h"
#include "../managers/panelsManager.h"
#include "../modes/caretLineHighlight.h"
#include "../panels/foldingPanel.h"
#include "../utils/utils.h"

namespace ito {

//-------------------------------------------------------------------
SyntaxHighlighterBase::SyntaxHighlighterBase(const QString &name, QTextDocument *parent, const QString &description /*= ""*/, QSharedPointer<CodeEditorStyle> editorStyle /*= QSharedPointer<CodeEditorStyle>()*/) :
    QSyntaxHighlighter(parent),
    Mode(name, description)
{
    m_editorStyle = editorStyle;
    m_regSpacesPtrn = QRegularExpression("[ \\t]+");
    m_regWhitespaces = QRegularExpression("\\s+");
    m_foldDetector.clear();
}

//-------------------------------------------------------------------
SyntaxHighlighterBase::~SyntaxHighlighterBase()
{
}

//-------------------------------------------------------------------
void SyntaxHighlighterBase::setFoldDetector(QSharedPointer<FoldDetector> foldDetector)
{
    m_foldDetector = foldDetector;
}

//-------------------------------------------------------------------
/*static*/ QTextBlock SyntaxHighlighterBase::findPrevNonBlankBlock(const QTextBlock &currentBlock)
{
    QTextBlock previousBlock = currentBlock.blockNumber() ? currentBlock.previous() : QTextBlock();
    QString t = previousBlock.text().trimmed();
    // find the previous non-blank block
    while (previousBlock.isValid() &&
        previousBlock.text().trimmed() == "")
    {
        previousBlock = previousBlock.previous();
        t = previousBlock.text().trimmed();
    }

    if (previousBlock.isValid() &&
        previousBlock.text().trimmed() == "")
    {
        previousBlock = QTextBlock();
    }

    return previousBlock;
}

//-------------------------------------------------------------------
void SyntaxHighlighterBase::highlightWhitespaces(const QString &text)
{
    QRegularExpressionMatch match = m_regWhitespaces.match(text);

    int index = match.capturedStart(0);
    int length;

    while (index >= 0)
    {
        index = match.capturedStart(0);
        length = match.captured(0).size();
        setFormat(index, length, m_editorStyle->format(StyleItem::KeyWhitespace));
        match = m_regWhitespaces.match(text, index + length);
        index = match.capturedStart(0);
    }
}

//-------------------------------------------------------------------
void SyntaxHighlighterBase::highlightBlock(const QString &text)
{
    if (!enabled())
    {
        return;
    }

    QTextBlock current_block = currentBlock();

    if (editor())
    {
        CodeEditor *e = editor();

        //qDebug() << current_block.blockNumber();
        TextBlockUserData *userData = e->getTextBlockUserData(current_block, false);
        if (userData && userData->m_syntaxStyle != TextBlockUserData::StylePython)
        {
            default_highlight_block(text, userData->m_syntaxStyle == TextBlockUserData::StyleOutput);
        }
        else
        {
            highlight_block(text, current_block);

            if (e->showWhitespaces())
            {
                highlightWhitespaces(text);
            }
        }

        if (m_foldDetector.isNull() == false)
        {
            QTextBlock previous_block = SyntaxHighlighterBase::findPrevNonBlankBlock(current_block);
            m_foldDetector->setEditor(e);
            m_foldDetector->processBlock(current_block, previous_block, text);
        }
    }
}

//-------------------------------------------------------------------
/*
*/
void SyntaxHighlighterBase::onStateChanged(bool state)
{
    if (onClose())
    {
        return;
    }

    if (state)
    {
        setDocument(editor()->document());
    }
    else
    {
        setDocument(NULL);
    }
}

//-------------------------------------------------------------------
/*
Rehighlight the entire document, may be slow.
*/
void SyntaxHighlighterBase::rehighlight()
{
    QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
    QSyntaxHighlighter::rehighlight();
    QApplication::restoreOverrideCursor();
}

//-------------------------------------------------------------------
/*
Rehighlight the entire document, may be slow.
*/
void SyntaxHighlighterBase::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
    refreshEditor(m_editorStyle);
    document()->setParent(editor);
    setParent(editor);
}

//-------------------------------------------------------------------
/*
Refresh editor settings (background and highlight colors) when color
scheme changed.

:param color_scheme: new color scheme.
*/
void SyntaxHighlighterBase::refreshEditor(QSharedPointer<CodeEditorStyle> editorStyle)
{
    if (editorStyle != m_editorStyle)
    {
        m_editorStyle = editorStyle;
    }

    editor()->setBackground(editorStyle->background());
    editor()->setWhitespacesForeground(editorStyle->format(StyleItem::KeyWhitespace).foreground().color());

    Panel::Ptr panel = editor()->panels()->get("FoldingPanel");

    if (panel)
    {
        QSharedPointer<FoldingPanel> fp = panel.dynamicCast<FoldingPanel>();

        if (fp)
        {
            fp->refreshDecorations(true);
        }
    }

    setDocument(editor()->document());   //update document

    editor()->resetStylesheet();
}

} //end namespace ito
