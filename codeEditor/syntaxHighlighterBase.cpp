#include "syntaxHighlighterBase.h"

#include "codeEditor.h"
#include <qapplication.h>
#include <qtextdocument.h>

#include "managers/modesManager.h"
#include "managers/panelsManager.h"
#include "modes/caretLineHighlight.h"

//------------------------------------------------------------------
ColorScheme::ColorScheme()
{
    //create defaults
    for (int i = 0; i < Last; ++i)
    {
        m_formats[i] = (QTextCharFormat());
    }
}

//------------------------------------------------------------------
ColorScheme::~ColorScheme()
{
}

//------------------------------------------------------------------
/*
Gets the background color.
:return:
*/
QColor ColorScheme::background() const
{
    return m_formats[KeyBackground].background().color();
}

//------------------------------------------------------------------
/*
Gets the highlight color.
:return:
*/
QColor ColorScheme::highlight() const
{
    return m_formats[KeyHighlight].background().color();
}

//------------------------------------------------------------------
QTextCharFormat ColorScheme::operator[](int idx) const
{
    if (idx >= 0 && idx < m_formats.size())
    {
        return m_formats[idx];
    }
    return QTextCharFormat();
}

//-------------------------------------------------------------------
SyntaxHighlighterBase::SyntaxHighlighterBase(const QString &name, const QString &description /*= ""*/, const ColorScheme &colorScheme /*=  = ColorScheme()*/, QObject *parent /*= NULL*/) :
    QSyntaxHighlighter(parent),
    Mode(name, description),
    m_colorScheme(colorScheme),
    m_regSpacesPtrn(QRegExp("[ \\t]+")),
    m_regWhitespaces(QRegExp("\\s+")),
    m_foldDetector(NULL)
{
}

//-------------------------------------------------------------------
SyntaxHighlighterBase::~SyntaxHighlighterBase()
{
}

//-------------------------------------------------------------------
/*static*/ QTextBlock SyntaxHighlighterBase::findPrevNonBlankBlock(const QTextBlock &currentBlock)
{
    QTextBlock previousBlock = currentBlock.blockNumber() ? currentBlock.previous() : QTextBlock();
    // find the previous non-blank block
    while (previousBlock.isValid() && previousBlock.blockNumber() && previousBlock.text().trimmed() == "")
    {
        previousBlock = previousBlock.previous();
    }
    return previousBlock;
}

//-------------------------------------------------------------------
void SyntaxHighlighterBase::highlightWhitespaces(const QString &text)
{
    int index = m_regWhitespaces.indexIn(text, 0);
    int length;

    while (index >= 0)
    {
        index = m_regWhitespaces.pos(0);
        length = m_regWhitespaces.cap(0).size();
        setFormat(index, length, m_colorScheme[ColorScheme::KeyWhitespace]);
        index = m_regWhitespaces.indexIn(text, index + length);
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
    QTextBlock previous_block = SyntaxHighlighterBase::findPrevNonBlankBlock(current_block);
    if (editor())
    {
        CodeEditor *e = editor();
        highlight_block(text, current_block);
        if (e->showWhitespaces())
        {
            highlightWhitespaces(text);
        }
        if (m_foldDetector.isNull() == false)
        {
            m_foldDetector->setEditor(editor());
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
    refreshEditor(m_colorScheme);
    document()->setParent(editor);
    setParent(editor);
}

//-------------------------------------------------------------------
/*
Refresh editor settings (background and highlight colors) when color
scheme changed.

:param color_scheme: new color scheme.
*/
void SyntaxHighlighterBase::refreshEditor(const ColorScheme &colorScheme)
{
    editor()->setBackground(colorScheme.background());
    editor()->setForeground(colorScheme[ColorScheme::KeyNormal].foreground().color());
    editor()->setWhitespacesForeground(colorScheme[ColorScheme::KeyWhitespace].foreground().color());
    Mode::Ptr mode = editor()->modes()->get("CaretLineHighlighterMode");
    if (mode)
    {
        CaretLineHighlighterMode* clh = static_cast<CaretLineHighlighterMode*>(mode.data());
        clh->setBackground(colorScheme.highlight());
        clh->refresh();
    }

    Panel* panel = editor()->panels()->get("FoldingPanel");
    if (panel)
    {
        static_cast<FoldingPanel*>(panel)->refreshDecorations(force=true);
    }
    editor()->resetStylesheet();
}