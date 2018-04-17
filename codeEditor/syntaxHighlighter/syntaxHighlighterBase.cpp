#include "syntaxHighlighterBase.h"

#include "codeEditor.h"
#include <qapplication.h>
#include <qtextdocument.h>

#include "managers/modesManager.h"
#include "managers/panelsManager.h"
#include "modes/caretLineHighlight.h"
#include "utils/utils.h"

//------------------------------------------------------------------
ColorScheme::ColorScheme()
{
    QBrush bgcolor;
    bgcolor.setColor("white");

    //create defaults
    for (int i = 0; i < Last; ++i)
    {
        m_formats[i] = createFormat(QBrush("black"), bgcolor);
    }

    m_formats[KeyKeyword] = createFormat(QBrush("blue"));
    m_formats[KeyInstance] = createFormat(QBrush("gray"));
    m_formats[KeyOperator] = createFormat(QBrush("red"));
    m_formats[KeyConstant] = createFormat(QBrush("blue"), QBrush(), true);
    m_formats[KeyNamespace] = createFormat(QBrush("red"), QBrush(), false, true);
    
    //m_formats[Keybrace] = createFormat(QBrush("darkGray"));
    m_formats[KeyClass] = createFormat(QBrush("black"), QBrush(), true);
    m_formats[KeyString] = createFormat(QBrush("magenta"));
    //m_formats[KeyString2] = createFormat(QBrush("darkMagenta"));
    m_formats[KeyComment] = createFormat(QBrush("darkGreen"), QBrush(), false, true);
    m_formats[KeySelf] = createFormat(QBrush("gray"), QBrush(), false, true);
    m_formats[KeyNumber] = createFormat(QBrush("darkYellow"));
    m_formats[KeyDecorator] = createFormat(QBrush("darkYellow"), QBrush(), true, true);
    m_formats[KeyHighlight] = createFormat(QBrush(), Utils::driftColor(bgcolor.color(), 110));
    m_formats[KeyBuiltin] = createFormat(QBrush("yellow"));
    m_formats[KeyOperatorWord] = createFormat(QBrush("pink"));
    m_formats[KeyFunction] = createFormat(QBrush("darkBlue"), QBrush(), true);
    m_formats[KeyDefinition] = createFormat(QBrush("darkRed"), QBrush(), true);
    m_formats[KeyDocstring] = createFormat(QBrush("darkCyan"), QBrush(), false, true);

    /*regExpressions["builtin_fct"] = QRegExp(builtin_fct);
    regExpressions["uf_sqstring"] = QRegExp(ufstring1);
    regExpressions["uf_dqstring"] = QRegExp(ufstring2);
    regExpressions["uf_sq3string"] = QRegExp(ufstring3);
    regExpressions["uf_dq3string"] = QRegExp(ufstring4);
    regExpressions["SYNC"] = QRegExp(any("SYNC", QStringList("\\n")));*/
    

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

//------------------------------------------------------------------
QTextCharFormat ColorScheme::createFormat(const QBrush &color, const QBrush &bgcolor /*= QBrush()*/, bool bold /*= false*/, \
    bool italic /*= false*/, bool underline /*= false*/, QFont::StyleHint styleHint /* = QFont::SansSerif*/)
{
    QTextCharFormat f;
    f.setForeground(color);
    f.setBackground(bgcolor);
    if (bold)
    {
        f.setFontWeight(QFont::Bold);
    }
    f.setFontItalic(italic);
    if (underline)
    {
        f.setUnderlineStyle(QTextCharFormat::SingleUnderline);
    }
    f.setFontStyleHint(styleHint);
    return f;
}

//-------------------------------------------------------------------
SyntaxHighlighterBase::SyntaxHighlighterBase(const QString &name, QTextDocument *parent, const QString &description /*= ""*/, const ColorScheme &colorScheme /*=  = ColorScheme()*/) :
    QSyntaxHighlighter(parent),
    Mode(name, description)
{
    m_colorScheme =(colorScheme);
    m_regSpacesPtrn=(QRegExp("[ \\t]+"));
    m_regWhitespaces=(QRegExp("\\s+"));
    m_foldDetector=(NULL);
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
        //todo: static_cast<FoldingPanel*>(panel)->refreshDecorations(force=true);
    }
    editor()->resetStylesheet();
}