#include "pythonSyntaxHighlighter.h"

#include "codeEditor.h"
#include <qapplication.h>
#include <qtextdocument.h>

#include "managers/modesManager.h"
#include "managers/panelsManager.h"
#include "modes/caretLineHighlight.h"


//-------------------------------------------------------------------
PythonSyntaxHighlighter::PythonSyntaxHighlighter(QTextDocument *parent, const QString &description /*= ""*/, const ColorScheme &colorScheme /*=  = ColorScheme()*/) :
    SyntaxHighlighterBase("PythonSyntaxHighlighter", parent, description, colorScheme)
{
    m_keywords = QStringList() << "and" << "assert" << "break" << "class" << "continue" << "def" <<
        "del" << "elif" << "else" << "except" << "exec" << "finally" <<
        "for" << "from" << "global" << "if" << "import" << "in" <<
        "is" << "lambda" << "not" << "or" << "pass" << "print" <<
        "raise" << "return" << "try" << "while" << "yield" <<
        "None" << "True" << "False";
 
    m_operators = QStringList() << "=" <<
        // Comparison
        "==" << "!=" << "<" << "<=" << ">" << ">=" <<
        // Arithmetic
        "\\+" << "-" << "\\*" << "/" << "//" << "%" << "\\*\\*" <<
        // In-place
        "\\+=" << "-=" << "\\*=" << "/=" << "%=" <<
        // Bitwise
        "\\^" << "\\|" << "&" << "~" << ">>" << "<<";

    m_braces = QStringList() << "{" << "}" << "\\(" << "\\)" << "\\[" << "]";

    m_triSingleQuote.setPattern("'''");
    m_triDoubleQuote.setPattern("\"\"\"");

    initializeRules();
}

//-------------------------------------------------------------------
PythonSyntaxHighlighter::~PythonSyntaxHighlighter()
{
}

//-------------------------------------------------------------------
const QTextCharFormat PythonSyntaxHighlighter::getTextCharFormat(const QString &colorName, const QString &style)
{
    QTextCharFormat charFormat;
    QColor color(colorName);
    charFormat.setForeground(color);
    if (style.contains("bold", Qt::CaseInsensitive))
    {
        charFormat.setFontWeight(QFont::Bold);
    }
    if (style.contains("italic", Qt::CaseInsensitive))
    {
        charFormat.setFontItalic(true);
    }
    return charFormat;
}

//-------------------------------------------------------------------
void PythonSyntaxHighlighter::onInstall(CodeEditor *editor)
{
    clearCaches();
    SyntaxHighlighterBase::onInstall(editor);
}


//-------------------------------------------------------------------
void PythonSyntaxHighlighter::initializeRules()
{
    foreach (const QString &curKeyword, m_keywords)
    {
        m_rules.append(HighlightingRule(QString("\\b%1\\b").arg(curKeyword), 0, ColorScheme::KeyKeyword));
    }
    foreach (const QString &curOperator, m_operators)
    {
        m_rules.append(HighlightingRule(QString("%1").arg(curOperator), 0, ColorScheme::KeyOperator));
    }
    /*foreach (const QString &curBrace, m_braces)
    {
        m_rules.append(HighlightingRule(QString("%1").arg(curBrace), 0, m_basicStyles.value("brace")));
    }*/
    // 'self'
    m_rules.append(HighlightingRule("\\bself\\b", 0, ColorScheme::KeySelf));

    // Double-quoted string, possibly containing escape sequences
    // FF: originally in python : r'"[^"\\]*(\\.[^"\\]*)*"'
    m_rules.append(HighlightingRule("\"[^\"\\\\]*(\\\\.[^\"\\\\]*)*\"", 0, ColorScheme::KeyString));
    // Single-quoted string, possibly containing escape sequences
    // FF: originally in python : r"'[^'\\]*(\\.[^'\\]*)*'"
    m_rules.append(HighlightingRule("'[^'\\\\]*(\\\\.[^'\\\\]*)*'", 0, ColorScheme::KeyString));

    // 'def' followed by an identifier
    // FF: originally: r'\bdef\b\s*(\w+)'
    m_rules.append(HighlightingRule("\\bdef\\b\\s*(\\w+)", 1, ColorScheme::KeyClass));
    //  'class' followed by an identifier
    // FF: originally: r'\bclass\b\s*(\w+)'
    m_rules.append(HighlightingRule("\\bclass\\b\\s*(\\w+)", 1, ColorScheme::KeyClass));

    // From '#' until a newline
    // FF: originally: r'#[^\\n]*'
    m_rules.append(HighlightingRule("#[^\\n]*", 0, ColorScheme::KeyComment));

    // Numeric literals
    m_rules.append(HighlightingRule("\\b[+-]?[0-9]+[lL]?\\b", 0, ColorScheme::KeyNumber)); // r'\b[+-]?[0-9]+[lL]?\b'
    m_rules.append(HighlightingRule("\\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\\b", 0, ColorScheme::KeyNumber)); // r'\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b'
    m_rules.append(HighlightingRule("\\b[+-]?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\\b", 0, ColorScheme::KeyNumber)); // r'\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b'
}

//-------------------------------------------------------------------
/*
Highlights the block using a pygments lexer.

:param text: text of the block to highlight
:param block: block to highlight
*/
void PythonSyntaxHighlighter::highlight_block(const QString &text, QTextBlock &block)
{
    /*if self.color_scheme.name != self._pygments_style
    {
        self._pygments_style = self.color_scheme.name
        self._update_style()
    }*/

    QString originalText = text;
    if (editor() && enabled()) //&& self._lexer
    {
        if (block.blockNumber())
        {
            TextBlockUserData *prev_data = static_cast<TextBlockUserData*>(m_previousBlock.userData());
            if (prev_data)
            {
                if (prev_data->m_syntaxStack.isNull() == false)
                {

                    m_savedStateStack = prev_data->m_syntaxStack;
                }
                else
                {
                    m_savedStateStack.clear();
                }
            }
        }

        //Lex the text using Pygments
        int index = 0;
        QTextCharFormat fmt;
        int length;
        QTextBlockUserData *usd = block.userData();
        TextBlockUserData *usdSpecific = static_cast<TextBlockUserData*>(usd);

        if (usd == NULL)
        {
            usdSpecific = new TextBlockUserData();
            block.setUserData(usdSpecific);
            usd = usdSpecific;
        }

        QList<LexerResult> tokens = execLexer(text);

        foreach (const LexerResult &r, tokens)
        {
            fmt = getFormatFromStyle(r.token);
            if ((r.token == ColorScheme::KeyString) ||
                (r.token == ColorScheme::KeyDocstring) ||
                (r.token == ColorScheme::KeyComment))
            {
                fmt.setObjectType(QTextCharFormat::UserObject);
            }
            setFormat(r.index, r.length, fmt);
        }

        if (!m_savedStateStack.isNull() && usdSpecific)
        {
            usdSpecific->m_syntaxStack = m_savedStateStack;
            //Clean up for the next go-round.
            m_savedStateStack.clear();
        }

        // spaces
        QRegExp expression("\\s+");
        index = expression.indexIn(originalText, 0);
        
        while (index >= 0)
        {
            index = expression.pos(0);
            length = expression.cap(0).size();
            setFormat(index, length, getFormatFromStyle(ColorScheme::KeyWhitespace));
            index = expression.indexIn(originalText, index + length);
        }

        m_previousBlock = block;
    }
}

//---------------------------------------------------------
QList<PythonSyntaxHighlighter::LexerResult> PythonSyntaxHighlighter::execLexer(const QString &text)
{ 
    QList<LexerResult> tokens;

    foreach (HighlightingRule curRule, m_rules)
    {
        int idx = curRule.m_pattern.indexIn(text, 0);
        while (idx >= 0)
        {
        // Get index of Nth match
        idx = curRule.m_pattern.pos(curRule.m_nth);
        int length = curRule.m_pattern.cap(curRule.m_nth).length();
        tokens.append(LexerResult(idx, length, curRule.m_token));
        idx = curRule.m_pattern.indexIn(text, idx + length);
        }
    }
    
    setCurrentBlockState(0);

    // Do multi-line strings
    bool isInMultiline = matchMultiline(text, m_triSingleQuote, 1, ColorScheme::KeyDocstring, tokens);
    if (!isInMultiline)
    {
        isInMultiline = matchMultiline(text, m_triDoubleQuote, 2, ColorScheme::KeyDocstring, tokens);
    }
    return tokens;
}

//-------------------------------------------------------------------
bool PythonSyntaxHighlighter::matchMultiline(const QString &text, const QRegExp &delimiter, const int inState, ColorScheme::Keys token, QList<PythonSyntaxHighlighter::LexerResult> &tokens)
{
    int start = -1;
    int add = -1;
    int end = -1;
    int length = 0;

    // If inside triple-single quotes, start at 0
    if (previousBlockState() == inState) 
    {
        start = 0;
        add = 0;
    }
    // Otherwise, look for the delimiter on this line
    else
    { 
        start = delimiter.indexIn(text);
        // Move past this match
        add = delimiter.matchedLength();
    }

    // As long as there's a delimiter match on this line...
    while (start >= 0) 
    {
        // Look for the ending delimiter
        end = delimiter.indexIn(text, start + add);
        // Ending delimiter on this line?
        if (end >= add) 
        {
            length = end - start + add + delimiter.matchedLength();
            setCurrentBlockState(0);
        }
        // No; multi-line string
        else 
        {
            setCurrentBlockState(inState);
            length = text.length() - start + add;
        }
        // Apply formatting and look for next

        tokens.append(LexerResult(start, length, token));       
        start = delimiter.indexIn(text, start + length);
    }

    // Return True if still inside a multi-line string, False otherwise
    if (currentBlockState() == inState)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//-------------------------------------------------------------------
/*
Returns a QTextCharFormat for token
*/
QTextCharFormat PythonSyntaxHighlighter::getFormatFromStyle(ColorScheme::Keys token) const
{
    return m_colorScheme[token];
}


//-------------------------------------------------------------------
/*
Clear caches for brushes and formats.
*/
void PythonSyntaxHighlighter::clearCaches()
{
}

