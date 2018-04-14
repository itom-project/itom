#ifndef PYSYNTAXHIGHLIGHER_H
#define PYSYNTAXHIGHLIGHER_H

#include "syntaxHighlighterBase.h"
#include <qregexp.h>
#include <qtextformat.h>

//! Container to describe a highlighting rule. Based on a regular expression, a relevant match # and the format.
class HighlightingRule
{
public: 
    HighlightingRule(const QString &patternStr, int n, ColorScheme::Keys token) :
        m_originalRuleStr(patternStr),
        m_pattern(QRegExp(patternStr)),
        m_nth(n),
        m_token(token)
    {
    } 

    QString m_originalRuleStr;
    QRegExp m_pattern;
    int m_nth;
    ColorScheme::Keys m_token;
};

/*
Python Syntax Highlighter
*/
class PythonSyntaxHighlighter : public SyntaxHighlighterBase
{
    Q_OBJECT
public:
    PythonSyntaxHighlighter(QTextDocument *parent, const QString &description = "", const ColorScheme &colorScheme = ColorScheme());

    virtual ~PythonSyntaxHighlighter();

    virtual void onInstall(CodeEditor *editor);

    /*
    Abstract method. Override this to apply syntax highlighting.
    
    :param text: Line of text to highlight.
    :param block: current block
    */
    void highlight_block(const QString &text, QTextBlock &block);

private:
    QStringList m_keywords;
    QStringList m_operators;
    QStringList m_braces;
 
    void initializeRules();
    void clearCaches();

    struct LexerResult
    {
        LexerResult(int idx, int len, ColorScheme::Keys t) : index(idx), length(len), token(t) {}
        int index;
        int length;
        ColorScheme::Keys token;
    };

    QTextBlock m_previousBlock;
    QSharedPointer<TextBlockUserData> m_savedStateStack;

    QTextCharFormat getFormatFromStyle(ColorScheme::Keys token) const;

    QList<LexerResult> execLexer(const QString &text);
 
    //! Highlights multi-line strings, returns true if after processing we are still within the multi-line section.
    bool matchMultiline(const QString &text, const QRegExp &delimiter, const int inState, const QTextCharFormat &style);
    const QTextCharFormat getTextCharFormat(const QString &colorName, const QString &style = QString());

    QList<HighlightingRule> m_rules;
    QRegExp m_triSingleQuote;
    QRegExp m_triDoubleQuote;
};

#endif