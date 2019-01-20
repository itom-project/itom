#ifndef PYSYNTAXHIGHLIGHER_H
#define PYSYNTAXHIGHLIGHER_H

#include "syntaxHighlighterBase.h"

#define MIN_QT_REGULAREXPRESSION_VERSION 0x050100
#if QT_VERSION >= MIN_QT_REGULAREXPRESSION_VERSION
    #include <qregularexpression.h>
#endif

#include <qregexp.h>
#include <qtextformat.h>

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

#if QT_VERSION >= MIN_QT_REGULAREXPRESSION_VERSION
    typedef QRegularExpression QQRegExp;
#else
    typedef QRegExp QQRegExp;
#endif

    PythonSyntaxHighlighter(QTextDocument *parent, const QString &description = "", const ColorScheme &colorScheme = ColorScheme());

    virtual ~PythonSyntaxHighlighter();

    /*
    Abstract method. Override this to apply syntax highlighting.
    
    :param text: Line of text to highlight.
    :param block: current block
    */
    void highlight_block(const QString &text, QTextBlock &block);

    virtual void rehighlight();

private:
    enum State //!< Syntax highlighting states (from one text block to another):
    {
        Normal = 0,
        InsideSq3String = 1,
        InsideDq3String = 2,
        InsideSqString = 3,
        InsideDqString = 4
    };

    //syntax highlighting rules
    static QMap<QString,QQRegExp> regExpProg;
    static QRegExp regExpIdProg;
    static QRegExp regExpAsProg;
    static QQRegExp regExpOeComment; //comments suitable for outline explorer

    QList<QTextBlock> m_docstrings;
    QList<QTextBlock> m_importStatements;

    QTextCharFormat getFormatFromStyle(ColorScheme::Keys token) const;
    const QTextCharFormat getTextCharFormat(const QString &colorName, const QString &style = QString());

    static QMap<QString,QQRegExp> makePythonPatterns(const QStringList &additionalKeywords = QStringList(), const QStringList &additionalBuiltins = QStringList());
};

#endif