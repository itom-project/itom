#ifndef SYNTAXHIGHLIGHTERBASE_H
#define SYNTAXHIGHLIGHTERBASE_H

#include <qstring.h>
#include <qtextedit.h>
#include <qsyntaxhighlighter.h>
#include <qregexp.h>
#include <qpointer.h>
#include <qmap.h>

#include "../textBlockUserData.h"
#include "../mode.h"
#include "../foldDetector/foldDetector.h"

class CodeEditor; //forware declaration

class ColorScheme
{
public:
    enum Keys
    {
        KeyBackground = 0,
        KeyHighlight = 1,
        KeyNormal = 2,
        KeyKeyword = 3,
        KeyNamespace = 4,
        KeyType = 5,
        KeyKeywordReserved = 6,
        KeyBuiltin = 7,
        KeyDefinition = 8,
        KeyComment = 9,
        KeyString = 10,
        KeyDocstring = 11,
        KeyNumber = 12,
        KeyInstance = 13,
        KeyWhitespace = 14,
        KeyTag = 15,
        KeySelf = 16,
        KeyDecorator = 17,
        KeyPunctuation = 18,
        KeyConstant = 19,
        KeyFunction = 20,
        KeyOperator = 21,
        KeyOperatorWord = 22,
        KeyClass = 23,
        Last = 24 /*always the highest number*/
    };

    ColorScheme();
    virtual ~ColorScheme();

    QTextCharFormat operator[](int idx) const;

    QTextCharFormat createFormat(const QBrush &color, const QBrush &bgcolor = QBrush(), bool bold = false, bool italic = false, bool underline = false, QFont::StyleHint styleHint = QFont::SansSerif);

    QColor background() const;
    QColor highlight() const;

private:
    QHash<int, QTextCharFormat> m_formats;
};



/*
Syntax Highlighters should derive from this mode instead of mode directly.
*/
class SyntaxHighlighterBase : public QSyntaxHighlighter, public Mode
{
    Q_OBJECT
public:
    SyntaxHighlighterBase(const QString &name, QTextDocument *parent, const QString &description = "", const ColorScheme &colorScheme = ColorScheme());

    virtual ~SyntaxHighlighterBase();

    void setFoldDetector(QSharedPointer<FoldDetector> foldDetector);

    virtual void onStateChanged(bool state);
    virtual void onInstall(CodeEditor *editor);

    const ColorScheme& colorScheme() const { return m_colorScheme; }
    
    /*
    Highlights a block of text. Please do not override, this method.
    Instead you should implement
    :func:`pyqode.core.api.SyntaxHighlighter.highlight_block`.

    :param text: text to highlight.
    */
    void highlightBlock(const QString &text);

    void refreshEditor(const ColorScheme &colorScheme);
    
    /*
    Abstract method. Override this to apply syntax highlighting.
    
    :param text: Line of text to highlight.
    :param block: current block
    */
    virtual void highlight_block(const QString &text, QTextBlock &block) = 0;

    virtual void rehighlight();

protected:
    static QTextBlock findPrevNonBlankBlock(const QTextBlock &currentBlock);

    void highlightWhitespaces(const QString &text);

    QRegExp m_regWhitespaces;
    QRegExp m_regSpacesPtrn;
    ColorScheme m_colorScheme;
    QSharedPointer<FoldDetector> m_foldDetector;
};

#endif