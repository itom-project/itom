#ifndef SYMBOLMATCHERMODE_H
#define SYMBOLMATCHERMODE_H

/*
This module contains the symbol matcher mode
*/

#include "../textDecoration.h"
#include "../mode.h"
#include "../utils/utils.h"

#include <qcolor.h>
#include <qbrush.h>
#include <qcolor.h>
#include <qtextcursor.h>

/*
Highlights matching symbols (parentheses, braces,...)

.. note:: This mode requires the document to be filled with
    :class:`pyqode.core.api.TextBlockUserData`, i.e. a
    :class:`pyqode.core.api.SyntaxHighlighter` must be installed on
    the editor instance.
*/
class SymbolMatcherMode : public QObject, public Mode
{
    Q_OBJECT
public:

    // symbols indices
    enum Symbols { Paren = 0, Square = 2, Brace = 4 }; //Position of Symbol in chars bytearray

    enum CharType { Open = 0, Close = 1}; //Sub-Position of Symbol in chars bytearray

    static const QByteArray chars;

    SymbolMatcherMode(const QString &description = "", QObject *parent = NULL);
    virtual ~SymbolMatcherMode();

    QBrush matchBackground() const;
    void setMatchBackground(const QBrush &value);

    QColor matchForeground() const;
    void setMatchForeground(const QColor &value);

    QBrush unmatchBackground() const;
    void setUnmatchBackground(const QBrush &value);

    QColor unmatchForeground() const;
    void setUnmatchForeground(const QColor &value);

    virtual void onStateChanged(bool state);

    QPoint symbolPos(const QTextCursor &cursor, CharType charType = Open, Symbols symbolType =Paren); //!< return value is line (x) and column (y)

private slots:
    void doSymbolsMatching();

protected:
    QTextCursor createDecoration(int pos, bool match = true);
    void clearDecorations();
    void match(Symbols symbol, QList<Utils::ParenthesisInfo> &data, int cursorPos);
    bool matchLeft(SymbolMatcherMode::Symbols symbol, const QTextBlock &currentBlock, int i, int cpt);
    bool matchRight(SymbolMatcherMode::Symbols symbol, const QTextBlock &currentBlock, int i, int nbRightParen);
    void refreshDecorations();

    QBrush m_matchBackground;
    QColor m_matchForeground;
    QBrush m_unmatchBackground;
    QColor m_unmatchForeground;
    QList<TextDecoration::Ptr> m_decorations;
};

#endif
