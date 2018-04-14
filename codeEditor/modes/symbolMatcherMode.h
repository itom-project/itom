#ifndef SYMBOLMATCHERMODE_H
#define SYMBOLMATCHERMODE_H

/*
This module contains the symbol matcher mode
*/

#include "textDecoration.h"
#include "mode.h"

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
    enum Symbols { Paren = 0, Square = 1, Brace = 2 };

    enum SymbolsSubtype { Open = 0, Close = 1};

    SymbolMatcherMode(QObject *parent = NULL);
    virtual ~SymbolMatcherMode();

    void doSymbolsMatching();

    QColor background() const;
    void setBackground(const QColor &color);

    virtual void onInstall(CodeEditor *editor);
    virtual void onStateChanged(bool state);

public slots:
    void refresh();

protected:
    QTextCursor createDecoration(int pos, bool match = true);
    void clearDecorations();
    void match(SymbolMatcherMode::Symbols symbol, QList<Utils::ParenthesisInfo> &data, int cursorPos);

    QBrush m_matchBackground;
    QColor m_matchForeground;
    QBrush m_unmatchBackground;
    QColor m_unmatchForeground;
    QList<TextDecoration::Ptr> m_decorations;
};

#endif
