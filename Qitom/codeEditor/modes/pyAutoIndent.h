#ifndef PYAUTOINDENT_H
#define PYAUTOINDENT_H

/*
Contains python smart indent modes
*/

#include "autoindent.h"
#include "../utils/utils.h"

#include <qtextcursor.h>
#include <qstring.h>


/*
Automatically indents text, respecting the PEP8 conventions.

Customised :class:`pyqode.core.modes.AutoIndentMode` for python
that tries its best to follow the pep8 indentation guidelines.
*/
class PyAutoIndentMode : public AutoIndentMode
{
    Q_OBJECT
public:
    PyAutoIndentMode(const QString &description = "", QObject *parent = NULL);
    virtual ~PyAutoIndentMode();

    virtual void onInstall(CodeEditor *editor);

private slots:

protected:
    virtual QPair<QString, QString> getIndent(const QTextCursor &cursor) const;

    QPair<QString, QString> handleIndentBetweenParen(int column, const QString &line, const QPair<QString, QString> &parent_impl, const QTextCursor &cursor) const;
    void handleIndentInsideString(const QString &c, const QTextCursor &cursor, const QString &fullline, QString &post, QString &pre) const;
    QString handleNewScopeIndentation(const QTextCursor &cursor, const QString &fullline) const;
    void handleIndentInStatement(const QString &fullline, const QString &lastword, QString &post, QString &pre) const;
    void handleIndentAfterParen(const QTextCursor &cursor, QString &post) const; 
    bool betweenParen(const QTextCursor &cursor, int column) const;
    QPair<int, QChar> getFirstOpenParen(const QTextCursor &tc, int column) const;
    void getParenPos(const QTextCursor &cursor, int column, int &ol, int &oc, int &cl, int &cc) const;
    void parensCountForBlock(int column, const QTextBlock &block, int &numOpenParentheses, int &numClosedParentheses) const;
    int getIndentOfOpeningParen(const QTextCursor &cursor) const;

    static QPair<bool, QChar> isInStringDef(const QString &fullline, int column);
    static bool isParenOpen(const Utils::ParenthesisInfo &paren);
    static bool isParenClosed(const Utils::ParenthesisInfo &paren);
    static QString getFullLine(const QTextCursor &cursor);
    static QString getLastWordUnstripped(const QTextCursor &cursor);
    static QString getLastWord(const QTextCursor &cursor);
    static QChar getPrevChar(const QTextCursor &cursor);
    static QChar getNextChar(const QTextCursor &cursor);
    static bool atBlockEnd(const QTextCursor &cursor, const QString &fullline);
    static bool atBlockStart(const QTextCursor &cursor, const QString &line);

};

#endif
