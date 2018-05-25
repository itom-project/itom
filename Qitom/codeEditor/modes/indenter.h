#ifndef INDENTER_H
#define INDENTER_H

/*
Contains the default indenter.
*/

#include "../mode.h"
#include <qobject.h>
#include <qstring.h>
#include <qtextcursor.h>



/*
Implements classic indentation/tabulation (Tab/Shift+Tab)

It inserts/removes tabulations (a series of spaces defined by the
tabLength settings) at the cursor position if there is no selection,
otherwise it fully indents/un-indents selected lines.

To trigger an indentation/un-indentation programatically, you must emit
:attr:`pyqode.core.api.CodeEdit.indent_requested` or
:attr:`pyqode.core.api.CodeEdit.unindent_requested`.
*/
class IndenterMode : public QObject, public Mode
{
    Q_OBJECT
public:
    IndenterMode(const QString &description = "", QObject *parent = NULL);
    virtual ~IndenterMode();

    virtual void onStateChanged(bool state);

    void indentSelection(QTextCursor cursor) const;
    QTextCursor unindentSelection(QTextCursor cursor) const;
    
    int countDeletableSpaces(const QTextCursor &cursor, int maxSpaces) const;

private slots:
    void indent() const;
    void unindent() const;

protected:
    

};

#endif
