#ifndef AUTOINDENT_H
#define AUTOINDENT_H

/*
Contains the automatic generic indenter
*/

#include "utils/utils.h"
#include "mode.h"
#include <qevent.h>
#include <qobject.h>
#include <qpair.h>
#include <qstring.h>


/*
Indents text automatically.
Generic indenter mode that indents the text when the user press RETURN.

You can customize this mode by overriding
:meth:`pyqode.core.modes.AutoIndentMode._get_indent`
*/
class AutoIndentMode : public QObject, public Mode
{
    Q_OBJECT
public:
    AutoIndentMode(const QString &name, const QString &description = "", QObject *parent = NULL);
    virtual ~AutoIndentMode();

    virtual void onStateChanged(bool state);

private slots:
    void onKeyPressed(QKeyEvent *e);

protected:
    virtual QPair<QString, QString> getIndent(const QTextCursor &cursor) const;

};

#endif
