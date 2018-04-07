#ifndef TEXTDECORATIONSMANAGER_H
#define TEXTDECORATIONSMANAGER_H

/*
Contains the text decorations manager
*/

#include "manager.h"

#include <qlist.h>
#include "textDecoration.h"

/*
Manages the collection of TextDecoration that have been set on the editor
widget.
*/
class TextDecorationsManager : public Manager
{
    Q_OBJECT

public:
    TextDecorationsManager(CodeEditor *editor, QObject *parent = NULL);
    virtual ~TextDecorationsManager();

    bool append(const TextDecoration &decoration);
    bool remove(const TextDecoration &decoration);
    void clear();

private:
    QList<QTextEdit::ExtraSelection> getExtraSelections() const;

    QList<TextDecoration> m_decorations;
};
    
#endif