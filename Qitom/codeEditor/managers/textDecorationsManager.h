#ifndef TEXTDECORATIONSMANAGER_H
#define TEXTDECORATIONSMANAGER_H

/*
Contains the text decorations manager
*/

#include "manager.h"

#include <qlist.h>
#include "../textDecoration.h"

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

    typedef QList<TextDecoration::Ptr>::const_iterator const_iterator;
    typedef QList<TextDecoration::Ptr>::iterator iterator;

    bool append(TextDecoration::Ptr decoration);
    bool remove(TextDecoration::Ptr decoration);
    void clear();

    bool contains(const TextDecoration::Ptr &deco)
    {
        foreach (const TextDecoration::Ptr &t, m_decorations)
        {
            if (t == deco)
            {
                return true;
            }
        }
        return false;
    }

    const_iterator constBegin() const 
    {
          return m_decorations.constBegin(); 
    }
    const_iterator constEnd() const 
    {
          return m_decorations.constEnd(); 
    }

    iterator begin() 
    {
          return m_decorations.begin(); 
    }
    iterator end() 
    {
          return m_decorations.end(); 
    }

private:
    QList<QTextEdit::ExtraSelection> getExtraSelections() const;

    QList<TextDecoration::Ptr> m_decorations;
};
    
#endif