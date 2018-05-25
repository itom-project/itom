#ifndef MODESMANAGER_H
#define MODESMANAGER_H

/*
This module contains the modes controller.
*/

#include "manager.h"

#include <qmap.h>
#include "../mode.h"

/*
Manages the list of modes of the code edit widget.
*/
class ModesManager : public Manager
{
    Q_OBJECT

public:
    ModesManager(CodeEditor *editor, QObject *parent = NULL);
    virtual ~ModesManager();

    typedef QMap<QString, Mode::Ptr>::const_iterator const_iterator;
    typedef QMap<QString, Mode::Ptr>::iterator iterator;

    Mode::Ptr append(Mode::Ptr mode);
    Mode::Ptr remove(Mode::Ptr mode);
    void clear();

    Mode::Ptr get(const QString &name) const
    {
        if (m_modes.contains(name))
        {
            return m_modes[name];
        }
        return Mode::Ptr();
    }

    const_iterator constBegin() const 
    {
          return m_modes.constBegin(); 
    }
    const_iterator constEnd() const 
    {
          return m_modes.constEnd(); 
    }

    iterator begin() 
    {
          return m_modes.begin(); 
    }
    iterator end() 
    {
          return m_modes.end(); 
    }

private:

    QMap<QString, Mode::Ptr> m_modes;
};
    
#endif