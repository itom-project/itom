#include "modesManager.h"

#include "../codeEditor.h"
#include "../panel.h"


#include <assert.h>
#include <vector>



//---------------------------------------------------------------------
//---------------------------------------------------------------------
ModesManager::ModesManager(CodeEditor *editor, QObject *parent /*= NULL*/) : 
    Manager(editor, parent)
{
}

//---------------------------------------------------------------------
ModesManager::~ModesManager()
{
}



//---------------------------------------------------------------------
/*
Adds a mode to the editor.

:param mode: The mode instance to append.
*/
Mode::Ptr ModesManager::append(Mode::Ptr mode)
{
    m_modes[mode->name()] = mode;
    mode->onInstall(editor());
    return mode;
}


//---------------------------------------------------------------------
/*
Removes a mode from the editor.

:param name_or_klass: The name (or class) of the mode to remove.
:returns: The removed mode.
*/
Mode::Ptr ModesManager::remove(Mode::Ptr mode)
{
    Mode::Ptr out;
    if (m_modes.contains(mode->name()))
    {
        Mode::Ptr &m = m_modes[mode->name()];
        m->onUninstall();
        m_modes.remove(m->name());
        out = m;
    }

    return out;
}

//---------------------------------------------------------------------
/*
Removes all modes from the editor. All modes are removed from list
and deleted.
*/
void ModesManager::clear()
{
    while (m_modes.size() > 0)
    {
        remove(m_modes.first());
    }
}