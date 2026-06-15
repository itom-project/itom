/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#include "modesManager.h"

#include "../codeEditor.h"
#include "../panel.h"


#include <assert.h>
#include <vector>

namespace ito {

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

} //end namespace ito
