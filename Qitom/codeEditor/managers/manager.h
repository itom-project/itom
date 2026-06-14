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

#ifndef MANAGER_H
#define MANAGER_H

#include <qobject.h>

namespace ito {

/*
This module contains the Manager API.
*/

class CodeEditor;

/*
A manager manages a specific aspect of a CodeEdit instance:
    - backend management (start/stop server, request work,...)
    - modes management
    - panels management and drawing
    - file manager
Managers are typically created internally when you create a CodeEdit.
You interact with them later, e.g. when you want to start the backend
process or when you want to install/retrieve a mode or a panel.
::
    editor = CodeEdit()
    # use the backend manager to start the backend server
    editor.backend.start(...)
    editor.backend.send_request(...)
    # use the panels controller to install a panel
    editor.panels.install(MyPanel(), MyPanel.Position.Right)
    my_panel = editor.panels.get(MyPanel)
    # and so on
*/

class Manager : public QObject
{
    Q_OBJECT

public:
    Manager(CodeEditor *editor, QObject *parent = NULL);
    virtual ~Manager();

    CodeEditor* editor() const;

private:
    CodeEditor* m_pEditor;
};

} //end namespace ito


#endif
