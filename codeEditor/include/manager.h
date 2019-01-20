#ifndef MANAGER_H
#define MANAGER_H

/*
This module contains the Manager API.
*/

class CodeEditor;

#include <qobject.h>

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


#endif