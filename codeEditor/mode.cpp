#include "mode.h"

//-------------------------------------------------------------------
Mode::Mode(const QString &name, const QString &description /*= ""*/) :
    m_name(name),
    m_description(description),
    m_enabled(false),
    m_editor(NULL),
    m_onClose(false)
{
}

//-------------------------------------------------------------------
Mode::~Mode()
{
}

//-------------------------------------------------------
QString Mode::name() const
{
    return m_name;
}

//-------------------------------------------------------------------
/*
Installs the extension on the editor.

:param editor: editor widget instance
:type editor: pyqode.core.api.code_edit.CodeEdit

.. note:: This method is called by editor when you install a Mode.
            You should never call it yourself, even in a subclasss.

.. warning:: Don't forget to call **super** when subclassing
*/
void Mode::onInstall(CodeEditor *editor)
{
    m_editor = editor;
    m_enabled = true;
}

//-------------------------------------------------------------------
/*
Uninstalls the mode from the editor.
*/
void Mode::onUninstall()
{
    m_onClose = true;
    m_editor = NULL;
    m_enabled = true;
}

//-------------------------------------------------------------------
/*
Called when the enable state has changed.

This method does not do anything, you may override it if you need
to connect/disconnect to the editor's signals (connect when state is
true and disconnect when it is false).

:param state: True = enabled, False = disabled
:type state: bool
*/
void Mode::onStateChanged(bool state)
{
}