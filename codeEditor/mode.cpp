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
Mode::Mode() :
    m_name(""),
    m_description(""),
    m_enabled(false),
    m_editor(NULL),
    m_onClose(false)
{
}
//-------------------------------------------------------------------
Mode::Mode(const Mode &copy) :
    m_name(copy.m_name),
    m_description(copy.m_description),
    m_enabled(copy.m_enabled),
    m_editor(copy.m_editor),
    m_onClose(copy.m_onClose)
{
}

//-------------------------------------------------------------------
Mode::~Mode()
{
}

//-------------------------------------------------------------------
bool Mode::operator==(const Mode &other) const
{
    return ((m_name == other.m_name) && \
        (m_description == other.m_description) && \
        (m_enabled == other.m_enabled) && \
        (m_editor == other.m_editor) && \
        (m_onClose == other.m_onClose));
}

//-------------------------------------------------------
QString Mode::name() const
{
    return m_name;
}

//-------------------------------------------------------------------
/*
Tells if the mode is enabled,
:meth:`pyqode.core.api.Mode.on_state_changed` will be called as soon
as the mode state changed.

:type: bool
*/
bool Mode::enabled() const
{
    return m_enabled;
}

//-------------------------------------------------------------------
/*
*/
void Mode::setEnabled(bool enabled)
{
    if (enabled != m_enabled)
    {
        m_enabled = enabled;
        onStateChanged(enabled);
    }
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
    setEnabled(true);
}

//-------------------------------------------------------------------
/*
Uninstalls the mode from the editor.
*/
void Mode::onUninstall()
{
    m_onClose = true;
    m_editor = NULL;
    setEnabled(false);
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