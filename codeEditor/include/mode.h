#ifndef MODE_H
#define MODE_H

#include <qstring.h>

class CodeEditor; //forware declaration

/*
Base class for editor extensions. An extension is a "thing" that can be
installed on an editor to add new behaviours or to modify its appearance.

A mode is added to an editor by using the ModesManager/PanelsManager:

    - :meth:`pyqode.core.api.CodeEdit.modes.append` or
    - :meth:`pyqode.core.api.CodeEdit.panels.append`

Subclasses may/should override the following methods:

    - :meth:`pyqode.core.api.Mode.on_install`
    - :meth:`pyqode.core.api.Mode.on_uninstall`
    - :meth:`pyqode.core.api.Mode.on_state_changed`

..warning: The mode will be identified by its class name, this means that

**there cannot be two modes of the same type on the same editor instance!**
*/
class Mode
{
public:
    Mode(const QString &name, const QString &description = "");
    virtual ~Mode();

    virtual void onInstall(CodeEditor *editor);
    virtual void onUninstall();
    virtual void onStateChanged(bool state);

    QString name() const;

protected:
    

private:
    QString m_name;
    QString m_description;
    bool m_enabled;
    CodeEditor *m_editor;
    bool m_onClose;
};

#endif