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

#ifndef MODE_H
#define MODE_H

#include <qstring.h>
#include <qsharedpointer.h>
#include <qaction.h>
#include <qlist.h>

namespace ito {

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
    typedef QSharedPointer<Mode> Ptr;

    Mode();
    Mode(const Mode &copy);
    Mode(const QString &name, const QString &description = "");
    virtual ~Mode();

    bool operator==(const Mode &other) const;

    virtual void onInstall(CodeEditor *editor);
    virtual void onUninstall();
    virtual void onStateChanged(bool state);

    QString name() const;

    bool enabled() const;
    void setEnabled(bool enabled);

    inline CodeEditor *editor() const { return m_editor; }
    bool onClose() const { return m_onClose; }

    virtual QList<QAction*> actions() const { return QList<QAction*>(); }

private:
    QString m_name;
    QString m_description;
    bool m_enabled;
    CodeEditor *m_editor;
    bool m_onClose;
};

} //end namespace ito

#endif
