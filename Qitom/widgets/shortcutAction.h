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
*********************************************************************** */

#ifndef SHORTCUTACTION_H
#define SHORTCUTACTION_H

#include "../global.h"
#include "../common/sharedStructures.h"

#include <qaction.h>
#include <qobject.h>
#include <qstring.h>
#include <qshortcut.h>
#include <qpointer.h>

namespace ito
{

class AbstractDockWidget; //forward declaration

class ShortcutAction : public QObject
{
public:
    //!< simple action with text only (no shortcut)
    ShortcutAction(const QString &text, AbstractDockWidget *parent);

    //!< simple action with text and icon only (no shortcut)
    ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent);

    //!< simple action with text, icon and shortcut (context is the same in docked and undocked state)
    ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent,
        const QKeySequence &key, Qt::ShortcutContext context = Qt::WindowShortcut);

    //!< simple action with text, icon and shortcut (different contexts for docked and undocked state)
    ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent,
        const QKeySequence &key, Qt::ShortcutContext dockedContext,
        Qt::ShortcutContext undockedContext);

    ~ShortcutAction();

    void connectTrigger(const QObject *receiver, const char *method, Qt::ConnectionType type = Qt::AutoConnection);

    void setEnabled(bool actionEnabled, bool shortcutEnabled);

    void setEnabled(bool enabled);

    void setVisible(bool actionVisible, bool shortcutEnabled);

    void setVisible(bool visible);

    QAction* action() const { return m_action.data(); }

private:
    QPointer<QAction> m_action;
    QPointer<QShortcut> m_shortcut;
    Qt::ShortcutContext m_dockedShortcut;
    Qt::ShortcutContext m_undockedShortcut;

private Q_SLOTS:
    void parentDockStateChanged(bool docked);

};

} //end namespace ito

#endif
