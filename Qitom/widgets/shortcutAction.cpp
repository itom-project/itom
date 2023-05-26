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

#include "shortcutAction.h"

#include "abstractDockWidget.h"

namespace ito
{

//------------------------------------------------------------------------------------
ShortcutAction::ShortcutAction(const QString &text, AbstractDockWidget *parent) :
    QObject(parent),
    m_dockedShortcut(Qt::WidgetShortcut),
    m_undockedShortcut(Qt::WidgetShortcut)
{
    m_action = new QAction(text, parent);
    QString toolTipText = text;
    toolTipText.replace("&", "");
    m_action->setToolTip(toolTipText);
}

//------------------------------------------------------------------------------------
ShortcutAction::ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent) :
    QObject(parent),
    m_dockedShortcut(Qt::WidgetShortcut),
    m_undockedShortcut(Qt::WidgetShortcut)
{
    m_action = new QAction(icon, text, parent);
    QString toolTipText = text;
    toolTipText.replace("&", "");
    m_action->setToolTip(toolTipText);
}

//------------------------------------------------------------------------------------
ShortcutAction::ShortcutAction(const QIcon &icon, const QString &text,
        AbstractDockWidget *parent, const QKeySequence &key,
        Qt::ShortcutContext context /*= Qt::WindowShortcut*/) :
    QObject(parent),
    m_dockedShortcut(context),
    m_undockedShortcut(context)
{
    QString text2 = text;
    QString text3 = text;
    text3.replace("&", "");

    //some key sequences do not exist as default on all operating systems
    if (key.isEmpty() == false)
    {
        text2 += "\t" + key.toString(QKeySequence::NativeText);
        text3 += " (" + key.toString(QKeySequence::NativeText) + ")";

        m_shortcut = new QShortcut(key, parent->getCanvas());
        m_shortcut->setContext(context);
    }

    m_action = new QAction(icon, text2, parent);
    m_action->setToolTip(text3);
}

//----------------------------------------------------------------
//!< Action with text, icon and shortcut (different contexts for docked and undocked state)
ShortcutAction::ShortcutAction(const QIcon &icon, const QString &text, AbstractDockWidget *parent,
    const QKeySequence &key, Qt::ShortcutContext dockedContext,
    Qt::ShortcutContext undockedContext) :
    QObject(parent),
    m_dockedShortcut(dockedContext),
    m_undockedShortcut(undockedContext)
{
    QString text2 = text;
    QString text3 = text;
    text3.replace("&", "");

    //some key sequences do not exist as default on all operating systems
    if (key.isEmpty() == false)
    {
        text2 += "\t" + key.toString(QKeySequence::NativeText);
        text3 += " (" + key.toString(QKeySequence::NativeText) + ")";

        m_shortcut = new QShortcut(key, parent->getCanvas());

        //this is the only case, where the dock state has to be tracked
        connect(parent, &AbstractDockWidget::dockStateChanged, this, &ShortcutAction::parentDockStateChanged);

        if (parent->docked())
        {
            m_shortcut->setContext(dockedContext);
        }
        else
        {
            m_shortcut->setContext(undockedContext);
        }
    }

    m_action = new QAction(icon, text2, parent);
    m_action->setToolTip(text3);
}

//----------------------------------------------------------------
ShortcutAction::~ShortcutAction()
{
    //do not delete action and shortcut here, since it will be deleted by common parent.
}

//----------------------------------------------------------------
void ShortcutAction::parentDockStateChanged(bool docked)
{
    if (m_shortcut)
    {
        if (docked)
        {
            m_shortcut->setContext(m_dockedShortcut);
        }
        else
        {
            m_shortcut->setContext(m_undockedShortcut);
        }
    }
}

//----------------------------------------------------------------
void ShortcutAction::connectTrigger(
    const QObject *receiver,
    const char *method,
    Qt::ConnectionType type /*= Qt::AutoConnection*/)
{
    if (m_action)
    {
        QObject::connect(m_action, SIGNAL(triggered()), receiver, method, type);
    }

    if (m_shortcut)
    {
        QObject::connect(m_shortcut, SIGNAL(activated()), receiver, method, type);
    }
}

//----------------------------------------------------------------
void ShortcutAction::setEnabled(bool actionEnabled, bool shortcutEnabled)
{
    if (m_action)
    {
        m_action->setEnabled(actionEnabled);
        if (m_shortcut) m_shortcut->setEnabled(shortcutEnabled);
    }
}

//----------------------------------------------------------------
void ShortcutAction::setEnabled(bool enabled)
{
    setEnabled(enabled, enabled);
}

//----------------------------------------------------------------
void ShortcutAction::setVisible(bool actionVisible, bool shortcutEnabled)
{
    if (m_action)
    {
        m_action->setVisible(actionVisible);
        if (m_shortcut) m_shortcut->setEnabled(shortcutEnabled);
    }
}

//----------------------------------------------------------------
void ShortcutAction::setVisible(bool visible)
{
    setVisible(visible, visible);
}

} //end namespace ito
