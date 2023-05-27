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

#include "panelsManager.h"

#include "../codeEditor.h"
#include "../panel.h"

#include <assert.h>
#include <vector>

namespace ito {

//---------------------------------------------------------------------
void PanelsManager::ZoneItems::add(Panel::Ptr p, const QString &name)
{
    panels.append(p);
    names.append(name);
}

//---------------------------------------------------------------------
Panel::Ptr PanelsManager::ZoneItems::removeFirst(const QString &name)
{
    Panel::Ptr p;
    for (int j = 0; j < panels.size(); ++j)
    {
        if (names[j] == name)
        {
            names.removeAt(j);
            p = panels[j];
            panels.removeAt(j);
            break;
        }
    }
    return p;
}

//---------------------------------------------------------------------
Panel::Ptr PanelsManager::ZoneItems::get(const QString &name) const
{
    for (int j = 0; j < panels.size(); ++j)
    {
        if (names[j] == name)
        {
            return panels[j];
        }
    }
    return Panel::Ptr();
}

//---------------------------------------------------------------------
//---------------------------------------------------------------------
PanelsManager::PanelsManager(CodeEditor *editor, QObject *parent /*= NULL*/) :
    Manager(editor, parent),
    m_cachedCursorPos(-1,-1),
    m_top(-1),
    m_left(-1),
    m_right(-1),
    m_bottom(-1)

{
    //foreach entry in enum Position:
    m_panels << ZoneItems(0) << ZoneItems(0) << ZoneItems(0) << ZoneItems(0);

    connect(this->editor(), SIGNAL(blockCountChanged(int)), this, SLOT(updateViewportMargins()));
    connect(this->editor(), SIGNAL(updateRequest(QRect,int)), this, SLOT(update(QRect,int)));

}


PanelsManager::~PanelsManager()
{
}

//-----------------------------------------------------------
/*< Gets the list of panels attached to the specified zone.*/
QList<Panel::Ptr> PanelsManager::panelsForZone(Panel::Position zone) const
{
    return m_panels[zone].panels;
}

bool cmpPanelByOrderInZoneReverse(const std::pair<Panel::Ptr,int> &a, const std::pair<Panel::Ptr,int> &b)
{
    return a.second > b.second; //todo: verify
}

//-----------------------------------------------------------
QList<Panel::Ptr> PanelsManager::panelsForZoneSortedByOrderReverse(Panel::Position zone) const
{
    QList<Panel::Ptr> panels = m_panels[zone].panels;
    std::vector<std::pair<Panel::Ptr, int> > sortlist;
    foreach(Panel::Ptr p, panels)
    {
        sortlist.push_back(std::pair<Panel::Ptr,int>(p, p->orderInZone()));
    }

    std::sort(sortlist.begin(), sortlist.end(), cmpPanelByOrderInZoneReverse);

    QList<Panel::Ptr> panelsOut;
    panelsOut.reserve((int)sortlist.size());

    for (size_t i = 0; i < sortlist.size(); ++i)
    {
        panelsOut << sortlist[i].first;
    }

    return panelsOut;
}

//-----------------------------------------------------------
void PanelsManager::updateViewportMargins()
{
        /*Update viewport margins*/
        int top = 0;
        int left = 0;
        int right = 0;
        int bottom = 0;

        int width, height;

        foreach (const Panel::Ptr &p, panelsForZone(Panel::Left))
        {
            if (p->isVisible())
            {
                width = p->sizeHint().width();
                left += width;
            }
        }

        foreach (const Panel::Ptr &p, panelsForZone(Panel::Right))
        {
            if (p->isVisible())
            {
                width = p->sizeHint().width();
                right += width;
            }
        }

        foreach (const Panel::Ptr &p, panelsForZone(Panel::Top))
        {
            if (p->isVisible())
            {
                height = p->sizeHint().height();
                top += height;
            }
        }

        foreach (const Panel::Ptr &p, panelsForZone(Panel::Bottom))
        {
            if (p->isVisible())
            {
                height = p->sizeHint().height();
                top += height;
            }
        }

        m_panels[Panel::Top].marginSize = top;
        m_panels[Panel::Left].marginSize = left;
        m_panels[Panel::Right].marginSize = right;
        m_panels[Panel::Bottom].marginSize = bottom;

        editor()->setViewportMargins(left, top, right, bottom);
}

//---------------------------------------------
/*> Gets the size of a specific margin. */
int PanelsManager::marginSize(Panel::Position zone /*= Panel::Left*/)
{
    return m_panels[zone].marginSize;
}

//---------------------------------------------
/*
Updates panels
*/
void PanelsManager::update(const QRect & rect, int dy, bool forceUpdateMargins /*= false*/)
{
    /*Updates panels*/
    int line, column;
    int oline, ocolumn;
    const CodeEditor* e = editor();

    for (int zones_id = Panel::Left; zones_id < m_panels.size(); ++zones_id)
    {
        if (zones_id == Panel::Top || zones_id == Panel::Bottom)
        {
            continue;
        }

        foreach (const Panel::Ptr &p, m_panels[zones_id].panels)
        {
            if (p->scrollable() && dy)
            {
                p->scroll(0, dy);
            }
            e->cursorPosition(line, column);
            oline = m_cachedCursorPos.rx();
            ocolumn = m_cachedCursorPos.ry();
            if (line != oline || column != ocolumn || p->scrollable())
            {
                p->update(0, rect.y(), p->width(), rect.height());
            }
            e->cursorPosition(line, column);
            m_cachedCursorPos = QPoint(line, column);
        }
    }

    if (rect.contains(e->viewport()->rect()) || \
            forceUpdateMargins)
    {
        updateViewportMargins();
    }
}


//----------------------------------------------------------
/*
Installs a panel on the editor.

:param panel: Panel to install
:param position: Position where the panel must be installed.
:return: The installed panel
*/
Panel::Ptr PanelsManager::append(Panel::Ptr panel, Panel::Position pos /*= Panel::Left*/)
{
    assert(panel != NULL);

    panel->setOrderInZone(m_panels[pos].len());
    m_panels[pos].add(panel, panel->name());
    panel->setPosition(pos);
    panel->onInstall(editor());
    return panel;
}

//----------------------------------------------------------
/*
"""
Removes the specified panel.

:param name_or_klass: Name or class of the panel to remove.
:return: The removed panel
"""
*/
Panel::Ptr PanelsManager::remove(const QString &nameOrClass)
{
    Panel::Ptr panel = get(nameOrClass);
    panel->onUninstall();
    panel->hide();
    panel->setParent(NULL);
    return m_panels[panel->position()].removeFirst(panel->name());
}

//-----------------------------------------------
/*
Gets a specific panel instance.
:param name_or_klass: Name or class of the panel to retrieve.

:return: The specified panel instance.
*/
Panel::Ptr PanelsManager::get(const QString &nameOrClass)
{
    Panel::Ptr p;

    foreach (const ZoneItems &item, m_panels)
    {
        p = item.get(nameOrClass);
        if (p)
        {
            return p;
        }
    }

    return p;
}

//----------------------------------------------------------
/*
Removes all panel from the editor.
*/
void PanelsManager::clear()
{
    Panel::Ptr p;
    for (int i = 0; i < m_panels.size(); ++i)
    {
        ZoneItems &item = m_panels[i];
        while (item.len() > 0)
        {
            p = item.removeFirst(item.names[0]);
            if (p)
            {
                p->setParent(NULL);
                p->deleteLater();
            }
        }
    }

    m_panels.clear();
}

//---------------------------------------------
/*
Refreshes the editor panels (resize and update margins)
*/
void PanelsManager::refresh()
{
    resize();
    update(editor()->contentsRect(), 0, true);
}




//---------------------------------------------
/*
Compute panel zone sizes
*/
QVector<int> PanelsManager::computeZonesSizes()
{
    //Left panels
    int left = 0;
    foreach (const Panel::Ptr &p, panelsForZone(Panel::Left))
    {
        if (p->isVisible())
        {
            left += p->sizeHint().width();
        }
    }

    //Right panels
    int right = 0;
    foreach (const Panel::Ptr &p, panelsForZone(Panel::Right))
    {
        if (p->isVisible())
        {
            right += p->sizeHint().width();
        }
    }

    //Top panels
    int top = 0;
    foreach (const Panel::Ptr &p, panelsForZone(Panel::Top))
    {
        if (p->isVisible())
        {
            top += p->sizeHint().height();
        }
    }

    //Bottom panels
    int bottom = 0;
    foreach (const Panel::Ptr &p, panelsForZone(Panel::Bottom))
    {
        if (p->isVisible())
        {
            bottom += p->sizeHint().height();
        }
    }

    m_top = top;
    m_left = left;
    m_right = right;
    m_bottom = bottom;

    return QVector<int>() << bottom << left << right << top;
}



//----------------------------------------------------
/*
Resizes panels
*/
void PanelsManager::resize()
{
    QRect crect = editor()->contentsRect();
    QRect view_crect = editor()->viewport()->contentsRect();
    QVector<int> zonesSizes = computeZonesSizes();
    int s_bottom = zonesSizes[0];
    int s_left = zonesSizes[1];
    int s_right = zonesSizes[2];
    int s_top = zonesSizes[3];
    int tw = s_left + s_right;
    int th = s_bottom + s_top;
    int w_offset = crect.width() - (view_crect.width() + tw);
    int h_offset = crect.height() - (view_crect.height() + th);
    QSize size_hint;

    //Left
    int left = 0;
    QList<Panel::Ptr> panels = panelsForZoneSortedByOrderReverse(Panel::Left);

    foreach(Panel::Ptr p, panels)
    {
        if (p->isVisible())
        {
            p->adjustSize();
            size_hint = p->sizeHint();
            p->setGeometry(crect.left() + left, \
                           crect.top() + s_top, \
                           size_hint.width(), \
                           crect.height() - s_bottom - s_top - h_offset);
            left += size_hint.width();
        }
    }

    //Right
    int right = 0;
    panels = panelsForZoneSortedByOrderReverse(Panel::Right);

    foreach(Panel::Ptr p, panels)
    {
        if (p->isVisible())
        {
            size_hint = p->sizeHint();
            p->setGeometry(crect.right() - right - size_hint.width() - w_offset, \
                crect.top() + s_top, \
                size_hint.width(), \
                crect.height() - s_bottom - s_top - h_offset);
            right += size_hint.width();
        }
    }

    //Top
    int top = 0;
    panels = panelsForZoneSortedByOrderReverse(Panel::Top);

    foreach(Panel::Ptr p, panels)
    {
        if (p->isVisible())
        {
            size_hint = p->sizeHint();
            p->setGeometry(crect.left(), \
                              crect.top() + top, \
                              crect.width() - w_offset, \
                              size_hint.height());
            top += size_hint.height();
        }
    }

    //Bottom
    int bottom = 0;
    panels = panelsForZoneSortedByOrderReverse(Panel::Bottom);

    foreach(Panel::Ptr p, panels)
    {
        if (p->isVisible())
        {
            size_hint = p->sizeHint();
            p->setGeometry(crect.left(), \
                crect.bottom() - bottom - size_hint.height() - h_offset, \
                crect.width() - w_offset, \
                size_hint.height());
            bottom += size_hint.height();
        }
    }
}

} //end namespace ito
