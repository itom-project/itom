#include "panelsManager.h"

#include "codeEditor.h"
#include "panel.h"

#include <assert.h>

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

    connect(this->editor(), SIGNAL(blockCountChanged()), this, SLOT(updateViewportMargins()));
    connect(this->editor(), SIGNAL(updateRequest(QRect,int)), this, SLOT(update(QRect,int)));
    
}

 /*< Gets the list of panels attached to the specified zone.*/
QList<Panel*> PanelsManager::panelsForZone(Panel::Position zone) const
{
    return m_panels[zone].panels;
}

void PanelsManager::updateViewportMargins()
{
        /*Update viewport margins*/
        int top = 0;
        int left = 0;
        int right = 0;
        int bottom = 0;

        int width, height;

        foreach (const Panel* p, panelsForZone(Panel::Left))
        {
            if (p->isVisible())
            {
                width = p->sizeHint().width();
                left += width;
            }
        }
        
        foreach (const Panel* p, panelsForZone(Panel::Right))
        {
            if (p->isVisible())
            {
                width = p->sizeHint().width();
                right += width;
            }
        }
        
        foreach (const Panel* p, panelsForZone(Panel::Top))
        {
            if (p->isVisible())
            {
                height = p->sizeHint().height();
                top += height;
            }
        }
        
        foreach (const Panel* p, panelsForZone(Panel::Bottom))
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

/*> Gets the size of a specific margin. */
int PanelsManager::marginSize(Panel::Position zone /*= Panel::Left*/)
{
    return m_panels[zone].marginSize;
}

void PanelsManager::update(const QRect & rect, int dy, bool forceUpdateMargins /*= false*/)
{
    /*Updates panels*/
    //helper = TextHelper(self.editor)
    int line, column;
    int oline, ocolumn;
    const CodeEditor* e = editor();

    for (int zones_id = Panel::Left; zones_id < m_panels.size(); ++zones_id)
    {
        if (zones_id == Panel::Top || zones_id == Panel::Bottom)
        {
            continue;
        }

        foreach (Panel* p, m_panels[zones_id].panels)
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
Panel* PanelsManager::append(Panel *panel, Panel::Position pos /*= Panel::Left*/)
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
Panel* PanelsManager::remove(const QString &nameOrClass)
{
    Panel* panel = get(nameOrClass);
    panel->onUninstall();
    panel->hide();
    panel->setParent(NULL);
    return m_panels[panel->position()].removeFirst(panel->name());
}

/*
Gets a specific panel instance.
:param name_or_klass: Name or class of the panel to retrieve.

:return: The specified panel instance.
*/
Panel* PanelsManager::get(const QString &nameOrClass)
{
    Panel *p = NULL;

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