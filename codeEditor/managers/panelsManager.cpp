#include "panelsManager.h"

#include "codeEditor.h"
#include "panel.h"

#include <assert.h>
#include <vector>

//---------------------------------------------------------------------
void PanelsManager::ZoneItems::add(Panel* p, const QString &name) 
{ 
    panels.append(p); 
    names.append(name); 
}

//---------------------------------------------------------------------
Panel* PanelsManager::ZoneItems::removeFirst(const QString &name)
{
    Panel *p = NULL;
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
Panel* PanelsManager::ZoneItems::get(const QString &name) const
{
    for (int j = 0; j < panels.size(); ++j)
    {
        if (names[j] == name)
        {
            return panels[j];
        }
    }
    return NULL;
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

    connect(this->editor(), SIGNAL(blockCountChanged()), this, SLOT(updateViewportMargins()));
    connect(this->editor(), SIGNAL(updateRequest(QRect,int)), this, SLOT(update(QRect,int)));
    
}

//-----------------------------------------------------------
/*< Gets the list of panels attached to the specified zone.*/
QList<Panel*> PanelsManager::panelsForZone(Panel::Position zone) const
{
    return m_panels[zone].panels;
}

bool cmpPanelByOrderInZoneReverse(const std::pair<Panel*,int> &a, const std::pair<Panel*,int> &b)
{
    return a.second > b.second; //todo: verify
}

//-----------------------------------------------------------
QList<Panel*> PanelsManager::panelsForZoneSortedByOrderReverse(Panel::Position zone) const
{
    QList<Panel*> panels = m_panels[zone].panels;
    std::vector<std::pair<Panel*, int> > sortlist;
    foreach(Panel* p, panels)
    {
        sortlist.push_back(std::pair<Panel*,int>(p, p->orderInZone()));
    }

    std::sort(sortlist.begin(), sortlist.end(), cmpPanelByOrderInZoneReverse);

    QList<Panel*> panelsOut;
    panelsOut.reserve(sortlist.size());
    
    for (int i = 0; i < sortlist.size(); ++i)
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

//-----------------------------------------------
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

//----------------------------------------------------------
/*
Removes all panel from the editor.
*/
void PanelsManager::clear()
{
    Panel *p = NULL;
    for (int i = 0; i < m_panels.size(); ++i)
    {
        ZoneItems &item = m_panels[i];
        while (item.len() > 0)
        {
            p = item.removeFirst(item.names[0]);
            p->setParent(NULL);
            p->deleteLater();
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
    foreach (Panel *p, panelsForZone(Panel::Left))
    {
        if (p->isVisible())
        {
            left += p->sizeHint().width();
        }
    }

    //Right panels
    int right = 0;
    foreach (Panel *p, panelsForZone(Panel::Right))
    {
        if (p->isVisible())
        {
            right += p->sizeHint().width();
        }
    }

    //Top panels
    int top = 0;
    foreach (Panel *p, panelsForZone(Panel::Top))
    {
        if (p->isVisible())
        {
            top += p->sizeHint().height();
        }
    }

    //Bottom panels
    int bottom = 0;
    foreach (Panel *p, panelsForZone(Panel::Bottom))
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
    QList<Panel*> panels = panelsForZoneSortedByOrderReverse(Panel::Left);
    
    foreach(Panel *p, panels)
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
    
    foreach(Panel *p, panels)
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

    foreach(Panel *p, panels)
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
    
    foreach(Panel *p, panels)
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
