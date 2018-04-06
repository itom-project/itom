#ifndef PANELS_H
#define PANELS_H

/*
This module contains the panels controller, responsible of drawing panel
inside CodeEdit's margins
*/

#include "manager.h"

#include <qpoint.h>
#include <qlist.h>
#include <qmap.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qrect.h>

#include "panel.h"

/*
Manages the list of panels and draws them inside the margin of the code
edit widget.
*/
class PanelsManager : public Manager
{
    Q_OBJECT

public:
    

    PanelsManager(CodeEditor *editor, QObject *parent = NULL);
    virtual ~PanelsManager();

    QList<Panel*> panelsForZone(Panel::Position zone) const;
    int marginSize(Panel::Position zone = Panel::Left);

    Panel* append(Panel *panel, Panel::Position pos = Panel::Left);
    Panel* remove(const QString &nameOrClass);

private:

    struct ZoneItems
    {
        ZoneItems(int margin) : marginSize(margin) {}
        int marginSize;
        QStringList names;
        QList<Panel*> panels;

        int len() const { return panels.size(); }

        void add(Panel* p, const QString &name) { panels.append(p); names.append(name); }

        Panel* removeFirst(const QString &name)
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

        Panel* get(const QString &name) const
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
    };

    QPoint m_cachedCursorPos;
    int m_top;
    int m_bottom;
    int m_left;
    int m_right;
    QList<ZoneItems> m_panels;

    Panel* get(const QString &nameOrClass);

private slots:
    void updateViewportMargins();
    void update(const QRect & rect, int dy, bool forceUpdateMargins = false);
};

    
/*


    def clear(self):
        """
        Removes all panel from the editor.
        """
        for i in range(4):
            while len(self._panels[i]):
                key = sorted(list(self._panels[i].keys()))[0]
                panel = self.remove(key)
                panel.setParent(None)
                panel.deleteLater()





    def __iter__(self):
        lst = []
        for zone, zone_dict in self._panels.items():
            for name, panel in zone_dict.items():
                lst.append(panel)
        return iter(lst)

    def __len__(self):
        lst = []
        for zone, zone_dict in self._panels.items():
            for name, panel in zone_dict.items():
                lst.append(panel)
        return len(lst)

    def panels_for_zone(self, zone):
        """
        Gets the list of panels attached to the specified zone.
        :param zone: Panel position.
        :return: List of panels instances.
        """
        return list(self._panels[zone].values())

    def refresh(self):
        """ Refreshes the editor panels (resize and update margins) """
        _logger().log(5, 'refresh_panels')
        self.resize()
        self._update(self.editor.contentsRect(), 0,
                     force_update_margins=True)

    def resize(self):
        """ Resizes panels """
        crect = self.editor.contentsRect()
        view_crect = self.editor.viewport().contentsRect()
        s_bottom, s_left, s_right, s_top = self._compute_zones_sizes()
        tw = s_left + s_right
        th = s_bottom + s_top
        w_offset = crect.width() - (view_crect.width() + tw)
        h_offset = crect.height() - (view_crect.height() + th)
        left = 0
        panels = self.panels_for_zone(Panel.Position.LEFT)
        panels.sort(key=lambda panel: panel.order_in_zone, reverse=True)
        for panel in panels:
            if not panel.isVisible():
                continue
            panel.adjustSize()
            size_hint = panel.sizeHint()
            panel.setGeometry(crect.left() + left,
                              crect.top() + s_top,
                              size_hint.width(),
                              crect.height() - s_bottom - s_top - h_offset)
            left += size_hint.width()
        right = 0
        panels = self.panels_for_zone(Panel.Position.RIGHT)
        panels.sort(key=lambda panel: panel.order_in_zone, reverse=True)
        for panel in panels:
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            panel.setGeometry(
                crect.right() - right - size_hint.width() - w_offset,
                crect.top() + s_top,
                size_hint.width(),
                crect.height() - s_bottom - s_top - h_offset)
            right += size_hint.width()
        top = 0
        panels = self.panels_for_zone(Panel.Position.TOP)
        panels.sort(key=lambda panel: panel.order_in_zone)
        for panel in panels:
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            panel.setGeometry(crect.left(),
                              crect.top() + top,
                              crect.width() - w_offset,
                              size_hint.height())
            top += size_hint.height()
        bottom = 0
        panels = self.panels_for_zone(Panel.Position.BOTTOM)
        panels.sort(key=lambda panel: panel.order_in_zone)
        for panel in panels:
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            panel.setGeometry(
                crect.left(),
                crect.bottom() - bottom - size_hint.height() - h_offset,
                crect.width() - w_offset,
                size_hint.height())
            bottom += size_hint.height()

    def _update(self, rect, delta_y, force_update_margins=False):
        """ Updates panels """
        helper = TextHelper(self.editor)
        if not self:
            return
        for zones_id, zone in self._panels.items():
            if zones_id == Panel.Position.TOP or \
               zones_id == Panel.Position.BOTTOM:
                continue
            panels = list(zone.values())
            for panel in panels:
                if panel.scrollable and delta_y:
                    panel.scroll(0, delta_y)
                line, col = helper.cursor_position()
                oline, ocol = self._cached_cursor_pos
                if line != oline or col != ocol or panel.scrollable:
                    panel.update(0, rect.y(), panel.width(), rect.height())
                self._cached_cursor_pos = helper.cursor_position()
        if (rect.contains(self.editor.viewport().rect()) or
                force_update_margins):
            self._update_viewport_margins()



    def _compute_zones_sizes(self):
        """ Compute panel zone sizes """
        # Left panels
        left = 0
        for panel in self.panels_for_zone(Panel.Position.LEFT):
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            left += size_hint.width()
        # Right panels
        right = 0
        for panel in self.panels_for_zone(Panel.Position.RIGHT):
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            right += size_hint.width()
        # Top panels
        top = 0
        for panel in self.panels_for_zone(Panel.Position.TOP):
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            top += size_hint.height()
        # Bottom panels
        bottom = 0
        for panel in self.panels_for_zone(Panel.Position.BOTTOM):
            if not panel.isVisible():
                continue
            size_hint = panel.sizeHint()
            bottom += size_hint.height()
        self._top, self._left, self._right, self._bottom = (
            top, left, right, bottom)
return bottom, left, right, top
*/
#endif