#ifndef PANELSMANAGER_H
#define PANELSMANAGER_H

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
#include <qvector.h>

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

    QList<Panel::Ptr> panelsForZone(Panel::Position zone) const;
    QList<Panel::Ptr> panelsForZoneSortedByOrderReverse(Panel::Position zone) const;
    int marginSize(Panel::Position zone = Panel::Left);

    Panel::Ptr append(Panel::Ptr panel, Panel::Position pos = Panel::Left);
    Panel::Ptr remove(const QString &nameOrClass);
    void clear();

    void resize();
    void refresh();

    Panel::Ptr get(const QString &nameOrClass);

private:
    struct ZoneItems
    {
        ZoneItems(int margin) : marginSize(margin) {}
        int marginSize;
        QStringList names;
        QList<Panel::Ptr> panels;

        int len() const { return panels.size(); }
        void add(Panel::Ptr p, const QString &name);
        Panel::Ptr removeFirst(const QString &name);
        Panel::Ptr get(const QString &name) const;
    };

    QPoint m_cachedCursorPos;
    int m_top;
    int m_bottom;
    int m_left;
    int m_right;
    QList<ZoneItems> m_panels;

    
    QVector<int> computeZonesSizes();
    

private slots:
    void updateViewportMargins();
    void update(const QRect & rect, int dy, bool forceUpdateMargins = false);
};

    
/*


    





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
*/
    
#endif //PANELSMANAGER_H