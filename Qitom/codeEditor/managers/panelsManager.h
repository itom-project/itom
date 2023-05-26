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

#include "../panel.h"

namespace ito {

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


} //end namespace ito

#endif //PANELSMANAGER_H
