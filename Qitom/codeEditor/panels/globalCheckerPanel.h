/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

#pragma once

/*
Global Checker Panels

Right side, overview about script outline, and info, warning
and error indicators.
*/

#include "../panel.h"
#include "../utils/utils.h"
#include "../textBlockUserData.h"

#include <qevent.h>
#include <qsize.h>
#include <qcolor.h>
#include <qicon.h>
#include <qmap.h>

class QMenu;
class QAction;

namespace ito {

/*
Displays all checker messages found in the document.
The user can click on a marker to quickly go the the error line.
*/
class GlobalCheckerPanel : public Panel
{
    Q_OBJECT
public:
    GlobalCheckerPanel(const QString &description = "", QWidget *parent = nullptr);
    virtual ~GlobalCheckerPanel();

    virtual QSize sizeHint() const;


protected:
    virtual void paintEvent(QPaintEvent *e);
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void wheelEvent(QWheelEvent* e);

    float getMarkerSpacing() const;
    QSize getMarkerSize() const;
    void drawVisibleArea(QPainter &painter);
    void drawMessages(QPainter& painter);

private:
    int verticalOffset() const;
    QRect getScrollbarGrooveRect() const;
    int getScrollbarValueHeight() const;
    constexpr int markerHeight() const { return 3; }

    QMap<TextBlockUserData::BreakpointType, QIcon> m_icons;
    QBrush m_backgroundBrush;
    QIcon m_breakpointIcon;
    QIcon m_bookmarkIcon;
};

} //end namespace ito




