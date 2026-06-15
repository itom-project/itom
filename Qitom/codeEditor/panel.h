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

#ifndef PANEL_H
#define PANEL_H

/*
This module contains the Panel API.
*/



#include <qwidget.h>
#include <qevent.h>
#include <qbrush.h>
#include <qpen.h>
#include "mode.h"

namespace ito {

class CodeEditor;

/*
Base class for editor panels.

A panel is a mode and a QWidget.

.. note:: Use enabled to disable panel actions and setVisible to change the
    visibility of the panel.
*/

class Panel : public QWidget, public Mode
{
    Q_OBJECT

public:
    enum Position
    {
        Top = 0,
        Left = 1,
        Right = 2,
        Bottom = 3,
        Floating = 4
    };

    typedef QSharedPointer<Panel> Ptr;

    Panel(const QString &name, bool dynamic, const QString &description = "", QWidget *parent = NULL);
    virtual ~Panel();

    void setVisible(bool visible);

    bool scrollable() const;
    void setScrollable(bool value);

    int orderInZone() const;
    void setOrderInZone(int orderInZone);

    Position position() const;
    void setPosition(Position pos);

    QBrush backgroundBrush() const { return m_backgroundBrush; }
    QPen foregroundPen() const { return m_foregroundPen; }

    virtual void onInstall(CodeEditor *editor);

protected:
    virtual void paintEvent(QPaintEvent *e);


private:
    bool m_dynamic;
    int m_orderInZone;
    bool m_scrollable;
    QBrush m_backgroundBrush;
    QPen m_foregroundPen;

    //!< position in the editor (top, left, right, bottom)
    Position m_position;

    Q_DISABLE_COPY(Panel)
};

} //end namespace ito

#endif
