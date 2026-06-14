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

#include "panel.h"

#include <qapplication.h>
#include <qpainter.h>
#include "codeEditor.h"
#include "managers/panelsManager.h"

namespace ito {

//-------------------------------------------------------
Panel::Panel(const QString &name, bool dynamic, const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Mode(name, description),
    QWidget(parent),
    m_dynamic(dynamic),
    m_orderInZone(-1),
    m_scrollable(false),
    m_position(Left)
{
    setContextMenuPolicy(Qt::PreventContextMenu);
}

//-------------------------------------------------------
Panel::~Panel()
{
}

//-------------------------------------------------------
/*
A scrollable panel will follow the editor's scroll-bars. Left and right
panels follow the vertical scrollbar. Top and bottom panels follow the
horizontal scrollbar.

:type: bool
*/
bool Panel::scrollable() const
{
    return m_scrollable;
}

//-------------------------------------------------------
void Panel::setScrollable(bool value)
{
    m_scrollable = value;
}

//-------------------------------------------------------
int Panel::orderInZone() const
{
    return m_orderInZone;
}

//-------------------------------------------------------
void Panel::setOrderInZone(int orderInZone)
{
    m_orderInZone = orderInZone;
}

//-------------------------------------------------------
Panel::Position Panel::position() const
{
    return m_position;
}

//-------------------------------------------------------
void Panel::setPosition(Position pos)
{
    m_position = pos;
}

//-------------------------------------------------------
/*
Fills the panel background using QPalette
*/
void Panel::paintEvent(QPaintEvent *e)
{
    if (isVisible())
    {
        //fill background
        m_backgroundBrush = QBrush(QColor(palette().window().color()));
        m_foregroundPen = QPen(QColor(palette().windowText().color()));
        QPainter painter(this);
        painter.fillRect(e->rect(), m_backgroundBrush);
    }
}

//-------------------------------------------------------
/*
Extends :meth:`pyqode.core.api.Mode.on_install` method to set the
editor instance as the parent widget.

.. warning:: Don't forget to call **super** if you override this
    method!

:param editor: editor instance
:type editor: pyqode.core.api.CodeEdit
*/
void Panel::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
    setParent(editor);
    setPalette(qApp->palette());

}

//-------------------------------------------------------
/*
Shows/Hides the panel

Automatically call CodeEdit.refresh_panels.

:param visible: Visible state
*/
void Panel::setVisible(bool visible)
{
    QWidget::setVisible(visible);
    if (editor())
    {
        editor()->panels()->refresh();
    }
}

} //end namespace ito
