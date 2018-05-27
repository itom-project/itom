/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include "breakpointPanel.h"

#include "../codeEditor.h"
#include <qpainter.h>
#include <qapplication.h>
#include <qicon.h>
#include <qtooltip.h>
#include "../textBlockUserData.h"
#include "../codeEditor.h"
#include "../delayJobRunner.h"

#include "helper/guiHelper.h"

namespace ito {

//----------------------------------------------------------
/*
*/
BreakpointPanel::BreakpointPanel(const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Panel("BreakpointPanel", false, description, parent)
{
    setScrollable(true);
    setMouseTracking(true);

    m_icons[TextBlockUserData::TypeNoBp] = QIcon();
    m_icons[TextBlockUserData::TypeBp] = QIcon(":/breakpoints/icons/itomBreak.png");
    m_icons[TextBlockUserData::TypeBpDisabled] = QIcon(":/breakpoints/icons/itomBreakDisabled.png");
    m_icons[TextBlockUserData::TypeBpEdit] = QIcon(":/breakpoints/icons/itomcBreak.png");
    m_icons[TextBlockUserData::TypeBpEditDisabled] = QIcon(":/breakpoints/icons/itomCBreakDisabled.png");
}

//----------------------------------------------------------
/*
*/
BreakpointPanel::~BreakpointPanel()
{

}



//------------------------------------------------------------
/*
Returns the panel size hint. (fixed with of 16px)
*/
QSize BreakpointPanel::sizeHint() const
{   
    int dpi = GuiHelper::getScreenLogicalDpi();
    int size = 16 * dpi / 96;

    QFontMetricsF metrics(editor()->font());
    QSize size_hint(metrics.height(), metrics.height());
    if (size_hint.width() > size)
    {
        size_hint.setWidth(size);
    }
    return size_hint;
}

//----------------------------------------------------------
void BreakpointPanel::paintEvent(QPaintEvent *e)
{
    Panel::paintEvent(e);
    QPainter painter(this);
    TextBlockUserData *tbud;
    QIcon icon;
    QRect rect;

    foreach (const VisibleBlock &b, editor()->visibleBlocks())
    {
        tbud = dynamic_cast<TextBlockUserData*>(b.textBlock.userData());

        if (tbud && tbud->m_breakpointType != TextBlockUserData::TypeNoBp)
        {
            icon = m_icons[tbud->m_breakpointType];
            if (!icon.isNull())
            {
                rect = QRect();
                rect.setX(0);
                rect.setY(b.topPosition);
                rect.setWidth(sizeHint().width());
                rect.setHeight(sizeHint().height());
                icon.paint(&painter, rect);
            }
        }
    }
}

//----------------------------------------------------------
/*
*/
void BreakpointPanel::mouseReleaseEvent(QMouseEvent *e)
{
    /*# Handle mouse press:
    # - emit add marker signal if there were no marker under the mouse
    #   cursor
    # - emit remove marker signal if there were one or more markers under
    #   the mouse cursor.*/
    int line = editor()->lineNbrFromPosition(e->pos().y());

    if (line != -1)
    {
        if (e->button() == Qt::LeftButton)
        {
            emit toggleBreakpointRequested(line);
        }
    }
}




} //end namespace ito