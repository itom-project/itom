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

#include "globalCheckerPanel.h"

#include "../codeEditor.h"
#include <qpainter.h>
#include <qapplication.h>
#include <qicon.h>
#include <qtooltip.h>
#include <qmenu.h>
#include <qaction.h>
#include <qscrollbar.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include "../textBlockUserData.h"
#include "../codeEditor.h"
#include "../delayJobRunner.h"

#include "helper/guiHelper.h"

#include <qdebug.h>
namespace ito {

//----------------------------------------------------------
/*
*/
GlobalCheckerPanel::GlobalCheckerPanel(const QString &description /*= ""*/, QWidget *parent /*= nullptr */) :
    Panel("GlobalCheckerPanel", false, description, parent)
{
    setScrollable(true);
    m_breakpointIcon = QIcon(":/breakpoints/icons/itomBreak.png");
    m_bookmarkIcon = QIcon(":/bookmark/icons/bookmark.png");
}

//----------------------------------------------------------
/*
*/
GlobalCheckerPanel::~GlobalCheckerPanel()
{

}

//------------------------------------------------------------
/*
Returns the panel size hint. (fixed with of 16px)
*/
QSize GlobalCheckerPanel::sizeHint() const
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
/*Paints the messages and the visible area on the panel.

*/
void GlobalCheckerPanel::paintEvent(QPaintEvent *e)
{
    Panel::paintEvent(e);

    if (isVisible())
    {
        m_backgroundBrush = QBrush(editor()->background());
        QPainter painter(this);
        painter.fillRect(e->rect(), m_backgroundBrush);
        drawVisibleArea(painter);
        drawMessages(painter);
    }
}

//----------------------------------------------------------
/*
This property holds the vertical offset of the scroll flag area
relative to the top of the text editor.
*/
int GlobalCheckerPanel::verticalOffset() const
{
    const auto vsb = editor()->verticalScrollBar();

    if (!vsb ||!vsb->isVisible())
    {
        return 0;
    }

    const auto style = qApp->style();
    QStyleOptionSlider opt;
    opt.initFrom(vsb);

    // Get the area in which the slider handle may move.
    QRect grooveRect = style->subControlRect(QStyle::CC_ScrollBar, &opt, QStyle::SC_ScrollBarGroove, this);
    
    if (opt.orientation == Qt::Horizontal)
    {
        return grooveRect.x();
    }

    return grooveRect.y();
}

//----------------------------------------------------------
QRect GlobalCheckerPanel::getScrollbarGrooveRect() const
{
    const auto vsb = editor()->verticalScrollBar();

    if (!vsb || !vsb->isVisible())
    {
        return QRect();
    }

    const auto style = qApp->style();
    QStyleOptionSlider opt;
    opt.initFrom(vsb);

    // Get the area in which the slider handle may move.
    QRect grooveRect = style->subControlRect(QStyle::CC_ScrollBar, &opt, QStyle::SC_ScrollBarGroove, this);

    grooveRect.setY(qMax(grooveRect.y(), grooveRect.x()));
    grooveRect.setX(0);
    grooveRect.setWidth(qAbs(grooveRect.width()));

    return grooveRect;
}

//----------------------------------------------------------
/* Return the value span height of the scrollbar
*/
int GlobalCheckerPanel::getScrollbarValueHeight() const
{
    const auto vsb = editor()->verticalScrollBar();

    // pageStep are the number of additional "non existing" lines at the end
    // of the script.
    int h = vsb->maximum() - vsb->minimum() + vsb->pageStep();
    return h;
}

//----------------------------------------------------------
/*
Draw messages from all subclass of CheckerMode currently
installed on the editor.
*/
void GlobalCheckerPanel::drawMessages(QPainter& painter)
{
    TextBlockUserData* tbud;
    QIcon icon;
    CodeCheckerItem::CheckerType worstStatus = CodeCheckerItem::Info;
    bool hasCheckerMessage;
    QRect rect;
    QBrush brushInfo(QColor(60, 111, 179));
    QBrush brushWarning(QColor(241, 133, 46));
    QBrush brushError(QColor(226, 0, 0));

    const QTextDocument* td = editor()->document();
    QTextBlock b = td->firstBlock();
    QSize markerSize = getMarkerSize();    
    auto grooveRect = getScrollbarGrooveRect();
    int voffset = grooveRect.y();

    // b.blockNumber() is zero-based
    int blockIndex = 0;
    float markerSpacing = getMarkerSpacing();

    while (b.isValid())
    {
        worstStatus = CodeCheckerItem::Info;
        hasCheckerMessage = false;
        tbud = dynamic_cast<TextBlockUserData*>(b.userData());

        if (b.isVisible())
        {
            if (tbud)
            {
                foreach(const CodeCheckerItem & cci, tbud->m_checkerMessages)
                {
                    hasCheckerMessage = true;

                    if (cci.type() > worstStatus)
                    {
                        worstStatus = cci.type();
                    }
                }

                if (hasCheckerMessage)
                {
                    rect = QRect();
                    rect.setX(sizeHint().width() / 6);
                    rect.setY(qRound(voffset + blockIndex * markerSpacing - markerSize.height() / 2.0));
                    rect.setSize(markerSize);

                    switch (worstStatus)
                    {
                    case ito::CodeCheckerItem::Info:
                        painter.fillRect(rect, brushInfo);
                        break;
                    case ito::CodeCheckerItem::Warning:
                        painter.fillRect(rect, brushWarning);
                        break;
                    case ito::CodeCheckerItem::Error:
                        painter.fillRect(rect, brushError);
                        break;
                    }
                }

                if (tbud->m_bookmark)
                {
                    int s = std::max(2 + sizeHint().width() / 2, 8);
                    rect = QRect();
                    rect.setX((sizeHint().width() - s) / 2);
                    rect.setY(qRound(voffset + blockIndex * markerSpacing - s / 2.0));
                    rect.setWidth(s);
                    rect.setHeight(s);
                    m_bookmarkIcon.paint(&painter, rect);
                }

                if (tbud->m_breakpointType != ito::TextBlockUserData::TypeNoBp)
                {
                    int s = std::max(sizeHint().width() / 2, 8);
                    rect = QRect();
                    rect.setX((sizeHint().width() - s) / 2);
                    rect.setY(qRound(voffset + blockIndex * markerSpacing - s / 2.0));
                    rect.setWidth(s);
                    rect.setHeight(s);
                    m_breakpointIcon.paint(&painter, rect);
                }
            }

            blockIndex++;
        }

        b = b.next();
    }
}


//----------------------------------------------------------
/*
Draw the visible area.
This method does not take folded blocks into account.
*/
void GlobalCheckerPanel::drawVisibleArea(QPainter &painter)
{
    if (editor()->visibleBlocks().size() > 0)
    {
        QRect rect;
        int voffset = verticalOffset();
        

        const int vh = editor()->viewport()->height();
        float contentHeight = static_cast<float>(vh - 2 * verticalOffset());

        // the first block of the document has the right height. Only the last block has a different height.
        // Therefore always refer to the first one, and not the first visible one.
        QRectF firstBlock = editor()->blockBoundingGeometry(editor()->document()->firstBlock());
        QScrollBar* sb = editor()->verticalScrollBar();

        // number of rows in the editor() widget
        float numRowsInEditor = (float)editor()->height() / firstBlock.height();

        rect.setX(0);
        rect.setWidth(sizeHint().width());

        // sb->minimum() is always zero, sb->maximum() is the index of the last line (zero-based)
        // and sb->value() is the index of the first visible line in the view point
        rect.setY(sb->value() * getMarkerSpacing() + voffset);
        rect.setHeight(numRowsInEditor * getMarkerSpacing());

        QColor c;

        if (editor()->background().lightness() < 128)
        {
            c = editor()->background().darker(180);
        }
        else
        {
            c = editor()->background().darker(110);
        }

        c.setAlpha(128);
        painter.fillRect(rect, c);
    }
}


//----------------------------------------------------------
/* Gets the distance from one line to another line in the panel
*/
float GlobalCheckerPanel::getMarkerSpacing() const
{
    const int vh = editor()->viewport()->height();
    const int numLinesTotal = getScrollbarValueHeight();
    return static_cast<float>(vh - 2 * verticalOffset()) / numLinesTotal;
}

//---------------------------------------------------------
/*
Gets the size of a message marker.
*/
QSize GlobalCheckerPanel::getMarkerSize() const
{
    return QSize(2.0 * sizeHint().width() / 3.0, markerHeight());
}

//----------------------------------------------------------
/*
* Moves the editor text cursor to the clicked line.
*/
void GlobalCheckerPanel::mousePressEvent(QMouseEvent *e)
{
    auto vsb = editor()->verticalScrollBar();
    auto height = e->pos().y() - verticalOffset();
    int line = qBound<int>(vsb->minimum(), qRound(height / getMarkerSpacing()), vsb->maximum());
    
    vsb->setValue(line);
}

//----------------------------------------------------------
/*
* Moves the editor text cursor to the clicked line.
*/
void GlobalCheckerPanel::wheelEvent(QWheelEvent* e)
{
    editor()->callWheelEvent(e);
}


} //end namespace ito