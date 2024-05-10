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

    //-------------------------------------------------------------------------------------
/*
*/
GlobalCheckerPanel::GlobalCheckerPanel(const QString &description /*= ""*/, QWidget *parent /*= nullptr */) :
    Panel("GlobalCheckerPanel", false, description, parent)
{
    setScrollable(true);
    m_breakpointIcon = QIcon(":/breakpoints/icons/itomBreak.png");
    m_bookmarkIcon = QIcon(":/bookmark/icons/bookmark.png");

    m_cacheRenewTimer.setSingleShot(true);
    m_cacheRenewTimer.setInterval(250);
    connect(&m_cacheRenewTimer, &QTimer::timeout, this, &GlobalCheckerPanel::renewItemCache);
}

//-------------------------------------------------------------------------------------
/*
*/
GlobalCheckerPanel::~GlobalCheckerPanel()
{

}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

    const auto style = vsb->style();// qApp->style();
    QStyleOptionSlider opt;
    opt.initFrom(vsb);

    // Get the area in which the slider handle may move.
    // usually grooveRect is correct, however if stylesheets are
    // applied, addPageRect seems to deliver better results.
    QRect grooveRect = style->subControlRect(QStyle::CC_ScrollBar, &opt, QStyle::SC_ScrollBarGroove, this);
    QRect addPageRect = style->subControlRect(QStyle::CC_ScrollBar, &opt, QStyle::SC_ScrollBarAddPage, this);

    if (opt.orientation == Qt::Horizontal)
    {
        return std::max(grooveRect.x(), addPageRect.x());
    }

    return std::max(grooveRect.y(), addPageRect.y());
}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
void GlobalCheckerPanel::renewItemCache()
{
    createItemCache();
    update();
}

//-------------------------------------------------------------------------------------
void GlobalCheckerPanel::createItemCache()
{
    const QTextDocument* td = editor()->document();
    const QScrollBar* sb = editor()->verticalScrollBar();
    QTextBlock b = td->firstBlock();
    TextBlockUserData* tbud;
    CheckerItem* current = nullptr;

    m_itemCache.clear();
    m_itemCache.reserve(sb->maximum() + 1);

    while (b.isValid())
    {
        tbud = dynamic_cast<TextBlockUserData*>(b.userData());

        if (b.isVisible())
        {
            m_itemCache.push_back(CheckerItem(b.blockNumber()));
            current = &(m_itemCache.last());
        }
        else if (current)
        {
            // there is at least one invisible item below current
            current->headOfCollapsedFold = true;
        }

        if (tbud && current)
        {
            foreach(const CodeCheckerItem & cci, tbud->m_checkerMessages)
            {
                if (cci.type() > current->worstCaseStatus)
                {
                    current->worstCaseStatus = cci.type();
                }
            }

            if (b.isVisible())
            {
                current->bookmark = tbud->m_bookmark;
                current->breakpoint = tbud->m_breakpointType != ito::TextBlockUserData::TypeNoBp;
            }
        }

        b = b.next();
    }
}

//-------------------------------------------------------------------------------------
/*
Draw messages from all subclass of CheckerMode currently
installed on the editor.
*/
void GlobalCheckerPanel::drawMessages(QPainter& painter)
{
    if (m_itemCache.size() != editor()->lineCount())
    {
        // immediate refresh the cache
        createItemCache();
    }
    else if (!m_cacheRenewTimer.isActive())
    {
        m_cacheRenewTimer.start();
    }
    /*else
    {
        // the timer exceeds soon, renews the cache and calls update() again
    }*/

    QIcon icon;
    QRect rect;
    QBrush brushInfo(QColor(60, 111, 179));
    QBrush brushWarning(QColor(241, 133, 46));
    QBrush brushError(QColor(226, 0, 0));
    QBrush brushCollapsedFold(QColor(145, 205, 251));

    QSize markerSize = getMarkerSize();
    int voffset = verticalOffset();

    // b.blockNumber() is zero-based
    float markerSpacing = getMarkerSpacing();
    const float offset1 = markerSpacing / 2.0 - markerSize.height() / 2.0;
    int sizeHintWidth = sizeHint().width();

    for (int blockIndex = 0; blockIndex < m_itemCache.size(); ++blockIndex)
    {
        const CheckerItem& item = m_itemCache[blockIndex];

        if (item.headOfCollapsedFold)
        {
            rect = QRect();
            rect.setX(sizeHintWidth * 2.0 / 3.0);
            rect.setWidth(sizeHintWidth / 3.0);
            rect.setY(voffset + blockIndex * markerSpacing + markerSpacing / 2.0 - 15.0 / 2.0);
            rect.setHeight(15.0);
            painter.fillRect(rect, brushCollapsedFold);
        }

        if (item.worstCaseStatus > 0)
        {
            rect = QRect();
            rect.setX(sizeHintWidth / 6);
            rect.setY(qRound(voffset + blockIndex * markerSpacing + offset1));
            rect.setSize(markerSize);

            switch (item.worstCaseStatus)
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

        if (item.bookmark)
        {
            int s = std::max(2 + sizeHintWidth / 2, 8);
            rect = QRect();
            rect.setX((sizeHintWidth - s) / 2);
            rect.setY(qRound(voffset + blockIndex * markerSpacing - s / 2.0 + markerSpacing / 2.0));
            rect.setWidth(s);
            rect.setHeight(s);
            m_bookmarkIcon.paint(&painter, rect);
        }

        if (item.breakpoint)
        {
            int s = std::max(sizeHintWidth / 2, 8);
            rect = QRect();
            rect.setX((sizeHintWidth - s) / 2);
            rect.setY(qRound(voffset + blockIndex * markerSpacing - s / 2.0 + markerSpacing / 2.0));
            rect.setWidth(s);
            rect.setHeight(s);
            m_breakpointIcon.paint(&painter, rect);
        }
    }
}


//-------------------------------------------------------------------------------------
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

        QColor c = editor()->background();

        if (c.lightness() < 128)
        {
            c = c.darker(50);
        }
        else
        {
            c = c.darker(120);
        }

        painter.fillRect(rect, c);
    }
}


//-------------------------------------------------------------------------------------
/* Gets the distance from one line to another line in the panel
*/
float GlobalCheckerPanel::getMarkerSpacing() const
{
    const int vh = editor()->viewport()->height();
    const int numLinesTotal = getScrollbarValueHeight();
    return static_cast<float>(vh - 2 * verticalOffset()) / numLinesTotal;
}

//-------------------------------------------------------------------------------------
/*
Gets the size of a message marker.
*/
QSize GlobalCheckerPanel::getMarkerSize() const
{
    return QSize(2.0 * sizeHint().width() / 3.0, markerHeight());
}

//-------------------------------------------------------------------------------------
/*
* Moves the editor text cursor to the clicked line.
*/
void GlobalCheckerPanel::mousePressEvent(QMouseEvent *e)
{
    auto vsb = editor()->verticalScrollBar();
    auto markerSpacing = getMarkerSpacing();
    auto height = e->pos().y() - verticalOffset() - markerSpacing / 2.0f;
    int line = qBound<int>(vsb->minimum(), qRound(height / markerSpacing), vsb->maximum());

    if (m_itemCache.size() >= line + 1)
    {
        editor()->setCursorPosition(line, 0);
        editor()->ensureLineVisible(line);
    }
}

//-------------------------------------------------------------------------------------
/*
* Moves the editor text cursor to the clicked line.
*/
void GlobalCheckerPanel::wheelEvent(QWheelEvent* e)
{
    editor()->callWheelEvent(e);
}


} //end namespace ito
