/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2024, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

// Drawing of the code editor (moved from codeEditor.cpp, see paintTextBlocks for the
// handling of very long lines).

#include "codeEditor.h"

#include <qabstracttextdocumentlayout.h>
#include <qevent.h>
#include <qmath.h>
#include <qpainter.h>
#include <qscrollbar.h>
#include <qtextdocument.h>

#include "managers/panelsManager.h"
#include "utils/utils.h"

namespace ito {

//-----------------------------------------------------------
/*
Overrides paint event to update the list of visible blocks and emit
the painted e->
:param e: paint event
*/
void CodeEditor::paintEvent(QPaintEvent* e)
{
    updateVisibleBlocks();

    paintEventWithoutVisibleBlockUpdate(e);
}

//------------------------------------------------------------------------------
void CodeEditor::paintEventWithoutVisibleBlockUpdate(QPaintEvent* e)
{
    QTextCursor tc = textCursor();
    tc.movePosition(QTextCursor::Start);
    int xoffset = cursorRect(tc).x(); // left offset of first character

    switch (m_edgeMode)
    {
    case EdgeNone:
        break;
    case EdgeLine: {
        QPainter painter(viewport());
        painter.setPen(m_edgeColor);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        int x = fontMetrics().horizontalAdvance(QString(m_edgeColumn, '9'));
#else
        int x = fontMetrics().width(QString(m_edgeColumn, '9'));
#endif
        x += xoffset;
        painter.drawLine(x, 0, x, size().height());
    }
    break;
    case EdgeBackground: {
        QPainter painter(viewport());
        painter.setBrush(m_edgeColor);
        painter.setPen(Qt::NoPen);

#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        int x = fontMetrics().horizontalAdvance(QString(m_edgeColumn, '9'));
#else
        int x = fontMetrics().width(QString(m_edgeColumn, '9'));
#endif
        x += xoffset;
        painter.drawRect(x, 0, size().width() - x, size().height());
    }
    break;
    }

    paintTextBlocks(e);

    if (m_showIndentationGuides)
    {
        QPainter painter(viewport());

        QColor color = m_indentationGuidesColor;
        color.setAlphaF(.5);
        painter.setPen(color);
        int bottom;
        int indentation;

        foreach (const VisibleBlock& block, visibleBlocks())
        {
            bottom = block.topPosition + blockBoundingRect(block.textBlock).height() - 1;
            indentation = Utils::TextBlockHelper::getFoldLvl(block.textBlock);

            //foldLvl is always in steps of 2, since only even numbers are real indentations.
            //Odd numbers are another fold level but with the same indentation than the previous
            //even number.
            indentation = indentation / 2;


            for (int i = 1; i < indentation; ++i)
            {
                painter.drawLine(
                    xoffset + m_indentationBarWidth * i,
                    block.topPosition,
                    xoffset + m_indentationBarWidth * i,
                    bottom);
            }
        }
    }

    emit painted(e);
}

//-----------------------------------------------------------
//! fills a background rectangle (same as the helper in QPlainTextEdit)
static void fillBlockBackground(QPainter* p, const QRectF& rect, const QBrush& brush)
{
    p->save();

    if (brush.style() >= Qt::LinearGradientPattern && brush.style() <= Qt::ConicalGradientPattern)
    {
        p->setBrushOrigin(rect.topLeft());
    }

    p->fillRect(rect, brush);
    p->restore();
}

//-----------------------------------------------------------
/*
Draws the text blocks of the viewport. This is a port of QPlainTextEdit::paintEvent
with one difference: Qt always draws complete lines, even if only a small part of
a very long line is visible (no horizontal clipping). For data listings with
thousands of characters per line, every repaint took hundreds of milliseconds and
the editor became unusable. Lines longer than LongLineThreshold are therefore
drawn by drawLongLineClipped, which only lays out and draws the visible part.
*/
void CodeEditor::paintTextBlocks(QPaintEvent* e)
{
    QPainter painter(viewport());
    QPointF offset(contentOffset());

    QRect er = e->rect();
    QRect viewportRect = viewport()->rect();

    bool editable = !isReadOnly();

    QTextBlock block = firstVisibleBlock();
    qreal maximumWidth = document()->documentLayout()->documentSize().width();

    // Set a brush origin so that the WaveUnderline knows where the wave started
    painter.setBrushOrigin(offset);

    // keep right margin clean from full-width selection
    int maxX = offset.x() + qMax((qreal)viewportRect.width(), maximumWidth) -
        document()->documentMargin() + cursorWidth();
    er.setRight(qMin(er.right(), maxX));
    painter.setClipRect(er);

    if (document()->isEmpty() && !placeholderText().isEmpty())
    {
        QColor col = palette().text().color();
        col.setAlpha(128);
        painter.setPen(col);
        const int margin = int(document()->documentMargin());
        painter.drawText(
            viewportRect.adjusted(margin, margin, -margin, -margin),
            Qt::AlignTop | Qt::TextWordWrap,
            placeholderText());
    }

    QAbstractTextDocumentLayout::PaintContext context = getPaintContext();
    painter.setPen(context.palette.text().color());

    while (block.isValid())
    {
        QRectF r = blockBoundingRect(block).translated(offset);
        QTextLayout* layout = block.layout();

        if (!block.isVisible())
        {
            offset.ry() += r.height();
            block = block.next();
            continue;
        }

        if (r.bottom() >= er.top() && r.top() <= er.bottom())
        {
            QTextBlockFormat blockFormat = block.blockFormat();
            QBrush bg = blockFormat.background();

            if (bg != Qt::NoBrush)
            {
                QRectF contentsRect = r;
                contentsRect.setWidth(qMax(r.width(), maximumWidth));
                fillBlockBackground(&painter, contentsRect, bg);
            }

            QList<QTextLayout::FormatRange> selections;
            int blpos = block.position();
            int bllen = block.length();

            for (int i = 0; i < context.selections.size(); ++i)
            {
                const QAbstractTextDocumentLayout::Selection& range = context.selections.at(i);
                const int selStart = range.cursor.selectionStart() - blpos;
                const int selEnd = range.cursor.selectionEnd() - blpos;

                if (selStart < bllen && selEnd > 0 && selEnd > selStart)
                {
                    QTextLayout::FormatRange o;
                    o.start = selStart;
                    o.length = selEnd - selStart;
                    o.format = range.format;
                    selections.append(o);
                }
                else if (
                    !range.cursor.hasSelection() &&
                    range.format.hasProperty(QTextFormat::FullWidthSelection) &&
                    block.contains(range.cursor.position()))
                {
                    // for full width selections we don't require an actual selection, just
                    // a position to specify the line. that's more convenience in usage.
                    QTextLayout::FormatRange o;
                    QTextLine l = layout->lineForTextPosition(range.cursor.position() - blpos);
                    o.start = l.textStart();
                    o.length = l.textLength();

                    if (o.start + o.length == bllen - 1)
                    {
                        ++o.length; // include newline
                    }

                    o.format = range.format;
                    selections.append(o);
                }
            }

            bool drawCursor =
                ((editable || (textInteractionFlags() & Qt::TextSelectableByKeyboard)) &&
                 context.cursorPosition >= blpos && context.cursorPosition < blpos + bllen);

            bool drawCursorAsBlock = drawCursor && overwriteMode();

            if (drawCursorAsBlock)
            {
                if (context.cursorPosition == blpos + bllen - 1)
                {
                    drawCursorAsBlock = false;
                }
                else
                {
                    QTextLayout::FormatRange o;
                    o.start = context.cursorPosition - blpos;
                    o.length = 1;
                    o.format.setForeground(palette().base());
                    o.format.setBackground(palette().text());
                    selections.append(o);
                }
            }

            int cpos = -1;

            if ((drawCursor && !drawCursorAsBlock) ||
                (editable && context.cursorPosition < -1 && !layout->preeditAreaText().isEmpty()))
            {
                cpos = context.cursorPosition;

                if (cpos < -1)
                {
                    cpos = layout->preeditAreaPosition() - (cpos + 2);
                }
                else
                {
                    cpos -= blpos;
                }
            }

            bool longLine = bllen > LongLineThreshold && layout->lineCount() == 1 &&
                lineWrapMode() == QPlainTextEdit::NoWrap && layout->preeditAreaText().isEmpty();

            if (longLine)
            {
                drawLongLineClipped(painter, block, offset, selections, er, cpos);
            }
            else
            {
                layout->draw(&painter, offset, selections, er);

                if (cpos >= 0)
                {
                    layout->drawCursor(&painter, offset, cpos, cursorWidth());
                }
            }
        }

        offset.ry() += r.height();

        if (offset.y() > viewportRect.height())
        {
            break;
        }

        block = block.next();
    }

    if (backgroundVisible() && !block.isValid() && offset.y() <= er.bottom() &&
        (centerOnScroll() || verticalScrollBar()->maximum() == verticalScrollBar()->minimum()))
    {
        painter.fillRect(
            QRect(QPoint((int)er.left(), (int)offset.y()), er.bottomRight()), palette().window());
    }
}

//-----------------------------------------------------------
/*
Draws only the visible part of a very long line (see paintTextBlocks).

A temporary layout with the visible characters is created and placed at the exact
x-position of its first character in the original layout. The tab stops are passed
as absolute positions, such that the characters keep their positions. The syntax
highlighting formats and the selections are cut to the visible range.
*/
void CodeEditor::drawLongLineClipped(
    QPainter& painter,
    const QTextBlock& block,
    const QPointF& offset,
    const QList<QTextLayout::FormatRange>& selections,
    const QRect& clip,
    int cursorPosInBlock)
{
    QTextLayout* layout = block.layout();
    QTextLine line = layout->lineAt(0);
    const QString text = block.text();
    const int textLen = text.size();

    // visible range in layout coordinates (cursorToX / xToCursor include line.x())
    const qreal left = clip.left() - offset.x();
    const qreal right = clip.right() - offset.x();

    // some characters more at both sides, such that kerning / ligatures at the cut
    // and the cursor at the border are not affected
    const int margin = 8;
    int start = qBound(0, line.xToCursor(left) - margin, textLen);
    int end = qBound(start, line.xToCursor(right) + margin, textLen);

    if (end <= start)
    {
        return;
    }

    const qreal xStart = line.cursorToX(start);

    // the font of the block (layout->font() is not the font used for drawing)
    QFont font = document()->defaultFont();

    if (block.charFormat().hasProperty(QTextFormat::FontFamilies) ||
        block.charFormat().hasProperty(QTextFormat::FontPointSize))
    {
        font = block.charFormat().font();
    }

    QTextLayout subLayout(text.mid(start, end - start), font);
    QTextOption option = layout->textOption();
    option.setWrapMode(QTextOption::NoWrap);

    // The tab stops of the original line are at line.x() + k * tabDistance. They are
    // passed as absolute positions, relative to the start of the sub layout.
    qreal tabDistance = option.tabStopDistance();

    if (tabDistance <= 0)
    {
        tabDistance = 80;
    }

    QList<qreal> tabs;
    const qreal maxTab = right - xStart + 2 * tabDistance;
    const qreal relStart = xStart - line.x();

    for (qreal t = line.x() + tabDistance * (qFloor(relStart / tabDistance) + 1); t - xStart < maxTab;
         t += tabDistance)
    {
        tabs.append(t - xStart);
    }

    option.setTabArray(tabs);
    subLayout.setTextOption(option);

    // cut the formats (syntax highlighting) and selections to the visible range
    QList<QTextLayout::FormatRange> formats;

    foreach (const QTextLayout::FormatRange& f, layout->formats())
    {
        int s = qMax(f.start, start);
        int e = qMin(f.start + f.length, end);

        if (e > s)
        {
            QTextLayout::FormatRange o;
            o.start = s - start;
            o.length = e - s;
            o.format = f.format;
            formats.append(o);
        }
    }

    subLayout.setFormats(formats);

    QList<QTextLayout::FormatRange> subSelections;

    foreach (const QTextLayout::FormatRange& f, selections)
    {
        int s = qMax(f.start, start);
        int e = qMin(f.start + f.length, end);

        if (e > s)
        {
            QTextLayout::FormatRange o;
            o.start = s - start;
            o.length = e - s;
            o.format = f.format;
            subSelections.append(o);
        }
    }

    subLayout.setCacheEnabled(false);
    subLayout.beginLayout();
    QTextLine subLine = subLayout.createLine();
    subLine.setPosition(QPointF(0.0, line.y()));
    subLayout.endLayout();

    const QPointF subOffset(offset.x() + xStart, offset.y());
    subLayout.draw(&painter, subOffset, subSelections, clip);

    if (cursorPosInBlock >= start && cursorPosInBlock <= end)
    {
        subLayout.drawCursor(&painter, subOffset, cursorPosInBlock - start, cursorWidth());
    }
}

} // end namespace ito
