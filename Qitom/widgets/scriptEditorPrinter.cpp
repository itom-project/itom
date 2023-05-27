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
*********************************************************************** */

#include "scriptEditorPrinter.h"

#include <qdatetime.h>
#include <qpainter.h>
#include <qstack.h>
#include <qrect.h>
#include <qdebug.h>

#include "../codeEditor/codeEditor.h"
#include "../global.h"
#include "../helper/guiHelper.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
ScriptEditorPrinter::ScriptEditorPrinter(QPrinter::PrinterMode mode /*= QPrinter::ScreenResolution*/)
    : QPrinter(mode),
        m_magnification(0),
        m_alphaLevel(0)
{
    setColorMode(QPrinter::Color);
    setPageOrder(QPrinter::FirstPageFirst);
}

//----------------------------------------------------------------------------------------------------------------------------------
ScriptEditorPrinter::~ScriptEditorPrinter()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
QRect ScriptEditorPrinter::formatPage(QPainter &painter, bool drawing, const QRect& area, int pageNumber, int pageCount)
{
    painter.save();
    painter.setFont(QFont("Verdana", 10, QFont::Normal, false));
    painter.setPen(QColor(Qt::black));

    QString filename = this->docName();
    QString date = QDateTime::currentDateTime().toString(QLocale::system().dateFormat(QLocale::ShortFormat));
    QString page = QObject::tr("Page %1/%2").arg(pageNumber).arg(pageCount);
    int width = area.width();
#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))

    int dateWidth = painter.fontMetrics().horizontalAdvance(date);
#else

    int dateWidth = painter.fontMetrics().width(date);
#endif

    filename = painter.fontMetrics().elidedText(filename, Qt::ElideMiddle, 0.8 * (width - dateWidth));

    if (drawing)
    {
        //painter.drawText(area.right() - painter.fontMetrics().width(header), area.top() + painter.fontMetrics().ascent(), header);
        painter.drawText(area.left(), area.top() + painter.fontMetrics().ascent(), filename);
#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
        painter.drawText(
            area.right() - painter.fontMetrics().horizontalAdvance(date),
            area.top() + painter.fontMetrics().ascent(),
            date);
#else
        painter.drawText(
            area.right() - painter.fontMetrics().width(date),
            area.top() + painter.fontMetrics().ascent(),
            date);
#endif

        painter.drawText((area.left() + area.right())*0.5, area.bottom() - painter.fontMetrics().ascent(), page);
    }

    QRect textArea = area;
    textArea.setTop(area.top() + 1.5 * painter.fontMetrics().height());
    textArea.setBottom(area.bottom() - 1.5 * painter.fontMetrics().height());
    painter.restore();

    return textArea;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Print a range of lines to a printer.
int ScriptEditorPrinter::printRange(CodeEditor *editor, int from, int to)
{
    // Sanity check.
    if (!editor)
    {
        return false;
    }

    QTextDocument *doc = editor->document();

    QSizeF f = doc->pageSize();
    const QSize pageSize = pageLayout().paintRectPixels(resolution()).size(); // page size in pixels
    QPainter painter(this);

    QTextDocument *clone = doc ? doc->clone() : NULL;

    if (clone)
    {
        for (QTextBlock srcBlock = doc->firstBlock(), dstBlock = clone->firstBlock();
            srcBlock.isValid() && dstBlock.isValid();
            srcBlock = srcBlock.next(), dstBlock = dstBlock.next())
        {
#if QT_VERSION < 0x050600
            dstBlock.layout()->setAdditionalFormats(srcBlock.layout()->additionalFormats());
#else
            dstBlock.layout()->setFormats(srcBlock.layout()->formats());
#endif
        }

        float scale = (float)GuiHelper::getScreenLogicalDpi() / (float)resolution();

        QRect textRect(0, 0, pageSize.width(), pageSize.height());
        QRect textOnlyRect = formatPage(painter, false, textRect, 1, 1); //do not print something, only reduce textRect to cover header and footer

        clone->setPageSize(textOnlyRect.size() * scale);
        const int pageCount = clone->pageCount();

        bool firstPage = true;
        for (int pageIndex = 0; pageIndex < pageCount; ++pageIndex)
        {
            if (!firstPage)
            {
                newPage();
            }

            formatPage(painter, true, textRect, pageIndex + 1, pageCount);
            paintPage(pageIndex, pageCount, &painter, clone, textOnlyRect, scale);
            firstPage = false;
        }

        DELETE_AND_SET_NULL(clone);
    }
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Print a range of lines to a printer.
void ScriptEditorPrinter::paintPage(int pageNumber, int pageCount, QPainter* painter, QTextDocument* doc, const QRectF& textRect, float scale)
{
    const QSizeF pageSize = pageLayout().paintRectPixels(resolution()).size();
    const QRect page = pageLayout().paintRectPixels(resolution());

    /*const QRectF borderRect(0, 0, page.width(), page.height());
    painter->drawRect(borderRect);*/

    painter->save();
    // textPageRect is the rectangle in the coordinate system of the QTextDocument, in pixels,
    // and starting at (0,0) for the first page. Second page is at y=doc->pageSize().height().
    const QRectF textPageRect(0, pageNumber * doc->pageSize().height(), doc->pageSize().width(), doc->pageSize().height());
    // Clip the drawing so that the text of the other pages doesn't appear in the margins
    //painter->setClipRect(textRect);
    // Translate so that 0,0 is the text rect corner
    painter->translate(textRect.left(), textRect.top());
    painter->scale(1. / scale, 1. / scale);
    // Translate so that 0,0 is now the page corner
    painter->translate(0, -textPageRect.top());

    doc->drawContents(painter, textPageRect);
    painter->restore();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Set the print magnification in points.
void ScriptEditorPrinter::setMagnification(int magnification)
{
    m_magnification = magnification;
}

} // end namespace ito
