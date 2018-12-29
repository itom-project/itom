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
void ScriptEditorPrinter::formatPage(QPainter &painter, bool drawing, QRect &area, int pagenr)
{
    QString filename = this->docName();
    QString date = QDateTime::currentDateTime().toString(Qt::LocalDate);
    QString page = QString::number(pagenr);
    int width = area.width();
    int dateWidth = painter.fontMetrics().width(date);
    filename = painter.fontMetrics().elidedText(filename, Qt::ElideMiddle, 0.8 * (width - dateWidth));
        
    painter.save();
    painter.setFont(QFont("Helvetica", 10, QFont::Normal, false));
    painter.setPen(QColor(Qt::black)); 
    if (drawing)
    {
        //painter.drawText(area.right() - painter.fontMetrics().width(header), area.top() + painter.fontMetrics().ascent(), header);
        painter.drawText(area.left() - 25, area.top() + painter.fontMetrics().ascent(), filename);
        painter.drawText(area.right() + 25 - painter.fontMetrics().width(date), area.top() + painter.fontMetrics().ascent(), date);
        painter.drawText((area.left() + area.right())*0.5, area.bottom() - painter.fontMetrics().ascent(), page);
    }
    area.setTop(area.top() + painter.fontMetrics().height() + 30);
    area.setBottom(area.bottom() - painter.fontMetrics().height() - 50);
    painter.restore();
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
    QSizeF pageSize = pageRect().size(); // page size in pixels
    // Calculate the rectangle where to lay out the text
    const double tm = mmToPixels(12);
    qreal footerHeight;

    QPainter painter(this);
    {
        footerHeight = painter.fontMetrics().height();
    }
    const QRectF textRect(0, 0, pageSize.width(), pageSize.height() - footerHeight);

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

        qDebug() << painter.transform() << resolution();

        float scale = (float)GuiHelper::getScreenLogicalDpi() / (float)resolution();

        clone->setPageSize(pageRect().size() * scale);
        const int pageCount = clone->pageCount();

        bool firstPage = true;
        for (int pageIndex = 0; pageIndex < pageCount; ++pageIndex)
        {
            if (!firstPage)
            {
                newPage();
            }

            paintPage(pageIndex, pageCount, &painter, clone, textRect, scale, footerHeight);
            firstPage = false;
        }

        DELETE_AND_SET_NULL(clone);
    }
    return true;
}

double ScriptEditorPrinter::mmToPixels(int mm)
{
    return mm * 0.039370147 * resolution();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Print a range of lines to a printer.
void ScriptEditorPrinter::paintPage(int pageNumber, int pageCount, QPainter* painter, QTextDocument* doc, const QRectF& textRect, float scale, qreal footerHeaderHeight)
{
    //qDebug() << "Printing page" << pageNumber;
    const QSizeF pageSize = paperRect().size();
    const QRect page = pageRect();

    const QRectF borderRect(0, 0, page.width(), page.height());
    painter->drawRect(borderRect);

    painter->save();
    // textPageRect is the rectangle in the coordinate system of the QTextDocument, in pixels,
    // and starting at (0,0) for the first page. Second page is at y=doc->pageSize().height().
    const QRectF textPageRect(0, pageNumber * doc->pageSize().height(), doc->pageSize().width(), doc->pageSize().height());
    // Clip the drawing so that the text of the other pages doesn't appear in the margins
    painter->setClipRect(textRect);
    painter->scale(1./scale, 1./scale);
    // Translate so that 0,0 is now the page corner
    painter->translate(0, -textPageRect.top());
    // Translate so that 0,0 is the text rect corner
    painter->translate(textRect.left(), textRect.top());
    doc->drawContents(painter);
    painter->restore();

    // Footer: page number or "end"
    QRectF footerRect = textRect;
    footerRect.setTop(textRect.bottom());
    footerRect.setHeight(footerHeaderHeight);

    QRectF headerRect = textRect;
    headerRect.setHeight(footerHeaderHeight);

    QString filename = this->docName();
    QString date = QDateTime::currentDateTime().toString(Qt::LocalDate);
    QString pageStr = QString("%1/%2").arg(pageNumber + 1).arg(pageCount);
    int width = footerRect.width();
    

    painter->save();
    painter->setFont(QFont("Helvetica", 10, QFont::Normal, false));
    painter->setPen(QColor(Qt::black));

    int dateWidth = painter->fontMetrics().width(date);
    filename = painter->fontMetrics().elidedText(filename, Qt::ElideMiddle, 0.8 * (width - dateWidth));

    //painter.drawText(area.right() - painter.fontMetrics().width(header), area.top() + painter.fontMetrics().ascent(), header);
    painter->drawText(footerHeaderHeight, Qt::AlignVCenter | Qt::AlignLeft, filename);
    painter->drawText(footerHeaderHeight, Qt::AlignVCenter | Qt::AlignRight, date);
    painter->drawText(footerRect, Qt::AlignVCenter | Qt::AlignRight, pageStr);

    painter->restore();

    /*if (pageNumber == pageCount - 1)
        painter->drawText(footerRect, Qt::AlignCenter, QObject::tr("Fin du Bordereau de livraison"));
    else
        painter->drawText(footerRect, Qt::AlignVCenter | Qt::AlignRight, QObject::tr("Page %1/%2").arg(pageNumber + 1).arg(pageCount));*/
}

//----------------------------------------------------------------------------------------------------------------------------------
// Set the print magnification in points.
void ScriptEditorPrinter::setMagnification(int magnification)
{
    m_magnification = magnification;
}

//----------------------------------------------------------------------------------------------------------------------------------
void ScriptEditorPrinter::setAlphaLevel(int level)
{
    m_alphaLevel = level;
}

} // end namespace ito
