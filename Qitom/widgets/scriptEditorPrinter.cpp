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

#include "../codeEditor/codeEditor.h"
#include "../global.h"

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
int ScriptEditorPrinter::printRange(CodeEditor *editor, int /*from*/, int /*to*/)
{
    
    // Sanity check.
    if (!editor)
    {
        return false;
    }
    
    const QTextDocument *doc = editor->document();
    doc->print(this);
    /*QTextDocument *clone = doc ? doc->clone() : NULL;

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

        clone->setPageSize(pageRect(QPrinter::Millimeter).size());
        clone->print(this);

        DELETE_AND_SET_NULL(clone);
    }*/
/*
    // Setup the printing area.
    QRect def_area;

    def_area.setX(0);
    def_area.setY(0);
    def_area.setWidth(width());
    def_area.setHeight(height());

    // Get the page range.
    int pgFrom, pgTo;

    pgFrom = fromPage();
    pgTo = toPage();

    // Find the position range.
    long startPos, endPos;

    endPos = qsb->SendScintilla(QsciScintillaBase::SCI_GETLENGTH);

    startPos = (from > 0 ? qsb -> SendScintilla(QsciScintillaBase::SCI_POSITIONFROMLINE,from) : 0);

    if (to >= 0)
    {
        long toPos = qsb -> SendScintilla(QsciScintillaBase::SCI_POSITIONFROMLINE,to + 1);

        if (endPos > toPos)
            endPos = toPos;
    }

    if (startPos >= endPos)
        return false;

    QPainter painter(this);
    bool reverse = (pageOrder() == LastPageFirst);
    bool needNewPage = false;

    qsb -> SendScintilla(QsciScintillaBase::SCI_SETPRINTMAGNIFICATION,mag);
    qsb -> SendScintilla(QsciScintillaBase::SCI_SETPRINTWRAPMODE,wrap);

    for (int i = 1; i <= numCopies(); ++i)
    {
        // If we are printing in reverse page order then remember the start
        // position of each page.
        QStack<long> pageStarts;

        int currPage = 1;
        long pos = startPos;

        while (pos < endPos)
        {
            // See if we have finished the requested page range.
            if (pgTo > 0 && pgTo < currPage)
                break;

            // See if we are going to render this page, or just see how much
            // would fit onto it.
            bool render = false;

            if (pgFrom == 0 || pgFrom <= currPage)
            {
                if (reverse)
                    pageStarts.push(pos);
                else
                {
                    render = true;

                    if (needNewPage)
                    {
                        if (!newPage())
                            return false;
                    }
                    else
                        needNewPage = true;
                }
            }

            QRect area = def_area;

            formatPage(painter,render,area,currPage);
            pos = qsb -> SendScintilla(QsciScintillaBase::SCI_FORMATRANGE,render,&painter,area,pos,endPos);

            ++currPage;
        }

        // All done if we are printing in normal page order.
        if (!reverse)
            continue;

        // Now go through each page on the stack and really print it.
        while (!pageStarts.isEmpty())
        {
            --currPage;

            long ePos = pos;
            pos = pageStarts.pop();

            if (needNewPage)
            {
                if (!newPage())
                    return false;
            }
            else
                needNewPage = true;

            QRect area = def_area;

            formatPage(painter,true,area,currPage);
            qsb->SendScintilla(QsciScintillaBase::SCI_FORMATRANGE,true,&painter,area,pos,ePos);
        }
    }*/

    return true;
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
