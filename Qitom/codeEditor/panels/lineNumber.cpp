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

#include "lineNumber.h"

#include "../codeEditor.h"
#include <qpainter.h>
#include <qapplication.h>

namespace ito {

//----------------------------------------------------------
/*
*/
LineNumberPanel::LineNumberPanel(const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Panel("LineNumberPanel", false, description, parent),
    m_selecting(false),
    m_startLine(-1),
    m_selStart(-1)
{
    setScrollable(true);
    m_lineColorU = palette().color(QPalette::Disabled, QPalette::WindowText);
    m_lineColorS = palette().color(QPalette::Normal, QPalette::WindowText);
}

//----------------------------------------------------------
/*
*/
LineNumberPanel::~LineNumberPanel()
{
}

//------------------------------------------------------------
/*
Returns the panel size hint (as the panel is on the left, we only need
to compute the width
*/
QSize LineNumberPanel::sizeHint() const
{
    return QSize(lineNumberAreaWidth(), 50);
}

//------------------------------------------------------------
/*
Computes the lineNumber area width depending on the number of lines
in the document

:return: Widtg
*/
int LineNumberPanel::lineNumberAreaWidth() const
{
    int digits = 1;
    int count = std::max(1, editor()->blockCount());
    while (count >= 10)
    {
        count /= 10;
        digits += 1;
    }
#if (QT_VERSION >= QT_VERSION_CHECK(5, 11, 0))
    return 5 + editor()->fontMetrics().horizontalAdvance("9") * digits; /*space*/
#else
    return 5 + editor()->fontMetrics().width("9") * digits; /*space*/
#endif
}

//----------------------------------------------------------
/*
Cancels line selection.
*/
void LineNumberPanel::cancelSelection()
{
    m_selecting = false;
    m_selStart = -1;
}

//----------------------------------------------------------
/*
Starts selecting
*/
void LineNumberPanel::paintEvent(QPaintEvent *e)
{
    //Paints the line numbers
    m_lineColorU = Utils::driftColor(backgroundBrush().color(), 250);
    m_lineColorS = Utils::driftColor(backgroundBrush().color(), 280);
    Panel::paintEvent(e);

    if (isVisible())
    {
        QPainter painter(this);
        CodeEditor *editor = this->editor();
        // get style options (font, size)
        int width = this->width();
        int height = editor->fontMetrics().height();
        QFont font = editor->font();
        QFont bold_font = editor->font();
        bold_font.setBold(true);
        QPen pen = QPen(m_lineColorU);
        QPen pen_selected = QPen(m_lineColorS);
        painter.setFont(font);
        // get selection range
        QPair<int,int> sel= editor->selectionRange();
        bool has_sel = (sel.first != sel.second);
        int cl = editor->currentLineNumber();

        // draw every visible blocks
        int top, line;
        foreach(const VisibleBlock &vb, editor->visibleBlocks())
        {
            line = vb.lineNumber;
            top = vb.topPosition;

            if ((has_sel && (sel.first <= line) && (line <= sel.second)) ||
                    (!has_sel && (cl == line)))
            {
                painter.setPen(pen_selected);
                painter.setFont(bold_font);
            }
            else
            {
                painter.setPen(pen);
                painter.setFont(font);
            }
            painter.drawText(-3, top, width, height,
                                Qt::AlignRight, QString::number(line + 1));
        }
    }
}

//----------------------------------------------------------
/*
Starts selecting
*/
void LineNumberPanel::mousePressEvent(QMouseEvent *e)
{
    m_selecting = true;
    m_selStart = e->pos().y();
    int start;
    int end;
    start = end = editor()->lineNbrFromPosition(m_selStart);
    m_startLine = start;
    if (start >= 0)
    {
        editor()->selectLines(start, end);
    }
}

//----------------------------------------------------------
/*
Updates end of selection if we are currently selecting
*/
void LineNumberPanel::mouseMoveEvent(QMouseEvent *e)
{
    if (m_selecting)
    {
        int end_pos = e->pos().y();
        int end_line = editor()->lineNbrFromPosition(end_pos);
        if (end_line == -1 && editor()->visibleBlocks().size() > 0)
        {
            //take last visible block
            if (end_pos < 50)
            {
                end_line = editor()->visibleBlocks()[0].lineNumber;
                //_, end_line, _ = self.editor.visible_blocks[0]
                end_line -= 1;
            }
            else
            {
                end_line = editor()->visibleBlocks().last().lineNumber;
                //_, end_line, _ = self.editor.visible_blocks[-1]
                end_line += 1;
            }
        }

        editor()->selectLines(m_startLine, end_line);
    }
}

//----------------------------------------------------------
/*
Cancels selection
*/
void LineNumberPanel::mouseReleaseEvent(QMouseEvent *e)
{
    cancelSelection();
}

//----------------------------------------------------------
/*
Override Qt method
*/
void LineNumberPanel::wheelEvent(QWheelEvent *e)
{
    editor()->callWheelEvent(e);
}

} //end namespace ito
