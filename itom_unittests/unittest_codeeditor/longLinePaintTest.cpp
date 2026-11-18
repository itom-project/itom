/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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
*********************************************************************** */

#include "codeEditorTestHelper.h"
#include "gtest/gtest.h"

#include <qcoreapplication.h>
#include <qelapsedtimer.h>
#include <qimage.h>
#include <qpainter.h>
#include <qplaintextedit.h>
#include <qscrollbar.h>

using namespace ito;
using namespace ito::test;

/* CodeEditor draws very long lines clipped to the visible part (CodeEditor::paintTextBlocks).
   These tests check that the result is identical to the drawing of QPlainTextEdit and
   that the drawing is fast. */

//-------------------------------------------------------------------------------------
static QString longLinesText(int lines, int numbersPerLine)
{
    QString line;

    for (int i = 0; i < numbersPerLine; ++i)
    {
        line += QString(" %1.123456E-01\t").arg(i % 10);
    }

    line += "end of line";
    QString text;

    for (int l = 0; l < lines; ++l)
    {
        // every row is shifted by one character, such that the tab stops differ
        text += QString("row%1 ").arg(l) + line.mid(l) + "\n";
    }

    return text;
}

//-------------------------------------------------------------------------------------
static int differingPixels(const QImage& a, const QImage& b)
{
    int n = 0;

    for (int y = 0; y < a.height(); ++y)
    {
        for (int x = 0; x < a.width(); ++x)
        {
            if (a.pixel(x, y) != b.pixel(x, y))
            {
                n++;
            }
        }
    }

    return n;
}

//-------------------------------------------------------------------------------------
static QImage renderViewport(QPlainTextEdit* w)
{
    QImage img(w->viewport()->size(), QImage::Format_ARGB32);
    img.fill(Qt::white);
    QPainter p(&img);
    w->viewport()->render(&p);
    return img;
}

//! applies a text selection and two extra selections (one full width like the caret line)
static void applySelections(QPlainTextEdit* w)
{
    QTextCursor s(w->document()->findBlockByNumber(5));
    s.setPosition(s.position() + 380);
    s.setPosition(s.position() + 150, QTextCursor::KeepAnchor);
    w->setTextCursor(s);

    QTextEdit::ExtraSelection fullWidth;
    fullWidth.cursor = QTextCursor(w->document()->findBlockByNumber(7));
    fullWidth.cursor.clearSelection();
    fullWidth.format.setBackground(QColor(255, 255, 0));
    fullWidth.format.setProperty(QTextFormat::FullWidthSelection, true);

    QTextEdit::ExtraSelection range;
    range.cursor = QTextCursor(w->document()->findBlockByNumber(9));
    range.cursor.setPosition(range.cursor.position() + 440);
    range.cursor.setPosition(range.cursor.position() + 30, QTextCursor::KeepAnchor);
    range.format.setBackground(QColor(0, 255, 255));

    w->setExtraSelections(QList<QTextEdit::ExtraSelection>() << fullWidth << range);
}

//! compares the code editor with a QPlainTextEdit for the same text and scroll position
static int compareWithQPlainTextEdit(bool showWhitespaces, int hscroll, bool withSelections)
{
    HighlightedEditor e(EditorFileType::PlainText);
    CodeEditor* ed = e.editor();
    ed->resize(800, 300);
    ed->setShowWhitespaces(showWhitespaces);

    QPlainTextEdit ref;
    ref.resize(800, 300);
    ref.setFont(ed->font());
    ref.setLineWrapMode(QPlainTextEdit::NoWrap);
    ref.setFrameShape(ed->frameShape());
    ref.setTabStopDistance(ed->tabStopDistance());
    ref.document()->setDocumentMargin(ed->document()->documentMargin());
    ref.document()->setDefaultTextOption(ed->document()->defaultTextOption());

    // the test style draws the default text with its unique color (see createUniqueColorStyle)
    QPalette pal = ed->palette();
    pal.setColor(QPalette::Text, QColor(1, 100, 0));
    ref.setPalette(pal);

    const QString text = longLinesText(30, 450);
    ed->setPlainText(text);
    ref.setPlainText(text);
    ed->show();
    ref.show();
    QCoreApplication::processEvents();

    // cursor in the visible range of a long line
    for (QPlainTextEdit* w : {static_cast<QPlainTextEdit*>(ed), &ref})
    {
        QTextCursor c(w->document()->findBlockByNumber(3));
        c.setPosition(c.position() + hscroll / 7 + 5);
        w->setTextCursor(c);

        if (withSelections)
        {
            applySelections(w);
        }

        w->horizontalScrollBar()->setValue(hscroll);
    }

    QCoreApplication::processEvents();
    EXPECT_EQ(ed->horizontalScrollBar()->value(), ref.horizontalScrollBar()->value());

    return differingPixels(renderViewport(ed), renderViewport(&ref));
}

//-------------------------------------------------------------------------------------
TEST(LongLinePaintTest, identicalToQPlainTextEdit)
{
    // a few pixels of tolerance for the blinking cursor
    EXPECT_LE(compareWithQPlainTextEdit(false, 0, false), 20);
    EXPECT_LE(compareWithQPlainTextEdit(false, 3000, false), 20);
    EXPECT_LE(compareWithQPlainTextEdit(false, 3000, true), 20);
    EXPECT_LE(compareWithQPlainTextEdit(false, 100000, false), 20); // scrolled to the end
}

//-------------------------------------------------------------------------------------
TEST(LongLinePaintTest, nearlyIdenticalWithVisibleWhitespaces)
{
    // the tab arrows are placed with sub pixel differences (< 0.5 % of the pixels)
    int total = 800 * 300;
    EXPECT_LE(compareWithQPlainTextEdit(true, 3000, true), total / 200);
}

//-------------------------------------------------------------------------------------
TEST(LongLinePaintTest, paintTimeIndependentOfLineLength)
{
    auto paintTime = [](int numbersPerLine) {
        HighlightedEditor e(EditorFileType::PlainText);
        CodeEditor* ed = e.editor();
        ed->resize(1200, 800);
        ed->show();
        ed->setPlainText(longLinesText(100, numbersPerLine));
        QCoreApplication::processEvents();

        QImage img(1200, 800, QImage::Format_ARGB32);
        QPainter warmup(&img);
        ed->render(&warmup);
        warmup.end();

        QElapsedTimer t;
        t.start();

        for (int i = 0; i < 5; ++i)
        {
            QPainter p(&img);
            ed->render(&p);
        }

        return t.elapsed() / 5.0;
    };

    double shortLines = paintTime(12); // ~ 160 characters
    double longLines = paintTime(450); // ~ 6000 characters

    // before the clipped drawing, the long lines were ~18 times slower than the short ones
    EXPECT_LT(longLines, 6 * shortLines + 20)
        << "short " << shortLines << " ms, long " << longLines << " ms";
}
