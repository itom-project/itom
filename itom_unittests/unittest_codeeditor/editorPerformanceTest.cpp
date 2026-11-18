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

#include "codeEditor/managers/modesManager.h"
#include "codeEditor/managers/panelsManager.h"
#include "codeEditor/modes/caretLineHighlight.h"
#include "codeEditor/modes/occurrences.h"
#include "codeEditor/modes/symbolMatcherMode.h"
#include "codeEditor/panels/foldingPanel.h"
#include "codeEditor/panels/lineNumber.h"

#include <qcoreapplication.h>
#include <qelapsedtimer.h>
#include <qevent.h>
#include <qimage.h>
#include <qpainter.h>

using namespace ito;
using namespace ito::test;

/* Performance guard for the script editor (runs in CI).

   The limits are generous (slow CI machines), but far below the values measured
   before the fixes of October 2026 with a 1 MB data listing (108 lines of
   5400 characters):

   * repaint of the viewport:  170 ms (400 ms with visible whitespaces)
   * one typed character:      180 ms
   * cursor onto a frequent word (occurrence highlighting): > 3000 ms

   If one of these tests fails, a feature does O(file size) or O(line length)
   work per keystroke again. See the git history of the fixed functions
   (CodeEditor::paintTextBlocks, OccurrencesHighlighterMode::findAll).
*/

//-------------------------------------------------------------------------------------
static QString dataListing(int lines, int numbersPerLine)
{
    QString text = "Listing of Universal Plot 2D Data\n\nkey: value (x)\n";

    for (int l = 0; l < lines; ++l)
    {
        for (int n = 0; n < numbersPerLine; ++n)
        {
            text += QString(" %1.%2E-01\t").arg(n % 10).arg(l % 100, 2, 10, QChar('0'));
        }

        text += "\n";
    }

    return text;
}

//-------------------------------------------------------------------------------------
static void pump(int ms)
{
    QElapsedTimer t;
    t.start();

    while (t.elapsed() < ms)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
    }
}

//! editor with the modes and panels, that are active for text files in itom
class PerformanceEditor
{
public:
    PerformanceEditor(bool showWhitespaces) : e(EditorFileType::PlainText), ed(e.editor())
    {
        ed->resize(1200, 800);
        ed->setShowWhitespaces(showWhitespaces);
        ed->panels()->append(Panel::Ptr(new FoldingPanel(false, "FoldingPanel")));
        ed->panels()->append(Panel::Ptr(new LineNumberPanel("LineNumberPanel")));
        auto occ = new OccurrencesHighlighterMode("OccurrencesHighlighterMode");
        occ->setDelay(20);
        ed->modes()->append(Mode::Ptr(occ));
        ed->modes()->append(Mode::Ptr(new SymbolMatcherMode("SymbolMatcherMode")));
        ed->modes()->append(Mode::Ptr(new CaretLineHighlighterMode("CaretLineHighlighterMode")));
        ed->show();
        ed->setPlainText(dataListing(108, 360)); // ~ 500 kB, 5400 characters per line
        pump(100);
        setCursor(40, 200);
    }

    void setCursor(int line, int column)
    {
        QTextCursor c(ed->document()->findBlockByNumber(line));
        c.setPosition(c.position() + column);
        ed->setTextCursor(c);
        pump(50);
    }

    double paintMs(int repetitions = 5)
    {
        QImage img(1200, 800, QImage::Format_ARGB32);
        {
            QPainter warmup(&img);
            ed->render(&warmup);
        }

        QElapsedTimer t;
        t.start();

        for (int i = 0; i < repetitions; ++i)
        {
            QPainter p(&img);
            ed->render(&p);
        }

        return t.elapsed() / (double)repetitions;
    }

    double keyMs(int repetitions = 5)
    {
        QImage img(1200, 800, QImage::Format_ARGB32);
        QElapsedTimer t;
        t.start();

        for (int i = 0; i < repetitions; ++i)
        {
            QKeyEvent k(QEvent::KeyPress, Qt::Key_X, Qt::NoModifier, "x");
            QCoreApplication::sendEvent(ed, &k);
            QCoreApplication::processEvents();
            QPainter p(&img); // the edit repaints the viewport
            ed->render(&p);
        }

        return t.elapsed() / (double)repetitions;
    }

    HighlightedEditor e;
    CodeEditor* ed;
};

//-------------------------------------------------------------------------------------
TEST(EditorPerformanceTest, repaintOfLongLines)
{
    PerformanceEditor pe(false);
    double ms = pe.paintMs();
    EXPECT_LT(ms, 100) << "repaint took " << ms << " ms (was 170 ms before the fix)";
}

//-------------------------------------------------------------------------------------
TEST(EditorPerformanceTest, repaintOfLongLinesWithVisibleWhitespaces)
{
    PerformanceEditor pe(true);
    double ms = pe.paintMs();
    EXPECT_LT(ms, 150) << "repaint took " << ms << " ms (was 400 ms before the fix)";
}

//-------------------------------------------------------------------------------------
TEST(EditorPerformanceTest, typingInLongLines)
{
    PerformanceEditor pe(false);
    double ms = pe.keyMs();
    EXPECT_LT(ms, 150) << "one typed character took " << ms << " ms (was 180 ms before the fix)";
}

//-------------------------------------------------------------------------------------
TEST(EditorPerformanceTest, cursorOntoFrequentWord)
{
    PerformanceEditor pe(false);

    // "01" occurs ~ 20000 times in the listing
    QTextBlock block = pe.ed->document()->findBlockByNumber(50);
    int col = block.text().indexOf("E-01") + 3;

    QElapsedTimer t;
    t.start();
    pe.setCursor(50, col);
    pump(300);
    double highlight = t.elapsed() - 350;

    t.restart();
    pe.setCursor(0, 2);
    pump(300);
    double clear = t.elapsed() - 350;

    EXPECT_LT(highlight, 500) << "highlighting took " << highlight << " ms (was > 3000 ms)";
    EXPECT_LT(clear, 500) << "clearing took " << clear << " ms";
}
