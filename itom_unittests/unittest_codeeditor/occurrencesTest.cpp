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
#include "codeEditor/managers/textDecorationsManager.h"
#include "codeEditor/modes/occurrences.h"

#include <qcoreapplication.h>
#include <qelapsedtimer.h>

using namespace ito;
using namespace ito::test;

//! gives access to the protected findAll method
class OccurrencesTestMode : public OccurrencesHighlighterMode
{
public:
    OccurrencesTestMode() : OccurrencesHighlighterMode("OccurrencesTestMode")
    {
    }

    using OccurrencesHighlighterMode::findAll;
    static const int kMaxResults = OccurrencesHighlighterMode::MaxResults;
};

//! processes events for the given time (the occurrences are highlighted with a delay)
static void pump(int ms)
{
    QElapsedTimer t;
    t.start();

    while (t.elapsed() < ms)
    {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
    }
}

//! a text like an exported data listing: many lines with many numbers
static QString dataListing(int lines, int numbersPerLine)
{
    QString text = "Listing of Data\nkey: value (x)\n";

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
TEST(OccurrencesTest, findAllWholeWordAndSubstring)
{
    OccurrencesTestMode m;
    QString text = "ab abc ab\nxab ab";

    auto whole = m.findAll(text, "ab", true, true);
    ASSERT_EQ(whole.size(), 3);
    EXPECT_EQ(whole[0], qMakePair(0, 2));
    EXPECT_EQ(whole[1], qMakePair(7, 9));
    EXPECT_EQ(whole[2], qMakePair(14, 16));

    auto sub = m.findAll(text, "ab", false, true);
    EXPECT_EQ(sub.size(), 5);

    // case insensitive
    EXPECT_EQ(m.findAll("Ab ab AB", "ab", true, false).size(), 3);
    EXPECT_EQ(m.findAll("Ab ab AB", "ab", true, true).size(), 1);

    // empty search string
    EXPECT_EQ(m.findAll(text, "", true, true).size(), 0);
}

//-------------------------------------------------------------------------------------
TEST(OccurrencesTest, findAllSearchesLiterally)
{
    OccurrencesTestMode m;

    // regex characters in the word under the cursor must not be interpreted
    EXPECT_EQ(m.findAll("a.b axb a.b", "a.b", false, true).size(), 2);
    EXPECT_EQ(m.findAll("f(x) f(x)", "f(x)", false, true).size(), 2);
    EXPECT_EQ(m.findAll("1+1 1+1 11", "1+1", false, true).size(), 2);
}

//-------------------------------------------------------------------------------------
TEST(OccurrencesTest, findAllMaxResults)
{
    OccurrencesTestMode m;
    QString text = QString("x ").repeated(1000);

    EXPECT_EQ(m.findAll(text, "x", true, true).size(), 1000);
    EXPECT_EQ(m.findAll(text, "x", true, true, 10).size(), 10);
    EXPECT_EQ(m.findAll(text, "x", true, true, 0).size(), 0);
    EXPECT_EQ(m.findAll(text, "x", true, true, -1).size(), 1000);
}

//-------------------------------------------------------------------------------------
TEST(OccurrencesTest, findAllIsFastForFrequentWordsInBigTexts)
{
    // 500 kB text with ~36000 occurrences of "01". The former implementation
    // restarted the regex match after every hit and needed several seconds.
    OccurrencesTestMode m;
    QString text = dataListing(100, 360);
    ASSERT_GT(text.size(), 300000);

    QElapsedTimer t;
    t.start();
    auto all = m.findAll(text, "01", true, true);
    qint64 elapsed = t.elapsed();

    EXPECT_GT(all.size(), 1000);
    EXPECT_LT(elapsed, 500) << "findAll took " << elapsed << " ms";
}

//-------------------------------------------------------------------------------------
TEST(OccurrencesTest, highlightFrequentWordInEditorStaysResponsive)
{
    HighlightedEditor e(EditorFileType::PlainText);
    CodeEditor* ed = e.editor();
    ed->resize(1000, 600);
    ed->show();

    auto occ = new OccurrencesHighlighterMode("OccurrencesHighlighterMode");
    occ->setDelay(20);
    ed->modes()->append(Mode::Ptr(occ));

    ed->setPlainText(dataListing(100, 360));
    pump(100);

    // cursor onto "01" of a number in the data area
    QTextBlock block = ed->document()->findBlockByNumber(50);
    int col = block.text().indexOf("E-01") + 3;
    ASSERT_GT(col, 3);

    QTextCursor c(block);
    c.setPosition(block.position() + col);

    QElapsedTimer t;
    t.start();
    ed->setTextCursor(c);
    pump(300);
    qint64 highlight = t.elapsed() - 300;

    // the highlighted occurrences are extra selections of the editor
    int selections = ed->extraSelections().size();
    EXPECT_GE(selections, OccurrencesTestMode::kMaxResults - 1);
    EXPECT_LE(selections, OccurrencesTestMode::kMaxResults + 5);

    // move away: decorations are removed
    t.restart();
    ed->setTextCursor(QTextCursor(ed->document()->findBlockByNumber(0)));
    pump(300);
    qint64 clear = t.elapsed() - 300;

    EXPECT_LT(ed->extraSelections().size(), 5);

    // generous limits for slow CI machines (it was > 3000 ms before the fix)
    EXPECT_LT(highlight, 1000) << "highlight took " << highlight << " ms";
    EXPECT_LT(clear, 1000) << "clearing took " << clear << " ms";
}

//-------------------------------------------------------------------------------------
TEST(TextDecorationsManagerTest, batchAppendAndRemove)
{
    HighlightedEditor e(EditorFileType::PlainText);
    CodeEditor* ed = e.editor();
    ed->setPlainText("0123456789\n0123456789\n");

    QList<TextDecoration::Ptr> decos;

    for (int i = 0; i < 6; ++i)
    {
        decos.append(TextDecoration::Ptr(new TextDecoration(ed->textCursor(), i, i + 1, -1, -1, 6 - i)));
    }

    EXPECT_EQ(ed->decorations()->append(decos), 6);
    EXPECT_EQ(ed->extraSelections().size(), 6);

    // already contained decorations are not added twice
    EXPECT_EQ(ed->decorations()->append(decos.mid(0, 2)), 0);
    EXPECT_EQ(ed->extraSelections().size(), 6);

    // the selections are sorted by draw order (last decoration has draw order 1)
    EXPECT_EQ(ed->extraSelections().first().cursor.selectionStart(), 5);

    EXPECT_EQ(ed->decorations()->remove(decos.mid(0, 4)), 4);
    EXPECT_EQ(ed->extraSelections().size(), 2);

    // removing unknown decorations is a no-op
    EXPECT_EQ(ed->decorations()->remove(decos.mid(0, 4)), 0);
    EXPECT_EQ(ed->extraSelections().size(), 2);

    ed->decorations()->clear();
    EXPECT_EQ(ed->extraSelections().size(), 0);
}
