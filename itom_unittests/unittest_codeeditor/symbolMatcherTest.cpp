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
#include "codeEditor/modes/lineBackgroundMarker.h"
#include "codeEditor/modes/symbolMatcherMode.h"

#include <qcoreapplication.h>
#include <qelapsedtimer.h>

#include <algorithm>

using namespace ito;
using namespace ito::test;

//-------------------------------------------------------------------------------------
static void setCursor(CodeEditor* ed, int line, int column)
{
    QTextCursor c(ed->document()->findBlockByNumber(line));
    c.setPosition(c.position() + column);
    ed->setTextCursor(c);
    QCoreApplication::processEvents();
}

//-------------------------------------------------------------------------------------
TEST(SymbolMatcherTest, highlightsMatchingBrackets)
{
    HighlightedEditor e(EditorFileType::PlainText);
    CodeEditor* ed = e.editor();
    ed->modes()->append(Mode::Ptr(new SymbolMatcherMode("SymbolMatcherMode")));
    ed->setPlainText("f(a, [1, 2], {x: (y)})\nno brackets\n");

    auto starts = [ed]() {
        QList<int> l;

        foreach (const QTextEdit::ExtraSelection& s, ed->extraSelections())
        {
            l << s.cursor.selectionStart();
        }

        std::sort(l.begin(), l.end());
        return l;
    };

    setCursor(ed, 0, 1); // before '('
    EXPECT_EQ(starts(), QList<int>() << 1 << 21);

    setCursor(ed, 0, 5); // before '['
    EXPECT_EQ(starts(), QList<int>() << 5 << 10);

    setCursor(ed, 1, 3); // no bracket
    EXPECT_EQ(ed->extraSelections().size(), 0);
}

//-------------------------------------------------------------------------------------
TEST(SymbolMatcherTest, longLinesStayFast)
{
    // data listing with 6000 characters per line, few brackets
    QString line = "(";

    for (int i = 0; i < 450; ++i)
    {
        line += QString(" %1.123456E-01\t").arg(i % 10);
    }

    line += ")";
    QString text;

    for (int l = 0; l < 100; ++l)
    {
        text += line + "\n";
    }

    HighlightedEditor e(EditorFileType::PlainText);
    CodeEditor* ed = e.editor();
    ed->resize(1000, 600);
    ed->show();
    ed->modes()->append(Mode::Ptr(new SymbolMatcherMode("SymbolMatcherMode")));
    ed->setPlainText(text);
    QCoreApplication::processEvents();

    QElapsedTimer t;
    t.start();

    for (int i = 0; i < 20; ++i)
    {
        setCursor(ed, 50, (i % 2) ? 1 : line.size() - 1); // next to '(' or ')'
    }

    qint64 withBrackets = t.restart();

    for (int i = 0; i < 20; ++i)
    {
        setCursor(ed, 50, 100 + i); // inside the numbers
    }

    qint64 withoutBrackets = t.restart();

    EXPECT_EQ(ed->extraSelections().size(), 0);
    EXPECT_LT(withBrackets, 2000) << withBrackets << " ms for 20 cursor moves";
    EXPECT_LT(withoutBrackets, 1000) << withoutBrackets << " ms for 20 cursor moves";
}

//-------------------------------------------------------------------------------------
TEST(LineBackgroundMarkerTest, addAndClearMarkers)
{
    HighlightedEditor e(EditorFileType::PlainText);
    CodeEditor* ed = e.editor();
    auto marker = new LineBackgroundMarkerMode("LineBackgroundMarkerMode", QColor(Qt::red));
    ed->modes()->append(Mode::Ptr(marker));
    ed->setPlainText("a\nb\nc\nd\n");

    marker->addMarker(1);
    marker->addMarker(3);
    EXPECT_EQ(ed->extraSelections().size(), 2);

    marker->clearAllMarkers();
    EXPECT_EQ(ed->extraSelections().size(), 0);
}
