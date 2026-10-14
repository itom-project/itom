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


#include "gtest/gtest.h"

#include "searchBox.h"
#include "widgetTestHelpers.h"

#include <QSignalSpy>
#include <QtTest/QTest>

using namespace itomWidgetsTest;

namespace {

class SearchBoxTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(showAndWait(box, QSize(250, 30)));
    }

    //! approximate center of the clear icon (right side of the line edit).
    QPoint clearIconPos() const
    {
        const int iconSize = box.contentsRect().height();
        return QPoint(box.width() - iconSize / 2 - 2, box.height() / 2);
    }

    //! approximate center of the search icon (left side of the line edit).
    QPoint searchIconPos() const
    {
        const int iconSize = box.contentsRect().height();
        return QPoint(iconSize / 2 + 2, box.height() / 2);
    }

    SearchBox box;
};

} // namespace

TEST_F(SearchBoxTest, DefaultState)
{
    EXPECT_TRUE(box.text().isEmpty());
    EXPECT_FALSE(box.placeholderText().isEmpty());
    EXPECT_FALSE(box.alwaysShowClearIcon());
}

TEST_F(SearchBoxTest, PlaceholderTextCanBeChanged)
{
    box.setPlaceholderText("Filter plugins");
    EXPECT_EQ(box.placeholderText(), QString("Filter plugins"));
    EXPECT_EQ(box.property("placeholderText").toString(), QString("Filter plugins"));
}

TEST_F(SearchBoxTest, TypingChangesText)
{
    QSignalSpy spyEdited(&box, SIGNAL(textEdited(QString)));

    box.setFocus();
    QTest::keyClicks(&box, "dataObject");

    EXPECT_EQ(box.text(), QString("dataObject"));
    EXPECT_EQ(spyEdited.count(), 10);
}

TEST_F(SearchBoxTest, ClickOnClearIconClearsText)
{
    box.setText("some filter");
    QSignalSpy spyEdited(&box, SIGNAL(textEdited(QString)));

    QTest::mouseClick(&box, Qt::LeftButton, Qt::NoModifier, clearIconPos());

    EXPECT_TRUE(box.text().isEmpty());
    ASSERT_EQ(spyEdited.count(), 1);
    EXPECT_TRUE(spyEdited.at(0).at(0).toString().isEmpty());
}

TEST_F(SearchBoxTest, ClickInTheMiddleDoesNotClearText)
{
    box.setText("some filter");

    QTest::mouseClick(&box, Qt::LeftButton, Qt::NoModifier, box.rect().center());

    EXPECT_EQ(box.text(), QString("some filter"));
}

TEST_F(SearchBoxTest, ClickOnSearchIconSelectsAllIfIconIsShown)
{
    box.setShowSearchIcon(true);
    box.setText("select me");
    box.deselect();

    QTest::mouseClick(&box, Qt::LeftButton, Qt::NoModifier, searchIconPos());

    EXPECT_EQ(box.selectedText(), QString("select me"));
}

TEST_F(SearchBoxTest, ClickOnSearchIconAreaDoesNotSelectIfIconIsHidden)
{
    box.setShowSearchIcon(false);
    box.setText("select me");
    box.deselect();

    QTest::mouseClick(&box, Qt::LeftButton, Qt::NoModifier, searchIconPos());

    EXPECT_TRUE(box.selectedText().isEmpty());
}

TEST_F(SearchBoxTest, EscapeDoesNotModifyText)
{
    box.setText("keep");
    box.setFocus();

    QTest::keyClick(&box, Qt::Key_Escape);

    EXPECT_EQ(box.text(), QString("keep"));
}
