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

#include "collapsibleGroupBox.h"
#include "widgetTestHelpers.h"

#include <QLabel>
#include <QPushButton>
#include <QSignalSpy>
#include <QStyle>
#include <QStyleOptionGroupBox>
#include <QVBoxLayout>
#include <QtTest/QTest>

using namespace itomWidgetsTest;

namespace {

//! exposes the protected style option to compute the position of the checkbox indicator.
class CollapsibleGroupBoxProbe : public CollapsibleGroupBox
{
public:
    explicit CollapsibleGroupBoxProbe(const QString& title) : CollapsibleGroupBox(title)
    {
    }

    QPoint indicatorCenter() const
    {
        QStyleOptionGroupBox option;
        initStyleOption(&option);
        return style()
            ->subControlRect(QStyle::CC_GroupBox, &option, QStyle::SC_GroupBoxCheckBox, this)
            .center();
    }
};

class CollapsibleGroupBoxTest : public ::testing::Test
{
protected:
    CollapsibleGroupBoxTest() : box("Settings")
    {
        auto* layout = new QVBoxLayout(&box);
        label = new QLabel("content", &box);
        button = new QPushButton("action", &box);
        layout->addWidget(label);
        layout->addWidget(button);
    }

    CollapsibleGroupBoxProbe box;
    QLabel* label;
    QPushButton* button;
};

} // namespace

TEST_F(CollapsibleGroupBoxTest, DefaultIsExpandedAndCheckable)
{
    EXPECT_TRUE(box.isCheckable());
    EXPECT_TRUE(box.isChecked());
    EXPECT_FALSE(box.collapsed());
    EXPECT_EQ(box.collapsedHeight(), 14);
}

TEST_F(CollapsibleGroupBoxTest, CollapseHidesChildren)
{
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));
    ASSERT_TRUE(label->isVisible());

    box.setCollapsed(true);

    EXPECT_TRUE(box.collapsed());
    EXPECT_FALSE(box.isChecked());
    EXPECT_FALSE(label->isVisible());
    EXPECT_FALSE(button->isVisible());
}

TEST_F(CollapsibleGroupBoxTest, CollapseReducesMaximumHeight)
{
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));
    const int expandedMaxHeight = box.maximumHeight();

    box.setCollapsed(true);
    EXPECT_LT(box.maximumHeight(), 150);

    box.setCollapsed(false);
    EXPECT_EQ(box.maximumHeight(), expandedMaxHeight);
}

TEST_F(CollapsibleGroupBoxTest, ExpandRestoresChildren)
{
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));

    box.setCollapsed(true);
    box.setCollapsed(false);

    EXPECT_FALSE(box.collapsed());
    EXPECT_TRUE(label->isVisible());
    EXPECT_TRUE(button->isVisible());
}

TEST_F(CollapsibleGroupBoxTest, ExplicitlyHiddenChildStaysHiddenAfterExpand)
{
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));

    button->hide();
    box.setCollapsed(true);
    box.setCollapsed(false);

    EXPECT_TRUE(label->isVisible());
    EXPECT_FALSE(button->isVisible());
}

TEST_F(CollapsibleGroupBoxTest, CollapsedPropertyViaMetaObject)
{
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));

    ASSERT_TRUE(box.setProperty("collapsed", true));
    EXPECT_TRUE(box.property("collapsed").toBool());
    EXPECT_FALSE(label->isVisible());

    ASSERT_TRUE(box.setProperty("collapsedHeight", 30));
    EXPECT_EQ(box.collapsedHeight(), 30);
}

TEST_F(CollapsibleGroupBoxTest, ClickOnIndicatorTogglesCollapsed)
{
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));
    QSignalSpy spy(&box, SIGNAL(toggled(bool)));

    QTest::mouseClick(&box, Qt::LeftButton, Qt::NoModifier, box.indicatorCenter());

    EXPECT_TRUE(box.collapsed());
    EXPECT_FALSE(label->isVisible());
    ASSERT_EQ(spy.count(), 1);
    EXPECT_FALSE(spy.at(0).at(0).toBool());

    QTest::mouseClick(&box, Qt::LeftButton, Qt::NoModifier, box.indicatorCenter());

    EXPECT_FALSE(box.collapsed());
    EXPECT_TRUE(label->isVisible());
}

TEST_F(CollapsibleGroupBoxTest, CollapsingBeforeShowAppliesOnShow)
{
    box.setCollapsed(true);
    ASSERT_TRUE(showAndWait(box, QSize(200, 150)));

    EXPECT_TRUE(box.collapsed());
    EXPECT_FALSE(label->isVisible());
}
