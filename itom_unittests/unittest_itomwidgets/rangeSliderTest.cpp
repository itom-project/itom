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

#include "rangeSlider.h"
#include "widgetTestHelpers.h"

#include <QSignalSpy>
#include <QStyle>
#include <QStyleOptionSlider>
#include <QtTest/QTest>

using namespace itomWidgetsTest;

namespace {

//! exposes the protected style option of RangeSlider to compute handle positions.
class RangeSliderProbe : public RangeSlider
{
public:
    explicit RangeSliderProbe(QWidget* parent = nullptr) :
        RangeSlider(Qt::Horizontal, parent)
    {
    }

    //! returns the center of the handle, if it is located at the given slider value.
    QPoint handleCenter(int value) const
    {
        QStyleOptionSlider option;
        initStyleOption(&option);
        option.sliderPosition = value;
        option.sliderValue = value;
        return style()
            ->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, this)
            .center();
    }
};

//! fixture: a shown horizontal range slider with the range [0, 100].
class RangeSliderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        slider.setRange(0, 100);
        slider.setMaximumRange(101);
        slider.setValues(0, 100);
    }

    RangeSliderProbe slider;
};

} // namespace

//--------------------------------------------------------------------------------------------------
// state and value logic
//--------------------------------------------------------------------------------------------------

TEST(RangeSliderDefaults, DefaultConstructedState)
{
    RangeSlider slider;

    // QSlider default range is [0, 99]
    EXPECT_EQ(slider.minimum(), 0);
    EXPECT_EQ(slider.maximum(), 99);
    EXPECT_EQ(slider.minimumValue(), 0);
    EXPECT_EQ(slider.maximumValue(), 99);
    EXPECT_EQ(slider.minimumPosition(), 0);
    EXPECT_EQ(slider.maximumPosition(), 99);
    EXPECT_EQ(slider.stepSizePosition(), 1u);
    EXPECT_EQ(slider.minimumRange(), 0u);
    EXPECT_EQ(slider.maximumRange(), 99u);
    EXPECT_FALSE(slider.symmetricMoves());
    EXPECT_FALSE(slider.rangeIncludeLimits());
    EXPECT_FALSE(slider.isMinimumSliderDown());
    EXPECT_FALSE(slider.isMaximumSliderDown());
}

TEST(RangeSliderDefaults, MaximumRangeIsNotUpdatedWhenRangeIsEnlarged)
{
    // Characterization test: the maximum range is computed once from the initial
    // QSlider range [0, 99]. Enlarging the slider range afterwards does not enlarge
    // the maximum range, such that the interval stays limited to a width of 99.
    // Callers have to call setMaximumRange() explicitly after setRange().
    RangeSlider slider;
    slider.setRange(0, 1000);
    slider.setValues(0, 1000);

    EXPECT_EQ(slider.maximumRange(), 99u);
    EXPECT_EQ(slider.maximumValue() - slider.minimumValue(), 99);
}

TEST_F(RangeSliderTest, SetValuesStoresBothValues)
{
    slider.setValues(10, 60);

    EXPECT_EQ(slider.minimumValue(), 10);
    EXPECT_EQ(slider.maximumValue(), 60);
    EXPECT_EQ(slider.minimumPosition(), 10);
    EXPECT_EQ(slider.maximumPosition(), 60);
}

TEST_F(RangeSliderTest, SetValuesSwapsReversedArguments)
{
    slider.setValues(70, 20);

    EXPECT_EQ(slider.minimumValue(), 20);
    EXPECT_EQ(slider.maximumValue(), 70);
}

TEST_F(RangeSliderTest, SetValuesIsClampedToSliderRange)
{
    slider.setValues(-50, 500);

    EXPECT_EQ(slider.minimumValue(), 0);
    EXPECT_EQ(slider.maximumValue(), 100);
}

TEST_F(RangeSliderTest, SetMinimumValueAboveMaximumPushesMaximum)
{
    slider.setValues(10, 30);
    slider.setMinimumValue(50);

    EXPECT_EQ(slider.minimumValue(), 50);
    EXPECT_EQ(slider.maximumValue(), 50);
}

TEST_F(RangeSliderTest, SetMaximumValueBelowMinimumPushesMinimum)
{
    slider.setValues(40, 80);
    slider.setMaximumValue(20);

    EXPECT_EQ(slider.minimumValue(), 20);
    EXPECT_EQ(slider.maximumValue(), 20);
}

TEST_F(RangeSliderTest, SetValuesEmitsSignalsOnlyOnChange)
{
    QSignalSpy spyValues(&slider, SIGNAL(valuesChanged(int, int)));
    QSignalSpy spyMin(&slider, SIGNAL(minimumValueChanged(int)));
    QSignalSpy spyMax(&slider, SIGNAL(maximumValueChanged(int)));

    slider.setValues(10, 100); // only the minimum changes

    ASSERT_EQ(spyValues.count(), 1);
    EXPECT_EQ(spyValues.at(0).at(0).toInt(), 10);
    EXPECT_EQ(spyValues.at(0).at(1).toInt(), 100);
    EXPECT_EQ(spyMin.count(), 1);
    EXPECT_EQ(spyMax.count(), 0);

    slider.setValues(10, 100); // no change at all

    EXPECT_EQ(spyValues.count(), 1);
    EXPECT_EQ(spyMin.count(), 1);
    EXPECT_EQ(spyMax.count(), 0);
}

TEST_F(RangeSliderTest, MinimumRangeIsEnforced)
{
    slider.setMinimumRange(20);
    slider.setValues(50, 55);

    EXPECT_GE(slider.maximumValue() - slider.minimumValue(), 20);
    EXPECT_GE(slider.minimumValue(), slider.minimum());
    EXPECT_LE(slider.maximumValue(), slider.maximum());
}

TEST_F(RangeSliderTest, MaximumRangeIsEnforced)
{
    slider.setMaximumRange(30);
    slider.setValues(10, 90);

    EXPECT_LE(slider.maximumValue() - slider.minimumValue(), 30);
}

TEST_F(RangeSliderTest, PositionStepSizeSnapsValues)
{
    slider.setStepSizePosition(10);
    slider.setValues(13, 67);

    EXPECT_EQ(slider.minimumValue() % 10, 0);
    EXPECT_EQ(slider.maximumValue() % 10, 0);
}

TEST_F(RangeSliderTest, ShrinkingSliderRangeClampsValues)
{
    slider.setValues(10, 90);
    slider.setRange(20, 50);

    EXPECT_GE(slider.minimumValue(), 20);
    EXPECT_LE(slider.maximumValue(), 50);
}

//--------------------------------------------------------------------------------------------------
// keyboard interaction
//--------------------------------------------------------------------------------------------------

TEST_F(RangeSliderTest, KeyRightMovesMinimumHandle)
{
    ASSERT_TRUE(showAndWait(slider));
    slider.setValues(10, 50);

    QTest::keyClick(&slider, Qt::Key_Right);

    EXPECT_EQ(slider.minimumValue(), 11);
    EXPECT_EQ(slider.maximumValue(), 50);
}

TEST_F(RangeSliderTest, KeyLeftMovesMinimumHandle)
{
    ASSERT_TRUE(showAndWait(slider));
    slider.setValues(10, 50);

    QTest::keyClick(&slider, Qt::Key_Left);

    EXPECT_EQ(slider.minimumValue(), 9);
    EXPECT_EQ(slider.maximumValue(), 50);
}

TEST_F(RangeSliderTest, KeyUpAndDownMoveMaximumHandle)
{
    ASSERT_TRUE(showAndWait(slider));
    slider.setValues(10, 50);

    QTest::keyClick(&slider, Qt::Key_Up);
    EXPECT_EQ(slider.maximumValue(), 51);

    QTest::keyClick(&slider, Qt::Key_Down);
    QTest::keyClick(&slider, Qt::Key_Down);
    EXPECT_EQ(slider.maximumValue(), 49);
    EXPECT_EQ(slider.minimumValue(), 10);
}

TEST_F(RangeSliderTest, ShiftKeyMovesWholeInterval)
{
    ASSERT_TRUE(showAndWait(slider));
    slider.setStepSizePosition(5);
    slider.setValues(20, 40);

    QTest::keyClick(&slider, Qt::Key_Right, Qt::ShiftModifier);
    EXPECT_EQ(slider.minimumValue(), 25);
    EXPECT_EQ(slider.maximumValue(), 45);

    QTest::keyClick(&slider, Qt::Key_Left, Qt::ShiftModifier);
    QTest::keyClick(&slider, Qt::Key_Left, Qt::ShiftModifier);
    EXPECT_EQ(slider.minimumValue(), 15);
    EXPECT_EQ(slider.maximumValue(), 35);
}

TEST_F(RangeSliderTest, KeysDoNotLeaveSliderRange)
{
    ASSERT_TRUE(showAndWait(slider));
    slider.setValues(0, 100);

    QTest::keyClick(&slider, Qt::Key_Left);
    QTest::keyClick(&slider, Qt::Key_Up);
    QTest::keyClick(&slider, Qt::Key_Left, Qt::ShiftModifier);
    QTest::keyClick(&slider, Qt::Key_Right, Qt::ShiftModifier);

    EXPECT_EQ(slider.minimumValue(), 0);
    EXPECT_EQ(slider.maximumValue(), 100);
}

//--------------------------------------------------------------------------------------------------
// mouse interaction
//--------------------------------------------------------------------------------------------------

TEST_F(RangeSliderTest, DragMaximumHandleToTheRight)
{
    ASSERT_TRUE(showAndWait(slider, QSize(400, 40)));
    slider.setValues(20, 50);

    QSignalSpy spyValues(&slider, SIGNAL(valuesChanged(int, int)));

    const QPoint start = slider.handleCenter(50);
    dragMouse(&slider, start, QPoint(slider.width() - 2, start.y()));

    EXPECT_EQ(slider.minimumValue(), 20);
    EXPECT_GT(slider.maximumValue(), 90);
    EXPECT_LE(slider.maximumValue(), 100);
    EXPECT_GE(spyValues.count(), 1);
    EXPECT_FALSE(slider.isSliderDown());
}

TEST_F(RangeSliderTest, DragMinimumHandleToTheLeft)
{
    ASSERT_TRUE(showAndWait(slider, QSize(400, 40)));
    slider.setValues(50, 80);

    const QPoint start = slider.handleCenter(50);
    dragMouse(&slider, start, QPoint(1, start.y()));

    EXPECT_LT(slider.minimumValue(), 10);
    EXPECT_GE(slider.minimumValue(), 0);
    EXPECT_EQ(slider.maximumValue(), 80);
}

TEST_F(RangeSliderTest, MinimumHandleCannotBeDraggedBeyondMaximumHandle)
{
    ASSERT_TRUE(showAndWait(slider, QSize(400, 40)));
    slider.setValues(20, 50);

    const QPoint start = slider.handleCenter(20);
    dragMouse(&slider, start, QPoint(slider.width() - 2, start.y()));

    EXPECT_LE(slider.minimumValue(), slider.maximumValue());
    EXPECT_EQ(slider.maximumValue(), 50);
}

TEST_F(RangeSliderTest, DragBetweenHandlesMovesWholeInterval)
{
    ASSERT_TRUE(showAndWait(slider, QSize(400, 40)));
    slider.setValues(30, 60);

    const QPoint left = slider.handleCenter(30);
    const QPoint right = slider.handleCenter(60);
    const QPoint start((left.x() + right.x()) / 2, left.y());
    const int deltaPixels = right.x() - left.x(); // approx. 30 values

    dragMouse(&slider, start, QPoint(start.x() + deltaPixels / 3, start.y()));

    // the width of the interval stays constant and the interval moved to the right
    EXPECT_EQ(slider.maximumValue() - slider.minimumValue(), 30);
    EXPECT_GT(slider.minimumValue(), 30);
}

TEST_F(RangeSliderTest, SymmetricMovesMirrorTheOtherHandle)
{
    ASSERT_TRUE(showAndWait(slider, QSize(400, 40)));
    slider.setSymmetricMoves(true);
    slider.setValues(40, 60);

    const QPoint start = slider.handleCenter(60);
    const QPoint target = slider.handleCenter(70);
    dragMouse(&slider, start, target);

    // the maximum handle moved to the right, the minimum handle by the same amount to the left
    const int deltaMax = slider.maximumValue() - 60;
    EXPECT_GT(deltaMax, 0);
    EXPECT_EQ(slider.minimumValue(), 40 - deltaMax);
}

TEST_F(RangeSliderTest, MouseIsIgnoredIfSliderRangeIsEmpty)
{
    ASSERT_TRUE(showAndWait(slider, QSize(400, 40)));
    slider.setRange(5, 5);

    const QPoint start = slider.handleCenter(5);
    dragMouse(&slider, start, QPoint(slider.width() - 2, start.y()));

    EXPECT_EQ(slider.minimumValue(), 5);
    EXPECT_EQ(slider.maximumValue(), 5);
}
