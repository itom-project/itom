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

#include "doubleSpinBox.h"
#include "widgetTestHelpers.h"

#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QLocale>
#include <QSignalSpy>
#include <QtTest/QTest>

using namespace itomWidgetsTest;

namespace {

class DoubleSpinBoxTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // fixed decimal point, independent of the locale of the test machine
        box.spinBox()->setLocale(QLocale::c());
        box.setRange(-100.0, 100.0);
        box.setValue(0.0);
        ASSERT_TRUE(showAndWait(box, QSize(150, 30)));
        box.spinBox()->setFocus();
    }

    //! replaces the text of the editor by typing and confirms with Enter.
    void typeValue(const QString& text)
    {
        box.lineEdit()->selectAll();
        QTest::keyClicks(box.spinBox(), text);
        QTest::keyClick(box.spinBox(), Qt::Key_Enter);
    }

    DoubleSpinBox box;
};

} // namespace

//--------------------------------------------------------------------------------------------------
// state logic
//--------------------------------------------------------------------------------------------------

TEST(DoubleSpinBoxDefaults, DefaultState)
{
    DoubleSpinBox box;

    EXPECT_DOUBLE_EQ(box.value(), 0.0);
    EXPECT_DOUBLE_EQ(box.minimum(), 0.0);
    EXPECT_DOUBLE_EQ(box.maximum(), 99.99);
    EXPECT_DOUBLE_EQ(box.singleStep(), 1.0);
    EXPECT_EQ(box.decimals(), 2);
    EXPECT_EQ(box.setMode(), DoubleSpinBox::SetIfDifferent);
    EXPECT_EQ(
        box.decimalsOption(),
        DoubleSpinBox::DecimalsOptions(
            DoubleSpinBox::DecimalsByShortcuts | DoubleSpinBox::InsertDecimals));
    EXPECT_FALSE(box.invertedControls());
}

TEST_F(DoubleSpinBoxTest, SetValueIsClampedToRange)
{
    box.setValue(250.0);
    EXPECT_DOUBLE_EQ(box.value(), 100.0);

    box.setValue(-250.0);
    EXPECT_DOUBLE_EQ(box.value(), -100.0);
}

TEST_F(DoubleSpinBoxTest, ValueIsRoundedToDecimals)
{
    box.setDecimals(1);
    box.setValue(1.26);

    EXPECT_DOUBLE_EQ(box.displayedValue(), 1.3);
    EXPECT_DOUBLE_EQ(box.round(1.26), 1.3);
}

TEST_F(DoubleSpinBoxTest, PrefixAndSuffixAreOnlyPartOfText)
{
    box.setPrefix("x = ");
    box.setSuffix(" mm");
    box.setValue(12.5);

    EXPECT_EQ(box.text(), QString("x = 12.50 mm"));
    EXPECT_EQ(box.cleanText(), QString("12.50"));
}

TEST_F(DoubleSpinBoxTest, ValueChangedIsEmittedOnlyOnChangeInSetIfDifferentMode)
{
    QSignalSpy spy(&box, SIGNAL(valueChanged(double)));

    box.setValue(5.0);
    box.setValue(5.0);

    ASSERT_EQ(spy.count(), 1);
    EXPECT_DOUBLE_EQ(spy.at(0).at(0).toDouble(), 5.0);
}

TEST_F(DoubleSpinBoxTest, ShrinkingRangeClampsValue)
{
    box.setValue(80.0);
    box.setRange(0.0, 50.0);

    EXPECT_DOUBLE_EQ(box.value(), 50.0);
}

//--------------------------------------------------------------------------------------------------
// keyboard interaction
//--------------------------------------------------------------------------------------------------

TEST_F(DoubleSpinBoxTest, ArrowKeysStepTheValue)
{
    box.setSingleStep(0.5);

    QTest::keyClick(box.spinBox(), Qt::Key_Up);
    QTest::keyClick(box.spinBox(), Qt::Key_Up);
    EXPECT_DOUBLE_EQ(box.value(), 1.0);

    QTest::keyClick(box.spinBox(), Qt::Key_Down);
    EXPECT_DOUBLE_EQ(box.value(), 0.5);
}

TEST_F(DoubleSpinBoxTest, InvertedControlsReverseTheArrowKeys)
{
    box.setInvertedControls(true);

    QTest::keyClick(box.spinBox(), Qt::Key_Up);

    EXPECT_DOUBLE_EQ(box.value(), -1.0);
}

TEST_F(DoubleSpinBoxTest, SteppingStopsAtMaximum)
{
    box.setValue(99.5);

    QTest::keyClick(box.spinBox(), Qt::Key_Up);
    QTest::keyClick(box.spinBox(), Qt::Key_Up);

    EXPECT_DOUBLE_EQ(box.value(), 100.0);
}

TEST_F(DoubleSpinBoxTest, TypingAValue)
{
    QSignalSpy spyFinished(&box, SIGNAL(editingFinished()));

    typeValue("42.25");

    EXPECT_DOUBLE_EQ(box.value(), 42.25);
    EXPECT_GE(spyFinished.count(), 1);
}

TEST_F(DoubleSpinBoxTest, TypingAValueOutsideTheRangeIsRejected)
{
    box.setValue(3.0);

    typeValue("500");

    EXPECT_LE(box.value(), 100.0);
    EXPECT_GE(box.value(), -100.0);
}

TEST_F(DoubleSpinBoxTest, CtrlPlusAndMinusChangeDecimals)
{
    QSignalSpy spy(&box, SIGNAL(decimalsChanged(int)));

    QTest::keyClick(box.spinBox(), Qt::Key_Plus, Qt::ControlModifier);
    EXPECT_EQ(box.decimals(), 3);

    QTest::keyClick(box.spinBox(), Qt::Key_Minus, Qt::ControlModifier);
    QTest::keyClick(box.spinBox(), Qt::Key_Minus, Qt::ControlModifier);
    EXPECT_EQ(box.decimals(), 1);

    QTest::keyClick(box.spinBox(), Qt::Key_0, Qt::ControlModifier);
    EXPECT_EQ(box.decimals(), 2);

    EXPECT_EQ(spy.count(), 4);
}

TEST_F(DoubleSpinBoxTest, DecimalShortcutsCanBeDisabled)
{
    box.setDecimalsOption(DoubleSpinBox::FixedDecimals);

    QTest::keyClick(box.spinBox(), Qt::Key_Plus, Qt::ControlModifier);

    EXPECT_EQ(box.decimals(), 2);
}

TEST_F(DoubleSpinBoxTest, IncreasingDecimalsRestoresPrecision)
{
    box.setValue(1.2345);
    EXPECT_DOUBLE_EQ(box.displayedValue(), 1.23);

    QTest::keyClick(box.spinBox(), Qt::Key_Plus, Qt::ControlModifier);
    QTest::keyClick(box.spinBox(), Qt::Key_Plus, Qt::ControlModifier);

    EXPECT_DOUBLE_EQ(box.displayedValue(), 1.2345);
}
