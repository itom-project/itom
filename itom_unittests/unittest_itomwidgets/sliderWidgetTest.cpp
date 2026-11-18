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

#include "doubleSlider.h"
#include "doubleSpinBox.h"
#include "sliderWidget.h"
#include "widgetTestHelpers.h"

#include <QDoubleSpinBox>
#include <QLocale>
#include <QSignalSpy>
#include <QSlider>
#include <QStyle>
#include <QStyleOptionSlider>
#include <QHBoxLayout>
#include <QtTest/QTest>

using namespace itomWidgetsTest;

namespace {

//! returns the center of the handle of a horizontal QSlider at its current position.
QPoint sliderHandleCenter(const QSlider* slider)
{
    QStyleOptionSlider option;
    option.initFrom(slider);
    option.subControls = QStyle::SC_None;
    option.activeSubControls = QStyle::SC_None;
    option.orientation = slider->orientation();
    option.maximum = slider->maximum();
    option.minimum = slider->minimum();
    option.tickPosition = slider->tickPosition();
    option.tickInterval = slider->tickInterval();
    option.upsideDown = slider->invertedAppearance();
    option.direction = slider->layoutDirection();
    option.sliderPosition = slider->sliderPosition();
    option.sliderValue = slider->value();
    option.singleStep = slider->singleStep();
    option.pageStep = slider->pageStep();
    return slider->style()
        ->subControlRect(QStyle::CC_Slider, &option, QStyle::SC_SliderHandle, slider)
        .center();
}

//! fixture: a SliderWidget inside a parent container (the SliderWidget requires a parent).
class SliderWidgetTest : public ::testing::Test
{
protected:
    SliderWidgetTest() : widget(*new SliderWidget(&container))
    {
        auto* layout = new QHBoxLayout(&container);
        layout->addWidget(&widget);
    }

    void SetUp() override
    {
        widget.spinBox()->spinBox()->setLocale(QLocale::c());
        widget.setRange(0.0, 10.0);
        widget.setSingleStep(0.1);
        widget.setValue(0.0);
        ASSERT_TRUE(showAndWait(container, QSize(400, 30)));

        innerSlider = widget.slider()->findChild<QSlider*>();
        ASSERT_NE(innerSlider, nullptr);
    }

    QWidget container;
    SliderWidget& widget; // owned by container
    QSlider* innerSlider = nullptr;
};

} // namespace

// Regression test: SliderWidgetPrivate::synchronizeSiblingWidth() and
// synchronizeSiblingDecimals() formerly dereferenced q->parent() without a nullptr check,
// such that a SliderWidget without parent crashed when its range or decimals changed.
TEST(SliderWidgetWithoutParent, SetRangeDoesNotCrash)
{
    SliderWidget widget;
    widget.setRange(0.0, 10.0);
    widget.setDecimals(3);

    EXPECT_DOUBLE_EQ(widget.maximum(), 10.0);
    EXPECT_EQ(widget.decimals(), 3);
}

TEST(SliderWidgetSiblings, DecimalsAreSynchronizedBetweenSiblings)
{
    QWidget container;
    auto* first = new SliderWidget(&container);
    auto* second = new SliderWidget(&container);
    auto* independent = new SliderWidget(&container);

    first->setSynchronizeSiblings(SliderWidget::SynchronizeDecimals);
    second->setSynchronizeSiblings(SliderWidget::SynchronizeDecimals);
    independent->setSynchronizeSiblings(SliderWidget::NoSynchronize);
    const int independentDecimals = independent->decimals();

    first->setDecimals(4);

    EXPECT_EQ(second->decimals(), 4);
    EXPECT_EQ(independent->decimals(), independentDecimals);
}

TEST_F(SliderWidgetTest, SetValueSynchronizesSliderAndSpinBox)
{
    widget.setValue(3.7);

    EXPECT_DOUBLE_EQ(widget.value(), 3.7);
    EXPECT_DOUBLE_EQ(widget.slider()->value(), 3.7);
    EXPECT_DOUBLE_EQ(widget.spinBox()->value(), 3.7);
}

TEST_F(SliderWidgetTest, SpinBoxChangesUpdateTheSlider)
{
    widget.spinBox()->setValue(8.2);

    EXPECT_DOUBLE_EQ(widget.slider()->value(), 8.2);
    EXPECT_DOUBLE_EQ(widget.value(), 8.2);
}

TEST_F(SliderWidgetTest, SliderChangesUpdateTheSpinBox)
{
    widget.slider()->setValue(1.5);

    EXPECT_DOUBLE_EQ(widget.spinBox()->value(), 1.5);
    EXPECT_DOUBLE_EQ(widget.value(), 1.5);
}

TEST_F(SliderWidgetTest, ValueChangedIsEmittedOncePerChange)
{
    QSignalSpy spy(&widget, SIGNAL(valueChanged(double)));

    widget.setValue(4.0);
    widget.setValue(4.0);

    ASSERT_EQ(spy.count(), 1);
    EXPECT_DOUBLE_EQ(spy.at(0).at(0).toDouble(), 4.0);
}

TEST_F(SliderWidgetTest, RangeIsAppliedToBothChildren)
{
    widget.setRange(-5.0, 5.0);
    widget.setValue(9.0);

    EXPECT_DOUBLE_EQ(widget.value(), 5.0);
    EXPECT_DOUBLE_EQ(widget.slider()->maximum(), 5.0);
    EXPECT_DOUBLE_EQ(widget.spinBox()->maximum(), 5.0);
    EXPECT_DOUBLE_EQ(widget.slider()->minimum(), -5.0);
    EXPECT_DOUBLE_EQ(widget.spinBox()->minimum(), -5.0);
}

TEST_F(SliderWidgetTest, ResetSetsValueToZero)
{
    widget.setValue(6.0);
    widget.reset();

    EXPECT_DOUBLE_EQ(widget.value(), 0.0);
}

TEST_F(SliderWidgetTest, SpinBoxCanBeHidden)
{
    widget.setSpinBoxVisible(false);

    EXPECT_FALSE(widget.isSpinBoxVisible());
    EXPECT_FALSE(widget.spinBox()->isVisible());
    EXPECT_TRUE(widget.slider()->isVisible());
}

TEST_F(SliderWidgetTest, KeyboardOnSliderUpdatesSpinBox)
{
    innerSlider->setFocus();

    QTest::keyClick(innerSlider, Qt::Key_Right);
    QTest::keyClick(innerSlider, Qt::Key_Right);

    EXPECT_NEAR(widget.value(), 0.2, 1e-9);
    EXPECT_NEAR(widget.spinBox()->value(), 0.2, 1e-9);
}

TEST_F(SliderWidgetTest, DraggingWithTrackingEmitsValueChangedContinuously)
{
    widget.setTracking(true);
    QSignalSpy spyChanged(&widget, SIGNAL(valueChanged(double)));

    const QPoint start = sliderHandleCenter(innerSlider);
    dragMouse(innerSlider, start, QPoint(innerSlider->width() / 2, start.y()), 10);

    EXPECT_GT(widget.value(), 3.0);
    EXPECT_LT(widget.value(), 7.0);
    EXPECT_DOUBLE_EQ(widget.spinBox()->value(), widget.value());
    EXPECT_GT(spyChanged.count(), 1);
}

TEST_F(SliderWidgetTest, DraggingWithoutTrackingEmitsValueChangedOnRelease)
{
    widget.setTracking(false);
    QSignalSpy spyChanged(&widget, SIGNAL(valueChanged(double)));
    QSignalSpy spyChanging(&widget, SIGNAL(valueIsChanging(double)));

    const QPoint start = sliderHandleCenter(innerSlider);
    dragMouse(innerSlider, start, QPoint(innerSlider->width() / 2, start.y()), 10);

    EXPECT_GT(widget.value(), 3.0);
    EXPECT_GT(spyChanging.count(), 1);
    ASSERT_EQ(spyChanged.count(), 1);
    EXPECT_DOUBLE_EQ(spyChanged.at(0).at(0).toDouble(), widget.value());
}
