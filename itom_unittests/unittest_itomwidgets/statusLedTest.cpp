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

#include "statusLed.h"
#include "widgetTestHelpers.h"

#include <QColor>
#include <QImage>
#include <QPixmap>

using namespace itomWidgetsTest;

namespace {

class StatusLedTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_TRUE(showAndWait(led, QSize(64, 64)));
    }

    //! renders the led and returns a pixel between center and edge of the circle.
    QColor sampleOuterRing()
    {
        const QImage image = led.grab().toImage();
        const QPoint center = led.rect().center();
        const int radius = qMin(led.width(), led.height()) / 2 - 2;
        return image.pixelColor(center + QPoint(radius * 3 / 4, 0));
    }

    StatusLed led;
};

} // namespace

TEST_F(StatusLedTest, DefaultState)
{
    EXPECT_FALSE(led.checked());
    EXPECT_EQ(led.colorOnEdge(), QColor(Qt::green));
    EXPECT_EQ(led.colorOffEdge(), QColor(Qt::red));
    EXPECT_EQ(led.colorOnCenter(), QColor(Qt::white));
    EXPECT_EQ(led.colorOffCenter(), QColor(Qt::white));
    EXPECT_EQ(led.sizeHint(), QSize(32, 32));
    EXPECT_EQ(led.heightForWidth(40), 40);
}

TEST_F(StatusLedTest, SettersAndProperties)
{
    led.setChecked(true);
    led.setColorOnEdge(Qt::blue);
    led.setColorOffCenter(Qt::yellow);

    EXPECT_TRUE(led.checked());
    EXPECT_TRUE(led.property("checked").toBool());
    EXPECT_EQ(led.property("colorOnEdge").value<QColor>(), QColor(Qt::blue));
    EXPECT_EQ(led.colorOffCenter(), QColor(Qt::yellow));
}

TEST_F(StatusLedTest, RendersGreenWhenOn)
{
    led.setChecked(true);
    const QColor c = sampleOuterRing();

    EXPECT_GT(c.green(), c.red() + 50) << c.name().toStdString();
    EXPECT_GT(c.green(), c.blue() + 50) << c.name().toStdString();
}

TEST_F(StatusLedTest, RendersRedWhenOff)
{
    led.setChecked(false);
    const QColor c = sampleOuterRing();

    EXPECT_GT(c.red(), c.green() + 50) << c.name().toStdString();
    EXPECT_GT(c.red(), c.blue() + 50) << c.name().toStdString();
}

TEST_F(StatusLedTest, RendersCustomOnColor)
{
    led.setColorOnEdge(Qt::blue);
    led.setChecked(true);
    const QColor c = sampleOuterRing();

    EXPECT_GT(c.blue(), c.red() + 50) << c.name().toStdString();
    EXPECT_GT(c.blue(), c.green() + 50) << c.name().toStdString();
}

TEST_F(StatusLedTest, RendersGrayWhenDisabled)
{
    led.setChecked(true);
    led.setEnabled(false);
    const QColor c = sampleOuterRing();

    EXPECT_NEAR(c.red(), c.green(), 10) << c.name().toStdString();
    EXPECT_NEAR(c.green(), c.blue(), 10) << c.name().toStdString();
}
