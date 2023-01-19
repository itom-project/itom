/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2022, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#include "qVector2DProperty.h"
#include "qVector3DProperty.h"
#include "qVector4DProperty.h"
#include "autoIntervalProperty.h"
#include "interval.h"

#include <qmap.h>
#include <qvector2d.h>

Q_DECLARE_METATYPE(ito::AutoInterval)

#ifdef ENABLE_INTERNAL_TESTS

TEST(QPropertyEditorTest, CheckVector2DSetValue)
{
    QObject propObj;
    ito::QVector2DProperty vecProperty("name", &propObj, nullptr);

    ASSERT_EQ(vecProperty.x(), 0.0);
    ASSERT_EQ(vecProperty.y(), 0.0);
    
    vecProperty.setValue("-2.756;4.5e-7");

    auto data = vecProperty.value().value<QVector2D>();
    ASSERT_FLOAT_EQ(data.x(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.property("x").toDouble(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.x(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.y(), 4.5e-7);

    vecProperty.setValue("-20.756//40.5E-7//55");

    data = vecProperty.value().value<QVector2D>();
    ASSERT_FLOAT_EQ(data.x(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.property("x").toDouble(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.x(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.y(), 40.5e-7);

    vecProperty.value();
}

TEST(QPropertyEditorTest, CheckVector3DSetValue)
{
    QObject propObj;
    ito::QVector3DProperty vecProperty("name", &propObj, nullptr);

    ASSERT_EQ(vecProperty.x(), 0.0);
    ASSERT_EQ(vecProperty.y(), 0.0);
    ASSERT_EQ(vecProperty.z(), 0.0);

    vecProperty.setValue("-2.756;4.5e-7;-4.23e2");

    auto data = vecProperty.value().value<QVector3D>();
    ASSERT_FLOAT_EQ(data.x(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.property("x").toDouble(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.x(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.y(), 4.5e-7);
    ASSERT_FLOAT_EQ(vecProperty.z(), -4.23e2);

    vecProperty.setValue("-20.756//40.5E-7//55//-44");

    data = vecProperty.value().value<QVector3D>();
    ASSERT_FLOAT_EQ(data.x(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.property("x").toDouble(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.x(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.y(), 40.5e-7);
    ASSERT_FLOAT_EQ(vecProperty.z(), 55);

    vecProperty.value();
}

TEST(QPropertyEditorTest, CheckVector4DSetValue)
{
    QObject propObj;
    ito::QVector4DProperty vecProperty("name", &propObj, nullptr);

    ASSERT_EQ(vecProperty.x(), 0.0);
    ASSERT_EQ(vecProperty.y(), 0.0);
    ASSERT_EQ(vecProperty.z(), 0.0);
    ASSERT_EQ(vecProperty.w(), 0.0);

    vecProperty.setValue("-2.756;4.5e-7;-4.23e2;34");

    auto data = vecProperty.value().value<QVector4D>();
    ASSERT_FLOAT_EQ(data.x(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.property("x").toDouble(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.x(), -2.756);
    ASSERT_FLOAT_EQ(vecProperty.y(), 4.5e-7);
    ASSERT_FLOAT_EQ(vecProperty.z(), -4.23e2);
    ASSERT_FLOAT_EQ(vecProperty.w(), 34);

    vecProperty.setValue("-20.756//40.5E-7//55//-44e4//45");

    data = vecProperty.value().value<QVector4D>();
    ASSERT_FLOAT_EQ(data.x(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.property("x").toDouble(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.x(), -20.756);
    ASSERT_FLOAT_EQ(vecProperty.y(), 40.5e-7);
    ASSERT_FLOAT_EQ(vecProperty.z(), 55);
    ASSERT_FLOAT_EQ(vecProperty.w(), -44e4);

    vecProperty.value();
}

TEST(QPropertyEditorTest, CheckVector2DParseHints)
{
    QObject propObj;
    ito::QVector2DProperty vecProperty("name", &propObj, nullptr);

    vecProperty.setEditorHints("minimumX=-2;maximumX=2.5;minimumY=-7.;maximumY= 23");
}

TEST(QPropertyEditorTest, CheckVector3DParseHints)
{
    QObject propObj;
    ito::QVector3DProperty vecProperty("name", &propObj, nullptr);

    vecProperty.setEditorHints("minimumX=-2;maximumX=2.5;minimumY=-7.;maximumY= 23;minimumZ=4;maximumZ=7");
}

TEST(QPropertyEditorTest, CheckVector4DParseHints)
{
    QObject propObj;
    ito::QVector4DProperty vecProperty("name", &propObj, nullptr);

    vecProperty.setEditorHints("minimumX=-2;maximumX=2.5;minimumY=-7.;maximumY= 23");
}

TEST(QPropertyEditorTest, CheckAutoIntervalSetValue)
{
    QObject propObj;
    auto interval = ito::AutoInterval(-5.5, 3.2, false);
    propObj.setProperty("name", QVariant::fromValue(interval));
    ito::AutoIntervalProperty aiProperty("name", &propObj, nullptr);

    ASSERT_EQ(aiProperty.minimum(), interval.minimum() );
    ASSERT_EQ(aiProperty.maximum(), interval.maximum() );
    ASSERT_EQ(aiProperty.autoScaling(), interval.isAuto());

    aiProperty.setValue("<auto>");
    ASSERT_EQ(aiProperty.minimum(), interval.minimum());
    ASSERT_EQ(aiProperty.maximum(), interval.maximum());
    ASSERT_EQ(aiProperty.autoScaling(), true);

    aiProperty.setValue("-6.5e2//3.245");
    ASSERT_EQ(aiProperty.minimum(), -6.5e2);
    ASSERT_EQ(aiProperty.maximum(), 3.245);
    ASSERT_EQ(aiProperty.autoScaling(), false);

    interval = ito::AutoInterval(-5.52, 3.25, true);
    aiProperty.setValue(QVariant::fromValue(interval));
    ASSERT_EQ(aiProperty.minimum(), interval.minimum());
    ASSERT_EQ(aiProperty.maximum(), interval.maximum());
    ASSERT_EQ(aiProperty.autoScaling(), interval.isAuto());

    aiProperty.value();
}

#endif
