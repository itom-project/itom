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

#include <qmap.h>
#include <qvector2d.h>

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
