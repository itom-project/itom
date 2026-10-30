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


#include "helper/lineEnding.h"
#include "gtest/gtest.h"

using namespace ito;

//-------------------------------------------------------------------------------------
TEST(LineEndingTest, detect)
{
    EXPECT_EQ(LineEnding::detect("a\r\nb\r\nc\r\n"), LineEnding::CRLF);
    EXPECT_EQ(LineEnding::detect("a\nb\nc\n"), LineEnding::LF);
    EXPECT_EQ(LineEnding::detect("a\r\nb\r\n"), LineEnding::CRLF); // no trailing newline
    EXPECT_EQ(LineEnding::detect("a\nb"), LineEnding::LF);

    // mixed: majority wins
    EXPECT_EQ(LineEnding::detect("a\r\nb\r\nc\nd\r\n"), LineEnding::CRLF);
    EXPECT_EQ(LineEnding::detect("a\nb\nc\r\nd\n"), LineEnding::LF);

    // no newline: platform default
    EXPECT_EQ(LineEnding::detect("single line"), LineEnding::platformDefault());
    EXPECT_EQ(LineEnding::detect(""), LineEnding::platformDefault());

#ifdef Q_OS_WIN
    EXPECT_EQ(LineEnding::platformDefault(), LineEnding::CRLF);
#else
    EXPECT_EQ(LineEnding::platformDefault(), LineEnding::LF);
#endif
}

//-------------------------------------------------------------------------------------
TEST(LineEndingTest, normalizeAndApply)
{
    QString text = "a\r\nb\nc\r\n";
    LineEnding::normalize(text);
    EXPECT_EQ(text, "a\nb\nc\n");

    EXPECT_EQ(LineEnding::apply("a\nb\n", LineEnding::CRLF), "a\r\nb\r\n");
    EXPECT_EQ(LineEnding::apply("a\nb\n", LineEnding::LF), "a\nb\n");
    EXPECT_EQ(LineEnding::apply("", LineEnding::CRLF), "");

    // round trip keeps the file identical
    QString file = "x\r\ny\r\n\r\nz";
    QString t = file;
    LineEnding::Type type = LineEnding::detect(t);
    LineEnding::normalize(t);
    EXPECT_EQ(LineEnding::apply(t, type), file);

    EXPECT_EQ(LineEnding::shortName(LineEnding::LF), "LF");
    EXPECT_EQ(LineEnding::shortName(LineEnding::CRLF), "CRLF");
}
