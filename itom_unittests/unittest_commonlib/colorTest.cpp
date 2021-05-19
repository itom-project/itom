#include "color.h"
#include "gtest/gtest.h"

TEST(ColorTest, Constructor)
{
    ito::Rgba32 zeros = ito::Rgba32::zeros();
    EXPECT_EQ(zeros.argb(), 0);

    ito::Rgba32 black = ito::Rgba32::black();
    EXPECT_EQ(black.r, 0);
    EXPECT_EQ(black.g, 0);
    EXPECT_EQ(black.b, 0);
    EXPECT_EQ(black.a, 255);

    ito::Rgba32 ful = ito::Rgba32::fromUnsignedLong((12 << 24) + (200 << 16) + (130 << 8) + (27 << 0)); // argb
    EXPECT_EQ(ful.r, 200);
    EXPECT_EQ(ful.g, 130);
    EXPECT_EQ(ful.b, 27);
    EXPECT_EQ(ful.a, 12);

    ito::Rgba32 rgba = ito::Rgba32(12, 200, 130, 27);
    EXPECT_EQ(rgba.r, 200);
    EXPECT_EQ(rgba.g, 130);
    EXPECT_EQ(rgba.b, 27);
    EXPECT_EQ(rgba.a, 12);

    ito::Rgba32 gray = ito::Rgba32(133); // from gray
    EXPECT_EQ(gray.r, 133);
    EXPECT_EQ(gray.g, 133);
    EXPECT_EQ(gray.b, 133);
    EXPECT_EQ(gray.a, 255);

    ito::Rgba32 rgba2(rgba); // copy constructor
    EXPECT_EQ(rgba2.r, 200);
    EXPECT_EQ(rgba2.g, 130);
    EXPECT_EQ(rgba2.b, 27);
    EXPECT_EQ(rgba2.a, 12);

    gray = rgba; // assignment
    EXPECT_EQ(gray.r, 200);
    EXPECT_EQ(gray.g, 130);
    EXPECT_EQ(gray.b, 27);
    EXPECT_EQ(gray.a, 12);

    gray = (33 << 24) + (55 << 16) + (77 << 8) + (99 << 0);
    EXPECT_EQ(gray.r, 55);
    EXPECT_EQ(gray.g, 77);
    EXPECT_EQ(gray.b, 99);
    EXPECT_EQ(gray.a, 33);
}

TEST(ColorTest, PlusMinusOperation)
{
    ito::Rgba32 val1 = ito::Rgba32(12, 20, 90, 100);
    ito::Rgba32 val2 = ito::Rgba32(20, 10, 1, 15);
    ito::Rgba32 val3 = ito::Rgba32(254, 254, 254, 254);

    ito::Rgba32 val1p2 = val1 + val2;
    ito::Rgba32 val1p3 = val1 + val3;

    ito::Rgba32 val3m1 = val3 - val1;
    ito::Rgba32 val2m3 = val2 - val3;

    EXPECT_EQ(val1p2.a, 32);
    EXPECT_EQ(val1p2.r, 30);
    EXPECT_EQ(val1p2.g, 91);
    EXPECT_EQ(val1p2.b, 115);

    EXPECT_EQ(val1p3.a, 255);
    EXPECT_EQ(val1p3.r, 255);
    EXPECT_EQ(val1p3.g, 255);
    EXPECT_EQ(val1p3.b, 255);

    EXPECT_EQ(val3m1.a, 242);
    EXPECT_EQ(val3m1.r, 234);
    EXPECT_EQ(val3m1.g, 164);
    EXPECT_EQ(val3m1.b, 154);

    EXPECT_EQ(val2m3.a, 0);
    EXPECT_EQ(val2m3.r, 0);
    EXPECT_EQ(val2m3.g, 0);
    EXPECT_EQ(val2m3.b, 0);

    val1 += val2;
    val3 -= val2;

    EXPECT_EQ(val1.a, 32);
    EXPECT_EQ(val1.r, 30);
    EXPECT_EQ(val1.g, 91);
    EXPECT_EQ(val1.b, 115);

    EXPECT_EQ(val3.a, 234);
    EXPECT_EQ(val3.r, 244);
    EXPECT_EQ(val3.g, 253);
    EXPECT_EQ(val3.b, 239);
}

TEST(ColorTest, MulDivOperation)
{
    ito::Rgba32 val1 = ito::Rgba32(12, 0, 255, 100);
    ito::Rgba32 val2 = ito::Rgba32(20, 0, 255, 15);
    ito::Rgba32 val3 = ito::Rgba32(254, 254, 254, 254);

    ito::Rgba32 val1m2 = val1 * val2;
    EXPECT_EQ(val1m2.a, 0);
    EXPECT_EQ(val1m2.r, 0);
    EXPECT_EQ(val1m2.g, 255);
    EXPECT_EQ(val1m2.b, 5);

    ito::Rgba32 val2d3 = val2 / val3;
    EXPECT_EQ(val2d3.a, 20);
    EXPECT_EQ(val2d3.r, 0);
    EXPECT_EQ(val2d3.g, 255);
    EXPECT_EQ(val2d3.b, 15);

    EXPECT_THROW(val3 / val2, std::runtime_error); // division by zero
}

TEST(ColorTest, GrayTest)
{
    ito::Rgba32 gray(130, 27, 100, 200);
    EXPECT_FLOAT_EQ(gray.gray(), 0.299 * 27 + 0.587 * 100 + 0.114 * 200);
}

TEST(ColorTest, Comparison)
{
    ito::Rgba32 gray(130, 27, 100, 200);
    ito::Rgba32 gray2(130, 27, 100, 200);

    EXPECT_TRUE(gray == gray2);
    EXPECT_FALSE(gray != gray2);
}

TEST(ColorTest, Access)
{
    ito::Rgba32 val(130, 27, 100, 200);
    EXPECT_EQ(val.alpha(), 130);
    EXPECT_EQ(val.a, 130);
    EXPECT_EQ(val.red(), 27);
    EXPECT_EQ(val.r, 27);
    EXPECT_EQ(val.green(), 100);
    EXPECT_EQ(val.g, 100);
    EXPECT_EQ(val.blue(), 200);
    EXPECT_EQ(val.b, 200);
    EXPECT_EQ(val.argb(), (130 << 24) + (27 << 16) + (100 << 8) + 200);
    EXPECT_EQ(val.u32ptr()[0], (130 << 24) + (27 << 16) + (100 << 8) + 200);
    EXPECT_EQ(200, val.u8ptr()[0]);

    val.a = 131;

    EXPECT_EQ(val.alpha(), 131);
    EXPECT_EQ(val.a, 131);
}
