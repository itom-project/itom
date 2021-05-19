#include <iostream>

#include "../../common/color.h"
#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "../../common/numeric.h"
#include "commonChannel.h"

/*! \class
    \brief
*/
template <typename _Tp> class rgbaOperatorTests : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        this->valWhite = _Tp(255);
        this->valBLACKTransparent = _Tp::zeros();
        this->valRED = ito::Rgba32(255, 255, 0, 0);
        this->valBLUE = ito::Rgba32(255, 0, 0, 255);
        this->valGREEN = ito::Rgba32(255, 0, 255, 0);
        this->valBLACK = _Tp::black();
        this->valOneGray = _Tp(1);
        this->valDarkGray = _Tp(64);
        this->valGray = _Tp(128);
    };

    virtual void TearDown(void){};

    _Tp valWhite;
    _Tp valBLACKTransparent;
    ito::Rgba32 valRED;
    ito::Rgba32 valBLUE;
    ito::Rgba32 valGREEN;
    _Tp valBLACK;
    _Tp valOneGray;
    _Tp valDarkGray;
    _Tp valGray;
};

TYPED_TEST_CASE(rgbaOperatorTests, ItomColorAllTypes);

/*!

*/
TYPED_TEST(rgbaOperatorTests, testMultiplication)
{
    TypeParam tempVal =
        this->valDarkGray * this->valBLACKTransparent; // Multiplication off zero with white must be zeros;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);

    // Multiplication off white with zero must be zeros;
    tempVal = this->valBLACKTransparent * this->valDarkGray;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);
    tempVal = this->valBLACKTransparent;
    tempVal *= this->valDarkGray;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);

    // Multiplication off ones with white must be ones;
    tempVal = this->valDarkGray * this->valWhite;
    EXPECT_EQ(tempVal, this->valDarkGray);
    tempVal = this->valDarkGray;
    tempVal *= this->valWhite;
    EXPECT_EQ(tempVal, this->valDarkGray);

    // Multiplication off gray128 with white must be gray128;
    tempVal = this->valGray * this->valWhite;
    EXPECT_EQ(tempVal, this->valGray);
    tempVal = this->valGray;
    tempVal *= this->valWhite;
    EXPECT_EQ(tempVal, this->valGray);

    // Multiplication off white with white must be white;
    tempVal = this->valWhite * this->valWhite;
    EXPECT_EQ(tempVal, this->valWhite);
    tempVal = this->valWhite;
    tempVal *= this->valWhite;
    EXPECT_EQ(tempVal, this->valWhite);

    // Multiplication off white with white must be white;
    tempVal = this->valGray * this->valGray;
    EXPECT_EQ(tempVal, this->valDarkGray);
    tempVal = this->valGray;
    tempVal *= this->valGray;
    EXPECT_EQ(tempVal, this->valDarkGray);
}

TYPED_TEST(rgbaOperatorTests, testAddition)
{

    TypeParam tempVal = this->valDarkGray + this->valBLACKTransparent; // Addition off zeros to value must be value;
    EXPECT_EQ(tempVal, this->valDarkGray);
    tempVal = this->valDarkGray;
    tempVal += this->valBLACKTransparent; // Addition off zeros to value must be value;
    EXPECT_EQ(tempVal, this->valDarkGray);

    tempVal = this->valBLACKTransparent + this->valDarkGray; // Addition off value to zeros must be value;
    EXPECT_EQ(tempVal, this->valDarkGray);
    tempVal = this->valBLACKTransparent;
    tempVal += this->valDarkGray; // Addition off value to zeros must be value;
    EXPECT_EQ(tempVal, this->valDarkGray);

    tempVal = this->valWhite + this->valDarkGray; // Addition off any value to white must be white;
    EXPECT_EQ(tempVal, this->valWhite);
    tempVal = this->valWhite;
    tempVal += this->valDarkGray; // Addition off any value to white must be white;
    EXPECT_EQ(tempVal, this->valWhite);

    tempVal = this->valDarkGray + this->valWhite; // Addition off white to any value must be white;
    EXPECT_EQ(tempVal, this->valWhite);
    tempVal = this->valDarkGray;
    tempVal += this->valWhite; // Addition off white to any value must be white;
    EXPECT_EQ(tempVal, this->valWhite);

    tempVal = this->valDarkGray + this->valDarkGray; // Addition off white to any value must be white;
    EXPECT_EQ(tempVal, this->valGray);
    tempVal = this->valDarkGray;
    tempVal += this->valDarkGray; // Addition off white to any value must be white;
    EXPECT_EQ(tempVal, this->valGray);

    tempVal = this->valGray + this->valGray; // Addition off white to any value must be white;
    EXPECT_EQ(tempVal, this->valWhite);
    EXPECT_NE(tempVal, this->valBLACKTransparent);
}

TYPED_TEST(rgbaOperatorTests, testSubstraction)
{

    // Addition off zeros to value must be value;
    TypeParam tempVal = this->valDarkGray - this->valBLACKTransparent;
    EXPECT_EQ(tempVal, this->valDarkGray);
    tempVal = this->valDarkGray;
    tempVal -= this->valBLACKTransparent;
    EXPECT_EQ(tempVal, this->valDarkGray);

    // Addition off value to zeros must be value;
    tempVal = this->valBLACKTransparent - this->valDarkGray;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);
    tempVal = this->valBLACKTransparent;
    tempVal -= this->valDarkGray;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);

    // Addition off white to any value must be white;
    tempVal = this->valDarkGray - this->valWhite;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);
    tempVal = this->valDarkGray;
    tempVal -= this->valWhite;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);

    // Addition off white to any value must be white;
    tempVal = this->valGray - this->valDarkGray;
    if (typeid(TypeParam) == typeid(ito::Rgba32))
    {
        (*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).alpha() = 255;
    }
    EXPECT_EQ(tempVal, this->valDarkGray);

    tempVal = this->valGray;
    tempVal -= this->valDarkGray;
    if (typeid(TypeParam) == typeid(ito::Rgba32))
    {
        (*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).alpha() = 255;
    }
    EXPECT_EQ(tempVal, this->valDarkGray);
}

TYPED_TEST(rgbaOperatorTests, testDividation)
{
    EXPECT_ANY_THROW(this->valDarkGray / this->valBLACKTransparent);

    TypeParam tempVal = this->valBLACKTransparent / this->valWhite; // Multiplication off zero with white must be zeros;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);

    tempVal = this->valBLACKTransparent;
    tempVal /= this->valWhite; // Multiplication off zero with white must be zeros;
    EXPECT_EQ(tempVal, this->valBLACKTransparent);

    tempVal = this->valWhite / this->valWhite; // Multiplication off white with zero must be zeros;
    EXPECT_EQ(tempVal, this->valWhite);

    tempVal = this->valWhite;
    tempVal /= this->valWhite; // Multiplication off white with zero must be zeros;
    EXPECT_EQ(tempVal, this->valWhite);

    tempVal = this->valWhite / this->valDarkGray; // Multiplication off white with zero must be zeros;
    EXPECT_EQ(tempVal, this->valWhite);
    tempVal = this->valWhite;
    tempVal /= this->valDarkGray; // Multiplication off white with zero must be zeros;
    EXPECT_EQ(tempVal, this->valWhite);

    tempVal = this->valDarkGray / this->valWhite; // Multiplication off white with zero must be zeros;
    EXPECT_EQ(tempVal, this->valDarkGray);
    tempVal = this->valDarkGray;
    tempVal /= this->valWhite; // Multiplication off white with zero must be zeros;
    EXPECT_EQ(tempVal, this->valDarkGray);

    tempVal = this->valDarkGray / this->valGray; // Multiplication off white with zero must be zeros;

    if (typeid(TypeParam) == typeid(ito::Rgba32))
    {
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).alpha(), 255);
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).red(), 127);
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).blue(), 127);
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).green(), 127);
    }
    else if (typeid(TypeParam) == typeid(ito::AlphaChannel))
    {
        EXPECT_EQ(tempVal, TypeParam(127));
    }

    tempVal = this->valDarkGray;
    tempVal /= this->valGray; // Multiplication off white with zero must be zeros;

    if (typeid(TypeParam) == typeid(ito::Rgba32))
    {
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).alpha(), 255);
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).red(), 127);
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).blue(), 127);
        EXPECT_EQ((*(reinterpret_cast<ito::Rgba32 *>(&tempVal))).green(), 127);
    }
    else
    {
        EXPECT_EQ(tempVal, TypeParam(127));
    }
}

TYPED_TEST(rgbaOperatorTests, testElementSize)
{
    EXPECT_EQ(sizeof(TypeParam), 4);
    EXPECT_EQ(sizeof(TypeParam), sizeof(ito::uint32));
}

TYPED_TEST(rgbaOperatorTests, testConstructors)
{
    if (typeid(TypeParam) == typeid(ito::Rgba32))
    {
        EXPECT_EQ(ito::Rgba32(255, 255, 255, 255), ito::Rgba32(255));
        EXPECT_EQ(TypeParam(255), TypeParam(255));
        EXPECT_EQ(ito::Rgba32(0, 0, 0, 0), ito::Rgba32::zeros());
        EXPECT_NE(TypeParam(255), TypeParam::zeros());

        EXPECT_EQ(ito::Rgba32(255, 255, 0, 0), ito::Rgba32(255, 255, 0, 0));
        EXPECT_NE(ito::Rgba32(255, 0, 255, 0), ito::Rgba32(255, 255, 0, 0));

        ito::Rgba32 c1(255, 255, 0, 0);
        ito::Rgba32 c2(255, 0, 255, 0);
        ito::Rgba32 c3(255, 0, 0, 255);
        ito::Rgba32 black = ito::Rgba32::black();

        EXPECT_EQ((*static_cast<ito::RGBChannel<ito::Rgba32::RGBA_R> *>(&c1)).value(),
                  ito::Rgba32(255, 255, 0, 0).red());

        EXPECT_EQ(ito::Rgba32(255, 0, 255, 0), ito::Rgba32(255, 0, 255, 0));
        EXPECT_NE(ito::Rgba32(255, 12, 255, 0), ito::Rgba32(255, 0, 255, 0));
        EXPECT_EQ((*static_cast<ito::RGBChannel<ito::Rgba32::RGBA_G> *>(&c2)).value(),
                  ito::Rgba32(255, 0, 255, 0).green());

        EXPECT_EQ(ito::Rgba32(255, 0, 0, 255), ito::Rgba32(255, 0, 0, 255));
        EXPECT_NE(ito::Rgba32(255, 12, 255, 0), ito::Rgba32(255, 0, 0, 255));
        EXPECT_EQ((*static_cast<ito::RGBChannel<ito::Rgba32::RGBA_B> *>(&c3)).value(),
                  ito::Rgba32(255, 0, 0, 255).blue());

        EXPECT_EQ(ito::Rgba32(255, 0, 0, 0), ito::Rgba32::black());
        EXPECT_NE(ito::Rgba32(255, 0, 1, 0), ito::Rgba32::black());
        EXPECT_EQ((*static_cast<ito::RGBChannel<ito::Rgba32::RGBA_A> *>(&black)).value(), 255);
        EXPECT_EQ(255, ito::Rgba32::black().alpha());

        ito::Rgba32 tempVal(128, 3, 4, 5);

        EXPECT_EQ(tempVal.alpha(), 128);
        EXPECT_EQ(tempVal.red(), 3);
        EXPECT_EQ(tempVal.green(), 4);
        EXPECT_EQ(tempVal.blue(), 5);
    }
    else if (typeid(TypeParam) == typeid(ito::AlphaChannel))
    {
        EXPECT_EQ(TypeParam(255), TypeParam(255));
        EXPECT_EQ(TypeParam(0), TypeParam::zeros());
        EXPECT_EQ(TypeParam::zeros(), TypeParam::zeros());

        ito::Rgba32 c1(255);
        ito::Rgba32 c2(128);
        ito::Rgba32 c3(255, 0, 0, 0);
        ito::Rgba32 c4(128, 0, 0, 0);
        EXPECT_EQ(TypeParam(255), *(reinterpret_cast<TypeParam *>(&c1)));
        EXPECT_NE(TypeParam(128), *(reinterpret_cast<TypeParam *>(&c2)));

        EXPECT_EQ(TypeParam(255), *(reinterpret_cast<TypeParam *>(&c3)));
        EXPECT_EQ(TypeParam(128), *(reinterpret_cast<TypeParam *>(&c4)));
    }
    else
    {
        TypeParam zero(0);

        EXPECT_EQ(TypeParam(255), TypeParam(255));
        EXPECT_EQ(TypeParam(0), TypeParam::zeros());
        EXPECT_NE((*reinterpret_cast<ito::Rgba32 *>(&zero)).alpha(), ito::Rgba32::zeros().alpha());
        EXPECT_EQ((*reinterpret_cast<ito::Rgba32 *>(&zero)).alpha(), 255);
        EXPECT_EQ(TypeParam::zeros(), TypeParam::zeros());

        ito::Rgba32 c1(255);
        ito::Rgba32 c2(128);

        EXPECT_EQ(TypeParam(255), *(reinterpret_cast<TypeParam *>(&c1)));
        EXPECT_EQ(TypeParam(128), *(reinterpret_cast<TypeParam *>(&c2)));
    }

    ito::Rgba32 t1_(255, 11, 125, 15);
    TypeParam t1 = *(reinterpret_cast<TypeParam *>(&t1_));
    TypeParam t2 = t1;
    TypeParam t3 = t2;

    EXPECT_EQ(t1, t3);
}

TYPED_TEST(rgbaOperatorTests, testDefines)
{
    if (typeid(TypeParam) == typeid(ito::Rgba32))
    {
        EXPECT_EQ(ito::Rgba32::RGBA_B, 0);
        EXPECT_EQ(ito::Rgba32::RGBA_G, 1);
        EXPECT_EQ(ito::Rgba32::RGBA_R, 2);
        EXPECT_EQ(ito::Rgba32::RGBA_A, 3);
        EXPECT_EQ(ito::Rgba32::RGBA_Y, 4);
        EXPECT_EQ(ito::Rgba32::RGBA_RGB, 5);
    }
}

TYPED_TEST(rgbaOperatorTests, testGrayConversion)
{
    ito::Rgba32 myVal(255);

    EXPECT_TRUE(
        ito::isZeroValue<ito::float32>(this->valWhite.gray() - 255.0, std::numeric_limits<ito::float32>::epsilon()));
    EXPECT_TRUE(
        ito::isZeroValue<ito::float32>(this->valBLACKTransparent.gray(), std::numeric_limits<ito::float32>::epsilon()));
}

TYPED_TEST(rgbaOperatorTests, testAssignment)
{
    ito::uint32 longWhite = 0xFFFFFFFF;
    ito::uint32 longTrans = 0x00000000;
    ito::uint32 longBlack = 0xFF000000;

    ito::uint32 longRED = 0xFFFF0000;
    ito::uint32 longGREEN = 0xFF00FF00;
    ito::uint32 longBLUE = 0xFF0000FF;

    TypeParam tmpVal;
    tmpVal = longWhite;
    EXPECT_EQ(tmpVal, this->valWhite);

    tmpVal = longTrans;
    EXPECT_EQ(tmpVal, this->valBLACKTransparent);

    tmpVal = longBlack;
    EXPECT_EQ(tmpVal, this->valBLACK);

    tmpVal = longRED;
    EXPECT_EQ(tmpVal, (*reinterpret_cast<TypeParam *>(&this->valRED)));

    tmpVal = longGREEN;
    EXPECT_EQ(tmpVal, (*reinterpret_cast<TypeParam *>(&this->valGREEN)));

    tmpVal = longBLUE;
    EXPECT_EQ(tmpVal, (*reinterpret_cast<TypeParam *>(&this->valBLUE)));

    TypeParam tempVal = this->valWhite; // Multiplication off zero with white must be zeros;
    tempVal = this->valBLACKTransparent;
    tempVal = this->valBLACK;
    TypeParam tempVal2(0);
    TypeParam tempVal3(0);
    tempVal2 = tempVal;
    tempVal3 = tempVal2;
    tempVal = tempVal3;

    EXPECT_EQ(tempVal, this->valBLACK);

    tempVal = this->valWhite;
    *(reinterpret_cast<ito::Rgba32 *>(&tempVal2)) = *(reinterpret_cast<ito::Rgba32 *>(&tempVal));
    EXPECT_EQ(tempVal2, this->valWhite);

    *(reinterpret_cast<ito::RedChannel *>(&tempVal2)) = *(reinterpret_cast<ito::RedChannel *>(&this->valWhite));
    EXPECT_EQ(tempVal2, this->valWhite);

    *(reinterpret_cast<ito::BlueChannel *>(&tempVal2)) = *(reinterpret_cast<ito::BlueChannel *>(&this->valWhite));
    EXPECT_EQ(tempVal2, this->valWhite);

    *(reinterpret_cast<ito::GreenChannel *>(&tempVal2)) = *(reinterpret_cast<ito::GreenChannel *>(&this->valWhite));
    EXPECT_EQ(tempVal2, this->valWhite);

    *(reinterpret_cast<ito::AlphaChannel *>(&tempVal2)) = *(reinterpret_cast<ito::AlphaChannel *>(&this->valWhite));
    EXPECT_EQ(tempVal2, this->valWhite);
}

TYPED_TEST(rgbaOperatorTests, testBoolEqual)
{
    ito::Rgba32 c1(0, 0, 0, 0);
    ito::Rgba32 c2(1, 0, 0, 0);
    ito::Rgba32 c3(0, 1, 0, 0);
    ito::Rgba32 c4(0, 0, 1, 0);
    ito::Rgba32 c5(0, 0, 0, 1);
    TypeParam val1 = *(static_cast<TypeParam *>(&c1));
    TypeParam val2 = *(static_cast<TypeParam *>(&c2));
    if (typeid(TypeParam) == typeid(ito::Rgba32) || typeid(TypeParam) == typeid(ito::AlphaChannel))
    {
        EXPECT_TRUE(val1 != val2);
        EXPECT_FALSE(val1 == val2);
    }
    else
    {
        EXPECT_TRUE(val1 == val2);
        EXPECT_FALSE(val1 != val2);
    }

    val2 = *(static_cast<TypeParam *>(&c3));
    if (typeid(TypeParam) == typeid(ito::Rgba32) || typeid(TypeParam) == typeid(ito::RedChannel))
    {
        EXPECT_TRUE(val1 != val2);
        EXPECT_FALSE(val1 == val2);
    }
    else
    {
        EXPECT_TRUE(val1 == val2);
        EXPECT_FALSE(val1 != val2);
    }

    val2 = *(static_cast<TypeParam *>(&c4));
    if (typeid(TypeParam) == typeid(ito::Rgba32) || typeid(TypeParam) == typeid(ito::GreenChannel))
    {
        EXPECT_TRUE(val1 != val2);
        EXPECT_FALSE(val1 == val2);
    }
    else
    {
        EXPECT_TRUE(val1 == val2);
        EXPECT_FALSE(val1 != val2);
    }

    val2 = *(static_cast<TypeParam *>(&c5));
    if (typeid(TypeParam) == typeid(ito::Rgba32) || typeid(TypeParam) == typeid(ito::BlueChannel))
    {
        EXPECT_TRUE(val1 != val2);
        EXPECT_FALSE(val1 == val2);
    }
    else
    {
        EXPECT_TRUE(val1 == val2);
        EXPECT_FALSE(val1 != val2);
    }

    EXPECT_TRUE(this->valWhite == TypeParam(255));
    EXPECT_FALSE(this->valWhite != TypeParam(255));

    if (typeid(TypeParam) == typeid(ito::Rgba32) || typeid(TypeParam) == typeid(ito::AlphaChannel))
    {
        EXPECT_FALSE(this->valBLACKTransparent == this->valBLACK);
        EXPECT_TRUE(this->valBLACKTransparent != this->valBLACK);
    }
    else
    {
        EXPECT_FALSE(this->valBLACKTransparent != this->valBLACK);
        EXPECT_TRUE(this->valBLACKTransparent == this->valBLACK);
    }
}

/*! \class
    \brief
*/
template <typename _Tp> class rgbaChannalTests : public ::testing::Test
{
  public:
    virtual void SetUp(void){

    };

    virtual void TearDown(void){};
    typedef _Tp TypeParam;
};

TYPED_TEST_CASE(rgbaChannalTests, ItomAllChannelTypes);
/*!

*/
TYPED_TEST(rgbaChannalTests, testChannelSpecificOperators)
{
}
