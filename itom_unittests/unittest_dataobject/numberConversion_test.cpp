
#include <iostream>

#include "../../common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.

#include "opencv2/opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"

template <typename _Tp> class Real_ComplexTest : public ::testing::Test 
{ 
    public:

};


TYPED_TEST_CASE(Real_ComplexTest, ItomDataStandardTypes);

//!< Test for numberConversion<type>() function.
TYPED_TEST(Real_ComplexTest, numberConversionIntToFloat_Test)
{
    //!< Declaring fixed-point variables.
    ito::int8 int8_var1, int8_var2, int8_var3;
    ito::int16 int16_var1, int16_var2, int16_var3;
    ito::int32 int32_var1, int32_var2, int32_var3;
    ito::uint8 uint8_var1, uint8_var2, uint8_var3;
    ito::uint16 uint16_var1, uint16_var2, uint16_var3;

    //!< Declaring ito::float[32/64] type variables.
    ito::float32 float32_var1, float32_var2, float32_var3;
    ito::float64 float64_var1, float64_var2, float64_var3;
    //!< Test for conversion from ito::int8 to ito::int16 using numberConversion<Type>(...) function.
    int16_var1,int16_var2,int16_var3 =0;
    int8_var1 = 5;
    int8_var2 = 0;
    int8_var3 = -5;
    int16_var1 = ito::numberConversion<ito::int16>(ito::tInt8, &int8_var1);        //!< Converting int8 type value into int16 type using numberConversion() method.
    int16_var2 = ito::numberConversion<ito::int16>(ito::tInt8, &int8_var2);        //!< Converting int8 type value into int16 type using numberConversion() method.
    int16_var3 = ito::numberConversion<ito::int16>(ito::tInt8, &int8_var3);        //!< Converting int8 type value into int16 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::int16>(5), int16_var1);                    //!< testing if int16_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int16>(0), int16_var2);                    //!< testing if int16_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int16>(-5), int16_var3);                    //!< testing if int16_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int16 to ito::int8 using numberConversion<Type>(...) function.
    int8_var1,int8_var2,int8_var3 = 0;
    int16_var1 = 5;
    int16_var2 = 0;
    int16_var3 = -5;
    int8_var1 = ito::numberConversion<ito::int8>(ito::tInt16, &int16_var1);        //!< Converting int16 type value into int8 type using numberConversion() method.
    int8_var2 = ito::numberConversion<ito::int8>(ito::tInt16, &int16_var2);        //!< Converting int16 type value into int8 type using numberConversion() method.
    int8_var3 = ito::numberConversion<ito::int8>(ito::tInt16, &int16_var3);        //!< Converting int16 type value into int8 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::int8>(5), int8_var1 );                    //!< testing if int8_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int8>(0), int8_var2 );                    //!< testing if int8_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int8>(-5), int8_var3 );                    //!< testing if int8_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int16 to ito::int32 using numberConversion<Type>(...) function.
    int32_var1,int32_var2,int32_var3 = 0;
    int16_var1 = 5;
    int16_var2 = 0;
    int16_var3 = -5;
    int32_var1 = ito::numberConversion<ito::int32>(ito::tInt16, &int16_var1);        //!< Converting int16 type value into int32 type using numberConversion() method.
    int32_var2 = ito::numberConversion<ito::int32>(ito::tInt16, &int16_var2);        //!< Converting int16 type value into int32 type using numberConversion() method.
    int32_var3 = ito::numberConversion<ito::int32>(ito::tInt16, &int16_var3);        //!< Converting int16 type value into int32 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(5), int32_var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(0), int32_var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(-5), int32_var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int8 to ito::int32 using numberConversion<Type>(...) function.
    int32_var1,int32_var2,int32_var3 = 0;
    int8_var1 = 5;
    int8_var2 = 0;
    int8_var3 = -5;
    int32_var1 = ito::numberConversion<ito::int32>(ito::tInt8, &int8_var1);        //!< Converting int8 type value into int32 type using numberConversion() method.
    int32_var2 = ito::numberConversion<ito::int32>(ito::tInt8, &int8_var2);        //!< Converting int8 type value into int32 type using numberConversion() method.
    int32_var3 = ito::numberConversion<ito::int32>(ito::tInt8, &int8_var3);        //!< Converting int8 type value into int32 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(5), int32_var1 );                    //!< testing if int32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(0), int32_var2 );                    //!< testing if int32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(-5), int32_var3 );                    //!< testing if int32_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int32 to ito::int16 using numberConversion<Type>(...) function.
    int16_var1,int16_var2,int16_var3 = 0;
    int32_var1 = 5;
    int32_var2 = 0;
    int32_var3 = -5;
    int16_var1 = ito::numberConversion<ito::int16>(ito::tInt32, &int32_var1);        //!< Converting int32 type value into int16 type using numberConversion() method.
    int16_var2 = ito::numberConversion<ito::int16>(ito::tInt32, &int32_var2);        //!< Converting int32 type value into int16 type using numberConversion() method.
    int16_var3 = ito::numberConversion<ito::int16>(ito::tInt32, &int32_var3);        //!< Converting int32 type value into int16 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::int16>(5), int16_var1 );                        //!< testing if int16_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int16>(0), int16_var2 );                        //!< testing if int16_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int16>(-5), int16_var3 );                        //!< testing if int16_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int32 to ito::int8 using numberConversion<Type>(...) function.
    int8_var1,int8_var2,int8_var3 = 0;
    int16_var1 = 0;
    int16_var2 = 0;
    int16_var3 = 0;
    int8_var1 = ito::numberConversion<ito::int8>(ito::tInt32, &int32_var1);        //!< Converting int32 type value into int8 type using numberConversion() method.
    int8_var2 = ito::numberConversion<ito::int8>(ito::tInt32, &int32_var2);        //!< Converting int32 type value into int8 type using numberConversion() method.
    int8_var3 = ito::numberConversion<ito::int8>(ito::tInt32, &int32_var3);        //!< Converting int32 type value into int8 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::int8>(5), int8_var1 );                    //!< testing if int8_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int8>(0), int8_var2 );                    //!< testing if int8_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int8>(-5), int8_var3 );                    //!< testing if int8_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int8 to ito::float32 using numberConversion<Type>(...) function.
    float32_var1,float32_var2,float32_var3 = 0;
    int16_var1 = 0;
    int16_var2 = 0;
    int16_var3 = 0;
    float32_var1 = ito::numberConversion<ito::float32>(ito::tInt8, &int8_var1);        //!< Converting int8 type value into float32 type using numberConversion() method.
    float32_var2 = ito::numberConversion<ito::float32>(ito::tInt8, &int8_var2);        //!< Converting int8 type value into float32 type using numberConversion() method.
    float32_var3 = ito::numberConversion<ito::float32>(ito::tInt8, &int8_var3);        //!< Converting int8 type value into float32 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(5), float32_var1 );                    //!< testing if float32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(0), float32_var2 );                    //!< testing if float32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(-5), float32_var3 );                    //!< testing if float32_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int16 to ito::float32 using numberConversion<Type>(...) function.
    float32_var1,float32_var2,float32_var3 = 0;
    int16_var1 = 5;
    int16_var2 = 0;
    int16_var3 = -5;
    float32_var1 = ito::numberConversion<ito::float32>(ito::tInt16, &int16_var1);        //!< Converting int16 type value into float32 type using numberConversion() method.
    float32_var2 = ito::numberConversion<ito::float32>(ito::tInt16, &int16_var2);        //!< Converting int16 type value into float32 type using numberConversion() method.
    float32_var3 = ito::numberConversion<ito::float32>(ito::tInt16, &int16_var3);        //!< Converting int16 type value into float32 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(5), float32_var1 );                        //!< testing if float32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(0), float32_var2 );                        //!< testing if float32_var2 contains the desired original value after conversion.    
    EXPECT_EQ( cv::saturate_cast<ito::float32>(-5), float32_var3 );                        //!< testing if float32_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::int32 to ito::float32 using numberConversion<Type>(...) function.
    float32_var1,float32_var2,float32_var3 = 0;
    int32_var1 = 5;
    int32_var2 = 0;
    int32_var3 = -5;
    float32_var1 = ito::numberConversion<ito::float32>(ito::tInt32, &int32_var1);        //!< Converting int32 type value into float32 type using numberConversion() method.
    float32_var2 = ito::numberConversion<ito::float32>(ito::tInt32, &int32_var2);        //!< Converting int32 type value into float32 type using numberConversion() method.
    float32_var3 = ito::numberConversion<ito::float32>(ito::tInt32, &int32_var3);        //!< Converting int32 type value into float32 type using numberConversion() method.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(5), float32_var1 );                        //!< testing if float32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::float32>(0), float32_var2 );                        //!< testing if float32_var2 contains the desired original value after conversion.    
    EXPECT_EQ( cv::saturate_cast<ito::float32>(-5), float32_var3 );                        //!< testing if float32_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::float32 to ito::int32 using numberConversion<Type>(...) function.
    float32_var1 = -5.0;    /*!< Initializing a ito::float32 type variable with negative value.*/
    float32_var2 = -4.9;
    float32_var3 = -4.1;

    int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var1);        //!< Converting float32 type value into int32 type using numberConversion() method.
    int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var2);        //!< Converting float32 type value into int32 type using numberConversion() method.
    int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var3);        //!< Converting float32 type value into int32 type using numberConversion() method.

    EXPECT_EQ( cv::saturate_cast<ito::int32>(-5.0), int32_var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.9), int32_var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.1), int32_var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.

    float32_var1 = 5.0;
    float32_var2 = 4.9;
    float32_var3 = 4.1;

    int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var1);        //!< Converting float32 type value into int32 type using numberConversion() method.
    int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var2);        //!< Converting float32 type value into int32 type using numberConversion() method.
    int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var3);        //!< Converting float32 type value into int32 type using numberConversion() method.

    EXPECT_EQ( cv::saturate_cast<ito::int32>(5.0), int32_var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(4.9), int32_var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(4.1), int32_var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.

    //!< Test for conversion from ito::float64 to ito::int32 using numberConversion<Type>(...) function.
    float64_var1 = -5.0;    /*!< Initializing a ito::float64 type variable with negative value.*/
    float64_var2 = -4.9;
    float64_var3 = -4.1;

    int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var1);        //!< Converting float64 type value into int32 type using numberConversion() method.
    int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var2);        //!< Converting float64 type value into int32 type using numberConversion() method.
    int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var3);        //!< Converting float64 type value into int32 type using numberConversion() method.

    EXPECT_EQ( cv::saturate_cast<ito::int32>(-5.0), int32_var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.9), int32_var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.1), int32_var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.

    float64_var1 = 5.0;
    float64_var2 = 4.9;
    float64_var3 = 4.1;

    int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var1);        //!< Converting float64 type value into int32 type using numberConversion() method.
    int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var2);        //!< Converting float64 type value into int32 type using numberConversion() method.
    int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var3);        //!< Converting float64 type value into int32 type using numberConversion() method.

    EXPECT_EQ( cv::saturate_cast<ito::int32>(5.0), int32_var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(4.9), int32_var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
    EXPECT_EQ( cv::saturate_cast<ito::int32>(4.1), int32_var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.
}

//!< Test for numberConversion<type>() function for floating point variables.
TYPED_TEST(Real_ComplexTest, numberConversionRGBAToVALUE_Test)
{
    ito::Rgba32 rgba_var1= ito::Rgba32::zeros();
    ito::Rgba32 rgba_var2= ito::Rgba32::black();
    ito::Rgba32 rgba_var3= ito::Rgba32(255,0,255,0);

    if(typeid(TypeParam) == typeid(ito::uint32) || typeid(TypeParam) == typeid(ito::int32))
    {
        TypeParam var1 = ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var1);        //!< Converting float64 type value into int32 type using numberConversion() method.
        TypeParam var2 = ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var2);        //!< Converting float64 type value into int32 type using numberConversion() method.
        TypeParam var3 = ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var3);        //!< Converting float64 type value into int32 type using numberConversion() method.

        EXPECT_EQ( (TypeParam)0x00000000, var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
        EXPECT_EQ( (TypeParam)0xFF000000, var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
        EXPECT_EQ( (TypeParam)0xFF00FF00, var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.
    }
    else if(typeid(TypeParam) == typeid(ito::uint8) || typeid(TypeParam) == typeid(ito::uint16) || typeid(TypeParam) == typeid(ito::float32)  || typeid(TypeParam) == typeid(ito::float64))
    {
        TypeParam var1 = ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var1);        //!< Converting float64 type value into int32 type using numberConversion() method.
        TypeParam var2 = ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var2);        //!< Converting float64 type value into int32 type using numberConversion() method.
        TypeParam var3 = ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var3);        //!< Converting float64 type value into int32 type using numberConversion() method.

        EXPECT_EQ( cv::saturate_cast<TypeParam>(rgba_var1.gray()), var1 );                        //!< testing if int32_var1 contains the desired original value after conversion.
        EXPECT_EQ( cv::saturate_cast<TypeParam>(rgba_var2.gray()), var2 );                        //!< testing if int32_var2 contains the desired original value after conversion.
        EXPECT_EQ( cv::saturate_cast<TypeParam>(rgba_var3.gray()), var3 );                        //!< testing if int32_var3 contains the desired original value after conversion.
    }
    else
    {
        EXPECT_ANY_THROW(ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var1));
        EXPECT_ANY_THROW(ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var2));
        EXPECT_ANY_THROW(ito::numberConversion<TypeParam>(ito::tRGBA32, &rgba_var3));
    }
}

//!< Test for numberConversion<type>() function for floating point variables.
TYPED_TEST(Real_ComplexTest, numberConversionFloatToFloat_Test)
{
    ito::float32 float32_var1,float32_var2,float32_var3,float32_var4,float32_var5,float32_var6,float32_var7;
    ito::float64 float64_var1,float64_var2,float64_var3,float64_var4,float64_var5,float64_var6,float64_var7;

    float64_var1,float64_var2,float64_var3,float64_var4,float64_var5,float64_var6,float64_var7 =0;
    float32_var1 = -5.0;
    float32_var2 = -4.9;
    float32_var3 = -4.1;
    float32_var4 = 0;
    float32_var5 = 4.1;
    float32_var6 = 4.9;
    float32_var7 = 5.0;

    //!< Converting ito::float32 type numbers of different critical values into ito::float64 type numbers.
    float64_var1 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var1);        //!< Converting float32 type value into float64 type using numberConversion() method.
    float64_var2 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var2);        //!< Converting float32 type value into float64 type using numberConversion() method.
    float64_var3 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var3);        //!< Converting float32 type value into float64 type using numberConversion() method.
    float64_var4 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var4);        //!< Converting float32 type value into float64 type using numberConversion() method.
    float64_var5 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var5);        //!< Converting float32 type value into float64 type using numberConversion() method.
    float64_var6 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var6);        //!< Converting float32 type value into float64 type using numberConversion() method.
    float64_var7 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var7);        //!< Converting float32 type value into float64 type using numberConversion() method.

    //!< Test for conversion from ito::float32 to ito::float64.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5.0), float64_var1);                    //!< testing if float64_var1 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.9), float64_var2);                    //!< testing if float64_var2 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.1), float64_var3);                    //!< testing if float64_var3 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), float64_var4);                        //!< testing if float64_var4 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.1), float64_var5);                    //!< testing if float64_var5 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.9), float64_var6);                    //!< testing if float64_var6 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5.0), float64_var7);                    //!< testing if float64_var7 contains the desired original value after conversion.

    float32_var1,float32_var2,float32_var3,float32_var4,float32_var5,float32_var6,float32_var7 = 0;

    //!< Converting ito::float64 type numbers of different critical values into ito::float32 type numbers.
    float32_var1 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var1);        //!< Converting float64 type value into float32 type using numberConversion() method.
    float32_var2 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var2);        //!< Converting float64 type value into float32 type using numberConversion() method.
    float32_var3 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var3);        //!< Converting float64 type value into float32 type using numberConversion() method.
    float32_var4 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var4);        //!< Converting float64 type value into float32 type using numberConversion() method.
    float32_var5 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var5);        //!< Converting float64 type value into float32 type using numberConversion() method.
    float32_var6 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var6);        //!< Converting float64 type value into float32 type using numberConversion() method.
    float32_var7 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var7);        //!< Converting float64 type value into float32 type using numberConversion() method.

        //!< Test for conversion from ito::float64 to ito::float32.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5.0), float32_var1);                    //!< testing if float64_var1 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.9), float32_var2);                    //!< testing if float64_var2 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.1), float32_var3);                    //!< testing if float64_var3 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), float32_var4);                        //!< testing if float64_var4 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.1), float32_var5);                    //!< testing if float64_var5 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.9), float32_var6);                    //!< testing if float64_var6 contains the desired original value after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5.0), float32_var7);                    //!< testing if float64_var7 contains the desired original value after conversion.
}

//!< Test for numberConversion<type>() function for floating point variables.
TYPED_TEST(Real_ComplexTest, numberConversionIntToComplex_Test)
{
    //!< Declaring fixed-point variables.
    ito::int8 int8_var1, int8_var2, int8_var3;
    ito::int16 int16_var1, int16_var2, int16_var3;
    ito::int32 int32_var1, int32_var2, int32_var3;
    ito::uint8 uint8_var2, uint8_var3;
    ito::uint16 uint16_var2, uint16_var3;

    //!< Declaring Complex64 type variables.
    ito::complex64 complex64_var1,complex64_var2,complex64_var3;
    ito::complex128 complex128_var1,complex128_var2,complex128_var3;

    int8_var1 = -5;
    int8_var2 = 0;
    int8_var3 = 5;

    complex64_var1 = ito::numberConversion<ito::complex64>(ito::tInt8,(void*)(&int8_var1) );    //!< converting the value in int8_var1 to complex64 value 
    complex128_var1 = ito::numberConversion<ito::complex128>(ito::tInt8,(void*)(&int8_var1) );    //!< converting the value in int8_var1 to complex128 value 
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5), complex64_var1.real() );                //!< Real part of complex64_var1 should be equal to the value in int8_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );                //!< Imaginary part of complex64_var1 should be 0.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5), complex128_var1.real() );                //!< Real part of complex128_var1 should be equal to the value in int8_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );                //!< Imaginary part of complex128_var1 should be 0.

    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tInt8,(void*)(&int8_var2) );    //!< converting the value in int8_var2 to complex64 value 
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );                //!< Real part of complex64_var2 should be equal to the value in int8_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                //!< Imaginary part of complex64_var2 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tInt8,(void*)(&int8_var2) );    //!< Converting the value in int8_var2 to complex128 value                
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );                //!< Real part of complex128_var2 should be equal to the value in int8_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                //!< Imaginary part of complex128_var2 should be 0.

    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tInt8,(void*)(&int8_var3) );    //!< converting the value in int8_var3 to complex64 value                 
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );                //!< Real part of complex64_var2 should be equal to the value in int8_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                //!< Imaginary part of complex64_var2 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tInt8,(void*)(&int8_var3) );    //!< Converting the value in int8_var2 to complex128 value                    
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );                //!< Real part of complex128_var3 should be equal to the value in int8_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                //!< Imaginary part of complex128_var3 should be 0.

    int16_var1 = -5;
    int16_var2 = 0;
    int16_var3 = 5;
    complex64_var1,complex64_var2,complex64_var3 = 0 ;
    complex128_var1,complex128_var2,complex128_var3 = 0 ;

    complex64_var1 = ito::numberConversion<ito::complex64>(ito::tInt16, &int16_var1);            //!< converting the value in int16_var1 to complex64 value 
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5), complex64_var1.real() );                //!< Real part of complex64_var1 should be equal to the value in int16_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );                //!< Imaginary part of complex64_var1 should be 0.
    complex128_var1 = ito::numberConversion<ito::complex128>(ito::tInt16, &int16_var1);            //!< Converting the value in int16_var1 to complex128 value        
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5), complex128_var1.real() );                //!< Real part of complex128_var1 should be equal to the value in int16_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );                //!< Imaginary part of complex128_var1 should be 0.

    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tInt16,(void*)(&int16_var2) );                //!< converting the value in int16_var2 to complex64 value     
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );                            //!< Real part of complex64_var2 should be equal to the value in int16_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                            //!< Imaginary part of complex64_var2 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tInt16,(void*)(&int16_var2) );            //!< Converting the value in int16_var2 to complex128 value                
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );                            //!< Real part of complex128_var2 should be equal to the value in int16_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                            //!< Imaginary part of complex128_var2 should be 0.

    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tInt16,(void*)(&int16_var3) );                //!< converting the value in int16_var3 to complex64 value     
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );                            //!< Real part of complex64_var3 should be equal to the value in int16_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                            //!< Imaginary part of complex64_var3 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tInt16,(void*)(&int16_var3) );            //!< Converting the value in int16_var3 to complex128 value            
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );                            //!< Real part of complex128_var3 should be equal to the value in int16_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                            //!< Imaginary part of complex128_var3 should be 0.

    int32_var1 = -5;
    int32_var2 = 0;
    int32_var3 = 5;
    complex64_var1,complex64_var2,complex64_var3 = 0 ;
    complex128_var1,complex128_var2,complex128_var3 = 0 ;

    complex64_var1 = ito::numberConversion<ito::complex64>(ito::tInt32, &int32_var1);                    //!< converting the value in int32_var1 to complex64 value     
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5), complex64_var1.real() );                        //!< Real part of complex64_var1 should be equal to the value in int32_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );                        //!< Imaginary part of complex64_var1 should be 0.
    complex128_var1 = ito::numberConversion<ito::complex128>(ito::tInt32, &int32_var1);                    //!< Converting the value in int32_var1 to complex128 value    
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5), complex128_var1.real() );                        //!< Real part of complex128_var1 should be equal to the value in int32_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );                        //!< Imaginary part of complex128_var1 should be 0.

    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tInt32,(void*)(&int32_var2) );                //!< converting the value in int32_var2 to complex64 value         
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );                            //!< Real part of complex64_var2 should be equal to the value in int32_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                            //!< Imaginary part of complex64_var2 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tInt32,(void*)(&int32_var2) );            //!< Converting the value in int32_var2 to complex128 value            
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );                            //!< Real part of complex128_var2 should be equal to the value in int32_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                            //!< Imaginary part of complex128_var2 should be 0.

    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tInt32,(void*)(&int32_var3) );                //!< converting the value in int32_var3 to complex64 value             
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );                            //!< Real part of complex64_var3 should be equal to the value in int32_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                            //!< Imaginary part of complex64_var3 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tInt32,(void*)(&int32_var3) );            //!< Converting the value in int32_var3 to complex128 value            
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );                            //!< Real part of complex128_var3 should be equal to the value in int32_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                            //!< Imaginary part of complex128_var3 should be 0.

    //!< Test for unsigned integer type variables.    
    uint8_var2 = 0;
    uint8_var3 = 5;

    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tUInt8,(void*)(&uint8_var2) );                    //!< converting the value in uint8_var2 to complex64 value     
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );                                //!< Real part of complex64_var2 should be equal to the value in uint8_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                                //!< Imaginary part of complex64_var2 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tUInt8,(void*)(&uint8_var2) );                //!< Converting the value in uint8_var2 to complex128 value        
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );                                //!< Real part of complex128_var2 should be equal to the value in uint8_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                                //!< Imaginary part of complex128_var2 should be 0.

    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tUInt8,(void*)(&uint8_var3) );                    //!< converting the value in uint8_var3 to complex64 value     
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );                                //!< Real part of complex64_var3 should be equal to the value in uint8_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                                //!< Imaginary part of complex64_var3 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tUInt8,(void*)(&uint8_var3) );                //!< Converting the value in uint8_var3 to complex128 value        
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );                                //!< Real part of complex128_var3 should be equal to the value in uint8_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                                //!< Imaginary part of complex128_var3 should be 0.

    uint16_var2 = 0;
    uint16_var3 = 5;
    complex64_var2,complex64_var3 = 0 ;
    complex128_var2,complex128_var3 = 0 ;

    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tUInt16,(void*)(&uint16_var2) );                //!< converting the value in uint16_var2 to complex64 value         
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );                                //!< Real part of complex64_var2 should be equal to the value in uint16_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                                //!< Imaginary part of complex64_var2 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tUInt16,(void*)(&uint16_var2) );                //!< Converting the value in uint16_var2 to complex128 value        
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );                                //!< Real part of complex128_var2 should be equal to the value in uint16_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                                //!< Imaginary part of complex128_var2 should be 0.

    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tUInt16,(void*)(&uint16_var3) );                //!< converting the value in uint16_var3 to complex64 value         
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );                                //!< Real part of complex64_var3 should be equal to the value in uint16_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                                //!< Imaginary part of complex64_var3 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tUInt16,(void*)(&uint16_var3) );                //!< Converting the value in uint16_var3 to complex128 value    
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );                                //!< Real part of complex128_var3 should be equal to the value in uint16_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                                //!< Imaginary part of complex128_var3 should be 0.
}


//!< Test for numberConversion<type>() function for floating point variables.
TYPED_TEST(Real_ComplexTest, numberConversionFloatToComplex_Test)
{
    //!< Declaring ito::float[32/64] type variables.
    ito::float32 float32_var1, float32_var2, float32_var3,float32_var4, float32_var5, float32_var6,float32_var7;
    ito::float64 float64_var1, float64_var2, float64_var3,float64_var4, float64_var5, float64_var6,float64_var7;
    //!< Declaring complex type variables.
    ito::complex64 complex64_var1,complex64_var2,complex64_var3,complex64_var4,complex64_var5,complex64_var6,complex64_var7;
    ito::complex128 complex128_var1,complex128_var2,complex128_var3,complex128_var4,complex128_var5,complex128_var6,complex128_var7;

    float32_var1 = -5;
    float32_var2 = -4.9;
    float32_var3 = -4.1;
    float32_var4 = 0;
    float32_var5 = 4.1;
    float32_var6 = 4.9;
    float32_var7 = 5.0;
    complex64_var1,complex64_var2,complex64_var3,complex64_var4,complex64_var5,complex64_var6,complex64_var7 = 0;
    complex128_var1,complex128_var2,complex128_var3,complex128_var4,complex128_var5,complex128_var6,complex128_var7 = 0;

    complex64_var1 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var1) );        //!< converting the value in float32_var1 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5.0), complex64_var1.real() );                    //!< Real part of complex64_var1 should be equal to the value in float32_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );                        //!< Imaginary part of complex64_var1 should be 0.
    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var2) );        //!< Converting the value in float32_var2 to complex64 value    
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.9), complex64_var2.real() );                    //!< Real part of complex64_var2 should be equal to the value in float32_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                        //!< Imaginary part of complex64_var2 should be 0.
    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var3) );        //!< converting the value in float32_var3 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.1), complex64_var3.real() );                    //!< Real part of complex64_var3 should be equal to the value in float32_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                        //!< Imaginary part of complex64_var3 should be 0.
    complex64_var4 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var4) );        //!< converting the value in float32_var4 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.real() );                        //!< Real part of complex64_var4 should be equal to the value in float32_var4 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.imag() );                        //!< Imaginary part of complex64_var4 should be 0.
    complex64_var5 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var5) );        //!< converting the value in float32_var5 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.1), complex64_var5.real() );                        //!< Real part of complex64_var5 should be equal to the value in float32_var5 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var5.imag() );                        //!< Imaginary part of complex64_var5 should be 0.
    complex64_var6 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var6) );        //!< converting the value in float32_var6 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.9), complex64_var6.real() );                        //!< Real part of complex64_var6 should be equal to the value in float32_var6 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var6.imag() );                        //!< Imaginary part of complex64_var6 should be 0.
    complex64_var7 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var7) );        //!< converting the value in float32_var7 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5.0), complex64_var7.real() );                        //!< Real part of complex64_var7 should be equal to the value in float32_var7 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var7.imag() );                        //!< Imaginary part of complex64_var7 should be 0.


    complex128_var1 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var1) );    //!< converting the value in float32_var1 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5.0), complex128_var1.real() );                    //!< Real part of complex128_var1 should be equal to the value in float32_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );                        //!< Imaginary part of complex128_var1 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var2) );    //!< converting the value in float32_var2 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.9), complex128_var2.real() );                    //!< Real part of complex128_var2 should be equal to the value in float32_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                        //!< Imaginary part of complex128_var2 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var3) );    //!< converting the value in float32_var3 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.1), complex128_var3.real() );                    //!< Real part of complex128_var3 should be equal to the value in float32_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                        //!< Imaginary part of complex128_var3 should be 0.
    complex128_var4 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var4) );    //!< converting the value in float32_var4 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.real() );                        //!< Real part of complex128_var4 should be equal to the value in float32_var4 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.imag() );                        //!< Imaginary part of complex128_var4 should be 0.
    complex128_var5 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var5) );    //!< converting the value in float32_var5 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.1), complex128_var5.real() );                    //!< Real part of complex128_var5 should be equal to the value in float32_var5 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var5.imag() );                        //!< Imaginary part of complex128_var5 should be 0.
    complex128_var6 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var6) );    //!< converting the value in float32_var6 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.9), complex128_var6.real() );                    //!< Real part of complex128_var6 should be equal to the value in float32_var6 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var6.imag() );                        //!< Imaginary part of complex128_var6 should be 0.
    complex128_var7 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var7) );    //!< converting the value in float32_var7 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5.0), complex128_var7.real() );                    //!< Real part of complex128_var7 should be equal to the value in float32_var7 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var7.imag() );                        //!< Imaginary part of complex128_var7 should be 0.


    //!< Test for float64 type variables.
    float64_var1 = -5;
    float64_var2 = -4.9;
    float64_var3 = -4.1;
    float64_var4 = 0;
    float64_var5 = 4.1;
    float64_var6 = 4.9;
    float64_var7 = 5.0;
    complex64_var1,complex64_var2,complex64_var3,complex64_var4,complex64_var5,complex64_var6,complex64_var7 = 0;
    complex128_var1,complex128_var2,complex128_var3,complex128_var4,complex128_var5,complex128_var6,complex128_var7 = 0;

    complex64_var1 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var1) );    //!< converting the value in float64_var1 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5.0), complex64_var1.real() );                //!< Real part of complex64_var1 should be equal to the value in float64_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );                    //!< Imaginary part of complex64_var1 should be 0.
    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var2) );    //!< converting the value in float64_var2 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.9), complex64_var2.real() );                //!< Real part of complex64_var2 should be equal to the value in float64_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );                    //!< Imaginary part of complex64_var2 should be 0.
    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var3) );    //!< converting the value in float64_var3 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.1), complex64_var3.real() );                //!< Real part of complex64_var3 should be equal to the value in float64_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );                    //!< Imaginary part of complex64_var3 should be 0.
    complex64_var4 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var4) );    //!< converting the value in float64_var4 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.real() );                    //!< Real part of complex64_var4 should be equal to the value in float64_var4 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.imag() );                    //!< Imaginary part of complex64_var4 should be 0.
    complex64_var5 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var5) );    //!< converting the value in float64_var5 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.1), complex64_var5.real() );                    //!< Real part of complex64_var5 should be equal to the value in float64_var5 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var5.imag() );                    //!< Imaginary part of complex64_var5 should be 0.
    complex64_var6 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var6) );    //!< converting the value in float64_var6 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.9), complex64_var6.real() );                    //!< Real part of complex64_var6 should be equal to the value in float64_var6 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var6.imag() );                    //!< Imaginary part of complex64_var6 should be 0.
    complex64_var7 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var7) );    //!< converting the value in float64_var7 to complex64 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5.0), complex64_var7.real() );                    //!< Real part of complex64_var7 should be equal to the value in float64_var7 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var7.imag() );                    //!< Imaginary part of complex64_var7 should be 0.


    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var1) );    //!< converting the value in float64_var1 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5.0), complex128_var1.real() );                    //!< Real part of complex128_var1 should be equal to the value in float64_var1 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );                        //!< Imaginary part of complex128_var1 should be 0.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var2) );    //!< converting the value in float64_var2 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.9), complex128_var2.real() );                    //!< Real part of complex128_var2 should be equal to the value in float64_var2 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );                        //!< Imaginary part of complex128_var2 should be 0.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var3) );    //!< converting the value in float64_var3 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.1), complex128_var3.real() );                    //!< Real part of complex128_var3 should be equal to the value in float64_var3 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );                        //!< Imaginary part of complex128_var3 should be 0.
    complex128_var4 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var4) );    //!< converting the value in float64_var4 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.real() );                        //!< Real part of complex128_var4 should be equal to the value in float64_var4 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.imag() );                        //!< Imaginary part of complex128_var4 should be 0.
    complex128_var5 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var5) );    //!< converting the value in float64_var5 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.1), complex128_var5.real() );                    //!< Real part of complex128_var5 should be equal to the value in float64_var5 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var5.imag() );                        //!< Imaginary part of complex128_var5 should be 0.
    complex128_var6 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var6) );    //!< converting the value in float64_var6 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.9), complex128_var6.real() );                    //!< Real part of complex128_var6 should be equal to the value in float64_var6 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var6.imag() );                        //!< Imaginary part of complex128_var6 should be 0.
    complex128_var7 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var7) );    //!< converting the value in float64_var7 to complex128 value
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5.0), complex128_var7.real() );                    //!< Real part of complex128_var7 should be equal to the value in float64_var7 after conversion.
    EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var7.imag() );                        //!< Imaginary part of complex128_var7 should be 0.
}

//!< Test for numberConversion<type>() function for floating point variables.
TYPED_TEST(Real_ComplexTest, numberConversionComplexToComplex_Test)
{
    ito::complex64 complex64_var1,complex64_var2,complex64_var3,complex64_var4,complex64_var5,complex64_var6,complex64_var7;
    ito::complex128 complex128_var1,complex128_var2,complex128_var3,complex128_var4,complex128_var5,complex128_var6,complex128_var7;

    complex64_var1 = (-5.0,5.0);
    complex64_var2 = (-5.0,0);
    complex64_var3 = (0,-5.0);
    complex64_var4 = (0,5.0);
    complex64_var5 = (5.0,-5.0);

    complex128_var1 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var1) );    //!< converting the value in complex64_var1 to complex128 value
    EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var1), complex128_var1 );                        //!< testing if complex128_var1 contains the desired original value after conversion.
    complex128_var2 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var2) );    //!< converting the value in complex64_var2 to complex128 value
    EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var2), complex128_var2 );                        //!< testing if complex128_var2 contains the desired original value after conversion.
    complex128_var3 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var3) );    //!< converting the value in complex64_var3 to complex128 value
    EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var3), complex128_var3 );                        //!< testing if complex128_var3 contains the desired original value after conversion.
    complex128_var4 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var4) );    //!< converting the value in complex64_var4 to complex128 value
    EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var4), complex128_var4 );                        //!< testing if complex128_var4 contains the desired original value after conversion.
    complex128_var5 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var5) );    //!< converting the value in complex64_var5 to complex128 value
    EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var5), complex128_var5 );                        //!< testing if complex128_var5 contains the desired original value after conversion.

    //!< Testing conversion from complex128 type variables to complex64 type variables.
    complex64_var1,complex64_var2,complex64_var3,complex64_var4,complex64_var5 = 0;    /*!< Defining complex64 type variables to 0 value. */
    complex128_var1 = (-5.0,5.0);                                                    /*!< Defining complex128 type variables to different test values. */
    complex128_var2 = (-5.0,0);
    complex128_var3 = (0,-5.0);
    complex128_var4 = (0,5.0);
    complex128_var5 = (5.0,-5.0);

    complex64_var1 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var1) );    //!< converting the value in complex128_var1 to complex64 value
    EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var1), complex64_var1 );                        //!< testing if complex64_var1 contains the desired original value after conversion.
    complex64_var2 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var2) );    //!< converting the value in complex128_var2 to complex64 value
    EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var2), complex64_var2 );                        //!< testing if complex64_var2 contains the desired original value after conversion.
    complex64_var3 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var3) );    //!< converting the value in complex128_var3 to complex64 value
    EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var3), complex64_var3 );                        //!< testing if complex64_var3 contains the desired original value after conversion.
    complex64_var4 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var4) );    //!< converting the value in complex128_var4 to complex64 value
    EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var4), complex64_var4 );                        //!< testing if complex64_var4 contains the desired original value after conversion.
    complex64_var5 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var5) );    //!< converting the value in complex128_var5 to complex64 value
    EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var5), complex64_var5 );                        //!< testing if complex64_var5 contains the desired original value after conversion.

    //!< Testing if exception is raised as it is supposed to, while converting from Complex datatype variables to any other type variables.
    EXPECT_ANY_THROW(ito::numberConversion<ito::float64>(ito::tComplex128,(void*)(&complex128_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::float32>(ito::tComplex128,(void*)(&complex128_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::int8>(ito::tComplex128,(void*)(&complex128_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::int16>(ito::tComplex128,(void*)(&complex128_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::int32>(ito::tComplex128,(void*)(&complex128_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::uint8>(ito::tComplex128,(void*)(&complex128_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::uint16>(ito::tComplex128,(void*)(&complex128_var1) ) );

    EXPECT_ANY_THROW(ito::numberConversion<ito::float64>(ito::tComplex64,(void*)(&complex64_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::float32>(ito::tComplex64,(void*)(&complex64_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::int8>(ito::tComplex64,(void*)(&complex64_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::int16>(ito::tComplex64,(void*)(&complex64_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::int32>(ito::tComplex64,(void*)(&complex64_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::uint8>(ito::tComplex64,(void*)(&complex64_var1) ) );
    EXPECT_ANY_THROW(ito::numberConversion<ito::uint16>(ito::tComplex64,(void*)(&complex64_var1) ) );
}
