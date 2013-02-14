#ifndef SATURATETEST_REAL_H
#define SATURATETEST_REAL_H

#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"

/*! \class SaturateTestReal
	\brief saturation test for real data types

	This test class tests saturation limits for real data types
*/
template <typename _Tp> class SaturateTestReal : public ::testing::Test { };


TYPED_TEST_CASE(SaturateTestReal, ItomRealDataTypes);


//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestReal, checkSaturateBoundaries)
{
    TypeParam max = std::numeric_limits<TypeParam>::max();
    TypeParam min = std::numeric_limits<TypeParam>::min();

    if(std::numeric_limits<TypeParam>::is_exact)
    {
    
        EXPECT_EQ(cv::saturate_cast<TypeParam>(max) , max);
        EXPECT_EQ(cv::saturate_cast<TypeParam>(min) , min );
		EXPECT_EQ(cv::saturate_cast<TypeParam>(0) , 0 );
		EXPECT_EQ(cv::saturate_cast<TypeParam>(1) , 1 );
		EXPECT_EQ(cv::saturate_cast<TypeParam>(2) , 2 );
        //EXPECT_EQ(cv::saturate_cast<TypeParam>(max+1) , max );
        EXPECT_EQ(cv::saturate_cast<TypeParam>(max-1), max - 1 );
        EXPECT_EQ(cv::saturate_cast<TypeParam>(min+1), min + 1 );
        //EXPECT_EQ(cv::saturate_cast<TypeParam>(min-1) , min );
    }
    else
    {
        min = -std::numeric_limits<TypeParam>::max(); //for float and double
        TypeParam epsilon = std::numeric_limits<TypeParam>::epsilon();
        EXPECT_NEAR(cv::saturate_cast<TypeParam>(max) , max, epsilon);
        EXPECT_NEAR(cv::saturate_cast<TypeParam>(min) , min, epsilon );
        //EXPECT_NEAR(cv::saturate_cast<TypeParam>(max+1) , max, epsilon );
        EXPECT_NEAR(cv::saturate_cast<TypeParam>(max-1), max - 1 , epsilon);
        EXPECT_NEAR(cv::saturate_cast<TypeParam>(min+1), min + 1, epsilon );
        //EXPECT_NEAR(cv::saturate_cast<TypeParam>(min-1) , min , epsilon);
		EXPECT_EQ(cv::saturate_cast<TypeParam>(0) , 0 );
		EXPECT_EQ(cv::saturate_cast<TypeParam>(1) , 1 );
		EXPECT_EQ(cv::saturate_cast<TypeParam>(2) , 2 );

		EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>( (ito::float64)std::numeric_limits<ito::float64>::max() ), std::numeric_limits<ito::float32>::max() );
        ito::float32 test = cv::saturate_cast<ito::float32>( -(ito::float64)std::numeric_limits<ito::float64>::min() );
		EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>( -(ito::float64)std::numeric_limits<ito::float64>::max() ), -std::numeric_limits<ito::float32>::max() );
    }
}


template <typename _Tp> class IsZeroValueRealTest : public ::testing::Test { };


TYPED_TEST_CASE(IsZeroValueRealTest, ItomRealDataTypes);

TYPED_TEST(IsZeroValueRealTest, checkZeroValueReal_Test)
{
	TypeParam ZeroVal= 0 ;
	TypeParam NonZeroVal1= 5;
	TypeParam NonZeroVal2= 23;
	TypeParam NonZeroVal3= 9;
	TypeParam NonZeroVal4= -4;
	TypeParam min;
	TypeParam max;

    if(std::numeric_limits<TypeParam>::is_exact)
    {
		EXPECT_TRUE( ito::isZeroValue<TypeParam>(ZeroVal,0) );
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal1,0) );
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal2,0) );
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal3,0) );
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal4,0) );
	}
    else
    {
		min= -std::numeric_limits<TypeParam>::epsilon();
		max= std::numeric_limits<TypeParam>::epsilon();

		EXPECT_TRUE( ito::isZeroValue<TypeParam>( ZeroVal , std::numeric_limits<TypeParam>::epsilon() ) );
		EXPECT_FALSE( ito::isZeroValue<TypeParam>( min, std::numeric_limits<TypeParam>::epsilon() ) );		//!< Note:------> Test is failing 
		EXPECT_FALSE( ito::isZeroValue<TypeParam>( max, std::numeric_limits<TypeParam>::epsilon() ) );		//!< Note:------> Test is failing
		EXPECT_FALSE( ito::isZeroValue<TypeParam>( min+1, std::numeric_limits<TypeParam>::epsilon() ) );
		EXPECT_FALSE( ito::isZeroValue<TypeParam>( max-1, std::numeric_limits<TypeParam>::epsilon() ) );
    }
}


template <typename _Tp> class IsZeroValueComplexTest : public ::testing::Test { };


TYPED_TEST_CASE(IsZeroValueComplexTest, ItomComplexDataTypes);

TYPED_TEST(IsZeroValueComplexTest, checkZeroValueComplex_Test)
{
	//!< Declaration for complex32 type variables.
	std::complex<ito::float32> ZeroVal1(0.0,0.0); 
	std::complex<ito::float32> NonZeroVal1(0.01,0.0);
	std::complex<ito::float32> NonZeroVal2(0.0,-0.05);
	std::complex<ito::float32> NonZeroVal3(-0.03,0.01);

	//!< Declaration for complex64 type variables.
	std::complex<ito::float64> ZeroVal2(0.0,0.0); 
	std::complex<ito::float64> NonZeroVal4(0.01,0.0);
	std::complex<ito::float64> NonZeroVal5(0.0,-0.05);
	std::complex<ito::float64> NonZeroVal6(-0.03,0.01);

	//!< Declaration for zero value variables. 
	std::complex<ito::float32> epsilon1(0.0,0.0) ;
	std::complex<ito::float64> epsilon2(0.0,0.0) ;
    if(std::numeric_limits<TypeParam>::is_exact)
    {

	}
    else
    {
		//!< Test fo isZeroValue() function.
        EXPECT_TRUE( ito::isZeroValue<TypeParam>(ZeroVal1, epsilon1 ) );			/*!< Test of isZeroValue() function for real part of Complex Type Variable with zero value.  */
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal1, epsilon1 ) );		/*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero positive value. */	
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal2, epsilon1) );			/*!< Test of isZeroValue() function for Complex Type variable with zero real part and nonzero negative imaginary part. */
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal3, epsilon1 ) );		/*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero negative value. */

		EXPECT_TRUE( ito::isZeroValue<TypeParam>(ZeroVal2, epsilon2 ) );			/*!< Test of isZeroValue() function for real part of Complex Type Variable with zero value.  */
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal4, epsilon2 ) );		/*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero positive value. */	
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal5, epsilon2) );			/*!< Test of isZeroValue() function for Complex Type variable with zero real part and nonzero negative imaginary part. */
		EXPECT_FALSE( ito::isZeroValue<TypeParam>(NonZeroVal6, epsilon2 ) );		/*!< Test of isZeroValue() function for real part of Complex Type Variable with nonzero negative value. */
    }
}

template <typename _Tp> class Real_ComplexTest : public ::testing::Test { };


TYPED_TEST_CASE(Real_ComplexTest, ItomDataTypes);

//! Test for convertCmplxTypeToRealType() function with different variables of different Datatypes.
TYPED_TEST(Real_ComplexTest, convertCmplxTypeToRealType_Test)
{
	//!< Declaration of different variables of different datatypes.
	ito::tDataType cmplxType1 = ito::tInt8;
	ito::tDataType cmplxType2 = ito::tInt16;
	ito::tDataType cmplxType3 = ito::tInt32;
	ito::tDataType cmplxType4 = ito::tUInt8;
	ito::tDataType cmplxType5 = ito::tUInt16;
	ito::tDataType cmplxType6 = ito::tFloat32;
	ito::tDataType cmplxType7 = ito::tFloat64;
	ito::tDataType cmplxType8 = ito::tComplex64;
	ito::tDataType cmplxType9 = ito::tComplex128;

	
	//!	 Test Details.

	/*!  ito::t(U)Int[8/16/32] must return the same type constants. 
		 ito::tFloat32/64 must return the same type constants.
		 ito::tComplex64 must return ito::tFloat32.
		 ito::tComplex128 must return ito::tFloat64. !
	*/
	EXPECT_EQ(cmplxType1, ito::convertCmplxTypeToRealType(cmplxType1) );	
	EXPECT_EQ(cmplxType2, ito::convertCmplxTypeToRealType(cmplxType2) );
	EXPECT_EQ(cmplxType3, ito::convertCmplxTypeToRealType(cmplxType3) );
	EXPECT_EQ(cmplxType4, ito::convertCmplxTypeToRealType(cmplxType4) );
	EXPECT_EQ(cmplxType5, ito::convertCmplxTypeToRealType(cmplxType5) );
	EXPECT_NE(cmplxType5, ito::convertCmplxTypeToRealType(cmplxType4) );	/*!< Testing if two datatype variables do not show equality in result. */
	EXPECT_NE(cmplxType4, ito::convertCmplxTypeToRealType(cmplxType1) );	/*!< Testing if two datatype variables do not show equality in result. */
	EXPECT_EQ(cmplxType6, ito::convertCmplxTypeToRealType(cmplxType6) );	/*!< Test for ito::tFloat32 type variable. */
	EXPECT_EQ(cmplxType7, ito::convertCmplxTypeToRealType(cmplxType7) );	/*!< Test for ito::tFloat64 type variable. */
	EXPECT_EQ(cmplxType6, ito::convertCmplxTypeToRealType(cmplxType8) );	/*!< Testing if the function returns ito::tFloat32 type for ito::tComplex64 type variable. */
	EXPECT_EQ(cmplxType7, ito::convertCmplxTypeToRealType(cmplxType9) );	/*!< Testing if the function returns ito::tFloat64 type for ito::tComplex128 type variable. */
}

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
	int16_var1 = ito::numberConversion<ito::int16>(ito::tInt8, &int8_var1);
	int16_var2 = ito::numberConversion<ito::int16>(ito::tInt8, &int8_var2);
	int16_var3 = ito::numberConversion<ito::int16>(ito::tInt8, &int8_var3);
	EXPECT_EQ( cv::saturate_cast<ito::int16>(5), int16_var1);
	EXPECT_EQ( cv::saturate_cast<ito::int16>(0), int16_var2);
	EXPECT_EQ( cv::saturate_cast<ito::int16>(-5), int16_var3);

	//!< Test for conversion from ito::int16 to ito::int8 using numberConversion<Type>(...) function.
	int8_var1,int8_var2,int8_var3 = 0;
	int16_var1 = 5;
	int16_var2 = 0;
	int16_var3 = -5;
	int8_var1 = ito::numberConversion<ito::int8>(ito::tInt16, &int16_var1);
	int8_var2 = ito::numberConversion<ito::int8>(ito::tInt16, &int16_var2);
	int8_var3 = ito::numberConversion<ito::int8>(ito::tInt16, &int16_var3);
	EXPECT_EQ( cv::saturate_cast<ito::int8>(5), int8_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int8>(0), int8_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int8>(-5), int8_var3 );

	//!< Test for conversion from ito::int16 to ito::int32 using numberConversion<Type>(...) function.
	int32_var1,int32_var2,int32_var3 = 0;
	int16_var1 = 5;
	int16_var2 = 0;
	int16_var3 = -5;
	int32_var1 = ito::numberConversion<ito::int32>(ito::tInt16, &int16_var1);
	int32_var2 = ito::numberConversion<ito::int32>(ito::tInt16, &int16_var2);
	int32_var3 = ito::numberConversion<ito::int32>(ito::tInt16, &int16_var3);
	EXPECT_EQ( cv::saturate_cast<ito::int32>(5), int32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(0), int32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(-5), int32_var3 );

		//!< Test for conversion from ito::int8 to ito::int32 using numberConversion<Type>(...) function.
	int32_var1,int32_var2,int32_var3 = 0;
	int8_var1 = 5;
	int8_var2 = 0;
	int8_var3 = -5;
	int32_var1 = ito::numberConversion<ito::int32>(ito::tInt8, &int8_var1);
	int32_var2 = ito::numberConversion<ito::int32>(ito::tInt8, &int8_var2);
	int32_var3 = ito::numberConversion<ito::int32>(ito::tInt8, &int8_var3);
	EXPECT_EQ( cv::saturate_cast<ito::int32>(5), int32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(0), int32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(-5), int32_var3 );

		//!< Test for conversion from ito::int32 to ito::int16 using numberConversion<Type>(...) function.
	int16_var1,int16_var2,int16_var3 = 0;
	int32_var1 = 5;
	int32_var2 = 0;
	int32_var3 = -5;
	int16_var1 = ito::numberConversion<ito::int16>(ito::tInt32, &int32_var1);
	int16_var2 = ito::numberConversion<ito::int16>(ito::tInt32, &int32_var2);
	int16_var3 = ito::numberConversion<ito::int16>(ito::tInt32, &int32_var3);
	EXPECT_EQ( cv::saturate_cast<ito::int16>(5), int16_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int16>(0), int16_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int16>(-5), int16_var3 );

	//!< Test for conversion from ito::int32 to ito::int8 using numberConversion<Type>(...) function.
	int8_var1,int8_var2,int8_var3 = 0;
	int16_var1 = 0;
	int16_var2 = 0;
	int16_var3 = 0;
	int8_var1 = ito::numberConversion<ito::int8>(ito::tInt32, &int32_var1);
	int8_var2 = ito::numberConversion<ito::int8>(ito::tInt32, &int32_var2);
	int8_var3 = ito::numberConversion<ito::int8>(ito::tInt32, &int32_var3);
	EXPECT_EQ( cv::saturate_cast<ito::int8>(5), int8_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int8>(0), int8_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int8>(-5), int8_var3 );

	//!< Test for conversion from ito::int8 to ito::float32 using numberConversion<Type>(...) function.
	float32_var1,float32_var2,float32_var3 = 0;
	int16_var1 = 0;
	int16_var2 = 0;
	int16_var3 = 0;
	float32_var1 = ito::numberConversion<ito::float32>(ito::tInt8, &int8_var1);
	float32_var2 = ito::numberConversion<ito::float32>(ito::tInt8, &int8_var2);
	float32_var3 = ito::numberConversion<ito::float32>(ito::tInt8, &int8_var3);
	EXPECT_EQ( cv::saturate_cast<ito::float32>(5), float32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::float32>(0), float32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::float32>(-5), float32_var3 );

	//!< Test for conversion from ito::int16 to ito::float32 using numberConversion<Type>(...) function.
	float32_var1,float32_var2,float32_var3 = 0;
	int16_var1 = 5;
	int16_var2 = 0;
	int16_var3 = -5;
	float32_var1 = ito::numberConversion<ito::float32>(ito::tInt16, &int16_var1);
	float32_var2 = ito::numberConversion<ito::float32>(ito::tInt16, &int16_var2);
	float32_var3 = ito::numberConversion<ito::float32>(ito::tInt16, &int16_var3);
	EXPECT_EQ( cv::saturate_cast<ito::float32>(5), float32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::float32>(0), float32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::float32>(-5), float32_var3 );

	//!< Test for conversion from ito::int32 to ito::float32 using numberConversion<Type>(...) function.
	float32_var1,float32_var2,float32_var3 = 0;
	int32_var1 = 5;
	int32_var2 = 0;
	int32_var3 = -5;
	float32_var1 = ito::numberConversion<ito::float32>(ito::tInt32, &int32_var1);
	float32_var2 = ito::numberConversion<ito::float32>(ito::tInt32, &int32_var2);
	float32_var3 = ito::numberConversion<ito::float32>(ito::tInt32, &int32_var3);
	EXPECT_EQ( cv::saturate_cast<ito::float32>(5), float32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::float32>(0), float32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::float32>(-5), float32_var3 );

	//!< Test for conversion from ito::float32 to ito::int32 using numberConversion<Type>(...) function.
	float32_var1 = -5.0;	/*!< Initializing a ito::float32 type variable with negative value.*/
	float32_var2 = -4.9;
	float32_var3 = -4.1;

	int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var1);
	int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var2);
	int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var3);

	EXPECT_EQ( cv::saturate_cast<ito::int32>(-5.0), int32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.9), int32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.1), int32_var3 );

	float32_var1 = 5.0;
	float32_var2 = 4.9;
	float32_var3 = 4.1;

	int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var1);
	int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var2);
	int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat32, &float32_var3);

	EXPECT_EQ( cv::saturate_cast<ito::int32>(5.0), int32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(4.9), int32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(4.1), int32_var3 );

	//!< Test for conversion from ito::float64 to ito::int32 using numberConversion<Type>(...) function.
	float64_var1 = -5.0;	/*!< Initializing a ito::float64 type variable with negative value.*/
	float64_var2 = -4.9;
	float64_var3 = -4.1;

	int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var1);
	int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var2);
	int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var3);

	EXPECT_EQ( cv::saturate_cast<ito::int32>(-5.0), int32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.9), int32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(-4.1), int32_var3 );

	float64_var1 = 5.0;
	float64_var2 = 4.9;
	float64_var3 = 4.1;

	int32_var1 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var1);
	int32_var2 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var2);
	int32_var3 = ito::numberConversion<ito::int32>(ito::tFloat64, &float64_var3);

	EXPECT_EQ( cv::saturate_cast<ito::int32>(5.0), int32_var1 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(4.9), int32_var2 );
	EXPECT_EQ( cv::saturate_cast<ito::int32>(4.1), int32_var3 );
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
	float64_var1 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var1);
	float64_var2 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var2);
	float64_var3 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var3);
	float64_var4 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var4);
	float64_var5 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var5);
	float64_var6 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var6);
	float64_var7 = ito::numberConversion<ito::float64>(ito::tFloat32, &float32_var7);

	//!< Test for conversion from ito::float32 to ito::float64.
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5.0), float64_var1);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.9), float64_var2);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.1), float64_var3);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), float64_var4);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.1), float64_var5);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.9), float64_var6);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5.0), float64_var7);

	float32_var1,float32_var2,float32_var3,float32_var4,float32_var5,float32_var6,float32_var7 = 0;

	//!< Converting ito::float64 type numbers of different critical values into ito::float32 type numbers.
	float32_var1 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var1);
	float32_var2 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var2);
	float32_var3 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var3);
	float32_var4 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var4);
	float32_var5 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var5);
	float32_var6 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var6);
	float32_var7 = ito::numberConversion<ito::float32>(ito::tFloat64, &float64_var7);

		//!< Test for conversion from ito::float64 to ito::float32.
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5.0), float32_var1);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.9), float32_var2);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.1), float32_var3);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), float32_var4);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.1), float32_var5);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.9), float32_var6);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5.0), float32_var7);
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

	complex64_var1 = ito::numberConversion<ito::complex64>(ito::tInt8,(void*)(&int8_var1) );	
	complex128_var1 = ito::numberConversion<ito::complex128>(ito::tInt8,(void*)(&int8_var1) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5), complex64_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5), complex128_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );

	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tInt8,(void*)(&int8_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tInt8,(void*)(&int8_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );

	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tInt8,(void*)(&int8_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tInt8,(void*)(&int8_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );

	int16_var1 = -5;
	int16_var2 = 0;
	int16_var3 = 5;
	complex64_var1,complex64_var2,complex64_var3 = 0 ;
	complex128_var1,complex128_var2,complex128_var3 = 0 ;

	complex64_var1 = ito::numberConversion<ito::complex64>(ito::tInt16, &int16_var1);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5), complex64_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );
	complex128_var1 = ito::numberConversion<ito::complex128>(ito::tInt16, &int16_var1);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5), complex128_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );

	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tInt16,(void*)(&int16_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tInt16,(void*)(&int16_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );

	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tInt16,(void*)(&int16_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tInt16,(void*)(&int16_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );

	int32_var1 = -5;
	int32_var2 = 0;
	int32_var3 = 5;
	complex64_var1,complex64_var2,complex64_var3 = 0 ;
	complex128_var1,complex128_var2,complex128_var3 = 0 ;

	complex64_var1 = ito::numberConversion<ito::complex64>(ito::tInt32, &int32_var1);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5), complex64_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );
	complex128_var1 = ito::numberConversion<ito::complex128>(ito::tInt32, &int32_var1);
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5), complex128_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );

	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tInt32,(void*)(&int32_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tInt32,(void*)(&int32_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );

	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tInt32,(void*)(&int32_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tInt32,(void*)(&int32_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );

	//!< Test for unsigned integer type variables.	
	uint8_var2 = 0;
	uint8_var3 = 5;

	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tUInt8,(void*)(&uint8_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tUInt8,(void*)(&uint8_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );

	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tUInt8,(void*)(&uint8_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tUInt8,(void*)(&uint8_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );

	uint16_var2 = 0;
	uint16_var3 = 5;
	complex64_var2,complex64_var3 = 0 ;
	complex128_var2,complex128_var3 = 0 ;

	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tUInt16,(void*)(&uint16_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tUInt16,(void*)(&uint16_var2) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );

	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tUInt16,(void*)(&uint16_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tUInt16,(void*)(&uint16_var3) );					
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );
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

	complex64_var1 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var1) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5.0), complex64_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );
	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var2) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.9), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var3) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.1), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex64_var4 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var4) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.imag() );
	complex64_var5 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var5) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.1), complex64_var5.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var5.imag() );
	complex64_var6 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var6) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.9), complex64_var6.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var6.imag() );
	complex64_var7 = ito::numberConversion<ito::complex64>(ito::tFloat32,(void*)(&float32_var7) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5.0), complex64_var7.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var7.imag() );


	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var1) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5.0), complex128_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var2) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.9), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var3) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.1), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );
	complex128_var4 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var4) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.imag() );
	complex128_var5 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var5) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.1), complex128_var5.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var5.imag() );
	complex128_var6 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var6) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.9), complex128_var6.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var6.imag() );
	complex128_var7 = ito::numberConversion<ito::complex128>(ito::tFloat32,(void*)(&float32_var7) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5.0), complex128_var7.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var7.imag() );


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

	complex64_var1 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var1) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-5.0), complex64_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var1.imag() );
	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var2) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.9), complex64_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var2.imag() );
	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var3) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(-4.1), complex64_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var3.imag() );
	complex64_var4 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var4) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var4.imag() );
	complex64_var5 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var5) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.1), complex64_var5.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var5.imag() );
	complex64_var6 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var6) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(4.9), complex64_var6.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var6.imag() );
	complex64_var7 = ito::numberConversion<ito::complex64>(ito::tFloat64,(void*)(&float64_var7) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(5.0), complex64_var7.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float32>(0), complex64_var7.imag() );


	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var1) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-5.0), complex128_var1.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var1.imag() );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var2) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.9), complex128_var2.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var2.imag() );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var3) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(-4.1), complex128_var3.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var3.imag() );
	complex128_var4 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var4) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var4.imag() );
	complex128_var5 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var5) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.1), complex128_var5.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var5.imag() );
	complex128_var6 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var6) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(4.9), complex128_var6.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var6.imag() );
	complex128_var7 = ito::numberConversion<ito::complex128>(ito::tFloat64,(void*)(&float64_var7) );	
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(5.0), complex128_var7.real() );
	EXPECT_FLOAT_EQ( cv::saturate_cast<ito::float64>(0), complex128_var7.imag() );
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

	complex128_var1 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var1) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var1), complex128_var1 );
	complex128_var2 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var2) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var2), complex128_var2 );
	complex128_var3 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var3) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var3), complex128_var3 );
	complex128_var4 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var4) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var4), complex128_var4 );
	complex128_var5 = ito::numberConversion<ito::complex128>(ito::tComplex64,(void*)(&complex64_var5) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex128>(complex64_var5), complex128_var5 );

	//!< Testing conversion from complex128 type variables to complex64 type variables.
	complex64_var1,complex64_var2,complex64_var3,complex64_var4,complex64_var5 = 0;	/*!< Defining complex64 type variables to 0 value. */
	complex128_var1 = (-5.0,5.0);													/*!< Defining complex128 type variables to different test values. */
	complex128_var2 = (-5.0,0);
	complex128_var3 = (0,-5.0);
	complex128_var4 = (0,5.0);
	complex128_var5 = (5.0,-5.0);

	complex64_var1 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var1) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var1), complex64_var1 );
	complex64_var2 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var2) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var2), complex64_var2 );
	complex64_var3 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var3) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var3), complex64_var3 );
	complex64_var4 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var4) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var4), complex64_var4 );
	complex64_var5 = ito::numberConversion<ito::complex64>(ito::tComplex128,(void*)(&complex128_var5) );	
	EXPECT_EQ( cv::saturate_cast<ito::complex64>(complex128_var5), complex64_var5 );

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


#endif