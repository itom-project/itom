#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"

/*! \class getDataTypetest
	\brief Test for getDataType and getDataType2 methods for all real data types

	This test contains 2 test cases for getDataType and getDataType2 functions each and checks the outputs of them when they are applied with respective data types.
*/
template <typename _Tp> class getDataTypeTest : public ::testing::Test { };

TYPED_TEST_CASE(getDataTypeTest, ItomDataAllTypes);

//! getDataType_Test
/*!
	This test checks the functionality of getDataType() function with different supported data types.
*/
TYPED_TEST(getDataTypeTest, getDataType_Test)
{
	ito::tDataType testType_var1 = ito::tInt8;
	ito::tDataType testType_var2 = ito::tInt16;
	ito::tDataType testType_var3 = ito::tInt32;
	ito::tDataType testType_var4 = ito::tUInt8;
	ito::tDataType testType_var5 = ito::tUInt16;
	ito::tDataType testType_var6 = ito::tFloat32;
	ito::tDataType testType_var7 = ito::tFloat64;
	ito::tDataType testType_var8 = ito::tComplex64;
	ito::tDataType testType_var9 = ito::tComplex128;
    ito::tDataType testType_var10 = ito::tRGBA32;

	//!< the following tests check if the getDataType() method returns the correct expected data types	
	EXPECT_EQ( testType_var1, ito::getDataType( (const ito::int8 *) NULL )	);
	EXPECT_EQ( testType_var2, ito::getDataType( (const ito::int16 *) NULL )	);
	EXPECT_EQ( testType_var3, ito::getDataType( (const ito::int32 *) NULL )	);
	EXPECT_EQ( testType_var4, ito::getDataType( (const ito::uint8 *) NULL )	);
	EXPECT_EQ( testType_var5, ito::getDataType( (const ito::uint16 *) NULL ) );
	EXPECT_EQ( testType_var6, ito::getDataType( (const ito::float32 *) NULL ) );
	EXPECT_EQ( testType_var7, ito::getDataType( (const ito::float64 *) NULL ) );
	EXPECT_EQ( testType_var8, ito::getDataType( (const ito::complex64 *) NULL )	);
	EXPECT_EQ( testType_var9, ito::getDataType( (const ito::complex128 *) NULL ) );
    EXPECT_EQ( testType_var10, ito::getDataType( (const ito::Rgba32 *) NULL ) );
}

//! getDataType2
/*!
	This test checks the functionality of getDataType2() function with different supported data types.
*/
TYPED_TEST(getDataTypeTest, getDataType2_Test)
{
	ito::tDataType testType_var1 = ito::tInt8;
	ito::tDataType testType_var2 = ito::tInt16;
	ito::tDataType testType_var3 = ito::tInt32;
	ito::tDataType testType_var4 = ito::tUInt8;
	ito::tDataType testType_var5 = ito::tUInt16;
	ito::tDataType testType_var6 = ito::tFloat32;
	ito::tDataType testType_var7 = ito::tFloat64;
	ito::tDataType testType_var8 = ito::tComplex64;
	ito::tDataType testType_var9 = ito::tComplex128;
    ito::tDataType testType_var10 = ito::tRGBA32;

	//!< the following tests check if the getDataType2() method returns the correct expected data types		
	EXPECT_EQ( testType_var4, ito::getDataType2<ito::uint8*>() );				
	EXPECT_EQ( testType_var2, ito::getDataType2<ito::int16 *>() );
	EXPECT_EQ( testType_var3, ito::getDataType2<ito::int32 *>() );
	EXPECT_EQ( testType_var4, ito::getDataType2<ito::uint8 *>() );
	EXPECT_EQ( testType_var5, ito::getDataType2<ito::uint16 *>() );
	EXPECT_EQ( testType_var6, ito::getDataType2<ito::float32 *>() );
	EXPECT_EQ( testType_var7, ito::getDataType2<ito::float64 *>() );
	EXPECT_EQ( testType_var8, ito::getDataType2<ito::complex64 *>() );
	EXPECT_EQ( testType_var9, ito::getDataType2<ito::complex128 *>() );
    EXPECT_EQ( testType_var10, ito::getDataType2<ito::Rgba32 *>() );
}
