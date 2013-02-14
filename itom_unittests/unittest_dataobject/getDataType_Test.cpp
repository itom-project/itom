#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"

template <typename _Tp> class getDataTypeTest : public ::testing::Test { };

TYPED_TEST_CASE(getDataTypeTest, ItomDataTypes);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
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

	EXPECT_EQ( testType_var1, ito::getDataType( (const ito::int8 *) NULL )	);
	EXPECT_EQ( testType_var2, ito::getDataType( (const ito::int16 *) NULL )	);
	EXPECT_EQ( testType_var3, ito::getDataType( (const ito::int32 *) NULL )	);
	EXPECT_EQ( testType_var4, ito::getDataType( (const ito::uint8 *) NULL )	);
	EXPECT_EQ( testType_var5, ito::getDataType( (const ito::uint16 *) NULL ) );
	EXPECT_EQ( testType_var6, ito::getDataType( (const ito::float32 *) NULL ) );
	EXPECT_EQ( testType_var7, ito::getDataType( (const ito::float64 *) NULL ) );
	EXPECT_EQ( testType_var8, ito::getDataType( (const ito::complex64 *) NULL )	);
	EXPECT_EQ( testType_var9, ito::getDataType( (const ito::complex128 *) NULL ) );
}

//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
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

	EXPECT_EQ( testType_var4, ito::getDataType2<ito::uint8*>() );				
	EXPECT_EQ( testType_var2, ito::getDataType2<ito::int16 *>() );
	EXPECT_EQ( testType_var3, ito::getDataType2<ito::int32 *>() );
	EXPECT_EQ( testType_var4, ito::getDataType2<ito::uint8 *>() );
	EXPECT_EQ( testType_var5, ito::getDataType2<ito::uint16 *>() );
	EXPECT_EQ( testType_var6, ito::getDataType2<ito::float32 *>() );
	EXPECT_EQ( testType_var7, ito::getDataType2<ito::float64 *>() );
	EXPECT_EQ( testType_var8, ito::getDataType2<ito::complex64 *>() );
	EXPECT_EQ( testType_var9, ito::getDataType2<ito::complex128 *>() );
}
