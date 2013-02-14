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



/*! \class dataTest
	\brief data test for all itom data types

	This test class checks functionality of different data test methods for data objects.
*/
template <typename _Tp> class TestTest : public ::testing::Test 
	{ 
public:

	
	 ito::DataObject matrixTest;


    virtual void SetUp(void)
    {
		
	};
	 virtual void TearDown(void) {};
	  typedef _Tp valueType;	
	};

TYPED_TEST_CASE(TestTest, ItomDataTypes);

//checkZeros
/*!
	This test checks functionality of "zeros" method for 1, 2 and 3 dimensional matrices by checking if the required matrix is zero matrix.
*/
TYPED_TEST(TestTest, TestTest1)
{
	int i=1;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




/*! \class SaturateTestReal
	\brief saturation test for real data types

	This test class tests saturation limits for real data types
*/
template <typename _Tp> class SaturateTestInt : public ::testing::Test { };


TYPED_TEST_CASE(SaturateTestInt, ItomIntDataTypes2);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestInt, saturate_cast_Test)
{
    TypeParam int_var1 = -5;
    TypeParam int_var2 = 0;
	TypeParam int_var3 = 5;
   
	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var1) , int_var1);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var2) , int_var2);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var3) , int_var3);
}

/*! \class SaturateTestReal
	\brief saturation test for real data types

	This test class tests saturation limits for real data types
*/
template <typename _Tp> class SaturateTestUInt : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestUInt, ItomUIntDataTypes);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestUInt, saturate_cast_Test)
{
    TypeParam uint_var1 = -5;
    TypeParam uint_var2 = 0;
	TypeParam uint_var3 = 5;
   
	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var1) , uint_var1);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var2) , uint_var2);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var3) , uint_var3);
}

/*! \class SaturateTestReal
	\brief saturation test for real data types

	This test class tests saturation limits for real data types
*/
template <typename _Tp> class SaturateTestFloat : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestFloat, ItomFloatDoubleDataTypes);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestFloat, saturate_cast_Test)
{
    TypeParam float_var1 = -5.0;
    TypeParam float_var2 = -4.9;
	TypeParam float_var3 = -4.1;
	TypeParam float_var4 = 0;
	TypeParam float_var5 = 4.1;
	TypeParam float_var6 = 4.9;
	TypeParam float_var7 = 5.0;
	ito::float32 float32_inf = std::numeric_limits<ito::float32>::infinity();
	ito::float64 float64_inf = std::numeric_limits<ito::float64>::infinity();
	ito::float32 float32_sigNan = std::numeric_limits<ito::float32>::signaling_NaN();
	ito::float32 float32_qNan = std::numeric_limits<ito::float32>::quiet_NaN();
	ito::float64 float64_sigNan = std::numeric_limits<ito::float64>::signaling_NaN();
	ito::float64 float64_qNan = std::numeric_limits<ito::float64>::quiet_NaN();
	
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var1) , float_var1);
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var2) , float_var2);
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var3) , float_var3);
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var4) , float_var4);
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var5) , float_var5);
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var6) , float_var6);
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var7) , float_var7);

	EXPECT_FLOAT_EQ(cv::saturate_cast<ito::float32>(float64_inf),float32_inf);						/*!< Testing if float64 infinity variable becomes same as float32 infinity variable after converting with saturate_cast<Type>(...) function. */

	ito::float32 float32_test1 = cv::saturate_cast<ito::float32>(float64_sigNan);
	EXPECT_EQ(0,std::memcmp(&float32_sigNan,&float32_test1,sizeof(float32_sigNan) ) );				/*!< Testing if float64 signalNaN variable becomes same as float32 signalNaN variable after converting with saturate_cast<Type>(...) function. */

	ito::float32 float32_test2 = cv::saturate_cast<ito::float32>(float64_qNan);			
	EXPECT_EQ(0,std::memcmp(&float32_qNan,&float32_test2,sizeof(float32_qNan) ) );		               //NOTE:: This test is failing for qNan value.................


	EXPECT_FLOAT_EQ(cv::saturate_cast<ito::float32>(float32_inf),float64_inf);

	ito::float64 float64_test1 = cv::saturate_cast<ito::float64>(float32_sigNan);
	EXPECT_EQ(0,std::memcmp(&float64_sigNan,&float64_test1,sizeof(float64_sigNan) ) );

	ito::float64 float64_test2 = cv::saturate_cast<ito::float64>(float32_qNan);
	EXPECT_EQ(0,std::memcmp(&float64_qNan,&float64_test2,sizeof(float64_qNan) ) );				      //NOTE:: This test is failing for qNan value................
}

template <typename _Tp> class SaturateTestFloat_Int : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestFloat_Int, ItomIntDataTypes2);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestFloat_Int, saturate_cast_Test)
{
	ito::float32 float_var1 = -5.0;
    ito::float32 float_var2 = -4.9;
	ito::float32 float_var3 = -4.1;
	ito::float32 float_var4 = 0;
	ito::float32 float_var5 = 4.1;
	ito::float32 float_var6 = 4.9;
	ito::float32 float_var7 = 5.0;
    ito::float32 float32_inf = std::numeric_limits<ito::float32>::infinity();
	ito::float64 float64_inf = std::numeric_limits<ito::float64>::infinity();
	ito::float32 float32_sigNan = std::numeric_limits<ito::float32>::signaling_NaN();
	ito::float32 float32_qNan = std::numeric_limits<ito::float32>::quiet_NaN();
	ito::float64 float64_sigNan = std::numeric_limits<ito::float64>::signaling_NaN();
	ito::float64 float64_qNan = std::numeric_limits<ito::float64>::quiet_NaN();
	

	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var1) , -5);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var2) , -5);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var3) , -4);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var4) , 0);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var5) , 4);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var6) , 5);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var7) , 5);
	TypeParam Param_test1 = cv::saturate_cast<TypeParam>(float32_inf);
	TypeParam Param_max1 = std::numeric_limits<TypeParam>::min();
	EXPECT_EQ(Param_max1 ,  Param_test1 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead...................

	TypeParam Param_test2 = cv::saturate_cast<TypeParam>(float32_sigNan);
	TypeParam Param_max2 = std::numeric_limits<TypeParam>::min();
	EXPECT_EQ(Param_max2, Param_test2 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................

	TypeParam Param_test3 = cv::saturate_cast<TypeParam>(float32_qNan);
	TypeParam Param_max3 = std::numeric_limits<TypeParam>::min();
	EXPECT_EQ(Param_max3, Param_test3 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................
}

/////////////////////////////////////////////////////////////////////////

template <typename _Tp> class SaturateTestFloat_UInt : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestFloat_UInt, ItomUIntDataTypes);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestFloat_UInt, saturate_cast_Test)
{
	ito::float32 float_var1 = -5.0;
    ito::float32 float_var2 = -4.9;
	ito::float32 float_var3 = -4.1;
	ito::float32 float_var4 = 0;
	ito::float32 float_var5 = 4.1;
	ito::float32 float_var6 = 4.9;
	ito::float32 float_var7 = 5.0;
	ito::float32 float_var8 = 258;
	ito::float32 float_var9 = 65538;

	ito::float32 float32_inf = std::numeric_limits<ito::float32>::infinity();
	ito::float64 float64_inf = std::numeric_limits<ito::float64>::infinity();
	ito::float32 float32_sigNan = std::numeric_limits<ito::float32>::signaling_NaN();
	ito::float32 float32_qNan = std::numeric_limits<ito::float32>::quiet_NaN();
	ito::float64 float64_sigNan = std::numeric_limits<ito::float64>::signaling_NaN();
	ito::float64 float64_qNan = std::numeric_limits<ito::float64>::quiet_NaN();

	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var1) , 0);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var2) , 0);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var3) , 0);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var4) , 0);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var5) , 4);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var6) , 5);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var7) , 5);
	EXPECT_EQ(cv::saturate_cast<ito::uint8>(float_var8),std::numeric_limits<ito::uint8>::max());
	EXPECT_EQ(cv::saturate_cast<ito::uint16>(float_var9),std::numeric_limits<ito::uint16>::max());

	TypeParam Param_test1 = cv::saturate_cast<TypeParam>(float32_inf);
	TypeParam Param_max1 = std::numeric_limits<TypeParam>::min();
	EXPECT_EQ(Param_max1 ,  Param_test1 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................

	TypeParam Param_test2 = cv::saturate_cast<TypeParam>(float32_sigNan);
	TypeParam Param_max2 = std::numeric_limits<TypeParam>::min();
	EXPECT_EQ(Param_max2, Param_test2 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................

	TypeParam Param_test3 = cv::saturate_cast<TypeParam>(float32_qNan);
	TypeParam Param_max3 = std::numeric_limits<TypeParam>::min();
	EXPECT_EQ(Param_max3, Param_test3 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................
}

////////////////////////////////////////////////////////////////////////

template <typename _Tp> class SaturateTestCmplx_NonCmplx : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestCmplx_NonCmplx, ItomRealDataTypes);
//! checkSaturateBoundaries
/*!
	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
*/
TYPED_TEST(SaturateTestCmplx_NonCmplx, saturate_cast_Test)
{
	TypeParam Real_var1 = 5;
    ito::complex64 complex_var1 ;
	ito::complex128 complex_var2 ;
	ito::float32 float32_var1 = -4.9;

    complex_var1 = cv::saturate_cast<TypeParam>(Real_var1) ;
	EXPECT_EQ(Real_var1 , complex_var1.real() );
	EXPECT_EQ(0 , complex_var1.imag() );

	complex_var2 = cv::saturate_cast<TypeParam>(float32_var1);
	EXPECT_EQ(cv::saturate_cast<TypeParam>(-4.9), complex_var2.real() );
	EXPECT_EQ(0,complex_var2.imag() );										//Note:: Test is failing only for double datatype...............

	//!< Testing if exception is raised as it is supposed to, while converting from Complex datatype variables to any other type variables.
	EXPECT_ANY_THROW(cv::saturate_cast<TypeParam>(complex_var1));
	
}

