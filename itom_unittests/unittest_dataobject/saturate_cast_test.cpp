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



//! SaturateTestInt
/*!
	This test checks if the saturate_cast<type>(...) function returns the same value for the same signed fixed point data type.
*/
template <typename _Tp> class SaturateTestInt : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestInt, ItomIntDataTypes2);

TYPED_TEST(SaturateTestInt, saturate_cast_Test)
{
	//!< declaring variables of signed fixed point types.
    TypeParam int_var1 = -5;
    TypeParam int_var2 = 0;
	TypeParam int_var3 = 5;
   
	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var1) , int_var1);		//!< checking if saturate_cast<type>(...) function returns the same value for conversion into the same fixed point signed data type as original.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var2) , int_var2);		//!< checking if saturate_cast<type>(...) function returns the same value for conversion into the same fixed point signed data type as original.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var3) , int_var3);		//!< checking if saturate_cast<type>(...) function returns the same value for conversion into the same fixed point signed data type as original.
}

//! SaturateTestUInt
/*!
	This test checks if the saturate_cast<type>(...) function returns the same value for the same unsigned fixed point data type.
*/
template <typename _Tp> class SaturateTestUInt : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestUInt, ItomUIntDataTypes);

TYPED_TEST(SaturateTestUInt, saturate_cast_Test)
{
	//!< declaring variables of unsigned fixed point types.
    TypeParam uint_var1 = -5;
    TypeParam uint_var2 = 0;
	TypeParam uint_var3 = 5;
   
	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var1) , uint_var1);		//!< checking if saturate_cast<type>(...) function returns the same value for conversion into the same fixed point unsigned data type as original.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var2) , uint_var2);		//!< checking if saturate_cast<type>(...) function returns the same value for conversion into the same fixed point unsigned data type as original.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var3) , uint_var3);		//!< checking if saturate_cast<type>(...) function returns the same value for conversion into the same fixed point unsigned data type as original.
}

//! SaturateTestFloat
/*!
	This test checks the functionality of saturate_cast<type>(...) method while converting from float32 and float64 into float32/64 data types.
*/
template <typename _Tp> class SaturateTestFloat : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestFloat, ItomFloatDoubleDataTypes);

TYPED_TEST(SaturateTestFloat, saturate_cast_Test)
{
    TypeParam float_var1 = -5.0;
    TypeParam float_var2 = -4.9;
	TypeParam float_var3 = -4.1;
	TypeParam float_var4 = 0;
	TypeParam float_var5 = 4.1;
	TypeParam float_var6 = 4.9;
	TypeParam float_var7 = 5.0;
	ito::float32 float32_inf = std::numeric_limits<ito::float32>::infinity();					//!< a float32 variable with a special value infinity()
	ito::float64 float64_inf = std::numeric_limits<ito::float64>::infinity();					//!< a float64 point variable with a special value infinity()
	ito::float32 float32_sigNan = std::numeric_limits<ito::float32>::signaling_NaN();			//!< a float32 variable with a special value signaling_NaN()
	ito::float32 float32_qNan = std::numeric_limits<ito::float32>::quiet_NaN();					//!< a float32 variable with a special value quiet_NaN()
	ito::float64 float64_sigNan = std::numeric_limits<ito::float64>::signaling_NaN();			//!< a float64 variable with a special value signaling_NaN()
	ito::float64 float64_qNan = std::numeric_limits<ito::float64>::quiet_NaN();					//!< a float64 variable with a special value quiet_NaN()
		
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var1) , float_var1);						//!< while converting from float32/64 into the same data type must retain the original value
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var2) , float_var2);						//!< while converting from float32/64 into the same data type must retain the original value
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var3) , float_var3);						//!< while converting from float32/64 into the same data type must retain the original value
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var4) , float_var4);						//!< while converting from float32/64 into the same data type must retain the original value
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var5) , float_var5);						//!< while converting from float32/64 into the same data type must retain the original value
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var6) , float_var6);						//!< while converting from float32/64 into the same data type must retain the original value
	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var7) , float_var7);						//!< while converting from float32/64 into the same data type must retain the original value

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

//! saturate_cast_Test1
/*!
	This test checks the functionality of saturate_cast<type>(...) method while converting from float64 into float32 for values which exceeds the maximum value of float32 type variable.
*/
TYPED_TEST(SaturateTestFloat, saturate_cast_Test1)
{
	ito::float64 float64_max = std::numeric_limits<ito::float64>::max();		//!< maximum value of data type float64
	ito::float32 float32_max = std::numeric_limits<ito::float32>::max();		//!< maximum value of data type float32
	EXPECT_FLOAT_EQ(cv::saturate_cast<ito::float32>(float64_max), float32_max);	//!< while converting from float64 into float32, saturate_cast<type>(...) function must return maximum limit for float32 data type.
}

//! SaturateTestFloat_Int
/*!
	This test checks the functionality of saturate_cast<type>(...) method while converting from floating point data types into signed fixed point data types.
*/
template <typename _Tp> class SaturateTestFloat_Int : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestFloat_Int, ItomIntDataTypes2);

TYPED_TEST(SaturateTestFloat_Int, saturate_cast_Test)
{
	//!< Declaring floating point variables to be used in this test.
	ito::float32 float_var1 = -5.0;
    ito::float32 float_var2 = -4.9;
	ito::float32 float_var3 = -4.1;
	ito::float32 float_var4 = 0;
	ito::float32 float_var5 = 4.1;
	ito::float32 float_var6 = 4.9;
	ito::float32 float_var7 = 5.0;
    ito::float32 float32_inf = std::numeric_limits<ito::float32>::infinity();				//!< a float32 variable with a special value infinity()
	ito::float64 float64_inf = std::numeric_limits<ito::float64>::infinity();				//!< a float64 point variable with a special value infinity()
	ito::float32 float32_sigNan = std::numeric_limits<ito::float32>::signaling_NaN();		//!< a float32 variable with a special value signaling_NaN()
	ito::float32 float32_qNan = std::numeric_limits<ito::float32>::quiet_NaN();				//!< a float32 variable with a special value quiet_NaN()
	ito::float64 float64_sigNan = std::numeric_limits<ito::float64>::signaling_NaN();		//!< a float64 variable with a special value signaling_NaN()
	ito::float64 float64_qNan = std::numeric_limits<ito::float64>::quiet_NaN();				//!< a float64 variable with a special value quiet_NaN()
	

	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var1) , -5);				//!< conversion from floating point value -5.0 ( using saturate_cast(...) method )to signed fixed type value should return -5.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var2) , -5);				//!< conversion from floating point value -4.9 ( using saturate_cast(...) method )to signed fixed type value should return -4.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var3) , -4);				//!< conversion from floating point value -4.1 ( using saturate_cast(...) method )to signed fixed type value should return -4.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var4) , 0);				//!< conversion from floating point value 0 ( using saturate_cast(...) method )to signed fixed type value should return 0.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var5) , 4);				//!< conversion from floating point value 4.1 ( using saturate_cast(...) method )to signed fixed type value should return 4.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var6) , 5);				//!< conversion from floating point value 4.9 ( using saturate_cast(...) method )to signed fixed type value should return 4.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var7) , 5);				//!< conversion from floating point value 5.0 ( using saturate_cast(...) method )to signed fixed type value should return 5.
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

//! SaturateTestFloat_UInt
/*!
	This test checks the functionality of saturate_cast<type>(...) method while converting from floating point data types into unsigned fixed point data types.
*/
template <typename _Tp> class SaturateTestFloat_UInt : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestFloat_UInt, ItomUIntDataTypes);

TYPED_TEST(SaturateTestFloat_UInt, saturate_cast_Test)
{
	ito::float32 float_var1 = -5.0;
    ito::float32 float_var2 = -4.9;
	ito::float32 float_var3 = -4.1;
	ito::float32 float_var4 = 0;
	ito::float32 float_var5 = 4.1;
	ito::float32 float_var6 = 4.9;
	ito::float32 float_var7 = 5.0;
	ito::float32 float_var8 = 258;		//!< declaring float32 type variable with value greater than maximum value for uint8 type variable for test purpose
	ito::float32 float_var9 = 65538;	//!< declaring float32 type variable with value greater than maximum value for uint16 type variable for test purpose

	ito::float32 float32_inf = std::numeric_limits<ito::float32>::infinity();			//!< a float32 variable with a special value infinity()
	ito::float64 float64_inf = std::numeric_limits<ito::float64>::infinity();			//!< a float64 point variable with a special value infinity()
	ito::float32 float32_sigNan = std::numeric_limits<ito::float32>::signaling_NaN();	//!< a float32 variable with a special value signaling_NaN()
	ito::float32 float32_qNan = std::numeric_limits<ito::float32>::quiet_NaN();			//!< a float32 variable with a special value quiet_NaN()
	ito::float64 float64_sigNan = std::numeric_limits<ito::float64>::signaling_NaN();	//!< a float64 variable with a special value signaling_NaN()
	ito::float64 float64_qNan = std::numeric_limits<ito::float64>::quiet_NaN();			//!< a float64 variable with a special value quiet_NaN()

	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var1) , 0);				//!< conversion from floating point value -5.0 ( using saturate_cast(...) method )to unsigned fixed type value should return 0.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var2) , 0);				//!< conversion from floating point value -4.9 ( using saturate_cast(...) method )to unsigned fixed type value should return 0.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var3) , 0);				//!< conversion from floating point value -4.1 ( using saturate_cast(...) method )to unsigned fixed type value should return 0.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var4) , 0);				//!< conversion from floating point value 0 ( using saturate_cast(...) method )to unsigned fixed type value should return 0.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var5) , 4);				//!< conversion from floating point value 4.1 ( using saturate_cast(...) method )to unsigned fixed type value should return 4.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var6) , 5);				//!< conversion from floating point value 4.9 ( using saturate_cast(...) method )to unsigned fixed type value should return 4.
	EXPECT_EQ(cv::saturate_cast<TypeParam>(float_var7) , 5);				//!< conversion from floating point value 5.0 ( using saturate_cast(...) method )to unsigned fixed type value should return 5.
	EXPECT_EQ(cv::saturate_cast<ito::uint8>(float_var8),std::numeric_limits<ito::uint8>::max());	//!< conversion from floating point variable greater than maximum limit of fixed point variable value of data type under test must return maximum limit value of that fixed point data type under test 
	EXPECT_EQ(cv::saturate_cast<ito::uint16>(float_var9),std::numeric_limits<ito::uint16>::max());	//!< conversion from floating point variable greater than maximum limit of fixed point variable value of data type under test must return maximum limit value of that fixed point data type under test 

	TypeParam Param_test1 = cv::saturate_cast<TypeParam>(float32_inf);		//!< converting special value infinity() variable into unsigned fixed point type variable using saturate_cast(...) method.
	TypeParam Param_max1 = std::numeric_limits<TypeParam>::min();			//!< Expected maximum value of the data type under test.
	EXPECT_EQ(Param_max1 ,  Param_test1 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................

	TypeParam Param_test2 = cv::saturate_cast<TypeParam>(float32_sigNan);	//!< converting special value signaling_NaN() variable into unsigned fixed point type variable using saturate_cast(...) method.
	TypeParam Param_max2 = std::numeric_limits<TypeParam>::min();			//!< Expected maximum value of the data type under test.
	EXPECT_EQ(Param_max2, Param_test2 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................

	TypeParam Param_test3 = cv::saturate_cast<TypeParam>(float32_qNan);		//!< converting special value quiet_NaN() variable into unsigned fixed point type variable using saturate_cast(...) method.
	TypeParam Param_max3 = std::numeric_limits<TypeParam>::min();			//!< Expected maximum value of the data type under test.
	EXPECT_EQ(Param_max3, Param_test3 );									//Note:: Test is failing. According to document, saturate_cast() must return max value of that datatype but it is returning min value of that datatype instead..................
}

//! SaturateTestCmplx_NonCmplx
/*!
	This test checks the functionality of saturate_cast<type>(...) method while converting from non-complex data type into complex data type and vice versa.
*/
template <typename _Tp> class SaturateTestCmplx_NonCmplx : public ::testing::Test { };

TYPED_TEST_CASE(SaturateTestCmplx_NonCmplx, ItomRealDataTypes);

TYPED_TEST(SaturateTestCmplx_NonCmplx, saturate_cast_Test)
{
	TypeParam Real_var1 = 5;												//!< declaring non-complex data type variable.
    ito::complex64 complex_var1 ;											//!< declaring complex64 type variable.
	ito::complex128 complex_var2 ;											//!< declaring complex128 type variable.
	ito::float32 float32_var1 = -4.9;										//!< declaring float32 type variable.

    complex_var1 = cv::saturate_cast<TypeParam>(Real_var1) ;				//!< converting non-complex type value into complex64 type 
	EXPECT_EQ(Real_var1 , complex_var1.real() );							//!< check if real part of complex64 type variable is same as assigned non-complex value 
	EXPECT_EQ(0 , complex_var1.imag() );									//!< check if real part of complex64 type variable is 0.

	complex_var2 = cv::saturate_cast<TypeParam>(float32_var1);				//!< converting float32 type value into complex128 type
	EXPECT_EQ(cv::saturate_cast<TypeParam>(-4.9), complex_var2.real() );	//!< check if real part of complex128 type variable is same as assigned float32 type value
	EXPECT_EQ(0,complex_var2.imag() );										//Note:: Test is failing only for double datatype...............

	//!< Testing if exception is raised as it is supposed to, while converting from Complex datatype variables to any other type variables.
	EXPECT_ANY_THROW(cv::saturate_cast<TypeParam>(complex_var1));	
}

