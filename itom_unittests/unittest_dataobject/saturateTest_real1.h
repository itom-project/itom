//#ifndef SATURATETEST_REAL1_H
//#define SATURATETEST_REAL1_H
//
//#include <iostream>
//
//#include "../../Common/sharedStructures.h"
//
////opencv
//#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
//#pragma once
//#include "opencv/cv.h"
//#include "../../DataObject/dataobj.h"
//#include "gtest/gtest.h"
//#include "commonChannel.h"
//
///*! \class SaturateTestReal
//	\brief saturation test for real data types
//
//	This test class tests saturation limits for real data types
//*/
//template <typename _Tp> class SaturateTestInt : public ::testing::Test { };
//
//
//TYPED_TEST_CASE(SaturateTestInt, ItomIntDataTypes2);
////! checkSaturateBoundaries
///*!
//	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
//*/
//TYPED_TEST(SaturateTestInt, saturate_cast_Test)
//{
//    TypeParam int_var1 = -5;
//    TypeParam int_var2 = 0;
//	TypeParam int_var3 = 5;
//   
//	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var1) , int_var1);
//	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var2) , int_var2);
//	EXPECT_EQ(cv::saturate_cast<TypeParam>(int_var3) , int_var3);
//}
//
///*! \class SaturateTestReal
//	\brief saturation test for real data types
//
//	This test class tests saturation limits for real data types
//*/
//template <typename _Tp> class SaturateTestUInt : public ::testing::Test { };
//
//TYPED_TEST_CASE(SaturateTestUInt, ItomUIntDataTypes);
////! checkSaturateBoundaries
///*!
//	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
//*/
//TYPED_TEST(SaturateTestUInt, saturate_cast_Test)
//{
//    TypeParam uint_var1 = -5;
//    TypeParam uint_var2 = 0;
//	TypeParam uint_var3 = 5;
//   
//	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var1) , uint_var1);
//	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var2) , uint_var2);
//	EXPECT_EQ(cv::saturate_cast<TypeParam>(uint_var3) , uint_var3);
//}
//
///*! \class SaturateTestReal
//	\brief saturation test for real data types
//
//	This test class tests saturation limits for real data types
//*/
//template <typename _Tp> class SaturateTestFloat : public ::testing::Test { };
//
//TYPED_TEST_CASE(SaturateTestFloat, ItomFloatDoubleDataTypes);
////! checkSaturateBoundaries
///*!
//	This test checks if the saturation limits for for all real data types as defined in std library and openCv library match or not .
//*/
//TYPED_TEST(SaturateTestFloat, saturate_cast_Test)
//{
//    TypeParam float_var1 = -5.0;
//    TypeParam float_var2 = -4.9;
//	TypeParam float_var3 = -4.1;
//	TypeParam float_var4 = 0;
//	TypeParam float_var5 = 4.1;
//	TypeParam float_var6 = 4.9;
//	TypeParam float_var7 = 5.0;
//   
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var1) , float_var1);
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var2) , float_var2);
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var3) , float_var3);
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var4) , float_var4);
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var5) , float_var5);
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var6) , float_var6);
//	EXPECT_FLOAT_EQ(cv::saturate_cast<TypeParam>(float_var7) , float_var7);
//}
//#endif