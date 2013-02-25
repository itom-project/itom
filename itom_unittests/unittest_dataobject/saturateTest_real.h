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






#endif