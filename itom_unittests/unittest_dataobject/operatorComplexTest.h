#ifndef OPERATORCOMPLEXTEST_H
#define OPERATORCOMPLEXTEST_H
#include <iostream>

#include "../../common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv/cv.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class ROITest
    \brief ROI methods test for real data types

    This test class checks functionality of different methods dealing with ROI for data objects.
*/
template <typename _Tp> class operatorComplexTest : public ::testing::Test 
    { 
public:

    
    virtual void SetUp(void)
    {
        mat1_1d = ito::DataObject(2,getTypeNumber<_Tp>());
        mat2_1d = ito::DataObject(2,getTypeNumber<_Tp>());
        mat3_1d = ito::DataObject(2,getTypeNumber<_Tp>());

        mat1_2d = ito::DataObject(3,3,getTypeNumber<_Tp>());
        mat2_2d = ito::DataObject(3,3,getTypeNumber<_Tp>());
        mat3_2d = ito::DataObject(3,3,getTypeNumber<_Tp>());

        mat1_3d = ito::DataObject(3,3,3,getTypeNumber<_Tp>());    
        mat2_3d = ito::DataObject(3,3,3,getTypeNumber<_Tp>());
        mat3_3d = ito::DataObject(3,3,3,getTypeNumber<_Tp>());
    };
 

    virtual void TearDown(void) {};
    typedef _Tp valueType;

    
    ito::DataObject mat1_1d;
    ito::DataObject mat2_1d;
    ito::DataObject mat3_1d;

    ito::DataObject mat1_2d;
    ito::DataObject mat2_2d;
    ito::DataObject mat3_2d;

    ito::DataObject mat1_3d;
    ito::DataObject mat2_3d;
    ito::DataObject mat3_3d;

    };
    

TYPED_TEST_CASE(operatorComplexTest, ItomComplexDataTypes);

//adjustROITest2d
/*!
    This test adjust the ROI of 2 dimensional matrices to check proper functionality of "adjustROI" method. It also checks "locateROI" method by comparing achieved offsets with original values.
*/
TYPED_TEST(operatorComplexTest, complexDivTest)
{
    mat1_2d = cv::saturate_cast<TypeParam>(TypeParam(4,6));
    mat2_2d = cv::saturate_cast<TypeParam>(TypeParam(1,1));
    //std::cout << "mat2:  " << mat2 << std::endl;
    mat3_2d = cv::saturate_cast<TypeParam>(TypeParam(1,1));

    //divFunc(mat1, mat2, mat3);
    mat3_2d=mat1_2d.div(mat2_2d);
    std::cout << "mat1:  " << mat1_2d << std::endl;
    std::cout << "mat2:  " << mat2_2d << std::endl;
//    std::cout << "mat3:  " << (mat2 == mat1) << std::endl;
    std::cout << "mat3:  " << mat3_2d << std::endl;
    EXPECT_EQ(mat3_2d.at<TypeParam>(0,0), cv::saturate_cast<TypeParam>(complex128(5,1)));
}

#endif