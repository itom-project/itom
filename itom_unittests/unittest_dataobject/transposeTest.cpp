#include <iostream>

#include "../../Common/sharedStructures.h"

//opencv
#pragma warning( disable : 4996 ) //C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead.
#pragma once
#include "opencv2\opencv.hpp"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "commonChannel.h"


/*! \class transposeTest
*/
template <typename _Tp> class transposeTest : public ::testing::Test 
    { 
public:

    virtual void SetUp(void)
    {
     
    };
 

    virtual void TearDown(void) 
    {
    };
    typedef _Tp valueType;
     

    };


TYPED_TEST_CASE(transposeTest, ItomRealDataTypes);

//CVTransTest
/*!
    Currently, this test only checks the behaviour of the tranpose operation in OpenCV.
    If an original 4x3 matrix is created, the step vector is [3,1]. If this
    matrix is transposed by OpenCV, OpenCV 2.4.10 or below returns a deep copy, hence the step vector is [4,1].

    If a shallow copy is created (maybe this is the case for OpenCV 3.x), [1,3] should be returned.
    If this is the case, further checks have to be implemented in the dataObject code, where possibly transposed
    cv::Mats are used within dataObjects.
*/
TYPED_TEST(transposeTest, CVTransTest)
{
    cv::Mat matrix = cv::Mat::zeros(4, 3, CV_8UC1);
    std::cout << matrix.step[0] << ";" << matrix.step[1] << "\n" << std::endl; //[3,1]
    cv::Mat matrix_transpose = matrix.t();
    cv::Mat matrix_transpose2;
    cv::transpose(matrix, matrix_transpose2);
    std::cout << matrix_transpose.step[0] << ";" << matrix_transpose.step[1] << "\n" << std::endl; //returns [4,1] for OpenCV 2.4.10
    std::cout << matrix_transpose2.step[0] << ";" << matrix_transpose2.step[1] << "\n" << std::endl; //returns [4,1] for OpenCV 2.4.10
}

