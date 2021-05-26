#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class locateROI_func_test
    \brief Test for locateROI method for real data types

    This test class checks functionality of locateROI method for different Data Objects of different real datatypes.
    This test contains 3 test cases for 2 dimensional, 3 dimensional and 5 dimensional data objects each.
*/
template <typename _Tp> class locateROI_func_test : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        int temp_size1[] = { 10,10 };
        dObj2 = ito::DataObject(2, temp_size1, ito::getDataType2<_Tp *>()); //!< creating 2 dimensional data object
        this->dObj3 = ito::DataObject(4, 5, 5, ito::getDataType2<_Tp *>()); //!< creating 3 dimensional data object

        int temp_size[] = { 4,5,5,4,3 };
        this->dObj4 = ito::DataObject(5, temp_size, ito::getDataType2<_Tp *>()); //!< creating 5 dimesional data object
    };

    virtual void TearDown(void){};

    ito::DataObject dObj2;
    ito::DataObject dObj3;
    ito::DataObject dObj4;
    ito::DataObject dObj5;
    ito::DataObject dObj6;
};

TYPED_TEST_CASE(locateROI_func_test, ItomRealDataTypes);

// locateROI_Test1
/*!
    This test checks the functionality of locateROI() function on 2 dimensional data object dObj2.
*/
TYPED_TEST(locateROI_func_test, locateROI_Test1)
{
    int matLimits2d[] = {-4, -1, -5, -3}; //!< defining ROI offsets for 2 Dimensional Data Object dObj2

    int lims2d[] = {0, 0, 0, 0};       //!< Empty array to store result of locateROI() function.
    int exptLims2d[] = {2, -3, 1, -3}; //!< expected result after applying locateROI() function on dObj2
    this->dObj2.adjustROI(
        -2, -1, -1, -2); //!< Adjusting the ROI of 2 dimensional data object dObj2 with four parameter implementation.
    this->dObj2.locateROI(lims2d); //!< adjusting ROI of dObj2 using locateROI() function.
    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(lims2d[i],
                  exptLims2d[i]); //!< Checking if the result of locateROI() function match with expected result.
    }
    this->dObj2.adjustROI(2, 1, 1, 2); //!< Adjusting back the ROI back to normal position.
    this->dObj2.adjustROI(
        2, matLimits2d); //!< adjusting ROI of dObj2 with general 2 parameter adjustROI method to desired position
    int lims2d1[] = {0, 0, 0, 0};       //!< Empty array to store result of locateROI() function.
    int exptLims2d1[] = {4, -5, 5, -8}; //!< expected result after applying locateROI() function on dObj2
    this->dObj2.locateROI(lims2d1);     //!< adjusting ROI of dObj2 using locateROI() function.
    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(exptLims2d1[i],
                  lims2d1[i]); //!< checking expected result of locateROI() function on 2 dimensional data object dObj2
    }
}

// locateROI_Test2
/*!
    This test checks the functionality of locateROI() function on 3 dimensional data object this->dObj3
*/
TYPED_TEST(locateROI_func_test, locateROI_Test2)
{
    int matLimits3d[] = {-1, -1, -1, -1, -2, -1}; //!< defining ROI offsets for 3 Dimensional Data Object this->dObj3
    int lims3d[] = {0, 0, 0, 0, 0, 0};            //!< Empty array to store result of locateROI() function.
    int exptLims3d[] = {1, -2, 1, -2, 2, -3}; //!< expected result after applying locateROI() function on this->dObj3
    this->dObj3.adjustROI(
        3, matLimits3d); //!< adjusting ROI of this->dObj3 with general 2 parameter adjustROI method to desired position
    this->dObj3.locateROI(lims3d); //!< locating ROI of this->dObj3 using locateROI() function.
    for (int i = 0; i < 6; i++)
    {
        EXPECT_EQ(lims3d[i],
                  exptLims3d[i]); //! Checking if the result of locateROI() function match with expected result.
    }
}

// locateROI_Test3
/*!
    This test checks the functionality of locateROI() function on 5 dimensional data object this->dObj4
*/
TYPED_TEST(locateROI_func_test, locateROI_Test3)
{
    int matLimits5d[] = {0, -3, 0,  -4, 0,
                         0, -1, -1, -2, 0};        //!< defining ROI offsets for 5 Dimensional Data Object this->dObj4
    int lims5d[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; //!< Empty array to store result of locateROI() function.
    int exptLims5d[] = {0, -3, 0,  -4, 0,
                        0, 1,  -2, 2,  -2}; //!< expected result after applying locateROI() function on this->dObj4
    this->dObj4.adjustROI(
        5, matLimits5d); //!< adjusting ROI of this->dObj4 with general 2 parameter adjustROI method to desired position
    this->dObj4.locateROI(lims5d); //!< locating ROI of this->dObj4 using locateROI() function.
    for (int i = 0; i < 10; i++)
    {
        EXPECT_EQ(lims5d[i],
                  exptLims5d[i]); //!< Checking if the result of locateROI() function match with expected result.
    }
}

// locateROI_Test4
/*!
    This test should check a bug in the size, osize and roi members of a dataObject.
    Since dObj2 and this->dObj3 have different number of dimensions, their size, roi and originalSize should be unequal.
    However, at least the original size, was pretent to be equal.
*/
TYPED_TEST(locateROI_func_test, locateROI_Test4)
{
    EXPECT_NE(this->dObj2.getSize(), this->dObj3.getSize());
    EXPECT_NE(this->dObj2.getOriginalSize(), this->dObj3.getOriginalSize());
}
