#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"
//#include "test_global.h"
#include "commonChannel.h"

/*! \class ROITest
    \brief ROI methods test for real data types

    This test class checks functionality of different methods dealing with ROI for data objects.
*/
template <typename _Tp> class miscellaneousTests : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        int temp_size1[] = { 10, 10 };
        this->dObj1 = ito::DataObject(0, temp_size1, ito::getDataType2<_Tp *>());
        this->dObj2 = ito::DataObject(2, temp_size1, ito::getDataType2<_Tp *>());
        this->dObj3 = ito::DataObject(3, 3, 10, ito::getDataType2<_Tp *>());

        int temp_size[] = { 3,4,2,10,10 };
        this->dObj4 = ito::DataObject(5, temp_size, ito::getDataType2<_Tp *>());

        int temp_size2[] = { 3 };
        this->dObj7 = ito::DataObject(1, temp_size2, ito::getDataType2<_Tp *>());
    };

    virtual void TearDown(void){};

    ito::DataObject dObj1;
    ito::DataObject dObj2;
    ito::DataObject dObj3;
    ito::DataObject dObj4;
    ito::DataObject dObj5;
    ito::DataObject dObj6;
    ito::DataObject dObj7;
};

TYPED_TEST_CASE(miscellaneousTests, ItomRealDataTypes);

// getDims_getType_Test
/*!

*/
TYPED_TEST(miscellaneousTests, getValueOffset_Test)
{
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;

    //!< Test for getValueOffset() function.
    EXPECT_FLOAT_EQ(0, this->dObj1.getValueOffset());
    EXPECT_FLOAT_EQ(0, this->dObj2.getValueOffset());
    EXPECT_FLOAT_EQ(0, this->dObj3.getValueOffset());
    EXPECT_FLOAT_EQ(0, this->dObj4.getValueOffset());
    EXPECT_FLOAT_EQ(0, this->dObj5.getValueOffset());
    EXPECT_FLOAT_EQ(0, this->dObj6.getValueOffset());
    EXPECT_FLOAT_EQ(0, this->dObj7.getValueOffset());
}

TYPED_TEST(miscellaneousTests, getValueScale_Test)
{
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;

    //!< Test for getValueScale() function.
    EXPECT_FLOAT_EQ(1, this->dObj1.getValueScale());
    EXPECT_FLOAT_EQ(1, this->dObj2.getValueScale());
    EXPECT_FLOAT_EQ(1, this->dObj3.getValueScale());
    EXPECT_FLOAT_EQ(1, this->dObj4.getValueScale());
    EXPECT_FLOAT_EQ(1, this->dObj5.getValueScale());
    EXPECT_FLOAT_EQ(1, this->dObj6.getValueScale());
    EXPECT_FLOAT_EQ(1, this->dObj7.getValueScale());
}

TYPED_TEST(miscellaneousTests, getValueUnit_Test)
{
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;

    //!< Test for getValueUnit() function.
    EXPECT_EQ("", this->dObj1.getValueUnit());
    EXPECT_EQ("", this->dObj2.getValueUnit());
    EXPECT_EQ("", this->dObj3.getValueUnit());
    EXPECT_EQ("", this->dObj4.getValueUnit());
    EXPECT_EQ("", this->dObj5.getValueUnit());
    EXPECT_EQ("", this->dObj6.getValueUnit());
    EXPECT_EQ("", this->dObj7.getValueUnit());
}

TYPED_TEST(miscellaneousTests, getValueDescription_Test)
{
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;
    int i;
    //!< Test for getValueDescription() function.
    EXPECT_EQ("", this->dObj1.getValueDescription());
    EXPECT_EQ("", this->dObj2.getValueDescription());
    EXPECT_EQ("", this->dObj3.getValueDescription());
    EXPECT_EQ("", this->dObj4.getValueDescription());
    EXPECT_EQ("", this->dObj5.getValueDescription());
    EXPECT_EQ("", this->dObj6.getValueDescription());
    EXPECT_EQ("", this->dObj7.getValueDescription());
}

TYPED_TEST(miscellaneousTests, getAxisOffset_Test)
{
    int i;
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;
    //!< Test for getAxisOffset() function.
    int dObj2_dim = this->dObj2.getDims();
    int dObj3_dim = this->dObj3.getDims();
    int dObj4_dim = this->dObj4.getDims();
    int dObj5_dim = this->dObj5.getDims();
    int dObj6_dim = this->dObj6.getDims();
    int dObj7_dim = this->dObj7.getDims();

    for (i = -1; i < 2; i++)
    {
        EXPECT_ANY_THROW(this->dObj1.getAxisOffset(i));
    }

    //!< Test for getAxisOffset() function for this->dObj2.
    EXPECT_ANY_THROW(this->dObj2.getAxisOffset(-1));
    for (i = 0; i < dObj2_dim; i++)
    {
        EXPECT_EQ(0, this->dObj2.getAxisOffset(i));
    }
    EXPECT_ANY_THROW(this->dObj2.getAxisOffset(
        dObj2_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj2.getAxisOffset(dObj2_dim + 1));

    //!< Test for getAxisOffset() function for this->dObj3.
    EXPECT_ANY_THROW(this->dObj3.getAxisOffset(-1));
    for (i = 0; i < dObj3_dim; i++)
    {
        EXPECT_EQ(0, this->dObj3.getAxisOffset(i));
    }
    EXPECT_ANY_THROW(this->dObj3.getAxisOffset(
        dObj3_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj3.getAxisOffset(dObj3_dim + 1));

    //!< Test for getAxisOffset() function for this->dObj4.
    EXPECT_ANY_THROW(this->dObj4.getAxisOffset(-1));
    for (i = 0; i < dObj4_dim; i++)
    {
        EXPECT_EQ(0, this->dObj4.getAxisOffset(i));
    }
    EXPECT_ANY_THROW(this->dObj4.getAxisOffset(
        dObj4_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj4.getAxisOffset(dObj4_dim + 1));

    //!< Test for getAxisOffset() function for this->dObj5.
    EXPECT_ANY_THROW(this->dObj5.getAxisOffset(-1));
    for (i = 0; i < dObj5_dim; i++)
    {
        EXPECT_EQ(0, this->dObj5.getAxisOffset(i));
    }
    EXPECT_ANY_THROW(this->dObj5.getAxisOffset(
        dObj5_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj5.getAxisOffset(dObj5_dim + 1));

    //!< Test for getAxisOffset() function for this->dObj6.
    EXPECT_ANY_THROW(this->dObj6.getAxisOffset(-1));
    for (i = 0; i < dObj6_dim; i++)
    {
        EXPECT_EQ(0, this->dObj6.getAxisOffset(i));
    }
    EXPECT_ANY_THROW(this->dObj6.getAxisOffset(
        dObj6_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj6.getAxisOffset(dObj6_dim + 1));

    //!< Test for getAxisOffset() function for this->dObj7.
    /*< this->dObj7 is explicitely defined as 1 dimensional Data Object.
    But as there is no existance of 1 dimensional Data Objects, this->dObj7 becomes 2 dimensional Data Object.
    So this test checks this type of conversion and result of getAxisOffset() function accordingly.
    */
    EXPECT_ANY_THROW(this->dObj7.getAxisOffset(-1));
    for (i = 0; i < dObj7_dim; i++)
    {
        EXPECT_EQ(0, this->dObj7.getAxisOffset(i));
    }
    EXPECT_ANY_THROW(this->dObj7.getAxisOffset(
        dObj7_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj7.getAxisOffset(dObj7_dim + 1));
}

TYPED_TEST(miscellaneousTests, getAxisScale_Test)
{
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;
    int i;
    int dObj2_dim = this->dObj2.getDims();
    int dObj3_dim = this->dObj3.getDims();
    int dObj4_dim = this->dObj4.getDims();
    int dObj5_dim = this->dObj5.getDims();
    int dObj6_dim = this->dObj6.getDims();
    int dObj7_dim = this->dObj7.getDims();
    //!< Test for getAxisOffset() function.

    for (i = -1; i < 2; i++)
    {
        EXPECT_ANY_THROW(this->dObj1.getAxisScale(i));
    }

    //!< Test for getAxisScale() function for this->dObj2.
    EXPECT_ANY_THROW(this->dObj2.getAxisScale(-1));
    for (i = 0; i < dObj2_dim; i++)
    {
        EXPECT_EQ(1.0, this->dObj2.getAxisScale(i));
    }
    EXPECT_ANY_THROW(this->dObj2.getAxisScale(
        dObj2_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj2.getAxisScale(dObj2_dim + 1));

    //!< Test for getAxisScale() function for this->dObj3.
    EXPECT_ANY_THROW(this->dObj3.getAxisScale(-1));
    for (i = 0; i < dObj3_dim; i++)
    {
        EXPECT_EQ(1.0, this->dObj3.getAxisScale(i));
    }
    EXPECT_ANY_THROW(this->dObj3.getAxisScale(
        dObj3_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj3.getAxisScale(dObj3_dim + 1));

    //!< Test for getAxisScale() function for this->dObj4.
    EXPECT_ANY_THROW(this->dObj4.getAxisScale(-1));
    for (i = 0; i < dObj4_dim; i++)
    {
        EXPECT_EQ(1.0, this->dObj4.getAxisScale(i));
    }
    EXPECT_ANY_THROW(this->dObj4.getAxisScale(
        dObj4_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj4.getAxisScale(dObj4_dim + 1));

    //!< Test for getAxisScale() function for this->dObj5.
    EXPECT_ANY_THROW(this->dObj5.getAxisScale(-1));
    for (i = 0; i < dObj5_dim; i++)
    {
        EXPECT_EQ(1.0, this->dObj5.getAxisScale(i));
    }
    EXPECT_ANY_THROW(this->dObj5.getAxisScale(
        dObj5_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj5.getAxisScale(dObj5_dim + 1));

    //!< Test for getAxisScale() function for this->dObj6.
    EXPECT_ANY_THROW(this->dObj6.getAxisScale(-1));
    for (i = 0; i < dObj6_dim; i++)
    {
        EXPECT_EQ(1.0, this->dObj6.getAxisScale(i));
    }
    EXPECT_ANY_THROW(this->dObj6.getAxisScale(
        dObj6_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj6.getAxisScale(dObj6_dim + 1));

    //!< Test for getAxisScale() function for this->dObj7.
    /*< this->dObj7 is explicitely defined as 1 dimensional Data Object.
    But as there is no existance of 1 dimensional Data Objects, this->dObj7 becomes 2 dimensional Data Object.
    So this test checks this type of conversion and result of getAxisScale() function accordingly.
    */
    EXPECT_ANY_THROW(this->dObj7.getAxisScale(-1));
    for (i = 0; i < dObj7_dim; i++)
    {
        EXPECT_EQ(1.0, this->dObj7.getAxisScale(i));
    }
    EXPECT_ANY_THROW(this->dObj7.getAxisScale(
        dObj7_dim)); //!< testing if this function throws an exception if the parameter is out of range.
    EXPECT_ANY_THROW(this->dObj7.getAxisScale(dObj7_dim + 1));
}

TYPED_TEST(miscellaneousTests, getXYRotationalMatrix_Test)
{
    this->dObj5 = ito::DataObject(this->dObj2);
    this->dObj6 = this->dObj2;
    double r00, r01, r02, r10, r11, r12, r20, r21, r22;

    //!< Test for getXYRotationMatrix() function.
    //    this->dObj1.getXYRotationalMatrix( r00,r01,r02,r10,r11,r12,r20,r21,r22 ) ;        //Note: This test fails for
    //    Obj1 (Empty DataObject).............................................
    // EXPECT_EQ(1,r00);
    // EXPECT_EQ(0,r01);
    // EXPECT_EQ(0,r02);
    // EXPECT_EQ(0,r10);
    // EXPECT_EQ(1,r11);
    // EXPECT_EQ(0,r12);
    // EXPECT_EQ(0,r20);
    // EXPECT_EQ(0,r21);
    // EXPECT_EQ(1,r22);

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = 0;
    this->dObj2.getXYRotationalMatrix(r00, r01, r02, r10, r11, r12, r20, r21, r22);
    EXPECT_EQ(1, r00);
    EXPECT_EQ(0, r01);
    EXPECT_EQ(0, r02);
    EXPECT_EQ(0, r10);
    EXPECT_EQ(1, r11);
    EXPECT_EQ(0, r12);
    EXPECT_EQ(0, r20);
    EXPECT_EQ(0, r21);
    EXPECT_EQ(1, r22);

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = 0;
    this->dObj3.getXYRotationalMatrix(r00, r01, r02, r10, r11, r12, r20, r21, r22);
    EXPECT_EQ(1, r00);
    EXPECT_EQ(0, r01);
    EXPECT_EQ(0, r02);
    EXPECT_EQ(0, r10);
    EXPECT_EQ(1, r11);
    EXPECT_EQ(0, r12);
    EXPECT_EQ(0, r20);
    EXPECT_EQ(0, r21);
    EXPECT_EQ(1, r22);

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = 0;
    this->dObj4.getXYRotationalMatrix(r00, r01, r02, r10, r11, r12, r20, r21, r22);
    EXPECT_EQ(1, r00);
    EXPECT_EQ(0, r01);
    EXPECT_EQ(0, r02);
    EXPECT_EQ(0, r10);
    EXPECT_EQ(1, r11);
    EXPECT_EQ(0, r12);
    EXPECT_EQ(0, r20);
    EXPECT_EQ(0, r21);
    EXPECT_EQ(1, r22);

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = 0;
    this->dObj5.getXYRotationalMatrix(r00, r01, r02, r10, r11, r12, r20, r21, r22);
    EXPECT_EQ(1, r00);
    EXPECT_EQ(0, r01);
    EXPECT_EQ(0, r02);
    EXPECT_EQ(0, r10);
    EXPECT_EQ(1, r11);
    EXPECT_EQ(0, r12);
    EXPECT_EQ(0, r20);
    EXPECT_EQ(0, r21);
    EXPECT_EQ(1, r22);

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = 0;
    this->dObj6.getXYRotationalMatrix(r00, r01, r02, r10, r11, r12, r20, r21, r22);
    EXPECT_EQ(1, r00);
    EXPECT_EQ(0, r01);
    EXPECT_EQ(0, r02);
    EXPECT_EQ(0, r10);
    EXPECT_EQ(1, r11);
    EXPECT_EQ(0, r12);
    EXPECT_EQ(0, r20);
    EXPECT_EQ(0, r21);
    EXPECT_EQ(1, r22);

    r00, r01, r02, r10, r11, r12, r20, r21, r22 = 0;
    this->dObj7.getXYRotationalMatrix(r00, r01, r02, r10, r11, r12, r20, r21, r22);
    EXPECT_EQ(1, r00);
    EXPECT_EQ(0, r01);
    EXPECT_EQ(0, r02);
    EXPECT_EQ(0, r10);
    EXPECT_EQ(1, r11);
    EXPECT_EQ(0, r12);
    EXPECT_EQ(0, r20);
    EXPECT_EQ(0, r21);
    EXPECT_EQ(1, r22);
}

//!< this test should check if a mask for all NaN values can be obtained
//! from OpenCV matrices if the array is compared to itself.
//! see: https://stackoverflow.com/questions/41759247/filter-opencv-mat-for-nan-values
TYPED_TEST(miscellaneousTests, openCVNanMask_Test)
{
    cv::Mat arr = cv::Mat::zeros(9, 10, CV_32FC1);

    for (int r = 3; r< 7; ++r)
    {
        for (int c = 4; c < 8; ++c)
        {
            arr.at<float>(r, c) = std::numeric_limits<float>::quiet_NaN();
        }
    }

    cv::Mat maskValid = (arr == arr);

    for (int r = 0; r < 9; ++r)
    {
        for (int c = 0; c < 10; ++c)
        {
            if (r >= 3 && r < 7 && c >= 4 && c < 8)
            {
                EXPECT_EQ(maskValid.at<unsigned char>(r, c), 0);
            }
            else
            {
                EXPECT_EQ(maskValid.at<unsigned char>(r, c), 255);
            }
        }
    }
}

//!< similar test than above, taken from https://github.com/opencv/opencv/issues/16465
//! It is indeed true to compare to matrices with an equal operator, however
//! if the != operator would be used, the result is undefined!
int count_NaNs_in_image(int rows, int cols)
{
    cv::Mat1f test(rows, cols);
    test.setTo(std::numeric_limits<float>::quiet_NaN());
    /*
    // this will not work for all cases
    cv::Mat1b is_nan = (test != test);
    return cv::countNonZero(is_nan);*/

    // this should work
    cv::Mat1b is_notnan = (test == test);
    return (rows * cols) - cv::countNonZero(is_notnan);
}

TYPED_TEST(miscellaneousTests, openCVNanMask2_Test)
{
    EXPECT_EQ(16, count_NaNs_in_image(4, 4)); // succeeds
    EXPECT_EQ(44, count_NaNs_in_image(4, 11)); // fails with 12 NaNs found instead of 44?
}
