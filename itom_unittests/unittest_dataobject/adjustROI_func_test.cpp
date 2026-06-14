#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class adjustROI_func_test
    \brief Test for adjustROI method for real data types

    This test class checks functionality of adjustROI method for different Data Objects of different real datatypes.
*/
template <typename _Tp> class adjustROI_func_test : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        int temp_size1[] = { 10,10 };
        dObj1 = ito::DataObject(0, temp_size1, ito::getDataType2<_Tp *>());
        dObj2 = ito::DataObject(2, temp_size1, ito::getDataType2<_Tp *>());
        dObj3 = ito::DataObject(4, 5, 5, ito::getDataType2<_Tp *>());

        int temp_size[] = { 4,5,5,4,3 };
        dObj4 = ito::DataObject(5, temp_size, ito::getDataType2<_Tp *>());
    };

    virtual void TearDown(void){};

    //! calcUniqueValue5D()
    /*!
         This function generates unique values for each element of 5 dimensional data object for test purpose.
    */
    int calcUniqueValue5D(int d1, int d2, int d3, int d4, int d5)
    {
        return d5 + d4 * 10 + d3 * 100 + d2 * 1000 + d1 * 10000;
    }

    typedef _Tp valueType;
    ito::DataObject dObj1;
    ito::DataObject dObj2;
    ito::DataObject dObj3;
    ito::DataObject dObj4;
};

TYPED_TEST_CASE(adjustROI_func_test, ItomRealDataTypes);

//!< This test checks the functionality of adjustROI function with four parameter implementation and general
//!< implementation.
TYPED_TEST(adjustROI_func_test, adjustROI_Test1)
{
    //    int matLimits1d[] = {-4,-6};            //!< defining ROI offsets for empty Data Object dObj1
    int matLimits2d[] = {-4, -4, -1, -3};         //!< defining ROI offsets for 2 Dimensional Data Object dObj2
    int matLimits3d[] = {-1, -1, -1, -1, -2, -1}; //!< defining ROI offsets for 3 Dimensional Data Object dObj3
    int matLimits5d[] = {0, -3, 0,  -4, 0,
                         0, -1, -1, -2, 0}; //!< defining ROI offsets for 5 Dimensional Data Object dObj5
    int test_res[] = {21, 22, 23, 24, 25, 26, 27, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43,
                      44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 61, 62, 63, 64, 65, 66,
                      67, 71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87}; //!< Expected result vector for dObj2
                                                                                   //!< after adjustROI method with 4
                                                                                   //!< parameters implementation.
    int test_res2d[] = {41, 42, 43, 44, 45, 46, 51,
                        52, 53, 54, 55, 56}; //!< Expected result vector for dObj2 after adjustROI method using 2
                                             //!< parameter (general) implementation
    int test_res3d[] = {32, 33, 37, 38, 42, 43, 57,
                        58, 62, 63, 67, 68}; //!< Expected result vector for dObj3 after adjustROI method using 2
                                             //!< parameter (general) implementation

    //!< Initializing 2 dimensional data object dObj2.
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            this->dObj2.template at<TypeParam>(i, j) = (TypeParam)(10 * i + j);
        }
    }

    //!< Initializing 3 dimensional data object dObj3.
    int temp = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                this->dObj3.template at<TypeParam>(i, j, k) = (TypeParam)temp++;
            }
        }
    }

    this->dObj2.adjustROI(
        -2, -1, -1, -2); //!< Adjusting the ROI of 2 dimensional data object dObj2 with four parameter implementation.
    temp = 0;
    for (int i = 0; i < 7; i++)
    {
        for (int j = 0; j < 7; j++)
        {
            EXPECT_EQ(this->dObj2.template at<TypeParam>(i, j),
                      test_res[temp++]); //!< Testing if the elements within the ROI contains same original value after
                                         //!< adjustROI method.
        }
    }
    std::cout << this->dObj3 << std::endl;
    this->dObj2.adjustROI(2, 1, 1, 2); //!< Adjusting back the ROI back to normal position.
    this->dObj2.adjustROI(
        2, matLimits2d); //!< adjusting ROI of dObj2 with general 2 parameter adjustROI method to desired position
    this->dObj3.adjustROI(
        3, matLimits3d); //!< adjusting ROI of dObj3 with general 2 parameter adjustROI method to desired position
    std::cout << this->dObj3 << std::endl;
    //!< Checking values of 2 dimensional data object dObj2 after applying adjustROI().
    temp = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            EXPECT_EQ(cv::saturate_cast<TypeParam>(test_res2d[temp++]),
                      this->dObj2.template at<TypeParam>(i, j)); //!< Testing if the elements within the ROI contains
                                                                 //!< same original value after adjustROI method.
        }
    }

    //!< Checking values of 3 dimensional data object dObj3 after applying adjustROI().
    temp = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                EXPECT_EQ(
                    cv::saturate_cast<TypeParam>(test_res3d[temp++]),
                    this->dObj3.template at<TypeParam>(i, j, k)); //!< Testing if the elements within the ROI contains
                                                                  //!< same original value after adjustROI method.
            }
        }
    }

    //!< test for checking values of  5 dimensional data object dObj5 after applying adjustROI().
    TypeParam *rowPtr1 = NULL;
    int dim1 = this->dObj4.getSize(0); //!< assigning size of 0th dimension of dObj4 to dim1 for test purpose
    int dim2 = this->dObj4.getSize(1); //!< assigning size of 1st dimension of dObj4 to dim2 for test purpose
    int dim3 = this->dObj4.getSize(2); //!< assigning size of 2nd dimension of dObj4 to dim3 for test purpose
    int dim4 = this->dObj4.getSize(3); //!< assigning size of 3rd dimension of dObj4 to dim4 for test purpose
    int dim5 = this->dObj4.getSize(4); //!< assigning size of 4th dimension of dObj4 to dim5 for test purpose
    int dataIdx = 0;
    temp = 0;
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            for (int k = 0; k < dim3; k++)
            {
                dataIdx = i * (dim2 * dim3) + j * dim3 + k;

                for (int l = 0; l < dim4; l++)
                {
                    rowPtr1 = (TypeParam *)this->dObj4.rowPtr(dataIdx, l);

                    for (int m = 0; m < dim5; m++)
                    {
                        rowPtr1[m] = cv::saturate_cast<TypeParam>(this->calcUniqueValue5D(
                            i, j, k, l, m)); //!< assign unique value to each element of data object dObj4
                    }
                }
            }
        }
    }
    this->dObj4.adjustROI(
        5, matLimits5d); //!< adjusting ROI of dObj5 with general 2 parameter adjustROI method to desired position
    dim1 = this->dObj4.getSize(0); //!< assigning size of 0th dimension of dObj4 to dim1 for test purpose
    dim2 = this->dObj4.getSize(1); //!< assigning size of 1st dimension of dObj4 to dim2 for test purpose
    dim3 = this->dObj4.getSize(2); //!< assigning size of 2nd dimension of dObj4 to dim3 for test purpose
    dim4 = this->dObj4.getSize(3); //!< assigning size of 3rd dimension of dObj4 to dim4 for test purpose
    dim5 = this->dObj4.getSize(4); //!< assigning size of 4th dimension of dObj4 to dim5 for test purpose
    int test_res5d[] = {12,  22,  112, 122, 212, 222,
                        312, 322, 412, 422}; //!< Expected result vector for dObj4 after adjustROI method using 2
                                             //!< parameter (general) implementation
    unsigned int idx[] = {0, 0, 0, 0, 0};
    TypeParam v1;
    TypeParam v2;
    temp = 0;
    for (int i = 0; i < dim1; i++)
    {
        idx[0] = i;
        for (int j = 0; j < dim2; j++)
        {
            idx[1] = j;
            for (int k = 0; k < dim3; k++)
            {
                idx[2] = k;
                for (int l = 0; l < dim4; l++)
                {
                    idx[3] = l;
                    for (int m = 0; m < dim5; m++)
                    {
                        idx[4] = m;
                        v1 = this->dObj4.template at<TypeParam>(idx);
                        v2 = cv::saturate_cast<TypeParam>(test_res5d[temp++]);
                        EXPECT_EQ(v1, v2); //!< Testing if the elements within the ROI contains same original value
                                           //!< after adjustROI method.
                    }
                }
            }
        }
    }
}

//!< This test checks the functionality of adjustROI function with wrong number of arguments to check if exception is
//!< raised or not.
TYPED_TEST(adjustROI_func_test, adjustROI_Test2)
{
    int matLimits2d[] = {-1, -2, -1, 0,
                         0,  0}; //!< defining ROI offsets for dObj2 with wrong number of offset values intending to
                                 //!< raise an exception while test
    int matLimits3d[] = {-2, -3, -5, -1}; //!< defining ROI offsets for dObj3 with wrong number of offset values
                                          //!< intending to raise an exception while test
    int matLimits5d[] = {-1, -2, -1, 0};  //!< defining ROI offsets for dObj5 with wrong number of offset values
                                          //!< intending to raise an exception while test

    EXPECT_NO_THROW(
        this->dObj1.adjustROI(0, matLimits2d)); //!< Does not throw any exception as the data object dObj1 is empty
    EXPECT_ANY_THROW(this->dObj2.adjustROI(1, matLimits2d)); //!< expect an exception as the dimension and offset limits
                                                             //!< for ROI are intensionally wrong declared
    EXPECT_ANY_THROW(this->dObj3.adjustROI(2, matLimits3d)); //!< expect an exception as the dimension and offset limits
                                                             //!< for ROI are intensionally wrong declared
    EXPECT_ANY_THROW(this->dObj4.adjustROI(3, matLimits5d)); //!< expect an exception as the dimension and offset limits
                                                             //!< for ROI are intensionally wrong declared
}

//!< This test checks the range of valid ROI.
TYPED_TEST(adjustROI_func_test, adjustROI_Test3)
{
    //!< Defining ROI partially outside the valid matrix-region.
    int matLimits1d[] = {-4, 1}; //!< defining ROI offsets for dObj1 (empty data object) which resides partially outside
                                 //!< the original size of data object
    int matLimits2d[] = {
        -4, 1, -1,
        -3}; //!< defining ROI offsets for dObj2 which resides partially outside the original size of data object
    int matLimits3d[] = {
        -6, 4, 1,
        -7, 2, -9}; //!< defining ROI offsets for dObj3 which resides partially outside the original size of data object
    int matLimits5d[] = {-2, 0,  -1, -2, 1, 0,
                         1,  -1, 2,  0}; //!< defining ROI offsets for dObj4 which resides partially outside the
                                         //!< original size of data object

    EXPECT_NO_THROW(
        this->dObj1.adjustROI(0, matLimits1d)); //!< Does not throw any exception as the data object dObj1 is empty
    EXPECT_ANY_THROW(
        this->dObj1.adjustROI(1, matLimits1d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared partially outside of the valied matrix-region.
    EXPECT_ANY_THROW(
        this->dObj2.adjustROI(2, matLimits2d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared partially outside of the valied matrix-region.
    EXPECT_ANY_THROW(
        this->dObj3.adjustROI(3, matLimits3d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared partially outside of the valied matrix-region.
    EXPECT_ANY_THROW(
        this->dObj4.adjustROI(5, matLimits5d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared partially outside of the valied matrix-region.
}

//!< This test checks the range of valid ROI.
TYPED_TEST(adjustROI_func_test, adjustROI_Test4)
{
    //!< Defining ROI completely outside the valid matrix-region.
    int matLimits1d[] = {4, 1}; //!< defining ROI offsets for dObj1 (empty data object) which resides fully outside the
                                //!< original size of data object
    int matLimits2d[] = {
        -12, 4, -13,
        5}; //!< defining ROI offsets for dObj2 which resides fully outside the original size of data object
    int matLimits3d[] = {
        -1, 1, -1,
        -1, 2, -1}; //!< defining ROI offsets for dObj3 which resides fully outside the original size of data object
    int matLimits5d[] = {2,   -9, -10, 15, 1, -13,
                         -12, 17, 2,   -11}; //!< defining ROI offsets for dObj4 which resides fully outside the
                                             //!< original size of data object

    EXPECT_NO_THROW(
        this->dObj1.adjustROI(0, matLimits1d)); //!< Does not throw any exception as the data object dObj1 is empty
    EXPECT_ANY_THROW(
        this->dObj2.adjustROI(2, matLimits2d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared fully outside of the valied matrix-region.
    EXPECT_ANY_THROW(
        this->dObj3.adjustROI(3, matLimits3d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared fully outside of the valied matrix-region.
    EXPECT_ANY_THROW(
        this->dObj4.adjustROI(5, matLimits5d)); //!< expect an exception as the offset limits for ROI are intensionally
                                                //!< declared fully outside of the valied matrix-region.
}
