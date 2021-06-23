#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class iterator_test
    \brief iterator_test checks the functionality of DObjIterator for data objects of various dimensions.

    This test is a group of 3 tests for 2, 3 and 4 dimensional data objects. In each test, an iterator object of type
   DObjIterator is declared and data values of each element of data object is being fetched using this iterator. Later
   on test is performed such that data values fetched by iterators should match the original values given to each data
   object. The test is performed for data objects with no change in ROI as well as data objects with changed ROI.
*/
template <typename _Tp> class iterator_test : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        this->dObj2d = ito::DataObject(21, 13, ito::getDataType2<_Tp *>());
        this->dObj3d = ito::DataObject(5, 10, 10, ito::getDataType2<_Tp *>());
        int temp_size[] = { 3,4,2,10 };
        this->dObj4d = ito::DataObject(4, temp_size, ito::getDataType2<_Tp *>());
    };

    virtual void TearDown(void){};

    ito::DataObject dObj2d;
    ito::DataObject dObj3d;
    ito::DataObject dObj4d;
};

TYPED_TEST_CASE(iterator_test, ItomRealDataTypes);

// iterator_test_2d
/*!
    This test checks the functionality of DObjIterator for 2 dimensional data objects with no change in ROI.
*/
TYPED_TEST(iterator_test, iterator_test_2d)
{
    int temp = 0;             //!< Temporary variable for indexing some arrays used in this test.
    TypeParam objdata2d[273]; //!< This array holds the data of data object this->dObj2d.
    TypeParam *dataptr2d;
    dataptr2d = objdata2d;   //!< Now pointer dataptr2d points to the array objdata2d.
    ito::DObjIterator it_2d; //!< Declaration of DObjIterator

    int dim1_2d = this->dObj2d.getSize(0);
    int dim2_2d = this->dObj2d.getSize(1);

    for (int i = 0; i < dim1_2d; i++)
    {
        for (int j = 0; j < dim2_2d; j++)
        {
            this->dObj2d.template at<TypeParam>(i, j) = cv::saturate_cast<TypeParam>(
                dim2_2d * i +
                j); //!< Assigning unique values to each element of 2 dimensional data object this->dObj2d.
            *dataptr2d = cv::saturate_cast<TypeParam>(
                dim2_2d * i + j); //!< Defining the array with the same data values as in this->dObj2d for test purpose.
            dataptr2d++;
        }
    }
    temp = 0;
    for (it_2d = this->dObj2d.begin(); it_2d != this->dObj2d.end(); ++it_2d)
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam *)(*it_2d))),
                  cv::saturate_cast<TypeParam>(
                      objdata2d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the
                                           //!< values of each element of data object this->dObj2d with the same values
                                           //!< stored in array objdata2d using iterator it_2d .
    }
}

// iteratorROI_test_2d
/*!
    This test checks the functionality of DObjIterator for 2 dimensional data objects with changed ROI.
*/
TYPED_TEST(iterator_test, iteratorROI_test_2d)
{
    int temp = 0;             //!< Temporary variable for indexing some arrays used in this test.
    TypeParam objdata2d[273]; //!< This array holds the data of data object this->dObj2d.
    TypeParam *dataptr2d;
    dataptr2d = objdata2d;   //!< Now pointer dataptr2d points to the array objdata2d.
    ito::DObjIterator it_2d; //!< Declaration of DObjIterator

    int matLimits2d[] = {-4, -4, -1, -3}; //!< defining ROI offsets for 2 Dimensional Data Object this->dObj2d

    int dim1_2d = this->dObj2d.getSize(0);
    int dim2_2d = this->dObj2d.getSize(1);

    for (int i = 0; i < dim1_2d; i++)
    {
        for (int j = 0; j < dim2_2d; j++)
        {
            this->dObj2d.template at<TypeParam>(i, j) = cv::saturate_cast<TypeParam>(
                dim2_2d * i +
                j); //!< Assigning unique values to each element of 2 dimensional data object this->dObj2d.
            if (i > (abs(matLimits2d[0]) - 1) && i < (dim1_2d - abs(matLimits2d[1])) && j > (abs(matLimits2d[2]) - 1) &&
                j < (dim2_2d - abs(matLimits2d[3])))
            {
                *dataptr2d = cv::saturate_cast<TypeParam>(
                    dim2_2d * i +
                    j); //!< Defining the array with the same data values as in ROI of this->dObj2d for test purpose.
                dataptr2d++;
            }
        }
    }

    this->dObj2d.adjustROI(
        2, matLimits2d); //!< adjusting ROI of dObj2 with general 2 parameter adjustROI method to desired position
    temp = 0;
    for (it_2d = this->dObj2d.begin(); it_2d != this->dObj2d.end(); ++it_2d)
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam *)(*it_2d))),
                  cv::saturate_cast<TypeParam>(
                      objdata2d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the
                                           //!< values of each element of data object this->dObj2d with the same values
                                           //!< stored in array objdata2d using iterator it_2d .
    }
}

// iterator_test_3d
/*!
    This test checks the functionality of DObjIterator for 3 dimensional data objects with no change in ROI.
*/
TYPED_TEST(iterator_test, iterator_test_3d)
{
    int temp;                 //!< Temporary variable for indexing some arrays used in this test.
    TypeParam objdata3d[500]; //!< This array holds the data of data object this->dObj3d.
    TypeParam *dataptr3d;
    dataptr3d = objdata3d; //!< Now pointer dataptr3d points to the array objdata3d.

    ito::DObjIterator it_3d; //!< Declaration of DObjIterator
    int dim1_3d = this->dObj3d.getSize(0);
    int dim2_3d = this->dObj3d.getSize(1);
    int dim3_3d = this->dObj3d.getSize(2);

    for (int i = 0; i < dim1_3d; i++)
    {
        for (int j = 0; j < dim2_3d; j++)
        {
            for (int k = 0; k < dim3_3d; k++)
            {
                this->dObj3d.template at<TypeParam>(i, j, k) = cv::saturate_cast<TypeParam>(
                    dim1_3d * i * j + dim2_3d * j +
                    k); //!< Assigning unique values to each element of 3 dimensional data object this->dObj3d.
                *dataptr3d = cv::saturate_cast<TypeParam>(
                    dim1_3d * i * j + dim2_3d * j +
                    k); //!< Defining the array with the same data values as in this->dObj3d for test purpose.
                dataptr3d++;
            }
        }
    }

    temp = 0;
    for (it_3d = this->dObj3d.begin(); it_3d != this->dObj3d.end(); ++it_3d)
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam *)(*it_3d))),
                  cv::saturate_cast<TypeParam>(
                      objdata3d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the
                                           //!< values of each element of data object this->dObj3d with the same values
                                           //!< stored in array objdata3d using iterator it_3d.
    }
}

// iteratorROI_test_3d
/*!
    This test checks the functionality of DObjIterator for 3 dimensional data objects with changed ROI.
*/
TYPED_TEST(iterator_test, iteratorROI_test_3d)
{
    int temp;                 //!< Temporary variable for indexing some arrays used in this test.
    TypeParam objdata3d[500]; //!< This array holds the data of data object this->dObj3d with size dim1_3d x dim2_3d x
                              //!< dim3_3d.The size of this array is more than enough because the ROI of this->dObj3d
                              //!< will be shrunk with the use of adjustROI method during the test.
    TypeParam *dataptr3d;
    dataptr3d = objdata3d;                        //!< Now pointer dataptr3d points to the array objdata3d.
    int matLimits3d[] = {-1, -1, -1, -1, -2, -1}; //!< defining ROI offsets for 3 Dimensional Data Object this->dObj3d

    ito::DObjIterator it_3d; //!< Declaration of DObjIterator
    int dim1_3d = this->dObj3d.getSize(0);
    int dim2_3d = this->dObj3d.getSize(1);
    int dim3_3d = this->dObj3d.getSize(2);

    for (int i = 0; i < dim1_3d; i++)
    {
        for (int j = 0; j < dim2_3d; j++)
        {
            for (int k = 0; k < dim3_3d; k++)
            {
                this->dObj3d.template at<TypeParam>(i, j, k) = cv::saturate_cast<TypeParam>(
                    dim1_3d * i * j + dim2_3d * j +
                    k); //!< Assigning unique values to each element of 3 dimensional data object this->dObj3d.
                if (i > (abs(matLimits3d[0]) - 1) && i < (dim1_3d - abs(matLimits3d[1])))
                {
                    if (j > (abs(matLimits3d[2]) - 1) && j < (dim2_3d - abs(matLimits3d[3])))
                    {
                        if (k > (abs(matLimits3d[4]) - 1) && k < (dim3_3d - abs(matLimits3d[5])))
                        {
                            *dataptr3d = cv::saturate_cast<TypeParam>(
                                dim1_3d * i * j + dim2_3d * j + k); //!< Defining the array with the same data values as
                                                                    //!< in ROI of this->dObj3d for test purpose.
                            dataptr3d++;
                        }
                    }
                }
            }
        }
    }

    this->dObj3d.adjustROI(
        3,
        matLimits3d); //!< adjusting ROI of this->dObj3d with general 2 parameter adjustROI method to desired position

    temp = 0;
    for (it_3d = this->dObj3d.begin(); it_3d != this->dObj3d.end(); ++it_3d)
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam *)(*it_3d))),
                  cv::saturate_cast<TypeParam>(
                      objdata3d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the
                                           //!< values of each element of data object this->dObj3d with the same values
                                           //!< stored in array objdata3d using iterator it_3d.
    }
}

// iterator_test_4d
/*!
    This test checks the functionality of DObjIterator for 3 dimensional data objects.
*/
TYPED_TEST(iterator_test, iterator_test_4d)
{
    int temp; //!< Temporary variable for indexing some arrays used in this test.
    TypeParam
        objdata4d[240]; //!< This array holds the data of data object this->dObj4d with size dim1 x dim2 x dim3 x dim4.
    TypeParam *dataptr4d;
    dataptr4d = objdata4d; //!< Now pointer dataptr3d points to the array objdata4d.

    ito::DObjIterator it_4d; //!< Declaration of DObjIterator

    TypeParam *rowPtr1 = NULL;
    int dim1 = this->dObj4d.getSize(0);
    int dim2 = this->dObj4d.getSize(1);
    int dim3 = this->dObj4d.getSize(2);
    int dim4 = this->dObj4d.getSize(3);
    int dataIdx = 0;
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            dataIdx = i * dim2 + j;
            for (int k = 0; k < dim3; k++)
            {
                rowPtr1 = (TypeParam *)this->dObj4d.rowPtr(dataIdx, k);
                for (int l = 0; l < dim4; l++)
                {
                    rowPtr1[l] = cv::saturate_cast<TypeParam>(dim1 * i * j * k + dim2 * j * k + dim3 * k +
                                                              l); //!< Assigning unique value to each element of dObj4.
                    *dataptr4d = cv::saturate_cast<TypeParam>(
                        dim1 * i * j * k + dim2 * j * k + dim3 * k +
                        l); //!< Defining the array with the same data values as in this->dObj3d for test purpose.
                    dataptr4d++;
                }
            }
        }
    }

    temp = 0;
    for (it_4d = this->dObj4d.begin(); it_4d != this->dObj4d.end(); ++it_4d)
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam *)(*it_4d))),
                  cv::saturate_cast<TypeParam>(
                      objdata4d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the
                                           //!< values of each element of data object this->dObj4d with the same values
                                           //!< stored in array objdata3d using iterator it_4d.
    }
}

// iteratorROI_test_4d
/*!
    This test checks the functionality of DObjIterator for 4 dimensional data objects with changed ROI.
*/
TYPED_TEST(iterator_test, iteratorROI_test_4d)
{
    int temp;                 //!< Temporary variable for indexing some arrays used in this test.
    TypeParam objdata4d[240]; //!< This array holds the data of data object this->dObj4d with size dim1 x dim2 x dim3 x
                              //!< dim4. The size of this array is more than enough because the ROI of this->dObj4d will
                              //!< be shrunk with the use of adjustROI method during the test.
    TypeParam *dataptr4d;
    dataptr4d = objdata4d; //!< Now pointer dataptr4d points to the array objdata4d.
    int matLimits4d[] = {0, -3, 0,  -4,
                         0, 0,  -1, -1}; //!< defining ROI offsets for 4 Dimensional Data Object this->dObj4d

    ito::DObjIterator it_4d; //!< Declaration of DObjIterator
    TypeParam *rowPtr1 = NULL;
    int dim1 = this->dObj4d.getSize(0);
    int dim2 = this->dObj4d.getSize(1);
    int dim3 = this->dObj4d.getSize(2);
    int dim4 = this->dObj4d.getSize(3);
    int dataIdx = 0;
    for (int i = 0; i < dim1; i++)
    {
        for (int j = 0; j < dim2; j++)
        {
            dataIdx = i * dim2 + j;
            for (int k = 0; k < dim3; k++)
            {
                rowPtr1 = (TypeParam *)this->dObj4d.rowPtr(dataIdx, k);
                for (int l = 0; l < dim4; l++)
                {
                    rowPtr1[l] =
                        cv::saturate_cast<TypeParam>(dim1 * i * j * k + dim2 * j * k + dim3 * k +
                                                     l); //!< Assigning unique value to each element of this->dObj4d.

                    if (i > (abs(matLimits4d[0]) - 1) && i < (dim1 - abs(matLimits4d[1])))
                    {
                        if (j > (abs(matLimits4d[2]) - 1) && j < (dim2 - abs(matLimits4d[3])))
                        {
                            if (k > (abs(matLimits4d[4]) - 1) && k < (dim3 - abs(matLimits4d[5])))
                            {
                                if (l > (abs(matLimits4d[6]) - 1) && l < (dim4 - abs(matLimits4d[7])))
                                {
                                    *dataptr4d = cv::saturate_cast<TypeParam>(
                                        dim1 * i * j * k + dim2 * j * k + dim3 * k +
                                        l); //!< Defining the array with the same data values as in ROI of this->dObj4d
                                            //!< for test purpose.
                                    dataptr4d++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    this->dObj4d.adjustROI(
        4,
        matLimits4d); //!< adjusting ROI of this->dObj4d with general 2 parameter adjustROI method to desired position
    temp = 0;
    for (it_4d = this->dObj4d.begin(); it_4d != this->dObj4d.end(); ++it_4d)
    {
        EXPECT_EQ(cv::saturate_cast<TypeParam>(*((TypeParam *)(*it_4d))),
                  cv::saturate_cast<TypeParam>(
                      objdata4d[temp++])); //!< Testing the functionality of declared DObjIterator by comparing the
                                           //!< values of each element of data object this->dObj4d with the same values
                                           //!< stored in array objdata4d using iterator it_4d.
    }
}
