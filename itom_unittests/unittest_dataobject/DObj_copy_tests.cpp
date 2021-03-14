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

/*! \class copyTests
    \brief test for copyTo(...) method

    This test class checks functionality of copyTo(...) method with its regionOnly parameter as "true" and "false" both.
*/
template <typename _Tp> class copyTests : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        int *temp_size1 = new int[2];
        temp_size1[0] = 10;
        temp_size1[1] = 10;
        this->dObj1_s = ito::DataObject(0, temp_size1, ito::getDataType2<_Tp *>());
        this->dObj2_s = ito::DataObject(2, temp_size1, ito::getDataType2<_Tp *>());
        this->dObj3_s = ito::DataObject(4, 5, 5, ito::getDataType2<_Tp *>());
        int *temp_size = new int[5];
        temp_size[0] = 10;
        temp_size[1] = 12;
        temp_size[2] = 16;
        temp_size[3] = 18;
        temp_size[4] = 10;
        this->dObj4_s = ito::DataObject(5, temp_size, ito::getDataType2<_Tp *>());
        int *temp_size3 = new int[5];
        temp_size3[0] = 4;
        temp_size3[1] = 5;
        temp_size3[2] = 5;
        temp_size3[3] = 4;
        temp_size3[4] = 3;
        this->dObj4_s2 = ito::DataObject(5, temp_size3, ito::getDataType2<_Tp *>());
        int *temp_size2 = new int[5];
        temp_size2[0] = 1;
        temp_size2[1] = 1;
        temp_size2[2] = 2;
        temp_size2[3] = 1;
        temp_size2[4] = 1;
        this->dObj4_s1 = ito::DataObject(5, temp_size2, ito::getDataType2<_Tp *>());

        this->dObj1_d = ito::DataObject(0, temp_size1, ito::getDataType2<_Tp *>());
        this->dObj2_d = ito::DataObject(2, temp_size1, ito::getDataType2<_Tp *>());
        this->dObj3_d = ito::DataObject(3, 3, 10, ito::getDataType2<_Tp *>());
        this->dObj4_d = ito::DataObject(5, temp_size, ito::getDataType2<_Tp *>());
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

    ito::DataObject dObj1_s;
    ito::DataObject dObj2_s;
    ito::DataObject dObj3_s;
    ito::DataObject dObj4_s;
    ito::DataObject dObj4_s1;
    ito::DataObject dObj4_s2;

    ito::DataObject dObj1_sr;
    ito::DataObject dObj2_sr;
    ito::DataObject dObj3_sr;
    ito::DataObject dObj4_sr;

    ito::DataObject dObj1_d;
    ito::DataObject dObj2_d;
    ito::DataObject dObj3_d;
    ito::DataObject dObj4_d;

    ito::DataObject dObj1_dr;
    ito::DataObject dObj2_dr;
    ito::DataObject dObj3_dr;
    ito::DataObject dObj4_dr;

    ito::DataObject dObj1_dr1;
    ito::DataObject dObj2_dr1;
    ito::DataObject dObj3_dr1;
    ito::DataObject dObj4_dr1;
};

TYPED_TEST_CASE(copyTests, ItomRealDataTypes);

// copyTo_True_Test
/*!
    This test checks the functionality of copyTo(...) method with its parameter regionOnly=true.
    After applying copyTo(...) method, the destination object must have the same size and number of dimensions as the
   ROI of the source. Values in ROI of source and destination must be equal.
*/
TYPED_TEST(copyTests, copyTo_True_Test)
{
    int temp = 0;
    unsigned int res1_str[] = {41, 42, 43, 44, 45, 51,
                               52, 53, 54, 55}; //!< Expected element values for this->dObj2_dr after copying ROI of
                                                //!< this->dObj2_sr by copyTo() function into it.
    unsigned int res2_str[] = {32, 33, 37, 38, 42, 43, 57,
                               58, 62, 63, 67, 68}; //!< Expected element values for this->dObj3_dr after copying ROI of
                                                    //!< this->dObj3_sr by copyTo() function into it.
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            this->dObj2_s.template at<TypeParam>(i, j) = cv::saturate_cast<TypeParam>(10 * i + j);
        } //!< assigning unique value to each element of Data Object this->dObj2_s.
    }

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                this->dObj3_s.template at<TypeParam>(i, j, k) = cv::saturate_cast<TypeParam>(
                    temp++); //!< assigning unique value to each element of 3 dimensional Data Object this->dObj3_s
            }
        }
    }

    this->dObj1_sr = this->dObj1_s; //!< copying this->dObj1_s into this->dObj1_sr with assignment operator.
    this->dObj2_sr = this->dObj2_s; //!< copying this->dObj2_s into this->dObj2_sr with assignment operator.
    this->dObj3_sr = this->dObj3_s; //!< copying this->dObj3_s into this->dObj3_sr with assignment operator.
    this->dObj4_sr = this->dObj4_s; //!< copying this->dObj4_s into this->dObj4_sr with assignment operator.

    int matLimits1d[] = {-4, -6};         //!< defining offsets for ROI of empty data object this->dObj1_sr
    int matLimits2d[] = {-4, -4, -1, -4}; //!< defining offsets for ROI of 2 dimensional data object this->dObj2_sr
    int matLimits3d[] = {-1, -1, -1,
                         -1, -2, -1}; //!< defining offsets for ROI of 3 dimensional data object this->dObj3_sr
    int matLimits5d[] = {-4, -4, -1, -4, -2,
                         -3, -1, -1, -2, -1}; //!< defining offsets for ROI of 5 dimensional data object this->dObj4_sr

    this->dObj1_sr.adjustROI(0, matLimits1d); //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_sr.adjustROI(2, matLimits2d); //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_sr.adjustROI(3, matLimits3d); //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_sr.adjustROI(5, matLimits5d); //!< adjust ROI (shrinking because offset values are negative)

    //!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = True.
    this->dObj1_s.copyTo(this->dObj1_d, true);
    EXPECT_EQ(0,
              this->dObj1_d
                  .getDims()); //!< Testing if dimension of this->dObj1_d is same as dimension of ROI of this->dObj1_s.
    EXPECT_EQ(
        this->dObj1_s.getSize(),
        this->dObj1_d.getSize()); //!< Testing if size of this->dObj1_d is same as dimension of ROI of this->dObj1_s.

    //!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = True.
    this->dObj2_s.copyTo(this->dObj2_d, true);
    EXPECT_EQ(2, this->dObj2_d.getDims());
    EXPECT_EQ(this->dObj2_s.getSize(), this->dObj2_d.getSize());

    //!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = True.
    this->dObj3_s.copyTo(this->dObj3_d, true);
    EXPECT_EQ(3, this->dObj3_d.getDims());
    EXPECT_EQ(this->dObj3_s.getSize(), this->dObj3_d.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                 //!< is same as the ROI of original Data Object.

    //!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = True.
    this->dObj4_s.copyTo(this->dObj4_d, true);
    EXPECT_EQ(5, this->dObj4_d.getDims());
    EXPECT_EQ(this->dObj4_s.getSize(), this->dObj4_d.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                 //!< is same as the ROI of original Data Object.

    //!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = True.
    this->dObj1_sr.copyTo(this->dObj1_dr, true);
    EXPECT_EQ(0, this->dObj1_dr.getDims());
    EXPECT_EQ(this->dObj1_sr.getSize(), this->dObj1_dr.getSize());

    //!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = True.
    this->dObj2_sr.copyTo(this->dObj2_dr, true); //!< Copying only ROI from this->dObj2_sr to this->dObj2_dr using
                                                 //!< copyto(...) method with its regionOnly parameter as "true".
    EXPECT_EQ(this->dObj2_sr.getDims(),
              this->dObj2_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj2_sr.getSize(), this->dObj2_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
            EXPECT_EQ(this->dObj2_dr.template at<TypeParam>(i, j),
                      cv::saturate_cast<TypeParam>(
                          res1_str[i * 5 + j])); //!< Check if the values of elements in ROI of this->dObj2_dr are same
                                                 //!< as in the original this->dObj2_s.
    }

    //!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = True.
    this->dObj3_sr.copyTo(this->dObj3_dr, true); //!< copying this->dObj3_sr into this->dObj3_dr using copyTo(...)
                                                 //!< method with its regionOnly parameter as "true".
    EXPECT_EQ(this->dObj3_sr.getDims(),
              this->dObj3_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj3_sr.getSize(), this->dObj3_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.

    temp = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                EXPECT_EQ(
                    this->dObj3_dr.template at<TypeParam>(i, j, k),
                    cv::saturate_cast<TypeParam>(
                        res2_str[temp++])); //!< Checking if elements of ROI of original Data Object this->dObj3_sr has
                                            //!< been changed while copying it to this->dObj3_dr.
            }
        }
    }

    //!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = True.
    this->dObj4_sr.copyTo(this->dObj4_dr, true); //!< copying this->dObj4_sr into this->dObj4_dr using copyTo(...)
                                                 //!< function with its regionOnly parameter as "true".
    EXPECT_EQ(this->dObj4_sr.getDims(),
              this->dObj4_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj4_sr.getSize(), this->dObj4_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.
}

// copyTo_False_Test
/*!
    This test checks the functionality of copyTo(..) function with "regionOnly" parameter set to "false" on data objects
   of different data types and dimensions except 5 dimension data objects. After applying this method, destination
   object must have the same original size and dimensions as the source data object. ROI must have the same position and
   size. Even after adjusting back ROI of the destination object, the contents of the data object should be same as the
   original data object.
*/
TYPED_TEST(copyTests, copyTo_False_Test)
{

    int expSize2d_sr[] = {12, 13};
    int expOffsets2d_sr[] = {4, 1};

    int orgSize2d_sr[] = {0, 0};
    int offsets2d_sr[] = {0, 0};
    int orgSize2d_dr[] = {0, 0};
    int offsets2d_dr[] = {0, 0};

    int orgSize3d_sr[] = {0, 0, 0};
    int offsets3d_sr[] = {0, 0, 0};
    int orgSize3d_dr[] = {0, 0, 0};
    int offsets3d_dr[] = {0, 0, 0};

    int temp_index[] = {0, 0, 0, 0, 0};

    int temp = 0;
    unsigned int res1_str[] = {41, 42, 43, 44, 45, 51,
                               52, 53, 54, 55}; //!< Expected element values for this->dObj2_dr after copying ROI of
                                                //!< this->dObj2_sr by copyTo() function into it.
    unsigned int res2_str[] = {32, 33, 37, 38, 42, 43, 57,
                               58, 62, 63, 67, 68}; //!< Expected element values for this->dObj3_dr after copying ROI of
                                                    //!< this->dObj3_sr by copyTo() function into it.
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            this->dObj2_s.template at<TypeParam>(i, j) = cv::saturate_cast<TypeParam>(10 * i + j);
        } //!< assigning unique value to each element of Data Object this->dObj2_s.
    }

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                this->dObj3_s.template at<TypeParam>(i, j, k) = cv::saturate_cast<TypeParam>(
                    temp++); //!< assigning unique value to each element of 3 dimensional Data Object this->dObj3_s
            }
        }
    }

    temp = 0;
    this->dObj1_sr = this->dObj1_s;
    this->dObj2_sr = this->dObj2_s;
    this->dObj3_sr = this->dObj3_s;
    this->dObj4_sr = this->dObj4_s;

    int matLimits1d[] = {-4, -6};           //!< defining offsets for ROI
    int matLimits2d_1[] = {-4, -4, -1, -4}; //!< defining offsets for ROI
    int matLimits2d_2[] = {4, 4, 1, 4};
    int matLimits3d_1[] = {-1, -1, -1, -1, -2, -1};
    int matLimits3d_2[] = {1, 1, 1, 1, 2, 1};
    int matLimits5d_1[] = {-4, -4, -1, -4, -2, -3, -1, -1, -2, -1};
    int matLimits5d_2[] = {4, 4, 1, 4, 2, 3, 1, 1, 2, 1};

    this->dObj1_sr.adjustROI(0, matLimits1d);   //!< adjust ROI (shrinking because offset values are negative)
    this->dObj2_sr.adjustROI(2, matLimits2d_1); //!< adjust ROI (shrinking because offset values are negative)
    this->dObj3_sr.adjustROI(3, matLimits3d_1); //!< adjust ROI (shrinking because offset values are negative)
    this->dObj4_sr.adjustROI(5, matLimits5d_1); //!< adjust ROI (shrinking because offset values are negative)

    //!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = False.
    this->dObj1_s.copyTo(this->dObj1_d, false);
    EXPECT_EQ(0,
              this->dObj1_d
                  .getDims()); //!< Testing if dimension of this->dObj1_d is same as dimension of ROI of this->dObj1_s.
    EXPECT_EQ(
        this->dObj1_s.getSize(),
        this->dObj1_d.getSize()); //!< Testing if size of this->dObj1_d is same as dimension of ROI of this->dObj1_s.

    //!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = False.
    this->dObj2_s.copyTo(this->dObj2_d, false);
    EXPECT_EQ(2, this->dObj2_d.getDims());
    EXPECT_EQ(this->dObj2_s.getSize(), this->dObj2_d.getSize());

    //!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = False.
    this->dObj3_s.copyTo(this->dObj3_d, false);
    EXPECT_EQ(3, this->dObj3_d.getDims());
    EXPECT_EQ(this->dObj3_s.getSize(), this->dObj3_d.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                 //!< is same as the ROI of original Data Object.

    //!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = False.
    this->dObj4_s.copyTo(this->dObj4_d, false);
    EXPECT_EQ(5, this->dObj4_d.getDims());
    EXPECT_EQ(this->dObj4_s.getSize(), this->dObj4_d.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                 //!< is same as the ROI of original Data Object.

    //!< Testing functionality of copyTo() function for empty Data Object with regionOnly parameter = True.
    this->dObj1_sr.copyTo(this->dObj1_dr, false);
    EXPECT_EQ(0, this->dObj1_dr.getDims());
    EXPECT_EQ(this->dObj1_sr.getSize(), this->dObj1_dr.getSize());

    //!< Testing functionality of copyTo() function for 2 dimensional Data Object with regionOnly parameter = False.
    this->dObj2_sr.copyTo(this->dObj2_dr, false); //!< Copying only ROI from this->dObj2_sr to this->dObj2_dr.
    EXPECT_EQ(this->dObj2_sr.getDims(),
              this->dObj2_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj2_sr.getSize(), this->dObj2_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
            EXPECT_EQ(this->dObj2_dr.template at<TypeParam>(i, j),
                      cv::saturate_cast<TypeParam>(
                          res1_str[i * 5 + j])); //!< Check if the values of elements in ROI are same as in the original
    }

    this->dObj2_dr.locateROI(orgSize2d_dr, offsets2d_dr);
    this->dObj2_sr.locateROI(orgSize2d_sr, offsets2d_sr);
    for (int i = 0; i < 2; i++)
        EXPECT_EQ(orgSize2d_dr[i], orgSize2d_sr[i]);
    for (int i = 0; i < 2; i++)
        EXPECT_EQ(offsets2d_dr[i], offsets2d_sr[i]);

    this->dObj2_dr.adjustROI(2, matLimits2d_2);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            EXPECT_EQ(this->dObj2_s.template at<TypeParam>(i, j), this->dObj2_dr.template at<TypeParam>(i, j));
        } //!< checking if the original values of all elements retained after expanding ROI same as original Data Object
          //!< this->dObj2_s.
    }

    //!< Testing functionality of copyTo() function for 3 dimensional Data Object with regionOnly parameter = False.
    this->dObj3_sr.copyTo(this->dObj3_dr, false);
    EXPECT_EQ(this->dObj3_sr.getDims(),
              this->dObj3_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj3_sr.getSize(), this->dObj3_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.

    temp = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                EXPECT_EQ(
                    this->dObj3_dr.template at<TypeParam>(i, j, k),
                    cv::saturate_cast<TypeParam>(
                        res2_str[temp++])); //!< Checking if elements of ROI of original Data Object this->dObj3_sr has
                                            //!< been changed while copying it to this->dObj3_dr.
            }
        }
    }

    this->dObj3_dr.locateROI(orgSize3d_dr, offsets3d_dr);
    this->dObj3_sr.locateROI(orgSize3d_sr, offsets3d_sr);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(orgSize3d_dr[i], orgSize3d_sr[i]);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(offsets3d_dr[i], offsets3d_sr[i]);

    this->dObj3_dr.adjustROI(3,
                             matLimits3d_2); //!< adjust ROI back to original size of source Data Object this->dObj3_s.

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                EXPECT_EQ(this->dObj3_dr.template at<TypeParam>(i, j, k),
                          this->dObj3_s.template at<TypeParam>(
                              i, j, k)); //!< Checking if elements of ROI of original Data Object this->dObj3_dr has
                                         //!< been changed while adusting the size of ROI back to original.
            }
        }
    }

    //!< Testing functionality of copyTo() function for 5 dimensional Data Object with regionOnly parameter = False.
    this->dObj4_sr.copyTo(this->dObj4_dr, false);
    EXPECT_EQ(this->dObj4_sr.getDims(),
              this->dObj4_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj4_sr.getSize(), this->dObj4_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.
}

//!< copyTo_False_Test1
/*!
    This test checks the functionality of copyTo(..) function with "regionOnly" parameter set to "false" on data objects
   of different data types but only 5 dimension. This test is basically extension of copyTo_False_Test adding 5
   dimensional data object.
*/
TYPED_TEST(copyTests, copyTo_False_Test1)
{
    int matLimits5d_1[] = {0, -3, 0, -4, 0, 0, -1, -1, -2, 0};
    int matLimits5d_2[] = {0, 3, 0, 4, 0, 0, 1, 1, 2, 0};
    int temp = 0;

    TypeParam *rowPtr1 = NULL;
    TypeParam *rowPtr_d1 = NULL;
    int dim1 = this->dObj4_s2.getSize(0); //!< assigning size of 0th dimension of dObj4 to dim1 for test purpose
    int dim2 = this->dObj4_s2.getSize(1); //!< assigning size of 1st dimension of dObj4 to dim2 for test purpose
    int dim3 = this->dObj4_s2.getSize(2); //!< assigning size of 2nd dimension of dObj4 to dim3 for test purpose
    int dim4 = this->dObj4_s2.getSize(3); //!< assigning size of 3rd dimension of dObj4 to dim4 for test purpose
    int dim5 = this->dObj4_s2.getSize(4); //!< assigning size of 4th dimension of dObj4 to dim5 for test purpose
    int dataIdx = 0;
    int dataIdx_d = 0;
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
                    rowPtr1 = (TypeParam *)this->dObj4_s2.rowPtr(dataIdx, l);

                    for (int m = 0; m < dim5; m++)
                    {
                        rowPtr1[m] = cv::saturate_cast<TypeParam>(this->calcUniqueValue5D(
                            i, j, k, l, m)); //!< assign unique value to each element of data object dObj4
                    }
                }
            }
        }
    }
    this->dObj4_sr = this->dObj4_s2;

    this->dObj4_sr.adjustROI(5, matLimits5d_1); //!< adjust ROI (shrinking because offset values are negative)

    this->dObj4_sr.copyTo(this->dObj4_dr, false);
    EXPECT_EQ(this->dObj4_sr.getDims(),
              this->dObj4_dr.getDims()); //!< Testing if the dimensions of ROI of copied Data Object are same as the ROI
                                         //!< of original Data Object.
    EXPECT_EQ(this->dObj4_sr.getSize(), this->dObj4_dr.getSize()); //!< Testing if the size of ROI of copied Data Object
                                                                   //!< is same as the ROI of original Data Object.
    dim1 = this->dObj4_dr.getSize(0); //!< assigning size of 0th dimension of dObj4 to dim1 for test purpose
    dim2 = this->dObj4_dr.getSize(1); //!< assigning size of 1st dimension of dObj4 to dim2 for test purpose
    dim3 = this->dObj4_dr.getSize(2); //!< assigning size of 2nd dimension of dObj4 to dim3 for test purpose
    dim4 = this->dObj4_dr.getSize(3); //!< assigning size of 3rd dimension of dObj4 to dim4 for test purpose
    dim5 = this->dObj4_dr.getSize(4); //!< assigning size of 4th dimension of dObj4 to dim5 for test purpose
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
                        v1 = this->dObj4_dr.template at<TypeParam>(idx);
                        v2 = cv::saturate_cast<TypeParam>(test_res5d[temp++]);
                        EXPECT_EQ(v1, v2); //!< Testing if the elements within the ROI contains same original value
                                           //!< after copyTo(...) method.
                    }
                }
            }
        }
    }

    this->dObj4_dr.adjustROI(
        5, matLimits5d_2); //!< adjusting ROI of dObj5 with general 2 parameter adjustROI method to desired position

    TypeParam *rowPtr1_dr = NULL;
    dim1 = this->dObj4_dr.getSize(0); //!< assigning size of 0th dimension of dObj4 to dim1 for test purpose
    dim2 = this->dObj4_dr.getSize(1); //!< assigning size of 1st dimension of dObj4 to dim2 for test purpose
    dim3 = this->dObj4_dr.getSize(2); //!< assigning size of 2nd dimension of dObj4 to dim3 for test purpose
    dim4 = this->dObj4_dr.getSize(3); //!< assigning size of 3rd dimension of dObj4 to dim4 for test purpose
    dim5 = this->dObj4_dr.getSize(4); //!< assigning size of 4th dimension of dObj4 to dim5 for test purpose
    dataIdx = 0;
    dataIdx_d = 0;
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
                    rowPtr1_dr = (TypeParam *)this->dObj4_dr.rowPtr(
                        dataIdx, l); //!< using row pointer accessing the data of data object this->dObj4_dr

                    for (int m = 0; m < dim5; m++)
                    {
                        EXPECT_EQ(rowPtr1_dr[m],
                                  cv::saturate_cast<TypeParam>(this->calcUniqueValue5D(
                                      i, j, k, l, m))); //!< checking if the ROI  of this->dObj4_dr retains same after
                                                        //!< applying copyTo(...) function.
                    }
                }
            }
        }
    }
}

//!< copyTo_NoRealloc_Test
/*!
    It is desired, that a copyTo operation only reallocates the rhs object if its
    sizes does not correspond to the copied range. This is tested here.
*/
TYPED_TEST(copyTests, copyTo_NoRealloc_RegionOnly_Test)
{
    ito::int16 data1[] = {1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24};
    ito::DataObject source(3, 4, ito::tInt16);
    memcpy(source.rowPtr<ito::int16>(0, 0), data1, 3 * 4 * sizeof(ito::int16));

    ito::int16 data2[] = {100, 200, 300, 400, 110, 120, 130, 140, 210, 220, 230, 240};
    ito::DataObject destination(3, 4, ito::tInt16);
    memcpy(destination.rowPtr<ito::int16>(0, 0), data2, 3 * 4 * sizeof(ito::int16));

    // roi of source
    ito::DataObject source_roi = source.at(ito::Range::all(), ito::Range(1, 2));
    ito::DataObject destination_roi = destination.at(ito::Range::all(), ito::Range(0, 1));

    // copy
    source_roi.copyTo(destination_roi, 1);

    ito::int16 *line0 = destination.rowPtr<ito::int16>(0, 0);
    ito::int16 *line1 = destination.rowPtr<ito::int16>(0, 1);
    ito::int16 *line2 = destination.rowPtr<ito::int16>(0, 2);
    EXPECT_EQ(2, line0[0]);
    EXPECT_EQ(200, line0[1]);
    EXPECT_EQ(300, line0[2]);
    EXPECT_EQ(400, line0[3]);

    EXPECT_EQ(12, line1[0]);
    EXPECT_EQ(120, line1[1]);
    EXPECT_EQ(130, line1[2]);
    EXPECT_EQ(140, line1[3]);

    EXPECT_EQ(22, line2[0]);
    EXPECT_EQ(220, line2[1]);
    EXPECT_EQ(230, line2[2]);
    EXPECT_EQ(240, line2[3]);
}
