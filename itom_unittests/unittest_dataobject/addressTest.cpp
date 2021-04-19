#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996)

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class AddressTest
    \brief Address test for all data types declared as "ItomDataAllTypes"

    This is a basic test class for any matrix of any data type. This test class confirms if the different parameters of
   already declared matrices are alright.
*/
template <typename _Tp> class AddressTest : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        matrix1x1 = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        matrix1x1.at<_Tp>(0, 0) = cv::saturate_cast<_Tp>(11);

        matrix1x2 = ito::DataObject(1, 2, ito::getDataType2<_Tp *>());
        matrix1x2.at<_Tp>(0, 0) = cv::saturate_cast<_Tp>(11);
        matrix1x2.at<_Tp>(0, 1) = cv::saturate_cast<_Tp>(12);

        matrix2x2 = ito::DataObject(2, 2, ito::getDataType2<_Tp *>());
        matrix2x2.at<_Tp>(0, 0) = cv::saturate_cast<_Tp>(11);
        matrix2x2.at<_Tp>(0, 1) = cv::saturate_cast<_Tp>(12);
        matrix2x2.at<_Tp>(1, 0) = cv::saturate_cast<_Tp>(21);
        matrix2x2.at<_Tp>(1, 1) = cv::saturate_cast<_Tp>(22);

        matrix1x1x1 = ito::DataObject(1, 1, 1, ito::getDataType2<_Tp *>());
        matrix1x1x1.at<_Tp>(0, 0, 0) = 111;
    };

    virtual void TearDown(void){};

    ito::DataObject matrix1x1; /*!< 1x1 matrix created with the template type */
    ito::DataObject matrix1x2;
    ito::DataObject matrix2x2;
    ito::DataObject matrix1x1x1;

    // ito::DataObject matrix2x1x2;

    typedef _Tp valueType;
};

// checkValues
/*!
    This test class declares and defines different possible multi dimensional matrices and tests if the values of
   different elements in matrices are unchanged.
*/
TYPED_TEST_CASE(AddressTest, ItomDataAllTypes);

TYPED_TEST(AddressTest, checkValues)
{
    int typeno = ito::getDataType((const TypeParam *)NULL);
    EXPECT_EQ(this->matrix1x1.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(11));
    EXPECT_EQ(this->matrix1x1.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(11));

    EXPECT_EQ(this->matrix1x2.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(11));
    EXPECT_EQ(this->matrix1x2.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(12));

    EXPECT_EQ(this->matrix2x2.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(11));
    EXPECT_EQ(this->matrix2x2.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(12));
    EXPECT_EQ(this->matrix2x2.template at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(21));
    EXPECT_EQ(this->matrix2x2.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(22));

    EXPECT_EQ(this->matrix1x1x1.template at<TypeParam>(0, 0, 0), cv::saturate_cast<TypeParam>(111));
}

// checkDim
/*!
    This test checks if the dimensions of different predefined matrices are unchanged using "getDims" method.
*/
TYPED_TEST(AddressTest, checkDim)
{

    // EXPECT_EQ ( this->matrix1x1.getDims<TypeParam>(), cv::saturate_cast<TypeParam>(2));

    EXPECT_EQ(this->matrix1x2.getDims(), 2);
    EXPECT_EQ(this->matrix1x2.getDims(), 2);

    EXPECT_EQ(this->matrix2x2.getDims(), 2);
    EXPECT_EQ(this->matrix2x2.getDims(), 2);
    EXPECT_EQ(this->matrix2x2.getDims(), 2);
    EXPECT_EQ(this->matrix2x2.getDims(), 2);

    EXPECT_EQ(this->matrix1x1x1.getDims(), 3);
}
