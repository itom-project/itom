#include <iostream>

#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class operatorTest
    \brief Operator test for real data types

    This test class checks functionality of different operators for data objects.
*/
template <typename _Tp> class operatorTest : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        /*!< declaring multi dimensional test data objects */
        this->mat1_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->mat1_2d = ito::DataObject(2, 2, ito::getDataType2<_Tp *>());
        this->mat1_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->mat2_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->mat2_2d = ito::DataObject(2, 2, ito::getDataType2<_Tp *>());
        this->mat2_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->mat3_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->mat3_2d = ito::DataObject(2, 2, ito::getDataType2<_Tp *>());
        this->mat3_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->mat4_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->mat4_2d = ito::DataObject(2, 2, ito::getDataType2<_Tp *>());
        this->mat4_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->add_mat3_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->add_mat3_2d = ito::DataObject(4, 5, ito::getDataType2<_Tp *>());
        this->add_mat3_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->sub_mat3_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->sub_mat3_2d = ito::DataObject(4, 5, ito::getDataType2<_Tp *>());
        this->sub_mat3_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->mul_mat3_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->mul_mat3_2d = ito::DataObject(4, 5, ito::getDataType2<_Tp *>());
        this->mul_mat3_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->div_mat3_1d = ito::DataObject(1, 1, ito::getDataType2<_Tp *>());
        this->div_mat3_2d = ito::DataObject(4, 5, ito::getDataType2<_Tp *>());
        this->div_mat3_3d = ito::DataObject(3, 2, 4, ito::getDataType2<_Tp *>());

        this->mulCross_mat1_2d = ito::DataObject(2, 3, ito::getDataType2<_Tp *>());
        this->mulCross_mat2_2d = ito::DataObject(3, 2, ito::getDataType2<_Tp *>());
        this->mulCross_mat3_2d = ito::DataObject(2, 2, ito::getDataType2<_Tp *>());
    };

    virtual void TearDown(void){};
    ito::DataObject mat1_1d;
    ito::DataObject mat1_2d;
    ito::DataObject mat1_3d;

    ito::DataObject mat2_1d;
    ito::DataObject mat2_2d;
    ito::DataObject mat2_3d;

    ito::DataObject mat3_1d;
    ito::DataObject mat3_2d;
    ito::DataObject mat3_3d;

    ito::DataObject mat4_1d;
    ito::DataObject mat4_2d;
    ito::DataObject mat4_3d;

    ito::DataObject add_mat3_1d;
    ito::DataObject add_mat3_2d;
    ito::DataObject add_mat3_3d;

    ito::DataObject sub_mat3_1d;
    ito::DataObject sub_mat3_2d;
    ito::DataObject sub_mat3_3d;

    ito::DataObject mul_mat3_1d;
    ito::DataObject mul_mat3_2d;
    ito::DataObject mul_mat3_3d;

    ito::DataObject div_mat3_1d;
    ito::DataObject div_mat3_2d;
    ito::DataObject div_mat3_3d;

    ito::DataObject mulCross_mat1_2d;
    ito::DataObject mulCross_mat2_2d;
    ito::DataObject mulCross_mat3_2d;
};

TYPED_TEST_CASE(operatorTest, ItomRealDataTypes);
// TransTest
/*!
This test checks functionality of "trans(void)" function for 2 dimensional matrices
*/
TYPED_TEST(operatorTest, TransTest)
{
    TypeParam temp = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            temp++;
            this->mat1_2d.template at<TypeParam>(i, j) = cv::saturate_cast<TypeParam>(temp);
        }
    ito::DataObject transDObj = this->mat1_2d.trans();
    temp = 1;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            EXPECT_EQ(
                transDObj.at<TypeParam>(j, i),
                cv::saturate_cast<TypeParam>(temp++)); // checks if the transposed matrix elements are same as expected.
        }
    }
}

// AddTest
/*!
This test checks functionality of "+" operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, AddTest)
{
    this->mat1_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(3);
    this->mat2_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2);

    this->mat1_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2);
    this->mat1_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(3);
    this->mat1_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(4);
    this->mat1_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(5);
    this->mat2_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(2);
    this->mat2_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(3);

    this->mat1_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(4);
    this->mat2_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(1);

    this->add_mat3_1d = this->mat1_1d + this->mat2_1d;
    this->add_mat3_2d = this->mat1_2d + this->mat2_2d;
    this->add_mat3_3d = this->mat1_3d + this->mat2_3d;

    EXPECT_EQ(this->add_mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(5));

    EXPECT_EQ(this->add_mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(3));
    EXPECT_EQ(this->add_mat3_2d.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(4));
    EXPECT_EQ(this->add_mat3_2d.template at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(6));
    EXPECT_EQ(this->add_mat3_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(8));

    EXPECT_EQ(this->add_mat3_3d.template at<TypeParam>(0, 0, 3), cv::saturate_cast<TypeParam>(5));
}

// SubTest
/*!
  This test checks functionality of "-" operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, SubTest)
{
    this->mat1_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(3);
    this->mat2_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2);

    this->mat1_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2);
    this->mat1_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(3);
    this->mat1_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(4);
    this->mat1_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(5);
    this->mat2_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(2);
    this->mat2_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(3);

    this->mat1_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(4);
    this->mat2_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(1);

    this->sub_mat3_1d = this->mat1_1d - this->mat2_1d;
    this->sub_mat3_2d = this->mat1_2d - this->mat2_2d;
    this->sub_mat3_3d = this->mat1_3d - this->mat2_3d;
    EXPECT_EQ(this->sub_mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(1));

    EXPECT_EQ(this->sub_mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(1));
    EXPECT_EQ(this->sub_mat3_2d.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(2));
    EXPECT_EQ(this->sub_mat3_2d.template at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(2));
    EXPECT_EQ(this->sub_mat3_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(2));

    EXPECT_EQ(this->sub_mat3_3d.template at<TypeParam>(0, 0, 3), cv::saturate_cast<TypeParam>(3));
}
// MulFactor_test
/*!
   This test checks functionality of "*" (multiplication with constant factor) operator for 1, 2 and 3 dimensional
   matrices
*/
TYPED_TEST(operatorTest, MulFactor_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(2);
    this->mat1_2d = cv::saturate_cast<TypeParam>(5);
    this->mat1_3d = cv::saturate_cast<TypeParam>(3);

    this->mat3_1d = this->mat1_1d * 7;
    this->mat3_2d = this->mat1_2d * 10;
    this->mat3_3d = this->mat1_3d * 25;

    EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(14));
    EXPECT_EQ(this->mat3_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(50));
    EXPECT_EQ(this->mat3_3d.template at<TypeParam>(0, 1, 0), cv::saturate_cast<TypeParam>(75));
}

// MulFactor1_test
/*!
   This test checks functionality of "*=" (multiplication with constant factor) operator for 1, 2 and 3 dimensional
   matrices
*/
TYPED_TEST(operatorTest, MulFactor1_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(2);
    this->mat1_2d = cv::saturate_cast<TypeParam>(5);
    this->mat1_3d = cv::saturate_cast<TypeParam>(3);

    this->mat1_1d *= 7;
    this->mat1_2d *= 10;
    this->mat1_3d *= 25;

    EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(14));
    EXPECT_EQ(this->mat1_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(50));
    EXPECT_EQ(this->mat1_3d.template at<TypeParam>(0, 1, 0), cv::saturate_cast<TypeParam>(75));
}

// MulFactor2_test
/*!
   This test checks that a scalar multiplication with a double value is casted in the right way
   (no rounding, double is multiplied at first with double precision, then the cast to the data type of the data object
   should be executed)
*/
TYPED_TEST(operatorTest, MulFactor2_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(2);
    this->mat1_2d = cv::saturate_cast<TypeParam>(5);
    this->mat1_3d = cv::saturate_cast<TypeParam>(3);

    this->mat1_1d *= 7.2;
    this->mat1_2d *= 10.8;
    this->mat1_3d *= 0.5;

    EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2 * (double)7.2));
    EXPECT_EQ(this->mat1_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(5 * (double)10.8));
    EXPECT_EQ(this->mat1_3d.template at<TypeParam>(0, 1, 0), cv::saturate_cast<TypeParam>(3 * (double)0.5));
}

// MulCross_test
/*!
   This test checks functionality of "*" (cross multiplication of matrices) operator for 1 and 2 dimensional matrices
*/
TYPED_TEST(operatorTest, MulCross_test) // Note: 1)exception in 3d cross multiplication
                                        // Note: 2)true for only "float32" and "float64" datatypes.
{
    unsigned int res_str[] = {22, 28, 49, 64};
    for (int i = 1; i <= 2; i++)
    {
        for (int j = 1; j <= 3; j++)
            this->mulCross_mat1_2d.template at<TypeParam>(i - 1, j - 1) = cv::saturate_cast<TypeParam>(3 * (i - 1) + j);
    }
    for (int i = 1; i <= 3; i++)
    {
        for (int j = 1; j <= 2; j++)
            this->mulCross_mat2_2d.template at<TypeParam>(i - 1, j - 1) = cv::saturate_cast<TypeParam>(2 * (i - 1) + j);
    }

    if (std::numeric_limits<TypeParam>::is_integer == false) // floating point numbers
    {
        this->mulCross_mat3_2d = this->mulCross_mat1_2d * this->mulCross_mat2_2d;
        int temp = 0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
                EXPECT_NEAR(this->mulCross_mat3_2d.template at<TypeParam>(i, j),
                            cv::saturate_cast<TypeParam>(res_str[temp++]), std::numeric_limits<TypeParam>::epsilon());
        }
    }
    else
    {
        EXPECT_THROW({ this->mulCross_mat1_2d *this->mulCross_mat2_2d; }, cv::Exception);
    }
}

// MulDotTest
/*!
   This test checks functionality of "mul" (elementwise multiplication) for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, MulDotTest)
{
    this->mat1_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(3);
    this->mat2_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2);

    this->mat1_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2);
    this->mat1_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(3);
    this->mat1_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(4);
    this->mat1_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(5);
    this->mat2_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(2);
    this->mat2_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(3);

    this->mat1_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(4);
    this->mat2_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(1);

    this->mul_mat3_1d = this->mat1_1d.mul(this->mat2_1d);
    this->mul_mat3_2d = this->mat1_2d.mul(this->mat2_2d);
    this->mul_mat3_3d = this->mat1_3d.mul(this->mat2_3d);

    EXPECT_EQ(this->mul_mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(6));

    EXPECT_EQ(this->mul_mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2));
    EXPECT_EQ(this->mul_mat3_2d.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(3));
    EXPECT_EQ(this->mul_mat3_2d.template at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(8));
    EXPECT_EQ(this->mul_mat3_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(15));

    EXPECT_EQ(this->mul_mat3_3d.template at<TypeParam>(0, 0, 3), cv::saturate_cast<TypeParam>(4));
    // std::cout << this->mat2_2d << std::endl;
}

// divTest
/*!
   This test checks functionality of "div" (elementwise division) for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, divTest)
{
    this->mat1_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(20.0);
    this->mat2_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2.0);

    this->mat1_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(2.0);
    this->mat1_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(3.0);
    this->mat1_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(4.0);
    this->mat1_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(5.0);
    this->mat2_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(1.0);
    this->mat2_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(1.0);
    this->mat2_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(2.0);
    this->mat2_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(4.0);

    this->mat1_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(4);
    this->mat2_3d = 6;
    this->mat2_3d.template at<TypeParam>(0, 0, 3) = cv::saturate_cast<TypeParam>(1);

    this->div_mat3_1d = this->mat1_1d.div(this->mat2_1d);
    this->div_mat3_2d = this->mat1_2d.div(this->mat2_2d);
    this->div_mat3_3d = this->mat1_3d.div(this->mat2_3d);

    EXPECT_EQ(this->div_mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(20.0 / 2.0));
    EXPECT_EQ(this->div_mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2.0 / 1.0));
    EXPECT_EQ(this->div_mat3_2d.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(3.0 / 1.0));
    EXPECT_EQ(this->div_mat3_2d.template at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(4.0 / 2.0));
    EXPECT_EQ(this->div_mat3_2d.template at<TypeParam>(1, 1), cv::saturate_cast<TypeParam>(5.0 / 4.0));

    EXPECT_EQ(this->div_mat3_3d.template at<TypeParam>(0, 0, 3), cv::saturate_cast<TypeParam>(4.0 / 1.0));
}

// Add1_test
/*!
   This test checks functionality of "+=" operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, Add1_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(3.0);
    this->mat2_1d = cv::saturate_cast<TypeParam>(2.0);

    this->mat1_2d = cv::saturate_cast<TypeParam>(2.0);

    this->mat2_2d = cv::saturate_cast<TypeParam>(1.0);

    this->mat1_3d = cv::saturate_cast<TypeParam>(4);
    this->mat2_3d = cv::saturate_cast<TypeParam>(1);

    this->mat2_1d += this->mat1_1d;
    this->mat2_2d += this->mat1_2d;
    this->mat2_3d += this->mat1_3d;

    EXPECT_EQ(this->mat2_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(3.0 + 2.0));
    EXPECT_EQ(this->mat2_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2.0 + 1.0));
    EXPECT_EQ(this->mat2_3d.template at<TypeParam>(0, 0, 3), cv::saturate_cast<TypeParam>(4.0 + 1.0));
}

// sub1_test
/*!
   This test checks functionality of "-=" operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, sub1_test) // Note:Test Fails for datatype Float
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(3.0);
    this->mat2_1d = cv::saturate_cast<TypeParam>(2.0);

    this->mat1_2d = cv::saturate_cast<TypeParam>(2.0);

    this->mat2_2d = cv::saturate_cast<TypeParam>(1.0);

    this->mat1_3d = cv::saturate_cast<TypeParam>(4);
    this->mat2_3d = cv::saturate_cast<TypeParam>(1);

    this->mat2_1d -= this->mat1_1d;
    this->mat1_2d -= this->mat2_2d;
    this->mat1_3d -= this->mat2_3d;

    EXPECT_EQ(this->mat2_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2.0 - 3.0));
    EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2.0 - 1.0));
    EXPECT_EQ(this->mat1_3d.template at<TypeParam>(0, 0, 3), cv::saturate_cast<TypeParam>(4.0 - 1.0));
}

// MulCross1_test
/*!
   This test checks functionality of "*=" (cross multiplication and assign) operator for 1 and 2 dimensional matrices
*/
TYPED_TEST(operatorTest, MulCross1_test)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(1);
    this->mat2_2d = cv::saturate_cast<TypeParam>(1);

    if (std::numeric_limits<TypeParam>::is_integer == false) // floating point numbers
    {
        this->mat1_2d *= this->mat2_2d;
        EXPECT_NEAR(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2),
                    std::numeric_limits<TypeParam>::epsilon());
    }
    else
    {
        EXPECT_THROW({ this->mat1_2d *= this->mat2_2d; }, cv::Exception);
    }
}

// CompareEQ_test
/*!
   This test checks functionality of "==" (Equal to) operator for 1 and 2 dimensional matrices
*/
TYPED_TEST(operatorTest, CompareEQ_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(1);
    this->mat2_1d = cv::saturate_cast<TypeParam>(1);

    this->mat1_2d.template at<TypeParam>(0, 0) = 3;
    this->mat1_2d.template at<TypeParam>(0, 1) = 2;
    this->mat1_2d.template at<TypeParam>(1, 0) = 5;
    this->mat1_2d.template at<TypeParam>(1, 1) = 8;

    this->mat2_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(3);
    this->mat2_2d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(3);
    this->mat2_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(3);
    this->mat2_2d.template at<TypeParam>(1, 1) = cv::saturate_cast<TypeParam>(8);

    this->mat1_3d.template at<TypeParam>(0, 0, 0) = cv::saturate_cast<TypeParam>(25);
    this->mat1_3d.template at<TypeParam>(0, 0, 1) = cv::saturate_cast<TypeParam>(12);
    this->mat1_3d.template at<TypeParam>(0, 1, 0) = cv::saturate_cast<TypeParam>(34);
    this->mat1_3d.template at<TypeParam>(1, 0, 2) = cv::saturate_cast<TypeParam>(65);
    this->mat1_3d.template at<TypeParam>(1, 1, 3) = cv::saturate_cast<TypeParam>(34);
    this->mat1_3d.template at<TypeParam>(2, 0, 0) = cv::saturate_cast<TypeParam>(85);
    this->mat1_3d.template at<TypeParam>(2, 0, 1) = cv::saturate_cast<TypeParam>(84);
    this->mat1_3d.template at<TypeParam>(2, 1, 0) = cv::saturate_cast<TypeParam>(95);
    this->mat1_3d.template at<TypeParam>(2, 1, 1) = cv::saturate_cast<TypeParam>(149);

    this->mat2_3d.template at<TypeParam>(0, 0, 0) = cv::saturate_cast<TypeParam>(25);
    this->mat2_3d.template at<TypeParam>(0, 0, 1) = cv::saturate_cast<TypeParam>(
        56); //!< This value is not kept equal to the same as on the same location in this->mat1_3d for testing purpose
    this->mat2_3d.template at<TypeParam>(0, 1, 0) = cv::saturate_cast<TypeParam>(34);
    this->mat2_3d.template at<TypeParam>(1, 0, 2) = cv::saturate_cast<TypeParam>(65);
    this->mat2_3d.template at<TypeParam>(1, 1, 3) = cv::saturate_cast<TypeParam>(
        35); //!< This value is not kept equal to the same as on the same location in this->mat1_3d for testing purpose
    this->mat2_3d.template at<TypeParam>(2, 0, 0) = cv::saturate_cast<TypeParam>(85);
    this->mat2_3d.template at<TypeParam>(2, 0, 1) = cv::saturate_cast<TypeParam>(
        82); //!< This value is not kept equal to the same as on the same location in this->mat1_3d for testing purpose
    this->mat2_3d.template at<TypeParam>(2, 1, 0) = cv::saturate_cast<TypeParam>(95);
    this->mat2_3d.template at<TypeParam>(2, 1, 1) = cv::saturate_cast<TypeParam>(149);

    if (std::numeric_limits<TypeParam>::max() ==
        std::numeric_limits<ito::int8>::max()) // compare not implemented for int8
    {
        EXPECT_THROW({ this->mat1_1d == this->mat2_1d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_2d == this->mat2_2d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_3d == this->mat2_3d; }, cv::Exception);
    }
    else
    {
        this->mat3_1d = (this->mat1_1d == this->mat2_1d);
        this->mat3_2d = (this->mat1_2d == this->mat2_2d);
        this->mat3_3d = (this->mat1_3d == this->mat2_3d);

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));

        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0),
                  cv::saturate_cast<ito::uint8>(
                      255)); //!< The values of the elements on locations (0,0) and (1,1) in both matrices are kept
                             //!< intentionally similar, which should be resulted as '255' in the result matrix
                             //!< this->mat3_2d on those same locations.
        EXPECT_EQ(
            this->mat3_2d.template at<ito::uint8>(0, 1),
            cv::saturate_cast<ito::uint8>(0)); //!< The values of the elements on locations (0,1) and (1,0) in both
                                               //!< matrices are kept intentionally different, which should be resulted
                                               //!< as '0' in the result matrix this->mat3_2d on those same locations.
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(1, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(1, 1), cv::saturate_cast<ito::uint8>(255));

        // EXPECT_EQ ( this->mat3_3d.template at<ito::uint8>(0,0,0), cv::saturate_cast<ito::uint8>(255));
        // EXPECT_NE ( this->mat3_3d.template at<ito::uint8>(0,0,0), cv::saturate_cast<ito::uint8>(0));

        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(0, 0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_EQ(
            this->mat3_3d.template at<ito::uint8>(0, 0, 1),
            cv::saturate_cast<ito::uint8>(0)); //!< Values of the elements on this location are kept different in both
                                               //!< matrices this->mat1_3d and this->mat2_3d to check if the resultant
                                               //!< matrix this->mat3_3d contains '0' value on the same location.
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(0, 1, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 0, 2), cv::saturate_cast<ito::uint8>(255));
        EXPECT_EQ(
            this->mat3_3d.template at<ito::uint8>(1, 1, 3),
            cv::saturate_cast<ito::uint8>(0)); //!< Values of the elements on this location are kept different in both
                                               //!< matrices this->mat1_3d and this->mat2_3d to check if the resultant
                                               //!< matrix this->mat3_3d contains '0' value on the same location.
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_EQ(
            this->mat3_3d.template at<ito::uint8>(2, 0, 1),
            cv::saturate_cast<ito::uint8>(0)); //!< Values of the elements on this location are kept different in both
                                               //!< matrices this->mat1_3d and this->mat2_3d to check if the resultant
                                               //!< matrix this->mat3_3d contains '0' value on the same location.
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 1, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 1, 1), cv::saturate_cast<ito::uint8>(255));

        this->mat3_1d = (this->mat1_1d == 1);
        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), 255);

        this->mat3_2d = (this->mat1_2d == 5.0);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), 0);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(1, 0), 255);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 1), 0);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(1, 1), 0);

        this->mat3_3d = (this->mat1_3d == 65);
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(0, 0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(0, 0, 1), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(0, 1, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 0, 2), cv::saturate_cast<ito::uint8>(255));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 3), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 0, 1), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 1, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(2, 1, 1), cv::saturate_cast<ito::uint8>(0));
    }
}

// CompareNE_test
/*!
   This test checks functionality of "!=" (Not Equal To) operator for 1 and 2 dimensional matrices
*/
TYPED_TEST(operatorTest, CompareNE_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(1);
    this->mat2_1d = cv::saturate_cast<TypeParam>(3);

    this->mat1_2d = cv::saturate_cast<TypeParam>(5);
    this->mat2_2d = cv::saturate_cast<TypeParam>(2);

    this->mat1_3d = cv::saturate_cast<TypeParam>(25);
    this->mat2_3d = cv::saturate_cast<TypeParam>(26);

    if (std::numeric_limits<TypeParam>::max() ==
        std::numeric_limits<ito::int8>::max()) // compare not implemented for int8
    {
        EXPECT_THROW({ this->mat1_1d != this->mat2_1d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_2d != this->mat2_2d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_3d != this->mat2_3d; }, cv::Exception);
    }
    else
    {
        this->mat3_1d = this->mat1_1d != this->mat2_1d;
        this->mat3_2d = this->mat1_2d != this->mat2_2d;
        this->mat3_3d = this->mat1_3d != this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(0));
    }
}

// CompareLE_test
/*!
   This test checks functionality of "<=" (Less Than or Equal to) operator for 1 and 2 dimensional matrices
*/
TYPED_TEST(operatorTest, CompareLE_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(1);
    this->mat2_1d = cv::saturate_cast<TypeParam>(3);

    this->mat1_2d = cv::saturate_cast<TypeParam>(5);
    this->mat2_2d = cv::saturate_cast<TypeParam>(7);

    this->mat1_3d = cv::saturate_cast<TypeParam>(25);
    this->mat2_3d = cv::saturate_cast<TypeParam>(25);

    if (std::numeric_limits<TypeParam>::max() ==
        std::numeric_limits<ito::int8>::max()) // compare not implemented for int8
    {
        EXPECT_THROW({ this->mat1_1d <= this->mat2_1d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_2d <= this->mat2_2d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_3d <= this->mat2_3d; }, cv::Exception);
    }
    else
    {
        this->mat3_1d = this->mat1_1d <= this->mat2_1d;
        this->mat3_2d = this->mat1_2d <= this->mat2_2d;
        this->mat3_3d = this->mat1_3d <= this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(0));
    }
}
// CompareGE_test
/*!
    This test checks functionality of ">=" (Greater than or equal to) operator for 1 and 2 dimensional matrices
*/
TYPED_TEST(operatorTest, CompareGE_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(10);
    this->mat2_1d = cv::saturate_cast<TypeParam>(3);

    this->mat1_2d = cv::saturate_cast<TypeParam>(7);
    this->mat2_2d = cv::saturate_cast<TypeParam>(7);

    this->mat1_3d = cv::saturate_cast<TypeParam>(29);
    this->mat2_3d = cv::saturate_cast<TypeParam>(25);

    if (std::numeric_limits<TypeParam>::max() ==
        std::numeric_limits<ito::int8>::max()) // compare not implemented for int8
    {
        EXPECT_THROW({ this->mat1_1d >= this->mat2_1d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_2d >= this->mat2_2d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_3d >= this->mat2_3d; }, cv::Exception);
    }
    else
    {
        this->mat3_1d = this->mat1_1d >= this->mat2_1d;
        this->mat3_2d = this->mat1_2d >= this->mat2_2d;
        this->mat3_3d = this->mat1_3d >= this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(0));

        this->mat3_1d = this->mat1_1d >= 7;
        this->mat3_2d = this->mat1_2d >= 7;
        this->mat3_3d = this->mat1_3d >= 7;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), 255);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), 255);
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), 255);

        this->mat3_1d = this->mat1_1d >= 35;
        this->mat3_2d = this->mat1_2d >= 35;
        this->mat3_3d = this->mat1_3d >= 35;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), 0);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), 0);
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), 0);
    }
}

// CompareLT_test
/*!
    This test checks functionality of "<" (less than) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, CompareLT_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(10);
    this->mat2_1d = cv::saturate_cast<TypeParam>(13);

    this->mat1_2d = cv::saturate_cast<TypeParam>(7);
    this->mat2_2d = cv::saturate_cast<TypeParam>(9);

    this->mat1_3d = cv::saturate_cast<TypeParam>(21);
    this->mat2_3d = cv::saturate_cast<TypeParam>(25);

    if (std::numeric_limits<TypeParam>::max() ==
        std::numeric_limits<ito::int8>::max()) // compare not implemented for int8
    {
        EXPECT_THROW({ this->mat1_1d < this->mat2_1d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_2d < this->mat2_2d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_3d < this->mat2_3d; }, cv::Exception);
    }
    else
    {
        this->mat3_1d = this->mat1_1d < this->mat2_1d;
        this->mat3_2d = this->mat1_2d < this->mat2_2d;
        this->mat3_3d = this->mat1_3d < this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(0));

        this->mat3_1d = this->mat1_1d < 7;
        this->mat3_2d = this->mat1_2d < 7;
        this->mat3_3d = this->mat1_3d < 7;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), 0);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), 0);
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), 0);

        this->mat3_1d = this->mat1_1d < 35;
        this->mat3_2d = this->mat1_2d < 35;
        this->mat3_3d = this->mat1_3d < 35;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), 255);
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), 255);
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), 255);
    }
}

// CompareGT_test
/*!
    This test checks functionality of ">" (greater than) operator for 1,2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, CompareGT_test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(17);
    this->mat2_1d = cv::saturate_cast<TypeParam>(13);

    this->mat1_2d = cv::saturate_cast<TypeParam>(10);
    this->mat2_2d = cv::saturate_cast<TypeParam>(9);

    this->mat1_3d = cv::saturate_cast<TypeParam>(26);
    this->mat2_3d = cv::saturate_cast<TypeParam>(22);

    if (std::numeric_limits<TypeParam>::max() ==
        std::numeric_limits<ito::int8>::max()) // compare not implemented for int8
    {
        EXPECT_THROW({ this->mat1_1d > this->mat2_1d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_2d > this->mat2_2d; }, cv::Exception);
        EXPECT_THROW({ this->mat1_3d > this->mat2_3d; }, cv::Exception);
    }
    else
    {
        this->mat3_1d = this->mat1_1d > this->mat2_1d;
        this->mat3_2d = this->mat1_2d > this->mat2_2d;
        this->mat3_3d = this->mat1_3d > this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_1d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_2d.template at<ito::uint8>(0, 0), cv::saturate_cast<ito::uint8>(0));
        EXPECT_EQ(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(255));
        EXPECT_NE(this->mat3_3d.template at<ito::uint8>(1, 1, 1), cv::saturate_cast<ito::uint8>(0));
    }
}

// ShiftL_test
/*!
    This test checks functionality of "<<" (shift left) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, ShiftL_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(17);
    this->mat1_2d = cv::saturate_cast<TypeParam>(2);
    this->mat1_3d = cv::saturate_cast<TypeParam>(4);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat3_1d = this->mat1_1d << 1;
        this->mat3_2d = this->mat1_2d << 1;
        this->mat3_3d = this->mat1_3d << 2;

        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(34));
        EXPECT_EQ(this->mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(4));
        EXPECT_EQ(this->mat3_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(16));
    }
    else
    {
        EXPECT_THROW(this->mat3_1d = this->mat1_1d << 1, cv::Exception);
    }
}

// ShiftR_test
/*!
    This test checks functionality of ">>" (shift right) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, ShiftR_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(18);
    this->mat1_2d = cv::saturate_cast<TypeParam>(2);
    this->mat1_3d = cv::saturate_cast<TypeParam>(24);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat3_1d = this->mat1_1d >> 1;
        this->mat3_2d = this->mat1_2d >> 1;
        this->mat3_3d = this->mat1_3d >> 2;

        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(9));
        EXPECT_EQ(this->mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(1));
        EXPECT_EQ(this->mat3_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(6));
    }
    else
    {
        EXPECT_THROW(this->mat3_1d = this->mat1_1d >> 1, cv::Exception);
    }
}

// ShiftL1_test
/*!
    This test checks functionality of "<<=" (shift left and assign) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, ShiftL1_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(18);
    this->mat1_2d = cv::saturate_cast<TypeParam>(2);
    this->mat1_3d = cv::saturate_cast<TypeParam>(24);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat1_1d <<= 1;
        this->mat1_2d <<= 1;
        this->mat1_3d <<= 2;

        std::cout << this->mat1_1d << std::endl;
        std::cout << this->mat1_2d << std::endl;
        std::cout << this->mat1_3d << std::endl;

        EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(36));
        EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(4));
        EXPECT_EQ(this->mat1_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(96));
    }
    else
    {
        EXPECT_THROW(this->mat1_1d <<= 1, cv::Exception);
    }
}

// ShiftR1_test
/*!
    This test checks functionality of ">>=" (shift right and assign) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, ShiftR1_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(18);
    this->mat1_2d = cv::saturate_cast<TypeParam>(4);
    this->mat1_3d = cv::saturate_cast<TypeParam>(24);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat1_1d >>= 1;
        this->mat1_2d >>= 1;
        this->mat1_3d >>= 2;

        std::cout << this->mat1_1d << std::endl;
        std::cout << this->mat1_2d << std::endl;
        std::cout << this->mat1_3d << std::endl;
        EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(9));
        EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2));
        EXPECT_EQ(this->mat1_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(6));
    }
    else
    {
        EXPECT_THROW(this->mat1_1d >>= 1, cv::Exception);
    }
}

// BitAND_test
/*!
    This test checks functionality of "&" (bitwise AND) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, BitAND_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(10);
    this->mat2_1d = cv::saturate_cast<TypeParam>(3);

    this->mat1_2d = cv::saturate_cast<TypeParam>(15);
    this->mat2_2d = cv::saturate_cast<TypeParam>(4);

    this->mat1_3d = cv::saturate_cast<TypeParam>(7);
    this->mat2_3d = cv::saturate_cast<TypeParam>(8);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat3_1d = this->mat1_1d & this->mat2_1d;
        this->mat3_2d = this->mat1_2d & this->mat2_2d;
        this->mat3_3d = this->mat1_3d & this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(2));
        EXPECT_EQ(this->mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(4));
        EXPECT_EQ(this->mat3_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(0));
    }
    else
    {
        EXPECT_THROW(this->mat3_1d = this->mat1_1d & this->mat2_1d, cv::Exception);
    }
}

// BitOR_test
/*!
    This test checks functionality of "|" (bitwise OR) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, BitOR_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(10);
    this->mat2_1d = cv::saturate_cast<TypeParam>(3);

    this->mat1_2d = cv::saturate_cast<TypeParam>(15);
    this->mat2_2d = cv::saturate_cast<TypeParam>(4);

    this->mat1_3d = cv::saturate_cast<TypeParam>(7);
    this->mat2_3d = cv::saturate_cast<TypeParam>(8);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat3_1d = this->mat1_1d | this->mat2_1d;
        this->mat3_2d = this->mat1_2d | this->mat2_2d;
        this->mat3_3d = this->mat1_3d | this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(11));
        EXPECT_EQ(this->mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(15));
        EXPECT_EQ(this->mat3_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(15));
    }
    else
    {
        EXPECT_THROW(this->mat3_1d = this->mat1_1d | this->mat2_1d, cv::Exception);
    }
}

// BitNOT_Test
/*!
    This test checks functionality of "^" (bitwise NOT) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, BitNOT_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(7);
    this->mat2_1d = cv::saturate_cast<TypeParam>(4);

    this->mat1_2d = cv::saturate_cast<TypeParam>(8);
    this->mat2_2d = cv::saturate_cast<TypeParam>(11);

    this->mat1_3d = cv::saturate_cast<TypeParam>(12);
    this->mat2_3d = cv::saturate_cast<TypeParam>(3);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat3_1d = this->mat1_1d ^ this->mat2_1d;
        this->mat3_2d = this->mat1_2d ^ this->mat2_2d;
        this->mat3_3d = this->mat1_3d ^ this->mat2_3d;

        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(3));
        EXPECT_EQ(this->mat3_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(3));
        EXPECT_EQ(this->mat3_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(15));
    }
    else
    {
        EXPECT_THROW(this->mat3_1d = this->mat1_1d ^ this->mat2_1d, cv::Exception);
    }
}

// BitAND1_test
/*!
    This test checks functionality of "&=" (bitwise AND and assign) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, BitAND1_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(25);
    this->mat2_1d = cv::saturate_cast<TypeParam>(15);

    this->mat1_2d = cv::saturate_cast<TypeParam>(48);
    this->mat2_2d = cv::saturate_cast<TypeParam>(12);

    this->mat1_3d = cv::saturate_cast<TypeParam>(13);
    this->mat2_3d = cv::saturate_cast<TypeParam>(21);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat1_1d &= this->mat2_1d;
        this->mat1_2d &= this->mat2_2d;
        this->mat1_3d &= this->mat2_3d;

        EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(9));
        EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(0));
        EXPECT_EQ(this->mat1_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(5));
    }
    else
    {
        EXPECT_THROW(this->mat1_1d &= this->mat2_1d, cv::Exception);
    }
}

// BitOR1_test
/*!
    This test checks functionality of "|=" (bitwise OR and assign) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, BitOR1_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(7);
    this->mat2_1d = cv::saturate_cast<TypeParam>(4);

    this->mat1_2d = cv::saturate_cast<TypeParam>(12);
    this->mat2_2d = cv::saturate_cast<TypeParam>(23);

    this->mat1_3d = cv::saturate_cast<TypeParam>(28);
    this->mat2_3d = cv::saturate_cast<TypeParam>(5);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat1_1d |= this->mat2_1d;
        this->mat1_2d |= this->mat2_2d;
        this->mat1_3d |= this->mat2_3d;

        EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(7));
        EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(31));
        EXPECT_EQ(this->mat1_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(29));
    }
    else
    {
        EXPECT_THROW(this->mat1_1d |= this->mat2_1d, cv::Exception);
    }
}

// BitNOT1_test
/*!
    This test checks functionality of "^=" (bitwise NOT and assign) operator for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, BitNOT1_test) // Note: Test fails for datatypes "float32" and "float64"
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(43);
    this->mat2_1d = cv::saturate_cast<TypeParam>(27);

    this->mat1_2d = cv::saturate_cast<TypeParam>(2);
    this->mat2_2d = cv::saturate_cast<TypeParam>(15);

    this->mat1_3d = cv::saturate_cast<TypeParam>(14);
    this->mat2_3d = cv::saturate_cast<TypeParam>(9);

    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat1_1d ^= this->mat2_1d;
        this->mat1_2d ^= this->mat2_2d;
        this->mat1_3d ^= this->mat2_3d;

        EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(48));
        EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(13));
        EXPECT_EQ(this->mat1_3d.template at<TypeParam>(1, 1, 1), cv::saturate_cast<TypeParam>(7));
    }
    else
    {
        EXPECT_THROW(this->mat1_1d ^= this->mat2_1d, cv::Exception);
    }
}

// Combination_Arith_Test
/*!
    This test checks functionality of mixed arithmetic operators for 1, 2 and 3 dimensional matrices
*/
TYPED_TEST(operatorTest, Combination_Arith_Test)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(15);
    this->mat2_1d = cv::saturate_cast<TypeParam>(3);
    this->mat3_1d = cv::saturate_cast<TypeParam>(5);

    this->mat4_1d = this->mat1_1d + this->mat2_1d - this->mat3_1d;
    EXPECT_EQ(this->mat4_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(15 + 3 - 5));
    this->mat4_1d = this->mat1_1d.mul(this->mat2_1d) - this->mat1_1d.div(this->mat3_1d);
    EXPECT_EQ(this->mat4_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>((15 * 3) - (15 / 5)));
    this->mat4_1d = cv::saturate_cast<TypeParam>(3);
    this->mat4_1d += this->mat1_1d.div(this->mat2_1d) + this->mat1_1d.div(this->mat2_1d.mul(this->mat3_1d));
    EXPECT_EQ(this->mat4_1d.template at<TypeParam>(0, 0),
              cv::saturate_cast<TypeParam>(3 + ((15 / 3) + (15 / (3 * 5)))));
    this->mat4_1d = cv::saturate_cast<TypeParam>(20);
    this->mat4_1d -= (this->mat1_1d + this->mat2_1d).div(this->mat3_1d - this->mat2_1d) +
                     (this->mat1_1d * 2).div(this->mat2_1d.mul(this->mat3_1d));
    EXPECT_EQ(this->mat4_1d.template at<TypeParam>(0, 0),
              cv::saturate_cast<TypeParam>(20 - ((15 + 3) / (5 - 3) + ((15 * 2) / (5 * 3)))));
    if (std::numeric_limits<TypeParam>::is_exact)
    {
        this->mat2_1d = cv::saturate_cast<TypeParam>(15);
        this->mat1_1d.ones(1, 1, ito::getDataType((const TypeParam *)NULL));
        this->mat4_1d = (this->mat1_1d & this->mat2_1d).mul(this->mat2_1d >> 2);
        EXPECT_EQ(this->mat4_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(3));

        this->mat4_1d = (this->mat1_1d | this->mat2_1d).div((this->mat1_1d << 1) + this->mat1_1d);
        EXPECT_EQ(this->mat4_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(5));
    }
}
/*! \class operatorTest_float
    \brief Operator test for float data types

    This test class specially checks functionality of different operators ('+','-','=+','=-') for data objects of float
   datatype.
*/
template <typename _Tp> class operatorTest_float : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        this->mat1_1d = ito::DataObject(3, ito::getDataType2<_Tp *>());
        this->mat2_1d = ito::DataObject(3, ito::getDataType2<_Tp *>());
        this->mat2_1d = ito::DataObject(3, ito::getDataType2<_Tp *>());

        this->mat1_2d = ito::DataObject(3, 4, ito::getDataType2<_Tp *>());
        this->mat2_2d = ito::DataObject(3, 4, ito::getDataType2<_Tp *>());
        this->mat2_2d = ito::DataObject(3, 4, ito::getDataType2<_Tp *>());

        this->mat1_3d = ito::DataObject(3, 3, 3, ito::getDataType2<_Tp *>());
        this->mat2_3d = ito::DataObject(3, 3, 3, ito::getDataType2<_Tp *>());
        this->mat2_3d = ito::DataObject(3, 3, 3, ito::getDataType2<_Tp *>());
    };
    virtual void TearDown(void){};

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

TYPED_TEST_CASE(operatorTest_float, ItomFloatDoubleDataTypes);

// Float_Add_Test1d
/*!
    This test checks functionality of "+" operator for 1 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Add_Test1d)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(245.313);
    this->mat2_1d = cv::saturate_cast<TypeParam>(7465.3768);
    this->mat3_1d = this->mat1_1d + this->mat2_1d;
    for (int i = 0; i < 2; i++)
        EXPECT_FLOAT_EQ(this->mat3_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(245.313 + 7465.3768));
};
// Float_Add_Test2d
/*!
    This test checks functionality of "+" operator for 2 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Add_Test2d)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(12.33);
    this->mat2_2d = cv::saturate_cast<TypeParam>(34.66);
    this->mat3_2d = this->mat1_2d + this->mat2_2d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_FLOAT_EQ(this->mat3_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(12.33 + 34.66));
};
// Float_Add_Test3d
/*!
    This test checks functionality of "+" operator for 3 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Add_Test3d)
{
    this->mat1_3d = cv::saturate_cast<TypeParam>(-2354.34);
    this->mat2_3d = cv::saturate_cast<TypeParam>(897.345);
    this->mat3_3d = this->mat1_3d + this->mat2_3d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat3_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-2354.34 + 897.345));
};

// Float_Sub_Test1d
/*!
    This test checks functionality of "-" operator for 1 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Sub_Test1d)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(245.313);
    this->mat2_1d = cv::saturate_cast<TypeParam>(7465.3768);
    this->mat3_1d = this->mat1_1d - this->mat2_1d;
    for (int i = 0; i < 2; i++)
        EXPECT_FLOAT_EQ(this->mat3_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(245.313 - 7465.3768));
};
// Float_Sub_Test2d
/*!
    This test checks functionality of "-" operator for 2 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Sub_Test2d)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(12.33);
    this->mat2_2d = cv::saturate_cast<TypeParam>(34.66);
    this->mat3_2d = this->mat1_2d - this->mat2_2d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_FLOAT_EQ(this->mat3_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(12.33 - 34.66));
};
// Float_Sub_Test3d
/*!
    This test checks functionality of "-" operator for 3 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Sub_Test3d)
{
    this->mat1_3d = cv::saturate_cast<TypeParam>(-2354.34);
    this->mat2_3d = cv::saturate_cast<TypeParam>(897.345);
    this->mat3_3d = this->mat1_3d - this->mat2_3d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat3_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-2354.34 - 897.345));
};

// Float_Add_Test1d
/*!
    This test checks functionality of "+=" operator for 1 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Add1_Test1d)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(245.313);
    this->mat2_1d = cv::saturate_cast<TypeParam>(7465.3768);
    this->mat1_1d += this->mat2_1d;
    for (int i = 0; i < 2; i++)
        EXPECT_FLOAT_EQ(this->mat1_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(245.313 + 7465.3768));
};
// Float_Add1_Test2d
/*!
    This test checks functionality of "+=" operator for 2 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Add1_Test2d)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(12.33);
    this->mat2_2d = cv::saturate_cast<TypeParam>(34.66);
    this->mat1_2d += this->mat2_2d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_FLOAT_EQ(this->mat1_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(12.33 + 34.66));
};
// Float_Add1_Test3d
/*!
    This test checks functionality of "+=" operator for 3 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Add1_Test3d)
{
    this->mat1_3d = cv::saturate_cast<TypeParam>(-2354.34);
    this->mat2_3d = cv::saturate_cast<TypeParam>(897.345);
    this->mat1_3d += this->mat2_3d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat1_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-2354.34 + 897.345));
};

// Float_Sub1_Test1d
/*!
    This test checks functionality of "-=" operator for 1 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Sub1_Test1d)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(245.313);
    this->mat2_1d = cv::saturate_cast<TypeParam>(7465.3768);
    this->mat1_1d -= this->mat2_1d;
    for (int i = 0; i < 2; i++)
        EXPECT_FLOAT_EQ(this->mat1_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(245.313 - 7465.3768));
};
// Float_Sub1_Test2d
/*!
    This test checks functionality of "-=" operator for 2 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Sub1_Test2d)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(12.33);
    this->mat2_2d = cv::saturate_cast<TypeParam>(34.66);
    this->mat1_2d -= this->mat2_2d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_FLOAT_EQ(this->mat1_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(12.33 - 34.66));
};
// Float_Sub1_Test3d
/*!
    This test checks functionality of "-=" operator for 3 dimensional matrices
*/
TYPED_TEST(operatorTest_float, Float_Sub1_Test3d)
{
    this->mat1_3d = cv::saturate_cast<TypeParam>(-2354.34);
    this->mat2_3d = cv::saturate_cast<TypeParam>(897.345);
    this->mat1_3d -= this->mat2_3d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat1_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-2354.34 - 897.345));
};
