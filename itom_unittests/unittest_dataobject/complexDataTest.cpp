
#include "../../common/sharedStructures.h"

// opencv
#pragma warning(disable : 4996) // C:\OpenCV2.3\build\include\opencv2/flann/logger.h(70): warning C4996: 'fopen': This
                                // function or variable may be unsafe. Consider using fopen_s instead.

#include "../../DataObject/dataobj.h"
#include "commonChannel.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

/*! \class operatorComplexTest
    \brief test for div and mul methods for complex datatype (ItomComplexDataTypes) data objects

    This test class checks functionality of div and mul methods for data objects with complex data types.
*/
template <typename _Tp> class operatorComplexTest : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        // Creating DataObjects for this Perticular Test class.
        mat1_1d = ito::DataObject(3, ito::getDataType2<_Tp *>());
        mat2_1d = ito::DataObject(3, ito::getDataType2<_Tp *>());
        mat3_1d = ito::DataObject(3, ito::getDataType2<_Tp *>());

        mat1_2d = ito::DataObject(3, 3, ito::getDataType2<_Tp *>());
        mat2_2d = ito::DataObject(3, 3, ito::getDataType2<_Tp *>());
        mat3_2d = ito::DataObject(3, 3, ito::getDataType2<_Tp *>());

        mat1_3d = ito::DataObject(3, 3, 3, ito::getDataType2<_Tp *>());
        mat2_3d = ito::DataObject(3, 3, 3, ito::getDataType2<_Tp *>());
        mat3_3d = ito::DataObject(3, 3, 3, ito::getDataType2<_Tp *>());
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

TYPED_TEST_CASE(operatorComplexTest, ItomComplexDataTypes);

// complexDivTest1d
/*!
    This test checks the functionality of "div" operator for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexDivTest1d)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(TypeParam(4, 6));
    this->mat2_1d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_1d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));

    this->mat3_1d = this->mat1_1d.div(this->mat2_1d);
    for (int i = 0; i < 2; i++)
        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(TypeParam(5, 1)));
}
// complexDivTest2d
/*!
    This test checks the functionality of "div" operator for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexDivTest2d)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(TypeParam(4, 6));
    this->mat2_2d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_2d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));

    this->mat3_2d = this->mat1_2d.div(this->mat2_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_EQ(this->mat3_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(TypeParam(5, 1)));
}
// complexDivTest3d
/*!
    This test checks the functionality of "div" operator for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexDivTest3d)
{
    this->mat1_3d = cv::saturate_cast<TypeParam>(TypeParam(4, 6));
    this->mat2_3d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_3d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_3d = this->mat1_3d.div(this->mat2_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_EQ(this->mat3_3d.template at<TypeParam>(i, j, k), cv::saturate_cast<TypeParam>(TypeParam(5, 1)));
}
// complexMulTest1d
/*!
    This test checks the functionality of "mul" operator for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexMulTest1d)
{
    this->mat1_1d = cv::saturate_cast<TypeParam>(TypeParam(4, 3));
    this->mat2_1d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_1d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));

    this->mat3_1d = this->mat1_1d.mul(this->mat2_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(this->mat3_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(TypeParam(1, 7)));
}
// complexMulTest2d
/*!
    This test checks the functionality of "mul" operator for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexMulTest2d)
{
    this->mat1_2d = cv::saturate_cast<TypeParam>(TypeParam(4, 3));
    this->mat2_2d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_2d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));

    this->mat3_2d = this->mat1_2d.mul(this->mat2_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_EQ(this->mat3_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(TypeParam(1, 7)));
}
// complexMulTest3d
/*!
    This test checks the functionality of "mul" operator for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexMulTest3d)
{
    this->mat1_3d = cv::saturate_cast<TypeParam>(TypeParam(4, 3));
    this->mat2_3d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_3d = cv::saturate_cast<TypeParam>(TypeParam(1, 1));
    this->mat3_3d = this->mat1_3d.mul(this->mat2_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_EQ(this->mat3_3d.template at<TypeParam>(i, j, k), cv::saturate_cast<TypeParam>(TypeParam(1, 7)));
}
// complexConjTest1d
/*!
    This test checks the functionality of "conj" method for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexConjTest1d)
{
    this->mat1_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(TypeParam(10.09, 0.012));
    this->mat1_1d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(TypeParam(1, -2.3));
    this->mat1_1d.template at<TypeParam>(0, 2) = cv::saturate_cast<TypeParam>(TypeParam(0, 2));
    this->mat1_1d.conj();
    EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(TypeParam(10.09, -0.012)));
    EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(TypeParam(1, 2.3)));
    EXPECT_EQ(this->mat1_1d.template at<TypeParam>(0, 2), cv::saturate_cast<TypeParam>(TypeParam(0, -2)));
}
// complexConjTest2d
/*!
    This test checks the functionality of "conj" method for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexConjTest2d)
{
    this->mat1_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(TypeParam(1, 0));
    this->mat1_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(TypeParam(3, -2.77));
    this->mat1_2d.template at<TypeParam>(2, 1) = cv::saturate_cast<TypeParam>(TypeParam(4, 1.22));
    this->mat1_2d.conj();
    EXPECT_EQ(this->mat1_2d.template at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(TypeParam(1, 0)));
    EXPECT_EQ(this->mat1_2d.template at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(TypeParam(3, 2.77)));
    EXPECT_EQ(this->mat1_2d.template at<TypeParam>(2, 1), cv::saturate_cast<TypeParam>(TypeParam(4, -1.22)));
}
// complexConjTest3d
/*!
    This test checks the functionality of "conj" method for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexConjTest3d)
{
    this->mat1_3d.template at<TypeParam>(0, 1, 2) = cv::saturate_cast<TypeParam>(TypeParam(23.2, 0));
    this->mat1_3d.template at<TypeParam>(1, 0, 1) = cv::saturate_cast<TypeParam>(TypeParam(0, 3));
    this->mat1_3d.template at<TypeParam>(2, 2, 1) = cv::saturate_cast<TypeParam>(TypeParam(1234, -23.34));
    this->mat1_3d.conj();
    EXPECT_EQ(this->mat1_3d.template at<TypeParam>(0, 1, 2), cv::saturate_cast<TypeParam>(TypeParam(23.2, 0)));
    EXPECT_EQ(this->mat1_3d.template at<TypeParam>(1, 0, 1), cv::saturate_cast<TypeParam>(TypeParam(0, -3)));
    EXPECT_EQ(this->mat1_3d.template at<TypeParam>(2, 2, 1), cv::saturate_cast<TypeParam>(TypeParam(1234, 23.34)));
}
// complexAdjTest1d
/*!
    This test checks the functionality of "adj" method for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexAdjTest1d)
{
    this->mat1_1d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(TypeParam(10.09, 0.012));
    this->mat1_1d.template at<TypeParam>(0, 1) = cv::saturate_cast<TypeParam>(TypeParam(1, -2.3));
    this->mat1_1d.template at<TypeParam>(0, 2) = cv::saturate_cast<TypeParam>(TypeParam(0, 2));
    ito::DataObject adjugatedDataObj = this->mat1_1d.adj();
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(TypeParam(10.09, -0.012)));
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(1, 0), cv::saturate_cast<TypeParam>(TypeParam(1, 2.3)));
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(2, 0), cv::saturate_cast<TypeParam>(TypeParam(0, -2)));
}
// complexAdjTest2d
/*!
    This test checks the functionality of "adj" method for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexAdjTest2d)
{
    this->mat1_2d.template at<TypeParam>(0, 0) = cv::saturate_cast<TypeParam>(TypeParam(1, 0));
    this->mat1_2d.template at<TypeParam>(1, 0) = cv::saturate_cast<TypeParam>(TypeParam(3, -2.77));
    this->mat1_2d.template at<TypeParam>(2, 1) = cv::saturate_cast<TypeParam>(TypeParam(4, 1.22));
    ito::DataObject adjugatedDataObj = this->mat1_2d.adj();
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(0, 0), cv::saturate_cast<TypeParam>(TypeParam(1, 0)));
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(0, 1), cv::saturate_cast<TypeParam>(TypeParam(3, 2.77)));
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(1, 2), cv::saturate_cast<TypeParam>(TypeParam(4, -1.22)));
}
// complexAdjTest3d
/*!
    This test checks the functionality of "adj" method for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest, complexAdjTest3d)
{
    this->mat1_3d.template at<TypeParam>(0, 1, 2) = cv::saturate_cast<TypeParam>(TypeParam(23.2, 0));
    this->mat1_3d.template at<TypeParam>(1, 0, 1) = cv::saturate_cast<TypeParam>(TypeParam(0, 3));
    this->mat1_3d.template at<TypeParam>(2, 2, 1) = cv::saturate_cast<TypeParam>(TypeParam(1234, -23.34));
    ito::DataObject adjugatedDataObj = this->mat1_3d.adj();
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(0, 2, 1), cv::saturate_cast<TypeParam>(TypeParam(23.2, 0)));
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(1, 1, 0), cv::saturate_cast<TypeParam>(TypeParam(0, -3)));
    EXPECT_EQ(adjugatedDataObj.at<TypeParam>(2, 1, 2), cv::saturate_cast<TypeParam>(TypeParam(1234, 23.34)));
}

/*! \class operatorComplexTest1
    \brief test for supported arithmatic operators and methods for complex datatype data objects

    This test class checks functionality of supported methods (imag, real, abs) for data objects with complex data
   types. This test class also checks functionality of supported arithmatic operators (+,-,+=,-=) for data objects with
   complex data types.
*/
template <typename _Tp> class operatorComplexTest1 : public ::testing::Test
{
  public:
    virtual void SetUp(void)
    {
        mat1_1d = ito::DataObject(3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat2_1d = ito::DataObject(3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat3_1d = ito::DataObject(3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat4_1d = ito::DataObject(3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat1_2d = ito::DataObject(3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat2_2d = ito::DataObject(3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat3_2d = ito::DataObject(3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat4_2d = ito::DataObject(3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat1_3d = ito::DataObject(3, 3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat2_3d = ito::DataObject(3, 3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat3_3d = ito::DataObject(3, 3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
        mat4_3d = ito::DataObject(3, 3, 3, ito::getDataType((const std::complex<_Tp> *)NULL));
    }

    virtual void TearDown(void){};

    ito::DataObject mat1_1d;
    ito::DataObject mat2_1d;
    ito::DataObject mat3_1d;
    ito::DataObject mat4_1d;
    ito::DataObject mat1_2d;
    ito::DataObject mat2_2d;
    ito::DataObject mat3_2d;
    ito::DataObject mat4_2d;
    ito::DataObject mat1_3d;
    ito::DataObject mat2_3d;
    ito::DataObject mat3_3d;
    ito::DataObject mat4_3d;
};

TYPED_TEST_CASE(operatorComplexTest1, ItomFloatDoubleDataTypes);

// imageValTest1d
/*!
    This test checks the functionality of "imag" mathod for 1 dimensional matrices. "imag" mathod should return the
   result matrix containing corresponding imaginary values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, imageValTest1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(0, 24));
    this->mat2_1d = ito::imag(this->mat1_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(this->mat2_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(24));
}
// imageValTest2d
/*!
    This test checks the functionality of "imag" mathod for 2 dimensional matrices. "imag" mathod should return the
   result matrix containing corresponding imaginary values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, imageValTest2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(4, 0));
    this->mat2_2d = ito::imag(this->mat1_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_EQ(this->mat2_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(0));
}
// imageValTest3d
/*!
    This test checks the functionality of "imag" mathod for 3 dimensional matrices. "imag" mathod should return the
   result matrix containing corresponding imaginary values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, imageValTest3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(4.2, 24.12));
    this->mat2_3d = ito::imag(this->mat1_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_EQ(this->mat2_3d.template at<TypeParam>(i, j, k), cv::saturate_cast<TypeParam>(24.12));
}
// realValTest1d
/*!
    This test checks the functionality of "real" mathod for 1 dimensional matrices. "real" mathod should return the
   result matrix containing corresponding Real values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, realValTest1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(4, 24));
    this->mat2_1d = ito::real(this->mat1_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(this->mat2_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(4));
}
// realValTest2d
/*!
    This test checks the functionality of "real" mathod for 2 dimensional matrices. "real" mathod should return the
   result matrix containing corresponding Real values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, realValTest2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(4, 24));
    this->mat2_2d = ito::real(this->mat1_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_EQ(this->mat2_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(4));
}
// realValTest3d
/*!
    This test checks the functionality of "real" mathod for 3 dimensional matrices. "real" mathod should return the
   result matrix containing corresponding Real values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, realValTest3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(4, 24));
    this->mat2_3d = ito::real(this->mat1_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_EQ(this->mat2_3d.template at<TypeParam>(i, j, k), cv::saturate_cast<TypeParam>(4));
}
// absValTest1d
/*!
    This test checks the functionality of "abs" mathod for 1 dimensional matrices. "abs" mathod should return the result
   matrix containing corresponding Absolute values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, absValTest1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(3, 4));
    this->mat2_1d = ito::abs(this->mat1_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(this->mat2_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(5));
}
// absValTest2d
/*!
    This test checks the functionality of "abs" mathod for 2 dimensional matrices. "abs" mathod should return the result
   matrix containing corresponding Absolute values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, absValTest2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(3, 4));
    this->mat2_2d = ito::abs(this->mat1_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_EQ(this->mat2_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(5));
}
// absValTest3d
/*!
    This test checks the functionality of "abs" mathod for 3 dimensional matrices. "abs" mathod should return the result
   matrix containing corresponding Absolute values as its elements of the test matrix.
*/
TYPED_TEST(operatorComplexTest1, absValTest3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(3, 4));
    this->mat2_3d = ito::abs(this->mat1_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_EQ(this->mat2_3d.template at<TypeParam>(i, j, k), cv::saturate_cast<TypeParam>(5));
}
// complexAddTest1d
/*!
    This test checks the functionality of "+" operator for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexAddTest1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-4.24, 6.345));
    this->mat2_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-7.321, -2.832));
    this->mat3_1d = this->mat1_1d + this->mat2_1d;
    this->mat4_1d = ito::imag(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(6.345 + (-2.832)));
    this->mat4_1d = ito::real(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(-4.24 + (-7.321)));
}
// complexAddTest2d
/*!
    This test checks the functionality of "+" operator for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexAddTest2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(91.234, 3.836));
    this->mat2_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(15.342, -17.375));
    this->mat3_2d = this->mat1_2d + this->mat2_2d;
    this->mat4_2d = ito::imag(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j),
                            cv::saturate_cast<TypeParam>(3.836 + (-17.375)));
    this->mat4_2d = ito::real(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(91.234 + 15.342));
}
// complexAddTest3d
/*!
    This test checks the functionality of "+" operator for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexAddTest3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(6.11, -4.11));
    this->mat2_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(1.59, -1.43));
    this->mat3_3d = this->mat1_3d + this->mat2_3d;
    this->mat4_3d = ito::imag(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-4.11 + (-1.43)));
    this->mat4_3d = ito::real(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(6.11 + 1.59));
}
// complexSubTest1d
/*!
    This test checks the functionality of "-" operator for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexSubTest1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-4.24, 6.345));
    this->mat2_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-7.321, -2.832));
    this->mat3_1d = this->mat1_1d - this->mat2_1d;
    this->mat4_1d = ito::imag(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(6.345 - (-2.832)));
    this->mat4_1d = ito::real(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(-4.24 - (-7.321)));
}
// complexSubTest2d
/*!
    This test checks the functionality of "-" operator for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexSubTest2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(91.234, 3.836));
    this->mat2_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(15.342, -17.375));
    this->mat3_2d = this->mat1_2d - this->mat2_2d;
    this->mat4_2d = ito::imag(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j),
                            cv::saturate_cast<TypeParam>(3.836 - (-17.375)));
    this->mat4_2d = ito::real(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(91.234 - 15.342));
}
// complexSubTest3d
/*!
    This test checks the functionality of "-" operator for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexSubTest3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(6.11, -4.11));
    this->mat2_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(1.59, -1.43));
    this->mat3_3d = this->mat1_3d - this->mat2_3d;
    this->mat4_3d = ito::imag(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-4.11 - (-1.43)));
    this->mat4_3d = ito::real(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(6.11 - 1.59));
}
// complexAdd1Test1d
/*!
    This test checks the functionality of "+=" operator for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexAdd1Test1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(339.24, 6.345));
    this->mat3_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-7.321, -28.832));
    this->mat3_1d += this->mat1_1d;
    this->mat4_1d = ito::imag(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(6.345 + (-28.832)));
    this->mat4_1d = ito::real(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(339.24 + (-7.321)));
}
// complexAdd1Test2d
/*!
    This test checks the functionality of "+=" operator for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexAdd1Test2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(91.234, 3.836));
    this->mat3_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(15.342, -17.375));
    this->mat3_2d += this->mat1_2d;
    this->mat4_2d = ito::imag(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j),
                            cv::saturate_cast<TypeParam>(3.836 + (-17.375)));
    this->mat4_2d = ito::real(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(91.234 + 15.342));
}
// complexAdd1Test3d
/*!
    This test checks the functionality of "+=" operator for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexAdd1Test3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(23.11, -445.11));
    this->mat3_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(1.59, -23.43));
    this->mat3_3d += this->mat1_3d;
    this->mat4_3d = ito::imag(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-445.11 + (-23.43)));
    this->mat4_3d = ito::real(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(23.11 + 1.59));
}
// complexSub1Test1d
/*!
    This test checks the functionality of "-=" operator for 1 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexSub1Test1d)
{
    this->mat1_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(35.947, 263.453));
    this->mat3_1d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-7.321, -235.832));
    this->mat3_1d -= this->mat1_1d;
    this->mat4_1d = ito::imag(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(-235.832 - 263.453));
    this->mat4_1d = ito::real(this->mat3_1d);
    for (int i = 0; i < 3; i++)
        EXPECT_FLOAT_EQ(this->mat4_1d.template at<TypeParam>(0, i), cv::saturate_cast<TypeParam>(-7.321 - 35.947));
}
// complexSub1Test2d
/*!
    This test checks the functionality of "-=" operator for 2 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexSub1Test2d)
{
    this->mat1_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(23.458, -34.215));
    this->mat3_2d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-46.768, 17.375));
    this->mat3_2d -= this->mat1_2d;
    this->mat4_2d = ito::imag(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j),
                            cv::saturate_cast<TypeParam>(17.375 - (-34.215)));
    this->mat4_2d = ito::real(this->mat3_2d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_FLOAT_EQ(this->mat4_2d.template at<TypeParam>(i, j), cv::saturate_cast<TypeParam>(-46.768 - 23.458));
}
// complexSub1Test3d
/*!
    This test checks the functionality of "-=" operator for 3 dimensional matrices.
*/
TYPED_TEST(operatorComplexTest1, complexSub1Test3d)
{
    this->mat1_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(45.654, 32.456));
    this->mat3_3d = cv::saturate_cast<std::complex<TypeParam>>(std::complex<TypeParam>(-87.546, 34.213));
    this->mat3_3d -= this->mat1_3d;
    this->mat4_3d = ito::imag(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(34.213 - 32.456));
    this->mat4_3d = ito::real(this->mat3_3d);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                EXPECT_FLOAT_EQ(this->mat4_3d.template at<TypeParam>(i, j, k),
                                cv::saturate_cast<TypeParam>(-87.546 - 45.654));
}
