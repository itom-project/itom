#include "numeric.h"
#include "gtest/gtest.h"

using namespace ito;

class NumericTest : public testing::Test
{
protected:
    // You can define per-test set-up logic as usual.
    void SetUp() override
    {
        nan64 = std::numeric_limits<ito::float64>::quiet_NaN();
        inf64 = std::numeric_limits<ito::float64>::infinity();
        eps64 = std::numeric_limits<ito::float64>::epsilon();

        nan32 = std::numeric_limits<ito::float32>::quiet_NaN();
        inf32 = std::numeric_limits<ito::float32>::infinity();
        eps32 = std::numeric_limits<ito::float32>::epsilon();
    }

    // Some expensive resource shared by all tests.
    ito::float64 nan64;
    ito::float64 inf64;
    ito::float64 eps64;

    ito::float32 nan32;
    ito::float32 inf32;
    ito::float32 eps32;
};

TEST_F(NumericTest, IsNotZero)
{
    EXPECT_TRUE(isNotZero<ito::uint8>(2));
    EXPECT_TRUE(isNotZero<ito::uint16>(2));
    EXPECT_TRUE(isNotZero<ito::uint32>(2));
    EXPECT_TRUE(isNotZero<ito::int8>(2));
    EXPECT_TRUE(isNotZero<ito::int16>(2));
    EXPECT_TRUE(isNotZero<ito::int32>(2));
    EXPECT_TRUE(isNotZero<ito::int8>(-2));
    EXPECT_TRUE(isNotZero<ito::int16>(-2));
    EXPECT_TRUE(isNotZero<ito::int32>(-2));

    EXPECT_FALSE(isNotZero<ito::uint8>(0));
    EXPECT_FALSE(isNotZero<ito::uint16>(0));
    EXPECT_FALSE(isNotZero<ito::uint32>(0));
    EXPECT_FALSE(isNotZero<ito::int8>(0));
    EXPECT_FALSE(isNotZero<ito::int16>(0));
    EXPECT_FALSE(isNotZero<ito::int32>(0));

    EXPECT_TRUE(isNotZero<ito::float32>(2.0));
    EXPECT_TRUE(isNotZero<ito::float32>(this->eps32));
    EXPECT_FALSE(isNotZero<ito::float32>(0.0));

    EXPECT_TRUE(isNotZero<ito::float64>(2.0));
    EXPECT_TRUE(isNotZero<ito::float64>(this->eps64));
    EXPECT_FALSE(isNotZero<ito::float64>(0.0));

    EXPECT_TRUE(isNotZero(ito::complex64(2, 0.0)));
    EXPECT_TRUE(isNotZero(ito::complex64(0.0, 2.0)));
    EXPECT_TRUE(isNotZero(ito::complex64(-2.0, 2.0)));
    EXPECT_TRUE(isNotZero(ito::complex64(this->eps32, 0.0)));
    EXPECT_TRUE(isNotZero(ito::complex64(0.0, this->eps32)));
    EXPECT_FALSE(isNotZero(ito::complex64(0.0, 0.0)));

    EXPECT_TRUE(isNotZero(ito::complex128(2, 0.0)));
    EXPECT_TRUE(isNotZero(ito::complex128(0.0, 2.0)));
    EXPECT_TRUE(isNotZero(ito::complex128(-2.0, 2.0)));
    EXPECT_TRUE(isNotZero(ito::complex128(this->eps64, 0.0)));
    EXPECT_TRUE(isNotZero(ito::complex128(0.0, this->eps64)));
    EXPECT_FALSE(isNotZero(ito::complex128(0.0, 0.0)));
}

TEST_F(NumericTest, IsFinite)
{
    EXPECT_TRUE(isFinite<ito::uint8>(2));
    EXPECT_TRUE(isFinite<ito::uint16>(2));
    EXPECT_TRUE(isFinite<ito::uint32>(2));
    EXPECT_TRUE(isFinite<ito::int8>(2));
    EXPECT_TRUE(isFinite<ito::int16>(2));
    EXPECT_TRUE(isFinite<ito::int32>(2));

    EXPECT_TRUE(isFinite<ito::float32>(2.0));
    EXPECT_TRUE(isFinite<ito::float32>(0.0));
    EXPECT_FALSE(isFinite<ito::float32>(this->nan32));
    EXPECT_FALSE(isFinite<ito::float32>(this->inf32));
    EXPECT_FALSE(isFinite<ito::float32>(-this->inf32));

    EXPECT_TRUE(isFinite<ito::float64>(2.0));
    EXPECT_TRUE(isFinite<ito::float64>(0.0));
    EXPECT_FALSE(isFinite<ito::float64>(this->nan64));
    EXPECT_FALSE(isFinite<ito::float64>(this->inf64));
    EXPECT_FALSE(isFinite<ito::float64>(-this->inf64));

    EXPECT_TRUE(isFinite(ito::complex64(0.0, 2.0)));
    EXPECT_TRUE(isFinite(ito::complex64(-2.0, 2.0)));
    EXPECT_TRUE(isFinite(ito::complex64(0.0, 0.0)));
    EXPECT_FALSE(isFinite(ito::complex64(this->inf32, 2.0)));
    EXPECT_FALSE(isFinite(ito::complex64(this->inf32, this->nan32)));
    EXPECT_FALSE(isFinite(ito::complex64(2.0, -this->inf32)));

    EXPECT_TRUE(isFinite(ito::complex128(0.0, 2.0)));
    EXPECT_TRUE(isFinite(ito::complex128(-2.0, 2.0)));
    EXPECT_TRUE(isFinite(ito::complex128(0.0, 0.0)));
    EXPECT_FALSE(isFinite(ito::complex128(this->inf64, 2.0)));
    EXPECT_FALSE(isFinite(ito::complex128(this->inf64, this->nan64)));
    EXPECT_FALSE(isFinite(ito::complex128(2.0, -this->inf64)));
}

TEST_F(NumericTest, IsNaN)
{
    EXPECT_FALSE(isNaN<ito::uint8>(2));
    EXPECT_FALSE(isNaN<ito::uint16>(2));
    EXPECT_FALSE(isNaN<ito::uint32>(2));
    EXPECT_FALSE(isNaN<ito::int8>(2));
    EXPECT_FALSE(isNaN<ito::int16>(2));
    EXPECT_FALSE(isNaN<ito::int32>(2));

    EXPECT_FALSE(isNaN<ito::float32>(2.0));
    EXPECT_FALSE(isNaN<ito::float32>(0.0));
    EXPECT_TRUE(isNaN<ito::float32>(this->nan32));
    EXPECT_FALSE(isNaN<ito::float32>(this->inf32));
    EXPECT_FALSE(isNaN<ito::float32>(-this->inf32));

    EXPECT_FALSE(isNaN<ito::float64>(2.0));
    EXPECT_FALSE(isNaN<ito::float64>(0.0));
    EXPECT_TRUE(isNaN<ito::float64>(this->nan64));
    EXPECT_FALSE(isNaN<ito::float64>(this->inf64));
    EXPECT_FALSE(isNaN<ito::float64>(-this->inf64));

    EXPECT_FALSE(isNaN(ito::complex64(0.0, 2.0)));
    EXPECT_FALSE(isNaN(ito::complex64(-2.0, 2.0)));
    EXPECT_FALSE(isNaN(ito::complex64(0.0, 0.0)));
    EXPECT_FALSE(isNaN(ito::complex64(this->inf32, 2.0)));
    EXPECT_TRUE(isNaN(ito::complex64(this->inf32, this->nan32)));
    EXPECT_FALSE(isNaN(ito::complex64(2.0, -this->inf32)));
    EXPECT_TRUE(isNaN(ito::complex64(this->nan32, this->nan32)));

    EXPECT_FALSE(isNaN(ito::complex128(0.0, 2.0)));
    EXPECT_FALSE(isNaN(ito::complex128(-2.0, 2.0)));
    EXPECT_FALSE(isNaN(ito::complex128(0.0, 0.0)));
    EXPECT_FALSE(isNaN(ito::complex128(this->inf64, 2.0)));
    EXPECT_TRUE(isNaN(ito::complex128(this->inf64, this->nan64)));
    EXPECT_FALSE(isNaN(ito::complex128(2.0, -this->inf64)));
    EXPECT_TRUE(isNaN(ito::complex128(this->nan64, this->nan64)));
}

TEST_F(NumericTest, IsInf)
{
    EXPECT_FALSE(isInf<ito::uint8>(2));
    EXPECT_FALSE(isInf<ito::uint16>(2));
    EXPECT_FALSE(isInf<ito::uint32>(2));
    EXPECT_FALSE(isInf<ito::int8>(2));
    EXPECT_FALSE(isInf<ito::int16>(2));
    EXPECT_FALSE(isInf<ito::int32>(2));

    EXPECT_FALSE(isInf<ito::float32>(2.0));
    EXPECT_FALSE(isInf<ito::float32>(0.0));
    EXPECT_FALSE(isInf<ito::float32>(this->nan32));
    EXPECT_TRUE(isInf<ito::float32>(this->inf32));
    EXPECT_TRUE(isInf<ito::float32>(-this->inf32));

    EXPECT_FALSE(isInf<ito::float64>(2.0));
    EXPECT_FALSE(isInf<ito::float64>(0.0));
    EXPECT_FALSE(isInf<ito::float64>(this->nan64));
    EXPECT_TRUE(isInf<ito::float64>(this->inf64));
    EXPECT_TRUE(isInf<ito::float64>(-this->inf64));

    EXPECT_FALSE(isInf(ito::complex64(0.0, 2.0)));
    EXPECT_FALSE(isInf(ito::complex64(-2.0, 2.0)));
    EXPECT_FALSE(isInf(ito::complex64(0.0, 0.0)));
    EXPECT_TRUE(isInf(ito::complex64(this->inf32, 2.0)));
    EXPECT_TRUE(isInf(ito::complex64(this->inf32, this->nan32)));
    EXPECT_TRUE(isInf(ito::complex64(2.0, -this->inf32)));
    EXPECT_FALSE(isInf(ito::complex64(this->nan32, this->nan32)));
    EXPECT_TRUE(isInf(ito::complex64(this->inf32, this->inf32)));
    EXPECT_TRUE(isInf(ito::complex64(this->inf32, -this->inf32)));
    EXPECT_TRUE(isInf(ito::complex64(-this->inf32, -this->inf32)));

    EXPECT_FALSE(isInf(ito::complex128(0.0, 2.0)));
    EXPECT_FALSE(isInf(ito::complex128(-2.0, 2.0)));
    EXPECT_FALSE(isInf(ito::complex128(0.0, 0.0)));
    EXPECT_TRUE(isInf(ito::complex128(this->inf64, 2.0)));
    EXPECT_TRUE(isInf(ito::complex128(this->inf64, this->nan64)));
    EXPECT_TRUE(isInf(ito::complex128(2.0, -this->inf64)));
    EXPECT_FALSE(isInf(ito::complex128(this->nan64, this->nan64)));
    EXPECT_TRUE(isInf(ito::complex128(this->inf64, this->inf64)));
    EXPECT_TRUE(isInf(ito::complex128(this->inf64, -this->inf64)));
    EXPECT_TRUE(isInf(ito::complex128(-this->inf64, -this->inf64)));
}

TEST_F(NumericTest, IsZeroValue)
{
    EXPECT_FALSE(isZeroValue<ito::uint8>(2, 0));
    EXPECT_FALSE(isZeroValue<ito::uint16>(2, 1));
    EXPECT_FALSE(isZeroValue<ito::uint32>(2, 1));
    EXPECT_FALSE(isZeroValue<ito::int8>(2, 1));
    EXPECT_FALSE(isZeroValue<ito::int16>(2, 0));
    EXPECT_FALSE(isZeroValue<ito::int32>(2, 0));

    EXPECT_TRUE(isZeroValue<ito::uint8>(0, 0));
    EXPECT_TRUE(isZeroValue<ito::uint16>(0, 1));
    EXPECT_TRUE(isZeroValue<ito::uint32>(0, 1));
    EXPECT_TRUE(isZeroValue<ito::int8>(0, 1));
    EXPECT_TRUE(isZeroValue<ito::int16>(0, 0));
    EXPECT_TRUE(isZeroValue<ito::int32>(0, 0));

    EXPECT_FALSE(isZeroValue<ito::float32>(2.0, this->eps32));
    EXPECT_TRUE(isZeroValue<ito::float32>(0.0, this->eps32));
    EXPECT_FALSE(isZeroValue<ito::float32>(this->eps32, this->eps32));
    EXPECT_FALSE(isZeroValue<ito::float32>(this->inf32, this->eps32));
    EXPECT_FALSE(isZeroValue<ito::float32>(-this->inf32, this->eps32));

    EXPECT_FALSE(isZeroValue<ito::float64>(2.0, this->eps64));
    EXPECT_TRUE(isZeroValue<ito::float64>(0.0, this->eps64));
    EXPECT_FALSE(isZeroValue<ito::float64>(this->inf64, this->eps64));
    EXPECT_FALSE(isZeroValue<ito::float64>(-this->inf64, this->eps64));

    EXPECT_FALSE(isZeroValue(ito::complex64(0.0, 2.0), ito::complex64(this->eps32, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex64(-2.0, 2.0), ito::complex64(this->eps32, 0.0)));
    EXPECT_TRUE(isZeroValue(ito::complex64(0.0, 0.0), ito::complex64(this->eps32, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex64(this->inf32, 2.0), ito::complex64(this->eps32, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex64(2.0, -this->inf32), ito::complex64(this->eps32, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex64(this->inf32, this->inf32), ito::complex64(this->eps32, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex64(this->inf32, -this->inf32), ito::complex64(this->eps32, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex64(-this->inf32, -this->inf32), ito::complex64(this->eps32, 0.0)));

    EXPECT_FALSE(isZeroValue(ito::complex128(0.0, 2.0), ito::complex128(this->eps64, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex128(-2.0, 2.0), ito::complex128(this->eps64, 0.0)));
    EXPECT_TRUE(isZeroValue(ito::complex128(0.0, 0.0), ito::complex128(this->eps64, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex128(this->inf64, 2.0), ito::complex128(this->eps64, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex128(2.0, -this->inf64), ito::complex128(this->eps64, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex128(this->inf64, this->inf64), ito::complex128(this->eps64, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex128(this->inf64, -this->inf64), ito::complex128(this->eps64, 0.0)));
    EXPECT_FALSE(isZeroValue(ito::complex128(-this->inf64, -this->inf64), ito::complex128(this->eps64, 0.0)));
}

TEST_F(NumericTest, IsEqual)
{
    EXPECT_TRUE(areEqual<ito::uint8>(1, 1));
    EXPECT_TRUE(areEqual<ito::uint16>(1, 1));
    EXPECT_TRUE(areEqual<ito::uint32>(1, 1));
    EXPECT_TRUE(areEqual<ito::int8>(1, 1));
    EXPECT_TRUE(areEqual<ito::int16>(1, 1));
    EXPECT_TRUE(areEqual<ito::int32>(1, 1));

    EXPECT_FALSE(areEqual<ito::uint8>(10, 1));
    EXPECT_FALSE(areEqual<ito::uint16>(1, 10));
    EXPECT_FALSE(areEqual<ito::uint32>(3, 1));
    EXPECT_FALSE(areEqual<ito::int8>(-10, 100));
    EXPECT_FALSE(areEqual<ito::int16>(1, -1));
    EXPECT_FALSE(areEqual<ito::int32>(10, 1));

    EXPECT_TRUE(areEqual<ito::float32>(0.0, 0.0));
    EXPECT_TRUE(areEqual<ito::float32>(24.444, 24.444));
    EXPECT_TRUE(areEqual<ito::float64>(0.0, 0.0));
    EXPECT_TRUE(areEqual<ito::float64>(24.444, 24.444));

    EXPECT_FALSE(areEqual<ito::float32>(0.0, this->eps32));
    EXPECT_FALSE(areEqual<ito::float32>(24.444, 24.445));
    EXPECT_FALSE(areEqual<ito::float64>(this->eps64, 0.0));
    EXPECT_FALSE(areEqual<ito::float64>(24.45, 24.444));

    EXPECT_TRUE(areEqual(complex64(1.2, -2.6), complex64(1.2, -2.6)));
    EXPECT_TRUE(areEqual(complex128(1.2, -2.6), complex128(1.2, -2.6)));
    EXPECT_FALSE(areEqual(complex64(1.2, -2.7), complex64(1.2, -2.6)));
    EXPECT_FALSE(areEqual(complex128(1.2, -2.6), complex128(1.2, +2.6)));
    EXPECT_FALSE(areEqual(complex64(-1.2, -2.6), complex64(1.2, -2.6)));
    EXPECT_FALSE(areEqual(complex128(1.2001, -2.6), complex128(1.2, -2.6)));
}
