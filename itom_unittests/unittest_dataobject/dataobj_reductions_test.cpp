/* Tests of the numpy-like members of ito::DataObject: sum, mean, min, max, argmin, argmax, std,
   var, median, ptp, count_nonzero, all, any, their nan* variants and clip.

   The expected values are those of the numpy functions of the same name (axis=None). */

#include <cmath>
#include <limits>

#include "../../common/sharedStructures.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

using namespace ito;

namespace
{
const float64 NaN = std::numeric_limits<float64>::quiet_NaN();
const float64 Inf = std::numeric_limits<float64>::infinity();

//! sets one element of a real data object of any type
void setValue(DataObject& obj, const float64 value, const int z, const int y, const int x)
{
    switch (obj.getType())
    {
    case tUInt8:
        obj.at<uint8>(z, y, x) = cv::saturate_cast<uint8>(value);
        break;
    case tInt8:
        obj.at<int8>(z, y, x) = cv::saturate_cast<int8>(value);
        break;
    case tUInt16:
        obj.at<uint16>(z, y, x) = cv::saturate_cast<uint16>(value);
        break;
    case tInt16:
        obj.at<int16>(z, y, x) = cv::saturate_cast<int16>(value);
        break;
    case tInt32:
        obj.at<int32>(z, y, x) = cv::saturate_cast<int32>(value);
        break;
    case tFloat32:
        obj.at<float32>(z, y, x) = static_cast<float32>(value);
        break;
    default:
        obj.at<float64>(z, y, x) = value;
        break;
    }
}

//! a 3 x 2 x 4 object with the values 0 .. 23 (numpy: np.arange(24).reshape(3, 2, 4))
DataObject arange24(tDataType type)
{
    DataObject obj(3, 2, 4, type);
    int v = 0;
    for (int z = 0; z < 3; ++z)
    {
        for (int y = 0; y < 2; ++y)
        {
            for (int x = 0; x < 4; ++x)
            {
                setValue(obj, v++, z, y, x);
            }
        }
    }
    obj.setTag("sensor", "abc");
    obj.setAxisScale(2, 0.5);
    obj.setAxisUnit(2, "mm");
    obj.setValueUnit("counts");
    obj.setValueDescription("intensity");
    return obj;
}

//! np.array([[4.0, NaN, -2.0], [8.0, 1.0, Inf]])
DataObject withNaN()
{
    DataObject obj(2, 3, tFloat64);
    obj.at<float64>(0, 0) = 4.0;
    obj.at<float64>(0, 1) = NaN;
    obj.at<float64>(0, 2) = -2.0;
    obj.at<float64>(1, 0) = 8.0;
    obj.at<float64>(1, 1) = 1.0;
    obj.at<float64>(1, 2) = Inf;
    return obj;
}

const tDataType realTypes[] = {tUInt8, tInt8, tUInt16, tInt16, tInt32, tFloat32, tFloat64};
} // namespace

//---------------------------------------------------------------------------------------------- all real types
TEST(DataObjectReductions, allRealTypesLikeArange24)
{
    for (tDataType type : realTypes)
    {
        SCOPED_TRACE(type);
        DataObject a = arange24(type);
        EXPECT_DOUBLE_EQ(a.sum(), 276.0);
        EXPECT_DOUBLE_EQ(a.nansum(), 276.0);
        EXPECT_DOUBLE_EQ(a.mean(), 11.5);
        EXPECT_DOUBLE_EQ(a.nanmean(), 11.5);
        EXPECT_DOUBLE_EQ(a.min(), 0.0);
        EXPECT_DOUBLE_EQ(a.max(), 23.0);
        EXPECT_DOUBLE_EQ(a.nanmin(), 0.0);
        EXPECT_DOUBLE_EQ(a.nanmax(), 23.0);
        EXPECT_DOUBLE_EQ(a.ptp(), 23.0);
        EXPECT_EQ(a.argmin(), 0u);
        EXPECT_EQ(a.argmax(), 23u);
        EXPECT_EQ(a.nanargmin(), 0u);
        EXPECT_EQ(a.nanargmax(), 23u);
        EXPECT_NEAR(a.var(), 47.916666666666664, 1e-12); // np.arange(24).var()
        EXPECT_NEAR(a.var(1), 50.0, 1e-12); // ddof=1
        EXPECT_NEAR(a.std(), 6.922186552431729, 1e-12);
        EXPECT_NEAR(a.nanstd(1), std::sqrt(50.0), 1e-12);
        EXPECT_DOUBLE_EQ(a.median(), 11.5);
        EXPECT_DOUBLE_EQ(a.nanmedian(), 11.5);
        EXPECT_EQ(a.count_nonzero(), 23u);
        EXPECT_FALSE(a.all()); // the first value is 0
        EXPECT_TRUE(a.any());
    }
}

//---------------------------------------------------------------------------------------------- NaN semantics
TEST(DataObjectReductions, nanPropagatesLikeNumpy)
{
    DataObject a = withNaN();
    EXPECT_TRUE(std::isnan(a.sum()));
    EXPECT_TRUE(std::isnan(a.mean()));
    EXPECT_TRUE(std::isnan(a.min()));
    EXPECT_TRUE(std::isnan(a.max()));
    EXPECT_TRUE(std::isnan(a.ptp()));
    EXPECT_TRUE(std::isnan(a.var()));
    EXPECT_TRUE(std::isnan(a.std()));
    EXPECT_TRUE(std::isnan(a.median()));
    // numpy: argmin/argmax return the index of the first NaN
    EXPECT_EQ(a.argmin(), 1u);
    EXPECT_EQ(a.argmax(), 1u);
}

TEST(DataObjectReductions, nanVariantsIgnoreNaN)
{
    DataObject a = withNaN();
    EXPECT_DOUBLE_EQ(a.nansum(), Inf); // 4 - 2 + 8 + 1 + inf
    EXPECT_DOUBLE_EQ(a.nanmin(), -2.0);
    EXPECT_DOUBLE_EQ(a.nanmax(), Inf);
    EXPECT_EQ(a.nanargmin(), 2u);
    EXPECT_EQ(a.nanargmax(), 5u);

    a.at<float64>(1, 2) = 3.0; // without the Inf: [4, nan, -2, 8, 1, 3]
    EXPECT_DOUBLE_EQ(a.nansum(), 14.0);
    EXPECT_DOUBLE_EQ(a.nanmean(), 2.8);
    EXPECT_DOUBLE_EQ(a.nanmedian(), 3.0); // median of [-2, 1, 3, 4, 8]
    EXPECT_NEAR(a.nanvar(), 10.96, 1e-12); // np.nanvar([4, nan, -2, 8, 1, 3])
    EXPECT_NEAR(a.nanstd(), std::sqrt(10.96), 1e-12);
    EXPECT_EQ(a.count_nonzero(), 6u); // NaN counts as non-zero
}

TEST(DataObjectReductions, onlyNaN)
{
    DataObject a(1, 2, tFloat32);
    a.at<float32>(0, 0) = std::numeric_limits<float32>::quiet_NaN();
    a.at<float32>(0, 1) = std::numeric_limits<float32>::quiet_NaN();
    EXPECT_DOUBLE_EQ(a.nansum(), 0.0);
    EXPECT_TRUE(std::isnan(a.nanmean()));
    EXPECT_TRUE(std::isnan(a.nanmin()));
    EXPECT_TRUE(std::isnan(a.nanmax()));
    EXPECT_TRUE(std::isnan(a.nanmedian()));
    EXPECT_TRUE(std::isnan(a.nanvar()));
    EXPECT_THROW(a.nanargmin(), cv::Exception);
    EXPECT_THROW(a.nanargmax(), cv::Exception);
}

//---------------------------------------------------------------------------------------------- ROI and index order
TEST(DataObjectReductions, respectsRoiAndFlatIndexIsRowMajor)
{
    DataObject a = arange24(tInt32);
    int offsets[6] = {-1, -1, 0, 0, -1, -1}; // plane 1, both rows, columns 1..2: [[9, 10], [13, 14]]
    a.adjustROI(3, offsets);

    EXPECT_DOUBLE_EQ(a.sum(), 46.0);
    EXPECT_DOUBLE_EQ(a.mean(), 11.5);
    EXPECT_DOUBLE_EQ(a.min(), 9.0);
    EXPECT_DOUBLE_EQ(a.max(), 14.0);
    EXPECT_EQ(a.argmin(), 0u);
    EXPECT_EQ(a.argmax(), 3u);
    EXPECT_DOUBLE_EQ(a.median(), 11.5); // (10 + 13) / 2
    EXPECT_EQ(a.count_nonzero(), 4u);
    EXPECT_TRUE(a.all());

    a.at<int32>(0, 1, 0) = -7; // [[9, 10], [-7, 14]]
    EXPECT_EQ(a.argmin(), 2u);
    EXPECT_DOUBLE_EQ(a.ptp(), 21.0);
}

TEST(DataObjectReductions, medianOddAndEvenCount)
{
    DataObject a(1, 5, tUInt8);
    const uint8 values[5] = {9, 1, 5, 3, 7};
    for (int i = 0; i < 5; ++i)
    {
        a.at<uint8>(0, i) = values[i];
    }
    EXPECT_DOUBLE_EQ(a.median(), 5.0);

    int offsets[4] = {0, 0, 0, -1}; // [9, 1, 5, 3]
    a.adjustROI(2, offsets);
    EXPECT_DOUBLE_EQ(a.median(), 4.0);
}

//---------------------------------------------------------------------------------------------- empty and errors
TEST(DataObjectReductions, emptyObjectLikeNumpy)
{
    DataObject e;
    EXPECT_DOUBLE_EQ(e.sum(), 0.0);
    EXPECT_DOUBLE_EQ(e.nansum(), 0.0);
    EXPECT_TRUE(std::isnan(e.mean()));
    EXPECT_TRUE(std::isnan(e.nanmean()));
    EXPECT_TRUE(std::isnan(e.var()));
    EXPECT_EQ(e.count_nonzero(), 0u);
    EXPECT_TRUE(e.all());
    EXPECT_FALSE(e.any());
    EXPECT_EQ(e.clip(0.0, 1.0).getDims(), 0);
    EXPECT_THROW(e.min(), cv::Exception);
    EXPECT_THROW(e.max(), cv::Exception);
    EXPECT_THROW(e.argmin(), cv::Exception);
    EXPECT_THROW(e.argmax(), cv::Exception);
    EXPECT_THROW(e.ptp(), cv::Exception);
    EXPECT_TRUE(std::isnan(e.median())); // numpy: nan with a RuntimeWarning
    EXPECT_TRUE(std::isnan(e.nanmedian()));
}

TEST(DataObjectReductions, unsupportedTypesThrow)
{
    DataObject c(2, 2, tComplex64);
    EXPECT_THROW(c.sum(), cv::Exception);
    EXPECT_THROW(c.min(), cv::Exception);
    EXPECT_THROW(c.clip(0.0, 1.0), cv::Exception);
    DataObject rgba(2, 2, tRGBA32);
    EXPECT_THROW(rgba.mean(), cv::Exception);
    EXPECT_THROW(rgba.count_nonzero(), cv::Exception);
}

TEST(DataObjectReductions, varWithTooLargeDdof)
{
    DataObject a(1, 2, tFloat64);
    a.at<float64>(0, 0) = 1.0;
    a.at<float64>(0, 1) = 3.0;
    EXPECT_DOUBLE_EQ(a.var(1), 2.0);
    EXPECT_DOUBLE_EQ(a.var(2), Inf); // numpy: inf with a RuntimeWarning
    DataObject b(1, 2, tFloat64);
    b.setTo(float64(2.0));
    EXPECT_TRUE(std::isnan(b.var(2))); // 0 / 0
}

//---------------------------------------------------------------------------------------------- clip
TEST(DataObjectReductions, clipReturnsNewObjectWithMeta)
{
    DataObject src = arange24(tInt32);
    DataObject dst = src.clip(5.0, 18.0);

    ASSERT_EQ(dst.getDims(), 3);
    EXPECT_EQ(dst.getType(), tInt32);
    EXPECT_EQ(dst.getSize(2), 4);
    bool valid = false;
    EXPECT_EQ(dst.getTag("sensor", valid).getVal_ToString(), "abc");
    EXPECT_DOUBLE_EQ(dst.getAxisScale(2), 0.5);
    EXPECT_EQ(dst.getAxisUnit(2, valid), "mm");
    EXPECT_EQ(dst.getValueUnit(), "counts");
    EXPECT_EQ(dst.getValueDescription(), "intensity");

    EXPECT_EQ(dst.at<int32>(0, 0, 0), 5);
    EXPECT_EQ(dst.at<int32>(1, 0, 2), 10);
    EXPECT_EQ(dst.at<int32>(2, 1, 3), 18);
    EXPECT_EQ(src.at<int32>(0, 0, 0), 0); // source unchanged
    EXPECT_EQ(src.at<int32>(2, 1, 3), 23);
}

TEST(DataObjectReductions, clipBoundsSaturateToTypeAndRoi)
{
    DataObject a = arange24(tUInt8);
    DataObject d = a.clip(-1000.0, 1e9); // nothing is clipped for uint8
    EXPECT_EQ(d.at<uint8>(0, 0, 0), 0);
    EXPECT_EQ(d.at<uint8>(2, 1, 3), 23);

    d = a.clip(2.6, 10.4); // fractional bounds are rounded for integer types
    EXPECT_EQ(d.at<uint8>(0, 0, 0), 3);
    EXPECT_EQ(d.at<uint8>(2, 1, 3), 10);

    int offsets[6] = {-1, -1, 0, 0, -1, -1}; // [[9, 10], [13, 14]]
    a.adjustROI(3, offsets);
    d = a.clip(10.0, 13.0);
    EXPECT_EQ(d.getSize(0), 1);
    EXPECT_EQ(d.getSize(1), 2);
    EXPECT_EQ(d.getSize(2), 2);
    EXPECT_EQ(d.at<uint8>(0, 0, 0), 10);
    EXPECT_EQ(d.at<uint8>(0, 1, 1), 13);
}

TEST(DataObjectReductions, clipKeepsNaNAndRejectsInvertedBounds)
{
    DataObject a = withNaN();
    DataObject d = a.clip(0.0, 5.0);
    EXPECT_DOUBLE_EQ(d.at<float64>(0, 0), 4.0);
    EXPECT_TRUE(std::isnan(d.at<float64>(0, 1)));
    EXPECT_DOUBLE_EQ(d.at<float64>(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(d.at<float64>(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(d.at<float64>(1, 2), 5.0); // Inf is clipped
    EXPECT_THROW(a.clip(5.0, 4.0), cv::Exception);
}
