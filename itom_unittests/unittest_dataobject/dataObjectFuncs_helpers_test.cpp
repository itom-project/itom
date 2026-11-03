/* Tests of the helper functions sumValue, clipValues and normalizeValues in dataObjectFuncs. */

#include <cmath>
#include <limits>

#include "opencv2/opencv.hpp"

#include "../../common/sharedStructures.h"
#include "../../DataObject/dataObjectFuncs.h"
#include "../../DataObject/dataobj.h"
#include "gtest/gtest.h"

using namespace ito;

namespace
{
const float64 NaN = std::numeric_limits<float64>::quiet_NaN();

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
    case tUInt32:
        obj.at<uint32>(z, y, x) = cv::saturate_cast<uint32>(value);
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

//! a 3 x 2 x 4 object with the values 0 .. 23 and some meta information
DataObject makeCounted(tDataType type)
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

void expectMetaCopied(const DataObject& src, const DataObject& dst)
{
    bool valid = false;
    EXPECT_EQ(dst.getTag("sensor", valid).getVal_ToString(), "abc");
    EXPECT_TRUE(valid);
    EXPECT_DOUBLE_EQ(dst.getAxisScale(2), src.getAxisScale(2));
    EXPECT_EQ(dst.getAxisUnit(2, valid), "mm");
    EXPECT_EQ(dst.getValueUnit(), "counts");
    EXPECT_EQ(dst.getValueDescription(), "intensity");
}
} // namespace

//---------------------------------------------------------------------------------------------- sumValue
TEST(DataObjectFuncsHelpers, sumValueIntegerTypes)
{
    const tDataType types[] = {tUInt8, tInt8, tUInt16, tInt16, tInt32, tFloat32, tFloat64};
    for (tDataType type : types)
    {
        DataObject obj = makeCounted(type);
        float64 sum = -1.0;
        EXPECT_EQ(dObjHelper::sumValue(&obj, sum), retOk) << "type " << type;
        EXPECT_DOUBLE_EQ(sum, 276.0) << "type " << type; // 0 + 1 + ... + 23
    }
}

TEST(DataObjectFuncsHelpers, sumValueNegativeAndRoi)
{
    DataObject obj = makeCounted(tInt16);
    obj.at<int16>(0, 0, 0) = -100;

    float64 sum = 0.0;
    EXPECT_EQ(dObjHelper::sumValue(&obj, sum), retOk);
    EXPECT_DOUBLE_EQ(sum, 176.0);

    // ROI: last plane, last row => 20 + 21 + 22 + 23
    DataObject roi = obj;
    int offsets[6] = {-2, 0, -1, 0, 0, 0};
    roi.adjustROI(3, offsets);
    EXPECT_EQ(dObjHelper::sumValue(&roi, sum), retOk);
    EXPECT_DOUBLE_EQ(sum, 86.0);
}

TEST(DataObjectFuncsHelpers, sumValueNaNHandling)
{
    DataObject obj(2, 3, tFloat64);
    obj.setTo(float64(1.5));
    obj.at<float64>(1, 1) = NaN;
    obj.at<float64>(0, 2) = std::numeric_limits<float64>::infinity();

    float64 sum = 0.0;
    EXPECT_EQ(dObjHelper::sumValue(&obj, sum, true), retOk);
    EXPECT_DOUBLE_EQ(sum, 6.0); // four finite values

    EXPECT_EQ(dObjHelper::sumValue(&obj, sum, false), retOk);
    EXPECT_TRUE(std::isnan(sum));
}

TEST(DataObjectFuncsHelpers, sumValueErrorsAndEmpty)
{
    float64 sum = 42.0;
    EXPECT_TRUE(dObjHelper::sumValue(NULL, sum).containsError());
    EXPECT_DOUBLE_EQ(sum, 0.0);

    DataObject empty;
    EXPECT_EQ(dObjHelper::sumValue(&empty, sum), retOk);
    EXPECT_DOUBLE_EQ(sum, 0.0);

    DataObject cmplx(2, 2, tComplex64);
    EXPECT_TRUE(dObjHelper::sumValue(&cmplx, sum).containsError());
    DataObject rgba(2, 2, tRGBA32);
    EXPECT_TRUE(dObjHelper::sumValue(&rgba, sum).containsError());
}

//---------------------------------------------------------------------------------------------- clipValues
TEST(DataObjectFuncsHelpers, clipValuesNewObject)
{
    DataObject src = makeCounted(tInt32);
    DataObject dst;
    EXPECT_EQ(dObjHelper::clipValues(src, dst, 5.0, 18.0), retOk);

    ASSERT_EQ(dst.getDims(), 3);
    EXPECT_EQ(dst.getType(), tInt32);
    EXPECT_EQ(dst.getSize(0), 3);
    EXPECT_EQ(dst.getSize(1), 2);
    EXPECT_EQ(dst.getSize(2), 4);
    expectMetaCopied(src, dst);

    EXPECT_EQ(dst.at<int32>(0, 0, 0), 5);
    EXPECT_EQ(dst.at<int32>(0, 1, 1), 5);
    EXPECT_EQ(dst.at<int32>(1, 0, 2), 10);
    EXPECT_EQ(dst.at<int32>(2, 0, 2), 18);
    EXPECT_EQ(dst.at<int32>(2, 1, 3), 18);

    // the source is unchanged
    EXPECT_EQ(src.at<int32>(0, 0, 0), 0);
    EXPECT_EQ(src.at<int32>(2, 1, 3), 23);
}

TEST(DataObjectFuncsHelpers, clipValuesInPlace)
{
    DataObject obj = makeCounted(tUInt8);
    EXPECT_EQ(dObjHelper::clipValues(obj, obj, 3.0, 20.0), retOk);
    EXPECT_EQ(obj.at<uint8>(0, 0, 0), 3);
    EXPECT_EQ(obj.at<uint8>(1, 0, 0), 8);
    EXPECT_EQ(obj.at<uint8>(2, 1, 3), 20);
    expectMetaCopied(obj, obj);
}

TEST(DataObjectFuncsHelpers, clipValuesBoundsSaturateToType)
{
    // bounds outside of uint8: nothing is clipped
    DataObject obj = makeCounted(tUInt8);
    DataObject dst;
    EXPECT_EQ(dObjHelper::clipValues(obj, dst, -1000.0, 1e9), retOk);
    EXPECT_EQ(dst.at<uint8>(0, 0, 0), 0);
    EXPECT_EQ(dst.at<uint8>(2, 1, 3), 23);

    // fractional bounds are rounded by saturate_cast for integer types
    EXPECT_EQ(dObjHelper::clipValues(obj, dst, 2.6, 10.4), retOk);
    EXPECT_EQ(dst.at<uint8>(0, 0, 0), 3);
    EXPECT_EQ(dst.at<uint8>(2, 1, 3), 10);
}

TEST(DataObjectFuncsHelpers, clipValuesKeepsNaN)
{
    DataObject src(1, 4, tFloat32);
    src.at<float32>(0, 0) = -5.0f;
    src.at<float32>(0, 1) = std::numeric_limits<float32>::quiet_NaN();
    src.at<float32>(0, 2) = 0.25f;
    src.at<float32>(0, 3) = std::numeric_limits<float32>::infinity();

    DataObject dst;
    EXPECT_EQ(dObjHelper::clipValues(src, dst, 0.0, 1.0), retOk);
    EXPECT_FLOAT_EQ(dst.at<float32>(0, 0), 0.0f);
    EXPECT_TRUE(std::isnan(dst.at<float32>(0, 1)));
    EXPECT_FLOAT_EQ(dst.at<float32>(0, 2), 0.25f);
    EXPECT_FLOAT_EQ(dst.at<float32>(0, 3), 1.0f);
}

TEST(DataObjectFuncsHelpers, clipValuesRoiGivesRoiSizedResult)
{
    DataObject src = makeCounted(tFloat64);
    int offsets[6] = {-1, -1, 0, 0, -1, -1}; // plane 1, both rows, columns 1..2
    src.adjustROI(3, offsets);

    DataObject dst;
    EXPECT_EQ(dObjHelper::clipValues(src, dst, 10.0, 13.0), retOk);
    EXPECT_EQ(dst.getSize(0), 1);
    EXPECT_EQ(dst.getSize(1), 2);
    EXPECT_EQ(dst.getSize(2), 2);
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 0, 0), 10.0); // src value 9
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 0, 1), 10.0);
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 1, 0), 13.0);
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 1, 1), 13.0); // src value 14
}

TEST(DataObjectFuncsHelpers, clipValuesErrorsAndEmpty)
{
    DataObject src = makeCounted(tInt16);
    DataObject dst;
    EXPECT_TRUE(dObjHelper::clipValues(src, dst, 5.0, 4.0).containsError());

    DataObject cmplx(2, 2, tComplex128);
    EXPECT_TRUE(dObjHelper::clipValues(cmplx, dst, 0.0, 1.0).containsError());

    DataObject empty;
    dst = makeCounted(tInt16);
    EXPECT_EQ(dObjHelper::clipValues(empty, dst, 0.0, 1.0), retOk);
    EXPECT_EQ(dst.getDims(), 0);
}

//---------------------------------------------------------------------------------------------- normalizeValues
TEST(DataObjectFuncsHelpers, normalizeValuesDefaultRange)
{
    DataObject src = makeCounted(tUInt16);
    DataObject dst;
    EXPECT_EQ(dObjHelper::normalizeValues(src, dst), retOk);
    EXPECT_EQ(dst.getType(), tFloat64);
    EXPECT_EQ(dst.getDims(), 3);
    expectMetaCopied(src, dst);

    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 0, 0), 0.0);
    EXPECT_DOUBLE_EQ(dst.at<float64>(2, 1, 3), 1.0);
    EXPECT_DOUBLE_EQ(dst.at<float64>(1, 0, 3), 11.0 / 23.0);
}

TEST(DataObjectFuncsHelpers, normalizeValuesToUInt8Range)
{
    DataObject src(2, 2, tFloat64);
    src.at<float64>(0, 0) = -2.0;
    src.at<float64>(0, 1) = 0.0;
    src.at<float64>(1, 0) = 2.0;
    src.at<float64>(1, 1) = 6.0;

    DataObject dst;
    EXPECT_EQ(dObjHelper::normalizeValues(src, dst, 0.0, 255.0, tUInt8), retOk);
    EXPECT_EQ(dst.getType(), tUInt8);
    EXPECT_EQ(dst.at<uint8>(0, 0), 0);
    EXPECT_EQ(dst.at<uint8>(0, 1), 64); // 2/8 * 255 = 63.75
    EXPECT_EQ(dst.at<uint8>(1, 0), 128); // 4/8 * 255 = 127.5 -> rounds to even 128
    EXPECT_EQ(dst.at<uint8>(1, 1), 255);
}

TEST(DataObjectFuncsHelpers, normalizeValuesInvertedAndShiftedRange)
{
    DataObject src = makeCounted(tInt8);
    DataObject dst;
    EXPECT_EQ(dObjHelper::normalizeValues(src, dst, 1.0, -1.0, tFloat32), retOk);
    EXPECT_FLOAT_EQ(dst.at<float32>(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(dst.at<float32>(2, 1, 3), -1.0f);
}

TEST(DataObjectFuncsHelpers, normalizeValuesConstantAndNaN)
{
    DataObject src(1, 3, tFloat64);
    src.setTo(float64(7.0));
    DataObject dst;
    EXPECT_EQ(dObjHelper::normalizeValues(src, dst, 0.25, 0.75), retOk);
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 0), 0.25);
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 2), 0.25);

    src.at<float64>(0, 0) = NaN;
    src.at<float64>(0, 1) = 1.0;
    src.at<float64>(0, 2) = 3.0;
    EXPECT_EQ(dObjHelper::normalizeValues(src, dst, 0.0, 1.0), retOk);
    EXPECT_TRUE(std::isnan(dst.at<float64>(0, 0)));
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(dst.at<float64>(0, 2), 1.0);

    // NaN and Inf become 0 for integer destination types
    EXPECT_EQ(dObjHelper::normalizeValues(src, dst, 10.0, 20.0, tInt16), retOk);
    EXPECT_EQ(dst.at<int16>(0, 0), 0);
    EXPECT_EQ(dst.at<int16>(0, 1), 10);
    EXPECT_EQ(dst.at<int16>(0, 2), 20);
}

TEST(DataObjectFuncsHelpers, normalizeValuesInPlaceAndRoi)
{
    DataObject obj = makeCounted(tFloat32);
    int offsets[6] = {0, -2, 0, 0, 0, 0}; // first plane only: values 0..7
    obj.adjustROI(3, offsets);

    // in-place with a different type is an error and leaves the object unchanged
    EXPECT_TRUE(dObjHelper::normalizeValues(obj, obj, 0.0, 1.0, tFloat64).containsError());
    EXPECT_FLOAT_EQ(obj.at<float32>(0, 1, 3), 7.0f);

    EXPECT_EQ(dObjHelper::normalizeValues(obj, obj, 0.0, 1.0, tFloat32), retOk);
    EXPECT_FLOAT_EQ(obj.at<float32>(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(obj.at<float32>(0, 1, 3), 1.0f);
    EXPECT_FLOAT_EQ(obj.at<float32>(0, 0, 1), 1.0f / 7.0f);

    // the planes outside of the ROI are untouched
    int back[6] = {0, 2, 0, 0, 0, 0};
    obj.adjustROI(3, back);
    EXPECT_FLOAT_EQ(obj.at<float32>(1, 0, 0), 8.0f);
    EXPECT_FLOAT_EQ(obj.at<float32>(2, 1, 3), 23.0f);
}

TEST(DataObjectFuncsHelpers, normalizeValuesErrors)
{
    DataObject src = makeCounted(tInt32);
    DataObject dst;
    EXPECT_TRUE(dObjHelper::normalizeValues(src, dst, 0.0, 1.0, tComplex64).containsError());
    DataObject rgba(2, 2, tRGBA32);
    EXPECT_TRUE(dObjHelper::normalizeValues(rgba, dst).containsError());
}
