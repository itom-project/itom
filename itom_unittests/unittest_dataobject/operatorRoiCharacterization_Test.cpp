/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */


/*
 * Characterization tests of the DataObject operators for regions of interest (ROI).
 *
 * The operators of the DataObject are implemented by large function tables (MAKEFUNCLIST),
 * with one entry per data type, and by separate code paths for objects that are continuous
 * in memory and for objects that are not (e.g. a ROI inside a larger object). Before the
 * implementation is split into several files, these tests record the current behaviour:
 *
 *  1. the result of every operator applied to ROI objects (non-continuous memory layout)
 *     must be identical to the result for a continuous deep copy of the same region,
 *  2. in-place operators applied to a ROI must change the elements inside of the ROI only,
 *  3. operators that are not supported for a data type must behave identically for both
 *     memory layouts (both succeed or both throw).
 *
 * The tests intentionally do not encode the arithmetic of each data type (saturation,
 * rounding, ...). This is covered by operatorTest.cpp. They are executed for the real
 * types, that are supported by the DataObject. uint32 is not supported, see the test
 * RoiOperatorUnsupported.UInt32IsNotSupportedByTheDataObject.
 *
 * The reference for a region is a deep copy created by DataObject::copyTo(copy, 1), not the
 * function makeContinuous: for 2d objects makeContinuous returns a view on the same memory
 * (see RoiOperatorMakeContinuous), such that it would not be an independent reference.
 *
 * Set the environment variable ITOM_TEST_LOG_UNSUPPORTED to list all operator / type
 * combinations, that throw an exception.
 */

#include "../../DataObject/dataobj.h"
#include "../../common/sharedStructures.h"
#include "opencv2/opencv.hpp"
#include "gtest/gtest.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

typedef ::testing::
    Types<ito::uint8, ito::int8, ito::uint16, ito::int16, ito::int32, ito::float32, ito::float64>
        RoiRealTypes;

typedef ::testing::Types<ito::uint8, ito::int8, ito::uint16, ito::int16, ito::int32>
    RoiIntegerTypes;

//! describes the test objects: a full object and two same-sized regions inside of it.
/*
    Index 0 is the z axis (unused for 2d objects), 1 is y, 2 is x.
*/
struct Geometry
{
    int dims;
    unsigned char continuous; //!< value of the argument 'continuous' of the DataObject constructor
    int full[3]; //!< size of the full objects
    int startA[3]; //!< offset of the first region
    int startB[3]; //!< offset of the second region
    int size[3]; //!< size of both regions
};

const std::vector<Geometry>& geometries()
{
    static std::vector<Geometry> g;

    if (g.empty())
    {
        for (unsigned char cont = 0; cont <= 1; ++cont)
        {
            Geometry g2 = {2, cont, {1, 5, 6}, {0, 1, 1}, {0, 1, 0}, {1, 3, 4}};
            g.push_back(g2);
        }

        for (unsigned char cont = 0; cont <= 1; ++cont)
        {
            Geometry g3 = {3, cont, {4, 5, 6}, {1, 1, 1}, {2, 1, 0}, {2, 3, 4}};
            g.push_back(g3);
        }
    }

    return g;
}

std::string describe(const Geometry& g)
{
    return std::to_string(g.dims) + "d, planes allocated " +
        (g.continuous ? "continuously" : "separately");
}

//-------------------------------------------------------------------------------------
// value generators. Values are in [1, 50], such that no data type saturates when creating
// the objects. Both generators have identical values at about one third of the elements,
// such that all comparison operators have true and false results.
//-------------------------------------------------------------------------------------
inline int pattern(int z, int y, int x)
{
    return 1 + (z * 30 + y * 6 + x) % 50;
}

template <typename T> T genA(int z, int y, int x)
{
    return cv::saturate_cast<T>(pattern(z, y, x));
}

template <typename T> T genB(int z, int y, int x)
{
    const int a = pattern(z, y, x);
    return cv::saturate_cast<T>(((z + y + x) % 3 == 0) ? a : 1 + (a * 7 + 3) % 50);
}

template <> ito::DateTime genA<ito::DateTime>(int z, int y, int x)
{
    return ito::DateTime(1000000LL * pattern(z, y, x));
}

template <> ito::DateTime genB<ito::DateTime>(int z, int y, int x)
{
    const int a = pattern(z, y, x);
    return ito::DateTime(1000000LL * (((z + y + x) % 3 == 0) ? a : 1 + (a * 7 + 3) % 50));
}

template <> ito::TimeDelta genA<ito::TimeDelta>(int z, int y, int x)
{
    return ito::TimeDelta(1000LL * pattern(z, y, x));
}

template <> ito::TimeDelta genB<ito::TimeDelta>(int z, int y, int x)
{
    const int a = pattern(z, y, x);
    return ito::TimeDelta(1000LL * (((z + y + x) % 3 == 0) ? a : 1 + (a * 7 + 3) % 50));
}

//-------------------------------------------------------------------------------------
// helpers
//-------------------------------------------------------------------------------------
template <typename T> T& elem(ito::DataObject& obj, int z, int y, int x)
{
    return (obj.getDims() == 3) ? obj.at<T>(z, y, x) : obj.at<T>(y, x);
}

template <typename T>
ito::DataObject makeFull(const Geometry& g, const std::function<T(int, int, int)>& gen)
{
    const int type = ito::getDataType2<T*>();
    ito::DataObject obj;

    if (g.dims == 3)
    {
        obj = ito::DataObject(g.full[0], g.full[1], g.full[2], type, g.continuous);
    }
    else
    {
        // the constructor DataObject(sizeY, sizeX, type) has no argument 'continuous'
        const int sizes[2] = {g.full[1], g.full[2]};
        obj = ito::DataObject(2, sizes, type, g.continuous);
    }

    for (int z = 0; z < g.full[0]; ++z)
    {
        for (int y = 0; y < g.full[1]; ++y)
        {
            for (int x = 0; x < g.full[2]; ++x)
            {
                elem<T>(obj, z, y, x) = gen(z, y, x);
            }
        }
    }

    return obj;
}

//! returns a shallow copy of 'full', that only shows the region at 'start' with the size g.size
ito::DataObject makeRoi(ito::DataObject& full, const Geometry& g, const int* start)
{
    if (g.dims == 3)
    {
        ito::Range r[3] = {
            ito::Range(start[0], start[0] + g.size[0]),
            ito::Range(start[1], start[1] + g.size[1]),
            ito::Range(start[2], start[2] + g.size[2])};
        return full.at(r);
    }

    ito::Range r[2] = {
        ito::Range(start[1], start[1] + g.size[1]), ito::Range(start[2], start[2] + g.size[2])};
    return full.at(r);
}

//! returns an independent deep copy of the region, that is continuous in memory and has no ROI.
ito::DataObject deepCopyRegion(ito::DataObject& roi)
{
    ito::DataObject copy;
    roi.copyTo(copy, 1);
    return copy;
}

//! returns the first plane (cv::Mat) of the data object, considering a ROI
cv::Mat* firstPlane(ito::DataObject& obj)
{
    return reinterpret_cast<cv::Mat*>(obj.get_mdata()[obj.seekMat(0)]);
}

void logUnsupported(const char* name, const Geometry& g, int type)
{
    if (std::getenv("ITOM_TEST_LOG_UNSUPPORTED"))
    {
        std::cout << "[ UNSUPPORTED ] type " << type << ", " << name << ", " << describe(g)
                  << std::endl;
    }
}

//! compares size, type and all elements of two data objects. Reports only the first mismatches.
template <typename R>
void expectSame(ito::DataObject& actual, ito::DataObject& expected, const std::string& what)
{
    ASSERT_EQ(actual.getDims(), expected.getDims()) << what;
    ASSERT_EQ(actual.getType(), expected.getType()) << what;

    for (int d = 0; d < actual.getDims(); ++d)
    {
        ASSERT_EQ(actual.getSize(d), expected.getSize(d)) << what << ", dimension " << d;
    }

    const int dims = actual.getDims();
    const int nz = (dims == 3) ? actual.getSize(0) : 1;
    const int ny = actual.getSize(dims - 2);
    const int nx = actual.getSize(dims - 1);
    int mismatches = 0;

    for (int z = 0; z < nz; ++z)
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                const R a = elem<R>(actual, z, y, x);
                const R b = elem<R>(expected, z, y, x);

                if (!(a == b) && ++mismatches <= 3)
                {
                    ADD_FAILURE() << what << ": element (" << z << ", " << y << ", " << x
                                  << ") differs. ROI: " << ::testing::PrintToString(a)
                                  << ", continuous: " << ::testing::PrintToString(b);
                }
            }
        }
    }

    if (mismatches > 3)
    {
        ADD_FAILURE() << what << ": " << mismatches << " elements differ in total.";
    }
}

typedef std::function<ito::DataObject(ito::DataObject&, ito::DataObject&)> BinaryOp;
typedef std::function<void(ito::DataObject&, ito::DataObject&)> InPlaceOp;

bool tryBinary(const BinaryOp& op, ito::DataObject& a, ito::DataObject& b, ito::DataObject& res)
{
    try
    {
        res = op(a, b);
        return true;
    }
    catch (const cv::Exception&)
    {
        return false;
    }
}

bool tryInPlace(const InPlaceOp& op, ito::DataObject& a, ito::DataObject& b)
{
    try
    {
        op(a, b);
        return true;
    }
    catch (const cv::Exception&)
    {
        return false;
    }
}

//! The operator is applied to two ROIs (non-continuous) and to continuous copies of them.
/*
    T is the element type of the first, U of the second operand, R of the result.
*/
template <typename T, typename U, typename R>
void checkBinary(
    const Geometry& g,
    const char* name,
    const std::function<T(int, int, int)>& genFirst,
    const std::function<U(int, int, int)>& genSecond,
    const BinaryOp& op)
{
    ito::DataObject fullA = makeFull<T>(g, genFirst);
    ito::DataObject fullB = makeFull<U>(g, genSecond);
    ito::DataObject roiA = makeRoi(fullA, g, g.startA);
    ito::DataObject roiB = makeRoi(fullB, g, g.startB);
    ito::DataObject contA = deepCopyRegion(roiA);
    ito::DataObject contB = deepCopyRegion(roiB);

    ito::DataObject resRoi, resCont;
    const bool okRoi = tryBinary(op, roiA, roiB, resRoi);
    const bool okCont = tryBinary(op, contA, contB, resCont);

    EXPECT_EQ(okRoi, okCont) << name << ": ROI and continuous objects behave differently";

    if (!okRoi && !okCont)
    {
        logUnsupported(name, g, fullA.getType());
    }

    if (okRoi && okCont)
    {
        expectSame<R>(resRoi, resCont, std::string(name) + ", " + describe(g));
    }
}

template <typename T, typename R>
void checkBinarySame(const Geometry& g, const char* name, const BinaryOp& op)
{
    checkBinary<T, T, R>(g, name, genA<T>, genB<T>, op);
}

//! In-place operator applied to a ROI of a larger object.
/*
    The elements in the ROI must be equal to the result for a continuous copy of the ROI, the
    elements outside of the ROI must not be changed.
*/
template <typename T>
void checkInPlace(
    const Geometry& g,
    const char* name,
    const std::function<T(int, int, int)>& genFirst,
    const std::function<T(int, int, int)>& genSecond,
    const InPlaceOp& op)
{
    ito::DataObject full = makeFull<T>(g, genFirst);
    ito::DataObject other = makeFull<T>(g, genSecond);
    ito::DataObject pristine;
    full.copyTo(pristine); // deep copy of the original values

    ito::DataObject roi = makeRoi(full, g, g.startA);
    ito::DataObject otherRoi = makeRoi(other, g, g.startB);
    ito::DataObject pristineRoi = makeRoi(pristine, g, g.startA);
    ito::DataObject ref = deepCopyRegion(pristineRoi);
    ito::DataObject otherCont = deepCopyRegion(otherRoi);

    const bool okRoi = tryInPlace(op, roi, otherRoi);
    const bool okCont = tryInPlace(op, ref, otherCont);

    EXPECT_EQ(okRoi, okCont) << name << ": ROI and continuous objects behave differently";

    if (!okRoi && !okCont)
    {
        logUnsupported(name, g, full.getType());
    }

    if (!(okRoi && okCont))
    {
        return;
    }

    int mismatches = 0;

    for (int z = 0; z < g.full[0]; ++z)
    {
        for (int y = 0; y < g.full[1]; ++y)
        {
            for (int x = 0; x < g.full[2]; ++x)
            {
                const bool inside = z >= g.startA[0] && z < g.startA[0] + g.size[0] &&
                    y >= g.startA[1] && y < g.startA[1] + g.size[1] && x >= g.startA[2] &&
                    x < g.startA[2] + g.size[2];

                const T actual = elem<T>(full, z, y, x);
                const T expected = inside
                    ? elem<T>(ref, z - g.startA[0], y - g.startA[1], x - g.startA[2])
                    : elem<T>(pristine, z, y, x);

                if (!(actual == expected) && ++mismatches <= 3)
                {
                    ADD_FAILURE() << name << ", " << describe(g) << ": element (" << z << ", " << y
                                  << ", " << x << ") " << (inside ? "inside" : "outside")
                                  << " of the ROI is " << ::testing::PrintToString(actual)
                                  << ", expected " << ::testing::PrintToString(expected);
                }
            }
        }
    }

    if (mismatches > 3)
    {
        ADD_FAILURE() << name << ", " << describe(g) << ": " << mismatches
                      << " elements differ in total.";
    }
}

template <typename T>
void checkInPlaceSame(const Geometry& g, const char* name, const InPlaceOp& op)
{
    checkInPlace<T>(g, name, genA<T>, genB<T>, op);
}

template <typename T> class RoiOperatorTest : public ::testing::Test
{
};

template <typename T> class RoiBitwiseTest : public ::testing::Test
{
};

} // namespace

TYPED_TEST_CASE(RoiOperatorTest, RoiRealTypes);
TYPED_TEST_CASE(RoiBitwiseTest, RoiIntegerTypes);

//-------------------------------------------------------------------------------------
// the foundations of all other tests: ROI and makeContinuous deliver the expected values
//-------------------------------------------------------------------------------------
TYPED_TEST(RoiOperatorTest, RoiAndDeepCopyDeliverTheExpectedValues)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));

        ito::DataObject full = makeFull<T>(g, genA<T>);
        ito::DataObject roi = makeRoi(full, g, g.startA);
        ito::DataObject cont = deepCopyRegion(roi);

        // the ROI must really be a part of the larger object, with a non-continuous plane
        ASSERT_EQ(roi.getDims(), g.dims);
        for (int d = 0; d < g.dims; ++d)
        {
            const int idx = (g.dims == 3) ? d : d + 1;
            EXPECT_EQ(roi.getSize(d), g.size[idx]);
            EXPECT_EQ(roi.getOriginalSize(d), g.full[idx]);
        }
        EXPECT_FALSE(firstPlane(roi)->isContinuous());

        // the deep copy has the size of the ROI, no ROI and an own continuous memory block
        ASSERT_EQ(cont.getDims(), g.dims);
        for (int d = 0; d < g.dims; ++d)
        {
            EXPECT_EQ(cont.getSize(d), cont.getOriginalSize(d));
        }
        EXPECT_TRUE(firstPlane(cont)->isContinuous());
        EXPECT_NE(firstPlane(cont)->data, firstPlane(roi)->data);

        for (int z = 0; z < g.size[0]; ++z)
        {
            for (int y = 0; y < g.size[1]; ++y)
            {
                for (int x = 0; x < g.size[2]; ++x)
                {
                    const T expected = genA<T>(z + g.startA[0], y + g.startA[1], x + g.startA[2]);
                    ASSERT_EQ(elem<T>(roi, z, y, x), expected);
                    ASSERT_EQ(elem<T>(cont, z, y, x), expected);
                }
            }
        }
    }
}

TYPED_TEST(RoiOperatorTest, AdjustRoiAndRangeAddressingAreEquivalent)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));

        ito::DataObject full = makeFull<T>(g, genA<T>);
        ito::DataObject viaRange = makeRoi(full, g, g.startA);
        ito::DataObject viaAdjust = full; // shallow copy

        // offsets of the borders: {start of dim 0, end of dim 0, start of dim 1, ...}
        int limits[6];
        for (int d = 0; d < g.dims; ++d)
        {
            const int idx = (g.dims == 3) ? d : d + 1;
            limits[2 * d] = -g.startA[idx];
            limits[2 * d + 1] = -(g.full[idx] - g.startA[idx] - g.size[idx]);
        }
        viaAdjust.adjustROI(g.dims, limits);

        expectSame<T>(viaAdjust, viaRange, "adjustROI vs. at(Range)");
    }
}

//-------------------------------------------------------------------------------------
// operators with two data objects
//-------------------------------------------------------------------------------------
TYPED_TEST(RoiOperatorTest, ArithmeticWithTwoObjects)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, T>(
            g, "a + b", [](ito::DataObject& a, ito::DataObject& b) { return a + b; });
        checkBinarySame<T, T>(
            g, "a - b", [](ito::DataObject& a, ito::DataObject& b) { return a - b; });
        checkBinarySame<T, T>(
            g, "a.mul(b)", [](ito::DataObject& a, ito::DataObject& b) { return a.mul(b); });
        checkBinarySame<T, T>(
            g, "a.div(b)", [](ito::DataObject& a, ito::DataObject& b) { return a.div(b); });
    }
}

TYPED_TEST(RoiOperatorTest, ArithmeticWithScalar)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, T>(
            g, "a + 3.0", [](ito::DataObject& a, ito::DataObject&) { return a + 3.0; });
        checkBinarySame<T, T>(
            g, "a - 3.0", [](ito::DataObject& a, ito::DataObject&) { return a - 3.0; });
        checkBinarySame<T, T>(
            g, "a * 2.0", [](ito::DataObject& a, ito::DataObject&) { return a * 2.0; });
    }
}

TYPED_TEST(RoiOperatorTest, ComparisonWithTwoObjects)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, ito::uint8>(
            g, "a < b", [](ito::DataObject& a, ito::DataObject& b) { return a < b; });
        checkBinarySame<T, ito::uint8>(
            g, "a > b", [](ito::DataObject& a, ito::DataObject& b) { return a > b; });
        checkBinarySame<T, ito::uint8>(
            g, "a <= b", [](ito::DataObject& a, ito::DataObject& b) { return a <= b; });
        checkBinarySame<T, ito::uint8>(
            g, "a >= b", [](ito::DataObject& a, ito::DataObject& b) { return a >= b; });
        checkBinarySame<T, ito::uint8>(
            g, "a == b", [](ito::DataObject& a, ito::DataObject& b) { return a == b; });
        checkBinarySame<T, ito::uint8>(
            g, "a != b", [](ito::DataObject& a, ito::DataObject& b) { return a != b; });
    }
}

TYPED_TEST(RoiOperatorTest, ComparisonWithScalar)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, ito::uint8>(
            g, "a < 25.0", [](ito::DataObject& a, ito::DataObject&) { return a < 25.0; });
        checkBinarySame<T, ito::uint8>(
            g, "a > 25.0", [](ito::DataObject& a, ito::DataObject&) { return a > 25.0; });
        checkBinarySame<T, ito::uint8>(
            g, "a <= 25.0", [](ito::DataObject& a, ito::DataObject&) { return a <= 25.0; });
        checkBinarySame<T, ito::uint8>(
            g, "a >= 25.0", [](ito::DataObject& a, ito::DataObject&) { return a >= 25.0; });
        checkBinarySame<T, ito::uint8>(
            g, "a == 25.0", [](ito::DataObject& a, ito::DataObject&) { return a == 25.0; });
        checkBinarySame<T, ito::uint8>(
            g, "a != 25.0", [](ito::DataObject& a, ito::DataObject&) { return a != 25.0; });
    }
}

//! == and != are complementary. Records which value is used for 'true'.
/*
    Both regions are at the same position, such that elements of both objects are equal where
    the generators deliver equal values.
*/
TYPED_TEST(RoiOperatorTest, EqualAndNotEqualAreComplementary)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));

        ito::DataObject fullA = makeFull<T>(g, genA<T>);
        ito::DataObject fullB = makeFull<T>(g, genB<T>);
        ito::DataObject roiA = makeRoi(fullA, g, g.startA);
        ito::DataObject roiB = makeRoi(fullB, g, g.startA);

        if (std::is_same<T, ito::int8>::value)
        {
            // known limitation: the compare operators are not implemented for int8
            EXPECT_THROW((void)(roiA == roiB), cv::Exception);
            EXPECT_THROW((void)(roiA != roiB), cv::Exception);
            continue;
        }

        ito::DataObject eq = (roiA == roiB);
        ito::DataObject ne = (roiA != roiB);
        ASSERT_EQ(eq.getType(), ito::tUInt8);
        ASSERT_EQ(ne.getType(), ito::tUInt8);

        int nrEqual = 0;

        for (int z = 0; z < g.size[0]; ++z)
        {
            for (int y = 0; y < g.size[1]; ++y)
            {
                for (int x = 0; x < g.size[2]; ++x)
                {
                    const ito::uint8 e = elem<ito::uint8>(eq, z, y, x);
                    const ito::uint8 n = elem<ito::uint8>(ne, z, y, x);
                    const bool equal = genA<T>(z + g.startA[0], y + g.startA[1], x + g.startA[2]) ==
                        genB<T>(z + g.startA[0], y + g.startA[1], x + g.startA[2]);

                    EXPECT_EQ(e != 0, equal) << "(" << z << ", " << y << ", " << x << ")";
                    EXPECT_EQ(n != 0, !equal) << "(" << z << ", " << y << ", " << x << ")";
                    EXPECT_EQ(e, equal ? 255 : 0); // 'true' is 255, not 1
                    nrEqual += equal ? 1 : 0;
                }
            }
        }

        // the test data contains equal and different elements
        EXPECT_GT(nrEqual, 0);
        EXPECT_LT(nrEqual, g.size[0] * g.size[1] * g.size[2]);
    }
}

//-------------------------------------------------------------------------------------
// in-place operators
//-------------------------------------------------------------------------------------
TYPED_TEST(RoiOperatorTest, InPlaceOperatorsChangeOnlyTheRoi)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkInPlaceSame<T>(g, "a += b", [](ito::DataObject& a, ito::DataObject& b) { a += b; });
        checkInPlaceSame<T>(g, "a -= b", [](ito::DataObject& a, ito::DataObject& b) { a -= b; });
        checkInPlaceSame<T>(g, "a += 3.0", [](ito::DataObject& a, ito::DataObject&) { a += 3.0; });
        checkInPlaceSame<T>(g, "a -= 3.0", [](ito::DataObject& a, ito::DataObject&) { a -= 3.0; });
        checkInPlaceSame<T>(g, "a *= 2.0", [](ito::DataObject& a, ito::DataObject&) { a *= 2.0; });
    }
}

TYPED_TEST(RoiOperatorTest, AssignAndSetToChangeOnlyTheRoi)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkInPlaceSame<T>(g, "a = value", [](ito::DataObject& a, ito::DataObject&) {
            a = cv::saturate_cast<T>(7);
        });
        checkInPlaceSame<T>(g, "a.setTo(value)", [](ito::DataObject& a, ito::DataObject&) {
            a.setTo(cv::saturate_cast<T>(7));
        });
    }
}

//-------------------------------------------------------------------------------------
// copy and conversion
//-------------------------------------------------------------------------------------
namespace {

//! convertTo for a ROI and a continuous copy of it, for the destination type D
template <typename S, typename D> void checkConvert(const Geometry& g)
{
    ito::DataObject full = makeFull<S>(g, genA<S>);
    ito::DataObject roi = makeRoi(full, g, g.startA);
    ito::DataObject cont = deepCopyRegion(roi);
    const int destType = ito::getDataType2<D*>();

    ito::DataObject resRoi, resCont;
    bool okRoi = true;
    bool okCont = true;

    try
    {
        okRoi = !roi.convertTo(resRoi, destType).containsError();
    }
    catch (const cv::Exception&)
    {
        okRoi = false;
    }

    try
    {
        okCont = !cont.convertTo(resCont, destType).containsError();
    }
    catch (const cv::Exception&)
    {
        okCont = false;
    }

    const std::string name = "convertTo type " + std::to_string(destType);
    EXPECT_EQ(okRoi, okCont) << name << ": ROI and continuous objects behave differently";

    if (!okRoi && !okCont)
    {
        logUnsupported(name.c_str(), g, full.getType());
    }

    if (okRoi && okCont)
    {
        EXPECT_EQ(resRoi.getType(), destType);
        expectSame<D>(resRoi, resCont, name + ", " + describe(g));
    }
}

} // namespace

TYPED_TEST(RoiOperatorTest, ConvertToEveryType)
{
    typedef TypeParam S;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkConvert<S, ito::uint8>(g);
        checkConvert<S, ito::int8>(g);
        checkConvert<S, ito::uint16>(g);
        checkConvert<S, ito::int16>(g);
        checkConvert<S, ito::uint32>(g);
        checkConvert<S, ito::int32>(g);
        checkConvert<S, ito::float32>(g);
        checkConvert<S, ito::float64>(g);
        checkConvert<S, ito::complex64>(g);
        checkConvert<S, ito::complex128>(g);
    }
}

TYPED_TEST(RoiOperatorTest, CopyToOfTheRegionOnly)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, T>(
            g, "a.copyTo(b, regionOnly)", [](ito::DataObject& a, ito::DataObject&) {
                ito::DataObject copy;
                a.copyTo(copy, 1);
                return copy;
            });
    }
}

TYPED_TEST(RoiOperatorTest, TransposeOfA2dRoi)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        if (g.dims != 2)
        {
            continue;
        }

        SCOPED_TRACE(describe(g));
        checkBinarySame<T, T>(
            g, "a.trans()", [](ito::DataObject& a, ito::DataObject&) { return a.trans(); });
    }
}

//-------------------------------------------------------------------------------------
// bit operators (integer types only. Floating point types throw an exception, see operatorTest)
//-------------------------------------------------------------------------------------
TYPED_TEST(RoiBitwiseTest, BitOperatorsWithTwoObjects)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, T>(
            g, "a & b", [](ito::DataObject& a, ito::DataObject& b) { return a & b; });
        checkBinarySame<T, T>(
            g, "a | b", [](ito::DataObject& a, ito::DataObject& b) { return a | b; });
        checkBinarySame<T, T>(
            g, "a ^ b", [](ito::DataObject& a, ito::DataObject& b) { return a ^ b; });
    }
}

TYPED_TEST(RoiBitwiseTest, BitNotAndShifts)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkBinarySame<T, T>(g, "a.bitwise_not()", [](ito::DataObject& a, ito::DataObject&) {
            return a.bitwise_not();
        });
        checkBinarySame<T, T>(
            g, "a << 1", [](ito::DataObject& a, ito::DataObject&) { return a << 1; });
        checkBinarySame<T, T>(
            g, "a >> 1", [](ito::DataObject& a, ito::DataObject&) { return a >> 1; });
    }
}

TYPED_TEST(RoiBitwiseTest, InPlaceBitOperatorsChangeOnlyTheRoi)
{
    typedef TypeParam T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));
        checkInPlaceSame<T>(g, "a &= b", [](ito::DataObject& a, ito::DataObject& b) { a &= b; });
        checkInPlaceSame<T>(g, "a |= b", [](ito::DataObject& a, ito::DataObject& b) { a |= b; });
        checkInPlaceSame<T>(g, "a ^= b", [](ito::DataObject& a, ito::DataObject& b) { a ^= b; });
        checkInPlaceSame<T>(g, "a <<= 1", [](ito::DataObject& a, ito::DataObject&) { a <<= 1; });
        checkInPlaceSame<T>(g, "a >>= 1", [](ito::DataObject& a, ito::DataObject&) { a >>= 1; });
    }
}

//-------------------------------------------------------------------------------------
// date and time types
//-------------------------------------------------------------------------------------
TEST(RoiOperatorDateTime, DateTimeAndTimeDeltaOperators)
{
    typedef ito::DateTime DT;
    typedef ito::TimeDelta TD;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));

        checkBinary<DT, TD, DT>(
            g,
            "datetime + timedelta object",
            genA<DT>,
            genA<TD>,
            [](ito::DataObject& a, ito::DataObject& b) { return a + b; });
        checkBinary<DT, TD, DT>(
            g,
            "datetime - timedelta object",
            genA<DT>,
            genA<TD>,
            [](ito::DataObject& a, ito::DataObject& b) { return a - b; });
        checkBinary<DT, DT, TD>(
            g,
            "datetime - datetime object",
            genA<DT>,
            genB<DT>,
            [](ito::DataObject& a, ito::DataObject& b) { return a - b; });
        checkBinary<TD, TD, TD>(
            g,
            "timedelta + timedelta object",
            genA<TD>,
            genB<TD>,
            [](ito::DataObject& a, ito::DataObject& b) { return a + b; });
        checkBinary<TD, TD, TD>(
            g,
            "timedelta - timedelta object",
            genA<TD>,
            genB<TD>,
            [](ito::DataObject& a, ito::DataObject& b) { return a - b; });

        checkBinary<DT, DT, DT>(
            g,
            "datetime + TimeDelta(5)",
            genA<DT>,
            genB<DT>,
            [](ito::DataObject& a, ito::DataObject&) { return a + ito::TimeDelta(5000000); });
        checkBinary<DT, DT, DT>(
            g,
            "datetime - TimeDelta(5)",
            genA<DT>,
            genB<DT>,
            [](ito::DataObject& a, ito::DataObject&) { return a - ito::TimeDelta(5000000); });
        checkBinary<TD, TD, TD>(
            g,
            "timedelta + TimeDelta(5)",
            genA<TD>,
            genB<TD>,
            [](ito::DataObject& a, ito::DataObject&) { return a + ito::TimeDelta(5000); });

        const ito::DateTime ref(25000000);
        checkBinary<DT, DT, ito::uint8>(
            g,
            "datetime < DateTime",
            genA<DT>,
            genB<DT>,
            [&](ito::DataObject& a, ito::DataObject&) { return a < ref; });
        checkBinary<DT, DT, ito::uint8>(
            g,
            "datetime >= DateTime",
            genA<DT>,
            genB<DT>,
            [&](ito::DataObject& a, ito::DataObject&) { return a >= ref; });
        checkBinary<DT, DT, ito::uint8>(
            g,
            "datetime == DateTime",
            genA<DT>,
            genB<DT>,
            [&](ito::DataObject& a, ito::DataObject&) { return a == ref; });
        checkBinary<DT, DT, ito::uint8>(
            g,
            "datetime != DateTime",
            genA<DT>,
            genB<DT>,
            [&](ito::DataObject& a, ito::DataObject&) { return a != ref; });

        const ito::TimeDelta refDelta(25000);
        checkBinary<TD, TD, ito::uint8>(
            g,
            "timedelta < TimeDelta",
            genA<TD>,
            genB<TD>,
            [&](ito::DataObject& a, ito::DataObject&) { return a < refDelta; });
        checkBinary<TD, TD, ito::uint8>(
            g,
            "timedelta == TimeDelta",
            genA<TD>,
            genB<TD>,
            [&](ito::DataObject& a, ito::DataObject&) { return a == refDelta; });
    }
}

TEST(RoiOperatorDateTime, InPlaceOperatorsChangeOnlyTheRoi)
{
    typedef ito::DateTime DT;
    typedef ito::TimeDelta TD;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));

        checkInPlace<DT>(
            g,
            "datetime += TimeDelta",
            genA<DT>,
            genB<DT>,
            [](ito::DataObject& a, ito::DataObject&) { a += ito::TimeDelta(5000000); });
        checkInPlace<DT>(
            g,
            "datetime -= TimeDelta",
            genA<DT>,
            genB<DT>,
            [](ito::DataObject& a, ito::DataObject&) { a -= ito::TimeDelta(5000000); });
        checkInPlace<DT>(
            g,
            "datetime.setTo(DateTime)",
            genA<DT>,
            genB<DT>,
            [](ito::DataObject& a, ito::DataObject&) { a.setTo(ito::DateTime(77000000)); });
        checkInPlace<TD>(
            g,
            "timedelta += timedelta object",
            genA<TD>,
            genB<TD>,
            [](ito::DataObject& a, ito::DataObject& b) { a += b; });
        checkInPlace<TD>(
            g,
            "timedelta.setTo(TimeDelta)",
            genA<TD>,
            genB<TD>,
            [](ito::DataObject& a, ito::DataObject&) { a.setTo(ito::TimeDelta(7000)); });
    }
}

//-------------------------------------------------------------------------------------
// recorded quirks
//-------------------------------------------------------------------------------------
TEST(RoiOperatorUnsupported, UInt32IsNotSupportedByTheDataObject)
{
    // uint32 is listed in the function tables and in the overloads of operator= and setTo,
    // but the data objects cannot be created. Therefore it is not part of the typed tests.
    EXPECT_THROW(ito::DataObject(2, 2, ito::tUInt32), cv::Exception);
    EXPECT_THROW(ito::DataObject(2, 2, 2, ito::tUInt32), cv::Exception);
}

//! Characterization of ito::makeContinuous for regions.
/*
    The function returns a view on the same memory (not a copy), if DataObject::getContinuous()
    is true. For 2d objects that flag is always true, since it only describes the arrangement
    of the planes, and a 2d object has one plane. The documentation of makeContinuous says
    that non-continuous data is copied. If this is changed, the 2d expectations below have to
    be updated.
*/
TEST(RoiOperatorMakeContinuous, ValuesAreCorrectForAllGeometries)
{
    typedef ito::uint8 T;

    for (const Geometry& g : geometries())
    {
        SCOPED_TRACE(describe(g));

        ito::DataObject full = makeFull<T>(g, genA<T>);
        ito::DataObject roi = makeRoi(full, g, g.startA);
        ito::DataObject cont = ito::makeContinuous(roi);
        ito::DataObject reference = deepCopyRegion(roi);

        expectSame<T>(cont, reference, "makeContinuous");
    }
}

TEST(RoiOperatorMakeContinuous, A2dRegionIsNotCopied)
{
    typedef ito::uint8 T;
    const Geometry& g = geometries()[0];
    ASSERT_EQ(g.dims, 2);

    ito::DataObject full = makeFull<T>(g, genA<T>);
    ito::DataObject roi = makeRoi(full, g, g.startA);
    ito::DataObject cont = ito::makeContinuous(roi);

    EXPECT_EQ(roi.getContinuous(), 1);
    EXPECT_EQ(firstPlane(cont)->data, firstPlane(roi)->data); // same memory
    EXPECT_FALSE(firstPlane(cont)->isContinuous());
    EXPECT_NE(cont.getOriginalSize(0), cont.getSize(0)); // still a ROI
}
