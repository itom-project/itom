/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

/*
    numpy-like reductions and element-wise functions of ito::DataObject (sum, min, argmax, clip,
    ...). See the documentation in dataobj.h. All functions iterate plane-wise over the ROI.
*/

#include "dataobj.h"
#include "../common/numeric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace ito
{
namespace
{
const float64 NaN = std::numeric_limits<float64>::quiet_NaN();

//! throws a cv::Exception if the data type is not a real type
void requireRealType(const DataObject& obj, const char* function)
{
    switch (obj.getType())
    {
    case tUInt8:
    case tInt8:
    case tUInt16:
    case tInt16:
    case tUInt32:
    case tInt32:
    case tFloat32:
    case tFloat64:
        return;
    default:
        CV_Error_(
            cv::Error::StsUnsupportedFormat,
            ("DataObject::%s is only defined for real data types", function));
    }
}

//! throws a cv::Exception if the object has no elements (numpy: ValueError)
void requireNonEmpty(const DataObject& obj, const char* function)
{
    if (obj.getDims() == 0 || obj.getTotal() == 0)
    {
        CV_Error_(
            cv::Error::StsBadArg,
            ("DataObject::%s of an empty object is not defined", function));
    }
}

//! calls visitor(value, flatIndex) for every element of the ROI, row-major like numpy
template <typename _Tp, typename Visitor> void forEach(const DataObject& obj, Visitor& visitor)
{
    const int numMats = obj.getNumPlanes();
    size_t index = 0;

    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        const cv::Mat* mat = obj.get_mdata()[obj.seekMat(nmat, numMats)];

        for (int m = 0; m < mat->rows; ++m)
        {
            const _Tp* rowPtr = mat->ptr<_Tp>(m);

            for (int n = 0; n < mat->cols; ++n)
            {
                visitor(rowPtr[n], index++);
            }
        }
    }
}

//! dispatches forEach over the real data types
template <typename Visitor> void forEachReal(const DataObject& obj, Visitor& visitor)
{
    switch (obj.getType())
    {
    case tUInt8:
        forEach<uint8>(obj, visitor);
        break;
    case tInt8:
        forEach<int8>(obj, visitor);
        break;
    case tUInt16:
        forEach<uint16>(obj, visitor);
        break;
    case tInt16:
        forEach<int16>(obj, visitor);
        break;
    case tUInt32:
        forEach<uint32>(obj, visitor);
        break;
    case tInt32:
        forEach<int32>(obj, visitor);
        break;
    case tFloat32:
        forEach<float32>(obj, visitor);
        break;
    default:
        forEach<float64>(obj, visitor);
        break;
    }
}

//! sums all values (ignoreNaN: NaN values are skipped, otherwise NaN propagates)
struct SumVisitor
{
    explicit SumVisitor(bool ignoreNaN) : sum(0.0), count(0), ignoreNaN(ignoreNaN)
    {
    }

    template <typename _Tp> void operator()(const _Tp& value, size_t)
    {
        if (ignoreNaN && isNaN<_Tp>(value))
        {
            return;
        }
        sum += static_cast<float64>(value);
        ++count;
    }

    float64 sum;
    size_t count;
    bool ignoreNaN;
};

//! minimum and maximum with their first flat indices
struct MinMaxVisitor
{
    explicit MinMaxVisitor(bool ignoreNaN) :
        minValue(NaN), maxValue(NaN), minIndex(0), maxIndex(0), nanIndex(0), count(0),
        hasNaN(false), ignoreNaN(ignoreNaN)
    {
    }

    template <typename _Tp> void operator()(const _Tp& value, size_t index)
    {
        if (isNaN<_Tp>(value))
        {
            if (!hasNaN)
            {
                hasNaN = true;
                nanIndex = index;
            }
            return;
        }

        const float64 v = static_cast<float64>(value);

        if (count == 0)
        {
            minValue = maxValue = v;
            minIndex = maxIndex = index;
        }
        else
        {
            if (v < minValue)
            {
                minValue = v;
                minIndex = index;
            }
            if (v > maxValue)
            {
                maxValue = v;
                maxIndex = index;
            }
        }

        ++count;
    }

    //! true if the result is NaN because of a NaN value in the data
    bool nanResult() const
    {
        return hasNaN && !ignoreNaN;
    }

    float64 minValue;
    float64 maxValue;
    size_t minIndex;
    size_t maxIndex;
    size_t nanIndex;
    size_t count;
    bool hasNaN;
    bool ignoreNaN;
};

//! collects all (non-NaN) values as float64 for the median
struct CollectVisitor
{
    explicit CollectVisitor(bool ignoreNaN) : hasNaN(false), ignoreNaN(ignoreNaN)
    {
    }

    template <typename _Tp> void operator()(const _Tp& value, size_t)
    {
        if (isNaN<_Tp>(value))
        {
            hasNaN = true;
            if (ignoreNaN)
            {
                return;
            }
        }
        values.push_back(static_cast<float64>(value));
    }

    std::vector<float64> values;
    bool hasNaN;
    bool ignoreNaN;
};

//! sum of squared deviations from a given mean (ignoreNaN: NaN values are skipped)
struct SquaredDeviationVisitor
{
    SquaredDeviationVisitor(float64 mean, bool ignoreNaN) :
        sum(0.0), count(0), mean(mean), ignoreNaN(ignoreNaN)
    {
    }

    template <typename _Tp> void operator()(const _Tp& value, size_t)
    {
        if (ignoreNaN && isNaN<_Tp>(value))
        {
            return;
        }
        const float64 d = static_cast<float64>(value) - mean;
        sum += d * d;
        ++count;
    }

    float64 sum;
    size_t count;
    float64 mean;
    bool ignoreNaN;
};

//! counts non-zero values (NaN is non-zero, like in numpy)
struct CountNonZeroVisitor
{
    CountNonZeroVisitor() : count(0)
    {
    }

    template <typename _Tp> void operator()(const _Tp& value, size_t)
    {
        if (value != _Tp(0))
        {
            ++count;
        }
    }

    size_t count;
};

float64 sumImpl(const DataObject& obj, bool ignoreNaN, const char* function)
{
    requireRealType(obj, function);
    if (obj.getDims() == 0)
    {
        return 0.0;
    }
    SumVisitor visitor(ignoreNaN);
    forEachReal(obj, visitor);
    return visitor.sum;
}

float64 meanImpl(const DataObject& obj, bool ignoreNaN, const char* function)
{
    requireRealType(obj, function);
    if (obj.getDims() == 0)
    {
        return NaN;
    }
    SumVisitor visitor(ignoreNaN);
    forEachReal(obj, visitor);
    return visitor.count > 0 ? visitor.sum / static_cast<float64>(visitor.count) : NaN;
}

MinMaxVisitor minMaxImpl(const DataObject& obj, bool ignoreNaN, const char* function)
{
    requireRealType(obj, function);
    requireNonEmpty(obj, function);
    MinMaxVisitor visitor(ignoreNaN);
    forEachReal(obj, visitor);
    return visitor;
}

float64 varImpl(const DataObject& obj, int ddof, bool ignoreNaN, const char* function)
{
    requireRealType(obj, function);
    if (obj.getDims() == 0)
    {
        return NaN;
    }

    const float64 mean = meanImpl(obj, ignoreNaN, function);

    if (std::isnan(mean))
    {
        return NaN;
    }

    SquaredDeviationVisitor visitor(mean, ignoreNaN);
    forEachReal(obj, visitor);

    const float64 divisor = static_cast<float64>(visitor.count) - static_cast<float64>(ddof);

    // numpy: a non-positive divisor gives inf or nan with a warning
    if (divisor <= 0.0)
    {
        return visitor.sum > 0.0 ? std::numeric_limits<float64>::infinity() : NaN;
    }

    return visitor.sum / divisor;
}

float64 medianImpl(const DataObject& obj, bool ignoreNaN, const char* function)
{
    requireRealType(obj, function);

    if (obj.getDims() == 0)
    {
        return NaN; // numpy: nan with a RuntimeWarning
    }

    CollectVisitor visitor(ignoreNaN);
    forEachReal(obj, visitor);

    if (visitor.hasNaN && !ignoreNaN)
    {
        return NaN;
    }

    std::vector<float64>& v = visitor.values;

    if (v.empty())
    {
        return NaN; // nanmedian of only NaN values
    }

    const size_t half = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + half, v.end());
    const float64 upper = v[half];

    if (v.size() % 2 == 1)
    {
        return upper;
    }

    // the largest value of the lower half
    const float64 lower = *std::max_element(v.begin(), v.begin() + half);
    return (lower + upper) / 2.0;
}

//! clips the values of src (type _Tp) to [lower, upper] and writes them to dst (same size and type)
template <typename _Tp>
void clipImpl(const DataObject& src, DataObject& dst, float64 a_min, float64 a_max)
{
    const _Tp lower = cv::saturate_cast<_Tp>(a_min);
    const _Tp upper = cv::saturate_cast<_Tp>(a_max);
    const int numMats = src.getNumPlanes();

    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        const cv::Mat* srcMat = src.get_mdata()[src.seekMat(nmat, numMats)];
        cv::Mat* dstMat = dst.get_mdata()[nmat];

        for (int m = 0; m < srcMat->rows; ++m)
        {
            const _Tp* srcPtr = srcMat->ptr<_Tp>(m);
            _Tp* dstPtr = dstMat->ptr<_Tp>(m);

            for (int n = 0; n < srcMat->cols; ++n)
            {
                // NaN compares false in both cases and is copied unchanged
                const _Tp v = srcPtr[n];
                dstPtr[n] = (v < lower) ? lower : ((v > upper) ? upper : v);
            }
        }
    }
}
} // namespace

//----------------------------------------------------------------------------------------------------------------------------------
float64 DataObject::sum() const
{
    return sumImpl(*this, false, "sum");
}

float64 DataObject::nansum() const
{
    return sumImpl(*this, true, "nansum");
}

float64 DataObject::mean() const
{
    return meanImpl(*this, false, "mean");
}

float64 DataObject::nanmean() const
{
    return meanImpl(*this, true, "nanmean");
}

float64 DataObject::min() const
{
    const MinMaxVisitor v = minMaxImpl(*this, false, "min");
    return v.nanResult() ? NaN : v.minValue;
}

float64 DataObject::max() const
{
    const MinMaxVisitor v = minMaxImpl(*this, false, "max");
    return v.nanResult() ? NaN : v.maxValue;
}

float64 DataObject::nanmin() const
{
    return minMaxImpl(*this, true, "nanmin").minValue;
}

float64 DataObject::nanmax() const
{
    return minMaxImpl(*this, true, "nanmax").maxValue;
}

float64 DataObject::ptp() const
{
    const MinMaxVisitor v = minMaxImpl(*this, false, "ptp");
    return v.nanResult() ? NaN : v.maxValue - v.minValue;
}

size_t DataObject::argmin() const
{
    const MinMaxVisitor v = minMaxImpl(*this, false, "argmin");
    return v.nanResult() ? v.nanIndex : v.minIndex;
}

size_t DataObject::argmax() const
{
    const MinMaxVisitor v = minMaxImpl(*this, false, "argmax");
    return v.nanResult() ? v.nanIndex : v.maxIndex;
}

size_t DataObject::nanargmin() const
{
    const MinMaxVisitor v = minMaxImpl(*this, true, "nanargmin");
    if (v.count == 0)
    {
        CV_Error(cv::Error::StsBadArg, "DataObject::nanargmin: all values are NaN");
    }
    return v.minIndex;
}

size_t DataObject::nanargmax() const
{
    const MinMaxVisitor v = minMaxImpl(*this, true, "nanargmax");
    if (v.count == 0)
    {
        CV_Error(cv::Error::StsBadArg, "DataObject::nanargmax: all values are NaN");
    }
    return v.maxIndex;
}

float64 DataObject::var(int ddof) const
{
    return varImpl(*this, ddof, false, "var");
}

float64 DataObject::std(int ddof) const
{
    return std::sqrt(varImpl(*this, ddof, false, "std"));
}

float64 DataObject::nanvar(int ddof) const
{
    return varImpl(*this, ddof, true, "nanvar");
}

float64 DataObject::nanstd(int ddof) const
{
    return std::sqrt(varImpl(*this, ddof, true, "nanstd"));
}

float64 DataObject::median() const
{
    return medianImpl(*this, false, "median");
}

float64 DataObject::nanmedian() const
{
    return medianImpl(*this, true, "nanmedian");
}

size_t DataObject::count_nonzero() const
{
    requireRealType(*this, "count_nonzero");
    if (getDims() == 0)
    {
        return 0;
    }
    CountNonZeroVisitor visitor;
    forEachReal(*this, visitor);
    return visitor.count;
}

bool DataObject::all() const
{
    return count_nonzero() == static_cast<size_t>(getTotal());
}

bool DataObject::any() const
{
    return count_nonzero() > 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObject DataObject::clip(float64 a_min, float64 a_max) const
{
    requireRealType(*this, "clip");

    if (a_min > a_max)
    {
        CV_Error(cv::Error::StsBadArg, "DataObject::clip: a_min must not be greater than a_max");
    }

    if (getDims() == 0)
    {
        return DataObject();
    }

    DataObject result(getDims(), getSize(), getType());
    copyTagMapTo(result);
    copyAxisTagsTo(result);
    result.setValueUnit(getValueUnit());
    result.setValueDescription(getValueDescription());

    switch (getType())
    {
    case tUInt8:
        clipImpl<uint8>(*this, result, a_min, a_max);
        break;
    case tInt8:
        clipImpl<int8>(*this, result, a_min, a_max);
        break;
    case tUInt16:
        clipImpl<uint16>(*this, result, a_min, a_max);
        break;
    case tInt16:
        clipImpl<int16>(*this, result, a_min, a_max);
        break;
    case tUInt32:
        clipImpl<uint32>(*this, result, a_min, a_max);
        break;
    case tInt32:
        clipImpl<int32>(*this, result, a_min, a_max);
        break;
    case tFloat32:
        clipImpl<float32>(*this, result, a_min, a_max);
        break;
    default:
        clipImpl<float64>(*this, result, a_min, a_max);
        break;
    }

    return result;
}

} // namespace ito
