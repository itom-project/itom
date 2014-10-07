/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef __DATAOBJH
#define __DATAOBJH

#include "defines.h"

//#include <crtdbg.h>
#include <cstdlib>
#include <iostream>
#include <complex>
#include <limits>
#include <string>

#if !linux
    #pragma warning(disable:4996)
#endif

#define NOMINMAX //see: http://social.msdn.microsoft.com/Forums/sv/vclanguage/thread/d986a370-d856-4f9e-9f14-53f3b18ab63e, this is only an issue with OpenCV 2.4.3, not 2.3.x

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

#include "../common/sharedStructures.h"
#include "../common/color.h"

#include "readWriteLock.h"



namespace cv 
{
   template<> inline ito::float32 saturate_cast<ito::float32>( ito::float64 v)
   {
       //return (float32)v;
       if(cvIsInf(v)) return std::numeric_limits<ito::float32>::infinity();
       if(cvIsNaN(v)) return std::numeric_limits<ito::float32>::quiet_NaN();
       return static_cast<ito::float32>(std::max ( (ito::float64)(- std::numeric_limits<ito::float32>::max()) ,  std::min ( v , (ito::float64) std::numeric_limits<ito::float32>::max() )));
   }

   template<> inline ito::float64 saturate_cast<ito::float64>( ito::float32 v)
   {
       //return (float64)v;
       if(cvIsInf(v)) return std::numeric_limits<ito::float64>::infinity();
       if(cvIsNaN(v)) return std::numeric_limits<ito::float64>::quiet_NaN();
       return static_cast<ito::float64>(v);
   }

   template<typename _Tp> static inline _Tp saturate_cast(ito::complex128 /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
   template<typename _Tp> static inline _Tp saturate_cast(ito::complex64 /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
   
   template<typename _Tp> static inline _Tp saturate_cast(ito::Rgba32 /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
//   template<typename _Tp> static inline ito::Rgba32 saturate_cast(_Tp /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
 
   template<> inline ito::complex64 saturate_cast(ito::uint8 v){ return ito::complex64(static_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::int8 v){ return ito::complex64(static_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::uint16 v){ return ito::complex64(static_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::int16 v){ return ito::complex64(static_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::uint32 v){ return ito::complex64(static_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::int32 v){ return ito::complex64(static_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::float32 v){ return ito::complex64(v,0.0); }
   template<> inline ito::complex64 saturate_cast(ito::float64 v){ return ito::complex64(saturate_cast<ito::float32>(v),0.0); }
   template<> inline ito::complex64 saturate_cast(ito::complex64 v){ return v; }
   template<> inline ito::complex64 saturate_cast(ito::complex128 v){ return std::complex<ito::float32>(saturate_cast<ito::float32>(v.real()),saturate_cast<ito::float32>(v.imag())); }

   template<> inline ito::complex128 saturate_cast(ito::uint8 v){ return ito::complex128(static_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::int8 v){ return ito::complex128(static_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::uint16 v){ return ito::complex128(static_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::int16 v){ return ito::complex128(static_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::uint32 v){ return ito::complex128(static_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::int32 v){ return ito::complex128(static_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::float32 v){ return ito::complex128(saturate_cast<ito::float64>(v),0.0); }
   template<> inline ito::complex128 saturate_cast(ito::float64 v){ return ito::complex128(v,0.0); }
   template<> inline ito::complex128 saturate_cast(ito::complex64 v){ return ito::complex128(saturate_cast<ito::float64>(v.real()),saturate_cast<ito::float64>(v.imag())); }
   template<> inline ito::complex128 saturate_cast(ito::complex128 v){ return v; }
   
// template<> inline ito::Rgba32 saturate_cast(ito::int8 v) {return ito::Rgba32(saturate_cast<ito::uint8>(v));}
   template<> inline ito::Rgba32 saturate_cast(ito::uint8 v) {return ito::Rgba32(v);}
   template<> inline ito::Rgba32 saturate_cast(ito::uint16 v){return ito::Rgba32(saturate_cast<ito::uint8>(v));}
// template<> inline ito::Rgba32 saturate_cast(ito::int16 v){return ito::Rgba32(saturate_cast<ito::uint8>(v));}
   template<> inline ito::Rgba32 saturate_cast(ito::uint32 v)
   {
       return ito::Rgba32::fromUnsignedLong(v);
   }
   template<> inline ito::Rgba32 saturate_cast(ito::int32 v)
   {
       ito::Rgba32 temp;
       temp.rgba = static_cast<ito::uint32>(v);
       return temp;
   }
   template<> inline ito::Rgba32 saturate_cast(ito::float32 v){return ito::Rgba32(saturate_cast<ito::uint8>(v));}
   template<> inline ito::Rgba32 saturate_cast(ito::float64 v){return ito::Rgba32(saturate_cast<ito::uint8>(v));}
   template<> inline ito::Rgba32 saturate_cast(ito::Rgba32 v){return v;}

   template<> inline ito::Rgba32 saturate_cast(ito::int8 /*v*/) { cv::error(cv::Exception(CV_StsAssert, "Cast from int8 to rgba32 not defined.", "", __FILE__, __LINE__)); return ito::Rgba32(); }
   template<> inline ito::Rgba32 saturate_cast(ito::int16 /*v*/) { cv::error(cv::Exception(CV_StsAssert, "Cast from int16 to rgba32 not defined.", "", __FILE__, __LINE__)); return ito::Rgba32(); }
   template<> inline ito::Rgba32 saturate_cast(ito::complex128 /*v*/) { cv::error(cv::Exception(CV_StsAssert, "Cast from complex128 to rgba32 not defined.", "", __FILE__, __LINE__)); return ito::Rgba32(); }
   template<> inline ito::Rgba32 saturate_cast(ito::complex64 /*v*/) {  cv::error(cv::Exception(CV_StsAssert, "Cast from complex64 to rgba32 not defined.", "", __FILE__, __LINE__)); return ito::Rgba32(); }

   template<> inline ito::uint8 saturate_cast(ito::Rgba32 v){return saturate_cast<ito::uint8>(v.gray());};
   //template<> inline ito::int16 saturate_cast(ito::Rgba32 v){return saturate_cast<ito::int16>(v.gray());};
   template<> inline ito::uint16 saturate_cast(ito::Rgba32 v){return saturate_cast<ito::uint16>(v.gray());};
   template<> inline ito::uint32 saturate_cast(ito::Rgba32 v){return v.argb();};
   template<> inline ito::int32 saturate_cast(ito::Rgba32 v){return (ito::int32)(v.argb());};
   template<> inline ito::float32 saturate_cast(ito::Rgba32 v){return v.gray();};
   template<> inline ito::float64 saturate_cast(ito::Rgba32 v){return (ito::float64)v.gray();};


    template<> class DataType<ito::Rgba32>
    {
        public:
        typedef ito::Rgba32 value_type;
        typedef ito::uint8 channel_type;
        typedef Vec<channel_type, 4> work_type; 
        typedef value_type vec_type;
        enum 
        {
            generic_type = 0, 
            depth = cv::DataDepth<channel_type>::value, 
            channels = 4,
            fmt = ((channels-1)<<8) + cv::DataDepth<channel_type>::fmt,
            type = CV_MAKETYPE(depth, channels)
        };
    };

    template<> class DataType<ito::RedChannel>
    {
        public:
        typedef ito::RedChannel value_type;
        typedef ito::uint8 channel_type;
        typedef Vec<channel_type, 4> work_type; 
        typedef value_type vec_type;
        enum 
        {
            generic_type = 0, 
            depth = cv::DataDepth<channel_type>::value, 
            channels = 4,
            fmt = ((channels-1)<<8) + cv::DataDepth<channel_type>::fmt,
            type = CV_MAKETYPE(depth, channels)
        };
    };

    template<> class DataType<ito::GreenChannel>
    {
        public:
        typedef ito::GreenChannel value_type;
        typedef ito::uint8 channel_type;
        typedef Vec<channel_type, 4> work_type; 
        typedef value_type vec_type;
        enum 
        {
            generic_type = 0, 
            depth = cv::DataDepth<channel_type>::value, 
            channels = 4,
            fmt = ((channels-1)<<8) + cv::DataDepth<channel_type>::fmt,
            type = CV_MAKETYPE(depth, channels)
        };
    };

    template<> class DataType<ito::BlueChannel>
    {
        public:
        typedef ito::BlueChannel value_type;
        typedef ito::uint8 channel_type;
        typedef Vec<channel_type, 4> work_type; 
        typedef value_type vec_type;
        enum 
        {
            generic_type = 0, 
            depth = cv::DataDepth<channel_type>::value, 
            channels = 4,
            fmt = ((channels-1)<<8) + cv::DataDepth<channel_type>::fmt,
            type = CV_MAKETYPE(depth, channels)
        };
    };

    template<> class DataType<ito::AlphaChannel>
    {
        public:
        typedef ito::AlphaChannel value_type;
        typedef ito::uint8 channel_type;
        typedef Vec<channel_type, 4> work_type; 
        typedef value_type vec_type;
        enum 
        {
            generic_type = 0, 
            depth = cv::DataDepth<channel_type>::value, 
            channels = 4,
            fmt = ((channels-1)<<8) + cv::DataDepth<channel_type>::fmt,
            type = CV_MAKETYPE(depth, channels)
        };
    };


} // namespace cv

namespace ito {

#define __ITODEBUG 1

class DObjIterator;  // Forward declaration
class DataObject;
class Range;
struct Scalar;
class DataObjectTags;
class DataObjectTagType;
class DataObjectTagsPrivate;


//! method which returns the value of enumeration ito::tDataType, which corresponds to the type of the given pointer parameter.
/*!
    If the parameter type cannot be transformed into a value of ito::tDataType, an exception is thrown.

    \param any pointer, whose type should be transformed
    \return ito::tDataType
    \throws cv::Exception if the input data type is unknown
    \sa getDataType2
*/
template<typename _Tp> static inline ito::tDataType getDataType(const _Tp* /*src*/)
{
    cv::error(cv::Exception(CV_StsAssert, "Input value type unkown", "", __FILE__, __LINE__));
    return ito::tInt8;
}

template<> inline ito::tDataType getDataType(const uint8* /*src*/)      { return ito::tUInt8; }
template<> inline ito::tDataType getDataType(const int8* /*src*/)       { return ito::tInt8; }
template<> inline ito::tDataType getDataType(const uint16* /*src*/)     { return ito::tUInt16; }
template<> inline ito::tDataType getDataType(const int16* /*src*/)      { return ito::tInt16; }
template<> inline ito::tDataType getDataType(const uint32* /*src*/)     { return ito::tUInt32; }
template<> inline ito::tDataType getDataType(const int32* /*src*/)      { return ito::tInt32; }
template<> inline ito::tDataType getDataType(const float32* /*src*/)    { return ito::tFloat32; }
template<> inline ito::tDataType getDataType(const float64* /*src*/)    { return ito::tFloat64; }
template<> inline ito::tDataType getDataType(const complex64* /*src*/)  { return ito::tComplex64; }
template<> inline ito::tDataType getDataType(const complex128* /*src*/) { return ito::tComplex128; }
template<> inline ito::tDataType getDataType(const Rgba32* /*src*/) { return ito::tRGBA32; }

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class Range
    \brief each range value has a start and end point. Optionally range can be marked as Range::all(), which indicates a full range
*/
class DATAOBJ_EXPORT Range
{
    public:
        Range() : start(0), end(0) {}                           /*!< empty constructor. start = end = 0 */
        Range(int _start, int _end) { _start < _end ? (end = _end, start = _start) : (start = _end, end = _start); }  /*!< constructor with given start and end value */
        inline unsigned int size() const { return end - start; }/*!< returns number of elements between the given range boundaries */
        inline bool empty() const { return (start == end); }    /*!< returns true if range is empty, that means if the start index is equal to the end index */
        static Range all() { return Range(INT_MIN, INT_MAX); }  /*!< static method which returns a new Range object with full range */

        int start;                                               /*!< member, start-index (zero-based) */
        int end;                                                 /*!< member, end-index (zero-based) */
};

//----------------------------------------------------------------------------------------------------------------------------------
class DATAOBJ_EXPORT DataObjectTagType
{
    public:
        enum tTagType
        {
            typeInvalid     = 0x000000,
            typeDouble      = 0x000008,
            typeString      = 0x000020
        };

    private:
        double m_dVal;
        tTagType m_type; //!< parameter type, maybe int, char, double or pointer
        ByteArray m_strValue;

    public:
        //!< Constructor
        DataObjectTagType() : m_dVal(0), m_strValue(""), m_type(DataObjectTagType::typeInvalid){};
        DataObjectTagType(double dVal) : m_dVal(dVal), m_strValue(""), m_type(DataObjectTagType::typeDouble){};
        DataObjectTagType(const std::string &str) : m_dVal(std::numeric_limits<double>::quiet_NaN()), m_type(DataObjectTagType::typeString){ m_strValue = str.data(); };
        DataObjectTagType(const ByteArray &str) : m_dVal(std::numeric_limits<double>::quiet_NaN()), m_type(DataObjectTagType::typeString){ m_strValue = str; };
        DataObjectTagType(const char* cVal) : m_dVal(std::numeric_limits<double>::quiet_NaN()), m_type(DataObjectTagType::typeString){ cVal == NULL ?  m_strValue = "" : m_strValue = cVal;};
        //!< Copy Constructor
        DataObjectTagType(const DataObjectTagType& a) : m_dVal(a.m_dVal), m_type(a.m_type), m_strValue(a.m_strValue) {};

        DataObjectTagType & operator = (const DataObjectTagType &rhs)
        {
            this->m_dVal = rhs.m_dVal;
            this->m_strValue = rhs.m_strValue;
            this->m_type = rhs.m_type;

            return *this;
        }

        inline int getType(void) const {return m_type;};

        inline bool isValid(void) const { return (m_type == DataObjectTagType::typeInvalid) ? false: true;};

        /** getVal_ToDouble  read parameter value and try to convert to double
        *   @return parameter value (numeric, casted) or quiet_NaN()
        *
        *   returns the actual parameter value as double. If conversion failes it returns a signaling_NaN()
        */
        inline double getVal_ToDouble(void)
        {
            if(m_type == DataObjectTagType::typeInvalid)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            else if(m_type == DataObjectTagType::typeDouble)
            {
                return m_dVal;
            }
            else
            {
                double dVal = std::numeric_limits<double>::quiet_NaN();
                //dVal = atof(m_strValue.c_str()); //sometimes the result of that line has been arbitrary, therefore this conversion should fail.
                return dVal;
            }
        }

        /** getVal_ToString  read parameter value and try to convert to std::string
        *   @return parameter value (numeric, casted) or 'NaN' || 'Inf'
        *
        *   returns the actual parameter value as std::string. If conversion from double failes it returns 'NaN' || 'Inf'
        */
        inline ByteArray getVal_ToString(void)
        {
            if(m_type == DataObjectTagType::typeInvalid)
            {
                return "";
            }
            else if(m_type == DataObjectTagType::typeString)
            {
                return m_strValue;
            }
            else
            {
                if (cvIsNaN(m_dVal)) return "NaN";
                if (cvIsInf(m_dVal)) return "Inf";
                /*if(m_dVal == std::numeric_limits<double>::quiet_NaN()) return std::string("NaN");
                if(m_dVal == std::numeric_limits<double>::signaling_NaN()) return std::string("NaN");
                if(m_dVal == std::numeric_limits<double>::infinity()) return std::string("Inf");*/

                std::ostringstream strs;
                strs << m_dVal;
                ByteArray ba(strs.str().data());

                return ba;
            }
        }
};

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class DataObjectTags
    \brief class for handling tags for class DataObject
    \detail This class contains meta-information for the dataObject. In case of a deep copy it is also deep copied. In case of a shallow copy it is also shallow copied.
            To copy the axis metainformation into another/new object use copyAxisTagsTo to copy m_axisOffsets, m_axisScales, m_axisDescription, m_axisUnit, m_valueOffset, m_valueScale, m_valueDescription, m_valueUnit,  m_rotMatrix.
            To copy the tag-space, e.g. the protocol, use copyTagMapTo to copy m_tags.
    \sa     DataObject::copyTagMapTo, DataObject::copyAxisTagsTo
*/
//class DATAOBJ_EXPORT DataObjectTags
//{
//private:
//    DataObjectTagsPrivate *d;
//
//    friend class DataObject;
//
//
//public:
//    //!< Constructor
//    DataObjectTags(unsigned int totalAxisNum);
//
//    //!< Destructor
//    ~DataObjectTags();
//
//    //!< Copy constructor
//    DataObjectTags(const DataObjectTags& copyConstr);
//};

//----------------------------------------------------------------------------------------------------------------------------------

class DATAOBJ_EXPORT DObjConstIterator
{
public:
     
    //! default constructor
    DObjConstIterator();
    //! constructor that sets the iterator to the beginning of the matrix 
    DObjConstIterator(const DataObject* _dObj, int pos = 0);
    //! copy constructor
    DObjConstIterator(const DObjConstIterator& it);
    
    //! copy operator
    DObjConstIterator& operator = (const DObjConstIterator& it);
    //! returns the current matrix element
    uchar* operator *() const;
    //! returns the i-th matrix element, relative to the current
    uchar* operator [](int i) const;
    
    //! shifts the iterator forward by the specified number of elements
    DObjConstIterator& operator += (int ofs);
    //! shifts the iterator backward by the specified number of elements
    DObjConstIterator& operator -= (int ofs);
    //! decrements the iterator
    DObjConstIterator& operator --();
    //! decrements the iterator
    DObjConstIterator operator --(int);
    //! increments the iterator
    DObjConstIterator& operator ++();
    //! increments the iterator
    DObjConstIterator operator ++(int);

    bool operator == (const DObjConstIterator& dObjIt);
    bool operator != (const DObjConstIterator& dObjIt);
    bool operator < (const DObjConstIterator& dObjIt);
    bool operator > (const DObjConstIterator& dObjIt);
    bool operator <= (const DObjConstIterator& dObjIt);
    bool operator >= (const DObjConstIterator& dObjIt);

protected:
    void seekAbs(int ofs);
    void seekRel(int ofs);
    
    const DataObject* dObj;
    bool   planeContinuous;
    int elemSize;
    uchar* ptr;
    uchar* sliceStart;
    uchar* sliceEnd;
    int plane;
};

//----------------------------------------------------------------------------------------------------------------------------------
class DATAOBJ_EXPORT DObjIterator : public DObjConstIterator
{
public:
     
    //! default constructor
    DObjIterator();
    //! constructor that sets the iterator to the beginning of the matrix 
    DObjIterator(DataObject* _dObj, int pos = 0);
    //! copy constructor
    DObjIterator(const DObjIterator& it);
    
    //! copy operator
    DObjIterator& operator = (const DObjIterator& it);
    //! returns the current matrix element
    uchar* operator *();
    //! returns the i-th matrix element, relative to the current
    uchar* operator [](int i);
    
    //! shifts the iterator forward by the specified number of elements
    DObjIterator& operator += (int ofs);
    //! shifts the iterator backward by the specified number of elements
    DObjIterator& operator -= (int ofs);
    //! decrements the iterator
    DObjIterator& operator --();
    //! decrements the iterator
    DObjIterator operator --(int);
    //! increments the iterator
    DObjIterator& operator ++();
    //! increments the iterator
    DObjIterator operator ++(int);
};

//----------------------------------------------------------------------------------------------------------------------------------
class DATAOBJ_EXPORT DataObject
{
    private:

        //! helper method for creation of header information
        /*!
            This method allocates memory for the member variables m_roi, m_osize and m_size. Therefore one memory-block is continuously allocated with length 3*(dims+1):

            [dimensions, roi1, ..., roiN, dimensions, osize1,..., osizeN, dimensions, size1,...,sizeN]
            m_roi.m_p points to roi1, m_osize.m_p points to osize1 and m_size.m_p points to size1

            \param dimensions indicates the number of dimensions
            \param *sizes is an array with length of 'dimensions'. Each element gives the size of the corresponding dimension
            \param *steps This parameter makes the data object compatible to numpy and opencv and is only used if a continuousDataPtr has been given to the data object. Else set steps = NULL.
                        Each element if steps indicates by how many bytes one has to go in order to get from one element in this dimension to the next one. Hence, the last element is equal to elemSize
            \param elemSize number of bytes each element requires
            \throws cv::Exception if dimensions is <= 1
            \sa CreateFunc
        */
        //length(sizes) == length(steps)
        void createHeader(const unsigned char dimensions, const int *sizes, const int *steps, const int elemSize)
        {
            if(dimensions == 1)
            {
                cv::error(cv::Exception(CV_StsAssert, "number of dimensions must be at least 2", "", __FILE__, __LINE__));
            }

            m_dims = dimensions;

            if(dimensions > 0)
            {
                if (m_roi.m_p)
                {
                    if (*(m_roi.m_p-1) != dimensions)
                    {
                        delete[] (m_roi.m_p - 1);

                        m_roi.m_p = new int [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                        m_osize.m_p = static_cast<int *>(m_roi.m_p + dimensions) + 1;
                        m_osize.m_p[-1] = dimensions;
                        m_size.m_p = static_cast<int *>(m_osize.m_p + dimensions) + 1;
                        m_size.m_p[-1] = dimensions;
                        m_roi.m_p = m_roi.m_p + 1;
                        m_roi.m_p[-1] = dimensions;
                    }
                }
                else
                {
                    m_roi.m_p = new int [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                    m_osize.m_p = static_cast<int *>(m_roi.m_p + dimensions) + 1;
                    m_osize.m_p[-1] = dimensions;
                    m_size.m_p = static_cast<int *>(m_osize.m_p + dimensions) + 1;
                    m_size.m_p[-1] = dimensions;
                    m_roi.m_p = m_roi.m_p + 1;
                    m_roi.m_p[-1] = dimensions;
                }

                for (uchar n = 0 ; n < dimensions ; n++)
                {
                    m_size.m_p[n] = sizes[n];
                    m_roi.m_p[n] = 0;

                    if(steps == NULL || n == 0)
                    {
                        m_osize.m_p[n] = sizes[n];
                    }
                    else if(n == dimensions - 1) //last element
                    {
                        if (elemSize == 0)
                            cv::Exception(CV_StsAssert, "elemSize is zero", "", __FILE__, __LINE__);
                        m_osize.m_p[n] = steps[n-1] / elemSize;
                    }
                    else /*if(n < dimensions -1)*/
                    {
                        if (steps[n] == 0)
                            cv::Exception(CV_StsAssert, "step size is zero", "", __FILE__, __LINE__);
                        m_osize.m_p[n] = steps[n - 1] / steps[n];
                    }
                }
            }
        }

        //! helper method for creation of header information considering the region of interest
        /*!
            This method allocates memory for the member variables m_roi, m_osize and m_size. Therefore one memory-block is continuously allocated with length 3*(dims+1):

            [dimensions, roi1, ..., roiN, dimensions, osize1,..., osizeN, dimensions, size1,...,sizeN]
            m_roi.m_p points to roi1, m_osize.m_p points to osize1 and m_size.m_p points to size1

            \param dimensions indicates the number of dimensions
            \param *sizes is an array with length of 'dimensions'. Each element gives the size of the corresponding dimension
            \param *osizes gives a vector with the original size of each dimension, which corresponds to the physical data in memory, if NULL, a full size ROI is assumed, hence osize is equal to size (default : NULL)
            \param *roi gives a vector with the offset from the starting point of the allocated data block to the first element in the region of interest, must be NULL if osizes is NULL too (default : NULL)
            \throws cv::Exception if dimensions is <= 1
            \sa CreateFunc
        */
        void createHeaderWithROI(const unsigned char dimensions, const int *sizes, const int *osizes = NULL, const int *roi = NULL)
        {
            if(dimensions == 1)
            {
                cv::error(cv::Exception(CV_StsAssert, "number of dimensions must be at least 2", "", __FILE__, __LINE__));
            }

            m_dims = dimensions;

            if(dimensions > 0)
            {

                if (m_roi.m_p)
                {
                    if (*(m_roi.m_p-1) != dimensions)
                    {
                        delete[] (m_roi.m_p-1);

                        m_roi.m_p = new int [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                        m_osize.m_p = static_cast<int *>(m_roi.m_p + dimensions) + 1;
                        m_osize.m_p[-1] = dimensions;
                        m_size.m_p = static_cast<int *>(m_osize.m_p + dimensions) + 1;
                        m_size.m_p[-1] = dimensions;
                        m_roi.m_p = m_roi.m_p + 1; //move m_p pointer by one
                        m_roi.m_p[-1] = dimensions;
                    }
                }
                else
                {
                    m_roi.m_p = new int [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                    m_osize.m_p = static_cast<int *>(m_roi.m_p + dimensions) + 1;
                    m_osize.m_p[-1] = dimensions;
                    m_size.m_p = static_cast<int *>(m_osize.m_p + dimensions) + 1;
                    m_size.m_p[-1] = dimensions;
                    m_roi.m_p = m_roi.m_p + 1; //move m_p pointer by one
                    m_roi.m_p[-1] = dimensions;
                }

                for (uchar n = 0; n < dimensions; n++)
                {
                    m_size.m_p[n] = sizes[n];

                    if (osizes)
                    {
                        m_osize.m_p[n] = osizes[n];
                    }
                    else
                    {
                        m_osize.m_p[n] = sizes[n];
                    }

                    if (roi)
                    {
                        m_roi.m_p[n] = roi[n];
                    }
                    else
                    {
                        m_roi.m_p[n] = 0;
                    }
                }
            }
        }

        void create(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous, const uchar* continuousDataPtr = NULL, const int* steps = NULL);  /*!< allocates new data */
        void create(const unsigned char dimensions, const int *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes);

        void freeData(void);     /*!< decrements reference counter and deletes data, if no other instance is using them (ref counter < 0) */
        void secureFreeData(void); /*!< decrements reference counter and deletes data, if no other instance is using them (ref counter < 0). This method makes a lot of security checks instead of freeFunc. */


        //----------------------------------------------------------------------------------------------------------------------------------
        ito::RetVal matNumToIdx(const int matNum, int *matIdx) const;

        //! brief calculates the index of the matrix-plane in the m_data-vector for a given vector of indices, which address one element in the n-dimensional matrix
        /*!
            The matrix indices are zero-based and consider the ROI of this data object.

            \param *matIdx is a vector containing indices which address one element in the n-dimensional matrix
            \param *matNum is a pointer, where the resulting matrix-plane-index is written.
            \return retOk
            \throws cv::Exception if the given indices are out of bounds
        */
        inline ito::RetVal matIdxToNum(const unsigned int *matIdx, int *matNum) const
        {
           *matNum = 0;
           if (m_dims <= 2)
           {
                 return 0;
           }

           int planeSize = 1;

           for (int n = m_dims - 3; n >= 0; n--)
           {
        #if __ITODEBUG
                 if (((int)matIdx[n] + m_roi[n]) >= m_osize[n])
                 {
                    cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
                 }
        #endif
                 (*matNum) += ((int)matIdx[n] + m_roi[n]) * planeSize; //CAST_TODO
                 planeSize *= m_osize[n];
           }

           return 0;
        }

        struct DATAOBJ_EXPORT MSize
        {
            inline MSize() : m_p(NULL) {}
            inline int operator [] (const int dim) const
            {
                return m_p[dim]; //return the size value. this operator corresponds to the real data representation in memory
            };
            inline operator const int * () const { return m_p; }
            bool operator == (const MSize& sz) const
            {
                if(m_p == NULL || sz.m_p == NULL)
                {
                    return sz.m_p == m_p;
                }

                int d = m_p[-1], dsz = sz.m_p[-1];
                if( d != dsz )
                    return false;
                if( d == 2 )
                {                    
                    return m_p[0] == sz.m_p[0] && m_p[1] == sz.m_p[1];            
                }

                for( int i = 0; i < d - 2; i++ )
                {
                    if( m_p[i] != sz.m_p[i] )
                    {
                        return false;
                    }
                }                
                return (m_p[d - 2] == sz.m_p[d - 2]) && (m_p[d - 1] == sz.m_p[d - 1]);               
            }

            inline bool operator != (const MSize& sz) const { return !(*this == sz); }

            int *m_p;
        };

        struct DATAOBJ_EXPORT MROI
        {
            inline MROI() : m_p(NULL) {};
            inline int operator [] (const int dim) const
            {
                return m_p[dim]; //return the size value. this operator corresponds to the real data representation in memory
            }

            bool operator == (const MROI & rroi) const
            {
                if(m_p == NULL || rroi.m_p == NULL)
                {
                    return rroi.m_p == m_p;
                }

                if (m_p[-1] != rroi.m_p[-1])
                {
                    return false;
                }

                int d = m_p[-1];
                for (int n = 0; n < d - 2; n++)
                {
                    if (m_p[n] != rroi.m_p[n])
                    {
                        return false;
                    }
                }

                return m_p[d - 2] == rroi.m_p[d - 2] && m_p[d - 1] == rroi.m_p[d - 1];
                
            }

            int *m_p;
        };

        DataObject(const DataObject& dObj, bool transposed);    /*!< copy constructor for transposed creation */

        char    m_continuous;                        /*!< continuous flag */
        char    m_owndata;                           /*!< owndata flag (false if the data object is constructed with a given continuousDataPointer, else true) */
        int     m_type;                              /*!< element data type */
        int     *m_pRefCount;                        /*!< pointer to the reference counter, the integer variable will be allocated once the data block is allocated. If only one data object has access to the data, the reference counter is 0. */
        int     m_dims;                              /*!< number of dimensions */
        MSize   m_osize;                             /*!< vector containing the original size of each dimension. The allocated data block corresponds to these sizes */
        MROI    m_roi;                               /*!< vector containing the offset to the starting point of the ROI for each dimension, is used for detecting and adjusting the ROI */
        MSize   m_size;                              /*!< vector containing the "virtual" size of each dimension considering the ROI */
        uchar  **m_data;                             /*!< vector with references to each matrix-plane. array of char pointers */
        ReadWriteLock      *m_objSharedDataLock;     /*!< readWriteLock for data block, this lock is shared within every instance which is using the same data. */
        DataObjectTagsPrivate *m_pDataObjectTags;    /*!< class containing the object metadata */
        ReadWriteLock       m_objHeaderLock;         /*!< readWriteLock for this instance of dataObject. */
        static const int m_sizeofs;

        int mdata_realloc(const int size);
        
        int mdata_size(void) const;
        
        int mdata_free();

        RetVal copyFromData2DInternal(const uchar* src, const int sizeOfElem, const int sizeX, const int sizeY);  
        RetVal copyFromData2DInternal(const uchar* src, const int sizeOfElem, const int sizeX, const int x0, const int y0, const int width, const int height);
        

        //low-level, templated methods
        //most low-level methods are marked "friend" such that they can access private members of their data object parameters
        template<typename _Tp> friend RetVal CreateFunc(DataObject *dObj, const unsigned char dimensions, const int *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const int* steps);
        template<typename _Tp> friend RetVal CreateFuncWithCVPlanes(DataObject *dObj, const unsigned char dimensions, const int *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes);
        template<typename _Tp> friend RetVal FreeFunc(DataObject *dObj);
        template<typename _Tp> friend RetVal SecureFreeFunc(DataObject *dObj);
        template<typename _Tp> friend RetVal CopyToFunc(const DataObject &lhs, DataObject &rhs, unsigned char regionOnly);
        template<typename _Tp> friend RetVal ConvertToFunc(const DataObject &lhs, DataObject &rhs, const int type, const double alpha, const double beta);
        template<typename _Tp> friend RetVal AdjustROIFunc(DataObject *dObj, const int *lims);
        template<typename _Tp> friend RetVal MinMaxLocFunc(const DataObject &dObj, double *minVal, double *maxVal, int *minPos, int *maxPos);
        template<typename _Tp> friend RetVal AssignScalarFunc(const DataObject *src, const ito::tDataType type, const void *scalar);
        template<typename _Tp> friend RetVal MakeContinuousFunc(const DataObject &dObj, DataObject &resDObj);
        template<typename _Tp> friend RetVal EvaluateTransposeFlagFunc(DataObject *dObj);
        template<typename _Tp> friend RetVal CalcMinMaxValues(DataObject *lhs, double &result_min, double &result_max, const int cmplxSel = 0);
        template<typename _Tp> friend std::ostream& coutFunc(std::ostream& out, const DataObject& dObj);

        // more friends due to change of std::vector to int ** for m_data ...
        template<typename _Tp> friend RetVal GetRangeFunc(DataObject *dObj, const int dtop, const int dbottom, const int dleft, const int dright);
        template<typename _Tp> friend RetVal AdjustROIFunc(DataObject *dObj, int dtop, int dbottom, int dleft, int dright);

    public:
        int seekMat(const int matNum, const int numMats) const;
        int seekMat(const int matNum) const;
        int calcNumMats(void) const;

        // TAGSPACEFUNCTIONS

        //!< Function return the offset of the values stored within the dataOject
        double getValueOffset() const;

        //!< Function return the scaling of values stored within the dataOject
        double getValueScale() const;

        //!< Function return the unit description for the values stoerd within the dataOject
        const std::string getValueUnit() const;

        //!< Function return the description for the values stored within the dataOject, if tagspace does not exist, NULL is returned.
        std::string getValueDescription() const;

        //!< Function return the axis-offset for the existing axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
        double getAxisOffset(const int axisNum) const;
        
        //!< Function returns the axis-description for the exist axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
        double getAxisScale(const int axisNum) const;
        
        //!< Function returns the axis-unit-description for the exist axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
        const std::string getAxisUnit(const int axisNum, bool &validOperation) const;
        
        //!< Function returns the axis-description for the exist specified by axisNum. If axisNum is out of dimension range it returns NULL.
        std::string getAxisDescription(const int axisNum, bool &validOperation) const;
        
        DataObjectTagType getTag(const std::string &key, bool &validOperation) const;
        
        bool getTagByIndex(const int tagNumber, std::string &key, DataObjectTagType &value) const;
        
        //!<  Function returns the string-value for 'key' identified by int tagNumber. If key in the TagMap do not exist NULL is returned
        std::string getTagKey(const int tagNumber, bool &validOperation) const;
        
        //!< Function returns the number of elements in the Tags-Maps
        int getTagListSize() const;

        //!<  Function to set the string-value of the value unit, return 1 if values does not exist
        int setValueUnit(const std::string &unit);

        //!<  Function to set the string-value of the value description, return 1 if values does not exist
        int setValueDescription(const std::string &description);

        //inline lead to a linker error on MSVC when calling from several methods
        int setAxisOffset(const unsigned int axisNum, const double offset);
        int setAxisScale(const unsigned int axisNum, const double scale);
        int setAxisUnit(const unsigned int axisNum, const std::string &unit);
        int setAxisDescription(const unsigned int axisNum, const std::string &description);
        int setTag(const std::string &key, const DataObjectTagType &value);
        bool existTag(const std::string &key) const;
        bool deleteTag(const std::string &key);
        bool deleteAllTags();
        int addToProtocol(const std::string &value);


        /**
        \brief Function returns the not rounded pixel index of a physical coordinate
        \detail Function returns the not rounded pixel index of a physical coordinate (Unit-Coordinate = ( px-Coordinate - Offset)* Scale).
                If the pixel is outside of the image, the isInsideImage-flag is set to false else it is set to true.
                To avoid memory access-error, the returnvalue is clipped within the range of the image ([0...imagesize-1])
        \param[in] dim  Axis-dimension for which the physical coordinate is calculated
        \param[in] pix  Pixel-index as double
        \param[out] isInsideImage   flag which is set to true if coordinate is within range of the image.
        \return (double)( pix / AxisScale + AxisOffset) & [0..imagesize-1]
        */
        inline double getPhysToPix(const unsigned int dim, const double phys, bool &isInsideImage) const
        {
            double tPx = 0.0;
            if(static_cast<int>(dim) >= m_dims)
            {
                if(phys == 0.0)
                {
                    isInsideImage = true;
                }
                else
                {
                    isInsideImage = false;
                }
                return 0.0;
            }

            if(m_pDataObjectTags)
            {
                tPx = (phys / getAxisScale(dim) + getAxisOffset(dim));
            }
            else
            {
                tPx = phys;
            }

            if(tPx > getSize(dim) - 1)
            {
                isInsideImage = false;
                tPx = static_cast<double>(getSize(dim)- 1);
            }
            else if( tPx < 0)
            {
                isInsideImage = false;
                tPx = 0;
            }
            else
            {
                isInsideImage = true;
            }

            return tPx;
        }

        /**
        \brief Function returns the not rounded pixel index of a physical coordinate
         */
        inline int getPhysToPix2D(const double physY, double &tPxY, bool &isInsideImageY, const double physX, double &tPxX, bool &isInsideImageX) const
        {
            if(m_dims < 2)
            {           
                    tPxY = physY;
                    if(physY != 0.0)
                    {
                        isInsideImageY = false;
                    }

                    tPxX = physX / getAxisScale(0) + getAxisOffset(0); //m_pDataObjectTags->m_axisScales[0] + m_pDataObjectTags->m_axisOffsets[0];

                    if(tPxX > m_size[0] - 1)
                    {
                        isInsideImageX = false;
                        tPxX = static_cast<double>(m_size[0] - 1);
                    }                
            }
            else
            {             
                    if(m_pDataObjectTags)
                    {
                        tPxX = physX / getAxisScale(m_dims - 1) + getAxisOffset(m_dims - 1);
                        tPxY = physY / getAxisScale(m_dims - 2) + getAxisOffset(m_dims - 2);
                    }
                    else
                    {
                        tPxX = physX;
                        tPxY = physY;
                    }

                    if(tPxY > m_size[m_dims - 2] - 1)
                    {
                        isInsideImageY = false;
                        tPxY = static_cast<double>(m_size[m_dims - 2] - 1);
                    }
                    if(tPxX > m_size[m_dims - 1] - 1)
                    {
                        isInsideImageX = false;
                        tPxY = static_cast<double>(m_size[m_dims - 1] - 1);
                    }                
            }

            if(tPxX < 0)
            {
                tPxX = 0;
                isInsideImageX = false;
            }
            if(tPxY < 0)
            {
                tPxY = 0;
                isInsideImageY = false;
            }
            return 0;
        }

        /**
        \brief Function returns the physical coordinate of a pixel
        \detail Function returns the physical coordinate of a pixel index (Unit-Coordinate = ( px-Coordinate - Offset)* Scale).
                If the pixel is outside of the image, the isInsideImage-flag is set to false else it is set to true
        \param[in] dim  Axis-dimension for which the physical coordinate is calculated
        \param[in] pix  Pixel-index as double
        \param[out] isInsideImage   flag which is set to true if coordinate is within range of the image.
        \return (double)( pix - AxisOffset)* AxisScale)
        */
        inline double getPixToPhys(const unsigned int dim, const double pix, bool &isInsideImage) const
        {
            double tPhys = 0.0;
            if(static_cast<int>(dim) >= m_dims)
            {
                if(pix == 0)
                {
                    isInsideImage = true;
                }
                else
                {
                    isInsideImage = false;
                }
                return 0.0;
            }
            if(m_pDataObjectTags)
            {
                tPhys = (pix - getAxisOffset(dim)) * getAxisScale(dim);
            }
            else
            {
                tPhys = pix;
            }

            if((pix > getSize(dim) - 1) || (pix < 0))
            {
                isInsideImage = false;
            }
            else
            {
                isInsideImage = true;
            }

            return tPhys;
        }

        /**
        \brief Function to access (set) the rotiational matrix by each element
        \param[in] r11  Upper left element
        \param[in] r12  Upper middle element
        \param[in] r13  Upper rigth element
        \param[in] r21  Middle left element
        \param[in] r22  Middle middle element
        \param[in] r23  Middle rigth element
        \param[in] r31  Lower left element
        \param[in] r32  Lower middle element
        \param[in] r33  Lower rigth element
        \return ito::retOk || ito::retError
        */
        RetVal setXYRotationalMatrix(double r11, double r12, double r13, double r21, double r22, double r23, double r31, double r32, double r33);

        /**
        \brief Function to access (get) the rotiational matrix by each element
        \param[out] r11  Upper left element
        \param[out] r12  Upper middle element
        \param[out] r13  Upper rigth element
        \param[out] r21  Middle left element
        \param[out] r22  Middle middle element
        \param[out] r23  Middle rigth element
        \param[out] r31  Lower left element
        \param[out] r32  Lower middle element
        \param[out] r33  Lower rigth element
        \return ito::retOk || ito::retError
        */
        RetVal getXYRotationalMatrix(double &r11, double &r12, double &r13, double &r21, double &r22, double &r23, double &r31, double &r32, double &r33) const;

        RetVal copyTagMapTo(DataObject &rhs) const;  /*!< Deep copies the tagmap with all entries to rhs object */
        RetVal copyAxisTagsTo(DataObject &rhs) const;  /*!< Deep copies the axistags to rhs object */

        // END TAGSPACE

        /*!< returns the number of dimensions */
        inline int getDims(void) const { return m_dims; }

        /*!< returns the element data type in form of its type-number */
        inline int getType(void) const { return m_type; }

        /*!< returns if the data in the first n-2 dimensions is stored within one entire block in memory (true), else (false) */
        inline char getContinuous(void) const { return m_continuous; }

        /*!< returns if the data object is owner of the data, hence, the data will be deleted by this data object, if nobody else is using the data any more */
        inline char getOwnData(void) const { return m_owndata; }
        
        //! gets total number of elements within the data object's ROI
        /*!
            \return number of elements
            \sa getDims, getSize
        */
        inline int getTotal() const
        {
            int dims = getDims();
            int total = dims > 0 ? 1 : 0;
            for(int i = 0 ; i<dims ; i++)
            {
                total *= getSize(i);
            }
            return total;

        }

        //! gets total number of elements of the whole data object
        /*!
            \return number of elements
            \sa getDims, getSize
        */
        inline int getOriginalTotal() const
        {
            int dims = getDims();
            int total = dims > 0 ? 1 : 0;
            for(int i = 0 ; i<dims ; i++)
            {
                total *= m_osize[i];
            }
            return total;

        }

        //! locks this dataObject (all header information) and the underlying data block for a read operation
        /*!
            \remark During the copy-constructor, operator=, eye, zero and ones method, the readWriteLock for the data block
                will be set to writeLock if any of the participating dataObjects are in writeLock mode. Then the number of readers
                will be decremented first. The lock of the dataObject, hence the lock for all header information, which are not
                shared, remains at its level.

            \sa ReadWriteLock
        */
        inline void lockRead()
        {
            m_objHeaderLock.lockRead();
            if(m_objSharedDataLock) m_objSharedDataLock->lockRead();
        }

        //! locks this dataObject (all header information) and the underlying data block for a write operation
        /*!
            \sa ReadWriteLock
        */
        inline void lockWrite()
        {
            m_objHeaderLock.lockWrite();
            if(m_objSharedDataLock) m_objSharedDataLock->lockWrite();
        }

        //! unlocks any lock. If lock is writeLock, lock is set to idle, if lock is readLock, then the number of readers is decremented and lock is freed if no more readers are available
        /*!
            \sa ReadWriteLock
        */
        inline void unlock()
        {
            if(m_objSharedDataLock) m_objSharedDataLock->unlock();
            m_objHeaderLock.unlock();
        }
        
        RetVal copyTo(DataObject &rhs, unsigned char regionOnly = 0);   /*!< deeply copies the data of this data object to the given rhs-dataObject, whose existing data will be deleted first. */
        RetVal convertTo(DataObject &rhs, const int type, const double alpha=1, const double beta=0 ) const;
        RetVal deepCopyPartial(DataObject &rhs);                         /*!< copies the values of each element from this data object to the ROI of the given rhs-dataObject. The rhs-dataObject must be allocated yet and its ROI must be the same size than this ROI */

        uchar ** get_mdata(void);
        uchar ** get_mdata(void) const;

        cv::Mat* getCvPlaneMat(const int planeIndex);
        const cv::Mat* getCvPlaneMat(const int planeIndex) const;

        //! returns the size-member. m_size fits to the physical organization of data in memory.
        /*!
            \return size-member of type MSize
        */
        inline MSize getSize(void) { return m_size; }

        //! returns the size-member. This member does not consider the transpose flag, hence, m_size fits to the physical organization of data in memory.
        /*!
            \return size-member of type MSize
        */
        inline const MSize getSize(void) const { return m_size; }

        //! gets the size of the given dimension (this is the size within the ROI)
        /*!
            \param index is the specific zero-based dimension-index whose size is requested           
            \return size or -1 if index is out of boundaries
        */
        int getSize(int index) const
        {
            if(index < 0 || index >= m_dims)
            {
                return -1;
            }
            else
            {
                return m_size[index];
            }
        }

        //! gets the original size of the given dimension (this is the size without considering any ROI)
        /*!
            \param index is the specific zero-based dimension-index whose size is requested           
            \return size or -1 if index is out of boundaries
        */
        int getOriginalSize(int index) const
        {
            if(index < 0 || index >= m_dims)
            {
                return -1;           
            }
            else
            {
                return m_osize[index];
            }
        }

        //! returns 0, this is the index of the first element in valid DataObject range.
        DObjIterator begin();
        //! returns index of last-element in DataObject range incremented by one. (equal to number of elements in total range)
        DObjIterator end();

        DObjConstIterator constBegin() const;
        DObjConstIterator constEnd() const;

        //! constructor for empty data object
        /*!
            no data will be allocated, the number of elements and dimensions is set to zero
        */
        DataObject(void) : m_continuous(1), m_owndata(1), m_type(0), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0) {}

        //! constructor for one-dimensional data object. The data is newly allocated and arbitrarily filled.
        /*!
            In fact, by this constructor a two-dimensional matrix with dimension 1 x size will be created.
            the owndata-flag is set to true, the continuously-flag, too (since only one matrix-plane will be created)

            \param size is the number of elements
            \param type is the data-type of each element (use type of enumeration tDataType)
            \sa create, tDataType
        */
        DataObject(const int size, const int type): m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            int sizes[2] = {1, size};
            this->create(2, sizes, type, 1);
        }

        //! constructor for two-dimensional data object. The data is newly allocated and arbitrarily filled.
        /*!
            the owndata-flag is set to true, the continuously-flag, too (since only one matrix-plane will be created)

            \param sizeY is the number of rows in each matrix-plane
            \param sizeX is the number of columns in each matrix-plane
            \param type is the data-type of each element (use type of enumeration tDataType)
            \sa create, tDataType
        */
        DataObject(const int sizeY, const int sizeX, const int type): m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            int sizes[2] = {sizeY, sizeX};
            this->create(2, sizes, type, 1);
        }

        //! constructor for three-dimensional data object. The data is newly allocated and arbitrarily filled.
        /*!
            the owndata-flag is set to true

            \param sizeZ is the number of images in the z-direction
            \param sizeY is the number of rows in each matrix-plane
            \param sizeX is the number of columns in each matrix-plane
            \param type is the data-type of each element (use type of enumeration tDataType)
            \param continuous indicates whether all matrix-planes should continuously lie in memory (1) or not (0) (default: 0)
            \sa create, tDataType
        */
        DataObject(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous = 0) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            int sizes[3] = {sizeZ, sizeY, sizeX};
            this->create(3, sizes, type, m_continuous);
        }

        //! constructor for 3-dimensional data object which uses the data given by the continuousDataPtr.
        /*!
            In case of the continuousDataPtr, the owndata-flag is set to false, hence this dataObj will not delete the data.
            Additionally the continuous-flag is set to true.

            \param sizeZ is the number of images in the z-direction
            \param sizeY is the number of rows in each matrix-plane
            \param sizeX is the number of columns in each matrix-plane
            \param type is the data-type of each element (use type of enumeration tDataType)
            \param *continuousDataPtr points to the first element of a continuous data block of the specific data type
            \param *steps may be NULL, if the data in continuousDataPtr should be taken continuously, hence the ROI is the whole matrix, else
                    this is a vector with three elements, where each elements indicates the number of bytes one has to move in order to get from
                    one element to the next one in the same dimension. Hence, the last element in this vector is equal to the size of one single element (in bytes)
            \sa create, tDataType
        */
        DataObject(const int sizeZ, const int sizeY, const int sizeX, const int type, const uchar* continuousDataPtr,  const int* steps = NULL) : m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            int sizes[3] = {sizeZ, sizeY, sizeX};
            this->create(3, sizes, type, m_continuous, continuousDataPtr, steps);
        }

        //! constructor for data object with given dimension. The data is newly allocated and arbitrarily filled.
        /*!
            the owndata-flag is set to true

            \param dimensions indicates the total number of dimensions
            \param *sizes is a vector of size 'dimensions', where each element gives the size (not osize) of the specific dimension
            \param type is the data-type of each element (use type of enumeration tDataType)
            \param continuous indicates whether all matrix-planes should continuously lie in memory (1) or not (0) (default: 0)
            \sa create, tDataType
        */
        DataObject(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous = 0) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            this->create(dimensions, sizes, type, m_continuous);
        }

        //! constructor for data object which uses the data given by the continuousDataPtr.
        /*!
            In case of the continuousDataPtr, the owndata-flag is set to false, hence this dataObj will not delete the data.
            Additionally the continuous-flag is set to true.

            \param dimensions indicates the total number of dimensions
            \param *sizes is a vector of size 'dimensions', where each element gives the size (not osize) of the specific dimension
            \param type is the data-type of each element (use type of enumeration tDataType)
            \param *continuousDataPtr points to the first element of a continuous data block of the specific data type
            \param *steps may be NULL, if the data in continuousDataPtr should be taken continuously, hence the ROI is the whole matrix, else
                    this is a vector of size 'dimensions', where each elements indicates the number of bytes one has to move in order to get from
                    one element to the next one in the same dimension. Hence, the last element in this vector is equal to the size of one single element (in bytes)
            \sa create, ito::tDataType
        */
        DataObject(const unsigned char dimensions, const int *sizes, const int type, const uchar* continuousDataPtr, const int* steps = NULL) : m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            this->create(dimensions, sizes, type, m_continuous, continuousDataPtr, steps);
        }

        DataObject(const unsigned char dimensions, const int *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes) : m_continuous(0), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            //usually it is dangerous to say that m_owndata is 1 in this case, since we cannot be sure if the given planes are the owner of their data.
            //however, in this case, owndata is unimportant since the created dataObject is always not continuous, therefore owndata will never
            //be analyzed and the destructor of the dataObject never tries to delete the continuous data block. The underlying cv::Mats however still know
            //whether they can or can't delete their data.
            this->create(dimensions, sizes, type, planes, nrOfPlanes);
        }

        DataObject(const DataObject& copyConstr);    /*!< copy constructor */

        //! destructor
        /*!
            reference pointer of data is decremented and if <0, data will be deleted if owndata-flag is true. Additionally the allocated memory for
            header information will be deleted, too.

            \sa freeData
        */
        ~DataObject(void)
        {
            freeData();
        }

        //// Arithmetic Operators
        DataObject & operator = (const cv::Mat &rhs);
        DataObject & operator = (const DataObject &rhs);
        DataObject & operator = (const int8 &value);          /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const uint8 &value);         /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const int16 &value);         /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const uint16 &value);        /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const int32 &value);         /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const uint32 &value);        /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const float32 &value);       /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const float64 &value);       /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const complex64 &value);     /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const complex128 &value);    /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const ito::Rgba32 &value);   /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */


        DataObject & operator += (const DataObject &rhs);
        DataObject & operator += (const float64 &value);

        DataObject operator + (const DataObject &rhs);
        DataObject operator + (const float64 &value);

        DataObject & operator -= (const DataObject &rhs);
        DataObject & operator -= (const float64 &value);

        DataObject operator - (const DataObject &rhs);
        DataObject operator - (const float64 &value);

        DataObject & operator *= (const DataObject &rhs);
        DataObject & operator *= (const float64 &factor);

        DataObject operator * (const DataObject &rhs);
        DataObject operator * (const float64 &factor);

        // Comparison Operators
        DataObject operator < (DataObject &rhs);
        DataObject operator > (DataObject &rhs);
        DataObject operator <= (DataObject &rhs);
        DataObject operator >= (DataObject &rhs);
        DataObject operator == (DataObject &rhs);
        DataObject operator != (DataObject &rhs);


        // bitshift operators
        DataObject operator << (const unsigned int shiftbit);
        DataObject & operator <<= (const unsigned int shiftbit);
        DataObject operator >> (const unsigned int shiftbit);
        DataObject & operator >>= (const unsigned int shiftbit);

        // bitwise operators
        DataObject operator & (const DataObject & rhs);
        DataObject & operator &= (const DataObject & rhs);
        DataObject operator | (const DataObject & rhs);
        DataObject & operator |= (const DataObject & rhs);
        DataObject operator ^ (const DataObject & rhs);
        DataObject & operator ^= (const DataObject & rhs);

        // allocates matrix with zero values
        RetVal zeros(const int type);
        RetVal zeros(const int size, const int type);
        RetVal zeros(const int sizeY, const int sizeX, const int type);
        RetVal zeros(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous = 0);
        RetVal zeros(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous = 0);

        // allocates matrix with all values set to one
        RetVal ones(const int type);
        RetVal ones(const int size, const int type);
        RetVal ones(const int sizeY, const int sizeX, const int type);
        RetVal ones(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous = 0);
        RetVal ones(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous = 0);

        // allocates matrix with uniform distributed noise
        RetVal rand(const int type, const bool randMode = false);
        RetVal rand(const int size, const int type, const bool randMode = false);
        RetVal rand(const int sizeY, const int sizeX, const int type, const bool randMode = false);
        RetVal rand(const int sizeZ, const int sizeY, const int sizeX, const int type, const bool randMode, const unsigned char continuous = 0);
        RetVal rand(const unsigned char dimensions, const int *sizes, const int type, const bool randMode, const unsigned char continuous = 0);

        // allocates matrix with eye-matrix representation
        RetVal eye(const int type);
        RetVal eye(const int size, const int type);

        RetVal conj();
        DataObject adj() const;
        DataObject trans() const;

        //RetVal makeContinuous(void);

        DataObject mul(const DataObject &mat2, const double scale = 1.0);
        DataObject div(const DataObject &mat2, const double scale = 1.0);
        DataObject squeeze() const;
        int elemSize() const;

        //! addressing method for two-dimensional data object.
        /*!
            \param y is the zero-based row-index to the element which is requested (considering any ROI)
            \param x is the zero-based column-index to the element which is requested (considering any ROI)
            \return const reference to specific element
        */
        template<typename _Tp> _Tp& at(const unsigned int y, const unsigned int x) const
        {
         #if __ITODEBUG
            if (m_dims != 2)
            {
               cv::error(cv::Exception(CV_StsAssert, "Dimension mismatch while addressing data field", "", __FILE__, __LINE__));
            }
            else if (((int)x >= m_size[1]) || ((int)y >= m_size[0]) )
            {
                cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__ , __LINE__));
            }
         #endif           
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(y, x);
        }

        //! addressing method for two-dimensional data object.
        /*!
            \param y is the zero-based row-index to the element which is requested (considering any ROI)
            \param x is the zero-based column-index to the element which is requested (considering any ROI)
            \return reference to specific element
        */
        template<typename _Tp> _Tp& at(const unsigned int y, const unsigned int x)
        {
         #if __ITODEBUG
            if (m_dims != 2)
            {
               cv::error(cv::Exception(CV_StsAssert, "Dimension mismatch while addressing data field", "", __FILE__, __LINE__));
            }          
            else if (((int)x >= m_size[1]) || ((int)y >= m_size[0]) )
            {
                cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__ , __LINE__));
            }
            
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(y, x);
        }

        //! addressing method for three-dimensional data object.
        /*!
            \param z is the zero-based z-index to the element which is requested (considering any ROI)
            \param y is the zero-based row-index to the element which is requested (considering any ROI)
            \param x is the zero-based column-index to the element which is requested (considering any ROI)
            \return const reference to specific element
        */
        template<typename _Tp> _Tp& at(const unsigned int z, const unsigned int y, const unsigned int x) const
        {
         #if __ITODEBUG
            if (m_dims != 3)
            {
               cv::error(cv::Exception(CV_StsAssert, "Dimension mismatch while addressing data field", "", __FILE__, __LINE__));
            }
            else if (((int)x >= m_size[2]) || ((int)y >= m_size[1]) || (((int)z + m_roi[0]) >= (m_roi[0] + m_size[0])))
            {
                cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__ , __LINE__));
            }
         #endif
            
               return (*(cv::Mat_<_Tp> *)(m_data[z + m_roi[0]]))(y, x);
        }

        //! addressing method for three-dimensional data object.
        /*!
            \param z is the zero-based z-index to the element which is requested (considering any ROI)
            \param y is the zero-based row-index to the element which is requested (considering any ROI)
            \param x is the zero-based column-index to the element which is requested (considering any ROI)
            \return reference to specific element
        */
        template<typename _Tp> _Tp& at(const unsigned int z, const unsigned int y, const unsigned int x)
        {
         #if __ITODEBUG
            if (m_dims != 3)
            {
               cv::error(cv::Exception(CV_StsAssert, "Dimension mismatch while addressing data field", "", __FILE__, __LINE__));
            }
            else if (((int)x >= m_size[2]) || ((int)y >= m_size[1]) || (((int)z + m_roi[0]) >= (m_roi[0] + m_size[0])))
            {
                cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__ , __LINE__));
            }
         #endif
            return (*(cv::Mat_<_Tp> *)(m_data[z + m_roi[0]]))(y, x);
        }

        //! addressing method for n-dimensional data object.
        /*!
            \param *idx is vector whose size is equal to the data object's dimensions. Each entry indicates the zero-based index of its specific dimension considering any ROI
            \remark The idx vector must indicate the indizes in "virtual"-order (user-friendly order)
            \return const reference to specific element
        */
        template<typename _Tp> _Tp& at(const unsigned int *idx) const //idx is in virtual order 
        {
            int matNum = 0;

            matIdxToNum(idx, &matNum);

            return (*(cv::Mat_<_Tp> *)(m_data[matNum]))(idx[m_dims - 2], idx[m_dims - 1]);
        }

        //! addressing method for n-dimensional data object.
        /*!
            \param *idx is vector whose size is equal to the data object's dimensions. Each entry indicates the zero-based index of its specific dimension considering any ROI
            \remark The idx vector must indicate the indizes in "virtual"-order (user-friendly order)
            \return reference to specific element
        */
        template<typename _Tp> _Tp& at(const unsigned int *idx) //idx is in virtual order 
        {
            int matNum = 0;

            matIdxToNum(idx, &matNum);

            return (*(cv::Mat_<_Tp> *)(m_data[matNum]))(idx[m_dims - 2], idx[m_dims - 1]);
        }

        DataObject at(const ito::Range &rowRange, const ito::Range &colRange);     /*!< addressing method for two-dimensional data object with two given range-values. returns shallow copy of addressed regions */
        DataObject at(ito::Range *ranges);                                       /*!< addressing method for n-dimensional data object with n given range-values. returns shallow copy of addressed regions */

        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
            cast this pointer to the data type of the matrix elements (as pointer).

            \remark No further error checking (e.g. boundaries)
            \return data-pointer
        */
        uchar* rowPtr(const int matNum, const int y)
        {
            int matIndex = seekMat(matNum);
            return ((cv::Mat*)m_data[matIndex])->ptr(y);
        }

        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
            cast this pointer to the data type of the matrix elements (as pointer).

            \remark No further error checking (e.g. boundaries)
            \return data-pointer
        */
        const uchar* rowPtr(const int matNum, const int y) const
        {
            int matIndex = seekMat(matNum);
            return ((cv::Mat*)m_data[matIndex])->ptr(y);
        }

        DataObject row(const int selRow);
        DataObject col(const int selCol);

        DataObject toGray(const int destinationType = ito::tUInt8);

        // ROI
        DataObject & adjustROI(const int dtop, const int dbottom, const int dleft, const int dright);   /*!< changes the boundaries of the ROI of a two-dimensional data object by the given incremental values */
        DataObject & adjustROI(const unsigned char dims, const int *lims);                              /*!< changes the boundaries of the ROI of a n-dimensional data object by the given incremental values */
        RetVal locateROI(int *wholeSizes, int *offsets);                              /*!< locates the boundaries of the ROI of a n-dimensional data object and returns the original size and the distances to the physical borders */
        RetVal locateROI(int *lims);                                                  /*!< locates the boundaries of the ROI of a n-dimensional data object the distances to the physical borders */

        //! copies the externally given source data inside this data object
        /*!
            This method obtains an externally given source array that must have the same
            element type than this data object. Its dimension is given by sizeX and sizeY and
            must correspond to the x-size and y-size of this data object. It is allowed that
            this data object is a shallow copy with a possible region of interest of another (bigger)
            object.

            Then, the given array is copied inside of the values of the data object. The external 
            array must have a row-wise data arrangment (c-style), hence, one row follows after the other one.
            
            \param _Tp* src is the source array. The type of the array is analyzed at compile time (_Tp is the placeholder for this type as template parameter)
            \param sizeX is the width of the array and must fit to the plane width of the data object
            \param sizeY is the height of the array and must fit to the plane height of the data object
            \return RetVal error if sizeX or sizeY does not fit to the size of the data object or if the type of the given array does not fit to the type of the data object
        */
        template<typename _Tp> RetVal copyFromData2D(const _Tp* src, const int sizeX, const int sizeY) { return copyFromData2DInternal((const uchar*)src, sizeof(_Tp), sizeX, sizeY); }        // copies 2D continuous data into data object, data object must have correct size and type, otherwise an error is returned

        //! copies the externally given source data inside this data object
        /*!
            This method obtains an externally given source array that must have the same
            element type than this data object. Its dimension is given by sizeX and sizeY and
            must correspond to the x-size and y-size of this data object. It is allowed that
            this data object is a shallow copy with a possible region of interest of another (bigger)
            object.

            Then, the given array is copied inside of the values of the data object. The external 
            array must have a row-wise data arrangment (c-style), hence, one row follows after the other one.

            In this method, it is allowed that the original width and height of the given data is different
            than the plane size of this data object. Then only a subregion of the external data is copied, indicated
            by the x0 and y0 indices of the first value and its width and height (sizeX and sizeY are the original size of
            the given data). width and height must correspond to the plane size of the data object.
            
            \param _Tp* src is the source array. The type of the array is analyzed at compile time (_Tp is the placeholder for this type as template parameter)
            \param sizeX is the width of the array.
            \param sizeY is the height of the array.
            \param x0 is the x-index of the first value of the source data that is copied.
            \param y0 is the y-index of the first value of the source data that is copied.
            \param width is the width of the sub-region of the source data that should be copied (must fit to the width of the data object)
            \param height is the height of the sub-region of the source data that should be copied (must fit to the height of the data object)
            \return RetVal error if sizeX or sizeY does not fit to the size of the data object or if the type of the given array does not fit to the type of the data object
        */
        template<typename _Tp> RetVal copyFromData2D(const _Tp *src, const int sizeX, const int sizeY, const int x0, const int y0, const int width, const int height) { return copyFromData2DInternal((const uchar*)src, sizeof(_Tp), sizeX, x0, y0, width, height); }      // copies 2D continuous data into data object, data object must have correct size and type, otherwise an error is returned
        
        //----------------------------------------------------------------------------------------------------------------------------------
        //! verifies if the data type of elements in this data object is equal to the type of the argument.
        /*
            \param [in] src is any variable whose type is checked
            \return retOk if both types are equal, retError if they are not equal or if the type of src is unknown
        */
        template<typename _Tp> RetVal checkType(const _Tp *src)    //must be inline since function template!
        {
            try
            {
                if(m_type == ito::getDataType(src))
                {
                    return ito::retOk;
                }
                return RetVal(retError,0,"CheckType failed: types are not equal");
            }
            catch(cv::Exception /*&ex*/)
            {
                return RetVal(retError,0,"Error during Type-Check. Type not templated");
            }
        }

        //
        template<typename T2> operator T2 ();  /*!< cast operator, tries to cast this data object to another element type */

        template<typename _Tp> RetVal linspace(const _Tp start, const _Tp end, const _Tp inc, const int transposed);

};

//template<> DATAOBJ_EXPORT RetVal ito::DataObject::linspace<ito::int8>(const ito::int8 /*start*/, const ito::int8 /*end*/, const ito::int8 /*inc*/, const int /*transposed*/);
//template<> DATAOBJ_EXPORT RetVal ito::DataObject::linspace<ito::uint8>(const ito::uint8 /*start*/, const ito::uint8 /*end*/, const ito::uint8 /*inc*/, const int /*transposed*/);


//----------------------------------------------------------------------------------------------------------------------------------
// functions for DataObject in namespace ITO, which are NOT member functions
//----------------------------------------------------------------------------------------------------------------------------------
DataObject DATAOBJ_EXPORT abs(const DataObject &dObj);              /*!< calculates the absolute values of each element in the given data object and returns the result as new data object */
DataObject DATAOBJ_EXPORT arg(const DataObject &dObj);              /*!< calculates the argument of each element in the given data object and returns the result as new data object */
DataObject DATAOBJ_EXPORT real(const DataObject &dObj);             /*!< calculates the real part of each element in the given data object and returns the result as new data object */
DataObject DATAOBJ_EXPORT imag(const DataObject &dObj);             /*!< calculates the imaginary part of each element in the given data object and returns the result as new data object */

DataObject DATAOBJ_EXPORT makeContinuous(const DataObject &dObj);   /*!< if the given data object is not continuously organized, copies the content to a new continuous data object */

template<typename _Tp, typename _T2> RetVal CastFunc(const DataObject *dObj, DataObject *resObj, double alpha = 1.0, double beta = 0.0);

//RetVal minMaxLoc(const DataObject &dObj, double *minVal, double *maxVal, int *minPos = NULL, int *maxPos = NULL);

//! templated method for converting a given scalar value to the data type, indicated by the template parameter
/*!
    \param fromType is the data type of the given scalar
    \param *scalar is the pointer to the scalar value, casted to void*
    \return the converted scalar value
    \throws cv::Exception if the input data type is unknown or if the conversion failed
    \sa saturate_cast
*/
template<typename _Tp> _Tp numberConversion(ito::tDataType fromType, void *scalar)
{
    //_Tp retValue = 0;
    _Tp retValue;

    switch(fromType)
    {
    case ito::tUInt8:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<uint8*>(scalar)));
        break;
    case ito::tInt8:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<int8*>(scalar)));
        break;
    case ito::tUInt16:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<uint16*>(scalar)));
        break;
    case ito::tInt16:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<int16*>(scalar)));
        break;
    case ito::tUInt32:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<uint32*>(scalar)));
        break;
    case ito::tInt32:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<int32*>(scalar)));
        break;
    case ito::tFloat32:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<ito::float32*>(scalar)));
        break;
    case ito::tFloat64:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<ito::float64*>(scalar)));
        break;
    case ito::tComplex64:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<ito::complex64*>(scalar)));
        break;
    case ito::tComplex128:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<ito::complex128*>(scalar)));
        break;
    case ito::tRGBA32:
        retValue = cv::saturate_cast<_Tp>(*(static_cast<ito::Rgba32*>(scalar)));
        break;
    default:
        cv::error(cv::Exception(CV_StsAssert, "Input value type unkown", "", __FILE__, __LINE__));
        retValue = 0;
    }

    return retValue;
};

//template<> ito::Rgba32 numberConversion<ito::Rgba32>(ito::tDataType fromType, void *scalar)
//{
//    _Tp retValue = 0;
//
//    switch(fromType)
//    {
//    case ito::tUInt8:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<uint8*>(scalar)));
//        break;
//    case ito::tInt8:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<int8*>(scalar)));
//        break;
//    case ito::tUInt16:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<uint16*>(scalar)));
//        break;
//    case ito::tInt16:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<int16*>(scalar)));
//        break;
//    case ito::tUInt32:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<uint32*>(scalar)));
//        break;
//    case ito::tInt32:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<int32*>(scalar)));
//        break;
//    case ito::tFloat32:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<ito::float32*>(scalar)));
//        break;
//    case ito::tFloat64:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<ito::float64*>(scalar)));
//        break;
//    case ito::tComplex64:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<ito::complex64*>(scalar)));
//        break;
//    case ito::tComplex128:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<ito::complex128*>(scalar)));
//        break;
//    case ito::tRGBA32:
//        retValue = cv::saturate_cast<ito::Rgba32>(*(static_cast<ito::Rgba32*>(scalar)));
//        break;
//    default:
//        cv::error(cv::Exception(CV_StsAssert, "Input value type unkown", "", __FILE__, __LINE__));
//        retValue = 0;
//    }
//
//    return retValue;
//}

//----------------------------------------------------------------------------------------------------------------------------------
// cout
//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> static std::ostream& coutFunc(std::ostream& out, const DataObject& dObj)
{
    //cv::Mat_<_Tp> *cvMat = NULL;
    int numMats = dObj.calcNumMats();
    int tMat = 0;

    if (dObj.getDims() == 0)
    {
        std::cout << "DataObject()\n" << std::endl;
    }
    else
    {
        std::cout << "DataObject(size=[" << dObj.getSize(0);
        for (int dim = 1; dim < dObj.getDims(); ++dim)
        {
            std::cout << "x" << dObj.getSize(dim);
        }
        std::cout << "]\n" << std::endl;

        int *idx = new int[dObj.getDims()];
         

        for (int nMat = 0; nMat < numMats; nMat++)
        {
            tMat = dObj.seekMat(nMat, numMats);
            //std::cout <<  tMat + 1 << "->(";

            dObj.matNumToIdx(tMat, idx);
            std::cout << "[";
            for (int i = 0; i < dObj.getDims() - 2; ++i)
            {
                std::cout << idx[i] << ",";
            }
            std::cout << ":,:]->(";

            std::cout << cv::format( (*((cv::Mat_<_Tp> *)((dObj.get_mdata())[tMat]))) , "numpy" ) << std::endl << std::endl;        

            std::cout << ")" << "\n" << std::endl;
        }

        delete[] idx;

        std::cout << ")" << "\n" << std::endl;
    }
    return out;

}

typedef std::ostream& (*tCoutFunc)(std::ostream& out, const DataObject& dObj);

static tCoutFunc fListCout[] =
{
   coutFunc<int8>,
   coutFunc<uint8>,
   coutFunc<int16>,
   coutFunc<uint16>,
   coutFunc<int32>,
   coutFunc<uint32>,
   coutFunc<ito::float32>,
   coutFunc<ito::float64>,
   coutFunc<ito::complex64>,
   coutFunc<ito::complex128>,
   coutFunc<ito::Rgba32>
};

static inline std::ostream& operator << (std::ostream& out, const DataObject& dObj)
{
   return fListCout[dObj.getType()](out, dObj);
}

//! static method which returns the real data type of any given data type
/*!
    If the given data type is already real, the same type is returned. Else the type of the real argument of the given complex type is returned.

    \param cmplxType is the input data type
    \return see method's description
    \throws cv::Exception if the input data type is unknown
*/
static ito::tDataType convertCmplxTypeToRealType(ito::tDataType cmplxType)
{
    switch(cmplxType)
    {
        case ito::tInt8:
        case ito::tUInt8:
        case ito::tInt16:
        case ito::tUInt16:
        case ito::tInt32:
        case ito::tUInt32:
        case ito::tFloat32:
        case ito::tFloat64:
        case ito::tRGBA32:
            return cmplxType;
        case ito::tComplex64:
            return ito::tFloat32;
        case ito::tComplex128:
            return ito::tFloat64;
    }

    cv::error(cv::Exception(CV_StsAssert, "Input data type unknown", "", __FILE__, __LINE__));
    return ito::tInt8;
}

//! static method which guesses the dataObject type from a given cv::Mat*
/*!
    If the given data type is already real, the same type is returned. Else the type of the real argument of the given complex type is returned.

    \param mat is the OpenCV matrix.
    \param retval an error value will be added if the type cannot be converted.
    \return ito::DataObject type that fits to the given matrix
*/
static ito::tDataType guessDataTypeFromCVMat(const cv::Mat* mat, ito::RetVal &retval)
{
    if (mat)
    {
        switch(mat->type())
        {
        case cv::DataType<ito::int8>::type:
            return ito::tInt8;
        case cv::DataType<ito::uint8>::type:
            return ito::tUInt8;
        case cv::DataType<ito::int16>::type:
            return ito::tInt16;
        case cv::DataType<ito::uint16>::type:
            return ito::tUInt16;
        case cv::DataType<ito::int32>::type:
            return ito::tInt32;
        case cv::DataType<ito::uint32>::type:
            return ito::tUInt32;
        case cv::DataType<ito::float32>::type:
            return ito::tFloat32;
        case cv::DataType<ito::float64>::type:
            return ito::tFloat64;
        case cv::DataType<ito::Rgba32>::type:
            return ito::tRGBA32;
        case cv::DataType<ito::complex64>::type:
            return ito::tComplex64;
        case cv::DataType<ito::complex128>::type:
            return ito::tComplex128;
        }

        retval += ito::RetVal(ito::retError, 0, "type of cv::Mat is incompatible to ito::DataObject");
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, "given cv::Mat is NULL.");
    }
    return ito::tInt8;
}




//! method which returns the value of enumeration ito::tDataType, which corresponds to the template parameter (must be a pointer).
/*!
    If the template parameter cannot be transformed into a value of ito::tDataType, an exception is thrown.

    Call is ito::tDataType result = getDataType2<uint8*>().

    \return ito::tDataType
    \throws cv::Exception if the template parameter is unknown (e.g. no pointer).
    \sa getDataType
*/
template<typename _Tp> static inline ito::tDataType getDataType2()
{
    cv::error(cv::Exception(CV_StsAssert, "Input value type unkown", "", __FILE__, __LINE__));
    return ito::tInt8;
}

template<> inline ito::tDataType getDataType2<uint8*>()      { return ito::tUInt8; }
template<> inline ito::tDataType getDataType2<int8*>()       { return ito::tInt8; }
template<> inline ito::tDataType getDataType2<uint16*>()     { return ito::tUInt16; }
template<> inline ito::tDataType getDataType2<int16*>()      { return ito::tInt16; }
template<> inline ito::tDataType getDataType2<uint32*>()     { return ito::tUInt32; }
template<> inline ito::tDataType getDataType2<int32*>()      { return ito::tInt32; }
template<> inline ito::tDataType getDataType2<float32*>()    { return ito::tFloat32; }
template<> inline ito::tDataType getDataType2<float64*>()    { return ito::tFloat64; }
template<> inline ito::tDataType getDataType2<complex64*>()  { return ito::tComplex64; }
template<> inline ito::tDataType getDataType2<complex128*>() { return ito::tComplex128; }
template<> inline ito::tDataType getDataType2<Rgba32*>() { return ito::tRGBA32; }

//! method returns whether a given variable is equal to zero.
/*!
    For floating point variables, this method considers a variable to be zero, if its value
    lie within the boundaries (-epsilon,epsilon). Epsilon can for example be obtained by
    std::numeric_limits<_Tp>::epsilon(). For floating point values only parameters of type
    float32, float64, complex64 or complex128 are treated in the desired way.

    \param v is value to check
    \param epsilon is epsilon boundary, for fixed-point values this value is ignored.
    \return true if value is zero or within the epsilon boundaries, else false
*/
template<typename _Tp> static inline bool isZeroValue(_Tp v, _Tp /*epsilon*/)
{
    return v == 0;
}
template<> inline bool isZeroValue(Rgba32 v, Rgba32 /*epsilon*/)
{
    return v == Rgba32::zeros();
}
template<> inline bool isZeroValue(float32 v, float32 epsilon)
{
    return v >= epsilon ? false : (v <= -epsilon ? false : true);
}
template<> inline bool isZeroValue(float64 v, float64 epsilon)
{
    return v >= epsilon ? false : (v <= -epsilon ? false : true);
}
template<> inline bool isZeroValue(std::complex<ito::float32> v, std::complex<ito::float32> epsilon)
{
    return isZeroValue<ito::float32>(v.real(),epsilon.real()) && isZeroValue<ito::float32>(v.imag(),epsilon.real());
}
template<> inline bool isZeroValue(std::complex<ito::float64> v, std::complex<ito::float64> epsilon)
{
    return isZeroValue<ito::float64>(v.real(),epsilon.real()) && isZeroValue<ito::float64>(v.imag(),epsilon.real());
}


//----------------------------------------------------------------------------------------------------------------------------------
// friend functions to access private members of DataObject
//----------------------------------------------------------------------------------------------------------------------------------
} //namespace ito

#endif //__DATAOBJH


