/* ********************************************************************
 itom software
 URL: http://www.uni-stuttgart.de/ito
 Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
 General Public Licence for more details.
 
 You should have received a copy of the GNU Library General Public License
 along with itom. If not, see <http://www.gnu.org/licenses/>.
 *********************************************************************** */

#ifndef __DATAOBJH
#define __DATAOBJH

#include "defines.h"

#include <cstdlib>
#include <iostream>
#include <complex>
#include <limits>
#include <string>

#ifdef WIN32
    #pragma warning(disable:4996)
#endif

#define NOMINMAX //see: http://social.msdn.microsoft.com/Forums/sv/vclanguage/thread/d986a370-d856-4f9e-9f14-53f3b18ab63e, this is only an issue with OpenCV 2.4.3, not 2.3.x

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

#include "../common/sharedStructures.h"
#include "../common/color.h"
#include "../common/byteArray.h"


namespace cv
{
    template<> inline ito::float32 saturate_cast<ito::float32>( ito::float64 v)
    {
        if(cvIsInf(v)) return std::numeric_limits<ito::float32>::infinity();
        if(cvIsNaN(v)) return std::numeric_limits<ito::float32>::quiet_NaN();
        return static_cast<ito::float32>(std::max ( (ito::float64)(- std::numeric_limits<ito::float32>::max()) ,  std::min ( v , (ito::float64) std::numeric_limits<ito::float32>::max() )));
    }
    
    template<> inline ito::float64 saturate_cast<ito::float64>( ito::float32 v)
    {
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

    //from CV 3.3.1 on, the default implementation of DataType is dropped:
    //Original note in traits.hpp of OpenCV: Default values were dropped to stop confusing developers about using of unsupported types (see #7599)
    template<> class DataType<ito::uint32>
    {
    public:
        typedef ito::uint32 value_type;
        typedef value_type work_type;
        typedef value_type channel_type;
        typedef value_type vec_type;
        enum { generic_type = 1, depth = -1, channels = 1, fmt=0,
            type = CV_MAKETYPE(depth, channels) };
    };
    
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
    
    //forward declarations
    class DObjIterator;
    class DataObject;
    class Range;
    class DataObjectTags;
    class DataObjectTagType;
    class DataObjectTagsPrivate;
    
    
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
     \class Range
     \brief each range value has a start and end point. Optionally range can be marked as Range::all(), which indicates a full range
     
     start always indicates the first zero-based index of a range, end is the excluded index of the range, hence one item after the last item in the range.
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
    /*!
     \class DataObjectTagType
     \brief Variant storage class for either a double or a string value.
     
     The tag map of a data object always contains values of this class such that tags can be either of type string or double.
     */
    class DATAOBJ_EXPORT DataObjectTagType
    {
    public:
        enum tTagType
        {
            typeInvalid     = 0x000000, //!< invalid tag type
            typeDouble      = 0x000008, //!< tag type double
            typeString      = 0x000020  //!< tag type string (\sa ByteArray)
        };
        
    private:
        double m_dVal;          //!< if the tag type is double, the real double value is stored here
        tTagType m_type;        //!< type indicator of this class (invalid, double or string)
        ByteArray m_strValue;   //!< if the tag type is string, the string data is stored in this ByteArray variable.
        
    public:
        //!< Constructor
        DataObjectTagType() : m_dVal(0), m_strValue(""), m_type(DataObjectTagType::typeInvalid){}
        DataObjectTagType(double dVal) : m_dVal(dVal), m_strValue(""), m_type(DataObjectTagType::typeDouble){}
        DataObjectTagType(const std::string &str) : m_dVal(std::numeric_limits<double>::quiet_NaN()), m_type(DataObjectTagType::typeString){ m_strValue = str.data(); }
        DataObjectTagType(const ByteArray &str) : m_dVal(std::numeric_limits<double>::quiet_NaN()), m_type(DataObjectTagType::typeString){ m_strValue = str; }
        DataObjectTagType(const char* cVal) : m_dVal(std::numeric_limits<double>::quiet_NaN()), m_type(DataObjectTagType::typeString){ cVal == NULL ?  m_strValue = "" : m_strValue = cVal;}
        //!< Copy Constructor
        DataObjectTagType(const DataObjectTagType& a) : m_dVal(a.m_dVal), m_type(a.m_type), m_strValue(a.m_strValue) {}
        
        //! assignment operator will copy the content of the right-hand-sided DataObjectTagType to this object.
        DataObjectTagType & operator = (const DataObjectTagType &rhs)
        {
            this->m_dVal = rhs.m_dVal;
            this->m_strValue = rhs.m_strValue;
            this->m_type = rhs.m_type;
            
            return *this;
        }
        
        //! returns type of tag (\sa tTagType)
        inline int getType(void) const {return m_type;}
        
        //! returns if tag is valid (double or string) or invalid (e.g. due to use of default constructor)
        inline bool isValid(void) const { return (m_type == DataObjectTagType::typeInvalid) ? false: true;}
        
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
                
                std::ostringstream strs;
                strs << m_dVal;
                ByteArray ba(strs.str().data());
                
                return ba;
            }
        }
    };
    
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
     \class DObjConstIterator
     \brief constant iterator through data object
     
     Use this iterator to iterate over all items within the related data object.
     The order of the iteration is a C-wise ordering, hence row after row for a 2D data object.
     This iterator only allows reading the item values, use DObjIterator for an additional write access.
     
     \sa DObjIterator
     */
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
        const uchar* operator *() const;
        
        //! returns the i-th matrix element, relative to the current
        const uchar* operator [](int i) const;
        
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
        //! moves the iterator to an absolute position
        void seekAbs(int ofs);
        
        //! moves the iterator by a certain number of elements
        void seekRel(int ofs);
        
        const DataObject* dObj; //!< reference to the related data object
        bool   planeContinuous; //!< indicates whether dObj is continuously organized in each plane for faster seek operations
        int elemSize;           //!<
        uchar* ptr;             //!< pointer to the current value of the iterator
        uchar* sliceStart;      //!< pointer to the first item within the current continuous slice
        uchar* sliceEnd;        //!< pointer to the last item within the current continuous slice
        int plane;              //!< plane index where the iterator is currently positioned
    };
    
    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
     \class DObjIterator
     \brief iterator through data object
     
     Use this iterator to iterate over all items within the related data object.
     The order of the iteration is a C-wise ordering, hence row after row for a 2D data object.
     This iterator allows reading and writing the content of the data object.
     */
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

// Forward declaration of friend methods
#ifdef __APPLE__
    template<typename _Tp> RetVal CreateFunc(DataObject *dObj, const unsigned char dimensions, const int *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const int* steps);
    template<typename _Tp> RetVal CreateFuncWithCVPlanes(DataObject *dObj, const unsigned char dimensions, const int *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes);
    template<typename _Tp> RetVal FreeFunc(DataObject *dObj);
    template<typename _Tp> RetVal SecureFreeFunc(DataObject *dObj);
    template<typename _Tp> RetVal CopyToFunc(const DataObject &lhs, DataObject &rhs, unsigned char regionOnly);
    template<typename _Tp> RetVal ConvertToFunc(const DataObject &lhs, DataObject &rhs, const int type, const double alpha, const double beta);
    template<typename _Tp> RetVal AdjustROIFunc(DataObject *dObj, const int *lims);
    //template<typename _Tp> RetVal MinMaxLocFunc(const DataObject &dObj, double *minVal, double *maxVal, int *minPos, int *maxPos);
    template<typename _Tp> RetVal AssignScalarFunc(const DataObject *src, const ito::tDataType type, const void *scalar);
    template<typename _Tp> RetVal MakeContinuousFunc(const DataObject &dObj, DataObject &resDObj);
    template<typename _Tp> RetVal EvaluateTransposeFlagFunc(DataObject *dObj);
    template<typename _Tp> std::ostream& coutFunc(std::ostream& out, const DataObject& dObj);
    
    // more friends due to change of std::vector to int ** for m_data ...
    template<typename _Tp> RetVal GetRangeFunc(DataObject *dObj, const int dtop, const int dbottom, const int dleft, const int dright);
    template<typename _Tp> RetVal AdjustROIFunc(DataObject *dObj, int dtop, int dbottom, int dleft, int dright);
#endif // __APPLE__
    
    class DATAOBJ_EXPORT DataObject
    {
    private:
        //! create header information for data objects with a given size and step sizes to jump from one element in a dimension to the next one.
        void createHeader(const unsigned char dimensions, const int *sizes, const int *steps, const int elemSize);
        
        //! create header information for data objects with a given size, optional roi indeces and a possible original size
        void createHeaderWithROI(const unsigned char dimensions, const int *sizes, const int *osizes = NULL, const int *roi = NULL);
        
        void create(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous, const uchar* continuousDataPtr = NULL, const int* steps = NULL);  /*!< allocates new data */
        void create(const unsigned char dimensions, const int *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes);
        
        void freeData(void);     /*!< decrements reference counter and deletes data, if no other instance is using them (ref counter < 0) */
        void secureFreeData(void); /*!< decrements reference counter and deletes data, if no other instance is using them (ref counter < 0). This method makes a lot of security checks instead of freeFunc. */
        
        
        //----------------------------------------------------------------------------------------------------------------------------------
        ito::RetVal matNumToIdx(const int matNum, int *matIdx) const;
        
        //! \brief calculates the index of the matrix-plane in the m_data-vector for a given vector of indices, which address one element in the n-dimensional matrix
        ito::RetVal matIdxToNum(const unsigned int *matIdx, int *matNum) const;
        
        
        struct DATAOBJ_EXPORT MSize
        {
            inline MSize() : m_p(NULL) {}
            inline int operator [] (const int dim) const
            {
                return m_p[dim]; //return the size value. this operator corresponds to the real data representation in memory
            };
            inline operator const int * () const { return m_p; }
            inline bool operator == (const MSize& sz) const
            {
                if(m_p == NULL || sz.m_p == NULL)
                {
                    return sz.m_p == m_p;
                }
                
                int d = m_p[-1]; 
                if( d != sz.m_p[-1] )
                    return false;

                return (memcmp(m_p, sz.m_p, d * sizeof(int)) == 0); //returns true if the size vector (having the same length) is equal
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
            
            inline bool operator == (const MROI & rroi) const
            {
                if(m_p == NULL || rroi.m_p == NULL)
                {
                    return rroi.m_p == m_p;
                }

                int d = m_p[-1]; 
                if( d != rroi.m_p[-1] )
                    return false;

                return (memcmp(m_p, rroi.m_p, d * sizeof(int)) == 0); //returns true if the size vector (having the same length) is equal
                
                if (m_p[-1] != rroi.m_p[-1])
                {
                    return false;
                }
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
        DataObjectTagsPrivate *m_pDataObjectTags;    /*!< class containing the object metadata */
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
        //template<typename _Tp> friend RetVal MinMaxLocFunc(const DataObject &dObj, double *minVal, double *maxVal, int *minPos, int *maxPos);
        template<typename _Tp> friend RetVal AssignScalarFunc(const DataObject *src, const ito::tDataType type, const void *scalar);
        template<typename _Tp> friend RetVal MakeContinuousFunc(const DataObject &dObj, DataObject &resDObj);
        template<typename _Tp> friend RetVal EvaluateTransposeFlagFunc(DataObject *dObj);
        template<typename _Tp> friend std::ostream& coutFunc(std::ostream& out, const DataObject& dObj);
        
        // more friends due to change of std::vector to int ** for m_data ...
        template<typename _Tp> friend RetVal GetRangeFunc(DataObject *dObj, const int dtop, const int dbottom, const int dleft, const int dright);
        template<typename _Tp> friend RetVal AdjustROIFunc(DataObject *dObj, int dtop, int dbottom, int dleft, int dright);
        
    public:
        //! constructor for empty data object
        DataObject(void);
        
        //! constructor for one-dimensional data object. The data is newly allocated and arbitrarily filled.
        DataObject(const int size, const int type);
        
        //! constructor for two-dimensional data object. The data is newly allocated and arbitrarily filled.
        DataObject(const int sizeY, const int sizeX, const int type);
        
        //! constructor for three-dimensional data object. The data is newly allocated and arbitrarily filled.
        DataObject(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous = 0);
        
        //! constructor for 3-dimensional data object which uses the data given by the continuousDataPtr.
        DataObject(const int sizeZ, const int sizeY, const int sizeX, const int type, const uchar* continuousDataPtr,  const int* steps = NULL);

        //! constructor for data object with given dimension. The data is newly allocated and arbitrarily filled.
        DataObject(const MSize &sizes, const int type, const unsigned char continuous = 0);
        
        //! constructor for data object with given dimension. The data is newly allocated and arbitrarily filled.
        DataObject(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous = 0);
        
        //! constructor for data object which uses the data given by the continuousDataPtr.
        DataObject(const unsigned char dimensions, const int *sizes, const int type, const uchar* continuousDataPtr, const int* steps = NULL);
        
        //! constructor for data object from a stack of cv::Mat
        DataObject(const unsigned char dimensions, const int *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes);

        //! constructor for data object from a single cv::Mat
        DataObject(const cv::Mat &data);
        
        DataObject(const DataObject& copyConstr);    /*!< copy constructor */
        
        //! destructor
        ~DataObject(void);
        
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
        
        //!< set the offset of the axisNum-th axis. Offset is in pixel. The relation is: physical unit = (pixel unit - offset) * scale
        int setAxisOffset(const unsigned int axisNum, const double offset); 

        //!< set the scaling of the axisNum-th axis. Scaling is in (physical unit / pixel). The relation is: physical unit = (pixel unit - offset) * scale
        int setAxisScale(const unsigned int axisNum, const double scale);

        //!< set the unit of the axisNum-th axis as latin1 encoded string
        int setAxisUnit(const unsigned int axisNum, const std::string &unit);

        //!< set the description of the axisNum-th axis as latin1 encoded string
        int setAxisDescription(const unsigned int axisNum, const std::string &description);
        int setTag(const std::string &key, const DataObjectTagType &value);
        bool existTag(const std::string &key) const;
        bool deleteTag(const std::string &key);
        bool deleteAllTags();
        int addToProtocol(const std::string &value);
        
        
        /**
         \brief Function returns the not rounded pixel index of a physical coordinate
         */
        double getPhysToPix(const unsigned int dim, const double phys, bool &isInsideImage) const;

        /**
         \brief Function returns the not rounded pixel index of a physical coordinate
         */
        double getPhysToPix(const unsigned int dim, const double phys) const;
        
        /**
         \brief Function returns the not rounded pixel index of a physical coordinate
         */
        int getPhysToPix2D(const double physY, double &tPxY, bool &isInsideImageY, const double physX, double &tPxX, bool &isInsideImageX) const;
        
        /**
         \brief Function returns the physical coordinate of a pixel
         */
        double getPixToPhys(const unsigned int dim, const double pix, bool &isInsideImage) const;

        /**
         \brief Function returns the physical coordinate of a pixel
         */
        double getPixToPhys(const unsigned int dim, const double pix) const;
        
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
        
        /*!< returns the real plane index of cv::Mat array returned by get_mdata() for a given plane number considering a possible roi. Use this method if you already know the total number of planes within the roi. */
        int seekMat(const int matNum, const int numMats) const;
        
        /*!< returns the real plane index of cv::Mat array returned by get_mdata() for a given plane number considering a possible roi. This method internally calculates the number of planes within the roi using getNumPlanes. */
        int seekMat(const int matNum) const;
        
        /*!< returns the number of planes of this data object (considering a possible ROI). This method simply calls getNumPlanes and is only there for historical reasons.*/
        int calcNumMats(void) const;
        
        //! calculates numbers of single opencv matrices which are part of the ROI which has previously been set.
        /*!
         \return 0 if empty range or empty matrix, 1 if two dimensional, else product of sizes of all dimensions besides the last two ones.
         
         This method replaces calcNumMats due to its more consistent method name.
         */
        inline int getNumPlanes(void) const
        {
            switch (m_dims)
            {
                case 0:
                    return 0;
                case 1:
                case 2:
                    return 1;
                case 3:
                    return m_size[0];
                case 4:
                    return m_size[0] * m_size[1];
                default:
                {
                    int numMat = 1;
                    for (int n = 0; n < m_dims - 2; n++)
                    {
                        numMat *= m_size[n];
                    }
                    return numMat;
                }
            }
        }
        
        //! returns pointer to cv::Mat plane with given index considering a possible roi.
        cv::Mat* getCvPlaneMat(const int planeIndex);
        
        //! returns pointer to cv::Mat plane with given index considering a possible roi.
        const cv::Mat* getCvPlaneMat(const int planeIndex) const;

        //! returns a shallow or deep copy of a cv::Mat plane with given index. If the current plane is not continuous (due to a roi), a cloned, continuous matrix is returned, else a shallow copy.
        const cv::Mat getContinuousCvPlaneMat(const int planeIndex) const;
        
        //! returns array of pointers to cv::_Mat-matrices (planes) of the data object.
        /*!
         The returned array of matrices contains all matrices of this object, including the matrices that may
         lie outside of a possible region of interest. In order to access the i-th plane considering any roi,
         use getCvPlaneMat or calculate the right accessing index using seekMat.
         
         \return pointer to vector of matrices
         \remark the returned type is an array of cv::Mat*, you should cast it to the appropriate type (e.g. cv::_Mat<int8>)
         \sa seekMat, getCvPlaneMat
         */
        inline cv::Mat** get_mdata(void)
        {
            return (cv::Mat**)m_data;
        }
        
        //! returns constant array of pointers to cv::_Mat-matrices (planes) of the data object
        /*!
         The returned array of matrices contains all matrices of this object, including the matrices that may
         lie outside of a possible region of interest. In order to access the i-th plane considering any roi,
         use getCvPlaneMat or calculate the right accessing index using seekMat.
         
         \return pointer to vector of matrices
         \remark the returned type is a const array of cv::Mat*, you should cast it to the appropriate type (e.g. cv::_Mat<int8>)
         \sa seekMat, getCvPlaneMat
         */
        inline const cv::Mat** get_mdata(void) const
        {
            return (const cv::Mat**)m_data;
        }
        
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
        inline int getSize(int index) const
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

        //! returns the original size-member. This is equal to getSize() if no roi is set to the dataObject.
        /*!
        \return osize-member of type MSize
        */
        inline MSize getOriginalSize(void) { return m_osize; }

        //! returns the original size-member. This is equal to getSize() if no roi is set to the dataObject.
        /*!
        \return osize-member of type MSize
        */
        inline const MSize getOriginalSize(void) const { return m_osize; }
        
        //! gets the original size of the given dimension (this is the size without considering any ROI)
        /*!
         \param index is the specific zero-based dimension-index whose size is requested
         \return size or -1 if index is out of boundaries
         */
        inline int getOriginalSize(int index) const
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

        //! returns a normalized step in the index-th axis, this is the number of values one has to walk in order to get the next value in the index-th axis.
        /*!
            Please consider, that this value can only be used for pointer-arithmetic operations if the dataObject is continuous.
            Else, it only indicates the number of values, however their pixel position might be interrupted at plane boundaries.

            \param index is the axis for which the step size should be determined
            \raises Exception if index is out of bounds
        */
        int getStep(int index) const;

        
        //! gets total number of elements within the data object's ROI
        /*!
         \return number of elements
         \sa getDims, getSize
         */
        inline int getTotal() const
        {
            int dims = getDims();
            int total = dims > 0 ? 1 : 0;
            for(int i = 0 ; i < dims ; i++)
            {
                total *= m_size[i];
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
        
        RetVal copyTo(DataObject &rhs, unsigned char regionOnly = 0) const;   /*!< deeply copies the data of this data object to the given rhs-dataObject. regionOnly defines if only data within the current ROI should be copied or the entire matrix with the current ROI borders. The destination object is newly allocated if its current number od dimensions, type or size of the ROI does not fit. */
        RetVal convertTo(DataObject &rhs, const int type, const double alpha=1, const double beta=0 ) const; /*!< Convertes an array to another data type with optional scaling (alpha * value + beta) */
        
        RetVal setTo(const int8 &value, const DataObject &mask = DataObject());        /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const uint8 &value, const DataObject &mask = DataObject());       /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const int16 &value, const DataObject &mask = DataObject());       /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const uint16 &value, const DataObject &mask = DataObject());      /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const int32 &value, const DataObject &mask = DataObject());       /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const uint32 &value, const DataObject &mask = DataObject());      /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const float32 &value, const DataObject &mask = DataObject());     /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const float64 &value, const DataObject &mask = DataObject());     /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const complex64 &value, const DataObject &mask = DataObject());   /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const complex128 &value, const DataObject &mask = DataObject());  /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        RetVal setTo(const ito::Rgba32 &value, const DataObject &mask = DataObject()); /*!< Sets all or some (if uint8 mask is given) of the array elements to the specified value. */
        
        //! copy all values of this data object to the copyTo data object. The copyTo-data object must be allocated and have the same type and size (of its roi) than this data object. The compared sequence of sizes only contains dimensions whose size is bigger than one (e.g. it is possible to copy a 5x1 object to a 1x1x5 object)
        RetVal deepCopyPartial(DataObject &copyTo);
        
        //! Returns the matrix iterator and sets it to the first matrix element.
        DObjIterator begin();
        //! Returns the matrix iterator and sets it to the after-last matrix element.
        DObjIterator end();
        
        //! Returns the matrix read-only iterator and sets it to the first matrix element.
        DObjConstIterator constBegin() const;
        
        //! Returns the matrix read-only iterator and sets it to the after-last matrix element.
        DObjConstIterator constEnd() const;
        
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
        DataObject & operator += (const complex128 &value);
        
        DataObject operator + (const DataObject &rhs);
        DataObject operator + (const float64 &value);
        DataObject operator + (const complex128 &value);
        
        DataObject & operator -= (const DataObject &rhs);
        DataObject & operator -= (const float64 &value);
        DataObject & operator -= (const complex128 &value);
        
        DataObject operator - (const DataObject &rhs);
        DataObject operator - (const float64 &value);
        DataObject operator - (const complex128 &value);
        
        DataObject & operator *= (const DataObject &rhs);
        DataObject & operator *= (const float64 &factor);
        DataObject & operator *= (const complex128 &factor);
        
        DataObject operator * (const DataObject &rhs);
        DataObject operator * (const float64 &factor);
        DataObject operator * (const complex128 &factor);
        
        // Comparison Operators
        DataObject operator < (DataObject &rhs);
        DataObject operator > (DataObject &rhs);
        DataObject operator <= (DataObject &rhs);
        DataObject operator >= (DataObject &rhs);
        DataObject operator == (DataObject &rhs);
        DataObject operator != (DataObject &rhs);
        
        DataObject operator < (const float64 &value);
        DataObject operator > (const float64 &value);
        DataObject operator <= (const float64 &value);
        DataObject operator >= (const float64 &value);
        DataObject operator == (const float64 &value);
        DataObject operator != (const float64 &value);
        
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
        DataObject bitwise_not() const; /*!< All other types will raise an exception. Compute bit-wise and element-wise inversion. For signed integer inputs, the two's complement is returned. For floating-point objects, its machine-specific bit representation is used for the operation*/
        
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
        
        // element-wise multiplication 
        DataObject mul(const DataObject &mat2, const double scale = 1.0) const;
        DataObject div(const DataObject &mat2, const double scale = 1.0) const;

        

		// power (power of 0.5 is the square root)
		DataObject pow(const ito::float64 &power); // returns a new data object with the same size and type than this data object and calculates src**power if power is an integer, else |src|**power (only for float32 and float64 data objects)
		void pow(const ito::float64 &power, DataObject &dst); // this function raises every element of this data object to *power* and saves the result in dst. Dst must be of the same size and type than this data object or empty. In the latter case, it is reassigned to the right size and type.

		DataObject sqrt(); // returns a new data object of the same size and type than this data object where the square root of every element is calculated. Is the same than pow(0.5)
		void sqrt(DataObject &dst); // this function calculates the square root of every element and saves the result in dst. Dst must be of the same size and type than this data object or empty. In the latter case, it is reassigned to the right size and type.. Is the same than pow(0.5, dst)

        DataObject squeeze() const;
        DataObject reshape(int newDims, const int *newSizes) const;

        int elemSize() const;  /*!< number of bytes that are required by each value inside of the data object array (e.g. 1 for uint8, 2 for int16...) */
        
        //! addressing method for two-dimensional data object.
        /*!
         \param y is the zero-based row-index to the element which is requested (considering any ROI)
         \param x is the zero-based column-index to the element which is requested (considering any ROI)
         \return const reference to specific element
         */
        template<typename _Tp> inline const _Tp& at(const unsigned int y, const unsigned int x) const
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
            return (*reinterpret_cast<const cv::Mat_<_Tp>*>(m_data[0]))(y, x);
        }
        
        //! addressing method for two-dimensional data object.
        /*!
         \param y is the zero-based row-index to the element which is requested (considering any ROI)
         \param x is the zero-based column-index to the element which is requested (considering any ROI)
         \return reference to specific element
         */
        template<typename _Tp> inline _Tp& at(const unsigned int y, const unsigned int x)
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
            return (*reinterpret_cast<cv::Mat_<_Tp>*>(m_data[0]))(y, x);
        }
        
        //! addressing method for three-dimensional data object.
        /*!
         \param z is the zero-based z-index to the element which is requested (considering any ROI)
         \param y is the zero-based row-index to the element which is requested (considering any ROI)
         \param x is the zero-based column-index to the element which is requested (considering any ROI)
         \return const reference to specific element
         */
        template<typename _Tp> inline const _Tp& at(const unsigned int z, const unsigned int y, const unsigned int x) const
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
            return (*reinterpret_cast<const cv::Mat_<_Tp>*>(m_data[z + m_roi[0]]))(y, x);
        }
        
        //! addressing method for three-dimensional data object.
        /*!
         \param z is the zero-based z-index to the element which is requested (considering any ROI)
         \param y is the zero-based row-index to the element which is requested (considering any ROI)
         \param x is the zero-based column-index to the element which is requested (considering any ROI)
         \return reference to specific element
         */
        template<typename _Tp> inline _Tp& at(const unsigned int z, const unsigned int y, const unsigned int x)
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
            return (*reinterpret_cast<cv::Mat_<_Tp>*>(m_data[z + m_roi[0]]))(y, x);
        }
        
        //! addressing method for n-dimensional data object.
        /*!
         \param *idx is vector whose size is equal to the data object's dimensions. Each entry indicates the zero-based index of its specific dimension considering any ROI
         \remark The idx vector must indicate the indizes in "virtual"-order (user-friendly order)
         \return const reference to specific element
         */
        template<typename _Tp> inline const _Tp& at(const unsigned int *idx) const //idx is in virtual order
        {
            int matNum = 0;
            matIdxToNum(idx, &matNum);
            return (*reinterpret_cast<const cv::Mat_<_Tp>*>(m_data[matNum]))(idx[m_dims - 2], idx[m_dims - 1]);
        }
        
        //! addressing method for n-dimensional data object.
        /*!
         \param *idx is vector whose size is equal to the data object's dimensions. Each entry indicates the zero-based index of its specific dimension considering any ROI
         \remark The idx vector must indicate the indizes in "virtual"-order (user-friendly order)
         \return reference to specific element
         */
        template<typename _Tp> inline _Tp& at(const unsigned int *idx) //idx is in virtual order
        {
            int matNum = 0;
            matIdxToNum(idx, &matNum);
            return (*reinterpret_cast<cv::Mat_<_Tp>*>(m_data[matNum]))(idx[m_dims - 2], idx[m_dims - 1]);
        }
        
        DataObject at(const ito::Range &rowRange, const ito::Range &colRange) const;    /*!< addressing method for two-dimensional data object with two given range-values. returns shallow copy of addressed regions */
        DataObject at(ito::Range *ranges) const;                                        /*!< addressing method for n-dimensional data object with n given range-values. returns shallow copy of addressed regions */
        DataObject at(const DataObject &mask) const;                                    /*!< addressing method that returns a Mx1 data object of the same type than this object with only values that are marked in the given uint8 mask object */
        
        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
         cast this pointer to the data type of the matrix elements (as pointer).
         
         \remark No further error checking (e.g. boundaries)
         \return data-pointer
         */
        inline uchar* rowPtr(const int matNum, const int y)
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
        inline const uchar* rowPtr(const int matNum, const int y) const
        {
            int matIndex = seekMat(matNum);
            return ((const cv::Mat*)m_data[matIndex])->ptr(y);
        }

        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
         This is a templated version to return the pointer already casted to the right type,
         e.g. ito::float64* myPtr = myObj->rowPtr<ito::float64>(0,0).
         
         \remark No further error checking (e.g. boundaries)
         \return data-pointer
         */
        template<typename _Tp> inline _Tp* rowPtr(const int matNum, const int y)
        {
            int matIndex = seekMat(matNum);
            return ((cv::Mat*)m_data[matIndex])->ptr<_Tp>(y);
        }
        
        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
         This is a templated version to return the pointer already casted to the right type,
         e.g. const ito::float64* myPtr = myObj->rowPtr<ito::float64>(0,0).
         
         \remark No further error checking (e.g. boundaries)
         \return data-pointer
         */
        template<typename _Tp> inline const _Tp* rowPtr(const int matNum, const int y) const
        {
            int matIndex = seekMat(matNum);
            return ((const cv::Mat*)m_data[matIndex])->ptr<_Tp>(y);
        }
        
        DataObject row(const int selRow) const;
        DataObject col(const int selCol) const;
        
        DataObject toGray(const int destinationType = ito::tUInt8) const;
        DataObject splitColor(const char* destinationColor, const int& dtype) const;
		DataObject lineCut(const double* coordinates, const int& len) const;
        
        // ROI
        DataObject & adjustROI(const int dtop, const int dbottom, const int dleft, const int dright);   /*!< changes the boundaries of the ROI of a two-dimensional data object by the given incremental values */
        DataObject & adjustROI(const unsigned char dims, const int *lims);                              /*!< changes the boundaries of the ROI of a n-dimensional data object by the given incremental values */
        RetVal locateROI(int *wholeSizes, int *offsets) const;                                          /*!< locates the boundaries of the ROI of a n-dimensional data object and returns the original size and the distances to the physical borders */
        RetVal locateROI(int *lims) const;                                                              /*!< locates the boundaries of the ROI of a n-dimensional data object the distances to the physical borders */
        
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
        
        template<typename T2> operator T2 ();  /*!< cast operator, tries to cast this data object to another element type */
        
        template<typename _Tp> RetVal linspace(const _Tp start, const _Tp end, const _Tp inc, const int transposed);

		//! returns a stack of multiple dataObjects (number is equal to num) along the given axis (default: 0). 
		/*! The axis is always mapped to the object with the largest number of dimensions ndim_max. 
		    All other dataObjects are considered to also have ndim_max dimensions, where additional
			dimensions are prepended having a size of 1.
		*/
		static DataObject stack(const DataObject *mats, int num, unsigned int axis = 0);
        
    };
    
    //template<> DATAOBJ_EXPORT RetVal ito::DataObject::linspace<ito::int8>(const ito::int8 /*start*/, const ito::int8 /*end*/, const ito::int8 /*inc*/, const int /*transposed*/);
    //template<> DATAOBJ_EXPORT RetVal ito::DataObject::linspace<ito::uint8>(const ito::uint8 /*start*/, const ito::uint8 /*end*/, const ito::uint8 /*inc*/, const int /*transposed*/);
    
    
    //----------------------------------------------------------------------------------------------------------------------------------
    // functions for DataObject in namespace ITO, which are NOT member functions
    //----------------------------------------------------------------------------------------------------------------------------------
    DATAOBJ_EXPORT DataObject abs(const DataObject &dObj);              /*!< calculates the absolute values of each element in the given data object and returns the result as new data object */
    DATAOBJ_EXPORT DataObject arg(const DataObject &dObj);              /*!< calculates the argument of each element in the given data object and returns the result as new data object */
    DATAOBJ_EXPORT DataObject real(const DataObject &dObj);             /*!< calculates the real part of each element in the given data object and returns the result as new data object */
    DATAOBJ_EXPORT DataObject imag(const DataObject &dObj);             /*!< calculates the imaginary part of each element in the given data object and returns the result as new data object */
    
    DATAOBJ_EXPORT DataObject makeContinuous(const DataObject &dObj);   /*!< if the given data object is not continuously organized, copies the content to a new continuous data object */
    
    //! templated method for converting a given scalar value to the data type, indicated by the template parameter
    /*!
     \param fromType is the data type of the given scalar
     \param *scalar is the pointer to the scalar value, casted to void*
     \return the converted scalar value
     \throws cv::Exception if the input data type is unknown or if the conversion failed
     \sa saturate_cast
     */
    template<typename _Tp> inline _Tp numberConversion(ito::tDataType fromType, const void *scalar)
    {
        _Tp retValue;
        
        switch(fromType)
        {
            case ito::tUInt8:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const uint8*>(scalar)));
                break;
            case ito::tInt8:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const int8*>(scalar)));
                break;
            case ito::tUInt16:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const uint16*>(scalar)));
                break;
            case ito::tInt16:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const int16*>(scalar)));
                break;
            case ito::tUInt32:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const uint32*>(scalar)));
                break;
            case ito::tInt32:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const int32*>(scalar)));
                break;
            case ito::tFloat32:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const ito::float32*>(scalar)));
                break;
            case ito::tFloat64:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const ito::float64*>(scalar)));
                break;
            case ito::tComplex64:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const ito::complex64*>(scalar)));
                break;
            case ito::tComplex128:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const ito::complex128*>(scalar)));
                break;
            case ito::tRGBA32:
                retValue = cv::saturate_cast<_Tp>(*(static_cast<const ito::Rgba32*>(scalar)));
                break;
            default:
                cv::error(cv::Exception(CV_StsAssert, "Input value type unkown", "", __FILE__, __LINE__));
                retValue = 0;
        }
        
        return retValue;
    };    
    
    //! streaming operator to stream the representation or contant of a data object
    DATAOBJ_EXPORT std::ostream& operator << (std::ostream& out, const DataObject& dObj);
    
    //! method which returns the real data type of any given data type
    /*!
     If the given data type is already real, the same type is returned. Else the type of the real argument of the given complex type is returned.
     
     \param cmplxType is the input data type
     \return see method's description
     \throws cv::Exception if the input data type is unknown
     */
    inline ito::tDataType convertCmplxTypeToRealType(ito::tDataType cmplxType)
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
    
    //! method which guesses the dataObject type from a given cv::Mat*
    /*!
     If the given data type is already real, the same type is returned. Else the type of the real argument of the given complex type is returned.
     
     \param mat is the OpenCV matrix.
     \param retval an error value will be added if the type cannot be converted.
     \return ito::DataObject type that fits to the given matrix
     */
    inline ito::tDataType guessDataTypeFromCVMat(const cv::Mat* mat, ito::RetVal &retval)
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
    
    
    //! method which returns the value of enumeration ito::tDataType, which corresponds to the type of the given pointer parameter.
    /*!
     If the parameter type cannot be transformed into a value of ito::tDataType, an exception is thrown.
     
     \param any pointer, whose type should be transformed
     \return ito::tDataType
     \throws cv::Exception if the input data type is unknown
     \sa getDataType2
     */
    template<typename _Tp> inline ito::tDataType getDataType(const _Tp* /*src*/)
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
    
    
    //! method which returns the value of enumeration ito::tDataType, which corresponds to the template parameter (must be a pointer).
    /*!
     If the template parameter cannot be transformed into a value of ito::tDataType, an exception is thrown.
     
     Call is ito::tDataType result = getDataType2<uint8*>().
     
     \return ito::tDataType
     \throws cv::Exception if the template parameter is unknown (e.g. no pointer).
     \sa getDataType
     */
    template<typename _Tp> inline ito::tDataType getDataType2()
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
    template<> inline ito::tDataType getDataType2<Rgba32*>()     { return ito::tRGBA32; }  
    
} //namespace ito

#endif //__DATAOBJH


