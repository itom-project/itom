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

//#include <crtdbg.h>
#include <cstdlib>
#include <iostream>
#include <complex>
#include <limits>

#if !linux
    #pragma warning(disable:4996)
#endif

#define NOMINMAX //see: http://social.msdn.microsoft.com/Forums/sv/vclanguage/thread/d986a370-d856-4f9e-9f14-53f3b18ab63e, this is only an issue with OpenCV 2.4.3, not 2.3.x

#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

#include "../common/sharedStructures.h"

#include "readWriteLock.h"

#include <vector>
#include <map>
#include <string>

namespace cv 
{
   template<> inline ito::float32 saturate_cast<ito::float32>( ito::float64 v)
   {
       //return (float32)v;
       if(cvIsInf(v)) return std::numeric_limits<ito::float32>::infinity();
       if(cvIsNaN(v)) return std::numeric_limits<ito::float32>::signaling_NaN();
       return static_cast<ito::float32>(std::max ( (ito::float64)(- std::numeric_limits<ito::float32>::max()) ,  std::min ( v , (ito::float64) std::numeric_limits<ito::float32>::max() )));
   }

   template<> inline ito::float64 saturate_cast<ito::float64>( ito::float32 v)
   {
       //return (float64)v;
       if(cvIsInf(v)) return std::numeric_limits<ito::float64>::infinity();
       if(cvIsNaN(v)) return std::numeric_limits<ito::float64>::signaling_NaN();
       return static_cast<ito::float64>(v);
   }

   template<typename _Tp> static inline _Tp saturate_cast(ito::complex128 /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
   template<typename _Tp> static inline _Tp saturate_cast(ito::complex64 /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
   
   template<typename _Tp> static inline _Tp saturate_cast(ito::rgba32 /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
   //template<typename _Tp> static inline ito::rgba32 saturate_cast(_Tp /*v*/) {     cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__)); return 0; }
 
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
   
   //template<> inline ito::rgba32 saturate_cast(ito::int8 v) {return saturate_cast<ito::uint8>(v);}
   template<> inline ito::rgba32 saturate_cast(ito::uint8 v) {return v;}
   template<> inline ito::rgba32 saturate_cast(ito::uint16 v){return saturate_cast<ito::uint8>(v);}
   //template<> inline ito::rgba32 saturate_cast(ito::int16 v){return saturate_cast<ito::uint8>(v);}
   template<> inline ito::rgba32 saturate_cast(ito::uint32 v){return v;}
   template<> inline ito::rgba32 saturate_cast(ito::int32 v){return (ito::uint32)v;}
   template<> inline ito::rgba32 saturate_cast(ito::float32 v){return saturate_cast<ito::uint8>(v);}
   template<> inline ito::rgba32 saturate_cast(ito::float64 v){return saturate_cast<ito::uint8>(v);}
   template<> inline ito::rgba32 saturate_cast(ito::rgba32 v){return v;}


   template<> inline ito::uint8 saturate_cast(ito::rgba32 v){return saturate_cast<ito::uint8>(v.gray());};
   template<> inline ito::uint16 saturate_cast(ito::rgba32 v){return saturate_cast<ito::uint16>(v.gray());};
   template<> inline ito::uint32 saturate_cast(ito::rgba32 v){return v.argb();};
   template<> inline ito::int32 saturate_cast(ito::rgba32 v){return (ito::int32)(v.argb());};
   template<> inline ito::float32 saturate_cast(ito::rgba32 v){return (ito::float32)v.gray();};
   template<> inline ito::float64 saturate_cast(ito::rgba32 v){return (ito::float64)v.gray();};


    template<> class cv::DataType<ito::rgba32>
    {
        public:
        typedef ito::rgba32 value_type;
        typedef ito::uint32 work_type;
        typedef ito::uint8 channel_type;
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

    template<> class cv::DataType<ito::redChannel>
    {
        public:
        typedef ito::redChannel value_type;
        typedef ito::uint32 work_type;
        typedef ito::uint8 channel_type;
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

    template<> class cv::DataType<ito::greenChannel>
    {
        public:
        typedef ito::greenChannel value_type;
        typedef ito::uint32 work_type;
        typedef ito::uint8 channel_type;
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

    template<> class cv::DataType<ito::blueChannel>
    {
        public:
        typedef ito::blueChannel value_type;
        typedef ito::uint32 work_type;
        typedef ito::uint8 channel_type;
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

    template<> class cv::DataType<ito::alphaChannel>
    {
        public:
        typedef ito::alphaChannel value_type;
        typedef ito::uint32 work_type;
        typedef ito::uint8 channel_type;
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

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class Range
    \brief each range value has a start and end point. Optionally range can be marked as Range::all(), which indicates a full range
*/
class Range
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
class DataObjectTagType
{
    private:
        double m_dVal;
        std::string m_strValue;
        int m_type;        //!< parameter type, maybe int, char, double or pointer

    public:
        //!< Constructor
        DataObjectTagType() : m_dVal(0), m_strValue(""), m_type(DataObjectTagType::typeInvalid){};
        DataObjectTagType(double dVal) : m_dVal(dVal), m_strValue(""), m_type(DataObjectTagType::typeDouble){};
        DataObjectTagType(std::string strVal) : m_dVal(std::numeric_limits<double>::signaling_NaN()), m_strValue(strVal), m_type(DataObjectTagType::typeString){};
        DataObjectTagType(const char* cVal) : m_dVal(std::numeric_limits<double>::signaling_NaN()), m_type(DataObjectTagType::typeString){ cVal == NULL ?  m_strValue = "" : m_strValue = cVal;};
        //!< Copy Constructor
        DataObjectTagType(const DataObjectTagType& copyConstr) : m_dVal(copyConstr.m_dVal), m_strValue(copyConstr.m_strValue), m_type(copyConstr.m_type){};

        DataObjectTagType & operator = (const DataObjectTagType rhs)
        {
            this->m_dVal = rhs.m_dVal;
            this->m_strValue = rhs.m_strValue;
            this->m_type = rhs.m_type;

            return *this;
        }

        enum tTagType
        {
            typeInvalid     = 0x000000,
            typeDouble      = 0x000008,
            typeString      = 0x000020
        };

        inline int getType(void) const {return m_type;};
        inline bool isValid(void) const { return (m_type == DataObjectTagType::typeInvalid) ? false: true;};

        /** getVal_ToDouble  read parameter value and try to convert to double
        *   @return parameter value (numeric, casted) or signaling_NaN()
        *
        *   returns the actual parameter value as double. If conversion failes it returns a signaling_NaN()
        */
        inline double getVal_ToDouble(void)
        {
            if(m_type == DataObjectTagType::typeInvalid)
            {
                return std::numeric_limits<double>::signaling_NaN();
            }
            else if(m_type == DataObjectTagType::typeDouble)
            {
                return m_dVal;
            }
            else
            {
                double dVal = std::numeric_limits<double>::signaling_NaN();
                //dVal = atof(m_strValue.c_str()); //sometimes the result of that line has been arbitrary, therefore this conversion should fail.
                return dVal;
            }
        }

        /** getVal_ToString  read parameter value and try to convert to std::string
        *   @return parameter value (numeric, casted) or 'NaN' || 'Inf'
        *
        *   returns the actual parameter value as std::string. If conversion from double failes it returns 'NaN' || 'Inf'
        */
        inline std::string getVal_ToString(void)
        {
            if(m_type == DataObjectTagType::typeInvalid)
            {
                return std::string();
            }
            else if(m_type == DataObjectTagType::typeString)
            {
                return m_strValue;
            }
            else
            {
				if(cvIsNaN(m_dVal)) return std::string("NaN");
				if(cvIsInf(m_dVal)) return std::string("Inf");
                /*if(m_dVal == std::numeric_limits<double>::quiet_NaN()) return std::string("NaN");
                if(m_dVal == std::numeric_limits<double>::signaling_NaN()) return std::string("NaN");
                if(m_dVal == std::numeric_limits<double>::infinity()) return std::string("Inf");*/

                std::ostringstream strs;
                strs << m_dVal;

                return strs.str();
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
class DataObjectTags
{
private:
    //std::map<std::string,std::string> m_tags;   /*!< map for tags with keyword (std::string) and value (std::string) */
    std::map<std::string, DataObjectTagType> m_tags;   /*!< map for tags with keyword (std::string) and value (either std::string or double) */

    std::vector<double> m_axisOffsets;          /*!< vector with offset-values for each axis (offset in dataObject-Pixel). Describes the distance from pixel [0,0,..0] to coordiante system origin. Unit-Coordinate = ( px-Coordinate - Offset)* Scale*/
    std::vector<double> m_axisScales;           /*!< vector with scale-values for each axis (unit / px). Unit-Coordinate = ( px-Coordinate - Offset)* Scale. Scale cannot be 0.0*/
    std::vector<std::string> m_axisDescription; /*!< vector with axis-describtions */
    std::vector<std::string> m_axisUnit;        /*!< vector with axis-units-description (e.g. 'mm') */
    double m_valueOffset;                       /*!< offset of the values within the dataObject. Currently as read only with value 0.0 */
    double m_valueScale;                        /*!< scale of the values within the dataObject. Currently as read only with value 1.0 */
    std::string m_valueDescription;             /*!< descriptions for the values (e.g. 'Intensity' or 'Heigth') */
    std::string m_valueUnit;                    /*!< unit description for the values (e.g. 'mm') */

    double m_rotMatrix[9];                      /*!< array containing the rotiational matrix for the yx-plane */

    friend class DataObject;


public:
    //!< Constructor
    DataObjectTags(unsigned int totalAxisNum) : m_valueOffset(0.0), m_valueScale(1.0), m_valueDescription(""), m_valueUnit("")
    {
        m_tags.clear();
        m_axisOffsets.resize(totalAxisNum, 0.0);
        m_axisScales.resize(totalAxisNum, 1.0);
        m_axisDescription.resize(totalAxisNum, "");
        m_axisUnit.resize(totalAxisNum, "");
        m_valueUnit = std::string();
        memset(m_rotMatrix, 0, sizeof(double)*9);
        m_rotMatrix[0] = 1; // r11
        m_rotMatrix[4] = 1; // r22
        m_rotMatrix[8] = 1; // r33
    }

    //!< Destructor
    ~DataObjectTags()
    {
        m_tags.clear();
        m_axisOffsets.clear();
        m_axisScales.clear();
        m_axisDescription.clear();
        m_axisUnit.clear();
    }

    //!< Copy constructor
    DataObjectTags(const DataObjectTags& copyConstr)
    {
        m_tags = copyConstr.m_tags;
        m_axisOffsets = copyConstr.m_axisOffsets;
        m_axisScales = copyConstr.m_axisScales;
        m_axisDescription = copyConstr.m_axisDescription;
        m_axisUnit = copyConstr.m_axisUnit;
        m_valueOffset = copyConstr.m_valueOffset;
        m_valueScale = copyConstr.m_valueScale;
        m_valueDescription = copyConstr.m_valueDescription;
        m_valueUnit = copyConstr.m_valueUnit;
        memcpy(m_rotMatrix,copyConstr.m_rotMatrix, sizeof(double)*9);
    }

    //friend class DataObjectTags; //I am my best friend (my own and only friend, and therefore my copy-constr has access to my friends members. nice.)

};

//----------------------------------------------------------------------------------------------------------------------------------

class DObjConstIterator
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
    size_t elemSize;
    uchar* ptr;
    uchar* sliceStart;
    uchar* sliceEnd;
    int plane;
};

//----------------------------------------------------------------------------------------------------------------------------------
class DObjIterator : public DObjConstIterator
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
class DataObject
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
        void createHeader(const unsigned char dimensions, const size_t *sizes, const size_t *steps, const size_t elemSize)
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

                        m_roi.m_p = new size_t [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                        m_osize.m_p = static_cast<size_t *>(m_roi.m_p + dimensions) + 1;
                        m_osize.m_p[-1] = dimensions;
                        m_size.m_p = static_cast<size_t *>(m_osize.m_p + dimensions) + 1;
                        m_size.m_p[-1] = dimensions;
                        m_roi.m_p = m_roi.m_p + 1;
                        m_roi.m_p[-1] = dimensions;
                    }
                }
                else
                {
                    m_roi.m_p = new size_t [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                    m_osize.m_p = static_cast<size_t *>(m_roi.m_p + dimensions) + 1;
                    m_osize.m_p[-1] = dimensions;
                    m_size.m_p = static_cast<size_t *>(m_osize.m_p + dimensions) + 1;
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
                        m_osize.m_p[n] = steps[n-1] / elemSize;
                    }
                    else /*if(n < dimensions -1)*/
                    {
                        m_osize.m_p[n] = steps[n-1] / steps[n];
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
        void createHeaderWithROI(const unsigned char dimensions, const size_t *sizes, const size_t *osizes = NULL, const size_t *roi = NULL)
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

                        m_roi.m_p = new size_t [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                        m_osize.m_p = static_cast<size_t *>(m_roi.m_p + dimensions) + 1;
                        m_osize.m_p[-1] = dimensions;
                        m_size.m_p = static_cast<size_t *>(m_osize.m_p + dimensions) + 1;
                        m_size.m_p[-1] = dimensions;
                        m_roi.m_p = m_roi.m_p + 1; //move m_p pointer by one
                        m_roi.m_p[-1] = dimensions;
                    }
                }
                else
                {
                    m_roi.m_p = new size_t [(dimensions + 1) + (dimensions + 1) + (dimensions + 1)];
                    m_osize.m_p = static_cast<size_t *>(m_roi.m_p + dimensions) + 1;
                    m_osize.m_p[-1] = dimensions;
                    m_size.m_p = static_cast<size_t *>(m_osize.m_p + dimensions) + 1;
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

        void create(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous, const uchar* continuousDataPtr = NULL, const size_t* steps = NULL);  /*!< allocates new data */

        void create(const unsigned char dimensions, const size_t *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes);

        void freeData(void);     /*!< decrements reference counter and deletes data, if no other instance is using them (ref counter < 0) */

        void secureFreeData(void); /*!< decrements reference counter and deletes data, if no other instance is using them (ref counter < 0). This method makes a lot of security checks instead of freeFunc. */



        //----------------------------------------------------------------------------------------------------------------------------------
        ito::RetVal matNumToIdx(const size_t matNum, size_t *matIdx) const;

        //! brief calculates the index of the matrix-plane in the m_data-vector for a given vector of indices, which address one element in the n-dimensional matrix
        /*!
            The matrix indices are zero-based and consider the ROI of this data object.

            \param *matIdx is a vector containing indices which address one element in the n-dimensional matrix
            \param *matNum is a pointer, where the resulting matrix-plane-index is written.
            \return retOk
            \throws cv::Exception if the given indices are out of bounds
        */
        inline ito::RetVal matIdxToNum(const unsigned int *matIdx, size_t *matNum) const
        {
           *matNum = 0;
           if (m_dims <= 2)
           {
                 return 0;
           }

           size_t planeSize = 1;

           for (int n = m_dims - 3; n >= 0; n--)
           {
        #if __ITODEBUG
                 if (((size_t)matIdx[n] + m_roi[n]) >= m_osize[n])
                 {
                    cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
                 }
        #endif
                 (*matNum) += ((size_t)matIdx[n] + m_roi[n]) * planeSize; //CAST_TODO
                 planeSize *= m_osize[n];
           }

           return 0;
        }

        struct MSize
        {
            inline MSize() : m_p(NULL) {}
            //inline MSize(size_t *_p, char *_transp) : m_p(_p) {}
//            Size operator()() const;
            inline size_t operator [] (const int dim) const
            {
                return m_p[dim]; //return the size value. this operator corresponds to the real data representation in memory
            };
//            inline unsigned int& operator [](const int i) { return m_p[i]; };
            inline operator const size_t * () const { return m_p; }
            bool operator == (const MSize& sz) const
            {
                if(m_p == NULL || sz.m_p == NULL)
                {
                    return sz.m_p == m_p;
                }

                size_t d = m_p[-1], dsz = sz.m_p[-1];
                if( d != dsz )
                    return false;
                if( d == 2 )
                {                    
                    return m_p[0] == sz.m_p[0] && m_p[1] == sz.m_p[1];            
                }

                for( size_t i = 0; i < d - 2; i++ )
                {
                    if( m_p[i] != sz.m_p[i] )
                    {
                        return false;
                    }
                }                
                return (m_p[d - 2] == sz.m_p[d - 2]) && (m_p[d - 1] == sz.m_p[d - 1]);               
            }

            inline bool operator != (const MSize& sz) const { return !(*this == sz); }

            size_t *m_p;
        };

        struct MROI
        {
            inline MROI() : m_p(NULL) {};
            //inline MROI(size_t *_p, char *_transp) : m_p(_p), m_pTransp(_transp) {}
            inline size_t operator [] (const int dim) const
            {
                return m_p[dim]; //return the size value. this operator corresponds to the real data representation in memory
            }

//            inline unsigned int operator () (unsigned char dim, unsigned char beginEnd) const { beginEnd > 0 ? beginEnd = 1 : beginEnd = 0; return m_p[dim + beginEnd]; };
//            inline unsigned int & operator () (unsigned char dim, unsigned char beginEnd) { return m_p[dim + beginEnd]; };
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

                size_t d = m_p[-1];
                for (size_t n = 0; n < d - 2; n++)
                {
                    if (m_p[n] != rroi.m_p[n])
                    {
                        return false;
                    }
                }

                return m_p[d - 2] == rroi.m_p[d - 2] && m_p[d - 1] == rroi.m_p[d - 1];
                
            }

            size_t *m_p;
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
//        std::vector<int *>  m_data;                  /*!< vector with references to each matrix-plane. Must be cast to cv::Mat_<"m_type"> */
        int     **m_data;
        ReadWriteLock      *m_objSharedDataLock;     /*!< readWriteLock for data block, this lock is shared within every instance which is using the same data. */
        DataObjectTags     *m_pDataObjectTags;        /*!< class containing the object metadata */
        ReadWriteLock       m_objHeaderLock;         /*!< readWriteLock for this instance of dataObject. */

        inline int mdata_realloc(const size_t size)
        {
            static size_t sizeofs = sizeof(size_t) / sizeof(int *);

            if (m_data)
            {
                m_data = static_cast<int **>(realloc(m_data - sizeofs, size * sizeof(int *) + sizeof(size_t)));
            }
            else
            {
                size_t numBytes = size * sizeof(int *) + sizeof(size_t);
                m_data = static_cast<int **>(calloc(numBytes, 1));
                memset( m_data, 0, numBytes );
            }
            (*reinterpret_cast<size_t*>(m_data)) = size;
            m_data += sizeofs;
            return 0;
        }
        inline size_t mdata_size(void) const
        {
            static size_t sizeofs = sizeof(size_t) / sizeof(int *);
            if (!m_data)
                return 0;
            else
                return (*reinterpret_cast<size_t *>(m_data - sizeofs));
        }
        inline int mdata_free()
        {
            static size_t sizeofs = sizeof(size_t) / sizeof(int *);
            if (m_data)
            {
                int **ptr = m_data - sizeofs;
                free(ptr);
            }
            m_data = NULL;
            return 0;
        }

        //low-level, templated methods
        //most low-level methods are marked "friend" such that they can access private members of their data object parameters
        template<typename _Tp> friend RetVal CreateFunc(DataObject *dObj, const unsigned char dimensions, const size_t *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const size_t* steps);
        template<typename _Tp> friend RetVal CreateFuncWithCVPlanes(DataObject *dObj, const unsigned char dimensions, const size_t *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes);
        template<typename _Tp> friend RetVal FreeFunc(DataObject *dObj);
        template<typename _Tp> friend RetVal SecureFreeFunc(DataObject *dObj);
        template<typename _Tp> friend RetVal CopyToFunc(const DataObject &lhs, DataObject &rhs, unsigned char regionOnly);
        template<typename _Tp> friend RetVal ConvertToFunc(const DataObject &lhs, DataObject &rhs, const int type, const double alpha, const double beta);
        template<typename _Tp> friend RetVal AdjustROIFunc(DataObject *dObj, const int *lims);
        template<typename _Tp> friend RetVal MinMaxLocFunc(const DataObject &dObj, double *minVal, double *maxVal, size_t *minPos, size_t *maxPos);
        template<typename _Tp> friend RetVal AssignScalarFunc(const DataObject *src, const ito::tDataType type, const void *scalar);
        template<typename _Tp> friend RetVal MakeContinuousFunc(const DataObject &dObj, DataObject &resDObj);
        template<typename _Tp> friend RetVal EvaluateTransposeFlagFunc(DataObject *dObj);
        template<typename _Tp> friend RetVal CalcMinMaxValues(DataObject *lhs, double &result_min, double &result_max, const int cmplxSel = 0);

        // more friends due to change of std::vector to int ** for m_data ...
        template<typename _Tp> friend RetVal GetRangeFunc(DataObject *dObj, const int dtop, const int dbottom, const int dleft, const int dright);
        template<typename _Tp> friend RetVal AdjustROIFunc(DataObject *dObj, int dtop, int dbottom, int dleft, int dright);

    public:
        size_t seekMat(const size_t matNum, const size_t numMats) const;
        size_t seekMat(const size_t matNum) const;
        size_t calcNumMats(void) const;

        // TAGSPACEFUNCTIONS

        //!< Function return the offset of the values stored within the dataOject
        inline double getValueOffset() const
        {
            if(!m_pDataObjectTags) return 0.0; // default
            return m_pDataObjectTags->m_valueOffset;
        }

        //!< Function return the scaling of values stored within the dataOject
        inline double getValueScale() const
        {
            if(!m_pDataObjectTags) return 1.0; // default
            return m_pDataObjectTags->m_valueScale;
        }

        //!< Function return the unit description for the values stoerd within the dataOject
        inline const std::string getValueUnit() const
        {
            if(!m_pDataObjectTags) return std::string(); //default
            return m_pDataObjectTags->m_valueUnit;
        }

        //!< Function return the description for the values stored within the dataOject, if tagspace does not exist, NULL is returned.
        inline std::string getValueDescription() const
        {
            if(!m_pDataObjectTags) return std::string(); //default
            return m_pDataObjectTags->m_valueDescription;
        }

        //!< Function return the axis-offset for the existing axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
        inline double getAxisOffset(const int axisNum) const
        {
            if(axisNum < 0 || axisNum >= m_dims)
            {
                cv::error(cv::Exception(CV_StsError, "Parameter axisNum out of range." ,"", __FILE__, __LINE__));
            }
            if(!m_pDataObjectTags) return 0.0; // default
           
            return m_pDataObjectTags->m_axisOffsets[axisNum] - m_roi[axisNum];
        }

        //!< Function returns the axis-description for the exist axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
        inline double getAxisScale(const int axisNum) const
        {
            if(axisNum < 0 || axisNum >= m_dims)
            {
                cv::error(cv::Exception(CV_StsError, "Parameter axisNum out of range." ,"", __FILE__, __LINE__));
            }
            if(!m_pDataObjectTags) return 1.0; // default

            return m_pDataObjectTags->m_axisScales[axisNum];
        }

        //!< Function returns the axis-unit-description for the exist axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
        inline const std::string getAxisUnit(const int axisNum, bool &validOperation) const
        {
            if(axisNum < 0 || axisNum >= m_dims)
            {
				validOperation = false;
                cv::error(cv::Exception(CV_StsError, "Parameter axisNum out of range." ,"", __FILE__, __LINE__));
            }
            if(!m_pDataObjectTags)
            {
                validOperation = false;
                return std::string(); //error
            }
            validOperation = true;            
            return m_pDataObjectTags->m_axisUnit[axisNum];
        }

        //!< Function returns the axis-description for the exist specified by axisNum. If axisNum is out of dimension range it returns NULL.
        const inline std::string getAxisDescription(const int axisNum, bool &validOperation) const
        {
            if(axisNum < 0 || axisNum >= m_dims)
            {
				validOperation = false;
                cv::error(cv::Exception(CV_StsError, "Parameter axisNum out of range." ,"", __FILE__, __LINE__));
            }
            if(!m_pDataObjectTags)
            {
                validOperation = false;
                return std::string(); //error
            }

            validOperation = true;          
            return m_pDataObjectTags->m_axisDescription[axisNum];
        }

        inline DataObjectTagType getTag(const std::string key, bool &validOperation) const
        {
            validOperation = false;
            if(!m_pDataObjectTags)
            {
                return DataObjectTagType(); //error
            }
            //std::map<std::string, std::string>::iterator it = m_pDataObjectTags->m_tags.find(key);
            std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find(key);
            if(it != m_pDataObjectTags->m_tags.end())
            {
                validOperation = true;
                return it->second;
            }
            return DataObjectTagType();
        }

        //!<
        //inline bool getTagByIndex(const int tagNumber, std::string &key, std::string &value) const
        inline bool getTagByIndex(const int tagNumber, std::string &key, DataObjectTagType &value) const
        {
            if(!m_pDataObjectTags)
            {
                key = std::string();
                value = std::string();
                return false;
            }

            if((tagNumber < 0) || ((size_t)(tagNumber + 1) > m_pDataObjectTags->m_tags.size()))
            {
                key = std::string();
                value = std::string();
                return false;
            }
            //std::map<std::string,std::string>::iterator it = m_pDataObjectTags->m_tags.begin();
            std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.begin();
            for(int i = 0; i < tagNumber; i++)
            {
                it++;
            }

            key = (*it).first;
            value = (*it).second;
            return true;
        }

        //!<  Function returns the string-value for 'key' identified by int tagNumber. If key in the TagMap do not exist NULL is returned
        inline std::string getTagKey(const int tagNumber, bool &validOperation) const
        {
            if(!m_pDataObjectTags)
            {
                validOperation = false;
                return std::string(""); //error
            }
            if((tagNumber < 0) || ((size_t)(tagNumber + 1) > m_pDataObjectTags->m_tags.size()))
            {
                validOperation = false;
                return std::string(""); //does not exist
            }
            //std::map<std::string,std::string>::iterator it = m_pDataObjectTags->m_tags.begin();
            std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.begin();
            validOperation = true;
            for(int i = 0; i < tagNumber; i++)
            {
                it++;
            }
            return (*it).first;
        }

        //!< Function returns the number of elements in the Tags-Maps
        inline int getTagListSize() const
        {
            if(!m_pDataObjectTags) return 0; //error
            return static_cast<int>(m_pDataObjectTags->m_tags.size());
        }

     //   inline void setValueOffset(double offset) { m_valueOffset =offset; }
     //   inline void setValueScale(double scale) { m_valueScale =scale; }

        //!<  Function to set the string-value of the value unit, return 1 if values does not exist
        inline int setValueUnit(const std::string &unit)
        {
            if(!m_pDataObjectTags) return 1;    //error
            m_pDataObjectTags->m_valueUnit = unit;
            return 0;
        }

        //!<  Function to set the string-value of the value description, return 1 if values does not exist
        inline int setValueDescription(const std::string &description)
        {
            if(!m_pDataObjectTags) return 1;    //error
            m_pDataObjectTags->m_valueDescription = description;
            return 0;
        }

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

                    if(m_pDataObjectTags)
                    {
                        tPxX = physX / m_pDataObjectTags->m_axisScales[0] + m_pDataObjectTags->m_axisOffsets[0];
                    }
                    else
                    {
                        tPxX = physX;
                    }

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
                        tPxX = physX / m_pDataObjectTags->m_axisScales[m_dims - 1] + m_pDataObjectTags->m_axisOffsets[m_dims - 1];
                        tPxY = physY / m_pDataObjectTags->m_axisScales[m_dims - 2] + m_pDataObjectTags->m_axisOffsets[m_dims - 2];
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

        inline RetVal setXYRotationalMatrix(double r11, double r12, double r13, double r21, double r22, double r23, double r31, double r32, double r33)
        {
            if(!m_pDataObjectTags) return RetVal(retError, 0, "Tagspace not initialized"); // error
            m_pDataObjectTags->m_rotMatrix[0] = r11;
            m_pDataObjectTags->m_rotMatrix[1] = r12;
            m_pDataObjectTags->m_rotMatrix[2] = r13;
            m_pDataObjectTags->m_rotMatrix[3] = r21;
            m_pDataObjectTags->m_rotMatrix[4] = r22;
            m_pDataObjectTags->m_rotMatrix[5] = r23;
            m_pDataObjectTags->m_rotMatrix[6] = r31;
            m_pDataObjectTags->m_rotMatrix[7] = r32;
            m_pDataObjectTags->m_rotMatrix[8] = r33;
            return retOk;
        }

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

        inline RetVal getXYRotationalMatrix(double &r11, double &r12, double &r13, double &r21, double &r22, double &r23, double &r31, double &r32, double &r33) const
        {
            if(!m_pDataObjectTags) return RetVal(retError, 0, "Tagspace not initialized"); // error
            r11 = m_pDataObjectTags->m_rotMatrix[0];
            r12 = m_pDataObjectTags->m_rotMatrix[1];
            r13 = m_pDataObjectTags->m_rotMatrix[2];
            r21 = m_pDataObjectTags->m_rotMatrix[3];
            r22 = m_pDataObjectTags->m_rotMatrix[4];
            r23 = m_pDataObjectTags->m_rotMatrix[5];
            r31 = m_pDataObjectTags->m_rotMatrix[6];
            r32 = m_pDataObjectTags->m_rotMatrix[7];
            r33 = m_pDataObjectTags->m_rotMatrix[8];
            return retOk;
        }

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
        inline size_t getTotal() const
        {
            int dims = getDims();
            size_t total = dims > 0 ? 1 : 0;
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
        inline size_t getOriginalTotal() const
        {
            int dims = getDims();
            size_t total = dims > 0 ? 1 : 0;
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



//        std::vector<int *> get_mdata(void);
        int ** get_mdata(void);
//        std::vector<int *> get_mdata(void) const;
        int ** get_mdata(void) const;

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
        size_t getSize(int index) const
        {
            if(index < 0 || index >= m_dims)
            {
                cv::error(cv::Exception(CV_StsAssert, "Requested dimension missmatch with object dimensions or maybe negative", "", __FILE__, __LINE__));
                return 0;
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
        size_t getOriginalSize(int index) const
        {
            if(index < 0 || index >= m_dims) return -1;           
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
        DataObject(const size_t size, const int type): m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            size_t sizes[2] = {1, size};
            this->create(2, sizes, type, 1);
            //DataObject(1, size, type);
        }

        //! constructor for two-dimensional data object. The data is newly allocated and arbitrarily filled.
        /*!
            the owndata-flag is set to true, the continuously-flag, too (since only one matrix-plane will be created)

            \param sizeY is the number of rows in each matrix-plane
            \param sizeX is the number of columns in each matrix-plane
            \param type is the data-type of each element (use type of enumeration tDataType)
            \sa create, tDataType
        */
        DataObject(const size_t sizeY, const size_t sizeX, const int type): m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            size_t sizes[2] = {sizeY, sizeX};
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
        DataObject(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const unsigned char continuous = 0) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
             size_t sizes[3] = {sizeZ, sizeY, sizeX};

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
        DataObject(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const uchar* continuousDataPtr,  const size_t* steps = NULL) : m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            size_t sizes[3] = {sizeZ, sizeY, sizeX};

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
        DataObject(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous = 0) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
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
        DataObject(const unsigned char dimensions, const size_t *sizes, const int type, const uchar* continuousDataPtr, const size_t* steps = NULL) : m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
            this->create(dimensions, sizes, type, m_continuous, continuousDataPtr, steps);
        }

        DataObject(const unsigned char dimensions, const size_t *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes) : m_continuous(0), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_objSharedDataLock(0), m_pDataObjectTags(0)
        {
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
        DataObject & operator = (const int8 value);          /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const uint8 value);         /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const int16 value);         /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const uint16 value);        /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const int32 value);         /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const uint32 value);        /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const float32 value);       /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const float64 value);       /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const complex64 value);     /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */
        DataObject & operator = (const complex128 value);    /*!< sets all elements of the data object to the given value. Value is cast to the data object's type */


        DataObject & operator += (const DataObject &rhs);
        DataObject & operator += (const float64 value);

        DataObject operator + (const DataObject &rhs);
        DataObject operator + (const float64 value);

        DataObject & operator -= (const DataObject &rhs);
        DataObject & operator -= (const float64 value);

        DataObject operator - (const DataObject &rhs);
        DataObject operator - (const float64 value);

        DataObject & operator *= (const DataObject &rhs);
        DataObject & operator *= (const float64 factor);

        DataObject operator * (const DataObject &rhs);
        DataObject operator * (const float64 factor);

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
        RetVal zeros(const size_t size, const int type);
        RetVal zeros(const size_t sizeY, const size_t sizeX, const int type);
        RetVal zeros(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const unsigned char continuous = 0);
        RetVal zeros(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous = 0);

        // allocates matrix with all values set to one
        RetVal ones(const int type);
        RetVal ones(const size_t size, const int type);
        RetVal ones(const size_t sizeY, const size_t sizeX, const int type);
        RetVal ones(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const unsigned char continuous = 0);
        RetVal ones(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous = 0);

        // allocates matrix with uniform distributed noise
        RetVal rand(const int type, const bool randMode = false);
        RetVal rand(const size_t size, const int type, const bool randMode = false);
        RetVal rand(const size_t sizeY, const size_t sizeX, const int type, const bool randMode = false);
        RetVal rand(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const bool randMode, const unsigned char continuous = 0);
        RetVal rand(const unsigned char dimensions, const size_t *sizes, const int type, const bool randMode, const unsigned char continuous = 0);

        // allocates matrix with eye-matrix representation
        RetVal eye(const int type);
        RetVal eye(const size_t size, const int type);

        RetVal conj();
        DataObject adj() const;
        DataObject trans() const;

        //RetVal makeContinuous(void);

        DataObject mul(const DataObject &mat2, const double scale = 1.0);
        DataObject div(const DataObject &mat2, const double scale = 1.0);

        DataObject squeeze() const;

        size_t elemSize() const;

/*
        // Adressing operators ()
        // You should NOT use DataObject::operator here! Otherwise the MsVc compiler will hate you  - at least today: 30.03.2011
        template<typename _Tp> _Tp& operator () (const unsigned int x) const;
        template<typename _Tp> _Tp& operator () (const unsigned int x);
        template<typename _Tp> _Tp& operator () (const unsigned int y, const unsigned int x) const;
        template<typename _Tp> _Tp& operator () (const unsigned int y, const unsigned int x);
        template<typename _Tp> _Tp& operator () (const unsigned int z, const unsigned int y, const unsigned int x) const;
        template<typename _Tp> _Tp& operator () (const unsigned int z, const unsigned int y, const unsigned int x);
        template<typename _Tp> _Tp& operator () (const unsigned int *idx) const;
        template<typename _Tp> _Tp& operator () (const unsigned int *idx);
        DataObject operator () (const ito::Range rowRange, const ito::Range colRange);
        DataObject operator () (ito::Range *ranges);

*/
        // Adressing functions

        //! addressing method for one-dimensional data object or two-dimensional data object having at least one dimension with size 1.
        /*!
            \param x is the zero-based index to the element which is requested (considering any ROI)
            \return reference to specific element
        */
        template<typename _Tp> _Tp& at(const int x) const
        {
         #if __ITODEBUG
            if ((m_dims != 1) && !((m_dims == 2) && ((m_size[0] == 1) || (m_size[1] == 1))))
            {
               cv::error(cv::Exception(CV_StsAssert, "Dimension mismatch while addressing data field", "", __FILE__, __LINE__));
            }
         #endif
            if (m_dims == 1)
            {
         #if __ITODEBUG
               if ((size_t)x >= m_size[0])
               {
                  cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
               }
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(0, x);
            }
            else if (m_size[0] == 1)
            {
         #if __ITODEBUG
               if ((size_t)x >= m_size[1])
               {
                  cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
               }
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(0, x);
            }
            else
            {
         #if __ITODEBUG
               if ((size_t)x >= m_size[0])
               {
                  cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
               }
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(x, 0);
            }
        }

        //! addressing method for one-dimensional data object or two-dimensional data object having at least one dimension with size 1.
        /*!
            \param x is the zero-based index to the element which is requested (considering any ROI)
            \return const reference to specific element
        */
        template<typename _Tp> _Tp& at(const int x)
        {
         #if __ITODEBUG
            if ((m_dims != 1) && !((m_dims == 2) && ((m_size[0] == 1) || (m_size[1] == 1))))
            {
               cv::error(cv::Exception(CV_StsAssert, "Dimension mismatch while addressing data field", "", __FILE__, __LINE__));
            }
         #endif
            if (m_dims == 1)
            {
         #if __ITODEBUG
               if ((size_t)x >= m_size[0])
               {
                  cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
               }
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(0, x);
            }
            else if (m_size[0] == 1)
            {
         #if __ITODEBUG
               if ((size_t)x >= m_size[1])
               {
                  cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
               }
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(0, x);
            }
            else
            {
         #if __ITODEBUG
               if ((size_t)x >= m_size[0])
               {
                  cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
               }
         #endif
               return (*(cv::Mat_<_Tp> *)(m_data[0]))(x, 0);
            }
        }

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
            else if ((x >= m_size[1]) || (y >= m_size[0]) )
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
            else if (((size_t)x >= m_size[1]) || ((size_t)y >= m_size[0]) )
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
            else if (((size_t)x >= m_size[2]) || ((size_t)y >= m_size[1]) || (((size_t)z + m_roi[0]) >= (m_roi[0] + m_size[0])))
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
            else if (((size_t)x >= m_size[2]) || ((size_t)y >= m_size[1]) || (((size_t)z + m_roi[0]) >= (m_roi[0] + m_size[0])))
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
            size_t matNum = 0;

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
            size_t matNum = 0;

            matIdxToNum(idx, &matNum);

               return (*(cv::Mat_<_Tp> *)(m_data[matNum]))(idx[m_dims - 2], idx[m_dims - 1]);
        }

        DataObject at(const ito::Range rowRange, const ito::Range colRange);     /*!< addressing method for two-dimensional data object with two given range-values. returns shallow copy of addressed regions */
        DataObject at(ito::Range *ranges);                                       /*!< addressing method for n-dimensional data object with n given range-values. returns shallow copy of addressed regions */

        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
            cast this pointer to the data type of the matrix elements (as pointer).

            \remark No further error checking (e.g. boundaries)
            \return data-pointer
        */
        uchar* rowPtr(const size_t matNum, const int y)
        {
            size_t matIndex = seekMat(matNum);
            return ((cv::Mat*)m_data[matIndex])->ptr(y);
        }

        //! returns pointer to the data in the y-th row in the 2d-matrix plane matNum
        /*!
            cast this pointer to the data type of the matrix elements (as pointer).

            \remark No further error checking (e.g. boundaries)
            \return data-pointer
        */
        const uchar* rowPtr(const size_t matNum, const int y) const
        {
            size_t matIndex = seekMat(matNum);
            return ((cv::Mat*)m_data[matIndex])->ptr(y);
        }

        DataObject row(const int selRow);
        DataObject col(const int selCol);
        //DataObject diag(void);

        // ROI
        DataObject & adjustROI(const int dtop, const int dbottom, const int dleft, const int dright);   /*!< changes the boundaries of the ROI of a two-dimensional data object by the given incremental values */
        DataObject & adjustROI(const unsigned char dims, const int *lims);                              /*!< changes the boundaries of the ROI of a n-dimensional data object by the given incremental values */
        RetVal locateROI(int *wholeSizes, int *offsets);                              /*!< locates the boundaries of the ROI of a n-dimensional data object and returns the original size and the distances to the physical borders */
        RetVal locateROI(int *lims);                                                  /*!< locates the boundaries of the ROI of a n-dimensional data object the distances to the physical borders */
        template<typename _Tp> RetVal copyFromData2D(const _Tp* src, const size_t sizeX, const size_t sizeY);        //!< copies 2D continuous data into data object, data object must have correct size and type, otherwise an error is returned
        template<typename _Tp> RetVal copyFromData2D(const _Tp *src, const size_t sizeX, const size_t sizeY, const int x0, const int y0, const size_t width, const size_t height);       //!< copies 2D continuous data into data object, data object must have correct size and type, otherwise an error is returned
        template<typename _Tp> RetVal checkType(const _Tp *src);    //compares type of elements in this data objects and type of given argument (doc in source)

        //
        template<typename T2> operator T2 ();  /*!< cast operator, tries to cast this data object to another element type */
};


//----------------------------------------------------------------------------------------------------------------------------------
// functions for DataObject in namespace ITO, which are NOT member functions
//----------------------------------------------------------------------------------------------------------------------------------
DataObject abs(const DataObject &dObj);              /*!< calculates the absolute values of each element in the given data object and returns the result as new data object */
DataObject arg(const DataObject &dObj);              /*!< calculates the argument of each element in the given data object and returns the result as new data object */
DataObject real(const DataObject &dObj);             /*!< calculates the real part of each element in the given data object and returns the result as new data object */
DataObject imag(const DataObject &dObj);             /*!< calculates the imaginary part of each element in the given data object and returns the result as new data object */

DataObject makeContinuous(const DataObject &dObj);   /*!< if the given data object is not continuously organized, copies the content to a new continuous data object */

template<typename _Tp, typename _T2> RetVal CastFunc(const DataObject *dObj, DataObject *resObj, double alpha = 1.0, double beta = 0.0);

//RetVal minMaxLoc(const DataObject &dObj, double *minVal, double *maxVal, size_t *minPos = NULL, size_t *maxPos = NULL);

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
    _Tp retValue = 0;

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
        retValue = cv::saturate_cast<_Tp>(*(static_cast<ito::rgba32*>(scalar)));
        break;
    default:
        cv::error(cv::Exception(CV_StsAssert, "Input value type unkown", "", __FILE__, __LINE__));
        retValue = 0;
    }

    return retValue;
};



//----------------------------------------------------------------------------------------------------------------------------------
// cout
//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> static std::ostream& coutFunc(std::ostream& out, const DataObject& dObj)
{
    //cv::Mat_<_Tp> *cvMat = NULL;
    size_t numMats = dObj.calcNumMats();
    size_t tMat = 0;

    std::cout << "Array(";

    for (size_t nMat = 0; nMat < numMats; nMat++)
    {
        tMat = dObj.seekMat(nMat, numMats);
        std::cout <<  tMat + 1 << "->(";

//#ifndef linux
      
            std::cout << cv::format( (*((cv::Mat_<_Tp> *)((dObj.get_mdata())[tMat]))) , "numpy" ) << std::endl << std::endl;        
//#endif

        std::cout << ")" << "\n" << std::endl;
    }
    std::cout << ")" << "\n" << std::endl;
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
   coutFunc<ito::complex128>
};

static inline std::ostream& operator << (std::ostream& out, const DataObject& dObj)
{
   return fListCout[dObj.getType()](out, dObj);
}

//! static method which returns the real data object of any given data type
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
template<> inline ito::tDataType getDataType(const rgba32* /*src*/) { return ito::tRGBA32; }


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
template<> inline ito::tDataType getDataType2<rgba32*>() { return ito::tRGBA32; }

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
template<> inline bool isZeroValue(rgba32 v, rgba32 /*epsilon*/)
{
    return v == rgba32();
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


