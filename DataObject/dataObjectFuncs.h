/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#ifndef __DATAOBJFUNCH
#define __DATAOBJFUNCH

#ifdef __GNUC__
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif

#include "dataobj.h"
#include <assert.h>     /* assert */

//! creates template defined function table for all supported data types
#define MAKEHELPERFUNCLIST(FuncName) static t##FuncName fList##FuncName[] = \
{                                                                           \
   FuncName<int8>,                                                          \
   FuncName<uint8>,                                                         \
   FuncName<int16>,                                                         \
   FuncName<uint16>,                                                        \
   FuncName<int32>,                                                         \
   FuncName<uint32>,                                                        \
   FuncName<ito::float32>,                                                  \
   FuncName<ito::float64>,                                                  \
   FuncName<ito::complex64>,                                                \
   FuncName<ito::complex128>,                                               \
   FuncName<ito::Rgba32>                                                    \
};

//! creates function table for the function (FuncName) and both complex data types. The destination method must be templated with two template values.
#define MAKEHELPERFUNCLIST_CMPLX_TO_REAL(FuncName) static t##FuncName fList##FuncName[] = \
{                                                                                         \
    FuncName<ito::complex64,float32>,                                                     \
    FuncName<ito::complex128,float64>                                                     \
};

namespace ito 
{
namespace dObjHelper
{
    enum CmplxSelectionFlags
    {
        CMPLX_ABS_VALUE         = 0,
        CMPLX_IMAGINARY_VALUE   = 1,
        CMPLX_REAL_VALUE        = 2,
        CMPLX_ARGUMENT_VALUE    = 3
    };

    // Invert the unitString
    inline std::string invertUnit(const std::string &oldUnit)
    {
        if(oldUnit.empty())
            return oldUnit;

        int found = (int)oldUnit.find('/');

        if(found!=std::string::npos)
        {
            if(found == 0)
            {
                return oldUnit.substr(1, oldUnit.length() - 1);
            }
            else if(oldUnit[0] == '1' && found == 1)
            {
                return oldUnit.substr(2, oldUnit.length() - 2);
            }
            else
            {
                std::string newString;
                newString.reserve(oldUnit.length());
                newString.append(oldUnit.substr(found + 1, oldUnit.length()- (found + 1)));
                newString.append("/");
                newString.append(oldUnit.substr(0, found));
            }
        }
        else
        {
            std::string newString;
            newString.reserve(oldUnit.length()+2);
            newString.append("1/");
            newString.append(oldUnit);
            return newString;    
        }
        return "";
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /** @fn     itomType2cvType
    *   @brief  This function return the Open-CV-datatype
    *   @param[in]  type    Datatype of the data-object of interest
    *   @return -1 or datatype
    *   @detail This helper function converts itom-dataojecttype to openCV type. If unkwown type occurs, this function returns -1
    */
    inline int itomType2cvType(const int type)
    {
        switch(type)
        {
        case ito::tUInt8:
            return CV_8U;
        case ito::tUInt16:
            return CV_16U;
        case ito::tInt8:
            return CV_8S;
        case ito::tInt16:
            return CV_16S;
        case ito::tInt32:
            return CV_32S;
        case ito::tFloat32:
            return CV_32F;
        case ito::tFloat64:
            return CV_64F;
        default:    //ito::tUInt32 and complextype
            return -1;
        }
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /** @fn     cvType2itomType
    *   @brief  This function return the ITOM-datatype
    *   @param[in]  type    Datatype of the current cv::Mat
    *   @return -1 or datatype
    *   @detail This helper function converts openCV-type to itom-datatype. If unkwown type occurs, this function returns -1
    */
    inline int cvType2itomType(const int type)
    {
        switch(type)
        {
        case CV_8U:
            return ito::tUInt8;
        case CV_16U:
            return ito::tUInt16;
        case CV_8S:
            return ito::tInt8;
        case CV_16S:
            return ito::tInt16;
        case CV_32S:
            return ito::tInt32;
        case CV_32F:
            return ito::tFloat32;
        case CV_64F:
            return ito::tFloat64;
        default:
            return -1;
        }
    }

    //! Copy a line from cv::Mat to int32 linebuffer
    template<typename _Tp> void GetHLineL(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);

    //! Copy a line from cv::Mat to float64 linebuffer
    template<typename _Tp> void GetHLineD(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);

    //! Copy a line from cv::Mat to float64 linebuffer
    template<typename _Tp> void GetHLineC(const cv::Mat *srcPlane, const int x0, const int y, const int length, complex128 * pData);

    //! Copy a line from int32 linebuffer to cv::Mat
    template<typename _Tp> void SetHLineL(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);

    //! Copy a line from float64 linebuffer to cv::Mat
    template<typename _Tp> void SetHLineD(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);

//-----------------------------------------------------------------------------------------------
    /*! \fn GetHLineL
       \brief Helper function which copies Data from a row of a cv:mat to a int32-buffer
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated int32-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<typename _Tp> inline void GetHLineL(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData)
    {
        _Tp* resRowPtr;
        resRowPtr = (_Tp*)(srcPlane->ptr(y));

        int x = x0;
        for(int i = 0; i < length; i++, x++)
        {
            pData[i] = (int32)(resRowPtr[x]);
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \brief Helper function which copies Data from a row of a cv:mat<int32> to a int32-buffer using memcopy
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated int32-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<> inline void GetHLineL<int32>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData)
    {
        int32* resRowPtr;
        resRowPtr = (int32*)(srcPlane->ptr(y));
        memcpy(pData, &resRowPtr[x0], length *sizeof(int32));
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn GetHLineD
       \brief Helper function which copies Data from a row of a cv:mat to a float64-buffer
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated float64-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<typename _Tp> inline void GetHLineD(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData)
    {
        _Tp* resRowPtr;
        resRowPtr = (_Tp*)(srcPlane->ptr(y));

        int x = x0;
        for(int i = 0; i < length; i++, x++)
        {
            pData[i] = cv::saturate_cast<ito::float64>(resRowPtr[x]);
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \brief Helper function which copies Data from a row of a cv:mat<float64> to a float64-buffer using memcopy
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated float64-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<> inline void GetHLineD<ito::float64>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData)
    {
        float64* resRowPtr;
        resRowPtr = (float64*)(srcPlane->ptr(y));
        memcpy(pData, &resRowPtr[x0], length *sizeof(float64));
    }

        //-----------------------------------------------------------------------------------------------
    /*! \fn GetHLineComplex
       \brief Helper function which copies Data from a row of a cv:mat to a complex128-buffer
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated complex128-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<typename _Tp> inline void GetHLineC(const cv::Mat *srcPlane, const int x0, const int y, const int length, complex128 * pData)
    {
        const _Tp* resRowPtr = srcPlane->ptr<_Tp>(y);

        int x = x0;
        for(int i = 0; i < length; i++, x++)
        {
            pData[i] = complex128(cv::saturate_cast<ito::float64>(resRowPtr[x]), 0.0);
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \brief Helper function which copies Data from a row of a cv:mat<complex128> to a complex128-buffer using memcopy
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated complex128-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<> inline void GetHLineC<complex64>(const cv::Mat *srcPlane, const int x0, const int y, const int length, complex128 * pData)
    {
        const complex64* resRowPtr = srcPlane->ptr<complex64>(y);

        int x = x0;
        for(int i = 0; i < length; i++, x++)
        {
            pData[i] = complex128(resRowPtr[x].real(), resRowPtr[x].imag());
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \brief Helper function which copies Data from a row of a cv:mat<complex128> to a complex128-buffer using memcopy
       \param[in] srcPlane  Source for the copy-action
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length    number of elements to copy
       \param[in|out] pData Pointer to allocated complex128-buffer of size elements
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<> inline void GetHLineC<complex128>(const cv::Mat *srcPlane, const int x0, const int y, const int length, complex128 * pData)
    {
        const complex128* resRowPtr = srcPlane->ptr<complex128>(y);
        memcpy(pData, &resRowPtr[x0], length *sizeof(complex128));
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn SetHLineL
       \brief Helper function which copies Data from a int32-buffer to a row of a cv:mat
       \param[in|out] destPlane     Destination for the data
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length number of elements to copy
       \param[in] pData Pointer to int32-buffer of size elements (source)
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<typename _Tp> inline void SetHLineL(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData)
    {
        _Tp* resRowPtr;
        resRowPtr = (_Tp*)(destPlane->ptr(y));

        int x = x0;
        for(int i = 0; i < length; i++, x++)
        {
            resRowPtr[x] = (_Tp)(pData[i]);
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn SetHLineL
       \brief Helper function which copies Data from a int32-buffer to a row of a cv:mat<int32> using memcopy
       \param[in|out] destPlane     Destination for the data
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length number of elements to copy
       \param[in] pData Pointer to int32-buffer of size elements (source)
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<> inline void SetHLineL<int32>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData)
    {
        int32* resRowPtr;
        resRowPtr = (int32*)(destPlane->ptr(y));
        memcpy(&resRowPtr[x0], pData, length *sizeof(int32));
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn SetHLineD
       \brief Helper function which copies Data from a doulbe-buffer to a row of a cv:mat
       \param[in|out] destPlane     Destination for the data
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length number of elements to copy
       \param[in] pData Pointer to float64-buffer of size elements (source)
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<typename _Tp> inline void SetHLineD(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData)
    {
        _Tp* resRowPtr;
        resRowPtr = (_Tp*)destPlane->ptr(y);

        int x = x0;
        for(int i = 0; i < length; i++, x++)
        {
            resRowPtr[x] = cv::saturate_cast<_Tp>(pData[i]);
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn SetHLineD
       \brief Helper function which copies Data from a float64-buffer to a row of a cv:mat<float64> using memcopy
       \param[in|out] destPlane     Destination for the data
       \param[in] x0    Startpixel
       \param[in] y     linenumber
       \param[in] length number of elements to copy
       \param[in] pData Pointer to float64-buffer of size elements (source)
       \author ITO
       \sa GetL, DoGenericFilter
       \date 12.2011
    */
    template<> inline void SetHLineD<ito::float64>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData)
    {
        float64* resRowPtr;
        resRowPtr = (float64*)(destPlane->ptr(y));
        memcpy(&resRowPtr[x0], pData, length *sizeof(float64));
    }

#if 1 // DOBJHELPER_OBSOLTE
    //! Check if a value is equal to zero (trivial)
	
	template<typename _Tp> DEPRECATED inline bool isNotZero(_Tp value)
    {
        if(value == 0)
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for float32
	template<> DEPRECATED inline bool isNotZero<float32>(float32 value)
    {
        float32 lowVal = std::numeric_limits<float32>::epsilon();
        if(fabs(value) < lowVal)
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for float64
	template<> DEPRECATED inline bool isNotZero<float64>(float64 value)
    {
        float64 lowVal = std::numeric_limits<float64>::epsilon();
        if(fabs(value) < lowVal)
            return false;
        else
            return true;
    }

    //! Check if a value is finite (this is for integer types --> always true)
	template<typename _Tp> DEPRECATED inline bool isFinite(_Tp /*value*/)
    {
        return true;
    }

    //! Check if a value is finite float32 values
	template<> DEPRECATED inline bool isFinite<float32>(float32 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[3] & 0x7f) != 0x7f || (ch[2] & 0x80) != 0x80;
    }

    //! Check if a value is finite float64 values
	template<> DEPRECATED inline bool isFinite<float64>(float64 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[7] & 0x7f) != 0x7f || (ch[6] & 0xf0) != 0xf0;
    }

    //! Check if both components of complex64 value are finite
	template<> DEPRECATED inline bool isFinite<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[3] & 0x7f) != 0x7f || (chreal[2] & 0x80) != 0x80) && ((chimag[3] & 0x7f) != 0x7f || (chimag[2] & 0x80) != 0x80);
    }

    //! Check if both components of complex128 value are finite
	template<> DEPRECATED inline bool isFinite<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[7] & 0x7f) != 0x7f || (chreal[6] & 0xf0) != 0xf0) && ((chimag[7] & 0x7f) != 0x7f || (chimag[6] & 0xf0) != 0xf0);
    }

    //! Check if a value is NaN (this is for integer types --> always false)
	template<typename _Tp> DEPRECATED inline bool isNaN(_Tp value)
    {
        return false;
    }

    //! Check if a value is isNaN float32 values
	template<> DEPRECATED inline bool isNaN<float32>(float32 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[3] & 0x7f) == 0x7f && ch[2] > 0x80;
    }
    //! Check if a value is isNaN float64 values
	template<> DEPRECATED inline bool isNaN<float64>(float64 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[7] & 0x7f) == 0x7f && ch[6] > 0xf0;
    }

    //! Check if one of the components of complex64 values are not a number
	template<> DEPRECATED inline bool isNaN<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[3] & 0x7f) == 0x7f && chreal[2] > 0x80) || ((chimag[3] & 0x7f) == 0x7f && chimag[2] > 0x80);
    }

    //! Check if one of the components of complex128 values are not a number
	template<> DEPRECATED inline bool isNaN<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[7] & 0x7f) == 0x7f && chreal[6] > 0xf0) || ((chimag[7] & 0x7f) == 0x7f && chimag[6] > 0xf0);
    }

    //! Check if a value is Inf (this is for integer types --> always false)
	template<typename _Tp> DEPRECATED inline bool isInf(_Tp /*value*/)
    {
        return false;
    }

    //! Check if a value is infinite float32 values
	template<> DEPRECATED inline bool isInf<float32>(float32 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[3] & 0x7f) == 0x7f && ch[2] == 0x80;
    }

    //! Check if a value is infinite float64 values
	template<> DEPRECATED inline bool isInf<float64>(float64 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[7] & 0x7f) == 0x7f && ch[6] == 0xf0;
    }

    //! Check if one of the components of complex64 values are infinite
	template<> DEPRECATED inline bool isInf<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[3] & 0x7f) == 0x7f && chreal[2] == 0x80) || ((chimag[3] & 0x7f) == 0x7f && chimag[2] == 0x80);
    }

    //! Check if one of the components of complex128 values are infinite
	template<> DEPRECATED inline bool isInf<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[7] & 0x7f) == 0x7f && chreal[6] == 0xf0) || ((chimag[7] & 0x7f) == 0x7f && chimag[6] == 0xf0);
    }
#endif // DOBJHELPER_OBSOLTE

    //! Find the min-value of this data object and the first position <Templated version>.
    template<typename _Tp> RetVal minValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf = true);
    
    //! Find the min-value of this data object and the first position.
    RetVal DATAOBJ_EXPORT minValue(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf = true);
       
    //! Find the max-value of this data object and the first position <Templated version>.
    template<typename _Tp> RetVal maxValueFunc(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreInf = true);

    //! Find the max-value of this data object and the first position.
    RetVal DATAOBJ_EXPORT maxValue(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreInf = true);
        
    //! <templated version> Find the minimal and maximal Value in the dataObject and saves their first occurence in firstMinLocation (uint32[3]-Array) and firstMaxLocation (uint32[3]-Array)
    template<typename _Tp> RetVal minMaxValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf = true, const int specialDataTypeFlags = CMPLX_ABS_VALUE);
    
    //! Find the minimal and maximal Value in the dataObject and saves their first occurence in firstMinLocation (uint32[3]-Array) and firstMaxLocation (uint32[3]-Array)
    RetVal DATAOBJ_EXPORT minMaxValue(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf = true, const int specialDataTypeFlags = CMPLX_ABS_VALUE);

    template<typename _Tp, typename _BufTp> RetVal meanValueFunc(const DataObject *dObj, float64 &meanResult, bool ignoreNaN = true);
    RetVal DATAOBJ_EXPORT meanValue(const DataObject *dObj, float64 &meanResult, bool ignoreNaN = true);

    template<typename _Tp, typename _BufTp> RetVal devValueFunc(const ito::DataObject *dObj, const int devTypFlag, float64 &meanResult, float64 &devResult, bool ignoreNaN = true);
    RetVal DATAOBJ_EXPORT devValue(const DataObject *dObj, const int devTypFlag, float64 &meanResult, float64 &devResult, bool ignoreNaN = true);

    RetVal DATAOBJ_EXPORT calcCVDFT(DataObject *dObjIO, const bool inverse, const bool inverseAsReal, const bool lineWise);

    //! Check if the dataObject is of right type
    ito::RetVal DATAOBJ_EXPORT verifyDataObjectType(const ito::DataObject* dObj, const char* name, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    
    //! Check if the dataObject is of right type and if it is 2D and if it is of right size
    ito::RetVal DATAOBJ_EXPORT verify2DDataObject(const ito::DataObject* dObj, const char* name, int sizeYMin, int sizeYMax, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    
    //! Check if the dataObject is of right type and if it is 2D or is 1x...xYxX and if it is of right size
    //ito::RetVal DATAOBJ_EXPORT verify1PlaneDObject(const ito::DataObject* dObj, const char* name, int &yIdx, int sizeYMin, int sizeYMax, int &xIdx, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    
    //! Check if the dataObject is of right type and if it is 3D and if it is of right size
    ito::RetVal DATAOBJ_EXPORT verify3DDataObject(const ito::DataObject* dObj, const char* name, int sizeZMin, int sizeZMax, int sizeYMin, int sizeYMax, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    
    ito::RetVal DATAOBJ_EXPORT verifySize(int size, int minSize, int maxSize, const char *axisName, const char* dObjName);

    //! returns a shallow or deep copy of a given data object that fits to given requirements
    ito::DataObject DATAOBJ_EXPORT squeezeConvertCheck2DDataObject(const ito::DataObject *dObj, const char* name, const ito::Range &sizeY, const ito::Range &sizeX, ito::RetVal &retval, int convertToType, uint8 numberOfAllowedTypes, ...);
    
    //-----------------------------------------------------------------------------------------------
    /*! \fn freeRowPointer
        \brief Delete a 3D-Pointer array previous created with getRowPointer(...)
               The function iters through all pointer[n] and deletes the underlying
               sizeY pointer** which were allocated by getRowPointer(...). The end of pointer*
               is marked by a NULL.

        \param[in]   pointer  The _Type *** pointer to be deleted

        \author  Lyda
        \sa      getRowPointer
        \date    12.2012
    */
    template<typename _Tp> ito::RetVal freeRowPointer(_Tp *** &pointer)
    {
        if(pointer == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "Delete rowpointer failed, pointer was not allocated freeRowPointer(...)");
        }

        int i = 0;

        while(pointer[i] != 0)
        {
            free(pointer[i]);
            i++;
            // end of pointer is marked by a NULL-Pointer
        }

        free(pointer);
        pointer = NULL;

        return ito::retOk;
    }

    /*! \fn getRowPointer
        \brief Allocate a 3D-Pointer array with address [plane, y, x] for a n-Dim dataObject
               This function allocates an type*** pointer to address each pixel by plane number, y-index, x-index.
               The size of the array is planes+1, ysize and row-length. The last plane index is set to NULL.
               The pointer array must be cleaned by freeRowPointer.
               The function checks for type compability of object and pointer.

        \param[in]   *dObj    pointer to an existing dataObject of right type
        \param[in]   pointer  A _Type *** pointer of correct type

        \author  Lyda
        \sa      freeRowPointer
        \date    12.2012 
    */

    template<typename _Tp> ito::RetVal getRowPointer(ito::DataObject* dObj, _Tp *** &pointer)  
    {

        if(!dObj)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject was a NULL pointer in function getRowPointer(...)");               
        }

        if(dObj->getType() != ito::getDataType2<_Tp*>())
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject and template Type differed in function getRowPointer(...)");               
        }

        if(dObj->getDims() < 2)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject was not initialized in function getRowPointer(...)");   
        }

        pointer = (_Tp***)calloc(dObj->getNumPlanes() + 1, sizeof(_Tp**));

        if(pointer == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "Allocate plane-pointer failed in function getRowPointer(...)");
        }

        ito::RetVal retVal(ito::retOk);
        int sizeY = dObj->getSize(dObj->getDims() - 2);

        cv::Mat** mdata = dObj->get_mdata();
        int numMats = dObj->getNumPlanes();
        for(int i = 0; i < dObj->getNumPlanes(); i++)
        {
            pointer[i] = (_Tp**)calloc(sizeY, sizeof(_Tp*));
            if(pointer[i])
            {
                for(int y = 0; y < sizeY; y++)
                {
                    pointer[i][y] = mdata[dObj->seekMat(i, numMats)]->ptr<_Tp>(y);
                }
            }
            else
            {
                retVal += ito::RetVal::format(ito::retError, 0, "Allocate row-pointer failed in function getRowPointer(...)");
                break;
            }
        }

        if(retVal.containsError())
        {
            freeRowPointer(pointer);
        }
        return retVal;
    } 
 
    //-----------------------------------------------------------------------------------------------
    /*! \fn isIntType 
        \brief  Helpfunction to check if object type is integer type or not and if integer point the size is spezified in size
        \param[in]   type    tDataType of a DataObject 
        \param[out]  size    Number of bytes if floating point type (1 or 4) else -1
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool isIntType(const int type, int *size)
    {
        if((type == ito::tInt8)||(type == ito::tUInt8))
        {
            *size = 1;
            return true;
        }
        else if((type == ito::tInt16)||(type == ito::tUInt16))
        {
            *size = 2;
            return true;
        }
        else if((type == ito::tInt32)||(type == ito::tUInt32))
        {
            *size = 4;
            return true;
        }
        else
        {
            *size = -1;
            return false;
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn isFPType 
        \brief  Helpfunction to check if object type is floating point type or not and if floating point point the size is spezified in size
        \param[in]   type    tDataType of a DataObject 
        \param[out]  size    Number of bytes if floating point type (4 or 8) else -1
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool isFPType(const int type, int *size)
    {
        if((type == ito::tFloat32))
        {
            *size = 4;
            return true;
        }
        else if((type == ito::tFloat64))
        {
            *size = 8;
            return true;
        }
        else
        {
            *size = -1;
            return false;
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn isCplxType 
        \brief  Helpfunction to check if object type is complex-type or not and if complex point the size is spezified in size
        \param[in]   type    tDataType of a DataObject 
        \param[out]  size    Number of bytes if complex type (8 or 16) else -1
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool isCplxType(const int type, int *size)
    {
        if((type == ito::tComplex64))
        {
            *size = 8;
            return true;
        }
        else if((type == ito::tComplex128))
        {
            *size = 16;
            return true;
        }
        else
        {
            *size = -1;
            return false;
        }
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn isEqualDObj 
        \brief  Helpfunction to check if two dataObjects are equal in size/dims and type. Returns false if not equal
        \param[in]   dObj1   DataObject1
        \param[in]   dObj2   DataObject2
        \param[out]  typeFlag    If true both objects are of same type
        \param[out]  dimsFlag    If true both object have same dims and sizes 
        \param[out]  last2DimsFlag   If true x/y of the object are equal
        \return true if equal else false
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool dObjareEqualDetail(const ito::DataObject *dObj1, const ito::DataObject *dObj2, bool &typeFlag, bool &dimsFlag, bool &last2DimsFlag)
    {
        assert(dObj1 && dObj2);

        bool sizeEqual;

        last2DimsFlag = true;

        int d1 = dObj1->getDims();
        int d2 = dObj2->getDims();
        dimsFlag = (d1 == d2);
        typeFlag = (dObj1->getType() == dObj2->getType());

        if(d1 != d2)
        {
            sizeEqual = false;

            if (d1 >= 2 && d2 >= 2)
            {
                if(dObj1->getSize(dObj1->getDims() - 2) != dObj2->getSize(dObj2->getDims() - 2))
                {
                    last2DimsFlag = false;
                }
                else if(dObj1->getSize(dObj1->getDims() - 1) != dObj2->getSize(dObj2->getDims() - 1))
                {
                    last2DimsFlag = false;
                }
            }
            else
            {
                last2DimsFlag = false;
            }
        }
        else
        {
            sizeEqual = (dObj1->getSize() == dObj2->getSize());

            if (d1 >= 2)
            {
                if(dObj1->getSize(dObj1->getDims() - 2) != dObj2->getSize(dObj2->getDims() - 2))
                {
                    last2DimsFlag = false;
                }
                else if(dObj1->getSize(dObj1->getDims() - 1) != dObj2->getSize(dObj2->getDims() - 1))
                {
                    last2DimsFlag = false;
                }
            }
            else
            {
                last2DimsFlag = false;
            }
        }
        return typeFlag && dimsFlag && last2DimsFlag && sizeEqual;
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn isEqualShort 
        \brief  check if both data objects are equal concerning their number of dimensions, sizes and type
        \param[in]   dObj1   first data object
        \param[in]   dObj2   second data obect
        \return true if size and type of both objects are equal, else false
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool dObjareEqualShort(const ito::DataObject *obj1, const ito::DataObject *obj2)
    {
        assert(obj1 && obj2);

        return (obj1->getType() == obj2->getType()) && (obj1->getSize() == obj2->getSize());
    }

    //! Helperfunction to copy axis related tags from a n-D-Object to a m-D-Object.
    ito::RetVal DATAOBJ_EXPORT dObjCopyLastNAxisTags(const ito::DataObject &DataObjectIn, ito::DataObject &DataObjectOut, const int copyLastNDims, const bool includeValueTags = true, const bool includeRotationMatrix = true);

    ito::RetVal DATAOBJ_EXPORT dObjSetScaleRectangle(ito::DataObject &DataObjectInOut, const double &x0, const double &x1, const double &y0, const double &y1);
} //end namespace dObjHelper
} //end namespace ito

#endif
