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

#ifndef __DATAOBJFUNCH
#define __DATAOBJFUNCH

#include "dataobj.h"

//! creates template defined function table for all supported data types
#define MAKEHELPERFUNCLIST(FuncName) static t##FuncName fList##FuncName[] =   \
{                                                                       \
   FuncName<int8>,                                                      \
   FuncName<uint8>,                                                     \
   FuncName<int16>,                                                     \
   FuncName<uint16>,                                                    \
   FuncName<int32>,                                                     \
   FuncName<uint32>,                                                    \
   FuncName<ito::float32>,                                                   \
   FuncName<ito::float64>,                                                   \
   FuncName<ito::complex64>,                                                 \
   FuncName<ito::complex128>,                                                 \
   FuncName<ito::Rgba32>                                                 \
};

//! creates function table for the function (FuncName) and both complex data types. The destination method must be templated with two template values.
#define MAKEHELPERFUNCLIST_CMPLX_TO_REAL(FuncName) static t##FuncName fList##FuncName[] =     \
{                                                                                       \
    FuncName<ito::complex64,float32>,                                                        \
    FuncName<ito::complex128,float64>                                                        \
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
    inline std::string invertUnit(const std::string oldUnit)
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
        default:	//ito::tUInt32 and complextype
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
        _Tp* resRowPtr;
        resRowPtr = (_Tp*)(srcPlane->ptr(y));

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
        complex64* resRowPtr;
        resRowPtr = (complex64*)(srcPlane->ptr(y));

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
        complex128* resRowPtr;
        resRowPtr = (complex128*)(srcPlane->ptr(y));
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

    //! Check if a value is equal to zero (trivial)
    template<typename _Tp> inline bool isNotZero(_Tp value)
    {
        if(value == 0)
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for float32
    template<> inline bool isNotZero<float32>(float32 value)
    {
        float32 lowVal = std::numeric_limits<float32>::epsilon();
        if(fabs(value) < lowVal)
            return false;
        else
            return true;
    }

    //! Check if a value is equal to zero for float64
    template<> inline bool isNotZero<float64>(float64 value)
    {
        float64 lowVal = std::numeric_limits<float64>::epsilon();
        if(fabs(value) < lowVal)
            return false;
        else
            return true;
    }

    //! Check if a value is finite (this is for integer types --> always true)
    template<typename _Tp> inline bool isFinite(_Tp /*value*/)
    {
        return true;
    }

    //! Check if a value is finite float32 values
    template<> inline bool isFinite<float32>(float32 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[3] & 0x7f) != 0x7f || (ch[2] & 0x80) != 0x80;
    }

    //! Check if a value is finite float64 values
    template<> inline bool isFinite<float64>(float64 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[7] & 0x7f) != 0x7f || (ch[6] & 0xf0) != 0xf0;
    }

    //! Check if both components of complex64 value are finite
    template<> inline bool isFinite<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[3] & 0x7f) != 0x7f || (chreal[2] & 0x80) != 0x80) && ((chimag[3] & 0x7f) != 0x7f || (chimag[2] & 0x80) != 0x80);
    }

    //! Check if both components of complex128 value are finite
    template<> inline bool isFinite<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[7] & 0x7f) != 0x7f || (chreal[6] & 0xf0) != 0xf0) && ((chimag[7] & 0x7f) != 0x7f || (chimag[6] & 0xf0) != 0xf0);
    }

    //! Check if a value is NaN (this is for integer types --> always false)
    template<typename _Tp> inline bool isNaN(_Tp value)
    {
        return false;
    }

    //! Check if a value is isNaN float32 values
    template<> inline bool isNaN<float32>(float32 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[3] & 0x7f) == 0x7f && ch[2] > 0x80;
    }
    //! Check if a value is isNaN float64 values
    template<> inline bool isNaN<float64>(float64 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[7] & 0x7f) == 0x7f && ch[6] > 0xf0;
    }

    //! Check if one of the components of complex64 values are not a number
    template<> inline bool isNaN<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[3] & 0x7f) == 0x7f && chreal[2] > 0x80) || ((chimag[3] & 0x7f) == 0x7f && chimag[2] > 0x80);
    }

    //! Check if one of the components of complex128 values are not a number
    template<> inline bool isNaN<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[7] & 0x7f) == 0x7f && chreal[6] > 0xf0) || ((chimag[7] & 0x7f) == 0x7f && chimag[6] > 0xf0);
    }

    //! Check if a value is Inf (this is for integer types --> always false)
    template<typename _Tp> inline bool isInf(_Tp /*value*/)
    {
        return false;
    }

    //! Check if a value is infinite float32 values
    template<> inline bool isInf<float32>(float32 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[3] & 0x7f) == 0x7f && ch[2] == 0x80;
    }

    //! Check if a value is infinite float64 values
    template<> inline bool isInf<float64>(float64 value)
    {
        uchar *ch = (uchar *)&value;
        return (ch[7] & 0x7f) == 0x7f && ch[6] == 0xf0;
    }

    //! Check if one of the components of complex64 values are infinite
    template<> inline bool isInf<complex64>(complex64 value)
    {
        float32 realVal = value.real();
        float32 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[3] & 0x7f) == 0x7f && chreal[2] == 0x80) || ((chimag[3] & 0x7f) == 0x7f && chimag[2] == 0x80);
    }

    //! Check if one of the components of complex128 values are infinite
    template<> inline bool isInf<complex128>(complex128 value)
    {
        float64 realVal = value.real();
        float64 imagVal = value.real();
        uchar *chreal = (uchar *)&realVal;
        uchar *chimag = (uchar *)&imagVal;
        return ((chreal[7] & 0x7f) == 0x7f && chreal[6] == 0xf0) || ((chimag[7] & 0x7f) == 0x7f && chimag[6] == 0xf0);
    }

    //! Find the min-value of this data object and the first position <Templated version>.
    template<typename _Tp> RetVal minValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf = true);
    
    //! Find the min-value of this data object and the first position.
    RetVal minValue(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf = true);
       
    //! Find the max-value of this data object and the first position <Templated version>.
    template<typename _Tp> RetVal maxValueFunc(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreInf = true);

    //! Find the max-value of this data object and the first position.
    RetVal maxValue(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreInf = true);
        
    //! <templated version> Find the minimal and maximal Value in the dataObject and saves their first occurence in firstMinLocation (uint32[3]-Array) and firstMaxLocation (uint32[3]-Array)
    template<typename _Tp> RetVal minMaxValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf = true, const int specialDataTypeFlags = CMPLX_ABS_VALUE);
    
    //! Find the minimal and maximal Value in the dataObject and saves their first occurence in firstMinLocation (uint32[3]-Array) and firstMaxLocation (uint32[3]-Array)
    RetVal minMaxValue(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf = true, const int specialDataTypeFlags = CMPLX_ABS_VALUE);

    template<typename _Tp, typename _BufTp> RetVal meanValueFunc(const DataObject *dObj, float64 &meanResult, bool ignoreNaN = true);
    RetVal meanValue(const DataObject *dObj, float64 &meanResult, bool ignoreNaN = true);

    template<typename _Tp, typename _BufTp> RetVal devValueFunc(const ito::DataObject *dObj, const int devTypFlag, float64 &meanResult, float64 &devResult, bool ignoreNaN = true);
    RetVal devValue(const DataObject *dObj, const int devTypFlag, float64 &meanResult, float64 &devResult, bool ignoreNaN = true);

    RetVal calcCVDFT(DataObject *dObjIO, const bool inverse, const bool inverseAsReal, const bool lineWise);

    ito::RetVal verifyDataObjectType(const ito::DataObject* dObj, const char* name, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    ito::RetVal verify2DDataObject(const ito::DataObject* dObj, const char* name, int sizeYMin, int sizeYMax, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    ito::RetVal verify3DDataObject(const ito::DataObject* dObj, const char* name, int sizeZMin, int sizeZMax, int sizeYMin, int sizeYMax, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...); //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    ito::RetVal verifySize(int size, int minSize, int maxSize, const char *axisName, const char* dObjName);
    
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

        pointer = (_Tp***)calloc(dObj->calcNumMats() + 1, sizeof(_Tp**));

        if(pointer == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "Allocate plane-pointer failed in function getRowPointer(...)");
        }

        ito::RetVal retVal(ito::retOk);
        int sizeY = dObj->getSize(dObj->getDims() - 2);

        uchar** mdata = dObj->get_mdata();
        for(int i = 0; i < dObj->calcNumMats(); i++)
        {
            pointer[i] = (_Tp**)calloc(sizeY, sizeof(_Tp*));
            if(pointer[i])
            {
                for(int y = 0; y < sizeY; y++)
                {
                    pointer[i][y] = ((cv::Mat*)(mdata[dObj->seekMat(i)]))->ptr<_Tp>(y);
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
    
    //template<typename _Tp> void minMaxValueHelper(ito::DataObject *dObj, float64 *min, float64 *max, int matNumber);

    //-----------------------------------------------------------------------------------------------
    /*! \fn checkITOMType
        \brief   This helperfunction compares the type to the allowed types and returns an error if the given type in not allowed 
        \param[in]   type    Type of the dataObject
        \param[in]   allow_int8  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_uint8  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_int16  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_uint16  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_int32  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_uint32  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_float32  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_float64  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_complex64  If true, type is allowed, else if type = allow_* an error is return
        \param[in]   allow_complex128  If true, type is allowed, else if type = allow_* an error is return
        \author  ITO 
        \sa 
        \date 12.2011
    */
    /*
    inline ito::RetVal checkITOMType(ito::DataObject *dObj, bool allow_int8, bool allow_uint8, bool allow_int16, bool allow_uint16, bool allow_int32, bool allow_uint32, bool allow_float32, bool allow_float64, bool allow_complex64, bool allow_complex128)
    {
        int type = dObj->getType();
	    switch(type)
	    {
	    case ito::tUInt8:
		    if(allow_uint8)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "UInt8 not allowed");
	    case ito::tUInt16:
		    if(allow_uint16)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "UInt16 not allowed");
	    case ito::tUInt32:
		    if(allow_uint32)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "UInt32 not allowed");
	    case ito::tInt8:
		    if(allow_int8)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "Int8 not allowed");
	    case ito::tInt16:
		    if(allow_int16)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "Int16 not allowed");
	    case ito::tInt32:
		    if(allow_int32)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "Int32 not allowed");
	    case ito::tFloat32:
		    if(allow_float32)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "Float32 not allowed");
	    case ito::tFloat64:
		    if(allow_float64)
			    return 0;
		    else return ito::RetVal(ito::retError, 0, "Float64 not allowed");
	    case ito::tComplex64:
		    if(allow_complex64)
			    return ito::retOk;
		    else return ito::RetVal(ito::retError, 0, "Complex64 not allowed");
	    case ito::tComplex128:
		    if(allow_complex128)
			    return 0;
		    else return ito::RetVal(ito::retError, 0, "Complex128 not allowed");
	    default:
		    return ito::RetVal(ito::retError, 0, "Unknown type or type not implemented");
	    }
    }
    */
 
    //-----------------------------------------------------------------------------------------------
    /*! \fn isIntType 
        \brief  Helpfunction to check if object type is integer type or not and if integer point the size is spezified in size
        \param[in]   type    Type of a DataObject 
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
        \param[in]   type    ITO-Objtype of a DataObject 
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
        \param[in]   type    ITO-Objtype of a DataObject 
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
        \brief  Helpfunction to check if to dataObjects are equal in size/dims and type. Returns false if not equal
        \param[in]   dObj1   DataObject1
        \param[in]   dObj2   DataObject2
        \param[out]  typeFlag    If true both objects are of same type
        \param[out]  dimsFlag    If true both object have same dims and sizes 
        \param[out]  last2DimsFlag   If true x/y of the object are equal
        \return true of equal else false
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool dObjareEqualDetail(ito::DataObject *dObj1, ito::DataObject *dObj2, bool &typeFlag, bool &dimsFlag, bool &last2DimsFlag)
    {
        bool retVal = true;
        typeFlag = true;
        dimsFlag = true;
        last2DimsFlag = true;
        //transFlag = true;

        if(dObj1->getType() != dObj2->getType())
        {
            retVal = false;
            typeFlag = false;
        }
        /*if(dObj1->isT() != dObj2->isT())
        {
            retVal = false;
            transFlag = false;        
        }*/

        if(dObj1->getDims() != dObj2->getDims())
        {
            retVal = false;
            dimsFlag = false;
            if(dObj1->getSize(dObj1->getDims() - 2) != dObj2->getSize(dObj2->getDims() - 2))
            {
                last2DimsFlag = false;
            }
            if(dObj1->getSize(dObj1->getDims() - 1) != dObj2->getSize(dObj2->getDims() - 1))
            {
                last2DimsFlag = false;
            }
        }
        else
        {
            for(int i = 0; i < dObj1->getDims()-2; i++)
            {
                if(dObj1->getSize(i) != dObj2->getSize(i))
                {
                    retVal = false;
                }
            }
            if(dObj1->getSize(dObj1->getDims() - 2) != dObj2->getSize(dObj2->getDims() - 2))
            {
                last2DimsFlag = false;
                retVal = false;
            }
            if(dObj1->getSize(dObj1->getDims() - 1) != dObj2->getSize(dObj2->getDims() - 1))
            {
                last2DimsFlag = false;
                retVal = false;
            }
        }
        return retVal;
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn isEqualShort 
        \brief  Helpfunction to check if to dataObjects are equal in size/dims and type. Returns false if not equal
        \param[in]   dObj1   DataObject1
        \param[in]   dObj2   DataObject2
        \return true of equal eslse false
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    inline bool dObjareEqualShort(ito::DataObject *obj1, ito::DataObject *obj2)
    {
        if(obj1->getDims() == obj2->getDims() && obj1->getType() == obj2->getType() /*&& obj1->isT() == obj2->isT()*/)
        {
            for(int i = 0; i < obj1->getDims(); i++)
            {
                if(obj1->getSize(i) != obj2->getSize(i))
                {
                    return false;
                }
            }
            return true;
        }
        else
            return false;
    }

    //! Helperfunction to copy axis related tags from a n-D-Object to a m-D-Object.
    ito::RetVal dObjCopyLastNAxisTags(const ito::DataObject &DataObjectIn, ito::DataObject &DataObjectOut, const int copyLastNDims, const bool includeValueTags = true, const bool includeRotationMatrix = true);

    ito::RetVal dObjSetScaleRectangle(ito::DataObject &DataObjectInOut, const double &x0, const double &x1, const double &y0, const double &y1);
}
}

#endif
