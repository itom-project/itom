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

#include "dataObjectFuncs.h"
#include "../common/numeric.h"

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
    //------------------------------------------------------------------------------------------------------------------
    // documentation in header file
    std::string invertUnit(const std::string &oldUnit)
    {
        if (oldUnit.empty())
            return oldUnit;

        int found = (int)oldUnit.find('/');

        if (found != std::string::npos)
        {
            if (found == 0)
            {
                return oldUnit.substr(1, oldUnit.length() - 1);
            }
            else if (oldUnit[0] == '1' && found == 1)
            {
                return oldUnit.substr(2, oldUnit.length() - 2);
            }
            else
            {
                std::string newString;
                newString.reserve(oldUnit.length());
                newString.append(oldUnit.substr(found + 1, oldUnit.length() - (found + 1)));
                newString.append("/");
                newString.append(oldUnit.substr(0, found));
            }
        }
        else
        {
            std::string newString;
            newString.reserve(oldUnit.length() + 2);
            newString.append("1/");
            newString.append(oldUnit);
            return newString;
        }
        return "";
    }


    //! Templated version of minValueFunc which calc min-value of this data object and the first position
    /*!
        NaN-Values will be ignored by this method
        Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored.
        The function does not check if dObj != NULL && firstLocation != NULL &&  dObj.type != tComplexXX !!!

        \param[in]      dObj            handle to the dataObject
        \param[out]     minValue        lowest value in this object
        \param[in|out]  firstLocation   Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf       Ignore Inf-Values

        \return retOK
    */

    template<typename _Tp> RetVal minValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf)
    {
        unsigned int numMats = dObj->getNumPlanes();
        int matIndex = 0;
        int m,n;

        const cv::Mat_<_Tp> *mat = NULL;
        const _Tp* rowPtr;

        _Tp tempMinValue = std::numeric_limits<_Tp>::max();

        if(ignoreInf)
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(isFinite<_Tp>(rowPtr[n]) && rowPtr[n] < tempMinValue) 
                        {
                            tempMinValue = rowPtr[n]; // This ignores Nan and Inf
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(rowPtr[n] < tempMinValue) 
                        {
                            tempMinValue = rowPtr[n];  // This should ignore NaN anyway, but allows Inf
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }

        minValue = cv::saturate_cast<float64>(tempMinValue);
        return ito::retOk;    
    }

    template<> RetVal minValueFunc<complex64>(const DataObject * /*dObj*/, float64 & /*minValue*/, uint32 * /*firstLocation*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "minValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal minValueFunc<complex128>(const DataObject * /*dObj*/, float64 & /*minValue*/, uint32 * /*firstLocation*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "minValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal minValueFunc<Rgba32>(const DataObject * /*dObj*/, float64 & /*minValue*/, uint32 * /*firstLocation*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "minValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    typedef RetVal (*tminValueFunc)(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf);
    MAKEHELPERFUNCLIST(minValueFunc)

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns min-value of this data object and the first position
    /*!
        NaN-Values will be ignored by this method
        Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored.
        The function checks if dObj != NULL && firstLocation != NULL &&  dObj.type != tComplexXX 

        \param[in]      dObj            handle to the dataObject
        \param[out]     minValue        lowest value in this object
        \param[in|out]  firstLocation   Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf       Ignore Inf-Values

        \return retOK or in case dObj == NULL || firstLocation == NULL || for complex Object it returns retError 
    */
    RetVal minValue(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf)
    {
        minValue = std::numeric_limits<float64>::max();

        if(firstLocation == NULL)
            return ito::RetVal(retError, 0, "firstLocation must not be NULL");

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "Input dataObject must not be empty");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128 || dObj->getType() == tRGBA32)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        return fListminValueFunc[dObj->getType()](dObj, minValue, firstLocation, ignoreInf);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Templated version of maxValueFunc which returns max-value of this data object and the first position
                NaN-Values will be ignored by this method
                Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored.
                The function does not check if dObj != NULL && firstLocation != NULL &&  dObj.type != tComplexXX !!!

        \param[in]      dObj            handle to the dataObject
        \param[out]     maxValue        highest value in this object
        \param[in|out]  firstLocation   Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf       Ignore Inf-Values

        \return retOK
    */
        
    template<typename _Tp> RetVal maxValueFunc(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreNaN)
    {
       unsigned int numMats = dObj->getNumPlanes();
       int matIndex = 0;
       int m,n;

       cv::Mat_<_Tp> *mat = NULL;
       const _Tp* rowPtr;

       _Tp tempMaxValue;

       if(std::numeric_limits<_Tp>::is_exact)
       {
           tempMaxValue = std::numeric_limits<_Tp>::min(); //integer numbers
       }
       else
       {
           tempMaxValue = -1 * std::numeric_limits<_Tp>::max();
       }

        if(ignoreNaN)
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(isFinite<_Tp>(rowPtr[n]) && rowPtr[n] > tempMaxValue) 
                        {
                            tempMaxValue = rowPtr[n];
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(rowPtr[n] > tempMaxValue) 
                        {
                            tempMaxValue = rowPtr[n];
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }

        maxValue = cv::saturate_cast<float64>(tempMaxValue);
        return ito::retOk;
    }

    template<> RetVal maxValueFunc<complex64>(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreNaN)
    {
        unsigned int numMats = dObj->getNumPlanes();
        int matIndex = 0;
        int m, n;

        const cv::Mat_<complex64> *mat = NULL;
        const complex64* rowPtr;

        float64 tempMaxValue;

        tempMaxValue = -1 * std::numeric_limits<float64>::max();

        if (ignoreNaN)
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<complex64> *)(dObj->get_mdata())[matIndex];

                for (m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<complex64>(m);
                    for (n = 0; n < mat->cols; n++)
                    {
                        if (isFinite<complex64>(rowPtr[n]) && abs(rowPtr[n]) > tempMaxValue)
                        {
                            tempMaxValue = abs(rowPtr[n]);
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<complex64> *)(dObj->get_mdata())[matIndex];

                for (m = 0; m < mat->rows; m++)
                {
                    rowPtr = (complex64*)mat->ptr(m);
                    for (n = 0; n < mat->cols; n++)
                    {
                        if (abs(rowPtr[n]) > tempMaxValue)
                        {
                            tempMaxValue = abs(rowPtr[n]);
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }

        maxValue = cv::saturate_cast<float64>(tempMaxValue);
        return ito::retOk;
    }

    template<> RetVal maxValueFunc<complex128>(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreNaN)
    {
        unsigned int numMats = dObj->getNumPlanes();
        int matIndex = 0;
        int m, n;

        cv::Mat_<complex128> *mat = NULL;
        const complex128* rowPtr;

        float64 tempMaxValue = -1 * std::numeric_limits<float64>::max();

        if (ignoreNaN)
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<complex128> *)(dObj->get_mdata())[matIndex];

                for (m = 0; m < mat->rows; m++)
                {
                    rowPtr = (complex128*)mat->ptr(m);
                    for (n = 0; n < mat->cols; n++)
                    {
                        if (isFinite<complex128>(rowPtr[n]) && abs(rowPtr[n]) > tempMaxValue)
                        {
                            tempMaxValue = abs(rowPtr[n]);
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<complex128> *)(dObj->get_mdata())[matIndex];

                for (m = 0; m < mat->rows; m++)
                {
                    rowPtr = (complex128*)mat->ptr(m);
                    for (n = 0; n < mat->cols; n++)
                    {
                        if (abs(rowPtr[n]) > tempMaxValue)
                        {
                            tempMaxValue = abs(rowPtr[n]);
                            firstLocation[0] = nmat;
                            firstLocation[1] = m;
                            firstLocation[2] = n;
                        }
                    }
                }
            }
        }

        maxValue = cv::saturate_cast<float64>(tempMaxValue);
        return ito::retOk;
    }

    template<> RetVal maxValueFunc<Rgba32>(const DataObject * /*dObj*/, float64 & /*maxValue*/, uint32 * /*firstLocation*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "maxValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    typedef RetVal (*tmaxValueFunc)(const DataObject *dObj, float64 &minValue, uint32 *firstLocation, bool ignoreInf);
    MAKEHELPERFUNCLIST(maxValueFunc)

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
        \detail Find the max-value of this data object and the first position.
                NaN-Values will be ignored by this method and the Inf-Value handling depends on inf-flag. 
                If ignoreInf == true, inf is ignored else inf is not ignored.
                For complex valued data types the position with the largest magnitude is returned.

        \param[in]      dObj            handle to the dataObject
        \param[out]     maxValue        highest value in this object
        \param[in|out]  firstLocation   Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf       Ignore Inf-Values

        \return retOK or in case dObj == NULL || firstLocation == NULL || for complex Object it returns retError 
    */
    RetVal maxValue(const DataObject *dObj, float64 &maxValue, uint32 *firstLocation, bool ignoreInf)
    {
        ito::RetVal retval = ito::retOk;

        if(firstLocation == NULL)
            return ito::RetVal(retError, 0, "firstLocation must not be NULL");

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "Input dataObject must not be empty");

        return fListmaxValueFunc[dObj->getType()](dObj, maxValue, firstLocation, ignoreInf);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
    \detail This function searches for min and max-value of the <_Type> data object and saves its first detects positions in firstMinLocation (must be uint32[3]-Array) and firstMaxLocation (must be  uint32[3]-Array).
            NaN-Values will be ignored by this method and Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored. 
            The specialDataTypeFlags for complex / rgba32 handling selection is unused in the FP and INT-versions of the function.
            Warning, does not check if dObj && firstMinLocation && firstMaxLocation are valid!

        \param[in]      dObj                    handle to the dataObject
        \param[out]     minValue                lowest value in this object
        \param[in|out]  firstMinLocation        Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue                highest value in this object
        \param[in|out]  firstMaxLocation        Allocated uint32[3]-array. Will be filled with [mat-Number, ymax, xmax]
        \param[in]      ignoreInf               Ignore Inf-Values
        \param[in]      specialDataTypeFlags                Toggle complex handling (not used) 

        \return retOk
    */
    template<typename _Tp> RetVal minMaxValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int /*specialDataTypeFlags*/)
    {
        int numMats = dObj->getNumPlanes();
        int matIndex = 0;

        int m,n;

        cv::Mat_<_Tp> *mat = NULL;
        const _Tp* rowPtr;
        _Tp tempResultMin;
        _Tp tempResultMax;

        tempResultMin = std::numeric_limits<_Tp>::max();
        if(std::numeric_limits<_Tp>::is_exact)
        {
            tempResultMax = std::numeric_limits<_Tp>::min(); //integer numbers
        }
        else
        {
            tempResultMax = -1 * std::numeric_limits<_Tp>::max();
        }

        if(ignoreInf)   // Ignores inf
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                    
                        if(isInf<_Tp>(rowPtr[n])) continue;

                        if(rowPtr[n] < tempResultMin) //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                        {
                            tempResultMin = rowPtr[n]; 
                            firstMinLocation[0] = nmat;
                            firstMinLocation[1] = m;
                            firstMinLocation[2] = n;
                        }
                            
                        if(rowPtr[n] > tempResultMax) 
                        {
                            tempResultMax = rowPtr[n]; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                            firstMaxLocation[0] = nmat;
                            firstMaxLocation[1] = m;
                            firstMaxLocation[2] = n;
                        }
                    }
                }
            }
        }
        else
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(rowPtr[n] < tempResultMin) 
                        {
                            tempResultMin = rowPtr[n]; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                            firstMinLocation[0] = nmat;
                            firstMinLocation[1] = m;
                            firstMinLocation[2] = n;
                        }
                        if(rowPtr[n] > tempResultMax) 
                        {
                            tempResultMax = rowPtr[n]; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                            firstMaxLocation[0] = nmat;
                            firstMaxLocation[1] = m;
                            firstMaxLocation[2] = n;
                        }
                    }
                }
            }
        }

        minValue = cv::saturate_cast<float64>(tempResultMin);
        maxValue = cv::saturate_cast<float64>(tempResultMax);

        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
    \brief  Search min/max in a <complex64> data object
    \detail This function searches for min and max-value of the <complex64> data object and saves its first detects positions in firstMinLocation (must be uint32[3]-Array) and firstMaxLocation (must be  uint32[3]-Array).
            NaN-Values will be ignored by this method and Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored.
            Warning, does not check if dObj && firstMinLocation && firstMaxLocation are valid!

        \param[in]      dObj                    handle to the dataObject
        \param[out]     minValue                lowest value in this object
        \param[in|out]  firstMinLocation        Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue                highest value in this object
        \param[in|out]  firstMaxLocation        Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf               Ignore Inf-Values
        \param[in]      specialDataTypeFlags    Toggle complex handling, 0:abs-Value, 1:imaginary-Value, 2:real-Value, 3: argument-Value, see ito::dObjHelper::CmplxSelectionFlags

        \return retOk
    */
    template<> RetVal minMaxValueFunc<ito::complex64>(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int specialDataTypeFlags)
    {
        int numMats = dObj->getNumPlanes();
        int matIndex = 0;

        uint32 m,n;
        uint32 cols,rows;

        cv::Mat_<ito::complex64> *mat = NULL;
        const ito::complex64* rowPtr;
        float tempResultMin;
        float tempResultMax;
        float tmpVal;

        tempResultMin = std::numeric_limits<float>::max();
        tempResultMax = -tempResultMin;

        if(ignoreInf)   // Ignores inf
        {
            for (int32 nmat = 0; nmat < (int32)numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex64> *)(dObj->get_mdata())[matIndex];
                cols = (uint32)mat->cols;
                rows = (uint32)mat->rows;
            
                switch (specialDataTypeFlags)
                {
                    default:    
                    case CMPLX_ABS_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                if(isInf<complex64>(rowPtr[n].real())) continue;
                                if(isInf<complex64>(rowPtr[n].imag())) continue;
                                tmpVal = abs(rowPtr[n]);
                                if(tmpVal < tempResultMin) 
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax) 
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_IMAGINARY_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                if(isInf<complex64>(rowPtr[n].imag())) continue;
                                tmpVal = rowPtr[n].imag();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax) 
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_REAL_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                if(isInf<complex64>(rowPtr[n].real())) continue;
                                tmpVal = rowPtr[n].real();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_ARGUMENT_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                if(isInf<complex64>(rowPtr[n].real())) continue;
                                if(isInf<complex64>(rowPtr[n].imag())) continue;
                                tmpVal = arg(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;
                }
            }
        }
        else
        {
            for (uint32 nmat = 0; nmat < (uint32)numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex64> *)(dObj->get_mdata())[matIndex];
                cols = (uint32)mat->cols;
                rows = (uint32)mat->rows;

                switch (specialDataTypeFlags)
                {
                    default:    
                    case CMPLX_ABS_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                tmpVal = abs(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_IMAGINARY_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                tmpVal = rowPtr[n].imag();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_REAL_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                tmpVal = rowPtr[n].real();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax) 
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_ARGUMENT_VALUE:
                        for(m = 0; m < rows; m++)
                        {
                            rowPtr = mat->ptr<complex64>(m);
                            for(n = 0; n < cols; n++)
                            {
                                tmpVal = arg(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMinLocation[0] = nmat;
                                    firstMinLocation[1] = m;
                                    firstMinLocation[2] = n;
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;
                }
            }
        }

        minValue = cv::saturate_cast<float64>(tempResultMin);
        maxValue = cv::saturate_cast<float64>(tempResultMax);

    //    cv::error(cv::Exception(CV_StsAssert, "getMinMaxValue not defined for complex type", "", __FILE__, __LINE__));
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*! 
    \brief  Search min/max in a <complex128> data object
    \detail This function searches for min and max-value of the <complex128> data object and saves its first detects positions in firstMinLocation (must be uint32[3]-Array) and firstMaxLocation (must be  uint32[3]-Array).
            NaN-Values will be ignored by this method and Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored.
            Warning, does not check if dObj && firstMinLocation && firstMaxLocation are valid!

        \param[in]      dObj                handle to the dataObject
        \param[out]     minValue            lowest value in this object
        \param[in|out]  firstMinLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue            highest value in this object
        \param[in|out]  firstMaxLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf           Ignore Inf-Values
        \param[in]      specialDataTypeFlags            Toggle complex handling, 0:abs-Value, 1:imaginary-Value, 2:real-Value, 3: argument-Value

        \return retOk
    */

    template<> RetVal minMaxValueFunc<ito::complex128>(const DataObject *dObj, float64 &minValue, uint32 * /*firstMinLocation*/, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int specialDataTypeFlags)
    {
        int numMats = dObj->getNumPlanes();
        int matIndex = 0;

        int m,n;

        cv::Mat_<ito::complex128> *mat = NULL;
        const ito::complex128* rowPtr;
        float64 tempResultMin;
        float64 tempResultMax;
        float64 tmpVal;

        tempResultMin = std::numeric_limits<float64>::max();
        tempResultMax = -tempResultMin;

        if(ignoreInf)   // Ignores inf
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex128> *)(dObj->get_mdata())[matIndex];
            
                switch (specialDataTypeFlags)
                {
                    default:    
                    case CMPLX_ABS_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                if(isInf<complex128>(rowPtr[n].real())) continue;
                                if(isInf<complex128>(rowPtr[n].imag())) continue;
                                tmpVal = abs(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_IMAGINARY_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                if(isInf<complex128>(rowPtr[n].imag())) continue;
                                tmpVal = rowPtr[n].imag();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_REAL_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                if(isInf<complex128>(rowPtr[n].real())) continue;
                                tmpVal = rowPtr[n].real();
                                
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_ARGUMENT_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                if(isInf<complex128>(rowPtr[n].real())) continue;
                                if(isInf<complex128>(rowPtr[n].imag())) continue;
                                tmpVal = arg(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;
                }
            }
        }
        else
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex128> *)(dObj->get_mdata())[matIndex];

                switch (specialDataTypeFlags)
                {
                    default:    
                    case CMPLX_ABS_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                tmpVal = abs(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_IMAGINARY_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                tmpVal = rowPtr[n].imag();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_REAL_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                tmpVal = rowPtr[n].real();
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;

                    case CMPLX_ARGUMENT_VALUE:
                        for(m = 0; m < mat->rows; m++)
                        {
                            rowPtr = mat->ptr<complex128>(m);
                            for(n = 0; n < mat->cols; n++)
                            {
                                tmpVal = arg(rowPtr[n]);
                                if(tmpVal < tempResultMin)
                                {
                                    tempResultMin = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                }
                                if(tmpVal > tempResultMax)
                                {
                                    tempResultMax = tmpVal; //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                    firstMaxLocation[0] = nmat;
                                    firstMaxLocation[1] = m;
                                    firstMaxLocation[2] = n;
                                }
                            }
                        }
                    break;
                }
            }
        }

        minValue = cv::saturate_cast<float64>(tempResultMin);
        maxValue = cv::saturate_cast<float64>(tempResultMax);

    //    cv::error(cv::Exception(CV_StsAssert, "getMinMaxValue not defined for complex type", "", __FILE__, __LINE__));
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
    \detail This function searches for min and max-value of the <_Type> data object and saves its first detects positions in firstMinLocation (must be uint32[3]-Array) and firstMaxLocation (must be  uint32[3]-Array).
            The specialDataTypeFlags for complex handling / rgba32 selection is used to toggle between the different channels.
            If specialDataTypeFlags == RGBA_B the maximal & minimal value for the blue channel is returned.
            If specialDataTypeFlags == RGBA_G the maximal & minimal value for the green channel is returned
            If specialDataTypeFlags == RGBA_R the maximal & minimal value for the red channel is returned
            If specialDataTypeFlags == RGBA_A the maximal & minimal value for the alpha channel is returned
            If specialDataTypeFlags == RGBA_Y for each pixel a gray-value transformation in YUV-space is done and the maximal & minimal value for the Y is returned
            If specialDataTypeFlags == RGBA_RGB the maximal and minimal value over all channels will be returned.
            Warning, does not check if dObj && firstMinLocation && firstMaxLocation are valid!

        \param[in]      dObj                handle to the dataObject
        \param[out]     minValue            lowest value in this object
        \param[in|out]  firstMinLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue            highest value in this object
        \param[in|out]  firstMaxLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf           Ignore Inf-Values
        \param[in]      specialDataTypeFlags            Toggle rgba-channel handling, see ito::Rgba32_t::RGBSelectionFlags

        \return retOk
    */
    template<> RetVal minMaxValueFunc<Rgba32>(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool /*ignoreInf*/, const int specialDataTypeFlags)
    {
        int numMats = dObj->getNumPlanes();
        int matIndex = 0;

        int m,n;

        const Rgba32* rowPtr;
        cv::Mat_<Rgba32> *mat = NULL;

//        rgba32 tempResultMin;
//        rgba32 tempResultMax;
//        tempResultMin = std::numeric_limits<ito::uint32>::max();
//        tempResultMax = std::numeric_limits<ito::uint32>::min(); //integer numbers

        uint8 tmpMin = 255;
        uint8 tmpMax = 0;
        float32 tmpMinFloat = std::numeric_limits<float32>::max();
        float32 tmpMaxFloat = -std::numeric_limits<float32>::max();

        for (int nmat = 0; nmat < numMats; nmat++)
        {
            matIndex = dObj->seekMat(nmat, numMats);
            mat = (cv::Mat_<Rgba32> *)(dObj->get_mdata())[matIndex];

            switch(specialDataTypeFlags)
            {
                case Rgba32::RGBA_B:
                {
                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (Rgba32*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(rowPtr[n].blue() < tmpMin) 
                            {
                                firstMinLocation[0] = nmat;
                                firstMinLocation[1] = m;
                                firstMinLocation[2] = n;
                                tmpMin = rowPtr[n].blue();
                            }
                            if(rowPtr[n].blue() > tmpMax) 
                            {
                                firstMaxLocation[0] = nmat;
                                firstMaxLocation[1] = m;
                                firstMaxLocation[2] = n;
                                tmpMax = rowPtr[n].blue();
                            }
                        }
                    }
                }
                break;
                case Rgba32::RGBA_G:
                {
                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (Rgba32*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(rowPtr[n].g < tmpMin) 
                            {
                                tmpMin = rowPtr[n].green();
                                firstMinLocation[0] = nmat;
                                firstMinLocation[1] = m;
                                firstMinLocation[2] = n;
                            }
                            if(rowPtr[n].g > tmpMin) 
                            {
                                tmpMax = rowPtr[n].green();
                                firstMaxLocation[0] = nmat;
                                firstMaxLocation[1] = m;
                                firstMaxLocation[2] = n;
                            }
                        }
                    }
                }
                break;
                case Rgba32::RGBA_R:
                {
                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (Rgba32*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(rowPtr[n].r < tmpMin) 
                            {
                                tmpMin = rowPtr[n].red();
                                firstMinLocation[0] = nmat;
                                firstMinLocation[1] = m;
                                firstMinLocation[2] = n;
                            }
                            if(rowPtr[n].r > tmpMax) 
                            {
                                tmpMax = rowPtr[n].red();
                                firstMaxLocation[0] = nmat;
                                firstMaxLocation[1] = m;
                                firstMaxLocation[2] = n;
                            }
                        }
                    }
                }
                break;
                case Rgba32::RGBA_A:
                {
                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (Rgba32*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(rowPtr[n].a < tmpMin) 
                            {
                                tmpMin = rowPtr[n].alpha(); //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                firstMinLocation[0] = nmat;
                                firstMinLocation[1] = m;
                                firstMinLocation[2] = n;
                            }
                            if(rowPtr[n].a > tmpMax) 
                            {
                                tmpMax = rowPtr[n].alpha(); //NaN will be ignored by this comparison (that means if rowPtr[n]=NaN, the if-result is always false)
                                firstMaxLocation[0] = nmat;
                                firstMaxLocation[1] = m;
                                firstMaxLocation[2] = n;
                            }
                        }
                    }
                }
                break;
                case Rgba32::RGBA_Y:
                {
                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (Rgba32*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(rowPtr[n].gray() < tmpMinFloat) 
                            {
                                firstMinLocation[0] = nmat;
                                firstMinLocation[1] = m;
                                firstMinLocation[2] = n;
                                tmpMinFloat = rowPtr[n].gray();
                            }
                            if(rowPtr[n].alpha() > tmpMaxFloat) 
                            {
                                firstMaxLocation[0] = nmat;
                                firstMaxLocation[1] = m;
                                firstMaxLocation[2] = n;
                                tmpMaxFloat = rowPtr[n].gray();
                            }
                        }
                    }
                }
                break;
                default:
                case Rgba32::RGBA_RGB:
                {
                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (Rgba32*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(rowPtr[n].red() < tmpMin || rowPtr[n].blue() < tmpMin || rowPtr[n].green() < tmpMin ) 
                            {
                                firstMinLocation[0] = nmat;
                                firstMinLocation[1] = m;
                                firstMinLocation[2] = n;
                                tmpMin = rowPtr[n].r < rowPtr[n].b ? rowPtr[n].r : (rowPtr[n].g < rowPtr[n].b ? rowPtr[n].g : rowPtr[n].b);
                            }
                            if(rowPtr[n].red() > tmpMax || rowPtr[n].blue() > tmpMax || rowPtr[n].green() > tmpMax) 
                            {
                               firstMaxLocation[0] = nmat;
                                firstMaxLocation[1] = m;
                                firstMaxLocation[2] = n;
                                tmpMax = rowPtr[n].r > rowPtr[n].b ? rowPtr[n].r : (rowPtr[n].g > rowPtr[n].b ? rowPtr[n].g : rowPtr[n].b);
                            }
                        }
                    }
                }
                break;
            }
        }

        if(specialDataTypeFlags == Rgba32::RGBA_Y)
        {
            minValue = static_cast<float64>(tmpMinFloat);
            maxValue = static_cast<float64>(tmpMaxFloat);
        }
        else
        {
            minValue = static_cast<float64>(tmpMin);
            maxValue = static_cast<float64>(tmpMax);
        }
        return ito::retOk;
    }

    typedef RetVal (*tminMaxValueFunc)(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int specialDataTypeFlags);
    MAKEHELPERFUNCLIST(minMaxValueFunc);

    //----------------------------------------------------------------------------------------------------------------------------------
    /*! \detail This function searches for min and max-value of the data object and saves its first detects positions in firstMinLocation (must be uint32[3]-Array) and firstMaxLocation (must be  uint32[3]-Array).
                NaN-Values will be ignored by this method and Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored.
                The function checks if dObj != NULL && firstLocation != NULL &&  dObj.type != tComplexXX 

        \param[in]      dObj                handle to the dataObject
        \param[out]     minValue            lowest value in this object
        \param[in|out]  firstMinLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue            highest value in this object
        \param[in|out]  firstMaxLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf           Ignore Inf-Values

        \return retOK or in case dObj == NULL || firstMinLocation == NULL || firstMinLocation == NULL it returns retError 
    */
    RetVal minMaxValue(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int specialDataTypeFlags)
    {
        minValue = std::numeric_limits<float64>::max();
        maxValue = -std::numeric_limits<float64>::max();

        if(firstMinLocation == NULL)
            return ito::RetVal(retError, 0, "firstMinLocation must not be NULL");

        if(firstMaxLocation == NULL)
            return ito::RetVal(retError, 0, "firstMaxLocation must not be NULL");

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "Input dataObject must not be empty");

        return fListminMaxValueFunc[dObj->getType()](dObj, minValue, firstMinLocation, maxValue, firstMaxLocation, ignoreInf, specialDataTypeFlags);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns mean-value of the data object
    /*!
        NaN-Values & Inf-Value handling depends on NaN-flag. If ignoreNaN == true, both are ignored else not.
        The function does not check if dObj != NULL &&  dObj.type != tComplexXX 

        \param[in]      dObj                handle to the dataObject
        \param[out]     meanResult          float64-type mean value
        \param[in]      ignoreNaN           Ignore NaN-Values && Inf-Values

        \return retOK
    */
    template<typename _Tp, typename _BufTp> RetVal meanValueFunc(const DataObject *dObj, float64 &meanResult, bool ignoreNaN)
    {
        unsigned int numMats = dObj->getNumPlanes();
        int matIndex = 0;

        int m,n;
        unsigned int nrOfValidElements = 0;

        cv::Mat_<_Tp> *mat = NULL;
        const _Tp* rowPtr;
        _BufTp sum = 0;

        if(ignoreNaN)
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<_Tp>(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(isFinite<_Tp>(rowPtr[n]))
                        {
                            sum += rowPtr[n];
                            nrOfValidElements ++;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = (_Tp*)mat->ptr(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        sum += rowPtr[n];
                        nrOfValidElements ++;
                    }
                }
            }
        }
        if(nrOfValidElements == 0) nrOfValidElements = 1; //in order to avoid divide-by-zero crash

        meanResult = static_cast<float64>(sum)/nrOfValidElements;
        return ito::retOk;
    }

    template<> RetVal meanValueFunc<complex64, float64>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "meanValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }
    template<> RetVal meanValueFunc<complex64, complex64>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "meanValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal meanValueFunc<complex128, float64>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "meanValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }
    template<> RetVal meanValueFunc<complex128, complex128>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "meanValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal meanValueFunc<Rgba32, uint32>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "meanValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }
    template<> RetVal meanValueFunc<Rgba32, Rgba32>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "meanValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns mean-value of the data object
    /*!
        NaN-Values & Inf-Value handling depends on NaN-flag. If ignoreNaN == true, both are ignored else not.
        The function checks if dObj != NULL &&  dObj.type != tComplexXX 

        \param[in]      dObj                handle to the dataObject
        \param[out]     meanResult          float64-type mean value
        \param[in]      ignoreNaN           Ignore NaN-Values && Inf-Values

        \return retOK or in case dObj == NULL || firstMinLocation == NULL || firstMinLocation == NULL it returns retError 
    */
    RetVal meanValue(const DataObject *dObj, float64 &meanResult, bool ignoreNaN)
    {
        ito::RetVal retval = ito::retOk;

        int dims = dObj->getDims();

        if(dObj == NULL || dims == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128 || dObj->getType() == tRGBA32)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        meanResult = std::numeric_limits<float64>::max();

        switch( dObj->getType() )
        {
        case tUInt8:
        {
            if((dObj->getNumPlanes() * dObj->getSize(dims-1) * dObj->getSize(dims-2)) > 8388000) retval += meanValueFunc<uint8, float64>(dObj, meanResult, false);
            else retval += meanValueFunc<uint8, int32>(dObj, meanResult, false);
            break;
        }
        case tInt8:
        {
            if((dObj->getNumPlanes() * dObj->getSize(dims-1) * dObj->getSize(dims-2)) > 8388000) retval += meanValueFunc<int8, float64>(dObj, meanResult, false);
            else retval += meanValueFunc<int8, int32>(dObj, meanResult, false);
            break;
        }   
    
        case tUInt16:
        {
            retval += meanValueFunc<uint16, float64>(dObj, meanResult, false);
            break;
        }
        case tInt16:
        {
            retval += meanValueFunc<int16, float64>(dObj, meanResult, false);
            break;
        }
        case tUInt32:
        {
            retval += meanValueFunc<uint32, float64>(dObj, meanResult, false);
            break;
        }
        case tInt32:
        {
            retval += meanValueFunc<int32, float64>(dObj, meanResult, false);
            break;
        }
        case tFloat32:
        {
            retval += meanValueFunc<float32, float64>(dObj, meanResult, ignoreNaN);
            break;
        }
        case tFloat64:
        {
            retval += meanValueFunc<float64, float64>(dObj, meanResult, ignoreNaN);
            break;
        }
        default:
        {
            retval += ito::RetVal(retError, 0, "data type not supported");
            break;
        }
        }
        return retval;
    }


    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns median-value of the data object
    /*!
    NaN-Values & Inf-Value handling depends on NaN-flag. If ignoreNaN == true, both are ignored else not.
    The function does not check if dObj != NULL &&  dObj.type != tComplexXX

    \param[in]      dObj                handle to the dataObject
    \param[out]     meanResult          float64-type mean value
    \param[in]      ignoreNaN           Ignore NaN-Values && Inf-Values

    \return retOK
    */
    template<typename _Tp> RetVal medianValueFunc(const DataObject *dObj, float64 &medianResult, bool ignoreNaN)
    {
        ito::DataObject temp;
        size_t num = dObj->getTotal();
        _Tp* values = new _Tp[num];
        const cv::Mat *mat;
        const _Tp* rowPtr;

        if (!ignoreNaN)
        {
            //copy all values into a continuous buffer which is also used for sorting (later on)
            size_t size;
            unsigned char* values_ptr = (unsigned char*)values;

            for (int p = 0; p < dObj->getNumPlanes(); ++p)
            {
                mat = dObj->getCvPlaneMat(p);
                if (mat->isContinuous())
                {
                    size = sizeof(_Tp) * mat->cols * mat->rows;
                    memcpy(values_ptr, mat->data, size);
                    values_ptr += size;
                }
                else
                {
                    size = sizeof(_Tp) * mat->cols;
                    for (int r = 0; r < mat->rows; ++r)
                    {
                        rowPtr = mat->ptr<_Tp>(r);
                        memcpy(values_ptr, rowPtr, size);
                        values_ptr += size;
                    }
                }
            }
        }
        else
        {
            size_t idx = 0;

            for (int p = 0; p < dObj->getNumPlanes(); ++p)
            {
                mat = dObj->getCvPlaneMat(p);
                for (int r = 0; r < mat->rows; ++r)
                {
                    rowPtr = mat->ptr<_Tp>(r);
                    for (int c = 0; c < mat->cols; ++c)
                    {
                        if (ito::isFinite(rowPtr[c]))
                        {
                            values[idx++] = rowPtr[c];
                        }
                    }
                }
            }

            num = idx;
        }

        //this algorithms seems to be like the following: http://www.i-programmer.info/babbages-bag/505-quick-median.html?start=1
        ito::uint32 halfKernSize = num / 2;
        ito::uint32 leftElement = 0;
        ito::uint32 rightElement = num - 1;
        ito::uint32 leftPos, rightPos;
        _Tp a;
        _Tp tempValue;
        while (leftElement < rightElement)
        {
            a = values[halfKernSize];
            leftPos = leftElement;
            rightPos = rightElement;
            do
            {
                while (values[leftPos] < a)
                {
                    leftPos++;
                }
                while (values[rightPos] > a)
                {
                    rightPos--;
                }
                if (leftPos <= rightPos)
                {
                    tempValue = values[leftPos];
                    values[leftPos] = values[rightPos];
                    values[rightPos] = tempValue;
                    leftPos++;
                    rightPos--;
                }
            } while (leftPos <= rightPos);

            if (rightPos < halfKernSize)
            {
                leftElement = leftPos;
            }
            if (halfKernSize < leftPos)
            {
                rightElement = rightPos;
            }
        }
        medianResult = values[halfKernSize];

        DELETE_AND_SET_NULL_ARRAY(values);
        
        return ito::retOk;
    }

    template<> RetVal medianValueFunc<complex64>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "medianValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal medianValueFunc<complex128>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "medianValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal medianValueFunc<Rgba32>(const ito::DataObject * /*dObj*/, float64 & /*meanResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "medianValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }


    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns median-value of the data object
    /*!
    NaN-Values & Inf-Value handling depends on NaN-flag. If ignoreNaN == true, both are ignored else not.
    The function checks if dObj != NULL &&  dObj.type != tComplexXX

    \param[in]      dObj                handle to the dataObject
    \param[out]     meanResult          float64-type mean value
    \param[in]      ignoreNaN           Ignore NaN-Values && Inf-Values

    \return retOK or in case dObj == NULL || firstMinLocation == NULL || firstMinLocation == NULL it returns retError
    */
    RetVal medianValue(const DataObject *dObj, float64 &medianResult, bool ignoreNaN)
    {
        ito::RetVal retval = ito::retOk;

        int dims = dObj->getDims();

        if (dObj == NULL || dims == 0)
            return ito::RetVal(retError, 0, "DataObject must not be empty");

        if (dObj->getType() == tComplex64 || dObj->getType() == tComplex128 || dObj->getType() == tRGBA32)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        medianResult = std::numeric_limits<float64>::max();

        switch (dObj->getType())
        {
        case tUInt8:
        {
            retval += medianValueFunc<uint8>(dObj, medianResult, false);
            break;
        }
        case tInt8:
        {
            retval += medianValueFunc<int8>(dObj, medianResult, false);
            break;
        }

        case tUInt16:
        {
            retval += medianValueFunc<uint16>(dObj, medianResult, false);
            break;
        }
        case tInt16:
        {
            retval += medianValueFunc<int16>(dObj, medianResult, false);
            break;
        }
        case tUInt32:
        {
            retval += medianValueFunc<uint32>(dObj, medianResult, false);
            break;
        }
        case tInt32:
        {
            retval += medianValueFunc<int32>(dObj, medianResult, false);
            break;
        }
        case tFloat32:
        {
            retval += medianValueFunc<float32>(dObj, medianResult, ignoreNaN);
            break;
        }
        case tFloat64:
        {
            retval += medianValueFunc<float64>(dObj, medianResult, ignoreNaN);
            break;
        }
        default:
        {
            retval += ito::RetVal(retError, 0, "data type not supported");
            break;
        }
        }
        return retval;
    }
    

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns mean-value and the standard deviation of the data object
    /*!
        NaN-Values & Inf-Value handling depends on NaN-flag. If ignoreNaN == true, both are ignored else not.
        The function does not check if dObj != NULL &&  dObj.type != tComplexXX 

        \param[in]      dObj                handle to the dataObject
        \param[in]      devTypFlag          should be 0 or 1, due to different definitions of standard deviation
        \param[out]     meanResult          float64-type mean value
        \param[out]     devResult           float64-type std-value
        \param[in]      ignoreNaN           Ignore NaN-Values && Inf-Values

        \return retOK
    */
    template<typename _Tp, typename _BufTp> RetVal devValueFunc(const ito::DataObject *dObj, const int devTypFlag, float64 &meanResult, float64 &devResult, bool ignoreNaN)
    {
        unsigned int numMats = dObj->getNumPlanes();
        int matIndex = 0;

        int m,n;
        unsigned int nrOfValidElements = 0;

        cv::Mat_<_Tp> *mat = NULL;
        const _Tp* rowPtr;
        _BufTp sum = 0;

        if(ignoreNaN && !std::numeric_limits<_Tp>::is_exact)
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = (_Tp*)mat->ptr(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        if(isFinite<_Tp>(rowPtr[n]))
                        {
                            sum += rowPtr[n];
                            nrOfValidElements ++;
                        }
                    }
                }
            }
        }
        else
        {
            for (unsigned int nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = (_Tp*)mat->ptr(m);
                    for(n = 0; n < mat->cols; n++)
                    {
                        sum += rowPtr[n];
                        nrOfValidElements ++;
                    }
                }
            }
        }

        if(nrOfValidElements == 0) nrOfValidElements = 1; //in order to avoid divide-by-zero crash

        float64 meanValue = static_cast<float64>(sum) / nrOfValidElements;

        float64 devValue = 0.0;
        float64 dev = 0.0;

        if(nrOfValidElements > 1)
        {
            nrOfValidElements = 0;
            float64 temp = 0.0;
            if(ignoreNaN && !std::numeric_limits<_Tp>::is_exact)
            {
                for (unsigned int nmat = 0; nmat < numMats; nmat++)
                {
                    matIndex = dObj->seekMat(nmat, numMats);
                    mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (_Tp*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            if(isFinite<_Tp>(rowPtr[n]))
                            {
                                temp = static_cast<float64>(rowPtr[n]) - meanValue;
                                dev += temp * temp;
                                nrOfValidElements ++;
                            }
                        }
                    }
                }
            }
            else
            {
                for (unsigned int nmat = 0; nmat < numMats; nmat++)
                {
                    matIndex = dObj->seekMat(nmat, numMats);
                    mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                    for(m = 0; m < mat->rows; m++)
                    {
                        rowPtr = (_Tp*)mat->ptr(m);
                        for(n = 0; n < mat->cols; n++)
                        {
                            temp = static_cast<float64>(rowPtr[n]) - meanValue;
                            dev += temp * temp;
                            nrOfValidElements ++;
                        }
                    }
                }
            }

            if ((nrOfValidElements - 1 + devTypFlag) != 0)
            {
                devValue = sqrt(dev / (nrOfValidElements - 1 + devTypFlag));  // if flag = 1, std = 1/n * sqrt(sum((x - xm)²)) else std = 1/(n-1) * sqrt(sum((x - xm)²)) else
            }
        }

        meanResult = meanValue;
        devResult = devValue;

        return ito::retOk;
    }

    template<> RetVal devValueFunc<complex64, float64>(const ito::DataObject * /*dObj*/, const int /*devTypFlag*/, float64 & /*meanResult*/, float64 & /*devResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "devValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }
    template<> RetVal devValueFunc<complex64, complex64>(const ito::DataObject * /*dObj*/, const int /*devTypFlag*/, float64 & /*meanResult*/, float64 & /*devResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "devValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal devValueFunc<complex128, float64>(const ito::DataObject * /*dObj*/, const int /*devTypFlag*/, float64 & /*meanResult*/, float64 & /*devResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "devValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }
    template<> RetVal devValueFunc<complex128, complex128>(const ito::DataObject * /*dObj*/, const int /*devTypFlag*/, float64 & /*meanResult*/, float64 & /*devResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "devValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal devValueFunc<Rgba32, uint32>(const ito::DataObject * /*dObj*/, const int /*devTypFlag*/, float64 & /*meanResult*/, float64 & /*devResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "devValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }
    template<> RetVal devValueFunc<Rgba32, Rgba32>(const ito::DataObject * /*dObj*/, const int /*devTypFlag*/, float64 & /*meanResult*/, float64 & /*devResult*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "devValueFunc not defined for rgba32 type", "", __FILE__, __LINE__));
        return retError;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! returns mean-value and the standard deviation of the data object
    /*!
        NaN-Values & Inf-Value handling depends on NaN-flag. If ignoreNaN == true, both are ignored else not.
        The function checks if dObj != NULL &&  dObj.type != tComplexXX 

        \param[in]      dObj                handle to the dataObject
        \param[in]      devTypFlag          should be 0 or 1, due to different definitions of standard deviation
        \param[out]     meanResult          float64-type mean value
        \param[out]     devResult           float64-type std-value
        \param[in]      ignoreNaN           Ignore NaN-Values && Inf-Values

        \return retOK
    */
    RetVal devValue(const DataObject *dObj, const int devTypFlag, float64 &meanValue, float64 &devValue, bool ignoreNaN)
    {
        ito::RetVal retval = ito::retOk;

        int dims = dObj->getDims();

        if(dObj == NULL || dims == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128 || dObj->getType() == tRGBA32)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        meanValue = std::numeric_limits<float64>::max();
        devValue = std::numeric_limits<float64>::max();

        switch( dObj->getType() )
        {
        case tUInt8:
        {
            if((dObj->getNumPlanes() * dObj->getSize(dims-1) * dObj->getSize(dims-2)) > 8388000) retval += devValueFunc<uint8, float64>(dObj, devTypFlag, meanValue, devValue, false);
            else retval += devValueFunc<uint8, int32>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tInt8:
        {
            if((dObj->getNumPlanes() * dObj->getSize(dims-1) * dObj->getSize(dims-2)) > 8388000) retval += devValueFunc<int8, float64>(dObj, devTypFlag, meanValue, devValue, false);
            else retval += devValueFunc<int8, int32>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }        
        case tUInt16:
        {
            retval += devValueFunc<uint16, float64>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tInt16:
        {
            retval += devValueFunc<int16, float64>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tUInt32:
        {
            retval += devValueFunc<uint32, float64>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tInt32:
        {
            retval += devValueFunc<int32, float64>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tFloat32:
        {
            retval += devValueFunc<float32, float64>(dObj, devTypFlag, meanValue, devValue, ignoreNaN);
            break;
        }
        case tFloat64:
        {
            retval += devValueFunc<float64, float64>(dObj, devTypFlag, meanValue, devValue, ignoreNaN);
            break;
        }
        default:
        {
            retval += ito::RetVal(retError, 0, "data type not supported");
            break;
        }
        }
        return retval;
    }

    //-----------------------------------------------------------------------------------------------
    //! calculate the 1D-FFT / 1D-IFFT or 2D-FFT / 2D-IFFT of a dataObject
    /*!
        This filter tries to perform an inplace FFT for a given 2D-dataObject. The FFT is calculated linewise or pointwise.

        \param[in|out]  dObjIO              handle to the dataObject. Must be float-type or complex-type
        \param[in]      inverse             toggle between IFFT or FFT
        \param[out]     inverseAsReal       toggle output for the IFFT between real and complex
        \param[out]     lineWise            toggle between 1D-linewise and 2D-FFT

        \return retOK
    */
    RetVal calcCVDFT(DataObject *dObjIO, const bool inverse, const bool inverseAsReal, const bool lineWise)
    {
        unsigned int numMats = dObjIO->getNumPlanes();
        std::string protocol("Applied ");
        ito::RetVal retval(ito::retOk);
        
        bool createNewObj = false;
        bool clearInMat = false;

        if(dObjIO == NULL || dObjIO->getDims() > 2 || numMats != 1)
            return ito::RetVal(ito::retError, 0, "DFT-Error: source object empty or not a single plane");

        int flags = 0;

        if(inverse && inverseAsReal)
        {
            flags += cv::DFT_REAL_OUTPUT;
        }
        else
        {
            flags += cv::DFT_COMPLEX_OUTPUT;
        }
        if(inverse)
        {
            flags += cv::DFT_INVERSE;
            flags += cv::DFT_SCALE;
            protocol.append("inverse ");
        }

        if(lineWise)
        {
            protocol.append("linewise ");
            flags += cv::DFT_ROWS;
        }

        protocol.append("cv::DFT");

        cv::Mat *cvplaneIn = NULL;
        cv::Mat *cvplaneOut = NULL;

        ito::DataObject tempObject;

        if(inverse)
        {
            switch(dObjIO->getType())
            {
                case ito::tComplex64:
                    if(inverseAsReal)
                    {
                        cvplaneIn = (cv::Mat_<complex64> *)(dObjIO->get_mdata())[0];

                        tempObject = ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tFloat32);             
                        dObjIO->copyAxisTagsTo(tempObject);
                        dObjIO->copyTagMapTo(tempObject);
                        cvplaneOut = (cv::Mat_<float32> *)(tempObject.get_mdata())[0];

                        createNewObj = true;
                    }
                    else
                    {
                        cvplaneIn = (cv::Mat_<complex64> *)(dObjIO->get_mdata())[0];
                        cvplaneOut = cvplaneIn;
                    }
                    break;
                case ito::tComplex128:
                    if(inverseAsReal)
                    {
                        cvplaneIn = (cv::Mat_<complex128> *)(dObjIO->get_mdata())[0];

                        tempObject = ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tFloat64);             
                        dObjIO->copyAxisTagsTo(tempObject);
                        dObjIO->copyTagMapTo(tempObject);
                        cvplaneOut = (cv::Mat_<float64> *)(tempObject.get_mdata())[0];
                        
                        createNewObj = true;
                    }
                    else
                    {
                        cvplaneIn = (cv::Mat_<complex128> *)(dObjIO->get_mdata())[0];
                        cvplaneOut = cvplaneIn;
                    }
                    break;
                default:
                    return ito::RetVal(ito::retError, 0, "DFT-Error: object must be complex-type");
            }
        }
        else
        {
            switch(dObjIO->getType())
            {
            
                case ito::tFloat32:
                {
                    cvplaneIn = ((cv::Mat_<float32> *)(dObjIO->get_mdata())[0]);
                    cv::Mat planes[] = {cv::Mat_<ito::float32>(*cvplaneIn), cv::Mat::zeros(cvplaneIn->size(), CV_32F)};
//                    cvplaneIn = NULL;
                    cvplaneIn = new cv::Mat;
                    cv::merge(planes, 2, *cvplaneIn);

                    tempObject = ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tComplex64);             
                    dObjIO->copyAxisTagsTo(tempObject);
                    dObjIO->copyTagMapTo(tempObject);
                    cvplaneOut = (cv::Mat_<complex64> *)(tempObject.get_mdata())[0];
                        
                    createNewObj = true;
                    clearInMat = true;
                }
                break;
                case ito::tFloat64:
                {
                    cvplaneIn = ((cv::Mat_<float64> *)(dObjIO->get_mdata())[0]);
                    cv::Mat planes[] = {cv::Mat_<ito::float64>(*cvplaneIn), cv::Mat::zeros(cvplaneIn->size(), CV_64F)};
//                    cvplaneIn = NULL;
                    cvplaneIn = new cv::Mat;
                    cv::merge(planes, 2, *cvplaneIn);

                    tempObject = ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tComplex128);             
                    dObjIO->copyAxisTagsTo(tempObject);
                    dObjIO->copyTagMapTo(tempObject);
                    cvplaneOut = (cv::Mat_<complex128> *)(tempObject.get_mdata())[0];
                        
                    createNewObj = true;
                    clearInMat = true;
                }
                break;
            
                case ito::tComplex64:
                    cvplaneIn = (cv::Mat_<complex64> *)(dObjIO->get_mdata())[0];
                    cvplaneOut = cvplaneIn;
                    break;
                case ito::tComplex128:
                    cvplaneIn = (cv::Mat_<complex128> *)(dObjIO->get_mdata())[0];
                    cvplaneOut = cvplaneIn;
                    break;
                default:
                    return ito::RetVal(ito::retError, 0, "DFT-Error: object must be complex-type");
            }
        }

        try
        {
            cv::dft(*cvplaneIn, *cvplaneOut, flags);
        }
        catch (cv::Exception &exc)
        {
//            std::string errBuf(exc.err);
#if (CV_MAJOR_VERSION < 3)
            retval += ito::RetVal(ito::retError, 0, exc.err.data());
#else
            retval += ito::RetVal(ito::retError, 0, exc.err.c_str());
#endif
        }

        if((clearInMat == true) && (cvplaneIn != NULL))
        {
            delete cvplaneIn;
            cvplaneIn = NULL;
        }

        if(createNewObj)
        {
            *dObjIO = tempObject;        
        }

        dObjIO->addToProtocol(protocol);
        
        bool test;
        int curDim = dObjIO->getDims()-1;
        std::string axisUnit;
        
        float64 newScale = dObjIO->getAxisScale(curDim);
        if(ito::isFinite<float64>(newScale) && ito::isNotZero<float64>(newScale))
        {
            newScale = 1/newScale / dObjIO->getSize(curDim);
            axisUnit = invertUnit(dObjIO->getAxisUnit(curDim, test));
            dObjIO->setAxisUnit(curDim, axisUnit);
        }
        else
        {
            newScale = 1.0;
            dObjIO->setAxisUnit(curDim, "");
        }
        dObjIO->setAxisScale(curDim, newScale );
        dObjIO->setAxisOffset(curDim, 0.0);

        

        if(!lineWise)
        {
            curDim--;
            newScale = dObjIO->getAxisScale(curDim);
            if(ito::isFinite<float64>(newScale) && ito::isNotZero<float64>(newScale))
            {
                newScale = 1/newScale / dObjIO->getSize(curDim);
                axisUnit = invertUnit(dObjIO->getAxisUnit(curDim, test));
                dObjIO->setAxisUnit(curDim, axisUnit);
            }
            else
            {
                newScale = 1.0;
                dObjIO->setAxisUnit(curDim, "");
            }
            dObjIO->setAxisScale(curDim, newScale );

        }

        return retval;
    }
    
    //-----------------------------------------------------------------------------------------------
    //this is a private function and not exported
    ito::RetVal verifyDataObjectType(const ito::DataObject* dObj, const char* name, uint8 numberOfAllowedTypes, va_list types) //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    {
        if(dObj == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object is NULL.", name);
        }

        if (numberOfAllowedTypes > 12)
        {
            return ito::RetVal::format(ito::retError, 0, "error in 'verifyDataObjectType': numberOfAllowedTypes must be in range [0,12]");
        }

        bool found = false;
        int type = dObj->getType();
        int temp = 0;
        ito::uint8 types_[12];
        memset(types_, 0, 12 * sizeof(ito::uint8));

        for (int i = 0; i < numberOfAllowedTypes; ++i)
        {
            temp = va_arg(types, int); //gcc complains that tio::tDataType is defaulted to int when passed to va_arg so the function call should be with in effectively
            types_[i] = temp;
            if(temp == type)
            {
                found = true;
                break;
            }
        }

        ito::RetVal retValue;

        if(!found)
        {
            char expected[160]; //max 12 types * 12 characters
            expected[0] = '\0';

            for (int i=0; i < numberOfAllowedTypes; ++i)
            {
                switch (types_[i])
                {
                case ito::tUInt8:
                    strcat (expected, "uint8, ");
                    break;
                case ito::tInt8:
                    strcat (expected, "int8, ");
                    break;
                case ito::tUInt16:
                    strcat (expected, "uint16, ");
                    break;
                case ito::tInt16:
                    strcat (expected, "int16, ");
                    break;
                case ito::tUInt32:
                    strcat (expected, "uint32, ");
                    break;
                case ito::tInt32:
                    strcat (expected, "int32, ");
                    break;
                case ito::tFloat32:
                    strcat (expected, "float32, ");
                    break;
                case ito::tFloat64:
                    strcat (expected, "float64, ");
                    break;
                case ito::tComplex64:
                    strcat (expected, "complex64, ");
                    break;
                case ito::tComplex128:
                    strcat (expected, "complex128, ");
                    break;
                case ito::tRGBA32:
                    strcat (expected, "rgba32, ");
                    break;
                }
            }

            size_t len = strlen(expected);
            if (len > 2)
            {
                expected[len - 2] = '\0'; //cut last ', '
            }
            else if (len == 0)
            {
                strcat (expected, "[none]");
            }

            retValue += ito::RetVal::format( ito::retError, 0, "DataObject '%s': wrong type. Expected: %s", name, expected);
        }

        return retValue;
    }

    //-----------------------------------------------------------------------------------------------
    /*!
        This function checks if the dataObject pointer is valid and of the object is of right type.
        If the type is not one of the given types, a specific error message containing the name is returned.

        \param[in]  dObj                    handle to the dataObject, NULL-Pointer is allowed
        \param[in]  name                    the name of the dataObject, will be added to the error message
        \param[in]  numberOfAllowedTypes    number of allowed types appened behind this.
        \param[in]  Allowed types(mul)      A number of additional variabled (number = numberOfAllowedTypes) containing the type definition e.g. ito::tUint8, ito::tInt8... (order does not care)

        \return retOk if valid and handle not NULL, else retError
    */
    ito::RetVal verifyDataObjectType(const ito::DataObject* dObj, const char* name, uint8 numberOfAllowedTypes, ...)
    {
        ito::RetVal retval;
        va_list va;
        va_start(va, numberOfAllowedTypes);
        retval = verifyDataObjectType(dObj, name, numberOfAllowedTypes, va);
        va_end(va);
        return retval;
    };

    //-----------------------------------------------------------------------------------------------
    /*!
        This function checks if the dataObject pointer is valid and of the object is of right type.
        Further more this function checks if the object is truely 2D (dims == 2).
        If the type is not one of the given types, a specific error message containing the name is returned.

        \param[in]  dObj                    handle to the dataObject, NULL-Pointer is allowed
        \param[in]  name                    the name of the dataObject, will be added to the error message
        \param[in]  sizeYMin                the minimum size in y direction
        \param[in]  sizeYMax                the maximum size in y direction
        \param[in]  sizeXMin                the minimum size in x direction
        \param[in]  sizeXMax                the maximum size in x direction
        \param[in]  numberOfAllowedTypes    number of allowed types appened behind this, if zero all types are allowed.
        \param[in]  Allowed types(mul)      A number of additional variables (number = numberOfAllowedTypes) containing the type definition e.g. ito::tUint8, ito::tInt8... (order does not care)

        \return retOk if valid and handle not NULL, else retError
    */
    ito::RetVal verify2DDataObject(const ito::DataObject* dObj, const char* name, int sizeYMin, int sizeYMax, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...)
    {
        if(dObj == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object is NULL.", name);
        }

        if(dObj->getDims() != 2)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object must be two-dimensional.", name);
        }

        ito::RetVal retValue;

        if (numberOfAllowedTypes > 0)
        {
            va_list vl;
            va_start(vl, numberOfAllowedTypes);
            retValue += verifyDataObjectType(dObj, name, numberOfAllowedTypes, vl);
            va_end(vl);
        }

        retValue += verifySize(dObj->getSize(0), sizeYMin, sizeYMax, "y-axis", name);
        retValue += verifySize(dObj->getSize(1), sizeXMin, sizeXMax, "x-axis", name);
    
        return retValue;
    }

    //-----------------------------------------------------------------------------------------------
    /*!
        This function checks if the dataObject pointer is valid and of the object is of right type.
        Further more this function checks if the object is truely 3D (dims == 3).
        If the type is not one of the given types, a specific error message containing the name is returned.

        \param[in]  dObj                    handle to the dataObject, NULL-Pointer is allowed
        \param[in]  name                    the name of the dataObject, will be added to the error message
        \param[in]  sizeZMin                the minimum size in z direction
        \param[in]  sizeZMax                the maximum size in z direction
        \param[in]  sizeYMin                the minimum size in y direction
        \param[in]  sizeYMax                the maximum size in y direction
        \param[in]  sizeXMin                the minimum size in x direction
        \param[in]  sizeXMax                the maximum size in x direction
        \param[in]  numberOfAllowedTypes    number of allowed types appened behind this, if zero all types are allowed.
        \param[in]  Allowed types(mul)      A number of additional variabled (number = numberOfAllowedTypes) containing the type definition e.g. ito::tUint8, ito::tInt8... (order does not care)

        \return retOk if valid and handle not NULL, else retError
    */
    ito::RetVal verify3DDataObject(const ito::DataObject* dObj, const char* name, int sizeZMin, int sizeZMax, int sizeYMin, int sizeYMax, int sizeXMin, int sizeXMax, uint8 numberOfAllowedTypes, ...)
    {
        if(dObj == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object is NULL.", name);
        }

        if(dObj->getDims() != 3)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object must be three-dimensional.", name);
        }

        ito::RetVal retValue;

        if (numberOfAllowedTypes > 0)
        {
            va_list vl;
            va_start(vl, numberOfAllowedTypes);
            retValue += verifyDataObjectType(dObj, name, numberOfAllowedTypes, vl);
            va_end(vl);
        }

        retValue += verifySize(dObj->getSize(0), sizeZMin, sizeZMax, "z-axis", name);
        retValue += verifySize(dObj->getSize(1), sizeYMin, sizeYMax, "y-axis", name);
        retValue += verifySize(dObj->getSize(2), sizeXMin, sizeXMax, "x-axis", name);
    
        return retValue;
    }

    //-----------------------------------------------------------------------------------------------
    ito::RetVal verifySize(int size, int minSize, int maxSize, const char *axisName, const char* dObjName)
    {
        if(minSize < 0) minSize = 0;
        if(maxSize == -1)
        {
            if(size < (int)minSize)
            {
                return ito::RetVal::format(ito::retError, 0, "DataObject '%s': size of %s must be %i or bigger.", dObjName, axisName, minSize);
            }
        }
        else
        {
            if(maxSize < minSize) std::swap(minSize, maxSize);
            if(size < (int)minSize || size > (int)maxSize)
            {
                return ito::RetVal::format(ito::retError, 0, "DataObject '%s': size of %s must be between %i and %i.", dObjName, axisName, minSize, maxSize);
            }
        }
        return ito::retOk;
    }

    //-----------------------------------------------------------------------------------------------
    /*!
        Use this method to test an incoming data object to be two-dimensional, have a certain width and height and types.

        If the data object has more than two dimensions, it is squeezed at the beginning and must than contain two dimensions.
        In this case a two-dimensional shallow copy is returned (hence, the squeezed version). If convertToType is != 0, the
        dataObject is finally converted to the desired new type (in terms of ito::tDataType), if it is 0, it stays like it is.

        Use the variable number of arguments at the end to pass all type enumeration that are allowed for the incoming data object.
        numberOfAllowedTypes indicates the number of following type arguments. Then pass all allowed types (ito::tDataType) as unique arguments.
        Don't pass an or combination of types, since they are not organized as bitmask. If numberOfAllowedTypes is equal to zero, all types
        are accepted.

        \param dObj the input data object
        \param name is the name of the data object (for error messages only)
        \param sizeX is the allowed range for the width of the 2d data object (start and end are included)
        \param sizeY is the allowed range for the height of the 2d data object (start and end are included)
        \param retval is the return value
        \param convertToType is the type the data object should be converted to (ito::tInt8, ito::tFloat32...) or -1 if no conversion should be done (changed after itom 1.4.0, was 0 before which stands for int8)
        \param numberOfAllowedTypes is the number of types that are allowed for the incoming object or 0 if all types are allowed
        \param ... is the list of allowed types in terms of ito::tDataType added as multiple arguments (their number must correspond to numberOfAllowedTypes)
        \return shallow or deep copy of the (squeezed) input data object.
    */
    ito::DataObject squeezeConvertCheck2DDataObject(const ito::DataObject *dObj, const char* name, const ito::Range &sizeY, const ito::Range &sizeX, ito::RetVal &retval, int convertToType, uint8 numberOfAllowedTypes, ...)
    {
        ito::DataObject out;

        if(dObj == NULL)
        {
            retval += ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object is NULL.", name);
        }
        else if (numberOfAllowedTypes > 0)
        {
            va_list vl;
            va_start(vl, numberOfAllowedTypes);
            retval += verifyDataObjectType(dObj, name, numberOfAllowedTypes, vl);
            va_end(vl);
        }

        if (!retval.containsError())
        {
            
            if (dObj->getDims() == 2)
            {
                //size checking
                int width = dObj->getSize(1);
                int height = dObj->getSize(0);

                if (width < sizeX.start || width > sizeX.end)
                {
                    retval += ito::RetVal::format( ito::retError, 0, "sizeX of dataObject '%s' out of range [%i,%i]", name, sizeX.start, sizeX.end);
                }
                else if (height < sizeY.start || height > sizeY.end)
                {
                    retval += ito::RetVal::format( ito::retError, 0, "sizeY of dataObject '%s' out of range [%i,%i]", name, sizeY.start, sizeY.end);
                }
                else
                {
                    //check if it needs to be converted
                    if (convertToType >= 0 && (convertToType != dObj->getType()))
                    {
                        retval += dObj->convertTo(out, convertToType);
                    }
                    else //take as is
                    {
                        out = *dObj;
                    }
                }
            }
            else if (dObj->getDims() > 2)
            {
                //check that the first dims-2 dimensions are 1
                out = dObj->squeeze();

                if (out.getDims() == 2)
                {
                    //size checking
                    int width = out.getSize(1);
                    int height = out.getSize(0);

                    if (width < sizeX.start || width > sizeX.end)
                    {
                        retval += ito::RetVal::format( ito::retError, 0, "sizeX of dataObject '%s' out of range [%i,%i]", name, sizeX.start, sizeX.end);
                    }
                    else if (height < sizeY.start || height > sizeY.end)
                    {
                        retval += ito::RetVal::format( ito::retError, 0, "sizeY of dataObject '%s' out of range [%i,%i]", name, sizeY.start, sizeY.end);
                    }
                    else
                    {
                        //check if it needs to be converted
                        if (convertToType > 0 && convertToType != out.getType())
                        {
                            ito::DataObject out2;
                            retval += out.convertTo(out2, convertToType); //non-inplace conversion
                            out = out2;
                        }
                        else {} //take as is
                    }
                }
                else
                {
                    retval += ito::RetVal::format( ito::retError, 0, "DataObject '%s' must have 2 dimensions (squeezed)", name);
                }
            }
            else
            {
                retval += ito::RetVal::format( ito::retError, 0, "DataObject '%s' must have 2 dimensions (squeezed)", name);
            }
        }
    
        return out;
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------
    //! returns a shallow or deep copy of a given data object that fits to given requirements
    /*!
        Use this simple api method to test a given data object if it fits some requirements.
        If this is the case, a shallow copy of the input data object is returned. Else, it is
        tried to convert into the required object and a converted deep copy is returned. If the
        input object does not fit the given requirements, NULL is returned and the ito::RetVal
        parameter contains an error status including error message.

        \note In any case you need to delete the returned data object

        \param dObj the input data object
        \param nrDims the required number of dimensions
        \param type the required type of the returned data object
        \param name name of the data object for an improved error message (zero-terminated string) or NULL if no name is known.
        \param sizeLimits can be NULL if the sizes should not be checked, else it is an array with length (2*nrDims). Every adjacent pair describes the minimum and maximum size of each dimension.
        \param retval can be a pointer to an instance of ito::RetVal or NULL. If given, the status of this method is added to this return value.
        \return shallow or deep copy of the input data object or NULL (in case of unsolvable incompatibility)
    */
    ito::DataObject* createFromNamedDataObject(const ito::DataObject *dObj, int nrDims, ito::tDataType type, const char *name /*= NULL*/, int *sizeLimits /*= NULL*/, ito::RetVal *retval /*= NULL*/)
    {
        ito::DataObject *output = NULL;
        ito::RetVal ret;

        if (dObj)
        {
            if (dObj->getDims() != nrDims)
            {
                if (name)
                {
                    ret += ito::RetVal::format(ito::retError, 0, "The data object '%s' must have %i dimensions (%i given)", name, nrDims, dObj->getDims());
                }
                else
                {
                    ret += ito::RetVal::format(ito::retError, 0, "The given data object must have %i dimensions (%i given)", nrDims, dObj->getDims());
                }
            }
            else if(sizeLimits) //check sizeLimits (must be twice as lang as nrDims)
            {
                for (int i = 0; i < nrDims; ++i)
                {
                    int s = dObj->getSize(i);
                    if (s < sizeLimits[i * 2] || s > sizeLimits[i * 2 + 1])
                    {
                        if (name)
                        {
                            ret += ito::RetVal::format(ito::retError, 0, "The size of the %i. dimension of data object '%s' exceeds the given boundaries [%i, %i]", i+1, name, sizeLimits[i * 2], sizeLimits[i * 2 + 1]);
                        }
                        else
                        {
                            ret += ito::RetVal::format(ito::retError, 0, "The size of the %i. dimension exceeds the given boundaries [%i, %i]", i+1, sizeLimits[i * 2], sizeLimits[i * 2 + 1]);
                        }
                        break;
                    }
                }
            }

            if (!ret.containsError())
            {
                if (dObj->getType() == type)
                {
                    output = new ito::DataObject(*dObj);
                }
                else
                {
                    output = new ito::DataObject();
                    ret += dObj->convertTo(*output, type);
                }
            }
        }

        if (ret.containsError())
        {
            DELETE_AND_SET_NULL(output);
        }

        if (retval) *retval += ret;
        return output;
    }

    //-----------------------------------------------------------------------------------------------
    /*! \fn dObjCopyLastNAxisTags 
        \detail  Helperfunction to copy axis related tags from a n-D-Object to a m-D-Object.
        \param[in]       DataObjectIn   Input-Object / Source
        \param[in|out]   DataObjectOut  Preallocated Output-Object / Destination
        \param[in]       copyLastNDims  Number of dimensions to copy, counted from the last Dimension
        \param[in]       includeValueTags  Toggle copying of value tags (default = true)
        \param[in]       includeRotationMatrix  Toggle copying of rotation matrix-meta-data (default = true)

        \return ito::retOk or ito::retError
        \author  Lyda
        \sa  
        \date    03.2013 
    */
    ito::RetVal dObjCopyLastNAxisTags(const ito::DataObject &DataObjectIn, ito::DataObject &DataObjectOut, const int copyLastNDims, const bool includeValueTags, const bool includeRotationMatrix)
    {
        int nDimOut = DataObjectOut.getDims();
        int nDimIn = DataObjectIn.getDims();

        if(nDimOut < 2)
        {
            return ito::RetVal(ito::retError, 0, "dataObjectOut is empty");
        }
        if(nDimIn < 2)
        {
            return ito::RetVal(ito::retError, 0, "DataObjectIn is empty");
        }

        if(nDimOut < copyLastNDims )
        {
            return ito::RetVal(ito::retError, 0, "Requested dimensions exceeded number of dataObjectOut dims");
        }
        if(nDimIn < copyLastNDims )
        {
            return ito::RetVal(ito::retError, 0, "Requested dimensions exceeded number of dataObjectIn dims");
        }

        bool test;

        for(int i = copyLastNDims; i > 0; i--)
        {
            DataObjectOut.setAxisDescription(nDimOut - i, DataObjectIn.getAxisDescription(nDimIn - i, test));
            DataObjectOut.setAxisUnit(nDimOut - i, DataObjectIn.getAxisUnit(nDimIn - i, test));
            DataObjectOut.setAxisOffset(nDimOut - i, DataObjectIn.getAxisOffset(nDimIn - i));
            DataObjectOut.setAxisScale(nDimOut - i, DataObjectIn.getAxisScale(nDimIn - i ));
        }
        
        if(includeValueTags)
        {
            DataObjectOut.setValueDescription(DataObjectIn.getValueDescription());
            DataObjectOut.setValueUnit(DataObjectIn.getValueUnit());            
        }

        if(includeRotationMatrix)
        {
            double r11, r12, r13, r21, r22, r23, r31, r32, r33;
            DataObjectIn.getXYRotationalMatrix(r11, r12, r13, r21, r22, r23, r31, r32, r33);
            DataObjectOut.setXYRotationalMatrix(r11, r12, r13, r21, r22, r23, r31, r32, r33);
        }
        
        return ito::retOk;
    }

    ito::RetVal dObjSetScaleRectangle(ito::DataObject &DataObjectInOut, const double &x0, const double &x1, const double &y0, const double &y1)
    {
        ito::int32 dims = DataObjectInOut.getDims();
        if(dims > 2)
        {
            return ito::RetVal(ito::retError, 0, "Set scale and offset to rectangle failed due to invalid dataObject size. Object to small.");
        }

        ito::float64 scaleX = (x1 - x0);
        ito::float64 scaleY = (y1 - y0);
        if(!ito::isNotZero(scaleX) || !ito::isNotZero(scaleY))
        {
            return ito::RetVal(ito::retError, 0, "Set scale and offset to rectangle failed due to zero scale error of at least one scale");
        }

        if(DataObjectInOut.getSize(dims - 1) > 1) scaleX /= (DataObjectInOut.getSize(dims - 1) - 1);  
        if(DataObjectInOut.getSize(dims - 2) > 1) scaleY /= (DataObjectInOut.getSize(dims - 2) - 1);  
        
        ito::float64 offset = x0 / scaleX;
        DataObjectInOut.setAxisScale(dims -1, scaleX);
        DataObjectInOut.setAxisOffset(dims -1, offset);

        offset = y0 / scaleY;
        DataObjectInOut.setAxisScale(dims -2, scaleY);
        DataObjectInOut.setAxisOffset(dims -2, offset);

        return ito::retOk;
    }



}
}
