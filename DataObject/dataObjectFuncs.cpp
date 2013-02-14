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

#include "dataObjectFuncs.h"

namespace ito 
{
namespace dObjHelper
{

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
        unsigned int numMats = dObj->calcNumMats();
        size_t matIndex = 0;
        int m,n;

        cv::Mat_<_Tp> *mat = NULL;
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
                    rowPtr = (_Tp*)mat->ptr(m);
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
                    rowPtr = (_Tp*)mat->ptr(m);
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
            return ito::RetVal(retError, 0, "Location pointer is invalid for Minimum");

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128)
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
       unsigned int numMats = dObj->calcNumMats();
       size_t matIndex = 0;
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
                    rowPtr = (_Tp*)mat->ptr(m);
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
                    rowPtr = (_Tp*)mat->ptr(m);
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

    template<> RetVal maxValueFunc<complex64>(const DataObject * /*dObj*/, float64 & /*maxValue*/, uint32 * /*firstLocation*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "maxValueFunc not defined for complex type", "", __FILE__, __LINE__));
        return retError;
    }

    template<> RetVal maxValueFunc<complex128>(const DataObject * /*dObj*/, float64 & /*maxValue*/, uint32 * /*firstLocation*/, bool /*ignoreNaN*/)
    {
        cv::error(cv::Exception(CV_StsAssert, "maxValueFunc not defined for complex type", "", __FILE__, __LINE__));
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
                This function has no complex 

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
            return ito::RetVal(retError, 0, "Location pointer is invalid");

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        return fListmaxValueFunc[dObj->getType()](dObj, maxValue, firstLocation, ignoreInf);
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /*!
    \detail This function searches for min and max-value of the <_Type> data object and saves its first detects positions in firstMinLocation (must be uint32[3]-Array) and firstMaxLocation (must be  uint32[3]-Array).
            NaN-Values will be ignored by this method and Inf-Value handling depends on inf-flag. If ignoreInf == true, inf is ignored else inf is not ignored. 
            The cmplxSel for complex handling selection is unused in the FP and INT-versions of the function.
            Warning, does not check if dObj && firstMinLocation && firstMaxLocation are valid!

        \param[in]      dObj                handle to the dataObject
        \param[out]     minValue            lowest value in this object
        \param[in|out]  firstMinLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue            highest value in this object
        \param[in|out]  firstMaxLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf           Ignore Inf-Values
        \param[in]      cmplxSel            Toggle complex handling (not used) 

        \return retOk
    */
    template<typename _Tp> RetVal minMaxValueFunc(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int /*cmplxSel*/)
    {
        size_t numMats = dObj->calcNumMats();
        size_t matIndex = 0;

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
            for (size_t nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = (_Tp*)mat->ptr(m);
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
            for (size_t nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

                for(m = 0; m < mat->rows; m++)
                {
                    rowPtr = (_Tp*)mat->ptr(m);
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

        \param[in]      dObj                handle to the dataObject
        \param[out]     minValue            lowest value in this object
        \param[in|out]  firstMinLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[out]     maxValue            highest value in this object
        \param[in|out]  firstMaxLocation    Allocated uint32[3]-array. Will be filled with [mat-Number, ymin, xmin]
        \param[in]      ignoreInf           Ignore Inf-Values
        \param[in]      cmplxSel            Toggle complex handling, 0:abs-Value, 1:imaginary-Value, 2:real-Value, 3: argument-Value

        \return retOk
    */
    template<> RetVal minMaxValueFunc<ito::complex64>(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int cmplxSel)
    {
        size_t numMats = dObj->calcNumMats();
        size_t matIndex = 0;

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
            for (uint32 nmat = 0; nmat < (uint32)numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex64> *)(dObj->get_mdata())[matIndex];
                cols = (uint32)mat->cols;
                rows = (uint32)mat->rows;
			
			    switch (cmplxSel)
			    {
				    default:	
				    case 0:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

				    case 1:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

				    case 2:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

				    case 3:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

			    switch (cmplxSel)
			    {
				    default:	
				    case 0:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

				    case 1:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

				    case 2:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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

				    case 3:
					    for(m = 0; m < rows; m++)
					    {
						    rowPtr = (ito::complex64*)mat->ptr(m);
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
        \param[in]      cmplxSel            Toggle complex handling, 0:abs-Value, 1:imaginary-Value, 2:real-Value, 3: argument-Value

        \return retOk
    */

    template<> RetVal minMaxValueFunc<ito::complex128>(const DataObject *dObj, float64 &minValue, uint32 * /*firstMinLocation*/, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int cmplxSel)
    {
        size_t numMats = dObj->calcNumMats();
        size_t matIndex = 0;

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
            for (size_t nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex128> *)(dObj->get_mdata())[matIndex];
			
			    switch (cmplxSel)
			    {
				    default:	
				    case 0:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

				    case 1:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

				    case 2:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

				    case 3:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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
            for (size_t nmat = 0; nmat < numMats; nmat++)
            {
                matIndex = dObj->seekMat(nmat, numMats);
                mat = (cv::Mat_<ito::complex128> *)(dObj->get_mdata())[matIndex];

			    switch (cmplxSel)
			    {
				    default:	
				    case 0:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

				    case 1:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

				    case 2:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

				    case 3:
					    for(m = 0; m < mat->rows; m++)
					    {
						    rowPtr = (ito::complex128*)mat->ptr(m);
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

    typedef RetVal (*tminMaxValueFunc)(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int cmplxSel);
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
    RetVal minMaxValue(const DataObject *dObj, float64 &minValue, uint32 *firstMinLocation, float64 &maxValue, uint32 *firstMaxLocation, bool ignoreInf, const int cmplxSel)
    {
        minValue = std::numeric_limits<float64>::max();
        maxValue = -std::numeric_limits<float64>::max();

        if(firstMinLocation == NULL)
            return ito::RetVal(retError, 0, "Location pointer is invalid for Minimum");

        if(firstMaxLocation == NULL)
            return ito::RetVal(retError, 0, "Location pointer is invalid for Minimum");

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        return fListminMaxValueFunc[dObj->getType()](dObj, minValue, firstMinLocation, maxValue, firstMaxLocation, ignoreInf, cmplxSel);
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
        unsigned int numMats = dObj->calcNumMats();
        size_t matIndex = 0;

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
        if(nrOfValidElements <= 0) nrOfValidElements = 1; //in order to avoid divide-by-zero crash

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

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        meanResult = std::numeric_limits<float64>::max();

        switch( dObj->getType() )
        {
        case tUInt8:
        {
            retval += meanValueFunc<uint8, int32>(dObj, meanResult, false);
            break;
        }
        case tInt8:
        {
            retval += meanValueFunc<int8, int32>(dObj, meanResult, false);
            break;
        }        
        case tUInt16:
        {
            retval += meanValueFunc<uint16, int32>(dObj, meanResult, false);
            break;
        }
        case tInt16:
        {
            retval += meanValueFunc<int16, int32>(dObj, meanResult, false);
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
        unsigned int numMats = dObj->calcNumMats();
        size_t matIndex = 0;

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

        if(nrOfValidElements <= 0) nrOfValidElements = 1; //in order to avoid divide-by-zero crash

        float64 meanValue = static_cast<float64>(sum) / nrOfValidElements;

        float64 devValue = 0.0;
        float64 dev = 0.0;

        if(nrOfValidElements > 1)
        {
            nrOfValidElements = 0;
            float64 temp = 0.0;
            if(ignoreNaN)
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

            if(nrOfValidElements > 2)
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

        if(dObj == NULL || dObj->getDims() == 0)
            return ito::RetVal(retError, 0, "DataObjectPointer is invalid");

        if(dObj->getType() == tComplex64 || dObj->getType() == tComplex128)
        {
            return ito::RetVal(retError, 0, "source matrix must be of type (u)int8, (u)int16, (u)int32, float32 or float64");
        }

        meanValue = std::numeric_limits<float64>::max();
        devValue = std::numeric_limits<float64>::max();

        switch( dObj->getType() )
        {
        case tUInt8:
        {
            retval += devValueFunc<uint8, int32>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tInt8:
        {
            retval += devValueFunc<int8, int32>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }        
        case tUInt16:
        {
            retval += devValueFunc<uint16, int32>(dObj, devTypFlag, meanValue, devValue, false);
            break;
        }
        case tInt16:
        {
            retval += devValueFunc<int16, int32>(dObj, devTypFlag, meanValue, devValue, false);
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
    RetVal calcCVDFT(DataObject *dObjIO, const bool inverse, const bool inverseAsReal, const bool lineWise)
    {
        unsigned int numMats = dObjIO->calcNumMats();
        std::string protocol("Applied ");
        ito::RetVal retval(ito::retOk);
        
        bool createNewObj = false;
        bool createNewInputMat = false;

        if(dObjIO == NULL || dObjIO->getDims() > 2 || numMats != 1)
            return ito::RetVal(ito::retError, 0, "DFT-Error: source object empty or not a single plane");

        int flags = cv::DFT_COMPLEX_OUTPUT;

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

        ito::DataObject *tempObject = NULL;

        if(inverse)
        {
            switch(dObjIO->getType())
            {
                case ito::tComplex64:
                    if(inverseAsReal)
                    {
                        cvplaneIn = (cv::Mat_<complex64> *)(dObjIO->get_mdata())[0];

                        tempObject = new ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tFloat32);             
                        dObjIO->copyAxisTagsTo(*tempObject);
                        dObjIO->copyTagMapTo(*tempObject);
                        cvplaneOut = (cv::Mat_<float32> *)(tempObject->get_mdata())[0];

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

                        tempObject = new ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tFloat64);             
                        dObjIO->copyAxisTagsTo(*tempObject);
                        dObjIO->copyTagMapTo(*tempObject);
                        cvplaneOut = (cv::Mat_<float64> *)(tempObject->get_mdata())[0];
                        
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
                    cvplaneIn = (cv::Mat_<float32> *)(dObjIO->get_mdata())[0];

                    tempObject = new ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tComplex64);             
                    dObjIO->copyAxisTagsTo(*tempObject);
                    dObjIO->copyTagMapTo(*tempObject);
                    cvplaneOut = (cv::Mat_<complex64> *)(tempObject->get_mdata())[0];
                        
                    createNewObj = true;
                    createNewInputMat = true;

                break;
                case ito::tFloat64:
                    cvplaneIn = new cv::Mat;
                    cv::merge((cv::Mat_<float64> *)(dObjIO->get_mdata())[0], 2, *cvplaneIn);

                    cvplaneIn = (cv::Mat_<float64> *)(dObjIO->get_mdata())[0];

                    tempObject = new ito::DataObject(dObjIO->getDims(), dObjIO->getSize(), ito::tComplex128);             
                    dObjIO->copyAxisTagsTo(*tempObject);
                    dObjIO->copyTagMapTo(*tempObject);
                    cvplaneOut = (cv::Mat_<complex128> *)(tempObject->get_mdata())[0];
                        
                    createNewObj = true;
                    createNewInputMat = true;
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
        catch (cv::Exception exc)
        {
            std::string errBuf(exc.err);
            retval += ito::RetVal(ito::retError, 0, exc.err.data());
        }


        if(createNewObj)
        {
            *dObjIO = *tempObject;        
        }

        dObjIO->addToProtocol(protocol);
        
        bool test;
        std::string axisUnit = invertUnit(dObjIO->getAxisUnit(dObjIO->getDims()-1, test, false));
        dObjIO->setAxisUnit(dObjIO->getDims()-1, axisUnit, false);
        dObjIO->setAxisOffset(dObjIO->getDims()-1, 0.0, false);

        float64 newScale = dObjIO->getAxisScale(dObjIO->getDims()-1, false);
        if(ito::dObjHelper::isFinite<float64>(newScale) && ito::dObjHelper::isNotZero<float64>(newScale))
        {
            newScale = 1/newScale;
        }
        else
        {
            newScale = 1.0;
        }
        dObjIO->setAxisScale(dObjIO->getDims()-1, newScale , false);

        if(!lineWise)
        {
            newScale = dObjIO->getAxisScale(dObjIO->getDims()-2, false);
            if(ito::dObjHelper::isFinite<float64>(newScale) && ito::dObjHelper::isNotZero<float64>(newScale))
            {
                newScale = 1/newScale;
            }
            else
            {
                newScale = 1.0;
            }
            dObjIO->setAxisScale(dObjIO->getDims()-2, newScale , false);

            axisUnit = invertUnit(dObjIO->getAxisUnit(dObjIO->getDims()-2, test, false));
            dObjIO->setAxisUnit(dObjIO->getDims()-2, axisUnit, false);
        }

        return retval;
    }
    
    //-----------------------------------------------------------------------------------------------
    ito::RetVal verifyDataObjectType(const ito::DataObject* dObj, const char* name, uint8 numberOfAllowedTypes, ...) //append allowed data types, e.g. ito::tUint8, ito::tInt8... (order does not care)
    {
        if(dObj == NULL)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object is NULL.", name);
        }

        if(dObj->getDims() < 2)
        {
            return ito::RetVal::format(ito::retError, 0, "DataObject '%s': data object must be at least 2D-dimensional.", name);
        }

        bool found = false;
        int type = dObj->getType();
        int temp = 0;
        va_list vl;
        va_start(vl, numberOfAllowedTypes);
        for(int i = 0; i < numberOfAllowedTypes; i++)
        {
            temp = va_arg(vl, int); //gcc complains that tio::tDataType is defaulted to int when passed to va_arg so the function call should be with in effectively
            if(temp == type)
            {
                found = true;
                break;
            }
        }
        va_end(vl);

        if(!found)
        {
            return ito::RetVal::format( ito::retError, 0, "DataObject '%s': wrong type", name);
        }

        ito::RetVal retValue;
    
        return retValue;
    }

    //-----------------------------------------------------------------------------------------------
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

        bool found = false;
        int type = dObj->getType();
        int temp = 0;
        va_list vl;
        va_start(vl, numberOfAllowedTypes);
        for(int i=0;i<numberOfAllowedTypes;i++)
        {
//            temp = va_arg(vl, ito::tDataType);
            temp = va_arg(vl, int); //gcc complains that tio::tDataType is defaulted to int when passed to va_arg so the function call should be with in effectively
            if(temp == type)
            {
                found = true;
                break;
            }
        }
        va_end(vl);

        if(!found)
        {
            return ito::RetVal::format( ito::retError, 0, "DataObject '%s': wrong type", name);
        }

        ito::RetVal retValue;
        retValue += verifySize(dObj->getSize(0,true), sizeYMin, sizeYMax, "y-axis", name);
        retValue += verifySize(dObj->getSize(1,true), sizeXMin, sizeXMax, "x-axis", name);
    
        return retValue;
    }

    //-----------------------------------------------------------------------------------------------
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

        bool found = false;
        int type = dObj->getType();
        int temp = 0;
        va_list vl;
        va_start(vl, numberOfAllowedTypes);
        for(int i=0;i<numberOfAllowedTypes;i++)
        {
            temp = va_arg(vl,int);
            if(temp == type)
            {
                found = true;
                break;
            }
        }
        va_end(vl);

        if(!found)
        {
            return ito::RetVal::format( ito::retError, 0, "DataObject '%s': wrong type", name);
        }

        ito::RetVal retValue;
        retValue += verifySize(dObj->getSize(0,true), sizeZMin, sizeZMax, "z-axis", name);
        retValue += verifySize(dObj->getSize(1,true), sizeYMin, sizeYMax, "y-axis", name);
        retValue += verifySize(dObj->getSize(2,true), sizeXMin, sizeXMax, "x-axis", name);
    
        return retValue;
    }

    //-----------------------------------------------------------------------------------------------
    ito::RetVal verifySize(size_t size, int minSize, int maxSize, const char *axisName, const char* dObjName)
    {
        if(minSize < 0) minSize = 0;
        if(maxSize == -1)
        {
            if(size < (size_t)minSize)
            {
                return ito::RetVal::format(ito::retError, 0, "DataObject '%s': size of %s must be %i or bigger.", dObjName, axisName, minSize);
            }
        }
        else
        {
            if(maxSize < minSize) std::swap(minSize, maxSize);
            if(size < (size_t)minSize || size > (size_t)maxSize)
            {
                return ito::RetVal::format(ito::retError, 0, "DataObject '%s': size of %s must be between %i and %i.", dObjName, axisName, minSize, maxSize);
            }
        }
        return ito::retOk;
    }
    //-----------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------
    /*! \fn minMaxValueHelper 
        \brief  Helpfunction to find min and maxValue in a dataObject
                The function searches either in all planes (matNumber < 0) or in one specific plane 
        \param[in]   dObj    dataObject
        \param[out]  min    Minimal value
        \param[out]  max    Minimal value
        \param[in]   matNumber   Specific mat to search in one or all planes (if < 0)
        \author  Lyda
        \sa  
        \date    12.2011 
    */
    /*
    template<typename _Tp> void minMaxValueHelper(ito::DataObject *dObj, float64 *min, float64 *max, int matNumber)
    {
        unsigned int numMats = 0;
        unsigned int matIndex = 0;
        unsigned int rhsMatNum = 0;
        unsigned int resMatNum = 0;
        unsigned int matStart = 0;

        if(matNumber < 0)
        {
            numMats = dObj->calcNumMats();
            matStart = 0;

        }
        else
        {
            numMats = (unsigned int)matNumber +1;
            matStart = (unsigned int)matNumber;
        }

        int m,n;

        cv::Mat_<_Tp> *mat = NULL;
        const _Tp* rowPtr;

        _Tp resultmax;

        if(std::numeric_limits<_Tp>::is_exact)
        {
            resultmax = std::numeric_limits<_Tp>::min(); //integer numbers
        }
        else
        {
            resultmax = -std::numeric_limits<_Tp>::max();
        }

        _Tp resultmin = std::numeric_limits<_Tp>::max();

        for (unsigned int nmat = matStart; nmat < numMats; nmat++)
        {
            matIndex = dObj->seekMat(nmat, numMats);
            mat = (cv::Mat_<_Tp> *)(dObj->get_mdata())[matIndex];

            for(m = 0; m < mat->rows; m++)
            {
                rowPtr = (_Tp*)mat->ptr(m);
                for(n = 0; n < mat->cols; n++)
                {
                    if(rowPtr[n] > resultmax) resultmax = rowPtr[n];
                    if(rowPtr[n] < resultmin) resultmin = rowPtr[n];
                }
            }
        }

        *max = (static_cast<float64>(resultmax));
        *min = (static_cast<float64>(resultmin));
    }
    */

    /*
    template<> void GetHLineL<int8>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<uint8>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<int16>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<uint16>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<uint32>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<float32>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<float64>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<complex64>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);
    template<> void GetHLineL<complex128>(const cv::Mat *srcPlane, const int x0, const int y, const int length, int32 * pData);

    template<> void GetHLineD<int8>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<uint8>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<int16>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<uint16>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<int32>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<uint32>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<float32>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<complex64>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);
    template<> void GetHLineD<complex128>(const cv::Mat *srcPlane, const int x0, const int y, const int length, float64 * pData);

    template<> void SetHLineL<int8>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<uint8>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<int16>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<uint16>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<uint32>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<float32>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<float64>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<complex64>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);
    template<> void SetHLineL<complex128>(cv::Mat *destPlane, const int x0, const int y, const int length, const int32 * pData);

    template<> void SetHLineD<int8>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<uint8>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<int16>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<uint16>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<int32>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<uint32>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<float32>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<complex64>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    template<> void SetHLineD<complex128>(cv::Mat *destPlane, const int x0, const int y, const int length, const float64 * pData);
    */

}
}
