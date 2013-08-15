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

//#include <crtdbg.h>
#include "dataobj.h"
#include <cmath>

namespace ito {

//-------------------------------------------------------------------------  
//! default constructor
DObjConstIterator::DObjConstIterator() :
    dObj(NULL),
    elemSize(0),
    ptr(NULL),
    sliceStart(NULL),
    sliceEnd(NULL),
    planeContinuous(false),
    plane(0)
{
}

//-------------------------------------------------------------------------  
//! constructor that sets the iterator to the beginning of the matrix 
DObjConstIterator::DObjConstIterator(const DataObject* _dObj, int pos /*= 0*/):
    dObj(_dObj)
{
    elemSize = _dObj->elemSize();
    size_t dims = dObj->getDims();

    if(dObj->getSize(dims-1) == dObj->getOriginalSize(dims-1))
    {
        planeContinuous = true;
    }
    else
    {
        planeContinuous = false;
    }

    seekAbs(pos);
}

//-------------------------------------------------------------------------  
//! copy constructor
DObjConstIterator::DObjConstIterator(const DObjConstIterator& it)
{
    dObj = it.dObj;
    planeContinuous = it.planeContinuous;
    elemSize = it.elemSize;
    ptr = it.ptr; //current pointer to current element
    sliceStart = it.sliceStart;
    sliceEnd = it.sliceEnd;
    plane = it.plane;
}

//-------------------------------------------------------------------------  
//! copy operator
DObjConstIterator& DObjConstIterator::operator = (const DObjConstIterator& it)
{
    dObj = it.dObj;
    planeContinuous = it.planeContinuous;
    elemSize = it.elemSize;
    ptr = it.ptr; //current pointer to current element
    sliceStart = it.sliceStart;
    sliceEnd = it.sliceEnd;
    plane = it.plane;
    return *this;
}

//-------------------------------------------------------------------------   
//! returns the current matrix element
uchar* DObjConstIterator::operator *() const
{
    return ptr;
}

//-------------------------------------------------------------------------  
//! returns the i-th matrix element, relative to the current
uchar* DObjConstIterator::operator [](int i) const
{
    DObjConstIterator it2(*this);
    it2+=i;
    return (*it2);
}
    

//-------------------------------------------------------------------------  
//! shifts the iterator forward by the specified number of elements
DObjConstIterator& DObjConstIterator::operator += (int ofs)
{
    if( !dObj || ofs == 0 )
        return *this;
    int ofsb = ofs*elemSize;
    ptr += ofsb;
    if( ptr < sliceStart || sliceEnd <= ptr )
    {
        ptr -= ofsb;
        seekRel(ofs);
    }
    return *this;
}

//-------------------------------------------------------------------------  
//! shifts the iterator backward by the specified number of elements
DObjConstIterator& DObjConstIterator::operator -= (int ofs)
{
    return (*this) += -ofs;
}

//-------------------------------------------------------------------------  
//! decrements the iterator
DObjConstIterator& DObjConstIterator::operator --()
{
    if( dObj && (ptr -= elemSize) < sliceStart )
    {
        ptr += elemSize;
        seekRel(-1); 
    }
    return *this;
}

//-------------------------------------------------------------------------  
//! decrements the iterator
DObjConstIterator DObjConstIterator::operator --(int)
{
    DObjConstIterator b = *this;
    *this -= 1;
    return b;
}

//-------------------------------------------------------------------------  
//! increments the iterator
DObjConstIterator& DObjConstIterator::operator ++()
{
    if(dObj && (ptr += elemSize) >= sliceEnd )
    {
        ptr -= elemSize;
        seekRel(1); 
    }
    return *this;
}
   
//-------------------------------------------------------------------------   
//! increments the iterator
DObjConstIterator DObjConstIterator::operator ++(int)
{
    DObjConstIterator b = *this;
    *this += 1;
    return b;
}

//-------------------------------------------------------------------------  
bool DObjConstIterator::operator == (const DObjConstIterator& dObjIt)
{
    return (plane == dObjIt.plane && ptr == dObjIt.ptr);
}

//-------------------------------------------------------------------------  
bool DObjConstIterator::operator != (const DObjConstIterator& dObjIt)
{
    return plane != dObjIt.plane || ptr != dObjIt.ptr;
}

//-------------------------------------------------------------------------  
bool DObjConstIterator::operator < (const DObjConstIterator& dObjIt)
{
    return (plane < dObjIt.plane) || (plane == dObjIt.plane && ptr < dObjIt.ptr);
}

//-------------------------------------------------------------------------  
bool DObjConstIterator::operator > (const DObjConstIterator& dObjIt)
{
    return (plane > dObjIt.plane) || (plane == dObjIt.plane && ptr > dObjIt.ptr);
}

//-------------------------------------------------------------------------  
bool DObjConstIterator::operator <= (const DObjConstIterator& dObjIt)
{
    return (plane < dObjIt.plane) || (plane == dObjIt.plane && ptr <= dObjIt.ptr);
}

//-------------------------------------------------------------------------  
bool DObjConstIterator::operator >= (const DObjConstIterator& dObjIt)
{
    return (plane > dObjIt.plane) || (plane == dObjIt.plane && ptr >= dObjIt.ptr);
}

//-------------------------------------------------------------------------  
//! moves the iterator at pos ofs.
void DObjConstIterator::seekAbs(int ofs)
{
    if(dObj)
    {
        size_t matIndex;
        size_t dims = dObj->getDims();

        if(ofs <= 0) //begin
        {
            //ptr = dObj->rowPtr(0,0);
            size_t matIndex = dObj->seekMat(0);
            ptr = ((cv::Mat*)dObj->get_mdata()[matIndex])->data;

            sliceStart = ptr;
            plane = 0;

            if(planeContinuous)
            {
                sliceEnd = ptr + elemSize * dObj->getSize(dims-1) * dObj->getSize(dims-2); //one after the last ptr
            }
            else
            {
                sliceEnd = ptr + elemSize * dObj->getSize(dims-1);
            }
        }
        else if((size_t)ofs >= dObj->getTotal()) //end, (ofs > 0)
        {
            plane = dObj->calcNumMats() - 1;

            if(planeContinuous)
            {
                
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + elemSize * dObj->getSize(dims-1) * dObj->getSize(dims-2);
            }
            else
            {
                //sliceStart = dObj->rowPtr(plane, dObj->getSize(dims-2)-1 );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->ptr( dObj->getSize(dims-2)-1 );
                ptr = sliceStart + elemSize * dObj->getSize(dims-1);
            }
            
            sliceEnd = ptr;
        }
        else
        {
            //determine the plane, where it lies in
            size_t planeSize = dObj->getSize( dims-1 ) * dObj->getSize( dims-2 );
            plane = ofs / planeSize; //floor value
            ofs -= (plane * planeSize);

            if (planeContinuous)
            {
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + elemSize * ofs;
                sliceEnd = sliceStart + elemSize * dObj->getSize(dims-1) * dObj->getSize(dims-2);
            }
            else
            {
                size_t row = ofs / dObj->getSize(dims-2); //floor value

                //sliceStart = dObj->rowPtr(plane,row);
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->ptr( row );

                sliceEnd = sliceStart + elemSize * dObj->getSize(dims-1);
                ptr = sliceStart + elemSize * (ofs - row * dObj->getSize(dims-2));
            }
        }
    }
}

//-------------------------------------------------------------------------  
//! moves the iterator by pos ofs.
void DObjConstIterator::seekRel(int ofs)
{
    if(dObj)
    {
        size_t matIndex;
        size_t dims = dObj->getDims();
        size_t width = dObj->getSize(dims-1);
        size_t stride = dObj->getOriginalSize(dims-1);

        size_t curRowIdx = (ptr - dObj->rowPtr(plane,0)) / (stride * elemSize); //floor
        int curElemIdxInPlane;
        if(planeContinuous)
        {
            curElemIdxInPlane = (ptr - dObj->rowPtr(plane,0)) / elemSize;
        }
        else
        {
            curElemIdxInPlane = curRowIdx * width + (ptr - sliceStart) / elemSize;
        }
        int planeSize = static_cast<int>(dObj->getSize(dims-1) * dObj->getSize(dims-2));

        curElemIdxInPlane += ofs;

        if(curElemIdxInPlane >= planeSize) //any plane after this plane
        {
            ofs = (curElemIdxInPlane - planeSize);
        }
        else if(curElemIdxInPlane < 0) //any plane before this plane
        {
            ofs = curElemIdxInPlane;
        }
        else //same plane
        {
            if(planeContinuous)
            {
                ptr = sliceStart + curElemIdxInPlane * elemSize; //sliceStart, sliceEnd still the same
            }
            else
            {
                curRowIdx = (curElemIdxInPlane / width);
                
                //sliceStart = dObj->rowPtr( plane, curRowIdx );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->ptr( curRowIdx );

                sliceEnd = sliceStart + width * elemSize;
                ptr = sliceStart + (curElemIdxInPlane - curRowIdx * width) * elemSize;
            }
            return;
        }

        //calc destination plane and adjust ofs
        int planeOffset = ofs / planeSize;
        plane += planeOffset;
        ofs -= (planeOffset * planeSize);
        plane += ((ofs >= 0) ? +1 : -1); //the destination plane is one before or after the last plane, that has been fully skipped
        
        if( plane < 0) //move to begin
        {
            //ptr = dObj->rowPtr(0,0);
            size_t matIndex = dObj->seekMat(0);
            ptr = ((cv::Mat*)dObj->get_mdata()[matIndex])->data;
            sliceStart = ptr;
            plane = 0;

            if(planeContinuous)
            {
                sliceEnd = ptr + elemSize * planeSize; //one after the last ptr
            }
            else
            {
                sliceEnd = ptr + elemSize * width;
            }
        }
        else if ( (size_t)plane >= dObj->calcNumMats() ) //move to end (plane cannot be negative again)
        {
            plane = dObj->calcNumMats() - 1;

            if(planeContinuous)
            {
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + elemSize * planeSize;
            }
            else
            {
                //sliceStart = dObj->rowPtr(plane, dObj->getSize(dims-2)-1 );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->ptr( dObj->getSize(dims-2)-1 );
                ptr = sliceStart + elemSize * width;
            }
            
            sliceEnd = ptr;
        }
        else //move to another plane or stay within this plane
        {
            curElemIdxInPlane = ((ofs >= 0) ? ofs : (planeSize - ofs));
            if(planeContinuous)
            {
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + curElemIdxInPlane * elemSize;
                sliceEnd = sliceStart + (planeSize * elemSize);
            }
            else
            {
                curRowIdx = (curElemIdxInPlane / width);
                //sliceStart = dObj->rowPtr( plane, curRowIdx );
                matIndex = dObj->seekMat(plane);
                sliceStart = ((cv::Mat*)dObj->get_mdata()[matIndex])->ptr( curRowIdx );
                sliceEnd = sliceStart + width * elemSize;
                ptr = sliceStart + (curElemIdxInPlane - curRowIdx * width) * elemSize;
            }
        }

    }
}

//-------------------------------------------------------------------------  
//-------------------------------------------------------------------------  
//! default constructor
DObjIterator::DObjIterator() : DObjConstIterator() {}

//-------------------------------------------------------------------------  
//! constructor that sets the iterator to the beginning of the matrix 
DObjIterator::DObjIterator(DataObject* _dObj, int pos /*= 0*/):
    DObjConstIterator(_dObj, pos)
{
}

//-------------------------------------------------------------------------  
//! copy constructor
DObjIterator::DObjIterator(const DObjIterator& it)
{
    dObj = it.dObj;
    planeContinuous = it.planeContinuous;
    elemSize = it.elemSize;
    ptr = it.ptr; //current pointer to current element
    sliceStart = it.sliceStart;
    sliceEnd = it.sliceEnd;
    plane = it.plane;
}

//-------------------------------------------------------------------------  
//! copy operator
DObjIterator& DObjIterator::operator = (const DObjIterator& it)
{
    dObj = it.dObj;
    planeContinuous = it.planeContinuous;
    elemSize = it.elemSize;
    ptr = it.ptr; //current pointer to current element
    sliceStart = it.sliceStart;
    sliceEnd = it.sliceEnd;
    plane = it.plane;
    return *this;
}

//-------------------------------------------------------------------------  
//! returns the current matrix element
uchar* DObjIterator::operator *()
{
    return ptr;
}

//-------------------------------------------------------------------------  
//! returns the i-th matrix element, relative to the current
uchar* DObjIterator::operator [](int i)
{
    DObjConstIterator it2(*this);
    it2 += i;
    return (*it2);
}
    

//-------------------------------------------------------------------------  
//! shifts the iterator forward by the specified number of elements
DObjIterator& DObjIterator::operator += (int ofs)
{
    if( !dObj || ofs == 0 )
        return *this;
    int ofsb = ofs*elemSize;
    ptr += ofsb;
    if( ptr < sliceStart || sliceEnd <= ptr )
    {
        ptr -= ofsb;
        seekRel(ofs);
    }
    return *this;
}

//-------------------------------------------------------------------------  
//! shifts the iterator backward by the specified number of elements
DObjIterator& DObjIterator::operator -= (int ofs)
{
    return (*this) += -ofs;
}

//-------------------------------------------------------------------------  
//! decrements the iterator
DObjIterator& DObjIterator::operator --()
{
    if( dObj && (ptr -= elemSize) < sliceStart )
    {
        ptr += elemSize;
        seekRel(-1); 
    }
    return *this;
}

//-------------------------------------------------------------------------  
//! decrements the iterator
DObjIterator DObjIterator::operator --(int)
{
    DObjIterator b = *this;
    *this -= 1;
    return b;
}

//-------------------------------------------------------------------------    
//! increments the iterator
DObjIterator& DObjIterator::operator ++()
{
    if(dObj && (ptr += elemSize) >= sliceEnd )
    {
        ptr -= elemSize;
        seekRel(1); 
    }
    return *this;
}

//-------------------------------------------------------------------------
//! increments the iterator
DObjIterator DObjIterator::operator ++(int)
{
    DObjIterator b = *this;
    *this += 1;
    return b;
}


/*!
    \class DataObject
    \brief dataObject contains a n-dimensional matrix

    The n-dimensional matrix can have different element types. Recently the following types are supported:
    int8, uint8, int16, uint16, int32, uint32, float32, float64 (=> double), complex64 (2x float32), complex128 (2x float64)

    In order to handle huge matrices, the data object can divide one matrix into subparts in memory. Each subpart (called matrix-plane)
    is two-dimensional and covers data of the last two dimensions. Each of these matrix-planes is of type cv::Mat_<type> and can be used with every
    operator given by the openCV-framework (version 2.3.1 or higher).

    We assume to have a n-dimensional matrix A, where each dimension has its size s_i, hence A=[s_1, s_2, ..., s_(n-2), s_(n-1), s_n]

    Hence, in total there are s_1 * s_2 * ... * s_(n-2) different matrix-planes, which are all accessible by the member m_data, which is a std::vector
    of the general type int*. This type has to be casted to the specific cv::Mat_<...> when one matrix-plane has to be accessed. Sometimes it is also possible
    to simply cast to cv::Mat.

    In order to make the data object compatible to continuously organized data structures, like numpy-arrays, it is also possible to have all matrix-planes
    in one data-block in memory. Then the continuous-flag will be set and the whole data block can be accessed by taking the pointer given by m_data[0]. Nevertheless,
    the indicated data structure with the two-dimensional sub-matrix-planes is still existing, hence, the pointer to each matrix-planes points to the entry point of its
    matrix-planes lying withing the huge data block.

    The data organization is equal to the one of open-cv, hence, two-dimensional matrices are stored row-by-row (C-style)...

    The real size of each dimension is stored in the vector m_osize. Since it is possible to set a n-dimensional region of interest (ROI) to each matrix, the virtual
    dimensions, which will be delivered if the user asks for the matrix size, are stored in the member vector m_size.

    Concept to handle templated and non-templated methods<BR>
    -----------------------------------------------------

    According to openCV, the class dataObject is not templated, because there are some structures in the entire itom-framework which does not support any templating concept,
    like the plugin-handling or communication with external dll-functions. Additionally the signal-slot-design of the Qt-framework does not accept templated parameters
    beside some standard-objects. Therefore the element-data-type is set by the integer-member m_type. The transformation between the real data type and the integer number
    is coded several times within the whole framework and can be accessed by the enumeration tDataType in typeDefs.h. Since templating has got many advantages concerning
    low-level calculation, we adapted the transformation-process which is used by openCV:

    1. define a templated helper-method in the following form:

        template<typename _Tp> returnType 'MethodName'Func(Parameters1)

    2. define the following two lines of code:
        typedef returnType (*t'MethodName'Func)(Parameters1);
        MAKEFUNCLIST('MethodName'Func);

    3. define the method, accessed for example as public-method of dataObject
        RetVal DataObject::'PublicMethodName'(Parameters2)
        {
            ...
            fList'MethodName'Func[getType()](Parameters1);
            ...
            return ...
        }

    ---
    By the macro MAKEFUNCLIST a list fList'MethodName'Func is generated with each entry being a function pointer to the specific templated version of
    'MethodName'Func. The specific method is accessed by using getType() of dataObject. Hence it is important to keep the element-data-types and their order
    consistent for the whole itom-project.


*/


////////int func1(void) { return 0; };
////////----------------------------------------------------------------------------------------------------------------------------------
///////
////// RetVal callBinFunc(const BinaryFunc *funcList, const int type, const DataObject *src1, const DataObject *src2, DataObject *dst)
//////{
//////   return (funcList[type])(src1, src2, dst);
//////}
//////

//! creates template defined function table for all supported data types
#define MAKEFUNCLIST(FuncName) static t##FuncName fList##FuncName[] =   \
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
   FuncName<ito::complex128>                                                 \
};

//! creates function table for the function (FuncName) and both complex data types. The destination method must be templated with two template values.
#define MAKEFUNCLIST_CMPLX_TO_REAL(FuncName) static t##FuncName fList##FuncName[] =     \
{                                                                                       \
    FuncName<ito::complex64,float32>,                                                        \
    FuncName<ito::complex128,float64>                                                        \
};

#define TYPE_OFFSET_COMPLEX  8

//----------------------------------------------------------------------------------------------------------------------------------
//! returns pointer to vector of cv::_Mat-matrices
/*!

    \return pointer to vector of matrices
    \remark the returned type of std::vector is int*, you should cast it to the appropriate type (e.g. cv::_Mat<int8>)
*/
//std::vector<int *> DataObject::get_mdata(void)
int ** DataObject::get_mdata(void)
{
   return this->m_data;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! constant version of get_mdata
/*!
    \sa get_mdata
*/
//std::vector<int *> DataObject::get_mdata(void) const
int ** DataObject::get_mdata(void) const
{
   return this->m_data;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \todo documentation is missing
/*!

*/
RetVal DataObject::matNumToIdx(const size_t matNum, size_t *matIdx) const
{
   size_t tMatNum = matNum;
   size_t planeSize = 1;
   size_t m_dims_us = m_dims;

   for (size_t nDim = 1; nDim < m_dims_us - 2; nDim++)
   {
         planeSize *= m_osize[nDim];
   }

   for (size_t nDim = 0; nDim < m_dims_us - 2; nDim++)
   {
         matIdx[nDim] = tMatNum / planeSize - m_roi[nDim];
         tMatNum %= planeSize;
         planeSize /= m_osize[nDim + 1];
   }

   return RetVal(retOk);
}



//----------------------------------------------------------------------------------------------------------------------------------
//! calculates numbers of single opencv matrices which are part of the ROI which has previously been set.
/*!
    \return 0 if empty range or empty matrix, 1 if two dimensional, else product of sizes of all dimensions besides the last two ones.
*/
size_t DataObject::calcNumMats(void) const
{
   if (m_dims > 2)
   {
      size_t numMat = 1;
      for (int n = 0; n < m_dims - 2; n++)
      {
         numMat *= m_size[n];
      }
      return numMat;
   }
   else if (m_dims == 0 /*|| m_size.m_p == 0*/)
   {
      return 0;
   }
   else
   {
      return 1;
   }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns the index vector-index of m_data which corresponds to the given zero-based two-dimensional matrix-index
/*!
    Since there might be a difference between the "real" matrix size in memory and the virtual size which is set by subslicing a matrix and hence
    setting any ROI, this method transforms a desired matrix-plane index to the real index in memory of the data-vector m_data

    \param matNum zero-based matrix-plane considering the virtual matrix size (ROI), 0<=matNum<calcNumMats
    \return real vector-index for the desired matrix-plane
    \sa seekMat
    \sa calcNumMats
*/
size_t DataObject::seekMat(const size_t matNum) const
{
   size_t numMats = calcNumMats();
   return seekMat(matNum, numMats);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! returns the index vector-index of m_data which corresponds to the given zero-based two-dimensional matrix-index
/*!
    Since there might be a difference between the "real" matrix size in memory and the virtual size which is set by subslicing a matrix and hence
    setting any ROI, this method transforms a desired matrix-plane index to the real index in memory of the data-vector m_data

    \param matNum zero-based matrix-plane-index, considering the virtual matrix size (ROI), 0<=matNum<calcNumMats
    \param numMats total number of matrix-planes, lying within the ROI
    \return real vector-index for the desired matrix-plane or 0 if matNum >= numMats.
*/
size_t DataObject::seekMat(const size_t matNum, const size_t numMats) const
{
    if( matNum >= numMats) return 0; //check boundaries

    if (m_dims <= 2)
    {
            return 0;
    }

    //in order to understand the calculation, consider first to determine the index-vector, where the last two items are set to zero.
    //the index vector starts counting at the first element of the ROI (zero-based), therefore:
    //
    // 3.dim-matrix: idx = { i & size[0] , 0 , 0 }
    //
    // 4.dim-matrix: idx = { ((i - idx[1]*t[1])/t[0]) % size[0] , i % size[1] , 0 , 0 }
    //        where    t = {         t[1] * size[1]             ,     1       , 0 , 0 }
    //
    // 5.dim-matrix: idx = { ((i - idx[1]*t[1] - idx[2]*t[2])/t[0]) % size[0] , ((i - idx[2]*t[2])/t[1]) % size[1] , i % size[2] , 0 , 0 }
    //        where    t = {                  t[1] * size[1]                  ,         t[2] * size[2]             ,     1       , 0 , 0 }
    //
    // Simplification of this scheme leads to first or second possibility.

    ////begin 1. possibility: determine indices within region of interest an call matIdxToNum in order to get the plane-number for this index
    ////begin 2. possibility: integrate matIdxToNum here.
    //size_t *idx = new size_t[m_dims];
    //idx[m_dims-2] = 0;
    //idx[m_dims-1] = 0;
    //size_t val = matNum;
    //size_t t = 1;
    //size_t result = 0;

    //for(int i = m_dims - 3 ; i >=0 ; i--)
    //{
    //    idx[i] = ( val/t ) % m_size[i];
    //    val -= (idx[i] * t);
    //    t *= m_size[i];
    //}

    ////1. possibility:
    //matIdxToNum(idx, &result);

    ////2. possibility:
    //size_t planeSize = 1;

    //for (int n = m_dims - 3; n >= 0; n--)
    //{
    //        result += (idx[n] + m_roi[n]) * planeSize; //CAST_TODO
    //        planeSize *= m_osize[n];
    //}

    //delete[] idx;
    //return result;

    ////end 1. and 2. possibility

    //begin 3. possibility, directly combine 2. possiblity into one loop without allocating idx-vector
    size_t val = matNum;
    size_t t = 1;
    size_t result = 0;
    size_t idx = 0;
    size_t planeSize = 1;

    for(int i = m_dims - 3 ; i >=0 ; i--)
    {
        idx = ( val/t ) % m_size[i];
        result += (idx + m_roi[i]) * planeSize;
        val -= (idx * t);
        t *= m_size[i];
        planeSize *= m_osize[i];
    }

    return result;


    //original version (buggy)

    /*size_t tmat = 0;
    size_t dmat = 0;
    int ndim;

    tmat = 0;
    dmat = matNum;
    ndim = 0;
    do
    {
            tmat += dmat % m_size[ndim] + m_roi[ndim];
            dmat /= m_size[ndim];
            tmat += dmat * (m_osize[ndim] - 1);
            ndim++;
    } while ((ndim < m_dims - 2) && (dmat > 0));

    return tmat;*/
};

//----------------------------------------------------------------------------------------------------------------------------------
//! templated method for create
/*!
    creates or initializes matrix with given parameters

    \param dObj DataObject, whose matrix is created here
    \param dimensions total number of dimensions (>=1), if dimensions == 1, dimensions will be set to two and a matrix with dimension [1 x orginial dimension] is created
    \param *sizes vector with size of dimensions, each element gives the size of elements in each dimension
    \param continuous, indicates whether the data stored in this data object is stored in one continuous data block or not. if dimension <= 2, matrix is always continuous
                be careful, continuous has not the same meaning than the continuous flag in opencv or numpy.
    \param continuousDataPtr if this pointer is NULL, new data will be allocated. Else the given data indicates data which will be used by this data object. only possible if continuous is true. m_ownflag will be set to 0 if this pointer is set
    \param *steps vector with size of dimensions, indicates how many bytes one has to move in order to get to the next element in the same dimension, the step-size for the last element must be equal to element-size (in byte)
    \return retOk
    \sa create
*/
template<typename _Tp> RetVal CreateFunc(DataObject *dObj, const unsigned char dimensions, const size_t *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const size_t* steps)
{
   size_t numMats = 0;

   if(dimensions == 0)
   {
       dObj->m_dims = 0;
   }
   else if(dimensions > 1)
   {
        dObj->createHeader(dimensions, sizes, steps, sizeof(_Tp));
   }
   else //if one-dimensional create a two-dimensional data-object
   {
       size_t sizes_inc[2] = {1,sizes[0]};
       size_t steps_inc[2] = {sizes[0] * sizeof(_Tp), sizeof(_Tp)};
       dObj->createHeader(2, sizes_inc, steps_inc, sizeof(_Tp));
   }

   if (!dObj->m_pRefCount && dimensions > 0)
   {
        dObj->m_pRefCount = new int(0);
        dObj->m_objSharedDataLock = new ReadWriteLock(dObj->m_objHeaderLock.getLockStatus());
        if(dimensions == 1)
        {
            dObj->m_pDataObjectTags = new DataObjectTags(2);
        }
        else
        {
            dObj->m_pDataObjectTags = new DataObjectTags(dimensions);
        }
   }

   cv::Mat_<_Tp> *dataMat = NULL;

   if(!continuous && continuousDataPtr)
   {
       cv::error(cv::Exception(CV_BadDataPtr, "data pointer must be empty if matrix is not continuous" ,"", __FILE__, __LINE__));
   }

   switch (dimensions)
   {
         case 0:
             break;
         case 1:
            dObj->mdata_realloc(1);

            try
            {
                if(continuousDataPtr)
                {
                    dataMat = new cv::Mat_<_Tp>(1, static_cast<int>(sizes[0]), (_Tp *)continuousDataPtr, sizes[0]*sizeof(_Tp));
                }
                else
                {
                    dataMat = new cv::Mat_<_Tp>(1, static_cast<int>(sizes[0]));
                }
            }
            catch(cv::Exception exc) //handle memory error
            {
                SecureFreeFunc<_Tp>(dObj);
                throw; //rethrow error
            }

            dObj->m_data[0] = reinterpret_cast<int *>(dataMat);
            break;

         case 2:
            dObj->mdata_realloc(1);

            try
            {
                if(continuousDataPtr)
                {
                    dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[0]), static_cast<int>(sizes[1]), (_Tp *)continuousDataPtr, steps[0]);
                }
                else
                {
                    dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[0]), static_cast<int>(sizes[1]));
                }
            }
            catch(cv::Exception exc) //handle memory error
            {
                SecureFreeFunc<_Tp>(dObj);
                throw; //rethrow error
            }

            dObj->m_data[0] = reinterpret_cast<int *>(dataMat);
            break;

         default:
            numMats = dObj->calcNumMats();
            dObj->mdata_realloc(numMats);

            try
            {
                if (!continuous)
                {
                   for (size_t n = 0; n < numMats; n++)
                   {
                      dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[dimensions - 2]), static_cast<int>(sizes[dimensions - 1]));
                      dObj->m_data[n] = reinterpret_cast<int *>(dataMat);
    //                  dObj->m_data.push_back((int *)dataMat);
                   }
                }
                else //!continuous
                {
                    size_t matSize = dObj->m_osize.m_p[dimensions - 2] * dObj->m_osize.m_p[dimensions - 1] * sizeof(_Tp);

                    if(continuousDataPtr)
                    {
                        dObj->mdata_realloc(numMats);
                        for (size_t n = 0; n < numMats; n++)
                        {
                            dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[dimensions - 2]), static_cast<int>(sizes[dimensions - 1]), (_Tp *)continuousDataPtr, steps[dimensions-2]);
    //                        dObj->m_data.push_back((int *)dataMat);
                            dObj->m_data[n] = reinterpret_cast<int *>(dataMat);
                            continuousDataPtr += matSize;
                        }
                    }
                    else
                    {
                        char *dataPtr = (char*)malloc(numMats * matSize);
                        if(dataPtr == NULL)
                        {
                            cv::error(cv::Exception(CV_StsNoMem, ("Failed to allocate memory"),"", __FILE__, __LINE__));
                        }

                        dObj->mdata_realloc(numMats);
                        for (size_t n = 0; n < numMats; n++)
                        {
                            dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[dimensions - 2]), static_cast<int>(sizes[dimensions - 1]), (_Tp *)dataPtr);
    //                        dObj->m_data.push_back((int *)dataMat);
                            dObj->m_data[n] = reinterpret_cast<int *>(dataMat);
                            dataPtr += matSize;
                        }
                    }

                }
            }
            catch(cv::Exception exc)
            {
                SecureFreeFunc<_Tp>(dObj);
                throw; //rethrow error
            }
            break;
   }

   return 0;
}

typedef RetVal (*tCreateFunc)(DataObject *dObj, const unsigned char dimensions, const size_t *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const size_t* steps);
MAKEFUNCLIST(CreateFunc)

//! high-level, non-templated method for data allocation
/*!
    \param dimensions is the total number of dimensions
    \param *sizes is a vector whose length is equal to dimensions. Each entry indicates the size of the specific dimension. Each matrix-plane is allocated with the size of the last two sizes
    \param type is the desired element data type (see tDataType)
    \param continuous indicates wether the entire array should be allocated in one connected data block in memory (true) or not (default, better for huge matrices)
    \param *continuousDataPtr is NULL if new data storage should be allocated (then m_owndata is true). Otherwise this pointer points to the starting point of a continuous data block, where this data-object should be refer to (then m_owndata is false)
    \param *steps vector with size of dimensions, indicates how many bytes one has to move in order to get to the next element in the same dimension, the step-size for the last element must be set to element-size
    \throws open-cv error in case of error
    \sa CreateFunc
*/
void DataObject::create(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous, const uchar* continuousDataPtr, const size_t* steps)
{
    m_type = type;

    if(dimensions <= 2)
    {
        m_continuous = 1; //matrix is always continuous if dimensions are <=2, since there exists only one cv::Mat-plane
    }
    else
    {
        m_continuous = continuous;
    }

    m_owndata = (continuousDataPtr == NULL);


    if(!m_continuous && continuousDataPtr)
    {
        cv::error(cv::Exception(CV_BadDataPtr, "data pointer must be empty if matrix is not continuous" ,"", __FILE__, __LINE__));
    }

    fListCreateFunc[type](this, dimensions, sizes, m_continuous, continuousDataPtr, steps);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! templated method for creation with given vector of cv::Mat-planes
/*!
    \param [in] dimensions is the total number of dimensions
    \param [in] *sizes is a vector whose length is equal to dimensions. Each entry indicates the size of the specific dimension. Each matrix-plane is allocated with the size of the last two sizes
    \param [in] type is the desired element data type (see tDataType)
    \param [in] *planes is an array of cv::Mat-planes which will be used as matrices for every single 2D-plane. Every Mat must have the same size and type. The type must correspond to the param type, the size must fit to the last two given sizes.
    \param [in] nrOfPlanes is the length of the planes-array. This value must be the same than (sizes[0]*sizes[1]*...*sizes[dimensions-2])
    \return retOk
    \sa create
*/
template<typename _Tp> RetVal CreateFuncWithCVPlanes(DataObject *dObj, const unsigned char dimensions, const size_t *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes)
{
    cv::Size tempOrgSize;
    cv::Size tempSize;
    cv::Point tempPoint;
    int dtop = 0;
    int dleft = 0;
    int dbottom = 0;
//    int dright = 0;

    if(dimensions == 0)
    {
        dObj->m_dims = 0;
    }
    else
    {
        planes[0].locateROI(tempOrgSize, tempPoint);
        tempSize = planes[0].size();
        dtop = tempPoint.y;
        dleft = tempPoint.x;
        dbottom = tempOrgSize.height - tempSize.height - tempPoint.y;
//        dright = tempOrgSize.width - tempSize.width - tempPoint.x;

        size_t* sizes_inc = new size_t[dimensions];
        size_t* osizes_inc = new size_t[dimensions];
        size_t* roi_inc = new size_t[dimensions];

        if(dimensions > 1)
        {
            for(int i=0;i<dimensions-2;i++)
            {
                sizes_inc[i] = osizes_inc[i] = sizes[i];
                roi_inc[i] = 0;
            }

            sizes_inc[dimensions - 2] = sizes[dimensions - 2];
            osizes_inc[dimensions - 2] = sizes[dimensions - 2] + dtop + dbottom;
            roi_inc[dimensions - 2] = +dtop;

            sizes_inc[dimensions - 1] = sizes[dimensions - 1];
            osizes_inc[dimensions - 1] = sizes[dimensions - 1] + dleft + dtop;
            roi_inc[dimensions - 1] = +dleft;

        }
        else //if one-dimensional create a two-dimensional data-object
        {
            sizes_inc[0] = 1; sizes_inc[1] = sizes[0];
            osizes_inc[0] = 1; osizes_inc[1] = sizes[0] + dleft + dtop;
            roi_inc[0] = 0; roi_inc[1] = +dleft;
        }

        dObj->createHeaderWithROI(dimensions, sizes_inc, osizes_inc, roi_inc);

        delete[] sizes_inc;
        delete[] osizes_inc;
        delete[] roi_inc;

        if (!dObj->m_pRefCount)
        {
            dObj->m_pRefCount = new int(0);
            dObj->m_objSharedDataLock = new ReadWriteLock(dObj->m_objHeaderLock.getLockStatus());
            dObj->m_pDataObjectTags = new DataObjectTags(dimensions);
        }

        cv::Mat_<_Tp> *dataMat;
        dObj->mdata_realloc(nrOfPlanes);

        try
        {
            for(unsigned int i = 0 ; i < nrOfPlanes ; i++)
            {
                dataMat = new cv::Mat_<_Tp>(planes[i]); //memory error might occur
                dObj->m_data[i] = reinterpret_cast<int *>(dataMat);
            }
        }
        catch(cv::Exception exc) //memory exception
        {
            SecureFreeFunc<_Tp>(dObj);
            throw; //rethrow error
        }
    }

    return retOk;
}

typedef RetVal (*tCreateFuncWithCVPlanes)(DataObject *dObj, const unsigned char dimensions, const size_t *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes);
MAKEFUNCLIST(CreateFuncWithCVPlanes)

//! high-level, non-templated method for data allocation
/*!
    \param [in] dimensions is the total number of dimensions
    \param [in] *sizes is a vector whose length is equal to dimensions. Each entry indicates the size of the specific dimension. Each matrix-plane is allocated with the size of the last two sizes
    \param [in] type is the desired element data type (see tDataType)
    \param [in] *planes is an array of cv::Mat-planes which will be used as matrices for every single 2D-plane. Every Mat must have the same size and type. The type must correspond to the param type, the size must fit to the last two given sizes.
    \param [in] nrOfPlanes is the length of the planes-array. This value must be the same than (sizes[0]*sizes[1]*...*sizes[dimensions-2])
    \throws open-cv error in case of error
    \sa CreateFuncWithCVPlanes
*/
void DataObject::create(const unsigned char dimensions, const size_t *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes)
{
    m_type = type;
    m_owndata = 1;
   

    if(dimensions == 0 || dimensions == 2)
    {
        m_continuous = 1; //matrix is always continuous if dimensions are <=2, since there exists only one cv::Mat-plane
    }
    else if(dimensions == 1)
    {
        cv::error(cv::Exception(CV_StsError, "DataObject with dimension = 1 not allowed." ,"", __FILE__, __LINE__));
    }
    else
    {
        m_continuous = 0;
    }

    //check whether planes do have the right size
    size_t sizex = dimensions ? sizes[dimensions - 1] : static_cast<size_t>(0);
    size_t sizey = dimensions ? sizes[dimensions - 2] : static_cast<size_t>(0);
    size_t numMats = dimensions ? 1 : 0;
    cv::Size planeSize;
    size_t requiredElemSize = 0;

    switch(type)
    {
    case ito::tUInt8:
    case ito::tInt8:
        requiredElemSize = 1;
        break;
    case ito::tUInt16:
    case ito::tInt16:
        requiredElemSize = 2;
        break;
    case ito::tUInt32:
    case ito::tInt32:
    case ito::tFloat32:
        requiredElemSize = 4;
        break;
    case ito::tFloat64:
    case ito::tComplex64:
        requiredElemSize = 8;
        break;
    case ito::tComplex128:
        requiredElemSize = 16;
        break;
    default:
        cv::error(cv::Exception(CV_StsError, "unkown type." ,"", __FILE__, __LINE__));
        break;
    }

    for(int i = 0 ; i < dimensions - 2 ; i++)
    {
        numMats *= sizes[i];
    }

    if(numMats != nrOfPlanes)
    {
        cv::error(cv::Exception(CV_BadImageSize, "nrOfPlanes must be equal to the product of the first (n-2) dimensions." ,"", __FILE__, __LINE__));
    }

    if ((type == ito::tComplex64) || (type ==ito::tComplex128))
    {
        for(size_t i = 0 ; i < numMats ; i++)
        {
            planeSize = planes[i].size();

            if((size_t)planeSize.height != sizey || (size_t)planeSize.width != sizex)
            {
                cv::error(cv::Exception(CV_BadImageSize, "image size of at least one cv::Mat-plane does not correspond to the given height and width.", "", __FILE__, __LINE__));
            }

            if(planes[i].channels() != 2)
            {
                cv::error(cv::Exception(CV_StsUnsupportedFormat, "at least one cv::Mat-plane has not two channels (complex type).", "", __FILE__, __LINE__));
            }

            if((planes[i].elemSize1()*2) != requiredElemSize)
            {
                cv::error(cv::Exception(CV_StsUnsupportedFormat, "the element size of at least one cv::Mat-plane does not correspond to the given dataObject-type.", "", __FILE__, __LINE__));
            }
        }
    }
    else
    {
        for(size_t i = 0 ; i < numMats ; i++)
        {
            planeSize = planes[i].size();

            if((size_t)planeSize.height != sizey || (size_t)planeSize.width != sizex)
            {
                cv::error(cv::Exception(CV_BadImageSize, "image size of at least one cv::Mat-plane does not correspond to the given height and width.", "", __FILE__, __LINE__));
            }

            if(planes[i].channels() != 1)
            {
                cv::error(cv::Exception(CV_StsUnsupportedFormat, "at least one cv::Mat-plane has not one channel.", "", __FILE__, __LINE__));
            }

            if(planes[i].elemSize1() != requiredElemSize)
            {
                cv::error(cv::Exception(CV_StsUnsupportedFormat, "the element size of at least one cv::Mat-plane does not correspond to the given dataObject-type.", "", __FILE__, __LINE__));
            }
        }
    }

    fListCreateFuncWithCVPlanes[type](this, dimensions, sizes, planes, nrOfPlanes);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for freeing allocated data blocks
/*!
    First, the header information of the corresponding data block is deleted. Then the reference counter of the data block is decremented.
    In the same way, the reference counter for every matrix-plane is incremented by calling the corresponding release-method. If the ref-counter is lower than zero
    no other instance needs this data block and it is deallocated if the m_owndata-flag is true.

    \param *dObj whose data block should be freed
    \return retOk
    \sa freeData
*/
template<typename _Tp> RetVal FreeFunc(DataObject *dObj)
{
   cv::Mat_<_Tp> *dataMat = NULL;

   // clear header
   if (dObj->m_roi.m_p) // m_roi.m_p-1 is the pointer mapping to the first element of [size of roi , roi-vector , size of osize, osize-vector, size of size, size-vector]
   {
      delete[] (dObj->m_roi.m_p - 1);
   }

   int old_m_dims = dObj->m_dims;

   dObj->m_size.m_p = NULL;
   dObj->m_osize.m_p = NULL;
   dObj->m_roi.m_p = NULL;
   dObj->m_dims = 0;

    // there is data?
    if (dObj->m_pRefCount)
    {
        // check if we are the last one
        if (*(dObj->m_pRefCount))
        {
            // no so just remove matrix headers and return
            CV_XADD(dObj->m_pRefCount, -1);
            //(*dObj->m_pRefCount)--;

            //this version of deleting the m_data vector is much faster than the version above (M. Gronle, 13.02.2012)
//            unsigned int size = dObj->m_data.size();
            size_t size = dObj->mdata_size();
            for ( size_t i = 0 ; i < size ; i++)
            {
                dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
                delete dataMat;
            }
            dObj->mdata_free();

            //unlock readwritelock of data block with respect to locking status of this dataObject
            //dObj->m_objSharedDataLock->_unlock(dObj->m_objHeaderLock.getLockStatus());

            return 0;
        }
        delete dObj->m_pRefCount;
        dObj->m_pRefCount = NULL;
        delete dObj->m_objSharedDataLock;
        dObj->m_objSharedDataLock = NULL;
        delete dObj->m_pDataObjectTags;
        dObj->m_pDataObjectTags = NULL;
    }

    // yes so really clean up
    size_t numMats;
    if (!(numMats = dObj->mdata_size()))
    {
        return 0;
    }

    if (dObj->m_continuous && old_m_dims > 2 && dObj->m_owndata)
    {
        dataMat = (cv::Mat_<_Tp> *)dObj->m_data[0];
        free(dataMat->datastart); //data is wrong, since data-pointer does not point to start in case of ROI
        //free(dataMat->data);
    }

    //this version of deleting the m_data vector is much faster than the version above (M. Gronle, 13.02.2012)
//    unsigned int size = dObj->m_data.size();
    size_t size = dObj->mdata_size();
    for ( size_t i = 0 ; i < size ; i++)
    {
        dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
        delete dataMat;
    }
    dObj->mdata_free();

    return 0;
};

typedef RetVal (*tFreeFunc)(DataObject *dObj);

MAKEFUNCLIST(FreeFunc);

//! high-level, non-templated method for freeing data
/*!
    \sa FreeFunc
*/
void DataObject::freeData(void)
{
   fListFreeFunc[m_type](this);
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> RetVal SecureFreeFunc(DataObject *dObj)
{
   cv::Mat_<_Tp> *dataMat = NULL;
   int old_m_dims = dObj->m_dims;

   // clear header
   if (dObj->m_roi.m_p) // m_roi.m_p-1 is the pointer mapping to the first element of [size of roi , roi-vector , size of osize, osize-vector, size of size, size-vector]
   {
      delete[] (dObj->m_roi.m_p - 1);
   }

   dObj->m_size.m_p = NULL;
   dObj->m_osize.m_p = NULL;
   dObj->m_roi.m_p = NULL;
   dObj->m_dims = 0;

    // does the data object contain a reference counter
    if (dObj->m_pRefCount)
    {
        if (*(dObj->m_pRefCount) > 0) //we are not the last to use the data
        {
            // decrease reference counter
            CV_XADD(dObj->m_pRefCount, -1);

            //delete cvMats in m_data array (OpenCV organizes the rest)
            if(dObj->m_data)
            {
                size_t size = dObj->mdata_size();
                for ( size_t i = 0 ; i < size ; i++)
                {
                    dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
                    if(dataMat) delete dataMat;
                }
                dObj->mdata_free();
            }

            return retOk;
        }
        else //this is the last instance to use this data
        {
            if(dObj->m_pRefCount) delete dObj->m_pRefCount;
            dObj->m_pRefCount = NULL;
            if(dObj->m_objSharedDataLock) delete dObj->m_objSharedDataLock;
            dObj->m_objSharedDataLock = NULL;
            if(dObj->m_pDataObjectTags) delete dObj->m_pDataObjectTags;
            dObj->m_pDataObjectTags = NULL;
        }
    }

    //this section is only entered if we are the last to use the data or if no reference counter has been set (the latter usually should not happen)
    size_t numMats = dObj->mdata_size();

    if(numMats > 0)
    {
        //check if the data has been allocated "en bloc" and delete the data first.
        if (dObj->m_continuous && old_m_dims > 2 && dObj->m_owndata)
        {
            dataMat = (cv::Mat_<_Tp> *)dObj->m_data[0];
            if(dataMat && dataMat->datastart)
            {
                free(dataMat->datastart);
            }
        }

        for ( size_t i = 0 ; i < numMats ; i++)
        {
            dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
            if(dataMat) delete dataMat;
        }

        dObj->mdata_free();
    }

    return retOk;
};

typedef RetVal (*tSecureFreeFunc)(DataObject *dObj);
MAKEFUNCLIST(SecureFreeFunc);

//! high-level, non-templated method for securely freeing data
/*!
    \sa SecureFreeFunc
*/
void DataObject::secureFreeData(void)
{
    fListSecureFreeFunc[m_type](this);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for deeply copying the data of one matrix to another given matrix
/*!
    At first, the memory of the new matrix is delete. Then the data of the lhs-matrix is deeply copied to the rhs-matrix.

    \param &lhs is the matrix whose data is copied
    \param &rhs is the matrix where the data is copied to. The old data of rhs is deleted first
    \param regionOnly, if true, only the data of the ROI in lhs is copied, hence, the org-size of rhs corresponds to the ROI-size of lhs, else the whole data block is copied and the ROI of rhs is set to the ROI of lhs
    \return retOk
    \sa copyTo, CreateFunc
*/
template<typename _Tp> RetVal CopyToFunc(const DataObject &lhs, DataObject &rhs, unsigned char regionOnly)
{
   if (&lhs == &rhs)
   {
         return 0;
   }

   size_t numMats = 0;
   size_t tMat = 0;
   cv::Mat_<_Tp> *tempMat = NULL;
   cv::Mat_<_Tp> *rhsMat = NULL;

   rhs.freeData();
   rhs.m_type = lhs.m_type;
   char rhsOldContinuous = rhs.getDims() > 2 ? rhs.m_continuous : 0; //if dims(rhs)<=2, then the continuity-flag should only be influenced by lhs, since then the continuity doesn't change the representation and the constructor of empty dataObject sets the flag to one (default)
   rhs.m_continuous = rhsOldContinuous | lhs.m_continuous;

   if (regionOnly || lhs.m_dims == 0) //Marc: bug, if empty data object, it is necessary to use this if case, too.
   {
         numMats = lhs.calcNumMats();
         CreateFunc<_Tp>(&rhs, lhs.m_dims, lhs.m_size, rhs.m_continuous, NULL, NULL);
         for (size_t nMat = 0; nMat < numMats; nMat++)
         {
            tMat = lhs.seekMat(nMat);
            tempMat = (cv::Mat_<_Tp> *)lhs.m_data[tMat];
            rhsMat = (cv::Mat_<_Tp> *)rhs.m_data[nMat];
            tempMat->copyTo(*rhsMat); //
         }
   }
   else
   {
//         numMats = lhs.m_data.size();
         numMats = lhs.mdata_size();
         CreateFunc<_Tp>(&rhs, lhs.m_dims, lhs.m_osize, rhs.m_continuous, NULL, NULL);

         for(unsigned int i = 0 ; i < rhs.m_size.m_p[-1]; i++)
         {
             rhs.m_size.m_p[i] = lhs.m_size.m_p[i];
         }
         for(unsigned int i = 0 ; i < rhs.m_roi.m_p[-1]; i++)
         {
             rhs.m_roi.m_p[i] = lhs.m_roi.m_p[i];
         }


         for (unsigned int nMat = 0; nMat < numMats; nMat++)
         {
            tempMat = (cv::Mat_<_Tp> *)lhs.m_data[nMat];
            rhsMat = (cv::Mat_<_Tp> *)rhs.m_data[nMat];

            cv::Size tempOrgSize;
            cv::Size tempSize = tempMat->size();
            cv::Point tempPoint;

            tempMat->locateROI(tempOrgSize, tempPoint);

            int dtop = tempPoint.y;
            int dleft = tempPoint.x;
            int dbottom = tempOrgSize.height - tempSize.height - tempPoint.y;
            int dright = tempOrgSize.width - tempSize.width - tempPoint.x;

            tempMat->adjustROI(dtop, dbottom , dleft, dright );

            tempMat->copyTo(*rhsMat);

            tempMat->adjustROI(-dtop, -dbottom, -dleft, -dright);
            rhsMat->adjustROI(-dtop, -dbottom, -dleft, -dright);


            //tempMat->adjustROI(-tempPoint.y, -tempOrg
         }
   }

   return 0;
}

typedef RetVal (*tCopyToFunc)(const DataObject &lhs, DataObject &rhs, unsigned char regionOnly);

MAKEFUNCLIST(CopyToFunc);

//! high-level, non-templated method to deeply copy the data of this matrix to another matrix rhs
/*!
    \param &rhs is the matrix where the data is copied to. The old data of rhs is deleted first
    \param regionOnly, if true, only the data of the ROI in lhs is copied, hence, the org-size of rhs corresponds to the ROI-size of lhs, else the whole data block is copied and the ROI of rhs is set to the ROI of lhs
    \return retOk
    \sa CopyToFunc
*/
RetVal DataObject::copyTo(DataObject &rhs, unsigned char regionOnly)
{
    ito::RetVal ret = fListCopyToFunc[m_type](*this, rhs, regionOnly);

    if(!ret.containsError())
    {
        this->copyTagMapTo(rhs);   //Deepcopy the tagspace
        this->copyAxisTagsTo(rhs); //Deepcopy the tagspace
    }
    return ret;
}


//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method to copy the values of the ROI of matrix lhs to the ROI of matrix rhs.
/*!
    the ROI of rhs must already correspond to the ROI of lhs, hence, rhs must have allocated data.

    \param &lhs is the original data object
    \param &rhs is the data object, where the values are copied to.
    \return retOk
    \sa deepCopyPartial
    \todo avoid DObjIterator for speed-up
*/
template<typename _Tp> RetVal DeepCopyPartialFunc(DataObject &lhs, DataObject &rhs)
{
   if (&lhs == &rhs)
   {
         return 0;
   }

   int sizeX = static_cast<int>(lhs.getSize(lhs.getDims() - 1));
   int sizeY = static_cast<int>(lhs.getSize(lhs.getDims() - 2));
   for (unsigned int nMat = 0; nMat < lhs.calcNumMats(); nMat++)
   {
        cv::Mat_<_Tp> *cvMatLhs = ((cv::Mat_<_Tp> *)lhs.get_mdata()[lhs.seekMat(nMat)]);
        cv::Mat_<_Tp> *cvMatRhs = ((cv::Mat_<_Tp> *)rhs.get_mdata()[rhs.seekMat(nMat)]);

        if (cvMatLhs->isContinuous() && cvMatRhs->isContinuous())
        {
           memcpy(cvMatRhs->ptr(0), cvMatLhs->ptr(0), sizeX * sizeY * sizeof(_Tp));
        }
        else
        {
           for (int y = 0; y < sizeY; y++)
           {
               memcpy(cvMatRhs->ptr(y), cvMatLhs->ptr(y), sizeX * sizeof(_Tp));
           }
        }
   }

//   DObjIterator<_Tp> itLhs_begin(lhs);
//   DObjIterator<_Tp> itRhs_begin(rhs);

//   while(itLhs_begin != lhs.end())
//   {
//       (*itRhs_begin) = (*itLhs_begin);
//       itLhs_begin++;
//       itRhs_begin++;
//   }

   return 0;
}

typedef RetVal (*tDeepCopyPartialFunc)(DataObject &lhs, DataObject &rhs);

MAKEFUNCLIST(DeepCopyPartialFunc);

//! high-level, non-templated method. Deeply copies data of this data object which is within its ROI to the ROI of rhs.
/*!
    \param &rhs is the right-handed data object, where data is copied to.
    \return retOk
    \throws cv::Exception(CV_StsAssert) if sizes or type of both matrices are not equal
    \sa DeepCopyPartialFunc
*/
RetVal DataObject::deepCopyPartial(DataObject &rhs)
{
    if(m_type != rhs.m_type)
    {
        cv::error(cv::Exception(CV_StsAssert, "DataObject - operands differ in type","", __FILE__,__LINE__));
    }

    //calc and compare squeezed dimensions
    int thisDims = this->getDims();
    int rhsDims = rhs.getDims();
    size_t *thisSizes = new size_t[thisDims];
    size_t *rhsSizes = new size_t[rhsDims];

    int j = 0;
    for(int i=0; i<thisDims; i++)
    {
        thisSizes[j] = this->getSize(i);
        if(thisSizes[j]>1) j++;
    }
    thisDims = j;

    j = 0;
    for(int i=0; i<rhsDims; i++)
    {
        rhsSizes[j] = rhs.getSize(i);
        if(rhsSizes[j]>1) j++;
    }
    rhsDims = j;

    if(thisDims != rhsDims)
    {
        cv::error(cv::Exception(CV_StsAssert, "DataObject - operands differ in number of dimensions","", __FILE__,__LINE__));
    }

    for(int i=0;i<thisDims;i++)
    {
        if(thisSizes[i] != rhsSizes[i])
        {
            cv::error(cv::Exception(CV_StsAssert, "DataObject - operands differ in size","", __FILE__,__LINE__));
        }
    }

    delete[] thisSizes;
    delete[] rhsSizes;

    ito::RetVal ret = fListDeepCopyPartialFunc[m_type](*this, rhs);

    if(!ret.containsError())
    {
        this->copyTagMapTo(rhs);   //Deepcopy the tagspace
        this->copyAxisTagsTo(rhs); //Deepcopy the tagspace
    }
    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! converts data in DataObject lhs to DataObject rhs with a given type
/*!
    Every element of the source data object is copied to the destionation data object by using this transformation<BR>
        elem_destination = static_cast<newType>(elem_source * alpha + beta)

    \param &lhs is the left-hand sided data object, whose data should be converted
    \param &rhs is the destination data object, whose memory is firstly deleted, then newly allocated
    \param type is the type-number of the destination element
    \param alpha scaling factor (default: 1.0)
    \param beta offset value (default: 0.0)
    \return retOk
    \throws cv::Exception(CV_StsAssert) if conversion type is unknown
    \sa convertTo, CastFunc
*/
template<typename _Tp> RetVal ConvertToFunc(const DataObject &lhs, DataObject &rhs, const int type, const double alpha, const double beta)
{
   if (&lhs == &rhs)
   {
         return 0;
   }

   //_Tp is source type
   

   if(type == lhs.getType())
   {
       rhs = lhs;
   }
   else
   {
       rhs.freeData();

       switch (type)
       {
          case ito::tInt8:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, int8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt8:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, uint8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt16:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, int16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt16:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, uint16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt32:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, uint32>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tFloat32:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, float32>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tFloat64:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, float64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex64:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, complex64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex128:
             rhs.create(lhs.m_dims, lhs.m_size, type, lhs.m_continuous);
             CastFunc<_Tp, complex128>(&lhs, &rhs, alpha, beta);
          break;

          default:
             cv::error(cv::Exception(CV_StsAssert, "not defined cast", "", __FILE__, __LINE__));
          break;
       }

        lhs.copyTagMapTo(rhs);   //Deepcopy the tagspace
        lhs.copyAxisTagsTo(rhs); //Deepcopy the tagspace
   }

   return 0;
}

typedef RetVal (*tConvertToFunc)(const DataObject &lhs, DataObject &rhs, const int type, const double alpha, const double beta);

MAKEFUNCLIST(ConvertToFunc);

//! high-level, non-templated matrix conversion
/*!
    Every element of the source matrix is converted to a new, given type. Additionally a floating-point scaling and offset parameter is possible.

    \param &rhs is the destination data object, whose memory is firstly deleted, then newly allocated
    \param type is the type-number of the destination element
    \param alpha scaling factor (default: 1.0)
    \param beta offset value (default: 0.0)
    \throws cv::Exception if cast failed, e.g. if cast not possible or types unknown
    \return retOk
    \sa fListConvertToFunc
*/
RetVal DataObject::convertTo(DataObject &rhs, const int type, const double alpha, const double beta ) const
{
    return fListConvertToFunc[m_type](*this, rhs, type, alpha, beta);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail this function makes a deepcopy of the tags map to rhs object from this object.
    \param &rhs is the matrix where the map is copied to. The old map of ths object is cleared first
    \return retOk
    \sa DataObjectTags
*/
RetVal DataObject::copyTagMapTo(DataObject &rhs) const
{
    if(this == &rhs)
    {
        return ito::retOk;
    }

    if(!m_pDataObjectTags)
    {
        return ito::RetVal(ito::retError, 0, "Source tagspace is not allocated");
    }
    if(!rhs.m_pDataObjectTags)
    {
        return ito::RetVal(ito::retError, 0, "Destination tagspace is not allocated");
    }

    rhs.m_pDataObjectTags->m_tags.clear();

    if(m_pDataObjectTags->m_tags.size() == 0)
    {
        return ito::RetVal(ito::retWarning, 0, "Source tag map was empty");
    }

    rhs.m_pDataObjectTags->m_tags = m_pDataObjectTags->m_tags;

    return ito::retOk;
}
//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \detail this function makes a deepcopy of the axis and value metadata from this object to rhs object.
            It copies
    \param &rhs is the matrix where the map is copied from. The old map of this object is cleared first
    \return retOk
    \sa DataObjectTags
*/
RetVal DataObject::copyAxisTagsTo(DataObject &rhs) const
{
    if(this == &rhs)
    {
        return ito::retOk;
    }

    if(!m_pDataObjectTags)
    {
        return ito::RetVal(ito::retError, 0, "Source tagspace is not allocated");
    }
    if(!rhs.m_pDataObjectTags)
    {
        return ito::RetVal(ito::retError, 0, "Destination tagspace is not allocated");
    }

    rhs.setXYRotationalMatrix(m_pDataObjectTags->m_rotMatrix[0], m_pDataObjectTags->m_rotMatrix[1], m_pDataObjectTags->m_rotMatrix[2], m_pDataObjectTags->m_rotMatrix[3], m_pDataObjectTags->m_rotMatrix[4], m_pDataObjectTags->m_rotMatrix[5], m_pDataObjectTags->m_rotMatrix[6], m_pDataObjectTags->m_rotMatrix[7], m_pDataObjectTags->m_rotMatrix[8]);

    int axisNumRhs = rhs.getDims()-1;

    for(int axisNum = getDims()-1; axisNum > -1 ; axisNum--)
    {

        rhs.setAxisOffset(axisNumRhs, getAxisOffset(axisNum));
        rhs.setAxisScale(axisNumRhs, getAxisScale(axisNum));

        bool isValid = false;
        std::string tempDes = getAxisDescription(axisNum, isValid);   // check this
        if(isValid) rhs.setAxisDescription(axisNumRhs, tempDes);

        isValid = false;
        std::string tempUnit = getAxisUnit(axisNum, isValid);
        if(isValid) rhs.setAxisUnit(axisNumRhs, tempUnit);

        axisNumRhs--;
        if(axisNumRhs < 0)
        {
            break;
        }
    }

    rhs.setValueDescription(getValueDescription());
    rhs.setValueUnit(getValueUnit());

    return ito::retOk;
}
//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a zero-value matrix of size 1x1 with the given type
/*!
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::zeros(const int type)
{
   size_t sizes[2] = {1, 1};
   return zeros(2, sizes, type);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a zero-value matrix of size 1 x size with the given type
/*!
    \param size is the desired length of the vector
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::zeros(const size_t size, const int type)
{
   size_t sizes[2] = {1, size};
   return zeros(2, sizes, type);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a zero-value matrix of size sizeY x sizeX with the given type
/*!
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::zeros(const size_t sizeY, const size_t sizeX, const int type)
{
   size_t sizes[2] = {sizeY, sizeX};
   return zeros(2, sizes, type);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a zero-value, 3D- matrix of size sizeZ x sizeY x sizeX with the given type
/*!
    \param sizeZ are the number of matrix-planes
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \param continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::zeros(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const unsigned char continuous)
{
   size_t sizes[3] = {sizeZ, sizeY, sizeX};
   return zeros(3, sizes, type, continuous);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for creation of zero-valued matrix-plane
/*!

    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param **dstMat is the pointer to the already allocated cv::Mat_<type>-matrix-plane
    \return retOk
    \sa zeros
*/
template<typename _Tp> RetVal ZerosFunc(const size_t sizeY, const size_t sizeX, int **dstMat)
{
   (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::zeros(static_cast<int>(sizeY), static_cast<int>(sizeX));

   return 0;
}

typedef RetVal (*tZerosFunc)(const size_t sizeY, const size_t sizeX, int **dstMat);
MAKEFUNCLIST(ZerosFunc);

//! high-level, non-templated base function for allocation of new matrix whose elements are all set to zero
/*!
    \param dimensions indicates the number of dimensions
    \param *sizes is a vector with the same length than dimensions. Every element indicates the size of the specific dimension
    \param type is the desired data-element-type
    \param continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa ZerosFunc
*/
RetVal DataObject::zeros(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous)
{
   freeData();
   create(dimensions, sizes, type, continuous);

   size_t numMats = calcNumMats();

    size_t sizeX = sizes[dimensions - 1];
    size_t sizeY = 1;
    if(dimensions > 1)
    {
        sizeY = sizes[dimensions - 2];
    }

    for (size_t matn = 0; matn < numMats; matn++)
    {
        fListZerosFunc[type](sizeY, sizeX, &(m_data[matn]));
    }

   return 0;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-value matrix of size 1x1 with the given type
/*!
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const int type)
{
   size_t sizes[2] = {1, 1};
   return ones(2, sizes, type);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-value matrix of size 1 x size with the given type
/*!
    \param size is the desired length of the vector
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const size_t size, const int type)
{
   size_t sizes[2] = {1, size};
   return ones(2, sizes, type);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-value matrix of size sizeY x sizeX with the given type
/*!
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const size_t sizeY, const size_t sizeX, const int type)
{
   size_t sizes[2] = {sizeY, sizeX};
   return ones(2, sizes, type);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-valued, 3D- matrix of size sizeZ x sizeY x sizeX with the given type
/*!
    \param sizeZ are the number of matrix-planes
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \param unsigned char continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const unsigned char continuous)
{
   size_t sizes[3] = {sizeZ, sizeY, sizeX};
   return ones(3, sizes, type, continuous);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for creation of one-valued matrix-plane
/*!

    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param **dstMat is the pointer to the already allocated cv::Mat_<type>-matrix-plane
    \return retOk
    \sa zeros
*/
template<typename _Tp> RetVal OnesFunc(const size_t sizeY, const size_t sizeX, int **dstMat)
{
   (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::ones(static_cast<int>(sizeY), static_cast<int>(sizeX));
   return 0;
}

typedef RetVal (*tOnesFunc)(const size_t sizeY, const size_t sizeX, int **dstMat);
MAKEFUNCLIST(OnesFunc);

//! high-level, non-templated base function for allocation of new matrix whose elements are all set to one
/*!
    \param dimensions indicates the number of dimensions
    \param *sizes is a vector with the same length than dimensions. Every element indicates the size of the specific dimension
    \param type is the desired data-element-type
    \param continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa OnesFunc
*/
RetVal DataObject::ones(const unsigned char dimensions, const size_t *sizes, const int type, const unsigned char continuous)
{
    freeData();
    create(dimensions, sizes, type, continuous);

    size_t numMats = calcNumMats();

    size_t sizeX = sizes[dimensions - 1];
    size_t sizeY = 1;
    if(dimensions > 1)
    {
        sizeY = sizes[dimensions - 2];
    }

    for (size_t matn = 0; matn < numMats; matn++)
    {
        fListOnesFunc[type](sizeY, sizeX, &(m_data[matn]));
    }

   return 0;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a random-value matrix of size 1x1 with the given type
/*!
    \detail this function allocates an random value matrix using cv::randu for uniform (randMode = false) or gausion noise (randMode = true).
            In case of an integer type, the uniform noise is from min(inclusiv) to max(inclusiv). For floating point types, the noise is between 0(inclusiv) and 1(exclusiv).
            In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0.
    \param type is the desired type-number
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::rand(const int type, const bool randMode)
{
   size_t sizes[2] = {1, 1};
   return rand(2, sizes, type, randMode);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a random-value matrix of size 1 x size with the given type
/*!
    \detail this function allocates an random value matrix using cv::randu for uniform (randMode = false) or gausion noise (randMode = true).
            In case of an integer type, the uniform noise is from min(inclusiv) to max(inclusiv). For floating point types, the noise is between 0(inclusiv) and 1(exclusiv).
            In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0.
    \param size is the desired length of the vector
    \param type is the desired type-number
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::rand(const size_t size, const int type, const bool randMode)
{
   size_t sizes[2] = {1, size};
   return rand(2, sizes, type, randMode);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a random-value matrix of size sizeY x sizeX with the given type
/*!
    \detail this function allocates an random value matrix using cv::randu for uniform (randMode = false) or gausion noise (randMode = true).
            In case of an integer type, the uniform noise is from min(inclusiv) to max(inclusiv). For floating point types, the noise is between 0(inclusiv) and 1(exclusiv).
            In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0.
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::rand(const size_t sizeY, const size_t sizeX, const int type, const bool randMode)
{
   size_t sizes[2] = {sizeY, sizeX};
   return rand(2, sizes, type, randMode);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a random-valued, 3D- matrix of size sizeZ x sizeY x sizeX with the given type
/*!
    \detail this function allocates an random value matrix using cv::randu for uniform (randMode = false) or gausion noise (randMode = true).
            In case of an integer type, the uniform noise is from min(inclusiv) to max(inclusiv). For floating point types, the noise is between 0(inclusiv) and 1(exclusiv).
            In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0.
    \param sizeZ are the number of matrix-planes
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \param unsigned char continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::rand(const size_t sizeZ, const size_t sizeY, const size_t sizeX, const int type, const bool randMode, const unsigned char continuous)
{
   size_t sizes[3] = {sizeZ, sizeY, sizeX};
   return rand(3, sizes, type, randMode, continuous);
};

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for creation of random-valued matrix-plane
/*!

    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired data-element-type
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \param **dstMat is the pointer to the already allocated cv::Mat_<type>-matrix-plane
    \return retOk
    \sa zeros
*/
template<typename _Tp> RetVal RandFunc(const size_t sizeY, const size_t sizeX, const double value1, const double value2, const bool randMode, int **dstMat)
{
    (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::zeros(static_cast<int>(sizeY), static_cast<int>(sizeX));

    if(randMode)
    {
        cv::randn((*((cv::Mat_<_Tp> *)(*dstMat))), value1, value2);
    }
    else
    {
        cv::randu((*((cv::Mat_<_Tp> *)(*dstMat))), value1, value2);
    }
   return 0;
}

//! template specialisation for low-level, templated method for creation of random-valued matrix-plane of type complex128
/*!
    \return retOk
    \sa  RandFunc, zeros, ones
*/
template<> RetVal RandFunc<ito::complex128>(const size_t sizeY, const size_t sizeX, const double value1, const double value2, const bool randMode, int **dstMat)
{
    (*((cv::Mat_<ito::complex128> *)(*dstMat))) = cv::Mat_<ito::complex128>::zeros(static_cast<int>(sizeY), static_cast<int>(sizeX));

    if(randMode)
    {
        cv::Mat_<ito::float64> tempMat(sizeY, sizeX * 2, ((cv::Mat*)(*dstMat))->ptr<ito::float64>());
        cv::randn(tempMat, value1, value2);       
    }
    else
    {
        cv::randu((*((cv::Mat_<ito::complex128> *)(*dstMat))), value1, value2);
    }
   return 0;
}

typedef RetVal (*tRandFunc)(const size_t sizeY, const size_t sizeX, const double value1, const double value2, const bool randMode, int **dstMat);
MAKEFUNCLIST(RandFunc);

//! high-level, non-templated base function for allocation of new matrix whose elements are all set to one
/*!
    \detail this function allocates an random value matrix using cv::randu for uniform (randMode = false) or gausion noise (randMode = true).
            In case of an integer type, the uniform noise is from min(inclusiv) to max(inclusiv). For floating point types, the noise is between 0(inclusiv) and 1(exclusiv).
            In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0.
    \param dimensions indicates the number of dimensions
    \param *sizes is a vector with the same length than dimensions. Every element indicates the size of the specific dimension
    \param type is the desired data-element-type
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \param continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa OnesFunc
*/
RetVal DataObject::rand(const unsigned char dimensions, const size_t *sizes, const int type, const bool randMode, const unsigned char continuous)
{
   freeData();
   create(dimensions, sizes, type, continuous);

   double val1 = 1.0;
   double val2 = 1.0;

   size_t numMats = calcNumMats();

    if(randMode)
    {
        switch(type)
        {
            case ito::tUInt8:
                val1 = ((double)std::numeric_limits<uint8>::max() + (double)std::numeric_limits<uint8>::min())/2.0;
                val2 = ((double)std::numeric_limits<uint8>::max() - (double)std::numeric_limits<uint8>::min())/6.0;
            break;
            case ito::tInt8:
                val1 = ((double)std::numeric_limits<int8>::max() + std::numeric_limits<int8>::min())/2.0;
                val2 = ((double)std::numeric_limits<int8>::max() - std::numeric_limits<int8>::min())/6.0;
            break;
            case ito::tUInt16:
                val1 = ((double)std::numeric_limits<uint16>::max() + (double)std::numeric_limits<uint16>::min())/2.0;
                val2 = ((double)std::numeric_limits<uint16>::max() - (double)std::numeric_limits<uint16>::min())/6.0;
            break;
            case ito::tInt16:
                val1 = ((double)std::numeric_limits<int16>::max() + (double)std::numeric_limits<int16>::min())/2.0;
                val2 = ((double)std::numeric_limits<int16>::max() - (double)std::numeric_limits<int16>::min())/6.0;
            break;
            case ito::tUInt32:
                val1 = ((double)std::numeric_limits<uint32>::max() + (double)std::numeric_limits<uint32>::min())/2.0;
                val2 = ((double)std::numeric_limits<uint32>::max() - (double)std::numeric_limits<uint32>::min())/6.0;
            break;
            case ito::tInt32:
                val1 = ((double)std::numeric_limits<int32>::max() + (double)std::numeric_limits<int32>::min())/2.0;
                val2 = ((double)std::numeric_limits<int32>::max() - (double)std::numeric_limits<int32>::min())/6.0;
            break;
            default:
                val1 = 0.0;
                val2 = 1.0/3.0;
                break;
        }
    }
    else
    {
        switch(type)
        {
            case ito::tUInt8:
                val1 = (double)std::numeric_limits<uint8>::min();
                val2 = (double)std::numeric_limits<uint8>::max() + 1;
            break;
            case ito::tInt8:
                val1 = (double)std::numeric_limits<int8>::min();
                val2 = (double)std::numeric_limits<int8>::max() + 1;
            break;
            case ito::tUInt16:
                val1 = (double)std::numeric_limits<uint16>::min();
                val2 = (double)std::numeric_limits<uint16>::max() + 1;
            break;
            case ito::tInt16:
                val1 = (double)std::numeric_limits<int16>::min();
                val2 = (double)std::numeric_limits<int16>::max() + 1;
            break;
            case ito::tUInt32:
                val1 = (double)std::numeric_limits<uint32>::min();
                val2 = (double)std::numeric_limits<uint32>::max();
            break;
            case ito::tInt32:
                val1 = (double)std::numeric_limits<int32>::min();
                val2 = (double)std::numeric_limits<int32>::max() + 1;
            break;
            default:
                val1 = 0.0;
                val2 = 1.0;
                break;
        }
    }


    size_t sizeX = sizes[dimensions - 1];
    size_t sizeY = 1;
    if(dimensions > 1)
    {
        sizeY = sizes[dimensions - 2];
    }

    for (size_t matn = 0; matn < numMats; matn++)
    {
        fListRandFunc[type](sizeY, sizeX, val1, val2, randMode, &(m_data[matn]));
    }

   return 0;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method that executes a shallow-copy of every matrix-plane in the source-vector and stores the copies in the destination-vector
/*!
    \param &src is the source vector which contains matrix-planes of type cv::Mat_<_Tp>
    \param &dst is the destination vector, where the shallow-copies are stored. dst should be empty at the beginning
    \return retOk
    \sa operator =, DataObject::DataObject(const DataObject& copyConstr)
*/
//template<typename _Tp> RetVal CopyMatFunc(const std::vector<int *> &src, std::vector<int *> &dst)
template<typename _Tp> RetVal CopyMatFunc(int **src, int **&dst, bool transposed)
{
    static size_t sizeofs = sizeof(size_t) / sizeof(int *);
    size_t size = (*reinterpret_cast<size_t *>(src - sizeofs));

    if (dst)
    {
        dst = reinterpret_cast<int **>(realloc((dst - sizeofs), size * sizeof(int *) + sizeof(size_t)));
    }
    else
    {
        size_t numBytes = size * sizeof(int *) + sizeof(size_t);
        dst = reinterpret_cast<int **>(calloc(numBytes, 1));
        memset(dst, 0, numBytes);
    }

    (*reinterpret_cast<size_t *>(dst)) = size;
    dst += sizeofs;

	if(transposed)
	{
		for( size_t i = 0 ; i < size ; i++)
		{
			dst[i] = reinterpret_cast<int*>( new cv::Mat_<_Tp>( ((cv::Mat_<_Tp> *)(src[i]))->t() ) );
		}
	}
	else
	{
		for( size_t i = 0 ; i < size ; i++)
		{
			dst[i] = reinterpret_cast<int*>( new cv::Mat_<_Tp>( *((cv::Mat_<_Tp> *)(src[i])) ) );
		}
	}

   return RetVal(retOk);
}

//typedef RetVal (*tCopyMatFunc)(const std::vector<int *> &src, std::vector<int *> &dst);
typedef RetVal (*tCopyMatFunc)(int **src, int  **&dst, bool transposed);
MAKEFUNCLIST(CopyMatFunc)


//! assign-operator which creates a two-dimensional data object as a shallow copy of a two dimensional cv::Mat object.
/*!
    shallow-copy means, that the header information of this data-object is physically created at the hard disk, while the data is shared
    with the original cv::Mat.

    \param &rhs is the cv::Mat where the shallow copy is taken from. At first, the existing data of this object is freed.
    \return this data object
    \throws cv::Exception if rhs is not two-dimensional or data type has no compatible data type of dataObject.
    \sa create
*/
DataObject & DataObject::operator = (const cv::Mat &rhs)
{
    int dataObjType = -1;

    //check data type of rhs
    switch(rhs.type())
    {
    case CV_8UC1: dataObjType = ito::tUInt8; break;
    case CV_8SC1: dataObjType = ito::tInt8; break;
    case CV_16UC1: dataObjType = ito::tUInt16; break;
    case CV_16SC1: dataObjType = ito::tInt16; break;
    //case CV_32UC1: dataObjType = ito::tUInt32; break; //does not exist in OpenCV cv::Mat
    case CV_32SC1: dataObjType = ito::tInt32; break;
    case CV_32FC1: dataObjType = ito::tFloat32; break;
    case CV_64FC1: dataObjType = ito::tFloat64; break;
    case CV_32FC2: dataObjType = ito::tComplex64; break;
    case CV_64FC2: dataObjType = ito::tComplex128; break;
    default: dataObjType = -1;
    }

    if(dataObjType == -1)
    {
        cv::error(cv::Exception(CV_StsAssert,"data type of cv::Mat is not compatible to dataObject.","", __FILE__, __LINE__));
    }

    if(rhs.dims != 2)
    {
        cv::error(cv::Exception(CV_StsAssert,"cv::Mat must have two dimensions.","", __FILE__, __LINE__));
    }

    freeData();

    const size_t sizes[2] = {(size_t)rhs.rows, (size_t)rhs.cols};
    create(2, sizes, dataObjType, &rhs, 1);

    return *this;
}

//! assign-operator which makes a shallow-copy of the rhs data object and stores it in this data object
/*!
    shallow-copy means, that the header information of the rhs data-object is physically copied to this-dataObject while
    the data is shared, hence, only its reference counter is incremented

    \param &rhs is the data object where the shallow copy is taken from. At first, the existing data of this object is freed.
    \return this data object
    \throws cv::Exception if lock state of both objects is not equal. Please make sure, that both lock states are equal
    \sa CopyMatFunc
*/
DataObject & DataObject::operator = (const DataObject &rhs)
{
    if(rhs.m_objSharedDataLock != NULL && rhs.m_objSharedDataLock->getLockStatus() == -1)
    {
        cv::error(cv::Exception(CV_StsAssert,"data of assigned data object may not be locked for writing","", __FILE__, __LINE__));
    }

   if (this == &rhs)
   {
      return *this;
   }

   freeData();

   createHeaderWithROI(rhs.m_dims, rhs.m_size.m_p, rhs.m_osize.m_p, rhs.m_roi.m_p);
   m_type = rhs.m_type;
   m_continuous = rhs.m_continuous;
//   m_owndata = m_owndata;
   m_pRefCount = rhs.m_pRefCount;

   if(rhs.m_dims > 0 || m_pRefCount)
   {
        CV_XADD((m_pRefCount),1); //++;

        m_objSharedDataLock = rhs.m_objSharedDataLock;
        m_pDataObjectTags = rhs.m_pDataObjectTags;

        int status = m_objHeaderLock.getLockStatus();

        if(status == -1) //this object is in write lock, then increment shared data lock for reading (for writing not possible since rhs also has access to the data)
        {
            m_objSharedDataLock->lockRead();
        }
        else if(status > 0)
        {
            m_objSharedDataLock->lockRead(status);
        }

        try
        {
            fListCopyMatFunc[m_type](rhs.get_mdata(), m_data, false);
        }
        catch(cv::Exception exc) //memory error
        {
            secureFreeData();
            throw; //rethrow exception
        }

   } //else: rhs is empty

   return *this;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! copy constructor for data object
/*!
    creates a data object with respect to the given data object. The header information is completely copied, while the data is
    a shallow copy. The lock of the new data object is unlocked while the lock for the common data block is taken from the current lock
    status of the given data object.

    \param &copyConstr is the data object, which will be copied
*/
DataObject::DataObject(const DataObject& copyConstr) : m_pRefCount(0), m_dims(0), m_data(NULL)
{
    /*if(copyConstr.m_objSharedDataLock->getLockStatus() == -1)
    {
        cv::error(cv::Exception(CV_StsAssert,"data of copyConstr may not be locked for writing","", __FILE__, __LINE__));
    }*/

    createHeaderWithROI(copyConstr.m_dims, copyConstr.m_size.m_p, copyConstr.m_osize.m_p, copyConstr.m_roi.m_p);
    m_pRefCount = copyConstr.m_pRefCount;
    m_objSharedDataLock = copyConstr.m_objSharedDataLock; //copies pointer
    m_pDataObjectTags = copyConstr.m_pDataObjectTags; // Make a shallowCopy of the TagSpace

    //header lock for new dataObject is unlocked, therefore does not increment shared data lock, since nobody is decrement this new object


    // only increment if data exists
    if(m_pRefCount != NULL)
    {
        CV_XADD((m_pRefCount),1);//++;
    }

    m_type = copyConstr.m_type;
    m_continuous = copyConstr.m_continuous;
    m_owndata = copyConstr.m_owndata;

    if(copyConstr.m_data != NULL)
    {
        mdata_realloc(const_cast<DataObject &>(copyConstr).mdata_size());

        try
        {
            fListCopyMatFunc[m_type](copyConstr.m_data, m_data, false);
        }
        catch(cv::Exception exc) //memory error
        {
            secureFreeData();
            throw; //rethrow exception
        }
    }
};


DataObject::DataObject(const DataObject& dObj, bool transposed)
{
	if(!transposed) //shallow copy of dataobject
	{
		createHeaderWithROI(dObj.m_dims, dObj.m_size.m_p, dObj.m_osize.m_p, dObj.m_roi.m_p);
		m_pRefCount = dObj.m_pRefCount;
		m_objSharedDataLock = dObj.m_objSharedDataLock; //copies pointer
		m_pDataObjectTags = dObj.m_pDataObjectTags; // Make a shallowCopy of the TagSpace

		//header lock for new dataObject is unlocked, therefore does not increment shared data lock, since nobody is decrement this new object


		// only increment if data exists
		if(m_pRefCount != NULL)
		{
			CV_XADD((m_pRefCount),1);//++;
		}

		m_type = dObj.m_type;
		m_continuous = dObj.m_continuous;
		m_owndata = dObj.m_owndata;

		if(dObj.m_data != NULL)
		{
			mdata_realloc(const_cast<DataObject &>(dObj).mdata_size());

			try
			{
				fListCopyMatFunc[m_type](dObj.m_data, m_data, false);
			}
			catch(cv::Exception exc) //memory error
			{
				secureFreeData();
				throw; //rethrow exception
			}
		}
	}
	else //deep, transposed copy of dataObject
	{
		int dims = dObj.m_dims;
		size_t *newSize = new size_t[dims];
		size_t *newOSize = new size_t[dims];
		size_t *newRoi = new size_t[dims];
		memcpy(newSize, dObj.m_size.m_p, dims * sizeof(size_t));
		memcpy(newOSize, dObj.m_osize.m_p, dims * sizeof(size_t));
		memcpy(newRoi, dObj.m_roi.m_p, dims * sizeof(size_t));

		//flip the last two dimensions
		if(dims >= 2)
		{
			std::swap( newSize[dims-1], newSize[dims-2] );
			std::swap( newOSize[dims-1], newOSize[dims-2] );
			std::swap( newRoi[dims-1], newRoi[dims-2] );
		}

		createHeaderWithROI(dims, newSize, newOSize, newRoi); 
		delete[] newSize;
		delete[] newOSize;
		delete[] newRoi;

		if (dims > 0)
		{
			m_pRefCount = new int(0);
			m_objSharedDataLock = new ReadWriteLock(m_objHeaderLock.getLockStatus());

			m_pDataObjectTags = new DataObjectTags( *dObj.m_pDataObjectTags ); //deep copy of tags

			//flip last two elements of axisDescription, axisOffsets, axisScale, axisUnit
			if(dims >= 2)
			{
				std::swap(m_pDataObjectTags->m_axisDescription[dims-1], m_pDataObjectTags->m_axisDescription[dims-2]);
				std::swap(m_pDataObjectTags->m_axisOffsets[dims-1], m_pDataObjectTags->m_axisOffsets[dims-2]);
				std::swap(m_pDataObjectTags->m_axisScales[dims-1], m_pDataObjectTags->m_axisScales[dims-2]);
				std::swap(m_pDataObjectTags->m_axisUnit[dims-1], m_pDataObjectTags->m_axisUnit[dims-2]);
			}
		}
		else
		{
			m_pRefCount = NULL;
		}

		m_type = dObj.m_type;
		m_continuous = dObj.m_continuous;
		m_owndata = dObj.m_owndata;

		m_data = NULL;
		mdata_realloc(const_cast<DataObject &>(dObj).mdata_size());

		try
		{
			fListCopyMatFunc[m_type](dObj.m_data, m_data, true);
		}
		catch(cv::Exception exc) //memory error
		{
			secureFreeData();
			throw; //rethrow exception
		}
	}
}


//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated helper method to assign the given scalar to every element within its ROI in DataObject src.
/*!
    The scalar value is converted to the type of the source data object

    \param *src is the source data object whose elements will be modified
    \param type is the data type of the scalar value
    \param *scalar is a void pointer to the scalar value
    \return retOk
    \throws cv::exception if conversion of scalar to dataObject's type is not possible
    \sa numberConcversion
*/
template<typename _Tp> RetVal AssignScalarFunc(const DataObject *src, const ito::tDataType type, const void *scalar)
{
    size_t numMats = src->calcNumMats();
    size_t MatNum = 0;

    _Tp scalar2 = ito::numberConversion<_Tp>(type, const_cast<void*>(scalar)); //convert the void* scalar to the data type of the source data object (throws error if not possible)

    cv::Mat_<_Tp> * tempMat = NULL;
    int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
    int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
       //TODO: check if non iterator version is working
        MatNum = src->seekMat(nmat, numMats);
        tempMat = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]));

        for (int y = 0; y < sizey; y++)
        {
            _Tp* dstPtr = (_Tp*)tempMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = (_Tp)scalar2;
            }
        }

//        cv::MatIterator_<_Tp> lhsIt = tempMat->begin();
//        cv::MatConstIterator_<_Tp> lhsIt_end = tempMat->end();

//        for (; lhsIt != lhsIt_end; ++lhsIt)
//        {
//            (*lhsIt) = scalar2;
//        }
    }

    return 0;
}

typedef RetVal (*tAssignScalarFunc)(const DataObject *src, const ito::tDataType type, const void *scalar);
MAKEFUNCLIST(AssignScalarFunc);

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const int8 value)
{
    fListAssignScalarFunc[m_type](this, ito::tInt8, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const uint8 value)
{
    fListAssignScalarFunc[m_type](this, ito::tUInt8, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const int16 value)
{
    fListAssignScalarFunc[m_type](this, ito::tInt16, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const uint16 value)
{
    fListAssignScalarFunc[m_type](this, ito::tUInt16, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const int32 value)
{
    fListAssignScalarFunc[m_type](this, ito::tInt32, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const uint32 value)
{
    fListAssignScalarFunc[m_type](this, ito::tUInt32, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const float32 value)
{
    fListAssignScalarFunc[m_type](this, ito::tFloat32, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const float64 value)
{
    fListAssignScalarFunc[m_type](this, ito::tFloat64, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const complex64 value)
{
    fListAssignScalarFunc[m_type](this, ito::tComplex64, static_cast<const void*>(&value));
    return *this;
};

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const complex128 value)
{
    fListAssignScalarFunc[m_type](this, ito::tComplex128, static_cast<const void*>(&value));
    return *this;
};


//----------------------------------------------------------------------------------------------------------------------------------
// arithmetic operators
//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for element-wise addition of two given data objects.
/*!
    dObjRes = dObj1 + dObj2

    \param *dObj1 is the first data object
    \param *dObj2 is the second data object
    \param *dObjRes is the pointer to the data object, where the values will be written to. This data object must already be allocated.
    \remark The size check for all data objects must be done before.
    \return retOk
    \sa operator +=, operator +
*/
template<typename _Tp> RetVal AddFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
   size_t srcTmat1 = 0;
   size_t srcTmat2 = 0;
   size_t dstTmat = 0;
   size_t numMats = dObj1->calcNumMats();
   cv::Mat_<_Tp> *cvSrcTmat1 = NULL;
   cv::Mat_<_Tp> *cvSrcTmat2 = NULL;
   cv::Mat_<_Tp> *cvDstTmat = NULL;

    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
		dstTmat = dObjRes->seekMat(nmat, numMats);
		srcTmat1 = dObj1->seekMat(nmat, numMats);
		srcTmat2 = dObj2->seekMat(nmat, numMats);
		cvSrcTmat1 = (cv::Mat_<_Tp> *) dObj1->get_mdata()[srcTmat1];
		cvSrcTmat2 = (cv::Mat_<_Tp> *) dObj2->get_mdata()[srcTmat2];
		cvDstTmat = (cv::Mat_<_Tp> *) dObjRes->get_mdata()[dstTmat];
		*cvDstTmat = *cvSrcTmat1 + *cvSrcTmat2;
    }

   return RetVal(retOk);
}

typedef RetVal (*tAddFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);
MAKEFUNCLIST(AddFunc);

template<typename _Tp> RetVal AddScalarFunc(const DataObject *dObjIn, ito::float64 scalar, DataObject *dObjOut)
{
   size_t srcTmat = 0;
   size_t dstTmat = 0;
   size_t numMats = dObjIn->calcNumMats();
   cv::Mat_<_Tp> *cvSrc = NULL;
   cv::Mat_<_Tp> *cvDest = NULL;
   cv::Scalar s = scalar;

    for (size_t nmat = 0; nmat < numMats; ++nmat)
    {
		dstTmat = dObjOut->seekMat(nmat, numMats);
		srcTmat = dObjIn->seekMat(nmat, numMats);
		cvSrc  = (cv::Mat_<_Tp> *) dObjIn->get_mdata()[srcTmat];
		cvDest = (cv::Mat_<_Tp> *) dObjOut->get_mdata()[dstTmat];
		*cvDest = *cvSrc + s;
    }

   return RetVal(retOk);
}

typedef RetVal (*tAddScalarFunc)(const DataObject *dObjIn, ito::float64 scalar, DataObject *dObjOut);
MAKEFUNCLIST(AddScalarFunc);

//----------------------------------------------------------------------------------------------------------------------------------
//! high-level, non-templated arithmetic operator for element-wise addition of values of given data object to this data object
/*!
    \param &rhs is the data object whose elements will be added to this data object
    \return this data object
    \throws cv::Exception if both data objects don't have the same size or type
    \sa AddFunc
*/
DataObject & DataObject::operator += (const DataObject &rhs)
{
    if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
        return *this;
    }

    (fListAddFunc[m_type])(this, &rhs, this);
    return *this;
}

DataObject & DataObject::operator += (const float64 value)
{
    (fListAddScalarFunc[m_type])(this, value, this);
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! high-level, non-templated arithmetic operator for element-wise addition of values of two given data objects
/*!
    \param &rhs is the data object whose elements will be added to this data object
    \return new resulting data object
    \throws cv::Exception if both data objects don't have the same size or type
    \sa AddFunc
*/
DataObject DataObject::operator + (const DataObject &rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   DataObject result;
   result.m_continuous = rhs.m_continuous;
   copyTo(result, 1);

   (fListAddFunc[m_type])(this, &rhs, &result);

   return result;
}

DataObject DataObject::operator + (const float64 value)
{
    DataObject result;
    result.m_continuous = this->m_continuous;
    copyTo(result, 1);

    (fListAddScalarFunc[m_type])(this, value, &result);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for element-wise subtraction of values from second data object from values of first data object
/*!
    dObjRes = dObj1 - dObj2

    \param *dObj1 is the first data object
    \param *dObj2 is the second data object
    \param *dObjRes is the pointer to the data object, where the values will be written to. This data object must already be allocated.
    \remark The size check for all data objects must be done before.
    \return retOk
    \sa operator -=, operator -
*/
template<typename _Tp> RetVal SubFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
   size_t nmat = 0;
   size_t srcTmat1 = 0;
   size_t srcTmat2 = 0;
   size_t dstTmat = 0;
   size_t numMats = dObj1->calcNumMats();
   cv::Mat_<_Tp> *cvSrcTmat1 = NULL;
   cv::Mat_<_Tp> *cvSrcTmat2 = NULL;
   cv::Mat_<_Tp> *cvDstTmat = NULL;

	for (nmat = 0; nmat < numMats; nmat++)
	{
		dstTmat = dObjRes->seekMat(nmat, numMats);
		srcTmat1 = dObj1->seekMat(nmat, numMats);
		srcTmat2 = dObj2->seekMat(nmat, numMats);
		cvSrcTmat1 = (cv::Mat_<_Tp> *) dObj1->get_mdata()[srcTmat1];
		cvSrcTmat2 = (cv::Mat_<_Tp> *) dObj2->get_mdata()[srcTmat2];
		cvDstTmat = (cv::Mat_<_Tp> *) dObjRes->get_mdata()[dstTmat];
		*cvDstTmat = *cvSrcTmat1 - *cvSrcTmat2;
	}
 
   return 0;
}

typedef RetVal (*tSubFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);

MAKEFUNCLIST(SubFunc);

//----------------------------------------------------------------------------------------------------------------------------------
//! high-level, non-templated arithmetic operator for element-wise subtraction of values of given data object from values of this data object
/*!
    \param &rhs is the data object whose elements will be subtracted from this data object
    \return this data object
    \throws cv::Exception if both data objects don't have the same size or type
    \sa SubFunc
*/
DataObject & DataObject::operator -= (const DataObject &rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   fListSubFunc[m_type](this, &rhs, this);

   return *this;
};

DataObject & DataObject::operator -= (const float64 value)
{
    (fListAddScalarFunc[m_type])(this, -value, this);
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! high-level, non-templated arithmetic operator for element-wise subtraction of values of given data object from values of this data object
/*!
    \param &rhs is the data object whose elements will be subtracted from this data object
    \return new resulting data object
    \throws cv::Exception if both data objects don't have the same size or type
    \sa SubFunc
*/
DataObject DataObject::operator - (const DataObject &rhs)
{
    if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
        return *this;
    }

    DataObject result;
    result.m_continuous = rhs.m_continuous;
    this->copyTo(result, 1);

    (fListSubFunc[m_type])(this, &rhs, &result);

    return result;
};

DataObject DataObject::operator - (const float64 value)
{
    DataObject result;
    result.m_continuous = this->m_continuous;
    copyTo(result, 1);

    (fListAddScalarFunc[m_type])(this, -value, &result);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! brief description
/*!
    \todo check for right definition of multiplication
*/
template<typename _Tp> RetVal OpMulFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
   size_t nmat = 0;
   size_t srcTmat1 = 0;
   size_t srcTmat2 = 0;
   size_t dstTmat = 0;
   size_t numMats = dObj1->calcNumMats();
   cv::Mat_<_Tp> *cvSrcTmat1 = NULL;
   cv::Mat_<_Tp> *cvSrcTmat2 = NULL;
   cv::Mat_<_Tp> *cvDstTmat = NULL;
  
    for (nmat = 0; nmat < numMats; nmat++)
    {
		dstTmat = dObjRes->seekMat(nmat, numMats);
		srcTmat1 = dObj1->seekMat(nmat, numMats);
		srcTmat2 = dObj2->seekMat(nmat, numMats);
		cvSrcTmat1 = (cv::Mat_<_Tp> *) dObj1->get_mdata()[srcTmat1];
		cvSrcTmat2 = (cv::Mat_<_Tp> *) dObj2->get_mdata()[srcTmat2];
		cvDstTmat = (cv::Mat_<_Tp> *) dObjRes->get_mdata()[dstTmat];
		*cvDstTmat = *cvSrcTmat1 * *cvSrcTmat2;
    }

   cv::Size msize = ((cv::Mat_<_Tp> *)(dObjRes->get_mdata()[0]))->size(); //for dObjRes, the transpose flag is 0.
   dObjRes->getSize().m_p[dObjRes->getDims() - 1] = msize.width;

   if(dObjRes->getDims() >= 2)
   {
    dObjRes->getSize().m_p[dObjRes->getDims() - 2] = msize.height;
   }

   return 0;
}

typedef RetVal (*tOpMulFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);

MAKEFUNCLIST(OpMulFunc);

//----------------------------------------------------------------------------------------------------------------------------------
//! brief description
/*!
    \todo think about definition for this operator
*/
DataObject & DataObject::operator *= (const DataObject &rhs)
{
   if (this->m_type != rhs.m_type)
   {
      cv::error(cv::Exception(CV_StsAssert, "Data type of objects different", "", __FILE__, __LINE__));
      return *this;
   }

   if ((m_size[m_dims - 1] != rhs.m_size[rhs.m_dims - 2]) || (m_dims > 2) || (rhs.m_dims > 2))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - matrix dimensions inapropriate for matrix multiplication","", __FILE__, __LINE__));
   }

   if(this->m_type != ito::tFloat32 && this->m_type != ito::tFloat64)
   {
       cv::error(cv::Exception(CV_StsAssert,"matrix multiplication is only implemented for float32 and float64 (due to openCV or BLAS restrictions)","",__FILE__,__LINE__));
   }

   fListOpMulFunc[m_type](this, &rhs, this);

   return *this;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! brief description
/*!
    \todo think about definition for this operator
*/
DataObject DataObject::operator * (const DataObject &rhs)
{
   if ((m_size[m_dims - 1] != rhs.m_size[rhs.m_dims - 2]) || (m_dims > 2) || (rhs.m_dims > 2))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - matrix dimensions inapropriate for matrix multiplication","", __FILE__, __LINE__));
      return *this;
   }

   if(this->m_type != ito::tFloat32 && this->m_type != ito::tFloat64)
   {
       cv::error(cv::Exception(CV_StsAssert,"matrix multiplication is only implemented for float32 and float64 (due to openCV or BLAS restrictions)","",__FILE__,__LINE__));
   }

   DataObject result;
   result.m_continuous = rhs.m_continuous;
   this->copyTo(result, 1);

   result *= rhs;
   return result;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which multiplies every element of Data Object with a factor
/*!
    \param *src
    \param factor
    \return retOk
*/
template<typename _Tp> RetVal OpScalarMulFunc(const DataObject *src, const double factor)
{
   size_t numMats = src->calcNumMats();
   size_t MatNum = 0;

   _Tp factor2 = cv::saturate_cast<_Tp>(factor);

   cv::Mat_<_Tp> * tempMat = NULL;
   int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
   int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
      //TODO: check if non iterator version is working
       MatNum = src->seekMat(nmat, numMats);
       tempMat = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]));

       for (int y = 0; y < sizey; y++)
       {
           _Tp* dstPtr = (_Tp*)tempMat->ptr(y);
           for (int x = 0; x < sizex; x++)
           {
               dstPtr[x] *= (_Tp)factor2;
           }
       }

//      cv::MatIterator_<_Tp> lhsIt = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]))->begin();
//      cv::MatConstIterator_<_Tp> lhsIt_end = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]))->end();
//
//      for (; lhsIt != lhsIt_end; ++lhsIt)
//      {
//         (*lhsIt) *= factor2;
//      }
   }

   return 0;
}

typedef RetVal (*tOpScalarMulFunc)(const DataObject *src, const double factor);
MAKEFUNCLIST(OpScalarMulFunc);

//! high-level method which multiplies every element in this data object by a given floating-point factor
/*!
    \param factor
    \sa OpScalarMulFunc
*/
DataObject & DataObject::operator *= (const double factor)
{
   fListOpScalarMulFunc[m_type](this, factor);

   return *this;
};

//! high-level method which multiplies every element in this data object by a given floating-point factor. The result matrix is returned as a new matrix.
/*!
    \param factor
    \sa operator *, OpScalarMulFunc
*/
DataObject DataObject::operator * (const double factor)
{
   DataObject result;
   result.m_continuous = (*this).m_continuous;
   this->copyTo(result, 1);

   result *= factor;
   return result;
};


//----------------------------------------------------------------------------------------------------------------------------------
// comparison operators
//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which compares each element in source-matrix1 with its corresponding element in source-matrix2 and saves the result in a destionation matrix
/*!
    \param *src1 is the first source matrix
    \param *src2 is the second source matrix
    \param *dst is the destination matrix, which must have the same ROI than src1 and src2 and must be of type uint8
    \param cmpOp is the compare operator (cv::CMP_EQ, cv::CMP_GT, cv::CMP_GE, cv::CMP_LT, cv::CMP_LE, cv::CMP_NE)
    \remark no comparison is possible for source matrices of type int8 (due to openCV-problems)
    \throws cv::Exception if source matrix is of type int8
    \return retOk
*/
template<typename _Tp> RetVal CmpFunc(const DataObject *src1, const DataObject *src2, DataObject *dst, int cmpOp)
{
   size_t numMats = src1->calcNumMats();
   size_t lhsMatNum = 0;
   size_t rhsMatNum = 0;
   size_t resMatNum = 0;

   cv::Mat_<_Tp> *src1mat;
   cv::Mat_<_Tp> *src2mat;
   cv::Mat_<uint8> *dest;

   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = src1->seekMat(nmat, numMats);
      rhsMatNum = src2->seekMat(nmat, numMats);
      resMatNum = dst->seekMat(nmat, numMats);

      src1mat = (cv::Mat_<_Tp> *)(src1->get_mdata())[lhsMatNum];
      src2mat = (cv::Mat_<_Tp> *)(src2->get_mdata())[rhsMatNum];
      dest = (cv::Mat_<uint8> *)(dst->get_mdata())[resMatNum];

      if(src1mat->depth() == 1 || src1mat->depth() == 7)
      {
          cv::error(cv::Exception(CV_StsAssert, "Compare operator not defined for int8.", "", __FILE__, __LINE__));
      }      
	  (*(cv::Mat_<char> *)((dst->get_mdata())[resMatNum])) =   src1mat < src2mat;
	  cv::compare(*src1mat, *src2mat, *dest, cmpOp);
   }

   return 0;
}

//! template specialisation for compare function of type complex64
/*!
    \throws cv::Exception since comparison is not defined for complex input types
*/
template<> RetVal CmpFunc<ito::complex64>(const DataObject * /*src1*/, const DataObject * /*src2*/, DataObject * /*dst*/, int /*cmpOp*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for compare function of type complex128
/*!
    \throws cv::Exception since comparison is not defined for complex input types
*/
template<> RetVal CmpFunc<ito::complex128>(const DataObject * /*src1*/, const DataObject * /*src2*/, DataObject * /*dst*/, int /*cmpOp*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tCmpFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst, int cmpOp);
MAKEFUNCLIST(CmpFunc);

//! compare operator, compares for "lower than"
/*!
    \param &rhs is the data object with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator < (DataObject &rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous | rhs.m_continuous);
   RetVal retValue = fListCmpFunc[m_type](this, &rhs, &resMat, cv::CMP_LT);

   return resMat;
}

//! compare operator, compares for "bigger than"
/*!
    \param &rhs is the data object with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator > (DataObject &rhs)
{
   return rhs < *this;
}

//! compare operator, compares for "lower or equal than"
/*!
    \param &rhs is the data object with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator <= (DataObject &rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous | rhs.m_continuous);
   RetVal retValue = fListCmpFunc[m_type](this, &rhs, &resMat, cv::CMP_LE);

   return resMat;
}

//! compare operator, compares for "bigger or equal than"
/*!
    \param &rhs is the data object with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator >= (DataObject &rhs)
{
   return rhs <= *this;
}

//! compare operator, compares for "equal to"
/*!
    \param &rhs is the data object with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator == (DataObject &rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
       cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
       return *this;
   }

   DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous | rhs.m_continuous);
   RetVal retValue = fListCmpFunc[m_type](this, &rhs, &resMat, cv::CMP_EQ);

   return resMat;
}

//! compare operator, compares for "unequal to"
/*!
    \param &rhs is the data object with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator != (DataObject &rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous | rhs.m_continuous);
   RetVal retValue = fListCmpFunc[m_type](this, &rhs, &resMat, cv::CMP_NE);

   return resMat;
}

//----------------------------------------------------------------------------------------------------------------------------------
// bitshift operators
//----------------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which element-wisely shifts the values of the source matrix by a certain value to the left
/*!
    \param *src
    \param shiftbit are the number bits the values are shifted
    \throws cv::Exception for unsupported data types (template specialization)
    \return retOk
*/
template<typename _Tp> RetVal ShiftLFunc(DataObject *src, const unsigned char shiftbit)
{
   size_t numMats = src->calcNumMats();
   size_t MatNum = 0;

   cv::Mat_<_Tp> * tempMat = NULL;
   int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
   int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
      MatNum = src->seekMat(nmat, numMats);
      //TODO: check if non iterator version is working
      tempMat = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]));

      for (int y = 0; y < sizey; y++)
      {
          _Tp* dstPtr = (_Tp*)tempMat->ptr(y);
          for (int x = 0; x < sizex; x++)
          {
              dstPtr[x] <<= shiftbit;
          }
      }

//      cv::MatIterator_<_Tp> lhsIt = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]))->begin();
//      cv::MatConstIterator_<_Tp> lhsIt_end = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]))->end();
//
//      for (; lhsIt != lhsIt_end; ++lhsIt)
//      {
//         (*lhsIt) <<= shiftbit;
//      }
   }

   return 0;
}

//! template specialisation for shift function of type float32
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftLFunc<ito::float32>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for shift function of type float64
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftLFunc<ito::float64>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for shift function of type complex64
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftLFunc<ito::complex64>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for shift function of type complex128
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftLFunc<ito::complex128>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tShiftLFunc)(DataObject *src, const unsigned char shiftbit);
MAKEFUNCLIST(ShiftLFunc);

//! high-level operator, which shifts the elements of this data objects by a given number of bits to the left
/*!
    \param shiftbit defines the number of bits to shift
    \return reference to this data object
    \sa ShiftLFunc
*/
DataObject & DataObject::operator <<= (const unsigned int shiftbit)
{
   if (shiftbit == 0)
   {
      return *this;
   }

   fListShiftLFunc[m_type](this, shiftbit);

   return (*this);
}

//! high-level operator, which shifts the elements of this data objects by a given number of bits to the left and returns the new data object
/*!
    \param shiftbit defines the number of bits to shift
    \return new data object with shifted values
    \sa operator <<=, ShiftLFunc
*/
DataObject DataObject::operator << (const unsigned int shiftbit)
{
   if (shiftbit == 0)
   {
      return *this;
   }

   DataObject result;
   this->copyTo(result, 1);

   result <<= shiftbit;
   return result;
}



//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which element-wisely shifts the values of the source matrix by a certain value to the right
/*!
    \param *src
    \param shiftbit are the number bits the values are shifted
    \throws cv::Exception for unsupported data types (template specialization)
    \return retOk
*/
template<typename _Tp> RetVal ShiftRFunc(DataObject *src, const unsigned char shiftbit)
{
   size_t numMats = src->calcNumMats();
   size_t MatNum = 0;

   cv::Mat_<_Tp> * tempMat = NULL;
   int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
   int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
        MatNum = src->seekMat(nmat, numMats);
        //TODO: check if non iterator version is working
        tempMat = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]));

        for (int y = 0; y < sizey; y++)
        {
            _Tp* dstPtr = (_Tp*)tempMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] >>= shiftbit;
            }
        }

//        MatNum = src->seekMat(nmat, numMats);
//        cv::MatIterator_<_Tp> lhsIt = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]))->begin();
//        cv::MatConstIterator_<_Tp> lhsIt_end = ((cv::Mat_<_Tp> *)((src->get_mdata())[MatNum]))->end();
//
//        for (; lhsIt != lhsIt_end; ++lhsIt)
//        {
//            (*lhsIt) >>= shiftbit;
//        }
   }

   return 0;
}

//! template specialisation for shift function of type float32
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftRFunc<ito::float32>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for shift function of type float64
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftRFunc<ito::float64>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for shift function of type complex64
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftRFunc<ito::complex64>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for shift function of type complex128
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftRFunc<ito::complex128>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tShiftRFunc)(DataObject *src, const unsigned char shiftbit);
MAKEFUNCLIST(ShiftRFunc)

//! high-level operator, which shifts the elements of this data objects by a given number of bits to the right
/*!
    \param shiftbit defines the number of bits to shift
    \return reference to this data object
    \sa ShiftRFunc
*/
DataObject & DataObject::operator >>= (const unsigned int shiftbit)
{
   if (shiftbit == 0)
   {
      return *this;
   }

    fListShiftRFunc[m_type](this, shiftbit);

   return (*this);
}

//! high-level operator, which shifts the elements of this data objects by a given number of bits to the right and returns the new data object
/*!
    \param shiftbit defines the number of bits to shift
    \return new data object with shifted values
    \sa operator >>=, ShiftRFunc
*/
DataObject DataObject::operator >> (const unsigned int shiftbit)
{
   if (shiftbit == 0)
   {
      return *this;
   }

   DataObject result;
   this->copyTo(result, 1);

   result >>= shiftbit;
   return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
// bitwise operators
//----------------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which element-wisely executes a bitwise 'and' comparison between values of two dataObjects.
/*!
    \param *dObj1 is the first data object
    \param *dObj2 is the second data object
    \param *dObjRes is the data object, where the result is stored
    \throws cv::Exception for unsupported data types (template specialization)
    \return retOk
*/
template<typename _Tp> RetVal BitAndFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
   size_t numMats = dObj1->calcNumMats();
   size_t lhsMatNum = 0;
   size_t rhsMatNum = 0;
   size_t resMatNum = 0;

   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = dObj1->seekMat(nmat, numMats);
      rhsMatNum = dObj2->seekMat(nmat, numMats);
      resMatNum = dObjRes->seekMat(nmat, numMats);
	  (*((cv::Mat_<_Tp> *)(dObjRes->get_mdata())[resMatNum])) = (*((cv::Mat_<_Tp> *)(dObj1->get_mdata())[lhsMatNum])) & (*((cv::Mat_<_Tp> *)(dObj2->get_mdata())[rhsMatNum]));      
   }
   return 0;
}

//! template specialisation for bitwise and function of type float32
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitAndFunc<ito::float32>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise and function of type float64
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitAndFunc<ito::float64>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise and function of type complex64
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitAndFunc<ito::complex64>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise and function of type complex128
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitAndFunc<ito::complex128>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tBitAndFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);

MAKEFUNCLIST(BitAndFunc)

//! high-level operator, which executes the element-wise operation "bitwise and" between this data object and a given data object
/*!
    \param &rhs is the matrix which is used for the operator
    \return reference to this data object, where the result of the operation is stored
    \throws cv::Exception if data type is not supported or both data objects differs either in their size or data type
    \sa BitAndFunc
*/
DataObject & DataObject::operator &= (const DataObject & rhs)
{
   if (this == &rhs)
   {
      return *this;
   }

   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   fListBitAndFunc[m_type](this, &rhs, this);

   return (*this);
}

//! high-level operator, which executes the element-wise operation "bitwise and" between this data object and a given data object
/*!
    the result is returned as a newly allocated data object.
    \param &rhs is the matrix which is used for the operator
    \return new data object, where the result of the operation is stored
    \throws cv::Exception if data type is not supported or both data objects differs either in their size or data type
    \sa operator &=, BitAndFunc
*/
DataObject DataObject::operator & (const DataObject & rhs)
{
   if (this == &rhs)
   {
      return *this;
   }

   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
      return *this;
   }

   DataObject result;
   result.m_continuous |= rhs.m_continuous;
   this->copyTo(result, 1);

   result &= rhs;
   return result;
}



//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which element-wisely executes a bitwise 'or' comparison between values of two dataObjects.
/*!
    \param *dObj1 is the first data object
    \param *dObj2 is the second data object
    \param *dObjRes is the data object, where the result is stored
    \throws cv::Exception for unsupported data types (template specialization)
    \return retOk
*/
template<typename _Tp> RetVal BitOrFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
   size_t numMats = dObj1->calcNumMats();
   size_t lhsMatNum = 0;
   size_t rhsMatNum = 0;
   size_t resMatNum = 0;

   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = dObj1->seekMat(nmat, numMats);
      rhsMatNum = dObj2->seekMat(nmat, numMats);
      resMatNum = dObjRes->seekMat(nmat, numMats);
	  (*((cv::Mat_<_Tp> *)(dObjRes->get_mdata())[resMatNum])) = (*((cv::Mat_<_Tp> *)(dObj1->get_mdata())[lhsMatNum])) | (*((cv::Mat_<_Tp> *)(dObj2->get_mdata())[rhsMatNum]));
   }
   return 0;
}

//! template specialisation for bitwise or function of type float32
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitOrFunc<ito::float32>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise or function of type float64
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitOrFunc<ito::float64>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise or function of type complex64
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitOrFunc<ito::complex64>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise or function of type complex128
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitOrFunc<ito::complex128>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tBitOrFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);
MAKEFUNCLIST(BitOrFunc)

//! high-level operator, which executes the element-wise operation "bitwise or" between this data object and a given data object
/*!
    \param &rhs is the matrix which is used for the operator
    \return reference to this data object, where the result of the operation is stored
    \throws cv::Exception if data type is not supported or both data objects differs either in their size or data type
    \sa BitOrFunc
*/
DataObject & DataObject::operator |= (const DataObject & rhs)
{
   if (this == &rhs)
   {
      return *this;
   }

   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
   }

   fListBitOrFunc[m_type](this, &rhs, this);

   return (*this);
}

//! high-level operator, which executes the element-wise operation "bitwise or" between this data object and a given data object
/*!
    the result is returned as a newly allocated data object.
    \param &rhs is the matrix which is used for the operator
    \return new data object, where the result of the operation is stored
    \throws cv::Exception if data type is not supported or both data objects differs either in their size or data type
    \sa operator |=, BitOrFunc
*/
DataObject DataObject::operator | (const DataObject & rhs)
{
   if (this == &rhs)
   {
      return *this;
   }

   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
   }

   DataObject result;
   result.m_continuous |= rhs.m_continuous;
   this->copyTo(result, 1);

   result |= rhs;
   return result;
}



//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which element-wisely executes a bitwise 'xor' comparison between values of two dataObjects.
/*!
    \param *dObj1 is the first data object
    \param *dObj2 is the second data object
    \param *dObjRes is the data object, where the result is stored
    \throws cv::Exception for unsupported data types (template specialization)
    \return retOk
*/
template<typename _Tp> RetVal BitXorFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
   size_t numMats = dObj1->calcNumMats();
   size_t lhsMatNum = 0;
   size_t rhsMatNum = 0;
   size_t resMatNum = 0;

   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = dObj1->seekMat(nmat, numMats);
      rhsMatNum = dObj2->seekMat(nmat, numMats);
      resMatNum = dObjRes->seekMat(nmat, numMats);
	  (*((cv::Mat_<_Tp> *)(dObjRes->get_mdata())[resMatNum])) = (*((cv::Mat_<_Tp> *)(dObj1->get_mdata())[lhsMatNum])) ^ (*((cv::Mat_<_Tp> *)(dObj2->get_mdata())[rhsMatNum]));   
   }
   return 0;
}

//! template specialisation for bitwise xor function of type float32
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitXorFunc<ito::float32>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise xor function of type float64
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitXorFunc<ito::float64>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise xor function of type complex64
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitXorFunc<ito::complex64>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for bitwise xor function of type complex128
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitXorFunc<ito::complex128>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tBitXorFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);
MAKEFUNCLIST(BitXorFunc)

//! high-level operator, which executes the element-wise operation "bitwise xor" between this data object and a given data object
/*!
    \param &rhs is the matrix which is used for the operator
    \return reference to this data object, where the result of the operation is stored
    \throws cv::Exception if data type is not supported or both data objects differs either in their size or data type
    \sa BitXorFunc
*/
DataObject & DataObject::operator ^= (const DataObject & rhs)
{
   if (this == &rhs)
   {
      return *this;
   }

   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
   }

   fListBitXorFunc[m_type](this, &rhs, this);

   return (*this);
}

//! high-level operator, which executes the element-wise operation "bitwise or" between this data object and a given data object
/*!
    the result is returned as a newly allocated data object.
    \param &rhs is the matrix which is used for the operator
    \return new data object, where the result of the operation is stored
    \throws cv::Exception if data type is not supported or both data objects differs either in their size or data type
    \sa operator ^=, BitXorFunc
*/
DataObject DataObject::operator ^ (const DataObject & rhs)
{
   if ((m_size != rhs.m_size) || (m_type != rhs.m_type))
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
   }

   DataObject result;
   result.m_continuous |= rhs.m_continuous;
   this->copyTo(result, 1);

   result ^= rhs;
   return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! addressing method for two-dimensional data object with two given range-values. returns shallow copy of addressed regions.
/*!
    \param rowRange is the desired rowRange which should be in the new ROI (considers any existing ROI, too)
    \param colRange is the desired colRange which should be in the new ROI (considers any existing ROI, too)
    \return new data object which is a shallow copy of this data object and whose ROI is set to the given row- and col-ranges
    \throws cv::Exception if number of dimensions is unequal to two.
*/
DataObject DataObject::at(const ito::Range rowRange, const ito::Range colRange)
{
   if (m_dims != 2)
   {
         cv::error(cv::Exception(CV_StsAssert,"DataObject::at with rowRange and colRange argument only defined for dims==2","", __FILE__, __LINE__));
   }

   Range ranges[2]; 
         ranges[1].start = colRange.start;
         ranges[1].end = colRange.end;
         ranges[0].start = rowRange.start;
         ranges[0].end = rowRange.end;
   return (*this).at(ranges);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for saving a shallow copy of a source cv::Mat_ to a destination cv::Mat_ with respect to given row- and col-ranges
/*!
    \param *SrcMat is the source matrix which is firstly cast to cv::Mat_<_Tp>*
    \param rowRange is the desired row-range
    \param colRange is the desired col-range
    \param **dstMat is the pointer to a destination matrix which is also cast to cv::Mat_<_Tp>*
    \return retOk
*/
template<typename _Tp> RetVal GetRangeFunc(DataObject *dObj, const int dtop, const int dbottom, const int dleft, const int dright)
{
   if (dObj->getDims() > 1) //new version: adjusts ROI for every plane
   {
      size_t numMats = dObj->mdata_size();
      for (size_t nmat = 0; nmat < numMats; nmat++)
      {
         ((cv::Mat_<_Tp> *)((dObj->get_mdata())[nmat]))->adjustROI(dtop, dbottom, dleft, dright);
      }
   }

   return 0;
}

//typedef RetVal (*tGetRangeFunc)(const int * SrcMat, const cv::Range rowRange, const cv::Range colRange, int **dstMat);
typedef RetVal (*tGetRangeFunc)(DataObject *dObj, const int dtop, const int dbottom, const int dleft, const int dright);
MAKEFUNCLIST(GetRangeFunc)

//! addressing method for n-dimensional data object with n given range-values. returns shallow copy of addressed regions
/*!
    If any of the given ranges exceed the boundaries of its corresponding dimension, the range will be set to the boundaries.
    ranges will be given in "virtual" order, hence, the transpose-flag is considered by this method.

    \param *ranges is vector of desired ranges for each dimension
    \return new data object with shallow copy of this data object and adjusted ROI with respect to the given ranges
    \sa GetRangeFunc
*/
DataObject DataObject::at(ito::Range *ranges)
{
    DataObject resMat = *this;

    int *lims = new int[m_dims*2];
    size_t size;
    int start, end;

    for(int n = 0; n < m_dims; n++)
    {
        start = ranges[n].start;
        end = ranges[n].end;
        if(start > end) std::swap(start,end);
        lims[ (n*2) ] = -ranges[n].start;
        if(ranges[n].start == INT_MIN) // range all
        {
            lims[ (n*2) ] = 0;
        }		
        size = m_size.m_p[n];
        if(ranges[n].end == INT_MAX) //range all
        {
            lims[ (n*2) + 1] = 0;
        }
        else
        {
            lims[ (n*2) + 1] = -((int)size - ranges[n].end);
        }
    }

    resMat.adjustROI(m_dims, lims);

    delete[] lims;

    return resMat;   
}

//----------------------------------------------------------------------------------------------------------------------------------

//! low-level, templated method for adjusting the ROI of a data object by the given incremental values
/*!
    \param *dObj is the data object, whose boundaries should be adjusted
    \param dtop - The shift of the top submatrix boundary upwards (positive value means upwards)
    \param dbottom - The shift of the bottom submatrix boundary downwards (positive value means downwards)
    \param dleft - The shift of the left submatrix boundary to the left (positive value means to the left)
    \param dright - The shift of the right submatrix boundary to the right (positive value means to the right)
    \remark for any n-dimensional data object, the ROI of every matrix-plane is adjusted, even if any specific matrix-plane is temporarily not inside of the ROI
    \return retOk
*/
template<typename _Tp> RetVal AdjustROIFunc(DataObject *dObj, int dtop, int dbottom, int dleft, int dright)
{
   if (dObj->getDims() > 1) //new version: adjusts ROI for every plane
   {
//      size_t numMats = dObj->get_mdata().size();
      size_t numMats = dObj->mdata_size();
      for (size_t nmat = 0; nmat < numMats; nmat++)
      {
         //(*((cv::Mat_<_Tp> *)((dObj->get_mdata())[nmat]))) = ((cv::Mat_<_Tp> *)((dObj->get_mdata())[nmat]))->adjustROI(dtop, dbottom, dleft, dright);
         ((cv::Mat_<_Tp> *)((dObj->get_mdata())[nmat]))->adjustROI(dtop, dbottom, dleft, dright);
      }
   }

   return 0;
}

typedef RetVal (*tAdjustROIFunc)(DataObject *dObj, int dtop, int dbottom, int dleft, int dright);
MAKEFUNCLIST(AdjustROIFunc)

//! adjust submatrix size and position within the two-dimensional data-object
/*!
    \param dtop  The shift of the top submatrix boundary upwards (positive value means upwards)
    \param dbottom  The shift of the bottom submatrix boundary downwards (positive value means downwards)
    \param dleft  The shift of the left submatrix boundary to the left (positive value means to the left)
    \param dright  The shift of the right submatrix boundary to the right (positive value means to the right)
    \remarks the parameters indicates the shift with respect to the virtual order of the matrix, hence, the transpose flag is considered in this method
    \return reference to this data object
    \throws cv::Exception if data object is not two-dimensional
    \sa adjustROI
*/
DataObject & DataObject::adjustROI(const int dtop, const int dbottom, const int dleft, const int dright)
{
   if (m_dims != 2)
   {
         cv::error(cv::Exception(CV_StsAssert,"DataObject::row only defined for dims==2","", __FILE__, __LINE__));
   }

   int lims[4] = {dtop, dbottom, dleft, dright};
   return adjustROI(2, lims);
}

//! adjust submatrix size and position within the n-dimensional data-object
/*!
    \params dims is the number of dimensions
    \param *lims is a integer array whose length is 2*dims.
        For every dimension, two adjacent values indicates the shift of the ROI. The first of both values indicates the shift of the ROI towards the first element in the matrix (positive direction).
        The second value indicates the shift of the ROI towards the last element in the matrix (positive direction).
    \return reference to this data object
    \remarks lims indicates the shift with respect to the virtual order of the matrix, hence, the transpose flag is considered in this method
    \sa adjustROI
*/
DataObject & DataObject::adjustROI(const unsigned char dims, const int *lims)
{
    // check values
    if(dims != m_dims)
    {
        cv::error(cv::Exception(CV_StsAssert,"adjustROI is called with the wrong number of dimensions.","", __FILE__, __LINE__));
    }


    int startIdx, endSize;
    int lim1, lim2;

    for(int n = 0; n < dims; n++)
    {
        //TODO: why lim1 & lim2 are set when they actually aren't used?
        lim1 = lims[n*2];
        lim2 = lims[n*2+1];       
        startIdx = (int)m_roi.m_p[n] - lims[n*2]; //new first index
        if( startIdx < 0 || startIdx > (int)m_osize[n] ) //((int)m_roi.m_p[n] - lims[n*2] + (int)m_size.m_p[n] + lims[n*2+1]) >= (int)m_osize[n])
        {
            cv::error(cv::Exception(CV_StsAssert,"adjustROI: resulting ROI will start outside of the original matrix.","", __FILE__, __LINE__));
        }

        endSize = lims[n*2] + (int)m_size.m_p[n] + lims[n * 2 + 1];
        if( endSize < 0 || startIdx + endSize > (int)m_osize[n] )
        {
            cv::error(cv::Exception(CV_StsAssert,"adjustROI: resulting ROI is bigger than the original matrix size.","", __FILE__, __LINE__));
        }

        m_roi.m_p[n] = startIdx;
        m_size.m_p[n] = endSize;
    }


   ////! adjust the object metadata
   //for (int n = 0; n < (dims - 2); n++)
   //{
   //   m_size.m_p[n] = (((int)m_roi[n] + (int)m_size[n] + lims[n * 2 + 1]) > (int)m_osize[n]) ? m_osize[n] : (m_roi[n] + m_size[n] + lims[n * 2 + 1]);
   //   m_roi.m_p[n] = ((int)m_roi[n] - lims[n * 2]) < 0 ? 0 : (size_t)(m_roi[n] - lims[n * 2]);
   //   m_size.m_p[n] -= m_roi[n];
   //}

   //if(dims > 1)
   //{
   //    if (isT())
   //    {
   //       m_size.m_p[dims - 2] = (((int)m_roi[dims - 2] + (int)m_size[dims - 2] + lims[(dims - 1) * 2 + 1]) > (int)m_osize[dims - 2]) ? m_osize[dims - 2] : (size_t)(m_roi[dims - 2] + m_size[dims - 2] + lims[(dims - 1) * 2 + 1]);
   //       m_roi.m_p[dims - 2] = ((int)m_roi[dims - 2] - lims[(dims - 1) * 2]) < 0 ? 0 : (size_t)(m_roi[dims - 2] - lims[(dims - 1) * 2]);
   //       m_size.m_p[dims - 2] -= m_roi[dims - 2];

   //       m_size.m_p[dims - 1] = (((int)m_roi[dims - 1] + (int)m_size[dims - 1] + lims[(dims - 2) * 2 + 1]) > (int)m_osize[dims - 1]) ? m_osize[dims - 1] : (size_t)(m_roi[dims - 1] + m_size[dims - 1] + lims[(dims - 2) * 2 + 1]);
   //       m_roi.m_p[dims - 1] = ((int)m_roi[dims - 1] - lims[(dims - 2) * 2]) < 0 ? 0 : m_roi[dims - 1] - lims[(dims - 2) * 2];
   //       m_size.m_p[dims - 1] -= m_roi[dims - 1];
   //    }
   //    else
   //    {
   //       m_size.m_p[dims - 2] = (((int)m_roi[dims - 2] + (int)m_size[dims - 2] + lims[(dims - 2) * 2 + 1]) > (int)m_osize[dims - 2]) ? m_osize[dims - 2] : (size_t)(m_roi[dims - 2] + m_size[dims - 2] + lims[(dims - 2) * 2 + 1]);
   //       m_roi.m_p[dims - 2] = ((int)m_roi[dims - 2] - lims[(dims - 2) * 2]) < 0 ? 0 : (size_t)(m_roi[dims - 2] - lims[(dims - 2) * 2]);
   //       m_size.m_p[dims - 2] -= m_roi[dims - 2];

   //       m_size.m_p[dims - 1] = (((int)m_roi.m_p[dims - 1] + (int)m_size[dims - 1] + lims[(dims - 1) * 2 + 1]) > (int)m_osize[dims - 1]) ? m_osize[dims - 1] : (size_t)(m_roi[dims - 1] + m_size[dims - 1] + lims[(dims - 1) * 2 + 1]);
   //       m_roi.m_p[dims - 1] = ((int)m_roi[dims - 1] - lims[(dims - 1) * 2]) < 0 ? 0 : (size_t)(m_roi[dims - 1] - lims[(dims - 1) * 2]);
   //       m_size.m_p[dims - 1] -= m_roi[dims - 1];
   //    }
   //}

   if(dims > 1)
   {
       {
            fListAdjustROIFunc[m_type](this, lims[(dims - 2) * 2], lims[(dims - 2) * 2 + 1], lims[(dims - 1) * 2], lims[(dims - 1) * 2 + 1]); //dtop, dbottom, dleft, dright
       }
   }

   return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method locates ROI of this data object within its original data block
/*!
    long description

    \param *wholeSizes is an allocated array of size m_dims, which is filled with the original matrix-sizes (considering the transpose-flag, hence, the output is in user-friendly form)
    \param *offsets is dimension-wise offset in order to get from the original first element of the matrix to the subpart within the region of interest, array must be pre-allocated, too.
    \return retOk
*/
RetVal DataObject::locateROI(int *wholeSizes, int *offsets)
{
   for (int nDim = 0; nDim < m_dims - 2; nDim++)
   {
      wholeSizes[nDim] = static_cast<int>(m_osize[nDim]);
      offsets[nDim] = static_cast<int>(m_roi[nDim]);
   }

   if(m_dims > 1)
   {
      wholeSizes[m_dims - 2] = static_cast<int>(m_osize[m_dims - 2]);
      offsets[m_dims - 2] = static_cast<int>(m_roi[m_dims - 2]);
      wholeSizes[m_dims - 1] = static_cast<int>(m_osize[m_dims - 1]);
      offsets[m_dims - 1] = static_cast<int>(m_roi[m_dims - 1]);       
   }
   else if(m_dims == 1)
   {
       wholeSizes[0] = static_cast<int>(m_osize[0]);
       offsets[0] = static_cast<int>(m_roi[0]);
   }

   return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method get ROI of this data object within its original data block
/*!
    \params dims is the number of dimensions
    \param *lims is a integer array whose length is 2*dims.
        For every dimension, two adjacent values indicates the shift of the ROI. The first of both values indicates the shift of the ROI towards the first element in the matrix (positive direction).
        The second value indicates the shift of the ROI towards the last element in the matrix (positive direction).
    \return retOk
*/
RetVal DataObject::locateROI(int *lims)
{
   for (int nDim = 0; nDim < m_dims - 2; nDim++)
   {
      lims[2*nDim] = static_cast<int>(m_roi[nDim]);
      lims[2*nDim+1] = static_cast<int>(m_size[nDim])-static_cast<int>(m_osize[nDim]);
   }

   if(m_dims > 1)
   {
       {
          lims[2*(m_dims-1)] = static_cast<int>(m_roi[(m_dims-1)]);
          lims[2*(m_dims-1)+1] = static_cast<int>(m_size[(m_dims-1)])-static_cast<int>(m_osize[(m_dims-1)]);
          lims[2*(m_dims-2)] = static_cast<int>(m_roi[(m_dims-2)]);
          lims[2*(m_dims-2)+1] = static_cast<int>(m_size[(m_dims-2)])-static_cast<int>(m_osize[(m_dims-2)]);
       }
   }
   else if(m_dims == 1)
   {
          lims[0] = static_cast<int>(m_roi[0]);
          lims[1] = static_cast<int>(m_size[0])-static_cast<int>(m_osize[0]);
   }

   return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for creating an eye-matrix
/*!
    \param size indicates the size of the square matrix
    \param **dstMat is a pointer to which the eye-matrix is assigned to. The eye matrix is of type cv::Mat_<_Tp>
    \return retOk
*/
template<typename _Tp> RetVal EyeFunc(const size_t size, int **dstMat)
{
   (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::eye(static_cast<int>(size), static_cast<int>(size));

   return 0;
}

typedef RetVal (*tEyeFunc)(const size_t size, int **dstMat);
MAKEFUNCLIST(EyeFunc)

//! sets the matrix of this data object to a two-dimensional eye-matrix of size 1, hence [1]
/*!
    \param type is the desired element data-type
    \return retOk
    \sa ones
*/
RetVal DataObject::eye(const int type)
{
   size_t sizes[2] = {1, 1};
   return ones(2, sizes, type);
}

//! sets the matrix of this data object to a two-dimensional eye-matrix of given size
/*!
    At first, a preexisting matrix is freed, before creating the eye-matrix

    \param size is the desired size of the squared eye-matrix
    \param type is the desired element data-type
    \return retOk
    \sa freeData, create, EyeFunc
*/
RetVal DataObject::eye(const size_t size, const int type)
{
   size_t sizes[2] = {size, size};

   freeData();
   create(2, sizes, type, 1);

   fListEyeFunc[m_type](size, &(m_data[0]));

   return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for calculating the conjugated value of each element within the ROI of this data object
/*!
    This method is only valid for complex data types.
    \todo avoid MatIterator
    \param *dObj
    \throws cv::Exception if data type is not complex. This is done by template specialization.
    \return retOk
    \sa std::conj
*/
template<typename _Tp> RetVal ConjFunc(DataObject *dObj)
{
   size_t numMats = dObj->calcNumMats();
   size_t MatNum = 0;

   cv::Mat_<_Tp> * tempMat = NULL;
   int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
   int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
        //TODO: check if non iterator version is working
       MatNum = dObj->seekMat(nmat, numMats);
       tempMat = ((cv::Mat_<_Tp> *)((dObj->get_mdata())[MatNum]));

       for (int y = 0; y < sizey; y++)
       {
           _Tp* dstPtr = (_Tp*)tempMat->ptr(y);
           for (int x = 0; x < sizex; x++)
           {
               dstPtr[x] = std::conj(dstPtr[x]);
           }
       }

//      MatNum = dObj->seekMat(nmat, numMats);
//      cv::MatIterator_<_Tp> it = ((cv::Mat_<_Tp> *)(dObj->get_mdata())[MatNum])->begin();
//      cv::MatIterator_<_Tp> it_end = ((cv::Mat_<_Tp> *)(dObj->get_mdata())[MatNum])->end();
//      for (; it != it_end; ++it)
//      {
//         *it = std::conj(*it);
//      }
   }

   return retOk;
}

//! template specialization for data object of type int8. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<int8>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type uint8. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<uint8>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type int16. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<int16>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type uint16. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<uint16>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type int32. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<int32>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialization for data object of type uint32. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<uint32>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialization for data object of type float32. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<ito::float32>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type float64. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<ito::float64>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type float64. throws cv::Exception, since the data type is not complex.
template<> RetVal ConjFunc<int64>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tConjFunc)(DataObject *dObj);
MAKEFUNCLIST(ConjFunc)

//! converts every element of the data object to its conjugate complex value
/*!
    \throws cv::Exception if data type is not complex.
    \return retOk
    \sa ConjFunc
*/
RetVal DataObject::conj()
{
   fListConjFunc[m_type](this);
   return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! converts every element of the data object to its adjungate value
/*!
    The adjungate is the transposed matrix, where each element is complex conjugated.
    \throws cv::Exception if data type is not complex.
    \return retOk
    \sa conj
*/
DataObject DataObject::adj() const
{
	DataObject newDataObj = trans();
	newDataObj.conj();
	return newDataObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! transposes this data object
/*!
    simply toggles the transpose flag
    \return reference to this data object
*/
DataObject DataObject::trans() const
{
	return DataObject(*this, true);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which changes the region of interest of the data object to the selected zero-based row index
/*!
    \param *dObj
    \param selRow indicates the zero-based row-index (considering any existing ROI)
    \return retOk
*/
template<typename _Tp> RetVal RowFunc(DataObject *dObj, const unsigned int selRow)
{
   (*((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))) = (((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))->row(selRow));

   return 0;
}

typedef RetVal (*tRowFunc)(DataObject *dObj, const unsigned int selRow);
MAKEFUNCLIST(RowFunc)

//! high-level method which makes a new header for the specified matrix row and returns it. The underlying data of the new matrix is shared with the original matrix.
/*!
    \param selRow is the specific zero-based row index
    \return new data object
    \throws cv::Exception if dimension is unequal to two.
    \sa RowFunc
*/
DataObject DataObject::row(const int selRow)
{
   if (m_dims != 2)
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject::row only defined for dims==2","", __FILE__, __LINE__));
   }

   DataObject resMat = *this;
   fListRowFunc[m_type](&resMat, selRow);
   resMat.m_size.m_p[0] = 1;
   resMat.m_roi.m_p[0] = selRow;

   return resMat;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which changes the region of interest of the data object to the selected zero-based col index
/*!
    \param *dObj
    \param unsigned int selCol indicates the zero-based col-index (considering any existing ROI)
    \return retOk
*/
template<typename _Tp> RetVal ColFunc(DataObject *dObj, const unsigned int selCol)
{
   (*((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))) = (((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))->col(selCol));
   return 0;
}

typedef RetVal (*tColFunc)(DataObject *dObj, const unsigned int selCol);
MAKEFUNCLIST(ColFunc)

//! high-level method which makes a new header for the specified matrix column and returns it. The underlying data of the new matrix is shared with the original matrix.
/*!
    \param selCol is the specific zero-based row index
    \return new data object
    \throws cv::Exception if dimension is unequal to two.
    \sa ColFunc
*/
DataObject DataObject::col(const int selCol)
{
   if (m_dims != 2)
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject::col only defined for dims==2","", __FILE__, __LINE__));
   }

   DataObject resMat = *this;
   fListColFunc[m_type](&resMat, selCol);
   resMat.m_size.m_p[1] = 1;
   resMat.m_roi.m_p[1] = selCol;

   return resMat;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*
DataObject DataObject::diag(void)
{
   if (m_dims > 2)
   {
      cv::error(cv::Exception(CV_StsAssert, "diag is only defined for 2D matrices", "", __FILE__, __LINE__));
   }

   unsigned int sizes = m_size[m_dims - 1] < m_size[m_dims - 2] ? m_size[m_dims - 1] : m_size[m_dims - 2];
   DataObject resMat(1, sizes);
   diagVals.create(1, &sizes);

   for (unsigned int nElem = 0; nElem < sizes; nElem++)
   {
      diagVals(nElem) = m_data[0](nElem, nElem);
   }

   return 0;
}
*/
//----------------------------------------------------------------------------------------------------------------------------------
//!
/*
    \todo think about the definition (operator * ...)
*/
template<typename _Tp> RetVal MulFunc(const DataObject *src1, const DataObject *src2, DataObject *res, const double /*scale*/)
{
   //the transpose flag of this matrix already is evaluated if src2 is not transposed
   size_t numMats = src1->calcNumMats();

   size_t lhsMatNum = 0;
   size_t rhsMatNum = 0;
   size_t resMatNum = 0;

   const _Tp* src1RowPtr;
   const _Tp* src2RowPtr;
   _Tp* resRowPtr;
//   _Tp zero = static_cast<_Tp>(0);
   cv::Mat_<_Tp>* srcMat1 = NULL;
   cv::Mat_<_Tp>* dstMat = NULL;
   cv::Mat_<_Tp>* srcMat2 = NULL;

   for (unsigned int nmat = 0; nmat < numMats; nmat++)
   {
        cv::Mat_<_Tp> tempMat;
        lhsMatNum = src1->seekMat(nmat, numMats);
        rhsMatNum = src2->seekMat(nmat, numMats);
        resMatNum = res->seekMat(nmat, numMats);
        srcMat1 = (cv::Mat_<_Tp> *)(src1->get_mdata()[lhsMatNum]);       
        srcMat2 = (cv::Mat_<_Tp> *)(src2->get_mdata()[rhsMatNum]);
        dstMat = (cv::Mat_<_Tp> *)(res->get_mdata()[resMatNum]);

        for(int i = 0; i < srcMat1->rows; i++)
        {
            src1RowPtr = (_Tp*)srcMat1->ptr(i);
            src2RowPtr = (_Tp*)srcMat2->ptr(i);
            resRowPtr = (_Tp*)dstMat->ptr(i);

            for(int j = 0; j < srcMat1->cols; j++)
            {
                resRowPtr[j] = src1RowPtr[j] * src2RowPtr[j];
            }
        }
   }
   return 0;
}

typedef RetVal (*tMulFunc)(const DataObject *src1, const DataObject *src2, DataObject *res, const double scale);
MAKEFUNCLIST(MulFunc)

//!
/*
    \todo think about the definition (operator * ...)
*/
DataObject DataObject::mul(const DataObject &mat2, const double scale)
{
    if ((m_size != mat2.m_size) || (m_type != mat2.m_type))
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
    }  

    unsigned char continuous = 0;
    DataObject result(m_dims,m_size,m_type,continuous);    
    //int64 start = cv::getCPUTickCount();
    fListMulFunc[m_type](this, &mat2, &result, scale);
    //start = cv::getCPUTickCount() -start;

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which does a element-wise division of elements in first source matrix by elements in second source matrix.
/*!
    The result is stored in a result matrix, optionally the division can be scaled by a scaling factor, which is set to one by default.
    For fixed point numbers or complex values, a division by zero will throw an error. For floating-point values the following (matlab-like)
    implementation is used:

    1.0/0.0 = Inf, 0.0/0.0 = Nan

    \param *src1 is the first source matrix
    \param *src2 is the second source matrix
    \param *res is the result matrix, which must have the same size than the source matrices
    \param double scale is the scaling factor (default: 1.0)
    \return retOk
*/
template<typename _Tp> RetVal DivFunc(const DataObject *src1, const DataObject *src2, DataObject *res, const double /*scale*/)
{
    //the transpose flag of this matrix already is evaluated if src2 is not transposed
   size_t numMats = src1->calcNumMats();

   size_t lhsMatNum = 0;
   size_t rhsMatNum = 0;
   size_t resMatNum = 0;

   const _Tp* src1RowPtr;
   const _Tp* src2RowPtr;
   _Tp* resRowPtr;
   _Tp nanNumber;
   _Tp infNumber;
   _Tp epsilon;
   _Tp zero = static_cast<_Tp>(0);
   cv::Mat_<_Tp>* srcMat1 = NULL;
   cv::Mat_<_Tp>* dstMat = NULL;
   cv::Mat_<_Tp>* srcMat2 = NULL;

   for (size_t nmat = 0; nmat < numMats; nmat++)
   {
        cv::Mat_<_Tp> tempMat;
        lhsMatNum = src1->seekMat(nmat, numMats);
        rhsMatNum = src2->seekMat(nmat, numMats);
        resMatNum = res->seekMat(nmat, numMats);
        srcMat1 = (cv::Mat_<_Tp> *)(src1->get_mdata()[lhsMatNum]);        
        srcMat2 = (cv::Mat_<_Tp> *)(src2->get_mdata()[rhsMatNum]);     
        dstMat = (cv::Mat_<_Tp> *)(res->get_mdata()[resMatNum]);

        if(std::numeric_limits<_Tp>::has_signaling_NaN)
        {
            nanNumber = std::numeric_limits<_Tp>::signaling_NaN();
            infNumber = std::numeric_limits<_Tp>::infinity();
            epsilon = std::numeric_limits<_Tp>::epsilon();

            for(int i = 0; i < srcMat1->rows; i++)
            {
                src1RowPtr = (_Tp*)srcMat1->ptr(i);
                src2RowPtr = (_Tp*)srcMat2->ptr(i);
                resRowPtr = (_Tp*)dstMat->ptr(i);

                for(int j = 0; j < srcMat1->cols; j++)
                {
                    resRowPtr[j] = (isZeroValue<_Tp>(src2RowPtr[j],epsilon)) ? ((isZeroValue<_Tp>(src1RowPtr[j],epsilon)) ? nanNumber : infNumber) : src1RowPtr[j] / src2RowPtr[j];
                }
            }
        }
        else
        {
            for(int i = 0; i < srcMat1->rows; i++)
            {
                src1RowPtr = (_Tp*)srcMat1->ptr(i);
                src2RowPtr = (_Tp*)srcMat2->ptr(i);
                resRowPtr = (_Tp*)dstMat->ptr(i);

                for(int j = 0; j < srcMat1->cols; j++)
                {
                    if(src2RowPtr[j] == zero) cv::error(cv::Exception(CV_StsAssert,"Division by zero not allowed for fixed point arithmetic and complex values","", __FILE__, __LINE__));
                    resRowPtr[j] = src1RowPtr[j] / src2RowPtr[j];
                }
            }
        }
   }

   return 0;
}

typedef RetVal (*tDivFunc)(const DataObject *src1, const DataObject *src2, DataObject *res, const double scale);
MAKEFUNCLIST(DivFunc)

//! high-level method which does a element-wise division of elements in this matrix by elements in second source matrix.
/*!
    The result is stored in a result matrix which is returned. Optionally the division can be scaled by a scaling factor, which is set to one by default.

    \param &mat2 is the second source matrix
    \param scale is the scaling factor (default: 1.0)
    \return result matrix
    \sa DivFunc
*/
DataObject DataObject::div(const DataObject &mat2, const double scale)
{
   if ((m_size != mat2.m_size) || (m_type != mat2.m_type))
   {
        cv::error(cv::Exception(CV_StsAssert,"DataObject - operands differ in size or type","", __FILE__, __LINE__));
   }  

   DataObject result;
   this->copyTo(result, 1); 
                     
   //int64 start = cv::getCPUTickCount();
   fListDivFunc[m_type](this, &mat2, &result, scale);
   //start = cv::getCPUTickCount() -start;

   return result;
}

DataObject DataObject::squeeze() const
{
    if(m_dims <= 0)
    {
        return DataObject();
    }
    else if(m_dims == 1)
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject to squeeze may not have dimension = 1",  "", __FILE__, __LINE__));
    }

    size_t numMats = calcNumMats();
    cv::Mat * planes = new cv::Mat[numMats];

    for(size_t i = 0 ; i < numMats ; i++)
    {
        planes[i] = *( (cv::Mat*)(m_data[this->seekMat(i)]) );
    }

    unsigned char newDimensions = 2;
    size_t *newSizes = new size_t[m_dims];
    int counter = 0;

    for(int i = 0; i < m_dims - 2 ; i++)
    {
        if( getSize(i) > 1 )
        {
            newDimensions ++;
            newSizes[counter] = getSize(i);
            counter++;
        }
    }

    //last two dimensions (do not squeeze)
    newSizes[counter] = getSize(m_dims - 2);
    newSizes[counter+1] = getSize(m_dims - 1);

    DataObject resObj = DataObject(newDimensions, newSizes, m_type, planes, static_cast<unsigned int>(numMats));

    if(!copyTagMapTo(resObj).containsError())   // Now deal with the tagspace
    {
        unsigned int counter = 0;
        for(int i = 0; i < m_dims - 2 ; i++)
        {
            if( getSize(i) > 1 )
            {
                bool test = false;
                resObj.setAxisDescription(counter, this->getAxisDescription(i, test));
                resObj.setAxisUnit(counter, this->getAxisUnit(i, test));
                resObj.setAxisOffset(counter,this->getAxisOffset(i));
                resObj.setAxisScale(counter, this->getAxisScale(i));
                counter++;
            }
        }
        for(int i = m_dims - 2; i < m_dims ; i++)
        {
            bool test = false;
            resObj.setAxisDescription(counter, this->getAxisDescription(i, test));
            resObj.setAxisUnit(counter, this->getAxisUnit(i, test));
            resObj.setAxisOffset(counter,this->getAxisOffset(i));
            resObj.setAxisScale(counter, this->getAxisScale(i));
            counter++;
        }
        resObj.setValueDescription(this->getValueDescription());
        resObj.setValueUnit(this->getValueUnit());
        //resObj.setValueOffset(counter,this->getAxisOffset(i));
        //resObj.setValueScales(counter, this->getAxisScales(i));
    }

    //tempMat->adjustROI(-dtop, -dbottom, -dleft, -dright);
    //resObj.adjustROI(newDimensions , );

    ////adjust ROI of resObj and both last dimensions (plane)

    delete[] newSizes;
    delete[] planes;
    newSizes = NULL;


    return resObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! verifies if the data type of elements in this data object is equal to the type of the argument.
/*
    \param [in] src is any variable whose type is checked
    \return retOk if both types are equal, retError if they are not equal or if the type of src is unknown
*/
template<typename _Tp> RetVal DataObject::checkType(const _Tp * src)
{
    try
    {
        ito::tDataType t = ito::getDataType(src);
        if(m_type == t)
        {
            return ito::retOk;
        }
        return RetVal(retError,0,"CheckType failed: types are not equal");
    }
    catch(cv::Exception ex)
    {
        return RetVal(retError,0,"Error during Type-Check. Type not templated");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> RetVal DataObject::copyFromData2D(const _Tp *src, const size_t sizeX, const size_t sizeY)
{
    ito::RetVal retval(ito::retOk);

    if ((calcNumMats() != 1) || (getSize(getDims() - 1) != sizeX) || (getSize(getDims() - 2) != sizeY))
    {
        retval = RetVal(ito::retError,0,"Error in copyFromData2D. Size of Buffer unequal size of DataObject");
        return retval;
    }
    retval = checkType(src);
    if (retval != ito::retOk)
       return retval;

    cv::Mat_<_Tp> *cvMat = ((cv::Mat_<_Tp> *)this->get_mdata()[this->seekMat(0)]);
    if (cvMat->isContinuous())
    {
        memcpy(cvMat->ptr(0), src, sizeX * sizeY * sizeof(_Tp));
    }
    else
    {
        for (size_t y = 0; y < sizeY; y++)
        {
            memcpy(cvMat->ptr(y), src + y * sizeX * sizeof(_Tp), sizeX * sizeof(_Tp));
        }
    }

    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> RetVal DataObject::copyFromData2D(const _Tp *src, const size_t sizeX, const size_t /* sizeY */, const int x0, const int y0, const size_t width, const size_t height)
{
    ito::RetVal retval;

    if ((calcNumMats() != 1) || (getSize(getDims() - 1) != width) || (getSize(getDims() - 2) != height))
    {
        retval = RetVal(ito::retError,0,"Error in copyFromData2D. Size of Buffer unequal size of DataObject");
        return retval;
    }
    retval = checkType(src);
    if (retval != ito::retOk)
        return retval;

    cv::Mat_<_Tp> *cvMat = ((cv::Mat_<_Tp> *)this->get_mdata()[this->seekMat(0)]);
    for (size_t y = 0; y < height; y++)
    {
        memcpy(cvMat->ptr(y), src + ((y0 + y) * sizeX + x0) * sizeof(_Tp) , width * sizeof(_Tp));
    }

    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level templated method to cast each element of source matrix to another type.
/*!
    The result is stored in the result matrix. Optionally a scaling and offsetting is possible.

    \param *dObj is the source data object
    \param *resObj is the result data object
    \param alpha is the scaling factor (default 1.0)
    \param beta is the ofset value (default 0.0)
    \return 0
    \throws cv::Exception if cast failed
    \sa cv::saturate_cast
*/
template<typename _Tp, typename _T2> RetVal CastFunc(const DataObject *dObj, DataObject *resObj, double alpha, double beta)
{
   size_t numMats = dObj->calcNumMats();
   size_t resTmat = 0;
   size_t srcTmat = 0;

   int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
   int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
   cv::Mat_<_Tp> * srcMat = NULL;
   cv::Mat_<_T2> * dstMat = NULL;

   if(alpha == 1.0 && beta == 0.0)
   {
       for (size_t nmat = 0; nmat < numMats; nmat++)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = dObj->seekMat(nmat, numMats);
          //TODO: check if non iterator version is working
          srcMat = ((cv::Mat_<_Tp> *)((dObj->get_mdata())[srcTmat]));
          dstMat = ((cv::Mat_<_T2> *)((resObj->get_mdata())[resTmat]));
          for (int y = 0; y < sizey; y++)
          {
              _T2* dstPtr = (_T2*)dstMat->ptr(y);
              _Tp* srcPtr = (_Tp*)srcMat->ptr(y);
              for (int x = 0; x < sizex; x++)
              {
                  dstPtr[x] = cv::saturate_cast<_T2>(srcPtr[x]);
              }
          }
       }
   }
   else
   {
       _Tp alpha2 = cv::saturate_cast<_Tp>(alpha);
       _Tp beta2 = cv::saturate_cast<_Tp>(beta);
       for (size_t nmat = 0; nmat < numMats; nmat++)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = dObj->seekMat(nmat, numMats);
          //TODO: check if non iterator version is working
          srcMat = ((cv::Mat_<_Tp> *)((dObj->get_mdata())[srcTmat]));
          dstMat = ((cv::Mat_<_T2> *)((resObj->get_mdata())[resTmat]));
          for (int y = 0; y < sizey; y++)
          {
              _T2* dstPtr = (_T2*)dstMat->ptr(y);
              _Tp* srcPtr = (_Tp*)srcMat->ptr(y);
              for (int x = 0; x < sizex; x++)
              {
                  dstPtr[x] = cv::saturate_cast<_T2>(srcPtr[x] * alpha2 + beta2);
              }
          }
       }
   }

   return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! cast operator for data object
/*!
    usage: res = static_cast<ito::float32>(sourceDataObject)

    \throws cv::Exception if cast failed, e.g. if cast not possible or types unknown
    \return cast data object
    \sa convertTo, CastFunc
*/
template<typename T2> DataObject::operator T2 ()
{
    ito::tDataType newType = getDataType2<T2*>();
    if(newType != m_type)
    {
        DataObject resObj;
        convertTo(resObj, newType);
        return resObj;
    }
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, double templated method to save the element-wise absolute value of each element in source matrix to result matrix
/*!
    This method takes the absolute value of a complex valued input matrix and stores it in the equivalent real typed result matrix

    \param *dObj is source matrix, must have complex type
    \param *resObj is the resulting data object, which has the real data type which corresponds to the complex type
    \return retOk
    \sa std::abs
*/
template<typename _CmplxTp, typename _Tp> RetVal AbsFunc(const DataObject *dObj, DataObject *resObj)
{
    size_t numMats = dObj->calcNumMats();
    size_t srcMatNum = 0;
    size_t dstMatNum = 0;

    cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    const _CmplxTp* srcPtr = NULL;
    _Tp* dstPtr = NULL;

    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = ((cv::Mat_<_CmplxTp> *)((dObj->get_mdata())[srcMatNum]));
        dstMat = ((cv::Mat_<_Tp> *)((resObj->get_mdata())[dstMatNum]));

        for (int y = 0; y < sizey; y++)
        {
            dstPtr = (_Tp*)dstMat->ptr(y);
            srcPtr = (_CmplxTp*)srcMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::abs(srcPtr[x]);
            }
        }

//        dstMatNum = resObj->seekMat(nmat, numMats);
//        cv::MatIterator_<_Tp> itRes = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->begin();
//        cv::MatIterator_<_Tp> itRes_end = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->end();
//
//        srcMatNum = dObj->seekMat(nmat, numMats);
//        cv::MatConstIterator_<_CmplxTp> itSrc = ((cv::Mat_<_CmplxTp> *)(dObj->get_mdata()[srcMatNum]))->begin();
//
//        for (; itRes != itRes_end; ++itRes, ++itSrc)
//        {
//            *itRes = std::abs(*itSrc);
//        }
    }
    return 0;
}

//! low-level, templated method to save the element-wise absolute value of each element in source matrix to result matrix
/*!
    This method takes the absolute value of a real typed input matrix and stores it in the equivalent real typed result matrix

    \param *dObj is source matrix, must have real value
    \param *resObj is the resulting data object
    \return retOk
    \sa std::abs
*/
template<typename _Tp> RetVal AbsFuncReal(const DataObject *dObj, DataObject *resObj)
{
    size_t numMats = dObj->calcNumMats();
    size_t srcMatNum = 0;
    size_t dstMatNum = 0;

    cv::Mat_<_Tp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = ((cv::Mat_<_Tp> *)((dObj->get_mdata())[srcMatNum]));
        dstMat = ((cv::Mat_<_Tp> *)((resObj->get_mdata())[dstMatNum]));

        for (int y = 0; y < sizey; y++)
        {
            _Tp* dstPtr = (_Tp*)dstMat->ptr(y);
            _Tp* srcPtr = (_Tp*)srcMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::abs(srcPtr[x]);
            }
        }

//        dstMatNum = resObj->seekMat(nmat, numMats);
//        cv::MatIterator_<_Tp> itRes = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->begin();
//        cv::MatIterator_<_Tp> itRes_end = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->end();
//
//        srcMatNum = dObj->seekMat(nmat, numMats);
//        cv::MatConstIterator_<_Tp> itSrc = ((cv::Mat_<_Tp> *)(dObj->get_mdata()[srcMatNum]))->begin();
//
//        for (; itRes != itRes_end; ++itRes, ++itSrc)
//        {
//            *itRes = std::abs(*itSrc);
//        }
    }
    return 0;
}

typedef RetVal (*tAbsFunc)(const DataObject *dObj, DataObject *resObj);
MAKEFUNCLIST_CMPLX_TO_REAL(AbsFunc)

//! high-level value which calculates the absolute value of each element of the input source data object and returns the resulting data object
/*!
    \param &dObj
    \return new data object with absolute values
    \throws cv::Exception if unknown data type
    \sa AbsFunc, AbsFuncReal
*/
DataObject abs(const DataObject &dObj)
{
    //resObj must be allocated with pysical data size of dObj since iterators in AbsFunc doesn't know anything about transpose-flag.
    //afterwards the transpose flag of resObj is set to this of dObj.
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX)
    {
        DataObject resObj(dObj.getDims(), dObj.getSize().m_p, ito::convertCmplxTypeToRealType((ito::tDataType)dObj.getType()));
        fListAbsFunc[dObj.getType() - TYPE_OFFSET_COMPLEX](&dObj, &resObj);
        return resObj;
    }
    else
    {
        DataObject resObj(dObj.getDims(), dObj.getSize().m_p, dObj.getType());

        switch(dObj.getType())
        {
        case ito::tInt8:
            AbsFuncReal<int8>(&dObj, &resObj);
            break;
        case ito::tInt16:
            AbsFuncReal<int16>(&dObj, &resObj);
            break;
        case ito::tInt32:
            AbsFuncReal<int32>(&dObj, &resObj);
            break;
        case ito::tFloat32:
            AbsFuncReal<ito::float32>(&dObj, &resObj);
            break;
        case ito::tFloat64:
            AbsFuncReal<ito::float64>(&dObj, &resObj);
            break;
        default:
            cv::error(cv::Exception(CV_StsAssert,"abs(), unkown type of source data object","", __FILE__, __LINE__));
        }
        return resObj;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------

//! low-level, double templated method to save the element-wise argument of each element in source matrix to result matrix
/*!
    This method takes the element-wise argument of a complex valued input matrix and stores it in the equivalent real typed result matrix

    \param *dObj is source matrix, must have complex type
    \param *resObj is the resulting data object, which has the real data type which corresponds to the complex type
    \return retOk
    \sa std::abs
*/
template<typename _CmplxTp, typename _Tp> RetVal ArgFunc(const DataObject *dObj, DataObject *resObj)
{
    size_t numMats = dObj->calcNumMats();
    size_t srcMatNum = 0;
    size_t dstMatNum = 0;

    cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = ((cv::Mat_<_CmplxTp> *)((dObj->get_mdata())[srcMatNum]));
        dstMat = ((cv::Mat_<_Tp> *)((resObj->get_mdata())[dstMatNum]));

        for (int y = 0; y < sizey; y++)
        {
            _Tp* dstPtr = (_Tp*)dstMat->ptr(y);
            _CmplxTp* srcPtr = (_CmplxTp*)srcMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::arg(srcPtr[x]);
            }
        }

//        dstMatNum = resObj->seekMat(nmat, numMats);
//        cv::MatIterator_<_Tp> itRes = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->begin();
//        cv::MatIterator_<_Tp> itRes_end = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->end();
//
//        srcMatNum = dObj->seekMat(nmat, numMats);
//        cv::MatConstIterator_<_CmplxTp> itSrc = ((cv::Mat_<_CmplxTp> *)(dObj->get_mdata()[srcMatNum]))->begin();
//
//        for (; itRes != itRes_end; ++itRes, ++itSrc)
//        {
//            *itRes = std::arg(*itSrc);
//        }
    }
    return 0;
}

typedef RetVal (*tArgFunc)(const DataObject *dObj, DataObject *resObj);
MAKEFUNCLIST_CMPLX_TO_REAL(ArgFunc)

//! high-level value which calculates the argument value of each element of the input source data object and returns the resulting data object
/*!
    \param &dObj
    \return new data object with argument values
    \throws cv::Exception if undefined data type
    \sa ArgFunc
*/
DataObject arg(const DataObject &dObj)
{
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX)
    {
        DataObject resObj(dObj.getDims(), dObj.getSize().m_p, ito::convertCmplxTypeToRealType((ito::tDataType)dObj.getType()));

        fListArgFunc[dObj.getType() - TYPE_OFFSET_COMPLEX](&dObj, &resObj);
        return resObj;
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert, "Arg() not defined for real input parameter type", "", __FILE__, __LINE__));
        return DataObject();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------

//! low-level, double templated method to save the element-wise real value of each element in source matrix to result matrix
/*!
    This method takes the real value of a complex valued input matrix and stores it in the equivalent real typed result matrix

    \param *dObj is source matrix, must have complex type
    \param *resObj is the resulting data object, which has the real data type which corresponds to the complex type
    \return retOk
    \sa std::abs
*/
template<typename _CmplxTp, typename _Tp> RetVal RealFunc(const DataObject *dObj, DataObject *resObj)
{
    size_t numMats = dObj->calcNumMats();
    size_t srcMatNum = 0;
    size_t dstMatNum = 0;

    cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = ((cv::Mat_<_CmplxTp> *)((dObj->get_mdata())[srcMatNum]));
        dstMat = ((cv::Mat_<_Tp> *)((resObj->get_mdata())[dstMatNum]));

        for (int y = 0; y < sizey; y++)
        {
            _CmplxTp* srcPtr = (_CmplxTp*)srcMat->ptr(y);
            _Tp* dstPtr = (_Tp*)dstMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::real(srcPtr[x]);
            }
        }

//        dstMatNum = resObj->seekMat(nmat, numMats);
//        cv::MatIterator_<_Tp> itRes = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->begin();
//        cv::MatIterator_<_Tp> itRes_end = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->end();
//
//        srcMatNum = dObj->seekMat(nmat, numMats);
//        cv::MatConstIterator_<_CmplxTp> itSrc = ((cv::Mat_<_CmplxTp> *)(dObj->get_mdata()[srcMatNum]))->begin();
//
//        for (; itRes != itRes_end; ++itRes, ++itSrc)
//        {
//            *itRes = std::real(*itSrc);
//        }
    }
    return 0;
}

typedef RetVal (*tRealFunc)(const DataObject *dObj, DataObject *resObj);
MAKEFUNCLIST_CMPLX_TO_REAL(RealFunc)

//! high-level value which calculates the real value of each element of the input source data object and returns the resulting data object
/*!
    \param &dObj
    \return new data object with real values
    \throws cv::Exception if undefined data type (e.g. real data types)
    \sa ArgFunc
*/
DataObject real(const DataObject &dObj)
{
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX)
    {
        DataObject resObj(dObj.getDims(), dObj.getSize().m_p, ito::convertCmplxTypeToRealType((ito::tDataType)dObj.getType()));

        fListRealFunc[dObj.getType() - TYPE_OFFSET_COMPLEX](&dObj, &resObj);
        return resObj;
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert, "Real() not defined for real input parameter type", "", __FILE__, __LINE__));
        return DataObject();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, double templated method to save the element-wise imaginary value of each element in source matrix to result matrix
/*!
    This method takes the imaginary value of a complex valued input matrix and stores it in the equivalent real typed result matrix

    \param *dObj is source matrix, must have complex type
    \param *resObj is the resulting data object, which has the real data type which corresponds to the complex type
    \return retOk
    \sa std::abs
*/
template<typename _CmplxTp, typename _Tp> RetVal ImagFunc(const DataObject *dObj, DataObject *resObj)
{
    size_t numMats = dObj->calcNumMats();
    size_t srcMatNum = 0;
    size_t dstMatNum = 0;

    cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (size_t nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = ((cv::Mat_<_CmplxTp> *)((dObj->get_mdata())[srcMatNum]));
        dstMat = ((cv::Mat_<_Tp> *)((resObj->get_mdata())[dstMatNum]));

        for (int y = 0; y < sizey; y++)
        {
            _CmplxTp* srcPtr = (_CmplxTp*)srcMat->ptr(y);
            _Tp* dstPtr = (_Tp*)dstMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::imag(srcPtr[x]);
            }
        }


//        dstMatNum = resObj->seekMat(nmat, numMats);
//        cv::MatIterator_<_Tp> itRes = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->begin();
//        cv::MatIterator_<_Tp> itRes_end = ((cv::Mat_<_Tp> *)(resObj->get_mdata()[dstMatNum]))->end();
//
//        srcMatNum = dObj->seekMat(nmat, numMats);
//        cv::MatConstIterator_<_CmplxTp> itSrc = ((cv::Mat_<_CmplxTp> *)(dObj->get_mdata()[srcMatNum]))->begin();
//
//        for (; itRes != itRes_end; ++itRes, ++itSrc)
//        {
//            *itRes = std::imag(*itSrc);
//        }
    }
    return 0;
}

typedef RetVal (*tImagFunc)(const DataObject *dObj, DataObject *resObj);
MAKEFUNCLIST_CMPLX_TO_REAL(ImagFunc)

//! high-level value which calculates the imaginary value of each element of the input source data object and returns the resulting data object
/*!
    \param &dObj
    \return new data object with imaginary values
    \throws cv::Exception if undefined data type (e.g. real data types)
    \sa ArgFunc
*/
DataObject imag(const DataObject &dObj)
{
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX)
    {//    unsigned int resTmat = 0;
        //    unsigned int srcTmat = 0;
        DataObject resObj(dObj.getDims(), dObj.getSize().m_p, ito::convertCmplxTypeToRealType((ito::tDataType)dObj.getType()));

        fListImagFunc[dObj.getType() - TYPE_OFFSET_COMPLEX](&dObj, &resObj);
        return resObj;
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert, "Imag() not defined for real input parameter type", "", __FILE__, __LINE__));
        return DataObject();
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which copies an incontinuously organized data object to a continuously organized resulting data object
/*!
    \param &dObj is the source data object
    \param &resDObj is the resulting data object
    \return retOk
*/
template<typename _Tp> RetVal MakeContinuousFunc(const DataObject &dObj, DataObject &resDObj)
{
    if(dObj.getContinuous())
    {
        resDObj = DataObject(dObj);
        return RetVal(retOk);
    }

    resDObj = DataObject(dObj.getDims() , dObj.m_osize, dObj.getType() , 1);
    resDObj.m_owndata = 1;

    int dims = dObj.getDims();
    for (int i = 0 ; i < dims ; i++)
    {
        resDObj.m_size.m_p[i] = dObj.m_size.m_p[i];
        resDObj.m_roi.m_p[i] = dObj.m_roi.m_p[i];
    }

    size_t roiOffset = 0;
    
    if(dims > 1)
    {
        roiOffset = sizeof(_Tp) * (dObj.m_roi.m_p[dims-1] + dObj.m_roi.m_p[dims-2] * dObj.m_osize.m_p[dims-1]);
    }

    size_t numMats = dObj.mdata_size();

    size_t matSize = sizeof(_Tp) * dObj.m_osize[dObj.getDims()-2] * dObj.m_osize[dObj.getDims()-1];

    if(numMats > 0)
    {

        uchar* newDataPtr = ((cv::Mat_<_Tp>*)(resDObj.m_data[0]))->data;
        cv::Mat_<_Tp> *tempMat;

        for (size_t n = 0; n < numMats; n++)
        {
            tempMat = (cv::Mat_<_Tp>*)(dObj.m_data[n]);
            memcpy((void*)newDataPtr , (void*)(tempMat->datastart), matSize);
            newDataPtr += matSize;
        }
    }

    if(roiOffset > 0)
    {
        int dtop = -(int)dObj.m_roi.m_p[dims-2];
        int dleft = -(int)dObj.m_roi.m_p[dims-1];
        int dbottom = -( (int)dObj.m_osize.m_p[dims-2] - (int)dObj.m_size.m_p[dims-2] + dtop );
        int dright = -( (int)dObj.m_osize.m_p[dims-1] - (int)dObj.m_size.m_p[dims-1] + dleft );
        for (size_t n = 0; n < numMats; n++)
        {
            ((cv::Mat*)resDObj.m_data[n])->adjustROI(dtop,dbottom,dleft,dright);
        }
    }

    return RetVal(retOk);
}

typedef RetVal (*tMakeContinuousFunc)(const DataObject &dObj, DataObject &resDObj);
MAKEFUNCLIST(MakeContinuousFunc)

//! high-level method which copies an incontinuously organized data object to a continuously organized resulting data object, which is returned
/*!
    \param &dObj is the source data object
    \return resulting data object
    \sa MakeContinuousFunc
*/
DataObject makeContinuous(const DataObject &dObj)
{
    if(!dObj.getContinuous())
    {
        DataObject retDataObject;
        fListMakeContinuousFunc[dObj.getType()](dObj, retDataObject);
        return retDataObject;
    }
    else
    {
        return DataObject(dObj);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the offset of the specified axis, return 1 if axis does not exist
int DataObject::setAxisOffset(const unsigned int axisNum, const double offset)
{
    if (!m_pDataObjectTags)
        return 1; // error
//            if (axisNum < 0 || axisNum >= m_pDataObjectTags->m_axisOffsets.size()) // gcc complains axisnum is ALWAYS > 0 so the first part of the if case is sensless
    if (axisNum >= m_pDataObjectTags->m_axisOffsets.size())
        return 1; //error
    uchar *ch = (uchar *)&offset;
    if (!((ch[7] & 0x7f) != 0x7f || (ch[6] & 0xf0) != 0xf0))
        return 1;           
    else
    {
        double t = m_pDataObjectTags->m_axisOffsets[axisNum] = offset  + m_roi[axisNum];
    }
    return 0; //ok
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the scale of the specified axis, return 1 if axis does not exist or scale is 0.0.
int DataObject::setAxisScale(const unsigned int axisNum, const double scale)
{
    if (!m_pDataObjectTags) return 1; //error
//            if (axisNum < 0 || axisNum >= m_pDataObjectTags->m_axisScales.size()) return 1; //error
    if (axisNum >= m_pDataObjectTags->m_axisScales.size()) return 1; //error
    if (fabs(scale) < std::numeric_limits<double>::epsilon()) return 1;
    uchar *ch = (uchar *)&scale;
    if (!((ch[7] & 0x7f) != 0x7f || (ch[6] & 0xf0) != 0xf0))
        return 1;           
    else
    {
        m_pDataObjectTags->m_axisScales[axisNum] = scale;
    }
    return 0; //ok
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the unit (string value) of the specified axis, return 1 if axis does not exist
int DataObject::setAxisUnit(const unsigned int axisNum, const std::string &unit)
{
    if (!m_pDataObjectTags) return 1; //error
//            if (axisNum < 0 || axisNum >= m_pDataObjectTags->m_axisUnit.size()) return 1; //error
    if (axisNum >= m_pDataObjectTags->m_axisUnit.size()) return 1; //error
			           
    else
    {
        m_pDataObjectTags->m_axisUnit[axisNum] = unit;
    }
    return 0; //ok
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the description (string value) of the specified axis, return 1 if axis does not exist
int DataObject::setAxisDescription(const unsigned int axisNum, const std::string &description)
{
    if (!m_pDataObjectTags) return 1; //error
//            if (axisNum < 0 || axisNum >= m_pDataObjectTags->m_axisDescription.size()) return 1; //error
    if (axisNum >= m_pDataObjectTags->m_axisDescription.size()) return 1; //error          
    else
    {
        m_pDataObjectTags->m_axisDescription[axisNum] = description;
    }
    return 0; //ok
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the string value of the specified tag, if the tag do not exist, it will be added automatically, return 1 if tagspace does not exist
//inline int setTag(const std::string &key, const std::string &value)
int DataObject::setTag(const std::string &key, const DataObjectTagType &value)
{
    if(!m_pDataObjectTags) return 1; //error
    m_pDataObjectTags->m_tags[key] = value;
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to check whether tag exist or not
bool DataObject::existTag(const std::string &key) const
{
    if(!m_pDataObjectTags) return false; //Tag does not existtemplate
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find(key);
    return (it != m_pDataObjectTags->m_tags.end());
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function deletes specified tag. If tag do not exist, return value is 1 else returnvalue is 0
bool DataObject::deleteTag(const std::string &key)
{
    if(!m_pDataObjectTags) return false; //tag not deleted
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find(key);
    if(it == m_pDataObjectTags->m_tags.end()) return false;
    m_pDataObjectTags->m_tags.erase(it);
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DataObject::deleteAllTags()
{
    if(!m_pDataObjectTags) return false; //tag not deleted
    m_pDataObjectTags->m_tags.clear();
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function adds value to the protocol-tag. If this object is an ROI, the ROI-coordinates are added. If string do not end with an \n, \n is added.
int DataObject::addToProtocol(const std::string &value)
{
    if(!m_pDataObjectTags) return 1; //error
    /* Check if object is only an ROI */
    bool isROI = false;
    std::string newcontent(""); // Start with an empty sting
    for(int dim = 0; dim < m_dims; dim++)   // Check if this is an ROI
    {
        if(m_size[dim] != m_osize[dim])
        {
            isROI = true;
        }
    }
    if(isROI)   // If this is an ROI get the position for all dimensions
    {
        int * sizeTotal = (int*)calloc(m_dims, sizeof(int));
        int * posROI = (int*)calloc(m_dims, sizeof(int));
        size_t sizeDim = 0;
        locateROI(sizeTotal, posROI);
        newcontent.append("ROI[");
        for(int dim = 0; dim < m_dims; dim++)
        {
            sizeDim = getSize(dim);
            if((int)sizeDim != sizeTotal[dim])
            {
                char buf[50] ={0};
                _snprintf(buf, 49, " %i : %i", posROI[dim], static_cast<int>(sizeDim) - 1 + posROI[dim]);
                newcontent.append(buf);
            }
            else
            {
                newcontent.append(" : ");
            }
            if(dim != m_dims-1)
            {
                newcontent.append(",");
            }
            else
            {
                newcontent.append("]");
            }
        }
        free(sizeTotal);
        free(posROI);
    }
    newcontent.append(value);   // Append the value to the content
    if(newcontent[newcontent.length()-1] != '\n')   // add a \n is not aready there
    {
        newcontent.append("\n");
    }
    // Check if there is already a protocol tag
    //std::map<std::string,std::string>::iterator it = m_pDataObjectTags->m_tags.find("protocol");
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find("protocol");
    if(it == m_pDataObjectTags->m_tags.end())  // is not, okay create a new
    {
        m_pDataObjectTags->m_tags["protocol"] = newcontent;
    }
    else
    {   // is there, so just append to existing tag
        //(*it).second.append(newcontent);
        std::string tempVal = (*it).second.getVal_ToString();
        tempVal.append(newcontent);
        (*it).second = tempVal;
    }
    return 0;
}


size_t DataObject::elemSize() const
{
    switch(m_type)
    {
    case tInt8:
    case tUInt8:
        return 1;
    case tInt16:
    case tUInt16:
        return 2;
    case tInt32:
    case tUInt32:
        return 4;
    case tFloat32:
        return 4;
    case tFloat64:
    case tComplex64:
        return 8;
    case tComplex128:
        return 16;
    default: return 0;
    }
}

DObjIterator DataObject::begin()
{
    return DObjIterator(this, 0);
}

DObjIterator DataObject::end()
{
    return DObjIterator(this, getTotal());
}

DObjConstIterator DataObject::constBegin() const
{
    return DObjConstIterator(this, 0);
}

DObjConstIterator DataObject::constEnd() const
{
    return DObjConstIterator(this, getTotal());
}









////! missing documentation
///*!
//    \todo check that function
//*/
//template <typename _Tp> RetVal MinMaxLocFunc(const DataObject &dObj, double *minVal, double *maxVal, size_t *minPos, size_t *maxPos)
//{
//    if(std::numeric_limits<_Tp>::is_exact)
//    {
//        *maxVal = std::numeric_limits<_Tp>::min();
//    }
//    else
//    {
//        *maxVal = -1 * std::numeric_limits<_Tp>::max();
//    }
//
//    *minVal = std::numeric_limits<_Tp>::max();
//
//    size_t minMatNum = 0;
//    size_t maxMatNum = 0;
//    size_t numMats = 0;
//    size_t tMat = 0;
//    numMats = dObj.calcNumMats();
//    cv::Point minPt;
//    cv::Point maxPt;
//    cv::Point absMinPt;
//    cv::Point absMaxPt;
//
//    double absMinVal = std::numeric_limits<_Tp>::max();
//
//    double absMaxVal = std::numeric_limits<_Tp>::min();
//
//    if(!std::numeric_limits<_Tp>::is_exact)
//    {
//        absMaxVal = -1 * std::numeric_limits<_Tp>::max();
//    }
//
//    double locMinVal = std::numeric_limits<_Tp>::max();
//
//    double locMaxVal = std::numeric_limits<_Tp>::min();
//    if(!std::numeric_limits<_Tp>::is_exact)
//    {
//        locMaxVal = -1 * std::numeric_limits<_Tp>::max();
//    }
//
//    for (size_t nMat = 0; nMat < numMats; nMat++)
//    {
//        tMat = dObj.seekMat(nMat, numMats);
//        if (dObj.isT())
//        {
//            cv::minMaxLoc(*(cv::Mat_<_Tp> *)((dObj.get_mdata())[tMat]), &locMinVal, &locMaxVal, &minPt, &maxPt);
//        }
//        else
//        {
//            cv::minMaxLoc(((cv::Mat_<_Tp> *)((dObj.get_mdata())[tMat]))->t(), &locMinVal, &locMaxVal, &minPt, &maxPt);
//        }
//        if (locMinVal < absMinVal)
//        {
//            absMinVal = locMinVal;
//            absMinPt = minPt;
//            minMatNum = tMat;
//        }
//        if (locMaxVal > absMaxVal)
//        {
//            absMaxVal = locMaxVal;
//            absMaxPt = maxPt;
//            maxMatNum = tMat;
//        }
//    }
//
//    *minVal = absMinVal;
//    *maxVal = absMaxVal;
//    if (minPos)
//    {
//       dObj.matNumToIdx(minMatNum, minPos);
//       minPos[dObj.getDims() - 1] = absMinPt.x;
//       minPos[dObj.getDims() - 2] = absMinPt.y;
//    }
//
//    if (maxPos)
//    {
//       dObj.matNumToIdx(maxMatNum, minPos);
//       maxPos[dObj.getDims() - 1] = absMaxPt.x;
//       maxPos[dObj.getDims() - 2] = absMaxPt.y;
//    }
//
//    return 0;
//}
//
////! template specialization for data object of type complex64. throws cv::Exception, since the data type is not supported.
//template <> RetVal MinMaxLocFunc<ito::complex64>(const DataObject & /*dObj*/, double * /*minVal*/, double * /*maxVal*/, size_t * /*minPos*/, size_t * /*maxPos*/)
//{
//   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
//   return 0;
//}
//
////! template specialization for data object of type complex128. throws cv::Exception, since the data type is not supported.
//template <> RetVal MinMaxLocFunc<ito::complex128>(const DataObject & /*dObj*/, double * /*minVal*/, double * /*maxVal*/, size_t * /*minPos*/, size_t * /*maxPos*/)
//{
//   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
//   return 0;
//}
//
//
//typedef RetVal (*tMinMaxLocFunc)(const DataObject &dObj, double *minVal, double *maxVal, size_t *minPos, size_t *maxPos);
//MAKEFUNCLIST(MinMaxLocFunc)
//
////! missing documentation
///*!
//    \todo check that function
//*/
//RetVal minMaxLoc(const DataObject &dObj, double *minVal, double *maxVal, size_t *minPos, size_t *maxPos)
//{
//    return fListMinMaxLocFunc[dObj.getType()](dObj, minVal, maxVal, minPos, maxPos);
//}

//----------------------------------------------------------------------------------------------------------------------------------
template RetVal DataObject::copyFromData2D<int8>(const int8*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<uint8>(const uint8*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<int16>(const int16*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<uint16>(const uint16*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<int32>(const int32*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<uint32>(const uint32*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::float32>(const float32*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::float64>(const float64*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::complex64>(const complex64*, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::complex128>(const complex128*, const size_t, const size_t);

//----------------------------------------------------------------------------------------------------------------------------------
template RetVal DataObject::copyFromData2D<int8>(const int8*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<uint8>(const uint8*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<int16>(const int16*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<uint16>(const uint16*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<int32>(const int32*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<uint32>(const uint32*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::float32>(const float32*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::float64>(const float64*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::complex64>(const complex64*, const size_t, const size_t, const int, const int, const size_t, const size_t);
template RetVal DataObject::copyFromData2D<ito::complex128>(const complex128*, const size_t, const size_t, const int, const int, const size_t, const size_t);

//----------------------------------------------------------------------------------------------------------------------------------
}// namespace ito
