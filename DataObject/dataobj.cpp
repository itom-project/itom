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

//#include <crtdbg.h>
#include "dataobj.h"
#include "../common/numeric.h"
#include <cmath>

#include <vector>
#include <map>

namespace ito {


const int DataObject::m_sizeofs = sizeof(int) < sizeof(int *) ? 1 : sizeof(int) / sizeof(int *);


//--------------------------------------------------------------------------
//       BEGIN DataObjectTagsPrivate
//--------------------------------------------------------------------------
class DataObjectTagsPrivate
{
public:
    std::map<std::string, DataObjectTagType> m_tags;   /*!< map for tags with keyword (std::string) and value (either std::string or double) */
    std::vector<double> m_axisOffsets;          /*!< vector with offset-values for each axis (offset in dataObject-Pixel). Describes the distance from pixel [0,0,..0] to coordiante system origin. Unit-Coordinate = ( px-Coordinate - Offset)* Scale*/
    std::vector<double> m_axisScales;           /*!< vector with scale-values for each axis (unit / px). Unit-Coordinate = ( px-Coordinate - Offset)* Scale. Scale cannot be 0.0*/
    std::vector<std::string> m_axisDescription; /*!< vector with axis-describtions */
    std::vector<std::string> m_axisUnit;        /*!< vector with axis-units-description (e.g. 'mm') */
    double m_valueOffset;                       /*!< offset of the values within the dataObject. Currently as read only with value 0.0 */
    double m_valueScale;                        /*!< scale of the values within the dataObject. Currently as read only with value 1.0 */
    std::string m_valueDescription;             /*!< descriptions for the values (e.g. 'Intensity' or 'Heigth') */
    std::string m_valueUnit;                    /*!< unit description for the values (e.g. 'mm') */

    double m_rotMatrix[9];                      /*!< array containing the rotation matrix for the yx-plane */;

    DataObjectTagsPrivate()
        : m_valueOffset(0.0), m_valueScale(1.0)
    {
        memset(m_rotMatrix, 0, sizeof(double)*9);
        m_rotMatrix[0] = 1; // r11
        m_rotMatrix[4] = 1; // r22
        m_rotMatrix[8] = 1; // r33
    }

    //!< Constructor
    DataObjectTagsPrivate(unsigned int totalAxisNum) : m_valueOffset(0.0), m_valueScale(1.0)
    {
        m_axisOffsets.resize(totalAxisNum, 0.0);
        m_axisScales.resize(totalAxisNum, 1.0);
        m_axisDescription.resize(totalAxisNum, "");
        m_axisUnit.resize(totalAxisNum, "");
        memset(m_rotMatrix, 0, sizeof(double)*9);
        m_rotMatrix[0] = 1; // r11
        m_rotMatrix[4] = 1; // r22
        m_rotMatrix[8] = 1; // r33
    }

    //!< Destructor
    ~DataObjectTagsPrivate()
    {
    }

    ////!< Copy constructor
    //DataObjectTagsPrivate(const DataObjectTags& copyConstr)
};

//--------------------------------------------------------------------------
//       END DataObjectTagsPrivate
//--------------------------------------------------------------------------
















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
    int dims = dObj->getDims();

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
const uchar* DObjConstIterator::operator *() const
{
    return ptr;
}

//-------------------------------------------------------------------------  
//! returns the i-th matrix element, relative to the current
const uchar* DObjConstIterator::operator [](int i) const
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
    {
        return *this;
    }

    int ofsb = ofs * elemSize;
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
        int matIndex;
        int dims = dObj->getDims();

        // crash running iterator on empty dataObject
        if (dims <= 0)
            return;

        if(ofs <= 0) //begin
        {
            //ptr = dObj->rowPtr(0,0);
            int matIndex = dObj->seekMat(0);
            ptr = (dObj->get_mdata()[matIndex])->data;

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
        else if((int)ofs >= dObj->getTotal()) //end, (ofs > 0)
        {
            plane = dObj->getNumPlanes() - 1;

            if(planeContinuous)
            {
                
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = (dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + elemSize * dObj->getSize(dims-1) * dObj->getSize(dims-2);
            }
            else
            {
                //sliceStart = dObj->rowPtr(plane, dObj->getSize(dims-2)-1 );
                matIndex = dObj->seekMat(plane);
                sliceStart = const_cast<uchar*>((dObj->get_mdata()[matIndex])->ptr( dObj->getSize(dims-2)-1 ));
                ptr = sliceStart + elemSize * dObj->getSize(dims-1);
            }
            
            sliceEnd = ptr;
        }
        else
        {
            //determine the plane, where it lies in
            int planeSize = dObj->getSize( dims-1 ) * dObj->getSize( dims-2 );
            plane = ofs / planeSize; //floor value
            ofs -= (plane * planeSize);

            if (planeContinuous)
            {
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = (dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + elemSize * ofs;
                sliceEnd = sliceStart + elemSize * dObj->getSize(dims-1) * dObj->getSize(dims-2);
            }
            else
            {
                int row = ofs / dObj->getSize(dims-2); //floor value

                //sliceStart = dObj->rowPtr(plane,row);
                matIndex = dObj->seekMat(plane);
                sliceStart = const_cast<uchar*>((dObj->get_mdata()[matIndex])->ptr( row ));

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
        int matIndex;
        int dims = dObj->getDims();
        int width = dObj->getSize(dims-1);
        int stride = dObj->getOriginalSize(dims-1);

        int curRowIdx = (ptr - dObj->rowPtr(plane,0)) / (stride * elemSize); //floor
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
            int matIndex = dObj->seekMat(0);
            ptr = (dObj->get_mdata()[matIndex])->data;
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
        else if ( (int)plane >= dObj->getNumPlanes() ) //move to end (plane cannot be negative again)
        {
            plane = dObj->getNumPlanes() - 1;

            if(planeContinuous)
            {
                //sliceStart = dObj->rowPtr(plane, 0 );
                matIndex = dObj->seekMat(plane);
                sliceStart = (dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + elemSize * planeSize;
            }
            else
            {
                //sliceStart = dObj->rowPtr(plane, dObj->getSize(dims-2)-1 );
                matIndex = dObj->seekMat(plane);
                sliceStart = const_cast<uchar*>((dObj->get_mdata()[matIndex])->ptr( dObj->getSize(dims-2)-1 ));
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
                sliceStart = (dObj->get_mdata()[matIndex])->data;
                ptr = sliceStart + curElemIdxInPlane * elemSize;
                sliceEnd = sliceStart + (planeSize * elemSize);
            }
            else
            {
                curRowIdx = (curElemIdxInPlane / width);
                //sliceStart = dObj->rowPtr( plane, curRowIdx );
                matIndex = dObj->seekMat(plane);
                sliceStart = const_cast<uchar*>((dObj->get_mdata()[matIndex])->ptr( curRowIdx ));
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
    DObjIterator it2(*this);
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


////////int func1(void) { return 0; }
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
   FuncName<ito::float32>,                                              \
   FuncName<ito::float64>,                                              \
   FuncName<ito::complex64>,                                            \
   FuncName<ito::complex128>,                                           \
   FuncName<ito::Rgba32>                                                \
};

//! creates function table for the function (FuncName) and both complex data types. The destination method must be templated with two template values.
#define MAKEFUNCLIST_CMPLX_TO_REAL(FuncName) static t##FuncName fList##FuncName[] =     \
{                                                                                       \
    FuncName<ito::complex64,float32>,                                                        \
    FuncName<ito::complex128,float64>                                                        \
};

#define TYPE_OFFSET_COMPLEX  8
#define TYPE_OFFSET_RGBA  10


#define CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(otherObject) \
    if (m_type != otherObject.m_type) \
    { \
        cv::error(cv::Exception(CV_StsUnmatchedFormats, "dataObjects differ in type", "", __FILE__, __LINE__)); \
    } \
    else if ((m_dims == otherObject.m_dims) && (m_size != otherObject.m_size)) \
    { \
        cv::error(cv::Exception(CV_StsUnmatchedSizes, "dataObjects differ in size", "", __FILE__, __LINE__)); \
    } \
    else if (getNumPlanes() != otherObject.getNumPlanes()) \
    { \
        /*dataObjects have different numbers of planes.*/ \
        cv::error(cv::Exception(CV_StsUnmatchedSizes, "dataObjects differ in size (non equal number of planes)", "", __FILE__, __LINE__)); \
    } \
    else if (m_dims > 0 && (get_mdata()[0]->size() != otherObject.get_mdata()[0]->size())) \
    { \
        /*both objects have at least dimension two (same number of planes, and this->m_dims > 0).*/ \
        /*but the size of both planes (last two dimensions) is not equal.*/ \
        cv::error(cv::Exception(CV_StsUnmatchedSizes, "dataObjects differ in size (non equal size of each plane)", "", __FILE__, __LINE__)); \
    } \
    else if (m_size.m_p[0] == 0 || m_size.m_p[1] == 0) \
    { \
        /*One of the matrix dimensions is zeros, so matrix operations are meaningless*/ \
        cv::error(cv::Exception(CV_StsOutOfRange, "one of the matrices dimension is zero, meaningless operation", "", __FILE__, __LINE__)); \
    }


//! constructor for empty data object
/*!
    no data will be allocated, the number of elements and dimensions is set to zero
*/
DataObject::DataObject(void) : m_continuous(1), m_owndata(1), m_type(0), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
{
}

//! constructor for one-dimensional data object. The data is newly allocated and arbitrarily filled.
/*!
    In fact, by this constructor a two-dimensional matrix with dimension 1 x size will be created.
    the owndata-flag is set to true, the continuously-flag, too (since only one matrix-plane will be created)

    \param size is the number of elements
    \param type is the data-type of each element (use type of enumeration tDataType)
    \sa create, tDataType
*/
DataObject::DataObject(const int size, const int type): m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
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
DataObject::DataObject(const int sizeY, const int sizeX, const int type): m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
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
DataObject::DataObject(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous /*= 0*/) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
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
DataObject::DataObject(const int sizeZ, const int sizeY, const int sizeX, const int type, const uchar* continuousDataPtr,  const int* steps /*= NULL*/) : m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
{
    int sizes[3] = {sizeZ, sizeY, sizeX};
    this->create(3, sizes, type, m_continuous, continuousDataPtr, steps);
}

//! constructor for data object with given dimension. The data is newly allocated and arbitrarily filled.
/*!
    the owndata-flag is set to true

    \param sizes
    \param type is the data-type of each element (use type of enumeration tDataType)
    \param continuous indicates whether all matrix-planes should continuously lie in memory (1) or not (0) (default: 0)
    \sa create, tDataType
*/
DataObject::DataObject(const MSize &sizes, const int type, const unsigned char continuous /*= 0*/) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
{
    this->create(sizes.m_p[-1], sizes.operator const int *(), type, m_continuous);
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
DataObject::DataObject(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous /*= 0*/) : m_continuous(continuous), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
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
DataObject::DataObject(const unsigned char dimensions, const int *sizes, const int type, const uchar* continuousDataPtr, const int* steps /*= NULL*/) : m_continuous(1), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
{
    this->create(dimensions, sizes, type, m_continuous, continuousDataPtr, steps);
}

DataObject::DataObject(const unsigned char dimensions, const int *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes) : m_continuous(0), m_owndata(1), m_pRefCount(0), m_dims(0), m_data(NULL), m_pDataObjectTags(0)
{
    //usually it is dangerous to say that m_owndata is 1 in this case, since we cannot be sure if the given planes are the owner of their data.
    //however, in this case, owndata is unimportant since the created dataObject is always not continuous, therefore owndata will never
    //be analyzed and the destructor of the dataObject never tries to delete the continuous data block. The underlying cv::Mats however still know
    //whether they can or can't delete their data.
    this->create(dimensions, sizes, type, planes, nrOfPlanes);
}



//! destructor
/*!
    reference pointer of data is decremented and if <0, data will be deleted if owndata-flag is true. Additionally the allocated memory for
    header information will be deleted, too.

    \sa freeData
*/
DataObject::~DataObject(void)
{
    freeData();
}



//----------------------------------------------------------------------------------------------------------------------------------
//! returns the pointer to the underlying cv::Mat that represents the plane with given planeIndex of the entire data object.
/*!
    This command is equivalent to get_mdata()[seekMat(planeIndex)] but checks for out-of-range errors.

    \param planeIndex is the zero-based index of the requested plane within the current ROI of the data object
    \return pointer to the cv::Mat plane or NULL if planeIndex is out of range
    \sa seekMat
    \sa get_mdata, getContinuousCvPlaneMat
*/
cv::Mat* DataObject::getCvPlaneMat(const int planeIndex)
{
    int numMats = getNumPlanes();

    if (planeIndex >= 0 && planeIndex < numMats)
    {
        return (cv::Mat*)(m_data[seekMat(planeIndex, numMats)]);
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns the pointer to the underlying cv::Mat that represents the plane with given planeIndex of the entire data object.
/*!
    This command is equivalent to get_mdata()[seekMat(planeIndex)] but checks for out-of-range errors.

    \param planeIndex is the zero-based index of the requested plane within the current ROI of the data object
    \return pointer to the cv::Mat plane or NULL if planeIndex is out of range
    \sa seekMat
    \sa get_mdata, getContinuousCvPlaneMat
*/
const cv::Mat* DataObject::getCvPlaneMat(const int planeIndex) const
{
    int numMats = getNumPlanes();

    if (planeIndex >= 0 && planeIndex < numMats)
    {
        return (const cv::Mat*)(m_data[seekMat(planeIndex, numMats)]);
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns a shallow or deep copy of a cv::Mat plane with given index. If the current plane is not continuous (due to a roi), a cloned, continuous matrix is returned, else a shallow copy.
/*!
    \param planeIndex is the zero-based index of the requested plane within the current ROI of the data object
    \return shallow copy or clone of desired plane, depending if the plane is continuous (no roi set in plane dimensions) or not.
    \sa seekMat
    \sa get_mdata, getCvPlaneMat
*/
const cv::Mat DataObject::getContinuousCvPlaneMat(const int planeIndex) const
{
    int numMats = getNumPlanes();

    if (planeIndex >= 0 && planeIndex < numMats)
    {
        const cv::Mat* mat = (const cv::Mat*)(m_data[seekMat(planeIndex, numMats)]);

        if (mat->isContinuous())
        {
            return *mat;
        }
        else
        {
            return mat->clone();
        }
    }

    return cv::Mat();
}

//----------------------------------------------------------------------------------------------------------------------------------
int DataObject::mdata_realloc(const int size)
{
    if (m_data)
    {
        m_data = static_cast<uchar **>(realloc(m_data - m_sizeofs, (size + m_sizeofs) * sizeof(uchar *)));
    }
    else
    {
        int numBytes = (size + m_sizeofs) * sizeof(uchar *);
        m_data = static_cast<uchar **>(calloc(numBytes, 1));
        memset( m_data, 0, numBytes );
    }
    (*reinterpret_cast<int*>(m_data)) = size;
    m_data += m_sizeofs;
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
int DataObject::mdata_size(void) const
{
    if (!m_data)
    {
        return 0;
    }
    else
    {
        return (*reinterpret_cast<int *>(m_data - m_sizeofs));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
int DataObject::mdata_free()
{
    if (m_data)
    {
        uchar **ptr = m_data - m_sizeofs;
        free(ptr);
    }
    m_data = NULL;
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \todo documentation is missing
/*!

*/
RetVal DataObject::matNumToIdx(const int matNum, int *matIdx) const
{
   int tMatNum = matNum;
   int planeSize = 1;
   int m_dims_us = m_dims;

   for (int nDim = 1; nDim < m_dims_us - 2; nDim++)
   {
         planeSize *= m_osize[nDim];
   }

   for (int nDim = 0; nDim < m_dims_us - 2; nDim++)
   {
         matIdx[nDim] = tMatNum / planeSize - m_roi[nDim];
         tMatNum %= planeSize;
         planeSize /= m_osize[nDim + 1];
   }

   return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! \brief calculates the index of the matrix-plane in the m_data-vector for a given vector of indices, which address one element in the n-dimensional matrix
/*!
    The matrix indices are zero-based and consider the ROI of this data object.

    \param *matIdx is a vector containing indices which address one element in the n-dimensional matrix
    \param *matNum is a pointer, where the resulting matrix-plane-index is written.
    \return retOk
    \throws cv::Exception if the given indices are out of bounds
*/
ito::RetVal DataObject::matIdxToNum(const unsigned int *matIdx, int *matNum) const
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



//----------------------------------------------------------------------------------------------------------------------------------
//! calculates numbers of single opencv matrices which are part of the ROI which has previously been set.
/*!
    \return 0 if empty range or empty matrix, 1 if two dimensional, else product of sizes of all dimensions besides the last two ones.

    \sa getNumPlanes
*/
int DataObject::calcNumMats(void) const
{
    return getNumPlanes();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns the index vector-index of m_data which corresponds to the given zero-based two-dimensional matrix-index
/*!
    Since there might be a difference between the "real" matrix size in memory and the virtual size which is set by subslicing a matrix and hence
    setting any ROI, this method transforms a desired matrix-plane index to the real index in memory of the data-vector m_data

    \param matNum zero-based matrix-plane considering the virtual matrix size (ROI), 0<=matNum<getNumPlanes
    \return real vector-index for the desired matrix-plane
    \sa seekMat
    \sa getNumPlanes
*/
int DataObject::seekMat(const int matNum) const
{
   int numMats = getNumPlanes();
   return seekMat(matNum, numMats);
}

//----------------------------------------------------------------------------------------------------------------------------------
int DataObject::getStep(int index) const
{
    if (index < 0 || index >= m_dims)
    {
        cv::error(cv::Exception(CV_StsAssert, "Index out of bounds", "", __FILE__, __LINE__));
    }

    int step = 1;
    for (int i = index + 1; i < m_dims; ++i)
    {
        step *= m_osize[i];
    }
    return step;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns the index vector-index of m_data which corresponds to the given zero-based two-dimensional matrix-index
/*!
    Since there might be a difference between the "real" matrix size in memory and the virtual size which is set by subslicing a matrix and hence
    setting any ROI, this method transforms a desired matrix-plane index to the real index in memory of the data-vector m_data

    \param matNum zero-based matrix-plane-index, considering the virtual matrix size (ROI), 0<=matNum<getNumPlanes
    \param numMats total number of matrix-planes, lying within the ROI
    \return real vector-index for the desired matrix-plane or 0 if matNum >= numMats.
*/
int DataObject::seekMat(const int matNum, const int numMats) const
{
    if( matNum >= numMats || matNum < 0) return 0; //check boundaries

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
    //int *idx = new int[m_dims];
    //idx[m_dims-2] = 0;
    //idx[m_dims-1] = 0;
    //int val = matNum;
    //int t = 1;
    //int result = 0;

    //for(int i = m_dims - 3 ; i >=0 ; i--)
    //{
    //    idx[i] = ( val/t ) % m_size[i];
    //    val -= (idx[i] * t);
    //    t *= m_size[i];
    //}

    ////1. possibility:
    //matIdxToNum(idx, &result);

    ////2. possibility:
    //int planeSize = 1;

    //for (int n = m_dims - 3; n >= 0; n--)
    //{
    //        result += (idx[n] + m_roi[n]) * planeSize; //CAST_TODO
    //        planeSize *= m_osize[n];
    //}

    //delete[] idx;
    //return result;

    ////end 1. and 2. possibility

    //begin 3. possibility, directly combine 2. possiblity into one loop without allocating idx-vector
    int result = 0;

    //a little bit of loop unrolling to allow faster run for the most common dimensions (0,1,2,3)
    switch (m_dims)
    {
    case 0:
    case 1:
    case 2:
        break;
    case 3:
        {
        int idx = matNum % m_size[0];
        result += (idx + m_roi[0]);
        }
        break;
    default:
        {
        int val = matNum;
        int t = 1;
        int idx = 0;
        int planeSize = 1;

        for(int i = m_dims - 3 ; i >= 0 ; --i)
        {
            idx = (val / t) % m_size[i];
            result += (idx + m_roi[i]) * planeSize;
            val -= (idx * t);
            t *= m_size[i];
            planeSize *= m_osize[i];
        }
        }
        break;
    }

    return result;
}


//----------------------------------------------------------------------------------------------------------------------------------
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
void DataObject::createHeader(const unsigned char dimensions, const int *sizes, const int *steps, const int elemSize)
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

//----------------------------------------------------------------------------------------------------------------------------------
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
void DataObject::createHeaderWithROI(const unsigned char dimensions, const int *sizes, const int *osizes /*= NULL*/, const int *roi /*= NULL*/)
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
template<typename _Tp> RetVal CreateFunc(DataObject *dObj, const unsigned char dimensions, const int *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const int* steps)
{
   int numMats = 0;

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
       int sizes_inc[2] = {1,sizes[0]};
       int steps_inc[2] = {sizes[0] * sizeof(_Tp), sizeof(_Tp)};
       dObj->createHeader(2, sizes_inc, steps_inc, sizeof(_Tp));
   }

   if (!dObj->m_pRefCount && dimensions > 0)
   {
        dObj->m_pRefCount = new int(0);
        if(dimensions == 1)
        {
            dObj->m_pDataObjectTags = new DataObjectTagsPrivate(2);
        }
        else
        {
            dObj->m_pDataObjectTags = new DataObjectTagsPrivate(dimensions);
        }
   }

   cv::Mat_<_Tp> *dataMat = NULL;

   if(!continuous && continuousDataPtr)
   {
       cv::error(cv::Exception(CV_BadDataPtr, "data pointer must be empty if matrix is not continuous" ,"", __FILE__, __LINE__));
   }

   dObj->m_owndata = (continuousDataPtr == NULL);

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
            catch(cv::Exception /*&exc*/) //handle memory error
            {
                SecureFreeFunc<_Tp>(dObj);
                throw; //rethrow error
            }

            dObj->m_data[0] = reinterpret_cast<uchar *>(dataMat);
            break;

         case 2:
            dObj->mdata_realloc(1);

            try
            {
                if(continuousDataPtr)
                {
                    dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[0]), static_cast<int>(sizes[1]), (_Tp *)continuousDataPtr, steps ? steps[0] : cv::Mat::AUTO_STEP);
                }
                else
                {
                    dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[0]), static_cast<int>(sizes[1]));
                }
            }
            catch(cv::Exception /*&exc*/) //handle memory error
            {
                SecureFreeFunc<_Tp>(dObj);
                throw; //rethrow error
            }

            dObj->m_data[0] = reinterpret_cast<uchar *>(dataMat);
            break;

         default:
            numMats = dObj->getNumPlanes();
            dObj->mdata_realloc(numMats);

            try
            {
                if (!continuous)
                {
                   for (int n = 0; n < numMats; ++n)
                   {
                      dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[dimensions - 2]), static_cast<int>(sizes[dimensions - 1]));
                      dObj->m_data[n] = reinterpret_cast<uchar *>(dataMat);
                   }
                }
                else //!continuous
                {
                    size_t matSize = static_cast<size_t>(dObj->m_osize.m_p[dimensions - 2]) * static_cast<size_t>(dObj->m_osize.m_p[dimensions - 1]) * sizeof(_Tp);

                    if(continuousDataPtr)
                    {
                        dObj->mdata_realloc(numMats);
                        for (int n = 0; n < numMats; n++)
                        {
                            dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[dimensions - 2]), static_cast<int>(sizes[dimensions - 1]), (_Tp *)continuousDataPtr, steps ? steps[dimensions-2] : cv::Mat::AUTO_STEP);
                            dObj->m_data[n] = reinterpret_cast<uchar *>(dataMat);
                            continuousDataPtr += matSize;
                        }
                    }
                    else
                    {
                        if (numMats > 0 && (std::numeric_limits<size_t>::max() / numMats) < matSize)
                        {
                            cv::error(cv::Exception(CV_StsNoMem, ("Failed to allocate memory"), "", __FILE__, __LINE__));
                        }

                        size_t bytesToAllocate = static_cast<size_t>(numMats) * matSize;
                        char *dataPtr = (char*)malloc(bytesToAllocate); //this continuous data block must be freed if dataObject is destroyed. (done in FreeFunc and SecureFreeFunc)
                        if(dataPtr == NULL)
                        {
                            cv::error(cv::Exception(CV_StsNoMem, ("Failed to allocate memory"),"", __FILE__, __LINE__));
                        }

                        dObj->mdata_realloc(numMats);
                        for (int n = 0; n < numMats; n++)
                        {
                            dataMat = new cv::Mat_<_Tp>(static_cast<int>(sizes[dimensions - 2]), static_cast<int>(sizes[dimensions - 1]), (_Tp *)dataPtr);
                            dObj->m_data[n] = reinterpret_cast<uchar *>(dataMat);
                            dataPtr += matSize;
                        }
                    }

                }
            }
            catch(cv::Exception /*&exc*/)
            {
                SecureFreeFunc<_Tp>(dObj);
                throw; //rethrow error
            }
            break;
   }

   return 0;
}

typedef RetVal (*tCreateFunc)(DataObject *dObj, const unsigned char dimensions, const int *sizes, const unsigned char continuous, const uchar* continuousDataPtr, const int* steps);
MAKEFUNCLIST(CreateFunc)

//! high-level, non-templated method for data allocation
/*!
    \param dimensions is the total number of dimensions
    \param *sizes is a vector whose length is equal to dimensions. Each entry indicates the size of the specific dimension. Each matrix-plane is allocated with the size of the last two sizes
    \param type is the desired element data type (see tDataType)
    \param continuous indicates wether the entire array should be allocated in one connected data block in memory (true) or not (default, better for huge matrices)
    \param *continuousDataPtr is NULL if new data storage should be allocated (then m_owndata is true). Otherwise this pointer points to the starting point of a continuous data block, where this data-object should be refer to (then m_owndata is false). The data is not copied and the dataObject does not take ownership of the external data, hence it must be allocated during the lifetime of the dataObject and deallocated afterwards.
    \param *steps vector with size of dimensions, indicates how many bytes one has to move in order to get to the next element in the same dimension, the step-size for the last element must be set to element-size
    \throws open-cv error in case of error
    \sa CreateFunc
*/
void DataObject::create(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous, const uchar* continuousDataPtr, const int* steps)
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
template<typename _Tp> RetVal CreateFuncWithCVPlanes(DataObject *dObj, const unsigned char dimensions, const int *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes)
{
    cv::Size tempOrgSize;
    cv::Size tempSize;
    cv::Point tempPoint;

    if(dimensions == 0)
    {
        dObj->m_dims = 0;
    }
    else
    {
        planes[0].locateROI(tempOrgSize, tempPoint);
        tempSize = planes[0].size();
        int dtop = tempPoint.y;
        int dleft = tempPoint.x;
        int dbottom = tempOrgSize.height - tempSize.height - dtop;
        int dright = tempOrgSize.width - tempSize.width - dleft;

        int* sizes_inc = new int[dimensions];
        int* osizes_inc = new int[dimensions];
        int* roi_inc = new int[dimensions];

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
            osizes_inc[dimensions - 1] = sizes[dimensions - 1] + dleft + dright;
            roi_inc[dimensions - 1] = +dleft;

        }
        else //if one-dimensional create a two-dimensional data-object
        {
            sizes_inc[0] = 1; sizes_inc[1] = sizes[0];
            osizes_inc[0] = 1; osizes_inc[1] = sizes[0] + dleft + dright;
            roi_inc[0] = 0; roi_inc[1] = +dleft;
        }

        dObj->createHeaderWithROI(dimensions, sizes_inc, osizes_inc, roi_inc);

        delete[] sizes_inc;
        delete[] osizes_inc;
        delete[] roi_inc;

        if (!dObj->m_pRefCount)
        {
            dObj->m_pRefCount = new int(0);
            dObj->m_pDataObjectTags = new DataObjectTagsPrivate(dimensions);
        }

        dObj->mdata_realloc(nrOfPlanes);

        try
        {
            for(unsigned int i = 0 ; i < nrOfPlanes ; i++)
            {
                cv::Mat_<_Tp> * dataMat = new cv::Mat_<_Tp>(planes[i]); //memory error might occur
                dObj->m_data[i] = reinterpret_cast<uchar *>(dataMat);
            }
        }
        catch(cv::Exception /*&exc*/) //memory exception
        {
            SecureFreeFunc<_Tp>(dObj);
            throw; //rethrow error
        }
    }

    return retOk;
}

typedef RetVal (*tCreateFuncWithCVPlanes)(DataObject *dObj, const unsigned char dimensions, const int *sizes, const cv::Mat* planes, const unsigned int nrOfPlanes);
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
void DataObject::create(const unsigned char dimensions, const int *sizes, const int type, const cv::Mat* planes, const unsigned int nrOfPlanes)
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
    int sizex = dimensions ? sizes[dimensions - 1] : static_cast<int>(0);
    int sizey = dimensions ? sizes[dimensions - 2] : static_cast<int>(0);
    int numMats = dimensions ? 1 : 0;
    cv::Size planeSize;
    int requiredElemSize = 0;

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
    case ito::tRGBA32:
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
        for(int i = 0 ; i < numMats ; i++)
        {
            planeSize = planes[i].size();

            if((int)planeSize.height != sizey || (int)planeSize.width != sizex)
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
    else if (type == ito::tRGBA32)
    {
        for(int i = 0 ; i < numMats ; i++)
        {
            planeSize = planes[i].size();

            if((int)planeSize.height != sizey || (int)planeSize.width != sizex)
            {
                cv::error(cv::Exception(CV_BadImageSize, "image size of at least one cv::Mat-plane does not correspond to the given height and width.", "", __FILE__, __LINE__));
            }

            if(planes[i].channels() != 4)
            {
                cv::error(cv::Exception(CV_StsUnsupportedFormat, "at least one cv::Mat-plane has not four channels (RGBA type).", "", __FILE__, __LINE__));
            }

            if((planes[i].elemSize1()*4) != requiredElemSize)
            {
                cv::error(cv::Exception(CV_StsUnsupportedFormat, "the element size of at least one cv::Mat-plane does not correspond to the given dataObject-type.", "", __FILE__, __LINE__));
            }
        }
    }
    else
    {
        for(int i = 0 ; i < numMats ; i++)
        {
            planeSize = planes[i].size();

            if((int)planeSize.height != sizey || (int)planeSize.width != sizex)
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

            //this version of deleting the m_data vector is much faster than the version above (M. Gronle, 13.02.2012)
//            unsigned int size = dObj->m_data.size();
            int size = dObj->mdata_size();
            for ( int i = 0 ; i < size ; i++)
            {
                dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
                delete dataMat;
            }
            dObj->mdata_free();

            dObj->m_pRefCount = NULL;
            dObj->m_pDataObjectTags = NULL;

            return ito::retOk;
        }
        delete dObj->m_pRefCount;
        dObj->m_pRefCount = NULL;
        delete dObj->m_pDataObjectTags;
        dObj->m_pDataObjectTags = NULL;
    }

    // yes so really clean up
    if (!dObj->mdata_size())
    {
        return ito::retOk;
    }

    //check if the data has been allocated "en bloc" and delete the data first.
    if (dObj->m_continuous && old_m_dims > 2 && dObj->m_owndata)
    {
        dataMat = (cv::Mat_<_Tp> *)dObj->m_data[0];
        free((void*)dataMat->datastart); //data is wrong, since data-pointer does not point to start in case of ROI
    }

    //this version of deleting the m_data vector is much faster than the version above (M. Gronle, 13.02.2012)
//    unsigned int size = dObj->m_data.size();
    int size = dObj->mdata_size();
    for ( int i = 0 ; i < size ; i++)
    {
        dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
        delete dataMat;
    }
    dObj->mdata_free();

    return ito::retOk;
}

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
                int size = dObj->mdata_size();
                for ( int i = 0 ; i < size ; i++)
                {
                    dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
                    if(dataMat) delete dataMat;
                }
                dObj->mdata_free();
            }

            dObj->m_pRefCount = NULL;
            dObj->m_pDataObjectTags = NULL;

            return retOk;
        }
        else //this is the last instance to use this data
        {
            if(dObj->m_pRefCount) delete dObj->m_pRefCount;
            dObj->m_pRefCount = NULL;
            if(dObj->m_pDataObjectTags) delete dObj->m_pDataObjectTags;
            dObj->m_pDataObjectTags = NULL;
        }
    }

    //this section is only entered if we are the last to use the data or if no reference counter has been set (the latter usually should not happen)
    int numMats = dObj->mdata_size();

    if(numMats > 0)
    {
        //check if the data has been allocated "en bloc" and delete the data first.
        if (dObj->m_continuous && old_m_dims > 2 && dObj->m_owndata)
        {
            dataMat = (cv::Mat_<_Tp> *)dObj->m_data[0];
            if(dataMat && dataMat->datastart)
            {
                free((void*)dataMat->datastart);
            }
        }

        for ( int i = 0 ; i < numMats ; i++)
        {
            dataMat = (cv::Mat_<_Tp> *)dObj->m_data[i];
            if(dataMat) delete dataMat;
        }

        dObj->mdata_free();
    }

    return retOk;
}

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



/**
\brief Function returns the not rounded pixel index of a physical coordinate
\detail Function returns the not rounded pixel index of a physical coordinate (Unit-Coordinate = ( px-Coordinate - Offset)* Scale).
        If the pixel is outside of the image, the isInsideImage-flag is set to false else it is set to true.
        To avoid memory access-error, the returnvalue is clipped within the range of the image ([0...imagesize-1])
\param[in] dim  Axis-dimension for which the physical coordinate is calculated
\param[in] pix  Pixel-index as double
\param[out] isInsideImage   flag which is set to true if coordinate is within range of the image.
\return (double)( phys / AxisScale + AxisOffset) & [0..imagesize-1]
*/
double DataObject::getPhysToPix(const unsigned int dim, const double phys, bool &isInsideImage) const
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
        //tPx = (phys / scale) + offset
        tPx = (phys / m_pDataObjectTags->m_axisScales[dim]) + (m_pDataObjectTags->m_axisOffsets[dim] - m_roi[dim]);
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
\detail Function returns the not rounded pixel index of a physical coordinate (Unit-Coordinate = ( px-Coordinate - Offset)* Scale).
        To avoid memory access-error, the return value is clipped within the range of the image ([0...imagesize-1])
\param[in] dim  Axis-dimension for which the physical coordinate is calculated
\param[in] pix  Pixel-index as double
\return (double)( phys / AxisScale + AxisOffset) & [0..imagesize-1]
*/
double DataObject::getPhysToPix(const unsigned int dim, const double phys) const
{
    double tPx = 0.0;
    if(static_cast<int>(dim) >= m_dims)
    {
        return 0.0;
    }

    if(m_pDataObjectTags)
    {
        //tPx = (phys / scale) + offset
        tPx = (phys / m_pDataObjectTags->m_axisScales[dim]) + (m_pDataObjectTags->m_axisOffsets[dim] - m_roi[dim]);
    }
    else
    {
        tPx = phys;
    }

    if(tPx > getSize(dim) - 1)
    {
        tPx = static_cast<double>(getSize(dim)- 1);
    }
    else if( tPx < 0)
    {
        tPx = 0;
    }

    return tPx;
}

/**
\brief Function returns the not rounded pixel index of a physical coordinate. 

This method only considers the x- and y-coordinates (last two dimensions of the dataObject).

\param physY is the physical coordinate of the y axis
\param tPxY [byRef] contains the corresponding pixel coordinate of physY after the function has been called
\param isInsideImageY [byRef] is true if physY is inside of the dataObject area, else false
\param physX is the physical coordinate of the x axis
\param tPxX [byRef] contains the corresponding pixel coordinate of physX after the function has been called
\param isInsideImageX [byRef] is true if physX is inside of the dataObject area, else false
\return 0 (always)
*/
int DataObject::getPhysToPix2D(const double physY, double &tPxY, bool &isInsideImageY, const double physX, double &tPxX, bool &isInsideImageX) const
{
    isInsideImageX = isInsideImageY = true;

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
                tPxX = static_cast<double>(m_size[m_dims - 1] - 1);
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
double DataObject::getPixToPhys(const unsigned int dim, const double pix, bool &isInsideImage) const
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
        //tPhys = (pix - offset) * scale
        tPhys = (pix - (m_pDataObjectTags->m_axisOffsets[dim] - m_roi[dim])) * m_pDataObjectTags->m_axisScales[dim];
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
\brief Function returns the physical coordinate of a pixel
\detail Function returns the physical coordinate of a pixel index (Unit-Coordinate = ( px-Coordinate - Offset)* Scale).
\param[in] dim  Axis-dimension for which the physical coordinate is calculated
\param[in] pix  Pixel-index as double
\return (double)( pix - AxisOffset)* AxisScale)
*/
double DataObject::getPixToPhys(const unsigned int dim, const double pix) const
{
    double tPhys = 0.0;
    if(static_cast<int>(dim) >= m_dims)
    {
        return 0.0;
    }

    if(m_pDataObjectTags)
    {
        //tPhys = (pix - offset) * scale
        tPhys = (pix - (m_pDataObjectTags->m_axisOffsets[dim] - m_roi[dim])) * m_pDataObjectTags->m_axisScales[dim];
    }
    else
    {
        tPhys = pix;
    }

    return tPhys;
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

   int numMats = 0;
   int tMat = 0;
   cv::Mat_<_Tp> *tempMat = NULL;
   cv::Mat_<_Tp> *rhsMat = NULL;

   rhs.freeData();
   rhs.m_type = lhs.m_type;
   char rhsOldContinuous = rhs.getDims() > 2 ? rhs.m_continuous : 0; //if dims(rhs)<=2, then the continuity-flag should only be influenced by lhs, since then the continuity doesn't change the representation and the constructor of empty dataObject sets the flag to one (default)
   rhs.m_continuous = rhsOldContinuous | lhs.m_continuous;

   if (regionOnly || lhs.m_dims == 0) //Marc: bug, if empty data object, it is necessary to use this if case, too.
   {
         numMats = lhs.getNumPlanes();
         CreateFunc<_Tp>(&rhs, lhs.m_dims, lhs.m_size, rhs.m_continuous, NULL, NULL);
         for (int nMat = 0; nMat < numMats; nMat++)
         {
            tMat = lhs.seekMat(nMat);
            tempMat = (cv::Mat_<_Tp> *)lhs.m_data[tMat];
            rhsMat = (cv::Mat_<_Tp> *)rhs.m_data[nMat];
            tempMat->copyTo(*rhsMat);
         }
   }
   else
   {
         numMats = lhs.mdata_size();
         CreateFunc<_Tp>(&rhs, lhs.m_dims, lhs.m_osize, rhs.m_continuous, NULL, NULL);

         for(int i = 0 ; i < rhs.m_size.m_p[-1]; i++)
         {
             rhs.m_size.m_p[i] = lhs.m_size.m_p[i];
         }
         for(int i = 0 ; i < rhs.m_roi.m_p[-1]; i++)
         {
             rhs.m_roi.m_p[i] = lhs.m_roi.m_p[i];
         }


         for (int nMat = 0; nMat < numMats; nMat++)
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
RetVal DataObject::copyTo(DataObject &rhs, unsigned char regionOnly) const
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
template<typename _Tp> RetVal DeepCopyPartialFunc(const DataObject &lhs, DataObject &rhs)
{
    if (&lhs == &rhs)
    {
            return ito::retOk;
    }

    int lhs_numPlanes = lhs.getNumPlanes();
    int rhs_numPlanes = rhs.getNumPlanes();
    int dims_lhs = lhs.getDims();
	int dims_rhs = rhs.getDims();

    const cv::Mat **lhs_mdata = lhs.get_mdata();
    cv::Mat **rhs_mdata = rhs.get_mdata();
    const cv::Mat *lhs_mat;
    cv::Mat *rhs_mat;

    if (lhs_numPlanes == rhs_numPlanes) //both planes have the same size, line wise or block wise memcpy is possible
    {
        int sizeX = lhs.getSize(dims_lhs - 1);
        int sizeY = lhs.getSize(dims_lhs - 2);
		int sizeX_rhs = rhs.getSize(dims_rhs - 1);

        int lineBytes = sizeX * sizeof(_Tp);
        int planeBytes = sizeY * lineBytes;

        const uchar *lhs_ptr;
        uchar *rhs_ptr;

        for (int nMat = 0; nMat < lhs_numPlanes; ++nMat)
        {
            lhs_mat = lhs_mdata[lhs.seekMat(nMat, lhs_numPlanes)];
            rhs_mat = rhs_mdata[rhs.seekMat(nMat, lhs_numPlanes)];

            if (lhs_mat->isContinuous() && rhs_mat->isContinuous())
            {
                memcpy(rhs_mat->data, lhs_mat->data, planeBytes);
            }
            else if (sizeX == sizeX_rhs) 
            {
                lhs_ptr = lhs_mat->data;
                rhs_ptr = rhs_mat->data;

                for (int y = 0; y < sizeY; ++y)
                {
                    memcpy(rhs_ptr, lhs_ptr, lineBytes);
                    lhs_ptr += lhs_mat->step[0];
                    rhs_ptr += rhs_mat->step[0];
                }
            }
			else //so lhs has shape nXm while rhs is of shape mXn so it is necessary to loop ofer the cv:mats 
			{
				lhs_ptr = lhs_mat->data;
				rhs_ptr = rhs_mat->data;
				int row;
				int col;
				
				
				for (row = 0; row < lhs_mat -> rows; ++row)
				{
					for (col=0; col < lhs_mat -> cols; ++col)
					{
                        rhs_mat->ptr<_Tp>(col)[row] = ((const _Tp*)((lhs_ptr + row * lineBytes)))[col];
					}
				}	

			}
        }
    }
    else
    {
        //the number of planes is not equal, hence each plane has another size (only the type, the sequence of the size of (non-one-size) dimensions is equal)
        //this can mean a partial copy from a 1x5 object to a 5x1x1 object.
        DObjConstIterator lhs_it = lhs.constBegin();
        DObjIterator rhs_it = rhs.begin();

        while (lhs_it != lhs.constEnd())
        {
            memcpy(*rhs_it, *lhs_it, sizeof(_Tp));
            ++lhs_it;
            ++rhs_it;
        }
    }

    return ito::retOk;
}

typedef RetVal (*tDeepCopyPartialFunc)(const DataObject &lhs, DataObject &rhs);

MAKEFUNCLIST(DeepCopyPartialFunc);

//! high-level, non-templated method. Deeply copies data of this data object which is within its ROI to the ROI of rhs.
/*!
    \param &rhs is the right-handed data object, where data is copied to.
    \return retOk
    \throws cv::Exception(CV_StsAssert) if sizes or type of both matrices are not equal
    \sa DeepCopyPartialFunc
*/
RetVal DataObject::deepCopyPartial(DataObject &copyTo)
{
    if(m_type != copyTo.m_type)
    {
        cv::error(cv::Exception(CV_StsAssert, "DataObject - operands differ in type","", __FILE__,__LINE__));
    }

    //calc and compare squeezed dimensions
    int thisDims = this->getDims();
    int rhsDims = copyTo.getDims();
    int *thisSizes = new int[thisDims];
    int *rhsSizes = new int[rhsDims];

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
        rhsSizes[j] = copyTo.getSize(i);
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

    ito::RetVal ret = fListDeepCopyPartialFunc[m_type](*this, copyTo);

    //in copypartial, no tags or axis (offsets, scales...) should be copy to core object!
    //if(!ret.containsError())
    //{
    //    this->copyTagMapTo(rhs);   //Deepcopy the tagspace
    //    this->copyAxisTagsTo(rhs); //Deepcopy the tagspace
    //}
    return ret;
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

    if(m_pDataObjectTags->m_tags.empty())
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

    const double *rot = m_pDataObjectTags->m_rotMatrix;
    rhs.setXYRotationalMatrix(rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);

    int axisNumRhs = rhs.getDims() - 1;
    bool isValid;

    for(int axisNum = getDims() - 1; axisNum > -1 ; axisNum--)
    {
        rhs.setAxisOffset(axisNumRhs, getAxisOffset(axisNum));
        rhs.setAxisScale(axisNumRhs, getAxisScale(axisNum));

        isValid = false;
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
   int sizes[2] = {1, 1};
   return zeros(2, sizes, type);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a zero-value matrix of size 1 x size with the given type
/*!
    \param size is the desired length of the vector
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::zeros(const int size, const int type)
{
   int sizes[2] = {1, size};
   return zeros(2, sizes, type);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a zero-value matrix of size sizeY x sizeX with the given type
/*!
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::zeros(const int sizeY, const int sizeX, const int type)
{
   int sizes[2] = {sizeY, sizeX};
   return zeros(2, sizes, type);
}

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
RetVal DataObject::zeros(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous)
{
   int sizes[3] = {sizeZ, sizeY, sizeX};
   return zeros(3, sizes, type, continuous);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for creation of zero-valued matrix-plane
/*!

    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param **dstMat is the pointer to the already allocated cv::Mat_<type>-matrix-plane
    \return retOk
    \sa zeros
*/
template<typename _Tp> RetVal ZerosFunc(const int sizeY, const int sizeX, uchar **dstMat)
{
   (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::zeros(static_cast<int>(sizeY), static_cast<int>(sizeX));

   return 0;
}

typedef RetVal (*tZerosFunc)(const int sizeY, const int sizeX, uchar **dstMat);
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
RetVal DataObject::zeros(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous)
{
   freeData();
   create(dimensions, sizes, type, continuous);

   int numMats = getNumPlanes();

    int sizeX = sizes[dimensions - 1];
    int sizeY = 1;
    if(dimensions > 1)
    {
        sizeY = sizes[dimensions - 2];
    }

    for (int matn = 0; matn < numMats; matn++)
    {
        fListZerosFunc[type](sizeY, sizeX, &(m_data[matn]));
    }

   return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-value matrix of size 1x1 with the given type
/*!
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const int type)
{
   int sizes[2] = {1, 1};
   return ones(2, sizes, type);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-value matrix of size 1 x size with the given type
/*!
    \param size is the desired length of the vector
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const int size, const int type)
{
   int sizes[2] = {1, size};
   return ones(2, sizes, type);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! allocates a one-value matrix of size sizeY x sizeX with the given type
/*!
    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param type is the desired type-number
    \return retOk
    \sa zeros, ZerosFunc
*/
RetVal DataObject::ones(const int sizeY, const int sizeX, const int type)
{
   int sizes[2] = {sizeY, sizeX};
   return ones(2, sizes, type);
}

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
RetVal DataObject::ones(const int sizeZ, const int sizeY, const int sizeX, const int type, const unsigned char continuous)
{
   int sizes[3] = {sizeZ, sizeY, sizeX};
   return ones(3, sizes, type, continuous);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method for creation of one-valued matrix-plane
/*!

    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param **dstMat is the pointer to the already allocated cv::Mat_<type>-matrix-plane
    \return retOk
    \sa zeros
*/
template<typename _Tp> RetVal OnesFunc(const int sizeY, const int sizeX, uchar **dstMat)
{
   (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::ones(static_cast<int>(sizeY), static_cast<int>(sizeX));
   return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, overloaded template method for creation of one-valued matrix-plane of RGBA32
/*!

    \param sizeY are the number of rows
    \param sizeX are the number of columns
    \param **dstMat is the pointer to the already allocated cv::Mat_<type>-matrix-plane
    \return retOk
    \sa zeros
*/
template<> RetVal OnesFunc<ito::Rgba32>(const int sizeY, const int sizeX, uchar **dstMat)
{
   (*((cv::Mat_<ito::Rgba32> *)(*dstMat))) = cv::Mat_<ito::Rgba32>(static_cast<int>(sizeY), static_cast<int>(sizeX));
   memset( (*((cv::Mat*)(*dstMat))).ptr<ito::uint8>(), 255, static_cast<int>(sizeY) * static_cast<int>(sizeX) * sizeof(ito::Rgba32));
   return 0;
}

typedef RetVal (*tOnesFunc)(const int sizeY, const int sizeX, uchar **dstMat);
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
RetVal DataObject::ones(const unsigned char dimensions, const int *sizes, const int type, const unsigned char continuous)
{
    freeData();
    create(dimensions, sizes, type, continuous);

    int numMats = getNumPlanes();

    int sizeX = sizes[dimensions - 1];
    int sizeY = 1;
    if(dimensions > 1)
    {
        sizeY = sizes[dimensions - 2];
    }

    for (int matn = 0; matn < numMats; matn++)
    {
        fListOnesFunc[type](sizeY, sizeX, &(m_data[matn]));
    }

   return 0;
}

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
   int sizes[2] = {1, 1};
   return rand(2, sizes, type, randMode);
}

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
RetVal DataObject::rand(const int size, const int type, const bool randMode)
{
   int sizes[2] = {1, size};
   return rand(2, sizes, type, randMode);
}

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
RetVal DataObject::rand(const int sizeY, const int sizeX, const int type, const bool randMode)
{
   int sizes[2] = {sizeY, sizeX};
   return rand(2, sizes, type, randMode);
}

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
RetVal DataObject::rand(const int sizeZ, const int sizeY, const int sizeX, const int type, const bool randMode, const unsigned char continuous)
{
   int sizes[3] = {sizeZ, sizeY, sizeX};
   return rand(3, sizes, type, randMode, continuous);
}

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
template<typename _Tp> RetVal RandFunc(const int sizeY, const int sizeX, const double value1, const double value2, const bool randMode, uchar **dstMat)
{
    //dstMat must already be preallocated concerning size and type!
    if(randMode)
    {
        cv::randn((*((cv::Mat_<_Tp> *)(*dstMat))), value1, value2);
    }
    else
    {
        cv::randu((*((cv::Mat_<_Tp> *)(*dstMat))), value1, value2);
    }
   return retOk;
}

//! template specialisation for low-level, templated method for creation of random-valued matrix-plane of type complex128
/*!
    \return retOk
    \sa  RandFunc, zeros, ones
*/
template<> RetVal RandFunc<ito::complex128>(const int sizeY, const int sizeX, const double value1, const double value2, const bool randMode, uchar **dstMat)
{
    //dstMat must already be preallocated concerning size and type!

    if(randMode)
    {
        cv::Mat_<ito::float64> tempMat(sizeY, sizeX * 2, ((cv::Mat*)(*dstMat))->ptr<ito::float64>());
        cv::randn(tempMat, value1, value2);       
    }
    else
    {
        cv::randu((*((cv::Mat_<ito::complex128> *)(*dstMat))), value1, value2);
    }
   return retOk;
}

//! template specialisation for low-level, templated method for creation of random-valued matrix-plane of type rgba32
/*!
    \return retOk
    \sa  RandFunc, zeros, ones
*/
template<> RetVal RandFunc<ito::Rgba32>(const int sizeY, const int sizeX, const double value1, const double value2, const bool randMode, uchar **dstMat)
{
    //dstMat must already be preallocated concerning size and type!
    cv::Mat_<ito::uint8> tempMat(sizeY, sizeX * 4, ((cv::Mat*)(*dstMat))->ptr<ito::uint8>());
    if(randMode)
    {
        cv::randn(tempMat, value1, value2);
    }
    else
    {
        cv::randu(tempMat, value1, value2);
    }
   return retOk;
}

typedef RetVal (*tRandFunc)(const int sizeY, const int sizeX, const double value1, const double value2, const bool randMode, uchar **dstMat);
MAKEFUNCLIST(RandFunc);

//! high-level, non-templated base function for allocation of new matrix whose elements are all set to one
/*!
    \detail this function allocates an random value matrix using cv::randu for uniform (randMode = false) or gausion noise (randMode = true).
            In case of an integer type, the uniform noise is from min(inclusiv) to max(exclusive). For floating point types, the noise is between 0(inclusiv) and 1(exclusiv).
            In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0.
    \param dimensions indicates the number of dimensions
    \param *sizes is a vector with the same length than dimensions. Every element indicates the size of the specific dimension
    \param type is the desired data-element-type
    \param randMode switch mode between uniform distributed(false) and normal distributed noise(true)
    \param continuous indicates wether the data should be in one continuous block (true) or not (false)
    \return retOk
    \sa OnesFunc
*/
RetVal DataObject::rand(const unsigned char dimensions, const int *sizes, const int type, const bool randMode, const unsigned char continuous)
{
   freeData();
   create(dimensions, sizes, type, continuous);

   double val1 = 1.0;
   double val2 = 1.0;

   int numMats = getNumPlanes();

    if(randMode)
    {
        switch(type)
        {
            case ito::tRGBA32:
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
            case ito::tRGBA32:
            case ito::tUInt8:
                val1 = (double)std::numeric_limits<uint8>::min();
                val2 = (double)std::numeric_limits<uint8>::max(); //was +1 in order to make it inclusive, however this leads to overflows with non-uniform distribution
            break;
            case ito::tInt8:
                val1 = (double)std::numeric_limits<int8>::min();
                val2 = (double)std::numeric_limits<int8>::max(); //was +1 in order to make it inclusive, however this leads to overflows with non-uniform distribution
            break;
            case ito::tUInt16:
                val1 = (double)std::numeric_limits<uint16>::min();
                val2 = (double)std::numeric_limits<uint16>::max(); //was +1 in order to make it inclusive, however this leads to overflows with non-uniform distribution
            break;
            case ito::tInt16:
                val1 = (double)std::numeric_limits<int16>::min();
                val2 = (double)std::numeric_limits<int16>::max(); //was +1 in order to make it inclusive, however this leads to overflows with non-uniform distribution
            break;
            case ito::tUInt32:
                val1 = (double)std::numeric_limits<uint32>::min();
                val2 = (double)std::numeric_limits<uint32>::max();
            break;
            case ito::tInt32:
                val1 = (double)std::numeric_limits<int32>::min();
                val2 = (double)std::numeric_limits<int32>::max(); //was +1 in order to make it inclusive, however this leads to overflows with non-uniform distribution
            break;
            default:
                val1 = 0.0;
                val2 = 1.0;
                break;
        }
    }


    int sizeX = sizes[dimensions - 1];
    int sizeY = 1;
    if(dimensions > 1)
    {
        sizeY = sizes[dimensions - 2];
    }

    for (int matn = 0; matn < numMats; matn++)
    {
        fListRandFunc[type](sizeY, sizeX, val1, val2, randMode, &(m_data[matn]));
    }

   return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method that executes a shallow-copy of every matrix-plane in the source-vector and stores the copies in the destination-vector
/*!
    \param &src is the source vector which contains matrix-planes of type cv::Mat_<_Tp>
    \param &dst is the destination vector, where the shallow-copies are stored. dst should be empty at the beginning
    \return retOk
    \sa operator =, DataObject::DataObject(const DataObject& copyConstr)
*/
//template<typename _Tp> RetVal CopyMatFunc(const std::vector<int *> &src, std::vector<int *> &dst)
template<typename _Tp> RetVal CopyMatFunc(const uchar **src, uchar **&dst, bool transposed, const int sizeofs)
{
    int size = (*reinterpret_cast<int *>(src - sizeofs));

    if (dst)
    {
        dst = reinterpret_cast<uchar **>(realloc((dst - sizeofs), (size + sizeofs) * sizeof(uchar *)));
    }
    else
    {
        int numBytes = (size + sizeofs) * sizeof(uchar *);
        dst = reinterpret_cast<uchar **>(calloc(numBytes, 1));
        memset(dst, 0, numBytes);
    }

    (*reinterpret_cast<int *>(dst)) = size;
    dst += sizeofs;

    if(transposed)
    {
        for( int i = 0 ; i < size ; i++)
        {
            dst[i] = reinterpret_cast<uchar *>( new cv::Mat_<_Tp>( ((const cv::Mat_<_Tp> *)(src[i]))->t() ) );
        }
    }
    else
    {
        for( int i = 0 ; i < size ; i++)
        {
            dst[i] = reinterpret_cast<uchar *>( new cv::Mat_<_Tp>( *((const cv::Mat_<_Tp> *)(src[i])) ) );
        }
    }

   return RetVal(retOk);
}

typedef RetVal (*tCopyMatFunc)(const uchar **src, uchar  **&dst, bool transposed, const int sizeofs);
MAKEFUNCLIST(CopyMatFunc)

//-----------------------------------------------------------------------------------------------------------
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
    case CV_8UC4: dataObjType = ito::tRGBA32; break;
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

    const int sizes[2] = {(int)rhs.rows, (int)rhs.cols};
    create(2, sizes, dataObjType, &rhs, 1);

    return *this;
}

//-----------------------------------------------------------------------------------------------------------
//! assign-operator which makes a shallow-copy of the rhs data object and stores it in this data object
/*!
    shallow-copy means, that the header information of the rhs data-object is physically copied to this-dataObject while
    the data is shared, hence, only its reference counter is incremented.

    The previous array covered by this data object is completely released before assigning the new rhs data object.
    In order to deeply copy the values from one object into another pre-allocated object use the method deepCopyPartial.

    \param &rhs is the data object where the shallow copy is taken from. At first, the existing data of this object is freed.
    \return this data object
    \throws cv::Exception if lock state of both objects is not equal. Please make sure, that both lock states are equal
    \sa CopyMatFunc, deepCopyPartial
*/
DataObject & DataObject::operator = (const DataObject &rhs)
{
   if (this == &rhs)
   {
      return *this;
   }

   freeData();

   createHeaderWithROI(rhs.m_dims, rhs.m_size.m_p, rhs.m_osize.m_p, rhs.m_roi.m_p);
   m_type = rhs.m_type;
   m_continuous = rhs.m_continuous;
   m_pRefCount = rhs.m_pRefCount;
   m_owndata = rhs.m_owndata; //if the rhs object already shared its data with another owner, this object will also shared data with the original owner!

   if(rhs.m_dims > 0 || m_pRefCount)
   {
        CV_XADD((m_pRefCount),1); //++;

        m_pDataObjectTags = rhs.m_pDataObjectTags;

        try
        {
            fListCopyMatFunc[m_type](const_cast<const uchar**>(rhs.m_data), m_data, false, m_sizeofs);
        }
        catch(cv::Exception /*&exc*/) //memory error
        {
            secureFreeData();
            throw; //rethrow exception
        }

   } //else: rhs is empty

   return *this;
}

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
    createHeaderWithROI(copyConstr.m_dims, copyConstr.m_size.m_p, copyConstr.m_osize.m_p, copyConstr.m_roi.m_p);
    m_pRefCount = copyConstr.m_pRefCount;
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
            fListCopyMatFunc[m_type](const_cast<const uchar**>(copyConstr.m_data), m_data, false, m_sizeofs);
        }
        catch(cv::Exception /*&exc*/) //memory error
        {
            secureFreeData();
            throw; //rethrow exception
        }
    }
}

//-----------------------------------------------------------------------------------------------------------
DataObject::DataObject(const DataObject& dObj, bool transposed)
{
    if(!transposed) //shallow copy of dataobject
    {
        createHeaderWithROI(dObj.m_dims, dObj.m_size.m_p, dObj.m_osize.m_p, dObj.m_roi.m_p);
        m_pRefCount = dObj.m_pRefCount;
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
                fListCopyMatFunc[m_type](const_cast<const uchar**>(dObj.m_data), m_data, false, m_sizeofs);
            }
            catch(cv::Exception /*&exc*/) //memory error
            {
                secureFreeData();
                throw; //rethrow exception
            }
        }
    }
    else //deep, transposed copy of dataObject
    {
        int dims = dObj.m_dims;
        int *newSize = new int[dims];
        int *newOSize = new int[dims];
        int *newRoi = new int[dims];
        memcpy(newSize, dObj.m_size.m_p, dims * sizeof(int));
        memcpy(newOSize, dObj.m_osize.m_p, dims * sizeof(int));
        memcpy(newRoi, dObj.m_roi.m_p, dims * sizeof(int));

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

            m_pDataObjectTags = new DataObjectTagsPrivate( *dObj.m_pDataObjectTags ); //deep copy of tags

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
            m_pDataObjectTags = NULL;
        }

        m_type = dObj.m_type;
        m_continuous = dObj.m_continuous;
        m_owndata = dObj.m_owndata;

        m_data = NULL;

        if (dObj.m_data != NULL)
        {
            mdata_realloc(const_cast<DataObject &>(dObj).mdata_size());

            try
            {
                fListCopyMatFunc[m_type](const_cast<const uchar**>(dObj.m_data), m_data, true, m_sizeofs);
            }
            catch(cv::Exception /*&exc*/) //memory error
            {
                secureFreeData();
                throw; //rethrow exception
            }
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
template<typename _Tp> RetVal AssignScalarFunc(DataObject *src, const ito::tDataType type, const void *scalar)
{
    int numMats = src->getNumPlanes();
    int MatNum = 0;

    _Tp scalar2 = ito::numberConversion<_Tp>(type, scalar); //convert the void* scalar to the data type of the source data object (throws error if not possible)

    cv::Mat** tempMats = src->get_mdata();
    cv::Mat* tempMat;
    _Tp* dstPtr = NULL;
    int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
    int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
    for (int nmat = 0; nmat < numMats; nmat++)
    {
        MatNum = src->seekMat(nmat, numMats);
        tempMat = tempMats[MatNum];

        for (int y = 0; y < sizey; y++)
        {
            dstPtr = tempMat->ptr<_Tp>(y);
            for (int x = 0; x < sizex; ++x)
            {
                dstPtr[x] = scalar2;
            }
        }
    }

    return retOk;
}

typedef RetVal (*tAssignScalarFunc)(DataObject *src, const ito::tDataType type, const void *scalar);
MAKEFUNCLIST(AssignScalarFunc);


//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const int8 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tInt8, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const uint8 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tUInt8, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const int16 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tInt16, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const uint16 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tUInt16, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const int32 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tInt32, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const uint32 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tUInt32, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const float32 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tFloat32, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const float64 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tFloat64, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const complex64 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tComplex64, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const complex128 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tComplex128, static_cast<const void*>(&value));
    return *this;
}

//! Every data element in this data object is set to the given value
/*!
    \param value is the scalar assignment value
    \return modified data object
    \sa AssignScalarValue
*/
DataObject & DataObject::operator = (const ito::Rgba32 &value)
{
    fListAssignScalarFunc[m_type](this, ito::tRGBA32, static_cast<const void*>(&value));
    return *this;
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
template<typename _Tp> RetVal AssignScalarMaskFunc(DataObject *src, const DataObject *mask, const ito::tDataType type, const void *scalar)
{
    if (mask->getDims() < 2)
    {
        return AssignScalarFunc<_Tp>(src, type, scalar);
    }
    else
    {
        ito::RetVal retval;
        if (mask->getType() != ito::tUInt8)
        {
            retval += ito::RetVal(ito::retError, 0, "mask must have type uint8");
        }
        else if (mask->getSize() != src->getSize())
        {
            retval += ito::RetVal(ito::retError, 0, "size of mask must correspond to size of data object");
        }
        else
        {
            int numMats = src->getNumPlanes();
            int MatNum = 0;

            _Tp scalar2 = ito::numberConversion<_Tp>(type, scalar); //convert the void* scalar to the data type of the source data object (throws error if not possible)

            cv::Mat** tempMats = src->get_mdata();
            cv::Mat* tempMat;
            _Tp* dstPtr = NULL;
            const cv::Mat** maskMats = mask->get_mdata();
            const cv::Mat* maskMat;
            const ito::uint8* maskPtr = NULL;

            int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
            int sizey = static_cast<int>(src->getSize(src->getDims() - 2));

            for (int nmat = 0; nmat < numMats; nmat++)
            {
                MatNum = src->seekMat(nmat, numMats);
                tempMat = tempMats[MatNum];
                MatNum = mask->seekMat(nmat, numMats);
                maskMat = maskMats[MatNum];

                for (int y = 0; y < sizey; y++)
                {
                    maskPtr = maskMat->ptr<ito::uint8>(y);
                    dstPtr = tempMat->ptr<_Tp>(y);
                    for (int x = 0; x < sizex; ++x)
                    {
                        if (maskPtr[x] > 0)
                        {
                            dstPtr[x] = scalar2;
                        }
                    }
                }
            }
        }

        return retval;
    }
}

typedef RetVal (*tAssignScalarMaskFunc)(DataObject *src, const DataObject *mask, const ito::tDataType type, const void *scalar);
MAKEFUNCLIST(AssignScalarMaskFunc);

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const int8 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tInt8, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const uint8 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tUInt8, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const int16 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tInt16, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const uint16 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tUInt16, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const int32 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tInt32, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const uint32 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tUInt32, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const float32 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tFloat32, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const float64 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tFloat64, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const complex64 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tComplex64, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const complex128 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tComplex128, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/
RetVal DataObject::setTo(const ito::Rgba32 &value, const DataObject &mask /*= DataObject()*/)
{
    return fListAssignScalarMaskFunc[m_type](this, &mask, ito::tRGBA32, static_cast<const void*>(&value));
}

//! Sets all or some of the array elements to the specific value
/*!
    \param assigned scalar converted to the actual array type
    \param mask Operation mask of the same size as *this and type uint8 or empty data object if no mask should be considered (default)
    \return retError in case of error
    \sa AssignScalarValue
*/

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
   int srcTmat1 = 0;
   int srcTmat2 = 0;
   int dstTmat = 0;
   int numMats = dObj1->getNumPlanes();
   const cv::Mat_<_Tp> *cvSrcTmat1 = NULL;
   const cv::Mat_<_Tp> *cvSrcTmat2 = NULL;
   cv::Mat_<_Tp> *cvDstTmat = NULL;

    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        dstTmat = dObjRes->seekMat(nmat, numMats);
        srcTmat1 = dObj1->seekMat(nmat, numMats);
        srcTmat2 = dObj2->seekMat(nmat, numMats);
        cvSrcTmat1 = static_cast<const cv::Mat_<_Tp>*>(dObj1->get_mdata()[srcTmat1]);
        cvSrcTmat2 = static_cast<const cv::Mat_<_Tp>*>(dObj2->get_mdata()[srcTmat2]);
        cvDstTmat = static_cast<cv::Mat_<_Tp>*>(dObjRes->get_mdata()[dstTmat]);
        *cvDstTmat = *cvSrcTmat1 + *cvSrcTmat2;
    }

   return RetVal(retOk);
}

typedef RetVal (*tAddFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);
MAKEFUNCLIST(AddFunc);

template<typename _Tp> RetVal AddScalarFunc(const DataObject *dObjIn, ito::float64 scalar, DataObject *dObjOut)
{
    int srcTmat = 0;
    int dstTmat = 0;
    int numMats = dObjIn->getNumPlanes();
    const cv::Mat_<_Tp> *cvSrc = NULL;
    cv::Mat_<_Tp> *cvDest = NULL;
    
    cv::Scalar s = scalar;
    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        dstTmat = dObjOut->seekMat(nmat, numMats);
        srcTmat = dObjIn->seekMat(nmat, numMats);
        cvSrc  = static_cast<const cv::Mat_<_Tp> *>(dObjIn->get_mdata()[srcTmat]);
        cvDest = static_cast<cv::Mat_<_Tp> *>(dObjOut->get_mdata()[dstTmat]);
        *cvDest = *cvSrc + s;
    }

    return RetVal(retOk);
}

template<> RetVal AddScalarFunc<ito::Rgba32>(const DataObject *dObjIn, ito::float64 scalar, DataObject *dObjOut)
{
   int srcTmat = 0;
   int dstTmat = 0;
   int numMats = dObjIn->getNumPlanes();
   const cv::Mat_<ito::Rgba32> *cvSrc = NULL;
   cv::Mat_<ito::Rgba32> *cvDest = NULL;
   cv::Scalar s;
   ito::int32 sign = scalar < 0.0 ? -1 : 1;
   ito::uint32 val = fabs(scalar) > 4294967295 ? 0xFFFFFFFF : (ito::uint32)fabs(scalar);
   s[0] = ((ito::uint8*)&val)[0] * sign;
   s[1] = ((ito::uint8*)&val)[1] * sign;
   s[2] = ((ito::uint8*)&val)[2] * sign;
   s[3] = ((ito::uint8*)&val)[3] * sign;

    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        dstTmat = dObjOut->seekMat(nmat, numMats);
        srcTmat = dObjIn->seekMat(nmat, numMats);
        cvSrc  = static_cast<const cv::Mat_<ito::Rgba32> *>(dObjIn->get_mdata()[srcTmat]);
        cvDest = static_cast<cv::Mat_<ito::Rgba32> *>(dObjOut->get_mdata()[dstTmat]);
        *cvDest = *cvSrc + s;
    }

   return RetVal(retOk);
}

typedef RetVal (*tAddScalarFunc)(const DataObject *dObjIn, ito::float64 scalar, DataObject *dObjOut);
MAKEFUNCLIST(AddScalarFunc);



template<typename _Tp> RetVal AddComplexScalarFunc(const DataObject *dObjIn, ito::complex128 scalar, DataObject *dObjOut)
{
   int srcTmat = 0;
   int dstTmat = 0;
   int numMats = dObjIn->getNumPlanes();
   const cv::Mat **cvSrc = dObjIn->get_mdata();
   cv::Mat **cvDest = dObjOut->get_mdata();

   if (std::abs(scalar.imag()) < std::numeric_limits<ito::float64>::epsilon())
   {
       cv::Scalar s = scalar.real();

        for (int nmat = 0; nmat < numMats; ++nmat)
        {
            dstTmat = dObjOut->seekMat(nmat, numMats);
            srcTmat = dObjIn->seekMat(nmat, numMats);
            *(cvDest[dstTmat]) = *(cvSrc[srcTmat]) + s;
        }
   }
   else
   {
       cv::error(cv::Exception(CV_StsAssert,"The given complex value cannot be converted to a real value. However the data object is real.","", __FILE__, __LINE__));
   }

   return RetVal(retOk);
}

template<> RetVal AddComplexScalarFunc<ito::complex64>(const DataObject *dObjIn, ito::complex128 scalar, DataObject *dObjOut)
{
    int numMats = dObjIn->getNumPlanes();
    const cv::Mat **cvSrcs = dObjIn->get_mdata();
    cv::Mat **cvDests = dObjOut->get_mdata();
    const cv::Mat *cvSrc;
    cv::Mat *cvDest;
    const ito::complex64 *srcPtr;
    ito::complex64 *destPtr;
    ito::complex64 value = cv::saturate_cast<ito::complex64>(scalar);

    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        cvDest = cvDests[dObjOut->seekMat(nmat, numMats)];
        cvSrc = cvSrcs[dObjIn->seekMat(nmat, numMats)];
        for (int r = 0; r < cvDest->rows; ++r)
        {
            srcPtr = cvSrc->ptr<ito::complex64>(r);
            destPtr = cvDest->ptr<ito::complex64>(r);
            for (int c = 0; c < cvDest->cols; ++c)
            {
                destPtr[c] = srcPtr[c] + value;
            }
        }
    }

   return RetVal(retOk);
}

template<> RetVal AddComplexScalarFunc<ito::complex128>(const DataObject *dObjIn, ito::complex128 scalar, DataObject *dObjOut)
{
    int numMats = dObjIn->getNumPlanes();
    const cv::Mat **cvSrcs = dObjIn->get_mdata();
    cv::Mat **cvDests = dObjOut->get_mdata();
    const cv::Mat *cvSrc;
    cv::Mat *cvDest;
    const ito::complex128 *srcPtr;
    ito::complex128 *destPtr;

    for (int nmat = 0; nmat < numMats; ++nmat)
    {
        cvDest = cvDests[dObjOut->seekMat(nmat, numMats)];
        cvSrc = cvSrcs[dObjIn->seekMat(nmat, numMats)];
        for (int r = 0; r < cvDest->rows; ++r)
        {
            srcPtr = cvSrc->ptr<ito::complex128>(r);
            destPtr = cvDest->ptr<ito::complex128>(r);
            for (int c = 0; c < cvDest->cols; ++c)
            {
                destPtr[c] = srcPtr[c] + scalar;
            }
        }
    }

   return RetVal(retOk);
}

template<> RetVal AddComplexScalarFunc<ito::Rgba32>(const DataObject *dObjIn, ito::complex128 scalar, DataObject *dObjOut)
{
   int srcTmat = 0;
   int dstTmat = 0;
   int numMats = dObjIn->getNumPlanes();
   const cv::Mat_<ito::Rgba32> **cvSrc = (const cv::Mat_<ito::Rgba32>**)dObjIn->get_mdata();
   cv::Mat_<ito::Rgba32> **cvDest = (cv::Mat_<ito::Rgba32>**)dObjOut->get_mdata();

   if (std::abs(scalar.imag()) < std::numeric_limits<ito::float64>::epsilon())
   {
       cv::Scalar s;
       ito::int32 sign = scalar.real() < 0.0 ? -1 : 1;
       ito::uint32 val = fabs(scalar.real()) > 4294967295 ? 0xFFFFFFFF : (ito::uint32)fabs(scalar.real());
       s[0] = ((ito::uint8*)&val)[0] * sign;
       s[1] = ((ito::uint8*)&val)[1] * sign;
       s[2] = ((ito::uint8*)&val)[2] * sign;
       s[3] = ((ito::uint8*)&val)[3] * sign;

        for (int nmat = 0; nmat < numMats; ++nmat)
        {
            dstTmat = dObjOut->seekMat(nmat, numMats);
            srcTmat = dObjIn->seekMat(nmat, numMats);
            *(cvDest[dstTmat]) = *(cvSrc[srcTmat]) + s;
        }
   }
   else
   {
       cv::error(cv::Exception(CV_StsAssert,"The given complex value cannot be converted to a real value. However the rgba32 data object only supports scalar operations with real scalars.","", __FILE__, __LINE__));
   }

   return RetVal(retOk);
}

typedef RetVal (*tAddComplexScalarFunc)(const DataObject *dObjIn, ito::complex128 scalar, DataObject *dObjOut);
MAKEFUNCLIST(AddComplexScalarFunc);

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
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)
    (fListAddFunc[m_type])(this, &rhs, this);
    return *this;
}

DataObject & DataObject::operator += (const float64 &value)
{
    (fListAddScalarFunc[m_type])(this, value, this);
    return *this;
}

DataObject & DataObject::operator += (const complex128 &value)
{
    (fListAddComplexScalarFunc[m_type])(this, value, this);
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
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

    DataObject result;
    result.m_continuous = rhs.m_continuous;
    copyTo(result, 1);

    (fListAddFunc[m_type])(this, &rhs, &result);

    return result;
}

DataObject DataObject::operator + (const float64 &value)
{
    DataObject result;
    result.m_continuous = this->m_continuous;
    copyTo(result, 1);

    (fListAddScalarFunc[m_type])(this, value, &result);
    return result;
}

DataObject DataObject::operator + (const complex128 &value)
{
    DataObject result;
    result.m_continuous = this->m_continuous;
    copyTo(result, 1);

    (fListAddComplexScalarFunc[m_type])(this, value, &result);
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
   int nmat = 0;
   int srcTmat1 = 0;
   int srcTmat2 = 0;
   int dstTmat = 0;
   int numMats = dObj1->getNumPlanes();
   const cv::Mat_<_Tp> *cvSrcTmat1 = NULL;
   const cv::Mat_<_Tp> *cvSrcTmat2 = NULL;
   cv::Mat_<_Tp> *cvDstTmat = NULL;

    for (nmat = 0; nmat < numMats; nmat++)
    {
        dstTmat = dObjRes->seekMat(nmat, numMats);
        srcTmat1 = dObj1->seekMat(nmat, numMats);
        srcTmat2 = dObj2->seekMat(nmat, numMats);
        cvSrcTmat1 = static_cast<const cv::Mat_<_Tp> *>(dObj1->get_mdata()[srcTmat1]);
        cvSrcTmat2 = static_cast<const cv::Mat_<_Tp> *>(dObj2->get_mdata()[srcTmat2]);
        cvDstTmat = static_cast<cv::Mat_<_Tp> *>(dObjRes->get_mdata()[dstTmat]);
        *cvDstTmat = *cvSrcTmat1 - *cvSrcTmat2;
    }
 
   return retOk;
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
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)
    fListSubFunc[m_type](this, &rhs, this);
    return *this;
}

DataObject & DataObject::operator -= (const float64 &value)
{
    (fListAddScalarFunc[m_type])(this, -value, this);
    return *this;
}

DataObject & DataObject::operator -= (const complex128 &value)
{
    (fListAddComplexScalarFunc[m_type])(this, -value, this);
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
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

    DataObject result;
    result.m_continuous = rhs.m_continuous;
    this->copyTo(result, 1);

    (fListSubFunc[m_type])(this, &rhs, &result);

    return result;
}

DataObject DataObject::operator - (const float64 &value)
{
    DataObject result;
    result.m_continuous = this->m_continuous;
    copyTo(result, 1);

    (fListAddScalarFunc[m_type])(this, -value, &result);
    return result;
}

DataObject DataObject::operator - (const complex128 &value)
{
    DataObject result;
    result.m_continuous = this->m_continuous;
    copyTo(result, 1);

    (fListAddComplexScalarFunc[m_type])(this, -value, &result);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! brief description
/*!
    \todo check for right definition of multiplication
*/
template<typename _Tp> RetVal OpMulFunc(const DataObject *dObj1, const DataObject *dObj2, DataObject *dObjRes)
{
    int nmat = 0;
    int srcTmat1 = 0;
    int srcTmat2 = 0;
    int dstTmat = 0;
    int numMats = dObj1->getNumPlanes();
    const cv::Mat_<_Tp> *cvSrcTmat1 = NULL;
    const cv::Mat_<_Tp> *cvSrcTmat2 = NULL;
    cv::Mat_<_Tp> *cvDstTmat = NULL;
  
    for (nmat = 0; nmat < numMats; nmat++)
    {
        dstTmat = dObjRes->seekMat(nmat, numMats);
        srcTmat1 = dObj1->seekMat(nmat, numMats);
        srcTmat2 = dObj2->seekMat(nmat, numMats);
        cvSrcTmat1 = static_cast<const cv::Mat_<_Tp> *>(dObj1->get_mdata()[srcTmat1]);
        cvSrcTmat2 = static_cast<const cv::Mat_<_Tp> *>(dObj2->get_mdata()[srcTmat2]);
        cvDstTmat = static_cast<cv::Mat_<_Tp> *>(dObjRes->get_mdata()[dstTmat]);
        *cvDstTmat = *cvSrcTmat1 * *cvSrcTmat2;
    }

    return retOk;
}

typedef RetVal (*tOpMulFunc)(const DataObject *src1, const DataObject *src2, DataObject *dst);

MAKEFUNCLIST(OpMulFunc);

//----------------------------------------------------------------------------------------------------------------------------------
//! inplace matrix multiplication of this dataObject with rhs (this *= rhs)
/*!
    This multiplication is only implemented for float32 and float64. The matrix multiplication is only executed plane-by-plane,
    hence, the multiplication is done separately for each plane. This operation is only inplace, if the second matrix is squared 
    and both matrices have the same number of columns. Else, this dataObject is reallocated to the new size.

    For an element wise multiplication use the mul-method.
*/
DataObject & DataObject::operator *= (const DataObject &rhs)
{
    if (this->m_type != rhs.m_type)
    {
        cv::error(cv::Exception(CV_StsAssert, "Data type of objects are different.", "", __FILE__, __LINE__));
        return *this;
    }

    if (m_dims != rhs.m_dims)
    {
        cv::error(cv::Exception(CV_StsAssert, "Number of dimensions of objects are different.", "", __FILE__, __LINE__));
        return *this;
    }

    if (m_dims < 2)
    {
        cv::error(cv::Exception(CV_StsAssert, "DataObjects must be at least two-dimensional.", "", __FILE__, __LINE__));
        return *this;
    }

    if ((m_size[m_dims - 1] != rhs.m_size[m_dims - 2]))
    {
        cv::error(cv::Exception(CV_StsAssert, "DataObject - matrix dimensions inapropriate for matrix multiplication.", "", __FILE__, __LINE__));
        return *this;
    }

    for (int i = 0; i < m_dims - 2; ++i)
    {
        if (getSize(i) != rhs.getSize(i))
        {
            cv::error(cv::Exception(CV_StsAssert, "DataObject - the first n-2 dimensions of both objects must be the same for matrix multiplication (n is the number of total dimensions).", "", __FILE__, __LINE__));
            return *this;
        }
    }

    if (this->m_type != ito::tFloat32 && this->m_type != ito::tFloat64)
    {
        cv::error(cv::Exception(CV_StsAssert, "matrix multiplication is only implemented for float32 and float64 (due to OpenCV or BLAS restrictions)", "", __FILE__, __LINE__));
    }

    if (m_size[m_dims - 1] == rhs.m_size[m_dims - 1]) //The shape of the multiplication result has the same shape than this matrix. Inplace is possible.
    {
        fListOpMulFunc[m_type](this, &rhs, this);
        return *this;
    }
    else
    {
        int *sizes = new int[m_dims];
        for (int i = 0; i < m_dims - 2; ++i)
        {
            sizes[i] = m_size[i];
        }
        sizes[m_dims - 2] = m_size[m_dims - 2];
        sizes[m_dims - 1] = rhs.m_size[m_dims - 1];
        DataObject result(m_dims, sizes, m_type);
        delete[] sizes;
        fListOpMulFunc[m_type](this, &rhs, &result);
        *this = result;
        return *this;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! matrix multiplication of this dataObject with rhs. The result is returned.
/*!
    This multiplication is only implemented for float32 and float64. The matrix multiplication is only executed plane-by-plane,
    hence, the multiplication is done separately for each plane. 

    For an element wise multiplication use the mul-method.
*/
DataObject DataObject::operator * (const DataObject &rhs)
{
    if (this->m_type != rhs.m_type)
    {
        cv::error(cv::Exception(CV_StsAssert, "Data type of objects are different.", "", __FILE__, __LINE__));
        return *this;
    }

    if (m_dims != rhs.m_dims)
    {
        cv::error(cv::Exception(CV_StsAssert, "Number of dimensions of objects are different.", "", __FILE__, __LINE__));
        return *this;
    }

    if (m_dims < 2)
    {
        cv::error(cv::Exception(CV_StsAssert, "DataObjects must be at least two-dimensional.", "", __FILE__, __LINE__));
        return *this;
    }

    if ((m_size[m_dims - 1] != rhs.m_size[m_dims - 2]))
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject - matrix dimensions inapropriate for matrix multiplication.","", __FILE__, __LINE__));
        return *this;
    }

    for (int i = 0; i < m_dims - 2; ++i)
    {
        if (getSize(i) != rhs.getSize(i))
        {
            cv::error(cv::Exception(CV_StsAssert, "DataObject - the first n-2 dimensions of both objects must be the same for matrix multiplication (n is the number of total dimensions).", "", __FILE__, __LINE__));
            return *this;
        }
    }

    if(this->m_type != ito::tFloat32 && this->m_type != ito::tFloat64)
    {
        cv::error(cv::Exception(CV_StsAssert,"matrix multiplication is only implemented for float32 and float64 (due to OpenCV or BLAS restrictions)","",__FILE__,__LINE__));
    }

    int *sizes = new int[m_dims];
    for (int i = 0; i < m_dims - 2; ++i)
    {
        sizes[i] = m_size[i];
    }
    sizes[m_dims - 2] = m_size[m_dims - 2];
    sizes[m_dims - 1] = rhs.m_size[m_dims - 1];
    DataObject result(m_dims, sizes, m_type);
    delete[] sizes;
    fListOpMulFunc[m_type](this, &rhs, &result);
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which multiplies every element of Data Object with a factor
/*!
    \param *src
    \param factor
    \return retOk
*/
template<typename _Tp> RetVal OpScalarMulFunc(DataObject *src, const float64 &factor)
{
   int numMats = src->getNumPlanes();
   int MatNum = 0;

   cv::Mat* tempMat = NULL;
   _Tp* dstPtr = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
       MatNum = src->seekMat(nmat, numMats);
       tempMat = src->get_mdata()[MatNum];

       for (int y = 0; y < tempMat->rows; ++y)
       {
           dstPtr = tempMat->ptr<_Tp>(y);
           for (int x = 0; x < tempMat->cols; ++x)
           {
               dstPtr[x] = cv::saturate_cast<_Tp>(dstPtr[x] * factor);
           }
       }
   }

   return retOk;
}

template<> RetVal OpScalarMulFunc<ito::complex64>(DataObject *src, const float64 &factor)
{
   int numMats = src->getNumPlanes();
   int MatNum = 0;
   ito::complex64 factor2 = ito::complex64(cv::saturate_cast<ito::float32>(factor), 0.0);

   cv::Mat* tempMat = NULL;
   ito::complex64*  dstPtr = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
       MatNum = src->seekMat(nmat, numMats);
       tempMat = src->get_mdata()[MatNum];

       for (int y = 0; y < tempMat->rows; y++)
       {
           dstPtr = tempMat->ptr<ito::complex64>(y);
           for (int x = 0; x < tempMat->cols; x++)
           {
               dstPtr[x] = cv::saturate_cast<ito::complex64>(dstPtr[x] * factor2);
           }
       }
   }

   return retOk;
}

template<> RetVal OpScalarMulFunc<ito::complex128>(DataObject *src, const float64 &factor)
{
   int numMats = src->getNumPlanes();
   int MatNum = 0;
   ito::complex128 factor2 = ito::complex128(cv::saturate_cast<ito::float64>(factor), 0.0);

   cv::Mat * tempMat = NULL;
   ito::complex128* dstPtr = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
       MatNum = src->seekMat(nmat, numMats);
       tempMat = src->get_mdata()[MatNum];

       for (int y = 0; y < tempMat->rows; y++)
       {
           dstPtr = tempMat->ptr<ito::complex128>(y);
           for (int x = 0; x < tempMat->cols; x++)
           {
               dstPtr[x] = cv::saturate_cast<ito::complex128>(dstPtr[x] * factor2);
           }
       }
   }

   return retOk;
}

template<> RetVal OpScalarMulFunc<ito::Rgba32>(DataObject *src, const float64 &factor)
{
   int numMats = src->getNumPlanes();
   int MatNum = 0;

   ito::Rgba32 factor2;
   factor2 = (factor < 0.0 ? (ito::uint32)0 : (factor > 4294967295 ? (ito::uint32)0xFFFFFFFF : (ito::uint32)(factor + 0.5)));

   cv::Mat* tempMat = NULL;
   ito::Rgba32* dstPtr = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
       MatNum = src->seekMat(nmat, numMats);
       tempMat = src->get_mdata()[MatNum];

       for (int y = 0; y < tempMat->rows; y++)
       {
           dstPtr = tempMat->ptr<ito::Rgba32>(y);
           for (int x = 0; x < tempMat->cols; x++)
           {
               dstPtr[x] *= factor2;
           }
       }
   }

   return retOk;
}

typedef RetVal (*tOpScalarMulFunc)(DataObject *src, const float64 &factor);
MAKEFUNCLIST(OpScalarMulFunc);

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level, templated method which multiplies every element of Data Object with a factor
/*!
    \param *src
    \param factor
    \return retOk
*/
template<typename _Tp> RetVal OpScalarComplexMulFunc(DataObject *src, const complex128 &factor)
{
    int numMats = src->getNumPlanes();
    int MatNum = 0;

    cv::Mat* tempMat = NULL;
    _Tp* dstPtr = NULL;

    if (std::abs(factor.imag()) < std::numeric_limits<ito::float64>::epsilon())
    {
        float64 factor2 = factor.real();

        for (int nmat = 0; nmat < numMats; nmat++)
        {
            MatNum = src->seekMat(nmat, numMats);
            tempMat = src->get_mdata()[MatNum];

            for (int y = 0; y < tempMat->rows; ++y)
            {
                dstPtr = tempMat->ptr<_Tp>(y);
                for (int x = 0; x < tempMat->cols; ++x)
                {
                    dstPtr[x] = cv::saturate_cast<_Tp>(dstPtr[x] * factor2);
                }
            }
        }
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert,"The given complex value cannot be converted to a real value. However the data object is real.","", __FILE__, __LINE__));
    }

   return retOk;
}

template<> RetVal OpScalarComplexMulFunc<ito::complex64>(DataObject *src, const complex128 &factor)
{
    int numMats = src->getNumPlanes();
    int MatNum = 0;
    ito::complex64 factor2 = cv::saturate_cast<complex64>(factor);

    cv::Mat* tempMat = NULL;
    ito::complex64*  dstPtr = NULL;

    for (int nmat = 0; nmat < numMats; nmat++)
    {
        MatNum = src->seekMat(nmat, numMats);
        tempMat = src->get_mdata()[MatNum];

        for (int y = 0; y < tempMat->rows; y++)
        {
            dstPtr = tempMat->ptr<ito::complex64>(y);
            for (int x = 0; x < tempMat->cols; x++)
            {
                dstPtr[x] = cv::saturate_cast<ito::complex64>(dstPtr[x] * factor2);
            }
        }
    }

    return retOk;
}

template<> RetVal OpScalarComplexMulFunc<ito::complex128>(DataObject *src, const complex128 &factor)
{
    int numMats = src->getNumPlanes();
    int MatNum = 0;

    cv::Mat * tempMat = NULL;
    ito::complex128* dstPtr = NULL;

    for (int nmat = 0; nmat < numMats; nmat++)
    {
        MatNum = src->seekMat(nmat, numMats);
        tempMat = src->get_mdata()[MatNum];

        for (int y = 0; y < tempMat->rows; y++)
        {
            dstPtr = tempMat->ptr<ito::complex128>(y);
            for (int x = 0; x < tempMat->cols; x++)
            {
                dstPtr[x] = cv::saturate_cast<ito::complex128>(dstPtr[x] * factor);
            }
        }
    }

    return retOk;
}

template<> RetVal OpScalarComplexMulFunc<ito::Rgba32>(DataObject *src, const complex128 &factor)
{
   int numMats = src->getNumPlanes();
   int MatNum = 0;

    if (std::abs(factor.imag()) < std::numeric_limits<ito::float64>::epsilon())
    {
        ito::Rgba32 factor2;
        factor2 = (factor.real() < 0.0 ? (ito::uint32)0 : (factor.real() > 4294967295 ? (ito::uint32)0xFFFFFFFF : (ito::uint32)(factor.real() + 0.5)));

        cv::Mat* tempMat = NULL;
        ito::Rgba32* dstPtr = NULL;

        for (int nmat = 0; nmat < numMats; nmat++)
        {
            MatNum = src->seekMat(nmat, numMats);
            tempMat = src->get_mdata()[MatNum];

            for (int y = 0; y < tempMat->rows; y++)
            {
                dstPtr = tempMat->ptr<ito::Rgba32>(y);
                for (int x = 0; x < tempMat->cols; x++)
                {
                    dstPtr[x] *= factor2;
                }
            }
        }
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert,"The given complex value cannot be converted to a real value. However the rgba32 data object only supports scalar operations with real scalars.","", __FILE__, __LINE__));
    }

    return retOk;
}

typedef RetVal (*tOpScalarComplexMulFunc)(DataObject *src, const complex128 &factor);
MAKEFUNCLIST(OpScalarComplexMulFunc);

//! high-level method which multiplies every element in this data object by a given floating-point factor
/*!
    \param factor
    \sa OpScalarMulFunc
*/
DataObject & DataObject::operator *= (const float64 &factor)
{
   fListOpScalarMulFunc[m_type](this, factor);

   return *this;
}

DataObject & DataObject::operator *= (const complex128 &factor)
{
   fListOpScalarComplexMulFunc[m_type](this, factor);

   return *this;
}

//! high-level method which multiplies every element in this data object by a given floating-point factor. The result matrix is returned as a new matrix.
/*!
    \param factor
    \sa operator *, OpScalarMulFunc
*/
DataObject DataObject::operator * (const float64 &factor)
{
   DataObject result;
   result.m_continuous = m_continuous;
   copyTo(result, 1);

   result *= factor;
   return result;
}

DataObject DataObject::operator * (const complex128 &factor)
{
   DataObject result;
   result.m_continuous = m_continuous;
   copyTo(result, 1);

   result *= factor;
   return result;
}


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
   int numMats = src1->getNumPlanes();
   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const cv::Mat *src1mat;
   const cv::Mat *src2mat;
   cv::Mat *dest;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = src1->seekMat(nmat, numMats);
      rhsMatNum = src2->seekMat(nmat, numMats);
      resMatNum = dst->seekMat(nmat, numMats);

      src1mat = src1->get_mdata()[lhsMatNum];
      src2mat = src2->get_mdata()[rhsMatNum];
      dest = dst->get_mdata()[resMatNum];

      if(src1mat->depth() == 1 || src1mat->depth() == 7)
      {
          cv::error(cv::Exception(CV_StsAssert, "Compare operator not defined for int8.", "", __FILE__, __LINE__));
      }      

      cv::compare(*src1mat, *src2mat, *dest, cmpOp);
   }

   return ito::retOk;
}

//! template specialisation for compare function of type complex64
/*!
    \throws cv::Exception since comparison is not defined for complex input types
*/
template<> RetVal CmpFunc<ito::complex64>(const DataObject * src1, const DataObject * src2, DataObject * dst, int cmpOp)
{
    if (cmpOp == cv::CMP_EQ || cmpOp == cv::CMP_NE)
    {
        int numMats = src1->getNumPlanes();
        const cv::Mat *src1mat;
        const cv::Mat *src2mat;
        cv::Mat *dest;
        ito::uint8 *rowDest;
        bool equal = (cmpOp == cv::CMP_EQ);
        cv::Mat destTemp;
        const ito::uint8 *rowVec2b;

        for (int nmat = 0; nmat < numMats; nmat++)
        {
            src1mat = src1->get_mdata()[src1->seekMat(nmat, numMats)];
            src2mat = src2->get_mdata()[src2->seekMat(nmat, numMats)];
            dest = dst->get_mdata()[dst->seekMat(nmat, numMats)];

            cv::compare(*src1mat, *src2mat, destTemp, cmpOp);

            for (int r = 0; r < src1mat->rows; ++r)
            {
                rowVec2b = destTemp.ptr<ito::uint8>(r);
                rowDest = dest->ptr<ito::uint8>(r);
                for (int c = 0; c < src1mat->cols; ++c)
                {
                    *rowDest++ = *rowVec2b++ & *rowVec2b++;
                }
            }
        }
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert, "complex64 is an unorderable type.", "", __FILE__, __LINE__));
    }
    return ito::retOk;
}

//! template specialisation for compare function of type complex128
/*!
    \throws cv::Exception since comparison is not defined for complex input types
*/
template<> RetVal CmpFunc<ito::complex128>(const DataObject * src1, const DataObject * src2, DataObject * dst, int cmpOp)
{
    if (cmpOp == cv::CMP_EQ || cmpOp == cv::CMP_NE)
    {
        int numMats = src1->getNumPlanes();
        const cv::Mat *src1mat;
        const cv::Mat *src2mat;
        cv::Mat *dest;
        ito::uint8 *rowDest;
        bool equal = (cmpOp == cv::CMP_EQ);
        cv::Mat destTemp;
        const ito::uint8 *rowVec2b;

        for (int nmat = 0; nmat < numMats; nmat++)
        {
            src1mat = src1->get_mdata()[src1->seekMat(nmat, numMats)];
            src2mat = src2->get_mdata()[src2->seekMat(nmat, numMats)];
            dest = dst->get_mdata()[dst->seekMat(nmat, numMats)];

            cv::compare(*src1mat, *src2mat, destTemp, cmpOp);

            for (int r = 0; r < src1mat->rows; ++r)
            {
                rowVec2b = destTemp.ptr<ito::uint8>(r);
                rowDest = dest->ptr<ito::uint8>(r);
                for (int c = 0; c < src1mat->cols; ++c)
                {
                    *rowDest++ = *rowVec2b++ & *rowVec2b++;
                }
            }
        }
    }
    else
    {
        cv::error(cv::Exception(CV_StsAssert, "complex128 is an unorderable type.", "", __FILE__, __LINE__));
    }
    return ito::retOk;
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
template<typename _Tp> RetVal CmpFuncScalar(const DataObject *src, const float64 &value, DataObject *dst, int cmpOp)
{
   int numMats = src->getNumPlanes();
   int matNum = 0;
   int resMatNum = 0;

   const cv::Mat *srcmat;
   cv::Mat *dest;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
      matNum = src->seekMat(nmat, numMats);
      resMatNum = dst->seekMat(nmat, numMats);

      srcmat = src->get_mdata()[matNum];
      dest = dst->get_mdata()[resMatNum];

      if(srcmat->depth() == 1 || srcmat->depth() == 7)
      {
          cv::error(cv::Exception(CV_StsAssert, "Compare operator not defined for int8.", "", __FILE__, __LINE__));
      }      

      cv::compare(*srcmat, value, *dest, cmpOp);
   }

   return ito::retOk;
}

//! template specialisation for compare function of type complex64
/*!
    \throws cv::Exception since comparison is not defined for complex input types
*/
template<> RetVal CmpFuncScalar<ito::complex64>(const DataObject * /*src*/, const float64 &/*value*/, DataObject * /*dst*/, int /*cmpOp*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

//! template specialisation for compare function of type complex128
/*!
    \throws cv::Exception since comparison is not defined for complex input types
*/
template<> RetVal CmpFuncScalar<ito::complex128>(const DataObject * /*src*/, const float64 &/*value*/, DataObject * /*dst*/, int /*cmpOp*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}

typedef RetVal (*tCmpFuncScalar)(const DataObject *src, const float64 &value, DataObject *dst, int cmpOp);
MAKEFUNCLIST(CmpFuncScalar);


//! compare operator, compares for "lower than"
/*!
    \param value is the value with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator < (const float64 &value)
{
    DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous);
    RetVal retValue = fListCmpFuncScalar[m_type](this, value, &resMat, cv::CMP_LT);

    return resMat;
}

//! compare operator, compares for "bigger than"
/*!
    \param value is the value with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator > (const float64 &value)
{
   DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous);
    RetVal retValue = fListCmpFuncScalar[m_type](this, value, &resMat, cv::CMP_GT);

    return resMat;
}

//! compare operator, compares for "lower or equal than"
/*!
    \param value is the value with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator <= (const float64 &value)
{
    DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous);
    RetVal retValue = fListCmpFuncScalar[m_type](this, value, &resMat, cv::CMP_LE);

    return resMat;
}

//! compare operator, compares for "bigger or equal than"
/*!
    \param value is the value with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator >= (const float64 &value)
{
   DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous);
    RetVal retValue = fListCmpFuncScalar[m_type](this, value, &resMat, cv::CMP_GE);

    return resMat;
}

//! compare operator, compares for "equal to"
/*!
    \param value is the value with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator == (const float64 &value)
{
    DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous);
    RetVal retValue = fListCmpFuncScalar[m_type](this, value, &resMat, cv::CMP_EQ);

    return resMat;
}

//! compare operator, compares for "unequal to"
/*!
    \param value is the value with which this data object should element-wisely be compared
    \return compare matrix of type uint8, which contains 0 or 1, depending on the result of the element-wise comparison
    \throws cv::Exception if both data objects doesn't have the same size or type
    \sa CmpFunc
*/
DataObject DataObject::operator != (const float64 &value)
{
    DataObject resMat(m_dims, m_size.m_p, tUInt8, this->m_continuous);
    RetVal retValue = fListCmpFuncScalar[m_type](this, value, &resMat, cv::CMP_NE);

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
   int numMats = src->getNumPlanes();
   int MatNum = 0;

   cv::Mat_<_Tp> * tempMat = NULL;
   _Tp* dstPtr = NULL;
   int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
   int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
   for (int nmat = 0; nmat < numMats; nmat++)
   {
      MatNum = src->seekMat(nmat, numMats);
      //TODO: check if non iterator version is working
      tempMat = static_cast<cv::Mat_<_Tp> *>(src->get_mdata()[MatNum]);

      for (int y = 0; y < sizey; y++)
      {
          dstPtr = (_Tp*)tempMat->ptr(y);
          for (int x = 0; x < sizex; x++)
          {
              dstPtr[x] <<= shiftbit;
          }
      }
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

//! template specialisation for shift function of type rgba32
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftLFunc<ito::Rgba32>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
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
   int numMats = src->getNumPlanes();
   int MatNum = 0;

   cv::Mat_<_Tp> * tempMat = NULL;
   _Tp* dstPtr = NULL;
   int sizex = static_cast<int>(src->getSize(src->getDims() - 1));
   int sizey = static_cast<int>(src->getSize(src->getDims() - 2));
   for (int nmat = 0; nmat < numMats; nmat++)
   {
        MatNum = src->seekMat(nmat, numMats);
        //TODO: check if non iterator version is working
        tempMat = static_cast<cv::Mat_<_Tp> *>((src->get_mdata())[MatNum]);

        for (int y = 0; y < sizey; y++)
        {
            dstPtr = (_Tp*)tempMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] >>= shiftbit;
            }
        }
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

//! template specialisation for shift function of type rgba32
/*!
    \throws cv::Exception since shifting is not defined for that input type
*/
template<> RetVal ShiftRFunc<ito::Rgba32>(DataObject * /*src*/, const unsigned char /*shiftbit*/)
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
   int numMats = dObj1->getNumPlanes();
   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const cv::Mat_<_Tp>** dobj1mats = reinterpret_cast<const cv::Mat_<_Tp>**>(dObj1->get_mdata());
   const cv::Mat_<_Tp>** dobj2mats = reinterpret_cast<const cv::Mat_<_Tp>**>(dObj2->get_mdata());
   cv::Mat_<_Tp>** dobjresmats = reinterpret_cast<cv::Mat_<_Tp>**>(dObjRes->get_mdata());

   for (int nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = dObj1->seekMat(nmat, numMats);
      rhsMatNum = dObj2->seekMat(nmat, numMats);
      resMatNum = dObjRes->seekMat(nmat, numMats);
      *(dobjresmats[resMatNum]) = *(dobj1mats[lhsMatNum]) & *(dobj2mats[rhsMatNum]);      
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

//! template specialisation for bitwise and function of type rgba32
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitAndFunc<ito::Rgba32>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
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

    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

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

    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

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
   int numMats = dObj1->getNumPlanes();
   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const cv::Mat_<_Tp>** dobj1mats = reinterpret_cast<const cv::Mat_<_Tp>**>(dObj1->get_mdata());
   const cv::Mat_<_Tp>** dobj2mats = reinterpret_cast<const cv::Mat_<_Tp>**>(dObj2->get_mdata());
   cv::Mat_<_Tp>** dobjresmats = reinterpret_cast<cv::Mat_<_Tp>**>(dObjRes->get_mdata());

   for (int nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = dObj1->seekMat(nmat, numMats);
      rhsMatNum = dObj2->seekMat(nmat, numMats);
      resMatNum = dObjRes->seekMat(nmat, numMats);
      *(dobjresmats[resMatNum]) = *(dobj1mats[lhsMatNum]) | *(dobj2mats[rhsMatNum]); 
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

//! template specialisation for bitwise or function of type rgba32
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitOrFunc<ito::Rgba32>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
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

    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

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

    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

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
   int numMats = dObj1->getNumPlanes();
   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const cv::Mat_<_Tp>** dobj1mats = reinterpret_cast<const cv::Mat_<_Tp>**>(dObj1->get_mdata());
   const cv::Mat_<_Tp>** dobj2mats = reinterpret_cast<const cv::Mat_<_Tp>**>(dObj2->get_mdata());
   cv::Mat_<_Tp>** dobjresmats = reinterpret_cast<cv::Mat_<_Tp>**>(dObjRes->get_mdata());

   for (int nmat = 0; nmat < numMats; nmat++)
   {
      lhsMatNum = dObj1->seekMat(nmat, numMats);
      rhsMatNum = dObj2->seekMat(nmat, numMats);
      resMatNum = dObjRes->seekMat(nmat, numMats);
      *(dobjresmats[resMatNum]) = *(dobj1mats[lhsMatNum]) ^ *(dobj2mats[rhsMatNum]);    
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

//! template specialisation for bitwise xor function of type complex128
/*!
    \throws cv::Exception since this operation is not defined for that input type
*/
template<> RetVal BitXorFunc<ito::Rgba32>(const DataObject * /*dObj1*/, const DataObject * /*dObj2*/, DataObject * /*dObjRes*/)
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

    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

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
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(rhs)

    DataObject result;
    result.m_continuous |= rhs.m_continuous;
    this->copyTo(result, 1);

    result ^= rhs;
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObject DataObject::bitwise_not() const
{
    DataObject result;
    copyTo(result, 1);

    int numMats = result.getNumPlanes();
    cv::Mat *plane;
    for (int nmat = 0; nmat < numMats; nmat++)
    {
        plane = result.get_mdata()[result.seekMat(nmat, numMats)];
        cv::bitwise_not(*plane, *plane);
    }

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
DataObject DataObject::at(const ito::Range &rowRange, const ito::Range &colRange) const
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
    return this->at(ranges);
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
      int numMats = dObj->mdata_size();
      cv::Mat_<_Tp> **mats = reinterpret_cast<cv::Mat_<_Tp>** >(dObj->get_mdata());

      for (int nmat = 0; nmat < numMats; nmat++)
      {
         mats[nmat]->adjustROI(dtop, dbottom, dleft, dright);
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
DataObject DataObject::at(ito::Range *ranges) const
{
    DataObject resMat = *this;

    int *lims = new int[m_dims*2];
    int size;
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
//! addressing method that returns a 1xM data object of the same type than this object with only values that are marked in the given uint8 mask object
/*!
    This method returns a new 1xM data object with the same type than this data object. The M columns are filled with a values
    of this data object whose corresponding mask value is != 0.

    \param mask is a uint8 mask data object with the same size than this object. Values != 0 are valid values in the mask.
    \return new data object with shallow copy of this data object and adjusted ROI with respect to the given ranges
*/
DataObject DataObject::at(const DataObject &mask) const
{
    if (mask.getDims() != m_dims || (mask.getSize() != getSize()))
    {
        cv::error(cv::Exception(CV_StsAssert,"The mask object must have the same size than this data object","", __FILE__, __LINE__));
    }
    else if (mask.getType() != ito::tUInt8)
    {
        cv::error(cv::Exception(CV_StsAssert,"The mask object must have type uint8","", __FILE__, __LINE__));
    }

    int numPlanes = getNumPlanes();
    int sizeY = m_size[m_dims - 2];
    int sizeX = m_size[m_dims - 1];
    int counts = 0;
    const cv::Mat *maskMat;
    const cv::Mat *mat;

    for (int i = 0; i < numPlanes; ++i)
    {
        maskMat = mask.getCvPlaneMat(i);
        counts += cv::countNonZero(*maskMat);
    }
    
    ito::DataObject result(1, counts, m_type);
    copyTagMapTo(result);

    ito::uint8 *dataRow = result.rowPtr(0,0);
    const ito::uint8 *maskRow;
    const ito::uint8 *srcRow;
    int es = elemSize();

    for (int i = 0; i < numPlanes; ++i)
    {
        maskMat = mask.getCvPlaneMat(i);
        mat = getCvPlaneMat(i);

        for (int y = 0; y < sizeY; ++y)
        {
            maskRow = maskMat->ptr(y);
            srcRow = mat->ptr(y);

            for (int x = 0; x < sizeX; ++x)
            {
                if (maskRow[x])
                {
                    memcpy(dataRow, srcRow, es);
                    dataRow += es;
                }

                srcRow += es;
            }
        }
    }

    return result;
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
      int numMats = dObj->mdata_size();
      cv::Mat_<_Tp> **mats = reinterpret_cast<cv::Mat_<_Tp>** >(dObj->get_mdata());

      for (int nmat = 0; nmat < numMats; nmat++)
      {
         mats[nmat]->adjustROI(dtop, dbottom, dleft, dright);
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
//    int lim1, lim2;

    for(int n = 0; n < dims; n++)
    {
        //TODO: why lim1 & lim2 are set when they actually aren't used?
//        lim1 = lims[n*2];
//        lim2 = lims[n*2+1];       
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
RetVal DataObject::locateROI(int *wholeSizes, int *offsets) const
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
RetVal DataObject::locateROI(int *lims) const
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
template<typename _Tp> RetVal EyeFunc(const int size, uchar **dstMat)
{
   (*((cv::Mat_<_Tp> *)(*dstMat))) = cv::Mat_<_Tp>::eye(static_cast<int>(size), static_cast<int>(size));

   return 0;
}

typedef RetVal (*tEyeFunc)(const int size, uchar **dstMat);
MAKEFUNCLIST(EyeFunc)

//! sets the matrix of this data object to a two-dimensional eye-matrix of size 1, hence [1]
/*!
    \param type is the desired element data-type
    \return retOk
    \sa ones
*/
RetVal DataObject::eye(const int type)
{
   int sizes[2] = {1, 1};
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
RetVal DataObject::eye(const int size, const int type)
{
   int sizes[2] = {size, size};

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
   int numMats = dObj->getNumPlanes();
   int MatNum = 0;

   cv::Mat_<_Tp> * tempMat = NULL;
   int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
   int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
   for (int nmat = 0; nmat < numMats; nmat++)
   {
        //TODO: check if non iterator version is working
       MatNum = dObj->seekMat(nmat, numMats);
       tempMat = static_cast<cv::Mat_<_Tp> *>((dObj->get_mdata())[MatNum]);

       for (int y = 0; y < sizey; y++)
       {
           _Tp* dstPtr = (_Tp*)tempMat->ptr(y);
           for (int x = 0; x < sizex; x++)
           {
               dstPtr[x] = std::conj(dstPtr[x]);
           }
       }
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
template<> RetVal ConjFunc<ito::Rgba32>(DataObject * /*dObj*/)
{
   cv::error(cv::Exception(CV_StsAssert, "Not defined for input parameter type", "", __FILE__, __LINE__));
   return 0;
}
//! template specialization for data object of type int64. throws cv::Exception, since the data type is not complex.
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
//template<typename _Tp> RetVal RowFunc(const DataObject *dObj, const unsigned int selRow)
//{
//   (*((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))) = (((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))->row(selRow));
//
//   return 0;
//}
//
//typedef RetVal (*tRowFunc)(const DataObject *dObj, const unsigned int selRow);
//MAKEFUNCLIST(RowFunc)

//! high-level method which makes a new header for the specified matrix row and returns it. The underlying data of the new matrix is shared with the original matrix.
/*!
    \param selRow is the specific zero-based row index
    \return new data object
    \throws cv::Exception if dimension is unequal to two.
    \sa RowFunc
*/
DataObject DataObject::row(const int selRow) const
{
   if (m_dims != 2)
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject::row only defined for dims==2","", __FILE__, __LINE__));
   }

   DataObject resMat = *this;
   *(resMat.get_mdata()[0]) = resMat.get_mdata()[0]->row(selRow);
   //fListRowFunc[m_type](&resMat, selRow);
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
/*template<typename _Tp> RetVal ColFunc(const DataObject *dObj, const unsigned int selCol)
{
   (*((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))) = (((cv::Mat_<_Tp> *)(dObj->get_mdata()[0]))->col(selCol));
   return 0;
}

typedef RetVal (*tColFunc)(const DataObject *dObj, const unsigned int selCol);
MAKEFUNCLIST(ColFunc)*/

//! high-level method which makes a new header for the specified matrix column and returns it. The underlying data of the new matrix is shared with the original matrix.
/*!
    \param selCol is the specific zero-based row index
    \return new data object
    \throws cv::Exception if dimension is unequal to two.
    \sa ColFunc
*/
DataObject DataObject::col(const int selCol) const
{
   if (m_dims != 2)
   {
      cv::error(cv::Exception(CV_StsAssert,"DataObject::col only defined for dims==2","", __FILE__, __LINE__));
   }

   DataObject resMat = *this;
   *(resMat.get_mdata()[0]) = resMat.get_mdata()[0]->col(selCol);
   //fListColFunc[m_type](&resMat, selCol);
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
   int numMats = src1->getNumPlanes();

   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const _Tp* src1RowPtr;
   const _Tp* src2RowPtr;
   _Tp* resRowPtr;
   const cv::Mat_<_Tp>* srcMat1 = NULL;
   const cv::Mat_<_Tp>* srcMat2 = NULL;
   cv::Mat_<_Tp>* dstMat = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
        cv::Mat_<_Tp> tempMat;
        lhsMatNum = src1->seekMat(nmat, numMats);
        rhsMatNum = src2->seekMat(nmat, numMats);
        resMatNum = res->seekMat(nmat, numMats);
        srcMat1 = static_cast<const cv::Mat_<_Tp> *>(src1->get_mdata()[lhsMatNum]);       
        srcMat2 = static_cast<const cv::Mat_<_Tp> *>(src2->get_mdata()[rhsMatNum]);
        dstMat = static_cast<cv::Mat_<_Tp> *>(res->get_mdata()[resMatNum]);

        for(int i = 0; i < srcMat1->rows; i++)
        {
            src1RowPtr = (const _Tp*)srcMat1->ptr(i);
            src2RowPtr = (const _Tp*)srcMat2->ptr(i);
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

//! high-level method which does a element-wise multiplication of elements in this matrix with elements in the second matrix.
/*!
    The result is returned as new data object with the same type and size than this object. The axis scale, offset, description and
    unit values are copied from this object. Tags are copied from this object, too.
    Optionally the multiplication can be scaled by a scaling factor, which is set to one by default.

    \param &mat2 is the second source matrix
    \param scale is the scaling factor (default: 1.0)
    \return result matrix
    \sa DivFunc
*/
DataObject DataObject::mul(const DataObject &mat2, const double scale) const
{
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(mat2)

    unsigned char continuous = 0;
    DataObject result(m_dims,m_size,m_type,continuous);    
    copyAxisTagsTo(result);
    copyTagMapTo(result);

    fListMulFunc[m_type](this, &mat2, &result, scale);

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
    \return retOk
*/
template<typename _Tp> RetVal DivFunc(const DataObject *src1, const DataObject *src2, DataObject *res)
{
    //the transpose flag of this matrix already is evaluated if src2 is not transposed
   int numMats = src1->getNumPlanes();

   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const _Tp* src1RowPtr;
   const _Tp* src2RowPtr;
   _Tp* resRowPtr;
   _Tp nanNumber;
   _Tp infNumber;
   _Tp epsilon;
   _Tp zero = static_cast<_Tp>(0);
   const cv::Mat_<_Tp>* srcMat1 = NULL;
   cv::Mat_<_Tp>* dstMat = NULL;
   const cv::Mat_<_Tp>* srcMat2 = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
        cv::Mat_<_Tp> tempMat;
        lhsMatNum = src1->seekMat(nmat, numMats);
        rhsMatNum = src2->seekMat(nmat, numMats);
        resMatNum = res->seekMat(nmat, numMats);
        srcMat1 = static_cast<const cv::Mat_<_Tp> *>(src1->get_mdata()[lhsMatNum]);        
        srcMat2 = static_cast<const cv::Mat_<_Tp> *>(src2->get_mdata()[rhsMatNum]);     
        dstMat = static_cast<cv::Mat_<_Tp> *>(res->get_mdata()[resMatNum]);

        if(std::numeric_limits<_Tp>::has_quiet_NaN)
        {
            nanNumber = std::numeric_limits<_Tp>::quiet_NaN();
            infNumber = std::numeric_limits<_Tp>::infinity();
            epsilon = std::numeric_limits<_Tp>::epsilon();

            for(int i = 0; i < srcMat1->rows; i++)
            {
                src1RowPtr = (const _Tp*)srcMat1->ptr(i);
                src2RowPtr = (const _Tp*)srcMat2->ptr(i);
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
                src1RowPtr = (const _Tp*)srcMat1->ptr(i);
                src2RowPtr = (const _Tp*)srcMat2->ptr(i);
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
    \return retOk
*/
template<> RetVal DivFunc<Rgba32>(const DataObject *src1, const DataObject *src2, DataObject *res)
{
    //the transpose flag of this matrix already is evaluated if src2 is not transposed
   int numMats = src1->getNumPlanes();

   int lhsMatNum = 0;
   int rhsMatNum = 0;
   int resMatNum = 0;

   const Rgba32* src1RowPtr;
   const Rgba32* src2RowPtr;
   ito::Rgba32* resRowPtr;
   const cv::Mat_<Rgba32>* srcMat1 = NULL;
   cv::Mat_<Rgba32>* dstMat = NULL;
   const cv::Mat_<Rgba32>* srcMat2 = NULL;

   for (int nmat = 0; nmat < numMats; nmat++)
   {
        cv::Mat_<Rgba32> tempMat;
        lhsMatNum = src1->seekMat(nmat, numMats);
        rhsMatNum = src2->seekMat(nmat, numMats);
        resMatNum = res->seekMat(nmat, numMats);
        srcMat1 = static_cast<const cv::Mat_<Rgba32> *>(src1->get_mdata()[lhsMatNum]);        
        srcMat2 = static_cast<const cv::Mat_<Rgba32> *>(src2->get_mdata()[rhsMatNum]);     
        dstMat = static_cast<cv::Mat_<Rgba32> *>(res->get_mdata()[resMatNum]);

        for(int i = 0; i < srcMat1->rows; i++)
        {
            src1RowPtr = srcMat1->ptr<Rgba32>(i);
            src2RowPtr = srcMat2->ptr<Rgba32>(i);
            resRowPtr = dstMat->ptr<Rgba32>(i);

            for(int j = 0; j < srcMat1->cols; j++)
            {
                resRowPtr[j] = src1RowPtr[j] / src2RowPtr[j];
            }
        }
   }

   return 0;
}

typedef RetVal (*tDivFunc)(const DataObject *src1, const DataObject *src2, DataObject *res);
MAKEFUNCLIST(DivFunc)

//! high-level method which does a element-wise division of elements in this matrix by elements in second source matrix.
/*!
    The result is returned as new data object with the same type and size than this object. The axis scale, offset, description and
    unit values are copied from this object. Tags are copied from this object, too.

    \param &mat2 is the second source matrix
    \param scale is the scaling factor (default: 1.0)
    \return result matrix
    \sa DivFunc
*/
DataObject DataObject::div(const DataObject &mat2, const double /*scale*/) const
{
    CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(mat2)

    unsigned char continuous = 0;
    DataObject result(m_dims,m_size,m_type,continuous);    
    copyAxisTagsTo(result);
    copyTagMapTo(result);
                     
    fListDivFunc[m_type](this, &mat2, &result);

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObject DataObject::pow(const ito::float64 &power)
{
	DataObject result(m_dims, getSize(), m_type);
	pow(power, result);
	return result;
	
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObject::pow(const ito::float64 &power, DataObject &dst)
{
	if (dst.getDims() == 0)
	{
		dst = DataObject(m_dims, getSize(), m_type);
	}

	CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(dst)

	int ipower = cvRound(power);
	bool is_ipower = fabs(ipower - power) < DBL_EPSILON;
	int num_planes = getNumPlanes();
	const cv::Mat* mat_src;
	cv::Mat* mat_dest;

	if (is_ipower)
	{
		switch (m_type)
		{
		case ito::tUInt8:
		case ito::tInt8:
		case ito::tUInt16:
		case ito::tInt16:
		case ito::tUInt32:
		case ito::tInt32:
		case ito::tFloat32:
		case ito::tFloat64:
		{
			for (int i = 0; i < num_planes; ++i)
			{
				mat_src = get_mdata()[seekMat(i, num_planes)];
				mat_dest = dst.get_mdata()[dst.seekMat(i, num_planes)];
				cv::pow(*mat_src, power, *mat_dest);
			}
		}
		break;
		default:
			cv::error(cv::Exception(CV_StsAssert, "an integer power requires a real-typed data object.", "", __FILE__, __LINE__));
		}
	}
	else
	{
		switch (m_type)
		{
		case ito::tFloat32:
		case ito::tFloat64:
		{
			for (int i = 0; i < num_planes; ++i)
			{
				mat_src = get_mdata()[seekMat(i, num_planes)];
				mat_dest = dst.get_mdata()[dst.seekMat(i, num_planes)];
				cv::pow(*mat_src, power, *mat_dest);
			}
		}
		break;
		default:
			cv::error(cv::Exception(CV_StsAssert, "an non-integer power requires a data object of type float32 or float64.", "", __FILE__, __LINE__));
		}
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObject DataObject::sqrt()
{
	DataObject result(m_dims, getSize(), m_type);
	sqrt(result);
	return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DataObject::sqrt(DataObject &dst)
{
	if (dst.getDims() == 0)
	{
		dst = DataObject(m_dims, getSize(), m_type);
	}

	CHECK_SAME_TYPE_AND_NUM_PLANES_AND_PLANE_SIZE(dst)

	int num_planes = getNumPlanes();
	const cv::Mat* mat_src;
	cv::Mat* mat_dest;

	switch (m_type)
	{
	case ito::tFloat32:
	case ito::tFloat64:
	{
		for (int i = 0; i < num_planes; ++i)
		{
			mat_src = get_mdata()[seekMat(i, num_planes)];
			mat_dest = dst.get_mdata()[dst.seekMat(i, num_planes)];
			cv::pow(*mat_src, 0.5, *mat_dest);
		}
	}
	break;
	default:
		cv::error(cv::Exception(CV_StsAssert, "sqrt requires a data object of type float32 or float64.", "", __FILE__, __LINE__));
	}
}

//----------------------------------------------------------------------------------------------------------------------------------
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

    int numMats = getNumPlanes();

    if (numMats == 0)
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject to squeeze must contain at least one value (has no planes).",  "", __FILE__, __LINE__));
    }

    unsigned char newDimensions = 0;
    int *newSizes = new int[m_dims];
    int *axesMap = new int[m_dims]; //axesMap[0] gives the axes index in this object of the 0th axis in resObj. axesMap[idx >= newDimensions] is invalid
    int counter = 0;
    bool test;
    bool planesUnchanged = true;
    int shapeOfLastDim = 0;

    for(int i = 0; i < m_dims - 2 ;i++)
    {
        if( getSize(i) > 1 )
        {
            axesMap[counter] = i;
            newSizes[counter] = getSize(i);
            newDimensions++;
            counter++;
        }
    }

    //last two dimensions
    if ( getSize(m_dims - 2) > 1 )
    {
        axesMap[counter] = m_dims - 2;
        newSizes[counter] = getSize(m_dims - 2);
        newDimensions++;
        counter++;
    }
    else
    {
        planesUnchanged = false;
    }
    
    if ( getSize(m_dims - 1) > 1 )
    {
        axesMap[counter] = m_dims - 1;
        newSizes[counter] = getSize(m_dims - 1);
        shapeOfLastDim = newSizes[counter];
        newDimensions++;
        counter++;
    }
    else
    {
        planesUnchanged = false;
    }

    if (newDimensions == 0) //1x1x1x... object cannot be totally squeezed. The last two dimensions will be kept.
    {
        newDimensions = 2;
        axesMap[0] = m_dims - 2;
        newSizes[0] = getSize(m_dims - 2);
        axesMap[1] = m_dims - 1;
        newSizes[1] = getSize(m_dims - 1);
    }
    else if (newDimensions == 1) //a 1-dim object cannot be built. Therefore
    {
        newDimensions = 2;
        if (axesMap[0] == (m_dims - 1)) //the last dimensions was > 1, add the second to last dimension
        {
            axesMap[1] = axesMap[0];
            newSizes[1] = newSizes[0];
            axesMap[0] = m_dims - 2;
            newSizes[0] = getSize(m_dims - 2);
        }
        else //the last dimension was 1, nevertheless add it...
        {
            axesMap[1] = m_dims - 1;
            newSizes[1] = getSize(m_dims - 1);
        }
    }

    DataObject resObj;
    
    if (planesUnchanged)
    {
        //shallow copy without change in any plane
        cv::Mat * planes = new cv::Mat[numMats];
        for(int i = 0 ; i < numMats ; i++)
        {
            planes[i] = *( (cv::Mat*)(m_data[this->seekMat(i)]) );
        }

        resObj = DataObject(newDimensions, newSizes, m_type, planes, static_cast<unsigned int>(numMats));

        delete[] planes;
    }
    else
    {
        //the dimension within a plane changed, therefore an element-wise deep copy is required.
        //this is done by an iterator (yes, there are faster ways, but this is a comfortable one)
        resObj = DataObject(newDimensions, newSizes, m_type);
        ito::DObjIterator resIt = resObj.begin();
        DObjConstIterator srcIt = constBegin();
        int es = elemSize();

        if (shapeOfLastDim > 0)
        {
            //copy line by line since the last dimension is not squeezed
            es *= shapeOfLastDim;

            while (resIt != resObj.end() && srcIt != constEnd())
            {
                memcpy(*resIt, *srcIt, es);
                resIt += shapeOfLastDim;
                srcIt += shapeOfLastDim;
            }
        }
        else
        {
            while (resIt != resObj.end() && srcIt != constEnd())
            {
                memcpy(*resIt, *srcIt, es);
                resIt++;
                srcIt++;
            }
        }
    }

    for (int i = 0; i < newDimensions; ++i)
    {
        resObj.setAxisDescription(i, this->getAxisDescription(axesMap[i], test));
        resObj.setAxisUnit(i, this->getAxisUnit(axesMap[i], test));
        resObj.setAxisOffset(i,this->getAxisOffset(axesMap[i]));
        resObj.setAxisScale(i, this->getAxisScale(axesMap[i]));
    }

    //copy tags
    copyTagMapTo(resObj);

    //copy rotation matrix
    const double *rot = m_pDataObjectTags->m_rotMatrix;
    resObj.setXYRotationalMatrix(rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);

    //copy value description and unit
    resObj.setValueDescription(getValueDescription());
    resObj.setValueUnit(getValueUnit());

    delete[] newSizes;
    delete[] axesMap;
    
    newSizes = NULL;

    return resObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
DataObject DataObject::reshape(int newDims, const int *newSizes) const
{
    int newTotal = 1;
    for (int i = 0; i < newDims; ++i)
    {
        newTotal *= newSizes[i];
    }

    if (getTotal() != newTotal)
    {
        cv::error(cv::Exception(CV_StsAssert,"Total size of new dataObject must be unchanged.",  "", __FILE__, __LINE__));
    }
    else if (newDims < 2)
    {
        cv::error(cv::Exception(CV_StsAssert, "New new object must have at least two dimensions (e.g. 1xM or Mx1).", "", __FILE__, __LINE__));
    }

    if(m_dims <= 0)
    {
        return DataObject();
    }
    else if(m_dims == 1)
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject to reshape may not have dimension = 1",  "", __FILE__, __LINE__));
    }

    int numMats = getNumPlanes();

    if (numMats == 0)
    {
        cv::error(cv::Exception(CV_StsAssert,"DataObject to squeeze must contain at least one value (has no planes).",  "", __FILE__, __LINE__));
    }

    int counter = 0;
    bool test;
    bool planesUnchanged = false;
    int shapeOfLastDim = 0;

    if (getSize(m_dims - 1) == newSizes[newDims - 1])
    {
        if (getSize(m_dims - 2) == newSizes[newDims - 2])
        {
            planesUnchanged = true;
        }
        else
        {
            shapeOfLastDim = newSizes[newDims - 1];
        }
    }

    DataObject resObj;
    
    if (planesUnchanged)
    {
        //shallow copy without change in any plane
        cv::Mat * planes = new cv::Mat[numMats];
        for(int i = 0 ; i < numMats ; i++)
        {
            planes[i] = *( (cv::Mat*)(m_data[this->seekMat(i)]) );
        }

        resObj = DataObject(newDims, newSizes, m_type, planes, static_cast<unsigned int>(numMats));

        delete[] planes;
    }
    else
    {
        //the dimension within a plane changed, therefore an element-wise deep copy is required.
        //this is done by an iterator (yes, there are faster ways, but this is a comfortable one)
        resObj = DataObject(newDims, newSizes, m_type);
        ito::DObjIterator resIt = resObj.begin();
        DObjConstIterator srcIt = constBegin();
        int es = elemSize();

        if (shapeOfLastDim > 0)
        {
            //copy line by line since the last dimension is not squeezed
            es *= shapeOfLastDim;

            while (resIt != resObj.end() && srcIt != constEnd())
            {
                memcpy(*resIt, *srcIt, es);
                resIt += shapeOfLastDim;
                srcIt += shapeOfLastDim;
            }
        }
        else
        {
            while (resIt != resObj.end() && srcIt != constEnd())
            {
                memcpy(*resIt, *srcIt, es);
                resIt++;
                srcIt++;
            }
        }
    }

    //copy axis description, unit, offset, scale... for all axes with same size beginning from the last axis index.
    //the copy operation is stopped if the first dimension with different sizes is detected.
    for (int i = 0; i < std::min(newDims, m_dims); ++i)
    {
        if (newSizes[newDims - i] == getSize(m_dims - i))
        {
            resObj.setAxisDescription(newDims - i, this->getAxisDescription(m_dims - i, test));
            resObj.setAxisUnit(newDims - i, this->getAxisUnit(m_dims - i, test));
            resObj.setAxisOffset(newDims - i,this->getAxisOffset(m_dims - i));
            resObj.setAxisScale(newDims - i, this->getAxisScale(m_dims - i));
        }
        else
        {
            break;
        }
    }

    //copy tags
    copyTagMapTo(resObj);

    //copy rotation matrix
    const double *rot = m_pDataObjectTags->m_rotMatrix;
    resObj.setXYRotationalMatrix(rot[0], rot[1], rot[2], rot[3], rot[4], rot[5], rot[6], rot[7], rot[8]);

    //copy value description and unit
    resObj.setValueDescription(getValueDescription());
    resObj.setValueUnit(getValueUnit());

    return resObj;
}



//----------------------------------------------------------------------------------------------------------------------------------
RetVal DataObject::copyFromData2DInternal(const uchar* src, const int sizeOfElem, const int sizeX, const int sizeY)
{
    ito::RetVal retval(ito::retOk);

    if ((getNumPlanes() != 1) || (getSize(getDims() - 1) != sizeX) || (getSize(getDims() - 2) != sizeY))
    {
        retval = RetVal(ito::retError,0,"Error in copyFromData2D. Size of Buffer unequal size of DataObject");
        return retval;
    }
    
    //retval = checkType(src); // This is bullshit because this type is anytype and src is always uchar!!!!
    //if (retval != ito::retOk)
    //   return retval;

    cv::Mat *cvMat = get_mdata()[this->seekMat(0)];

    if (cvMat->elemSize() !=  sizeOfElem)
        return retval;

    if (cvMat->isContinuous())
    {
        memcpy(cvMat->ptr(0), src, sizeX * sizeY * sizeOfElem);
    }
    else
    {
        for (int y = 0; y < sizeY; y++)
        {
            memcpy(cvMat->ptr(y), src + y * sizeX * sizeOfElem, sizeX * sizeOfElem);
        }
    }

    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal DataObject::copyFromData2DInternal(const uchar* src, const int sizeOfElem, const int sizeX, const int x0, const int y0, const int width, const int height)
{
    ito::RetVal retval;

    if ((getNumPlanes() != 1) || (getSize(getDims() - 1) != width) || (getSize(getDims() - 2) != height))
    {
        retval = RetVal(ito::retError,0,"Error in copyFromData2D. Size of Buffer unequal size of DataObject");
        return retval;
    }
    //retval = checkType(src); // This is bullshit because this type is anytype and src is always uchar!!!!
    //if (retval != ito::retOk)
    //    return retval;

    cv::Mat *cvMat = get_mdata()[this->seekMat(0)];

    if (cvMat->elemSize() !=  sizeOfElem)
        return retval;
    for (int y = 0; y < height; y++)
    {
        memcpy(cvMat->ptr(y), src + ((y0 + y) * sizeX + x0) * sizeOfElem , width * sizeOfElem);
    }

    return retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> RetVal GrayScaleCastFunc(const DataObject *dObj, DataObject *resObj, double alpha = 1.0)
{
    int numMats = dObj->getNumPlanes();

    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    const cv::Mat * srcMat = NULL;
    cv::Mat * dstMat = NULL;
    const ito::Rgba32* srcPtr;
    _Tp* dstPtr;

    if (alpha == 1.0)
    {
        for (int nmat = 0; nmat < numMats; nmat++)
        {
            srcMat = dObj->getCvPlaneMat(nmat);
            dstMat = resObj->getCvPlaneMat(nmat);
for (int y = 0; y < sizey; y++)
{
    dstPtr = dstMat->ptr<_Tp>(y);
    srcPtr = srcMat->ptr<const ito::Rgba32>(y);
    for (int x = 0; x < sizex; x++)
    {
        dstPtr[x] = cv::saturate_cast<_Tp>(0.299 * srcPtr[x].r + 0.587 * srcPtr[x].g + 0.114 * srcPtr[x].b);
    }
}
        }
    }
    else
    {
        for (int nmat = 0; nmat < numMats; nmat++)
        {
            srcMat = dObj->getCvPlaneMat(nmat);
            dstMat = resObj->getCvPlaneMat(nmat);
            for (int y = 0; y < sizey; y++)
            {
                dstPtr = dstMat->ptr<_Tp>(y);
                srcPtr = srcMat->ptr<const ito::Rgba32>(y);
                for (int x = 0; x < sizex; x++)
                {
                    dstPtr[x] = cv::saturate_cast<_Tp>(alpha * (0.299 * srcPtr[x].r + 0.587 * srcPtr[x].g + 0.114 * srcPtr[x].b));
                }
            }
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! converts a color image (rgba32) to a gray-scale image
/*!
    usage: res = static_cast<ito::float32>(sourceDataObject)

    \throws cv::Exception if cast failed, e.g. if cast not possible or types unknown
    \return cast data object
    \sa convertTo, CastFunc
    */
DataObject DataObject::toGray(const int destinationType /*= ito::tUInt8*/) const
{
    if (this->m_type != ito::tRGBA32)
    {
        cv::error(cv::Exception(CV_StsAssert, "data type of dataObject must be rgba32.", "", __FILE__, __LINE__));
    }
    else if (destinationType == ito::tComplex64 || destinationType == ito::tComplex128)
    {
        cv::error(cv::Exception(CV_StsAssert, "destinationType must be real.", "", __FILE__, __LINE__));
    }

    DataObject resObj = DataObject(m_dims, m_size, destinationType);

    switch (destinationType)
    {
    case ito::tInt8:
        GrayScaleCastFunc<int8>(this, &resObj);
        break;

    case ito::tUInt8:
        GrayScaleCastFunc<uint8>(this, &resObj);
        break;

    case ito::tInt16:
        GrayScaleCastFunc<int16>(this, &resObj);
        break;

    case ito::tUInt16:
        GrayScaleCastFunc<uint16>(this, &resObj);
        break;

    case ito::tInt32:
        GrayScaleCastFunc<uint32>(this, &resObj);
        break;

    case ito::tFloat32:
        GrayScaleCastFunc<float32>(this, &resObj);
        break;

    case ito::tFloat64:
        GrayScaleCastFunc<float64>(this, &resObj);
        break;

    default:
        cv::error(cv::Exception(CV_StsAssert, "destinationType must be real.", "", __FILE__, __LINE__));
        break;
    }

    copyTagMapTo(resObj);
    copyAxisTagsTo(resObj);

    return resObj;
}
//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> void extractColor(const DataObject *dObj, DataObject &resObj, const char *color, const int &type)
{
    int numChannels = strlen(color);

    switch (numChannels)
    {
    case 1:
        resObj = DataObject(dObj->getDims(), dObj->getSize(), type);
        break;
    case 0:
        return;
        break;
    default:
        {
        int *sizes = new int[dObj->getDims() + 1];
        for (int i = 0; i < dObj->getDims(); ++i)
        {
            sizes[i + 1] = dObj->getSize(i);
        }
        sizes[0] = numChannels;
        resObj = DataObject(dObj->getDims() + 1, sizes, type); 
        delete sizes;
        break;    
        }
    }

    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));

    const cv::Mat * srcMat = NULL;
    cv::Mat * dstMat = NULL;
    const ito::Rgba32* srcPtr;
    _Tp* dstPtr;

    int numMats = dObj->getNumPlanes();

    for (int channel = 0; channel < numChannels; ++channel)
    {
        if (color[channel] == 'b')
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                srcMat = dObj->getCvPlaneMat(nmat);
                dstMat = resObj.getCvPlaneMat(channel * numMats + nmat);
                for (int y = 0; y < sizey; ++y)
                {
                    dstPtr = dstMat->ptr<_Tp>(y);
                    srcPtr = srcMat->ptr<const ito::Rgba32>(y);
                    for (int x = 0; x < sizex; ++x)
                    {
                        dstPtr[x] = cv::saturate_cast<_Tp>(srcPtr[x].b);
                    }
                }
            }
        }
        else if (color[channel] == 'r')
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                srcMat = dObj->getCvPlaneMat(nmat);
                dstMat = resObj.getCvPlaneMat(channel * numMats + nmat);
                for (int y = 0; y < sizey; ++y)
                {
                    dstPtr = dstMat->ptr<_Tp>(y);
                    srcPtr = srcMat->ptr<const ito::Rgba32>(y);
                    for (int x = 0; x < sizex; ++x)
                    {
                        dstPtr[x] = cv::saturate_cast<_Tp>(srcPtr[x].r);
                    }
                }
            }
        }
        else if (color[channel] == 'g')
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                srcMat = dObj->getCvPlaneMat(nmat);
                dstMat = resObj.getCvPlaneMat(channel * numMats + nmat);
                for (int y = 0; y < sizey; y++)
                {
                    dstPtr = dstMat->ptr<_Tp>(y);
                    srcPtr = srcMat->ptr<const ito::Rgba32>(y);
                    for (int x = 0; x < sizex; x++)
                    {
                        dstPtr[x] = cv::saturate_cast<_Tp>(srcPtr[x].g);
                    }
                }
            }
        }
        else if (color[channel] == 'a')
        {
            for (int nmat = 0; nmat < numMats; nmat++)
            {
                srcMat = dObj->getCvPlaneMat(nmat);
                dstMat = resObj.getCvPlaneMat(channel * numMats + nmat);
                for (int y = 0; y < sizey; y++)
                {
                    dstPtr = dstMat->ptr<_Tp>(y);
                    srcPtr = srcMat->ptr<const ito::Rgba32>(y);
                    for (int x = 0; x < sizex; x++)
                    {
                        dstPtr[x] = cv::saturate_cast<_Tp>(srcPtr[x].a);
                    }
                }
            }
        }
    }
}
//----------------------------------------------------------------------------------------------------------------------------------
//! returns a color channel of a color image (rgba32)
/*!
\throws cv::Exception if cast failed, e.g. if cast not possible or types unknown
\return data object
*/
DataObject DataObject::splitColor(const char* destinationColor, const int& dtype /*= ito::tUInt8*/) const
{
    if (this->m_type != ito::tRGBA32)
    {
        cv::error(cv::Exception(CV_StsAssert, "data type of dataObject must be rgba32.", "", __FILE__, __LINE__));
    }

    if (destinationColor[0] != 'b' && destinationColor[0] != 'r' \
        && destinationColor[0] != 'g' && destinationColor[0] != 'a')
    {
        cv::error(cv::Exception(CV_StsAssert, "unknown color.", "", __FILE__, __LINE__));
    }

    DataObject resObj;
    
    switch (dtype)
    {
    case ito::tUInt8:
        extractColor<uint8>(this, resObj, destinationColor, dtype);
        break;

    case ito::tInt16:
        extractColor<int16>(this, resObj, destinationColor, dtype);
        break;

    case ito::tUInt16:
        extractColor<uint16>(this, resObj, destinationColor, dtype);
        break;

    case ito::tInt32:
        extractColor<uint32>(this, resObj, destinationColor, dtype);
        break;

    case ito::tFloat32:
        extractColor<float32>(this, resObj, destinationColor, dtype);
        break;

    case ito::tFloat64:
        extractColor<float64>(this, resObj, destinationColor, dtype);
        break;

    default:
        cv::error(cv::Exception(CV_StsAssert, "destinationType must be real.", "", __FILE__, __LINE__));
        break;
    }
  


    copyTagMapTo(resObj);
    copyAxisTagsTo(resObj);

    return resObj;

}

//----------------------------------------------------------------------------------------------------------------------------------
//! low-level templated method to cast each element of source matrix to another type.
/*!
    The result is stored in the result matrix. Optionally a scaling and offsetting is possible.

    \param *srcObj is the source data object
    \param *resObj is the result data object
    \param alpha is the scaling factor (default 1.0)
    \param beta is the offset value (default 0.0)
    \return 0
    \throws cv::Exception if cast failed
    \sa cv::saturate_cast
*/
template<typename _TSrc, typename _TDest> RetVal CastFunc(const DataObject *srcObj, DataObject *resObj, double alpha, double beta)
{
   int numMats = srcObj->getNumPlanes();
   int resTmat = 0;
   int srcTmat = 0;

   int sizex = srcObj->getSize(srcObj->getDims() - 1);
   int sizey = srcObj->getSize(srcObj->getDims() - 2);
   const cv::Mat* srcMat = NULL;
   cv::Mat* dstMat = NULL;
   const _TSrc* srcPtr = NULL;
   _TDest* dstPtr = NULL;

   if(alpha == 1.0 && beta == 0.0)
   {
       for (int nmat = 0; nmat < numMats; ++nmat)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = srcObj->seekMat(nmat, numMats);
          srcMat = (srcObj->get_mdata())[srcTmat];
          dstMat = (resObj->get_mdata())[resTmat];
          for (int y = 0; y < sizey; ++y)
          {
              dstPtr = dstMat->ptr<_TDest>(y);
              srcPtr = srcMat->ptr<const _TSrc>(y);
              for (int x = 0; x < sizex; ++x)
              {
                  dstPtr[x] = cv::saturate_cast<_TDest>(srcPtr[x]);
              }
          }
       }
   }
   else
   {
       for (int nmat = 0; nmat < numMats; ++nmat)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = srcObj->seekMat(nmat, numMats);
          srcMat = (srcObj->get_mdata())[srcTmat];
          dstMat = (resObj->get_mdata())[resTmat];
          for (int y = 0; y < sizey; ++y)
          {
              dstPtr = dstMat->ptr<_TDest>(y);
              srcPtr = srcMat->ptr<const _TSrc>(y);
              for (int x = 0; x < sizex; ++x)
              {
                  dstPtr[x] = cv::saturate_cast<_TDest>(srcPtr[x] * alpha + beta);
              }
          }
       }
   }

   return ito::retOk;
}

template<typename _TDest> RetVal CastFuncFromComplex64(const DataObject *srcObj, DataObject *resObj, double alpha, double beta)
{
   int numMats = srcObj->getNumPlanes();
   int resTmat = 0;
   int srcTmat = 0;

   int sizex = srcObj->getSize(srcObj->getDims() - 1);
   int sizey = srcObj->getSize(srcObj->getDims() - 2);
   const cv::Mat* srcMat = NULL;
   cv::Mat* dstMat = NULL;
   const ito::complex64* srcPtr = NULL;
   _TDest* dstPtr = NULL;

   if(alpha == 1.0 && beta == 0.0)
   {
       for (int nmat = 0; nmat < numMats; ++nmat)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = srcObj->seekMat(nmat, numMats);
          srcMat = (srcObj->get_mdata())[srcTmat];
          dstMat = (resObj->get_mdata())[resTmat];
          for (int y = 0; y < sizey; ++y)
          {
              dstPtr = dstMat->ptr<_TDest>(y);
              srcPtr = srcMat->ptr<const ito::complex64>(y);
              for (int x = 0; x < sizex; ++x)
              {
                  dstPtr[x] = cv::saturate_cast<_TDest>(srcPtr[x]);
              }
          }
       }
   }
   else
   {
       ito::complex64 alpha2(cv::saturate_cast<ito::float32>(alpha), 0.0);
       ito::complex64 beta2(cv::saturate_cast<ito::float32>(beta), 0.0);

       for (int nmat = 0; nmat < numMats; ++nmat)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = srcObj->seekMat(nmat, numMats);
          srcMat = (srcObj->get_mdata())[srcTmat];
          dstMat = (resObj->get_mdata())[resTmat];
          for (int y = 0; y < sizey; ++y)
          {
              dstPtr = dstMat->ptr<_TDest>(y);
              srcPtr = srcMat->ptr<const ito::complex64>(y);
              for (int x = 0; x < sizex; ++x)
              {
                  dstPtr[x] = cv::saturate_cast<_TDest>(srcPtr[x] * alpha2 + beta2);
              }
          }
       }
   }

   return ito::retOk;
}

template<typename _TDest> RetVal CastFuncFromComplex128(const DataObject *srcObj, DataObject *resObj, double alpha, double beta)
{
   int numMats = srcObj->getNumPlanes();
   int resTmat = 0;
   int srcTmat = 0;

   int sizex = srcObj->getSize(srcObj->getDims() - 1);
   int sizey = srcObj->getSize(srcObj->getDims() - 2);
   const cv::Mat* srcMat = NULL;
   cv::Mat* dstMat = NULL;
   const ito::complex128* srcPtr = NULL;
   _TDest* dstPtr = NULL;

   if(alpha == 1.0 && beta == 0.0)
   {
       for (int nmat = 0; nmat < numMats; ++nmat)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = srcObj->seekMat(nmat, numMats);
          srcMat = (srcObj->get_mdata())[srcTmat];
          dstMat = (resObj->get_mdata())[resTmat];
          for (int y = 0; y < sizey; ++y)
          {
              dstPtr = dstMat->ptr<_TDest>(y);
              srcPtr = srcMat->ptr<const ito::complex128>(y);
              for (int x = 0; x < sizex; ++x)
              {
                  dstPtr[x] = cv::saturate_cast<_TDest>(srcPtr[x]);
              }
          }
       }
   }
   else
   {
       ito::complex128 alpha2(cv::saturate_cast<ito::float64>(alpha), 0.0);
       ito::complex128 beta2(cv::saturate_cast<ito::float64>(beta), 0.0);

       for (int nmat = 0; nmat < numMats; ++nmat)
       {
          resTmat = resObj->seekMat(nmat, numMats);
          srcTmat = srcObj->seekMat(nmat, numMats);
          srcMat = (srcObj->get_mdata())[srcTmat];
          dstMat = (resObj->get_mdata())[resTmat];
          for (int y = 0; y < sizey; ++y)
          {
              dstPtr = dstMat->ptr<_TDest>(y);
              srcPtr = srcMat->ptr<const ito::complex128>(y);
              for (int x = 0; x < sizex; ++x)
              {
                  dstPtr[x] = cv::saturate_cast<_TDest>(srcPtr[x] * alpha2 + beta2);
              }
          }
       }
   }

   return ito::retOk;
}

template<typename _TDest> RetVal CastFuncFromRgba32(const DataObject *srcObj, DataObject *resObj, double alpha, double beta)
{
    switch (resObj->getType())
    {
    case ito::tUInt8:
    case ito::tInt8:
    case ito::tUInt16:
    case ito::tInt16:
    case ito::tUInt32:
    case ito::tInt32:
    case ito::tFloat32:
    case ito::tFloat64:
        if (beta != 0.0)
        {
            cv::error(cv::Exception(CV_StsAssert, "beta value != 0.0 not allowed for conversion from rgba32 to real value type (to gray-scale)", "", __FILE__, __LINE__));
        }
        GrayScaleCastFunc<_TDest>(srcObj, resObj, alpha);
        break;
    case ito::tRGBA32:
        {
            int numMats = srcObj->getNumPlanes();
            int resTmat = 0;
            int srcTmat = 0;

            int sizex = srcObj->getSize(srcObj->getDims() - 1);
            int sizey = srcObj->getSize(srcObj->getDims() - 2);
            const cv::Mat* srcMat = NULL;
            cv::Mat* dstMat = NULL;
            const ito::Rgba32* srcPtr = NULL;
            ito::Rgba32* dstPtr = NULL;

            for (int nmat = 0; nmat < numMats; ++nmat)
            {
                resTmat = resObj->seekMat(nmat, numMats);
                srcTmat = srcObj->seekMat(nmat, numMats);
                srcMat = (srcObj->get_mdata())[srcTmat];
                dstMat = (resObj->get_mdata())[resTmat];
                for (int y = 0; y < sizey; ++y)
                {
                    dstPtr = dstMat->ptr<ito::Rgba32>(y);
                    srcPtr = srcMat->ptr<const ito::Rgba32>(y);
                    for (int x = 0; x < sizex; ++x)
                    {
                        dstPtr[x].r = cv::saturate_cast<ito::uint8>((double)srcPtr[x].r * alpha + beta);
                        dstPtr[x].g = cv::saturate_cast<ito::uint8>((double)srcPtr[x].g * alpha + beta);
                        dstPtr[x].b = cv::saturate_cast<ito::uint8>((double)srcPtr[x].b * alpha + beta);
                        dstPtr[x].a = cv::saturate_cast<ito::uint8>((double)srcPtr[x].a * alpha + beta);
                    }
                }
            }
        }
        break;
    default:
        cv::error(cv::Exception(CV_StsAssert, "conversion from ito::Rgba32 to complex data types not supported.", "", __FILE__, __LINE__));
        break;
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! converts data in DataObject lhs to DataObject rhs with a given type
/*!
    Every element of the source data object is copied to the destionation data object by using this transformation<BR>
        elem_destination = static_cast<newType>(elem_source * alpha + beta)

    \param &lhs is the left-hand sided data object, whose data should be converted
    \param &rhs is the destination data object, whose memory is firstly deleted, then newly allocated
    \param dest_type is the type-number of the destination element
    \param alpha scaling factor (default: 1.0)
    \param beta offset value (default: 0.0)
    \return retOk
    \throws cv::Exception(CV_StsAssert) if conversion type is unknown
    \sa convertTo, CastFunc
*/
template<typename _Tp> RetVal ConvertToFunc(const DataObject &lhs, DataObject &rhs, const int dest_type, const double alpha, const double beta)
{
   if (&lhs == &rhs)
   {
         cv::error(cv::Exception(CV_StsAssert, "inplace-conversion of dataObject not possible", "", __FILE__, __LINE__));
   }
   //_Tp is source type

   if(dest_type == lhs.getType() && alpha == 1.0 && beta == 0.0)
   {
       rhs = lhs;
   }
   else
   {
       rhs.freeData();

       switch (dest_type)
       {
          case ito::tInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, int8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, uint8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, int16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, uint16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, int32>(&lhs, &rhs, alpha, beta);
          break;

          //uint32 is not fully supported by OpenCV -> constructor is blocked
          /*case ito::tUInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, uint32>(&lhs, &rhs, alpha, beta);
          break;*/

          case ito::tFloat32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, float32>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tFloat64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, float64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, complex64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex128:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFunc<_Tp, complex128>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tRGBA32:
              rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
              CastFunc<_Tp, Rgba32>(&lhs, &rhs, alpha, beta);
          break;

          default:
             cv::error(cv::Exception(CV_StsAssert, "cast to destination type not defined (e.g. uint32 is not supported)", "", __FILE__, __LINE__));
          break;
       }

        lhs.copyTagMapTo(rhs);   //Deepcopy the tagspace
        lhs.copyAxisTagsTo(rhs); //Deepcopy the tagspace
   }

   return ito::retOk;
}

template<> RetVal ConvertToFunc<ito::complex64>(const DataObject &lhs, DataObject &rhs, const int dest_type, const double alpha, const double beta)
{
   if (&lhs == &rhs)
   {
         cv::error(cv::Exception(CV_StsAssert, "inplace-conversion of dataObject not possible", "", __FILE__, __LINE__));
   }
   //_Tp is source type

   if(dest_type == lhs.getType() && alpha == 1.0 && beta == 0.0)
   {
       rhs = lhs;
   }
   else
   {
       rhs.freeData();

       switch (dest_type)
       {
          case ito::tInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<int8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<uint8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<int16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<uint16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<int32>(&lhs, &rhs, alpha, beta);
          break;

          //uint32 is not fully supported by OpenCV -> constructor is blocked
          /*case ito::tUInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<uint32>(&lhs, &rhs, alpha, beta);
          break;*/

          case ito::tFloat32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<float32>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tFloat64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<float64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<complex64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex128:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex64<complex128>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tRGBA32:
              rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
              CastFuncFromComplex64<Rgba32>(&lhs, &rhs, alpha, beta);
          break;

          default:
             cv::error(cv::Exception(CV_StsAssert, "cast to destination type not defined (e.g. uint32 is not supported)", "", __FILE__, __LINE__));
          break;
       }

        lhs.copyTagMapTo(rhs);   //Deepcopy the tagspace
        lhs.copyAxisTagsTo(rhs); //Deepcopy the tagspace
   }

   return ito::retOk;
}

template<> RetVal ConvertToFunc<ito::complex128>(const DataObject &lhs, DataObject &rhs, const int dest_type, const double alpha, const double beta)
{
   if (&lhs == &rhs)
   {
         cv::error(cv::Exception(CV_StsAssert, "inplace-conversion of dataObject not possible", "", __FILE__, __LINE__));
   }
   //_Tp is source type

   if(dest_type == lhs.getType() && alpha == 1.0 && beta == 0.0)
   {
       rhs = lhs;
   }
   else
   {
       rhs.freeData();

       switch (dest_type)
       {
          case ito::tInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<int8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<uint8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<int16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<uint16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<int32>(&lhs, &rhs, alpha, beta);
          break;

          //uint32 is not fully supported by OpenCV -> constructor is blocked
          /*case ito::tUInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<uint32>(&lhs, &rhs, alpha, beta);
          break;*/

          case ito::tFloat32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<float32>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tFloat64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<float64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<complex64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex128:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromComplex128<complex128>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tRGBA32:
              rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
              CastFuncFromComplex128<Rgba32>(&lhs, &rhs, alpha, beta);
          break;

          default:
             cv::error(cv::Exception(CV_StsAssert, "cast to destination type not defined (e.g. uint32 is not supported)", "", __FILE__, __LINE__));
          break;
       }

        lhs.copyTagMapTo(rhs);   //Deepcopy the tagspace
        lhs.copyAxisTagsTo(rhs); //Deepcopy the tagspace
   }

   return ito::retOk;
}

template<> RetVal ConvertToFunc<ito::Rgba32>(const DataObject &lhs, DataObject &rhs, const int dest_type, const double alpha, const double beta)
{
    if (&lhs == &rhs)
   {
       cv::error(cv::Exception(CV_StsAssert, "inplace-conversion of dataObject not possible", "", __FILE__, __LINE__));
   }
   //_Tp is source type

   if(dest_type == lhs.getType() && alpha == 1.0 && beta == 0.0)
   {
       rhs = lhs;
   }
   else
   {
       rhs.freeData();

       switch (dest_type)
       {
          case ito::tInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<int8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt8:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<uint8>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<int16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tUInt16:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<uint16>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<int32>(&lhs, &rhs, alpha, beta);
          break;

          //uint32 is not fully supported by OpenCV -> constructor is blocked
          /*case ito::tUInt32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<uint32>(&lhs, &rhs, alpha, beta);
          break;*/

          case ito::tFloat32:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<float32>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tFloat64:
             rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
             CastFuncFromRgba32<float64>(&lhs, &rhs, alpha, beta);
          break;

          case ito::tComplex64:
          case ito::tComplex128:
             cv::error(cv::Exception(CV_StsAssert, "cast from rgba32 to complex destination type not supported", "", __FILE__, __LINE__));
          break;

          case ito::tRGBA32:
              rhs.create(lhs.m_dims, lhs.m_size, dest_type, lhs.m_continuous);
              CastFuncFromRgba32<Rgba32>(&lhs, &rhs, alpha, beta);
          break;

          default:
             cv::error(cv::Exception(CV_StsAssert, "cast to destination type not defined (e.g. uint32 is not supported)", "", __FILE__, __LINE__));
          break;
       }

        lhs.copyTagMapTo(rhs);   //Deepcopy the tagspace
        lhs.copyAxisTagsTo(rhs); //Deepcopy the tagspace
   }

   return ito::retOk;
}

typedef RetVal (*tConvertToFunc)(const DataObject &lhs, DataObject &rhs, const int dest_type, const double alpha, const double beta);

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
    dObj->copyTagMapTo(*resObj);
    dObj->copyAxisTagsTo(*resObj);

    int numMats = dObj->getNumPlanes();
    int srcMatNum = 0;
    int dstMatNum = 0;

    const cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    const _CmplxTp* srcPtr = NULL;
    _Tp* dstPtr = NULL;

    for (int nmat = 0; nmat < numMats; nmat++)
    {
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = static_cast<const cv::Mat_<_CmplxTp> *>((dObj->get_mdata())[srcMatNum]);
        dstMat = static_cast<cv::Mat_<_Tp> *>((resObj->get_mdata())[dstMatNum]);

        for (int y = 0; y < sizey; y++)
        {
            dstPtr = (_Tp*)dstMat->ptr(y);
            srcPtr = (const _CmplxTp*)srcMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::abs(srcPtr[x]);
            }
        }
    }
    return ito::retOk;
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
    dObj->copyTagMapTo(*resObj);
    dObj->copyAxisTagsTo(*resObj);

    int numMats = dObj->getNumPlanes();
    int srcMatNum = 0;
    int dstMatNum = 0;

    const cv::Mat* srcMat = NULL;
    cv::Mat* dstMat = NULL;
    _Tp* dstPtr = NULL;
    const _Tp* srcPtr = NULL;
    int sizex = dObj->getSize(dObj->getDims() - 1);
    int sizey = dObj->getSize(dObj->getDims() - 2);
    for (int nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = dObj->get_mdata()[srcMatNum];
        dstMat = resObj->get_mdata()[dstMatNum];

        for (int y = 0; y < sizey; y++)
        {
            dstPtr = dstMat->ptr<_Tp>(y);
            srcPtr = srcMat->ptr<const _Tp>(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::abs(srcPtr[x]);
            }
        }
    }
    return ito::retOk;
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
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX && dObj.getType() < TYPE_OFFSET_RGBA)
    {
        DataObject resObj(dObj.getDims(), dObj.getSize().m_p, ito::convertCmplxTypeToRealType((ito::tDataType)dObj.getType()));
        fListAbsFunc[dObj.getType() - TYPE_OFFSET_COMPLEX](&dObj, &resObj);
        return resObj;
    }
    else
    {
        DataObject resObj;

        switch(dObj.getType())
        {
        case ito::tInt8:
            resObj = ito::DataObject(dObj.getDims(), dObj.getSize().m_p, dObj.getType());
            AbsFuncReal<int8>(&dObj, &resObj);
            break;
		case ito::tUInt8:
            resObj = dObj;
			break;
        case ito::tInt16:
            resObj = ito::DataObject(dObj.getDims(), dObj.getSize().m_p, dObj.getType());
            AbsFuncReal<int16>(&dObj, &resObj);
            break;
		case ito::tUInt16:
            resObj = dObj;
			break;
        case ito::tInt32:
            resObj = ito::DataObject(dObj.getDims(), dObj.getSize().m_p, dObj.getType());
            AbsFuncReal<int32>(&dObj, &resObj);
            break;
		case ito::tUInt32:
            resObj = dObj;
			break;
        case ito::tFloat32:
            resObj = ito::DataObject(dObj.getDims(), dObj.getSize().m_p, dObj.getType());
            AbsFuncReal<ito::float32>(&dObj, &resObj);
            break;
        case ito::tFloat64:
            resObj = ito::DataObject(dObj.getDims(), dObj.getSize().m_p, dObj.getType());
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

    dObj->copyTagMapTo(*resObj);
    dObj->copyAxisTagsTo(*resObj);

    int numMats = dObj->getNumPlanes();
    int srcMatNum = 0;
    int dstMatNum = 0;

    const cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    _Tp* dstPtr = NULL;
    const _CmplxTp* srcPtr = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (int nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = static_cast<const cv::Mat_<_CmplxTp> *>((dObj->get_mdata())[srcMatNum]);
        dstMat = static_cast<cv::Mat_<_Tp> *>((resObj->get_mdata())[dstMatNum]);

        for (int y = 0; y < sizey; y++)
        {
            dstPtr = (_Tp*)dstMat->ptr(y);
            srcPtr = (const _CmplxTp*)srcMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::arg(srcPtr[x]);
            }
        }
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
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX && dObj.getType() < TYPE_OFFSET_RGBA)
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

    dObj->copyTagMapTo(*resObj);
    dObj->copyAxisTagsTo(*resObj);

    int numMats = dObj->getNumPlanes();
    int srcMatNum = 0;
    int dstMatNum = 0;

    const cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    const _CmplxTp* srcPtr = NULL;
    _Tp* dstPtr = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (int nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = static_cast<const cv::Mat_<_CmplxTp> *>(dObj->get_mdata()[srcMatNum]);
        dstMat = static_cast<cv::Mat_<_Tp> *>(resObj->get_mdata()[dstMatNum]);

        for (int y = 0; y < sizey; y++)
        {
            srcPtr = (const _CmplxTp*)srcMat->ptr(y);
            dstPtr = (_Tp*)dstMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::real(srcPtr[x]);
            }
        }
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
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX && dObj.getType() < TYPE_OFFSET_RGBA)
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

    dObj->copyTagMapTo(*resObj);
    dObj->copyAxisTagsTo(*resObj);

    int numMats = dObj->getNumPlanes();
    int srcMatNum = 0;
    int dstMatNum = 0;

    const cv::Mat_<_CmplxTp> * srcMat = NULL;
    cv::Mat_<_Tp> * dstMat = NULL;
    const _CmplxTp* srcPtr = NULL;
    _Tp* dstPtr = NULL;
    int sizex = static_cast<int>(dObj->getSize(dObj->getDims() - 1));
    int sizey = static_cast<int>(dObj->getSize(dObj->getDims() - 2));
    for (int nmat = 0; nmat < numMats; nmat++)
    {
         //TODO: check if non iterator version is working
        srcMatNum = dObj->seekMat(nmat, numMats);
        dstMatNum = resObj->seekMat(nmat, numMats);
        srcMat = static_cast<const cv::Mat_<_CmplxTp> *>((dObj->get_mdata())[srcMatNum]);
        dstMat = static_cast<cv::Mat_<_Tp> *>((resObj->get_mdata())[dstMatNum]);

        for (int y = 0; y < sizey; y++)
        {
            srcPtr = (const _CmplxTp*)srcMat->ptr(y);
            dstPtr = (_Tp*)dstMat->ptr(y);
            for (int x = 0; x < sizex; x++)
            {
                dstPtr[x] = std::imag(srcPtr[x]);
            }
        }
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
    if(dObj.getType() >= TYPE_OFFSET_COMPLEX && dObj.getType() < TYPE_OFFSET_RGBA)
    {
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
    this templated helper function should only be called if dObj is non-continuous. This is already checked by the calling function makeContinuous
    The hidden data which is out of a possible roi will not be part of the new continuous matrix.

    \param &dObj is the source data object
    \param &resDObj is the resulting data object
    \return retOk
*/
template<typename _Tp> RetVal MakeContinuousFunc(const DataObject &dObj, DataObject &resDObj)
{
    resDObj = DataObject(dObj.getDims() , dObj.m_size, dObj.getType() , 1);
    resDObj.m_owndata = 1;
    dObj.copyAxisTagsTo(resDObj);
    dObj.copyTagMapTo(resDObj);

    int dims = dObj.getDims();
    int newNumMats = resDObj.mdata_size();

    if(newNumMats > 0)
    {
        uchar* newDataPtr = ((cv::Mat_<_Tp>*)(resDObj.m_data[0]))->data;
        int newMatSize = sizeof(_Tp) * resDObj.m_size[dims-2] * resDObj.m_size[dims-1];
        const cv::Mat *tempMat = dObj.getCvPlaneMat(0);

        if (tempMat->isContinuous())
        {
            //first plane
            memcpy((void*)newDataPtr , (void*)(tempMat->data), newMatSize);
            newDataPtr += newMatSize;

            //further planes
            for (int n = 1; n < newNumMats; ++n)
            {
                tempMat = dObj.getCvPlaneMat(n);
                memcpy((void*)newDataPtr , (void*)(tempMat->data), newMatSize);
                newDataPtr += newMatSize;
            }
        }
        else
        {
            int rows = tempMat->rows;
            newMatSize = sizeof(_Tp) * resDObj.m_osize[dims-1]; //only the size of one row in bytes
            const uchar *srcPtr = tempMat->data;

            //first plane
            for (int r = 0; r < rows; ++r)
            {
                memcpy((void*)newDataPtr , (void*)(srcPtr), newMatSize);
                newDataPtr += newMatSize;
                srcPtr += tempMat->step[0];
            }

            //further planes
            for (int n = 1; n < newNumMats; ++n)
            {
                tempMat = dObj.getCvPlaneMat(n);
                srcPtr = tempMat->data;

                for (int r = 0; r < rows; ++r)
                {
                    memcpy((void*)newDataPtr , (void*)(srcPtr), newMatSize);
                    newDataPtr += newMatSize;
                    srcPtr += tempMat->step[0];
                }
            }
        }
    }


    //#### OLD VERSION: all data is copied and a roi of the source object is finally applied to the destination, continous object (memory intense)
    /*resDObj = DataObject(dObj.getDims() , dObj.m_osize, dObj.getType() , 1);
    resDObj.m_owndata = 1;

    int dims = dObj.getDims();
    for (int i = 0 ; i < dims ; i++)
    {
        resDObj.m_size.m_p[i] = dObj.m_size.m_p[i];
        resDObj.m_roi.m_p[i] = dObj.m_roi.m_p[i];
    }

    int roiOffset = 0;
    
    if(dims > 1)
    {
        roiOffset = sizeof(_Tp) * (dObj.m_roi.m_p[dims-1] + dObj.m_roi.m_p[dims-2] * dObj.m_osize.m_p[dims-1]);
    }

    int numMats = dObj.mdata_size();

    int matSize = sizeof(_Tp) * dObj.m_osize[dObj.getDims()-2] * dObj.m_osize[dObj.getDims()-1];

    if(numMats > 0)
    {

        uchar* newDataPtr = ((cv::Mat_<_Tp>*)(resDObj.m_data[0]))->data;
        cv::Mat_<_Tp> *tempMat;

        for (int n = 0; n < numMats; n++)
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
        for (int n = 0; n < numMats; n++)
        {
            ((cv::Mat*)resDObj.m_data[n])->adjustROI(dtop,dbottom,dleft,dright);
        }
    }*/

    return RetVal(retOk);
}

typedef RetVal (*tMakeContinuousFunc)(const DataObject &dObj, DataObject &resDObj);
MAKEFUNCLIST(MakeContinuousFunc)

//! high-level method which copies an incontinuously organized data object to a continuously organized resulting data object, which is returned
/*!
    If the given data object already is in a continuous form (e.g. 2D object or continuous representation for higher dimensions), a shallow
    copy to the given object is returned. In any other cases, a deep copy of the given object is returned, where the entire data block is
    continuously aligned in memory. Additionally, only values within the current region of interest are copied to the new, continous object (in order
    to safe memory).

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

//! equivalent to matlab linspace functino
/*!
    \param &dObj is the source data object
    \return 
    \sa MakeContinuousFunc
*/
template<typename _Tp> ito::RetVal DataObject::linspace(const _Tp start, const _Tp end, const _Tp inc, const int transposed)
{
    ito::RetVal ret(ito::retOk);
    // nan check
    if (0.0 * start != 0.0 || 0.0 * end != 0.0 || 0.0 * inc != 0.0)
    {
        cv::error(cv::Exception(CV_StsAssert, "input parameter is nan or inf", "", __FILE__, __LINE__));
        return RetVal(ito::retError, 0, "nan error");
    }

    int numElements = (int)((end - start) / inc + 1);
    int ne = 0;
    _Tp *dataPtr = NULL;
    const int sizes[2] = {numElements, 1};
    const int sizesT[2] = {1, numElements};
    
    if (!transposed)
    {
        if ((getDims() != 2) || (getSize(0) != numElements) || (getSize(1) != 1))
        {
            freeData();
            CreateFunc<_Tp>(this, 2, sizes, 0, NULL, NULL);
        }
        dataPtr = (_Tp*)(static_cast<cv::Mat_<_Tp>*>(get_mdata()[0])->ptr(0));
        for (_Tp count = start; count <= end; count = count + inc, ne++)
        {
            dataPtr[ne] = count;
        }
    }
    else
    {
        if ((getDims() != 2) || (getSize(1) != numElements) || (getSize(0) != 1))
        {
            freeData();
            CreateFunc<_Tp>(this, 2, sizesT, 0, NULL, NULL);
        }
//        for (_Tp count = start; count <= end; count = count + inc, ne++)
//            at<_Tp>(ne, 0) = count;
        dataPtr = (_Tp*)(static_cast<cv::Mat_<_Tp>*>(get_mdata()[0])->ptr(0));
        for (_Tp count = start; count <= end; count = count + inc, ne++)
        {
            dataPtr[ne] = count;
        }
    }
    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the offset of the specified axis, return 1 if axis does not exist
int DataObject::setAxisOffset(const unsigned int axisNum, const double offset)
{
    if (!m_pDataObjectTags || m_dims < 1)
        return 1; // error

    if (axisNum >= m_pDataObjectTags->m_axisOffsets.size())
        return 1; //error
    uchar *ch = (uchar *)&offset;
    if (!((ch[7] & 0x7f) != 0x7f || (ch[6] & 0xf0) != 0xf0))
        return 1;           
    else
    {
        m_pDataObjectTags->m_axisOffsets[axisNum] = offset  + m_roi[axisNum];
    }
    return 0; //ok
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the scale of the specified axis, return 1 if axis does not exist or scale is 0.0.
int DataObject::setAxisScale(const unsigned int axisNum, const double scale)
{
    if (!m_pDataObjectTags || m_dims < 1) return 1; //error

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
    if (!m_pDataObjectTags || m_dims < 1) return 1; //error

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
    if (!m_pDataObjectTags || m_dims < 1) return 1; //error

    if (axisNum >= m_pDataObjectTags->m_axisDescription.size()) return 1; //error          
    else
    {
        m_pDataObjectTags->m_axisDescription[axisNum] = description;
    }
    return 0; //ok
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the string value of the specified tag, if the tag do not exist, it will be added automatically, return 1 if tagspace does not exist
int DataObject::setTag(const std::string &key, const DataObjectTagType &value)
{
    if(!m_pDataObjectTags || m_dims < 1) return 1; //error
    m_pDataObjectTags->m_tags[key] = value;
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to check whether tag exist or not
bool DataObject::existTag(const std::string &key) const
{
    if(!m_pDataObjectTags || m_dims < 1) return false; //Tag does not existtemplate
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find(key);
    return (it != m_pDataObjectTags->m_tags.end());
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function deletes specified tag. If tag do not exist, return value is 1 else returnvalue is 0
bool DataObject::deleteTag(const std::string &key)
{
    if(!m_pDataObjectTags || m_dims < 1) return false; //tag not deleted
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
    if(!m_pDataObjectTags || m_dims < 1) return 1; //error
    /* Check if object is only an ROI */
    bool isROI = false;
    ByteArray newcontent; // Start with an empty sting
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
        int sizeDim = 0;
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
    newcontent.append(value.data());   // Append the value to the content
    if(newcontent[newcontent.length()-1] != '\n')   // add a \n is not aready there
    {
        newcontent.append("\n");
    }
    // Check if there is already a protocol tag
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find("protocol");
    if(it == m_pDataObjectTags->m_tags.end())  // is not, okay create a new
    {
        m_pDataObjectTags->m_tags["protocol"] = newcontent;
    }
    else
    {   // is there, so just append to existing tag
        //(*it).second.append(newcontent);
        ByteArray tempVal = (*it).second.getVal_ToString();
        tempVal.append(newcontent);
        (*it).second = tempVal;
    }
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns number of bytes required by each value in the array.
/*!

    \return the size of each array element in bytes.
*/
int DataObject::elemSize() const
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
    case tRGBA32:
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

//----------------------------------------------------------------------------------------------------------------------------------
//! returns iterator to the first item in the data object array
/*!
    \return iterator
    \sa DObjIterator
*/
DObjIterator DataObject::begin()
{
    return DObjIterator(this, 0);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns iterator to the end value of this data object array
/*!
    The end value is the first item outside of the data object array.

    \return iterator
    \sa DObjIterator
*/
DObjIterator DataObject::end()
{
    return DObjIterator(this, getTotal());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns constant iterator to the first item in the data object array
/*!
    \return iterator
    \sa DObjConstIterator
*/
DObjConstIterator DataObject::constBegin() const
{
    return DObjConstIterator(this, 0);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! returns constant iterator to the end value of this data object array
/*!
    The end value is the first item outside of the data object array.

    \return iterator
    \sa DObjConstIterator
*/
DObjConstIterator DataObject::constEnd() const
{
    return DObjConstIterator(this, getTotal());
}

//----------------------------------------------------------------------------------------------------------------------------------
// Function return the offset of the values stored within the dataOject
double DataObject::getValueOffset() const
{
    if(!m_pDataObjectTags || m_dims < 1) return 0.0; // default
    return m_pDataObjectTags->m_valueOffset;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Function return the scaling of values stored within the dataOject
double DataObject::getValueScale() const
{
    if(!m_pDataObjectTags || m_dims < 1) return 1.0; // default
    return m_pDataObjectTags->m_valueScale;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Function return the unit description for the values stored within the dataOject
const std::string DataObject::getValueUnit() const
{
    if(!m_pDataObjectTags || m_dims < 1) return std::string(); //default
    return m_pDataObjectTags->m_valueUnit;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Function return the description for the values stored within the dataOject, if tagspace does not exist, NULL is returned.
std::string DataObject::getValueDescription() const
{
    if(!m_pDataObjectTags || m_dims < 1) return std::string(); //default
    return m_pDataObjectTags->m_valueDescription;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Function return the axis-offset for the existing axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
double DataObject::getAxisOffset(const int axisNum) const
{
    if(axisNum < 0 || axisNum >= m_dims)
    {
        cv::error(cv::Exception(CV_StsError, "Parameter axisNum out of range." ,"", __FILE__, __LINE__));
    }
    if(!m_pDataObjectTags) return 0.0; // default
           
    return m_pDataObjectTags->m_axisOffsets[axisNum] - m_roi[axisNum];
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< Function returns the axis-description for the exist axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
double DataObject::getAxisScale(const int axisNum) const
{
    if(axisNum < 0 || axisNum >= m_dims)
    {
        cv::error(cv::Exception(CV_StsError, "Parameter axisNum out of range." ,"", __FILE__, __LINE__));
    }
    if(!m_pDataObjectTags) return 1.0; // default

    return m_pDataObjectTags->m_axisScales[axisNum];
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< Function returns the axis-unit-description for the exist axis specified by axisNum. If axisNum is out of dimension range it returns NULL.
const std::string DataObject::getAxisUnit(const int axisNum, bool &validOperation) const
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

//----------------------------------------------------------------------------------------------------------------------------------
//!< Function returns the axis-description for the exist specified by axisNum. If axisNum is out of dimension range it returns NULL.
std::string DataObject::getAxisDescription(const int axisNum, bool &validOperation) const
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

//----------------------------------------------------------------------------------------------------------------------------------
DataObjectTagType DataObject::getTag(const std::string &key, bool &validOperation) const
{
    validOperation = false;
    if(!m_pDataObjectTags || m_dims < 1)
    {
        return DataObjectTagType(); //error
    }
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.find(key);
    if(it != m_pDataObjectTags->m_tags.end())
    {
        validOperation = true;
        return it->second;
    }
    return DataObjectTagType();
}

//----------------------------------------------------------------------------------------------------------------------------------
bool DataObject::getTagByIndex(const int tagNumber, std::string &key, DataObjectTagType &value) const
{
    if(!m_pDataObjectTags || m_dims < 1)
    {
        key = std::string();
        value = "";
        return false;
    }

    if((tagNumber < 0) || ((int)(tagNumber + 1) > (int)m_pDataObjectTags->m_tags.size()))
    {
        key = std::string();
        value = "";
        return false;
    }
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.begin();
    for(int i = 0; i < tagNumber; i++)
    {
        ++it;
    }

    key = (*it).first;
    value = (*it).second;
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function returns the string-value for 'key' identified by int tagNumber. If key in the TagMap do not exist NULL is returned
std::string DataObject::getTagKey(const int tagNumber, bool &validOperation) const
{
    if(!m_pDataObjectTags || m_dims < 1)
    {
        validOperation = false;
        return std::string(""); //error
    }
    if((tagNumber < 0) || ((int)(tagNumber + 1) > (int)m_pDataObjectTags->m_tags.size()))
    {
        validOperation = false;
        return std::string(""); //does not exist
    }
    std::map<std::string, DataObjectTagType>::iterator it = m_pDataObjectTags->m_tags.begin();
    validOperation = true;
    for(int i = 0; i < tagNumber; i++)
    {
        ++it;
    }
    return (*it).first;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< Function returns the number of elements in the Tags-Maps
int DataObject::getTagListSize() const
{
    if(!m_pDataObjectTags || m_dims < 1) return 0; //error
    return static_cast<int>(m_pDataObjectTags->m_tags.size());
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the string-value of the value unit, return 1 if values does not exist
int DataObject::setValueUnit(const std::string &unit)
{
    if(!m_pDataObjectTags || m_dims < 1) return 1;    //error
    m_pDataObjectTags->m_valueUnit = unit;
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!<  Function to set the string-value of the value description, return 1 if values does not exist
int DataObject::setValueDescription(const std::string &description)
{
    if(!m_pDataObjectTags || m_dims < 1) return 1;    //error
    m_pDataObjectTags->m_valueDescription = description;
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal DataObject::getXYRotationalMatrix(double &r11, double &r12, double &r13, double &r21, double &r22, double &r23, double &r31, double &r32, double &r33) const
{
    if(!m_pDataObjectTags || m_dims < 1) return RetVal(retError, 0, "Tagspace not initialized"); // error
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

//----------------------------------------------------------------------------------------------------------------------------------
RetVal DataObject::setXYRotationalMatrix(double r11, double r12, double r13, double r21, double r22, double r23, double r31, double r32, double r33)
{
    if(!m_pDataObjectTags || m_dims < 1) return RetVal(retError, 0, "Tagspace not initialized"); // error
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

//----------------------------------------------------------------------------------------------------------------------------------
template DATAOBJ_EXPORT RetVal DataObject::linspace<int8>(const int8, const int8, const int8, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<uint8>(const uint8, const uint8, const uint8, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<int16>(const int16, const int16, const int16, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<uint16>(const uint16, const uint16, const uint16, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<int32>(const int32, const int32, const int32, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<uint32>(const uint32, const uint32, const uint32, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<float32>(const float32, const float32, const float32, const int);
template DATAOBJ_EXPORT RetVal DataObject::linspace<float64>(const float64, const float64, const float64, const int);


template<typename _Tp> void coutValue(const _Tp* val, char *buf)
{
}

template<> void coutValue(const ito::uint8* val, char *buf)
{
    sprintf(buf, "%3d", *val);
}

template<> void coutValue(const ito::int8* val, char *buf)
{
    sprintf(buf, "%3d", *val);
}

template<> void coutValue(const ito::uint16* val, char *buf)
{
    sprintf(buf, "%d", *val);
}

template<> void coutValue(const ito::int16* val, char *buf)
{
    sprintf(buf, "%d", *val);
}

template<> void coutValue(const ito::uint32* val, char *buf)
{
    sprintf(buf, "%d", *val);
}

template<> void coutValue(const ito::int32* val, char *buf)
{
    sprintf(buf, "%d", *val);
}

template<> void coutValue(const ito::float32* val, char *buf)
{
    sprintf(buf, "%.8g", *val);
}

template<> void coutValue(const ito::float64* val, char *buf)
{
    sprintf(buf, "%.8g", *val);
}

template<> void coutValue(const ito::complex64* val, char *buf)
{
    if (val->imag() >= 0)
    {
        sprintf(buf, "%.8g+%.8gj", val->real(), val->imag());
    }
    else
    {
        sprintf(buf, "%.8g-%.8gj", val->real(), -val->imag());
    }
}

template<> void coutValue(const ito::complex128* val, char *buf)
{
    if (val->imag() >= 0)
    {
        sprintf(buf, "%.8g+%.8gj", val->real(), val->imag());
    }
    else
    {
        sprintf(buf, "%.8g-%.8gj", val->real(), -val->imag());
    }
}

template<> void coutValue(const ito::Rgba32* val, char *buf)
{
    sprintf(buf, "(%d,%d,%d,%d)", val->r, val->g, val->b, val->a);
}


//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> std::ostream& coutPlane(std::ostream& out, const cv::Mat *plane, int firstLineIndent, int otherIndent)
{
    char buf[128];
    const _Tp *ptr;
    otherIndent = std::min(otherIndent, 127);
    firstLineIndent = std::min(firstLineIndent, 127);

    for (int r = 0; r < plane->rows; ++r)
    {
        if (r == 0)
        {
            memset(buf, ' ', sizeof(char) * firstLineIndent);
            buf[firstLineIndent] = 0;
            std::cout << buf << "[";
            buf[0] = 0;
        }
        else
        {
            memset(buf, ' ', sizeof(char) * otherIndent);
            buf[otherIndent] = 0;
        }

        if (plane->cols > 0)
        {
            ptr = (const _Tp*)plane->ptr(r);
            std::cout << buf << "[";
            for (int c = 0; c < plane->cols - 1; ++c)
            {
                coutValue<_Tp>(ptr, buf);
                ptr++;
                std::cout << buf << ", ";
            }
            coutValue<_Tp>(ptr, buf);
            std::cout << buf << "]";
        }
        else
        {
            std::cout << buf << "[]";
        }

        if (r != plane->rows - 1) //not last row
        {
            std::cout << ",\n" << std::endl;
        }
        else
        {
            std::cout << "]" << std::endl;
        }
    }

    return out;
}


//----------------------------------------------------------------------------------------------------------------------------------
template<typename _Tp> std::ostream& coutFunc(std::ostream& out, const DataObject& dObj)
{
    static const char* types[] =
    {
        "int8", "uint8", "int16", "uint16", "int32", "uint32", "float32", "float64", "complex64", "complex128", "rgba32"
    };

    int numMats = dObj.getNumPlanes();
    int tMat = 0;
    int dims = dObj.getDims();

    if (dims == 0)
    {
        std::cout << "dataObject()\n" << std::endl;
    }
    else
    {
        std::cout << "dataObject(size=[" << dObj.getSize(0);
        for (int dim = 1; dim < dims; ++dim)
        {
            std::cout << "x" << dObj.getSize(dim);
        }
        std::cout << "], dtype='" << types[dObj.getType()] << "'\n" << std::endl;

        if (numMats == 1)
        {
            coutPlane<_Tp>(out, dObj.get_mdata()[dObj.seekMat(tMat, numMats)], 4, 5);
        }
        else
        {

            int *idx = new int[dims];

            for (int nMat = 0; nMat < numMats; nMat++)
            {
                tMat = dObj.seekMat(nMat, numMats);

                dObj.matNumToIdx(tMat, idx);
                std::cout << "[";
                for (int i = 0; i < dims - 2; ++i)
                {
                    std::cout << idx[i] << ",";
                }
                std::cout << ":,:]->(";
                coutPlane<_Tp>(out, dObj.get_mdata()[tMat], 0, 2*dims+5);
                std::cout << ")" << "\n" << std::endl;
            }

            delete[] idx;
        }

        std::cout << ")" << "\n" << std::endl;
    }
    return out;

}

//----------------------------------------------------------------------------------------------------------------------------------
typedef std::ostream& (*tCoutFunc)(std::ostream& out, const DataObject& dObj);

//----------------------------------------------------------------------------------------------------------------------------------
tCoutFunc fListCout[] =
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

//----------------------------------------------------------------------------------------------------------------------------------
std::ostream& operator << (std::ostream& out, const DataObject& dObj)
{
   return fListCout[dObj.getType()](out, dObj);
}



}// namespace ito
