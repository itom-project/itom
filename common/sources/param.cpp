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

#include "../param.h"

#include <assert.h>

namespace ito
{

int ParamBase::numericTypeMask = ito::ParamBase::Char | ParamBase::Int | ParamBase::Double;

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name only
*   @param [in] name  name of new ParamBase
*   @return     new   ParamBase name "name"
*
*   creates a new ParamBase with name "name", string is copied
*/
ParamBase::ParamBase(const char *name) : 
    m_type(0), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    InOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type
*   @param [in] name  name of new ParamBase
*   @param [in] type  type of new ParamBase for possible types see \ref Type
*   @return     new Param with name and type
*
*   creates a new Param with name and type, string is copied
*/
ParamBase::ParamBase(const char *name, const uint32 type) : 
    m_type(type),
    m_name(name),
    m_dVal(0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    InOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type, char val and optional info
*   @param [in] name  name of new ParamBase
*   @param [in] type  type of new ParamBase for possible types see \ref Type
*   @param [in] val   character pointer to string pointer
*   @param [in] info  character pointer to string pointer holding information about this ParamBase
*   @return     new ParamBase with name, type, string value
*
*   creates a new ParamBase with name, type, string value. Strings are copied
*/
ParamBase::ParamBase(const char *name, const uint32 type, const char *val) : 
    m_type(type),
    m_name(name),
    m_dVal(0.0), 
    m_iVal(-1), 
    m_cVal(NULL)
{
    if (val)
    {
        if ((m_type & paramTypeMask) == String)
        {
            m_cVal = _strdup(val);
            m_iVal = static_cast<int>(strlen(m_cVal));
        }
        else
        {
            m_cVal = const_cast<char*>(val);
            m_iVal = -1;
        }
    }

    InOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type, char val and optional info
*   @param [in] name  name of new ParamBase
*   @param [in] type  type of new ParamBase for possible types see \ref Type
*   @param [in] val   character pointer to string pointer
*   @param [in] info  character pointer to string pointer holding information about this ParamBase
*   @return     new ParamBase with name, type, string value
*
*   creates a new ParamBase with name, type, string value. Strings are copied
*/
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const char *val) : 
    m_type(type),
    m_name(name),
    m_dVal(0.0), 
    m_iVal(-1), 
    m_cVal(NULL)
{
    if (val)
    {
        if ((m_type & paramTypeMask) == String)
        {
            m_cVal = _strdup(val);
            m_iVal = static_cast<int>(strlen(m_cVal));
        }
        else
        {
            m_cVal = const_cast<char*>(val);
            m_iVal = -1;
        }
    }

    InOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type, double val
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] val    actual value
*   @return     new ParamBase with name, type and val
*
*   creates a new ParamBase with name, type and val. Strings are copied.
*/
ParamBase::ParamBase(const char *name, const uint32 type, const double val) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    switch(typeFilter(type))
    {
        case Char:
            m_iVal = (char)val;
        break;
        case Int:
            m_iVal = (int)val;
        break;
        case Double:
            m_dVal = val;
        break;
        default:
            throw std::logic_error("constructor with double val is only callable for types Int and Double");
        break;
    }

    InOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type, double val
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] val    actual value
*   @return     new ParamBase with name, type and val
*
*   creates a new ParamBase with name, type and val. Strings are copied.
*/
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const double val) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    switch(typeFilter(type))
    {
        case Char:
            m_iVal = (char)val;
        break;
        case Int:
            m_iVal = (int)val;
        break;
        case Double:
            m_dVal = val;
        break;
        default:
            throw std::logic_error("constructor with double val is only callable for types Int and Double");
        break;
    }

    InOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type and int val
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] val    actual value
*   @return     new ParamBase with name, type andval.
*
*   creates a new ParamBase with name, type and val
*/
ParamBase::ParamBase(const char *name, const uint32 type, const int val) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    InOutCheck();

    switch(typeFilter(type))
    {
        case Char & paramTypeMask:
            m_iVal = (char)val;
        break;
        case Int & paramTypeMask:
            m_iVal = val;
        break;
        case Double & paramTypeMask:
            m_dVal = (double)val;
        break;
        case String & paramTypeMask:
            if (val == 0)
            {
                m_iVal = -1;
                m_cVal = NULL;
            }
            else
            {
                throw std::runtime_error("constructor with int val and String type is not callable for val != NULL");
            }
        break;
        case HWRef & paramTypeMask:
            if (val == 0)
            {
                m_iVal = -1;
                m_cVal = NULL;
            }
            else
            {
                throw std::runtime_error("constructor with int val and Hardware type is not callable for val != NULL");
            }
        break;
        default:
            throw std::runtime_error("constructor with int val is only callable for types Int, Double, String (for val==0 only) and Hardware (for val==0 only)");
        break;
    }
}
                             
//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type and int val
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] val    actual value
*   @return     new ParamBase with name, type andval.
*
*   creates a new ParamBase with name, type and val
*/
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const int val): 
    m_type(type), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    InOutCheck();

    switch(typeFilter(type))
    {
        case Char & paramTypeMask:
            m_iVal = (char)val;
        break;
        case Int & paramTypeMask:
            m_iVal = val;
        break;
        case Double & paramTypeMask:
            m_dVal = (double)val;
        break;
        case String & paramTypeMask:
            if (val == 0)
            {
                m_iVal = -1;
                m_cVal = NULL;
            }
            else
            {
                throw std::runtime_error("constructor with int val and String type is not callable for val != NULL");
            }
        break;
        case HWRef & paramTypeMask:
            if (val == 0)
            {
                m_iVal = -1;
                m_cVal = NULL;
            }
            else
            {
                throw std::runtime_error("constructor with int val and Hardware type is not callable for val != NULL");
            }
        break;
        default:
            throw std::runtime_error("constructor with int val is only callable for types Int, Double, String (for val==0 only) and Hardware (for val==0 only)");
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] size   array size
*   @param [in] val    values
*   @return     new ParamBase (array) with name, type, size and values.
*
*   creates a new ParamBase (array) with name, type, size and values.
*/
ParamBase::ParamBase(const char *name, const uint32 type, const unsigned int size, const char *values) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    InOutCheck();
    
    if (values == NULL)
    {
        m_iVal = -1;
        m_cVal = NULL;
    }
    else
    {
        switch (m_type & paramTypeMask)
        {
            case String & paramTypeMask:
                m_cVal = _strdup(values);
                m_iVal = static_cast<int>(strlen(m_cVal));
            break;

            case CharArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(char));
                    memcpy(m_cVal, values, size * sizeof(char));
                }
                else
                {
                    m_cVal = 0;
                }
            break;

            case IntArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(int));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<int*>(m_cVal)[n] = static_cast<int>(values[n]);
                }
                else
                {
                    m_cVal = 0;
                }
            break;

            case DoubleArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(double));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<double*>(m_cVal)[n] = static_cast<double>(values[n]);
                }
                else
                {
                    m_cVal = 0;
                }
            break;

            case HWRef & paramTypeMask:
            case DObjPtr & paramTypeMask:
            case Pointer & paramTypeMask:
            case PointCloudPtr & paramTypeMask:
            case PointPtr & paramTypeMask:
            case PolygonMeshPtr & paramTypeMask:
                m_cVal = const_cast<char*>(values);
                m_iVal = -1;
            break;

            default:
                m_cVal = NULL;
                m_iVal = -1;
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] size   array size
*   @param [in] val    values
*   @return     new ParamBase (array) with name, type, size and values.
*
*   creates a new ParamBase (array) with name, type, size and values.
*/
ParamBase::ParamBase(const char *name, const uint32 type, const unsigned int size, const int *values) : 
    m_type(type),
    m_name(name),
    m_dVal(0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    InOutCheck();

    if ((size <= 0) || (values == NULL))
    {
        m_iVal = -1;
        m_cVal = NULL;
    }
    else
    {
        switch (m_type & paramTypeMask)
        {
            case CharArray & paramTypeMask:
                throw std::invalid_argument("int array cannot be converted to char array");
                m_iVal = -1;
                m_cVal = 0;
            break;
            case IntArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(int));
                    memcpy(m_cVal, values, size * sizeof(int));
                }
                else
                {
                    m_cVal = 0;
                }
            break;
            case DoubleArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(double));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<double*>(m_cVal)[n] = static_cast<double>(values[n]);
                }
                else
                {
                    m_iVal = 0;
                }
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] size   array size
*   @param [in] val    values
*   @return     new ParamBase (array) with name, type, size and values.
*
*   creates a new ParamBase (array) with name, type, size and values.
*/
ParamBase::ParamBase(const char *name, const uint32 type, const unsigned int size, const double *values) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    InOutCheck();
    
    if ((size <= 0) || (values == NULL))
    {
        m_iVal = -1;
        m_cVal = NULL;
    }
    else
    {
        switch (m_type & paramTypeMask)
        {
            case CharArray & paramTypeMask:
                throw std::invalid_argument("int array cannot be converted to char array");
                m_iVal = -1;
                m_cVal = 0;
            break;
            case DoubleArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(double));
                    memcpy(m_cVal, values, size * sizeof(double));
                }
                else
                {
                    m_cVal = 0;
                }
            break;
            case IntArray & paramTypeMask:
                throw std::invalid_argument("double array cannot be converted to char array");
                m_iVal = -1;
                m_cVal = 0;
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** destructor
*
*   clear (frees) the name and in case a string value.
*/
ParamBase::~ParamBase()
{
    if ((m_cVal) && ((typeFilter(m_type) == typeFilter(String))
        || (typeFilter(m_type) == typeFilter(DoubleArray))
        || (typeFilter(m_type) == typeFilter(IntArray))
        || (typeFilter(m_type) == typeFilter(CharArray))))
    {
        free(m_cVal);
        m_cVal = NULL;
    };
}

//----------------------------------------------------------------------------------------------------------------------------------
/** copy constructor
*   @param [in] copyConstr ParamBase to copy from
*   @return     new ParamBase with copied values
*
*   creates ParamBase according to passed Param, strings are copied
*/
ParamBase::ParamBase(const ParamBase &copyConstr) : m_type(copyConstr.m_type), m_name(copyConstr.m_name), m_dVal(copyConstr.m_dVal), m_iVal(copyConstr.m_iVal), m_cVal(0)
{
    switch (copyConstr.m_type & paramTypeMask)
    {
        case Int & ito::paramTypeMask:
        case Char & ito::paramTypeMask:
            m_iVal = copyConstr.m_iVal;
        break;

        case Double & ito::paramTypeMask:
            m_dVal = copyConstr.m_dVal;
        break;

        case String & paramTypeMask:
            if (copyConstr.m_cVal)
            {
                m_cVal = _strdup(copyConstr.m_cVal);
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case CharArray & paramTypeMask:
            m_iVal = copyConstr.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(char));
                memcpy(m_cVal, copyConstr.m_cVal, m_iVal * sizeof(char));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case IntArray & paramTypeMask:
            m_iVal = copyConstr.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(int));
                memcpy(m_cVal, copyConstr.m_cVal, m_iVal * sizeof(int));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case DoubleArray & paramTypeMask:
            m_iVal = copyConstr.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(double));
                memcpy(m_cVal, copyConstr.m_cVal, m_iVal * sizeof(double));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case HWRef & paramTypeMask:
        case DObjPtr & paramTypeMask:
        case Pointer & paramTypeMask:
        case PointCloudPtr & paramTypeMask:
        case PointPtr & paramTypeMask:
        case PolygonMeshPtr & paramTypeMask:
            m_cVal = copyConstr.m_cVal;
        break;

        default:
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ParamBase::InOutCheck()
{
    if ((m_type & (ParamBase::In | ParamBase::Out)) == 0)
    {
        m_type |= ParamBase::In;
    }
    if (((m_type & ParamBase::In) | ParamBase::Out) == ParamBase::Out)
    {
        //out only not allowed for pointer-based types (beside arrays)
        if (m_type & ((DObjPtr | PointCloudPtr | PolygonMeshPtr| HWRef) ^ (Pointer | NoAutosave)))
        {
            throw std::logic_error("It is not allowed to delcare a parameter as OUT for types DObjPtr, PointCloudPtr, PolygonMeshPtr or HWRef");
        }
    }
}


//--------------------------------------------------------------------------------------------
//  ASSIGNMENT AND OPERATORS
//--------------------------------------------------------------------------------------------
            
//----------------------------------------------------------------------------------------------------------------------------------
/** braces operator
*   @param [in] num array index for which the value should be returned
*   @return     new tParam with values of ParamBase[num] in the array
*
*   returns the value of the index num from the array
*/
const ParamBase ParamBase::operator [] (const int num) const
{
    if ((typeFilter(m_type) == CharArray) 
        || (typeFilter(m_type) == IntArray) 
        || (typeFilter(m_type) == DoubleArray))
    {
        if (num > m_iVal)
        {
            return ParamBase();
        }
        else
        {
            int len = 0;
            switch (m_type & ~Pointer)
            {
                case Char:
                    return ParamBase(m_name, Char,  (getVal<char *>(len))[num]);
                break;

                case Int:
                    return ParamBase(m_name, Int, (getVal<int *>(len))[num]);
                break;

                case Double:
                    return ParamBase(m_name, Double, (getVal<double *>(len))[num]);
                break;

                default:
                    return ParamBase();
                break;
            }
        }
    }
    else
    {
        return ParamBase();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** assignment operator
*   @param [in] rhs ParamBase to copy from
*   @return     new ParamBase with copied values
*
*   sets values of lhs to values of rhs ParamBase, strings are copied
*/
ParamBase& ParamBase::operator = (const ParamBase &rhs)
{
    //first clear old ParamBase:
    if ((m_cVal) && ((typeFilter(m_type) == typeFilter(String))
            || (typeFilter(m_type) == typeFilter((CharArray)))
        || (typeFilter(m_type) == typeFilter(DoubleArray))
        || (typeFilter(m_type) == typeFilter(IntArray))))
    {
        free(m_cVal);
        m_cVal = NULL;
    };

    //now set new parameters:
    m_type = rhs.m_type;
    m_name = rhs.m_name;
    m_dVal = rhs.m_dVal;
    m_iVal = rhs.m_iVal;

    switch (typeFilter(rhs.m_type))
    {
        case String & paramTypeMask:
            if (rhs.m_cVal)
            {
                m_cVal = _strdup(rhs.m_cVal);
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case CharArray & paramTypeMask:
            m_iVal = rhs.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(char));
                memcpy(m_cVal, rhs.m_cVal, m_iVal * sizeof(char));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case IntArray & paramTypeMask:
            m_iVal = rhs.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(int));
                memcpy(m_cVal, rhs.m_cVal, m_iVal * sizeof(int));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case DoubleArray & paramTypeMask:
            m_iVal = rhs.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(double));
                memcpy(m_cVal, rhs.m_cVal, m_iVal * sizeof(double));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case HWRef & paramTypeMask:
        case DObjPtr & paramTypeMask:
        case Pointer & paramTypeMask:
        case PointCloudPtr & paramTypeMask:
        case PointPtr & paramTypeMask:
        case PolygonMeshPtr & paramTypeMask:
            m_cVal = rhs.m_cVal;
        break;

        default:
            m_cVal = rhs.m_cVal;
        break;
    }

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamBase::copyValueFrom(const ParamBase *rhs)
{
    if (typeFilter(m_type) != rhs->getType())
    {
        return ito::RetVal(ito::retError, 0, "tParam types are not equal");
    }

    switch (m_type & paramTypeMask)
    {
        case Char:
        case Int:
            m_iVal = rhs->m_iVal;
        break;

        case Double:
            m_dVal = rhs->m_dVal;
        break;

        case String:
            if (m_cVal)
            {
                free(m_cVal); //must have been a string, too (since no type-change)
            }
            if (rhs->m_cVal)
            {
                m_cVal = _strdup(rhs->m_cVal);
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case CharArray & paramTypeMask:
            if (m_cVal)
            {
                free(m_cVal); //must have been an int-array, too
            }
            m_iVal = rhs->m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(char));
                memcpy(m_cVal, rhs->m_cVal, m_iVal * sizeof(char));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case IntArray & paramTypeMask:
            if (m_cVal)
            {
                free(m_cVal); //must have been an int-array, too
            }
            m_iVal = rhs->m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(int));
                memcpy(m_cVal, rhs->m_cVal, m_iVal * sizeof(int));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case DoubleArray & paramTypeMask:
            if (m_cVal)
            {
                free(m_cVal); //must have been a double-array, too
            }
            m_iVal = rhs->m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(double));
                memcpy(m_cVal, rhs->m_cVal, m_iVal * sizeof(double));
            }
            else
            {
                m_iVal = 0;
            }
        break;

        case HWRef & paramTypeMask:
        case DObjPtr & paramTypeMask:
        case Pointer & paramTypeMask:
        case PointCloudPtr & paramTypeMask:
        case PointPtr & paramTypeMask:
        case PolygonMeshPtr & paramTypeMask:
            m_cVal = rhs->m_cVal;
        break;

        default:
            return ito::RetVal(ito::retError, 0, "unknown parameter type (ParamBase)");
        break;
    }

    return ito::RetVal(ito::retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const char *val, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const double minVal, const double maxVal, const double val, const char *info) :
    ParamBase(name, type, val),
    m_info(info)
{
    assert((type & ParamBase::Double));
    m_pMeta = new DoubleMeta(minVal, maxVal);
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const int minVal, const int maxVal, const int val, const char *info) :
    ParamBase(name, type, val),
    m_info(info)
{
    assert((type & ParamBase::Int));
    m_pMeta = new IntMeta(minVal, maxVal);
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const char minVal, const char maxVal, const char val, const char *info) :
    ParamBase(name, type, val),
    m_info(info)
{
    assert((type & ParamBase::Char));
    m_pMeta = new CharMeta(minVal, maxVal);
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const unsigned int size, const char *values, const char *info):
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const unsigned int size, const int *values, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const unsigned int size, const double *values, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const int val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    if (!setMeta(meta, true))
    {
        throw std::runtime_error("Type of meta [ParamMeta] must be included in given type of Param.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const double val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    if (!setMeta(meta, true))
    {
        throw std::runtime_error("Type of meta [ParamMeta] must be included in given type of Param.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const char val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    if (!setMeta(meta, true))
    {
        throw std::runtime_error("Type of meta [ParamMeta] must be included in given type of Param.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const unsigned int size, const double *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    if (!setMeta(meta, true))
    {
        throw std::runtime_error("Type of meta [ParamMeta] must be included in given type of Param.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const unsigned int size, const int *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    if (!setMeta(meta, true))
    {
        throw std::runtime_error("Type of meta [ParamMeta] must be included in given type of Param.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const char *name, const uint32 type, const unsigned int size, const char *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    if (!setMeta(meta, true))
    {
        throw std::runtime_error("Type of meta [ParamMeta] must be included in given type of Param.");
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::~Param()
{
    if (m_pMeta)
    {
        delete m_pMeta;
        m_pMeta = NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const Param &copyConstr) : ParamBase(copyConstr), m_pMeta(NULL), m_info(copyConstr.m_info)
{
    setMeta(copyConstr.m_pMeta);
}


//--------------------------------------------------------------------------------------------
//  ASSIGNMENT AND OPERATORS
//--------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------------
const Param Param::operator [] (const int num) const
{
    if (((m_type & paramTypeMask) == CharArray) 
        || ((m_type & paramTypeMask) == IntArray) 
        || ((m_type & paramTypeMask) == DoubleArray))
    {
        if (num > getLen())
        {
            return Param();
        }
        else
        {
            int len = 0;
            switch (m_type & ~Pointer & paramTypeMask)
            {
                case Char:
                    {
                    CharMeta *cMeta = static_cast<CharMeta*>(m_pMeta);
                    if (cMeta) cMeta = new CharMeta(*cMeta);
                    return Param(m_name.data(), m_type & ~Pointer, (getVal<char*>(len))[num], cMeta, m_info.data());
                    }
                break;

                case Int:
                    {
                    IntMeta *iMeta = static_cast<IntMeta*>(m_pMeta);
                    if (iMeta) iMeta = new IntMeta(*iMeta);
                    return Param(m_name.data(), m_type & ~Pointer, (getVal<int*>(len))[num], iMeta, m_info.data());
                    }
                break;

                case Double:
                    {
                    DoubleMeta *dMeta = static_cast<DoubleMeta*>(m_pMeta);
                    if (dMeta) dMeta = new DoubleMeta(*dMeta);
                    return Param(m_name.data(), m_type & ~Pointer, (getVal<double*>(len))[num], dMeta, m_info.data());
                    }
                break;

                default:
                    return Param();
                break;
            }
        }
    }
    else
    {
        return Param();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
Param& Param::operator = (const Param &rhs)
{
    ParamBase::operator=(rhs);
    m_info = rhs.m_info;
    setMeta(const_cast<ito::ParamMeta*>(rhs.getMeta()));
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal Param::copyValueFrom(const ParamBase *rhs)
{
    return ParamBase::copyValueFrom(rhs);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool Param::setMeta(ParamMeta* meta, bool takeOwnership)
{
    if (m_pMeta) 
    {
        delete m_pMeta; 
        m_pMeta = NULL;
    }

    if (meta)
    {
        uint32 metaType = meta->getType();
        if ((metaType & m_type) != metaType)
        {
            return false;
        }
        else if (takeOwnership)
        {
            m_pMeta = meta; //Param takes ownership of meta
        }
        else
        {
            switch(metaType & ito::paramTypeMask)
            {
            case Param::Int:
                m_pMeta = new IntMeta(*(IntMeta*)(meta));
                break;
            case Param::Double:
                m_pMeta = new DoubleMeta(*(DoubleMeta*)(meta));
                break;
            case Param::Char:
                m_pMeta = new CharMeta(*(CharMeta*)(meta));
                break;
            case Param::String:
                m_pMeta = new StringMeta(*(StringMeta*)(meta));
                break;
            case Param::DObjPtr & ito::paramTypeMask:
                m_pMeta = new DObjMeta(*(DObjMeta*)(meta));
                break;
            case Param::HWRef & ito::paramTypeMask:
                m_pMeta = new HWMeta(*(HWMeta*)(meta));
                break;
            default:
                throw std::logic_error("Type of meta [ParamMeta] is unknown and cannot not be copied or assigned.");
            }
        }
    }
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool Param::copyMetaFrom(const ParamMeta *meta)
{
    if (m_pMeta) 
    {
        delete m_pMeta; 
        m_pMeta = NULL;
    }

    if (meta)
    {
        uint32 metaType = meta->getType();
        if ((metaType & m_type) != metaType)
        {
            return false;
        }
        else
        {
            switch(metaType & ito::paramTypeMask)
            {
            case Param::Int:
                m_pMeta = new IntMeta(*(IntMeta*)(meta));
                break;
            case Param::Double:
                m_pMeta = new DoubleMeta(*(DoubleMeta*)(meta));
                break;
            case Param::Char:
                m_pMeta = new CharMeta(*(CharMeta*)(meta));
                break;
            case Param::String:
                m_pMeta = new StringMeta(*(StringMeta*)(meta));
                break;
            case Param::DObjPtr & ito::paramTypeMask:
                m_pMeta = new DObjMeta(*(DObjMeta*)(meta));
                break;
            case Param::HWRef & ito::paramTypeMask:
                m_pMeta = new HWMeta(*(HWMeta*)(meta));
                break;
            default:
                throw std::logic_error("Type of meta [ParamMeta] is unknown and cannot not be copied or assigned.");
            }
        }
    }
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
double Param::getMin() const
{
    if (m_pMeta)
    {
        switch(m_pMeta->getType())
        {
        case ParamBase::Char:
            return static_cast<const CharMeta*>(m_pMeta)->getMin();
        case ParamBase::Int:
            return static_cast<const IntMeta*>(m_pMeta)->getMin();
        case ParamBase::Double:
            return static_cast<const DoubleMeta*>(m_pMeta)->getMin();
        }
    }
    return -std::numeric_limits<double>::max();
}

//----------------------------------------------------------------------------------------------------------------------------------
double Param::getMax() const
{
    if (m_pMeta)
    {
        switch(m_pMeta->getType())
        {
        case ParamBase::Char:
            return static_cast<const CharMeta*>(m_pMeta)->getMax();
        case ParamBase::Int:
            return static_cast<const IntMeta*>(m_pMeta)->getMax();
        case ParamBase::Double:
            return static_cast<const DoubleMeta*>(m_pMeta)->getMax();
        }
    }
    return -std::numeric_limits<double>::max();
}

}; //end namespace ito






