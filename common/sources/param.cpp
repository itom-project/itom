/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#include "../param.h"

#include "../sharedStructures.h"

#include <assert.h>
#include "../numeric.h"

namespace ito
{



//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name only
*   @param [in] name  name of new ParamBase
*   @return     new   ParamBase name "name"
*
*   creates a new ParamBase with name "name", string is copied
*/
ParamBase::ParamBase(const ByteArray &name) : 
    m_type(0), 
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    inOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type
*   @param [in] name  name of new ParamBase
*   @param [in] type  type of new ParamBase for possible types see \ref Type
*   @return     new Param with name and type
*
*   creates a new Param with name and type, string is copied
*/
ParamBase::ParamBase(const ByteArray &name, const uint32 type) : 
    m_type(type),
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    inOutCheck();
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
    m_dVal(0.0, 0.0), 
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

    inOutCheck();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** constructor with name and type, float64 val
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] val    actual value
*   @return     new ParamBase with name, type and val
*
*   creates a new ParamBase with name, type and val. Strings are copied.
*/
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const float64 val) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0, 0.0), 
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
        case Complex:
        case Double:
            m_dVal.real = val;
        break;
        default:
            throw std::logic_error("constructor with float64 val is only callable for types Int, Complex and Double");
        break;
    }

    inOutCheck();
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
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const int32 val) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    inOutCheck();

    switch(typeFilter(type))
    {
        case Char & paramTypeMask:
            m_iVal = (char)val;
        break;
        case Int & paramTypeMask:
            m_iVal = val;
        break;
        case Complex & paramTypeMask:
        case Double & paramTypeMask:
            m_dVal.real = (float64)val;
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
            throw std::runtime_error("constructor with int32 val is only callable for types Int, Complex, Double, String (for val==0 only) and Hardware (for val==0 only)");
        break;
    }
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
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const complex128 val) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(0), 
    m_cVal(NULL)
{
    switch(typeFilter(type))
    {
        case Complex:
            m_dVal.real = val.real();
            m_dVal.imag = val.imag();
        break;
        default:
            throw std::logic_error("constructor with complex128 val is only callable for type Complex");
        break;
    }

    inOutCheck();
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
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const unsigned int size, const char *values) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    inOutCheck();
    
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
                    m_cVal = (char*)malloc(size * sizeof(int32));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<int32*>(m_cVal)[n] = static_cast<int32>(values[n]);
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
                    m_cVal = (char*)malloc(size * sizeof(float64));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<float64*>(m_cVal)[n] = static_cast<float64>(values[n]);
                }
                else
                {
                    m_cVal = 0;
                }
            break;

            case ComplexArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(complex128));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<complex128*>(m_cVal)[n] = static_cast<complex128>(values[n]);
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
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const unsigned int size, const int32 *values) : 
    m_type(type),
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    inOutCheck();

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
                    m_cVal = (char*)malloc(size * sizeof(int32));
                    memcpy(m_cVal, values, size * sizeof(int32));
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
                    m_cVal = (char*)malloc(size * sizeof(float64));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<float64*>(m_cVal)[n] = static_cast<float64>(values[n]);
                }
                else
                {
                    m_iVal = 0;
                }
            break;
            case ComplexArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(complex128));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<complex128*>(m_cVal)[n] = static_cast<complex128>(values[n]);
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
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const unsigned int size, const float64 *values) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    inOutCheck();
    
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
                throw std::invalid_argument("int32 array cannot be converted to char array");
                m_iVal = -1;
                m_cVal = 0;
            break;
            case DoubleArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(float64));
                    memcpy(m_cVal, values, size * sizeof(float64));
                }
                else
                {
                    m_cVal = 0;
                }
            break;
            case ComplexArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(complex128));
                    for (unsigned int n = 0; n < size; n++)
                        reinterpret_cast<complex128*>(m_cVal)[n] = static_cast<complex128>(values[n]);
                }
                else
                {
                    m_iVal = 0;
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
/** array constructor with name and type, size and array
*   @param [in] name   name of new ParamBase
*   @param [in] type   type of new ParamBase for possible types see \ref Type
*   @param [in] size   array size
*   @param [in] val    values
*   @return     new ParamBase (array) with name, type, size and values.
*
*   creates a new ParamBase (array) with name, type, size and values.
*/
ParamBase::ParamBase(const ByteArray &name, const uint32 type, const unsigned int size, const complex128 *values) : 
    m_type(type), 
    m_name(name),
    m_dVal(0.0, 0.0), 
    m_iVal(size), 
    m_cVal(NULL)
{
    inOutCheck();
    
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
                m_iVal = -1;
                m_cVal = 0;
                throw std::invalid_argument("complex128 array cannot be converted to char array");
            break;
            case DoubleArray & paramTypeMask:
                m_iVal = -1;
                m_cVal = 0;
                throw std::invalid_argument("complex128 array cannot be converted to float64 array");
            break;
            case ComplexArray & paramTypeMask:
                m_iVal = size;
                if (m_iVal > 0)
                {
                    m_cVal = (char*)malloc(size * sizeof(complex128));
                    memcpy(m_cVal, values, size * sizeof(complex128));
                }
                else
                {
                    m_iVal = 0;
                }
            break;
            case IntArray & paramTypeMask:
                m_iVal = -1;
                m_cVal = 0;
                throw std::invalid_argument("complex128 array cannot be converted to int32 array");
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
    // 01.12.16 crash when trying to free a fixed pointer assigned to a paramBase copied to another
    // via = operator, using broken assignment. Guess we must not free pointers, when ival == -1 
    // int useLim[10] = {1, 0 ,0, 0, 0, 0};
    // (*param) = ito::ParamBase("useLimits", ito::ParamBase::IntArray, (char*)useLim);
    // but if we do not free we might provocate memory leaks ... needs review
    if ((m_cVal) && ((typeFilter(m_type) == String)
        || (m_iVal > 0) && (typeFilter(m_type) == DoubleArray)
        || (m_iVal > 0) && (typeFilter(m_type) == IntArray)
        || (m_iVal > 0) && (typeFilter(m_type) == CharArray)
        || (m_iVal > 0) && (typeFilter(m_type) == ComplexArray)))
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
        case Complex & ito::paramTypeMask:
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
                m_cVal = (char*)malloc(m_iVal * sizeof(int32));
                memcpy(m_cVal, copyConstr.m_cVal, m_iVal * sizeof(int32));
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
                m_cVal = (char*)malloc(m_iVal * sizeof(float64));
                memcpy(m_cVal, copyConstr.m_cVal, m_iVal * sizeof(float64));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case ComplexArray & paramTypeMask:
            m_iVal = copyConstr.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(complex128));
                memcpy(m_cVal, copyConstr.m_cVal, m_iVal * sizeof(complex128));
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
bool ParamBase::operator == (const ParamBase &rhs) const
{
    if ((m_type & paramTypeMask) == (rhs.m_type & paramTypeMask))
    {
        switch (m_type & paramTypeMask)
        {
        case 0:
            return true; // both are invalid ParamBase objects.
        case Int & ito::paramTypeMask:
        case Char & ito::paramTypeMask:
            return (m_iVal == rhs.m_iVal);

        case Double & ito::paramTypeMask:
        case Complex & ito::paramTypeMask:
            return ito::areEqual(m_dVal.real, rhs.m_dVal.real) && ito::areEqual(m_dVal.imag, rhs.m_dVal.imag);

        case String & paramTypeMask:
            if (m_cVal && rhs.m_cVal)
            {
                return (strcmp(m_cVal, rhs.m_cVal) == 0);
            }
            else
            {
                return m_cVal == rhs.m_cVal;
            }

        case CharArray & paramTypeMask:
            if (m_iVal > 0 && rhs.m_iVal > 0)
            {
                return (memcmp(m_cVal, rhs.m_cVal, m_iVal * sizeof(char)) == 0);
            }
            else
            {
                return (m_iVal <= 0) && (rhs.m_iVal <= 0);
            }

        case IntArray & paramTypeMask:
            if (m_iVal > 0 && (m_iVal == rhs.m_iVal))
            {
                return (memcmp(m_cVal, rhs.m_cVal, m_iVal * sizeof(int32)) == 0);
            }
            else
            {
                return (m_iVal == rhs.m_iVal);
            }

        case DoubleArray & paramTypeMask:
            if (m_iVal > 0 && (m_iVal == rhs.m_iVal))
            {
                for (int i = 0; i < m_iVal; ++i)
                {
                    if (!ito::areEqual(((float64*)m_cVal)[i], ((float64*)rhs.m_cVal)[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return (m_iVal == rhs.m_iVal);
            }

        case ComplexArray & paramTypeMask:
            if (m_iVal > 0 && (m_iVal == rhs.m_iVal))
            {
                for (int i = 0; i < m_iVal; ++i)
                {
                    if (!ito::areEqual(((complex128*)m_cVal)[i], ((complex128*)rhs.m_cVal)[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return (m_iVal == rhs.m_iVal);
            }

        case HWRef & paramTypeMask:
        case DObjPtr & paramTypeMask:
        case Pointer & paramTypeMask:
        case PointCloudPtr & paramTypeMask:
        case PointPtr & paramTypeMask:
        case PolygonMeshPtr & paramTypeMask:
            return (m_cVal == rhs.m_cVal);

        default:
            return false;
        }
    }
    else
    {
        return false; //type is not equal
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< Verifies and possibly corrects the proper set of the direction flags depending on the type.
/* 
    Every parameter also has some "direction" flags as part of its type value.
    If the parameter is used to transport a generic value from a caller to a called
    method or returns a new parameter back, the direction indicates, if the called method

    * is consuming this parameter (In flag must be set),
    * will consume the parameter and will change its value 
      (only the value, not the type) (In | Out flag must both be set),
    * or will create a new value and set it to a previously empty parameter
      (only if the parameter is a return value of a method) (Out flag only must be set).

    If no in/out flag is set, the in-flag as default is automatically added to m_type.

    \seealso ito::ParamBase::Type
*/
void ParamBase::inOutCheck()
{
    if ((m_type & (ParamBase::Type::In | ParamBase::Type::Out)) == 0)
    {
        // if no direction is set, set at least In...
        m_type |= ParamBase::Type::In;
    }

    // verify that Out-only parameters as part of out-vectors of methods
    // in algorithm plugins must not contain any pointer types (like
    // dataObject, pointCloud, polygonMesh, HWRef, point), since
    // the destruction of the created value inside of the algorithm
    // will usually be earlier than the consumer will read these values.
    if ((m_type & ParamBase::Type::Out) && 
        !(m_type & ParamBase::Type::In))
    {
        //These types are not allowed to be output-only.
        //        DObjPtr         = 0x000010 | Pointer | NoAutosave,
        //        HWRef           = 0x000040 | Pointer | NoAutosave,
        //        PointCloudPtr   = 0x000080 | Pointer | NoAutosave,
        //        PointPtr        = 0x000100 | Pointer | NoAutosave,
        //        PolygonMeshPtr  = 0x000200 | Pointer | NoAutosave
        //since NoAutosave is not in the type part of m_type it needs to be appended to the
        //comparison mask.
        switch (m_type & (paramTypeMask | ParamBase::Type::NoAutosave))
        {
        case DObjPtr:
        case PointCloudPtr:
        case PointPtr:
        case PolygonMeshPtr:
        case HWRef:
        {
            // throw exception only in debug mode. You don't want
            // Exceptions of this type in a production system.
            // You cannot check where it comes from then.
            // To check the origin of this exception you would need
            // a debugger attached or a call stack at hand.
            assert(m_type & ParamBase::Type::In && "An out-only param must not be a Ptr-type"); // will always be false!
            
            // do not force the type to be In, too, here, since the
            // parameter is likely to be defined in an out-vector of
            // an algorithm plugin and then it is strictly forbidden
            // to have pointer-like parameters there (beside string).

            //throw std::logic_error("It is not allowed to declare a parameter as OUT"
            //    "only for types DObjPtr, PointCloudPtr, PolygonMeshPtr or HWRef");

            break;
        }

        default:
            //well nothing to be done here
            break;

        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
int ParamBase::getLen(void) const
{
    switch (m_type & paramTypeMask)
    {
        case DoubleArray:
        case IntArray:
        case CharArray:
        case ComplexArray:
            if (m_cVal)
            {
                return m_iVal;
            }
            else
            {
                return -1;
            }

        case String:
            if (m_cVal)
            {
                return static_cast<int>(strlen(m_cVal));
            }
            else
            {
                return 0;
            }
        case Char:
        case Double:
        case Int:
        case Complex:
            return 1;

        default:
            return -1;
    }
}

//-------------------------------------------------------------------------------------
ito::ByteArray ParamBase::getNameWithIndexSuffix(int index) const
{
    char suffix[16];
    sprintf_s(suffix, 16, "[%i]", index);
    ByteArray newName = m_name;
    newName.append(suffix);
    return newName;
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
const ParamBase ParamBase::operator [] (const int index) const
{
    int typeFiltered = getType();

    if ((typeFiltered == CharArray)
        || (typeFiltered == IntArray)
        || (typeFiltered == DoubleArray)
        || (typeFiltered == ComplexArray))
    {
        if (index >= m_iVal || index < 0)
        {
            return ParamBase();
        }
        else
        {
            int len = 0;
            ito::ByteArray newName = getNameWithIndexSuffix(index);

            switch (typeFiltered & ~Pointer)
            {
                case Char:
                    return ParamBase(newName, Char,  (getVal<char *>(len))[index]);
                break;

                case Int:
                    return ParamBase(newName, Int, (getVal<int32 *>(len))[index]);
                break;

                case Double:
                    return ParamBase(newName, Double, (getVal<float64 *>(len))[index]);
                break;

                case Complex:
                    return ParamBase(newName, Complex, (getVal<complex128 *>(len))[index]);
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
            || (typeFilter(m_type) == typeFilter(CharArray))
        || (typeFilter(m_type) == typeFilter(DoubleArray))
        || (typeFilter(m_type) == typeFilter(IntArray))
        || (typeFilter(m_type) == typeFilter(ComplexArray))))
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
                m_cVal = (char*)malloc(m_iVal * sizeof(int32));
                memcpy(m_cVal, rhs.m_cVal, m_iVal * sizeof(int32));
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
                m_cVal = (char*)malloc(m_iVal * sizeof(float64));
                memcpy(m_cVal, rhs.m_cVal, m_iVal * sizeof(float64));
            }
            else
            {
                m_cVal = 0;
            }
        break;

        case ComplexArray & paramTypeMask:
            m_iVal = rhs.m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(complex128));
                memcpy(m_cVal, rhs.m_cVal, m_iVal * sizeof(complex128));
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
        case Complex:
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
                m_cVal = (char*)malloc(m_iVal * sizeof(int32));
                memcpy(m_cVal, rhs->m_cVal, m_iVal * sizeof(int32));
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
                m_cVal = (char*)malloc(m_iVal * sizeof(float64));
                memcpy(m_cVal, rhs->m_cVal, m_iVal * sizeof(float64));
            }
            else
            {
                m_iVal = 0;
            }
        break;

        case ComplexArray & paramTypeMask:
            if (m_cVal)
            {
                free(m_cVal); //must have been a double-array, too
            }
            m_iVal = rhs->m_iVal;
            if (m_iVal > 0)
            {
                m_cVal = (char*)malloc(m_iVal * sizeof(complex128));
                memcpy(m_cVal, rhs->m_cVal, m_iVal * sizeof(complex128));
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
Param::Param(const ByteArray &name, const uint32 type, const char *val, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const char minVal, const char maxVal, const char val, const char *info) :
    ParamBase(name, type, val),
    m_info(info)
{
    assert((type & ParamBase::Char)); //use this constructor only for type character
    m_pMeta = new CharMeta(minVal, maxVal);
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const int32 minVal, const int32 maxVal, const int32 val, const char *info) :
    ParamBase(name, type, val),
    m_info(info)
{
    assert((type & ParamBase::Int)); //use this constructor only for type integer
    m_pMeta = new IntMeta(minVal, maxVal);
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const float64 minVal, const float64 maxVal, const float64 val, const char *info) :
    ParamBase(name, type, val),
    m_info(info)
{
    assert((type & ParamBase::Double)); //use this constructor only for type double
    m_pMeta = new DoubleMeta(minVal, maxVal);
}


//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const char *values, const char *info):
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const int32 *values, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const float64 *values, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const complex128 *values, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const char val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const int32 val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const float64 val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const complex128 val, ParamMeta *meta, const char *info) :
    ParamBase(name, type, val),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const char *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const int32 *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const float64 *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}

//----------------------------------------------------------------------------------------------------------------------------------
Param::Param(const ByteArray &name, const uint32 type, const unsigned int size, const complex128 *values, ParamMeta *meta, const char *info) :
    ParamBase(name, type, size, values),
    m_pMeta(NULL),
    m_info(info)
{
    setMeta(meta, true); //throws exception if meta does not fit to type
}


//----------------------------------------------------------------------------------------------------------------------------------
Param::~Param()
{
    DELETE_AND_SET_NULL(m_pMeta);
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
const Param Param::operator [] (const int index) const
{
    if ((typeFilter(m_type) == CharArray) 
        || (typeFilter(m_type) == IntArray) 
        || (typeFilter(m_type) == DoubleArray)
        || (typeFilter(m_type) == ComplexArray))
    {
        if (index >= getLen() || index < 0)
        {
            return Param();
        }
        else
        {
            ito::ByteArray newName = getNameWithIndexSuffix(index);
            int len;

            switch ((m_type & ~Pointer) & paramTypeMask)
            {
                case Char:
                    {
                    
                    CharMeta *cMeta = nullptr;

                    if (m_pMeta && m_pMeta->getType() == ParamMeta::rttiCharArrayMeta)
                    {
                        const CharArrayMeta *caMeta = static_cast<const CharArrayMeta*>(m_pMeta);
                        cMeta = new CharMeta(caMeta->getMin(), caMeta->getMax(), caMeta->getStepSize());
                    }
                    return Param(newName.data(), m_type & ~Pointer, (getVal<char*>(len))[index], cMeta, m_info.data());
                    }
                break;

                case Int:
                    {
                    IntMeta *iMeta = nullptr;

                    if (m_pMeta && 
                        (m_pMeta->getType() == ParamMeta::rttiIntArrayMeta 
                        || m_pMeta->getType() == ParamMeta::rttiIntervalMeta 
                        || m_pMeta->getType() == ParamMeta::rttiRangeMeta))
                    {
                        const IntMeta *iaMeta = static_cast<const IntMeta*>(m_pMeta);
                        iMeta = new IntMeta(*iaMeta);
                    }

                    //no conversion from RectMeta to single valued met
                    return Param(newName, m_type & ~Pointer, (getVal<int32*>(len))[index], iMeta, m_info.data());
                    }
                break;

                case Double:
                    {
                    DoubleMeta *dMeta = nullptr;

                    if (m_pMeta && (
                        m_pMeta->getType() == ParamMeta::rttiDoubleIntervalMeta 
                        || m_pMeta->getType() == ParamMeta::rttiDoubleArrayMeta))
                    {
                        const DoubleMeta *daMeta = static_cast<const DoubleMeta*>(m_pMeta);
                        dMeta = new DoubleMeta(*daMeta);
                    }

                    return Param(newName, m_type & ~Pointer, (getVal<float64*>(len))[index], dMeta, m_info.data());
                    }
                break;

                case Complex:
                    {
                        //complex has no meta, since no min or max comparison is defined for complex values
                    return Param(newName, m_type & ~Pointer, (getVal<complex128*>(len))[index], nullptr, m_info.data());
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
void Param::setMeta(ParamMeta* meta, bool takeOwnership)
{
    ParamMeta *oldMeta = m_pMeta;

    if (meta)
    {
        ito::ParamMeta::MetaRtti metaType = meta->getType();

#if _DEBUG
        bool valid = false;
        ito::uint32 ptype = getType();

        switch(metaType)
        {
        case ParamMeta::rttiCharMeta:
            if (ptype == ito::ParamBase::Char) valid = true;
            break;
        case ParamMeta::rttiIntMeta:
            if (ptype == ito::ParamBase::Int) valid = true;
            break;
        case ParamMeta::rttiDoubleMeta:
            if (ptype == ito::ParamBase::Double) valid = true;
            break;
        case ParamMeta::rttiStringMeta:
            if (ptype == (ito::ParamBase::String & paramTypeMask)) valid = true;
            break;
        case ParamMeta::rttiDObjMeta:
            if (ptype == (ito::ParamBase::DObjPtr & paramTypeMask)) valid = true;
            break;
        case ParamMeta::rttiHWMeta:
            if (ptype == (ito::ParamBase::HWRef & paramTypeMask)) valid = true;
            break;
        case ParamMeta::rttiCharArrayMeta:
            if (ptype == (ito::ParamBase::CharArray & paramTypeMask)) valid = true;
            break;
        case ParamMeta::rttiIntArrayMeta:
        case ParamMeta::rttiIntervalMeta:
        case ParamMeta::rttiRangeMeta:
        case ParamMeta::rttiRectMeta:
            if (ptype == (ito::ParamBase::IntArray & paramTypeMask)) valid = true;
            break;
        case ParamMeta::rttiDoubleArrayMeta:
        case ParamMeta::rttiDoubleIntervalMeta:
            if (ptype == (ito::ParamBase::DoubleArray & paramTypeMask)) valid = true;
            break;
        default:
            valid = false;
        }

        if (!valid)
        {
            throw std::logic_error("type of meta information does not fit to given parameter type");
        }
#endif

        if (takeOwnership)
        {
            m_pMeta = meta; //Param takes ownership of meta
        }
        else
        {
            switch(metaType)
            {
            case ParamMeta::rttiCharMeta:
                m_pMeta = new CharMeta(*(CharMeta*)(meta));
                break;
            case ParamMeta::rttiIntMeta:
                m_pMeta = new IntMeta(*(IntMeta*)(meta));
                break;
            case ParamMeta::rttiDoubleMeta:
                m_pMeta = new DoubleMeta(*(DoubleMeta*)(meta));
                break;
            case ParamMeta::rttiStringMeta:
                m_pMeta = new StringMeta(*(StringMeta*)(meta));
                break;
            case ParamMeta::rttiDObjMeta:
                m_pMeta = new DObjMeta(*(DObjMeta*)(meta));
                break;
            case ParamMeta::rttiHWMeta:
                m_pMeta = new HWMeta(*(HWMeta*)(meta));
                break;
            case ParamMeta::rttiCharArrayMeta:
                m_pMeta = new CharArrayMeta(*(CharArrayMeta*)(meta));
                break;
            case ParamMeta::rttiIntArrayMeta:
                m_pMeta = new IntArrayMeta(*(IntArrayMeta*)(meta));
                break;
            case ParamMeta::rttiDoubleArrayMeta:
                m_pMeta = new DoubleArrayMeta(*(DoubleArrayMeta*)(meta));
                break;
            case ParamMeta::rttiIntervalMeta:
                m_pMeta = new IntervalMeta(*(IntervalMeta*)(meta));
                break;
            case ParamMeta::rttiRangeMeta:
                m_pMeta = new RangeMeta(*(RangeMeta*)(meta));
                break;
            case ParamMeta::rttiDoubleIntervalMeta:
                m_pMeta = new DoubleIntervalMeta(*(DoubleIntervalMeta*)(meta));
                break;
            case ParamMeta::rttiRectMeta:
                m_pMeta = new RectMeta(*(RectMeta*)(meta));
                break;
            default:
                throw std::logic_error("Type of meta [ParamMeta] is unknown and cannot not be copied or assigned.");
            }
        }
    }
    else
    {
        m_pMeta = NULL;
    }

    if (oldMeta) 
    {
        delete oldMeta; 
        oldMeta = NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/** returns minimum value of parameter if this is available and exists.
*   
*   This method is a wrapper method for ((ito::IntMeta*)getMeta())->getMax()...
*   and returns the minimum value of the underlying meta information. It only
*   returns a valid value for meta structures of type char, charArray, int, intArray, interval, range,
*   double, doubleMeta.
*
*   @return     minimum value of -Inf maximum does not exist
*/
float64 Param::getMin() const
{
    if (m_pMeta)
    {
        switch(m_pMeta->getType())
        {
        case ParamMeta::rttiCharMeta:
        case ParamMeta::rttiCharArrayMeta:
            return static_cast<const CharMeta*>(m_pMeta)->getMin();
        case ParamMeta::rttiIntMeta:
        case ParamMeta::rttiIntArrayMeta:
        case ParamMeta::rttiIntervalMeta:
        case ParamMeta::rttiRangeMeta:
            return static_cast<const IntMeta*>(m_pMeta)->getMin();
        case ParamMeta::rttiDoubleMeta:
        case ParamMeta::rttiDoubleArrayMeta:
            return static_cast<const DoubleMeta*>(m_pMeta)->getMin();
        }
    }
    return -std::numeric_limits<float64>::infinity();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** returns maximum value of parameter if this is available and exists.
*   
*   This method is a wrapper method for ((ito::IntMeta*)getMeta())->getMax()...
*   and returns the maximum value of the underlying meta information. It only
*   returns a valid value for meta structures of type char, charArray, int, intArray, range,
*   double, doubleMeta.
*
*   @return     maximum value of Inf maximum does not exist
*/
float64 Param::getMax() const
{
    if (m_pMeta)
    {
        switch(m_pMeta->getType())
        {
        case ParamMeta::rttiCharMeta:
        case ParamMeta::rttiCharArrayMeta:
            return static_cast<const CharMeta*>(m_pMeta)->getMax();
        case ParamMeta::rttiIntMeta:
        case ParamMeta::rttiIntArrayMeta:
        case ParamMeta::rttiIntervalMeta:
        case ParamMeta::rttiRangeMeta:
            return static_cast<const IntMeta*>(m_pMeta)->getMax();
        case ParamMeta::rttiDoubleMeta:
        case ParamMeta::rttiDoubleArrayMeta:
            return static_cast<const DoubleMeta*>(m_pMeta)->getMax();
        }
    }
    return std::numeric_limits<float64>::infinity();
}

}; //end namespace ito






