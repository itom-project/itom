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

#include "../numeric.h"
#include <assert.h>

namespace ito {

//-------------------------------------------------------------------------------------
/* converts a flags value, e.g. ParamBase::Readonly | ParamBase::Out to the
   internal 16bit representation
*/
constexpr uint16 toFlagsInternal(uint32 flags)
{
    return (uint16)((flags & ito::paramFlagMask) >> 16);
}

//-------------------------------------------------------------------------------------
/* converts a m_flags value to its representation using the ParamBase::Flags enum flag
 */
constexpr uint32 toFlagsExternal(uint16 flags)
{
    return ((uint32)flags) << 16;
}

//-------------------------------------------------------------------------------------
ParamBase::ParamBase() : d(new Data())
{
    setDefaultAutosaveFlag();
}

//-------------------------------------------------------------------------------------
/** constructor with name only
 *   @param [in] name  name of new ParamBase
 *   @return     new   ParamBase name "name"
 *
 *   creates a new ParamBase with name "name", string is copied
 */
ParamBase::ParamBase(const ByteArray& name) : d(new Data(name))
{
    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();
}

//-------------------------------------------------------------------------------------
/** constructor with name and type
 *   @param [in] name  name of new ParamBase
 *   @param [in] type  type of new ParamBase for possible types see \ref Type
 *   @return     new Param with name and type
 *
 *   creates a new Param with name and type, string is copied
 */
ParamBase::ParamBase(const ByteArray& name, const uint32 typeAndFlags) : d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);
    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();
}

//-------------------------------------------------------------------------------------
/** constructor with name and type, char val and optional info
 *   @param [in] name  name of new ParamBase
 *   @param [in] type  type of new ParamBase for possible types see \ref Type
 *   @param [in] val   character pointer to string pointer
 *   @param [in] info  character pointer to string pointer holding information about this ParamBase
 *   @return     new ParamBase with name, type, string value
 *
 *   creates a new ParamBase with name, type, string value. Strings are copied
 */
ParamBase::ParamBase(const ByteArray& name, const uint32 typeAndFlags, const char* val) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    //todo: add scalar types with value 0, since this can occur, too!!!

    switch (d->type)
    {
    case Char:
    case Int:
    case Complex:
    case Double:
        if (val == 0)
        {
            memset(&(d->data), 0, sizeof(ParamBaseData));
            d->len = 0;
        }
        else
        {
            // delete d since destructor of ParamBase will not be called if exception is thrown.
            DELETE_AND_SET_NULL(d);
            throw std::logic_error(
                "valie must be 0 only for Char, "
                "Int, Double, Complex.");
        }
    case DObjPtr:
    case PointCloudPtr:
    case PolygonMeshPtr:
    case PointPtr:
    case HWRef:
        d->data.ptrVal = const_cast<char*>(val);
        d->len = 0;
        break;
    case String:
        if (val)
        {
            d->len = strlen(val);
            d->data.ptrVal = new char[d->len + 1];
            std::copy_n(val, d->len + 1, (char*)(d->data.ptrVal));
        }
        break;
    case CharArray:
    case IntArray:
    case DoubleArray:
    case ComplexArray:
    case StringList:
        if (val== nullptr)
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
        else
        {
            // delete d since destructor of ParamBase will not be called if exception is thrown.
            DELETE_AND_SET_NULL(d);
            throw std::logic_error(
                "value must be nullptr for CharArray, "
                "IntArray, DoubleArray, ComplexArray or StringList.");
        }
        break;
    default:
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error(
            "invalid type for constructor with const char* value.");
        break;
    }
}

//-------------------------------------------------------------------------------------
/** constructor with name and type, float64 val
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] val    actual value
 *   @return     new ParamBase with name, type and val
 *
 *   creates a new ParamBase with name, type and val. Strings are copied.
 */
ParamBase::ParamBase(const ByteArray& name, const uint32 typeAndFlags, const float64 val) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    switch (d->type)
    {
    case Char:
        d->data.i8Val = (int8)val;
        break;
    case Int:
        d->data.i32Val = (int32)val;
        break;
    case Complex:
        d->data.c128Val.real = val;
        break;
    case Double:
        d->data.f64Val = val;
        break;
    default:
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error("constructor with float64 val is only callable for types Char, Int, "
                               "Complex and Double");
        break;
    }
}

//-------------------------------------------------------------------------------------
/** constructor with name and type and int val
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] val    actual value
 *   @return     new ParamBase with name, type andval.
 *
 *   creates a new ParamBase with name, type and val
 */
ParamBase::ParamBase(const ByteArray& name, const uint32 typeAndFlags, const int32 val) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    switch (d->type)
    {
    case Char:
        d->data.i8Val = (int8)val;
        break;
    case Int:
        d->data.i32Val = val;
        break;
    case Complex:
        d->data.c128Val.real = (float64)val;
        break;
    case Double:
        d->data.f64Val = (float64)val;
        break;
    case String:
        if (val == 0)
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
        else
        {
            // delete d since destructor of ParamBase will not be called if exception is thrown.
            DELETE_AND_SET_NULL(d);
            throw std::runtime_error(
                "constructor with int32 value and type String is not callable for val != nullptr");
        }
        break;
    case HWRef:
        if (val == 0)
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
        else
        {
            // delete d since destructor of ParamBase will not be called if exception is thrown.
            DELETE_AND_SET_NULL(d);
            throw std::runtime_error(
                "constructor with int32 value and type HWRef is not callable for val != nullptr");
        }
        break;
    default:
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::runtime_error(
            "constructor with int32 val is only callable for types Int, Complex, Double, String "
            "(for val==0 only) and Hardware (for val==0 only)");
        break;
    }
}

//-------------------------------------------------------------------------------------
/** constructor with name and type, complex128 val
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] val    actual value
 *   @return     new ParamBase with name, type and val
 *
 *   creates a new ParamBase with name, type and val. Strings are copied.
 */
ParamBase::ParamBase(const ByteArray& name, const uint32 typeAndFlags, const complex128 val) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    switch (d->type)
    {
    case Complex:
        d->data.c128Val.real = val.real();
        d->data.c128Val.imag = val.imag();
        break;
    default:
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error("constructor with complex128 val is only callable for type Complex");
        break;
    }

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();
}

//-------------------------------------------------------------------------------------
/** constructor with name and type, and a list of strings
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] val    actual value
 *   @return     new ParamBase with name, type and val
 *
 *   creates a new ParamBase with name, type and val. Strings are copied.
 */
ParamBase::ParamBase(
    const ByteArray& name, const uint32 typeAndFlags, const uint32 size, const ByteArray* values) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    switch (d->type)
    {
    case StringList: {
        d->len = size;
        ByteArray* buf = new ByteArray[size];
        std::copy_n(values, size, buf);
        d->data.ptrVal = buf;
        break;
    }
    default:
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error(
            "constructor with ByteArray values is only callable for type StringList");
        break;
    }

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();
}

//-------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] size   array size
 *   @param [in] val    values
 *   @return     new ParamBase (array) with name, type, size and values.
 *
 *   creates a new ParamBase (array) with name, type, size and values.
 */
ParamBase::ParamBase(
    const ByteArray& name, const uint32 typeAndFlags, const unsigned int size, const char* values) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    if (d->type == CharArray)
    {
        d->len = size;

        if (size > 0 && values)
        {
            d->len = size;
            d->data.ptrVal = new char[size];
            std::copy_n(values, size, (char*)d->data.ptrVal);
        }
        else
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
    }
    else
    {
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error(
            "constructor with const char* values and size is only callable for type CharArray.");
    }
}

//-------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] size   array size
 *   @param [in] val    values
 *   @return     new ParamBase (array) with name, type, size and values.
 *
 *   creates a new ParamBase (array) with name, type, size and values.
 */
ParamBase::ParamBase(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const int32* values) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    if (d->type == IntArray)
    {
        d->len = size;

        if (size > 0 && values)
        {
            d->len = size;
            d->data.ptrVal = new int32[size];
            std::copy_n(values, size, (int32*)d->data.ptrVal);
        }
        else
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
    }
    else
    {
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error(
            "constructor with const int32* values and size is only callable for type IntArray.");
    }
}

//-------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] size   array size
 *   @param [in] val    values
 *   @return     new ParamBase (array) with name, type, size and values.
 *
 *   creates a new ParamBase (array) with name, type, size and values.
 */
ParamBase::ParamBase(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const float64* values) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    if (d->type == DoubleArray)
    {
        d->len = size;

        if (size > 0 && values)
        {
            d->len = size;
            d->data.ptrVal = new float64[size];
            std::copy_n(values, size, (float64*)d->data.ptrVal);
        }
        else
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
    }
    else
    {
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error("constructor with const float64* values and size is only callable "
                               "for type DoubleArray.");
    }
}

//-------------------------------------------------------------------------------------
/** array constructor with name and type, size and array
 *   @param [in] name   name of new ParamBase
 *   @param [in] type   type of new ParamBase for possible types see \ref Type
 *   @param [in] size   array size
 *   @param [in] val    values
 *   @return     new ParamBase (array) with name, type, size and values.
 *
 *   creates a new ParamBase (array) with name, type, size and values.
 */
ParamBase::ParamBase(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const complex128* values) :
    d(new Data(name))
{
    d->type = typeAndFlags & ito::paramTypeMask;
    d->flags = toFlagsInternal(typeAndFlags);

    checkAndCorrectInOutFlag();
    setDefaultAutosaveFlag();

    if (d->type == ComplexArray)
    {
        d->len = size;

        if (size > 0 && values)
        {
            d->len = size;
            d->data.ptrVal = new complex128[size];
            std::copy_n(values, size, (complex128*)d->data.ptrVal);
        }
        else
        {
            d->len = 0;
            d->data.ptrVal = nullptr;
        }
    }
    else
    {
        // delete d since destructor of ParamBase will not be called if exception is thrown.
        DELETE_AND_SET_NULL(d);
        throw std::logic_error("constructor with const complex128* values and size is only "
                               "callable for type ComplexArray.");
    }
}

//-------------------------------------------------------------------------------------
/** destructor
 *
 *   clear (frees) the name and in case a string value.
 */
ParamBase::~ParamBase()
{
    decRefAndFree(d);
}

//-------------------------------------------------------------------------------------
//!< depending on the type, set the default value for the autosave flag.
void ParamBase::setDefaultAutosaveFlag()
{
    ito::uint16 flags = d->flags;

    switch (d->type)
    {
    case ito::ParamBase::DObjPtr:
    case ito::ParamBase::PointCloudPtr:
    case ito::ParamBase::PolygonMeshPtr:
    case ito::ParamBase::PointPtr:
    case ito::ParamBase::HWRef:
        flags |= toFlagsInternal(ito::ParamBase::NoAutosave);
        break;
    default:
        flags &= ~toFlagsInternal(ito::ParamBase::NoAutosave);
        break;
    }

    if (flags != d->flags)
    {
        detach();
        d->flags = flags;
    }
}

//-------------------------------------------------------------------------------------
void ParamBase::clearData(Data* data)
{
    bool arrType = true;

    switch (data->type)
    {
    case String:
    case CharArray:
        delete[]((char*)data->data.ptrVal);
        break;
    case DoubleArray:
        delete[]((ito::float64*)data->data.ptrVal);
        break;
    case IntArray:
        delete[]((ito::int32*)data->data.ptrVal);
        break;
    case ComplexArray:
        delete[]((ito::complex128*)data->data.ptrVal);
        break;
    case StringList:
        delete[]((ito::ByteArray*)data->data.ptrVal);
        break;
    default:
        arrType = false;
        break;
    }

    if (arrType)
    {
        data->data.ptrVal = nullptr;
        data->len = 0;
    }
}

//-------------------------------------------------------------------------------------
/** copy constructor
 *   @param [in] copyConstr ParamBase to copy from
 *   @return     new ParamBase with copied values
 *
 *   creates ParamBase according to passed Param, strings are copied
 */
ParamBase::ParamBase(const ParamBase& other) : d(other.d)
{
    incRef(d);
}

//-------------------------------------------------------------------------------------
ParamBase::ParamBase(ParamBase&& other) noexcept : d(other.d)
{
    other.d = nullptr;
}

//-------------------------------------------------------------------------------------
bool ParamBase::operator==(const ParamBase& rhs) const
{
    if (d == rhs.d)
    {
        return true;
    }

    if ((d->type) == (rhs.d->type))
    {
        switch (d->type)
        {
        case 0:
            return true; // both are invalid ParamBase objects.
        case Int:
            return (d->data.i32Val == rhs.d->data.i32Val);
        case Char:
            return (d->data.i8Val == rhs.d->data.i8Val);
        case Double:
            return ito::areEqual(d->data.f64Val, rhs.d->data.f64Val);
        case Complex:
            return ito::areEqual(d->data.c128Val.real, rhs.d->data.c128Val.real) &&
                ito::areEqual(d->data.c128Val.imag, rhs.d->data.c128Val.imag);

        case String:
            if (d->data.ptrVal && rhs.d->data.ptrVal)
            {
                return (strcmp((char*)d->data.ptrVal, (char*)rhs.d->data.ptrVal) == 0);
            }
            else
            {
                return d->data.ptrVal == rhs.d->data.ptrVal;
            }

        case CharArray:
            if (d->len > 0 && rhs.d->len > 0)
            {
                return (memcmp(d->data.ptrVal, rhs.d->data.ptrVal, d->len * sizeof(char)) == 0);
            }
            else
            {
                return (d->len <= 0) && (rhs.d->len <= 0);
            }

        case IntArray:
            if (d->len > 0 && (d->len == rhs.d->len))
            {
                return (memcmp(d->data.ptrVal, rhs.d->data.ptrVal, d->len * sizeof(int32)) == 0);
            }
            else
            {
                return (d->len == rhs.d->len);
            }

        case DoubleArray:
            if (d->len > 0 && (d->len == rhs.d->len))
            {
                for (uint32 i = 0; i < d->len; ++i)
                {
                    if (!ito::areEqual(
                            ((float64*)d->data.ptrVal)[i], ((float64*)rhs.d->data.ptrVal)[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return (d->len == rhs.d->len);
            }

        case ComplexArray:
            if (d->len > 0 && (d->len == rhs.d->len))
            {
                for (ito::uint32 i = 0; i < d->len; ++i)
                {
                    if (!ito::areEqual(
                            ((complex128*)d->data.ptrVal)[i], ((complex128*)rhs.d->data.ptrVal)[i]))
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return (d->len == rhs.d->len);
            }

        case StringList:
            if (d->len > 0 && rhs.d->len > 0)
            {
                const ByteArray* list1 = (const ByteArray*)d->data.ptrVal;
                const ByteArray* list2 = (const ByteArray*)rhs.d->data.ptrVal;

                for (ito::uint32 i = 0; i < d->len; ++i)
                {
                    if (list1[i] != list2[i])
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return (d->len <= 0) && (rhs.d->len <= 0);
            }

        case HWRef:
        case DObjPtr:
        case Pointer:
        case PointCloudPtr:
        case PointPtr:
        case PolygonMeshPtr:
            return (d->data.ptrVal == rhs.d->data.ptrVal);

        default:
            return false;
        }
    }
    else
    {
        return false; // type is not equal
    }
}

//-------------------------------------------------------------------------------------
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
      f
    If no in/out flag is set, the in-flag as default is automatically added to d->type.

    \seealso ito::ParamBase::Type
*/
void ParamBase::checkAndCorrectInOutFlag()
{
    const uint16 in_internal = toFlagsInternal(ParamBase::In);
    const uint16 out_internal = toFlagsInternal(ParamBase::Out);

    if ((d->flags & (in_internal | out_internal)) == 0)
    {
        detach();

        // if no direction is set, set at least In...
        d->flags |= in_internal;
    }

    // verify that Out-only parameters as part of out-vectors of methods
    // in algorithm plugins must not contain any pointer types (like
    // dataObject, pointCloud, polygonMesh, HWRef, point), since
    // the destruction of the created value inside of the algorithm
    // will usually be earlier than the consumer will read these values.
    if ((d->flags & out_internal) && !(d->flags & in_internal))
    {
        // These types are not allowed to be output-only.
        //        DObjPtr         = 0x000010 | Pointer | NoAutosave,
        //        HWRef           = 0x000040 | Pointer | NoAutosave,
        //        PointCloudPtr   = 0x000080 | Pointer | NoAutosave,
        //        PointPtr        = 0x000100 | Pointer | NoAutosave,
        //        PolygonMeshPtr  = 0x000200 | Pointer | NoAutosave
        // since NoAutosave is not in the type part of d->type it needs to be appended to the
        // comparison mask.
        switch (d->type)
        {
        case DObjPtr:
        case PointCloudPtr:
        case PointPtr:
        case PolygonMeshPtr:
        case HWRef: {
            // throw exception only in debug mode. You don't want
            // Exceptions of this type in a production system.
            // You cannot check where it comes from then.
            // To check the origin of this exception you would need
            // a debugger attached or a call stack at hand.
            assert(
                d->flags & in_internal &&
                "An out-only param must not be a Ptr-type"); // will always be false!

            // do not force the type to be In, too, here, since the
            // parameter is likely to be defined in an out-vector of
            // an algorithm plugin and then it is strictly forbidden
            // to have pointer-like parameters there (beside string).

            // throw std::logic_error("It is not allowed to declare a parameter as OUT"
            //    "only for types DObjPtr, PointCloudPtr, PolygonMeshPtr or HWRef");

            break;
        }

        default:
            // well nothing to be done here
            break;
        }
    }
}

//-------------------------------------------------------------------------------------
int ParamBase::getLen() const
{
    switch (d->type)
    {
    case DoubleArray:
    case IntArray:
    case CharArray:
    case ComplexArray:
    case StringList:
        if (d->data.ptrVal)
        {
            return d->len;
        }
        else
        {
            return 0; // changed in itom 5.0 (was -1 before)
        }

    case String:
        if (d->data.ptrVal)
        {
            return d->len;
        }
        else
        {
            return -1;
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
    ByteArray newName = d->name;
    newName.append(suffix);
    return newName;
}

//-------------------------------------------------------------------------------------
// SET/GET FURTHER PROPERTIES
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
bool ParamBase::isNumeric(void) const
{
    static const int numericTypeMask =
        ito::ParamBase::Char | ParamBase::Int | ParamBase::Double | ParamBase::Complex;
    return ((d->type & numericTypeMask) > 0) && !(d->type & ito::ParamBase::Pointer);
}

//-------------------------------------------------------------------------------------
bool ParamBase::isNumericArray(void) const
{
    static const int numericTypeMask =
        ito::ParamBase::Char | ParamBase::Int | ParamBase::Double | ParamBase::Complex;
    return (d->type & numericTypeMask) && (d->type & ito::ParamBase::Pointer);
}

//-------------------------------------------------------------------------------------
bool ParamBase::isValid(void) const
{
    return d->type != 0;
}

//-------------------------------------------------------------------------------------
uint16 ParamBase::getType() const
{
    return d->type;
}


//-------------------------------------------------------------------------------------
uint32 ParamBase::getFlags() const
{
    return toFlagsExternal(d->flags);
}

//-------------------------------------------------------------------------------------
void ParamBase::setFlags(const uint32 flags)
{
    detach();
    d->flags = toFlagsInternal(flags);
}

//-------------------------------------------------------------------------------------
const char* ParamBase::getName(void) const
{
    return d->name.data();
}

//-------------------------------------------------------------------------------------
bool ParamBase::getAutosave(void) const
{
    return (getFlags() & NoAutosave) > 0;
}

//-------------------------------------------------------------------------------------
void ParamBase::setAutosave(const bool autosave)
{
    int32 f = getFlags();
    int32 fnew = autosave ? (f | NoAutosave) : (f & ~NoAutosave);

    if (f != fnew)
    {
        detach();
        setFlags(fnew);
    }
}


//--------------------------------------------------------------------------------------------
//  ASSIGNMENT AND OPERATORS
//--------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
/** braces operator
 *   @param [in] num array index for which the value should be returned
 *   @return     new tParam with values of ParamBase[num] in the array
 *
 *   returns the value of the index num from the array
 */
const ParamBase ParamBase::operator[](const int index) const
{
    auto type = getType();

    if ((type == CharArray) || (type == IntArray) || (type == DoubleArray) ||
        (type == ComplexArray) || (type == StringList))
    {
        if (index >= d->len || index < 0)
        {
            return ParamBase();
        }
        else
        {
            int len = 0;
            ito::ByteArray newName = getNameWithIndexSuffix(index);

            if (type == StringList)
            {
                const ByteArray* ba = getVal<const ByteArray*>(len);
                return ParamBase(newName, String, ba[index].data());
            }

            uint32 flags = getFlags();
            flags &= ~toFlagsInternal(NoAutosave); // remove "no autosave"

            switch (type & ~Pointer)
            {
            case Char:
                return ParamBase(newName, Char | flags, (getVal<const char*>(len))[index]);
                break;

            case Int:
                return ParamBase(newName, Int | flags, (getVal<const int32*>(len))[index]);
                break;

            case Double:
                return ParamBase(newName, Double | flags, (getVal<const float64*>(len))[index]);
                break;

            case Complex:
                return ParamBase(newName, Complex | flags, (getVal<const complex128*>(len))[index]);
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

//-------------------------------------------------------------------------------------
/** assignment operator
 *   @param [in] rhs ParamBase to copy from
 *   @return     new ParamBase with copied values
 *
 *   sets values of lhs to values of rhs ParamBase, strings are copied
 */
ParamBase& ParamBase::operator=(const ParamBase& rhs)
{
    if (this != &rhs)
    {
        incRef(rhs.d);
        decRefAndFree(d);
        d = rhs.d;
    }
    return *this;
}

//-------------------------------------------------------------------------------------
void ParamBase::detach(bool allocNewArray /*= true*/)
{
    if (d && d->ref > 0)
    {
        auto new_d = new Data(d->name);
        new_d->type = d->type;
        new_d->flags = d->flags;
        new_d->len = d->len;
        new_d->data = d->data; // in order to copy single numbers...

        if (d->data.ptrVal != nullptr)
        {
            if (allocNewArray)
            {
                switch (d->type)
                {
                case ParamBase::CharArray:
                    new_d->data.ptrVal = new char[d->len];
                    std::copy_n((const char*)d->data.ptrVal, d->len, (char*)new_d->data.ptrVal);
                    break;
                case ParamBase::IntArray:
                    new_d->data.ptrVal = new int32[d->len];
                    std::copy_n((const int32*)d->data.ptrVal, d->len, (int32*)new_d->data.ptrVal);
                    break;
                case ParamBase::DoubleArray:
                    new_d->data.ptrVal = new float64[d->len];
                    std::copy_n(
                        (const float64*)d->data.ptrVal, d->len, (float64*)new_d->data.ptrVal);
                    break;
                case ParamBase::ComplexArray:
                    new_d->data.ptrVal = new complex128[d->len];
                    std::copy_n(
                        (const complex128*)d->data.ptrVal, d->len, (complex128*)new_d->data.ptrVal);
                    break;
                case ParamBase::StringList: {
                    new_d->data.ptrVal = new ByteArray[d->len];
                    std::copy_n(
                        (const ByteArray*)d->data.ptrVal, d->len, (ByteArray*)new_d->data.ptrVal);
                    break;
                }
                case ParamBase::String:
                    new_d->data.ptrVal = new char[d->len + 1];
                    std::copy_n((const char*)d->data.ptrVal, d->len + 1, (char*)new_d->data.ptrVal);
                    break;
                }
            }
            else if (
                (d->type == ParamBase::CharArray) || (d->type == ParamBase::IntArray) ||
                (d->type == ParamBase::DoubleArray) || (d->type == ParamBase::ComplexArray) ||
                (d->type == ParamBase::StringList) || (d->type == ParamBase::String))
            {
                new_d->data.ptrVal = nullptr;
                new_d->len = 0;
            }
        }

        decRefAndFree(d);
        d = new_d;
    }
}

//-------------------------------------------------------------------------------------
ito::RetVal ParamBase::copyValueFrom(const ParamBase* rhs)
{
    if (getType() != rhs->getType())
    {
        return ito::RetVal(ito::retError, 0, "param types are not equal");
    }

    auto new_d = new Data(d->name);
    new_d->type = d->type;
    new_d->flags = d->flags;
    decRefAndFree(d);
    d = new_d;

    switch (d->type)
    {
    case Char:
        d->data.i8Val = rhs->d->data.i8Val;
        break;

    case Int:
        d->data.i32Val = rhs->d->data.i32Val;
        break;

    case Double:
        d->data.f64Val = rhs->d->data.f64Val;
        break;

    case Complex:
        d->data.c128Val = rhs->d->data.c128Val;
        break;

    case String:
        if (rhs->d->data.ptrVal)
        {
            d->len = rhs->d->len;
            d->data.ptrVal = new char[d->len + 1];
            std::copy_n((const char*)(rhs->d->data.ptrVal), d->len + 1, (char*)d->data.ptrVal);
        }
        break;

    case CharArray:
        if (rhs->d->data.ptrVal)
        {
            d->len = rhs->d->len;
            d->data.ptrVal = new char[d->len];
            std::copy_n((const char*)(rhs->d->data.ptrVal), d->len, (char*)d->data.ptrVal);
        }
        break;

    case IntArray:
        if (rhs->d->data.ptrVal)
        {
            d->len = rhs->d->len;
            d->data.ptrVal = new int32[d->len];
            std::copy_n((const int32*)(rhs->d->data.ptrVal), d->len, (int32*)d->data.ptrVal);
        }
        break;

    case DoubleArray:
        if (rhs->d->data.ptrVal)
        {
            d->len = rhs->d->len;
            d->data.ptrVal = new float64[d->len];
            std::copy_n((const float64*)(rhs->d->data.ptrVal), d->len, (float64*)d->data.ptrVal);
        }
        break;

    case ComplexArray:
        if (rhs->d->data.ptrVal)
        {
            d->len = rhs->d->len;
            d->data.ptrVal = new complex128[d->len];
            std::copy_n(
                (const complex128*)(rhs->d->data.ptrVal), d->len, (complex128*)d->data.ptrVal);
        }
        break;

    case StringList:
        if (rhs->d->data.ptrVal)
        {
            d->len = rhs->d->len;
            d->data.ptrVal = new ByteArray[d->len];
            std::copy_n(
                (const ByteArray*)(rhs->d->data.ptrVal), d->len, (ByteArray*)d->data.ptrVal);
        }
        break;

    case HWRef:
    case DObjPtr:
    case Pointer:
    case PointCloudPtr:
    case PointPtr:
    case PolygonMeshPtr:
        d->data.ptrVal = rhs->d->data.ptrVal;
        break;

    default:
        return ito::RetVal(ito::retError, 0, "unknown parameter type (ParamBase)");
        break;
    }

    return ito::RetVal(ito::retOk);
}

//-------------------------------------------------------------------------------------
Param::Param() : ParamBase(), m_pMetaShared(nullptr), m_info(nullptr)
{
}

//-------------------------------------------------------------------------------------
Param::Param(const ByteArray& name) : ParamBase(name), m_pMetaShared(nullptr), m_info(nullptr)
{
}

//-------------------------------------------------------------------------------------
Param::Param(const ByteArray& name, const uint32 typeAndFlags) :
    ParamBase(name, typeAndFlags), m_pMetaShared(nullptr), m_info(nullptr)
{
}

//-------------------------------------------------------------------------------------
Param::Param(const ByteArray& name, const uint32 typeAndFlags, const char* val, const char* info) :
    ParamBase(name, typeAndFlags, val), m_pMetaShared(nullptr), m_info(info)
{
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const char minVal,
    const char maxVal,
    const char val,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_info(info)
{
    assert((typeAndFlags & ParamBase::Char)); // use this constructor only for type character
    m_pMetaShared = new MetaShared(new CharMeta(minVal, maxVal));
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const int32 minVal,
    const int32 maxVal,
    const int32 val,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_info(info)
{
    assert((typeAndFlags & ParamBase::Int)); // use this constructor only for type integer
    m_pMetaShared = new MetaShared(new IntMeta(minVal, maxVal));
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const float64 minVal,
    const float64 maxVal,
    const float64 val,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_info(info)
{
    assert((typeAndFlags & ParamBase::Double)); // use this constructor only for type double
    m_pMetaShared = new MetaShared(new DoubleMeta(minVal, maxVal));
}


//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const char* values,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const int32* values,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const float64* values,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const complex128* values,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const ByteArray* values,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const char val,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const int32 val,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const float64 val,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const complex128 val,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, val),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const char* values,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const int32* values,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const float64* values,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const complex128* values,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}

//-------------------------------------------------------------------------------------
Param::Param(
    const ByteArray& name,
    const uint32 typeAndFlags,
    const unsigned int size,
    const ByteArray* values,
    ParamMeta* meta,
    const char* info) :
    ParamBase(name, typeAndFlags, size, values),
    m_pMetaShared(nullptr), m_info(info)
{
    setMeta(meta, true); // throws exception if meta does not fit to type
}


//-------------------------------------------------------------------------------------
Param::~Param()
{
    decRefMeta(m_pMetaShared);
    m_pMetaShared = nullptr;
}

//-------------------------------------------------------------------------------------
Param::Param(const Param& copyConstr) :
    ParamBase(copyConstr), m_pMetaShared(copyConstr.m_pMetaShared), m_info(copyConstr.m_info)
{
    incRefMeta(m_pMetaShared);
}

//-------------------------------------------------------------------------------------
Param::Param(Param&& rvalue) : ParamBase(rvalue), m_pMetaShared(nullptr)
{
    std::swap(m_pMetaShared, rvalue.m_pMetaShared);
    std::swap(m_info, rvalue.m_info);
}

//--------------------------------------------------------------------------------------------
//  ASSIGNMENT AND OPERATORS
//--------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
const Param Param::operator[](const int index) const
{
    auto type = getType();
    const ito::ParamMeta* pMeta = getMeta();

    if ((type == CharArray) || (type == IntArray) || (type == DoubleArray) ||
        (type == ComplexArray) || (type == StringList))
    {
        if (index >= getLen() || index < 0)
        {
            return Param();
        }
        else
        {
            ito::ByteArray newName = getNameWithIndexSuffix(index);
            int len;
            uint32 newSingleType = (uint32)type;
            newSingleType &= ~Pointer; // remove pointer
            uint32 flags = getFlags();
            flags &= ~toFlagsInternal(NoAutosave); // remove "no autosave"
            newSingleType |= flags;

            switch ((type & ~Pointer))
            {
            case Char: {
                CharMeta* cMeta = nullptr;

                if (pMeta && pMeta->getType() == ParamMeta::rttiCharArrayMeta)
                {
                    const CharArrayMeta* caMeta = static_cast<const CharArrayMeta*>(pMeta);
                    cMeta = new CharMeta(caMeta->getMin(), caMeta->getMax(), caMeta->getStepSize());
                }
                return Param(
                    newName.data(),
                    newSingleType,
                    (getVal<const char*>(len))[index],
                    cMeta,
                    m_info.data());
            }
            break;

            case Int: {
                IntMeta* iMeta = nullptr;

                if (pMeta &&
                    (pMeta->getType() == ParamMeta::rttiIntArrayMeta ||
                     pMeta->getType() == ParamMeta::rttiIntervalMeta ||
                     pMeta->getType() == ParamMeta::rttiRangeMeta))
                {
                    const IntMeta* iaMeta = static_cast<const IntMeta*>(pMeta);
                    iMeta = new IntMeta(*iaMeta);
                }

                // no conversion from RectMeta to single valued met
                return Param(
                    newName,
                    newSingleType,
                    (getVal<const int32*>(len))[index],
                    iMeta,
                    m_info.data());
            }
            break;

            case Double: {
                DoubleMeta* dMeta = nullptr;

                if (pMeta &&
                    (pMeta->getType() == ParamMeta::rttiDoubleIntervalMeta ||
                     pMeta->getType() == ParamMeta::rttiDoubleArrayMeta))
                {
                    const DoubleMeta* daMeta = static_cast<const DoubleMeta*>(pMeta);
                    dMeta = new DoubleMeta(*daMeta);
                }

                return Param(
                    newName,
                    newSingleType,
                    (getVal<const float64*>(len))[index],
                    dMeta,
                    m_info.data());
            }
            break;

            case Complex: {
                // complex has no meta, since no min or max comparison is defined for complex values
                return Param(
                    newName,
                    newSingleType,
                    (getVal<complex128*>(len))[index],
                    nullptr,
                    m_info.data());
            }
            break;

            case StringList & ~Pointer: {
                // no meta up to now
                return Param(
                    newName,
                    String | flags,
                    (getVal<const ByteArray*>(len))[index].data(),
                    m_info.data());
            }

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

//-------------------------------------------------------------------------------------
Param& Param::operator=(const Param& rhs)
{
    if (this != &rhs)
    {
        ParamBase::operator=(rhs);
        m_info = rhs.m_info;
        m_pMetaShared = rhs.m_pMetaShared;
        incRefMeta(m_pMetaShared);
    }

    return *this;
}

//-------------------------------------------------------------------------------------
Param& Param::operator=(Param&& rvalue) noexcept
{
    if (this != &rvalue)
    {
        ParamBase::operator=(std::move(rvalue));
        std::swap(m_info, rvalue.m_info);
        std::swap(m_pMetaShared, rvalue.m_pMetaShared);
    }

    return *this;
}

//-------------------------------------------------------------------------------------
ito::RetVal Param::copyValueFrom(const ParamBase* rhs)
{
    return ParamBase::copyValueFrom(rhs);
}

//-------------------------------------------------------------------------------------
void Param::detachMeta()
{
    if (m_pMetaShared)
    {
        // make a copy of the current meta
        setMeta(m_pMetaShared->meta, false);
    }
}

//-------------------------------------------------------------------------------------
void Param::setMeta(ParamMeta* meta, bool takeOwnership)
{
    MetaShared* oldMetaShared = m_pMetaShared;
    m_pMetaShared = nullptr;

    if (meta)
    {
        ito::ParamMeta::MetaRtti metaType = meta->getType();

#if _DEBUG
        bool valid = false;
        ito::uint16 ptype = getType();

        switch (metaType)
        {
        case ParamMeta::rttiCharMeta:
            if (ptype == ito::ParamBase::Char)
                valid = true;
            break;
        case ParamMeta::rttiIntMeta:
            if (ptype == ito::ParamBase::Int)
                valid = true;
            break;
        case ParamMeta::rttiDoubleMeta:
            if (ptype == ito::ParamBase::Double)
                valid = true;
            break;
        case ParamMeta::rttiStringMeta:
            if (ptype == (ito::ParamBase::String))
                valid = true;
            break;
        case ParamMeta::rttiDObjMeta:
            if (ptype == (ito::ParamBase::DObjPtr))
                valid = true;
            break;
        case ParamMeta::rttiHWMeta:
            if (ptype == (ito::ParamBase::HWRef))
                valid = true;
            break;
        case ParamMeta::rttiCharArrayMeta:
            if (ptype == (ito::ParamBase::CharArray))
                valid = true;
            break;
        case ParamMeta::rttiIntArrayMeta:
        case ParamMeta::rttiIntervalMeta:
        case ParamMeta::rttiRangeMeta:
        case ParamMeta::rttiRectMeta:
            if (ptype == (ito::ParamBase::IntArray))
                valid = true;
            break;
        case ParamMeta::rttiDoubleArrayMeta:
        case ParamMeta::rttiDoubleIntervalMeta:
            if (ptype == (ito::ParamBase::DoubleArray))
                valid = true;
            break;
        case ParamMeta::rttiStringListMeta:
            if (ptype == (ito::ParamBase::StringList))
                valid = true;
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
            // Param takes ownership of meta
            m_pMetaShared = new MetaShared(meta);
        }
        else
        {
            ito::ParamMeta* newMeta = nullptr;

            switch (metaType)
            {
            case ParamMeta::rttiCharMeta:
                newMeta = new CharMeta(*(CharMeta*)(meta));
                break;
            case ParamMeta::rttiIntMeta:
                newMeta = new IntMeta(*(IntMeta*)(meta));
                break;
            case ParamMeta::rttiDoubleMeta:
                newMeta = new DoubleMeta(*(DoubleMeta*)(meta));
                break;
            case ParamMeta::rttiStringMeta:
                newMeta = new StringMeta(*(StringMeta*)(meta));
                break;
            case ParamMeta::rttiDObjMeta:
                newMeta = new DObjMeta(*(DObjMeta*)(meta));
                break;
            case ParamMeta::rttiHWMeta:
                newMeta = new HWMeta(*(HWMeta*)(meta));
                break;
            case ParamMeta::rttiCharArrayMeta:
                newMeta = new CharArrayMeta(*(CharArrayMeta*)(meta));
                break;
            case ParamMeta::rttiIntArrayMeta:
                newMeta = new IntArrayMeta(*(IntArrayMeta*)(meta));
                break;
            case ParamMeta::rttiDoubleArrayMeta:
                newMeta = new DoubleArrayMeta(*(DoubleArrayMeta*)(meta));
                break;
            case ParamMeta::rttiIntervalMeta:
                newMeta = new IntervalMeta(*(IntervalMeta*)(meta));
                break;
            case ParamMeta::rttiRangeMeta:
                newMeta = new RangeMeta(*(RangeMeta*)(meta));
                break;
            case ParamMeta::rttiDoubleIntervalMeta:
                newMeta = new DoubleIntervalMeta(*(DoubleIntervalMeta*)(meta));
                break;
            case ParamMeta::rttiRectMeta:
                newMeta = new RectMeta(*(RectMeta*)(meta));
                break;
            case ParamMeta::rttiStringListMeta:
                newMeta = new StringListMeta(*(StringListMeta*)(meta));
                break;
            default:
                throw std::logic_error(
                    "Type of meta [ParamMeta] is unknown and cannot not be copied or assigned.");
            }

            if (newMeta)
            {
                m_pMetaShared = new MetaShared(newMeta);
            }
        }
    }

    // remove old shared meta
    decRefMeta(oldMetaShared);
}

//-------------------------------------------------------------------------------------
/** returns minimum value of parameter if this is available and exists.
 *
 *   This method is a wrapper method for ((ito::IntMeta*)getMeta())->getMax()...
 *   and returns the minimum value of the underlying meta information. It only
 *   returns a valid value for meta structures of type char, charArray, int, intArray, interval,
 * range, double, doubleMeta.
 *
 *   @return     minimum value of -Inf maximum does not exist
 */
float64 Param::getMin() const
{
    if (m_pMetaShared && m_pMetaShared->meta)
    {
        switch (m_pMetaShared->meta->getType())
        {
        case ParamMeta::rttiCharMeta:
        case ParamMeta::rttiCharArrayMeta:
            return static_cast<const CharMeta*>(m_pMetaShared->meta)->getMin();
        case ParamMeta::rttiIntMeta:
        case ParamMeta::rttiIntArrayMeta:
        case ParamMeta::rttiIntervalMeta:
        case ParamMeta::rttiRangeMeta:
            return static_cast<const IntMeta*>(m_pMetaShared->meta)->getMin();
        case ParamMeta::rttiDoubleMeta:
        case ParamMeta::rttiDoubleArrayMeta:
            return static_cast<const DoubleMeta*>(m_pMetaShared->meta)->getMin();
        }
    }
    return -std::numeric_limits<float64>::infinity();
}

//-------------------------------------------------------------------------------------
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
    if (m_pMetaShared && m_pMetaShared->meta)
    {
        switch (m_pMetaShared->meta->getType())
        {
        case ParamMeta::rttiCharMeta:
        case ParamMeta::rttiCharArrayMeta:
            return static_cast<const CharMeta*>(m_pMetaShared->meta)->getMax();
        case ParamMeta::rttiIntMeta:
        case ParamMeta::rttiIntArrayMeta:
        case ParamMeta::rttiIntervalMeta:
        case ParamMeta::rttiRangeMeta:
            return static_cast<const IntMeta*>(m_pMetaShared->meta)->getMax();
        case ParamMeta::rttiDoubleMeta:
        case ParamMeta::rttiDoubleArrayMeta:
            return static_cast<const DoubleMeta*>(m_pMetaShared->meta)->getMax();
        }
    }
    return std::numeric_limits<float64>::infinity();
}

}; // end namespace ito
