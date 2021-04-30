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

#pragma once

/* includes */

#include "byteArray.h"
#include "commonGlobal.h"
#include "paramMeta.h"
#include "retVal.h"
#include "typeDefs.h"

#include <limits>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>

/* definition and macros */
/* global variables (avoid) */
/* content */

namespace ito {
class Param;
template <typename _Tp> struct ItomParamHelper;

const uint32 paramFlagMask = 0xFFFF0000; //!< bits of type lying within this mask are flags (e.g.
                                         //!< typeNoAutosave, typeReadonly...)
const uint32 paramTypeMask = 0x0000FFFF; //!< bits of param type lying withing this mask describe
                                         //!< the type (typeNoAutosave must be included there)

//! wrapper class for a complex128 value. This class is used, since the std::complex stl class is
//! not exported over DLLs
struct complex128_
{
    float64 real;
    float64 imag;
};

union ParamBaseData
{
    //!< 1 byte
    int8 i8Val;

    //!< 4 bytes
    int32 i32Val;

    //!< 8 bytes
    float64 f64Val;

    //!< 16 bytes
    complex128_ c128Val;

    //!< 8 bytes
    void* ptrVal;
};

class ITOMCOMMON_EXPORT ParamBase
{
protected:
    void inOutCheck();

private:
    struct Data
    {
        Data(const ByteArray &name_ = "") :
            ref(0),
            flags(0),
            type(0),
            len(0),
            name(name_)
        {
            memset(&data, 0, sizeof(ParamBaseData));
        }

        std::atomic_int ref;               /*!< reference counter for implicit sharing (0: means one reference, ...) */

        ParamBaseData data;

        uint32 len;

        //!< flags, bitmask of the higher level values in ParamBase::Flag
        uint16 flags;

        //!< type, correspond to ParamBase::Type
        uint16 type;

        //!< name of this parameter
        ByteArray name;
    };

    mutable Data *d;

    void freeMemory(Data *data); //!< free allocated memory, if memory has been allocated.

    //!< if data is currently shared with another object, detach this
    //!< data d from the shared ones by making a copy. If this data is not
    //!< shared, this method is a noop.
    void detach();

    inline void decRefAndFree(Data *x)
    {
        if (x && (!(x->ref--)))
        {
            freeMemory(x);
            delete (x);
            x = nullptr;
        }
    }

    inline void incRef(Data *x)
    {
        if (x)
        {
            (x->ref)++;
        }
    }

    //!< depending on the type, set the default value for the autosave flag.
    void setDefaultAutosave();

    template <typename _Tp> friend struct ItomParamHelper;

public:
    //!< Flag section, new for itom > 4.1. Before it was part of the Type enumeration.
    /* For compatibility reasons with older versions of itom, values in this enumeration
       must only set bits in the range 17-32 of an uint32 value, since they can be 
       used together with values of the Type enum below, using bits 1-16. 
       Internally, the flags are stored as uint16 object, where these enumeration
       values are shifted by 16bits. 
    */
    enum Flag
    {
        /* If this bit is set, this parameter should be automatically
        stored if for instance a plugin is closed and if the plugin
        with the same identifier is opened again, the parameter
        value will be tried to be reconstructed from the stored value. */
        NoAutosave = 0x010000,

        /* Flag to define this parameter to be readonly. It cannot for instance not be changed
        from Python. Internally, the setVal method does not check for Readonly.
        This has to be programmed manually. Plugins can also set or unset this flag
        during the runtime. */
        Readonly = 0x020000,

        /* Flag to mark this parameter as input value only.
        If a plugin defines this flag without the Out-flag, it pretends
        to only consume the given value, but the consumer will not change the value at any time.
        If the Out-flag is set, too, the consumer (e.g. plugin) will read and write the
        value, such that for instance a given dataObject will have another content after having
        called a method of a plugin. Return parameters of a method must never have the In-flag set.
        */
        In = 0x040000,

        /* Flag to mark this parameter, that the consumer will create or change
        the value of this parameter.
        This can be set together with In (or alone). All return parameters of an method in
        an algorithm plugin, must have this flag set. For other mandatory or optional
        input parameter, this flag can only be set together with In and means then,
        that the content of the given parameter will be changed by the called method. */
        Out = 0x080000,

        /* Flag to mark a parameter to be temporarily not available.
        Plugins can also set or unset this flag
        during the runtime. */
        NotAvailable = 0x100000,
    };

    enum Type
    {
        /* Helper-bit for all types that only contain a pointer to the real value and
        not the value itself. For these types, the caller need to keep the pointed object
        until both caller and called method do not use it any more! */
        Pointer = 0x0001,

        //!< character (int8) parameter
        Char = 0x0002,

        //!< integer (int32) parameter
        Int = 0x0004,

        //!< double (float64) parameter
        Double = 0x0008,

        //!< complex (complex128) parameter
        Complex = 0x0400,

        //!< dataObject parameter (pointer, no auto-save possible)
        DObjPtr = 0x0010 | Pointer,

        //!< string parameter
        String = 0x0020 | Pointer,

        //!< reference to another plugin instance (pointer, no auto-save possible)
        HWRef = 0x0040 | Pointer,

        //!< array of characters
        CharArray = Char | Pointer,

        //!< array of integers
        IntArray = Int | Pointer,

        //!< array of doubles
        DoubleArray = Double | Pointer,

        //!< array of complex numbers
        ComplexArray = Complex | Pointer,

        //!< point cloud parameter (pointer, no auto-safe possible)
        PointCloudPtr = 0x0080 | Pointer,

        //!< point parameter (pointer, no auto-safe possible)
        PointPtr = 0x0100 | Pointer,

        //!< polygon mesh parameter (pointer, no auto-safe possible)
        PolygonMeshPtr = 0x0200 | Pointer,

        //!< list of strings, given as ito::ByteArray
        StringList = 0x0800 | Pointer
    };


    //--------------------------------------------------------------------------------------------
    //  CONSTRUCTORS, COPY-CONSTRUCTOR, DESTRUCTOR
    //--------------------------------------------------------------------------------------------
    //! default constructor, creates "empty" ParamBase
    ParamBase();

    // type-less ParamBase with name only
    ParamBase(const ByteArray& name);

    // constructor with type and name
    ParamBase(const ByteArray& name, const uint32 typeAndFlags);

    // constructor with name and type and char val
    ParamBase(const ByteArray& name, const uint32 typeAndFlags, const char* val);

    // constructor with name and type and float64 val
    ParamBase(const ByteArray& name, const uint32 typeAndFlags, const float64 val);

    // constructor with name and type and int32 val
    ParamBase(const ByteArray& name, const uint32 typeAndFlags, const int32 val);

    // constructor with name and type and complex128 val
    ParamBase(const ByteArray& name, const uint32 typeAndFlags, const complex128 val);

    // array constructor with name and type, size and array
    ParamBase(
        const ByteArray& name, const uint32 typeAndFlags, const uint32 size, const char* values);

    // array constructor with name and type, size and array
    ParamBase(
        const ByteArray& name, const uint32 typeAndFlags, const uint32 size, const int32* values);

    // array constructor with name and type, size and array
    ParamBase(
        const ByteArray& name, const uint32 typeAndFlags, const uint32 size, const float64* values);

    // array constructor with name and type, size and array
    ParamBase(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const uint32 size,
        const complex128* values);

    // array constructor with name and type, size and string list
    ParamBase(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const uint32 size,
        const ByteArray* values);

    virtual ~ParamBase(); // Destructor

    //!< copy constructor
    ParamBase(const ParamBase& other);

    //!< copy constructor for rvalues
    ParamBase(ParamBase &&other) noexcept;

    //--------------------------------------------------------------------------------------------
    //  ASSIGNMENT AND OPERATORS
    //--------------------------------------------------------------------------------------------

    //!< braces operator for element-wise access in arrays
    const ParamBase operator[](const int index) const;

    //!< assignment operator (sets values of lhs to
    //!< values of rhs Param, strings are copied)
    ParamBase& operator=(const ParamBase& rhs);

    inline ParamBase &operator=(ParamBase &&other) noexcept
    {
        std::swap(d, other.d);
        return *this;
    }

    //!< just copies the value from the right-hand-side tParam (rhs) to this tParam.
    ito::RetVal copyValueFrom(const ito::ParamBase* rhs);

    //--------------------------------------------------------------------------------------------
    // COMPARISON OPERATORS (two values are equal if both their type and the content is equal)
    //--------------------------------------------------------------------------------------------
    bool operator==(const ParamBase& rhs) const;

    inline bool operator!=(const ParamBase& rhs) const
    {
        return !(*this == rhs);
    }

    //--------------------------------------------------------------------------------------------
    //  SET/GET FURTHER PROPERTIES
    //--------------------------------------------------------------------------------------------

    //! returns true if Param is of type char, int, double or complex
    bool isNumeric(void) const;

    //! returns true if Param is of type char array, int array, double array or complex array
    bool isNumericArray(void) const;

    //! returns whether Param contains a valid type (true) or is an empty parameter (false, type ==
    //! 0). The default tParam-constructor is always an invalid tParam.
    bool isValid(void) const;

    //! returns parameter type
    uint16 getType() const;

    //! returns parameter flags
    uint32 getFlags() const;

    //! sets parameter flags for possible flags see \ref tParamType
    void setFlags(const uint32 flags);

    //! returns parameter name (returned string is no copy, do not delete it)
    const char* getName(void) const;

    //! return the name, where an integer index is appended with bracket squares, e.g. myArray
    //! becomes myArray[2].
    ito::ByteArray getNameWithIndexSuffix(int index) const;

    //! returns content of autosave flag - this flag determines whether the parameter value gets
    //! automagically saved to xml file when an instance of a plugin class is deleted (closed)
    bool getAutosave() const;

    //! sets content of autosave flag - this flag determines whether the parameter value gets
    //! automagically saved to xml file when an instance of a plugin class is deleted (closed)
    void setAutosave(const bool autosave);

    //! returns length of array parameters or 0 if no array is given. For string parameter returns
    //! length of string or 0 if not given, for number parameters return 1. In all other cases -1.
    /* From itom 5.0 the behaviour in case of arrays or string list changed. Before, -1 was
       returned if the internal array is a nullptr and 0 if the array has zero items. From itom
       5.0 on, an empty or non-existing array or list always returns 0.
    */
    int getLen(void) const;

    /** setVal  set parameter value - templated version
     *   @param [in] val  value to set to
     *   @return     RetVal with operation status
     *   sets the parameter value to the passed value, if the parameter type is inadequate it is set
     * to the maximum value of template type
     */
    template <typename _Tp> inline ito::RetVal setVal(_Tp val)
    {
        return ItomParamHelper<_Tp>::setVal(this, val, 0);
    }

    /** setVal  set parameter value - templated version for arrays
     *   @param [in] val  value to set to
     *   @param [in] len  length of array
     *   @return     RetVal with operation status
     *   sets the parameter value to the passed value, if the length is below 1 or a Null pointer is
     * passed an error is returned
     */
    template <typename _Tp> inline ito::RetVal setVal(_Tp val, int len)
    {
        return ItomParamHelper<_Tp>::setVal(this, val, len);
    }

    /** getVal  read parameter value - templated version
     *   @return parameter value (numeric, casted)
     *
     *   returns the actual parameter value casted to the template parameter type. If the tParam has
     * a non numeric type the largest value for the template type is passed.
     */
    template <typename _Tp> inline _Tp getVal() const
    {
        int len = 0;
        return ItomParamHelper<_Tp>::getVal(this, len);
    }

    /** getVal  read parameter value - templated version for arrays
     *   @param [out] len  length of array
     *
     *   returns the actual parameter value casted to the template parameter type. In 'len' is
     * returned what is supposed to be the length of the array. As only array references are used
     * within tParam the actual size may differ.
     */
    template <typename _Tp> inline _Tp getVal(int& len) const
    {
        return ItomParamHelper<_Tp>::getVal(this, len);
    }
};

//----------------------------------------------------------------------------------------------------------------------------------
/** @class Param
 *   @brief  class for parameter handling e.g. to pass paramters to plugins
 *
 *   The plugins use this class to organize their parameters (internally) and for the paramList
 * which is used for type checking whilst parsing parameters passed from python to c.
 */
class ITOMCOMMON_EXPORT Param : public ParamBase
{
private:
    ParamMeta* m_pMeta;
    ByteArray m_info;

public:
    //--------------------------------------------------------------------------------------------
    //  CONSTRUCTORS, COPY-CONSTRUCTOR, DESTRUCTOR
    //--------------------------------------------------------------------------------------------
    //! default constructor, creates "empty" Param
    /*
    This parameter has no documentation string and no meta information.
    The type is 0 (invalid).
    The name is empty.
    */
    Param() : ParamBase(), m_pMeta(nullptr), m_info(nullptr)
    {
    }

    //!< type-less Param with name only
    /*
    This parameter has no documentation string and no meta information.
    The type is 0 (invalid).

    \param name is the name of the parameter
    */
    Param(const ByteArray& name) : ParamBase(name), m_pMeta(nullptr), m_info(nullptr)
    {
    }

    //!< type-less Param with name and type
    /*
    This parameter has no documentation string and no meta information.

    \param name is the name of the parameter
    \type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
    e.g. ito::ParamBase::Char | ito::ParamBase::In for a read-only input parameter
    */
    Param(const ByteArray& name, const uint32 typeAndFlags) :
        ParamBase(name, typeAndFlags), m_pMeta(nullptr), m_info(nullptr)
    {
    }

    //!< Constructor for a string value (const char*)
    /*
    This parameter has no meta information. They can be added using setMetaInfo.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::String | ito::ParamBase::In for a read-only input parameter
    \param val is the default string (const char*) of this parameter
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(const ByteArray& name, const uint32 typeAndFlags, const char* val, const char* info);

    //!< Constructor for a char value parameter (int8)
    /*
    This parameter will automatically get meta information of type ito::CharMeta with a minimum and
    maximum value.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::Char | ito::ParamBase::In for a read-only input parameter
    \param minVal is the minimum allowed value (added to meta information)
    \param maxVal is the maximum allowed value (added to meta information)
    \param val is the default int8 value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const char minVal,
        const char maxVal,
        const char val,
        const char* info);

    //!< Constructor for an integer value parameter (int32)
    /*
    This parameter will automatically get meta information of type ito::IntMeta with a minimum and
    maximum value.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
    e.g. ito::ParamBase::Int | ito::ParamBase::In for a read-only input parameter
    \param minVal is the minimum allowed value (added to meta information)
    \param maxVal is the maximum allowed value (added to meta information)
    \param val is the default int32 value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const int32 minVal,
        const int32 maxVal,
        const int32 val,
        const char* info);

    //!< Constructor for an double value parameter (float64)
    /*
    This parameter will automatically get meta information of type ito::DoubleMeta with a minimum
    and maximum value.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
    e.g. ito::ParamBase::Double | ito::ParamBase::In for a read-only input parameter
    \param minVal is the minimum allowed value (added to meta information)
    \param maxVal is the maximum allowed value (added to meta information)
    \param val is the default float64 value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const float64 minVal,
        const float64 maxVal,
        const float64 val,
        const char* info);

    //!< Constructor for a character array parameter (int8)
    /*
    This parameter has no meta information. They can be added using setMetaInfo.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::CharArray | ito::ParamBase::In for a read-only input parameter
    \param size is the size of the given default values array
    \param values is the default int8 array value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const char* values,
        const char* info); // array constructor with name and type, size and array

    //!< Constructor for an integer array parameter (int32)
    /*
    This parameter has no meta information. They can be added using setMetaInfo.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::IntArray | ito::ParamBase::In for a read-only input parameter
    \param size is the size of the given default values array
    \param values is the default int32 array value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const int32* values,
        const char* info);

    //!< Constructor for a double array parameter (float64)
    /*
    This parameter has no meta information. They can be added using setMetaInfo.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::DoubleArray | ito::ParamBase::In for a read-only input parameter
    \param size is the size of the given default values array
    \param values is the default float64 array value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const float64* values,
        const char* info);

    //!< Constructor for a complex128 array parameter
    /*
    This parameter has no meta information.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::ComplexArray | ito::ParamBase::In for a read-only input parameter
    \param size is the size of the given default values array
    \param values is the default complex array value
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const complex128* values,
        const char* info);

    //!< Constructor for a string list parameter
    /*
    This parameter has no meta information.

    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::ComplexArray | ito::ParamBase::In for a read-only input parameter
    \param size is the size of the given default values array
    \param values is the default ByteArray list
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const ByteArray* values,
        const char* info);

    //!< constructor for a character value (int8)
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::Char | ito::ParamBase::In for a read-only input parameter
    \param val is the default value
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, it has to be an object of ito::CharMeta.
        The ownership is taken by this parameter!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const char val,
        ParamMeta* meta,
        const char* info);

    //!< constructor for an integer value (int32)
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::Int | ito::ParamBase::In for a read-only input parameter
    \param val is the default value
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, it has to be an object of ito::IntMeta.
        The ownership is taken by this parameter!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const int32 val,
        ParamMeta* meta,
        const char* info);

    //!< constructor for a double value (float64)
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::Double | ito::ParamBase::In for a read-only input parameter
    \param val is the default value
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, it has to be an object of ito::DoubleMeta.
        The ownership is taken by this parameter!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const float64 val,
        ParamMeta* meta,
        const char* info);

    //!< constructor for a complex value (complex128)
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
    e.g. ito::ParamBase::Complex | ito::ParamBase::In for a read-only input parameter
    \param val is the default value
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        Currently, there is no meta information class for complex value parameters!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const complex128 val,
        ParamMeta* meta,
        const char* info);

    //!< constructor for int32 arrays.
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::CharArray | ito::ParamBase::In for a read-only input parameter
    \param size is the length of the default array, passed to values
    \param values is the pointer to the default array of values
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, it has to be an object of ito::CharArrayMeta.
        The ownership is taken by this parameter!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const char* values,
        ParamMeta* meta,
        const char* info);

    //!< constructor for int32 arrays.
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::IntArray | ito::ParamBase::In for a read-only input parameter
    \param size is the length of the default array, passed to values
    \param values is the pointer to the default array of values
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, it has to be an object of ito::IntArrayMeta, ito::IntervalMeta,
        ito::RangeMeta or ito::RectMeta.
        The ownership is taken by this parameter!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const int32* values,
        ParamMeta* meta,
        const char* info);

    //!< constructor for float64 arrays.
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::DoubleArray | ito::ParamBase::In for a read-only input parameter
    \param size is the length of the default array, passed to values
    \param values is the pointer to the default array of values
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, it has to be an object of ito::DoubleArrayMeta or
    ito::DoubleIntervalMeta. The ownership is taken by this parameter! \param info can be a
    documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const float64* values,
        ParamMeta* meta,
        const char* info);

    //!< constructor for complex128 arrays.
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::ComplexArray | ito::ParamBase::In for a read-only input parameter
    \param size is the length of the default array, passed to values
    \param values is the pointer to the default array of values
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, the ownership is taken by this parameter!
        Currently, there is no meta information class available for complex array parameters!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const complex128* values,
        ParamMeta* meta,
        const char* info);

    //!< constructor for a string list.
    /*
    \param name is the name of the parameter
    \param type is a flag mask, that might consists of combinations of ito::ParamBase::Type,
        e.g. ito::ParamBase::ComplexArray | ito::ParamBase::In for a read-only input parameter
    \param size is the length of the default array, passed to values
    \param values is the pointer to the default string list values
    \param meta might be nullptr, if no meta information should be passed to this parameter.
        If a pointer is given, the ownership is taken by this parameter!
        Currently, there is no meta information class available for complex array parameters!
    \param info can be a documentation string for this parameter, an empty string or nullptr
    */
    Param(
        const ByteArray& name,
        const uint32 typeAndFlags,
        const unsigned int size,
        const ByteArray* values,
        ParamMeta* meta,
        const char* info);

    //!< Destructor
    ~Param();

    //!< Copy-Constructor
    Param(const Param& copyConstr);

    //--------------------------------------------------------------------------------------------
    //  ASSIGNMENT AND OPERATORS
    //--------------------------------------------------------------------------------------------

    //!< braces operator for element-wise access in arrays
    const Param operator[](const int index) const;

    //!< assignment operator (sets values of lhs to values of
    //!< rhs Param, strings are copied)
    Param& operator=(const Param& rhs);

    //!< just copies the value from the right-hand-side ParamBase (rhs)
    //!< to this tParam.
    ito::RetVal copyValueFrom(const ParamBase* rhs);

    //--------------------------------------------------------------------------------------------
    //  SET/GET FURTHER PROPERTIES
    //--------------------------------------------------------------------------------------------
    //!< returns content of info string (string is not copied)
    inline const char* getInfo(void) const
    {
        return m_info.data();
    }

    //!< sets content of info string, if necessary the info buffer is freed first, passed string is
    //!< copied
    inline void setInfo(const char* info)
    {
        m_info = info;
    }

    inline void setInfo(const ByteArray& info)
    {
        m_info = info;
    }

    //!< returns const-pointer to meta-information instance or nullptr if not available. Cast this
    //!< pointer to the right class of the parameter.
    inline const ParamMeta* getMeta(void) const
    {
        return m_pMeta;
    }

    //!< returns pointer to meta-information instance or nullptr if not available. Cast this pointer
    //!< to the right class of the parameter.
    inline ParamMeta* getMeta(void)
    {
        return m_pMeta;
    }

    //!< returns const-pointer to meta-information instance casted to 'const _Tp*' or nullptr if not
    //!< available or cast failed.
    /*
    Example: intParam.getMetaT<ito::IntMeta>();
    */
    template <typename _Tp> inline const _Tp* getMetaT(void) const
    {
        return static_cast<const _Tp*>(m_pMeta);
    }

    //!< returns pointer to meta-information instance casted to '_Tp*' or nullptr if not available
    //!< or cast failed.
    /*
    Example: intParam.getMetaT<ito::IntMeta>();
    */
    template <typename _Tp> inline _Tp* getMetaT(void)
    {
        return static_cast<_Tp*>(m_pMeta);
    }

    //! sets a new ParamMeta-instance as meta information for this Param
    /*!
        \param meta is the pointer to any instance derived from ParamMeta
        \param takeOwnership (default: false) defines if this Param should take
            the ownership of the given meta object. If false, a deep copy
            of the meta object is stored in this param.
        \sa ito::ParamMeta
    */
    void setMeta(ParamMeta* meta, bool takeOwnership = false);

    float64 getMin() const;
    float64 getMax() const;
};

//---------------------------------------------------------------------------------------------------------------------
template <typename _Tp> struct ItomParamHelper
{
    static ito::RetVal setVal(
        ito::ParamBase *param, const _Tp val, int len = 0)
    {
        static_assert(std::is_pointer<_Tp>::value, "invalid template type");

        param->detach();

        switch (param->d->type)
        {
        case ito::ParamBase::HWRef:
        case ito::ParamBase::DObjPtr:
        case ito::ParamBase::PointCloudPtr:
        case ito::ParamBase::PointPtr:
        case ito::ParamBase::PolygonMeshPtr:
            param->d->data.ptrVal = (void*)(reinterpret_cast<const void*>(val));
            return ito::retOk;

        case ito::ParamBase::String: {
            auto cVal_ = param->d->data.ptrVal;
            if (val)
            {
                size_t len = strlen(reinterpret_cast<const char*>(val));
                param->d->data.ptrVal =new char[len + 1];
                memcpy(param->d->data.ptrVal, val, len + 1);
                param->d->len = static_cast<int>(strlen((char*)param->d->data.ptrVal));
            }
            else
            {
                param->d->data.ptrVal =0;
                param->d->len = -1;
            }

            if (cVal_)
            {
                delete[](char*) cVal_;
            }
        }
            return ito::retOk;

        case ito::ParamBase::CharArray: {
            auto cVal_ = param->d->data.ptrVal;
            if ((val) && (len > 0))
            {
                param->d->data.ptrVal = new char[len];
                memcpy(param->d->data.ptrVal, val, len * sizeof(char));
                param->d->len = len;
            }
            else
            {
                param->d->data.ptrVal = nullptr;
                param->d->len = -1;
            }

            if (cVal_)
            {
                delete[](char*) cVal_;
            }
        }
            return ito::retOk;

        case ito::ParamBase::IntArray: {
            auto cVal_ = param->d->data.ptrVal;
            if ((val) && (len > 0))
            {
                param->d->data.ptrVal = new ito::int32[len];
                memcpy(param->d->data.ptrVal, val, len * sizeof(ito::int32));
                param->d->len = len;
            }
            else
            {
                param->d->data.ptrVal =nullptr;
                param->d->len = -1;
            }

            if (cVal_)
            {
                delete[](ito::int32*) cVal_;
            }
        }
            return ito::retOk;

        case ito::ParamBase::DoubleArray: {
            auto cVal_ = param->d->data.ptrVal;
            if ((val) && (len > 0))
            {
                param->d->data.ptrVal =new ito::float64[len];
                memcpy(param->d->data.ptrVal, val, len * sizeof(ito::float64));
                param->d->len = len;
            }
            else
            {
                param->d->data.ptrVal =nullptr;
                param->d->len = -1;
            }

            if (cVal_)
            {
                delete[](ito::float64*) cVal_;
            }
        }
            return ito::retOk;

        case ito::ParamBase::ComplexArray: {
            auto cVal_ = param->d->data.ptrVal;

            if ((val) && (len > 0))
            {
                param->d->data.ptrVal =new ito::complex128[len];
                memcpy(param->d->data.ptrVal, val, len * sizeof(ito::complex128));
                param->d->len = len;
            }
            else
            {
                param->d->data.ptrVal =nullptr;
                param->d->len = -1;
            }

            if (cVal_)
            {
                delete[](ito::complex128*) cVal_;
            }
        }
            return ito::retOk;

        case ito::ParamBase::StringList: {
            auto cVal_ = param->d->data.ptrVal;

            if ((val) && (len > 0))
            {
                ito::ByteArray* dest = new ito::ByteArray[len];
                param->d->data.ptrVal =dest;
                const ito::ByteArray* src = (const ito::ByteArray*)val;

                for (int i = 0; i < len; ++i)
                {
                    dest[i] = src[i]; // operator=
                }

                param->d->len = len;
            }
            else
            {
                param->d->data.ptrVal =nullptr;
                param->d->len = -1;
            }

            if (cVal_)
            {
                delete[](ito::ByteArray*) cVal_;
            }
        }
            return ito::retOk;

        default:
            return ito::RetVal(
                ito::retError,
                0,
                "_Tp parameter of setVal<_Tp> does not match the type of the parameter");
        }
    }

    static _Tp getVal(
        const ito::ParamBase *param,
        int& len)
    {
        static_assert(std::is_pointer<_Tp>::value, "invalid template type");

        if (std::is_pointer<_Tp>::value)
        {
            const_cast<ito::ParamBase*>(param)->detach();
        }

        switch (param->d->type)
        {
        case ito::ParamBase::String:
            if (param->d->data.ptrVal)
            {
                len = static_cast<int>(strlen((const char*)param->d->data.ptrVal));
                return reinterpret_cast<_Tp>((char*)param->d->data.ptrVal);
            }
            else
            {
                len = 0;
                return 0;
            }

        case ito::ParamBase::CharArray:
        case ito::ParamBase::IntArray:
        case ito::ParamBase::DoubleArray:
        case ito::ParamBase::ComplexArray:
        case ito::ParamBase::StringList:
            if (param->d->data.ptrVal)
            {
                len = param->d->len;
                return reinterpret_cast<_Tp>((char*)param->d->data.ptrVal);
            }
            else
            {
                len = 0;
                return 0;
            }

        case ito::ParamBase::HWRef:
        case ito::ParamBase::DObjPtr:
        case ito::ParamBase::PointCloudPtr:
        case ito::ParamBase::PointPtr:
        case ito::ParamBase::PolygonMeshPtr:
            return reinterpret_cast<_Tp>((char*)param->d->data.ptrVal);

        default:
            throw std::logic_error("Param::getVal<_Tp>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<const ByteArray*>
{
    static const ByteArray* getVal(
        const ito::ParamBase *param,
        int& len)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::StringList:
            len = param->d->len;
            return static_cast<const ByteArray*>(param->d->data.ptrVal);
        default:
            throw std::logic_error("Param::getVal<const ByteArray*>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<const char*>
{
    static const char* getVal(
        const ito::ParamBase *param,
        int& len)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::CharArray:
        case ito::ParamBase::String:
            len = param->d->len;
            return static_cast<const char*>(param->d->data.ptrVal);
        default:
            throw std::logic_error("Param::getVal<const char*>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<const int8*>
{
    static const int8* getVal(
        const ito::ParamBase *param,
        int& len)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::IntArray:
            len = param->d->len;
            return static_cast<const int8*>(param->d->data.ptrVal);
        default:
            throw std::logic_error("Param::getVal<const int8*>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<const int32*>
{
    static const int32* getVal(
        const ito::ParamBase *param,
        int& len)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::IntArray:
            len = param->d->len;
            return static_cast<const int32*>(param->d->data.ptrVal);
        default:
            throw std::logic_error("Param::getVal<const int32*>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<const float64*>
{
    static const float64* getVal(
        const ito::ParamBase *param,
        int& len)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::DoubleArray:
            len = param->d->len;
            return static_cast<const float64*>(param->d->data.ptrVal);
        default:
            throw std::logic_error("Param::getVal<const float64*>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<const complex128*>
{
    static const complex128* getVal(
        const ito::ParamBase *param,
        int& len)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::ComplexArray:
            len = param->d->len;
            return static_cast<const complex128*>(param->d->data.ptrVal);
        default:
            throw std::logic_error("Param::getVal<const complex128*>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<float64>
{
    static ito::RetVal setVal(
        ito::ParamBase *param, float64 val, int /*len = 0*/)
    {
        param->detach();

        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            param->d->data.i8Val = static_cast<int8>(val);
            return ito::retOk;
        case ito::ParamBase::Int:
            param->d->data.i32Val = static_cast<int32>(val);
            return ito::retOk;
        case ito::ParamBase::Double:
            param->d->data.f64Val = val;
            return ito::retOk;
        case ito::ParamBase::Complex:
            param->d->data.c128Val.real = val;
            param->d->data.c128Val.imag = 0.0;
            return ito::retOk;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                "float64 value passed to setVal<float64> does not match the type of the parameter");
        }
    }

    static float64 getVal(
        const ito::ParamBase *param,
        int& /*len*/)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            return static_cast<float64>(param->d->data.i8Val);
        case ito::ParamBase::Int:
            return static_cast<float64>(param->d->data.i32Val);
        case ito::ParamBase::Double:
            return param->d->data.f64Val;
        case ito::ParamBase::Complex:
            return param->d->data.c128Val.real;
        default:
            throw std::logic_error("Param::getVal<float64>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<int32>
{
    static ito::RetVal setVal(
        ito::ParamBase *param, int32 val, int /*len = 0*/)
    {
        param->detach();

        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            param->d->data.i8Val = static_cast<int8>(val);
            return ito::retOk;
        case ito::ParamBase::Int:
            param->d->data.i32Val = val;
            return ito::retOk;
        case ito::ParamBase::Double:
            param->d->data.f64Val = static_cast<float64>(val);
            return ito::retOk;
        case ito::ParamBase::Complex:
            param->d->data.c128Val.real = static_cast<float64>(val);
            param->d->data.c128Val.imag = 0.0;
            return ito::retOk;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                "int32 value passed to setVal<int32> does not match the type of the parameter");
        }
    }

    static int32 getVal(
        const ito::ParamBase *param,
        int& /*len*/)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            return param->d->data.i8Val;
        case ito::ParamBase::Int:
            return param->d->data.i32Val;
        case ito::ParamBase::Double:
            return static_cast<int32>(param->d->data.f64Val);
        case ito::ParamBase::Complex:
            return static_cast<int32>(param->d->data.c128Val.real);
        case 0:
            throw std::invalid_argument("Param::getVal<int32>: non existent parameter");

        default:
            throw std::logic_error("Param::getVal<int32>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<int8>
{
    static ito::RetVal setVal(
        ito::ParamBase *param, int8 val, int /*len = 0*/)
    {
        param->detach();

        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            param->d->data.i8Val = static_cast<int8>(val);
            return ito::retOk;
        case ito::ParamBase::Int:
            param->d->data.i32Val = static_cast<int32>(val);
            return ito::retOk;
        case ito::ParamBase::Double:
            param->d->data.f64Val = static_cast<float64>(val);
            return ito::retOk;
        case ito::ParamBase::Complex:
            param->d->data.c128Val.real = static_cast<float64>(val);
            param->d->data.c128Val.imag = 0.0;
            return ito::retOk;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                "char value passed to setVal<int8> does not match the type of the parameter");
        }
    }

    static int8 getVal(
        const ito::ParamBase *param,
        int& /*len*/)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::Int:
            return static_cast<char>(param->d->data.i8Val);
        case ito::ParamBase::Char:
            return static_cast<char>(param->d->data.i32Val);
        case ito::ParamBase::Double:
            return static_cast<char>(param->d->data.f64Val);
        case ito::ParamBase::Complex:
            return static_cast<char>(param->d->data.c128Val.real);
        case 0:
            throw std::invalid_argument("Param::getVal<int8>: non existent parameter");

        default:
            throw std::logic_error("Param::getVal<int8>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<char>
{
    static ito::RetVal setVal(
        ito::ParamBase *param, char val, int /*len = 0*/)
    {
        param->detach();

        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            param->d->data.i8Val = static_cast<int8>(val);
            return ito::retOk;
        case ito::ParamBase::Int:
            param->d->data.i32Val = static_cast<int32>(val);
            return ito::retOk;
        case ito::ParamBase::Double:
            param->d->data.f64Val = static_cast<float64>(val);
            return ito::retOk;
        case ito::ParamBase::Complex:
            param->d->data.c128Val.real = static_cast<float64>(val);
            param->d->data.c128Val.imag = 0.0;
            return ito::retOk;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                "char value passed to setVal<char> does not match the type of the parameter");
        }
    }

    static char getVal(
        const ito::ParamBase *param,
        int& /*len*/)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::Int:
            return static_cast<char>(param->d->data.i8Val);
        case ito::ParamBase::Char:
            return static_cast<char>(param->d->data.i32Val);
        case ito::ParamBase::Double:
            return static_cast<char>(param->d->data.f64Val);
        case ito::ParamBase::Complex:
            return static_cast<char>(param->d->data.c128Val.real);
        case 0:
            throw std::invalid_argument("Param::getVal<char>: non existent parameter");

        default:
            throw std::logic_error("Param::getVal<char>: Non-matching type!");
        }
    }
};

template <> struct ItomParamHelper<complex128>
{
    static ito::RetVal setVal(
        ito::ParamBase *param,
        const complex128 val,
        int /*len = 0*/)
    {
        param->detach();

        switch (param->d->type)
        {
        case ito::ParamBase::Complex:
            param->d->data.c128Val.real = val.real();
            param->d->data.c128Val.imag = val.imag();
            return ito::retOk;
        default:
            return ito::RetVal(
                ito::retError,
                0,
                "complex128 value passed to setVal<complex128> does not match the type of the "
                "parameter");
        }
    }

    static complex128 getVal(
        const ito::ParamBase *param,
        int& /*len*/)
    {
        switch (param->d->type)
        {
        case ito::ParamBase::Char:
            return complex128(param->d->data.i8Val, 0.0);
        case ito::ParamBase::Int:
            return complex128(param->d->data.i32Val, 0.0);
        case ito::ParamBase::Double:
            return complex128(param->d->data.f64Val, 0.0);
        case ito::ParamBase::Complex:
            return complex128(param->d->data.c128Val.real, param->d->data.c128Val.imag);
        default:
            throw std::logic_error("Param::getVal<complex128>: Non-matching type!");
        }
    }
};

} // end namespace ito
