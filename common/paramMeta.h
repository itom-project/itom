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
#include "retVal.h"
#include "typeDefs.h"

#include <limits>

/* definition and macros */
/* global variables (avoid) */
/* content */

namespace ito
{
/*!
\class ParamMeta
\brief Base class for all meta-information classes

Parameters of type ito::Param can have a pointer to this class. Consider this base class to be abstract, such that
it is only allowed to pass the right implementation (derived from this class) that fits to the type of the parameter.
The runtime type information value m_type indicates the real type of this pointer, such that a direct cast
can be executed.

\sa ito::CharMeta, ito::IntMeta, ito::DoubleMeta, ito::StringMeta, ito::HWMeta, ito::DObjMeta, ito::CharArrayMeta,
ito::IntArrayMeta, ito::DoubleArrayMeta, ito::StringListMeta
*/
class ITOMCOMMON_EXPORT ParamMeta
{
public:
    /*!
        \brief Runtime type information

        MetaRtti is used to cast param meta objects, without
        having to enable runtime type information of the compiler.
    */
    enum MetaRtti
    {
        rttiUnknown = 0,         /*!< unknown parameter */
        rttiCharMeta = 1,        /*!< meta for a char parameter */
        rttiIntMeta = 2,         /*!< meta for an integer parameter */
        rttiDoubleMeta = 3,      /*!< meta for a double parameter */
        rttiStringMeta = 4,      /*!< meta for a string parameter */
        rttiHWMeta = 5,          /*!< meta for a hardware plugin parameter */
        rttiDObjMeta = 6,        /*!< meta for a data object parameter */
        rttiIntArrayMeta = 7,    /*!< meta for an integer array parameter */
        rttiDoubleArrayMeta = 8, /*!< meta for a double array parameter */
        rttiCharArrayMeta = 9,   /*!< meta for a char array parameter */
        rttiIntervalMeta = 10, /*!< meta for an integer array with two values that represent an interval [value1, value2] parameter */
        rttiDoubleIntervalMeta = 11, /*!< meta for a double array with two values that represent an interval [value1,
                                        value2] parameter (size of the interval is value2-value1) */
        rttiRangeMeta = 12, /*!< meta for an integer array with two values that represent a range [value1, value2]
                               parameter (size of a range is 1+value2-value1) */
        rttiRectMeta = 13,  /*!< meta for an integer array with four values that consists of two ranges (vertical and
                               horizontal, e.g. for ROIs of cameras) */
        rttiStringListMeta = 14 /*!< meta for a string list */
    };

    /*!
    \brief The representation of number types indicates the type of widget that is suited best to display and change the
    value

    Not all representations can be applied to all types of number values, e.g. IPV4 can not be used for char-types.
    e.g. - Char, CharArray: Linear, Boolean, Logarithmic, PureNumber
         - IntegerArray, Range, Interval: Linear, Boolean, Logarithmic, PureNumber
         - Integer: Linear, Boolean, Logarithmic, PureNumber, HexNumber, IPV4Address, MACAddress
         - Double, DoubleArray: Linear, Boolean, Logarithmic, PureNumber
    */
    enum tRepresentation
    {
        Linear = 0x0001,      //!< Slider with linear behavior
        Logarithmic = 0x0002, //!< Slider with logarithmic behaviour
        Boolean = 0x0004,     //!< Check box
        PureNumber = 0x0008,  //!< Decimal number in an edit control
        HexNumber = 0x0010,   //!< Hex number in an edit control
        IPV4Address = 0x0020, //!< IP-Address
        MACAddress = 0x0040,  //!< MAC-Address
        UnknownRepresentation = 0x0080
    };

    //!< default constructor with an unknown meta information type
    explicit ParamMeta(ito::ByteArray category = ito::ByteArray());

    //!< constructor used by derived classes to indicate their real type
    explicit ParamMeta(MetaRtti type, ito::ByteArray category = ito::ByteArray());

    ParamMeta(const ParamMeta &copy);

    ParamMeta &operator=(const ParamMeta &rhs);

    //!< destructor
    virtual ~ParamMeta();

    //!< returns runtime type information value
    inline MetaRtti getType() const
    {
        return m_type;
    }

    //!< returns category name of this parameter (default: empty ByteArray)
    inline ito::ByteArray getCategory() const
    {
        return m_category;
    }

    void setCategory(const ito::ByteArray &category);

    virtual bool operator==(const ParamMeta &other) const;

    bool operator!=(const ParamMeta &other) const
    {
        return !(*this == other);
    }

protected:
    MetaRtti m_type;
    ito::ByteArray m_category; //!< optional category name of this parameter
};

/*!
\class CharMeta
\brief meta-information for Param of type Char.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::Char. If set, the given
char number can be limited with respect to given minimum and maximum values as well as an optional step size (default:
1).

\sa ito::Param, ito::ParamMeta
*/
class ITOMCOMMON_EXPORT CharMeta : public ParamMeta
{
public:
    //!< constructor with minimum and maximum value as
    //!< well as optional step size (default: 1)
    explicit CharMeta(char minVal, char maxVal, char stepSize = 1,
                      ito::ByteArray category = ito::ByteArray());

    CharMeta(const CharMeta &copy);

    CharMeta &operator=(const CharMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;

    //!< returns a new instance of CharMeta, where the min and max are set to the full range
    //!< available for char. The caller has to take care of memory.
    static CharMeta *all(ito::ByteArray category = ito::ByteArray());

    //!< returns minimum value
    inline char getMin() const
    {
        return m_minVal;
    }

    //!< returns maximum value
    inline char getMax() const
    {
        return m_maxVal;
    }

    //!< returns step size
    inline char getStepSize() const
    {
        return m_stepSize;
    }

    //!< returns unit
    inline ito::ByteArray getUnit() const
    {
        return m_unit;
    }

    //!< sets unit string of this parameter
    void setUnit(const ito::ByteArray &unit);

    //!< returns display representation
    inline ParamMeta::tRepresentation getRepresentation() const
    {
        return m_representation;
    }

    //!< sets display representation
    void setRepresentation(ParamMeta::tRepresentation representation);

    //! sets the minimum value
    /*!
        \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is
       changed to val, too
    */
    void setMin(char val);

    //! sets the maximum value
    /*!
        \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is
       changed to val, too
    */
    void setMax(char val);

    //! sets the step size
    /*!
        \param val is the new step size, hence only discrete values [minVal, minVal+stepSize,
       minVal+2*stepSize...,maxVal] are allowed
    */
    void setStepSize(char val);

  private:
    char m_minVal;
    char m_maxVal;
    char m_stepSize;                             // >= 1
    ito::ByteArray m_unit;                       //!< unit of value, e.g. 'mm', ...
    ParamMeta::tRepresentation m_representation; //!< hint for display representation in GUI widget
};

/*!
\class IntMeta
\brief Meta-information for Param of type Int.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::Int. If set, the given
integer number can be limited with respect to given minimum and maximum values as well as an optional step size
(default: 1).

\sa ito::Param, ito::ParamMeta
*/
class ITOMCOMMON_EXPORT IntMeta : public ParamMeta
{
  public:
    //!< constructor with minimum and maximum value as
    //!< well as optional step size (default: 1)
    explicit IntMeta(int32 minVal, int32 maxVal, int32 stepSize = 1,
                     ito::ByteArray category = ito::ByteArray());

    IntMeta(const IntMeta &copy);

    IntMeta &operator=(const IntMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;

    //!< returns a new instance of IntMeta, where the min and max are set to the full range
    //!< available for integers. The caller has to take care of memory.
    static IntMeta *all(ito::ByteArray category = ito::ByteArray());

    //!< returns minimum value
    inline int32 getMin() const
    {
        return m_minVal;
    }

    //!< returns maximum value
    inline int32 getMax() const
    {
        return m_maxVal;
    }

    //!< returns step size
    inline int32 getStepSize() const
    {
        return m_stepSize;
    }

    //!< returns unit
    inline ito::ByteArray getUnit() const
    {
        return m_unit;
    }

    //!< sets unit string of this parameter
    void setUnit(const ito::ByteArray &unit);

    //!< returns display representation
    inline ParamMeta::tRepresentation getRepresentation() const
    {
        return m_representation;
    }

    //!< sets display representation
    void setRepresentation(ParamMeta::tRepresentation behaviour);

    //! sets the minimum value
    /*!
        \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is
       changed to val, too
    */
    void setMin(int32 val);

    //! sets the maximum value
    /*!
        \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is
       changed to val, too
    */
    void setMax(int32 val);

    //! sets the step size
    /*!
        \param val is the new step size, hence only discrete values [minVal, minVal+stepSize,
       minVal+2*stepSize...,maxVal] are allowed
    */
    void setStepSize(int32 val);

  private:
    int32 m_minVal;
    int32 m_maxVal;
    int32 m_stepSize;                            // >= 1
    ito::ByteArray m_unit;                       //!< unit of value, e.g. 'mm', ...
    ParamMeta::tRepresentation m_representation; //!< hint for display behaviour in GUI widget
};

/*!
\class DoubleMeta
\brief Meta-information for ito::Param of type Double.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::Double. If set, the given
double number can be limited with respect to given minimum and maximum values as well as an optional step size (default:
0.0 -> no step size).

\sa ito::Param, ito::ParamMeta
*/
class ITOMCOMMON_EXPORT DoubleMeta : public ParamMeta
{
  public:
    /*!
    \brief Display notation style if the related parameters is displayed in any widget
    */
    enum tDisplayNotation
    {
        Automatic, //!< double number is automatically rendered in any GUI element (default)
        Fixed,     //!< if possible, the double number should be shown as fixed number, e.g. 1000.00
        Scientific //!< if possible, the double number should be rendered in a scientific notation, e.g. 1.0E3
    };

    //! constructor with minimum and maximum value
    explicit DoubleMeta(float64 minVal, float64 maxVal, float64 stepSize = 0.0 /*0.0 means no specific step size*/,
                        ito::ByteArray category = ito::ByteArray());

    DoubleMeta(const DoubleMeta &copy);

    DoubleMeta &operator=(const DoubleMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;

    //!< returns a new instance of DoubleMeta, where the min and max are set to the full
    //!< range available for double. The caller has to take care of memory.
    static DoubleMeta *all(ito::ByteArray category = ito::ByteArray());

    //!< returns minimum value
    inline float64 getMin() const
    {
        return m_minVal;
    }

    //!< returns maximum value
    inline float64 getMax() const
    {
        return m_maxVal;
    }

    //!< returns step size
    inline float64 getStepSize() const
    {
        return m_stepSize;
    }

    //!< returns unit
    inline ito::ByteArray getUnit() const
    {
        return m_unit;
    }

    //!< sets unit string of this parameter
    void setUnit(const ito::ByteArray &unit);

    //!< returns display precision
    inline int getDisplayPrecision() const
    {
        return m_displayPrecision;
    }

    //!< sets display precision
    void setDisplayPrecision(int displayPrecision);

    //!< returns display notation
    inline DoubleMeta::tDisplayNotation getDisplayNotation() const
    {
        return m_displayNotation;
    }

    //!< sets display notation
    void setDisplayNotation(DoubleMeta::tDisplayNotation displayNotation);

    //!< returns display representation
    inline ParamMeta::tRepresentation getRepresentation() const
    {
        return m_representation;
    }

    //!< sets display representation
    void setRepresentation(ParamMeta::tRepresentation representation);

    //! sets the minimum value
    /*!
        \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is
       changed to val, too
    */
    void setMin(float64 val);

    //! sets the maximum value
    /*!
        \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is
       changed to val, too
    */
    void setMax(float64 val);

    //! sets the step size
    /*!
        \param val is the new step size, hence only discrete values [minVal, minVal+stepSize,
       minVal+2*stepSize...,maxVal] are allowed
    */
    void setStepSize(float64 val);

  private:
    float64 m_minVal;
    float64 m_maxVal;
    float64 m_stepSize;     // >= 0, 0.0 means no specific step size
    ito::ByteArray m_unit;  //!< unit of value, e.g. 'mm', ...
    int m_displayPrecision; //!< hint for the number of decimal digits that should be shown in any GUI widget, default:
                            //!< 3
    tDisplayNotation m_displayNotation; //!< indicates how this double number should be rendered (e.g. in GUI widgets)
    ParamMeta::tRepresentation m_representation; //!< hint for display representation in GUI widget
};

/*!
\class HWMeta
\brief Meta-information for Param of type HWPtr.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::HWPtr, that is an instance
of another hardware plugin. If set, it is possible to restrict the given hardware plugin to a specific type (e.g.
dataIO, dataIO + grabber, actuator...) and/or to limit it to a specific name of the plugin (e.g. SerialIO).

\sa ito::Param, ito::ParamMeta
*/
class ITOMCOMMON_EXPORT HWMeta : public ParamMeta
{
  public:
    //! constructor
    /*!
        creates HWMeta-information struct where you can pass a bitmask which consists of values of the enumeration
        ito::tPluginType. The plugin reference of the corresponding Param should then only accept plugins, where
        all bits are set, too.
        \sa ito::Plugin, ito::tPluginType
    */
    explicit HWMeta(uint32 minType, ito::ByteArray category = ito::ByteArray());

    //! constructor
    /*!
        creates HWMeta-information struct where you can pass a specific name of a plugin, which only is
        allowed by the corresponding plugin-instance.
        \sa ito::Plugin
    */
    explicit HWMeta(const ito::ByteArray &hwAddInName, ito::ByteArray category = ito::ByteArray());

    //!< copy constructor
    HWMeta(const HWMeta &cpy);

    virtual bool operator==(const ParamMeta &other) const;

    //!< assignment operator
    HWMeta &operator=(const HWMeta &rhs);

    //!< returns type-bitmask which is minimally required by plugin-reference. Default 0. \sa ito::tPluginType
    inline uint32 getMinType() const
    {
        return m_minType;
    }

    //!< returns name of specific hardware plugin
    inline ito::ByteArray getHWAddInName() const
    {
        return m_HWName;
    }

  private:
    uint32 m_minType;        //!< type-bitmask which is minimally required. default: 0
    ito::ByteArray m_HWName; //!< zero-terminated name of specific plugin-name or invalid if not defined
};

class StringMetaPrivate; // forward declaration

/*!
\class StringMeta
\brief Meta-information for Param of type String.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::String.
If set, it is possible to restrict the a given string to fit to a given list of strings. This list of strings
might be interpreted in an exact way (tType::String), as wildcard expressions (tType::Wildcard) or as regular
expressions (tType::RegExp).

\sa ito::Param, ito::ParamMeta
*/
class ITOMCOMMON_EXPORT StringMeta : public ParamMeta
{
  public:
    enum tType
    {
        String,   //!< string elements should be considered as strings (exact match)
        Wildcard, //!< string elements should be considered as wildcard-expressions (e.g. *.doc)
        RegExp    //!< string elements should be considered as regular expressions (e.g. ^(.*)[abc]{1,5}$)
    };

    //! constructor
    /*!
        Returns a meta information class for string-types.
        \param type indicates how the string elements should be considered
        \sa tType
    */
    StringMeta(tType type, ito::ByteArray category = ito::ByteArray());

    //! constructor
    /*!
        Returns a meta information class for string-types.
        \param type indicates how the string elements should be considered
        \param val adds a first string to the element list
        \sa tType
    */
    StringMeta(tType type, const char *val, ito::ByteArray category = ito::ByteArray());

    //! constructor
    /*!
    Returns a meta information class for string-types.
    \param type indicates how the string elements should be considered
    \param val adds a first string to the element list
    \sa tType
    */
    StringMeta(tType type, const ito::ByteArray &val, ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    StringMeta(const StringMeta &cpy);

    //! destructor
    virtual ~StringMeta();

    StringMeta &operator+=(const char *val); //!< add another pattern string to the list of patterns.
    StringMeta &operator=(const StringMeta &rhs);
    virtual bool operator==(const ParamMeta &other) const;

    tType getStringType() const;    //!< returns the type how strings in list should be considered. \sa tType
    void setStringType(tType type); //!< sets the type how strings in pattern list should be considered. \sa tType
    int getLen() const;             //!< returns the number of string elements in meta information class.
    const char *getString(
        int idx = 0) const;        //!< returns string from list at index position or NULL, if index is out of range.
    bool addItem(const char *val); //!< adds another element to the list of patterns.
    bool addItem(const ito::ByteArray &val); //!< adds another element to the list of patterns.
    void clearItems();                       //!< clear all elements from the pattern list.

  private:
    StringMetaPrivate *p;
};

/*!
\class DObjMeta
\brief Meta-information for Param of type DObjPtr.

(not used yet)

\sa ito::Param, ito::ParamMeta
*/
class ITOMCOMMON_EXPORT DObjMeta : public ParamMeta
{
  public:

    //!< constructor. This default constructor has no data type restrictions.
    //!< use appendAllowedDataType to add one of multiple allowed data types.
    explicit DObjMeta(int minDim = 0, int maxDim = (std::numeric_limits<int>::max)(),
          ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    DObjMeta(const DObjMeta &cpy);

    //!< assignment operator
    DObjMeta &operator=(const DObjMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;

    //!< returns the number of allowed data types or 0 if no type restriction is given.
    int getNumAllowedDataTypes() const;

    //!< returns the i-th allowed data type. i must be in range [0, getNumAllowedDataTypes).
    ito::tDataType getAllowedDataType(int index) const;

    //!< returns true if the given dataType is allowed
    bool isDataTypeAllowed(ito::tDataType dataType) const;

    //!< add a new data type to the list of allowed data types.
    void appendAllowedDataType(ito::tDataType dataType);

    //!< returns minimum allowed dimensions of data object
    inline int getMinDim() const
    {
        return m_minDim;
    }

    //!< returns maximum number of dimensions of data object
    inline int getMaxDim() const
    {
        return m_maxDim;
    }

    //!< set the minimum allowed dimensions of data object
    inline void setMinDim(int minDim)
    {
        m_minDim = minDim;
    }

    //!< set the maximum number of dimensions of data object
    inline void setMaxDim(int maxDim)
    {
        m_maxDim = maxDim;
    }

  private:
    //!< all allowed data types, every character in this array has to
    //!< be interpreted as ito::tDataType value. If the size of this
    //!< array is 0, no type restriction is set.
    ito::ByteArray m_allowedTypes;
    int m_minDim;
    int m_maxDim;
};

/*!
\class ListMeta
\brief Additional base class for all list or array meta information.

Meta objects that derive from this class, too get further meta
information about the minimum and maximum number of elements
in a list, including an optional step size.
*/
class ITOMCOMMON_EXPORT ListMeta
{
public:
    //!< default constructor with an arbitrary list size
    ListMeta();

    explicit ListMeta(size_t numMin, size_t numMax, size_t numStepSize = 1);

    //!< destructor
    virtual ~ListMeta();

    virtual bool operator==(const ListMeta &other) const;

    //!< returns minimum number of values
    inline size_t getNumMin() const
    {
        return m_numMin;
    }

    //!< returns maximum number of values
    inline size_t getNumMax() const
    {
        return m_numMax;
    }

    //!< returns step size of number of values
    inline size_t getNumStepSize() const
    {
        return m_numStep;
    }

    //! sets the minimum number of values
    /*!
        \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is
       changed to val, too
    */
    void setNumMin(size_t val);

    //! sets the maximum number of values
    /*!
        \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is
       changed to val, too
    */
    void setNumMax(size_t val);

    //! sets the step size of the number of values
    /*!
        \param val is the new step size, hence only discrete values [minVal, minVal+stepSize,
       minVal+2*stepSize...,maxVal] are allowed
    */
    void setNumStepSize(size_t val);

private:
    size_t m_numMin;
    size_t m_numMax;
    size_t m_numStep;
};

/*!
\class CharArrayMeta
\brief Meta-information for Param of type CharArrayMeta.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::CharArray.
Since this meta information class is derived from ito::CharMeta, it is possible to restrict each value to the single
value contraints of ito::CharMeta. Furthermore, this class allows restricting the minimum and maximum length of the
array as well as the optional step size of the array's length.

\sa ito::Param, ito::ParamMeta, ito::CharMeta
*/
class ITOMCOMMON_EXPORT CharArrayMeta : public CharMeta, public ListMeta
{
  public:
    explicit CharArrayMeta(char minVal, char maxVal, char stepSize = 1, ito::ByteArray category = ito::ByteArray());
    explicit CharArrayMeta(char minVal, char maxVal, char stepSize, size_t numMin, size_t numMax,
                           size_t numStepSize = 1, ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    CharArrayMeta(const CharArrayMeta &cpy);

    //!< assignment operator
    CharArrayMeta &operator=(const CharArrayMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;
};

/*!
\class CharArrayMeta
\brief Meta-information for Param of type IntArrayMeta.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray.
Since this meta information class is derived from ito::IntMeta, it is possible to restrict each value to the single
value contraints of ito::IntMeta. Furthermore, this class allows restricting the minimum and maximum length of the array
as well as the optional step size of the array's length.

\sa ito::Param, ito::ParamMeta, ito::IntArray
*/
class ITOMCOMMON_EXPORT IntArrayMeta : public IntMeta, public ListMeta
{
  public:
    explicit IntArrayMeta(int32 minVal, int32 maxVal, int stepSize = 1, ito::ByteArray category = ito::ByteArray());
    explicit IntArrayMeta(int32 minVal, int32 maxVal, int stepSize, size_t numMin, size_t numMax,
                          size_t numStepSize = 1, ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    IntArrayMeta(const IntArrayMeta &cpy);

    //!< assignment operator
    IntArrayMeta &operator=(const IntArrayMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;
};

/*!
\class DoubleArrayMeta
\brief Meta-information for Param of type DoubleArrayMeta.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::DoubleArray.
Since this meta information class is derived from ito::DoubleArray, it is possible to restrict each value to the single
value contraints of ito::DoubleArray. Furthermore, this class allows restricting the minimum and maximum length of the
array as well as the optional step size of the array's length.

\sa ito::Param, ito::ParamMeta, ito::DoubleMeta
*/
class ITOMCOMMON_EXPORT DoubleArrayMeta : public DoubleMeta, public ListMeta
{
  public:
    explicit DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize = 0.0,
                             ito::ByteArray category = ito::ByteArray());

    explicit DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize, size_t numMin, size_t numMax,
                             size_t numStepSize = 1, ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    DoubleArrayMeta(const DoubleArrayMeta &cpy);

    //!< assignment operator
    DoubleArrayMeta &operator=(const DoubleArrayMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;
};

/*!
\class DoubleIntervalMeta
\brief Meta-information for Param of type DoubleIntervalMeta.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::DoubleArray.
This meta information class indicates that the corresponding double array parameter is interpreted as an interval,
hence, only an array consisting of two values is accepted. The size of the interval is defined by the difference
(value[1] - value[0]). You can restrict this size to a certain minimum and maximum value as well as indicating a
specific step size (default: 0.0 -> no step size).

\sa ito::Param, ito::ParamMeta, ito::DoubleMeta
*/
class ITOMCOMMON_EXPORT DoubleIntervalMeta : public DoubleMeta
{
  public:
    explicit DoubleIntervalMeta(float64 minVal, float64 maxVal, float64 stepSize = 0.0,
                                ito::ByteArray category = ito::ByteArray());
    explicit DoubleIntervalMeta(float64 minVal, float64 maxVal, float64 stepSize, float64 sizeMin, float64 sizeMax,
                                float64 sizeStep = 0.0, ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    DoubleIntervalMeta(const DoubleIntervalMeta &cpy);

    virtual bool operator==(const ParamMeta &other) const;

    //!< assignment operator
    DoubleIntervalMeta &operator=(const DoubleIntervalMeta &rhs);

    //!< returns minimum size of range
    inline float64 getSizeMin() const
    {
        return m_sizeMin;
    }

    //!< returns maximum size of range
    inline float64 getSizeMax() const
    {
        return m_sizeMax;
    }

    //!< returns step size of size of range
    inline float64 getSizeStepSize() const
    {
        return m_sizeStep;
    }

    //! sets the minimum size of the interval (= max-min)
    /*!
        \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is
       changed to val, too
    */
    void setSizeMin(float64 val);

    //! sets the maximum size of the interval (= max-min)
    /*!
        \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is
       changed to val, too
    */
    void setSizeMax(float64 val);

    //! sets the step size of the size of the interval (= max-min)
    /*!
        \param val is the new step size, hence only discrete values [minVal, minVal+stepSize,
       minVal+2*stepSize...,maxVal] are allowed
    */
    void setSizeStep(float64 val);

  private:
    float64 m_sizeMin;
    float64 m_sizeMax;
    float64 m_sizeStep;
};

/*!
\class StringListMeta
\brief Meta-information for Param of type STringListMeta.

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::StringListMeta.
Since this meta information class is derived from ito::StringMeta,
it is possible to restrict each value to the single value contraints of ito::StringMeta.
Furthermore, this class allows restricting the minimum and maximum length of the list as well as the optional step size
of the list's length.

\sa ito::Param, ito::ParamMeta, ito::StringMeta
*/
class ITOMCOMMON_EXPORT StringListMeta : public StringMeta, public ListMeta
{
  public:
    //! constructor
    /*!
        Returns a meta information class for string list types.
        \param type indicates how the string elements should be considered
        \param numMin is the minimum number of values in the list
        \param numMax is the maximum number of values in the list
        \param numStepSize is the step size of the number of values in the list
        \sa tType
    */
    explicit StringListMeta(tType type, size_t numMin, size_t numMax, size_t numStepSize = 1,
                            ito::ByteArray category = ito::ByteArray());

    //! constructor
    /*!
        Returns a meta information class for string list types.
        \param type indicates how the string elements should be considered
        \param val adds a first string to the element list
        \param numMin is the minimum number of values in the list
        \param numMax is the maximum number of values in the list
        \param numStepSize is the step size of the number of values in the list
        \sa tType
    */
    explicit StringListMeta(tType type, const char *val, size_t numMin, size_t numMax, size_t numStepSize = 1,
                            ito::ByteArray category = ito::ByteArray());

    //! constructor
    /*!
        Returns a meta information class for string list types.
        \param type indicates how the string elements should be considered
        \param val adds a first string to the element list
        \param numMin is the minimum number of values in the list
        \param numMax is the maximum number of values in the list
        \param numStepSize is the step size of the number of values in the list
        \sa tType
    */
    explicit StringListMeta(tType type, const ito::ByteArray &val, size_t numMin, size_t numMax, size_t numStepSize = 1,
                            ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    StringListMeta(const StringListMeta &cpy);

    //! destructor
    virtual ~StringListMeta();

    virtual bool operator==(const ParamMeta &other) const;

    //!< assignment operator
    StringListMeta &operator=(const StringListMeta &rhs);
};

/*!
\class IntervalMeta
\brief Meta-information for Param of type IntArrayMeta that represent an interval [minimum, maximum).

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray.
This meta information class indicates that the corresponding integer array parameter is interpreted as an interval,
hence, only an array consisting of two values is accepted. The size of the interval is defined by the difference
(value[1] - value[0]). You can restrict this size to a certain minimum and maximum value as well as indicating a
specific step size (default: 1).

An example for an interval might be a certain interval of allowed radius values when optimizing a cylinder fit.

\sa ito::Param, ito::ParamMeta, ito::RangeMeta, ito::IntMeta, ito::IntervalMeta
*/
class ITOMCOMMON_EXPORT IntervalMeta : public IntMeta
{
  public:
    explicit IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize = 1, ito::ByteArray category = ito::ByteArray());
    explicit IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize, int32 sizeMin, int32 sizeMax,
                          int32 intervalStep = 1, ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    IntervalMeta(const IntervalMeta &cpy);

    //!< assignment operator
    IntervalMeta &operator=(const IntervalMeta &rhs);

    virtual bool operator==(const ParamMeta &other) const;

    //!< returns minimum size of interval or range
    inline int getSizeMin() const
    {
        return m_sizeMin;
    }

    //!< returns maximum size of interval or range
    inline int getSizeMax() const
    {
        return m_sizeMax;
    }

    //!< returns step size of size of interval or range
    inline int getSizeStepSize() const
    {
        return m_sizeStep;
    }

    inline bool isIntervalNotRange() const
    {
        return m_isIntervalNotRange;
    }

    //! sets the minimum size of the interval (= max-min)
    /*!
        \param val is the new minimum value, if this is bigger than the current maximum value, the maximum value is
       changed to val, too
    */
    void setIntervalMin(int32 val);

    //! sets the maximum size of the interval (= max-min)
    /*!
        \param val is the new maximum value, if this is smaller than the current minimum value, the minimum value is
       changed to val, too
    */
    void setIntervalMax(int32 val);

    //! sets the step size of the size of the interval (= max-min)
    /*!
        \param val is the new step size, hence only discrete values [minVal, minVal+stepSize,
       minVal+2*stepSize...,maxVal] are allowed
    */
    void setIntervalStep(int32 val);

  protected:
    int32 m_sizeMin;
    int32 m_sizeMax;
    int32 m_sizeStep;
    bool m_isIntervalNotRange; //!< this flag describes if this object is an interval where its interval/range is
                               //!< (end-begin) or a range with (1+end-begin)
};

/*!
\class RangeMeta
\brief Meta-information for Param of type IntArrayMeta that represent a range [minVal, maxVal].

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray.
This meta information class indicates that the corresponding integer array parameter is interpreted as a range, hence,
only an array consisting of two values is accepted. The size of the interval is defined by the difference (1 + value[1]
- value[0]). You can restrict this size to a certain minimum and maximum value as well as indicating a specific step
size (default: 1).

An example for a range might be a one dimension (vertical or horizontal) of a ROI (region of interest) of a camera,
where the range determines the first and last pixel value inside of the ROI, such that the total size is the difference
between both limits + 1.

The range object is defined by its first and last value, that are both inside of the range, hence the size of the range
is (1+last-first). This is the difference to IntervalMeta, where the size of the interval is last-first only.

\sa ito::Param, ito::ParamMeta, ito::IntervalMeta, ito::IntArrayMeta
*/
class ITOMCOMMON_EXPORT RangeMeta : public IntervalMeta
{
  public:
    explicit RangeMeta(int32 minVal, int32 maxVal, int32 stepSize = 1, ito::ByteArray category = ito::ByteArray());
    explicit RangeMeta(int32 minVal, int32 maxVal, int32 stepSize, size_t sizeMin, size_t sizeMax, size_t sizeStep = 1,
                       ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    RangeMeta(const RangeMeta &cpy);

    //!< assignment operator
    RangeMeta &operator=(const RangeMeta &rhs);
};

/*!
\class RectMeta
\brief Meta-information for Param of type IntArrayMeta that represent a rectangle (left, top, width, height).

An object of this class can be used to parametrize a parameter whose type is ito::ParamBase::IntArray.
This meta information class indicates that the corresponding integer array parameter is interpreted as a rectangle,
hence, only an array consisting of four values is accepted. This meta information consists of two object of type
ito::RangeMeta, describing the contraints of the horizontal and vertical axes of the rectangle.

\sa ito::Param, ito::ParamMeta, ito::RangeMeta, ito::IntArrayMeta
*/
class ITOMCOMMON_EXPORT RectMeta : public ParamMeta
{
  public:
    explicit RectMeta(const ito::RangeMeta &widthMeta, const ito::RangeMeta &heightMeta,
                      ito::ByteArray category = ito::ByteArray());

    //! copy constructor
    RectMeta(const RectMeta &cpy);

    //!< assignment operator
    RectMeta &operator=(const RectMeta &rhs);

    inline const ito::RangeMeta& getWidthRangeMeta() const
    {
        return m_widthMeta;
    }

    inline const ito::RangeMeta& getHeightRangeMeta() const
    {
        return m_heightMeta;
    }

    void setWidthRangeMeta(const ito::RangeMeta &widthMeta);
    void setHeightRangeMeta(const ito::RangeMeta &heightMeta);

    //!< returns unit
    inline ito::ByteArray getUnit() const
    {
        return m_heightMeta.getUnit();
    }

    //!< sets unit string of this parameter
    void setUnit(const ito::ByteArray &unit);

    virtual bool operator==(const ParamMeta &other) const;

  protected:
    ito::RangeMeta m_heightMeta;
    ito::RangeMeta m_widthMeta;
};

} // end namespace ito
