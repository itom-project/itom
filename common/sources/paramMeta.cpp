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

/* includes */
#include "../paramMeta.h"

#include <vector>

#if _DEBUG
#include <stdexcept>
#endif


namespace ito
{
//--------------------------------------------------------------------------------
ParamMeta::ParamMeta(ByteArray category /*= ito::ByteArray()*/) :
    m_type(rttiUnknown),
    m_category(category)
{
}

//--------------------------------------------------------------------------------
ParamMeta::ParamMeta(MetaRtti type, ito::ByteArray category /*= ito::ByteArray()*/) : m_type(type), m_category(category)
{
}

//--------------------------------------------------------------------------------
ParamMeta::~ParamMeta()
{
}

//--------------------------------------------------------------------------------
ParamMeta::ParamMeta(const ParamMeta &copy) :
    m_category(copy.m_category),
    m_type(rttiUnknown)
{

}

//--------------------------------------------------------------------------------
ParamMeta &ParamMeta::operator=(const ParamMeta &rhs)
{
    m_category = rhs.m_category;
    m_type = rttiUnknown;
    return *this;
}

//--------------------------------------------------------------------------------
void ParamMeta::setCategory(const ito::ByteArray &category)
{
    m_category = category;
}

//--------------------------------------------------------------------------------
bool ParamMeta::operator==(const ParamMeta& other) const
{
    return (m_type == other.m_type && \
        m_category == other.m_category);
}

//---------------------------------------------------------------------------------
CharMeta::CharMeta(char minVal, char maxVal, char stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    ParamMeta(rttiCharMeta, category),
    m_minVal(minVal),
    m_maxVal(maxVal),
    m_stepSize(stepSize),
    m_representation(ParamMeta::PureNumber)
{
    if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal);

    if (m_minVal == 0 && m_maxVal == 1 && m_stepSize == 1)
    {
        m_representation = ParamMeta::Boolean;
    }

#if _DEBUG
    if (stepSize <= 0)
    {
        throw std::logic_error("stepSize of CharMeta must be >= 1");
    }
#endif
}

//---------------------------------------------------------------------------------
CharMeta::CharMeta(const CharMeta &copy) :
    ParamMeta(rttiCharMeta, copy.m_category),
    m_minVal(copy.m_minVal),
    m_maxVal(copy.m_maxVal),
    m_stepSize(copy.m_stepSize),
    m_representation(copy.m_representation)
{

}

//---------------------------------------------------------------------------------
CharMeta& CharMeta::operator=(const CharMeta &rhs)
{
    ParamMeta::operator=(rhs);
    m_type = rttiCharMeta;
    m_minVal = rhs.m_minVal;
    m_maxVal = rhs.m_maxVal;
    m_stepSize = rhs.m_stepSize;
    m_representation = rhs.m_representation;
    return *this;
}

//---------------------------------------------------------------------------------
CharMeta* CharMeta::all(ito::ByteArray category /*= ito::ByteArray()*/)
{
    return new CharMeta(std::numeric_limits<char>::min(), std::numeric_limits<char>::max(), 1, category);
}

//---------------------------------------------------------------------------------
void CharMeta::setMin(char val)
{
    m_minVal = val;
    m_maxVal = std::max(m_maxVal,m_minVal);
}

//---------------------------------------------------------------------------------
void CharMeta::setMax(char val)
{
    m_maxVal = val;
    m_minVal = std::min(m_maxVal,m_minVal);
}

//---------------------------------------------------------------------------------
void CharMeta::setStepSize(char val)
{
#if _DEBUG
    if (val <= 0)
    {
        throw std::logic_error("stepSize of CharMeta must be >= 1");
    }
#endif
    m_stepSize = val;
}

//---------------------------------------------------------------------------------
//!< sets unit string of this parameter
void CharMeta::setUnit(const ito::ByteArray &unit)
{
    m_unit = unit;
}

//---------------------------------------------------------------------------------
void CharMeta::setRepresentation(ParamMeta::tRepresentation representation)
{
    m_representation = representation;
}

//---------------------------------------------------------------------------------
bool CharMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const CharMeta *other_ = (const CharMeta*)(&other);
    return ((m_minVal == other_->m_minVal) && \
        (m_maxVal == other_->m_maxVal) && \
        (m_stepSize == other_->m_stepSize) && \
        (m_unit == other_->m_unit) && \
        (m_representation == other_->m_representation));
}



//---------------------------------------------------------------------------------
IntMeta::IntMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    ParamMeta(rttiIntMeta, category),
    m_minVal(minVal),
    m_maxVal(maxVal),
    m_stepSize(stepSize),
    m_representation(ParamMeta::PureNumber)
{
    if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal);

    if (m_minVal == 0 && m_maxVal == 1 && m_stepSize == 1)
    {
        m_representation = ParamMeta::Boolean;
    }

#if _DEBUG
    if (stepSize <= 0)
    {
        throw std::logic_error("stepSize of IntMeta must be >= 1");
    }
#endif
}

//---------------------------------------------------------------------------------
IntMeta::IntMeta(const IntMeta &copy) :
    ParamMeta(rttiIntMeta, copy.m_category),
    m_minVal(copy.m_minVal),
    m_maxVal(copy.m_maxVal),
    m_stepSize(copy.m_stepSize),
    m_representation(copy.m_representation)
{

}

//---------------------------------------------------------------------------------
IntMeta& IntMeta::operator=(const IntMeta &rhs)
{
    ParamMeta::operator=(rhs);
    m_type = rttiIntMeta;
    m_minVal = rhs.m_minVal;
    m_maxVal = rhs.m_maxVal;
    m_stepSize = rhs.m_stepSize;
    m_representation = rhs.m_representation;
    return *this;
}

//---------------------------------------------------------------------------------
IntMeta* IntMeta::all(ito::ByteArray category /*= ito::ByteArray()*/)
{
    return new IntMeta(std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), 1, category);
}

//---------------------------------------------------------------------------------
void IntMeta::setMin(int32 val)
{
    m_minVal = val;
    m_maxVal = std::max(m_maxVal,m_minVal);
}

//---------------------------------------------------------------------------------
void IntMeta::setMax(int32 val)
{
    m_maxVal = val;
    m_minVal = std::min(m_maxVal,m_minVal);
}

//---------------------------------------------------------------------------------
void IntMeta::setStepSize(int32 val)
{
#if _DEBUG
    if (val <= 0)
    {
        throw std::logic_error("stepSize of IntMeta must be >= 1");
    }
#endif
    m_stepSize = val;
}

//---------------------------------------------------------------------------------
//!< sets unit string of this parameter
void IntMeta::setUnit(const ito::ByteArray &unit)
{
    m_unit = unit;
}

//---------------------------------------------------------------------------------
void IntMeta::setRepresentation(ParamMeta::tRepresentation representation)
{
    m_representation = representation;
}

//---------------------------------------------------------------------------------
bool IntMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const IntMeta *other_ = (const IntMeta*)(&other);
    return ((m_minVal == other_->m_minVal) && \
        (m_maxVal == other_->m_maxVal) && \
        (m_stepSize == other_->m_stepSize) && \
        (m_unit == other_->m_unit) && \
        (m_representation == other_->m_representation));
}


//---------------------------------------------------------------------------------
DoubleMeta::DoubleMeta(float64 minVal, float64 maxVal, float64 stepSize /*=0.0*/ /*0.0 means no specific step size*/, ito::ByteArray category /*= ito::ByteArray()*/)
    : ParamMeta(rttiDoubleMeta, category),
    m_minVal(minVal),
    m_maxVal(maxVal),
    m_stepSize(stepSize),
    m_displayNotation(Automatic),
    m_displayPrecision(3),
    m_representation(ParamMeta::PureNumber)
{
    if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal);

    if (m_minVal == 0.0 && m_maxVal == 1.0 && m_stepSize == 1.0)
    {
        m_representation = ParamMeta::Boolean;
    }

#if _DEBUG
    if (stepSize < 0.0)
    {
        throw std::logic_error("stepSize of DoubleMeta must be >= 0.0");
    }
#endif
}

//---------------------------------------------------------------------------------
DoubleMeta::DoubleMeta(const DoubleMeta &copy) :
    ParamMeta(rttiDoubleMeta, copy.m_category),
    m_minVal(copy.m_minVal),
    m_maxVal(copy.m_maxVal),
    m_stepSize(copy.m_stepSize),
    m_displayNotation(copy.m_displayNotation),
    m_displayPrecision(copy.m_displayPrecision),
    m_representation(copy.m_representation)
{

}

//---------------------------------------------------------------------------------
DoubleMeta& DoubleMeta::operator=(const DoubleMeta &rhs)
{
    ParamMeta::operator=(rhs);
    m_type = rttiDoubleMeta;
    m_minVal = rhs.m_minVal;
    m_maxVal = rhs.m_maxVal;
    m_stepSize = rhs.m_stepSize;
    m_displayNotation = rhs.m_displayNotation;
    m_displayPrecision = rhs.m_displayPrecision;
    m_representation = rhs.m_representation;
    return *this;
}

//---------------------------------------------------------------------------------
DoubleMeta* DoubleMeta::all(ito::ByteArray category /*= ito::ByteArray()*/)
{
    return new DoubleMeta(-std::numeric_limits<float64>::max(), std::numeric_limits<float64>::max() );
}

//---------------------------------------------------------------------------------
//!< sets unit string of this parameter
void DoubleMeta::setUnit(const ito::ByteArray &unit)
{
    m_unit = unit;
}

//---------------------------------------------------------------------------------
//!< sets display precision
void DoubleMeta::setDisplayPrecision(int displayPrecision)
{
    m_displayPrecision = displayPrecision;
}


//---------------------------------------------------------------------------------
//!< sets display notation
void DoubleMeta::setDisplayNotation(DoubleMeta::tDisplayNotation displayNotation)
{
    m_displayNotation = displayNotation;
}

//---------------------------------------------------------------------------------
void DoubleMeta::setMin(float64 val)
{
    m_minVal = val;
    m_maxVal = std::max(m_maxVal,m_minVal);
}

//---------------------------------------------------------------------------------
void DoubleMeta::setMax(float64 val)
{
    m_maxVal = val;
    m_minVal = std::min(m_maxVal,m_minVal);
}

//---------------------------------------------------------------------------------
void DoubleMeta::setStepSize(float64 val)
{
#if _DEBUG
    if (val < 0.0)
    {
        throw std::logic_error("stepSize of DoubleMeta must be >= 0.0");
    }
#endif
    m_stepSize = val;
}

//---------------------------------------------------------------------------------
void DoubleMeta::setRepresentation(ParamMeta::tRepresentation representation)
{
    m_representation = representation;
}

//---------------------------------------------------------------------------------
bool DoubleMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const DoubleMeta *other_ = (const DoubleMeta*)(&other);
    double eps = std::numeric_limits<double>::epsilon();
    return ((std::abs(m_minVal - other_->m_minVal) < eps) && \
        (std::abs(m_maxVal - other_->m_maxVal) < eps) && \
        (std::abs(m_stepSize - other_->m_stepSize) < eps) && \
        (m_unit == other_->m_unit) && \
        (m_displayPrecision == other_->m_displayPrecision) && \
        (m_displayNotation == other_->m_displayNotation) && \
        (m_representation == other_->m_representation));
}

//---------------------------------------------------------------------------------
//! constructor
    /*!
        creates HWMeta-information struct where you can pass a bitmask which consists of values of the enumeration
        ito::tPluginType. The plugin reference of the corresponding Param should then only accept plugins, where
        all bits are set, too.
        \sa ito::Plugin, ito::tPluginType
    */
HWMeta::HWMeta(uint32 minType, ito::ByteArray category /*= ito::ByteArray()*/) :
    ParamMeta(rttiHWMeta, category),
    m_minType(minType)
{
}

//---------------------------------------------------------------------------------
//! constructor
/*!
    creates HWMeta-information struct where you can pass a specific name of a plugin, which only is
    allowed by the corresponding plugin-instance.
    \sa ito::Plugin
*/
HWMeta::HWMeta(const ito::ByteArray &hwAddInName, ito::ByteArray category /*= ito::ByteArray()*/) :
    ParamMeta(rttiHWMeta, category),
    m_minType(0),
    m_HWName(hwAddInName)
{
}

//---------------------------------------------------------------------------------
//!< copy constructor
HWMeta::HWMeta(const HWMeta &cpy) :
    ParamMeta(rttiHWMeta, cpy.m_category),
    m_minType(cpy.m_minType),
    m_HWName(cpy.m_HWName)
{
}

//---------------------------------------------------------------------------------
bool HWMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const HWMeta *other_ = (const HWMeta*)(&other);
    return ((m_minType == other_->m_minType) && \
        (m_HWName == other_->m_HWName));
}

//---------------------------------------------------------------------------------
HWMeta& HWMeta::operator=(const HWMeta &rhs)
{
    ParamMeta::operator=(rhs);
    m_type = rttiHWMeta;
    m_HWName = rhs.m_HWName;
    m_minType = rhs.m_minType;
    return *this;
}

//---------------------------------------------------------------------------------
class StringMetaPrivate
{
public:
    StringMetaPrivate(StringMeta::tType type) :
        m_stringType(type),
        m_len(0)
    {
    }

    StringMeta::tType m_stringType;
    int m_len;
    std::vector<ito::ByteArray> m_items;
};

//---------------------------------------------------------------------------------
StringMeta::StringMeta(tType type, ito::ByteArray category /*= ito::ByteArray()*/)
    : ParamMeta(rttiStringMeta, category),
    p(new StringMetaPrivate(type))
{
}

//---------------------------------------------------------------------------------
StringMeta::StringMeta(tType type, const char* val, ito::ByteArray category /*= ito::ByteArray()*/)
    : ParamMeta(rttiStringMeta, category),
    p(new StringMetaPrivate(type))
{
    if(val)
    {
        p->m_items.push_back(val);
        p->m_len = 1;
    }
    else
    {
        p->m_len = 0;
    }
}

//---------------------------------------------------------------------------------
StringMeta::StringMeta(tType type, const ito::ByteArray &val, ito::ByteArray category /*= ito::ByteArray()*/)
    : ParamMeta(rttiStringMeta, category),
    p(new StringMetaPrivate(type))
{
    if (val.empty() == false)
    {
        p->m_items.push_back(val);
        p->m_len = 1;
    }
    else
    {
        p->m_len = 0;
    }
}

//---------------------------------------------------------------------------------
StringMeta::StringMeta(const StringMeta& cpy) :
    ParamMeta(rttiStringMeta, cpy.m_category),
    p(new StringMetaPrivate(*(cpy.p)))
{
}

//---------------------------------------------------------------------------------
/*virtual*/ StringMeta::~StringMeta()
{
    delete p;
    p = nullptr;
}

//---------------------------------------------------------------------------------
bool StringMeta::addItem(const char *val)
{
    p->m_items.push_back(val);
    p->m_len++;
    return true;
}

//---------------------------------------------------------------------------------
bool StringMeta::addItem(const ito::ByteArray &val)
{
    p->m_items.push_back(val);
    p->m_len++;
    return true;
}

//---------------------------------------------------------------------------------
void StringMeta::clearItems()
{
    p->m_len = 0;
    p->m_items.clear();
}

//---------------------------------------------------------------------------------
void StringMeta::setStringType(tType type)
{
    p->m_stringType = type;
}

//---------------------------------------------------------------------------------
StringMeta::tType StringMeta::getStringType() const
{
    return p->m_stringType;
}

//---------------------------------------------------------------------------------
int StringMeta::getLen() const
{
    return p->m_len;
}

//---------------------------------------------------------------------------------
StringMeta & StringMeta::operator += (const char *val)
{
    addItem(val);
    return *this;
}

//---------------------------------------------------------------------------------
StringMeta & StringMeta::operator = (const StringMeta &rhs)
{
    ParamMeta::operator=(rhs);

    m_type = rttiStringMeta;

    if (rhs.p)
    {
        p->m_items = rhs.p->m_items;
        p->m_len = rhs.p->m_len;
        p->m_stringType = rhs.p->m_stringType;
    }

    return *this;
}

//---------------------------------------------------------------------------------
const char* StringMeta::getString(int idx /*= 0*/) const
{
    return (idx >= p->m_len) ? nullptr : p->m_items[idx].data();
}

//---------------------------------------------------------------------------------
bool StringMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const StringMeta *other_ = (const StringMeta*)(&other);
    if ((p->m_len == other_->p->m_len) && \
        (p->m_stringType == other_->p->m_stringType) && \
        (p->m_items.size() == other_->p->m_items.size()))
    {
        for (size_t i = 0; i < p->m_items.size(); ++i)
        {
            if (p->m_items[i] != other_->p->m_items[i])
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

//---------------------------------------------------------------------------------
DObjMeta::DObjMeta(int minDim /*= 0*/, int maxDim /*= (std::numeric_limits<int>::max)()*/,
    ito::ByteArray category /*= ito::ByteArray()*/)
    : ParamMeta(rttiDObjMeta, category), m_allowedTypes(ito::ByteArray()), m_minDim(minDim), m_maxDim(maxDim)
{
}

//---------------------------------------------------------------------------------
bool DObjMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const DObjMeta *other_ = (const DObjMeta*)(&other);
    return ((m_allowedTypes == other_->m_allowedTypes) && \
        (m_minDim == other_->m_minDim) && \
        (m_maxDim == other_->m_maxDim));
}

//---------------------------------------------------------------------------------
//! copy constructor
DObjMeta::DObjMeta(const DObjMeta &cpy) :
    ParamMeta(rttiDObjMeta, cpy.m_category),
    m_allowedTypes(cpy.m_allowedTypes),
    m_maxDim(cpy.m_maxDim),
    m_minDim(cpy.m_minDim)
{
}

//---------------------------------------------------------------------------------
DObjMeta& DObjMeta::operator=(const DObjMeta &rhs)
{
    ParamMeta::operator=(rhs);
    m_type = rttiDObjMeta;
    m_maxDim = rhs.m_maxDim;
    m_minDim = rhs.m_minDim;
    m_allowedTypes = rhs.m_allowedTypes;
    return *this;
}

//---------------------------------------------------------------------------------
void DObjMeta::appendAllowedDataType(ito::tDataType dataType)
{
    bool exists = false;
    int num = getNumAllowedDataTypes();

    if (num > 0)
    {
        exists = isDataTypeAllowed(dataType);
    }

    if (!exists)
    {
        const char* buf_old = m_allowedTypes.data();
        char* buf = new char[num + 2];
        char new_val = dataType;
        buf[num] = new_val;
        buf[num + 1] = '\0';
        size_t new_i = 0;
        bool added = false;

        for (int i = 0; i < num; ++i)
        {
            if (added || (buf_old[i] < new_val))
            {
                buf[new_i] = buf_old[i];
                new_i++;
            }
            else
            {
                buf[new_i] = new_val;
                buf[new_i + 1] = buf_old[i];
                new_i += 2;
                added = true;
            }
        }

        m_allowedTypes = buf;
        delete[] buf;
    }
}

//---------------------------------------------------------------------------------
bool DObjMeta::isDataTypeAllowed(ito::tDataType dataType) const
{
    if (m_allowedTypes.size() == 0)
    {
        return true;
    }

    const char* types = m_allowedTypes.data();
    ito::tDataType type;

    for (int i = 0; i < m_allowedTypes.size(); ++i)
    {
        type = (ito::tDataType)(types[i]);

        if (type == dataType)
        {
            return true;
        }
    }

    return false;
}

//---------------------------------------------------------------------------------
int DObjMeta::getNumAllowedDataTypes() const
{
    return m_allowedTypes.size();
}

//---------------------------------------------------------------------------------
ito::tDataType DObjMeta::getAllowedDataType(int index) const
{
    if (index < 0 || index >= m_allowedTypes.size())
    {
        throw std::logic_error("index must be in range [0, getNumAllowedDataTypes())");
    }

    return (ito::tDataType)(m_allowedTypes.data()[index]);
}

//---------------------------------------------------------------------------------
ListMeta::ListMeta(size_t numMin, size_t numMax, size_t numStepSize /*= 1*/) :
    m_numMin(numMin),
    m_numMax(numMax),
    m_numStep(numStepSize)
{
    if (m_numMax < m_numMin)
    {
        m_numMax = m_numMin;
    }
}

//---------------------------------------------------------------------------------
ListMeta::ListMeta() :
    m_numMin(0),
    m_numMax(std::numeric_limits<size_t>::max()),
    m_numStep(1)
{

}

//---------------------------------------------------------------------------------
//!< destructor
ListMeta::~ListMeta()
{

}

//---------------------------------------------------------------------------------
void ListMeta::setNumMin(size_t val)
{
    m_numMin = val;
}

//---------------------------------------------------------------------------------
void ListMeta::setNumMax(size_t val)
{
    m_numMax = val;
}

//---------------------------------------------------------------------------------
void ListMeta::setNumStepSize(size_t val)
{
    m_numStep = val;
}

//---------------------------------------------------------------------------------
bool ListMeta::operator==(const ListMeta& other) const
{
    return ((m_numMin == other.m_numMin) && \
        (m_numMax == other.m_numMax) && \
        (m_numStep == other.m_numStep));
}


//---------------------------------------------------------------------------------
CharArrayMeta::CharArrayMeta(char minVal, char maxVal, char stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    CharMeta(minVal, maxVal, stepSize, category),
    ListMeta()
{
    m_type = rttiCharArrayMeta;
}

//---------------------------------------------------------------------------------
CharArrayMeta::CharArrayMeta(char minVal, char maxVal, char stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    CharMeta(minVal, maxVal, stepSize, category),
    ListMeta(numMin, numMax, numStepSize)
{
    m_type = rttiCharArrayMeta;
}

//---------------------------------------------------------------------------------
CharArrayMeta::CharArrayMeta(const CharArrayMeta &cpy) :
    CharMeta(cpy),
    ListMeta(cpy)
{
    m_type = rttiCharArrayMeta;
}

//---------------------------------------------------------------------------------
bool CharArrayMeta::operator==(const ParamMeta& other) const
{
    if (!CharMeta::operator==(other))
        return false;

    const ListMeta* lm = dynamic_cast<const ListMeta*>(&other);

    if (lm)
    {
        return ListMeta::operator==(*lm);
    }

    return false;
}

//---------------------------------------------------------------------------------
//!< assignment operator
CharArrayMeta &CharArrayMeta::operator=(const CharArrayMeta &rhs)
{
    CharMeta::operator=(rhs);
    ListMeta::operator=(rhs);
    m_type = rttiCharArrayMeta;
    return *this;
}

//---------------------------------------------------------------------------------
IntArrayMeta::IntArrayMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    IntMeta(minVal, maxVal, stepSize, category),
    ListMeta()
{
    m_type = rttiIntArrayMeta;
}

//---------------------------------------------------------------------------------
IntArrayMeta::IntArrayMeta(int32 minVal, int32 maxVal, int32 stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    IntMeta(minVal, maxVal, stepSize, category),
    ListMeta(numMin, numMax, numStepSize)
{
    m_type = rttiIntArrayMeta;
}

//---------------------------------------------------------------------------------
IntArrayMeta::IntArrayMeta(const IntArrayMeta &cpy) :
    IntMeta(cpy),
    ListMeta(cpy)
{
    m_type = rttiIntArrayMeta;
}

//---------------------------------------------------------------------------------
bool IntArrayMeta::operator==(const ParamMeta& other) const
{
    if (!IntMeta::operator==(other))
        return false;

    const ListMeta* lm = dynamic_cast<const ListMeta*>(&other);

    if (lm)
    {
        return ListMeta::operator==(*lm);
    }

    return false;
}

//---------------------------------------------------------------------------------
//!< assignment operator
IntArrayMeta &IntArrayMeta::operator=(const IntArrayMeta &rhs)
{
    IntMeta::operator=(rhs);
    ListMeta::operator=(rhs);
    m_type = rttiIntArrayMeta;
    return *this;
}


//---------------------------------------------------------------------------------
DoubleArrayMeta::DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    DoubleMeta(minVal, maxVal, stepSize, category),
    ListMeta()
{
    m_type = rttiDoubleArrayMeta;
}

//---------------------------------------------------------------------------------
DoubleArrayMeta::DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    DoubleMeta(minVal, maxVal, stepSize, category),
    ListMeta(numMin, numMax, numStepSize)
{
    m_type = rttiDoubleArrayMeta;
}

//---------------------------------------------------------------------------------
DoubleArrayMeta::DoubleArrayMeta(const DoubleArrayMeta &cpy) :
    DoubleMeta(cpy),
    ListMeta(cpy)
{
    m_type = rttiDoubleArrayMeta;
}

//---------------------------------------------------------------------------------
bool DoubleArrayMeta::operator==(const ParamMeta& other) const
{
    if (!DoubleMeta::operator==(other))
        return false;

    const ListMeta* lm = dynamic_cast<const ListMeta*>(&other);

    if (lm)
    {
        return ListMeta::operator==(*lm);
    }

    return false;
}

//---------------------------------------------------------------------------------
//!< assignment operator
DoubleArrayMeta &DoubleArrayMeta::operator=(const DoubleArrayMeta &rhs)
{
    DoubleMeta::operator=(rhs);
    ListMeta::operator=(rhs);
    m_type = rttiDoubleArrayMeta;
    return *this;
}

//---------------------------------------------------------------------------------
IntervalMeta::IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    IntMeta(minVal, maxVal, stepSize, category),
    m_sizeMin(0),
    m_sizeMax(std::numeric_limits<int32>::max()),
    m_sizeStep(1),
    m_isIntervalNotRange(true)
{
    m_type = rttiIntervalMeta;
}

//---------------------------------------------------------------------------------
IntervalMeta::IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize, int32 intervalMin, int32 intervalMax, int32 intervalStep /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    IntMeta(minVal, maxVal, stepSize, category),
    m_sizeMin(intervalMin),
    m_sizeMax(intervalMax),
    m_sizeStep(intervalStep),
    m_isIntervalNotRange(true)
{
    if (m_sizeMax < m_sizeMin)
        m_sizeMax = m_sizeMin;
    m_type = rttiIntervalMeta;
}

//---------------------------------------------------------------------------------
IntervalMeta::IntervalMeta(const IntervalMeta &cpy) :
    IntMeta(cpy),
    m_sizeMin(cpy.m_sizeMin),
    m_sizeMax(cpy.m_sizeMax),
    m_sizeStep(cpy.m_sizeStep),
    m_isIntervalNotRange(true)
{
    m_type = rttiIntervalMeta;
}

//---------------------------------------------------------------------------------
//!< assignment operator
IntervalMeta &IntervalMeta::operator=(const IntervalMeta &rhs)
{
    IntMeta::operator=(rhs);
    m_sizeMin = rhs.m_sizeMin;
    m_sizeMax = rhs.m_sizeMax;
    m_sizeStep = rhs.m_sizeStep;
    m_isIntervalNotRange = true;
    m_type = rttiIntervalMeta;
    return *this;
}

//---------------------------------------------------------------------------------
void IntervalMeta::setIntervalMin(int32 val)
{
    m_sizeMin = val;
    m_sizeMax = std::max(m_sizeMin, m_sizeMax);
}

//---------------------------------------------------------------------------------
void IntervalMeta::setIntervalMax(int32 val)
{
    m_sizeMax = val;
    m_sizeMin = std::min(m_sizeMin, m_sizeMax);
}

//---------------------------------------------------------------------------------
void IntervalMeta::setIntervalStep(int32 val)
{
    m_sizeStep = val;
}

//---------------------------------------------------------------------------------
bool IntervalMeta::operator==(const ParamMeta& other) const
{
    if (!IntMeta::operator==(other))
        return false;

    const IntervalMeta *other_ = (const IntervalMeta*)(&other);
    return ((m_sizeMin == other_->m_sizeMin) && \
        (m_sizeMax == other_->m_sizeMax) && \
        (m_sizeStep == other_->m_sizeStep) && \
        (m_isIntervalNotRange == other_->m_isIntervalNotRange));
}

//---------------------------------------------------------------------------------
RangeMeta::RangeMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    IntervalMeta(minVal, maxVal, stepSize, category)
{
    m_type = rttiRangeMeta;
    m_isIntervalNotRange = false;
}

//---------------------------------------------------------------------------------
RangeMeta::RangeMeta(int32 minVal, int32 maxVal, int32 stepSize, size_t sizeMin, size_t sizeMax, size_t sizeStep /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    IntervalMeta(minVal, maxVal, stepSize, sizeMin, sizeMax, sizeStep, category)
{
    m_type = rttiRangeMeta;
    m_isIntervalNotRange = false;
}

//---------------------------------------------------------------------------------
RangeMeta::RangeMeta(const RangeMeta &cpy) :
    IntervalMeta(cpy)
{
    m_isIntervalNotRange = false;
    m_type = rttiRangeMeta;
}

//---------------------------------------------------------------------------------
//!< assignment operator
RangeMeta &RangeMeta::operator=(const RangeMeta &rhs)
{
    IntervalMeta::operator=(rhs);
    m_isIntervalNotRange = false;
    m_type = rttiRangeMeta;
    return *this;
}


//---------------------------------------------------------------------------------
DoubleIntervalMeta::DoubleIntervalMeta(float64 minVal, float64 maxVal, float64 stepSize /*= 0.0*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    DoubleMeta(minVal, maxVal, stepSize, category),
    m_sizeMin(0.0),
    m_sizeMax(std::numeric_limits<float64>::max()),
    m_sizeStep(0.0)
{
    m_type = rttiDoubleIntervalMeta;
}

//---------------------------------------------------------------------------------
DoubleIntervalMeta::DoubleIntervalMeta(float64 minVal, float64 maxVal, float64 stepSize, float64 sizeMin, float64 sizeMax, float64 sizeStep /*= 0.0*/, ito::ByteArray category /*= ito::ByteArray()*/) :
    DoubleMeta(minVal, maxVal, stepSize, category),
    m_sizeMin(sizeMin),
    m_sizeMax(sizeMax),
    m_sizeStep(sizeStep)
{
    if (m_sizeMax < m_sizeMin)
    {
        m_sizeMax = m_sizeMin;
    }

    m_type = rttiDoubleIntervalMeta;
}

//---------------------------------------------------------------------------------
DoubleIntervalMeta::DoubleIntervalMeta(const DoubleIntervalMeta &cpy) :
    DoubleMeta(cpy),
    m_sizeMin(cpy.m_sizeMin),
    m_sizeMax(cpy.m_sizeMax),
    m_sizeStep(cpy.m_sizeStep)
{
    m_type = rttiDoubleIntervalMeta;
}

//---------------------------------------------------------------------------------
//!< assignment operator
DoubleIntervalMeta &DoubleIntervalMeta::operator=(const DoubleIntervalMeta &rhs)
{
    DoubleMeta::operator=(rhs);
    m_sizeMin = rhs.m_sizeMin;
    m_sizeMax = rhs.m_sizeMax;
    m_sizeStep = rhs.m_sizeStep;
    m_type = rttiDoubleIntervalMeta;
    return *this;
}

//---------------------------------------------------------------------------------
void DoubleIntervalMeta::setSizeMin(float64 val)
{
    m_sizeMin = val;
    m_sizeMax = std::max(m_sizeMin, m_sizeMax);
}

//---------------------------------------------------------------------------------
void DoubleIntervalMeta::setSizeMax(float64 val)
{
    m_sizeMax = val;
    m_sizeMin = std::min(m_sizeMin, m_sizeMax);
}

//---------------------------------------------------------------------------------
void DoubleIntervalMeta::setSizeStep(float64 val)
{
    m_sizeStep = val;
}

//---------------------------------------------------------------------------------
bool DoubleIntervalMeta::operator==(const ParamMeta& other) const
{
    if (!DoubleMeta::operator==(other))
        return false;

    double eps = std::numeric_limits<double>::epsilon();
    const DoubleIntervalMeta *other_ = (const DoubleIntervalMeta*)(&other);
    return ((std::abs(m_sizeMin - other_->m_sizeMin) < eps) && \
        (std::abs(m_sizeMax - other_->m_sizeMax) < eps) && \
        (std::abs(m_sizeStep - other_->m_sizeStep) < eps));
}

//---------------------------------------------------------------------------------
RectMeta::RectMeta(const ito::RangeMeta &widthMeta, const ito::RangeMeta &heightMeta, ito::ByteArray category /*= ito::ByteArray()*/) :
    ParamMeta(rttiRectMeta, category),
    m_widthMeta(widthMeta),
    m_heightMeta(heightMeta)
{

}

//---------------------------------------------------------------------------------
//! copy constructor
RectMeta::RectMeta(const RectMeta &cpy) :
    ParamMeta(rttiRectMeta, cpy.m_category),
    m_widthMeta(cpy.m_widthMeta),
    m_heightMeta(cpy.m_heightMeta)
{
    m_type = rttiRectMeta;
}

//---------------------------------------------------------------------------------
//!< assignment operator
RectMeta &RectMeta::operator=(const RectMeta &rhs)
{
    ParamMeta::operator=(rhs);
    m_widthMeta = rhs.m_widthMeta;
    m_heightMeta = rhs.m_heightMeta;
    m_type = rttiRectMeta;
    return *this;
}

//---------------------------------------------------------------------------------
void RectMeta::setWidthRangeMeta(const ito::RangeMeta &widthMeta)
{
    m_widthMeta = widthMeta;
}

//---------------------------------------------------------------------------------
void RectMeta::setHeightRangeMeta(const ito::RangeMeta &heightMeta)
{
    m_heightMeta = heightMeta;
}

//---------------------------------------------------------------------------------
bool RectMeta::operator==(const ParamMeta& other) const
{
    if (!ParamMeta::operator==(other))
        return false;

    const RectMeta *other_ = (const RectMeta*)(&other);
    return ((m_heightMeta == other_->m_heightMeta) && \
        (m_widthMeta == other_->m_widthMeta));
}

//---------------------------------------------------------------------------------
//!< sets unit string of this parameter
void RectMeta::setUnit(const ito::ByteArray &unit)
{
    m_heightMeta.setUnit(unit);
}

//---------------------------------------------------------------------------------
StringListMeta::StringListMeta(tType type, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/,
    ito::ByteArray category /*= ito::ByteArray()*/) :
    StringMeta(type),
    ListMeta(numMin, numMax, numStepSize)
{
    m_type = rttiStringListMeta;

    if (!category.empty())
    {
        setCategory(category);
    }
}

//---------------------------------------------------------------------------------
StringListMeta::StringListMeta(tType type, const char *val, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/,
    ito::ByteArray category /*= ito::ByteArray()*/) :
    StringMeta(type, val, category),
    ListMeta(numMin, numMax, numStepSize)
{
    m_type = rttiStringListMeta;
}

//---------------------------------------------------------------------------------
StringListMeta::StringListMeta(tType type, const ito::ByteArray &val, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/,
    ito::ByteArray category /*= ito::ByteArray()*/) :
    StringMeta(type, val, category),
    ListMeta(numMin, numMax, numStepSize)
{
    m_type = rttiStringListMeta;
}

//---------------------------------------------------------------------------------
StringListMeta::StringListMeta(const StringListMeta &cpy) :
    StringMeta(cpy),
    ListMeta(cpy)
{
    m_type = rttiStringListMeta;
}

//---------------------------------------------------------------------------------
StringListMeta::~StringListMeta()
{

}

//---------------------------------------------------------------------------------
bool StringListMeta::operator==(const ParamMeta& other) const
{
    if (!StringMeta::operator==(other))
        return false;

    const ListMeta* lm = dynamic_cast<const ListMeta*>(&other);

    if (lm)
    {
        return ListMeta::operator==(*lm);
    }

    return false;
}

//---------------------------------------------------------------------------------
//!< assignment operator
StringListMeta &StringListMeta::operator=(const StringListMeta &rhs)
{
    StringMeta::operator=(rhs);
    ListMeta::operator=(rhs);
    m_type = rttiStringListMeta;
    return *this;
}


} //end namespace ito
