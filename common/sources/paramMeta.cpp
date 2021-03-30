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
    CharMeta::CharMeta(char minVal, char maxVal, char stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/)
        : ParamMeta(rttiCharMeta, category), 
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
    IntMeta::IntMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/)
        : ParamMeta(rttiIntMeta, category), 
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
        m_HWName = rhs.m_HWName;
        m_minType = rhs.m_minType;
        return *this;
    }

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
    StringMeta::StringMeta(const StringMeta& cpy) 
        : ParamMeta(cpy),
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
    DObjMeta& DObjMeta::operator=(const DObjMeta &rhs)
    {
        ParamMeta::operator=(rhs);
        m_maxDim = rhs.m_maxDim;
        m_minDim = rhs.m_minDim;
        m_allowedTypes = rhs.m_allowedTypes;
        return *this;
    }


    //---------------------------------------------------------------------------------
    CharArrayMeta::CharArrayMeta(char minVal, char maxVal, char stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        CharMeta(minVal, maxVal, stepSize, category),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiCharArrayMeta;
    }

    //---------------------------------------------------------------------------------
    CharArrayMeta::CharArrayMeta(char minVal, char maxVal, char stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        CharMeta(minVal, maxVal, stepSize, category),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
    {
        if (m_numMax < m_numMin)
        {
            m_numMax = m_numMin;
        }

        m_type = rttiCharArrayMeta;
    }

    //---------------------------------------------------------------------------------
    CharArrayMeta::CharArrayMeta(const CharArrayMeta &cpy) :
        CharMeta(cpy),
        m_numMin(cpy.m_numMin),
        m_numMax(cpy.m_numMax),
        m_numStep(cpy.m_numStep)
    {
        m_type = rttiCharArrayMeta;
    }

    //---------------------------------------------------------------------------------
    void CharArrayMeta::setNumMin(size_t val)
    {
        m_numMin = val; 
        m_numMax = std::max(m_numMin,m_numMax); 
    }
        
    //---------------------------------------------------------------------------------
    void CharArrayMeta::setNumMax(size_t val)
    {
        m_numMax = val; 
        m_numMin = std::min(m_numMin,m_numMax);
    }

    //---------------------------------------------------------------------------------
    void CharArrayMeta::setNumStepSize(size_t val)
    {
        m_numStep = val;
    }

    //---------------------------------------------------------------------------------
    bool CharArrayMeta::operator==(const ParamMeta& other) const
    {
        if (!CharMeta::operator==(other))
            return false;

        const CharArrayMeta *other_ = (const CharArrayMeta*)(&other);
        return ((m_numMin == other_->m_numMin) && \
            (m_numMax == other_->m_numMax) && \
            (m_numStep == other_->m_numStep));
    }


    
    //---------------------------------------------------------------------------------
    IntArrayMeta::IntArrayMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        IntMeta(minVal, maxVal, stepSize, category),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiIntArrayMeta;
    }

    //---------------------------------------------------------------------------------
    IntArrayMeta::IntArrayMeta(int32 minVal, int32 maxVal, int32 stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        IntMeta(minVal, maxVal, stepSize, category),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
    {
        if (m_numMax < m_numMin)
        {
            m_numMax = m_numMin;
        }

        m_type = rttiIntArrayMeta;
    }

    //---------------------------------------------------------------------------------
    IntArrayMeta::IntArrayMeta(const IntArrayMeta &cpy) :
        IntMeta(cpy),
        m_numMin(cpy.m_numMin),
        m_numMax(cpy.m_numMax),
        m_numStep(cpy.m_numStep)
    {
        m_type = rttiIntArrayMeta;
    }

    //---------------------------------------------------------------------------------
    void IntArrayMeta::setNumMin(size_t val)
    {
        m_numMin = val; 
        m_numMax = std::max(m_numMin,m_numMax); 
    }
        
    //---------------------------------------------------------------------------------
    void IntArrayMeta::setNumMax(size_t val)
    {
        m_numMax = val; 
        m_numMin = std::min(m_numMin,m_numMax);
    }

    //---------------------------------------------------------------------------------
    void IntArrayMeta::setNumStepSize(size_t val)
    {
        m_numStep = val;
    }

    //---------------------------------------------------------------------------------
    bool IntArrayMeta::operator==(const ParamMeta& other) const
    {
        if (!IntMeta::operator==(other))
            return false;

        const IntArrayMeta *other_ = (const IntArrayMeta*)(&other);
        return ((m_numMin == other_->m_numMin) && \
            (m_numMax == other_->m_numMax) && \
            (m_numStep == other_->m_numStep));
    }


    //---------------------------------------------------------------------------------
    DoubleArrayMeta::DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        DoubleMeta(minVal, maxVal, stepSize, category),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiDoubleArrayMeta;
    }

    //---------------------------------------------------------------------------------
    DoubleArrayMeta::DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        DoubleMeta(minVal, maxVal, stepSize, category),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
    {
        if (m_numMax < m_numMin)
        {
            m_numMax = m_numMin;
        }

        m_type = rttiDoubleArrayMeta;
    }

    //---------------------------------------------------------------------------------
    DoubleArrayMeta::DoubleArrayMeta(const DoubleArrayMeta &cpy) :
        DoubleMeta(cpy),
        m_numMin(cpy.m_numMin),
        m_numMax(cpy.m_numMax),
        m_numStep(cpy.m_numStep)
    {
        m_type = rttiDoubleArrayMeta;
    }

    //---------------------------------------------------------------------------------
    void DoubleArrayMeta::setNumMin(size_t val)
    {
        m_numMin = val; 
        m_numMax = std::max(m_numMin,m_numMax); 
    }
        
    //---------------------------------------------------------------------------------
    void DoubleArrayMeta::setNumMax(size_t val)
    {
        m_numMax = val; 
        m_numMin = std::min(m_numMin,m_numMax);
    }

    //---------------------------------------------------------------------------------
    void DoubleArrayMeta::setNumStepSize(size_t val)
    {
        m_numStep = val;
    }

    //---------------------------------------------------------------------------------
    bool DoubleArrayMeta::operator==(const ParamMeta& other) const
    {
        if (!DoubleMeta::operator==(other))
            return false;

        const DoubleArrayMeta *other_ = (const DoubleArrayMeta*)(&other);
        return ((m_numMin == other_->m_numMin) && \
            (m_numMax == other_->m_numMax) && \
            (m_numStep == other_->m_numStep));
    }
    

    


    //---------------------------------------------------------------------------------
    IntervalMeta::IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        IntMeta(minVal, maxVal, stepSize, category),
        m_sizeMin(0),
        m_sizeMax(std::numeric_limits<int32>::max()),
        m_sizeStep(1),
        m_isIntervalNotRange(false)
    {
        m_type = rttiIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    IntervalMeta::IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize, int32 intervalMin, int32 intervalMax, int32 intervalStep /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/) :
        IntMeta(minVal, maxVal, stepSize, category),
        m_sizeMin(intervalMin),
        m_sizeMax(intervalMax),
        m_sizeStep(intervalStep),
        m_isIntervalNotRange(false)
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
        m_isIntervalNotRange(cpy.m_isIntervalNotRange)
    {
        m_type = rttiIntervalMeta;
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
    StringListMeta::StringListMeta(tType type, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/,
        ito::ByteArray category /*= ito::ByteArray()*/) :
        StringMeta(type),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
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
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
    {
        m_type = rttiStringListMeta;
    }

    //---------------------------------------------------------------------------------
    StringListMeta::StringListMeta(tType type, const ito::ByteArray &val, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/,
        ito::ByteArray category /*= ito::ByteArray()*/) :
        StringMeta(type, val, category),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
    {
        m_type = rttiStringListMeta;

    }

    //---------------------------------------------------------------------------------
    StringListMeta::StringListMeta(const StringListMeta &cpy) :
        StringMeta(cpy),
        m_numMin(cpy.m_numMin),
        m_numMax(cpy.m_numMax),
        m_numStep(cpy.m_numStep)
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

        const StringListMeta *other_ = (const StringListMeta*)(&other);
        return ((m_numMin == other_->m_numMin) && \
            (m_numMax == other_->m_numMax) && \
            (m_numStep == other_->m_numStep));
    }
        

} //end namespace ito

