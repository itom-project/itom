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

    //---------------------------------------------------------------------------------
	CharMeta::CharMeta(char minVal, char maxVal, char stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/)
        : ParamMeta(rttiCharMeta, category), 
        m_minVal(minVal), 
        m_maxVal(maxVal), 
		m_stepSize(stepSize),
		m_representation(ParamMeta::Linear)
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
	IntMeta::IntMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/, ito::ByteArray category /*= ito::ByteArray()*/)
        : ParamMeta(rttiIntMeta, category), 
        m_minVal(minVal), 
        m_maxVal(maxVal), 
        m_stepSize(stepSize),
		m_representation(ParamMeta::Linear)
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
	DoubleMeta::DoubleMeta(float64 minVal, float64 maxVal, float64 stepSize /*=0.0*/ /*0.0 means no specific step size*/, ito::ByteArray category /*= ito::ByteArray()*/)
        : ParamMeta(rttiDoubleMeta, category), 
        m_minVal(minVal), 
        m_maxVal(maxVal), 
        m_stepSize(stepSize),
		m_displayNotation(Automatic),
		m_displayPrecision(3),
		m_representation(ParamMeta::Linear)
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
		p = NULL;
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
		return (idx >= p->m_len) ? NULL : p->m_items[idx].data();
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
            m_numMax = m_numMin;
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
            m_numMax = m_numMin;
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
            m_numMax = m_numMin;
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
            m_sizeMax = m_sizeMin;
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
        

} //end namespace ito

