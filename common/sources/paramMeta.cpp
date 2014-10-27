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

/* includes */
#include "../paramMeta.h"

#if _DEBUG
#include <stdexcept>
#endif


namespace ito
{
    //---------------------------------------------------------------------------------
    CharMeta::CharMeta(char minVal, char maxVal, char stepSize /*= 1*/)
        : ParamMeta(rttiCharMeta), 
        m_minVal(minVal), 
        m_maxVal(maxVal), 
        m_stepSize(stepSize) 
    { 
        if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); 

#if _DEBUG
        if (stepSize <= 0)
        {
            throw std::logic_error("stepSize of CharMeta must be >= 1");
        }
#endif
    }

    //---------------------------------------------------------------------------------
    CharMeta* CharMeta::all() 
    { 
        return new CharMeta(std::numeric_limits<char>::min(), std::numeric_limits<char>::max() ); 
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
    IntMeta::IntMeta(int minVal, int maxVal, int stepSize /*= 1*/)
        : ParamMeta(rttiIntMeta), 
        m_minVal(minVal), 
        m_maxVal(maxVal), 
        m_stepSize(stepSize) 
    { 
        if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); 

#if _DEBUG
        if (stepSize <= 0)
        {
            throw std::logic_error("stepSize of IntMeta must be >= 1");
        }
#endif
    }

    //---------------------------------------------------------------------------------
    IntMeta* IntMeta::all() 
    { 
        return new IntMeta(std::numeric_limits<int>::min(), std::numeric_limits<int>::max() ); 
    }

    //---------------------------------------------------------------------------------
    void IntMeta::setMin(int val)
    { 
        m_minVal = val; 
        m_maxVal = std::max(m_maxVal,m_minVal); 
    }
    
    //---------------------------------------------------------------------------------
    void IntMeta::setMax(int val)
    { 
        m_maxVal = val; 
        m_minVal = std::min(m_maxVal,m_minVal); 
    }

    //---------------------------------------------------------------------------------
    void IntMeta::setStepSize(int val)
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
    DoubleMeta::DoubleMeta(double minVal, double maxVal, double stepSize /*=0.0*/ /*0.0 means no specific step size*/)
        : ParamMeta(rttiDoubleMeta), 
        m_minVal(minVal), 
        m_maxVal(maxVal), 
        m_stepSize(stepSize) 
    { 
        if(m_maxVal < m_minVal) std::swap(m_minVal,m_maxVal); 

#if _DEBUG
        if (stepSize < 0.0)
        {
            throw std::logic_error("stepSize of DoubleMeta must be >= 0.0");
        }
#endif
    }

    //---------------------------------------------------------------------------------
    DoubleMeta* DoubleMeta::all() 
    { 
        return new DoubleMeta(-std::numeric_limits<double>::max(), std::numeric_limits<double>::max() ); 
    }

    //---------------------------------------------------------------------------------
    void DoubleMeta::setMin(double val)
    { 
        m_minVal = val; 
        m_maxVal = std::max(m_maxVal,m_minVal); 
    }
    
    //---------------------------------------------------------------------------------
    void DoubleMeta::setMax(double val)
    { 
        m_maxVal = val; 
        m_minVal = std::min(m_maxVal,m_minVal); 
    }

    //---------------------------------------------------------------------------------
    void DoubleMeta::setStepSize(double val)
    { 
#if _DEBUG
        if (val < 0.0)
        {
            throw std::logic_error("stepSize of IntMeta must be >= 0.0");
        }
#endif
        m_stepSize = val; 
    }

    //---------------------------------------------------------------------------------
    StringMeta::StringMeta(tType type) : ParamMeta(rttiStringMeta), m_stringType(type), m_len(0), m_val(NULL) 
    {
    }

    //---------------------------------------------------------------------------------
    StringMeta::StringMeta(tType type, const char* val) : ParamMeta(rttiStringMeta), m_stringType(type), m_len(1)
    {
        if(val)
        {
            m_val = (char**) calloc(1, sizeof(char*));
            m_val[0] = _strdup(val);
        }
        else
        {
            m_len = 0;
            m_val = NULL;
        }
    }

    //---------------------------------------------------------------------------------
    StringMeta::StringMeta(const StringMeta& cpy) : ParamMeta(rttiStringMeta), m_stringType(cpy.m_stringType), m_len(cpy.m_len), m_val(NULL)
    {
        if(m_len > 0)
        {
            m_val = (char**) calloc(m_len, sizeof(char*));
            for(int i=0;i<m_len;++i) m_val[i] = _strdup(cpy.m_val[i]);
        }
    }

    //---------------------------------------------------------------------------------
    /*virtual*/ StringMeta::~StringMeta()
    {
        for(int i=0;i<m_len;++i) free(m_val[i]);
        free(m_val);
    }

    //---------------------------------------------------------------------------------
    bool StringMeta::addItem(const char *val)
    {
        if(m_val)
        {
            char **m_val_old = m_val;
            m_val = (char**)realloc(m_val, sizeof(char*) * (++m_len) ); //m_val can change its address. if NULL, reallocation failed and m_val_old still contains old values
            if (!m_val)
            {
                m_val = m_val_old;
                m_len--; //failed to add new value
                return false;
            }
        }
        else
        {
            m_val = (char**) calloc(++m_len, sizeof(char*));
        }
        m_val[m_len-1] = _strdup(val);
        return true;
    }

    //---------------------------------------------------------------------------------
    StringMeta & StringMeta::operator += (const char *val)
    {
        addItem(val);
        return *this;
    }

    //---------------------------------------------------------------------------------
    const char* StringMeta::getString(int idx /*= 0*/) const 
    { 
        return (idx >= m_len) ? NULL : m_val[idx]; 
    }


    //---------------------------------------------------------------------------------
    CharArrayMeta::CharArrayMeta(char minVal, char maxVal, char stepSize /*= 1*/) :
        CharMeta(minVal, maxVal, stepSize),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiCharArrayMeta;
    }

    //---------------------------------------------------------------------------------
    CharArrayMeta::CharArrayMeta(char minVal, char maxVal, char stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/) :
        CharMeta(minVal, maxVal, stepSize),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
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
    IntArrayMeta::IntArrayMeta(int minVal, int maxVal, int stepSize /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiIntArrayMeta;
    }

    //---------------------------------------------------------------------------------
    IntArrayMeta::IntArrayMeta(int minVal, int maxVal, int stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
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
    DoubleArrayMeta::DoubleArrayMeta(double minVal, double maxVal, double stepSize /*= 1*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiDoubleArrayMeta;
    }

    //---------------------------------------------------------------------------------
    DoubleArrayMeta::DoubleArrayMeta(double minVal, double maxVal, double stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_numMin(numMin),
        m_numMax(numMax),
        m_numStep(numStepSize)
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
    RangeMeta::RangeMeta(int minVal, int maxVal, int stepSize /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_rangeMin(0),
        m_rangeMax(std::numeric_limits<int>::max()),
        m_rangeStep(1)
    {
        m_type = rttiRangeMeta;
    }

    //---------------------------------------------------------------------------------
    RangeMeta::RangeMeta(int minVal, int maxVal, int stepSize, int rangeMin, int rangeMax, int rangeStep /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_rangeMin(rangeMin),
        m_rangeMax(rangeMax),
        m_rangeStep(rangeStep)
    {
        m_type = rttiRangeMeta;
    }

    //---------------------------------------------------------------------------------
    void RangeMeta::setRangeMin(int val)
    {
        m_rangeMin = val; 
        m_rangeMax = std::max(m_rangeMin, m_rangeMax); 
    }
        
    //---------------------------------------------------------------------------------
    void RangeMeta::setRangeMax(int val)
    {
        m_rangeMax = val; 
        m_rangeMin = std::min(m_rangeMin, m_rangeMax);
    }

    //---------------------------------------------------------------------------------
    void RangeMeta::setRangeStep(int val)
    {
        m_rangeStep = val;
    }


    //---------------------------------------------------------------------------------
    IntervalMeta::IntervalMeta(int minVal, int maxVal, int stepSize /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_ivalMin(0),
        m_ivalMax(std::numeric_limits<int>::max()),
        m_ivalStep(1)
    {
        m_type = rttiIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    IntervalMeta::IntervalMeta(int minVal, int maxVal, int stepSize, int intervalMin, int intervalMax, int intervalStep /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_ivalMin(intervalMin),
        m_ivalMax(intervalMax),
        m_ivalStep(intervalStep)
    {
        m_type = rttiIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    void IntervalMeta::setIntervalMin(int val)
    {
        m_ivalMin = val; 
        m_ivalMax = std::max(m_ivalMin, m_ivalMax); 
    }
        
    //---------------------------------------------------------------------------------
    void IntervalMeta::setIntervalMax(int val)
    {
        m_ivalMax = val; 
        m_ivalMin = std::min(m_ivalMin, m_ivalMax);
    }

    //---------------------------------------------------------------------------------
    void IntervalMeta::setIntervalStep(int val)
    {
        m_ivalStep = val;
    }


    //---------------------------------------------------------------------------------
    DoubleIntervalMeta::DoubleIntervalMeta(double minVal, double maxVal, double stepSize /*= 0.0*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_ivalMin(0.0),
        m_ivalMax(std::numeric_limits<double>::max()),
        m_ivalStep(0.0)
    {
        m_type = rttiDoubleIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    DoubleIntervalMeta::DoubleIntervalMeta(double minVal, double maxVal, double stepSize, double intervalMin, double intervalMax, double intervalStep /*= 0.0*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_ivalMin(intervalMin),
        m_ivalMax(intervalMax),
        m_ivalStep(intervalStep)
    {
        m_type = rttiDoubleIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    void DoubleIntervalMeta::setIntervalMin(double val)
    {
        m_ivalMin = val; 
        m_ivalMax = std::max(m_ivalMin, m_ivalMax); 
    }
        
    //---------------------------------------------------------------------------------
    void DoubleIntervalMeta::setIntervalMax(double val)
    {
        m_ivalMax = val; 
        m_ivalMin = std::min(m_ivalMin, m_ivalMax);
    }

    //---------------------------------------------------------------------------------
    void DoubleIntervalMeta::setIntervalStep(double val)
    {
        m_ivalStep = val;
    }

    //---------------------------------------------------------------------------------
    RectMeta::RectMeta(const ito::RangeMeta &widthMeta, const ito::RangeMeta &heightMeta) :
        ParamMeta(rttiRectMeta),
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

