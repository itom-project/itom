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
    IntMeta::IntMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/)
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
    DoubleMeta::DoubleMeta(float64 minVal, float64 maxVal, float64 stepSize /*=0.0*/ /*0.0 means no specific step size*/)
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
    IntArrayMeta::IntArrayMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiIntArrayMeta;
    }

    //---------------------------------------------------------------------------------
    IntArrayMeta::IntArrayMeta(int32 minVal, int32 maxVal, int32 stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/) :
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
    DoubleArrayMeta::DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize /*= 1*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_numMin(0),
        m_numMax(std::numeric_limits<size_t>::max()),
        m_numStep(1)
    {
        m_type = rttiDoubleArrayMeta;
    }

    //---------------------------------------------------------------------------------
    DoubleArrayMeta::DoubleArrayMeta(float64 minVal, float64 maxVal, float64 stepSize, size_t numMin, size_t numMax, size_t numStepSize /*= 1*/) :
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
    IntervalMeta::IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_sizeMin(0),
        m_sizeMax(std::numeric_limits<int32>::max()),
        m_sizeStep(1),
        m_isIntervalNotRange(false)
    {
        m_type = rttiIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    IntervalMeta::IntervalMeta(int32 minVal, int32 maxVal, int32 stepSize, int32 intervalMin, int32 intervalMax, int32 intervalStep /*= 1*/) :
        IntMeta(minVal, maxVal, stepSize),
        m_sizeMin(intervalMin),
        m_sizeMax(intervalMax),
        m_sizeStep(intervalStep),
        m_isIntervalNotRange(false)
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
    RangeMeta::RangeMeta(int32 minVal, int32 maxVal, int32 stepSize /*= 1*/) :
        IntervalMeta(minVal, maxVal, stepSize)
    {
        m_type = rttiRangeMeta;
        m_isIntervalNotRange = false;
    }

    //---------------------------------------------------------------------------------
    RangeMeta::RangeMeta(int32 minVal, int32 maxVal, int32 stepSize, size_t sizeMin, size_t sizeMax, size_t sizeStep /*= 1*/) :
        IntervalMeta(minVal, maxVal, stepSize, sizeMin, sizeMax, sizeStep)
    {
        m_type = rttiRangeMeta;
        m_isIntervalNotRange = false;
    }


    //---------------------------------------------------------------------------------
    DoubleIntervalMeta::DoubleIntervalMeta(float64 minVal, float64 maxVal, float64 stepSize /*= 0.0*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_sizeMin(0.0),
        m_sizeMax(std::numeric_limits<float64>::max()),
        m_sizeStep(0.0)
    {
        m_type = rttiDoubleIntervalMeta;
    }

    //---------------------------------------------------------------------------------
    DoubleIntervalMeta::DoubleIntervalMeta(float64 minVal, float64 maxVal, float64 stepSize, float64 sizeMin, float64 sizeMax, float64 sizeStep /*= 0.0*/) :
        DoubleMeta(minVal, maxVal, stepSize),
        m_sizeMin(sizeMin),
        m_sizeMax(sizeMax),
        m_sizeStep(sizeStep)
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

