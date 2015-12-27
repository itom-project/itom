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

#include "../interval.h"

#include <limits>
#include <algorithm>

namespace ito {


//----------------------------------------------------------------------------------------------------------------------------------
AutoInterval::AutoInterval()
    : m_min(-std::numeric_limits<float>::infinity()),
    m_max(std::numeric_limits<float>::infinity()),
    m_auto(true)
{
    if (m_min > m_max)
    {
        std::swap(m_min, m_max);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
AutoInterval::AutoInterval(float min, float max, bool autoInterval)
    : m_min(min),
    m_max(max),
    m_auto(autoInterval)
{
    if (m_min > m_max)
    {
        std::swap(m_min, m_max);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
AutoInterval::~AutoInterval()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void AutoInterval::setRange(float min, float max)
{
    m_min = min;
    m_max = max;

    if (m_min > m_max)
    {
        std::swap(m_min, m_max);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AutoInterval::setMinimum(float min)
{
    m_min = min;

    if (m_min > m_max)
    {
        m_max = m_min;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AutoInterval::setMaximum(float max)
{
    m_max = max;

    if (m_max < m_min)
    {
        m_min = m_max;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void AutoInterval::setAuto(bool autoInterval)
{
    m_auto = autoInterval;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AutoInterval::operator==( const AutoInterval &other ) const
{
    if (m_auto != other.m_auto)
    {
        return false;
    }
    
    return (m_auto || (m_min == other.m_min && m_max == other.m_max));
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AutoInterval::operator!=( const AutoInterval &other ) const
{
    return ( !( *this == other ) );
}



} //end namespace ito