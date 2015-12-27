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

#ifndef INTERVAL_H
#define INTERVAL_H

#ifdef __APPLE__
extern "C++" {
#endif

/* includes */
#include "commonGlobal.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
/** @class AutoInterval
*   @brief  class for a interval type containing a min-max-range and an auto-flag.
*
*   This class can be used as datatype if you want to provide a range- or interval-object that
*   contains of two float min and max boundaries as well as an auto-flag. If auto is set, the
*   min and max boundaries can be calculated by an automatic mode within your code.
*/
class ITOMCOMMON_EXPORT AutoInterval
{       
    private:
        float m_min;
        float m_max;
        bool m_auto;

    public:
        AutoInterval(); //!> default constructor: auto-mode, min=-Inf, max=+Inf
        AutoInterval(float min, float max, bool autoInterval = false);
        virtual ~AutoInterval();

        inline float minimum() const { return m_min; }
        inline float maximum() const { return m_max; }
        inline float & rmin() { return m_min; }
        inline float & rmax() { return m_max; }

        inline bool isAuto() const { return m_auto; }
        inline bool &rauto() { return m_auto; }

        void setRange(float min, float max);
        void setMinimum(float min);
        void setMaximum(float max);
        void setAuto(bool autoInterval);

        bool operator==( const AutoInterval & ) const;
        bool operator!=( const AutoInterval & ) const;
};


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
