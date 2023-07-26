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
        double m_min; /*!< minimum value that is included in the interval */
        double m_max; /*!< maximum value that is included in the interval */
        bool m_auto; /*!< true if the interval can be automatically be adjusted by the code using it or if m_min and m_max are fixed boundaries */

    public:
        AutoInterval(); //!< default constructor: auto-mode: true, min=-Inf, max=+Inf

        //! constructor
        /*!
        \param min is the included minimum value
        \param max is the included maximum value
        \param autoInterval is the state of auto-flag.
        */
        AutoInterval(double min, double max, bool autoInterval = false);

        virtual ~AutoInterval(); //!< destructor */

        //! return the minimum value of the interval (included)
        inline double minimum() const { return m_min; }

        //! return the maximum value of the interval (included)
        inline double maximum() const { return m_max; }

        //! return a reference to the minimum value of the interval. Assigning a float to this reference will change the minimum in the interval.
        inline double & rmin() { return m_min; }

        //! return a reference to the maximum value of the interval. Assigning a float to this reference will change the maximum in the interval.
        inline double & rmax() { return m_max; }

        //! return the state of the auto-flag as boolean variable
        inline bool isAuto() const { return m_auto; }

        //! return the reference to the auto-flag. Assigning a boolean to this reference will change the auto-flag.
        inline bool &rauto() { return m_auto; }

        //! set the boundary values of the AutoInterval without changing the auto flag
        /*!
        \param min is the new included minimum value
        \param max is the new included maximum value
        */
        void setRange(double min, double max);

        //! change the included minimum value
        /*!
        \param min is the new included minimum value
        */
        void setMinimum(double min);

        //! change the included maximum value
        /*!
        \param max is the new included maximum value
        */
        void setMaximum(double max);

        //! set the auto-flag to a given boolean value
        /*!
        \param autoInterval is the new state of the auto-flag
        */
        void setAuto(bool autoInterval);

        //! comparison operator between the AutoInterval and another AutoInterval.
        /*!
        \param rhs is the AutoInterval that is compared to this one.
        \return true if both AutoInterval instances have the auto-flag set or if both boundaries are equal.
        */
        bool operator==( const AutoInterval & ) const;

        //! comparison operator between the AutoInterval and another AutoInterval.
        /*!
        \param rhs is the AutoInterval that is compared to this one.
        \return false if both AutoInterval instances have the auto-flag set or if both boundaries are equal.
        */
        bool operator!=( const AutoInterval & ) const;
};


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
