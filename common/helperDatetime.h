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

#ifdef __APPLE__
extern "C++"
{
#endif

/* includes */
#include "commonGlobal.h"
#include "typeDefs.h"

#include <qdatetime.h>


namespace ito {
namespace datetime {
    void ITOMCOMMONQT_EXPORT toYMDHMSU(const DateTime &dt, int &year, int &month, int &day, int &hour, int &minute, int &second, int &usecond);
    DateTime ITOMCOMMONQT_EXPORT fromYMDHMSU(int year, int month, int day, int hour, int minute, int second, int usecond, int utcoffset);

    //!< converts a DateTime object to QDateTime (microseconds are ignored)
    QDateTime ITOMCOMMONQT_EXPORT toQDateTime(const DateTime &dt);

    //!< converts a QDateTime to DateTime
    DateTime ITOMCOMMONQT_EXPORT toDateTime(const QDateTime &dt);
} // end namespace datetime

namespace timedelta {
    //!< convert the time delta structure into days, seconds and microseconds.
    void ITOMCOMMONQT_EXPORT toDSU(const TimeDelta &td, int &days, int &seconds, int &useconds);
    TimeDelta ITOMCOMMONQT_EXPORT fromDSU(int days, int seconds, int useconds);

} // end namespace timedelta
} // end namespace ito

#ifdef __APPLE__
}
#endif
