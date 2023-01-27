/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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

#include "../helperDatetime.h"

#include <qdatetime.h>
#include <qtimezone.h>

namespace ito {
namespace datetime {
void toYMDHMSU(
    const DateTime& dt,
    int& year,
    int& month,
    int& day,
    int& hour,
    int& minute,
    int& second,
    int& usecond)
{
    // usecond contains milliseconds and seconds
    usecond = static_cast<int>(dt.datetime % 1000000);

    // milliseconds since 01.01.1970, 00:00
    int64 secs = static_cast<time_t>((dt.datetime - usecond) / 1000000);

    if (dt.datetime < 0 && usecond != 0)
    {
        usecond = 1000000 + usecond;
        secs -= 1;
    }

    QDateTime qdt(QDate(1970, 1, 1), QTime(0, 0, 0));
    qdt.setOffsetFromUtc(dt.utcOffset);
    qdt = qdt.addSecs(secs);

    second = qdt.time().second();
    minute = qdt.time().minute();
    hour = qdt.time().hour();
    day = qdt.date().day();
    month = qdt.date().month();
    year = qdt.date().year();
}

DateTime fromYMDHMSU(
    int year, int month, int day, int hour, int minute, int second, int usecond, int utcoffset)
{
    QDate date(year, 1, 1);
    date = date.addMonths(month - 1);
    date = date.addDays(day - 1);
    QDateTime qdt(date, QTime(hour, minute, second), QTimeZone(0));
    DateTime dt;
    dt.datetime = qdt.toMSecsSinceEpoch() * 1000 + usecond;
    dt.utcOffset = utcoffset;
    return dt;
}

QDateTime toQDateTime(const DateTime& dt)
{
    // usecond contains milliseconds and seconds
    int usecond = static_cast<int>(dt.datetime % 1000000);

    // milliseconds since 01.01.1970, 00:00
    int64 secs = static_cast<time_t>((dt.datetime - usecond) / 1000000);

    if (dt.datetime < 0 && usecond != 0)
    {
        usecond = 1000000 + usecond;
        secs -= 1;
    }

    QDateTime qdt(QDate(1970, 1, 1), QTime(0, 0, 0));
    qdt.setOffsetFromUtc(dt.utcOffset);
    qdt = qdt.addSecs(secs);
    return qdt.addMSecs(usecond / 1000.0);
}

DateTime toDateTime(const QDateTime& dt)
{
    QDate date = dt.date();
    QTime time = dt.time();
    return fromYMDHMSU(
        date.year(),
        date.month(),
        date.day(),
        time.hour(),
        time.minute(),
        time.second(),
        time.msec() * 1000,
        dt.offsetFromUtc());
}

} // end namespace datetime

namespace timedelta {

//!< convert the time delta structure into days, seconds and microseconds.
void toDSU(const TimeDelta& td, int& days, int& seconds, int& useconds)
{
    useconds = static_cast<int>(td.delta % 1000000);
    int64 sec = (td.delta - useconds) / 1000000;

    const int64 secPerDay = 3600 * 24;
    days = static_cast<int>(sec / secPerDay);
    seconds = static_cast<int>(sec % secPerDay);
}

TimeDelta fromDSU(int days, int seconds, int useconds)
{
    int64 ms = useconds;
    ms += (int64)seconds * 1000000;
    const int64 secPerDay = 3600 * 24;
    ms += secPerDay * days * 1000000;
    return TimeDelta(ms);
}

} // end namespace timedelta
} // end namespace ito
