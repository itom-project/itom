/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "pythonDateTime.h"

#include <qdatetime.h>

namespace ito {
//---------------------------------------------------------------------------------
// checks for Python datetime and corresponding numpy types
bool PythonDateTime::PyDateTime_CheckExt(PyObject* obj)
{
    // import the datetime api
    Itom_PyDateTime_IMPORT;

    return PyDateTime_Check(obj);
}

//---------------------------------------------------------------------------------
// checks for Python time delta and corresponding numpy types
bool PythonDateTime::PyTimeDelta_CheckExt(PyObject* obj)
{
    // import the datetime api
    Itom_PyDateTime_IMPORT;

    return PyDelta_Check(obj);
}

//---------------------------------------------------------------------------------
DateTime PythonDateTime::GetDateTime(PyObject* obj, bool& ok)
{
    // import the datetime api
    Itom_PyDateTime_IMPORT;

    ok = true;

    if (PyDateTime_Check(obj))
    {
        PyDateTime_DateTime* dt = (PyDateTime_DateTime*)(obj);

        int year = PyDateTime_GET_YEAR(dt);
        int month = PyDateTime_GET_MONTH(dt);
        int day = PyDateTime_GET_DAY(dt);
        int hour = PyDateTime_DATE_GET_HOUR(dt);
        int minute = PyDateTime_DATE_GET_MINUTE(dt);
        int second = PyDateTime_DATE_GET_SECOND(dt);
        int usecond = PyDateTime_DATE_GET_MICROSECOND(dt);
        int utcoffset = 0;

        if (dt->hastzinfo)
        {
            PyDateTime_TZInfo* tz = (PyDateTime_TZInfo*)(dt->tzinfo);

            PyObject* name = PyUnicode_FromString("utcoffset");
            PyObject* ret = PyObject_CallMethodObjArgs(dt->tzinfo, name, Py_None, NULL);
            Py_DECREF(name);

            if (ret && PyDelta_Check(ret))
            {
                utcoffset = PyDateTime_DELTA_GET_SECONDS(ret);
            }
            else
            {
                ok = false;
            }

            Py_XDECREF(ret);
        }

        return ito::datetime::fromYMDHMSU(
            year, month, day, hour, minute, second, usecond, utcoffset);
    }
    else
    {
        ok = false;
        PyErr_Format(PyExc_RuntimeError, "object cannot be converted to a datetime object.");
    }

    return ito::DateTime(0);
}

//---------------------------------------------------------------------------------
TimeDelta PythonDateTime::GetTimeDelta(PyObject* obj, bool& ok)
{
    // import the datetime api
    Itom_PyDateTime_IMPORT;

    if (PyDelta_Check(obj))
    {
        int days = PyDateTime_DELTA_GET_DAYS(obj);
        int seconds = PyDateTime_DELTA_GET_SECONDS(obj);
        int useconds = PyDateTime_DELTA_GET_MICROSECONDS(obj);

        return ito::timedelta::fromDSU(days, seconds, useconds);
    }
    else
    {
        ok = false;
        PyErr_Format(PyExc_RuntimeError, "object cannot be converted to a timedelta object.");
    }

    return ito::TimeDelta(0);
}

//---------------------------------------------------------------------------------
// new ref, sets a PyException if an error occurs
PyObject* PythonDateTime::GetPyDateTime(const DateTime& datetime)
{
    Itom_PyDateTime_IMPORT;
    int year, month, day, hour, minute, sec, usec;

    ito::datetime::toYMDHMSU(datetime, year, month, day, hour, minute, sec, usec);

    PyObject* d = PyDateTime_FromDateAndTime(year, month, day, hour, minute, sec, usec);

    if (datetime.utcOffset != 0)
    {
        PyDateTime_DateTime* dt = (PyDateTime_DateTime*)(d);
        auto delta = PyDelta_FromDSU(0, datetime.utcOffset, 0); // new ref
        PyObject* oldTzInfo = dt->hastzinfo ? dt->tzinfo : nullptr;
        dt->tzinfo = PyTimeZone_FromOffset(delta); // new ref, passed to tzinfo
        dt->hastzinfo = true;
        Py_DECREF(delta);
        Py_XDECREF(oldTzInfo);
    }

    return d;
}

//---------------------------------------------------------------------------------
// new ref, sets a PyException if an error occurs
PyObject* PythonDateTime::GetPyTimeDelta(const TimeDelta& delta)
{
    Itom_PyDateTime_IMPORT;
    int days, secs, usecs;
    ito::timedelta::toDSU(delta, days, secs, usecs);
    PyObject* d = PyDelta_FromDSU(days, secs, usecs);
    return d;
}

//-------------------------------------------------------------------------------------
bool PythonDateTime::ItoDatetime2npyDatetime(
    const ito::DateTime& src, npy_datetime& dest, const PyArray_DatetimeMetaData& meta)
{
    /* Cannot instantiate a datetime with generic units */
    if (meta.base == NPY_FR_GENERIC)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Cannot convert a dateTime value into a dateTime value with generic units.");
        return false;
    }

    if (src.utcOffset != 0)
    {
        if (PyErr_WarnEx(
                PyExc_RuntimeWarning,
                "UTC offset is ignored when converted to a numpy.ndarray",
                1) == -1)
        {
            return false; // warning was turned into a real exception,
        }
    }

    if (meta.base == NPY_FR_Y)
    {
        /* Truncate to the year */
        int year, month, day, hour, minute, second, usecond;
        ito::datetime::toYMDHMSU(src, year, month, day, hour, minute, second, usecond);
        dest = year - 1970;
    }
    else if (meta.base == NPY_FR_M)
    {
        /* Truncate to the month */
        int year, month, day, hour, minute, second, usecond;
        ito::datetime::toYMDHMSU(src, year, month, day, hour, minute, second, usecond);
        dest = 12 * (year - 1970) + (month - 1);
    }
    else if (meta.base == NPY_FR_us)
    {
        dest = src.datetime;
    }
    else if (meta.base == NPY_FR_ns)
    {
        dest = src.datetime * 1000LL;
    }
    else if (meta.base == NPY_FR_ps)
    {
        dest = src.datetime * 1000000LL;
    }
    else if (meta.base == NPY_FR_fs)
    {
        dest = src.datetime * 1000000LL * 1000LL;
    }
    else if (meta.base == NPY_FR_as)
    {
        dest = src.datetime * 1000000LL * 1000000LL;
    }
    else
    {
        // usecond contains milliseconds and seconds
        int64 usecond = static_cast<int>(src.datetime % 1000000);

        // milliseconds since 01.01.1970, 00:00
        int64 secs = static_cast<time_t>((src.datetime - usecond) / 1000000);

        while (usecond < 0)
        {
            secs--;
            usecond += 1000000LL;
        }

        const QDateTime epochDate(QDate(1970, 1, 1));
        QDateTime date = epochDate.addSecs(secs);

        if (date.isDaylightTime())
        {
            // numpy does not consider daylight saving time
            date = date.addSecs(-3600);
        }

        QTime time = date.time();
        auto secsSinceEpoch = epochDate.secsTo(date);

        /* Otherwise calculate the number of days to start from 1.1.1970 */
        auto daysSinceEpoch = secsSinceEpoch / (3600LL * 24LL);

        switch (meta.base)
        {
        case NPY_FR_W:
            /* Truncate to weeks */
            if (daysSinceEpoch >= 0)
            {
                dest = daysSinceEpoch / 7;
            }
            else
            {
                dest = (daysSinceEpoch - 6) / 7;
            }
            break;
        case NPY_FR_D:
            dest = daysSinceEpoch;
            break;
        case NPY_FR_h:
            dest = daysSinceEpoch * 24 + time.hour();
            break;
        case NPY_FR_m:
            dest = (daysSinceEpoch * 24 + time.hour()) * 60 + time.minute();
            break;
        case NPY_FR_s:
            dest = ((daysSinceEpoch * 24 + time.hour()) * 60 + time.minute()) * 60 + time.second();
            break;
        case NPY_FR_ms:
            dest =
                (((daysSinceEpoch * 24 + time.hour()) * 60 + time.minute()) * 60 + time.second()) *
                    1000 +
                usecond / 1000;
            break;
        default:
            /* Something got corrupted */
            PyErr_SetString(PyExc_ValueError, "NumPy datetime metadata with corrupt unit value");
            return false;
        }
    }

    /* Divide by the multiplier */
    if (meta.num > 1)
    {
        if (dest >= 0)
        {
            dest /= meta.num;
        }
        else
        {
            dest = (dest - meta.num + 1) / meta.num;
        }
    }

    return true;
}

 //------------------------------------------------------------------------------------
ito::int64 LLDivide(const long long &nom, const long long &den)
{
    if (nom >= 0 || nom % den == 0)
    {
        return nom / den;
    }
    else
    {
        return nom / den - 1;
    }
}

//-------------------------------------------------------------------------------------
bool PythonDateTime::ItoTimedelta2npyTimedleta(
    const ito::TimeDelta& src, npy_timedelta& dest, const PyArray_DatetimeMetaData& meta)
{
    /* Cannot instantiate a datetime with generic units */
    if (meta.base == NPY_FR_GENERIC)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Cannot convert a timeDelta value into a timeDelta value with generic units.");
        return false;
    }

    switch (meta.base)
    {
    case NPY_FR_Y:
    case NPY_FR_M:
        PyErr_SetString(
            PyExc_ValueError, "A timedelta value in ms cannot be converted into months or years.");
        return false;
        break;
    case NPY_FR_W:
        dest = LLDivide(src.delta, (1000000LL * 3600LL * 24LL * 7LL));
        break;
    case NPY_FR_D:
        dest = LLDivide(src.delta, (1000000LL * 3600LL * 24LL));
        break;
    case NPY_FR_h:
        dest = LLDivide(src.delta, (1000000LL * 3600LL));
        break;
    case NPY_FR_m:
        dest = LLDivide(src.delta, (1000000LL * 60LL));
        break;
    case NPY_FR_s:
        dest = LLDivide(src.delta, 1000000LL);
        break;
    case NPY_FR_ms:
        dest = LLDivide(src.delta, 1000LL);
        break;
    case NPY_FR_us:
        dest = src.delta;
        break;
    case NPY_FR_ns:
        dest = src.delta * 1000LL;
        break;
    case NPY_FR_ps:
        dest = src.delta * 1000000LL;
        break;
    case NPY_FR_fs:
        dest = src.delta * 1000000LL * 1000LL;
        break;
    case NPY_FR_as:
        dest = src.delta * 1000000LL * 1000000LL;
        break;
    default:
        /* Something got corrupted */
        PyErr_SetString(PyExc_ValueError, "NumPy timedelta metadata with corrupt unit value");
        return false;
    }

    /* Divide by the multiplier */
    if (meta.num > 1)
    {
        if (dest >= 0)
        {
            dest /= meta.num;
        }
        else
        {
            dest = (dest - meta.num + 1) / meta.num;
        }
    }

    return true;
}


}; // end namespace ito
