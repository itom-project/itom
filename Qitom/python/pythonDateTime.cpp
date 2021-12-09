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

namespace ito
{
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
    DateTime PythonDateTime::GetDateTime(PyObject* obj, bool &ok)
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

                PyObject *name = PyUnicode_FromString("utcoffset");
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

            return ito::datetime::fromYMDHMSU(year, month, day, hour, minute, second, usecond, utcoffset);
        }
        else
        {
            ok = false;
            PyErr_Format(PyExc_RuntimeError, "object cannot be converted to a datetime object.");
        }

        return ito::DateTime(0);
    }

    //---------------------------------------------------------------------------------
    TimeDelta PythonDateTime::GetTimeDelta(PyObject* obj, bool &ok)
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
    PyObject* PythonDateTime::GetPyDateTime(const DateTime &datetime)
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
    PyObject* PythonDateTime::GetPyTimeDelta(const TimeDelta &delta)
    {
        Itom_PyDateTime_IMPORT;
        int days, secs, usecs;
        ito::timedelta::toDSU(delta, days, secs, usecs);
        PyObject* d = PyDelta_FromDSU(days, secs, usecs);
        return d;
    }


}; //end namespace ito
