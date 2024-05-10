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

#pragma once

/* includes */
#ifndef Q_MOC_RUN
    // see numpy help ::array api :: Miscellaneous :: Importing the api (this line
    // must before include global.h)
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API
    #define NO_IMPORT_ARRAY

    #include "python/pythonWrapper.h"
#endif

#include "DataObject/dataobj.h"
#include "common/helperDatetime.h"
#include "common/typeDefs.h"

namespace ito {

class PythonDateTime
{
public:
    // checks for Python datetime and corresponding numpy types
    static bool PyDateTime_CheckExt(PyObject* obj);

    // checks for Python time delta and corresponding numpy types
    static bool PyTimeDelta_CheckExt(PyObject* obj);

    static DateTime GetDateTime(PyObject* obj, bool& ok);
    static TimeDelta GetTimeDelta(PyObject* obj, bool& ok);

    // new ref, sets a PyException if an error occurs
    static PyObject* GetPyDateTime(const DateTime& datetime);

    // new ref, sets a PyException if an error occurs
    static PyObject* GetPyTimeDelta(const TimeDelta& delta);

    static bool ItoDatetime2npyDatetime(
        const ito::DateTime& src, npy_datetime& dest, const PyArray_DatetimeMetaData& meta);

    static bool NpyDatetime2itoDatetime(
        const npy_datetime& dt, const PyArray_DatetimeMetaData &md, ito::DateTime& out);

    static bool ItoTimedelta2npyTimedleta(
        const ito::TimeDelta& src, npy_timedelta& dest, const PyArray_DatetimeMetaData& meta);

    static bool NpyTimedelta2itoTimedelta(
        const npy_timedelta& dt, const PyArray_DatetimeMetaData &md, ito::TimeDelta& out);

    template <typename _Tp, size_t timeMemberOffset>
    static void GuessDateTimeMetaFromDataObjectValues(
        const ito::DataObject* dobj, PyArray_DatetimeMetaData& descr_meta)
    {
        ito::int64 combined = 0LL;
        int dims = dobj->getDims();
        const cv::Mat* plane;
        const _Tp* rowPtr;
        const uchar* item;

        if (dims > 0)
        {
            for (int p = 0; p < dobj->getNumPlanes(); ++p)
            {
                plane = dobj->getCvPlaneMat(p);

                for (int r = 0; r < plane->rows; ++r)
                {
                    rowPtr = plane->ptr<_Tp>(r);

                    for (int c = 0; c < plane->cols; ++c)
                    {
                        item = reinterpret_cast<const uchar*>(&(rowPtr[c]));

                        combined |= *reinterpret_cast<const ito::int64*>(item + timeMemberOffset);
                    }
                }
            }
        }

        // set defaults
        descr_meta.num = 1;
        descr_meta.base = NPY_FR_us;

        if (combined % 1000LL == 0)
        {
            // no microseconds --> milli
            if (combined % 1000000LL == 0)
            {
                // no milliseconds
                if (combined % 1000000000LL == 0)
                {
                    // no seconds
                    if (combined % (60LL * 1000000000LL) == 0)
                    {
                        // no minutes
                        if (combined % (3600LL * 1000000000LL) == 0)
                        {
                            // no hours
                            descr_meta.base = NPY_FR_D;
                        }
                        else
                        {
                            descr_meta.base = NPY_FR_h;
                        }
                    }
                    else
                    {
                        descr_meta.base = NPY_FR_m;
                    }
                }
                else
                {
                    descr_meta.base = NPY_FR_s;
                }
            }
            else
            {
                descr_meta.base = NPY_FR_ms;
            }
        }
        else
        {
            descr_meta.base = NPY_FR_us;
        }
    }
};

}; // end namespace ito
