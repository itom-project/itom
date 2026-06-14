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

#include "../byteArray.h"

namespace ito {

/*static*/ char ByteArray::emptyChar = '\0';

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
ByteArray::ByteArray(const char *str) : d(NULL)
{
    if (str)
    {
        size_t len = strlen(str);
        d = static_cast<Data*>(malloc(sizeof(Data)+len));
        d->m_ref = 0;
        d->m_pData = d->m_buffer;
        memcpy(d->m_buffer, str, len+1);
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
ByteArray &ByteArray::operator=(const ByteArray &rhs)
{
    if (rhs.d)
    {
        BYTEARRAY_INCCREF(rhs.d);
    }

    decAndFree(d);
    d = rhs.d;
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
ByteArray &ByteArray::operator=(ByteArray &&rhs)
{
    std::swap(d, rhs.d);
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
ByteArray &ByteArray::operator=(const char *str)
{
    decAndFree(d);

    if (str)
    {
        size_t len = strlen(str);
        d = static_cast<Data*>(malloc(sizeof(Data)+len));
        d->m_ref = 0;
        d->m_pData = d->m_buffer;
        memcpy(d->m_buffer, str, len+1);
    }
    else
    {
        d = NULL;
    }

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
ByteArray &ByteArray::append(const char *str)
{
    if (str)
    {
        size_t newlen = strlen(str);

        if (newlen > 0)
        {
            if (d)
            {
                size_t oldlen = strlen(d->m_pData);

                if (d->m_ref == 0) //this is the only user
                {
                    d = static_cast<Data*>(realloc(d, sizeof(Data) + newlen + oldlen));
                    d->m_pData = d->m_buffer;
                    memcpy(d->m_buffer + oldlen * sizeof(char), str, newlen+1);
                }
                else
                {
                    Data *oldData = d;
                    d = static_cast<Data*>(malloc(sizeof(Data) + newlen + oldlen));
                    d->m_ref = 0;
                    d->m_pData = d->m_buffer;
                    memcpy(d->m_buffer, oldData->m_pData, oldlen);
                    memcpy(d->m_buffer + oldlen * sizeof(char), str, newlen+1);

                    //decref the old data structure
                    decAndFree(oldData);
                }
            }
            else
            {
                d = static_cast<Data*>(malloc(sizeof(Data) + newlen));
                d->m_ref = 0;
                d->m_pData = d->m_buffer;
                memcpy(d->m_buffer, str, newlen+1);
            }
        }
    }

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
ByteArray &ByteArray::append(const ByteArray &str)
{
    int newlen = str.length();
    Data *oldData = d;
    size_t oldlen = strlen(d->m_pData);

    if (newlen > 0)
    {
        d = static_cast<Data*>(malloc(sizeof(Data) + newlen + oldlen));
        d->m_ref = 0;
        d->m_pData = d->m_buffer;
        memcpy(d->m_buffer, oldData->m_pData, oldlen);
        memcpy(d->m_buffer + oldlen * sizeof(char), str.d->m_pData, newlen+1);

        //decref the old data structure
        if (oldData)
        {
            decAndFree(oldData);
        }
    }

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
//doc in header file
bool ByteArray::operator==(const ByteArray &a) const
{
    if (d && a.d)
    {
        return (strcmp(d->m_pData,a.d->m_pData)==0);
    }
    else if (!d && !a.d)
    {
        return true;
    }
    return false;
}


} //end namespace ito
