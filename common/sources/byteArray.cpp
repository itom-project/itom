/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut f�r Technische
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

//----------------------------------------------------------------------------------------------------------------------------------
ByteArray::ByteArray(const char *str) : d(NULL)
{
    if (str)
    {
        int len = strlen(str);
        d = static_cast<Data*>(malloc(sizeof(Data)+len));
        d->m_ref = 0;
        d->m_pData = d->m_buffer;
        memcpy(d->m_buffer, str, len+1);
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
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
void ByteArray::append(const char *str)
{
    if (str)
    {
        int newlen = strlen(str);
        Data *oldData = d;
        int oldlen = strlen(d->m_pData);

        if (newlen > 0)
        {
            d = static_cast<Data*>(malloc(sizeof(Data) + newlen + oldlen));
            d->m_ref = 0;
            d->m_pData = d->m_buffer;
            memcpy(d->m_buffer, oldData->m_pData, oldlen);
            memcpy(d->m_buffer + oldlen * sizeof(char), str, newlen+1);

            //decref the old data structure
            if (oldData)
            {
                decAndFree(oldData);
            }
        }
    }
}


} //end namespace ito