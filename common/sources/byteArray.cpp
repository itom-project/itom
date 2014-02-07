/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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
        
        if (newlen > 0)
        {
            if (d)
            {
                int oldlen = strlen(d->m_pData);

                if (d->m_ref == 0) //this is the only user
                {
                    d = static_cast<Data*>(realloc(d, sizeof(Data) + newlen + oldlen));
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
}

//----------------------------------------------------------------------------------------------------------------------------------
void ByteArray::append(const ByteArray &str)
{
    int newlen = str.length();
    Data *oldData = d;
    int oldlen = strlen(d->m_pData);

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
}

//----------------------------------------------------------------------------------------------------------------------------------
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