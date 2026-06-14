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

#ifndef BYTEARRAY_H
#define BYTEARRAY_H

#ifdef __APPLE__
extern "C++" {
#endif

/* includes */
#include "commonGlobal.h"
#include "typeDefs.h"

#include <stdarg.h>
#include <stdio.h>
#include <assert.h>     /* assert */

#include <cstdlib>

namespace ito
{


/*thread safe*/
#define BYTEARRAY_DECREF(d) ITOM_DECREF(&(d->m_ref))
#define BYTEARRAY_INCCREF(d) ITOM_INCREF(&(d->m_ref))

//----------------------------------------------------------------------------------------------------------------------------------

/*!
\class ByteArray
\brief This is a Qt-free class for byte arrays (strings) without specific encoding information

This class uses implicit sharing (copy-on-write) to reduce memory usage and to avoid the needless
copying of data. It can be used to store 8-bit '\0' terminated strings and is a easy and safe
way to handle const char* types. ByteArray is a leightweight class without any dependency to Qt
and is used in many itom libraries e.g. the Qt-free itomCommonLib library.
*/
class ITOMCOMMON_EXPORT ByteArray
{
    private:
        /*!
        \struct Data
        \brief basic data container for class ByteArray that is implicitely shared over multiple instances of ByteArray of the same content

        The address of the real character array is stored in m_pData. The data buffer itself starts at m_buffer
        and continues at the end of the struct. Therefore, ByteArray allocates memory for Data that has the length
        of data + the length of the character array to store. The number of ByteArray instances that have access
        to exactly the same Data container is stored in the reference counter m_ref.

        The basic principle of the implicit sharing is taken from QByteArray.
        */
        struct Data
        {
            int m_ref;               /*!< reference counter for implicit sharing (0: means one reference, ...) */
            char *m_pData;           /*!< pointer to character array that begins at m_buffer */
            char m_buffer[1];        /*!< start of character array buffer. The real buffer is longer than one char value, therefore no member must be appended after this point. */
            //do not append further members add, only prepend!!!
        };

        static char emptyChar; /*!< static character that represents an empty byte array ('\0'). */

    public:
        //! default constructor. The ByteArray is empty.
        inline ByteArray() : d(NULL) {}

        //! constructor that copies the content of str ('\0'-terminated) to this ByteArray.
        ByteArray(const char *str);

        //! copy constructor: the given byte array is implicitely shared between both instances until its content is changed by one of both participating instances.
        inline ByteArray(const ByteArray& copyConstr) : d(copyConstr.d)
        {
            if (d)
            {
                BYTEARRAY_INCCREF(d);
            }
        }

        inline ByteArray(ByteArray&& other) : d(other.d)
        {
            other.d = nullptr;
        }

        //! destructor: the internal data is deleted if no other instance needs it.
        inline ~ByteArray() { decAndFree(d); }

        //! another ByteArray is assigned to this ByteArray. The old content is deleted and the given byte array is implicitely shared between both instances.
        ByteArray &operator=(const ByteArray &rhs);

        ByteArray &operator=(ByteArray &&rhs);

        //! a zero-terminated string is assigned to this ByteArray. The given char* is copied.
        ByteArray &operator=(const char *str);

        /*! a zero-terminated string is appended to this ByteArray.
        If the ByteArray implictely shared its content with another one,
        the contents are duplicated before appending the new string.*/
        ByteArray &append(const char *str);

        /*! the content of another ByteArray is appended to this ByteArray.
        If the ByteArray implictely shared its content with another one,
        the contents are duplicated before appending the new ByteArray. */
        ByteArray &append(const ByteArray &str);

        /*! return the length of the byte array or 0 if it is empty
            \sa size
        */
        int length() const { if(d){ return (int)strlen(d->m_pData); } return 0; }

        /*! return the length of the byte array or 0 if it is empty
        \sa length
        */
        int size() const { if(d){ return (int)strlen(d->m_pData); } return 0; }

        //! return true if the ByteArray is empty hence its length is 0
        bool empty() const { if(d) { return strlen(d->m_pData) == 0; } return true; }

        //! return the pointer to the internal character array. If it is empty, the returned pointer still points to a '\0' character. It is never NULL.
        const char *data() const { return d ? d->m_pData : &emptyChar; };

        //! access the i-th character of the ByteArray. An assertion is raised, if i is out of range
        /*!
            \param i is the index of the requested character [0, length()-1]
            \return i-th character
        */
        inline char &operator[](unsigned int i) const
        {
            assert(i >= 0 && i < (unsigned int)(size()));
            if (d)
            {
                return d->m_pData[i];
            }
            return emptyChar; //will never occur
        }

        //! access the i-th character of the ByteArray. An assertion is raised, if i is out of range
        /*!
        \param i is the index of the requested character [0, length()-1]
        \return i-th character
        */
        inline char &operator[](int i) const
        {
            assert(i >= 0 && i < size());
            if (d)
            {
                return d->m_pData[i];
            }
            return emptyChar; //will never occur
        }

        //! return true, if the content of this ByteArray is equal to the given ByteArray a.
        bool operator==(const ByteArray &a) const;

        //! return false, if the content of this ByteArray is equal to the given ByteArray a.
        inline bool operator!=(const ByteArray &a) const { return !(operator==(a)); }


    private:
        Data *d;  /*!< pointer to Data container */

        inline void decAndFree(Data *x)
        {
            if (x && !(BYTEARRAY_DECREF(x)))
            {
                free(x);
            }
        }


};

//! comparison operator that returns true if the content of a1 is equal to the given zero-terminated string a2.
inline bool operator==(const ByteArray &a1, const char *a2)
{
    return a2 ? strcmp(a1.data(),a2) == 0 : (a1.size() == 0);
}

//! comparison operator that returns true if the content of a2 is equal to the given zero-terminated string a1.
inline bool operator==(const char *a1, const ByteArray &a2)
{
    return a1 ? strcmp(a1,a2.data()) == 0 : (a2.size() == 0);
}

//! comparison operator that returns true if the content of a1 is not equal to the given zero-terminated string a2.
inline bool operator!=(const ByteArray &a1, const char *a2)
{
    return a2 ? strcmp(a1.data(),a2) != 0 : (a1.size() > 0);
}

//! comparison operator that returns true if the content of a2 is not equal to the given zero-terminated string a1.
inline bool operator!=(const char *a1, const ByteArray &a2)
{
    return a1 ? strcmp(a1,a2.data()) != 0 : (a2.size() > 0);
}


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
