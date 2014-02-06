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

namespace ito
{

#if 1
    /*non thread-safe but faster*/
    #define BYTEARRAY_DECREF(d) d->m_ref--
    #define BYTEARRAY_INCCREF(d) d->m_ref++
#else
    /*thread safe*/
    #define BYTEARRAY_DECREF(d) ITOM_DECREF(&(d->m_ref))
    #define BYTEARRAY_INCCREF(d) ITOM_INCREF(&(d->m_ref))
#endif

//----------------------------------------------------------------------------------------------------------------------------------
// @class ByteArray
class ITOMCOMMON_EXPORT ByteArray
{       
    private:
        struct Data
        {
            int m_ref;               /*!< reference counter (0: means one reference, ...) */
            char *m_pData;    
            char m_buffer[1];        
            //do not append further members add, only prepend!!!
        };

        static char emptyChar;

    public:
        inline ByteArray() : d(NULL) {}
        
        ByteArray(const char *str);
        
        inline ByteArray(const ByteArray& copyConstr) : d(copyConstr.d) { if (d) {BYTEARRAY_INCCREF(d);} }
        
        inline ~ByteArray() { decAndFree(d); }

        ByteArray &operator=(const ByteArray &rhs);

        void append(const char *str);

        void append(const ByteArray &str);

        int length() const { if(d){ return strlen(d->m_pData); } return 0; }

        int size() const { if(d){ return strlen(d->m_pData); } return 0; }

        const char *data() const { return d ? d->m_pData : &emptyChar; };

        const char *lazyData() const { return d ? d->m_pData : NULL; };

        inline char &operator[](unsigned int i) const
        {
            assert(i >= 0 && i < (unsigned int)(size()));
            if (d)
            {
                return d->m_pData[i];
            }
            return emptyChar; //will never occur
        }

        inline char &operator[](int i) const
        {
            assert(i >= 0 && i < size());
            if (d)
            {
                return d->m_pData[i];
            }
            return emptyChar; //will never occur
        }

        bool operator==(const ByteArray &a) const;
        inline bool operator!=(const ByteArray &a) const { return !(operator==(a)); }       
        

    private:
        Data *d;

        inline void decAndFree(Data *x) 
        { 
            if (x && !(BYTEARRAY_DECREF(x))) 
            { 
                free(x); 
            }
        }


};

inline bool operator==(const ByteArray &a1, const char *a2)
{ 
    return a2 ? strcmp(a1.data(),a2) == 0 : (a1.size() == 0);  
}

inline bool operator==(const char *a1, const ByteArray &a2)
{ 
    return a1 ? strcmp(a1,a2.data()) == 0 : (a2.size() == 0);
}

inline bool operator!=(const ByteArray &a1, const char *a2)
{ 
    return a2 ? strcmp(a1.data(),a2) != 0 : (a1.size() > 0); 
}
inline bool operator!=(const char *a1, const ByteArray &a2)
{ 
    return a1 ? strcmp(a1,a2.data()) != 0 : (a2.size() > 0); 
}


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
