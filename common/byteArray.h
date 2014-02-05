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

    public:
        inline ByteArray() : d(NULL) {}
        
        ByteArray(const char *str);
        
        inline ByteArray(const ByteArray& copyConstr) : d(copyConstr.d) { if (d) {BYTEARRAY_INCCREF(d);} }
        
        inline ~ByteArray() { decAndFree(d); }

        ByteArray &operator=(const ByteArray &rhs);

        void append(const char *str);

        const char *data() const { return d ? d->m_pData : NULL; };

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


} //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
