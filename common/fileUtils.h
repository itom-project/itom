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

#ifndef FILEUTILS_H
#define FILEUTILS_H

#include "retVal.h"
#include "typeDefs.h"
#include <qendian.h>
#include <qiodevice.h>


#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    //all these methods increment the input pointer by the size of the returned number
    inline ito::int8 getInt8(const uchar **ppval);
    inline ito::uint8 getUInt8(const uchar **ppval);
    inline ito::int16 getInt16LE(const uchar **ppval);
    inline ito::int16 getInt16BE(const uchar **ppval);
    inline ito::uint16 getUInt16LE(const uchar **ppval);
    inline ito::uint16 getUInt16BE(const uchar **ppval);
    inline ito::int32 getInt32LE(const uchar **ppval);
    inline ito::int16 getInt16BE(const uchar **ppval);
    inline ito::uint32 getUInt32LE(const uchar **ppval);
    inline ito::uint32 getUInt32BE(const uchar **ppval);
    inline qint64 getInt64LE(const uchar **ppval);
    inline qint64 getInt64BE(const uchar **ppval);
    inline quint64 getUInt64LE(const uchar **ppval);
    inline quint64 getUInt64BE(const uchar **ppval);
    inline ito::float32 getFloat32LE(const uchar **ppval);
    inline ito::float32 getFloat32BE(const uchar **ppval);
    inline ito::float64 getFloat64LE(const uchar **ppval);
    inline ito::float64 getFloat64BE(const uchar **ppval);

    inline ito::int16 swapInt16(ito::int16 val);
    inline ito::uint16 swapUInt16(ito::uint16 val);
    inline ito::int32 swapInt32(ito::int32 val);
    inline ito::uint32 swapUInt32(ito::uint32 val);
    inline ito::float32 swapFloat32(ito::float32 val);
    inline ito::float64 swapFloat64(ito::float64 val);

    //!< reads exactly numBytes from device into data and returns an error if less or no data is available
    ito::RetVal ITOMCOMMONQT_EXPORT readFromDevice(QIODevice *device, char *data, qint64 numBytes);

    //sources

    //--------------------------------------------------------------------------------------------
    inline ito::int8 getInt8(const uchar **ppval)
    {
        const ito::int8 *pval = (const ito::int8*)(*ppval);
        ito::int8 v = *pval;
        *ppval += sizeof(ito::uint8);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::uint8 getUInt8(const uchar **ppval)
    {
        const ito::uint8 *pval = (const ito::uint8*)(*ppval);
        ito::uint8 v = *pval;
        *ppval += sizeof(ito::uint8);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::int16 getInt16LE(const uchar **ppval)
    {
        const ito::int16 *pval = (const ito::int16*)(*ppval);
        ito::int16 v = qFromLittleEndian(*pval);
        *ppval += sizeof(ito::int16);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::int16 getInt16BE(const uchar **ppval)
    {
        const ito::int16 *pval = (const ito::int16*)(*ppval);
        ito::int16 v = qFromBigEndian(*pval);
        *ppval += sizeof(ito::int16);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::uint16 getUInt16LE(const uchar **ppval)
    {
        const ito::uint16 *pval = (const ito::uint16*)(*ppval);
        ito::uint16 v = qFromLittleEndian(*pval);
        *ppval += sizeof(ito::uint16);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::uint16 getUInt16BE(const uchar **ppval)
    {
        const ito::uint16 *pval = (const ito::uint16*)(*ppval);
        ito::uint16 v = qFromBigEndian(*pval);
        *ppval += sizeof(ito::uint16);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::int32 getInt32LE(const uchar **ppval)
    {
        const ito::int32 *pval = (const ito::int32*)(*ppval);
        ito::int32 v = qFromLittleEndian(*pval);
        *ppval += sizeof(ito::int32);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::int32 getInt32BE(const uchar **ppval)
    {
        const ito::int32 *pval = (const ito::int32*)(*ppval);
        ito::int32 v = qFromBigEndian(*pval);
        *ppval += sizeof(ito::int32);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::uint32 getUInt32LE(const uchar **ppval)
    {
        const ito::uint32 *pval = (const ito::uint32*)(*ppval);
        ito::uint32 v = qFromLittleEndian(*pval);
        *ppval += sizeof(ito::uint32);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::uint32 getUInt32BE(const uchar **ppval)
    {
        const ito::uint32 *pval = (const ito::uint32*)(*ppval);
        ito::uint32 v = qFromBigEndian(*pval);
        *ppval += sizeof(ito::uint32);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline qint64 getInt64LE(const uchar **ppval)
    {
        const qint64 *pval = (const qint64*)(*ppval);
        qint64 v = qFromLittleEndian(*pval);
        *ppval += sizeof(qint64);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline qint64 getInt64BE(const uchar **ppval)
    {
        const qint64 *pval = (const qint64*)(*ppval);
        qint64 v = qFromBigEndian(*pval);
        *ppval += sizeof(qint64);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline quint64 getUInt64LE(const uchar **ppval)
    {
        const quint64 *pval = (const quint64*)(*ppval);
        quint64 v = qFromLittleEndian(*pval);
        *ppval += sizeof(quint64);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline quint64 getUInt64BE(const uchar **ppval)
    {
        const quint64 *pval = (const quint64*)(*ppval);
        quint64 v = qFromBigEndian(*pval);
        *ppval += sizeof(quint64);
        return v;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::float32 getFloat32LE(const uchar **ppval)
    {
        union
        {
            ito::uint8 pp[4];
            ito::float32 f;
        } z;

#if (Q_BYTE_ORDER == Q_LITTLE_ENDIAN)
        memcpy(z.pp, *ppval, sizeof(ito::float32));
#else
        z.pp[0] = (*ppval)[3];
        z.pp[1] = (*ppval)[2];
        z.pp[2] = (*ppval)[1];
        z.pp[3] = (*ppval)[0];
#endif
        *ppval += sizeof(ito::float32);
        return z.f;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::float32 getFloat32BE(const uchar **ppval)
    {
        union
        {
            ito::uint8 pp[4];
            ito::float32 f;
        } z;

#if (Q_BYTE_ORDER == Q_BIG_ENDIAN)
        memcpy(z.pp, *ppval, sizeof(ito::float32));
#else
        z.pp[0] = (*ppval)[3];
        z.pp[1] = (*ppval)[2];
        z.pp[2] = (*ppval)[1];
        z.pp[3] = (*ppval)[0];
#endif
        *ppval += sizeof(ito::float32);
        return z.f;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::float64 getFloat64LE(const uchar **ppval)
    {
        union
        {
            ito::uint8 pp[8];
            ito::float64 d;
        } z;

#if (Q_BYTE_ORDER == Q_LITTLE_ENDIAN)
        memcpy(z.pp, *ppval, sizeof(ito::float64));
#else
        z.pp[0] = (*ppval)[7];
        z.pp[1] = (*ppval)[6];
        z.pp[2] = (*ppval)[5];
        z.pp[3] = (*ppval)[4];
        z.pp[4] = (*ppval)[3];
        z.pp[5] = (*ppval)[2];
        z.pp[6] = (*ppval)[1];
        z.pp[7] = (*ppval)[0];
#endif
        *ppval += sizeof(ito::float64);
        return z.d;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::float64 getFloat64BE(const uchar **ppval)
    {
        union
        {
            ito::uint8 pp[8];
            ito::float64 d;
        } z;

#if (Q_BYTE_ORDER == Q_BIG_ENDIAN)
        memcpy(z.pp, *ppval, sizeof(ito::float64));
#else
        z.pp[0] = (*ppval)[7];
        z.pp[1] = (*ppval)[6];
        z.pp[2] = (*ppval)[5];
        z.pp[3] = (*ppval)[4];
        z.pp[4] = (*ppval)[3];
        z.pp[5] = (*ppval)[2];
        z.pp[6] = (*ppval)[1];
        z.pp[7] = (*ppval)[0];
#endif
        *ppval += sizeof(ito::float64);
        return z.d;
    }





    //--------------------------------------------------------------------------------------------
    inline ito::uint16 swapUInt16(ito::uint16 val)
    {
        union s
        {
            char sa[2];
            ito::uint16 res;
        } temp;
        temp.res = val;
        s uint16out;
        for (int teller = 0; teller<2; ++teller){
            uint16out.sa[teller] = temp.sa[1 - teller];
        }
        return uint16out.res;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::int16 swapInt16(ito::int16 val)
    {
        union s
        {
            char sa[2];
            ito::int16 res;
        } temp;
        temp.res = val;
        s int16out;
        for (int teller = 0; teller<2; ++teller){
            int16out.sa[teller] = temp.sa[1 - teller];
        }
        return int16out.res;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::uint32 swapUInt32(ito::uint32 val)
    {
        union s
        {
            char sa[4];
            ito::uint32 res;
        } temp;
        temp.res = val;
        s uint32out;
        for (int teller = 0; teller<4; ++teller){
            uint32out.sa[teller] = temp.sa[3 - teller];
        }
        return uint32out.res;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::int32 swapInt32(ito::int32 val)
    {
        union s
        {
            char sa[4];
            ito::int32 res;
        } temp;
        temp.res = val;
        s int32out;
        for (int teller = 0; teller<4; ++teller){
            int32out.sa[teller] = temp.sa[3 - teller];
        }
        return int32out.res;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::float32 swapFloat32(ito::float32 val)
    {
        union s
        {
            char pp[4];
            ito::float32 f;
        } temp;

        temp.f = val;
        s float32out;
        float32out.pp[0] = temp.pp[3];
        float32out.pp[1] = temp.pp[2];
        float32out.pp[2] = temp.pp[1];
        float32out.pp[3] = temp.pp[0];
        return float32out.f;
    }

    //--------------------------------------------------------------------------------------------
    inline ito::float64 swapFloat64(ito::float64 val)
    {
        union s
        {
            char pp[8];
            ito::float64 f;
        } temp;

        temp.f = val;
        s float64out;
        float64out.pp[0] = temp.pp[7];
        float64out.pp[1] = temp.pp[6];
        float64out.pp[2] = temp.pp[5];
        float64out.pp[3] = temp.pp[4];
        float64out.pp[4] = temp.pp[3];
        float64out.pp[5] = temp.pp[2];
        float64out.pp[6] = temp.pp[1];
        float64out.pp[7] = temp.pp[0];
        return float64out.f;
    }

};   // end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
