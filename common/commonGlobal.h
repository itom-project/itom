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

#ifndef COMMONGLOBAL_H
#define COMMONGLOBAL_H

#if (defined ITOMLIBS_SHARED && ( defined(_Windows) || defined(_WINDOWS) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) ))

    #ifndef ITOMCOMMON_EXPORT

        /* Borland/Microsoft */
        #if defined(_MSC_VER) || defined(__BORLANDC__)
            #if (_MSC_VER >= 800) || (__BORLANDC__ >= 0x500)
            #else
                #ifdef ITOMCOMMON_DLL
                    #define ITOMCOMMON_EXPORT __export
                #else
                    #define ITOMCOMMON_EXPORT /*__import */ /* doesn't exist AFAIK in VC++ */
                #endif                              /* Exists in Borland C++ for
                                                                C++ classes (== huge) */
            #endif
        #endif

        #ifndef ITOMCOMMON_EXPORT //ITOMCOMMON_EXPORT has not be defined yet
            #ifdef ITOMCOMMON_DLL
                #define ITOMCOMMON_EXPORT __declspec(dllexport)
            #else
                #define ITOMCOMMON_EXPORT __declspec(dllimport)
            #endif
        #endif

    #endif //ITOMCOMMON_EXPORT

#endif //windows

#ifndef ITOMCOMMON_EXPORT
    #define ITOMCOMMON_EXPORT
#endif


#if (defined ITOMLIBS_SHARED && ( defined(_Windows) || defined(_WINDOWS) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) ))

    #ifndef ITOMCOMMONQT_EXPORT

        /* Borland/Microsoft */
        #if defined(_MSC_VER) || defined(__BORLANDC__)
            #if (_MSC_VER >= 800) || (__BORLANDC__ >= 0x500)
            #else
                #ifdef ITOMCOMMONQT_DLL
                    #define ITOMCOMMONQT_EXPORT __export
                #else
                    #define ITOMCOMMONQT_EXPORT /*__import */ /* doesn't exist AFAIK in VC++ */
                #endif                              /* Exists in Borland C++ for
                                                                C++ classes (== huge) */
            #endif
        #endif

        #ifndef ITOMCOMMONQT_EXPORT //ITOMCOMMONQT_EXPORT has not be defined yet
            #ifdef ITOMCOMMONQT_DLL
                #define ITOMCOMMONQT_EXPORT __declspec(dllexport)
            #else
                #define ITOMCOMMONQT_EXPORT __declspec(dllimport)
            #endif
        #endif

    #endif //ITOMCOMMONQT_EXPORT

#endif //windows

#ifndef ITOMCOMMONQT_EXPORT
    #define ITOMCOMMONQT_EXPORT
#endif

/////// exchange-add operation for atomic operations on reference counters ///////
    #ifdef __GNUC__

      #if __GNUC__*10 + __GNUC_MINOR__ >= 42

        #if !defined WIN32 && (defined __i486__ || defined __i586__ || defined __arm64__ ||\
            defined __i686__ || defined __MMX__ || defined __SSE__  || defined __ppc__)
          #define ITOM_XADD __sync_fetch_and_add
        #else
          #include <ext/atomicity.h>
          #define ITOM_XADD __gnu_cxx::__exchange_and_add
        #endif

      #else
        #include <bits/atomicity.h>
        #if __GNUC__*10 + __GNUC_MINOR__ >= 34
          #define ITOM_XADD __gnu_cxx::__exchange_and_add
        #else
          #define ITOM_XADD __exchange_and_add
        #endif
      #endif

    #elif (defined WIN32 || defined _WIN32)
      #include <intrin.h>
      #define ITOM_XADD(addr,delta) _InterlockedExchangeAdd((long volatile*)(addr), (delta))
    #else
      template<typename _Tp> static inline _Tp ITOM_XADD(_Tp* addr, _Tp delta)
      { int tmp = *addr; *addr += delta; return tmp; }
    #endif

    #define ITOM_INCREF(intvar) ITOM_XADD(intvar,1)
    #define ITOM_DECREF(intvar) ITOM_XADD(intvar,-1)

#endif
