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

#if (defined ITOMLIBS_SHARED && ( defined(_Windows) || defined(_WINDOWS) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) ))

    #ifndef DATAOBJ_EXPORT

        /* Borland/Microsoft */
        #if defined(_MSC_VER) || defined(__BORLANDC__)
            #if (_MSC_VER >= 800) || (__BORLANDC__ >= 0x500)
            #else
                #ifdef DATAOBJECT_DLL
                    #define DATAOBJ_EXPORT __export
                #else
                    #define DATAOBJ_EXPORT /*__import */ /* doesn't exist AFAIK in VC++ */
                #endif                              /* Exists in Borland C++ for
                                                                C++ classes (== huge) */
            #endif
        #endif

        #ifndef DATAOBJ_EXPORT //DATAOBJ_EXPORT has not be defined yet
            #ifdef DATAOBJECT_DLL
                #define DATAOBJ_EXPORT __declspec(dllexport)
            #else
                #define DATAOBJ_EXPORT __declspec(dllimport)
            #endif
        #endif

    #endif //DATAOBJ_EXPORT

#endif //windows

#ifndef DATAOBJ_EXPORT
    #define DATAOBJ_EXPORT
#endif
