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

#if (defined ITOMLIBS_SHARED && ( defined(_Windows) || defined(_WINDOWS) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) )) 
    
    #ifndef ADDINMGR_EXPORT

        // Borland/Microsoft
        #if defined(_MSC_VER) || defined(__BORLANDC__)
            #if (_MSC_VER >= 800) || (__BORLANDC__ >= 0x500)
            #else
                #ifdef ADDINMGR_DLL
                    #define ADDINMGR_EXPORT __export
                    #define static AddInManager *AddInManagerInst = NULL;
                #else
                    #define ADDINMGR_EXPORT     //__import doesn't exist AFAIK in VC++ Exists in Borland C++ for C++ classes (== huge)
                    #define extern AddInManager *AddInManagerInst;
                #endif
            #endif
        #endif

        #ifndef ADDINMGR_EXPORT //ADDINMGR_EXPORT has not be defined yet
            #ifdef ADDINMGR_DLL
                #define ADDINMGR_EXPORT __declspec(dllexport)
                namespace ito {
                    class AddInManager;
                    static AddInManager *AddInManagerInst = 0;
                }
            #else
                #define ADDINMGR_EXPORT __declspec(dllimport)
                namespace ito {
                    class AddInManager;
                    extern AddInManager *AddInManagerInst;
                }
            #endif
        #endif

    #endif //ADDINMGR_EXPORT

#endif //windows

#ifndef ADDINMGR_EXPORT
    #define ADDINMGR_EXPORT
    #define extern AddInManager *AddInManagerInst;
#endif