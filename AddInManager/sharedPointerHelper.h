/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef SHAREDPOINTERHELPER_H
#define SHAREDPOINTERHELPER_H


template<typename _Tp> static void delete1DArray(_Tp *ptrToArray)
{
    delete[] ptrToArray;
};

template<typename _Tp> static void deleteVoidPtr(void *ptr)
{
    _Tp *ptr2 = reinterpret_cast<_Tp>(ptr);
    if(ptr2) delete ptr2;
};

#endif //SHAREDPOINTERHELPER_H
