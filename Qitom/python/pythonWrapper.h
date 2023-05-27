/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2022, Institut fuer Technische Optik (ITO),
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

// python
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (defined WIN32)
    // see https://github.com/microsoft/onnxruntime/issues/9735#issuecomment-970718821
    #include <corecrt.h>
    #undef _DEBUG

    // work around following: https://stackoverflow.com/questions/23068700/embedding-python3-in-qt-5
    #pragma push_macro("slots")
    #undef slots
    #include "Python.h"
    #include "numpy/arrayobject.h"
    #include "numpy/arrayscalars.h"
    #include "datetime.h"
    #pragma pop_macro("slots")

    #define _DEBUG
#else
    // work around following: https://stackoverflow.com/questions/23068700/embedding-python3-in-qt-5
    #pragma push_macro("slots")
    #undef slots
    #include "Python.h"
    #include "numpy/arrayobject.h"
    #include "numpy/arrayscalars.h"
    #include "datetime.h"
    #pragma pop_macro("slots")
#endif


// use this macro if a method of the C-API of the Python datetime module should be used.
#define Itom_PyDateTime_IMPORT                                                                     \
    if (PyDateTimeAPI == nullptr)                                                                  \
    {                                                                                              \
        PyDateTime_IMPORT;                                                                         \
    }
