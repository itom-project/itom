#ifndef AAA_H
#define AAA_H

// This file is a dirty litte trick for the automoccer of qt - we MUST include our python stuff before ANY
// qt stuff so we make this class which is first in the file list and first in alphabet and gets included
// in the automoccer cpp file first 

#ifndef ITOM_NPDATAOBJECT
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //see comment in pythonNpDataObject.cpp
#endif

//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (!defined linux)
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
        #include "node.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "node.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
    #define _DEBUG
#else
    #ifdef linux
        #include "Python.h"
        #include "node.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "node.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
#endif

#include <qobject.h>

class qDummyClass : public QObject
{
    Q_OBJECT
    public:
        qDummyClass() {};
        ~qDummyClass() {};

    private:
};

#endif // AAA_H