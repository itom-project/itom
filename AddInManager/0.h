#ifndef A_H
#define A_H

// This file is a dirty litte trick for the automoccer of qt - we MUST include our python stuff before ANY
// qt stuff so we make this class which is first in the file list and first in alphabet and gets included
// in the automoccer cpp file first

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

///python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (defined WIN32)
    // see https://github.com/microsoft/onnxruntime/issues/9735#issuecomment-970718821
    #include <corecrt.h>


    #undef _DEBUG

    //workaround following: https://stackoverflow.com/questions/23068700/embedding-python3-in-qt-5
    #pragma push_macro("slots")
    #undef slots
    #include "Python.h"
    #pragma pop_macro("slots")
    #include "numpy/arrayobject.h" //for numpy arrays
    #define _DEBUG
#else
    //workaround following: https://stackoverflow.com/questions/23068700/embedding-python3-in-qt-5
    #pragma push_macro("slots")
    #undef slots
    #include "Python.h"
    #pragma pop_macro("slots")
    #include "numpy/arrayobject.h"
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

#endif // A_H
