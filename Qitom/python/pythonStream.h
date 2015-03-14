/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

//###################
//# pythonStream.h  #
//###################

#ifndef PYTHONSTREAM_H
#define PYTHONSTREAM_H

#ifndef Q_MOC_RUN
    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && ((!defined linux) || (!defined __APPLE__))
        #undef _DEBUG
        #include "Python.h"
        #define _DEBUG
    #else
        #include "Python.h"
    #endif
#endif


class PyStream
{

public:

    //!* this struct builds the basic structure of the new python type.
    /*
        The struct's name is PythonStream. This python type consists of the basic elements for every python type,
        included an integer value type, which indicates whether this stream corresponds to the stream \a cout or \a cerr.
    */
    typedef struct 
    {
        PyObject_HEAD
        int type;   /*!<  1: stream catches cout-stream, 2: stream catches cerr-stream */
        PyObject *encoding;
        char closed;
    } 
    PythonStream;

    static void PythonStream_dealloc(PythonStream* self);
    //static PyObject *PythonStream_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PythonStream_init(PythonStream *self, PyObject *args, PyObject *kwds);

    static PyObject *PythonStream_name(PythonStream* self);
    static PyObject *PythonStream_fileno(PythonStream* self);
    static PyObject *PythonStream_write(PythonStream* self, PyObject *args);
    static PyObject *PythonStream_flush(PythonStream* self, PyObject *args);
    
    static PyMemberDef PythonStream_members[];
    static PyMethodDef PythonStream_methods[];
    static PyTypeObject PythonStreamType;
    static PyModuleDef pythonStreamModule;
};


#endif
