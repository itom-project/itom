/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

//###################
//# pythonStream.h  #
//###################

#ifndef PYTHONSTREAM_H
#define PYTHONSTREAM_H

#ifndef Q_MOC_RUN
#include "python/pythonWrapper.h"
#endif

namespace ito
{

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
                int type;   /*!<  1: stream catches cout-stream, 2: stream catches cerr-stream, 3: stream catches cin-stream */
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
        static PyObject *PythonStream_readline(PythonStream* self, PyObject *args);
        static PyObject *PythonStream_isatty(PythonStream* self, PyObject *args);
        static PyObject *PythonStream_seekable(PythonStream* self, PyObject *args);
        static PyObject *PythonStream_writable(PythonStream* self, PyObject *args);
        static PyObject *PythonStream_readable(PythonStream* self, PyObject *args);

        static PyMemberDef PythonStream_members[];
        static PyMethodDef PythonStream_methods[];
        static PyTypeObject PythonStreamType;
    };

} //end namespace ito


#endif
