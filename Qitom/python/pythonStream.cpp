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

#include <iostream>

#include "pythonStream.h"
#include "pythonEngine.h"
#include "pythonCommon.h"
#include "pythonQtConversion.h"
#include "structmember.h"

#include "../AppManagement.h"
#include "../common/sharedStructuresQt.h"

namespace ito {

/*!
    \class PyStream
    \brief static class which implements a new python type. The members cout and cerr of the python system are set to variables of this type PyStream
            in order to observe the python's \a cout and \a cerr stream and to transmit the stream's content to the main application.
*/

//! static method which is called if variable of type PyStream in python workspace has been deleted in order to free allocated memory.
void PyStream::PythonStream_dealloc(PythonStream* self)
{
    Py_XDECREF(self->encoding);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//! static init method, which is called if any variable of type PyStream is initialized. This method extracts type value from args.
int PyStream::PythonStream_init(PythonStream *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"type", NULL};

    if(args==NULL)
    {
        self->type = 1;
        return 0;
    }

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|i", const_cast<char**>(kwlist), &self->type))
    {
        return -1;
    }

    Py_XDECREF(self->encoding);
    self->encoding = PyUnicode_FromString("utf-8"); //the python stdout and stderr streams will be encoded using latin_1
    self->closed = 0;

    return 0;
}

//! static PyMemberDef table which describes every member of PyStream type
PyMemberDef PyStream::PythonStream_members[] = {
    {"type", T_INT, offsetof(PyStream::PythonStream, type), READONLY, "PythonStream type"},
    {"encoding", T_OBJECT_EX, offsetof(PyStream::PythonStream, encoding), READONLY, "Encoding of stream"},
    {"closed", T_BOOL, offsetof(PyStream::PythonStream, closed), READONLY, "Indicates if stream is closed"},
    {NULL}  /* Sentinel */
};

//! static method returning name representation of this type
PyObject* PyStream::PythonStream_name(PythonStream* /*self*/)
{
    return PyUnicode_FromString("pythonStream");
}

PyObject* PyStream::PythonStream_fileno(PythonStream* self)
{
    return PyLong_FromLong( self->type ); //0: in, 1: out, 2: err
}
//! setting IOBase params to False since help(bla) will open a terminal if True.idfc.
PyObject* PyStream::PythonStream_isatty(PythonStream* self, PyObject* args)
{
    Py_RETURN_TRUE;
}
PyObject* PyStream::PythonStream_seekable(PythonStream* self, PyObject* args)
{
    Py_RETURN_FALSE;
}
PyObject* PyStream::PythonStream_writable(PythonStream* self, PyObject* args)
{
    Py_RETURN_FALSE;
}
PyObject* PyStream::PythonStream_readable(PythonStream* self, PyObject* args)
{
    Py_RETURN_FALSE;
}
//! static table of type PyMethodDef which contains function pointers and descriptions to all methods, belonging to this type
PyMethodDef PyStream::PythonStream_methods[] = {
    {"name", (PyCFunction)PyStream::PythonStream_name, METH_NOARGS, "name"},
    {"write", (PyCFunction)PyStream::PythonStream_write, METH_VARARGS, "write function"},
    {"flush", (PyCFunction)PyStream::PythonStream_flush, METH_VARARGS, "flush function"},
    {"readline", (PyCFunction)PyStream::PythonStream_readline, METH_VARARGS, "readline function" },
    {"fileno", (PyCFunction)PyStream::PythonStream_fileno, METH_NOARGS, "returns the virtual file number of this stream (0: in [not supported yet], 1: out, 2: err, 3: in)"},
    {"isatty", (PyCFunction)PyStream::PythonStream_isatty, METH_NOARGS, "returns True."},
    {"seekable", (PyCFunction)PyStream::PythonStream_writable, METH_NOARGS, "returns False." },
    {"writable", (PyCFunction)PyStream::PythonStream_writable, METH_NOARGS, "returns False." },
    {"readable", (PyCFunction)PyStream::PythonStream_seekable, METH_NOARGS, "returns False." },

    {NULL}  /* Sentinel */
};

//! static PyTypeObject for type PyStream with function pointers to all required static methods.
PyTypeObject PyStream::PythonStreamType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.pythonStream",             /* tp_name */
        sizeof(PythonStream),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PythonStream_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        "PythonStream objects",           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,                       /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PythonStream_methods,             /* tp_methods */
        PythonStream_members,             /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyStream::PythonStream_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyType_GenericNew /*PythonStream_new,*/                 /* tp_new */
    };

//! static method invoked if string has been written to stream
/*!
    string is contained in args reference and then sent to std::cout or std::cerr-stream, depending on value of member type.

    \return PyNone
*/
PyObject* PyStream::PythonStream_write(PythonStream* self, PyObject *args)
{

    PyObject* text = PyTuple_GetItem(args, 0);

    if(text != NULL)
    {
        PyObject *v = NULL;
        if (PyUnicode_Check(text))
        {
            v = PyUnicode_AsUTF8String(text); //default encoding of itom/Qt (C-side) is utf8
            if (v == NULL)
            {
				//utf8 failed, try latin1 instead
				PyErr_Clear();
				v = PyUnicode_AsLatin1String(text);

				if (v == NULL)
				{
					PyErr_Print();
					PyErr_Clear();
					v = PyBytes_FromString("");
				}
            }
        }
        else if(PyBytes_Check(text))
        {
            v = text;
            Py_INCREF(v);
        }
        else
        {
            v =  PyObject_Str(text);

            if (v == nullptr)
            {
                PyErr_Print();
                PyErr_Clear();
                v = PyBytes_FromString("");
            }
            else
            {
                PyObject *temp = v;
                v = PyUnicode_AsUTF8String(temp); //default encoding of itom/Qt (C-side) is utf8
                Py_DECREF(temp);

                if (v == nullptr)
                {
                    PyErr_Print();
                    PyErr_Clear();
                    v = PyBytes_FromString("");
                }
            }
        }

        const char* v_ = PyBytes_AsString(v);

        if(self->type == 1)
        {
            std::cout << v_; // endl is added directly by Python
        }
        else if (self->type == 2)
        {
            std::cerr << v_; // endl is added directly by Python
        }

        Py_DECREF(v);
        text = NULL;
    }

    Py_RETURN_NONE;
}

//! static method is invoked if stream has been flushed
/*!
    stream is flushed if python is destroyed. Must return Py_None.

    \return PyNone
*/
PyObject* PyStream::PythonStream_flush(PythonStream* self, PyObject * /*args*/)
{
    if (self->type == 1)
    {
        std::cout << std::flush;
    }
    else if (self->type == 2)
    {
        std::cerr << std::flush;
    }

    Py_RETURN_NONE; //args should be empty, if calling PythonStream_write from this position, garbage collector crash might occure
}

//---------------------------------------------------------------------------------
PyObject* PyStream::PythonStream_readline(PythonStream* self, PyObject *args)
{
    Py_ssize_t size;
    PyObject *arg = Py_None;

    if (!PyArg_ParseTuple(args, "|O:readline", &arg))
        return NULL;

    if (PyLong_Check(arg)) {
        size = PyLong_AsSsize_t(arg);
        if (size == -1 && PyErr_Occurred())
            return NULL;
    }
    else if (arg == Py_None) {
        /* No size limit, by default. */
        size = -1;
    }
    else {
        PyErr_Format(PyExc_TypeError, "integer argument expected, got '%s'",
            Py_TYPE(arg)->tp_name);
        return NULL;
    }

    QSharedPointer<QByteArray> buffer(new QByteArray());

    if (size > 0)
    {
        buffer->resize(size);
        (*buffer)[0] = '\0';
    }

    PythonEngine *pyEng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyEng)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

        emit pyEng->startInputCommandLine(buffer, locker.getSemaphore());

        locker->waitAndProcessEvents(-1);

        if (PythonCommon::setReturnValueMessage(locker->returnValue, "command line", PythonCommon::noMsg))
        {
            if (buffer->size() == 0)
            {
                PyErr_CheckSignals();
                if (!PyErr_Occurred())
                {
                    PyErr_SetNone(PyExc_KeyboardInterrupt);
                }
                return NULL;
            }
            else
            {
                if (buffer->size() > PY_SSIZE_T_MAX)
                {
                    PyErr_SetString(PyExc_OverflowError, "input: input too long");
                    return NULL;
                }
            }

        }
        else
        {
            return NULL;
        }
    }
    else
    {
        PyErr_Format(PyExc_RuntimeError, "Python engine not available for handling input command");
        return NULL;
    }

    return PythonQtConversion::QByteArrayUtf8ToPyUnicodeSecure(*buffer);
}

} //end namespace ito
