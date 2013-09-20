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

#include <iostream>
#include "pythonStream.h"
#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "structmember.h"
#endif

/*!
    \class PyStream
    \brief static class which implements a new python type. The members cout and cerr of the python system are set to variables of this type PyStream
            in order to observe the python's \a cout and \a cerr stream and to transmit the stream's content to the main application.
*/

//! static method which is called if variable of type PyStream in python workspace has been deleted in order to free allocated memory.
void PyStream::PythonStream_dealloc(PythonStream* self)
{
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//PyObject* PyStream::PythonStream_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
//{
//    PythonStream *self;
//
//    self = (PythonStream *)type->tp_alloc(type, 0);
//    if (self != NULL) 
//    {
//        self->type = 0;
//    }
//
//    return (PyObject *)self;
//};

//! static init method, which is called if any variable of type PyStream is initialized. This method extracts type value from args.
int PyStream::PythonStream_init(PythonStream *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"type", NULL};

    if(args==NULL)
    {
        self->type = 1;
        return 0;
    }

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &self->type))
    {
        return -1; 
    }

    return 0;
};

//! static PyMemberDef table which describes every member of PyStream type
PyMemberDef PyStream::PythonStream_members[] = {
        {"type", T_INT, offsetof(PyStream::PythonStream, type), 0,
         "PythonStream type"},
        {NULL}  /* Sentinel */
    };

//! static method returning name representation of this type
PyObject* PyStream::PythonStream_name(PythonStream* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("pythonStream"); 
    return result;
};

//! static table of type PyMethodDef which contains function pointers and descriptions to all methods, belonging to this type
PyMethodDef PyStream::PythonStream_methods[] = {
        {"name", (PyCFunction)PyStream::PythonStream_name, METH_NOARGS, "name"},
        {"write", (PyCFunction)PyStream::PythonStream_write, METH_VARARGS, "write function"},
        {"flush", (PyCFunction)PyStream::PythonStream_flush, METH_VARARGS, "flush function"},
        {NULL}  /* Sentinel */
    };

//! PyModuleDef table, containing description for PyStream type
PyModuleDef PyStream::pythonStreamModule = {
        PyModuleDef_HEAD_INIT,
        "pythonStream",
        "Example module that creates an extension type.",
        -1,
        NULL, NULL, NULL, NULL, NULL
    };

//! static PyTypeObject for type PyStream with function pointers to all required static methods.
PyTypeObject PyStream::PythonStreamType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "pythonStream.PythonStream",             /* tp_name */
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
        0,		               /* tp_traverse */
        0,		               /* tp_clear */
        0,		               /* tp_richcompare */
        0,		               /* tp_weaklistoffset */
        0,		               /* tp_iter */
        0,		               /* tp_iternext */
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
            v = PyUnicode_AsUTF8String(text);
            if (v == NULL)
            {
                v = PyBytes_FromString("Display error: error parsing the data stream");
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
            if (v == NULL)
            {
                v = PyBytes_FromString("Display error: error parsing the data stream");
            }
            else
            {
                PyObject *temp = v;
                v = PyUnicode_AsUTF8String(temp);
                Py_DECREF(temp);
                if (v == NULL)
                {
                    v = PyBytes_FromString("Display error: error parsing the data stream");
                }
            }
        }
        
        if(self->type == 1)
        {
        
            std::cout << PyBytes_AsString(v); // endl is added directly by Python
        }
        else
        {
            //qDebug() << PyBytes_AsString(v);
            std::cerr << PyBytes_AsString(v); // endl is added directly by Python
        }

        Py_DECREF(v);
        text = NULL;
    }
    Py_INCREF(Py_None);
    return Py_None; //Py_BuildValue("i", 1);
};

//! static method is invoked if stream has been flushed
/*!
    stream is flushed if python is destroyed. Must return Py_None.

    \return PyNone
*/
PyObject* PyStream::PythonStream_flush(PythonStream* /*self*/, PyObject * /*args*/)
{
    Py_INCREF(Py_None);
    return Py_None; //args should be empty, if calling PythonStream_write from this position, garbage collector crash might occure
};
