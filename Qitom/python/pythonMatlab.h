/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONMATLAB_H
#define PYTHONMATLAB_H

#if ITOM_PYTHONMATLAB == 1
/* * *
 * Copyright 2010 Joakim Mller
 *
 * This file is part of pymatlab.
 * 
 * pymatlab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * pymatlab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with pymatlab.  If not, see <http://www.gnu.org/licenses/>.
 * * */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#if (defined _DEBUG) && (defined WIN32)
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #elif (defined __APPLE__) | (defined CMAKE)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
    #define _DEBUG
#else
    #if (defined linux)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #elif (defined __APPLE__)
        #include "Python.h"
        #include "numpy/arrayobject.h"
    #else
        #include "Python.h"
        #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
    #endif
#endif


#include <stdio.h>
#include <engine.h>

#define PYMATLAB_VERSION "0.2"
#define AUTHOR "Joakim Moller, Marc Gronle (Python 3)"
#define PYTHON3

/* This wraps a command call to the MATLAB interpretor */
const char * function_wrap = "try\n\t%s;\ncatch err\n\tpymatlaberrstring = sprintf('Error: %%s with message: %%s\\n',err.identifier,err.message);\n\tfor i = 1:length(err.stack)\n\t\tpymatlaberrstring = sprintf('%%sError: in fuction %%s in file %%s line %%i\\n',pymatlaberrstring,err.stack(i,1).name,err.stack(i,1).file,err.stack(i,1).line);\n\tend\nend\nif exist('pymatlaberrstring','var')==0\n\tpymatlaberrstring='';\nend";

int mxtonpy[17] = {
    NPY_USERDEF,    /* mxUNKNOWN_CLASS (0) */
    NPY_USERDEF,    /* mxCELL_CLASS (1) */
    NPY_USERDEF,     /* mxSTRUCT_CLASS (2) */
    NPY_BOOL,     /* mxLOGICAL_CLASS (3) */
    NPY_CHAR,     /* mxCHAR_CLASS (4) */
    NPY_USERDEF,     /* mxVOID_CLASS (5) */
    NPY_DOUBLE,         /* mxDOUBLE_CLASS (6) */
    NPY_FLOAT,     /* mxSINGLE_CLASS (7) */
    NPY_BYTE,     /* mxINT8_CLASS (8) */
    NPY_UBYTE,     /* mxUINT8_CLASS (9) */
    NPY_SHORT,    /* mxINT16_CLASS (10) */
    NPY_USHORT,     /* mxUINT16_CLASS (11) */
    NPY_INT,     /* mxINT32_CLASS (12) */
    NPY_UINT,     /* mxUINT32_CLASS (13) */
    NPY_LONG,     /* mxINT64_CLASS (14) */
    NPY_ULONG,     /* mxUINT64_CLASS (15) */
    NPY_USERDEF     /* mxFUNCTION_CLASS (16) */
};

mxClassID npytomx[23]={ mxLOGICAL_CLASS, /*NPY_BOOL (0)*/
                        mxINT8_CLASS, /*NPY_BYTE (1)*/
                        mxUINT8_CLASS, /*NPY_UBYTE (2)*/
                        mxINT16_CLASS,  /*NPY_SHORT (3)*/
                        mxUINT16_CLASS, /*NPY_USHORT (4)*/
                        mxINT32_CLASS,  /*NPY_INT (5)*/
                        mxUINT32_CLASS, /*NPY_UINT (6)*/
                        mxINT64_CLASS,  /*NPY_LONG (7)*/
                        mxUINT64_CLASS, /*NPY_ULONG (8)*/
                        mxUNKNOWN_CLASS,  /*NPY_LONGLONG (9)*/
                        mxUNKNOWN_CLASS, /*NPY_ULONGLONG (10)*/
                        mxSINGLE_CLASS,  /*NPY_FLOAT (11)*/
                        mxDOUBLE_CLASS,  /*NPY_DOUBLE (12)*/
                        mxUNKNOWN_CLASS, /*NPY_LONGDOUBLE (13)*/
                        mxSINGLE_CLASS,  /*NPY_CFLOAT (14)*/
                        mxDOUBLE_CLASS,  /*NPY_CDOUBLE (15)*/
                        mxUNKNOWN_CLASS, /*NPY_CLONGDOUBLE (16)*/
                        mxUNKNOWN_CLASS, /*NPY_OBJECT (17)*/
                        mxCHAR_CLASS,  /*NPY_STRING (18)*/
                        mxCHAR_CLASS, /*NPY_UNICODE (19)*/
                        mxUNKNOWN_CLASS, /*NPY_VOID (20)*/
                        mxUNKNOWN_CLASS, /*NPY_DATETIME (21)*/
                        mxCHAR_CLASS, /*NPY_TIMEDELTA (22)*/
};

typedef struct 
{
    PyObject_HEAD
    Engine *ep;
} PyMatlabSessionObject;

static PyObject * PyMatlabSessionObject_new(PyTypeObject *type, PyObject *args, PyObject * kwds)
{
    PyMatlabSessionObject *self;
    self = (PyMatlabSessionObject *) type->tp_alloc(type,0);
    if (self!=NULL)
    {
        self->ep = NULL;
    }
    return (PyObject *) self;
}

static int PyMatlabSessionObject_init(PyMatlabSessionObject *self, PyObject *args, PyObject *kwds)
{
    int status;
    char *startstr=NULL;
    if (!PyArg_ParseTuple(args,"|s",&startstr))
        return EXIT_FAILURE;
    if (!(self->ep = engOpen(startstr))) {
        fprintf(stderr, "\nCan't start MATLAB engine\n");
        return EXIT_FAILURE;
    }
    status = engOutputBuffer(self->ep,NULL,0);
    return 0;
}

static void
PyMatlabSessionObject_dealloc(PyMatlabSessionObject *self)
{
    if(self->ep)
    {
        engClose(self->ep);
    }
    self->ob_base.ob_type->tp_free((PyObject*)self);
}

static PyObject * PyMatlabSessionObject_run(PyMatlabSessionObject *self, PyObject *args)
{
    char * stringarg;
    char * command;
    char * errmsg;
    int status;
    const mxArray * mxresult;
    PyObject * result;

    if(self->ep == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Matlab engine has not been started");
        return NULL;
    }

    if (!PyArg_ParseTuple(args,"s",&stringarg))
        return NULL;
    if (!(command = (char*)malloc(sizeof(char)*3000)))
        return NULL;
    sprintf(command,function_wrap,stringarg);
    if (engEvalString(self->ep,command)!=0)
    {
        return PyErr_Format(PyExc_RuntimeError, "Was not able to evaluate command: %s",command);
    }
    if ((mxresult = engGetVariable(self->ep,"pymatlaberrstring"))==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "can't get internal variable: pymatlaberrstring");
        return NULL;
    }
    if (strcmp( mxArrayToString(mxresult),"")!=0)
    {
        /*make sure 'pymatlaberrstring' is empty or not exist until next call*/
        status = engEvalString(self->ep,"clear pymatlaberrstring");
        return PyErr_Format(PyExc_RuntimeError,"Error from Matlab: %s end.", mxArrayToString(mxresult));
    }
    free((void*)command);
    Py_RETURN_NONE;
}

static PyObject * PyMatlabSessionObject_setValue(PyMatlabSessionObject *self, PyObject *args)
{
    const char * name;
    //PyArrayObject * ndarray;
    PyArrayObject * cont_ndarray;
    PyObject *obj;
    mxArray * mxarray;
    void *mx,*nd;
    float *mxReal32, *mxImag32, *nd32;
    double *mxReal64, *mxImag64, *nd64;
    int i,j;
    mwSize dims[] = {1};

    if(self->ep == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Matlab engine has not been started");
        return NULL;
    }

    if (!PyArg_ParseTuple(args,"sO",&name,&obj))
    {
        return NULL;
    }

    if(PyLong_Check(obj))
    {
        if (!(mxarray=mxCreateNumericArray(1, dims, mxINT64_CLASS , mxREAL)))
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't create mxarray.");
            return NULL;
        }

        ((long*)mxGetData(mxarray))[0] = PyLong_AsLong(obj);
    }
    else if(PyFloat_Check(obj))
    {
        if (!(mxarray=mxCreateNumericArray(1, dims, mxDOUBLE_CLASS , mxREAL)))
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't create mxarray.");
            return NULL;
        }

        ((double*)mxGetData(mxarray))[0] = PyFloat_AsDouble(obj);
    }
    else if(PyComplex_Check(obj))
    {
        if (!(mxarray=mxCreateNumericArray(1, dims, mxDOUBLE_CLASS , mxCOMPLEX)))
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't create mxarray.");
            return NULL;
        }

        ((double*)mxGetData(mxarray))[0] = PyComplex_RealAsDouble(obj);
        ((double*)mxGetImagData(mxarray))[0] = PyComplex_ImagAsDouble(obj);
    }
    else
    {
        #if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
            cont_ndarray = (PyArrayObject*)PyArray_FROM_OF(obj,NPY_F_CONTIGUOUS | NPY_ALIGNED | NPY_WRITEABLE);
        #else
            cont_ndarray = (PyArrayObject*)PyArray_FROM_OF(obj,NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE);
        #endif

        if(cont_ndarray == NULL || (PyObject*)cont_ndarray == Py_NotImplemented)
        {
            PyErr_SetString(PyExc_RuntimeError,"object cannot be interpreted as an array");
            return NULL;
        }
        /*
        allocating and zero initialise */
        if(PyArray_TYPE(cont_ndarray) != NPY_CFLOAT && PyArray_TYPE(cont_ndarray) != NPY_CDOUBLE && PyArray_TYPE(cont_ndarray) != NPY_CLONGDOUBLE) //real values
        {
            if (!(mxarray=mxCreateNumericArray((mwSize)PyArray_NDIM(cont_ndarray),
                            (mwSize*)PyArray_DIMS(cont_ndarray),
                            npytomx[PyArray_TYPE(cont_ndarray)],
                            mxREAL)))
            {
                return PyErr_Format(PyExc_RuntimeError, "Couldn't create mxarray: NPYTYPE:%i - mxtype:%i", PyArray_TYPE(cont_ndarray), npytomx[PyArray_TYPE(cont_ndarray)]);
            }

            nd=PyArray_DATA(cont_ndarray);
            mx=mxGetData(mxarray);
            j=PyArray_SIZE(cont_ndarray);

            memcpy(mx, nd, PyArray_NBYTES(cont_ndarray));
            /*for (i=0;i<j;i++)
                mx[i]=nd[i];*/
        

        } 
        else //else complex
        {
            if (!(mxarray=mxCreateNumericArray((mwSize)PyArray_NDIM(cont_ndarray),
                            (mwSize*)PyArray_DIMS(cont_ndarray),
                            npytomx[PyArray_TYPE(cont_ndarray)],
                            mxCOMPLEX)))
            {
                return PyErr_Format(PyExc_RuntimeError, "Couldn't create mxarray: NPYTYPE:%i - mxtype:%i", PyArray_TYPE(cont_ndarray), npytomx[PyArray_TYPE(cont_ndarray)]);
            }

        
            j=PyArray_SIZE(cont_ndarray);

            switch(PyArray_TYPE(cont_ndarray))
            {
            case NPY_CFLOAT:
                nd32=(float*)PyArray_DATA(cont_ndarray);
                mxReal32 = (float*)mxGetData(mxarray);
                mxImag32 = (float*)mxGetImagData(mxarray);
                for(i=0;i<j;i++)
                {
                    mxReal32[i]=nd32[2*i];
                    mxImag32[i]=nd32[2*i+1];
                }
                break;
            case NPY_CDOUBLE:
                nd64=(double*)PyArray_DATA(cont_ndarray);
                mxReal64 = (double*)mxGetData(mxarray);
                mxImag64 = (double*)mxGetImagData(mxarray);
                for(i=0;i<j;i++)
                {
                    mxReal64[i]=nd64[2*i];
                    mxImag64[i]=nd64[2*i+1];
                }
                break;
            case NPY_CLONGDOUBLE:
                PyErr_SetString(PyExc_RuntimeError,"matrix of type complex long double cannot be sent to Matlab");
                return NULL;
            }
                
        }
    }

    if ((engPutVariable(self->ep,name,mxarray)!=0))
    {
        PyErr_SetString(PyExc_RuntimeError,"Couldn't place string on workspace");
        return NULL;
    }
    /*
    if (ndarray!=cont_ndarray)
        Py_DECREF(cont_ndarray);
    Py_DECREF(ndarray);
        */

    Py_RETURN_NONE;
}

static PyObject * PyMatlabSessionObject_setString(PyMatlabSessionObject *self, PyObject *args)
{
    const char * name, * command_string;
    mxArray * variable;

    if(self->ep == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Matlab engine has not been started");
        return NULL;
    }

    if (!PyArg_ParseTuple(args,"ss",&name,&command_string))
        return NULL;
    if (!(variable=mxCreateString(command_string)))
    {
        PyErr_SetString(PyExc_RuntimeError,"Couldn't create mxarray");
        return NULL;
    }
    if ((engPutVariable(self->ep,name,variable)!=0))
    {
        PyErr_SetString(PyExc_RuntimeError,"Couldn't place string on workspace");
        return NULL;
    }
    mxDestroyArray(variable);
    Py_RETURN_NONE;
}

static PyObject * PyMatlabSessionObject_getString(PyMatlabSessionObject *self, PyObject *args)
{
    const char * name;
    mxArray * variable;
    PyObject *pyString;

    if(self->ep == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Matlab engine has not been started");
        return NULL;
    }

    if (!PyArg_ParseTuple(args,"s",&name))
        return NULL;

    if (!(variable = engGetVariable(self->ep,name)))
    {
        return PyErr_Format(PyExc_AttributeError,"Couldn't find '%s' at MATLAB desktop",name);
    }

    if(mxIsChar(variable))
    {
        size_t sizebuf = mxGetN(variable);
        //char *string = new char[sizebuf+1];
        char *string = (char*)malloc(sizeof(char)*(sizebuf+1));
        mxGetString(variable, string, sizebuf+1);
        //pyString = PyUnicode_FromString(string);
        pyString = PyUnicode_DecodeLatin1(string, strlen(string), NULL);
        free ((void*)string);
        return pyString;
    }
    else
    {
        return PyErr_Format(PyExc_AttributeError,"Variable '%s' is no string.",name);
    }


    Py_RETURN_NONE;
}

static PyObject * PyMatlabSessionObject_GetValue(PyMatlabSessionObject * self, PyObject *args)
{
    const char * variable;
    mxArray * mx;
    PyObject *result;
    void *real, *imag;

    if(self->ep == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Matlab engine has not been started");
        return NULL;
    }

    if (!PyArg_ParseTuple(args,"s",&variable))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument must be a string denoting the matlab variable name");
        return NULL;
    }

    if (!(mx = engGetVariable(self->ep,variable)))
    {
        return PyErr_Format(PyExc_AttributeError,"Couldn't find '%s' at MATLAB desktop",variable);
    }

    //check whether only one element is returned:
    if(mxGetM(mx) == 1 && mxGetN(mx) == 1)
    {
        if(mxIsComplex(mx))
        {
            real = mxGetData(mx);
            imag = mxGetImagData(mx);

            if(real != NULL && imag != NULL && mxGetClassID(mx) == mxSINGLE_CLASS)
            {
                return PyComplex_FromDoubles( (double)((float*)real)[0] , (double)((float*)imag)[0]);
            }
            else if(real != NULL && imag != NULL && mxGetClassID(mx) == mxDOUBLE_CLASS)
            {
                return PyComplex_FromDoubles( (double)((double*)real)[0] , (double)((double*)imag)[0]);
            }
        }
        else
        {
            real = mxGetData(mx);

            if(real != NULL)
            {
                switch(mxGetClassID(mx))
                {
                case mxINT8_CLASS:
                case mxINT16_CLASS:
                case mxINT32_CLASS:
                case mxINT64_CLASS:
                    return PyLong_FromLong((long)((long*)real)[0]);
                    break;
                case mxUINT8_CLASS:
                case mxUINT16_CLASS:
                case mxUINT32_CLASS:
                case mxUINT64_CLASS:
                    return PyLong_FromLong((unsigned long)((unsigned long*)real)[0]);
                    break;
                case mxSINGLE_CLASS:
                case mxDOUBLE_CLASS:
                    return PyFloat_FromDouble( (double)((double*)real)[0] );
                    break;
                default:
                    PyErr_SetString(PyExc_RuntimeError,"unknown datatype of matlab matrix");
                    return NULL;
                }
            }
        }
    }
    else
    {
        /*   This is how we could make it own data to avoid memory leak: (set OWN_DATA)
        *    data = malloc(sizeof(double[n*m])); 
        *    memcpy((void * )data,(void *)mxGetPr(mx),sizeof(double[n*m]));
        */
#if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
        int flag = NPY_F_CONTIGUOUS;
#else
        int flag = NPY_ARRAY_F_CONTIGUOUS;
#endif

        if (!(result=PyArray_New(&PyArray_Type,(int) mxGetNumberOfDimensions(mx), 
                        (npy_intp*) mxGetDimensions(mx), mxtonpy[mxGetClassID(mx)],
                        NULL, mxGetData(mx), NULL, flag, NULL)))
        {
            PyErr_SetString(PyExc_AttributeError,"Couldn't convert to PyArray");
            return NULL;
        }

         return (PyObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)result); //make array c-contiguous
    }


    
    /*
    mxDestroyArray(mx);
    free((void*)data);
    */
    //return result;
}

static PyObject * PyMatlabSessionObject_close(PyMatlabSessionObject * self, PyObject *args)
{
    if(self->ep)  engClose(self->ep);
    self->ep = NULL;
    Py_RETURN_NONE;
}

PyDoc_STRVAR(module_doc, "A wrapper for executing matlab commands.");

static PyMethodDef PyMatlabSessionObject_methods[] =
{
    {"run", (PyCFunction)PyMatlabSessionObject_run, METH_VARARGS, "Launch a command in MATLAB."},
    {"close", (PyCFunction)PyMatlabSessionObject_close, METH_VARARGS, "Close the open MATLAB session"},
    {"setString", (PyCFunction)PyMatlabSessionObject_setString, METH_VARARGS, "Put a string on the workspace"},
    {"getString", (PyCFunction)PyMatlabSessionObject_getString, METH_VARARGS, "Get a string-variable from the workspace"},
    {"getValue", (PyCFunction)PyMatlabSessionObject_GetValue, METH_VARARGS, "Get a variable from the workspace and return a ndarray"},
    {"setValue", (PyCFunction)PyMatlabSessionObject_setValue, METH_VARARGS, "Put a variable to the workspace"},
    {NULL,NULL,0,NULL}
};

static PyMemberDef PyMatlabSessionObject_members[] = 
{
    { NULL }
};


static PyTypeObject PyMatlabSessionObjectType =
    {
        PyVarObject_HEAD_INIT(NULL,0)
            "PyMatlabSessionObject",                             /*tp_name*/
    sizeof(PyMatlabSessionObject),                              /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyMatlabSessionObject_dealloc,                  /*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    0,                        /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash*/
    0,                                          /*tp_call*/
    0,                        /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    0,                            /*tp_doc*/
    0,                                            /*tp_traverse*/
    0,                                            /*tp_clear*/
    0,                                            /*tp_richcompare*/
    0,                                            /*tp_weaklistoffset*/
    0,                                            /*tp_iter*/
    0,                                            /*tp_iternext*/
    PyMatlabSessionObject_methods,                              /*tp_methods*/
    PyMatlabSessionObject_members,                              /*tp_members*/
    0,                                          /*tp_getset*/
    0,                                          /*tp_base*/
    0,                                          /*tp_dict*/
    0,                                          /*tp_descr_get*/
    0,                                          /*tp_descr_set*/
    0,                                          /*tp_dictoffset*/
    (initproc)PyMatlabSessionObject_init,                                          /*tp_init*/
    0, /*PyType_GenericAlloc,*/                        /*tp_alloc*/
    PyMatlabSessionObject_new,                                  /*tp_new*/
    };

//static PyMethodDef matlab_methods[] = 
//{
//    { NULL }
//};

#ifdef PYTHON3
static PyModuleDef PyMatlabSessionObject_Module = {
    PyModuleDef_HEAD_INIT, "matlab", NULL, -1, 
    NULL, NULL, NULL, NULL
};
#endif



#ifdef PYTHON3
PyMODINIT_FUNC PyInit_matlab(void)
#else
PyMODINIT_FUNC init_matlab()
#endif
{
    PyObject *module;
    
#ifdef PYTHON3
    module = PyModule_Create(&PyMatlabSessionObject_Module);
#else
    module = Py_InitModule3("MatlabSession", NULL, module_doc);
#endif

    if (module == NULL)
    {
        PyErr_SetString(PyExc_AttributeError,"fail in init: module is zero");
        goto fail;
    }

    import_array();

    if (PyType_Ready(&PyMatlabSessionObjectType) < 0)
    {
        //PyErr_SetString(PyExc_AttributeError,"fail in init: pyMatlabSessionObjectType is not ready");
        goto fail;
    }
    Py_INCREF(&PyMatlabSessionObjectType);
    PyModule_AddObject(module, "MatlabSession", (PyObject*)&PyMatlabSessionObjectType);
    
    PyModule_AddStringConstant(module, "__version__", PYMATLAB_VERSION);
    PyModule_AddStringConstant(module, "__author__", AUTHOR);

#ifdef PYTHON3
    return module;
#endif

fail:
#ifdef PYTHON3
    return NULL;
#else
    return;
#endif
}


#endif

#endif