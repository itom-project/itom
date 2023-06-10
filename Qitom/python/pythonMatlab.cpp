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

#include "pythonMatlab.h"


#include "pythonCommon.h"

#if ITOM_PYTHONMATLAB == 1

#define PYMATLAB_VERSION "0.2"
#define AUTHOR "Joakim Moller, Marc Gronle (Python 3)"

#define CREATE_MATLABDEF(retArg, funcName, ...) \
    typedef retArg (*f_##funcName)(##__VA_ARGS__); \
    f_##funcName func_##funcName = NULL;

#define LOAD_MATLABDEF(funcName, libName) func_##funcName = (f_##funcName)##libName.resolve(#funcName); \
    if (!func_##funcName) { \
    retVal += ito::RetVal::format(ito::retError, 0, "the required function " #funcName " could not be resolved: %s", ##libName.errorString().toLatin1().data()); \
    }

namespace ito
{
    //functions from engine.h
    CREATE_MATLABDEF(Engine*, engOpen, const char*)
    CREATE_MATLABDEF(int, engClose, Engine*)
    CREATE_MATLABDEF(int, engPutVariable, Engine*, const char*, const mxArray*)
    CREATE_MATLABDEF(mxArray*, engGetVariable, Engine*, const char*)
    CREATE_MATLABDEF(int, engEvalString, Engine*, const char*)
    CREATE_MATLABDEF(int, engOutputBuffer, Engine*, char*, int)

    //functions from matrix.h
    CREATE_MATLABDEF(char*, mxArrayToString, const mxArray *)
    CREATE_MATLABDEF(void*, mxGetImagData, const mxArray *)
    CREATE_MATLABDEF(size_t, mxGetN, const mxArray *)
    CREATE_MATLABDEF(size_t, mxGetM, const mxArray *)
    CREATE_MATLABDEF(void*, mxGetData, const mxArray *)
    CREATE_MATLABDEF(mxArray*, mxCreateNumericArray_730, mwSize, const mwSize *, mxClassID, mxComplexity)
    CREATE_MATLABDEF(void, mxDestroyArray, mxArray *)
    CREATE_MATLABDEF(mxArray*, mxCreateString, const char *)
    CREATE_MATLABDEF(int, mxGetString_730, const mxArray *, char*, mwSize)
    CREATE_MATLABDEF(bool, mxIsChar, const mxArray *)
    CREATE_MATLABDEF(mwSize, mxGetNumberOfDimensions_730, const mxArray *)
    CREATE_MATLABDEF(const mwSize*, mxGetDimensions_730, const mxArray *)
    CREATE_MATLABDEF(mxClassID, mxGetClassID, const mxArray *)
    CREATE_MATLABDEF(bool, mxIsComplex, const mxArray *)


    bool PythonMatlab::initialized = false;
    QLibrary PythonMatlab::engineLibrary;
    QLibrary PythonMatlab::mxLibrary;

    /* This wraps a command call to the MATLAB interpretor */
const char * function_wrap = "try\n\t%s;\ncatch err\n\tpymatlaberrstring = sprintf('Error: %%s with message: %%s\\n',err.identifier,err.message);\n\tfor i = 1:length(err.stack)\n\t\tpymatlaberrstring = sprintf('%%sError: in fuction %%s in file %%s line %%i\\n',pymatlaberrstring,err.stack(i,1).name,err.stack(i,1).file,err.stack(i,1).line);\n\tend\nend\nif exist('pymatlaberrstring','var')==0\n\tpymatlaberrstring='';\nend";

int mxtonpy[17] = {
    NPY_USERDEF,    /* mxUNKNOWN_CLASS (0) */
    NPY_USERDEF,    /* mxCELL_CLASS (1) */
    NPY_USERDEF,     /* mxSTRUCT_CLASS (2) */
    NPY_BOOL,     /* mxLOGICAL_CLASS (3) */
    NPY_STRING,     /* mxCHAR_CLASS (4) */
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

mxClassID npytomx[27]={ mxLOGICAL_CLASS, /*NPY_BOOL (0)*/
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
                        mxUNKNOWN_CLASS, /*NPY_HALF (23)*/
                        mxUNKNOWN_CLASS, /*NPY_NTYPES (24)*/
                        mxUNKNOWN_CLASS, /*NPY_NOTYPE (25)*/
                        mxUNKNOWN_CLASS, /*NPY_CHAR (26)*/
};

//-------------------------------------------------------------------------------------------------------
/*static*/ ito::RetVal PythonMatlab::loadLibrary()
{
    ito::RetVal retVal;

    if (!initialized)
    {
#if WIN32
        engineLibrary.setFileName("libeng.dll");
        mxLibrary.setFileName("libmx.dll");
#else
        engineLibrary.setFileName("libeng.so");
        mxLibrary.setFileName("libmx.so");
#endif

        if (engineLibrary.load())
        {
            //func_engOpen = (f_engOpen)engineLibrary.resolve("engOpen");
            LOAD_MATLABDEF(engOpen, engineLibrary)
            LOAD_MATLABDEF(engClose, engineLibrary)
            LOAD_MATLABDEF(engPutVariable, engineLibrary)
            LOAD_MATLABDEF(engGetVariable, engineLibrary)
            LOAD_MATLABDEF(engEvalString, engineLibrary)
            LOAD_MATLABDEF(engOutputBuffer, engineLibrary)
        }
        else
        {
            retVal += ito::RetVal::format(ito::retError, 0, "error loading matlab engine: %s", engineLibrary.errorString().toLatin1().data());
        }

        if (mxLibrary.load())
        {
            LOAD_MATLABDEF(mxArrayToString, mxLibrary)
            LOAD_MATLABDEF(mxGetImagData, mxLibrary)
            LOAD_MATLABDEF(mxGetN, mxLibrary)
            LOAD_MATLABDEF(mxGetM, mxLibrary)
            LOAD_MATLABDEF(mxGetData, mxLibrary)
            LOAD_MATLABDEF(mxCreateNumericArray_730, mxLibrary)
            LOAD_MATLABDEF(mxDestroyArray, mxLibrary)
            LOAD_MATLABDEF(mxCreateString, mxLibrary)
            LOAD_MATLABDEF(mxGetString_730, mxLibrary)
            LOAD_MATLABDEF(mxIsChar, mxLibrary)
            LOAD_MATLABDEF(mxGetNumberOfDimensions_730, mxLibrary)
            LOAD_MATLABDEF(mxGetDimensions_730, mxLibrary)
            LOAD_MATLABDEF(mxGetClassID, mxLibrary)
            LOAD_MATLABDEF(mxIsComplex, mxLibrary)
        }
        else
        {
            retVal += ito::RetVal::format(ito::retError, 0, "error loading matlab engine: %s", mxLibrary.errorString().toLatin1().data());
        }

        if (!retVal.containsError())
        {
            initialized = true;
        }
    }

    return retVal;
}


//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonMatlab::PyMatlabSessionObject_new(PyTypeObject *type, PyObject *args, PyObject * kwds)
{
    if (!initialized)
    {
        ito::RetVal init = loadLibrary();
        if (!PythonCommon::transformRetValToPyException(init))
        {
            return NULL;
        }
    }

    PyMatlabSessionObject *self;
    self = (PyMatlabSessionObject *) type->tp_alloc(type,0);
    if (self!=NULL)
    {
        self->ep = NULL;
    }
    return (PyObject *) self;
}

//-------------------------------------------------------------------------------------------------------
/*static*/ int PythonMatlab::PyMatlabSessionObject_init(PyMatlabSessionObject *self, PyObject *args, PyObject *kwds)
{
    int status;
    char *startstr=NULL;
    if (!PyArg_ParseTuple(args,"|s",&startstr))
        return -1;
    if (!(self->ep = func_engOpen(startstr)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Can't start MATLAB engine. Maybe you have to properly register Matlab in the registry.");
        return -1;
    }
    status = func_engOutputBuffer(self->ep,NULL,0);
    return 0;
}

//-------------------------------------------------------------------------------------------------------
/*static*/ void PythonMatlab::PyMatlabSessionObject_dealloc(PyMatlabSessionObject *self)
{
    if(self->ep)
    {
        func_engClose(self->ep);
    }
    self->ob_base.ob_type->tp_free((PyObject*)self);
}

//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonMatlab::PyMatlabSessionObject_run(PyMatlabSessionObject *self, PyObject *args)
{
    char * stringarg;
    char * command;
    int status;
    const mxArray * mxresult;

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
    if (func_engEvalString(self->ep,command)!=0)
    {
        PyObject* error = PyErr_Format(PyExc_RuntimeError, "Was not able to evaluate command: %s", command);
        free((void*)command);
        return error;
    }
    if ((mxresult = func_engGetVariable(self->ep,"pymatlaberrstring"))==NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "can't get internal variable: pymatlaberrstring");
        free((void*)command);
        return NULL;
    }
    if (strcmp( func_mxArrayToString(mxresult),"")!=0)
    {
        /*make sure 'pymatlaberrstring' is empty or not exist until next call*/
        status = func_engEvalString(self->ep,"clear pymatlaberrstring");
        free((void*)command);
        return PyErr_Format(PyExc_RuntimeError,"Error from Matlab: %s end.", func_mxArrayToString(mxresult));
    }
    free((void*)command);
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject * PythonMatlab::PyMatlabSessionObject_setValue(PyMatlabSessionObject *self, PyObject *args)
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
        if (!(mxarray = func_mxCreateNumericArray_730(1, dims, mxINT64_CLASS , mxREAL)))
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't create mxarray.");
            return NULL;
        }

        ((long*)func_mxGetData(mxarray))[0] = PyLong_AsLong(obj);
    }
    else if(PyFloat_Check(obj))
    {
        if (!(mxarray = func_mxCreateNumericArray_730(1, dims, mxDOUBLE_CLASS , mxREAL)))
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't create mxarray.");
            return NULL;
        }

        ((double*)func_mxGetData(mxarray))[0] = PyFloat_AsDouble(obj);
    }
    else if(PyComplex_Check(obj))
    {
        if (!(mxarray = func_mxCreateNumericArray_730(1, dims, mxDOUBLE_CLASS , mxCOMPLEX)))
        {
            PyErr_SetString(PyExc_RuntimeError, "Couldn't create mxarray.");
            return NULL;
        }

        ((double*)func_mxGetData(mxarray))[0] = PyComplex_RealAsDouble(obj);
        ((double*)func_mxGetImagData(mxarray))[0] = PyComplex_ImagAsDouble(obj);
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
            if (!(mxarray = func_mxCreateNumericArray_730((mwSize)PyArray_NDIM(cont_ndarray),
                            (mwSize*)PyArray_DIMS(cont_ndarray),
                            npytomx[PyArray_TYPE(cont_ndarray)],
                            mxREAL)))
            {
                return PyErr_Format(PyExc_RuntimeError, "Couldn't create mxarray: NPYTYPE:%i - mxtype:%i", PyArray_TYPE(cont_ndarray), npytomx[PyArray_TYPE(cont_ndarray)]);
            }

            nd = PyArray_DATA(cont_ndarray);
            mx = func_mxGetData(mxarray);
            j = PyArray_SIZE(cont_ndarray);

            memcpy(mx, nd, PyArray_NBYTES(cont_ndarray));
            /*for (i=0;i<j;i++)
                mx[i]=nd[i];*/


        }
        else //else complex
        {
            if (!(mxarray = func_mxCreateNumericArray_730((mwSize)PyArray_NDIM(cont_ndarray),
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
                mxReal32 = (float*)func_mxGetData(mxarray);
                mxImag32 = (float*)func_mxGetImagData(mxarray);
                for(i=0;i<j;i++)
                {
                    mxReal32[i]=nd32[2*i];
                    mxImag32[i]=nd32[2*i+1];
                }
                break;
            case NPY_CDOUBLE:
                nd64=(double*)PyArray_DATA(cont_ndarray);
                mxReal64 = (double*)func_mxGetData(mxarray);
                mxImag64 = (double*)func_mxGetImagData(mxarray);
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

    if ((func_engPutVariable(self->ep,name,mxarray)!=0))
    {
        PyErr_SetString(PyExc_RuntimeError,"Couldn't place string on workspace");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject * PythonMatlab::PyMatlabSessionObject_setString(PyMatlabSessionObject *self, PyObject *args)
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
    if (!(variable = func_mxCreateString(command_string)))
    {
        PyErr_SetString(PyExc_RuntimeError,"Couldn't create mxarray");
        return NULL;
    }
    if ((func_engPutVariable(self->ep,name,variable)!=0))
    {
        PyErr_SetString(PyExc_RuntimeError,"Couldn't place string on workspace");
        return NULL;
    }
    func_mxDestroyArray(variable);
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject * PythonMatlab::PyMatlabSessionObject_getString(PyMatlabSessionObject *self, PyObject *args)
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

    if (!(variable = func_engGetVariable(self->ep,name)))
    {
        return PyErr_Format(PyExc_AttributeError,"Couldn't find '%s' at MATLAB desktop",name);
    }

    if(func_mxIsChar(variable))
    {
        size_t sizebuf = func_mxGetN(variable);
        //char *string = new char[sizebuf+1];
        char *string = (char*)malloc(sizeof(char)*(sizebuf+1));
        func_mxGetString_730(variable, string, sizebuf+1);
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

//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject * PythonMatlab::PyMatlabSessionObject_GetValue(PyMatlabSessionObject * self, PyObject *args)
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

    if (!(mx = func_engGetVariable(self->ep,variable)))
    {
        return PyErr_Format(PyExc_AttributeError,"Couldn't find '%s' at MATLAB desktop",variable);
    }

    //check whether only one element is returned:
    if(func_mxGetM(mx) == 1 && func_mxGetN(mx) == 1)
    {
        if(func_mxIsComplex(mx))
        {
            real = func_mxGetData(mx);
            imag = func_mxGetImagData(mx);

            if(real != NULL && imag != NULL && func_mxGetClassID(mx) == mxSINGLE_CLASS)
            {
                return PyComplex_FromDoubles( (double)((float*)real)[0] , (double)((float*)imag)[0]);
            }
            else if(real != NULL && imag != NULL && func_mxGetClassID(mx) == mxDOUBLE_CLASS)
            {
                return PyComplex_FromDoubles( (double)((double*)real)[0] , (double)((double*)imag)[0]);
            }
        }
        else
        {
            real = func_mxGetData(mx);

            if(real != NULL)
            {
                switch(func_mxGetClassID(mx))
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
            else
            {
                PyErr_SetString(PyExc_RuntimeError,"error parsing matlab matrix");
                return NULL;
            }
        }
    }
    else
    {
        /*   This is how we could make it own data to avoid memory leak: (set OWN_DATA)
        *    data = malloc(sizeof(double[n*m]));
        *    memcpy((void * )data,(void *)func_mxGetPr(mx),sizeof(double[n*m]));
        */
#if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
        int flag = NPY_F_CONTIGUOUS;
#else
        int flag = NPY_ARRAY_F_CONTIGUOUS;
#endif

        mwSize numDim = func_mxGetNumberOfDimensions_730(mx);
        const mwSize* dims = func_mxGetDimensions_730(mx);
        real = func_mxGetData(mx);

        if (func_mxIsComplex(mx))
        {
            imag = func_mxGetImagData(mx);
            mxClassID classID = func_mxGetClassID(mx);

            if (!(result = PyArray_New(&PyArray_Type,(int) numDim,
                            (npy_intp*) dims, classID == mxSINGLE_CLASS ? NPY_CFLOAT : NPY_CDOUBLE,
                            NULL, NULL, NULL, flag, NULL)))
            {
                PyErr_SetString(PyExc_AttributeError,"Couldn't convert to PyArray");
                return NULL;
            }
            else
            {
                if (classID == mxSINGLE_CLASS)
                {
                    ito::float32 *ptr = (ito::float32*)PyArray_DATA((PyArrayObject*)result);
                    ito::float32 *realf = (ito::float32*)real;
                    ito::float32 *imagf = (ito::float32*)imag;
                    for (npy_intp s = 0; s < PyArray_SIZE((PyArrayObject*)result); ++s)
                    {
                        (*ptr++) = (*realf++);
                        (*ptr++) = (*imagf++);
                    }
                }
                else
                {
                    ito::float64 *ptr = (ito::float64*)PyArray_DATA((PyArrayObject*)result);
                    ito::float64 *realf = (ito::float64*)real;
                    ito::float64 *imagf = (ito::float64*)imag;
                    for (npy_intp s = 0; s < PyArray_SIZE((PyArrayObject*)result); ++s)
                    {
                        (*ptr++) = (*realf++);
                        (*ptr++) = (*imagf++);
                    }
                }
            }
        }
        else
        {
            //result share the values with the matlab data, however the PyArray_GETCONTIGUOUS method below will make a deep copy!
            if (!(result = PyArray_New(&PyArray_Type,(int) numDim,
                            (npy_intp*) dims, mxtonpy[func_mxGetClassID(mx)],
                            NULL, real, NULL, flag, NULL)))
            {
                PyErr_SetString(PyExc_AttributeError,"Couldn't convert to PyArray");
                return NULL;
            }
        }

        return (PyObject*)PyArray_GETCONTIGUOUS((PyArrayObject*)result); //make array c-contiguous
    }

    PyErr_SetString(PyExc_RuntimeError,"error parsing matlab matrix");
    return NULL;
}

//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject * PythonMatlab::PyMatlabSessionObject_close(PyMatlabSessionObject * self, PyObject *args)
{
    if(self->ep)  func_engClose(self->ep);
    self->ep = NULL;
    Py_RETURN_NONE;
}


PyDoc_STRVAR(module_doc, "A wrapper for executing matlab commands.");

//-------------------------------------------------------------------------------------------------------
/*static*/ PyMethodDef PythonMatlab::PyMatlabSessionObject_methods[] =
{
    {"run", (PyCFunction)PyMatlabSessionObject_run, METH_VARARGS, "Launch a command in MATLAB."},
    {"close", (PyCFunction)PyMatlabSessionObject_close, METH_VARARGS, "Close the open MATLAB session"},
    {"setString", (PyCFunction)PyMatlabSessionObject_setString, METH_VARARGS, "Put a string on the workspace"},
    {"getString", (PyCFunction)PyMatlabSessionObject_getString, METH_VARARGS, "Get a string-variable from the workspace"},
    {"getValue", (PyCFunction)PyMatlabSessionObject_GetValue, METH_VARARGS, "Get a variable from the workspace and return a ndarray"},
    {"setValue", (PyCFunction)PyMatlabSessionObject_setValue, METH_VARARGS, "Put a variable to the workspace"},
    {NULL,NULL,0,NULL}
};

//-------------------------------------------------------------------------------------------------------
/*static*/ PyMemberDef PythonMatlab::PyMatlabSessionObject_members[] = {
    {NULL},
};


//-------------------------------------------------------------------------------------------------------
/*static*/ PyTypeObject PythonMatlab::PyMatlabSessionObjectType =
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
    PythonMatlab::PyMatlabSessionObject_methods,                              /*tp_methods*/
    PythonMatlab::PyMatlabSessionObject_members,                              /*tp_members*/
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

/*static*/ PyModuleDef PythonMatlab::PyMatlabSessionObject_Module = {
    PyModuleDef_HEAD_INIT, "matlab", NULL, -1,
    NULL, NULL, NULL, NULL
};


//-------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonMatlab::PyInit_matlab(void)
{
    PyObject *module;

    module = PyModule_Create(&PyMatlabSessionObject_Module);

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


    return module;

fail:

    return NULL;
}

}

#endif
