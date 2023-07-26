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

#include "pythonDataObject.h"
#include "pythonEngineInc.h"

#include "numpy/arrayscalars.h"

#include "structmember.h"

#include "../global.h"

#include "../common/helperDatetime.h"
#include "../common/shapeDObject.h"
#include "pythonCommon.h"
#include "pythonDateTime.h"
#include "pythonRgba.h"
#include "pythonShape.h"

#include "dataObjectFuncs.h"
#include "pythonQtConversion.h"

#include <qdatetime.h>

#include <memory>
#include <stdexcept>

#define PROTOCOL_STR_LENGTH 128

namespace ito {
template<class T>
std::unique_ptr<T> make_unique(std::size_t size)
{
    return std::unique_ptr<T>(new typename std::remove_extent<T>::type[size]());
}
template <typename... Args> std::string string_format(const std::string& format, Args... args)
{
    int size_s = _snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf = make_unique<char[]>(size);
    _snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

//-------------------------------------------------------------------------------------
void PythonDataObject::PyDataObject_dealloc(PyDataObject* self)
{
    if (self->dataObject != NULL)
    {
        DELETE_AND_SET_NULL(self->dataObject);
    }

    Py_XDECREF(self->base); // this will free another pyobject (e.g. numpy array), with which this
                            // data object shared its data (base != NULL if owndata=0)

    Py_TYPE(self)->tp_free((PyObject*)self);
};

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_new(
    PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyDataObject* self = (PyDataObject*)type->tp_alloc(type, 0);

    if (self != NULL)
    {
        self->dataObject = NULL;
        self->base = NULL;
    }

    return (PyObject*)self;
};

//-------------------------------------------------------------------------------------
//! brief description
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
PyDoc_STRVAR(
    dataObjectInit_doc,
    "dataObject(dims = [], dtype = \"uint8\", continuous = 0, data = None) -> dataObject \n\
\n\
Creates a new n-dimensional dataObject array. \n\
\n\
The :class:`dataObject` represents a multidimensional array of fixed-size items \n\
(integer, floating-point or complex values) and contains further, optional, meta \n\
information, like units, axis descriptions, scalings, general tags, ... \n\
Recently the following data types (dtype) are supported: \n\
\n\
* Integer (int8, uint8, int16, uint16, int32),\n\
* Floating point (float32, float64 (=> double)),\n\
* Complex (complex64 (2x float32), complex128 (2x float64)).\n\
* Color (rgba32 (uint32 or uint[4] containing the four 8bit values [R, G, B, Alpha])).\n\
* Datetime and Timedelta (datetime, timedelta). \n\
\n\
Arrays can also be constructed using some of the static pre-initialization methods \n\
:meth:`zeros`, :meth:`ones`, :meth:`rand` or :meth:`randN`  \n\
(refer to the See Also section below). \n\
\n\
Parameters \n\
---------- \n\
dims : sequence of int, optional \n\
    ``dims`` is a list or tuple indicating the size of each dimension. The length \n\
    of this sequence defines the dimension of this dataObject. As an example, \n\
    ``dims = [2, 3]`` creates a two-dimensional dataObject with two rows and three columns. \n\
    If ``dims`` is not given, an empty data object is created. \n\
dtype : str, optional \n\
    Data type of each element in the array. Possible values are: \n\
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'float32', 'float64', 'complex64', \n\
    'complex128', 'rgba32', 'datetime' or 'timedelta'. \n\
continuous : int, optional \n\
    The last two dimensions of a dataObject are always stored as continuous junk of memory, \n\
    denoted as plane. If ``continuous`` is set to ``1``, even a dataObject with a dimension \n\
    ``n > 2`` will allocate one big block of memory and continuously stores the matrix data \n\
    there. \n\
    If ``continuous`` is 0, different junks of memory are allocated for each plane, the planes \n\
    are referenced by means of an index vector. This is recommended for large arrays, since \n\
    the operating system might get trouble allocated one very big continuous junk of memory, \n\
    instead of multiple smaller ones. \n\
data : int or float or complex or rgba or datetime.datetime or datetime.timedelta or sequence of int or sequence of float or sequence of complex or dataObject or np.ndarray, optional \n\
    If ``data`` is a single value, all values in the dataObject are set to this single value. \n\
    Else, the sequence or array-like object must have the same number of values than \n\
    the data object. These values will then be assigned to the new data object (filled row by row).\n\
\n\
Notes \n\
----- \n\
The :class:`itom.dataObject` is a direct wrapper for the underlying C++ class *dataObject*. \n\
This array class mainly is based on the class *Mat* of the computer vision library (OpenCV). \n\
\n\
In order to handle huge matrices, the data object can divide one array into chunks in memory.\n\
Each subpart (called matrix-plane) is two-dimensional and covers data of the last two dimensions.\n\
In C++-context each of these matrix-planes is of type cv::Mat_<type> and can be used with \n\
every operator given by the openCV-framework (version 2.3.1 or higher).\n\
\n\
The dimensions of the matrix are structured descending. So if we assume to have a n-dimensional \n\
matrix ``A``, where each dimension has its size s_i, the dimensions order is n, .., z, y, x and \n\
the corresponding sizes of ``A`` are [s_n, s_(n-1),  s_(n-2), ..., s_y, s_x].\n\
\n\
In order to make the data object compatible to continuously organized data structures, like \n\
numpy-arrays, it is also possible to have all matrix-planes in one data-block in memory \n\
(not recommended for huge matrices). Nevertheless, the indicated data structure with the \n\
two-dimensional sub-matrix-planes is still existing. The data organization is equal to the \n\
one of openCV, hence, two-dimensional matrices are stored row-by-row (C-style)...\n\
\n\
In addition to OpenCV, itom.dataObject supports complex valued data types for all operators and methods. \n\
\n\
Warning 'uint32' is currently not available, since it is not fully supported by the underlying OpenCV matrices.\n\
\n\
**Deep Copy, Shallow Copy and ROI** \n\
\n\
It is possible to set a n-dimensional region of interest (ROI) to each matrix, the virtual dimensions,\n\
which will be delivered if the user asks for the matrix size.\n\
To avoid copy operations where possible a simple =_Operator will also make a shallow copy of the object.\n\
Shallow copies share the same data (elements and meta data) with the original object, hence manipulations of one object will affect the\n\
original object and all shallow copies.\n\
\n\
The opposite a deep copy of a dataObject (by sourceObject.copy()) creates a complete mew matrix with own meta data object.\n\
\n\
Example::\n\
    \n\
    #Create an object \n\
    dObj = dataObject([5, 10, 10], 'int8')\n\
    \n\
    # Make a shallow copy \n\
    dObjShallow = dObj \n\
    \n\
    # Make a shallow copy on ROI\n\
    dObjROI = dObj[1, :, :] \n\
    \n\
    # Set the value of element [1, 0, 0] to 0\n\
    dObj[1, 0, 0] = 0\n\
    \n\
    # Make a deep copy of the dObjROI\n\
    dObjROICopy = dObjROI.copy()\n\
    \n\
    # Set the value of dObjROICopy element [0, 0, 0] to 127 without effecting other objects\n\
    dObjROICopy[0, 0, 0] = 127\n\
\n\
**Constructor** \n\
\n\
The function dataObject([dims [, dtype='uint8'[, continuous = 0][, data = valueOrSequence]]])\n\
creates a new itom-dataObject filled with undefined data.\n\
If no parameters are given, an uninitilized DataObject (dims = 0, no sizes) is created.\n\
\n\
As second possibility you can also use the copy-constructor 'dataObject(anyArray : Union[dataObject, np.ndarray], dtype : str = '', continuous : int = 0)', \n\
where 'anyArray' must be any array-like structure which can be parsed by the numpy-interface. If a dtype is given or if continuous is 1, \n\
the new data object will be a type-casted (and / or continuous) copy of 'anyArray'.\n\
\n\
See Also \n\
-------- \n\
ones : Static method to construct a data object filled with ones. \n\
zeros : Static method to construct a data object filled with zeros. \n\
nans : Static method to construct a data object (float or complex only) with NaNs. \n\
rand : Static method to construct a randomly filled data object (uniform distribution). \n\
randN : Static method to construct a randomly filled data object (gaussian distribution).");
int PythonDataObject::PyDataObject_init(PyDataObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t lengthArgs = args ? PyTuple_Size(args) : 0;
    Py_ssize_t lengthKwds = kwds ? PyDict_Size(kwds) : 0;

    PyObject* copyObject = NULL;
    const char* cpykwlist[] = {"object", "dtype", "continuous", NULL};
    const char* typeName = ""; // do not place the default value here: this will be done
                               // individually by the different sub-cases
    int typeno = 0;
    unsigned char continuous = 0;

    RetVal retValue(retOk);
    bool done = false;

    // clear base (if available)
    PyDataObject_SetBase(self, NULL);

    // clear existing dataObject (if exists)
    DELETE_AND_SET_NULL(self->dataObject);

    // The order of argument check is:
    /*
    1. no arguments --> create empty dataObject
    2. basic copy constructor --> first argument is another dataObject, followed by optional type
    and/or continuous flag
    3. general copy constructor --> first argument is a compatible np.array, followed by optional
    type and/or continuous flag
    4. generation from given shape, optional dtype, continuous flag and data
    */

    // 1. check for call without arguments
    if ((lengthArgs + lengthKwds) == 0 && !done)
    {
        DELETE_AND_SET_NULL(self->dataObject);
        self->dataObject = new ito::DataObject();
        self->base = NULL;
        retValue += RetVal(retOk);
        done = true;
    }

    // 2.  check for copy constructor of type PyDataObject (same type)
    if (!retValue.containsError())
        PyErr_Clear();
    // todo: default of type and continuous must be the same than of rhs object! not uint8 and 0!
    if (!done &&
        PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!|sb",
            const_cast<char**>(cpykwlist),
            &PyDataObjectType,
            &copyObject,
            &typeName,
            &continuous))
    {
        PyDataObject* rhsDataObj = (PyDataObject*)(copyObject);

        if (strlen(typeName) > 0)
        {
            typeno = dObjTypeFromName(typeName);
        }
        else
        {
            // same type than given copyObject
            typeno = rhsDataObj->dataObject->getType();
        }

        if (typeno >= 0)
        {
            bool differentType = (typeno != rhsDataObj->dataObject->getType());
            DELETE_AND_SET_NULL(self->dataObject);

            if (differentType)
            {
                self->dataObject = new ito::DataObject();
                retValue += rhsDataObj->dataObject->convertTo(*self->dataObject, typeno);
                if (retValue.containsError())
                {
                    PyErr_SetString(PyExc_RuntimeError, retValue.errorMessage());
                }
                else
                {
                    if (continuous > 0 && self->dataObject->getContinuous() == 0)
                    {
                        // try to make this object continuous
                        ito::DataObject tempObj = ito::makeContinuous(*(self->dataObject));
                        *(self->dataObject) = tempObj;
                    }

                    done = true;
                }
            }
            else
            {
                self->dataObject = new ito::DataObject(*rhsDataObj->dataObject);

                if (continuous > 0 && self->dataObject->getContinuous() == 0)
                {
                    // try to make this object continuous. The continous object cannot share any
                    // memory with any base objects, since it has to be reallocated as independent
                    // object
                    ito::DataObject tempObj = ito::makeContinuous(*(self->dataObject));
                    *(self->dataObject) = tempObj;
                }
                else
                {
                    PyDataObject_SetBase(self, rhsDataObj->base);
                }

                done = true;
            }
        }
        else
        {
            retValue += ito::retError;
            PyErr_Format(PyExc_ValueError, "Invalid dtype '%s'.", typeName);
        }
    }

    if (!retValue.containsError())
    {
        // the previous PyArg_ParseTupleAndKeywords returned false ans et an error. Delete this
        // error and try to go on.
        PyErr_Clear();
    }

    // 2. check for argument object : np.ndarray, dtype : str = "", continuous : int = 1 (continuous
    // has no impact)
    if (!done)
    {
        int result = PyDataObj_CreateFromNpNdArrayAndType(self, args, kwds, false);

        if (result == 0)
        {
            done = true;
        }
        else if (result == -1)
        {
            // general error: Python error is set and should be used
            retValue = ito::retError;
            done = true;
        }
        else
        {
            // argument parse error: jump into the general error message block
            done = false;
            retValue = ito::retOk;
            PyErr_Clear();
        }
    }

    if (!retValue.containsError())
    {
        // the previous PyArg_ParseTupleAndKeywords returned false ans et an error. Delete this
        // error and try to go on.
        PyErr_Clear();
    }

    // 3. check for argument: list/tuple/seq.(int size1, int size2,...,int sizeLast)[,
    // dtype='typename'][, continuous=[0|1]
    if (!done)
    {
        int result = PyDataObj_CreateFromShapeTypeData(self, args, kwds);

        if (result == 0)
        {
            done = true;
        }
        else if (result == -1)
        {
            // general error: Python error is set and should be used
            retValue = ito::retError;
            done = true;
        }
        else
        {
            // argument parse error: jump into the general error message block
            done = false;
            retValue = ito::retOk;
            PyErr_Clear();
        }
    }

    if (!done && retValue.containsError())
    {
        PyErr_SetString(
            PyExc_TypeError,
            "Required arguments are: No arguments OR obj : Union[dataObject, np.ndarray], dtype : "
            "str = 'uint8', continuous : int = 0 OR shape : Sequence[int], dtype : str = 'uint8', "
            "continuous : int = 0, data : Union[None,int,float,complex,list,tuple] = None");
    }
    else if (!done && !retValue.containsError())
    {
        PyErr_SetString(PyExc_TypeError, "number or arguments are invalid.");
    }
    else if (done)
    {
        return retValue.containsError() ? -1 : 0;
    }

    return -1;
};

//-------------------------------------------------------------------------------------
PythonDataObject::PyDataObjectTypes PythonDataObject::PyDataObject_types[] = {
    {"int8", tInt8},
    {"uint8", tUInt8},
    {"int16", tInt16},
    {"uint16", tUInt16},
    {"int32", tInt32},
    {"uint32", tUInt32},
    {"float32", tFloat32},
    {"float64", tFloat64},
    {"complex64", tComplex64},
    {"complex128", tComplex128},
    {"rgba32", tRGBA32},
    {"datetime", tDateTime},
    {"timedelta", tTimeDelta}};

//-------------------------------------------------------------------------------------
int PythonDataObject::dObjTypeFromName(const char* name)
{
    int length = numDataTypes();
    int i;

    for (i = 0; i < length; i++)
    {
        if (!strcmp(name, PyDataObject_types[i].name))
        {
            return PyDataObject_types[i].typeno;
        }
    }

    return -1;
}

//-------------------------------------------------------------------------------------
const char* PythonDataObject::typeNumberToName(int typeno)
{
    if (typeno < 0 || typeno >= numDataTypes())
    {
        return nullptr;
    }
    else
    {
        return PyDataObject_types[typeno].name;
    }
}

//-------------------------------------------------------------------------------------
int PythonDataObject::numDataTypes()
{
    int length = sizeof(PyDataObject_types) / sizeof(PyDataObject_types[0]);
    return length;
}

//-------------------------------------------------------------------------------------
/*
return 0 if dataObject could be created. self->dataObject is allocated then.
return -1 in case of a general error, Python error message is set
return -2 if args / kwds cannot be parsed, Python error message is set, too
*/
int PythonDataObject::PyDataObj_CreateFromShapeTypeData(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    PyObject* data = nullptr;
    const char* kwlist[] = {"dims", "dtype", "continuous", "data", nullptr};
    PyObject* dimList = nullptr;
    const char* typeName = "uint8\0";
    unsigned char continuous = 0;
    Py_ssize_t dims = 0;
    int intDims = 0;
    ito::RetVal retVal;
    int* sizes = nullptr;
    int tempSizes = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O|sbO",
            const_cast<char**>(kwlist),
            &dimList,
            &typeName,
            &continuous,
            &data))
    {
        return -2; // Python error is set, too
    }

    int typeno = dObjTypeFromName(typeName);

    if (typeno < 0)
    {
        PyErr_Format(PyExc_TypeError, "dtype name '%s' is unknown.", typeName);
        return -1;
    }
    else if (!PySequence_Check(dimList))
    {
        PyErr_SetString(
            PyExc_TypeError,
            "a non-empty list or tuple of integer is expected for the parameter 'shape'.");
        return -1;
    }
    else
    {
        dims = PySequence_Size(dimList);

        if (dims < 0 || dims > 255)
        {
            PyErr_SetString(PyExc_TypeError, "Number of dimensions must be in range [1,255].");
            return -1;
        }

        intDims = Py_SAFE_DOWNCAST(dims, Py_ssize_t, int);

        unsigned char dimensions = static_cast<unsigned char>(intDims);
        sizes = new int[intDims];
        for (int i = 0; i < intDims; i++)
        {
            sizes[i] = 0;
        }

        int totalElems = 1;
        PyObject* dimListItem = NULL;
        bool ok;

        // try to parse list to values of unsigned int
        for (Py_ssize_t i = 0; i < dims; i++)
        {
            dimListItem = PySequence_GetItem(dimList, i); // new reference
            tempSizes = PythonQtConversion::PyObjGetInt(dimListItem, true, ok);
            if (!ok)
            {
                PyErr_Format(
                    PyExc_TypeError,
                    "Size of %d. dimension is no integer number or exceeds the valid value range.",
                    i + 1);
                retVal += ito::retError;
                break;
            }
            else if (tempSizes <= 0)
            {
                PyErr_Format(
                    PyExc_TypeError, "Size of %d. dimension must be a positive number.", i + 1);
                retVal += ito::retError;
                break;
            }

            Py_XDECREF(dimListItem);
            sizes[i] = tempSizes;
            totalElems *= tempSizes;
        }

        // pre-check data
        if (!retVal.containsError() && data)
        {
            if (PySequence_Check(data) && PySequence_Length(data) != totalElems)
            {
                PyErr_SetString(
                    PyExc_TypeError,
                    "The sequence provided by data must have the same length than the total number "
                    "of elements of the data object.");
                retVal += RetVal(retError);
            }
            else if (
                !PySequence_Check(data) && !PyFloat_Check(data) && !PyLong_Check(data) &&
                !PyComplex_Check(data) && !PyRgba_Check(data) &&
                !PythonDateTime::PyDateTime_CheckExt(data) &&
                !PythonDateTime::PyTimeDelta_CheckExt(data))
            {
                PyErr_SetString(
                    PyExc_TypeError,
                    "The single value provided by data must be a numeric type (int, float, "
                    "complex) or a scalar value of type rgba, datetime.datetime or "
                    "datetime.timedelta.");
                retVal += RetVal(retError);
            }
        }

        if (!retVal.containsError())
        {
            DELETE_AND_SET_NULL(self->dataObject);
            try
            {
                self->dataObject = new ito::DataObject(dimensions, sizes, typeno, continuous);
            }
            catch (cv::Exception& exc)
            {
                PyErr_Format(
                    PyExc_RuntimeError, "failed to create data object: %s", (exc.err).c_str());
                self->dataObject = nullptr;
                retVal += RetVal(retError);
            }

            if (!retVal.containsError() && data)
            {
                try
                {
                    if (PyLong_Check(data))
                    {
                        int overflow;
                        *(self->dataObject) = (int32)PyLong_AsLongAndOverflow(data, &overflow);
                        if (overflow)
                        {
                            throw cv::Exception(
                                0,
                                "overflow: given data exceeds the integer boundaries.",
                                "PyDataObject_init",
                                __FILE__,
                                __LINE__);
                        }
                    }
                    else if (PyFloat_Check(data))
                    {
                        *(self->dataObject) = (float64)PyFloat_AsDouble(data);
                    }
                    else if (PyComplex_Check(data))
                    {
                        *(self->dataObject) =
                            complex128(PyComplex_RealAsDouble(data), PyComplex_ImagAsDouble(data));
                    }
                    else if (PyRgba_Check(data))
                    {
                        const auto* rgba = reinterpret_cast<const PythonRgba::PyRgba*>(data);
                        *(self->dataObject) = rgba->rgba;
                    }
                    else if (PythonDateTime::PyDateTime_CheckExt(data))
                    {
                        bool ok;
                        const auto dt = PythonDateTime::GetDateTime(data, ok);

                        if (!ok)
                        {
                            throw cv::Exception(
                                0,
                                "Value could not be parsed to an itom datetime value.",
                                "PyDataObject_init",
                                __FILE__,
                                __LINE__);
                        }

                        *(self->dataObject) = dt;
                    }
                    else if (PythonDateTime::PyTimeDelta_CheckExt(data))
                    {
                        bool ok;
                        const auto td = PythonDateTime::GetTimeDelta(data, ok);

                        if (!ok)
                        {
                            throw cv::Exception(
                                0,
                                "Value could not be parsed to an itom timedelta value.",
                                "PyDataObject_init",
                                __FILE__,
                                __LINE__);
                        }

                        *(self->dataObject) = td;
                    }
                    else if (PySequence_Check(data))
                    {
                        if (typeno == ito::tDateTime)
                        {
                            // a bypass by a numpy array would lead to a removal of
                            // timezone information, since they are not available
                            // in np.datetime64. Therefore, the assignment is done
                            // manually for this specific data type.
                            Py_ssize_t seqIdx = 0;
                            PyObject *item;
                            int numPlanes = self->dataObject->getNumPlanes();
                            ito::DateTime *rowPtr;

                            for (int planeIdx = 0; planeIdx < numPlanes; ++planeIdx)
                            {
                                cv::Mat *mat = self->dataObject->getCvPlaneMat(planeIdx);

                                for (int c = 0; c < mat->cols; ++c)
                                {
                                    rowPtr = mat->ptr<ito::DateTime>(0, c);

                                    for (int r = 0; r < mat->rows; ++r)
                                    {
                                        item = PySequence_GetItem(data, seqIdx); // new ref

                                        if (PythonDateTime::PyDateTime_CheckExt(item))
                                        {
                                            rowPtr[r] = PythonDateTime::GetDateTime(item, ok);

                                            if (!ok)
                                            {
                                                PyErr_Format(
                                                    PyExc_ValueError,
                                                    "The %i. value cannot be converted to a datetime.", seqIdx + 1);
                                                retVal += RetVal(retError);
                                            }
                                        }
                                        else
                                        {
                                            PyErr_Format(
                                                PyExc_ValueError,
                                                "The %i. value cannot be converted to a datetime.", seqIdx + 1);
                                            retVal += RetVal(retError);
                                        }

                                        seqIdx++;
                                        Py_XDECREF(item);
                                    }
                                }
                            }
                        }
                        else
                        {
                            int npTypenum = getNpTypeFromDataObjectType(typeno);

                            if (npTypenum == -1)
                            {
                                throw cv::Exception(
                                    0,
                                    "No compatible np datatype found for desired dtype",
                                    "PyDataObject_init",
                                    __FILE__,
                                    __LINE__);
                            }

                            PyObject* npArray = PyArray_ContiguousFromAny(data, npTypenum, 1, 1);

                            if (npArray == nullptr)
                            {
                                // Python error is set... Therefore just throw an exception without
                                // message
                                throw cv::Exception(0, "", "PyDataObject_init", __FILE__, __LINE__);
                            }
                            else
                            {
                                retVal += copyNpArrayValuesToDataObject(
                                    (PyArrayObject*)npArray, self->dataObject, (ito::tDataType)typeno);

                                if (retVal.containsError())
                                {
                                    throw cv::Exception(
                                        0,
                                        retVal.errorMessage(),
                                        "PyDataObject_init",
                                        __FILE__,
                                        __LINE__);
                                }
                            }

                            Py_XDECREF(npArray);
                        }
                    }
                    else
                    {
                        throw cv::Exception(
                            0, "invalid data value", "PyDataObject_init", __FILE__, __LINE__);
                    }
                }
                catch (cv::Exception& exc)
                {
                    if (!PyErr_Occurred())
                    {
                        // no python error set, yet: set it from the exception message
                        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                    }

                    DELETE_AND_SET_NULL(self->dataObject);
                    retVal += RetVal(retError);
                }
            }
        }

        DELETE_AND_SET_NULL_ARRAY(sizes);
    }

    return retVal.containsError() ? -1 : 0;
}

//-------------------------------------------------------------------------------------
/*
return 0 if dataObject could be created. self->dataObject is allocated then.
return -1 in case of a general error, Python error message is set
return -2 if args / kwds cannot be parsed, Python error message is set, too
*/
int PythonDataObject::PyDataObj_CreateFromNpNdArrayAndType(
    PyDataObject* self,
    PyObject* args,
    PyObject* kwds,
    bool addNpOrgTags) // helper method for PyDataObject_init
{
    const char* kwlist[] = {"object", "dtype", "continuous", nullptr};
    PyArrayObject* ndArrayRef = nullptr;
    PyObject* dimList = nullptr;
    const char* typeName = "\0";

    // continuous is not used in this method, but can be part of the arguments
    unsigned char continuous = 0;

#if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
    int C_CONTIGUOUS = NPY_C_CONTIGUOUS;
#else
    int C_CONTIGUOUS = NPY_ARRAY_C_CONTIGUOUS;
#endif

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!|sb",
            const_cast<char**>(kwlist),
            &PyArray_Type,
            &ndArrayRef,
            &typeName,
            &continuous))
    {
        return -2;
    }

    const PyArray_Descr* descr = PyArray_DESCR(ndArrayRef);
    int destDObjTypeNo = -1;
    int inputNpArrayTypeNo = getDObjTypeOfNpArray(descr->kind, PyArray_ITEMSIZE(ndArrayRef));
    PyArrayObject* ndArrayNew = nullptr;

    // if true, stride values must be given to the dataObject constructor.
    // Else, a contiguous array is assumed an no stride values are required.
    // The latter is necessary, since some contiguous np.ndarrays have also
    // stride values of 0 (e.g. possible if its dimension is equal to 1) and
    // this is not supported by the dataObject.
    bool stridesRequired = false;

    // at first, check copyObject. there are three cases: 1. we can take it as it is, 2. it is
    // compatible but has to be converted, 3. it is incompatible
    if (!(descr->byteorder == '<' || descr->byteorder == '|' ||
          (descr->byteorder == '=' && NPY_NATBYTE == NPY_LITTLE)))
    {
        PyErr_SetString(
            PyExc_TypeError,
            "Given numpy array has wrong byteorder (litte endian desired), which cannot be "
            "transformed to dataObject");
        return -1;
    }

    // now get the desired output type
    if (strlen(typeName) != 0)
    {
        destDObjTypeNo = dObjTypeFromName(typeName);

        if (destDObjTypeNo == -1 || destDObjTypeNo == ito::tUInt32)
        {
            PyErr_SetString(
                PyExc_ValueError,
                "Invalid type name. Allowed type names are 'uint8', 'int8', 'uint16', 'int16', "
                "'int32', 'float32', 'float64', 'complex64', 'complext128', 'rgba32', 'datetime', "
                "'timedelta'");
            return -1;
        }
    }
    else
    {
        // guess the type from the type of the given numpy array.

        // check whether type of ndarray exists for data object
        destDObjTypeNo = inputNpArrayTypeNo;

        if (inputNpArrayTypeNo == -1)
        {
            // check whether type is compatible
            destDObjTypeNo =
                getCompatibleDObjTypeOfNpArray(descr->kind, PyArray_ITEMSIZE(ndArrayRef));

            if (destDObjTypeNo == -1) // no compatible type found
            {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Could not find a compatible Numpy.dtype. Allowed types are 'uint8', 'int8', "
                    "'uint16', 'int16', 'int32', 'float32', 'float64', 'complex64', 'complext128', "
                    "'rgba32', 'datetime', 'timedelta'");
                return -1;
            }
        }
    }

    if (destDObjTypeNo == inputNpArrayTypeNo)
    {
        // the original np.ndarray dtype and the dtype of
        // the new dataObject are exactly the same.
        int flags = PyArray_FLAGS(ndArrayRef);
        int dims = PyArray_NDIM(ndArrayRef);
        const npy_intp* strides = PyArray_STRIDES(ndArrayRef);
        const npy_intp* sizes = PyArray_DIMS(ndArrayRef);
        bool ok = true;

        if (((flags & NPY_ARRAY_BEHAVED) != NPY_ARRAY_BEHAVED) ||
            (dims > 0 && strides[dims - 1] != PyArray_ITEMSIZE(ndArrayRef)))
        {
            ok = false;
        }
        else
        {
            // check if all strides are descending.
            npy_intp current = 0;

            for (int d = dims - 1; d >= 0; --d)
            {
                if (strides[d] < current)
                {
                    ok = false;
                    break;
                }

                current = strides[d];
            }
        }

        if (!ok)
        {
            // the contiguous array is c-style, contiguous, well behaved...
            // forces a contiguous copy, see PyArray_GETCONTIGUOUS(ndArrayRef);
            ndArrayNew = (PyArrayObject*)PyArray_Copy(ndArrayRef); // new ref
            stridesRequired = false;
        }
        else
        {
            ndArrayNew = ndArrayRef;
            Py_INCREF((PyObject*)ndArrayNew);
            stridesRequired = true;
        }
    }
    else
    {
        int newNumpyTypeNum = getNpTypeFromDataObjectType(destDObjTypeNo);

        if (newNumpyTypeNum != -1)
        {
            // now we always have an increased reference of
            // ndArray (either reference of old ndArray or new
            // object with new reference)
            ndArrayNew = (PyArrayObject*)PyArray_FROM_OTF(
                (PyObject*)ndArrayRef, newNumpyTypeNum, C_CONTIGUOUS | NPY_ARRAY_FORCECAST);

            // verify that the strides are descending and the last stride corresponds to item size
            const npy_intp* npstrides = (npy_intp*)PyArray_STRIDES(ndArrayNew);
            npy_intp currentStride = std::numeric_limits<npy_intp>::max();
            int dims = PyArray_NDIM(ndArrayNew);

            for (int i = 0; i < dims; ++i)
            {
                if ((npstrides[i] > currentStride) ||
                    ((i == dims - 1) && (npstrides[i] != PyArray_ITEMSIZE(ndArrayNew))))
                {
                    PyErr_Format(
                        PyExc_TypeError,
                        "Invalid format of the converted numpy.ndarray (unsupported strides).");
                    Py_XDECREF(ndArrayNew);
                    return -1;
                }

                currentStride = npstrides[i];
            }

            // For the returned array, all values are continous in memory.
            // Therefore no special strides are required.
            stridesRequired = false;
        }
        else
        {
            PyErr_Format(
                PyExc_TypeError,
                "Could not find a Numpy.dtype, compatible to the desired type '%s'",
                typeName);
            return -1;
        }
    }

    if (ndArrayNew == nullptr)
    {
        if (!PyErr_Occurred())
        {
            // error message is usually set by PyArray_FROM_OTF or PyArray_GETCONTIGUOUS itself. If
            // not...
            PyErr_SetString(
                PyExc_TypeError,
                "An error occurred while transforming the given np.array to a c-contiguous array "
                "with a compatible type.");
        }

        return -1;
    }
    else
    {
        // final check if the numpy array is really compatible.
        descr = PyArray_DESCR(ndArrayNew);
        destDObjTypeNo = getDObjTypeOfNpArray(descr->kind, PyArray_ITEMSIZE(ndArrayNew));

        if (destDObjTypeNo == -1 || destDObjTypeNo == ito::tUInt32)
        {
            PyErr_SetString(
                PyExc_TypeError,
                "While converting the given np.array to a compatible data type with respect to "
                "data object, an error occurred.");
            Py_DECREF(ndArrayNew);
            return -1;
        }
    }

    int dimensions = PyArray_NDIM(ndArrayNew);

    if (dimensions <= 0 || PyArray_SIZE(ndArrayNew) <= 0)
    {
        // create an empty dataObject
        DELETE_AND_SET_NULL(self->dataObject);

        if (destDObjTypeNo >= 0)
        {
            self->dataObject = new ito::DataObject(0, destDObjTypeNo);
        }
        else
        {
            self->dataObject = new ito::DataObject();
        }

        Py_XDECREF(ndArrayNew);
        return 0;
    }
    else
    {
        const uchar* data = (uchar*)PyArray_DATA(ndArrayNew);
        const npy_intp* npsizes = PyArray_DIMS(ndArrayNew);

        // number of bytes to jump from one element in
        // one dimension to the next one
        const npy_intp* npstrides = (npy_intp*)PyArray_STRIDES(ndArrayNew);

        int* sizes = new int[dimensions];

        for (int n = 0; n < dimensions; n++)
        {
            sizes[n] = npsizes[n];
        }

        int* steps = nullptr;

        if (stridesRequired)
        {
            steps = new int[dimensions];

            for (int n = 0; n < dimensions; n++)
            {
                steps[n] = npstrides[n];
            }
        }

        bool error = false;

        // here size of steps is equal to size of sizes, DataObject only requires the first
        // dimensions-1 elements of steps

        // verify that last dimension has steps size equal to itemsize
        // or the last dimension has a step size of 0, but then the size of this last dimension
        // must be 0.
        DELETE_AND_SET_NULL(self->dataObject);

        switch (destDObjTypeNo)
        {
        case ito::tDateTime:
            // always deep copy
            error = PyDataObj_CopyFromDatetimeNpNdArray(self, ndArrayNew, dimensions, sizes);
            break;
        case ito::tTimeDelta:
            // always deep copy
            error = PyDataObj_CopyFromTimedeltaNpNdArray(self, ndArrayNew, dimensions, sizes);
            break;
        default:
            // always shallow copy
            try
            {
                self->dataObject = new ito::DataObject(
                    static_cast<unsigned char>(dimensions), sizes, destDObjTypeNo, data, steps);
            }
            catch (cv::Exception& exc)
            {
                PyErr_Format(
                    PyExc_RuntimeError, "failed to create data object: %s", (exc.err).c_str());
                self->dataObject = nullptr;
                error = true;
            }

            PyDataObject_SetBase(self, (PyObject*)ndArrayNew);
            Py_XDECREF(ndArrayNew);
        }

        // If dataObject.continuous is set to 255 in the python_unittests
        // the tags _orgNp... are added, only for the python_unittests
        if (continuous == 255)
        {
            addNpOrgTags = true;
        }

        if (!error && addNpOrgTags)
        {
            // add tag _dtype with original shape of numpy.ndarray
            self->dataObject->setTag(
                "_orgNpDType", getNpDTypeStringFromNpDTypeEnum(PyArray_TYPE(ndArrayRef)));

            // add tag _shape with original shape of numpy.ndarray
            QString npArrayNewShape = "[";

            if (dimensions == 1)
            {
                npArrayNewShape.append(QString::number(sizes[0]));
            }
            else if (dimensions > 1)
            {
                for (int n = 0; n < dimensions - 1; ++n)
                {
                    npArrayNewShape.append(QString("%1 x ").arg(QString::number(sizes[n])));
                }

                npArrayNewShape.append(QString::number(sizes[dimensions - 1]));
            }

            npArrayNewShape.append("]");

            self->dataObject->setTag("_orgNpShape", npArrayNewShape.toStdString());
        }

        DELETE_AND_SET_NULL_ARRAY(sizes);
        DELETE_AND_SET_NULL_ARRAY(steps);

        return error ? -1 : 0;
    }
}

//-------------------------------------------------------------------------------------
bool PythonDataObject::PyDataObj_CopyFromDatetimeNpNdArray(
    PyDataObject* self, PyArrayObject* dateTimeArray, int dims, const int* sizes)
{
    bool error = false;

    const uchar* data = (uchar*)PyArray_DATA(dateTimeArray);

    // number of bytes to jump from one element in
    // one dimension to the next one
    const npy_intp* npstrides = PyArray_STRIDES(dateTimeArray);

    const auto descr = PyArray_DESCR(dateTimeArray);

    // in case of datetime or timedelta: The values are int64, based on 1.1.1970
    // the timebase is given by:
    const auto md = (PyArray_DatetimeDTypeMetaData*)(descr->c_metadata);
    // timezone is ignored in numpy. If dataObject contains a timezone, ignore it and raise a
    // warning.

    if (md == nullptr)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to read the time unit of the numpy.ndarray.");
        return true;
    }

    try
    {
        // create a continuous object for an easier iteration
        self->dataObject = new ito::DataObject(
            static_cast<unsigned char>(dims), sizes, ito::tDateTime, (unsigned char)1);
    }
    catch (cv::Exception& exc)
    {
        PyErr_Format(PyExc_RuntimeError, "failed to create data object: %s", (exc.err).c_str());
        self->dataObject = nullptr;
        error = true;
    }

    if (self->dataObject->getTotal() == 0)
    {
        return false;
    }

    ito::DateTime* dObjData = self->dataObject->rowPtr<ito::DateTime>(0, 0);

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    NpyIter* iter = NpyIter_New(
        dateTimeArray,
        NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_CORDER,
        NPY_NO_CASTING,
        nullptr);

    if (iter == nullptr)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        DELETE_AND_SET_NULL(self->dataObject);
        return true;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, nullptr);

    if (iternext == nullptr)
    {
        NpyIter_Deallocate(iter);

        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        DELETE_AND_SET_NULL(self->dataObject);
        return true;
    }

    /* The location of the data pointer which the iterator may update */
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    npy_intp* innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    npy_datetime dt;

    do
    {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--)
        {
            dt = *((npy_datetime*)data);

            if (!PythonDateTime::NpyDatetime2itoDatetime(dt, md->meta, *dObjData))
            {
                error = true;
                break;
            }

            dObjData++;

            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while (iternext(iter) && !error);

    NpyIter_Deallocate(iter);

    return error;
}

//-------------------------------------------------------------------------------------
bool PythonDataObject::PyDataObj_CopyFromTimedeltaNpNdArray(
    PyDataObject* self, PyArrayObject* timeDeltaArray, int dims, const int* sizes)
{
    bool error = false;

    const uchar* data = (uchar*)PyArray_DATA(timeDeltaArray);

    // number of bytes to jump from one element in
    // one dimension to the next one
    const npy_intp* npstrides = PyArray_STRIDES(timeDeltaArray);

    const auto descr = PyArray_DESCR(timeDeltaArray);

    // in case of datetime or timedelta: The values are int64, based on 1.1.1970
    // the timebase is given by:
    const auto md = (PyArray_DatetimeDTypeMetaData*)(descr->c_metadata);

    if (md == nullptr)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to read the time unit of the numpy.ndarray.");
        return true;
    }
    else if (md->meta.base == NPY_FR_Y || md->meta.base == NPY_FR_M)
    {
        PyErr_Format(
            PyExc_RuntimeError,
            "Cannot convert a year or month timebase into a dataObject timedelta data type.");
        return true;
    }

    try
    {
        // create a continuous object for an easier iteration
        self->dataObject = new ito::DataObject(
            static_cast<unsigned char>(dims), sizes, ito::tTimeDelta, (unsigned char)1);
    }
    catch (cv::Exception& exc)
    {
        PyErr_Format(PyExc_RuntimeError, "failed to create data object: %s", (exc.err).c_str());
        self->dataObject = nullptr;
        error = true;
    }

    if (self->dataObject->getTotal() == 0)
    {
        return false;
    }

    ito::TimeDelta* dObjData = self->dataObject->rowPtr<ito::TimeDelta>(0, 0);

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    NpyIter* iter = NpyIter_New(
        timeDeltaArray,
        NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_CORDER,
        NPY_NO_CASTING,
        nullptr);

    if (iter == nullptr)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        DELETE_AND_SET_NULL(self->dataObject);
        return true;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, nullptr);

    if (iternext == nullptr)
    {
        NpyIter_Deallocate(iter);

        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        DELETE_AND_SET_NULL(self->dataObject);
        return true;
    }

    /* The location of the data pointer which the iterator may update */
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    npy_intp* innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    npy_timedelta dt;

    do
    {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--)
        {
            dt = *((npy_timedelta*)data);

            if (!PythonDateTime::NpyTimedelta2itoTimedelta(dt, md->meta, *dObjData))
            {
                error = true;
            }

            dObjData++;

            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while (iternext(iter) && !error);

    NpyIter_Deallocate(iter);

    return error;
}

//-------------------------------------------------------------------------------------
RetVal PythonDataObject::PyDataObj_ParseCreateArgs(
    PyObject* args,
    PyObject* kwds,
    int& typeno,
    std::vector<unsigned int>& sizes,
    unsigned char& continuous)
{
    static const char* kwlist[] = {"dims", "dtype", "continuous", NULL};
    PyObject* dimList = NULL;
    unsigned int dims = 0;
    int tempSizes = 0;

    const char* type = typeNumberToName(typeno);
    if (type == NULL)
    {
        type = "uint8\0"; // default
    }

    RetVal retValue(retOk);

    // check for argument: list(int size1, int size2,...,int sizeLast)[, dtype='typename'][,
    // continuous=[0|1]
    if (PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!|sb",
            const_cast<char**>(kwlist),
            &PyList_Type,
            &dimList,
            &type,
            &continuous))
    {
        typeno = dObjTypeFromName(type);

        if (typeno >= 0)
        {
            dims = PyList_Size(dimList);

            if (dims < 0)
            {
                retValue += RetVal(retError);
                PyErr_SetString(PyExc_TypeError, "Number of dimensions must be bigger than zero.");
            }
            else if (dims > 255)
            {
                retValue += RetVal(retError);
                PyErr_SetString(PyExc_TypeError, "Number of dimensions must be lower than 256.");
            }

            if (!retValue.containsError())
            {
                sizes.clear();
                sizes.resize(dims);

                // try to parse list to values of unsigned int
                for (unsigned int i = 0; i < dims; i++)
                {
                    if (!PyArg_Parse(PyList_GetItem(dimList, i), "I", &tempSizes)) // borrowed ref
                    {
                        PyErr_PrintEx(0);
                        PyErr_Clear();
                        PyErr_Format(
                            PyExc_TypeError,
                            "Element %d of dimension-list is no integer number",
                            i + 1);
                        retValue += RetVal(retError);
                        break;
                    }
                    else if (tempSizes <= 0)
                    {
                        PyErr_SetString(PyExc_TypeError, "Element %d must be bigger than 1");
                        retValue += RetVal(retError);
                        break;
                    }

                    sizes[i] = tempSizes;
                }
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "dtype name is unknown.");
            retValue += RetVal(retError);
        }
    }
    else if (
        PyErr_Clear(),
        PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!|sb",
            const_cast<char**>(kwlist),
            &PyTuple_Type,
            &dimList,
            &type,
            &continuous))
    {
        typeno = dObjTypeFromName(type);
        if (typeno >= 0)
        {
            dims = PyTuple_Size(dimList);

            if (dims < 0)
            {
                retValue += RetVal(retError);
                PyErr_SetString(PyExc_TypeError, "Number of dimensions must be bigger than zero.");
            }
            else if (dims > 255)
            {
                retValue += RetVal(retError);
                PyErr_SetString(PyExc_TypeError, "Number of dimensions must be lower than 256.");
            }

            if (!retValue.containsError())
            {
                sizes.clear();
                sizes.resize(dims);

                // try to parse list to values of unsigned int
                for (unsigned int i = 0; i < dims; i++)
                {
                    if (!PyArg_Parse(PyTuple_GetItem(dimList, i), "I", &tempSizes)) // borrowed ref
                    {
                        PyErr_PrintEx(0);
                        PyErr_Clear();
                        PyErr_Format(
                            PyExc_TypeError,
                            "Element %d of dimension-tuple is no integer number",
                            i + 1);
                        retValue += RetVal(retError);
                        break;
                    }
                    else if (tempSizes <= 0)
                    {
                        PyErr_SetString(PyExc_TypeError, "Element %d must be bigger than 1");
                        retValue += RetVal(retError);
                        break;
                    }

                    sizes[i] = tempSizes;
                }
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "dtype name is unknown.");
            retValue += RetVal(retError);
        }
    }
    else
    {
        PyErr_SetString(
            PyExc_TypeError,
            "required arguments: list/tuple(int size1, int size2,...,int sizeLast)[, "
            "dtype='typename'][, continuous=[0|1]");
        retValue += RetVal(retError);
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrDims_doc, "int : Gets the number of dimensions of this data object.");
PyObject* PythonDataObject::PyDataObj_GetDims(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PyLong_FromLong(self->dataObject->getDims());
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrType_doc, "str : Gets the data type name of the values in this dataObject. \n\
\n\
This type string has one of these values: ``uint8``, ``int8``, ``uint16``, \n\
``int16``, ``int32``, ``float32``, ``float64``, ``complex64``, ``complex128``, \n\
``rgba32``, ``datetime`` or ``timedelta``.");
PyObject* PythonDataObject::PyDataObj_GetType(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == nullptr)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", typeNumberToName(self->dataObject->getType()));
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrContinuous_doc,
    "bool : Returns ``True`` if this dataObject is continuous, otherwise ``False``. \n\
\n\
If ``True``, the whole matrix is allocated in one huge block in memory, hence, \n\
this data object can be transformed into a numpy representation \n\
without reallocating memory.");
PyObject* PythonDataObject::PyDataObj_GetContinuous(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        Py_RETURN_NONE;
    }
    else
    {
        bool cont = self->dataObject->getContinuous();

        if (cont)
        {
            Py_RETURN_TRUE;
        }
        else
        {
            Py_RETURN_FALSE;
        }
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrShape_doc, "tuple of int : Gets the shape of this data object. \n\
\n\
The shape is a tuple where each element is the size of one dimension of this \n\
dataObject. As an example ``shape = [2, 3]`` corresponds to a ``2 x 3`` dataObject. \n\
\n\
Notes\n\
-----\n\
In difference to the shape attribute of :class:`numpy.ndarray`, this attribute cannot \n\
be set. \n\
\n\
See Also \n\
-------- \n\
size : Alternative method to return the size of all or any specific axis");
PyObject* PythonDataObject::PyDataObj_GetShape(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    int dims = self->dataObject->getDims();
    PyObject* retList = NULL;
    int desiredDim = 0;

    retList = PyTuple_New(dims); // new reference

    for (int i = 0; i < dims; i++)
    {
        PyTuple_SetItem(retList, i, PyLong_FromLong(self->dataObject->getSize(i)));
    }

    return retList;
}

//-------------------------------------------------------------------------------------

//---------------------------------------Get / Set metadata / objecttags---------------
PyDoc_STRVAR(
    dataObjectAttrTags_doc,
    "types.MappingProxyType : Gets or sets a dictionary with tags of this data object. \n\
\n\
This attribute returns a :obj:`dict_proxy` object of the tag dictionary of this \n\
data object. This object is read-only. However you can assign an entire new \n\
dictionary to this attribute that fully replaces the old tag dictionary. \n\
The tag dictionary can contain arbitrary pairs of key -> value where value is either \n\
a :class:`str` or a :class:`float` value. \n\
\n\
Special tags are the key ``protocol`` that contains the newline-separated protocol \n\
string of the data object (see: :meth:`addToProtocol`) or the key ``title`` that  \n\
can for instance be used as title in any plots. \n\
\n\
You can add single elements using the method :meth:`setTag` or you can delete tags \n\
using :meth:`deleteTag`.\n\
\n\
Do NOT use **special character** within the tag key because they are not XML-save.");
// getter and setter methods
PyObject* PythonDataObject::PyDataObject_getTags(PyDataObject* self, void* /*closure*/)
{
    PyObject* ret = PyDict_New();
    int size = self->dataObject->getTagListSize();
    bool valid;
    std::string key;
    // std::string value;
    DataObjectTagType value;

    for (int i = 0; i < size; i++)
    {
        valid = self->dataObject->getTagByIndex(i, key, value);
        if (valid)
        {
            if (value.getType() == DataObjectTagType::typeDouble)
            {
                PyObject* item = PyFloat_FromDouble(value.getVal_ToDouble());
                PyDict_SetItemString(ret, key.data(), item);
                Py_DECREF(item);
            }
            else
            {
                PyObject* text =
                    PythonQtConversion::ByteArrayToPyUnicode(value.getVal_ToString().data());
                if (text)
                {
                    PyDict_SetItemString(ret, key.data(), text);
                    Py_DECREF(text);
                }
                else
                {
                    text = PythonQtConversion::ByteArrayToPyUnicode("<encoding error>");
                    PyDict_SetItemString(ret, key.data(), text);
                    Py_DECREF(text);
                }
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "error while reading tags from data object");
            return NULL;
        }
    }

    PyObject* proxy = PyDictProxy_New(ret);
    Py_DECREF(ret);
    return proxy;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setTags(PyDataObject* self, PyObject* value, void* /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (!PyDict_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "The tag attribute must be a dictionary");
        return -1;
    }

    self->dataObject->deleteAllTags();

    PyObject* key;
    PyObject* content;
    std::string keyString, contentString;
    Py_ssize_t pos = 0;
    bool stringOk;

    while (PyDict_Next(value, &pos, &key, &content))
    {
        keyString = PythonQtConversion::PyObjGetStdStringAsLatin1(key, false, stringOk);
        if (stringOk)
        {
            if (PyFloat_Check(content) || PyLong_Check(content))
            {
                self->dataObject->setTag(keyString, PyFloat_AsDouble(content));
            }
            else
            {
                contentString =
                    PythonQtConversion::PyObjGetStdStringAsLatin1(content, false, stringOk);
                if (stringOk)
                {
                    self->dataObject->setTag(keyString, contentString);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "tags must be convertable into strings");
                    return -1;
                }
            }
        }
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrAxisScales_doc,
    "tuple of float : Gets or sets the optional scale values for each axis [unit/px]. \n\
\n\
This attribute gives access to the internal axis scales [unit/px] expressed as \n\
a :class:`tuple` of :class:`float` values. The i-th value in the tuple corresponds \n\
to the scaling factor of the i-th axis. Either assign a new tuple with the same \n\
length than the number of dimensions or change single values using tuple indexing. \n\
\n\
Definition: ``Physical unit = (px-Coordinate - offset)* scale`` \n\
\n\
If the data object is plot with scalings != 1, the scaled (physical) units are \n\
displayed in the plot. \n\
\n\
See Also \n\
-------- \n\
setAxisScale : Alternative method to set the scale value of one single axis");
PyObject* PythonDataObject::PyDataObject_getAxisScales(PyDataObject* self, void* /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    PyObject* ret =
        PyTuple_New(dims); // must be tuple, such that items cannot be changed, since this tuple is
                           // no reference but deep copy to the real tags in self->dataObject
    double temp;

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisScale(i);
        PyTuple_SetItem(ret, i, PyFloat_FromDouble(temp)); // steals reference
    }

    return ret;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisScales(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    double scale;
    PyObject* tempObj;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis scales must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(
            PyExc_TypeError, "length of axis scale sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        tempObj = PySequence_GetItem(value, i); // new reference
        if (PyFloat_Check(tempObj))
        {
            scale = PyFloat_AsDouble(tempObj);
        }
        else if (PyLong_Check(tempObj))
        {
            scale = static_cast<double>(PyLong_AsLong(tempObj));
        }
        else
        {
            Py_XDECREF(tempObj);
            PyErr_SetString(PyExc_TypeError, "elements of axis scale vector must be a number");
            return -1;
        }
        Py_XDECREF(tempObj);

        if (self->dataObject->setAxisScale(i, scale))
        {
            PyErr_SetString(PyExc_ValueError, "invalid scale value");
            return -1;
        }
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrAxisOffsets_doc,
    "tuple of float : Gets or sets the optional offset values for each axis [px]. \n\
\n\
This attribute gives access to the internal axis offsets [px] expressed as \n\
a :class:`tuple` of :class:`float` values. The i-th value in the tuple corresponds \n\
to the pixel-offset of the i-th axis. Either assign a new tuple with the same length \n\
than the number of dimensions or change single values using tuple indexing. \n\
\n\
Definition: ``Physical unit = (px-Coordinate - offset)* scale`` \n\
\n\
If the data object is plot with offsets != 0, the scaled (physical) units are \n\
displayed in the plot. \n\
\n\
See Also \n\
-------- \n\
setAxisOffset : Alternative method to set the offset value of one single axis");
PyObject* PythonDataObject::PyDataObject_getAxisOffsets(PyDataObject* self, void* /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    PyObject* ret =
        PyTuple_New(dims); // must be tuple, such that items cannot be changed, since this tuple is
                           // no reference but deep copy to the real tags in self->dataObject
    double temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisOffset(i);
        PyTuple_SetItem(ret, i, PyFloat_FromDouble(temp));
    }

    return ret;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisOffsets(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    double offset;
    PyObject* tempObj;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis offsets must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(
            PyExc_TypeError,
            "length of axis offset sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        tempObj = PySequence_GetItem(value, i); // new reference
        if (PyFloat_Check(tempObj))
        {
            offset = PyFloat_AsDouble(tempObj);
        }
        else if (PyLong_Check(tempObj))
        {
            offset = static_cast<double>(PyLong_AsLong(tempObj));
        }
        else
        {
            Py_XDECREF(tempObj);
            PyErr_SetString(PyExc_TypeError, "elements of axis offset vector must be a number");
            return -1;
        }
        Py_XDECREF(tempObj);

        if (self->dataObject->setAxisOffset(i, offset))
        {
            PyErr_SetString(PyExc_ValueError, "invalid offset value");
            return -1;
        }
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrAxisDescriptions_doc,
    "tuple of str : Gets or sets the optional description of each axis. \n\
\n\
This tuple contains the description of each axis. The length of this tuple \n\
is equal to the number of dimensions of the dataObject. \n\
\n\
You can either assign a new tuple with the same length or change single values using \n\
tuple indexing. \n\
\n\
The axis descriptions are considered if the data object is plotted. \n\
\n\
See Also \n\
-------- \n\
setAxisDescriptions : alternative method to change the description string of one single axis.");
PyObject* PythonDataObject::PyDataObject_getAxisDescriptions(PyDataObject* self, void* /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    PyObject* ret =
        PyTuple_New(dims); // must be tuple, such that items cannot be changed, since this tuple is
                           // no reference but deep copy to the real tags in self->dataObject
    bool valid;
    std::string temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisDescription(i, valid);
        if (valid)
        {
            // PyObject *string = PyUnicode_FromString(temp.data());
            PyObject* string = PyUnicode_DecodeLatin1(temp.data(), temp.length(), NULL);
            if (string == NULL)
            {
                string = PyUnicode_FromString("<encoding error>"); // TODO
            }
            PyTuple_SetItem(ret, i, string); // steals reference from string
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "error while reading axis units from data object");
            return NULL;
        }
    }

    return ret;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisDescriptions(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    std::string tempString;
    PyObject* seqItem = NULL;
    bool ok;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis descriptions must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(
            PyExc_TypeError,
            "length of axis description sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        seqItem = PySequence_GetItem(value, i); // new reference
        tempString = PythonQtConversion::PyObjGetStdStringAsLatin1(seqItem, true, ok);
        Py_XDECREF(seqItem);
        if (!ok)
        {
            PyErr_SetString(
                PyExc_TypeError, "elements of axis description vector must be string types");
            return -1;
        }
        self->dataObject->setAxisDescription(i, tempString);
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrAxisUnits_doc,
    "tuple of str : Gets or sets the optional unit value of each axis. \n\
\n\
This tuple contains the unit value of each axis. The length of this tuple \n\
is equal to the number of dimensions of the dataObject. \n\
\n\
You can either assign a new tuple with the same length or change single values using \n\
tuple indexing. \n\
\n\
The axis units are considered if the data object is plotted. \n\
\n\
See Also \n\
-------- \n\
setAxisDescriptions : alternative method to change the description string of one single axis.");
PyObject* PythonDataObject::PyDataObject_getAxisUnits(PyDataObject* self, void* /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    PyObject* ret =
        PyTuple_New(dims); // must be tuple, such that items cannot be changed, since this tuple is
                           // no reference but deep copy to the real tags in self->dataObject
    bool valid;
    std::string temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisUnit(i, valid);
        if (valid)
        {
            // PyTuple_SetItem(ret, i, PyUnicode_FromString(temp.data()));
            PyTuple_SetItem(ret, i, PyUnicode_DecodeLatin1(temp.data(), temp.length(), NULL));
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "error while reading axis units from data object");
            return NULL;
        }
    }

    return ret;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisUnits(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    std::string tempString;
    PyObject* seqItem = NULL;
    bool ok;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1)
        dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis units must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(
            PyExc_TypeError, "length of axis unit sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        seqItem = PySequence_GetItem(value, i); // new reference
        tempString = PythonQtConversion::PyObjGetStdStringAsLatin1(seqItem, true, ok);
        Py_XDECREF(seqItem);
        if (!ok)
        {
            PyErr_SetString(PyExc_TypeError, "elements of axis unit vector must be string types");
            return -1;
        }
        self->dataObject->setAxisUnit(i, tempString);
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrValueUnit_doc, "str : Gets or sets the unit of the values of the dataObject. \n\
\n\
The value unit is considered if the dataObject is plotted.");
PyObject* PythonDataObject::PyDataObject_getValueUnit(PyDataObject* self, void* /*closure*/)
{
    // return PyUnicode_FromString(self->dataObject->getValueUnit().data());
    std::string temp = self->dataObject->getValueUnit().data();
    return PyUnicode_DecodeLatin1(temp.data(), temp.length(), NULL);
}

int PythonDataObject::PyDataObject_setValueUnit(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    bool ok;
    std::string unit = PythonQtConversion::PyObjGetStdStringAsLatin1(value, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "unit value is no string type.");
        return -1;
    }

    if (self->dataObject->setValueUnit(unit))
    {
        PyErr_SetString(PyExc_RuntimeError, "set value unit failed");
        return -1;
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrValueDescription_doc,
    "str : Gets or sets the description of the values of the dataObject. \n\
\n\
The value description is considered if the dataObject is plotted.");
PyObject* PythonDataObject::PyDataObject_getValueDescription(PyDataObject* self, void* /*closure*/)
{
    std::string tempString = self->dataObject->getValueDescription().data();
    // PyObject *temp = PyUnicode_FromString(self->dataObject->getValueDescription().data());
    PyObject* temp = PyUnicode_DecodeLatin1(tempString.data(), tempString.length(), NULL);

    if (temp)
    {
        return temp;
    }

    return PyUnicode_FromString("<encoding error>"); // TODO
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setValueDescription(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    bool ok;
    std::string unit = PythonQtConversion::PyObjGetStdStringAsLatin1(value, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "value description is no string type.");
        return -1;
    }

    if (self->dataObject->setValueDescription(unit))
    {
        PyErr_SetString(PyExc_RuntimeError, "set value unit failed");
        return -1;
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrValueScale_doc,
    "float : Gets the scaling factor for the values. This value is always 1.0.");
PyObject* PythonDataObject::PyDataObject_getValueScale(PyDataObject* self, void* /*closure*/)
{
    return PyFloat_FromDouble(self->dataObject->getValueScale());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrValueOffset_doc,
    "float : Gets the offset value for the values. This value is always 0.0.");
PyObject* PythonDataObject::PyDataObject_getValueOffset(PyDataObject* self, void* /*closure*/)
{
    return PyFloat_FromDouble(self->dataObject->getValueOffset());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrValue_doc,
    "tuple : Gets or sets the values of this dataObject, defined as flattened tuple \n\
\n\
If this attribute is called by means of a getter, a tuple is returned which is \n\
created by iterating through the values of the data object (row-wise). The values \n\
in the tuple depend on the :attr:`dtype` of this dataObject and can be :class:`int`, \n\
:class:`float`, :class:`complex`, :class:`rgba`, :class:`datetime.datetime` \n\
or :class:`datetime.timedelta`. Analog to this, pass a new tuple \n\
of number values and the correct size to change the values of this dataObject. The \n\
size and shape of the object cannot be changed. \n\
\n\
Currently, the setter is only implemented for scalar number types. \n\
\n\
Example: ::\n\
\n\
    b = dataObject[1, 1:10, 1, 1].value\n\
    # or for the first value \n\
    b = dataObject[1, 1:10, 1, 1].value[0]\n\
    # The elements of the tuple are addressed with b[idx].");
PyObject* PythonDataObject::PyDataObject_getValue(PyDataObject* self, void* /*closure*/)
{
    PyObject* outputTuple = NULL;

    int dims = self->dataObject->getDims();

    if (dims == 0)
    {
        return outputTuple = PyTuple_New(0);
    }

    outputTuple = PyTuple_New(self->dataObject->getTotal());

    ito::DObjConstIterator it = self->dataObject->constBegin();
    ito::DObjConstIterator itEnd = self->dataObject->constEnd();
    Py_ssize_t cnt = 0;

    switch (self->dataObject->getType())
    {
    case ito::tInt8:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(outputTuple, cnt++, PyLong_FromLong((long)(*((ito::int8*)(*it)))));
        }
        break;
    case ito::tUInt8:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(outputTuple, cnt++, PyLong_FromLong((long)(*((ito::uint8*)(*it)))));
        }
        break;
    case ito::tInt16:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(outputTuple, cnt++, PyLong_FromLong((long)(*((ito::int16*)(*it)))));
        }
        break;
    case ito::tUInt16:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(outputTuple, cnt++, PyLong_FromLong((long)(*((ito::uint16*)(*it)))));
        }
        break;
    case ito::tInt32:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(outputTuple, cnt++, PyLong_FromLong((long)(*((ito::int32*)(*it)))));
        }
        break;
    case ito::tRGBA32: {
        ito::PythonRgba::PyRgba* color;
        for (; it < itEnd; ++it)
        {
            color = ito::PythonRgba::createEmptyPyRgba();
            if (color)
            {
                color->rgba = ((ito::Rgba32*)(*it))->rgba;
                PyTuple_SetItem(outputTuple, cnt++, (PyObject*)color);
            }
            else
            {
                cnt++;
            }
        }
    }
    break;
    case ito::tFloat32:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(
                outputTuple, cnt++, PyFloat_FromDouble((double)(*((ito::float32*)(*it)))));
        }
        break;
    case ito::tFloat64:
        for (; it < itEnd; ++it)
        {
            PyTuple_SetItem(
                outputTuple, cnt++, PyFloat_FromDouble((double)(*((ito::float64*)(*it)))));
        }
        break;
    case ito::tComplex64: {
        const complex64* value;
        for (; it < itEnd; ++it)
        {
            value = (const complex64*)(*it);
            // steals a reference
            PyTuple_SetItem(
                outputTuple,
                cnt++,
                PyComplex_FromDoubles((double)value->real(), (double)value->imag()));
        }
        break;
    }
    case ito::tComplex128: {
        const complex128* value;
        for (; it < itEnd; ++it)
        {
            value = (const complex128*)(*it);
            // steals a reference
            PyTuple_SetItem(
                outputTuple,
                cnt++,
                PyComplex_FromDoubles((double)value->real(), (double)value->imag()));
        }
        break;
    }
    case ito::tDateTime: {
        const DateTime* value;
        PyObject* obj;
        for (; it < itEnd; ++it)
        {
            value = (const DateTime*)(*it);
            obj = PythonDateTime::GetPyDateTime(*value);

            if (obj)
            {
                // steals a reference
                PyTuple_SetItem(outputTuple, cnt++, obj);
            }
            else
            {
                Py_INCREF(Py_None);
                // steals a reference
                PyTuple_SetItem(outputTuple, cnt++, Py_None);
            }
        }
        break;
    }
    case ito::tTimeDelta: {
        const TimeDelta* value;
        PyObject* obj;
        for (; it < itEnd; ++it)
        {
            value = (const TimeDelta*)(*it);
            obj = PythonDateTime::GetPyTimeDelta(*value);

            if (obj)
            {
                // steals a reference
                PyTuple_SetItem(outputTuple, cnt++, obj);
            }
            else
            {
                Py_INCREF(Py_None);
                // steals a reference
                PyTuple_SetItem(outputTuple, cnt++, Py_None);
            }
        }
        break;
    }
    default:
        Py_XDECREF(outputTuple);
        PyErr_SetString(PyExc_NotImplementedError, "Type not implemented yet");
        return NULL;
    }

    return outputTuple;
}

//-------------------------------------------------------------------------------------
/*static*/ int PythonDataObject::PyDataObject_setValue(
    PyDataObject* self, PyObject* value, void* closure)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "dataObject is NULL");
        return -1;
    }

    int total = self->dataObject->getTotal();
    int typenum;

    switch (self->dataObject->getType())
    {
    case ito::tInt8:
        typenum = NPY_INT8;
        break;
    case ito::tUInt8:
        typenum = NPY_UINT8;
        break;
    case ito::tInt16:
        typenum = NPY_INT16;
        break;
    case ito::tUInt16:
        typenum = NPY_UINT16;
        break;
    case ito::tInt32:
        typenum = NPY_INT32;
        break;
    case ito::tUInt32:
        typenum = NPY_UINT32;
        break;
    case ito::tRGBA32:
        typenum = NPY_UINT32;
        break;
    case ito::tFloat32:
        typenum = NPY_FLOAT32;
        break;
    case ito::tFloat64:
        typenum = NPY_FLOAT64;
        break;
    case ito::tComplex64:
        typenum = NPY_COMPLEX64;
        break;
    case ito::tComplex128:
        typenum = NPY_COMPLEX128;
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "type of dataObject is unknown.");
        return -1;
    }

// try to convert value to a numpy-array
#if !defined(NPY_NO_DEPRECATED_API) || (NPY_NO_DEPRECATED_API < NPY_1_7_API_VERSION)
    PyObject* arr = PyArray_FromObject(value, typenum, 1, 1); // new ref
#else
    PyArrayObject* arr = (PyArrayObject*)PyArray_FromObject(value, typenum, 1, 1); // new ref
#endif

    if (arr == NULL)
    {
        return -1;
    }

    if (PyArray_DIM(arr, 0) != total)
    {
        Py_DECREF(arr);
        PyErr_Format(
            PyExc_RuntimeError,
            "The given array-like object (array, tuple, list...) must have a length of %i",
            total);
        return -1;
    }

    ito::DObjIterator it = self->dataObject->begin();
    ito::DObjIterator itEnd = self->dataObject->end();

    Py_ssize_t cnt = 0;

    switch (self->dataObject->getType())
    {
    case ito::tInt8:
        for (; it < itEnd; ++it)
        {
            *((ito::int8*)(*it)) = *((ito::int8*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tUInt8:
        for (; it < itEnd; ++it)
        {
            *((ito::uint8*)(*it)) = *((ito::uint8*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tInt16:
        for (; it < itEnd; ++it)
        {
            *((ito::int16*)(*it)) = *((ito::int16*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tUInt16:
        for (; it < itEnd; ++it)
        {
            *((ito::uint16*)(*it)) = *((ito::uint16*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tInt32:
        for (; it < itEnd; ++it)
        {
            *((ito::int32*)(*it)) = *((ito::int32*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tRGBA32:
        for (; it < itEnd; ++it)
        {
            ((ito::Rgba32*)(*it))->rgba = *((ito::uint32*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tFloat32:
        for (; it < itEnd; ++it)
        {
            *((ito::float32*)(*it)) = *((ito::float32*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tFloat64:
        for (; it < itEnd; ++it)
        {
            *((ito::float64*)(*it)) = *((ito::float64*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    case ito::tComplex64: {
        for (; it < itEnd; ++it)
        {
            *((ito::complex64*)(*it)) = *((ito::complex64*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    }
    case ito::tComplex128: {
        for (; it < itEnd; ++it)
        {
            *((ito::complex128*)(*it)) = *((ito::complex128*)(PyArray_GETPTR1(arr, cnt++)));
        }
        break;
    }
    default:
        Py_XDECREF(arr);
        PyErr_SetString(PyExc_NotImplementedError, "Type not implemented yet");
        return -1;
    }

    Py_XDECREF(arr);

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrRotationalMatrix_doc,
    "list of list of float : Gets or sets the 3x3 rotation matrix of this dataObject. \n\
\n\
This rotation matrix is part of the meta information section and is not used \n\
for any other purposes. \n\
\n\
The rotation matrix is given as nested list of three elements. Each element is \n\
another list of three float values and correspond to one row in the ``3 x 3`` \n\
rotation matrix.");
int PythonDataObject::PyDataObject_setXYRotationalMatrix(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    if (self == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "dataObject is NULL");
        return -1;
    }

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "content of dataObject is NULL");
        return -1;
    }

    DataObject* dObj = self->dataObject;

    if (PyList_Size(value) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "row number do not match, matrix must be 3x3");
        return -1;
    }

    double ryx[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 3; i++)
    {
        PyObject* slice = PyList_GetItem(value, i);

        if (PyList_Size(slice) != 3)
        {
            PyErr_SetString(PyExc_ValueError, "col number do not match, matrix must be 3x3");
            return -1;
        }

        ryx[i * 3 + 0] = PyFloat_AsDouble(PyList_GetItem(slice, 0));
        ryx[i * 3 + 1] = PyFloat_AsDouble(PyList_GetItem(slice, 1));
        ryx[i * 3 + 2] = PyFloat_AsDouble(PyList_GetItem(slice, 2));
    }

    dObj->setXYRotationalMatrix(
        ryx[0], ryx[1], ryx[2], ryx[3], ryx[4], ryx[5], ryx[6], ryx[7], ryx[8]);

    return 0;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_getXYRotationalMatrix(
    PyDataObject* self, void* /*closure*/)
{
    if (self == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "dataObject is NULL");
        return NULL;
    }

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "content of dataObject is NULL");
        return NULL;
    }

    PyObject* matrix = PyList_New(3);

    DataObject* dObj = self->dataObject;

    double ryx[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    dObj->getXYRotationalMatrix(
        ryx[0], ryx[1], ryx[2], ryx[3], ryx[4], ryx[5], ryx[6], ryx[7], ryx[8]);

    PyObject* slice0 = PyList_New(3);
    PyObject* slice1 = PyList_New(3);
    PyObject* slice2 = PyList_New(3);
    PyList_SetItem(slice0, 0, PyFloat_FromDouble(ryx[0]));
    PyList_SetItem(slice0, 1, PyFloat_FromDouble(ryx[1]));
    PyList_SetItem(slice0, 2, PyFloat_FromDouble(ryx[2]));
    PyList_SetItem(slice1, 0, PyFloat_FromDouble(ryx[3]));
    PyList_SetItem(slice1, 1, PyFloat_FromDouble(ryx[4]));
    PyList_SetItem(slice1, 2, PyFloat_FromDouble(ryx[5]));
    PyList_SetItem(slice2, 0, PyFloat_FromDouble(ryx[6]));
    PyList_SetItem(slice2, 1, PyFloat_FromDouble(ryx[7]));
    PyList_SetItem(slice2, 2, PyFloat_FromDouble(ryx[8]));

    PyList_SetItem(matrix, 0, slice0);
    PyList_SetItem(matrix, 1, slice1);
    PyList_SetItem(matrix, 2, slice2);

    return matrix;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisOffset_doc, "setAxisOffset(axisNum, axisOffset) \n\
\n\
Sets the offset value of one specific axis of this dataObject. \n\
\n\
Each axis in the data object can get a specific offset value, given in pixels. \n\
The offset value for one axis can be set by this method. Getting or setting \n\
single or all offset values for all axis can also be achieved by the attribute \n\
:attr:`axisOffsets`. \n\
\n\
The conversion between physical and pixel units is: \n\
\n\
``physical_value = (pixel_value - axisOffset) * axisScale`` \n\
\n\
Parameters  \n\
----------\n\
axisNum : int\n\
    The axis index in the range [0, n), where ``n`` is the dimension of this dataObject. \n\
axisOffset : float\n\
    New axis offset value in pixels. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the given ``axisNum`` is out of range. \n\
\n\
See Also \n\
-------- \n\
axisOffsets : this attribute can directly be used to get or set the axis offset(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisOffset(PyDataObject* self, PyObject* args)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    int axisnum;
    double axisOffset;

    if (!PyArg_ParseTuple(args, "id", &axisnum, &axisOffset))
    {
        return NULL;
    }

    if (self->dataObject->setAxisOffset(axisnum, axisOffset))
    {
        PyErr_SetString(PyExc_RuntimeError, "Set axis offset failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisScale_doc, "setAxisScale(axisNum, axisScale) \n\
\n\
Sets the scaling value of one specific axis of this dataObject. \n\
\n\
Each axis in the data object can get a specific scale value, given in ``axisUnits`` \n\
per pixels. The scale value for one axis can be set by this method. Getting or setting \n\
single or all scaling values for all axis can also be achieved by the attribute \n\
:attr:`axisScales`. \n\
\n\
The conversion between physical and pixel units is: \n\
\n\
``physical_value = (pixel_value - axisOffset) * axisScale`` \n\
\n\
Parameters \n\
----------\n\
axisNum : int\n\
    The axis index in the range [0, n), where ``n`` is the dimension of this dataObject. \n\
axisScale : float\n\
    New scale value for this axis in [unit/px]. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the given ``axisNum`` is out of range. \n\
\n\
See Also \n\
-------- \n\
axisScales : this attribute can directly be used to get or set the axis scale(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisScale(PyDataObject* self, PyObject* args)
{
    int axisnum;
    double axisscale;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "id", &axisnum, &axisscale))
    {
        return NULL;
    }

    if (self->dataObject->setAxisScale(axisnum, axisscale))
    {
        PyErr_SetString(PyExc_RuntimeError, "Set axis scale failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisDescription_doc, "setAxisDescription(axisNum, axisDescription) \n\
\n\
Sets the axis description of one axis. \n\
\n\
Each axis in the data object can get a specific description string (e.g. 'x-axis'). \n\
\n\
Parameters  \n\
----------\n\
axisNum : int\n\
    The axis index in the range [0, n), where ``n`` is the dimension of this dataObject. \n\
axisDescription : str\n\
    New axis description.\n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if the given ``axisNum`` is out of range \n\
\n\
See Also \n\
-------- \n\
axisDescriptions : this attribute can directly be used to get or set the axis description(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisDescription(PyDataObject* self, PyObject* args)
{
    int axisNum = 0;
    PyObject* tagvalue = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "iO", &axisNum, &tagvalue))
    {
        return NULL;
    }

    bool ok;
    std::string tagValString = PythonQtConversion::PyObjGetStdStringAsLatin1(tagvalue, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "axis description value is no string type.");
        return NULL;
    }

    if (self->dataObject->setAxisDescription(axisNum, tagValString))
    {
        PyErr_SetString(PyExc_RuntimeError, "set axis description failed");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisUnit_doc, "setAxisUnit(axisNum, axisUnit) \n\
\n\
Sets the unit of the specified axis. \n\
\n\
Each axis in the data object can get a specific unit string (e.g. 'mm'). \n\
\n\
Parameters  \n\
----------\n\
axisNum : int\n\
    The axis index in the range [0, n), where ``n`` is the dimension of this dataObject. \n\
axisUnit : str\n\
    New axis unit.\n\
\n\
Raises \n\
------ \n\
RuntimeError  \n\
    if the given ``axisNum`` is out of range. \n\
\n\
See Also \n\
-------- \n\
axisUnits : this attribute can directly be used to get or set the axis unit(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisUnit(PyDataObject* self, PyObject* args)
{
    int axisNum = 0;
    PyObject* tagvalue = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "iO", &axisNum, &tagvalue))
    {
        return NULL;
    }

    bool ok;
    std::string tagValString = PythonQtConversion::PyObjGetStdStringAsLatin1(tagvalue, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "axis unit value is no string type.");
        return NULL;
    }

    if (self->dataObject->setAxisUnit(axisNum, tagValString))
    {
        PyErr_SetString(PyExc_RuntimeError, "set axis unit failed");
    }
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectPhysToPix_doc, "physToPix(values, axes = 0) -> Union[float, Tuple[float]] \n\
\n\
Returns transformed values from physical to pixel coordinates. \n\
\n\
This method transforms a physical axis coordinate into its corresponding pixel \n\
coordinate. The transformation is defined by the current offset and scale value of \n\
the specific axis: \n\
\n\
``phys = (pix - offset) * scaling`` \n\
\n\
If no axes parameter is given, the values are assumed to belong the the ascending axis \n\
list (0, 1, 2, 3...). The returned pixel value is clipped by the real size of the data \n\
object in the requested dimension ``[0, shape[axis] - 1]``. \n\
\n\
Parameters  \n\
----------\n\
values : float or sequence of float\n\
    One single physical coordinate or a tuple of physical coordinates.\n\
axes : int or sequence of int, optional \n\
    If ``values`` is a single value, axes must be ``None`` or one integer, that defines \n\
    the axis for which the transformation should be calculated. \n\
    If ``values`` is a tuple of float values, axes can be one single value (all values \n\
    are transformed with respect to the same axis), or a tuple of int, whose size must be \n\
    equal to the size of the ``axes`` tuple. Each value is then transformed with the \n\
    corresponding value in ``axes``. \n\
    If ``None`` is given, ``axes`` is assumed to be an ascending list of values ``0, 1, 2, ...``. \n\
\n\
Returns \n\
------- \n\
float or tuple of float \n\
    The transformed physical coordinates for the given axes to pixel coordinates. \n\
\n\
Raises \n\
------ \n\
ValueError \n\
    if the given axes is out of range \n\
RuntimeWarning \n\
    if requested physical unit is outside of the range of the requested axis. \n\
    The returned pixel value is clipped to the closest boundary value.");
PyObject* PythonDataObject::PyDataObj_PhysToPix(PyDataObject* self, PyObject* args, PyObject* kwds)
{
    static const char* kwlist[] = {"values", "axes", NULL};
    double value;
    int axis = 0;
    bool axisScalar = false;
    PyObject* values = NULL;
    PyObject* axes = NULL;
    bool single = false;
    int dims = self->dataObject->getDims();

    PyErr_Clear();
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|i", const_cast<char**>(kwlist), &value, &axis))
    {
        single = true;
    }
    else if (
        PyErr_Clear(),
        !PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &values, &axes))
    {
        return NULL;
    }

    if (single)
    {
        if (dims <= axis || (axis < 0))
        {
            return PyErr_Format(
                PyExc_ValueError, "axis %i is out of bounds [0,%i]", axis, dims - 1);
        }
        else
        {
            return Py_BuildValue("d", self->dataObject->getPhysToPix(axis, value));
        }
    }
    else
    {
        PyObject* valuesSeq =
            PySequence_Fast(values, "values must be a float value or a sequence of floats.");

        if (!valuesSeq)
        {
            return NULL;
        }

        PyObject* axesSeq = NULL;

        if (axes)
        {
            if (PyLong_Check(axes))
            {
                axis = PyLong_AsLong(axes);
                axisScalar = true;
            }
            else
            {
                axesSeq = PySequence_Fast(
                    axes, "axes must be an integer value or a sequence of integers.");

                if (!axesSeq)
                {
                    Py_XDECREF(valuesSeq);
                    return NULL;
                }

                if (PySequence_Length(valuesSeq) != PySequence_Length(axes))
                {
                    Py_XDECREF(valuesSeq);
                    PyErr_SetString(
                        PyExc_ValueError,
                        "values and axes must have the same size or axes has to be a scalar "
                        "integer value.");
                    return NULL;
                }
            }
        }

        PyObject* v = NULL;
        PyObject* a = NULL;
        PyObject* result = PyTuple_New(PySequence_Length(valuesSeq));
        bool isInsideImage;

        for (Py_ssize_t i = 0; i < PySequence_Length(valuesSeq); ++i)
        {
            v = PySequence_Fast_GET_ITEM(valuesSeq, i); // borrowed
            if (axesSeq)
            {
                a = PySequence_Fast_GET_ITEM(axesSeq, i); // borrowed
            }

            if (PyFloat_Check(v))
            {
                value = PyFloat_AsDouble(v);
            }
            else if (PyLong_Check(v))
            {
                value = PyLong_AsLong(v);
            }
            else
            {
                Py_DECREF(result);
                Py_XDECREF(axesSeq);
                Py_XDECREF(valuesSeq);
                return PyErr_Format(
                    PyExc_ValueError, "%i. value cannot be interpreted as float.", i);
            }

            if (a)
            {
                if (PyLong_Check(a))
                {
                    axis = PyLong_AsLong(a);
                }
                else
                {
                    Py_DECREF(result);
                    Py_XDECREF(axesSeq);
                    Py_XDECREF(valuesSeq);
                    return PyErr_Format(
                        PyExc_ValueError, "%i. axis cannot be interpreted as integer.", i);
                }
            }
            else if (!axisScalar)
            {
                axis = i;
            }

            if (axis < 0 || axis >= dims)
            {
                Py_DECREF(result);
                Py_XDECREF(axesSeq);
                Py_XDECREF(valuesSeq);
                return PyErr_Format(
                    PyExc_ValueError, "%i. axis index out of range [0,%i]", i, dims - 1);
            }

            PyTuple_SetItem(
                result,
                i,
                PyFloat_FromDouble(self->dataObject->getPhysToPix(axis, value, isInsideImage)));

            if (!isInsideImage)
            {
                if (PyErr_WarnFormat(
                        PyExc_RuntimeWarning,
                        1,
                        "the returned pixel for axis %i is clipped to the boundaries of the axis "
                        "[0,%i]",
                        axis,
                        self->dataObject->getSize(axis) - 1) == -1)
                {
                    Py_DECREF(result);
                    Py_XDECREF(axesSeq);
                    Py_XDECREF(valuesSeq);
                    return NULL; // warning was turned into a real exception,
                }
                // else
                //{
                // warning is a warning, go on with the script
                //}
            }
        }

        Py_XDECREF(axesSeq);
        Py_XDECREF(valuesSeq);

        return result;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectPixToPhys_doc, "pixToPhys(values, axes = 0) -> Union[float, Tuple[float]] \n\
\n\
Returns transformed values from pixel to physical coordinates. \n\
\n\
This method transforms a pixel axis coordinate into its corresponding physical \n\
coordinate. The transformation is defined by the current offset and scale value of \n\
the specific axis: \n\
\n\
``pix = (phys / scaling) + offset`` \n\
\n\
If no axes parameter is given, the values are assumed to belong the the ascending axis \n\
list (0, 1, 2, 3...). The returned pixel value is clipped by the real size of the data \n\
object in the requested dimension ``[0, shape[axis] - 1]``. \n\
\n\
Parameters  \n\
----------\n\
values : float or sequence of float\n\
    One single pixel coordinate or a tuple of pixel coordinates.\n\
axes : int or sequence of int, optional \n\
    If ``values`` is a single value, axes must be ``None`` or one integer, that defines \n\
    the axis for which the transformation should be calculated. \n\
    If ``values`` is a tuple of float values, axes can be one single value (all values \n\
    are transformed with respect to the same axis), or a tuple of int, whose size must be \n\
    equal to the size of the ``axes`` tuple. Each value is then transformed with the \n\
    corresponding value in ``axes``. \n\
    If ``None`` is given, ``axes`` is assumed to be an ascending list of values ``0, 1, 2, ...``. \n\
\n\
Returns \n\
------- \n\
float or tuple of float \n\
    The transformed pixel coordinates for the given axes to physical coordinates. \n\
\n\
Raises \n\
------ \n\
ValueError \n\
    if the given axes is out of range.");
PyObject* PythonDataObject::PyDataObj_PixToPhys(PyDataObject* self, PyObject* args, PyObject* kwds)
{
    static const char* kwlist[] = {"values", "axes", NULL};
    double value;
    int axis = 0;
    bool axisScalar = false;
    PyObject* values = NULL;
    PyObject* axes = NULL;
    bool single = false;
    int dims = self->dataObject->getDims();

    PyErr_Clear();
    if (PyArg_ParseTupleAndKeywords(args, kwds, "d|i", const_cast<char**>(kwlist), &value, &axis))
    {
        single = true;
    }
    else if (
        PyErr_Clear(),
        !PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char**>(kwlist), &values, &axes))
    {
        return NULL;
    }

    if (single)
    {
        if (dims <= axis || (axis < 0))
        {
            return PyErr_Format(
                PyExc_ValueError, "axis %i is out of bounds [0,%i]", axis, dims - 1);
        }
        else
        {
            return Py_BuildValue("d", self->dataObject->getPixToPhys(axis, value));
        }
    }
    else
    {
        PyObject* valuesSeq =
            PySequence_Fast(values, "values must be a float value or a sequence of floats.");
        if (!valuesSeq)
        {
            return NULL;
        }

        PyObject* axesSeq = NULL;

        if (axes)
        {
            if (PyLong_Check(axes))
            {
                axis = PyLong_AsLong(axes);
                axisScalar = true;
            }
            else
            {
                axesSeq = PySequence_Fast(
                    axes, "axes must be an integer value or a sequence of integers.");

                if (!axesSeq)
                {
                    Py_XDECREF(valuesSeq);
                    return NULL;
                }

                if (PySequence_Length(valuesSeq) != PySequence_Length(axes))
                {
                    Py_XDECREF(valuesSeq);
                    PyErr_SetString(
                        PyExc_ValueError,
                        "values and axes must have the same size or axes has to be a scalar "
                        "integer value.");
                    return NULL;
                }
            }
        }

        PyObject* v = NULL;
        PyObject* a = NULL;
        PyObject* result = PyTuple_New(PySequence_Length(valuesSeq));

        for (Py_ssize_t i = 0; i < PySequence_Length(valuesSeq); ++i)
        {
            v = PySequence_Fast_GET_ITEM(valuesSeq, i); // borrowed
            if (axesSeq)
            {
                a = PySequence_Fast_GET_ITEM(axesSeq, i); // borrowed
            }

            if (PyFloat_Check(v))
            {
                value = PyFloat_AsDouble(v);
            }
            else if (PyLong_Check(v))
            {
                value = PyLong_AsLong(v);
            }
            else
            {
                Py_XDECREF(valuesSeq);
                Py_XDECREF(axesSeq);
                Py_DECREF(result);
                return PyErr_Format(
                    PyExc_ValueError, "%i. value cannot be interpreted as float", i);
            }

            if (a)
            {
                if (PyLong_Check(a))
                {
                    axis = PyLong_AsLong(a);
                }
                else
                {
                    Py_XDECREF(valuesSeq);
                    Py_XDECREF(axesSeq);
                    Py_DECREF(result);
                    return PyErr_Format(
                        PyExc_ValueError, "%i. axis cannot be interpreted as integer", i);
                }
            }
            else if (!axisScalar)
            {
                axis = i;
            }

            if (axis < 0 || axis >= dims)
            {
                Py_XDECREF(valuesSeq);
                Py_XDECREF(axesSeq);
                Py_DECREF(result);
                return PyErr_Format(
                    PyExc_ValueError, "%i. axis index out of range [0,%i]", i, dims - 1);
            }

            PyTuple_SetItem(
                result, i, PyFloat_FromDouble(self->dataObject->getPixToPhys(axis, value)));
        }

        Py_XDECREF(valuesSeq);
        Py_XDECREF(axesSeq);

        return result;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetTag_doc, "setTag(key, value) \n\
\n\
Set the ``value`` of a tag with the given ``key`` name. \n\
\n\
If a tag with the given ``key`` exists, its value is overwritten. Else, a tag with \n\
that ``key`` is added to the tags. \n\
\n\
Parameters  \n\
----------\n\
key : str\n\
    the name of the tag.\n\
tagvalue : str or float\n\
    The new value of the tag. Must be a :obj:`str` or a :obj:`float` value. \n\
\n\
Notes \n\
----- \n\
Do NOT use 'special character' within the tag key because they are not XML-save.");
PyObject* PythonDataObject::PyDataObj_SetTag(PyDataObject* self, PyObject* args)
{
    const char* tagName = NULL;
    PyObject* tagvalue = NULL;
    double tagvalueD = 0;
    bool dType = true;

    if (!PyArg_ParseTuple(args, "sd", &tagName, &tagvalueD))
    {
        PyErr_Clear();
        dType = false;

        if (!PyArg_ParseTuple(args, "sO", &tagName, &tagvalue))
        {
            return NULL;
        }
    }

    if (self->dataObject->getDims() == 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "an empty dataObject cannot have any tags.");
        return NULL;
    }

    std::string tagNameString(tagName);

    if (dType)
    {
        if (self->dataObject->setTag(tagNameString, tagvalueD))
        {
            PyErr_SetString(PyExc_RuntimeError, "set tag value as double failed");
            return NULL;
        }
    }
    else
    {
        bool ok;
        std::string tagValString =
            PythonQtConversion::PyObjGetStdStringAsLatin1(tagvalue, true, ok);

        if (!ok)
        {
            PyErr_SetString(PyExc_TypeError, "unit value is no string type.");
            return NULL;
        }

        if (self->dataObject->setTag(tagNameString, tagValString))
        {
            PyErr_SetString(PyExc_RuntimeError, "set tag value string failed");
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectDeleteTag_doc, "deleteTag(key) -> bool \n\
\n\
Deletes a tag specified by ``key`` from the tag dictionary. \n\
\n\
Parameters  \n\
----------\n\
key : str\n\
    the name of the tag to be deleted.\n\
\n\
Returns \n\
------- \n\
success : bool \n\
    ``True`` if tag with given key existed and could be deleted, otherwise ``False``.");
PyObject* PythonDataObject::PyDataObj_DeleteTag(PyDataObject* self, PyObject* args)
{
    // int length = PyTuple_Size(args);
    const char* tagName = NULL;

    if (!PyArg_ParseTuple(args, "s", &tagName))
    {
        return NULL;
    }

    std::string tagNameString(tagName);
    if (self->dataObject->deleteTag(tagNameString))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectTagExists_doc, "existTag(key) -> bool \n\
\n\
Checks if a certain tag key exists. \n\
\n\
Checks whether a tag with the given ``key`` exists in tag dictionary of this \n\
data object and returns ``True`` if such a tag exists, else ``False``. \n\
\n\
Parameters  \n\
----------\n\
key : str\n\
    the key of the tag.\n\
\n\
Returns \n\
------- \n\
bool\n\
    ``True`` if tag exists, else ``False``");
PyObject* PythonDataObject::PyDataObj_TagExists(PyDataObject* self, PyObject* args)
{
    //    int length = PyTuple_Size(args);
    const char* tagName = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &tagName))
    {
        return NULL;
    }

    std::string tagNameString(tagName);
    if (self->dataObject->existTag(tagNameString))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    };
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectGetTagListSize_doc, "getTagListSize() -> int \n\
\n\
Returns the number of tags in the tag dictionary.\n\
\n\
Every data object can have an arbitrary number of tags stored in the tag dictionary. \n\
This method returns the number of different tags, where the protocol is also one \n\
tag with the key ``protocol``. \n\
\n\
Returns \n\
------- \n\
length : int \n\
    Size of the tag dictionary. The optional protocol also counts as one item.");
PyObject* PythonDataObject::PyDataObj_GetTagListSize(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }
    return PyLong_FromLong(self->dataObject->getTagListSize());
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAddToProtocol_doc, "addToProtocol(newLine) \n\
\n\
Appends one string entry to the protocol list. \n\
\n\
Appends a line of text to the protocol string of this data object. \n\
If this data object has got a region of interest defined, the rectangle of the ROI is \n\
automatically appended to ``newLine``. The protocol string ends with a newline character. \n\
\n\
Address the content of the protocol by ``obj.tags[\"protocol\"]``. The protocol is \n\
contained in the ordinary tag dictionary of this data object under the key ``protocol``. \n\
\n\
Parameters  \n\
----------\n\
newLine : str\n\
    The text to be added to the protocol.");
PyObject* PythonDataObject::PyDataObj_AddToProtocol(PyDataObject* self, PyObject* args)
{
    PyObject* unit = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }
    else if (self->dataObject->getDims() == 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "an empty dataObject cannot have a protocol.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O", &unit))
    {
        return NULL;
    }

    bool ok;
    std::string unitString = PythonQtConversion::PyObjGetStdStringAsLatin1(unit, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "unit value is no string type.");
        return NULL;
    }

    if (self->dataObject->addToProtocol(unitString))
    {
        PyErr_SetString(PyExc_RuntimeError, "Add line to protocol unit failed");
        return NULL;
    }

    Py_RETURN_NONE;
}
// Tag information functions

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_RichCompare(
    PyDataObject* self, PyObject* other, int cmp_op)
{
    if (self->dataObject == nullptr)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return nullptr;
    }

    if (other == nullptr)
    {
        PyErr_SetString(PyExc_TypeError, "compare object is empty.");
        return nullptr;
    }

    // check type of other
    PyDataObject* otherDataObj = nullptr;
    ito::DataObject resDataObj;
    PyDataObject* resultObject = nullptr;

    if (PyDataObject_Check(other))
    {
        otherDataObj = (PyDataObject*)(other);
        if (otherDataObj->dataObject == nullptr)
        {
            PyErr_SetString(PyExc_TypeError, "internal data object of compare object is empty.");
            return nullptr;
        }

        try
        {
            switch (cmp_op)
            {
            case Py_LT:
                resDataObj = *(self->dataObject) < *(otherDataObj->dataObject);
                break;
            case Py_LE:
                resDataObj = *(self->dataObject) <= *(otherDataObj->dataObject);
                break;
            case Py_EQ:
                resDataObj = *(self->dataObject) == *(otherDataObj->dataObject);
                break;
            case Py_NE:
                resDataObj = *(self->dataObject) != *(otherDataObj->dataObject);
                break;
            case Py_GT:
                resDataObj = *(self->dataObject) > *(otherDataObj->dataObject);
                break;
            case Py_GE:
                resDataObj = *(self->dataObject) >= *(otherDataObj->dataObject);
                break;
            }
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return nullptr;
        }

        resultObject = createEmptyPyDataObject();
        // resDataObj should always be the owner of its data,
        // therefore base of resultObject remains None
        resultObject->dataObject = new ito::DataObject(resDataObj);
        return (PyObject*)resultObject;
    }
    else if (PyFloat_Check(other) || PyLong_Check(other))
    {
        double value = PyFloat_AsDouble(other);
        if (!PyErr_Occurred())
        {
            try
            {
                switch (cmp_op)
                {
                case Py_LT:
                    resDataObj = *(self->dataObject) < value;
                    break;
                case Py_LE:
                    resDataObj = *(self->dataObject) <= value;
                    break;
                case Py_EQ:
                    resDataObj = *(self->dataObject) == value;
                    break;
                case Py_NE:
                    resDataObj = *(self->dataObject) != value;
                    break;
                case Py_GT:
                    resDataObj = *(self->dataObject) > value;
                    break;
                case Py_GE:
                    resDataObj = *(self->dataObject) >= value;
                    break;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            resultObject = createEmptyPyDataObject();
            // resDataObj should always be the owner of its
            // data, therefore base of resultObject remains None
            resultObject->dataObject = new ito::DataObject(resDataObj);
            return (PyObject*)resultObject;
        }
        else
        {
            return nullptr;
        }
    }
    else if (PyComplex_Check(other))
    {
        if (!PyErr_Occurred())
        {
            ito::complex128 cmplxValue =
                ito::complex128(PyComplex_AsCComplex(other).real, PyComplex_AsCComplex(other).imag);
            try
            {
                switch (cmp_op)
                {
                case Py_EQ:
                    resDataObj = *(self->dataObject) == cmplxValue;
                    break;
                case Py_NE:
                    resDataObj = *(self->dataObject) != cmplxValue;
                    break;
                default:
                    PyErr_SetString(
                        PyExc_TypeError,
                        "Not a valid operation for complex values (not orderable, use real, imag, "
                        "or abs).");
                    return NULL;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            resultObject = createEmptyPyDataObject();
            // resDataObj should always be the owner of its
            // data, therefore base of resultObject remains None
            resultObject->dataObject = new ito::DataObject(resDataObj);
            return (PyObject*)resultObject;
        }
        else
        {
            return nullptr;
        }
    }
    else if (PyRgba_Check(other))
    {
        if (!PyErr_Occurred())
        {
            const auto* color = (PythonRgba::PyRgba*)(other);
            try
            {
                switch (cmp_op)
                {
                case Py_EQ:
                    resDataObj = *(self->dataObject) == color->rgba;
                    break;
                case Py_NE:
                    resDataObj = *(self->dataObject) != color->rgba;
                    break;
                default:
                    PyErr_SetString(
                        PyExc_TypeError,
                        "Not a valid operation for rgba32 values (not orderable).");
                    return nullptr;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            resultObject = createEmptyPyDataObject();
            // resDataObj should always be the owner of its
            // data, therefore base of resultObject remains None
            resultObject->dataObject = new ito::DataObject(resDataObj);
            return (PyObject*)resultObject;
        }
        else
        {
            return nullptr;
        }
    }
    else if (PythonDateTime::PyDateTime_CheckExt(other))
    {
        if (!PyErr_Occurred())
        {
            bool ok;
            const auto& value = PythonDateTime::GetDateTime(other, ok);

            try
            {
                switch (cmp_op)
                {
                case Py_LT:
                    resDataObj = *(self->dataObject) < value;
                    break;
                case Py_LE:
                    resDataObj = *(self->dataObject) <= value;
                    break;
                case Py_EQ:
                    resDataObj = *(self->dataObject) == value;
                    break;
                case Py_NE:
                    resDataObj = *(self->dataObject) != value;
                    break;
                case Py_GT:
                    resDataObj = *(self->dataObject) > value;
                    break;
                case Py_GE:
                    resDataObj = *(self->dataObject) >= value;
                    break;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return nullptr;
            }

            resultObject = createEmptyPyDataObject();
            // resDataObj should always be the owner of its
            // data, therefore base of resultObject remains None
            resultObject->dataObject = new ito::DataObject(resDataObj);
            return (PyObject*)resultObject;
        }
        else
        {
            return nullptr;
        }
    }
    else if (PythonDateTime::PyTimeDelta_CheckExt(other))
    {
        if (!PyErr_Occurred())
        {
            bool ok;
            const auto& value = PythonDateTime::GetTimeDelta(other, ok);

            try
            {
                switch (cmp_op)
                {
                case Py_LT:
                    resDataObj = *(self->dataObject) < value;
                    break;
                case Py_LE:
                    resDataObj = *(self->dataObject) <= value;
                    break;
                case Py_EQ:
                    resDataObj = *(self->dataObject) == value;
                    break;
                case Py_NE:
                    resDataObj = *(self->dataObject) != value;
                    break;
                case Py_GT:
                    resDataObj = *(self->dataObject) > value;
                    break;
                case Py_GE:
                    resDataObj = *(self->dataObject) >= value;
                    break;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return nullptr;
            }

            resultObject = createEmptyPyDataObject();
            // resDataObj should always be the owner of its
            // data, therefore base of resultObject remains None
            resultObject->dataObject = new ito::DataObject(resDataObj);
            return (PyObject*)resultObject;
        }
        else
        {
            return nullptr;
        }
    }
    else
    {
        PyErr_SetString(
            PyExc_TypeError,
            "second argument of comparison operator is no dataObject or scalar value.");
        return nullptr;
    }
}

//-------------------------------------------------------------------------------------
PythonDataObject::PyDataObject* PythonDataObject::createEmptyPyDataObject()
{
    PyDataObject* result =
        (PyDataObject*)PyObject_Call((PyObject*)&PyDataObjectType, nullptr, nullptr);
    if (result != nullptr)
    {
        DELETE_AND_SET_NULL(result->dataObject);
        return result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return nullptr;
    }
}

//-------------------------------------------------------------------------------------
// returns NULL with set Python exception if npArray could not be converted to data
// object. If everything ok, returns a new reference of the PyDataObject
/*static*/ PyObject* PythonDataObject::createPyDataObjectFromArray(PyObject* npArray)
{
    PyObject* args = Py_BuildValue("(O)", npArray);
    ito::PythonDataObject::PyDataObject* result =
        (ito::PythonDataObject::PyDataObject*)PyObject_Call(
            (PyObject*)&ito::PythonDataObject::PyDataObjectType, args, NULL); // new reference
    Py_DECREF(args);
    return (PyObject*)result;
}

//-------------------------------------------------------------------------------------
bool PythonDataObject::checkPyDataObject(
    int number, PyObject* o1 /*= NULL*/, PyObject* o2 /*= NULL*/, PyObject* o3 /*= NULL*/)
{
    PyObject* temp;
    for (int i = 0; i < number; ++i)
    {
        switch (i)
        {
        case 0:
            temp = o1;
            break;
        case 1:
            temp = o2;
            break;
        case 2:
            temp = o3;
            break;
        default:
            continue;
        }

        if (temp == nullptr)
        {
            PyErr_Format(PyExc_TypeError, "%i. operand is NULL", i + 1);
            return false;
        }
        else if (!PyDataObject_Check(temp) || ((PyDataObject*)(temp))->dataObject == nullptr)
        {
            PyErr_Format(PyExc_TypeError, "%i. operand must be a valid data object", i);
            return false;
        }
    }
    return true;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbAdd(PyObject* o1, PyObject* o2)
{
    PyDataObject* dobj1 = nullptr;
    PyObject* obj2 = nullptr;

    if (PyDataObject_Check(o1))
    {
        dobj1 = (PyDataObject*)o1;
        obj2 = o2;
    }
    else if (PyDataObject_Check(o2))
    {
        dobj1 = (PyDataObject*)o2;
        obj2 = o1;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "at least one operand must be a dataObject");
        return nullptr;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    if (PyDataObject_Check(obj2))
    {
        try
        {
            // resDataObj should always be the owner of its data,
            // therefore base of resultObject remains None
            retObj->dataObject =
                new ito::DataObject(*(dobj1->dataObject) + *(((PyDataObject*)obj2)->dataObject));
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return nullptr;
        }

        retObj->dataObject->addToProtocol("Created by adding two dataObjects.");
    }
    else if (PyFloat_Check(o2) || PyLong_Check(o2))
    {
        double scalar = PyFloat_AsDouble(o2);

        try
        {
            // resDataObj should always be the owner of its data,
            // therefore base of resultObject remains None
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) + scalar);
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return nullptr;
        }

        retObj->dataObject->addToProtocol(string_format("Added %g scalar to dataObject.", scalar));
    }
    else if (PyComplex_Check(o2))
    {
        auto cscalar = ito::complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

        try
        {
            // resDataObj should always be the owner of its data,
            // therefore base of resultObject remains None
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) + cscalar);
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return nullptr;
        }

        if (cscalar.imag() > 0)
        {
            retObj->dataObject->addToProtocol(string_format(
                "Added %g+i%g scalar to dataObject.", cscalar.real(), cscalar.imag()));
        }
        else
        {
            retObj->dataObject->addToProtocol(string_format(
                "Added %g-i%g scalar to dataObject.", cscalar.real(), cscalar.imag()));
        }
    }
    else if (PythonDateTime::PyTimeDelta_CheckExt(o2))
    {
        bool ok;
        auto timedelta = PythonDateTime::GetTimeDelta(o2, ok);

        if (!ok)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_RuntimeError, "Timedelta value cannot be parsed.");
            return nullptr;
        }

        try
        {
            // resDataObj should always be the owner of its data,
            // therefore base of resultObject remains None
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) + timedelta);
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return nullptr;
        }

        retObj->dataObject->addToProtocol("timedelta added to dataObject");
    }
    else
    {
        Py_DECREF(retObj);
        PyErr_SetString(
            PyExc_RuntimeError,
            "Only a dataObject, integer, float, complex or timedelta value can be added to a "
            "dataObject.");
        return nullptr;
    }

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbSubtract(PyObject* o1, PyObject* o2)
{
    PyDataObject* dobj1 = NULL;
    PyDataObject* dobj2 = NULL;
    ito::float64 scalar = 0;
    ito::complex128 cscalar = 0;
    bool doneScalar = false;
    bool complexScalar = false;

    if (PyDataObject_Check(o1) && PyDataObject_Check(o2))
    {
        dobj1 = (PyDataObject*)o1;
        dobj2 = (PyDataObject*)o2;
    }
    else if (PyDataObject_Check(o1))
    {
        dobj1 = (PyDataObject*)o1;
        if (PyFloat_Check(o2) || PyLong_Check(o2))
        {
            scalar = PyFloat_AsDouble(o2);
        }
        else if (PyComplex_Check(o2))
        {
            cscalar = ito::complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));
            complexScalar = true;
        }
        else
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "second operand must be a dataObject, integer, float or complex");
            return NULL;
        }
    }
    else if (PyDataObject_Check(o2))
    {
        dobj2 = (PyDataObject*)o2;
        if (PyFloat_Check(o1) || PyLong_Check(o1))
        {
            scalar = PyFloat_AsDouble(o1);
        }
        else if (PyComplex_Check(o2))
        {
            cscalar = ito::complex128(PyComplex_RealAsDouble(o1), PyComplex_ImagAsDouble(o1));
            complexScalar = true;
        }
        else
        {
            PyErr_SetString(
                PyExc_RuntimeError,
                "first operand must be a dataObject, integer, float or complex");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "at least one operand must be a dataObject");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        if (dobj1 && dobj2)
        {
            retObj->dataObject = new ito::DataObject(
                *(dobj1->dataObject) -
                *(dobj2->dataObject)); // resDataObj should always be the owner of its data,
                                       // therefore base of resultObject remains None
        }
        else if (dobj1)
        {
            if (complexScalar)
            {
                doneScalar = true;
                retObj->dataObject = new ito::DataObject(
                    *(dobj1->dataObject) -
                    cscalar); // resDataObj should always be the owner of its data, therefore base
                              // of resultObject remains None
            }
            else
            {
                doneScalar = true;
                retObj->dataObject = new ito::DataObject(
                    *(dobj1->dataObject) -
                    scalar); // resDataObj should always be the owner of its data, therefore base of
                             // resultObject remains None
            }
        }
        else
        {
            if (complexScalar)
            {
                doneScalar = true;
                // this step is necessary in order to allow e.g. 255 - (uint8dataobject) without
                // buffer overflows.
                retObj->dataObject =
                    new ito::DataObject(dobj2->dataObject->getSize(), dobj2->dataObject->getType());
                retObj->dataObject->setTo(complexScalar);
                *(retObj->dataObject) -=
                    *(dobj2->dataObject); // resDataObj should always be the owner of its data,
                                          // therefore base of resultObject remains None
            }
            else
            {
                doneScalar = true;
                // this step is necessary in order to allow e.g. 255 - (uint8dataobject) without
                // buffer overflows.
                retObj->dataObject =
                    new ito::DataObject(dobj2->dataObject->getSize(), dobj2->dataObject->getType());
                retObj->dataObject->setTo(scalar);

                // resDataObj should always be the owner of its data,
                // therefore base of resultObject remains None
                *(retObj->dataObject) -= *(dobj2->dataObject);
            }
        }
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (doneScalar)
    {
        char buf[PROTOCOL_STR_LENGTH] = {0};
        if (complexScalar)
        {
            if (cscalar.imag() > 0)
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Subtracted %g+i%g scalar to dataObject.",
                    cscalar.real(),
                    cscalar.imag());
            }
            else
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Subtracted %g-i%g scalar to dataObject.",
                    cscalar.real(),
                    -cscalar.imag());
            }
        }
        else
        {
            sprintf_s(buf, PROTOCOL_STR_LENGTH, "Subtracted %g scalar to dataObject.", scalar);
        }
        if (retObj)
            retObj->dataObject->addToProtocol(buf);
    }
    else
    {
        if (retObj)
            retObj->dataObject->addToProtocol("Created by subtracting two dataObjects.");
    }

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbMultiply(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (Py_TYPE(o1) == &PyDataObjectType && Py_TYPE(o2) == &PyDataObjectType)
    {
        PyDataObject* dobj1 = (PyDataObject*)(o1);
        PyDataObject* dobj2 = (PyDataObject*)(o2);

        PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

        try
        {
            // resDataObj should always be the owner of its data, therefore base of resultObject
            // remains None
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) * *(dobj2->dataObject));
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        if (retObj)
            retObj->dataObject->addToProtocol("Multiplication of two dataObjects.");
        return (PyObject*)retObj;
    }
    else if (Py_TYPE(o1) == &PyDataObjectType)
    {
        PyDataObject* dobj1 = (PyDataObject*)(o1);

        if (PyComplex_Check(o2))
        {
            complex128 factor = complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

            PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

            try
            {
                // resDataObj should always be the owner of its data, therefore base of resultObject
                // remains None
                retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) * factor);
            }
            catch (cv::Exception& exc)
            {
                Py_DECREF(retObj);
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            char buf[PROTOCOL_STR_LENGTH] = {0};

            if (factor.imag() > 0)
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Multiplied dataObject with %g+i%g.",
                    factor.real(),
                    factor.imag());
            }
            else
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Multiplied dataObject with %g-i%g.",
                    factor.real(),
                    -factor.imag());
            }

            if (retObj)
                retObj->dataObject->addToProtocol(buf);

            return (PyObject*)retObj;
        }
        else
        {
            double factor = PyFloat_AsDouble((PyObject*)o2);

            if (PyErr_Occurred())
            {
                return NULL;
            }

            PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

            try
            {
                // resDataObj should always be the owner of its data, therefore base of resultObject
                // remains None
                retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) * factor);
            }
            catch (cv::Exception& exc)
            {
                Py_DECREF(retObj);
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            char buf[PROTOCOL_STR_LENGTH] = {0};
            sprintf_s(buf, PROTOCOL_STR_LENGTH, "Multiplied dataObject with %g.", factor);

            if (retObj)
                retObj->dataObject->addToProtocol(buf);

            return (PyObject*)retObj;
        }
    }
    else if (Py_TYPE(o2) == &PyDataObjectType)
    {
        double factor = PyFloat_AsDouble((PyObject*)o1);
        PyDataObject* dobj2 = (PyDataObject*)(o2);

        if (PyErr_Occurred())
        {
            return NULL;
        }

        PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

        try
        {
            // resDataObj should always be the owner of its data, therefore base of resultObject
            // remains None
            retObj->dataObject = new ito::DataObject(*(dobj2->dataObject) * factor);
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        sprintf_s(buf, PROTOCOL_STR_LENGTH, "Multiplied dataObject scalar with %g.", factor);

        if (retObj)
            retObj->dataObject->addToProtocol(buf);

        return (PyObject*)retObj;
    }
    return NULL;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbMatrixMultiply(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // resDataObj should always be the owner of its data, therefore base of resultObject remains
        // None
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) * *(dobj2->dataObject));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
    {
        retObj->dataObject->addToProtocol("Matrix multiplication of two dataObjects.");
    }

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbDivide(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (Py_TYPE(o2) == &PyDataObjectType)
    {
        PyErr_SetString(PyExc_RuntimeError, "division by a dataObject not implemented.");
        return NULL;
    }
    if (Py_TYPE(o1) == &PyDataObjectType)
    {
        PyDataObject* dobj1 = (PyDataObject*)(o1);

        if (PyComplex_Check(o2))
        {
            complex128 factor = complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

            PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

            try
            {
                retObj->dataObject = new ito::DataObject(
                    *(dobj1->dataObject) *
                    (complex128(1.0, 0.0) /
                     factor)); // resDataObj should always be the owner of its data, therefore base
                               // of resultObject remains None
            }
            catch (cv::Exception& exc)
            {
                Py_DECREF(retObj);
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            char buf[PROTOCOL_STR_LENGTH] = {0};
            if (factor.imag() > 0)
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Divided dataObject by %g+i%g.",
                    factor.real(),
                    factor.imag());
            }
            else
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Divided dataObject by %g-i%g.",
                    factor.real(),
                    -factor.imag());
            }

            if (retObj)
                retObj->dataObject->addToProtocol(buf);

            return (PyObject*)retObj;
        }
        else
        {
            double factor = PyFloat_AsDouble((PyObject*)o2);

            if (PyErr_Occurred())
            {
                return NULL;
            }

            PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

            try
            {
                retObj->dataObject = new ito::DataObject(
                    *(dobj1->dataObject) *
                    (1.0 / factor)); // resDataObj should always be the owner of its data, therefore
                                     // base of resultObject remains None
            }
            catch (cv::Exception& exc)
            {
                Py_DECREF(retObj);
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            char buf[PROTOCOL_STR_LENGTH] = {0};
            sprintf_s(buf, PROTOCOL_STR_LENGTH, "Divided dataObject by %g.", factor);

            if (retObj)
                retObj->dataObject->addToProtocol(buf);

            return (PyObject*)retObj;
        }
    }

    return NULL;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbRemainder(PyObject* /*o1*/, PyObject* /*o2*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbDivmod(PyObject* /*o1*/, PyObject* /*o2*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbPower(PyObject* o1, PyObject* o2, PyObject* o3)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    if (o3 != Py_None)
    {
        PyErr_SetString(PyExc_TypeError, "Modulo in power-method not supported");
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    ito::float64 power;

    if (PyLong_Check(o2))
    {
        power = PyLong_AsLong(o2);
    }
    else if (PyFloat_Check(o2))
    {
        power = PyFloat_AsDouble(o2);
    }
    else
    {
        PyErr_SetString(
            PyExc_TypeError, "2nd operand of power-method must be an integer of float.");
        return NULL;
    }
    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject();
        dobj1->dataObject->pow(power, *retObj->dataObject);
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj && retObj->dataObject)
        retObj->dataObject->addToProtocol("Created by dataObject ** power");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbNegative(PyObject* o1)
{
    if (!checkPyDataObject(1, o1))
    {
        return nullptr;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // resDataObj should always be the owner of its data,
        // therefore base of resultObject remains None
        retObj->dataObject = new ito::DataObject((*(dobj1->dataObject) * -1.0));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
        retObj->dataObject->addToProtocol(
            "Created by scalar multiplication of dataObject with -1.0.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbPositive(PyObject* o1)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject));

        if (!retObj->dataObject->getOwnData())
        {
            PyDataObject_SetBase(retObj, (PyObject*)o1);
        }
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Created by python function positive.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbAbsolute(PyObject* o1)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            ito::abs(*(dobj1->dataObject))); // resDataObj should always be the owner of its data,
                                             // therefore base of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject->addToProtocol("Absolute values of calculated via abs(dataObject).");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInvert(PyObject* o1)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // resDataObj should always be the owner of its data,
        // therefore base of resultObject remains None
        retObj->dataObject = new ito::DataObject(dobj1->dataObject->bitwise_not());
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject->addToProtocol("Bitwise inversion.");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbLshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    int overflow;
    int shift = PyLong_AsLongAndOverflow(o2, &overflow);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    if (overflow)
    {
        PyErr_SetString(PyExc_ValueError, "shift value exceeds integer range");
        return NULL;
    }

    if (shift < 0)
    {
        PyErr_SetString(PyExc_TypeError, "shift value must not be negative");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // resDataObj should always be the owner of its data, therefore base of
        // resultObject remains None
        retObj->dataObject =
            new ito::DataObject(*(dobj1->dataObject) << static_cast<unsigned int>(shift));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    char buf[PROTOCOL_STR_LENGTH] = {0};
    sprintf_s(buf, PROTOCOL_STR_LENGTH, "Left shift by %i on dataObject.", shift);

    retObj->dataObject->addToProtocol(buf);
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbRshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    int overflow;
    int shift = PyLong_AsLongAndOverflow(o2, &overflow);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    if (overflow)
    {
        PyErr_SetString(PyExc_ValueError, "shift value exceeds integer range");
        return NULL;
    }

    if (shift < 0)
    {
        PyErr_SetString(PyExc_TypeError, "shift value must not be negative");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // resDataObj should always be the owner of its data,
        // therefore base of resultObject remains None
        retObj->dataObject =
            new ito::DataObject(*(dobj1->dataObject) >> static_cast<unsigned int>(shift));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    char buf[PROTOCOL_STR_LENGTH] = {0};
    sprintf_s(buf, PROTOCOL_STR_LENGTH, "Right shift by %i on dataObject.", shift);

    retObj->dataObject->addToProtocol(buf);
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbAnd(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // resDataObj should always be the owner of its data, therefore
        // base of resultObject remains None
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) & *(dobj2->dataObject));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
        retObj->dataObject->addToProtocol("By elementwise AND comparison of two dataObjects.");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbXor(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            *(dobj1->dataObject) ^
            *(dobj2->dataObject)); // resDataObj should always be the owner of its data, therefore
                                   // base of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
        retObj->dataObject->addToProtocol("By elementwise XOR comparison of two dataObjects.");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbOr(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            *(dobj1->dataObject) |
            *(dobj2->dataObject)); // resDataObj should always be the owner of its data, therefore
                                   // base of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
        retObj->dataObject->addToProtocol("By elementwise OR comparison of two dataObjects.");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceAdd(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    if (PyDataObject_Check(o2))
    {
        PyDataObject* dobj2 = (PyDataObject*)(o2);

        try
        {
            *(dobj1->dataObject) += *(dobj2->dataObject);
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        dobj1->dataObject->addToProtocol("Inplace addition of two dataObjects");
    }
    else if (PyFloat_Check(o2) || PyLong_Check(o2))
    {
        double val = PyFloat_AsDouble(o2);

        try
        {
            *(dobj1->dataObject) += val;
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace scalar addition of %g.", val);

        dobj1->dataObject->addToProtocol(buf);
    }
    else if (PyComplex_Check(o2))
    {
        ito::complex128 val =
            ito::complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

        try
        {
            *(dobj1->dataObject) += val;
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        if (val.imag() > 0)
        {
            sprintf_s(
                buf,
                PROTOCOL_STR_LENGTH,
                "Inplace scalar addition of %g+i%g.",
                val.real(),
                val.imag());
        }
        else
        {
            sprintf_s(
                buf,
                PROTOCOL_STR_LENGTH,
                "Inplace scalar addition of %g-i%g.",
                val.real(),
                -val.imag());
        }

        dobj1->dataObject->addToProtocol(buf);
    }
    else if (PythonDateTime::PyTimeDelta_CheckExt(o2))
    {
        bool ok;
        const auto td = PythonDateTime::GetTimeDelta(o2, ok);

        try
        {
            *(dobj1->dataObject) += td;
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace scalar addition of timedelta.");
        dobj1->dataObject->addToProtocol(buf);
    }
    else
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "the second operand must be either a data object or an integer, floating point, "
            "complex or timedelta value");
        return NULL;
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceSubtract(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    if (PyDataObject_Check(o2))
    {
        PyDataObject* dobj2 = (PyDataObject*)(o2);

        try
        {
            *(dobj1->dataObject) -= *(dobj2->dataObject);
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        dobj1->dataObject->addToProtocol("Inplace subtraction of two dataObjects.");
    }
    else if (PyFloat_Check(o2) || PyLong_Check(o2))
    {
        double val = PyFloat_AsDouble(o2);

        try
        {
            *(dobj1->dataObject) -= val;
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace scalar subtraction of %g.", val);

        dobj1->dataObject->addToProtocol(buf);
    }
    else if (PyComplex_Check(o2))
    {
        ito::complex128 val =
            ito::complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

        try
        {
            *(dobj1->dataObject) -= val;
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        if (val.imag() > 0)
        {
            sprintf_s(
                buf,
                PROTOCOL_STR_LENGTH,
                "Inplace scalar subtraction of %g+i%g.",
                val.real(),
                val.imag());
        }
        else
        {
            sprintf_s(
                buf,
                PROTOCOL_STR_LENGTH,
                "Inplace scalar subtraction of %g-i%g.",
                val.real(),
                -val.imag());
        }

        dobj1->dataObject->addToProtocol(buf);
    }
    else if (PythonDateTime::PyTimeDelta_CheckExt(o2))
    {
        bool ok;
        const auto td = PythonDateTime::GetTimeDelta(o2, ok);

        try
        {
            *(dobj1->dataObject) -= td;
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace scalar subtraction of timedelta.");
        dobj1->dataObject->addToProtocol(buf);
    }
    else
    {
        PyErr_SetString(
            PyExc_RuntimeError,
            "the second operand must be either a data object, an integer, floating point, "
            "complex or timedelta value");
        return NULL;
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceMultiply(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    if (Py_TYPE(o2) == &PyDataObjectType)
    {
        PyDataObject* dobj2 = (PyDataObject*)(o2);

        try
        {
            *(dobj1->dataObject) *= *(dobj2->dataObject);
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        dobj1->dataObject->addToProtocol("Inplace multiplication of two dataObjects");
    }
    else
    {
        if (PyComplex_Check(o2))
        {
            ito::complex128 factor(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

            if (PyErr_Occurred())
            {
                return NULL;
            }

            try
            {
                *(dobj1->dataObject) *= factor;
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            char buf[PROTOCOL_STR_LENGTH] = {0};
            if (factor.imag() > 0)
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Inplace scalar multiplication of %g+i%g.",
                    factor.real(),
                    factor.imag());
            }
            else
            {
                sprintf_s(
                    buf,
                    PROTOCOL_STR_LENGTH,
                    "Inplace scalar multiplication of %g-i%g.",
                    factor.real(),
                    -factor.imag());
            }

            dobj1->dataObject->addToProtocol(buf);
        }
        else
        {
            double factor = PyFloat_AsDouble(o2);

            if (PyErr_Occurred())
            {
                return NULL;
            }

            try
            {
                *(dobj1->dataObject) *= factor;
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return NULL;
            }

            char buf[PROTOCOL_STR_LENGTH] = {0};
            sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace scalar multiplication of %g.", factor);

            dobj1->dataObject->addToProtocol(buf);
        }
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceMatrixMultiply(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    try
    {
        *(dobj1->dataObject) *= *(dobj2->dataObject);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    dobj1->dataObject->addToProtocol("Inplace matrix multiplication of two dataObjects");

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceTrueDivide(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    if (Py_TYPE(o2) == &PyDataObjectType)
    {
        PyErr_SetString(PyExc_RuntimeError, "division by another dataObject is not implemented.");
        // dobj1->dataObject->addToProtocol("Inplace division of two dataObjects");
        return NULL;
    }
    else if (PyComplex_Check(o2))
    {
        complex128 factor = complex128(PyComplex_RealAsDouble(o2), PyComplex_ImagAsDouble(o2));

        try
        {
            *(dobj1->dataObject) *= (complex128(1.0, 0.0) / factor);
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        if (factor.real() > 0)
        {
            sprintf_s(
                buf,
                PROTOCOL_STR_LENGTH,
                "Inplace scalar division by %g+i%g.",
                factor.real(),
                factor.imag());
        }
        else
        {
            sprintf_s(
                buf,
                PROTOCOL_STR_LENGTH,
                "Inplace scalar division by %g-i%g.",
                factor.real(),
                -factor.imag());
        }

        dobj1->dataObject->addToProtocol(buf);
    }
    else
    {
        double factor = PyFloat_AsDouble(o2);

        if (PyErr_Occurred())
        {
            return NULL;
        }

        try
        {
            *(dobj1->dataObject) *= (1.0 / factor);
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        char buf[PROTOCOL_STR_LENGTH] = {0};
        sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace scalar division by %g.", factor);

        dobj1->dataObject->addToProtocol(buf);
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceRemainder(PyObject* /*o1*/, PyObject* /*o2*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplacePower(
    PyObject* /*o1*/, PyObject* /*o2*/, PyObject* /*o3*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceLshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    int overflow;
    int shift = PyLong_AsLongAndOverflow(o2, &overflow);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    if (overflow)
    {
        PyErr_SetString(PyExc_ValueError, "shift value exceeds integer range");
        return NULL;
    }

    if (shift < 0)
    {
        PyErr_SetString(PyExc_TypeError, "shift value must not be negative");
        return NULL;
    }

    try
    {
        *(dobj1->dataObject) <<= static_cast<unsigned int>(shift);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    Py_INCREF(o1);

    char buf[PROTOCOL_STR_LENGTH] = {0};
    sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace left shift by %i.", shift);

    dobj1->dataObject->addToProtocol(buf);

    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceRshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1, o1))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);

    int overflow;
    int shift = PyLong_AsLongAndOverflow(o2, &overflow);

    if (PyErr_Occurred())
    {
        return NULL;
    }

    if (overflow)
    {
        PyErr_SetString(PyExc_ValueError, "shift value exceeds integer range");
        return NULL;
    }

    if (shift < 0)
    {
        PyErr_SetString(PyExc_TypeError, "shift value must not be negative");
        return NULL;
    }

    try
    {
        *(dobj1->dataObject) >>= static_cast<unsigned int>(shift);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    Py_INCREF(o1);
    char buf[PROTOCOL_STR_LENGTH] = {0};
    sprintf_s(buf, PROTOCOL_STR_LENGTH, "Inplace right shift by %i.", shift);

    dobj1->dataObject->addToProtocol(buf);

    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceAnd(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    try
    {
        *(dobj1->dataObject) &= *(dobj2->dataObject);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    dobj1->dataObject->addToProtocol("Inplace elementwise AND comparison with second dataObject.");

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceXor(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    try
    {
        *(dobj1->dataObject) ^= *(dobj2->dataObject);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    dobj1->dataObject->addToProtocol("Inplace elementwise XOR comparison with second dataObject.");
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceOr(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2, o1, o2))
    {
        return NULL;
    }

    PyDataObject* dobj1 = (PyDataObject*)(o1);
    PyDataObject* dobj2 = (PyDataObject*)(o2);

    try
    {
        *(dobj1->dataObject) |= *(dobj2->dataObject);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    dobj1->dataObject->addToProtocol("Inplace elementwise OR comparison with second dataObject.");
    Py_INCREF(o1);
    return (PyObject*)o1;
}

//-------------------------------------------------------------------------------------
/*static*/ int PythonDataObject::PyDataObj_nbBool(PyDataObject* self)
{
    if (self->dataObject == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "DataObject is nullptr.");
        return -1;
    }

    switch (self->dataObject->getTotal())
    {
    case 0:
        return 0;
        break;
    case 1: {
        // currently the biggest data type of dataObject is complex128 -> 16 * 8 bit.
        // Therefore zeros is initialized with 16 * 8bit value 0 for a zero comparison.
        int elemSize = self->dataObject->elemSize();

        if (elemSize > 16)
        {
            PyErr_SetString(
                PyExc_RuntimeError, "Datatype of dataObject is too large for bool implementation.");
            return -1;
        }

        const uchar zeros[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const uchar* data = self->dataObject->getCvPlaneMat(0)->data;
        if (memcmp(zeros, data, elemSize) == 0)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }
    break;
    default:
        PyErr_SetString(
            PyExc_ValueError,
            "The truth value of a dataObject with more than one element is ambiguous.");
        return -1;
    }
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_getiter(PyDataObject* self)
{
    PyObject* args = PyTuple_Pack(1, self); // new ref
    PyDataObjectIter* result =
        (PyDataObjectIter*)PyObject_Call((PyObject*)&PyDataObjectIterType, args, NULL);
    Py_DECREF(args);
    if (result != NULL)
    {
        return (PyObject*)result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectName_doc, "name() -> str \n\
\n\
Returns the name of this object \n\
\n\
Returns \n\
------- \n\
str \n\
    the name of this object (``dataObject``)");
PyObject* PythonDataObject::PyDataObject_name(PyDataObject* /*self*/)
{
    PyObject* result;
    result = PyUnicode_FromString("dataObject");
    return result;
};

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_repr(PyDataObject* self)
{
    PyObject* result;
    int dims;
    if (self->dataObject == NULL)
    {
        result = PyUnicode_FromFormat("dataObject(empty)");
    }
    else
    {
        ito::DataObject* dObj = self->dataObject;
        dims = dObj->getDims();
        switch (dims)
        {
        case 2:
            result = PyUnicode_FromFormat(
                "dataObject('%s', [%i x %i], continuous: %i, owndata: %i)",
                typeNumberToName(dObj->getType()),
                dObj->getSize(0),
                dObj->getSize(1),
                dObj->getContinuous(),
                dObj->getOwnData());
            break;
        case 3:
            result = PyUnicode_FromFormat(
                "dataObject('%s', [%i x %i x %i], continuous: %i, owndata: %i)",
                typeNumberToName(dObj->getType()),
                dObj->getSize(0),
                dObj->getSize(1),
                dObj->getSize(2),
                dObj->getContinuous(),
                dObj->getOwnData());
            break;
        default:
            result = PyUnicode_FromFormat(
                "dataObject('%s', %i dims, continuous: %i, owndata: %i)",
                typeNumberToName(dObj->getType()),
                dObj->getDims(),
                dObj->getContinuous(),
                dObj->getOwnData());
            break;
        }
    }
    return result;
};

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectData_doc, "data() \n\
\n\
Prints the content of the dataObject to the command line in a readable form.");
PyObject* PythonDataObject::PyDataObject_data(PyDataObject* self)
{
    try
    {
        std::cout << *(self->dataObject);
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return nullptr;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectConj_doc, "conj() \n\
\n\
Converts this dataObject into its complex-conjugate (inline). \n\
\n\
Every value of this :class:`dataObject` is replaced by its complex-conjugate value. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if the data type of this data object is not complex.\n\
\n\
See Also \n\
-------- \n\
conjugate : does the same operation but returns a complex-conjugated copy of this data object");
PyObject* PythonDataObject::PyDataObject_conj(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    try
    {
        self->dataObject->conj();
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }
    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
// return the complex-conjugate, element-wise
PyDoc_STRVAR(pyDataObjectConjugate_doc, "conjugate() -> dataObject \n\
\n\
Returns a copy of this dataObject where every element is complex-conjugated. \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    element-wise complex conjugate of this data object \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if the data type of this data object is not complex.\n\
\n\
See Also \n\
-------- \n\
conj : does the same operation but manipulates this object inline.");
PyObject* PythonDataObject::PyDataObject_conjugate(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    retObj->dataObject = new ito::DataObject();
    try
    {
        self->dataObject->copyTo(*(retObj->dataObject), 1);
        retObj->dataObject->conj();
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAdj_doc, "adj() \n\
\n\
Adjugates this dataObject (plane-by-plane). \n\
\n\
Every plane (spanned by the last two axes) is transposed and every element is \n\
replaced by its complex conjugate value. This is done in-line. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if the data type of this data object is not complex.\n\
\n\
See Also \n\
-------- \n\
adjugate : does the same operation but returns the resulting data object");
PyObject* PythonDataObject::PyDataObject_adj(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    try
    {
        ito::DataObject* newDataObj = new ito::DataObject(self->dataObject->adj());
        delete self->dataObject;
        self->dataObject = newDataObj;
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    self->dataObject->addToProtocol("Run inplace adjugate function on this dataObject.");

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAdjugate_doc, "adjugate() -> dataObject \n\
\n\
Returns the plane-wise adjugated array of this dataObject. \n\
\n\
If this data object has a complex type, the transposed data object is returned where \n\
every element is complex conjugated. For data objects with more than two dimensions \n\
the transposition is done plane-wise, hence, only the last two dimensions are permuted. \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    adjugate of this dataObject. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if the data type of this data object is not complex.\n\
\n\
See Also \n\
-------- \n\
adj : does the same operation but manipulates this object inline.");
PyObject* PythonDataObject::PyDataObject_adjugate(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->adj());
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol(
            "Created by calculation of adjugate value from a dataObject.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectTrans_doc, "trans() -> dataObject \n\
\n\
Returns the (plane-wise) transposed dataObject. \n\
\n\
Return a new data object with the same data type than this object and where every \n\
plane (data spanned by the last two dimensions) is transposed respectively \n\
such that the last two axes are permuted. The :attr:`shape` of the returned \n\
dataObject is then equal to the :attr:`shape` of this dataObject, but the last two \n\
values in the shape tuple are swapped. \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    A copy of this dataObject is returned where every plane is its transposed plane. \n\
\n\
See Also \n\
-------- \n\
T : this method is equal to the attribute :attr:`dataObject.T`.");
PyObject* PythonDataObject::PyDataObject_trans(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->trans());
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Created by transponation of a dataObject.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectTranspose_doc,
    "dataObject : Returns a copy of this dataObject where the two last dimensions are swapped. \n\
\n\
Return a new data object with the same data type than this object and where every \n\
plane (data spanned by the last two dimensions) is transposed respectively \n\
such that the last two axes are permuted. The :attr:`shape` of the returned \n\
dataObject is then equal to the :attr:`shape` of this dataObject, but the last two \n\
values in the shape tuple are swapped. \n\
\n\
This attribute was added with itom 5.0. \n\
\n\
See Also \n\
-------- \n\
trans : This method is equal to the method :meth:`dataObject.trans`.");
PyObject* PythonDataObject::PyDataObject_transpose(PyDataObject* self, void* closure)
{
    return PyDataObject_trans(self);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectMakeContinuous_doc, "makeContinuous() -> dataObject \n\
\n\
Returns a continuous representation of this dataObject.\n\
\n\
Per default a dataObject with more than two dimensions allocates separated chunks of \n\
memory for every plane, where a plane is always the matrix given by the last two \n\
dimensions. This separated storage usually allows allocating more memory for huge for \n\
instance three dimensional matrices. However, in order to generate a dataObject that is \n\
directly compatible to Numpy or other C-style matrix structures, the entire allocated \n\
memory must be in one block, that is called continuous. If you create a Numpy array \n\
from a dataObject that is not continuous, this function is implicitely called in order \n\
to firstly make the dataObject continuous before passing to Numpy. \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    If this dataObject is not continuous, its continuous representation is returned \n\
    as deep copy. A deep copy is also returned if this object is already :attr:`continuous`.");
PyObject* PythonDataObject::PyDataObject_makeContinuous(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    retObj->dataObject = new ito::DataObject(ito::makeContinuous(*(self->dataObject)));

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Made dataObject continuous.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSize_doc, "size(axisIndex = None) -> Union[int, Tuple[int]] \n\
\n\
Returns the size of the selected axis of this dataObject or the entire shape tuple, if no specific axis is given. \n\
\n\
Parameters  \n\
----------\n\
axisIndex : int, optional\n\
    If ``axisIndex`` is given, only the size of the indicated axis is returned as \n\
    single number. \n\
\n\
Returns \n\
------- \n\
int or tuple of int \n\
    A tuple containing the sizes of all dimensions or one single size value \n\
    if ``axisIndex`` is given. \n\
\n\
Raises \n\
------ \n\
DeprecatedWarning \n\
    This method is deprecated. For a more consistent syntax with \n\
    :class:`numpy.ndarray` objects, use :attr:`shape` instead. \n\
\n\
See Also \n\
-------- \n\
shape : the read-only attribute shape is equal to ``size()``.");
PyObject* PythonDataObject::PyDataObject_size(PyDataObject* self, PyObject* args)
{
    if (PyErr_WarnEx(
            PyExc_DeprecationWarning,
            "size([idx]) is deprecated. Use attribute shape "
            "instead (more consistent to numpy)",
            1) == -1)
    {
        // exception is raised instead of warning (depending on user defined warning levels)
        return NULL;
    }

    int desiredDim = -1;

    if (!PyArg_ParseTuple(args, "|i", &desiredDim))
    {
        return NULL;
    }

    PyObject* shapes = PyDataObj_GetShape(self, NULL);

    if (desiredDim >= 0 && desiredDim < PyTuple_Size(shapes))
    {
        PyObject* temp = shapes;
        shapes = PyTuple_GetItem(shapes, desiredDim);
        Py_INCREF(shapes);
        Py_DECREF(temp);
        return shapes;
    }
    else if (desiredDim != -1)
    {
        Py_DECREF(shapes);
        PyErr_SetString(PyExc_TypeError, "index argument out of boundaries.");
        return NULL;
    }

    return shapes;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectCopy_doc, "copy(regionOnly = False) -> dataObject \n\
\n\
Returns a deep copy of this dataObject\n\
\n\
Parameters \n\
---------- \n\
regionOnly : bool, optional \n\
    If ``regionOnly`` is ``True``, only the current region of interest of this \n\
    dataObject is copied, else the entire dataObject including the shaded areas outside \n\
    of the current region of interest are copied, including the ROI settings [default].\n\
\n\
Returns \n\
------- \n\
dataObject \n\
    Deep copy of this dataObject. \n\
\n\
See Also \n\
-------- \n\
locateROI");
PyObject* PythonDataObject::PyDataObject_copy(PyDataObject* self, PyObject* args, PyObject* kwds)
{
    if (self->dataObject == NULL)
        return 0;

    unsigned char regionOnly = 0;
    const char* kwlist[] = {"regionOnly", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|b", const_cast<char**>(kwlist), &regionOnly))
    {
        PyErr_SetString(PyExc_TypeError, "the region only flag must be 0 or 1");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    retObj->dataObject = new ito::DataObject();

    try
    {
        if (regionOnly)
        {
            // self->dataObject should always be the owner of its data, therefore base of
            // resultObject remains None
            self->dataObject->copyTo(*(retObj->dataObject), 1);
        }
        else
        {
            // self->dataObject should always be the owner of its data, therefore base of
            // resultObject remains None
            self->dataObject->copyTo(*(retObj->dataObject), 0);
        }
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (regionOnly)
    {
        if (retObj)
            retObj->dataObject->addToProtocol("Copied region of dataObject to new object.");
    }
    else
    {
        if (retObj)
            retObj->dataObject->addToProtocol("Copied dataObject to new object.");
    }
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectMul_doc, "mul(otherObj) -> dataObject \n\
\n\
Returns the result of the element-wise multiplication of this dataObject with otherObj. \n\
\n\
This :class:`dataObject` and ``otherObj`` must have the same :attr:`shape` and \n\
:attr:`dtype` for the element-wise multiplication. \n\
\n\
All meta information (axis scales, offsets, descriptions, units, tags...) of the \n\
resulting object are copied from this data object. \n\
\n\
Parameters  \n\
----------\n\
otherObj : dataObject \n\
    The returned :class:`dataObject` contains the result of the element-wise \n\
    multiplication of all values in this object and ``otherObj``. Must have the \n\
    same shape and data type than this object. \n\
\n\
Returns \n\
------- \n\
result : dataObject \n\
    Resulting multiplied data object. Values, that exceed the range of the current \n\
    data type, will be set to the ``result modulo max(dtype)``. \n\
\n\
Notes \n\
----- \n\
For a mathematical multiplication see the @-operator.");
PyObject* PythonDataObject::PyDataObject_mul(PyDataObject* self, PyObject* args)
{
    if (self->dataObject == NULL)
        return 0;

    PyObject* pyDataObject = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &pyDataObject))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument is no data object");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    PyDataObject* obj2 = (PyDataObject*)pyDataObject;

    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->mul(
            *(obj2->dataObject))); // new dataObject should always be the owner of its data,
                                   // therefore base of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
        retObj->dataObject->addToProtocol(
            "Created by elementwise multiplication of two dataObjects.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectDiv_doc, "div(otherObj) -> dataObject \n\
\n\
Returns the result of the element-wise division of this dataObject by otherObj. \n\
\n\
This :class:`dataObject` and ``otherObj`` must have the same :attr:`shape` and \n\
:attr:`dtype` for the element-wise division. \n\
\n\
All meta information (axis scales, offsets, descriptions, units, tags...) of the \n\
resulting object are copied from this data object. \n\
\n\
Parameters  \n\
----------\n\
otherObj : dataObject \n\
    The returned :class:`dataObject` contains the result of the element-wise \n\
    division of all values in this object by ``otherObj``. Must have the \n\
    same shape and data type than this object. \n\
\n\
Returns \n\
------- \n\
result : dataObject \n\
    Resulting divided data object. Values, that exceed the range of the current \n\
    data type, will be set to the ``result modulo max(dtype)``. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if a **division by zero** occurs for integer or complex data types.");
PyObject* PythonDataObject::PyDataObject_div(PyDataObject* self, PyObject* args)
{
    if (self->dataObject == NULL)
        return 0;

    PyObject* pyDataObject = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &pyDataObject))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument is no data object");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    PyDataObject* obj2 = (PyDataObject*)pyDataObject;

    try
    {
        // new dataObject should always be the owner of its data, therefore base of resultObject
        // remains None
        retObj->dataObject = new ito::DataObject((*(self->dataObject)).div(*(obj2->dataObject)));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
    {
        retObj->dataObject->addToProtocol("Created by elementwise division of two dataObjects.");
    }

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectReshape_doc, "reshape(shape) -> dataObject \n\
\n\
Returns a reshaped (shallow) copy of this dataObject. \n\
\n\
Reshaping means, that the shape (and optionally number of dimensions) of a \n\
:class:`dataObject` might be changed, unless the total number of elements is \n\
not changed. The reshaped and returned :class:`dataObject` has the same data type and \n\
data than this :class:`dataObject`. \n\
\n\
The shape of the returned object corresponds to the parameter ``shape``.  \n\
If the last two dimensions of ``shape`` and of this object are equal and if the \n\
data is not continously organized, a shallow copy can be returned, else a deep \n\
copy has to be created. \n\
\n\
Tags and the rotation matrix are copied. The axis tags are only copied for all axes \n\
whose size will not change beginning from the last axis (``x``). Copying the axis \n\
meta information is stopped after the first axis with a differing new size. \n\
\n\
Parameters \n\
---------- \n\
shape : sequence of int \n\
    New shape of the returned object. A minimal size of this list or tuple is two. \n\
\n\
Returns \n\
------- \n\
reshaped : dataObject \n\
    The reshaped data object. \n\
\n\
Notes \n\
----- \n\
This method is similar to :meth:`numpy.reshape`.");
PyObject* PythonDataObject::PyDataObject_reshape(PyDataObject* self, PyObject* args, PyObject* kwds)
{
    if (self->dataObject == NULL)
        return NULL;

    PyObject* shape = NULL;
    const char* kwlist[] = {"shape", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &shape))
    {
        return NULL;
    }

    bool ok;
    QVector<int> shapes = PythonQtConversion::PyObjGetIntArray(shape, false, ok);

    if (!ok)
    {
        PyErr_Format(PyExc_TypeError, "The argument 'newShape' must be a sequence of integers");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        ito::DataObject resObj = self->dataObject->reshape(shapes.size(), shapes.data());
        retObj->dataObject = new ito::DataObject(resObj);
    }
    catch (cv::Exception& exc)
    {
        retObj->dataObject = NULL;

        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Reshaped dataObject.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAstype_doc, "astype(typestring) -> dataObject \n\
\n\
Converts this dataObject to another data type.\n\
\n\
Converts this :class:`dataObject` to a new dataObject with another data type, given by \n\
the string ``typestring`` (e.g. 'uint8'). The converted dataObject is a deep copy of \n\
this object if the new type does not correspond to the current type, else a shallow \n\
copy of this object is returned. \n\
\n\
Parameters \n\
---------- \n\
typestring : str \n\
    Type string indicating the new type (``uint8``, ..., ``float32``, ..., \n\
    ``complex128``). \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    The converted :class:`dataObject`.");
PyObject* PythonDataObject::PyDataObject_astype(PyDataObject* self, PyObject* args, PyObject* kwds)
{
    const char* type;
    int typeno = 0;

    const char* kwlist[] = {"typestring", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &type))
    {
        return NULL;
    }

    typeno = dObjTypeFromName(type);

    if (typeno == -1)
    {
        PyErr_Format(PyExc_TypeError, "The given type string %s is unknown", type);
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    retObj->dataObject = new ito::DataObject();

    try
    {
        self->dataObject->convertTo(*(retObj->dataObject), typeno);
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    char buf[PROTOCOL_STR_LENGTH] = {0};
    sprintf_s(
        buf,
        PROTOCOL_STR_LENGTH,
        "Converted from dataObject of type %s to type %s",
        typeNumberToName(self->dataObject->getType()),
        type);


    if (retObj)
        retObj->dataObject->addToProtocol(buf);

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectNormalize_doc,
    "normalize(minValue = 0.0, maxValue = 1.0, typestring = \"\") -> dataObject \n\
\n\
Returns a normalized version of this dataObject. \n\
\n\
All values in the returned :class:`dataObject` are normalized with respect to the given \n\
range ``[minValue, maxValue]``. Additionally it is also possible to convert the \n\
resulting data object to another data type (given by the parameter ``typestring``). \n\
Per default, no such a type conversion is done (empty ``typestring`` argument). \n\
\n\
For the normalization, the current minimum and maximum value of this object is \n\
determined: \n\
\n\
.. math:: \n\
\n\
    min_{cur} = min(thisObj) \n\
    max_{cur} = max(thisObj) \n\
\n\
Each value `v` is then normalized by: \n\
\n\
.. math:: v_{norm} = minValue + (v - min_{cur}) * (maxValue - minValue) / (max_{cur} - min_{cur}) \n\
\n\
Parameters \n\
---------- \n\
minValue : float \n\
    minimum value of the normalized range. \n\
maxValue : float \n\
    maximum value of the normalized range. \n\
typestring : str \n\
    Data type for an optional type conversion. If an empty :obj:`str` is given, \n\
    no such a conversion is done. Else possible values are (among others): \n\
    (``uint8``, ..., ``float32``, ..., ``complex128``). \n\
\n\
Returns \n\
------- \n\
normalized : dataObject \n\
    normalized data object \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if a ``DateTime`` or ``TimeDelta`` dataObject should be normalized. Not supported. \n\
\n\
Notes \n\
----- \n\
For complex data types, the current minimum and maximum values are calculated \n\
based on the absolute value of the complex values. Therefore, the normalization \n\
can have a different result, than maybe expected.");
PyObject* PythonDataObject::PyDataObject_normalize(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    const char* type = NULL;
    double minVal = 0.0;
    double maxVal = 1.0;
    int typeno = 0;

    const char* kwlist[] = {"minValue", "maxValue", "typestring", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|dds", const_cast<char**>(kwlist), &minVal, &maxVal, &type))
    {
        return nullptr;
    }

    if (type != nullptr)
    {
        typeno = dObjTypeFromName(type);

        if (typeno == -1)
        {
            PyErr_Format(PyExc_TypeError, "The given type string %s is unknown", type);
            return nullptr;
        }
    }
    else
    {
        typeno = self->dataObject->getType();
    }

    double smin, smax;
    ito::uint32 loc1[] = {0, 0, 0};
    ito::uint32 loc2[] = {0, 0, 0};
    ito::RetVal retval =
        ito::dObjHelper::minMaxValue(self->dataObject, smin, loc1, smax, loc2, true);

    if (!PythonCommon::transformRetValToPyException(retval))
    {
        return nullptr;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    ito::DataObject dataObj;

    double dmin = std::min(minVal, maxVal);
    double dmax = std::max(minVal, maxVal);
    double scale = (dmax - dmin) *
        ((smax - smin) > std::numeric_limits<double>::epsilon() ? (1. / (smax - smin)) : 0.0);
    double shift = dmin - smin * scale;

    try
    {
        self->dataObject->convertTo(dataObj, typeno, scale, shift);
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject = new ito::DataObject(dataObj);

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    char buf[200] = {0};
    sprintf_s(
        buf,
        200,
        "Normalized from dataObject of type %s to type %s between %g and %g.",
        typeNumberToName(self->dataObject->getType()),
        type,
        dmin,
        dmax);

    if (retObj)
        retObj->dataObject->addToProtocol(buf);

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectLocateROI_doc, "locateROI() -> Tuple[List[int], List[int]] \n\
\n\
Returns information about the current region of interest of this dataObject. \n\
\n\
In Python, it is common to use slices of other objects, like lists, tuples, \n\
:class:`numpy.ndarray` among others. This slice represents a subpart of the original \n\
object, however the values within the slice are still the same than in the original \n\
object. \n\
\n\
The same holds for :class:`dataObject`, where a slice will return a shallow \n\
copy of the original object with a maybe reduced size. This is denoted as region \n\
of interest (ROI). Here an example:: \n\
    \n\
    org = dataObject.ones([100, 200], 'float32') \n\
    roi = org[10:20, 5:9]  # slicing \n\
    \n\
    roi[0, 0] = 100  # change one value in roi \n\
    print(org[10, 5])  # returns 100 \n\
\n\
Although the first value in ``roi`` is changed, its corresponding value in ``org`` \n\
is changed, too. This is the principle of shallow copies and slicing / region of \n\
interests. \n\
\n\
This method returns information about the exact **position** of the region of \n\
interest within its original :class:`dataObject`. This is defined by two values \n\
for each axis. The first value indicates the distance between the left, top, etc. border \n\
of the original object and the border of this object. If no region of interest is set, \n\
these values are ``0`` everywhere. \n\
\n\
The second values define the distances between the right, bottom, ... margins of this \n\
object and its original object (or ``0`` everywhere, too). \n\
\n\
This method returns a tuple with two elements: The first is a list with the original \n\
sizes of this data object (if no ROI would have been set), the second is a list with \n\
the offsets from the original data object to the first value in the current region of \n\
interest. \n\
\n\
If no region of interest is set (hence: full region of interest), the first list \n\
corresponds to the one returned by :attr:`shape`, the 2nd list contains ``0`` everyhwere. \n\
\n\
The output of the example above would be:: \n\
    \n\
    print(roi.locateROI()) \n\
    # >>> ([100, 200], [10, 5]) \n\
\n\
Returns \n\
------- \n\
orgSize : list of int \n\
    The original sizes of this object (without ROI). This is equal to :attr:`shape` \n\
    of the original object. \n\
offsets : list of int \n\
    A list with ``N`` values, where ``N`` is the number of dimensions of this object. \n\
    Each value ``n = 1 .. N`` is the offset of the first value of axis ``n`` in this \n\
    object with respect to the original object. \n\
\n\
See Also \n\
-------- \n\
adjustROI : method to change the current region of interest");
PyObject* PythonDataObject::PyDataObject_locateROI(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    int dims = self->dataObject->getDims();
    int* osize = new int[dims];
    int* offsets = new int[dims];

    self->dataObject->locateROI(osize, offsets);

    PyObject* osize_obj = PyList_New(dims);
    PyObject* offsets_obj = PyList_New(dims);

    for (int i = 0; i < dims; i++)
    {
        PyList_SetItem(osize_obj, i, Py_BuildValue("i", osize[i]));
        PyList_SetItem(offsets_obj, i, Py_BuildValue("i", offsets[i]));
    }

    DELETE_AND_SET_NULL_ARRAY(osize);
    DELETE_AND_SET_NULL_ARRAY(offsets);

    PyObject* result = Py_BuildValue("(OO)", osize_obj, offsets_obj);
    Py_DECREF(osize_obj);
    Py_DECREF(offsets_obj);

    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAdjustROI_doc, "adjustROI(offsets) \n\
\n\
Adjusts the size and position of the region of interest of this object.\n\
\n\
In Python, it is common to use slices of other objects, like lists, tuples, \n\
:class:`numpy.ndarray` among others. This slice represents a subpart of the original \n\
object, however the values within the slice are still the same than in the original \n\
object. \n\
\n\
The same holds for :class:`dataObject`, where a slice will return a shallow \n\
copy of the original object with a maybe reduced size. This is denoted as region \n\
of interest (ROI). Here an example:: \n\
    \n\
    org = dataObject.ones([100, 200], 'float32') \n\
    roi = org[10:20, 5:9]  # slicing \n\
    \n\
    roi[0, 0] = 100  # change one value in roi \n\
    print(org[10, 5])  # returns 100 \n\
\n\
Although the first value in ``roi`` is changed, its corresponding value in ``org`` \n\
is changed, too. This is the principle of shallow copies and slicing / region of \n\
interests. \n\
\n\
This method is used to change to offset and / or size of the current ROI of this \n\
object. Of course, this ROI can never be bigger than the original array data. \n\
In order to change the position and / or size of the current region of interest, \n\
pass a sequence (list or tuple) of integer values. The length of this sequence \n\
must be ``2 * ndim``, where ``ndim`` is the number of dimensions of this object. \n\
Always two adjacent values in this sequence refer to one axis, starting with \n\
the first axis index and ending with the last one. The first value of such a pair \n\
of two values indicate the offset of the region of interest with respect to one \n\
border of this axis (e.g. the left or top border), the 2nd value is the offset \n\
with respect to the other side of this axis (e.g. the right or bottom border). \n\
Negative values decrease the size of the ROI towards the center, positive values \n\
will increase its current size. \n\
\n\
Example: :: \n\
\n\
    d = dataObject([5, 4]) \n\
    droi = dataObject(d)  # make a shallow copy \n\
    droi.adjustROI([-2, 0, -1, -1]) \n\
\n\
Now, ``droi`` has a ROI, whose first value is equal to ``d[2, 1]`` and its shape \n\
is ``(3, 2)``. \n\
\n\
Parameters \n\
---------- \n\
offsets : list of int or tuple of int \n\
    This sequence must have twice as many values than the number of dimensions of \n\
    this :class:`dataObject`. A pair of numbers indicates the shift of the \n\
    current boundaries of the region of interest in every dimension. The first value \n\
    of each pair is the offset of the **left** boundary, the second the shift of the \n\
    **right** boundary. A positive value means a growth of the region of interest, \n\
    a negative one let the region of interest shrink towards the center. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if desired, new ROI exceeds the original shape of this :class:`dataObject`. \n\
\n\
See Also \n\
-------- \n\
locateROI : method to get the borders of the current ROI");
PyObject* PythonDataObject::PyDataObject_adjustROI(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    // args is supposed to be a list of offsets for each dimensions on the "left" and "right" side.
    // e.g. 2D-Object [dtop, dbottom, dleft, dright], negative d-value means offset towards the
    // center
    PyObject* offsetsArg = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    const char* kwlist[] = {"offsets", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char**>(kwlist), &offsetsArg))
    {
        return NULL;
    }

    bool ok;
    QVector<int> offsets = PythonQtConversion::PyObjGetIntArray(offsetsArg, true, ok);

    int dims = self->dataObject->getDims();

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "offsets must be a sequence of integer values.");
        return NULL;
    }
    else if (offsets.size() != 2 * dims)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "offsets must be a sequence of integer values. "
            "Its length must be two times the number of dimensions.");
        return NULL;
    }

    bool error = false;

    if (dims > 0)
    {
        try
        {
            self->dataObject->adjustROI(dims, offsets.constData());
        }
        catch (cv::Exception& exc)
        {
            PyErr_SetString(PyExc_RuntimeError, (exc.err).c_str());
            error = true;
        }
    }

    if (error)
    {
        return NULL;
    }
    else
    {
        Py_RETURN_NONE;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSqueeze_doc, "squeeze() -> dataObject \n\
\n\
Returns a squeezed shallow copy (if possible) of this dataObject. \n\
\n\
This method removes every dimension with size equal to ``1``. A shallow copy is only \n\
returned, if the sub-arrays, spanned by the last two dimensions (denoted as planes), \n\
are not affected by the squeeze operation and if the data block in the \n\
:class:`dataObject` is not continuous. Else a deep copy has to be returned due to an \n\
overall re-alignment of the matrix. The returned object can never have less than \n\
two dimensions. If this is the case, the last or second to last dimensions with a size \n\
of ``1`` is not deleted. If :this method returns a shallow copy, a change in a \n\
value will change the same value in the original object, too. \n\
\n\
Returns \n\
------- \n\
squeezed : dataObject \n\
    The squeezed data object. \n\
\n\
Notes \n\
----- \n\
This method is similar to :meth:`numpy.squeeze`.");
PyObject* PythonDataObject::PyDataObject_squeeze(PyDataObject* self, PyObject* /*args*/)
{
    if (self->dataObject == NULL)
        return NULL;

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        ito::DataObject resObj = self->dataObject->squeeze();
        retObj->dataObject = new ito::DataObject(resObj);
    }
    catch (cv::Exception& exc)
    {
        retObj->dataObject = NULL;

        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Squeezed dataObject.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrReal_doc,
    "dataObject : Gets or sets the `real` part of this ``complex64`` or ``complex128`` object. \n\
\n\
The real part object has the same shape than this :class:`dataObject`. If the data type \n\
of this object is ``complex64``, the real part object has the data type ``float32``. \n\
For a ``complex128`` object, the real part is ``float64``. \n\
\n\
If a real part object is set to this attribute, it can be either a :class:`numpy.ndarray` \n\
or a :class:`dataObject` with the same shape than this object and the appropriate data type. \n\
However, it is also possible to pass an :obj:`int` or :obj:`float` value. This value is \n\
then assigned to the real part of all complex values. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this :class:`dataObject` has no complex data type.");
PyObject* PythonDataObject::PyDataObject_getReal(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject* d = self->dataObject;

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            ito::real(*(d))); // resDataObj should always be the owner of its data, therefore base
                              // of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject->addToProtocol("Extracted real part of a complex dataObject via real().");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setReal(PyDataObject* self, PyObject* value, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return -1;
    }

    ito::DataObject* newValues = NULL;
    bool deleteNewValues = false;
    bool singleValue = false;
    PyObject* pyNewValues = NULL; // if new values are borrowed from a PyObject, set it here (it
                                  // will be decremented at the end)

    int dataObjectType = self->dataObject->getType();

    if (dataObjectType != ito::tComplex64 &&
        dataObjectType != ito::tComplex128) // input object must be complex
    {
        PyErr_SetString(PyExc_TypeError, "type of dataObject is not complex.");
        return -1;
    }

    // newvalues as data object
    if (PyDataObject_Check(value)) // check if value is dataObject
    {
        newValues = (((PyDataObject*)(value))->dataObject);
    }
    else if (PyArray_Check(value)) // check if value is numpy array
    {
        pyNewValues = createPyDataObjectFromArray(value); // new reference
        if (pyNewValues)
        {
            newValues = ((PyDataObject*)(pyNewValues))->dataObject;
        }
        else
        {
            return -1; // error is already set by createPyDataObjectFromArray
        }
    }
    else if (PyLong_Check(value)) // check if value is integer single value
    {
        if (dataObjectType == ito::tComplex64)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat32);
            ito::float32 newValue = PyLong_AsLong(value);
            newValues->at<ito::float32>(0, 0) = newValue;
            deleteNewValues = true;
        }
        else if (dataObjectType == ito::tComplex128)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat64);
            ito::float64 newValue = PyLong_AsLong(value);
            newValues->at<ito::float64>(0, 0) = newValue;
            deleteNewValues = true;
        }
        singleValue = true;
    }
    else if (PyFloat_Check(value)) // check if value is float single value
    {
        if (dataObjectType == ito::tComplex64)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat32);
            ito::float32 newValue = PyFloat_AsDouble(value);
            newValues->at<ito::float32>(0, 0) = newValue;
            deleteNewValues = true;
        }
        else if (dataObjectType == ito::tComplex128)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat64);
            ito::float64 newValue = PyFloat_AsDouble(value);
            newValues->at<ito::float64>(0, 0) = newValue;
            deleteNewValues = true;
        }
        singleValue = true;
    }
    else // error
    {
        Py_XDECREF(pyNewValues);
        pyNewValues = NULL;
        if (deleteNewValues)
        {
            DELETE_AND_SET_NULL(newValues);
        }
        PyErr_SetString(
            PyExc_TypeError,
            "Type of assigned value is invalid (real dataObject, real np.array or real scalar "
            "value).");
        return -1;
    }

    if (dataObjectType == ito::tComplex64)
    {
        if (newValues->getType() != ito::tFloat32)
        {
            // try to convert newValues to float32...
            ito::DataObject* newValuesFloat = new ito::DataObject();
            ito::RetVal ret;
            try
            {
                ret = newValues->convertTo(*newValuesFloat, ito::tFloat32);
            }
            catch (cv::Exception& exc)
            {
                ret = ito::RetVal::format(
                    ito::retError,
                    0,
                    "Cannot convert assigned value to a float32 dataObject (%s)",
                    exc.err.c_str());
            }

            if (ret == ito::retOk)
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                newValues = newValuesFloat;
                deleteNewValues = true;
            }
            else
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                DELETE_AND_SET_NULL(newValuesFloat);

                PythonCommon::transformRetValToPyException(ret);
                return -1;
            }
        }
    }
    else if (dataObjectType == ito::tComplex128)
    {
        if (newValues->getType() != ito::tFloat64)
        {
            // try to convert newValues to float64...
            ito::DataObject* newValuesFloat = new ito::DataObject();
            ito::RetVal ret;
            try
            {
                ret = newValues->convertTo(*newValuesFloat, ito::tFloat64);
            }
            catch (cv::Exception& exc)
            {
                ret = ito::RetVal::format(
                    ito::retError,
                    0,
                    "Cannot convert assigned value to a float64 dataObject (%s)",
                    exc.err.c_str());
            }

            if (ret == ito::retOk)
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                newValues = newValuesFloat;
                deleteNewValues = true;
            }
            else
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                DELETE_AND_SET_NULL(newValuesFloat);

                PythonCommon::transformRetValToPyException(ret);
                return -1;
            }
        }
    }

    const int dObjDims = self->dataObject->getDims();
    const int valDims = newValues->getDims();

    if (singleValue)
    {
    }
    else if (dObjDims == valDims) // error if same dimensions but different shape
    {
        if (self->dataObject->getSize() != newValues->getSize())
        {
            Py_XDECREF(pyNewValues);
            pyNewValues = NULL;
            if (deleteNewValues)
            {
                DELETE_AND_SET_NULL(newValues);
            }

            PyErr_Format(
                PyExc_IndexError,
                "The size of this dataObject and the assigned dataObject or np.array must be "
                "equal.");
            return -1;
        }
    }
    else // dObjDims must be greater than valDims
    {
        if (dObjDims > valDims && valDims == 2)
        {
            if (!(self->dataObject->getSize(self->dataObject->getDims() - 1) ==
                      newValues->getSize(1) &&
                  self->dataObject->getSize(self->dataObject->getDims() - 2) ==
                      newValues->getSize(0))) // last 2 dimensions are the same
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                PyErr_SetString(PyExc_IndexError, "last 2 dimensions differs in size.");
                return -1;
            }
        }
        else
        {
            Py_XDECREF(pyNewValues);
            pyNewValues = NULL;
            if (deleteNewValues)
            {
                DELETE_AND_SET_NULL(newValues);
            }

            PyErr_SetString(
                PyExc_IndexError,
                "the shape of the data object must be greater than the shape of the values.");
            return -1;
        }
    }

    try
    {
        self->dataObject->setReal(*newValues);
    }
    catch (cv::Exception& exc)
    {
        Py_XDECREF(pyNewValues);
        pyNewValues = NULL;
        if (deleteNewValues)
        {
            DELETE_AND_SET_NULL(newValues);
        }

        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return -1;
    }

    self->dataObject->addToProtocol("Changed real part of complex data object via real.");

    Py_XDECREF(pyNewValues);
    pyNewValues = NULL;
    if (deleteNewValues)
    {
        DELETE_AND_SET_NULL(newValues);
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectAttrImag_doc,
    "dataObject : Gets or sets the `imag` part of this ``complex64`` or ``complex128`` object. \n\
\n\
The imaginary part object has the same shape than this :class:`dataObject`. If the data type \n\
of this object is ``complex64``, the imag part object has the data type ``float32``. \n\
For a ``complex128`` object, the imag part is ``float64``. \n\
\n\
If an imaginary part object is set to this attribute, it can be either a :class:`numpy.ndarray` \n\
or a :class:`dataObject` with the same shape than this object and the appropriate data type. \n\
However, it is also possible to pass an :obj:`int` or :obj:`float` value. This value is \n\
then assigned to the imaginary part of all complex values. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this :class:`dataObject` has no complex data type.");
PyObject* PythonDataObject::PyDataObject_getImag(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject* d = self->dataObject;

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            ito::imag(*(d))); // resDataObj should always be the owner of its data, therefore base
                              // of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject->addToProtocol("Extracted imaginary part of a complex dataObject via imag.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setImag(PyDataObject* self, PyObject* value, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return -1;
    }

    ito::DataObject* newValues = NULL;
    bool deleteNewValues = false;
    bool singleValue = false;
    PyObject* pyNewValues = NULL; // if new values are borrowed from a PyObject, set it here (it
                                  // will be decremented at the end)

    int dataObjectType = self->dataObject->getType();

    if (dataObjectType != ito::tComplex64 &&
        dataObjectType != ito::tComplex128) // input object must be complex
    {
        PyErr_SetString(PyExc_TypeError, "type of dataObject is not complex.");
        return -1;
    }

    // newvalues as data object
    if (PyDataObject_Check(value)) // check if value is dataObject
    {
        newValues = (((PyDataObject*)(value))->dataObject);
    }
    else if (PyArray_Check(value)) // check if value is numpy array
    {
        pyNewValues = createPyDataObjectFromArray(value); // new reference
        if (pyNewValues)
        {
            newValues = ((PyDataObject*)(pyNewValues))->dataObject;
        }
        else
        {
            return -1; // error is already set by createPyDataObjectFromArray
        }
    }
    else if (PyLong_Check(value)) // check if value is integer single value
    {
        if (dataObjectType == ito::tComplex64)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat32);
            ito::float32 newValue = PyLong_AsLong(value);
            newValues->at<ito::float32>(0, 0) = newValue;
            deleteNewValues = true;
        }
        else if (dataObjectType == ito::tComplex128)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat64);
            ito::float64 newValue = PyLong_AsLong(value);
            newValues->at<ito::float64>(0, 0) = newValue;
            deleteNewValues = true;
        }
        singleValue = true;
    }
    else if (PyFloat_Check(value)) // check if value is float single value
    {
        if (dataObjectType == ito::tComplex64)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat32);
            ito::float32 newValue = PyFloat_AsDouble(value);
            newValues->at<ito::float32>(0, 0) = newValue;
            deleteNewValues = true;
        }
        else if (dataObjectType == ito::tComplex128)
        {
            newValues = new ito::DataObject(1, 1, ito::tFloat64);
            ito::float64 newValue = PyFloat_AsDouble(value);
            newValues->at<ito::float64>(0, 0) = newValue;
            deleteNewValues = true;
        }
        singleValue = true;
    }
    else // error
    {
        Py_XDECREF(pyNewValues);
        pyNewValues = NULL;
        if (deleteNewValues)
        {
            DELETE_AND_SET_NULL(newValues);
        }
        PyErr_SetString(
            PyExc_TypeError,
            "Type of assigned value is invalid (real dataObject, real np.array or real scalar "
            "value)");
        return -1;
    }

    if (dataObjectType == ito::tComplex64)
    {
        if (newValues->getType() != ito::tFloat32)
        {
            // try to convert newValues to float32...
            ito::DataObject* newValuesFloat = new ito::DataObject();
            ito::RetVal ret;
            try
            {
                ret = newValues->convertTo(*newValuesFloat, ito::tFloat32);
            }
            catch (cv::Exception& exc)
            {
                ret = ito::RetVal::format(
                    ito::retError,
                    0,
                    "Cannot convert assigned value to a float32 dataObject (%s)",
                    exc.err.c_str());
            }

            if (ret == ito::retOk)
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                newValues = newValuesFloat;
                deleteNewValues = true;
            }
            else
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                DELETE_AND_SET_NULL(newValuesFloat);

                PythonCommon::transformRetValToPyException(ret);
                return -1;
            }
        }
    }
    else if (dataObjectType == ito::tComplex128)
    {
        if (newValues->getType() != ito::tFloat64)
        {
            // try to convert newValues to float64...
            ito::DataObject* newValuesFloat = new ito::DataObject();
            ito::RetVal ret;
            try
            {
                ret = newValues->convertTo(*newValuesFloat, ito::tFloat64);
            }
            catch (cv::Exception& exc)
            {
                ret = ito::RetVal::format(
                    ito::retError,
                    0,
                    "Cannot convert assigned value to a float64 dataObject (%s)",
                    exc.err.c_str());
            }

            if (ret == ito::retOk)
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                newValues = newValuesFloat;
                deleteNewValues = true;
            }
            else
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                DELETE_AND_SET_NULL(newValuesFloat);

                PythonCommon::transformRetValToPyException(ret);
                return -1;
            }
        }
    }

    const int dObjDims = self->dataObject->getDims();
    const int valDims = newValues->getDims();

    if (singleValue)
    {
    }
    else if (dObjDims == valDims) // error if same dimensions but different shape
    {
        if (self->dataObject->getSize() != newValues->getSize())
        {
            Py_XDECREF(pyNewValues);
            pyNewValues = NULL;
            if (deleteNewValues)
            {
                DELETE_AND_SET_NULL(newValues);
            }

            PyErr_Format(
                PyExc_IndexError,
                "The size of this dataObject and the assigned dataObject or np.array must be "
                "equal.");
            return -1;
        }
    }
    else // dObjDims must be greater than valDims
    {
        if (dObjDims > valDims && valDims == 2)
        {
            if (!(self->dataObject->getSize(self->dataObject->getDims() - 1) ==
                      newValues->getSize(1) &&
                  self->dataObject->getSize(self->dataObject->getDims() - 2) ==
                      newValues->getSize(0))) // last 2 dimensions are the same
            {
                Py_XDECREF(pyNewValues);
                pyNewValues = NULL;
                if (deleteNewValues)
                {
                    DELETE_AND_SET_NULL(newValues);
                }

                PyErr_SetString(PyExc_IndexError, "last 2 dimensions differs in size.");
                return -1;
            }
        }
        else
        {
            Py_XDECREF(pyNewValues);
            pyNewValues = NULL;
            if (deleteNewValues)
            {
                DELETE_AND_SET_NULL(newValues);
            }

            PyErr_SetString(
                PyExc_IndexError,
                "the shape of the data object must be greater than the shape of the values.");
            return -1;
        }
    }

    try
    {
        self->dataObject->setImag(*newValues);
    }
    catch (cv::Exception& exc)
    {
        Py_XDECREF(pyNewValues);
        pyNewValues = NULL;
        if (deleteNewValues)
        {
            DELETE_AND_SET_NULL(newValues);
        }

        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return -1;
    }

    self->dataObject->addToProtocol("Changed imaginary part of complex data object via imag.");

    Py_XDECREF(pyNewValues);
    pyNewValues = NULL;
    if (deleteNewValues)
    {
        DELETE_AND_SET_NULL(newValues);
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAbs_doc, "abs() -> dataObject \n\
\n\
Returns a new dataObject with the absolute values of this object. \n\
\n\
The absolute values in the resulting :class:`dataObject` are determined for \n\
both real (integer and floating point) and complex data types of this object. \n\
This method raises a ``TypeError`` for a ``rgba32`` data type. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this method is called for a dataObject of data type ``rgba32``. \n\
\n\
Returns \n\
------- \n\
absObj : dataObject \n\
    Array with the same size than this object, that contains the absolute values \n\
    of this object. If the data type of this object is complex, the returned \n\
    object has the corresponding floating point data type. Else, the data type \n\
    is unchanged. If this :class:`dataObject` has an unsigned integer data type, \n\
    its shallow copy is returned without any changes.");
PyObject* PythonDataObject::PyDataObject_abs(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject* d = self->dataObject;

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            ito::abs(*(d))); // resDataObj should always be the owner of its
                             // data, therefore base of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject->addToProtocol("Absolute values of calculated via abs().");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectArg_doc, "arg() -> dataObject \n\
\n\
Returns a new data object with the argument values of this complex type dataObject. \n\
\n\
This method calculates the argument value of each element in this :class:`dataObject`\n\
and returns these values as new dataObject with the same shape than this object. \n\
This object must be of complex data type (``complex128`` or ``complex64``). \n\
The output data type will be float then (``float64`` or ``float32``).\n\
\n\
Returns \n\
------- \n\
argObj : dataObject \n\
    is the argument function applied to all values of this dataObject.");
PyObject* PythonDataObject::PyDataObject_arg(PyDataObject* self, void* /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject* d = self->dataObject;

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(
            ito::arg(*(d))); // resDataObj should always be the owner of its
                             // data, therefore base of resultObject remains None
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    retObj->dataObject->addToProtocol(
        "Extracted phase/argument of a complex dataObject via arg().");
    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObj_mappingLength(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        return 0;
    }

    return self->dataObject->getTotal();
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_mappingGetElem(PyDataObject* self, PyObject* key)
{
    PyObject* retObj = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    int dims = self->dataObject->getDims();
    ito::Range* ranges = NULL;
    unsigned int* singlePointIdx = NULL;
    PyDataObject* mask = NULL;
    bool singlePoint = true;
    bool error = false;

    if (dims <= 0)
    {
        PyErr_SetString(PyExc_IndexError, "too many indices for array");
    }
    else if (dims == 1)
    {
        PyErr_SetString(PyExc_TypeError, "data object dimension must not be one, but two instead");
        return NULL;
    }

    if (PyDataObject_Check(key))
    {
        mask = (PyDataObject*)(key); // borrowed
        Py_INCREF(mask);
        Py_INCREF(key);
    }
    else if (PyArray_Check(key))
    {
        mask = (PyDataObject*)createPyDataObjectFromArray(key); // new reference
        Py_INCREF(key);

        if (!mask)
        {
            error = true;
        }
    }
    else
    {
        if (!PyTuple_Check(key))
        {
            key = PyTuple_Pack(1, key); // new reference
        }
        else
        {
            Py_INCREF(key);
        }

        if (PyTuple_Size(key) != dims)
        {
            Py_DECREF(key);
            PyErr_SetString(
                PyExc_TypeError, "length of key-tuple does not fit to dimension of data object");
            return NULL;
        }

        Py_ssize_t length = PyTuple_Size(key);
        ranges = new ito::Range[dims];
        singlePointIdx = new unsigned int[dims];
        PyObject* elem = NULL;
        int temp1;
        int axisSize;

        for (Py_ssize_t i = 0; i < length && !error; i++)
        {
            elem = PyTuple_GetItem(key, i);
            axisSize = self->dataObject->getSize(i);

            // check type of elem, must be int or stride
            if (PyLong_Check(elem))
            {
                int overflow;
                temp1 = PyLong_AsLongAndOverflow(elem, &overflow);

                // index -1 will be the last element, -2 the element before the last...
                if (!overflow && (temp1 < 0))
                {
                    temp1 = axisSize + temp1;
                }

                if (!overflow &&
                    (temp1 >= 0 &&
                     temp1 < axisSize)) // temp1 is still the virtual order, therefore check agains
                                        // the getSize-method which considers the transpose-flag
                {
                    ranges[i].start = temp1;
                    ranges[i].end = temp1 + 1;
                    singlePointIdx[i] = temp1;
                }
                else
                {
                    singlePointIdx[i] = 0;
                    error = true;
                    PyErr_Format(
                        PyExc_IndexError,
                        "index %i is out of bounds for axis %i with size %i.",
                        PyLong_AsLong(elem),
                        i,
                        axisSize);
                }
            }
            else if (PySlice_Check(elem))
            {
                singlePoint = false;

                Py_ssize_t start, stop, step, slicelength;
                if (PySlice_GetIndicesEx(elem, axisSize, &start, &stop, &step, &slicelength) == 0)
                {
                    if (slicelength < 1)
                    {
                        error = true;
                        PyErr_SetString(PyExc_IndexError, "length of slice must be >= 1");
                    }
                    else if (step != 1)
                    {
                        error = true;
                        PyErr_SetString(PyExc_IndexError, "step size must be one.");
                    }
                    else
                    {
                        ranges[i].start = start;
                        ranges[i].end =
                            stop; // stop already points one index after the last index within the
                                  // range, this is the same definition than openCV has.
                    }
                }
                else
                {
                    error = true;
                    // error is already set by command
                    // PyErr_SetString(PyExc_TypeError, "no valid start and stop element can be
                    // found for given slice");
                }
            }
            else
            {
                error = true;
                PyErr_SetString(
                    PyExc_TypeError,
                    "range tuple element is neither of type integer nor of type slice");
            }
        }
    }

    if (!error)
    {
        if (mask)
        {
            PyDataObject* retObj2 = PythonDataObject::createEmptyPyDataObject(); // new reference
            try
            {
                retObj2->dataObject =
                    new ito::DataObject(self->dataObject->at(*(mask->dataObject)));

                if (!retObj2->dataObject->getOwnData())
                {
                    PyDataObject_SetBase(retObj2, (PyObject*)self);
                }

                retObj = (PyObject*)retObj2;
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                Py_DECREF(retObj2);
                retObj2 = NULL;
            }
        }
        else if (singlePoint)
        {
            retObj = PyDataObj_At(self->dataObject, singlePointIdx);
        }
        else
        {
            PyDataObject* retObj2 = PythonDataObject::createEmptyPyDataObject(); // new reference

            try
            {
                retObj2->dataObject = new ito::DataObject(self->dataObject->at(ranges));

                if (!retObj2->dataObject->getOwnData())
                {
                    PyDataObject_SetBase(retObj2, (PyObject*)self);
                }

                retObj = (PyObject*)retObj2;
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                Py_DECREF(retObj2);
                retObj2 = NULL;
            }
        }
    }

    DELETE_AND_SET_NULL_ARRAY(ranges);
    DELETE_AND_SET_NULL_ARRAY(singlePointIdx);

    Py_DECREF(key);
    Py_XDECREF(mask);

    return retObj;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObj_mappingSetElem(PyDataObject* self, PyObject* key, PyObject* value)
{
    DataObject dataObj;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return -1;
    }

    int dims = self->dataObject->getDims();
    ito::Range* ranges = NULL;
    unsigned int* idx = NULL; // redundant to range, if only single indices are addressed
    PyDataObject* mask = NULL;

    if (dims <= 0)
    {
        PyErr_SetString(PyExc_TypeError, "empty data object.");
        return -1;
    }

    if (PyDataObject_Check(key))
    {
        mask = (PyDataObject*)key;
        Py_INCREF(key); // increment reference
        Py_INCREF(mask);
    }
    else if (PyArray_Check(key))
    {
        mask = (PyDataObject*)createPyDataObjectFromArray(key); // new reference
        Py_INCREF(key);

        if (!mask)
        {
            return -1;
        }
    }
    else
    {
        if (!PyTuple_Check(key))
        {
            key = PyTuple_Pack(1, key); // new reference
        }
        else
        {
            Py_INCREF(key); // increment reference
        }

        if (PyTuple_Size(key) != dims)
        {
            Py_DECREF(key);
            PyErr_SetString(
                PyExc_TypeError, "length of key-tuple does not fit to dimension of data object");
            return -1;
        }
    }

    Py_ssize_t length = 0;
    bool error = false;
    bool containsSlices = false;

    if (!mask)
    {
        length = PyTuple_Size(key);
        ranges = new ito::Range[dims];
        idx = new unsigned int[dims];

        PyObject* elem = NULL;
        int temp1;
        int axisSize;

        for (Py_ssize_t i = 0; i < length && !error; i++)
        {
            elem = PyTuple_GetItem(key, i); // borrowed reference
            axisSize = self->dataObject->getSize(i);

            // check type of elem, must be int or stride
            if (PyLong_Check(elem))
            {
                int overflow;
                temp1 = PyLong_AsLongAndOverflow(elem, &overflow);

                // index -1 will be the last element, -2 the element before the last...
                if (!overflow && (temp1 < 0))
                {
                    temp1 = axisSize + temp1;
                }

                if (!overflow && (temp1 >= 0 && temp1 < axisSize))
                {
                    ranges[i].start = temp1;
                    ranges[i].end = temp1 + 1;
                    idx[i] = temp1;
                }
                else
                {
                    error = true;
                    PyErr_Format(
                        PyExc_IndexError,
                        "index %i is out of bounds for axis %i with size %i",
                        PyLong_AsLong(elem),
                        i,
                        axisSize);
                }
            }
            else if (PySlice_Check(elem))
            {
                containsSlices = true;
                Py_ssize_t start, stop, step, slicelength;

                if (PySlice_GetIndicesEx(elem, axisSize, &start, &stop, &step, &slicelength) == 0)
                {
                    if (step != 1)
                    {
                        error = true;
                        PyErr_SetString(PyExc_TypeError, "step size must be one.");
                    }
                    else
                    {
                        ranges[i].start = start;
                        ranges[i].end =
                            stop; // stop already points one index after the last index within the
                                  // range, this is the same definition than openCV has.
                    }
                }
                else
                {
                    error = true;
                    // error is already set by command
                    // PyErr_SetString(PyExc_TypeError, "no valid start and stop element can be
                    // found for given slice");
                }
            }
            else
            {
                error = true;
                PyErr_SetString(
                    PyExc_TypeError,
                    "range tuple element is neither of type integer nor of type slice");
            }
        }
    }

    if (containsSlices) // key is no mask data object
    {
        if (!error)
        {
            try
            {
                // self->dataObject in readLock, dataObj will become readLock, too
                dataObj = self->dataObject->at(ranges); // self->dataObject must not be locked for
                                                        // writing, since dataObj will read it
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                error = true;
            }
        }

        // no parse value and assign it to dataObj
        if (!error)
        {
            if (PyLong_Check(value))
            {
                int overflow;
                long l = PyLong_AsLongAndOverflow(value, &overflow);

                if (overflow == 0)
                {
                    dataObj = (ito::int32)l;
                }
                else if (overflow == -1)
                {
                    PyErr_SetString(
                        PyExc_ValueError, "value exceeds the negative boundary of int32.");
                    error = true;
                }
                else // overflow = 1
                {
                    dataObj = (ito::uint32)PyLong_AsUnsignedLong(value);
                }
            }
            else if (PyFloat_Check(value))
            {
                dataObj = PyFloat_AsDouble(value);
            }
            else if (PyComplex_Check(value))
            {
                dataObj = complex128(PyComplex_RealAsDouble(value), PyComplex_ImagAsDouble(value));
            }
            else if (Py_TYPE(value) == &PyDataObjectType)
            {
                try
                {
                    ((PyDataObject*)value)->dataObject->deepCopyPartial(dataObj);
                }
                catch (cv::Exception& exc)
                {
                    PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                    error = true;
                }
            }
            else if (Py_TYPE(value) == &(PythonRgba::PyRgbaType))
            {
                if (dataObj.getType() == ito::tRGBA32)
                {
                    dataObj = ((PythonRgba::PyRgba*)value)->rgba;
                }
                else
                {
                    PyErr_SetString(
                        PyExc_TypeError,
                        "An assignment of type itom.rgba is only possible for data objects of type "
                        "rgba32");
                    error = true;
                }
            }
            else if (PythonDateTime::PyDateTime_CheckExt(value))
            {
                if (dataObj.getType() == ito::tDateTime)
                {
                    bool ok = true;
                    dataObj = PythonDateTime::GetDateTime(value, ok);

                    if (!ok)
                    {
                        error = true;
                    }
                }
                else
                {
                    PyErr_SetString(
                        PyExc_TypeError,
                        "An assignment of type datetime is only possible for data objects of type "
                        "datetime");
                    error = true;
                }
            }
            else if (PythonDateTime::PyTimeDelta_CheckExt(value))
            {
                if (dataObj.getType() == ito::tTimeDelta)
                {
                    bool ok = true;
                    dataObj = PythonDateTime::GetTimeDelta(value, ok);

                    if (!ok)
                    {
                        error = true;
                    }
                }
                else
                {
                    PyErr_SetString(
                        PyExc_TypeError,
                        "An assignment of type timedelta is only possible for data objects of type "
                        "timedelta");
                    error = true;
                }
            }
            else
            {
                // try to convert the assigned value to a numpy array and then read the values
                int npTypenum = getNpTypeFromDataObjectType(dataObj.getType());

                if (dataObj.getDims() < 2)
                {
                    PyErr_SetString(PyExc_TypeError, "the destination data object is empty.");
                }
                else if (npTypenum >= 0)
                {
                    int dims = dataObj.getDims();
                    PyObject* npArray = PyArray_ContiguousFromAny(value, npTypenum, 1, dims);

                    if (npArray)
                    {
                        PyArrayObject* npArray_ = (PyArrayObject*)npArray;
                        int npdims = PyArray_NDIM(npArray_);

                        const npy_intp* npArrayShape = PyArray_SHAPE(npArray_);
                        int* map_dims_to_npdims = new int[dims];

                        if (dataObj.getTotal() != PyArray_SIZE(npArray_))
                        {
                            PyErr_Format(
                                PyExc_ValueError,
                                "size of given data does not fit to size of data object");
                            error = true;
                        }

                        int c = 0;

                        if (!error)
                        {
                            // check dimensions
                            for (int d = 0; d < dims; ++d)
                            {
                                if ((c < npdims) && (npArrayShape[c] == dataObj.getSize(d)))
                                {
                                    map_dims_to_npdims[d] = c;
                                    c++;
                                }
                                else if (dataObj.getSize(d) == 1) // this dimension is not required
                                                                  // in np-array
                                {
                                    // d.th dimension of dataObj is not
                                    // available in np-array (squeezed)
                                    map_dims_to_npdims[d] = -1;
                                }
                                else
                                {
                                    PyErr_Format(
                                        PyExc_ValueError,
                                        "%i. dimension of given data does not fit to given "
                                        "dimension. %i obtained, %i required",
                                        d,
                                        npArrayShape[c],
                                        dataObj.getSize(d));
                                    error = true;
                                }
                            }
                        }

                        if (!error)
                        {
                            npy_intp* ind = new npy_intp[npdims];
                            memset(ind, 0, npdims * sizeof(npy_intp));
                            const void* npPtr = NULL;
                            int numPlanes = dataObj.getNumPlanes();
                            cv::Mat* mat;

                            // the following part is inspired by DataObject::matNumToIdx:

                            int orgPlaneSize = 1;

                            for (int nDim = 1; nDim < dims - 2; nDim++)
                            {
                                orgPlaneSize *= dataObj.getSize(nDim);
                            }

                            int tMatNum;
                            int planeSize;

                            for (int plane = 0; plane < numPlanes; ++plane)
                            {
                                mat = dataObj.getCvPlaneMat(plane);

                                tMatNum = plane;
                                planeSize = orgPlaneSize;
                                for (int nDim = 0; nDim < dims - 2; nDim++)
                                {
                                    if (map_dims_to_npdims[nDim] >= 0)
                                    {
                                        ind[map_dims_to_npdims[nDim]] = tMatNum / planeSize;
                                    }
                                    tMatNum %= planeSize;
                                    planeSize /= dataObj.getSize(nDim + 1);
                                }


                                for (int row = 0; row < mat->rows; ++row)
                                {
                                    if (map_dims_to_npdims[dims - 2] >= 0)
                                    {
                                        ind[map_dims_to_npdims[dims - 2]] = row;
                                    }
                                    npPtr = PyArray_GetPtr(npArray_, ind);
                                    memcpy(mat->ptr(row), npPtr, mat->cols * mat->elemSize());
                                }
                            }

                            DELETE_AND_SET_NULL_ARRAY(ind);
                        }

                        DELETE_AND_SET_NULL_ARRAY(map_dims_to_npdims);

                        Py_DECREF(npArray);
                    }
                    else
                    {
                        // pyerror is already set
                        error = true;
                    }
                }
                else
                {
                    PyErr_SetString(
                        PyExc_TypeError,
                        "Allowed values for assignment: int, float, complex, itom.rgba, "
                        "datetime.datetime, datetime.timedelta, np.ndarray or itom.dataObject");
                    error = true;
                }
            }
            /*else
            {
                error = true;
                PyErr_SetString(PyExc_TypeError, "assign value has no of the following types:
            integer, floating point, complex, rgba (type rgba32 only) or data object");
            }*/
        }
    }
    else if (mask)
    {
        void* valuePtr;
        ito::tDataType fromType = ito::tInt8;
        int32 value1 = 0;
        float64 value2 = 0.0;
        complex128 value3 = 0.0;
        ito::Rgba32 value4;
        ito::DateTime value5;
        ito::TimeDelta value6;

        if (!error)
        {
            if (PyLong_Check(value))
            {
                value1 = PyLong_AsLong(value);
                valuePtr = static_cast<void*>(&value1);
                fromType = ito::tInt32;
            }
            else if (PyFloat_Check(value))
            {
                value2 = PyFloat_AsDouble(value);
                valuePtr = static_cast<void*>(&value2);
                fromType = ito::tFloat64;
            }
            else if (PyComplex_Check(value))
            {
                value3 = complex128(PyComplex_RealAsDouble(value), PyComplex_ImagAsDouble(value));
                valuePtr = static_cast<void*>(&value3);
                fromType = ito::tComplex128;
            }
            else if (Py_TYPE(value) == &ito::PythonRgba::PyRgbaType)
            {
                ito::PythonRgba::PyRgba* rgba = (ito::PythonRgba::PyRgba*)(value);
                fromType = ito::tRGBA32;
                value4 = rgba->rgba;
                // will be valid until end of function since this
                // is a direct access to the underlying structure.
                valuePtr = static_cast<void*>(&value4);
            }
            else if (PythonDateTime::PyDateTime_CheckExt(value))
            {
                bool ok = true;
                value5 = PythonDateTime::GetDateTime(value, ok);
                fromType = ito::tDateTime;
                valuePtr = static_cast<void*>(&value5);
            }
            else if (PythonDateTime::PyTimeDelta_CheckExt(value))
            {
                bool ok = true;
                value6 = PythonDateTime::GetTimeDelta(value, ok);
                fromType = ito::tDateTime;
                valuePtr = static_cast<void*>(&value6);
            }
            else
            {
                error = true;
                PyErr_SetString(
                    PyExc_TypeError,
                    "assign value has no of the following types: integer, floating point, complex");
            }
        }

        if (!error && fromType != ito::tInt8)
        {
            ito::RetVal retval2;

            try
            {
                switch (self->dataObject->getType())
                {
                case ito::tUInt8:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<uint8>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tInt8:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<int8>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tUInt16:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<uint16>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tInt16:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<int16>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tUInt32:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<uint32>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tInt32:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<int32>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tRGBA32:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<ito::Rgba32>(fromType, valuePtr),
                        *(mask->dataObject));
                    break;
                case ito::tDateTime:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<ito::DateTime>(fromType, valuePtr),
                        *(mask->dataObject));
                    break;
                case ito::tTimeDelta:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<ito::TimeDelta>(fromType, valuePtr),
                        *(mask->dataObject));
                    break;
                case ito::tFloat32:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<float32>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tFloat64:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<float64>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tComplex64:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<complex64>(fromType, valuePtr), *(mask->dataObject));
                    break;
                case ito::tComplex128:
                    retval2 = self->dataObject->setTo(
                        ito::numberConversion<complex128>(fromType, valuePtr), *(mask->dataObject));
                    break;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                error = true;
            }

            if (PythonCommon::transformRetValToPyException(retval2) == false)
            {
                error = true;
            }
        }
    }
    else // contains no slices and key is no mask
    {
        void* valuePtr;
        ito::tDataType fromType = ito::tInt8;
        int32 value1 = 0;
        float64 value2 = 0.0;
        complex128 value3 = 0.0;
        ito::Rgba32 value4;
        ito::DateTime value5;
        ito::TimeDelta value6;

        if (!error)
        {
            if (PyLong_Check(value))
            {
                value1 = PyLong_AsLong(value);
                valuePtr = static_cast<void*>(&value1);
                fromType = ito::tInt32;
            }
            else if (PyFloat_Check(value))
            {
                value2 = PyFloat_AsDouble(value);
                valuePtr = static_cast<void*>(&value2);
                fromType = ito::tFloat64;
            }
            else if (PyComplex_Check(value))
            {
                value3 = complex128(PyComplex_RealAsDouble(value), PyComplex_ImagAsDouble(value));
                valuePtr = static_cast<void*>(&value3);
                fromType = ito::tComplex128;
            }
            else if (Py_TYPE(value) == &PyDataObjectType)
            {
                fromType = ito::tInt8;

                try
                {
                    dataObj = self->dataObject->at(ranges); // dataObj in readLock
                    ((PyDataObject*)value)->dataObject->deepCopyPartial(dataObj);
                }
                catch (cv::Exception& exc)
                {
                    PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                    error = true;
                }
            }
            else if (Py_TYPE(value) == &ito::PythonRgba::PyRgbaType)
            {
                ito::PythonRgba::PyRgba* rgba = (ito::PythonRgba::PyRgba*)(value);
                fromType = ito::tRGBA32;
                value4 = rgba->rgba;
                // will be valid until end of function since this
                // is a direct access to the underlying structure.
                valuePtr = static_cast<void*>(&value4);
            }
            else if (PythonDateTime::PyDateTime_CheckExt(value))
            {
                bool ok = true;
                value5 = PythonDateTime::GetDateTime(value, ok);
                fromType = ito::tDateTime;
                valuePtr = static_cast<void*>(&value5);

                if (!ok)
                {
                    error = true;
                }
            }
            else if (PythonDateTime::PyTimeDelta_CheckExt(value))
            {
                bool ok = true;
                value6 = PythonDateTime::GetTimeDelta(value, ok);
                fromType = ito::tTimeDelta;
                valuePtr = static_cast<void*>(&value6);

                if (!ok)
                {
                    error = true;
                }
            }
            else
            {
                error = true;
                PyErr_SetString(
                    PyExc_TypeError,
                    "assign value has none of the following types: integer, floating point, complex, "
                    "dataObject, rgba, datetime, timedelta");
            }
        }

        if (!error && fromType != ito::tInt8)
        {
            try
            {
                switch (self->dataObject->getType())
                {
                case ito::tUInt8:
                    self->dataObject->at<uint8>(idx) =
                        ito::numberConversion<uint8>(fromType, valuePtr);
                    break;
                case ito::tInt8:
                    self->dataObject->at<int8>(idx) =
                        ito::numberConversion<int8>(fromType, valuePtr);
                    break;
                case ito::tUInt16:
                    self->dataObject->at<uint16>(idx) =
                        ito::numberConversion<uint16>(fromType, valuePtr);
                    break;
                case ito::tInt16:
                    self->dataObject->at<int16>(idx) =
                        ito::numberConversion<int16>(fromType, valuePtr);
                    break;
                case ito::tUInt32:
                    self->dataObject->at<uint32>(idx) =
                        ito::numberConversion<uint32>(fromType, valuePtr);
                    break;
                case ito::tInt32:
                    self->dataObject->at<int32>(idx) =
                        ito::numberConversion<int32>(fromType, valuePtr);
                    break;
                case ito::tRGBA32:
                    self->dataObject->at<ito::Rgba32>(idx) =
                        ito::numberConversion<ito::Rgba32>(fromType, valuePtr);
                    break;
                case ito::tDateTime:
                    self->dataObject->at<ito::DateTime>(idx) =
                        ito::numberConversion<ito::DateTime>(fromType, valuePtr);
                    break;
                case ito::tTimeDelta:
                    self->dataObject->at<ito::TimeDelta>(idx) =
                        ito::numberConversion<ito::TimeDelta>(fromType, valuePtr);
                    break;
                case ito::tFloat32:
                    self->dataObject->at<float32>(idx) =
                        ito::numberConversion<float32>(fromType, valuePtr);
                    break;
                case ito::tFloat64:
                    self->dataObject->at<float64>(idx) =
                        ito::numberConversion<float64>(fromType, valuePtr);
                    break;
                case ito::tComplex64:
                    self->dataObject->at<complex64>(idx) =
                        ito::numberConversion<complex64>(fromType, valuePtr);
                    break;
                case ito::tComplex128:
                    self->dataObject->at<complex128>(idx) =
                        ito::numberConversion<complex128>(fromType, valuePtr);
                    break;
                }
            }
            catch (cv::Exception& exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                error = true;
            }
        }
    }

    Py_DECREF(key);
    Py_XDECREF(mask);
    DELETE_AND_SET_NULL_ARRAY(ranges);
    DELETE_AND_SET_NULL_ARRAY(idx);

    return error ? -1 : 0;
}

//-------------------------------------------------------------------------------------
RetVal PythonDataObject::parseTypeNumber(int typeno, char& typekind, int& itemsize)
{
    switch (typeno)
    {
    case ito::tUInt8:
        typekind = NPY_UNSIGNEDLTR;
        itemsize = sizeof(uint8);
        break;
    case ito::tInt8:
        typekind = NPY_SIGNEDLTR;
        itemsize = sizeof(int8);
        break;
    case ito::tUInt16:
        typekind = NPY_UNSIGNEDLTR;
        itemsize = sizeof(uint16);
        break;
    case ito::tInt16:
        typekind = NPY_SIGNEDLTR;
        itemsize = sizeof(int16);
        break;
    case ito::tUInt32:
    case ito::tRGBA32:
        typekind = NPY_UNSIGNEDLTR;
        itemsize = sizeof(uint32);
        break;
    case ito::tInt32:
        typekind = NPY_SIGNEDLTR;
        itemsize = sizeof(int32);
        break;
    case ito::tFloat32:
        typekind = NPY_FLOATINGLTR;
        itemsize = sizeof(float32);
        break;
    case ito::tFloat64:
        typekind = NPY_FLOATINGLTR;
        itemsize = sizeof(float64);
        break;
    case ito::tComplex64:
        typekind = NPY_COMPLEXLTR;
        itemsize = sizeof(complex64);
        break;
    case ito::tComplex128:
        typekind = NPY_COMPLEXLTR;
        itemsize = sizeof(complex128);
        break;
    case ito::tDateTime: {
        // todo: maybe kind and size can be hard coded
        PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_DATETIME);
        typekind = descr->kind; // NPY_DATETIMELTR
        itemsize = descr->elsize; // 8
        Py_DECREF(descr);

        // PyDatetimeScalarObject
        break;
    }
    case ito::tTimeDelta: {
        // todo: maybe kind and size can be hard coded
        PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_TIMEDELTA);
        typekind = descr->kind; // NPY_TIMEDELTALTR
        itemsize = descr->elsize; // 8
        Py_DECREF(descr);
        break;
    }
    default:
        return RetVal(retError, 0, "type conversion failed");
    }

    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------
int PythonDataObject::getDObjTypeOfNpArray(char typekind, int itemsize)
{
    if (typekind == NPY_SIGNEDLTR)
    {
        switch (itemsize)
        {
        case 1:
            return ito::tInt8;
        case 2:
            return ito::tInt16;
        case 4:
            return ito::tInt32;
        }
    }
    else if (typekind == NPY_UNSIGNEDLTR)
    {
        switch (itemsize)
        {
        case 1:
            return ito::tUInt8;
        case 2:
            return ito::tUInt16;
        case 4:
            return ito::tUInt32;
        }
    }
    else if (typekind == NPY_FLOATINGLTR)
    {
        switch (itemsize)
        {
        case 4:
            return ito::tFloat32;
        case 8:
            return ito::tFloat64;
        }
    }
    else if (typekind == NPY_COMPLEXLTR)
    {
        switch (itemsize)
        {
        case 8:
            return ito::tComplex64;
        case 16:
            return ito::tComplex128;
        }
    }
    else if (typekind == NPY_DATETIMELTR)
    {
        switch (itemsize)
        {
        case 8:
            return ito::tDateTime;
        }
    }
    else if (typekind == NPY_TIMEDELTALTR)
    {
        switch (itemsize)
        {
        case 8:
            return ito::tTimeDelta;
        }
    }

    return -1;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::getCompatibleDObjTypeOfNpArray(char typekind, int itemsize)
{
    if (typekind == 'b')
    {
        // bool object

        switch (itemsize)
        {
        case 1:
            return ito::tUInt8; // convert bool to uint8
        }
    }

    return getDObjTypeOfNpArray(typekind, itemsize);
}

//-------------------------------------------------------------------------------------
std::string PythonDataObject::getNpDTypeStringFromNpDTypeEnum(const int type)
{
    std::string typeStr;

    switch (type)
    {
    case NPY_BOOL:
        typeStr = "bool";
        break;
    case NPY_BYTE:
        typeStr = "int8";
        break;
    case NPY_UBYTE:
        typeStr = "uint8";
        break;
    case NPY_SHORT:
        typeStr = "int16";
        break;
    case NPY_USHORT:
        typeStr = "uint16";
        break;
    case NPY_INT:
        typeStr = "int32";
        break;
    case NPY_UINT:
        typeStr = "uint32";
        break;
    case NPY_LONG:
        typeStr = "long";
        break;
    case NPY_ULONG:
        typeStr = "ulong";
        break;
    case NPY_LONGLONG:
        typeStr = "int64";
        break;
    case NPY_ULONGLONG:
        typeStr = "uint64";
        break;
    case NPY_FLOAT:
        typeStr = "float32";
        break;
    case NPY_DOUBLE:
        typeStr = "float64";
        break;
    case NPY_LONGDOUBLE:
        typeStr = "longdouble";
        break;
    case NPY_CFLOAT:
        typeStr = "complex64";
        break;
    case NPY_CDOUBLE:
        typeStr = "complex128";
        break;
    case NPY_CLONGDOUBLE:
        typeStr = "clongdouble";
        break;
    case NPY_OBJECT:
        typeStr = "object";
        break;
    case NPY_STRING:
        typeStr = "string";
        break;
    case NPY_UNICODE:
        typeStr = "unicode";
        break;
    case NPY_VOID:
        typeStr = "void";
        break;
    case NPY_DATETIME:
        typeStr = "datetime";
        break;
    case NPY_TIMEDELTA:
        typeStr = "timedelta";
        break;
    case NPY_HALF:
        typeStr = "half";
        break;
    case NPY_NTYPES:
        typeStr = "ntypes";
        break;
    case NPY_NOTYPE:
        typeStr = "notype";
        break;
    case NPY_USERDEF:
        typeStr = "userdef";
        break;
    default:
        typeStr = "others";
        break;
    }

    return typeStr;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::getNpTypeFromDataObjectType(int type)
{
    int npTypenum;

    switch (type)
    {
    case ito::tInt8:
        npTypenum = NPY_BYTE;
        break;
    case ito::tUInt8:
        npTypenum = NPY_UBYTE;
        break;
    case ito::tInt16:
        npTypenum = NPY_SHORT;
        break;
    case ito::tUInt16:
        npTypenum = NPY_USHORT;
        break;
    case ito::tInt32:
        npTypenum = NPY_INT;
        break;
    case ito::tUInt32:
        npTypenum = NPY_UINT;
        break;
    case ito::tRGBA32:
        npTypenum = NPY_UINT;
        break;
    case ito::tFloat32:
        npTypenum = NPY_FLOAT;
        break;
    case ito::tFloat64:
        npTypenum = NPY_DOUBLE;
        break;
    case ito::tComplex64:
        npTypenum = NPY_CFLOAT;
        break;
    case ito::tComplex128:
        npTypenum = NPY_CDOUBLE;
        break;
    case ito::tDateTime:
        npTypenum = NPY_DATETIME;
        break;
    case ito::tTimeDelta:
        npTypenum = NPY_TIMEDELTA;
        break;
    default:
        npTypenum = -1;
    }

    return npTypenum;
}

//-------------------------------------------------------------------------------------
/* npNdArray and dataObject must be allocated with the same type and shape.*/
ito::RetVal PythonDataObject::copyNpArrayValuesToDataObject(
    PyArrayObject* npNdArray, ito::DataObject* dataObject, ito::tDataType type)
{
    ito::RetVal retVal;
    void* data = PyArray_DATA(npNdArray);

    int numMats = dataObject->calcNumMats();
    int matIndex = 0;
    int c = 0;
    cv::Mat* mat = NULL;
    int m, n;

    for (int i = 0; i < numMats; i++)
    {
        matIndex = dataObject->seekMat(i, numMats);
        mat = (cv::Mat*)(dataObject->get_mdata())[matIndex];

        switch (type)
        {
        case ito::tInt8: {
            int8* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<int8>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<int8*>(data))[c++];
                }
            }
        }
        break;
        case ito::tUInt8: {
            uint8* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<uint8>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<uint8*>(data))[c++];
                }
            }
        }
        break;
        case ito::tInt16: {
            int16* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<int16>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<int16*>(data))[c++];
                }
            }
        }
        break;
        case ito::tUInt16: {
            uint16* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<uint16>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<uint16*>(data))[c++];
                }
            }
        }
        break;
        case ito::tInt32: {
            int32* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<int32>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<int32*>(data))[c++];
                }
            }
        }
        break;
        case ito::tUInt32: {
            uint32* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<uint32>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<uint32*>(data))[c++];
                }
            }
        }
        break;
        case ito::tRGBA32: {
            ito::Rgba32* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<ito::Rgba32>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n].rgba = (reinterpret_cast<uint32*>(data))[c++];
                }
            }
        }
        break;
        case ito::tFloat32: {
            float32* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<float32>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<float32*>(data))[c++];
                }
            }
        }
        break;
        case ito::tFloat64: {
            float64* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<float64>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<float64*>(data))[c++];
                }
            }
        }
        break;
        case ito::tComplex64: {
            complex64* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<complex64>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<complex64*>(data))[c++];
                }
            }
        }
        break;
        case ito::tComplex128: {
            complex128* rowPtr;
            for (m = 0; m < mat->rows; m++)
            {
                rowPtr = mat->ptr<complex128>(m);
                for (n = 0; n < mat->cols; n++)
                {
                    rowPtr[n] = (reinterpret_cast<complex128*>(data))[c++];
                }
            }
        }
        break;
        case ito::tDateTime: {
            const npy_datetime* td = reinterpret_cast<npy_datetime*>(data);
            ito::DateTime* rowPtr;
            PyArray_Descr* dtype = PyArray_DESCR(npNdArray);

            const auto md = (PyArray_DatetimeDTypeMetaData*)(dtype->c_metadata);
            // timezone is ignored in numpy. If dataObject contains a timezone, ignore it and raise a
            // warning.

            if (md == nullptr)
            {
                retVal += ito::RetVal(ito::retError, 0, "Failed to read the time unit of the numpy.ndarray.");
            }
            else
            {
                for (m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<DateTime>(m);
                    for (n = 0; n < mat->cols; n++)
                    {
                        if (!PythonDateTime::NpyDatetime2itoDatetime(td[c++], md->meta, rowPtr[n]))
                        {
                            PyErr_Clear();
                            retVal += ito::RetVal(ito::retError, 0, "invalid or unsupported datetime format (e.g. NaT).");
                        }
                    }
                }
            }
        }
        break;
        case ito::tTimeDelta: {
            const npy_timedelta* td = reinterpret_cast<npy_timedelta*>(data);
            ito::TimeDelta* rowPtr;
            PyArray_Descr* dtype = PyArray_DESCR(npNdArray);

            const auto md = (PyArray_DatetimeDTypeMetaData*)(dtype->c_metadata);
            // timezone is ignored in numpy. If dataObject contains a timezone, ignore it and raise a
            // warning.

            if (md == nullptr)
            {
                retVal += ito::RetVal(ito::retError, 0, "Failed to read the time unit of the numpy.ndarray.");
            }
            else
            {
                for (m = 0; m < mat->rows; m++)
                {
                    rowPtr = mat->ptr<TimeDelta>(m);

                    for (n = 0; n < mat->cols; n++)
                    {
                        if (!PythonDateTime::NpyTimedelta2itoTimedelta(td[c++], md->meta, rowPtr[n]))
                        {
                            PyErr_Clear();
                            retVal += ito::RetVal(ito::retError, 0, "invalid timedelta format or unsupported value (NaT).");
                        }
                    }
                }
            }
        }
        break;
        default:
            retVal += ito::RetVal(ito::retError, 0, "unknown dtype");
        }
    }

    return retVal;
}

//---------------------------------------Get / Set metaDict ---------------------------
PyDoc_STRVAR(
    dataObjectAttrTagDict_doc,
    "dict : Gets or sets a dictionary with all meta information of this dataObject. \n\
\n\
The dictionary contains the following key-value-pairs. If a new dictionary \n\
is set to this attribute, all these values must be contained in the dict: \n\
\n\
* axisOffsets : List with offsets of each axis. \n\
* axisScales : List with the scales of each axis. \n\
* axisUnits : List with the unit strings of each axis. \n\
* axisDescriptions : List with the description strings of each axis. \n\
* tags : Dictionary with all tags including the tag **protocol** if at least \n\
  one protocol entry has been added using :meth:`addToProtocol`. \n\
* valueOffset : Offset of each value (0.0). \n\
* valueScale : Scale of each value (1.0). \n\
* valueDescription : Description of the values. \n\
* valueUnit : The unit string of the values. \n\
\n\
This attribute was read-only until itom 4.0. It is settable from itom 4.1 on. \n\
\n\
See Also \n\
-------- \n\
addToProtocol, axisOffsets, axisScales, axisUnits, axisDescriptions, \n\
valueUnit, valueDescription, tags");
PyObject* PythonDataObject::PyDataObject_getTagDict(PyDataObject* self, void* /*closure*/)
{
    PyObject* item = NULL;

    if (self == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "dataObject is NULL");
        return NULL;
    }

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "content of dataObject is NULL");
        return NULL;
    }

    PyObject* dict = PyDict_New();

    DataObject* dObj = self->dataObject;
    int tagSize = dObj->getTagListSize();
    // std::string tempString;
    DataObjectTagType tempTag;
    std::string tempKey;
    bool validOp;

    // 1. tags (here it is bad to use the tags-getter, since this returns a dict_proxy, which cannot
    // directly be pickled
    PyObject* tempTagDict = PyDict_New();

    for (int i = 0; i < tagSize; i++)
    {
        tempKey = dObj->getTagKey(i, validOp);

        if (validOp)
        {
            // tempString = dObj->getTag(tempKey, validOp);
            // if (validOp) PyDict_SetItem(tempTagDict, PyUnicode_FromString(tempKey.data()),
            // PyUnicode_FromString(tempString.data()));
            dObj->getTagByIndex(i, tempKey, tempTag);

            if (tempTag.getType() == DataObjectTagType::typeDouble)
            {
                item = PyFloat_FromDouble(tempTag.getVal_ToDouble());
                PyDict_SetItemString(tempTagDict, tempKey.data(), item);
                Py_DECREF(item);
            }
            else
            {
                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(
                    tempTag.getVal_ToString().data());
                PyDict_SetItemString(tempTagDict, tempKey.data(), item);
                Py_DECREF(item);
            }
        }
    }
    // 1. tags
    PyDict_SetItemString(dict, "tags", tempTagDict);
    Py_DECREF(tempTagDict);

    // 2. axisScales
    item = PyDataObject_getAxisScales(self, NULL);
    PyDict_SetItemString(dict, "axisScales", item);
    Py_DECREF(item);

    // 3. axisOffsets
    item = PyDataObject_getAxisOffsets(self, NULL);
    PyDict_SetItemString(dict, "axisOffsets", item);
    Py_DECREF(item);

    // 4. axisDescriptions
    item = PyDataObject_getAxisDescriptions(self, NULL);
    PyDict_SetItemString(dict, "axisDescriptions", item);
    Py_DECREF(item);

    // 5. axisUnits
    item = PyDataObject_getAxisUnits(self, NULL);
    PyDict_SetItemString(dict, "axisUnits", item);
    Py_DECREF(item);

    // 6. valueUnit
    item = PyDataObject_getValueUnit(self, NULL);
    PyDict_SetItemString(dict, "valueUnit", item);
    Py_DECREF(item);

    // 7. valueDescription
    item = PyDataObject_getValueDescription(self, NULL);
    PyDict_SetItemString(dict, "valueDescription", item);
    Py_DECREF(item);

    // 8.
    item = PyDataObject_getValueOffset(self, NULL);
    PyDict_SetItemString(dict, "valueOffset", item);
    Py_DECREF(item);

    // 9.
    item = PyDataObject_getValueScale(self, NULL);
    PyDict_SetItemString(dict, "valueScale", item);
    Py_DECREF(item);

    return dict;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setTagDict(
    PyDataObject* self, PyObject* value, void* /*closure*/)
{
    void* closure = NULL;

    if (self == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "dataObject is NULL");
        return -1;
    }

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "content of dataObject is NULL");
        return -1;
    }

    if (!PyDict_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "The input value must be a dictionary");
        return -1;
    }

    PyObject* tags = PyDict_GetItemString(value, "tags"); // borrowed
    if (!PyDict_Check(tags))
    {
        PyErr_SetString(PyExc_TypeError, "tags must be a dictionary");
        return -1;
    }

    PyObject* axisScales = PyDict_GetItemString(value, "axisScales"); // borrowed
    if (!PySequence_Check(axisScales))
    {
        PyErr_SetString(PyExc_TypeError, "axisScales must be a sequence");
        return -1;
    }

    PyObject* axisOffsets = PyDict_GetItemString(value, "axisOffsets"); // borrowed
    if (!PySequence_Check(axisOffsets))
    {
        PyErr_SetString(PyExc_TypeError, "axisOffsets must be a sequence");
        return -1;
    }

    PyObject* axisDescriptions = PyDict_GetItemString(value, "axisDescriptions"); // borrowed
    if (!PySequence_Check(axisDescriptions))
    {
        PyErr_SetString(PyExc_TypeError, "axisDescriptions must be a sequence");
        return -1;
    }

    PyObject* axisUnits = PyDict_GetItemString(value, "axisUnits"); // borrowed
    if (!PySequence_Check(axisUnits))
    {
        PyErr_SetString(PyExc_TypeError, "axisUnits must be a sequence");
        return -1;
    }

    PyObject* valueUnit = PyDict_GetItemString(value, "valueUnit"); // borrowed
    if (!PyBytes_Check(valueUnit) && !PyUnicode_Check(valueUnit))
    {
        PyErr_SetString(PyExc_TypeError, "valueUnit must be a string");
        return -1;
    }

    PyObject* valueDescription = PyDict_GetItemString(value, "valueDescription"); // borrowed
    if (!PyBytes_Check(valueDescription) && !PyUnicode_Check(valueDescription))
    {
        PyErr_SetString(PyExc_TypeError, "valueDescription must be a string");
        return -1;
    }

    self->dataObject->deleteAllTags();

    if (PyDataObject_setTags(self, tags, closure))
    {
        return -1;
    }

    if (PyDataObject_setAxisScales(self, axisScales, closure))
    {
        return -1;
    }

    if (PyDataObject_setAxisOffsets(self, axisOffsets, closure))
    {
        return -1;
    }

    if (PyDataObject_setAxisDescriptions(self, axisDescriptions, closure))
    {
        return -1;
    }

    if (PyDataObject_setAxisUnits(self, axisUnits, closure))
    {
        return -1;
    }

    if (PyDataObject_setValueUnit(self, valueUnit, closure))
    {
        return -1;
    }

    if (PyDataObject_setValueDescription(self, valueDescription, closure))
    {
        return -1;
    }

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectArray_StructGet_doc,
    "Any : General python-array interface (do not call this directly) \n\
\n\
This interface makes the data object compatible to every array structure in Python \n\
which does equally implement the array interface (e.g. NumPy). This method is \n\
therefore a helper method for the array interface.");
PyObject* PythonDataObject::PyDataObj_Array_StructGet(PyDataObject* self)
{
    PyArrayInterface* inter;
    const ito::DataObject* selfDO = self->dataObject;

    if (selfDO == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is nullptr");
        return nullptr;
    }
    else if (selfDO->getType() == ito::tDateTime || selfDO->getType() == ito::tTimeDelta)
    {
        // it is not allowed to set a Python error here, else the
        // fallback method PyDataObj_Array_ will not be called afterwards.
        // PyErr_SetString(PyExc_NotImplementedError, "__array_struct__ not supported for
        // dataObjects of type dateTime or timeDelta");
        return nullptr;
    }
    else if (selfDO->getContinuous() == false)
    {
        // For Numpy >= 1.18 it seems, that an exception set will
        // change the behaviour. We want, that if this method
        // fails, numpy tries to call __array__(). This is only done
        // for Numpy >= 1.18 if no exception is set here!

        /*PyErr_SetString(
            PyExc_RuntimeError,
            "the dataObject cannot be directly converted into a numpy array since"
            "it is not continuous (call dataObject.makeContinuous() for conversion first)."
        );*/

        return nullptr;
    }

    inter = new PyArrayInterface();

    if (inter == nullptr)
    {
        return PyErr_NoMemory();
    }

    inter->two = 2;
    inter->nd = selfDO->getDims();

    if (inter->nd == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        DELETE_AND_SET_NULL(inter)
        return nullptr;
    }

    RetVal ret = parseTypeNumber(selfDO->getType(), inter->typekind, inter->itemsize);

    if (ret.containsError())
    {
        DELETE_AND_SET_NULL(inter)

        if (ret.hasErrorMessage())
        {
            PythonCommon::transformRetValToPyException(ret, PyExc_TypeError);
            return nullptr;
        }

        PyErr_SetString(
            PyExc_TypeError, "Error converting type of dataObject to corresponding numpy type");
        return nullptr;
    }

#if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
    // NPY_NOTSWAPPED indicates, that both data in opencv and data in numpy
    // should have the same byteorder (Intel: little-endian)
    inter->flags = NPY_WRITEABLE | NPY_ALIGNED | NPY_NOTSWAPPED;
#else
    // NPY_NOTSWAPPED indicates, that both data in opencv and data in
    // numpy should have the same byteorder (Intel: little-endian)
    inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
#endif

    // check if size and osize are totally equal, then set continuous flag
    if (selfDO->getTotal() == selfDO->getOriginalTotal())
    {
#if (NPY_FEATURE_VERSION < 0x00000007)
        inter->flags |= NPY_C_CONTIGUOUS;
#else
        inter->flags |= NPY_ARRAY_C_CONTIGUOUS;
#endif
    }

    inter->descr = nullptr;
    inter->data = nullptr;
    inter->shape = nullptr;
    inter->strides = nullptr;

    if (selfDO->getDims() > 0)
    {
        unsigned int firstMDataIndex = selfDO->seekMat(0);
        inter->data = (void*)((cv::Mat*)selfDO->get_mdata()[firstMDataIndex])->data;

        inter->shape = (npy_intp*)malloc(inter->nd * sizeof(npy_intp));
        inter->strides = (npy_intp*)malloc(inter->nd * sizeof(npy_intp));

        // since transpose flag has been evaluated and is false now, everything is ok here
        inter->shape[inter->nd - 1] = (npy_intp)selfDO->getSize(inter->nd - 1);
        inter->strides[inter->nd - 1] = inter->itemsize;

        for (int i = inter->nd - 2; i >= 0; i--)
        {
            // since transpose flag has been evaluated and is
            // false now, everything is ok here
            inter->shape[i] = (npy_intp)selfDO->getSize(i);

            // since transpose flag has been evaluated and is
            // false now, everything is ok here
            inter->strides[i] = inter->strides[i + 1] * selfDO->getOriginalSize(i + 1);
        }
    }

    // don't increment SELF here, since the receiver of the capsule (e.g. numpy-method) will
    // increment the refcount of the PyDataObject SELF by itself.
    return PyCapsule_New((void*)inter, nullptr, &PyDataObj_Capsule_Destructor);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    dataObjectArray_Interface_doc,
    "dict : General python-array interface (do not call this directly). \n\
\n\
This interface makes the data object compatible to every array structure in python \n\
which does equally implement the array interface (e.g. NumPy).");
PyObject* PythonDataObject::PyDataObj_Array_Interface(PyDataObject* self)
{
    const ito::DataObject* selfDO = self->dataObject;

    if (selfDO == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is nullptr");
        return nullptr;
    }
    else if (selfDO->getType() == ito::tDateTime || selfDO->getType() == ito::tTimeDelta)
    {
        return nullptr;
    }
    else if (selfDO->getContinuous() == false)
    {
        // For Numpy >= 1.18 it seems, that an exception set will
        // change the behaviour. We want, that if this method
        // fails, numpy tries to call __array__(). This is only done
        // for Numpy >= 1.18 if no exception is set here!

        /*PyErr_SetString(
            PyExc_RuntimeError,
            "the dataObject cannot be directly converted into a numpy array since"
            "it is not continuous (call dataObject.makeContinuous() for conversion first)."
        );*/

        return nullptr;
    }

    PyObject* item = nullptr;
    int itemsize;
    char typekind;
    char typekind2[] = "a\0";

    RetVal ret = parseTypeNumber(selfDO->getType(), typekind, itemsize);

    if (ret.containsError())
    {
        if (ret.hasErrorMessage())
        {
            PythonCommon::transformRetValToPyException(ret, PyExc_TypeError);
            return nullptr;
        }

        PyErr_SetString(
            PyExc_TypeError, "Error converting type of dataObject to corresponding numpy type");
        return nullptr;
    }

    PyObject* retDict = PyDict_New();
    item = PyLong_FromLong(3);
    PyDict_SetItemString(retDict, "version", item);
    Py_DECREF(item);

    typekind2[0] = typekind;

    PyObject* typestr = PyUnicode_FromFormat("|%s%d", &typekind2, itemsize);
    PyDict_SetItemString(retDict, "typestr", typestr);
    Py_XDECREF(typestr);

    if (selfDO->getDims() > 0)
    {
        unsigned int firstMDataIndex = selfDO->seekMat(0);
        int dims = selfDO->getDims();
        PyObject* shape = PyTuple_New(dims);
        PyObject* data = PyTuple_New(2);
        PyObject* strides = PyTuple_New(dims);
        npy_intp strides_iPlus1;

        bool isFullyContiguous = true;

        for (int i = 0; i < dims; i++)
        {
            if (selfDO->getSize(i) != selfDO->getOriginalSize(i))
                isFullyContiguous = false;
        }

        PyTuple_SetItem(
            data,
            0,
            PyLong_FromVoidPtr((void*)((cv::Mat*)selfDO->get_mdata()[firstMDataIndex])->data));
        Py_INCREF(Py_False);
        PyTuple_SetItem(data, 1, Py_False);

        PyTuple_SetItem(shape, dims - 1, PyLong_FromLong(selfDO->getSize(dims - 1)));
        strides_iPlus1 = itemsize;
        PyTuple_SetItem(strides, dims - 1, PyLong_FromLong(itemsize));

        for (int i = dims - 2; i >= 0; i--)
        {
            // since transpose flag has been evaluated and is false now, everything is ok here
            PyTuple_SetItem(shape, i, PyLong_FromLong(selfDO->getSize(i)));
            strides_iPlus1 = (strides_iPlus1 * selfDO->getOriginalSize(i + 1));
            PyTuple_SetItem(strides, i, PyLong_FromLong(strides_iPlus1));
        }

        PyDict_SetItemString(retDict, "shape", shape);

        if (!isFullyContiguous)
        {
            PyDict_SetItemString(retDict, "strides", strides);
        }

        PyDict_SetItemString(retDict, "data", data);

        Py_XDECREF(shape);
        Py_XDECREF(data);
        Py_XDECREF(strides);
    }

    // don't icrement SELF here, since the receiver of the capsule (e.g. numpy-method)
    // will increment the refcount of then PyDataObject SELF by itself.
    return retDict;
}


//-------------------------------------------------------------------------------------
PyArrayObject* nparrayFromTimeDeltaDataObject(
    const ito::DataObject* dobj, const PyArray_DatetimeMetaData* meta)
{
    // step 1: create numpy array
    PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_TIMEDELTA);
    auto metaData = (PyArray_DatetimeDTypeMetaData*)(descr->c_metadata);

    if (meta != nullptr)
    {
        metaData->meta.base = meta->base;
        metaData->meta.num = meta->num;
    }
    else
    {
        // guess best datetime meta from dataObject
        PythonDateTime::
            GuessDateTimeMetaFromDataObjectValues<ito::TimeDelta, offsetof(ito::TimeDelta, delta)>(
                dobj, metaData->meta);
    }

    npy_intp* sizes = new npy_intp[dobj->getDims()];
    for (int i = 0; i < dobj->getDims(); ++i)
    {
        sizes[i] = dobj->getSize(i);
    }

    // steals a ref to descr
    PyArrayObject* timeDeltaArray =
        (PyArrayObject*)PyArray_SimpleNewFromDescr(dobj->getDims(), sizes, descr);
    DELETE_AND_SET_NULL_ARRAY(sizes);

    if (!timeDeltaArray)
    {
        return nullptr;
    }

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    NpyIter* iter = NpyIter_New(
        timeDeltaArray,
        NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_CORDER,
        NPY_NO_CASTING,
        nullptr);

    if (iter == nullptr)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        Py_DECREF(timeDeltaArray);
        return nullptr;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, nullptr);

    if (iternext == nullptr)
    {
        NpyIter_Deallocate(iter);

        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        Py_DECREF(timeDeltaArray);
        return nullptr;
    }

    /* The location of the data pointer which the iterator may update */
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    npy_intp* innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    ito::DObjConstIterator it = dobj->constBegin();
    ito::DObjConstIterator itEnd = dobj->constEnd();

    do
    {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--)
        {
            if (!PythonDateTime::ItoTimedelta2npyTimedleta(
                    *((const ito::TimeDelta*)(*it)), *((npy_datetime*)data), metaData->meta))
            {
                // error message set in method above
                Py_DECREF(timeDeltaArray);
                return nullptr;
            }

            it++;
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while (iternext(iter));

    NpyIter_Deallocate(iter);

    return timeDeltaArray;
}

//-------------------------------------------------------------------------------------
PyArrayObject* nparrayFromDateTimeDataObject(
    const ito::DataObject* dobj, const PyArray_DatetimeMetaData* meta)
{
    // step 1: create numpy array
    PyArray_Descr* descr = PyArray_DescrNewFromType(NPY_DATETIME);
    auto metaData = (PyArray_DatetimeDTypeMetaData*)(descr->c_metadata);

    if (meta != nullptr)
    {
        metaData->meta.base = meta->base;
        metaData->meta.num = meta->num;
    }
    else
    {
        // guess best datetime meta from dataObject
        PythonDateTime::
            GuessDateTimeMetaFromDataObjectValues<ito::DateTime, offsetof(ito::DateTime, datetime)>(
                dobj, metaData->meta);
    }

    npy_intp* sizes = new npy_intp[dobj->getDims()];
    for (int i = 0; i < dobj->getDims(); ++i)
    {
        sizes[i] = dobj->getSize(i);
    }

    // steals a ref to descr
    PyArrayObject* dateTimeArray =
        (PyArrayObject*)PyArray_SimpleNewFromDescr(dobj->getDims(), sizes, descr);
    DELETE_AND_SET_NULL_ARRAY(sizes);

    if (!dateTimeArray)
    {
        return nullptr;
    }

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
    NpyIter* iter = NpyIter_New(
        dateTimeArray,
        NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
        NPY_CORDER,
        NPY_NO_CASTING,
        nullptr);

    if (iter == nullptr)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        Py_DECREF(dateTimeArray);
        return nullptr;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, nullptr);

    if (iternext == nullptr)
    {
        NpyIter_Deallocate(iter);

        PyErr_Format(PyExc_RuntimeError, "Failed to iterate over numpy.ndarray.");
        Py_DECREF(dateTimeArray);
        return nullptr;
    }

    /* The location of the data pointer which the iterator may update */
    char** dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    npy_intp* strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    npy_intp* innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    ito::DObjConstIterator it = dobj->constBegin();
    ito::DObjConstIterator itEnd = dobj->constEnd();

    do
    {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--)
        {
            if (!PythonDateTime::ItoDatetime2npyDatetime(
                    *((const ito::DateTime*)(*it)), *((npy_datetime*)data), metaData->meta))
            {
                // error message set in method above
                Py_DECREF(dateTimeArray);
                return nullptr;
            }

            it++;
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while (iternext(iter));

    NpyIter_Deallocate(iter);

    return dateTimeArray;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObject_Array__doc, "__array__(dtype = None) -> np.ndarray \n\
\n\
Returns a numpy.ndarray from this dataObject. If possible a shallow copy is returned. \n\
\n\
If no ``dtype`` is given and if the this :class:`dataObject` is continuous, \n\
a :class:`numpy.ndarray` that shares its memory with this dataObject is returned. \n\
If the desired ``dtype`` does not fit to the type of this :class:`dataObject`, \n\
a casted deep copy is returned, this is also always the case for the data types \n\
dateTime and timeDelta, as well as for non-continuous dataObjects. \n\
Then a continuous dataObject is created that is the base object of \n\
the returned :class:`numpy.ndarray`. \n\
\n\
Parameters \n\
---------- \n\
dtype : numpy.dtype, optional \n\
    A :class:`numpy.dtype` object that describes the data type, data alignment etc. \n\
    for the returned :class:`numpy.ndarray`. \n\
\n\
Returns \n\
------- \n\
arr : numpy.ndarray \n\
    The converted :class:`numpy.ndarray`");
PyObject* PythonDataObject::PyDataObj_Array_(PyDataObject* self, PyObject* args)
{
    if (self->dataObject == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is nullptr");
        return nullptr;
    }

    PyArray_Descr* newtype = nullptr;
    PyArrayObject* newArray = nullptr;

    if (!PyArg_ParseTuple(args, "|O&", PyArray_DescrConverter, &newtype))
    {
        Py_XDECREF(newtype);
        return nullptr;
    }

    PyObject* item = nullptr;
    ito::DataObject* selfDO = self->dataObject;

    if (selfDO->getType() == ito::tDateTime)
    {
        const PyArray_DatetimeMetaData* meta = nullptr;

        if (newtype && PyDataType_ISDATETIME(newtype))
        {
            meta = &(((PyArray_DatetimeDTypeMetaData*)newtype->c_metadata)->meta);
        }

        newArray = nparrayFromDateTimeDataObject(selfDO, meta);
    }
    else if (selfDO->getType() == ito::tTimeDelta)
    {
        const PyArray_DatetimeMetaData* meta = nullptr;

        if (newtype && PyDataType_ISDATETIME(newtype))
        {
            meta = &(((PyArray_DatetimeDTypeMetaData*)newtype->c_metadata)->meta);
        }

        newArray = nparrayFromTimeDeltaDataObject(selfDO, meta);
    }
    else if (selfDO->getContinuous())
    {
        newArray = (PyArrayObject*)PyArray_FromStructInterface((PyObject*)self);
    }
    else
    {
        // at first try to make continuous copy of data object and handle possible exceptions before
        // going on
        ito::DataObject* continuousObject = nullptr;

        try
        {
            continuousObject = new ito::DataObject(ito::makeContinuous(*selfDO));
        }
        catch (cv::Exception& exc)
        {
            continuousObject = NULL;
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        PyDataObject* newDO = PythonDataObject::createEmptyPyDataObject();
        newDO->dataObject = continuousObject;

        PyDataObject_SetBase(newDO, self->base);

        newArray = (PyArrayObject*)PyArray_FromStructInterface((PyObject*)newDO);
        Py_DECREF(newDO);
    }

    if (newArray == nullptr)
    {
        return nullptr;
    }

    if ((newtype == nullptr) || PyArray_EquivTypes(PyArray_DESCR(newArray) /*->descr*/, newtype))
    {
        return (PyObject*)newArray;
    }
    else
    {
        PyObject* ret = PyArray_CastToType(newArray, newtype, 0);
        Py_DECREF(newArray);
        return ret;
    }
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_Reduce(PyDataObject* self, PyObject* /*args*/)
{
    // version history:
    // 21120:
    //  - each plane is stored as a bytearray in the data tuple (this needs 16bit for values bigger
    //  than 100 since it is transformed to an unicode value)
    //
    // 21121:
    //  - each plane is now stored as a byte object, this can natively be pickled (faster, bytearray
    //  contains a reduce method)

    long version = 21121;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_NotImplementedError, "data object is NULL");
        return NULL;
    }

    int dims = self->dataObject->getDims();

    PyObject* sizeList = PyList_New(dims);
    for (int i = 0; i < dims; i++)
    {
        // since transpose flag has been evaluated and
        // is false now, everything is ok here
        PyList_SetItem(sizeList, i, Py_BuildValue("I", self->dataObject->getSize(i)));
    }

    // 1. elem -> callable object
    // 2. elem -> arguments for init-method
    // 3. elem -> state tuple (tuple1, tuple2,tuple3), tuple1 = (bool)transposed), tuple2 = size of
    // dataObject->calcNumMats(), tuple3 = tuple with tags..., each element is of type ByteArray
    int vectorLength = self->dataObject->calcNumMats();

    PyObject* dataTuple = PyTuple_New(vectorLength);
    PyObject* byteArray = NULL;
    cv::Mat* tempMat;
    unsigned int seekNr;
    Py_ssize_t sizeU = 0;
    Py_ssize_t sizeV = 0;
    Py_ssize_t elemSize = 0;
    char* dummy = 0;
    char* startingPoint = NULL;
    // int res;


    if (dims == 1)
    {
        sizeU = 1;
        sizeV = (Py_ssize_t)self->dataObject->getSize(dims - 1);
    }
    else if (dims > 1)
    {
        sizeU = (Py_ssize_t)self->dataObject->getSize(dims - 2);
        sizeV = (Py_ssize_t)self->dataObject->getSize(dims - 1);
    }

    if (version == 21120)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (Py_ssize_t)tempMat->elemSize();

            // in version (checksum) 21120 the data has been stored as bytearray, which is reduced
            // to a unicode and needs a lot of space
            byteArray = PyByteArray_FromStringAndSize(dummy, 0);
            if (PyByteArray_Resize(byteArray, sizeV * sizeU * elemSize) != 0)
            {
                // err, message already set
                Py_XDECREF(byteArray);
                Py_XDECREF(dataTuple);
                Py_XDECREF(sizeList);
                return NULL;
            }

            startingPoint = PyByteArray_AsString(byteArray);

            for (Py_ssize_t row = 0; row < sizeU; row++)
            {
                if (memcpy((void*)startingPoint, (void*)(tempMat->ptr(row)), sizeV * elemSize) ==
                    NULL)
                {
                    Py_XDECREF(byteArray);
                    Py_XDECREF(dataTuple);
                    Py_XDECREF(sizeList);
                    PyErr_Format(
                        PyExc_NotImplementedError,
                        "memcpy failed. (index m_data-vector: %d, row-index: %d)",
                        i,
                        row);
                    return NULL;
                }
                startingPoint +=
                    (sizeV * elemSize); // move startingPoint by length (in byte) of one image row
            }

            PyTuple_SetItem(dataTuple, i, byteArray); // steals ref from byteArray
            byteArray = NULL;
        }
    }
    else if (version = 21121)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (int)tempMat->elemSize();

            // in version (checksum) 21120 the data has been stored as bytearray, which is reduced
            // to a unicode and needs a lot of space
            byteArray = PyBytes_FromStringAndSize(NULL, sizeV * sizeU * elemSize);
            if (!byteArray /* || _PyBytes_Resize(&byteArray, sizeV * sizeU * elemSize) != 0 */)
            {
                // err, message already set
                Py_XDECREF(byteArray);
                Py_XDECREF(dataTuple);
                Py_XDECREF(sizeList);
                return NULL;
            }

            startingPoint = PyBytes_AS_STRING(byteArray);

            for (Py_ssize_t row = 0; row < sizeU; row++)
            {
                if (memcpy((void*)startingPoint, (void*)(tempMat->ptr(row)), sizeV * elemSize) ==
                    NULL)
                {
                    Py_XDECREF(byteArray);
                    Py_XDECREF(dataTuple);
                    Py_XDECREF(sizeList);
                    PyErr_Format(
                        PyExc_NotImplementedError,
                        "memcpy failed. (index m_data-vector: %d, row-index: %d)",
                        i,
                        row);
                    return NULL;
                }
                startingPoint +=
                    (sizeV * elemSize); // move startingPoint by length (in byte) of one image row
            }

            PyTuple_SetItem(dataTuple, i, byteArray); // steals ref from byteArray
            byteArray = NULL;
        }
    }


    // load tags
    PyObject* tagTuple = PyTuple_New(10);
    PyTuple_SetItem(tagTuple, 0, PyLong_FromLong(version));

    PyObject* newTagDict = PyDataObject_getTagDict(self, NULL); // new ref
    PyObject* tempItem;
    PyObject* item = NULL;

    if (!PyErr_Occurred())
    {
        DataObject* dObj = self->dataObject;
        int tagSize = dObj->getTagListSize();
        // std::string tempString;
        DataObjectTagType tempTagValue;
        std::string tempKey;
        bool validOp;
        PyObject* tempTag;

        // 1. tags (here it is bad to use the tags-getter, since this returns a dict_proxy, which
        // cannot directly be pickled
        tempTag = PyDict_New();
        for (int i = 0; i < tagSize; i++)
        {
            tempKey = dObj->getTagKey(i, validOp);
            if (validOp)
            {
                // tempString = dObj->getTag(tempKey, validOp);
                // if (validOp) PyDict_SetItem(tempTag, PyUnicode_FromString(tempKey.data()),
                // PyUnicode_FromString(tempString.data()));
                dObj->getTagByIndex(i, tempKey, tempTagValue);
                if (tempTagValue.getType() == DataObjectTagType::typeDouble)
                {
                    item = PyFloat_FromDouble(tempTagValue.getVal_ToDouble());
                    PyDict_SetItemString(tempTag, tempKey.data(), item);
                    Py_DECREF(item);
                }
                else
                {
                    item = PythonQtConversion::QByteArrayToPyUnicodeSecure(
                        tempTagValue.getVal_ToString().data());
                    PyDict_SetItemString(tempTag, tempKey.data(), item);
                    Py_DECREF(item);
                }
            }
        }
        PyTuple_SetItem(tagTuple, 1, tempTag); // steals ref from tempTag

        // 2. axisScales
        tempItem = PyDict_GetItemString(newTagDict, "axisScales"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 2, tempItem); // steals ref from tempItem

        // 3. axisOffsets
        tempItem = PyDict_GetItemString(newTagDict, "axisOffsets"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 3, tempItem); // steals ref from tempItem

        // 4. axisDescriptions
        tempItem = PyDict_GetItemString(newTagDict, "axisDescriptions"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 4, tempItem); // steals ref from tempItem

        // 5. axisUnits
        tempItem = PyDict_GetItemString(newTagDict, "axisUnits"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 5, tempItem); // steals ref from tempItem

        // 6. valueUnit
        tempItem = PyDict_GetItemString(newTagDict, "valueUnit"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 6, tempItem); // steals ref from tempItem

        // 7. valueDescription
        tempItem = PyDict_GetItemString(newTagDict, "valueDescription"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 7, tempItem); // steals ref from tempItem

        // 8.
        tempItem = PyDict_GetItemString(newTagDict, "valueOffset"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 8, tempItem); // steals ref from tempItem

        // 9.
        tempItem = PyDict_GetItemString(newTagDict, "valueScale"); // borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple, 9, tempItem); // steals ref from tempItem
    }

    Py_XDECREF(newTagDict);

    PyObject* stateTuple =
        Py_BuildValue("(bOO)", false /*self->dataObject->isT()*/, dataTuple, tagTuple);

    Py_DECREF(dataTuple);
    Py_DECREF(tagTuple);

    PyObject* tempOut = Py_BuildValue(
        "(O(Osb)O)",
        Py_TYPE(self),
        sizeList,
        typeNumberToName(self->dataObject->getType()),
        self->dataObject->getContinuous(),
        stateTuple);

    Py_DECREF(sizeList);
    Py_DECREF(stateTuple);

    return tempOut;

    // PyErr_SetString(PyExc_NotImplementedError, "pickling for dataObject not possible");
    // return NULL;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_SetState(PyDataObject* self, PyObject* args)
{
    // version history:
    // see log in PyDataObj_Reduce

    bool transpose = false;
    PyObject* dataTuple = nullptr; // borrowed reference
    PyObject* tagTuple = nullptr; // borrowed reference
    PyObject* tempTag = nullptr; // borrowed reference
    long version = 21120; // this is the first version, current is 21121

    if (!PyArg_ParseTuple(
            args, "(bO!O!)", &transpose, &PyTuple_Type, &dataTuple, &PyTuple_Type, &tagTuple))
    {
        PyErr_Clear();
        // test if maybe no tagTuple is available
        tagTuple = nullptr;
        if (!PyArg_ParseTuple(args, "(bO!)", &transpose, &PyTuple_Type, &dataTuple))
        {
            PyErr_SetString(
                PyExc_NotImplementedError,
                "unpickling for dataObject not possible since state vector is invalid");
            return nullptr;
        }
    }

    // pre-check tags
    if (tagTuple != NULL)
    {
        if (PyTuple_Size(tagTuple) != 10)
        {
            PyErr_SetString(
                PyExc_NotImplementedError,
                "tags in pickled data object does not have the required number of elements (10)");
            return nullptr;
        }
        else
        {
            tempTag = PyTuple_GetItem(tagTuple, 0); // borrowed ref

            if (!PyLong_Check(tempTag))
            {
                PyErr_SetString(
                    PyExc_NotImplementedError,
                    "first element in tag tuple must be an integer number, which it is not.");
                return nullptr;
            }

            version = PyLong_AsLong(tempTag);

            if (version != 21120 && version != 21121)
            {
                PyErr_SetString(
                    PyExc_NotImplementedError,
                    "This version of itom does not support the version number of this pickled "
                    "dataset. Consider to update itom.");
                return nullptr;
            }
        }
    }

    if (transpose == true)
    {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "transpose flag of unpickled data must be false (since the transposition has been "
            "evaluated before pickling). Transpose flag is obsolete now.");
        return nullptr;
    }

    if (self->dataObject == nullptr)
    {
        PyErr_SetString(PyExc_NotImplementedError, "unpickling for dataObject failed");
        return nullptr;
    }

    int vectorLength = self->dataObject->calcNumMats();

    if (PyTuple_Size(dataTuple) != vectorLength)
    {
        PyErr_SetString(
            PyExc_NotImplementedError,
            "unpickling for dataObject failed since data dimensions does not fit");
        return nullptr;
    }

    int dims = self->dataObject->getDims();
    PyObject* byteArray = nullptr;
    cv::Mat* tempMat;
    unsigned int seekNr;
    Py_ssize_t sizeU = 0;
    Py_ssize_t sizeV = 0;
    uchar* startPtr = nullptr;
    char* byteArrayContent = nullptr;
    Py_ssize_t elemSize = 0;
    std::string tempString;
    std::string keyString;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    PyObject* seqItem = nullptr;
    bool stringOk;

    if (dims == 1)
    {
        sizeU = 1;
        sizeV = (Py_ssize_t)self->dataObject->getSize(dims - 1);
    }
    else if (dims > 1)
    {
        sizeU = (Py_ssize_t)self->dataObject->getSize(dims - 2);
        sizeV = (Py_ssize_t)self->dataObject->getSize(dims - 1);
    }

    if (version == 21120)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (int)tempMat->elemSize();
            startPtr = tempMat->ptr(0); // mat is continuous!!! (should be;))
            byteArray = PyTuple_GetItem(dataTuple, i); // borrowed ref

            byteArrayContent = PyByteArray_AsString(byteArray); // borrowed ref
            memcpy((void*)startPtr, (void*)byteArrayContent, sizeU * sizeV * elemSize);
        }
    }
    else if (version == 21121)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (int)tempMat->elemSize();
            startPtr = tempMat->ptr(0); // mat is continuous!!! (should be;))
            byteArray = PyTuple_GetItem(dataTuple, i); // borrowed ref

            byteArrayContent = PyBytes_AsString(byteArray); // borrowed ref
            memcpy((void*)startPtr, (void*)byteArrayContent, sizeU * sizeV * elemSize);
        }
    }

    // transpose must be false (checked above)

    // check tags
    if (tagTuple != NULL && PyTuple_Size(tagTuple) == 10)
    {
        // 1. tags
        tempTag = PyTuple_GetItem(tagTuple, 1); // borrowed
        if (PyDict_Check(tempTag))
        {
            while (PyDict_Next(tempTag, &pos, &key, &value))
            {
                keyString = PythonQtConversion::PyObjGetStdStringAsLatin1(key, false, stringOk);
                if (stringOk)
                {
                    if (PyFloat_Check(value) || PyLong_Check(value))
                    {
                        self->dataObject->setTag(keyString, PyFloat_AsDouble(value));
                    }
                    else
                    {
                        tempString =
                            PythonQtConversion::PyObjGetStdStringAsLatin1(value, false, stringOk);
                        if (stringOk)
                        {
                            self->dataObject->setTag(keyString, tempString);
                        }
                    }
                }
            }
        }

        // 2. axisScales
        tempTag = PyTuple_GetItem(tagTuple, 2);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i = 0; i < PySequence_Size(tempTag); i++)
            {
                seqItem = PySequence_GetItem(tempTag, i); // new reference
                self->dataObject->setAxisScale(i, PyFloat_AsDouble(seqItem));
                Py_XDECREF(seqItem);
            }
        }

        // 3. axisOffsets
        tempTag = PyTuple_GetItem(tagTuple, 3);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i = 0; i < PySequence_Size(tempTag); i++)
            {
                seqItem = PySequence_GetItem(tempTag, i); // new reference
                self->dataObject->setAxisOffset(i, PyFloat_AsDouble(seqItem));
                Py_XDECREF(seqItem);
            }
        }

        // 4. axisDescriptions
        tempTag = PyTuple_GetItem(tagTuple, 4);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i = 0; i < PySequence_Size(tempTag); i++)
            {
                seqItem = PySequence_GetItem(tempTag, i); // new reference
                tempString =
                    PythonQtConversion::PyObjGetStdStringAsLatin1(seqItem, false, stringOk);
                if (stringOk)
                {
                    self->dataObject->setAxisDescription(i, tempString);
                }
                Py_XDECREF(seqItem);
            }
        }

        // 5. axisUnits
        tempTag = PyTuple_GetItem(tagTuple, 5);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i = 0; i < PySequence_Size(tempTag); i++)
            {
                seqItem = PySequence_GetItem(tempTag, i); // new reference
                tempString =
                    PythonQtConversion::PyObjGetStdStringAsLatin1(seqItem, false, stringOk);
                if (stringOk)
                {
                    self->dataObject->setAxisUnit(i, tempString);
                }
                Py_XDECREF(seqItem);
            }
        }

        // 6. valueUnit
        tempTag = PyTuple_GetItem(tagTuple, 6); // borrowed
        tempString = PythonQtConversion::PyObjGetStdStringAsLatin1(tempTag, false, stringOk);
        if (stringOk)
        {
            self->dataObject->setValueUnit(tempString);
        }

        // 7. valueDescription
        tempTag = PyTuple_GetItem(tagTuple, 7); // borrowed
        tempString = PythonQtConversion::PyObjGetStdStringAsLatin1(tempTag, false, stringOk);
        if (stringOk)
        {
            self->dataObject->setValueDescription(tempString);
        }

        // 8.
        // tempTag = PyTuple_GetItem(tagTuple,8);
        // 9.
        // tempTag = PyTuple_GetItem(tagTuple,9);
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObj_ToGray_doc, "toGray(destinationType = \"uint8\") -> dataObject \n\
\n\
Converts this ``rgba32`` coloured dataObject into a gray-scale dataObject. \n\
\n\
The returned :class:`dataObject` has the same size than this :class:`dataObject` \n\
and the real-value data type, that is given by ``destinationType``. The pixel-wise \n\
conversion is done using the formula: \n\
\n\
.. math: gray = 0.299 * red + 0.587 * green + 0.114 * blue.\n\
\n\
Parameters \n\
---------- \n\
destinationType : {\"uint8\", \"int8\", \"uint16\", \"int16\", \"int32\", \"float32\", \"float64\"}, optional \n\
    Desired data type of the returned dataObject (only real value data types allowed). \n\
\n\
Returns \n\
------- \n\
gray : dataObject \n\
    converted gray-scale data object of desired type. \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this dataObject is no ``rgba32`` object or if the ``destinationType`` is invalid.");
/*static*/ PyObject* PythonDataObject::PyDataObj_ToGray(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    const char* type = NULL;
    int typeno = ito::tUInt8;

    const char* kwlist[] = {"destinationType", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", const_cast<char**>(kwlist), &type))
    {
        return NULL;
    }

    if (type)
    {
        typeno = dObjTypeFromName(type);
    }

    if (typeno == -1)
    {
        PyErr_Format(PyExc_TypeError, "The given type string '%s' is unknown", type);
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->toGray(typeno));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Extracted gray-Value from RGBA32-type dataObject.");

    return (PyObject*)retObj;
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObj_SplitColor_doc, "splitColor(color, destinationType = \"uint8\") -> dataObject \n\
\n\
Splits selected color channels from this coloured ``rgba32`` dataObject. \n\
\n\
A ``rgba32`` coloured :class:`dataObject` contains color values for each item. \n\
Each color value contains a red, green, blue and alpha (transparancy) component (uint8 \n\
each). This method allows extracting one or several of these components from this \n\
dataObject. These components are then returned in single slices of a new, first axis \n\
of the returned dataObject. \n\
\n\
The returned :class:`dataObject` has one axis more than this object. This new axis \n\
is prepended to the existing axes, that have the same shape than this object. The data \n\
type of the returned object is ``destinationType``. \n\
\n\
The size of the first, new axis is equal to the number of letters in ``color``. \n\
Each letter must be one of the characters ``b``, ``r``, ``g`` or ``a``, that stand \n\
for the available channels of the color, that can be extracted. \n\
\n\
Example: :: \n\
    \n\
    color = dataObject.zeros([20, 10], 'rgba32') \n\
    split_colors = color.splitColor(\"rgb\") \n\
    print(split_colors.shape, split_colors.dtype) \n\
    # printout: [3, 20, 10], \"uint8\" \n\
\n\
In this example, the :attr:`shape` of ``split_colors`` is ``[3, 20, 10]``, since \n\
three channels (red, green and blue) should have been splitted, such that \n\
``split_colors[0, :, :]`` contains the red component, etc. \n\
\n\
Parameters \n\
---------- \n\
color : str \n\
    Desired color string, that indicates the type and order of extracted color \n\
    components. This string can consist of the following letters: ``('b', 'r', 'g', 'a')``. \n\
    It is possible to combine different channels, like ``\"arg\"`` which extracts the \n\
    alpha channel, followed by red and gree. \n\
destinationType : {\"uint8\", \"int8\", \"uint16\", \"int16\", \"int32\", \"float32\", \"float64\"}, optional \n\
    Desired data type of the returned dataObject (only real value data types allowed). \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    containing the selected channel values \n\
\n\
Raises \n\
------ \n\
TypeError \n\
    if this :class:`dataObject` is no ``rgba32`` object or if ``destinationType`` \n\
    is no real data type.");
/*static*/ PyObject* PythonDataObject::PyDataObj_SplitColor(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    int typeno = ito::tUInt8;
    const char* type = NULL;
    const char* color = NULL;
    const char* kwlist[] = {"color", "destinationType", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|s", const_cast<char**>(kwlist), &color, &type))
    {
        return NULL;
    }

    if (type)
    {
        typeno = dObjTypeFromName(type);
    }

    if (typeno == -1)
    {
        PyErr_Format(PyExc_TypeError, "The given type string '%s' is unknown", type);
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->splitColor(color, typeno));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    if (retObj)
        retObj->dataObject->addToProtocol("Extracted color data from RGBA32-type dataObject.");

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObj_ToNumpyColor_doc, "toNumpyColor(addAlphaChannel = 0) -> np.ndarray \n\
\n\
Converts a 2D dataObject of type ``rgba32`` to a 3D numpy.ndarray of type ``uint8``. \n\
\n\
Many Python packages, e.g. OpenCV (cv2) or PIL store coloured array such that the color \n\
components are stored in an additional axis, which is the last axis of all axes. \n\
Hence, there is no specific ``rgba2`` data type for :class:`numpy.ndarray`, like it \n\
is the case for :class:`dataObject`. \n\
\n\
This method converts a coloured :class:`dataObject` of dtype ``rgba32`` to a compatible \n\
:class:`numpy.ndarray`, where the color components are stored in an additional last axis. \n\
The size of this last axis is either ``3`` if ``addAlphaChannel = 0`` or ``4`` otherwise. \n\
The order of this last axis is ``blue``, ``green``, ``red`` and optional ``alpha``. \n\
The remaining first axes of the returned object have the same shape than this dataObject. \n\
\n\
Parameters \n\
---------- \n\
addAlphaChannel : int, optional \n\
    If ``0``, the last dimension of the returned :class:`numpy.ndarray` has a size of ``3`` \n\
    and contains the blue, green and red value, whereas ``1`` adds the alpha value as \n\
    fourth value. \n\
\n\
Returns \n\
------- \n\
arr : numpy.ndarray \n\
    The 3D :class:`numpy.ndarray` of dtype ``uint8``. The shape is ``[*obj.shape, 3]`` or \n\
    ``[*obj.shape, 4]``, depending on ``addAlphaChannel``, where ``obj`` is this \n\
    :class:`dataObject`.");
PyObject* PythonDataObject::PyDataObj_ToNumpyColor(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    int addAlphaChannel = 0;

    const char* kwlist[] = {"addAlphaChannel", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|i", const_cast<char**>(kwlist), &addAlphaChannel))
    {
        return NULL;
    }

    if (self->dataObject == NULL)
    {
        return PyErr_Format(PyExc_TypeError, "This dataObject is empty");
    }
    else if (self->dataObject->getType() != ito::tRGBA32)
    {
        return PyErr_Format(
            PyExc_TypeError,
            "This dataObject must have the type 'rgba32' to be converted to a coloured "
            "numpy.array.");
    }
    else if (self->dataObject->getNumPlanes() != 1)
    {
        return PyErr_Format(PyExc_TypeError, "This dataObject must be two-dimensional.");
    }

    int dims = self->dataObject->getDims();
    npy_intp sizes[] = {
        self->dataObject->getSize(dims - 2),
        self->dataObject->getSize(dims - 1),
        addAlphaChannel ? 4 : 3};
    PyObject* npArray = PyArray_EMPTY(3, sizes, NPY_UBYTE, 0);

    if (npArray)
    {
        npy_intp* npsizes = PyArray_DIMS((PyArrayObject*)npArray);
        npy_intp* npsteps = (npy_intp*)PyArray_STRIDES(
            (PyArrayObject*)npArray); // number of bytes to jump from one element in one dimension
                                      // to the next one
        uchar* data = (uchar*)PyArray_DATA((PyArrayObject*)npArray);

        const ito::Rgba32* srcRow;
        uchar* destRow;
        const cv::Mat* src = self->dataObject->getCvPlaneMat(0);

        for (int r = 0; r < sizes[0]; ++r)
        {
            srcRow = src->ptr<ito::Rgba32>(r);
            destRow = data + (r * npsteps[0]);

            for (int c = 0; c < sizes[1]; ++c)
            {
                destRow[0] = srcRow[c].b;
                destRow[npsteps[2]] = srcRow[c].g;
                destRow[2 * npsteps[2]] = srcRow[c].r;

                if (addAlphaChannel)
                {
                    destRow[3 * npsteps[2]] = srcRow[c].a;
                }

                destRow += npsteps[1];
            }
        }
    }

    return npArray;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectToList_doc, "tolist() -> list \n\
\n\
Returns a nested list with all values of this dataObject. \n\
\n\
An empty :class:`dataObject` with zero dimensions will return an empty list. \n\
Else, the depth of the nested list corresponds to the number of dimensions \n\
of this :class:`dataObject`. The innermost level corresponds to one ``row`` \n\
of this dataObject, or in general, to one set of values along the last \n\
axis of this object. This innermost list contains all these values. \n\
\n\
Returns \n\
------- \n\
list \n\
    Nested list with values of data object. The data types depend on the ``dtype`` \n\
    of this dataObject and can be :obj:`int`, :obj:`float`, :obj:`complex` or \n\
    :class:`rgba`.");
PyObject* PythonDataObject::PyDataObj_ToList(PyDataObject* self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject* d = self->dataObject;

    PyObject* result = NULL;

    unsigned int* iter = new unsigned int[d->getDims()];

    for (int i = 0; i < d->getDims(); i++)
    {
        iter[i] = 0;
    }

    result = PyDataObj_ToListRecursive(d, iter, 0);

    delete[] iter;

    return result;
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_ToListRecursive(
    ito::DataObject* dataObj, unsigned int* currentIdx, int iterationIndex)
{
    if (dataObj == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    PyObject* temp = NULL;

    if (iterationIndex == dataObj->getDims() - 1) // last index
    {
        int len = dataObj->getSize(iterationIndex);
        PyObject* result = PyList_New(len);

        for (int i = 0; i < len; i++)
        {
            currentIdx[iterationIndex] = i;
            temp = PyDataObj_At(dataObj, currentIdx);

            if (temp == NULL)
            {
                Py_DECREF(result);
                return NULL;
            }

            PyList_SetItem(result, i, temp); // steals a ref
        }
        return result;
    }
    else if (iterationIndex < dataObj->getDims() - 1) // previous indexes (besides last one)
    {
        int len = dataObj->getSize(iterationIndex);
        PyObject* result = PyList_New(len);

        for (int i = 0; i < len; i++)
        {
            currentIdx[iterationIndex] = i;
            temp = PyDataObj_ToListRecursive(dataObj, currentIdx, iterationIndex + 1);

            if (temp == NULL)
            {
                Py_DECREF(result);
                return NULL;
            }

            PyList_SetItem(result, i, temp); // steals a ref
        }

        return result;
    }
    else
    {
        return PyList_New(0);
    }
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_At(ito::DataObject* dataObj, const unsigned int* idx)
{
    if (dataObj == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return nullptr;
    }

    switch (dataObj->getType())
    {
    case ito::tUInt8:
        return PyLong_FromUnsignedLong(dataObj->at<uint8>(idx));
    case ito::tInt8:
        return PyLong_FromLong(dataObj->at<int8>(idx));
    case ito::tUInt16:
        return PyLong_FromUnsignedLong(dataObj->at<uint16>(idx));
    case ito::tInt16:
        return PyLong_FromLong(dataObj->at<int16>(idx));
    case ito::tUInt32:
        return PyLong_FromUnsignedLong(dataObj->at<uint32>(idx));
    case ito::tInt32:
        return PyLong_FromLong(dataObj->at<int32>(idx));
    case ito::tRGBA32: {
        ito::PythonRgba::PyRgba* color = ito::PythonRgba::createEmptyPyRgba();
        if (color)
        {
            color->rgba = dataObj->at<ito::Rgba32>(idx).rgba;
        }
        return (PyObject*)color;
    }
    case ito::tFloat32:
        return PyFloat_FromDouble(dataObj->at<float32>(idx));
    case ito::tFloat64:
        return PyFloat_FromDouble(dataObj->at<float64>(idx));
    case ito::tComplex64: {
        const ito::complex64 value = dataObj->at<complex64>(idx);
        return PyComplex_FromDoubles(value.real(), value.imag());
    }
    case ito::tComplex128: {
        const ito::complex128 value = dataObj->at<complex128>(idx);
        return PyComplex_FromDoubles(value.real(), value.imag());
    }
    case ito::tTimeDelta: {
        const auto value = dataObj->at<TimeDelta>(idx);
        PyObject* d = PythonDateTime::GetPyTimeDelta(value);
        return d;
    }
    case ito::tDateTime: {
        const auto value = dataObj->at<DateTime>(idx);
        PyObject* d = PythonDateTime::GetPyDateTime(value);
        return d;
    }
    default:
        PyErr_SetString(PyExc_TypeError, "type of data object not supported");
        return nullptr;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectCreateMask_doc, "createMask(shapes, inverse = False) -> dataObject \n\
\n\
Returns an ``uint8`` mask dataObject where all pixels of this object that are contained in any shape are masked. \n\
\n\
The returned :class:`dataObject` has the same shape than this object and the data type \n\
``uint8``. All pixels in this object, that are contained in any of the given :class:`shape` \n\
will be set to ``255`` in the returned array, otherwise ``0``. \n\
\n\
*New in itom 5.0*: always return a 2d dataObject \n\
(see return value below)\n\
\n\
Parameters \n\
---------- \n\
shapes : shape or list of shape or tuple of shape \n\
    The union of all given shapes (polygons, rectangles, squares, circles and ellipes \n\
    are considered, only) is used to determine if any pixel should be masked in the \n\
    returned mask (value ``255``) or not. \n\
inverse : bool \n\
    If ``True``, masked values are set to ``0`` (instead of ``255``) and all other \n\
    values are set to ``255`` (instead of ``0``). The default is ``False`` (masked = ``255``). \n\
\n\
Returns \n\
------- \n\
mask : dataObject \n\
    uint8 :class:`dataObject` mask (0: not contained, else: contained) whose shape is equal \n\
    to the last two dimensions of this object. The tags :attr:`axisScales`, \n\
    :attr:`axisOffsets`, :attr:`axisDescriptions` and :attr:`axisUnits` are accordingly copied from the \n\
    last two axes of this object.");
PyObject* PythonDataObject::PyDataObject_createMask(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    PyObject* shapes = NULL;
    int inverse = 0;

    const char* kwlist[] = {"shapes", "inverse", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|i", const_cast<char**>(kwlist), &shapes, &inverse))
    {
        return NULL;
    }

    if (PyShape_Check(shapes))
    {
        PythonShape::PyShape* shape = (PythonShape::PyShape*)shapes;
        if (shape && shape->shape)
        {
            // ito::DataObject mask = shape->shape->mask(*self->dataObject, inverse > 0);
            ito::DataObject mask =
                ShapeDObject::mask(*self->dataObject, *shape->shape, inverse > 0);
            PyDataObject* ret = createEmptyPyDataObject();
            ret->dataObject = new ito::DataObject(mask);
            return (PyObject*)ret;
        }

        PyErr_SetString(PyExc_TypeError, "at least one shape item is invalid.");
        return NULL;
    }
    else if (PySequence_Check(shapes))
    {
        QVector<ito::Shape> shape_vector;
        PyObject* obj;
        PythonShape::PyShape* shape;

        PyObject* shapeseq = PySequence_Fast(shapes, "shape is no sequence.");
        if (!shapeseq)
        {
            return NULL;
        }

        for (Py_ssize_t i = 0; i < PySequence_Length(shapeseq); ++i)
        {
            obj = PySequence_Fast_GET_ITEM(shapeseq, i); // borrowed

            if (PyShape_Check(obj))
            {
                shape = (PythonShape::PyShape*)obj;

                if (shape && shape->shape)
                {
                    shape_vector << *shape->shape;
                }
                else
                {
                    Py_DECREF(shapeseq);
                    PyErr_SetString(PyExc_TypeError, "at least one shape item is invalid.");
                    return NULL;
                }
            }
            else
            {
                Py_DECREF(shapeseq);
                PyErr_SetString(
                    PyExc_TypeError,
                    "at least one item of parameter 'shape' is no type itom.shape.");
                return NULL;
            }
        }

        Py_DECREF(shapeseq);

        // ito::DataObject mask = ito::Shape::maskFromMultipleShapes(*self->dataObject,
        // shape_vector, inverse > 0);
        ito::DataObject mask =
            ito::ShapeDObject::maskFromMultipleShapes(*self->dataObject, shape_vector, inverse > 0);

        PyDataObject* ret = createEmptyPyDataObject();
        ret->dataObject = new ito::DataObject(mask);
        return (PyObject*)ret;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "shape required.");
        return NULL;
    }
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectDstack_doc, "dstack(objects, copyAxisInfo = False) -> dataObject \n\
\n\
Returns a 3D dataObject with the stacked dataObjects in the objects sequence. \n\
\n\
The given dataObjects must all have the same type as well as the same size of both \n\
last axes / dimensions. This method then returns a 3d :class:`dataObject` of the same \n\
type, whose size of the two last axes correspond to those of the input ``objects``. \n\
The returned 3D :class:`dataObject` contains then a stacked representation of all \n\
given input dataObjects depth wise (along first axis). \n\
\n\
If any of the input dataObjects has more than two dimensions, all contained planes \n\
(x,y-matrices) are also stacked in the resulting object.\n\
\n\
Parameters \n\
---------- \n\
objects : list of dataObject or tuple of dataObject \n\
    Sequence (list) of dataObjects containing planes that will be stacked together. \n\
    All dataObjects must be of the same type and have the same shape of planes \n\
    (last two dimensions).\n\
copyAxisInfo : bool \n\
    If ``True``, the axis information (description, unit, scale, offset) is copied \n\
    from the first ``dataObject`` in ``objects`` to the returned object. If the first \n\
    dataObject has less dimensions than the returned 3D object, only the axis information \n\
    of the last two dimensions is copied. The description and unit of the first axis (``z``) \n\
    is an empty string (default) and the scale and offset of this first axis is set to \n\
    the default values ``1.0`` and ``0.0`` respectively. \n\
\n\
Returns \n\
------- \n\
stack : dataObject \n\
    If ``objects`` is an empty list or tuple, an empty :class:`dataObject` is returned. \n\
    Else if ``objects`` only contains one array, this array is returned. Otherwise, \n\
    all dataObjects (2D or 3D) in ``objects`` are vertically stacked along the first \n\
    axis, which is prepended to the existing axes before.");
PyObject* PythonDataObject::PyDataObj_dstack(PyObject* self, PyObject* args, PyObject* kwds)
{
    PyObject* sequence = nullptr;
    unsigned int axis = 0;
    int copyAxisInfo = 0;

    const char* kwlist[] = {"objects", "copyAxisInfo", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O|p", const_cast<char**>(kwlist), &sequence, &copyAxisInfo))
    {
        return nullptr;
    }

    if (PySequence_Check(sequence))
    {
        Py_ssize_t len = PySequence_Size(sequence);
        PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

        if (len > 0)
        {
            ito::DataObject* vector = new ito::DataObject[len];

            for (Py_ssize_t i = 0; i < len; ++i)
            {
                PyObject* item = PySequence_GetItem(sequence, i); // new reference

                if (!PyDataObject_Check(item))
                {
                    Py_DECREF(item);
                    Py_DECREF(retObj);
                    DELETE_AND_SET_NULL_ARRAY(vector);

                    return PyErr_Format(
                        PyExc_RuntimeError, "%i-th element of sequence is no dataObject.", i);
                }
                else
                {
                    vector[i] = *(((PyDataObject*)(item))->dataObject);
                }

                Py_DECREF(item);
            }

            try
            {
                retObj->dataObject = new ito::DataObject(ito::DataObject::stack(vector, len, axis));
            }
            catch (cv::Exception& exc)
            {
                DELETE_AND_SET_NULL_ARRAY(vector);
                Py_DECREF(retObj);
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                return nullptr;
            }

            if (retObj)
            {
                retObj->dataObject->addToProtocol("Created by stacking multiple dataObjects.");

                if (copyAxisInfo > 0)
                {
                    // the method copyAxisTagsTo is also able to copy from a 2d
                    // object to a 3d object. Axes information is always copied
                    // from the last axis towards the first one.
                    vector[0].copyAxisTagsTo(*(retObj->dataObject));
                }
            }

            DELETE_AND_SET_NULL_ARRAY(vector);

            return (PyObject*)retObj;
        }
        else
        {
            return (PyObject*)retObj;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "argument must be a sequence of dataObjects.");
        return nullptr;
    }
}
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectLineCut_doc, "lineCut(coordinates) -> dataObject \n\
\n\
Returns a dataObject with the values of this object along a line with the given coordinates. \n\
\n\
This method uses the **Bresenham** algorithm to get the nearest values along \n\
a line, whose start- and end-point is given by ``coordinates``. These values \n\
are returned in a new :class:`dataObject` with the same data type than this \n\
object. \n\
\n\
This method can be applied to 2D and 3D dataObjects. In the case of a 3D object, \n\
the line cut is defined plane-by-plane and the values are put in one row \n\
for each plane of this object. \n\
\n\
Parameters \n\
---------- \n\
coordinates : list of float or tuple of float \n\
    A sequence of 4 :class:`float` values, that define the physical coordinates \n\
    of the start- and end point of the desired line along which the nearest values \n\
    should be gathered. The values are: ``[x0, y0, x1, y1]``. \n\
\n\
Returns \n\
------- \n\
lineCut : dataObject \n\
    An array of the same data type than this object and shape ``P x N``, that \n\
    contains the nearest values along the given line coordinates. If this \n\
    :class:`dataObject` has two dimensions, ``P = 1``, else ``P`` is equal \n\
    to the size of the first dimension (``shape[0]``). ``N`` corresponds to \n\
    the number of points along the line, defined by the used **Bresenham** \n\
    algorithm. \n\
\n\
Raises \n\
------ \n\
RuntimeError \n\
    if this dataObject has more than three dimensions.");
PyObject* PythonDataObject::PyDataObj_lineCut(PyDataObject* self, PyObject* args)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "data object is NULL");
        return NULL;
    }

    PyObject* sequence = NULL;

    if (!PyArg_ParseTuple(args, "O", &sequence))
    {
        return NULL;
    }

    bool ok;
    QVector<double> coordinates = PythonQtConversion::PyObjGetDoubleArray(sequence, true, ok);

    if (!ok || coordinates.size() != 4)
    {
        return PyErr_Format(PyExc_ValueError, "coordinates must be a sequence of 4 float values.");
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    try
    {
        // new dataObject should always be the owner of its data, therefore base of resultObject
        // remains None
        retObj->dataObject = new ito::DataObject(
            self->dataObject->lineCut(coordinates.constData(), coordinates.size()));
    }
    catch (cv::Exception& exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (retObj)
    {
        retObj->dataObject->addToProtocol("Created taking a lineCut across a dataObject.");
    }

    return (PyObject*)retObj;
}

//-------------------------------------------------------------------------------------
void PythonDataObject::PyDataObj_Capsule_Destructor(PyObject* capsule)
{
    PyArrayInterface* inter = (PyArrayInterface*)PyCapsule_GetPointer(capsule, nullptr);

    if (inter != nullptr)
    {
        free(inter->shape);
        free(inter->strides);
    }

    DELETE_AND_SET_NULL(inter);
}

// PyObject* PythonDataObject::PyDataObj_StaticArange(PyDataObject *self, PyObject *args)
//{
//    return PyObject_Call((PyObject*)&PyDataObjectType, NULL, NULL);
//}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectStaticZeros_doc, "dims, dtype= \"uint8\", continuous = 0) -> dataObject \n\
\n\
Creates a dataObject filled with zeros. \n\
\n\
Parameters \n\
---------- \n\
dims : tuple of int or list of int \n\
    ``dims`` is the shape of the new :class:`dataObject`. The length of this list \n\
    or tuple defines the number of dimensions, e.g. ``[2, 3]`` creates a 2D dataObject\n\
    with two rows and three columns. \n\
dtype : str, optional \n\
    The desired data type for the elements in the returned :class:`dataObject`. \n\
    Possible values are: ``int8``, ``uint8``, ..., ``int32``, ``float32``, \n\
    ``float64``, ``complex64``, ``complex128``, ``rgba32``. \n\
continuous : int, optional \n\
    This value defines if the planes (each sub-array of the last two dimensions) \n\
    are continuously allocated in memory (``1``) or distributed in various smaller \n\
    junks (``0``, default). The latter is recommended for huge, n-dimensional matrices. \n\
    This argument is only considered for ``len(dims) > 2``. \n\
\n\
Returns \n\
------- \n\
array : dataObject \n\
    The newly created dataObject of shape ``dims`` and data type ``dtype``, filled with \n\
    zeros. \n\
\n\
See Also \n\
-------- \n\
eye : method for creating an eye matrix \n\
ones : method for creating a matrix filled with ones \n\
\n\
Notes \n\
----- \n\
For the color data type ``rgba32``, every value will be black and transparent: \n\
``(r=0, g=0, b=0, alpha=0)``.");
PyObject* PythonDataObject::PyDataObj_StaticZeros(
    PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    int typeno = -1;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError())
        return NULL;

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int* sizes2 = new int[sizes.size()];
        for (unsigned int i = 0; i < sizes.size(); i++)
            sizes2[i] = sizes[i];
        // no lock is necessary since eye is allocating the data block and no other access is
        // possible at this moment
        selfDO->dataObject->zeros(sizes.size(), sizes2, typeno, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectStaticOnes_doc, "ones(dims, dtype= \"uint8\", continuous = 0) -> dataObject \n\
\n\
Creates a dataObject filled ones. \n\
\n\
Parameters \n\
---------- \n\
dims : tuple of int or list of int \n\
    ``dims`` is the shape of the new :class:`dataObject`. The length of this list \n\
    or tuple defines the number of dimensions, e.g. ``[2, 3]`` creates a 2D dataObject\n\
    with two rows and three columns. \n\
dtype : str, optional \n\
    The desired data type for the elements in the returned :class:`dataObject`. \n\
    Possible values are: ``int8``, ``uint8``, ..., ``int32``, ``float32``, \n\
    ``float64``, ``complex64``, ``complex128``, ``rgba32``. \n\
continuous : int, optional \n\
    This value defines if the planes (each sub-array of the last two dimensions) \n\
    are continuously allocated in memory (``1``) or distributed in various smaller \n\
    junks (``0``, default). The latter is recommended for huge, n-dimensional matrices. \n\
    This argument is only considered for ``len(dims) > 2``. \n\
\n\
Returns \n\
------- \n\
array : dataObject \n\
    The newly created dataObject of shape ``dims`` and data type ``dtype``, filled with \n\
    ones. \n\
\n\
See Also \n\
-------- \n\
eye : method for creating an eye matrix \n\
zeros : method for creating a matrix filled with zeros \n\
\n\
Notes \n\
----- \n\
For the color data type ``rgba32``, every value will be white: \n\
``(r=255, g=255, b=255, alpha=255)``.");
PyObject* PythonDataObject::PyDataObj_StaticOnes(PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    int typeno = -1;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError())
        return NULL;

    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(
            PyExc_TypeError, "Type uint32 not supported due to incompatibility with OpenCV.");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != nullptr)
    {
        int* sizes2 = new int[sizes.size()];

        for (unsigned int i = 0; i < sizes.size(); i++)
        {
            sizes2[i] = sizes[i];
        }

        // no lock is necessary since eye is allocating the data block and no other access is
        // possible at this moment
        try
        {
            selfDO->dataObject->ones(sizes.size(), sizes2, typeno, continuous);
        }
        catch (cv::Exception& exc)
        {
            Py_DECREF(selfDO);
            selfDO = nullptr;
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        }

        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    return (PyObject*)selfDO;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectStaticNans_doc, "nans(dims, dtype= \"float32\", continuous = 0) -> dataObject \n\
\n\
Creates a floating-point dataObject filled with ``NaN`` values. \n\
\n\
Parameters \n\
---------- \n\
dims : tuple of int or list of int \n\
    ``dims`` is the shape of the new :class:`dataObject`. The length of this list \n\
    or tuple defines the number of dimensions, e.g. ``[2, 3]`` creates a 2D dataObject\n\
    with two rows and three columns. \n\
dtype : str, optional \n\
    The desired data type for the elements in the returned :class:`dataObject`. \n\
    Possible values are: ``float32``, ``float64``, ``complex64``, ``complex128``. \n\
continuous : int, optional \n\
    This value defines if the planes (each sub-array of the last two dimensions) \n\
    are continuously allocated in memory (``1``) or distributed in various smaller \n\
    junks (``0``, default). The latter is recommended for huge, n-dimensional matrices. \n\
    This argument is only considered for ``len(dims) > 2``. \n\
\n\
Returns \n\
------- \n\
array : dataObject \n\
    The newly created dataObject of shape ``dims`` and data type ``dtype``, filled with \n\
    ``NaN``. \n\
\n\
See Also \n\
-------- \n\
eye : method for creating an eye matrix \n\
zeros : method for creating a matrix filled with zeros \n\
ones : method for creating a matrix filled with ones.");
PyObject* PythonDataObject::PyDataObj_StaticNans(PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    int typeno = dObjTypeFromName("float32");
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError())
        return NULL;

    if (!(typeno == ito::tFloat32 || typeno == ito::tFloat64 || typeno == ito::tComplex64 ||
          typeno == ito::tComplex128))
    {
        // NaN values can only fill arrays float and complex dtypes!
        PyErr_SetString(
            PyExc_TypeError,
            "This function is only supported for float32, float64, complex64 and complex128!");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int* sizes2 = new int[sizes.size()];

        for (unsigned int i = 0; i < sizes.size(); i++)
        {
            sizes2[i] = sizes[i];
        }

        // no lock is necessary since eye is allocating the data block and no other access is
        // possible at this moment
        selfDO->dataObject->nans(sizes.size(), sizes2, typeno, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectStaticRand_doc, "rand(dims, dtype= \"uint8\", continuous = 0) -> dataObject \n\
\n\
Creates a dataObject filled with uniformly distributed random values. \n\
\n\
The value range of the random numbers depend on the desired data type ``dtype``: \n\
\n\
1. **integer types**: The random values are in the range ``[min(dtype), max(dtype)]``. \n\
2. **floating point types**: The random values are in the range ``[0, 1)``. \n\
3. **rgba32**: All colours as well as the alpha value is independently distributed in \n\
   the range ``[0, 255]``. \n\
4. **complex types**: Both the real as well as imaginary part is independently \n\
   distributed in the range ``[0, 1)``. \n\
\n\
Parameters \n\
---------- \n\
dims : tuple of int or list of int \n\
    ``dims`` is the shape of the new :class:`dataObject`. The length of this list \n\
    or tuple defines the number of dimensions, e.g. ``[2, 3]`` creates a 2D dataObject\n\
    with two rows and three columns. \n\
dtype : str, optional \n\
    The desired data type for the elements in the returned :class:`dataObject`. \n\
    Possible values are: ``int8``, ``uint8``, ..., ``int32``, ``float32``, \n\
    ``float64``, ``complex64``, ``complex128``, ``rgba32``. \n\
continuous : int, optional \n\
    This value defines if the planes (each sub-array of the last two dimensions) \n\
    are continuously allocated in memory (``1``) or distributed in various smaller \n\
    junks (``0``, default). The latter is recommended for huge, n-dimensional matrices. \n\
    This argument is only considered for ``len(dims) > 2``. \n\
\n\
Returns \n\
------- \n\
array : dataObject \n\
    The newly created dataObject of shape ``dims`` and data type ``dtype``, filled with \n\
    random numbers. \n\
\n\
See Also \n\
-------- \n\
randN : method for creating a matrix filled with gaussian distributed values");
PyObject* PythonDataObject::PyDataObj_StaticRand(PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    int typeno = -1;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError())
        return NULL;

    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(
            PyExc_TypeError, "Type uint32 not supported due to incompatibility with OpenCV.");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int* sizes2 = new int[sizes.size()];


        for (unsigned int i = 0; i < sizes.size(); i++)
        {
            sizes2[i] = sizes[i];
        }

        // no lock is necessary since eye is allocating the data block and no other access is
        // possible at this moment
        selfDO->dataObject->rand(sizes.size(), sizes2, typeno, false, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectStaticRandN_doc, "randN(dims, dtype= \"uint8\", continuous = 0) -> dataObject \n\
\n\
Creates a dataObject filled with Gaussian distributed random values. \n\
\n\
The value range of the random numbers depend on the desired data type ``dtype``: \n\
\n\
1. **integer types**: The random values are in the range ``[min(dtype), max(dtype)]``. \n\
2. **floating point types**: The random values are in the range ``[0, 1)``. \n\
3. **rgba32**: All colours as well as the alpha value is independently distributed in \n\
   the range ``[0, 255]``. \n\
4. **complex types**: Both the real as well as imaginary part is independently \n\
   distributed in the range ``[0, 1)``. \n\
\n\
The mean ``m`` and standard deviation ``s`` of the Gaussian distribution is as follows: \n\
\n\
* For **integer** and **rgba32** types holds: ``m = (min + max) / 2.0`` and \n\
  ``s = (max - min) / 6.0``. \n\
* For all **floating point** types holds: ``m = 0.0`` and ``s = 1/3``. \n\
\n\
Parameters \n\
---------- \n\
dims : tuple of int or list of int \n\
    ``dims`` is the shape of the new :class:`dataObject`. The length of this list \n\
    or tuple defines the number of dimensions, e.g. ``[2, 3]`` creates a 2D dataObject\n\
    with two rows and three columns. \n\
dtype : str, optional \n\
    The desired data type for the elements in the returned :class:`dataObject`. \n\
    Possible values are: ``int8``, ``uint8``, ..., ``int32``, ``float32``, \n\
    ``float64``, ``complex64``, ``complex128``, ``rgba32``. \n\
continuous : int, optional \n\
    This value defines if the planes (each sub-array of the last two dimensions) \n\
    are continuously allocated in memory (``1``) or distributed in various smaller \n\
    junks (``0``, default). The latter is recommended for huge, n-dimensional matrices. \n\
    This argument is only considered for ``len(dims) > 2``. \n\
\n\
Returns \n\
------- \n\
array : dataObject \n\
    The newly created dataObject of shape ``dims`` and data type ``dtype``, filled with \n\
    random numbers. \n\
\n\
See Also \n\
-------- \n\
rand : method for creating a matrix filled with unformly distributed values");
PyObject* PythonDataObject::PyDataObj_StaticRandN(
    PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    int typeno = -1;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError())
        return NULL;

    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(
            PyExc_TypeError, "Type uint32 not supported due to incompatibility with OpenCV.");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int* sizes2 = new int[sizes.size()];

        for (unsigned int i = 0; i < sizes.size(); i++)
        {
            sizes2[i] = sizes[i];
        }

        // no lock is necessary since eye is allocating the data block and no other access is
        // possible at this moment
        selfDO->dataObject->rand(sizes.size(), sizes2, typeno, true, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticEye_doc, "eye(size, dtype= \"uint8\") -> dataObject \n\
\n\
Creates a two-dimensional, squared ``eye`` matrix.\n\
\n\
An eye matrix is an array where all elements are equal to zero, except for \n\
the diagonal values which are set to ``1``. For ``dtype == rgba32``, the \n\
diagonal values are ``r = 0, g = 0, b = 1, alpha = 0``. \n\
\n\
Parameters \n\
---------- \n\
size : int \n\
    The size of the squared matrix (single integer value).\n\
dtype : str, optional \n\
    The desired data type for the elements in the returned :class:`dataObject`. \n\
    Possible values are: ``int8``, ``uint8``, ..., ``int32``, ``float32``, \n\
    ``float64``, ``complex64``, ``complex128``, ``rgba32``. \n\
\n\
Returns \n\
------- \n\
eyeMatrix : dataObject \n\
    The created eye-matrix as ``size x size`` :class:`dataObject`. \n\
\n\
See Also \n\
-------- \n\
ones : method for creating a matrix filled with ones \n\
zeros : method for creating a matrix filled with zeros");
PyObject* PythonDataObject::PyDataObj_StaticEye(PyObject* /*self*/, PyObject* args, PyObject* kwds)
{
    static const char* kwlist[] = {"size", "dtype", NULL};
    int size = 0;
    const char* type = "uint8";
    RetVal retValue(retOk);

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|s", const_cast<char**>(kwlist), &size, &type))
    {
        return NULL;
    }

    int typeno = dObjTypeFromName(type);

    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(
            PyExc_TypeError, "Type uint32 not supported due to incompatibility with OpenCV.");
        return nullptr;
    }
    else if (typeno == ito::tDateTime || typeno == ito::tTimeDelta)
    {
        PyErr_SetString(PyExc_TypeError, "Eye not possible for type datetime or timedelta.");
        return nullptr;
    }

    if (typeno >= 0)
    {
        if (size > 0)
        {
            PyDataObject* selfDO = createEmptyPyDataObject();
            selfDO->dataObject = new ito::DataObject();
            // no lock is necessary since eye is allocating the data block and no other access is
            // possible at this moment
            selfDO->dataObject->eye(size, typeno);
            return (PyObject*)selfDO;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "size must be bigger than zero.");
            return nullptr;
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "unknown dtype");
        return nullptr;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticFromNumpyColor_doc, "fromNumpyColor(array) -> dataObject \n\
\n\
Creates a ``rgba32`` dataObject from a three-dimensional numpy.ndarray. \n\
\n\
Static method for creating a 2D ``M x N`` :class:`dataObject` of data type ``rgba32`` \n\
from a three-dimensional, ``uint8`` :class:`numpy.ndarray``. This ``array`` must have \n\
the shape ``M x N x 3`` or ``M x N x 4``. Each vector ``array[i, j, :]`` is then \n\
used to create one ``rgba32`` value in the returned :class:`dataObject`. The meaning \n\
of this vector is: \n\
\n\
1. (blue, green, red) if ``array`` consists of three channels (last dimension). \n\
   The ``rgba32`` value then gets an alpha value set to 255 everywhere. \n\
2. (blue, green, red, alpha) if ``array`` consists of four channels (last dimension). \n\
\n\
This method can especially be used to convert numpy.arrays that are obtained by methods \n\
from packages like ``OpenCV (cv2)`` or ``PIL`` to dataObjects. \n\
\n\
Parameters \n\
---------- \n\
array : numpy.ndarray \n\
    ``M x N x 3`` or ``M x N x 4``, uint8 :class:`numpy.ndarray` \n\
\n\
Returns \n\
------- \n\
dataObject \n\
    Coloured dataObject of shape ``M x N`` and data type ``rgba32``.");
PyObject* PythonDataObject::PyDataObj_StaticFromNumpyColor(
    PyObject* self, PyObject* args, PyObject* kwds)
{
    static const char* kwlist[] = {"array", NULL};

    // will be a borrowed reference
    PyObject* obj = nullptr;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!", const_cast<char**>(kwlist), &PyArray_Type, &obj))
    {
        return nullptr;
    }

    PyArrayObject* ndArray = (PyArrayObject*)obj;
    PyArray_Descr* descr = PyArray_DESCR(ndArray);
    int typeno = -1;
    uchar* data = nullptr;

    // at first, check copyObject. there are three cases: 1. we can take it as it is, 2. it is
    // compatible but has to be converted, 3. it is incompatible
    if (!(descr->byteorder == '<' || descr->byteorder == '|' ||
          (descr->byteorder == '=' && NPY_NATBYTE == NPY_LITTLE)))
    {
        PyErr_SetString(
            PyExc_TypeError,
            "Given numpy array has wrong byteorder (litte endian desired), which cannot be "
            "transformed to dataObject");
        return nullptr;
    }
    else
    {
        // check whether type of ndarray exists for data object
        typeno = getDObjTypeOfNpArray(descr->kind, PyArray_ITEMSIZE(ndArray));

        if (typeno != ito::tUInt8)
        {
            PyErr_SetString(
                PyExc_TypeError,
                "Only numpy arrays of type uint8 can be transformed to a rgba32 dataObject");
            return nullptr;
        }

        // verify that ndArray is c-contiguous
        // now we always have an increased reference of ndArray (either reference of
        // old ndArray or new object with new reference)
        ndArray = PyArray_GETCONTIGUOUS(ndArray);

        if (ndArray == nullptr)
        {
            PyErr_SetString(
                PyExc_TypeError,
                "An error occurred while transforming the given numpy array to a c-contiguous "
                "array.");
            return nullptr;
        }

        int dimensions = PyArray_NDIM(ndArray); //->nd;
        npy_intp* npsizes = PyArray_DIMS(ndArray);

        // number of bytes to jump from one element in one dimension to the next one
        npy_intp* npsteps = (npy_intp*)PyArray_STRIDES(ndArray);

        if (dimensions != 3 || (npsizes[2] != 3 && npsizes[2] != 4))
        {
            PyErr_SetString(
                PyExc_ValueError,
                "The numpy.array must have three dimensions whereas the size of the last dimension "
                "must be three or four");
            Py_DECREF(ndArray);
            return NULL;
        }

        PyDataObject* pyDataObject = createEmptyPyDataObject();
        int sizes[] = {(int)npsizes[0], (int)npsizes[1]};
        int steps[] = {(int)npsteps[0], (int)npsteps[1], (int)npsteps[2]};
        int chn = (int)npsizes[2];
        data = (uchar*)PyArray_DATA(ndArray);

        if (chn == 4 && npsteps[2] == PyArray_ITEMSIZE(ndArray))
        {
            pyDataObject->dataObject = new ito::DataObject(2, sizes, ito::tRGBA32, data, steps);
        }
        else // 3
        {
            pyDataObject->dataObject = new ito::DataObject(2, sizes, ito::tRGBA32);
            ito::Rgba32* destRow;
            const uchar* srcRow;

            for (int r = 0; r < sizes[0]; ++r)
            {
                srcRow = data + (r * steps[0]);
                destRow = pyDataObject->dataObject->rowPtr<ito::Rgba32>(0, r);
                for (int c = 0; c < sizes[1]; ++c)
                {
                    destRow[c].b = srcRow[0];
                    destRow[c].g = srcRow[steps[2]];
                    destRow[c].r = srcRow[2 * steps[2]];
                    destRow[c].a = (chn == 3) ? 255 : srcRow[3 * steps[2]];
                    srcRow += steps[1];
                }
            }
        }

        Py_DECREF(ndArray);
        return (PyObject*)pyDataObject;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(
    pyDataObjectCopyMetaInfo_doc,
    "copyMetaInfo(sourceObj, copyAxisInfo = True, copyTags = False) \n\
\n\
Copies meta information of another dataObject to this object. \n\
\n\
This method can be used to copy all or parts of meta information of the \n\
:class:`dataObject` ``sourceObj`` to this object. The following things \n\
are copied, depending on the arguments of this method: \n\
\n\
Axis meta information: \n\
\n\
* axis scaling and offset (see :attr:`axisScales` and :attr:`axisOffsets`) \n\
* axis descriptions and units (see :attr:`axisDescriptions` and :attr:`axisUnits`) \n\
\n\
Tags: \n\
\n\
* the entire tag map (string key vs. string or float value), including the protocol \n\
  string. The existing tag map in this object is deleted first. \n\
\n\
Parameters  \n\
----------\n\
sourceObj : dataObject \n\
    source object, where meta information is copied from. \n\
copyAxisInfo : bool, optional \n\
    If ``True``, all axis meta information is copied. \n\
copyTags : bool, optional \n\
    If ``True``, the tags of this data object are cleared and then set to a copy \n\
    of the tags of ``sourceObj``. \n\
\n\
\n\
See Also \n\
-------- \n\
metaDict : this attribute can directly be used to print meta information of a dataObject.");
PyObject* PythonDataObject::PyDataObj_CopyMetaInfo(
    PyDataObject* self, PyObject* args, PyObject* kwds)
{
    Py_ssize_t length = 0;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "DataObject is NULL.");
        return NULL;
    }

    static const char* kwlist[] = {"sourceObj", "copyAxisInfo", "copyTags", NULL};
    PyObject* pyObj = NULL;
    unsigned char copyAxesInfo = 1;
    unsigned char copyTags = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "O!|bb",
            const_cast<char**>(kwlist),
            &PythonDataObject::PyDataObjectType,
            &pyObj,
            &copyAxesInfo,
            &copyTags)) // obj is a borrowed reference
    {
        return NULL;
    }

    PyDataObject* dObj = (PyDataObject*)pyObj;
    try
    {
        if (copyAxesInfo)
        {
            dObj->dataObject->copyAxisTagsTo(*(self->dataObject));
        }

        if (copyTags)
        {
            dObj->dataObject->copyTagMapTo(*(self->dataObject));
        }
    }
    catch (cv::Exception& exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    if (self->dataObject)
    {
        self->dataObject->addToProtocol("Copied meta information from another dataObject.");
    }

    Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------------
PyMethodDef PythonDataObject::PyDataObject_methods[] = {
    {"name", (PyCFunction)PythonDataObject::PyDataObject_name, METH_NOARGS, pyDataObjectName_doc},
    {"data", (PyCFunction)PythonDataObject::PyDataObject_data, METH_NOARGS, pyDataObjectData_doc},

    {"setAxisOffset",
     (PyCFunction)PyDataObj_SetAxisOffset,
     METH_VARARGS,
     pyDataObjectSetAxisOffset_doc},
    {"setAxisScale",
     (PyCFunction)PyDataObj_SetAxisScale,
     METH_VARARGS,
     pyDataObjectSetAxisScale_doc},
    {"setAxisDescription",
     (PyCFunction)PyDataObj_SetAxisDescription,
     METH_VARARGS,
     pyDataObjectSetAxisDescription_doc},
    {"setAxisUnit", (PyCFunction)PyDataObj_SetAxisUnit, METH_VARARGS, pyDataObjectSetAxisUnit_doc},
    {"setTag", (PyCFunction)PyDataObj_SetTag, METH_VARARGS, pyDataObjectSetTag_doc},
    {"deleteTag", (PyCFunction)PyDataObj_DeleteTag, METH_VARARGS, pyDataObjectDeleteTag_doc},
    {"existTag", (PyCFunction)PyDataObj_TagExists, METH_VARARGS, pyDataObjectTagExists_doc},
    {"getTagListSize",
     (PyCFunction)PyDataObj_GetTagListSize,
     METH_NOARGS,
     pyDataObjectGetTagListSize_doc},
    {"addToProtocol",
     (PyCFunction)PyDataObj_AddToProtocol,
     METH_VARARGS,
     pyDataObjectAddToProtocol_doc},
    {"physToPix",
     (PyCFunction)PyDataObj_PhysToPix,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObjectPhysToPix_doc},
    {"pixToPhys",
     (PyCFunction)PyDataObj_PixToPhys,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObjectPixToPhys_doc},
    {"copyMetaInfo",
     (PyCFunction)PyDataObj_CopyMetaInfo,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObjectCopyMetaInfo_doc},

    {"copy",
     (PyCFunction)PythonDataObject::PyDataObject_copy,
     METH_VARARGS | METH_KEYWORDS,
     pyDataObjectCopy_doc},
    {"astype",
     (PyCFunction)PythonDataObject::PyDataObject_astype,
     METH_VARARGS | METH_KEYWORDS,
     pyDataObjectAstype_doc},
    {"normalize",
     (PyCFunction)PythonDataObject::PyDataObject_normalize,
     METH_VARARGS | METH_KEYWORDS,
     pyDataObjectNormalize_doc},
    {"locateROI",
     (PyCFunction)PythonDataObject::PyDataObject_locateROI,
     METH_NOARGS,
     pyDataObjectLocateROI_doc},
    {"adjustROI",
     (PyCFunction)PythonDataObject::PyDataObject_adjustROI,
     METH_VARARGS | METH_KEYWORDS,
     pyDataObjectAdjustROI_doc},
    {"squeeze",
     (PyCFunction)PythonDataObject::PyDataObject_squeeze,
     METH_NOARGS,
     pyDataObjectSqueeze_doc},
    {"size", (PyCFunction)PythonDataObject::PyDataObject_size, METH_VARARGS, pyDataObjectSize_doc},
    {"conj", (PyCFunction)PythonDataObject::PyDataObject_conj, METH_NOARGS, pyDataObjectConj_doc},
    {"conjugate",
     (PyCFunction)PythonDataObject::PyDataObject_conjugate,
     METH_NOARGS,
     pyDataObjectConjugate_doc},
    {"adj", (PyCFunction)PythonDataObject::PyDataObject_adj, METH_NOARGS, pyDataObjectAdj_doc},
    {"adjugate", (PyCFunction)PyDataObject_adjugate, METH_NOARGS, pyDataObjectAdjugate_doc},
    {"trans",
     (PyCFunction)PythonDataObject::PyDataObject_trans,
     METH_NOARGS,
     pyDataObjectTrans_doc},
    {"div", (PyCFunction)PythonDataObject::PyDataObject_div, METH_VARARGS, pyDataObjectDiv_doc},
    {"mul", (PyCFunction)PythonDataObject::PyDataObject_mul, METH_VARARGS, pyDataObjectMul_doc},
    {"makeContinuous",
     (PyCFunction)PythonDataObject::PyDataObject_makeContinuous,
     METH_NOARGS,
     pyDataObjectMakeContinuous_doc},
    {"reshape",
     (PyCFunction)PythonDataObject::PyDataObject_reshape,
     METH_VARARGS | METH_KEYWORDS,
     pyDataObjectReshape_doc},
    {"zeros",
     (PyCFunction)PythonDataObject::PyDataObj_StaticZeros,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticZeros_doc},
    {"ones",
     (PyCFunction)PythonDataObject::PyDataObj_StaticOnes,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticOnes_doc},
    {"nans",
     (PyCFunction)PythonDataObject::PyDataObj_StaticNans,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticNans_doc},
    {"rand",
     (PyCFunction)PythonDataObject::PyDataObj_StaticRand,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticRand_doc},
    {"randN",
     (PyCFunction)PythonDataObject::PyDataObj_StaticRandN,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticRandN_doc},
    {"eye",
     (PyCFunction)PythonDataObject::PyDataObj_StaticEye,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticEye_doc},
    {"fromNumpyColor",
     (PyCFunction)PythonDataObject::PyDataObj_StaticFromNumpyColor,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectStaticFromNumpyColor_doc},
    {"__reduce__",
     (PyCFunction)PythonDataObject::PyDataObj_Reduce,
     METH_VARARGS,
     "__reduce__ method for handle pickling commands"},
    {"__setstate__",
     (PyCFunction)PythonDataObject::PyDataObj_SetState,
     METH_VARARGS,
     "__setstate__ method for handle unpickling commands"},
    {"__array__",
     (PyCFunction)PythonDataObject::PyDataObj_Array_,
     METH_VARARGS,
     dataObject_Array__doc},
    {"createMask",
     (PyCFunction)PythonDataObject::PyDataObject_createMask,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObjectCreateMask_doc},
    {"dstack",
     (PyCFunction)PythonDataObject::PyDataObj_dstack,
     METH_KEYWORDS | METH_VARARGS | METH_STATIC,
     pyDataObjectDstack_doc},
    {"lineCut",
     (PyCFunction)PythonDataObject::PyDataObj_lineCut,
     METH_VARARGS,
     pyDataObjectLineCut_doc},
    {"abs", (PyCFunction)PythonDataObject::PyDataObject_abs, METH_NOARGS, pyDataObjectAbs_doc},
    {"arg", (PyCFunction)PythonDataObject::PyDataObject_arg, METH_NOARGS, pyDataObjectArg_doc},

    {"tolist",
     (PyCFunction)PythonDataObject::PyDataObj_ToList,
     METH_NOARGS,
     pyDataObjectToList_doc}, //"returns nested list of content of data object"
    {"toGray",
     (PyCFunction)PythonDataObject::PyDataObj_ToGray,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObj_ToGray_doc},
    {"toNumpyColor",
     (PyCFunction)PythonDataObject::PyDataObj_ToNumpyColor,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObj_ToNumpyColor_doc},
    {"splitColor",
     (PyCFunction)PythonDataObject::PyDataObj_SplitColor,
     METH_KEYWORDS | METH_VARARGS,
     pyDataObj_SplitColor_doc},
    {NULL} /* Sentinel */
};

PyDoc_STRVAR(
    pyDataObject_base_doc,
    "None or dataObject or np.ndarray : Optional base object, this object shares its memory with "
    "(read-only).");

//-------------------------------------------------------------------------------------
PyMemberDef PythonDataObject::PyDataObject_members[] = {
    {"base", T_OBJECT, offsetof(PyDataObject, base), READONLY, pyDataObject_base_doc},
    {NULL} /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonDataObject::PyDataObjectModule = {
    PyModuleDef_HEAD_INIT,
    "dataObject",
    "itom DataObject type in python",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL};

//-------------------------------------------------------------------------------------
PyGetSetDef PythonDataObject::PyDataObject_getseters[] = {
    {"dims", (getter)PyDataObj_GetDims, NULL, dataObjectAttrDims_doc, NULL},
    {"ndim", (getter)PyDataObj_GetDims, NULL, dataObjectAttrDims_doc, NULL},
    {"dtype", (getter)PyDataObj_GetType, NULL, dataObjectAttrType_doc, NULL},
    {"shape", (getter)PyDataObj_GetShape, NULL, dataObjectAttrShape_doc, NULL},
    {"continuous", (getter)PyDataObj_GetContinuous, NULL, dataObjectAttrContinuous_doc, NULL},
    {"metaDict",
     (getter)PyDataObject_getTagDict,
     (setter)PyDataObject_setTagDict,
     dataObjectAttrTagDict_doc,
     NULL},

    {"tags",
     (getter)PyDataObject_getTags,
     (setter)PyDataObject_setTags,
     dataObjectAttrTags_doc,
     NULL},
    {"axisScales",
     (getter)PyDataObject_getAxisScales,
     (setter)PyDataObject_setAxisScales,
     dataObjectAttrAxisScales_doc,
     NULL},
    {"axisOffsets",
     (getter)PyDataObject_getAxisOffsets,
     (setter)PyDataObject_setAxisOffsets,
     dataObjectAttrAxisOffsets_doc,
     NULL},
    {"axisDescriptions",
     (getter)PyDataObject_getAxisDescriptions,
     (setter)PyDataObject_setAxisDescriptions,
     dataObjectAttrAxisDescriptions_doc,
     NULL},
    {"axisUnits",
     (getter)PyDataObject_getAxisUnits,
     (setter)PyDataObject_setAxisUnits,
     dataObjectAttrAxisUnits_doc,
     NULL},
    {"valueUnit",
     (getter)PyDataObject_getValueUnit,
     (setter)PyDataObject_setValueUnit,
     dataObjectAttrValueUnit_doc,
     NULL},
    {"valueDescription",
     (getter)PyDataObject_getValueDescription,
     (setter)PyDataObject_setValueDescription,
     dataObjectAttrValueDescription_doc,
     NULL},
    {"valueScale", (getter)PyDataObject_getValueScale, NULL, dataObjectAttrValueScale_doc, NULL},
    {"valueOffset", (getter)PyDataObject_getValueOffset, NULL, dataObjectAttrValueOffset_doc, NULL},
    {"value",
     (getter)PyDataObject_getValue,
     (setter)PyDataObject_setValue,
     dataObjectAttrValue_doc,
     NULL},
    {"xyRotationalMatrix",
     (getter)PyDataObject_getXYRotationalMatrix,
     (setter)PyDataObject_setXYRotationalMatrix,
     dataObjectAttrRotationalMatrix_doc,
     NULL},
    {"real",
     (getter)PyDataObject_getReal,
     (setter)PyDataObject_setReal,
     dataObjectAttrReal_doc,
     NULL},
    {"imag",
     (getter)PyDataObject_getImag,
     (setter)PyDataObject_setImag,
     dataObjectAttrImag_doc,
     NULL},

    {"__array_struct__",
     (getter)PyDataObj_Array_StructGet,
     NULL,
     dataObjectArray_StructGet_doc,
     NULL},
    {"__array_interface__",
     (getter)PyDataObj_Array_Interface,
     NULL,
     dataObjectArray_Interface_doc,
     NULL},

    {"T", (getter)PyDataObject_transpose, nullptr, dataObjectTranspose_doc, nullptr},

    {NULL} /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonDataObject::PyDataObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0) "itom.dataObject", /* tp_name */
    sizeof(PyDataObject), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)PyDataObject_dealloc, /* tp_dealloc */
    0, /* tp_print */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    (reprfunc)PyDataObject_repr, /* tp_repr */
    &PyDataObject_numberProtocol, /* tp_as_number */
    0, /* tp_as_sequence */
    &PyDataObject_mappingProtocol, /* tp_as_mapping */
    0, /* tp_hash  */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    dataObjectInit_doc /*"dataObject objects"*/, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    (richcmpfunc)PyDataObject_RichCompare, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    (getiterfunc)PyDataObj_getiter, /* tp_iter */
    (iternextfunc)PyDataObjectIter_new, /* tp_iternext */
    PyDataObject_methods, /* tp_methods */
    PyDataObject_members, /* tp_members */
    PyDataObject_getseters, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)PythonDataObject::PyDataObject_init, /* tp_init */
    0, /* tp_alloc */
    PyDataObject_new /*PyType_GenericNew*/ /*PythonStream_new,*/ /* tp_new */
};

//-------------------------------------------------------------------------------------
PyNumberMethods PythonDataObject::PyDataObject_numberProtocol = {
    (binaryfunc)PyDataObj_nbAdd, /* nb_add */
    (binaryfunc)PyDataObj_nbSubtract, /* nb_subtract */
    (binaryfunc)PyDataObj_nbMultiply, /* nb_multiply */
    (binaryfunc)PyDataObj_nbRemainder, /* nb_remainder */
    (binaryfunc)PyDataObj_nbDivmod, /* nb_divmod */
    (ternaryfunc)PyDataObj_nbPower, /* nb_power */
    (unaryfunc)PyDataObj_nbNegative, /* nb_negative */
    (unaryfunc)PyDataObj_nbPositive, /* nb_positive */
    (unaryfunc)PyDataObj_nbAbsolute, /* nb_absolute */
    (inquiry)PyDataObj_nbBool, /* nb_bool */
    (unaryfunc)PyDataObj_nbInvert, /* nb_invert */
    (binaryfunc)PyDataObj_nbLshift, /* nb_lshift */
    (binaryfunc)PyDataObj_nbRshift, /* nb_rshift */
    (binaryfunc)PyDataObj_nbAnd, /* nb_and */
    (binaryfunc)PyDataObj_nbXor, /* nb_xor */
    (binaryfunc)PyDataObj_nbOr, /* nb_or */
    0, /* nb_int */
    0, /* nb_reserved */
    0, /* nb_float */
    (binaryfunc)PyDataObj_nbInplaceAdd, /* nb_inplace_add */
    (binaryfunc)PyDataObj_nbInplaceSubtract, /* nb_inplace_subtract */
    (binaryfunc)PyDataObj_nbInplaceMultiply, /* nb_inplace_multiply*/
    (binaryfunc)PyDataObj_nbInplaceRemainder, /* nb_inplace_remainder */
    (ternaryfunc)PyDataObj_nbInplacePower, /* nb_inplace_power */
    (binaryfunc)PyDataObj_nbInplaceLshift, /* nb_inplace_lshift */
    (binaryfunc)PyDataObj_nbInplaceRshift, /* nb_inplace_rshift */
    (binaryfunc)PyDataObj_nbInplaceAnd, /* nb_inplace_and */
    (binaryfunc)PyDataObj_nbInplaceXor, /* nb_inplace_xor */
    (binaryfunc)PyDataObj_nbInplaceOr, /* nb_inplace_or */
    (binaryfunc)0, /* nb_floor_divide */
    (binaryfunc)PyDataObj_nbDivide, /* nb_true_divide */
    0, /* nb_inplace_floor_divide */
    (binaryfunc)PyDataObj_nbInplaceTrueDivide /* nb_inplace_true_divide */
    ,
    0, /* np_index */
    (binaryfunc)PyDataObj_nbMatrixMultiply, /* nb_matrix_multiply */
    (binaryfunc)PyDataObj_nbInplaceMatrixMultiply /* nb_inplace_matrix_multiply */
};

//-------------------------------------------------------------------------------------
PyMappingMethods PythonDataObject::PyDataObject_mappingProtocol = {
    (lenfunc)PyDataObj_mappingLength,
    (binaryfunc)PyDataObj_mappingGetElem,
    (objobjargproc)PyDataObj_mappingSetElem};

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObjectIter_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyObject* dataObject = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &dataObject))
    {
        return NULL;
    }

    PyDataObjectIter* self = (PyDataObjectIter*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        PythonDataObject::PyDataObject* dobj = (PyDataObject*)dataObject;
        Py_INCREF(dataObject);
        self->base = dataObject;

        if (dobj->dataObject)
        {
            self->it = dobj->dataObject->begin();
            self->itEnd = dobj->dataObject->end();
            self->len = dobj->dataObject->getTotal();
        }
        else
        {
            self->len = 0;
        }
    }

    return (PyObject*)self;
}

//-------------------------------------------------------------------------------------
int PythonDataObject::PyDataObjectIter_init(
    PyDataObjectIter* /*self*/, PyObject* /*args*/, PyObject* /*kwds*/)
{
    return 0;
}

//-------------------------------------------------------------------------------------
void PythonDataObject::PyDataObjectIter_dealloc(PyDataObjectIter* self)
{
    self->it = ito::DObjConstIterator();
    self->itEnd = self->it;
    Py_XDECREF(self->base);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//-------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObjectIter_iternext(PyDataObjectIter* self)
{
    if (self->it == self->itEnd)
    {
        PyErr_SetString(PyExc_StopIteration, "");
        return nullptr;
    }

    PyDataObject* dObj = (PyDataObject*)self->base;
    PyObject* output = nullptr;

    if (dObj->dataObject == nullptr)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return nullptr;
    }

    switch (dObj->dataObject->getType())
    {
    case ito::tInt8:
        output = PyLong_FromLong((long)(*((ito::int8*)(*(self->it)))));
        break;
    case ito::tUInt8:
        output = PyLong_FromLong((long)(*((ito::uint8*)(*(self->it)))));
        break;
    case ito::tInt16:
        output = PyLong_FromLong((long)(*((ito::int16*)(*(self->it)))));
        break;
    case ito::tUInt16:
        output = PyLong_FromLong((long)(*((ito::uint16*)(*(self->it)))));
        break;
    case ito::tInt32:
        output = PyLong_FromLong((long)(*((ito::int32*)(*(self->it)))));
        break;
    case ito::tRGBA32: {
        ito::PythonRgba::PyRgba* color = ito::PythonRgba::createEmptyPyRgba();
        if (color)
        {
            color->rgba = ((Rgba32*)(*(self->it)))->rgba;
        }
        output = (PyObject*)color;
    }
    break;
    case ito::tFloat32:
        output = PyFloat_FromDouble((double)(*((ito::float32*)(*(self->it)))));
        break;
    case ito::tFloat64:
        output = PyFloat_FromDouble((double)(*((ito::float64*)(*(self->it)))));
        break;
    case ito::tComplex64: {
        complex64* value = (complex64*)(*(self->it));
        output = PyComplex_FromDoubles((double)value->real(), (double)value->imag());
        break;
    }
    case ito::tComplex128: {
        complex128* value = (complex128*)(*(self->it));
        output = PyComplex_FromDoubles((double)value->real(), (double)value->imag());
        break;
    }
    case ito::tDateTime: {
        const ito::DateTime* value = reinterpret_cast<const ito::DateTime*>(*(self->it));
        output = PythonDateTime::GetPyDateTime(*value);
        break;
    }
    case ito::tTimeDelta: {
        const ito::TimeDelta* value = reinterpret_cast<const ito::TimeDelta*>(*(self->it));
        output = PythonDateTime::GetPyTimeDelta(*value);
        break;
    }
    default:
        PyErr_SetString(PyExc_NotImplementedError, "Type not implemented yet");
    }

    self->it++;
    return output;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectIterLen_doc, "Private method returning an estimate of len(list(it)).");
PyObject* PythonDataObject::PyDataObjectIter_len(PyDataObjectIter* self)
{
    return PyLong_FromUnsignedLong(self->len);
}

//-------------------------------------------------------------------------------------
PyMethodDef PythonDataObject::PyDataObjectIter_methods[] = {
    {"__length_hint__", (PyCFunction)PyDataObjectIter_len, METH_NOARGS, pyDataObjectIterLen_doc},
    {NULL, NULL} /* sentinel */
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonDataObject::PyDataObjectIterType = {
    PyVarObject_HEAD_INIT(NULL, 0) "itom.dataObjectIterator", /* tp_name */
    sizeof(PyDataObjectIter), /* tp_basicsize */
    0, /* tp_itemsize */
    /* methods */
    (destructor)PyDataObjectIter_dealloc, /* tp_dealloc */
    0, /* tp_print */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    0, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash */
    0, /* tp_call */
    0, /* tp_str */
    PyObject_GenericGetAttr, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    0, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    PyObject_SelfIter, /* tp_iter */
    (iternextfunc)PyDataObjectIter_iternext, /* tp_iternext */
    PyDataObjectIter_methods, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)PythonDataObject::PyDataObjectIter_init, /* tp_init */
    0, /* tp_alloc */
    PyDataObjectIter_new,
    /*PyType_GenericNew*/ /*PythonStream_new,*/ /* tp_new */
    0};

} // end namespace ito
