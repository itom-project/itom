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

#include "pythonEngineInc.h"
#include "pythonDataObject.h"
#if (defined linux) | (defined CMAKE)
    #include "structmember.h"
#else
    #include "structmember.h"
#endif
#include "../global.h"

#include "pythonNpDataObject.h"
#include "pythonRgba.h"

#include "pythonQtConversion.h"
#include "dataObjectFuncs.h"

//PyDataObject
using namespace ito;

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
void PythonDataObject::PyDataObject_dealloc(PyDataObject* self)
{
    if (self->dataObject != NULL)
    {
        self->dataObject->lockWrite(); //will be unlocked automatically
        self->dataObject->unlock();
        DELETE_AND_SET_NULL(self->dataObject);
    }

    Py_XDECREF(self->base); //this will free another pyobject (e.g. numpy array), with which this data object shared its data (base != NULL if owndata=0)

    Py_TYPE(self)->tp_free((PyObject*)self);
};

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyDataObject* self = (PyDataObject *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->dataObject = NULL;
        self->base = NULL;
    }

    return (PyObject *)self;
};

//----------------------------------------------------------------------------------------------------------------------------------
//! brief description
/*!
    long description

    \param name description
    \return description
    \sa (see also) keywords (comma-separated)
*/
PyDoc_STRVAR(dataObjectInit_doc,"dataObject([dims [, dtype='uint8'[, continuous = 0][, data = valueOrSequence]]]) -> constructor to get a new dataObject.\n\
\n\
The itom.dataObject represents a multidimensional array of fixed-size items with corresponding meta information (units, axes descriptions, scalings, tags, protocol...). \n\
Recently the following data types (dtype) are supported: \n\
\n\
* Integer-type (int8, uint8, int16, uint16, int32, uint32),\n\
* Floating-type (float32, float64 (=> double)),\n\
* Complex-type  (complex64 (2x float32), complex128 (2x float64)).\n\
* Color-type  (rgba32 (uint32 or uint[4] containing the four 8bit values [R, G, B, Alpha])).\n\
\n\
Arrays can also be constructed using some of the static pre-initialization methods 'zeros', 'ones', 'rand' or 'randN' (refer to the See Also section below). \n\
\n\
Parameters \n\
----------- \n\
dims : {sequence of integers}, optional \n\
    'dims' is a list or tuple indicating the size of each dimension, e.g. [2,3] is a matrix with 2 rows and 3 columns. If not given, an empty data object is created.\n\
dtype : {str}, optional \n\
    'dtype' is the data type of each element, possible values: 'int8','uint8',...,'int32','uint32','float32','float64','complex64','complex128', 'rgba32'\n\
continuous : {str}, optional \n\
    'continuous' [0|1] defines whether the data block should be continuously allocated in memory [1] or in different smaller blocks [0] (recommended for huge matrices).\n\
data : {str}, optional \n\
    'data' is a single value or a sequence with the same amount of values than the data object. The values from data will be assigned to the new data object (filled row by row).\n\
\n\
Notes \n\
------ \n\
\n\
The itom.dataObject is a direct wrapper for the underlying C++ class *dataObject*. This array class mainly is based on the class *Mat* of the computer vision library (OpenCV). \n\
\n\
In order to handle huge matrices, the data object can divide one array into chunks in memory.\n\
Each subpart (called matrix-plane) is two-dimensional and covers data of the last two dimensions.\n\
In c++-context each of these matrix-planes is of type cv::Mat_<type> and can be used with every operator given by the openCV-framework (version 2.3.1 or higher).\n\
\n\
The dimensions of the matrix are structured descending. So if we assume to have a n-dimensional matrix A,\n\
where each dimension has its size s_i, the dimensions order is n, .., z, y, x and the corresponding sizes of A are [s_n, s_(n-1),  s_(n-2), ..., s_y, s_x].\n\
\n\
In order to make the data object compatible to continuously organized data structures, like numpy-arrays, \n\
it is also possible to have all matrix-planes in one data-block in memory (not recommended for huge matrices).\n\
Nevertheless, the indicated data structure with the two-dimensional sub-matrix-planes is still existing. \n\
The data organization is equal to the one of openCV, hence, two-dimensional matrices are stored row-by-row (C-style)...\n\
\n\
In addition to OpenCV, itom.dataObject supports complex valued data types for all operators and methods. \n\
\n\
Warning 'uint32' is not fully openCV-compatible and hence causes instability!\n\
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
Example:\n\
::\n\
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
    \n\
\n\
**Constructor** \n\
\n\
The function dataObject([dims [, dtype='uint8'[, continuous = 0][, data = valueOrSequence]]])\n\
creates a new itom-dataObject filled with undefined data.\n\
If no parameters are given, an uninitilized DataObject (dims = 0, no sizes) is created.\n\
\n\
As second possibility you can also use the copy-constructor 'dataObject(AnyArray)', \n\
where AnyArray must be any array-like structure which is parsable by the numpy-interface.\n\
\n\
See Also \n\
---------- \n\
ones() : Static method to construct a data object filled with ones. \n\
zeros() : Static method to construct a data object filled with zeros. \n\
rand() : Static method to construct a randomly filled data object (uniform distribution). \n\
randN() : Static method to construct a randomly filled data object (gaussian distribution).");
int PythonDataObject::PyDataObject_init(PyDataObject *self, PyObject *args, PyObject *kwds)
{
    Py_ssize_t length = 0;
    Py_ssize_t lengthKwds = 0;
    PyObject *tmp = NULL;
    PyObject* copyObject = NULL;
    PyObject* data = NULL;
    const char *kwlist[] = {"dims", "dtype", "continuous", "data", NULL};
    PyObject *dimList = NULL;
    const char *type = "uint8\0";
    unsigned char continuous = 0;
    unsigned char dimensions = 0;
    Py_ssize_t dims = 0;
    int intDims = 0;
    int typeno = 0;
    int *sizes = NULL;
    int tempSizes = 0;

    RetVal retValue(retOk);
    bool done = false;

    if (args != NULL) length = PyTuple_Size(args);
    if (kwds != NULL) lengthKwds = PyDict_Size(kwds);

    //clear base (if available)
    tmp = self->base;
    self->base = NULL;
    Py_XDECREF(tmp);

    //clear existing dataObject (if exists)
    DELETE_AND_SET_NULL(self->dataObject);

    //1. check for call without arguments
    if ((length + lengthKwds) == 0 && !done)
    {
        DELETE_AND_SET_NULL(self->dataObject);
        self->dataObject = new ito::DataObject();
        self->base = NULL;
        retValue += RetVal(retOk);
        done = true;
    }

    //2.  check for copy constructor of type PyDataObject (same type)
    if (!retValue.containsError()) PyErr_Clear();
    if (!done && PyArg_ParseTuple(args, "O!", &PyDataObjectType, &copyObject))
    {
        PyDataObject* tempObject = (PyDataObject*)(copyObject);
        DELETE_AND_SET_NULL(self->dataObject);
        tempObject->dataObject->lockRead(); //lock
        self->dataObject = new ito::DataObject(*tempObject->dataObject);
        self->dataObject->unlock();
        Py_XINCREF(tempObject->base);
        self->base = tempObject->base;
        tempObject->dataObject->unlock(); //unlock
        retValue += RetVal(retOk);
        done = true;
    }

    if (!retValue.containsError()) PyErr_Clear();

    // temporary fix for numpy-arrays. This used to live in the python engine, but at the moment
    // it doesn't seem to have an effect here, so we need to do it again in this place :-/
 //   if (_import_array() < 0)
    //{
    //    PyErr_Print();
    //    PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    //    return -1;
    //}
    // end fix

    if (!done && PyArg_ParseTuple(args, "O!", &PyArray_Type, &copyObject)) // copyObject is a borrowed reference
    {
        PyArrayObject *ndArray = (PyArrayObject*)copyObject; //reference (from now on, copyObject is only used once when the tags are copied, don't use it for further tasks)
        PyArray_Descr *descr = PyArray_DESCR(ndArray);
        unsigned char dimensions = -1;
        int typeno = -1;
        uchar* data = NULL;

        //at first, check copyObject. there are three cases: 1. we can take it as it is, 2. it is compatible but has to be converted, 3. it is incompatible
        if (! (descr->byteorder == '<' || descr->byteorder == '|' || (descr->byteorder == '=' && NPY_NATBYTE == NPY_LITTLE)))
        {
            retValue += RetVal(retError);
            PyErr_Format(PyExc_TypeError,"Given ndarray or ndDataObject has wrong byteorder (litte endian desired), which cannot be transformed to dataObject");
            done = true;
        }
        else
        {
            //check whether type of ndarray exists for data object
            typeno = parseTypeNumberInverse(descr->kind , PyArray_ITEMSIZE(ndArray));

            if (typeno > -1)
            {
                //verify that ndArray is c-contiguous
                ndArray = PyArray_GETCONTIGUOUS(ndArray); //now we always have an increased reference of ndArray (either referen of old ndArray or new object with new reference)
                if (ndArray == NULL)
                {
                    retValue += RetVal(retError);
                    PyErr_SetString(PyExc_TypeError,"An error occurred while transforming the given ndArray (or ndDataObject) to a c-contiguous array.");
                    done = true;
                }
                else
                {
                    descr = PyArray_DESCR(ndArray);
                    dimensions = PyArray_NDIM(ndArray); //->nd;
                }
            }
            else
            {
                //check whether type is compatible
                int newNumpyTypeNum = getTypenumOfCompatibleType(descr->kind, PyArray_ITEMSIZE(ndArray));
                if (newNumpyTypeNum == -1)
                {
                    retValue += RetVal(retError);
                    PyErr_SetString(PyExc_TypeError,"The data type of the given ndarray (or ndDataObject) is not compatible to any data type provided by dataObject");
                    done = true;
                }
                else
                {
#if (NPY_FEATURE_VERSION < 0x00000007)
                    ndArray = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)ndArray, newNumpyTypeNum, NPY_C_CONTIGUOUS); //now we always have an increased reference of ndArray (either referen of old ndArray or new object with new reference)
#else
                    ndArray = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)ndArray, newNumpyTypeNum, NPY_ARRAY_C_CONTIGUOUS); //now we always have an increased reference of ndArray (either referen of old ndArray or new object with new reference)
#endif
                    if (ndArray == NULL)
                    {
                        retValue += RetVal(retError);
                        PyErr_Format(PyExc_TypeError,"An error occurred while transforming the given ndArray (or ndDataObject) to a c-contiguous array with a compatible type.");
                        done = true;
                    }
                    else
                    {
                        descr = PyArray_DESCR(ndArray);
                        dimensions = PyArray_NDIM(ndArray); //->nd;

                        typeno = parseTypeNumberInverse(descr->kind , PyArray_ITEMSIZE(ndArray));
                        if (typeno == -1)
                        {
                            retValue += RetVal(retError);
                            PyErr_Format(PyExc_TypeError,"While converting the given ndarray or ndDataObject to a compatible data type with respect to data object, an error occurred.");
                            done = true;
                        }
                    }
                }
            }

            if (!retValue.containsError())
            {
                if (dimensions <= 0 || PyArray_SIZE(ndArray) <= 0)
                {
                    DELETE_AND_SET_NULL(self->dataObject);
                    self->dataObject = new ito::DataObject();
                    Py_XDECREF((PyObject*)ndArray);
                    done = true;
                }
                else
                {
                    data = (uchar*)PyArray_DATA(ndArray);
                    npy_intp* npsizes = PyArray_DIMS(ndArray);
                    npy_intp *npsteps = (npy_intp *)PyArray_STRIDES(ndArray); //number of bytes to jump from one element in one dimension to the next one

                    int *sizes = new int[dimensions];
                    int *steps = new int[dimensions];
                    for (int n = 0; n < dimensions; n++)
                    {
                        sizes[n] = npsizes[n];
                        steps[n] = npsteps[n];
                    }

                    //here size of steps is equal to size of sizes, DataObject only requires the first dimensions-1 elements of steps

                    //verify that last dimension has steps size equal to itemsize
                    if(steps[dimensions-1] == PyArray_ITEMSIZE(ndArray))
                    {
                        DELETE_AND_SET_NULL(self->dataObject);
                        self->dataObject = new ito::DataObject(dimensions, sizes, typeno, data, steps);
                    }
                    else
                    {
                        //increase dimension by one and add last dimension with size 1 in order to realize a last step size equal to itemsize
                        dimensions = dimensions + 1;
                        int *sizes_inc = new int[dimensions];
                        int *steps_inc = new int[dimensions];

                        for(uchar i = 0 ; i < dimensions - 1 ; i++)
                        {
                            sizes_inc[i] = sizes[i];
                            steps_inc[i] = steps[i];
                        }
                        sizes_inc[dimensions - 1] = 1;
                        steps_inc[dimensions - 1] = PyArray_ITEMSIZE(ndArray);
                        DELETE_AND_SET_NULL(self->dataObject);
                        self->dataObject = new ito::DataObject(dimensions, sizes_inc, typeno, data, steps_inc);

                        DELETE_AND_SET_NULL_ARRAY(sizes_inc);
                        DELETE_AND_SET_NULL_ARRAY(steps_inc);

                    }

                    DELETE_AND_SET_NULL(sizes);
                    DELETE_AND_SET_NULL(steps);

                    int retCode = copyNpDataObjTags2DataObj(copyObject, self->dataObject);

                    //Py_XINCREF(copyObject); (we don't have to increment reference of ndArray here, since this is already done in the steps above, where the flags c_contiguous and the type is checked)
                    self->base = (PyObject*)ndArray;
                    done = true;

                    if (retCode != 0)
                    {
                        retValue += RetVal(retError);
                        PyErr_Format(PyExc_TypeError,"error while copying tags from npDataObject to this dataObject.");
                    }
                    else
                    {
                        retValue += RetVal(retOk);
                    }
                }
            }
        }
    }

    //3. check for argument: list(int size1, int size2,...,int sizeLast)[, dtype='typename'][, continuous=[0|1]
    if (!retValue.containsError()) PyErr_Clear();
    if (!done && PyArg_ParseTupleAndKeywords(args, kwds, "O|sbO", const_cast<char**>(kwlist), &dimList, &type, &continuous, &data))
    {
        done = true;

        //check if dimList supports sequence protocol
        if (PySequence_Check(dimList))
        {
            typeno = typeNameToNumber(type);
            if (typeno >= 0)
            {
                dims = PySequence_Size(dimList);

                if (dims < 0)
                {
                    retValue += RetVal(retError);
                    PyErr_Format(PyExc_TypeError,"Number of dimensions must be bigger than zero.");
                }
                else if (dims > 255)
                {
                    retValue += RetVal(retError);
                    PyErr_Format(PyExc_TypeError,"Number of dimensions must be lower than 256.");
                }

                intDims = Py_SAFE_DOWNCAST(dims, Py_ssize_t, int);

                if (!retValue.containsError())
                {
                    dimensions = static_cast<unsigned char>(intDims);
                    sizes = new int[intDims];
                    for (int i = 0; i<intDims; i++) sizes[i]=0;

                    int totalElems = 1;
                    PyObject *dimListItem = NULL;

                    //try to parse list to values of unsigned int
                    for (Py_ssize_t i = 0; i < dims; i++)
                    {
                        dimListItem = PySequence_GetItem(dimList,i); //new reference
                        if (!PyArg_Parse(dimListItem , "I" , &tempSizes /*&sizes[i]*/)) //borrowed ref
                        {
                            PyErr_Print();
                            PyErr_Clear();
                            PyErr_Format(PyExc_TypeError,"Element %d of dimension-list is no integer number", i+1);
                            retValue += RetVal(retError);
                            break;
                        }
                        else if (tempSizes <= 0)
                        {
                            PyErr_Format(PyExc_TypeError,"Element %d must be bigger than 1");
                            retValue += RetVal(retError);
                            break;
                        }

                        Py_XDECREF(dimListItem);
                        sizes[i] = tempSizes;
                        totalElems *= tempSizes;
                    }

                    //pre-check data
                    if (data)
                    {
                        if (PySequence_Check(data) && PySequence_Length(data) != totalElems)
                        {
                            PyErr_Format(PyExc_TypeError,"The sequence provided by data must have the same length than the total number of elements of the data object.");
                            retValue += RetVal(retError);
                        }
                        else if (!PySequence_Check(data) && PyFloat_Check(data) == false && PyLong_Check(data) == false && PyComplex_Check(data) == false)
                        {
                            PyErr_Format(PyExc_TypeError,"The single value provided by data must be a numeric type.");
                            retValue += RetVal(retError);
                        }
                    }

                    if (!retValue.containsError())
                    {
                        DELETE_AND_SET_NULL(self->dataObject);
                        try
                        {
                            self->dataObject = new ito::DataObject(dimensions, sizes, typeno, continuous);
                        }
                        catch(cv::Exception exc)
                        {
                            PyErr_Format(PyExc_RuntimeError, "failed to create data object: %s", (exc.err).c_str());
                            self->dataObject = NULL;
                            data = NULL; //this has been a borrowed reference or NULL before, we set it to NULL such that the next if-case is not entered
                            retValue += RetVal(retError);
                        }
                        done = true;

                        if (data)
                        {
                            try
                            {
                                if (PyLong_Check(data))
                                {
                                    *(self->dataObject) = (float64)PyFloat_AsDouble(data);
                                }
                                else if (PyFloat_Check(data))
                                {
                                    *(self->dataObject) = (int32)PyLong_AsLong(data);
                                }
                                else if (PyComplex_Check(data))
                                {
                                    *(self->dataObject) = complex128(PyComplex_RealAsDouble(data), PyComplex_ImagAsDouble(data));
                                }
                                else if (PySequence_Check(data))
                                {
                                    int npTypenum;
                                    switch(typeno)
                                    {
                                    case ito::tInt8:        npTypenum = NPY_BYTE; break;
                                    case ito::tUInt8:       npTypenum = NPY_UBYTE; break;
                                    case ito::tInt16:       npTypenum = NPY_SHORT; break;
                                    case ito::tUInt16:      npTypenum = NPY_USHORT; break;
                                    case ito::tInt32:       npTypenum = NPY_INT; break;
                                    case ito::tUInt32:      npTypenum = NPY_UINT; break;
                                    case ito::tRGBA32:      npTypenum = NPY_UINT; break;
                                    case ito::tFloat32:     npTypenum = NPY_FLOAT; break;
                                    case ito::tFloat64:     npTypenum = NPY_DOUBLE; break;
                                    case ito::tComplex64:   npTypenum = NPY_CFLOAT; break;
                                    case ito::tComplex128:  npTypenum = NPY_CDOUBLE; break;
                                    default: npTypenum = -1;
                                    }

                                    PyObject *npArray = PyArray_ContiguousFromAny(data, npTypenum, 1, 1);

                                    if (npArray == NULL)
                                    {
                                        throw cv::Exception(0, "given data could not entirely transformed to the required data type.","PyDataObject_init",__FILE__,__LINE__);
                                    }
                                    else
                                    {
                                        void *data = PyArray_DATA((PyArrayObject*)npArray);

                                        int numMats = self->dataObject->calcNumMats();
                                        int matIndex = 0;
                                        int c=0;
                                        //PyObject *temp = NULL;
                                        cv::Mat *mat = NULL;
                                        int m,n;

                                        for (int i=0;i<numMats;i++)
                                        {
                                            matIndex = self->dataObject->seekMat(i, numMats);
                                            mat = (cv::Mat*)(self->dataObject->get_mdata())[matIndex];

                                            switch(typeno)
                                            {
                                            case ito::tInt8:
                                                {
                                                    int8 *rowPtr;
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
                                            case ito::tUInt8:
                                                {
                                                    uint8 *rowPtr;
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
                                            case ito::tInt16:
                                                {
                                                    int16 *rowPtr;
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
                                            case ito::tUInt16:
                                                {
                                                    uint16 *rowPtr;
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
                                            case ito::tInt32:
                                                {
                                                    int32 *rowPtr;
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
                                            case ito::tUInt32:
                                                {
                                                    uint32 *rowPtr;
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
                                            case ito::tRGBA32:
                                                {
                                                    ito::Rgba32 *rowPtr;
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
                                            case ito::tFloat32:
                                                {
                                                    float32 *rowPtr;
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
                                            case ito::tFloat64:
                                                {
                                                    float64 *rowPtr;
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
                                            case ito::tComplex64:
                                                {
                                                    complex64 *rowPtr;
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
                                            case ito::tComplex128:
                                                {
                                                    complex128 *rowPtr;
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
                                            }
                                        }
                                    }

                                    Py_XDECREF(npArray);
                                }
                                else
                                {
                                    throw cv::Exception(0, "invalid data value","PyDataObject_init",__FILE__,__LINE__);
                                }
                            }
                            catch(cv::Exception exc)
                            {
                                //PyErr_Format(PyExc_TypeError, "file: %s, line: %d, error: %s", (exc.file).c_str(), exc.line, (exc.err).c_str());
                                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                                delete self->dataObject;
                                self->dataObject = NULL;
                                retValue += RetVal(retError);
                            }
                        }
                    }

                    DELETE_AND_SET_NULL_ARRAY(sizes);
                }
            }
            else
            {
                PyErr_Format(PyExc_TypeError,"dtype name is unknown.");
                retValue += RetVal(retError);
            }
        }
        else
        {
            PyErr_Format(PyExc_TypeError, "dimensions must be of type list(int size1, int size2, ...) or type tuple(int size1, int size2, ...)");
            retValue += RetVal(retError);
        }
    }

    if (!done && retValue.containsError())
    {
        PyErr_Format(PyExc_TypeError,"required arguments: list(int size1, int size2,...,int sizeLast)[, dtype='typename'][, continuous=[0|1]][, data=SequenceOfSingleValue]");
        retValue += RetVal(retError);
    }
    else if (!done && !retValue.containsError())
    {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,"number or arguments are invalid.");
        retValue += RetVal(retError);
    }

    if (retValue.containsError())
    {
        return -1;
    }
    else
    {
        return 0;
    }
};

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::copyNpDataObjTags2DataObj(PyObject* npDataObject, DataObject* dataObj)
{
    if (dataObj == NULL)
    {
        return -1;
    }

    if (npDataObject->ob_type != &PythonNpDataObject::PyNpDataObjectType) //if no npDataObject (numpy-array e.g.) no tags are available, therefore quit here
    {
        return 0;
    }
    PythonNpDataObject::PyNpDataObject* npDO = (PythonNpDataObject::PyNpDataObject*)npDataObject;
    if (npDO == NULL)
    {
        return 0; //no error (same than above)
    }

    PyObject *key, *value;
    PyObject *temp;
    Py_ssize_t pos = 0;
    std::string tempString;
    std::string tempKey;
    int retCode;
    bool error = false;

        //    PyObject* tags;             //PyDictObject
        //PyObject* axisScales;       //PyListObject
        //PyObject* axisOffsets;      //PyListObject
        //PyObject* axisDescriptions; //PyListObject
        //PyObject* axisUnits;        //PyListObject
        //PyObject* valueUnit;        //PyUnicode

    //1. copy tags-dict
    while (PyDict_Next(npDO->tags, &pos, &key, &value))
    {
        retCode = parsePyObject2StdString(key, tempKey);
        if (retCode == 0)
        {
            if (PyFloat_Check(value)||PyLong_Check(value))
            {
                dataObj->setTag(tempKey, PyFloat_AsDouble(value));
            }
            else
            {
                retCode = parsePyObject2StdString(value, tempString);
                if (retCode == 0)
                {
                    dataObj->setTag(tempKey, tempString);
                }
                else
                {
                    error = true;
                }
            }
        }
        else
        {
            error = true;
        }
    }

    //2. copy axisScales
    if (PySequence_Check(npDO->axisScales))
    {
        for (Py_ssize_t i = 0; i<PySequence_Size(npDO->axisScales); i++)
        {
            temp = PySequence_GetItem(npDO->axisScales,i); //new reference
            if (PyFloat_Check(temp))
            {
                dataObj->setAxisScale(i, PyFloat_AsDouble(temp));
            }
            else
            {
                error = true;
            }
            Py_XDECREF(temp);
        }
    }

    //3. copy axisOffsets
    if (PySequence_Check(npDO->axisOffsets))
    {
        for (Py_ssize_t i = 0; i<PySequence_Size(npDO->axisOffsets); i++)
        {
            temp = PySequence_GetItem(npDO->axisOffsets,i); //new reference
            if (PyFloat_Check(temp))
            {
                dataObj->setAxisOffset(i, PyFloat_AsDouble(temp));
            }
            else
            {
                error = true;
            }
            Py_XDECREF(temp);
        }
    }

    //4. copy axisDescriptions
    if (PySequence_Check(npDO->axisDescriptions))
    {
        for (Py_ssize_t i = 0; i<PySequence_Size(npDO->axisDescriptions); i++)
        {
            temp = PySequence_GetItem(npDO->axisDescriptions,i); //new reference
            retCode = parsePyObject2StdString(temp, tempString);
            if (retCode == 0)
            {
                dataObj->setAxisDescription(i, tempString);
            }
            else
            {
                error = true;
            }
            Py_XDECREF(temp);
        }
    }

    //5. copy axisUnits
    if (PySequence_Check(npDO->axisUnits))
    {
        for (Py_ssize_t i = 0; i<PySequence_Size(npDO->axisUnits); i++)
        {
            temp = PySequence_GetItem(npDO->axisUnits,i); //new referene
            retCode = parsePyObject2StdString(temp, tempString);
            if (retCode == 0)
            {
                dataObj->setAxisUnit(i, tempString);
            }
            else
            {
                error = true;
            }
            Py_XDECREF(temp);
        }
    }

    //6. copy valueUnit
    retCode = parsePyObject2StdString(npDO->valueUnit, tempString);
    if (retCode == 0)
    {
        dataObj->setValueUnit(tempString);
    }
    else
    {
        error = true;
    }

    //7. copy valueDescription
    retCode = parsePyObject2StdString(npDO->valueDescription, tempString);
    if (retCode == 0)
    {    //Beim Dokumentieren ausversehen verändert. Stimmt die folgende Zeile???
        dataObj->setValueDescription(tempString);
    }
    else
    {
        error = true;
    }

    //8. valueOffset (ignored)
    //9. valueScale (ignored)
    return error ? -1 : 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::parsePyObject2StdString(PyObject* pyObj, std::string &str)
{
    PyObject* temp = NULL;

    if (pyObj == NULL) return -1;
    if (PyBytes_Check(pyObj))
    {
        str = PyBytes_AsString(pyObj);
    }
    else if (PyUnicode_Check(pyObj))
    {
        bool ok = false;
        str = PythonQtConversion::PyObjGetBytes(pyObj, false, ok).data(); //PyBytes_AsString(temp2);
        //Py_XDECREF(temp);
        if (!ok) 
        {
            PyErr_Clear();
            return -1;
        }
/*
        temp = PyUnicode_AsASCIIString(pyObj);
        if (temp == NULL)
        {
            PyErr_Clear();
            return -1;
        }
        str = PyBytes_AsString(temp);
        Py_XDECREF(temp);*/
    }
    else
    {
        temp = PyObject_Str(pyObj);
        if (temp == NULL)
        {
            PyErr_Clear();
            return -1;
        }
        /*temp2 = PyUnicode_AsASCIIString(temp);
        Py_XDECREF(temp);
        if (temp2 == NULL)
        {
            PyErr_Clear();
            return -1;
        }*/

        bool ok = false;
        str = PythonQtConversion::PyObjGetBytes(temp,false,ok).data(); //PyBytes_AsString(temp2);
        Py_XDECREF(temp);
        if (!ok) 
        {
            PyErr_Clear();
            return -1;
        }
    }
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PythonDataObject::PyDataObjectTypes PythonDataObject::PyDataObject_types[] = {
    {"int8", tInt8},
    {"uint8", tUInt8},
    {"int16", tInt16},
    {"uint16" ,tUInt16},
    {"int32", tInt32},
    {"uint32", tUInt32},
    {"float32", tFloat32},
    {"float64", tFloat64},
    {"complex64", tComplex64},
    {"complex128", tComplex128},
    {"rgba32", tRGBA32}
};

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::typeNameToNumber(const char *name)
{
    int length = sizeof(PyDataObject_types) / sizeof(PyDataObject_types[0]);
    int i;

    for (i=0; i<length; i++)
    {
        if (!strcmp(name, PyDataObject_types[i].name))
        {
            return PyDataObject_types[i].typeno;
        }
    }
    return -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
char * PythonDataObject::typeNumberToName(int typeno)
{
    int length = sizeof(PyDataObject_types) / sizeof(PyDataObject_types[0]);

    if (typeno < 0 || typeno >= length)
    {
        return 0;
    }
    else
    {
        return PyDataObject_types[typeno].name;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal PythonDataObject::PyDataObj_ParseCreateArgs(PyObject *args, PyObject *kwds, int &typeno, std::vector<unsigned int> &sizes, unsigned char &continuous)
{
    static const char *kwlist[] = {"dims","dtype","continuous", NULL};
    PyObject *dimList = NULL;
    const char *type;
    unsigned int dims = 0;
    int dimensions = 0;
    int tempSizes = 0;

    type = typeNumberToName(typeno);
    if (strlen(type)<4) type="uint8\0"; //default

    RetVal retValue(retOk);

    //3. check for argument: list(int size1, int size2,...,int sizeLast)[, dtype='typename'][, continuous=[0|1]
    PyErr_Clear();
    if (PyArg_ParseTupleAndKeywords(args, kwds, "O!|sb", const_cast<char**>(kwlist), &PyList_Type, &dimList, &type, &continuous))
    {
        typeno = typeNameToNumber(type);
        if (typeno >= 0)
        {
            dims = PyList_Size(dimList);

            if (dims < 0)
            {
                retValue += RetVal(retError);
                PyErr_Format(PyExc_TypeError,"Number of dimensions must be bigger than zero.");
            }
            else if (dims > 255)
            {
                retValue += RetVal(retError);
                PyErr_Format(PyExc_TypeError,"Number of dimensions must be lower than 256.");
            }

            if (!retValue.containsError())
            {
                dimensions = static_cast<unsigned char>(dims);
                sizes.clear();
                sizes.resize(dims);

                //try to parse list to values of unsigned int
                for (unsigned int i = 0; i < dims; i++)
                {
                    if (!PyArg_Parse(PyList_GetItem(dimList,i) , "I" , &tempSizes)) //borrowed ref
                    {
                        PyErr_Print();
                        PyErr_Clear();
                        PyErr_Format(PyExc_TypeError,"Element %d of dimension-list is no integer number", i+1);
                        retValue += RetVal(retError);
                        break;

                    }
                    else if (tempSizes <= 0)
                    {
                        PyErr_Format(PyExc_TypeError,"Element %d must be bigger than 1");
                        retValue += RetVal(retError);
                        break;
                    }

                    sizes[i] = tempSizes;
                }
            }
        }
        else
        {
            PyErr_Format(PyExc_TypeError,"dtype name is unknown.");
            retValue += RetVal(retError);
        }
    }
    else
    {
        PyErr_Format(PyExc_TypeError,"required arguments: list(int size1, int size2,...,int sizeLast)[, dtype='typename'][, continuous=[0|1]");
        retValue += RetVal(retError);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrDims_doc,"number of dimensions of this data object\n\
\n\
Notes \n\
----- \n\
read-only property, this property is readable both by the attributes ndim and dims. \n\
");
PyObject* PythonDataObject::PyDataObj_GetDims(PyDataObject *self, void * /*closure*/)
{
    if (self->dataObject == NULL)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("i",self->dataObject->getDims());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrType_doc,"get type string of data in this data object \n\
\n\
This type string has one of these values: 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', \n\
'float32', 'float64', 'complex64', 'complex128', 'rgba32'\n\
\n\
Notes \n\
----- \n\
This attribute is read-only");
PyObject* PythonDataObject::PyDataObj_GetType(PyDataObject *self, void * /*closure*/)
{
    if (self->dataObject == NULL)
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s",typeNumberToName(self->dataObject->getType()));
    }

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrContinuous_doc,"true if matrix is continuously organized, else false. \n\
\n\
If true, the whole matrix is allocated in one huge block in memory, hence, \n\
this data object can be transformed into a numpy representation (npDataObject) \n\
without reallocating memory.\n\
\n\
Notes \n\
----- \n\
read-only\n\
");
PyObject* PythonDataObject::PyDataObj_GetContinuous(PyDataObject *self, void * /*closure*/)
{
    if (self->dataObject == NULL)
    {
        Py_RETURN_NONE;
    }
    else
    {
        self->dataObject->lockRead();
        bool cont = self->dataObject->getContinuous();
        self->dataObject->unlock();

        if (cont)
        {
            Py_INCREF(Py_True);
            return Py_True;
        }
        else
        {
            Py_INCREF(Py_False);
            return Py_False;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrShape_doc,"tuple with the sizes of each dimension / axis of this data object. \n\
\n\
Notes\n\
------\n\
In difference to the shape attribute of numpy arrays, no new shape tuple can be assigned to \
this value (used to 'reshape' the array). Read-only.\n\
\n\
See Also \n\
--------- \n\
size() : Alternative method to return the size of all or any specific axis");
PyObject* PythonDataObject::PyDataObj_GetShape(PyDataObject *self, void * /*closure*/)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    self->dataObject->lockRead();

    int dims = self->dataObject->getDims();
    PyObject* retList = NULL;
    int desiredDim = 0;

    retList = PyTuple_New(dims); //new reference

    for (int i = 0; i < dims; i++)
    {
        PyTuple_SetItem(retList, i, PyLong_FromLong(self->dataObject->getSize(i)));
    }

    self->dataObject->unlock();
    return retList;
}

//----------------------------------------------------------------------------------------------------------------------------------

//---------------------------------------Get / Set metadata / objecttags-----------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrTags_doc,  "tag dictionary of this data object. \n\
\n\
This attribute returns a dict_proxy object of the tag dictionary of this data object. This object is read-only. \n\
However you can assign an entire new dictionary to this attribute that fully replaces the old tag dictionary. \n\
The tag dictionary can contain arbitrary pairs of key -> value where value is either a string or a double value. \n\
\n\
Special tags are the key 'protocol' that contains the newline-separated protocol string of the data object (see: addToProtocol()) \n\
or the key 'title' that can for instance be used as title in any plots. \n\
\n\
You can add single elements using the method setTag(key,value) or you can delete tags using deleteTag(key).\n\
\n\
Do NOT use 'special character' within the tag key because they are not XML-save.\n\
\n\
Notes \n\
----- \n\
read-only / write only for fully new dictionary");
//getter and setter methods
PyObject* PythonDataObject::PyDataObject_getTags(PyDataObject *self, void * /*closure*/)
{
    PyObject* ret = PyDict_New();
    int size = self->dataObject->getTagListSize();
    bool valid;
    std::string key;
    //std::string value;
    DataObjectTagType value;

    for (int i = 0; i < size; i++)
    {
        valid = self->dataObject->getTagByIndex(i, key, value);
        if (valid)
        {
            //PyDict_SetItemString(ret, key.data(), PyUnicode_FromString(value.data()));
            if (value.getType() == DataObjectTagType::typeDouble)
            {
                PyObject *item = PyFloat_FromDouble(value.getVal_ToDouble());
                PyDict_SetItemString(ret, key.data(), item);
                Py_DECREF(item);
            }
            else
            {
                PyObject *text = PythonQtConversion::ByteArrayToPyUnicode(value.getVal_ToString().data());
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

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setTags(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (! PyDict_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "The tag attribute must be a dictionary");
        return -1;
    }

    self->dataObject->deleteAllTags();

    PyObject *key;
    PyObject *content;
    std::string keyString, contentString;
    Py_ssize_t pos = 0;

    while (PyDict_Next(value, &pos, &key, &content))
    {
        if (parsePyObject2StdString(key, keyString) >= 0)
        {
            if (PyFloat_Check(value)||PyLong_Check(value))
            {
                self->dataObject->setTag(keyString, PyFloat_AsDouble(value));
            }
            else if (parsePyObject2StdString(content, contentString) >= 0)
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

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrAxisScales_doc, "tuple containing the axis scales [unit/px]. \n\
\n\
This attribute gives access to the internal axis scales [unit/px] expressed as \n\
a tuple of double values. The i-th value in the tuple corresponds to the scaling factor of the i-th \n\
axis. Either assign a new tuple with the same length than the number of dimensions or change single values \n\
using tuple indexing. \n\
\n\
Definition: Physical unit = (px-Coordinate - offset)* scale\n\
\n\
Notes \n\
----- \n\
read / write\n\
\n\
See Also \n\
--------- \n\
setAxisScale() : Alternative method to set the scale value of one single axis");
PyObject* PythonDataObject::PyDataObject_getAxisScales(PyDataObject *self, void * /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    PyObject *ret = PyTuple_New(dims); //must be tuple, such that items cannot be changed, since this tuple is no reference but deep copy to the real tags in self->dataObject
    double temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisScale(i);
        PyTuple_SetItem(ret, i,PyFloat_FromDouble(temp)); //steals reference
    }

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisScales(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    double scale;
    PyObject *tempObj;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis scales must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis scale sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        tempObj = PySequence_GetItem(value,i); //new reference
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

        self->dataObject->setAxisScale(i, scale);
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrAxisOffsets_doc, "tuple containing the axis offsets [px]. \n\
\n\
This attribute gives access to the internal axis offsets [px] expressed as \n\
a tuple of double values. The i-th value in the tuple corresponds to the pixel-offset of the i-th \n\
axis. Either assign a new tuple with the same length than the number of dimensions or change single values \n\
using tuple indexing. \n\
\n\
Definition: Physical unit = (px-Coordinate - offset)* scale\n\
\n\
Notes \n\
----- \n\
read / write\n\
\n\
See Also \n\
--------- \n\
setAxisOffset() : Alternative method to set the offset value of one single axis");
PyObject* PythonDataObject::PyDataObject_getAxisOffsets(PyDataObject *self, void * /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    PyObject *ret = PyTuple_New(dims); //must be tuple, such that items cannot be changed, since this tuple is no reference but deep copy to the real tags in self->dataObject
    double temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisOffset(i);
        PyTuple_SetItem(ret, i,PyFloat_FromDouble(temp));
    }

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisOffsets(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    double offset;
    PyObject *tempObj;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis offsets must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis offset sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        tempObj = PySequence_GetItem(value,i); //new reference
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

        self->dataObject->setAxisOffset(i, offset);
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrAxisDescriptions_doc, "tuple containing the axis descriptions {str}. \n\
\n\
This attribute gives access to the internal axis descriptions expressed as \n\
a tuple of strings. The tuple has the same length than the number of dimensions of this data object. \n\
\n\
You can either assign a new tuple with the same length or change single values using tuple indexing. \n\
\n\
See Also \n\
--------- \n\
setAxisDescriptions : alternative method to change the description string of one single axis \n\
\n\
Notes \n\
------- \n\
read / write");
PyObject* PythonDataObject::PyDataObject_getAxisDescriptions(PyDataObject *self, void * /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    PyObject *ret = PyTuple_New(dims); //must be tuple, such that items cannot be changed, since this tuple is no reference but deep copy to the real tags in self->dataObject
    bool valid;
    std::string temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisDescription(i, valid);
        if (valid)
        {
            PyObject *string = PyUnicode_FromString(temp.data());
            if (string == NULL)
            {
                string = PyUnicode_FromString("<encoding error>"); //TODO
            }
            PyTuple_SetItem(ret, i, string); //steals reference from string
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "error while reading axis units from data object");
            return NULL;
        }
    }

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisDescriptions(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    std::string tempString;
    PyObject *seqItem = NULL;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis descriptions must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis description sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        seqItem = PySequence_GetItem(value,i); //new reference
        if (PythonDataObject::parsePyObject2StdString(PySequence_GetItem(value,i), tempString) == -1)
        {
            Py_XDECREF(seqItem);
            PyErr_SetString(PyExc_TypeError, "elements of axis description vector must be a string");
            return -1;
        }
        Py_XDECREF(seqItem);
        self->dataObject->setAxisDescription(i, tempString);
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrAxisUnits_doc, "tuple containing the axis units {str}. \n\
\n\
This attribute gives access to the internal axis units expressed as \n\
a tuple of strings. The tuple has the same length than the number of dimensions of this data object. \n\
\n\
You can either assign a new tuple with the same length or change single values using tuple indexing. \n\
\n\
See Also \n\
--------- \n\
setAxisUnits : alternative method to change the unit string of one single axis \n\
\n\
Notes \n\
------- \n\
read / write");
PyObject* PythonDataObject::PyDataObject_getAxisUnits(PyDataObject *self, void * /*closure*/)
{
    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    PyObject *ret = PyTuple_New(dims); //must be tuple, such that items cannot be changed, since this tuple is no reference but deep copy to the real tags in self->dataObject
    bool valid;
    std::string temp;
    for (Py_ssize_t i = 0; i < dims; i++)
    {
        temp = self->dataObject->getAxisUnit(i, valid);
        if (valid)
        {
            PyTuple_SetItem(ret, i, PyUnicode_FromString(temp.data()));
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "error while reading axis units from data object");
            return NULL;
        }
    }

    return ret;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setAxisUnits(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    std::string tempString;
    PyObject *seqItem = NULL;

    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    Py_ssize_t dims = static_cast<Py_ssize_t>(self->dataObject->getDims());
    if (dims == 1) dims = 2;

    if (!PySequence_Check(value))
    {
        PyErr_SetString(PyExc_TypeError, "axis units must be a sequence");
        return -1;
    }
    if (PySequence_Size(value) != dims)
    {
        PyErr_SetString(PyExc_TypeError, "length of axis unit sequence must be equal to number of dimensions");
        return -1;
    }

    for (Py_ssize_t i = 0; i < dims; i++)
    {
        seqItem = PySequence_GetItem(value,i); //new reference
        if (PythonDataObject::parsePyObject2StdString(seqItem, tempString) == -1)
        {
            Py_XDECREF(seqItem);
            PyErr_SetString(PyExc_TypeError, "elements of axis unit vector must be a string");
            return -1;
        }
        Py_XDECREF(seqItem);
        self->dataObject->setAxisUnit(i, tempString);
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrValueUnit_doc, "value unit. \n\
\n\
Attribute to read or write the unit string of the values in this data object. \n\
\n\
Notes \n\
----- \n\
read / write");
PyObject* PythonDataObject::PyDataObject_getValueUnit(PyDataObject *self, void * /*closure*/)
{
    return PyUnicode_FromString(self->dataObject->getValueUnit().data());
}

int PythonDataObject::PyDataObject_setValueUnit(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    std::string unit;
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (PythonDataObject::parsePyObject2StdString(value, unit) == -1)
    {
        PyErr_SetString(PyExc_TypeError, "The value unit must be a string");
        return -1;
    }

    if (self->dataObject->setValueUnit(unit))
    {
        PyErr_Format(PyExc_RuntimeError, "set value unit failed");
        return -1;
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrValueDescription_doc, "value unit description. \n\
\n\
Attribute to read or write the unit description string of the values in this data object. \n\
\n\
Notes \n\
----- \n\
read / write");
PyObject* PythonDataObject::PyDataObject_getValueDescription(PyDataObject *self, void * /*closure*/)
{
    PyObject *temp = PyUnicode_FromString(self->dataObject->getValueDescription().data());
    if (temp)
    {
        return temp;
    }
    return PyUnicode_FromString("<encoding error>"); //TODO
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObject_setValueDescription(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    std::string unit;
    if (value == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Cannot delete this attribute");
        return -1;
    }

    if (PythonDataObject::parsePyObject2StdString(value, unit) == -1)
    {
        PyErr_SetString(PyExc_TypeError, "The value unit must be a string");
        return -1;
    }

    if (self->dataObject->setValueDescription(unit))
    {
        PyErr_Format(PyExc_RuntimeError, "set value unit failed");
        return -1;
    }

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrValueScale_doc, "value scale [default: 0.0].\n\
\n\
This attribute gives the scaling factor of each value in the data object. This value is always 1.0. \n\
\n\
Notes \n\
----- \n\
This attribute is read only");
PyObject* PythonDataObject::PyDataObject_getValueScale(PyDataObject *self, void * /*closure*/)
{
    return PyFloat_FromDouble(self->dataObject->getValueScale());
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrValueOffset_doc, "value offset [default: 0.0].\n\
\n\
This attribute gives the offset of each value in the data object. This value is always 0.0. \n\
\n\
Notes \n\
----- \n\
This attribute is read only");
PyObject* PythonDataObject::PyDataObject_getValueOffset(PyDataObject *self, void * /*closure*/)
{
    return PyFloat_FromDouble(self->dataObject->getValueOffset());
}

//---------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrValue_doc, "get/set the values within the ROI as a one-dimensional tuple.\n\
\n\
This method gets or sets the values within the ROI. If this attribute is called by means of a getter, \n\
a tuple is returned which is created by iterating through the values of the data object (row-wise). \n\
In the same way of iterating, the values are set to the data object if you provide a tuple of the size of the data object \n\
or its ROI, respectively. \n\
\n\
Example: ::\n\
\n\
    b = dataObject[1,1:10,1,1].value\n\
    # or for the first value \n\
    b = dataObject[1,1:10,1,1].value[0]\n\
    # The elements of the tuple are adressed with b[idx].");
PyObject* PythonDataObject::PyDataObject_getValue(PyDataObject *self, void * /*closure*/)
{
    PyObject *OutputTuple = NULL;

    int dims = self->dataObject->getDims();

    if (dims == 0)
    {
        return OutputTuple = PyTuple_New(0);
    }

    OutputTuple = PyTuple_New(self->dataObject->getTotal());

    ito::DObjConstIterator it = self->dataObject->constBegin();
    ito::DObjConstIterator itEnd = self->dataObject->constEnd();
    Py_ssize_t cnt = 0;

    switch (self->dataObject->getType())
    {
        case ito::tInt8:
            for (; it < itEnd; ++it)
            {
                PyTuple_SetItem(OutputTuple, cnt++, PyLong_FromLong((long)(*((ito::int8*)(*it)))));
            }
            break;
        case ito::tUInt8:
            for (; it < itEnd; ++it)
            {
                PyTuple_SetItem(OutputTuple, cnt++, PyLong_FromLong((long)(*((ito::uint8*)(*it)))));
            }
            break;
        case ito::tInt16:
            for (; it < itEnd; ++it)
            {
                PyTuple_SetItem(OutputTuple, cnt++, PyLong_FromLong((long)(*((ito::int16*)(*it)))));
            }
            break;
        case ito::tUInt16:
            for (; it < itEnd; ++it)
            {
                PyTuple_SetItem(OutputTuple, cnt++, PyLong_FromLong((long)(*((ito::uint16*)(*it)))));
            }
            break;
        case ito::tInt32:
            for (; it < itEnd; ++it)
            {
                PyTuple_SetItem(OutputTuple, cnt++, PyLong_FromLong((long)(*((ito::int32*)(*it)))));
            }
            break;
        case ito::tRGBA32:
            {
                ito::PythonRgba::PyRgba *color;
                for (; it < itEnd; ++it)
                {
                    color = ito::PythonRgba::createEmptyPyRgba();
                    if (color)
                    {
                        color->rgba = ((ito::Rgba32*)(*it))->rgba;
                        PyTuple_SetItem(OutputTuple, cnt++, (PyObject*)color);
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
                PyTuple_SetItem(OutputTuple, cnt++, PyFloat_FromDouble((double)(*((ito::float32*)(*it)))));
            }
            break;
        case ito::tFloat64:
            for (; it < itEnd; ++it)
            {
                PyTuple_SetItem(OutputTuple, cnt++, PyFloat_FromDouble((double)(*((ito::float64*)(*it)))));
            }
            break;
        case ito::tComplex64:
        {
            complex64 *value;
            for (; it < itEnd; ++it)
            {
                value = (complex64*)(*it);
                PyTuple_SetItem(OutputTuple, cnt++, PyComplex_FromDoubles((double)value->real(),(double)value->imag()));
            }
            break;
        }
        case ito::tComplex128:
        {
            complex128 *value;
            for (; it < itEnd; ++it)
            {
                value = (complex128*)(*it);
                PyTuple_SetItem(OutputTuple, cnt++, PyComplex_FromDoubles((double)value->real(),(double)value->imag()));
            }
            break;
        }
        default:
            Py_XDECREF(OutputTuple);
            PyErr_Format(PyExc_NotImplementedError, "Type not implemented yet");
            return NULL;
    }

    return OutputTuple;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ int PythonDataObject::PyDataObject_setValue(PyDataObject *self, PyObject *value, void *closure)
{
    if (self->dataObject == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "dataObject is NULL");
        return -1;
    }
    
    int total = self->dataObject->getTotal();
    int typenum;

    switch(self->dataObject->getType())
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
        PyErr_Format(PyExc_RuntimeError, "type of dataObject is unknown.");
        return -1;
    }

    //try to convert value to a numpy-array
    PyObject* arr = PyArray_FromObject(value, typenum, 1, 1);  //new ref

    if (arr == NULL)
    {
        return -1;
    }

    if (PyArray_DIM(arr, 0) != total)
    {
        Py_DECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "The given array-like object (array, tuple, list...) must have a length of %i", total);
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
        case ito::tComplex64:
        {
            for (; it < itEnd; ++it)
            {
                *((ito::complex64*)(*it)) = *((ito::complex64*)(PyArray_GETPTR1(arr, cnt++)));
            }
            break;
        }
        case ito::tComplex128:
        {
            for (; it < itEnd; ++it)
            {
                *((ito::complex128*)(*it)) = *((ito::complex128*)(PyArray_GETPTR1(arr, cnt++)));
            }
            break;
        }
        default:
            Py_XDECREF(arr);
            PyErr_Format(PyExc_NotImplementedError, "Type not implemented yet");
            return -1;
    }

    Py_XDECREF(arr);

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrRotationalMatrix_doc, "Access the 3x3 rotational matrix in the dataObject tagspace \n\
\n\
This attribute gives access to the xyRotationalMatrix in the metaData-Tagspace.\n\
The getter method retuns a 3x3-Array deep copied from the internal matrix,\n\
Implemented to offer compability to x3p format.\n\
\n\
Notes \n\
----- \n\
{3x3 array of doubles} : ReadWrite\n\
");
int PythonDataObject::PyDataObject_setXYRotationalMatrix(PyDataObject *self, PyObject *value, void * /*closure*/)
{
    if (self == NULL)
    {
        PyErr_Format(PyExc_ValueError, "dataObject is NULL");
        return -1;
    }

    if (self->dataObject == NULL)
    {
        PyErr_Format(PyExc_ValueError, "content of dataObject is NULL");
        return -1;
    }

    DataObject *dObj = self->dataObject;

    if (PyList_Size(value) != 3)
    {
        PyErr_Format(PyExc_ValueError, "row number do not match, matrix must be 3x3");
        return -1;
    }

    double ryx[9] = {0,0,0,0,0,0,0,0,0};

    for (int i = 0; i < 3; i++)
    {
        PyObject *slice = PyList_GetItem(value, i);

        if (PyList_Size(slice) != 3)
        {
            PyErr_Format(PyExc_ValueError, "col number do not match, matrix must be 3x3");
            return -1;
        }

        ryx[i*3 + 0] = PyFloat_AsDouble(PyList_GetItem(slice, 0));
        ryx[i*3 + 1] = PyFloat_AsDouble(PyList_GetItem(slice, 1));
        ryx[i*3 + 2] = PyFloat_AsDouble(PyList_GetItem(slice, 2));
    }

    dObj->setXYRotationalMatrix(ryx[0], ryx[1], ryx[2], ryx[3], ryx[4], ryx[5], ryx[6], ryx[7], ryx[8]);

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_getXYRotationalMatrix(PyDataObject *self, void * /*closure*/)
{
    if (self == NULL)
    {
        PyErr_Format(PyExc_ValueError, "dataObject is NULL");
        return NULL;
    }

    if (self->dataObject == NULL)
    {
        PyErr_Format(PyExc_ValueError, "content of dataObject is NULL");
        return NULL;
    }

    PyObject *matrix = PyList_New(3);

    DataObject *dObj = self->dataObject;

    double ryx[9] = {0,0,0,0,0,0,0,0,0};

    dObj->getXYRotationalMatrix(ryx[0], ryx[1], ryx[2], ryx[3], ryx[4], ryx[5], ryx[6], ryx[7], ryx[8]);

    PyObject *slice0 = PyList_New(3);
    PyObject *slice1 = PyList_New(3);
    PyObject *slice2 = PyList_New(3);
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisOffset_doc,"setAxisOffset(axisNum, axisOffset) -> Set the offset of the specified axis. \n\
\n\
Each axis in the data object can get a specific scale value, described in axisUnits per pixel. Use this method to set the scale of one specific axis. \n\
The value of each pixel in its physical unit is the (px-Coordinate - axisOffset) * axisScale \n\
\n\
Parameters  \n\
------------\n\
axisNum : {int}\n\
    The addressed axis index\n\
axisOffset : {double}\n\
    New axis offset in [px]\n\
\n\
Raises \n\
------- \n\
Runtime error : \n\
    if the given axisNum is invalid (out of range) \n\
\n\
See Also \n\
--------- \n\
axisOffsets : this attribute can directly be used to read/write the axis offset(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisOffset(PyDataObject *self, PyObject *args)
{
    int length = PyTuple_Size(args);
    int axisnum;
    double axisOffset;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }
    if (length < 2)
    {
        PyErr_Format(PyExc_TypeError, "inputparameters are (int) axisnumber and (double) axis scale");
        return NULL;
    }
    else if (length == 2)
    {
        if (!PyArg_ParseTuple(args, "id", &axisnum, &axisOffset))
        {
            PyErr_Format(PyExc_TypeError, "inputparameters are (int) axisnumber and (double) axis scale");
            return NULL;
        }
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "to many input parameters specified");
        return NULL;
    }
    if (self->dataObject->setAxisOffset(axisnum, axisOffset))
    {
        PyErr_Format(PyExc_RuntimeError, "Set axisoffset failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisScale_doc,"setAxisScale(axisNum, axisScale) -> Set the scale value of the specified axis. \n\
\n\
Each axis in the data object can get a specific scale value, described in axisUnits per pixel. Use this method to set the scale of one specific axis. \n\
\n\
Parameters  \n\
------------\n\
axisNum : {int}\n\
    The addressed axis index\n\
axisScale : {double}\n\
    New axis scale in axisUnit/px\n\
\n\
Raises \n\
------- \n\
Runtime error : \n\
    if the given axisNum is invalid (out of range) \n\
\n\
See Also \n\
--------- \n\
axisScales : this attribute can directly be used to read/write the axis scale(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisScale(PyDataObject *self, PyObject *args)
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
        PyErr_Format(PyExc_ValueError, "inputparameters are (int) axisnumber and (double) axis scale");
        return NULL;
    }

    if (self->dataObject->setAxisScale(axisnum, axisscale))
    {
        PyErr_Format(PyExc_RuntimeError, "Set axis scale failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisDescription_doc,"setAxisDescription(axisNum, axisDescription) -> Set the description of the specified axis. \n\
\n\
Each axis in the data object can get a specific axisDescription string (e.g. mm). Use this method to set the axisDescription of one specific axis. \n\
\n\
Parameters  \n\
------------\n\
axisNum : {int}\n\
    The addressed axis index\n\
axisDescription : {str}\n\
    New axis description\n\
\n\
Raises \n\
------- \n\
Runtime error : \n\
    if the given axisNum is invalid (out of range) \n\
\n\
See Also \n\
--------- \n\
axisDescriptions : this attribute can directly be used to read/write the axis description(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisDescription(PyDataObject *self, PyObject *args)
{
    int axisNum = 0;
    const char *tagvalue = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "is", &axisNum, &tagvalue))
    {
        PyErr_Format(PyExc_ValueError, "Inputarguments are axisnumber and  value description");
        return NULL;
    }

    std::string tagValString(tagvalue);
    if (self->dataObject->setAxisDescription(axisNum, tagValString))
    {
        PyErr_Format(PyExc_RuntimeError, "set axis description failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetAxisUnit_doc,"setAxisUnit(axisNum, axisUnit) -> Set the unit of the specified axis. \n\
\n\
Each axis in the data object can get a specific unit string (e.g. mm). Use this method to set the unit of one specific axis. \n\
\n\
Parameters  \n\
------------\n\
axisNum : {int}\n\
    The addressed axis index\n\
axisUnit : {str}\n\
    New axis unit\n\
\n\
Raises \n\
------- \n\
Runtime error : \n\
    if the given axisNum is invalid (out of range) \n\
\n\
See Also \n\
--------- \n\
axisUnits : this attribute can directly be used to read/write the axis unit(s) of single or all axes");
PyObject* PythonDataObject::PyDataObj_SetAxisUnit(PyDataObject *self, PyObject *args)
{
    int axisNum = 0;
    const char *tagvalue = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "is", &axisNum, &tagvalue))
    {
        return NULL;
    }

    std::string tagValString(tagvalue);
    if (self->dataObject->setAxisUnit(axisNum, tagValString))
    {
        PyErr_Format(PyExc_RuntimeError, "set axis unit failed");
    }
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSetTag_doc,"setTag(key, tagvalue) -> Set the value of tag specified by key. \n\
\n\
Sets the value of an existing tag (defined by key) in the tag dictionary to the string or double tagvalue or \
adds a new item with key. \n\
\n\
Parameters  \n\
------------\n\
key : {str}\n\
    the name of the tag to set\n\
tagvalue : {str or double}\n\
    the new value of the tag, either string or double value\n\
\n\
Notes \n\
----- \n\
Do NOT use 'special character' within the tag key because they are not XML-save.\n\
");
PyObject* PythonDataObject::PyDataObj_SetTag(PyDataObject *self, PyObject *args)
{
    const char *tagName = NULL;
    const char *tagvalue = NULL;
    double tagvalueD = 0;
    bool dType = true;

    if (!PyArg_ParseTuple(args, "sd", &tagName, &tagvalueD))
    {
        PyErr_Clear();
        dType = false;
        if (!PyArg_ParseTuple(args, "ss", &tagName, &tagvalue))
        {
            PyErr_Format(PyExc_ValueError, "input must be key (string) of tag and new tagvalue (str or double)");
            return NULL;
        }
    }

    std::string tagNameString(tagName);
    if (dType)
    {
        if (self->dataObject->setTag(tagNameString, tagvalueD))
        {
            PyErr_Format(PyExc_RuntimeError, "set tag value as double failed");
            return NULL;
        }
    }
    else
    {
        std::string tagValString(tagvalue);
        if (self->dataObject->setTag(tagNameString, tagValString))
        {
            PyErr_Format(PyExc_RuntimeError, "set tag value string failed");
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectDeleteTag_doc,"deleteTag(key) -> Delete a tag specified by key from the tag dictionary. \n\
\n\
Checks whether a tag with the given key exists in the tag dictionary and if so deletes it. \n\
\n\
Parameters  \n\
------------\n\
key : {str}\n\
    the name of the tag to be deleted\n\
\n\
Returns \n\
-------- \n\
success : {bool}: \n\
    True if tag with given key existed and could be deleted, else False");
PyObject* PythonDataObject::PyDataObj_DeleteTag(PyDataObject *self, PyObject *args)
{
    //int length = PyTuple_Size(args);
    const char *tagName = NULL;

    if (!PyArg_ParseTuple(args, "s", &tagName))
    {
        return NULL;
    }

    std::string tagNameString(tagName);
    return PyFloat_FromDouble(self->dataObject->deleteTag(tagNameString));
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectTagExists_doc,"existTag(key) -> return True if tag with given key exists, else False \n\
\n\
Checks whether a tag with the given key exists in tag dictionary of this data object and returns True if such a tag exists, else False. \n\
\n\
Parameters  \n\
------------\n\
key : {str}\n\
    the key of the tag\n\
\n\
Returns \n\
-------- \n\
result : {bool}\n\
    True if tag exists, else False");
PyObject* PythonDataObject::PyDataObj_TagExists(PyDataObject *self, PyObject *args)
{
//    int length = PyTuple_Size(args);
    const char *tagName = NULL;

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
    if (self->dataObject->existTag(tagNameString)) { Py_RETURN_TRUE; } else { Py_RETURN_FALSE; };
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectGetTagListSize_doc,"getTagListSize() -> returns the number of tags in the tag dictionary\n\
\n\
Every data object can have an arbitrary number of tags stored in the tag dictionary. This method returns the number of different tags, \
where the protocol is also one tag with the key 'protocol'. \n\
\n\
Returns \n\
------- \n\
length : {int}: \n\
    size of the tag dictionary. The optional protocol also counts as one item.");
PyObject* PythonDataObject::PyDataObj_GetTagListSize(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }
    return PyFloat_FromDouble(self->dataObject->getTagListSize());
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAddToProtocol_doc,"addToProtocol(newLine) -> Appends a protocol line to the protocol. \n\
\n\
Appends a line of text to the protocol string of this data object. If this data object has got a region of interest defined, \
the rectangle of the ROI is automatically appended to newLine. The protocol string ends with a newline character. \n\
\n\
Address the content of the protocol by obj.tags[\"protocol\"]. The protocol is contained in the ordinary tag dictionary of this data object under the key 'protocol'. \n\
\n\
Parameters  \n\
------------\n\
newLine : {str}\n\
    The text to be added to the protocol.");
PyObject* PythonDataObject::PyDataObj_AddToProtocol(PyDataObject *self, PyObject *args)
{
    const char *unit = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &unit))
    {
        return NULL;
    }
    
    std::string unitString(unit);
    if (self->dataObject->addToProtocol(unitString))
    {
        PyErr_Format(PyExc_RuntimeError, "Add line to protocol unit failed");
        return NULL;
    }
    Py_RETURN_NONE;
}
// Tag information functions

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_RichCompare(PyDataObject *self, PyObject *other, int cmp_op)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        Py_RETURN_NONE;
    }

    if (other == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "compare object is empty.");
        Py_RETURN_NONE;
    }

    //check type of other
    PyDataObject* otherDataObj = NULL;
    ito::DataObject resDataObj;
    PyDataObject* resultObject = NULL;

    if (PyArg_Parse(other, "O!", &PyDataObjectType, &otherDataObj))
    {
        if (otherDataObj->dataObject == NULL)
        {
            PyErr_SetString(PyExc_TypeError, "data object is empty.");
            Py_RETURN_NONE;
        }

        self->dataObject->lockRead();
        otherDataObj->dataObject->lockRead();

        try
        {
            switch (cmp_op)
            {
            case Py_LT: resDataObj = *(self->dataObject) < *(otherDataObj->dataObject); break;
            case Py_LE: resDataObj = *(self->dataObject) <= *(otherDataObj->dataObject); break;
            case Py_EQ: resDataObj = *(self->dataObject) == *(otherDataObj->dataObject); break;
            case Py_NE: resDataObj = *(self->dataObject) != *(otherDataObj->dataObject); break;
            case Py_GT: resDataObj = *(self->dataObject) > *(otherDataObj->dataObject); break;
            case Py_GE: resDataObj = *(self->dataObject) >= *(otherDataObj->dataObject); break;
            }
        }
        catch(cv::Exception exc)
        {
            //PyErr_Format(PyExc_TypeError, "file: %s, line: %d, error: %s", (exc.file).c_str(), exc.line, (exc.err).c_str());
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            return NULL;
        }

        self->dataObject->unlock();
        otherDataObj->dataObject->unlock();

        resultObject = createEmptyPyDataObject();
        resultObject->dataObject = new ito::DataObject(resDataObj); //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        return (PyObject*)resultObject;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        Py_RETURN_NONE;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PythonDataObject::PyDataObject* PythonDataObject::createEmptyPyDataObject()
{
    //PyDataObject* result = (PyDataObject*)PyType_Type.tp_call((PyObject*)&PyDataObjectType, NULL, NULL);
    PyDataObject* result = (PyDataObject*)PyObject_Call((PyObject*)&PyDataObjectType, NULL, NULL);
    if (result != NULL)
    {
        DELETE_AND_SET_NULL(result->dataObject);
        return result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PythonDataObject::checkPyDataObject(int number, PyObject* o1 /*= NULL*/, PyObject* o2 /*= NULL*/, PyObject* o3 /*= NULL*/)
{
    PyObject *temp;
    for (int i = 0; i < number; ++i)
    {
        switch(i)
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

        if (temp == NULL)
        {
            PyErr_Format(PyExc_TypeError, "%i. operand is NULL", i+1);
            return false;
        }
        else if (!PyDataObject_Check(temp) || ((PyDataObject*)(temp))->dataObject == NULL)
        {
            PyErr_Format(PyExc_TypeError, "%i. operand must be a valid data object", i);
            return false;
        }
    }
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbAdd(PyObject* o1, PyObject* o2)
{
    PyDataObject *dobj1 = NULL;
    PyDataObject *dobj2 = NULL;
    double scalar = 0;

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
        else
        {
//            return PyErr_Format(PyExc_RuntimeError,0,"second operand must be a dataObject, integer of float");
            return PyErr_Format(PyExc_RuntimeError, "second operand must be a dataObject, integer of float");
        }
    }
    else if (PyDataObject_Check(o2))
    {
        dobj1 = (PyDataObject*)o2; //dobj1 is always a dataobject!!! (difference to nbSub)
        if (PyFloat_Check(o1) || PyLong_Check(o1))
        {
            scalar = PyFloat_AsDouble(o1);
        }
        else
        {
//            return PyErr_Format(PyExc_RuntimeError,0,"second operand must be a dataObject, integer of float");
            return PyErr_Format(PyExc_RuntimeError, "second operand must be a dataObject, integer of float");
        }
    }
    else
    {
//        return PyErr_Format(PyExc_RuntimeError,0,"at least one operand must be a dataObject");
        return PyErr_Format(PyExc_RuntimeError, "at least one operand must be a dataObject");
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    if (dobj1) dobj1->dataObject->lockRead();
    if (dobj2) dobj2->dataObject->lockRead();

    try
    {
        if (dobj2)
        {
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) + *(dobj2->dataObject));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
        else
        {
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) + scalar);  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        if (dobj1) dobj1->dataObject->unlock();
        if (dobj2) dobj2->dataObject->unlock();
        return NULL;
    }

    if (dobj1) dobj1->dataObject->unlock();
    if (dobj2) dobj2->dataObject->unlock();


    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbSubtract(PyObject* o1, PyObject* o2)
{
    PyDataObject *dobj1 = NULL;
    PyDataObject *dobj2 = NULL;
    double scalar = 0;

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
        else
        {
//            return PyErr_Format(PyExc_RuntimeError,0,"second operand must be a dataObject, integer of float");
            return PyErr_Format(PyExc_RuntimeError, "second operand must be a dataObject, integer of float");
        }
    }
    else if (PyDataObject_Check(o2))
    {
        dobj2 = (PyDataObject*)o2;
        if (PyFloat_Check(o1) || PyLong_Check(o1))
        {
            scalar = PyFloat_AsDouble(o1);
        }
        else
        {
//            return PyErr_Format(PyExc_RuntimeError,0,"second operand must be a dataObject, integer of float");
            return PyErr_Format(PyExc_RuntimeError, "second operand must be a dataObject, integer of float");
        }
    }
    else
    {
//        return PyErr_Format(PyExc_RuntimeError,0,"at least one operand must be a dataObject");
        return PyErr_Format(PyExc_RuntimeError, "at least one operand must be a dataObject");
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    if (dobj1) dobj1->dataObject->lockRead();
    if (dobj2) dobj2->dataObject->lockRead();

    try
    {
        if (dobj1 && dobj2)
        {
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) - *(dobj2->dataObject));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
        else if (dobj1)
        {
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) - scalar);  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
        else
        {
            retObj->dataObject = new ito::DataObject((*(dobj2->dataObject)*(-1.0)) + scalar);  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        if (dobj1) dobj1->dataObject->unlock();
        if (dobj2) dobj2->dataObject->unlock();
        return NULL;
    }

    if (dobj1) dobj1->dataObject->unlock();
    if (dobj2) dobj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbMultiply(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (Py_TYPE(o1) == &PyDataObjectType && Py_TYPE(o2) == &PyDataObjectType)
    {
        PyDataObject *dobj1 = (PyDataObject*)(o1);
        PyDataObject *dobj2 = (PyDataObject*)(o2);

        PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

        dobj1->dataObject->lockRead();
        dobj2->dataObject->lockRead();

        try
        {
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) * *(dobj2->dataObject));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
        catch(cv::Exception exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            dobj2->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();

        return (PyObject*)retObj;
    }
    else if (Py_TYPE(o1) == &PyDataObjectType)
    {
        double factor = PyFloat_AsDouble((PyObject*)o2);
        PyDataObject *dobj1 = (PyDataObject*)(o1);

        if (PyErr_Occurred())
        {
            return NULL;
        }

        PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

        dobj1->dataObject->lockRead();

        try
        {
            retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) * factor);  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
        catch(cv::Exception exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();

        return (PyObject*)retObj;
    }
    else if (Py_TYPE(o2) == &PyDataObjectType)
    {
        double factor = PyFloat_AsDouble((PyObject*)o1);
        PyDataObject *dobj2 = (PyDataObject*)(o2);

        if (PyErr_Occurred())
        {
            return NULL;
        }

        PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

        dobj2->dataObject->lockRead();

        try
        {
            retObj->dataObject = new ito::DataObject(*(dobj2->dataObject) * factor);  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
        }
        catch(cv::Exception exc)
        {
            Py_DECREF(retObj);
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj2->dataObject->unlock();
            return NULL;
        }

        dobj2->dataObject->unlock();

        return (PyObject*)retObj;
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbRemainder(PyObject* /*o1*/, PyObject* /*o2*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbDivmod(PyObject* /*o1*/, PyObject* /*o2*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbPower(PyObject* o1, PyObject* o2, PyObject* o3)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    if ((PyObject*)o3 != Py_None)
    {
        PyErr_SetString(PyExc_TypeError, "Modulo in power-method not supported");
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();
    dobj2->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) ^ *(dobj2->dataObject));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbNegative(PyObject* o1)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject((*(dobj1->dataObject) * -1.0));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbPositive(PyObject* o1)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject));

        if (!retObj->dataObject->getOwnData())
        {
            PyDataObject_SetBase(retObj, (PyObject*)o1);
        }
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbAbsolute(PyObject* o1)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(ito::abs(*(dobj1->dataObject)));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInvert(PyObject* /*o1*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbLshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    int shift = PyLong_AsLong(o2);

    if (PyErr_Occurred()) return NULL;
    if (shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) << static_cast<unsigned int>(shift));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbRshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    int shift = PyLong_AsLong(o2);

    if (PyErr_Occurred())
    {
        return NULL;
    }
    if (shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) >> static_cast<unsigned int>(shift));  //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbAnd(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();
    dobj2->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) & *(dobj2->dataObject)); //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbXor(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();
    dobj2->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) ^ *(dobj2->dataObject)); //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbOr(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    dobj1->dataObject->lockRead();
    dobj2->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject(*(dobj1->dataObject) | *(dobj2->dataObject)); //resDataObj should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceAdd(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    if (PyDataObject_Check(o2))
    {
        PyDataObject *dobj2 = (PyDataObject*)(o2);

        dobj1->dataObject->lockWrite();
        dobj2->dataObject->lockRead();

        try
        {
            *(dobj1->dataObject) += *(dobj2->dataObject);
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            dobj2->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
    }
    else if (PyFloat_Check(o2) || PyLong_Check(o2))
    {
        double val = PyFloat_AsDouble(o2);

        dobj1->dataObject->lockWrite();

        try
        {
            *(dobj1->dataObject) += val;
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
    }
    else
    {
        return PyErr_Format(PyExc_RuntimeError, "the second operand must be either a data object or an integer or floating point value");
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceSubtract(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    if (PyDataObject_Check(o2))
    {
        PyDataObject *dobj2 = (PyDataObject*)(o2);

        dobj1->dataObject->lockWrite();
        dobj2->dataObject->lockRead();

        try
        {
            *(dobj1->dataObject) -= *(dobj2->dataObject);
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            dobj2->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
    }
    else if (PyFloat_Check(o2) || PyLong_Check(o2))
    {
        double val = PyFloat_AsDouble(o2);

        dobj1->dataObject->lockWrite();

        try
        {
            *(dobj1->dataObject) -= val;
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
    }
    else
    {
        return PyErr_Format(PyExc_RuntimeError, "the second operand must be either a data object or an integer or floating point value");
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceMultiply(PyObject* o1, PyObject* o2)
{
    if (o1 == NULL || o2 == NULL)
    {
        return NULL;
    }

    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    if (Py_TYPE(o2) == &PyDataObjectType)
    {
        PyDataObject *dobj2 = (PyDataObject*)(o2);

        dobj1->dataObject->lockWrite();
        dobj2->dataObject->lockRead();

        try
        {
            *(dobj1->dataObject) *= *(dobj2->dataObject);
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            dobj2->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
    }
    else
    {
        double factor = PyFloat_AsDouble((PyObject*)o2);

        if (PyErr_Occurred())
        {
            return NULL;
        }

        dobj1->dataObject->lockWrite();
        try
        {
            *(dobj1->dataObject) *= factor;
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            dobj1->dataObject->unlock();
            return NULL;
        }

        dobj1->dataObject->unlock();
    }

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceRemainder(PyObject* /*o1*/, PyObject* /*o2*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplacePower(PyObject* /*o1*/, PyObject* /*o2*/, PyObject* /*o3*/)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceLshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    int shift = PyLong_AsLong(o2);

    if (PyErr_Occurred()) return NULL;
    if (shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    Py_INCREF(o1);

    dobj1->dataObject->lockWrite();
    *(dobj1->dataObject) <<= static_cast<unsigned int>(shift);
    dobj1->dataObject->unlock();

    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceRshift(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(1,o1))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);

    int shift = PyLong_AsLong(o2);

    if (PyErr_Occurred()) return NULL;
    if (shift<0)
    {
        PyErr_SetString(PyExc_TypeError,"shift value must not be negative");
        return NULL;
    }

    Py_INCREF(o1);
    dobj1->dataObject->lockWrite();
    *(dobj1->dataObject) >>= static_cast<unsigned int>(shift);
    dobj1->dataObject->unlock();

    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceAnd(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    dobj1->dataObject->lockWrite();
    dobj2->dataObject->lockRead();

    try
    {
        *(dobj1->dataObject) &= *(dobj2->dataObject);
    }
    catch(cv::Exception exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceXor(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    dobj1->dataObject->lockWrite();
    dobj2->dataObject->lockRead();

    try
    {
        *(dobj1->dataObject) ^= *(dobj2->dataObject);
    }
    catch(cv::Exception exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_nbInplaceOr(PyObject* o1, PyObject* o2)
{
    if (!checkPyDataObject(2,o1,o2))
    {
        return NULL;
    }

    PyDataObject *dobj1 = (PyDataObject*)(o1);
    PyDataObject *dobj2 = (PyDataObject*)(o2);

    dobj1->dataObject->lockWrite();
    dobj2->dataObject->lockRead();

    try
    {
        *(dobj1->dataObject) |= *(dobj2->dataObject);
    }
    catch(cv::Exception exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        dobj1->dataObject->unlock();
        dobj2->dataObject->unlock();
        return NULL;
    }

    dobj1->dataObject->unlock();
    dobj2->dataObject->unlock();

    Py_INCREF(o1);
    return (PyObject*)o1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_getiter(PyDataObject* self)
{
    PyObject *args = PyTuple_Pack(1, self); //new ref
    PyDataObjectIter* result = (PyDataObjectIter*)PyObject_Call((PyObject*)&PyDataObjectIterType, args, NULL);
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectName_doc,"name() -> returns the name of this object (dataObject)");
PyObject* PythonDataObject::PyDataObject_name(PyDataObject* /*self*/)
{
    PyObject *result;
    result = PyUnicode_FromString("dataObject");
    return result;
};

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObject_repr(PyDataObject *self)
{
    PyObject *result;
    int dims;
    if (self->dataObject == NULL)
    {
        result = PyUnicode_FromFormat("DataObject(empty)");
    }
    else
    {
        self->dataObject->lockRead();
        ito::DataObject *dObj = self->dataObject;
        dims = dObj->getDims();
        switch(dims)
        {
        case 2:
            result = PyUnicode_FromFormat("DataObject('%s', [%i x %i], continuous: %i, owndata: %i)", typeNumberToName(dObj->getType()), dObj->getSize(0), dObj->getSize(1), dObj->getContinuous(), dObj->getOwnData());
            break;
        case 3:
            result = PyUnicode_FromFormat("DataObject('%s', [%i x %i x %i], continuous: %i, owndata: %i)", typeNumberToName(dObj->getType()), dObj->getSize(0), dObj->getSize(1), dObj->getSize(2), dObj->getContinuous(), dObj->getOwnData());
            break;
        default:
            result = PyUnicode_FromFormat("DataObject('%s', %i dims, continuous: %i, owndata: %i)", typeNumberToName(dObj->getType()), dObj->getDims(), dObj->getContinuous(), dObj->getOwnData());
            break;
        }
        self->dataObject->unlock();
    }
    return result;
};

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectData_doc,"data() -> prints the content of the dataObject in a readable form. \n\
\n\
Notes \n\
----- \n\
When calling this method, the complete content of the dataObject is printed to the standard output stream.");
PyObject* PythonDataObject::PyDataObject_data(PyDataObject *self)
{
    self->dataObject->lockRead();
    ito::fListCout[self->dataObject->getType()](std::cout,*(self->dataObject));
    self->dataObject->unlock();
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectConj_doc,"conj() -> complex-conjugates all elements of this dataObject (inline). \n\
\n\
Every value of this dataObject is replaced by its complex-conjugate value. \n\
\n\
Raises \n\
------- \n\
TypeError : \n\
    if data type of this data object is not complex.\n\
\n\
See Also \n\
--------- \n\
conjugate() : does the same operation but returns a complex-conjugated copy of this data object");
PyObject* PythonDataObject::PyDataObject_conj(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    self->dataObject->lockWrite();
    try
    {
        self->dataObject->conj();
    }
    catch(cv::Exception exc)
    {
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        return NULL;
    }
    self->dataObject->unlock();
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
//return the complex-conjugate, element-wise
PyDoc_STRVAR(pyDataObjectConjugate_doc,"conjugate() -> return a copy of this dataObject where every element is complex-conjugated. \n\
\n\
Returns \n\
------- \n\
out : {dataObject} \n\
    element-wise complex conjugate of this data object \n\
\n\
Raises \n\
------- \n\
TypeError : \n\
    if data type of this data object is not complex.\n\
\n\
See Also \n\
--------- \n\
conj() : does the same operation but manipulates this object inline.");
PyObject* PythonDataObject::PyDataObject_conjugate(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
        
    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    self->dataObject->lockWrite();
    
    retObj->dataObject = new ito::DataObject();
    try
    {
        self->dataObject->copyTo( *(retObj->dataObject), 1);
        retObj->dataObject->conj();
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    self->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAdj_doc, "adj() -> Adjugate all elements (inline)\n\
\n\
Every plane (spanned by the last two axes) is transposed and every element is replaced by its complex conjugate value. \n\
\n\
Raises \n\
------- \n\
TypeError : \n\
    if data type of this data object is not complex.\n\
\n\
See Also \n\
--------- \n\
adjugate() : does the same operation but returns the resulting data object");
PyObject* PythonDataObject::PyDataObject_adj(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    self->dataObject->lockWrite();

    try
    {
        ito::DataObject *newDataObj = new ito::DataObject(self->dataObject->adj());
        self->dataObject->unlock();
        delete self->dataObject;
        self->dataObject = newDataObj;
    }
    catch(cv::Exception exc)
    {
        self->dataObject->unlock();
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        return NULL;
    }
    
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAdjugate_doc, "adjugate() -> returns the plane-wise adjugated array of this dataObject. \n\
\n\
If this data object has a complex type, the tranposed data object is returned where every element is complex conjugated. \
For data objects with more than two dimensions the tranposition is done plane-wise, hence, only the last two dimensions are permutated. \n\
\n\
Returns \n\
------- \n\
out : {dataObject} \n\
    adjugate of this dataObject \n\
\n\
Raises \n\
------- \n\
TypeError : \n\
    if data type of this data object is not complex.\n\
\n\
See Also \n\
--------- \n\
adj() : does the same operation but manipulates this object inline.");
PyObject* PythonDataObject::PyDataObject_adjugate(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    
    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    self->dataObject->lockRead();
    
    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->adj());
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    self->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectTrans_doc, "trans() -> return a plane-wise transposed dataObject\n\
\n\
Return a new data object with the same data type than this object and where every plane (data spanned by the last two dimensions) \
is transposed respectively such that the last two axes are permuted. \n\
\n\
Returns \n\
-------- \n\
out : {dataObject} \n\
    A copy of this dataObject is returned where every plane is its transposed plane.");
PyObject* PythonDataObject::PyDataObject_trans(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    
    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    self->dataObject->lockRead();
    
    try
    {
        retObj->dataObject = new ito::DataObject(self->dataObject->trans());
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        return NULL;
    }

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    self->dataObject->unlock();

    return (PyObject*)retObj;

    //return PyErr_Format(PyExc_ValueError, "TODO: due to removal of transpose flag (obsolete?)");
    //if (self->dataObject == NULL)
    //{
    //    PyErr_SetString(PyExc_ValueError, "data object is NULL");
    //    return NULL;
    //}
    ////self->dataObject->lockWrite();
    ////self->dataObject->trans();
    ////self->dataObject->unlock();
    ////Py_RETURN_NONE;

    //PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    //self->dataObject->lockWrite();
    //
    //retObj->dataObject = new ito::DataObject(*(self->dataObject));
    //retObj->dataObject->trans();

    //if (!retObj->dataObject->getOwnData())
    //{
    //    PyDataObject_SetBase(retObj, (PyObject*)self);
    //}

    //self->dataObject->unlock();

    //return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectMakeContinuous_doc, "makeContinuous() -> return continuous representation of dataObject\n\
\n\
Per default a dataObject with more than two dimensions allocates separated chunks of memory for every plane, where \
a plane is always the matrix given by the last two dimensions. This separated storage usually allows allocating more \
memory for huge for instance three dimensional matrices. However, in order to generate a dataObject that is directly \
compatible to Numpy or other C-style matrix structures, the entire allocated memory must be in one block, that is called \
continuous. If you create a Numpy array from a dataObject that is not continuous, this function is implicitely called \
in order to firstly make the dataObject continuous before passing to Numpy. \n\
\n\
Returns \n\
-------- \n\
obj : {dataObject} \n\
    continuous dataObject\n\
\n\
Notes \n\
----- \n\
if this dataObject already is continuous, a simple shallow copy is returned");
PyObject* PythonDataObject::PyDataObject_makeContinuous(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    self->dataObject->lockWrite();
    retObj->dataObject = new ito::DataObject(ito::makeContinuous(*(self->dataObject)));

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    self->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSize_doc,"size([index]) -> returns the size of this dataObject (tuple of the sizes in all dimensions or size in dimension indicated by optional axis index). \n\
\n\
Parameters  \n\
------------\n\
index : {int}, optional\n\
    If index is given, only the size of the indicated dimension is returned as single number (0 <= index < number of dimensions) \n\
\n\
Returns \n\
-------- \n\
A tuple containing the sizes of all dimensions or one single size value if 'index' is indicated. \n\
\n\
Notes \n\
--------- \n\
For a more consistent syntax with respect to numpy arrays, the same result is obtained by the attribute shape. Please use the attribute shape for future implementations \
since this method is marked as deprecated.\n\
\n\
See Also \n\
--------- \n\
shape : the read-only attribute shape is equal to size()");
PyObject* PythonDataObject::PyDataObject_size(PyDataObject *self, PyObject* args)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning, "size([idx]) is deprecated. Use attribute shape instead (more consistent to numpy)",1) == -1) //exception is raised instead of warning (depending on user defined warning levels)
    {
        return NULL;
    }

    PyObject *shapes = PyDataObj_GetShape(self, NULL);
    int desiredDim = 0;

    if (PyTuple_Size(args) > 0 && shapes)
    {
        if (PyArg_ParseTuple(args, "i", &desiredDim))
        {
            if (desiredDim >= 0 && desiredDim < PyTuple_Size(shapes))
            {
                PyObject *temp = shapes;
                shapes = PyTuple_GetItem(shapes,desiredDim);
                Py_INCREF(shapes);
                Py_DECREF(temp);
            }
            else
            {
                Py_DECREF(shapes);
                PyErr_SetString(PyExc_TypeError, "index argument out of boundaries.");
            }
        }
        else
        {
            Py_DECREF(shapes);
            PyErr_SetString(PyExc_TypeError, "argument must be valid index or nothing.");
        }
    }

    return shapes;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectCopy_doc,"copy(regionOnly=0) -> return a deep copy of this dataObject\n\
\n\
Parameters \n\
----------- \n\
regionOnly : {bool}, optional \n\
    If regionOnly is 1, only the current region of interest of this dataObject is copied, else the entire dataObject \
    including the current settings concerning the region of interest are deeply copied [default].\n\
\n\
Returns \n\
------- \n\
cpy : {dataObject} \n\
    Deep copy of this dataObject");
PyObject* PythonDataObject::PyDataObject_copy(PyDataObject *self, PyObject* args)
{
    if (self->dataObject == NULL) return 0;

    unsigned char regionOnly = 0;

    if (!PyArg_ParseTuple(args, "|b", &regionOnly))
    {
        PyErr_Format(PyExc_TypeError,"the region only flag must be 0 or 1");
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    retObj->dataObject = new ito::DataObject();

    self->dataObject->lockRead();

    try
    {
        if (regionOnly)
        {
            self->dataObject->copyTo(*(retObj->dataObject),1);  //self->dataObject should always be the owner of its data, therefore base of resultObject remains None
        }
        else
        {
            self->dataObject->copyTo(*(retObj->dataObject),0);  //self->dataObject should always be the owner of its data, therefore base of resultObject remains None
        }
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        return NULL;
    }

    self->dataObject->unlock();

    return (PyObject*)retObj;

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectMul_doc, "mul(obj) -> a.mul(b) returns element wise multiplication of a*b\n\
\n\
Parameters  \n\
------------\n\
obj : {dataObject} \n\
    dataObject whose values are element-wisely multiplied with the values in this dataObject. \n\
\n\
Returns \n\
-------- \n\
c : {dataObject} \n\
    Resulting multiplied data object.");
PyObject* PythonDataObject::PyDataObject_mul(PyDataObject *self, PyObject *args)
{
    if (self->dataObject == NULL) return 0;

    PyObject *pyDataObject = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &pyDataObject))
    {
        return PyErr_Format(PyExc_RuntimeError,"argument is no data object");
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    PyDataObject* obj2 = (PyDataObject*)pyDataObject;

    self->dataObject->lockRead();
    obj2->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject((*(self->dataObject)).mul(*(obj2->dataObject)));  //new dataObject should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        obj2->dataObject->unlock();
        return NULL;
    }

    self->dataObject->unlock();
    obj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectDiv_doc, "div(obj) -> a.div(b) return result of element wise division of a./b \n\
\n\
Parameters  \n\
------------\n\
obj : {dataObject} \n\
    Every value in this data object is divided by the corresponding value in obj. \n\
\n\
Returns \n\
-------- \n\
c : {dataObject} \n\
    Resulting divided data object.");
PyObject* PythonDataObject::PyDataObject_div(PyDataObject *self, PyObject *args)
{
    if (self->dataObject == NULL) return 0;

    PyObject *pyDataObject = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &pyDataObject))
    {
        return PyErr_Format(PyExc_RuntimeError,"argument is no data object");
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    PyDataObject* obj2 = (PyDataObject*)pyDataObject;

    self->dataObject->lockRead();
    obj2->dataObject->lockRead();

    try
    {
        retObj->dataObject = new ito::DataObject((*(self->dataObject)).div(*(obj2->dataObject)));//new dataObject should always be the owner of its data, therefore base of resultObject remains None
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        self->dataObject->unlock();
        obj2->dataObject->unlock();
        return NULL;
    }

    self->dataObject->unlock();
    obj2->dataObject->unlock();

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectReshape_doc,"reshape(newSizes) -> Returns reshaped shallow copy of data object  \n\
\n\
Notes \n\
----- \n\
Not implemented yet.\n\
\n\
");
PyObject* PythonDataObject::PyDataObject_reshape(PyDataObject *self, PyObject *args)
{
    PyErr_Format(PyExc_NotImplementedError,"Not implemented yet");
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAstype_doc,"astype(typestring) -> converts this data object to another type\n\
\n\
Converts this data object to a new data object with another type, given by the string newTypestring (e.g. 'uint8'). The converted data object \
is a deep copy of this object if the new type does not correspond to the current type, else a shallow copy of this object is returned. \n\
\n\
Parameters \n\
----------- \n\
typestring : {str} \n\
    Type string indicating the new type ('uint8',...'float32',..,'complex64') \n\
\n\
Returns \n\
-------- \n\
c : {dataObject} \n\
    type-converted data object \n\
\n\
Notes \n\
----- \n\
This method mainly uses the method convertTo of OpenCV. \n\
");
PyObject* PythonDataObject::PyDataObject_astype(PyDataObject *self, PyObject* args, PyObject* kwds)
{
    const char* type;
    int typeno = 0;

    const char *kwlist[] = {"typestring", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds,"s",const_cast<char**>(kwlist), &type))
    {
        return NULL;
    }

    typeno = typeNameToNumber(type);

    if (typeno == -1)
    {
        PyErr_Format(PyExc_TypeError,"The given type string %s is unknown", type);
        return NULL;
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    retObj->dataObject = new ito::DataObject();

    self->dataObject->lockRead();
    try
    {
        self->dataObject->convertTo(*(retObj->dataObject), typeno);
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    self->dataObject->unlock();

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectNormalize_doc,"normalize([minValue=0.0, maxValue=1.0, typestring='']) -> returns the normalization of this dataObject\n\
\n\
Returns the normalized version of this data object, where the values lie in the range [minValue,maxValue]. Additionally it is also \n\
possible to convert the resulting data object to another type (given by the parameter typestring). As default no type conversion is executed.\n\
\n\
Parameters \n\
----------- \n\
minValue : {double} \n\
    minimum value of the normalized range \n\
maxValue : {double} \n\
    maximum value of the normalized range \n\
typestring : {String} \n\
    Type string indicating the new type ('uint8',...'float32',..,'complex64'), default: '' (no type conversion) \n\
\n\
Returns \n\
-------- \n\
normalized : {dataObject} \n\
    normalized data object");
PyObject* PythonDataObject::PyDataObject_normalize(PyDataObject *self, PyObject* args, PyObject* kwds)
{
    const char* type = NULL;
    double minVal = 0.0;
    double maxVal = 1.0;
    int typeno = 0;

    const char *kwlist[] = {"minValue", "maxValue", "typestring", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds,"|dds",const_cast<char**>(kwlist), &minVal, &maxVal, &type))
    {
        return NULL;
    }

    if (type != NULL)
    {
        typeno = typeNameToNumber(type);

        if (typeno == -1)
        {
            PyErr_Format(PyExc_TypeError,"The given type string %s is unknown", type);
            return NULL;
        }
    }
    else
    {
        typeno = self->dataObject->getType();
    }

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference
    ito::DataObject dataObj;

    self->dataObject->lockRead();
    double smin, smax;
    ito::uint32 loc1[] = {0,0,0};
    ito::uint32 loc2[] = {0,0,0};
    ito::dObjHelper::minMaxValue(self->dataObject, smin, loc1, smax, loc2, true);

    double dmin = std::min(minVal, maxVal);
    double dmax = std::max(minVal, maxVal);
    double scale = (dmax-dmin)*((smax - smin) > std::numeric_limits<double>::epsilon() ? (1./(smax-smin)) : 0.0);
    double shift = dmin-smin*scale;
    try
    {
        self->dataObject->convertTo(dataObj, typeno, scale, shift);
    }
    catch(cv::Exception exc)
    {
        Py_DECREF(retObj);
        PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
        return NULL;
    }

    self->dataObject->unlock();

    retObj->dataObject = new ito::DataObject(dataObj);

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectLocateROI_doc,"locateROI() -> returns information about the current region of interest of this data object\n\
\n\
A region of interest (ROI) of a data object is defined by the two values per axis. The first element always indicates the size between the \n\
real border of the data object and the region of interest on the left / top ... side and the second value the margin of the right / bottom ... side. \n\
\n\
This method returns a tuple with two elements: The first is a list with the original sizes of this data object, \
the second is a list with the offsets from the original data object to the first value in the current region of interest \n\
\n\
If no region of interest is set (hence: full region of interest), the first list corresponds to the one returned by size(), \
the second list is a zero-vector. \n\
\n\
See Also \n\
-------- \n\
adjustROI(offsetList) : method to change the current region of interest");
PyObject* PythonDataObject::PyDataObject_locateROI(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }
    self->dataObject->lockRead();
    int dims = self->dataObject->getDims();
    int *osize = new int[dims];
    int *offsets = new int[dims];

    self->dataObject->locateROI(osize, offsets);
    self->dataObject->unlock();

    PyObject *osize_obj = PyList_New(dims);
    PyObject *offsets_obj = PyList_New(dims);

    for (int i=0;i<dims;i++)
    {
        PyList_SetItem(osize_obj, i, Py_BuildValue("i",osize[i]));
        PyList_SetItem(offsets_obj, i, Py_BuildValue("i",offsets[i]));
    }

    DELETE_AND_SET_NULL_ARRAY(osize);
    DELETE_AND_SET_NULL_ARRAY(offsets);

    PyObject *result = Py_BuildValue("(OO)", osize_obj, offsets_obj);
    Py_DECREF(osize_obj);
    Py_DECREF(offsets_obj);

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectAdjustROI_doc, "adjustROI(offsetList) -> adjust the size and position of the region of interest of this data object\n\
\n\
For every data object, it is possible to define a region of interest such that subsequent commands only refer to this subpart. However, if values within \n\
the region of interest (ROI) are changed, this also affects the original data object due to the shallow copy principal of python. \n\
Use this command to adjust the current size and position of this region of interest by passing an offset list, that contains \
integer numbers with twice the size than the number of dimensions. \n\
\n\
Example: :: \n\
\n\
    d = dataObject([5,4]) \n\
    droi = d \n\
    droi.adjustROI([-2,0,-1,-1]) \n\
    \n\
Now *droi* is a region of interest of the original data object whose first value is equal to d[2,1] and its size is (3,2) \n\
\n\
Parameters \n\
----------- \n\
offsetList : {list of integers} \n\
    This list must have twice as many values than the number of dimensions of this data object. A pair of numbers indicates the shift of the \
    current boundaries of the region of interest in every dimension. The first value of each pair is the offset of the 'left' boundary, the \
    second the shift of the right boundary. A positive value means a growth of the region of interest, a negative one let the region of interest \
    shrink towards the center. \n\
\n\
See Also \n\
--------- \n\
locateROI() : method to get the borders of the current ROI");
PyObject* PythonDataObject::PyDataObject_adjustROI(PyDataObject *self, PyObject* args)
{
    //args is supposed to be a list of offsets for each dimensions on the "left" and "right" side.
    //e.g. 2D-Object [dtop, dbottom, dleft, dright], negative d-value means offset towards the center
    Py_ssize_t sizeOffsets;
    int sizeOffsetsInt;
    PyObject* offsets = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "data object is NULL");
        return NULL;
    }

    self->dataObject->lockWrite();


    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &offsets))
    {
        PyErr_SetString(PyExc_ValueError, "argument must be a list of offset-values. Its length must be two times the number of matrix-dimensions");
        return NULL;
    }

    sizeOffsets = PyList_Size(offsets);

    if (sizeOffsets >= INT_MAX)
    {
        PyErr_SetString(PyExc_ValueError, "length of argument list must be smaller than the maximum integer value");
        return NULL;
    }

    sizeOffsetsInt = Py_SAFE_DOWNCAST(sizeOffsets, Py_ssize_t, int);

    if (sizeOffsetsInt != 2*self->dataObject->getDims())
    {
        Py_DECREF(offsets);
        PyErr_SetString(PyExc_ValueError, "argument must be a list of offset-values. Its length must be two times the number of matrix-dimensions");
        return NULL;
    }

    int dims = self->dataObject->getDims();
    bool error = false;

    if (dims > 0)
    {
        int *offsetVector = new int[2*dims];
        PyObject *temp;

        for (int i=0;i<2*dims;i++)
        {
            temp = PyList_GetItem(offsets,i); //borrowed
            if (!PyLong_Check(temp))
            {
                PyErr_SetString(PyExc_ValueError, "at least one element in the offset list has no integer type");
                break;
            }
            offsetVector[i] = PyLong_AsLong(temp);
        }
        try
        {
            self->dataObject->adjustROI(dims, offsetVector);
        }
        catch(cv::Exception exc)
        {
            PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
            error = true;
        }

        DELETE_AND_SET_NULL_ARRAY(offsetVector);
    }

    self->dataObject->unlock();

    Py_DECREF(offsets);

    if (error)
    {
        return NULL;
    }
    else
    {
        Py_RETURN_NONE;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectSqueeze_doc,"squeeze() -> return a squeezed shallow copy (if possible) of this dataObject. \n\
\n\
This method removes every dimension with size equal to 1. Take care, that \n\
none of the last two dimensions is considered by this squeeze-command. \n\
\n\
Returns \n\
-------- \n\
squeezed : {dataObject} \n\
    The squeezed data object where all kept planes are shallow copies of the original plane. \n\
\n\
Notes \n\
----- \n\
The returned squeezed data object is a shallow copy of the original data object and hence changes in its values\n\
will also change the original data set.\n\
This method is equal to numpy.squeeze");
PyObject* PythonDataObject::PyDataObject_squeeze(PyDataObject *self, PyObject* /*args*/)
{
    if (self->dataObject == NULL) return 0;

    PyDataObject* retObj = PythonDataObject::createEmptyPyDataObject(); // new reference

    self->dataObject->lockRead();
    ito::DataObject resObj = self->dataObject->squeeze();
    self->dataObject->unlock();

    retObj->dataObject = new ito::DataObject(resObj);

    if (!retObj->dataObject->getOwnData())
    {
        PyDataObject_SetBase(retObj, (PyObject*)self);
    }

    return (PyObject*)retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObj_mappingLength(PyDataObject* self)
{
    if (self->dataObject == NULL) return 0;

    self->dataObject->lockRead();

    int dims = self->dataObject->getDims();
    int count = dims > 0 ? 1 : 0;

    for (int i = 0; i < dims; i++)
    {
        count *= self->dataObject->getSize(i); //independent on transpose flag
    }

    self->dataObject->unlock();

    return count;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_mappingGetElem(PyDataObject* self, PyObject* key)
{
    PyObject *retObj = NULL;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    self->dataObject->lockRead();

    int dims = self->dataObject->getDims();
    ito::Range *ranges = NULL;
    unsigned int *singlePointIdx = NULL;
    bool singlePoint = true;

    if (dims <= 0)
    {
        self->dataObject->unlock();
        Py_RETURN_NONE;
    }

    if (dims == 1)
    {
        self->dataObject->unlock();
        PyErr_SetString(PyExc_TypeError, "data object dimension must not be one, but two instead");
        return NULL;
    }

    if (!PyTuple_Check(key))
    {
        key = PyTuple_Pack(1,key);
    }

    if (PyTuple_Size(key) != dims)
    {
        self->dataObject->unlock();
        PyErr_SetString(PyExc_TypeError, "length of key-tuple does not fit to dimension of data object");
        return NULL;
    }

    Py_ssize_t length = PyTuple_Size(key);
    ranges = new ito::Range[dims];
    singlePointIdx = new unsigned int[dims];
    bool error = false;
    PyObject* elem = NULL;
    int temp1;

    for (Py_ssize_t i = 0; i < length && !error; i++)
    {
        elem = PyTuple_GetItem(key, i);

        //check type of elem, must be int or stride
        if (PyLong_Check(elem))
        {
            temp1 = PyLong_AsLong(elem);

            if (temp1 >= 0 && temp1 < static_cast<long>(self->dataObject->getSize(i))) //temp1 is still the virtual order, therefore check agains the getSize-method which considers the transpose-flag
            {
                ranges[i].start = temp1;
                ranges[i].end = temp1 + 1;
                singlePointIdx[i] = temp1;
            }
            else
            {
                singlePointIdx[i] = 0;
                error = true;
                PyErr_SetString(PyExc_TypeError, "length of key-tuple exceeds dimension of data object");
            }
        }
        else if (PySlice_Check(elem))
        {
            singlePoint = false;

            Py_ssize_t start, stop, step, slicelength;
            if (PySlice_GetIndicesEx(elem, self->dataObject->getSize(i), &start, &stop, &step, &slicelength) == 0)
            {
                if (step != 1)
                {
                    error = true;
                    PyErr_SetString(PyExc_TypeError, "step size must be one.");
                }
                else
                {
                    ranges[i].start = start;
                    ranges[i].end = stop; //stop already points one index after the last index within the range, this is the same definition than openCV has.
                }
            }
            else
            {
                error = true;
                //error is already set by command
                //PyErr_SetString(PyExc_TypeError, "no valid start and stop element can be found for given slice");
            }
        }
        else
        {
            error = true;
            PyErr_SetString(PyExc_TypeError, "range tuple element is neither of type integer nor of type slice");
        }
    }

    if (!error)
    {
        if (singlePoint)
        {
            retObj = PyDataObj_At(self->dataObject, singlePointIdx);
        }
        else
        {
            PyDataObject *retObj2 = PythonDataObject::createEmptyPyDataObject(); // new reference
            try
            {
                retObj2->dataObject = new ito::DataObject(self->dataObject->at(ranges));

                if (!retObj2->dataObject->getOwnData())
                {
                    PyDataObject_SetBase(retObj2, (PyObject*)self);
                }

                retObj2->dataObject->unlock();
                retObj = (PyObject*)retObj2;
            }
            catch(cv::Exception exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                Py_DECREF(retObj2);
                retObj2 = NULL;
            }
        }
    }

    self->dataObject->unlock();

    DELETE_AND_SET_NULL_ARRAY(ranges);
    DELETE_AND_SET_NULL_ARRAY(singlePointIdx);

    return retObj;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObj_mappingSetElem(PyDataObject* self, PyObject* key, PyObject* value)
{
    DataObject dataObj;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return 0;
    }

    self->dataObject->lockRead();

    int dims = self->dataObject->getDims();
    ito::Range *ranges = NULL;
    unsigned int *idx = NULL; //redundant to range, if only single indizes are addressed

    if (dims <= 0)
    {
        self->dataObject->unlock();
        return -1;
    }

    if (!PyTuple_Check(key))
    {
        key = PyTuple_Pack(1,key);
    }

    if (PyTuple_Size(key) != dims)
    {
        self->dataObject->unlock();
        PyErr_SetString(PyExc_TypeError, "length of key-tuple does not fit to dimension of data object");
        return -1;
    }

    Py_ssize_t length = PyTuple_Size(key);
    ranges = new ito::Range[dims];
    idx = new unsigned int[dims];

    bool error = false;
    bool containsSlices = false;
    PyObject* elem = NULL;
    int temp1;

    for (Py_ssize_t i = 0; i < length && !error; i++)
    {
        elem = PyTuple_GetItem(key, i);

        //check type of elem, must be int or stride
        if (PyLong_Check(elem))
        {
            temp1 = PyLong_AsLong(elem);

            if (temp1 >= 0 && temp1 < static_cast<long>(self->dataObject->getSize(i)))
            {
                ranges[i].start = temp1;
                ranges[i].end = temp1+1;
                idx[i] = temp1;
            }
            else
            {
                error = true;
                PyErr_SetString(PyExc_TypeError, "length of key-tuple exceeds dimension of data object");
            }
        }
        else if (PySlice_Check(elem))
        {
            containsSlices = true;
            Py_ssize_t start, stop, step, slicelength;
            if (PySlice_GetIndicesEx(elem, self->dataObject->getSize(i), &start, &stop, &step, &slicelength) == 0)
            {
                if (step != 1)
                {
                    error = true;
                    PyErr_SetString(PyExc_TypeError, "step size must be one.");
                }
                else
                {
                    ranges[i].start = start;
                    ranges[i].end = stop; //stop already points one index after the last index within the range, this is the same definition than openCV has.
                }
            }
            else
            {
                error = true;
                //error is already set by command
                //PyErr_SetString(PyExc_TypeError, "no valid start and stop element can be found for given slice");
            }
        }
        else
        {
            error = true;
            PyErr_SetString(PyExc_TypeError, "range tuple element is neither of type integer nor of type slice");
        }

    }

    if (containsSlices)
    {
        if (!error)
        {
            try
            {
                //self->dataObject in readLock, dataObj will become readLock, too
                dataObj = self->dataObject->at(ranges); //self->dataObject must not be locked for writing, since dataObj will read it
            }
            catch(cv::Exception exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                error = true;
            }
        }

        dataObj.unlock();
        dataObj.lockWrite(); //dataObj is a shallow copy of dataObject and is in readLock, right now. This lock must be switched to writeLock (usually unlock and lockWrite should be done in one "protected" step)

        //no parse value and assign it to dataObj
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
                    PyErr_SetString(PyExc_ValueError, "value exceeds the negative boundary of int32.");
                    error = true;
                }
                else //overflow = 1
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
                catch(cv::Exception exc)
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
                    PyErr_SetString(PyExc_TypeError, "An assignment of type itom.rgba is only possible for data objects of type rgba32");
                    error = true;
                }
            }
            else
            {
                error = true;
                PyErr_SetString(PyExc_TypeError, "assign value has no of the following types: integer, floating point, complex, rgba (type rgba32 only)");
            }
        }

        self->dataObject->unlock(); //self->dataObject is not used any more (unlocked from readLock)
        dataObj.unlock();

    }
    else //contains no slices
    {

        void* valuePtr;
        ito::tDataType fromType = ito::tInt8;
        int32 value1;
        float64 value2;
        complex128 value3;

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
                    dataObj = self->dataObject->at(ranges); //dataObj in readLock
                    dataObj.unlock();
                    dataObj.lockWrite();
                    ((PyDataObject*)value)->dataObject->deepCopyPartial(dataObj);
                    dataObj.unlock();
                }
                catch(cv::Exception exc)
                {
                    PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                    error = true;
                }

            }
            else
            {
                error = true;
                PyErr_SetString(PyExc_TypeError, "assign value has no of the following types: integer, floating point, complex");
            }

            self->dataObject->unlock();
        }

        if (!error && fromType != ito::tInt8)
        {

            self->dataObject->lockWrite();

            try
            {
                switch(self->dataObject->getType())
                {
                case ito::tUInt8:
                    self->dataObject->at<uint8>(idx) = ito::numberConversion<uint8>(fromType, valuePtr);
                    break;
                case ito::tInt8:
                    self->dataObject->at<int8>(idx) = ito::numberConversion<int8>(fromType, valuePtr);
                    break;
                case ito::tUInt16:
                    self->dataObject->at<uint16>(idx) = ito::numberConversion<uint16>(fromType, valuePtr);
                    break;
                case ito::tInt16:
                    self->dataObject->at<int16>(idx) = ito::numberConversion<int16>(fromType, valuePtr);
                    break;
                case ito::tUInt32:
                    self->dataObject->at<uint32>(idx) = ito::numberConversion<uint32>(fromType, valuePtr);
                    break;
                case ito::tInt32:
                    self->dataObject->at<int32>(idx) = ito::numberConversion<int32>(fromType, valuePtr);
                    break;
                case ito::tRGBA32:
                    self->dataObject->at<ito::Rgba32>(idx) = ito::numberConversion<ito::Rgba32>(fromType, valuePtr);
                    break;
                case ito::tFloat32:
                    self->dataObject->at<float32>(idx) = ito::numberConversion<float32>(fromType, valuePtr);
                    break;
                case ito::tFloat64:
                    self->dataObject->at<float64>(idx) = ito::numberConversion<float64>(fromType, valuePtr);
                    break;
                case ito::tComplex64:
                    self->dataObject->at<complex64>(idx) = ito::numberConversion<complex64>(fromType, valuePtr);
                    break;
                case ito::tComplex128:
                    self->dataObject->at<complex128>(idx) = ito::numberConversion<complex128>(fromType, valuePtr);
                    break;
                }

            }
            catch(cv::Exception exc)
            {
                PyErr_SetString(PyExc_TypeError, (exc.err).c_str());
                error = true;
            }

            self->dataObject->unlock();
        }
    }

    DELETE_AND_SET_NULL_ARRAY(ranges);
    DELETE_AND_SET_NULL_ARRAY(idx);

    return error ? -1 : 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal PythonDataObject::parseTypeNumber(int typeno, char &typekind, int &itemsize)
{
    switch(typeno)
    {
    case ito::tUInt8:
        typekind = 'u';
        itemsize = sizeof(uint8);
        break;
    case ito::tInt8:
        typekind = 'i';
        itemsize = sizeof(int8);
        break;
    case ito::tUInt16:
        typekind = 'u';
        itemsize = sizeof(uint16);
        break;
    case ito::tInt16:
        typekind = 'i';
        itemsize = sizeof(int16);
        break;
    case ito::tUInt32:
        typekind = 'u';
        itemsize = sizeof(uint32);
        break;
    case ito::tInt32:
        typekind = 'i';
        itemsize = sizeof(int32);
        break;
    case ito::tFloat32:
        typekind = 'f';
        itemsize = sizeof(float32);
        break;
    case ito::tFloat64:
        typekind = 'f';
        itemsize = sizeof(float64);
        break;
    case ito::tComplex64:
        typekind = 'c';
        itemsize = sizeof(complex64);
        break;
    case ito::tComplex128:
        typekind = 'c';
        itemsize = sizeof(complex128);
        break;
    default:
        return RetVal(retError);
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::parseTypeNumberInverse(char typekind, int itemsize)
{
    if (typekind == 'i')
    {
        switch(itemsize)
        {
        case 1: return ito::tInt8;
        case 2: return ito::tInt16;
        case 4: return ito::tInt32;
        }
    }
    else if (typekind == 'u')
    {
        switch(itemsize)
        {
        case 1: return ito::tUInt8;
        case 2: return ito::tUInt16;
        case 4: return ito::tUInt32;
        }
    }
    else if (typekind == 'f')
    {
        switch(itemsize)
        {
        case 4: return ito::tFloat32;
        case 8: return ito::tFloat64;
        }
    }
    else if (typekind == 'c')
    {
        switch(itemsize)
        {
        case 8: return ito::tComplex64;
        case 16: return ito::tComplex128;
        }
    }

    return -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::getTypenumOfCompatibleType(char typekind, int itemsize)
{
    if (typekind == 'b')
    {
        switch(itemsize)
        {
        case 1: return NPY_UBYTE; //convert bool to uint8
        }
    }
    return -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectAttrTagDict_doc,"return dictionary with all meta information of this dataObject \n\
\n\
Returns a new dictionary with the following meta information: \n\
\n\
* axisOffsets : List with offsets of each axis \n\
* axisScales : List with the scales of each axis \n\
* axisUnits : List with the unit strings of each axis \n\
* axisDescriptions : List with the description strings of each axis \n\
* tags : Dictionary with all tags including the tag 'protocol' if at least one protocol entry has been added using addToProtocol \n\
* valueOffset : Offset of each value (0.0) \n\
* valueScale : Scale of each value (1.0) \n\
* valueDescription : Description of the values \n\
* valueUnit : The unit string of the values \n\
\n\
Notes \n\
----- \n\
Adding or changing values to / in the dictionary does not change the meta information of the dataObject. \
Use the corresponding setters like setTag... instead.");
PyObject* PythonDataObject::PyDataObject_getTagDict(PyDataObject *self, void * /*clousure*/)
{
    PyObject *item = NULL;

    if (self == NULL)
    {
        PyErr_Format(PyExc_ValueError, "dataObject is NULL");
        return NULL;
    }

    if (self->dataObject == NULL)
    {
        PyErr_Format(PyExc_ValueError, "content of dataObject is NULL");
        return NULL;
    }

    PyObject *dict = PyDict_New();

    DataObject *dObj = self->dataObject;
    int tagSize = dObj->getTagListSize();
    //std::string tempString;
    DataObjectTagType tempTag;
    std::string tempKey;
    bool validOp;

    //1. tags (here it is bad to use the tags-getter, since this returns a dict_proxy, which cannot directly be pickled
    PyObject *tempTagDict = PyDict_New();
    for (int i=0;i<tagSize;i++)
    {
        tempKey = dObj->getTagKey(i,validOp);
        if (validOp)
        {
            //tempString = dObj->getTag(tempKey, validOp);
            //if (validOp) PyDict_SetItem(tempTagDict, PyUnicode_FromString(tempKey.data()), PyUnicode_FromString(tempString.data()));
            dObj->getTagByIndex(i, tempKey, tempTag);
            if (tempTag.getType() == DataObjectTagType::typeDouble)
            {
                item = PyFloat_FromDouble(tempTag.getVal_ToDouble());
                PyDict_SetItemString(tempTagDict, tempKey.data(), item);
                Py_DECREF(item);
            }
            else
            {
                item = PythonQtConversion::QByteArrayToPyUnicodeSecure(tempTag.getVal_ToString().data());
                PyDict_SetItemString(tempTagDict, tempKey.data(), item);
                Py_DECREF(item);
            }
        }
    }
    //1. tags
    PyDict_SetItemString(dict, "tags", tempTagDict);
    Py_DECREF(tempTagDict);

    //2. axisScales
    item = PyDataObject_getAxisScales(self,NULL);
    PyDict_SetItemString(dict, "axisScales", item);
    Py_DECREF(item);

    //3. axisOffsets
    item = PyDataObject_getAxisOffsets(self,NULL);
    PyDict_SetItemString(dict, "axisOffsets", item);
    Py_DECREF(item);

    //4. axisDescriptions
    item = PyDataObject_getAxisDescriptions(self,NULL);
    PyDict_SetItemString(dict, "axisDescriptions", item);
    Py_DECREF(item);

    //5. axisUnits
    item = PyDataObject_getAxisUnits(self,NULL);
    PyDict_SetItemString(dict, "axisUnits", item);
    Py_DECREF(item);

    //6. valueUnit
    item = PyDataObject_getValueUnit(self,NULL);
    PyDict_SetItemString(dict, "valueUnit", item);
    Py_DECREF(item);

    //7. valueDescription
    item = PyDataObject_getValueDescription(self,NULL);
    PyDict_SetItemString(dict, "valueDescription", item);
    Py_DECREF(item);

    //8.
    item = PyDataObject_getValueOffset(self,NULL);
    PyDict_SetItemString(dict, "valueOffset", item);
    Py_DECREF(item);

    //9.
    item = PyDataObject_getValueScale(self,NULL);
    PyDict_SetItemString(dict, "valueScale", item);
    Py_DECREF(item);

    return dict;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectArray_StructGet_doc,"__array_struct__ -> general python-array interface (do not call this directly) \n\
                                           This interface makes the data object compatible to every array structure in python \n\
                                           which does equally implement the array interface (e.g. NumPy). This method is \n\
                                           therefore a helper method for the array interface.");
PyObject* PythonDataObject::PyDataObj_Array_StructGet(PyDataObject *self)
{
    PyArrayInterface *inter;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject* selfDO = self->dataObject;

    if (selfDO->getContinuous() == false)
    {
        return PyErr_Format(PyExc_RuntimeError, "the dataObject cannot be directly converted into a numpy array since it is not continuous.");
    }

    selfDO->lockRead();

    /*if (selfDO->isT())
    {
        selfDO->unlock();
        selfDO->lockWrite();
        selfDO->evaluateTransposeFlag();
        selfDO->unlock();
        selfDO->lockRead();
    }*/

    inter = new PyArrayInterface;
    if (inter==NULL) {
        selfDO->unlock();
        return PyErr_NoMemory();
    }

    inter->two = 2;
    inter->nd = selfDO->getDims();

    if (inter->nd == 0)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        DELETE_AND_SET_NULL(inter)
        selfDO->unlock();
        return NULL;
    }

    parseTypeNumber(selfDO->getType(), inter->typekind, inter->itemsize);
#if (NPY_FEATURE_VERSION < 0x00000007)
    inter->flags = NPY_WRITEABLE | NPY_ALIGNED | NPY_NOTSWAPPED; //NPY_NOTSWAPPED indicates, that both data in opencv and data in numpy should have the same byteorder (Intel: little-endian)
#else
    inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED; //NPY_NOTSWAPPED indicates, that both data in opencv and data in numpy should have the same byteorder (Intel: little-endian)
#endif

    //check if size and osize are totally equal, then set continuous flag
    if (selfDO->getTotal() == selfDO->getOriginalTotal())
    {
#if (NPY_FEATURE_VERSION < 0x00000007)
        inter->flags |= NPY_C_CONTIGUOUS;
#else
        inter->flags |= NPY_ARRAY_C_CONTIGUOUS;
#endif
    }

    inter->descr = NULL;
    inter->data = NULL;
    inter->shape = NULL;
    inter->strides = NULL;

    if (selfDO->getDims() > 0)
    {
        unsigned int firstMDataIndex = selfDO->seekMat(0);
        inter->data = (void*)((cv::Mat*)selfDO->get_mdata()[firstMDataIndex])->data;

        inter->shape = (npy_intp *)malloc(inter->nd * sizeof(npy_intp));
        inter->strides = (npy_intp *)malloc(inter->nd * sizeof(npy_intp));

        inter->shape[inter->nd - 1] = (npy_intp)selfDO->getSize(inter->nd - 1); //since transpose flag has been evaluated and is false now, everything is ok here
        inter->strides[inter->nd - 1] = inter->itemsize;
        for (int i = inter->nd - 2; i >= 0; i--)
        {
            inter->shape[i] = (npy_intp)selfDO->getSize(i); //since transpose flag has been evaluated and is false now, everything is ok here
            inter->strides[i] = inter->strides[i+1] * selfDO->getOriginalSize(i+1); //since transpose flag has been evaluated and is false now, everything is ok here
        }
    }

    selfDO->unlock();

    //don't icrement SELF here, since the receiver of the capsule (e.g. numpy-method) will increment the refcount of then PyDataObject SELF by itself.
    return PyCapsule_New((void*)inter, NULL, &PyDataObj_Capsule_Destructor);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObjectArray_Interface_doc,"__array_interface__ -> general python-array interface (do not call this directly) \n\
                                           This interface makes the data object compatible to every array structure in python \n\
                                           which does equally implement the array interface (e.g. NumPy).");
PyObject* PythonDataObject::PyDataObj_Array_Interface(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    PyObject *item = NULL;

    ito::DataObject* selfDO = self->dataObject;

    if (selfDO->getContinuous() == false)
    {
        return PyErr_Format(PyExc_RuntimeError, "the dataObject cannot be directly converted into a numpy array since it is not continuous.");
    }

    selfDO->lockRead();

    /*if (selfDO->isT())
    {
        selfDO->unlock();
        selfDO->lockWrite();
        selfDO->evaluateTransposeFlag();
        selfDO->unlock();
        selfDO->lockRead();
    }*/

    PyObject *retDict = PyDict_New();
    item = PyLong_FromLong(3);
    PyDict_SetItemString(retDict, "version", item);
    Py_DECREF(item);

    int itemsize;
    char typekind;
    char typekind2[] = "a\0";

    //inter = new PyArrayInterface;
    //if (inter==NULL) {
    //    selfDO->unlock();
    //    return PyErr_NoMemory();
    //}

    //inter->two = 2;
    //inter->nd = selfDO->getDims();

    //if (inter->nd == 0)
    //{
    //    PyErr_SetString(PyExc_TypeError, "data object is empty.");
    //    delete inter;
    //    selfDO->unlock();
    //    return NULL;
    //}

    parseTypeNumber(selfDO->getType(), typekind, itemsize);
    typekind2[0]=typekind;

    PyObject *typestr = PyUnicode_FromFormat("|%s%d", &typekind2, itemsize);
    PyDict_SetItemString(retDict, "typestr", typestr);
    Py_XDECREF(typestr);

    //inter->flags = NPY_WRITEABLE | NPY_ALIGNED | NPY_NOTSWAPPED; //NPY_NOTSWAPPED indicates, that both data in opencv and data in numpy should have the same byteorder (Intel: little-endian)

    ////check if size and osize are totally equal, then set continuous flag
    //if (selfDO->getTotal() == selfDO->getOriginalTotal())
    //{
    //    inter->flags |= NPY_C_CONTIGUOUS;
    //}

    //inter->descr = NULL;
    //inter->data = NULL;
    //inter->shape = NULL;
    //inter->strides = NULL;

    if (selfDO->getDims() > 0)
    {
        unsigned int firstMDataIndex = selfDO->seekMat(0);
        int dims = selfDO->getDims();
        PyObject *shape = PyTuple_New(dims);
        PyObject *data = PyTuple_New(2);
        PyObject *strides = PyTuple_New(dims);
        npy_intp strides_iPlus1;

        bool isFullyContiguous = true;
        for (int i = 0; i < dims; i++)
        {
            if (selfDO->getSize(i) != selfDO->getOriginalSize(i)) isFullyContiguous = false;
        }


        PyTuple_SetItem(data,0, PyLong_FromVoidPtr((void*)((cv::Mat*)selfDO->get_mdata()[firstMDataIndex])->data));
        Py_INCREF(Py_False);
        PyTuple_SetItem(data,1, Py_False);


        //inter->shape = (npy_intp *)malloc(inter->nd * sizeof(npy_intp));
        //inter->strides = (npy_intp *)malloc(inter->nd * sizeof(npy_intp));

        //inter->shape[inter->nd - 1] = (npy_intp)selfDO->getSize(inter->nd - 1); //since transpose flag has been evaluated and is false now, everything is ok here
        PyTuple_SetItem(shape, dims-1, PyLong_FromLong(selfDO->getSize(dims-1)));
        strides_iPlus1 = itemsize;
        PyTuple_SetItem(strides, dims-1, PyLong_FromLong(itemsize));
        //inter->strides[inter->nd - 1] = inter->itemsize;
        for (int i = dims - 2; i >= 0; i--)
        {
            PyTuple_SetItem(shape, i, PyLong_FromLong(selfDO->getSize(i))); //since transpose flag has been evaluated and is false now, everything is ok here
            strides_iPlus1 = (strides_iPlus1 * selfDO->getOriginalSize(i+1));
            PyTuple_SetItem(strides, i, PyLong_FromLong(strides_iPlus1));

            //inter->shape[i] = (npy_intp)selfDO->getSize(i);
            //inter->strides[i] = inter->strides[i+1] * selfDO->getOriginalSize(i+1); //since transpose flag has been evaluated and is false now, everything is ok here
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

    selfDO->unlock();

    //don't icrement SELF here, since the receiver of the capsule (e.g. numpy-method) will increment the refcount of then PyDataObject SELF by itself.
    return retDict;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(dataObject_Array__doc,"__array__([dtype]) -> returns a numpy.ndarray from this dataObject. If possible a shallow copy is returned. \n\
                                   If no desired dtype is given and if the this dataObject is continuous, a ndarray sharing its memory with this dataObject is returned. \n\
                                   If the desired dtype does not fit to the type of this dataObject, a casted deep copy is returned. This is also the case if \n\
                                   this dataObject is not continuous. Then a continuous dataObject is created that is the base object of the returned ndarray.");
PyObject* PythonDataObject::PyDataObj_Array_(PyDataObject *self, PyObject *args)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    PyArray_Descr *newtype = NULL;
    PyArrayObject *newArray = NULL;

    if (!PyArg_ParseTuple(args, "|O&", PyArray_DescrConverter, &newtype)) 
    {
        Py_XDECREF(newtype);
        return NULL;
    }

    PyObject *item = NULL;

    ito::DataObject* selfDO = self->dataObject;

    selfDO->lockRead();

    /*if (selfDO->isT())
    {
        selfDO->unlock();
        selfDO->lockWrite();
        selfDO->evaluateTransposeFlag();
        selfDO->unlock();
        selfDO->lockRead();
    }*/

    if (selfDO->getContinuous()/* == true*/)
    {
        newArray = (PyArrayObject*)PyArray_FromStructInterface((PyObject*)self);
        selfDO->unlock();
    }
    else
    {
        PyDataObject *newDO = PythonDataObject::createEmptyPyDataObject();
        newDO->dataObject = new ito::DataObject(ito::makeContinuous(*selfDO));

        PyDataObject_SetBase(newDO, self->base);
        selfDO->unlock();

        newArray = (PyArrayObject*)PyArray_FromStructInterface((PyObject*)newDO);
        Py_DECREF(newDO);
    }

    if ((newtype == NULL) || PyArray_EquivTypes(PyArray_DESCR(newArray) /*->descr*/, newtype)) 
    {
        return (PyObject *)newArray;
    }
    else 
    {
        PyObject* ret = PyArray_CastToType(newArray, newtype, 0);
        Py_DECREF(newArray);
        return ret;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_Reduce(PyDataObject *self, PyObject * /*args*/)
{
    //version history:
    //21120:
    //  - each plane is stored as a bytearray in the data tuple (this needs 16bit for values bigger than 100 since it is transformed to an unicode value)
    //
    //21121:
    //  - each plane is now stored as a byte object, this can natively be pickled (faster, bytearray contains a reduce method)

    long version = 21121;

    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_NotImplementedError, "data object is NULL");
        return NULL;
    }

    self->dataObject->lockRead();

    int dims = self->dataObject->getDims();

    PyObject *sizeList = PyList_New(dims);
    for (int i = 0; i < dims; i++) PyList_SetItem(sizeList, i, Py_BuildValue("I", self->dataObject->getSize(i))); //since transpose flag has been evaluated and is false now, everything is ok here

    //1. elem -> callable object
    //2. elem -> arguments for init-method
    //3. elem -> state tuple (tuple1, tuple2,tuple3), tuple1 = (bool)transposed), tuple2 = size of dataObject->calcNumMats(), tuple3 = tuple with tags..., each element is of type ByteArray
    int vectorLength = self->dataObject->calcNumMats();

    PyObject *dataTuple = PyTuple_New(vectorLength);
    PyObject *byteArray = NULL;
    cv::Mat* tempMat;
    unsigned int seekNr;
    int sizeU = 0;
    int sizeV = 0;
    int elemSize = 0;
    char *dummy = 0;
    char *startingPoint = NULL;
    //int res;
    

    if (dims == 1)
    {
        sizeU = 1;
        sizeV = self->dataObject->getSize(dims-1); 
    }
    else if (dims > 1)
    {
        sizeU = self->dataObject->getSize(dims-2); 
        sizeV = self->dataObject->getSize(dims-1); 
    }

    if (version == 21120)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (int)tempMat->elemSize();

            //in version (checksum) 21120 the data has been stored as bytearray, which is reduced to a unicode and needs a lot of space
            byteArray = PyByteArray_FromStringAndSize(dummy,0);
            if (PyByteArray_Resize(byteArray, sizeV * sizeU * elemSize) != 0)
            {
                //err, message already set
                self->dataObject->unlock();
                Py_XDECREF(byteArray);
                Py_XDECREF(dataTuple);
                Py_XDECREF(sizeList);
                return NULL;
            }

            startingPoint = PyByteArray_AsString(byteArray);

            for (int row = 0; row < sizeU; row++)
            {
                if (memcpy((void*)startingPoint, (void*)(tempMat->ptr(row)), sizeV * elemSize) == NULL)
                {
                    self->dataObject->unlock();
                    Py_XDECREF(byteArray);
                    Py_XDECREF(dataTuple);
                    Py_XDECREF(sizeList);
                    PyErr_Format(PyExc_NotImplementedError, "memcpy failed. (index m_data-vector: %d, row-index: %d)", i, row);
                    return NULL;
                }
                startingPoint += (sizeV * elemSize); //move startingPoint by length (in byte) of one image row
            }

            PyTuple_SetItem(dataTuple, i, byteArray); //steals ref from byteArray
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

            //in version (checksum) 21120 the data has been stored as bytearray, which is reduced to a unicode and needs a lot of space
            byteArray = PyBytes_FromStringAndSize(NULL, sizeV * sizeU * elemSize);
            if (!byteArray /* || _PyBytes_Resize(&byteArray, sizeV * sizeU * elemSize) != 0 */)
            {
                //err, message already set
                self->dataObject->unlock();
                Py_XDECREF(byteArray);
                Py_XDECREF(dataTuple);
                Py_XDECREF(sizeList);
                return NULL;
            }

            startingPoint = PyBytes_AS_STRING(byteArray);

            for (int row = 0; row < sizeU; row++)
            {
                if (memcpy((void*)startingPoint, (void*)(tempMat->ptr(row)), sizeV * elemSize) == NULL)
                {
                    self->dataObject->unlock();
                    Py_XDECREF(byteArray);
                    Py_XDECREF(dataTuple);
                    Py_XDECREF(sizeList);
                    PyErr_Format(PyExc_NotImplementedError, "memcpy failed. (index m_data-vector: %d, row-index: %d)", i, row);
                    return NULL;
                }
                startingPoint += (sizeV * elemSize); //move startingPoint by length (in byte) of one image row
            }

            PyTuple_SetItem(dataTuple, i, byteArray); //steals ref from byteArray
            byteArray = NULL;
        }
    }


    //load tags
    PyObject *tagTuple = PyTuple_New(10);
    PyTuple_SetItem(tagTuple,0,PyLong_FromLong(version));

    PyObject *newTagDict = PyDataObject_getTagDict(self, NULL); //new ref
    PyObject *tempItem;
    PyObject *item = NULL;

    if (!PyErr_Occurred())
    {
        DataObject *dObj = self->dataObject;
        int tagSize = dObj->getTagListSize();
        //std::string tempString;
        DataObjectTagType tempTagValue;
        std::string tempKey;
        bool validOp;
        PyObject *tempTag;

        //1. tags (here it is bad to use the tags-getter, since this returns a dict_proxy, which cannot directly be pickled
        tempTag = PyDict_New();
        for (int i=0;i<tagSize;i++)
        {
            tempKey = dObj->getTagKey(i,validOp);
            if (validOp)
            {
                //tempString = dObj->getTag(tempKey, validOp);
                //if (validOp) PyDict_SetItem(tempTag, PyUnicode_FromString(tempKey.data()), PyUnicode_FromString(tempString.data()));
                dObj->getTagByIndex(i, tempKey, tempTagValue);
                if (tempTagValue.getType() == DataObjectTagType::typeDouble)
                {
                    item = PyFloat_FromDouble(tempTagValue.getVal_ToDouble());
                    PyDict_SetItemString(tempTag, tempKey.data(), item);
                    Py_DECREF(item);
                }
                else
                {
                    item = PythonQtConversion::QByteArrayToPyUnicodeSecure(tempTagValue.getVal_ToString().data());
                    PyDict_SetItemString(tempTag, tempKey.data(), item);
                    Py_DECREF(item);
                }
            }
        }
        PyTuple_SetItem(tagTuple,1,tempTag); //steals ref from tempTag

        //2. axisScales
        tempItem = PyDict_GetItemString(newTagDict, "axisScales"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,2,tempItem);//steals ref from tempItem

        //3. axisOffsets
        tempItem = PyDict_GetItemString(newTagDict, "axisOffsets"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,3,tempItem);//steals ref from tempItem

        //4. axisDescriptions
        tempItem = PyDict_GetItemString(newTagDict, "axisDescriptions"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,4,tempItem);//steals ref from tempItem

        //5. axisUnits
        tempItem = PyDict_GetItemString(newTagDict, "axisUnits"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,5,tempItem);//steals ref from tempItem

        //6. valueUnit
        tempItem = PyDict_GetItemString(newTagDict, "valueUnit"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,6,tempItem);//steals ref from tempItem

        //7. valueDescription
        tempItem = PyDict_GetItemString(newTagDict, "valueDescription"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,7,tempItem);//steals ref from tempItem

        //8.
        tempItem = PyDict_GetItemString(newTagDict, "valueOffset"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,8,tempItem);//steals ref from tempItem

        //9.
        tempItem = PyDict_GetItemString(newTagDict, "valueScale"); //borrowed
        Py_INCREF(tempItem);
        PyTuple_SetItem(tagTuple,9,tempItem);//steals ref from tempItem
    }

    Py_XDECREF(newTagDict);

    PyObject *stateTuple = Py_BuildValue("(bOO)", false /*self->dataObject->isT()*/, dataTuple, tagTuple);

    Py_DECREF(dataTuple);
    Py_DECREF(tagTuple);

    PyObject *tempOut = Py_BuildValue("(O(Osb)O)", Py_TYPE(self), sizeList, typeNumberToName(self->dataObject->getType()), self->dataObject->getContinuous(), stateTuple);

    Py_DECREF(sizeList);
    Py_DECREF(stateTuple);

    self->dataObject->unlock();

    return tempOut;

    //PyErr_SetString(PyExc_NotImplementedError, "pickling for dataObject not possible");
    //return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_SetState(PyDataObject *self, PyObject *args)
{
    //version history:
    // see log in PyDataObj_Reduce

    bool transpose = false;
    PyObject *dataTuple = NULL; //borrowed reference
    PyObject *tagTuple = NULL;  //borrowed reference
    PyObject *tempTag = NULL;   //borrowed reference
    long version = 21120; //this is the first version, current is 21121

    if (!PyArg_ParseTuple(args,"(bO!O!)", &transpose, &PyTuple_Type, &dataTuple, &PyTuple_Type, &tagTuple))
    {
        PyErr_Clear();
        //test if maybe no tagTuple is available
        tagTuple = NULL;
        if (!PyArg_ParseTuple(args,"(bO!)", &transpose, &PyTuple_Type, &dataTuple))
        {
            PyErr_SetString(PyExc_NotImplementedError, "unpickling for dataObject not possible since state vector is invalid");
            return NULL;
        }
    }

    //pre-check tags
    if (tagTuple != NULL)
    {
        if (PyTuple_Size(tagTuple) != 10)
        {
            //Py_XDECREF(dataTuple);
            //Py_XDECREF(tagTuple);
            PyErr_SetString(PyExc_NotImplementedError, "tags in pickled data object does not have the required number of elements (10)");
            return NULL;
        }
        else
        {
            tempTag = PyTuple_GetItem(tagTuple,0); //borrowed ref
            if (!PyLong_Check(tempTag))
            {
                //Py_XDECREF(dataTuple);
                //Py_XDECREF(tagTuple);
                PyErr_SetString(PyExc_NotImplementedError, "first element in tag tuple must be an integer number, which it is not.");
                return NULL;
            }

            version = PyLong_AsLong(tempTag);
            if (version != 21120 && version != 21121)
            {
                //Py_XDECREF(dataTuple);
                //Py_XDECREF(tagTuple);
                PyErr_SetString(PyExc_NotImplementedError, "first element in tag tuple is a check sum and does not have the right value.");
                return NULL;
            }
        }
    }

    if (transpose == true)
    {
        //Py_XDECREF(dataTuple);
        //Py_XDECREF(tagTuple);
        PyErr_SetString(PyExc_NotImplementedError, "transpose flag of unpickled data must be false (since the transposition has been evaluated before pickling). Transpose flag is obsolete now.");
        return NULL;
    }

    if (self->dataObject == NULL)
    {
        //Py_XDECREF(dataTuple);
        //Py_XDECREF(tagTuple);
        PyErr_SetString(PyExc_NotImplementedError, "unpickling for dataObject failed");
        return NULL;
    }

    self->dataObject->lockWrite();

    int vectorLength = self->dataObject->calcNumMats();

    if (PyTuple_Size(dataTuple) != vectorLength)
    {
        self->dataObject->unlock();
        //Py_XDECREF(dataTuple);
        //Py_XDECREF(tagTuple);
        PyErr_SetString(PyExc_NotImplementedError, "unpickling for dataObject failed since data dimensions does not fit");
        return NULL;
    }

    int dims = self->dataObject->getDims();
    PyObject *byteArray = NULL;
    cv::Mat* tempMat;
    unsigned int seekNr;
    int sizeU = 0;
    int sizeV = 0;
    uchar* startPtr = NULL;
    char* byteArrayContent = NULL;
    int elemSize = 0;
    std::string tempString;
    std::string keyString;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    PyObject *seqItem = NULL;

    if (dims == 1)
    {
        sizeU = 1;
        sizeV = self->dataObject->getSize(dims-1); 
    }
    else if (dims > 1)
    {
        sizeU = self->dataObject->getSize(dims-2); 
        sizeV = self->dataObject->getSize(dims-1); 
    }

    if (version == 21120)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (int)tempMat->elemSize();
            startPtr = tempMat->ptr(0); //mat is continuous!!! (should be;))
            byteArray = PyTuple_GetItem(dataTuple, i); //borrowed ref

            byteArrayContent = PyByteArray_AsString(byteArray); //borrowed ref
            memcpy((void*)startPtr, (void*)byteArrayContent, sizeU*sizeV*elemSize);
        }
    }
    else if (version == 21121)
    {
        for (int i = 0; i < vectorLength; i++)
        {
            seekNr = self->dataObject->seekMat(i);
            tempMat = (cv::Mat*)(self->dataObject->get_mdata()[seekNr]);
            elemSize = (int)tempMat->elemSize();
            startPtr = tempMat->ptr(0); //mat is continuous!!! (should be;))
            byteArray = PyTuple_GetItem(dataTuple, i); //borrowed ref

            byteArrayContent = PyBytes_AsString(byteArray); //borrowed ref
            memcpy((void*)startPtr, (void*)byteArrayContent, sizeU*sizeV*elemSize);
        }
    }

    //transpose must be false (checked above)

    //check tags
    if (tagTuple != NULL && PyTuple_Size(tagTuple) == 10)
    {
        //1. tags
        tempTag = PyTuple_GetItem(tagTuple,1); //borrowed
        if (PyDict_Check(tempTag))
        {
            while (PyDict_Next(tempTag, &pos, &key, &value))
            {
                if (parsePyObject2StdString(key, keyString) == 0)
                {
                    if (PyFloat_Check(value)||PyLong_Check(value))
                    {
                        self->dataObject->setTag(keyString, PyFloat_AsDouble(value));
                    }
                    else if (parsePyObject2StdString(value, tempString) == 0)
                    {
                        self->dataObject->setTag(keyString, tempString);
                    }
                }
            }
        }

        //2. axisScales
        tempTag = PyTuple_GetItem(tagTuple,2);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i=0;i<PySequence_Size(tempTag);i++)
            {
                seqItem = PySequence_GetItem(tempTag,i); //new reference
                self->dataObject->setAxisScale(i, PyFloat_AsDouble(seqItem));
                Py_XDECREF(seqItem);
            }
        }

        //3. axisOffsets
        tempTag = PyTuple_GetItem(tagTuple,3);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i=0;i<PySequence_Size(tempTag);i++)
            {
                seqItem = PySequence_GetItem(tempTag,i); //new reference
                self->dataObject->setAxisOffset(i, PyFloat_AsDouble(seqItem));
                Py_XDECREF(seqItem);
            }
        }

        // 4. axisDescriptions
        tempTag = PyTuple_GetItem(tagTuple,4);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i=0;i<PySequence_Size(tempTag);i++)
            {
                seqItem = PySequence_GetItem(tempTag,i); //new reference
                if (parsePyObject2StdString(seqItem, tempString) == 0)
                {
                    self->dataObject->setAxisDescription(i, tempString);
                }
                Py_XDECREF(seqItem);
            }
        }

        // 5. axisUnits
        tempTag = PyTuple_GetItem(tagTuple,5);
        if (PySequence_Check(tempTag))
        {
            for (Py_ssize_t i=0;i<PySequence_Size(tempTag);i++)
            {
                seqItem = PySequence_GetItem(tempTag,i); //new reference
                if (parsePyObject2StdString(seqItem, tempString) == 0)
                {
                    self->dataObject->setAxisUnit(i, tempString);
                }
                Py_XDECREF(seqItem);
            }
        }

        // 6. valueUnit
        tempTag = PyTuple_GetItem(tagTuple,6);
        if (parsePyObject2StdString(tempTag, tempString) == 0)
        {
            self->dataObject->setValueUnit(tempString);
        }

        // 7. valueDescription
        tempTag = PyTuple_GetItem(tagTuple,7);
        if (PythonDataObject::parsePyObject2StdString(tempTag, tempString) == 0)
        {
            self->dataObject->setValueDescription(tempString);
        }

        // 8.
        //tempTag = PyTuple_GetItem(tagTuple,8);
        // 9.
        //tempTag = PyTuple_GetItem(tagTuple,9);
    }

    self->dataObject->unlock();

    //Py_XDECREF(dataTuple);
    //Py_XDECREF(tagTuple);

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectToList_doc, "tolist() -> return the data object as a (possibly nested) list\n\
\n\
This method returns a nested list with all values of this data object. The recursion level of this nested list \
corresponds to the number of dimensions. The outer list corresponds to the first dimension. \n\
\n\
Returns \n\
------- \n\
y : {list} \n\
    Nested list with values of data object (int, float or complex depending on type of data object)");
PyObject* PythonDataObject::PyDataObj_ToList(PyDataObject *self)
{
    if (self->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    ito::DataObject *d = self->dataObject;

    PyObject *result = NULL;

    unsigned int *iter = new unsigned int[d->getDims()];
    for (int i = 0; i < d->getDims(); i++)
    {
        iter[i] = 0;
    }

    result = PyDataObj_ToListRecursive(d, iter, 0);

    delete[] iter;

    return result;
    
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_ToListRecursive(ito::DataObject *dataObj, unsigned int *currentIdx, int iterationIndex)
{
    if (dataObj == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    PyObject *temp = NULL;

    if ((int)iterationIndex == dataObj->getDims() - 1) //last index
    {
        int len = dataObj->getSize(iterationIndex);
        PyObject *result = PyList_New(len);
        for (int i = 0; i < len; i++)
        {
            currentIdx[iterationIndex] = i;
            temp = PyDataObj_At(dataObj, currentIdx);
            if (temp == NULL) return NULL;
            PyList_SetItem(result, i, temp);
        }
        return result;
    }
    else if ((int)iterationIndex < dataObj->getDims() - 1) //previous indexes (besides last one)
    {
        int len = dataObj->getSize(iterationIndex);
        PyObject *result = PyList_New(len);
        for (int i = 0; i < len; i++)
        {
            currentIdx[iterationIndex] = i;
            temp = PyDataObj_ToListRecursive(dataObj, currentIdx, iterationIndex + 1);
            if (temp == NULL) return NULL;
            PyList_SetItem(result, i, temp);
        }
        return result;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "iterationIndex is bigger than dimensions of data object");
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_At(ito::DataObject *dataObj, unsigned int *idx)
{
    if (dataObj == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    switch(dataObj->getType())
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
    case ito::tRGBA32:
        {
            ito::PythonRgba::PyRgba *color = ito::PythonRgba::createEmptyPyRgba();
            if (color) color->rgba = dataObj->at<ito::Rgba32>(idx).rgba;
            return (PyObject*)color;
        }
    case ito::tFloat32:
        return PyFloat_FromDouble(dataObj->at<float32>(idx));
    case ito::tFloat64:
        return PyFloat_FromDouble(dataObj->at<float64>(idx));
    case ito::tComplex64:
        {
        ito::complex64 value = dataObj->at<complex64>(idx);
        return PyComplex_FromDoubles(value.real(),value.imag());
        }
    case ito::tComplex128:
        {
        ito::complex128 value = dataObj->at<complex128>(idx);
        return PyComplex_FromDoubles(value.real(),value.imag());
        }
    default:
        PyErr_SetString(PyExc_TypeError, "type of data object not supported");
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObj_At(ito::DataObject *dataObj, int continuousIdx)
{
    if (dataObj == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is NULL");
        return NULL;
    }

    if (continuousIdx >= dataObj->getTotal())
    {
        PyErr_SetString(PyExc_TypeError, "continuous index is out of range.");
        return NULL;
    }

    int dims = dataObj->getDims();
    int planeSize = dataObj->getSize(dims - 1) * dataObj->getSize(dims - 2);
    int planeIdx = continuousIdx % planeSize;
    int col = planeIdx % dataObj->getSize(dims - 1);
    int row = (planeIdx - col) / dataObj->getSize(dims - 1);
    int mat = (continuousIdx - planeIdx) / planeSize;
    mat = dataObj->seekMat(mat);

    cv::Mat* m = (cv::Mat*)dataObj->get_mdata()[mat];

    switch(dataObj->getType())
    {
    case ito::tUInt8:
        return PyLong_FromUnsignedLong(m->at<uint8>(row,col));
    case ito::tInt8:
        return PyLong_FromLong(m->at<int8>(row,col));
    case ito::tUInt16:
        return PyLong_FromUnsignedLong(m->at<uint16>(row,col));
    case ito::tInt16:
        return PyLong_FromLong(m->at<int16>(row,col));
    case ito::tUInt32:
        return PyLong_FromUnsignedLong(m->at<uint32>(row,col));
    case ito::tInt32:
        return PyLong_FromLong(m->at<int32>(row,col));
    case ito::tRGBA32:
        {
            ito::PythonRgba::PyRgba *color = ito::PythonRgba::createEmptyPyRgba();
            if (color) color->rgba = m->at<Rgba32>(row,col).rgba;
            return (PyObject*)color;
        }
    case ito::tFloat32:
        return PyFloat_FromDouble(m->at<float32>(row,col));
    case ito::tFloat64:
        return PyFloat_FromDouble(m->at<float64>(row,col));
    case ito::tComplex64:
        {
        ito::complex64 value = (m->at<complex64>(row,col));
        return PyComplex_FromDoubles(value.real(),value.imag());
        }
    case ito::tComplex128:
        {
        ito::complex128 value = (m->at<complex128>(row,col));
        return PyComplex_FromDoubles(value.real(),value.imag());
        }
    default:
        PyErr_SetString(PyExc_TypeError, "type of data object not supported");
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonDataObject::PyDataObj_Capsule_Destructor(PyObject* capsule)
{
    PyArrayInterface *inter = (PyArrayInterface*)PyCapsule_GetPointer(capsule, NULL);

    if (inter != NULL)
    {
        free(inter->shape);
        free(inter->strides);
    }

    DELETE_AND_SET_NULL(inter);
}

//PyObject* PythonDataObject::PyDataObj_StaticArange(PyDataObject *self, PyObject *args)
//{
//    return PyObject_Call((PyObject*)&PyDataObjectType, NULL, NULL);
//}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticZeros_doc,"zeros(dims [, dtype='uint8'[, continuous = 0]]) -> creates new dataObject filled with zeros.  \n\
\n\
Static method for creating a new n-dimensional itom.dataObject with given number of dimensions and dtype, filled with zeros. \n\
\n\
Parameters \n\
----------- \n\
dims : {integer list} \n\
    'dims' is list indicating the size of each dimension, e.g. [2,3] is a matrix with 2 rows and 3 columns\n\
dtype : {str}, optional \n\
    'dtype' is the data type of each element, possible values: 'int8','uint8',...,'int32','float32','float64','complex64','complex128', 'rgba32'\n\
continuous : {str}, optional \n\
    'continuous' [0|1] defines whether the data block should be continuously allocated in memory [1] or in different smaller blocks [0] (recommended for huge matrices).\n\
\n\
Returns \n\
------- \n\
I : {dataObject} of shape (size,size)\n\
    An array where all elements are equal to zero. \n\
\n\
See Also \n\
--------- \n\
eye: method for creating an eye matrix \n\
ones: method for creating a matrix filled with ones \n\
\n\
Notes \n\
------ \n\
For color-types (rgba32) every item / cell will be black and transparent: [r=0 g=0 b=0 a=0].");
PyObject* PythonDataObject::PyDataObj_StaticZeros(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    int typeno = 0;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError()) return NULL;

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int *sizes2 = new int[sizes.size()];
        for (unsigned int i = 0; i < sizes.size(); i++)
            sizes2[i] = sizes[i];
        //no lock is necessary since eye is allocating the data block and no other access is possible at this moment
        selfDO->dataObject->zeros(sizes.size(), sizes2, typeno, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticOnes_doc,"ones(dims [, dtype='uint8'[, continuous = 0]]) -> creates new dataObject filled with ones.  \n\
\n\
Static method for creating a new n-dimensional itom.dataObject with given number of dimensions and dtype, filled with ones. \n\
\n\
Parameters \n\
----------- \n\
dims : {integer list} \n\
    'dims' is list indicating the size of each dimension, e.g. [2,3] is a matrix with 2 rows and 3 columns\n\
dtype : {str}, optional \n\
    'dtype' is the data type of each element, possible values: 'int8','uint8',...,'int32','float32','float64','complex64','complex128', 'rgba32'\n\
continuous : {str}, optional \n\
    'continuous' [0|1] defines whether the data block should be continuously allocated in memory [1] or in different smaller blocks [0] (recommended for huge matrices).\n\
\n\
Returns \n\
------- \n\
I : {dataObject} of shape (size,size)\n\
    An array where all elements are equal to one. \n\
\n\
See Also \n\
--------- \n\
eye: method for creating an eye matrix \n\
zeros: method for creating a matrix filled with zeros \n\
\n\
Notes \n\
------ \n\
For color-types (rgba32) every item / cell will be white: [r=255 g=255 b=255 a=255].");
PyObject* PythonDataObject::PyDataObj_StaticOnes(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    int typeno = 0;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError()) return NULL;
    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(PyExc_TypeError, "Type uint32 currently not supported due to incompability to openCV!");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int *sizes2 = new int[sizes.size()];
        for (unsigned int i = 0; i < sizes.size(); i++)
            sizes2[i]=sizes[i];
        //no lock is necessary since eye is allocating the data block and no other access is possible at this moment
        selfDO->dataObject->ones(sizes.size(), sizes2, typeno, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticRand_doc,"rand([dims [, dtype='uint8'[, continuous = 0]]]) -> creates new dataObject filled with uniform distributed random values.  \n\
\n\
Static method to create a new itom.dataObject filled with uniform distributed random numbers.\n\
In case of an integer type, the uniform noise is from min<ObjectType>(inclusiv) to max<ObjectType>(inclusiv).\n\
For floating point types, the noise is between 0(inclusiv) and 1(exclusiv). \n\
\n\
Parameters \n\
----------- \n\
dims : {integer list} \n\
    'dims' is list indicating the size of each dimension, e.g. [2,3] is a matrix with 2 rows and 3 columns.\n\
dtype : {str}, optional \n\
    'dtype' is the data type of each element, possible values: 'int8','uint8',...,'int32','float32','float64','complex64','complex128'\n\
continuous : {str}, optional \n\
    'continuous' [0|1] defines whether the data block should be continuously allocated in memory [1] or in different smaller blocks [0] (recommended for huge matrices).\n\
\n\
Returns \n\
------- \n\
out : {dataObject} \n\
    Array of random numbers with the given dimensions, dtype. \n\
\n\
See Also \n\
--------- \n\
randN: method for creating a matrix filled with gaussianly distributed values");
PyObject* PythonDataObject::PyDataObj_StaticRand(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    int typeno = 0;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError()) return NULL;
    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(PyExc_TypeError, "Type uint32 currently not supported due to incompability to openCV!");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int *sizes2 = new int[sizes.size()];
        for (unsigned int i = 0; i < sizes.size(); i++)
            sizes2[i]=sizes[i];

        //no lock is necessary since eye is allocating the data block and no other access is possible at this moment
        selfDO->dataObject->rand(sizes.size(),sizes2, typeno, false, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticRandN_doc,"randN(dims [, dtype='uint8'[, continuous = 0]]) -> creates dataObject filled with gaussian distributed random values.  \n\
\n\
Static method to create a new itom.dataObject filled with gaussian distributed random numbers. \n\
In case of an integer type, the gausian noise mean value is (max+min)/2.0 and the standard deviation is (max-min/)6.0 to max. \n\
For floating point types, the noise mean value is 0 and the standard deviation is 1.0/3.0. \n\
\n\
Parameters \n\
----------- \n\
dims : {integer list} \n\
    'dims' is list indicating the size of each dimension, e.g. [2,3] is a matrix with 2 rows and 3 columns.\n\
dtype : {str}, optional \n\
    'dtype' is the data type of each element, possible values: 'int8','uint8',...,'int32', 'float32','float64','complex64','complex128'\n\
continuous : {str}, optional \n\
    'continuous' [0|1] defines whether the data block should be continuously allocated in memory [1] or in different smaller blocks [0] (recommended for huge matrices).\n\
\n\
Returns \n\
------- \n\
out : {dataObject} \n\
    Array of random numbers with the given dimensions, dtype. \n\
\n\
See Also \n\
--------- \n\
rand: method for creating a matrix filled with unformly distributed values");
PyObject* PythonDataObject::PyDataObj_StaticRandN(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    int typeno = 0;
    std::vector<unsigned int> sizes;
    sizes.clear();
    unsigned char continuous = 0;

    RetVal retValue = PyDataObj_ParseCreateArgs(args, kwds, typeno, sizes, continuous);

    if (retValue.containsError()) return NULL;
    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(PyExc_TypeError, "Type uint32 currently not supported due to incompability to openCV!");
        return NULL;
    }

    PyDataObject* selfDO = createEmptyPyDataObject();
    selfDO->dataObject = new ito::DataObject();

    if (selfDO->dataObject != NULL)
    {
        int *sizes2 = new int[sizes.size()];
        for (unsigned int i = 0; i < sizes.size(); i++)
            sizes2[i]=sizes[i];

        //no lock is necessary since eye is allocating the data block and no other access is possible at this moment
        selfDO->dataObject->rand(sizes.size(),sizes2, typeno, true, continuous);
        DELETE_AND_SET_NULL_ARRAY(sizes2);
    }

    sizes.clear();

    return (PyObject*)selfDO;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectStaticEye_doc,"eye(size [, dtype='uint8']) -> creates a 2D, square, eye-matrix.\n\
\n\
Static method for creating a two-dimensional, square, eye-matrix of type itom.dataObject. \n\
\n\
Parameters \n\
----------- \n\
size : {int}, \n\
    the size of the square matrix (single value)\n\
dtype : {str}, optional \n\
    'dtype' is the data type of each element, possible values: 'int8','uint8',...,'int32','uint32','float32','float64','complex64','complex128'\n\
\n\
Returns \n\
------- \n\
I : {dataObject} of shape (size,size)\n\
    An array where all elements are equal to zero, except for the 'k-th diagonal, whose values are equal to one. \n\
\n\
See Also \n\
--------- \n\
ones: method for creating a matrix filled with ones \n\
zeros: method for creating a matrix filled with zeros");
PyObject* PythonDataObject::PyDataObj_StaticEye(PyObject * /*self*/, PyObject *args /*, PyObject *kwds*/)
{
    //static const char *kwlist[] = {"size","dtype","continuous", NULL};

    int length = PyTuple_Size(args);

    const char *type = 0;
    int size = 0;
    int typeno = 0;

    type = typeNumberToName(typeno);
    if (strlen(type) < 4)
    {
//        type = "uint8\0";
        type = "uint8";
    }

    RetVal retValue(retOk);


    if (length < 1)
    {
        PyErr_Format(PyExc_TypeError, "Argument of Eye-Method must be: int size [, char dtype (optional)]");
        return NULL;
    }
    else if (length == 1)
    {
        if (!PyArg_ParseTuple(args, "i", &size))
        {
            PyErr_Format(PyExc_TypeError, "Argument of Eye-Method must be: int size [, char dtype (optional)]");
            return NULL;
        }
    }
    else if (length == 2)
    {
        if (!PyArg_ParseTuple(args, "is", &size, &type))
        {
            PyErr_Format(PyExc_TypeError, "Argument of Eye-Method must be: int size [, char dtype (optional)]");
            return NULL;
        }
    }
    else
    {
        PyErr_Format(PyExc_TypeError,"Argument of Eye-Method must be: int size [, char dtype (optional)]");
        return NULL;
    }

    typeno = typeNameToNumber(type);

    if (typeno == ito::tUInt32)
    {
        PyErr_SetString(PyExc_TypeError, "Type uint32 currently not supported due to incompability to openCV!");
        return NULL;
    }

    if (typeno >= 0)
    {
        if (size > 0)
        {
            PyDataObject* selfDO = createEmptyPyDataObject();
            selfDO->dataObject = new ito::DataObject();
            //no lock is necessary since eye is allocating the data block and no other access is possible at this moment
            selfDO->dataObject->eye(size, typeno);
            return (PyObject*)selfDO;
        }
        else
        {
            PyErr_Format(PyExc_TypeError,"size must be bigger than zero.");
            return NULL;
        }
    }
    else
    {
        PyErr_Format(PyExc_TypeError,"unknown dtype");
        return NULL;
    }


}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonDataObject::PyDataObject_methods[] = {
        {"name", (PyCFunction)PythonDataObject::PyDataObject_name, METH_NOARGS, pyDataObjectName_doc},
        {"data", (PyCFunction)PythonDataObject::PyDataObject_data, METH_NOARGS, pyDataObjectData_doc},

        {"setAxisOffset",(PyCFunction)PyDataObj_SetAxisOffset, METH_VARARGS, pyDataObjectSetAxisOffset_doc},
        {"setAxisScale",(PyCFunction)PyDataObj_SetAxisScale, METH_VARARGS, pyDataObjectSetAxisScale_doc},
        {"setAxisDescription",(PyCFunction)PyDataObj_SetAxisDescription, METH_VARARGS, pyDataObjectSetAxisDescription_doc},
        {"setAxisUnit",(PyCFunction)PyDataObj_SetAxisUnit, METH_VARARGS, pyDataObjectSetAxisUnit_doc},
        {"setTag",(PyCFunction)PyDataObj_SetTag, METH_VARARGS, pyDataObjectSetTag_doc},
        {"deleteTag",(PyCFunction)PyDataObj_DeleteTag, METH_VARARGS, pyDataObjectDeleteTag_doc},
        {"existTag",(PyCFunction)PyDataObj_TagExists, METH_VARARGS, pyDataObjectTagExists_doc},
        {"getTagListSize",(PyCFunction)PyDataObj_GetTagListSize, METH_NOARGS, pyDataObjectGetTagListSize_doc},
        {"addToProtocol",(PyCFunction)PyDataObj_AddToProtocol, METH_VARARGS, pyDataObjectAddToProtocol_doc},
        
        {"copy",(PyCFunction)PythonDataObject::PyDataObject_copy, METH_VARARGS, pyDataObjectCopy_doc},
        {"astype", (PyCFunction)PythonDataObject::PyDataObject_astype, METH_VARARGS | METH_KEYWORDS, pyDataObjectAstype_doc},
        {"normalize", (PyCFunction)PythonDataObject::PyDataObject_normalize, METH_VARARGS | METH_KEYWORDS, pyDataObjectNormalize_doc},
        {"locateROI", (PyCFunction)PythonDataObject::PyDataObject_locateROI, METH_NOARGS, pyDataObjectLocateROI_doc},
        {"adjustROI", (PyCFunction)PythonDataObject::PyDataObject_adjustROI, METH_VARARGS, pyDataObjectAdjustROI_doc},
        {"squeeze", (PyCFunction)PythonDataObject::PyDataObject_squeeze, METH_NOARGS, pyDataObjectSqueeze_doc},
        {"size", (PyCFunction)PythonDataObject::PyDataObject_size, METH_VARARGS, pyDataObjectSize_doc},
        {"conj", (PyCFunction)PythonDataObject::PyDataObject_conj, METH_NOARGS, pyDataObjectConj_doc},
        {"conjugate", (PyCFunction)PythonDataObject::PyDataObject_conjugate, METH_NOARGS, pyDataObjectConjugate_doc},
        {"adj", (PyCFunction)PythonDataObject::PyDataObject_adj, METH_NOARGS, pyDataObjectAdj_doc},
        {"adjugate", (PyCFunction)PyDataObject_adjugate, METH_NOARGS, pyDataObjectAdjugate_doc}, 
        {"trans", (PyCFunction)PythonDataObject::PyDataObject_trans, METH_NOARGS, pyDataObjectTrans_doc},
        {"div", (PyCFunction)PythonDataObject::PyDataObject_div, METH_VARARGS, pyDataObjectDiv_doc},
        {"mul", (PyCFunction)PythonDataObject::PyDataObject_mul, METH_VARARGS, pyDataObjectMul_doc},
        {"makeContinuous", (PyCFunction)PythonDataObject::PyDataObject_makeContinuous, METH_NOARGS, pyDataObjectMakeContinuous_doc},
        {"reshape", (PyCFunction)PythonDataObject::PyDataObject_reshape, METH_VARARGS, pyDataObjectReshape_doc},
        {"zeros", (PyCFunction)PythonDataObject::PyDataObj_StaticZeros, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyDataObjectStaticZeros_doc},
        {"ones",(PyCFunction)PythonDataObject::PyDataObj_StaticOnes, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyDataObjectStaticOnes_doc},
        {"rand",(PyCFunction)PythonDataObject::PyDataObj_StaticRand, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyDataObjectStaticRand_doc},
        {"randN",(PyCFunction)PythonDataObject::PyDataObj_StaticRandN, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyDataObjectStaticRandN_doc},
        {"eye",(PyCFunction)PythonDataObject::PyDataObj_StaticEye, METH_VARARGS | METH_STATIC, pyDataObjectStaticEye_doc},
        {"__reduce__", (PyCFunction)PythonDataObject::PyDataObj_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
        {"__setstate__", (PyCFunction)PythonDataObject::PyDataObj_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
        {"__array__", (PyCFunction)PythonDataObject::PyDataObj_Array_, METH_VARARGS, dataObject_Array__doc},

        {"tolist", (PyCFunction)PythonDataObject::PyDataObj_ToList, METH_NOARGS, pyDataObjectToList_doc}, //"returns nested list of content of data object"
        {NULL}  /* Sentinel */
    };

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonDataObject::PyDataObject_members[] = {
        {"base", T_OBJECT, offsetof(PyDataObject, base), READONLY, "base object"}, 
        {NULL}  /* Sentinel */
    };

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonDataObject::PyDataObjectModule = {
        PyModuleDef_HEAD_INIT,
        "dataObject",
        "itom DataObject type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
    };

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonDataObject::PyDataObject_getseters[] = {
    {"dims", (getter)PyDataObj_GetDims, NULL, dataObjectAttrDims_doc, NULL},
    {"ndim", (getter)PyDataObj_GetDims, NULL, dataObjectAttrDims_doc, NULL},
    {"dtype", (getter)PyDataObj_GetType, NULL, dataObjectAttrType_doc, NULL},
    {"shape", (getter)PyDataObj_GetShape, NULL, dataObjectAttrShape_doc, NULL},
    {"continuous", (getter)PyDataObj_GetContinuous, NULL, dataObjectAttrContinuous_doc, NULL},
    {"metaDict", (getter)PyDataObject_getTagDict, NULL, dataObjectAttrTagDict_doc, NULL},

    {"tags", (getter)PyDataObject_getTags, (setter)PyDataObject_setTags, dataObjectAttrTags_doc, NULL},
    {"axisScales", (getter)PyDataObject_getAxisScales, (setter)PyDataObject_setAxisScales, dataObjectAttrAxisScales_doc, NULL},
    {"axisOffsets", (getter)PyDataObject_getAxisOffsets, (setter)PyDataObject_setAxisOffsets, dataObjectAttrAxisOffsets_doc, NULL},
    {"axisDescriptions", (getter)PyDataObject_getAxisDescriptions, (setter)PyDataObject_setAxisDescriptions, dataObjectAttrAxisDescriptions_doc, NULL},
    {"axisUnits", (getter)PyDataObject_getAxisUnits, (setter)PyDataObject_setAxisUnits, dataObjectAttrAxisUnits_doc, NULL},
    {"valueUnit", (getter)PyDataObject_getValueUnit, (setter)PyDataObject_setValueUnit, dataObjectAttrValueUnit_doc, NULL},
    {"valueDescription", (getter)PyDataObject_getValueDescription, (setter)PyDataObject_setValueDescription, dataObjectAttrValueDescription_doc, NULL},
    {"valueScale", (getter)PyDataObject_getValueScale, NULL, dataObjectAttrValueScale_doc, NULL},
    {"valueOffset", (getter)PyDataObject_getValueOffset, NULL, dataObjectAttrValueOffset_doc, NULL},
    {"value", (getter)PyDataObject_getValue, (setter)PyDataObject_setValue, dataObjectAttrValue_doc, NULL},
    {"xyRotationalMatrix", (getter)PyDataObject_getXYRotationalMatrix, (setter)PyDataObject_setXYRotationalMatrix, dataObjectAttrRotationalMatrix_doc, NULL},

    {"__array_struct__", (getter)PyDataObj_Array_StructGet, NULL, dataObjectArray_StructGet_doc, NULL},
    {"__array_interface__", (getter)PyDataObj_Array_Interface, NULL, dataObjectArray_Interface_doc ,NULL},
    
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonDataObject::PyDataObjectType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.dataObject",             /* tp_name */
        sizeof(PyDataObject),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyDataObject_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyDataObject_repr,         /* tp_repr */
        &PyDataObject_numberProtocol,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        &PyDataObject_mappingProtocol,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        dataObjectInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        (richcmpfunc)PyDataObject_RichCompare,            /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        (getiterfunc)PyDataObj_getiter,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyDataObject_methods,             /* tp_methods */
        PyDataObject_members,             /* tp_members */
        PyDataObject_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PythonDataObject::PyDataObject_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyDataObject_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
    };

//----------------------------------------------------------------------------------------------------------------------------------
PyNumberMethods PythonDataObject::PyDataObject_numberProtocol = {
    (binaryfunc)PyDataObj_nbAdd,                    /* nb_add */
    (binaryfunc)PyDataObj_nbSubtract,                    /* nb_subtract */
    (binaryfunc)PyDataObj_nbMultiply,                    /* nb_multiply */
    (binaryfunc)PyDataObj_nbRemainder,              /* nb_remainder */
    (binaryfunc)PyDataObj_nbDivmod,                 /* nb_divmod */
    (ternaryfunc)PyDataObj_nbPower,                   /* nb_power */
    (unaryfunc)PyDataObj_nbNegative,                     /* nb_negative */
    (unaryfunc)PyDataObj_nbPositive,                     /* nb_positive */
    (unaryfunc)PyDataObj_nbAbsolute,                     /* nb_absolute */
    (inquiry)0,                      /* nb_bool */
    (unaryfunc)PyDataObj_nbInvert,                                          /* nb_invert */
    (binaryfunc)PyDataObj_nbLshift,                                          /* nb_lshift */
    (binaryfunc)PyDataObj_nbRshift,                                          /* nb_rshift */
    (binaryfunc)PyDataObj_nbAnd,                                          /* nb_and */
    (binaryfunc)PyDataObj_nbXor,                                          /* nb_xor */
    (binaryfunc)PyDataObj_nbOr,                                          /* nb_or */
    0,                                /* nb_int */
    0,                                          /* nb_reserved */
    0,                              /* nb_float */
    (binaryfunc)PyDataObj_nbInplaceAdd,                                          /* nb_inplace_add */
    (binaryfunc)PyDataObj_nbInplaceSubtract,                                          /* nb_inplace_subtract */
    (binaryfunc)PyDataObj_nbInplaceMultiply,                                          /* nb_inplace_multiply*/
    (binaryfunc)PyDataObj_nbInplaceRemainder,                                          /* nb_inplace_remainder */
    (ternaryfunc)PyDataObj_nbInplacePower,                                          /* nb_inplace_power */
    (binaryfunc)PyDataObj_nbInplaceLshift,                                          /* nb_inplace_lshift */
    (binaryfunc)PyDataObj_nbInplaceRshift,                                          /* nb_inplace_rshift */
    (binaryfunc)PyDataObj_nbInplaceAnd,                                          /* nb_inplace_and */
    (binaryfunc)PyDataObj_nbInplaceXor,                                          /* nb_inplace_xor */
    (binaryfunc)PyDataObj_nbInplaceOr,                                          /* nb_inplace_or */
    (binaryfunc)0,                /* nb_floor_divide */
    (binaryfunc)0,                    /* nb_true_divide */
    0,                                          /* nb_inplace_floor_divide */
    0,                                          /* nb_inplace_true_divide */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMappingMethods PythonDataObject::PyDataObject_mappingProtocol = {
    (lenfunc)PyDataObj_mappingLength,
    (binaryfunc)PyDataObj_mappingGetElem,
    (objobjargproc)PyDataObj_mappingSetElem
};

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObjectIter_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyObject *dataObject = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PythonDataObject::PyDataObjectType, &dataObject))
    {
        return NULL;
    }

    PyDataObjectIter* self = (PyDataObjectIter *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        PythonDataObject::PyDataObject *dobj = (PyDataObject*)dataObject;
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

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonDataObject::PyDataObjectIter_init(PyDataObjectIter* /*self*/, PyObject* /*args*/, PyObject* /*kwds*/)
{
    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PythonDataObject::PyDataObjectIter_dealloc(PyDataObjectIter *self)
{
    self->it = ito::DObjConstIterator();
    self->itEnd = self->it;
    Py_XDECREF(self->base);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonDataObject::PyDataObjectIter_iternext(PyDataObjectIter* self)
{
    if (self->it == self->itEnd)
    {
        PyErr_SetString(PyExc_StopIteration, "");
        return NULL;
    }

    PyDataObject* dObj = (PyDataObject*)self->base;
    if (dObj->dataObject == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "data object is empty.");
        return NULL;
    }

    PyObject *output = NULL;

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
        case ito::tRGBA32:
            {
                ito::PythonRgba::PyRgba *color = ito::PythonRgba::createEmptyPyRgba();
                if (color) color->rgba = ((Rgba32*)(*(self->it)))->rgba;
                output = (PyObject*)color;
            }
            break;
        case ito::tFloat32:
            output = PyFloat_FromDouble((double)(*((ito::float32*)(*(self->it)))));
            break;
        case ito::tFloat64:
            output = PyFloat_FromDouble((double)(*((ito::float64*)(*(self->it)))));
            break;
        case ito::tComplex64:
        {
            complex64 *value = (complex64*)(*(self->it));
            output = PyComplex_FromDoubles((double)value->real(),(double)value->imag());
            break;
        }
        case ito::tComplex128:
        {
            complex128 *value = (complex128*)(*(self->it));
            output = PyComplex_FromDoubles((double)value->real(),(double)value->imag());
            break;
        }
        default:
            PyErr_Format(PyExc_NotImplementedError, "Type not implemented yet");
    }

    self->it++;
    return output;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyDataObjectIterLen_doc, "Private method returning an estimate of len(list(it)).");
PyObject * PythonDataObject::PyDataObjectIter_len(PyDataObjectIter* self)
{
    return PyLong_FromUnsignedLong(self->len);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonDataObject::PyDataObjectIter_methods[] = {
    {"__length_hint__", (PyCFunction)PyDataObjectIter_len, METH_NOARGS, pyDataObjectIterLen_doc},
    {NULL,              NULL}           /* sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyTypeObject PythonDataObject::PyDataObjectIterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "itom.dataObjectIterator",                           /* tp_name */
    sizeof(PyDataObjectIter),                    /* tp_basicsize */
    0,                                          /* tp_itemsize */
    /* methods */
    (destructor)PyDataObjectIter_dealloc,              /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    PyObject_GenericGetAttr,                    /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,/* tp_flags */
    0,                                          /* tp_doc */
    0,           /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    PyObject_SelfIter,                          /* tp_iter */
    (iternextfunc)PyDataObjectIter_iternext,               /* tp_iternext */
    PyDataObjectIter_methods,                          /* tp_methods */
    0,             /* tp_members */
    0,            /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PythonDataObject::PyDataObjectIter_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyDataObjectIter_new, /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
    0
};

} //end namespace ito
