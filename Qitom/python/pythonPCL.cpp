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
#include <vector>
#include "pythonPCL.h"
#include "../global.h"
#include "common/typeDefs.h"

#if ITOM_POINTCLOUDLIBRARY > 0

#include "pythonQtConversion.h"
#include "pythonDataObject.h"

#include "PointCloud/pclFunctions.h"
#include "DataObject/dataObjectFuncs.h"
#include "pythonCommon.h"

//#include "api/apiFunctions.h"
#include "../../common/apiFunctionsInc.h"

#include <qbytearray.h>
#include <qstring.h>

//for generating a temporary file name (for pickling point clouds)
#include <stdio.h>
#include <qdir.h>
#include <QTemporaryFile>
#include <QDir>

//------------------------------------------------------------------------------------------------------

namespace ito
{
//------------------------------------------------------------------------------------------------------
//fix since PCL 1.8.0 changed its definition for the detailed error messages of their exception class from std::string to const char*
/*static*/ void PythonPCL::PythonPCL_SetString(PyObject *exception, const char *string)
{
    PyErr_SetString(exception, string);
}

//------------------------------------------------------------------------------------------------------
//fix since PCL 1.8.0 changed its definition for the detailed error messages of their exception class from std::string to const char*
/*static*/ void PythonPCL::PythonPCL_SetString(PyObject *exception, const std::string &string)
{
    PyErr_SetString(exception, string.c_str());
}

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPointCloud_addTpDict(PyObject * /*tp_dict*/)
{
}

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPointCloud_dealloc(PyPointCloud* self)
{
    DELETE_AND_SET_NULL(self->data);
    Py_XDECREF(self->base);

    Py_TYPE(self)->tp_free((PyObject*)self);
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyPointCloud* self = reinterpret_cast<PyPointCloud *>(type->tp_alloc(type, 0));
    if (self != NULL)
    {
        self->base = NULL;
        self->data = NULL;
    }

    return reinterpret_cast<PyObject *>(self);
};

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pointCloudInit_doc,"pointCloud(type = point.PointInvalid | pointCloud, indices = None | width, height, point = None | point) -> creates new point cloud.  \n\
\n\
Parameters \n\
----------- \n\
type : point-type (e.g. point.PointXYZ, see Notes) \n\
pointCloud : {pointCloud} \n\
    another pointCloud instance which is appended at this point cloud \n\
indices : {sequence}, optional \n\
    only the indices of the given pointCloud will be appended to this point cloud \n\
width : {int} \n\
    width of the new point cloud \n\
height : {int} \n\
    height of the new point cloud (the point cloud is dense if height > 1) \n\
point : {point} \n\
    single point instance. This point cloud is filled up with elements of this point. If width and height \n\
    are given, every element is set to this point, else the point cloud only consists of this single point. \n\
\n\
Notes \n\
------ \n\
Possible types: \n\
\n\
* point.PointXYZ \n\
* point.PointXYZI \n\
* point.PointXYZRGBA \n\
* point.PointXYZNormal \n\
* point.PointXYZINormal \n\
* point.PointXYZRGBNormal\n\
");
int PythonPCL::PyPointCloud_init(PyPointCloud *self, PyObject *args, PyObject * /*kwds*/)
{

    int pclType = ito::pclInvalid;
    PyObject *copyConstr = NULL;
    PyObject *singlePoint = NULL;
    PyObject *pySeq = NULL;
    unsigned int width, height;
    bool done = false;

    //0. check for call without arguments
    if (args == NULL)
    {
        try
        {
            self->data = new ito::PCLPointCloud();
        }
        catch(std::bad_alloc &/*ba*/)
        {
            self->data = NULL;
            PyErr_SetString(PyExc_RuntimeError, "no more memory when creating point cloud");
            return -1;
        }
        catch(...)
        {
            self->data = NULL;
            PyErr_SetString(PyExc_RuntimeError, "an exception has been raised when creating point cloud");
            return -1;
        }

        done = true;
    }

    //1. check for type only
    if (!done && PyArg_ParseTuple(args,"|i", &pclType))
    {
        done = true;
        switch(pclType)
        {
        case ito::pclInvalid:
            self->data = new ito::PCLPointCloud(ito::pclInvalid);
        break;

        case ito::pclXYZ:
            self->data = new ito::PCLPointCloud(ito::pclXYZ);
        break;

        case ito::pclXYZI:
            self->data = new ito::PCLPointCloud(ito::pclXYZI);
        break;

        case ito::pclXYZRGBA:
            self->data = new ito::PCLPointCloud(ito::pclXYZRGBA);
        break;

        case ito::pclXYZNormal:
            self->data = new ito::PCLPointCloud(ito::pclXYZNormal);
        break;

        case ito::pclXYZINormal:
            self->data = new ito::PCLPointCloud(ito::pclXYZINormal);
        break;

        case ito::pclXYZRGBNormal:
            self->data = new ito::PCLPointCloud(ito::pclXYZRGBNormal);
        break;

        default:
            PyErr_SetString(PyExc_TypeError, "The point cloud type is unknown");
            return -1;
        }
    }

    //2. check for copy constructor
    PyErr_Clear();
    if (!done && PyArg_ParseTuple(args, "O!|O", &PythonPCL::PyPointCloudType, &copyConstr, &pySeq))
    {
        PyPointCloud *copyConstr2 = reinterpret_cast<PyPointCloud*>(copyConstr);
        if (copyConstr2->data != NULL)
        {
            if (pySeq == NULL)
            {
                try
                {
                    self->data = new ito::PCLPointCloud(*copyConstr2->data);
                }
                catch(std::bad_alloc &/*ba*/)
                {
                    self->data = NULL;
                    PyErr_SetString(PyExc_RuntimeError, "no more memory when creating point cloud");
                    return -1;
                }
                catch(...)
                {
                    self->data = NULL;
                    PyErr_SetString(PyExc_RuntimeError, "an exception has been raised when creating point cloud");
                    return -1;
                }
            }
            else
            {
                if (PyIter_Check(pySeq) || PySequence_Check(pySeq))
                {
                    std::vector< int > indices;
                    PyObject *item = NULL;

                    if (PyIter_Check(pySeq))
                    {
                        PyObject *iterator = PyObject_GetIter(pySeq);

                        if (PySequence_Check(pySeq) && PySequence_Length(pySeq)>0)
                        {
                            indices.reserve(PySequence_Length(pySeq));
                        }

                        if (iterator == NULL)
                        {
                            PyErr_SetString(PyExc_TypeError, "error creating iterator");
                        }
                        else
                        {
                            //TODO: gcc wants paraentheses around assignment in while condition
                            //item gets loaded with the iterator and the result is implicitly compared to zero anyways,
                            //so i just made this explicit to suppress the compilerwarning
                            while ((item = PyIter_Next(iterator)) != NULL)
                            {
                                if (PyLong_Check(item))
                                {
                                    indices.push_back(PyLong_AsLong(item));
                                    Py_DECREF(item);
                                }
                                else
                                {
                                    PyErr_SetString(PyExc_TypeError, "indices must only contain integer values");
                                    Py_DECREF(item);
                                    break;
                                }
                            }

                            Py_DECREF(iterator);
                        }
                    }
                    else if (PySequence_Check(pySeq))
                    {
                        indices.reserve(PySequence_Length(pySeq));

                        for (Py_ssize_t i = 0; i < PySequence_Length(pySeq); ++i)
                        {
                            item = PySequence_GetItem(pySeq, i); //new ref

                            if (PyLong_Check(item))
                            {
                                indices.push_back(PyLong_AsLong(item));
                                Py_DECREF(item);
                            }
                            else
                            {
                                PyErr_SetString(PyExc_TypeError, "indices must only contain integer values");
                                Py_DECREF(item);
                                break;
                            }
                        }
                    }

                    if (!PyErr_Occurred())
                    {
                        try
                        {
                            self->data = new ito::PCLPointCloud(*copyConstr2->data, indices);
                        }
                        catch(std::bad_alloc &/*ba*/)
                        {
                            self->data = NULL;
                            PyErr_SetString(PyExc_RuntimeError, "no more memory when creating point cloud");
                            return -1;
                        }
                        catch(...)
                        {
                            self->data = NULL;
                            PyErr_SetString(PyExc_RuntimeError, "an exception has been raised when creating point cloud");
                            return -1;
                        }
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "indices must be an iteratible object");
                    return -1;
                }
            }
        }
        else
        {
            try
            {
                self->data = new ito::PCLPointCloud(ito::pclInvalid);
            }
            catch(std::bad_alloc &/*ba*/)
            {
                self->data = NULL;
                PyErr_SetString(PyExc_RuntimeError, "no more memory when creating point cloud");
                return -1;
            }
            catch(...)
            {
                self->data = NULL;
                PyErr_SetString(PyExc_RuntimeError, "an exception has been raised when creating point cloud");
                return -1;
            }
        }
        done = true;
    }

    //3. check for width_, height_, value
    PyErr_Clear();
    if (!done && PyArg_ParseTuple(args, "II|O!", &width, &height, &PythonPCL::PyPointType, &singlePoint))
    {
        ito::PCLPoint point;
        if (singlePoint != NULL)
        {
            PyPoint *pyPoint = (PyPoint*)singlePoint;
            point = *(pyPoint->point);
        }

        done = true;

        try
        {
            self->data = new ito::PCLPointCloud((uint32_t)width, (uint32_t)height, point.getType(), point);
        }
        catch(std::bad_alloc &/*ba*/)
        {
            self->data = NULL;
            PyErr_SetString(PyExc_RuntimeError, "no more memory when creating point cloud");
            return -1;
        }
        catch(...)
        {
            self->data = NULL;
            PyErr_SetString(PyExc_RuntimeError, "an exception has been raised when creating point cloud");
            return -1;
        }
    }

    //4. check for single point
    PyErr_Clear();
    if (!done && PyArg_ParseTuple(args, "O!", &PythonPCL::PyPointType, &singlePoint))
    {
        ito::PCLPoint point;
        if (singlePoint != NULL)
        {
            PyPoint *pyPoint = (PyPoint*)singlePoint;
            point = *(pyPoint->point);
        }

        done = true;

        try
        {
            self->data = new ito::PCLPointCloud(1, 1, point.getType(), point);
        }
        catch(std::bad_alloc &/*ba*/)
        {
            self->data = NULL;
            PyErr_SetString(PyExc_RuntimeError, "no more memory when creating point cloud");
            return -1;
        }
        catch(...)
        {
            self->data = NULL;
            PyErr_SetString(PyExc_RuntimeError, "an exception has been raised when creating point cloud");
            return -1;
        }
    }

    if (done == false)
    {
        PyErr_SetString(PyExc_RuntimeError, "arguments for constructor must be a type value, another instance of point cloud, an instance of point or width, height and an instance of point");
        return -1;
    }

    return 0;
};

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudType_doc,"returns point type of point cloud [read-only]\n\
\n\
Point types can be: \n\
\n\
* pointCloud.PointInvalid \n\
* pointCloud.PointXYZ \n\
* pointCloud.PointXYZI \n\
* pointCloud.PointXYZRGBA \n\
* pointCloud.PointXYZNormal \n\
* pointCloud.PointXYZINormal \n\
* pointCloud.PointXYZRGBNormal \n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetType(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    ito::tPCLPointType type;
    try
    {
        type = self->data->getType();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    if (PythonPCL::PyPointType.tp_dict != NULL)
    {
        PyObject *dict = PythonPCL::PyPointType.tp_dict;
        PyObject *value = NULL;
        switch(type)
        {
        default:
            PyErr_SetString(PyExc_ValueError, "undefined type of point cloud detected.");
        case ito::pclInvalid:
            value = PyDict_GetItemString(dict, "PointInvalid");
            break;
        case ito::pclXYZ:
            value = PyDict_GetItemString(dict, "PointXYZ");
            break;
        case ito::pclXYZI:
            value = PyDict_GetItemString(dict, "PointXYZI");
            break;
        case ito::pclXYZRGBA:
            value = PyDict_GetItemString(dict, "PointXYZRGBA");
            break;
        case ito::pclXYZNormal:
            value = PyDict_GetItemString(dict, "PointXYZNormal");
            break;
        case ito::pclXYZINormal:
            value = PyDict_GetItemString(dict, "PointXYZINormal");
            break;
        case ito::pclXYZRGBNormal:
            value = PyDict_GetItemString(dict, "PointXYZRGBNormal");
            break;
        }
        Py_XINCREF(value);
        return value;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "class point is not available");
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudSize_doc,"returns total number of points in point cloud\n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetSize(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    size_t size;
    try
    {
        size = self->data->size();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
    }

    return Py_BuildValue("i", size);
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudHeight_doc,"returns height of point cloud if organized as regular grid (organized == true), else 1. \n\
\n\
The height of a point cloud is equal to 1, if the points in the point cloud are not organized. If it is organized, \n\
all points are assumed to lie on a regular grid, such that neighbouring points in the grid are adjacent in space, too. \n\
In this case, height is the number of rows in this grid. The total number of points is then height * width. \n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetHeight(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    uint32_t height;
    try
    {
        height = self->data->height();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    return Py_BuildValue("I", height);
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudWidth_doc,"returns width of point cloud if organized as regular grid (organized == true), else equal to size \n\
\n\
The width of a point cloud is equal to the number of total points, if the points in the point cloud are not organized. If it is organized, \n\
all points are assumed to lie on a regular grid, such that neighbouring points in the grid are adjacent in space, too. \n\
In this case, width is the number of columns in this grid. The total number of points is then height * width. \n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetWidth(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    uint32_t width;
    try
    {
        width = self->data->width();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    return Py_BuildValue("I", width);
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudEmpty_doc,"returns whether this point cloud is empty (size == 0)\n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetEmpty(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    bool empty;
    try
    {
        empty = self->data->empty();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    if (empty) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudOrganized_doc,"returns True if this point cloud is organized, else False\n\
\n\
Points in an organized point cloud are assumed to lie on a regular grid, defined by height and width. \n\
Neighbouring points in this grid are the neighbours in space, too. An unorganized point cloud has \n\
always a height equal to 1 and width equal to the total number of points. \n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetOrganized(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    bool organized;
    try
    {
        organized = self->data->isOrganized();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    if (organized) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudDense_doc,"specifies if all the data in points is finite (true), or whether it might contain Inf/NaN values (false)");
PyObject* PythonPCL::PyPointCloud_GetDense(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    bool dense;
    try
    {
        dense = self->data->is_dense();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    if (dense) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

int PythonPCL::PyPointCloud_SetDense(PyPointCloud *self, PyObject *value, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return -1;
    }

    bool dense;
    if (PyArg_Parse(value,"b", &dense))
    {
        try
        {
            self->data->set_dense(dense);
        }
        catch(pcl::PCLException &exc)
        {
            PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
            return -1;
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Argument must have a boolean value.");
        return -1;
    }

    try
    {
        self->data->set_dense(dense);
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return -1;
    }

    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFields_doc,"get available field names of point cloud\n\
\n\
This property returns a list of field names that are contained in this cloud, e.g. ['x', 'y', 'z']. \n\
\n\
Notes \n\
----- \n\
This attribute is readonly");
PyObject* PythonPCL::PyPointCloud_GetFields(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    std::string names;
    try
    {
        names = self->data->getFieldsList();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    QStringList a = QString::fromStdString(names).split(" ");
    PyObject *result = PyList_New(a.size());
    QByteArray ba;

    for (int i = 0 ; i < a.size() ; i++)
    {
        ba = a[i].toLatin1();
        //PyList_SetItem(result,i, PyUnicode_FromStringAndSize(ba.data(), ba.size()));
        PyList_SetItem(result,i, PyUnicode_DecodeLatin1(ba.data(), ba.size(), NULL));
    }

    return result;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudScaleXYZ_doc, "scaleXYZ(x = 1.0, y = 1.0, z = 1.0) -> scale the x, y and z components of every point by the given values . \n\
\n\
Parameters \n\
----------- \n\
x : {float}, optional \n\
    scaling factor for x-component (default: 1.0) \n\
y : {float}, optional \n\
    scaling factor for y-component (default: 1.0) \n\
z : {float}, optional \n\
    scaling factor for z-component (default: 1.0)");
PyObject* PythonPCL::PyPointCloud_scaleXYZ(PyPointCloud *self, PyObject *args, PyObject *kwds)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    //check if args contains only one point cloud
    //g++ doesn't like char* to be cast from strings...
    static const char *kwlist[] = { "x", "y", "z", NULL };
    ito::float32 x = 1.0;
    ito::float32 y = 1.0;
    ito::float32 z = 1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|fff", const_cast<char **>(kwlist), &x, &y, &z))
    {
        return NULL;
    }

    self->data->scaleXYZ(x, y, z);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudMoveXYZ_doc, "moveXYZ(x = 0.0, y = 0.0, z = 0.0) -> move the x, y and z components of every point by the given values. \n\
\n\
Parameters \n\
----------- \n\
x : {float}, optional \n\
    offset value for x-component (default: 0.0) \n\
y : {float}, optional \n\
    offset value for y-component (default: 0.0) \n\
z : {float}, optional \n\
    offset value for z-component (default: 0.0)");
PyObject* PythonPCL::PyPointCloud_moveXYZ(PyPointCloud *self, PyObject *args, PyObject *kwds)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    //check if args contains only one point cloud
    static const char *kwlist[] = { "x", "y", "z", NULL };
    ito::float32 x = 0.0;
    ito::float32 y = 0.0;
    ito::float32 z = 0.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|fff", const_cast<char**>(kwlist), &x, &y, &z))
    {
        return NULL;
    }

    self->data->moveXYZ(x, y, z);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudAppend_doc,"append(point) -> appends point or all points from a given point cloud to the end of the point cloud. \n\
\n\
Parameters \n\
----------- \n\
point : {point or pointCloud} \n\
    point or points from pointCloud that should be appended at the list of points of this pointCloud. \n\
\n\
Notes \n\
----- \n\
The type of point must fit to the type of the point cloud. If the point cloud is \n\
invalid, its type is set to the type of the point.");
PyObject* PythonPCL::PyPointCloud_append(PyPointCloud *self, PyObject *args, PyObject *kwds)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

        //check if args contains only one point cloud
    static const char *kwlist0[] = {"pointCloud", NULL};
    static const char *kwlist1[] = {"point", NULL};
    PyObject *pclObj = NULL;
    if (PyArg_ParseTupleAndKeywords(args,kwds,"O!",const_cast<char**>(kwlist0),
                 &PyPointCloudType, &pclObj))
    {
        PyPointCloud *pcl = (PyPointCloud*)pclObj;
        if (pcl == NULL || pcl->data == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "argument is of type pointCloud, but this point cloud or the underlying point cloud structure is NULL");
            return NULL;
        }

        *(self->data) += *(pcl->data);

    }
    else if (PyErr_Clear(), PyArg_ParseTupleAndKeywords(args,kwds,"O!",const_cast<char**>(kwlist1),
                 &PyPointType, &pclObj))
    {
        PyPoint *point = (PyPoint*)pclObj;
        if (point == NULL || point->point == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "argument is of type point, but this point or the underlying point structure is NULL");
            return NULL;
        }
        if (self->data->getType() != point->point->getType() && self->data->getType() != ito::pclInvalid)
        {
            PyErr_SetString(PyExc_RuntimeError, "point cloud and this point do not have the same type");
            return NULL;
        }
        self->data->push_back(*(point->point));
    }
    else
    {
        PyErr_Clear();

        switch(self->data->getType())
        {
        case ito::pclXYZ:
        {
            static const char *kwlist[] = {"xyz", NULL};
            PyObject *xyz = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",const_cast<char**>(kwlist),
                             &xyz)) return NULL;
            return PyPointCloud_XYZ_append(self, xyz);
        }
        case ito::pclXYZI:
        {
            static const char *kwlist[] = {"xyzi", NULL};
            PyObject *xyzi = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",const_cast<char**>(kwlist),&xyzi)) return NULL;
            return PyPointCloud_XYZI_append(self, xyzi);
        }
        case ito::pclXYZRGBA:
        {
            static const char *kwlist[] = {"xyz", "rgba", NULL};
            PyObject *xyz = NULL;
            PyObject *rgba = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",const_cast<char**>(kwlist),&xyz, &rgba)) return NULL;
            return PyPointCloud_XYZRGBA_append(self, xyz, rgba);
        }
        case ito::pclXYZNormal:
        {
            static const char *kwlist[] = {"xyz_normal_curvature", NULL};
            PyObject *xyz_normal_curvature = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",const_cast<char**>(kwlist),&xyz_normal_curvature)) return NULL;
            return PyPointCloud_XYZNormal_append(self, xyz_normal_curvature);
        }
        case ito::pclXYZINormal:
        {
            static const char *kwlist[] = {"xyz_i_normal_curvature", NULL};
            PyObject *xyz_i_normal_curvature = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",const_cast<char**>(kwlist),&xyz_i_normal_curvature)) return NULL;
            return PyPointCloud_XYZINormal_append(self, xyz_i_normal_curvature);
        }
        case ito::pclXYZRGBNormal:
        {
            static const char *kwlist[] = {"xyz_normal_curvature", "rgba", NULL};
            PyObject *xyz_normal_curvature = NULL;
            PyObject *rgba = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",const_cast<char**>(kwlist),&xyz_normal_curvature, &rgba)) return NULL;
            return PyPointCloud_XYZRGBNormal_append(self, xyz_normal_curvature, rgba);
        }
        default:
            PyErr_SetString(PyExc_RuntimeError, "point cloud must have a valid point type");
            return NULL;
        }
    }

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_XYZ_append(PyPointCloud *self, PyObject *xyzObj)
{
    if (self->data == NULL || self->data->getType() != ito::pclXYZ)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZ");
        return NULL;
    }

    PyObject *arrayObj = NULL;
    npy_intp n = 0;
    float32** xyz = new float32*[3];
    arrayObj = PythonPCL::parseObjAsFloat32Array(xyzObj, 3, n, xyz);
    if (arrayObj == NULL)
    {
        delete[] xyz; xyz = NULL;
        return NULL; //error already set by parseObjAsFloat32Array
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPtr = self->data->toPointXYZ();
    pclPtr->reserve(pclPtr->size() + n);
    for (npy_intp i = 0 ; i < n; i++)
    {
        pclPtr->push_back(pcl::PointXYZ(xyz[0][i],xyz[1][i],xyz[2][i]));
    }

    delete[] xyz; xyz = NULL;

    Py_XDECREF(arrayObj);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_XYZI_append(PyPointCloud *self, PyObject *xyziObj)
{
    if (self->data == NULL || self->data->getType() != ito::pclXYZI)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZI");
        return NULL;
    }

    PyObject *arrayObj = NULL;
    npy_intp n = 0;
    float32** xyzi = new float32*[4];
    arrayObj = PythonPCL::parseObjAsFloat32Array(xyziObj, 4, n, xyzi);
    if (arrayObj == NULL)
    {
        delete[] xyzi; xyzi = NULL;
        return NULL; //error already set by parseObjAsFloat32Array
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pclPtr = self->data->toPointXYZI();
    pclPtr->reserve(pclPtr->size() + n);
    pcl::PointXYZI p;
    for (npy_intp i = 0 ; i < n; i++)
    {
        p.x = xyzi[0][i]; p.y = xyzi[1][i]; p.z = xyzi[2][i]; p.intensity = xyzi[3][i];
        pclPtr->push_back(p);
    }

    delete[] xyzi; xyzi = NULL;

    Py_XDECREF(arrayObj);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_XYZRGBA_append(PyPointCloud *self, PyObject *xyzObj, PyObject *rgbaObj)
{
    if (self->data == NULL || self->data->getType() != ito::pclXYZI)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZRGBA");
        return NULL;
    }

    PyObject *arrayObj = NULL;
    npy_intp n1 = 0;
    float32** xyz = new float32*[3];
    arrayObj = PythonPCL::parseObjAsFloat32Array(xyzObj, 3, n1, xyz);
    if (arrayObj == NULL)
    {
        delete[] xyz; xyz = NULL;
        return NULL; //error already set by parseObjAsFloat32Array
    }

    PyObject *arrayObj2 = NULL;
    npy_intp n2 = 0;
    uint8_t** rgba = new uint8_t*[4];
    arrayObj2 = PythonPCL::parseObjAsUInt8Array(rgbaObj, 4, n2, rgba);
    if (arrayObj2 == NULL)
    {
        delete[] rgba; rgba = NULL;
        delete[] xyz; xyz = NULL;
        Py_XDECREF(arrayObj);
        return NULL; //error already set by parseObjAsFloat32Array
    }

    if (n1 != n2)
    {
        delete[] rgba; rgba = NULL;
        delete[] xyz; xyz = NULL;
        Py_XDECREF(arrayObj);
        Py_XDECREF(arrayObj2);
        PyErr_SetString(PyExc_RuntimeError, "length of xyz and rgba-arrays must be equal");
        return NULL;
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pclPtr = self->data->toPointXYZRGBA();
    pclPtr->reserve(pclPtr->size() + n1);
    pcl::PointXYZRGBA p;
    for (npy_intp i = 0 ; i < n1; i++)
    {
        p.x = xyz[0][i]; p.y = xyz[1][i]; p.z = xyz[2][i];
        p.r = rgba[0][i]; p.g = rgba[1][i]; p.b = rgba[2][i]; p.PCLALPHA = rgba[3][i];
        pclPtr->push_back(p);
    }

    delete[] rgba; rgba = NULL;
    delete[] xyz; xyz = NULL;
    Py_XDECREF(arrayObj);
    Py_XDECREF(arrayObj2);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_XYZNormal_append(PyPointCloud *self, PyObject *xyz_nxnynz_curvObj)
{
    if (self->data == NULL || self->data->getType() != ito::pclXYZNormal)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZNormal");
        return NULL;
    }

    PyObject *arrayObj = NULL;
    npy_intp n = 0;
    float32** xyznxnynzc = new float32*[7];
    arrayObj = PythonPCL::parseObjAsFloat32Array(xyz_nxnynz_curvObj, 7, n, xyznxnynzc);
    if (arrayObj == NULL)
    {
        delete[] xyznxnynzc; xyznxnynzc = NULL;
        return NULL; //error already set by parseObjAsFloat32Array
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr pclPtr = self->data->toPointXYZNormal();
    pclPtr->reserve(pclPtr->size() + n);
    pcl::PointNormal p;
    for (npy_intp i = 0 ; i < n; i++)
    {
        p.x = xyznxnynzc[0][i]; p.y = xyznxnynzc[1][i]; p.z = xyznxnynzc[2][i];
        p.normal_x = xyznxnynzc[3][i]; p.normal_y = xyznxnynzc[4][i]; p.normal_z = xyznxnynzc[5][i];
        p.curvature = xyznxnynzc[6][i];
        pclPtr->push_back(p);
    }

    delete[] xyznxnynzc; xyznxnynzc = NULL;

    Py_XDECREF(arrayObj);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_XYZINormal_append(PyPointCloud *self, PyObject *xyz_i_nxnynz_curvObj)
{
    if (self->data == NULL || self->data->getType() != ito::pclXYZINormal)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZINormal");
        return NULL;
    }

    PyObject *arrayObj = NULL;
    npy_intp n = 0;
    float32** xyzinxnynzc = new float32*[7];
    arrayObj = PythonPCL::parseObjAsFloat32Array(xyz_i_nxnynz_curvObj, 7, n, xyzinxnynzc);
    if (arrayObj == NULL)
    {
        delete[] xyzinxnynzc; xyzinxnynzc = NULL;
        return NULL; //error already set by parseObjAsFloat32Array
    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pclPtr = self->data->toPointXYZINormal();
    pclPtr->reserve(pclPtr->size() + n);
    pcl::PointXYZINormal p;
    for (npy_intp i = 0 ; i < n; i++)
    {
        p.x = xyzinxnynzc[0][i]; p.y = xyzinxnynzc[1][i]; p.z = xyzinxnynzc[2][i];
        p.intensity = xyzinxnynzc[3][i];
        p.normal_x = xyzinxnynzc[4][i]; p.normal_y = xyzinxnynzc[5][i]; p.normal_z = xyzinxnynzc[6][i];
        p.curvature = xyzinxnynzc[7][i];
        pclPtr->push_back(p);
    }

    delete[] xyzinxnynzc; xyzinxnynzc = NULL;

    Py_XDECREF(arrayObj);
    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_XYZRGBNormal_append(PyPointCloud *self, PyObject *xyz_i_nxnynz_curvObj, PyObject *rgbaObj)
{
    if (self->data == NULL || self->data->getType() != ito::pclXYZRGBNormal)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZRGBNormal");
        return NULL;
    }

    PyObject *arrayObj = NULL;
    npy_intp n1 = 0;
    float32** xyznxnynzc = new float32*[7];
    arrayObj = PythonPCL::parseObjAsFloat32Array(xyz_i_nxnynz_curvObj, 7, n1, xyznxnynzc);
    if (arrayObj == NULL)
    {
        delete[] xyznxnynzc; xyznxnynzc = NULL;
        return NULL; //error already set by parseObjAsFloat32Array
    }

    PyObject *arrayObj2 = NULL;
    npy_intp n2 = 0;
    uint8_t** rgba = new uint8_t*[4];
    arrayObj2 = PythonPCL::parseObjAsUInt8Array(rgbaObj, 4, n2, rgba);
    if (arrayObj2 == NULL)
    {
        delete[] rgba; rgba = NULL;
        delete[] xyznxnynzc; xyznxnynzc = NULL;
        Py_XDECREF(arrayObj);
        return NULL; //error already set by parseObjAsFloat32Array
    }

    if (n1 != n2)
    {
        delete[] rgba; rgba = NULL;
        delete[] xyznxnynzc; xyznxnynzc = NULL;
        Py_XDECREF(arrayObj);
        Py_XDECREF(arrayObj2);
        PyErr_SetString(PyExc_RuntimeError, "length of xyz and rgba-arrays must be equal");
        return NULL;
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclPtr = self->data->toPointXYZRGBNormal();
    pclPtr->reserve(pclPtr->size() + n1);
    pcl::PointXYZRGBNormal p;

    for (npy_intp i = 0 ; i < n1; i++)
    {
        p.x = xyznxnynzc[0][i]; p.y = xyznxnynzc[1][i]; p.z = xyznxnynzc[2][i];
        p.normal_x = xyznxnynzc[3][i]; p.normal_y = xyznxnynzc[4][i]; p.normal_z = xyznxnynzc[5][i];
        p.curvature = xyznxnynzc[6][i];
        p.r = rgba[0][i]; p.g = rgba[1][i]; p.b = rgba[2][i]; p.PCLALPHA = rgba[3][i];
        pclPtr->push_back(p);
    }

    delete[] rgba; rgba = NULL;
    delete[] xyznxnynzc; xyznxnynzc = NULL;
    Py_XDECREF(arrayObj);
    Py_XDECREF(arrayObj2);

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_name(PyPointCloud* /*self*/)
{
    return PyUnicode_FromString("PointCloud");
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_repr(PyPointCloud *self)
{
    if (self->data)
    {
        int size = 0;
        int width = 0;
        int height = 0;
        ito::tPCLPointType type = self->data->getType();

        try
        {
        size = (int)self->data->size();
        width = self->data->width();
        height = self->data->height();
        }
        catch(pcl::PCLException &/*exc*/) {};

        switch(type)
        {
        case ito::pclXYZ: return PyUnicode_FromFormat("PointCloud (type: PointXYZ, size: %d, [%d x %d])", size, height, width);
        case ito::pclXYZI: return PyUnicode_FromFormat("PointCloud (type: PointXYZI, size: %d, [%d x %d])", size, height, width);
        case ito::pclXYZRGBA: return PyUnicode_FromFormat("PointCloud (type: PointXYZRGBA, size: %d, [%d x %d])", size, height, width);
        case ito::pclXYZNormal: return PyUnicode_FromFormat("PointCloud (type: PointXYZNormal, size: %d, [%d x %d])", size, height, width);
        case ito::pclXYZINormal: return PyUnicode_FromFormat("PointCloud (type: PointXYZINormal, size: %d, [%d x %d])", size, height, width);
        case ito::pclXYZRGBNormal: return PyUnicode_FromFormat("PointCloud (type: PointXYZRGBNormal, size: %d, [%d x %d])", size, height, width);
        default: return PyUnicode_FromString("PointCloud (type: PointInvalid)");
        }
    }
    else
    {
        return PyUnicode_FromString("PointCloud (empty)");
    }
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudClear_doc,"clear() -> clears the whole point cloud");
PyObject* PythonPCL::PyPointCloud_clear(PyPointCloud *self)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    try
    {
        self->data->clear();
    }
    catch(pcl::PCLException &exc)
    {
        PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
        return NULL;
    }

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
Py_ssize_t PythonPCL::PyPointCloud_seqLength(PyPointCloud *self)
{
    size_t s = 0;
    if (self->data)
    {
        try
        {
            s = self->data->size();
        }
        catch(pcl::PCLException &/*exc*/)
        {
            s = 0;
        }
    }
    return s;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqConcat(PyPointCloud *self, PyObject *rhs) //returns new reference
{
    if (Py_TYPE(rhs) != &PyPointCloudType)
    {
        PyErr_SetString(PyExc_TypeError, "object must be of type pointCloud");
        return NULL;
    }

    PyPointCloud *rhs_ = reinterpret_cast<PyPointCloud*>(rhs);
    if (self->data && rhs_->data)
    {
        PyObject *args = Py_BuildValue("(i)", self->data->getType());
        PyPointCloud *result = (PyPointCloud*)PyObject_Call((PyObject*)&PyPointCloudType, args, NULL);
        Py_DECREF(args);
        if (result)
        {
            try
            {
                *result->data = *self->data + *rhs_->data;
            }
            catch(pcl::PCLException &exc)
            {
                PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
                return NULL;
            }
            return reinterpret_cast<PyObject*>(result);
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "could not allocate object of type pointCloud");
            return NULL;
        }
    }
    PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqRepeat(PyPointCloud *self, Py_ssize_t size)//returns new reference
{
    if (self->data)
    {
        PyObject *args = Py_BuildValue("(i)", self->data->getType());
        PyPointCloud *result = reinterpret_cast<PyPointCloud*>(PyObject_Call((PyObject*)&PyPointCloudType, args, NULL)); //new reference
        Py_DECREF(args);
        if (result)
        {
            try
            {
                for (Py_ssize_t i = 0; i < size; i++)
                {
                    *result->data += *self->data;
                }
            }
            catch(pcl::PCLException &exc)
            {
                PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
                return NULL;
            }

            return reinterpret_cast<PyObject*>(result);
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "could not allocate object of type pointCloud");
            return NULL;
        }
    }
    PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqItem(PyPointCloud *self, Py_ssize_t size) //returns new reference
{
    if (self->data)
    {
        if (size < 0 || size >= static_cast<Py_ssize_t>(self->data->size()))
        {
            PyErr_Format(PyExc_IndexError, "index must be in range [%d,%d]", 0, self->data->size()-1); //it is important to have an index or stopIteration exception here, such that "for i in pointCloud:" will stop
            return NULL;
        }

        PyPoint *result = (PyPoint*)PyObject_Call((PyObject*)&PyPointType, NULL, NULL); //new reference
        if (result)
        {
            try
            {
                result->point = new ito::PCLPoint(self->data->at(size));
                return (PyObject*)result;
            }
            catch(pcl::PCLException &exc)
            {
                PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
                return NULL;
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "could not allocate object of type point");
            return NULL;
        }
    }
    PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPointCloud_seqAssItem(PyPointCloud *self, Py_ssize_t size, PyObject *point)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
        return -1;
    }

    if (size < 0 || size >= static_cast<Py_ssize_t>(self->data->size()))
    {
        PyErr_Format(PyExc_RuntimeError, "index must be in range [%d,%d]", 0, self->data->size()-1);
        return 0;
    }

    if (point == NULL) //value should be deleted
    {
        self->data->erase(size, size+1);
    }
    else
    {
        if (Py_TYPE(point) != &PyPointType)
        {
            PyErr_SetString(PyExc_TypeError, "assigned value must be of type point");
            return -1;
        }
        PyPoint *point_ = reinterpret_cast<PyPoint*>(point);

        try
        {
            self->data->set_item(size, *point_->point);
        }
        catch (pcl::PCLException exc)
        {
            PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
            return -1;
        }
    }

    return 0;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqInplaceConcat(PyPointCloud *self, PyObject *rhs) //returns new reference
{
    if (Py_TYPE(rhs) != &PyPointCloudType)
    {
        PyErr_SetString(PyExc_TypeError, "object must be of type pointCloud");
        return NULL;
    }

    PyPointCloud *rhs_ = (PyPointCloud*)rhs;
    if (self->data && rhs_->data)
    {
        try
        {
            *self->data = *self->data + *rhs_->data;
        }
        catch(pcl::PCLException &exc)
        {
            PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
            return NULL;
        }

        Py_INCREF(self);
        return reinterpret_cast<PyObject*>(self);
    }
    PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqInplaceRepeat(PyPointCloud * /*self*/, Py_ssize_t /*size*/)//returns new reference
{
    PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
Py_ssize_t PythonPCL::PyPointCloud_mappingLength(PyPointCloud *self)
{
    return PyPointCloud_seqLength(self);
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_mappingGetElem(PyPointCloud *self, PyObject *key)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
        return NULL;
    }

    Py_ssize_t start, stop, step, slicelength;
    //check type of elem, must be int or stride
    if (PyLong_Check(key))
    {
        start = PyLong_AsLong(key);
        if (start < 0) start = self->data->size() + start;

        if (start < 0 || start >= static_cast<Py_ssize_t>(self->data->size()))
        {
            PyErr_Format(PyExc_TypeError, "index exceeds dimensions of point cloud [0,%d]", self->data->size());
            return NULL;
        }
        stop = start;
        step = 1;
        slicelength = 1;
    }
    else if (PySlice_Check(key))
    {
        if (!PySlice_GetIndicesEx(key, self->data->size(), &start, &stop, &step, &slicelength) == 0)
        {
            return NULL; //error message already set
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "range tuple element is neither of type integer nor of type slice");
        return NULL;
    }

    if (slicelength > 1) //two or more points -> return point cloud with selected points only
    {
        PyObject *indices = PyTuple_New(slicelength);
        PyObject *result = NULL;
        Py_ssize_t c = 0;
        for (Py_ssize_t i = start ; i < stop ; i += step)
        {
            PyTuple_SetItem(indices, c, PyLong_FromLong(i)); //steals a reference
            c++;
        }

        PyObject *args = Py_BuildValue("OO", (PyObject*)self, indices);
        PyObject *kwds = PyDict_New();

        result = PyObject_Call((PyObject*)&PyPointCloudType, args, kwds); //new reference
        if (!result)
        {
            PyErr_SetString(PyExc_RuntimeError, "could not allocate object of type point cloud");
        }

        Py_DECREF(kwds);
        Py_DECREF(args);
        Py_DECREF(indices);

        return result;
    }
    else //one element -> return point
    {
        PyPoint *tempPt = (PyPoint*)PyObject_Call((PyObject*)&PyPointType, NULL, NULL); //new reference
        if (tempPt)
        {
            tempPt->point = new ito::PCLPoint(self->data->at(start));
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "could not allocate object of type point");
            return NULL;
        }
        return (PyObject*)tempPt;
    }
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPointCloud_mappingSetElem(PyPointCloud *self, PyObject *key, PyObject *value)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
        return -1;
    }

    Py_ssize_t start, stop, step, slicelength;
    //check type of elem, must be int or stride
    if (PyLong_Check(key))
    {
        start = PyLong_AsLong(key);
        if (start < 0) start = self->data->size() + start;

        if (start < 0 || start >= static_cast<Py_ssize_t>(self->data->size()))
        {
            PyErr_Format(PyExc_TypeError, "index exceeds dimensions of point cloud [0,%d]", self->data->size());
            return -1;
        }
        stop = start;
        step = 1;
        slicelength = 1;
    }
    else if (PySlice_Check(key))
    {
        if (!PySlice_GetIndicesEx(key, self->data->size(), &start, &stop, &step, &slicelength) == 0)
        {
            return -1; //error message already set
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "range tuple element is neither of type integer nor of type slice");
        return -1;
    }

    if (value == NULL) //delete items
    {
        if (step == 1)
        {
            self->data->erase(start,stop+1);
        }
        else
        {
            for (Py_ssize_t i = stop-1 ; i>=start ; i-=step) //erase backwards
            {
                self->data->erase(i,i+1);
            }
        }
    }
    else // set items
    {
        if (slicelength == 1)
        {
            return PyPointCloud_seqAssItem(self, start, value);
        }
        else if (PySequence_Check(value))
        {
            if (PySequence_Length(value) == slicelength)
            {
                int retCode = 0;
                PyObject *temp;
                for (Py_ssize_t i = start ; i < stop ; i+=step)
                {
                    temp = PySequence_GetItem(value, i); //new ref
                    retCode |= PyPointCloud_seqAssItem(self, i, temp);
                    Py_XDECREF(temp);
                }
                return retCode;
            }
            else
            {
                PyErr_Format(PyExc_RuntimeError, "sequence must contain %d elements", slicelength);
                return -1;
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "value should be any sequence object filled with points (tuple, list,...)");
            return -1;
        }
    }
    return 0;
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudInsert_doc,"insert(index, values) -> inserts a single point or a sequence of points before position given by index\n\
\n\
By this method, an itom.point or a sequence of points is inserted into the existing point cloud. \n\
The new points are inserted before the existing value at the index-th position. \n\
\n\
The type of the inserted points must fit to the type of the point cloud. \n\
\n\
Parameters \n\
----------- \n\
index : {int}\n\
    the new point(s) is / are inserted before the existing value at the index-th position. index must be in range [0, size of point cloud - 1]. Negative values signify a count from the end of the existing cloud. \n\
values : {point, seq. of point}\n\
    itom.point or sequence of itom.point that should be inserted.");
PyObject* PythonPCL::PyPointCloud_insert(PyPointCloud *self, PyObject *args)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "this point cloud is empty");
        return NULL;
    }

    PyObject *index = NULL;
    PyObject *points = NULL;
    Py_ssize_t start = 0;
    if (!PyArg_ParseTuple(args,"OO", &index, &points))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument must be an fixed-point index number followed by a single point or a sequence of points");
        return NULL;
    }

    if (PyLong_Check(index))
    {
        start = PyLong_AsLong(index);
        if (start < 0) start = self->data->size() + start;

        if (start < 0 || start > static_cast<Py_ssize_t>(self->data->size()))
        {
            PyErr_Format(PyExc_TypeError, "index exceeds dimensions of point cloud [0,%d]", self->data->size());
            return NULL;
        }
    }

    if (Py_TYPE(points) == &PyPointType)
    {
        try
        {
            self->data->insert(start, *(((PyPoint*)points)->point));
        }
        catch(pcl::PCLException &exc)
        {
            PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
            return NULL;
        }
    }
    else if (PySequence_Check(points))
    {
        PyObject *sequence = PySequence_Fast(points, "");

        for (Py_ssize_t i = 0 ; i < PySequence_Length(points) ; i++)
        {
            if (Py_TYPE(PySequence_Fast_GET_ITEM(sequence,i)) != &PyPointType)
            {
                PyErr_SetString(PyExc_TypeError, "not every element in sequence is of type point");
                Py_DECREF(sequence);
                return NULL;
            }
        }

        for (Py_ssize_t i = 0 ; i < PySequence_Length(points) ; i++)
        {
            try
            {
                self->data->insert(start + i, *(((PyPoint*)PySequence_Fast_GET_ITEM(sequence,i))->point));
            }
            catch(pcl::PCLException &exc)
            {
                Py_DECREF(sequence);
                PythonPCL_SetString(PyExc_TypeError, exc.detailedMessage());
                return NULL;
            }
        }
        Py_DECREF(sequence);
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudErase_doc,"erase(indices) -> erases the points in point clouds indicated by indices (single number of slice) \n\
\n\
Parameters \n\
----------- \n\
indices : {int or slice}\n\
    Single index or slice of indices whose corresponding points should be deleted. \n\
\n\
Notes \n\
----- \n\
This method is the same than command 'del pointCloudVariable[indices]'");
PyObject* PythonPCL::PyPointCloud_erase(PyPointCloud *self, PyObject *args)
{
    PyObject *indices = NULL;
    if (!PyArg_ParseTuple(args,"O", &indices))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument must be a number of a slice object");
        return NULL;
    }

    if (!PyPointCloud_mappingSetElem(self, indices, NULL)) return NULL;

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudToDataObject_doc,"toDataObject() -> returns a PxN data object, where P is determined by the point type in the point cloud. N is the number of points.\n\
\n\
The output has at least 3 elements per column. Onr got each coordinate (xyz). These will always be the first 3 elements. \n\
If the pointcloud type has normals defined, these will be added in the next 4 columns as [nx, ny, nz, curvature]. \n\
If the pointcloud type has an intensity, these will be added in the next 1 column as [intensity]. \n\
If the pointcloud type has an RGB(A)-intensity, these will be added in the next 4 column as [r, g, b, a]. \n\
Hence following combinations are possible [x,y,z], [x,y,z,i], [x,y,z,r,g,b,a], [x,y,z,nx,ny,nz, curvature], [x,y,z,nx,ny,nz, curvature, i], ...\n\
\n\
Returns \n\
------- \n\
dObj : {dataObject}\n\
    A dataObject with P (cols) by N elements (Points), where the elements per column depend on the point cloud type");
/*static*/ PyObject* PythonPCL::PyPointCloud_toDataObject(PyPointCloud *self)
{
    if (self->data)
    {
        ito::PythonDataObject::PyDataObject *pyDObj = ito::PythonDataObject::createEmptyPyDataObject(); //new reference
        pyDObj->dataObject = new ito::DataObject();
        ito::RetVal retval = ito::pclHelper::pointCloudToDObj(self->data, *(pyDObj->dataObject));

        if (ito::PythonCommon::transformRetValToPyException(retval) == false)
        {
            Py_DECREF(pyDObj);
            pyDObj = NULL;
            return NULL;
        }
        return (PyObject*)pyDObj;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "point cloud is empty");
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudCopy_doc,"copy() -> returns a deep copy of this point cloud.\n\
\n\
Returns \n\
------- \n\
cloud : {pointCloud}\n\
    An exact copy if this point cloud.");
/*static*/ PyObject* PythonPCL::PyPointCloud_copy(PyPointCloud *self)
{
    PyPointCloud* result = (PyPointCloud*)PyObject_Call((PyObject*)&PyPointCloudType, NULL, NULL);

    if (result && self->data)
    {
        result->data = new PCLPointCloud(self->data->copy());
    }

    return (PyObject*)result;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_Reduce(PyPointCloud *self, PyObject * /*args*/)
{
    if (self->data == NULL)
    {
        PyErr_SetString(PyExc_NotImplementedError, "point cloud is NULL");
        return NULL;
    }

    int type = self->data->getType();
    PyObject *stateTuple = NULL;

    if (type != ito::pclInvalid && self->data->size() > 0)
    {
        //from the tempnam manpage:
        //Never use this function.  Use mkstemp(3) or tmpfile(3) instead.
        //char *buf = tmpnam(NULL);
                     //actually is not a temp file, but also the target one...
        QTemporaryFile pclTempSaveFile;//name is autogenerated in locals temp dir. Pathes change, if filename is pased to it, so it's left away-
        bool success = pclTempSaveFile.open();//creates Name here
        pclTempSaveFile.close();//hmmm better let others use it solely
        if(!success)
         {
             PyErr_SetString(PyExc_RuntimeError, QObject::tr("Temporary file for writing point cloud binary data could not be created").toLatin1().data());
             return NULL;
         }

        ito::RetVal retval;
        try
        {
            if (apiFilterCall)
            {
                QVector<ito::ParamBase> paramsMand;
                QVector<ito::ParamBase> paramsOpt;
                QVector<ito::ParamBase> paramsOut;

                paramsMand.append(ito::ParamBase("pointCloud",
                          ito::ParamBase::PointCloudPtr | ito::ParamBase::In,
                          reinterpret_cast<const char*>(self->data)));
                paramsMand.append(ito::ParamBase("filename",
                          ito::ParamBase::String | ito::ParamBase::In,
                          pclTempSaveFile.fileName().toLatin1().data()));

                paramsOpt.append(ito::ParamBase("mode", ito::ParamBase::String, "b"));
                paramsOpt.append(ito::ParamBase("type", ito::ParamBase::String, "pcd"));

                retval = apiFilterCall("savePointCloud", &paramsMand, &paramsOpt, &paramsOut);
            }
            else
            {
                retval = ito::RetVal(ito::retError, 0, QObject::tr("api function pointer not set").toLatin1().data());

            }
        }
        catch(std::bad_alloc &/*ba*/)
        {
            retval += RetVal(retError, 0, QObject::tr("No more memory available when saving point cloud").toLatin1().data());
        }
        catch(std::exception &exc)
        {
            if (exc.what())
            {
                retval += ito::RetVal::format(ito::retError, 0,
                             QObject::tr("The exception '%s' has been thrown when saving a point cloud.").toLatin1().data(), exc.what());
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0,
                             QObject::tr("An unspecified exception has been thrown when saving a point cloud.").toLatin1().data());
            }
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError, 0,
                         QObject::tr("An unspecified exception has been thrown when saving a point cloud.").toLatin1().data());
        }

        if (PythonCommon::transformRetValToPyException(retval) == false)
        {
            return NULL;
        }
        if (pclTempSaveFile.open() == false)
        {
            PyErr_SetString(PyExc_RuntimeError, QObject::tr("Temporary file for writing point cloud binary data could not be opened").toLatin1().data());
            return NULL;
        }
        //allocate memory
        stateTuple = PyBytes_FromStringAndSize(NULL, pclTempSaveFile.size());
        //get pointer to datapart of memory
        char *data = PyBytes_AsString(stateTuple);
        //write to memory
        uint64_t bytesRead = pclTempSaveFile.peek(data, pclTempSaveFile.size());
        Q_ASSERT(bytesRead == pclTempSaveFile.size());//check if file was fully read...
    }
    else
    {
        Py_INCREF(Py_None);
        stateTuple = Py_None;
    }

    //the stateTuple is simply a byte array with the binary content
    //of the temporarily written pcd file or None, if the point cloud is invalid or empty
    //the type-number is passed as argument to the constructor of the point cloud class, if it is reconstructed.
    PyObject *tempOut = Py_BuildValue("(O(i)O)", Py_TYPE(self), type, stateTuple);
    Py_XDECREF(stateTuple);

    return tempOut;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_SetState(PyPointCloud *self, PyObject *args)
{
    PyObject *data = NULL;
    if (!PyArg_ParseTuple(args, "O", &data))
    {
        return NULL;
    }

    if (data == Py_None)
    {
        Py_RETURN_NONE;
    }
    else if (PyBytes_Check(data))
    {
        QTemporaryFile pclTempSaveFile;
        bool success = pclTempSaveFile.open();//creates the file
        if (!success)
        {
            PyErr_SetString(PyExc_RuntimeError, QObject::tr("Temporary file for writing point cloud binary data could not be created").toLatin1().data());
            return NULL;
        }

        qint64 bytesWritten = pclTempSaveFile.write(PyBytes_AsString(data), PyBytes_GET_SIZE(data));
        pclTempSaveFile.close();
        Q_ASSERT(bytesWritten == PyBytes_GET_SIZE(data));
        QVector<ito::ParamBase> paramsMand;
        QVector<ito::ParamBase> paramsOpt;
        QVector<ito::ParamBase> paramsOut;

        paramsMand.append(ito::ParamBase("pointCloud",
                     ito::ParamBase::PointCloudPtr | ito::ParamBase::In,
                     reinterpret_cast<char*>(self->data)));
        paramsMand.append(ito::ParamBase("filename",
                     ito::ParamBase::String | ito::ParamBase::In,
                     pclTempSaveFile.fileName().toLatin1().data()));
        paramsOpt.append(ito::ParamBase("type", ito::ParamBase::String | ito::ParamBase::In, "pcd"));

        ito::RetVal retval;

        try
        {
            if (apiFilterCall)
                retval = apiFilterCall("loadPointCloud", &paramsMand, &paramsOpt, &paramsOut);
            else
                retval = ito::RetVal(ito::retError, 0, QObject::tr("api function pointer not set").toLatin1().data());
        }
        catch(std::bad_alloc &/*ba*/)
        {
            retval += RetVal(retError, 0, QObject::tr("No more memory available when loading point cloud").toLatin1().data());
        }
        catch(std::exception &exc)
        {
            if (exc.what())
            {
                retval += ito::RetVal::format(ito::retError, 0,
                             QObject::tr("The exception '%s' has been thrown when loading a point cloud.").toLatin1().data(), exc.what());
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0,
                             QObject::tr("An unspecified exception has been thrown when loading a point cloud.").toLatin1().data());
            }
        }
        catch (...)
        {
            retval += ito::RetVal(ito::retError, 0,
                         QObject::tr("An unspecified exception has been thrown when loading a point cloud.").toLatin1().data());
        }

        if (PythonCommon::transformRetValToPyException(retval) == false)
        {
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError,
                     QObject::tr("The pickled data must be a byte array for establishing the pointCloud.").toLatin1().data());
        return NULL;
    }

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFromXYZ_doc,"fromXYZ(X, Y, Z, deleteNaN = False | XYZ, deleteNaN = False) -> creates a point cloud from three X,Y,Z data objects or from one 3xMxN data object\n\
\n\
The created point cloud is not organized (height=1) and dense, if no NaN or Inf values are within the point cloud (deleteNaN:true results in a dense point cloud) \n\
\n\
Parameters \n\
----------- \n\
X,Y,Z : {MxN data objects} \n\
    Three 2D data objects with the same size.\n\
XYZ : {3xMxN or Mx3 data object} \n\
    either 3xMxN data object, such that the first plane is X, the second is Y and the third is Z, \n\
    or: Mx3 data object, such that the point cloud consists of M points where every coordinate is given by the three values in each row. \n\
deleteNaN : {bool} \n\
    default = false\n\
    if True all NaN values are skipped, hence, the resulting point cloud is not dense any more\n\
\n\
Returns \n\
------- \n\
pointCloud.");
/*static*/ PyObject* PythonPCL::PyPointCloud_fromXYZ(PyPointCloud * /*self*/, PyObject *args)
{
    PyObject *objX = NULL;
    PyObject *objY = NULL;
    PyObject *objZ = NULL;
    bool deleteNaN = false;

    QSharedPointer<ito::DataObject> X, Y, Z, XYZ;
    bool ok = true;
    ito::RetVal retval = ito::retOk;

    if (!PyArg_ParseTuple(args,"OOO|b", &objX, &objY, &objZ, &deleteNaN))
    {
        if (!PyArg_ParseTuple(args, "O|b", &objX, &deleteNaN))
        {
            return NULL;
        }
        else
        {
            XYZ = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objX, false, ok, &retval));
            if (!ok)
            {
                return PyErr_Format(PyExc_RuntimeError, "XYZ argument could not be converted to a data object (%s)", retval.errorMessage());
            }

            ito::RetVal tmpRetval = ito::dObjHelper::verify3DDataObject(XYZ.data(), "XYZ", 3, 3, 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
            if (tmpRetval.containsWarningOrError() && XYZ.data()->getDims() == 2)
            {
                retval += ito::dObjHelper::verify2DDataObject(XYZ.data(), "XYZ", 1, std::numeric_limits<int>::max(), 3, 3, ito::tFloat32);

                if (PythonCommon::transformRetValToPyException(retval) == false)
                {
                    return NULL;
                }

                ito::Range ranges[2] = { ito::Range::all(), ito::Range(0,0) };

                ranges[1] = ito::Range(0,1);
                X = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));

                ranges[1] = ito::Range(1,2);
                Y = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()) );

                ranges[1] = ito::Range(2,3);
                Z = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));
            }
            else
            {
                retval += tmpRetval;

                if (PythonCommon::transformRetValToPyException(retval) == false)
                {
                    return NULL;
                }

                ito::Range ranges[3] = { ito::Range(0,0), ito::Range::all(), ito::Range::all() };

                ranges[0] = ito::Range(0,1);
                X = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));

                ranges[0] = ito::Range(1,2);
                Y = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()) );

                ranges[0] = ito::Range(2,3);
                Z = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));
            }
        }
    }
    else
    {
        X = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objX, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "X argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        Y = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objY, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Y argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        Z = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objZ, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Z argument could not be converted to a data object (%s)", retval.errorMessage());
        }
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    if (cloud != NULL)
    {
        cloud->data = new ito::PCLPointCloud(ito::pclInvalid);
        retval += ito::pclHelper::pointCloudFromXYZ(X.data(), Y.data(), Z.data(), *(cloud->data), deleteNaN);
    }

    if (PythonCommon::transformRetValToPyException(retval) == false)
    {
        Py_XDECREF(cloud);
        cloud = NULL;
        return NULL;
    }
    else
    {
        return (PyObject*)cloud;
    }
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFromXYZI_doc,"fromXYZI(X, Y, Z, I, deleteNaN = False | XYZ, I, deleteNaN = False) -> creates a point cloud from four X,Y,Z,I data objects or from one 3xMxN data object and one intensity data object\n\
\n\
The created point cloud is not organized (height=1) and dense, if no NaN or Inf values are within the point cloud (deleteNaN:true results in a dense point cloud) \n\
\n\
Parameters \n\
----------- \n\
X,Y,Z,I : {MxN data objects} \n\
    Four 2D data objects with the same size.\n\
OR \n\
XYZ : {3xMxN data object} \n\
    3xMxN data object, such that the first plane is X, the second is Y and the third is Z\n\
I : {MxN data object} \n\
    MxN data object with the same size than the single X,Y or Z planes \n\
deleteNaN : {bool} \n\
    default = false\n\
    if True all NaN values in X, Y or Z data objects are skipped, hence, the resulting point cloud is not dense any more\n\
\n\
Returns \n\
------- \n\
pointCloud.");
/*static*/ PyObject* PythonPCL::PyPointCloud_fromXYZI(PyPointCloud * /*self*/, PyObject *args)
{
    PyObject *objX = NULL;
    PyObject *objY = NULL;
    PyObject *objZ = NULL;
    PyObject *objI = NULL;
    bool deleteNaN = false;

    QSharedPointer<ito::DataObject> X, Y, Z, XYZ, I;
    bool ok = true;
    ito::RetVal retval = ito::retOk;

    if (!PyArg_ParseTuple(args,"OOOO|b", &objX, &objY, &objZ, &objI, &deleteNaN))
    {
        if (!PyArg_ParseTuple(args, "OO|b", &objX, &objI, &deleteNaN))
        {
            return NULL;
        }
        else
        {
            XYZ = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objX, false, ok, &retval));
            if (!ok)
            {
                return PyErr_Format(PyExc_RuntimeError, "XYZ argument could not be converted to a data object (%s)", retval.errorMessage());
            }

            ito::RetVal tmpRetval = ito::dObjHelper::verify3DDataObject(XYZ.data(), "XYZ", 3, 3, 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
            if (tmpRetval.containsWarningOrError())
            {
                ito::RetVal tmpRetval = ito::dObjHelper::verify2DDataObject(XYZ.data(), "XYZ", 1, std::numeric_limits<int>::max(), 3, 3, ito::tFloat32);
                if (PythonCommon::transformRetValToPyException(retval) == false)
                {
                    return NULL;
                }

                ito::Range ranges[2] = { ito::Range::all(), ito::Range(0,0) };

                ranges[1] = ito::Range(0,1);
                X = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));

                ranges[1] = ito::Range(1,2);
                Y = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()) );

                ranges[1] = ito::Range(2,3);
                Z = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));
            }
            else
            {
                ito::Range ranges[3] = { ito::Range(0,0), ito::Range::all(), ito::Range::all() };

                ranges[0] = ito::Range(0,1);
                X = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));

                ranges[0] = ito::Range(1,2);
                Y = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()) );

                ranges[0] = ito::Range(2,3);
                Z = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));
            }

            I = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objI, false, ok, &retval));
            if (!ok)
            {
                return PyErr_Format(PyExc_RuntimeError, "Intensity argument could not be converted to a data object (%s)", retval.errorMessage());
            }
        }
    }
    else
    {
        X = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objX, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "X argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        Y = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objY, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Y argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        Z = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objZ, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Z argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        I = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objI, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Intensity argument could not be converted to a data object (%s)", retval.errorMessage());
        }
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    cloud->data = new ito::PCLPointCloud(ito::pclInvalid);
    retval += ito::pclHelper::pointCloudFromXYZI(X.data(), Y.data(), Z.data(), I.data(), *(cloud->data), deleteNaN);

    if (PythonCommon::transformRetValToPyException(retval) == false)
    {
        Py_XDECREF(cloud);
        cloud = NULL;
        return NULL;
    }
    else
    {
        return (PyObject*)cloud;
    }
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFromXYZRGBA_doc,"fromXYZRGBA(X, Y, Z, color, deleteNaN = False | XYZ, color, deleteNaN = False) -> creates a point cloud from four X,Y,Z,color data objects or from one 3xMxN data object and one coloured data object\n\
\n\
The created point cloud is not organized (height=1) and dense, if no NaN or Inf values are within the point cloud (deleteNaN:true results in a dense point cloud) \n\
\n\
Parameters \n\
----------- \n\
X,Y,Z,color : {MxN data objects} \n\
    Four 2D data objects with the same size. X,Y,Z must have the same type, color must be rgba32.\n\
OR \n\
XYZ : {3xMxN data object} \n\
    3xMxN data object, such that the first plane is X, the second is Y and the third is Z\n\
color : {MxN data object} \n\
    MxN data object with the same size than the single X,Y or Z planes, type: rgba32 \n\
deleteNaN : {bool} \n\
    default = false\n\
    if True all NaN or Inf values are skipped, hence, the resulting point cloud does not contain this values\n\
\n\
Returns \n\
------- \n\
pointCloud.");
/*static*/ PyObject* PythonPCL::PyPointCloud_fromXYZRGBA(PyPointCloud * /*self*/, PyObject *args)
{
    PyObject *objX = NULL;
    PyObject *objY = NULL;
    PyObject *objZ = NULL;
    PyObject *objColor = NULL;
    bool deleteNaN = false;

    QSharedPointer<ito::DataObject> X, Y, Z, XYZ, color;
    bool ok = true;
    ito::RetVal retval = ito::retOk;

    if (!PyArg_ParseTuple(args,"OOOO|b", &objX, &objY, &objZ, &objColor, &deleteNaN))
    {
        if (!PyArg_ParseTuple(args, "OO|b", &objX, &objColor, &deleteNaN))
        {
            return NULL;
        }
        else
        {
            XYZ = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objX, false, ok, &retval));
            if (!ok)
            {
                return PyErr_Format(PyExc_RuntimeError, "XYZ argument could not be converted to a data object (%s)", retval.errorMessage());
            }

            ito::RetVal tmpRetval = ito::dObjHelper::verify3DDataObject(XYZ.data(), "XYZ", 3, 3, 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
            if (tmpRetval.containsWarningOrError())
            {
                ito::RetVal tmpRetval = ito::dObjHelper::verify2DDataObject(XYZ.data(), "XYZ", 1, std::numeric_limits<int>::max(), 3, 3, ito::tFloat32);
                if (PythonCommon::transformRetValToPyException(retval) == false)
                {
                    return NULL;
                }

                ito::Range ranges[2] = { ito::Range::all(), ito::Range(0,0) };

                ranges[1] = ito::Range(0,1);
                X = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));

                ranges[1] = ito::Range(1,2);
                Y = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()) );

                ranges[1] = ito::Range(2,3);
                Z = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));
            }
            else
            {
                ito::Range ranges[3] = { ito::Range(0,0), ito::Range::all(), ito::Range::all() };

                ranges[0] = ito::Range(0,1);
                X = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));

                ranges[0] = ito::Range(1,2);
                Y = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()) );

                ranges[0] = ito::Range(2,3);
                Z = QSharedPointer<ito::DataObject>(new ito::DataObject(XYZ->at(ranges).squeeze()));
            }

            color = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objColor, false, ok, &retval));
            if (!ok)
            {
                return PyErr_Format(PyExc_RuntimeError, "color argument could not be converted to a data object (%s)", retval.errorMessage());
            }
        }
    }
    else
    {
        X = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objX, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "X argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        Y = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objY, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Y argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        Z = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objZ, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "Z argument could not be converted to a data object (%s)", retval.errorMessage());
        }

        color = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objColor, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "color argument could not be converted to a data object (%s)", retval.errorMessage());
        }
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    cloud->data = new ito::PCLPointCloud(ito::pclInvalid);
    retval += ito::pclHelper::pointCloudFromXYZRGBA(X.data(), Y.data(), Z.data(), color.data(), *(cloud->data), deleteNaN);

    if (PythonCommon::transformRetValToPyException(retval) == false)
    {
        Py_XDECREF(cloud);
        cloud = NULL;
        return NULL;
    }
    else
    {
        return (PyObject*)cloud;
    }
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFromDisparity_doc,"fromDisparity(disparity, intensity = None, deleteNaN = False, color = None) -> creates a point cloud from a given topography dataObject.\n\
\n\
Creates a point cloud from the 2.5D data set given by the topography dataObject. The x and y-components of each point are taken from the regular grid \n\
values of 'topography' (considering the scaling and offset of the object). The corresponding z-value is the topography's value itself. \n\
\n\
This method is deprecated and has been renamed to 'fromTopography' due to the wrong usage of the name topography in this case. \n\
\n\
Parameters \n\
----------- \n\
disparity : {MxN data object, float32} \n\
    The values of this dataObject represent the z-components.\n\
intensity : {MxN data object, float32}, optional \n\
    If given, an XYZI-point cloud is created whose intensity values are determined by this dataObject (cannot be used together with 'color')\n\
deleteNaN : {bool}, optional \n\
    If true (default: false), NaN or Inf-values (z) in the topography map will not be copied into the point cloud (the point cloud is not organized any more).\n\
color : {MxN data object, rgba32}, optional \n\
    If given, a XYZRGBA-point cloud is created whose color values are determined by this dataObject (cannot be used together with 'intensity')\n\
\n\
Returns \n\
------- \n\
pointCloud.");
//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFromTopography_doc, "fromTopography(topography, intensity = None, deleteNaN = False, color = None) -> creates a point cloud from a given topography dataObject.\n\
\n\
Creates a point cloud from the 2.5D data set given by the topography dataObject. The x and y-components of each point are taken from the regular grid \n\
values of 'topography' (considering the scaling and offset of the object). The corresponding z-value is the topography's value itself. \n\
\n\
Parameters \n\
----------- \n\
topography : {MxN data object, float32} \n\
    The values of this dataObject represent the topography values.\n\
intensity : {MxN data object, float32}, optional \n\
    If given, an XYZI-point cloud is created whose intensity values are determined by this dataObject (cannot be used together with 'color')\n\
deleteNaN : {bool}, optional \n\
    If true (default: false), NaN or Inf-values (z) in the disparity map will not be copied into the point cloud (the point cloud is not organized any more).\n\
color : {MxN data object, rgba32}, optional \n\
    If given, a XYZRGBA-point cloud is created whose color values are determined by this dataObject (cannot be used together with 'intensity')\n\
\n\
Returns \n\
------- \n\
PointCloud.");
/*static*/ PyObject* PythonPCL::PyPointCloud_fromTopography(PyPointCloud * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *objDisp = NULL;
    PyObject *objI = NULL;
    PyObject *objColor = NULL;
    bool deleteNaN = false;
    const char *kwlist[] = {"topography", "intensity", "deleteNaN", "color", NULL};
    const char *kwlist2[] = { "disparity", "intensity", "deleteNaN", "color", NULL };

    QSharedPointer<ito::DataObject> dispMap, IntMap, colorMap;
    bool ok = true;
    ito::RetVal retval = ito::retOk;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ObO", const_cast<char**>(kwlist), &objDisp, &objI, &deleteNaN, &objColor))
    {
        PyErr_Clear();
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ObO", const_cast<char**>(kwlist2), &objDisp, &objI, &deleteNaN, &objColor))
        {
            return NULL;
        }
    }

    dispMap = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objDisp, false, ok, &retval));
    if (!ok)
    {
        return PyErr_Format(PyExc_RuntimeError, "topography map argument could not be converted to a data object (%s)", retval.errorMessage());
    }

    if (objI)
    {
        IntMap = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objI, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "topography map argument could not be converted to a data object (%s)", retval.errorMessage());
        }
        colorMap = QSharedPointer<ito::DataObject>(NULL);
    }
    else if (objColor)
    {
        colorMap = QSharedPointer<ito::DataObject>(PythonQtConversion::PyObjGetDataObjectNewPtr(objColor, false, ok, &retval));
        if (!ok)
        {
            return PyErr_Format(PyExc_RuntimeError, "color map argument could not be converted to a data object (%s)", retval.errorMessage());
        }
        IntMap = QSharedPointer<ito::DataObject>(NULL);
    }
    else if (objI && objColor)
    {
        PyErr_SetString(PyExc_RuntimeError, "either give an intensity map or a color map");
        return NULL;
    }
    else
    {
        IntMap = QSharedPointer<ito::DataObject>(NULL);
        colorMap = QSharedPointer<ito::DataObject>(NULL);
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    cloud->data = new ito::PCLPointCloud(ito::pclInvalid);

    if (!objColor)
    {
        retval += ito::pclHelper::pointCloudFromDisparityI(dispMap.data(), IntMap.data(), *(cloud->data), deleteNaN);
    }
    else
    {
        retval += ito::pclHelper::pointCloudFromDisparityRGBA(dispMap.data(), colorMap.data(), *(cloud->data), deleteNaN);
    }

    if (PythonCommon::transformRetValToPyException(retval) == false)
    {
        Py_XDECREF(cloud);
        cloud = NULL;
        return NULL;
    }
    else
    {
        return (PyObject*)cloud;
    }
}

//---------------------------------------------------------------------------------------
/*static*/ PythonPCL::PyPointCloud* PythonPCL::createEmptyPyPointCloud()
{
    PyPointCloud* result = (PyPointCloud*)PyObject_Call((PyObject*)&PyPointCloudType, NULL, NULL);
    if (result != NULL)
    {
        DELETE_AND_SET_NULL(result->data);
        return result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//---------------------------------------------------------------------------------------
PyGetSetDef PythonPCL::PyPointCloud_getseters[] = {
    {"type",        (getter)PyPointCloud_GetType,       NULL,                           pyPointCloudType_doc,     NULL},
    {"size",        (getter)PyPointCloud_GetSize,       NULL,                           pyPointCloudSize_doc,     NULL},
    {"height",      (getter)PyPointCloud_GetHeight,     NULL,                           pyPointCloudHeight_doc,   NULL},
    {"width",       (getter)PyPointCloud_GetWidth,      NULL,                           pyPointCloudWidth_doc,    NULL},
    {"empty",       (getter)PyPointCloud_GetEmpty,      NULL,                           pyPointCloudEmpty_doc,    NULL},
    {"organized",   (getter)PyPointCloud_GetOrganized,  NULL,                           pyPointCloudOrganized_doc,NULL},
    {"dense",       (getter)PyPointCloud_GetDense,      (setter)PyPointCloud_SetDense,  pyPointCloudDense_doc,    NULL},
    {"fields",      (getter)PyPointCloud_GetFields,     NULL,                           pyPointCloudFields_doc,   NULL},
    {NULL}  /* Sentinel */
};

//---------------------------------------------------------------------------------------
PyMethodDef PythonPCL::PyPointCloud_methods[] = {
    {"name",          (PyCFunction)PyPointCloud_name, METH_NOARGS, "name"},
    {"scaleXYZ",      (PyCFunction)PyPointCloud_scaleXYZ, METH_KEYWORDS | METH_VARARGS, pyPointCloudScaleXYZ_doc },
    {"moveXYZ",      (PyCFunction)PyPointCloud_moveXYZ, METH_KEYWORDS | METH_VARARGS, pyPointCloudMoveXYZ_doc },
    {"append",        (PyCFunction)PyPointCloud_append, METH_KEYWORDS | METH_VARARGS, pyPointCloudAppend_doc},
    {"clear",         (PyCFunction)PyPointCloud_clear, METH_NOARGS, pyPointCloudClear_doc},
    {"insert",        (PyCFunction)PyPointCloud_insert, METH_VARARGS, pyPointCloudInsert_doc},
    {"erase",         (PyCFunction)PyPointCloud_erase, METH_VARARGS, pyPointCloudErase_doc},
    {"toDataObject",  (PyCFunction)PyPointCloud_toDataObject, METH_NOARGS, pyPointCloudToDataObject_doc},
    {"__reduce__",    (PyCFunction)PyPointCloud_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
    {"__setstate__",  (PyCFunction)PyPointCloud_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},

    {"fromXYZ",       (PyCFunction)PyPointCloud_fromXYZ, METH_VARARGS | METH_STATIC, pyPointCloudFromXYZ_doc},
    {"fromXYZI",      (PyCFunction)PyPointCloud_fromXYZI, METH_VARARGS | METH_STATIC, pyPointCloudFromXYZI_doc},
    {"fromXYZRGBA",   (PyCFunction)PyPointCloud_fromXYZRGBA, METH_VARARGS | METH_STATIC, pyPointCloudFromXYZRGBA_doc},
    {"fromDisparity", (PyCFunction)PyPointCloud_fromTopography, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyPointCloudFromDisparity_doc},
    {"fromTopography",  (PyCFunction)PyPointCloud_fromTopography, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyPointCloudFromTopography_doc },

    {"copy",          (PyCFunction)PyPointCloud_copy, METH_NOARGS, pyPointCloudCopy_doc},

    {NULL}  /* Sentinel */
};

//---------------------------------------------------------------------------------------
PySequenceMethods PythonPCL::PyPointCloud_sequenceProtocol = {
    (lenfunc)PyPointCloud_seqLength,                      /*sq_length*/
    (binaryfunc)PyPointCloud_seqConcat,               /*sq_concat*/
    (ssizeargfunc)PyPointCloud_seqRepeat,                 /*sq_repeat*/
    (ssizeargfunc)PyPointCloud_seqItem,                           /*sq_item*/
    0,                                          /*sq_slice*/
    (ssizeobjargproc)PyPointCloud_seqAssItem,                    /*sq_ass_item*/
    0,                                          /*sq_ass_slice*/
    (objobjproc)NULL,                 /*sq_contains*/
    (binaryfunc)PyPointCloud_seqInplaceConcat,           /*sq_inplace_concat*/
    (ssizeargfunc)PyPointCloud_seqInplaceRepeat          /*sq_inplace_repeat*/
};

//---------------------------------------------------------------------------------------
PyMappingMethods PythonPCL::PyPointCloud_mappingProtocol = {
    (lenfunc)PyPointCloud_mappingLength,
    (binaryfunc)PyPointCloud_mappingGetElem,
    (objobjargproc)PyPointCloud_mappingSetElem
};

//---------------------------------------------------------------------------------------
PyModuleDef PythonPCL::PyPointCloudModule = {
    PyModuleDef_HEAD_INIT, "PointCloud", "PointCloud wrapper for pcl::PointCloud<PointT>", -1,
    NULL, NULL, NULL, NULL, NULL
};

//---------------------------------------------------------------------------------------
PyTypeObject PythonPCL::PyPointCloudType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.pointCloud",             /* tp_name */
    sizeof(PyPointCloud),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PythonPCL::PyPointCloud_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PythonPCL::PyPointCloud_repr,                         /* tp_repr */
    0,                         /* tp_as_number */
    &PyPointCloud_sequenceProtocol,                         /* tp_as_sequence */
    &PyPointCloud_mappingProtocol,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    pointCloudInit_doc,           /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    PythonPCL::PyPointCloud_methods,             /* tp_methods */
    0, /*PyNpDataObject_members,*/             /* tp_members */
    PythonPCL::PyPointCloud_getseters,                         /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PythonPCL::PyPointCloud_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PythonPCL::PyPointCloud_new         /* tp_new */
};

//---------------------------------------------------------------------------------------
// private helpers for PyPointCloud
//---------------------------------------------------------------------------------------
PyObject* PythonPCL::parseObjAsFloat32Array(PyObject *obj, npy_intp mRequired, npy_intp &n, float32 **elemRows)
{
    if (mRequired < 1 || mRequired > 7)
    {
        PyErr_SetString(PyExc_RuntimeError, "the number of required rows must be between 1 and 7");
        return NULL;
    }
#if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#else
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#endif

    if (arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "argument cannot be interpreted as float32, c-contiguous numpy.ndarray");
        return NULL;
    }

    //check arr
    if (PyArray_NDIM((PyArrayObject*)arr) != 2)
    {
        Py_XDECREF(arr);
        PyErr_SetString(PyExc_RuntimeError, "input array must have two dimensions");
        return NULL;
    }
    if (PyArray_DIM((PyArrayObject*)arr,0) != mRequired)
    {
        Py_XDECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "ndArray must have %c rows", mRequired);
        return NULL;
    }

    n = PyArray_DIM((PyArrayObject*)arr,1);
    npy_intp strideDim0 = PyArray_STRIDE((PyArrayObject*)arr,0);

    for (npy_intp i = 0; i<mRequired;i++)
    {
        elemRows[i] = reinterpret_cast<float32*>(PyArray_BYTES((PyArrayObject*)arr) + i*strideDim0);
    }

    return arr;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::parseObjAsUInt8Array(PyObject *obj, npy_intp mRequired, npy_intp &n, uint8_t **elemRows)
{
    if (mRequired < 1 || mRequired > 4)
    {
        PyErr_SetString(PyExc_RuntimeError, "the number of required rows must be between 1 and 4");
        return NULL;
    }

#if (NPY_FEATURE_VERSION < NPY_1_7_API_VERSION)
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_UBYTE, NPY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#else
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_UBYTE, NPY_ARRAY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#endif

    if (arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "argument cannot be interpreted as uint8, c-contiguous numpy.ndarray");
        return NULL;
    }

    //check arr
    if (PyArray_NDIM((PyArrayObject*)arr) != 2)
    {
        Py_XDECREF(arr);
        PyErr_SetString(PyExc_RuntimeError, "input array must have two dimensions");
        return NULL;
    }
    if (PyArray_DIM((PyArrayObject*)arr,0) != mRequired)
    {
        Py_XDECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "ndArray must have %c rows", mRequired);
        return NULL;
    }

    n = PyArray_DIM((PyArrayObject*)arr,1);
    npy_intp strideDim0 = PyArray_STRIDE((PyArrayObject*)arr,0);

    for (npy_intp i = 0; i<mRequired;i++)
    {
        elemRows[i] = reinterpret_cast<uint8_t*>(PyArray_BYTES((PyArrayObject*)arr) + i*strideDim0);
    }

    return arr;
}


//---------------------------------------------------------------------------------------
//
// PyPoint
//
//---------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPoint_dealloc(PyPoint* self)
{
    Py_XDECREF(self->base);
    DELETE_AND_SET_NULL(self->point);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPoint_new(PyTypeObject *type, PyObject* /*args*/, PyObject * /*kwds*/)
{
    PyPoint* self = (PyPoint *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->base = NULL;
        self->point = NULL;
    }

    return (PyObject *)self;
};

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pointInit_doc,"point(type = point.PointInvalid, xyz = None, intensity = None, rgba = None, normal = None, curvature = None) -> creates new point used for class 'pointCloud'.  \n\
\n\
Parameters \n\
------------ \n\
type : {int} \n\
    the desired type of this point (default: point.PointInvalid). Depending on the type, some of the following parameters must be given: \n\
xyz : {seq}, all types besides PointInvalid \n\
    sequence with three floating point elements (x,y,z) \n\
intensity : {float}, only PointXYZI or PointXYZINormal \n\
    is a floating point value for the intensity \n\
rgba, {seq. of uint8, three or four values}, only PointXYZRGBA or PointXYZRGBNormal \n\
    a uint8-sequence with either three or four values (r,g,b,a). If alpha value is not given, 255 is assumed \n\
normal : {seq}, only PointXYZNormal, PointXYZINormal and PointXYZRGBNormal \n\
    is a sequence with three floating point elements (nx, ny, nz) \n\
curvature : {float}, only PointXYZNormal, PointXYZINormal and PointXYZRGBNormal \n\
    is the curvature value for the normal (float)");
int PythonPCL::PyPoint_init(PyPoint *self, PyObject *args, PyObject *kwds)
{
    int pclType = ito::pclInvalid;

    if (args == NULL) //point will be allocated later
    {
        self->point = NULL;
        return 0;
    }

    if (PyTuple_Size(args) > 0 || kwds != NULL)
    {
        PyObject *temp = kwds == NULL ? (PyObject*)NULL : PyDict_GetItemString(kwds, "type"); //borrowed
        if (temp == NULL && PyTuple_Size(args) > 0) temp = PyTuple_GetItem(args,0); //borrowed
        if (temp)
        {
            if (!PyLong_Check(temp))
            {
                PyErr_SetString(PyExc_TypeError, "The argument must contain the type of the point, e.g. point.PointXYZ");
                return -1;
            }
            pclType = PyLong_AsLong(temp);
        }
    }

    switch(pclType)
    {
        case ito::pclInvalid:
        {
            self->point = new ito::PCLPoint();
            break;
        }
        case ito::pclXYZ:
        {
            static const char *kwlist[] = {"type","xyz", NULL};
            float x,y,z;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)",const_cast<char**>(kwlist),&pclType,&x,&y,&z)) return -1;
            self->point = new ito::PCLPoint(pcl::PointXYZ(x,y,z));
            break;
        }
        case ito::pclXYZI:
        {
            static const char *kwlist[] = {"type","xyz","intensity", NULL};
            float x,y,z,intensity;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)f",const_cast<char**>(kwlist),&pclType,&x,&y,&z,&intensity)) return -1;
            pcl::PointXYZI pt(intensity);
            pt.x=x;pt.y=y;pt.z=z;
            self->point = new ito::PCLPoint(pt);
            break;
        }
        case ito::pclXYZRGBA:
        {
            static const char *kwlist[] = {"type","xyz","rgba", NULL};
            float x,y,z;
            uint8_t r,g,b,a;
            a=255;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhhh)",
                         const_cast<char**>(kwlist),&pclType,&x,&y,&z,&r,&g,&b,&a))
            {
                PyErr_Clear();
                if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhh)",
                             const_cast<char**>(kwlist),&pclType,&x,&y,&z,&r,&g,&b)) return -1;
                a=255;
            }
            pcl::PointXYZRGBA pt;
            pt.x=x;pt.y=y;pt.z=z;
            pt.r=r;pt.g=g;pt.b=b;pt.PCLALPHA=a;
            self->point = new ito::PCLPoint(pt);
            break;
        }
        case ito::pclXYZNormal:
        {
            static const char *kwlist[] = {"type","xyz","normal","curvature", NULL};
            float x,y,z;
            float nx,ny,nz, curvature;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(fff)f",const_cast<char**>(kwlist),&pclType,&x,&y,&z,&nx,&ny,&nz,&curvature)) return -1;
            pcl::PointNormal pt;
            pt.x = x; pt.y = y; pt.z = z;
            pt.normal_x = nx; pt.normal_y = ny; pt.normal_z = nz;
            pt.curvature = curvature;
            self->point = new ito::PCLPoint(pt);
            break;
        }
        case ito::pclXYZINormal:
        {
            static char const *kwlist[] = {"type","xyz","intensity","normal","curvature", NULL};
            float x,y,z, intensity;
            float nx,ny,nz, curvature;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)f(fff)f",const_cast<char**>(kwlist),&pclType,&x,&y,&z,&intensity,&nx,&ny,&nz,&curvature)) return -1;
            pcl::PointXYZINormal pt;
            pt.x = x; pt.y = y; pt.z = z;
            pt.normal_x = nx; pt.normal_y = ny; pt.normal_z = nz;
            pt.curvature = curvature;
            pt.intensity = intensity;
            self->point = new ito::PCLPoint(pt);
            break;
        }
        case ito::pclXYZRGBNormal:
        {
            static const char *kwlist[] = {"type","xyz","rgba","normal","curvature", NULL};
            float x,y,z;
            float nx,ny,nz, curvature;
            uint8_t r,g,b,a;
            a=255;

            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhhh)(fff)f",const_cast<char**>(kwlist),
                         &pclType,&x,&y,&z,&r,&g,&b,&a,&nx,&ny,&nz,&curvature))
            {
                PyErr_Clear();
                if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhh)(fff)f",const_cast<char**>(kwlist),
                             &pclType,&x,&y,&z,&r,&g,&b,&nx,&ny,&nz,&curvature)) return -1;
                a=255;
            }

            pcl::PointXYZRGBNormal pt;
            pt.x = x; pt.y = y; pt.z = z;
            pt.normal_x = nx; pt.normal_y = ny; pt.normal_z = nz;
            pt.curvature = curvature;
            pt.r = r; pt.g = g; pt.b = b; pt.PCLALPHA = a;
            self->point = new ito::PCLPoint(pt);
            break;
        }
        default:
            PyErr_SetString(PyExc_TypeError, "The point type is unknown");
            return -1;
    }
    return 0;
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPoint_name(PyPoint* /*self*/)
{
    return PyUnicode_FromString("Point");
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPoint_repr(PyPoint *self)
{
    if (self->point == NULL)
    {
        return PyUnicode_FromString("Point (empty)");
    }
    else
    {
        QString str;
        switch(self->point->getType())
        {
        case ito::pclXYZ:
            {
                const pcl::PointXYZ pt = self->point->getPointXYZ();
                str = QString("Point (XYZ=[%1,%2,%3])").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3);
                break;
            }
        case ito::pclXYZI:
            {
                const pcl::PointXYZI pt = self->point->getPointXYZI();
                str = QString("Point (XYZ=[%1,%2,%3], intensity=%4)").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3).arg(pt.intensity, 0, 'G', 1);
                break;
            }
        case ito::pclXYZRGBA:
            {
                const pcl::PointXYZRGBA pt = self->point->getPointXYZRGBA();
                str = QString("Point (XYZ=[%1,%2,%3], RGB=[%4,%5,%6], alpha=%7)").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3).arg(pt.r).arg(pt.g).arg(pt.b).arg(pt.PCLALPHA);
                break;
            }
        case ito::pclXYZNormal:
            {
                const pcl::PointNormal pt = self->point->getPointXYZNormal();
                str = QString("Point (XYZ=[%1,%2,%3], Normal=[%4,%5,%6], curvature=%7)").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3).arg(pt.normal_x, 0, 'G', 3).arg(pt.normal_y, 0, 'G', 3).arg(pt.normal_z, 0, 'G', 3).arg(pt.curvature, 0, 'G', 3);
                break;
            }
        case ito::pclXYZINormal:
            {
                const pcl::PointXYZINormal pt = self->point->getPointXYZINormal();
                str = QString("Point (XYZ=[%1,%2,%3], Normal=[%4,%5,%6], curvature=%7, intensity=%8)").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3).arg(pt.normal_x, 0, 'G', 3).arg(pt.normal_y, 0, 'G', 3).arg(pt.normal_z, 0, 'G', 3).arg(pt.curvature, 0, 'G', 3).arg(pt.intensity, 0, 'G', 3);
                break;
            }
        case ito::pclXYZRGBNormal:
            {
                const pcl::PointXYZRGBNormal pt = self->point->getPointXYZRGBNormal();
                str = QString("Point (XYZ=[%1,%2,%3], Normal=[%4,%5,%6], curvature=%7, RGB=[%8,%9,%10], alpha=%11)").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3).arg(pt.normal_x, 0, 'G', 3).arg(pt.normal_y, 0, 'G', 3).arg(pt.normal_z, 0, 'G', 3).arg(pt.curvature, 0, 'G', 3).arg(pt.r).arg(pt.g).arg(pt.b).arg(pt.PCLALPHA);
                break;
            }
        default:
            return PyUnicode_FromString("Point (invalid)");
        }

        QByteArray ba = str.toLatin1();
        return PyUnicode_FromString(ba.data());
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPoint_mappingGetElem(PyPoint* self, PyObject* key)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point is invalid");
        return NULL;
    }
    bool ok;
    int type = self->point->getType();

    QString keyString = PythonQtConversion::PyObjGetString(key,true,ok);
    if (ok == false)
    {
        PyErr_SetString(PyExc_ValueError, "key must be a string");
        return NULL;
    }

    switch(type)
    {
    case ito::pclXYZ:
        {
            const pcl::PointXYZ pt = self->point->getPointXYZ();

            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.x, pt.y, pt.z);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz'");
                return NULL;
            }
            break;
        }
    case ito::pclXYZI:
        {
            const pcl::PointXYZI pt = self->point->getPointXYZI();
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.x, pt.y, pt.z);
            }
            else if (QString::compare(keyString, "intensity", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("f", pt.intensity);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz' or 'intensity'");
                return NULL;
            }
            break;
        }
    case ito::pclXYZRGBA:
        {
            const pcl::PointXYZRGBA pt = self->point->getPointXYZRGBA();
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.x, pt.y, pt.z);
            }
            else if (QString::compare(keyString, "rgba", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(hhhh)", pt.r, pt.g, pt.b, pt.PCLALPHA);
            }
            else if (QString::compare(keyString, "rgb", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(hhh)", pt.r, pt.g, pt.b);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'rgb' or 'rgba'");
                return NULL;
            }
            break;
        }
    case ito::pclXYZNormal:
        {
            const pcl::PointNormal pt = self->point->getPointXYZNormal();
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.x, pt.y, pt.z);
            }
            else if (QString::compare(keyString, "normal", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.normal_x, pt.normal_y, pt.normal_z);
            }
            else if (QString::compare(keyString, "curvature", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("f", pt.curvature);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'normal' or 'curvature'");
                return NULL;
            }
            break;
        }
    case ito::pclXYZINormal:
        {
            const pcl::PointXYZINormal pt = self->point->getPointXYZINormal();
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.x, pt.y, pt.z);
            }
            else if (QString::compare(keyString, "normal", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.normal_x, pt.normal_y, pt.normal_z);
            }
            else if (QString::compare(keyString, "curvature", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("f", pt.curvature);
            }
            else if (QString::compare(keyString, "intensity", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("f", pt.intensity);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature' or 'intensity'");
                return NULL;
            }
            break;
        }
    case ito::pclXYZRGBNormal:
        {
            const pcl::PointXYZRGBNormal pt = self->point->getPointXYZRGBNormal();
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.x, pt.y, pt.z);
            }
            else if (QString::compare(keyString, "normal", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(fff)", pt.normal_x, pt.normal_y, pt.normal_z);
            }
            else if (QString::compare(keyString, "curvature", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("f", pt.curvature);
            }
            else if (QString::compare(keyString, "rgba", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(hhhh)", pt.r, pt.g, pt.b, pt.PCLALPHA);
            }
            else if (QString::compare(keyString, "rgb", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(hhh)", pt.r, pt.g, pt.b, pt.PCLALPHA);
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature', 'rgb' or 'rgba'");
                return NULL;
            }
            break;
        }
    default:
        return PyUnicode_FromString("Point (invalid)");
    }

    return NULL;
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_mappingSetElem(PyPoint* self, PyObject* key, PyObject* value)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "point is invalid");
        return -1;
    }
    bool ok;

    QString keyString = PythonQtConversion::PyObjGetString(key,true,ok);
    if (ok == false)
    {
        PyErr_SetString(PyExc_ValueError, "key must be a string");
        return -1;
    }

    if (value == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "value must not be empty");
        return -1;
    }

    float x,y,z;
    float nx,ny,nz;
    uint8_t r,g,b,alpha;
    float curvature, intensity;
    PyObject *tuple = PyTuple_New(1);
    Py_INCREF(value);
    PyTuple_SetItem(tuple, 0, value);

    switch(self->point->getType())
    {
    case ito::pclXYZ:
        {
            pcl::PointXYZ *pt = &(self->point->getPointXYZ());

            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z)) goto end;
                pt->x = x; pt->y = y; pt->z = z;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz'");
                return 0;
            }
            break;
        }
    case ito::pclXYZI:
        {
            pcl::PointXYZI *pt = &(self->point->getPointXYZI());
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z)) goto end;
                pt->x = x; pt->y = y; pt->z = z;
            }
            else if (QString::compare(keyString, "intensity", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "f", &intensity)) goto end;
                pt->intensity = intensity;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz' or 'intensity'");
                return 0;
            }
            break;
        }
    case ito::pclXYZRGBA:
        {
            pcl::PointXYZRGBA *pt = &(self->point->getPointXYZRGBA());
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z)) goto end;
                pt->x = x; pt->y = y; pt->z = z;
            }
            else if (QString::compare(keyString, "rgba", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(hhhh)", &r, &g, &b, &alpha)) goto end;
                pt->r = r; pt->b = b; pt->g = g; pt->PCLALPHA = alpha;
            }
            else if (QString::compare(keyString, "rgb", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(hhh)", &r, &g, &b)) goto end;
                pt->r = r; pt->b = b; pt->g = g;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'rgb' or 'rgba'");
                return 0;
            }
            break;
        }
    case ito::pclXYZNormal:
        {
            pcl::PointNormal *pt = &(self->point->getPointXYZNormal());
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z)) goto end;
                pt->x = x; pt->y = y; pt->z = z;
            }
            else if (QString::compare(keyString, "normal", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &nx, &ny, &nz)) goto end;
                pt->normal_x = nx; pt->normal_y = ny; pt->normal_z = nz;
            }
            else if (QString::compare(keyString, "curvature", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "f", &curvature)) goto end;
                pt->curvature = curvature;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'normal' or 'curvature'");
                return 0;
            }
            break;
        }
    case ito::pclXYZINormal:
        {
            pcl::PointXYZINormal *pt = &(self->point->getPointXYZINormal());
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z)) goto end;
                pt->x = x; pt->y = y; pt->z = z;
            }
            else if (QString::compare(keyString, "normal", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &nx, &ny, &nz)) goto end;
                pt->normal_x = nx; pt->normal_y = ny; pt->normal_z = nz;
            }
            else if (QString::compare(keyString, "curvature", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "f", &curvature)) goto end;
                pt->curvature = curvature;
            }
            else if (QString::compare(keyString, "intensity", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "f", &intensity)) goto end;
                pt->intensity = intensity;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature' or 'intensity'");
                return 0;
            }
            break;
        }
    case ito::pclXYZRGBNormal:
        {
            pcl::PointXYZRGBNormal *pt = &(self->point->getPointXYZRGBNormal());
            if (QString::compare(keyString, "xyz", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z)) goto end;
                pt->x = x; pt->y = y; pt->z = z;
            }
            else if (QString::compare(keyString, "normal", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(fff)", &nx, &ny, &nz)) goto end;
                pt->normal_x = nx; pt->normal_y = ny; pt->normal_z = nz;
            }
            else if (QString::compare(keyString, "curvature", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "f", &curvature)) goto end;
                pt->curvature = curvature;
            }
            else if (QString::compare(keyString, "rgba", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(hhhh)", &r, &g, &b, &alpha)) goto end;
                pt->r = r; pt->b = b; pt->g = g; pt->PCLALPHA = alpha;
            }
            else if (QString::compare(keyString, "rgb", Qt::CaseInsensitive) == 0)
            {
                if (!PyArg_ParseTuple(tuple, "(hhh)", &r, &g, &b)) goto end;
                pt->r = r; pt->b = b; pt->g = g;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature', 'rgb' or 'rgba'");
                return 0;
            }
            break;
        }
    default:
        PyErr_SetString(PyExc_ValueError, "Point is invalid");
        goto end;
    }
    Py_DECREF(tuple);

    return 0; //error

end:
    Py_DECREF(tuple);
    return -1; //error
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointGetType_doc,"returns type-object for this point");
PyObject* PythonPCL::PyPoint_GetType(PyPoint *self, void * /*closure*/)
{
    PyObject *type = NULL;
    PyObject *dict = NULL; //borrowed
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    if (PythonPCL::PyPointType.tp_dict != NULL)
    {
        dict = PythonPCL::PyPointType.tp_dict;
        int pType = self->point->getType();
        switch(pType)
        {
        default:
            PyErr_SetString(PyExc_ValueError, "point type is not defined");
        case ito::pclInvalid:
            type = PyDict_GetItemString(dict, "PointInvalid");
            break;
        case ito::pclXYZ:
            type = PyDict_GetItemString(dict, "PointXYZ");
            break;
        case ito::pclXYZI:
            type = PyDict_GetItemString(dict, "PointXYZI");
            break;
        case ito::pclXYZRGBA:
            type = PyDict_GetItemString(dict, "PointXYZRGBA");
            break;
        case ito::pclXYZNormal:
            type = PyDict_GetItemString(dict, "PointXYZNormal");
            break;
        case ito::pclXYZINormal:
            type = PyDict_GetItemString(dict, "PointXYZINormal");
            break;
        case ito::pclXYZRGBNormal:
            type = PyDict_GetItemString(dict, "PointXYZRGBNormal");
            break;
        }
        Py_XINCREF(type);
        return type;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "class point is not available");
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointXYZ_doc,"get or set x,y,z-values of point as tuple (x,y,z)");
PyObject* PythonPCL::PyPoint_GetXYZ(PyPoint *self, void * /*closure*/)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    //XYZ is available by every point-type
    float x,y,z;
    if (!self->point->getXYZ(x,y,z))
    {
        PyErr_SetString(PyExc_ValueError, "could not read x,y,z");
        return NULL;
    }
    return Py_BuildValue("(fff)",x,y,z);
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_SetXYZ(PyPoint *self, PyObject *value, void * /*closure*/)
{
    bool ok = false;
    float x,y,z;
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return 0;
    }

    PyObject *tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple,0,value);
    if (PyArg_ParseTuple(tuple, "(fff)", &x, &y, &z))
    {
        ok = self->point->setXYZ(x,y,z);
    }
    Py_DECREF(tuple);

    if (!ok)
    {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "x,y,z could not be assigned");
        return -1;
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointIntensity_doc,"gets or sets intensity if type of point supports intensity values \n\
\n\
Raises \n\
-------- \n\
ValueError : \n\
    if type of point does not support an intensity value.");
PyObject* PythonPCL::PyPoint_GetIntensity(PyPoint *self, void * /*closure*/)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    //XYZ is available by every point-type
    float i;
    if (!self->point->getIntensity(i))
    {
        PyErr_SetString(PyExc_ValueError, "could not read intensity");
        return NULL;
    }
    return Py_BuildValue("f",i);
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_SetIntensity(PyPoint *self, PyObject *value, void * /*closure*/)
{
    bool ok = false;
    float intensity;
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return 0;
    }

    PyObject *tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple,0,value);
    if (PyArg_ParseTuple(tuple, "f", &intensity))
    {
        ok = self->point->setIntensity(intensity);
    }
    Py_DECREF(tuple);

    if (!ok)
    {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "intensity could not be assigned");
        return -1;
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointRgb_doc,"gets or sets rgb-values as tuple (r,g,b), where each color component is in range [0, 255]\n\
\n\
Raises \n\
-------- \n\
ValueError : \n\
    if type of point does not support r,g,b values.");
PyObject* PythonPCL::PyPoint_GetRgb(PyPoint *self, void * /*closure*/)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    //XYZ is available by every point-type
    uint8_t r,g,b,a;
    if (!self->point->getRGBA(r,g,b,a))
    {
        PyErr_SetString(PyExc_ValueError, "could not read r,g,b");
        return NULL;
    }
    return Py_BuildValue("(hhh)",r,g,b);
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_SetRgb(PyPoint *self, PyObject *value, void * /*closure*/)
{
    bool ok = false;
    uint8_t r,g,b,dummy;
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return 0;
    }

    PyObject *tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple,0,value);
    if (PyArg_ParseTuple(tuple, "(hhh)", &r, &g, &b))
    {
        dummy = 255;
        ok = self->point->setRGBA(r,g,b,dummy, 0x01 | 0x02 | 0x04);
    }
    Py_DECREF(tuple);

    if (!ok)
    {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "r,g,b could not be assigned");
        return -1;
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointRgba_doc,"gets or sets rgba-values as tuple (r,g,b), where each color component is in range [0, 255]\n\
\n\
Raises \n\
-------- \n\
ValueError : \n\
    if type of point does not support r,g,b,a values.");
PyObject* PythonPCL::PyPoint_GetRgba(PyPoint *self, void * /*closure*/)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    //XYZ is available by every point-type
    uint8_t r,g,b,a;
    if (!self->point->getRGBA(r,g,b,a))
    {
        PyErr_SetString(PyExc_ValueError, "could not read r,g,b,a");
        return NULL;
    }
    return Py_BuildValue("(hhhh)",r,g,b,a);
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_SetRgba(PyPoint *self, PyObject *value, void * /*closure*/)
{
    bool ok = false;
    uint8_t r,g,b,a;
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return 0;
    }

    PyObject *tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple,0,value);
    if (PyArg_ParseTuple(tuple, "(hhhh)", &r, &g, &b, &a))
    {
        ok = self->point->setRGBA(r,g,b,a);
    }
    Py_DECREF(tuple);

    if (!ok)
    {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "r,g,b,a could not be assigned");
        return -1;
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCurvature_doc,"gets or sets curvature value\n\
\n\
Raises \n\
-------- \n\
ValueError : \n\
    if type of point does not support a curvature value.");
PyObject* PythonPCL::PyPoint_GetCurvature(PyPoint *self, void * /*closure*/)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    //XYZ is available by every point-type
    float c;
    if (!self->point->getCurvature(c))
    {
        PyErr_SetString(PyExc_ValueError, "could not read curvature");
        return NULL;
    }
    return Py_BuildValue("f",c);
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_SetCurvature(PyPoint *self, PyObject *value, void * /*closure*/)
{
    bool ok = false;
    float curvature;
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return 0;
    }

    PyObject *tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple,0,value);
    if (PyArg_ParseTuple(tuple, "f", &curvature))
    {
        ok = self->point->setCurvature(curvature);
    }
    Py_DECREF(tuple);

    if (!ok)
    {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "curvature could not be assigned");
        return -1;
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointNormal_doc,"gets or sets normal vector as tuple (nx,ny,nz)\n\
\n\
Raises \n\
-------- \n\
ValueError : \n\
    if type of point does not support normal vector data.");
PyObject* PythonPCL::PyPoint_GetNormal(PyPoint *self, void * /*closure*/)
{
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return NULL;
    }

    //XYZ is available by every point-type
    float nx,ny,nz;
    if (!self->point->getNormal(nx,ny,nz))
    {
        PyErr_SetString(PyExc_ValueError, "could not read nx,ny,nz");
        return NULL;
    }
    return Py_BuildValue("(fff)",nx,ny,nz);
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPoint_SetNormal(PyPoint *self, PyObject *value, void * /*closure*/)
{
    bool ok = false;
    float nx,ny,nz;
    if (self->point == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "point is NULL");
        return 0;
    }

    PyObject *tuple = PyTuple_New(1);
    PyTuple_SetItem(tuple,0,value);
    if (PyArg_ParseTuple(tuple, "(fff)", &nx, &ny, &nz))
    {
        ok = self->point->setNormal(nx,ny,nz);
    }
    Py_DECREF(tuple);

    if (!ok)
    {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "nx,ny,nz could not be assigned");
        return -1;
    }
    return 0;
}

//------------------------------------------------------------------------------------------------------
PyGetSetDef PythonPCL::PyPoint_getseters[] = {
    {"type", (getter)PyPoint_GetType, NULL, pyPointGetType_doc, NULL},
    {"xyz", (getter)PyPoint_GetXYZ, (setter)PyPoint_SetXYZ, pyPointXYZ_doc, NULL},
    {"intensity", (getter)PyPoint_GetIntensity, (setter)PyPoint_SetIntensity, pyPointIntensity_doc, NULL},
    {"rgb", (getter)PyPoint_GetRgb, (setter)PyPoint_SetRgb, pyPointRgb_doc, NULL},
    {"rgba", (getter)PyPoint_GetRgba, (setter)PyPoint_SetRgba, pyPointRgba_doc, NULL},
    {"curvature", (getter)PyPoint_GetCurvature, (setter)PyPoint_SetCurvature, pyPointCurvature_doc, NULL},
    {"normal", (getter)PyPoint_GetNormal, (setter)PyPoint_SetNormal, pyPointNormal_doc, NULL},
    {NULL}  /* Sentinel */
};

//------------------------------------------------------------------------------------------------------
PyMethodDef PythonPCL::PyPoint_methods[] = {
    {"name", (PyCFunction)PyPoint_name, METH_NOARGS, "name"},
    {NULL}  /* Sentinel */
};

//------------------------------------------------------------------------------------------------------
PyModuleDef PythonPCL::PyPointModule = {
    PyModuleDef_HEAD_INIT, "Point", "wrapper for PCL points", -1,
    NULL, NULL, NULL, NULL, NULL
};

//------------------------------------------------------------------------------------------------------
PyMappingMethods PythonPCL::PyPoint_mappingProtocol = {
    (lenfunc)NULL,
    (binaryfunc)PyPoint_mappingGetElem,
    (objobjargproc)PyPoint_mappingSetElem
};

//------------------------------------------------------------------------------------------------------
PyTypeObject PythonPCL::PyPointType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.point",             /* tp_name */
    sizeof(PyPoint),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PythonPCL::PyPoint_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PythonPCL::PyPoint_repr,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    &PyPoint_mappingProtocol,   /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    pointInit_doc,           /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    PythonPCL::PyPoint_methods,             /* tp_methods */
    0, /*PyNpDataObject_members,*/             /* tp_members */
    PythonPCL::PyPoint_getseters,                         /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PythonPCL::PyPoint_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PythonPCL::PyPoint_new         /* tp_new */
};

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPoint_addTpDict(PyObject *tp_dict)
{
    PyObject *value;

    value = Py_BuildValue("i",ito::pclInvalid);
    PyDict_SetItemString(tp_dict, "PointInvalid", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",ito::pclXYZ);
    PyDict_SetItemString(tp_dict, "PointXYZ", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",ito::pclXYZI);
    PyDict_SetItemString(tp_dict, "PointXYZI", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",ito::pclXYZRGBA);
    PyDict_SetItemString(tp_dict, "PointXYZRGBA", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",ito::pclXYZNormal);
    PyDict_SetItemString(tp_dict, "PointXYZNormal", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",ito::pclXYZINormal);
    PyDict_SetItemString(tp_dict, "PointXYZINormal", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",ito::pclXYZRGBNormal);
    PyDict_SetItemString(tp_dict, "PointXYZRGBNormal", value);
    Py_DECREF(value);
}

//---------------------------------------------------------------------------------------
//
// PyPoint
//
//---------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPolygonMesh_dealloc(PyPolygonMesh* self)
{
    Py_XDECREF(self->base);
    DELETE_AND_SET_NULL(self->polygonMesh);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_new(PyTypeObject *type, PyObject* /*args*/, PyObject * /*kwds*/)
{
    PyPolygonMesh* self = (PyPolygonMesh *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->base = NULL;
        self->polygonMesh = NULL;
    }

    return (PyObject *)self;
};

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(polygonMeshInit_doc,"polygonMesh(mesh = None, polygons = None) -> creates a polygon mesh.\n\
\n\
This constructor either creates an empty polygon mesh, a shallow copy of another polygon mesh (mesh parameter only) or a deep copy of \n\
another polygon mesh where only the polygons, given by the list of indices in the parameter 'polygons', are taken. \n\
In this case, the containing cloud is reduced and no longer organized (height=1, dense=false) \n\
\n\
Parameters \n\
----------- \n\
mesh : {polygonMesh}, optional \n\
    another polygon mesh instance (shallow or deep copy depending on polygons-parameter)\n\
polygons : {sequence or array-like}, optional \n\
    If given, polygons must be a sequence or one-dimensional array-like structure, where all values can be transformed into unsigned integer values. \n\
    Polygons must contain a list of indices pointing to all polygon from the given mesh that should be copied to this new instance.");
int PythonPCL::PyPolygonMesh_init(PyPolygonMesh * self, PyObject * args, PyObject * kwds)
{
    const char *kwlist[] = {"mesh", "polygons", NULL};
    PyPolygonMesh *mesh = NULL;
    PyObject *polygons = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O!O", const_cast<char**>(kwlist), &PyPolygonMeshType, &mesh, &polygons))
    {
        return -1;
    }

    if (!mesh && polygons)
    {
        PyErr_SetString(PyExc_RuntimeError,"it is not allowed to pass polygon-indices but now polygon mesh.");
        return -1;
    }

    if (mesh)
    {
        if (polygons)
        {
            if (sizeof(npy_uint) != sizeof(uint32_t))
            {
                PyErr_SetString(PyExc_RuntimeError,"polygon indexing not possible since size of NPY_UINT32 does not correspond to size of uint32_t");
                return -1;
            }

            //try to convert polygons into a numpy-array of desired type
            #if !defined(NPY_NO_DEPRECATED_API) || (NPY_NO_DEPRECATED_API < NPY_1_7_API_VERSION)
                PyObject *polygonArray = PyArray_ContiguousFromAny(polygons, NPY_UINT32, 1, 1);
            #else
                PyArrayObject *polygonArray = (PyArrayObject*)PyArray_ContiguousFromAny(polygons, NPY_UINT32, 1, 1);
            #endif


            if (polygonArray)
            {
                npy_intp len = PyArray_SIZE(polygonArray);

                npy_uint *arrayStart = (npy_uint*)PyArray_GETPTR1(polygonArray,0);
                npy_uint maxIdx = mesh->polygonMesh->polygonMesh()->polygons.size() - 1;

                //check values for validity
                for (npy_intp l = 0; l < len; ++l)
                {
                    if (arrayStart[l] > maxIdx)
                    {
                        Py_XDECREF(polygonArray);
                        PyErr_Format(PyExc_RuntimeError,"The given polygon index %i is bigger than the biggest allowed index %i.", arrayStart[l], maxIdx);
                        return -1;
                    }
                }

                std::vector<uint32_t> polygonIndices;
                polygonIndices.resize(len);

                void *vecStart = (void*)(&polygonIndices[0]);
                memcpy(vecStart, arrayStart, len * sizeof(uint32_t));

                self->polygonMesh = new ito::PCLPolygonMesh(*(mesh->polygonMesh), polygonIndices);
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError,"polygon indices must be a one-dimensional sequence or array of unsigned integer indices.");
                return -1;
            }

            Py_XDECREF(polygonArray);
        }
        else
        {
            self->polygonMesh = new ito::PCLPolygonMesh(*(mesh->polygonMesh));
        }
    }
    else
    {
        self->polygonMesh = new ito::PCLPolygonMesh();
    }

    return 0;
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_name(PyPolygonMesh* /*self*/)
{
    return PyUnicode_FromString("PolygonMesh");
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_repr(PyPolygonMesh *self)
{
    if (self->polygonMesh == NULL || self->polygonMesh->polygonMesh().get() == NULL)
    {
        return PyUnicode_FromString("PolygonMesh (empty)");
    }
    else
    {
        return PyUnicode_FromFormat("PolygonMesh (%u polygons, [%u x %u] points, fields: %s)", self->polygonMesh->polygonMesh()->polygons.size(), self->polygonMesh->height(), self->polygonMesh->width(), self->polygonMesh->getFieldsList().data());
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_data(PyPolygonMesh *self)
{
    if (self->polygonMesh)
    {
        self->polygonMesh->streamOut(std::cout);
        Py_RETURN_NONE;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError,"point cloud is NULL");
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_Reduce(PyPolygonMesh *self, PyObject * /*args*/)
{
    if (self->polygonMesh == NULL)
    {
        PyErr_SetString(PyExc_NotImplementedError, "point cloud is NULL");
        return NULL;
    }

    QByteArray content = "";
    PyObject *stateTuple = NULL;

    if (self->polygonMesh->valid())
    {
        QTemporaryFile pclTempSaveFile;
        bool success = pclTempSaveFile.open();
        if (!success)
        {
            PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing polygon mesh binary data could not be created");
            return NULL;
        }
        //give access to the next one who wants to write to it...
        pclTempSaveFile.close();

        QVector<ito::ParamBase> paramsMand;
        QVector<ito::ParamBase> paramsOpt;
        QVector<ito::ParamBase> paramsOut;

        paramsMand.append(ito::ParamBase("polygonMesh",
                    ito::ParamBase::PolygonMeshPtr | ito::ParamBase::In,
                    reinterpret_cast<const char*>(self->polygonMesh)));
        paramsMand.append(ito::ParamBase("filename",
                     ito::ParamBase::String | ito::ParamBase::In,
                     pclTempSaveFile.fileName().toLatin1().data()));

        paramsOpt.append(ito::ParamBase("type", ito::ParamBase::String, "obj"));

        ito::RetVal retval;
        if (apiFilterCall)
            retval = apiFilterCall("savePolygonMesh", &paramsMand, &paramsOpt, &paramsOut);
        else
            retval = ito::RetVal(ito::retError, 0, QObject::tr("api function pointer not set").toLatin1().data());

        if (PythonCommon::transformRetValToPyException(retval) == false)
        {
            return NULL;
        }

        if (pclTempSaveFile.open() == false)
        {
            PyErr_SetString(PyExc_RuntimeError, "Temporary file for reading back polygon mesh binary data could not be opened");
            return NULL;
        }

        //allocate memory
        stateTuple = PyBytes_FromStringAndSize(NULL, pclTempSaveFile.size());
        //get pointer to datapart of memory
        char *data = PyBytes_AsString(stateTuple);
        //write to memory
        uint64_t bytesRead = pclTempSaveFile.peek(data, pclTempSaveFile.size());
        Q_ASSERT(bytesRead == pclTempSaveFile.size());//check if file was fully read...
    }
    else
    {
        Py_INCREF(Py_None);
        stateTuple = Py_None;
    }

    //the stateTuple is simply a byte array with the binary content of the temporarily written obj file or None, if the point cloud is invalid or empty
    //the type-number is passed as argument to the constructor of the polygon mesh class, if it is reconstructed.
    PyObject *tempOut = Py_BuildValue("(O()O)", Py_TYPE(self), stateTuple);
    Py_XDECREF(stateTuple);

    return tempOut;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_SetState(PyPolygonMesh *self, PyObject *args)
{
    PyObject *data = NULL;
    if (!PyArg_ParseTuple(args, "O", &data))
    {
        return NULL;
    }

    if (data == Py_None)
    {
        Py_RETURN_NONE;
    }
    else if (PyBytes_Check(data))
    {
        QTemporaryFile pclTempSaveFile;
        bool success = pclTempSaveFile.open();//creates the file
        if (success == false)
        {
            PyErr_SetString(PyExc_RuntimeError, QObject::tr("Temporary file for writing point cloud binary data could not be created").toLatin1().data());
            return NULL;
        }

        pclTempSaveFile.write(PyBytes_AsString(data), PyBytes_GET_SIZE(data));
        pclTempSaveFile.close();

        QVector<ito::ParamBase> paramsMand;
        QVector<ito::ParamBase> paramsOpt;
        QVector<ito::ParamBase> paramsOut;

        paramsMand.append(ito::ParamBase("polygonMesh",
                                         ito::ParamBase::PointCloudPtr | ito::ParamBase::In,
                                         reinterpret_cast<const char*>(self->polygonMesh)));
        paramsMand.append(ito::ParamBase("filename",
                                         ito::ParamBase::String | ito::ParamBase::In,
                                         pclTempSaveFile.fileName().toLatin1().data()));
        paramsOpt.append(ito::ParamBase("type", ito::ParamBase::String | ito::ParamBase::In, "obj"));

        ito::RetVal retval;
        if (apiFilterCall)
            retval = apiFilterCall("loadPolygonMesh", &paramsMand, &paramsOpt, &paramsOut);
        else
            retval = ito::RetVal(ito::retError, 0, QObject::tr("api function pointer not set").toLatin1().data());
        //Zumindest im DebugModus kann man ja mal den retval checken,
        //sonst ist das hier ja komplett gratis
        Q_ASSERT(retval.containsWarningOrError() == false);

        if (PythonCommon::transformRetValToPyException(retval) == false)
        {
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "The pickled data must be a byte array for establishing the polygonMesh.");
        return NULL;
    }

    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------------------------------
/*static*/ PythonPCL::PyPolygonMesh* PythonPCL::createEmptyPyPolygonMesh()
{
    PyObject *args = PyTuple_New(0);

    PyPolygonMesh* result = (PyPolygonMesh*)PyObject_Call((PyObject*)&PyPolygonMeshType, args, NULL);

    Py_XDECREF(args);

    if (result != NULL)
    {
        DELETE_AND_SET_NULL(result->polygonMesh);
        return result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonPCL::PyPolygonMesh_mappingGetElem(PyPolygonMesh* self, PyObject* key)
{
    ito::PCLPolygonMesh *pm = self->polygonMesh;

    if (pm == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Polygon mesh is NULL");
        return NULL;
    }

    size_t dims = pm->height() > 1 ? 2 : 1;

    if (!PyTuple_Check(key))
    {
        key = PyTuple_Pack(1,key);
    }

    PyErr_SetString(PyExc_TypeError, "mapping get not implemented yet.");
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

//------------------------------------------------------------------------------------------------------
/*static*/ int PythonPCL::PyPolygonMesh_mappingLength(PyPolygonMesh* self)
{
    if (self->polygonMesh)
    {
        return self->polygonMesh->height() * self->polygonMesh->width();
    }
    else
    {
        return 0;
    }
}

//------------------------------------------------------------------------------------------------------
//PyDoc_STRVAR(pyPolygonMeshGetCloud_doc,"cloud -> ");
///*static*/ PyObject* PythonPCL::PyPolygonMesh_getCloud(PyPolygonMesh *self, void *closure)
//{
//    return NULL;
//}

//------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonPCL::PyPolygonMesh_get(PyPolygonMesh *self, PyObject *args, PyObject *kwds)
{
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPolygonMeshGetCloud_docs,"getCloud(pointType = point.PointInvalid) -> returns the point cloud of this polygon mesh converted to the desired type.\n\
\n\
If the pointType is not given or point.PointInvalid, the type of the internal pointCloud is guessed with respect to available types. \n\
\n\
Parameters \n\
----------- \n\
pointType : {int, enum point.PointXXX}, optional \n\
    the point type value of the desired type, the point cloud should be converted too (default: point.PointInvalid)");
/*static*/ PyObject* PythonPCL::PyPolygonMesh_getCloud(PyPolygonMesh *self, PyObject *args)
{
    int pointType = ito::pclInvalid;

    if (!PyArg_ParseTuple(args, "|i", &pointType))
    {
        return NULL;
    }

    ito::PCLPolygonMesh *pm = self->polygonMesh;

    if (pm == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Polygon mesh is NULL");
        return NULL;
    }

    if (pointType == pclInvalid) //try to guess the type
    {
        ito::tPCLPointType t = ito::pclHelper::guessPointType(pm->polygonMesh()->cloud);
        if (t == pclInvalid)
        {
            PyErr_SetString(PyExc_RuntimeError, "The native pointType of the given polygon mesh cannot be guessed.");
            return NULL;
        }
        else
        {
            pointType = t;
        }
    }

    if (pointType == ito::pclXYZ || pointType == pclXYZ || pointType == pclXYZI
        || pointType == pclXYZRGBA || pointType == pclXYZNormal || pointType == pclXYZINormal || pointType == pclXYZRGBNormal)
    {
        PyPointCloud* pc = createEmptyPyPointCloud();
        pc->data = new ito::PCLPointCloud((ito::tPCLPointType)pointType);
        ito::RetVal retval = ito::pclHelper::pointCloud2ToPCLPointCloud(pm->polygonMesh()->cloud, pc->data);

        if (!PythonCommon::transformRetValToPyException(retval))
        {
            Py_DECREF(pc);
            return NULL;
        }

        return (PyObject*)pc;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "The given pointType is unknown");
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPolygonMeshGetPolygons_docs,"getPolygons() -> returns MxN int32 dataObject with the polygon description. M is the number of polygons and N is the biggest number of vertices.");
/*static*/ PyObject* PythonPCL::PyPolygonMesh_getPolygons(PyPolygonMesh *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
    {
        return NULL;
    }

    ito::PCLPolygonMesh *pm = self->polygonMesh;

    if (pm == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Polygon mesh is NULL");
        return NULL;
    }

    std::vector<pcl::Vertices> *p = &(pm->polygonMesh()->polygons);

    PythonDataObject::PyDataObject* dataObj = PythonDataObject::createEmptyPyDataObject();

    size_t numVertices = 0;
    pcl::Vertices *verticePtr = &(p->front());
    size_t psize = p->size();

    for (size_t i = 0 ; i < psize; i++)
    {
        numVertices = std::max(numVertices, verticePtr->vertices.size());
        verticePtr++;
    }

    dataObj->dataObject = new ito::DataObject((int)p->size(), (int)numVertices, ito::tInt32);
    ito::int32 *ptr = (ito::int32*)dataObj->dataObject->rowPtr(0,0);

    verticePtr = &(p->front());

    for (size_t i = 0 ; i < psize; i++)
    {
        memcpy(ptr, &(verticePtr->vertices.front()), verticePtr->vertices.size() * sizeof(ito::int32));

        for (size_t j = verticePtr->vertices.size(); j < numVertices; j++)
        {
            ptr[j] = -1;
        }

        ptr += numVertices;
        verticePtr++;
    }

    return (PyObject*)dataObj;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPolygonMeshFromCloudAndPolygons_docs,"fromCloudAndPolygons(cloud, polygons) -> creates a polygon mesh from cloud and polygons. \n\
\n\
Parameters \n\
----------- \n\
cloud : {pointCloud} \n\
    the input point cloud \n\
polygons : {array-like, MxN} \n\
    an array-like matrix with the indices of the polygons. The array contains M polygons and every row gives the indices of the vertices of the cloud belonging to the polygon.");
/*static*/ PyObject* PythonPCL::PyPolygonMesh_FromCloudAndPolygons(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyPointCloud *cloud = NULL;
    PyObject *polygons = NULL;
    const char *kwlist[] = {"cloud", "polygons", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O", const_cast<char**>(kwlist), &PythonPCL::PyPointCloudType, &cloud, &polygons))
    {
        return NULL;
    }

    if (sizeof(npy_int) != sizeof(ito::int32))
    {
        PyErr_SetString(PyExc_RuntimeError,"polygon indexing not possible since size of NPY_UINT32 does not correspond to size of uint32_t");
        return NULL;
    }

    //try to convert polygons into a numpy-array of desired type
    #if !defined(NPY_NO_DEPRECATED_API) || (NPY_NO_DEPRECATED_API < NPY_1_7_API_VERSION)
        PyObject *polygonArray = PyArray_ContiguousFromAny(polygons, NPY_INT32, 2, 2);
    #else
        PyArrayObject *polygonArray = (PyArrayObject*)PyArray_ContiguousFromAny(polygons, NPY_INT32, 2, 2);
    #endif

    if (polygonArray)
    {
        npy_intp len = PyArray_SIZE(polygonArray);
        npy_intp y = PyArray_DIM(polygonArray, 0);
        npy_intp x = PyArray_DIM(polygonArray, 1);

        std::vector<pcl::Vertices> p;
        pcl::Vertices v;
        p.resize(y);
        v.vertices.resize(x);
        ito::int32 *linePtr;

        for (npy_intp m = 0; m < y; m++)
        {
            linePtr = (ito::int32*)(PyArray_GETPTR1(polygonArray, m));
            memcpy(&v.vertices.front(), linePtr, sizeof(ito::int32) * x);
            p[m] = v;
        }

        Py_DECREF(polygonArray);

        PyPolygonMesh *mesh = PythonPCL::createEmptyPyPolygonMesh();
        mesh->polygonMesh = new ito::PCLPolygonMesh(*(cloud->data), p);
        return (PyObject*)mesh;
    }
    else
    {
        return NULL;
    }
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPolygonMeshFromTopography_docs, "fromTopography(topography, triangulationType = 1) -> creates a polygon mesh from a dataObject whose values are the z-components. \n\
\n\
The polygons are created either as rectangles (quads) or triangles. Some other algorithms only support meshes with triangles. \n\
This method is the same than calling polygonMesh.fromOrganizedCloud(pointCloud.fromTopography(topography)). \n\
\n\
Parameters \n\
----------- \n\
topography : {dataObject} \n\
    the input data object. The grid of the data object including its axisScales and axisOffsets value indicate the X and Y values whereas the Z values are given by the data object \n\
triangulationType : {int} \n\
    type of triangulation. 0: quads, 1: triangles [default]");
/*static*/ PyObject* PythonPCL::PyPolygonMesh_FromTopography(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PythonDataObject::PyDataObject *topography = NULL;
    unsigned char triangulationType = 1;
    const char *kwlist[] = { "topography", "triangulationType", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|b", const_cast<char**>(kwlist), &PythonDataObject::PyDataObjectType, &topography, &triangulationType))
    {
        return NULL;
    }

    PyObject *args2 = Py_BuildValue("(O)", topography);
    PyObject *kwds2 = PyDict_New();
    PyObject* cloud = PyPointCloud_fromTopography(NULL, args2, kwds2);
    PyObject* mesh = NULL;
    Py_DECREF(args2);

    if (cloud)
    {
        args2 = Py_BuildValue("(Oi)", cloud, triangulationType);
        mesh = PyPolygonMesh_FromOrganizedCloud(NULL, args2, kwds2);
        Py_DECREF(args2);
    }

    Py_DECREF(kwds2);

    return mesh;
}


//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPolygonMeshFromOrganizedCloud_docs,"fromOrganizedCloud(cloud, triangulationType = 1) -> creates a polygon mesh from an organized cloud using triangles. \n\
\n\
The polygons are created as triangles. Triangles are also created for non-finite points. \n\
\n\
Parameters \n\
----------- \n\
cloud : {pointCloud} \n\
    the input point cloud (must be organized, see attribute organized of a cloud) \n\
triangulationType : {int} \n\
    type of triangulation. 0: quads, 1 : triangles [default]");
/*static*/ PyObject* PythonPCL::PyPolygonMesh_FromOrganizedCloud(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
    PyPointCloud *cloud = NULL;
    unsigned char triangulationType = 1;
    const char *kwlist[] = { "cloud", "triangulationType", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|b", const_cast<char**>(kwlist), &PythonPCL::PyPointCloudType, &cloud, &triangulationType))
    {
        return NULL;
    }

    if (cloud->data->isOrganized() == false)
    {
        PyErr_SetString(PyExc_RuntimeError,"given cloud must be organized (e.g. creating from topography map without NaN or Inf values)");
        return NULL;
    }

    std::vector<pcl::Vertices> p;
    uint32_t h = cloud->data->height();
    uint32_t w = cloud->data->width();

    if (triangulationType > 0)
    {
        //triangles
        p.resize((h - 1) * (w - 1) * 2);
        pcl::Vertices v;
        v.vertices.resize(3);
        uint32_t i = 0;

        for (uint32_t r = 0; r < (w - 1); ++r)
        {
            for (uint32_t c = 0; c < (h - 1); ++c)
            {
                //points: p1 - p2
                //         |    |
                //        p3 - p4
                //triangles: p1,p3,p2 and p3,p4,p2
                v.vertices[0] = r * w + c;
                v.vertices[1] = (r + 1)*w + c;
                v.vertices[2] = v.vertices[0] + 1;
                p[i++] = v;
                v.vertices[0] += w;
                v.vertices[1] += 1;
                //v.vertices[2] is the same than above
                p[i++] = v;
            }
        }
    }
    else
    {
        //quads
        p.resize((h - 1) * (w - 1));
        pcl::Vertices v;
        v.vertices.resize(4);
        uint32_t i = 0;

        for (uint32_t r = 0; r < (w - 1); ++r)
        {
            for (uint32_t c = 0; c < (h - 1); ++c)
            {
                //points: p1 - p2
                //         |    |
                //        p3 - p4
                //triangles: p1,p3,p2 and p3,p4,p2
                v.vertices[0] = r * w + c;
                v.vertices[1] = r * w + c + 1;
                v.vertices[2] = (r + 1) * w + c + 1;
                v.vertices[3] = (r + 1) * w + c;
                p[i++] = v;
            }
        }
    }


    PyPolygonMesh *mesh = PythonPCL::createEmptyPyPolygonMesh();
    mesh->polygonMesh = new ito::PCLPolygonMesh(*(cloud->data), p);
    return (PyObject*)mesh;
}

//------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonPCL::PyPolygonMesh_getNrOfPolygons(PyPolygonMesh *self, void * /*closure*/)
{
    ito::PCLPolygonMesh *pm = self->polygonMesh;

    if (pm == NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Polygon mesh is NULL");
        return NULL;
    }

    if (pm->polygonMesh().get() == NULL)
    {
        return Py_BuildValue("i", 0);
    }

    return Py_BuildValue("i", pm->polygonMesh()->polygons.size());
}

//------------------------------------------------------------------------------------------------------
PyGetSetDef PythonPCL::PyPolygonMesh_getseters[] = {
    //{"cloud", (getter)PyPolygonMesh_getCloud, NULL, pyPolygonMeshGetCloud_doc, NULL},
    {"nrOfPolygons", (getter)PyPolygonMesh_getNrOfPolygons, NULL,
    "returns the number of polygons in this mesh", NULL},
    {NULL}  /* Sentinel */
};

//------------------------------------------------------------------------------------------------------
PyMethodDef PythonPCL::PyPolygonMesh_methods[] = {
    {"name", (PyCFunction)PyPolygonMesh_name, METH_NOARGS, "name"},
    {"__reduce__", (PyCFunction)PyPolygonMesh_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
    {"__setstate__", (PyCFunction)PyPolygonMesh_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
    {"data", (PyCFunction)PyPolygonMesh_data, METH_NOARGS, "prints content of polygon mesh"},
    //{"get", (PyCFunction)PyPolygonMesh_get, METH_VARARGS | METH_KEYWORDS, pyPolygonMeshGet_docs},
    {"getCloud", (PyCFunction)PyPolygonMesh_getCloud, METH_VARARGS, pyPolygonMeshGetCloud_docs},
    {"getPolygons", (PyCFunction)PyPolygonMesh_getPolygons, METH_VARARGS, pyPolygonMeshGetPolygons_docs},
    {"fromCloudAndPolygons", (PyCFunction)PyPolygonMesh_FromCloudAndPolygons, METH_VARARGS | METH_KEYWORDS | METH_STATIC, pyPolygonMeshFromCloudAndPolygons_docs},
    {"fromOrganizedCloud", (PyCFunction)PyPolygonMesh_FromOrganizedCloud, METH_VARARGS | METH_KEYWORDS | METH_STATIC, pyPolygonMeshFromOrganizedCloud_docs},
    {"fromTopography", (PyCFunction)PyPolygonMesh_FromTopography, METH_VARARGS | METH_KEYWORDS | METH_STATIC, pyPolygonMeshFromTopography_docs },
    {NULL}  /* Sentinel */
};

//------------------------------------------------------------------------------------------------------
PyModuleDef PythonPCL::PyPolygonMeshModule = {
    PyModuleDef_HEAD_INIT, "PolygonMesh", "wrapper for PCL polygon mesh", -1,
    NULL, NULL, NULL, NULL, NULL
};

//------------------------------------------------------------------------------------------------------
PyTypeObject PythonPCL::PyPolygonMeshType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.polygonMesh",             /* tp_name */
    sizeof(PyPolygonMesh),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PythonPCL::PyPolygonMesh_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PythonPCL::PyPolygonMesh_repr,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    &PyPolygonMesh_mappingProtocol, /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    polygonMeshInit_doc,           /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    PythonPCL::PyPolygonMesh_methods,             /* tp_methods */
    0, /*PyNpDataObject_members,*/             /* tp_members */
    PythonPCL::PyPolygonMesh_getseters,                         /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PythonPCL::PyPolygonMesh_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PythonPCL::PyPolygonMesh_new         /* tp_new */
};

//------------------------------------------------------------------------------------------------------
PyMappingMethods PythonPCL::PyPolygonMesh_mappingProtocol = {
    (lenfunc)PyPolygonMesh_mappingLength,
    (binaryfunc)PyPolygonMesh_mappingGetElem,
    NULL
};

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPolygonMesh_addTpDict(PyObject * /*tp_dict*/)
{
}

//------------------------------------------------------------------------------------------------------
} //end namespace ito

#endif //#if ITOM_POINTCLOUDLIBRARY > 0
