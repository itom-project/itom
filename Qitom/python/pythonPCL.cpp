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

#include "pythonPCL.h"

#include "../global.h"

#if ITOM_POINTCLOUDLIBRARY > 0

#ifndef CMAKE
    #ifdef _DEBUG
	    #pragma comment(lib, "PointCloudd.lib")
	    #pragma comment(lib, "pcl_common_debug.lib")
    #else
	    #pragma comment(lib, "PointCloud.lib")
	    #pragma comment(lib, "pcl_common_release.lib")
    #endif
#else
    /*#ifdef _DEBUG
	    #pragma comment(lib, "PointCloud.lib")
	    #pragma comment(lib, "pcl_common_debug.lib")
    #else
	    #pragma comment(lib, "PointCloud.lib")
	    #pragma comment(lib, "pcl_common_release.lib")
    #endif*/
#endif

#include "pythonQtConversion.h"
#include "pythonDataObject.h"

#include "../PointCloud/pclFunctions.h"
#include "../DataObject/dataObjectFuncs.h"
#include "pythonCommon.h"

#include "../api/apiFunctions.h"

//#include <pcl/point_types.h>
//#include <pcl/point_cloud.h>

#include <qbytearray.h>
#include <qstring.h>
//#include <qtemporaryfile.h>

#include <vector>

//for generating a temporary file name (for pickling point clouds)
#include <stdio.h> 
#include <qdir.h>

//------------------------------------------------------------------------------------------------------

namespace ito
{

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPointCloud_addTpDict(PyObject * /*tp_dict*/)
{
}

//------------------------------------------------------------------------------------------------------
void PythonPCL::PyPointCloud_dealloc(PyPointCloud* self)
{
    if (self->data)
    {
        delete self->data;
        self->data = NULL;
    }
    Py_XDECREF(self->base);

    Py_TYPE(self)->tp_free((PyObject*)self);
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyPointCloud* self = (PyPointCloud *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->base = NULL;
        self->data = NULL;
    }

    return (PyObject *)self;
};

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pointCloudInit_doc,"pointCloud([type] | pointCloud [,indices] | width, height [,point] | point) -> creates new point cloud.  \n\
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
    if( args == NULL )
    {
        self->data = new ito::PCLPointCloud();
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
            PyErr_Format(PyExc_TypeError, "The point cloud type is unknown");
            return -1;
            break;
        }
    }

    //2. check for copy constructor
    if (!done && PyArg_ParseTuple(args, "O!|O", &PythonPCL::PyPointCloudType, &copyConstr, &pySeq))
    {
        PyPointCloud *copyConstr2 = (PyPointCloud*)copyConstr; 
        if (copyConstr2->data != NULL)
        {
            if (pySeq == NULL)
            {
                self->data = new ito::PCLPointCloud(*copyConstr2->data);
            }
            else
            {
                if (PyIter_Check(pySeq))
                {
                    PyObject *iterator = PyObject_GetIter(pySeq);
                    PyObject *item = NULL;
                    std::vector< int > indices;
                    if (PySequence_Check(pySeq) && PySequence_Length(pySeq)>0)
                    {
                        indices.reserve( PySequence_Length(pySeq) );
                    }

                    if (iterator == NULL) 
                    {
                        PyErr_Format(PyExc_TypeError, "error creating iterator");
                    }
                    else
                    {
                        //TODO: gcc wants paraentheses around assignment in while condition
                        while (item = PyIter_Next(iterator)) 
                        {
                            if (PyLong_Check(item))
                            {
                                indices.push_back( PyLong_AsLong(item) );
                                Py_DECREF(item);
                            }
                            else
                            {
                                PyErr_Format(PyExc_TypeError, "indices must only contain integer values");
                                Py_DECREF(item);
                                break;
                            }
                        }

                        Py_DECREF(iterator);

                        if (!PyErr_Occurred()) 
                        {
                            self->data = new ito::PCLPointCloud(*copyConstr2->data, indices);
                        }
                        
                    }
                }
                else
                {
                    PyErr_Format(PyExc_TypeError, "indices must be an iteratible object");
                }
            }
        }
        else
        {
            self->data = new ito::PCLPointCloud(ito::pclInvalid);
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
        self->data = new ito::PCLPointCloud( (uint32_t)width, (uint32_t)height, point.getType(), point );
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
        self->data = new ito::PCLPointCloud( 1, 1, point.getType(), point );
    }

    if (done == false)
    {
        PyErr_SetString(PyExc_RuntimeError, "arguments for constructor must be a type value, another instance of point cloud, an instance of point or width, height and an instance of point");
        return -1;
    }


    return 0;
};

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudType_doc,"returns point type of point cloud\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetType(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    ito::tPCLPointType type;
    try
    {
        type = self->data->getType();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    if (PythonPCL::PyPointType.tp_dict != NULL)
    {
        PyObject *dict = PythonPCL::PyPointType.tp_dict;
        PyObject *value = NULL;
        switch(type)
        {
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
PyDoc_STRVAR(pyPointCloudSize_doc,"returns number of points in point cloud\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetSize(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    size_t size;
    try
    {
        size = self->data->size();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    return Py_BuildValue("i", size);
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudHeight_doc,"returns height of point cloud if organized as regular grid (organized == true), else 1 \n\
specifies the height of the point cloud dataset in the number of points. HEIGHT has two meanings: \n\
    * it can specify the height (total number of rows) of an organized point cloud dataset; \n\
    * it is set to 1 for unorganized datasets (thus used to check whether a dataset is organized or not).\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetHeight(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    uint32_t height;
    try
    {
        height = self->data->height();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    return Py_BuildValue("I", height);
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudWidth_doc,"returns width of point cloud if organized as regular grid (organized == true), else equal to size \n\
specifies the width of the point cloud dataset in the number of points. WIDTH has two meanings: \n\
    * it can specify the total number of points in the cloud (equal with POINTS see below) for unorganized datasets; \n\
    * it can specify the width (total number of points in a row) of an organized point cloud dataset.\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetWidth(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    uint32_t width;
    try
    {
        width = self->data->width();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    return Py_BuildValue("I", width);
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudEmpty_doc,"returns whether this point cloud is empty, hence size == 1\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetEmpty(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    bool empty;
    try
    {
        empty = self->data->empty();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    if (empty) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudOrganized_doc,"returns whether this point cloud is organized as regular grid, hence height != 1\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetOrganized(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    bool organized;
    try
    {
        organized = self->data->isOrganized();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    if (organized) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudDense_doc,"specifies if all the data in points is finite (true), or whether it might contain Inf/NaN values (false).\n\
\n\
Notes \n\
----- \n\
{bool} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetDense(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    bool dense;
    try
    {
        dense = self->data->is_dense();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    if (dense) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

int PythonPCL::PyPointCloud_SetDense(PyPointCloud *self, PyObject *value, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return -1;
    }

    bool dense;
    if(PyArg_Parse(value,"b", &dense))
    {
        try
        {
            self->data->set_dense(dense);
        }
        catch(pcl::PCLException exc)
        {
            PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
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
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return -1;
    }

    return 0;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFields_doc,"get available field names of point cloud\n\
\n\
Notes \n\
----- \n\
{} : ReadOnly ReadWrite \n\
");
PyObject* PythonPCL::PyPointCloud_GetFields(PyPointCloud *self, void * /*closure*/)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    std::string names;
    try
    {
        names = self->data->getFieldsList();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
        return NULL;
    }

    QStringList a = QString::fromStdString(names).split(" ");
    PyObject *result = PyList_New(a.size());
    QByteArray ba;

    for(int i = 0 ; i < a.size() ; i++)
    {
        ba = a[i].toAscii();
        PyList_SetItem(result,i, PyUnicode_FromStringAndSize( ba.data(), ba.size() ) );
    }
    
    return result;
}

//------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudAppend_doc,"append(point) -> appends point at the end of the point cloud. \n\
\n\
Parameters \n\
----------- \n\
point : {point???}, optional \n\
\n\
Notes \n\
----- \n\
The type of point must fit to the type of the point cloud. If the point cloud is \n\
invalid, its type is set to the type of the point.\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonPCL::PyPointCloud_append(PyPointCloud *self, PyObject *args, PyObject *kwds)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

        //check if args contains only one point cloud
    static char *kwlist0[] = {"pointCloud", NULL};
    static char *kwlist1[] = {"point", NULL};
    PyObject *pclObj = NULL;
    if (PyArg_ParseTupleAndKeywords(args,kwds,"O!",kwlist0, &PyPointCloudType, &pclObj))
    {
        PyPointCloud *pcl = (PyPointCloud*)pclObj;
        if (pcl == NULL || pcl->data == NULL)
        {
            PyErr_Format(PyExc_RuntimeError, "argument is of type pointCloud, but this point cloud or the underlying point cloud structure is NULL");
            return NULL;
        }

        *(self->data) += *(pcl->data);
        
    }
    else if (PyErr_Clear(), PyArg_ParseTupleAndKeywords(args,kwds,"O!",kwlist1, &PyPointType, &pclObj) )
    {
        PyPoint *point = (PyPoint*)pclObj;
        if (point == NULL || point->point == NULL)
        {
            PyErr_Format(PyExc_RuntimeError, "argument is of type point, but this point or the underlying point structure is NULL");
            return NULL;
        }
        if (self->data->getType() != point->point->getType() && self->data->getType() != ito::pclInvalid)
        {
            PyErr_Format(PyExc_RuntimeError, "point cloud and this point do not have the same type");
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
            static char *kwlist[] = {"xyz", NULL};
            PyObject *xyz = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",kwlist,&xyz)) return NULL;
            return PyPointCloud_XYZ_append(self, xyz);
            break;
        }
        case ito::pclXYZI:
        {
            static char *kwlist[] = {"xyzi", NULL};
            PyObject *xyzi = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",kwlist,&xyzi)) return NULL;
            return PyPointCloud_XYZI_append(self, xyzi);
            break;
        }
        case ito::pclXYZRGBA:
        {
            static char *kwlist[] = {"xyz", "rgba", NULL};
            PyObject *xyz = NULL;
            PyObject *rgba = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",kwlist,&xyz, &rgba)) return NULL;
            return PyPointCloud_XYZRGBA_append(self, xyz, rgba);
            break;
        }
        case ito::pclXYZNormal:
        {
            static char *kwlist[] = {"xyz_normal_curvature", NULL};
            PyObject *xyz_normal_curvature = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",kwlist,&xyz_normal_curvature)) return NULL;
            return PyPointCloud_XYZNormal_append(self, xyz_normal_curvature);
            break;
        }
        case ito::pclXYZINormal:
        {
            static char *kwlist[] = {"xyz_i_normal_curvature", NULL};
            PyObject *xyz_i_normal_curvature = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",kwlist,&xyz_i_normal_curvature)) return NULL;
            return PyPointCloud_XYZINormal_append(self, xyz_i_normal_curvature);
            break;
        }
        case ito::pclXYZRGBNormal:
        {
            static char *kwlist[] = {"xyz_normal_curvature", "rgba", NULL};
            PyObject *xyz_normal_curvature = NULL;
            PyObject *rgba = NULL;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",kwlist,&xyz_normal_curvature, &rgba)) return NULL;
            return PyPointCloud_XYZRGBNormal_append(self, xyz_normal_curvature, rgba);
            break;
        }
        default:
            PyErr_Format(PyExc_RuntimeError, "point cloud must have a valid point type");
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
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZ");
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
    pclPtr->reserve( pclPtr->size() + n );
    for(npy_intp i = 0 ; i < n; i++)
    {
        pclPtr->push_back( pcl::PointXYZ(xyz[0][i],xyz[1][i],xyz[2][i]) );
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
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZI");
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
    pclPtr->reserve( pclPtr->size() + n );
    pcl::PointXYZI p;
    for(npy_intp i = 0 ; i < n; i++)
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
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZRGBA");
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
        PyErr_Format(PyExc_RuntimeError, "length of xyz and rgba-arrays must be equal");
        return NULL;
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pclPtr = self->data->toPointXYZRGBA();
    pclPtr->reserve( pclPtr->size() + n1 );
    pcl::PointXYZRGBA p;
    for(npy_intp i = 0 ; i < n1; i++)
    {
        p.x = xyz[0][i]; p.y = xyz[1][i]; p.z = xyz[2][i];
        p.r = rgba[0][i]; p.g = rgba[1][i]; p.b = rgba[2][i]; p.PCLALPHA = rgba[3][i];
        pclPtr->push_back( p );
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
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZNormal");
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
    pclPtr->reserve( pclPtr->size() + n );
    pcl::PointNormal p;
    for(npy_intp i = 0 ; i < n; i++)
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
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZINormal");
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
    pclPtr->reserve( pclPtr->size() + n );
    pcl::PointXYZINormal p;
    for(npy_intp i = 0 ; i < n; i++)
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
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL or not of type pointXYZRGBNormal");
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
        PyErr_Format(PyExc_RuntimeError, "length of xyz and rgba-arrays must be equal");
        return NULL;
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pclPtr = self->data->toPointXYZRGBNormal();
    pclPtr->reserve( pclPtr->size() + n1 );
    pcl::PointXYZRGBNormal p;
    for(npy_intp i = 0 ; i < n1; i++)
    {
        p.x = xyznxnynzc[0][i]; p.y = xyznxnynzc[1][i]; p.z = xyznxnynzc[2][i];
        p.normal_x = xyznxnynzc[3][i]; p.normal_y = xyznxnynzc[4][i]; p.normal_z = xyznxnynzc[5][i];
        p.curvature = xyznxnynzc[6][i];
        p.r = rgba[0][i]; p.g = rgba[1][i]; p.b = rgba[2][i]; p.PCLALPHA = rgba[3][i];
        pclPtr->push_back( p );
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
        size = self->data->size();
        width = self->data->width();
        height = self->data->height();
        }
        catch(pcl::PCLException exc) {};

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
PyDoc_STRVAR(pyPointCloudClear_doc,"clear() -> clears the whole point cloud\n\
\n\
Notes \n\
----- \n\
Clears the whole pointcloud.\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonPCL::PyPointCloud_clear(PyPointCloud *self)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is NULL");
        return NULL;
    }

    try
    {
        self->data->clear();
    }
    catch(pcl::PCLException exc)
    {
        PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
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
        catch(pcl::PCLException exc)
        {
            s = 0;
        }
    }
    return s;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqConcat(PyPointCloud *self, PyObject *rhs) //returns new reference
{
    if ( Py_TYPE(rhs) != &PyPointCloudType )
    {
        PyErr_Format(PyExc_TypeError, "object must be of type pointCloud");
        return NULL;
    }

    PyPointCloud *rhs_ = (PyPointCloud*)rhs;
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
            catch(pcl::PCLException exc)
            {
                PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
                return NULL;
            }
            return (PyObject*)result;
        }
        else
        {
            PyErr_Format(PyExc_RuntimeError, "could not allocate object of type pointCloud");
            return NULL;
        }
    }
    PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqRepeat(PyPointCloud *self, Py_ssize_t size)//returns new reference
{
    if (self->data)
    {
        PyObject *args = Py_BuildValue("(i)", self->data->getType());
        PyPointCloud *result = (PyPointCloud*)PyObject_Call((PyObject*)&PyPointCloudType, args, NULL); //new reference
        Py_DECREF(args);
        if (result)
        {
            try
            {
                for(Py_ssize_t i = 0; i < size; i++)
                {
                    *result->data += *self->data;
                }
            }
            catch(pcl::PCLException exc)
            {
                PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
                return NULL;
            }
            
            return (PyObject*)result;
        }
        else
        {
            PyErr_Format(PyExc_RuntimeError, "could not allocate object of type pointCloud");
            return NULL;
        }
    }
    PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqItem(PyPointCloud *self, Py_ssize_t size) //returns new reference
{
    if (self->data)
    {
        if (size < 0 || size >= static_cast<Py_ssize_t>(self->data->size()))
        {
            PyErr_Format(PyExc_RuntimeError, "index must be in range [%d,%d]", 0, self->data->size()-1);
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
            catch(pcl::PCLException exc)
            {
                PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
                return NULL;
            }
            
        }
        else
        {
            PyErr_Format(PyExc_RuntimeError, "could not allocate object of type point");
            return NULL;
        }
    }
    PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
int PythonPCL::PyPointCloud_seqAssItem(PyPointCloud *self, Py_ssize_t size, PyObject *point)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
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
        if (Py_TYPE(point) != &PyPointType )
        {
            PyErr_Format(PyExc_TypeError, "assigned value must be of type point");
            return -1;
        }
        PyPoint *point_ = (PyPoint*)point;

        try
        {
            self->data->set_item(size, *point_->point);
        }
        catch (pcl::PCLException exc)
        {
            PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
            return -1;
        }
    }

    return 0;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqInplaceConcat(PyPointCloud *self, PyObject *rhs) //returns new reference
{
    if ( Py_TYPE(rhs) != &PyPointCloudType )
    {
        PyErr_Format(PyExc_TypeError, "object must be of type pointCloud");
        return NULL;
    }

    PyPointCloud *rhs_ = (PyPointCloud*)rhs;
    if (self->data && rhs_->data)
    {
        try
        {
            *self->data = *self->data + *rhs_->data;
        }
        catch(pcl::PCLException exc)
        {
            PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
            return NULL;
        }
        
        Py_INCREF(self);
        return (PyObject*)self;
    }
    PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
    return NULL;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_seqInplaceRepeat(PyPointCloud * /*self*/, Py_ssize_t /*size*/)//returns new reference
{
    PyErr_Format(PyExc_NotImplementedError, "not implemented yet");
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
        PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
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

    if (slicelength > 1) //two or more points -> return tuple of points
    {
        PyObject *retValue = PyTuple_New(slicelength);
        PyPoint *tempPt = NULL;
        Py_ssize_t c = 0;
        for(Py_ssize_t i = start ; i < stop ; i += step)
        {
            tempPt = (PyPoint*)PyObject_Call((PyObject*)&PyPointType, NULL, NULL); //new reference
            if (tempPt)
            {
                tempPt->point = new ito::PCLPoint(self->data->at(i));
            }
            else
            {
                Py_XDECREF(retValue);
                PyErr_Format(PyExc_RuntimeError, "could not allocate object of type point");
                return NULL;
            }

            PyTuple_SetItem(retValue, c, (PyObject*)tempPt); //steals reference
            c++;
        }

        return retValue;
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
            PyErr_Format(PyExc_RuntimeError, "could not allocate object of type point");
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
        PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
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
            for(Py_ssize_t i = stop-1 ; i>=start ; i-=step) //erase backwards
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
                for(Py_ssize_t i = start ; i < stop ; i+=step)
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
Parameters \n\
----------- \n\
index : {}\n\
values : {}\n\
\n\
Notes \n\
----- \n\
doctodo\n\
\n\
See Also \n\
--------- \n\
\n\
");

PyObject* PythonPCL::PyPointCloud_insert(PyPointCloud *self, PyObject *args)
{
    if (self->data == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "this point cloud is empty");
        return NULL;
    }

    PyObject *index = NULL;
    PyObject *points = NULL;
    Py_ssize_t start = 0;
    if (!PyArg_ParseTuple(args,"OO", &index, &points))
    {
        PyErr_Format(PyExc_RuntimeError, "argument must be an fixed-point index number followed by a single point or a sequence of points");
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

    if ( Py_TYPE(points) == &PyPointType )
    {
        try
        {
            self->data->insert(start, *(((PyPoint*)points)->point));
        }
        catch(pcl::PCLException exc)
        {
            PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
            return NULL;
        }
    }
    else if (PySequence_Check(points))
    {
        PyObject *sequence = PySequence_Fast(points, "");

        for(Py_ssize_t i = 0 ; i < PySequence_Length(points) ; i++)
        {
            if (Py_TYPE( PySequence_Fast_GET_ITEM(sequence,i) ) != &PyPointType)
            {
                PyErr_Format(PyExc_TypeError, "not every element in sequence is of type point");
                Py_DECREF(sequence);
                return NULL;
            }
        }

        for(Py_ssize_t i = 0 ; i < PySequence_Length(points) ; i++)
        {
            try
            {
                self->data->insert(start + i, *(((PyPoint*)PySequence_Fast_GET_ITEM(sequence,i))->point) );
            }
            catch(pcl::PCLException exc)
            {
                Py_DECREF(sequence);
                PyErr_SetString(PyExc_TypeError, exc.detailedMessage().c_str());
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
indices : {}\n\
\n\
Notes \n\
----- \n\
This method is the same than command 'del pointCloudVariable[indices]'\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonPCL::PyPointCloud_erase(PyPointCloud *self, PyObject *args)
{
    PyObject *indices = NULL;
    if (!PyArg_ParseTuple(args,"O", &indices))
    {
        PyErr_Format(PyExc_RuntimeError, "argument must be a number of a slice object");
        return NULL;
    }

    if (!PyPointCloud_mappingSetElem(self, indices, NULL)) return NULL;

    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudToDataObject_doc,"toDataObject() -> returns a PxN data object, where P is determined by the point type in the point cloud. N is the number of points.\n\
\n\
Returns \n\
------- \n\
PxN : {float}\n\
\n\
Notes \n\
----- \n\
doctodo\n\
\n\
See Also \n\
--------- \n\
\n\
");
/*static*/ PyObject* PythonPCL::PyPointCloud_toDataObject(PyPointCloud *self)
{
    if (self->data)
    {
        ito::PythonDataObject::PyDataObject *pyDObj = ito::PythonDataObject::createEmptyPyDataObject(); //new reference
        pyDObj->dataObject = new ito::DataObject();
        ito::RetVal retval = ito::pclHelper::pointCloudToDObj(self->data, *(pyDObj->dataObject));

        if( ito::PythonCommon::transformRetValToPyException( retval ) == false)
        {
            Py_DECREF(pyDObj);
            pyDObj = NULL;
            return NULL;
        }
        return (PyObject*)pyDObj;
    }
    else
    {
        PyErr_Format(PyExc_RuntimeError, "point cloud is empty");
        return NULL;
    }  
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_Reduce(PyPointCloud *self, PyObject * /*args*/)
{
    if(self->data == NULL)
    {
        PyErr_SetString(PyExc_NotImplementedError, "point cloud is NULL");
        return NULL;
    }

    int type = self->data->getType();
	QByteArray content = "";
	PyObject *stateTuple = NULL;

	if(type != ito::pclInvalid && self->data->size() > 0)
	{

		char *buf = tmpnam(NULL);
		if(buf == NULL)
		{
			PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing point cloud binary data could not be created");
			return NULL;
		}

		QString tempFilename = buf;
		while(tempFilename.size() > 0 && (tempFilename.startsWith("/") || tempFilename.startsWith("\\") ))
		{
			tempFilename.remove(0,1);
		}
		
		tempFilename = QDir::temp().absoluteFilePath( tempFilename ); //creates unique, temporary filename
		QFile tempFile2;

		QVector<ito::ParamBase> paramsMand;
		QVector<ito::ParamBase> paramsOpt;
		QVector<ito::ParamBase> paramsOut;

		paramsMand.append( ito::ParamBase("pointCloud", ito::ParamBase::PointCloudPtr | ito::ParamBase::In, (const char*)self->data) );
		paramsMand.append( ito::ParamBase("filename", ito::ParamBase::String | ito::ParamBase::In, tempFilename.toAscii().data() ) );

		paramsOpt.append( ito::ParamBase("mode", ito::ParamBase::String, "b") );
		paramsOpt.append( ito::ParamBase("type", ito::ParamBase::String, "pcd") );

		ito::RetVal retval = ito::apiFunctions::mfilterCall( "savePointCloud", &paramsMand, &paramsOpt, &paramsOut);

		if( PythonCommon::transformRetValToPyException(retval) == false )
		{
            return NULL;
		}
		
		tempFile2.setFileName(tempFilename);
		if(tempFile2.open(QIODevice::ReadOnly) == false)
		{
			PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing point cloud binary data could not be opened");
			return NULL;
		}

		stateTuple = PyBytes_FromStringAndSize(NULL, tempFile2.size() + 10 );
		char *data = PyBytes_AsString( stateTuple );
		tempFile2.peek( data, tempFile2.size() + 10 );
		tempFile2.close();
		tempFile2.remove();
	}
	else
	{
		Py_INCREF(Py_None);
		stateTuple = Py_None;
	}
	
	//the stateTuple is simply a byte array with the binary content of the temporarily written pcd file or None, if the point cloud is invalid or empty
	//the type-number is passed as argument to the constructor of the point cloud class, if it is reconstructed.
    PyObject *tempOut = Py_BuildValue("(O(i)O)", Py_TYPE(self), type, stateTuple);
    Py_XDECREF(stateTuple);

    return tempOut;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPointCloud_SetState(PyPointCloud *self, PyObject *args)
{
	PyObject *data = NULL;
	if(!PyArg_ParseTuple(args, "O", &data))
	{
		return NULL;
	}

	if(data == Py_None)
	{
		Py_RETURN_NONE;
	}
	else if(PyBytes_Check(data))
	{
		char *buf = tmpnam(NULL);
		if(buf == NULL)
		{
			PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing point cloud binary data could not be created");
			return NULL;
		}

		QString tempFilename = buf;
		while(tempFilename.size() > 0 && (tempFilename.startsWith("/") || tempFilename.startsWith("\\") ))
		{
			tempFilename.remove(0,1);
		}
		
		tempFilename = QDir::temp().absoluteFilePath( tempFilename ); //creates unique, temporary filename

		QFile tempFile2(tempFilename);
		if(tempFile2.open( QIODevice::WriteOnly ) == false)
		{
			PyErr_SetString(PyExc_RuntimeError, "temporary file could not be opened (II)");
			return NULL;
		}
		tempFile2.write(PyBytes_AsString(data), PyBytes_GET_SIZE(data));
		tempFile2.close();

		QVector<ito::ParamBase> paramsMand;
		QVector<ito::ParamBase> paramsOpt;
		QVector<ito::ParamBase> paramsOut;

		paramsMand.append( ito::ParamBase("pointCloud", ito::ParamBase::PointCloudPtr | ito::ParamBase::In, (const char*)self->data) );
		paramsMand.append( ito::ParamBase("filename", ito::ParamBase::String | ito::ParamBase::In, tempFilename.toAscii().data() ) );
		paramsOpt.append( ito::ParamBase("type", ito::ParamBase::String | ito::ParamBase::In, "pcd" ) );

		ito::RetVal retval = ito::apiFunctions::mfilterCall( "loadPointCloud", &paramsMand, &paramsOpt, &paramsOut);

		tempFile2.remove();

		if( PythonCommon::transformRetValToPyException(retval) == false )
		{
			return NULL;
		}
	}
	else
	{
		PyErr_SetString(PyExc_RuntimeError, "The pickled data must be a byte array for establishing the pointCloud.");
		return NULL;
	}


    Py_RETURN_NONE;
}

//---------------------------------------------------------------------------------------
PyDoc_STRVAR(pyPointCloudFromXYZ_doc,"fromXYZ(X,Y,Z | XYZ) -> creates a point cloud from three X,Y,Z data objects or from one 3xMxN data object\n\
\n\
Parameters \n\
----------- \n\
X,Y,Z : {MxN data objects} \n\
    Three 2D data objects with the same size.\n\
XYZ : {3xMxN data object} \n\
    OR: 3xMxN data object, such that the first plane is X, the second is Y and the third is Z\n\
deleteNaN : {bool} \n\
    default = false\n\
    if True all NaN values are skipped, hence, the resulting point cloud is not dense (organized) any more\n\
\n\
Returns \n\
------- \n\
PointCloud.");
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
        if( !PyArg_ParseTuple(args, "O|b", &objX, &deleteNaN) )
        {
            return NULL;
        }
        else
        {
            XYZ = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objX, false, ok) );
            if(!ok)
            {
                PyErr_Format(PyExc_RuntimeError, "XYZ argument could not be converted to a data object");
                return NULL;
            }
            
            retval += ito::dObjHelper::verify3DDataObject(XYZ.data(), "XYZ", 3, 3, 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
            
            if(PythonCommon::transformRetValToPyException( retval ) == false)
            {
                return NULL;
            }

            ito::Range ranges[3] = { ito::Range(0,0), ito::Range::all(), ito::Range::all() };

            ranges[0] = ito::Range(0,1);
            X = QSharedPointer<ito::DataObject>( new ito::DataObject( XYZ->at(ranges) )  );

            ranges[0] = ito::Range(1,2);
            Y = QSharedPointer<ito::DataObject>( new ito::DataObject( XYZ->at(ranges) )  );

            ranges[0] = ito::Range(2,3);
            Z = QSharedPointer<ito::DataObject>( new ito::DataObject( XYZ->at(ranges) )  );

        }
    }
    else
    {
        X = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objX, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "X argument could not be converted to a data object");
            return NULL;
        }

        Y = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objY, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "Y argument could not be converted to a data object");
            return NULL;
        }

        Z = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objZ, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "Z argument could not be converted to a data object");
            return NULL;
        }
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    cloud->data = new ito::PCLPointCloud(ito::pclInvalid);
    retval += ito::pclHelper::pointCloudFromXYZ(X.data(), Y.data(), Z.data(), *(cloud->data), deleteNaN);

    if(PythonCommon::transformRetValToPyException( retval ) == false)
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
PyDoc_STRVAR(pyPointCloudFromXYZI_doc,"fromXYZI(X,Y,Z,I | XYZ,I) -> creates a point cloud from four X,Y,Z,I data objects or from one 3xMxN data object and one intensity data object\n\
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
    if True all NaN values are skipped, hence, the resulting point cloud is not dense (organized) any more\n\
\n\
Returns \n\
------- \n\
PointCloud.");
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
        if( !PyArg_ParseTuple(args, "OO|b", &objX, &objI, &deleteNaN) )
        {
            return NULL;
        }
        else
        {
            XYZ = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objX, false, ok) );
            if(!ok)
            {
                PyErr_Format(PyExc_RuntimeError, "XYZ argument could not be converted to a data object");
                return NULL;
            }
            
            retval += ito::dObjHelper::verify3DDataObject(XYZ.data(), "XYZ", 3, 3, 1, std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 1, ito::tFloat32);
            
            if(PythonCommon::transformRetValToPyException( retval ) == false)
            {
                return NULL;
            }

            ito::Range ranges[3] = { ito::Range(0,0), ito::Range::all(), ito::Range::all() };

            ranges[0] = ito::Range(0,1);
            X = QSharedPointer<ito::DataObject>( new ito::DataObject( XYZ->at(ranges).squeeze() )  );

            ranges[0] = ito::Range(1,2);
            Y = QSharedPointer<ito::DataObject>( new ito::DataObject( XYZ->at(ranges).squeeze() )  );

            ranges[0] = ito::Range(2,3);
            Z = QSharedPointer<ito::DataObject>( new ito::DataObject( XYZ->at(ranges).squeeze() )  );

            I = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objI, false, ok) );
            if(!ok)
            {
                PyErr_Format(PyExc_RuntimeError, "Intensity argument could not be converted to a data object");
                return NULL;
            }

        }
    }
    else
    {
        X = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objX, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "X argument could not be converted to a data object");
            return NULL;
        }

        Y = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objY, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "Y argument could not be converted to a data object");
            return NULL;
        }

        Z = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objZ, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "Z argument could not be converted to a data object");
            return NULL;
        }

        I = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objI, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "Intensity argument could not be converted to a data object");
            return NULL;
        }
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    cloud->data = new ito::PCLPointCloud(ito::pclInvalid);
    retval += ito::pclHelper::pointCloudFromXYZI(X.data(), Y.data(), Z.data(), I.data(), *(cloud->data), deleteNaN);

    if(PythonCommon::transformRetValToPyException( retval ) == false)
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
PyDoc_STRVAR(pyPointCloudFromDisparity_doc,"fromDisparity(disparity [,intensity] [,deleteNaN]) -> creates a point cloud from a given disparity dataObject.\n\
\n\
Parameters \n\
----------- \n\
disparity : {MxN data object, float32} \n\
    The values of this dataObject represent the disparity values.\n\
intensity : {MxN data object, float32}, optional \n\
    If given, an XYZI-point cloud is created whose intensity values are determined by this dataObject \n\
    deleteNaN : {bool}, optional \n\
    If true (default: false), all NaN-values in the disparity map will not be copied into the point cloud.\n\
\n\
Returns \n\
------- \n\
PointCloud.");
/*static*/ PyObject* PythonPCL::PyPointCloud_fromDisparity(PyPointCloud * /*self*/, PyObject *args, PyObject *kwds)
{
    PyObject *objDisp = NULL;
    PyObject *objI = NULL;
    bool deleteNaN = false;
    const char *kwlist[] = {"disparity", "intensity", "deleteNaN", NULL};

    QSharedPointer<ito::DataObject> dispMap, IntMap;
    bool ok = true;
    ito::RetVal retval = ito::retOk;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Ob", const_cast<char**>(kwlist), &objDisp, &objI, &deleteNaN))
    {
        return NULL; 
    }
    
    dispMap = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objDisp, false, ok) );
    if(!ok)
    {
        PyErr_Format(PyExc_RuntimeError, "disparity map argument could not be converted to a data object");
        return NULL;
    }

    if(objI)
    {
        IntMap = QSharedPointer<ito::DataObject>( PythonQtConversion::PyObjGetDataObjectNewPtr( objI, false, ok) );
        if(!ok)
        {
            PyErr_Format(PyExc_RuntimeError, "intensity map argument could not be converted to a data object");
            return NULL;
        }
    }
    else
    {
        IntMap = QSharedPointer<ito::DataObject>(NULL);
    }

    PyPointCloud *cloud = createEmptyPyPointCloud();
    cloud->data = new ito::PCLPointCloud(ito::pclInvalid);
    retval += ito::pclHelper::pointCloudFromDisparityI(dispMap.data(), IntMap.data(), *(cloud->data), deleteNaN);

    if(PythonCommon::transformRetValToPyException( retval ) == false)
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
    if(result != NULL)
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
    {"type",        (getter)PyPoint_GetType,            NULL,                           pyPointCloudType_doc,     NULL},
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
    {"append",        (PyCFunction)PyPointCloud_append, METH_KEYWORDS | METH_VARARGS, pyPointCloudAppend_doc},
    {"clear",         (PyCFunction)PyPointCloud_clear, METH_NOARGS, pyPointCloudClear_doc},
    {"insert",        (PyCFunction)PyPointCloud_insert, METH_VARARGS, pyPointCloudInsert_doc},
    {"erase",         (PyCFunction)PyPointCloud_erase, METH_VARARGS, pyPointCloudErase_doc},
    {"toDataObject",  (PyCFunction)PyPointCloud_toDataObject, METH_NOARGS, pyPointCloudToDataObject_doc},
    {"__reduce__",    (PyCFunction)PyPointCloud_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
    {"__setstate__",  (PyCFunction)PyPointCloud_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
    
    {"fromXYZ",       (PyCFunction)PyPointCloud_fromXYZ, METH_VARARGS | METH_STATIC, pyPointCloudFromXYZ_doc},
    {"fromXYZI",      (PyCFunction)PyPointCloud_fromXYZI, METH_VARARGS | METH_STATIC, pyPointCloudFromXYZI_doc},
    {"fromDisparity", (PyCFunction)PyPointCloud_fromDisparity, METH_VARARGS | METH_STATIC, pyPointCloudFromDisparity_doc},
    
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
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
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
        PyErr_Format(PyExc_RuntimeError, "the number of required rows must be between 1 and 7");
        return NULL;
    }
#if (NPY_FEATURE_VERSION < 0x00000007)
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#else
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#endif
    
    if (arr == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "argument cannot be interpreted as float32, c-contiguous numpy.ndarray");
        return NULL;
    }

    //check arr
    if (PyArray_NDIM( (PyArrayObject*)arr) != 2)
    {
        Py_XDECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "input array must have two dimensions");
        return NULL;
    }
    if (PyArray_DIM( (PyArrayObject*)arr,0) != mRequired)
    {
        Py_XDECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "ndArray must have %c rows", mRequired);
        return NULL;
    }

    n = PyArray_DIM( (PyArrayObject*)arr,1);
    npy_intp strideDim0 = PyArray_STRIDE( (PyArrayObject*)arr,0);

    for(npy_intp i = 0; i<mRequired;i++)
    {
        elemRows[i] = reinterpret_cast<float32*>(PyArray_BYTES( (PyArrayObject*)arr) + i*strideDim0);
    }
    
    return arr;
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::parseObjAsUInt8Array(PyObject *obj, npy_intp mRequired, npy_intp &n, uint8_t **elemRows)
{
    if (mRequired < 1 || mRequired > 4)
    {
        PyErr_Format(PyExc_RuntimeError, "the number of required rows must be between 1 and 4");
        return NULL;
    }

#if (NPY_FEATURE_VERSION < 0x00000007)
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_UBYTE, NPY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#else
    PyObject *arr = PyArray_FROM_OTF(obj, NPY_UBYTE, NPY_ARRAY_IN_ARRAY); //maybe NPY_IN_ARRAY must be changed to NPY_ARRAY_IN_ARRAY
#endif
    
    if (arr == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "argument cannot be interpreted as uint8, c-contiguous numpy.ndarray");
        return NULL;
    }

    //check arr
    if (PyArray_NDIM( (PyArrayObject*)arr) != 2)
    {
        Py_XDECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "input array must have two dimensions");
        return NULL;
    }
    if (PyArray_DIM( (PyArrayObject*)arr,0) != mRequired)
    {
        Py_XDECREF(arr);
        PyErr_Format(PyExc_RuntimeError, "ndArray must have %c rows", mRequired);
        return NULL;
    }

    n = PyArray_DIM( (PyArrayObject*)arr,1);
    npy_intp strideDim0 = PyArray_STRIDE( (PyArrayObject*)arr,0);

    for(npy_intp i = 0; i<mRequired;i++)
    {
        elemRows[i] = reinterpret_cast<uint8_t*>(PyArray_BYTES( (PyArrayObject*)arr) + i*strideDim0);
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
    if (self->point)
    {
        delete self->point;
        self->point = NULL;
    }
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
PyDoc_STRVAR(pointInit_doc,"point([type, [xyz, [intensity, ][rgba, ][normal, ][curvature]]) -> creates new point used for class 'pointCloud'.  \n\
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

    if (PyTuple_Size(args)>0 || kwds != NULL)
    {
        PyObject *temp = kwds == NULL ? (PyObject*)NULL : PyDict_GetItemString(kwds, "type");
        if (temp == NULL && PyTuple_Size(args)>0) temp = PyTuple_GetItem(args,0); //borrowed
        if (temp)
        {
            if (!PyLong_Check(temp))
            {
                PyErr_Format(PyExc_TypeError, "The argument must contain the type of the point, e.g. point.PointXYZ");
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
            static char *kwlist[] = {"type","xyz", NULL};
            float x,y,z;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)",kwlist,&pclType,&x,&y,&z)) return -1;
            self->point = new ito::PCLPoint( pcl::PointXYZ(x,y,z) );
            break;
        }
        case ito::pclXYZI:
        {
            static char *kwlist[] = {"type","xyz","intensity", NULL};
            float x,y,z,intensity;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)f",kwlist,&pclType,&x,&y,&z,&intensity)) return -1;
            pcl::PointXYZI pt(intensity);
            pt.x=x;pt.y=y;pt.z=z;
            self->point = new ito::PCLPoint( pt );
            break;
        }
        case ito::pclXYZRGBA:
        {
            static char *kwlist[] = {"type","xyz","rgba", NULL};
            float x,y,z;
            uint8_t r,g,b,a;
            a=255;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhhh)",kwlist,&pclType,&x,&y,&z,&r,&g,&b,&a))
            {
                PyErr_Clear();
                if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhh)",kwlist,&pclType,&x,&y,&z,&r,&g,&b)) return -1;
                a=255;
            }
            pcl::PointXYZRGBA pt;
            pt.x=x;pt.y=y;pt.z=z;
            pt.r=r;pt.g=g;pt.b=b;pt.PCLALPHA=a;
            self->point = new ito::PCLPoint( pt );
            break;
        }
        case ito::pclXYZNormal:
        {
            static char *kwlist[] = {"type","xyz","normal","curvature", NULL};
            float x,y,z;
            float nx,ny,nz, curvature;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(fff)f",kwlist,&pclType,&x,&y,&z,&nx,&ny,&nz,&curvature)) return -1;
            pcl::PointNormal pt;
            pt.x = x; pt.y = y; pt.z = z;
            pt.normal_x = nx; pt.normal_y = ny; pt.normal_z = nz;
            pt.curvature = curvature;
            self->point = new ito::PCLPoint( pt );
            break;
        }
        case ito::pclXYZINormal:
        {
            static char *kwlist[] = {"type","xyz","intensity","normal","curvature", NULL};
            float x,y,z, intensity;
            float nx,ny,nz, curvature;
            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)f(fff)f",kwlist,&pclType,&x,&y,&z,&intensity,&nx,&ny,&nz,&curvature)) return -1;
            pcl::PointXYZINormal pt;
            pt.x = x; pt.y = y; pt.z = z;
            pt.normal_x = nx; pt.normal_y = ny; pt.normal_z = nz;
            pt.curvature = curvature;
            pt.intensity = intensity;
            self->point = new ito::PCLPoint( pt );
            break;
        }
        case ito::pclXYZRGBNormal:
        {
            static char *kwlist[] = {"type","xyz","rgba","normal","curvature", NULL};
            float x,y,z;
            float nx,ny,nz, curvature;
            uint8_t r,g,b,a;
            a=255;

            if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhhh)(fff)f",kwlist,&pclType,&x,&y,&z,&r,&g,&b,&a,&nx,&ny,&nz,&curvature))
            {
                PyErr_Clear();
                if (!PyArg_ParseTupleAndKeywords(args,kwds,"i(fff)(hhh)(fff)f",kwlist,&pclType,&x,&y,&z,&r,&g,&b,&nx,&ny,&nz,&curvature)) return -1;
                a=255;
            }
            
            pcl::PointXYZRGBNormal pt;
            pt.x = x; pt.y = y; pt.z = z;
            pt.normal_x = nx; pt.normal_y = ny; pt.normal_z = nz;
            pt.curvature = curvature;
            pt.r = r; pt.g = g; pt.b = b; pt.PCLALPHA = a;
            self->point = new ito::PCLPoint( pt );
            break;
        }
        default:
            PyErr_Format(PyExc_TypeError, "The point type is unknown");
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
                str = QString("Point (XYZ=[%1,%2,%3], Normal=[%4,%5,%6], curvature=%7, intensity=%8)").arg(pt.x, 0, 'G', 3).arg(pt.y, 0, 'G', 3).arg(pt.z, 0, 'G', 3).arg(pt.normal_x, 0, 'G', 3).arg(pt.normal_y, 0, 'G', 3).arg(pt.normal_z, 0, 'G', 3).arg(pt.curvature, 0, 'G', 3).arg(pt.intensity);
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

        QByteArray ba = str.toAscii();
        return PyUnicode_FromString(ba.data());
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPoint_mappingGetElem(PyPoint* self, PyObject* key)
{
    if (self->point == NULL)
    {
        PyErr_Format(PyExc_RuntimeError, "point is invalid");
        return NULL;
    }
    bool ok;
    int type = self->point->getType();

    QString keyString = PythonQtConversion::PyObjGetString(key,true,ok);
    if (ok == false)
    {
        PyErr_Format(PyExc_ValueError, "key must be a string");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz' or 'intensity'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'rgb' or 'rgba'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'normal' or 'curvature'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature' or 'intensity'");
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
                return Py_BuildValue("(hhhh)", pt.r, pt.g, pt.b, pt.PCLALPHA );
            }
            else if (QString::compare(keyString, "rgb", Qt::CaseInsensitive) == 0)
            {
                return Py_BuildValue("(hhh)", pt.r, pt.g, pt.b, pt.PCLALPHA );
            }
            else
            {
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature', 'rgb' or 'rgba'");
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
        PyErr_Format(PyExc_RuntimeError, "point is invalid");
        return -1;
    }
    bool ok;

    QString keyString = PythonQtConversion::PyObjGetString(key,true,ok);
    if (ok == false)
    {
        PyErr_Format(PyExc_ValueError, "key must be a string");
        return -1;
    }

    if (value == NULL)
    {
        PyErr_Format(PyExc_ValueError, "value must not be empty");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz' or 'intensity'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'rgb' or 'rgba'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'normal' or 'curvature'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature' or 'intensity'");
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
                PyErr_Format(PyExc_ValueError, "key must be 'xyz', 'normal', 'curvature', 'rgb' or 'rgba'");
                return 0;
            }
            break;
        }
    default:
        PyErr_Format(PyExc_ValueError, "Point is invalid");
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
        switch(self->point->getType())
        {
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
PyDoc_STRVAR(pyPointXYZ_doc,"gets or sets x,y,z-values (x,y,z)\n\
\n\
Notes \n\
----- \n\
{float-list} : ReadWrite \n\
");
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
PyDoc_STRVAR(pyPointIntensity_doc,"gets or sets intensity\n\
\n\
Notes \n\
----- \n\
{float} : ReadWrite \n\
");
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
PyDoc_STRVAR(pyPointRgb_doc,"gets or sets rgb-values (r,g,b)\n\
\n\
Notes \n\
----- \n\
{uint8-list} : ReadWrite \n\
");
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
PyDoc_STRVAR(pyPointRgba_doc,"gets or sets rgba-values (r,g,b,a)\n\
\n\
Notes \n\
----- \n\
{uint8-list} : ReadWrite \n\
");
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
Notes \n\
----- \n\
{float} : ReadWrite \n\
");
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
PyDoc_STRVAR(pyPointNormal_doc,"gets or sets normal vector (nx,ny,nz)\n\
\n\
Notes \n\
----- \n\
{float-list} : ReadWrite \n\
");
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
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
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
    if (self->polygonMesh)
    {
        delete self->polygonMesh;
        self->polygonMesh = NULL;
    }
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
PyDoc_STRVAR(polygonMeshInit_doc,"polygonMesh() -> creates empty polygon mesh.");    
int PythonPCL::PyPolygonMesh_init(PyPolygonMesh * self, PyObject * /*args*/, PyObject * /*kwds*/)
{
    self->polygonMesh = new ito::PCLPolygonMesh();
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
    if (self->polygonMesh == NULL)
    {
        return PyUnicode_FromString("PolygonMesh (empty)");
    }
    else
    {
        return PyUnicode_FromFormat("PolygonMesh (size: [%u x %u], fields: %s)", self->polygonMesh->height(), self->polygonMesh->width(), self->polygonMesh->getFieldsList().data() );
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_data(PyPolygonMesh *self)
{
    if( self->polygonMesh )
    {
        self->polygonMesh->streamOut(std::cout);
        Py_RETURN_NONE;
    }
    else
    {
        return PyErr_Format(PyExc_RuntimeError,"point cloud is NULL");
    }
}

//------------------------------------------------------------------------------------------------------
PyObject* PythonPCL::PyPolygonMesh_Reduce(PyPolygonMesh *self, PyObject * /*args*/)
{
    if(self->polygonMesh == NULL)
    {
        PyErr_SetString(PyExc_NotImplementedError, "point cloud is NULL");
        return NULL;
    }

	QByteArray content = "";
	PyObject *stateTuple = NULL;

	if( self->polygonMesh->valid() )
	{

		char *buf = tmpnam(NULL);
		if(buf == NULL)
		{
			PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing polygon mesh binary data could not be created");
			return NULL;
		}

		QString tempFilename = buf;
		while(tempFilename.size() > 0 && (tempFilename.startsWith("/") || tempFilename.startsWith("\\") ))
		{
			tempFilename.remove(0,1);
		}
		
		tempFilename = QDir::temp().absoluteFilePath( tempFilename ); //creates unique, temporary filename
		QFile tempFile2;

		QVector<ito::ParamBase> paramsMand;
		QVector<ito::ParamBase> paramsOpt;
		QVector<ito::ParamBase> paramsOut;

		paramsMand.append( ito::ParamBase("polygonMesh", ito::ParamBase::PolygonMeshPtr | ito::ParamBase::In, (const char*)self->polygonMesh) );
		paramsMand.append( ito::ParamBase("filename", ito::ParamBase::String | ito::ParamBase::In, tempFilename.toAscii().data() ) );

		paramsOpt.append( ito::ParamBase("type", ito::ParamBase::String, "obj") );

		ito::RetVal retval = ito::apiFunctions::mfilterCall( "savePolygonMesh", &paramsMand, &paramsOpt, &paramsOut);

		if( PythonCommon::transformRetValToPyException(retval) == false )
		{
            return NULL;
		}
		
		tempFile2.setFileName(tempFilename);
		if(tempFile2.open(QIODevice::ReadOnly) == false)
		{
			PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing polygon mesh binary data could not be opened");
			return NULL;
		}

		stateTuple = PyBytes_FromStringAndSize(NULL, tempFile2.size() + 10 );
		char *data = PyBytes_AsString( stateTuple );
		tempFile2.peek( data, tempFile2.size() + 10 );
		tempFile2.close();
		tempFile2.remove();
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
	if(!PyArg_ParseTuple(args, "O", &data))
	{
		return NULL;
	}

	if(data == Py_None)
	{
		Py_RETURN_NONE;
	}
	else if(PyBytes_Check(data))
	{
		char *buf = tmpnam(NULL);
		if(buf == NULL)
		{
			PyErr_SetString(PyExc_RuntimeError, "Temporary file for writing polygon mesh binary data could not be created");
			return NULL;
		}

		QString tempFilename = buf;
		while(tempFilename.size() > 0 && (tempFilename.startsWith("/") || tempFilename.startsWith("\\") ))
		{
			tempFilename.remove(0,1);
		}
		
		tempFilename = QDir::temp().absoluteFilePath( tempFilename ); //creates unique, temporary filename

		QFile tempFile2(tempFilename);
		if(tempFile2.open( QIODevice::WriteOnly ) == false)
		{
			PyErr_SetString(PyExc_RuntimeError, "temporary file could not be opened (II)");
			return NULL;
		}
		tempFile2.write(PyBytes_AsString(data), PyBytes_GET_SIZE(data));
		tempFile2.close();

		QVector<ito::ParamBase> paramsMand;
		QVector<ito::ParamBase> paramsOpt;
		QVector<ito::ParamBase> paramsOut;

		paramsMand.append( ito::ParamBase("polygonMesh", ito::ParamBase::PointCloudPtr | ito::ParamBase::In, (const char*)self->polygonMesh) );
		paramsMand.append( ito::ParamBase("filename", ito::ParamBase::String | ito::ParamBase::In, tempFilename.toAscii().data() ) );
		paramsOpt.append( ito::ParamBase("type", ito::ParamBase::String | ito::ParamBase::In, "obj" ) );

		ito::RetVal retval = ito::apiFunctions::mfilterCall( "loadPolygonMesh", &paramsMand, &paramsOpt, &paramsOut);

		tempFile2.remove();

		if( PythonCommon::transformRetValToPyException(retval) == false )
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
    PyPolygonMesh* result = (PyPolygonMesh*)PyObject_Call((PyObject*)&PyPolygonMeshType, NULL, NULL);
    if(result != NULL)
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
PyGetSetDef PythonPCL::PyPolygonMesh_getseters[] = {
    {NULL}  /* Sentinel */
};

//------------------------------------------------------------------------------------------------------
PyMethodDef PythonPCL::PyPolygonMesh_methods[] = {
    {"name", (PyCFunction)PyPolygonMesh_name, METH_NOARGS, "name"},
	{"__reduce__", (PyCFunction)PyPolygonMesh_Reduce, METH_VARARGS, "__reduce__ method for handle pickling commands"},
    {"__setstate__", (PyCFunction)PyPolygonMesh_SetState, METH_VARARGS, "__setstate__ method for handle unpickling commands"},
    {"data", (PyCFunction)PyPolygonMesh_data, METH_NOARGS, "prints content of polygon mesh"},
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
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    polygonMeshInit_doc,           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
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
void PythonPCL::PyPolygonMesh_addTpDict(PyObject * /*tp_dict*/)
{
}



} //end namespace ito

#endif //#if ITOM_POINTCLOUDLIBRARY > 0
