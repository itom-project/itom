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

#ifndef PYTHONPCL_H
#define PYTHONPCL_H


/* includes */
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "pythonWrapper.h"
#endif

#include "../global.h"

#if ITOM_POINTCLOUDLIBRARY > 0

#include "PointCloud/pclStructures.h"


namespace ito
{

class PythonPCL
{
public:
    typedef struct
    {
        PyObject_HEAD
        ito::PCLPointCloud *data;
        PyObject* base;
    }
    PyPointCloud;

    #define PyPointCloud_Check(op) PyObject_TypeCheck(op, &PythonPCL::PyPointCloudType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyPointCloud_dealloc(PyPointCloud *self);
    static PyObject* PyPointCloud_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyPointCloud_init(PyPointCloud *self, PyObject *args, PyObject *kwds);


    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject *PyPointCloud_name(PyPointCloud *self);
    static PyObject *PyPointCloud_repr(PyPointCloud *self);
    static PyObject *PyPointCloud_append(PyPointCloud *self, PyObject *args, PyObject *kwds);
    static PyObject *PyPointCloud_clear(PyPointCloud *self);
    static PyObject *PyPointCloud_insert(PyPointCloud *self, PyObject *args);
    static PyObject *PyPointCloud_erase(PyPointCloud *self, PyObject *args);
    static PyObject *PyPointCloud_toDataObject(PyPointCloud *self);
    static PyObject *PyPointCloud_copy(PyPointCloud *self);
    static PyObject *PyPointCloud_scaleXYZ(PyPointCloud *self, PyObject *args, PyObject *kwds);
    static PyObject *PyPointCloud_moveXYZ(PyPointCloud *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // helper methods
    //-------------------------------------------------------------------------------------------------
    static PyObject *PyPointCloud_XYZ_append(PyPointCloud *self, PyObject *xyzObj);
    static PyObject *PyPointCloud_XYZI_append(PyPointCloud *self, PyObject *xyziObj);
    static PyObject *PyPointCloud_XYZRGBA_append(PyPointCloud *self, PyObject *xyzObj, PyObject *rgbaObj);
    static PyObject *PyPointCloud_XYZNormal_append(PyPointCloud *self, PyObject *xyz_nxnynz_curvObj);
    static PyObject *PyPointCloud_XYZINormal_append(PyPointCloud *self, PyObject *xyz_i_nxnynz_curvObj);
    static PyObject *PyPointCloud_XYZRGBNormal_append(PyPointCloud *self, PyObject *xyz_i_nxnynz_curvObj, PyObject *rgbaObj);

    static PyPointCloud* createEmptyPyPointCloud();

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPointCloud_GetType(PyPointCloud *self, void *closure);
    static PyObject* PyPointCloud_GetSize(PyPointCloud *self, void *closure);
    static PyObject* PyPointCloud_GetHeight(PyPointCloud *self, void *closure);
    static PyObject* PyPointCloud_GetWidth(PyPointCloud *self, void *closure);
    static PyObject* PyPointCloud_GetEmpty(PyPointCloud *self, void *closure);
    static PyObject* PyPointCloud_GetOrganized(PyPointCloud *self, void *closure);
    static PyObject* PyPointCloud_GetDense(PyPointCloud *self, void *closure);
    static int       PyPointCloud_SetDense(PyPointCloud *self, PyObject *value, void *closure);
    static PyObject* PyPointCloud_GetFields(PyPointCloud *self, void *closure);

    //-------------------------------------------------------------------------------------------------
    // sequence protocol
    //-------------------------------------------------------------------------------------------------
    static Py_ssize_t PyPointCloud_seqLength(PyPointCloud *self);
    static PyObject *PyPointCloud_seqConcat(PyPointCloud *self, PyObject *rhs);
    static PyObject *PyPointCloud_seqRepeat(PyPointCloud *self, Py_ssize_t size);
    static PyObject *PyPointCloud_seqItem(PyPointCloud *self, Py_ssize_t size);
    static int PyPointCloud_seqAssItem(PyPointCloud *self, Py_ssize_t size, PyObject *point);
    static PyObject *PyPointCloud_seqInplaceConcat(PyPointCloud *self, PyObject *rhs);
    static PyObject *PyPointCloud_seqInplaceRepeat(PyPointCloud *self, Py_ssize_t size);

    //-------------------------------------------------------------------------------------------------
    // mapping protocol
    //-------------------------------------------------------------------------------------------------
    static Py_ssize_t PyPointCloud_mappingLength(PyPointCloud *self);
    static PyObject *PyPointCloud_mappingGetElem(PyPointCloud *self, PyObject *key);
    static int PyPointCloud_mappingSetElem(PyPointCloud *self, PyObject *key, PyObject *value);

    //-------------------------------------------------------------------------------------------------
    // pickling
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPointCloud_Reduce(PyPointCloud *self, PyObject *args);
    static PyObject* PyPointCloud_SetState(PyPointCloud *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // static methods
    //-------------------------------------------------------------------------------------------------
    static PyObject *PyPointCloud_fromXYZ(PyPointCloud *self, PyObject *args, PyObject *kwds);
    static PyObject *PyPointCloud_fromXYZI(PyPointCloud *self, PyObject *args, PyObject *kwds);
    static PyObject *PyPointCloud_fromXYZRGBA(PyPointCloud *self, PyObject *args, PyObject *kwds);
    static PyObject *PyPointCloud_fromTopography(PyPointCloud *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    //static PyMemberDef PyNpDataObject_members[];
    static PyMethodDef PyPointCloud_methods[];
    static PyTypeObject PyPointCloudType;
    static PyModuleDef PyPointCloudModule;
    static PyGetSetDef PyPointCloud_getseters[];
    static PySequenceMethods PyPointCloud_sequenceProtocol;
    static PyMappingMethods PyPointCloud_mappingProtocol;

    static void PyPointCloud_addTpDict(PyObject *tp_dict);

    //--------------------------------------------------------------------------------------------------
    // PCL Point
    //--------------------------------------------------------------------------------------------------
    typedef struct
    {
        PyObject_HEAD
        ito::PCLPoint *point;
        PyObject* base;
    }
    PyPoint;

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyPoint_dealloc(PyPoint *self);
    static PyObject* PyPoint_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyPoint_init(PyPoint *self, PyObject *args, PyObject *kwds);


    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject *PyPoint_name(PyPoint *self);
    static PyObject *PyPoint_repr(PyPoint *self);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPoint_GetType(PyPoint *self, void *closure);

    static PyObject* PyPoint_GetXYZ(PyPoint *self, void *closure);
    static int PyPoint_SetXYZ(PyPoint *self, PyObject *value, void *closure);

    static PyObject* PyPoint_GetIntensity(PyPoint *self, void *closure);
    static int PyPoint_SetIntensity(PyPoint *self, PyObject *value, void *closure);

    static PyObject* PyPoint_GetRgb(PyPoint *self, void *closure);
    static int PyPoint_SetRgb(PyPoint *self, PyObject *value, void *closure);

    static PyObject* PyPoint_GetRgba(PyPoint *self, void *closure);
    static int PyPoint_SetRgba(PyPoint *self, PyObject *value, void *closure);

    static PyObject* PyPoint_GetCurvature(PyPoint *self, void *closure);
    static int PyPoint_SetCurvature(PyPoint *self, PyObject *value, void *closure);

    static PyObject* PyPoint_GetNormal(PyPoint *self, void *closure);
    static int PyPoint_SetNormal(PyPoint *self, PyObject *value, void *closure);

    //-------------------------------------------------------------------------------------------------
    // mapping members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPoint_mappingGetElem(PyPoint* self, PyObject* key);
    static int PyPoint_mappingSetElem(PyPoint* self, PyObject* key, PyObject* value);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    //static PyMemberDef PyNpDataObject_members[];
    static PyMethodDef PyPoint_methods[];
    static PyTypeObject PyPointType;
    static PyModuleDef PyPointModule;
    static PyGetSetDef PyPoint_getseters[];
    static PyMappingMethods PyPoint_mappingProtocol;

    static void PyPoint_addTpDict(PyObject *tp_dict);






    //--------------------------------------------------------------------------------------------------
    // PCL PolygonMesh
    //--------------------------------------------------------------------------------------------------
    typedef struct
    {
        PyObject_HEAD
        ito::PCLPolygonMesh *polygonMesh;
        PyObject* base;
    }
    PyPolygonMesh;

    #define PyPolygonMesh_Check(op) PyObject_TypeCheck(op, &PythonPCL::PyPolygonMeshType)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyPolygonMesh_dealloc(PyPolygonMesh *self);
    static PyObject* PyPolygonMesh_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyPolygonMesh_init(PyPolygonMesh *self, PyObject *args, PyObject *kwds);


    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject *PyPolygonMesh_name(PyPolygonMesh *self);
    static PyObject *PyPolygonMesh_repr(PyPolygonMesh *self);
    static PyObject *PyPolygonMesh_data(PyPolygonMesh *self);

    static PyObject *PyPolygonMesh_get(PyPolygonMesh *self, PyObject *args, PyObject *kwds);
    static PyObject* PyPolygonMesh_getCloud(PyPolygonMesh *self, PyObject *args);
    static PyObject* PyPolygonMesh_getPolygons(PyPolygonMesh *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // pickling
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPolygonMesh_Reduce(PyPolygonMesh *self, PyObject *args);
    static PyObject* PyPolygonMesh_SetState(PyPolygonMesh *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // mapping
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPolygonMesh_mappingGetElem(PyPolygonMesh* self, PyObject* key);
    static int PyPolygonMesh_mappingLength(PyPolygonMesh* self);

    //-------------------------------------------------------------------------------------------------
    // static methods
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPolygonMesh_FromCloudAndPolygons(PyObject *self, PyObject *args, PyObject *kwds);
    static PyObject* PyPolygonMesh_FromOrganizedCloud(PyObject * self, PyObject *args, PyObject *kwds);
    static PyObject* PyPolygonMesh_FromTopography(PyObject * self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // getter / setter
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyPolygonMesh_getNrOfPolygons(PyPolygonMesh *self, void *closure);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    //static PyMemberDef PyNpDataObject_members[];
    static PyMethodDef PyPolygonMesh_methods[];
    static PyTypeObject PyPolygonMeshType;
    static PyModuleDef PyPolygonMeshModule;
    static PyGetSetDef PyPolygonMesh_getseters[];
    static PyMappingMethods PyPolygonMesh_mappingProtocol;

    static void PyPolygonMesh_addTpDict(PyObject *tp_dict);

    static PyPolygonMesh* createEmptyPyPolygonMesh();

private:
    static PyObject* parseObjAsFloat32Array(PyObject *obj, npy_intp mRequired, npy_intp &n, float32 **elemRows);
    static PyObject* parseObjAsUInt8Array(PyObject *obj, npy_intp mRequired, npy_intp &n, uint8_t **elemRows);

    static void PythonPCL_SetString(PyObject *exception, const char *string);
    static void PythonPCL_SetString(PyObject *exception, const std::string &string);

};

}; //end namespace ito


#endif //#if ITOM_POINTCLOUDLIBRARY > 0

#endif
