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

#include "pythontParamConversion.h"

#include "pythonDataObject.h"
#include "pythonPCL.h"
#include "pythonPlugins.h"
#include "pythonQtConversion.h"

namespace ito
{

//! converts ito::ParamBase to the most appropriate PyObject (returns nullptr, if conversion not possible /
//! implemented).
/*!
    This methods returns the given ParamBase object to an appropriate
    Python object. If the conversion is not possible or implemented, nullptr
    is returned and a Python exception is set.

    \param param is the given ito::ParamBase parameter
    \return The corresponding PyObject, whose content correspond to param.
    \sa PyObjectToParamBase
*/
/*static*/ PyObject *PythonParamConversion::ParamBaseToPyObject(const ito::ParamBase &param)
{
    int len;
    PyObject *result = NULL;

    switch (param.getType())
    {
    case ito::ParamBase::Char:
    case ito::ParamBase::String: {
        char *value = param.getVal<char *>();
        if (value)
        {
            // result = PyUnicode_FromString(param.getVal<char*>());
            result = PyUnicode_DecodeLatin1(param.getVal<char *>(), param.getLen(), NULL);
        }
        else
        {
            Py_INCREF(Py_None);
            result = Py_None;
        }
    }

    break;

    case ito::ParamBase::Int:
        result = PyLong_FromLong(param.getVal<int32>());
        break;

    case ito::ParamBase::Double:
        result = PyFloat_FromDouble(param.getVal<float64>());
        break;

    case ito::ParamBase::Complex:
        result = PyComplex_FromDoubles(param.getVal<complex128>().real(), param.getVal<complex128>().imag());
        break;

    case (ito::ParamBase::CharArray): {
        const char *carr = param.getVal<const char *>();
        len = param.getLen();
        if (len < 0)
            len = 0;
        result = PyByteArray_FromStringAndSize(carr, len);
    }
    break;

    case (ito::ParamBase::IntArray):
        len = param.getLen();
        if (len > 0)
        {
            result = PyTuple_New(len);
            const int32 *iarr = param.getVal<const int32 *>();
            for (int n = 0; n < len; n++)
            {
                PyTuple_SetItem(result, n, PyLong_FromLong(iarr[n]));
            }
        }
        else
        {
            result = PyTuple_New(0);
        }
        break;

    case (ito::ParamBase::DoubleArray):
        len = param.getLen();
        if (len > 0)
        {
            result = PyTuple_New(len);
            const float64 *darr = param.getVal<const float64 *>();
            for (int n = 0; n < len; n++)
            {
                PyTuple_SetItem(result, n, PyFloat_FromDouble(darr[n]));
            }
        }
        else
        {
            result = PyTuple_New(0);
        }
        break;

    case (ito::ParamBase::ComplexArray):
        len = param.getLen();
        if (len > 0)
        {
            result = PyTuple_New(len);
            const complex128 *darr = param.getVal<const complex128 *>();
            for (int n = 0; n < len; n++)
            {
                PyTuple_SetItem(result, n, PyComplex_FromDoubles(darr[n].real(), darr[n].imag()));
            }
        }
        else
        {
            result = PyTuple_New(0);
        }
        break;

    case (ito::ParamBase::StringList):
        len = param.getLen();
        if (len > 0)
        {
            result = PyTuple_New(len);
            const ito::ByteArray *values = param.getVal<const ito::ByteArray *>();
            for (int n = 0; n < len; n++)
            {
                PyObject *str = PythonQtConversion::QByteArrayToPyUnicode(QByteArray(values[n].data()));
                PyTuple_SetItem(result, n, str);
            }
        }
        else
        {
            result = PyTuple_New(0);
        }
        break;

    case (ito::ParamBase::DObjPtr): {
        ito::DataObject* dObj = param.getVal<ito::DataObject*>();
        if(dObj)
        {
            ito::PythonDataObject::PyDataObject *pyDataObj = ito::PythonDataObject::createEmptyPyDataObject();
            if (pyDataObj)
            {
                pyDataObj->dataObject = new ito::DataObject(*dObj);
                result = (PyObject *)pyDataObj;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "Could not create instance of dataObject");
                return NULL;
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "Data object in parameter is invalid or empty (NULL).");
            return NULL;
        }
    }
    break;
#if ITOM_POINTCLOUDLIBRARY > 0
    case (ito::ParamBase::PointCloudPtr): {
        ito::PCLPointCloud* pointCloud = param.getVal<ito::PCLPointCloud*>();
        if(pointCloud)
        {
            ito::PythonPCL::PyPointCloud *pyPointCloud = ito::PythonPCL::createEmptyPyPointCloud();
            if (pyPointCloud)
            {
                pyPointCloud->data = new ito::PCLPointCloud(*pointCloud);
                result = (PyObject *)pyPointCloud;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "Could not create instance of pointCloud");
                return NULL;
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "PointCloud in parameter is invalid or empty (NULL).");
            return NULL;
        }
    }
    break;

    case (ito::ParamBase::PolygonMeshPtr): {
        ito::PCLPolygonMesh* polygonMesh = param.getVal<ito::PCLPolygonMesh*>();
        if (polygonMesh)
        {
            ito::PythonPCL::PyPolygonMesh *pyPolygonMesh = ito::PythonPCL::createEmptyPyPolygonMesh();
            if (pyPolygonMesh)
            {
                pyPolygonMesh->polygonMesh = new ito::PCLPolygonMesh(*polygonMesh);
                result = (PyObject *)pyPolygonMesh;
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "Could not create instance of polygonMesh");
                return NULL;
            }
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "PolygonMesh in parameter is invalid or empty (NULL).");
            return NULL;
        }
    }
    break;
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    case (ito::ParamBase::HWRef): {
        void *ptr = param.getVal<void *>();
        if (ptr)
        {
            ito::AddInBase *ptr2 = (ito::AddInBase *)(ptr);
            if (ptr2 && ptr2->getBasePlugin())
            {
                if (ptr2->getBasePlugin()->getType() & ito::typeActuator)
                {
                    ito::AddInActuator *aia = static_cast<ito::AddInActuator *>(ptr2);
                    PythonPlugins::PyActuatorPlugin *res2 = (PythonPlugins::PyActuatorPlugin *)PyObject_Call(
                        (PyObject *)&(PythonPlugins::PyActuatorPluginType), NULL, NULL);
                    if (res2) // I assume that res2 is clean (hence actuatorObj-member is NULL)
                    {
                        res2->actuatorObj = aia;
                        aia->getBasePlugin()->incRef(ptr2);
                        result = (PyObject *)res2;
                    }
                    else
                    {
                        PyErr_SetString(PyExc_RuntimeError, "Could not create instance of type actuator");
                        return NULL;
                    }
                }
                else if (ptr2->getBasePlugin()->getType() & ito::typeDataIO)
                {
                    ito::AddInDataIO *aia = static_cast<ito::AddInDataIO *>(ptr2);
                    PythonPlugins::PyDataIOPlugin *res2 = (PythonPlugins::PyDataIOPlugin *)PyObject_Call(
                        (PyObject *)&(PythonPlugins::PyDataIOPluginType), NULL, NULL);
                    if (res2) // I assume that res2 is clean (hence actuatorObj-member is NULL)
                    {
                        res2->dataIOObj = aia;
                        aia->getBasePlugin()->incRef(ptr2);
                        result = (PyObject *)res2;
                    }
                    else
                    {
                        PyErr_SetString(PyExc_RuntimeError, "Could not create instance of type dataIO");
                        return NULL;
                    }
                }
                else if (ptr2->getBasePlugin()->getType() & ito::typeAlgo)
                {
                    PyErr_SetString(PyExc_TypeError, "Parameter of type 'hardware reference' is a reference to an "
                                                     "algorithm plugin, which cannot be returned.");
                    return NULL;
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError,
                                    "Parameter of type 'hardware reference' cannot be casted to any plugin type.");
                    return NULL;
                }
            }
            else
            {
                PyErr_SetString(PyExc_TypeError,
                                "Parameter of type 'hardware reference' cannot be casted to any plugin type.");
                return NULL;
            }
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Parameter of type 'hardware reference' is NULL");
            return NULL;
        }
    }
    break;

    default:
        PyErr_SetString(PyExc_TypeError, "Undefined parameter type");
        return NULL;
        break;
    }

    return result;
}

//! converts a given PyObject to an appropriate ito::ParamBase.
/*!
    This method can be used as deleter callback for a shared pointer, covering
    an ito::ParamBase object. If this object is then deleted, this deleter
    callback will also finally deleted the internally stored object, if it
    is among one of the types ito::DataObject*, ito::PCLPointCloud or
    ito::PCLPolygonMesh.

    If the conversion is related to references to dataObjects, point clouds
    or polygon meshes, a special deleter callback is assigned to the
    returned shared pointer, such that the referenced object is deleted, too,
    if the returned ParamBase is deleted within the shared pointer.

    \param obj is the PyObject to be converted
    \param name is the name of the returned ParamBase
    \param retVal is used to store the success state of the conversion
    \param paramBaseType is the desired type of the returned ParamBase, or
        0, if the most appropriate type should be guessed (only implemented
        for some conversions / types).
    \param strict if true, the conversion is only done if the given
        obj can be directly converted to the desired paramBaseType, else
        implicit conversions (e.g. from integer to string) are allowed, too.
    \sa PyObjectToParamBaseDeleter
*/
/*static*/ SharedParamBasePointer PythonParamConversion::PyObjectToParamBase(PyObject *obj, const char *name,
                                                                             ito::RetVal &retVal,
                                                                             int paramBaseType /*= 0*/,
                                                                             bool strict /*= true*/)
{
    ito::RetVal retVal2; // used for return values from conversion methods (if available). In case of an error, retVal
                         // is set to retVal2 if it is set, else a generic error message is generated.

    // if paramBaseType == 0, type has to be guessed first
    if (paramBaseType <= 0)
    {
        // guess type by PyObject's type
        if (PyBytes_Check(obj) || PyUnicode_Check(obj))
        {
            paramBaseType = ito::ParamBase::String;
        }
        else if (PyLong_Check(obj) || obj == Py_False || obj == Py_True)
        {
            paramBaseType = ito::ParamBase::Int;
        }
        else if (PyFloat_Check(obj))
        {
            paramBaseType = ito::ParamBase::Double;
        }
        else if (PyComplex_Check(obj))
        {
            paramBaseType = ito::ParamBase::Complex;
        }
        else if (PyDataObject_Check(obj))
        {
            paramBaseType = ito::ParamBase::DObjPtr;
        }
        else if (PyArray_CheckScalar(obj))
        {
            switch (PyArray_DescrFromScalar(obj)->type_num)
            {
            case NPY_BOOL:
            case NPY_UBYTE:
            case NPY_BYTE:
            case NPY_USHORT:
            case NPY_SHORT:
            case NPY_INT:
            case NPY_UINT:
            case NPY_LONG:
            case NPY_ULONG:
            case NPY_LONGLONG:
            case NPY_ULONGLONG:
                paramBaseType = ito::ParamBase::Int;
                break;
            case NPY_FLOAT:
            case NPY_DOUBLE:
                paramBaseType = ito::ParamBase::Double;
                break;
            case NPY_CDOUBLE:
            case NPY_CFLOAT:
                paramBaseType = ito::ParamBase::Complex;
                break;
            }
        }
#if ITOM_POINTCLOUDLIBRARY > 0
        else if (PyPointCloud_Check(obj))
        {
            paramBaseType = ito::ParamBase::PointCloudPtr;
        }
        else if (PyPolygonMesh_Check(obj))
        {
            paramBaseType = ito::ParamBase::PolygonMeshPtr;
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    }

    if (paramBaseType <= 0)
    {
        retVal +=
            ito::RetVal(ito::retError, 0,
                        QObject::tr("Type of ParamBase could not be guessed with given PyObject.").toLatin1().data());
        return QSharedPointer<ito::ParamBase>();
    }

    bool ok = true;
    switch (paramBaseType)
    {
    case ito::ParamBase::Char: {
        if (obj == Py_False)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::Char, 0)); // does not require the special deleter
        }
        else if (obj == Py_True)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::Char, 1)); // does not require the special deleter
        }
        else
        {
            int i = PythonQtConversion::PyObjGetInt(obj, strict, ok);
            char c;

            if (i >= (int)std::numeric_limits<char>::min() && i <= (int)std::numeric_limits<char>::max())
            {
                c = (char)i;
            }
            else
            {
                ok = false;
            }

            if (ok)
            {
                return QSharedPointer<ito::ParamBase>(
                    new ito::ParamBase(name, ito::ParamBase::Char, c)); // does not require the special deleter
            }
        }
        break;
    }
    case ito::ParamBase::Int: {
        int32 i = 0;

        if (obj == Py_False)
        {
            i = 0;
        }
        else if (obj == Py_True)
        {
            i = 1;
        }
        else
        {
            i = PythonQtConversion::PyObjGetInt(obj, strict, ok);
        }

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::Int, i)); // does not require the special deleter
        }
    }
    break;
    case ito::ParamBase::Double: {
        float64 d = PythonQtConversion::PyObjGetDouble(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::Double, d)); // does not require the special deleter
        }
    }
    break;
    case ito::ParamBase::Complex: {
        complex128 d = PythonQtConversion::PyObjGetComplex(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::Complex, d)); // does not require the special deleter
        }
    }
    break;
    case ito::ParamBase::String: {
        QString s = PythonQtConversion::PyObjGetString(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(new ito::ParamBase(
                name, ito::ParamBase::String, s.toLatin1().data())); // does not require the special deleter
        }
    }
    break;
    case ito::ParamBase::IntArray: {
        QVector<int32> ia = PythonQtConversion::PyObjGetIntArray(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::IntArray, ia.size(), ia.constData()));
        }
        break;
    }
    case ito::ParamBase::DoubleArray: {
        QVector<float64> da = PythonQtConversion::PyObjGetDoubleArray(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::DoubleArray, da.size(), da.constData()));
        }
        break;
    }
    case ito::ParamBase::ComplexArray: {
        QVector<complex128> da = PythonQtConversion::PyObjGetComplexArray(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::ComplexArray, da.size(), da.constData()));
        }
        break;
    }
    case ito::ParamBase::StringList: {
        QVector<ito::ByteArray> list = PythonQtConversion::PyObjGetByteArrayList(obj, strict, ok);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(new ito::ParamBase(name, ito::ParamBase::StringList, list.size(), list.constData()));
        }
        break;
    }
    case ito::ParamBase::CharArray: {
        if (PyByteArray_Check(obj))
        {
            char *buf = (char *)PyByteArray_AsString(obj);
            Py_ssize_t listlen = PyByteArray_Size(obj);
            return QSharedPointer<ito::ParamBase>(new ito::ParamBase(name, ito::ParamBase::CharArray, listlen, buf));
        }
        else
        {
            ok = false;
        }
        break;
    }
    case ito::ParamBase::DObjPtr: {
        ito::DataObject *dObj = PythonQtConversion::PyObjGetDataObjectNewPtr(obj, strict, ok, &retVal2, true);
        ok &= (dObj != nullptr);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::DObjPtr, (char *)dObj),
                PyObjectToParamBaseDeleter); // deleter deletes newly created dataObject, if parambase is deleted
        }
    }
    break;
#if ITOM_POINTCLOUDLIBRARY > 0
    case ito::ParamBase::PointCloudPtr: {
        ito::PCLPointCloud *p = PythonQtConversion::PyObjGetPointCloudNewPtr(obj, strict, ok);
        ok &= (p != nullptr);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::PointCloudPtr, (char *)p),
                PyObjectToParamBaseDeleter); // deleter deletes newly created dataObject, if parambase is deleted
        }
    }
    break;
    case ito::ParamBase::PolygonMeshPtr: {
        ito::PCLPolygonMesh *m = PythonQtConversion::PyObjGetPolygonMeshNewPtr(obj, strict, ok);
        ok &= (m != nullptr);

        if (ok)
        {
            return QSharedPointer<ito::ParamBase>(
                new ito::ParamBase(name, ito::ParamBase::PolygonMeshPtr, (char *)m),
                PyObjectToParamBaseDeleter); // deleter deletes newly created dataObject, if parambase is deleted
        }
    }
    break;
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    default: {
        retVal += ito::RetVal(ito::retError, 0, QObject::tr("Given paramBaseType is unsupported.").toLatin1().data());
        return QSharedPointer<ito::ParamBase>();
    }
    }

    if (!ok)
    {
        if (retVal2 != ito::retError)
        {
            retVal +=
                ito::RetVal(ito::retError, 0,
                            QObject::tr("Error while converting value from PyObject to ParamBase").toLatin1().data());
        }
        else
        {
            retVal += retVal2;
        }
    }

    return QSharedPointer<ito::ParamBase>();
}


//------------------------------------------------------------------------------------
//! special deleter for param, where the wrapped object is deleted, too.
/*!
    This method can be used as deleter callback for a shared pointer, covering
    an ito::ParamBase object. If this object is then deleted, this deleter
    callback will also finally deleted the internally stored object, if it
    is among one of the types ito::DataObject*, ito::PCLPointCloud or
    ito::PCLPolygonMesh.

    This is necessary, since ito::ParamBase only hold a pointer (like a weak pointer)
    to these objects and will therefore not automatically deleted them.

    \param param is the ito::ParamBase object to be deleted.
    \sa PyObjectToParamBase
*/
/*static*/ void PythonParamConversion::PyObjectToParamBaseDeleter(ito::ParamBase *param)
{
    if (param)
    {
        switch (param->getType())
        {
        case ito::ParamBase::DObjPtr: {
            ito::DataObject *d = param->getVal<ito::DataObject *>();
            DELETE_AND_SET_NULL(d);
        }
        break;
#if ITOM_POINTCLOUDLIBRARY > 0
        case ito::ParamBase::PointCloudPtr: {
            ito::PCLPointCloud *d = param->getVal<ito::PCLPointCloud *>();
            DELETE_AND_SET_NULL(d);
        }
        break;
        case ito::ParamBase::PolygonMeshPtr: {
            ito::PCLPolygonMesh *d = param->getVal<ito::PCLPolygonMesh *>();
            DELETE_AND_SET_NULL(d);
        }
        break;
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
        default:
            // do nothing
            break;
        }
        DELETE_AND_SET_NULL(param);
    }
}

}; // end namespace ito
