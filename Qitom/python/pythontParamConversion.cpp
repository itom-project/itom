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

#include "pythontParamConversion.h"

#include "pythonDataObject.h"
#include "pythonQtConversion.h"
#include "pythonPlugins.h"
#include "pythonPCL.h"

namespace ito
{

/*static*/ PyObject *PythonParamConversion::ParamBaseToPyObject(const ito::ParamBase &param)
{
    int len;
    int *iarr = NULL;
    double *darr = NULL;
    char *carr = NULL;
    PyObject *result = NULL;

    switch(param.getType())
    {
        case ito::ParamBase::Char & ito::paramTypeMask:
        case ito::ParamBase::String & ito::paramTypeMask:
            {
            char *value = param.getVal<char*>();
            if(value)
            {
                result = PyUnicode_FromString(param.getVal<char*>());
            }
            else
            {
                Py_INCREF(Py_None);
                result = Py_None;
            }
            }
            
        break;

        case ito::ParamBase::Int & ito::paramTypeMask:
            result = PyLong_FromLong( param.getVal<int>() );
        break;

        case ito::ParamBase::Double & ito::paramTypeMask:
            result = PyFloat_FromDouble( param.getVal<double>());
        break;

        case (ito::ParamBase::CharArray & ito::paramTypeMask):
            //result = PyTuple_New(1);
            carr = param.getVal<char*>();
            len = param.getLen();
            //PyTuple_SetItem(result, 0, PyByteArray_FromStringAndSize(carr, len));
            result = PyByteArray_FromStringAndSize(carr, len);
        break;

        case (ito::ParamBase::IntArray & ito::paramTypeMask):
            len = param.getLen();
            if(len > 0)
            {
                result = PyTuple_New(len);
                iarr = param.getVal<int*>();
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

        case (ito::ParamBase::DoubleArray & ito::paramTypeMask):
            len = param.getLen();
            if(len > 0)
            {
                result = PyTuple_New(len);
                darr = param.getVal<double*>();
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

        case (ito::ParamBase::DObjPtr & ito::paramTypeMask):
        {
            ito::DataObject* dObj = (ito::DataObject*)(param.getVal<void*>());
            if(dObj)
            {
                ito::PythonDataObject::PyDataObject *pyDataObj = ito::PythonDataObject::createEmptyPyDataObject();
                if(pyDataObj)
                {
                    pyDataObj->dataObject = new ito::DataObject(*dObj);
                    result = (PyObject*)pyDataObj;
                }
                else
                {
                    PyErr_SetString(PyExc_RuntimeError, "could not create instance of dataObject");
                    return NULL;
                }
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "data object in parameter is invalid or empty (NULL).");
                return NULL;
            }

            
        }
        break;
#if ITOM_POINTCLOUDLIBRARY > 0
        case (ito::ParamBase::PointCloudPtr & ito::paramTypeMask):
        {
            ito::PCLPointCloud* pointCloud = (ito::PCLPointCloud*)(param.getVal<void*>());
            if(pointCloud)
            {
                ito::PythonPCL::PyPointCloud *pyPointCloud = ito::PythonPCL::createEmptyPyPointCloud();
                if(pyPointCloud)
                {
                    pyPointCloud->data = new ito::PCLPointCloud(*pointCloud);
                    result = (PyObject*)pyPointCloud;
                }
                else
                {
                    PyErr_SetString(PyExc_RuntimeError, "could not create instance of pointCloud");
                    return NULL;
                }
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "pointCloud in parameter is invalid or empty (NULL).");
                return NULL;
            }

            
        }
        break;

        case (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask):
        {
            ito::PCLPolygonMesh* polygonMesh = (ito::PCLPolygonMesh*)(param.getVal<void*>());
            if(polygonMesh)
            {
                ito::PythonPCL::PyPolygonMesh *pyPolygonMesh = ito::PythonPCL::createEmptyPyPolygonMesh();
                if(pyPolygonMesh)
                {
                    pyPolygonMesh->polygonMesh = new ito::PCLPolygonMesh(*polygonMesh);
                    result = (PyObject*)pyPolygonMesh;
                }
                else
                {
                    PyErr_SetString(PyExc_RuntimeError, "could not create instance of polygonMesh");
                    return NULL;
                }
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "polygonMesh in parameter is invalid or empty (NULL).");
                return NULL;
            }

            
        }
        break;
#endif //#if ITOM_POINTCLOUDLIBRARY > 0        
        case (ito::ParamBase::HWRef & ito::paramTypeMask):
            {
                void *ptr = param.getVal<void*>();
                if(ptr)
                {
                    ito::AddInBase* ptr2 = (ito::AddInBase*)(ptr);
                    if(ptr2 && ptr2->getBasePlugin())
                    {
                        if( ptr2->getBasePlugin()->getType() & ito::typeActuator )
                        {
                            ito::AddInActuator* aia = static_cast<ito::AddInActuator*>(ptr2);
                            PythonPlugins::PyActuatorPlugin *res2 = (PythonPlugins::PyActuatorPlugin*)PyObject_Call((PyObject*)&(PythonPlugins::PyActuatorPluginType), NULL, NULL);
                            if(res2) //I assume that res2 is clean (hence actuatorObj-member is NULL)
                            {
                                res2->actuatorObj = aia;
                                aia->getBasePlugin()->incRef(ptr2);
                                result = (PyObject*)res2;
                            }
                            else
                            {
                                PyErr_SetString(PyExc_RuntimeError, "could not create instance of type actuator");
                                return NULL;
                            }
                        }
                        else if( ptr2->getBasePlugin()->getType() & ito::typeDataIO )
                        {
                            ito::AddInDataIO* aia = static_cast<ito::AddInDataIO*>(ptr2);
                            PythonPlugins::PyDataIOPlugin *res2 = (PythonPlugins::PyDataIOPlugin*)PyObject_Call((PyObject*)&(PythonPlugins::PyDataIOPluginType), NULL, NULL);
                            if(res2) //I assume that res2 is clean (hence actuatorObj-member is NULL)
                            {
                                res2->dataIOObj = aia;
                                aia->getBasePlugin()->incRef(ptr2);
                                result = (PyObject*)res2;
                            }
                            else
                            {
                                PyErr_SetString(PyExc_RuntimeError, "could not create instance of type dataIO");
                                return NULL;
                            }
                        }
                        else if( ptr2->getBasePlugin()->getType() & ito::typeAlgo )
                        {
                            PyErr_Format(PyExc_TypeError, "parameter of type 'hardware reference' is a reference to an algorithm plugin, which cannot be returned.");
                            return NULL;
                        }
                        else
                        {
                            PyErr_Format(PyExc_TypeError, "parameter of type 'hardware reference' cannot be casted to any plugin type.");
                            return NULL;
                        }
                    }
                    else
                    {
                        PyErr_Format(PyExc_TypeError, "parameter of type 'hardware reference' cannot be casted to any plugin type.");
                        return NULL;
                    }
                }
                else
                {
                    PyErr_Format(PyExc_TypeError, "parameter of type 'hardware reference' is NULL");
                    return NULL;
                }
                
                
            }
        break;

        default:
            PyErr_Format(PyExc_TypeError, "undefined parameter type");
            return NULL;
        break;
    }

    return result;
}

/*static*/ SharedParamBasePointer PythonParamConversion::PyObjectToParamBase(PyObject* obj, const char *name, ito::RetVal &retVal, int paramBaseType /*= 0*/, bool strict /*= true*/)
{
    //if paramBaseType == 0, type has to be guessed first
    if (paramBaseType <= 0) 
    {
        //guess type by PyObject's type
        if(PyBytes_Check(obj) || PyUnicode_Check(obj))
        {
            paramBaseType = ito::ParamBase::String;
        }
        else if(PyLong_Check(obj) || obj == Py_False || obj == Py_True)
        {
            paramBaseType = ito::ParamBase::Int;
        }
        else if(PyFloat_Check(obj))
        {
            paramBaseType = ito::ParamBase::Double;
        }
        else if(PyDataObject_Check(obj))
        {
            paramBaseType = ito::ParamBase::DObjPtr;
        }
#if ITOM_POINTCLOUDLIBRARY > 0
        else if(PyPointCloud_Check(obj))
        {
            paramBaseType = ito::ParamBase::PointCloudPtr;
        }
        else if(PyPolygonMesh_Check(obj))
        {
            paramBaseType = ito::ParamBase::PolygonMeshPtr;
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    }

    if(paramBaseType <= 0)
    {
        retVal += ito::RetVal(ito::retError,0,"type of ParamBase could not be guessed with given PyObject.");
        return QSharedPointer<ito::ParamBase>();
    }
    
    bool ok = true;
    switch(paramBaseType & ito::paramTypeMask)
    {
    case ito::ParamBase::Char:
        {
            if(obj == Py_False)
            {
                return QSharedPointer<ito::ParamBase>(new ito::ParamBase(name, ito::ParamBase::Char, 0) ); //does not require the special deleter
            }
            else if(obj == Py_True)
            {
                return QSharedPointer<ito::ParamBase>(new ito::ParamBase(name, ito::ParamBase::Char, 1) ); //does not require the special deleter
            }
            else
            {
                int i = PythonQtConversion::PyObjGetInt(obj, strict, ok);
                char c;
                if(i >= (int)std::numeric_limits<char>::min() && i <= (int)std::numeric_limits<char>::max())
                {
                    c = (char)i;
                }
                else
                {
                    ok = false;
                }
                if(ok)
                {
                    return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::Char, c) ); //does not require the special deleter
                }
            }
        }
    case ito::ParamBase::Int:
        {
            int i = 0;
            if(obj == Py_False)
            {
                i = 0;
            }
            else if(obj == Py_True)
            {
                i = 1;
            }
            else
            {
                i = PythonQtConversion::PyObjGetInt(obj, strict, ok);
            }
            if(ok)
            {
                return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::Int, i) ); //does not require the special deleter
            }
        }
        break;
    case ito::ParamBase::Double:
        {
            double d = PythonQtConversion::PyObjGetDouble(obj, strict, ok);
            if(ok)
            {
                return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::Double, d) ); //does not require the special deleter
            }
        }
        break;
    case ito::ParamBase::String:
        {
            QString s = PythonQtConversion::PyObjGetString(obj, strict, ok);
            if(ok)
            {
                return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::String, s.toAscii().data()) ); //does not require the special deleter
            }
        }
        break;
    case ito::ParamBase::DObjPtr & ito::paramTypeMask:
        {
            ito::DataObject *dObj = PythonQtConversion::PyObjGetDataObjectNewPtr(obj, strict, ok);
            ok &= (dObj != NULL);
            if(ok)
            {
                return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::DObjPtr, (char*)dObj), PyObjectToParamBaseDeleter ); //deleter deletes newly created dataObject, if parambase is deleted
            }
        }
        break;
#if ITOM_POINTCLOUDLIBRARY > 0
    case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
        {
            ito::PCLPointCloud *p = PythonQtConversion::PyObjGetPointCloudNewPtr(obj, strict, ok);
            ok &= (p != NULL);
            if(ok)
            {
                return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::PointCloudPtr, (char*)p), PyObjectToParamBaseDeleter ); //deleter deletes newly created dataObject, if parambase is deleted
            }
        }
        break;
    case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
        {
            ito::PCLPolygonMesh *m = PythonQtConversion::PyObjGetPolygonMeshNewPtr(obj, strict, ok);
            ok &= (m != NULL);
            if(ok)
            {
                return QSharedPointer<ito::ParamBase>( new ito::ParamBase(name, ito::ParamBase::PolygonMeshPtr, (char*)m), PyObjectToParamBaseDeleter ); //deleter deletes newly created dataObject, if parambase is deleted
            }
        }
        break;
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    default:
        {
        retVal += ito::RetVal(ito::retError,0,"given paramBaseType is unsupported.");
        return QSharedPointer<ito::ParamBase>();
        break;
        }
    }

    if(!ok)
    {
        retVal += ito::RetVal(ito::retError,0,"error while converting value from PyObject to ParamBase");
    }
    return QSharedPointer<ito::ParamBase>();


}

/*static*/ void PythonParamConversion::PyObjectToParamBaseDeleter(ito::ParamBase *param)
{
    if(param)
    {
        switch(param->getType())
        {
        case ito::ParamBase::DObjPtr & ito::paramTypeMask:
            {
                ito::DataObject *d = (ito::DataObject*)param->getVal<void*>();
                if(d) delete d;
            }
            break;
#if ITOM_POINTCLOUDLIBRARY > 0
         case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
            {
                ito::PCLPointCloud *d = (ito::PCLPointCloud*)param->getVal<void*>();
                if(d) delete d;
            }
            break;
         case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
            {
                ito::PCLPolygonMesh *d = (ito::PCLPolygonMesh*)param->getVal<void*>();
                if(d) delete d;
            }
            break;
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
        default:
            //do nothing
            break;
        }
        delete param;
        param = NULL;
    }
}


}; //end namespace ito