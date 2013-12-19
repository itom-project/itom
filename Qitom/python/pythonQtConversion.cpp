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

#include "pythonQtConversion.h"

#include "pythonUi.h"
#include "pythonCommon.h"
#include "pythonRgba.h"

#include <qstringlist.h>
#include <qurl.h>
#include <qtextcodec.h>

#include "pythonSharedPointerGuard.h"


//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class PythonQtConversion
    \brief Conversion class with mainly static methods, which convert values between Qt and standard C++ data types
        and PyObject* values.

        Parts of this class are taken from the project PythonQt (http://pythonqt.sourceforge.net/)
*/
PythonQtConversion::unicodeEncodings PythonQtConversion::textEncoding = PythonQtConversion::utf_8;
QByteArray PythonQtConversion::textEncodingName = "utf8";
QHash<void*,PyObject*> PythonQtConversion::m_pyBaseObjectStorage = QHash<void*, PyObject*>();

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to QStringList
/*!
    tries to interprete given PyObject* as list of strings and converts it to QStringList.
    If strict is true, we do not want to convert a string to a stringlist, since single strings in python
    are also detected to be sequences.

    \param val is the given python object
    \param strict indicates if any object fitting to the sequence protocol is interpreted as string list, too [true]
    \param ok (ByRef) is set to true if conversion was successful, else false
    \return resulting QStringList
*/
QStringList PythonQtConversion::PyObjToStringList(PyObject* val, bool strict, bool& ok) 
{
    QStringList v;
    ok = false;
    // if we are strict, we do not want to convert a string to a stringlist
    // (strings in python are detected to be sequences)
    if (strict && (PyBytes_Check(val) || PyUnicode_Check(val))) 
    {
        ok = false;
        return v;
    }
    if (PySequence_Check(val)) 
    {
        int count = PySequence_Size(val);
        PyObject *value = NULL;
        for (int i = 0; i < count; i++) 
        {
            value = PySequence_GetItem(val, i); //new reference
            v.append(PyObjGetString(value, false, ok));
            Py_XDECREF(value);
        }
        ok = true;
    }
    return v;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! string representation of PyObject*
/*!
    returns a string or a string-like interpretation of the given PyObject*. If this object is no byte- or unicode-object,
    its string representation is returned, obtained by calling the tp_repr-slot of the corresponding Type-struct.

    \param val is the given python object
    \return resulting QString
*/
QString PythonQtConversion::PyObjGetRepresentation(PyObject* val)
{
    QString r;
    PyObject* str =  PyObject_Repr(val);
    if (str) 
    {
        bool ok;
        r = PyObjGetString(val, false, ok);
        Py_DECREF(str);
    }
    return r;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to QString
/*!
    If val is a byte-object, it is directly converted into a QString. If val is an unicode-object,
    its value is converted using the current encoding and returned. In any other case the string-like-representation is only
    returned if strict is set to false.

    \param val is the given python object
    \param strict indicates if only real byte or unicode objects can be converted to string
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting QString
*/
QString PythonQtConversion::PyObjGetString(PyObject* val, bool strict, bool& ok) 
{
    QString r;
    ok = true;
    if (PyBytes_Check(val))
    {
        r = QString(PyObjGetBytes(val, strict, ok));
    }
    else if (PyUnicode_Check(val))
    {
        PyObject *repr2 = PyUnicodeToPyByteObject(val);
        if (repr2 != NULL)
        {
            r = QString(PyObjGetBytes(repr2, strict, ok));
            Py_XDECREF(repr2);
        }
    } 
    else if (!strict) 
    {
        // EXTRA: could also use _Unicode, but why should we?
        PyObject* str =  PyObject_Str(val);
        if (str) 
        {
            r = PyObjGetString(str, strict, ok);
            Py_DECREF(str);
        } 
        else 
        {
            ok = false;
        }
    } 
    else 
    {
        ok = false;
    }
    return r;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to QString
/*!
    If val is a byte-object, it is directly converted into a QString. If val is an unicode-object,
    its value is converted using the current encoding and returned. In any other case the string-like-representation is only
    returned if strict is set to false.

    \param val is the given python object
    \param strict indicates if only real byte or unicode objects can be converted to string
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting QString
*/
QByteArray PythonQtConversion::PyObjGetBytes(PyObject* val, bool strict, bool& ok) 
{
    // TODO: support buffer objects in general
    QByteArray r;
    ok = true;
    if (PyBytes_Check(val)) 
    {
        Py_ssize_t size = PyBytes_GET_SIZE(val);
        r = QByteArray(PyBytes_AS_STRING(val), size);
    } 
    else if (strict)
    {
        ok = false;
    }
    else
    {
        if (PyUnicode_Check(val))
        {
            PyObject *repr2 = PyUnicodeToPyByteObject(val);
            if (repr2 != NULL)
            {
                r = PyObjGetBytes(repr2, strict, ok);
                Py_DECREF(repr2);
            }
            else
            {
                ok = false;
            }
        } 
        else 
        {
            // EXTRA: could also use _Unicode, but why should we?
            PyObject* str =  PyObject_Str(val);
            if (str) 
            {
                r = PyObjGetBytes(str, strict, ok);
                Py_DECREF(str);
            } 
            else 
            {
                ok = false;
            }
        } 
    }
    return r;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to bool
/*!
    tries to convert the given PyObject* val as boolean variable and returns its value. If val is no
    boolean object the output depends on the param strict. If strict==false, the output is true, if the integer
    conversion (strict==false) of val is unequal to zero.

    \param val is the given python object
    \param strict indicates if only real boolean types should be converted
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting bool
*/
bool PythonQtConversion::PyObjGetBool(PyObject* val, bool strict, bool &ok) 
{
    bool d = false;
    ok = false;
    if (val == Py_False) 
    {
        d = false;
        ok = true;
    } 
    else if (val == Py_True) 
    {
        d = true;
        ok = true;
    } 
    else if (!strict) 
    {
        d = PyObjGetInt(val, false, ok) != 0;
        ok = true;
    }
    return d;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to integer
/*!
    If val is a fixed-point object, it is directly converted into an integer variable. Otherwise, the output depends
    on strict. If strict is equal to false, any floating point object is rounded using floor and the result is returned.
    Additionally boolean variables are returned as 0 (false) or 1 (true) if strict is false, too.

    \param val is the given python object
    \param strict indicates if only real integer or long types should be converted
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting integer
*/
int PythonQtConversion::PyObjGetInt(PyObject* val, bool strict, bool &ok) 
{
    int d = 0;
    ok = true;
    if (PyLong_Check(val)) 
    {
        int overflow;
        d = PyLong_AsLongAndOverflow(val, &overflow);
        if (overflow) //1: too big, -1: too small
        {
            ok = false;
        }
    } 
    else if (!strict) 
    {
        if (PyFloat_Check(val)) 
        {
            d = floor(PyFloat_AS_DOUBLE(val));
        } 
        else if (val == Py_False) 
        {
            d = 0;
        } 
        else if (val == Py_True) 
        {
            d = 1;
        } 
        else 
        {
            ok = false;
        }
    } 
    else 
    {
        ok = false;
    }
    return d;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to long long (64bit, if possible)
/*!
    If val is a fixed-point object, it is directly converted into an qint64 variable. Otherwise, the output depends
    on strict. If strict is equal to false, any floating point object is rounded using floor and the result is returned.
    Additionally boolean variables are returned as 0 (false) or 1 (true) if strict is false, too.

    \param val is the given python object
    \param strict indicates if only real integer or long types should be converted
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting qint64
*/
qint64 PythonQtConversion::PyObjGetLongLong(PyObject* val, bool strict, bool &ok) 
{
    qint64 d = 0;
    ok = true;
    if (PyLong_Check(val)) 
    {
        int overflow;
        d = PyLong_AsLongLongAndOverflow(val, &overflow);
        if (overflow) //1: too big, -1: too small
        {
            ok = false;
        }
    } 
    else if (!strict) 
    {
        if (PyFloat_Check(val)) 
        {
            d = floor(PyFloat_AS_DOUBLE(val));
        } 
        else if (val == Py_False) 
        {
            d = 0;
        } 
        else if (val == Py_True) 
        {
            d = 1;
        } 
        else 
        {
            ok = false;
        }
    } 
    else 
    {
        ok = false;
    }
    return d;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to unsigned integer 64bit
/*!
    If val is a fixed-point object, it is directly converted into an quint64 variable. Otherwise, the output depends
    on strict. If strict is equal to false, any floating point object is rounded using floor and the result is returned.
    Additionally boolean variables are returned as 0 (false) or 1 (true) if strict is false, too.

    \param val is the given python object
    \param strict indicates if only real integer or long types should be converted
    \param ok (ByRef) is set to true if conversion succeeded. Conversion fails if number is smaller than zero.
    \return resulting quint64
*/
quint64 PythonQtConversion::PyObjGetULongLong(PyObject* val, bool strict, bool &ok) 
{
    quint64 d = 0;
    ok = true;
    if (PyLong_Check(val)) {
        d = PyLong_AsUnsignedLongLong(val);
        if (PyErr_Occurred())
        {
            ok = false;
            PyErr_Clear();
        }
    } 
    else if (!strict) 
    {
        if (PyFloat_Check(val)) 
        {
            d = floor(PyFloat_AS_DOUBLE(val));
        } 
        else if (val == Py_False) 
        {
            d = 0;
        } 
        else if (val == Py_True) 
        {
            d = 1;
        } 
        else 
        {
            ok = false;
        }
    } 
    else 
    {
        ok = false;
    }
    return d;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to double
/*!
    If val is a floating-point object, it is directly converted into a double variable. Otherwise, the output depends
    on strict. If strict is equal to false, any fixed-point object is interpreted as double and the result is returned.
    Additionally boolean variables are returned as 0.0 (false) or 1.0 (true) if strict is false, too.

    \param val is the given python object
    \param strict indicates if only real floating point numbers should be converted
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting double value
*/
double PythonQtConversion::PyObjGetDouble(PyObject* val, bool strict, bool &ok) 
{
    double d = 0;
    ok = true;
    if (PyFloat_Check(val)) 
    {
        d = PyFloat_AS_DOUBLE(val);
    } 
    else if (!strict) 
    {
        if (PyLong_Check(val)) 
        {
            int overflow;
            d = PyLong_AsLongAndOverflow(val, &overflow);
            if (overflow)
            {
                ok = false;
            }
        } 
        else if (val == Py_False) 
        {
            d = 0.0;
        } 
        else if (val == Py_True) 
        {
            d = 1.0;
        } 
        else 
        {
            ok = false;
        }
    } 
    else 
    {
        ok = false;
    }
    return d;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! get double-array from py object
QVector<double> PythonQtConversion::PyObjGetDoubleArray(PyObject* val, bool strict, bool &ok)
{
    ok = true;
    QVector<double> v;
    if (PySequence_Check(val) == false)
    {
        ok = false;
        return v;
    }

    Py_ssize_t len = PySequence_Size(val);
    PyObject *t = NULL;

    if (strict)
    {
        for (Py_ssize_t i = 0; i < len; i++)
        {
            t = PySequence_GetItem(val, i); //new reference
            if (PyFloat_Check(t))
            {
                v.append(PyFloat_AS_DOUBLE(t));
            }
            else
            {
                ok = false;
                Py_XDECREF(t);
                break;
            }
            Py_XDECREF(t);
        }
    }
    else
    {
        int overflow;
        for (Py_ssize_t i = 0; i < len; i++)
        {
            t = PySequence_GetItem(val, i); //new reference
            
            if (PyFloat_Check(t)) 
            {
                v.append(PyFloat_AS_DOUBLE(t));
            } 
            else if (PyLong_Check(t))
            {
                qreal v2 = PyLong_AsLongAndOverflow(t, &overflow);
                if (overflow)
                {
                    v2 = PyLong_AsLongLongAndOverflow(t, &overflow);
                    if (overflow)
                    {
                        ok = false;
                        Py_XDECREF(t);
                        break;
                    }
                }
                v.append(v2);
            }
            else if (t == Py_False) 
            {
                v.append(0);
            } 
            else if (t == Py_True) 
            {
                v.append(1);
            } 
            else
            {
                ok = false;
                Py_XDECREF(t);
                break;
            }
            Py_XDECREF(t);
        }
    }

    if (!ok)
    {
        v.clear();
    }

    return v;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! get int-array from py object
QVector<int> PythonQtConversion::PyObjGetIntArray(PyObject* val, bool strict, bool &ok)
{
    QVector<int> v;
    int overflow;
    ok = true;

    if (PySequence_Check(val) == false)
    {
        ok = false;
        return v;
    }

    Py_ssize_t len = PySequence_Size(val);
    PyObject *t = NULL;

    if (strict)
    {
        for (Py_ssize_t i = 0; i < len; i++)
        {
            t = PySequence_GetItem(val,i); //new reference
            if (PyLong_Check(t))
            {
                v.append(PyLong_AsLongAndOverflow(t, &overflow));
                if (overflow)
                {
                    ok = false;
                    Py_XDECREF(t);
                    break;
                }
            }
            else
            {
                ok = false;
                Py_XDECREF(t);
                break;
            }
            Py_XDECREF(t);
        }
    }
    else
    {
        for (Py_ssize_t i = 0; i < len; i++)
        {
            t = PySequence_GetItem(val,i); //new reference
            if (PyLong_Check(t))
            {
                v.append(PyLong_AsLongAndOverflow(t, &overflow));
                if (overflow)
                {
                    ok = false;
                    Py_XDECREF(t);
                    break;
                }
            }
            else if (PyFloat_Check(t)) 
            {
                v.append(floor(PyFloat_AS_DOUBLE(t)));
            } 
            else if (t == Py_False) 
            {
                v.append(0);
            } 
            else if (t == Py_True) 
            {
                v.append(1);
            } 
            else
            {
                ok = false;
                Py_XDECREF(t);
                break;
            }
            Py_XDECREF(t);
        }
    }

    if (!ok)
    {
        v.clear();
    }

    return v;
}

#if ITOM_POINTCLOUDLIBRARY > 0
//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to ito::PCLPointCloud
/*!
    If val is of type pointCloud (Python class), the containing ito::PCLPointCloud is returned, otherwise an empty
    point cloud is returned.

    \param val is the given python object
    \param strict - no functionality in this case
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting ito::PCLPointCloud
*/
ito::PCLPointCloud PythonQtConversion::PyObjGetPointCloud(PyObject *val, bool /*strict*/, bool &ok)
{
    if (Py_TYPE(val) == &ito::PythonPCL::PyPointCloudType)
    {
        ito::PythonPCL::PyPointCloud* pyPlc = (ito::PythonPCL::PyPointCloud*)val;
        if (pyPlc->data)
        {
            ok = true;
            return *(pyPlc->data);
        }
        ok = false;
        return ito::PCLPointCloud();
    }
    else
    {
        ok = false;
        return ito::PCLPointCloud();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to ito::PCLPoint
/*!
    If val is of type point (Python class), the containing ito::PCLPoint is returned, otherwise an empty
    point is returned.

    \param val is the given python object
    \param strict - no functionality in this case
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting ito::PCLPoint
*/
ito::PCLPoint PythonQtConversion::PyObjGetPoint(PyObject *val, bool /*strict*/, bool &ok)
{
    if (Py_TYPE(val) == &ito::PythonPCL::PyPointType)
    {
        ito::PythonPCL::PyPoint* pyPt = (ito::PythonPCL::PyPoint*)val;
        if (pyPt->point)
        {
            ok = true;
            return *(pyPt->point);
        }
        ok = false;
        return ito::PCLPoint();
    }
    else
    {
        ok = false;
        return ito::PCLPoint();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to ito::PCLPolygonMesh
/*!
    If val is of type polygonMesh (Python class), the containing ito::PCLPolygonMesh is returned, otherwise an empty
    point is returned.

    \param val is the given python object
    \param strict - no functionality in this case
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting ito::PCLPolygonMesh
*/
ito::PCLPolygonMesh PythonQtConversion::PyObjGetPolygonMesh(PyObject *val, bool /*strict*/, bool &ok)
{
    if (Py_TYPE(val) == &ito::PythonPCL::PyPolygonMeshType)
    {
        ito::PythonPCL::PyPolygonMesh* pyMesh = (ito::PythonPCL::PyPolygonMesh*)val;
        if (pyMesh->polygonMesh)
        {
            ok = true;
            return *(pyMesh->polygonMesh);
        }
        ok = false;
        return ito::PCLPolygonMesh();
    }
    else
    {
        ok = false;
        return ito::PCLPolygonMesh();
    }
}

#endif //#if ITOM_POINTCLOUDLIBRARY > 0

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::DataObject* PythonQtConversion::PyObjGetDataObjectNewPtr(PyObject *val, bool strict, bool &ok)
{
    if (Py_TYPE(val) == &ito::PythonDataObject::PyDataObjectType)
    {
        ito::PythonDataObject::PyDataObject* dObj = (ito::PythonDataObject::PyDataObject*)val;
        if (dObj->dataObject && dObj->base == NULL)
        {
            ok = true;
            return (new ito::DataObject(*(dObj->dataObject)));
        }
        else if (dObj->dataObject && dObj->base != NULL) //make deep copy since ref of base cannot be incremented (it can, but nobody will decrement it if the dataObject is deleted)
        {
            ok = true;
            ito::DataObject *dObj2 = new ito::DataObject();
            if (dObj->dataObject->copyTo(*dObj2, true) == ito::retOk)
            {
                return dObj2;
            }
            else
            {
                DELETE_AND_SET_NULL(dObj2);
                ok = false;
                return NULL;
            }
        }
        ok = false;
        return NULL;
    }
    else if (strict == false) //try to convert to dataObject
    {
        PyObject *args = Py_BuildValue("(O)", val);
        ito::PythonDataObject::PyDataObject *result = (ito::PythonDataObject::PyDataObject*)PyObject_Call((PyObject*)&ito::PythonDataObject::PyDataObjectType, args, NULL); //new reference
        ito::DataObject *dObj = NULL;
        Py_DECREF(args);
        if (result)
        {
            dObj = PyObjGetDataObjectNewPtr((PyObject*)result, true, ok);
            Py_XDECREF(result);
            return dObj;
        }
        else
        {
            ok = false;
            return NULL;
        }
    }
    else
    {
        ok = false;
        return NULL;
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QSharedPointer<ito::DataObject> PythonQtConversion::PyObjGetSharedDataObject(PyObject *val, bool &ok) //is always strict, only dataobjects are allowed
{
    if (Py_TYPE(val) == &ito::PythonDataObject::PyDataObjectType)
    {
        ito::PythonDataObject::PyDataObject* dObj = (ito::PythonDataObject::PyDataObject*)val;
        if (dObj->dataObject)
        {
            ok = true;
            return ito::PythonSharedPointerGuard::createPythonSharedPointer<ito::DataObject>(dObj->dataObject, val);
        }
    }
    ok = false;
    return QSharedPointer<ito::DataObject>();
}

#if ITOM_POINTCLOUDLIBRARY > 0
//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::PCLPointCloud* PythonQtConversion::PyObjGetPointCloudNewPtr(PyObject *val, bool /*strict*/, bool &ok)
{
    if (Py_TYPE(val) == &ito::PythonPCL::PyPointCloudType)
    {
        ito::PythonPCL::PyPointCloud* pyPointCloud = (ito::PythonPCL::PyPointCloud*)val;
        if (pyPointCloud->data)
        {
            ok = true;
            return (new ito::PCLPointCloud(*(pyPointCloud->data)));
        }
        ok = false;
        return NULL;
    }
    else
    {
        ok = false;
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ ito::PCLPolygonMesh* PythonQtConversion::PyObjGetPolygonMeshNewPtr(PyObject *val, bool /*strict*/, bool &ok)
{
    if (Py_TYPE(val) == &ito::PythonPCL::PyPolygonMeshType)
    {
        ito::PythonPCL::PyPolygonMesh* pyMesh = (ito::PythonPCL::PyPolygonMesh*)val;
        if (pyMesh->polygonMesh)
        {
            ok = true;
            return (new ito::PCLPolygonMesh(*(pyMesh->polygonMesh)));
        }
        ok = false;
        return NULL;
    }
    else
    {
        ok = false;
        return NULL;
    }
}

#endif //#if ITOM_POINTCLOUDLIBRARY > 0

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from PyObject* to QVariant
/*!
    methods tries to convert PyObject* to QVariant. Type indicates the desired type of QVariant, given by the type-number of QMetaType.
    If type==-1, the right type is guessed by checking the PyObject*-type. If conversion failed, QVariant() is returned.

    \param val is the given python object
    \param type is the desired Qt-type (with respect to QMetaType) or -1, if resulting type should be guessed by input
    \return resulting QVariant or empty QVariant(), if conversion failed
*/
QVariant PythonQtConversion::PyObjToQVariant(PyObject* val, int type)
{
    QVariant v;
    bool ok = true;

    if (type==-1) 
    {
        //guess type by PyObject's type
        if (PyBytes_Check(val) || PyUnicode_Check(val))
        {
            type = QVariant::String;
        }
        else if (PyLong_Check(val))
        {
            int overflow = 0;
            long value = PyLong_AsLongAndOverflow(val, &overflow);
            if (overflow)
            {
                type = QVariant::LongLong;
            }
            else if (value < std::numeric_limits<int>::max() && value > std::numeric_limits<int>::min())
            {
                type = QVariant::Int;
            }
            else
            {
                type = QVariant::LongLong;
            }
        }
        else if (PyFloat_Check(val))
        {
            type = QVariant::Double;
        }
        else if (val == Py_False || val == Py_True)
        {
            type = QVariant::Bool;
        }
        else if (PyDict_Check(val))
        {
            type = QVariant::Map;
        }
        else if (PyList_Check(val) || PyTuple_Check(val) || PySequence_Check(val))
        {
            type = QVariant::List;
        }
        else if (val == Py_None)
        {
            // none is invalid
            type = QVariant::Invalid;
        }
        else if (Py_TYPE(val) == &ito::PythonRegion::PyRegionType)
        {
            type = QVariant::Region;
        }

#if ITOM_POINTCLOUDLIBRARY > 0
        else if (Py_TYPE(val) == &ito::PythonPCL::PyPointCloudType)
        {
            type = QMetaType::type("ito::PCLPointCloud");
        }
        else if (Py_TYPE(val) == &ito::PythonPCL::PyPointType)
        {
            type = QMetaType::type("ito::PCLPoint");
        }
        else if (Py_TYPE(val) == &ito::PythonPCL::PyPolygonMeshType)
        {
            type = QMetaType::type("ito::PCLPolygonMesh");
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
        else if (Py_TYPE(val) == &ito::PythonDataObject::PyDataObjectType)
        {
            type = QMetaType::type("QSharedPointer<ito::DataObject>");
        }
        else if (Py_TYPE(val) == &ito::PythonPlugins::PyActuatorPluginType)
        {
            type = QMetaType::type("QPointer<ito::AddInActuator>");
        }
        else if (Py_TYPE(val) == &ito::PythonPlugins::PyDataIOPluginType)
        {
            type = QMetaType::type("QPointer<ito::AddInDataIO>");
        }
    }

    // special type request:
    switch (type) 
    {
    case QVariant::Invalid:
        return v;
        break;
    case QVariant::Int:
    {
        int d = PyObjGetInt(val, false, ok);
        if (ok) return QVariant(d);
    }
    break;
    case QVariant::UInt:
    {
        int d = PyObjGetInt(val, false, ok);
        if (ok) v = QVariant((unsigned int)d);
    }
    break;
    case QVariant::Bool:
    {
        int d = PyObjGetBool(val, false, ok);
        if (ok) v =  QVariant((bool)(d!=0));
    }
    break;
    case QVariant::Double:
    {
        double d = PyObjGetDouble(val, false, ok);
        if (ok) v =  QVariant(d);
        break;
    }
    case QMetaType::Float:
    {
        float d = (float) PyObjGetDouble(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }
    case QMetaType::Long:
    {
        long d = (long) PyObjGetLongLong(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }
    case QMetaType::ULong:
    {
        unsigned long d = (unsigned long) PyObjGetLongLong(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }
    case QMetaType::LongLong:
    {
        qint64 d = PyObjGetLongLong(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
    }
    break;
    case QMetaType::ULongLong:
    {
        quint64 d = PyObjGetULongLong(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
    }
    break;
    case QMetaType::Short:
    {
        short d = (short) PyObjGetInt(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }
    case QMetaType::UShort:
    {
        unsigned short d = (unsigned short) PyObjGetInt(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }
    case QMetaType::Char:
    {
        char d = (char) PyObjGetInt(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }
    case QMetaType::UChar:
    {
        unsigned char d = (unsigned char) PyObjGetInt(val, false, ok);
        if (ok) v =  qVariantFromValue(d);
        break;
    }

    case QVariant::ByteArray:
    {
        QByteArray ba = PyObjGetBytes(val, true, ok);
        if (ok) v = qVariantFromValue(ba);
        break;
    }
       
    case QVariant::String:
    {
        bool ok;
        v = QVariant(PyObjGetString(val, false, ok));
    }
    break;

    case QVariant::Map:
    {
        if (PyMapping_Check(val)) 
        {
            QMap<QString,QVariant> map;
            PyObject* items = PyMapping_Items(val);
            if (items) 
            {
                int count = PyList_Size(items);
                PyObject* value;
                PyObject* key;
                PyObject* tuple;
                for (int i = 0;i<count;i++) 
                {
                    tuple = PyList_GetItem(items,i);
                    key = PyTuple_GetItem(tuple, 0);
                    value = PyTuple_GetItem(tuple, 1);
                    map.insert(PyObjGetString(key), PyObjToQVariant(value, -1));
                }
                Py_DECREF(items);
                v = map;
            }
        }
    }
    break;
    case QVariant::List:
    if (PySequence_Check(val)) 
    {
        QVariantList list;
        int count = PySequence_Size(val);
        PyObject* value = NULL;
        for (int i = 0;i<count;i++) 
        {
            value = PySequence_GetItem(val,i); //new reference
            list.append(PyObjToQVariant(value, -1));
            Py_XDECREF(value);
        }
        v = list;
    }
    break;
    case QVariant::StringList:
    {
        bool ok;
        QStringList l = PyObjToStringList(val, false, ok);
        if (ok) {
        v = l;
        }
    }
    break;

    case QVariant::Region:
    {
        ito::PythonRegion::PyRegion *pyReg = (ito::PythonRegion::PyRegion*)val;
        if (pyReg && pyReg->r)
        {
            v = *(pyReg->r);
        }
    }
    break;

    default:
    {
		if (type == QMetaType::type("QSharedPointer<ito::DataObject>"))
        {
            ito::PythonDataObject::PyDataObject *dataObj = (ito::PythonDataObject::PyDataObject*)val;
            if (dataObj)
            {
                if (dataObj->dataObject == NULL)
                {
                    v = QVariant();
                }
                else if (dataObj->base != NULL) //if the python-dataObject shares memory with other arrays (like a numpy array, we need to make a deep copy here, since we cannot increment the reference of 
                {
                    //QSharedPointer<ito::DataObject> value(new ito::DataObject());
                    //dataObj->dataObject->copyTo(*value);
                    //v = qVariantFromValue<QSharedPointer<ito::DataObject> >(value);

                    //baseObjectDeleter
                    Py_XINCREF(dataObj->base);
                    ito::DataObject *copy = new ito::DataObject(*dataObj->dataObject);
                    m_pyBaseObjectStorage.insert((void*)copy, dataObj->base); //unique
                    QSharedPointer<ito::DataObject> value(copy , baseObjectDeleterDataObject);
                    v = qVariantFromValue<QSharedPointer<ito::DataObject> >(value);
                }
                else
                {
                    QSharedPointer<ito::DataObject> value(new ito::DataObject(*dataObj->dataObject));
                    v = qVariantFromValue<QSharedPointer<ito::DataObject> >(value);
                }
            }
            else
            {
                v = QVariant();
            }
        }
#if ITOM_POINTCLOUDLIBRARY > 0
        else if (type == QMetaType::type("ito::PCLPointCloud"))
        {
			bool ok;
            ito::PCLPointCloud pcl = PyObjGetPointCloud(val, true, ok);
            if (ok)
            {
                v = qVariantFromValue<ito::PCLPointCloud >(pcl);
            }
        }
        else if (type == QMetaType::type("ito::PCLPoint"))
        {
			bool ok;
            ito::PCLPoint pcl = PyObjGetPoint(val, true, ok);
            if (ok)
            {
                v = qVariantFromValue<ito::PCLPoint>(pcl);
            }
        }
        else if (type == QMetaType::type("ito::PCLPolygonMesh"))
        {
			bool ok;
            ito::PCLPolygonMesh pcl = PyObjGetPolygonMesh(val, true, ok);
            if (ok)
            {
                v = qVariantFromValue<ito::PCLPolygonMesh >(pcl);
            }
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0        
        else if (type == QMetaType::type("QPointer<ito::AddInDataIO>"))
        {
            ito::PythonPlugins::PyDataIOPlugin *plugin = (ito::PythonPlugins::PyDataIOPlugin*)val;
            if (plugin)
            {
                v = qVariantFromValue<QPointer<ito::AddInDataIO> >(QPointer<ito::AddInDataIO>(plugin->dataIOObj));
            }
            else
            {
                v = QVariant();
            }
        }
        else if (type == QMetaType::type("QPointer<ito::AddInActuator>"))
        {
            ito::PythonPlugins::PyActuatorPlugin *plugin = (ito::PythonPlugins::PyActuatorPlugin*)val;
            if (plugin)
            {
                v = qVariantFromValue<QPointer<ito::AddInActuator> >(QPointer<ito::AddInActuator>(plugin->actuatorObj));
            }
            else
            {
                v = QVariant();
            }
        }
        else
        {
            v = QVariant();
        }
    }
    break;
        
    }
    return v;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QVariant PythonQtConversion::QVariantCast(const QVariant &item, QVariant::Type destType, ito::RetVal &retval)
{
	if (item.type() == destType)
	{
		retval += ito::retOk;
		return item;
	}

    bool ok = false;
	QVariant result;

    if (item.type() == QVariant::List)
    {
        if (destType == QVariant::PointF)
	    {
		    const QVariantList list = item.toList();
		    if (list.size() == 2)
		    {
                bool ok2;
			    result = QPointF(list[0].toFloat(&ok), list[1].toFloat(&ok2));
			    ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to PointF: at least one value could not be transformed to float.");
                }
		    }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to PointF: 2 values required.");
            }
	    }
	    else if (destType == QVariant::Point)
	    {
		    const QVariantList list = item.toList();
		    if (list.size() == 2)
		    {
			    bool ok2;
			    result = QPoint(list[0].toInt(&ok), list[1].toInt(&ok2));
			    ok &= ok2;
            
                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Point: at least one value could not be transformed to integer.");
                }
		    }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Point: 2 values required.");
            }
	    }
        else if (destType == QVariant::Rect)
        {
            const QVariantList list = item.toList();
		    if (list.size() == 4)
		    {
			    bool ok2, ok3, ok4;
			    result = QRect(list[0].toInt(&ok), list[1].toInt(&ok2), list[2].toInt(&ok3), list[3].toInt(&ok4));
			    ok &= ok2;
                ok &= ok3;
                ok &= ok4;
            
                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Rect: at least one value could not be transformed to integer.");
                }
		    }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Rect: 4 values required.");
            }
        }
        else if (destType == QVariant::RectF)
        {
            const QVariantList list = item.toList();
		    if (list.size() == 4)
		    {
			    bool ok2, ok3, ok4;
			    result = QRectF(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3), list[3].toFloat(&ok4));
			    ok &= ok2;
                ok &= ok3;
                ok &= ok4;
            
                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to RectF: at least one value could not be transformed to float.");
                }
		    }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to RectF: 4 values required.");
            }
        }
        else if (destType == QVariant::Size)
        {
            const QVariantList list = item.toList();
            if (list.size() == 2)
            {
                bool ok2;
                result = QSize(list[0].toInt(&ok), list[1].toInt(&ok2));
                ok &= ok2;
            
                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Size: at least one value could not be transformed to integer.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Size: 2 values required.");
            }
        }
    } //end item.type() == QVariant::List
    

    if (!ok && !retval.containsError()) //not yet converted, try to convert it using QVariant internal conversion method
    {
        if (item.canConvert(destType))
        {
            result = item;
            result.convert(destType);
            ok = true;
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, "no conversion from QVariant type %i to %i is possible", item.type(), destType);
        }
    }

	if (ok)
	{
		return result;
	}
	return item;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QVariant PythonQtConversion::QVariantToEnumCast(const QVariant &item, const QMetaEnum &enumerator, ito::RetVal &retval)
{
    int val;
    bool ok;
    val = item.toInt(&ok);
    QVariant result;

    if (ok) //integer
    {
        const char *key = enumerator.valueToKey(val);
        if (key)
        {
            result = val;
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, "The value %i does not exist in the enumeration %s::%s", val,enumerator.scope(), enumerator.name());
            return result;
        }
    }
    else //
    {
        QString str = item.toString();
        if (str.isEmpty() == false) //string
        {
            val = enumerator.keyToValue(str.toAscii().data());
            if (val >= 0)
            {
                result = val;
            }
            else
            {
                retval += ito::RetVal::format(ito::retError, 0, "The key %s does not exist in the enumeration %s::%s",str.toAscii().data(), enumerator.scope(), enumerator.name());
                return result;
            }
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, "Use an integer or a string to for a value of the enumeration %s::%s", enumerator.scope(), enumerator.name());
            return result;
        }
    }

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! tries to convert PyObject* to known data type and returns deep-copy of the value, given as void*
/*!
    methods tries to convert PyObject* to QVariant. Type indicates the desired type of QVariant, given by the type-number of QMetaType.
    If type==-1, the right type is guessed by checking the PyObject*-type. If conversion failed, QVariant() is returned.

    The deep-copy is created using QMetaType::construct(...)

    \param val is the given python object
    \param retPtr is the resulting pointer to type void*, containing the converted deep copy or NULL, if error
    \param retType is an integer-pointer, containing the type-id of the converted result (with respect to QMetaType) or -1, if failure
    \param type is the desired QMetaType-type-id or -1, if conversion-type should be guessed considering the type of val.
    \param strict decides whether the given PyObject should strictly be converted into the C-type or a cast like int->string is allowed, too.
    \return true if conversion succeeded or false otherwise
    \see PyObjToQVariant
*/
bool PythonQtConversion::PyObjToVoidPtr(PyObject* val, void **retPtr, int *retType, int type /*= -1*/, bool strict /*= false*/)
{
    QVariant v;
    bool ok = true;
    bool qvariant = false;

    //check for variant type
    if (type == QMetaType::QVariant)
    {
        qvariant = true;
        type = -1; //let guess the type and finally transform to QVariant
    }

    if (type==-1) 
    {
        //guess type by PyObject's type
        if (PyBytes_Check(val) || PyUnicode_Check(val))
        {
            type = QMetaType::QString;
        }
        else if (PyLong_Check(val))
        {
            int overflow = 0;
            long value = PyLong_AsLongAndOverflow(val, &overflow);
            if (overflow)
            {
                type = QMetaType::LongLong;
            }
            else if (value < std::numeric_limits<int>::max() && value > std::numeric_limits<int>::min())
            {
                type = QMetaType::Int;
            }
            else
            {
                type = QMetaType::LongLong;
            }
        }
        else if (PyFloat_Check(val))
        {
            type = QVariant::Double;
        }
        else if (val == Py_False || val == Py_True)
        {
            type = QMetaType::Bool;
        }
        else if (PyDict_Check(val))
        {
            type = QMetaType::QVariantMap;
        }
        else if (PyList_Check(val) || PyTuple_Check(val) || PySequence_Check(val))
        {
            type = QMetaType::QVariantList;
        }
        else if (val == Py_None)
        {
            // none is invalid
            type = QMetaType::Void;
        }
        else if (PyUiItem_Check(val))
        {
            type = QMetaType::type("ito::PythonQObjectMarshal");
        }
        else if (PyRegion_Check(val))
        {
            type = QMetaType::QRegion;
        }
        else if (PyRgba_Check(val))
        {
            type = QMetaType::QColor;
        }
    }

    if (QMetaType::isRegistered(type))
    {
        // special type request:
        switch (type) 
        {
        case QMetaType::Void:
            *retPtr = QMetaType::construct(type, NULL);
            break;
        case QMetaType::Int:
        {
            int d = PyObjGetInt(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::UInt:
        {
            unsigned int d = (unsigned int)PyObjGetInt(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::Bool:
        {
            int d = PyObjGetBool(val, strict, ok);
            if (ok)
            {
                bool d2 = (d != 0);
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d2));
            }
            break;
        }
        case QMetaType::Double:
        {
            double d = PyObjGetDouble(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }  
            break;
        }
        case QMetaType::Float:
        {
            float d = (float) PyObjGetDouble(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::Long:
        {
            long d = (long) PyObjGetLongLong(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::ULong:
        {
            unsigned long d = (unsigned long) PyObjGetLongLong(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::LongLong:
        {
            qint64 d = (qint64) PyObjGetLongLong(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::ULongLong:
        {
            quint64 d = (quint64) PyObjGetULongLong(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::Short:
        {
            short d = (short) PyObjGetInt(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::UShort:
        {
            unsigned short d = (unsigned short) PyObjGetInt(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::Char:
        {
            char d = (char) PyObjGetInt(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }
        case QMetaType::UChar:
        {
            unsigned char d = (unsigned char) PyObjGetInt(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&d));
            }
            break;
        }

        case QMetaType::QByteArray:
        {
            QByteArray text = PyObjGetBytes(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&text));
            }
            break;
        }
        case QMetaType::QString:
        {
            QString text = PyObjGetString(val, strict, ok);
            if (ok)
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&text));
            }
            break;
        }
        case QMetaType::QUrl:
        {
            QString text = PyObjGetString(val, strict, ok);
            if (ok)
            {
                QUrl url = QUrl(text);
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&url));
            }
            break;
        }
        case QMetaType::QVariantMap:
        {
            if (PyMapping_Check(val)) 
            {
                QVariantMap map;
                PyObject* items = PyMapping_Items(val);
                if (items) 
                {
                    int count = PyList_Size(items);
                    PyObject* value;
                    PyObject* key;
                    PyObject* tuple;
                    for (int i = 0; i < count; i++) 
                    {
                        tuple = PyList_GetItem(items,i);
                        key = PyTuple_GetItem(tuple, 0);
                        value = PyTuple_GetItem(tuple, 1);
                        map.insert(PyObjGetString(key), PyObjToQVariant(value, -1));
                    }
                    Py_DECREF(items);
                }
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&map));
            }
            break;
        }
        
        case QMetaType::QVariantList:
        if (PySequence_Check(val)) 
        {
            QVariantList list;
            int count = PySequence_Size(val);
            PyObject* value = NULL;
            for (int i = 0; i < count; i++) 
            {
                value = PySequence_GetItem(val, i); //new reference
                list.append(PyObjToQVariant(value, -1));
                Py_XDECREF(value);
            }
            *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&list));
            break;
        }
        
        case QMetaType::QStringList:
        {
            bool ok;
            QStringList l = PyObjToStringList(val, strict, ok);
            if (ok) 
            {
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&l));
            }
            break;
        }

        case QMetaType::QRegion:
        {
            ito::PythonRegion::PyRegion *reg = (ito::PythonRegion::PyRegion*)val;
            if (reg && reg->r)
            {
                QRegion r = *(reg->r);
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&r));
            }
            break;
        }

        case QMetaType::QColor:
        {
            if (PyRgba_Check(val))
            {
                ito::PythonRgba::PyRgba *rgba = (ito::PythonRgba::PyRgba*)val;
                if (rgba)
                {
                    //QColor c(rgba->rgba.red(), rgba->rgba.green(), rgba->rgba.blue(), rgba->rgba.alpha());
                    QColor c(rgba->rgba.argb());
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&c));
                }
            }
            else
            {
                QString text = PyObjGetString(val, strict, ok);
                if (ok)
                {
                    QColor c(text);
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&c));
                }
            }
            break;
        }

        default:
        //check user defined types
        {
			if (type == QMetaType::type("ito::PythonQObjectMarshal"))
            {
                ito::PythonUi::PyUiItem *val2 = (ito::PythonUi::PyUiItem*)val;
                
                ito::PythonQObjectMarshal m = ito::PythonQObjectMarshal();
                m.m_objectID = val2->objectID;
                m.m_object = NULL;
                m.m_objName = val2->objName;
                m.m_className = val2->widgetClassName;
                *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&m));
            }
            else if (type == QMetaType::type("QSharedPointer<ito::DataObject>"))
            {
                ito::PythonDataObject::PyDataObject *val2 = (ito::PythonDataObject::PyDataObject*)val;
                if (val2 && val2->dataObject)
                {
                    QSharedPointer<ito::DataObject> sharedBuffer = ito::PythonSharedPointerGuard::createPythonSharedPointer<ito::DataObject>(val2->dataObject, val);
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&sharedBuffer));
                }
            }
			else if (type == QMetaType::type("ito::DataObject"))
			{
				ito::PythonDataObject::PyDataObject *val2 = (ito::PythonDataObject::PyDataObject*)val;
                if (val2 && val2->dataObject)
                {
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(val2->dataObject));
                }
			}
#if ITOM_POINTCLOUDLIBRARY > 0
            else if (type == QMetaType::type("ito::PCLPointCloud"))
            {
				bool ok;
                ito::PCLPointCloud pcl = PyObjGetPointCloud(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&pcl));
                }
            }
            else if (type == QMetaType::type("ito::PCLPoint"))
            {
				bool ok;
                ito::PCLPoint pt = PyObjGetPoint(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&pt));
                }
            }
            else if (type == QMetaType::type("ito::PCLPolygonMesh"))
            {
				bool ok;
                ito::PCLPolygonMesh mesh = PyObjGetPolygonMesh(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&mesh));
                }
            }
            else if (type == QMetaType::type("QVector<double>"))
            {
                bool ok;
                QVector<double> arr = PyObjGetDoubleArray(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&arr));
                }
            }
            else if (type == QMetaType::type("QVector<int>"))
            {
                bool ok;
                QVector<int> arr = PyObjGetIntArray(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::construct(type, reinterpret_cast<void*>(&arr));
                }
            }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
            else
            {
                *retPtr = NULL;
            }
            break;
        }
            
        } //end switch case

        //now check if qvariant is true, then retPtr and retType needs to be retransformed to QVariant
        if (qvariant && *retPtr)
        {
            void *ptrToOriginalValue = *retPtr;

            QVariant *variantValue = new QVariant(type, *retPtr);
            if (variantValue->isValid())
            {
                *retType = QMetaType::QVariant;
                *retPtr = (void*)variantValue; //here no QMetaType::construct is used since construct does not more than a copy constructor of QVariant casted to void*
            }
            else
            {
                *retType = -1;
                *retPtr = NULL;
                delete variantValue;
                variantValue = NULL;
            }

            QMetaType::destroy(type, ptrToOriginalValue); //type is type-number of original value
        }
        else if (*retPtr)
        {
            *retType = type;
        }
        else
        {
            *retType = -1;
        }
    }
    else
    {
        *retPtr = NULL;
        *retType = -1;
    }

    
    return (*retPtr != NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from bool to python boolean type
/*!
    Returns new reference to Py_True or Py_False depending on input val.

    \param val is the given boolean input
    \return resulting PyObject* (new reference)
*/
PyObject* PythonQtConversion::GetPyBool(bool val)
{
    PyObject* r = val?Py_True:Py_False;
    Py_INCREF(r);
    return r;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from given QString to PyObject*
/*!
    returns new reference to Python-unicode object, containing the content of given QString. If str is empty or null, 
    the resulting string is empty string ("")

    \param str is reference to QString
    \return is the resulting PyUnicode-Object (new reference)
*/
PyObject* PythonQtConversion::QStringToPyObject(const QString& str)
{
    if (str.isNull()) 
    {
        return ByteArrayToPyUnicode("",0);
    } 
    else 
    {
        return QByteArrayToPyUnicode(str.toAscii()); //str.toAscii() decodes with current encoding and then it is transformed to PyObject with the same encoding
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from given QStringList to tuple of python-unicode objects
/*!
    returns new reference to tuple of Python-unicode objects, containing the content of given QStringList. Every single string is converted
    using \a QStringToPyObject.

    \param list is reference to QStringList
    \return is the resulting PyObject-tuple (new reference)
*/
PyObject* PythonQtConversion::QStringListToPyObject(const QStringList& list)
{
    PyObject* result = PyTuple_New(list.count());
    int i = 0;
    QString str;
    foreach (str, list) 
    {
        PyTuple_SET_ITEM(result, i, PythonQtConversion::QStringToPyObject(str));
        i++;
    }
    // why is the error state bad after this?
    PyErr_Clear();
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from given QStringList to list of python-unicode objects
/*!
    returns new reference to list of Python-unicode objects, containing the content of given QStringList. Every single string is converted
    using \a QStringToPyObject.

    \param list is reference to QStringList
    \return is the resulting PyObject-list (new reference)
*/
PyObject* PythonQtConversion::QStringListToPyList(const QStringList& list)
{
    PyObject* result = PyList_New(list.count());
    int i = 0;
    for (QStringList::ConstIterator it = list.begin(); it != list.end(); ++it) 
    {
        PyList_SET_ITEM(result, i, PythonQtConversion::QStringToPyObject(*it));
        i++;
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from given QVariant to appropriate PyObject*
/*!
    returns new reference to PyObject*, which contains the conversion from given QVariant-variable.

    \param v is reference to QVariant
    \return is the resulting PyObject*
    \see ConvertQtValueToPytonInternal
*/
PyObject* PythonQtConversion::QVariantToPyObject(const QVariant& v)
{
    return ConvertQtValueToPythonInternal(v.userType(), (void*)v.constData());
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from given QVariantMap to python-dictionary.
/*!
    returns new reference to python-dictionary type. Each key of QVariantMap is one element in dictionary. The values are converted
    using \a QVariantToPyObject.

    \param m is reference to QVariantMap
    \return is the resulting PyObject*
    \see QVariantToPyObject
*/
PyObject* PythonQtConversion::QVariantMapToPyObject(const QVariantMap& m) 
{
    PyObject* result = PyDict_New();
    QVariantMap::const_iterator t = m.constBegin();
    PyObject* key;
    PyObject* val;
    for (; t != m.constEnd(); t++) 
    {
        key = PythonQtConversion::QStringToPyObject(t.key());
        val = PythonQtConversion::QVariantToPyObject(t.value());
        PyDict_SetItem(result, key, val);
        Py_DECREF(key);
        Py_DECREF(val);
    }
    return result;
}

#if ITOM_POINTCLOUDLIBRARY > 0
//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonQtConversion::PCLPointCloudToPyObject(const ito::PCLPointCloud& c)
{
    PyObject *args = Py_BuildValue("(i)", c.getType());
    ito::PythonPCL::PyPointCloud *result = (ito::PythonPCL::PyPointCloud*)PyObject_Call((PyObject*)&(ito::PythonPCL::PyPointCloudType), args, NULL);
    Py_DECREF(args);
    if (result)
    {
        *result->data = c;
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_RuntimeError, "could not create instance of pclPointCloud");
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonQtConversion::PCLPointToPyObject(const ito::PCLPoint& c)
{
    ito::PythonPCL::PyPoint *result = (ito::PythonPCL::PyPoint*)PyObject_Call((PyObject*)&(ito::PythonPCL::PyPointType), NULL, NULL);
    if (result)
    {
        if (result->point)
        {
            delete result->point;
            result->point = NULL;
        }
        *result->point = c;
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_RuntimeError, "could not create instance of pclPoint");
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonQtConversion::PCLPolygonMeshToPyObject(const ito::PCLPolygonMesh& c)
{
    ito::PythonPCL::PyPolygonMesh *result = (ito::PythonPCL::PyPolygonMesh*)PyObject_Call((PyObject*)&(ito::PythonPCL::PyPolygonMeshType), NULL, NULL);
    if (result)
    {
        if (result->polygonMesh)
        {
            delete result->polygonMesh;
            result->polygonMesh = NULL;
        }
        result->polygonMesh = new ito::PCLPolygonMesh(c);
        //*result->polygonMesh = c;
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_RuntimeError, "could not create instance of pclPolygonMesh");
    return NULL;
}

#endif //#if ITOM_POINTCLOUDLIBRARY > 0

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonQtConversion::DataObjectToPyObject(const ito::DataObject& dObj)
{
    ito::PythonDataObject::PyDataObject *pyDataObj = ito::PythonDataObject::createEmptyPyDataObject();
    if (pyDataObj)
    {
        pyDataObj->dataObject = new ito::DataObject(dObj);
        return (PyObject*)pyDataObj;
    }

    PyErr_SetString(PyExc_RuntimeError, "could not create instance of dataObject");
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! conversion from given QVariantList to python-tuple.
/*!
    returns new reference to python-tuple type. Each item of QVariantList is one element in tuple. The values are converted
    using \a QVariantToPyObject.

    \param m is reference to QVariantMap
    \return is the resulting PyObject*
    \see QVariantToPyObject
*/
PyObject* PythonQtConversion::QVariantListToPyObject(const QVariantList& l) 
{
    PyObject* result = PyTuple_New(l.count());
    int i = 0;
    QVariant v;
    foreach (v, l) 
    {
        PyTuple_SET_ITEM(result, i, PythonQtConversion::QVariantToPyObject(v));
        i++;
    }
    // why is the error state bad after this?
    PyErr_Clear();
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method internally used for conversion from given type-id (QMetaType) and corresponding void*-pointer to PyObject*
/*!
    This method is the opposite from \a PyObjToVoidPtr and converts a pair given by type-id (see QMetaType) and corresponding void*-pointer,
    which to the variable's content to the appropriate python type.

    A python error is returned if conversion failed.

    \param type is given type-id (\a QMetaType)
    \param data is the content, casted to void*
    \return is the resulting PyObject* (new reference)
*/
PyObject* PythonQtConversion::ConvertQtValueToPythonInternal(int type, const void* data) 
{
    switch (type) 
    {
    case QMetaType::Void:
        Py_INCREF(Py_None);
        return Py_None;
    case QMetaType::Char:
        return PyLong_FromLong(*((char*)data));
    case QMetaType::UChar:
        return PyLong_FromLong(*((unsigned char*)data));
    case QMetaType::Short:
        return PyLong_FromLong(*((short*)data));
    case QMetaType::UShort:
        return PyLong_FromLong(*((unsigned short*)data));
    case QMetaType::Long:
        return PyLong_FromLong(*((long*)data));
    case QMetaType::ULong:
    // does not fit into simple int of python
        return PyLong_FromUnsignedLong(*((unsigned long*)data));
    case QMetaType::Bool:
        return PythonQtConversion::GetPyBool(*((bool*)data));
    case QMetaType::Int:
        return PyLong_FromLong(*((int*)data));
    case QMetaType::UInt:
    // does not fit into simple int of python
        return PyLong_FromUnsignedLong(*((unsigned int*)data));
    case QMetaType::QChar:
        return PyLong_FromLong(*((short*)data));
    case QMetaType::Float:
        return PyFloat_FromDouble(*((float*)data));
    case QMetaType::Double:
        return PyFloat_FromDouble(*((double*)data));
    case QMetaType::LongLong:
        return PyLong_FromLongLong(*((qint64*)data));
    case QMetaType::ULongLong:
        return PyLong_FromUnsignedLongLong(*((quint64*)data));
    case QMetaType::QByteArray: 
    {
        QByteArray* v = (QByteArray*) data;
        return PyBytes_FromStringAndSize(*v, v->size());
    }
    case QMetaType::QVariantMap:
        return PythonQtConversion::QVariantMapToPyObject(*((QVariantMap*)data));
    case QMetaType::QVariantList:
        return PythonQtConversion::QVariantListToPyObject(*((QVariantList*)data));
    case QMetaType::QString:
        return PythonQtConversion::QStringToPyObject(*((QString*)data));
    case QMetaType::QStringList:
        return PythonQtConversion::QStringListToPyObject(*((QStringList*)data));
    case QMetaType::QSize:
        {
            PyObject *temp = PyList_New(2);
            QSize *temp2 = (QSize*)data;
            PyList_SetItem(temp, 0, PyLong_FromLong(temp2->width()));
            PyList_SetItem(temp, 1, PyLong_FromLong(temp2->height()));
            return temp;
        }
    case QMetaType::QRect:
        {
            PyObject *temp = PyList_New(4);
            QRect *temp2 = (QRect*)data;
            PyList_SetItem(temp, 0, PyLong_FromLong(temp2->x()));
            PyList_SetItem(temp, 1, PyLong_FromLong(temp2->y()));
            PyList_SetItem(temp, 2, PyLong_FromLong(temp2->width()));
            PyList_SetItem(temp, 3, PyLong_FromLong(temp2->height()));
            return temp;
        }
	case QMetaType::QPointF:
		{
			PyObject *temp = PyList_New(2);
            QPointF *temp2 = (QPointF*)data;
            PyList_SetItem(temp, 0, PyFloat_FromDouble(temp2->x()));
            PyList_SetItem(temp, 1, PyFloat_FromDouble(temp2->y()));
            return temp;
		}
	case QMetaType::QPoint:
		{
			PyObject *temp = PyList_New(2);
            QPoint *temp2 = (QPoint*)data;
            PyList_SetItem(temp, 0, PyLong_FromLong(temp2->x()));
            PyList_SetItem(temp, 1, PyLong_FromLong(temp2->y()));
            return temp;
		}
    case QMetaType::QRegion:
        {
            return ito::PythonRegion::createPyRegion(*((QRegion*)data));
        }
    case QMetaType::QColor:
        {
            ito::PythonRgba::PyRgba *rgba = ito::PythonRgba::createEmptyPyRgba();
            QColor* color = (QColor*)data;
            if (rgba)
            {
                rgba->rgba.argb() = (ito::uint32)color->value();
                //rgba->b = color->blue();
                //rgba->g = color->green();
                //rgba->a = color->alpha();
            }
            return (PyObject*)rgba;
        }
    case QMetaType::QVariant:
        {
            QVariant temp = *(QVariant*)data;
            return PythonQtConversion::QVariantToPyObject(temp);
        }
    }

    //until now, type did not fit
    if (QMetaType::isRegistered(type))
    {
        const char *name = QMetaType::typeName(type);
        if (strcmp(name, "ito::RetVal") == 0)
        {
            ito::RetVal *v = (ito::RetVal*)data;
            if (ito::PythonCommon::transformRetValToPyException(*v))
            {
                Py_RETURN_NONE;
            }
            return NULL;
        }
		if (strcmp(name, "ito::DataObject") == 0)
        {
            return DataObjectToPyObject(*((ito::DataObject*)data));
        }
#if ITOM_POINTCLOUDLIBRARY > 0
		else if (strcmp(name, "ito::PCLPointCloud") == 0)
        {
            return PCLPointCloudToPyObject(*((ito::PCLPointCloud*)data));
        }
        else if (strcmp(name, "ito::PCLPoint") == 0)
        {
            return PCLPointToPyObject(*((ito::PCLPoint*)data));
        }
        
        else if (strcmp(name, "ito::PCLPolygonMesh") == 0)
        {
            return PCLPolygonMeshToPyObject(*((ito::PCLPolygonMesh*)data));
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
        else if (strcmp(name, "QSharedPointer<ito::DataObject>") == 0)
        {
            QSharedPointer<ito::DataObject> *sharedPtr = (QSharedPointer<ito::DataObject>*)data;
            if (sharedPtr == NULL)
            {
                return PyErr_Format(PyExc_TypeError, "The given QSharedPointer is NULL");
            }
            if (sharedPtr->data() == NULL)
            {
                Py_RETURN_NONE;
                //return PyErr_Format(PyExc_TypeError, "Internal dataObject of QSharedPointer is NULL");
            }
            return DataObjectToPyObject(*(sharedPtr->data()));
        }
    }
    else
    {
        return PyErr_Format(PyExc_TypeError, "The given Qt-type is not registered in the Qt-MetaType system.");
    }

    return PyErr_Format(PyExc_TypeError, "The given Qt-type can not be parsed into an appropriate python type.");
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonQtConversion::convertPyObjectToQVariant(PyObject *argument, QVariant &qVarArg)
{
    if (PyList_Check(argument))
    {
        PyObject* tempArg = NULL;
        PyObject* retValue = NULL;
        QVariantList list;
        for (Py_ssize_t i = 0; i < PyList_Size(argument); i++)
        {
            tempArg = PyList_GetItem(argument, i);
            list.append(QVariant());
            retValue = convertPyObjectToQVariant(tempArg, list[i]);

            if (PyErr_Occurred())
            {
                return NULL;
            }
            if (retValue != NULL) Py_DECREF(retValue);
        }

        qVarArg = list;

        Py_RETURN_NONE;
    }

    //check for elementary types char*, int, double
    char* textArg;
    if (PyLong_Check(argument))
    {
        qVarArg = (int)PyLong_AsLong(argument); //overflow error is checked here and returned as error.
    }
    else if (PyFloat_Check(argument))
    {
        qVarArg = PyFloat_AsDouble(argument);
    }
    else if (PyArg_Parse(argument, "s", &textArg))
    {
        qVarArg = QString(textArg);
    }
    else if ((Py_TYPE(argument) == &ito::PythonDataObject::PyDataObjectType))
    {
        ito::PythonDataObject::PyDataObject *dataObj = (ito::PythonDataObject::PyDataObject*)argument;
        if (dataObj)
        {
            if (dataObj->dataObject == NULL)
            {
                PyErr_Format(PyExc_ValueError, "internal dataObject is NULL");
            }
            else
            {
                QSharedPointer<ito::DataObject> value(new ito::DataObject(*dataObj->dataObject));
                qVarArg = QVariant::fromValue(value);
                if (qVarArg.isNull())
                {
                    PyErr_Format(PyExc_ValueError, "dataObject could not be converted to QVariant (QSharedPointer<ito::DataObject>)");
                }
            }
        }
        else
        {
            PyErr_Format(PyExc_ValueError, "Cannot cast to python dataObject instance");
        }
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "argument does not fit to char*, int, long or double");
        qVarArg = QVariant();
    }

    if (PyErr_Occurred())
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::QByteArrayToPyUnicode(const QByteArray &ba, const char *errors)
{
    return ByteArrayToPyUnicode(ba.data(), ba.length(), errors);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::QByteArrayToPyUnicodeSecure(const QByteArray &ba, const char *errors /*= "replace"*/)
{
    PyObject *temp = ByteArrayToPyUnicode(ba.data(), ba.length(), errors);
    if (temp == NULL)
    {
        PyErr_Clear();
        temp = ByteArrayToPyUnicode("<encoding error>");
    }
    return temp;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::ByteArrayToPyUnicode(const char* byteArray, Py_ssize_t len, const char *errors)
{
    int bo;

    if (len <= 0)
    {
        len = strlen(byteArray);
    }

    switch(textEncoding)
    {
    case utf_8:
        return PyUnicode_DecodeUTF8(byteArray,len,errors);
    case latin_1:
    case iso_8859_1:
        return PyUnicode_DecodeLatin1(byteArray,len,errors);
#if defined(WIN) || defined(WIN32) || defined(_WIN64) || defined(_WINDOWS)
    case mbcs:
        return PyUnicode_DecodeMBCS(byteArray, len, errors);
#endif
    case ascii:
            return PyUnicode_DecodeASCII(byteArray, len, errors);
    case utf_16:
            return PyUnicode_DecodeUTF16(byteArray, len, errors, 0);
    case utf_16_LE:
            bo = -1;
            return PyUnicode_DecodeUTF16(byteArray, len, errors, &bo);
    case utf_16_BE:
            bo = 1;
            return PyUnicode_DecodeUTF16(byteArray, len, errors, &bo);
    case utf_32:
            return PyUnicode_DecodeUTF32(byteArray, len, errors, 0);
    case utf_32_LE:
            bo = -1;
            return PyUnicode_DecodeUTF32(byteArray, len, errors, &bo);
    case utf_32_BE:
            bo = 1;
            return PyUnicode_DecodeUTF32(byteArray, len, errors, &bo);
    case other:
    default:
        {
            PyObject *res = PyUnicode_Decode(byteArray, len, textEncodingName.data(), errors);
            return res;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::PyUnicodeToPyByteObject(PyObject *unicode, const char *errors /*= "replace"*/)
{
    int bo;

    if (!PyUnicode_Check(unicode)) 
    {
        PyErr_BadArgument();
        return NULL;
    }

    switch(textEncoding)
    {
    case utf_8:
        return PyUnicode_AsUTF8String(unicode);
    case latin_1:
    case iso_8859_1:
        return PyUnicode_AsLatin1String(unicode);
#if defined(WIN) || defined(WIN32) || defined(_WIN64) || defined(_WINDOWS)
    case mbcs:
        return PyUnicode_AsMBCSString(unicode);
#endif
    case ascii:
            return PyUnicode_AsASCIIString(unicode);
    case utf_16:
            return PyUnicode_EncodeUTF16(PyUnicode_AS_UNICODE(unicode), PyUnicode_GET_SIZE(unicode), errors, 0);
    case utf_16_LE:
            bo = -1;
            return PyUnicode_EncodeUTF16(PyUnicode_AS_UNICODE(unicode), PyUnicode_GET_SIZE(unicode), errors, bo);
    case utf_16_BE:
            bo = 1;
            return PyUnicode_EncodeUTF16(PyUnicode_AS_UNICODE(unicode), PyUnicode_GET_SIZE(unicode), errors, bo);
    case utf_32:
            return PyUnicode_EncodeUTF32(PyUnicode_AS_UNICODE(unicode), PyUnicode_GET_SIZE(unicode), errors, 0);
    case utf_32_LE:
            bo = -1;
            return PyUnicode_EncodeUTF32(PyUnicode_AS_UNICODE(unicode), PyUnicode_GET_SIZE(unicode), errors, bo);
    case utf_32_BE:
            bo = 1;
            return PyUnicode_EncodeUTF32(PyUnicode_AS_UNICODE(unicode), PyUnicode_GET_SIZE(unicode), errors, bo);
    case other:
    default:
        {
            PyObject *res = PyUnicode_AsEncodedString(unicode, textEncodingName.data(), errors);
            return res;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class MethodDescription
    \brief Small wrapper class with all necessary information for any method, signal or slot of class which should be
        inherited from QObject*. 
*/

MethodDescription::MethodDescription() :
    m_methodIndex(-1),
    m_type(QMetaMethod::Method),
    m_access(QMetaMethod::Public),
    m_retType(-1),
    m_nrOfArgs(0),
    m_argTypes(NULL)
{
}

MethodDescription::MethodDescription(QByteArray &name, QByteArray &signature, QMetaMethod::MethodType type, QMetaMethod::Access access, int methodIndex, int retType, int nrOfArgs, int *argTypes) : 
    m_name(name),
    m_methodIndex(methodIndex),
    m_signature(signature),
    m_type(type),
    m_access(access),
    m_retType(retType),
    m_nrOfArgs(nrOfArgs),
    m_argTypes(NULL)
{
    m_argTypes = new int[nrOfArgs];
    memcpy(m_argTypes, argTypes, nrOfArgs * sizeof(int));
}

MethodDescription::MethodDescription(QMetaMethod &method) : 
    m_type(QMetaMethod::Method),
    m_access(QMetaMethod::Public),
    m_retType(-1),
    m_nrOfArgs(0),
    m_argTypes(NULL)
{
    m_methodIndex = method.methodIndex();
    m_signature = QByteArray(method.signature());
    QByteArray sig(method.signature());
    int beginArgs = sig.indexOf("(");
    m_name = sig.left(beginArgs);
    m_type = method.methodType();
    m_access = method.access();
    m_retType = QMetaType::type(method.typeName());
    QList<QByteArray> types = method.parameterTypes();
    m_nrOfArgs = types.size();
    m_argTypes = new int[m_nrOfArgs];

    for (int i = 0; i < m_nrOfArgs; i++)
    {
        m_argTypes[i] = QMetaType::type(types[i].data());
    } 
}

MethodDescription::MethodDescription(const MethodDescription &copy) : 
    m_name(copy.m_name),
    m_methodIndex(copy.m_methodIndex),
    m_signature(copy.m_signature),
    m_type(copy.m_type),
    m_access(copy.m_access),
    m_retType(copy.m_retType),
    m_nrOfArgs(copy.m_nrOfArgs),
    m_argTypes(NULL)
{
    DELETE_AND_SET_NULL_ARRAY(m_argTypes);
    m_argTypes = new int[m_nrOfArgs];
    memcpy(m_argTypes, copy.m_argTypes, m_nrOfArgs * sizeof(int));
}

MethodDescription & MethodDescription::operator = (const MethodDescription &other)
{
    DELETE_AND_SET_NULL_ARRAY(m_argTypes);
    m_type = other.m_type;
    m_access = other.m_access;
    m_methodIndex = other.m_methodIndex;
    m_signature = other.m_signature;
    m_name = other.m_name;
    m_retType = other.m_retType;
    m_nrOfArgs = other.m_nrOfArgs;
    m_argTypes = new int[m_nrOfArgs];
    memcpy(m_argTypes, other.m_argTypes, m_nrOfArgs * sizeof(int));
    return *this;
}

MethodDescription::~MethodDescription()
{
    DELETE_AND_SET_NULL_ARRAY(m_argTypes);
}
