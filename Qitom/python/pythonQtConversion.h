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

#ifndef PYTHONQTCONVERSION_H
#define PYTHONQTCONVERSION_H

//python
// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
#ifdef _DEBUG
    #undef _DEBUG
    #if (defined linux) | (defined CMAKE)
        #include "Python.h"
    #else
        #include "Python.h"
    #endif
    #define _DEBUG
#else
#ifdef linux
    #include "Python.h"
#else
    #include "Python.h"
#endif
#endif

#include "pythonDataObject.h"
#include "pythonPlugins.h"
#include "pythonRegion.h"

#include "pythonPCL.h"

#include "../global.h"

#include "opencv/cv.h"
#include <qvariant.h>
#include <qmetaobject.h>
#include <qlist.h>
#include <qbytearray.h>
#include <qpointer.h>

#if ITOM_POINTCLOUDLIBRARY > 0
Q_DECLARE_METATYPE(ito::PCLPointCloud)
Q_DECLARE_METATYPE(ito::PCLPoint)
Q_DECLARE_METATYPE(ito::PCLPolygonMesh)
#endif

Q_DECLARE_METATYPE(QSharedPointer<ito::DataObject>)
Q_DECLARE_METATYPE(QPointer<ito::AddInDataIO>)
Q_DECLARE_METATYPE(QPointer<ito::AddInActuator>)

class PythonQtConversion
{
public:
    enum unicodeEncodings { utf_8, latin_1, iso_8859_1, mbcs, ascii, utf_16, utf_16_LE, utf_16_BE, utf_32, utf_32_BE, utf_32_LE, other };

    //! converts QString to Python string (unicode!)
    static PyObject* QStringToPyObject(const QString& str);

    //! converts QStringList to Python tuple
    static PyObject* QStringListToPyObject(const QStringList& list);

    //! converts QStringList to Python list
    static PyObject* QStringListToPyList(const QStringList& list);

    //! get string representation of py object
    static QString PyObjGetRepresentation(PyObject* val);

    //! get string value from py object
    static QString PyObjGetString(PyObject* val) { bool ok; QString s = PyObjGetString(val, false, ok); return s; }
    //! get string value from py object
    static QString PyObjGetString(PyObject* val, bool strict, bool &ok);
    //! get bytes from py object
    static QByteArray PyObjGetBytes(PyObject* val, bool strict, bool &ok);
    //! get int from py object
    static int     PyObjGetInt(PyObject* val, bool strict, bool &ok);
    //! get int64 from py object
    static qint64  PyObjGetLongLong(PyObject* val, bool strict, bool &ok);
    //! get int64 from py object
    static quint64  PyObjGetULongLong(PyObject* val, bool strict, bool &ok);
    //! get double from py object
    static double  PyObjGetDouble(PyObject* val, bool strict, bool &ok);
    //! get double-array from py object
    static QVector<double>  PyObjGetDoubleArray(PyObject* val, bool strict, bool &ok);
    //! get int-array from py object
    static QVector<int>  PyObjGetIntArray(PyObject* val, bool strict, bool &ok);
    //! get bool from py object
    static bool    PyObjGetBool(PyObject* val, bool strict, bool &ok);

#if ITOM_POINTCLOUDLIBRARY > 0
    static ito::PCLPointCloud PyObjGetPointCloud(PyObject *val, bool strict, bool &ok);
    static ito::PCLPoint PyObjGetPoint(PyObject *val, bool strict, bool &ok);
    static ito::PCLPolygonMesh PyObjGetPolygonMesh(PyObject *val, bool strict, bool &ok);

    
    static ito::PCLPointCloud* PyObjGetPointCloudNewPtr(PyObject *val, bool strict, bool &ok);
    static ito::PCLPolygonMesh* PyObjGetPolygonMeshNewPtr(PyObject *val, bool strict, bool &ok);
#endif

	static ito::DataObject* PyObjGetDataObjectNewPtr(PyObject *val, bool strict, bool &ok);

    static QSharedPointer<ito::DataObject> PyObjGetSharedDataObject(PyObject *val, bool &ok); //is always strict, only dataobjects are allowed

    //! create a string list from python sequence
    static QStringList PyObjToStringList(PyObject* val, bool strict, bool& ok);


    //! convert python object to qvariant, if type is given it will try to create a qvariant of that type, otherwise
    //! it will guess from the python type
    static QVariant PyObjToQVariant(PyObject* val, int type = -1);

	static QVariant QVariantCast(const QVariant &item, QVariant::Type destType, ito::RetVal &retval);

    static QVariant QVariantToEnumCast(const QVariant &item, QMetaEnum &enumerator, ito::RetVal &retval);

    //! convert python object to void* using QMetaType. if type is given it will try to create a void* of that type, otherwise
    //! it will guess from the python type. If fails, NULL is returned
    static bool PyObjToVoidPtr(PyObject* val, void **retPtr, int *retType, int type = -1, bool strict = false);

    //! convert QVariant from PyObject
    static PyObject* GetPyBool(bool val);
    static PyObject* QVariantToPyObject(const QVariant& v);

    static PyObject* QVariantMapToPyObject(const QVariantMap& m);
    static PyObject* QVariantListToPyObject(const QVariantList& l);

#if ITOM_POINTCLOUDLIBRARY > 0
    static PyObject* PCLPointCloudToPyObject(const ito::PCLPointCloud& c);
    static PyObject* PCLPointToPyObject(const ito::PCLPoint& c);
    static PyObject* PCLPolygonMeshToPyObject(const ito::PCLPolygonMesh& c);
#endif

    static PyObject* DataObjectToPyObject(const ito::DataObject& dObj);

    static PyObject* ConvertQtValueToPythonInternal(int type, const void* data); 
    static PyObject* convertPyObjectToQVariant(PyObject *argument, QVariant &qVarArg);

    static PyObject* QByteArrayToPyUnicode(const QByteArray &ba, const char *errors = "replace");
    static PyObject* QByteArrayToPyUnicodeSecure(const QByteArray &ba, const char *errors = "replace");
    static PyObject* ByteArrayToPyUnicode(const char* byteArray, Py_ssize_t len = 0, const char *errors = "replace");

    static PyObject* PyUnicodeToPyByteObject(PyObject *unicode, const char *errors = "replace");

    friend class PythonEngine; //such that the pythonEngine can set the encoding values below

private:
    static unicodeEncodings textEncoding;
    static QByteArray textEncodingName;

    /*!
    if any PyObject is converted into a QVariant-object, dataObject or any other class, and if this
    PyObject has a base-pointer unequal to 
    */
    static QHash<void*,PyObject*> m_pyBaseObjectStorage; 
    static void baseObjectDeleterDataObject(ito::DataObject *sharedObject)
    {
        QHash<void*,PyObject*>::iterator i = m_pyBaseObjectStorage.find((void*)sharedObject);
        if(i != m_pyBaseObjectStorage.end())
        {
            Py_XDECREF(i.value());
            m_pyBaseObjectStorage.erase(i);
        }

        delete sharedObject;
    }
};

#endif