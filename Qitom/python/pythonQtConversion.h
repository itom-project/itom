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

#ifndef PYTHONQTCONVERSION_H
#define PYTHONQTCONVERSION_H

#include "pythonDataObject.h"
#include "pythonPlugins.h"
#include "pythonRegion.h"
#include "pythonPCL.h"
#include "pythonAutoInterval.h"
#include "pythonNone.h"
#include "datetime.h"
#include "patchlevel.h"

#include "../global.h"
#include "../../common/itomPlotHandle.h"
#include "../../common/shape.h"

#include "opencv2/opencv.hpp"
#include <qvariant.h>
#include <qmetaobject.h>
#include <qlist.h>
#include <qbytearray.h>
#include <qpointer.h>

#include "../helper/qpropertyHelper.h"

Q_DECLARE_METATYPE(ito::PythonNone)

namespace ito
{

class PythonQtConversion
{
public:
    enum UnicodeEncodings { utf_8, latin_1, iso_8859_1, mbcs, ascii, utf_16, utf_16_LE, utf_16_BE, utf_32, utf_32_BE, utf_32_LE, other };

    //! converts QString to Python string (unicode!)
    static PyObject* QStringToPyObject(const QString& str);

    //! converts QStringList to Python tuple
    static PyObject* QStringListToPyObject(const QStringList& list);

    //! converts QStringList to Python list
    static PyObject* QStringListToPyList(const QStringList& list);

    //! converts QDate to Python datetime.date object
    static PyObject* QDateToPyDate(const QDate& date);

    //! converts QDateTime to Python datetime.datetime object
    static PyObject* QDateTimeToPyDateTime(const QDateTime& datetime);

    //! converts QTime to Python datetime.time object
    static PyObject* QTimeToPyTime(const QTime& time);

    //! get string representation of py object
    static QString PyObjGetRepresentation(PyObject* val);

    //! get string value from py object
    static QString PyObjGetString(PyObject* val) { bool ok; QString s = PyObjGetString(val, false, ok); return s; }
    //! get string value from py object
    static QString PyObjGetString(PyObject* val, bool strict, bool &ok);
    //! get std::string value from py object. std::string has latin1 encoding
    static std::string PyObjGetStdStringAsLatin1(PyObject* val, bool strict, bool &ok);
    //! get bytes from py object
    static QByteArray PyObjGetBytes(PyObject* val, bool strict, bool &ok);
    //! get bytes from py object
    static QSharedPointer<char> PyObjGetBytesShared(PyObject* val, bool strict, bool &ok);
    //! get int from py object
    static int PyObjGetInt(PyObject* val, bool strict, bool &ok);
    //! get unsigned int from py object
    static unsigned int PyObjGetUInt(PyObject* val, bool strict, bool &ok);
    //! get int64 from py object
    static qint64  PyObjGetLongLong(PyObject* val, bool strict, bool &ok);
    //! get int64 from py object
    static quint64  PyObjGetULongLong(PyObject* val, bool strict, bool &ok);
    //! get double from py object
    static double  PyObjGetDouble(PyObject* val, bool strict, bool &ok);
    //! get double-array from py object
    static QVector<double>  PyObjGetDoubleArray(PyObject* val, bool strict, bool &ok);
    //! get complex from py object
    static complex128  PyObjGetComplex(PyObject* val, bool strict, bool &ok);
    //! get complex-array from py object
    static QVector<complex128>  PyObjGetComplexArray(PyObject* val, bool strict, bool &ok);
    //! get int-array from py object
    static QVector<int>  PyObjGetIntArray(PyObject* val, bool strict, bool &ok);
    //! get bool from py object
    static bool    PyObjGetBool(PyObject* val, bool strict, bool &ok);
    //! get shape vector from py object
    static QVector<ito::Shape> PyObjGetShapeVector(PyObject* val, bool &ok);

    //! get ito::ByteArray list from py object
    static QVector<ito::ByteArray> PyObjGetByteArrayList(PyObject *val, bool strict, bool &ok);

#if ITOM_POINTCLOUDLIBRARY > 0
    static ito::PCLPointCloud PyObjGetPointCloud(PyObject *val, bool strict, bool &ok);
    static ito::PCLPoint PyObjGetPoint(PyObject *val, bool strict, bool &ok);
    static ito::PCLPolygonMesh PyObjGetPolygonMesh(PyObject *val, bool strict, bool &ok);


    static ito::PCLPointCloud* PyObjGetPointCloudNewPtr(PyObject *val, bool strict, bool &ok);
    static ito::PCLPolygonMesh* PyObjGetPolygonMeshNewPtr(PyObject *val, bool strict, bool &ok);
#endif

    static ito::DataObject* PyObjGetDataObjectNewPtr(PyObject *val, bool strict, bool &ok, ito::RetVal *retVal = nullptr, bool addNumpyOrgTags = false);

    //! converts the python object to a DataObject. If the returned value is destroyed, possible base objects will be safely
    //! removed, too. If strict is true, the returned DataObject is exactly the same object, than wrapped by the given
    //! python object. Else, numpy-like arrays can also be accepted and will be converted (shallow or deep copy, whatever
    //! is possible) to a dataObject.
    static QSharedPointer<ito::DataObject> PyObjGetSharedDataObject(PyObject *val, bool strict, bool &ok, ito::RetVal *retVal = nullptr);

    //! create a string list from python sequence
    static QStringList PyObjToStringList(PyObject* val, bool strict, bool& ok);


    //! convert python object to qvariant, if type is given it will try to create a qvariant of that type, otherwise
    //! it will guess from the python type
    static QVariant PyObjToQVariant(PyObject* val, int type = -1);

    //! convert python object to char* using QMetaType. if type is given it will try to create a char* of that type, otherwise
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
    static PyObject* AddInBaseToPyObject(ito::AddInBase* aib);

    static PyObject* ConvertQtValueToPythonInternal(int type, const void* data);

    static PyObject* QByteArrayToPyUnicode(const QByteArray &ba, const char *errors = "replace");
    static PyObject* QByteArrayToPyUnicodeSecure(const QByteArray &ba, const char *errors = "replace");
    static PyObject* QByteArrayUtf8ToPyUnicode(const QByteArray &ba, const char *errors = "replace");
    static PyObject* QByteArrayUtf8ToPyUnicodeSecure(const QByteArray &ba, const char *errors = "replace");
    static PyObject* ByteArrayToPyUnicode(const char* byteArray, Py_ssize_t len = 0, const char *errors = "replace");

    static PyObject* PyUnicodeToPyByteObject(PyObject *unicode, const char *errors = "replace");

    friend class PythonEngine; //such that the pythonEngine can set the encoding values below

private:
    static UnicodeEncodings textEncoding;
    static QByteArray textEncodingName;

    static int guessQMetaTypeFromPyObject(PyObject* val);

    /*!
    if any PyObject is converted into a QVariant-object, dataObject or any other class, and if this
    PyObject has a base-pointer unequal to None, this base pointer is incremented during the lifetime of the
    dataObject, passed to QVariant. If this dataObject is destroyed, the baseObjectDeleterDataObject deleter method
    is called and decrements the base PyObject.

    Be careful: For decrementing the refcount, the GIL must be hold by this deleter!
    */
    static QHash<char*,PyObject*> m_pyBaseObjectStorage;
    static void baseObjectDeleterDataObject(ito::DataObject *sharedObject);
};

} //end namespace ito

#endif
