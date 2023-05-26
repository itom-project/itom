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

#include "pythonQtConversion.h"

#include "pythonUi.h"
#include "pythonPlotItem.h"
#include "pythonCommon.h"
#include "pythonRgba.h"
#include "pythonFont.h"
#include "pythonShape.h"
#include "pythonAutoInterval.h"

#include <qstringlist.h>
#include <qurl.h>
#include <qdatetime.h>
#include <qvector2d.h>
#include <qvector3d.h>
#include <qvector4d.h>
#include <qelapsedtimer.h>
#include <QtConcurrent/qtconcurrentrun.h>

#include "pythonSharedPointerGuard.h"

#define METATYPE_CONSTRUCT(type, ptr) QMetaType::create(type, ptr)

namespace ito
{


//-------------------------------------------------------------------------------------
/*!
    \class PythonQtConversion
    \brief Conversion class with mainly static methods, which convert values between Qt and standard C++ data types
        and PyObject* values.

        Parts of this class are taken from the project PythonQt (http://pythonqt.sourceforge.net/)
*/
//PythonQtConversion::unicodeEncodings PythonQtConversion::textEncoding = PythonQtConversion::utf_8;
PythonQtConversion::UnicodeEncodings PythonQtConversion::textEncoding = PythonQtConversion::latin_1;

//QByteArray PythonQtConversion::textEncodingName = "utf8";
QByteArray PythonQtConversion::textEncodingName = "latin_1";
QHash<char*,PyObject*> PythonQtConversion::m_pyBaseObjectStorage = QHash<char*, PyObject*>();

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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
        r = PyObjGetString(str, false, ok);
        Py_DECREF(str);
    }
	else
	{
		PyErr_Clear();
	}

    return r;
}

//-------------------------------------------------------------------------------------
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
        r = QString::fromUtf8(PyObjGetBytes(val, strict, ok));
    }
    else if (PyUnicode_Check(val))
    {
        //we need to have a latin1-decoded string, since we assume to have latin1 in the QString conversion below.
        PyObject *utf8repr = PyUnicode_AsUTF8String(val);

        if (utf8repr != nullptr)
        {
            r = QString::fromUtf8(PyObjGetBytes(utf8repr, strict, ok));
            Py_DECREF(utf8repr);
        }
        else
        {
            PyErr_Clear();
            PyObject* utf16repr = PyUnicode_AsUTF16String(val);

            if (utf16repr)
            {
                Py_ssize_t bytes_length = PyBytes_GET_SIZE(utf16repr);
                r = QString::fromUtf16((const char16_t*)PyBytes_AS_STRING(utf16repr), bytes_length / 2);
                Py_DECREF(utf16repr);
            }
            else
            {
                PyErr_Clear();
                ok = false;
            }
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

//-------------------------------------------------------------------------------------
//! conversion from PyObject* to std::string
/*!
    If val is a byte-object, it is directly converted into a std::string. If val is an unicode-object,
    its value is converted using the current encoding and returned. In any other case the string-like-representation is only
    returned if strict is set to false.

    \param val is the given python object
    \param strict indicates if only real byte or unicode objects can be converted to string
    \param ok (ByRef) is set to true if conversion succeeded.
    \return resulting QString
*/
std::string PythonQtConversion::PyObjGetStdStringAsLatin1(PyObject* val, bool strict, bool& ok)
{
    std::string r;
    ok = true;
    if (PyBytes_Check(val))
    {
        r = std::string(PyObjGetBytes(val, strict, ok));
    }
    else if (PyUnicode_Check(val))
    {
        //we need to have a latin1-decoded string, since we assume to have latin1 in the QString conversion below.
        PyObject *latin1repr = PyUnicode_AsLatin1String(val);
        if (latin1repr != NULL)
        {
            r = std::string(PyObjGetBytes(latin1repr, strict, ok));
            Py_XDECREF(latin1repr);
        }
    }
    else if (!strict)
    {
        // EXTRA: could also use _Unicode, but why should we?
        PyObject* str =  PyObject_Str(val);
        if (str)
        {
            r = PyObjGetStdStringAsLatin1(str, strict, ok);
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

//-------------------------------------------------------------------------------------
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
        const char *b = PyBytes_AS_STRING(val);
        r = QByteArray(b, size);
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

//-------------------------------------------------------------------------------------
QSharedPointer<char> PythonQtConversion::PyObjGetBytesShared(PyObject* val, bool strict, bool& ok)
{
    // TODO: support buffer objects in general
    ok = true;
    if (PyBytes_Check(val))
    {
        Py_ssize_t size = PyBytes_GET_SIZE(val);
        char *b = PyBytes_AS_STRING(val);
        return ito::PythonSharedPointerGuard::createPythonSharedPointer<char>(b, val);
    }
    else if (strict)
    {
        ok = false;
    }
    else
    {
        QSharedPointer<char> r;
        if (PyUnicode_Check(val))
        {
            PyObject *repr2 = PyUnicodeToPyByteObject(val);
            if (repr2 != NULL)
            {
                r = PyObjGetBytesShared(repr2, strict, ok);
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
                r = PyObjGetBytesShared(str, strict, ok);
                Py_DECREF(str);
            }
            else
            {
                ok = false;
            }
        }

        return r;
    }
    return QSharedPointer<char>();
}

//-------------------------------------------------------------------------------------
//! conversion from PyObject* to vector of ito::ByteArray
/*!
    tries to interprete given PyObject* as list of strings and converts it to QVector<ito::ByteArray>.
    If strict is true, we do not want to convert a string to a stringlist, since single strings in python
    are also detected to be sequences.

    Strings are converted to the byte array using the latin1 coded, if possible.

    \param val is the given python object
    \param strict indicates if any object fitting to the sequence protocol is interpreted as string list, too [true]
    \param ok (ByRef) is set to true if conversion was successful, else false
    \return resulting QVector<ito::ByteArray>
*/
QVector<ito::ByteArray> PythonQtConversion::PyObjGetByteArrayList(PyObject *val, bool strict, bool &ok)
{
    QVector<ito::ByteArray> v;
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
        PyObject *value = nullptr;
        QByteArray ba;

        for (int i = 0; i < count; i++)
        {
            value = PySequence_GetItem(val, i); //new reference

            ba = PyObjGetBytes(value, strict, ok);

            v.append(ito::ByteArray(ba.constData()));

            Py_XDECREF(value);
        }

        ok = true;
    }

    return v;
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
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_BOOL) // Scalar
    {
        // cast the scalar numpy type to bool
        PyArray_ScalarAsCtype(val, &d);
    }
    else if (!strict)
    {
        d = (PyObjGetInt(val, false, ok) != 0);
        ok = true;
    }
    return d;
}

template <typename _Tp>
int castIntOverflow(const _Tp val, bool &ok)
{
    static_assert(std::numeric_limits<_Tp>::is_integer, "_Tp must be an integer");
    qint64 val_ = (qint64)val;

    ok = (val_ >= (qint64)std::numeric_limits<int>::min() && val_ <= (qint64)std::numeric_limits<int>::max());

    return static_cast<int>(val);
}

template <typename _Tp>
int castUIntOverflow(const _Tp val, bool &ok)
{
    static_assert(std::numeric_limits<_Tp>::is_integer, "_Tp must be an integer");
    qint64 val_ = (qint64)val;

    ok = (val_ >= 0 && val_ <= (qint64)std::numeric_limits<unsigned int>::max());

    return static_cast<int>(val);
}

//-------------------------------------------------------------------------------------
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
    bool processed = false;

    if (PyLong_Check(val))
    {
        int overflow;
        d = PyLong_AsLongAndOverflow(val, &overflow);
        processed = true;

        if (overflow) //1: too big, -1: too small
        {
            ok = false;
        }
    }
    else if (PyArray_CheckScalar(val))
    {
        int typeNum = PyArray_DescrFromScalar(val)->type_num;
        processed = true;

        switch (typeNum)
        {
        case NPY_ULONGLONG:
        {
            unsigned long long v;
            PyArray_ScalarAsCtype(val, &v);
            d = castIntOverflow(v, ok);
        }
        break;
        case NPY_LONGLONG:
        {
            long long v;
            PyArray_ScalarAsCtype(val, &v);
            d = castIntOverflow(v, ok);
        }
        break;
        case NPY_ULONG:
        {
            unsigned long v;
            PyArray_ScalarAsCtype(val, &v);
            d = castIntOverflow(v, ok);
        }
        break;
        case NPY_LONG:
        {
            long v;
            PyArray_ScalarAsCtype(val, &v);
            d = castIntOverflow(v, ok);
        }
        break;
        case NPY_UINT:
        {
            unsigned int v;
            PyArray_ScalarAsCtype(val, &v);
            d = castIntOverflow(v, ok);
        }
        break;
        case NPY_INT:
        {
            int v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_USHORT:
        {
            unsigned short v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_SHORT:
        {
            short v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_UBYTE:
        {
            unsigned char v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_BYTE:
        {
            char v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        default:
            processed = false;
        break;
        }
    }

    if (!processed && !strict)
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
        else if (PyArray_CheckScalar(val)) // Scalar
        {
            // cast the scalar numpy type to int
            PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_INT);
            PyArray_CastScalarToCtype(val, &d, descr);
            Py_DECREF(descr);
        }
        else
        {
            //try to convert to long (e.g. numpy scalars or other objects that have a __int__() method defined
            int overflow;
            d = PyLong_AsLongAndOverflow(val, &overflow);
            if (PyErr_Occurred())
            {
                //error during conversion
                PyErr_Clear();
                ok = false;
            }
            else if (overflow) //1: too big, -1: too small
            {
                ok = false;
            }
        }
    }
    else if (!processed)
    {
        ok = false;
    }

    return d;
}

//-------------------------------------------------------------------------------------
//! conversion from PyObject* to unsigned integer
/*!
If val is a fixed-point object, it is directly converted into an integer variable. Otherwise, the output depends
on strict. If strict is equal to false, any floating point object is rounded using floor and the result is returned.
Additionally boolean variables are returned as 0 (false) or 1 (true) if strict is false, too.

\param val is the given python object
\param strict indicates if only real integer or long types should be converted
\param ok (ByRef) is set to true if conversion succeeded.
\return resulting integer
*/
unsigned int PythonQtConversion::PyObjGetUInt(PyObject* val, bool strict, bool &ok)
{
    unsigned int uInt = 0;
    ok = true;
    bool processed = false;

    if (PyLong_Check(val))
    {
        uInt = PyLong_AsUnsignedLong(val);
    }
    else if (PyArray_CheckScalar(val))
    {
        int typeNum = PyArray_DescrFromScalar(val)->type_num;
        processed = true;

        switch (typeNum)
        {
        case NPY_ULONGLONG:
        {
            unsigned long long v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_LONGLONG:
        {
            long long v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_ULONG:
        {
            unsigned long v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_LONG:
        {
            long v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_UINT:
        {
            PyArray_ScalarAsCtype(val, &uInt);
        }
        break;
        case NPY_INT:
        {
            int v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        case NPY_USHORT:
        {
            unsigned short v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_SHORT:
        {
            short v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_UBYTE:
        {
            unsigned char v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        case NPY_BYTE:
        {
            char v;
            PyArray_ScalarAsCtype(val, &v);
            uInt = castUIntOverflow(v, ok);
        }
        break;
        default:
            processed = false;
            break;
        }
    }

    if (!processed && !strict)
    {
        if (PyFloat_Check(val))
        {
            uInt = floor(PyFloat_AS_DOUBLE(val));
        }
        else if (val == Py_False)
        {
            uInt = 0;
        }
        else if (val == Py_True)
        {
            uInt = 1;
        }
        else if (PyArray_CheckScalar(val)) // Scalar
        {
            // cast the scalar numpy type to int
            PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_UINT);
            PyArray_CastScalarToCtype(val, &uInt, descr);
            Py_DECREF(descr);
        }
        else
        {
            //try to convert to long (e.g. numpy scalars or other objects that have a __int__() method defined
            uInt = PyLong_AsUnsignedLong(val);
            if (PyErr_Occurred())
            {
                //error during conversion
                PyErr_Clear();
                ok = false;
            }
        }
    }
    else if (!processed)
    {
        ok = false;
    }

    return uInt;
}

//-------------------------------------------------------------------------------------
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
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_LONGLONG) // Scalar
    {
        // cast the scalar numpy type to long long
        PyArray_ScalarAsCtype(val, &d);
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
        else if (PyArray_CheckScalar(val)) // Scalar
        {
            // cast the scalar numpy type to int64
            PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_LONGLONG);
            PyArray_CastScalarToCtype(val, &d, descr);
            Py_DECREF(descr);
        }
        else
        {
            //try to convert to long (e.g. numpy scalars or other objects that have a __int__() method defined
            int overflow;
            d = PyLong_AsLongAndOverflow(val, &overflow);
            if (PyErr_Occurred())
            {
                //error during conversion
                PyErr_Clear();
                ok = false;
            }
            else if (overflow) //1: too big, -1: too small
            {
                ok = false;
            }
        }
    }
    else if (PyArray_CheckScalar(val))
    {
        // try to directly convert from an integer number of smaller size
        int typeNum = PyArray_DescrFromScalar(val)->type_num;

        switch (typeNum)
        {
        case NPY_ULONG:
        {
            unsigned long v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_LONG:
        {
            long v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_UINT:
        {
            unsigned int v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_INT:
        {
            int v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        case NPY_USHORT:
        {
            unsigned short v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_SHORT:
        {
            short v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_UBYTE:
        {
            unsigned char v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_BYTE:
        {
            char v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        default:
            ok = false;
            break;
        }
    }
    else
    {
        ok = false;
    }

    return d;
}

//-------------------------------------------------------------------------------------
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

    if (PyLong_Check(val))
    {
        d = PyLong_AsUnsignedLongLong(val);
        if (PyErr_Occurred())
        {
            ok = false;
            PyErr_Clear();
        }
    }
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_ULONGLONG) // Scalar
    {
        // cast the scalar numpy type to long long
        PyArray_ScalarAsCtype(val, &d);
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
        else if (PyArray_CheckScalar(val)) // Scalar
        {
            // cast the scalar numpy type to uint64
            PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_ULONGLONG);
            PyArray_CastScalarToCtype(val, &d, descr);
            Py_DECREF(descr);
        }
        else
        {
            ok = false;
        }
    }
    else if (PyArray_CheckScalar(val))
    {
        // try to directly convert from an integer number of smaller size
        int typeNum = PyArray_DescrFromScalar(val)->type_num;

        switch (typeNum)
        {
        case NPY_ULONG:
        {
            unsigned long v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_LONG:
        {
            long v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_UINT:
        {
            unsigned int v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_INT:
        {
            int v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        case NPY_USHORT:
        {
            unsigned short v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_SHORT:
        {
            short v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_UBYTE:
        {
            unsigned char v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        case NPY_BYTE:
        {
            char v;
            PyArray_ScalarAsCtype(val, &v);
            d = v;
        }
        break;
        default:
            ok = false;
            break;
        }
    }
    else
    {
        ok = false;
    }
    return d;
}

//-------------------------------------------------------------------------------------
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
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_FLOAT64) // Scalar
    {
        // cast the scalar numpy type to float64
        PyArray_ScalarAsCtype(val, &d);
    }
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_FLOAT32) // Scalar
    {
        // cast the scalar numpy type to float32, then to float64
        float f;
        PyArray_ScalarAsCtype(val, &f);
        d = f;
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
        else if (PyArray_CheckScalar(val)) // Scalar
        {
            // cast the scalar numpy type to float64
            PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_FLOAT64);
            PyArray_CastScalarToCtype(val, &d, descr);
            Py_DECREF(descr);
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
            //try to convert to float (e.g. numpy scalars or other objects that have a __float__() method defined
            d = PyFloat_AsDouble(val);
            if (PyErr_Occurred())
            {
                //error during conversion
                PyErr_Clear();
                ok = false;
            }
        }
    }
    else
    {
        ok = false;
    }
    return d;
}

//-------------------------------------------------------------------------------------
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
    PyObject *t = nullptr;

    for (Py_ssize_t i = 0; i < len; i++)
    {
        t = PySequence_GetItem(val, i); //new reference
        v << PyObjGetDouble(t, strict, ok);
        Py_XDECREF(t);

        if (!ok)
        {
            break;
        }
    }

    if (!ok)
    {
        v.clear();
    }

    return v;
}

//-------------------------------------------------------------------------------------
//! get complex from py object
complex128  PythonQtConversion::PyObjGetComplex(PyObject* val, bool strict, bool &ok)
{
    complex128 d = 0;
    ok = true;

    if (PyComplex_Check(val))
    {
        d = complex128(PyComplex_RealAsDouble(val), PyComplex_ImagAsDouble(val));
    }
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_COMPLEX128) // Scalar
    {
        // cast the scalar numpy type to complex128
        PyArray_ScalarAsCtype(val, &d);
    }
    else if (PyArray_CheckScalar(val) && PyArray_DescrFromScalar(val)->type_num == NPY_COMPLEX64) // Scalar
    {
        // cast the scalar numpy type to complex64
        complex64 cmplx64;
        PyArray_ScalarAsCtype(val, &cmplx64);
        d.real(cmplx64.real());
        d.imag(cmplx64.imag());
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
        else if (PyFloat_Check(val))
        {
            d = PyFloat_AS_DOUBLE(val);
        }
        else if (val == Py_False)
        {
            d = 0.0;
        }
        else if (val == Py_True)
        {
            d = 1.0;
        }
        else if (PyArray_CheckScalar(val)) // Scalar
        {
            // cast the scalar numpy type to complex128
            PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_COMPLEX128);
            PyArray_CastScalarToCtype(val, &d, descr);
            Py_DECREF(descr);
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

//-------------------------------------------------------------------------------------
//! get complex-array from py object
QVector<complex128>  PythonQtConversion::PyObjGetComplexArray(PyObject* val, bool strict, bool &ok)
{
    ok = true;
    QVector<complex128> v;

    if (PySequence_Check(val) == false)
    {
        ok = false;
        return v;
    }

    Py_ssize_t len = PySequence_Size(val);
    PyObject *t = nullptr;

    for (Py_ssize_t i = 0; i < len; i++)
    {
        t = PySequence_GetItem(val, i); //new reference
        v << PyObjGetComplex(t, strict, ok);
        Py_XDECREF(t);

        if (!ok)
        {
            break;
        }
    }

    if (!ok)
    {
        v.clear();
    }

    return v;
}

//-------------------------------------------------------------------------------------
//! get int-array from py object
QVector<int> PythonQtConversion::PyObjGetIntArray(PyObject* val, bool strict, bool &ok)
{
    ok = true;
    QVector<int> v;

    if (PySequence_Check(val) == false)
    {
        ok = false;
        return v;
    }

    Py_ssize_t len = PySequence_Size(val);
    PyObject *t = nullptr;

    for (Py_ssize_t i = 0; i < len; i++)
    {
        t = PySequence_GetItem(val, i); //new reference
        v << PyObjGetInt(t, strict, ok);
        Py_XDECREF(t);

        if (!ok)
        {
            break;
        }
    }

    if (!ok)
    {
        v.clear();
    }

    return v;
}

//-------------------------------------------------------------------------------------
//! get shape vector from pyObj
QVector<ito::Shape> PythonQtConversion::PyObjGetShapeVector(PyObject* val, bool &ok)
{
    QVector<ito::Shape> v;
    ok = true;

    if (PySequence_Check(val) == false)
    {
        ok = false;
        return v;
    }

    Py_ssize_t len = PySequence_Size(val);
    PyObject *t = NULL;

    for (Py_ssize_t i = 0; i < len; i++)
    {
        t = PySequence_GetItem(val, i); //new reference
        if (PyShape_Check(t))
        {
            if (((ito::PythonShape::PyShape*)t)->shape)
                v.append(*((ito::PythonShape::PyShape*)t)->shape);
        }
        else
        {
            ok = false;
            Py_XDECREF(t);
            break;
        }
        Py_XDECREF(t);
    }

    if (!ok)
    {
        v.clear();
    }

    return v;
}

#if ITOM_POINTCLOUDLIBRARY > 0
//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
/*static*/ ito::DataObject* PythonQtConversion::PyObjGetDataObjectNewPtr(PyObject *val, bool strict, bool &ok, ito::RetVal *retVal /*= nullptr*/, bool addNumpyOrgTags /*= false*/)
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
    else if (strict == false) //try to convert numpy.array to dataObject
    {
        if (PyArray_Check(val))
        {
            ito::PythonDataObject::PyDataObject *result;
            result = PyObject_New(ito::PythonDataObject::PyDataObject, &ito::PythonDataObject::PyDataObjectType);

            if (result)
            {
                result->base = nullptr;
                result->dataObject = nullptr;

                PyObject *args = Py_BuildValue("(O)", val);
                PyObject *kwds = PyDict_New();

                if (ito::PythonDataObject::PyDataObj_CreateFromNpNdArrayAndType(result, args, kwds, addNumpyOrgTags) != 0)
                {
                    Py_DECREF(result);
                    result = nullptr;
                }

                Py_DECREF(args);
                Py_DECREF(kwds);
            }

            if (result)
            {
                ito::DataObject *dObj = nullptr;
                dObj = PyObjGetDataObjectNewPtr((PyObject*)result, true, ok, retVal);
                Py_XDECREF(result);
                return dObj;
            }
            else
            {
                ito::RetVal ret = PythonCommon::checkForPyExceptions(true);
                if (retVal)
                {
                    *retVal += ret;
                }

                ok = false;
                return NULL;
            }
        }
        else
        {
            if (retVal)
            {
                *retVal += ito::RetVal(ito::retError, 0, "given object must be of type itom.dataObject or numpy.array.");
            }

            ok = false;
            return NULL;
        }
    }
    else //strict
    {
        if (retVal)
        {
            *retVal += ito::RetVal(ito::retError, 0, "given object must be of type itom.dataObject.");
        }

        ok = false;
        return NULL;
    }
}

//-------------------------------------------------------------------------------------
/*static*/ QSharedPointer<ito::DataObject> PythonQtConversion::PyObjGetSharedDataObject(PyObject *val, bool strict, bool &ok, ito::RetVal *retVal /*= nullptr*/)
{
    QSharedPointer<ito::DataObject> result = QSharedPointer<ito::DataObject>();
    ok = false;

    if (Py_TYPE(val) == &ito::PythonDataObject::PyDataObjectType)
    {
        ito::PythonDataObject::PyDataObject* dObj = (ito::PythonDataObject::PyDataObject*)val;

        if (dObj->dataObject)
        {
            ok = true;
            // returns the internal dataObject of val and increments val to keep the dataObject.
            // The refcount of val is decrementetd by the deleter of the returned shared pointer.
            result = ito::PythonSharedPointerGuard::createPythonSharedPointer<ito::DataObject>(dObj->dataObject, val);
        }
        else if (retVal)
        {
            *retVal += ito::RetVal(ito::retError, 0, "given object must contain a valid dataObject.");
        }
    }
    else if (strict == false) //try to convert numpy.array to dataObject
    {
        if (PyArray_Check(val))
        {
            PyObject *args = Py_BuildValue("(O)", val);

            ito::PythonDataObject::PyDataObject *newPyDataObject =
                (ito::PythonDataObject::PyDataObject*)PyObject_Call(
                    (PyObject*)&ito::PythonDataObject::PyDataObjectType,
                    args,
                    nullptr
                ); //new reference

            ito::DataObject *dObj = nullptr;
            Py_DECREF(args);

            if (newPyDataObject)
            {
                // returns the internal dataObject of val and increments val to keep the dataObject.
                // The refcount of val is decrementetd by the deleter of the returned shared pointer.
                result = ito::PythonSharedPointerGuard::createPythonSharedPointer<ito::DataObject>(
                    newPyDataObject->dataObject,
                    (PyObject*)newPyDataObject
                );

                ok = true;
                Py_XDECREF(newPyDataObject);
            }
            else
            {
                ito::RetVal ret = PythonCommon::checkForPyExceptions(true);

                if (retVal)
                {
                    *retVal += ret;
                }
            }
        }
        else
        {
            if (retVal)
            {
                *retVal += ito::RetVal(ito::retError, 0, "given object must be of type itom.dataObject or numpy.array.");
            }
        }
    }
    else //strict
    {
        if (retVal)
        {
            *retVal += ito::RetVal(ito::retError, 0, "given object must be of type itom.dataObject.");
        }
    }

    return result;
}

#if ITOM_POINTCLOUDLIBRARY > 0
//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
int PythonQtConversion::guessQMetaTypeFromPyObject(PyObject* val)
{
    int type = -1;

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
        else if (value <= std::numeric_limits<int>::max() && value >= std::numeric_limits<int>::min())
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
    else if (PyList_Check(val) || PyTuple_Check(val))
    {
        type = QVariant::List;
    }
    else if (val == Py_None)
    {
        // none is PythonNone
        type = QMetaType::type("ito::PythonNone");
    }
    else if (Py_TYPE(val) == &ito::PythonRegion::PyRegionType)
    {
        type = QVariant::Region;
    }
    else if (Py_TYPE(val) == &ito::PythonFont::PyFontType)
    {
        type = QVariant::Font;
    }
    else if (PyDateTime_Check(val)) //must be checked before PyDate_Check since PyDateTime is derived from PyDate
    {
        type = QVariant::DateTime;
    }
    else if (PyTime_Check(val))
    {
        type = QVariant::Time;
    }
    else if (PyDate_Check(val))
    {
        type = QVariant::Date;
    }

#if ITOM_POINTCLOUDLIBRARY > 0
    else if (Py_TYPE(val) == &ito::PythonPCL::PyPointCloudType)
    {
        //type = QMetaType::type("ito::PCLPointCloud");
        type = QMetaType::type("QSharedPointer<ito::PCLPointCloud>");
    }
    else if (Py_TYPE(val) == &ito::PythonPCL::PyPointType)
    {
        //type = QMetaType::type("ito::PCLPoint");
        type = QMetaType::type("QSharedPointer<ito::PCLPoint>");
    }
    else if (Py_TYPE(val) == &ito::PythonPCL::PyPolygonMeshType)
    {
        //type = QMetaType::type("ito::PCLPolygonMesh");
        type = QMetaType::type("QSharedPointer<ito::PCLPolygonMesh>");
    }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    else if ((Py_TYPE(val) == &ito::PythonDataObject::PyDataObjectType) || PyArray_Check(val))
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
    else if (Py_TYPE(val) == &ito::PythonRgba::PyRgbaType)
    {
        type = QVariant::Color;
    }
    else if (PySequence_Check(val))
    {
        type = QVariant::List;
    }
    else if (Py_TYPE(val) == &ito::PythonAutoInterval::PyAutoIntervalType)
    {
        type = QMetaType::type("ito::AutoInterval");
    }
    else if (Py_TYPE(val) == &ito::PythonPlotItem::PyPlotItemType)
    {
        type = QMetaType::type("ito::ItomPlotHandle");
    }
    else if (Py_TYPE(val) == &ito::PythonUi::PyUiItemType)
    {
        type = QMetaType::type("ito::ItomPlotHandle");
    }
    else if (Py_TYPE(val) == &ito::PythonShape::PyShapeType)
    {
        type = QMetaType::type("ito::Shape");
    }
    else if (PyArray_CheckScalar(val))
    {
        int typeNum = PyArray_DescrFromScalar(val)->type_num;

        switch (typeNum)
        {
        case NPY_BOOL:
            type = QVariant::Bool;
            break;
        case NPY_BYTE:
            type = QMetaType::Char;
            break;
        case NPY_UBYTE:
            type = QMetaType::UChar;
            break;
        case NPY_SHORT:
            type = QMetaType::Short;
            break;
        case NPY_USHORT:
            type = QMetaType::UShort;
            break;
        case NPY_INT:
            type = QMetaType::Int;
            break;
        case NPY_UINT:
            type = QMetaType::UInt;
            break;
        case NPY_LONG:
            type = QMetaType::Long;
            break;
        case NPY_ULONG:
            type = QMetaType::ULong;
            break;
        case NPY_LONGLONG:
            type = QMetaType::LongLong;
            break;
        case NPY_ULONGLONG:
            type = QMetaType::ULongLong;
            break;
        };
    }

    return type;
}

//-------------------------------------------------------------------------------------
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

    //initialize datetime api that is used for date, time, datetime conversion
    if (!PyDateTimeAPI)
    {
        PyDateTime_IMPORT;
    }

    if (type == -1)
    {
        type = guessQMetaTypeFromPyObject(val);
    }

    // special type request:
    switch (type)
    {
    case QVariant::Invalid:
        return v;

    case QVariant::Int:
    {
        int d = PyObjGetInt(val, false, ok);
        if (ok) return QVariant(d);
    }
    break;

    case QVariant::UInt:
    {
        unsigned int d = PyObjGetUInt(val, false, ok);
        if (ok) v = QVariant(d);
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
    }
    break;

    case QMetaType::Float:
    {
        float d = (float) PyObjGetDouble(val, false, ok);
        if (ok) v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::Long:
    {
        long d = static_cast<long>(PyObjGetLongLong(val, false, ok));
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::ULong:
    {
        unsigned long d = static_cast<unsigned long>(PyObjGetLongLong(val, false, ok));
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::LongLong:
    {
        qint64 d = PyObjGetLongLong(val, false, ok);
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::ULongLong:
    {
        quint64 d = PyObjGetULongLong(val, false, ok);
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::Short:
    {
        short d = cv::saturate_cast<short>(PyObjGetInt(val, false, ok));
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::UShort:
    {
        unsigned short d = cv::saturate_cast<unsigned short>(PyObjGetInt(val, false, ok));
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::Char:
    {
        char d = cv::saturate_cast<char>(PyObjGetInt(val, false, ok));
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QMetaType::UChar:
    {
        unsigned char d = cv::saturate_cast<unsigned char>(PyObjGetInt(val, false, ok));
        if (ok)
            v = QVariant::fromValue(d);
    }
    break;

    case QVariant::ByteArray:
    {
        QByteArray ba = PyObjGetBytes(val, true, ok);
        if (ok)
            v = QVariant::fromValue(ba);
    }
    break;

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

    case QVariant::Font:
    {
        ito::PythonFont::PyFont *pyFont = (ito::PythonFont::PyFont*)val;
        if (pyFont && pyFont->font)
        {
            v = *(pyFont->font);
        }
    }
    break;

    case QVariant::Time:
    {
        PyDateTime_Time *o = (PyDateTime_Time*)val;
        v = QTime(PyDateTime_TIME_GET_HOUR(o), PyDateTime_TIME_GET_MINUTE(o), PyDateTime_TIME_GET_SECOND(o), PyDateTime_TIME_GET_MICROSECOND(o));
    }
    break;

    case QVariant::Date:
    {
        PyDateTime_Date *o = (PyDateTime_Date*)val;
        v = QDate(PyDateTime_GET_YEAR(o), PyDateTime_GET_MONTH(o), PyDateTime_GET_DAY(o));
    }
    break;

    case QVariant::DateTime:
    {
        PyDateTime_DateTime *o = (PyDateTime_DateTime*)val;
        QDate date(PyDateTime_GET_YEAR(o), PyDateTime_GET_MONTH(o), PyDateTime_GET_DAY(o));
        QTime time(PyDateTime_DATE_GET_HOUR(o), PyDateTime_DATE_GET_MINUTE(o), PyDateTime_DATE_GET_SECOND(o), PyDateTime_DATE_GET_MICROSECOND(o));
        v = QDateTime(date, time);
    }
    break;

    case QVariant::Color:
    {
        if (PyRgba_Check(val))
        {
            ito::PythonRgba::PyRgba *rgba = (ito::PythonRgba::PyRgba*)val;
            v = QColor(rgba->rgba.r, rgba->rgba.g, rgba->rgba.b, rgba->rgba.a);
        }
        else
        {
            v = QVariant();
        }
    }
    break;


    default:
    {
        if (type == QMetaType::type("QSharedPointer<ito::DataObject>"))
        {
            if (PyDataObject_Check(val))
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
                        //baseObjectDeleter
                        Py_XINCREF(dataObj->base);
                        ito::DataObject *copy = new ito::DataObject(*dataObj->dataObject);
                        m_pyBaseObjectStorage.insert((char*)copy, dataObj->base); //unique
                        QSharedPointer<ito::DataObject> value(copy , baseObjectDeleterDataObject);
                        v = QVariant::fromValue<QSharedPointer<ito::DataObject>>(value);
                    }
                    else
                    {
                        QSharedPointer<ito::DataObject> value(new ito::DataObject(*dataObj->dataObject));
                        v = QVariant::fromValue<QSharedPointer<ito::DataObject>>(value);
                    }
                }
                else
                {
                    v = QVariant();
                }
            }
            else if (PyArray_Check(val))
            {
                //try to create a dataObject (Python object) from given numpy array
                PyObject *pyDataObj = PythonDataObject::PyDataObjectType.tp_new(&PythonDataObject::PyDataObjectType,NULL,NULL);  //new ref
                if (pyDataObj)
                {
                    PyObject *args = Py_BuildValue("(O)", val);
                    PyObject *kwds = PyDict_New();
                    int result = PythonDataObject::PyDataObjectType.tp_init(pyDataObj, args, kwds);
                    Py_XDECREF(args);
                    Py_XDECREF(kwds);

                    if (result == 0)
                    {
                        PythonDataObject::PyDataObject *pyDataObj2 = (PythonDataObject::PyDataObject*)pyDataObj;
                        if (pyDataObj2->dataObject == NULL)
                        {
                            v = QVariant();
                        }
                        else if (pyDataObj2->base != NULL) //if the python-dataObject shares memory with other arrays (like a numpy array, we need to make a deep copy here, since we cannot increment the reference of
                        {
                            //baseObjectDeleter
                            Py_XINCREF(pyDataObj2->base);
                            ito::DataObject *copy = new ito::DataObject(*pyDataObj2->dataObject);
                            m_pyBaseObjectStorage.insert((char*)copy, pyDataObj2->base); //unique
                            QSharedPointer<ito::DataObject> value(copy , baseObjectDeleterDataObject);
                            v = QVariant::fromValue<QSharedPointer<ito::DataObject>>(value);
                        }
                        else
                        {
                            QSharedPointer<ito::DataObject> value(new ito::DataObject(*pyDataObj2->dataObject));
                            v = QVariant::fromValue<QSharedPointer<ito::DataObject>>(value);
                        }
                    }
                    else
                    {
                        PyErr_PrintEx(0);
                        v = QVariant();
                    }

                    Py_DECREF(pyDataObj);
                }
                else
                {
                    v = QVariant();
                }
            }
        }
#if ITOM_POINTCLOUDLIBRARY > 0
        else if (type == QMetaType::type("ito::PCLPointCloud"))
        {
            bool ok;
            ito::PCLPointCloud pcl = PyObjGetPointCloud(val, true, ok);
            if (ok)
            {
                v = QVariant::fromValue<ito::PCLPointCloud>(pcl);
            }
        }
        else if (type == QMetaType::type("QSharedPointer<ito::PCLPointCloud>"))
        {
            if (PyPointCloud_Check(val))
            {
                ito::PythonPCL::PyPointCloud* pyPlc = (ito::PythonPCL::PyPointCloud*)val;
                if (pyPlc && pyPlc->data)
                {
                    if (pyPlc->data == NULL)
                    {
                        v = QVariant();
                    }
                    else
                    {
                        QSharedPointer<ito::PCLPointCloud> value(new ito::PCLPointCloud(*pyPlc->data));
                        v = QVariant::fromValue<QSharedPointer<ito::PCLPointCloud>>(value);
                    }
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
        else if (type == QMetaType::type("ito::PCLPoint"))
        {
            bool ok;
            ito::PCLPoint pcl = PyObjGetPoint(val, true, ok);
            if (ok)
            {
                v = QVariant::fromValue<ito::PCLPoint>(pcl);
            }
        }
        else if (type == QMetaType::type("ito::PCLPolygonMesh"))
        {
            bool ok;
            ito::PCLPolygonMesh pcl = PyObjGetPolygonMesh(val, true, ok);
            if (ok)
            {
                v = QVariant::fromValue<ito::PCLPolygonMesh>(pcl);
            }
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
        else if (type == QMetaType::type("QPointer<ito::AddInDataIO>"))
        {
            ito::PythonPlugins::PyDataIOPlugin *plugin = (ito::PythonPlugins::PyDataIOPlugin*)val;
            if (plugin)
            {
                v = QVariant::fromValue<QPointer<ito::AddInDataIO>>(
                    QPointer<ito::AddInDataIO>(plugin->dataIOObj));
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
                v = QVariant::fromValue<QPointer<ito::AddInActuator>>(
                    QPointer<ito::AddInActuator>(plugin->actuatorObj));
            }
            else
            {
                v = QVariant();
            }
        }
        else if (type == QMetaType::type("ito::AutoInterval"))
        {
            ito::PythonAutoInterval::PyAutoInterval *ai = (ito::PythonAutoInterval::PyAutoInterval*)val;
            if (ai)
            {
                v = QVariant::fromValue<ito::AutoInterval>(ai->interval);
            }
            else
            {
                v = QVariant();
            }
        }
        else if (type == QMetaType::type("ito::ItomPlotHandle"))
        {
            if (PyPlotItem_Check(val))
            {
                ito::PythonPlotItem::PyPlotItem *plot = (ito::PythonPlotItem::PyPlotItem*)val;
                if (plot)
                {
                    ito::ItomPlotHandle myHandle(plot->uiItem.objName, plot->uiItem.widgetClassName, plot->uiItem.objectID);
                    v = QVariant::fromValue<ito::ItomPlotHandle>(myHandle);
                }
                else
                {
                    v = QVariant();
                }
            }
            else if (PyUiItem_Check(val))
            {
                ito::PythonUi::PyUiItem *ui = (ito::PythonUi::PyUiItem*)val;
                if (ui)
                {
                    ito::ItomPlotHandle myHandle(ui->objName, ui->widgetClassName, ui->objectID);
                    v = QVariant::fromValue<ito::ItomPlotHandle>(myHandle);
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
        else if (type == QMetaType::type("ito::Shape"))
        {
            ito::PythonShape::PyShape *shape = (ito::PythonShape::PyShape*)val;
            if (shape)
            {
                v = QVariant::fromValue<ito::Shape>(*(shape->shape));
            }
            else
            {
                v = QVariant();
            }
        }
        else if (type == QMetaType::type("ito::PythonNone"))
        {
            ito::PythonNone none;
            v = QVariant::fromValue<ito::PythonNone>(none);
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

//-------------------------------------------------------------------------------------
//! tries to convert PyObject* to known data type and returns deep-copy of the value, given as char*
/*!
    methods tries to convert PyObject* to QVariant. Type indicates the desired type of QVariant, given by the type-number of QMetaType.
    If type==-1, the right type is guessed by checking the PyObject*-type. If conversion failed, QVariant() is returned.

    The deep-copy is created using QMetaType::create(...)

    \param val is the given python object
    \param retPtr is the resulting pointer to type char*, containing the converted deep copy or NULL, if error
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

    //initialize datetime api that is used for date, time, datetime conversion
    if (!PyDateTimeAPI)
    {
        PyDateTime_IMPORT;
    }

    if (type == -1)
    {
        type = guessQMetaTypeFromPyObject(val);
    }

    *retPtr = nullptr; //invalidate it first

    if (QMetaType::isRegistered(type))
    {

        if (QMetaType::typeFlags(type) & QMetaType::IsEnumeration)
        {
            unsigned int d = PyObjGetUInt(val, strict, ok);
            if (ok)
            {
                *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
            }
            else
            {
                QString s = PyObjGetString(val, strict, ok);
                if (ok)
                {
                    ito::RetVal retval;
                    QMetaType mt(type);
                    const QMetaObject *mo = mt.metaObject();
                    if (mo)
                    {
                        QByteArray name = QMetaType::typeName(type);
                        int idx = name.indexOf("::");
                        if (idx >= 0)
                        {
                            name = name.mid(idx + 2);
                        }

                        idx = mo->indexOfEnumerator(name.data());

                        if (idx >= 0)
                        {
                            int d = QPropertyHelper::QVariantToEnumCast(s, mo->enumerator(idx), retval).toInt(&ok);
                            if (ok && !retval.containsError())
                            {
                                *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                            }
                        }
                    }
                }
            }
        }

        if (*retPtr == nullptr)
        {
            // special type request:
            switch (type)
            {
            case QMetaType::Void:
                *retPtr = METATYPE_CONSTRUCT(type, nullptr);
                break;
            case QMetaType::Int:
            {
                int d = PyObjGetInt(val, strict, ok);
                if (ok)
                {
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::UInt:
            {
                unsigned int d = PyObjGetUInt(val, strict, ok);
                if (ok)
                {
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::Bool:
            {
                int d = PyObjGetBool(val, strict, ok);
                if (ok)
                {
                    bool d2 = (d != 0);
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d2));
                }
                break;
            }
            case QMetaType::Double:
            {
                double d = PyObjGetDouble(val, strict, ok);
                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::Float:
            {
                float d = (float) PyObjGetDouble(val, strict, ok);
                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::Long:
            {
                long d = static_cast<long>(PyObjGetLongLong(val, strict, ok));
                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::ULong:
            {
                unsigned long d = static_cast<unsigned long>(PyObjGetULongLong(val, strict, ok));
                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::LongLong:
            {
                qint64 d = PyObjGetLongLong(val, strict, ok);
                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::ULongLong:
            {
                quint64 d = PyObjGetULongLong(val, strict, ok);

                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::Short:
            {
                short d = cv::saturate_cast<short>(PyObjGetInt(val, strict, ok));
                if (ok)
                {
                    * retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::UShort:
            {
                unsigned short d = cv::saturate_cast<unsigned short>(PyObjGetInt(val, strict, ok));
                if (ok)
                {
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }
            case QMetaType::Char:
            {
                char d = cv::saturate_cast<char>(PyObjGetInt(val, strict, ok));
                if (ok)
                {
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));

                }
                break;
            }
            case QMetaType::UChar:
            {
                unsigned char d = cv::saturate_cast<unsigned char>(PyObjGetInt(val, strict, ok));
                if (ok)
                {
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                }
                break;
            }

            case QMetaType::QByteArray:
            {
                QByteArray text = PyObjGetBytes(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::create(type, reinterpret_cast<char*>(&text));
                }
                break;
            }
            case QMetaType::QString:
            {
                QString text = PyObjGetString(val, strict, ok);
                if (ok)
                {
                    *retPtr = QMetaType::create(type, reinterpret_cast<char*>(&text));
                }
                break;
            }
            case QMetaType::QUrl:
            {
                QString text = PyObjGetString(val, strict, ok);
                if (ok)
                {
                    QUrl url = QUrl(text);
                    *retPtr = QMetaType::create(type, reinterpret_cast<char*>(&url));
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
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&map));
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
                *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&list));
                break;
            }

            case QMetaType::QStringList:
            {
                bool ok;
                QStringList l = PyObjToStringList(val, strict, ok);
                if (ok)
                {
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&l));
                }
                break;
            }

            case QMetaType::QRegion:
            {
                if (PyRegion_Check(val))
                {
                    ito::PythonRegion::PyRegion *reg = (ito::PythonRegion::PyRegion*)val;
                    if (reg && reg->r)
                    {
                        QRegion r = *(reg->r);
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&r));
                    }
                }
                break;
            }

            case QMetaType::QFont:
            {
                if (PyFont_Check(val))
                {
                    ito::PythonFont::PyFont *font = (ito::PythonFont::PyFont*)val;
                    if (font && font->font)
                    {
                        QFont f = *(font->font);
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&f));
                    }
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
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&c));
                    }
                }
                else
                {
                    QString text = PyObjGetString(val, strict, ok);
                    if (ok)
                    {
                        QColor c(text);
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&c));
                    }
                }
                break;
            }
            case QVariant::Time:
            {
                if (PyTime_Check(val))
                {
                    PyDateTime_Time *o = (PyDateTime_Time*)val;
                    QTime l = QTime(PyDateTime_TIME_GET_HOUR(o), PyDateTime_TIME_GET_MINUTE(o), PyDateTime_TIME_GET_SECOND(o), PyDateTime_TIME_GET_MICROSECOND(o));
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&l));
                }
                break;
            }
            case QVariant::Date:
            {
                if (PyDate_Check(val))
                {
                    PyDateTime_Date *o = (PyDateTime_Date*)val;
                    QDate l = QDate(PyDateTime_GET_YEAR(o), PyDateTime_GET_MONTH(o), PyDateTime_GET_DAY(o));
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&l));
                }
                break;
            }
            case QVariant::DateTime:
            {
                if (PyDateTime_Check(val))
                {
                    PyDateTime_DateTime *o = (PyDateTime_DateTime*)val;
                    QDate date(PyDateTime_GET_YEAR(o), PyDateTime_GET_MONTH(o), PyDateTime_GET_DAY(o));
                    QTime time(PyDateTime_DATE_GET_HOUR(o), PyDateTime_DATE_GET_MINUTE(o), PyDateTime_DATE_GET_SECOND(o), PyDateTime_DATE_GET_MICROSECOND(o));
                    QDateTime l = QDateTime(date, time);
                    *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&l));
                }
                break;
            }
            default:
            //check user defined types
            {
                if (type == QMetaType::type("ito::PythonQObjectMarshal"))
                {
                    if (PyUiItem_Check(val))
                    {
                        ito::PythonUi::PyUiItem *val2 = (ito::PythonUi::PyUiItem*)val;

                        ito::PythonQObjectMarshal m = ito::PythonQObjectMarshal();
                        m.m_objectID = val2->objectID;
                        m.m_object = NULL;
                        m.m_objName = val2->objName;
                        m.m_className = val2->widgetClassName;
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&m));
                    }
                }
                else if (type == QMetaType::type("QSharedPointer<ito::DataObject>"))
                {
                    if (PyDataObject_Check(val))
                    {
                        ito::PythonDataObject::PyDataObject *val2 = (ito::PythonDataObject::PyDataObject*)val;
                        if (val2 && val2->dataObject)
                        {
                            QSharedPointer<ito::DataObject> sharedBuffer = ito::PythonSharedPointerGuard::createPythonSharedPointer<ito::DataObject>(val2->dataObject, val);
                            *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&sharedBuffer));
                        }
                    }
                }
                else if (type == QMetaType::type("ito::DataObject"))
                {
                    if (PyDataObject_Check(val))
                    {
                        ito::PythonDataObject::PyDataObject *val2 = (ito::PythonDataObject::PyDataObject*)val;
                        if (val2 && val2->dataObject)
                        {
                            *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(val2->dataObject));
                        }
                    }
                }
#if ITOM_POINTCLOUDLIBRARY > 0
                else if (type == QMetaType::type("ito::PCLPointCloud") || type == QMetaType::type("ito::PCLPointCloud&"))
                {
                    bool ok;
                    ito::PCLPointCloud pcl = PyObjGetPointCloud(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&pcl));
                    }
                }
                else if (type == QMetaType::type("QSharedPointer<ito::PCLPointCloud>"))
                {
                    if (PyPointCloud_Check(val))
                    {
                        ito::PythonPCL::PyPointCloud* pyPlc = (ito::PythonPCL::PyPointCloud*)val;
                        if (pyPlc && pyPlc->data)
                        {
                            QSharedPointer<ito::PCLPointCloud> sharedBuffer = ito::PythonSharedPointerGuard::createPythonSharedPointer<ito::PCLPointCloud>(pyPlc->data, val);
                            *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&sharedBuffer));
                        }
                    }
                }
                else if (type == QMetaType::type("ito::PCLPoint"))
                {
                    bool ok;
                    ito::PCLPoint pt = PyObjGetPoint(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&pt));
                    }
                }
                else if (type == QMetaType::type("ito::PCLPolygonMesh") || type == QMetaType::type("ito::PCLPolygonMesh&"))
                {
                    bool ok;
                    ito::PCLPolygonMesh mesh = PyObjGetPolygonMesh(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&mesh));
                    }
                }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
                else if (type == QMetaType::type("QSharedPointer<char>"))
                {
                    QSharedPointer<char> text = PyObjGetBytesShared(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = QMetaType::create(type, reinterpret_cast<char*>(&text));
                    }
                    break;
                }
                else if (type == QMetaType::type("QVector<double>"))
                {
                    bool ok;
                    QVector<double> arr = PyObjGetDoubleArray(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&arr));
                    }
                }
                else if (type == QMetaType::type("QVector<int>"))
                {
                    bool ok;
                    QVector<int> arr = PyObjGetIntArray(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&arr));
                    }
                }
				else if (type == QMetaType::type("QList<double>"))
				{
					bool ok;
					QVector<double> arr = PyObjGetDoubleArray(val, strict, ok);
					if (ok)
					{
						QList<double> arr2 = arr.toList();
						*retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&arr2));
					}
				}
				else if (type == QMetaType::type("QList<int>"))
				{
					bool ok;
					QVector<int> arr = PyObjGetIntArray(val, strict, ok);
					if (ok)
					{
						QList<int> arr2 = arr.toList();
						*retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&arr2));
					}
				}
                else if (type == QMetaType::type("ito::Shape"))
                {
                    if (PyShape_Check(val))
                    {
                        ito::PythonShape::PyShape* shape = (ito::PythonShape::PyShape*)val;
                        if (shape)
                        {
                            *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(shape->shape));
                        }
                    }
                }
                else if (type == QMetaType::type("QVector<ito::Shape>"))
                {
                    bool ok;
                    QVector<ito::Shape> vec = PyObjGetShapeVector(val, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&vec));
                    }
                }
                else if (type == QMetaType::type("Qt::ItemFlags"))
                {
                    bool ok;
                    int d = PyObjGetInt(val, strict, ok);
                    if (ok)
                    {
                        *retPtr = METATYPE_CONSTRUCT(type, reinterpret_cast<char*>(&d));
                    }
                }
                else
                {
                    *retPtr = NULL;
                }
                break;
            }

            } //end switch case

        }

        //now check if qvariant is true, then retPtr and retType needs to be retransformed to QVariant
        if (qvariant && *retPtr)
        {
            void *ptrToOriginalValue = *retPtr;

            QVariant* variantValue = nullptr;
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
            variantValue = new QVariant(QMetaType(type), *retPtr);
#else
            variantValue = new QVariant(type, *retPtr);
#endif

            if (variantValue->isValid())
            {
                *retType = QMetaType::QVariant;
                *retPtr = (char*)variantValue; //here no QMetaType::create is used since construct does not more than a copy constructor of QVariant casted to char*
            }
            else
            {
                *retType = -1;
                *retPtr = NULL;
                DELETE_AND_SET_NULL(variantValue);
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

//-------------------------------------------------------------------------------------
//! conversion from bool to python boolean type
/*!
    Returns new reference to Py_True or Py_False depending on input val.

    \param val is the given boolean input
    \return resulting PyObject* (new reference)
*/
PyObject* PythonQtConversion::GetPyBool(bool val)
{
    PyObject* r = val ? Py_True : Py_False;
    Py_INCREF(r);
    return r;
}

//-------------------------------------------------------------------------------------
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
        return ByteArrayToPyUnicode("", 0);
    }
    else
    {
        const QByteArray ba = str.toUtf8();
        PyObject *unicode = PyUnicode_DecodeUTF8(ba.constData(), ba.size(), nullptr);
        return unicode;

        //str.toLatin1() decodes with current encoding and then it is transformed to PyObject with the same encoding
        //return QByteArrayToPyUnicode(str.toLatin1());
    }
}

//-------------------------------------------------------------------------------------
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
        PyTuple_SET_ITEM(result, i, PythonQtConversion::QStringToPyObject(str)); //steals reference
        i++;
    }
    // why is the error state bad after this?
    PyErr_Clear();
    return result;
}

//-------------------------------------------------------------------------------------
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

//! converts QDate to Python datetime.date object
PyObject* PythonQtConversion::QDateToPyDate(const QDate& date)
{
    if (!PyDateTimeAPI)
    {
        PyDateTime_IMPORT;
    }
    return PyDate_FromDate(date.year(),date.month(),date.day()); //new reference
}

//! converts QDateTime to Python datetime.datetime object
PyObject* PythonQtConversion::QDateTimeToPyDateTime(const QDateTime& datetime)
{
    if (!PyDateTimeAPI)
    {
        PyDateTime_IMPORT;
    }
    QTime time = datetime.time();
    QDate date = datetime.date();
    return PyDateTime_FromDateAndTime(date.year(),date.month(),date.day(), time.hour(), time.minute(), time.second(), time.msec()); //new reference
}

//! converts QTime to Python datetime.time object
PyObject* PythonQtConversion::QTimeToPyTime(const QTime& time)
{
    if (!PyDateTimeAPI)
    {
        PyDateTime_IMPORT;
    }
    return PyTime_FromTime(time.hour(), time.minute(), time.second(), time.msec()); //new reference
}

//-------------------------------------------------------------------------------------
//! conversion from given QVariant to appropriate PyObject*
/*!
    returns new reference to PyObject*, which contains the conversion from given QVariant-variable.

    \param v is reference to QVariant
    \return is the resulting PyObject*
    \see ConvertQtValueToPytonInternal
*/
PyObject* PythonQtConversion::QVariantToPyObject(const QVariant& v)
{
    return ConvertQtValueToPythonInternal(v.userType(), (char*)v.constData());
}

//-------------------------------------------------------------------------------------
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
//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
PyObject* PythonQtConversion::PCLPointToPyObject(const ito::PCLPoint& c)
{
    ito::PythonPCL::PyPoint *result = (ito::PythonPCL::PyPoint*)PyObject_Call((PyObject*)&(ito::PythonPCL::PyPointType), NULL, NULL);
    if (result)
    {
        DELETE_AND_SET_NULL(result->point);
        *(result->point) = c;
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_RuntimeError, "could not create instance of pclPoint");
    return NULL;
}

//-------------------------------------------------------------------------------------
PyObject* PythonQtConversion::PCLPolygonMeshToPyObject(const ito::PCLPolygonMesh& c)
{
    ito::PythonPCL::PyPolygonMesh *result = (ito::PythonPCL::PyPolygonMesh*)PyObject_Call((PyObject*)&(ito::PythonPCL::PyPolygonMeshType), NULL, NULL);
    if (result)
    {
        DELETE_AND_SET_NULL(result->polygonMesh);
        result->polygonMesh = new ito::PCLPolygonMesh(c);
        //*result->polygonMesh = c;
        return (PyObject*)result;
    }
    PyErr_SetString(PyExc_RuntimeError, "could not create instance of pclPolygonMesh");
    return NULL;
}

#endif //#if ITOM_POINTCLOUDLIBRARY > 0

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
PyObject* PythonQtConversion::AddInBaseToPyObject(ito::AddInBase* aib)
{
    if (aib->getBasePlugin()->getType() & ito::typeDataIO)
    {
        ito::PythonPlugins::PyDataIOPlugin *dataIOPlugin = (ito::PythonPlugins::PyDataIOPlugin*)PythonPlugins::PyDataIOPluginType.tp_new(&ito::PythonPlugins::PyDataIOPluginType, NULL, NULL); //new ref
        if (dataIOPlugin == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "No instance of python class dataIO could be created");
            return NULL;
        }
        else
        {
            aib->getBasePlugin()->incRef(aib);
            dataIOPlugin->dataIOObj = (ito::AddInDataIO*)aib;
            return (PyObject*)dataIOPlugin;
        }
    }
    else if (aib->getBasePlugin()->getType() & ito::typeActuator)
    {
        ito::PythonPlugins::PyActuatorPlugin *actuatorPlugin = (ito::PythonPlugins::PyActuatorPlugin*)PythonPlugins::PyActuatorPluginType.tp_new(&ito::PythonPlugins::PyActuatorPluginType, NULL, NULL); //new ref
        if (actuatorPlugin == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "No instance of python class actuator could be created");
            return NULL;
        }
        else
        {
            aib->getBasePlugin()->incRef(aib);
            actuatorPlugin->actuatorObj = (ito::AddInActuator*)aib;
            return (PyObject*)actuatorPlugin;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "AddIn must be of type dataIO or actuator");
    }

    return NULL;

}

//-------------------------------------------------------------------------------------
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
    PyObject* item = nullptr;

    for (int i = 0; i < l.count(); ++i)
    {
        item = PythonQtConversion::QVariantToPyObject(l[i]);

        if (!item)
        {
            PyErr_Format(PyExc_ValueError, "The QVariant value cannot be converted to a Python type.");
            Py_XDECREF(result);
            return nullptr;
        }

        PyTuple_SET_ITEM(result, i, item);
    }

    return result;
}

//-------------------------------------------------------------------------------------
//! method internally used for conversion from given type-id (QMetaType) and corresponding char*-pointer to PyObject*
/*!
    This method is the opposite from \a PyObjToVoidPtr and converts a pair given by type-id (see QMetaType) and corresponding char*-pointer,
    which to the variable's content to the appropriate python type.

    A python error is returned if conversion failed.

    \param type is given type-id (\a QMetaType)
    \param data is the content, casted to char*
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
    case QMetaType::QFont:
        {
            return ito::PythonFont::createPyFont(*((QFont*)data));
        }
    case QMetaType::QColor:
        {
            ito::PythonRgba::PyRgba *rgba = ito::PythonRgba::createEmptyPyRgba();
            QColor* color = (QColor*)data;
            if (rgba)
            {
                rgba->rgba.r = color->red();
                rgba->rgba.b = color->blue();
                rgba->rgba.g = color->green();
                rgba->rgba.a = color->alpha();
            }
            return (PyObject*)rgba;
        }
    case QMetaType::QTime:
        {
            QTime temp = *(QTime*)data;
            return PythonQtConversion::QTimeToPyTime(temp);
        }
    case QMetaType::QDate:
        {
            QDate temp = *(QDate*)data;
            return PythonQtConversion::QDateToPyDate(temp);
        }
    case QMetaType::QDateTime:
        {
            QDateTime temp = *(QDateTime*)data;
            return PythonQtConversion::QDateTimeToPyDateTime(temp);
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
        if (strcmp(name, "ito::AutoInterval") == 0)
        {
            ito::AutoInterval *v = (ito::AutoInterval*)data;
            ito::PythonAutoInterval::PyAutoInterval *ai = ito::PythonAutoInterval::createEmptyPyAutoInterval();
            ai->interval = *v;
            return (PyObject*)ai;

        }
        if (strcmp(name, "ito::ItomPlotHandle") == 0)
        {
            ito::ItomPlotHandle *v = (ito::ItomPlotHandle*)data;

            if (v->getObjectID() > 0)
            {
                ito::PythonPlotItem::PyPlotItem *plotItem = (ito::PythonPlotItem::PyPlotItem*) ito::PythonPlotItem::PyPlotItem_new(&ito::PythonUi::PyUiItemType, NULL, NULL);
                plotItem->uiItem.objectID = v->getObjectID();

                DELETE_AND_SET_NULL_ARRAY(plotItem->uiItem.objName);
                plotItem->uiItem.objName = new char[v->getObjName().length()+1];
                strcpy_s(plotItem->uiItem.objName, v->getObjName().length()+1, v->getObjName().data());
                DELETE_AND_SET_NULL_ARRAY(plotItem->uiItem.widgetClassName);
                plotItem->uiItem.widgetClassName = new char[v->getWidgetClassName().length()+1];
                strcpy_s(plotItem->uiItem.widgetClassName, v->getWidgetClassName().length()+1, v->getWidgetClassName().data());

                return (PyObject*)plotItem;
            }
            else
            {
                //invalid plot handle
                Py_RETURN_NONE;
            }

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
        else if (strcmp(name, "QSharedPointer<ito::PCLPointCloud>") == 0)
        {
            QSharedPointer<ito::PCLPointCloud> *sharedPtr = (QSharedPointer<ito::PCLPointCloud>*)data;
            if (sharedPtr == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "The given QSharedPointer is NULL");
                return NULL;
            }
            if (sharedPtr->data() == NULL)
            {
                Py_RETURN_NONE;
                //return PyErr_SetString(PyExc_TypeError, "Internal dataObject of QSharedPointer is NULL");
            }
            return PCLPointCloudToPyObject(*(sharedPtr->data()));
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
                PyErr_SetString(PyExc_TypeError, "The given QSharedPointer is NULL");
                return NULL;
            }
            if (sharedPtr->data() == NULL)
            {
                Py_RETURN_NONE;
                //return PyErr_SetString(PyExc_TypeError, "Internal dataObject of QSharedPointer is NULL");
            }
            return DataObjectToPyObject(*(sharedPtr->data()));
        }
        else if (strcmp(name, "QPointer<ito::AddInDataIO>") == 0 || \
            strcmp(name, "QPointer<ito::AddInActuator>") == 0 || \
            strcmp(name, "QPointer<ito::AddInBase>") == 0)
        {
            QPointer<ito::AddInBase> *ptr = (QPointer<ito::AddInBase>*)data;
            if (ptr == NULL)
            {
                PyErr_SetString(PyExc_TypeError, "The given QPointer is NULL");
                return NULL;
            }
            if (ptr->data() == NULL)
            {
                Py_RETURN_NONE;
            }
            return AddInBaseToPyObject(ptr->data());
        }
        else if (strcmp(name, "QVector<int>") == 0)
        {
            QVector<int> *temp2 = (QVector<int>*)data;
            PyObject *temp = PyTuple_New(temp2->size());
            for (Py_ssize_t i = 0; i < temp2->size(); ++i)
            {
                PyTuple_SetItem(temp, i, PyLong_FromLong(temp2->at(i)));
            }
            return temp;
        }
		else if (strcmp(name, "QList<int>") == 0)
		{
			QList<int> *temp2 = (QList<int>*)data;
			PyObject *temp = PyTuple_New(temp2->size());
			for (Py_ssize_t i = 0; i < temp2->size(); ++i)
			{
				PyTuple_SetItem(temp, i, PyLong_FromLong(temp2->at(i)));
			}
			return temp;
		}
        else if (strcmp(name, "QVector<double>") == 0)
        {
            QVector<double> *temp2 = (QVector<double>*)data;
            PyObject *temp = PyTuple_New(temp2->size());
            for (Py_ssize_t i = 0; i < temp2->size(); ++i)
            {
                PyTuple_SetItem(temp, i, PyFloat_FromDouble(temp2->at(i)));
            }
            return temp;
        }
		else if (strcmp(name, "QList<double>") == 0)
		{
			QList<double> *temp2 = (QList<double>*)data;
			PyObject *temp = PyTuple_New(temp2->size());
			for (Py_ssize_t i = 0; i < temp2->size(); ++i)
			{
				PyTuple_SetItem(temp, i, PyFloat_FromDouble(temp2->at(i)));
			}
			return temp;
		}
        else if (strcmp(name, "QVector<float>") == 0)
        {
            QVector<float> *temp2 = (QVector<float>*)data;
            PyObject *temp = PyTuple_New(temp2->size());
            for (Py_ssize_t i = 0; i < temp2->size(); ++i)
            {
                PyTuple_SetItem(temp, i, PyFloat_FromDouble(temp2->at(i)));
            }
            return temp;
        }
		else if (strcmp(name, "QList<float>") == 0)
		{
			QList<float> *temp2 = (QList<float>*)data;
			PyObject *temp = PyTuple_New(temp2->size());
			for (Py_ssize_t i = 0; i < temp2->size(); ++i)
			{
				PyTuple_SetItem(temp, i, PyFloat_FromDouble(temp2->at(i)));
			}
			return temp;
		}
        else if (strcmp(name, "QVector2D") == 0)
        {
            QVector2D *temp2 = (QVector2D*)data;
            PyObject *temp = PyTuple_New(2);
            PyTuple_SetItem(temp, 0, PyFloat_FromDouble(temp2->x()));
            PyTuple_SetItem(temp, 1, PyFloat_FromDouble(temp2->y()));
            return temp;
        }
        else if (strcmp(name, "QVector3D") == 0)
        {
            QVector3D *temp2 = (QVector3D*)data;
            PyObject *temp = PyTuple_New(3);
            PyTuple_SetItem(temp, 0, PyFloat_FromDouble(temp2->x()));
            PyTuple_SetItem(temp, 1, PyFloat_FromDouble(temp2->y()));
            PyTuple_SetItem(temp, 2, PyFloat_FromDouble(temp2->z()));
            return temp;
        }
        else if (strcmp(name, "QVector4D") == 0)
        {
            QVector4D *temp2 = (QVector4D*)data;
            PyObject *temp = PyTuple_New(4);
            PyTuple_SetItem(temp, 0, PyFloat_FromDouble(temp2->x()));
            PyTuple_SetItem(temp, 1, PyFloat_FromDouble(temp2->y()));
            PyTuple_SetItem(temp, 2, PyFloat_FromDouble(temp2->z()));
            PyTuple_SetItem(temp, 2, PyFloat_FromDouble(temp2->w()));
            return temp;
        }
        else if (strcmp(name, "QVector<ito::Shape>") == 0)
        {
            QVector<ito::Shape> *temp2 = (QVector<ito::Shape>*)data;
            PyObject *temp = PyTuple_New(temp2->size());
            for (int i = 0; i < temp2->size(); ++i)
            {
                PyTuple_SetItem(temp, i, ito::PythonShape::createPyShape(temp2->at(i))); //steals reference
            }
            return temp;
        }
        else if (strcmp(name, "QList<ito::Shape>") == 0)
        {
        QList<ito::Shape>* temp2 = (QList<ito::Shape>*)data;
        PyObject* temp = PyTuple_New(temp2->size());
        for (int i = 0; i < temp2->size(); ++i)
        {
            PyTuple_SetItem(temp, i, ito::PythonShape::createPyShape(temp2->at(i))); //steals reference
        }
        return temp;
        }
        else if (strcmp(name, "ito::Shape") == 0)
        {
            return ito::PythonShape::createPyShape(*((ito::Shape*)data));
        }
        else if (strcmp(name, "Qt::CheckState") == 0)
        {
            return PyLong_FromLong(*((Qt::CheckState*)data));
        }
        else if (strcmp(name, "Qt::ItemFlags") == 0)
        {
            return PyLong_FromLong(*((Qt::ItemFlags*)data));
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "The given Qt-type is not registered in the Qt-MetaType system.");
        return NULL;
    }

    // since Qt6 it seems that an enumeration value is now returned as
    // user-defined meta type. Since we cannot transfer every available
    // enum, the enumeration is tried to be converted to an integer
    // variable and returned as Python integer value.
    auto flags = QMetaType::typeFlags(type);

    if (flags.testFlag(QMetaType::IsEnumeration))
    {
        long val;

        if (QMetaType::convert(data, type, &val, QMetaType::Long))
        {
            return PyLong_FromLong(val);
        }
    }

    PyErr_SetString(PyExc_TypeError, "The given Qt-type cannot be parsed into an appropriate python type.");
    return NULL;
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::QByteArrayToPyUnicode(const QByteArray &ba, const char *errors)
{
    return ByteArrayToPyUnicode(ba.data(), ba.length(), errors);
}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
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
        return PyUnicode_DecodeUTF8(byteArray, len, errors);
    case latin_1:
    case iso_8859_1:
        return PyUnicode_DecodeLatin1(byteArray, len, errors);
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

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::QByteArrayUtf8ToPyUnicodeSecure(const QByteArray &ba, const char *errors /*= "replace"*/)
{
    PyObject *temp = QByteArrayUtf8ToPyUnicode(ba, errors);

    if (temp == nullptr)
    {
        PyErr_Clear();
        temp = ByteArrayToPyUnicode("<encoding error>");
    }

    return temp;
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::QByteArrayUtf8ToPyUnicode(const QByteArray &ba, const char *errors)
{
    int bo;
    int len = ba.size();
    const char* data = ba.constData();

    return PyUnicode_DecodeUTF8(data, len, errors);
}

//-------------------------------------------------------------------------------------
/*static*/ PyObject* PythonQtConversion::PyUnicodeToPyByteObject(PyObject *unicode, const char *errors /*= "replace"*/)
{
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
#if defined(WIN) || defined(WIN32) || defined(_WIN64)
    case mbcs:
        return PyUnicode_AsMBCSString(unicode);
#endif
    case ascii:
        return PyUnicode_AsASCIIString(unicode);
    case utf_16:
        return PyUnicode_AsEncodedString(unicode, "utf_16", errors);
    case utf_16_LE:
        return PyUnicode_AsEncodedString(unicode, "utf_16_le", errors);
    case utf_16_BE:
        return PyUnicode_AsEncodedString(unicode, "utf_16_be", errors);
    case utf_32:
        return PyUnicode_AsEncodedString(unicode, "utf_32", errors);
    case utf_32_LE:
        return PyUnicode_AsEncodedString(unicode, "utf_32_le", errors);
    case utf_32_BE:
        return PyUnicode_AsEncodedString(unicode, "utf_32_be", errors);
    case other:
    default:
        {
            PyObject *res = PyUnicode_AsEncodedString(unicode, textEncodingName.data(), errors);
            return res;
        }
    }
}

//-------------------------------------------------------------------------------------
// the following method is only called by baseObjectDeleterDataObject within a QtConcurrent::run worker thread
void safeDecrefPyObject(PyObject *obj)
{
    if (PyGILState_Check())
    {
        Py_DECREF(obj);
    }
    else
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_DECREF(obj);
        PyGILState_Release(gstate);
    }
}


//-------------------------------------------------------------------------------------
/*static*/ void PythonQtConversion::baseObjectDeleterDataObject(ito::DataObject *sharedObject)
{
    QHash<char*, PyObject*>::iterator i = m_pyBaseObjectStorage.find((char*)sharedObject);
    if (i != m_pyBaseObjectStorage.end())
    {
        if (i.value())
        {
            if (PyGILState_Check())
            {
                Py_DECREF(i.value());
            }
            else
            {
                //the current thread has no Python GIL. However, it might be
                //that the GIL is currently hold by another thread, which has called
                //the current thread, such that directly waiting for the GIL here might
                //lead to a dead-lock. Therefore, we open a worker thread to finally delete the guarded base object!
                QtConcurrent::run(safeDecrefPyObject, i.value());
            }
        }

        m_pyBaseObjectStorage.erase(i);
    }

    delete sharedObject;
}

//-------------------------------------------------------------------------------------
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
    m_signature = method.methodSignature();
    QByteArray sig(method.methodSignature());

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

} //end namespace ito
