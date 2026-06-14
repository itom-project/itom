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

#include "pythonFont.h"

#include "../global.h"
#include "pythonQtConversion.h"
#include <qfontdatabase.h>



//-------------------------------------------------------------------------------------

namespace ito
{

//-------------------------------------------------------------------------------------
void PythonFont::PyFont_addTpDict(PyObject * tp_dict)
{
    PyObject *value;

    value = Py_BuildValue("i",QFont::Light);
    PyDict_SetItemString(tp_dict, "Light", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",QFont::Normal);
    PyDict_SetItemString(tp_dict, "Normal", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",QFont::DemiBold);
    PyDict_SetItemString(tp_dict, "DemiBold", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",QFont::Bold);
    PyDict_SetItemString(tp_dict, "Bold", value);
    Py_DECREF(value);

    value = Py_BuildValue("i",QFont::Black);
    PyDict_SetItemString(tp_dict, "Black", value);
    Py_DECREF(value);
}

//-------------------------------------------------------------------------------------
void PythonFont::PyFont_dealloc(PyFont* self)
{
    DELETE_AND_SET_NULL(self->font);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//-------------------------------------------------------------------------------------
PyObject* PythonFont::PyFont_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyFont* self = (PyFont *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->font = NULL;
    }

    return (PyObject *)self;
};

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyFont_doc,"font(family, pointSize = 0, weight = -1, italic = False) -> font \n\
\n\
Creates a font object. \n\
\n\
This class is a wrapper for the class `QFont` of the Qt framework. \n\
It provides possibilities for creating a font type. \n\
\n\
Parameters \n\
---------- \n\
family : str \n\
    The family name may optionally also include a foundry name, e.g. \"Helvetica [Cronyx]\". \n\
    If the family is available from more than one foundry and the foundry isn't specified, \n\
    an arbitrary foundry is chosen. If the family isn't available a family will be set \n\
    using a best-matching algorithm. \n\
pointSize : int, optional \n\
    If pointSize is zero or negative, the point size of the font is set to a \n\
    system-dependent default value. Generally, this is 12 points. \n\
weight : int, optional \n\
    Weighting scale from 0 to 99, e.g. ``font.Light``, ``font.Normal`` (default), \n\
    ``font.DemiBold``, ``font.Bold``, ``font.Black``. \n\
italic : bool, optional \n\
    Defines if font is italic or not (default)");
int PythonFont::PyFont_init(PyFont *self, PyObject *args, PyObject * kwds)
{
    int pointSize = 0;
    int weight = -1;
    bool italic = false;
    const char* family = NULL;

    const char *kwlist[] = {"family", "pointSize", "weight", "italic", NULL};

    if (args == NULL && kwds == NULL)
    {
        return 0; //call from createPyFont
    }

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "s|iiB", const_cast<char**>(kwlist), &(family), &(pointSize), &(weight), &(italic)))
    {
        return -1;
    }

    if (weight >= 100)
    {
        PyErr_SetString(PyExc_ValueError, "weight must be in the range [0, 99]");
    }

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    if (weight >= 0)
    {
        // Qt6: weight is between 0 and 1000.
        weight *= 10;
    }
#endif

    QFontInfo info(QFont(family, pointSize, weight, italic));
    self->font = new QFont(info.family(), info.pointSize(), info.weight(), info.italic());

    return 0;
};

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonFont::createPyFont(const QFont &font)
{
    PyFont* result = (PyFont*)PyObject_Call((PyObject*)&PyFontType, NULL, NULL);
    if(result != NULL)
    {
        result->font = new QFont(font);
        return (PyObject*)result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return NULL;
    }
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonFont::PyFont_repr(PyFont *self)
{
    PyObject *result;
    if(self->font == NULL)
    {
        result = PyUnicode_FromFormat("font(NULL)");
    }
    else
    {
        result = PyUnicode_FromFormat(
            "font(%s, %ipt, weight: %i)",
            self->font->family().toUtf8().data(),
            self->font->pointSize(),
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
            ((int)self->font->weight()) / 10
#else
            self->font->weight()
#endif
        );
    }
    return result;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonFont::PyFont_Reduce(PyFont *self, PyObject *args)
{
    PyObject *stateTuple = NULL;

    if(self->font)
    {
        QByteArray ba = self->font->toString().toLatin1();
        stateTuple = PyBytes_FromStringAndSize( ba.data(), ba.size() );
    }
    else
    {
        Py_INCREF(Py_None);
        stateTuple = Py_None;
    }

    //the stateTuple is simply a byte array with the stream data of the QRegion.
    PyObject *tempOut = Py_BuildValue(
        "(O(s)O)",
        Py_TYPE(self),
        self->font ? self->font->family().toLatin1().data() : "",
        stateTuple
    );
    Py_XDECREF(stateTuple);

    return tempOut;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonFont::PyFont_SetState(PyFont *self, PyObject *args)
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
    else
    {
        QByteArray ba( PyBytes_AS_STRING(data), PyBytes_GET_SIZE(data) );

        if(self->font)
        {
            self->font->fromString(QLatin1String(ba));
        }
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(font_getFamily_doc,
"str : gets / sets the family name of the font. \n\
\n\
The name is case insensitive. It may optionally also include a foundry name, \n\
e.g.  \"Helvetica [Cronyx]\". If the family is available from more than one \n\
foundry and the foundry isn't specified, an arbitrary foundry is chosen. If \n\
the family isn't available a family will be set using a font matching algorithm.");
PyObject* PythonFont::PyFont_getFamily(PyFont *self, void * /*closure*/)
{
    if(!self || self->font == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "font is not available");
        return NULL;
    }

    return PythonQtConversion::QStringToPyObject(self->font->family());
}

int PythonFont::PyFont_setFamily(PyFont *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    QString family = PythonQtConversion::PyObjGetString(value, true, ok);
    if (ok)
    {
        QFont temp = *self->font;
        temp.setFamily(family);
        QFontInfo info(temp);
        self->font->setFamily(info.family());
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the font family as string.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(font_getWeight_doc,
"int : gets or sets the weight of the font. \n\
\n\
This should be one of the constant values ``font.Light``, ``font.Normal``, \n\
``font.DemiBold``, ``font.Bold``, ``font.Black`` enumeration or any value \n\
in the range [0, 99].");
PyObject* PythonFont::PyFont_getWeight(PyFont *self, void * /*closure*/)
{
    if(!self || self->font == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "font is not available");
        return NULL;
    }

    int weight = self->font->weight();

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    weight /= 10;
#endif

    return PyLong_FromLong(weight);
}

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
QFont::Weight fontWeightConversion(int weight)
{
    if (weight <= 10)
    {
        return QFont::Thin;
    }
    else if (weight <= 20)
    {
        return QFont::ExtraLight;
    }
    else if (weight <= 30)
    {
        return QFont::Light;
    }
    else if (weight <= 40)
    {
        return QFont::Normal;
    }
    else if (weight <= 50)
    {
        return QFont::Medium;
    }
    else if (weight <= 60)
    {
        return QFont::DemiBold;
    }
    else if (weight <= 70)
    {
        return QFont::Bold;
    }
    else if (weight <= 80)
    {
        return QFont::ExtraBold;
    }
    else
    {
        return QFont::Black;
    }
}
#else
int fontWeightConversion(int weight)
{
    return weight;
}
#endif

int PythonFont::PyFont_setWeight(PyFont *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    int weight = PythonQtConversion::PyObjGetInt(value, true, ok);

    if (ok)
    {
        if (weight <= 99)
        {
            QFont temp = *self->font;
            temp.setWeight(fontWeightConversion(weight));
            QFontInfo info(temp);
            weight = info.weight(); // Qt5: [0,99], Qt6: [0, 999]
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
            weight /= 10;
#endif
            self->font->setWeight(fontWeightConversion(weight));
            return 0;
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "weight must be in range [0,99].");
            return -1;
        }
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the weight as uint.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(font_getPointSize_doc,
"int : gets or sets the point size (> 0) of the font.");
PyObject* PythonFont::PyFont_getPointSize(PyFont *self, void * /*closure*/)
{
    if(!self || self->font == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "font is not available");
        return NULL;
    }

    return PyLong_FromLong(self->font->pointSize());
}

int PythonFont::PyFont_setPointSize(PyFont *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    int pointSize = PythonQtConversion::PyObjGetInt(value, true, ok);

    if (ok)
    {
        QFont temp = *self->font;
        temp.setPointSize(pointSize);
        QFontInfo info(temp);
        self->font->setPointSize(info.pointSize());
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the point size as int.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(font_getItalic_doc,
"bool : gets or sets the italic attribute of the font.");
PyObject* PythonFont::PyFont_getItalic(PyFont *self, void * /*closure*/)
{
    if(!self || self->font == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "font is not available");
        return NULL;
    }

    if (self->font->italic())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

int PythonFont::PyFont_setItalic(PyFont *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    bool s = PythonQtConversion::PyObjGetBool(value, true, ok);
    if (ok)
    {
        QFont temp = *self->font;
        temp.setItalic(s);
        QFontInfo info(temp);
        self->font->setItalic(info.italic());
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the italic attribute as boolean.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(font_getStrikeOut_doc,
"bool : gets or sets the strikeout attribute of the font.");
PyObject* PythonFont::PyFont_getStrikeOut(PyFont *self, void * /*closure*/)
{
    if(!self || self->font == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "font is not available");
        return NULL;
    }

    if (self->font->strikeOut())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

int PythonFont::PyFont_setStrikeOut(PyFont *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    bool s = PythonQtConversion::PyObjGetBool(value, true, ok);
    if (ok)
    {
        QFont temp = *self->font;
        temp.setStrikeOut(s);
        QFontInfo info(temp);
        self->font->setStrikeOut(info.strikeOut());
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the strikeOut attribute as boolean.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(font_getUnderline_doc,
"bool : gets or sets the underline attribute of the font.");
PyObject* PythonFont::PyFont_getUnderline(PyFont *self, void * /*closure*/)
{
    if(!self || self->font == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "font is not available");
        return NULL;
    }

    if (self->font->underline())
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

int PythonFont::PyFont_setUnderline(PyFont *self, PyObject *value, void * /*closure*/)
{
    bool ok;
    bool s = PythonQtConversion::PyObjGetBool(value, true, ok);
    if (ok)
    {
        QFont temp = *self->font;
        temp.setUnderline(s);
        QFontInfo info(temp);
        self->font->setUnderline(info.underline());
        return 0;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "error interpreting the underline attribute as boolean.");
        return -1;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyFont_isFamilyInstalled_DOC, "isFamilyInstalled(family) -> bool \n\
\n\
Checks if the given font family is installed on this computer. \n\
\n\
Parameters \n\
---------- \n\
family : str \n\
    The name of the font family that should be checked \n\
\n\
Returns \n\
------- \n\
installed : bool \n\
    ``True`` if family is installed, else ``False``.");
PyObject* PythonFont::PyFont_isFamilyInstalled(PyFont * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "family", NULL };
    const char *family = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &(family)))
    {
        return NULL;
    }

    if (QFontDatabase().families().contains(family, Qt::CaseInsensitive))
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(pyFont_installedFontFamilies_DOC, "installedFontFamilies() -> List[str] \n\
\n\
Returns a list of all installed font families. \n\
\n\
Returns \n\
------- \n\
list of str : \n\
    list of the names of all installed font families");
PyObject* PythonFont::PyFont_installedFontFamilies(PyFont * /*self*/)
{
    return PythonQtConversion::QStringListToPyList(QFontDatabase().families());
}

//-----------------------------------------------------------------------------
PyGetSetDef PythonFont::PyFont_getseters[] = {
    {"family", (getter)PyFont_getFamily,       (setter)PyFont_setFamily, font_getFamily_doc, NULL},
    {"pointSize", (getter)PyFont_getPointSize, (setter)PyFont_setPointSize, font_getPointSize_doc, NULL},
    {"weight", (getter)PyFont_getWeight,       (setter)PyFont_setWeight, font_getWeight_doc, NULL},
    {"italic", (getter)PyFont_getItalic,       (setter)PyFont_setItalic, font_getItalic_doc, NULL},
    {"underline", (getter)PyFont_getUnderline, (setter)PyFont_setUnderline, font_getUnderline_doc, NULL},
    {"strikeOut", (getter)PyFont_getStrikeOut, (setter)PyFont_setStrikeOut, font_getStrikeOut_doc, NULL},
    {NULL}  /* Sentinel */
};

//-----------------------------------------------------------------------------
PyMethodDef PythonFont::PyFont_methods[] = {
    {"__reduce__", (PyCFunction)PyFont_Reduce, METH_VARARGS,      "__reduce__ method for handle pickling commands"},
    {"__setstate__", (PyCFunction)PyFont_SetState, METH_VARARGS,  "__setstate__ method for handle unpickling commands"},
    { "isFamilyInstalled", (PyCFunction)PyFont_isFamilyInstalled, METH_VARARGS | METH_KEYWORDS | METH_STATIC, pyFont_isFamilyInstalled_DOC },
    { "installedFontFamilies", (PyCFunction)PyFont_installedFontFamilies, METH_NOARGS | METH_STATIC, pyFont_installedFontFamilies_DOC },
    {NULL}  /* Sentinel */
};




//-----------------------------------------------------------------------------
PyModuleDef PythonFont::PyFontModule = {
    PyModuleDef_HEAD_INIT, "font", "Font object.", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-----------------------------------------------------------------------------
PyTypeObject PythonFont::PyFontType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.font",             /* tp_name */
    sizeof(PyFont),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyFont_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyFont_repr,                         /* tp_repr */
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
    PyFont_doc,              /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    PyFont_methods,          /* tp_methods */
    0,                         /* tp_members */
    PyFont_getseters,        /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyFont_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PyFont_new               /* tp_new */
};



} //end namespace ito
