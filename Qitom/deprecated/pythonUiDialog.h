/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2018, Institut fuer Technische Optik (ITO), 
   Universität Stuttgart, Germany 
 
   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef PYTHONUIDIALOG_H
#define PYTHONUIDIALOG_h

////python
//// see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
//#ifdef _DEBUG
//    #undef _DEBUG
//    #ifdef linux
//        #include "Python.h"
//    #else
//        #include "include\Python.h"
//    #endif
//    #define _DEBUG
//#else
//    #ifdef linux
//        #include "Python.h"
//    #else
//        #include "include\Python.h"
//    #endif
//#endif

#include "pythonCommon.h"
#include "pythonQtConversion.h"
#include "pythonQtSignalMapper.h"

#include <qstring.h>
#include <qvariant.h>
#include <qobject.h>
#include <qhash.h>


namespace ito 
{

class timerCallback : public QObject
{
    Q_OBJECT
    public:
        timerCallback() : m_function(NULL), m_boundedInstance(NULL), m_boundedMethod(0), m_callbackArgs(0) {};
        ~timerCallback() {};
        PyObject *m_function; //pyFunctionObject
        PyObject *m_boundedInstance; //self if bounded method, else null
//        PyObject m_callbackFunc;
//        PyObject m_callbackArgs;
        PyObject *m_callbackArgs;
        int m_boundedMethod;

    public slots:
        void timeout(); 
};

class PythonUiDialog
{
public:

    //-------------------------------------------------------------------------------------------------
    // typedefs
    //------------------------------------------------------------------------------------------------- 
    typedef struct
    {
        PyObject_HEAD
        int uiHandle;
        int winType;
        int buttonBarType;
        bool childOfMainWindow;
        PyObject *dialogButtons;
        char* filename;
        PythonQtSignalMapper *signalMapper;
        PyObject *weakreflist;
    }
    PyUiDialog;

    typedef struct
    {
        PyObject_HEAD
        QTimer *timer;
        PyObject* base;
        timerCallback *callbackFunc;
    }
    PyUiTimer;

    #define PyUiDialog_Check(op) PyObject_TypeCheck(op, &PythonUiDialog::PyUiDialogType)

    //-------------------------------------------------------------------------------------------------
    // Timer
    //------------------------------------------------------------------------------------------------- 
    static void PyUiTimer_dealloc(PyUiTimer *self);
    static PyObject *PyUiTimer_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyUiTimer_init(PyUiTimer *self, PyObject *args, PyObject *kwds);
    static PyObject *PyUiTimer_name(PyUiTimer *self);
    static PyObject *PyUiTimer_repr(PyUiTimer *self);

    static PyGetSetDef  PyUiTimer_getseters[];
    static PyMemberDef  PyUiTimer_members[];
    static PyMethodDef  PyUiTimer_methods[];
    static PyTypeObject PyUiTimerType;
    static PyModuleDef  PyUiTimerModule;
    static PyObject *PyUiTimer_start(PyUiTimer *self);
    static PyObject *PyUiTimer_stop(PyUiTimer *self);
    static PyObject *PyUiTimer_isActive(PyUiTimer *self);
    static PyObject *PyUiTimer_setInterval(PyUiTimer *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //------------------------------------------------------------------------------------------------- 
    static void PyUiDialog_dealloc(PyUiDialog *self);
    static PyObject *PyUiDialog_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyUiDialog_init(PyUiDialog *self, PyObject *args, PyObject *kwds);


    //-------------------------------------------------------------------------------------------------
    // general members
    //------------------------------------------------------------------------------------------------- 
    static PyObject *PyUiDialog_name(PyUiDialog *self);
    static PyObject* PyUiDialog_repr(PyUiDialog *self);

    static PyObject *PyUiDialog_show(PyUiDialog *self, PyObject *args);
    static PyObject *PyUiDialog_hide(PyUiDialog *self);
    static PyObject *PyUiDialog_isVisible(PyUiDialog *self);

    static PyObject *PyUiDialog_getPropertyInfo(PyUiDialog *self, PyObject *args);
    static PyObject *PyUiDialog_getProperties(PyUiDialog *self, PyObject *args);
    static PyObject *PyUiDialog_setProperties(PyUiDialog *self, PyObject *args);
    static PyObject *PyUiDialog_getattro(PyUiDialog *self, PyObject *args);
    static PyObject *PyUiDialog_setattro(PyUiDialog *self, PyObject *args);
    static PyObject *PyUiDialog_getuimetaobject(PyUiDialog *self);

    //-------------------------------------------------------------------------------------------------
    // static members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyUiDialog_getDouble(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_getInt(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_getItem(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_getText(PyUiDialog *self, PyObject *args, PyObject *kwds);

    static PyObject* PyUiDialog_msgInformation(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_msgQuestion(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_msgWarning(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_msgCritical(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_msgGeneral(PyUiDialog *self, PyObject *args, PyObject *kwds, int type);

    static PyObject* PyUiDialog_getExistingDirectory(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_getOpenFileName(PyUiDialog *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiDialog_getSaveFileName(PyUiDialog *self, PyObject *args, PyObject *kwds);

    static PyObject* PyUiDialog_createNewAlgoWidget(PyUiDialog *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // helper
    //-------------------------------------------------------------------------------------------------
    //static QString pythonStringAsString(PyObject* pyObj);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //------------------------------------------------------------------------------------------------- 
    static PyGetSetDef  PyUiDialog_getseters[];
    static PyMemberDef  PyUiDialog_members[];
    static PyMethodDef  PyUiDialog_methods[];
    static PyTypeObject PyUiDialogType;
    static PyModuleDef  PyUiDialogModule;

    static void PyUiDialog_addTpDict(PyObject *tp_dict);
};


class PyUiDialogMetaObject
{
    public:

    //-------------------------------------------------------------------------------------------------
    // typedefs
    //------------------------------------------------------------------------------------------------- 
    typedef struct
    {
        PyObject_HEAD
        PythonUiDialog::PyUiDialog *dialog;
        char* objName;
        unsigned int objectID;
        int methodDescriptionListLoaded;
        MethodDescriptionList* methodList;
        QMultiHash<QString, unsigned int>* methodListHash;
        PyObject *weakreflist;
    }
    PyMetaObject;

    #define PyMetaObject_Check(op) PyObject_TypeCheck(op, &PyUiDialogMetaObject::PyMetaObject)

    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //------------------------------------------------------------------------------------------------- 

    static void PyMetaObject_dealloc(PyMetaObject *self);
    static PyObject *PyMetaObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyMetaObject_init(PyMetaObject *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // mapping members
    //-------------------------------------------------------------------------------------------------
    static int PyMetaObject_mappingLength(PyMetaObject* self);
    static PyObject* PyMetaObject_mappingGetElem(PyMetaObject* self, PyObject* key);
    static int PyMetaObject_mappingSetElem(PyMetaObject* self, PyObject* key, PyObject* value);

    //-------------------------------------------------------------------------------------------------
    // general members
    //------------------------------------------------------------------------------------------------- 
    static PyObject *PyMetaObject_name(PyMetaObject *self);
    static PyObject* PyMetaObject_repr(PyMetaObject *self);
    static PyObject* PyMetaObject_call(PyMetaObject *self, PyObject* args);
    static PyObject* PyMetaObject_connect(PyMetaObject *self, PyObject* args);
    static PyObject* PyMetaObject_connectKeyboardInterrupt(PyMetaObject *self, PyObject* args);
    static PyObject* PyMetaObject_disconnect(PyMetaObject *self, PyObject* args);
    static PyObject *PyMetaObject_getProperties(PyMetaObject *self, PyObject *args);
    static PyObject *PyMetaObject_setProperties(PyMetaObject *self, PyObject *args);
    static PyObject *PyMetaObject_getattro(PyMetaObject *self, PyObject *args);
    static PyObject *PyMetaObject_setattro(PyMetaObject *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // helpers
    //------------------------------------------------------------------------------------------------- 
    static bool loadMethodDescriptionList(PyMetaObject *self);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //------------------------------------------------------------------------------------------------- 
    static PyMemberDef  PyMetaObject_members[];
    static PyGetSetDef  PyMetaObject_getseters[];
    static PyMethodDef  PyMetaObject_methods[];
    static PyTypeObject PyMetaObjectType;
    static PyModuleDef  PyMetaObjectModule;
    static PyMappingMethods PyMetaObject_mappingProtocol;

    static void PyMetaObject_addTpDict(PyObject *tp_dict);
};

}; //end namespace ito

#endif