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

#include "pythonUi.h"

#include "structmember.h"

#include "../global.h"
#include "../organizer/uiOrganizer.h"
#include "../../AddInManager/addInManager.h"
#include <qcoreapplication.h>

#include "pythonQtConversion.h"
#include "pythonFigure.h"
#include "pythonPlotItem.h"
#include "pythonProgressObserver.h"
#include "AppManagement.h"

#include <qsharedpointer.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qelapsedtimer.h>

QHash<QByteArray, QSharedPointer<ito::MethodDescriptionList> > ito::PythonUi::methodDescriptionListStorage;

namespace ito
{
// ------------------------------------------------------------------------------------
//
//  pyUiItem
//
// ------------------------------------------------------------------------------------
void PythonUi::PyUiItem_dealloc(PyUiItem* self)
{
    Py_XDECREF(self->baseItem);
    DELETE_AND_SET_NULL_ARRAY(self->objName);
    DELETE_AND_SET_NULL_ARRAY(self->widgetClassName);
    self->methodList = NULL; //this has only been a borrowed reference

    //clear weak reference to this object
    if (self->weakreflist != NULL)
    {
        PyObject_ClearWeakRefs((PyObject *) self);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUiItem_new(PyTypeObject *type, PyObject * /*args*/, PyObject * /*kwds*/)
{
    PyUiItem* self = (PyUiItem *)type->tp_alloc(type, 0);

    if (self != NULL)
    {
        self->baseItem = NULL;
        self->objName = NULL;
        self->widgetClassName = NULL;
        self->objectID = 0; //invalid
        self->methodList = NULL;
        self->weakreflist = NULL;
    }

    return (PyObject *)self;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemInit_doc,"uiItem(objectID, objName, widgetClassName, parentObj = None) -> uiItem \\\n\
uiItem(parentObj, objName) -> uiItem \n\
\n\
Base class that represents any widget or layout of an user interface. \n\
\n\
This class represents any widget (graphical, interactive element like a button or \n\
checkbox) on a graphical user interface. An object of this class provides many \n\
functionalities given by the underlying Qt system. For instance, it is posible to \n\
call a public slot of the corresponding widget, connect signals to specific python \n\
methods or functions or change properties of the widget represeted by this object. \n\
\n\
The overall dialog or window as main element of a graphical user interface itself are \n\
instances of the class :class:`ui`. However, they are derived from :class:`uiItem`, \n\
since dialogs or windows internally are widgets as well. \n\
\n\
Widgets, placed at a user interface using the Qt Designer, can be referenced by an \n\
:class:`uiItem` object by their specific ``objectName``, assigned in the Qt Designer \n\
as well. As an example, a simple dialog with one button is created and the text of \n\
the button (objectName: btn) is set to OK: :: \n\
    \n\
    dialog = ui('filename.ui', type=ui.TYPEDIALOG) \n\
    button = dialog.btn #here the reference to the button is obtained \n\
    button[\"text\"] = \"OK\" #set the property text of the button \n\
    \n\
Information about available properties, signals and slots can be obtained using the \n\
method :meth:`uiItem.info`. For more information about creating customized user \n\
interfaces, reference widgets and layouts etc, see the section :ref:`qtdesigner`. \n\
\n\
Parameters \n\
---------- \n\
objectID : int \n\
    is the itom internal identifier number for the widget or layout to be wrapped. \n\
objName : str \n\
    is the ``objectName`` property of the wrapped widget or layout. \n\
widgetClassName : str \n\
    is the Qt class name of the wrapped widget or layout (see :meth:`getClassName`). \n\
parentObj : uiItem \n\
    is the parent :class:`uiItem` of this wrapped widget or layout. \n\
\n\
Returns \n\
------- \n\
uiItem \n\
    is the new :class:`uiItem` object that wraps the indicated widget or layout. \n\
\n\
Notes \n\
----- \n\
It is not intended to directly instantiate this class. Either create a user interface \n\
using the class :class:`ui` or obtain a reference to an existing widget (this is then \n\
an instance of :class:`uiItem`) using the dot-operator of a parent widget or the entire \n\
user interface.");
int PythonUi::PyUiItem_init(PyUiItem *self, PyObject *args, PyObject *kwds)
{
    ito::RetVal retValue = retOk;
    QSharedPointer<unsigned int> objectID(new unsigned int());
    *objectID = 0;
    QSharedPointer<QByteArray> widgetClassNameBA(new QByteArray());
    const char *objName = NULL;
    const char *widgetClassName = NULL;
    PyObject *parentObj = NULL;
    PythonUi::PyUiItem *parentItem = NULL;

    const char *kwlist1[] = { "objectID", "objName", "widgetClassName", "parentObj", NULL };
    const char *kwlist2[] = { "parentObj", "objName" , NULL };

    if (PyArg_ParseTupleAndKeywords(args, kwds, "Iss|O!", const_cast<char**>(kwlist1), &(*objectID), &objName, &widgetClassName, &PythonUi::PyUiItemType, &parentObj))
    {
        self->baseItem = parentObj;
        Py_XINCREF(self->baseItem); //if parent available increment its reference
        DELETE_AND_SET_NULL_ARRAY(self->objName);
        self->objName = new char[strlen(objName) + 1];
        strcpy_s(self->objName, strlen(objName) + 1, objName);
        DELETE_AND_SET_NULL_ARRAY(self->widgetClassName);
        self->widgetClassName = new char[strlen(widgetClassName)+1];
        strcpy_s(self->widgetClassName, strlen(widgetClassName)+1, widgetClassName);
        self->objectID = *objectID;
        *widgetClassNameBA = widgetClassName;
    }
    else if(PyErr_Clear(), PyArg_ParseTupleAndKeywords(args, kwds, "O!s", const_cast<char**>(kwlist2), &PythonUi::PyUiItemType, &parentObj, &objName))
    {
        UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

        if (uiOrga == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
            return -1;
        }

        parentItem = (PythonUi::PyUiItem*)parentObj;
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

        QMetaObject::invokeMethod(
            uiOrga,
            "getChildObject3",
            Q_ARG(uint, static_cast<unsigned int>(parentItem->objectID)),
            Q_ARG(QString, QString(objName)),
            Q_ARG(QSharedPointer<uint>, objectID),
            Q_ARG(QSharedPointer<QByteArray>, widgetClassNameBA),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        ); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

        locker.getSemaphore()->wait(-1);
        retValue += locker.getSemaphore()->returnValue;

        if(*objectID == 0)
        {
            if (retValue.hasErrorMessage())
            {
                PyErr_SetString(PyExc_RuntimeError, retValue.errorMessage());
            }
            else
            {
                PyErr_SetString(PyExc_RuntimeError, "attribute is no widget name of this user interface");
            }
            return -1;
        }
        else
        {
            Py_XINCREF(parentObj);
            self->baseItem = parentObj;
            DELETE_AND_SET_NULL_ARRAY(self->objName);
            self->objName = new char[strlen(objName)+1];
            strcpy_s(self->objName, strlen(objName)+1, objName);
            DELETE_AND_SET_NULL_ARRAY(self->widgetClassName);
            self->widgetClassName = new char[widgetClassNameBA->size()+1];
            strcpy_s(self->widgetClassName, widgetClassNameBA->size()+1, widgetClassNameBA->data());
            self->objectID = *objectID;
        }
    }
    else
    {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "Arguments must be an object of type ui followed by an object name (string).");
        return -1;
    }

    self->methodList = NULL;
    //if the following if-block is commented, the methodDescriptionList will be delay-loaded at the time when it is needed for the first time.
    /*if(loadMethodDescriptionList(self) == false)
    {
        PyErr_SetString(PyExc_TypeError, "MethodDescriptionList for this UiItem could not be loaded");
        return -1;
    }*/

    return 0;
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUiItem_repr(PyUiItem *self)
{
    if(self->objName && self->widgetClassName)
    {
        return PyUnicode_FromFormat("UiItem(class: %s, name: %s)", self->widgetClassName, self->objName);
    }
    else
    {
        return PyUnicode_FromString("UiItem(<unknown>)");
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetClassName_doc, "getClassName() -> str \n\
\n\
Returns the Qt class name of this uiItem (widget or layout).  \n\
\n\
Every :class:`uiItem` wraps a widget or layout of the user interface. \n\
This method returns the class name of this item, as it is given by the \n\
Qt framework. \n\
\n\
New in itom 4.1. \n\
\n\
Returns \n\
------- \n\
className : str \n\
    The class name of this :class:`uiItem`.");
PyObject* PythonUi::PyUiItem_getClassName(PyUiItem *self)
{
    if (self->widgetClassName)
    {
        return PyUnicode_FromFormat("%s", self->widgetClassName);
    }
    else
    {
        return PyUnicode_FromString("");
    }
}

//--------------------------------------------------------------------------------------------
// mapping methods
//--------------------------------------------------------------------------------------------
int PythonUi::PyUiItem_mappingLength(PyUiItem* self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return 0;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return 0;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<int> classInfoCount(new int());
    QSharedPointer<int> enumeratorCount(new int());
    QSharedPointer<int> methodCount(new int());
    QSharedPointer<int> propertiesCount(new int());
    *classInfoCount = -1;
    *enumeratorCount = -1;
    *methodCount = -1;
    *propertiesCount = -1;

    QMetaObject::invokeMethod(
        uiOrga,
        "widgetMetaObjectCounts",
        Q_ARG(uint, static_cast<unsigned int>(self->objectID)),
        Q_ARG(QSharedPointer<int>, classInfoCount),
        Q_ARG(QSharedPointer<int>, enumeratorCount),
        Q_ARG(QSharedPointer<int>, methodCount),
        Q_ARG(QSharedPointer<int>, propertiesCount),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    ); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting number of properties");
        return -1;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (retValue.containsError() && retValue.errorCode() == UiOrganizer::errorObjDoesNotExist)
    {
        return 0; //special case: if the widget does not exist any more, return 0 mapping items such the check "if plotHandle:" becomes False, instead of True
    }

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return -1;
    }

    return *propertiesCount; //nr of properties in the corresponding QMetaObject
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUiItem_mappingGetElem(PyUiItem* self, PyObject* key)
{
    QStringList propNames;
    bool ok = false;
    QString propName = PythonQtConversion::PyObjGetString(key, false, ok);

    if(!ok)
    {
        PyErr_SetString(PyExc_RuntimeError, "property name string could not be parsed.");
        return NULL;
    }

    propNames << propName;

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());

    for(int i = 0 ; i < propNames.count() ; i++)
    {
        (*retPropMap)[propNames.at(i)] = QVariant();
    }

    QMetaObject::invokeMethod(
        uiOrga,
        "readProperties",
        Q_ARG(uint, self->objectID),
        Q_ARG(QSharedPointer<QVariantMap>, retPropMap),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    ); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    return PythonQtConversion::QVariantToPyObject(retPropMap->value(propNames[0]));
}

//-------------------------------------------------------------------------------------
int PythonUi::PyUiItem_mappingSetElem(PyUiItem* self, PyObject* key, PyObject* value)
{
    QString keyString;
    bool ok = false;
    QVariantMap propMap;
    QVariant valueV;

    keyString = PythonQtConversion::PyObjGetString(key,false,ok);

    if(!ok)
    {
        PyErr_SetString(PyExc_RuntimeError, "key must be a string");
        return -1;
    }

    valueV = PythonQtConversion::PyObjToQVariant(value);

    if(valueV.isValid())
    {
        propMap[keyString] = valueV;
    }
    else
    {
        PyErr_SetString(PyExc_ValueError, "property value could not be transformed to QVariant.");
        return -1;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return -1;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return -1;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "writeProperties",
        Q_ARG(uint, self->objectID),
        Q_ARG(QVariantMap, propMap),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    ); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while writing property");
        return -1;
    }

    retValue += locker.getSemaphore()->returnValue;

    if(!PythonCommon::transformRetValToPyException(retValue)) return -1;

    return 0;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemCall_doc,"call(publicSlotName, *args) \n\
\n\
Calls any public slot or other accessible public method of the widget or layout, referenced by this uiItem. \n\
\n\
This method calls a public or a 'wrapped' slot (see section :ref:`qtdesigner-wrappedslots`) \n\
of the widget or layout, that is referenced by this :class:`uiItem`. \n\
\n\
If only one slot with the given ``publicSlotName`` is available, all arguments ``*args`` \n\
are tried to be cast to the requested types and the slot is called then. If the \n\
designated slot has multiple possible overloads, at first, it is intended to find the \n\
overload where all arguments can be strictly cast from Python types to the indicated \n\
C-types. If this fails, the next overload with a successful, non-strict conversion is \n\
chosen. \n\
\n\
Information about all possible slots of this :class:`uiItem` can be obtained by the \n\
official Qt help or the method :meth:`uiItem.info`. \n\
\n\
Parameters \n\
---------- \n\
publicSlotName : str \n\
    name of the public slot or a specially wrapped slot of the widget or layout. \n\
*args : Any, optional\n\
    Variable length argument list, that is passed to the called slot. The type of each \n\
    value must be convertible to the requested C++ based argument type of the slot \n\
    (see section :ref:`qtdesigner-datatypes`).\n\
\n\
See Also \n\
-------- \n\
info");
PyObject* PythonUi::PyUiItem_call(PyUiItem *self, PyObject* args)
{
    int argsSize = PyTuple_Size(args);
    int nrOfParams = argsSize - 1;
    bool ok;

    if(argsSize < 1)
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a function name string, optionally followed by the necessary function parameters");
        return NULL;
    }

    QByteArray slotName = PythonQtConversion::PyObjGetBytes(PyTuple_GetItem(args,0),false,ok);

    if(!ok)
    {
        PyErr_SetString(PyExc_TypeError, "First given parameter cannot be interpreted as byteArray.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    if (!loadMethodDescriptionList(self))
    {
        return NULL;
    }

    QSharedPointer<FctCallParamContainer> paramContainer;

    //scan for method
    //step 1: check if method exists
    QList<const MethodDescription*> possibleMethods;

    for (int i = 0; i < self->methodList->size(); ++i)
    {
        if (self->methodList->at(i).name() == slotName)
        {
            possibleMethods.append( &(self->methodList->at(i)) );
        }
    }

    if (possibleMethods.size() == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No slot or method with name '%s' available.", slotName.data());
        return NULL;
    }

    //create function container
    paramContainer = QSharedPointer<FctCallParamContainer>(new FctCallParamContainer(nrOfParams));
    void *ptr = NULL;
    int typeNr = 0;
    bool found = false;
    QByteArray possibleSignatures = "";
    const MethodDescription *foundMethod = NULL;

    // if more than one possible method is availabe,
    // at first, try to strictly cast all parameters...
    if (possibleMethods.count() > 1)
    {
        foreach(const MethodDescription *method, possibleMethods)
        {
            ok = true;

            if (method->checkMethod(slotName, nrOfParams))
            {
                for (int j = 0; j < nrOfParams; j++)
                {
                    // first try to find strict conversions only (in order to
                    // better handle methods with different possible argument types
                    if (PythonQtConversion::PyObjToVoidPtr(
                        PyTuple_GetItem(args,j + 1),  //GetItem is a borrowed reference
                        &ptr,
                        &typeNr,
                        method->argTypes()[j],
                        true))
                    {
                        paramContainer->setParamArg(j, ptr, typeNr);
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }

                if (ok)
                {
                    // init retArg after all other parameters fit to requirements
                    paramContainer->initRetArg(method->retType());

                    found = true;
                    foundMethod = method;
                    break; //everything ok, we found the method and could convert all given parameters
                }
                else
                {
                    possibleSignatures += QByteArray("'" + method->signature() + "', ");
                }

            }
            else
            {
                possibleSignatures += QByteArray("'" + method->signature() + "', ");
                ok = false;
            }
        }
    }
    else
    {
        // ... directly allow the non-strict conversion of
        // all parameters (ok = false enters the next if case ;) )
        ok = false;
    }

    // until now, there is no possibility to directly, strictly cast all
    // parameters to available signatures. Therefore try now also to not-strictly cast
    if (!ok)
    {
        foreach (const MethodDescription *method, possibleMethods)
        {
            ok = true;

            if (method->checkMethod(slotName, nrOfParams))
            {
                ok = true;

                for (int j = 0; j < nrOfParams; j++)
                {
                    // first try to find strict conversions only (in order
                    // to better handle methods with different possible argument types
                    if (PythonQtConversion::PyObjToVoidPtr(
                        PyTuple_GetItem(args,j + 1), //GetItem is a borrowed reference
                        &ptr,
                        &typeNr,
                        method->argTypes()[j],
                        false))
                    {
                        paramContainer->setParamArg(j, ptr, typeNr);
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }

                if (ok)
                {
                    // init retArg after all other parameters fit to requirements
                    paramContainer->initRetArg(method->retType());

                    found = true;
                    foundMethod = method;
                    break; //everything ok, we found the method and could convert all given parameters
                }
                else
                {
                    possibleSignatures += QByteArray("'" + method->signature() + "', ");
                }

            }
            else
            {
                possibleSignatures += QByteArray("'" + method->signature() + "', ");
            }
        }
    }

    if (!found)
    {
        PyErr_Format(
            PyExc_RuntimeError,
            "None of the following possible signatures fit to the given set of parameters: %s",
            possibleSignatures.data()
        );
        return NULL;
    }

    ItomSharedSemaphoreLocker waitForInvokationEnd(new ItomSharedSemaphore());
    int method_type = foundMethod->type();

    if (method_type == QMetaMethod::Slot || method_type == QMetaMethod::Method)
    {
        QMetaObject::invokeMethod(
            uiOrga,
            "callSlotOrMethod",
            Q_ARG(bool, method_type == QMetaMethod::Slot),
            Q_ARG(uint, self->objectID),
            // 'unsigned int' leads to overhead and is automatically
            // transformed to uint in invokeMethod command
            Q_ARG(int, foundMethod->methodIndex()),
            Q_ARG(QSharedPointer<FctCallParamContainer>, paramContainer),
            Q_ARG(ItomSharedSemaphore*, waitForInvokationEnd.getSemaphore()));
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError,
            QString("unknown method type: %1").arg(method_type).toLatin1().data());
        return NULL;
    }

    if (!waitForInvokationEnd.getSemaphore()->wait(50000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while calling slot");
        return NULL;
    }

    if (PythonCommon::transformRetValToPyException(
        waitForInvokationEnd.getSemaphore()->returnValue) == false)
    {
        return NULL;
    }

    if (paramContainer->getRetType() > 0)
    {
        if (paramContainer->getRetType() == QMetaType::type("ito::PythonQObjectMarshal"))
        {
            ito::PythonQObjectMarshal *m = (ito::PythonQObjectMarshal*)paramContainer->args()[0];

            PyObject *newArgs = PyTuple_New(4);
            PyTuple_SetItem(newArgs, 0, PyLong_FromLong(m->m_objectID));
            PyTuple_SetItem(newArgs, 1, PyUnicode_FromString( m->m_objName.data() ));
            PyTuple_SetItem(newArgs, 2, PyUnicode_FromString( m->m_className.data() ));
            Py_INCREF(self);
            PyTuple_SetItem(newArgs, 3, (PyObject*)self);
            PyObject *newUiItem = PyObject_CallObject((PyObject *) &PythonUi::PyUiItemType, newArgs);
            Py_DECREF(newArgs);
            return newUiItem;
        }
        else
        {
            return PythonQtConversion::ConvertQtValueToPythonInternal(
                paramContainer->getRetType(),
                paramContainer->args()[0]
            );
        }
    }

    Py_RETURN_NONE;

}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemConnect_doc, "connect(signalSignature, callableMethod, minRepeatInterval = 0) \n\
\n\
Connects a signal of this widget or layout with the given Python callback method. \n\
\n\
The widget or layout class, referenced by an :class:`uiItem` object, can emit \n\
different signals whenever a certain event occurs. See the official Qt help \n\
about a list of all possible signals or use the method :meth:`info` to get a \n\
print-out of a list of possible signals. This method is used to connect a certain \n\
callable Python callback method or function to a specific signal. The callable \n\
function can be bounded as well as unbounded. \n\
\n\
The connection is described by the string signature of the signal (hence the source of \n\
the connection). Such a signature is the name of the signal, followed by the types of \n\
its arguments (the original C++ types). An example is ``clicked(bool)``, \n\
emitted if a button has been clicked. This signal can be connected to a callback function \n\
with one argument, that will then contain the boolean click state of this signal. \n\
In case of a bounded method, the ``self`` argument must be given in any case. \n\
\n\
If the signal should have further arguments with specific datatypes, they are transformed \n\
into corresponding Python data types. A table of supported conversions is given in section \n\
:ref:`qtdesigner-datatypes`. In general, a ``callableMethod`` must be a method or \n\
function with the same number of parameters than the signal has (besides the \n\
``self`` argument). \n\
\n\
If a signal is emitted very often, it can be necessary to limit the call of the callback \n\
function to a certain minimum time interval. This can be given by the ``minRepeatInterval`` \n\
parameter. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. ``targetChanged(QVector<double>)``) \n\
callableMethod : callable \n\
    valid method or function that is called if the signal is emitted. \n\
minRepeatInterval : int, optional \n\
    If > 0, the same signal only invokes a slot once within the given interval (in ms). \n\
    Default: 0 (all signals will invoke the callable python method. \n\
\n\
Notes \n\
----- \n\
The Python callback method can only be executed if Python is in an idle state. Else, \n\
the trigger is postponed to the next possible time. However, if you want for instance \n\
to have a button that interrupts a long Python operation, it is not possible to use \n\
this :meth:`connect` method to bind the click signal of this button with any \n\
Python script interruption, since the callback method will only be called if the long \n\
operation has finished. For these cases it is recommenden to connect the triggering \n\
signal (e.g. `clicked()`) by the :meth:`invokeKeyboardInterrupt` method. \n\
\n\
See Also \n\
-------- \n\
disconnect, info, invokeKeyboardInterrupt");
PyObject* PythonUi::PyUiItem_connect(PyUiItem *self, PyObject* args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", "minRepeatInterval", NULL };
    const char* signalSignature;
    int minRepeatInterval = 0;
    PyObject *callableMethod;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "sO|i", const_cast<char**>(kwlist), &signalSignature, &callableMethod, &minRepeatInterval))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }
    if(!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }
    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    QByteArray signature(signalSignature);
    QSharedPointer<int> sigId(new int);

    QSharedPointer<QObject*> objPtr(new (QObject*));
    QSharedPointer<IntList> argTypes(new IntList);

    *sigId = -1;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    //returns the Qt-internal signal index of the requested signal signature of the QWidget that can emit this signal
    /* Hint: by Qt's moc process, all signals and slots of a class (having the Q_OBJECT macro) get an auto-incremented index
       and the 'hidden' method qt_metacall is added to the specific class via its auto-generated _moc.cpp code.
       This qt_metacall method mainly consists of a switch-case that maps the signal and slot indices to real slot calls
       or signal invocations (in other objects via their qt_metacall-method).
    */
    QMetaObject::invokeMethod(
        uiOrga,
        "getSignalIndex",
        Q_ARG(uint, self->objectID),
        Q_ARG(QByteArray, signature),
        Q_ARG(QSharedPointer<int>, sigId),
        Q_ARG(QSharedPointer<QObject*>, objPtr),
        Q_ARG(QSharedPointer<IntList>, argTypes),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    ); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if(*sigId == -1)
    {
        PyErr_SetString(PyExc_RuntimeError, "signal signature is invalid.");
        return NULL;
    }

    PythonQtSignalMapper *signalMapper = PyUiItem_getTopLevelSignalMapper(self);
    if(signalMapper)
    {
        if(!signalMapper->addSignalHandler(*objPtr, signalSignature, *sigId, callableMethod, *argTypes, minRepeatInterval))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be established.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No user interface for this UiItem could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemConnectKeyboardInterrupt_doc,"invokeKeyboardInterrupt(signalSignature) \n\
\n\
Connects the given signal with the immediate invokation of a Python interrupt signal. \n\
\n\
If you use the connect method to link a signal with a python method or function, this \n\
method can only be executed if Python is in an idle status. However, if you want to \n\
immediately raise the Python interrupt signal, use this method to establish the \n\
connection instead of the :meth:`uiItem.connect` command. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. 'clicked(bool)') \n\
\n\
See Also \n\
-------- \n\
connect, invokeProgressObserverCancellation");
PyObject* PythonUi::PyUiItem_connectKeyboardInterrupt(PyUiItem *self, PyObject* args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", NULL };
    const char* signalSignature;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &signalSignature))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    QByteArray signature(signalSignature);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "connectWithKeyboardInterrupt",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QByteArray, signature),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemConnectProgressObserverInterrupt_doc,"invokeProgressObserverCancellation(signalSignature, observer) \n\
\n\
Connects the given signal to a slot immediately setting the cancellation flag of this object. \n\
\n\
This method immediately calls the ``requestCancellation`` slot of the given observer \n\
if the signal with the ``signalSignature`` is emitted (independent on the current \n\
state of the Python script execution). \n\
\n\
For more information about the class :class:`requestCancellation`, see also this \n\
section: :ref:`filter_interruptible`. \n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. 'clicked(bool)') \n\
observer : progressObserver \n\
    This must be a :class:`progressObserver` object. The given signal is connected \n\
    to the slot ``requestCancellation`` of this progressObserver.\n\
\n\
See Also \n\
-------- \n\
connect, invokeKeyboardInterrupt");
PyObject* PythonUi::PyUiItem_connectProgressObserverInterrupt(PyUiItem *self, PyObject* args, PyObject *kwds)
{
    const char* signalSignature;
    PyObject *observer = NULL;

    const char *kwlist[] = { "signalSignature", "observer", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO!", const_cast<char**>(kwlist), &signalSignature, &PythonProgressObserver::PyProgressObserverType, &observer))
    {
        return NULL;
    }

    QSharedPointer<ito::FunctionCancellationAndObserver> obs = *(((PythonProgressObserver::PyProgressObserver*)observer)->progressObserver);

    if (obs.isNull())
    {
        PyErr_SetString(PyExc_RuntimeError, "The observer is invalid or does not exist anymore.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    QByteArray signature(signalSignature);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "connectProgressObserverInterrupt",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QByteArray, signature),
        Q_ARG(QPointer<QObject>, QPointer<QObject>(obs.data())),
        Q_ARG(ItomSharedSemaphore*,
        locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemDisconnect_doc, "disconnect(signalSignature, callableMethod) \n\
\n\
Disconnects a connection which must have been established before with exactly the same parameters.\n\
\n\
Parameters \n\
---------- \n\
signalSignature : str \n\
    This must be the valid signature, known from the Qt-method *connect* \n\
    (e.g. ``clicked(bool)``) \n\
callableMethod : callable \n\
    valid method or function, that should not be called any more if the \n\
    given signal is emitted. \n\
\n\
See Also \n\
-------- \n\
connect, info");
PyObject* PythonUi::PyUiItem_disconnect(PyUiItem *self, PyObject* args, PyObject *kwds)
{
    const char *kwlist[] = { "signalSignature", "callableMethod", NULL };
    const char* signalSignature;
    PyObject *callableMethod;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO", const_cast<char**>(kwlist), &signalSignature, &callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "Arguments must be a signal signature and a callable method reference");
        return NULL;
    }

    if(!PyCallable_Check(callableMethod))
    {
        PyErr_SetString(PyExc_TypeError, "given method reference is not callable.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    QByteArray signature(signalSignature);
    QSharedPointer<int> sigId(new int);
    QSharedPointer<QObject*> objPtr(new (QObject*));
    QSharedPointer<IntList> argTypes(new IntList);

    *sigId = -1;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "getSignalIndex",
        Q_ARG(uint, self->objectID),
        Q_ARG(QByteArray, signature),
        Q_ARG(QSharedPointer<int>, sigId),
        Q_ARG(QSharedPointer<QObject*>, objPtr),
        Q_ARG(QSharedPointer<IntList>, argTypes),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    ); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if(*sigId == -1)
    {
        PyErr_SetString(PyExc_RuntimeError, "signal signature is invalid.");
        return NULL;
    }

    PythonQtSignalMapper *signalMapper = PyUiItem_getTopLevelSignalMapper(self);

    if(signalMapper)
    {
        if(!signalMapper->removeSignalHandler(*objPtr, *sigId, callableMethod))
        {
            PyErr_SetString(PyExc_RuntimeError, "the connection could not be disconnected.");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "No user interface for this UiItem could be found");
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetProperty_doc,"getProperty(propertyName) -> Union[Any, List[Any]] \n\
n\
Returns the requested property or a list of values for a sequence of requested properties. \n\
\n\
Use this method or the operator [] in order to get the value of one specific property \n\
of this widget or layout or of multiple properties. \n\
\n\
Multiple properties are given by a tuple or list of property names. For one single \n\
property, its value is returned as it is. If the property names are passed as sequence, \n\
a list of same size is returned with the corresponding values. \n\
\n\
Parameters \n\
---------- \n\
propertyName : str or list of str or tuple of str \n\
    Name of one property or sequence (tuple, list...) of property names. \n\
\n\
Returns \n\
------- \n\
value : Any or list of Any \n\
    the value of one single property of a list of values, if a sequence of ``propertyNames`` \n\
    is given as parameter. \n\
\n\
See Also \n\
-------- \n\
setProperty");
PyObject* PythonUi::PyUiItem_getProperties(PyUiItem *self, PyObject *args)
{
    PyObject *propertyNames = NULL;
    QStringList propNames;
    bool ok = false;
    bool returnList = true;

    if(!PyArg_ParseTuple(args, "O", &propertyNames))
    {
        return NULL;
    }

    if(PyBytes_Check(propertyNames) || PyUnicode_Check(propertyNames))
    {
        QString temp = PythonQtConversion::PyObjGetString(propertyNames, true, ok);

        if(ok)
        {
            returnList = false;
            propNames << temp;
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "property name string could not be parsed.");
            return NULL;
        }
    }
    else if(PySequence_Check(propertyNames))
    {
        propNames = PythonQtConversion::PyObjToStringList(propertyNames, true, ok);

        if(!ok)
        {
            PyErr_SetString(PyExc_RuntimeError, "list or tuple of property names could not be converted to a list of strings");
            return NULL;
        }
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "property name must be a string or tuple/list of strings");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());
    for(int i = 0 ; i < propNames.count() ; i++)
    {
        (*retPropMap)[propNames.at(i)] = QVariant();
    }

    QMetaObject::invokeMethod(
        uiOrga,
        "readProperties",
        Q_ARG(unsigned int, self->objectID),
        Q_ARG(QSharedPointer<QVariantMap>, retPropMap),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if(returnList)
    {
        PyObject *retObj = PyList_New(propNames.count());

        for(int i = 0 ; i < propNames.count() ; i++)
        {
            PyList_SetItem(retObj,i, PythonQtConversion::QVariantToPyObject(retPropMap->value(propNames.at(i))));
        }

        return retObj;
    }
    else
    {
        return PythonQtConversion::QVariantToPyObject( retPropMap->value(propNames.at(0)) );
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemSetProperty_doc,"setProperty(propertyDict) \n\
\n\
Each property in the ``propertyDict`` is set to the dictionaries value. \n\
\n\
As an alternative, a single property can also be set using the operator []. \n\
\n\
Parameters \n\
---------- \n\
propertyDict : dict\n\
    Dictionary with properties (the keys are the property names) and the values \n\
    that should be set.\n\
\n\
See Also \n\
-------- \n\
getProperty");
PyObject* PythonUi::PyUiItem_setProperties(PyUiItem *self, PyObject *args)
{
    PyObject *propDict = NULL;
    bool ok = false;
    QVariantMap propMap;

    if(!PyArg_ParseTuple(args, "O!", &PyDict_Type, &propDict))
    {
        return NULL;
    }

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    QVariant valueV;
    QString keyS;

    while (PyDict_Next(propDict, &pos, &key, &value))
    {
        keyS = PythonQtConversion::PyObjGetString(key,true,ok);
        valueV = PythonQtConversion::PyObjToQVariant(value);

        if(valueV.isValid())
        {
            propMap[keyS] = valueV;
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "at least one property value could not be parsed to QVariant.");
            Py_DECREF(propDict);
            return NULL;
        }
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "writeProperties",
        Q_ARG(uint, static_cast<unsigned int>(self->objectID)), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QVariantMap, propMap),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while writing property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetPropertyInfo_doc,"getPropertyInfo(propertyName = None) -> Union[dict, List[str]] \n\
\n\
Returns a list of all available property names or a dict of meta information of one given ``propertyName``. \n\
\n\
if ``propertyName`` is ``None``, a list of all property names is returned. Else, \n\
a ``Dict[str, Any]`` is returned with meta information about this property. \n\
The structure of this dictionary is as follows: \n\
\n\
* **name**: Name of the property (str). \n\
* **valid**: ``True`` if this property is valid (readable), otherwise ``False``. \n\
* **readable**: ``True`` if this property is readable, otherwise ``False``. \n\
* **writable**: ``True`` if this property can be set to another value, otherwise ``False``. \n\
* **resettable**: ``True`` if this property can be reset to a default value; otherwise returns ``False``. \n\
* **final**: ``True`` if this property is final and cannot be overwritten in derived classes, otherwise ``False``.\n\
* **constant**: ``True`` if this property is constant, otherwise ``False``.\n\
\n\
Parameters \n\
---------- \n\
propertyName : str, optional \n\
    The name of the property whose detailed information should be returned or \n\
    ``None``, if a list of all property names should be returned. \n\
\n\
Returns \n\
------- \n\
names : list of str \n\
    A list of all available property names. \n\
information : dict \n\
    The dictionary with meta information about this property (see above).");
PyObject* PythonUi::PyUiItem_getPropertyInfo(PyUiItem *self, PyObject *args)
{
    const char *propertyName = NULL;

    if(!PyArg_ParseTuple(args, "|s", &propertyName))
    {
        PyErr_SetString(PyExc_RuntimeError, "argument only accepts one optional name of a property (string, optional)");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QVariantMap> retPropMap(new QVariantMap());
    QMetaObject::invokeMethod(
        uiOrga,
        "getPropertyInfos",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QSharedPointer<QVariantMap>, retPropMap),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting property information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if(retValue.containsWarningOrError())
    {
        if (!PythonCommon::setReturnValueMessage(retValue, propertyName ? propertyName : "getPropertyInfo", PythonCommon::getProperty))
        {
            return NULL;
        }
    }

    QStringList stringList = retPropMap->keys();
    QString propNameString = QString(propertyName);

    if(propertyName == NULL)
    {
        PyObject *retObj = PythonQtConversion::QStringListToPyList(stringList);
        return retObj;
    }
    else if(retPropMap->contains(propNameString))
    {
        int flags = retPropMap->value(propNameString).toInt();
        PyObject *retObj = PyDict_New();

        PyObject *item = PythonQtConversion::QByteArrayToPyUnicodeSecure( propNameString.toLatin1() );
        PyDict_SetItemString(retObj, "name", item);
        Py_DECREF(item);

        item = PythonQtConversion::GetPyBool( flags & UiOrganizer::propValid );
        PyDict_SetItemString(retObj, "valid", item);
        Py_DECREF(item);

        item = PythonQtConversion::GetPyBool( flags & UiOrganizer::propReadable );
        PyDict_SetItemString(retObj, "readable", item);
        Py_DECREF(item);

        item = PythonQtConversion::GetPyBool( flags & UiOrganizer::propWritable );
        PyDict_SetItemString(retObj, "writable", item);
        Py_DECREF(item);

        item = PythonQtConversion::GetPyBool( flags & UiOrganizer::propResettable );
        PyDict_SetItemString(retObj, "resettable", item);
        Py_DECREF(item);

        item = PythonQtConversion::GetPyBool( flags & UiOrganizer::propFinal );
        PyDict_SetItemString(retObj, "final", item);
        Py_DECREF(item);

        item = PythonQtConversion::GetPyBool( flags & UiOrganizer::propConstant );
        PyDict_SetItemString(retObj, "constant", item);
        Py_DECREF(item);

        PyObject *proxyDict = PyDictProxy_New(retObj);
        Py_DECREF(retObj);

        return proxyDict;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, QString("the property '%1' does not exist.").arg(propNameString).toUtf8().data());
        return NULL;
    }
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetAttribute_doc,"getAttribute(attributeNumber) -> bool \n\
\n\
Returns if a specific WidgetAttribute is set for the referenced widget. \n\
\n\
Widgets have specific attributes that influence their behaviour. These attributes \n\
are contained in the Qt-enumeration ``Qt::WidgetAttribute``. Use this method to \n\
query if the requested ``attributeNumber`` is set / enabled for the referenced widget. \n\
\n\
Important attributes are: \n\
\n\
* Qt::WA_DeleteOnClose (55) -> deletes the widget when it is closed, else it is only \n\
  hidden [default] \n\
* Qt::WA_MouseTracking (2) -> indicates that the widget has mouse tracking enabled \n\
\n\
Parameters \n\
---------- \n\
attributeNumber : int \n\
    Number of the attribute of the widget to query (see Qt enumeration \n\
    ``Qt::WidgetAttribute``) \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if attribute is set (enabled), otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
setAttribute");
PyObject* PythonUi::PyUiItem_getAttribute(PyUiItem *self, PyObject *args)
{
    int attributeNumber;

    if(!PyArg_ParseTuple(args, "i", &attributeNumber))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<bool> value(new bool);

    QMetaObject::invokeMethod(
        uiOrga,
        "getAttribute",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(int, attributeNumber),
        Q_ARG(QSharedPointer<bool>, value),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting attribute");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if (*value)
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemSetAttribute_doc,"setAttribute(attributeNumber, value) \n\
\n\
Enables or disables the attribute of the referenced widget.\n\
\n\
Widgets have specific attributes that influence their behaviour. These attributes \n\
are contained in the Qt-enumeration ``Qt::WidgetAttribute``. Use this method to \n\
enable or disable the requested widget attribute, given by its ``attributeNumber``. \n\
\n\
Important attributes are: \n\
\n\
* Qt::WA_DeleteOnClose (55) -> deletes the widget when it is closed, else it is \n\
  only hidden [default]. \n\
* Qt::WA_MouseTracking (2) -> indicates that the widget has mouse tracking enabled. \n\
\n\
Parameters \n\
---------- \n\
attributeNumber : int \n\
    Number of the attribute of the widget to set (enum ``Qt::WidgetAttribute``). \n\
value : bool \n\
    ``True`` if attribute should be enabled, else ``False``. \n\
\n\
See Also \n\
-------- \n\
getAttribute");
PyObject* PythonUi::PyUiItem_setAttribute(PyUiItem *self, PyObject *args)
{
    int attributeNumber;
    bool value;

    if(!PyArg_ParseTuple(args, "ib", &attributeNumber, &value))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "setAttribute",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(int, attributeNumber),
        Q_ARG(bool, value),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while setting attribute");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemSetWindowFlags_doc,"setWindowFlags(flags) \n\
\n\
Set the window flags of the referenced widget.\n\
\n\
The window flags are used to set the type of a widget, dialog or window including \n\
further hints to the window system. This method is used to set the entire \n\
or-combination of all ``flags``, contained in the Qt-enumeration ``Qt::WindowType``. \n\
\n\
Please consider, that you have to set all values in ``flags``, that should be \n\
active in the referenced widget. It is possible to get the current flags value of \n\
this widget using :meth:`getWindowFlags``, set or unset some enum values (bits) \n\
and set it again using this method. \n\
\n\
The most important types are: \n\
\n\
* Qt::Widget (0) -> default type for widgets \n\
* Qt::Window (1) -> the widget looks and behaves like a windows (title bar, window \n\
  frame...) \n\
* Qt::Dialog (3) -> window decorated as dialog (no minimize or maximize button...) \n\
\n\
Further hints can be (among others): \n\
\n\
* Qt::FramelessWindowHint (0x00000800) -> borderless window (user cannot move or \n\
  resize the window) \n\
* Qt::WindowTitleBar (0x00001000) -> gives the window a title bar \n\
* Qt::WindowMinimizeButtonHint (0x00004000) -> adds a minimize button to the \n\
  title bar \n\
* Qt::WindowMaximizeButtonHint (0x00008000) -> adds a maximize button to the \n\
  title bar \n\
* Qt::WindowCloseButtonHint (0x00010000) -> adds a close button. \n\
* Qt::WindowStaysOnTopHint (0x00040000) -> this ui element always stays on top of \n\
  other windows \n\
* Qt::WindowCloseButtonHint (0x08000000) -> remove this flag in order to disable the \n\
  close button \n\
\n\
Parameters \n\
---------- \n\
flags : int \n\
    window flags to set (or-combination, see ``Qt::WindowFlags``). \n\
\n\
See Also \n\
-------- \n\
getWindowFlags");
PyObject* PythonUi::PyUiItem_setWindowFlags(PyUiItem *self, PyObject *args)
{
    int value;

    if(!PyArg_ParseTuple(args, "i", &value))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "setWindowFlags",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(int, value),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while setting window flags");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetWindowFlags_doc,"getWindowFlags() -> int \n\
\n\
Gets the window flags of the referenced widget. \n\
\n\
The returned ``flags`` value is an or-combination, hence bitmask, of enumeration \n\
values of the Qt enumeration ``Qt::WindowType``. \n\
\n\
Returns \n\
------- \n\
flags : int \n\
    or-combination of ``Qt::WindowType`` describing the type and further hints \n\
    of the referenced widget. \n\
\n\
See Also \n\
-------- \n\
setWindowFlags");
PyObject* PythonUi::PyUiItem_getWindowFlags(PyUiItem *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<int> value(new int);

    QMetaObject::invokeMethod(
        uiOrga,
        "getWindowFlags",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QSharedPointer<int>, value),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting window flag");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    return Py_BuildValue("i", *value);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemInfo_doc,"info(verbose = 0) \n\
\n\
Prints information about properties, public accessible slots and signals of the wrapped widget. \n\
\n\
Parameters \n\
---------- \n\
verbose : int \n\
    * ``0``: only properties, slots and signals that do not come from Qt-classes are \n\
      printed (default) \n\
    * ``1``: properties, slots and signals are printed up to Qt GUI base classes \n\
    * ``2``: all properties, slots and signals are printed");
/*static*/ PyObject* PythonUi::PyUiItem_info(PyUiItem *self, PyObject *args)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    int showAll = 0;

    if(!PyArg_ParseTuple(args, "|i", &showAll))
    {
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    //!> we need this as otherwise the Q_ARG macro does not recognize our templated QMap
    int type = UiOrganizer::infoShowAllInheritance;

    if (showAll == 1)
    {
        type = UiOrganizer::infoShowInheritanceUpToWidget;
    }
    else if (showAll < 1)
    {
        type = UiOrganizer::infoShowItomInheritance;
    }

    QMetaObject::invokeMethod(
        uiOrga,
        "getObjectInfo",
        Q_ARG(uint, self->objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(int, type), Q_ARG(bool, true),
        Q_ARG(ito::UiOrganizer::ClassInfoContainerList*, NULL),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if (showAll < 2)
    {
        std::cout << "For more properties, slots and signals call info(1) for properties, slots and signals \n" \
                       "besides the ones that originate from Qt GUI base classes " \
                      "or info(2) for all properties, slots and signals\n" << std::endl;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemExists_doc,"exists() -> bool \n\
\n\
Returns True if the widget or layout still exists, otherwise False. \n\
\n\
Returns \n\
------- \n\
bool \n\
    ``True`` if the referenced widget or layout still exists, otherwise ``False``.");
/*static*/ PyObject* PythonUi::PyUiItem_exists(PyUiItem *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer< bool > exists(new bool );

    //!> we need this as otherwise the Q_ARG macro does not recognize our templated QMap
    QMetaObject::invokeMethod(
        uiOrga,
        "exists",
        Q_ARG(uint, self->objectID),
        Q_ARG(QSharedPointer<bool>,exists),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if (*exists)
    {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemChildren_doc,"children(recursive = False) -> Dict[str, str] \n\
\n\
Returns a dict with all child items of the referenced widget. \n\
\n\
Each widget in an user interface can have multiple child items, like radio buttons \n\
within a group box or widgets within a layout. This method returns information about \n\
all child items of this :class:`uiItem`. A dictionary is returned with key-value \n\
pairs, where the key is the ``objectName`` of the child item, and the value its \n\
Qt class name (see :meth:`getClassName`). \n\
\n\
Child items without valid ``objectName`` are not contained in the returned dict. \n\
\n\
Parameters \n\
---------- \n\
recursive : bool \n\
    ``True``: all objects including sub-widgets of widgets are returned, \n\
    ``False``: only children of this :class:`uiItem` are returned (default). \n\
\n\
Returns \n\
------- \n\
dict \n\
    All child items of this item are returned.");
/*static*/ PyObject* PythonUi::PyUiItem_children(PyUiItem *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"recursive", NULL};
    unsigned char recursive = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|b", const_cast<char**>(kwlist), &recursive))
    {
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->objectID <= 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer< QStringList > objectNames(new QStringList() );
    QSharedPointer< QStringList > classNames(new QStringList() );

    //!> we need this as otherwise the Q_ARG macro does not recognize our templated QMap
    QMetaObject::invokeMethod(
        uiOrga,
        "getObjectChildrenInfo",
        Q_ARG(uint, self->objectID),
        Q_ARG(bool, recursive > 0),
        Q_ARG(QSharedPointer<QStringList>,objectNames),
        Q_ARG(QSharedPointer<QStringList>,classNames),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    PyObject *dict = PyDict_New();
    PyObject *value = NULL;

    for (int i = 0; i < objectNames->size(); ++i)
    {
        value = PythonQtConversion::QStringToPyObject(classNames->at(i));
        PyDict_SetItemString(dict, objectNames->at(i).toUtf8().data(), value);
        Py_DECREF(value);
    }

    return dict;
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetChild_doc, "getChild(widgetName) -> uiItem \n\
\n\
Returns the uiItem of the child widget with the given ``widgetName``. \n\
\n\
This call is equal to ``self.widgetName``, where ``self`` is this :class:`uiItem`. \n\
\n\
Parameters \n\
---------- \n\
widgetName : str \n\
    ``objectName`` of the requested child widget or layout. \n\
\n\
Returns \n\
------- \n\
item : uiItem \n\
    The reference to the searched sub-widget (or layout).\n\
\n\
Raises \n\
------ \n\
AttributeError \n\
    if no widget / layout with ``widgetName`` as ``objectName`` exists.");
/*static*/ PyObject* PythonUi::PyUiItem_getChild(PyUiItem *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "widgetName", NULL };
    PyObject *name;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "U", const_cast<char**>(kwlist), &name))
    {
        return NULL;
    }

    return PyUiItem_getattro(self, name);
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetLayout_doc, "getLayout() -> Optional[uiItem] \n\
\n\
Returns the uiItem of the layout item of this widget (or None). \n\
\n\
Container widgets, like group boxes, tab widgets etc. as well as top level widgets \n\
of a custom user interface can have layouts, that are responsible to arrange \n\
possible child widgets. \n\
\n\
If this uiItem has such a layout, its reference is returned as :class:`uiItem`, too. \n\
Else ``None`` is returned. \n\
\n\
Returns \n\
------- \n\
layout : None or uiItem \n\
    The reference to the searched layout, or ``None`` if no such a layout exists.");
/*static*/ PyObject* PythonUi::PyUiItem_getLayout(PyUiItem *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<unsigned int> layoutObjectID(new unsigned int());
    QSharedPointer<QByteArray> layoutClassName(new QByteArray());
    QSharedPointer<QString> layoutObjectName(new QString());

    //!> we need this as otherwise the Q_ARG macro does not recognize our templated QMap
    QMetaObject::invokeMethod(
        uiOrga,
        "getLayout",
        Q_ARG(uint, self->objectID),
        Q_ARG(QSharedPointer<unsigned int>, layoutObjectID),
        Q_ARG(QSharedPointer<QByteArray>, layoutClassName),
        Q_ARG(QSharedPointer<QString>, layoutObjectName),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting layout");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    PyUiItem *newself = (PyUiItem*)PythonUi::PyUiItemType.tp_new(&PythonUi::PyUiItemType, NULL, NULL);

    if (newself != NULL)
    {
        Py_XINCREF(self);
        newself->baseItem = (PyObject*)self;

        DELETE_AND_SET_NULL_ARRAY(newself->objName);
        newself->objName = new char[layoutObjectName->size() + 1];
        strcpy_s(newself->objName, layoutObjectName->size() + 1, layoutObjectName->toLatin1().data());

        DELETE_AND_SET_NULL_ARRAY(newself->widgetClassName);
        newself->widgetClassName = new char[layoutClassName->size() + 1];
        strcpy_s(newself->widgetClassName, layoutClassName->size() + 1, layoutClassName->data());

        newself->objectID = *layoutObjectID;
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "Error creating new uiItem.");
    }

    return (PyObject *)newself;
}


//-------------------------------------------------------------------------------------
bool PythonUi::loadMethodDescriptionList(PyUiItem *self)
{
    if(self->methodList == NULL)
    {
        QByteArray className(self->widgetClassName);
        QHash<QByteArray, QSharedPointer<ito::MethodDescriptionList> >::const_iterator it = methodDescriptionListStorage.constFind( className );

        if (it != methodDescriptionListStorage.constEnd())
        {
            self->methodList = it->data();
            return true;
        }
        else
        {
            UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
            if(uiOrga == NULL)
            {
                PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
                return false;
            }
            if(self->objectID <= 0)
            {
                PyErr_SetString(PyExc_RuntimeError, "No valid objectID is assigned to this uiItem-instance");
                return false;
            }

            QSharedPointer<MethodDescriptionList> methodList(new MethodDescriptionList);
            ItomSharedSemaphoreLocker locker1(new ItomSharedSemaphore());

            QMetaObject::invokeMethod(
                uiOrga,
                "getMethodDescriptions",
                Q_ARG(uint, self->objectID),
                Q_ARG(QSharedPointer<MethodDescriptionList>, methodList),
                Q_ARG(ItomSharedSemaphore*, locker1.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

            if(!locker1.getSemaphore()->wait(PLUGINWAIT))
            {
                PyErr_SetString(PyExc_RuntimeError, "timeout while analysing method description list");
                return false;
            }

            methodDescriptionListStorage[className] = methodList;
            self->methodList = methodList.data();
        }
    }

    return true;
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUiItem_getattro(PyUiItem *self, PyObject *name)
{
    //UiItem has no __dict__ and __slots__ attribute and this is no widget either, therefore filter it out and raise an exception
    if (PyUnicode_CompareWithASCIIString(name, "__slots__") == 0)
    {
        return PyErr_Format(PyExc_AttributeError, "'%.50s' object has no attribute '__slots__'.", self->objName);
    }
    if (PyUnicode_CompareWithASCIIString(name, "__dict__") == 0)
    {
        return PyErr_Format(PyExc_AttributeError, "'%.50s' object has no attribute '__dict__'.", self->objName);
    }
    else if (PyUnicode_CompareWithASCIIString(name, "__getstate__") == 0)
    {
        return PyErr_Format(PyExc_AttributeError, "'%.50s' object has no attribute '%U' (e.g. it cannot be pickled).", self->objName, name);
    }

    PyObject *ret = PyObject_GenericGetAttr((PyObject*)self,name); //new reference

    if (ret != NULL)
    {
        return ret;
    }
    PyErr_Clear(); //genericgetattr throws an error, if attribute is not available, which it isn't for attributes pointing to widgetNames

    //return new instance of pyUiItem
    PyObject *arg2 = Py_BuildValue("OO", self, name);
    PythonUi::PyUiItem *pyUiItem = (PythonUi::PyUiItem *)PyObject_CallObject((PyObject *)&PythonUi::PyUiItemType, arg2);
    Py_DECREF(arg2);

    if (pyUiItem == NULL)
    {
        //check if this uiItem exists:
        PyObject *exists = PyUiItem_exists(self);

        if (exists == Py_False)
        {
            Py_XDECREF(exists);
            PyErr_SetString(PyExc_AttributeError, "This uiItem does not exist any more.");
            return NULL;
        }
        else
        {
            Py_XDECREF(exists);
            bool ok;
            QString name_str = PythonQtConversion::PyObjGetString(name, true, ok);

            if (ok)
            {
                return PyErr_Format(
                    PyExc_AttributeError,
                    "This uiItem has neither a child item nor a method defined with the name '%s'.",
                    name_str.toLatin1().data());
            }
            else
            {
                PyErr_SetString(
                    PyExc_AttributeError,
                    "This uiItem has neither a child item nor a method defined with the given name (string).");
                return NULL;
            }
        }
    }

    if (PyErr_Occurred())
    {
        Py_XDECREF(pyUiItem);
        pyUiItem = NULL;
    }

    return (PyObject*)pyUiItem;
}

//-------------------------------------------------------------------------------------
int PythonUi::PyUiItem_setattro(PyUiItem *self, PyObject *name, PyObject *value)
{
    return PyObject_GenericSetAttr( (PyObject*)self, name, value );
}

//returns borrowed reference to signal mapper of overall ui- or figure-instance or NULL, if this does not exist
/*static*/ PythonQtSignalMapper* PythonUi::PyUiItem_getTopLevelSignalMapper(PyUiItem *self)
{
    PyUiItem *item = self;
    while(item)
    {
        if(PyUi_Check(item))
        {
            return ((PyUi*)item)->signalMapper;
        }
        else if(PyFigure_Check(item))
        {
            return ((PythonFigure::PyFigure*)item)->signalMapper;
        }
        item = (PyUiItem*)(item->baseItem);
    }
    return NULL;
}

//-------------------------------------------------------------------------------------
PyMethodDef PythonUi::PyUiItem_methods[] = {
        {"call", (PyCFunction)PyUiItem_call, METH_VARARGS, PyUiItemCall_doc},
        {"connect", (PyCFunction)PyUiItem_connect, METH_KEYWORDS | METH_VARARGS, PyUiItemConnect_doc},
        {"disconnect", (PyCFunction)PyUiItem_disconnect, METH_KEYWORDS | METH_VARARGS, PyUiItemDisconnect_doc},
        {"setProperty", (PyCFunction)PyUiItem_setProperties, METH_VARARGS, PyUiItemSetProperty_doc},
        {"getProperty", (PyCFunction)PyUiItem_getProperties, METH_VARARGS, PyUiItemGetProperty_doc},
        {"getPropertyInfo", (PyCFunction)PyUiItem_getPropertyInfo, METH_VARARGS, PyUiItemGetPropertyInfo_doc},
        {"setAttribute", (PyCFunction)PyUiItem_setAttribute, METH_VARARGS, PyUiItemSetAttribute_doc},
        {"getAttribute", (PyCFunction)PyUiItem_getAttribute, METH_VARARGS, PyUiItemGetAttribute_doc},
        {"getWindowFlags", (PyCFunction)PyUiItem_getWindowFlags, METH_NOARGS, PyUiItemGetWindowFlags_doc},
        {"setWindowFlags", (PyCFunction)PyUiItem_setWindowFlags, METH_VARARGS, PyUiItemSetWindowFlags_doc},
        {"invokeKeyboardInterrupt", (PyCFunction)PyUiItem_connectKeyboardInterrupt, METH_KEYWORDS | METH_VARARGS, PyUiItemConnectKeyboardInterrupt_doc},
        {"invokeProgressObserverCancellation", (PyCFunction)PyUiItem_connectProgressObserverInterrupt, METH_KEYWORDS | METH_VARARGS, PyUiItemConnectProgressObserverInterrupt_doc},
        {"info", (PyCFunction)PyUiItem_info, METH_VARARGS, PyUiItemInfo_doc},
        {"exists", (PyCFunction)PyUiItem_exists, METH_NOARGS, PyUiItemExists_doc},
        {"children", (PyCFunction)PyUiItem_children, METH_KEYWORDS | METH_VARARGS, PyUiItemChildren_doc},
        {"getChild", (PyCFunction)PyUiItem_getChild, METH_KEYWORDS | METH_VARARGS, PyUiItemGetChild_doc},
        {"getClassName", (PyCFunction)PyUiItem_getClassName, METH_NOARGS, PyUiItemGetClassName_doc},
        {"getLayout", (PyCFunction)PyUiItem_getLayout, METH_NOARGS, PyUiItemGetLayout_doc},
        {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMemberDef PythonUi::PyUiItem_members[] = {
        {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonUi::PyUiItemModule = {
        PyModuleDef_HEAD_INIT,
        "uiItem",
        "Any item of user interface (dialog, windows...). The item corresponds to any child-object of the overall dialog or window.",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyGetSetDef PythonUi::PyUiItem_getseters[] = {
    {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyTypeObject PythonUi::PyUiItemType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.uiItem",             /* tp_name */
        sizeof(PyUiItem),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyUiItem_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyUiItem_repr,         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        &PyUiItem_mappingProtocol,   /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        (getattrofunc)PyUiItem_getattro, /* tp_getattro */
        (setattrofunc)PyUiItem_setattro,  /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        PyUiItemInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,            /* tp_richcompare */
        offsetof(PyUiItem, weakreflist),    /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyUiItem_methods,             /* tp_methods */
        PyUiItem_members,             /* tp_members */
        PyUiItem_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyUiItem_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyUiItem_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//-------------------------------------------------------------------------------------
PyMappingMethods PythonUi::PyUiItem_mappingProtocol = {
    (lenfunc)PyUiItem_mappingLength,
    (binaryfunc)PyUiItem_mappingGetElem,
    (objobjargproc)PyUiItem_mappingSetElem
};

//-------------------------------------------------------------------------------------
void PythonUi::PyUiItem_addTpDict(PyObject * /*tp_dict*/)
{
    //nothing
}





//-------------------------------------------------------------------------------------
void PythonUi::PyUi_dealloc(PyUi* self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga && self->uiHandle >= 0)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        ito::RetVal retValue = retOk;

        QMetaObject::invokeMethod(
            uiOrga,
            "deleteDialog",
            Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
        );

        if(!locker.getSemaphore()->wait(PLUGINWAIT))
        {
            std::cerr << "timeout while closing dialog" << std::endl;
        }
    }

    DELETE_AND_SET_NULL( self->signalMapper );
    DELETE_AND_SET_NULL_ARRAY( self->filename );
    Py_XDECREF(self->dialogButtons);

    PyUiItemType.tp_dealloc( (PyObject*)self );
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUi_new(PyTypeObject *type, PyObject * args, PyObject * kwds)
{
    PyUi *self = (PyUi*)PyUiItemType.tp_new(type,args,kwds);

    if(self != NULL)
    {
        self->uiHandle = -1; //default: invalid
        self->dialogButtons = PyDict_New();
        PyObject *text = PythonQtConversion::ByteArrayToPyUnicode( "OK" );
        PyDict_SetItemString(self->dialogButtons, "AcceptRole", text);
        Py_DECREF(text);
        text = PythonQtConversion::ByteArrayToPyUnicode( "Cancel" );
        PyDict_SetItemString(self->dialogButtons, "RejectRole", text);
        Py_DECREF(text);
        text = NULL;
        self->winType = UiOrganizer::typeDialog;
        self->buttonBarType = UserUiDialog::bbTypeNo;
        self->childOfMainWindow = true; //default
        self->deleteOnClose = false; //default
        self->filename = NULL;
        self->signalMapper = NULL;
    }

    return (PyObject *)self;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiInit_doc,"ui(filename, type = ui.TYPEDIALOG, dialogButtonBar = ui.BUTTONBAR_NO, dialogButtons = {}, childOfMainWindow = True, deleteOnClose = False, dockWidgetArea = ui.TOPDOCKWIDGETAREA) -> ui \n\
\n\
Loads a user interface file (`ui`) and references this loaded interface by the new ui object. \n\
\n\
If the ui file is created in the `QtDesigner`, you can choose from which base type \n\
you would like to create the user interface (e.g. from a dialog, a window or a widget). \n\
This together with the argument ``type`` will mainly define the kind of user interface \n\
that is actually displayed in `itom`. \n\
\n\
If you want to add a customized user interface as toolbox or into the central part of \n\
the main window of `itom`, it is either recommended to design the interface from a \n\
widget or a main window. The latter has the advantage, that an individual menu or toolbar \n\
can be added. \n\
\n\
If you want to create a standalone window, it is recommended to already design the \n\
user interface from a main window, such that menus, toolbars as well as access to \n\
the statusbar is possible (if desired). \n\
\n\
For the creation of (modal) dialogs, where the user should configure settings or pass \n\
some inputs, it is recommended to either design the interface from a dialog on, or \n\
it is also possible to create a simple widget. In the latter case, itom will put \n\
this interface into a dialog (for ``type = ui.TYPEDIALOG``) and add optional buttons \n\
(like the ``OK`` and ``Cancel`` button). These buttons are then already configured \n\
to work. If you design a dialog from a dialog as base element, you have to connect \n\
buttons for instance with the ``accept()`` or ``reject()`` slot of the dialog by hand. \n\
\n\
For more information see also the section :ref:`qtdesigner` of the user documentation. \n\
\n\
Parameters \n\
---------- \n\
filename : str \n\
    path to the user interface file (.ui), absolute or relative to current directory. \n\
type : int, optional \n\
    This ``type`` defines how the loaded user interface is displayed: \n\
    \n\
    * ``ui.TYPEDIALOG`` (0): The ui-file is the content of a dialog window or, if the \n\
      file already defines a `QDialog`, this dialog is shown as it is. \n\
      This is recommended for the creation of modal dialogs, like settings... \n\
    * ``ui.TYPEWINDOW`` (1): The ui-file must be a `QMainWindow` or its outer widget \n\
      is turned into a main window. This window is then shown. This is recommended \n\
      for \"standalone\" windows, that should be able to be minimized, maximized, contain \n\
      menus or toolbars etc. \n\
    * ``ui.TYPEDOCKWIDGET`` (2): The loaded widget is the content of a dock widget (toolbox) \n\
      and is added to the indicated ``dockWidgetArea`` of the main window of `itom`. \n\
    * ``ui.TYPECENTRALWIDGET`` (3): The loaded ui-file must define a `QWidget` or \n\
      `QMainWindow` and is then added to the central area of `itom`, above the command line. \n\
      It is not allowed to choose this type if the user interface is created from \n\
      a `QDialog`. \n\
    \n\
dialogButtonBar : int, optional \n\
    This argument is only used if ``type == ui.TYPEDIALOG`` and defines if a button bar \n\
    with buttons, given by ``dialogButtons`` should be automatically added to the dialog. \n\
    If this is the case, the role of the buttons is considered, such that clicking the \n\
    ``OK`` or ``Cancel`` button  will automatically close the dialog and return the \n\
    role to the :meth:`show` method (if the dialog is displayed modal). Allowed values: \n\
    \n\
    * ``ui.BUTTONBAR_NO`` (0): do not add any button bar and buttons (default), \n\
    * ``ui.BUTTONBAR_HORIZONTAL`` (1): add a horizontal button bar at the bottom, \n\
    * ``ui.BUTTONBAR_VERTICAL`` (2): add vertical button bar on the right side. \n\
    \n\
dialogButtons : dict, optional \n\
    Only relevant if ``dialogButtonBar`` is not ``ui.BUTTONBAR_NO``: This dictionary \n\
    contains all buttons, that should be added to the button bar. For every entry, \n\
    the key is the role name of the button (enum ``QDialogButtonBox::ButtonRole``, \n\
    e.g. 'AcceptRole', 'RejectRole', 'ApplyRole', 'YesRole', 'NoRole'). The value is \n\
    the text of the button. \n\
childOfMainWindow : bool, optional \n\
    For type ``ui.TYPEDIALOG`` and ``ui.TYPEWINDOW`` only: Indicates if the window \n\
    should be a child of the itom main window. If ``False``, this window has its own \n\
    icon in the taskbar of the operating system. \n\
deleteOnClose : bool, optional \n\
    Indicates if the widget / window / dialog should be deleted if the user closes it \n\
    or if it is hidden. If it is hidden, it can be shown again using :meth:`show`. \n\
dockWidgetArea : int, optional \n\
    Only for ``type == ui.TYPEDOCKWIDGET (2)``. Indicates the position where the \n\
    dock widget should be placed: \n\
    \n\
    * 1 : ``ui.LEFTDOCKWIDGETAREA`` \n\
    * 2 : ``ui.RIGHTDOCKWIDGETAREA`` \n\
    * 4 : ``ui.TOPDOCKWIDGETAREA`` \n\
    * 8 : ``ui.BOTTOMDOCKWIDGETAREA`` \n\
\n\
Returns \n\
------- \n\
window : ui \n\
    A :class:`ui` object, that references the loaded ui-file.");
int PythonUi::PyUi_init(PyUi *self, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"filename", "type", "dialogButtonBar", "dialogButtons", "childOfMainWindow", "deleteOnClose", "dockWidgetArea", NULL};
    PyObject *dialogButtons = NULL;
    PyObject *tmp;
    PyBytesObject *bytesFilename = NULL; //weak reference
    char *internalFilename;
    int dockWidgetArea = Qt::TopDockWidgetArea;
    //PyUnicode_FSConverter

    if(args == NULL || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    if(!PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "O&|iiO!bbi",
        const_cast<char**>(kwlist),
        &PyUnicode_FSConverter, &bytesFilename,
        &self->winType,
        &self->buttonBarType,
        &PyDict_Type, &dialogButtons,
        &self->childOfMainWindow,
        &self->deleteOnClose, &dockWidgetArea))
    {
        return -1;
    }

    //check values:
    if(self->winType < 0 || self->winType > 3)
    {
        PyErr_SetString(
            PyExc_ValueError,
            "Argument 'type' must have one of the values TYPEDIALOG (0), TYPEWINDOW (1), TYPEDOCKWIDGET (2) or TYPECENTRALWIDGET (3)");
        Py_XDECREF(bytesFilename);
        return -1;
    }

    if(self->buttonBarType < 0 || self->buttonBarType > 2)
     {
        PyErr_SetString(
            PyExc_ValueError,
            "Argument 'dialogButtonBar' must have one of the values BUTTONBAR_NO (0), BUTTONBAR_HORIZONTAL (1) or BUTTONBAR_VERTICAL (2)");
        Py_XDECREF(bytesFilename);
        return -1;
    }


    DELETE_AND_SET_NULL_ARRAY(self->filename);
    internalFilename = PyBytes_AsString((PyObject*)bytesFilename);
    self->filename = new char[strlen(internalFilename)+1];
    strcpy_s(self->filename, strlen(internalFilename)+1, internalFilename);
    internalFilename = NULL;
    Py_XDECREF(bytesFilename);

    if(dialogButtons)
    {
        tmp = self->dialogButtons;
        Py_INCREF(dialogButtons);
        self->dialogButtons = dialogButtons;
        if(PyDict_Check(tmp)) PyDict_Clear(tmp);
        Py_XDECREF(tmp);
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    QSharedPointer<unsigned int> dialogHandle(new unsigned int);
    *dialogHandle = 0;
    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    StringMap dialogButtonMap;
    ito::RetVal retValue;

    if(self->dialogButtons)
    {
        //transfer dialogButtons dict to dialogButtonMap
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QString keyString, valueString;
        bool ok=false;

        while (PyDict_Next(self->dialogButtons, &pos, &key, &value))
        {
            keyString = PythonQtConversion::PyObjGetString(key,true,ok);
            valueString = PythonQtConversion::PyObjGetString(value,true,ok);
            if(keyString.isNull() || valueString.isNull())
            {
                std::cout << "Warning while parsing dialogButtons-dictionary. At least one element does not contain a string as key and value\n" << std::endl;
            }
            else
            {
                dialogButtonMap[keyString] = valueString;
            }
        }
    }

    int uiDescription = UiOrganizer::createUiDescription(self->winType, self->buttonBarType, self->childOfMainWindow, self->deleteOnClose, dockWidgetArea);
    QSharedPointer<QByteArray> className(new QByteArray());
    QSharedPointer<unsigned int> objectID(new unsigned int);

    QMetaObject::invokeMethod(
        uiOrga,
        "createNewDialog",
        Q_ARG(QString,QString(self->filename)),
        Q_ARG(int, uiDescription),
        Q_ARG(StringMap, dialogButtonMap),
        Q_ARG(QSharedPointer<uint>, dialogHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(QSharedPointer<QByteArray>, className),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));


    if(!locker.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while opening dialog");
        return -1;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return -1;
    }

    self->uiHandle = static_cast<int>(*dialogHandle);
    DELETE_AND_SET_NULL( self->signalMapper );
    self->signalMapper = new PythonQtSignalMapper();

    PyObject *args2 = PyTuple_New(3);
    PyTuple_SetItem(args2,0,PyLong_FromLong(*objectID) );
    PyTuple_SetItem(args2,1, PyUnicode_FromString("<ui>") );
    PyTuple_SetItem(args2,2, PyUnicode_FromString(className->data()) );
    int result = PyUiItemType.tp_init((PyObject*)self,args2,NULL);
    Py_DECREF(args2);

    return result;
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUi_repr(PyUi *self)
{
    PyObject *result;

    if(self->uiHandle < 0)
    {
        result = PyUnicode_FromFormat("Ui(empty)");
    }
    else
    {
        UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

        if (uiOrga == NULL)
        {
            if (self->filename == NULL)
            {
                result = PyUnicode_FromFormat("Ui(handle: %i) + %U", self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
            }
            else
            {
                result = PyUnicode_FromFormat("Ui(filename: '%s', handle: %i) + %U", self->filename, self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
            }
        }
        else
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QSharedPointer<bool> exist(new bool);

            QMetaObject::invokeMethod(
                uiOrga,
                "handleExist",
                Q_ARG(uint, self->uiHandle), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                Q_ARG(QSharedPointer<bool>, exist),
                Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
            );

            if (!locker.getSemaphore()->wait(PLUGINWAIT))
            {
                if (self->filename == NULL)
                {
                    result = PyUnicode_FromFormat("Ui(handle: %i) + %U", self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
                }
                else
                {
                    result = PyUnicode_FromFormat("Ui(filename: '%s', handle: %i) + %U", self->filename, self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
                }
            }
            else
            {
                if (*exist == true)
                {
                    if (self->filename == NULL)
                    {
                        result = PyUnicode_FromFormat("Ui(handle: %i) + %U", self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
                    }
                    else
                    {
                        result = PyUnicode_FromFormat("Ui(filename: '%s', handle: %i) + %U", self->filename, self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
                    }
                }
                else
                {
                    if (self->filename == NULL)
                    {
                        result = PyUnicode_FromFormat("Ui(handle: %i, ui is not longer available) + %U", self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
                    }
                    else
                    {
                        result = PyUnicode_FromFormat("Ui(filename: '%s', handle: %i, ui is not longer available) + %U", self->filename, self->uiHandle, PyUiItemType.tp_repr((PyObject*)self));
                    }
                }
            }
        }
    }
    return result;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiShow_doc,"show(modal = 0) -> Optional[int] \n\
\n\
Shows the window or dialog. \n\
\n\
Parameters \n\
---------- \n\
modal : int, optional \n\
    * 0: non-modal, the opened GUI does not block other windows of itom (default)\n\
    * 1: modal (python waits until dialog is hidden)\n\
    * 2: modal (python returns immediately)\n\
\n\
Returns \n\
------- \n\
None or int \n\
    Usually the value -1 is returned. Only if a dialog is shown with ``modal = 1``, \n\
    the exit code of the shown dialog is returned, once this dialog is closed again. \n\
    This code is: ``1`` if the dialog has been accepted (e.g. by closing it by an OK button \n\
    or ``0`` if the dialog has been rejected (Cancel button or directly closing the dialog \n\
    via the close icon in its title bar. \n\
\n\
See Also \n\
-------- \n\
hide");
PyObject* PythonUi::PyUi_show(PyUi *self, PyObject *args)
{
    int modalLevel = 0;

    if(!PyArg_ParseTuple(args, "|i", &modalLevel))
    {
        PyErr_SetString(PyExc_RuntimeError, "Parameter modal must be a boolean value");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of Ui");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<int> retCodeIfModal(new int);
    *retCodeIfModal = -1;
    ito::RetVal retValue = retOk;

    //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    QMetaObject::invokeMethod(
        uiOrga,
        "showDialog",
        Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)) ,
        Q_ARG(int,modalLevel),
        Q_ARG(QSharedPointer<int>, retCodeIfModal),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(modalLevel == 1)
    {
        Py_BEGIN_ALLOW_THREADS //release GIL when showing a modal dialog in order not to block background python operations during the visibility of the dialog

        locker.getSemaphore()->waitAndProcessEvents(-1);

        Py_END_ALLOW_THREADS //re-acquire GIL to process
    }
    else
    {
        if(!locker.getSemaphore()->wait(30000))
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
            return NULL;
        }
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(*retCodeIfModal >= 0)
    {
        return Py_BuildValue("i",*retCodeIfModal);
    }
    else
    {
        Py_RETURN_NONE;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiHide_doc, "hide() \n\
\n\
Hides the user interface reference by this ui object. \n\
\n\
A hidden window or dialog can be shown again via the method :py:meth:`show`.\n\
\n\
See Also \n\
-------- \n\
show");
PyObject* PythonUi::PyUi_hide(PyUi *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of Ui");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    QMetaObject::invokeMethod(
        uiOrga,
        "hideDialog",
        Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while hiding dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiIsVisible_doc,"isVisible() -> bool \n\
\n\
Returns ``True`` if the referenced window or dialog is still visible. \n\
\n\
Returns \n\
------- \n\
visible : bool \n\
    ``True`` if user interface is visible, ``False`` if it is hidden.");
PyObject* PythonUi::PyUi_isVisible(PyUi *self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    if(self->uiHandle < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "No valid dialog handle is assigned to this instance of Ui");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    QSharedPointer<bool> visible(new bool);
    *visible = false;
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(
        uiOrga,
        "isVisible",
        Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QSharedPointer<bool>, visible),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting visible status");
        return NULL;
    }

    if(*visible)
    {
        Py_RETURN_TRUE;
    }
    else
    {
        Py_RETURN_FALSE;
    }
}


//#################################################################################################################
//
//      STATIC METHODS OF UI
//
//#################################################################################################################

int PyUiItem_Converter(PyObject *object, PythonUi::PyUiItem **address)
{
    if (object == NULL || object == Py_None)
    {
        *address = NULL;
        return 1;
    }

    if (PyUiItem_Check(object))
    {
        *address = (PythonUi::PyUiItem*)object;
        return 1;
    }
    else if (PyUi_Check(object))
    {
        *address = (PythonUi::PyUiItem*)object;
        return 1;
    }
    else if (PyFigure_Check(object))
    {
        *address = (PythonUi::PyUiItem*)object;
        return 1;
    }
    else if (PyPlotItem_Check(object))
    {
        *address = (PythonUi::PyUiItem*)object;
        return 1;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "argument 'parent' must be a type derived from uiItem (e.g. ui, plotItem, figure, uiItem)");
        *address = NULL;
        return 0;
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetDouble_doc,"getDouble(title, label, defaultValue, min = -2147483647.0, max = 2147483647.0, decimals = 1, parent = None) -> Tuple[float, bool] \n\
\n\
Shows a dialog to get a float value from the user. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str\n\
    is the title of the dialog. \n\
label : str \n\
    is the label above the input box. \n\
defaultValue : float \n\
    is the default value in the input box. \n\
min : float, optional \n\
    is the allowed minimal value. \n\
max : float, optional \n\
    is the allowed maximal value. \n\
decimals : int, optional \n\
    the maximum number of decimal places. \n\
parent : uiItem, optional \n\
    the dialog is modal with respect to ``parent`` or with respect to the \n\
    main window of `itom`, if ``None``. \n\
\n\
Returns \n\
------- \n\
value : float \n\
    The entered float value. \n\
success : bool \n\
    ``True`` if the dialog has been accepted, otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
getInt, getText, getItem");
PyObject* PythonUi::PyUi_getDouble(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultValue", "min", "max", "decimals", "parent", NULL};
    PyObject *titleObj = NULL;
    PyObject *labelObj = NULL;
    QString title;
    QString label;
    double defaultValue = 0;
    double minValue = -2147483647;
    double maxValue = 2147483647;
    int decimals = 1;
    PythonUi::PyUiItem *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOd|ddiO&", const_cast<char**>(kwlist), &titleObj, &labelObj, &defaultValue, &minValue, &maxValue, &decimals, &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok;
    title = PythonQtConversion::PyObjGetString(titleObj, true, ok);
    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "title must be a string.");
        return NULL;
    }

    label = PythonQtConversion::PyObjGetString(labelObj, true, ok);
    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "label must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = ito::retOk;

    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<double> retDblValue(new double);
    *retDblValue = defaultValue;
    unsigned int objectID= parentItem ? parentItem->objectID : 0;

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetDouble", Q_ARG(uint, objectID), Q_ARG(QString, title), Q_ARG(QString, label), Q_ARG(double, defaultValue), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<double>, retDblValue), Q_ARG(double,minValue), Q_ARG(double,maxValue), Q_ARG(int,decimals), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

#if QT_VERSION < 0x050600
    //workaround for special notebook ;)
    //A simple wait(-1) sometimes lead to a deadlock when pushing any arrow key
    //therefore we implemented this special while-wait-combination. The simple
    //call of hasPendingEvents was sufficient to avoid the deadlock.
    //counter is incremented in both cases in order to avoid that this case
    //is deleted in optimized release compilation
    int timeout = -1; //set the real timeout here (ms)
    int counter = 0;
    int c = 0;

    while (!locker.getSemaphore()->wait(100))
    {
        counter++;

        if (QCoreApplication::hasPendingEvents())
        {
            c++; //dummy action
            //QCoreApplication::processEvents(); //it is not necessary to call this here
        }

        if (timeout >= 0 && counter > (timeout / 100) && c >= 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
            return NULL;
        }
    }
#else
    locker.getSemaphore()->wait(-1);
#endif

    if(*retOk == true)
    {
        return Py_BuildValue("dO", *retDblValue, Py_True );
    }
    else
    {
        return Py_BuildValue("dO", defaultValue, Py_False );
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetInt_doc,"getInt(title, label, defaultValue, min = -2147483647, max = 2147483647, step = 1, parent = None) -> Tuple[int, bool] \n\
\n\
Shows a dialog to get an integer value from the user. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str\n\
    is the title of the dialog. \n\
label : str\n\
    is the label above the input box. \n\
defaultValue : int\n\
    is the default value in the input box. \n\
min : int, optional\n\
    is the allowed minimal value. \n\
max : int, optional\n\
    is the allowed maximal value. \n\
step : int, optional\n\
    is the step size if user presses the up/down arrow. \n\
parent : uiItem, optional \n\
    the dialog is modal with respect to ``parent`` or with respect to the \n\
    main window of `itom`, if ``None``. \n\
\n\
Returns \n\
------- \n\
value : int \n\
    The entered integer value. \n\
success : bool \n\
    ``True`` if the dialog has been accepted, otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
getDouble, getText, getItem");
PyObject* PythonUi::PyUi_getInt(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultValue", "min", "max", "step", "parent", NULL};
    PyObject *titleObj = NULL;
    PyObject *labelObj = NULL;
    QString title;
    QString label;
    int defaultValue = 0;
    int minValue = -2147483647;
    int maxValue = 2147483647;
    int step = 1;
    PythonUi::PyUiItem *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOi|iiiO&", const_cast<char**>(kwlist), &titleObj, &labelObj, &defaultValue, &minValue, &maxValue, &step, &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok;
    title = PythonQtConversion::PyObjGetString(titleObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "title must be a string.");
        return NULL;
    }

    label = PythonQtConversion::PyObjGetString(labelObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "label must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = ito::retOk;

    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<int> retIntValue(new int);
    *retIntValue = defaultValue;
    unsigned int objectID = parentItem ? parentItem->objectID : 0;

    QMetaObject::invokeMethod(
        uiOrga,
        "showInputDialogGetInt",
        Q_ARG(uint, objectID),
        Q_ARG(QString, title),
        Q_ARG(QString, label),
        Q_ARG(int, defaultValue),
        Q_ARG(QSharedPointer<bool>, retOk),
        Q_ARG(QSharedPointer<int>, retIntValue),
        Q_ARG(int,minValue),
        Q_ARG(int,maxValue),
        Q_ARG(int,step),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

#if QT_VERSION < 0x050600
    //workaround for special notebook ;)
    //A simple wait(-1) sometimes lead to a deadlock when pushing any arrow key
    //therefore we implemented this special while-wait-combination. The simple
    //call of hasPendingEvents was sufficient to avoid the deadlock.
    //counter is incremented in both cases in order to avoid that this case
    //is deleted in optimized release compilation
    int timeout = -1; //set the real timeout here (ms)
    int counter = 0;
    int c = 0;

    while (!locker.getSemaphore()->wait(100))
    {
        counter++;

        if (QCoreApplication::hasPendingEvents())
        {
            c++; //dummy action
            //QCoreApplication::processEvents(); //it is not necessary to call this here
        }

        if (timeout >= 0 && counter > (timeout / 100) && c >= 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
            return NULL;
        }
    }
#else
    locker.getSemaphore()->wait(-1);
#endif

    if (*retOk == true)
    {
        return Py_BuildValue("iO", *retIntValue, Py_True );
    }
    else
    {
        return Py_BuildValue("iO", defaultValue, Py_False );
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetItem_doc,"getItem(title, label, stringList, currentIndex = 0, editable = False, parent = None) -> Tuple[str, bool] \n\
\n\
Shows a dialog to let the user select an item from a string list. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str \n\
    is the title of the dialog.\n\
label : str \n\
    is the label above the text box. \n\
stringList : list of str or tuple of str \n\
    is a list or tuple of possible string values. \n\
currentIndex : int, optional\n\
    defines the pre-selected value index from ``stringList``. \n\
editable : bool, optional\n\
    defines whether new entries can be added (``True``) or not (``False``) \n\
parent : uiItem, optional\n\
    the dialog is modal with respect to ``parent`` or with respect to the \n\
    main window of `itom`, if ``None``. \n\
\n\
Returns \n\
------- \n\
value : str \n\
    The currently selected or entered string value. \n\
success : bool \n\
    ``True`` if the dialog has been accepted, otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
getInt, getDouble, getText");
PyObject* PythonUi::PyUi_getItem(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "stringList", "currentIndex", "editable","parent", NULL};
    PyObject *titleObj = NULL;
    PyObject *labelObj = NULL;
    QString title;
    QString label;
    PyObject *stringList = NULL;
    int currentIndex = 0;
    bool editable = false;
    QStringList stringListQt;
    QString temp;
    PythonUi::PyUiItem *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|ibO&", const_cast<char**>(kwlist), &titleObj, &labelObj, &stringList, &currentIndex, &editable, &PyUiItem_Converter ,&parentItem))
    {
        return NULL;
    }

    bool ok;
    title = PythonQtConversion::PyObjGetString(titleObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "title must be a string.");
        return NULL;
    }

    label = PythonQtConversion::PyObjGetString(labelObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "label must be a string.");
        return NULL;
    }

    if(!PySequence_Check(stringList))
    {
        PyErr_SetString(PyExc_TypeError, "string list must be a sequence of elements (tuple or list)");
        return NULL;
    }
    else
    {
        Py_ssize_t length = PySequence_Size(stringList);
        PyObject *stringListItem = NULL;
        bool ok = false;

        for(Py_ssize_t i = 0 ; i < length ; i++)
        {
            stringListItem = PySequence_GetItem(stringList,i); //new reference
            temp = PythonQtConversion::PyObjGetString(stringListItem,true,ok);
            Py_XDECREF(stringListItem);

            if(!temp.isNull())
            {
                stringListQt << temp;
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "string list must only contain string values!");
                return NULL;
            }
        }
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<QString> retString(new QString());
    unsigned int objectID = parentItem ? parentItem->objectID : 0;

    QMetaObject::invokeMethod(
        uiOrga,
        "showInputDialogGetItem",
        Q_ARG(uint, objectID),
        Q_ARG(QString, title),
        Q_ARG(QString, label),
        Q_ARG(QStringList, stringListQt),
        Q_ARG(QSharedPointer<bool>, retOk),
        Q_ARG(QSharedPointer<QString>, retString),
        Q_ARG(int, currentIndex),
        Q_ARG(bool, editable),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

#if QT_VERSION < 0x050600
    //workaround for special notebook ;)
    //A simple wait(-1) sometimes lead to a deadlock when pushing any arrow key
    //therefore we implemented this special while-wait-combination. The simple
    //call of hasPendingEvents was sufficient to avoid the deadlock.
    //counter is incremented in both cases in order to avoid that this case
    //is deleted in optimized release compilation
    int timeout = -1; //set the real timeout here (ms)
    int counter = 0;
    int c=0;

    while(!locker.getSemaphore()->wait(100))
    {
        counter++;

        if (QCoreApplication::hasPendingEvents())
        {
            c++; //dummy action
            //QCoreApplication::processEvents(); //it is not necessary to call this here
        }

        if (timeout >= 0 && counter > (timeout / 100) && c>=0)
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
            return NULL;
        }
    }
#else
    locker.getSemaphore()->wait(-1);
#endif

    if(*retOk == true)
    {
        return Py_BuildValue("NO", PythonQtConversion::QStringToPyObject(*retString), Py_True ); //"N" -> Py_BuildValue steals reference from QStringToPyObject
    }
    else
    {
        return Py_BuildValue("sO", "", Py_False );
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetText_doc,"getText(title, label, defaultString, parent = None) -> Tuple[str, bool] \n\
\n\
Opens a dialog to ask the user for a string value. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str \n\
    is the title of the dialog. \n\
label : str \n\
    is the label above the text box. \n\
defaultString : str \n\
    is the default string in the text box. \n\
parent : uiItem, optional \n\
    the dialog is modal with respect to ``parent`` or with respect to the \n\
    main window of `itom`, if ``None``. \n\
\n\
Returns \n\
------- \n\
value : str \n\
    The entered string value. \n\
success : bool \n\
    ``True`` if dialog has been accepted, otherwise ``False``. \n\
\n\
See Also \n\
-------- \n\
getInt, getDouble, getItem");
PyObject* PythonUi::PyUi_getText(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultString", "parent", NULL};
    PyObject *titleObj = NULL;
    PyObject *labelObj = NULL;
    PyObject *defaultObj = NULL;
    QString title;
    QString label;
    QString defaultString;
    PythonUi::PyUiItem *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|O&", const_cast<char**>(kwlist), &titleObj, &labelObj, &defaultObj, &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok;
    title = PythonQtConversion::PyObjGetString(titleObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "title must be a string.");
        return NULL;
    }

    label = PythonQtConversion::PyObjGetString(labelObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "label must be a string.");
        return NULL;
    }

    defaultString = PythonQtConversion::PyObjGetString(defaultObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "default string must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<bool> retOk(new bool);
    *retOk = false;
    QSharedPointer<QString> retStringValue(new QString(defaultString));
    unsigned int objectID = parentItem ? parentItem->objectID : 0;

    QMetaObject::invokeMethod(
        uiOrga,
        "showInputDialogGetText",
        Q_ARG(uint,objectID),
        Q_ARG(QString, title),
        Q_ARG(QString, label),
        Q_ARG(QString, defaultString),
        Q_ARG(QSharedPointer<bool>, retOk),
        Q_ARG(QSharedPointer<QString>, retStringValue),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

#if QT_VERSION < 0x050600
    //workaround for special notebook ;)
    //A simple wait(-1) sometimes lead to a deadlock when pushing any arrow key
    //therefore we implemented this special while-wait-combination. The simple
    //call of hasPendingEvents was sufficient to avoid the deadlock.
    //counter is incremented in both cases in order to avoid that this case
    //is deleted in optimized release compilation
    int timeout = -1; //set the real timeout here (ms)
    int counter = 0;
    int c = 0;

    while (!locker.getSemaphore()->wait(100))
    {
        counter++;

        if (QCoreApplication::hasPendingEvents())
        {
            c++; //dummy action
            //QCoreApplication::processEvents(); //it is not necessary to call this here
        }

        if (timeout >= 0 && counter > (timeout / 100) && c >= 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
            return NULL;
        }
    }
#else
    locker.getSemaphore()->wait(-1);
#endif

    if(*retOk == true)
    {
        return Py_BuildValue("NO", PythonQtConversion::QStringToPyObject(*retStringValue), Py_True ); //"N" -> Py_BuildValue steals reference from QStringToPyObject
    }
    else
    {
        return Py_BuildValue("NO", PythonQtConversion::QStringToPyObject(defaultString), Py_False ); //"N" -> Py_BuildValue steals reference from QStringToPyObject
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgInformation_doc,"msgInformation(title, text, buttons = ui.MsgBoxOk, defaultButton = 0, parent = None) -> Tuple[int, str] \n\
\n\
Opens an information message box. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str \n\
    is the title of the message box. \n\
text : str \n\
    is the message text \n\
buttons : int, optional \n\
    is a flag value (bitmask) of the constants ``ui.MsgBoxXYZ``, where ``XYZ`` is \n\
    a placeholder for different values. Each selected constant indicates the \n\
    corresponding button to display (combine values be the | operator). \n\
defaultButton : int, optional \n\
    is the button constant (see ``buttons``, that should be set as default. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
buttonID : int \n\
    constant of the button that has been clicked to close the message box. \n\
buttonText : str \n\
    caption of the button that has been clicked to close the message box. \n\
\n\
See Also \n\
-------- \n\
msgCritical, msgQuestion, msgWarning");
PyObject* PythonUi::PyUi_msgInformation(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,1);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgQuestion_doc,"msgQuestion(title, text, buttons = ui.MsgBoxOk, defaultButton = 0, parent = None) -> Tuple[int, str] \n\
\n\
Opens a question message box. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str \n\
    is the title of the message box. \n\
text : str \n\
    is the message text \n\
buttons : int, optional \n\
    is a flag value (bitmask) of the constants ``ui.MsgBoxXYZ``, where ``XYZ`` is \n\
    a placeholder for different values. Each selected constant indicates the \n\
    corresponding button to display (combine values be the | operator). \n\
defaultButton : int, optional \n\
    is the button constant (see ``buttons``, that should be set as default. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
buttonID : int \n\
    constant of the button that has been clicked to close the message box. \n\
buttonText : str \n\
    caption of the button that has been clicked to close the message box. \n\
\n\
See Also \n\
-------- \n\
msgCritical, msgWarning, msgInformation");
PyObject* PythonUi::PyUi_msgQuestion(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,2);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgWarning_doc,"msgWarning(title, text, buttons = ui.MsgBoxOk, defaultButton = 0, parent = None) -> Tuple[int, str] \n\
\n\
Opens a warning message box. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str \n\
    is the title of the message box. \n\
text : str \n\
    is the message text \n\
buttons : int, optional \n\
    is a flag value (bitmask) of the constants ``ui.MsgBoxXYZ``, where ``XYZ`` is \n\
    a placeholder for different values. Each selected constant indicates the \n\
    corresponding button to display (combine values be the | operator). \n\
defaultButton : int, optional \n\
    is the button constant (see ``buttons``, that should be set as default. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
buttonID : int \n\
    constant of the button that has been clicked to close the message box. \n\
buttonText : str \n\
    caption of the button that has been clicked to close the message box. \n\
\n\
See Also \n\
-------- \n\
msgCritical, msgQuestion, msgInformation");
PyObject* PythonUi::PyUi_msgWarning(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,3);
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgCritical_doc,"msgCritical(title, text, buttons = ui.MsgBoxOk, defaultButton = 0, parent = None) -> Tuple[int, str] \n\
\n\
Opens a critical message box. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
title : str \n\
    is the title of the message box. \n\
text : str \n\
    is the message text \n\
buttons : int, optional \n\
    is a flag value (bitmask) of the constants ``ui.MsgBoxXYZ``, where ``XYZ`` is \n\
    a placeholder for different values. Each selected constant indicates the \n\
    corresponding button to display (combine values be the | operator). \n\
defaultButton : int, optional \n\
    is the button constant (see ``buttons``, that should be set as default. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
buttonID : int \n\
    constant of the button that has been clicked to close the message box. \n\
buttonText : str \n\
    caption of the button that has been clicked to close the message box. \n\
\n\
See Also \n\
-------- \n\
msgWarning, msgQuestion, msgInformation");
PyObject* PythonUi::PyUi_msgCritical(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,4);
}

//-------------------------------------------------------------------------------------
PyObject* PythonUi::PyUi_msgGeneral(PyUi * /*self*/, PyObject *args, PyObject *kwds, int type)
{
    const char *kwlist[] = {"title", "text", "buttons", "defaultButton", "parent", NULL};
    PyObject *titleObj = NULL;
    PyObject *textObj = NULL;
    QString title;
    QString text;
    int buttons = QMessageBox::Ok;
    int defaultButton = QMessageBox::NoButton;
    PythonUi::PyUiItem *parentItem = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iiO&", const_cast<char**>(kwlist), &titleObj, &textObj, &buttons, &defaultButton, &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok;
    title = PythonQtConversion::PyObjGetString(titleObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "title must be a string.");
        return NULL;
    }

    text = PythonQtConversion::PyObjGetString(textObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "text must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<int> retButton(new int);
    *retButton = QMessageBox::Escape;
    QSharedPointer<QString> retButtonText(new QString());
    unsigned int objectID = parentItem ? parentItem->objectID : 0;

    QMetaObject::invokeMethod(
        uiOrga,
        "showMessageBox",
        Q_ARG(uint, objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(int, type),
        Q_ARG(QString, title),
        Q_ARG(QString, text),
        Q_ARG(int, buttons),
        Q_ARG(int, defaultButton),
        Q_ARG(QSharedPointer<int>, retButton),
        Q_ARG(QSharedPointer<QString>, retButtonText),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

#if QT_VERSION < 0x050600
    //workaround for special notebook ;)
    //A simple wait(-1) sometimes lead to a deadlock when pushing any arrow key
    //therefore we implemented this special while-wait-combination. The simple
    //call of hasPendingEvents was sufficient to avoid the deadlock.
    //counter is incremented in both cases in order to avoid that this case
    //is deleted in optimized release compilation
    int timeout = -1; //set the real timeout here (ms)
    int counter = 0;
    int c = 0;

    while (!locker.getSemaphore()->wait(100))
    {
        counter++;

        if (QCoreApplication::hasPendingEvents())
        {
            c++; //dummy action
            //QCoreApplication::processEvents(); //it is not necessary to call this here
        }

        if (timeout >= 0 && counter > (timeout / 100) && c >= 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
            return NULL;
        }
    }
#else
    locker.getSemaphore()->wait(-1);
#endif

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    return Py_BuildValue("iN", *retButton, PythonQtConversion::QStringToPyObject(*retButtonText)); //"N" -> Py_BuildValue steals reference from QStringToPyObject
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetExistingDirectory_doc,"getExistingDirectory(caption, startDirectory, options = 0, parent = None) -> Optional[str] \n\
\n\
Opens a dialog to choose an existing directory. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
caption : str \n\
    is the caption of this dialog. \n\
startDirectory : str \n\
    is the start directory, visible in the dialog. \n\
options : int, optional\n\
    is a flag value (bitmask) of the following options (see ``QFileDialog::Option``): \n\
    \n\
    * 1: ShowDirsOnly [default] \n\
    * 2: DontResolveSymlinks \n\
    * ... (for others see Qt-Help) \n\
    \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
directory : None or str \n\
    The absolute path of the selected directory is returned or ``None`` if the dialog \n\
    has been rejected. \n\
\n\
See Also \n\
-------- \n\
getSaveFileName, getOpenFileName");
PyObject* PythonUi::PyUi_getExistingDirectory(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"caption", "directory", "options", "parent", NULL};
    PyObject *captionObj = NULL;
    PyObject *directoryObj = NULL;
    QString caption;
    QString directory;
    int options = 1; //QFileDialog::ShowDirsOnly
    PythonUi::PyUiItem *parentItem = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iO&", const_cast<char**>(kwlist), &captionObj, &directoryObj, &options, &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok;
    caption = PythonQtConversion::PyObjGetString(captionObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "caption must be a string.");
        return NULL;
    }

    directory = PythonQtConversion::PyObjGetString(directoryObj, true, ok);

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "directory must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    unsigned int objectID = parentItem ? parentItem->objectID : 0;
    QSharedPointer<QString> sharedDir(new QString(directory));

    QMetaObject::invokeMethod(
        uiOrga,
        "showFileDialogExistingDir",
        Q_ARG(uint, objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QString, caption),
        Q_ARG(QSharedPointer<QString>, sharedDir),
        Q_ARG(int, options),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if(sharedDir->isEmpty() || sharedDir->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PythonQtConversion::QStringToPyObject(*sharedDir);
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetOpenFileNames_doc,
"getOpenFileNames(caption = \"\", startDirectory = \"\", filters = \"\", selectedFilterIndex = 0, options = 0, parent = None) -> Optional[List[str]] \n\
\n\
Shows a dialog for chosing one or multiple file names. The selected file(s) must exist. \n\
\n\
This method creates a modal file dialog to let the user select one or multiple file \n\
names used for opening these files. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
caption : str, optional \n\
    This is the title of the dialog. \n\
startDirectory : str, optional \n\
    The intial directory, shown in the dialog. If an empty string, the current working \n\
    directory will be taken. \n\
filters : str, optional \n\
    Possible filter list or allowed file types / suffixes etc. The entries should be \n\
    separated by ``;;``, for example ``Images (*.png *.jpg);;Text files (*.txt)``. \n\
selectedFilterIndex : int, optional \n\
    The index of the currently selected filter from ``filters``. \n\
options : int, optional\n\
    This corresponds to the Qt flag ``QFileDialog::Options``. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
selectedFileNames : None or list of str \n\
    The selected file pathes or ``None`` if the dialog has been aborted. \n\
\n\
See Also \n\
-------- \n\
getOpenFileName, getSaveFileName");
PyObject* PythonUi::PyUi_getOpenFileNames(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "caption", "startDirectory", "filters", "selectedFilterIndex", "options", "parent", NULL };
    PyObject *captionObj = NULL;
    PyObject *directoryObj = NULL;
    PyObject *filtersObj = NULL;
    QString caption;
    QString directory;
    QString filters;
    int selectedFilterIndex = 0;
    int options = 0;
    PythonUi::PyUiItem *parentItem = NULL;

    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "|OOOiiO&",
        const_cast<char**>(kwlist),
        &captionObj,
        &directoryObj,
        &filtersObj,
        &selectedFilterIndex,
        &options,
        &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok = true;

    caption = captionObj ? PythonQtConversion::PyObjGetString(captionObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "caption must be a string.");
        return NULL;
    }

    directory = directoryObj ? PythonQtConversion::PyObjGetString(directoryObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "directory must be a string.");
        return NULL;
    }

    filters = filtersObj ? PythonQtConversion::PyObjGetString(filtersObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "filters must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available.");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    unsigned int objectID = parentItem ? parentItem->objectID : 0;
    QSharedPointer<QStringList> files(new QStringList());

    QMetaObject::invokeMethod(
        uiOrga,
        "showFilesOpenDialog",
        Q_ARG(uint, objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QString, caption),
        Q_ARG(QString, directory),
        Q_ARG(QString, filters),
        Q_ARG(QSharedPointer<QStringList>, files),
        Q_ARG(int, selectedFilterIndex),
        Q_ARG(int, options),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if (files->isEmpty())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PythonQtConversion::QStringListToPyObject(*files);
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetOpenFileName_doc,
"getOpenFileName(caption = \"\", startDirectory = \"\", filters = \"\", selectedFilterIndex = 0, options = 0, parent = None) -> Optional[str] \n\
\n\
Shows a dialog for chosing a file name. The selected file must exist. \n\
\n\
This method creates a modal file dialog to let the user select a file name used for opening a file. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
caption : str, optional \n\
    This is the title of the dialog. \n\
startDirectory : str, optional \n\
    The intial directory, shown in the dialog. If an empty string, the current working \n\
    directory will be taken. \n\
filters : str, optional \n\
    Possible filter list or allowed file types / suffixes etc. The entries should be \n\
    separated by ``;;``, for example ``Images (*.png *.jpg);;Text files (*.txt)``. \n\
selectedFilterIndex : int, optional \n\
    The index of the currently selected filter from ``filters``. \n\
options : int, optional\n\
    This corresponds to the Qt flag ``QFileDialog::Options``. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
selectedFileName : None or str \n\
    The selected file path or ``None`` if the dialog has been aborted. \n\
\n\
See Also \n\
-------- \n\
getOpenFileNames, getSaveFileName");
PyObject* PythonUi::PyUi_getOpenFileName(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{

    const char *kwlist[] = {"caption", "startDirectory", "filters", "selectedFilterIndex", "options", "parent", NULL};
    PyObject *captionObj = NULL;
    PyObject *directoryObj = NULL;
    PyObject *filtersObj = NULL;
    QString caption;
    QString directory;
    QString filters;
    int selectedFilterIndex = 0;
    int options = 0;
    PythonUi::PyUiItem *parentItem = NULL;

    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "|OOOiiO&",
        const_cast<char**>(kwlist),
        &captionObj,
        &directoryObj,
        &filtersObj,
        &selectedFilterIndex,
        &options,
        &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok = true;
    caption = captionObj ? PythonQtConversion::PyObjGetString(captionObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "caption must be a string.");
        return NULL;
    }

    directory = directoryObj ? PythonQtConversion::PyObjGetString(directoryObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "directory must be a string.");
        return NULL;
    }

    filters = filtersObj ? PythonQtConversion::PyObjGetString(filtersObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "filters must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    unsigned int objectID = parentItem ? parentItem->objectID : 0;
    QSharedPointer<QString> file(new QString());

    QMetaObject::invokeMethod(
        uiOrga,
        "showFileOpenDialog",
        Q_ARG(uint, objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QString, caption),
        Q_ARG(QString, directory),
        Q_ARG(QString, filters),
        Q_ARG(QSharedPointer<QString>, file),
        Q_ARG(int, selectedFilterIndex),
        Q_ARG(int, options),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if(file->isEmpty() || file->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PythonQtConversion::QStringToPyObject(*file);
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetSaveFileName_doc,
"getSaveFileName(caption = \"\", startDirectory = \"\", filters = \"\", selectedFilterIndex = 0, options = 0, parent = None) -> Optional[str] \n\
\n\
Shows a dialog for chosing a file name. The selected file must not exist. \n\
\n\
This method creates a modal file dialog to let the user select a file name used for saving a file. \n\
\n\
For more information, see also the section :ref:`msgInputBoxes` of the documentation. \n\
\n\
Parameters \n\
---------- \n\
caption : str, optional \n\
    This is the title of the dialog. \n\
startDirectory : str, optional \n\
    The intial directory, shown in the dialog. If an empty string, the current working \n\
    directory will be taken. \n\
filters : str, optional \n\
    Possible filter list or allowed file types / suffixes etc. The entries should be \n\
    separated by ``;;``, for example ``Images (*.png *.jpg);;Text files (*.txt)``. \n\
selectedFilterIndex : int, optional \n\
    The index of the currently selected filter from ``filters``. \n\
options : int, optional\n\
    This corresponds to the Qt flag ``QFileDialog::Options``. \n\
parent : uiItem, optional \n\
    If not ``None``, the dialog will be shown modal to this ``parent`` window. \n\
    Else, it is modal with respect to the main window of `itom`. \n\
\n\
Returns \n\
------- \n\
selectedFileName : None or str \n\
    The selected file path or ``None`` if the dialog has been aborted. \n\
\n\
See Also \n\
-------- \n\
getOpenFileName");
PyObject* PythonUi::PyUi_getSaveFileName(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{

    const char *kwlist[] = {"caption", "startDirectory", "filters", "selectedFilterIndex", "options", "parent", NULL};
    PyObject *captionObj = NULL;
    PyObject *directoryObj = NULL;
    PyObject *filtersObj = NULL;
    QString caption;
    QString directory;
    QString filters;
    int selectedFilterIndex = 0;
    int options = 0;
    PythonUi::PyUiItem *parentItem = NULL;

    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "|OOOiiO&",
        const_cast<char**>(kwlist),
        &captionObj,
        &directoryObj,
        &filtersObj,
        &selectedFilterIndex,
        &options,
        &PyUiItem_Converter, &parentItem))
    {
        return NULL;
    }

    bool ok = true;

    caption = captionObj ? PythonQtConversion::PyObjGetString(captionObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "caption must be a string.");
        return NULL;
    }

    directory = directoryObj ? PythonQtConversion::PyObjGetString(directoryObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "directory must be a string.");
        return NULL;
    }

    filters = filtersObj ? PythonQtConversion::PyObjGetString(filtersObj, true, ok) : "";

    if (!ok)
    {
        PyErr_SetString(PyExc_TypeError, "filters must be a string.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    unsigned int objectID = parentItem ? parentItem->objectID : 0;
    QSharedPointer<QString> file(new QString());

    QMetaObject::invokeMethod(
        uiOrga,
        "showFileSaveDialog",
        Q_ARG(uint, objectID), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
        Q_ARG(QString, caption),
        Q_ARG(QString, directory),
        Q_ARG(QString, filters),
        Q_ARG(QSharedPointer<QString>, file),
        Q_ARG(int, selectedFilterIndex),
        Q_ARG(int, options),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())
    );

    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    if(file->isEmpty() || file->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return PythonQtConversion::QStringToPyObject(*file);
    }
}

//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiCreateNewPluginWidget_doc, "createNewPluginWidget(widgetName, *args, **kwds) -> ui \n\
\n\
Loads a widget, defined in an itom algorithm plugin, and returns the :class:`ui` object, that references this widget. \n\
\n\
Itom algorithm plugins cannot only contain algorithms, callable by Python, but also \n\
methods, that return a customized user-interface, widget etc. Use this method to \n\
initialize such an user-interface and returns its corresponding :class:`ui` object. \n\
\n\
For a list of available widget methods, see :meth:`widgetHelp`. Compared to the more \n\
detailed method :meth:`createNewPluginWidget2`, this method uses the following defaults \n\
for the windows appearance: \n\
\n\
* The ``type`` of the widget is derived from the widget itself and cannot be adjusted, \n\
* ``deleteOnClose = False``: The widget or windows will only be hidden if the user \n\
  clicks the close button, \n\
* ``childOfMainWindow = True``: The widget or windows is a child of the main window \n\
  without own symbol in the taskbar, \n\
* ``dockWidgetArea = ui.TOPDOCKWIDGETAREA``: If the widget is derived from `QDockWidget`, \n\
  the dock widget is docked at that location \n\
* ``buttonBarType = ui.BUTTONBAR_NO``, if a dialog is created (if the plugin delivers a \n\
  widget and no windows, dialog or dock widget), the dialog has no automatically \n\
  generated ``OK``, ``Cancel``, ``...`` buttons \n\
\n\
If you want to have other default parameters than these ones, call :meth:`createNewPluginWidget2`. \n\
\n\
Parameters \n\
---------- \n\
widgetName : str \n\
    Name of algorithm widget method. \n\
*args \n\
    Further positional arguments, that are parsed and passed to the widget creation method. \n\
    These arguments are used first to initialize all mandatory parameters, followed by \n\
    the optional ones. \n\
**kwds \n\
    Keyword-based arguments, that are parsed and passed together with the positional \n\
    arguments to the widget creation method. If one argument is given by its keyword, \n\
    no further positional arguments can follow. For this, the mandatory and optional \n\
    parameters of the widget creation method can be considered to be in one list, where \n\
    the optional parameters follow after the mandatory ones. \n\
\n\
Returns \n\
------- \n\
ui \n\
    :class:`ui` object, that represents the loaded widget, dialog or window. The type of \n\
    the ui is mainly defined by the type of the widget. If it is derived from `QMainWindow`, \n\
    a window is opened; if it is derived from `QDockWidget` a dock widget is created, in \n\
    all other cases a dialog is created. \n\
\n\
Notes \n\
----- \n\
Unlike it is the case at the creation of ui's from ui files, you cannot directly \n\
parameterize behaviours like the ``deleteOnClose`` flag. This can however be done using \n\
:meth:`setAttribute`. \n\
\n\
See Also \n\
-------- \n\
createNewPluginWidget2, widgetHelp");
PyObject* PythonUi::PyUi_createNewAlgoWidget(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    int length = PyTuple_Size(args);

    if (length == 0)
    {
        PyErr_SetString(PyExc_ValueError, QObject::tr("no widget name specified").toUtf8().data());
        return NULL;
    }

    PyErr_Clear();
    QVector<ito::ParamBase> paramsMandBase, paramsOptBase;
    ito::RetVal retVal = ito::retOk;
    PyObject *pnameObj = NULL;
    PyObject *params = NULL;
    QString algoWidgetName;
    bool ok = true;

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("no addin-manager found").toUtf8().data());
        return NULL;
    }

    pnameObj = PyTuple_GetItem(args, 0);
    algoWidgetName = PythonQtConversion::PyObjGetString(pnameObj, true, ok);

    if(!ok)
    {
        PyErr_SetString(PyExc_TypeError, QObject::tr("the first parameter must contain the widget name as string").toUtf8().data());
        return NULL;
    }

    const ito::AddInAlgo::AlgoWidgetDef *def = AIM->getAlgoWidgetDef( algoWidgetName );

    if(def == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("Could not find plugin widget with name '%1'").arg(algoWidgetName).toUtf8().data());
        return NULL;
    }

    const ito::FilterParams *filterParams = AIM->getHashedFilterParams(def->m_paramFunc);

    if(!filterParams)
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("Could not get parameters for plugin widget '%1'").arg(algoWidgetName).toUtf8().data());
        return NULL;
    }

    params = PyTuple_GetSlice(args, 1, PyTuple_Size(args)); //new reference

    if(parseInitParams(&(filterParams->paramsMand), &(filterParams->paramsOpt), params, kwds, paramsMandBase, paramsOptBase) != ito::retOk)
    {
        Py_XDECREF(params);
        PyErr_SetString(PyExc_RuntimeError, "error while parsing parameters.");
        return NULL;
    }
    Py_XDECREF(params);

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    int winType = 0xff;
    bool deleteOnClose = false;
    bool childOfMainWindow = true;
    Qt::DockWidgetArea dockWidgetArea = Qt::TopDockWidgetArea;
    int buttonBarType = UserUiDialog::bbTypeNo;
    StringMap dialogButtons;
    int uiDescription = UiOrganizer::createUiDescription(winType, buttonBarType, childOfMainWindow, deleteOnClose, dockWidgetArea);

    QSharedPointer<unsigned int> dialogHandle(new unsigned int);
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QSharedPointer<QByteArray> className(new QByteArray());
    *dialogHandle = 0;
    *objectID = 0;

    QMetaObject::invokeMethod(
        uiOrga,
        "loadPluginWidget",
        Q_ARG(void*, reinterpret_cast<void*>(def->m_widgetFunc)),
        Q_ARG(int, uiDescription),
        Q_ARG(StringMap, dialogButtons),
        Q_ARG(QVector<ito::ParamBase>*, &paramsMandBase),
        Q_ARG(QVector<ito::ParamBase>*, &paramsOptBase),
        Q_ARG(QSharedPointer<uint>, dialogHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(QSharedPointer<QByteArray>, className),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while loading plugin widget");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    PythonUi::PyUi *dialog;

    PyObject *emptyTuple = PyTuple_New(0);
    dialog = (PyUi*)PyObject_Call((PyObject*)&PyUiType, NULL, NULL); //new ref, tp_new of PyUi is called, init not
    Py_XDECREF(emptyTuple);

    if(dialog == NULL)
    {
        if(*dialogHandle)
        {
            ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(
                uiOrga,
                "deleteDialog",
                Q_ARG(uint, static_cast<unsigned int>(*dialogHandle)), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore()));

            if(!locker2.getSemaphore()->wait(PLUGINWAIT))
            {
                PyErr_SetString(PyExc_RuntimeError, "timeout while closing dialog");
            }
        }

        PyErr_SetString(PyExc_RuntimeError, "could not create a new instance of class ui.");
        return NULL;
    }

    dialog->uiHandle = static_cast<int>(*dialogHandle);
    dialog->signalMapper = new PythonQtSignalMapper();
    dialog->uiItem.methodList = NULL;
    dialog->uiItem.objectID = *objectID;
    dialog->uiItem.widgetClassName = new char[className->size()+1];
    strcpy_s(dialog->uiItem.widgetClassName, className->size()+1, className->data());

    const char *objName = "<plugin-widget>\0";
    dialog->uiItem.objName = new char[strlen(objName)+1];
    strcpy_s(dialog->uiItem.objName, strlen(objName)+1, objName);

    return (PyObject*)dialog;
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiCreateNewPluginWidget2_doc,
"createNewPluginWidget2(widgetName, paramsArgs = [], paramsDict = {}, type = 0xFF, dialogButtonBar = ui.BUTTONBAR_NO, dialogButtons = {}, childOfMainWindow = True, deleteOnClose = False, dockWidgetArea = ui.TOPDOCKWIDGETAREA) -> ui \n\
\n\
Loads a widget, defined in an itom algorithm plugin, and returns the :class:`ui` object, that references this widget. \n\
\n\
Itom algorithm plugins cannot only contain algorithms, callable by Python, but also \n\
methods, that return a customized user-interface, widget etc. Use this method to \n\
initialize such an user-interface and returns its corresponding :class:`ui` object. \n\
\n\
For a list of available widget methods, see :meth:`widgetHelp`. \n\
\n\
Parameters \n\
---------- \n\
widgetName : str \n\
    Name of algorithm widget method. \n\
paramsArgs : tuple \n\
    See ``paramsDict``. \n\
paramsDict : dict \n\
    The widget creation method in the algorithm plugin can depend on several mandatory \n\
    and / or optional parameters. For their initialization, the mandatory and optional \n\
    parameters are considered to be stacked together. At first, the ``paramsArgs`` \n\
    sequence is used to assign a certain number of parameters beginning with the \n\
    mandatory ones. If all ``paramsArgs`` values are assigned, the keyword-based \n\
    values in ``paramsDict`` are tried to be assigned to not yet used mandatory or \n\
    optional parameters. All mandatory parameters must be given (see \n\
    ``widgetHelp(widgetName)`` to obtain information about all required parameters. \n\
type : int, optional \n\
    Desired type of the newly created widget (a widget can also be a standalone dialog, \n\
    dockwidget or window): \n\
    \n\
    * 255 (default) : the type is derived from the original type of the widget, \n\
    * 0 (``ui.TYPEDIALOG``): the ui-file is embedded in auto-created dialog, \n\
    * 1 (``ui.TYPEWINDOW``): the ui-file is handled as main window, \n\
    * 2 (``ui.TYPEDOCKWIDGET``): the ui-file is handled as dock-widget and appended \n\
        to the main-window dock area, \n\
    * 3 (``ui.TYPECENTRALWIDGET``): the ui-file must be a widget or main window \n\
        and is included in the central area of itom, above the command line. \n\
    \n\
dialogButtonBar : int, optional \n\
    Only for ``type`` ``ui.TYPEDIALOG (0)``: Indicates if buttons should be automatically \n\
    added to the dialog: \n\
    \n\
    * 0 (``ui.BUTTONBAR_NO``): do not add any buttons (default), \n\
    * 1 (``ui.BUTTONBAR_HORIZONTAL``): add a horizontal button bar, \n\
    * 2 (``ui.BUTTONBAR_VERTICAL``): add a vertical button bar. \n\
    \n\
dialogButtons : dict, optional \n\
    Only relevant if ``dialogButtonBar`` is not ``ui.BUTTONBAR_NO``: This dictionary \n\
    contains all buttons, that should be added to the button bar. For every entry, \n\
    the key is the role name of the button (enum ``QDialogButtonBox::ButtonRole``, \n\
    e.g. 'AcceptRole', 'RejectRole', 'ApplyRole', 'YesRole', 'NoRole'). The value is \n\
    the text of the button. \n\
childOfMainWindow : bool, optional \n\
    For type ``ui.TYPEDIALOG`` and ``ui.TYPEWINDOW`` only: Indicates if the window \n\
    should be a child of the itom main window. If ``False``, this window has its own \n\
    icon in the taskbar of the operating system. \n\
deleteOnClose : bool, optional \n\
    Indicates if the widget / window / dialog should be deleted if the user closes it \n\
    or if it is hidden. If it is hidden, it can be shown again using :meth:`show`. \n\
dockWidgetArea : int, optional \n\
    Only for ``type`` ``ui.TYPEDOCKWIDGET (2)``. Indicates the position where the \n\
    dock widget should be placed: \n\
    \n\
    * 1 : ``ui.LEFTDOCKWIDGETAREA`` \n\
    * 2 : ``ui.RIGHTDOCKWIDGETAREA`` \n\
    * 4 : ``ui.TOPDOCKWIDGETAREA`` \n\
    * 8 : ``ui.BOTTOMDOCKWIDGETAREA`` \n\
\n\
Returns \n\
------- \n\
ui \n\
    :class:`ui` object, that represents the loaded widget, dialog or window. The type of \n\
    the ui is mainly defined by the type of the widget. If it is derived from `QMainWindow`, \n\
    a window is opened; if it is derived from `QDockWidget` a dock widget is created, in \n\
    all other cases a dialog is created. \n\
\n\
Notes \n\
----- \n\
Unlike it is the case at the creation of ui's from ui files, you cannot directly \n\
parameterize behaviours like the ``deleteOnClose`` flag. This can however be done using \n\
:meth:`setAttribute`. \n\
\n\
See Also \n\
-------- \n\
createNewPluginWidget, widgetHelp");
PyObject* PythonUi::PyUi_createNewAlgoWidget2(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = { "widgetName", "paramsArgs", "paramsDict", "type", "dialogButtonBar", "dialogButtons", "childOfMainWindow", "deleteOnClose", "dockWidgetArea", NULL };
    PyObject *dialogButtons = NULL;
    int dockWidgetArea = Qt::TopDockWidgetArea;
    const char* widgetName = NULL;
    PyObject *paramsArgs = NULL;
    PyObject *paramsDict = NULL;
    bool deleteOnClose = false;
    bool childOfMainWindow = true;
    int winType = 0xFF;
    int buttonBarType = UserUiDialog::bbTypeNo;

    if (args == NULL || PyTuple_Size(args) == 0) //empty constructor
    {
        return 0;
    }

    if (!PyArg_ParseTupleAndKeywords(
        args,
        kwds,
        "s|O!O!iiO!bbi",
        const_cast<char**>(kwlist),
        &widgetName,
        &PyTuple_Type, &paramsArgs,
        &PyDict_Type, &paramsDict,
        &winType,
        &buttonBarType,
        &PyDict_Type, &dialogButtons,
        &childOfMainWindow,
        &deleteOnClose,
        &dockWidgetArea))
    {
        return NULL;
    }

    QVector<ito::ParamBase> paramsMandBase, paramsOptBase;
    QString algoWidgetName = widgetName;

    ito::AddInManager *AIM = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    if (!AIM)
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("no addin-manager found").toUtf8().data());
        return NULL;
    }

    const ito::AddInAlgo::AlgoWidgetDef *def = AIM->getAlgoWidgetDef(algoWidgetName);

    if (def == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("Could not find plugin widget with name '%1'").arg(algoWidgetName).toUtf8().data());
        return NULL;
    }

    const ito::FilterParams *filterParams = AIM->getHashedFilterParams(def->m_paramFunc);

    if (!filterParams)
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("Could not get parameters for plugin widget '%1'").arg(algoWidgetName).toUtf8().data());
        return NULL;
    }

    if (parseInitParams(&(filterParams->paramsMand), &(filterParams->paramsOpt), paramsArgs, paramsDict, paramsMandBase, paramsOptBase) != ito::retOk)
    {
        PyErr_SetString(PyExc_RuntimeError, "error while parsing parameters.");
        return NULL;
    }

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    int uiDescription = UiOrganizer::createUiDescription(winType, buttonBarType, childOfMainWindow, deleteOnClose, dockWidgetArea);
    StringMap dialogButtonMap;

    if (dialogButtons)
    {
        //transfer dialogButtons dict to dialogButtonMap
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        QString keyString, valueString;
        bool ok = false;

        while (PyDict_Next(dialogButtons, &pos, &key, &value))
        {
            keyString = PythonQtConversion::PyObjGetString(key, true, ok);
            valueString = PythonQtConversion::PyObjGetString(value, true, ok);

            if (keyString.isNull() || valueString.isNull())
            {
                std::cout << "Warning while parsing dialogButtons-dictionary. At least one element does not contain a string as key and value\n" << std::endl;
            }
            else
            {
                dialogButtonMap[keyString] = valueString;
            }
        }
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;
    QSharedPointer<unsigned int> dialogHandle(new unsigned int);
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QSharedPointer<QByteArray> className(new QByteArray());
    *dialogHandle = 0;
    *objectID = 0;

    QMetaObject::invokeMethod(
        uiOrga,
        "loadPluginWidget",
        Q_ARG(void*, reinterpret_cast<void*>(def->m_widgetFunc)),
        Q_ARG(int, uiDescription),
        Q_ARG(StringMap, dialogButtonMap),
        Q_ARG(QVector<ito::ParamBase>*, &paramsMandBase),
        Q_ARG(QVector<ito::ParamBase>*, &paramsOptBase),
        Q_ARG(QSharedPointer<uint>, dialogHandle),
        Q_ARG(QSharedPointer<uint>, objectID),
        Q_ARG(QSharedPointer<QByteArray>, className),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while loading plugin widget");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;

    if (!PythonCommon::transformRetValToPyException(retValue))
    {
        return NULL;
    }

    PythonUi::PyUi *dialog;

    PyObject *emptyTuple = PyTuple_New(0);
    dialog = (PyUi*)PyObject_Call((PyObject*)&PyUiType, NULL, NULL); //new ref, tp_new of PyUi is called, init not
    Py_XDECREF(emptyTuple);

    if (dialog == NULL)
    {
        if (*dialogHandle)
        {
            ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(
                uiOrga,
                "deleteDialog",
                Q_ARG(uint, static_cast<unsigned int>(*dialogHandle)), // 'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
                Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore()));

            if (!locker2.getSemaphore()->wait(PLUGINWAIT))
            {
                PyErr_SetString(PyExc_RuntimeError, "timeout while closing dialog");
            }
        }

        PyErr_SetString(PyExc_RuntimeError, "could not create a new instance of class ui.");
        return NULL;
    }

    dialog->uiHandle = static_cast<int>(*dialogHandle);
    dialog->signalMapper = new PythonQtSignalMapper();
    dialog->uiItem.methodList = NULL;
    dialog->uiItem.objectID = *objectID;
    dialog->uiItem.widgetClassName = new char[className->size() + 1];
    strcpy_s(dialog->uiItem.widgetClassName, className->size() + 1, className->data());

    const char *objName = "<plugin-widget>\0";
    dialog->uiItem.objName = new char[strlen(objName) + 1];
    strcpy_s(dialog->uiItem.objName, strlen(objName) + 1, objName);

    return (PyObject*)dialog;
}


//-------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiAvailableWidgets_doc, "availableWidgets() -> List[str] \n\
\n\
List of class names of all available widgets that can be directly loaded in an ui-file at runtime. \n\
\n\
Returns \n\
------- \n\
list of str \n\
    A list of the class names of all widgets, that can be directly loaded in an \n\
    user interface at runtime. These widgets can be built-in widgets of Qt as well \n\
    as additional widgets from designer plugins (like itom plots or other itom widgets.");
PyObject* PythonUi::PyUi_availableWidgets(PyUi * /*self*/)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());

    if (uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<QStringList> widgetNames(new QStringList);
    QMetaObject::invokeMethod(
        uiOrga,
        "getAvailableWidgetNames",
        Q_ARG(QSharedPointer<QStringList>, widgetNames),
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(5000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while request");
        return NULL;
    }

    return PythonQtConversion::QStringListToPyList(*widgetNames);
}


//-------------------------------------------------------------------------------------
PyMethodDef PythonUi::PyUi_methods[] = {
        {"show", (PyCFunction)PyUi_show,     METH_VARARGS, pyUiShow_doc},
        {"hide", (PyCFunction)PyUi_hide, METH_NOARGS, pyUiHide_doc},
        {"isVisible", (PyCFunction)PyUi_isVisible, METH_NOARGS, pyUiIsVisible_doc},

        {"getDouble",(PyCFunction)PyUi_getDouble, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetDouble_doc},
        {"getInt",(PyCFunction)PyUi_getInt, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetInt_doc},
        {"getItem",(PyCFunction)PyUi_getItem, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetItem_doc},
        {"getText",(PyCFunction)PyUi_getText, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetText_doc},
        {"msgInformation", (PyCFunction)PyUi_msgInformation, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgInformation_doc},
        {"msgQuestion", (PyCFunction)PyUi_msgQuestion, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgQuestion_doc},
        {"msgWarning", (PyCFunction)PyUi_msgWarning, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgWarning_doc},
        {"msgCritical", (PyCFunction)PyUi_msgCritical, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiMsgCritical_doc},
        {"getExistingDirectory", (PyCFunction)PyUi_getExistingDirectory, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetExistingDirectory_doc},
        {"getOpenFileName", (PyCFunction)PyUi_getOpenFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetOpenFileName_doc},
        {"getOpenFileNames", (PyCFunction)PyUi_getOpenFileNames, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiGetOpenFileNames_doc },
        {"getSaveFileName", (PyCFunction)PyUi_getSaveFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetSaveFileName_doc},
        {"createNewPluginWidget", (PyCFunction)PyUi_createNewAlgoWidget, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiCreateNewPluginWidget_doc},
        { "createNewPluginWidget2", (PyCFunction)PyUi_createNewAlgoWidget2, METH_KEYWORDS | METH_VARARGS | METH_STATIC, pyUiCreateNewPluginWidget2_doc },
        { "availableWidgets", (PyCFunction)PyUi_availableWidgets, METH_NOARGS | METH_STATIC, pyUiAvailableWidgets_doc },
        {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyMemberDef PythonUi::PyUi_members[] = {
        {NULL}  /* Sentinel */
};

//-------------------------------------------------------------------------------------
PyModuleDef PythonUi::PyUiModule = {
        PyModuleDef_HEAD_INIT,
        "ui",
        "Itom userInterfaceDialog type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//-------------------------------------------------------------------------------------
PyGetSetDef PythonUi::PyUi_getseters[] = {
    {NULL}  /* Sentinel */
};

PyTypeObject PythonUi::PyUiType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "itom.ui",             /* tp_name */
        sizeof(PyUi),             /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)PyUi_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        (reprfunc)PyUi_repr,         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0, /* tp_getattro */
        0,  /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
        pyUiInit_doc /*"dataObject objects"*/,           /* tp_doc */
        0,                       /* tp_traverse */
        0,                       /* tp_clear */
        0,            /* tp_richcompare */
        0,                       /* tp_weaklistoffset */
        0,                       /* tp_iter */
        0,                       /* tp_iternext */
        PyUi_methods,             /* tp_methods */
        PyUi_members,             /* tp_members */
        PyUi_getseters,            /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)PyUi_init,      /* tp_init */
        0,                         /* tp_alloc */
        PyUi_new /*PyType_GenericNew*/ /*PythonStream_new,*/                 /* tp_new */
};

//-------------------------------------------------------------------------------------
void PythonUi::PyUi_addTpDict(PyObject *tp_dict)
{
    PyObject *value;
    QMetaObject metaObject = QMessageBox::staticMetaObject;
    QMetaEnum metaEnum = metaObject.enumerator( metaObject.indexOfEnumerator( "StandardButtons" ));
    QString key;
    QStringList obsoleteKeys = QStringList() << "YesAll" << "NoAll" << "Default" << "Escape" << "FlagMask" << "ButtonMask";

    //auto-parsing of StandardButtons-enumeration for key-value-pairs
    for(int i = 0 ; i < metaEnum.keyCount() ; i++)
    {
        value = Py_BuildValue("i", metaEnum.value(i));
        key = metaEnum.key(i);

        if(obsoleteKeys.contains(key) == false)
        {
            key.prepend("MsgBox"); //Button-Constants will be accessed by ui.MsgBoxOk, ui.MsgBoxError...
            PyDict_SetItemString(tp_dict, key.toLatin1().data(), value);
            Py_DECREF(value);
        }
    }

    //add dialog types
    value = Py_BuildValue("i", 0);
    PyDict_SetItemString(tp_dict, "TYPEDIALOG", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 1);
    PyDict_SetItemString(tp_dict, "TYPEWINDOW", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 2);
    PyDict_SetItemString(tp_dict, "TYPEDOCKWIDGET", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 3);
    PyDict_SetItemString(tp_dict, "TYPECENTRALWIDGET", value);
    Py_DECREF(value);

    //add button orientation
    value = Py_BuildValue("i", 0);
    PyDict_SetItemString(tp_dict, "BUTTONBAR_NO", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 1);
    PyDict_SetItemString(tp_dict, "BUTTONBAR_HORIZONTAL", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", 2);
    PyDict_SetItemString(tp_dict, "BUTTONBAR_VERTICAL", value);
    Py_DECREF(value);

    //add dock widget area
    value = Py_BuildValue("i", Qt::LeftDockWidgetArea);
    PyDict_SetItemString(tp_dict, "LEFTDOCKWIDGETAREA", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", Qt::RightDockWidgetArea);
    PyDict_SetItemString(tp_dict, "RIGHTDOCKWIDGETAREA", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", Qt::TopDockWidgetArea);
    PyDict_SetItemString(tp_dict, "TOPDOCKWIDGETAREA", value);
    Py_DECREF(value);
    value = Py_BuildValue("i", Qt::BottomDockWidgetArea);
    PyDict_SetItemString(tp_dict, "BOTTOMDOCKWIDGETAREA", value);
    Py_DECREF(value);
}


} //end namespace ito
