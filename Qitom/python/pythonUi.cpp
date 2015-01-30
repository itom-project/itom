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

#include "pythonUi.h"

#include "structmember.h"

#include "../global.h"
#include "../organizer/uiOrganizer.h"
#include "../organizer/addInManager.h"
#include <qcoreapplication.h>

#include "pythonQtConversion.h"
#include "pythonFigure.h"
#include "AppManagement.h"

#include <qsharedpointer.h>
#include <qmessagebox.h>
#include <qmetaobject.h>

QHash<QByteArray, QSharedPointer<MethodDescriptionList> > ito::PythonUi::methodDescriptionListStorage;

namespace ito
{
// -------------------------------------------------------------------------------------------------------------------------
//
//  PyUiItem
//
// -------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemInit_doc,"uiItem(...) -> base class representing any widget of a graphical user interface \n\
\n\
This class represents any widget (graphical, interactive element like a button or checkbox) on a graphical user interface. \n\
An instance of this class provides many functionalities given by the underlying Qt system. For instance, it is posible to \n\
call a public slot of the corresponding widget, connect signals to specific python methods or functions or change properties \n\
of the widget represeted by the instance. \n\
\n\
The overall dialog or window as main element of a graphical user interface itself are instances of the class *ui*. However, \n\
they are derived from *uiItem*, since dialogs or windows internally are widgets as well. \n\
\n\
Widgets placed at a user interface using the Qt Designer can be referenced by an *uiItem* instance by their specific objectName, \n\
assigned in the Qt Designer as well. As an example, a simple dialog with one button is created and the text of the button (objectName: btn) \n\
is set to OK: :: \n\
    \n\
    dialog = ui('filename.ui', type=ui.TYPEDIALOG) \n\
    button = dialog.btn #here the reference to the button is obtained \n\
    button[\"text\"] = \"OK\" #set the property text of the button \n\
    \n\
Information about available properties, signals and slots can be obtained using the method **info()** of *uiItem*. \n\
\n\
Notes \n\
------ \n\
It is not intended to directly instantiate this class. Either create a user interface using the class *ui* or obtain \n\
a reference to an existing widget (this is then an instance of *uiItem*) using the dot-operator of a \n\
parent widget or the entire user interface.");
int PythonUi::PyUiItem_init(PyUiItem *self, PyObject *args, PyObject * /*kwds*/)
{
    ito::RetVal retValue = retOk;
    QSharedPointer<unsigned int> objectID(new unsigned int());
    *objectID = 0;
    QSharedPointer<QByteArray> widgetClassNameBA(new QByteArray());
    const char *objName = NULL;
    const char *widgetClassName = NULL;
    PyObject *parentObj = NULL;
    PythonUi::PyUiItem *parentItem = NULL;

    if(PyArg_ParseTuple(args,"Iss|O!",&(*objectID),&objName,&widgetClassName,&PythonUi::PyUiItemType,&parentObj))
    {
        self->baseItem = parentObj;
        Py_XINCREF(self->baseItem); //if parent available increment its reference
        DELETE_AND_SET_NULL_ARRAY(self->objName);
        self->objName = new char[strlen(objName)+1];
        strcpy_s(self->objName, strlen(objName)+1, objName);
        DELETE_AND_SET_NULL_ARRAY(self->widgetClassName);
        self->widgetClassName = new char[strlen(widgetClassName)+1];
        strcpy_s(self->widgetClassName, strlen(widgetClassName)+1, widgetClassName);
        self->objectID = *objectID;
        *widgetClassNameBA = widgetClassName;
    }
    else if(PyErr_Clear(), PyArg_ParseTuple(args, "O!s", &PythonUi::PyUiItemType, &parentObj, &objName))
    {
        UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
        if(uiOrga == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
            return -1;
        }

        parentItem = (PythonUi::PyUiItem*)parentObj;
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(uiOrga, "getChildObject3", Q_ARG(uint, static_cast<unsigned int>(parentItem->objectID)), Q_ARG(QString, QString(objName)), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(QSharedPointer<QByteArray>, widgetClassNameBA), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command

        locker.getSemaphore()->wait(-1);
        retValue += locker.getSemaphore()->returnValue;

        if(*objectID == 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "attribute is no widget name of this user interface");
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

//----------------------------------------------------------------------------------------------------------------------------------
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

    QMetaObject::invokeMethod(uiOrga, "widgetMetaObjectCounts", Q_ARG(uint, static_cast<unsigned int>(self->objectID)), Q_ARG(QSharedPointer<int>, classInfoCount), Q_ARG(QSharedPointer<int>, enumeratorCount), Q_ARG(QSharedPointer<int>, methodCount),Q_ARG(QSharedPointer<int>, propertiesCount), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting number of properties");
        return 0;
    }

    retValue += locker.getSemaphore()->returnValue;
    return *propertiesCount; //nr of properties in the corresponding QMetaObject
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUi::PyUiItem_mappingGetElem(PyUiItem* self, PyObject* key)
{
    QStringList propNames;
    bool ok = false;
    QString propName = PythonQtConversion::PyObjGetString(key,false,ok);
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

    QMetaObject::invokeMethod(uiOrga, "readProperties", Q_ARG(uint, self->objectID), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    return PythonQtConversion::QVariantToPyObject(retPropMap->value(propNames[0]));
}

//----------------------------------------------------------------------------------------------------------------------------------
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

    QMetaObject::invokeMethod(uiOrga, "writeProperties", Q_ARG(uint, self->objectID), Q_ARG(QVariantMap, propMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while writing property");
        return -1;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return -1;

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemCall_doc,"call(slotOrPublicMethod [,argument1, argument2, ...]) -> calls any public slot of this widget or any accessible public method.  \n\
\n\
This method invokes (calls) a method of the underlying widget that is marked as public slot. Besides slots there are some public methods of specific \n\
widget classes that are wrapped by itom and therefore are callable by this method, too. \n\
\n\
If only method is available, all arguments are tried to be cast to the requested types and the slot is called on conversion success. If the method has \n\
multiple overloaded possibilities in the underlying C++ classes, at first, it is intended to find the variant where all arguments can be strictly casted \n\
from Python types to the necessary C-types. If this fails, the next variant with a non-strict conversion is chosen. \n\
\n\
Parameters \n\
----------- \n\
slotOrPublicMethod : {str} \n\
    name of the slot or method \n\
arguments : {various types}, optional\n\
    Here you must indicate every argument, that the definition of the slot indicates. The type must be convertable into the \n\
    requested C++ based argument type.\n\
\n\
Notes \n\
----- \n\
If you want to know all possible slots of a specific widget, see the Qt help or call the member *info()* of the widget. \n\
\n\
See Also \n\
--------- \n\
info()");
PyObject* PythonUi::PyUiItem_call(PyUiItem *self, PyObject* args)
{
    int argsSize = PyTuple_Size(args);
    int nrOfParams = argsSize - 1;
    bool ok;
    FctCallParamContainer *paramContainer;

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

    if(!loadMethodDescriptionList(self)) return NULL;

    //scan for method
    //step 1: check if method exists
    QList<const MethodDescription*> possibleMethods;
    for(int i=0;i<self->methodList->size();i++)
    {
        if(self->methodList->at(i).name() == slotName)
        {
            possibleMethods.append( &(self->methodList->at(i)) );
        }
    }
    
    if(possibleMethods.size() == 0)
    {
        PyErr_Format(PyExc_RuntimeError, "No slot or method with name '%s' available.", slotName.data());
        return NULL;
    }

    //create function container
    paramContainer = new FctCallParamContainer(nrOfParams);
    void *ptr = NULL;
    int typeNr = 0;
    bool found = false;
    QByteArray possibleSignatures = "";
    const MethodDescription *foundMethod = NULL;

    if (possibleMethods.count() > 1) //if more than one possible method is availabe, at first, try to strictly cast all parameters...
    {
        foreach(const MethodDescription *method, possibleMethods)
        {
            ok = true;
            if(method->checkMethod(slotName, nrOfParams))
            {
                for(int j=0;j<nrOfParams;j++)
                {
                    //first try to find strict conversions only (in order to better handle methods with different possible argument types
                    if(PythonQtConversion::PyObjToVoidPtr(PyTuple_GetItem(args,j+1), &ptr, &typeNr, method->argTypes()[j], true)) //GetItem is a borrowed reference
                    {
                        paramContainer->setParamArg(j, ptr, typeNr);
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }

                if(ok)
                {
                    paramContainer->initRetArg( method->retType() ); //init retArg after all other parameters fit to requirements

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
    else //... directly allow the non-strict conversion of all parameters (ok = false enters the next if case ;) )
    {
        ok = false;
    }

    if (!ok) //until now, there is no possibility to directly, strictly cast all parameters to available signatures. Therefore try now also to not-strictly cast
    {
        foreach(const MethodDescription *method, possibleMethods)
        {
            ok = true;
            if(method->checkMethod(slotName, nrOfParams))
            {
                ok = true;
                for(int j=0;j<nrOfParams;j++)
                {
                    //first try to find strict conversions only (in order to better handle methods with different possible argument types
                    if(PythonQtConversion::PyObjToVoidPtr(PyTuple_GetItem(args,j+1), &ptr, &typeNr, method->argTypes()[j], false)) //GetItem is a borrowed reference
                    {
                        paramContainer->setParamArg(j, ptr, typeNr);
                    }
                    else
                    {
                        ok = false;
                        break;
                    }
                }

                if(ok)
                {
                    paramContainer->initRetArg( method->retType() ); //init retArg after all other parameters fit to requirements

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

    if(!found)
    {
        DELETE_AND_SET_NULL(paramContainer);
        PyErr_Format(PyExc_RuntimeError, "None of the following possible signatures fit to the given set of parameters: %s", possibleSignatures.data());
        return NULL;
    }

    QSharedPointer<FctCallParamContainer> sharedParamContainer(paramContainer); //from now on, do not directly delete paramContainer any more
    ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());

    if(foundMethod->type() == QMetaMethod::Slot)
    {
        QMetaObject::invokeMethod(uiOrga, "callSlotOrMethod", Q_ARG(bool,true), Q_ARG(uint, self->objectID), Q_ARG(int, foundMethod->methodIndex()), Q_ARG(QSharedPointer<FctCallParamContainer>, sharedParamContainer), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    }   
    else if(foundMethod->type() == QMetaMethod::Method)
    {
        QMetaObject::invokeMethod(uiOrga, "callSlotOrMethod", Q_ARG(bool,false), Q_ARG(uint, self->objectID), Q_ARG(int, foundMethod->methodIndex()), Q_ARG(QSharedPointer<FctCallParamContainer>, sharedParamContainer), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    }
    else
    {
        PyErr_SetString(PyExc_RuntimeError, "unknown method type.");
        return NULL;
    }

    if(!locker2.getSemaphore()->wait(50000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while calling slot");
        return NULL;
    }

    if(PythonCommon::transformRetValToPyException( locker2.getSemaphore()->returnValue ) == false) return NULL;

    if(sharedParamContainer->getRetType() > 0)
    {
        if(sharedParamContainer->getRetType() == QMetaType::type("ito::PythonQObjectMarshal"))
        {
            ito::PythonQObjectMarshal *m = (ito::PythonQObjectMarshal*)sharedParamContainer->args()[0];

            PyObject *newArgs = PyTuple_New(4);
            PyTuple_SetItem(newArgs,0, PyLong_FromLong(m->m_objectID));
            PyTuple_SetItem(newArgs,1, PyUnicode_FromString( m->m_objName.data() ));
            PyTuple_SetItem(newArgs,2, PyUnicode_FromString( m->m_className.data() ));
            Py_INCREF(self);
            PyTuple_SetItem(newArgs,3, (PyObject*)self);
            PyObject *newUiItem = PyObject_CallObject((PyObject *) &PythonUi::PyUiItemType, newArgs);
            Py_DECREF(newArgs);
            return newUiItem;
        }
        else
        {
            return PythonQtConversion::ConvertQtValueToPythonInternal(sharedParamContainer->getRetType(), sharedParamContainer->args()[0]);
        }
    }

    Py_RETURN_NONE;

}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemConnect_doc,"connect(signalSignature, callableMethod) -> connects the signal of the widget with the given callable python method \n\
\n\
This instance of *uiItem* wraps a widget, that is defined by a C++-class, that is finally derived from *QWidget*. See Qt-help for more information \n\
about the capabilities of every specific widget. Every widget can send various signals. Use this method to connect any signal to any \n\
callable python method (bounded or unbounded). This method must have the same number of arguments than the signal and the types of the \n\
signal definition must be convertable into a python object. \n\
\n\
Parameters \n\
----------- \n\
signalSignature : {str} \n\
    This must be the valid signature, known from the Qt-method *connect* (e.g. 'clicked(bool)') \n\
callableMethod : {python method or function} \n\
    valid method or function that is called if the signal is emitted. \n\
\n\
See Also \n\
--------- \n\
disconnect, invokeKeyboardInterrupt");
PyObject* PythonUi::PyUiItem_connect(PyUiItem *self, PyObject* args)
{
    const char* signalSignature;
    PyObject *callableMethod;

    if(!PyArg_ParseTuple(args, "sO", &signalSignature, &callableMethod))
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

    QString signature(signalSignature);
    QSharedPointer<int> sigId(new int);
    
    QSharedPointer<QObject*> objPtr(new (QObject*));
    QSharedPointer<IntList> argTypes(new IntList);

    *sigId = -1;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "getSignalIndex", Q_ARG(uint, self->objectID), Q_ARG(QString, signature), Q_ARG(QSharedPointer<int>, sigId), Q_ARG(QSharedPointer<QObject*>, objPtr), Q_ARG(QSharedPointer<IntList>, argTypes), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
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
        if(!signalMapper->addSignalHandler(*objPtr, signalSignature, *sigId, callableMethod, *argTypes))
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemConnectKeyboardInterrupt_doc,"invokeKeyboardInterrupt(signalSignature) -> connects the given signal with a slot immediately invoking a python interrupt signal. \n\
\n\
Parameters \n\
----------- \n\
signalSignature : {str} \n\
    This must be the valid signature, known from the Qt-method *connect* (e.g. 'clicked(bool)') \n\
\n\
Returns \n\
------- \n\
\n\
Notes \n\
----- \n\
If you use the connect method to link a signal with a python method or function, this method can only be executed if python is in an idle status. \n\
However, if you want raise the python interrupt signal if a specific signal is emitted, this interruption should be immediately invoked. Therefore \n\
\n\
See Also \n\
--------- \n\
connect");
PyObject* PythonUi::PyUiItem_connectKeyboardInterrupt(PyUiItem *self, PyObject* args)
{
    const char* signalSignature;

    if(!PyArg_ParseTuple(args, "s", &signalSignature))
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

    QString signature(signalSignature);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "connectWithKeyboardInterrupt", Q_ARG(uint, self->objectID), Q_ARG(QString, signature), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while analysing signal signature");
        return NULL;
    }

    if(!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemDisconnect_doc,"disconnect(signalSignature, callableMethod) -> disconnects a connection which must have been established with exactly the same parameters.\n\
\n\
Parameters \n\
----------- \n\
signalSignature : {str} \n\
callableMethod : {python method or function} \n\
\n\
Returns \n\
------- \n\
\n\
Notes \n\
----- \n\
doctodo\n\
\n\
See Also \n\
--------- \n\
\n\
");
PyObject* PythonUi::PyUiItem_disconnect(PyUiItem *self, PyObject* args)
{
    const char* signalSignature;
    PyObject *callableMethod;

    if(!PyArg_ParseTuple(args, "sO", &signalSignature, &callableMethod))
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

    QString signature(signalSignature);
    QSharedPointer<int> sigId(new int);
    QSharedPointer<QObject*> objPtr(new (QObject*));
    QSharedPointer<IntList> argTypes(new IntList);

    *sigId = -1;

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QMetaObject::invokeMethod(uiOrga, "getSignalIndex", Q_ARG(uint, self->objectID), Q_ARG(QString, signature), Q_ARG(QSharedPointer<int>, sigId), Q_ARG(QSharedPointer<QObject*>, objPtr), Q_ARG(QSharedPointer<IntList>, argTypes), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
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
        if(signalMapper->removeSignalHandler(*objPtr, signalSignature, *sigId, callableMethod))
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetProperty_doc,"getProperty(propertyName | listOfPropertyNames) -> returns tuple of requested properties (single property or tuple of properties)\n\
Use this method or the operator [] in order to get the value of one specific property of this widget or of multiple properties. \n\
Multiple properties are given by a tuple or list of property names. For one single property, its value is returned as it is. \n\
If the property names are passed as sequence, a sequence of same size is returned with the corresponding values. \n\
\n\
Parameters \n\
----------- \n\
property : {string, string-list} \n\
    Name of one property or sequence (tuple,list...) of property names \n\
\n\
Returns \n\
------- \n\
out : {variant, sequence of variants} \n\
    the value of one single property of a list of values, if a sequence of names is given as parameter. \n\
\n\
See Also \n\
--------- \n\
setProperty \n\
");
PyObject* PythonUi::PyUiItem_getProperties(PyUiItem *self, PyObject *args)
{
    PyObject *propertyNames = NULL;
    QStringList propNames;
    bool ok = false;
    bool returnTuple = true;

    if(!PyArg_ParseTuple(args, "O", &propertyNames))
    {
        return NULL;
    }

    if(PyBytes_Check(propertyNames) || PyUnicode_Check(propertyNames))
    {
        QString temp = PythonQtConversion::PyObjGetString(propertyNames, true, ok);
        if(ok)
        {
            returnTuple = false;
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

    QMetaObject::invokeMethod(uiOrga, "readProperties", Q_ARG(unsigned int, self->objectID), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while reading property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(returnTuple)
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemSetProperty_doc,"setProperty(propertyDict) -> each property in the parameter dictionary is set to the dictionaries value.\n\
\n\
Parameters \n\
----------- \n\
propertyDict : {dict}\n\
    Dictionary with properties (keyword) and the values that should be set.\n\
\n\
See Also \n\
--------- \n\
getProperty \n\
");
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

    QMetaObject::invokeMethod(uiOrga, "writeProperties", Q_ARG(uint, static_cast<unsigned int>(self->objectID)), Q_ARG(QVariantMap, propMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while writing property/properties");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetPropertyInfo_doc,"getPropertyInfo([propertyName]) -> returns information about the property 'propertyName' of this widget or all properties, if no name indicated.\n\
\n\
Parameters \n\
----------- \n\
propertyName : {tuple}, optional \n\
\n\
Returns \n\
------- \n\
");
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
    QMetaObject::invokeMethod(uiOrga, "getPropertyInfos", Q_ARG(uint, self->objectID), Q_ARG(QSharedPointer<QVariantMap>, retPropMap), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting property information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;

    if(retValue.containsError())
    {

        if(propertyName) PythonCommon::setReturnValueMessage(retValue, propertyName, PythonCommon::getProperty);
        else PythonCommon::setReturnValueMessage(retValue, "???", PythonCommon::getProperty);
        return NULL;
    }
    else if(retValue.containsWarning())
    {
        std::cout << "Warning while getting property infos with message: " << QObject::tr(retValue.errorMessage()).toLatin1().data() << std::endl;
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

    Py_RETURN_NONE;
}


//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetAttribute_doc,"getAttribute(attributeNumber) -> returns specified attribute of corresponding widget.\n\
\n\
Widgets have specific attributes that influence their behaviour. These attributes are contained in the Qt-enumeration \n\
Qt::WidgetAttribute. Use this method to query the current status of one specific attributes. \n\
\n\
Important attributes are: \n\
\n\
* Qt::WA_DeleteOnClose (55) -> deletes the widget when it is closed, else it is only hidden [default] \n\
* Qt::WA_MouseTracking (2) -> indicates that the widget has mouse tracking enabled \n\
\n\
Parameters \n\
----------- \n\
attributeNumber : {int} \n\
    Number of the attribute of the widget to query (enum Qt::WidgetAttribute) \n\
\n\
Returns \n\
------- \n\
out : {bool} \n\
    True if attribute is set, else False \n\
\n\
See Also \n\
--------- \n\
setAttribute()\n\
");
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

    QMetaObject::invokeMethod(uiOrga, "getAttribute", Q_ARG(uint, self->objectID), Q_ARG(int, attributeNumber), Q_ARG(QSharedPointer<bool>, value), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting attribute");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(*value)
    {
        Py_INCREF(Py_True);
        return Py_True;
    }
    else
    {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemSetAttribute_doc,"setAttribute(attributeNumber, value) -> sets attribute of corresponding widget.\n\
\n\
Widgets have specific attributes that influence their behaviour. These attributes are contained in the Qt-enumeration \n\
Qt::WidgetAttribute. Use this method to enable/disable one specific attribute.\n\
\n\
Important attributes are: \n\
\n\
* Qt::WA_DeleteOnClose (55) -> deletes the widget when it is closed, else it is only hidden [default] \n\
* Qt::WA_MouseTracking (2) -> indicates that the widget has mouse tracking enabled \n\
\n\
Parameters \n\
----------- \n\
attributeNumber : {int} \n\
    Number of the attribute of the widget to set (enum Qt::WidgetAttribute) \n\
value : {bool} \n\
    True if attribute should be enabled, else False \n\
\n\
See Also \n\
--------- \n\
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

    QMetaObject::invokeMethod(uiOrga, "setAttribute", Q_ARG(uint, self->objectID), Q_ARG(int, attributeNumber), Q_ARG(bool, value), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while setting attribute");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}


//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemSetWindowFlags_doc,"setWindowFlags(flags) -> set window flags of corresponding widget.\n\
\n\
The window flags are used to set the type of a widget, dialog or window including further hints to the window system. \n\
This method is used to set the entire or-combination of all flags, contained in the Qt-enumeration Qt::WindowType. \n\
\n\
The most important types are: \n\
\n\
* Qt::Widget (0) -> default type for widgets \n\
* Qt::Window (1) -> the widget looks and behaves like a windows (title bar, window frame...) \n\
* Qt::Dialog (3) -> window decorated as dialog (no minimize or maximize button...) \n\
\n\
Further hints can be (among others): \n\
\n\
* Qt::FramelessWindowHint (0x00000800) -> borderless window (user cannot move or resize the window) \n\
* Qt::WindowTitleBar (0x00001000) -> gives the window a title bar \n\
* Qt::WindowMinimizeButtonHint (0x00004000) -> adds a minimize button to the title bar \n\
* Qt::WindowMaximizeButtonHint (0x00008000) -> adds a maximize button to the title bar \n\
* Qt::WindowCloseButtonHint (0x00010000) -> adds a close button. \n\
\n\
If you simply want to change one hint, get the current set of flags using **getWindowFlags**, change the necessary bitmask and \n\
set it again using this method. \n\
\n\
Parameters \n\
----------- \n\
flags : {int} \n\
    window flags to set (or-combination, see Qt::WindowFlags) \n\
\n\
See Also \n\
---------- \n\
getWindowFlags()");
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

    QMetaObject::invokeMethod(uiOrga, "setWindowFlags", Q_ARG(uint, self->objectID), Q_ARG(int, value), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while setting window flags");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemGetWindowFlags_doc,"getWindowFlags(flags) -> gets window flags of corresponding widget. \n\
\n\
The flags-value is an or-combination of the enumeration Qt::WindowType. See Qt documentation for more information. \n\
\n\
Returns \n\
-------- \n\
flags {int}: \n\
    or-combination of Qt::WindowType describing the type and further hints of the user interface \n\
\n\
See Also \n\
--------- \n\
setWindowFlags()");
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

    QMetaObject::invokeMethod(uiOrga, "getWindowFlags", Q_ARG(uint, self->objectID), Q_ARG(QSharedPointer<int>, value), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting window flag");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    return Py_BuildValue("i", *value);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ PyObject* PythonUi::PyUiItem_info(PyUiItem *self)
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
//    QSharedPointer< QVariantMap > value(new QVariantMap );

    //!> we need this as otherwise the Q_ARG macro does not recognize our templated QMap
//    QMetaObject::invokeMethod(uiOrga, "getObjectInfo", Q_ARG(uint, self->objectID), Q_ARG(int,UiOrganizer::infoShowItomInheritance), Q_ARG(QSharedPointer<QVariantMap>, value), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    QMetaObject::invokeMethod(uiOrga, "getObjectInfo", Q_ARG(uint, self->objectID), Q_ARG(int,UiOrganizer::infoShowItomInheritance), Q_ARG(ito::UiOrganizer::tQMapArg*, NULL), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyUiItemExists_doc,"exists() -> returns true if widget still exists, else false.");
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
    QMetaObject::invokeMethod(uiOrga, "exists", Q_ARG(uint, self->objectID), Q_ARG(QSharedPointer<bool>,exists), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while getting information");
        return NULL;
    }

    retValue += locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if (*exists)
    {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PythonUi::loadMethodDescriptionList(PyUiItem *self)
{
    if(self->methodList == NULL)
    {
        QByteArray className(self->widgetClassName);
        QHash<QByteArray, QSharedPointer<MethodDescriptionList> >::const_iterator it = methodDescriptionListStorage.constFind( className );
        if(it != methodDescriptionListStorage.constEnd())
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

            QMetaObject::invokeMethod(uiOrga, "getMethodDescriptions", Q_ARG(uint, self->objectID), Q_ARG(QSharedPointer<MethodDescriptionList>, methodList), Q_ARG(ItomSharedSemaphore*, locker1.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
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

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUi::PyUiItem_getattro(PyUiItem *self, PyObject *name)
{
    //UiItem has no __dict__ attribute and this is no widget either, therefore filter it out and raise an exception
    if (PyUnicode_CompareWithASCIIString(name, "__dict__") == 0)
    {
        return PyErr_Format(PyExc_AttributeError, "'%.50s' object has no attribute '%U'", self->objName, name);
    }

    PyObject *ret = PyObject_GenericGetAttr((PyObject*)self,name);
    if(ret != NULL)
    {
        return ret;
    }
    PyErr_Clear(); //genericgetattr throws an error, if attribute is not available, which it isn't for attributes pointing to widgetNames

    //return new instance of PyUiItem
    PyObject *arg2 = Py_BuildValue("OO", self, name);
    PythonUi::PyUiItem *PyUiItem = (PythonUi::PyUiItem *)PyObject_CallObject((PyObject *)&PythonUi::PyUiItemType, arg2);
    Py_DECREF(arg2);

    if(PyUiItem == NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Could not create uiItem of requested widget");
        return NULL;
    }

    if(PyErr_Occurred())
    {
        Py_XDECREF(PyUiItem);
        PyUiItem = NULL;
    }

    return (PyObject *)PyUiItem;
}

//----------------------------------------------------------------------------------------------------------------------------------
int PythonUi::PyUiItem_setattro(PyUiItem *self, PyObject *name, PyObject *value)
{
    return PyObject_GenericSetAttr( (PyObject*)self, name, value );
}

//----------------------------------------------------------------------------------------------------------------------------------
//returns borrowed reference to overall ui-instance or NULL, if not existing
///*static*/ PythonUi::PyUi* PythonUi::PyUiItem_getParentUI(PyUiItem *self)
//{
//    PyUi *result = NULL;
//    PyUiItem *item = self;
//    while(item && !result)
//    {
//        if(PyUi_Check(item))
//        {
//            result = (PythonUi::PyUi*)(item);
//        }
//        item = (PyUiItem*)(item->baseItem);
//    }
//    return result;
//}

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

//----------------------------------------------------------------------------------------------------------------------------------
PyMethodDef PythonUi::PyUiItem_methods[] = {
        {"call", (PyCFunction)PyUiItem_call, METH_VARARGS, PyUiItemCall_doc},
        {"connect", (PyCFunction)PyUiItem_connect, METH_VARARGS, PyUiItemConnect_doc},
        {"disconnect", (PyCFunction)PyUiItem_disconnect, METH_VARARGS, PyUiItemDisconnect_doc},
        {"setProperty", (PyCFunction)PyUiItem_setProperties, METH_VARARGS, PyUiItemSetProperty_doc},
        {"getProperty", (PyCFunction)PyUiItem_getProperties, METH_VARARGS, PyUiItemGetProperty_doc},
        {"getPropertyInfo", (PyCFunction)PyUiItem_getPropertyInfo, METH_VARARGS, PyUiItemGetPropertyInfo_doc},
        {"setAttribute", (PyCFunction)PyUiItem_setAttribute, METH_VARARGS, PyUiItemSetAttribute_doc},
        {"getAttribute", (PyCFunction)PyUiItem_getAttribute, METH_VARARGS, PyUiItemGetAttribute_doc},
        {"getWindowFlags", (PyCFunction)PyUiItem_getWindowFlags, METH_NOARGS, PyUiItemGetWindowFlags_doc},
        {"setWindowFlags", (PyCFunction)PyUiItem_setWindowFlags, METH_VARARGS, PyUiItemSetWindowFlags_doc},
        {"invokeKeyboardInterrupt", (PyCFunction)PyUiItem_connectKeyboardInterrupt, METH_VARARGS, PyUiItemConnectKeyboardInterrupt_doc},
        {"info", (PyCFunction)PyUiItem_info, METH_NOARGS, NULL},
        {"exists", (PyCFunction)PyUiItem_exists, METH_NOARGS, PyUiItemExists_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonUi::PyUiItem_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonUi::PyUiItemModule = {
        PyModuleDef_HEAD_INIT,
        "uiItem",
        "Any item of user interface (dialog, windows...). The item corresponds to any child-object of the overall dialog or window.",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
PyGetSetDef PythonUi::PyUiItem_getseters[] = {
    {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
PyMappingMethods PythonUi::PyUiItem_mappingProtocol = {
    (lenfunc)PyUiItem_mappingLength,
    (binaryfunc)PyUiItem_mappingGetElem,
    (objobjargproc)PyUiItem_mappingSetElem
};

//----------------------------------------------------------------------------------------------------------------------------------
void PythonUi::PyUiItem_addTpDict(PyObject * /*tp_dict*/)
{
    //nothing
}





//----------------------------------------------------------------------------------------------------------------------------------OK
void PythonUi::PyUi_dealloc(PyUi* self)
{
    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga && self->uiHandle >= 0)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        ito::RetVal retValue = retOk;

        QMetaObject::invokeMethod(uiOrga, "deleteDialog", Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
        if(!locker.getSemaphore()->wait(PLUGINWAIT))
        {
            std::cerr << "timeout while closing dialog" << std::endl;
            //PyErr_SetString(PyExc_RuntimeError, "timeout while closing dialog");
        }
    }

    DELETE_AND_SET_NULL( self->signalMapper );
    DELETE_AND_SET_NULL_ARRAY( self->filename );
    Py_XDECREF(self->dialogButtons);

    PyUiItemType.tp_dealloc( (PyObject*)self );
}

//----------------------------------------------------------------------------------------------------------------------------------
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
        self->winType = 0;
        self->buttonBarType = 0;
        self->childOfMainWindow = true; //default
        self->deleteOnClose = false; //default
        self->filename = NULL;
        self->signalMapper = NULL;
    }

    return (PyObject *)self;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiInit_doc,"ui(filename, [type, dialogButtonBar, dialogButtons, childOfMainWindow, deleteOnClose, dockWidgetArea]) -> instance of user interface \n\
\n\
The class **ui** wraps a user interface, externally designed and given by a ui-file. If your user interface is a dialog or window, \n\
chose *ui.TYPEWINDOW* as type, if the user interface is a widget (simplest case), chose *ui.TYPEDIALOG* and your widget \n\
will be embedded in a dialog, provided by *itom*. This dialog can be equiped with a button bar, whose buttons are already \n\
connected to *itom* internal methods. If you then show your dialog in a modal mode, *itom* knows which button has been \n\
clicked in order to accept or reject the dialog. \n\
\n\
Parameters \n\
----------- \n\
filename : {str} \n\
    path to user interface file (*.ui), absolute or relative to current directory \n\
type : {int}, optional \n\
    display type: \n\
    \n\
        * 0 (ui.TYPEDIALOG): ui-file is embedded in auto-created dialog (default), \n\
        * 1 (ui.TYPEWINDOW): ui-file is handled as main window, \n\
        * 2 (ui.TYPEDOCKWIDGET): ui-file is handled as dock-widget and appended to the main-window dock area \n\
dialogButtonBar :  {int}, optional \n\
    Only for type ui.TYPEDIALOG (0). Indicates whether buttons should automatically be added to the dialog: \n\
    \n\
        * 0 (ui.BUTTONBAR_NO): do not add any buttons (default) \n\
        * 1 (ui.BUTTONBAR_HORIZONTAL): add horizontal button bar \n\
        * 2 (ui.BUTTONBAR_VERTICAL): add vertical button bar \n\
dialogButtons : {dict}, optional \n\
    every dictionary-entry is one button. key is the role, value is the button text \n\
childOfMainWindow :  {bool}, optional \n\
    for type TYPEDIALOG and TYPEWINDOW only. Indicates whether window should be a child of itom main window (default: True) \n\
deleteOnClose : {bool}, optional \n\
    Indicates whether window should be deleted if user closes it or if it is hidden (default: Hidden, False) \n\
dockWidgetArea : {int}, optional \n\
    Only for type ui.TYPEDOCKWIDGET (2). Indicates the position where the dock widget should be placed: \n\
    \n\
        * 1 (ui.LEFTDOCKWIDGETAREA) \n\
        * 2 (ui.RIGHTDOCKWIDGETAREA) \n\
        * 4 (ui.TOPDOCKWIDGETAREA): default \n\
        * 8 (ui.BOTTOMDOCKWIDGETAREA)");
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

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iiO!bbi", const_cast<char**>(kwlist), &PyUnicode_FSConverter, &bytesFilename, &self->winType, &self->buttonBarType, &PyDict_Type, &dialogButtons, &self->childOfMainWindow, &self->deleteOnClose, &dockWidgetArea))
    {
        //PyErr_SetString(PyExc_TypeError,"Arguments does not fit to required list of arguments. See help(ui)."); //message is already set by method above and the text is more specific.
        //Py_XDECREF(bytesFilename); //error: crash if bytesFilename is deleted here. Why?
        return -1;
    }

    //check values:
    if(self->winType < 0 || self->winType > 2)
    {
        PyErr_SetString(PyExc_ValueError,"Argument 'type' must have one of the values TYPEDIALOG (0), TYPEWINDOW (1) or TYPEDOCKWIDGET (2)");
        Py_XDECREF(bytesFilename);
        return -1;
    }

    if(self->buttonBarType < 0 || self->buttonBarType > 2)
     {
        PyErr_SetString(PyExc_ValueError,"Argument 'dialogButtonBar' must have one of the values BUTTONBAR_NO (0), BUTTONBAR_HORIZONTAL (1) or BUTTONBAR_VERTICAL (2)");
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
    QSharedPointer<unsigned int> initSlotCount(new unsigned int);
    *dialogHandle = 0;
    *initSlotCount = 0;
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
                std::cout << "Warning while parsing dialogButtons-dictionary. At least one element does not contain a string as key and value" << std::endl;
            }
            else
            {
                dialogButtonMap[keyString] = valueString;
            }
        }
    }

    int uiDescription = UiOrganizer::createUiDescription(self->winType,self->buttonBarType,self->childOfMainWindow,self->deleteOnClose, dockWidgetArea);
    QSharedPointer<QByteArray> className(new QByteArray());
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QMetaObject::invokeMethod(uiOrga, "createNewDialog",Q_ARG(QString,QString(self->filename)), Q_ARG(int, uiDescription), Q_ARG(StringMap, dialogButtonMap), Q_ARG(QSharedPointer<uint>, dialogHandle),Q_ARG(QSharedPointer<uint>, initSlotCount), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(QSharedPointer<QByteArray>, className), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    

    if(!locker.getSemaphore()->wait(60000))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while opening dialog");
        return -1;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return -1;

    self->uiHandle = static_cast<int>(*dialogHandle);
    DELETE_AND_SET_NULL( self->signalMapper );
    self->signalMapper = new PythonQtSignalMapper(*initSlotCount);

    PyObject *args2 = PyTuple_New(3);
    PyTuple_SetItem(args2,0,PyLong_FromLong(*objectID) );
    PyTuple_SetItem(args2,1, PyUnicode_FromString("<ui>") );
    PyTuple_SetItem(args2,2, PyUnicode_FromString(className->data()) );
    int result = PyUiItemType.tp_init((PyObject*)self,args2,NULL);
    Py_DECREF(args2);

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUi::PyUi_repr(PyUi *self)
{
    PyObject *result;
    if(self->uiHandle < 0)
    {
        result = PyUnicode_FromFormat("Ui(empty)");
    }
    else
    {
        if(self->filename == NULL)
        {
            result = PyUnicode_FromFormat("Ui(handle: %i) + %U", self->uiHandle, PyUiItemType.tp_repr((PyObject*)self) );
        }
        else
        {
            result = PyUnicode_FromFormat("Ui(filename: '%s', handle: %i) + %U", self->filename, self->uiHandle, PyUiItemType.tp_repr((PyObject*)self) );
        }
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiShow_doc,"show([modal=0]) -> shows initialized UI-Dialog \n\
\n\
Parameters \n\
----------- \n\
modal : {int}, optional \n\
    * 0: non-modal (default)\n\
    * 1: modal (python waits until dialog is hidden)\n\
    * 2: modal (python returns immediately)\n\
\n\
See Also \n\
--------- \n\
hide()");
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

    QMetaObject::invokeMethod(uiOrga, "showDialog", Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)) , Q_ARG(int,modalLevel), Q_ARG(QSharedPointer<int>, retCodeIfModal), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(modalLevel == 1)
    {
        /*while(!locker.getSemaphore()->wait(10))
        {
            QCoreApplication::processEvents();
            QCoreApplication::sendPostedEvents();
        }
*/
        locker.getSemaphore()->waitAndProcessEvents(-1);
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

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiHide_doc, "hide() -> hides initialized user interface \n\
\n\
See Also \n\
--------- \n\
show(modal)");
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

    QMetaObject::invokeMethod(uiOrga, "hideDialog", Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while hiding dialog");
        return NULL;
    }
    
    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiIsVisible_doc,"isVisible() -> returns true if dialog is still visible\n\
\n\
Returns \n\
------- \n\
visibility : {bool} \n\
    True if user interface is visible, False if it is hidden");
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

    QMetaObject::invokeMethod(uiOrga, "isVisible", Q_ARG(uint, static_cast<unsigned int>(self->uiHandle)), Q_ARG(QSharedPointer<bool>, visible), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
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
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetDouble_doc,"getDouble(title, label, defaultValue [, min, max, decimals=3]) -> shows a dialog to get a double value from the user\n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the dialog title \n\
label : {str}\n\
    is the label above the spin box \n\
defaultValue : {double}, optional\n\
    is the default value in the spin box \n\
min : {double}, optional\n\
    default = -2147483647.0\n\
    is the allowed minimal value\n\
max : {double}, optional\n\
    default = 2147483647.0\n\
    is the allowed maximal value\n\
decimals : {int}, optional\n\
    the maximum number of decimal places (default: 1) \n\
\n\
Returns \n\
------- \n\
out : {tuple (double, bool)} \n\
    A tuple where the first value contains the current double value. The second value is True if the dialog has been accepted, else False. \n\
\n\
See Also \n\
--------- \n\
getInt, getText, getItem");
PyObject* PythonUi::PyUi_getDouble(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultValue", "min", "max", "decimals", NULL};
    const char *title = 0;
    const char *label = 0;
    double defaultValue = 0;
    double minValue = -2147483647;
    double maxValue = 2147483647;
    int decimals = 1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ssd|ddi", const_cast<char**>(kwlist), &title, &label, &defaultValue, &minValue, &maxValue, &decimals))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), default value (double), min (double, optional), max (double, optional), decimals (int, optional)");
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

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetDouble", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(double, defaultValue), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<double>, retDblValue), Q_ARG(double,minValue), Q_ARG(double,maxValue), Q_ARG(int,decimals), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
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
    /*if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }*/
    
    if(*retOk == true)
    {
        //Py_INCREF(Py_True);
        return Py_BuildValue("dO", *retDblValue, Py_True );
    }
    else
    {
        //Py_INCREF(Py_False);
        return Py_BuildValue("dO", defaultValue, Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetInt_doc,"getInt(title, label, defaultValue [, min, max, step=1]) -> shows a dialog to get an integer value from the user\n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the dialog title \n\
label : {str}\n\
    is the label above the spinbox \n\
defaultValue : {int}, optional\n\
    is the default value in the spinbox \n\
min : {int}, optional\n\
    is the allowed minimal value (default: -2147483647) \n\
max : {int}, optional\n\
    is the allowed maximal value (default: 2147483647) \n\
step : {int}, optional\n\
    is the step size if user presses the up/down arrow (default: 1)\n\
\n\
Returns \n\
------- \n\
out : {tuple (int, bool)} \n\
    A tuple where the first value contains the current integer value. The second value is True if the dialog has been accepted, else False. \n\
\n\
See Also \n\
--------- \n\
getDouble, getText, getItem");
PyObject* PythonUi::PyUi_getInt(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultValue", "min", "max", "step", NULL};
    const char *title = 0;
    const char *label = 0;
    int defaultValue = 0;
    int minValue = -2147483647;
    int maxValue = 2147483647;
    int step = 1;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ssi|iii", const_cast<char**>(kwlist), &title, &label, &defaultValue, &minValue, &maxValue, &step))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), default value (int), min (int, optional), max (int, optional), step (int, optional)");
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

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetInt", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(int, defaultValue), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<int>, retIntValue), Q_ARG(int,minValue), Q_ARG(int,maxValue), Q_ARG(int,step), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
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
    /*if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }*/
    
    if(*retOk == true)
    {
        //Py_INCREF(Py_True);
        return Py_BuildValue("iO", *retIntValue, Py_True );
    }
    else
    {
        //Py_INCREF(Py_False);
        return Py_BuildValue("iO", defaultValue, Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetItem_doc,"getItem(title, label, stringList [, currentIndex=0, editable=True]) -> shows a dialog to let the user select an item from a string list\n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the dialog title \n\
label : {str}\n\
    is the label above the text box \n\
stringList : {tuple or list}, optional \n\
    is a list or tuple of possible string values \n\
currentIndex : {int}, optional\n\
    defines the preselected value index (default: 0)\n\
editable : {bool}, optional\n\
    defines whether new entries can be added (True) or not (False, default)\n\
\n\
Returns \n\
------- \n\
out : {tuple (str, bool)} \n\
    A tuple where the first value contains the current active or typed string value. The second value is True if the dialog has been accepted, else False. \n\
\n\
See Also \n\
--------- \n\
getInt, getDouble, getText");
PyObject* PythonUi::PyUi_getItem(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "stringList", "currentIndex", "editable", NULL};
    const char *title = 0;
    const char *label = 0;
    PyObject *stringList = NULL;
    int currentIndex = 0;
    bool editable = false;
    QStringList stringListQt;
    QString temp;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ssO|ib", const_cast<char**>(kwlist), &title, &label, &stringList, &currentIndex, &editable))
    {
        //PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), string list (list, tuple), currentIndex (int), editable (bool)");
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
    

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetItem", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(QStringList, stringListQt), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<QString>, retString), Q_ARG(int, currentIndex), Q_ARG(bool, editable), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
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
    /*if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }*/
    
    if(*retOk == true)
    {
        //Py_INCREF(Py_True);
        QByteArray ba = retString->toLatin1();
        return Py_BuildValue("sO", ba.data(), Py_True );
    }
    else
    {
        //Py_INCREF(Py_False);
        return Py_BuildValue("sO", "", Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetText_doc,"getText(title, label, defaultString) -> opens a dialog in order to ask the user for a string \n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the dialog title \n\
label : {str}\n\
    is the label above the text box \n\
defaultString : {str}\n\
    is the default string in the text box\n\
\n\
Returns \n\
------- \n\
out : {tuple (str, bool)} \n\
    A tuple where the first value contains the current string value. The second value is True if the dialog has been accepted, else False. \n\
\n\
See Also \n\
--------- \n\
getInt, getDouble, getItem");
PyObject* PythonUi::PyUi_getText(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"title", "label", "defaultString", NULL};
    const char *title = 0;
    const char *label = 0;
    const char *defaultString = 0;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "sss", const_cast<char**>(kwlist), &title, &label, &defaultString))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), default string (string)");
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

    QMetaObject::invokeMethod(uiOrga, "showInputDialogGetText", Q_ARG(QString, QString(title)), Q_ARG(QString, QString(label)), Q_ARG(QString, QString(defaultString)), Q_ARG(QSharedPointer<bool>, retOk), Q_ARG(QSharedPointer<QString>, retStringValue), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
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
    /*if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing input dialog");
        return NULL;
    }*/
    
    if(*retOk == true)
    {
        //Py_INCREF(Py_True);
        return Py_BuildValue("sO", retStringValue->toLatin1().data(), Py_True );
    }
    else
    {
        //Py_INCREF(Py_False);
        return Py_BuildValue("sO", defaultString, Py_False );
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgInformation_doc,"msgInformation(title, text [, buttons, defaultButton, parent]) -> opens an information message box \n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the message box title \n\
text : {str}\n\
    is the message text \n\
buttons : {int}, optional\n\
    is an or-combination of ui.MsgBox[...]-constants indicating the buttons to display. Use | for the or-combination. \n\
defaultButton : {int}, optional\n\
    is a value of ui.MsgBox[...] which indicates the default button \n\
parent : {ui}, optional\n\
    is the parent dialog of the message box.\n\
\n\
See Also \n\
--------- \n\
msgCritical, msgQuestion, msgWarning");                                  
PyObject* PythonUi::PyUi_msgInformation(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,1);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgQuestion_doc,"msgQuestion(title, text [, buttons, defaultButton, parent]) -> opens a question message box \n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the message box title \n\
text : {str}\n\
    is the message text \n\
buttons : {int}, optional\n\
    is an or-combination of ui.MsgBox[...]-constants indicating the buttons to display. Use | for the or-combination. \n\
defaultButton : {int}, optional\n\
    is a value of ui.MsgBox[...] which indicates the default button \n\
parent : {ui}, optional\n\
    is the parent dialog of the message box.\n\
\n\
See Also \n\
--------- \n\
msgCritical, msgWarning, msgInformation");
PyObject* PythonUi::PyUi_msgQuestion(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,2);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgWarning_doc,"msgWarning(title, text [, buttons, defaultButton, parent]) -> opens a warning message box \n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the message box title \n\
text : {str}\n\
    is the message text \n\
buttons : {int}, optional\n\
    is an or-combination of ui.MsgBox[...]-constants indicating the buttons to display. Use | for the or-combination. \n\
defaultButton : {int}, optional\n\
    is a value of ui.MsgBox[...] which indicates the default button \n\
parent : {ui}, optional\n\
    is the parent dialog of the message box.\n\
\n\
See Also \n\
--------- \n\
msgCritical, msgQuestion, msgInformation");
PyObject* PythonUi::PyUi_msgWarning(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,3);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiMsgCritical_doc,"msgCritical(title, text [, buttons, defaultButton, parent]) -> opens a critical message box \n\
\n\
Parameters \n\
----------- \n\
title : {str}\n\
    is the message box title \n\
text : {str}\n\
    is the message text \n\
buttons : {int}, optional\n\
    is an or-combination of ui.MsgBox[...]-constants indicating the buttons to display. Use | for the or-combination. \n\
defaultButton : {int}, optional\n\
    is a value of ui.MsgBox[...] which indicates the default button \n\
parent : {ui}, optional\n\
    is the parent dialog of the message box.\n\
\n\
See Also \n\
--------- \n\
msgWarning, msgQuestion, msgInformation");
PyObject* PythonUi::PyUi_msgCritical(PyUi *self, PyObject *args, PyObject *kwds)
{
    return PyUi_msgGeneral(self,args,kwds,4);
}

//----------------------------------------------------------------------------------------------------------------------------------
PyObject* PythonUi::PyUi_msgGeneral(PyUi * /*self*/, PyObject *args, PyObject *kwds, int type)
{
    const char *kwlist[] = {"title", "text", "buttons", "defaultButton", "parent", NULL};
    const char *title = 0;
    const char *text = 0;
    int buttons = QMessageBox::Ok;
    int defaultButton = QMessageBox::NoButton;
    PythonUi::PyUi *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ss|iiO!", const_cast<char**>(kwlist), &title, &text, &buttons, &defaultButton, &PythonUi::PyUiType, &parentItem))
    {
        PyErr_SetString(PyExc_TypeError, "arguments must be title (string), label (string), buttons (combination of ui.MsgBox[...]), defaultButton (ui.MsgBox[...])");
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
    unsigned int parentUiHandle = parentItem ? parentItem->uiHandle : 0;
    QMetaObject::invokeMethod(uiOrga, "showMessageBox", Q_ARG(uint, parentUiHandle), Q_ARG(int, type), Q_ARG(QString, QString(title)), Q_ARG(QString, QString(text)), Q_ARG(int, buttons), Q_ARG(int, defaultButton), Q_ARG(QSharedPointer<int>, retButton), Q_ARG(QSharedPointer<QString>, retButtonText), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
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
            PyErr_SetString(PyExc_RuntimeError, "timeout while showing message box");
            return NULL;
        }
    }
    /*if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing message box");
        return NULL;
    }*/

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;
    
    return Py_BuildValue("is", *retButton, retButtonText->toLatin1().data());
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetExistingDirectory_doc,"getExistingDirectory(caption, startDirectory [, options, parent]) -> opens a dialog to choose an existing directory \n\
\n\
Parameters \n\
----------- \n\
caption : {str}\n\
    is the caption of this dialog \n\
startDirectory : {str}\n\
    is the start directory \n\
options : {int}, optional\n\
    is an or-combination of the following options (see 'QFileDialog::Option'): \n\
    \n\
        * 1: ShowDirsOnly [default] \n\
        * 2: DontResolveSymlinks \n\
        * ... (for others see Qt-Help) \n\
parent : {ui}, optional\n\
    is a parent dialog or window, this dialog becomes modal.\n\
\n\
Returns \n\
------- \n\
out : {str, None} \n\
    The selected directory is returned as absolute path or None if the dialog has been rejected. \n\
\n\
See Also \n\
--------- \n\
getSaveFileName, getOpenFileName");
PyObject* PythonUi::PyUi_getExistingDirectory(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    const char *kwlist[] = {"caption", "directory", "options", "parent", NULL};
    const char *caption = 0;
    const char *directory = 0;
    int options = 1; //QFileDialog::ShowDirsOnly
    PythonUi::PyUi *parentItem = NULL;


    if(!PyArg_ParseTupleAndKeywords(args, kwds, "ss|iO!", const_cast<char**>(kwlist), &caption, &directory, &options, &PythonUi::PyUiType, &parentItem))
    {
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

    unsigned int parentHandle = (parentItem) ? parentItem->uiHandle : 0;
    QSharedPointer<QString> sharedDir(new QString(directory));

    QMetaObject::invokeMethod(uiOrga, "showFileDialogExistingDir", Q_ARG(uint, parentHandle), Q_ARG(QString, QString(caption)), Q_ARG(QSharedPointer<QString>, sharedDir), Q_ARG(int, options), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(sharedDir->isEmpty() || sharedDir->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", sharedDir->toLatin1().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetOpenFileName_doc,"getOpenFileName([caption, startDirectory, filters, selectedFilterIndex, options, parent]) -> opens dialog for chosing an existing file. \n\
\n\
Parameters \n\
----------- \n\
caption : {str}, optional\n\
    This is the optional title of the dialog, default: no title \n\
startDirectory {str}, optional\n\
    optional, if not indicated currentDirectory will be taken\n\
filters : {str}, optional\n\
    default = 0\n\
    possible filter list, entries should be separated by ;; , e.g. 'Images (*.png *.jpg);;Text files (*.txt)' \n\
selectedFilterIndex : {int}, optional \n\
    is the index of filters which is set by default (0 is first entry) \n\
options : {int}, optional\n\
    default =  0 \n\
    or-combination of enum values QFileDialog::Options \n\
parent : {ui}, optional\n\
    is the parent widget of this dialog \n\
\n\
Returns \n\
------- \n\
out : {str, None} \n\
    filename as string or None if dialog has been aborted.\n\
\n\
See Also \n\
--------- \n\
getSaveFileName");
PyObject* PythonUi::PyUi_getOpenFileName(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    
    const char *kwlist[] = {"caption", "startDirectory", "filters", "selectedFilterIndex", "options", "parent", NULL};
    const char *caption = "";
    const char *directory = "";
    const char *filters = "";
    int selectedFilterIndex = 0;
    int options = 0;
    PythonUi::PyUi *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sssiiO!", const_cast<char**>(kwlist), &caption, &directory, &filters, &selectedFilterIndex, &options, &PythonUi::PyUiType, &parentItem))
    {
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
    unsigned int parentHandle = (parentItem) ? parentItem->uiHandle : 0;

    QSharedPointer<QString> file(new QString());
    //QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore
    QMetaObject::invokeMethod(uiOrga, "showFileOpenDialog", Q_ARG(uint, parentHandle), Q_ARG(QString, QString(caption)), Q_ARG(QString, QString(directory)), Q_ARG(QString, QString(filters)), Q_ARG(QSharedPointer<QString>, file), Q_ARG(int, selectedFilterIndex), Q_ARG(int, options), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(file->isEmpty() || file->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", file->toLatin1().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiGetSaveFileName_doc,"getSaveFileName([caption, startDirectory, filters, selectedFilterIndex, options, parent]) -> opens dialog for chosing a file to save. \n\
\n\
This method creates a modal file dialog to let the user select a file name used for saving a file. \n\
\n\
Parameters \n\
----------- \n\
caption : {str}, optional\n\
    This is the title of the dialog \n\
startDirectory : {String}, optional\n\
    if not indicated, the current working directory will be taken\n\
filters : {str}, optional\n\
    possible filter list, entries should be separated by ;; , e.g. 'Images (*.png *.jpg);;Text files (*.txt)' \n\
selectedFilterIndex : {int}, optional\n\
    default = 0\n\
    is the index of filters which is set by default (0 is first entry) \n\
options : {int}, optional\n\
    default = 0\n\
    or-combination of enum values QFileDialog::Options \n\
parent : {ui}, optional\n\
    is the parent widget of this dialog\n\
\n\
Returns \n\
------- \n\
out : {str, None} \n\
    filename as string or None if dialog has been aborted.\n\
\n\
See Also \n\
--------- \n\
getOpenFileName");
PyObject* PythonUi::PyUi_getSaveFileName(PyUi * /*self*/, PyObject *args, PyObject *kwds)
{
    
    const char *kwlist[] = {"caption", "startDirectory", "filters", "selectedFilterIndex", "options", "parent", NULL};
    const char *caption = "";
    const char *directory = "";
    const char *filters = "";
    int selectedFilterIndex = 0;
    int options = 0;
    PythonUi::PyUi *parentItem = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|sssiiO!", const_cast<char**>(kwlist), &caption, &directory, &filters, &selectedFilterIndex, &options, &PythonUi::PyUiType, &parentItem))
    {
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
    unsigned int parentHandle = (parentItem) ? parentItem->uiHandle : 0;

    QSharedPointer<QString> file(new QString());
    //QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore
    QMetaObject::invokeMethod(uiOrga, "showFileSaveDialog", Q_ARG(uint, parentHandle), Q_ARG(QString, QString(caption)), Q_ARG(QString, QString(directory)), Q_ARG(QString, QString(filters)), Q_ARG(QSharedPointer<QString>, file), Q_ARG(int, selectedFilterIndex), Q_ARG(int, options), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while showing dialog");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    if(file->isEmpty() || file->isNull())
    {
        Py_RETURN_NONE;
    }
    else
    {
        return Py_BuildValue("s", file->toLatin1().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(pyUiCreateNewPluginWidget_doc, "createNewPluginWidget(widgetName[, mandparams, optparams]) -> creates widget defined by any algorithm plugin and returns the instance of type 'ui' \n\
Parameters \n\
----------- \n\
widgetName : {}\n\
    name algorithm widget \n\
    parameters to pass to the plugin. The parameters are parsed and unnamed parameters are used in their \
    incoming order to fill first mandatory parameters and afterwards optional parameters. Parameters may be passed \
    with name as well but after the first named parameter no more unnamed parameters are allowed.\n\
\n\
Returns \n\
------- \n\
\n\
Notes \n\
----- \n\
doctodo\n\
\n\
See Also \n\
--------- \n\
\n\
");
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
    ito::RetVal retVal = 0;
    PyObject *pnameObj = NULL;
    PyObject *params = NULL;
    QString algoWidgetName;
    bool ok;

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
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

/*
    
    retVal = def->m_paramFunc(&paramsMand, &paramsOpt);
    if (retVal.containsWarningOrError())
    {
        PyErr_SetString(PyExc_RuntimeError, QObject::tr("Could not load default parameter set for loading plugin widget. Error-Message: \n%s\n").toLatin1(), QObject::tr(retVal.errorMessage()).toLatin1().data());
        return NULL;
    }*/

    params = PyTuple_GetSlice(args, 1, PyTuple_Size(args));
    if(parseInitParams(&(filterParams->paramsMand), &(filterParams->paramsOpt), params, kwds, paramsMandBase, paramsOptBase) != ito::retOk)
    //if (parseInitParams(&paramsMand, &paramsOpt, params, kwds) != ito::retOk)
    {
        PyErr_SetString(PyExc_RuntimeError, "error while parsing parameters.");
        return NULL;
    }
    Py_DECREF(params);

    UiOrganizer *uiOrga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    if(uiOrga == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Instance of UiOrganizer not available");
        return NULL;
    }

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    ito::RetVal retValue = retOk;

    QSharedPointer<unsigned int> dialogHandle(new unsigned int);
    QSharedPointer<unsigned int> initSlotCount(new unsigned int);
    QSharedPointer<unsigned int> objectID(new unsigned int);
    QSharedPointer<QByteArray> className(new QByteArray());
    *dialogHandle = 0;
    *initSlotCount = 0;
    *objectID = 0;
    QMetaObject::invokeMethod(uiOrga, "loadPluginWidget", Q_ARG(void*, reinterpret_cast<void*>(def->m_widgetFunc)), Q_ARG(QVector<ito::ParamBase>*, &paramsMandBase), Q_ARG(QVector<ito::ParamBase>*, &paramsOptBase), Q_ARG(QSharedPointer<uint>, dialogHandle), Q_ARG(QSharedPointer<uint>, initSlotCount), Q_ARG(QSharedPointer<uint>, objectID), Q_ARG(QSharedPointer<QByteArray>, className), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
    
    if(!locker.getSemaphore()->wait(-1))
    {
        PyErr_SetString(PyExc_RuntimeError, "timeout while loading plugin widget");
        return NULL;
    }

    retValue = locker.getSemaphore()->returnValue;
    if(!PythonCommon::transformRetValToPyException(retValue)) return NULL;

    PythonUi::PyUi *dialog;

    PyObject *emptyTuple = PyTuple_New(0);
    dialog = (PyUi*)PyObject_Call((PyObject*)&PyUiType, NULL, NULL); //new ref, tp_new of PyUi is called, init not
    Py_XDECREF(emptyTuple);

    if(dialog == NULL)
    {
        if(*dialogHandle)
        {
            ItomSharedSemaphoreLocker locker2(new ItomSharedSemaphore());
            QMetaObject::invokeMethod(uiOrga, "deleteDialog", Q_ARG(uint, static_cast<unsigned int>(*dialogHandle)), Q_ARG(ItomSharedSemaphore*, locker2.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    
            if(!locker2.getSemaphore()->wait(PLUGINWAIT))
            {
                PyErr_SetString(PyExc_RuntimeError, "timeout while closing dialog");
            }
        }

        PyErr_SetString(PyExc_RuntimeError, "could not create a new instance of class ui.");
        return NULL;
    }

    dialog->uiHandle = static_cast<int>(*dialogHandle);
    dialog->signalMapper = new PythonQtSignalMapper(*initSlotCount);
    dialog->uiItem.methodList = NULL;
    dialog->uiItem.objectID = *objectID;
    dialog->uiItem.widgetClassName = new char[className->size()+1];
    strcpy_s(dialog->uiItem.widgetClassName, className->size()+1, className->data());

    const char *objName = "<plugin-widget>\0";
    dialog->uiItem.objName = new char[strlen(objName)+1];
    strcpy_s(dialog->uiItem.objName, strlen(objName)+1, objName);  

    return (PyObject*)dialog;
}


//----------------------------------------------------------------------------------------------------------------------------------
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
        {"getSaveFileName", (PyCFunction)PyUi_getSaveFileName, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiGetSaveFileName_doc},
        {"createNewPluginWidget", (PyCFunction)PyUi_createNewAlgoWidget, METH_KEYWORDS | METH_VARARGS |METH_STATIC, pyUiCreateNewPluginWidget_doc},
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyMemberDef PythonUi::PyUi_members[] = {
        {NULL}  /* Sentinel */
};

//----------------------------------------------------------------------------------------------------------------------------------
PyModuleDef PythonUi::PyUiModule = {
        PyModuleDef_HEAD_INIT,
        "ui",
        "Itom userInterfaceDialog type in python",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

//----------------------------------------------------------------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------------------------------------------
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

