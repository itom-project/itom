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

#include "../python/pythonQtConversion.h"
#include "widgetWrapper.h"


#include <qlistwidget.h>
#include <qcombobox.h>
#include <qmainwindow.h>
#include <qtablewidget.h>
#include <qheaderview.h>
#include <qtreeview.h>
#include <qtableview.h>

namespace ito
{

/*!
    \class WidgetWrapper
    \brief One instance of this class acts as wrapper for several import public methods of classes, derived from QObject,
    which should be made available by the call-method in python. 

    Usually, the huge meta system, provided by QMetaObject, of Qt gives the possibility to call slots and changes properties
    of all classes, derived from QObject, at runtime. Other public methods of these classes however can not be accessed by this
    runtime-system. Since the python-bindings which are integrated in the python-class UiDialog and UiDialogMetaObject use the
    Qt-meta system in order to access properties and connect to signals or call slots, it is also desirable to access other public
    methods of Qt-based classes. Frameworks, like PythonQt or PyQt have an internal parsing system which automatically creates
    wrappers for all public methods; here the class WidgetWrapper is a manually created wrapper for the most important methods.
*/

//! constructor
/*!
    initializes a hash table containing information about all public-methods which should be wrapped and therefore accessed 
    for instance from the python-method "call".

    \sa PythonUiDialog, UiOrganizer
*/
WidgetWrapper::WidgetWrapper() : initialized(false)
{
    initMethodHash();
}

//! destructor
WidgetWrapper::~WidgetWrapper()
{
}

//! initializes the hash table containing information about all methods which should be wrapped.
/*!
    Every public method of a class derived from QObject can be made available for access by Python if some information
    about this method are contained in a corresponding instance of the class MethodDescription. A list of such instances is linked
    to its appropriate class and therefore stored in the hash table methodHash whose key is the Qt-internal name of the corresponding
    class. Classes which are derived from that class also have access to the recently wrapped methods.

    If you want to register new wrapper methods for an Qt-based class with name "newClass", you should at first create a temporary
    variable of type MethodDescriptionList, which is nothing else but a QList<MethodDescription>. Then you can add elements to this list
    for example using the helper-method \sa buildMethodDescription. This method returns a new value of class MethodDescription.
    Finally you add this list of MethodDescription to methodHash with the key "newClass".

    All wrapped methods can finally be called by the method \sa call, which needs the method's index and a pointer to the base class
    QObject*. You must also adapt the method call, in order to successfully finish the wrapping process.

    \sa buildMethodDescription, MethodDescription, MethodDescriptionList
*/
void WidgetWrapper::initMethodHash()
{
    if(!initialized)
    {
	    bool ok;
		
        //QWidget
        MethodDescriptionList qWidgetList;
        qWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("resize(int,int)"), "void", 1001, ok );
        qWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("setGeometry(int,int,int,int)"), "void", 1002, ok );
        methodHash["QWidget"] = qWidgetList;


        //QListWidget
        MethodDescriptionList qListWidgetList;
        qListWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("addItem(QString)"), "void", 2001, ok );
        qListWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("addItems(QStringList)"), "void", 2002, ok );
        methodHash["QListWidget"] = qListWidgetList;

        //QComboBox
        MethodDescriptionList qComboBoxList;
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("addItem(QString)"), "void", 3001, ok );
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("addItems(QStringList)"), "void", 3002, ok );
        methodHash["QComboBox"] = qComboBoxList;

        //QTabWidget
        MethodDescriptionList qTabWidgetList;
        qTabWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("isTabEnabled(int)"), "bool", 4001, ok );
        qTabWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("setTabEnabled(int,bool)"), "void", 4002, ok );
        methodHash["QTabWidget"] = qTabWidgetList;

        //QMainWindow
        MethodDescriptionList qMainWindow;
        qMainWindow << buildMethodDescription(QMetaObject::normalizedSignature("statusBar()"), "ito::PythonQObjectMarshal", 5001, ok );
        qMainWindow << buildMethodDescription(QMetaObject::normalizedSignature("centralWidget()"), "ito::PythonQObjectMarshal", 5002, ok );
        methodHash["QMainWindow"] = qMainWindow;

        //QTableWidget
        MethodDescriptionList qTableWidget;
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setHorizontalHeaderLabels(QStringList)"), "void", 6001, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setVerticalHeaderLabels(QStringList)"), "void", 6002, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("getItem(int,int)"), "QVariant", 6003, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setItem(int,int,QVariant)"), "void", 6004, ok );
        methodHash["QTableWidget"] = qTableWidget;

        //QTableView
        MethodDescriptionList qTableView;
        qTableView << buildMethodDescription(QMetaObject::normalizedSignature("horizontalHeader()"), "ito::PythonQObjectMarshal", 7001, ok );
        qTableView << buildMethodDescription(QMetaObject::normalizedSignature("verticalHeader()"), "ito::PythonQObjectMarshal", 7002, ok );
        methodHash["QTableView"] = qTableView;
    }
}

//! returns a list of MethodDescription, which contains all wrapped methods of the given object and all its base classes.
/*!
    Methods, contained in this list, can be accessed for example by python finally using the method call.

    \param [in] object is the pointer to an instance derived from QObject whose wrapped public methods should be returned
    \return list of wrapped public methods
    \sa methodHash
*/
MethodDescriptionList WidgetWrapper::getMethodList(QObject *object)
{
    MethodDescriptionList list;
    const QMetaObject *tempMO = object->metaObject();
    QString className;

    while(tempMO != NULL)
    {
        className = tempMO->className();
        if(methodHash.contains(className))
        {
            list.append( methodHash[className] );
        }
        tempMO = tempMO->superClass();
    }
    return list;
}

//! creates an instance of MethodDescription which contains all necessary information in order to call the corresponding method at runtime using the method \sa call.
/*!
    This method creates the neccesary input for the constructor of MethodDescription, which has pretty much the same functionality than
    the call of slots in Qt at runtime using the signal/slot and Qt-meta system.

    \param [in] signature is the Qt-like signature string for the method to wrap (e.g. "methodName(argType1,argType2,argType3,...)" without argument names)
    \param [in] typename of the return value or "void" if no return value. This return type must be known by QMetaType, which is the case for all standard types, else use \sa qRegisterMetaType.
    \param [in] a self defined, unique ID for the method; this ID must only be unique within the corresponding Qt-class.
    \param [out] ok returns whether the MethodDescription instance could successfully be built (true).
    \return instance of MethodDescription, empty MethodDescription in case of error
    \sa MethodDescription, call
*/
MethodDescription WidgetWrapper::buildMethodDescription(QByteArray signature, QString retType, int methodIndex, bool &ok)
{
    ok = false;
    int retTypeInt = 0;
    if(retType != "" && QString::compare(retType, "void", Qt::CaseInsensitive) != 0)
    {
        retTypeInt = QMetaType::type(retType.toLatin1().data());
        if(retTypeInt == 0)
            return MethodDescription(); //error
    }

    int start = signature.indexOf("(");
    int end = signature.lastIndexOf(")");
    if(end<=start)
    {
        return MethodDescription(); //error
    }
    QString params = signature.mid(start+1, end-start-1);
    QStringList paramList;
    if(params != "")
    {
        paramList = params.split(",");
    }
    QByteArray name = signature.left(start);
    int nrOfArgs = paramList.size();
    int* args = new int[nrOfArgs];
    bool ok2 = true;
    int type;
    int counter = 0;

    foreach(const QString& param, paramList)
    {
        type = QMetaType::type( param.trimmed().toLatin1().data() );
        if(type == 0)
        {
            ok2 = false;
            break;
        }
        args[counter++] = type;
    }

    if(ok2 == false)
    {
        delete[] args;
        return MethodDescription();
    }

    MethodDescription method(name, signature, QMetaMethod::Method, QMetaMethod::Public, methodIndex, retTypeInt, nrOfArgs, args);
    delete[] args;
    ok = true;
    return method;
}

//! call method which calls a wrapped method of any class derived from QObject at runtime. This call is for exampled executed by UiOrganizer::callMethod.
/*!
    This method uses the class-name of object and checks if there is an call-implementation for the given methodIndex. If so, the
    call is executed (see switch-case inside of the method). If the method index does not fit, the class name of the base class
    of object is iteratively checked until either an appropriate method call could be executed or no other base class is available.

    Please adapt the switch case in order to check for wrapped methods of other classes. Use the void-array _a and a reinterpret_cast-
    operation for accessing the necessary parameters and write back the return value of the "real" call to the wrapped public method.

    \param [in] object is the instance casted to its base class QObject, whose wrapped public method should be called
    \param [in] methodIndex is the ID of the wrapped method to call
    \param [in/out] _a is a void-array containing the value for the return value as first element and the content of all argument as following elements. (similar to qt_metacall)
    \return true if call could successfully be executed (only if call itsself was successfull), false if method could not be found
    \sa UiOrganizer::callMethod
*/
bool WidgetWrapper::call(QObject *object, int methodIndex, void **_a)
{
    //parse the class hierarchie of object recursively and check for possible methods:
    const QMetaObject *tempMetaObject = object->metaObject();
    QString className;

    while( tempMetaObject != NULL )
    {
        className = tempMetaObject->className();
        if(QString::compare(className, "QListWidget", Qt::CaseInsensitive) == 0)
        {
            QListWidget *object2 = qobject_cast<QListWidget*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 2001: //addItem
                object2->addItem((*reinterpret_cast< const QString(*)>(_a[1])));
                //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                return true;
                break;
            case 2002: //addItems
                object2->addItems((*reinterpret_cast< const QStringList(*)>(_a[1])));
                //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                return true;
                break;
            }
        
        }
        else if(QString::compare(className, "QComboBox", Qt::CaseInsensitive) == 0)
        {
            QComboBox *object2 = qobject_cast<QComboBox*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 3001: //addItem
                {
                object2->addItem((*reinterpret_cast< const QString(*)>(_a[1])));
                //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                return true;
                }
            case 3002: //addItems
                {
                object2->addItems((*reinterpret_cast< const QStringList(*)>(_a[1])));
                //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                return true;
                }
            }
        }
        else if(QString::compare(className, "QTabWidget", Qt::CaseInsensitive) == 0)
        {
            QTabWidget *object2 = qobject_cast<QTabWidget*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 4001: //isTabEnabled
                {
                bool _r = object2->isTabEnabled((*reinterpret_cast< const int(*)>(_a[1])));
                (*reinterpret_cast< bool*>(_a[0])) = _r;
                return true;
                }
            case 4002: //setTabEnabled
                {
                object2->setTabEnabled((*reinterpret_cast< const int(*)>(_a[1])),(*reinterpret_cast< const bool(*)>(_a[2])));
                //(*reinterpret_cast< bool*>(_a[0])) = _r;
                return true;
                }
            }
        }
        else if(QString::compare(className, "QMainWindow", Qt::CaseInsensitive) == 0)
        {
            QMainWindow *object2 = qobject_cast<QMainWindow*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 5001: //statusBar
                {
                QWidget* _r = (QWidget*)( object2->statusBar() );
                (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), (void*)_r);
                return true;
                }
            case 5002: //centralWidget
                {
                QWidget* _r = ( object2->centralWidget() );
                (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), (void*)_r);
                return true;
                }
            }
        }
        else if(QString::compare(className, "QWidget", Qt::CaseInsensitive) == 0)
        {
            QWidget *object2 = qobject_cast<QWidget*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 1001: //resize
                {
                object2->resize((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])));
                //(*reinterpret_cast< bool*>(_a[0])) = _r;
                return true;
                }
            case 1002: //setGeometry
                {
                object2->setGeometry((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])), (*reinterpret_cast< const int(*)>(_a[3])), (*reinterpret_cast< const int(*)>(_a[4])));
                //(*reinterpret_cast< bool*>(_a[0])) = _r;
                return true;
                }
            }
        }
        else if(QString::compare(className, "QTableWidget", Qt::CaseInsensitive) == 0)
        {
            QTableWidget *object2 = qobject_cast<QTableWidget*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 6001: //setHorizontalHeaderLabels
                {
                object2->setHorizontalHeaderLabels( (*reinterpret_cast< const QStringList(*)>(_a[1])) );
                //(*reinterpret_cast< bool*>(_a[0])) = _r;
                return true;
                }
            case 6002: //setVerticalHeaderLabels
                {
                object2->setVerticalHeaderLabels( (*reinterpret_cast< const QStringList(*)>(_a[1])) );
                //(*reinterpret_cast< bool*>(_a[0])) = _r;
                return true;
                }
            case 6003: //getItem
                {
                int row = (*reinterpret_cast< const int(*)>(_a[1]));
                int col = (*reinterpret_cast< const int(*)>(_a[2]));
                QTableWidgetItem *item = object2->item(row,col);
                if(item)
                {
                    (*reinterpret_cast<QVariant*>(_a[0])) = item->data(Qt::DisplayRole);
                    return true;
                }
                return false;
                }
            case 6004: //setItem
                {
                    int row = (*reinterpret_cast< const int(*)>(_a[1]));
                    int col = (*reinterpret_cast< const int(*)>(_a[2]));
                    if(row < 0 || row >= object2->rowCount()) return false;
                    if(col < 0 || col >= object2->columnCount()) return false;
                    QTableWidgetItem *item = new QTableWidgetItem();
                    object2->setItem(row,col,item);
                    item->setData(Qt::DisplayRole, (*reinterpret_cast< const QVariant(*)>(_a[3])));
                    return true;
                }
            }
        }
        else if(QString::compare(className, "QTableView", Qt::CaseInsensitive) == 0)
        {
            QTableView *object2 = qobject_cast<QTableView*>(object);
            if(object2 == NULL) return false;
            switch(methodIndex)
            {
            case 7001: //horizontalHeader
                {
                QHeaderView* _r = (QHeaderView*)( object2->horizontalHeader() );
                (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), (void*)_r);
                return true;
                }
            case 7002: //verticalHeader
                {
                QHeaderView* _r = (QHeaderView*)( object2->verticalHeader() );
                (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), (void*)_r);
                return true;
                }
            }
        }

        //nothing found yet, check superclass
        tempMetaObject = tempMetaObject->superClass();
    }


    return false;
}

//! Method is able to handle unexisting properties and map them to existing ones (compatibility to QtDesigner)
/*!
    In QtDesigner, sometimes it is possible to change properties that are not directly extracted from the QMetaObject-system.
    However, QtDesigner adds some artifical sets of properties, especially for widgets derived from QTreeView and QTableView.
    Therefore, if methods UiOrganizer::writeProperties and UiOrganizer::readProperties fail to address the given property,
    they call this method. In the given property can be transformed into a new property of a new object, then this is returned,
    else an empty QMetaProperty is returned.

    \param [in] baseObject is the original input object
    \param [in] fakePropertyName is the artificial property name
    \param [out] destinationObject will be filled with a pointer to the new object, the new property is valid for (or NULL)
    \return instance of QMetaProperty (valid, if fakePropertyName could be converted to another property and object)
    \sa UiOrganizer::writeProperties, UiOrganizer::readProperties
*/
QMetaProperty WidgetWrapper::fakeProperty(const QObject *baseObject, const QString &fakePropertyName, QObject **destinationObject)
{
    //parse the class hierarchie of object recursively and check for possible methods:
    const QMetaObject *tempMetaObject = baseObject->metaObject();
    QString className;
    QString newProperty;
    *destinationObject = NULL;

    while( tempMetaObject != NULL )
    {
        className = tempMetaObject->className();
        if(QString::compare(className, "QTreeView", Qt::CaseInsensitive) == 0) 
        {
            QHeaderView *header = NULL;

            //transforms a property with name "headerProperty" to the property "property" of the header widget
            if(fakePropertyName.startsWith("header"))
            {
                header = (qobject_cast<const QTreeView*>(baseObject))->header();
                *destinationObject = header;
            }
            else
            {
                return QMetaProperty();
            }

            newProperty = fakePropertyName.mid( QString("header").length() );
            newProperty[0] = newProperty[0].toLower();

            tempMetaObject = header->metaObject();
            int idx = tempMetaObject->indexOfProperty(newProperty.toLatin1().data());
            if(idx >= 0)
            {
                return tempMetaObject->property(idx);
            }
            return QMetaProperty();
        }
        else if(QString::compare(className, "QTableView", Qt::CaseInsensitive) == 0)
        {
            QHeaderView *header = NULL;

            //transforms a property with name "verticalHeaderProperty" to the property "property" of the vertical header widget
            if(fakePropertyName.startsWith("verticalHeader"))
            {
                header = (qobject_cast<const QTableView*>(baseObject))->verticalHeader();
                newProperty = fakePropertyName.mid( QString("verticalHeader").length() );
                newProperty[0] = newProperty[0].toLower();
                *destinationObject = header;
            }

            //transforms a property with name "horizontalHeaderProperty" to the property "property" of the horizontal header widget
            else if(fakePropertyName.startsWith("horizontalHeader"))
            {
                header = (qobject_cast<const QTableView*>(baseObject))->horizontalHeader();
                newProperty = fakePropertyName.mid( QString("horizontalHeader").length() );
                newProperty[0] = newProperty[0].toLower();
                *destinationObject = header;
            }
            else
            {
                return QMetaProperty();
            }

            tempMetaObject = header->metaObject();
            int idx = tempMetaObject->indexOfProperty(newProperty.toLatin1().data());
            if(idx >= 0)
            {
                return tempMetaObject->property(idx);
            }
            return QMetaProperty();
        }


        //nothing found yet, check superclass
        tempMetaObject = tempMetaObject->superClass();
    }


    return QMetaProperty();
}

} //end namespace ito
