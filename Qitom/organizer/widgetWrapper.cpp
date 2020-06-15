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

#include "../python/pythonQtConversion.h"
#include "uiOrganizer.h"
#include "widgetWrapper.h"


#include <qlistwidget.h>
#include <qcombobox.h>
#include <qmainwindow.h>
#include <qtablewidget.h>
#include <qheaderview.h>
#include <qtreeview.h>
#include <qtableview.h>
#include <qsplitter.h>
#include <qstatusbar.h>
#include <qtoolbar.h>
#include <qlabel.h>
#include <qpixmap.h>
#include <qlayout.h>
#include <qformlayout.h>
#include <qgridlayout.h>

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

//--------------------------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    initializes a hash table containing information about all public-methods which should be wrapped and therefore accessed 
    for instance from the python-method "call".

    \sa PythonUiDialog, UiOrganizer
*/
WidgetWrapper::WidgetWrapper(UiOrganizer *uiOrganizer) : 
    initialized(false),
    m_pUiOrganizer(uiOrganizer)
{
    initMethodHash();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
//! destructor
WidgetWrapper::~WidgetWrapper()
{
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
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
        qWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("setCursor(int)"), "void", 1003, ok);
        qWidgetList << buildMethodDescription(QMetaObject::normalizedSignature("devicePixelRatioF()"), "float", 1004, ok);
        methodHash["QWidget"] = qWidgetList;


        //QListWidget
        MethodDescriptionList qListWidget;
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("addItem(QString)"), "void", 2001, ok );
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("addItems(QStringList)"), "void", 2002, ok );
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("selectedRows()"), "QVector<int>", 2003, ok );
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("selectedTexts()"), "QStringList", 2004, ok );
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("selectRows(QVector<int>)"), "void", 2005, ok );
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("takeItem(int)"), "QString", 2006, ok);
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("item(int)"), "QString", 2007, ok);
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("checkState(int)"), "Qt::CheckState", 2008, ok);
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("setCheckState(int,Qt::CheckState)"), "void", 2009, ok);
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("flags(int)"), "Qt::ItemFlags", 2010, ok);
		qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("setFlags(int,Qt::ItemFlags)"), "void", 2011, ok);
        qListWidget << buildMethodDescription(QMetaObject::normalizedSignature("setItemText(int,QString)"), "void", 2012, ok);
        methodHash["QListWidget"] = qListWidget;

        //QComboBox
        MethodDescriptionList qComboBoxList;
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("addItem(QString)"), "void", 3001, ok );
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("addItems(QStringList)"), "void", 3002, ok );
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("removeItem(int)"), "void", 3003, ok );
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("setItemData(int,QVariant)"), "void", 3004, ok );
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("insertItem(int,QString)"), "void", 3005, ok );
        qComboBoxList << buildMethodDescription(QMetaObject::normalizedSignature("itemText(int)"), "QString", 3006, ok);
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
        qMainWindow << buildMethodDescription(QMetaObject::normalizedSignature("addToolBar(QString,QString)"), "ito::PythonQObjectMarshal", 5003, ok);
        methodHash["QMainWindow"] = qMainWindow;

        //QTableWidget
        MethodDescriptionList qTableWidget;
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setHorizontalHeaderLabels(QStringList)"), "void", 6001, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setVerticalHeaderLabels(QStringList)"), "void", 6002, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("getItem(int,int)"), "QVariant", 6003, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setItem(int,int,QVariant)"), "void", 6004, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("currentColumn()"), "int", 6005, ok );
        qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("currentRow()"), "int", 6006, ok );
		qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("checkState(int,int)"), "Qt::CheckState", 6007, ok);
		qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setCheckState(int,int,Qt::CheckState)"), "void", 6008, ok);
		qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("flags(int,int)"), "Qt::ItemFlags", 6009, ok);
		qTableWidget << buildMethodDescription(QMetaObject::normalizedSignature("setFlags(int,int,Qt::ItemFlags)"), "void", 6010, ok);
        methodHash["QTableWidget"] = qTableWidget;

        //QTableView
        MethodDescriptionList qTableView;
        qTableView << buildMethodDescription(QMetaObject::normalizedSignature("horizontalHeader()"), "ito::PythonQObjectMarshal", 7001, ok );
        qTableView << buildMethodDescription(QMetaObject::normalizedSignature("verticalHeader()"), "ito::PythonQObjectMarshal", 7002, ok );
        methodHash["QTableView"] = qTableView;

		//QSplitter
		MethodDescriptionList qSplitter;
		qSplitter << buildMethodDescription(QMetaObject::normalizedSignature("setStretchFactor(int,int)"), "void", 8001, ok);
		qSplitter << buildMethodDescription(QMetaObject::normalizedSignature("sizes()"), "QList<int>", 8002, ok);
		qSplitter << buildMethodDescription(QMetaObject::normalizedSignature("setSizes(QList<int>)"), "void", 8003, ok);
		qSplitter << buildMethodDescription(QMetaObject::normalizedSignature("isCollapsible(int)"), "bool", 8004, ok);
		qSplitter << buildMethodDescription(QMetaObject::normalizedSignature("setCollapsible(int,bool)"), "void", 8005, ok);
		methodHash["QSplitter"] = qSplitter;

        //QStatusBar
        MethodDescriptionList qStatusBar;
        qStatusBar << buildMethodDescription(QMetaObject::normalizedSignature("addLabelWidget(QString)"), "ito::PythonQObjectMarshal", 9001, ok);
        qStatusBar << buildMethodDescription(QMetaObject::normalizedSignature("currentMessage()"), "QString", 9002, ok);
        methodHash["QStatusBar"] = qStatusBar;

        //QToolBar
        MethodDescriptionList qToolBar;
        qToolBar << buildMethodDescription(QMetaObject::normalizedSignature("addSeparator()"), "ito::PythonQObjectMarshal", 10001, ok);
        qToolBar << buildMethodDescription(QMetaObject::normalizedSignature("addAction(QString,QString)"), "ito::PythonQObjectMarshal", 10002, ok);
        methodHash["QToolBar"] = qToolBar;

        //QAction
        MethodDescriptionList qAction;
        qAction << buildMethodDescription(QMetaObject::normalizedSignature("setIcon(QString,double)"), "void", 11001, ok);
        methodHash["QAction"] = qAction;

        //QLayout
        MethodDescriptionList qLayout;
        qLayout << buildMethodDescription(QMetaObject::normalizedSignature("itemAt(int)"), "ito::PythonQObjectMarshal", 12001, ok);
        qLayout << buildMethodDescription(QMetaObject::normalizedSignature("count()"), "int", 12002, ok);
        qLayout << buildMethodDescription(QMetaObject::normalizedSignature("addItem(QString,QString)"), "ito::PythonQObjectMarshal", 12003, ok);
        methodHash["QLayout"] = qLayout;

        //QFormLayout
        MethodDescriptionList qFormLayout;
        qFormLayout << buildMethodDescription(QMetaObject::normalizedSignature("removeRow(int)"), "void", 13001, ok);
        qFormLayout << buildMethodDescription(QMetaObject::normalizedSignature("rowCount()"), "int", 13002, ok);
        methodHash["QFormLayout"] = qFormLayout;

        //QGridLayout
        MethodDescriptionList qGridLayout;
        qGridLayout << buildMethodDescription(QMetaObject::normalizedSignature("itemAtPosition(int,int)"), "ito::PythonQObjectMarshal", 14001, ok);
        qGridLayout << buildMethodDescription(QMetaObject::normalizedSignature("rowCount()"), "int", 14002, ok);
        qGridLayout << buildMethodDescription(QMetaObject::normalizedSignature("columnCount()"), "int", 14003, ok);
        methodHash["QGridLayout"] = qGridLayout;
    }
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
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

//--------------------------------------------------------------------------------------------------------------------------------------------------
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

//--------------------------------------------------------------------------------------------------------------------------------------------------
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
ito::RetVal WidgetWrapper::call(QObject *object, int methodIndex, void **_a)
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

            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QListWidget object is null").toLatin1().data());
            switch(methodIndex)
            {
                case 2001: //addItem
                    object2->addItem((*reinterpret_cast< const QString(*)>(_a[1])));
                    return ito::retOk;
                break;
            
                case 2002: //addItems
                    object2->addItems((*reinterpret_cast< const QStringList(*)>(_a[1])));
                    return ito::retOk;
                break;
            
                case 2003: //selectedRows
                {
                    QVector<int> _r;
                    for (int i = 0; i < object2->count(); ++i)
                    {
                        if (object2->item(i)->isSelected())
                        {
                            _r.append(i);
                        }
                    }
                    (*reinterpret_cast< QVector<int>*>(_a[0])) = _r;
                }
                return ito::retOk;
                break;
            
                case 2004: //selectedTexts
                {
                    QStringList _r;
                    QListWidgetItem *_item;
                    for (int i = 0; i < object2->count(); ++i)
                    {
                        _item = object2->item(i);
                        if (_item->isSelected())
                        {
                            _r.append(_item->text());
                        }
                    }
                    (*reinterpret_cast< QStringList*>(_a[0])) = _r;
                }
                return ito::retOk;
                break;

                case 2005: //selectRows
                {
                    const QVector<int> *vals = reinterpret_cast< const QVector<int>(*)>(_a[1]);
                    QListWidgetItem *_item;
                    for (int i = 0; i < object2->count(); ++i)
                    {
                        _item = object2->item(i);
                        _item->setSelected(vals->contains(i));
                    }
                }
                return ito::retOk;
                break;

                case 2006: //takeItem
                {
                    QListWidgetItem *_r;
                    _r = object2->takeItem((*reinterpret_cast< const int(*)>(_a[1])));
                    (*reinterpret_cast< QString*>(_a[0])) = _r ? _r->text() : QString();
                    if (_r)
                    {
                        (*reinterpret_cast< QString*>(_a[0])) = _r->text();
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;

                case 2007: //item
                {
                    QListWidgetItem *_r;
                    _r = object2->item((*reinterpret_cast< const int(*)>(_a[1])));
                    if (_r)
                    {
                        (*reinterpret_cast< QString*>(_a[0])) = _r->text();
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;

                case 2008: //checkState
                {
                    QListWidgetItem *_r;
                    _r = object2->item((*reinterpret_cast< const int(*)>(_a[1])));
                    if (_r)
                    {
                        (*reinterpret_cast< Qt::CheckState*>(_a[0])) = _r->checkState();
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;

                case 2009: //setCheckState
                {
                    QListWidgetItem *_r;
                    Qt::CheckState state = *reinterpret_cast< const Qt::CheckState(*)>(_a[2]);
                    _r = object2->item((*reinterpret_cast< const int(*)>(_a[1])));
                    if (_r)
                    {
                        _r->setCheckState(state);
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;

                case 2010: //flags
                {
                    QListWidgetItem *_r;
                    _r = object2->item((*reinterpret_cast< const int(*)>(_a[1])));
                    if (_r)
                    {
                        (*reinterpret_cast< Qt::ItemFlags*>(_a[0])) = _r->flags();
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;

                case 2011: //setItemFlags
                {
                    QListWidgetItem *_r;
                    Qt::ItemFlags flags = *reinterpret_cast< const Qt::ItemFlags(*)>(_a[2]);
                    _r = object2->item((*reinterpret_cast< const int(*)>(_a[1])));
                    if (_r)
                    {
                        _r->setFlags(flags);
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;
                
                case 2012: //setItemText
                {
                    QListWidgetItem *_r;
                    QString text = *reinterpret_cast< const QString(*)>(_a[2]);
                    _r = object2->item((*reinterpret_cast< const int(*)>(_a[1])));
                    if (_r)
                    {
                        _r->setText(text);
                    }
                    else
                    {
                        return ito::RetVal(ito::retError, 0, "item in given row does not exist");
                    }
                }
                return ito::retOk;
                break;
            }
        
        }
        else if(QString::compare(className, "QComboBox", Qt::CaseInsensitive) == 0)
        {
            QComboBox *object2 = qobject_cast<QComboBox*>(object);

            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("ComboBox object is null").toLatin1().data());
            switch(methodIndex)
            {
                case 3001: //addItem
                {
                    object2->addItem((*reinterpret_cast< const QString(*)>(_a[1])));
                    //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                    return ito::retOk;
                }
                case 3002: //addItems
                {
                    object2->addItems((*reinterpret_cast< const QStringList(*)>(_a[1])));
                    //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                    return ito::retOk;
                }
                case 3003: //removeItem
                {
                    object2->removeItem((*reinterpret_cast< const int(*)>(_a[1])));
                    //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                    return ito::retOk;
                }
                case 3004: //setItemData
                {
                    object2->setItemData((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const QVariant(*)>(_a[2])), Qt::DisplayRole);
                    //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                    return ito::retOk;
                }
                case 3005: //insertItem
                {
                    object2->insertItem((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const QString(*)>(_a[2])));
                    //*reinterpret_cast< ito::RetVal*>(_a[0]) = _r;
                    return ito::retOk;
                }
                case 3006: //itemText
                {
                    QString _r = object2->itemText((*reinterpret_cast<const int(*)>(_a[1])));
                    (*reinterpret_cast<QString*>(_a[0])) = _r;
                    return ito::retOk;
                }
            }
        }
        else if(QString::compare(className, "QTabWidget", Qt::CaseInsensitive) == 0)
        {
            QTabWidget *object2 = qobject_cast<QTabWidget*>(object);

            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QTabWidget object is null").toLatin1().data());
            switch(methodIndex)
            {
                case 4001: //isTabEnabled
                {
                    bool _r = object2->isTabEnabled((*reinterpret_cast< const int(*)>(_a[1])));
                    (*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
                case 4002: //setTabEnabled
                {
                    object2->setTabEnabled((*reinterpret_cast< const int(*)>(_a[1])),(*reinterpret_cast< const bool(*)>(_a[2])));
                    //(*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
            }
        }
        else if(QString::compare(className, "QMainWindow", Qt::CaseInsensitive) == 0)
        {
            QMainWindow *object2 = qobject_cast<QMainWindow*>(object);

            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QMainWindow object is null").toLatin1().data());
            switch(methodIndex)
            {
                case 5001: //statusBar
                {
                    QWidget* _r = (QWidget*)( object2->statusBar() );
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), _r);
                    return ito::retOk;
                }
                case 5002: //centralWidget
                {
                    QWidget* _r = ( object2->centralWidget() );
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), _r);
                    return ito::retOk;
                }
                case 5003: //addToolBar
                {
                    QToolBar *_r = object2->addToolBar(*reinterpret_cast<QString(*)>(_a[1]));
                    QString objectName = *reinterpret_cast<QString(*)>(_a[2]);
                    if (objectName != "")
                    {
                        _r->setObjectName(objectName);
                    }
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), _r);
                    return ito::retOk;
                }
            }
        }
        else if(QString::compare(className, "QWidget", Qt::CaseInsensitive) == 0)
        {
            QWidget *object2 = qobject_cast<QWidget*>(object);

            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QWidget object is null").toLatin1().data());
            switch(methodIndex)
            {
                case 1001: //resize
                {
                    object2->resize((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])));
                    //(*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
                case 1002: //setGeometry
                {
                    object2->setGeometry((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])), (*reinterpret_cast< const int(*)>(_a[3])), (*reinterpret_cast< const int(*)>(_a[4])));
                    //(*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
                case 1003: //setCursor
                {
                    int c = (*reinterpret_cast<const int(*)>(_a[1]));
                    if (c < 0)
                    {
                        object2->unsetCursor();
                    }
                    else
                    {
                        QCursor cursor((Qt::CursorShape)(c));
                        object2->setCursor(cursor);
                    }
                    //(*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
                case 1004: //devicePixelRatioF
                {
#if QT_VERSION >= 0x050600
                    (*reinterpret_cast< float*>(_a[0])) = object2->devicePixelRatioF();
#else
                    (*reinterpret_cast<float*>(_a[0])) = 1.0;
#endif
                    return ito::retOk;
                }
            }
        }
        else if(QString::compare(className, "QTableWidget", Qt::CaseInsensitive) == 0)
        {
            QTableWidget *object2 = qobject_cast<QTableWidget*>(object);

            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QTableWidget object is null").toLatin1().data());
            switch(methodIndex)
            {
                case 6001: //setHorizontalHeaderLabels
                {
                    object2->setHorizontalHeaderLabels( (*reinterpret_cast< const QStringList(*)>(_a[1])) );
                    //(*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
                case 6002: //setVerticalHeaderLabels
                {
                    object2->setVerticalHeaderLabels( (*reinterpret_cast< const QStringList(*)>(_a[1])) );
                    //(*reinterpret_cast< bool*>(_a[0])) = _r;
                    return ito::retOk;
                }
                case 6003: //getItem
                {
                    int row = (*reinterpret_cast< const int(*)>(_a[1]));
                    int col = (*reinterpret_cast< const int(*)>(_a[2]));
                    QTableWidgetItem *item = object2->item(row,col);

                    if(item)
                    {
                        (*reinterpret_cast<QVariant*>(_a[0])) = item->data(Qt::DisplayRole);
                        return ito::retOk;
                    }

                    return ito::RetVal(ito::retError, 0, QObject::tr("Could not access row / col, maybe out of range").toLatin1().data());
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
                    return ito::retOk;
                }
                case 6005: //currentColumn
                {
                    (*reinterpret_cast<int*>(_a[0])) = object2->currentColumn();
                    return ito::retOk;
                }
                case 6006: //currentRow
                {
                    (*reinterpret_cast<int*>(_a[0])) = object2->currentRow();
                    return ito::retOk;
                }

				case 6007: //checkState
				{
					QTableWidgetItem *_r;
					_r = object2->item((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])));
					if (_r)
					{
						(*reinterpret_cast< Qt::CheckState*>(_a[0])) = _r->checkState();
					}
					else
					{
						return ito::RetVal(ito::retError, 0, "item in given row does not exist");
					}
				}
				return ito::retOk;
				break;

				case 6008: //setCheckState
				{
					QTableWidgetItem *_r;
					Qt::CheckState state = *reinterpret_cast< const Qt::CheckState(*)>(_a[3]);
					_r = object2->item((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])));
					if (_r)
					{
						_r->setCheckState(state);
					}
					else
					{
						return ito::RetVal(ito::retError, 0, "item in given row does not exist");
					}
				}
				return ito::retOk;
				break;

				case 6009: //flags
				{
					QTableWidgetItem *_r;
					_r = object2->item((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])));
					if (_r)
					{
						(*reinterpret_cast< Qt::ItemFlags*>(_a[0])) = _r->flags();
					}
					else
					{
						return ito::RetVal(ito::retError, 0, "item in given row does not exist");
					}
				}
				return ito::retOk;
				break;

				case 6010: //setItemFlags
				{
					QTableWidgetItem *_r;
					Qt::ItemFlags flags = *reinterpret_cast< const Qt::ItemFlags(*)>(_a[3]);
					_r = object2->item((*reinterpret_cast< const int(*)>(_a[1])), (*reinterpret_cast< const int(*)>(_a[2])));
					if (_r)
					{
						_r->setFlags(flags);
					}
					else
					{
						return ito::RetVal(ito::retError, 0, "item in given row does not exist");
					}
				}
				return ito::retOk;
            }
        }
        else if(QString::compare(className, "QTableView", Qt::CaseInsensitive) == 0)
        {
            QTableView *object2 = qobject_cast<QTableView*>(object);
            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QTableView object is null").toLatin1().data());

            switch(methodIndex)
            {
                case 7001: //horizontalHeader
                {
                    QHeaderView* _r = (QHeaderView*)( object2->horizontalHeader() );
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), _r);
                    return ito::retOk;
                }
                case 7002: //verticalHeader
                {
                    QHeaderView* _r = (QHeaderView*)( object2->verticalHeader() );
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(_r->objectName().toLatin1(), _r->metaObject()->className(), _r);
                    return ito::retOk;
                }
            }
        }
		else if(QString::compare(className, "QSplitter", Qt::CaseInsensitive) == 0)
        {
            QSplitter *object2 = qobject_cast<QSplitter*>(object);
            if(object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QTableView object is null").toLatin1().data());

            switch(methodIndex)
            {
                case 8001: //setStretchFactor
                {
					object2->setStretchFactor((*reinterpret_cast< int(*)>(_a[1])), (*reinterpret_cast< int(*)>(_a[2])));
					//(*reinterpret_cast< bool*>(_a[0])) = _r;
					return ito::retOk;
                }
				case 8002: //sizes
				{
					QList<int> _r = object2->sizes();
					(*reinterpret_cast< QList<int>*>(_a[0])) = _r;
					return ito::retOk;
				}
				case 8003: //setSizes
				{
					object2->setSizes(*reinterpret_cast< QList<int>(*)>(_a[1]));
					//(*reinterpret_cast< bool*>(_a[0])) = _r;
					return ito::retOk;
				}
				case 8004: //isCollapsible
				{
					bool _r = object2->isCollapsible(*reinterpret_cast< int(*)>(_a[1]));
					(*reinterpret_cast< bool*>(_a[0])) = _r;
					return ito::retOk;
				}
				case 8005: //setCollapsible
				{
					object2->setCollapsible((*reinterpret_cast< int(*)>(_a[1])), (*reinterpret_cast< bool(*)>(_a[2])));
					//(*reinterpret_cast< bool*>(_a[0])) = _r;
					return ito::retOk;
				}
            }
        }
        else if (QString::compare(className, "QStatusBar", Qt::CaseInsensitive) == 0)
        {
            QStatusBar *object2 = qobject_cast<QStatusBar*>(object);
            if (object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QStatusBar object is null").toLatin1().data());

            switch (methodIndex)
            {
                case 9001: //addLabelWidget
                {
                    QLabel *lbl = new QLabel(object2);
                    lbl->setObjectName(*reinterpret_cast<QString(*)>(_a[1]));
                    object2->addWidget(lbl);
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(lbl->objectName().toLatin1(), lbl->metaObject()->className(), lbl);
                    return ito::retOk;
                }
                case 9002: //currentMessage
                {
                    QString _r = object2->currentMessage();
                    (*reinterpret_cast<QString*>(_a[0])) = _r;
                    return ito::retOk;
                }
            }
        }
        else if (QString::compare(className, "QToolBar", Qt::CaseInsensitive) == 0)
        {
            QToolBar *object2 = qobject_cast<QToolBar*>(object);
            if (object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QToolBar object is null").toLatin1().data());

            switch (methodIndex)
            {
                case 10001: //addSeparator
                {
                    QAction* a = object2->addSeparator();
                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(a->objectName().toLatin1(), a->metaObject()->className(), a);
                    return ito::retOk;
                }
                case 10002: //addAction
                {
                    QString text = *reinterpret_cast<QString(*)>(_a[1]);
                    QAction *a = object2->addAction(text);

                    QString objectName = *reinterpret_cast<QString(*)>(_a[2]);
                    if (objectName != "")
                    {
                        a->setObjectName(objectName);
                    }

                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(a->objectName().toLatin1(), a->metaObject()->className(), a);
                    return ito::retOk;
                }
            }
        }
        else if (QString::compare(className, "QAction", Qt::CaseInsensitive) == 0)
        {
            QAction *object2 = qobject_cast<QAction*>(object);
            if (object2 == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QAction object is null").toLatin1().data());

            switch (methodIndex)
            {
                case 11001: //setIcon
                {
                    QPixmap pm(*reinterpret_cast<QString(*)>(_a[1]));
                    pm.setDevicePixelRatio(*reinterpret_cast<double(*)>(_a[2]));
                    object2->setIcon(QIcon(pm));
                    return ito::retOk;
                }
            }
        }
        else if (QString::compare(className, "QLayout", Qt::CaseInsensitive) == 0)
        {
            QLayout *layout = qobject_cast<QLayout*>(object);
            if (layout == NULL)
            {
                return ito::RetVal(ito::retError, 0, QObject::tr("QLayout object is null").toLatin1().data());
            }

            switch (methodIndex)
            {
                case 12001: //itemAt
                {
                    int index = *reinterpret_cast<int(*)>(_a[1]);

                    if (index < 0 || index >= layout->count())
                    {
                        return ito::RetVal::format(ito::retError, 0, "index exceeds the valid range [0, %i]", layout->count() - 1);
                    }

                    QLayoutItem *item = layout->itemAt(index);

                    if (item == NULL)
                    {
                        return ito::RetVal::format(ito::retError, 0, "Layout has no item at index %i", index);
                    }
                
                    QObject *layoutItem = item->widget();

                    if (layoutItem == NULL)
                    {
                        layoutItem = item->layout();
                    }

                    if (layoutItem == NULL)
                    {
                        return ito::RetVal::format(ito::retError, 0, "Layout has no item at index %i", index);
                    }

                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(layoutItem);
                    return ito::retOk;
                }
                case 12002: //count
                {
                    (*reinterpret_cast<int*>(_a[0])) = layout->count();
                    return ito::retOk;
                }
                case 12003: // addItem(QString,QString) -> ito::PythonQObjectMarshal
                {
                    QString className = *reinterpret_cast<QString(*)>(_a[1]);
                    QString objectName = *reinterpret_cast<QString(*)>(_a[2]);
                    ito::RetVal retValue;

                    if (objectName == "")
                    {
                        objectName = QString();
                    }

                    QWidget *widget = m_pUiOrganizer->loadDesignerPluginWidget(
                        className,
                        retValue,
                        ito::AbstractFigure::WindowMode::ModeInItomFigure,
                        layout->parentWidget());
                    
                    if (!retValue.containsError())
                    {
                        if (objectName != "")
                        {
                            widget->setObjectName(objectName);
                        }

                        layout->addWidget(widget);

                        (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(widget);
                    }

                    return retValue;
                }
            }
        }
        else if (QString::compare(className, "QFormLayout", Qt::CaseInsensitive) == 0)
        {
            QFormLayout *layout = qobject_cast<QFormLayout*>(object);
            if (layout == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QLayout object is null").toLatin1().data());

            switch (methodIndex)
            {
                case 13001: //removeRow
                {
                    int row = *reinterpret_cast<int(*)>(_a[1]);

                    if (row < 0 || row >= layout->rowCount())
                    {
                        return ito::RetVal::format(ito::retError, 0, "row exceeds the valid range [0, %i]", layout->rowCount() - 1);
                    }

                    layout->removeRow(row);

                    return ito::retOk;
                }
                case 13002: //rowCount
                {
                    (*reinterpret_cast<int*>(_a[0])) = layout->rowCount();
                    return ito::retOk;
                }
            }
        }
        else if (QString::compare(className, "QGridLayout", Qt::CaseInsensitive) == 0)
        {
            QGridLayout *layout = qobject_cast<QGridLayout*>(object);
            if (layout == NULL) return ito::RetVal(ito::retError, 0, QObject::tr("QLayout object is null").toLatin1().data());

            switch (methodIndex)
            {
                case 14001: //itemAtPosition
                {
                    int row = *reinterpret_cast<int(*)>(_a[1]);
                    int column = *reinterpret_cast<int(*)>(_a[2]);

                    if (row < 0 || row >= layout->rowCount())
                    {
                        return ito::RetVal::format(ito::retError, 0, "row exceeds the valid range [0, %i]", layout->rowCount() - 1);
                    }

                    if (column < 0 || column >= layout->columnCount())
                    {
                        return ito::RetVal::format(ito::retError, 0, "row exceeds the valid range [0, %i]", layout->columnCount() - 1);
                    }

                    QLayoutItem *item = layout->itemAtPosition(row, column);

                    if (item == NULL)
                    {
                        return ito::RetVal::format(ito::retError, 0, "Layout has no item at row %i and column %i", row, column);
                    }

                    QObject *layoutItem = item->widget();

                    if (layoutItem == NULL)
                    {
                        layoutItem = item->layout();
                    }

                    if (layoutItem == NULL)
                    {
                        return ito::RetVal::format(ito::retError, 0, "Layout has no item at row %i and column %i", row, column);
                    }

                    (*reinterpret_cast<ito::PythonQObjectMarshal*>(_a[0])) = ito::PythonQObjectMarshal(layoutItem);

                    return ito::retOk;
                }
                case 14002: //rowCount
                {
                    (*reinterpret_cast<int*>(_a[0])) = layout->rowCount();
                    return ito::retOk;
                }
                case 14003: //columnCount
                {
                    (*reinterpret_cast<int*>(_a[0])) = layout->columnCount();
                    return ito::retOk;
                }
            }
        }

        //nothing found yet, check superclass
        tempMetaObject = tempMetaObject->superClass();
    }


    return ito::RetVal(ito::retError, 0, QObject::tr("Slot or widget not found").toLatin1().data());
}

//--------------------------------------------------------------------------------------------------------------------------------------------------
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
