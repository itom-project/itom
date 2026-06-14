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

#ifndef PYTHONUI_H
#define PYTHONUI_H

#include "pythonCommon.h"
#include "pythonQtConversion.h"
#include "pythonQtSignalMapper.h"
#include "pythonItomMetaObject.h"

#include <qstring.h>
#include <qvariant.h>
#include <qobject.h>
#include <qhash.h>

namespace ito
{

class PythonUi
{
public:

    //#################################################################################################
    // UiItem
    //#################################################################################################

    //-------------------------------------------------------------------------------------------------
    // typedefs
    //-------------------------------------------------------------------------------------------------
    typedef struct
    {
        PyObject_HEAD
        PyObject *baseItem; //parent UiItem (e.g. the whole dialog), NULL if no parent.
        char* objName;          //object name of corresponding widget in UI
        char* widgetClassName;  //class name of corresponding widget in UI
        unsigned int objectID;  //itom internal ID of hashed meta object of this widget (for communication with uiOrganizer)
        const MethodDescriptionList* methodList; //borrowed pointer to an item in methodDescriptionListStorage.
        PyObject *weakreflist;
    }
    PyUiItem;

    typedef struct
    {
        PyUiItem uiItem;
        int uiHandle;
        int winType;
        int buttonBarType;
        bool childOfMainWindow;
        bool deleteOnClose;
        PyObject *dialogButtons;
        char* filename;
        PythonQtSignalMapper *signalMapper;
    }
    PyUi;


    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyUiItem_dealloc(PyUiItem *self);
    static PyObject *PyUiItem_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyUiItem_init(PyUiItem *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // mapping members
    //-------------------------------------------------------------------------------------------------
    static int PyUiItem_mappingLength(PyUiItem* self);
    static PyObject* PyUiItem_mappingGetElem(PyUiItem* self, PyObject* key);
    static int PyUiItem_mappingSetElem(PyUiItem* self, PyObject* key, PyObject* value);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyUiItem_repr(PyUiItem *self);
    static PyObject* PyUiItem_getClassName(PyUiItem *self);
    static PyObject* PyUiItem_call(PyUiItem *self, PyObject* args);
    static PyObject* PyUiItem_connect(PyUiItem *self, PyObject* args, PyObject *kwds);
    static PyObject* PyUiItem_connectKeyboardInterrupt(PyUiItem *self, PyObject* args, PyObject *kwds);
    static PyObject* PyUiItem_connectProgressObserverInterrupt(PyUiItem *self, PyObject* args, PyObject *kwds);
    static PyObject* PyUiItem_disconnect(PyUiItem *self, PyObject* args, PyObject *kwds);
    static PyObject* PyUiItem_getProperties(PyUiItem *self, PyObject *args);
    static PyObject* PyUiItem_setProperties(PyUiItem *self, PyObject *args);
    static PyObject *PyUiItem_getPropertyInfo(PyUiItem *self, PyObject *args);
    static PyObject* PyUiItem_getattro(PyUiItem *self, PyObject *name);
    static int       PyUiItem_setattro(PyUiItem *self, PyObject *name, PyObject *value);

    static PyObject* PyUiItem_setAttribute(PyUiItem *self, PyObject *args);
    static PyObject* PyUiItem_getAttribute(PyUiItem *self, PyObject *args);

    static PyObject* PyUiItem_setWindowFlags(PyUiItem *self, PyObject *args);
    static PyObject* PyUiItem_getWindowFlags(PyUiItem *self);

    static PyObject* PyUiItem_info(PyUiItem *self, PyObject *args);
    static PyObject* PyUiItem_exists(PyUiItem *self);
    static PyObject* PyUiItem_children(PyUiItem *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiItem_getChild(PyUiItem *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUiItem_getLayout(PyUiItem *self);

    //-------------------------------------------------------------------------------------------------
    // helpers
    //-------------------------------------------------------------------------------------------------
    static bool loadMethodDescriptionList(PyUiItem *self);
    static PythonQtSignalMapper* PyUiItem_getTopLevelSignalMapper(PyUiItem *self);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    static PyMemberDef  PyUiItem_members[];
    static PyGetSetDef  PyUiItem_getseters[];
    static PyMethodDef  PyUiItem_methods[];
    static PyTypeObject PyUiItemType;
    static PyModuleDef  PyUiItemModule;
    static PyMappingMethods PyUiItem_mappingProtocol;
    static void PyUiItem_addTpDict(PyObject *tp_dict);



    //#################################################################################################
    // Ui
    //#################################################################################################


    //-------------------------------------------------------------------------------------------------
    // constructor, deconstructor, alloc, dellaoc
    //-------------------------------------------------------------------------------------------------
    static void PyUi_dealloc(PyUi *self);
    static PyObject *PyUi_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
    static int PyUi_init(PyUi *self, PyObject *args, PyObject *kwds);

    //-------------------------------------------------------------------------------------------------
    // general members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyUi_repr(PyUi *self);

    static PyObject *PyUi_show(PyUi *self, PyObject *args);
    static PyObject *PyUi_hide(PyUi *self);
    static PyObject *PyUi_isVisible(PyUi *self);

    static PyObject *PyUi_getPropertyInfo(PyUi *self, PyObject *args);
    static PyObject *PyUi_getProperties(PyUi *self, PyObject *args);
    static PyObject *PyUi_setProperties(PyUi *self, PyObject *args);

    //-------------------------------------------------------------------------------------------------
    // static members
    //-------------------------------------------------------------------------------------------------
    static PyObject* PyUi_getDouble(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_getInt(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_getItem(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_getText(PyUi *self, PyObject *args, PyObject *kwds);

    static PyObject* PyUi_msgInformation(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_msgQuestion(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_msgWarning(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_msgCritical(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_msgGeneral(PyUi *self, PyObject *args, PyObject *kwds, int type);

    static PyObject* PyUi_getExistingDirectory(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_getOpenFileNames(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_getOpenFileName(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_getSaveFileName(PyUi *self, PyObject *args, PyObject *kwds);

    static PyObject* PyUi_createNewAlgoWidget(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject* PyUi_createNewAlgoWidget2(PyUi *self, PyObject *args, PyObject *kwds);
    static PyObject *PyUi_availableWidgets(PyUi *self);

    //-------------------------------------------------------------------------------------------------
    // type structures
    //-------------------------------------------------------------------------------------------------
    static PyGetSetDef  PyUi_getseters[];
    static PyMemberDef  PyUi_members[];
    static PyMethodDef  PyUi_methods[];
    static PyTypeObject PyUiType;
    static PyModuleDef  PyUiModule;
    static void PyUi_addTpDict(PyObject *tp_dict);

    //-------------------------------------------------------------------------------------------------
    // macros
    //-------------------------------------------------------------------------------------------------
    #define PyUiItem_Check(op) PyObject_TypeCheck(op, &ito::PythonUi::PyUiItemType)
    #define PyUi_Check(op) PyObject_TypeCheck(op, &ito::PythonUi::PyUiType)

private:
    static QHash<QByteArray, QSharedPointer<ito::MethodDescriptionList> > methodDescriptionListStorage; //key is a widget-className, every PyUiItem which needs a methodDescriptionList gets it from this storage or if not available from UiOrganizer and puts it then to this storage
};

}; //end namespace ito


#endif
