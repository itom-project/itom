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

#include "pythonWorkspace.h"
#include "../../common/sharedStructures.h"

#include "pythonDataObject.h"
#include "pythonPCL.h"
#include "pythonPlugins.h"
#include "pythonQtConversion.h"

namespace ito {

//-----------------------------------------------------------------------------------------------------------
PyWorkspaceItem::~PyWorkspaceItem()
{
    foreach (const PyWorkspaceItem* child, m_childs)
    {
        delete child;
    }

    m_childs.clear();
}

//-----------------------------------------------------------------------------------------------------------
PyWorkspaceItem::PyWorkspaceItem(const PyWorkspaceItem& other)
{
    m_name = other.m_name;
    m_key = other.m_key;
    m_type = other.m_type;
    m_value = other.m_value;
    m_extendedValue = other.m_extendedValue;
    m_childState = other.m_childState;
    m_exist = other.m_exist;
    m_childs = other.m_childs;
    m_isarrayelement = other.m_isarrayelement;
    m_compatibleParamBaseType = other.m_compatibleParamBaseType;
}

/*!< delimiter between the parent and child(ren) item
of the full path to a python variable. */
QChar PyWorkspaceContainer::delimiter = QChar(0x1C, 0x00); // '/';

//-----------------------------------------------------------------------------------------------------------
PyWorkspaceContainer::PyWorkspaceContainer(bool globalNotLocal) :
    m_globalNotLocal(globalNotLocal),
    m_dictUnicode(nullptr),
    m_slotsUnicode(nullptr),
    m_mroUnicode(nullptr)
{
    int i = 0;
}

//-----------------------------------------------------------------------------------------------------------
PyWorkspaceContainer::~PyWorkspaceContainer()
{
    Py_XDECREF(m_dictUnicode);
    Py_XDECREF(m_slotsUnicode);
    Py_XDECREF(m_mroUnicode);
}

//-----------------------------------------------------------------------------------------------------------
void PyWorkspaceContainer::initUnicodeConstants()
{
    if (!m_dictUnicode)
    {
        m_dictUnicode = PyUnicode_FromString("__dict__");
        m_slotsUnicode = PyUnicode_FromString("__slots__");
        m_mroUnicode = PyUnicode_FromString("__mro__");
    }
}

//-----------------------------------------------------------------------------------------------------------
bool PyWorkspaceContainer::isNotInBlacklist(PyObject* obj) const
{
    return !(
        PyFunction_Check(obj) || PyMethod_Check(obj) || PyType_Check(obj) || PyModule_Check(obj) ||
        PyCFunction_Check(obj));
}

//-----------------------------------------------------------------------------------------------------------
void PyWorkspaceContainer::clear()
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    loadDictionary(nullptr, "");
    PyGILState_Release(gstate);
}

//-----------------------------------------------------------------------------------------------------------
// Python GIL must be locked when calling this function!
void PyWorkspaceContainer::loadDictionary(PyObject* obj, const QString& fullNameParentItem)
{
#if defined _DEBUG
    if (!PyGILState_Check())
    {
        std::cerr << "Python GIL must be locked when calling loadDictionaryRec\n" << std::endl;
        return;
    }
#endif

    QStringList deleteList;

    if (fullNameParentItem == "")
    {
        loadDictionaryRec(obj, "", &m_rootItem, deleteList);
        emit updateAvailable(&m_rootItem, fullNameParentItem, deleteList);
    }
    else
    {
        QStringList nameSplit = fullNameParentItem.split(ito::PyWorkspaceContainer::delimiter);

        if (nameSplit[0] == "")
        {
            nameSplit.removeFirst();
        }

        PyWorkspaceItem* parent = &m_rootItem;
        QHash<QString, ito::PyWorkspaceItem*>::iterator it;

        while (nameSplit.count() > 0)
        {
            it = parent->m_childs.find(nameSplit.takeFirst());

            if (it != parent->m_childs.end())
            {
                parent = *it;
            }
            else
            {
                return;
            }
        }

        loadDictionaryRec(obj, fullNameParentItem, parent, deleteList);

        emit updateAvailable(parent, fullNameParentItem, deleteList);
    }
}

//-------------------------------------------------------------------------------------
void PyWorkspaceContainer::appendSlotNamesToList(PyObject* objOrType, PyObject* slotNamesList)
{
    //__slots__ can return any sequence, here list and tuple are supported.
    PyObject* subitem = nullptr;
    PyObject* slotNames =
        PyObject_GetAttr(objOrType, m_slotsUnicode); // new ref (list, tuple or string)

    if (slotNames)
    {
        if (PyList_Check(slotNames))
        {
            for (Py_ssize_t idx = 0; idx < PyList_Size(slotNames); ++idx)
            {
                PyList_Append(slotNamesList, PyList_GET_ITEM(slotNames, idx));
            }
        }
        else if (PyTuple_Check(slotNames))
        {
            for (Py_ssize_t idx = 0; idx < PyTuple_Size(slotNames); ++idx)
            {
                PyList_Append(slotNamesList, PyTuple_GET_ITEM(slotNames, idx));
            }
        }
        else if ((PyUnicode_Check(slotNames) || PyBytes_Check(slotNames)))
        {
            PyList_Append(slotNamesList, slotNames);
        }
    }

    Py_XDECREF(slotNames);
    slotNames = nullptr;
}

//-----------------------------------------------------------------------------------------------------------
void PyWorkspaceContainer::loadDictionaryRec(
    PyObject* obj,
    const QString& fullNameParentItem,
    PyWorkspaceItem* parentItem,
    QStringList& deletedKeys)
{
#if defined _DEBUG
    if (!PyGILState_Check())
    {
        std::cerr << "Python GIL must be locked when calling loadDictionaryRec\n" << std::endl;
        return;
    }
#endif

    if (!m_dictUnicode)
    {
        initUnicodeConstants();
    }

    // To call this method, the Python GIL must already be locked!
    PyObject* keys = nullptr;
    PyObject* values = nullptr;
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    QHash<QString, PyWorkspaceItem*>::iterator it;
    Py_ssize_t i;
    QString keyText;
    QString keyKey;
    PyObject* keyUTF8String = nullptr;
    PyWorkspaceItem* actItem;
    QString fullName;
    char keyType[] = {0, 0};

    // at first, set status of all childs of parentItem to "not-existing"
    it = parentItem->m_childs.begin();

    while (it != parentItem->m_childs.end())
    {
        (*it)->m_exist = false;
        ++it;
    }

    if (Py_IsInitialized() && obj != nullptr)
    {
        // was PySequence_Check(obj) before, however a class can also
        // implement the sequence protocol
        if (PyTuple_Check(obj) || PyList_Check(obj))
        {
            for (i = 0; i < PySequence_Size(obj); i++)
            {
                value = PySequence_GetItem(obj, i); // new reference

                if (isNotInBlacklist(value))
                {
                    // only if not on blacklist

                    keyText = QString::number(i);
                    keyKey = "xx:" + keyText; // list + number
                    keyKey[0] = PY_LIST_TUPLE;
                    keyKey[1] = PY_NUMBER;

                    it = parentItem->m_childs.find(keyKey);

                    if (it == parentItem->m_childs.end()) // not existing yet
                    {
                        actItem = new PyWorkspaceItem();
                        actItem->m_name = keyText;
                        actItem->m_key = keyKey;
                        actItem->m_exist = true;
                        actItem->m_isarrayelement = true;
                        fullName = fullNameParentItem + ito::PyWorkspaceContainer::delimiter +
                            actItem->m_key;
                        parseSinglePyObject(actItem, value, fullName, deletedKeys);

                        if (m_expandedFullNames.contains(fullName))
                        {
                            // load subtree
                            loadDictionaryRec(value, fullName, actItem, deletedKeys);
                        }

                        parentItem->m_childs.insert(keyKey, actItem);
                    }
                    else // item with this name already exists
                    {
                        actItem = *it;
                        actItem->m_name = keyText;
                        actItem->m_exist = true;
                        actItem->m_isarrayelement = true;
                        fullName = fullNameParentItem + ito::PyWorkspaceContainer::delimiter +
                            actItem->m_key;
                        parseSinglePyObject(actItem, value, fullName, deletedKeys);

                        if (m_expandedFullNames.contains(fullName))
                        {
                            // load subtree
                            loadDictionaryRec(value, fullName, actItem, deletedKeys);
                        }
                    }
                }

                Py_DECREF(value);
            }
        }
        else
        {
            if (PyDict_Check(obj))
            {
                keys = PyDict_Keys(obj); // new ref
                values = PyDict_Values(obj); // new ref

                if (PyErr_Occurred())
                {
                    PyErr_Clear();
                }

                keyType[0] = PY_DICT;
            }
            else if (PyMapping_Check(obj) && PyMapping_Size(obj) > 0)
            {
                keys = PyMapping_Keys(obj); // new ref
                values = PyMapping_Values(obj); // new ref

                if (PyErr_Occurred())
                {
                    // maybe a dataObject... implements the mapping
                    // protocol, but does not have keys and values.
                    PyErr_Clear();
                }

                keyType[0] = PY_MAPPING;
            }
            else if (PyObject_HasAttr(obj, m_dictUnicode) || PyObject_HasAttr(obj, m_slotsUnicode))
            {
                // get the dict, containing all values from object and its bases classes
                PyObject* subdict = PyObject_GetAttr(obj, m_dictUnicode); // new ref

                if (subdict)
                {
                    keys = PyDict_Keys(subdict); // new ref (list)
                    values = PyDict_Values(subdict); // new ref (list)
                    Py_DECREF(subdict);

                    if (PyErr_Occurred())
                    {
                        PyErr_Clear();
                    }
                }
                else
                {
                    keys = PyList_New(0);
                    values = PyList_New(0);
                }

                // get all slots (here, we have to go through all base classes)
                PyObject* slotNames = PyList_New(0);
                PyObject* thisType = PyObject_Type(obj);
                PyObject* mro = PyObject_GetAttr(thisType, m_mroUnicode);

                if (mro)
                {
                    for (Py_ssize_t idx = 0; idx < PyTuple_Size(mro); ++idx)
                    {
                        appendSlotNamesToList(PyTuple_GET_ITEM(mro, idx), slotNames);
                    }
                }

                Py_XDECREF(mro);
                Py_XDECREF(thisType);

                keyType[0] = PY_ATTR;

                if (PyErr_Occurred())
                {
                    PyErr_Clear();
                }

                PyObject* subitem = nullptr;
                PyObject* name = nullptr;

                for (Py_ssize_t idx = 0; idx < PyList_GET_SIZE(slotNames); ++idx)
                {
                    name = PyList_GET_ITEM(slotNames, idx); // borrowed
                    subitem = PyObject_GetAttr(obj, name); // new ref

                    if (subitem)
                    {
                        PyList_Append(keys, name); // does not steal a ref
                        PyList_Append(values, subitem); // does not steal a ref
                        Py_DECREF(subitem);
                        subitem = nullptr;
                    }
                    else
                    {
                        // this slot is not available in this object (name contained in __slots__,
                        // but attribute does not exist)
                        qDebug() << "error parsing attribute of PyObject";
                    }
                }

                if (PyErr_Occurred())
                {
                    PyErr_Clear();
                }

                Py_DECREF(slotNames);
            }

            if (keys && values)
            {
                int overflow;

                for (i = 0; i < PyList_Size(values); i++)
                {
                    value = PyList_GET_ITEM(values, i); // borrowed
                    key = PyList_GET_ITEM(keys, i); // borrowed

                    if (isNotInBlacklist(value)) // only if not on blacklist
                    {
                        keyUTF8String = PyUnicode_AsUTF8String(key); // new

                        if (keyUTF8String)
                        {
                            // borrowed reference to
                            // char-pointer in keyUTF8String
                            keyText = PyBytes_AS_STRING(keyUTF8String);
                            keyKey = "xx:" + keyText;
                            keyKey[0] = keyType[0];
                            keyKey[1] = PY_STRING;
                        }
                        else
                        {
                            PyErr_Clear();

                            if (PyLong_Check(key))
                            {
                                keyText = QString::number(PyLong_AsLongAndOverflow(key, &overflow));
                                if (overflow)
                                {
                                    keyText = QString::number(PyLong_AsLongLong(key));
                                }
                                keyKey = "xx:" + keyText;
                                keyKey[0] = keyType[0];
                                keyKey[1] = PY_NUMBER;
                            }
                            else
                            {
                                // store the pointer of the key object as hex number
                                quintptr objId = reinterpret_cast<quintptr>(key);
                                keyKey = "xx:" + QString::number(objId, 16);
                                keyKey[0] = keyType[0];
                                keyKey[1] = PY_OBJID;
                                keyText = PythonQtConversion::PyObjGetRepresentation(key);
                            }
                        }

                        it = parentItem->m_childs.find(keyKey);

                        if (it == parentItem->m_childs.end()) // not existing yet
                        {
                            actItem = new PyWorkspaceItem();
                            actItem->m_key = keyKey;
                            actItem->m_name = keyText;
                            actItem->m_exist = true;
                            actItem->m_isarrayelement = true;
                            fullName = fullNameParentItem + ito::PyWorkspaceContainer::delimiter +
                                actItem->m_key;
                            parseSinglePyObject(actItem, value, fullName, deletedKeys);

                            if (m_expandedFullNames.contains(fullName))
                            {
                                // load subtree
                                loadDictionaryRec(value, fullName, actItem, deletedKeys);
                            }

                            parentItem->m_childs.insert(keyKey, actItem);
                        }
                        else // item with this name already exists
                        {
                            actItem = *it;
                            actItem->m_key = keyKey;
                            actItem->m_exist = true;
                            actItem->m_isarrayelement = true;
                            fullName = fullNameParentItem + ito::PyWorkspaceContainer::delimiter +
                                actItem->m_key;
                            parseSinglePyObject(actItem, value, fullName, deletedKeys);

                            if (m_expandedFullNames.contains(fullName))
                            {
                                // load subtree
                                loadDictionaryRec(value, fullName, actItem, deletedKeys);
                            }
                        }

                        Py_XDECREF(keyUTF8String);
                    }
                }
            }
            else
            {
                // this item does not seem to contain any children any more. Maybe it was a
                // container item before and now it was directly replaced by a non-container item.
                // Then, the current item cannot be expanded any more:
                if (m_expandedFullNames.contains(fullNameParentItem))
                {
                    m_expandedFullNames.remove(fullNameParentItem);
                }
            }

            Py_XDECREF(keys);
            Py_XDECREF(values);
            keys = nullptr;
            values = nullptr;
        }
    }

    it = parentItem->m_childs.begin();

    while (it != parentItem->m_childs.end())
    {
        if ((*it)->m_exist == false)
        {
            deletedKeys << fullNameParentItem + ito::PyWorkspaceContainer::delimiter + (*it)->m_key;
            delete (*it);
            it = parentItem->m_childs.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (PyErr_Occurred())
    {
        PyErr_PrintEx(0);
    }
}

//-----------------------------------------------------------------------------------------------------------
void PyWorkspaceContainer::parseSinglePyObject(
    PyWorkspaceItem* item, PyObject* value, const QString& fullName, QStringList& deletedKeys)
{
    // To call this method, the Python GIL must already be locked!
    Py_ssize_t size;
    bool expandableType = false;

    if (!m_dictUnicode)
    {
        initUnicodeConstants();
    }

    // check new value
    item->m_exist = true;
    item->m_type = value->ob_type->tp_name;

    // at first check for possible types which have children (dict,list,tuple) or its subtypes
    if (PyDict_Check(value))
    {
        size = PyDict_Size(value);
        item->m_value = QString("[%1 element(s)]").arg(size);
        expandableType = true;
        item->m_extendedValue = "";
        item->m_compatibleParamBaseType = 0; // not compatible
    }
    else if (PyList_Check(value))
    {
        size = PyList_Size(value);
        item->m_value = QString("[%1 element(s)]").arg(size);
        expandableType = true;
        item->m_extendedValue = "";
        item->m_compatibleParamBaseType = 0; // not compatible
    }
    else if (PyTuple_Check(value))
    {
        size = PyTuple_Size(value);
        item->m_value = QString("[%1 element(s)]").arg(size);
        expandableType = true;
        item->m_extendedValue = "";
        item->m_compatibleParamBaseType = 0; // not compatible
    }
    else if (PyObject_HasAttr(value, m_dictUnicode) || PyObject_HasAttr(value, m_slotsUnicode))
    {
        // user-defined class (has attr '__dict__' or '__slots__')
        expandableType = true;
        item->m_compatibleParamBaseType = 0; // not compatible

        // TODO: increase speed
        PyObject* repr = PyObject_Repr(value);

        if (repr == nullptr)
        {
            PyErr_Clear();
            item->m_extendedValue = item->m_value = "<error during call of repr()>";
        }
        else if (PyUnicode_Check(repr))
        {
            PyObject* encodedByteArray = PyUnicode_AsUTF8String(repr);

            if (!encodedByteArray)
            {
                PyErr_Clear();
                encodedByteArray = PyUnicode_AsLatin1String(repr);

                if (!encodedByteArray)
                {
                    PyErr_Clear();
                    encodedByteArray = PyUnicode_AsASCIIString(repr);
                }
            }

            if (encodedByteArray)
            {
                const char* bytes = PyBytes_AS_STRING(encodedByteArray);
                item->m_extendedValue = item->m_value = QString::fromUtf8(bytes);

                if (item->m_value.length() > 100)
                {
                    item->m_value = "<double-click to show value>";
                }
                else if (item->m_value.length() > 20)
                {
                    item->m_value = item->m_value.replace("\n", ";");
                }

                Py_XDECREF(encodedByteArray);
            }
            else
            {
                PyErr_Clear();

                // maybe, encoding of str is unknown, therefore you could decode the
                // string to a new encoding and parse it afterwards
                item->m_extendedValue = item->m_value = "unknown";
            }
        }
        else
        {
            item->m_extendedValue = item->m_value = "unknown";
        }

        Py_XDECREF(repr);
        repr = nullptr;
    }

    if (expandableType)
    {
        // stateChildsAvailable will be set afterwards (if
        // necessary) by loadDictionaryRec
        item->m_childState = PyWorkspaceItem::stateChilds;
    }
    else // the new element is not an expandable type, if the old value has been one, delete the
         // existing elements
    {
        item->m_childState = PyWorkspaceItem::stateNoChilds;

        foreach (const PyWorkspaceItem* child, item->m_childs)
        {
            deletedKeys << fullName + "." + child->m_key;
            delete child;
        }

        item->m_childs.clear();

        // base types first
        if (PyFloat_Check(value))
        {
            item->m_extendedValue = item->m_value = QString("%1").arg(PyFloat_AsDouble(value));
            item->m_compatibleParamBaseType = ito::ParamBase::Double;
        }
        else if (value == Py_None)
        {
            item->m_extendedValue = item->m_value = QString("None");
            item->m_compatibleParamBaseType = 0; // not compatible
        }
        else if (PyBool_Check(value))
        {
            item->m_extendedValue = item->m_value =
                (value == Py_True) ? QString("True") : QString("False");
            item->m_compatibleParamBaseType = ito::ParamBase::Char;
        }
        else if (PyLong_Check(value))
        {
            int overflow;
            item->m_extendedValue = item->m_value =
                QString("%1").arg(PyLong_AsLongLongAndOverflow(value, &overflow));

            if (overflow)
            {
                item->m_extendedValue = item->m_value =
                    (overflow > 0 ? "int too big" : "int too small");
            }
            item->m_compatibleParamBaseType = ito::ParamBase::Int;
        }
        else if (PyComplex_Check(value))
        {
            item->m_extendedValue = item->m_value = QString("%1+%2j")
                                                        .arg(PyComplex_RealAsDouble(value))
                                                        .arg(PyComplex_ImagAsDouble(value));
            item->m_compatibleParamBaseType = 0; // not compatible
        }
        else if (PyBytes_Check(value))
        {
            char* buffer;
            Py_ssize_t length;
            PyBytes_AsStringAndSize(value, &buffer, &length);

            if (length < 350)
            {
                item->m_extendedValue = item->m_value = buffer;
                if (length > 20)
                    item->m_value = item->m_value.replace("\n", ";");
            }
            else
            {
                item->m_value = QString("[String with %1 characters]").arg(length);
                item->m_extendedValue = "";
            }

            item->m_compatibleParamBaseType = ito::ParamBase::String;
        }
        else if (PyArray_Check(value) && PyArray_SIZE((PyArrayObject*)value) > 10)
        {
            PyArrayObject* a = (PyArrayObject*)value;
            // long array
            item->m_extendedValue = "";
            item->m_value =
                QString("[dims: %1, total: %2]").arg(PyArray_NDIM(a)).arg(PyArray_SIZE(a));
            item->m_compatibleParamBaseType = ito::ParamBase::DObjPtr;
        }
        else if (Py_TYPE(value)->tp_repr == nullptr) // no detailed information provided by this
                                                     // type
        {
            item->m_value = value->ob_type->tp_name;
            item->m_extendedValue = "";
            item->m_compatibleParamBaseType = 0; // not compatible
        }
        else
        {
            if (PyDataObject_Check(value) || PyArray_Check(value))
            {
                item->m_compatibleParamBaseType = ito::ParamBase::DObjPtr;
            }
#if ITOM_POINTCLOUDLIBRARY > 0
            else if (PyPointCloud_Check(value))
            {
                item->m_compatibleParamBaseType = ito::ParamBase::PointCloudPtr;
            }
            else if (PyPolygonMesh_Check(value))
            {
                item->m_compatibleParamBaseType = ito::ParamBase::PolygonMeshPtr;
            }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
            else
            {
                item->m_compatibleParamBaseType = 0; // not compatible
            }

            bool reload = true;

            if (Py_TYPE(value) == &PythonPlugins::PyDataIOPluginType && item->m_value != "")
            {
                reload = false;
            }
            else if (Py_TYPE(value) == &PythonPlugins::PyActuatorPluginType && item->m_value != "")
            {
                reload = false;
            }

            if (reload)
            {
                // TODO: increase speed
                PyObject* repr = PyObject_Repr(value);

                if (repr == nullptr)
                {
                    PyErr_Clear();
                    item->m_extendedValue = item->m_value = "<error during call of repr()>";
                }
                else if (PyUnicode_Check(repr))
                {
                    PyObject* encodedByteArray = PyUnicode_AsUTF8String(repr);
                    if (!encodedByteArray)
                    {
                        PyErr_Clear();
                        encodedByteArray = PyUnicode_AsLatin1String(repr);

                        if (!encodedByteArray)
                        {
                            PyErr_Clear();
                            encodedByteArray = PyUnicode_AsASCIIString(repr);
                        }
                    }
                    if (encodedByteArray)
                    {
                    	const char* bytesString = PyBytes_AS_STRING(encodedByteArray);
                        item->m_extendedValue = item->m_value = QString::fromUtf8(bytesString);

                        if (item->m_value.length() > 100)
                        {
                            item->m_value = "<double-click to show value>";
                        }
                        else if (item->m_value.length() > 20)
                        {
                            item->m_value = item->m_value.replace("\n", ";");
                        }

                        Py_XDECREF(encodedByteArray);
                    }
                    else
                    {
                        PyErr_Clear();
                        // maybe, encoding of str is unknown, therefore you could
                        // decode the string to a new encoding and parse it
                        // afterwards
                        item->m_extendedValue = item->m_value = "unknown";
                    }
                }
                else
                {
                    item->m_extendedValue = item->m_value = "unknown";
                }

                Py_XDECREF(repr);
                repr = nullptr;
            }
        }
    }

    if (PyErr_Occurred())
    {
        PyErr_PrintEx(0);
    }
}

//-----------------------------------------------------------------------------------------------------------
ito::PyWorkspaceItem* PyWorkspaceContainer::getItemByFullName(const QString& fullname)
{
    PyWorkspaceItem* result = &m_rootItem;
    QStringList names = fullname.split(ito::PyWorkspaceContainer::delimiter);
    QHash<QString, PyWorkspaceItem*>::iterator it;

    if (names.count() > 0 && names[0] == "")
    {
        names.removeFirst();
    }

    if (names.count() == 0)
    {
        result = nullptr;
    }

    while (names.count() > 0 && result)
    {
        it = result->m_childs.find(names.takeFirst());

        if (it != result->m_childs.end())
        {
            result = (*it);
        }
        else
        {
            result = nullptr;
        }
    }

    return result;
}

} // end namespace ito
