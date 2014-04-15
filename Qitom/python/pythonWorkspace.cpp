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

#include "pythonWorkspace.h"
#include "../../common/sharedStructures.h"

#include "pythonPlugins.h"
#include "pythonDataObject.h"
#include "pythonPCL.h"

namespace ito
{

PyWorkspaceItem::~PyWorkspaceItem()
{
    foreach(const PyWorkspaceItem *child, m_childs)
    {
        delete child;
    }
    m_childs.clear();
}

PyWorkspaceItem::PyWorkspaceItem(const PyWorkspaceItem &other)
{
    m_name = other.m_name;
    m_type = other.m_type;
    m_value = other.m_value;
    m_extendedValue = other.m_extendedValue;
    m_childState = other.m_childState;
    m_exist = other.m_exist;
    m_childs = other.m_childs;
    m_isarrayelement = other.m_isarrayelement;
    m_compatibleParamBaseType = other.m_compatibleParamBaseType;
}

PyWorkspaceContainer::PyWorkspaceContainer(bool globalNotLocal) : m_globalNotLocal(globalNotLocal)
{
    m_delimiter = QString( QByteArray::fromHex("AAD791A8") );
    m_blackListType = QSet<QString>() << "builtin_function_or_method" << "module" << "type" << "function"; // << "dict"; //blacklist of python types, which should not be displayed in the workspace

    dictUnicode = PyUnicode_FromString("__dict__");
}

PyWorkspaceContainer::~PyWorkspaceContainer()
{
    Py_XDECREF(dictUnicode);
}

void PyWorkspaceContainer::clear()
{
    loadDictionary(NULL, "");
}

void PyWorkspaceContainer::loadDictionary(PyObject *obj, QString fullNameParentItem)
{
    QStringList deleteList;
    
    
    if(fullNameParentItem == "")
    {
        loadDictionaryRec(obj,"",&m_rootItem,deleteList);
        emit updateAvailable(&m_rootItem, fullNameParentItem, deleteList);
    }
    else
    {
        QStringList nameSplit = fullNameParentItem.split(m_delimiter);
        if(nameSplit[0] == "") nameSplit.removeFirst();
        PyWorkspaceItem *parent = &m_rootItem;
        QHash<QString, ito::PyWorkspaceItem*>::iterator it;

        while(nameSplit.count() > 0)
        {
            it = parent->m_childs.find(nameSplit.takeFirst());
            if(it != parent->m_childs.end())
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

void PyWorkspaceContainer::loadDictionaryRec(PyObject *obj, QString fullNameParentItem, PyWorkspaceItem *parentItem, QStringList &deletedKeys)
{
    PyObject* keys = NULL;
    PyObject* values = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;
    QHash<QString, PyWorkspaceItem*>::iterator it;
    Py_ssize_t i;
    QString keyText;
    PyObject* keyUTF8String = NULL;
    PyWorkspaceItem *actItem;
    QString fullName;

    //at first, set status of all childs of parentItem to "not-existing"
    it = parentItem->m_childs.begin();
    while (it != parentItem->m_childs.end()) 
    {
        (*it)->m_exist = false;
        ++it;
    }

    if(Py_IsInitialized() && obj != NULL)
    {

        if(PySequence_Check(obj))
        {
            for( i = 0 ; i < PySequence_Size(obj) ; i++)
            {
                value = PySequence_GetItem(obj,i); //new reference

                if(!m_blackListType.contains(value->ob_type->tp_name)) // only if not on blacklist
                {
                    keyText = QString::number(i);

                    it = parentItem->m_childs.find(keyText);
                    if(it == parentItem->m_childs.end()) //not existing yet
                    {
                        actItem = new PyWorkspaceItem();
                        actItem->m_name = keyText;
                        actItem->m_exist = true;
                        actItem->m_isarrayelement = true;
                        fullName = fullNameParentItem + m_delimiter + actItem->m_name;
                        parseSinglePyObject(actItem, value, fullName, deletedKeys, actItem->m_compatibleParamBaseType );
                        if(m_expandedFullNames.contains(fullName))
                        {
                            //load subtree
                            loadDictionaryRec(value, fullName, actItem, deletedKeys);
                        }
                        parentItem->m_childs.insert(keyText,actItem);
                    }
                    else //item with this name already exists
                    {
                        actItem = *it;
                        actItem->m_exist = true;
                        actItem->m_isarrayelement = true;
                        fullName = fullNameParentItem + m_delimiter + actItem->m_name;
                        parseSinglePyObject(actItem, value, fullName, deletedKeys, actItem->m_compatibleParamBaseType );
                        if(m_expandedFullNames.contains(fullName))
                        {
                            //load subtree
                            loadDictionaryRec(value, fullName, actItem, deletedKeys);
                        }
                    }
                }

                Py_DECREF(value);
            }
        }
        else
        {
            if(PyDict_Check(obj))
            {
                keys = PyDict_Keys(obj); //new ref
                values = PyDict_Values(obj); //new ref
            }
            else if(PyMapping_Check(obj))
            {
                keys = PyMapping_Keys(obj); //new ref
                values = PyMapping_Values(obj); //new ref
            }
            else if(PyObject_HasAttr(obj, dictUnicode))
            {
                PyObject *subdict = PyObject_GetAttr(obj, dictUnicode);
                keys = PyDict_Keys(subdict);
                values = PyDict_Values(subdict);
                Py_DECREF(subdict);
            }

            if(keys && values)
            {
                int overflow;
                for( i = 0 ; i < PyList_Size(values) ; i++)
                {
                    value = PyList_GetItem(values, i); //borrowed
                    key = PyList_GetItem(keys, i); //borrowed

                    if(!m_blackListType.contains(value->ob_type->tp_name)) // only if not on blacklist
                    {
                        keyUTF8String = PyUnicode_AsUTF8String(key); //new
                        if(keyUTF8String == NULL)
                        {
                            PyErr_Clear();
                            if(PyLong_Check(key))
                            {
                                keyText = QString::number( PyLong_AsLongAndOverflow(key, &overflow) );
                                if (overflow)
                                {
                                    keyText = QString::number( PyLong_AsLongLong(key) );
                                }
                            }
                            else if(PyFloat_Check(key))
                            {
                                keyText = QString::number( PyFloat_AsDouble(key) );
                            }
                            else
                            {
                                keyText = "<unknown>";
                            }
                        }
                        else
                        {
                            keyText = PyBytes_AsString(keyUTF8String); //borrowed reference to char-pointer in keyUTF8String
                        }

                        it = parentItem->m_childs.find(keyText);
                        if(it == parentItem->m_childs.end()) //not existing yet
                        {
                            actItem = new PyWorkspaceItem();
                            actItem->m_name = keyText;
                            actItem->m_exist = true;
                            actItem->m_isarrayelement = true;
                            fullName = fullNameParentItem + m_delimiter + actItem->m_name;
                            parseSinglePyObject(actItem, value, fullName, deletedKeys, actItem->m_compatibleParamBaseType);
                            if(m_expandedFullNames.contains(fullName))
                            {
                                //load subtree
                                loadDictionaryRec(value, fullName, actItem, deletedKeys);
                            }
                            parentItem->m_childs.insert(keyText,actItem);
                        }
                        else //item with this name already exists
                        {
                            actItem = *it;
                            actItem->m_exist = true;
                            actItem->m_isarrayelement = true;
                            fullName = fullNameParentItem + m_delimiter + actItem->m_name;
                            parseSinglePyObject(actItem, value, fullName, deletedKeys, actItem->m_compatibleParamBaseType);
                            if(m_expandedFullNames.contains(fullName))
                            {
                                //load subtree
                                loadDictionaryRec(value, fullName, actItem, deletedKeys);
                            }
                        }

                        Py_XDECREF(keyUTF8String);
                    }
                }

                Py_DECREF(keys);
                Py_DECREF(values);
            }
        }
    }

    it = parentItem->m_childs.begin();
    while (it != parentItem->m_childs.end()) 
    {
        if( (*it)->m_exist == false)
        {
            deletedKeys << fullNameParentItem + m_delimiter + (*it)->m_name;
            delete (*it);
            it = parentItem->m_childs.erase(it);
        }
        else
        {
            ++it;
        }
    }
}


void PyWorkspaceContainer::parseSinglePyObject(PyWorkspaceItem *item, PyObject *value, QString &fullName, QStringList &deletedKeys, int & /*m_compatibleParamBaseType*/)
{
    Py_ssize_t size;
    bool expandableType = false;

    //check new value
    item->m_exist = true;
    item->m_type = value->ob_type->tp_name;
//    PyObject *subdict = NULL;

    //at first check for possible types which have children (dict,list,tuple) or its subtypes
    if(PyDict_Check(value))
    {
        size = PyDict_Size(value);
        item->m_value = QString("[%1 element(s)]").arg(size);
        expandableType = true;
        item->m_extendedValue = "";
        item->m_compatibleParamBaseType = 0; //not compatible
    }
    else if(PyList_Check(value))
    {
        size = PyList_Size(value);
        item->m_value = QString("[%1 element(s)]").arg(size);
        expandableType = true;
        item->m_extendedValue = "";
        item->m_compatibleParamBaseType = 0; //not compatible
    }
    else if(PyTuple_Check(value))
    {
        size = PyTuple_Size(value);
        item->m_value = QString("[%1 element(s)]").arg(size);
        expandableType = true;
        item->m_extendedValue = "";
        item->m_compatibleParamBaseType = 0; //not compatible
    }
    else if(PyObject_HasAttr(value,dictUnicode))
    {
        //user-defined class (has attr '__dict__')
        expandableType = true;
        item->m_compatibleParamBaseType = 0; //not compatible

        //TODO: increase speed
        PyObject *repr = PyObject_Repr(value);
        if(repr == NULL)
        {
            PyErr_Clear();
            item->m_extendedValue = item->m_value = "unknown";
        }
        else if(PyUnicode_Check(repr))
        {
            PyObject *encodedByteArray = PyUnicode_AsLatin1String(repr);
            if (!encodedByteArray)
            {
                PyErr_Clear();
                encodedByteArray = PyUnicode_AsASCIIString(repr);
                if (!encodedByteArray)
                {
                    PyErr_Clear();
                    encodedByteArray = PyUnicode_AsUTF8String(repr);
                }
            }
            if (encodedByteArray)
            {
                item->m_extendedValue = item->m_value = PyBytes_AS_STRING(encodedByteArray);
                if(item->m_value.length()>20)
                {
                    item->m_value = item->m_value.replace("\n",";");
                }
                else if(item->m_value.length() > 100)
                {
                    item->m_value = "<double-click to show value>";
                }
                Py_XDECREF(encodedByteArray);
            }
            else
            {
                PyErr_Clear();
                item->m_extendedValue = item->m_value = "unknown"; //maybe, encoding of str is unknown, therefore you could decode the string to a new encoding and parse it afterwards
            }
            Py_XDECREF(repr);
        
        }
        else
        {
            item->m_extendedValue = item->m_value = "unknown";
            Py_XDECREF(repr);
        }
    }

    if(expandableType)
    {
        item->m_childState = PyWorkspaceItem::stateChilds; //stateChildsAvailable will be set afterwards (if necessary) by loadDictionaryRec
    }
    else //the new element is not an expandable type, if the old value has been one, delete the existing elements
    {
        item->m_childState = PyWorkspaceItem::stateNoChilds;
        foreach(const PyWorkspaceItem *child, item->m_childs)
        {
            deletedKeys << fullName + "." + child->m_name;
            delete child;
        }
        item->m_childs.clear();

        //base types first
        if(PyFloat_Check(value))
        {
            item->m_extendedValue = item->m_value = QString("%1").arg(PyFloat_AsDouble(value));
            item->m_compatibleParamBaseType = ito::ParamBase::Double;
        }
        else if(value == Py_None)
        {
            item->m_extendedValue = item->m_value = QString("None");
            item->m_compatibleParamBaseType = 0; //not compatible
        }
        else if(PyBool_Check(value))
        {
            item->m_extendedValue = item->m_value = (value == Py_True) ? QString("True") : QString("False");
            item->m_compatibleParamBaseType = ito::ParamBase::Char;
        }
        else if(PyLong_Check(value))
        {
            int overflow;
            item->m_extendedValue = item->m_value = QString("%1").arg(PyLong_AsLongAndOverflow(value, &overflow));
            if (overflow)
            {
                item->m_extendedValue = item->m_value = (overflow > 0 ? "int too big" : "int too small");
            }
            item->m_compatibleParamBaseType = ito::ParamBase::Int;
        }
        else if(PyComplex_Check(value))
        {
            item->m_extendedValue = item->m_value = QString("%1+%2j").arg(PyComplex_RealAsDouble(value)).arg(PyComplex_ImagAsDouble(value));
            item->m_compatibleParamBaseType = 0; //not compatible
        }
        else if(PyBytes_Check(value))
        {
            char* buffer;
            Py_ssize_t length;
            PyBytes_AsStringAndSize(value, &buffer, &length);

            if(length < 350)
            {
                item->m_extendedValue = item->m_value = buffer;
                if(length > 20) item->m_value = item->m_value.replace("\n",";");
            }
            else
            {
                item->m_value = QString("[String with %1 characters]").arg(length);
                item->m_extendedValue = "";
            }
            item->m_compatibleParamBaseType = ito::ParamBase::String;
        }
        else if(PyArray_Check(value) && PyArray_SIZE( (PyArrayObject*)value ) > 10)
        {
            PyArrayObject *a = (PyArrayObject*)value;
            //long array
            item->m_extendedValue = "";
            item->m_value = QString("[dims: %1, total: %2]").arg( PyArray_NDIM(a) ).arg(PyArray_SIZE( a ));
            item->m_compatibleParamBaseType = ito::ParamBase::DObjPtr;
        }
        else if(Py_TYPE(value)->tp_repr == NULL) //no detailed information provided by this type
        {
            item->m_value = value->ob_type->tp_name;
            item->m_extendedValue = "";
            item->m_compatibleParamBaseType = 0; //not compatible
        }
        else
        {
            if(PyDataObject_Check(value) || PyArray_Check(value))
            {
                item->m_compatibleParamBaseType = ito::ParamBase::DObjPtr;
            }
#if ITOM_POINTCLOUDLIBRARY > 0
            else if(PyPointCloud_Check(value))
            {
                item->m_compatibleParamBaseType = ito::ParamBase::PointCloudPtr;
            }
            else if(PyPolygonMesh_Check(value))
            {
                item->m_compatibleParamBaseType = ito::ParamBase::PolygonMeshPtr;
            }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
            else
            {
                item->m_compatibleParamBaseType = 0; //not compatible
            }

            bool reload = true;
            if(Py_TYPE(value) == &PythonPlugins::PyDataIOPluginType && item->m_value != "")
            {
                reload = false;
            }
            else if(Py_TYPE(value) == &PythonPlugins::PyActuatorPluginType && item->m_value != "")
            {
                reload = false;
            }

            if(reload)
            {
                //TODO: increase speed
                PyObject *repr = PyObject_Repr(value);
                if(repr == NULL)
                {
                    PyErr_Clear();
                    item->m_extendedValue = item->m_value = "unknown";
                }
                else if(PyUnicode_Check(repr))
                {
                    PyObject *encodedByteArray = PyUnicode_AsLatin1String(repr);
                    if (!encodedByteArray)
                    {
                        PyErr_Clear();
                        encodedByteArray = PyUnicode_AsASCIIString(repr);
                        if (!encodedByteArray)
                        {
                            PyErr_Clear();
                            encodedByteArray = PyUnicode_AsUTF8String(repr);
                        }
                    }
                    if (encodedByteArray)
                    {
                        item->m_extendedValue = item->m_value = PyBytes_AS_STRING(encodedByteArray);
                        if(item->m_value.length()>20)
                        {
                            item->m_value = item->m_value.replace("\n",";");
                        }
                        else if(item->m_value.length() > 100)
                        {
                            item->m_value = "<double-click to show value>";
                        }
                        Py_XDECREF(encodedByteArray);
                    }
                    else
                    {
                        PyErr_Clear();
                        item->m_extendedValue = item->m_value = "unknown"; //maybe, encoding of str is unknown, therefore you could decode the string to a new encoding and parse it afterwards
                    }
                    Py_XDECREF(repr);
                }
                else
                {
                    item->m_extendedValue = item->m_value = "unknown";
                    Py_XDECREF(repr);
                }
            }
        }
    }
}


ito::PyWorkspaceItem* PyWorkspaceContainer::getItemByFullName(const QString &fullname)
{
    PyWorkspaceItem* result = &m_rootItem;
    QStringList names = fullname.split(m_delimiter);
    QHash<QString, PyWorkspaceItem*>::iterator it;

    if(names.count() > 0 && names[0] == "") names.removeFirst();
    if(names.count() == 0) result = NULL;

    while(names.count() > 0 && result)
    {
        it = result->m_childs.find(names.takeFirst());
        if(it != result->m_childs.end())
        {
            result = (*it);
        }
        else
        {
            result = NULL;
        }
    }

    return result;
}

} //end namespace ito
