/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

#pragma once

#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before include global.h)
    #define NO_IMPORT_ARRAY

    #include "pythonWrapper.h"
#endif

#include "../global.h"

#include <qstring.h>
#include <qhash.h>
#include <qmutex.h>
#include <qchar.h>
#include <qset.h>
#include <qstringlist.h>
#include <qobject.h>

// type of outer container
#define PY_LIST_TUPLE 'l'
#define PY_MAPPING 'm'
#define PY_ATTR 'a'
#define PY_DICT 'd'

// type of key
#define PY_NUMBER 'n'
#define PY_STRING 's'
#define PY_OBJID 'h' // object id (ptr) as hex for all other objects

namespace ito
{

class PyWorkspaceItem;

//----------------------------------------------------------------------------------------------------------------------------------
/*!
\class PyWorkspaceItem
\brief every item in the workspace is represented by one PyWorkspaceItem
*/
class PyWorkspaceItem
{
public:
    PyWorkspaceItem() : m_compatibleParamBaseType(0), m_exist(false), m_isarrayelement(false), m_childState(stateNoChilds)
    {
    }
    ~PyWorkspaceItem();
    PyWorkspaceItem(const PyWorkspaceItem &other);

    enum ChildState
    {
        stateNoChilds = 0x00, /*!< this variable has no children (no list items, no attributes, no dict items). Therefore no expand indicators are displayed in the tree view. */
        stateChilds = 0x01    /*!< this variable can have children. An expand indicator is shown in the tree view. */
    };

    QString m_name; /*!< name of the item as it is visible in the first column of the workspace (either name of variable or index of list, tuple...) */
    QString m_key;  /*!< type of this item. The string has the following form XY:name, where X is PY_LIST_TUPLE, PY_MAPPING, PY_ATTR or PY_DICT (depends where this variable is member from), Y is PY_NUMBER or PY_STRING (depends on the type of m_name, e.g. variable string name or index of list or tuple) and name is m_name.*/
    QString m_type; /*!< Python internal type name of the variable (ob_type->tp_name of PyObject) */
    QString m_value;
    QString m_extendedValue;
    int m_compatibleParamBaseType; /*!< sets the corresponding type of ito::ParamBase::Type that fits to the variable or 0 if no ito::ParamBase::Type fits. */
    bool m_exist;
    bool m_isarrayelement; /*!< true if this variable is part of a list, tuple, dict, mapping, ... If the python type does not allow any child, m_isarrayelement is set to false. */
    ChildState m_childState; /*!< indicates if this type of variable can have any childs and the expand indicator should be displayed in the tree view. */
    QHash<QString, PyWorkspaceItem*> m_childs;
};


class PyWorkspaceContainer : public QObject //each container has one view
{
    Q_OBJECT
public:

    PyWorkspaceContainer(bool globalNotLocal);
    ~PyWorkspaceContainer();

    void clear();
    void loadDictionary(PyObject *obj, const QString &fullNameParentItem = "");

    inline bool isGlobalWorkspace() const { return m_globalNotLocal; }
    inline bool isRoot(PyWorkspaceItem *item) const { return item == &m_rootItem; }
    inline void emitGetChildNodes(PyWorkspaceContainer *container, QString fullNameParentItem) { emit getChildNodes(container,fullNameParentItem); }

    ito::PyWorkspaceItem* getItemByFullName(const QString &fullname);

    QMutex m_accessMutex;
    QSet<QString> m_expandedFullNames; //this full names are recently expanded in the corresponding view (full name is "." + name + "." + subname + "." + subsubname ...)
    PyWorkspaceItem m_rootItem;

    static QChar delimiter;

private:
    void loadDictionaryRec(PyObject *obj, const QString &fullNameParentItem, PyWorkspaceItem *parentItem, QStringList &deletedKeys);
    void parseSinglePyObject(PyWorkspaceItem *item, PyObject *value, const QString &fullName, QStringList &deletedKeys);

    bool isNotInBlacklist(PyObject *obj) const;

    //!< appends a possible names in __slots__ attribute of objOrType (object or type object) to pre-defined list slotNamesList.
    void appendSlotNamesToList(PyObject *objOrType, PyObject *slotNamesList);

    //!< initializes some Python unicode constant strings. Requires the GIL to do this.
    void initUnicodeConstants();

    bool m_globalNotLocal;
    PyObject *m_dictUnicode;
    PyObject *m_slotsUnicode;
    PyObject *m_mroUnicode;

signals:
    void updateAvailable(PyWorkspaceItem *rootItem, QString fullNameRoot, QStringList recentlyDeletedFullNames);   //TODO
    void getChildNodes(PyWorkspaceContainer *container, QString fullNameParentItem); //signal catched by python    //TODO
};

} //end namespace ito
