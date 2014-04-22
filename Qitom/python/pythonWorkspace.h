/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONWORKSPACE_H
#define PYTHONWORKSPACE_H

#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must bebefore include global.h)
    #define NO_IMPORT_ARRAY

    #ifndef ITOM_NPDATAOBJECT
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //see comment in pythonNpDataObject.cpp
    #endif

    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (!defined linux)
        #undef _DEBUG
        #if (defined linux) | (defined CMAKE)
            #include "Python.h"
            #include "numpy/arrayobject.h"
        #else
            #include "Python.h"
            #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
        #endif
        #define _DEBUG
    #else
        #ifdef linux
            #include "Python.h"
            #include "numpy/arrayobject.h"
        #else
            #include "Python.h"
            #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
        #endif
    #endif
#endif

#include "../global.h"

#include <qstring.h>
#include <qhash.h>
#include <qmutex.h>
#include <qset.h>
#include <qstringlist.h>

namespace ito
{

class PyWorkspaceItem;

class PyWorkspaceItem
{
public:
    PyWorkspaceItem() : m_compatibleParamBaseType(0), m_exist(false), m_isarrayelement(false), m_childState(stateNoChilds)
    {
    }
    ~PyWorkspaceItem();
    PyWorkspaceItem(const PyWorkspaceItem &other);

    enum childState { stateNoChilds = 0x00, stateChilds = 0x01};

    QString m_name;
    QString m_type;
    QString m_value;
    QString m_extendedValue;
    int m_compatibleParamBaseType;
    bool m_exist;
    bool m_isarrayelement;
    int m_childState;
    QHash<QString, PyWorkspaceItem*> m_childs;
};


class PyWorkspaceContainer : public QObject //each container has one view
{
    Q_OBJECT
public:

    PyWorkspaceContainer(bool globalNotLocal);
    ~PyWorkspaceContainer();

    void clear();
    void loadDictionary(PyObject *obj, QString fullNameParentItem = "");

    inline bool isGlobalWorkspace() const { return m_globalNotLocal; }
    inline bool isRoot(PyWorkspaceItem *item) const { return item == &m_rootItem; }
    inline void emitGetChildNodes(PyWorkspaceContainer *container, QString fullNameParentItem) { emit getChildNodes(container,fullNameParentItem); }

    inline QString getDelimiter() const { return m_delimiter; };

    ito::PyWorkspaceItem* getItemByFullName(const QString &fullname);

    QMutex m_accessMutex;
    QSet<QString> m_expandedFullNames; //this full names are recently expanded in the corresponding view (full name is "." + name + "." + subname + "." + subsubname ...)
    PyWorkspaceItem m_rootItem;

private:
    void loadDictionaryRec(PyObject *obj, QString fullNameParentItem, PyWorkspaceItem *parentItem, QStringList &deletedKeys);
    void parseSinglePyObject(PyWorkspaceItem *item, PyObject *value, QString &fullName, QStringList &deletedKeys, int &m_compatibleParamBaseType);

    QSet<QString> m_blackListType;
    bool m_globalNotLocal;

    QString m_delimiter;

    PyObject *dictUnicode;

signals:
    void updateAvailable(PyWorkspaceItem *rootItem, QString fullNameRoot, QStringList recentlyDeletedFullNames);   //TODO
    void getChildNodes(PyWorkspaceContainer *container, QString fullNameParentItem); //signal catched by python    //TODO
};

} //end namespace ito


#endif
