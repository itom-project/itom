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

#ifndef WORKSPACEWIDGET_H
#define WORKSPACEWIDGET_H

#ifndef Q_MOC_RUN
    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #if (defined _DEBUG) && (!defined linux)
        #undef _DEBUG
        #include "Python.h"
        #define _DEBUG
    #else
        #include "Python.h"
    #endif
#endif

#include "../global.h"
#include "../common/sharedStructures.h"
#include "../common/sharedStructuresQt.h"

#include "../python/pythonWorkspace.h"

#include <qtreewidget.h>

#include <qhash.h>
#include <qdebug.h>
#include <qset.h>

namespace ito
{

class WorkspaceWidget : public QTreeWidget
{
    Q_OBJECT
public:
    WorkspaceWidget(bool globalNotLocal, QWidget* parent = NULL);
    ~WorkspaceWidget();

    inline int numberOfSelectedItems() const { return selectedItems().count(); }
    int numberOfSelectedMainItems() const;
    inline ito::PyWorkspaceContainer* getWorkspaceContainer() { return m_workspaceContainer; }

protected:

private:

    ////! item of WorkspaceWidget
    ///*! this struct corresponds to one item in the workspace widget */
    //struct WorkspaceItem
    //{
    //    WorkspaceItem() : m_item(NULL), m_exists(true) {}; /*!< constructor */
    //    WorkspaceItem(QTreeWidgetItem* item, PyObject* pyValue, QString extendedValue) : m_item(item), m_exists(true), m_pyValue(pyValue), m_extendedValue(extendedValue) {}; /*!< constructor with given values */
    //    QTreeWidgetItem* m_item;    /*!< pointer to the corresponding QTreeWidgetItem */
    //    bool m_exists;              /*!< true: this item is still existing, false: this value is not longer available in the given python dictionary and should be removed */
    //    PyObject* m_pyValue;        /*!< pointer to the corresponding PyObject in the python dictionary. be careful. only use this pointer for comparing, this pointer may be corrupt (if variable does not exist any more)!!! */
    //    QString m_extendedValue;    /*!< String with the full text of this variable (if e.g. a matrix) */
    //};

    //void transformPyValue(PyObject* value, QString &valueLong, QString &valueShort);
    void updateView(QHash<QString,ito::PyWorkspaceItem*> items, QString baseName, QTreeWidgetItem *parent = NULL);
    void recursivelyDeleteHash(QTreeWidgetItem *item);
    void recursivelyDeleteHash(const QString &fullBaseName);

    bool m_globalNotLocal;                              /*!< flag indicated whether this workspaceWidget shows a global (true) or a local (false) dictionary */
    //QHash<QString,WorkspaceItem> m_hashTable;           /*!< hash table which maps a workspace variable name to its corresponding WorkspaceItem */
    QSet<QString> m_pyTypeBlacklist;                    /*!< set of type-string, which will be ignored in this workspaceWidget */
    QHash<QString,QTreeWidgetItem*> m_itemHash;
    ito::PyWorkspaceContainer *m_workspaceContainer;

    QString m_delimiter;

signals:

public slots:
    //ito::RetVal loadDictionary(PyObject *dict, ItomSharedSemaphore *semaphore = NULL);                      /*!< slot invoked by PythonEngine if dictionary has changed */

    void workspaceContainerUpdated(PyWorkspaceItem *rootItem, QString fullNameRoot, QStringList recentlyDeletedFullNames);

private slots:
    void itemDoubleClicked(QTreeWidgetItem* item, int column);  /*!< slot invoked if item has been double-clicked */
    void itemExpanded(QTreeWidgetItem* item);
    void itemCollapsed(QTreeWidgetItem* item);
};

} //end namespace ito

#endif
