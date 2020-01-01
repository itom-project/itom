/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
    #if (defined _DEBUG) && (defined WIN32)
        #undef _DEBUG
        #include "python/pythonWrapper.h"
        #define _DEBUG
    #else
        #include "python/pythonWrapper.h"
    #endif
#endif

#include "../global.h"
#include "../common/sharedStructures.h"
#include "../common/sharedStructuresQt.h"

#include "../python/pythonWorkspace.h"

#include <qtreewidget.h>
#include <qmimedata.h>
#include <qpixmap.h>
#include <qhash.h>
#include <qset.h>

namespace ito
{

//! tiny derivative of QTreeWidgetItem, that overwrites the comparison operator
/*
The variable names of a WorkspaceWidget are usually strings, however these strings can
also be numbers (e.g. children of a list or tuple). In this case, it is desired that numbers
are compared to numbers based on their number value and not the text, such that 1 < 2 < 10 instead of "1" < "10" < "2".
*/
class WorkspaceTreeItem : public QTreeWidgetItem
{
public:
    explicit WorkspaceTreeItem(int type = Type) : QTreeWidgetItem(type) {}
    explicit WorkspaceTreeItem(const QStringList &strings, int type = Type) : QTreeWidgetItem(strings, type) {}
    explicit WorkspaceTreeItem(QTreeWidget *view, int type = Type) : QTreeWidgetItem(view, type) {}
    WorkspaceTreeItem(QTreeWidget *view, const QStringList &strings, int type = Type) : QTreeWidgetItem(strings, type) {}
    WorkspaceTreeItem(QTreeWidget *view, QTreeWidgetItem *after, int type = Type) : QTreeWidgetItem(view, after, type) {}
    explicit WorkspaceTreeItem(QTreeWidgetItem *parent, int type = Type) : QTreeWidgetItem(parent, type) {}
    WorkspaceTreeItem(QTreeWidgetItem *parent, const QStringList &strings, int type = Type) : QTreeWidgetItem(parent, strings, type) {}
    WorkspaceTreeItem(QTreeWidgetItem *parent, QTreeWidgetItem *after, int type = Type) : QTreeWidgetItem(parent, after, type) {}
    WorkspaceTreeItem(const QTreeWidgetItem &other) : QTreeWidgetItem(other) {}

    virtual ~WorkspaceTreeItem() {}

    //!overwritten operator for better number comparison
    virtual bool operator<(const QTreeWidgetItem &other) const
    {
        int column = treeWidget()->sortColumn();
        QString thisText = text(column);
        QString otherText = other.text(column);

        bool ok;
        float a = thisText.toFloat(&ok);
        
        if (ok)
        {
            float b = otherText.toFloat(&ok);
            if (ok)
            {
                return a < b;
            }
            
        }
        
        return thisText.localeAwareCompare(otherText) < 0;
    }

};

//! major class WorkspaceWidget to show a tree widget for the global and local workspace toolbox
class WorkspaceWidget : public QTreeWidget
{
    Q_OBJECT
public:
    WorkspaceWidget(bool globalNotLocal, QWidget* parent = NULL);
    ~WorkspaceWidget();

    int numberOfSelectedItems(bool ableToBeRenamed = false) const;
    int numberOfSelectedMainItems() const;
    inline ito::PyWorkspaceContainer* getWorkspaceContainer() { return m_workspaceContainer; }

    enum WorkspaceRole
    {
        RoleFullName = Qt::UserRole + 1, /*!< the fullName role indicates the full, encrypted path name to the variable such that the PythonEngine can decode back the corresponding PyObject */
        RoleCompatibleTypes = Qt::UserRole + 2,
        RoleType     = Qt::UserRole + 3
    };

    QString getPythonReadableName(const QTreeWidgetItem *item) const;

protected:
    QStringList mimeTypes() const;
    QMimeData * mimeData(const QList<QTreeWidgetItem *> items) const;
    void startDrag(Qt::DropActions supportedActions);

private:
    void updateView(QHash<QString,ito::PyWorkspaceItem*> items, QString baseName, QTreeWidgetItem *parent = NULL);
    void recursivelyDeleteHash(QTreeWidgetItem *item);
    void recursivelyDeleteHash(const QString &fullBaseName);

    bool m_globalNotLocal;                              /*!< flag indicated whether this workspaceWidget shows a global (true) or a local (false) dictionary */
    QHash<QString,QTreeWidgetItem*> m_itemHash;
    ito::PyWorkspaceContainer *m_workspaceContainer;

    QPixmap m_dragPixmap;
    Qt::DropActions supportedDragActions() const;

signals:

public slots:
    void workspaceContainerUpdated(PyWorkspaceItem *rootItem, QString fullNameRoot, QStringList recentlyDeletedFullNames);

private slots:
    void itemDoubleClicked(QTreeWidgetItem* item, int column);  /*!< slot invoked if item has been double-clicked */
    void itemExpanded(QTreeWidgetItem* item);
    void itemCollapsed(QTreeWidgetItem* item);
};

} //end namespace ito

#endif
