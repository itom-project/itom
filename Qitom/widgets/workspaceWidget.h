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
    #if (defined _DEBUG) && (defined WIN32)
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
#include <qmimedata.h>
#include <qpixmap.h>
#include <qhash.h>
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

    QString m_delimiter;
    QPixmap m_dragPixmap;
#if QT_VERSION >= 0x050000
    Qt::DropActions supportedDragActions() const;
#endif

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
