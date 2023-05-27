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

#ifndef PYTHONMESSAGEDOCKWIDGET_H
#define PYTHONMESSAGEDOCKWIDGET_H

#include "abstractDockWidget.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qpoint.h>

#include <qtreewidget.h>

namespace ito
{
    class PythonMessageTreeWidget : public QTreeWidget
    {
        Q_OBJECT

    public:
        PythonMessageTreeWidget(QWidget * parent = 0) : QTreeWidget(parent) {};
        virtual ~PythonMessageTreeWidget() {};

    protected:
        QStringList mimeTypes() const;
        QMimeData * mimeData(const QList<QTreeWidgetItem *> items) const;
    };

    class PythonMessageDockWidget: public AbstractDockWidget
    {
        Q_OBJECT

        public:
            PythonMessageDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL, bool docked = true, bool isDockAvailable = true, tFloatingStyle floatingStyle = floatingNone, tMovingStyle movingStyle = movingEnabled);
            ~PythonMessageDockWidget();

        protected:
            void createActions();
            void createMenus();
            void createToolBars();
            void createStatusBar(){};
            void updateActions();
            void updatePythonActions(){ updateActions(); }

        private:
            PythonMessageTreeWidget *m_pythonMessageTreeWidget;
            QMenu* m_pContextMenu;
            ShortcutAction* m_pActClearList;
            QTreeWidgetItem *m_pythonTreeWidgetParent;
            bool m_enabled;
            QString m_dateColor;
            QString m_message;
            bool m_doubleCommand;

        signals:
            void runPythonCommand(const QString cmd);

        private slots:
            void itemDoubleClicked(QTreeWidgetItem *item, int column);
            void mnuClearList();
            void treeWidgetContextMenuRequested(const QPoint &pos);

        public slots:
            void addPythonMessage(QString cmd);
            void propertiesChanged();
    };

} //end namespace ito

#endif
