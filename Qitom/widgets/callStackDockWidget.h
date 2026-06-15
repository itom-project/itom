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

#ifndef CALLSTACKDOCKWIDGET_H
#define CALLSTACKDOCKWIDGET_H

#include "abstractDockWidget.h"

#include <qwidget.h>
#include <qaction.h>
#include <qtoolbar.h>
#include <qtablewidget.h>


namespace ito {


class CallStackDockWidget : public AbstractDockWidget
{
    Q_OBJECT

public:
    CallStackDockWidget(const QString &title, const QString &objName, QWidget *parent = NULL, bool docked = true, bool isDockAvailable = true, tFloatingStyle floatingStyle = floatingNone, tMovingStyle movingStyle = movingEnabled);
    ~CallStackDockWidget();

protected:
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar(){}
    //void updateActions();
    void updatePythonActions(){ updateActions(); }

private:
    QTableWidget *m_table;
    QStringList m_headers;
    int m_currentRow;
    QIcon m_emptyIcon;
    QIcon m_currentIcon;
    QIcon m_selectedIcon;

    enum CallStackColumns
    {
        ColFilename = 0,
        ColMethod = 1,
        ColLine = 2
    };

private slots:
    void itemDoubleClicked(QTableWidgetItem *item);

public slots:
    void updateCallStack(QStringList filenames, IntList lines, QStringList methods);
    void deleteCallStack();

};

} //end namespace ito

#endif
