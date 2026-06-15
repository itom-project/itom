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

    This class is a port of the Python class TabSwitcherWidget
    of the Spyder IDE (https://github.com/spyder-ide),
    licensed under the MIT License and developed by the Spyder Project
    Contributors.
*********************************************************************** */

#pragma once

#include <qlistwidget.h>
#include <qpointer.h>
#include <qtabwidget.h>
#include <qevent.h>
#include <qlist.h>



namespace ito
{
class ScriptDockWidget;

//!< Show tabs in mru order and change between them.
class TabSwitcherWidget : public QListWidget
{
    Q_OBJECT
public:
    TabSwitcherWidget(QTabWidget *tabWidget, const QList<int> &stackHistory, ScriptDockWidget *scriptDockWidget, QWidget *parent = nullptr);
    virtual ~TabSwitcherWidget();

    void selectRow(int steps);

protected:
    void setDialogPosition();
    int loadData();

    void keyReleaseEvent(QKeyEvent* ev);
    void keyPressEvent(QKeyEvent* ev);
    void focusOutEvent(QFocusEvent* ev);

private:
    QPointer<QTabWidget> m_tabs;
    ScriptDockWidget* m_pScriptDockWidget;
    QList<int> m_stackHistory;

private slots:
    void itemSelected(QListWidgetItem *item = nullptr);
};

} //end namespace ito
