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

#pragma once

/*
This module contains the OutlineSelectorWidget
*/

#include <qwidget.h>
#include <qsharedpointer.h>
#include <qlist.h>
#include <qtreewidget.h>

#include "../models/outlineItem.h"


namespace ito {

class ScriptDockWidget;

//!< Show tabs in mru order and change between them.
class OutlineSelectorWidget : public QWidget
{
    Q_OBJECT
public:
    struct EditorOutline
    {
        QSharedPointer<OutlineItem> rootOutline;
        int editorUID;
        QString filename;
    };

    OutlineSelectorWidget(const QList<EditorOutline> &outlines, int currentOutlineIndex, ScriptDockWidget *scriptDockWidget, QWidget *parent = nullptr);
    virtual ~OutlineSelectorWidget();

    //void selectRow(int steps);

protected:
    void setDialogPosition();
    QList<QTreeWidgetItem*> parseTree(const QString &filename, int editorUID, const QSharedPointer<OutlineItem> &root) const;
    bool filterItemRec(QTreeWidgetItem *root, const QString &text);

    void keyReleaseEvent(QKeyEvent* ev);
    void keyPressEvent(QKeyEvent* ev);
    void focusOutEvent(QFocusEvent* ev);
    bool eventFilter(QObject* obj, QEvent *ev);

private:
    ScriptDockWidget* m_pScriptDockWidget;
    QTreeWidget *m_pTreeWidget;

private slots:
    void filterTextChanged(const QString &text);
    void itemActivated(QTreeWidgetItem *item, int column);
};

} //end namespace ito
