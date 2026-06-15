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
#include <qdialog.h>
#include <qsharedpointer.h>
#include <qlist.h>
#include <qtreewidget.h>
#include <qitemdelegate.h>

#include "../models/outlineItem.h"


namespace ito {

class ScriptDockWidget;


//!< framless dialog, shown as popup over a script editor, to display the outline.
/* Usually, this popup is displayed by a menu of the script dock widget
or if the Ctrl+D key is pressed.

It then shows the outline of all classes and methods of the current script
or of all opened scripts.

This outline can be filtered by a filter text box and if a method is
selected, the corresponding script is shown and the cursor is placed
at the definition of the selected method or class.
*/
class OutlineSelectorWidget : public QDialog
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

protected:
    enum Scope { AllScripts, SingleScript };

    void setDialogPosition();
    void fillContent();
    QList<QTreeWidgetItem*> parseTree(const QString &filename, int editorUID, const QSharedPointer<OutlineItem> &root) const;
    bool filterItemRec(QTreeWidgetItem *root, const QString &text);
    QString renderTooltipText(const QSharedPointer<OutlineItem> &item) const;
    QString argsWordWrap(QString argText, int maxLineLength) const;
    bool selectFirstVisibleChild(QTreeWidgetItem *parent);

    void keyReleaseEvent(QKeyEvent* ev);
    void keyPressEvent(QKeyEvent* ev);
    void focusOutEvent(QFocusEvent* ev);
    bool eventFilter(QObject* obj, QEvent *ev);

private:
    ScriptDockWidget* m_pScriptDockWidget;
    QTreeWidget *m_pTreeWidget;
    QLineEdit *m_pLineEdit;
    Scope m_currentScope;
    QList<EditorOutline> m_outlines;
    int m_currentOutlineIndex;
    bool m_sortItems;
    QAction *m_actScopeChange;

private slots:
    void filterTextChanged(const QString &text);
    void itemActivated(QTreeWidgetItem *item, int column);
    void actSort(bool triggered);
    void actScopeChanged(bool triggered);
};


/*
This delegate allows the list view of the switcher to look like it has
    the focus, even when its focus policy is set to Qt.NoFocus.
*/
class SelectorDelegate : public QItemDelegate
{
public:
    SelectorDelegate(QObject *parent = nullptr);
    ~SelectorDelegate();

protected:
    virtual void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const;

};


} //end namespace ito
