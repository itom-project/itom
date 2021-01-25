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

#include "outlineSelectorWidget.h"

#include "common/sharedStructures.h"
#include "../helper/guiHelper.h"
#include "scriptDockWidget.h"

#include <qboxlayout.h>
#include <qlineedit.h>
#include <qtreewidget.h>

namespace ito
{

//-------------------------------------------------------------------------------------
OutlineSelectorWidget::OutlineSelectorWidget(
        const QList<EditorOutline> &outlines, 
        int currentOutlineIndex,
        ScriptDockWidget *scriptDockWidget, 
        QWidget *parent /*= nullptr*/) :
    m_pScriptDockWidget(scriptDockWidget),
    QWidget(parent)
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::Dialog);

    QVBoxLayout *layout = new QVBoxLayout(this);
    QLineEdit *lineEdit = new QLineEdit(this);
    connect(lineEdit, &QLineEdit::textChanged,
        this, &OutlineSelectorWidget::filterTextChanged);
    lineEdit->installEventFilter(this);
    layout->addWidget(lineEdit);

    m_pTreeWidget = new QTreeWidget(this);
    m_pTreeWidget->setHeaderHidden(true);
    m_pTreeWidget->setIndentation(15);
    connect(m_pTreeWidget, &QTreeWidget::itemActivated,
        this, &OutlineSelectorWidget::itemActivated);

    layout->addWidget(m_pTreeWidget);
    setLayout(layout);

    //setFocusPolicy(Qt::NoFocus);
    setFocusProxy(lineEdit);
    setContentsMargins(0,0,0,0);

    float f = GuiHelper::screenDpiFactor();
    //int maxWidth = loadData() * 1.1 + f * 80;  // additional for icon...
    int width = qBound((int)(200 * f), (int)(f * 400), (int)(scriptDockWidget->geometry().width() / 1.2));
    int height = qBound((int)(300 * f), (int)(f * 400), (int)(scriptDockWidget->geometry().height() * 1.6));
    QSize size(width, height);

    resize(size);
    setDialogPosition();

    const EditorOutline &eo = outlines[currentOutlineIndex];
    m_pTreeWidget->addTopLevelItems(parseTree(eo.filename, eo.editorUID, eo.rootOutline));
    m_pTreeWidget->expandAll();
}

//-------------------------------------------------------------------------------------
OutlineSelectorWidget::~OutlineSelectorWidget()
{
}

//-------------------------------------------------------------------------------------
QList<QTreeWidgetItem*> OutlineSelectorWidget::parseTree(
    const QString &filename,
    int editorUID,
    const QSharedPointer<OutlineItem> &root) const
{
    if (root.isNull())
    {
        return QList<QTreeWidgetItem*>();
    }

    QList<QTreeWidgetItem*> items;

    if (root->m_type == OutlineItem::typeRoot)
    {
        for (int i = 0; i < root->m_childs.count(); ++i)
        {
            items.append(parseTree(filename, editorUID, root->m_childs[i]));
        }
    }
    else
    {
        QTreeWidgetItem *item = new QTreeWidgetItem();
        item->setData(0, Qt::DisplayRole, root->m_name);
        item->setIcon(0, root->icon());
        item->setData(0, Qt::UserRole, filename);
        item->setData(0, Qt::UserRole + 1, editorUID);
        item->setData(0, Qt::UserRole + 2, root->m_startLineIdx);

        for (int i = 0; i < root->m_childs.count(); ++i)
        {
            item->addChildren(parseTree(filename, editorUID, root->m_childs[i]));
        }

        items << item;
    }

    return items;
}

//-------------------------------------------------------------------------------------
/*
Positions the tab switcher in the top-center of the editor.
*/
void OutlineSelectorWidget::setDialogPosition()
{
    int left = m_pScriptDockWidget->geometry().width() / 2 - width() / 2;
    int top = m_pScriptDockWidget->geometry().top();

    move(m_pScriptDockWidget->mapToGlobal(QPoint(left, top)));
}

//-------------------------------------------------------------------------------------
/*
Reimplement Qt method.

    Handle "most recent used" tab behavior,
    When ctrl is released and tab_switcher is visible, tab will be changed.
*/
void OutlineSelectorWidget::keyReleaseEvent(QKeyEvent* ev)
{
    if (isVisible())
    {
        if (ev->modifiers() == Qt::NoModifier)
        {
            // Ctrl released
            //itemSelected();
            ev->accept();
        }
    }
}

//-------------------------------------------------------------------------------------
bool OutlineSelectorWidget::eventFilter(QObject* obj, QEvent *ev)
{
    if (ev->type() == QEvent::FocusOut) 
    {
        focusOutEvent((QFocusEvent*)ev);
        return true;
    }
    else {
        // standard event processing
        return QObject::eventFilter(obj, ev);
    }
}

//-------------------------------------------------------------------------------------
/*
Reimplement Qt method to allow cyclic behavior.
*/
void OutlineSelectorWidget::keyPressEvent(QKeyEvent* ev)
{
    if (ev->key() == Qt::Key_Escape)
    {
        ev->accept();
        close();
    }
    else if (ev->key() == Qt::Key_Down ||
        ev->key() == Qt::Key_Up)
    {
        ev->accept();
        QKeyEvent ev2(ev->type(),
            ev->key(),
            ev->modifiers(),
            ev->text(),
            ev->isAutoRepeat(),
            ev->count());
        QApplication::sendEvent(m_pTreeWidget, &ev2);
        
    }
    else if (ev->key() == Qt::Key_Return ||
        ev->key() == Qt::Key_Enter)
    {
        ev->accept();
        itemActivated(m_pTreeWidget->currentItem(), 0);
    }

    if (ev->modifiers() & Qt::ControlModifier)
    {
        /*if (ev->key() == Qt::Key_Tab ||
            ev->key() == Qt::Key_Down)
        {
            selectRow(1);
        }
        else if (ev->key() == Qt::Key_Up)
        {
            selectRow(-1);
        }*/
    }
}

//-------------------------------------------------------------------------------------
/*
Reimplement Qt method to close the widget when loosing focus.
*/
void OutlineSelectorWidget::focusOutEvent(QFocusEvent* ev)
{
    ev->ignore();

    // Inspired from CompletionWidget.focusOutEvent() in file
    // widgets / sourcecode / base.py line 212
#ifdef Q_OS_DARWIN
    if (ev->reason() != Qt::ActiveWindowFocusReason)
    {
        close();
    }
#else
    close();
#endif
}

//-------------------------------------------------------------------------------------
bool OutlineSelectorWidget::filterItemRec(QTreeWidgetItem *root, const QString &text)
{
    if (text == "")
    {
        root->setHidden(false);

        for (int i = 0; i < root->childCount(); ++i)
        {
            filterItemRec(root->child(i), text);
        }

        return true; // all visible
    }
    
    bool childsVisible = false;

    for (int i = 0; i < root->childCount(); ++i)
    {
        childsVisible |= filterItemRec(root->child(i), text);
    }

    if (childsVisible)
    {
        root->setHidden(false);
    }
    else
    {
        QString name = root->data(0, Qt::DisplayRole).toString();

        if (name.contains(text, Qt::CaseInsensitive))
        {
            root->setHidden(false);
            childsVisible = true;
        }
        else
        {
            root->setHidden(true);
        }
    }

    return childsVisible;
}

//-------------------------------------------------------------------------------------
void OutlineSelectorWidget::filterTextChanged(const QString &text)
{
    for (int i = 0; i < m_pTreeWidget->topLevelItemCount(); ++i)
    {
        filterItemRec(m_pTreeWidget->topLevelItem(i), text);
    }
}

//-------------------------------------------------------------------------------------
void OutlineSelectorWidget::itemActivated(QTreeWidgetItem *item, int column)
{
    if (item)
    {
        QString filename = item->data(0, Qt::UserRole).toString();
        int editorUID = item->data(0, Qt::UserRole + 1).toInt();
        int startLineIdx = item->data(0, Qt::UserRole + 2).toInt();
        QString name = item->data(0, Qt::DisplayRole).toString();

        m_pScriptDockWidget->activateTabByFilename(filename, -1, editorUID);

        if (startLineIdx >= 0)
        {
            m_pScriptDockWidget->activeTabShowLineAndHighlightWord(
                startLineIdx,
                name
            );
        }
    }

    close();
}

} // end namespace ito