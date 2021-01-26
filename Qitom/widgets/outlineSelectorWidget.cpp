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
#include <qicon.h>
#include <qfileinfo.h>
#include <qlabel.h>
#include <qtabwidget.h>

namespace ito
{

//-------------------------------------------------------------------------------------
OutlineSelectorWidget::OutlineSelectorWidget(
        const QList<EditorOutline> &outlines, 
        int currentOutlineIndex,
        ScriptDockWidget *scriptDockWidget, 
        QWidget *parent /*= nullptr*/) :
    QDialog(parent),
    m_pScriptDockWidget(scriptDockWidget),
    m_pTreeWidget(nullptr),
    m_pLineEdit(nullptr),
    m_currentScope(Scope::SingleScript),
    m_outlines(outlines),
    m_currentOutlineIndex(currentOutlineIndex)
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::Popup);

    QVBoxLayout *layout = new QVBoxLayout(this);
    QLineEdit *m_pLineEdit = new QLineEdit(this);
    m_pLineEdit->setText("@");
    m_pLineEdit->setToolTip(tr("Limit the search to the current script with a leading @ sign."));
    connect(m_pLineEdit, &QLineEdit::textChanged,
        this, &OutlineSelectorWidget::filterTextChanged);
    m_pLineEdit->installEventFilter(this);
    layout->addWidget(m_pLineEdit);

    m_pTreeWidget = new QTreeWidget(this);
    m_pTreeWidget->setHeaderHidden(true);
    m_pTreeWidget->setIndentation(15);
    m_pTreeWidget->setItemDelegate(new SelectorDelegate(this));
    connect(m_pTreeWidget, &QTreeWidget::itemActivated,
        this, &OutlineSelectorWidget::itemActivated);

    layout->addWidget(m_pTreeWidget);

    setLayout(layout);

    setContentsMargins(0,0,0,0);

    float f = GuiHelper::screenDpiFactor();
    //int maxWidth = loadData() * 1.1 + f * 80;  // additional for icon...
    int width = qBound((int)(200 * f), (int)(f * 400), (int)(scriptDockWidget->geometry().width() / 1.2));
    int height = qBound((int)(300 * f), (int)(f * 400), (int)(scriptDockWidget->geometry().height() * 1.6));
    QSize size(width, height);

    resize(size);
    setDialogPosition();

    fillContent();

    m_pLineEdit->setFocus();
}

//-------------------------------------------------------------------------------------
OutlineSelectorWidget::~OutlineSelectorWidget()
{
}

//-------------------------------------------------------------------------------------
void OutlineSelectorWidget::fillContent()
{
    m_pTreeWidget->clear();

    if (m_currentScope == Scope::SingleScript)
    {
        const EditorOutline &eo = m_outlines[m_currentOutlineIndex];
        m_pTreeWidget->addTopLevelItems(parseTree(eo.filename, eo.editorUID, eo.rootOutline));
    }
    else
    {
        foreach(const EditorOutline &eo, m_outlines)
        {
            QTreeWidgetItem *toplevel = new QTreeWidgetItem();
            toplevel->setIcon(0, QIcon(":/files/icons/filePython.png"));

            if (eo.filename != "")
            {
                QFileInfo fi(eo.filename);
                toplevel->setData(0, Qt::DisplayRole, fi.fileName());
                toplevel->setData(0, Qt::ToolTipRole, eo.filename);
                toplevel->setData(0, Qt::UserRole, eo.filename);
            }
            else
            {
                toplevel->setData(0, Qt::DisplayRole, tr("Untitled%1").arg(eo.editorUID));
                toplevel->setData(0, Qt::UserRole, "");
            }

            
            toplevel->setData(0, Qt::UserRole + 1, eo.editorUID);
            toplevel->setData(0, Qt::UserRole + 2, -1);
            toplevel->setData(0, Qt::UserRole + 3, "");

            toplevel->addChildren(parseTree(eo.filename, eo.editorUID, eo.rootOutline));

            m_pTreeWidget->addTopLevelItem(toplevel);
        }
    }

    m_pTreeWidget->expandAll();

    if (m_pTreeWidget->topLevelItemCount() > 0)
    {
        m_pTreeWidget->setCurrentItem(m_pTreeWidget->topLevelItem(0));
    }
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
        item->setIcon(0, root->icon());
        item->setData(0, Qt::UserRole, filename);
        item->setData(0, Qt::UserRole + 1, editorUID);
        item->setData(0, Qt::UserRole + 2, root->m_startLineIdx);
        item->setData(0, Qt::UserRole + 3, root->m_name);

        QString displayName;

        switch (root->m_type)
        {
        case OutlineItem::typeFunction:
        case OutlineItem::typeMethod:
            if (root->m_async)
            {
                displayName = "[async] " + root->m_name;
            }
            else
            {
                displayName = root->m_name;
            }
            break;
        case OutlineItem::typePropertyGet:
            if (root->m_async)
            {
                displayName = "[get, async] " + root->m_name;
            }
            else
            {
                displayName = "[get] " + root->m_name;
            }
            break;
        case OutlineItem::typePropertySet:
            if (root->m_async)
            {
                displayName = "[set, async] " + root->m_name;
            }
            else
            {
                displayName = "[set] " + root->m_name;
            }
            break;
        case OutlineItem::typeStaticMethod:
            if (root->m_async)
            {
                displayName = "[static, async] " + root->m_name;
            }
            else
            {
                displayName = "[static] " + root->m_name;
            }
            break;
        case OutlineItem::typeClassMethod:
            if (root->m_async)
            {
                displayName = "[classmethod] async " + root->m_name;
            }
            else
            {
                displayName = "[classmethod] " + root->m_name;
            }
            break;
        case OutlineItem::typeClass:
        default:
            displayName = root->m_name;
            break;
        }

        item->setData(0, Qt::DisplayRole, displayName);

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
    const QTabWidget *tab = m_pScriptDockWidget->tabWidget();

    int left = tab->geometry().width() / 2 - width() / 2;
    int top = 0;

    QPoint pt = m_pScriptDockWidget->tabWidget()->mapToGlobal(QPoint(left, top));

    move(pt);
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
    if (ev->type() == QEvent::KeyPress)
    {
        QKeyEvent *keyev = (QKeyEvent*)ev;

        if (keyev->key() == Qt::Key_Down ||
            keyev->key() == Qt::Key_Up)
        {
            keyev->accept();
            QKeyEvent ev2(keyev->type(),
                keyev->key(),
                keyev->modifiers(),
                keyev->text(),
                keyev->isAutoRepeat(),
                keyev->count());
            QApplication::sendEvent(m_pTreeWidget, &ev2);

        }
        else if (keyev->key() == Qt::Key_Return ||
            keyev->key() == Qt::Key_Enter)
        {
            keyev->accept();
            itemActivated(m_pTreeWidget->currentItem(), 0);
        }
    }

    return QObject::eventFilter(obj, ev);
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
    }
    else if (ev->key() == Qt::Key_Return ||
        ev->key() == Qt::Key_Enter)
    {
        ev->accept();
        itemActivated(m_pTreeWidget->currentItem(), 0);
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
    if (m_currentScope == Scope::SingleScript && !text.startsWith("@"))
    {
        m_currentScope = Scope::AllScripts;
        fillContent();
    }
    else if (m_currentScope == Scope::AllScripts && text.startsWith("@"))
    {
        m_currentScope = Scope::SingleScript;
        fillContent();
    }

    QString text_ = text;

    if (text_.startsWith("@"))
    {
        text_ = text_.mid(1); // remove the @ sign
    }

    for (int i = 0; i < m_pTreeWidget->topLevelItemCount(); ++i)
    {
        filterItemRec(m_pTreeWidget->topLevelItem(i), text_);
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
        QString name = item->data(0, Qt::UserRole + 3).toString();

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

//-------------------------------------------------------------------------------------
SelectorDelegate::SelectorDelegate(QObject *parent /*= nullptr*/) :
    QItemDelegate(parent)
{

}

//-------------------------------------------------------------------------------------
SelectorDelegate::~SelectorDelegate()
{

}

//-------------------------------------------------------------------------------------
/* Override Qt method to force this delegate to look active at all times.
*/
void SelectorDelegate::paint(
    QPainter *painter,
    const QStyleOptionViewItem &option,
    const QModelIndex &index) const
{
    QStyleOptionViewItem option2(option);
    option2.state |= QStyle::State_Active;

    QItemDelegate::paint(painter, option2, index);
}

} // end namespace ito