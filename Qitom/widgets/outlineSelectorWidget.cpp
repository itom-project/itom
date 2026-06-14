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
#include "../AppManagement.h"
#include "../organizer/scriptEditorOrganizer.h"

#include <qboxlayout.h>
#include <qlineedit.h>
#include <qtreewidget.h>
#include <qicon.h>
#include <qfileinfo.h>
#include <qlabel.h>
#include <qtabwidget.h>
#include <qtoolbar.h>
#include <qsettings.h>

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
    m_currentOutlineIndex(currentOutlineIndex),
    m_sortItems(false),
    m_actScopeChange(nullptr)
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::Popup);
    float f = GuiHelper::screenDpiFactor();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("OutlineSelectorWidget");
    m_sortItems = settings.value("sortItems", false).toBool();
    settings.endGroup();

    QVBoxLayout *layout = new QVBoxLayout(this);

    QToolBar *toolBar = new QToolBar(this);

    // adjust the height of the toolbar in to 16 px with
    // a 96dpi screen and scale it to other pixels for higher resolving screens.
    toolBar->setIconSize(QSize(16 * f, 16 * f));

    layout->addWidget(toolBar);
    layout->setContentsMargins(f * 4, f*4, f*4, f*4);

    QAction *act = toolBar->addAction(
        QIcon(":/classNavigator/icons/sortAZAsc.png"),
        tr("Sort alphabetically"),
        this, SLOT(actSort(bool))
    );
    act->setCheckable(true);
    act->setChecked(m_sortItems);

    m_actScopeChange = toolBar->addAction(
        QIcon(":/files/icons/multiFilePython.png"),
        tr("Show the outline of all opened files"),
        this, SLOT(actScopeChanged(bool))
    );
    m_actScopeChange->setCheckable(true);
    m_actScopeChange->setChecked(m_currentScope != Scope::SingleScript);

    m_pLineEdit = new QLineEdit(this);
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

    setTabOrder(toolBar, m_pLineEdit);
    setTabOrder(m_pLineEdit, m_pTreeWidget);

    setLayout(layout);

    setContentsMargins(0,0,0,0);


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

    if (m_sortItems)
    {
        m_pTreeWidget->sortItems(0, Qt::AscendingOrder);
    }
}

//-------------------------------------------------------------------------------------
//!< wraps the arguments text of a signature with a maximum line length
QString OutlineSelectorWidget::argsWordWrap(QString argText, int maxLineLength) const
{
    QString result;
    int j, i;
    bool firstWrap = true;

    for (;;)
    {
        i = std::min(maxLineLength, (int)(argText.length()));
        j = argText.lastIndexOf(", ", i);

        if (j == -1)
        {
            j = argText.indexOf(", ", i);
        }

        if (j > 0)
        {
            result += argText.left(j);
            result += ",\n    ";
            argText = argText.mid(j + 2);

            if (firstWrap)
            {
                firstWrap = false;
                maxLineLength -= 4;
            }
        }
        else
        {
            break;
        }

        if (maxLineLength >= argText.length())
        {
            break;
        }
    }

    return result + argText;
}

//-------------------------------------------------------------------------------------
QString OutlineSelectorWidget::renderTooltipText(const QSharedPointer<OutlineItem> &item) const
{
    QString name = item->m_name;

    switch (item->m_type)
    {
    case OutlineItem::typeFunction:
    case OutlineItem::typeMethod:
        if (item->m_async)
        {
            name = "[async] def " + item->m_name;
        }
        else
        {
            name = "def " + item->m_name;
        }
        break;
    case OutlineItem::typePropertyGet:
        if (item->m_async)
        {
            name = "[get, async] def " + item->m_name;
        }
        else
        {
            name = "[get] def " + item->m_name;
        }
        break;
    case OutlineItem::typePropertySet:
        if (item->m_async)
        {
            name = "[set, async] def " + item->m_name;
        }
        else
        {
            name = "[set] def " + item->m_name;
        }
        break;
    case OutlineItem::typeStaticMethod:
        if (item->m_async)
        {
            name = "[static, async] def " + item->m_name;
        }
        else
        {
            name = "[static] def " + item->m_name;
        }
        break;
    case OutlineItem::typeClassMethod:
        if (item->m_async)
        {
            name = "[classmethod] async def " + item->m_name;
        }
        else
        {
            name = "[classmethod] def " + item->m_name;
        }
        break;
    case OutlineItem::typeClass:
    default:
        name = "class " + item->m_name;
        break;
    }

    QString fullSig = QString("%1(%2)").arg(name, item->m_args);

    if (item->m_returnType != "")
    {
        fullSig += " -> " + item->m_returnType;
    }

    const int maxLength = 150;

    if (fullSig.size() > maxLength)
    {
        QString methArgsWrapped = argsWordWrap(item->m_args, 100);

        if (item->m_returnType == "")
        {
            fullSig = QString("%1(\n    %2\n)").arg(name, methArgsWrapped);
        }
        else
        {
            fullSig = QString("%1(\n    %2\n) -> %3").arg(name, methArgsWrapped, item->m_returnType);
        }
    }

    return fullSig;
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
        item->setData(0, Qt::ToolTipRole, renderTooltipText(root));

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
    int top = 8;

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
        /*else if (keyev->key() == Qt::Key_Tab)
        {
            keyev->accept();
            m_pTreeWidget->setFocus();
        }*/
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
    m_actScopeChange->blockSignals(true);

    if (m_currentScope == Scope::SingleScript && !text.startsWith("@"))
    {
        m_currentScope = Scope::AllScripts;
        m_actScopeChange->setChecked(true);
        fillContent();
    }
    else if (m_currentScope == Scope::AllScripts && text.startsWith("@"))
    {
        m_currentScope = Scope::SingleScript;
        m_actScopeChange->setChecked(false);
        fillContent();
    }

    m_actScopeChange->blockSignals(false);

    QString text_ = text;

    if (text_.startsWith("@"))
    {
        text_ = text_.mid(1); // remove the @ sign
    }

    for (int i = 0; i < m_pTreeWidget->topLevelItemCount(); ++i)
    {
        filterItemRec(m_pTreeWidget->topLevelItem(i), text_);
    }

    QTreeWidgetItem *item = m_pTreeWidget->currentItem();

    if (!item || item->isHidden())
    {
        QTreeWidgetItem *item;

        for (int i = 0; i < m_pTreeWidget->topLevelItemCount(); ++i)
        {
            item = m_pTreeWidget->topLevelItem(i);

            if (!item->isHidden())
            {
                if (!selectFirstVisibleChild(item))
                {
                    item->setSelected(true);
                    m_pTreeWidget->setCurrentItem(item);
                    m_pTreeWidget->expandItem(item);
                }

                break;
            }
        }
    }
}

//-------------------------------------------------------------------------------------
bool OutlineSelectorWidget::selectFirstVisibleChild(QTreeWidgetItem *parent)
{
    bool found = false;
    QTreeWidgetItem *item;

    for (int i = 0; i < parent->childCount(); ++i)
    {
        item = parent->child(i);

        if (item && !item->isHidden())
        {
            if (!selectFirstVisibleChild(item))
            {
                item->setSelected(true);
                m_pTreeWidget->setCurrentItem(item);
                m_pTreeWidget->expandItem(item);
            }

            found = true;
            break;
        }
    }

    return found;
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

        auto *seo = qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

        if (seo)
        {
            ScriptDockWidget* scriptDockWidget =
                seo->activateOpenedScriptByFilename(filename, -1, editorUID);

            if (scriptDockWidget && startLineIdx >= 0)
            {
                scriptDockWidget->activeTabShowLineAndHighlightWord(
                    startLineIdx,
                    name
                );
            }
        }
    }

    close();
}

//-------------------------------------------------------------------------------------
void OutlineSelectorWidget::actSort(bool triggered)
{
    if (triggered != m_sortItems)
    {
        m_sortItems = triggered;

        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
        settings.beginGroup("OutlineSelectorWidget");
        settings.setValue("sortItems", m_sortItems);
        settings.endGroup();

        if (triggered)
        {
            m_pTreeWidget->sortItems(0, Qt::AscendingOrder);
        }
        else
        {
            fillContent();
        }
    }
}

//-------------------------------------------------------------------------------------
void OutlineSelectorWidget::actScopeChanged(bool triggered)
{
    if (m_currentScope == Scope::SingleScript && triggered)
    {
        //m_currentScope is changed by text change of m_pLineEdit!

        QString txt = m_pLineEdit->text();

        if (txt.startsWith("@"))
        {
            m_pLineEdit->setText(txt.mid(1).trimmed());
        }
    }
    else if (m_currentScope == Scope::AllScripts && !triggered)
    {
        //m_currentScope is changed by text change of m_pLineEdit!

        QString txt = m_pLineEdit->text();

        if (!txt.startsWith("@"))
        {
            m_pLineEdit->setText("@" + txt);
        }
    }
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
