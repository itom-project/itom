/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.

    This file is a port and modified version of the
    Common framework (http://www.commontk.org)
*********************************************************************** */

// Qt includes
#include <QAbstractItemView>
#include <QActionEvent>
#include <QCompleter>
#include <QDebug>
#include <QEvent>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QStringList>
#include <QStringListModel>
#include <QToolButton>

// CTK includes
#include "helper/completer.h"
#include "searchBox.h"
#include "menuComboBox_p.h"

// -------------------------------------------------------------------------
MenuComboBoxInternal::MenuComboBoxInternal()
{
}
// -------------------------------------------------------------------------
MenuComboBoxInternal::~MenuComboBoxInternal()
{
}

// -------------------------------------------------------------------------
void MenuComboBoxInternal::showPopup()
{
    QMenu* menu = this->Menu;
    if (!menu)
    {
        return;
    }
    menu->popup(this->mapToGlobal(this->rect().bottomLeft()));
    static int minWidth = menu->sizeHint().width();
    menu->setFixedWidth(qMax(this->width(), minWidth));
    emit popupShown();
}

// -------------------------------------------------------------------------
QSize MenuComboBoxInternal::minimumSizeHint()const
{
    // Cached QComboBox::minimumSizeHint is not recomputed when the current
    // index change, however QComboBox::sizeHint is. Use it instead.
    return this->sizeHint();
}

// -------------------------------------------------------------------------
MenuComboBoxPrivate::MenuComboBoxPrivate(MenuComboBox& object)
    :q_ptr(&object)
{
    this->mMenuComboBox = 0;
    this->SearchCompleter = 0;
    this->EditBehavior = MenuComboBox::NotEditable;
    this->IsDefaultTextCurrent = true;
    this->IsDefaultIconCurrent = true;
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::init()
{
    Q_Q(MenuComboBox);
    this->setParent(q);

    QHBoxLayout* layout = new QHBoxLayout(q);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSizeConstraint(QLayout::SetMinimumSize);
    layout->setSpacing(0);

    // SearchButton
    this->SearchButton = new QToolButton();
    this->SearchButton->setText(q->tr("Search"));
    this->SearchButton->setIcon(QIcon(":/icons/search.svg"));
    this->SearchButton->setCheckable(true);
    this->SearchButton->setAutoRaise(true);
    layout->addWidget(this->SearchButton);
    q->connect(this->SearchButton, SIGNAL(toggled(bool)),
        this, SLOT(setComboBoxEditable(bool)));

    // MenuComboBox
    this->mMenuComboBox = new MenuComboBoxInternal();
    this->mMenuComboBox->setMinimumContentsLength(12);
    layout->addWidget(this->mMenuComboBox);
    this->mMenuComboBox->installEventFilter(q);
    this->mMenuComboBox->setInsertPolicy(QComboBox::NoInsert);
    this->mMenuComboBox->setSizeAdjustPolicy(QComboBox::AdjustToContents);
    this->mMenuComboBox->addItem(this->DefaultIcon, this->DefaultText);
    q->connect(this->mMenuComboBox, SIGNAL(popupShown()),
        q, SIGNAL(popupShown()));

    this->SearchCompleter = new Completer(QStringList(), this->mMenuComboBox);
    this->SearchCompleter->popup()->setParent(q);
    this->SearchCompleter->setCaseSensitivity(Qt::CaseInsensitive);
    this->SearchCompleter->setModelFiltering(Completer::FilterWordStartsWith);
    q->connect(this->SearchCompleter, SIGNAL(activated(QString)),
        this, SLOT(onCompletion(QString)));

    // Automatically set the minimumSizeHint of the layout to the widget
    layout->setSizeConstraint(QLayout::SetMinimumSize);
    // Behave like a QComboBox
    q->setSizePolicy(QSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed,
        QSizePolicy::ComboBox));

    q->setDefaultText(MenuComboBox::tr("Search..."));
}

//  ------------------------------------------------------------------------
QAction* MenuComboBoxPrivate::actionByTitle(const QString& text, const QMenu* parentMenu)
{
    if (!parentMenu || parentMenu->title() == text)
    {
        return 0;
    }
    foreach(QAction* action, parentMenu->actions())
    {
        if (!action->menu() && action->text().toLower() == text.toLower())
        {
            return action;
        }
        if (action->menu())
        {
            QAction* subAction = this->actionByTitle(text, action->menu());
            if (subAction)
            {
                return subAction;
            }
        }
    }
    return 0;
}

//  ------------------------------------------------------------------------
void MenuComboBoxPrivate::setCurrentText(const QString& newCurrentText)
{
    if (this->mMenuComboBox->lineEdit())
    {
        static_cast<SearchBox*>(this->mMenuComboBox->lineEdit())
            ->setPlaceholderText(newCurrentText);
    }

    this->mMenuComboBox->setItemText(this->mMenuComboBox->currentIndex(),
        newCurrentText);
}

//  ------------------------------------------------------------------------
QString MenuComboBoxPrivate::currentText()const
{
    return this->mMenuComboBox->itemText(this->mMenuComboBox->currentIndex());
}

//  ------------------------------------------------------------------------
QIcon MenuComboBoxPrivate::currentIcon()const
{
    return this->mMenuComboBox->itemIcon(this->mMenuComboBox->currentIndex());
}

//  ------------------------------------------------------------------------
void MenuComboBoxPrivate::setCurrentIcon(const QIcon& newCurrentIcon)
{
    this->mMenuComboBox->setItemIcon(this->mMenuComboBox->currentIndex(),
        newCurrentIcon);
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::setComboBoxEditable(bool edit)
{
    Q_Q(MenuComboBox);
    if (edit)
    {
        if (!this->mMenuComboBox->lineEdit())
        {
            SearchBox* line = new SearchBox();
            this->mMenuComboBox->setLineEdit(line);
            if (q->isSearchIconVisible())
            {
                this->mMenuComboBox->lineEdit()->selectAll();
                this->mMenuComboBox->setFocus();
            }
            q->connect(line, SIGNAL(editingFinished()),
                q, SLOT(onEditingFinished()));
        }
        this->mMenuComboBox->setCompleter(this->SearchCompleter);
    }

    this->mMenuComboBox->setEditable(edit);
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::onCompletion(const QString& text)
{
    Q_Q(MenuComboBox);

    // In Qt5, when QCompleter sends its activated() signal, QComboBox sets
    // its current index to the activated item, if found. Work around that behavior
    // by re-selecting the original item.
    this->mMenuComboBox->setCurrentIndex(0);

    // Set text to the completed string
    if (this->mMenuComboBox->lineEdit())
    {
        this->mMenuComboBox->lineEdit()->setText(text);
    }

    q->onEditingFinished();
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::addAction(QAction *action)
{
    if (action->menu())
    {
        this->addMenuToCompleter(action->menu());
    }
    else
    {
        this->addActionToCompleter(action);
    }
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::removeAction(QAction *action)
{
    if (action->menu())
    {
        this->removeMenuFromCompleter(action->menu());
    }
    else
    {
        this->removeActionFromCompleter(action);
    }
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::addMenuToCompleter(QMenu* menu)
{
    Q_Q(MenuComboBox);

    menu->installEventFilter(q);

    // Bug QT : see this link for more details
    // https://bugreports.qt.nokia.com/browse/QTBUG-20929?focusedCommentId=161370#comment-161370
    // if the submenu doesn't have a parent, the submenu triggered(QAction*)
    // signal is not propagated. So we listened this submenu to fix the bug.
    QObject* emptyObject = 0;
    if (menu->parent() == emptyObject)
    {
        q->connect(menu, SIGNAL(triggered(QAction*)),
            q, SLOT(onActionSelected(QAction*)), Qt::UniqueConnection);
    }

    foreach(QAction* action, menu->actions())
    {
        this->addAction(action);
    }
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::removeMenuFromCompleter(QMenu* menu)
{
    Q_Q(MenuComboBox);

    menu->removeEventFilter(q);

    foreach(QAction* action, menu->actions())
    {
        this->removeAction(action);
    }
}

// -------------------------------------------------------------------------
void MenuComboBoxPrivate::addActionToCompleter(QAction *action)
{
    QStringListModel* model = qobject_cast<QStringListModel*>(
        this->SearchCompleter->sourceModel());
    Q_ASSERT(model);
    QModelIndex start = model->index(0, 0);
    QModelIndexList indexList = model->match(start, 0, action->text(), 1, Qt::MatchFixedString | Qt::MatchWrap);
    if (indexList.count())
    {
        return;
    }

    int actionCount = model->rowCount();
    model->insertRow(actionCount);
    QModelIndex index = model->index(actionCount, 0);
    model->setData(index, action->text());
}

//  ------------------------------------------------------------------------
void MenuComboBoxPrivate::removeActionFromCompleter(QAction *action)
{
    QStringListModel* model = qobject_cast<QStringListModel*>(
        this->SearchCompleter->sourceModel());
    Q_ASSERT(model);
    if (!model->stringList().contains(action->text()))
    {
        return;
    }

    QModelIndex start = model->index(0, 0);
    QModelIndexList indexList = model->match(start, 0, action->text());
    Q_ASSERT(indexList.count() == 1);
    foreach(QModelIndex index, indexList)
    {
        // Search completer model is a flat list
        model->removeRow(index.row());
    }
}

//  ------------------------------------------------------------------------
MenuComboBox::MenuComboBox(QWidget* _parent)
    :QWidget(_parent)
    , d_ptr(new MenuComboBoxPrivate(*this))
{
    Q_D(MenuComboBox);
    d->init();
}

//  ------------------------------------------------------------------------
MenuComboBox::~MenuComboBox()
{
}

//  ------------------------------------------------------------------------
void MenuComboBox::setMenu(QMenu* menu)
{
    Q_D(MenuComboBox);
    if (d->mMenuComboBox->Menu == menu)
    {
        return;
    }
    d->mMenuComboBox->Menu = menu;
    this->setCompleterMenu(menu);
}

// -------------------------------------------------------------------------
QMenu* MenuComboBox::menu()const
{
    Q_D(const MenuComboBox);
    return d->mMenuComboBox->Menu;
}

// -------------------------------------------------------------------------
void MenuComboBox::setCompleterMenu(QMenu* menu)
{
    Q_D(MenuComboBox);

    if (d->CompleterMenu == menu)
    {
        return;
    }

    if (d->CompleterMenu)
    {
        QObject::disconnect(d->CompleterMenu, SIGNAL(triggered(QAction*)),
            this, SLOT(onActionSelected(QAction*)));
        this->removeAction(d->CompleterMenu->menuAction());
        d->removeMenuFromCompleter(d->CompleterMenu);
    }

    d->CompleterMenu = menu;

    if (d->CompleterMenu)
    {
        d->addMenuToCompleter(d->CompleterMenu);
        this->addAction(d->CompleterMenu->menuAction());
        QObject::connect(d->CompleterMenu, SIGNAL(triggered(QAction*)),
            this, SLOT(onActionSelected(QAction*)), Qt::UniqueConnection);
    }
}

// -------------------------------------------------------------------------
QMenu* MenuComboBox::completerMenu()const
{
    Q_D(const MenuComboBox);
    return d->CompleterMenu;
}

// -------------------------------------------------------------------------
void MenuComboBox::setDefaultText(const QString& newDefaultText)
{
    Q_D(MenuComboBox);
    d->DefaultText = newDefaultText;
    if (d->IsDefaultTextCurrent)
    {
        d->setCurrentText(d->DefaultText);
    }
}

// -------------------------------------------------------------------------
QString MenuComboBox::defaultText()const
{
    Q_D(const MenuComboBox);
    return d->DefaultText;
}

// -------------------------------------------------------------------------
void MenuComboBox::setDefaultIcon(const QIcon& newIcon)
{
    Q_D(MenuComboBox);
    d->DefaultIcon = newIcon;
    if (d->IsDefaultIconCurrent)
    {
        d->setCurrentIcon(d->DefaultIcon);
    }
}

// -------------------------------------------------------------------------
QIcon MenuComboBox::defaultIcon()const
{
    Q_D(const MenuComboBox);
    return d->DefaultIcon;
}

// -------------------------------------------------------------------------
void MenuComboBox::setEditableBehavior(MenuComboBox::EditableBehavior edit)
{
    Q_D(MenuComboBox);
    d->EditBehavior = edit;
    this->disconnect(d->mMenuComboBox, SIGNAL(popupShown()),
        d, SLOT(setComboBoxEditable()));
    switch (edit)
    {
    case MenuComboBox::Editable:
        d->mMenuComboBox->setContextMenuPolicy(Qt::DefaultContextMenu);
        d->setComboBoxEditable(true);
        break;
    case MenuComboBox::NotEditable:
        d->mMenuComboBox->setContextMenuPolicy(Qt::DefaultContextMenu);
        d->setComboBoxEditable(false);
        break;
    case MenuComboBox::EditableOnFocus:
        d->setComboBoxEditable(this->hasFocus());
        // Here we set the context menu policy to fix a crash on the right click.
        // Opening the context menu removes the focus on the line edit,
        // the comboBox becomes not editable, and the line edit is deleted.
        // The opening of the context menu is done in the line edit and lead to
        // a crash because it infers that the line edit is valid. Another fix
        // could be to delete the line edit later (deleteLater()).
        d->mMenuComboBox->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    case MenuComboBox::EditableOnPopup:
        d->setComboBoxEditable(false);
        this->connect(d->mMenuComboBox, SIGNAL(popupShown()),
            d, SLOT(setComboBoxEditable()));
        // Same reason as in MenuComboBox::EditableOnFocus.
        d->mMenuComboBox->setContextMenuPolicy(Qt::NoContextMenu);
        break;
    }
}

// -------------------------------------------------------------------------
MenuComboBox::EditableBehavior MenuComboBox::editableBehavior()const
{
    Q_D(const MenuComboBox);
    return d->EditBehavior;
}

// -------------------------------------------------------------------------
void MenuComboBox::setSearchIconVisible(bool state)
{
    Q_D(MenuComboBox);
    d->SearchButton->setVisible(state);
}

// -------------------------------------------------------------------------
bool MenuComboBox::isSearchIconVisible() const
{
    Q_D(const MenuComboBox);
    return d->SearchButton->isVisibleTo(const_cast<MenuComboBox*>(this));
}

// -------------------------------------------------------------------------
void MenuComboBox::setToolButtonStyle(Qt::ToolButtonStyle style)
{
    Q_D(MenuComboBox);
    d->SearchButton->setToolButtonStyle(style);
}

// -------------------------------------------------------------------------
Qt::ToolButtonStyle MenuComboBox::toolButtonStyle() const
{
    Q_D(const MenuComboBox);
    return d->SearchButton->toolButtonStyle();
}
// -------------------------------------------------------------------------
void MenuComboBox::setMinimumContentsLength(int characters)
{
    Q_D(MenuComboBox);
    d->mMenuComboBox->setMinimumContentsLength(characters);
}

// -------------------------------------------------------------------------
QComboBox* MenuComboBox::menuComboBoxInternal() const
{
    Q_D(const MenuComboBox);
    return d->mMenuComboBox;
}

// -------------------------------------------------------------------------
QToolButton* MenuComboBox::toolButtonInternal() const
{
    Q_D(const MenuComboBox);
    return d->SearchButton;
}

// -------------------------------------------------------------------------
Completer* MenuComboBox::searchCompleter() const
{
    Q_D(const MenuComboBox);
    return d->SearchCompleter;
}

// -------------------------------------------------------------------------
void MenuComboBox::onActionSelected(QAction* action)
{
    Q_D(MenuComboBox);
    /// Set the action selected in the combobox.

    d->IsDefaultTextCurrent = true;
    QString newText = d->DefaultText;
    if (action && !action->text().isEmpty())
    {
        newText = action->text();
        d->IsDefaultTextCurrent = false;
    }
    d->setCurrentText(newText);

    d->IsDefaultIconCurrent = true;
    QIcon newIcon = d->DefaultIcon;
    if (action && !action->icon().isNull())
    {
        d->IsDefaultIconCurrent = false;
        newIcon = action->icon();
    }
    d->setCurrentIcon(newIcon);

    d->mMenuComboBox->clearFocus();

    emit MenuComboBox::actionChanged(action);
}

// -------------------------------------------------------------------------
void MenuComboBox::clearActiveAction()
{
    this->onActionSelected(0);
}

// -------------------------------------------------------------------------
void MenuComboBox::onEditingFinished()
{
    Q_D(MenuComboBox);
    if (!d->mMenuComboBox->lineEdit())
    {
        return;
    }
    QAction* action = d->actionByTitle(d->mMenuComboBox->lineEdit()->text(), d->CompleterMenu);
    if (!action)
    {
        return;
    }
    if (this->isSearchIconVisible())
    {
        d->SearchButton->setChecked(false);
    }

    action->trigger();
}

// -------------------------------------------------------------------------
bool MenuComboBox::eventFilter(QObject* target, QEvent* event)
{
    Q_D(MenuComboBox);

    if (target == d->mMenuComboBox)
    {
        if (event->type() == QEvent::Resize)
        {
            this->layout()->invalidate();
        }
        if (event->type() == QEvent::FocusIn &&
            d->EditBehavior == MenuComboBox::EditableOnFocus)
        {
            d->setComboBoxEditable(true);
        }
        if (event->type() == QEvent::FocusOut &&
            (d->EditBehavior == MenuComboBox::EditableOnFocus ||
                d->EditBehavior == MenuComboBox::EditableOnPopup))
        {
            d->setComboBoxEditable(false);
        }
    }
    else if (event->type() == QEvent::ActionAdded)
    {
        QActionEvent* actionEvent = static_cast<QActionEvent *>(event);
        d->addAction(actionEvent->action());
    }
    else if (event->type() == QEvent::ActionRemoved)
    {
        QActionEvent* actionEvent = static_cast<QActionEvent *>(event);
        QAction* action = actionEvent->action();
        // Maybe the action is present multiple times in different submenus
        // Don't remove its entry from the completer model if there are still some action instances
        // in the menus.
        if (!d->actionByTitle(action->text(), this->menu()))
        {
            d->removeActionFromCompleter(action);
        }
    }
    return this->Superclass::eventFilter(target, event);
}

/// this slot only returns the argument list for unittest purposes.
    /* This slot has no further functionality and can be used
    to test the proper marshalling between Python, C++ and Qt.
    */
QVariantList MenuComboBox::__unittestVariantList(const QVariantList &list)
{
    return list;
}

int MenuComboBox::__unittestInt(int value) { return value; }
qint64 MenuComboBox::__unittestInt64(qint64 value) { return value; }
quint64 MenuComboBox::__unittestUInt64(quint64 value) { return value; }
float MenuComboBox::__unittestFloat(float value) { return value; }
double MenuComboBox::__unittestDouble(double value) { return value; }
short MenuComboBox::__unittestShort(short value) { return value; }
