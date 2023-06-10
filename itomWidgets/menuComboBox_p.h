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

#ifndef MENUCOMBOBOX_P_H
#define MENUCOMBOBOX_P_H

// Qt includes
#include <QComboBox>
#include <QPointer>

// CTK includes
#include "menuComboBox.h"
class Completer;
class QToolButton;

/// \ingroup Widgets
class MenuComboBoxInternal : public QComboBox
{
    Q_OBJECT
public:
    /// Superclass typedef
    typedef QComboBox Superclass;

    MenuComboBoxInternal();
    virtual ~MenuComboBoxInternal();
    virtual void showPopup();

    virtual QSize minimumSizeHint()const;
Q_SIGNALS:
    void popupShown();
public:
    QPointer<QMenu>  Menu;
};

// -------------------------------------------------------------------------
/// \ingroup Widgets
class MenuComboBoxPrivate : public QObject
{
    Q_OBJECT
    Q_DECLARE_PUBLIC(MenuComboBox);

protected:
    MenuComboBox* const q_ptr;
public:
    MenuComboBoxPrivate(MenuComboBox& object);
    void init();
    QAction* actionByTitle(const QString& text, const QMenu* parentMenu);
    void setCurrentText(const QString& newCurrentText);
    QString currentText()const;

    void setCurrentIcon(const QIcon& newCurrentIcon);
    QIcon currentIcon()const;

    void addAction(QAction* action);
    void addMenuToCompleter(QMenu* menu);
    void addActionToCompleter(QAction* action);

    void removeAction(QAction* action);
    void removeMenuFromCompleter(QMenu* menu);
    void removeActionFromCompleter(QAction* action);

public Q_SLOTS:
    void setComboBoxEditable(bool editable = true);
    void onCompletion(const QString& text);

protected:
    QIcon         DefaultIcon;
    QString       DefaultText;
    bool          IsDefaultTextCurrent;
    bool          IsDefaultIconCurrent;

    MenuComboBox::EditableBehavior EditBehavior;

    MenuComboBoxInternal*    mMenuComboBox;
    Completer*               SearchCompleter;
    QPointer<QMenu>          CompleterMenu;
    QToolButton*             SearchButton;
};

#endif
