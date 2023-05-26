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

#include "tabSwitcherWidget.h"
#include "../helper/guiHelper.h"
#include "scriptDockWidget.h"

#include <qstring.h>
#include <qicon.h>
#include <qtabbar.h>
#include <qfileinfo.h>
#include <qdebug.h>


namespace ito
{

//-------------------------------------------------------------------------------------
TabSwitcherWidget::TabSwitcherWidget(
        QTabWidget *tabWidget,
        const QList<int> &stackHistory,
        ScriptDockWidget *scriptDockWidget,
        QWidget *parent /*= nullptr*/) :
    QListWidget(parent),
    m_tabs(tabWidget),
    m_pScriptDockWidget(scriptDockWidget)
{
    setWindowFlags(Qt::FramelessWindowHint | Qt::Dialog);
    m_stackHistory = stackHistory;
    setSelectionMode(QListWidget::SingleSelection);

    float f = GuiHelper::screenDpiFactor();
    int maxWidth = loadData() * 1.1 + f * 80;  // additional for icon...
    int width = qBound((int)(300 * f), maxWidth, m_tabs->geometry().width());
    QSize size(width, 180 * f);

    if (count() < 6)
    {
        size.setHeight(100 * f);
    }

    resize(size);
    setDialogPosition();
    setCurrentRow(0);
}

//-------------------------------------------------------------------------------------
TabSwitcherWidget::~TabSwitcherWidget()
{
}


//-------------------------------------------------------------------------------------
/*Fill ListWidget with the tabs texts.

Add elements in inverse order of stack_history.
*/
int TabSwitcherWidget::loadData()
{
    int index;
    QString text;
    QString path;
    QIcon icon;
    QFileInfo fi;
    QListWidgetItem *item = nullptr;
    int maxWidth = 0;
    const QFontMetrics &fm = fontMetrics();

    for (int i = 0; i < m_stackHistory.size(); ++i)
    {
        index = m_stackHistory[i];
        text = m_tabs->tabText(index);
        icon = m_tabs->tabIcon(index);

        if (m_tabs->tabToolTip(index) != text)
        {
            // real path, else: unsaved script
            fi.setFile(m_tabs->tabToolTip(index));
            path = fi.absolutePath();

            if (path.size() > 60)
            {
                path = path.left(20) + "..." + path.right(37);
            }

            text += QString(" [%1]").arg(path);
        }

        maxWidth = std::max(fm.boundingRect(text).width(), maxWidth);
        item = new QListWidgetItem(icon, text);
        addItem(item);
    }

    return maxWidth;
}

//-------------------------------------------------------------------------------------
/*
Change to the selected document and hide this widget.
*/
void TabSwitcherWidget::itemSelected(QListWidgetItem *item /*= nullptr*/)
{
    if (item == nullptr)
    {
        item = currentItem();
    }

    // stack history is in inverse order
    int idx = row(item);

    if (idx >= 0 && idx < m_stackHistory.size())
    {
        int tabIndex = m_stackHistory[idx];

        if (m_pScriptDockWidget)
        {
            m_pScriptDockWidget->setCurrentIndex(tabIndex);
        }
    }

    hide();
}

//-------------------------------------------------------------------------------------
/*
Move selected row a number of steps.

    Iterates in a cyclic behaviour.
*/
void TabSwitcherWidget::selectRow(int steps)
{
    int row = (currentRow() + steps) % count();

    if (row < 0)
    {
        row = count() - 1;
    }

    setCurrentRow(row);
}

//-------------------------------------------------------------------------------------
/*
Positions the tab switcher in the top-center of the editor.
*/
void TabSwitcherWidget::setDialogPosition()
{
    int left = m_tabs->geometry().width() / 2 - width() / 2;
    int top = m_tabs->tabBar()->geometry().top() - height();

    move(m_tabs->mapToGlobal(QPoint(left, top)));
}

//-------------------------------------------------------------------------------------
/*
Reimplement Qt method.

    Handle "most recent used" tab behavior,
    When ctrl is released and tab_switcher is visible, tab will be changed.
*/
void TabSwitcherWidget::keyReleaseEvent(QKeyEvent* ev)
{
    if (isVisible())
    {
        if (ev->modifiers() == Qt::NoModifier)
        {
            // Ctrl released
            itemSelected();
            ev->accept();
        }
    }
}

//-------------------------------------------------------------------------------------
/*
Reimplement Qt method to allow cyclic behavior.
*/
void TabSwitcherWidget::keyPressEvent(QKeyEvent* ev)
{
    if (ev->modifiers() & Qt::ControlModifier)
    {
        if (ev->key() == Qt::Key_Tab ||
            ev->key() == Qt::Key_Down)
        {
            selectRow(1);
        }
        else if (ev->key() == Qt::Key_Up)
        {
            selectRow(-1);
        }
    }
}

//-------------------------------------------------------------------------------------
/*
Reimplement Qt method to close the widget when loosing focus.
*/
void TabSwitcherWidget::focusOutEvent(QFocusEvent* ev)
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

} // end namespace ito
