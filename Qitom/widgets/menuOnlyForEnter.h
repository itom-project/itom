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

#include <qmenu.h>
#include <qevent.h>
#include <qapplication.h>

namespace ito
{

/*
The class executes the selected action when "enter key" is input.
If a input of keyboard is not the "enter key", the menu is closed and
the input is inserted to code editor.
*/
class MenuOnlyForEnter : public QMenu
{
public:
    explicit MenuOnlyForEnter(QWidget *parent = nullptr) :
        QMenu(parent),
        m_pEditor(parent)
    {
    }

protected:
    //!< close the instance if key is not enter key.
    void keyPressEvent(QKeyEvent *e)
    {
        if (e->key() != Qt::Key_Enter &&
            e->key() != Qt::Key_Return)
        {
            e->accept();

            if (m_pEditor)
            {
                QKeyEvent ev2(e->type(),
                    e->key(),
                    e->modifiers(),
                    e->text(),
                    e->isAutoRepeat(),
                    e->count());
                QApplication::sendEvent(m_pEditor, &ev2);
            }

            close();
        }
        else
        {
            QMenu::keyPressEvent(e);
        }
    }

private:
    QWidget *m_pEditor;
};

} //end namespace ito
