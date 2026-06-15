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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#include "textDecorationsManager.h"

#include "../codeEditor.h"
#include "../panel.h"


#include <assert.h>
#include <vector>
#include <qdebug.h>

namespace ito {

//---------------------------------------------------------------------
//---------------------------------------------------------------------
TextDecorationsManager::TextDecorationsManager(CodeEditor *editor, QObject *parent /*= NULL*/) :
    Manager(editor, parent)
{
}

//---------------------------------------------------------------------
TextDecorationsManager::~TextDecorationsManager()
{
}

//---------------------------------------------------------------------
QList<QTextEdit::ExtraSelection> TextDecorationsManager::getExtraSelections() const
{
    QList<QTextEdit::ExtraSelection> s;
    for (int i = 0; i < m_decorations.size(); ++i)
    {
        if (m_decorations[i].isNull() == false)
        {
            s << *(dynamic_cast<const QTextEdit::ExtraSelection*>(m_decorations[i].data()));
        }
    }
    return s;
}

bool sortDecorationsByDrawOrder(const TextDecoration::Ptr &a, const TextDecoration::Ptr &b)
{
    return a->drawOrder() < b->drawOrder();
}

//---------------------------------------------------------------------
/*
Adds a text decoration on a CodeEdit instance

:param decoration: Text decoration to add
:type decoration: pyqode.core.api.TextDecoration
*/
bool TextDecorationsManager::append(TextDecoration::Ptr decoration)
{
    if (m_decorations.contains(decoration))
    {
        return false;
    }

    m_decorations.append(decoration);
    std::sort(m_decorations.begin(), m_decorations.end(), sortDecorationsByDrawOrder);
    editor()->setExtraSelections(getExtraSelections());

    //qDebug() << "deco #" << m_decorations.size() << "(append)";

    return true;
}


//---------------------------------------------------------------------
/*
Removes a text decoration from the editor.

:param decoration: Text decoration to remove
:type decoration: pyqode.core.api.TextDecoration
*/
bool TextDecorationsManager::remove(TextDecoration::Ptr decoration)
{
   if (m_decorations.removeOne(decoration))
   {
       editor()->setExtraSelections(getExtraSelections());
       //qDebug() << "deco #" << m_decorations.size() << "(remove)";
       return true;
   }
   return false;
}

//---------------------------------------------------------------------
/*
Removes all text decoration from the editor.
*/
void TextDecorationsManager::clear()
{
    m_decorations.clear();
    editor()->setExtraSelections(QList<QTextEdit::ExtraSelection>());
    //qDebug() << "deco # 0 (clear)";
}

} //end namespace ito
