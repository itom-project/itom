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

#ifndef TEXTDECORATIONSMANAGER_H
#define TEXTDECORATIONSMANAGER_H

/*
Contains the text decorations manager
*/

#include "manager.h"

#include <qlist.h>
#include "../textDecoration.h"

namespace ito {

/*
Manages the collection of TextDecoration that have been set on the editor
widget.
*/
class TextDecorationsManager : public Manager
{
    Q_OBJECT

public:
    TextDecorationsManager(CodeEditor *editor, QObject *parent = NULL);
    virtual ~TextDecorationsManager();

    typedef QList<TextDecoration::Ptr>::const_iterator const_iterator;
    typedef QList<TextDecoration::Ptr>::iterator iterator;

    bool append(TextDecoration::Ptr decoration);
    bool remove(TextDecoration::Ptr decoration);
    void clear();

    bool contains(const TextDecoration::Ptr &deco)
    {
        foreach (const TextDecoration::Ptr &t, m_decorations)
        {
            if (t == deco)
            {
                return true;
            }
        }
        return false;
    }

    const_iterator constBegin() const
    {
          return m_decorations.constBegin();
    }
    const_iterator constEnd() const
    {
          return m_decorations.constEnd();
    }

    iterator begin()
    {
          return m_decorations.begin();
    }
    iterator end()
    {
          return m_decorations.end();
    }

private:
    QList<QTextEdit::ExtraSelection> getExtraSelections() const;

    QList<TextDecoration::Ptr> m_decorations;
};

} //end namespace ito

#endif
