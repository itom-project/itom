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

#ifndef MODESMANAGER_H
#define MODESMANAGER_H

/*
This module contains the modes controller.
*/

#include "manager.h"

#include <qmap.h>
#include "../mode.h"

namespace ito {

/*
Manages the list of modes of the code edit widget.
*/
class ModesManager : public Manager
{
    Q_OBJECT

public:
    ModesManager(CodeEditor *editor, QObject *parent = NULL);
    virtual ~ModesManager();

    typedef QMap<QString, Mode::Ptr>::const_iterator const_iterator;
    typedef QMap<QString, Mode::Ptr>::iterator iterator;

    Mode::Ptr append(Mode::Ptr mode);
    Mode::Ptr remove(Mode::Ptr mode);
    void clear();

    Mode::Ptr get(const QString &name) const
    {
        if (m_modes.contains(name))
        {
            return m_modes[name];
        }
        return Mode::Ptr();
    }

    template <typename _Tp> _Tp* getT(const QString &name) const
    {
        _Tp* ptr = NULL;
        if (m_modes.contains(name))
        {
            Mode::Ptr p = m_modes[name];
            if (p)
            {
                ptr = dynamic_cast<_Tp*>(p.data());
            }
        }

        return ptr;
    }

    const_iterator constBegin() const
    {
          return m_modes.constBegin();
    }
    const_iterator constEnd() const
    {
          return m_modes.constEnd();
    }

    iterator begin()
    {
          return m_modes.begin();
    }
    iterator end()
    {
          return m_modes.end();
    }

private:

    QMap<QString, Mode::Ptr> m_modes;
};

} //end namespace ito

#endif
