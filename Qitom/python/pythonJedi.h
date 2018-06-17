/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef PYTHONJEDI_H
#define PYTHONJEDI_H

#include <QMetaType>
#include <qstring.h>

namespace ito
{
    struct JediCalltip
    {
        JediCalltip() : m_column(-1), m_bracketStartCol(-1), m_bracketStartLine(-1) {};
        JediCalltip(const QString &calltip, int column, int bracketStartLine, int bracketStartCol) :
            m_calltipText(calltip), 
            m_column(column),
            m_bracketStartCol(bracketStartCol),
            m_bracketStartLine(bracketStartLine)
        {}

        QString m_calltipText;
        int m_column;
        int m_bracketStartCol;
        int m_bracketStartLine;
    };

    Q_DECLARE_METATYPE(JediCalltip)

} //end namespace ito

#endif
