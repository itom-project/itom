/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

    --------------------------------
    This class is a modified version of the class QToolTip of the
    Qt framework (licensed under LGPL):
    https://code.woboq.org/qt5/qtbase/src/widgets/kernel/qtooltip.cpp.html
*********************************************************************** */

#include "pyCodeVariableRenamer.h"
#include "../python/pythonEngine.h"
#include "../AppManagement.h"
#include "../python/pythonJedi.h"

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeVariableRenamer::PyCodeVariableRenamer(QObject* parent /*= nullptr*/) :
    QObject(parent)
{
    m_pPythonEngine = AppManagement::getPythonEngine();

}

//-------------------------------------------------------------------------------------
PyCodeVariableRenamer::~PyCodeVariableRenamer()
{
}
} // namespace ito
