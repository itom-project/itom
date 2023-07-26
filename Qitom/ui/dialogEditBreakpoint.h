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

#ifndef DIALOGEDITBREAKPOINT_H
#define DIALOGEDITBREAKPOINT_H

#include <qdialog.h>

#include "ui_dialogEditBreakpoint.h"

namespace ito
{

class DialogEditBreakpoint : public QDialog
{
public:
    DialogEditBreakpoint(QString filename, int linenr, bool enabled, bool temporary, int ignoreCount, QString condition, QWidget *parent = NULL);
    ~DialogEditBreakpoint() {};

    void getData(bool &enabled, bool &temporary, int &ignoreCount, QString &condition);

private:
    Ui::DialogEditBreakpoint ui;

private slots:

};

} //end namespace ito

#endif
