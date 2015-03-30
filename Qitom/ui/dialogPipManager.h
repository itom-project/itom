/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#ifndef DIALOGPIPMANAGER_H
#define DIALOGPIPMANAGER_H

#include "../../common/addInInterface.h"
#include "../../common/sharedStructures.h"

#include <qdialog.h>
#include <qvector.h>

#include "ui_dialogPipManager.h"

namespace ito {

class DialogPipManager : public QDialog 
{
    Q_OBJECT

public:
    DialogPipManager(QWidget *parent = NULL );
    ~DialogPipManager() {};

protected:

private:

    Ui::DialogPipManager ui;

private slots:

};

} //end namespace ito

#endif
