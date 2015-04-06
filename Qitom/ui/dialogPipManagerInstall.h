/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef DIALOGPIPMANAGERINSTALL_H
#define DIALOGPIPMANAGERINSTALL_H

#include "../../common/addInInterface.h"
#include "../../common/sharedStructures.h"

#include <qdialog.h>
#include <qvector.h>
#include <qevent.h>

#include "ui_dialogPipManagerInstall.h"

namespace ito {

class DialogPipManagerInstall : public QDialog 
{
    Q_OBJECT

public:
    DialogPipManagerInstall(QWidget *parent = NULL );
    ~DialogPipManagerInstall();

    void getResult(int &type, QString &packageName, bool &upgrade, bool &installDeps, QString &findLinks, bool &ignoreIndex);

private:
    Ui::DialogPipManagerInstall ui;

protected:
    
};

} //end namespace ito

#endif
