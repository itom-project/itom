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

#ifndef DIALOGPIPMANAGER_H
#define DIALOGPIPMANAGER_H

#include "../../common/addInInterface.h"
#include "../../common/sharedStructures.h"

#include "../models/pipManager.h"

#include <qdialog.h>
#include <qvector.h>

#include "ui_dialogPipManager.h"

namespace ito {

class DialogPipManager : public QDialog 
{
    Q_OBJECT

public:
    DialogPipManager(QWidget *parent = NULL );
    ~DialogPipManager();

protected:

private:
    PipManager *m_pPipManager;
    Ui::DialogPipManager ui;
    QString logHtml;

private slots:
    void pipVersion(const QString &version);
    void outputReceived(const QString &text, bool isError);
};

} //end namespace ito

#endif
