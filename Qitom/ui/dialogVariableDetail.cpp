/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#include "dialogVariableDetail.h"

namespace ito
{

DialogVariableDetail::DialogVariableDetail(QString name, QString type, QString value) :
    QDialog()
{
    ui.setupUi(this);
    
    ui.txtName->setText(name);
    ui.txtType->setText(type);
    ui.txtValue->setPlainText(value);
    /*ui.
    ui.lblFilename->setText(filename);
    ui.lblLineNr->setText(QString::number(linenr));

    ui.checkEnabled->setChecked(enabled);
    ui.checkTemporaryBP->setChecked(temporary);

    ui.spinBoxIgnoreCount->setValue(ignoreCount);

    ui.txtCondition->setText(condition);
*/

}

} //end namespace ito