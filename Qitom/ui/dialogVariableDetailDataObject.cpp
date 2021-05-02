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

#include "dialogVariableDetailDataObject.h"
#include <qsharedpointer.h>
#include "dataobj.h"

#include <qclipboard.h>
#include <dataObjectMetaTable.h>

namespace ito
{

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::DialogVariableDetailDataObject(const QString& name, const QString& type, const char* dtype, QSharedPointer<ito::DataObject> dObj, QWidget* parent):
    QDialog(parent)
{
    ui.setupUi(this);

    m_dObj = *dObj;

    ui.txtName->setText(name);
    ui.txtType->setText(type);
    ui.txtDType->setText(dtype);

    ui.dataTable->setData(dObj);
    ui.dataTable->setReadOnly(true);

    ui.metaTable->setData(dObj);    
    ui.metaTable->setReadOnly(true);

    if (m_dObj.getDims() < 3)
    {
        ui.frameAxesVisible->setVisible(false);
    }
    else
    {
        ui.spinBoxDObjColAxis->setValue(0);
        ui.spinBoxDObjColAxis->setMinimum(0);
        ui.spinBoxDObjColAxis->setMaximum(m_dObj.getDims() - 1);
        
        ui.spinBoxDObjRowAxis->setValue(1);
        ui.spinBoxDObjRowAxis->setMinimum(1);
        ui.spinBoxDObjRowAxis->setMaximum(m_dObj.getDims());
    }
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_btnCopyClipboard_clicked()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(ui.txtName->text(), QClipboard::Clipboard);
}

void DialogVariableDetailDataObject::on_spinBoxDObjRowAxis_valueChanged()
{
    changeDObjAxes();
}

void DialogVariableDetailDataObject::on_spinBoxDObjColAxis_valueChanged()
{   
    changeDObjAxes();
}

void DialogVariableDetailDataObject::changeDObjAxes()
{
    int colIdx = ui.spinBoxDObjColAxis->value();
    int rowIdx = ui.spinBoxDObjRowAxis->value();
    /*ito::DataObject roiObj = m_dObj->at(rowIdx, colIdx);
    ui.table->setData(roiObj);*/
}

} //end namespace ito