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

#include "checkableComboBox.h"

#include <qclipboard.h>

namespace ito
{

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::DialogVariableDetailDataObject(
    const QString& name,
    const QString& type,
    const char* dtype,
    QSharedPointer<ito::DataObject> dObj,
    QWidget* parent) :
    QDialog(parent),
    m_isChanging(true), 
    m_AxesRanges(nullptr)
{
    ui.setupUi(this);

    m_dObj = dObj;

    ui.txtName->setText(name);
    ui.txtType->setText(type);
    ui.txtDType->setText(dtype);

    
    ui.dataTable->setReadOnly(true);

    ui.metaWidget->setData(m_dObj);    
    ui.metaWidget->setReadOnly(true);

    int dims = m_dObj->getDims();
    if (dims >= 3)
    {
        int col = dims - 2;
        int row = dims - 1;
        ui.groupBoxTableAxes->setVisible(true);

        ui.spinBoxTableCol->setMinimum(0);
        ui.spinBoxTableCol->setMaximum(col);
        ui.spinBoxTableCol->setValue(col);

        ui.spinBoxTableRow->setMinimum(1);
        ui.spinBoxTableRow->setMaximum(row);
        ui.spinBoxTableRow->setValue(row);

        m_AxesRanges = new ito::Range[dims];
        m_isChanging = false;

        // show last two axes after start
        changeDObjAxes(row, col);
    }
    else
    {
        ui.groupBoxTableAxes->setVisible(false);
        ui.dataTable->setData(m_dObj);
    }

}

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::~DialogVariableDetailDataObject()
{
    DELETE_AND_SET_NULL_ARRAY(m_AxesRanges);
}
    //----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_btnCopyClipboard_clicked()
{
    QClipboard *clipboard = QApplication::clipboard();
    clipboard->setText(ui.txtName->text(), QClipboard::Clipboard);
}

void DialogVariableDetailDataObject::on_spinBoxTableCol_valueChanged()
{
    ui.spinBoxTableRow->setEnabled(false);
    changeDObjAxes(ui.spinBoxTableRow->value(), ui.spinBoxTableCol->value());
    ui.spinBoxTableRow->setEnabled(true);
}

void DialogVariableDetailDataObject::on_spinBoxTableRow_valueChanged()
{   
    ui.spinBoxTableCol->setEnabled(false);
    changeDObjAxes(ui.spinBoxTableRow->value(), ui.spinBoxTableCol->value());
    ui.spinBoxTableCol->setEnabled(true);
}

void DialogVariableDetailDataObject::changeDObjAxes(const int row, const int col)
{
    if (!m_isChanging)
    {
        int dims = m_dObj->getDims();
        for (int idx = 0; idx < dims; idx++)
        {
            // first time show last 2 axes
            if (dims - idx == row)
            {
                m_AxesRanges[idx] = ito::Range(ito::Range::all());
            }
            else if (dims - idx == col)
            {
                m_AxesRanges[idx] = ito::Range(ito::Range::all());
            }
            else
            {
                m_AxesRanges[idx] = ito::Range(0, 1);
            }
        }

        ui.dataTable->setData(QSharedPointer<ito::DataObject>(
            new ito::DataObject(m_dObj->at(m_AxesRanges).squeeze())));

        
    }
}

} //end namespace ito