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
#include <qtablewidgetitem>
#include <qspinbox.h>

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
        bool valid = true;

        QStringList items;
        QString itemDescription;
        std::string axisDescription;
        for (int idx = 0; idx < dims; idx++)
        {
            itemDescription = "";
            axisDescription = m_dObj->getAxisDescription(idx, valid);  
            if (!axisDescription.empty())
            {
                itemDescription += QString::fromUtf8(axisDescription.data()); 
                itemDescription += " (";
            }

            itemDescription += QString::number(idx);

            if (!axisDescription.empty())
            {
                itemDescription += ")";
            }      

            items += itemDescription;

            if (idx == 0)
            {
                ui.horizontalLayoutSlicing->addWidget(new QLabel("["));
            }

            if (dims - idx == 2) // row
            {
                ui.horizontalLayoutSlicing->addWidget(new QLabel(":"));
            }
            else if (dims - idx == 1) // col
            {
                ui.horizontalLayoutSlicing->addWidget(new QLabel(":"));
            }
            else
            {
                QSpinBox* spin = new QSpinBox();
                spin->setMaximum(m_dObj->getSize(idx));
                connect(spin, SIGNAL(valueChanged(int)), this, SLOT(spinBoxValueChanged(int)));
                ui.horizontalLayoutSlicing->addWidget(spin);   
            }

            if (idx + 1 == dims)
            {
                ui.horizontalLayoutSlicing->addWidget(new QLabel("]"));
            }
            else
            {
                ui.horizontalLayoutSlicing->addWidget(new QLabel(", "));
            }

        }

        ui.comboBoxDisplayedRow->addItems(items);
        ui.comboBoxDisplayedRow->setCurrentIndex(dims - 2);
        ui.comboBoxDisplayedCol->addItems(items);
        ui.comboBoxDisplayedCol->setCurrentIndex(dims - 1);

        m_AxesRanges = new ito::Range[dims];
        m_isChanging = false;

        // show last two axes after start
        changeDObjAxes(row, col);
    }
    else
    {
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

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_comboBoxDisplayedRow_currentIndexChanged(int idx)
{
    // update slicing
    // update max value of new spinbox axes
    // update changeDObjAxes here
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_comboBoxDisplayedCol_currentIndexChanged(int idx)
{
    // update slicing
    // update max value of new spinbox axes
    // update changeDObjAxes here
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::spinBoxValueChanged(int idx)
{
    // update changeDObjAxes here
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::changeDObjAxes(const int row, const int col)
{
    if (!m_isChanging)
    {
        int dims = m_dObj->getDims();
        for (int idx = 0; idx < dims; idx++)
        {
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