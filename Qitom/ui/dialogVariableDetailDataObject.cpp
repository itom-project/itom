/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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
#include "dataobj.h"
#include <qsharedpointer.h>

#include <qclipboard.h>
#include <qmap.h>
#include <qspinbox.h>

namespace ito {

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::DialogVariableDetailDataObject(
    const QString& name,
    const QString& type,
    const char* dtype,
    QSharedPointer<ito::DataObject> dObj,
    QWidget* parent) :
    QDialog(parent),
    m_axesRanges(nullptr), 
    m_dObj(dObj)
{
    ui.setupUi(this);

    ui.txtName->setText(name);
    ui.txtType->setText(type);
    ui.txtDType->setText(dtype);

    ui.dataTable->setReadOnly(true);

    ui.metaWidget->setData(m_dObj);
    ui.metaWidget->setReadOnly(true);

    // add combobox for displayed axis and slicing of dataObject axis
    int dims = m_dObj->getDims();

    QString dObjSize = "[";

    for (int dim = 0; dim < dims; dim++)
    {
        if (dim < dims - 1)
        {
            dObjSize.append(QString("%1 x ").arg(m_dObj->getSize(dim)));
        }
        else
        {
            dObjSize.append(QString("%1").arg(m_dObj->getSize(dim)));
        }
        
    }
    dObjSize.append("]");

    ui.txtDSize->setText(dObjSize);

    if (dims < 3)
    {
        ui.dataTable->setData(QSharedPointer<ito::DataObject>(m_dObj));
        ui.frameSlicing->setVisible(false);
    }
    else
    {
        int col = dims - 1;
        int row = dims - 2;
        bool valid = true;

        m_axesRanges = new ito::Range[dims];

        QStringList itemsRow;
        QStringList itemsCol;
        QString itemDescription;
        std::string axisDescription;
        for (int idx = 0; idx < dims; idx++)
        {
            // define combobox for row without last axis
            if (idx != dims - 1)
            {
                itemDescription = "";
                axisDescription = m_dObj->getAxisDescription(idx, valid);
                if (!axisDescription.empty())
                {
                    itemDescription += QString::fromUtf8(axisDescription.data());
                    itemDescription += " (Axis ";
                }

                itemDescription += QString::number(idx);

                if (!axisDescription.empty())
                {
                    itemDescription += ")";
                }

                itemsRow += itemDescription;
                m_rowAxisToIndex.insert(itemDescription, idx);
            }

            // define combobox for col
            if (idx != 0)
            {
                itemDescription = "";
                axisDescription = m_dObj->getAxisDescription(idx, valid);
                if (!axisDescription.empty())
                {
                    itemDescription += QString::fromUtf8(axisDescription.data());
                    itemDescription += " (Axis ";
                }

                itemDescription += QString::number(idx);

                if (!axisDescription.empty())
                {
                    itemDescription += ")";
                }

                itemsCol += itemDescription;
                m_colAxisToIndex.insert(itemDescription, idx);
            }            
        }

        ui.comboBoxDisplayedCol->blockSignals(true);
        ui.comboBoxDisplayedRow->blockSignals(true);

        ui.comboBoxDisplayedRow->addItems(itemsRow);
        ui.comboBoxDisplayedRow->setCurrentIndex(row);

        ui.comboBoxDisplayedCol->addItems(itemsCol);
        ui.comboBoxDisplayedCol->setCurrentIndex(col-1);  // - 1 because first axis is not accessible with col. ComboBox with one item less

        ui.comboBoxDisplayedCol->blockSignals(false);
        ui.comboBoxDisplayedRow->blockSignals(false);

        addSlicingWidgets();
        changeDisplayedAxes(-1);
    }
}

//----------------------------------------------------------------------------------------------
DialogVariableDetailDataObject::~DialogVariableDetailDataObject()
{
    DELETE_AND_SET_NULL_ARRAY(m_axesRanges);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_btnCopyClipboard_clicked()
{
    QClipboard* clipboard = QApplication::clipboard();
    clipboard->setText(ui.txtName->text(), QClipboard::Clipboard);
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_comboBoxDisplayedRow_currentIndexChanged(int idx)
{
    deleteSlicingWidgets();
    addSlicingWidgets();
    changeDisplayedAxes(0); // 0: row, 1: col, -1: default
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::on_comboBoxDisplayedCol_currentIndexChanged(int idx)
{
    deleteSlicingWidgets();
    addSlicingWidgets();
    changeDisplayedAxes(1); // 0: row, 1: col, -1: default
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::spinBoxValueChanged(int idx)
{
    changeDisplayedAxes(-1); // 0: row, 1: col, -1: default
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::deleteSlicingWidgets()
{
    while (ui.horizontalLayoutSlicing->count() != 0)
    {
        QLayoutItem* item = ui.horizontalLayoutSlicing->takeAt(0);
        delete item->widget();
        delete item;
    }
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::addSlicingWidgets()
{
    int dims = m_dObj->getDims();
    int col = m_colAxisToIndex.value(
        ui.comboBoxDisplayedCol->itemText(ui.comboBoxDisplayedCol->currentIndex()));
    
    int row = m_rowAxisToIndex.value(
        ui.comboBoxDisplayedRow->itemText(ui.comboBoxDisplayedRow->currentIndex()));

    ui.horizontalLayoutSlicing->addWidget(new QLabel("["));

    m_spinBoxToIdxMap.clear();

    for (int idx = 0; idx < dims; idx++)
    {
        if ((idx == row) || (idx == col)) // row
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel(":"));
        }
        else
        {
            QSpinBox* spinBox = new QSpinBox();
            spinBox->setMaximum(m_dObj->getSize(idx) - 1);
            connect(spinBox, SIGNAL(valueChanged(int)), this, SLOT(spinBoxValueChanged(int)));
            ui.horizontalLayoutSlicing->addWidget(spinBox);
            m_spinBoxToIdxMap.insert(idx, spinBox);
        }

        if (idx < dims - 1)
        {
            ui.horizontalLayoutSlicing->addWidget(new QLabel(", "));
        }
    }

    ui.horizontalLayoutSlicing->addWidget(new QLabel("]"));
}

//----------------------------------------------------------------------------------------------
void DialogVariableDetailDataObject::changeDisplayedAxes(int isColNotRow = -1)
{
    ui.comboBoxDisplayedCol->blockSignals(true);
    ui.comboBoxDisplayedRow->blockSignals(true);

    int dims = m_dObj->getDims();
    int col = m_colAxisToIndex.value(
        ui.comboBoxDisplayedCol->itemText(ui.comboBoxDisplayedCol->currentIndex()));
    int row = m_rowAxisToIndex.value(
        ui.comboBoxDisplayedRow->itemText(ui.comboBoxDisplayedRow->currentIndex()));

    if (row > col)
    {
        if (isColNotRow) // 0: row, 1: col, -1: default
        {
            row = col;
            ui.comboBoxDisplayedRow->setCurrentIndex(row);
        }
        else
        {
            col = row;
            ui.comboBoxDisplayedCol->setCurrentIndex(col - 1);
            col = m_colAxisToIndex.value(
                ui.comboBoxDisplayedCol->itemText(ui.comboBoxDisplayedCol->currentIndex()));
        }
        deleteSlicingWidgets();
        addSlicingWidgets();
    }

    for (int idx = 0; idx < dims; idx++)
    {
        if (idx == row)
        {
            m_axesRanges[idx] = ito::Range(ito::Range::all());
        }
        else if (idx == col)
        {
            m_axesRanges[idx] = ito::Range(ito::Range::all());
        }
        else
        {
            int index;
            if (m_spinBoxToIdxMap.empty())
            {
                index = 0;
            }
            else
            {
                index = m_spinBoxToIdxMap.value(idx)->value();
            }
            m_axesRanges[idx] = ito::Range(index, index + 1);
        }
    }

    ui.dataTable->setData(QSharedPointer<ito::DataObject>(
        new ito::DataObject(m_dObj->at(m_axesRanges).squeeze())));

    ui.comboBoxDisplayedCol->blockSignals(false);
    ui.comboBoxDisplayedRow->blockSignals(false);
    
}

} // end namespace ito
